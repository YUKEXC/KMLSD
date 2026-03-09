#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unordered beam search over an arbitrary number of mutation sites using a LoRA-PLM regressor.

Idea:
  - Evaluate all single mutations (num_sites × 20) and take top seeds.
  - Iteratively expand any remaining position by 20 letters (unordered),
    score with the LoRA model, merge all expansions, keep top-K as beam.
  - Optional epsilon-greedy and diversity (approx Hamming) during pruning.
  - Output final Top-K (default 30) fully-specified combos with scores.

Usage (example):
  python beam/beam_search_lora.py \
    --model_path model/esm2_650M \
    --peft_dir results/lora_plm/esm2_650m_run1_173 \
    --wt_fasta WT.fasta \
    --crossmap stage1_data/p450/refpos_crossmap.csv \
    --enzyme_name CYP107D1 \
    --ref_positions 68,96,173,192,294,296 \
    --out_dir results/lora_plm/esm2_650m_run1_173/beam \
    --beam 128 --epsilon 0.05 --diversity_dmin 0 \
    --seeds_from_singles 400 --pairwise_top_per_site 0 \
    --batch_size 128 --device cuda --local_files_only --trust_remote_code \
    --topk_final 30
"""
import argparse
import os
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from peft import PeftModel, LoraConfig
import json
import inspect

# Reuse utils from lora_plm and regressor head
# Import from package; fallback to add repo root to sys.path when run by path
try:
    from lora_plm.utils import read_fasta_first_seq, load_pos_map
    from lora_plm.model import SeqRegressor, SixSiteAttentionRegressor, MultiSiteAttentionRegressor
except Exception:
    import sys as _sys, os as _os
    _sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from lora_plm.utils import read_fasta_first_seq, load_pos_map
    from lora_plm.model import SeqRegressor, SixSiteAttentionRegressor, MultiSiteAttentionRegressor

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def load_base_encoder(model_path: str, local_files_only: bool, trust_remote_code: bool):
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    try:
        enc = AutoModel.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    except Exception:
        mlm = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
        enc = mlm.esm
    return tok, enc


def apply_letters_to_wt(wt_seq: str, ref_positions: List[int], r2s: Dict[int, int], letters: List[Optional[str]]) -> str:
    s_list = list(wt_seq)
    for i, rp in enumerate(ref_positions):
        ch = letters[i]
        if ch is None:
            continue
        si = r2s[rp]
        s_list[si] = ch
    return ''.join(s_list)


def batch_predict(seqs: List[str], tokenizer, model: SeqRegressor, device: torch.device, batch_size: int) -> List[float]:
    preds: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            y = out['logits'].detach().cpu().numpy().tolist()
            preds.extend(y)
    return preds

def batch_predict_mc(seqs: List[str], tokenizer, model: SeqRegressor, device: torch.device, batch_size: int, passes: int) -> Tuple[List[float], List[float]]:
    import numpy as np
    means: List[float] = []
    stds: List[float] = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            preds_all = []
            for _ in range(passes):
                model.train()  # enable dropout
                out = model(**enc)
                y = out['logits'].detach().cpu().numpy()
                preds_all.append(y)
            arr = np.stack(preds_all, axis=0)  # [T, B]
            means.extend(arr.mean(axis=0).tolist())
            stds.extend(arr.std(axis=0, ddof=0).tolist())
    return means, stds


def hamming_at_known(a: List[Optional[str]], b: List[Optional[str]]) -> int:
    d = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if x != y:
            d += 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--peft_dir', required=True)
    ap.add_argument('--wt_fasta', required=True)
    ap.add_argument('--crossmap', required=True)
    ap.add_argument('--enzyme_name', required=True)
    ap.add_argument('--ref_positions', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--alphabet', default=''.join(AA20))
    ap.add_argument('--beam', type=int, default=128)
    ap.add_argument('--epsilon', type=float, default=0.05)
    ap.add_argument('--diversity_dmin', type=int, default=0)
    ap.add_argument('--seeds_from_singles', type=int, default=400)
    ap.add_argument('--pairwise_top_per_site', type=int, default=0)
    ap.add_argument('--topk_final', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--local_files_only', action='store_true')
    ap.add_argument('--trust_remote_code', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--head', default='auto', choices=['auto','meanpool','sixsite_attn','site_attn'])
    # MC-dropout for uncertainty (UCB)
    ap.add_argument('--mc_passes', type=int, default=0, help='>1 to enable MC-dropout averaging')
    ap.add_argument('--kappa', type=float, default=0.0, help='UCB bonus multiplier (score = mean + kappa*std)')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.makedirs(args.out_dir, exist_ok=True)

    ref_positions = [int(x) for x in str(args.ref_positions).split(',') if x.strip()]
    num_sites = len(ref_positions)
    if num_sites == 0:
        raise SystemExit('Please specify at least one reference position.')
    letters_all = list(str(args.alphabet))
    if len(letters_all) < 20:
        raise SystemExit('alphabet must contain at least the 20 canonical AAs')

    wt_seq = read_fasta_first_seq(args.wt_fasta)
    r2s = load_pos_map(args.crossmap, args.enzyme_name, ref_positions)

    # Load encoder + LoRA adapters + regression head
    tokenizer, encoder = load_base_encoder(args.model_path, args.local_files_only, args.trust_remote_code)
    # Try to load adapters; if missing/invalid, fall back to base encoder (for zero-shot)
    if args.peft_dir:
        cfg_path = os.path.join(args.peft_dir, 'adapter_config.json')
        if not os.path.exists(cfg_path):
            print(f"[INFO] adapter_config.json not found in {args.peft_dir}, skip PEFT; using base encoder.")
        else:
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                sig = inspect.signature(LoraConfig.__init__)
                allowed = set(k for k in sig.parameters.keys() if k not in ('self',))
                filtered = {k: v for k, v in raw.items() if k in allowed}
                filtered.setdefault('r', 8)
                filtered.setdefault('lora_alpha', 16)
                filtered.setdefault('lora_dropout', 0.05)
                filtered.setdefault('bias', 'none')
                filtered.setdefault('task_type', 'FEATURE_EXTRACTION')
                if 'target_modules' not in filtered:
                    filtered['target_modules'] = []
                lora_cfg = LoraConfig(**filtered)
                encoder = PeftModel.from_pretrained(encoder, args.peft_dir, config=lora_cfg,
                                                    local_files_only=args.local_files_only)
            except Exception as e2:
                print(f"[WARN] failed to load LoRA adapters from {args.peft_dir}: {e2}; using base encoder.")
    hidden_size = getattr(encoder.base_model.config, 'hidden_size', getattr(encoder.config, 'hidden_size', 768))
    # determine head type
    meta_path = os.path.join(args.peft_dir, 'meta.txt')
    head_type = None
    # default attn hyperparams
    attn_heads = 4
    attn_layers = 1
    attn_dropout = 0.1
    attn_ff_mult = 2
    if args.head != 'auto':
        head_type = args.head
    else:
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('head='):
                            head_type = line.strip().split('=',1)[1]
                        elif line.startswith('attn_heads='):
                            try:
                                attn_heads = int(line.strip().split('=',1)[1])
                            except Exception:
                                pass
                        elif line.startswith('attn_layers='):
                            try:
                                attn_layers = int(line.strip().split('=',1)[1])
                            except Exception:
                                pass
                        elif line.startswith('attn_dropout='):
                            try:
                                attn_dropout = float(line.strip().split('=',1)[1])
                            except Exception:
                                pass
                        elif line.startswith('attn_ff_mult='):
                            try:
                                attn_ff_mult = int(line.strip().split('=',1)[1])
                            except Exception:
                                pass
            except Exception:
                head_type = None
    if head_type == 'sixsite_attn':
        if num_sites != 6:
            print("[WARN] sixsite_attn head requires exactly 6 positions; falling back to meanpool.")
            head_type = 'meanpool'
    if head_type == 'sixsite_attn':
        site_seq_positions = [int(r2s[rp]) for rp in ref_positions]
        model = SixSiteAttentionRegressor(encoder, hidden_size, site_seq_positions,
                                          n_heads=attn_heads, n_layers=attn_layers, ff_mult=attn_ff_mult, dropout=attn_dropout)
        # load site_encoder and reg_head if present
        se_path = os.path.join(args.peft_dir, 'site_encoder.pt')
        if os.path.exists(se_path):
            try:
                sd = torch.load(se_path, map_location='cpu')
                model.site_encoder.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[WARN] failed to load site_encoder.pt: {e}")
        reg_head_path = os.path.join(args.peft_dir, 'reg_head.pt')
        if os.path.exists(reg_head_path):
            sd = torch.load(reg_head_path, map_location='cpu')
            model.reg_head.load_state_dict(sd, strict=False)
    elif head_type == 'site_attn':
        site_seq_positions = [int(r2s[rp]) for rp in ref_positions]
        model = MultiSiteAttentionRegressor(encoder, hidden_size, site_seq_positions,
                                            n_heads=attn_heads, n_layers=attn_layers, ff_mult=attn_ff_mult, dropout=attn_dropout)
        se_path = os.path.join(args.peft_dir, 'site_encoder.pt')
        if os.path.exists(se_path):
            try:
                sd = torch.load(se_path, map_location='cpu')
                model.site_encoder.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[WARN] failed to load site_encoder.pt: {e}")
        reg_head_path = os.path.join(args.peft_dir, 'reg_head.pt')
        if os.path.exists(reg_head_path):
            sd = torch.load(reg_head_path, map_location='cpu')
            model.reg_head.load_state_dict(sd, strict=False)
    else:
        model = SeqRegressor(encoder, hidden_size)
        reg_head_path = os.path.join(args.peft_dir, 'reg_head.pt')
        if os.path.exists(reg_head_path):
            sd = torch.load(reg_head_path, map_location='cpu')
            model.reg_head.load_state_dict(sd, strict=False)
    device = torch.device(args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    # Helper to score a list of partial assignments
    def score_partials(partials: List[List[Optional[str]]]) -> List[float]:
        seqs = [apply_letters_to_wt(wt_seq, ref_positions, r2s, p) for p in partials]
        if int(args.mc_passes) and int(args.mc_passes) > 1 and float(args.kappa) != 0.0:
            mu, sd = batch_predict_mc(seqs, tokenizer, model, device, args.batch_size, int(args.mc_passes))
            k = float(args.kappa)
            return [m + k*s for m, s in zip(mu, sd)]
        else:
            return batch_predict(seqs, tokenizer, model, device, args.batch_size)

    # Stage 1: singles (num_sites × 20)
    singles: List[List[Optional[str]]] = []
    for i in range(num_sites):
        for a in letters_all[:20]:
            arr = [None]*num_sites
            arr[i] = a
            singles.append(arr)
    scores = score_partials(singles)
    df_s1 = []
    for arr, sc in zip(singles, scores):
        combo = ''.join(ch if ch is not None else '-' for ch in arr)
        df_s1.append({'partial': combo, 'score': sc, 'filled': sum(ch is not None for ch in arr)})
    s1 = pd.DataFrame(df_s1).sort_values('score', ascending=False).reset_index(drop=True)
    seeds = s1.head(int(args.seeds_from_singles)).copy()
    seeds_path = os.path.join(args.out_dir, 'stage1_singles.csv')
    s1.to_csv(seeds_path, index=False)
    print(f"[S1] singles done → {seeds_path}")

    # Initialize beam with seeds (convert back to list[List[Optional[str]]])
    def str_to_partial(s: str) -> List[Optional[str]]:
        return [None if c=='-' else c for c in s]

    beam: List[List[Optional[str]]] = [str_to_partial(s) for s in seeds['partial'].tolist()]
    beam_scores: List[float] = seeds['score'].tolist()

    # Stage 2+: iterative unordered expansions to fill all positions
    for layer in range(1, num_sites):  # after singles, need num_sites-1 more letters
        expansions: List[List[Optional[str]]] = []
        # generate expansions by adding one letter at any empty position
        for arr in beam:
            for pos_idx in range(num_sites):
                if arr[pos_idx] is not None:
                    continue
                for a in letters_all[:20]:
                    new_arr = list(arr)
                    new_arr[pos_idx] = a
                    expansions.append(new_arr)
        # deduplicate
        seen = set()
        uniq_exp: List[List[Optional[str]]] = []
        for arr in expansions:
            key = tuple(ch if ch is not None else '-' for ch in arr)
            if key in seen:
                continue
            seen.add(key)
            uniq_exp.append(arr)
        print(f"[L{layer}] candidates before score: {len(uniq_exp)}")

        exp_scores = score_partials(uniq_exp)
        dfL = []
        for arr, sc in zip(uniq_exp, exp_scores):
            combo = ''.join(ch if ch is not None else '-' for ch in arr)
            dfL.append({'partial': combo, 'score': sc, 'filled': sum(ch is not None for ch in arr)})
        dfl = pd.DataFrame(dfL).sort_values('score', ascending=False).reset_index(drop=True)
        dfl.to_csv(os.path.join(args.out_dir, f'layer{layer}_candidates.csv'), index=False)

        # epsilon-greedy + optional diversity
        beam_next: List[List[Optional[str]]] = []
        kept = 0
        top_cut = max(1, int(args.beam * (1.0 - args.epsilon)))
        # deterministic picks
        for _, r in dfl.head(top_cut).iterrows():
            arr = str_to_partial(r['partial'])
            if args.diversity_dmin > 0:
                ok = True
                for b in beam_next:
                    if hamming_at_known(arr, b) < args.diversity_dmin:
                        ok = False
                        break
                if not ok:
                    continue
            beam_next.append(arr)
            kept += 1
            if kept >= args.beam:
                break
        # random picks
        if kept < args.beam and len(dfl) > top_cut:
            pool_idx = list(range(top_cut, len(dfl)))
            random.shuffle(pool_idx)
            for idx in pool_idx:
                arr = str_to_partial(dfl.iloc[idx]['partial'])
                if args.diversity_dmin > 0:
                    ok = True
                    for b in beam_next:
                        if hamming_at_known(arr, b) < args.diversity_dmin:
                            ok = False
                            break
                    if not ok:
                        continue
                beam_next.append(arr)
                kept += 1
                if kept >= args.beam:
                    break

        beam = beam_next
        beam_scores = score_partials(beam)
        df_beam = []
        for arr, sc in zip(beam, beam_scores):
            combo = ''.join(ch if ch is not None else '-' for ch in arr)
            df_beam.append({'partial': combo, 'score': sc})
        pd.DataFrame(df_beam).sort_values('score', ascending=False).to_csv(
            os.path.join(args.out_dir, f'layer{layer}_beam.csv'), index=False)
        print(f"[L{layer}] beam kept: {len(beam)}")

    # Finalize: ensure full combos (fill remaining None with WT AA at that pos)
    # Better: only keep those fully filled (should be, after 5 layers)
    finals: List[str] = []
    for arr in beam:
        full = [ch if ch is not None else '-' for ch in arr]
        if '-' in full:
            # fill with WT letter in case of shortfall
            for i, ch in enumerate(full):
                if ch == '-':
                    wt_ch = wt_seq[r2s[ref_positions[i]]]
                    full[i] = wt_ch
        finals.append(''.join(full))
    # dedup + score precise finals
    finals = list(dict.fromkeys(finals))
    seqs = []
    for c in finals:
        letters = [ch for ch in c]
        seqs.append(apply_letters_to_wt(wt_seq, ref_positions, r2s, letters))
    final_scores = batch_predict(seqs, tokenizer, model, device, args.batch_size)
    df_final = pd.DataFrame({'Combo': finals, 'y_pred': final_scores}).sort_values('y_pred', ascending=False)
    out_all = os.path.join(args.out_dir, 'beam_all_final.csv')
    df_final.to_csv(out_all, index=False)
    out_top = os.path.join(args.out_dir, f'beam_top{args.topk_final}.csv')
    df_final.head(int(args.topk_final)).to_csv(out_top, index=False)
    print(f"[OK] wrote {out_all} and {out_top}")


if __name__ == '__main__':
    main()
