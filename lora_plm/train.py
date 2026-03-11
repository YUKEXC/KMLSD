#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
from typing import List

import numpy as np
import pandas as pd
# For deterministic CUDA behavior with torch.use_deterministic_algorithms on CUDA>=10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None  # runtime check later

try:
    from lora_plm.utils import read_fasta_first_seq, load_pos_map
    from lora_plm.data import ComboRegressionDataset
    from lora_plm.model import load_encoder, SeqRegressor, SixSiteAttentionRegressor, MultiSiteAttentionRegressor, attach_lora
except Exception:
    from utils import read_fasta_first_seq, load_pos_map
    from data import ComboRegressionDataset
    from model import load_encoder, SeqRegressor, SixSiteAttentionRegressor, MultiSiteAttentionRegressor, attach_lora


def collate_tokenize(batch, tokenizer, device):
    seqs = [b["seq"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b.get("weight", 1.0) for b in batch], dtype=torch.float32)
    enc = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, labels.to(device), weights.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--wt_fasta', required=True)
    ap.add_argument('--crossmap', required=True)
    ap.add_argument('--enzyme_name', required=True)
    ap.add_argument('--ref_positions', required=True, help='Comma-separated ref_pos, e.g., 68,96,192,195')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--obj_col', default='PlateNormIso2')
    ap.add_argument('--mdca_col', default=None)
    ap.add_argument('--obj_lambda', type=float, default=0.0)
    ap.add_argument('--weight_col', default=None, help='Optional sample weight column in train_csv')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup_ratio', type=float, default=0.06)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--loss', choices=['mse','huber','logcosh'], default='mse')
    ap.add_argument('--huber_delta', type=float, default=1.0)
    ap.add_argument('--rank_loss_weight', type=float, default=0.0, help='Pairwise logistic ranking regularizer weight')
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--local_files_only', action='store_true')
    ap.add_argument('--trust_remote_code', action='store_true')
    ap.add_argument('--lora_targets', default='auto', help='Comma-separated target module names or "auto" to autodetect')
    ap.add_argument('--head', choices=['meanpool','sixsite_attn','site_attn'], default='sixsite_attn')
    ap.add_argument('--attn_heads', type=int, default=4)
    ap.add_argument('--attn_layers', type=int, default=1)
    ap.add_argument('--attn_dropout', type=float, default=0.1)
    ap.add_argument('--attn_ff_mult', type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 确保 GPU 可复现
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    # Load mapping and WT
    ref_positions = [int(x) for x in str(args.ref_positions).split(',') if x.strip()]
    wt_seq = read_fasta_first_seq(args.wt_fasta)
    r2s = load_pos_map(args.crossmap, args.enzyme_name, ref_positions)

    # Load data
    df = pd.read_csv(args.train_csv)
    # Optional: composite objective y = UDCA - lambda*MDCA
    obj_col = args.obj_col
    if args.mdca_col and args.obj_lambda and float(args.obj_lambda) > 0:
        if args.obj_col not in df.columns or args.mdca_col not in df.columns:
            raise ValueError(f"Missing columns for composite objective: need {args.obj_col} and {args.mdca_col}")
        df['_Obj'] = pd.to_numeric(df[args.obj_col], errors='coerce').fillna(0.0) - float(args.obj_lambda) * pd.to_numeric(df[args.mdca_col], errors='coerce').fillna(0.0)
        obj_col = '_Obj'
    dataset = ComboRegressionDataset(df, wt_seq, ref_positions, r2s, obj_col, weight_col=args.weight_col)

    # Split
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    enc_tok = load_encoder(args.model_path, local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code)
    encoder = enc_tok.encoder
    tokenizer = enc_tok.tokenizer
    hidden_size = enc_tok.hidden_size

    # Build regressor on top
    # Attach LoRA to the encoder (PreTrainedModel) so that PEFT can rely on .config
    targets = None if args.lora_targets.strip().lower() == 'auto' else [t.strip() for t in args.lora_targets.split(',') if t.strip()]
    encoder = attach_lora(encoder, target_modules=targets, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, auto=True)
    if args.head == 'sixsite_attn':
        if len(ref_positions) != 6:
            raise SystemExit("sixsite_attn head requires exactly 6 reference positions.")
        site_seq_positions = [int(r2s[rp]) for rp in ref_positions]
        base = SixSiteAttentionRegressor(encoder, hidden_size,
                                         site_seq_positions=site_seq_positions,
                                         n_heads=args.attn_heads,
                                         n_layers=args.attn_layers,
                                         ff_mult=args.attn_ff_mult,
                                         dropout=args.attn_dropout)
    elif args.head == 'site_attn':
        site_seq_positions = [int(r2s[rp]) for rp in ref_positions]
        base = MultiSiteAttentionRegressor(encoder, hidden_size,
                                           site_seq_positions=site_seq_positions,
                                           n_heads=args.attn_heads,
                                           n_layers=args.attn_layers,
                                           ff_mult=args.attn_ff_mult,
                                           dropout=args.attn_dropout)
    else:
        base = SeqRegressor(encoder, hidden_size)
    base = base.to(device)

    # Optim & sched
    trainable = [p for p in base.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_tokenize(b, tokenizer, device))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_tokenize(b, tokenizer, device))

    t_total = args.epochs * max(1, len(train_loader))
    warmup = int(t_total * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=t_total)

    best_val = None
    loss_mode = args.loss
    huber_delta = float(args.huber_delta)
    rank_w = float(args.rank_loss_weight)

    def _primary_loss(preds: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None):
        e = preds - labels
        if loss_mode == 'mse':
            per = e * e
        elif loss_mode == 'huber':
            abs_e = torch.abs(e)
            per = torch.where(abs_e <= huber_delta, 0.5 * e * e, huber_delta * (abs_e - 0.5 * huber_delta))
        else:  # logcosh
            per = torch.log(torch.cosh(e.clamp(min=-20, max=20)))
        if weights is None:
            return per.mean()
        return (per * weights).sum() / weights.sum().clamp_min(1e-6)

    def _pairwise_rank_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = preds.size(0)
        if B < 2 or rank_w <= 0:
            return preds.new_tensor(0.0)
        p_i = preds.view(B, 1)
        p_j = preds.view(1, B)
        y_i = labels.view(B, 1)
        y_j = labels.view(1, B)
        sign = torch.sign(y_i - y_j)
        mask = sign.ne(0.0)
        if not mask.any():
            return preds.new_tensor(0.0)
        diff_p = p_i - p_j
        term = torch.nn.functional.softplus(-(sign * diff_p))[mask]
        return term.mean()
    for epoch in range(1, args.epochs + 1):
        base.train()
        tr_loss = 0.0
        for step, (enc, labels, weights) in enumerate(train_loader, 1):
            out = base(**enc)
            preds = out['logits']
            loss_main = _primary_loss(preds, labels, weights)
            loss_rank = _pairwise_rank_loss(preds, labels)
            loss = loss_main + rank_w * loss_rank
            if torch.isnan(loss):
                continue
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # eval
        base.eval()
        val_loss = 0.0
        with torch.no_grad():
            for enc, labels, weights in val_loader:
                out = base(**enc)
                preds = out['logits']
                loss_main = _primary_loss(preds, labels, weights)
                loss_rank = _pairwise_rank_loss(preds, labels)
                loss = loss_main + rank_w * loss_rank
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        if best_val is None or val_loss < best_val - 1e-6:
            best_val = val_loss
            # save LoRA adapters on encoder and tokenizer; save regression head separately
            saved = False
            if hasattr(base.encoder, 'save_pretrained'):
                try:
                    base.encoder.save_pretrained(args.out_dir)
                    saved = True
                except Exception as e:
                    print(f"[WARN] save_pretrained failed: {e}")
            # sanity check: ensure adapter_config.json exists
            cfg_path = os.path.join(args.out_dir, 'adapter_config.json')
            if not os.path.exists(cfg_path):
                print("[WARN] adapter_config.json not found after save; the encoder may not be a PEFT model. Training will continue, but prediction with --peft_dir will require this file.")
            # save head weights
            if hasattr(base, 'reg_head'):
                torch.save(base.reg_head.state_dict(), os.path.join(args.out_dir, 'reg_head.pt'))
            if hasattr(base, 'site_encoder'):
                torch.save(base.site_encoder.state_dict(), os.path.join(args.out_dir, 'site_encoder.pt'))
            tokenizer.save_pretrained(args.out_dir)
            with open(os.path.join(args.out_dir, 'meta.txt'), 'w', encoding='utf-8') as f:
                f.write(f"ref_positions={','.join(map(str, ref_positions))}\n")
                f.write(f"enzyme_name={args.enzyme_name}\n")
                f.write(f"model_path={args.model_path}\n")
                f.write(f"val_loss={val_loss:.6f}\n")
                f.write(f"head={args.head}\n")
                if args.head in ('sixsite_attn', 'site_attn'):
                    f.write(f"attn_heads={args.attn_heads}\n")
                    f.write(f"attn_layers={args.attn_layers}\n")
                    f.write(f"attn_dropout={args.attn_dropout}\n")
                    f.write(f"attn_ff_mult={args.attn_ff_mult}\n")
            print(f"[SAVE] best adapters → {args.out_dir}")

    print("[DONE] Training complete.")


if __name__ == '__main__':
    main()
