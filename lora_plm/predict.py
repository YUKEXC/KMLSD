#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import pandas as pd
import torch

try:
    # when running as module
    from lora_plm.utils import read_fasta_first_seq, apply_combo_to_wt
    from lora_plm.checkpoint import load_regressor
except ImportError:
    # when running as script
    from utils import read_fasta_first_seq, apply_combo_to_wt
    from checkpoint import load_regressor


def collate_tokenize_seqs(seqs, tokenizer, device):
    enc = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True)
    return {k: v.to(device) for k, v in enc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True, help='Base local model dir (e.g., model/esm2_650M)')
    ap.add_argument('--peft_dir', required=True, help='Directory containing saved LoRA adapters (from train.py)')
    ap.add_argument('--wt_fasta', required=True)
    ap.add_argument('--crossmap', required=True)
    ap.add_argument('--enzyme_name', required=True)
    ap.add_argument('--ref_positions', required=True)
    ap.add_argument('--head', default='auto', choices=['auto', 'meanpool', 'site_attn', 'sixsite_attn'])
    ap.add_argument('--indexing', default='auto', choices=['auto', 'one_based', 'legacy_shifted'])
    ap.add_argument('--candidates_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--overwrite', action='store_true', help='Replace an existing output; never append implicitly')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--progress_every', type=int, default=50000,
                    help='Print progress every N sequences (approx). Set 0 to disable.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--local_files_only', action='store_true')
    ap.add_argument('--trust_remote_code', action='store_true')
    args = ap.parse_args()

    if os.path.exists(args.out_csv) and not args.overwrite:
        raise FileExistsError(f'{args.out_csv} already exists; use --overwrite to replace it')
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    ref_positions = [int(x) for x in str(args.ref_positions).split(',') if x.strip()]
    wt_seq = read_fasta_first_seq(args.wt_fasta)
    tokenizer, base, r2s, spec = load_regressor(
        args.model_path, args.peft_dir, args.crossmap, args.enzyme_name, ref_positions,
        head=args.head, indexing=args.indexing, local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code, device=device)
    print(f"[MODEL] head={spec.head} heads={spec.attn_heads} layers={spec.attn_layers} indexing={spec.indexing}")

    total = 0
    global_start = time.time()
    last_print_k = 0  # last multiple of progress_every already printed
    write_header = True
    for chunk in pd.read_csv(args.candidates_csv, chunksize=200000):
        if 'Combo' not in chunk.columns:
            if chunk.shape[1] == 1:
                chunk.columns = ['Combo']
            else:
                raise SystemExit('candidates_csv must have Combo column')
        combos = chunk['Combo'].astype(str).tolist()
        seqs = [apply_combo_to_wt(wt_seq, ref_positions, r2s, c) for c in combos]
        preds = []
        with torch.no_grad():
            # processed before entering this chunk
            base_count = total
            for i in range(0, len(seqs), args.batch_size):
                batch = seqs[i:i+args.batch_size]
                enc = collate_tokenize_seqs(batch, tokenizer, device)
                out = base(**enc)
                y_hat = out['logits']
                preds.extend(y_hat.detach().cpu().numpy().tolist())
                # progress
                if args.progress_every and args.progress_every > 0:
                    done = base_count + i + len(batch)
                    k = done // args.progress_every
                    if k > last_print_k:
                        last_print_k = k
                        elapsed = time.time() - global_start
                        rate = done / max(elapsed, 1e-9)
                        print(f"[pred] processed={done}  rate={rate:.1f}/s  elapsed={elapsed:.1f}s")

        out = pd.DataFrame({'Combo': combos, 'y_pred': preds})
        out.to_csv(args.out_csv, index=False, mode='a' if not write_header else 'w', header=write_header)
        write_header = False
        total += len(out)
        elapsed = time.time() - global_start
        rate = total / max(elapsed, 1e-9)
        print(f"[chunk] wrote {len(out)} rows, total={total}, rate={rate:.1f}/s, elapsed={elapsed:.1f}s")

    print(f"[OK] predictions saved: {args.out_csv}, total={total}")


if __name__ == '__main__':
    main()
