LoRA‑PLM Reranking (Small‑Sample Fine‑Tuning)

This folder provides a minimal, clear pipeline to fine‑tune a local ESM‑style protein language model with LoRA on small labeled data, then rerank large candidate sets.

What it does
- Build full sequences by applying 6‑site combos onto WT using refpos_crossmap.csv.
- Fine‑tune a regression head on top of the PLM encoder with LoRA (Q/K/V/O) only.
- Train with your small Round‑2 data and predict for large candidate batches in chunks.

Local model
- Pass your local model dir via --model_path (e.g. model/esm2_650M). The dir must contain config.json and weights.
- If your model requires trust_remote_code, add --trust_remote_code and --local_files_only.

Quickstart
1) Train (LoRA)
   ```bash
   python -m lora_plm.train \
     --model_path model/esm2_650M \
     --wt_fasta WT.fasta \
     --crossmap stage1_data/p450/refpos_crossmap.csv \
     --enzyme_name CYP107D1 \
     --ref_positions 68,96,192,195,294,296 \
     --train_csv data/P450/fitness_round1_training.csv \
     --obj_col PlateNormIso2 \
     --out_dir results/lora_plm/esm2_650m_run1 \
     --epochs 8 --batch_size 16 --lr 2e-4 --device cuda --local_files_only --trust_remote_code
   ```

   If you hit a PEFT error like "Target modules ... not found", try auto detection (default) or pass explicit names:
   ```bash
   # auto (default)
   python -m lora_plm.train ... --lora_targets auto

   # or explicit (examples; use one that matches your model)
   python -m lora_plm.train ... --lora_targets q_proj,k_proj,v_proj,out_proj
   python -m lora_plm.train ... --lora_targets query,key,value,dense
   ```

2) Predict candidates in chunks
   ```bash
   python -m lora_plm.predict \
     --model_path model/esm2_650M \
     --peft_dir results/lora_plm/esm2_650m_run1 \
     --wt_fasta WT.fasta \
     --crossmap stage1_data/p450/refpos_crossmap.csv \
     --enzyme_name CYP107D1 \
     --ref_positions 68,96,192,195,294,296 \
     --candidates_csv data/P450/all_combos.csv \
     --out_csv results/lora_plm/predicted_fitness_lora.csv \
     --batch_size 64 --device cuda --local_files_only --trust_remote_code
   ```

Notes
- This is a simple regression fine‑tuning. For tiny data, prefer fewer epochs, small lr, and early stop.
- To speed up ranking of 20^6, consider using a prior‑additive pre‑filter (e.g., PLM+ddG additive) to Top‑N, then LoRA rerank.
 - The training script saves LoRA adapters (in `--out_dir`) and a separate regression head `reg_head.pt` that predict.py loads automatically.
