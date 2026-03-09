# P450 and GB1 Two-Stage Pipeline (Review Package)

This repository package is intentionally scoped to:
- Stage 1 hotspot screening
- Stage 2 LoRA training
- Final beam search

Supported targets in this package:
- P450 (CYP107D1)
- GB1

## Package Layout

- `10.14/03_make_feature_table_and_score.py`: Stage 1 hotspot scoring for P450
- `10.14/tools/beam_search_lora.py`: beam search for Stage 2 models
- `lora_plm/train.py`: LoRA training
- `lora_plm/predict.py`: batch prediction utility
- `data/P450/`: P450 training and candidate data
- `data/GB1/`: GB1 training and truth data (no large ALDE-only embedding file)
- `10.14/out/`: included intermediate Stage 1 inputs/outputs
- `results/lora_plm/esm2_650m_run_six_attn1/`: pretrained P450 Stage 2 weights
- `results/lora_plm/gb1_siteattn/`: pretrained GB1 Stage 2 weights
- `results/lora_plm/gb1_beam/`: GB1 beam search outputs
- `WT.fasta`: WT sequence used by the P450 command line below

## Included Data and Intermediate Files

P450:
- `data/P450/fitness_round1_training_six_with_aux.csv`
- `data/P450/all_combos.csv`
- `10.14/out/msa_site_features.csv`
- `10.14/out/alanine_labels.csv`
- `10.14/out/plm_srs_site_summary.csv`
- `10.14/out/ddg_srs_site_summary.csv`
- `10.14/out/stage1_scores.csv`
- `10.14/out/top6.csv`

GB1:
- `data/GB1/GB1.CSV`
- `data/GB1/GB1_WT.fasta`
- `data/GB1/gb1_refpos_crossmap.csv`
- `data/GB1/gb1_stage2_train.csv`
- `data/GB1/gb1_stage2_singles.csv`
- `10.14/out/gb1_stage1_whole_seq/site_features_stage1.csv`
- `10.14/out/gb1_stage1_whole_seq/top4.csv`

## Prerequisites

- Python 3.9+
- PyTorch, Transformers, PEFT, NumPy, Pandas
- A local base model at `model/esm2_650M` (not bundled here)
- GPU is recommended for training and beam search

## Stage 1 (P450) Historical Hotspot Scoring Command

The following command keeps the historical scoring weights in this project line:

```bash
python 10.14/03_make_feature_table_and_score.py \
  --in_dir 10.14/out \
  --out_dir 10.14/out/TOP \
  --topk 6 --srs_only \
  --w_model 0.8 \
  --w_alpha 0.8 \
  --w_alpha_udca_sel 0.5 \
  --w_alpha_mdca 0 \
  --w_alpha_mdca_sel 0 \
  --w_delta 0.8 \
  --w_lambda 0.5 \
  --plm_csv 10.14/out/plm_srs_site_summary.csv --w_plm 0.3 \
  --ddg_csv 10.14/out/ddg_srs_site_summary.csv --w_ddg 0.7
```

## Stage 2 Training (P450)

Command requested for this package:

```bash
python lora_plm/train.py --model_path model/esm2_650M --wt_fasta WT.fasta --crossmap 10.14/out/refpos_crossmap.csv --enzyme_name CYP107D1 --ref_positions 68,96,173,192,294,296 --train_csv data/P450/fitness_round1_training_six_with_aux.csv --obj_col PlateNormIso2 --mdca_col PlateNormIso1 --obj_lambda 0 --head sixsite_attn --attn_heads 4 --attn_layers 2 --attn_dropout 0.1 --attn_ff_mult 2 --epochs 12 --batch_size 2 --lr 1e-4 --out_dir results/lora_plm/esm2_650m_run_six_attn1 --device cuda:2 --local_files_only --trust_remote_code
```

## Beam Search (P450)

Command requested for this package:

```bash
python 10.14/tools/beam_search_lora.py --model_path model/esm2_650M --peft_dir results/lora_plm/esm2_650m_run_six_attn1 --wt_fasta WT.fasta --crossmap 10.14/out/refpos_crossmap.csv --enzyme_name CYP107D1 --ref_positions 68,96,173,192,294,296 --out_dir results/lora_plm/esm2_650m_run_six_attn/beam1 --beam 256 --epsilon 0.05 --diversity_dmin 0 --seeds_from_singles 400 --batch_size 128 --device cuda:2 --local_files_only --trust_remote_code --head auto
```

## Stage 1 and Stage 2 (GB1)

GB1 Stage 1 processed hotspot files are already included:
- `10.14/out/gb1_stage1_whole_seq/site_features_stage1.csv`
- `10.14/out/gb1_stage1_whole_seq/top4.csv`

GB1 Stage 2 training:

```bash
python lora_plm/train.py --model_path model/esm2_650M --wt_fasta data/GB1/GB1_WT.fasta --crossmap data/GB1/gb1_refpos_crossmap.csv --enzyme_name GB1 --ref_positions 39,40,41,54 --train_csv data/GB1/gb1_stage2_train.csv --obj_col Fitness --head site_attn --attn_heads 2 --attn_layers 1 --attn_dropout 0.1 --attn_ff_mult 2 --rank_loss_weight 0.1 --epochs 20 --batch_size 4 --lr 2e-4 --out_dir results/lora_plm/gb1_siteattn --device cuda --local_files_only --trust_remote_code
```

GB1 beam search:

```bash
python 10.14/tools/beam_search_lora.py --model_path model/esm2_650M --peft_dir results/lora_plm/gb1_siteattn --wt_fasta data/GB1/GB1_WT.fasta --crossmap data/GB1/gb1_refpos_crossmap.csv --enzyme_name GB1 --ref_positions 39,40,41,54 --out_dir results/lora_plm/gb1_beam --beam 256 --epsilon 0.05 --diversity_dmin 0 --seeds_from_singles 200 --batch_size 128 --device cuda --local_files_only --trust_remote_code --head site_attn
```

## GitHub Upload Note (Large Files)

`results/lora_plm/*/site_encoder.pt` is around 100MB and should be uploaded with Git LFS.

Example:

```bash
git lfs install
git lfs track "results/lora_plm/*/site_encoder.pt"
git add .gitattributes
```

Then commit and push as usual.
