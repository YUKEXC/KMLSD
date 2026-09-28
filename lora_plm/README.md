# Stage-II surrogate training and prediction

See the repository [README](../README.md) for training and beam-search commands.
P450 combinations use six letters, ordered as 68, 96, 173, 192, 294, 296.
GB1 combinations use four letters, ordered as 39, 40, 41, 54.

To score a six-site P450 candidate list after training:

```bash
python -m lora_plm.predict --model_path model/esm2_650M --peft_dir outputs/p450_model --wt_fasta WT.fasta --crossmap stage1_data/p450/refpos_crossmap.csv --enzyme_name CYP107D1 --ref_positions 68,96,173,192,294,296 --candidates_csv data/P450/six_site_example.csv --out_csv outputs/p450_predictions.csv --device cuda --local_files_only
```

Prediction and beam search load the architecture and indexing from `meta.txt`.
Existing prediction files are not appended to; use a new filename or `--overwrite`.
