# P450 experimental measurements

These tables contain the experimental measurements supplied on 30 September
2026. English filenames are used consistently for each Excel/CSV pair.

| Dataset | Excel | CSV | Records |
| --- | --- | --- | ---: |
| Alanine scanning | [alanine_labels.xlsx](../../../stage1_data/p450/alanine_labels.xlsx) | [alanine_labels.csv](../../../stage1_data/p450/alanine_labels.csv) | 85 |
| Six-site saturation mutagenesis | [fitness_round1_training_six_with_aux.xlsx](../fitness_round1_training_six_with_aux.xlsx) | [fitness_round1_training_six_with_aux.csv](../fitness_round1_training_six_with_aux.csv) | 114 |
| OM3 reference | [om3_reference.xlsx](om3_reference.xlsx) | [om3_reference.csv](om3_reference.csv) | 1 |

## Measurement fields

| Column | Meaning |
| --- | --- |
| `Variant` | Variant identifier as supplied |
| `YUDCA` | Mean UDCA yield |
| `SUDCA` | Mean UDCA selectivity |
| `YMDCA` | Mean MDCA yield |
| `SMDCA` | Mean MDCA selectivity |
| `Conversion` | Mean substrate conversion |

All five numeric fields are fractions from 0 to 1. Multiply by 100 to express
them as percentages. The workbooks display percentages and contain reported
means, without individual replicate measurements. The CSV measurement columns
preserve the supplied values without rounding or recalculation.

The saturation CSV also contains `Combo`, a derived identifier with site order
68, 96, 173, 192, 294, 296. Each of the 114 variants changes one position of the
parental combination `SQTVGF`; all 19 non-parental substitutions are represented
at each site. No additional measurement or parental record is inferred.

The six alanine variants at the saturation sites occur in both scanning tables.
Their measurements differ between the two experimental series and are retained
separately. Recorded zeros and conversion values are preserved. The supplied
alanine identifier `S295A` is retained; the repository reference sequence has
G at position 295, which must be resolved before using this identifier for
sequence construction. Stage-I uses its reported position only.

## Model inputs

The current Stage-I input is `stage1_data/p450/alanine_labels.csv`. Its reader
maps the position in `Variant` to `ref_pos` and uses `YUDCA` as `y`. It does not
infer risk flags or additional derived features from the reported measurements.

The current Stage-II input is `data/P450/fitness_round1_training_six_with_aux.csv`, using
`Combo` for sequence construction and `--obj_col YUDCA` for the target.

The supplied checkpoints and saved rankings predate this replacement. Their
original input files remain accessible in Git history. Other
`fitness_round1_training*.csv` tables are older processed datasets and are not
the current saturation measurement table.
