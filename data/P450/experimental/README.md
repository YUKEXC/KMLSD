# P450 experimental measurements

This directory contains the experimental measurement tables supplied on
30 September 2026. Each Excel workbook is preserved as supplied, with an English
filename. Its CSV counterpart contains the same variant identifiers and numeric
values, with the two spreadsheet header rows combined into a single CSV header.

| File stem | Content | Number of records |
| --- | --- | ---: |
| `alanine_scanning` | Single alanine substitutions | 85 |
| `saturation_mutagenesis` | All 19 non-parental substitutions at S68, Q96, T173, V192, G294, and F296 | 114 |
| `om3_reference` | OM3 reference measurements | 1 |

## Fields

| Column | Meaning |
| --- | --- |
| `Variant` | Variant identifier as supplied |
| `YUDCA` | Mean UDCA yield |
| `SUDCA` | Mean UDCA selectivity |
| `YMDCA` | Mean MDCA yield |
| `SMDCA` | Mean MDCA selectivity |
| `Conversion` | Mean substrate conversion |

All five numeric fields are fractions from 0 to 1. Multiply by 100 to express
them as percentages. Excel displays these fields as percentages. The tables
contain reported means, without individual replicate measurements.

The six alanine variants at the saturation sites occur in both scanning tables.
Their measurements differ between the two experimental series and are retained
separately. Recorded zeros and conversion values are preserved. No rounding,
normalization, averaging across tables, or recalculation of selectivity has
been applied to the CSV exports.

## Relationship to model inputs

`stage1_data/p450/alanine_labels.csv` and
`data/P450/fitness_round1_training*.csv` retain the processed inputs for the
existing model runs. Their normalized fields are not direct exports of the
measurement tables in this directory. The supplied checkpoints and saved
rankings remain associated with those existing inputs.
