# vit5_base_baseline_constrained — test

| set | BLEU↑ | chrF↑ | WER↓ | TER↓ | EM↑ | n |
|---|---:|---:|---:|---:|---:|---:|
| overall | 94.23 | 96.41 | 3.90 | 3.60 | 81.88 | 949 |
| overall (lc) | 94.37 | 96.54 | 3.77 | 3.60 | 82.51 | 949 |

## By transformation category

| category | BLEU↑ | chrF↑ | WER↓ | TER↓ | EM↑ | n |
|---|---:|---:|---:|---:|---:|---:|
| identical | 93.51 | 96.08 | 4.18 | 4.18 | 77.78 | 225 |
| reorder_only | 52.19 | 72.69 | 32.35 | 19.12 | 30.00 | 10 |
| deletion_only | 96.47 | 97.86 | 2.18 | 2.18 | 87.50 | 592 |
| deletion_reorder | 90.35 | 93.75 | 6.23 | 5.08 | 69.81 | 106 |
| lexical | 57.14 | 76.28 | 34.75 | 32.20 | 43.75 | 16 |
