# vit5_large_constrained — test

| set | BLEU↑ | chrF↑ | WER↓ | TER↓ | EM↑ | n |
|---|---:|---:|---:|---:|---:|---:|
| overall | 89.12 | 92.49 | 6.74 | 6.47 | 68.49 | 949 |
| overall (lc) | 89.23 | 92.58 | 6.66 | 6.47 | 68.92 | 949 |

## By transformation category

| category | BLEU↑ | chrF↑ | WER↓ | TER↓ | EM↑ | n |
|---|---:|---:|---:|---:|---:|---:|
| identical | 82.95 | 87.67 | 9.66 | 9.53 | 55.56 | 225 |
| reorder_only | 52.64 | 69.81 | 30.88 | 22.06 | 30.00 | 10 |
| deletion_only | 91.86 | 94.48 | 4.71 | 4.69 | 74.32 | 592 |
| deletion_reorder | 91.59 | 94.28 | 6.10 | 5.34 | 70.75 | 106 |
| lexical | 56.58 | 73.69 | 37.29 | 33.90 | 43.75 | 16 |
