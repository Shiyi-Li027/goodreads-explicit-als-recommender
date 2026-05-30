# Model Comparison Results

This file summarizes the model comparison from the broader Goodreads recommendation project.

## Evaluation Setup

Relevant held-out items were defined as:

```text
is_read == 1 and rating >= 4
```

The main ranking metrics were:

- MAP
- NDCG@100
- Precision@100
- Recall@100

## Results Table

| Model | Split | MAP | NDCG@100 | Precision@100 | Recall@100 |
|---|---|---:|---:|---:|---:|
| Popularity baseline | Validation | 0.00257 | 0.01997 | 0.00721 | 0.03667 |
| Popularity baseline | Test | 0.00251 | 0.01971 | 0.00716 | 0.03622 |
| Explicit ALS | Validation | 0.00000897 | 0.000123 | 0.000068 | 0.000211 |
| Explicit ALS | Test | 0.00000724 | 0.000119 | 0.0000718 | 0.000187 |
| Implicit ALS | Validation | 0.03597 | 0.11262 | 0.02878 | 0.16581 |
| Implicit ALS | Test | 0.04033 | 0.12916 | 0.03219 | 0.20891 |
| Explicit + Implicit ALS | Validation | 0.06426 | 0.17196 | 0.04168 | 0.22404 |
| Explicit + Implicit ALS | Test | 0.06408 | 0.17111 | 0.03955 | 0.21518 |

## Interpretation

The popularity baseline was non-personalized, so it recommended the same globally popular books to many users. Its metrics were low but useful as an initial benchmark.

The explicit ALS model used only numerical rating scores. It performed poorly in this evaluation setting because many useful read-only interactions were excluded from training, while the held-out target focused on books that users read and rated positively.

The implicit ALS model used behavioral signals such as reading and reviewing. It performed much better than the popularity baseline and explicit ALS because it captured more user-book interaction history.

The combined explicit + implicit feedback model achieved the best overall metrics by combining reading behavior, normalized ratings, and review signals.
