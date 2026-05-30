# Goodreads Explicit-Feedback ALS Recommender

This repository contains a distributed Goodreads book recommendation pipeline built with **PySpark** and **Spark MLlib ALS**. The repo focuses on my implementation of the **explicit-feedback ALS** module, while also summarizing how the explicit model compared with popularity, implicit-feedback ALS, and combined explicit + implicit feedback ALS in the broader recommendation system project.

## Why This Project Matters

Recommendation systems often need to combine different feedback signals:

- **Explicit feedback:** user ratings, such as 1--5 star scores
- **Implicit feedback:** user behavior, such as reading or reviewing a book
- **Combined feedback:** a hybrid signal that uses both ratings and behavioral engagement

This project uses Goodreads user-book interaction data to study how explicit ratings alone perform compared with behavior-based and combined-feedback recommendation models.

## My Contribution

My primary contribution was the **explicit-feedback ALS module**:

- Implemented the explicit-feedback collaborative filtering pipeline using **Spark MLlib ALS**
- Preprocessed rating-based user-book interactions for ALS training
- Generated **Top-100** recommendations per user
- Filtered already-seen books from recommendation lists
- Evaluated ranking quality using **NDCG@100, MAP, Precision@100, and Recall@100**
- Tuned ALS hyperparameters, including `rank`, `regParam`, and `maxIter`
- Participated in comparing explicit ALS with implicit ALS and combined explicit + implicit feedback models

## Dataset and Split Design

The broader project converted raw Goodreads CSV files into Parquet format on HDFS. The converted interaction dataset contained:

| Statistic | Value |
|---|---:|
| Interaction rows | 228,648,342 |
| Distinct users | 876,145 |
| Distinct books | 2,360,650 |

For evaluation, the project kept users with at least 5 read interactions. Eligible users were split into:

| Split | Design |
|---|---|
| Train | 80% of eligible users |
| Validation | 10% of eligible users, with observed / held-out interactions |
| Test | 10% of eligible users, with observed / held-out interactions |

For validation and test users, the project used an approximately **70% observed / 30% held-out** split within each user. The observed portion was used as available user history, while the held-out portion was used for recommendation evaluation.

A held-out book was treated as relevant if:

```text
is_read == 1 and rating >= 4
```

## Repository Structure

```text
goodreads-explicit-als-recommender/
├── src/
│   └── explicit_als.py
├── results/
│   ├── validation_tuning_results.md
│   └── model_comparison_results.md
├── docs/
│   └── project_summary.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Explicit-Feedback ALS Method

The explicit-feedback ALS model uses numerical Goodreads ratings as the preference signal.

```python
ALS(
    userCol="user_id",
    itemCol="book_id",
    ratingCol="rating",
    implicitPrefs=False,
    coldStartStrategy="drop"
)
```

The pipeline performs the following steps:

1. Load Goodreads Parquet split files from HDFS
2. Keep valid explicit ratings on the 1--5 scale
3. Train ALS on observed rating interactions
4. Generate candidate recommendations using `recommendForUserSubset`
5. Filter books already seen by each user
6. Keep the final Top-100 recommendations
7. Evaluate against held-out relevant books

## Hyperparameter Tuning

The current script uses a conservative 9-grid search:

```python
RANK_LIST = [5, 10, 20]
REG_PARAM_LIST = [0.05, 0.1, 0.5]
MAX_ITER_LIST = [5]
```

One validation run selected the following best setting:

| rank | regParam | maxIter | NDCG@100 | Recall@100 | MAP | Eval users |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.5 | 5 | 0.0005677491 | 0.0009798163 | 0.0000593816 | 4,935 |

## Model Comparison Summary

The broader project compared four recommendation approaches:

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

The explicit ALS model served as a rating-based baseline. It performed worse than implicit ALS because it only used explicit rated interactions, while the evaluation target focused on held-out books that users actually read and rated positively. The implicit and combined models performed better because they used broader behavioral signals such as reading and reviewing.

## Key Takeaways

- Explicit ratings alone can be sparse and may miss many read-only interactions.
- Implicit behavioral signals are stronger for large-scale Goodreads recommendation evaluation.
- The combined explicit + implicit feedback model achieved the best ranking performance in the broader project.
- The explicit ALS module is still useful as a clear collaborative filtering baseline and a controlled comparison point.

## Running the Code

First, update the HDFS paths in `src/explicit_als.py`:

```python
parquet_files = {
    "train": "hdfs:///user/<username>/goodreads/splits/train.parquet",
    "validation_observed": "hdfs:///user/<username>/goodreads/splits/validation_observed.parquet",
    "validation_heldout": "hdfs:///user/<username>/goodreads/splits/validation_heldout.parquet",
    "test_observed": "hdfs:///user/<username>/goodreads/splits/test_observed.parquet",
    "test_heldout": "hdfs:///user/<username>/goodreads/splits/test_heldout.parquet",
}

output_base ="hdfs:///user/<username>/goodreads_outputs/explicit_als_full2p"
```

Then run:

```bash
spark-submit --deploy-mode client src/explicit_als.py
```

Or run with memory settings:

```bash
nohup spark-submit \
  --deploy-mode client \
  --driver-memory 3g \
  --executor-memory 2g \
  --conf spark.sql.shuffle.partitions=400 \
  --conf spark.executor.memoryOverhead=1g \
  --conf spark.network.timeout=800s \
  --conf spark.executor.heartbeatInterval=60s \
  src/explicit_als.py > explicit_als_9grid.log 2>&1 &
```

## Notes

- The raw Goodreads dataset and HDFS Parquet split files are not included in this repository.
- This repository is designed for offline recommendation experiments, not production serving.
- The repo centers on my explicit-feedback ALS implementation. Comparison results for implicit ALS and combined ALS are included to show the broader experimental context and model trade-offs.
