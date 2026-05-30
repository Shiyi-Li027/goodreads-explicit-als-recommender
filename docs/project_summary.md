# Goodreads Recommendation Project Summary

## Data Preprocessing

The raw Goodreads CSV files were converted to Parquet and stored on HDFS for scalable Spark processing.

The converted interaction dataset contained:

- 228,648,342 interaction rows
- 876,145 distinct users
- 2,360,650 distinct books

## Split Strategy

The project kept users with at least 5 read interactions, where a read interaction was defined as `is_read == 1`.

Eligible users were split into:

- 80% train users
- 10% validation users
- 10% test users

For validation and test users, each user's interactions were further split into observed and held-out portions. The observed portion represented available user history, and the held-out portion was used for offline recommendation evaluation.

## Explicit-Feedback ALS

The explicit-feedback ALS model used numerical ratings as the preference signal.

Preprocessing steps:

1. Keep `user_id`, `book_id`, `rating`, and `is_read`
2. Cast IDs and ratings into Spark ALS-compatible numeric types
3. Remove missing or invalid ratings outside the 1--5 scale
4. Keep one record per `(user_id, book_id)` pair

The model was implemented with Spark MLlib ALS using:

```python
implicitPrefs=False
ratingCol="rating"
```

## Broader Model Comparison

The full project compared:

1. Popularity baseline
2. Explicit-feedback ALS
3. Implicit-feedback ALS
4. Combined explicit + implicit feedback ALS

The combined model performed best overall, suggesting that rating information is more useful when combined with behavioral engagement signals rather than used alone.

## Contribution Scope

This standalone repository focuses on the explicit-feedback ALS implementation and includes comparison summaries from the broader team project for context.
