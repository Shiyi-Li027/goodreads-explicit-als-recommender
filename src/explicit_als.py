#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    $ spark-submit --deploy-mode client explicit_als.py
Or
    $ nohup spark-submit \
        --deploy-mode client \
            --driver-memory 3g \
                --executor-memory 2g \
                    --conf spark.sql.shuffle.partitions=400 \
                        --conf spark.executor.memoryOverhead=1g \
                            --conf spark.network.timeout=800s \
                                --conf spark.executor.heartbeatInterval=60s \
                                    explicit_als.py > explicit_als_full2p_9grid.log 2>&1 &
'''

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics

SEED = 42
TRAIN_SAMPLE_FRAC = 0.02  # Train ALS on 2% of the preprocessed training interactions.
TRAIN_SAMPLE_SEED = 37
VALIDATION_USER_SAMPLE_SIZE = 5000  # Number of validation users to keep for validation evaluation.
VALIDATION_USER_SEED = 37
TEST_USER_SAMPLE_SIZE = 5000  # Number of test users to keep for final test evaluation.
TEST_USER_SEED = 38

# Define the paths to the parquet files. 
# Update these paths for your own HDFS or local Spark environment.
parquet_files = {
    "train": "hdfs:///user/<username>/goodreads/splits/train.parquet",
    "validation_observed": "hdfs:///user/<username>/goodreads/splits/validation_observed.parquet",
    "validation_heldout": "hdfs:///user/<username>/goodreads/splits/validation_heldout.parquet",
    "test_observed": "hdfs:///user/<username>/goodreads/splits/test_observed.parquet",
    "test_heldout": "hdfs:///user/<username>/goodreads/splits/test_heldout.parquet",
}

# Define output paths for saving explicit ALS results.
# Update these paths for your own HDFS or local Spark environment.
output_base ="hdfs:///user/<username>/goodreads_outputs/explicit_als_full2p"

validation_results_path = output_base + "/validation_results_csv"
test_metrics_path = output_base + "/test_metrics_csv"
test_recs_parquet_path = output_base + "/test_top100_recommendations_parquet"
test_recs_sample_csv_path = output_base + "/test_top100_sample_20_users_csv"

# Full hyperparameter grid:
"""
rank = [5, 10, 20, 50]
regParam = [0.01, 0.05, 0.1, 0.5]
maxIter = [5, 10]
"""

K = 100  # Number of top recommendations to generate for each user.
num_candidate_items = 500  # Number of candidate items to generate before filtering out books already seen by the user. （num_candidate_items = 200）

# Conservative hyperparameter grid.
# This keeps already-seen filtering but avoids training too many ALS models at once.
rank = [5, 10, 20]
regParam = [0.05, 0.1, 0.5]
maxIter = [5]  # maxIter=2

# Create a SparkSession.
spark = (
    SparkSession.builder
    .appName("Explicit_ALS")
    .config("spark.sql.shuffle.partitions", "400")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

# Preprocess the data.
def preprocess_explicit_df(df):
    """
    Select explicit-rating columns and filter out invalid ratings.

    Args:
        df (DataFrame): The raw interaction DataFrame.

    Returns:
        DataFrame: A preprocessed explicit-rating DataFrame with user_id, book_id, and rating columns.
    """
    # Create a new DataFrame with only the relevant columns and filter out invalid ratings.
    explicit_df = (
        df
        .select(
            F.col("user_id").cast("int").alias("user_id"),
            F.col("book_id").cast("int").alias("book_id"),
            F.col("is_read").cast("int").alias("is_read")
            F.col("rating").cast("float").alias("rating")  # Spark ALS expects integer user/item ids and a numeric rating column.
        )
        # Keep only valid explicit ratings. Rating 0 or null does not provide an explicit preference score.
        .filter(F.col("rating").isNotNull())
        .filter((F.col("rating") >= 1.0) & (F.col("rating") <= 5.0))
        .groupBy("user_id", "book_id")
        .agg(F.max("rating").alias("rating"))  # Keep the maximum rating record for each user-book pair.
    )

    return explicit_df


def load_and_preprocess_data(spark, parquet_files=parquet_files):
    """
    Preprocess the data by reading parquet files, selecting explicit rating columns, filtering out invalid ratings, and creating smaller train/validation/test splits.

    Args:
        spark (SparkSession): The SparkSession object.
        parquet_files (dict): A dictionary containing the paths to the parquet files.

    Returns:
        dict: A dictionary containing the preprocessed and sampled DataFrames for each dataset.
    """
    # Read the original parquet split files.
    train_raw = spark.read.parquet(parquet_files["train"])
    validation_observed_raw = spark.read.parquet(parquet_files["validation_observed"])
    validation_heldout_raw = spark.read.parquet(parquet_files["validation_heldout"])
    test_observed_raw = spark.read.parquet(parquet_files["test_observed"])
    test_heldout_raw = spark.read.parquet(parquet_files["test_heldout"])

    # Train on a 2% random sample of train.parquet with seed 37.
    train_sample_raw = train_raw.sample(
        withReplacement=False,
        fraction=TRAIN_SAMPLE_FRAC,
        seed=TRAIN_SAMPLE_SEED
    )

    # Preprocess the sampled training data for explicit ALS.
    train_sample = preprocess_explicit_df(train_sample_raw)

    # Preprocess validation and test observed/heldout data before selecting users for ALS evaluation.
    validation_observed = preprocess_explicit_df(validation_observed_raw)
    validation_heldout = preprocess_explicit_df(validation_heldout_raw)
    test_observed = preprocess_explicit_df(test_observed_raw)
    test_heldout = preprocess_explicit_df(test_heldout_raw)

    # Sample 5000 validation users from validation_observed with seed 37.
    # Then keep both validation_observed and validation_heldout records for those users.
    validation_users = (
        validation_observed
        .select("user_id")
        .distinct()
        .orderBy(F.rand(VALIDATION_USER_SEED))
        .limit(VALIDATION_USER_SAMPLE_SIZE)
    )

    validation_observed_sample = validation_observed.join(
        validation_users,
        on="user_id",
        how="inner"
    )

    validation_heldout_sample = validation_heldout.join(
        validation_users,
        on="user_id",
        how="inner"
    )

    # Sample 5000 test users from test_observed with seed 38.
    # Then keep both test_observed and test_heldout records for those users.
    test_users = (
        test_observed
        .select("user_id")
        .distinct()
        .orderBy(F.rand(TEST_USER_SEED))
        .limit(TEST_USER_SAMPLE_SIZE)
    )

    test_observed_sample = test_observed.join(
        test_users,
        on="user_id",
        how="inner"
    )

    test_heldout_sample = test_heldout.join(
        test_users,
        on="user_id",
        how="inner"
    )

    sampled_dfs = {
        "train": train_sample,
        "validation_observed": validation_observed_sample,
        "validation_heldout": validation_heldout_sample,
        "test_observed": test_observed_sample,
        "test_heldout": test_heldout_sample
    }

    # Create temporary views for the sampled DataFrames in case we need SQL queries later.
    for name, df in sampled_dfs.items():
        df.createOrReplaceTempView(name)

    print("Sampled split summary:")
    for name, df in sampled_dfs.items():
        print(name, "rows:", df.count(), "users:", df.select("user_id").distinct().count())

    print(
        "validation sampled users:",
        validation_users.count()
    )

    print(
        "validation overlap users:",
        validation_observed_sample.select("user_id").distinct()
        .join(validation_heldout_sample.select("user_id").distinct(), on="user_id", how="inner")
        .count()
    )

    print(
        "test sampled users:",
        test_users.count()
    )

    print(
        "test overlap users:",
        test_observed_sample.select("user_id").distinct()
        .join(test_heldout_sample.select("user_id").distinct(), on="user_id", how="inner")
        .count()
    )

    return sampled_dfs

def train_als_model(train_df, rank=5, maxIter=5, regParam=0.1, numUserBlocks = 50, numItemBlocks=50):
    """
    Train an explicit-feedback ALS model using the preprocessed training DataFrame.

    Args:
        train_df (DataFrame): The preprocessed training DataFrame containing user_id, book_id, and rating columns.
        rank (int): The number of latent factors for each user and item.
        maxIter (int): The maximum number of ALS iterations.
        regParam (float): The regularization parameter.

    Returns:
        ALS_Model: The trained ALS model.
    """
    als = ALS(
        userCol="user_id",
        itemCol="book_id",
        ratingCol="rating",
        implicitPrefs=False,  # Since we are working with explicit ratings, we set implicitPrefs to False.
        rank=rank,  # The number of hidden factors for each user and item automatically learned by the model.
        maxIter=maxIter,  # The maximum number of iterations used to alternate between optimizing user factors and item factors.
        regParam=regParam,  # The regularization parameter used to prevent overfitting by adding a penalty term to the ALS objective function.
        numUserBlocks=numUserBlocks,  # Split users into blocks to make ALS training more stable on the full dataset.
        numItemBlocks=numItemBlocks,  # Split books into blocks to reduce the work done by each ALS task.
        coldStartStrategy="drop"  # Drop NaN predictions caused by users/items unseen during model fitting.
    )

    ALS_model = als.fit(train_df)

    return ALS_model


def recommend_for_users(model, observed_df, k=100, num_candidate_items=500):
    """
    Generate top-k recommendations for users in the observed dataset using the trained ALS model.

    Args:
        model (ALSModel): The trained ALS model.
        observed_df (DataFrame): The observed interactions used to identify users and filter out already seen books.
        k (int): The number of final recommendations to keep for each user.
        num_candidate_items (int): The number of candidate recommendations generated before filtering.

    Returns:
        DataFrame: A DataFrame containing user_id, book_id, predicted_rating, and rank_order for the top-k recommendations.
    """
    # Get the unique users from observed_df to generate recommendations for these users.
    unique_users = observed_df.select("user_id").distinct()

    # Generate candidate recommendations for the users in the observed set.
    raw_rec = model.recommendForUserSubset(unique_users, num_candidate_items)

    # Explode the recommendations column so that each recommended book becomes one row.
    # Repartition by user_id because the later filtering and ranking steps are also based on user_id.
    recs = (
        raw_rec
        .select(
            F.col("user_id"),
            F.explode(F.col("recommendations")).alias("rec")
        )
        .select(
            F.col("user_id"),
            F.col("rec.book_id").cast("int").alias("book_id"),
            F.col("rec.rating").alias("predicted_rating")
        )
        .repartition("user_id")
    )

    # Get the books that each user has already seen in the observed set.
    books_already_seen = (
        observed_df
        .select("user_id", "book_id")
        .dropDuplicates(["user_id", "book_id"])
        .repartition("user_id")
    )

    # Filter out books that the user has already seen.
    filtered_recs = (
        recs
        .join(books_already_seen, on=["user_id", "book_id"], how="left_anti")
    )

    # Rank the remaining recommendations for each user by predicted rating.
    window = Window.partitionBy("user_id").orderBy(F.desc("predicted_rating"))

    # Keep only the top-k recommendations for each user.
    reordered_recs = (
        filtered_recs
        .withColumn("rank_order", F.row_number().over(window))
        .filter(F.col("rank_order") <= k)
    )

    return reordered_recs


def evaluate_model(recs_df, heldout_df, k=100):
    """
    Evaluate the recommendation results by comparing recommended books with held-out books.

    Args:
        recs_df (DataFrame): The recommendation DataFrame containing user_id, book_id, predicted_rating, and rank_order.
        heldout_df (DataFrame): The held-out interactions used as the ground truth.
        k (int): The cutoff for ranking metrics.

    Returns:
        dict: A dictionary containing ndcg@k, recall@k, MAP, and number of evaluated users.
    """
    # Collect held-out books as the ground-truth item list for each user.
    true_recs = (
        heldout_df
        .select(
            F.col("user_id").cast("int").alias("user_id"),
            F.col("book_id").cast("int").alias("book_id")
        )
        .filter((F.col("is_read") == 1) & (F.col("rating") >= 4))
        .groupBy("user_id")
        .agg(F.collect_set("book_id").alias("true_items"))
    )

    # Collect recommended books as an ordered predicted item list for each user.
    pred_recs = (
        recs_df
        .groupBy("user_id")
        .agg(
            F.expr(
                "transform(sort_array(collect_list(struct(rank_order, book_id))), x -> x.book_id)"
            )
            .alias("pred_items")
        )
    )

    # Join predicted recommendation lists with the held-out ground truth by user_id.
    eval_df = (
        pred_recs
        .join(true_recs, on="user_id", how="inner")
    )

    n_eval_users = eval_df.count()

    if n_eval_users == 0:
        return {
            "MAP": 0.0,
            f"ndcg@{k}": 0.0,
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            "num_eval_users": 0
        }

    # RankingMetrics expects an RDD of (predicted_items, true_items).
    prediction_and_labels = (
        eval_df
        .select("pred_items", "true_items")
        .rdd
        .map(lambda row: (row["pred_items"], row["true_items"]))
    )

    # Create a RankingMetrics object using the predicted and true item lists.
    ranking_metrics = RankingMetrics(prediction_and_labels)

    # Compute ranking metrics based on the top-k recommendation results.
    ndcg_at_k = ranking_metrics.ndcgAt(k)
    precision_at_k = ranking_metrics.precisionAt(k)
    map_score = ranking_metrics.meanAveragePrecision

    # Compute recall@k using Spark SQL to avoid version differences in RankingMetrics.
    recall_at_k = (
        eval_df
        .withColumn("hits", F.size(F.array_intersect("pred_items", "true_items")))
        .withColumn("num_true", F.size("true_items"))
        .withColumn("recall", F.col("hits") / F.col("num_true"))
        .agg(F.avg("recall").alias("recall"))
        .first()["recall"]
    )

    return {
        "MAP": float(map_score),
        f"ndcg@{k}": float(ndcg_at_k),
        f"precision@{k}": float(precision_at_k),
        f"recall@{k}": float(recall_at_k),
        "num_eval_users": n_eval_users
    }


def tune_hyperparameters(train_df, validation_observed_df, validation_heldout_df, rank_list=rank, regParam_list=regParam, maxIter_list=maxIter, k=100):
    """
    Tune ALS hyperparameters using the validation set.

    For each hyperparameter setting, we train the model on train_df plus validation_observed_df,
    generate recommendations for validation users, and evaluate the recommendations against validation_heldout_df.
    
    Args:
        train_df (DataFrame): The preprocessed training DataFrame.
        validation_observed_df (DataFrame): The observed interactions for validation users.
        validation_heldout_df (DataFrame): The held-out interactions used to evaluate validation recommendations.
        rank_list (list): A list of rank values to try.
        regParam_list (list): A list of regularization parameter values to try.
        maxIter_list (list): A list of maximum iteration values to try.
        k (int): The cutoff for top-k recommendation evaluation.

    Returns:
        tuple: best_params, best_metrics, and a list of all validation tuning results.
    """
    best_params = None
    best_metrics = None
    best_ndcg = float("-inf")  # The range of ndcg@k is between 0 and 1, where 1 represents an ideal ranking.
    results = []

    # Combine train_df and validation_observed_df for validation tuning.
    # train_df helps the model learn the general user-book patterns, while validation_observed_df gives the model the observed history of validation users.
    # validation_heldout_df is not included in training and is only used for evaluation.
    validation_train_df = train_df.unionByName(validation_observed_df)

    for rank in rank_list:
        for regParam in regParam_list:
            for maxIter in maxIter_list:

                print(f"Starting to train ALS model with parameters: rank={rank}, regParam={regParam}, maxIter={maxIter}")

                # Train ALS model using train_df and validation_observed_df.
                model = train_als_model(validation_train_df, rank=rank, maxIter=maxIter, regParam=regParam)

                # Generate recommendations for validation users based on their observed history.
                recs = recommend_for_users(model=model, observed_df=validation_observed_df, k=k, num_candidate_items=num_candidate_items)

                # Evaluate the generated recommendations against validation held-out interactions.
                metrics = evaluate_model(recs_df=recs, heldout_df=validation_heldout_df, k=k)

                result = {
                    "rank": rank,
                    "regParam": regParam,
                    "maxIter": maxIter,
                    **metrics
                }

                print(result)
                results.append(result)

                # Use ndcg@k as the main metric for selecting hyperparameters because it considers both whether relevant books are recommended and whether they are ranked near the top.
                if metrics[f"ndcg@{k}"] > best_ndcg:
                    best_ndcg = metrics[f"ndcg@{k}"]
                    best_params = {
                        "rank": rank,
                        "regParam": regParam,
                        "maxIter": maxIter
                    }
                    best_metrics = metrics
                    
    if best_params is None:
        raise RuntimeError("No valid validation result was produced.")

    return best_params, best_metrics, results


def final_test_evaluation(train_df, validation_observed_df, test_observed_df, test_heldout_df, best_params):
    """
    Train the final ALS model using the best hyperparameters and evaluate it on the test set.

    test_observed_df is included so the model can learn the observed history of test users.
    test_heldout_df is not used in training and is only used for final evaluation.
    
    Args:
        train_df (DataFrame): The preprocessed training DataFrame.
        validation_observed_df (DataFrame): The observed interactions for validation users.
        test_observed_df (DataFrame): The observed interactions for test users.
        test_heldout_df (DataFrame): The held-out interactions used to evaluate test recommendations.
        best_params (dict): The best hyperparameters selected from validation tuning.

    Returns:
        tuple: test_metrics and test_recs. test_metrics contains the final test metrics, and test_recs contains the top-k recommendations for test users.
    """
    # Combine all observed interactions available before test evaluation.
    # test_heldout_df is not included to avoid data leakage.
    final_train_df = (
        train_df
        .unionByName(validation_observed_df)
        .unionByName(test_observed_df)
    )

    # Train the final ALS model using the selected best hyperparameters.
    final_model = train_als_model(
        final_train_df,
        rank=best_params["rank"],
        regParam=best_params["regParam"],
        maxIter=best_params["maxIter"]
    )

    # Generate recommendations for test users based on their observed history.
    test_recs = recommend_for_users(
        model=final_model,
        observed_df=test_observed_df,
        k=K,
        num_candidate_items=num_candidate_items
    )

    # Evaluate the test recommendations against test held-out interactions.
    test_metrics = evaluate_model(
        recs_df=test_recs,
        heldout_df=test_heldout_df,
        k=K
        )

    return test_metrics, test_recs


if __name__ == "__main__":
    # Load and preprocess all train, validation, and test split files.
    dfs = load_and_preprocess_data(spark, parquet_files)

    # Get each preprocessed DataFrame from the dictionary.
    train_df = dfs["train"]
    validation_observed_df = dfs["validation_observed"]
    validation_heldout_df = dfs["validation_heldout"]
    test_observed_df = dfs["test_observed"]
    test_heldout_df = dfs["test_heldout"]

    # Tune hyperparameters using validation data.
    best_params, best_val_metrics, tuning_results = tune_hyperparameters(
        train_df=train_df,
        validation_observed_df=validation_observed_df,
        validation_heldout_df=validation_heldout_df
    )

    # Print all validation tuning results for comparison.
    print("All validation tuning results:")
    for row in tuning_results:
        print(row)

    # Evaluate the final model on the test set using the best hyperparameters.
    test_metrics, test_recs = final_test_evaluation(
        train_df=train_df,
        validation_observed_df=validation_observed_df,
        test_observed_df=test_observed_df,
        test_heldout_df=test_heldout_df,
        best_params=best_params
    )

    print("Best hyperparameter set selected:")
    print(best_params)

    print("Best validation metrics:")
    print(best_val_metrics)

    print("Test metrics:")
    print(test_metrics)

    # Save validation tuning results as CSV.
    spark.createDataFrame(tuning_results) \
        .coalesce(1) \
        .write.mode("overwrite") \
        .option("header", True) \
        .csv(validation_results_path)

    # Save final test metrics as CSV.
    spark.createDataFrame([test_metrics]) \
        .coalesce(1) \
        .write.mode("overwrite") \
        .option("header", True) \
        .csv(test_metrics_path)

    # Save full test top-100 recommendations as compressed Parquet.
    test_recs.write.mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(test_recs_parquet_path)

    # Get the first 20 users from the test recommendation results.
    sample_users = test_recs.select("user_id") \
        .distinct() \
        .orderBy("user_id") \
        .limit(20)

    # Save these 20 users' top-100 recommendations as CSV for the report.
    test_recs.join(sample_users, on="user_id", how="inner") \
        .orderBy("user_id", "rank_order") \
        .coalesce(1) \
        .write.mode("overwrite") \
        .option("header", True) \
        .csv(test_recs_sample_csv_path)

    print("Saved validation tuning results to:")
    print(validation_results_path)

    print("Saved test metrics to:")
    print(test_metrics_path)

    print("Saved full test top-100 recommendations to:")
    print(test_recs_parquet_path)

    print("Saved sample top-100 recommendations for 20 users to:")
    print(test_recs_sample_csv_path)

    spark.stop()
