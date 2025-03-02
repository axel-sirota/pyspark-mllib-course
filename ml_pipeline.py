"""
ml_pipeline.py

This script demonstrates an end-to-end machine learning workflow on an Azure HDInsight Spark cluster.
It covers:
  - Synthetic data generation (for regression and classification)
  - Feature engineering using Spark's Pipeline API
  - Training a regression model (using GBTRegressor)
  - Training a classification model (using XGBoost via SparkXGBClassifier)
  - Hyperparameter tuning with CrossValidator and ParamGridBuilder
  - Logging metrics and models using MLflow
  - Distributed inference using broadcast joins and performance optimizations

Before running this script:
  - Ensure that MLflow is installed (e.g., pip install mlflow)
  - Ensure that the XGBoost Spark integration is available (xgboost4j-spark)
  - Ensure you have the Hadoop configuration files (core-site.xml, yarn-site.xml, etc.)
    for your HDInsight cluster if running locally.
  - This script explicitly sets the master to "yarn" so that when executed, it connects to your HDInsight cluster.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, when, col, broadcast

# Import components for building ML pipelines
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, PCA
)
# Regression model and evaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# For hyperparameter tuning
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Classification: using XGBoost for Spark
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# MLflow for experiment tracking; ensure MLflow is installed and configured
import mlflow
import mlflow.spark

# -------------------------------------------------------------------
# Section 0: Connect to the HDInsight Cluster
# -------------------------------------------------------------------
# When running locally, ensure your local environment has the HDInsight cluster's Hadoop configuration.
# Setting master to "yarn" will instruct Spark to submit jobs to the HDInsight cluster.
spark = SparkSession.builder \
    .appName("HDInsight_ML_Pipeline") \
    .master("yarn") \
    .config("spark.jars", "hdfs:///user/spark/libs/xgboost4j-spark_2.13-2.1.4.jar,hdfs:///user/spark/libs/xgboost4j_2.13-2.1.4.jar") \
    .getOrCreate()

# -------------------------------------------------------------------
# Section 1: Data Generation and Preprocessing
# -------------------------------------------------------------------
# Generate synthetic datasets (1 million rows each) to simulate large-scale data.
# In production, load your data from external sources (e.g. Azure Blob Storage).

# --- A. Regression Dataset ---
regression_df = (
    spark.range(0, 1000000)
         .withColumn("feature1", rand() * 100)  # Numeric values between 0 and 100
         .withColumn("feature2", rand() * 50)   # Numeric values between 0 and 50
         .withColumn("category", when(rand() > 0.5, "A").otherwise("B"))
         .withColumn("target", col("feature1") * 2.5 + col("feature2") * -1.5 + (rand() * 10))
)
regression_df = regression_df.repartition(10).cache()

# --- B. Classification Dataset ---
classification_df = (
    spark.range(0, 1000000)
         .withColumn("feat1", rand() * 10)
         .withColumn("feat2", rand() * 5)
         .withColumn("cat_feature", when(rand() > 0.5, "X").otherwise("Y"))
         .withColumn("label", when((col("feat1") + col("feat2") + rand() * 5) > 10, 1).otherwise(0))
)
classification_df = classification_df.repartition(10).cache()

# -------------------------------------------------------------------
# Section 2: Feature Engineering Pipelines
# -------------------------------------------------------------------
# Regression Pipeline: Assemble numeric features, scale, and optionally apply PCA.
reg_numeric_cols = ["feature1", "feature2"]
assembler_reg = VectorAssembler(inputCols=reg_numeric_cols, outputCol="num_features")
scaler_reg = StandardScaler(inputCol="num_features", outputCol="scaled_features")
pca_reg = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
reg_pipeline = Pipeline(stages=[assembler_reg, scaler_reg, pca_reg])

# Classification Pipeline: Index categorical feature, one-hot encode, and assemble with numeric features.
indexer = StringIndexer(inputCol="cat_feature", outputCol="cat_index")
encoder = OneHotEncoder(inputCols=["cat_index"], outputCols=["cat_encoded"])
assembler_clf = VectorAssembler(inputCols=["feat1", "feat2", "cat_encoded"], outputCol="features")
clf_pipeline = Pipeline(stages=[indexer, encoder, assembler_clf])

# -------------------------------------------------------------------
# Section 3: Regression Model Training using Gradient Boosted Trees
# -------------------------------------------------------------------
with mlflow.start_run(run_name="Regression_GBT"):
    reg_prepared = reg_pipeline.fit(regression_df).transform(regression_df)
    train_reg, test_reg = reg_prepared.randomSplit([0.8, 0.2], seed=42)

    gbt = GBTRegressor(featuresCol="pca_features", labelCol="target", maxIter=20)
    paramGrid_reg = (ParamGridBuilder()
                     .addGrid(gbt.maxDepth, [3, 5])
                     .addGrid(gbt.maxIter, [10, 20])
                     .build())

    cv_reg = CrossValidator(estimator=gbt,
                            estimatorParamMaps=paramGrid_reg,
                            evaluator=RegressionEvaluator(labelCol="target", metricName="rmse"),
                            numFolds=3)
    reg_cv_model = cv_reg.fit(train_reg)

    predictions_reg = reg_cv_model.transform(test_reg)
    evaluator_reg = RegressionEvaluator(labelCol="target", metricName="rmse")
    rmse = evaluator_reg.evaluate(predictions_reg)

    mlflow.log_metric("reg_rmse", rmse)
    mlflow.spark.log_model(reg_cv_model.bestModel, "regression_model")
    print(f"Regression Model RMSE: {rmse}")

# -------------------------------------------------------------------
# Section 4: Classification Model Training using XGBoost
# -------------------------------------------------------------------
with mlflow.start_run(run_name="Classification_XGBoost"):
    clf_prepared = clf_pipeline.fit(classification_df).transform(classification_df)
    train_clf, test_clf = clf_prepared.randomSplit([0.8, 0.2], seed=42)

    xgb_clf = SparkXGBClassifier(featuresCol="features", labelCol="label", numWorkers=2)
    paramGrid_clf = (ParamGridBuilder()
                     .addGrid(xgb_clf.maxDepth, [3, 5])
                     .addGrid(xgb_clf.eta, [0.1, 0.3])
                     .build())

    cv_clf = CrossValidator(estimator=xgb_clf,
                            estimatorParamMaps=paramGrid_clf,
                            evaluator=BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC"),
                            numFolds=3)
    clf_cv_model = cv_clf.fit(train_clf)

    predictions_clf = clf_cv_model.transform(test_clf)
    evaluator_clf = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator_clf.evaluate(predictions_clf)
    multi_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    accuracy = multi_eval.evaluate(predictions_clf)

    mlflow.log_metric("clf_auc", auc)
    mlflow.log_metric("clf_accuracy", accuracy)
    mlflow.spark.log_model(clf_cv_model.bestModel, "classification_model")
    print(f"Classification Model AUC: {auc}, Accuracy: {accuracy}")

# -------------------------------------------------------------------
# Section 5: Distributed Inference and Workflow Optimization
# -------------------------------------------------------------------
lookup_data = [("A", 1), ("B", 2)]
lookup_df = spark.createDataFrame(lookup_data, ["category", "lookup_value"])
joined_df = regression_df.join(broadcast(lookup_df), on="category", how="left")
joined_df = joined_df.cache()
joined_df.show(5)

# -------------------------------------------------------------------
# Best Practices and Comments:
# -------------------------------------------------------------------
# 1. Resource Management:
#    - Repartition and cache large DataFrames to improve performance.
#
# 2. Hyperparameter Tuning:
#    - The parameter grids here are minimal for demonstration. Expand them cautiously in production.
#
# 3. MLflow:
#    - This script logs metrics and models using MLflow. Configure a tracking server if desired.
#
# 4. Distributed Inference:
#    - Broadcast joins are used to optimize joining with small lookup tables.
#
# 5. Reproducibility:
#    - A fixed random seed (42) is used for data splits and model training.

# End of script.
