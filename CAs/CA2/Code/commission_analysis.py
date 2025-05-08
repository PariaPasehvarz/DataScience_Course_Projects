from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg, col, when

# MongoDB connection parameters
database = "darooghe_db"
input_collection = "transactions"
output_collection = "commission_summary"

# Create SparkSession with MongoDB read and write configs
spark = SparkSession.builder \
    .appName("CommissionAnalysis") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/") \
    .config("spark.mongodb.input.database", database) \
    .config("spark.mongodb.input.collection", input_collection) \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/") \
    .config("spark.mongodb.output.database", database) \
    .config("spark.mongodb.output.collection", output_collection) \
    .getOrCreate()

# Load and filter data
df = spark.read.format("mongo").load()
df = df.filter(col("status") == "approved")

# Simulate commission structures on transaction level
df = df.withColumn(
    "flat_model_commission", col("amount") * 0.02
).withColumn(
    "progressive_model_commission",
    when(col("amount") < 500000, col("amount") * 0.01)
    .when(col("amount") < 1500000, col("amount") * 0.02)
    .otherwise(col("amount") * 0.025)
).withColumn(
    "regressive_model_commission",
    when(col("amount") < 500000, col("amount") * 0.03)
    .when(col("amount") < 1500000, col("amount") * 0.02)
    .otherwise(col("amount") * 0.015)
)

# Aggregate metrics by merchant category
commission_summary = df.groupBy("merchant_category").agg(
    sum("commission_amount").alias("actual_total_commission"),
    sum("flat_model_commission").alias("flat_model_total"),
    sum("progressive_model_commission").alias("progressive_model_total"),
    sum("regressive_model_commission").alias("regressive_model_total"),
    avg("amount").alias("avg_txn_amount"),
    sum("amount").alias("total_transaction_amount")
)

# Identify best model per category
commission_summary = commission_summary.withColumn(
    "best_model",
    when(
        (col("progressive_model_total") > col("flat_model_total")) &
        (col("progressive_model_total") > col("regressive_model_total")),
        "progressive"
    ).when(
        col("flat_model_total") > col("regressive_model_total"),
        "flat"
    ).otherwise("regressive")
)

# Show results
commission_summary.select(
    "merchant_category",
    "actual_total_commission",
    "flat_model_total",
    "progressive_model_total",
    "regressive_model_total",
    "best_model"
).show(truncate=False)

# Save to MongoDB
commission_summary.write \
    .format("mongo") \
    .mode("overwrite") \
    .save()

spark.stop()
