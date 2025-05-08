from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, to_date, hour, count, sum, when

# MongoDB connection parameters
database = "darooghe_db"
collection = "transactions"

# Create Spark session
spark = SparkSession.builder \
    .appName("TransactionPatternAnalysis") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/") \
    .config("spark.mongodb.input.database", database) \
    .config("spark.mongodb.input.collection", collection) \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/") \
    .config("spark.mongodb.output.database", database) \
    .getOrCreate()

# Load and filter data
df = spark.read.format("mongo").load()
df = df.filter(col("status") == "approved")

# Parse timestamp field to Spark timestamp type
df = df.withColumn("parsed_timestamp", to_timestamp("timestamp"))
df = df.withColumn("txn_date", to_date("parsed_timestamp"))
df = df.withColumn("txn_hour", hour("parsed_timestamp"))

# DAILY TRENDS: count and total amount per day
daily_summary = df.groupBy("txn_date").agg(
    count("*").alias("transaction_count"),
    sum("amount").alias("total_amount")
)

# HOURLY TRENDS: average transaction count by hour
hourly_summary = df.groupBy("txn_hour").agg(
    count("*").alias("transaction_count"),
    sum("amount").alias("total_amount")
)

# CUSTOMER TYPE analysis
customer_type_summary = df.groupBy("customer_type").agg(
    count("*").alias("transaction_count"),
    sum("amount").alias("total_amount")
)

# MERCHANT CATEGORY trends
merchant_summary = df.groupBy("merchant_category").agg(
    count("*").alias("transaction_count"),
    sum("amount").alias("total_amount")
)

# CUSTOMER SEGMENTATION based on transaction patterns
customer_stats = df.groupBy("customer_id").agg(
    count("*").alias("txn_count"),
    sum("amount").alias("total_spent")
)

# Segment customers
segmented_customers = customer_stats.withColumn(
    "segment",
    when((col("total_spent") > 100000000) & (col("txn_count") >= 10), "High Value")
    .when((col("txn_count") >= 90), "Frequent")
    .when((col("txn_count") > 50) & (col("total_spent") / col("txn_count") < 950000), "Low Value")
    .otherwise("Other")
)

# ANALYZE SEGMENT STATS
segment_summary = segmented_customers.groupBy("segment").agg(
    count("customer_id").alias("customer_count"),
    sum("txn_count").alias("total_transactions"),
    sum("total_spent").alias("total_spent"),
    (sum("total_spent") / sum("txn_count")).alias("avg_transaction_value")
)

# Write all outputs back to MongoDB
daily_summary.write.mode("overwrite").format("mongo").option("collection", "txn_daily_summary").save()
hourly_summary.write.mode("overwrite").format("mongo").option("collection", "txn_hourly_summary").save()
customer_type_summary.write.mode("overwrite").format("mongo").option("collection", "txn_customer_type_summary").save()
merchant_summary.write.mode("overwrite").format("mongo").option("collection", "txn_merchant_summary").save()
segmented_customers.write.mode("overwrite").format("mongo").option("collection", "txn_customer_segments").save()
segment_summary.write.mode("overwrite").format("mongo").option("collection", "txn_segment_summary").save()

# MERCHANT trends (based on individual merchant_id)
merchant_trends = df.groupBy("merchant_id").agg(
    count("*").alias("transaction_count"),
    sum("amount").alias("total_amount")
)

merchant_trends.write.mode("overwrite").format("mongo").option("collection", "txn_merchant_trends").save()

# Show in console
print("Daily Summary:")
daily_summary.show(truncate=False)
print("Hourly Summary:")
hourly_summary.orderBy("txn_hour").show(truncate=False)
print("Customer Type Summary:")
customer_type_summary.show(truncate=False)
print("Merchant Category Summary:")
merchant_summary.show(truncate=False)
print("Customer Segments:")
segmented_customers.show(truncate=False)
print("Customer Segment Summary:")
segment_summary.show(truncate=False)

spark.stop()
