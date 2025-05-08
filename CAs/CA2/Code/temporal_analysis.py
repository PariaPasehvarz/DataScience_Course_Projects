
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, to_timestamp, when, window, count, avg, row_number
from pyspark.sql.window import Window
from pyspark.sql import Row

# Create Spark session with Mongo v3.0.1 compatible config
spark = SparkSession.builder \
    .appName("TemporalAnalysis") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/darooghe_db.transactions") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/darooghe_db") \
    .getOrCreate()

# Load transaction data
df = spark.read.format("mongo").load()
df = df.withColumn("timestamp", to_timestamp("timestamp"))
df = df.filter(col("status") == "approved")

# ======================================================
# PART I: Transactions Outside Category Business Hours
# ======================================================
df = df.withColumn("txn_hour", hour("timestamp"))

category_hours = spark.createDataFrame([
    ("retail", 9, 21),
    ("food_service", 10, 23),
    ("entertainment", 12, 2),
    ("transportation", 0, 24),
    ("government", 8, 16)
], ["merchant_category", "open_hour", "close_hour"])

df = df.join(category_hours, on="merchant_category", how="left")

# Correct logic for overnight hours
outside_hours_df = df.filter(
    ((col("open_hour") < col("close_hour")) & 
     ((col("txn_hour") < col("open_hour")) | (col("txn_hour") >= col("close_hour")))) |
    ((col("open_hour") > col("close_hour")) & 
     ~((col("txn_hour") >= col("open_hour")) | (col("txn_hour") < col("close_hour"))))
)

outside_hours_df.write.format("mongo") \
    .option("collection", "outside_business_hours") \
    .mode("overwrite") \
    .save()

# ======================================================
# PART II: Most Active Part of Day by Merchant Category
# ======================================================
df_daypart = df.withColumn("day_part",
    when(hour("timestamp").between(6, 11), "morning")
    .when(hour("timestamp").between(12, 17), "afternoon")
    .when(hour("timestamp").between(18, 21), "evening")
    .otherwise("night")
)

category_counts = df_daypart.groupBy("merchant_category", "day_part").count()

category_ref = spark.createDataFrame([
    Row(merchant_category="retail"),
    Row(merchant_category="food_service"),
    Row(merchant_category="entertainment"),
    Row(merchant_category="transportation"),
    Row(merchant_category="government")
])

joined_category_activity = category_ref.join(
    category_counts,
    on="merchant_category",
    how="left"
)

w = Window.partitionBy("merchant_category").orderBy(col("count").desc_nulls_last())
most_active = joined_category_activity.withColumn("rank", row_number().over(w)) \
    .filter(col("rank") == 1).drop("rank")

most_active.write.format("mongo") \
    .option("collection", "category_active_hours") \
    .mode("overwrite") \
    .save()

# ======================================================
# PART III: Sudden Transaction Spikes per Merchant
# ======================================================

# Spike: Any 1-minute window where the number of transactions exceeds 3× the merchant's average.

txn_per_min = df.groupBy(window("timestamp", "1 minute"), "merchant_id").count()

avg_txn_rate = txn_per_min.groupBy("merchant_id") \
    .agg(avg("count").alias("avg_txn_per_min"))

spikes_df = txn_per_min.join(avg_txn_rate, "merchant_id") \
    .filter(col("count") > col("avg_txn_per_min") * 3)

spikes_df.write.format("mongo") \
    .option("collection", "transaction_spikes") \
    .mode("overwrite") \
    .save()

spark.stop()
