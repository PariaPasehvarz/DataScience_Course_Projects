
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, to_json, struct
from pyspark.sql.types import *
from pyspark.sql.streaming import DataStreamWriter

# 1. Spark session setup
spark = SparkSession.builder \
    .appName("CommissionAuditSystem") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/darooghe_db.commission_summary") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Define transaction schema
transaction_schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("customer_id", StringType()),
    StructField("merchant_id", StringType()),
    StructField("merchant_category", StringType()),
    StructField("payment_method", StringType()),
    StructField("amount", LongType()),
    StructField("commission_type", StringType()),
    StructField("commission_amount", LongType()),
    StructField("vat_amount", LongType()),
    StructField("total_amount", LongType()),
    StructField("customer_type", StringType()),
    StructField("risk_level", IntegerType()),
    StructField("status", StringType()),
    StructField("failure_reason", StringType(), True),
    StructField("location", StructType([
        StructField("lat", DoubleType()),
        StructField("lng", DoubleType())
    ]), True),
    StructField("device_info", StructType([
        StructField("os", StringType()),
        StructField("app_version", StringType()),
        StructField("device_model", StringType())
    ]), True)
])

# 3. Kafka stream read
kafka_stream = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "darooghe.transactions") \
    .option("startingOffsets", "latest") \
    .load()

parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), transaction_schema).alias("data")) \
    .select("data.*") \
    .filter(col("status") == "approved")

# 4. UDF: Recommend commission type
def recommend_commission(amount):
    if amount < 500000:
        return "regressive"
    elif amount < 1500000:
        return "flat"
    else:
        return "progressive"

from pyspark.sql.types import StringType
recommend_udf = udf(recommend_commission, StringType())

stream_with_recommendation = parsed_stream.withColumn(
    "recommended_commission", recommend_udf(col("amount"))
)

# 5. Static lookup: historical best models per merchant category
commission_summary = spark.read.format("mongo") \
    .option("collection", "commission_summary") \
    .load() \
    .select("merchant_category", "best_model")

# 6. Join and audit logic
audited_stream = stream_with_recommendation.join(
    commission_summary,
    on="merchant_category",
    how="left"
)

discrepancies = audited_stream.filter(
    col("recommended_commission") != col("best_model")
)

# 7. Convert to JSON and send to Kafka topic
output_df = discrepancies.select(to_json(struct("*")).alias("value"))

output_df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.commission_audit") \
    .option("checkpointLocation", "/tmp/commission_audit_checkpoint") \
    .outputMode("append") \
    .start() \
    .awaitTermination()
