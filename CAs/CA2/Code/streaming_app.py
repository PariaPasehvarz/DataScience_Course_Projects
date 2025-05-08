from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window, sum, expr, desc, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

# 1. Create Spark session
spark = SparkSession.builder \
    .appName("DaroogheStreamingApp") \
    .getOrCreate()

# 2. Define the schema of the transaction
transactionSchema = StructType([
    StructField("transaction_id", StringType()),
    StructField("timestamp", StringType()),  # parse to Timestamp later
    StructField("customer_id", StringType()),
    StructField("merchant_id", StringType()),
    StructField("merchant_category", StringType()),
    StructField("payment_method", StringType()),
    StructField("amount", DoubleType()),
    StructField("location", StructType([
        StructField("lat", DoubleType()),
        StructField("lng", DoubleType()),
    ])),
    StructField("device_info", StructType([
        StructField("os", StringType()),
        StructField("app_version", StringType()),
        StructField("device_model", StringType()),
    ])),
    StructField("status", StringType()),
    StructField("commission_type", StringType()),
    StructField("commission_amount", DoubleType()),
    StructField("vat_amount", DoubleType()),
    StructField("total_amount", DoubleType()),
    StructField("customer_type", StringType()),
    StructField("risk_level", IntegerType()),
    StructField("failure_reason", StringType())
])

# 3. Read stream from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "darooghe.transactions") \
    .option("startingOffsets", "earliest") \
    .load()

# 4. Parse the Kafka 'value' field as JSON
parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_value") \
    .select(from_json(col("json_value"), transactionSchema).alias("data")) \
    .select("data.*")

# 5. Convert string timestamp to actual TimestampType
parsed_df = parsed_df.withColumn("timestamp", to_timestamp(col("timestamp")))

# A. Total commission by type per minute
commission_by_type = parsed_df \
    .withWatermark("timestamp", "2 minutes") \
    .groupBy(window(col("timestamp"), "1 minute"), col("commission_type")) \
    .agg(sum("commission_amount").alias("total_commission")) \
    .selectExpr("to_json(struct(*)) AS value")

query1 = commission_by_type.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.commission_by_type") \
    .option("checkpointLocation", "/tmp/checkpoints/commission_type") \
    .outputMode("update") \
    .start()

# B. Commission ratio by merchant category
commission_ratio = parsed_df \
    .withWatermark("timestamp", "2 minutes") \
    .filter(col("amount") > 0) \
    .groupBy(window(col("timestamp"), "1 minute"), col("merchant_category")) \
    .agg(
        (sum("commission_amount") / sum("amount")).alias("commission_ratio")
    ) \
    .selectExpr("to_json(struct(*)) AS value")

query2 = commission_ratio.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.commission_ratio") \
    .option("checkpointLocation", "/tmp/checkpoints/commission_ratio") \
    .outputMode("update") \
    .start()

# C. Top commission-generating merchants in 5-minute windows
highest_commission_merchants = parsed_df \
    .withWatermark("timestamp", "6 minutes") \
    .groupBy(window(col("timestamp"), "5 minutes"), col("merchant_id")) \
    .agg(sum("commission_amount").alias("total_commission")) \
    .orderBy(desc("total_commission")) \
    .limit(5) \
    .selectExpr("to_json(struct(*)) AS value")

query3 = highest_commission_merchants.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.top_merchants") \
    .option("checkpointLocation", "/tmp/checkpoints/top_merchants") \
    .outputMode("complete") \
    .start()

# Block the application from exiting
query1.awaitTermination()
query2.awaitTermination()
query3.awaitTermination()