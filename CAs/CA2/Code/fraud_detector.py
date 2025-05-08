from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window, sum, expr, desc, udf, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from dateutil.parser import parse as parse_time
from pyspark.sql import DataFrame
from pymongo import MongoClient
from geopy.distance import geodesic
import json

# 1. Create Spark session
spark = SparkSession.builder \
    .appName("DaroogheStreamingApp") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

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
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# 4. Parse the Kafka 'value' field as JSON
parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_value") \
    .select(from_json(col("json_value"), transactionSchema).alias("data")) \
    .select("data.*")

# 5. Convert string timestamp to actual TimestampType
parsed_df = parsed_df.withColumn("timestamp", to_timestamp(col("timestamp")))

# Fraud Detection Rule C - Amount Anomaly
client = MongoClient("mongodb://localhost:27017")
avg_data = client.darooghe_db.customer_averages.find()
avg_map = {doc['customer_id']: doc['avg_amount'] for doc in avg_data}
client.close()

broadcast_avg_map = spark.sparkContext.broadcast(avg_map)

def amount_anomaly(customer_id, amount):
    avg = broadcast_avg_map.value.get(customer_id)
    if avg is not None and amount > 10 * avg:
        return True
    return False

amount_anomaly_udf = udf(amount_anomaly, StringType())

amount_anomaly_df = parsed_df.withColumn("amount_anomaly", amount_anomaly_udf(col("customer_id"), col("amount"))) \
    .filter(col("amount_anomaly") == "true") \
    .selectExpr("transaction_id", "customer_id", "'amount_anomaly' as fraud_type")

# Fraud Detection Rule A - Velocity Check
velocity_df = parsed_df.withWatermark("timestamp", "3 minutes") \
    .groupBy(window(col("timestamp"), "2 minutes"), col("customer_id")) \
    .agg(count("transaction_id").alias("tx_count")) \
    .filter("tx_count > 5") \
    .selectExpr("customer_id", "'velocity_check' as fraud_type") \
    .withColumn("transaction_id", expr("null"))

# Fraud Detection Rule B - Geographical Impossibility

def process_geo_batch(df, epoch_id):
    alerts = []
    client = MongoClient("mongodb://localhost:27017")
    location_collection = client.darooghe_db.customer_last_location

    if df.limit(1).count() == 0:
        print("[INFO] Skipping empty micro-batch")
        return

    rows = df.collect()

    for row in rows:
        customer_id = row['customer_id']
        transaction_id = row['transaction_id']
        curr_time = row['timestamp']
        curr_lat = row['location']['lat']
        curr_lng = row['location']['lng']

        # Parse current timestamp if needed
        if isinstance(curr_time, str):
            curr_time = parse_time(curr_time)

        prev = location_collection.find_one({ "customer_id": customer_id })

        if prev:
            prev_time = prev['timestamp']
            prev_lat = prev['lat']
            prev_lng = prev['lng']

            # Parse previous timestamp if needed
            if isinstance(prev_time, str):
                prev_time = parse_time(prev_time)

            time_diff = (curr_time - prev_time).total_seconds() / 60
            distance = geodesic((prev_lat, prev_lng), (curr_lat, curr_lng)).km


            if time_diff <= 5 and distance > 50:
                alert = json.dumps({
                    "transaction_id": transaction_id,
                    "customer_id": customer_id,
                    "fraud_type": "geographical_impossibility"
                })
                print(f"[GEO FRAUD DETECTED] {alert}")
                alerts.append(alert)

        # Always update latest known location
        location_collection.update_one(
            { "customer_id": customer_id },
            {
                "$set": {
                    "timestamp": curr_time.isoformat(),
                    "lat": curr_lat,
                    "lng": curr_lng
                }
            },
            upsert=True
        )

    client.close()

    if alerts:
        alert_df = spark.createDataFrame([(a,) for a in alerts], ["value"])
        alert_df \
            .write \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("topic", "darooghe.fraud_alerts") \
            .option("checkpointLocation", "/tmp/checkpoints/geo_kafka") \
            .save()


parsed_df.writeStream \
    .foreachBatch(process_geo_batch) \
    .option("checkpointLocation", "/tmp/checkpoints/geo_check") \
    .start()

def process_amount_velocity_batch(df: DataFrame, epoch_id):
    if df.isEmpty():
        print("[INFO] No amount/velocity frauds in this micro-batch.")
        return

    rows = df.collect()
    alerts = []

    for row in rows:
        alert = {
            "transaction_id": row["transaction_id"] if row["transaction_id"] else "N/A",
            "customer_id": row["customer_id"],
            "fraud_type": row["fraud_type"]
        }
        if alert["fraud_type"] == "amount_anomaly":
            print(f"[AMOUNT FRAUD DETECTED] {json.dumps(alert)}")
        elif alert["fraud_type"] == "velocity_check":
            print(f"[VELOCITY FRAUD DETECTED] {json.dumps(alert)}")
        alerts.append(json.dumps(alert))

    if alerts:
        alert_df = spark.createDataFrame([(a,) for a in alerts], ["value"])
        alert_df \
            .write \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("topic", "darooghe.fraud_alerts") \
            .option("checkpointLocation", "/tmp/checkpoints/fraud_av_kafka") \
            .save()

amount_velocity_df = amount_anomaly_df.unionByName(velocity_df)

amount_velocity_df.writeStream \
    .foreachBatch(process_amount_velocity_batch) \
    .option("checkpointLocation", "/tmp/checkpoints/fraud_av_foreach") \
    .start() \
    .awaitTermination()

