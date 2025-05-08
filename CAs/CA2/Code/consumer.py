from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient
import json
import logging
import datetime
import time
import os
from pymongo import MongoClient
from prometheus_client import start_http_server, Gauge

# ---------- Prometheus Metric ----------
consumer_lag_gauge = Gauge(
    'kafka_consumer_lag',
    'Kafka Consumer Lag',
    ['topic', 'partition', 'group']
)

# ---------- Logging ----------
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level_str, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------- Schema ----------
expected_schema = {
    "transaction_id": str,
    "timestamp": str,
    "customer_id": str,
    "merchant_id": str,
    "merchant_category": str,
    "payment_method": str,
    "amount": int,
    "location": dict,
    "device_info": dict,
    "status": str,
    "commission_type": str,
    "commission_amount": int,
    "vat_amount": int,
    "total_amount": int,
    "customer_type": str,
    "risk_level": int,
    "failure_reason": (str, type(None))
}

# ---------- Kafka Config ----------
kafka_broker = os.getenv("KAFKA_BROKER", "localhost:9092")
consume_topic = "darooghe.transactions"
error_topic = "darooghe.error_logs"

consumer_conf = {
    'bootstrap.servers': kafka_broker,
    'group.id': 'darooghe-consumer-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_conf)
consumer.subscribe([consume_topic])

producer_conf = {'bootstrap.servers': kafka_broker}
producer = Producer(producer_conf)

# ---------- MongoDB ----------
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client['darooghe_db']
mongo_collection = mongo_db['transactions']

# ---------- Prometheus Monitoring ----------
def update_consumer_lag():
    admin = AdminClient({"bootstrap.servers": kafka_broker})
    group_id = consumer_conf['group.id']
    topic_partitions = consumer.assignment()

    if not topic_partitions:
        return

    committed = consumer.committed(topic_partitions, timeout=5)
    watermarks = {tp: consumer.get_watermark_offsets(tp, timeout=5) for tp in topic_partitions}

    for tp, offset in zip(topic_partitions, committed):
        high = watermarks.get(tp, (None, None))[1]
        if high is not None and offset.offset != -1001:
            lag = max(0, high - offset.offset)
            consumer_lag_gauge.labels(
                topic=tp.topic, partition=tp.partition, group=group_id
            ).set(lag)

# ---------- Validation ----------
def validate_schema(event):
    for field, expected_type in expected_schema.items():
        if field not in event:
            return False, f"Missing field: {field}"
        if not isinstance(event[field], expected_type):
            return False, f"Wrong type for {field}: expected {expected_type}, got {type(event[field])}"
    return True, None

def validate_transaction(event):
    errors = []
    try:
        total_amount = event["total_amount"]
        expected_total = event["amount"] + event["vat_amount"] + event["commission_amount"]
        if total_amount != expected_total:
            errors.append("ERR_AMOUNT")

        event_time = datetime.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        if event_time > now or (now - event_time) > datetime.timedelta(days=1):
            errors.append("ERR_TIME")

        if event["payment_method"] == "mobile":
            if event["device_info"].get("os") not in ["iOS", "Android"]:
                errors.append("ERR_DEVICE")
    except Exception as e:
        logging.error(f"Validation error: {e}")

    return errors

# ---------- Error Logging ----------
def produce_error_log(transaction_id, errors, event):
    error_event = {
        "transaction_id": transaction_id,
        "errors": errors,
        "data": event
    }
    producer.produce(
        error_topic,
        key=transaction_id,
        value=json.dumps(error_event),
        callback=lambda err, msg: logging.error(f"Failed to produce error log: {err}") if err else None
    )

# ---------- Cleanup ----------
def delete_old_transactions():
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    cutoff_time = now - timedelta(days=1)
    cutoff_time_str = cutoff_time.isoformat() + "Z"
    result = mongo_collection.delete_many({"timestamp": {"$lt": cutoff_time_str}})
    logging.info(f"Deleted {result.deleted_count} old transactions.")

# ---------- Main ----------
def main_loop():
    last_cleanup = time.time()
    last_metrics_update = time.time()

    mongo_collection.create_index([("transaction_day", 1)])
    mongo_collection.create_index([("merchant_id", 1)])

    # Start Prometheus server
    start_http_server(8000)
    logging.info("Prometheus metrics available at http://localhost:8000/metrics")

    try:
        flush_topic(kafka_broker, error_topic)
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logging.error(f"Consumer error: {msg.error()}")
                continue

            event = json.loads(msg.value().decode('utf-8'))
            transaction_id = event["transaction_id"]
            schema_valid, error_msg = validate_schema(event)

            if not schema_valid:
                produce_error_log(transaction_id, ["ERR_SCHEMA"], event)
                continue

            errors = validate_transaction(event)
            if errors:
                produce_error_log(transaction_id, errors, event)
            else:
                logging.info(f"Stored valid transaction: {transaction_id}")
                event["transaction_day"] = event["timestamp"][:10]
                mongo_collection.insert_one(event)

                customer_id = event["customer_id"]
                amount = event["amount"]

                try:
                    pipeline = [
                        {"$match": {"customer_id": customer_id, "status": "approved", "amount": {"$gt": 0}}},
                        {"$group": {"_id": "$customer_id", "avg_amount": {"$avg": "$amount"}}}
                    ]
                    result = list(mongo_collection.aggregate(pipeline))
                    if result:
                        avg_amount = result[0]["avg_amount"]
                        mongo_db.customer_averages.update_one(
                            {"customer_id": customer_id},
                            {"$set": {"avg_amount": avg_amount}},
                            upsert=True
                        )
                except Exception as e:
                    logging.error(f"Failed to update customer average for {customer_id}: {e}")

            producer.poll(0)

            if time.time() - last_cleanup > 300:
                delete_old_transactions()
                last_cleanup = time.time()

            if time.time() - last_metrics_update > 10:
                update_consumer_lag()
                last_metrics_update = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        producer.flush()

def flush_topic(broker, topic):
    admin_client = AdminClient({"bootstrap.servers": broker})
    topics = admin_client.list_topics(timeout=10).topics
    if topic in topics:
        fs = admin_client.delete_topics([topic], operation_timeout=30)
        for t, f in fs.items():
            try:
                f.result()
                logging.info(f"Topic {t} deleted")
            except Exception as e:
                logging.error(f"Deletion failed for topic {t}: {e}")
        time.sleep(10)

if __name__ == "__main__":
    main_loop()
