
from kafka import KafkaProducer
import json
from datetime import datetime, timedelta
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_transaction(tx, delay=0.1):
    producer.send("darooghe.transactions", tx)
    time.sleep(delay)

# Use a timestamp 1 minute ago for all transactions to ensure it's within watermark
base_time = datetime.utcnow() - timedelta(minutes=1)
iso_time = base_time.isoformat() + "Z"

# === A. Amount Anomaly ===
amount_anomaly_tx = {
    "transaction_id": "fraud-amount-anomaly",
    "timestamp": iso_time,
    "customer_id": "cust_307",
    "merchant_id": "merch_001",
    "merchant_category": "retail",
    "payment_method": "mobile",
    "amount": 99541900,
    "location": {"lat": 35.7, "lng": 51.4},
    "device_info": {"os": "Android", "app_version": "1.0", "device_model": "Pixel"},
    "status": "approved",
    "commission_type": "flat",
    "commission_amount": 50000,
    "vat_amount": 1000000,
    "total_amount": 16000000,
    "customer_type": "individual",
    "risk_level": 3,
    "failure_reason": None
}
send_transaction(amount_anomaly_tx)

# === B. Stable Velocity Check: same timestamp for all 6 events ===
for i in range(6):
    velocity_tx = {
        "transaction_id": f"fraud-velocity-{i}",
        "timestamp": iso_time,
        "customer_id": "cust_velocity",
        "merchant_id": "merch_002",
        "merchant_category": "food_service",
        "payment_method": "pos",
        "amount": 200000,
        "location": {"lat": 35.71, "lng": 51.41},
        "device_info": {"os": "Android", "app_version": "2.1", "device_model": "Galaxy"},
        "status": "approved",
        "commission_type": "progressive",
        "commission_amount": 8000,
        "vat_amount": 20000,
        "total_amount": 228000,
        "customer_type": "individual",
        "risk_level": 2,
        "failure_reason": None
    }
    send_transaction(velocity_tx, delay=0.05)

# === C. Geographical Impossibility ===
geo_tx_1 = {
    "transaction_id": "fraud-geo-1",
    "timestamp": iso_time,
    "customer_id": "cust_geo",
    "merchant_id": "merch_geo1",
    "merchant_category": "electronics",
    "payment_method": "mobile",
    "amount": 300000,
    "location": {"lat": 35.7, "lng": 51.4},
    "device_info": {"os": "iOS", "app_version": "3.0", "device_model": "iPhone"},
    "status": "approved",
    "commission_type": "flat",
    "commission_amount": 10000,
    "vat_amount": 30000,
    "total_amount": 340000,
    "customer_type": "individual",
    "risk_level": 1,
    "failure_reason": None
}
geo_tx_2 = {
    "transaction_id": "fraud-geo-2",
    "timestamp": (base_time + timedelta(minutes=3)).isoformat() + "Z",
    "customer_id": "cust_geo",
    "merchant_id": "merch_geo2",
    "merchant_category": "electronics",
    "payment_method": "mobile",
    "amount": 320000,
    "location": {"lat": 36.3, "lng": 59.6},
    "device_info": {"os": "iOS", "app_version": "3.0", "device_model": "iPhone"},
    "status": "approved",
    "commission_type": "flat",
    "commission_amount": 10000,
    "vat_amount": 32000,
    "total_amount": 362000,
    "customer_type": "individual",
    "risk_level": 1,
    "failure_reason": None
}
send_transaction(geo_tx_1)
time.sleep(0.5)
send_transaction(geo_tx_2)

producer.flush()
print("✅ All test transactions sent with stable timestamps. Waiting for Spark to process...")
time.sleep(10)
