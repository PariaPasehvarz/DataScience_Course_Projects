from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['darooghe_db']
transactions = db['transactions']

def summarize_transactions():
    pipeline = [
        {
            "$group": {
                "_id": {
                    "merchant_id": "$merchant_id",
                    "transaction_day": "$transaction_day"
                },
                "total_transactions": {"$sum": 1},
                "total_amount": {"$sum": "$amount"},
                "total_vat": {"$sum": "$vat_amount"},
                "total_commission": {"$sum": "$commission_amount"}
            }
        },
        {
            "$out": "summarized_transactions"
        }
    ]
    transactions.aggregate(pipeline)
    print("Summarized transactions saved.")

def summarize_commissions():
    pipeline = [
        {
            "$group": {
                "_id": {
                    "merchant_category": "$merchant_category",
                    "transaction_day": "$transaction_day"
                },
                "total_commission": {"$sum": "$commission_amount"},
                "transaction_count": {"$sum": 1}
            }
        },
        {
            "$out": "commission_reports"
        }
    ]
    transactions.aggregate(pipeline)
    print("Commission reports saved.")

if __name__ == "__main__":
    summarize_transactions()
    summarize_commissions()
