import pandas as pd
import numpy as np
import argparse
from database_connection import get_engine
from import_to_db import import_to_database
from load_data import load_data
import os

# Paths for train and test databases
base_dir = os.path.dirname(__file__)
db_dir = os.path.join(base_dir, '../database')
train_db_path = os.path.join(db_dir, 'IMDB_movies_train.db')
test_db_path = os.path.join(db_dir, 'IMDB_movies_test.db')

def preprocess_and_save(db_path, label):
    engine = get_engine(db_path=db_path)
    
    df = load_data(engine)
    df = df.replace("nan", np.nan)
    df["original_index"] += 1 
    
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    mask = ((df.isnull()) | (df == 0))
    rows_with_issues = df[mask.any(axis=1)]
    df = df.drop(index=rows_with_issues.index)
    df = df[(df["revenue"] > 10000) & (df["budget_x"] > 10000)]
    df = df[df["status"] == "Released"]
    df = df.drop_duplicates(subset='name', keep='first')
    
    import_to_database(df=df, engine=engine)
    print(f"Data preprocessed successfully for {label}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess train or test database.")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode: train or test")
    args = parser.parse_args()
    
    if args.mode == 'train':
        preprocess_and_save(train_db_path, 'train')
    elif args.mode == 'test':
        preprocess_and_save(test_db_path, 'test')