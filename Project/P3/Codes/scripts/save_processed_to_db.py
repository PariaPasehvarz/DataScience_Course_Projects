import pandas as pd
import os
from sqlalchemy import text
from database_connection import get_engine
import argparse # Added argparse

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/processed_features/")
TABLE_PREFIX = "processed_"

# Added TRAIN_DB_PATH and TEST_DB_PATH
TRAIN_DB_PATH = os.path.join(os.path.dirname(__file__), "../database/IMDB_movies_train.db")
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "../database/IMDB_movies_test.db")

def save_feature_set_to_db(engine, feature_set_name: str):
    csv_file_name = f"{feature_set_name}.csv"
    csv_path = os.path.join(PROCESSED_DATA_DIR, csv_file_name)
    table_name = f"{TABLE_PREFIX}{feature_set_name.lower()}" 

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    with engine.connect() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
        df.to_sql(table_name, con=connection, if_exists='replace', index=False)
        connection.commit()
    print(f"Saved {feature_set_name} to {table_name} in {engine.url.database}.")

def main(db_path=None, mode=None): # Modified main to accept db_path and mode
    """
    Save processed features to database.
    Args:
        db_path: Path to the database.
        mode: 'train' or 'test'.
    """
    print(f"Saving feature sets to database in {mode} mode...")

    # Determine which database to use
    if db_path is None:
        if mode == 'train':
            db_path = TRAIN_DB_PATH
        elif mode == 'test':
            db_path = TEST_DB_PATH
        else:
            # Default behavior - save to train database if mode is not specified clearly
            print("Warning: Mode not specified or invalid. Defaulting to 'train' mode.")
            db_path = TRAIN_DB_PATH
            mode = 'train' # Ensure mode is set for feature_sets logic
            
    engine = get_engine(db_path=db_path)
    
    if mode == 'train':
        feature_sets = ["X_train_final", "Y_train"]
    elif mode == 'test':
        feature_sets = ["X_test_final", "Y_test"]
    else: # Should not happen if mode is validated by argparse or defaulted above
        print(f"Error: Invalid mode '{mode}'. No features will be saved.")
        return
        
    for fs_name in feature_sets:
        save_feature_set_to_db(engine, fs_name)
        
    print(f"Finished saving {mode} feature sets to database: {db_path}")

if __name__ == "__main__":
    # Added argparse for mode
    parser = argparse.ArgumentParser(description="Save processed features to the specified database.")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode: 'train' or 'test' to determine which database and feature sets to use.")
    args = parser.parse_args()
    
    main(mode=args.mode)
