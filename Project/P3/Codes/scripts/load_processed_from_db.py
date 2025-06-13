import pandas as pd
import os
import argparse
from database_connection import get_engine

TABLE_PREFIX = "processed_"
TRAIN_DB_PATH = os.path.join(os.path.dirname(__file__), "../database/IMDB_movies_train.db")
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "../database/IMDB_movies_test.db")

def load_feature_set_from_db(engine, feature_set_name: str, as_numpy: bool = True):
    table_name = f"{TABLE_PREFIX}{feature_set_name.lower()}"
    df = pd.read_sql_table(table_name, con=engine)
    if as_numpy:
        return df.to_numpy()
    return df

def main(db_path=None, mode=None):
    """
    Load processed features from database
    Args:
        db_path: Path to the database
        mode: 'train' or 'test'
    """
    # Determine which database to use
    if db_path is None:
        if mode == 'train':
            db_path = TRAIN_DB_PATH
        elif mode == 'test':
            db_path = TEST_DB_PATH
        else:
            # Default behavior - load from train database
            db_path = TRAIN_DB_PATH
            
    engine = get_engine(db_path=db_path)
    
    # Load data based on mode
    if mode == 'test':
        # In test mode, only load test data
        X_test_final = load_feature_set_from_db(engine, "X_test_final", as_numpy=True)
        Y_test = load_feature_set_from_db(engine, "Y_test", as_numpy=True)
        X_train_final, Y_train = None, None
        
        if X_test_final is not None:
            print(f"--- Shape of X_test_final: {X_test_final.shape}")
        if Y_test is not None:
            print(f"--- Shape of Y_test: {Y_test.shape}")
    else:
        # In train mode or default mode, only load train data
        X_train_final = load_feature_set_from_db(engine, "X_train_final", as_numpy=True)
        Y_train = load_feature_set_from_db(engine, "Y_train", as_numpy=True) 
        X_test_final, Y_test = None, None
        
        if X_train_final is not None:
            print(f"--- Shape of X_train_final: {X_train_final.shape}")
        if Y_train is not None:
            print(f"--- Shape of Y_train: {Y_train.shape}")
        
    return X_train_final, Y_train, X_test_final, Y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load processed features from database.")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help="Mode: train or test")
    args = parser.parse_args()
    
    X_train, Y_train_data, X_test, Y_test_data = main(mode=args.mode)
    
    if args.mode == 'train' and X_train is not None:
        print(f"Loaded X_train shape: {X_train.shape}")
    elif args.mode == 'test' and X_test is not None:
        print(f"Loaded X_test shape: {X_test.shape}")
