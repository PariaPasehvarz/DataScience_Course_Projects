import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
import joblib
from load_data import load_data # Restored
from database_connection import get_engine # Restored
import os
import argparse

# Define base directory for data consistently
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/")
# PREPROCESSED_DATA_DIR = BASE_DATA_DIR # No longer loading from preprocessed CSVs here
DATA_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "processed_features/")
TRANSFORMER_DIR = os.path.join(BASE_DATA_DIR, "transformers/")
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRANSFORMER_DIR, exist_ok=True)

MLB_PATH = os.path.join(TRANSFORMER_DIR, "mlb.joblib")
PREPROCESSOR_PATH = os.path.join(TRANSFORMER_DIR, "preprocessor.joblib")

# Path for saving original_index for test set
X_TEST_ORIGINAL_INDEX_PATH = os.path.join(DATA_OUTPUT_DIR, "X_test_original_index.csv")

# Database paths (similar to preprocess.py)
DB_DIR = os.path.join(os.path.dirname(__file__), '../database')
TRAIN_DB_PATH = os.path.join(DB_DIR, 'IMDB_movies_train.db')
TEST_DB_PATH = os.path.join(DB_DIR, 'IMDB_movies_test.db')

def main(mode):
    fit_transformers = False
    engine = None
    if mode == 'train':
        print("Loading raw train data from database...")
        # input_csv_path = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_train.csv")
        engine = get_engine(db_path=TRAIN_DB_PATH)
        df_raw = load_data(engine)
        fit_transformers = True
    elif mode == 'test':
        print("Loading raw test data from database...")
        # input_csv_path = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_test.csv")
        engine = get_engine(db_path=TEST_DB_PATH)
        df_raw = load_data(engine)
        fit_transformers = False
    else:
        raise ValueError("Mode must be 'train' or 'test'")

    print(f"Initial {mode} data shape: {df_raw.shape}")

    # --- Re-integrate preprocessing steps --- 
    df_raw = df_raw.replace("nan", np.nan)
    df_raw = df_raw.replace("None", np.nan) # Added for robustness

    if "original_index" in df_raw.columns:
        # Ensure original_index is numeric before incrementing, coerce errors
        df_raw["original_index"] = pd.to_numeric(df_raw["original_index"], errors='coerce')
        # Increment only non-NaN values. If it was coerced to NaN, it remains NaN.
        df_raw.loc[df_raw["original_index"].notna(), "original_index"] += 1
    else:
        print(f"Warning: 'original_index' column not found in {mode} data. Will not be incremented or saved for test.")

    # Strip whitespace from all string columns
    for col in df_raw.select_dtypes(include=['object']).columns:
        df_raw[col] = df_raw[col].str.strip()
    
    # Convert budget_x and revenue to numeric, coercing errors. This is crucial before filtering.
    if 'budget_x' in df_raw.columns:
        df_raw['budget_x'] = pd.to_numeric(df_raw['budget_x'], errors='coerce')
    if 'revenue' in df_raw.columns:
        df_raw['revenue'] = pd.to_numeric(df_raw['revenue'], errors='coerce')
    
    # Create revenue_to_budget ratio feature
    if 'budget_x' in df_raw.columns and 'revenue' in df_raw.columns:
        df_raw['revenue_to_budget'] = df_raw['revenue'] / df_raw['budget_x'].replace(0, np.nan)
        df_raw['revenue_to_budget'] = df_raw['revenue_to_budget'].fillna(0)
    
    # Create year feature from release_date or date columns
    date_columns = ['release_date', 'date_x', 'year']
    year_created = False
    for col in date_columns:
        if col in df_raw.columns and not year_created:
            if col == 'year':
                df_raw['year'] = pd.to_numeric(df_raw['year'], errors='coerce')
                year_created = True
            else:
                try:
                    df_raw['year'] = pd.to_datetime(df_raw[col], errors='coerce').dt.year
                    year_created = True
                except:
                    continue
    
    if not year_created and 'year' not in df_raw.columns:
        raise ValueError("Cannot create 'year' feature: no suitable date column found")

    # Define columns that are critical and should not be NaN or zero after conversion for certain filters
    # For example, budget and revenue for the > 10000 filter.
    # The generic isnull() | (df == 0) mask might be too broad if 0 is valid for some numeric cols.
    # Let's be more specific for budget/revenue filtering.
    initial_rows = len(df_raw)
    df_raw = df_raw.dropna(subset=['budget_x', 'revenue']) # Drop rows where budget or revenue became NaN after coercion
    df_raw = df_raw[(df_raw["revenue"] > 10000) & (df_raw["budget_x"] > 10000)]
    print(f"Rows after budget/revenue filtering (>10000 and not NaN): {len(df_raw)} (dropped {initial_rows - len(df_raw)})")

    if 'status' in df_raw.columns:
        initial_rows = len(df_raw)
        df_raw = df_raw[df_raw["status"] == "Released"]
        print(f"Rows after status='Released' filtering: {len(df_raw)} (dropped {initial_rows - len(df_raw)})")
    else:
        print("Warning: 'status' column not found. Skipping status filter.")

    if 'name' in df_raw.columns:
        initial_rows = len(df_raw)
        df_raw = df_raw.drop_duplicates(subset='name', keep='first')
        print(f"Rows after dropping duplicates by 'name': {len(df_raw)} (dropped {initial_rows - len(df_raw)})")
    else:
        print("Warning: 'name' column not found. Skipping duplicate removal by name.")
    
    print(f"Shape of df_raw after preprocessing: {df_raw.shape}")
    if df_raw.empty:
        raise ValueError(f"DataFrame is empty after preprocessing steps for mode '{mode}'. Check data quality and filters.")

    # --- End of re-integrated preprocessing --- 

    if 'overview' not in df_raw.columns:
        print(f"Warning: 'overview' column not found in {mode} data after preprocessing. Filling with empty strings.")
        df_raw['overview'] = ""
    else:
        df_raw['overview'] = df_raw['overview'].fillna("")

    if 'genre' not in df_raw.columns:
        # load_data is expected to create a 'genre' column (comma-separated string)
        raise ValueError(f"'genre' column not found in {mode} data after preprocessing. Check load_data function.")

    # Retain original_index for test mode before dropping it for X features
    original_index_series = None
    if mode == 'test':
        if 'original_index' in df_raw.columns:
            print("Extracting original_index for test set.")
            original_index_series = df_raw['original_index']
        else:
            # This warning was already present, but now it refers to data from DB
            print(f"Warning: 'original_index' column not found in data loaded from DB for test mode. Cannot save it.")

    print("Generating overview embeddings...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    df_raw['overview_vector'] = df_raw['overview'].apply(lambda x: st_model.encode(x).tolist())

    print("Binarizing genres...")
    if fit_transformers:
        mlb = MultiLabelBinarizer()
        # The 'genre' column from load_data is a comma-separated string of genre names.
        # We need to split it into lists of strings.
        # df_raw['genre_list'] = df_raw['genre'].fillna('').astype(str).apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
        df_raw['genre'] = df_raw['genre'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])
        Y = mlb.fit_transform(df_raw['genre'])
        joblib.dump(mlb, MLB_PATH)
        print(f"MultiLabelBinarizer saved to {MLB_PATH}")
    else:
        df_raw['genre'] = df_raw['genre'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])
        mlb = joblib.load(MLB_PATH)
        # df_raw['genre_list'] = df_raw['genre'].fillna('').astype(str).apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
        Y = mlb.transform(df_raw['genre'])

    # Columns to drop to create the feature set X.
    # original_index is handled separately for test mode now.
    columns_to_drop_for_X = ['genre', 'genre_list', 'overview', 'overview_vector']
    if 'original_index' in df_raw.columns: # Always drop for X features, but it's saved for test.
        columns_to_drop_for_X.append('original_index')
    if 'name' in df_raw.columns:
        columns_to_drop_for_X.append('name')
    # Avoid dropping 'id' if it's one of the numerical features by mistake.
    # This logic seems a bit complex, ensure 'id' is not a feature name.
    if 'id' in df_raw.columns and 'id' not in ['score', 'budget_x', 'revenue', 'year', 'revenue_to_budget', 'orig_lang', 'country']:
        columns_to_drop_for_X.append('id')

    # Ensure all columns in columns_to_drop_for_X actually exist in df_raw to avoid errors with errors='ignore'
    columns_to_drop_for_X = [col for col in columns_to_drop_for_X if col in df_raw.columns]

    X_df_for_struct = df_raw.drop(columns=columns_to_drop_for_X)
    available_cols = X_df_for_struct.columns.tolist()
    numerical_cols_ideal = ['score', 'budget_x', 'revenue', 'year', 'revenue_to_budget']
    categorical_cols_ideal = ['orig_lang', 'country']
    numerical_cols = [col for col in numerical_cols_ideal if col in available_cols]
    categorical_cols = [col for col in categorical_cols_ideal if col in available_cols]

    X_structured_raw = X_df_for_struct[numerical_cols + categorical_cols]
    X_overview_embeddings = np.array(df_raw['overview_vector'].tolist())

    print("Preprocessing structured data...")
    if fit_transformers:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='drop' 
        )
        X_struct_processed = preprocessor.fit_transform(X_structured_raw)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        print(f"Preprocessor saved to {PREPROCESSOR_PATH}")
    else:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        X_struct_processed = preprocessor.transform(X_structured_raw)

    if hasattr(X_struct_processed, "toarray"):
        X_struct_processed = X_struct_processed.toarray()

    print("Combining structured and overview features...")
    X_final = np.hstack([X_struct_processed, X_overview_embeddings])

    if mode == 'train':
        print(f"X_train_final shape: {X_final.shape}, Y_train shape: {Y.shape}")
        pd.DataFrame(X_final).to_csv(os.path.join(DATA_OUTPUT_DIR, "X_train_final.csv"), index=False)
        pd.DataFrame(Y).to_csv(os.path.join(DATA_OUTPUT_DIR, "Y_train.csv"), index=False)
    else: # mode == 'test'
        print(f"X_test_final shape: {X_final.shape}, Y_test shape: {Y.shape}")
        pd.DataFrame(X_final).to_csv(os.path.join(DATA_OUTPUT_DIR, "X_test_final.csv"), index=False)
        pd.DataFrame(Y).to_csv(os.path.join(DATA_OUTPUT_DIR, "Y_test.csv"), index=False)
        if original_index_series is not None:
            pd.DataFrame(original_index_series).to_csv(X_TEST_ORIGINAL_INDEX_PATH, index=False)
            print(f"Original index for test set saved to {X_TEST_ORIGINAL_INDEX_PATH}")

    print(f"Processed feature CSVs saved to {DATA_OUTPUT_DIR}")

    # Save to database using the mode
    print(f"Saving processed features to database for mode: {mode}...")
    from save_processed_to_db import main as save_to_db_main
    save_to_db_main(mode=mode) # Pass the mode
    print(f"Successfully initiated saving processed data to database for mode: {mode}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering for train or test mode.")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode: train or test")
    args = parser.parse_args()
    main(args.mode)