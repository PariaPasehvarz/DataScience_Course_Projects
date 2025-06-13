import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, hamming_loss
import os
import joblib
from load_processed_from_db import main as load_data_from_db
from database_connection import get_engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")
TRANSFORMERS_DIR = os.path.join(os.path.dirname(__file__), "../data/transformers/")

MODEL_PATH = os.path.join(MODEL_DIR, "genre_prediction_model.keras")
EVALUATION_METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.joblib")
MLB_PATH = os.path.join(TRANSFORMERS_DIR, "mlb.joblib")

def evaluate_model_and_save_predictions(engine):
    # Load test data using mode='test'
    loaded_data = load_data_from_db(mode='test')
    X_test_final, Y_test = None, None

    if loaded_data and len(loaded_data) == 4:
        _, _, X_test_final, Y_test = loaded_data
    else:
        print("Failed to load test data from database. Exiting evaluation.")
        return

    if X_test_final is None or Y_test is None:
        print("Failed to load test data. Exiting evaluation.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Exiting evaluation.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    mlb = None
    if os.path.exists(MLB_PATH):
        mlb = joblib.load(MLB_PATH)
    else:
        print(f"Warning: MLB file not found at {MLB_PATH}. Predictions may lack class names; evaluation might be affected by dimension mismatch.")
        if model.output_shape[-1] != Y_test.shape[1]:
             print(f"Model output dimension ({model.output_shape[-1]}) and Y_test dimension ({Y_test.shape[1]}) mismatch without MLB. Exiting.")
             return

    Y_probs = model.predict(X_test_final)
    Y_pred = (Y_probs >= 0.35).astype(int)

    f1_micro = f1_score(Y_test, Y_pred, average='micro')
    f1_macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    hamming = hamming_loss(Y_test, Y_pred)

    print("\nModel Evaluation Results:")
    print(f"Micro F1-score:  {f1_micro:.4f}")
    print(f"Macro F1-score:  {f1_macro:.4f}")
    print(f"Hamming Loss:    {hamming:.4f}")
    
    evaluation_metrics = {
        "micro_f1": f1_micro,
        "macro_f1": f1_macro,
        "hamming_loss": hamming
    }
    joblib.dump(evaluation_metrics, EVALUATION_METRICS_PATH)
    print(f"Evaluation metrics saved to {EVALUATION_METRICS_PATH}")

    if mlb is None:
        if Y_probs.shape[1] == Y_test.shape[1]:
            prob_cols = [f'prob_class_{i}' for i in range(Y_probs.shape[1])]
            pred_cols = [f'pred_class_{i}' for i in range(Y_pred.shape[1])]
        else:
            print("MLB not found and dimension mismatch. Skipping DB save for predictions.")
            return
    else:
        prob_cols = [f'prob_{cls}' for cls in mlb.classes_]
        pred_cols = [f'pred_{cls}' for cls in mlb.classes_]

    expected_cols = len(mlb.classes_) if mlb else Y_test.shape[1]
    if Y_probs.shape[1] != expected_cols or Y_pred.shape[1] != expected_cols:
        print(f"Mismatch in prediction columns vs expected. Skipping DB save.")
        return

    Y_probs_df = pd.DataFrame(Y_probs, columns=prob_cols)
    Y_pred_df = pd.DataFrame(Y_pred, columns=pred_cols)
    
    if isinstance(X_test_final, pd.DataFrame) and X_test_final.index.name is not None:
        ids = X_test_final.index.to_series()
    else:
        ids = pd.Series(range(len(Y_probs_df)), name="sample_id")

    predictions_df = pd.concat([ids.reset_index(drop=True), Y_probs_df, Y_pred_df], axis=1)
    id_col_name = str(ids.name) if ids.name is not None else "sample_id"
    predictions_df = predictions_df.rename(columns={predictions_df.columns[0]: id_col_name})

    def clean_column_names(columns):
        cleaned = []
        seen = set()
        for col in columns:
            # Normalize column name: lowercase and stripped
            new_col = str(col).strip().lower()
            # Deduplicate by adding suffix if needed
            count = 1
            original = new_col
            while new_col in seen:
                new_col = f"{original}_{count}"
                count += 1
            seen.add(new_col)
            cleaned.append(new_col)
        return cleaned

    predictions_df.columns = clean_column_names(predictions_df.columns)

    # Use connection context for proper database handling
    with engine.connect() as conn:
        predictions_df.to_sql('test_set_predictions', conn, if_exists='replace', index=False)
        conn.commit()
    print(f"Predictions saved to 'test_set_predictions' table in test database.")

def main():
    # Use test database for evaluation
    test_db_path = os.path.join(os.path.dirname(__file__), "../database/IMDB_movies_test.db")
    engine = get_engine(db_path=test_db_path)
    evaluate_model_and_save_predictions(engine)

if __name__ == "__main__":
    main()