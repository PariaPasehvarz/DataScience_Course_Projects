import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import os
import joblib
import tensorflow as tf

from load_processed_from_db import main as load_data_from_db
from database_connection import get_engine # Though imported, not directly used in this script's logic

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")
MODEL_PATH = os.path.join(MODEL_DIR, "genre_prediction_model.keras")
METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.joblib") # Points to evaluation metrics
TRANSFORMER_DIR = os.path.join(os.path.dirname(__file__), "../data/transformers/")
MLB_PATH = os.path.join(TRANSFORMER_DIR, "mlb.joblib")
PREPROCESSOR_PATH = os.path.join(TRANSFORMER_DIR, "preprocessor.joblib")

MLFLOW_EXPERIMENT_NAME = "genre_prediction_pipeline_experiment"
MLFLOW_RUN_NAME = "DNN_Pipeline_Training_Run"
REGISTERED_MODEL_NAME = "genre_dnn_pipeline_predictor"

# os.environ['MLFLOW_TRACKING_URI'] = 'file:/path/to/your/mlruns' # Set if needed

def log_and_register_model_with_mlflow(model_path, metrics, X_test, Y_test, mlb_path, preprocessor_path):
    print(f"Starting MLflow logging for model: {model_path}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log parameters
        params_to_log = {
            "model_type": "Sequential_DNN",
            "dense_1_units": 384,
            "dense_2_units": 192,
            "dropout_1_rate": 0.4,
            "dropout_2_rate": 0.3,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss_function": "binary_crossentropy",
            "prediction_threshold": 0.35
        }
        mlflow.log_params(params_to_log)
        print(f"Logged parameters: {params_to_log}")

        # Log metrics
        mlflow.log_metrics(metrics)
        print(f"Logged metrics: {metrics}")

        # Log artifacts
        print(f"Logging MultiLabelBinarizer from: {mlb_path}")
        mlflow.log_artifact(mlb_path, artifact_path="transformers/mlb.joblib")
        print(f"Logging Preprocessor from: {preprocessor_path}")
        mlflow.log_artifact(preprocessor_path, artifact_path="transformers/preprocessor.joblib")
        
        plot_path = os.path.join(os.path.dirname(__file__), "../plots/training_history.png")
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, artifact_path="plots")
            print(f"Logged training history plot: {plot_path}")

        # Log Keras model
        print(f"Logging Keras model from: {model_path}")
        loaded_model = tf.keras.models.load_model(model_path)
        if X_test is not None and X_test.shape[0] > 0:
            input_example = X_test[0:1]
            mlflow.keras.log_model(
                loaded_model, 
                artifact_path="model_files",
                signature=mlflow.models.infer_signature(input_example, loaded_model.predict(input_example))
            )
        else:
             mlflow.keras.log_model(loaded_model, artifact_path="model_files")
        print("Keras model logged to MLflow.")
        model_uri = f"runs:/{run_id}/model_files"

        # Register model
        client = MlflowClient()
        try:
            print(f"Attempting to create registered model: {REGISTERED_MODEL_NAME}")
            client.create_registered_model(REGISTERED_MODEL_NAME)
            print(f"Registered model '{REGISTERED_MODEL_NAME}' created successfully.")
        except mlflow.exceptions.MlflowException as e:
            if "Registered Model" in str(e) and "already exists" in str(e):
                print(f"Registered model '{REGISTERED_MODEL_NAME}' already exists. Will proceed to create a new version.")
            else:
                print(f"An unexpected MLflow error occurred while trying to create registered model '{REGISTERED_MODEL_NAME}': {e}")
                raise
        
        print(f"Creating new version for model '{REGISTERED_MODEL_NAME}' from URI: {model_uri}")
        client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )
        print("Model version created and model registered.")

    print("MLflow logging and registration process complete.")

def main():
    print("Starting MLflow logging and registration script...")

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Ensure train_model.py has run successfully.")
        return
    if not os.path.exists(METRICS_PATH):
        print(f"Metrics file not found at {METRICS_PATH}. Ensure evaluate_model.py has run successfully and saved metrics.")
        return
    if not os.path.exists(MLB_PATH):
        print(f"MultiLabelBinarizer not found at {MLB_PATH}. Ensure feature_engineering.py has run.")
        return
    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Preprocessor not found at {PREPROCESSOR_PATH}. Ensure feature_engineering.py has run.")
        return

    metrics = joblib.load(METRICS_PATH)
    print(f"Loaded metrics: {metrics}")

    # Load test data for MLflow model signature
    print("Loading test data for MLflow model signature from test database...")
    # Pass mode='test' to ensure test data is loaded
    loaded_data = load_data_from_db(mode='test') 
    if loaded_data and len(loaded_data) == 4: # load_data_from_db returns (X_train, Y_train, X_test, Y_test)
        _, _, X_test_final, Y_test_final = loaded_data # For mode='test', X_train and Y_train will be None
    else:
        X_test_final, Y_test_final = None, None
        print(f"Warning: load_data_from_db(mode='test') did not return the expected data structure. Loaded: {loaded_data}")


    if X_test_final is None:
        print("Failed to load test data. Model signature will not be inferred by X_test sample.")

    log_and_register_model_with_mlflow(MODEL_PATH, metrics, X_test_final, Y_test_final, MLB_PATH, PREPROCESSOR_PATH)

    print("MLflow script finished.")

if __name__ == "__main__":
    main()
