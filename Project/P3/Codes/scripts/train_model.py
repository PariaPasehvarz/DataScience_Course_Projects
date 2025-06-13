import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os
import joblib

from load_processed_from_db import main as load_data_from_db
from database_connection import get_engine


# Set the path to the train database
TRAIN_DB_PATH = os.path.join(os.path.dirname(__file__), "../database/IMDB_movies_train.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "../plots/")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "genre_prediction_model.keras")
HISTORY_PLOT_PATH = os.path.join(PLOTS_DIR, "training_history.png")
TRAINING_PROCESS_METRICS_PATH = os.path.join(MODEL_DIR, "training_process_metrics.joblib") 

def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(384, activation='relu', kernel_regularizer=l2(0.0005), input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(192, activation='relu', kernel_regularizer=l2(0.0005)),
        Dropout(0.3),
        Dense(output_dim, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def plot_history(history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

def train_model_procedure(X_train_final, Y_train, X_val_for_split, Y_val_for_split):
    input_dim = X_train_final.shape[1]
    output_dim = Y_train.shape[1]
    model = build_model(input_dim, output_dim)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    if X_val_for_split is not None and Y_val_for_split is not None:
        validation_data = (X_val_for_split, Y_val_for_split)
        validation_split = None
    else:
        validation_data = None
        validation_split = 0.1

    history = model.fit(
        X_train_final, Y_train,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=30,
        batch_size=64,
        verbose=1,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    plot_history(history, HISTORY_PLOT_PATH)

    training_metrics = {
        "final_train_loss": history.history['loss'][-1] if 'loss' in history.history and history.history['loss'] else None,
        "final_val_loss": history.history['val_loss'][-1] if 'val_loss' in history.history and history.history['val_loss'] else None
    }
    joblib.dump(training_metrics, TRAINING_PROCESS_METRICS_PATH)
    
    return model, history

def main():
    # Explicitly use the train database and mode
    loaded_data = load_data_from_db(db_path=TRAIN_DB_PATH, mode='train')
    
    X_train_final, Y_train, X_val_for_split, Y_val_for_split = None, None, None, None

    if loaded_data and len(loaded_data) == 4: 
        X_train_final, Y_train, _, _ = loaded_data
    elif loaded_data and len(loaded_data) == 6: 
        X_train_final, Y_train, X_val_for_split, Y_val_for_split, _, _ = loaded_data
    else:
        print("Loaded data is None or does not have the expected number of elements (4 or 6). Exiting.")
        return

    if X_train_final is None or Y_train is None:
        print("Failed to load training data. Exiting training.")
        return

    train_model_procedure(X_train_final, Y_train, X_val_for_split, Y_val_for_split)

if __name__ == "__main__":
    main()
