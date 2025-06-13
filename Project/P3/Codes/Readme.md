# DataScience_Final_Project

This project is a data science pipeline for analyzing and modeling IMDB movie data. It includes data preprocessing, feature engineering, database management, model training, evaluation, and experiment tracking using MLflow.

## Pipeline Workflow

The pipeline follows a **train-then-test** approach:

### Training Phase:
1. **Database Initialization** - Import raw data into separate train/test databases
2. **Data Preprocessing (Train)** - Clean and preprocess training data
3. **Feature Engineering (Train)** - Generate features from training data
4. **Save Processed Data (Train)** - Store processed training features
5. **Model Training** - Train neural network on training data

### Testing Phase:
6. **Data Preprocessing (Test)** - Clean and preprocess test data
7. **Feature Engineering (Test)** - Generate features from test data using trained transformers
8. **Save Processed Data (Test)** - Store processed test features
9. **Model Evaluation** - Evaluate trained model on test data
10. **MLflow Logging** - Log model and metrics to MLflow

## Project Structure

- `pipeline.py`: Main pipeline script with train/test workflow support.
- `requirements.txt`: Python dependencies for the project.
- `data/`: Contains raw and processed data files.
  - `imdb_movies_train.csv`: Training dataset.
  - `imdb_movies_test.csv`: Test dataset.
  - `imdb_movies.csv`: Complete dataset.
  - `processed_features/`: Directory for processed feature files.
  - `transformers/`: Directory for trained transformers (scalers, encoders).
- `database/`: SQLite databases for storing movie data.
  - `IMDB_movies_train.db`: Training data database.
  - `IMDB_movies_test.db`: Test data database.
- `mlruns/`: MLflow experiment tracking directory.
- `models/`: Saved machine learning models and metrics.
- `plots/`: Generated plots and visualizations.
- `scripts/`: Utility scripts for each step of the pipeline, including:
  - `database_connection.py`: Handles database connections.
  - `evaluate_model.py`: Evaluates trained models on test data.
  - `feature_engineering.py`: Feature engineering steps (supports --mode train/test).
  - `import_to_db.py`: Imports data into the database.
  - `init_db.py`: Initializes the database schema and imports raw data.
  - `load_data.py`: Loads raw data from database.
  - `load_processed_from_db.py`: Loads processed data from the database.
  - `log_to_mlflow.py`: Logs experiments to MLflow.
  - `preprocess.py`: Data preprocessing steps (supports --mode train/test).
  - `save_processed_to_db.py`: Saves processed data to the database (supports --mode train/test).
  - `train_model.py`: Trains the neural network model.

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline**
   
   **Full Pipeline (Train + Test):**
   ```bash
   python pipeline.py
   # or explicitly
   python pipeline.py --workflow full
   ```
   
   **Training Only:**
   ```bash
   python pipeline.py --workflow full_train
   ```
   
   **Testing Only (requires trained model):**
   ```bash
   python pipeline.py --workflow full_test
   ```
   
   **Individual Steps:**
   ```bash
   # Run specific step
   python pipeline.py --step train
   
   # Start from specific step
   python pipeline.py --start evaluate
   
   # List all available steps
   python pipeline.py --list
   ```

## Pipeline Commands

| Command | Description |
|---------|-------------|
| `python pipeline.py` | Run complete train + test pipeline |
| `python pipeline.py --workflow full_train` | Run training workflow only |
| `python pipeline.py --workflow full_test` | Run testing workflow only |
| `python pipeline.py --step <step_name>` | Run single step |
| `python pipeline.py --start <step_name>` | Start from specific step |
| `python pipeline.py --list` | Show all available steps |

## Features
- **Train/Test Workflow**: Proper separation of training and testing phases
- **Data loading and preprocessing**: Separate processing for train and test datasets
- **Feature engineering**: Consistent feature transformation using fitted transformers
- **Database storage and querying**: SQLite databases for train and test data
- **Model training and evaluation**: Neural network training on training data, evaluation on test data
- **Experiment tracking**: Comprehensive logging with MLflow
- **Visualization**: Training history and performance plots with matplotlib
- **Pipeline automation**: Flexible pipeline execution with workflow options

## Model Architecture
- **Type**: Sequential Deep Neural Network
- **Framework**: TensorFlow/Keras
- **Features**: Text embeddings (SentenceTransformers) + structured features
- **Output**: Multi-label genre classification
- **Evaluation**: F1-score, Hamming loss, and accuracy metrics

## Requirements
See `requirements.txt` for the full list of dependencies.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
