# IMDB Movie Data Science Pipeline

This repository contains a comprehensive data science pipeline for processing and analyzing IMDB movie data. The pipeline includes data loading, preprocessing, feature engineering, and database management.

## Overview

This project implements an end-to-end data processing pipeline that:
1. Initializes a SQLite database with IMDB movie data
2. Preprocesses the data to clean and filter records
3. Performs feature engineering to create meaningful variables for analysis
4. Stores the transformed data back to the database

## Detailed Pipeline Workflow

The pipeline follows these detailed steps:

### 1. Database Initialization (`init_db.py`)
- Creates a SQLite database in the `database/IMDB_movies.db` location
- Establishes connection using SQLAlchemy
- Imports the raw IMDB movie data from `data/imdb_movies.csv`
- Creates necessary tables and schema

### 2. Data Loading (`load_data.py`)
- Connects to the SQLite database using the engine
- Executes SQL queries to retrieve movie data
- Loads the data into pandas DataFrames for processing
- Returns a structured DataFrame with all movie attributes

### 3. Data Preprocessing (`preprocess.py`)
- Handles missing values by replacing "nan" strings with proper NaN values
- Indexes movies with a consistent numbering system (original_index + 1)
- Removes leading/trailing whitespace from string columns
- Identifies and removes problematic rows with null values or zeros
- Filters movies:
  - Keeps only movies with revenue > 10,000
  - Keeps only movies with budget > 10,000
  - Retains only "Released" status movies
- Removes duplicate movies by name (keeping the first occurrence)
- Imports the cleaned dataset back to the database

### 4. Feature Engineering (`feature_engineering.py`)
- Extracts temporal features:
  - Converts date_x to datetime format
  - Extracts year from the date
  - Filters movies to those released between 1972-2022
- Creates derived metrics:
  - Calculates revenue-to-budget ratio
- Removes unnecessary columns (orig_title, crew, status, date_x, id)
- Performs NLP processing:
  - Uses spaCy's 'en_core_web_md' model
  - Converts movie overviews to 300-dimensional word vectors
- Scales numeric features using MinMaxScaler:
  - Normalizes score, budget, year, and revenue_to_budget
- Encodes categorical variables:
  - One-hot encodes language and country columns
  - Uses MultiLabelBinarizer for genre lists
- Reduces text vector dimensionality:
  - Applies PCA to reduce 300D vectors to 75D
  - Uses whitening for better feature distribution
- Assembles the final feature table:
  - Combines processed numeric features
  - Adds categorical one-hot encoded features
  - Integrates genre encodings
  - Incorporates reduced text PCA features
- Saves the fully processed dataset:
  - Exports to CSV as `data/cleaned_imdb_movies_wide.csv`
  - Imports the feature-engineered dataset back to the database

### 5. Database Operations (`import_to_db.py`, `run_queries.py`)
- `import_to_db.py`:
  - Handles the creation and updating of database tables
  - Maps DataFrame columns to SQL data types
  - Manages efficient data insertion
  - Handles both CSV file and DataFrame imports
- `run_queries.py`:
  - Provides utilities for executing SQL queries against the database
  - Supports custom query execution
  - Enables data retrieval for analysis

### 6. Main Pipeline Execution (`pipeline.py`)
- Orchestrates the entire pipeline flow
- Executes the components in sequence:
  1. Runs database initialization
  2. Performs data preprocessing
  3. Conducts feature engineering
- Manages dependencies between pipeline steps
- Ensures proper execution order and data flow

## Data Transformations

### Raw Data (`imdb_movies.csv`)
- Contains basic movie information:
  - Title, original title
  - Release date
  - Budget and revenue figures
  - Overview/description
  - Genre, language, country
  - Crew information
  - Movie status

### Processed Data (`cleaned_imdb_movies_wide.csv`)
A wide-format dataset with:
- Original movie metadata (cleaned)
- Derived features:
  - Year
  - Revenue-to-budget ratio
- Encoded categorical features:
  - One-hot encoded language (prefix: 'lang_')
  - One-hot encoded country (prefix: 'ctr_')
  - Multi-label encoded genres
- NLP features:
  - 75 PCA components from movie overview text (prefix: 'text_pca_')

## Data

The project uses IMDB movie data with the following files:
- `data/imdb_movies.csv` - Raw IMDB movie data
- `data/cleaned_imdb_movies_wide.csv` - Processed and feature-engineered dataset

## Project Structure

```
.
├── data/                   # Data files
│   ├── imdb_movies.csv     # Raw IMDB movie data
│   └── cleaned_imdb_movies_wide.csv # Processed data
├── database/               # SQLite database files
├── Doc/                    # Documentation files
├── scripts/                # Data processing scripts
│   ├── database_connection.py # Database connection utilities
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── import_to_db.py     # Database import functions
│   ├── init_db.py          # Database initialization
│   ├── load_data.py        # Data loading utilities
│   ├── preprocess.py       # Data preprocessing
│   └── run_queries.py      # Database query utilities
├── Dockerfile              # Docker configuration
├── ds_env.yml              # Conda environment specification
├── job.yaml                # Kubernetes job configuration
├── pipeline.py             # Main pipeline script
└── requirements.txt        # Python dependencies
```

## Features

1. **Data Preprocessing**:
   - Handling missing values
   - Removing duplicates
   - Filtering out invalid or irrelevant records
   - Text cleaning and normalization

2. **Feature Engineering**:
   - Temporal features (year extraction)
   - Ratio calculations (revenue to budget)
   - Text vectorization using spaCy
   - Dimensionality reduction with PCA
   - One-hot encoding for categorical variables
   - Multi-label encoding for genres

3. **Database Management**:
   - SQLite database integration
   - Data import and export utilities
   - Query capabilities

## Technology Stack

- **Programming Language**: Python 3.10
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Natural Language Processing**: spaCy
- **Database**: SQLite, SQLAlchemy
- **Containerization**: Docker
- **Orchestration**: Kubernetes

## Setup and Installation

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t ds-pipeline-with-spacy .
   ```

2. Run the container:
   ```bash
   docker run ds-pipeline-with-spacy
   ```

### Local Installation

1. Create a conda environment (recommended):
   ```bash
   conda env create -f ds_env.yml
   conda activate ds-env
   ```

   OR install using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_md
   ```

3. Run the pipeline:
   ```bash
   python pipeline.py
   ```

## Running in Kubernetes

The project includes a Kubernetes job configuration in `job.yaml`:

```bash
kubectl apply -f job.yaml
```

## Pipeline Components

1. **Database Initialization** (`init_db.py`):
   - Creates SQLite database
   - Imports raw IMDB movie data

2. **Preprocessing** (`preprocess.py`):
   - Cleans the data
   - Removes duplicates and invalid entries
   - Standardizes text fields

3. **Feature Engineering** (`feature_engineering.py`):
   - Extracts year from date
   - Calculates revenue-to-budget ratio
   - Generates text vectors using spaCy
   - Performs dimensionality reduction
   - Creates one-hot encodings for categorical variables

## License

This project is licensed under the terms included in the LICENSE file.

## Requirements

See `requirements.txt` for a complete list of dependencies. 