from database_connection import get_engine
from import_to_db import import_to_database
import os

# Paths for train and test CSVs and databases
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, '../data')
db_dir = os.path.join(base_dir, '../database')

train_csv_path = os.path.join(data_dir, 'imdb_movies_train.csv')
test_csv_path = os.path.join(data_dir, 'imdb_movies_test.csv')
train_db_path = os.path.join(db_dir, 'IMDB_movies_train.db')
test_db_path = os.path.join(db_dir, 'IMDB_movies_test.db')

# Initialize train database
train_engine = get_engine(db_path=train_db_path)
import_to_database(engine=train_engine, csv_path=train_csv_path, clear_existing=True)
print('Train database initiated successfully.')

# Initialize test database
test_engine = get_engine(db_path=test_db_path)
import_to_database(engine=test_engine, csv_path=test_csv_path, clear_existing=True)
print('Test database initiated successfully.')