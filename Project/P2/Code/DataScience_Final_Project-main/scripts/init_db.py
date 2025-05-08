from database_connection import get_engine
from import_to_db import import_to_database
import os

engine = get_engine()

csv_path = os.path.join(os.path.dirname(__file__), '../data/imdb_movies.csv')

import_to_database(engine=engine, csv_path=csv_path)

print("Database initiated successfully.")