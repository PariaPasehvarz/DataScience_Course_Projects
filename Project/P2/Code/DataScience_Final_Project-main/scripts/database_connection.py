from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Get the absolute path to the database file from the current script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../database/IMDB_movies.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)

# Optional session factory for transactional work
Session = sessionmaker(bind=engine)

def get_engine():
    """Return the SQLAlchemy engine for executing queries."""
    return engine

def get_session():
    """Return a new SQLAlchemy session."""
    return Session()
