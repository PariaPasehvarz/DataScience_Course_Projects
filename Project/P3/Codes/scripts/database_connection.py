from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

def get_engine(db_path: str):
    """
    Return a new SQLAlchemy engine for the specified database path.

    Args:
        db_path (str): The path to the SQLite database file.
    """
    if not db_path:
        raise ValueError("A database path must be provided to get_engine.")
    
    # Ensure the path is absolute for SQLite connection
    abs_db_path = os.path.abspath(db_path)
    database_url = f"sqlite:///{abs_db_path}"
    return create_engine(database_url, echo=False)

def get_session(engine_instance):
    """
    Return a new SQLAlchemy session from the given engine instance.

    Args:
        engine_instance: An SQLAlchemy engine instance.
    """
    if engine_instance is None:
        raise ValueError("An engine instance must be provided to get_session.")
    
    LocalSession = sessionmaker(bind=engine_instance)
    session = LocalSession()
    return session
