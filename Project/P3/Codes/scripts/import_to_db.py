import os
import pandas as pd
from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, MetaData, Table, text, UniqueConstraint
from sqlalchemy.types import Boolean
from sqlalchemy.engine import Engine

def infer_sqlalchemy_type(pd_dtype, sample_value=None):
    """Infer SQLAlchemy column type from pandas dtype."""
    if pd_dtype == 'int64':
        return Integer
    elif pd_dtype == 'float64':
        return Float
    elif pd_dtype == 'bool':
        return Boolean
    elif pd_dtype == 'object':
        if sample_value and len(str(sample_value)) < 255:
            return String(255)
        return Text
    else:
        return Text

def import_to_database(engine: Engine, df=None, csv_path=None, table_name='movies', clear_existing=True):
    """
    Imports data into a normalized database schema (movies, genres, movie_genres).
    
    Args:
        engine (Engine): SQLAlchemy engine.
        df (pd.DataFrame): Optional DataFrame. If None, loads from csv_path.
        csv_path (str): Path to CSV file (used if df is None).
        table_name (str): Name for the main movie table.
        clear_existing (bool): If True, drops existing tables and recreates them.
    """
    if df is None:
        if csv_path is None:
            raise ValueError("Either 'df' or 'csv_path' must be provided.")
        df = pd.read_csv(csv_path)

    df['original_index'] = df.index
    df = df.rename(columns={'names': 'name'})

    metadata = MetaData()

    with engine.begin() as conn:
        if clear_existing:
            for tbl in ['movie_genres', 'genres', table_name]:
                conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))

        # Rebuild metadata after drop
        metadata = MetaData()

        # Dynamically define movie table columns
        columns = []
        if 'id' not in df.columns:
            columns.append(Column('id', Integer, primary_key=True, autoincrement=True))

        for col in df.columns:
            if col == 'genre':
                continue
            sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            col_type = infer_sqlalchemy_type(df[col].dtype, sample_value)
            columns.append(Column(col, col_type))

        movies = Table(table_name, metadata, *columns)

        genres = Table('genres', metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('genre_name', String(50), unique=True, nullable=False)
        )

        movie_genres = Table('movie_genres', metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('movie_id', Integer, ForeignKey(f'{table_name}.id')),
            Column('genre_id', Integer, ForeignKey('genres.id')),
            UniqueConstraint('movie_id', 'genre_id', name='uq_movie_genre')
        )

        metadata.create_all(engine)

        # Insert movies
        df_movies = df.drop(columns=['genre'], errors='ignore')
        df_movies.to_sql(table_name, con=conn, if_exists='append', index=False)

        movie_ids_df = pd.read_sql(f'SELECT id, original_index FROM {table_name}', con=conn)
        movie_id_map = dict(zip(movie_ids_df['original_index'], movie_ids_df['id']))

        # Handle genres
        genre_set = set()
        movie_to_genres = {}

        for idx, row in df.iterrows():
            genres_list = str(row.get('genre', '')).split(',')
            genres_list = [g.strip() for g in genres_list if g.strip()]
            movie_to_genres[idx] = genres_list
            genre_set.update(genres_list)

        for genre in genre_set:
            conn.execute(text("INSERT OR IGNORE INTO genres (genre_name) VALUES (:genre)"), {"genre": genre})

        genre_df = pd.read_sql('SELECT id, genre_name FROM genres', con=conn)
        genre_name_to_id = dict(zip(genre_df['genre_name'], genre_df['id']))

        for orig_index, genres_list in movie_to_genres.items():
            movie_db_id = movie_id_map.get(orig_index)
            if movie_db_id is None:
                print(f"Warning: Could not find database ID for original_index {orig_index}. Skipping genre linking.")
                continue
            for genre_name in genres_list:
                genre_db_id = genre_name_to_id.get(genre_name)
                if genre_db_id is None:
                    print(f"Warning: Could not find database ID for genre_name {genre_name}. Skipping.")
                    continue
                conn.execute(text("""
                    INSERT INTO movie_genres (movie_id, genre_id) 
                    VALUES (:movie_id, :genre_id)
                    ON CONFLICT(movie_id, genre_id) DO NOTHING; 
                    """), 
                    {"movie_id": movie_db_id, "genre_id": genre_db_id}
                )
        print(f"Data imported into '{table_name}', 'genres', and 'movie_genres' tables.")
