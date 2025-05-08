import pandas as pd

def load_data(engine, movie_table='movies', genre_table='genres',
              movie_genre_table='movie_genres', genre_label='genre_name'):
    """
    Load movie records and attach genre labels as a single comma-separated column.

    Args:
        engine: SQLAlchemy engine object.
        movie_table (str): Name of the movies table.
        genre_table (str): Name of the genres table.
        movie_genre_table (str): Name of the many-to-many join table.
        genre_label (str): Name of the column in the genres table containing genre names.

    Returns:
        pd.DataFrame: DataFrame combining all movie columns and a 'genre' column.
    """
    # Fetch movie table columns dynamically
    with engine.connect() as conn:
        movie_columns_df = pd.read_sql(f"PRAGMA table_info({movie_table});", conn)
        if movie_columns_df.empty:
            raise ValueError(f"Table '{movie_table}' not found or has no columns.")

        movie_column_str = ', '.join([f"m.{col}" for col in movie_columns_df['name']])

    # Construct the dynamic query
    query = f"""
    SELECT 
        {movie_column_str},
        GROUP_CONCAT(g.{genre_label}, ', ') AS genre
    FROM {movie_table} m
    LEFT JOIN {movie_genre_table} mg ON m.id = mg.movie_id
    LEFT JOIN {genre_table} g ON g.id = mg.genre_id
    GROUP BY m.id
    ORDER BY m.id;
    """

    return pd.read_sql(query, con=engine)
