import pandas as pd
from database_connection import get_engine 

# Get the SQLAlchemy engine from database_connection.py
engine = get_engine()

queries = {
    
    # 1. Retrieve the top 10 highest-grossing movies.
    # This query selects movie names and their revenue from the 'movies' table.
    # It filters out any movies where revenue is NULL, then sorts the remaining rows
    # in descending order by revenue to find the highest-grossing ones.
    # Finally, it limits the output to the top 10 results.
    "Top 10 highest-grossing movies (by revenue)": """
        SELECT name, revenue
        FROM movies
        WHERE revenue IS NOT NULL
        ORDER BY revenue DESC
        LIMIT 10;
    """,

    # 2. Calculate average budget and average revenue per original language.
    # This query groups the data by 'orig_lang' (original language of the movie),
    # and computes three things per group:
    #   - number of movies (COUNT)
    #   - average budget (AVG of budget_x)
    #   - average revenue (AVG of revenue)
    # The HAVING clause ensures we only include languages that appear in more than 30 movies.
    # The result is sorted by average revenue in descending order.
    "Average budget and revenue per original languages that have more than 30 movies": """
        SELECT orig_lang,
               COUNT(*) AS movie_count,
               AVG(budget_x) AS avg_budget,
               AVG(revenue) AS avg_revenue
        FROM movies
        WHERE budget_x IS NOT NULL AND revenue IS NOT NULL
        GROUP BY orig_lang
        HAVING COUNT(*) > 30
        ORDER BY avg_revenue DESC;
    """,

    # 3. Identify movies where the revenue is at least 5 times the budget.
    # This query selects movies where:
    #   - budget_x is greater than 0 (to avoid division by zero),
    #   - and revenue is at least 5 times the budget.
    # It computes the revenue-to-budget ratio as a derived column and
    # sorts the results by total revenue in descending order.
    # The output is limited to the top 10 high-performing movies financially.
    "Top 10 movies with revenue >= 5x budget (sorted by revenue)": """
        SELECT id, name, budget_x, revenue, (revenue / budget_x) AS revenue_ratio
        FROM movies
        WHERE budget_x > 0
          AND revenue >= 5 * budget_x
        ORDER BY revenue DESC
        LIMIT 10;
    """,

    # 4. Find genres that have an average movie score above 65.
    # This query uses JOINs to link the 'genres' table with the 'movies' table
    # via the intermediate 'movie_genres' many-to-many mapping table.
    # It calculates the average score for all movies in each genre,
    # and filters out genres where the average score is 65 or below.
    # Results are sorted by average score in descending order.
    "Genres with average score > 65": """
        SELECT g.genre_name, AVG(m.score) AS avg_score
        FROM genres g
        JOIN movie_genres mg ON g.id = mg.genre_id
        JOIN movies m ON m.id = mg.movie_id
        WHERE m.score IS NOT NULL
        GROUP BY g.genre_name
        HAVING avg_score > 65
        ORDER BY avg_score DESC;
    """,

    # 5. Determine the 5 most frequent genre combinations among all movies.
    # The inner query collects the genres associated with each movie by JOINing all three tables,
    # and then uses GROUP_CONCAT to merge them into a single comma-separated string per movie.
    # The outer query then counts how many times each unique genre combination appears.
    # Finally, it returns the top 5 most frequent combinations sorted by their count.
    "Top 5 most common genre combinations": """
        SELECT genre_combination, COUNT(*) AS count
        FROM (
            SELECT m.id AS movie_id,
                   GROUP_CONCAT(g.genre_name, ', ') AS genre_combination
            FROM movies m
            JOIN movie_genres mg ON m.id = mg.movie_id
            JOIN genres g ON g.id = mg.genre_id
            GROUP BY m.id
        ) AS combinations
        GROUP BY genre_combination
        ORDER BY count DESC
        LIMIT 5;
    """
}

# Execute and display each query
for title, query in queries.items():
    print(f"\n\n--- {title} ---\n")
    df = pd.read_sql(query, con=engine)
    print(df)
