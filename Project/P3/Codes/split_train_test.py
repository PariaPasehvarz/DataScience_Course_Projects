import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the original CSV file
data_path = 'data/imdb_movies.csv'

# Load the data
df = pd.read_csv(data_path)

# Split the data (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the train and test sets to new CSV files
train_df.to_csv('data/imdb_movies_train.csv', index=False)
test_df.to_csv('data/imdb_movies_test.csv', index=False)

print('Train and test CSV files have been created.')
