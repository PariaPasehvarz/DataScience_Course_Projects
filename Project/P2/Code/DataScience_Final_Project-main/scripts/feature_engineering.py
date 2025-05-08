import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from database_connection import get_engine
from import_to_db import import_to_database
from load_data import load_data
import spacy

# 1. Load data
engine = get_engine()
df = load_data(engine)

# 2. Year, ratio, basic drops
df["year"] = pd.to_datetime(
    df["date_x"], errors="coerce", dayfirst=True, format="mixed"
).dt.year
df = df[df["year"].between(1972, 2022)]

df["revenue_to_budget"] = df["revenue"] / df["budget_x"]

df.drop(columns=["orig_title", "crew", "status", "date_x", "id"], inplace=True)

# 3. spaCy vectors
nlp = spacy.load("en_core_web_md")
df["overview_vector"] = df["overview"].fillna("").map(lambda t: nlp(t).vector)

# 4. Scale numeric columnss
numeric_cols = ["score", "budget_x", "year", "revenue_to_budget"]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 5. Categorical encoding
cat_onehot = pd.get_dummies(df[["orig_lang", "country"]],
                            prefix=["lang", "ctr"],
                            dtype="int8")

mlb = MultiLabelBinarizer()
genre_lists = (
    df["genre"]
    .fillna("")
    .str.split(",")
    .apply(lambda lst: [g.strip().title() for g in lst if g.strip()])
)
genre_ohe = pd.DataFrame(
    mlb.fit_transform(genre_lists),
    columns=[g.replace(" ", "_") for g in mlb.classes_],
    index=df.index,
    dtype="int8"
)

# 6. PCA on text (300 → 75)
text_matrix = np.vstack(df["overview_vector"].values)
pca = PCA(n_components=75, whiten=True, random_state=42)
text_pca_df = pd.DataFrame(
    pca.fit_transform(text_matrix),
    columns=[f"text_pca_{i+1}" for i in range(75)],
    index=df.index
)

# 7. Assemble final table (do NOT drop name or orig_index)
drop_cols_final = ["overview", "revenue", "genre",
                   "orig_lang", "country", "overview_vector"]

df_final = pd.concat(
    [
        df.drop(columns=drop_cols_final),
        cat_onehot,
        genre_ohe,
        text_pca_df
    ],
    axis=1
)

# 8. Save
csv_path = os.path.join(
    os.path.dirname(__file__), "../data/cleaned_imdb_movies_wide.csv"
)
df_final.to_csv(csv_path, index=False)

import_to_database(df=df_final, engine=engine)

print("Feature engineering finished.")
