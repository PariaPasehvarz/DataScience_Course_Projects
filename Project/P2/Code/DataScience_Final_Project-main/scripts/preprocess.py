import pandas as pd
import numpy as np
from database_connection import get_engine
from import_to_db import import_to_database
from load_data import load_data
import os

engine = get_engine()


df = load_data(engine)

df = df.replace("nan", np.nan)
df["original_index"] += 1
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

mask = ((df.isnull()) | (df == 0))
rows_with_issues = df[mask.any(axis=1)]
df.drop(index=rows_with_issues.index)
df = df[(df["revenue"] > 10000) & (df["budget_x"] > 10000)]
df = df[df["status"] == "Released"]
df = df.drop_duplicates(subset='name', keep='first')

import_to_database(df=df,engine=engine)

print("Data preprocessed successfully.")