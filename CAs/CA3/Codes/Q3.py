# Final Version

# ------------------- INSTALL REQUIRED LIBRARIES -------------------
# If you're running this in a fresh environment (like Colab), uncomment:

# import os

# if not os.path.exists("data_flag"):
#   !gdown 1CyR9bMjbgOGDSRIgcZHrB5oWN4rJmyvb
#   !gdown 1fWw22JQbvOxkYdQ3nL5ruP7gNzXzLzFd
#   !gdown 1GS5Bf81MDbI_yAZMrhZs2MJDT6aC8jiR

#   !touch data_flag

# if not os.path.exists("lib_flag"):
#   !pip uninstall -y numpy
#   !pip install "numpy<2.0"
#   !pip install scikit-surprise

#   !touch lib_flag

#   import os

#   # Force restart
#   import os, signal; os.kill(os.getpid(), signal.SIGKILL)

# ------------------- IMPORT LIBRARIES -------------------
import pandas as pd                  # Data manipulation
import numpy as np                   # Numerical operations
import random                        # For seeding random operations
import os                            # For environment variable setting
from pathlib import Path             # For handling file paths

from surprise import Dataset, Reader, SVDpp  # For building recommender systems

from surprise import accuracy                                 # For RMSE, MAE
from surprise.model_selection import train_test_split         # For validation split

# ------------------- CONFIGURATION -------------------
SEED     = 42  # Seed for reproducibility
DATA_DIR = Path("/content")          # Directory for input CSVs
OUT_DIR  = Path("/content/outputs")  # Output directory for final submission
OUT_DIR.mkdir(exist_ok=True)         # Ensure output dir exists

# ------------------- SET RANDOM SEEDS -------------------
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ------------------- 1) LOAD DATA -------------------
ratings = pd.read_csv(DATA_DIR / "train_data_movie_rate.csv")  # Training ratings data
test_df = pd.read_csv(DATA_DIR / "test_data.csv")              # Test dataset

# ------------------- 2) CLEAN AND INDEX DATA -------------------
ratings = ratings.drop_duplicates(subset=["user_id", "item_id"], keep="last").copy()  # Remove duplicate ratings
ratings["label"] = pd.to_numeric(ratings["label"], errors="coerce").clip(1, 5)  # Ensure ratings are numeric and in valid range
ratings = ratings.dropna(subset=["user_id", "item_id", "label"]).reset_index(drop=True)  # Drop missing data

user2idx = {u: i for i, u in enumerate(sorted(ratings.user_id.unique()))}  # Map users to indices
item2idx = {m: i for i, m in enumerate(sorted(ratings.item_id.unique()))}  # Map items to indices

ratings["uid"] = ratings.user_id.map(user2idx)
ratings["iid"] = ratings.item_id.map(item2idx)
test_df["uid"] = test_df.user_id.map(user2idx)
test_df["iid"] = test_df.item_id.map(item2idx)

# ------------------- 3) TRAIN SVD++ MODEL -------------------
reader = Reader(rating_scale=(1, 5))
full_data = Dataset.load_from_df(ratings[["uid", "iid", "label"]], reader)

trainset, valset = train_test_split(full_data, test_size=0.1, random_state=SEED)  # Split data into train/val

# Initialize SVD++ model
svdpp = SVDpp(
    n_factors=164,        # Latent factors
    n_epochs=100,         # Training epochs
    lr_all=0.001,         # Learning rate
    reg_all=0.07,         # Regularization
    random_state=SEED     # Reproducibility
)
svdpp.fit(trainset)

# ------------------- EVALUATE MODEL ON VALIDATION SET -------------------
val_preds = svdpp.test(valset)  # Predict on validation set
rmse = accuracy.rmse(val_preds, verbose=True)  # RMSE
mae  = accuracy.mae(val_preds, verbose=True)   # MAE

# ------------------- 4) PREDICT ON TEST DATA -------------------
preds = [svdpp.predict(int(u), int(i)).est for u, i in zip(test_df.uid, test_df.iid)]  # Predict ratings on test set

preds = np.clip(preds, 1, 5)  # Ensure predictions are within valid range (1-5)

# ------------------- 5) SAVE SUBMISSION -------------------
submission = pd.DataFrame({
    "id": test_df.get("id", pd.Series(np.arange(len(test_df)))),  # Use test IDs or default to index
    "label": np.round(preds, 3)  # Round predictions to 3 decimal places
})

fname = OUT_DIR / "submission.csv"
submission.to_csv(fname, index=False)

print("Saved →", fname)
submission.head()

# ------------------- Other Versions -------------------

# Version 1

# import os

# if not os.path.exists("data_flag"):
#     !gdown 1CyR9bMjbgOGDSRIgcZHrB5oWN4rJmyvb
#     !gdown 1fWw22JQbvOxkYdQ3nL5ruP7gNzXzLzFd
#     !gdown 1GS5Bf81MDbI_yAZMrhZs2MJDT6aC8jiR
#     !touch data_flag

# if not os.path.exists("lib_flag"):
#     !pip uninstall -y numpy
#     !pip install "numpy<2.0"
#     !pip install scikit-surprise
#     !pip install implicit
#     !pip install node2vec
#     !touch lib_flag
#     import os
#     import os, signal; os.kill(os.getpid(), signal.SIGKILL)

# import numpy as np, pandas as pd, networkx as nx
# from tqdm.notebook import tqdm
# from pathlib import Path
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.linear_model import Ridge
# import lightgbm as lgb
# from surprise import Dataset, Reader, SVD, SVDpp
# import implicit
# from node2vec import Node2Vec
# import warnings, random, pickle, json, os, sys, shutil, tempfile
# warnings.filterwarnings("ignore")

# SEED = 42
# random.seed(SEED); np.random.seed(SEED)

# DATA_DIR = Path("/content")
# OUT_DIR  = Path("/content/outputs")
# OUT_DIR.mkdir(exist_ok=True); (OUT_DIR/"cache").mkdir(exist_ok=True)

# ratings = pd.read_csv(DATA_DIR/"train_data_movie_rate.csv")
# trust   = pd.read_csv(DATA_DIR/"train_data_movie_trust.csv")
# test_df = pd.read_csv(DATA_DIR/"test_data.csv")

# print(ratings.head(), "\n\n", trust.head(), "\n\n", test_df.head())
# print(f"#ratings = {len(ratings):,}  |  #trust edges = {len(trust):,}")

# def clean_frames(ratings, trust):
#     ratings = ratings.drop_duplicates(subset=["user_id", "item_id"], keep="last").copy()
#     trust   = trust.drop_duplicates(subset=["user_id_trustor", "user_id_trustee"], keep="last").copy()
#     ratings["label"]       = pd.to_numeric(ratings["label"], errors="coerce").clip(1, 5)
#     trust["trust_value"]   = pd.to_numeric(trust["trust_value"], errors="coerce").fillna(0)
#     trust["trust_value"]   = trust["trust_value"].clip(0, 5)
#     ratings = ratings.dropna(subset=["user_id", "item_id", "label"])
#     trust   = trust.dropna(subset=["user_id_trustor", "user_id_trustee", "trust_value"])
#     ratings[["user_id","item_id"]] = ratings[["user_id","item_id"]].astype("int32")
#     trust[["user_id_trustor","user_id_trustee"]] = trust[["user_id_trustor","user_id_trustee"]].astype("int32")
#     ratings["label"]     = ratings["label"].astype("float32")
#     trust["trust_value"] = trust["trust_value"].astype("float32")
#     return ratings.reset_index(drop=True), trust.reset_index(drop=True)

# ratings, trust = clean_frames(ratings, trust)
# print(f"Ratings cleaned → {len(ratings):,} rows  |  Trust cleaned → {len(trust):,} rows")

# (OUT_DIR/"cache").mkdir(exist_ok=True)
# ratings.to_parquet(OUT_DIR/"cache/ratings_clean.parquet")
# trust.to_parquet(OUT_DIR/"cache/trust_clean.parquet")
# print("✔️  Cached cleaned datasets")

# user_ids = pd.concat([ratings.user_id, trust.user_id_trustor, trust.user_id_trustee]).unique()
# item_ids = ratings.item_id.unique()
# user2idx = {u:i for i,u in enumerate(sorted(user_ids))}
# item2idx = {m:i for i,m in enumerate(sorted(item_ids))}
# ratings["uid"] = ratings.user_id.map(user2idx)
# ratings["iid"] = ratings.item_id.map(item2idx)
# trust ["uid_src"] = trust.user_id_trustor.map(user2idx)
# trust ["uid_dst"] = trust.user_id_trustee.map(user2idx)
# test_df["uid"]   = test_df.user_id.map(user2idx)
# test_df["iid"]   = test_df.item_id.map(item2idx)

# val_idx = ratings.groupby("uid").sample(1, random_state=SEED).index
# val     = ratings.loc[val_idx].reset_index(drop=True)
# train   = ratings.drop(val_idx).reset_index(drop=True)
# print(f"train:{len(train):,}  val:{len(val):,}")

# global_mean = train.label.mean()
# item_mean   = train.groupby("iid").label.mean().to_dict()
# user_mean   = train.groupby("uid").label.mean().to_dict()

# def predict_baseline(df):
#     return np.column_stack([
#         np.full(len(df), global_mean),
#         df.iid.map(item_mean).fillna(global_mean),
#         df.uid.map(user_mean).fillna(global_mean)
#     ])

# val_base = predict_baseline(val)
# for i, name in enumerate(["Global","ItemMean","UserMean"]):
#     print(f"{name:10s} RMSE:",
#           np.sqrt(mean_squared_error(val.label, val_base[:, i])))

# def surprise_fit(algo, df):
#     reader = Reader(rating_scale=(1,5))
#     data   = Dataset.load_from_df(df[["uid","iid","label"]], reader)
#     algo.fit(data.build_full_trainset())
#     return algo

# def surprise_pred(algo, df):
#     return np.array([algo.predict(int(u), int(i)).est for u,i in zip(df.uid, df.iid)])

# svd  = surprise_fit(SVD  (n_factors=128, n_epochs=30, reg_all=0.02, random_state=SEED), train)
# svdpp= surprise_fit(SVDpp(n_factors= 64, n_epochs=20, reg_all=0.02, random_state=SEED), train)

# val_svd   = surprise_pred(svd, val)
# val_svdpp = surprise_pred(svdpp, val)

# print("SVD    RMSE:", np.sqrt(mean_squared_error(val.label, val_svd)))
# print("SVD++  RMSE:", np.sqrt(mean_squared_error(val.label, val_svdpp)))

# from scipy.sparse import coo_matrix
# conf_mat = coo_matrix((train.label, (train.iid, train.uid)))
# als = implicit.als.AlternatingLeastSquares(factors=128, regularization=0.01,
#                                            iterations=20, random_state=SEED)
# als.fit(conf_mat)

# def als_predict(df):
#     user_factors = als.user_factors
#     item_factors = als.item_factors
#     raw = np.array([
#         np.dot(user_factors[int(u)], item_factors[int(i)])
#         if int(u) < user_factors.shape[0] and int(i) < item_factors.shape[0]
#         else global_mean
#         for u, i in zip(df.uid, df.iid)
#     ])
#     mn, mx = raw.min(), raw.max()
#     return 1 + 4 * (raw - mn) / (mx - mn) if mx > mn else np.full_like(raw, global_mean)

# val_als = als_predict(val)
# print("ALS    RMSE:", np.sqrt(mean_squared_error(val.label, val_als)))

# n_items   = ratings.iid.max()+1
# fake_rows = pd.DataFrame({
#     "uid":   trust.uid_src,
#     "iid":   trust.uid_dst + n_items,
#     "label": trust.trust_value.clip(1,5)
# })
# aug_train = pd.concat([train[["uid","iid","label"]], fake_rows], ignore_index=True)
# trust_mf = surprise_fit(SVD(n_factors=128, n_epochs=25, reg_all=0.02, random_state=SEED),
#                         aug_train)
# val_trust = surprise_pred(trust_mf, val)
# print("TrustMF RMSE:", np.sqrt(mean_squared_error(val.label, val_trust)))

# g = nx.from_pandas_edgelist(trust, "uid_src", "uid_dst", create_using=nx.DiGraph())
# n2v = Node2Vec(g, dimensions=128, walk_length=20, num_walks=10,
#                workers=os.cpu_count(), seed=SEED).fit(window=10, min_count=1)
# u_emb = {int(n): n2v.wv[n] for n in n2v.wv.key_to_index}

# def build_feats(df):
#     feats = []
#     for _,r in df.iterrows():
#         uid, iid = int(r.uid), int(r.iid)
#         emb = u_emb.get(uid, np.zeros(128))
#         feats.append(np.hstack([emb,
#                                 user_mean.get(uid, global_mean),
#                                 item_mean.get(iid, global_mean),
#                                 train.groupby('uid').size().get(uid,0),
#                                 train.groupby('iid').size().get(iid,0)]))
#     return np.vstack(feats).astype(np.float32)

# X_tr, X_val = build_feats(train), build_feats(val)

# lgbm = lgb.LGBMRegressor(n_estimators=700, num_leaves=64,
#                          learning_rate=0.05, subsample=0.8,
#                          random_state=SEED)
# from lightgbm import early_stopping, log_evaluation

# lgbm.fit(
#     X_tr, train.label,
#     eval_set=[(X_val, val.label)],
#     eval_metric="rmse",
#     callbacks=[
#         early_stopping(stopping_rounds=50),
#         log_evaluation(period=0)
#     ]
# )

# val_lgb = lgbm.predict(X_val)
# print("LightGBM RMSE:", np.sqrt(mean_squared_error(val.label, val_lgb)))

# val_stack = np.column_stack([val_base[:,0],
#                              val_base[:,1],
#                              val_base[:,2],
#                              val_svd, val_svdpp, val_als, val_trust, val_lgb])

# blender = Ridge(alpha=0.1, random_state=SEED).fit(val_stack, val.label)
# val_blend = blender.predict(val_stack)
# print("BLENDED  RMSE:", np.sqrt(mean_squared_error(val.label, val_blend)))

# full_svd   = surprise_fit(SVD(n_factors=128, n_epochs=30, reg_all=0.02, random_state=SEED), ratings)
# full_svdpp = surprise_fit(SVDpp(n_factors=64, n_epochs=20, reg_all=0.02, random_state=SEED), ratings)
# fake_rows_full = fake_rows
# aug_full = pd.concat([ratings[["uid","iid","label"]], fake_rows_full])
# full_trustmf = surprise_fit(SVD(n_factors=128, n_epochs=25, reg_all=0.02, random_state=SEED), aug_full)

# conf_full = coo_matrix((ratings.label, (ratings.iid, ratings.uid)))
# full_als   = implicit.als.AlternatingLeastSquares(factors=128, regularization=0.01,
#                                                   iterations=20, random_state=SEED)
# full_als.fit(conf_full)

# X_test = build_feats(test_df)
# pred_matrix = np.column_stack([
#     np.full(len(test_df), global_mean),
#     test_df.iid.map(item_mean).fillna(global_mean),
#     test_df.uid.map(user_mean).fillna(global_mean),
#     surprise_pred(full_svd,   test_df),
#     surprise_pred(full_svdpp, test_df),
#     als_predict(test_df),
#     surprise_pred(full_trustmf, test_df),
#     lgbm.predict(X_test)
# ])

# test_preds = np.clip(blender.predict(pred_matrix), 1, 5)

# submission = pd.DataFrame({
#     "id":   test_df["id"] if "id" in test_df.columns else np.arange(len(test_df)),
#     "label": np.round(test_preds, 3)
# })

# submission.to_csv(OUT_DIR/"submission.csv", index=False)

# Version 2

# -*- coding: utf-8 -*-
"""Final_DS_CA3_Part3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iIUbt4WqE92BSw8kARryqAiqkRzU6j-f
"""

# import os

# if not os.path.exists("data_flag"):
#   !gdown 1CyR9bMjbgOGDSRIgcZHrB5oWN4rJmyvb
#   !gdown 1fWw22JQbvOxkYdQ3nL5ruP7gNzXzLzFd
#   !gdown 1GS5Bf81MDbI_yAZMrhZs2MJDT6aC8jiR

#   !touch data_flag

# if not os.path.exists("lib_flag"):
#   !pip uninstall -y numpy
#   !pip install "numpy<2.0"
#   !pip install scikit-surprise
#   !pip install implicit
#   !pip install node2vec


#   !touch lib_flag

#   import os


#   # Force restart
#   import os, signal; os.kill(os.getpid(), signal.SIGKILL)

# # Limit the number of threads used by NumPy, OpenBLAS, MKL, etc.
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# # ============================================================
# # 1 ▸ Imports, seeds, paths
# # ============================================================
# import os, random, warnings, pickle
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import networkx as nx
# from tqdm.notebook import tqdm

# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import Ridge
# import lightgbm as lgb
# from node2vec import Node2Vec
# from surprise import Dataset, Reader, SVD, SVDpp

# warnings.filterwarnings("ignore")

# SEED = 42
# random.seed(SEED);  np.random.seed(SEED)
# os.environ["PYTHONHASHSEED"] = str(SEED)

# DATA_DIR = Path("/content")
# OUT_DIR  = Path("/content/outputs")
# OUT_DIR.mkdir(exist_ok=True);  (OUT_DIR/"cache").mkdir(exist_ok=True)

# # ============================================================
# # 2 ▸ Load raw CSVs
# # ============================================================
# ratings = pd.read_csv(DATA_DIR/"train_data_movie_rate.csv")
# trust   = pd.read_csv(DATA_DIR/"train_data_movie_trust.csv")
# test_df = pd.read_csv(DATA_DIR/"test_data.csv")

# test_df.head()

# nig = test_df[["user_id","item_id"]].drop_duplicates(
#     subset=["user_id","item_id"], keep="last").copy()

# print(len(test_df),len(nig))

# test_df[["id","user_id"]].nunique()

# # ============================================================
# # 3 ▸ Clean obvious issues (duplicates, dtypes, clipping)
# # ============================================================
# def clean_frames(ratings, trust):
#     ratings = ratings.drop_duplicates(
#         subset=["user_id", "item_id"], keep="last").copy()
#     trust   = trust.drop_duplicates(
#         subset=["user_id_trustor", "user_id_trustee"], keep="last").copy()

#     ratings["label"]     = pd.to_numeric(ratings["label"], errors="coerce").clip(1, 5)
#     trust["trust_value"] = pd.to_numeric(trust["trust_value"], errors="coerce").fillna(0).clip(0, 5)

#     ratings = ratings.dropna(subset=["user_id", "item_id", "label"])
#     trust   = trust.dropna(subset=["user_id_trustor", "user_id_trustee", "trust_value"])

#     ratings[["user_id","item_id"]] = ratings[["user_id","item_id"]].astype("int32")
#     trust[["user_id_trustor","user_id_trustee"]] = trust[["user_id_trustor","user_id_trustee"]].astype("int32")
#     ratings["label"]     = ratings["label"].astype("float32")
#     trust["trust_value"] = trust["trust_value"].astype("float32")
#     return ratings.reset_index(drop=True), trust.reset_index(drop=True)

# ratings, trust = clean_frames(ratings, trust)

# # ============================================================
# # 4 ▸ Integer-index users & items
# # ============================================================
# user_ids = pd.concat([ratings.user_id, trust.user_id_trustor, trust.user_id_trustee]).unique()
# item_ids = ratings.item_id.unique()

# user2idx = {u:i for i,u in enumerate(sorted(user_ids))}
# item2idx = {m:i for i,m in enumerate(sorted(item_ids))}

# for col in ["user_id", "item_id"]:
#     ratings[col.replace("id", "id_old")] = ratings[col]          # keep originals if wanted
# ratings["uid"] = ratings.user_id.map(user2idx)
# ratings["iid"] = ratings.item_id.map(item2idx)

# trust["uid_src"] = trust.user_id_trustor.map(user2idx)
# trust["uid_dst"] = trust.user_id_trustee.map(user2idx)

# test_df["uid"] = test_df.user_id.map(user2idx)
# test_df["iid"] = test_df.item_id.map(item2idx)

# # ============================================================
# # 5 ▸ Helper functions (Surprise wrappers & feature builder)
# # ============================================================
# def surprise_fit(algo, df):
#     reader = Reader(rating_scale=(1,5))
#     data   = Dataset.load_from_df(df[["uid","iid","label"]], reader)
#     algo.fit(data.build_full_trainset())
#     return algo

# def surprise_pred(algo, df):
#     return np.array([algo.predict(int(u), int(i)).est for u,i in zip(df.uid, df.iid)])

# # Node2Vec + simple stats → feature matrix
# def build_feats(df, u_emb, user_mean, item_mean, global_mean, freq_u, freq_i):
#     feats = []
#     for _, r in df.iterrows():
#         uid, iid = int(r.uid), int(r.iid)
#         emb = u_emb.get(uid, np.zeros(128, dtype=np.float32))
#         feats.append(np.hstack([
#             emb,
#             user_mean.get(uid, global_mean),
#             item_mean.get(iid, global_mean),
#             freq_u.get(uid, 0),
#             freq_i.get(iid, 0)
#         ]))
#     return np.vstack(feats).astype(np.float32)

# # ============================================================
# # 6 ▸ Fast 3‑fold OOF  (single‑CPU version)
# # ============================================================
# K        = 3
# oof_cols = ["global","item","user","svd","svdpp","trust","lgb"]
# oof_stack = np.zeros((len(ratings), len(oof_cols)), dtype=np.float32)

# # 0️⃣ ── Node2Vec (single core)
# g_full = nx.from_pandas_edgelist(trust, "uid_src", "uid_dst",
#                                  create_using=nx.DiGraph())
# n2v    = Node2Vec(g_full,
#                   dimensions = 128,
#                   walk_length= 20,
#                   num_walks  = 10,
#                   workers    = 1,     # ← one CPU
#                   seed       = SEED)
# w2v    = n2v.fit(window=10, min_count=1)
# u_emb  = {int(k): w2v.wv[k] for k in w2v.wv.key_to_index}

# kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

# for fold,(tr_idx,va_idx) in enumerate(kf.split(ratings), 1):
#     tr = ratings.iloc[tr_idx].reset_index(drop=True)
#     va = ratings.iloc[va_idx].reset_index(drop=True)

#     # --- baselines
#     g_mean = tr.label.mean()
#     i_mean = tr.groupby('iid').label.mean().to_dict()
#     u_mean = tr.groupby('uid').label.mean().to_dict()
#     oof_stack[va_idx,0] = g_mean
#     oof_stack[va_idx,1] = va.iid.map(i_mean).fillna(g_mean)
#     oof_stack[va_idx,2] = va.uid.map(u_mean).fillna(g_mean)

#     # --- MF models (epochs trimmed)
#     svd   = surprise_fit(SVD  (n_factors=128, n_epochs=30, reg_all=0.02, random_state=SEED), tr)
#     svdpp = surprise_fit(SVDpp(n_factors=64, n_epochs=20, reg_all=0.02, random_state=SEED), tr)
#     oof_stack[va_idx,3] = surprise_pred(svd,   va)
#     oof_stack[va_idx,4] = surprise_pred(svdpp, va)

#     # --- TrustMF inside fold
#     n_items = ratings.iid.max() + 1
#     fake    = pd.DataFrame({"uid": trust.uid_src,
#                             "iid": trust.uid_dst + n_items,
#                             "label": trust.trust_value.clip(1,5)})
#     trust_m = surprise_fit(
#                  SVD(n_factors=128, n_epochs=15, reg_all=0.02, random_state=SEED),
#                  pd.concat([tr[["uid","iid","label"]], fake], ignore_index=True))
#     oof_stack[va_idx,5] = surprise_pred(trust_m, va)

#     # --- LightGBM (single core, 400 trees)
#     freq_u = tr.groupby('uid').size()
#     freq_i = tr.groupby('iid').size()
#     X_tr   = build_feats(tr, u_emb, u_mean, i_mean, g_mean, freq_u, freq_i)
#     X_va   = build_feats(va, u_emb, u_mean, i_mean, g_mean, freq_u, freq_i)

#     lgbm = lgb.LGBMRegressor(
#               n_estimators=700,
#               num_leaves  =64,
#               learning_rate=0.05,
#               subsample   =0.8,
#               random_state=SEED,
#               deterministic=True,
#               n_jobs      =1)
#     lgbm.fit(X_tr, tr.label)
#     oof_stack[va_idx,6] = lgbm.predict(X_va)

#     print(f"fold {fold}/{K} done")

# print("Fast single‑CPU OOF stack:", oof_stack.shape)

# # ============================================================
# # 7 ▸ Train blender on OOF predictions
# # ============================================================
# blender = Ridge(alpha=1.0, random_state=SEED)
# blender.fit(oof_stack, ratings.label)

# rmse_oof = np.sqrt(mean_squared_error(ratings.label, blender.predict(oof_stack)))
# print("OOF blend RMSE:", rmse_oof)

# # ============================================================
# # 8 ▸ Re-fit every base model on **all** data
# # ============================================================
# # Baselines (recomputed)
# global_mean = ratings.label.mean()
# item_mean   = ratings.groupby('iid').label.mean().to_dict()
# user_mean   = ratings.groupby('uid').label.mean().to_dict()

# # MF models (same hyper-params as in CV)
# full_svd   = surprise_fit(SVD  (n_factors=128, n_epochs=30,  reg_all=0.02, random_state=SEED), ratings)
# full_svdpp = surprise_fit(SVDpp(n_factors=64, n_epochs=20, reg_all=0.02, random_state=SEED), ratings)

# # TrustMF on full data
# n_items = ratings.iid.max()+1
# fake_rows = pd.DataFrame({"uid": trust.uid_src,
#                           "iid": trust.uid_dst + n_items,
#                           "label": trust.trust_value.clip(1,5)})
# aug_full  = pd.concat([ratings[["uid","iid","label"]], fake_rows])
# full_trust = surprise_fit(SVD(n_factors=128, n_epochs=25, reg_all=0.02, random_state=SEED), aug_full)

# # ============================================================
# # 9 ▸ LightGBM on full data (graph + stats)
# # ============================================================
# g_full = nx.from_pandas_edgelist(trust, "uid_src", "uid_dst", create_using=nx.DiGraph())
# n2v_full = Node2Vec(g_full, dimensions=128, walk_length=20, num_walks=10, workers=1, seed=SEED)
# w2v_full = n2v_full.fit(window=10, min_count=1)
# u_emb_full = {int(k): w2v_full.wv[k] for k in w2v_full.wv.key_to_index}

# freq_u_full = ratings.groupby('uid').size()
# freq_i_full = ratings.groupby('iid').size()
# X_full = build_feats(ratings, u_emb_full, user_mean, item_mean, global_mean,
#                      freq_u_full, freq_i_full)

# lgbm_full = lgb.LGBMRegressor(
#               n_estimators=700, num_leaves=64, learning_rate=0.05,
#               subsample=0.8, random_state=SEED, deterministic=True, n_jobs=1)
# lgbm_full.fit(X_full, ratings.label)

# # ============================================================
# # 10 ▸ Predict on TEST and write submission
# # ============================================================
# X_test = build_feats(test_df, u_emb_full, user_mean, item_mean, global_mean,
#                      freq_u_full, freq_i_full)

# test_stack = np.column_stack([
#     np.full(len(test_df), global_mean),
#     test_df.iid.map(item_mean).fillna(global_mean),
#     test_df.uid.map(user_mean).fillna(global_mean),
#     surprise_pred(full_svd,    test_df),
#     surprise_pred(full_svdpp,  test_df),
#     surprise_pred(full_trust,  test_df),
#     lgbm_full.predict(X_test)
# ])

# test_preds = np.clip(blender.predict(test_stack), 1, 5)
# submission = pd.DataFrame({
#     "id": test_df.get("id", pd.Series(np.arange(len(test_df)))),
#     "label": np.round(test_preds, 3)
# })
# submission.to_csv(OUT_DIR/"submission.csv", index=False)
# print("Saved →", OUT_DIR/"submission.csv")
# submission.head()

# Version 3

# ------------------- INSTALL REQUIRED LIBRARIES -------------------
# import os

# if not os.path.exists("data_flag"):
#   !gdown 1CyR9bMjbgOGDSRIgcZHrB5oWN4rJmyvb
#   !gdown 1fWw22JQbvOxkYdQ3nL5ruP7gNzXzLzFd
#   !gdown 1GS5Bf81MDbI_yAZMrhZs2MJDT6aC8jiR
#   !touch data_flag

# if not os.path.exists("lib_flag"):
#   !pip uninstall -y numpy
#   !pip install "numpy<2.0"
#   !pip install scikit-surprise
#   !touch lib_flag
#   import os, signal; os.kill(os.getpid(), signal.SIGKILL)

# # ------------------- IMPORT LIBRARIES -------------------
# import pandas as pd
# import numpy as np
# import random
# import os
# from pathlib import Path

# from surprise import Dataset, Reader, SVDpp
# from surprise import accuracy
# from surprise.model_selection import train_test_split

# # ------------------- CONFIGURATION -------------------
# SEED     = 42
# DATA_DIR = Path("/content")
# OUT_DIR  = Path("/content/outputs")
# OUT_DIR.mkdir(exist_ok=True)

# random.seed(SEED)
# np.random.seed(SEED)
# os.environ["PYTHONHASHSEED"] = str(SEED)

# # ------------------- 1) LOAD DATA -------------------
# ratings = pd.read_csv(DATA_DIR / "train_data_movie_rate.csv")
# test_df = pd.read_csv(DATA_DIR / "test_data.csv")
# trust_df = pd.read_csv(DATA_DIR / "train_data_movie_trust.csv")

# # ------------------- 2) CLEAN AND INDEX DATA -------------------
# ratings = ratings.drop_duplicates(subset=["user_id", "item_id"], keep="last").copy()
# ratings["label"] = pd.to_numeric(ratings["label"], errors="coerce").clip(1, 5)
# ratings = ratings.dropna(subset=["user_id", "item_id", "label"]).reset_index(drop=True)

# # Encode users/items
# user2idx = {u: i for i, u in enumerate(sorted(ratings.user_id.unique()))}
# item2idx = {m: i for i, m in enumerate(sorted(ratings.item_id.unique()))}
# ratings["uid"] = ratings.user_id.map(user2idx)
# ratings["iid"] = ratings.item_id.map(item2idx)
# test_df["uid"] = test_df.user_id.map(user2idx)
# test_df["iid"] = test_df.item_id.map(item2idx)

# # ------------------- 3) TRUST AUGMENTATION -------------------
# # Add pseudo-ratings based on trusted users' ratings
# trust_df = trust_df.copy()
# trust_df["trustor"] = trust_df["user_id_trustor"].map(user2idx)
# trust_df["trustee"] = trust_df["user_id_trustee"].map(user2idx)

# # Remove NaNs caused by mapping (users not in train set)
# trust_df = trust_df.dropna(subset=["trustor", "trustee"]).astype({"trustor": int, "trustee": int})

# # Merge ratings from trustee for each trustor
# augmented = []  # List to store augmented ratings

# MAX_PSEUDO_RATINGS_PER_TRUSTEE = 10

# for _, row in trust_df.iterrows():
#     trustor = row["user_id_trustor"]
#     trustee = row["user_id_trustee"]
#     trustee_ratings = ratings[ratings["uid"] == user2idx.get(trustee, None)]

#     if trustee_ratings.empty:
#         continue

#     # Limit the number of ratings to augment
#     trustee_ratings = trustee_ratings.sample(min(len(trustee_ratings), MAX_PSEUDO_RATINGS_PER_TRUSTEE))

#     for _, r in trustee_ratings.iterrows():
#         augmented.append({
#             "uid": user2idx.get(trustor, None),
#             "iid": r["iid"],
#             "label": min(5.0, r["label"] * 0.9)
#         })

# # Convert augmented list into a DataFrame
# augmented_df = pd.DataFrame(augmented)

# # Combine original ratings with augmented ratings
# augmented_ratings = pd.concat([ratings[["uid", "iid", "label"]], augmented_df], ignore_index=True)

# # ------------------- 4) TRAIN SVD++ MODEL -------------------
# reader = Reader(rating_scale=(1, 5))
# full_data = Dataset.load_from_df(augmented_ratings, reader)
# trainset, valset = train_test_split(full_data, test_size=0.1, random_state=SEED)

# svdpp = SVDpp(
#     n_factors=175,
#     n_epochs=100,
#     lr_all=0.0007,
#     reg_all=0.07,
#     random_state=SEED
# )
# svdpp.fit(trainset)

# # ------------------- 5) EVALUATE MODEL -------------------
# val_preds = svdpp.test(valset)
# rmse = accuracy.rmse(val_preds, verbose=True)
# mae  = accuracy.mae(val_preds, verbose=True)

# # ------------------- 6) PREDICT ON TEST DATA -------------------
# preds = [svdpp.predict(int(u), int(i)).est for u, i in zip(test_df.uid, test_df.iid)]
# preds = np.clip(preds, 1, 5)

# # ------------------- 7) SAVE SUBMISSION -------------------
# submission = pd.DataFrame({
#     "id": test_df.get("id", pd.Series(np.arange(len(test_df)))),
#     "label": np.round(preds, 3)
# })
# fname = OUT_DIR / "submission.csv"
# submission.to_csv(fname, index=False)
# print("Saved →", fname)
# submission.head()
