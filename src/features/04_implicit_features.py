import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "user_behavior_cleaned.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(INPUT_FILE)
df["datetime"] = pd.to_datetime(df["time"])
print("数据读取成功")
print(df.head())
print(df.columns)
behavior_score_map = {1:1,2:2,3:3,4:4}
df["behavior_score"] = df["behavior_type"].map(behavior_score_map)
from scipy.sparse import coo_matrix
user_codes = df["user_id"].astype("category").cat.codes
item_codes = df["item_id"].astype("category").cat.codes
interaction_matrix = coo_matrix((df["behavior_score"],(user_codes, item_codes)))
#SVD矩阵分解
svd = TruncatedSVD(n_components=20,random_state=42)
user_svd_matrix = svd.fit_transform(interaction_matrix )
print(user_svd_matrix[:5])

import numpy as np
user_svd_features = pd.DataFrame(user_svd_matrix,columns=[f"user_feature_{i}" for i in range(20)])
user_id_map = (df[["user_id"]].drop_duplicates().reset_index(drop=True))
#把 user_id 加回 SVD 特征表
user_svd_features = pd.concat([user_id_map, user_svd_features],axis=1)
print(user_svd_features.head())

USER_SVD_OUTPUT = OUTPUT_DIR / "user_svd_features.parquet"
user_svd_features.to_parquet(USER_SVD_OUTPUT,index=False)
print("用户SVD隐式特征已保存：")
print(USER_SVD_OUTPUT)