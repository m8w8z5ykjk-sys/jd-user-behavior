import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

time_features = pd.read_parquet(PROCESSED_DIR / "time_features.parquet")
behavior_sequence_features = pd.read_parquet(PROCESSED_DIR / "behavior_sequence_features.parquet")
business_features = pd.read_parquet(PROCESSED_DIR / "business_features.parquet")
user_svd_features = pd.read_parquet(PROCESSED_DIR / "user_svd_features.parquet")

print("时间行为特征：", time_features.shape)
print("行为序列特征：", behavior_sequence_features.shape)
print("业务导向特征：", business_features.shape)
print("SVD隐式特征：", user_svd_features.shape)

feature_table = time_features.copy()
feature_table = feature_table.merge(behavior_sequence_features,on="user_id",how="left")
feature_table = feature_table.merge(business_features,on="user_id",how="left")
feature_table = feature_table.merge(user_svd_features,on="user_id",how="left")
feature_table.to_parquet(OUTPUT_DIR / "feature_table.parquet",index=False)

#宽表校验
print("所有字段：")
print(feature_table.columns.tolist())

duplicate_users = feature_table["user_id"].duplicated().sum()
print("重复用户数：")
print(duplicate_users)

missing_rate = (feature_table.isnull().mean().sort_values(ascending=False))
print("缺失率最高的前20个字段：")
print(missing_rate.head(20))

unique_count = feature_table.nunique()
print("每个字段唯一值数量：")
print(unique_count)

low_variance_features = unique_count[unique_count <= 1].index.tolist()
print("低区分度特征：")
print(low_variance_features)

duplicate_columns_count = feature_table.columns.duplicated().sum()
print("重复列名数量：")
print(duplicate_columns_count)

#只保留主要特征,删除低区分度特征
feature_table_cleaned = feature_table.drop(columns=low_variance_features)
print("删除低区分度特征后的宽表大小：")
print(feature_table_cleaned.shape)

#缺失值处理
numeric_cols = feature_table_cleaned.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
for col in numeric_cols:
    if col != "user_id":
        feature_table_cleaned[col] = feature_table_cleaned[col].fillna(0)


object_cols = feature_table_cleaned.select_dtypes(include=["object"]).columns.tolist()
for col in object_cols:
    feature_table_cleaned[col] = feature_table_cleaned[col].fillna("unknown")

#保存校验后的特征宽表
FEATURE_TABLE_OUTPUT = OUTPUT_DIR / "feature_table_cleaned.parquet"
feature_table_cleaned.to_parquet(FEATURE_TABLE_OUTPUT,index=False)

print("=" * 60)
print("清洗后的特征宽表已保存：")
print(FEATURE_TABLE_OUTPUT)
print(feature_table_cleaned.head())
