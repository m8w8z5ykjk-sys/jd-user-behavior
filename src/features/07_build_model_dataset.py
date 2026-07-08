import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR

feature_table = pd.read_parquet(PROCESSED_DIR / "feature_table_cleaned.parquet")
df = pd.read_parquet(PROCESSED_DIR / "user_behavior_cleaned.parquet")
df["datetime"] = pd.to_datetime(df["time"])
df["date"] = pd.to_datetime(df["datetime"].dt.date)

#用时间窗口构建未来7天购买标签
max_date = df["date"].max()
feature_end_date = max_date - pd.Timedelta(days=7)
label_df = df[df["date"] > feature_end_date]
future_buy_users = (label_df[label_df["behavior_type"] == 4]["user_id"].drop_duplicates())
feature_table["label"] = feature_table["user_id"].isin(future_buy_users).astype(int)

print("数据最后一天：", max_date)
print("特征截止日期：", feature_end_date)
print("未来7天购买用户数：", feature_table["label"].sum())
print("标签分布：")
print(feature_table["label"].value_counts())

#按时间窗口过滤特征用户，避免数据泄露
history_users = (df[df["date"] <= feature_end_date]["user_id"].drop_duplicates())
model_dataset = feature_table[feature_table["user_id"].isin(history_users)].copy()
print("时间窗口过滤后的数据：", model_dataset.shape)
print(model_dataset["label"].value_counts())

#类别不平衡处理：下采样
positive_samples = model_dataset[model_dataset["label"] == 1]
negative_samples = model_dataset[model_dataset["label"] == 0]
sample_size = min(len(positive_samples), len(negative_samples))
positive_sampled = positive_samples.sample(n=sample_size,random_state=42)
negative_sampled = negative_samples.sample(n=sample_size,random_state=42)
balanced_dataset = pd.concat([positive_sampled, negative_sampled],axis=0).sample(frac=1,random_state=42).reset_index(drop=True)
print(balanced_dataset["label"].value_counts())

MODEL_DATASET_OUTPUT = OUTPUT_DIR / "model_dataset_balanced.parquet"
balanced_dataset.to_parquet(MODEL_DATASET_OUTPUT,index=False)
print("最终建模数据集已保存：")
print(MODEL_DATASET_OUTPUT)
print(balanced_dataset.head())
