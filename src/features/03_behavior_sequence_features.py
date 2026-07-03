import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "user_behavior_cleaned.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(INPUT_FILE)
df["datetime"] = pd.to_datetime(df["time"])
print("数据读取成功")
print(df.head())
print(df.columns)
df = df.sort_values(["user_id", "datetime"])
print(df[["user_id", "behavior_type", "datetime"]].head(20))
#计算相邻行为时间间隔
df["previous_datetime"] = (df.groupby("user_id")["datetime"].shift(1))
df["behavior_interval"] = (df["datetime"] - df["previous_datetime"])
df["behavior_interval_seconds"] = (df["behavior_interval"].dt.total_seconds())
print(df[["user_id", "behavior_interval","datetime", "previous_datetime","behavior_type","behavior_interval_seconds"]].head(20))
#行为跳转概率
df["previous_behavior_type"] = (df.groupby("user_id")["behavior_type"].shift(1))
print(df[["user_id","previous_behavior_type","behavior_type"]].head(20))
transition_df = df.dropna(subset=["previous_behavior_type"])
transition_count = (transition_df.groupby(["previous_behavior_type", "behavior_type"]).size().reset_index(name="transition_count"))
print(transition_count.head())
previous_behavior_total = (transition_count.groupby("previous_behavior_type")["transition_count"].sum().reset_index(name="previous_behavior_total"))
print(previous_behavior_total.head())
behavior_transition_features = transition_count.merge(previous_behavior_total,on="previous_behavior_type",how="left")
behavior_transition_features["transition_probability"] = (
    behavior_transition_features["transition_count"]
    / behavior_transition_features["previous_behavior_total"])
print(behavior_transition_features.head())
#连续行为模式
df["is_same_behavior_as_previous"] = (df["behavior_type"] == df["previous_behavior_type"])
print(df[["user_id","previous_behavior_type","behavior_type","is_same_behavior_as_previous"]].head(20))
continuous_behavior_count = (df.groupby("user_id")["is_same_behavior_as_previous"].sum().reset_index(name="continuous_same_behavior_count"))
print(continuous_behavior_count.head())
behavior_sequence_user_features = (df.groupby("user_id")["behavior_interval_seconds"].mean().reset_index(name="avg_behavior_interval_seconds"))
behavior_sequence_user_features = behavior_sequence_user_features.merge(continuous_behavior_count,on="user_id",how="left")
print(behavior_sequence_user_features.head())
BEHAVIOR_SEQUENCE_OUTPUT = OUTPUT_DIR / "behavior_sequence_features.parquet"
behavior_sequence_user_features.to_parquet(BEHAVIOR_SEQUENCE_OUTPUT,index=False)
print("行为序列特征已保存：")
print(BEHAVIOR_SEQUENCE_OUTPUT)