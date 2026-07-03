import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
print(BASE_DIR)
INPUT_FILE = BASE_DIR / "data" / "processed" / "user_behavior_cleaned.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(INPUT_FILE)
print("数据读取成功")
print(df.head())
print(df.info())
print(df.columns)
df["date"] = df["time"].dt.date
df["date"] = pd.to_datetime(df["date"])
#计算每个用户的总行为次数
user_total_behavior = (df.groupby("user_id").size().reset_index(name="total_behavior_count"))
print(user_total_behavior.head())
#计算每个用户的活跃天数
user_active_days = (df.groupby("user_id")["date"].nunique().reset_index(name="active_days"))
print(user_active_days.head())
#计算每个用户最近30天行为次数
max_date = df["date"].max()
recent_30_days = max_date - pd.Timedelta(days=30)
df_recent = df[df["date"] >= recent_30_days]
user_recent_30_behavior = (df_recent.groupby("user_id").size().reset_index(name="recent_30_behavior_count"))
#给用户划分活跃度等级
def active_level(count):
    if count >= 100:
        return "高活跃用户"
    elif count >= 30:
        return "中活跃用户"
    else:
        return "低活跃用户"
user_recent_30_behavior["active_level"] = user_recent_30_behavior["recent_30_behavior_count"].apply(active_level)
print(user_recent_30_behavior.head())
#计算每个用户最后一次活跃日期
user_last_active = (df.groupby("user_id")["date"].max().reset_index(name="last_active_date"))
#计算流失间隔
user_last_active["churn_gap_days"] = (max_date - user_last_active["last_active_date"]).dt.days
print(user_last_active.head())
#计算连续活跃天数
user_dates = (df[["user_id", "date"]].drop_duplicates().sort_values(["user_id", "date"]))
user_dates["date_diff"] = (user_dates.groupby("user_id")["date"].diff().dt.days)
user_dates["is_new_streak"] = user_dates["date_diff"] != 1
user_dates["streak_group"] = (user_dates.groupby("user_id")["is_new_streak"].cumsum())
user_streak = (user_dates.groupby(["user_id", "streak_group"]).size().reset_index(name="streak_days"))
user_max_streak = (user_streak.groupby("user_id")["streak_days"].max().reset_index(name="max_continuous_active_days"))
#合并所有用户生命周期特征
user_lifecycle_features = user_total_behavior.merge(user_active_days,on="user_id",how="left")
user_lifecycle_features = user_lifecycle_features.merge(user_recent_30_behavior,on="user_id",how="left")
user_lifecycle_features = user_lifecycle_features.merge(user_last_active,on="user_id",how="left")
user_lifecycle_features = user_lifecycle_features.merge(user_max_streak,on="user_id",how="left")
print(user_lifecycle_features.head())
#保存用户生命周期特征
USER_LIFECYCLE_OUTPUT = OUTPUT_DIR   / "user_lifecycle_features.parquet"
user_lifecycle_features.to_parquet(USER_LIFECYCLE_OUTPUT, index=False)
print("用户生命周期特征已保存：", USER_LIFECYCLE_OUTPUT)