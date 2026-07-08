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
df["hour"] = pd.to_datetime(df["time"]).dt.hour
print(df[["time", "hour"]].head())

#每个用户每小时行为次数
user_hour_behavior = (df.groupby(["user_id", "hour"]).size().reset_index(name="hour_behavior_count"))
print(user_hour_behavior.head())
#每个用户总行为次数
user_total_behavior = (df.groupby("user_id").size().reset_index(name="total_behavior_count"))
#计算每小时行为占比
user_hour_behavior = user_hour_behavior.merge(user_total_behavior,on="user_id",how="left")
user_hour_behavior["hour_behavior_ratio"] = (user_hour_behavior["hour_behavior_count"]/ user_hour_behavior["total_behavior_count"])
#找到每个用户最活跃小时
user_peak_hour = (user_hour_behavior.sort_values(["user_id", "hour_behavior_count"],ascending=[True, False])
    .drop_duplicates("user_id").reset_index(drop=True))
user_peak_hour = user_peak_hour[["user_id", "hour", "hour_behavior_count", "hour_behavior_ratio"]]
user_peak_hour = user_peak_hour.rename(
    columns={"hour": "peak_active_hour","hour_behavior_count": "peak_hour_behavior_count","hour_behavior_ratio": "peak_hour_behavior_ratio"})
print(user_peak_hour.head())
#划分时间段
def time_period(hour):
    if 0 <= hour < 6:
        return "凌晨"
    elif 6 <= hour < 12:
        return "上午"
    elif 12 <= hour < 18:
        return "下午"
    else:
        return "晚上"
df["time_period"] = df["hour"].apply(time_period)
#每个用户各时间段行为次数
user_period_behavior = (df.groupby(["user_id", "time_period"]).size().reset_index(name="period_behavior_count"))
user_period_behavior = user_period_behavior.merge(user_total_behavior,on="user_id",how="left")
user_period_behavior["period_behavior_ratio"] = (user_period_behavior["period_behavior_count"]/ user_period_behavior["total_behavior_count"])
#找到每个用户最活跃时间段
user_peak_period = (user_period_behavior.sort_values(["user_id", "period_behavior_count"],ascending=[True, False]).drop_duplicates("user_id").reset_index(drop=True))
user_peak_period = user_peak_period[["user_id", "time_period", "period_behavior_count", "period_behavior_ratio"]]
user_peak_period = user_peak_period.rename(columns={"time_period": "peak_active_period",
        "period_behavior_count": "peak_period_behavior_count",
        "period_behavior_ratio": "peak_period_behavior_ratio"})
#夜间行为占比
df["is_night"] = df["hour"].apply(lambda x: 1 if x >= 18 or x < 6 else 0)
user_night_behavior = (df.groupby("user_id")["is_night"].sum().reset_index(name="night_behavior_count"))
user_night_behavior = user_night_behavior.merge(user_total_behavior,on="user_id",how="left")
user_night_behavior["night_behavior_ratio"] = (user_night_behavior["night_behavior_count"]/ user_night_behavior["total_behavior_count"])
#合并用户时间行为特征
user_time_behavior_features = user_total_behavior.merge(user_peak_hour,on="user_id",how="left")
user_time_behavior_features = user_time_behavior_features.merge(user_peak_period,on="user_id",how="left")
user_time_behavior_features = user_time_behavior_features.merge(user_night_behavior[["user_id", "night_behavior_count", "night_behavior_ratio"]],on="user_id",how="left")

print("用户时间行为特征已保存：")