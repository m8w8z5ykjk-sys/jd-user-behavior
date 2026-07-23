"""
文件名称：01_user_lifecycle_features.py

功能：
1. 读取由 build_feature_table.py 生成的清洗后用户行为 Parquet 数据；
2. 将 time 字段转换为日期字段；
3. 统计每个用户的总行为次数；
4. 统计每个用户的活跃天数；
5. 统计每个用户最近30天的行为次数；
6. 根据最近30天行为次数划分短期高活跃、中活跃和低活跃用户；
7. 计算每个用户最后一次活跃日期和流失间隔；
8. 计算每个用户的最大连续活跃天数；
9. 合并生成短期用户生命周期特征表；
10. 将本文件产生的所有 processed 输出保存到以当前 Python 文件名命名的目录中。

输入文件名：
- data/processed/build_feature_table/user_behavior_cleaned.parquet
  由 build_feature_table.py 生成的清洗后用户行为 Parquet 明细数据。
  主要使用字段：
  user_id、time。

输出目录：
- data/processed/01_user_lifecycle_features/

输出文件名：
- data/processed/01_user_lifecycle_features/user_lifecycle_features.parquet
  用户短期生命周期特征表，主要包含：
  user_id、total_behavior_count、active_days、
  recent_30_behavior_count、short_term_active_level、
  last_active_date、churn_gap_days、max_continuous_active_days。

目录规则：
- 本文件输出属于 data/processed；
- 程序会自动创建：
  data/processed/01_user_lifecycle_features/
- 本文件生成的所有 processed 文件都保存到该目录。

说明：
- 当前数据集时间跨度较短，因此本文件构建的是短期用户活跃特征，
  不能完全代表用户长期生命周期阶段。
"""

import pandas as pd
from pathlib import Path

# 自动定位项目根目录
# 当前文件位于：项目根目录/src/features/
BASE_DIR = Path(__file__).resolve().parents[2]

# 输入文件：由 build_feature_table.py 生成
INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "build_feature_table"
    / "user_behavior_cleaned.parquet"
)

# 本脚本专属输出目录
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "01_user_lifecycle_features"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("项目根目录：", BASE_DIR)
print("输入文件：", INPUT_FILE)
print("输出目录：", OUTPUT_DIR)


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"未找到输入文件：{INPUT_FILE}\n"
        "请先运行 src/build_feature_table.py，生成 "
        "data/processed/build_feature_table/user_behavior_cleaned.parquet"
    )

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
        return "短期高活跃用户"
    elif count >= 30:
        return "短期中活跃用户"
    else:
        return "短期低活跃用户"
user_recent_30_behavior["short_term_active_level"] = user_recent_30_behavior["recent_30_behavior_count"].apply(active_level)
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
#合并所有短期用户生命周期特征
user_lifecycle_features = user_total_behavior.merge(user_active_days,on="user_id",how="left")
user_lifecycle_features = user_lifecycle_features.merge(user_recent_30_behavior,on="user_id",how="left")
user_lifecycle_features = user_lifecycle_features.merge(user_last_active,on="user_id",how="left")
user_lifecycle_features = user_lifecycle_features.merge(user_max_streak,on="user_id",how="left")
print(user_lifecycle_features.head())
#保存短期用户生命周期特征
USER_LIFECYCLE_OUTPUT = OUTPUT_DIR / "user_lifecycle_features.parquet"
user_lifecycle_features.to_parquet(USER_LIFECYCLE_OUTPUT, index=False)
print("短期用户生命周期特征已保存：", USER_LIFECYCLE_OUTPUT)
#由于当前数据集时间跨度较短，用户行为不足以体现完整生命周期阶段变化
