"""
文件名称：01_time_features.py

功能：
1. 读取由 build_feature_table.py 生成的用户行为 Parquet 明细数据；
2. 将 time 字段转换为标准日期时间，并提取行为发生小时；
3. 统计每个用户在每个小时的行为次数和行为占比；
4. 识别每个用户最活跃的小时；
5. 将一天划分为凌晨、上午、下午和晚上四个时间段；
6. 统计每个用户在各时间段的行为次数和占比；
7. 识别每个用户最活跃的时间段；
8. 统计每个用户的夜间行为次数和夜间行为占比；
9. 合并生成用户时间行为特征表；
10. 将本文件产生的所有 processed 输出保存到以当前 Python 文件名命名的目录中。

输入文件名：
- data/processed/build_feature_table/user_behavior_cleaned.parquet
  由 build_feature_table.py 生成的清洗后用户行为 Parquet 明细数据。

输入字段要求：
- user_id：用户编号；
- time：用户行为发生时间。

输出目录：
- data/processed/01_time_features/

输出文件名：
- data/processed/01_time_features/time_features.parquet

输出文件主要字段：
- user_id：用户编号；
- total_behavior_count：用户总行为次数；
- peak_active_hour：用户最活跃小时；
- peak_hour_behavior_count：最活跃小时的行为次数；
- peak_hour_behavior_ratio：最活跃小时行为次数占总行为次数的比例；
- peak_active_period：用户最活跃时间段；
- peak_period_behavior_count：最活跃时间段的行为次数；
- peak_period_behavior_ratio：最活跃时间段行为次数占总行为次数的比例；
- night_behavior_count：夜间行为次数；
- night_behavior_ratio：夜间行为次数占总行为次数的比例。

目录规则：
- 本文件输出属于 data/processed；
- 程序会自动创建：
  data/processed/01_time_features/
- 本文件产生的所有 processed 文件均保存在该目录中。

路径检查：
- 当前文件应位于：项目根目录/src/features/；
- 项目根目录通过 Path(__file__).resolve().parents[2] 自动定位；
- 输入文件读取路径：
  项目根目录/data/processed/build_feature_table/user_behavior_cleaned.parquet；
- 输出文件保存路径：
  项目根目录/data/processed/01_time_features/time_features.parquet。
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

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

# 本文件专属 processed 输出目录
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "01_time_features"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 明确输出文件
TIME_FEATURES_OUTPUT = OUTPUT_DIR / "time_features.parquet"

print("项目根目录：", BASE_DIR)
print("输入文件：", INPUT_FILE)
print("输出目录：", OUTPUT_DIR)
print("输出文件：", TIME_FEATURES_OUTPUT)


# ============================================================
# 2. 输入文件与字段检查
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"未找到输入文件：{INPUT_FILE}\n"
        "请先运行 src/build_feature_table.py，生成：\n"
        "data/processed/build_feature_table/user_behavior_cleaned.parquet"
    )

df = pd.read_parquet(INPUT_FILE)

required_columns = {"user_id", "time"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "输入文件缺少以下必要字段："
        + ", ".join(sorted(missing_columns))
    )

if df.empty:
    raise ValueError(f"输入文件为空：{INPUT_FILE}")

# 统一时间格式
df["time"] = pd.to_datetime(df["time"], errors="coerce")

invalid_time_count = int(df["time"].isna().sum())
if invalid_time_count > 0:
    print(f"警告：发现 {invalid_time_count} 条无效时间记录，已删除。")
    df = df.dropna(subset=["time"]).copy()

if df.empty:
    raise ValueError("删除无效时间记录后数据为空，无法构建时间特征。")

print("数据读取成功")
print("数据行数：", len(df))
print("字段名称：", list(df.columns))
print(df.head())


# ============================================================
# 3. 构建用户时间行为特征
# ============================================================

# 提取小时
df["hour"] = df["time"].dt.hour

# 3.1 每个用户每小时行为次数
user_hour_behavior = (
    df.groupby(["user_id", "hour"])
    .size()
    .reset_index(name="hour_behavior_count")
)

# 3.2 每个用户总行为次数
user_total_behavior = (
    df.groupby("user_id")
    .size()
    .reset_index(name="total_behavior_count")
)

# 3.3 计算每小时行为占比
user_hour_behavior = user_hour_behavior.merge(
    user_total_behavior,
    on="user_id",
    how="left"
)

user_hour_behavior["hour_behavior_ratio"] = (
    user_hour_behavior["hour_behavior_count"]
    / user_hour_behavior["total_behavior_count"]
)

# 3.4 找到每个用户最活跃小时
user_peak_hour = (
    user_hour_behavior
    .sort_values(
        ["user_id", "hour_behavior_count", "hour"],
        ascending=[True, False, True]
    )
    .drop_duplicates("user_id")
    .reset_index(drop=True)
)

user_peak_hour = user_peak_hour[
    [
        "user_id",
        "hour",
        "hour_behavior_count",
        "hour_behavior_ratio"
    ]
].rename(
    columns={
        "hour": "peak_active_hour",
        "hour_behavior_count": "peak_hour_behavior_count",
        "hour_behavior_ratio": "peak_hour_behavior_ratio"
    }
)

# 3.5 划分时间段
def time_period(hour: int) -> str:
    """将小时划分为凌晨、上午、下午和晚上。"""
    if 0 <= hour < 6:
        return "凌晨"
    if 6 <= hour < 12:
        return "上午"
    if 12 <= hour < 18:
        return "下午"
    return "晚上"


df["time_period"] = df["hour"].apply(time_period)

# 3.6 每个用户各时间段行为次数
user_period_behavior = (
    df.groupby(["user_id", "time_period"])
    .size()
    .reset_index(name="period_behavior_count")
)

user_period_behavior = user_period_behavior.merge(
    user_total_behavior,
    on="user_id",
    how="left"
)

user_period_behavior["period_behavior_ratio"] = (
    user_period_behavior["period_behavior_count"]
    / user_period_behavior["total_behavior_count"]
)

# 3.7 找到每个用户最活跃时间段
user_peak_period = (
    user_period_behavior
    .sort_values(
        ["user_id", "period_behavior_count", "time_period"],
        ascending=[True, False, True]
    )
    .drop_duplicates("user_id")
    .reset_index(drop=True)
)

user_peak_period = user_peak_period[
    [
        "user_id",
        "time_period",
        "period_behavior_count",
        "period_behavior_ratio"
    ]
].rename(
    columns={
        "time_period": "peak_active_period",
        "period_behavior_count": "peak_period_behavior_count",
        "period_behavior_ratio": "peak_period_behavior_ratio"
    }
)

# 3.8 夜间行为占比
df["is_night"] = (
    (df["hour"] >= 18) | (df["hour"] < 6)
).astype("int8")

user_night_behavior = (
    df.groupby("user_id")["is_night"]
    .sum()
    .reset_index(name="night_behavior_count")
)

user_night_behavior = user_night_behavior.merge(
    user_total_behavior,
    on="user_id",
    how="left"
)

user_night_behavior["night_behavior_ratio"] = (
    user_night_behavior["night_behavior_count"]
    / user_night_behavior["total_behavior_count"]
)

# 3.9 合并用户时间行为特征
user_time_behavior_features = user_total_behavior.merge(
    user_peak_hour,
    on="user_id",
    how="left"
)

user_time_behavior_features = user_time_behavior_features.merge(
    user_peak_period,
    on="user_id",
    how="left"
)

user_time_behavior_features = user_time_behavior_features.merge(
    user_night_behavior[
        [
            "user_id",
            "night_behavior_count",
            "night_behavior_ratio"
        ]
    ],
    on="user_id",
    how="left"
)

print("用户时间行为特征示例：")
print(user_time_behavior_features.head())


# ============================================================
# 4. 保存输出文件
# ============================================================

user_time_behavior_features.to_parquet(
    TIME_FEATURES_OUTPUT,
    index=False
)

if not TIME_FEATURES_OUTPUT.exists():
    raise RuntimeError(
        f"输出文件保存失败：{TIME_FEATURES_OUTPUT}"
    )

print("用户时间行为特征构建完成")
print("输出数据行数：", len(user_time_behavior_features))
print("输出数据列数：", len(user_time_behavior_features.columns))
print("用户时间行为特征已保存：", TIME_FEATURES_OUTPUT)
