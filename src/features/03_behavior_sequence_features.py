"""
文件名称：03_behavior_sequence_features.py

功能：
1. 读取由 build_feature_table.py 生成的用户行为 Parquet 明细数据；
2. 将 time 字段转换为标准日期时间；
3. 按 user_id 和行为时间排序用户行为；
4. 计算同一用户相邻两次行为之间的时间间隔；
5. 计算不同行为类型之间的跳转次数和跳转概率；
6. 识别用户连续执行相同行为的情况；
7. 统计每个用户的平均行为间隔和连续相同行为次数；
8. 生成用户行为序列特征表；
9. 将本文件产生的所有 processed 输出保存到以当前 Python 文件名命名的目录中。

输入文件名：
- data/processed/build_feature_table/user_behavior_cleaned.parquet
  由 build_feature_table.py 生成的清洗后用户行为 Parquet 明细数据。

输入字段要求：
- user_id：用户编号；
- behavior_type：行为类型；
- time：行为发生时间。

输出目录：
- data/processed/03_behavior_sequence_features/

输出文件名：
- data/processed/03_behavior_sequence_features/behavior_sequence_features.parquet

输出文件主要字段：
- user_id：用户编号；
- avg_behavior_interval_seconds：用户相邻行为的平均时间间隔，单位为秒；
- continuous_same_behavior_count：用户连续执行相同行为的次数。

目录规则：
- 本文件输出属于 data/processed；
- 程序会自动创建：
  data/processed/03_behavior_sequence_features/
- 本文件产生的所有 processed 文件均保存在该目录中。

路径检查：
- 当前文件应位于：项目根目录/src/features/；
- 项目根目录通过 Path(__file__).resolve().parents[2] 自动定位；
- 输入文件读取路径：
  项目根目录/data/processed/build_feature_table/user_behavior_cleaned.parquet；
- 输出文件保存路径：
  项目根目录/data/processed/03_behavior_sequence_features/behavior_sequence_features.parquet。
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
    / "03_behavior_sequence_features"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 明确输出文件
BEHAVIOR_SEQUENCE_OUTPUT = (
    OUTPUT_DIR / "behavior_sequence_features.parquet"
)

print("项目根目录：", BASE_DIR)
print("输入文件：", INPUT_FILE)
print("输出目录：", OUTPUT_DIR)
print("输出文件：", BEHAVIOR_SEQUENCE_OUTPUT)


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

required_columns = {"user_id", "behavior_type", "time"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "输入文件缺少以下必要字段："
        + ", ".join(sorted(missing_columns))
    )

if df.empty:
    raise ValueError(f"输入文件为空：{INPUT_FILE}")

df["datetime"] = pd.to_datetime(df["time"], errors="coerce")

invalid_time_count = int(df["datetime"].isna().sum())
if invalid_time_count > 0:
    print(f"警告：发现 {invalid_time_count} 条无效时间记录，已删除。")
    df = df.dropna(subset=["datetime"]).copy()

if df.empty:
    raise ValueError("删除无效时间记录后数据为空，无法构建行为序列特征。")

print("数据读取成功")
print("数据行数：", len(df))
print("字段名称：", list(df.columns))
print(df.head())


# ============================================================
# 3. 行为序列特征计算
# ============================================================

# 按用户和行为时间排序
df = df.sort_values(["user_id", "datetime"]).reset_index(drop=True)

# 3.1 计算同一用户相邻行为时间间隔
df["previous_datetime"] = (
    df.groupby("user_id")["datetime"].shift(1)
)

df["behavior_interval"] = (
    df["datetime"] - df["previous_datetime"]
)

df["behavior_interval_seconds"] = (
    df["behavior_interval"].dt.total_seconds()
)

print("相邻行为时间间隔示例：")
print(
    df[
        [
            "user_id",
            "behavior_interval",
            "datetime",
            "previous_datetime",
            "behavior_type",
            "behavior_interval_seconds"
        ]
    ].head(20)
)

# 3.2 计算行为跳转次数和跳转概率
df["previous_behavior_type"] = (
    df.groupby("user_id")["behavior_type"].shift(1)
)

transition_df = df.dropna(
    subset=["previous_behavior_type"]
).copy()

transition_count = (
    transition_df.groupby(
        ["previous_behavior_type", "behavior_type"]
    )
    .size()
    .reset_index(name="transition_count")
)

previous_behavior_total = (
    transition_count.groupby("previous_behavior_type")[
        "transition_count"
    ]
    .sum()
    .reset_index(name="previous_behavior_total")
)

behavior_transition_features = transition_count.merge(
    previous_behavior_total,
    on="previous_behavior_type",
    how="left"
)

behavior_transition_features["transition_probability"] = (
    behavior_transition_features["transition_count"]
    / behavior_transition_features["previous_behavior_total"]
)

print("行为跳转概率示例：")
print(behavior_transition_features.head())

# 3.3 识别连续相同行为
df["is_same_behavior_as_previous"] = (
    df["behavior_type"] == df["previous_behavior_type"]
)

continuous_behavior_count = (
    df.groupby("user_id")["is_same_behavior_as_previous"]
    .sum()
    .reset_index(name="continuous_same_behavior_count")
)

# 3.4 构建用户级行为序列特征
behavior_sequence_user_features = (
    df.groupby("user_id")["behavior_interval_seconds"]
    .mean()
    .reset_index(name="avg_behavior_interval_seconds")
)

behavior_sequence_user_features = (
    behavior_sequence_user_features.merge(
        continuous_behavior_count,
        on="user_id",
        how="left"
    )
)

behavior_sequence_user_features[
    "continuous_same_behavior_count"
] = (
    behavior_sequence_user_features[
        "continuous_same_behavior_count"
    ]
    .fillna(0)
    .astype("int64")
)

print("用户行为序列特征示例：")
print(behavior_sequence_user_features.head())


# ============================================================
# 4. 保存输出文件
# ============================================================

behavior_sequence_user_features.to_parquet(
    BEHAVIOR_SEQUENCE_OUTPUT,
    index=False
)

if not BEHAVIOR_SEQUENCE_OUTPUT.exists():
    raise RuntimeError(
        f"输出文件保存失败：{BEHAVIOR_SEQUENCE_OUTPUT}"
    )

print("行为序列特征构建完成")
print("输出数据行数：", len(behavior_sequence_user_features))
print("行为序列特征已保存：", BEHAVIOR_SEQUENCE_OUTPUT)
