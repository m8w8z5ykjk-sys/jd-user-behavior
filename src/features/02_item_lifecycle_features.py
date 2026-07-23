"""
文件名称：02_item_lifecycle_features.py

功能：
1. 读取由 build_feature_table.py 生成的用户行为 Parquet 明细数据；
2. 将 time 字段统一转换为日期时间格式并提取 date；
3. 统计每个商品的总行为次数；
4. 统计每个商品最近7天的行为次数；
5. 计算商品热度衰减系数；
6. 获取每个商品所属的商品类别；
7. 统计每个类别包含的商品数量，构建类别竞争度特征；
8. 合并生成商品生命周期特征表；
9. 将本文件产生的所有 processed 输出保存到以当前 Python 文件名命名的目录中。

输入文件名：
- data/processed/build_feature_table/user_behavior_cleaned.parquet
  由 build_feature_table.py 生成的清洗后用户行为 Parquet 明细数据。

输入字段要求：
- item_id：商品编号；
- item_category：商品类别；
- time：用户行为发生时间。

输出目录：
- data/processed/02_item_lifecycle_features/

输出文件名：
- data/processed/02_item_lifecycle_features/item_lifecycle_features.parquet

输出文件主要字段：
- item_id：商品编号；
- item_total_behavior_count：商品总行为次数；
- item_recent_7_behavior_count：商品最近7天行为次数；
- heat_decay_score：最近7天行为次数占总行为次数的比例；
- item_category：商品类别；
- category_item_count：该类别包含的商品数量；
- category_competition_degree：商品类别竞争度。

目录规则：
- 本文件输出属于 data/processed；
- 程序会自动创建：
  data/processed/02_item_lifecycle_features/
- 本文件产生的所有 processed 文件均保存在该目录中。

路径检查：
- 当前文件应位于：项目根目录/src/features/；
- 项目根目录通过 Path(__file__).resolve().parents[2] 自动定位；
- 输入文件读取路径：
  项目根目录/data/processed/build_feature_table/user_behavior_cleaned.parquet；
- 输出文件保存路径：
  项目根目录/data/processed/02_item_lifecycle_features/item_lifecycle_features.parquet。
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
    / "02_item_lifecycle_features"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 明确输出文件
ITEM_LIFECYCLE_OUTPUT = OUTPUT_DIR / "item_lifecycle_features.parquet"

print("项目根目录：", BASE_DIR)
print("输入文件：", INPUT_FILE)
print("输出目录：", OUTPUT_DIR)
print("输出文件：", ITEM_LIFECYCLE_OUTPUT)


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

required_columns = {"item_id", "item_category", "time"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "输入文件缺少以下必要字段："
        + ", ".join(sorted(missing_columns))
    )

if df.empty:
    raise ValueError(f"输入文件为空：{INPUT_FILE}")

# 保证 time 字段为日期时间类型
df["time"] = pd.to_datetime(df["time"], errors="coerce")

invalid_time_count = int(df["time"].isna().sum())
if invalid_time_count > 0:
    print(f"警告：发现 {invalid_time_count} 条无效时间记录，已删除。")
    df = df.dropna(subset=["time"]).copy()

if df.empty:
    raise ValueError("删除无效时间记录后数据为空，无法构建商品生命周期特征。")

print("数据读取成功")
print("数据行数：", len(df))
print("字段名称：", list(df.columns))
print(df.head())


# ============================================================
# 3. 构建商品生命周期特征
# ============================================================

df["date"] = df["time"].dt.normalize()

# 1. 每个商品总行为次数
item_total_behavior = (
    df.groupby("item_id")
    .size()
    .reset_index(name="item_total_behavior_count")
)
print("商品总行为次数：")
print(item_total_behavior.head())

# 2. 最近7天开始日期
max_date = df["date"].max()
recent_7_days = max_date - pd.Timedelta(days=7)

print("数据最后一天：", max_date)
print("最近7天开始日期：", recent_7_days)

# 3. 筛选最近7天商品行为
df_recent_7 = df[df["date"] >= recent_7_days].copy()

# 4. 最近7天商品行为次数
item_recent_7_behavior = (
    df_recent_7.groupby("item_id")
    .size()
    .reset_index(name="item_recent_7_behavior_count")
)

# 5. 合并商品总行为次数和最近7天行为次数
item_lifecycle_features = item_total_behavior.merge(
    item_recent_7_behavior,
    on="item_id",
    how="left"
)

item_lifecycle_features["item_recent_7_behavior_count"] = (
    item_lifecycle_features["item_recent_7_behavior_count"]
    .fillna(0)
    .astype("int64")
)

# 6. 计算热度衰减系数
item_lifecycle_features["heat_decay_score"] = (
    item_lifecycle_features["item_recent_7_behavior_count"]
    / item_lifecycle_features["item_total_behavior_count"]
)

# 7. 获取每个商品所属类目
item_category = (
    df.groupby("item_id", as_index=False)["item_category"]
    .first()
)

item_lifecycle_features = item_lifecycle_features.merge(
    item_category,
    on="item_id",
    how="left"
)

# 8. 计算类目竞争度
category_item_count = (
    item_lifecycle_features.groupby(
        "item_category",
        dropna=False
    )["item_id"]
    .nunique()
    .reset_index(name="category_item_count")
)

item_lifecycle_features = item_lifecycle_features.merge(
    category_item_count,
    on="item_category",
    how="left"
)

item_lifecycle_features["category_competition_degree"] = (
    item_lifecycle_features["category_item_count"]
)


# ============================================================
# 4. 保存输出文件
# ============================================================

item_lifecycle_features.to_parquet(
    ITEM_LIFECYCLE_OUTPUT,
    index=False
)

if not ITEM_LIFECYCLE_OUTPUT.exists():
    raise RuntimeError(f"输出文件保存失败：{ITEM_LIFECYCLE_OUTPUT}")

print("商品生命周期特征构建完成")
print(item_lifecycle_features.head())
print("输出数据行数：", len(item_lifecycle_features))
print("商品生命周期特征已保存：", ITEM_LIFECYCLE_OUTPUT)
