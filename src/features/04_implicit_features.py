"""
文件名称：04_implicit_features.py

功能：
1. 读取由 build_feature_table.py 生成的用户行为 Parquet 明细数据；
2. 检查隐式特征计算所需的字段和数据有效性；
3. 根据行为类型设置交互权重：
   浏览=1、收藏=2、加购=3、购买=4；
4. 构建用户—商品稀疏交互矩阵；
5. 使用 TruncatedSVD 对交互矩阵进行降维；
6. 提取用户隐式语义特征；
7. 将用户编号与对应的 SVD 特征正确对齐；
8. 将本文件产生的所有 processed 输出保存到以当前 Python 文件名命名的目录中。

输入文件名：
- data/processed/build_feature_table/user_behavior_cleaned.parquet
  由 build_feature_table.py 生成的清洗后用户行为 Parquet 明细数据。

输入字段要求：
- user_id：用户编号；
- item_id：商品编号；
- behavior_type：行为类型；
- time：行为发生时间。

输出目录：
- data/processed/04_implicit_features/

输出文件名：
- data/processed/04_implicit_features/user_svd_features.parquet

输出文件主要字段：
- user_id：用户编号；
- user_feature_0 至 user_feature_N：SVD提取的用户隐式语义特征。
  默认最多生成20维；当用户数或商品数不足时，会自动降低维度。

目录规则：
- 本文件的输出属于 data/processed；
- 程序会自动创建：
  data/processed/04_implicit_features/
- 本文件产生的所有 processed 文件均保存在该目录中。

路径检查：
- 当前文件应位于：项目根目录/src/features/；
- 项目根目录通过 Path(__file__).resolve().parents[2] 自动定位；
- 输入文件读取路径：
  项目根目录/data/processed/build_feature_table/user_behavior_cleaned.parquet；
- 输出文件保存路径：
  项目根目录/data/processed/04_implicit_features/user_svd_features.parquet。
"""

from pathlib import Path

import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD


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
PROCESSED_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "04_implicit_features"
)
PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 明确输出文件
USER_SVD_OUTPUT = (
    PROCESSED_OUTPUT_DIR / "user_svd_features.parquet"
)

print("项目根目录：", BASE_DIR)
print("输入文件：", INPUT_FILE)
print("输出目录：", PROCESSED_OUTPUT_DIR)
print("输出文件：", USER_SVD_OUTPUT)


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

required_columns = {"user_id", "item_id", "behavior_type", "time"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "输入文件缺少以下必要字段："
        + ", ".join(sorted(missing_columns))
    )

if df.empty:
    raise ValueError(f"输入文件为空：{INPUT_FILE}")

# 统一时间格式，主要用于检查数据有效性
df["datetime"] = pd.to_datetime(df["time"], errors="coerce")

invalid_time_count = int(df["datetime"].isna().sum())
if invalid_time_count > 0:
    print(f"警告：发现 {invalid_time_count} 条无效时间记录，已删除。")
    df = df.dropna(subset=["datetime"]).copy()

# 删除隐式特征计算所需字段为空的记录
before_drop = len(df)
df = df.dropna(
    subset=["user_id", "item_id", "behavior_type"]
).copy()
dropped_count = before_drop - len(df)

if dropped_count > 0:
    print(f"警告：删除了 {dropped_count} 条关键字段缺失记录。")

if df.empty:
    raise ValueError("清理无效记录后数据为空，无法计算隐式特征。")

print("数据读取成功")
print("数据行数：", len(df))
print("字段名称：", list(df.columns))
print(df.head())


# ============================================================
# 3. 构建用户—商品交互矩阵
# ============================================================

# 行为权重：浏览 < 收藏 < 加购 < 购买
behavior_score_map = {
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 4.0
}

df["behavior_score"] = df["behavior_type"].map(
    behavior_score_map
)

invalid_behavior_count = int(df["behavior_score"].isna().sum())
if invalid_behavior_count > 0:
    print(
        f"警告：发现 {invalid_behavior_count} 条非法行为类型记录，已删除。"
    )
    df = df.dropna(subset=["behavior_score"]).copy()

if df.empty:
    raise ValueError("没有合法行为类型数据，无法构建交互矩阵。")

# 使用明确的类别顺序，保证编码与最终ID映射完全一致
user_category = pd.Categorical(df["user_id"])
item_category = pd.Categorical(df["item_id"])

user_codes = user_category.codes
item_codes = item_category.codes

user_ids = pd.Index(user_category.categories)
item_ids = pd.Index(item_category.categories)

n_users = len(user_ids)
n_items = len(item_ids)

if n_users < 2 or n_items < 2:
    raise ValueError(
        "用户数或商品数少于2，无法执行TruncatedSVD。"
    )

interaction_matrix = coo_matrix(
    (
        df["behavior_score"].astype(float),
        (user_codes, item_codes)
    ),
    shape=(n_users, n_items)
).tocsr()

# 同一用户—商品对如果存在多次行为，稀疏矩阵会自动累加
interaction_matrix.sum_duplicates()

print("用户数量：", n_users)
print("商品数量：", n_items)
print("交互矩阵形状：", interaction_matrix.shape)
print("非零交互数量：", interaction_matrix.nnz)


# ============================================================
# 4. SVD矩阵分解
# ============================================================

# TruncatedSVD维度必须小于矩阵较小维度
max_components = min(
    20,
    n_users - 1,
    n_items - 1
)

if max_components < 1:
    raise ValueError(
        "当前用户—商品矩阵维度不足，无法生成SVD隐式特征。"
    )

svd = TruncatedSVD(
    n_components=max_components,
    random_state=42
)

user_svd_matrix = svd.fit_transform(
    interaction_matrix
)

feature_columns = [
    f"user_feature_{i}"
    for i in range(max_components)
]

user_svd_features = pd.DataFrame(
    user_svd_matrix,
    columns=feature_columns
)

# 使用类别编码时的categories顺序恢复user_id，避免ID与特征错位
user_svd_features.insert(
    0,
    "user_id",
    user_ids.to_numpy()
)

print("实际SVD维度：", max_components)
print(
    "累计解释方差比例：",
    float(svd.explained_variance_ratio_.sum())
)
print("用户SVD隐式特征示例：")
print(user_svd_features.head())


# ============================================================
# 5. 保存输出文件
# ============================================================

user_svd_features.to_parquet(
    USER_SVD_OUTPUT,
    index=False
)

if not USER_SVD_OUTPUT.exists():
    raise RuntimeError(
        f"输出文件保存失败：{USER_SVD_OUTPUT}"
    )

print("用户SVD隐式特征构建完成")
print("输出数据行数：", len(user_svd_features))
print("输出数据列数：", len(user_svd_features.columns))
print("用户SVD隐式特征已保存：", USER_SVD_OUTPUT)
