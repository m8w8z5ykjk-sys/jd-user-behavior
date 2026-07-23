"""
文件名称：06_feature_preprocessing.py

功能：
1. 读取时间特征、行为序列特征、业务特征和SVD隐式特征；
2. 按 user_id 合并生成用户级特征宽表；
3. 检查重复用户、缺失率、字段唯一值数量、低区分度字段和重复列名；
4. 删除只有一个唯一值的低区分度字段；
5. 对数值型字段使用0填充缺失值；
6. 对字符型字段使用 unknown 填充缺失值；
7. 保存合并前的原始特征宽表和清洗后的特征宽表；
8. 保存缺失率、字段唯一值和质量检查摘要；
9. 将 processed 输出和 results 输出分别保存到以当前Python文件名命名的目录。

输入文件名：
1. data/processed/01_time_features/time_features.parquet
   时间维度用户特征。

2. data/processed/03_behavior_sequence_features/behavior_sequence_features.parquet
   用户行为序列特征。

3. data/processed/05_business_features/business_features.parquet
   用户业务导向特征。

4. data/processed/04_implicit_features/user_svd_features.parquet
   用户SVD隐式特征。

processed输出目录：
- data/processed/06_feature_preprocessing/

processed输出文件名：
1. data/processed/06_feature_preprocessing/feature_table.parquet
   四类特征合并后的原始特征宽表。

2. data/processed/06_feature_preprocessing/feature_table_cleaned.parquet
   删除低区分度特征并完成缺失值处理后的特征宽表。

results输出目录：
- results/06_feature_preprocessing/

results输出文件名：
1. results/06_feature_preprocessing/missing_rate.csv
   各字段缺失率统计。

2. results/06_feature_preprocessing/unique_count.csv
   各字段唯一值数量统计。

3. results/06_feature_preprocessing/feature_quality_summary.txt
   特征宽表质量检查摘要。

目录规则：
- Parquet等中间数据保存到：
  data/processed/06_feature_preprocessing/
- CSV、TXT等分析结果保存到：
  results/06_feature_preprocessing/
- 两个目录均由程序自动创建。

路径检查：
- 当前文件应位于：项目根目录/src/features/；
- 项目根目录通过 Path(__file__).resolve().parents[2] 自动定位。
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# 输入文件
TIME_FEATURES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "01_time_features"
    / "time_features.parquet"
)

BEHAVIOR_SEQUENCE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "03_behavior_sequence_features"
    / "behavior_sequence_features.parquet"
)

BUSINESS_FEATURES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "05_business_features"
    / "business_features.parquet"
)

USER_SVD_FEATURES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "04_implicit_features"
    / "user_svd_features.parquet"
)

# 本文件专属输出目录
PROCESSED_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "06_feature_preprocessing"
)

RESULTS_OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "06_feature_preprocessing"
)

PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 明确输出文件
FEATURE_TABLE_OUTPUT = (
    PROCESSED_OUTPUT_DIR / "feature_table.parquet"
)

FEATURE_TABLE_CLEANED_OUTPUT = (
    PROCESSED_OUTPUT_DIR / "feature_table_cleaned.parquet"
)

MISSING_RATE_OUTPUT = (
    RESULTS_OUTPUT_DIR / "missing_rate.csv"
)

UNIQUE_COUNT_OUTPUT = (
    RESULTS_OUTPUT_DIR / "unique_count.csv"
)

QUALITY_SUMMARY_OUTPUT = (
    RESULTS_OUTPUT_DIR / "feature_quality_summary.txt"
)

print("项目根目录：", BASE_DIR)
print("processed输出目录：", PROCESSED_OUTPUT_DIR)
print("results输出目录：", RESULTS_OUTPUT_DIR)


# ============================================================
# 2. 输入文件检查
# ============================================================

input_files = {
    "时间特征": TIME_FEATURES_FILE,
    "行为序列特征": BEHAVIOR_SEQUENCE_FILE,
    "业务特征": BUSINESS_FEATURES_FILE,
    "SVD隐式特征": USER_SVD_FEATURES_FILE,
}

missing_files = [
    f"{name}: {path}"
    for name, path in input_files.items()
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "以下输入文件不存在：\n"
        + "\n".join(missing_files)
        + "\n请先运行对应的特征工程脚本。"
    )


# ============================================================
# 3. 读取特征文件
# ============================================================

time_features = pd.read_parquet(TIME_FEATURES_FILE)
behavior_sequence_features = pd.read_parquet(
    BEHAVIOR_SEQUENCE_FILE
)
business_features = pd.read_parquet(BUSINESS_FEATURES_FILE)
user_svd_features = pd.read_parquet(USER_SVD_FEATURES_FILE)

feature_frames = {
    "时间特征": time_features,
    "行为序列特征": behavior_sequence_features,
    "业务特征": business_features,
    "SVD隐式特征": user_svd_features,
}

for name, frame in feature_frames.items():
    if frame.empty:
        raise ValueError(f"{name}文件为空，无法继续处理。")

    if "user_id" not in frame.columns:
        raise ValueError(
            f"{name}缺少必要字段 user_id。"
        )

    print(f"{name}形状：", frame.shape)


# ============================================================
# 4. 合并用户特征宽表
# ============================================================

feature_table = time_features.copy()

feature_table = feature_table.merge(
    behavior_sequence_features,
    on="user_id",
    how="left"
)

feature_table = feature_table.merge(
    business_features,
    on="user_id",
    how="left"
)

feature_table = feature_table.merge(
    user_svd_features,
    on="user_id",
    how="left"
)

feature_table.to_parquet(
    FEATURE_TABLE_OUTPUT,
    index=False
)

print("原始特征宽表已保存：", FEATURE_TABLE_OUTPUT)
print("原始特征宽表形状：", feature_table.shape)


# ============================================================
# 5. 宽表质量检查
# ============================================================

duplicate_users = int(
    feature_table["user_id"].duplicated().sum()
)

missing_rate = (
    feature_table.isnull()
    .mean()
    .sort_values(ascending=False)
)

unique_count = (
    feature_table.nunique(dropna=False)
    .sort_values(ascending=True)
)

low_variance_features = (
    unique_count[unique_count <= 1]
    .index
    .tolist()
)

duplicate_columns_count = int(
    feature_table.columns.duplicated().sum()
)

print("所有字段：")
print(feature_table.columns.tolist())

print("重复用户数：", duplicate_users)

print("缺失率最高的前20个字段：")
print(missing_rate.head(20))

print("低区分度特征：")
print(low_variance_features)

print("重复列名数量：", duplicate_columns_count)

# 保存质量检查结果
missing_rate.rename("missing_rate").to_csv(
    MISSING_RATE_OUTPUT,
    encoding="utf-8-sig"
)

unique_count.rename("unique_count").to_csv(
    UNIQUE_COUNT_OUTPUT,
    encoding="utf-8-sig"
)


# ============================================================
# 6. 删除低区分度特征并处理缺失值
# ============================================================

feature_table_cleaned = feature_table.drop(
    columns=low_variance_features,
    errors="ignore"
).copy()

numeric_cols = (
    feature_table_cleaned
    .select_dtypes(include="number")
    .columns
    .tolist()
)

for col in numeric_cols:
    if col != "user_id":
        feature_table_cleaned[col] = (
            feature_table_cleaned[col].fillna(0)
        )

object_cols = (
    feature_table_cleaned
    .select_dtypes(include=["object", "string", "category"])
    .columns
    .tolist()
)

for col in object_cols:
    feature_table_cleaned[col] = (
        feature_table_cleaned[col]
        .astype("object")
        .fillna("unknown")
    )

feature_table_cleaned.to_parquet(
    FEATURE_TABLE_CLEANED_OUTPUT,
    index=False
)


# ============================================================
# 7. 保存质量摘要
# ============================================================

summary_text = f"""特征预处理质量检查摘要

输入特征表数量：4
原始特征宽表行数：{len(feature_table)}
原始特征宽表列数：{len(feature_table.columns)}
重复用户数：{duplicate_users}
重复列名数量：{duplicate_columns_count}
低区分度特征数量：{len(low_variance_features)}
低区分度特征：{low_variance_features}
清洗后特征宽表行数：{len(feature_table_cleaned)}
清洗后特征宽表列数：{len(feature_table_cleaned.columns)}
清洗后总缺失值数量：{int(feature_table_cleaned.isna().sum().sum())}

原始特征宽表：
{FEATURE_TABLE_OUTPUT}

清洗后特征宽表：
{FEATURE_TABLE_CLEANED_OUTPUT}
"""

QUALITY_SUMMARY_OUTPUT.write_text(
    summary_text,
    encoding="utf-8"
)


# ============================================================
# 8. 输出文件检查
# ============================================================

expected_outputs = [
    FEATURE_TABLE_OUTPUT,
    FEATURE_TABLE_CLEANED_OUTPUT,
    MISSING_RATE_OUTPUT,
    UNIQUE_COUNT_OUTPUT,
    QUALITY_SUMMARY_OUTPUT,
]

failed_outputs = [
    str(path)
    for path in expected_outputs
    if not path.exists()
]

if failed_outputs:
    raise RuntimeError(
        "以下输出文件保存失败：\n"
        + "\n".join(failed_outputs)
    )

print("=" * 60)
print("特征预处理完成")
print("清洗后的特征宽表形状：", feature_table_cleaned.shape)
print("清洗后的特征宽表已保存：")
print(FEATURE_TABLE_CLEANED_OUTPUT)
