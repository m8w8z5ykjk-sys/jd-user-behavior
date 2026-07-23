"""
文件名称：07_build_model_dataset.py

功能：
1. 读取特征预处理后的用户特征宽表；
2. 读取清洗后的用户行为明细数据；
3. 使用时间窗口构建“未来7天是否购买”的目标变量；
4. 按历史窗口过滤用户，降低数据泄露风险；
5. 使用欠采样构建类别平衡数据集；
6. 对类别型特征执行Target Encoding；
7. 对数值特征进行标准化；
8. 计算特征与目标变量的相关性；
9. 使用随机森林评估特征重要性；
10. 选取重要特征构建最终建模数据集；
11. 按7:2:1比例分层划分训练集、验证集和测试集；
12. 生成SMOTE过采样训练集；
13. 计算类别权重；
14. 将processed输出和results输出分别保存到以当前Python文件名命名的目录。

输入文件名：
1. data/processed/06_feature_preprocessing/feature_table_cleaned.parquet
   经06_feature_preprocessing.py处理后的用户特征宽表。

2. data/processed/build_feature_table/user_behavior_cleaned.parquet
   由build_feature_table.py生成的清洗后用户行为明细数据。

processed输出目录：
- data/processed/07_build_model_dataset/

processed输出文件名：
1. data/processed/07_build_model_dataset/model_dataset_balanced.parquet
2. data/processed/07_build_model_dataset/model_dataset_final.parquet
3. data/processed/07_build_model_dataset/train_dataset.parquet
4. data/processed/07_build_model_dataset/valid_dataset.parquet
5. data/processed/07_build_model_dataset/test_dataset.parquet
6. data/processed/07_build_model_dataset/train_dataset_smote.parquet

results输出目录：
- results/07_build_model_dataset/

results输出文件名：
1. results/07_build_model_dataset/feature_importance.csv
2. results/07_build_model_dataset/feature_label_correlation.csv
3. results/07_build_model_dataset/class_weights.csv
4. results/07_build_model_dataset/dataset_summary.txt

目录规则：
- Parquet等建模中间数据保存到：
  data/processed/07_build_model_dataset/
- CSV、TXT等分析报告保存到：
  results/07_build_model_dataset/
- 两个目录均由程序自动创建。

路径检查：
- 当前文件应位于：项目根目录/src/features/；
- 项目根目录通过Path(__file__).resolve().parents[2]自动定位。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_TABLE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "06_feature_preprocessing"
    / "feature_table_cleaned.parquet"
)

USER_BEHAVIOR_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "build_feature_table"
    / "user_behavior_cleaned.parquet"
)

PROCESSED_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "07_build_model_dataset"
)

RESULTS_OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "07_build_model_dataset"
)

PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DATASET_BALANCED_OUTPUT = (
    PROCESSED_OUTPUT_DIR / "model_dataset_balanced.parquet"
)
MODEL_DATASET_FINAL_OUTPUT = (
    PROCESSED_OUTPUT_DIR / "model_dataset_final.parquet"
)
TRAIN_OUTPUT = PROCESSED_OUTPUT_DIR / "train_dataset.parquet"
VALID_OUTPUT = PROCESSED_OUTPUT_DIR / "valid_dataset.parquet"
TEST_OUTPUT = PROCESSED_OUTPUT_DIR / "test_dataset.parquet"
SMOTE_OUTPUT = PROCESSED_OUTPUT_DIR / "train_dataset_smote.parquet"

IMPORTANCE_OUTPUT = RESULTS_OUTPUT_DIR / "feature_importance.csv"
CORRELATION_OUTPUT = RESULTS_OUTPUT_DIR / "feature_label_correlation.csv"
CLASS_WEIGHTS_OUTPUT = RESULTS_OUTPUT_DIR / "class_weights.csv"
SUMMARY_OUTPUT = RESULTS_OUTPUT_DIR / "dataset_summary.txt"

print("项目根目录：", BASE_DIR)
print("特征宽表输入：", FEATURE_TABLE_FILE)
print("用户行为输入：", USER_BEHAVIOR_FILE)
print("processed输出目录：", PROCESSED_OUTPUT_DIR)
print("results输出目录：", RESULTS_OUTPUT_DIR)


# ============================================================
# 2. 输入文件检查
# ============================================================

missing_files = [
    str(path)
    for path in [FEATURE_TABLE_FILE, USER_BEHAVIOR_FILE]
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "以下输入文件不存在：\n"
        + "\n".join(missing_files)
        + "\n请先运行06_feature_preprocessing.py和build_feature_table.py。"
    )

feature_table = pd.read_parquet(FEATURE_TABLE_FILE)
df = pd.read_parquet(USER_BEHAVIOR_FILE)

if feature_table.empty:
    raise ValueError("特征宽表为空，无法构建模型数据集。")

if df.empty:
    raise ValueError("用户行为数据为空，无法构建购买标签。")

required_feature_columns = {"user_id"}
required_behavior_columns = {"user_id", "behavior_type", "time"}

missing_feature_columns = required_feature_columns - set(feature_table.columns)
missing_behavior_columns = required_behavior_columns - set(df.columns)

if missing_feature_columns:
    raise ValueError(
        "特征宽表缺少字段："
        + ", ".join(sorted(missing_feature_columns))
    )

if missing_behavior_columns:
    raise ValueError(
        "用户行为数据缺少字段："
        + ", ".join(sorted(missing_behavior_columns))
    )


# ============================================================
# 3. 构建未来7天购买标签
# ============================================================

df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["datetime"]).copy()
df["date"] = df["datetime"].dt.normalize()

if df.empty:
    raise ValueError("时间字段清洗后数据为空。")

max_date = df["date"].max()
feature_end_date = max_date - pd.Timedelta(days=7)

label_df = df[df["date"] > feature_end_date].copy()

future_buy_users = (
    label_df.loc[
        label_df["behavior_type"] == 4,
        "user_id"
    ]
    .drop_duplicates()
)

feature_table["label"] = (
    feature_table["user_id"]
    .isin(future_buy_users)
    .astype("int8")
)

print("数据最后一天：", max_date)
print("特征截止日期：", feature_end_date)
print("未来7天购买用户数：", int(feature_table["label"].sum()))
print("标签分布：")
print(feature_table["label"].value_counts())


# ============================================================
# 4. 按历史窗口过滤用户
# ============================================================

history_users = (
    df.loc[df["date"] <= feature_end_date, "user_id"]
    .drop_duplicates()
)

model_dataset = feature_table[
    feature_table["user_id"].isin(history_users)
].copy()

if model_dataset.empty:
    raise ValueError("时间窗口过滤后没有可用样本。")

print("时间窗口过滤后的数据形状：", model_dataset.shape)
print(model_dataset["label"].value_counts())


# ============================================================
# 5. 欠采样构建平衡数据集
# ============================================================

positive_samples = model_dataset[model_dataset["label"] == 1]
negative_samples = model_dataset[model_dataset["label"] == 0]

if positive_samples.empty or negative_samples.empty:
    raise ValueError(
        "正样本或负样本为空，无法执行类别平衡处理。"
    )

sample_size = min(len(positive_samples), len(negative_samples))

positive_sampled = positive_samples.sample(
    n=sample_size,
    random_state=42
)

negative_sampled = negative_samples.sample(
    n=sample_size,
    random_state=42
)

balanced_dataset = (
    pd.concat([positive_sampled, negative_sampled], axis=0)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

balanced_dataset.to_parquet(
    MODEL_DATASET_BALANCED_OUTPUT,
    index=False
)

print("欠采样后标签分布：")
print(balanced_dataset["label"].value_counts())


# ============================================================
# 6. 类别编码与数值特征标准化
# ============================================================

X = balanced_dataset.drop(columns=["user_id", "label"]).copy()
y = balanced_dataset["label"].astype(int).copy()

category_cols = (
    X.select_dtypes(include=["object", "category", "string"])
    .columns
    .tolist()
)

print("类别型特征：", category_cols)

for col in category_cols:
    temp = balanced_dataset[[col, "label"]].copy()
    target_mean = temp.groupby(col, dropna=False)["label"].mean()
    X[f"{col}_target_encoded"] = X[col].map(target_mean)

X = X.drop(columns=category_cols, errors="ignore")

svd_cols = [
    col
    for col in X.columns
    if "svd" in col.lower() or "feature_" in col.lower()
]

print("稀疏预嵌入特征数量：", len(svd_cols))
print("部分稀疏预嵌入特征：", svd_cols[:20])

X = X.select_dtypes(include="number").copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

if X.empty:
    raise ValueError("没有可用于建模的数值特征。")

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled_array,
    columns=X.columns,
    index=X.index
)

print("标准化完成：")
print(X_scaled.head())


# ============================================================
# 7. 特征相关性与重要性
# ============================================================

corr_data = X_scaled.copy()
corr_data["label"] = y.to_numpy()

label_corr = (
    corr_data.corr(numeric_only=True)["label"]
    .drop("label")
    .abs()
    .sort_values(ascending=False)
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_scaled, y)

feature_importance = pd.DataFrame(
    {
        "feature": X_scaled.columns,
        "importance": rf.feature_importances_
    }
).sort_values(
    by="importance",
    ascending=False
)

feature_importance.to_csv(
    IMPORTANCE_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

label_corr.rename(
    "correlation_with_label"
).to_csv(
    CORRELATION_OUTPUT,
    encoding="utf-8-sig"
)

print("特征重要性前20：")
print(feature_importance.head(20))


# ============================================================
# 8. 构建最终建模数据集
# ============================================================

top_n = min(30, len(feature_importance))
top_features = (
    feature_importance.head(top_n)["feature"].tolist()
)

final_dataset = X_scaled[top_features].copy()
final_dataset["label"] = y.to_numpy()

final_dataset.to_parquet(
    MODEL_DATASET_FINAL_OUTPUT,
    index=False
)

print("最终建模数据集已保存：", MODEL_DATASET_FINAL_OUTPUT)


# ============================================================
# 9. 分层划分训练集、验证集、测试集
# ============================================================

X_final = final_dataset.drop(columns=["label"])
y_final = final_dataset["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X_final,
    y_final,
    test_size=0.30,
    random_state=42,
    stratify=y_final
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=1 / 3,
    random_state=42,
    stratify=y_temp
)

train_dataset = X_train.copy()
train_dataset["label"] = y_train.to_numpy()

valid_dataset = X_valid.copy()
valid_dataset["label"] = y_valid.to_numpy()

test_dataset = X_test.copy()
test_dataset["label"] = y_test.to_numpy()

train_dataset.to_parquet(TRAIN_OUTPUT, index=False)
valid_dataset.to_parquet(VALID_OUTPUT, index=False)
test_dataset.to_parquet(TEST_OUTPUT, index=False)

print("训练集大小：", train_dataset.shape)
print("验证集大小：", valid_dataset.shape)
print("测试集大小：", test_dataset.shape)


# ============================================================
# 10. SMOTE过采样与类别权重
# ============================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

train_smote_dataset = pd.DataFrame(
    X_train_smote,
    columns=X_train.columns
)

train_smote_dataset["label"] = y_train_smote
train_smote_dataset.to_parquet(SMOTE_OUTPUT, index=False)

classes = np.array(sorted(y_train.unique()))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_df = pd.DataFrame(
    {
        "class": classes,
        "class_weight": class_weights
    }
)

class_weight_df.to_csv(
    CLASS_WEIGHTS_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("SMOTE训练集标签分布：")
print(train_smote_dataset["label"].value_counts())

print("类别权重：")
print(class_weight_df)


# ============================================================
# 11. 保存数据集摘要
# ============================================================

summary_text = f"""模型数据集构建摘要

原始特征用户数：{len(feature_table)}
时间窗口过滤后样本数：{len(model_dataset)}
正样本数：{len(positive_samples)}
负样本数：{len(negative_samples)}
欠采样后样本数：{len(balanced_dataset)}
最终特征数量：{len(top_features)}

训练集样本数：{len(train_dataset)}
验证集样本数：{len(valid_dataset)}
测试集样本数：{len(test_dataset)}
SMOTE训练集样本数：{len(train_smote_dataset)}

特征截止日期：{feature_end_date}
标签窗口结束日期：{max_date}
"""

SUMMARY_OUTPUT.write_text(
    summary_text,
    encoding="utf-8"
)


# ============================================================
# 12. 输出检查
# ============================================================

expected_outputs = [
    MODEL_DATASET_BALANCED_OUTPUT,
    MODEL_DATASET_FINAL_OUTPUT,
    TRAIN_OUTPUT,
    VALID_OUTPUT,
    TEST_OUTPUT,
    SMOTE_OUTPUT,
    IMPORTANCE_OUTPUT,
    CORRELATION_OUTPUT,
    CLASS_WEIGHTS_OUTPUT,
    SUMMARY_OUTPUT,
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
print("07_build_model_dataset.py运行完成")
print("processed输出目录：", PROCESSED_OUTPUT_DIR)
print("results输出目录：", RESULTS_OUTPUT_DIR)
