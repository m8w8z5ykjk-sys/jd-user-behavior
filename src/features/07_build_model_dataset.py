import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR

feature_table = pd.read_parquet(PROCESSED_DIR / "feature_table_cleaned.parquet")
df = pd.read_parquet(PROCESSED_DIR / "user_behavior_cleaned.parquet")
df["datetime"] = pd.to_datetime(df["time"])
df["date"] = pd.to_datetime(df["datetime"].dt.date)

#用时间窗口构建未来7天购买标签
max_date = df["date"].max()
feature_end_date = max_date - pd.Timedelta(days=7)
label_df = df[df["date"] > feature_end_date]
future_buy_users = (label_df[label_df["behavior_type"] == 4]["user_id"].drop_duplicates())
feature_table["label"] = feature_table["user_id"].isin(future_buy_users).astype(int)

print("数据最后一天：", max_date)
print("特征截止日期：", feature_end_date)
print("未来7天购买用户数：", feature_table["label"].sum())
print("标签分布：")
print(feature_table["label"].value_counts())

#按时间窗口过滤特征用户，避免数据泄露
history_users = (df[df["date"] <= feature_end_date]["user_id"].drop_duplicates())
model_dataset = feature_table[feature_table["user_id"].isin(history_users)].copy()
print("时间窗口过滤后的数据：", model_dataset.shape)
print(model_dataset["label"].value_counts())

#类别不平衡处理：下采样
positive_samples = model_dataset[model_dataset["label"] == 1]
negative_samples = model_dataset[model_dataset["label"] == 0]
sample_size = min(len(positive_samples), len(negative_samples))
positive_sampled = positive_samples.sample(n=sample_size,random_state=42)
negative_sampled = negative_samples.sample(n=sample_size,random_state=42)
balanced_dataset = pd.concat([positive_sampled, negative_sampled],axis=0).sample(frac=1,random_state=42).reset_index(drop=True)
print(balanced_dataset["label"].value_counts())

MODEL_DATASET_OUTPUT = OUTPUT_DIR / "model_dataset_balanced.parquet"
balanced_dataset.to_parquet(MODEL_DATASET_OUTPUT,index=False)
print("最终建模数据集已保存：")
print(MODEL_DATASET_OUTPUT)
print(balanced_dataset.head())

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 数值特征标准化
X = balanced_dataset.drop(columns=["user_id", "label"])
y = balanced_dataset["label"]


#类别型特征 Target Encoding
category_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("类别型特征：")
print(category_cols)
for col in category_cols:
    target_mean = balanced_dataset.groupby(col)["label"].mean()
    X[col + "_target_encoded"] = X[col].map(target_mean)
X = X.drop(columns=category_cols)


#稀疏特征预嵌入处理
# 本项目中 user_svd_features 已经通过用户-商品交互矩阵生成，
# 可作为稀疏用户-商品行为的预嵌入表示。
# 这些 SVD 特征已在 feature_table_cleaned 中合并进来，
# 后续会一起进入标准化和特征重要性评估。
svd_cols = [col for col in X.columns if "svd" in col.lower() or "feature_" in col.lower()]
print("稀疏预嵌入特征：")
print(svd_cols[:20])
print("稀疏预嵌入特征数量：", len(svd_cols))

X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])
X = X.fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
print("标准化完成：")
print(X_scaled.head())

#验证特征与目标变量相关性
corr_data = X_scaled.copy()
corr_data["label"] = y.values
label_corr = (corr_data.corr(numeric_only=True)["label"].drop("label").abs().sort_values(ascending=False))
print("与目标变量相关性最高的前20个特征：")
print(label_corr.head(20))


#随机森林特征重要性
rf = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
rf.fit(X_scaled, y)
feature_importance = pd.DataFrame({"feature": X_scaled.columns,"importance": rf.feature_importances_}).sort_values(by="importance",ascending=False)
print("特征重要性前20：")
print(feature_importance.head(20))

#只保留主要特征
top_features = feature_importance.head(30)["feature"].tolist()
final_dataset = X_scaled[top_features].copy()
final_dataset["label"] = y.values


MODEL_OUTPUT = OUTPUT_DIR / "model_dataset_final.parquet"
IMPORTANCE_OUTPUT = OUTPUT_DIR / "feature_importance.csv"
CORRELATION_OUTPUT = OUTPUT_DIR / "feature_label_correlation.csv"
final_dataset.to_parquet(MODEL_OUTPUT,index=False)
feature_importance.to_csv(IMPORTANCE_OUTPUT,index=False)
label_corr.to_csv(CORRELATION_OUTPUT,header=["correlation_with_label"])

print("最终建模数据已保存：")
print(MODEL_OUTPUT)
print("特征重要性报告已保存：")
print(IMPORTANCE_OUTPUT)
print("相关性报告已保存：")
print(CORRELATION_OUTPUT)
