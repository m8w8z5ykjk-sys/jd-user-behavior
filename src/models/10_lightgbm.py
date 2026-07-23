"""
文件名称：10_lightgbm.py

功能：
1. 读取07_build_model_dataset.py生成的训练集、验证集和测试集；
2. 训练LightGBM基础模型；
3. 使用Optuna搜索最佳超参数；
4. 计算Accuracy、Precision、Recall、F1和ROC-AUC；
5. 保存模型评价指标、最佳参数、调参记录、测试集预测和混淆矩阵；
6. 保存最优LightGBM模型及训练集缺失值填充中位数；
7. 将processed输出和results输出分别保存到以当前Python文件名命名的目录。

输入文件名：
1. data/processed/07_build_model_dataset/train_dataset.parquet
2. data/processed/07_build_model_dataset/valid_dataset.parquet
3. data/processed/07_build_model_dataset/test_dataset.parquet

processed输出目录：
- data/processed/10_lightgbm/

processed输出文件名：
1. best_lightgbm_optuna.pkl
2. lightgbm_feature_medians.pkl

results输出目录：
- results/10_lightgbm/

results输出文件名：
1. lightgbm_metrics.csv
2. lightgbm_optuna_best_params.csv
3. lightgbm_optuna_trials.csv
4. lightgbm_test_predictions.csv
5. lightgbm_confusion_matrix.csv

目录规则：
- PKL等模型和中间对象保存到：
  data/processed/10_lightgbm/
- CSV等分析结果保存到：
  results/10_lightgbm/
- 两个目录均由程序自动创建。
"""


from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ==================================================
# 1. 设置项目路径
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "07_build_model_dataset"
)

PROCESSED_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "10_lightgbm"
)

RESULTS_OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "10_lightgbm"
)

PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. 设置输入数据路径
# ==================================================

TRAIN_PATH = PROCESSED_INPUT_DIR / "train_dataset.parquet"
VALID_PATH = PROCESSED_INPUT_DIR / "valid_dataset.parquet"
TEST_PATH = PROCESSED_INPUT_DIR / "test_dataset.parquet"

print("项目根目录：", BASE_DIR)
print("训练集：", TRAIN_PATH)
print("验证集：", VALID_PATH)
print("测试集：", TEST_PATH)
print("processed输出目录：", PROCESSED_OUTPUT_DIR)
print("results输出目录：", RESULTS_OUTPUT_DIR)


# ==================================================
# 3. 检查数据文件是否存在
# ==================================================

required_files = [TRAIN_PATH, VALID_PATH, TEST_PATH]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"没有找到数据文件：{file_path}\n"
            "请先运行建模数据集构建代码，生成训练集、验证集和测试集。"
        )


# ==================================================
# 4. 读取训练集、验证集和测试集
# ==================================================

print("=" * 60)
print("开始读取训练集、验证集和测试集")
print("=" * 60)

train_df = pd.read_parquet(TRAIN_PATH)
valid_df = pd.read_parquet(VALID_PATH)
test_df = pd.read_parquet(TEST_PATH)

print("训练集形状：", train_df.shape)
print("验证集形状：", valid_df.shape)
print("测试集形状：", test_df.shape)


# ==================================================
# 5. 检查label列是否存在
# ==================================================

datasets = {
    "训练集": train_df,
    "验证集": valid_df,
    "测试集": test_df
}

for dataset_name, dataset in datasets.items():
    if "label" not in dataset.columns:
        raise ValueError(
            f"{dataset_name}中不存在label列，无法进行模型训练。"
        )


# ==================================================
# 6. 拆分特征X和目标变量y
# ==================================================

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"].astype(int)

X_valid = valid_df.drop(columns=["label"])
y_valid = valid_df["label"].astype(int)

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"].astype(int)


# ==================================================
# 7. 检查三个数据集的特征是否一致
# ==================================================

if list(X_train.columns) != list(X_valid.columns):
    raise ValueError("训练集和验证集的特征列不一致。")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("训练集和测试集的特征列不一致。")


# ==================================================
# 8. 处理无穷值和缺失值
# ==================================================

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_valid = X_valid.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

train_medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(train_medians)
X_valid = X_valid.fillna(train_medians)
X_test = X_test.fillna(train_medians)

if X_train.isnull().sum().sum() > 0:
    raise ValueError("训练集仍然存在未处理的缺失值。")

if X_valid.isnull().sum().sum() > 0:
    raise ValueError("验证集仍然存在未处理的缺失值。")

if X_test.isnull().sum().sum() > 0:
    raise ValueError("测试集仍然存在未处理的缺失值。")


# ==================================================
# 9. 计算类别不平衡权重
# ==================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0:
    raise ValueError("训练集中没有正样本label=1，无法训练二分类模型。")

base_scale_pos_weight = negative_count / positive_count

print("\n训练集标签数量：")
print(y_train.value_counts())

print("\n训练集标签比例：")
print(y_train.value_counts(normalize=True))

print("\n类别不平衡基础权重：")
print(base_scale_pos_weight)


# ==================================================
# 10. 定义模型评价函数
# ==================================================

def evaluate_model(model_name, model, X_data, y_true, dataset_name):
    """
    对LightGBM模型进行统一评价。

    返回：
    metrics：评价指标字典。
    y_pred：预测类别。
    y_prob：正类预测概率。
    """

    y_pred = model.predict(X_data)
    y_prob = model.predict_proba(X_data)[:, 1]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print("\n" + "=" * 60)
    print(f"{model_name}——{dataset_name}评价结果")
    print("=" * 60)
    print(f"Accuracy ：{accuracy:.6f}")
    print(f"Precision：{precision:.6f}")
    print(f"Recall   ：{recall:.6f}")
    print(f"F1 Score ：{f1:.6f}")
    print(f"ROC-AUC  ：{auc:.6f}")

    print("\n分类报告：")
    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    metrics = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    }

    return metrics, y_pred, y_prob


# ==================================================
# 11. 建立并训练LightGBM基础模型
# ==================================================

print("\n" + "=" * 60)
print("开始训练LightGBM基础模型")
print("=" * 60)

baseline_model = LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    scale_pos_weight=base_scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

baseline_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc"
)

print("LightGBM基础模型训练完成。")


# ==================================================
# 12. 在验证集上评价基础模型
# ==================================================

baseline_metrics, baseline_pred, baseline_prob = evaluate_model(
    model_name="LightGBM Baseline",
    model=baseline_model,
    X_data=X_valid,
    y_true=y_valid,
    dataset_name="Validation"
)


# ==================================================
# 13. 定义Optuna目标函数
# ==================================================

def objective(trial):
    """
    每次trial尝试一组LightGBM参数，
    以验证集ROC-AUC作为优化目标。
    """

    params = {
        "objective": "binary",
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            800
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.2,
            log=True
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves",
            16,
            128
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            12
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            5,
            100
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            20.0,
            log=True
        ),
        "min_split_gain": trial.suggest_float(
            "min_split_gain",
            0.0,
            5.0
        ),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            max(1.0, base_scale_pos_weight * 0.5),
            max(1.1, base_scale_pos_weight * 1.5)
        ),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }

    model = LGBMClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc"
    )

    valid_prob = model.predict_proba(X_valid)[:, 1]
    valid_auc = roc_auc_score(y_valid, valid_prob)

    return valid_auc


# ==================================================
# 14. 创建并执行Optuna研究
# ==================================================

print("\n" + "=" * 60)
print("开始进行LightGBM Optuna自动调参")
print("核心优化指标：验证集ROC-AUC")
print("=" * 60)

study = optuna.create_study(
    direction="maximize",
    study_name="lightgbm_auc_optimization",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)


# ==================================================
# 15. 输出Optuna最佳结果
# ==================================================

print("\n" + "=" * 60)
print("LightGBM Optuna调参完成")
print("=" * 60)

print("\n最佳验证集AUC：")
print(study.best_value)

print("\n最佳参数：")
print(study.best_params)


# ==================================================
# 16. 使用最佳参数重新训练模型
# ==================================================

best_model = LGBMClassifier(
    objective="binary",
    **study.best_params,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    subsample_freq=1
)

best_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc"
)

print("\n最优LightGBM模型训练完成。")


# ==================================================
# 17. 评价优化模型
# ==================================================

tuned_valid_metrics, tuned_valid_pred, tuned_valid_prob = evaluate_model(
    model_name="LightGBM Optuna",
    model=best_model,
    X_data=X_valid,
    y_true=y_valid,
    dataset_name="Validation"
)

tuned_test_metrics, test_pred, test_prob = evaluate_model(
    model_name="LightGBM Optuna",
    model=best_model,
    X_data=X_test,
    y_true=y_test,
    dataset_name="Test"
)


# ==================================================
# 18. 保存模型评价指标
# ==================================================

metrics_df = pd.DataFrame([
    baseline_metrics,
    tuned_valid_metrics,
    tuned_test_metrics
])

metrics_path = RESULTS_OUTPUT_DIR / "lightgbm_metrics.csv"

metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 19. 保存Optuna最佳参数
# ==================================================

best_params_row = {
    "Model": "LightGBM",
    "Best_Validation_AUC": study.best_value,
    **study.best_params
}

best_params_df = pd.DataFrame([best_params_row])

best_params_path = (
    RESULTS_OUTPUT_DIR /
    "lightgbm_optuna_best_params.csv"
)

best_params_df.to_csv(
    best_params_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 20. 保存Optuna全部试验记录
# ==================================================

trials_df = study.trials_dataframe()
trials_path = RESULTS_OUTPUT_DIR / "lightgbm_optuna_trials.csv"

trials_df.to_csv(
    trials_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 21. 保存测试集预测结果
# ==================================================

prediction_df = pd.DataFrame({
    "true_label": y_test.reset_index(drop=True),
    "predicted_label": test_pred,
    "purchase_probability": test_prob
})

prediction_path = (
    RESULTS_OUTPUT_DIR /
    "lightgbm_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 22. 保存测试集混淆矩阵
# ==================================================

test_confusion_matrix = confusion_matrix(
    y_test,
    test_pred
)

confusion_matrix_df = pd.DataFrame(
    test_confusion_matrix,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)

confusion_matrix_path = (
    RESULTS_OUTPUT_DIR /
    "lightgbm_confusion_matrix.csv"
)

confusion_matrix_df.to_csv(
    confusion_matrix_path,
    encoding="utf-8-sig"
)


# ==================================================
# 23. 保存最优模型和中位数
# ==================================================

model_path = (
    PROCESSED_OUTPUT_DIR /
    "best_lightgbm_optuna.pkl"
)

median_path = (
    PROCESSED_OUTPUT_DIR /
    "lightgbm_feature_medians.pkl"
)

joblib.dump(best_model, model_path)
joblib.dump(train_medians, median_path)


# ==================================================
# 24. 检查输出文件
# ==================================================

expected_outputs = [
    metrics_path,
    best_params_path,
    trials_path,
    prediction_path,
    confusion_matrix_path,
    model_path,
    median_path
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


# ==================================================
# 25. 输出最终总结
# ==================================================

print("\n" + "=" * 60)
print("LightGBM基础训练和Optuna调参全部完成")
print("=" * 60)

print("\n最佳参数：")
print(study.best_params)

print("\n最佳验证集AUC：")
print(study.best_value)

print("\n测试集最终结果：")
print(
    metrics_df[
        metrics_df["Dataset"] == "Test"
    ].to_string(index=False)
)

print("\n结果文件已保存：")
print(metrics_path)
print(best_params_path)
print(trials_path)
print(prediction_path)
print(confusion_matrix_path)
print(model_path)
print(median_path)