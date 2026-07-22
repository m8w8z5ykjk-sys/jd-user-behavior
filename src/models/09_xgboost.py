"""
文件名称：09_xgboost.py

功能说明：
1. 读取训练集、验证集和测试集。
2. 完成XGBoost基础分类模型训练。
3. 计算基础模型的Accuracy、Precision、Recall、F1和ROC-AUC。
4. 使用Optuna自动搜索XGBoost关键参数。
5. 以验证集ROC-AUC作为核心优化指标。
6. 使用最优参数重新训练XGBoost模型。
7. 在测试集上完成最终评价。
8. 保存模型评价结果、最佳参数、调参记录、预测结果和最优模型。

输入文件：
- data/processed/train_dataset.parquet
- data/processed/valid_dataset.parquet
- data/processed/test_dataset.parquet

输出文件：
- results/xgboost_metrics.csv
- results/xgboost_optuna_best_params.csv
- results/xgboost_optuna_trials.csv
- results/xgboost_test_predictions.csv
- results/xgboost_confusion_matrix.csv
- results/models/best_xgboost_optuna.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ==================================================
# 1. 设置项目路径
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. 设置数据文件路径
# ==================================================

TRAIN_PATH = PROCESSED_DIR / "train_dataset.parquet"
VALID_PATH = PROCESSED_DIR / "valid_dataset.parquet"
TEST_PATH = PROCESSED_DIR / "test_dataset.parquet"


# ==================================================
# 3. 检查数据文件是否存在
# ==================================================

for file_path in [TRAIN_PATH, VALID_PATH, TEST_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"没有找到数据文件：{file_path}"
        )


# ==================================================
# 4. 读取数据
# ==================================================

print("=" * 60)
print("开始读取数据")
print("=" * 60)

train_df = pd.read_parquet(TRAIN_PATH)
valid_df = pd.read_parquet(VALID_PATH)
test_df = pd.read_parquet(TEST_PATH)

print("训练集形状：", train_df.shape)
print("验证集形状：", valid_df.shape)
print("测试集形状：", test_df.shape)


# ==================================================
# 5. 检查label列
# ==================================================

for dataset_name, dataset in {
    "训练集": train_df,
    "验证集": valid_df,
    "测试集": test_df
}.items():

    if "label" not in dataset.columns:
        raise ValueError(
            f"{dataset_name}中不存在label列。"
        )


# ==================================================
# 6. 拆分特征X和标签y
# ==================================================

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"].astype(int)

X_valid = valid_df.drop(columns=["label"])
y_valid = valid_df["label"].astype(int)

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"].astype(int)


# ==================================================
# 7. 检查特征列是否一致
# ==================================================

if list(X_train.columns) != list(X_valid.columns):
    raise ValueError(
        "训练集和验证集的特征列不一致。"
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "训练集和测试集的特征列不一致。"
    )


# ==================================================
# 8. 处理无穷值和缺失值
# ==================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_valid = X_valid.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

train_medians = X_train.median(
    numeric_only=True
)

X_train = X_train.fillna(train_medians)
X_valid = X_valid.fillna(train_medians)
X_test = X_test.fillna(train_medians)


# ==================================================
# 9. 计算类别不平衡权重
# ==================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0:
    raise ValueError(
        "训练集中没有正样本label=1，无法训练模型。"
    )

base_scale_pos_weight = (
    negative_count / positive_count
)

print("\n训练集标签数量：")
print(y_train.value_counts())

print("\n基础scale_pos_weight：")
print(base_scale_pos_weight)


# ==================================================
# 10. 定义模型评价函数
# ==================================================

def evaluate_model(
    model_name,
    model,
    X_data,
    y_true,
    dataset_name
):
    """
    对XGBoost模型进行评价。

    返回：
    1. 评价指标字典
    2. 预测类别
    3. 正类预测概率
    """

    y_pred = model.predict(X_data)

    y_prob = model.predict_proba(
        X_data
    )[:, 1]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    print("\n" + "=" * 60)
    print(f"{model_name}——{dataset_name}结果")
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
# 11. 建立XGBoost基础模型
# ==================================================

print("\n" + "=" * 60)
print("开始训练XGBoost基础模型")
print("=" * 60)

baseline_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=base_scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)


# ==================================================
# 12. 训练基础模型
# ==================================================

baseline_model.fit(
    X_train,
    y_train
)

print("XGBoost基础模型训练完成。")


# ==================================================
# 13. 在验证集上评价基础模型
# ==================================================

baseline_metrics, baseline_pred, baseline_prob = (
    evaluate_model(
        model_name="XGBoost Baseline",
        model=baseline_model,
        X_data=X_valid,
        y_true=y_valid,
        dataset_name="Validation"
    )
)


# ==================================================
# 14. 定义Optuna目标函数
# ==================================================

def objective(trial):
    """
    Optuna每次尝试一组XGBoost参数。

    以验证集ROC-AUC作为优化目标。
    """

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",

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

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            1.0,
            20.0
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

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

        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            max(1.0, base_scale_pos_weight * 0.5),
            max(1.1, base_scale_pos_weight * 1.5)
        ),

        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist"
    }

    model = XGBClassifier(
        **params
    )

    model.fit(
        X_train,
        y_train
    )

    valid_prob = model.predict_proba(
        X_valid
    )[:, 1]

    valid_auc = roc_auc_score(
        y_valid,
        valid_prob
    )

    return valid_auc


# ==================================================
# 15. 创建Optuna研究对象
# ==================================================

print("\n" + "=" * 60)
print("开始进行XGBoost Optuna自动调参")
print("优化目标：验证集ROC-AUC")
print("=" * 60)

study = optuna.create_study(
    direction="maximize",
    study_name="xgboost_auc_optimization",
    sampler=optuna.samplers.TPESampler(
        seed=42
    )
)


# ==================================================
# 16. 执行Optuna参数搜索
# ==================================================

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)


# ==================================================
# 17. 输出最佳结果
# ==================================================

print("\n" + "=" * 60)
print("XGBoost Optuna调参完成")
print("=" * 60)

print("最佳验证集AUC：")
print(study.best_value)

print("\n最佳参数：")
print(study.best_params)


# ==================================================
# 18. 使用最佳参数建立优化模型
# ==================================================

best_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    **study.best_params,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)


# ==================================================
# 19. 重新训练最优模型
# ==================================================

best_model.fit(
    X_train,
    y_train
)

print("\n最优XGBoost模型训练完成。")


# ==================================================
# 20. 在验证集上评价优化模型
# ==================================================

tuned_valid_metrics, tuned_valid_pred, tuned_valid_prob = (
    evaluate_model(
        model_name="XGBoost Optuna",
        model=best_model,
        X_data=X_valid,
        y_true=y_valid,
        dataset_name="Validation"
    )
)


# ==================================================
# 21. 在测试集上完成最终评价
# ==================================================

tuned_test_metrics, test_pred, test_prob = (
    evaluate_model(
        model_name="XGBoost Optuna",
        model=best_model,
        X_data=X_test,
        y_true=y_test,
        dataset_name="Test"
    )
)


# ==================================================
# 22. 保存评价结果
# ==================================================

metrics_df = pd.DataFrame(
    [
        baseline_metrics,
        tuned_valid_metrics,
        tuned_test_metrics
    ]
)

metrics_path = (
    RESULTS_DIR /
    "xgboost_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 23. 保存Optuna最佳参数
# ==================================================

best_params_row = {
    "Model": "XGBoost",
    "Best_Validation_AUC": study.best_value
}

for key, value in study.best_params.items():
    best_params_row[key] = value

best_params_df = pd.DataFrame(
    [best_params_row]
)

best_params_path = (
    RESULTS_DIR /
    "xgboost_optuna_best_params.csv"
)

best_params_df.to_csv(
    best_params_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 24. 保存Optuna全部试验记录
# ==================================================

trials_df = study.trials_dataframe()

trials_path = (
    RESULTS_DIR /
    "xgboost_optuna_trials.csv"
)

trials_df.to_csv(
    trials_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 25. 保存测试集预测结果
# ==================================================

prediction_df = pd.DataFrame({
    "true_label": y_test.reset_index(drop=True),
    "predicted_label": test_pred,
    "purchase_probability": test_prob
})

prediction_path = (
    RESULTS_DIR /
    "xgboost_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 26. 保存混淆矩阵
# ==================================================

test_confusion_matrix = confusion_matrix(
    y_test,
    test_pred
)

confusion_matrix_df = pd.DataFrame(
    test_confusion_matrix,
    index=[
        "Actual_0",
        "Actual_1"
    ],
    columns=[
        "Predicted_0",
        "Predicted_1"
    ]
)

confusion_matrix_path = (
    RESULTS_DIR /
    "xgboost_confusion_matrix.csv"
)

confusion_matrix_df.to_csv(
    confusion_matrix_path,
    encoding="utf-8-sig"
)


# ==================================================
# 27. 保存最优模型
# ==================================================

model_path = (
    MODEL_DIR /
    "best_xgboost_optuna.pkl"
)

joblib.dump(
    best_model,
    model_path
)


# ==================================================
# 28. 保存缺失值填充中位数
# ==================================================

median_path = (
    MODEL_DIR /
    "xgboost_feature_medians.pkl"
)

joblib.dump(
    train_medians,
    median_path
)


# ==================================================
# 29. 输出最终总结
# ==================================================

print("\n" + "=" * 60)
print("XGBoost基础训练和Optuna调参全部完成")
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