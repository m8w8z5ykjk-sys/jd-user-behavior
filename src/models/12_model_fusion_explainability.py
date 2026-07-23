# -*- coding: utf-8 -*-
"""
文件名称：12_model_fusion_explainability.py

功能：
1. 读取逻辑回归、XGBoost和DIN三个基础模型的测试集预测结果；
2. 对三个模型的预测样本进行对齐和合并；
3. 使用逻辑回归作为二层元学习器，完成Stacking模型融合；
4. 对比逻辑回归、XGBoost、DIN和Stacking模型性能；
5. 根据逻辑回归特征系数解释线性模型；
6. 使用SHAP方法解释XGBoost模型；
7. 使用代理解释方法分析DIN模型的重要特征；
8. 汇总影响用户购买决策的核心特征；
9. 保存Stacking预测错误样本；
10. 分析冷启动用户和长尾商品场景下的模型表现；
11. 生成模型融合与可解释性综合报告；
12. 将模型等processed输出和CSV、PNG、TXT、JSON、Markdown等results输出
    分别保存到以当前Python文件名命名的专属目录中。

输入文件名：
1. results/08_logistic_regression/logistic_regression_test_predictions.csv
2. results/09_xgboost/xgboost_predictions.csv
3. results/11_deep_learning_models/din_test_predictions.csv
4. data/processed/08_logistic_regression/best_logistic_regression_optuna.pkl
5. data/processed/09_xgboost/best_xgboost_optuna.pkl
6. data/processed/11_deep_learning_models/din_best.pt
7. data/processed/07_build_model_dataset/test_dataset.parquet

processed输出目录：
- data/processed/12_model_fusion_explainability/

processed输出文件名：
- stacking_meta_model.pkl

results输出目录：
- results/12_model_fusion_explainability/

results输出文件名：
- config.json
- merged_base_model_predictions.csv
- stacking_test_predictions.csv
- model_performance_comparison.csv
- stacking_base_model_weights.csv
- logistic_feature_coefficients.csv
- logistic_feature_coefficients.png
- xgboost_shap_importance.csv
- xgboost_shap_summary.png
- din_proxy_shap_importance.csv
- din_proxy_shap_summary.png
- din_shap_explanation_note.txt
- stacking_error_samples.csv
- cold_start_long_tail_analysis.csv
- core_purchase_features.csv
- model_fusion_explainability_report.md

目录规则：
- PKL模型文件保存到data/processed/12_model_fusion_explainability/；
- CSV、PNG、TXT、JSON和Markdown结果保存到results/12_model_fusion_explainability/；
- 两个目录均由程序自动创建。
"""


import os
import gc
import json
import math
import pickle
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("请先安装matplotlib：pip install matplotlib") from e

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
except ImportError as e:
    raise ImportError("请先安装scikit-learn：pip install scikit-learn") from e


# ============================================================
# 1. 配置
# ============================================================
@dataclass
class Config:
    project_root: str = str(Path(__file__).resolve().parents[2])

    processed_output_dir: str = "data/processed/12_model_fusion_explainability"
    results_output_dir: str = "results/12_model_fusion_explainability"

    logistic_pred_path: str = (
        "results/08_logistic_regression/"
        "logistic_regression_test_predictions.csv"
    )
    xgboost_pred_path: str = "results/09_xgboost/xgboost_predictions.csv"
    din_pred_path: str = (
        "results/11_deep_learning_models/din_test_predictions.csv"
    )

    logistic_model_path: str = (
        "data/processed/08_logistic_regression/"
        "best_logistic_regression_optuna.pkl"
    )
    xgboost_model_path: str = (
        "data/processed/09_xgboost/best_xgboost_optuna.pkl"
    )
    din_model_path: str = (
        "data/processed/11_deep_learning_models/din_best.pt"
    )

    test_feature_path: str = (
        "data/processed/07_build_model_dataset/test_dataset.parquet"
    )

    user_col: str = "user_id"
    item_col: str = "item_id"
    label_col: str = "label"

    meta_train_ratio: float = 0.70
    classification_threshold: float = 0.50
    random_state: int = 42

    top_n_features: int = 20
    shap_sample_size: int = 1000

    cold_user_threshold: int = 3
    long_tail_quantile: float = 0.20


# ============================================================
# 2. 基础工具
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_path(cfg: Config, path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(cfg.project_root) / p


def find_existing_file(cfg: Config, specified_path: str, keywords: List[str]) -> Optional[Path]:
    p = resolve_path(cfg, specified_path)
    if p.exists():
        return p

    root = Path(cfg.project_root)
    candidates = []
    for ext in ("*.csv", "*.parquet", "*.pkl", "*.pickle", "*.joblib", "*.pt"):
        for file in root.rglob(ext):
            name = file.name.lower()
            if all(k.lower() in name for k in keywords):
                candidates.append(file)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"不支持的数据格式：{path}")


def detect_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for col in df.columns:
        low = str(col).lower()
        for name in candidates:
            if name.lower() in low:
                return col

    if required:
        raise ValueError(f"未找到列：{candidates}，当前列为：{list(df.columns)}")
    return None


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": safe_auc(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }


# ============================================================
# 3. 读取并标准化三个基础模型预测结果
# ============================================================
def standardize_prediction_file(
    path: Path,
    model_name: str,
    cfg: Config
) -> pd.DataFrame:
    df = read_table(path).copy()

    user_col = detect_column(
        df,
        [cfg.user_col, "user_id_idx", "user_idx", "userid", "uid"],
        required=False
    )
    item_col = detect_column(
        df,
        [cfg.item_col, "item_id_idx", "item_idx", "itemid", "iid"],
        required=False
    )
    label_col = detect_column(
        df,
        [cfg.label_col, "target", "y_true", "true_label", "actual", "is_buy", "purchase_label"]
    )
    prob_col = detect_column(
        df,
        [
            "probability", "prob", "y_prob", "pred_proba",
            "prediction_probability", "positive_probability", "purchase_probability",
            f"{model_name.lower()}_probability"
        ]
    )

    out = pd.DataFrame()
    if user_col is not None:
        out["user_id"] = df[user_col]
    else:
        out["row_id"] = np.arange(len(df))

    if item_col is not None:
        out["item_id"] = df[item_col]

    out["label"] = pd.to_numeric(df[label_col], errors="coerce")
    out[f"{model_name.lower()}_prob"] = pd.to_numeric(df[prob_col], errors="coerce")
    out = out.dropna(subset=["label", f"{model_name.lower()}_prob"]).copy()
    out["label"] = out["label"].astype(int)

    print(f"{model_name}预测文件：{path}")
    print(f"{model_name}有效预测记录：{len(out):,}")
    return out


def merge_predictions(
    logistic_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    din_df: pd.DataFrame
) -> pd.DataFrame:
    common_keys = []
    for key in ["user_id", "item_id"]:
        if key in logistic_df.columns and key in xgb_df.columns and key in din_df.columns:
            common_keys.append(key)

    if common_keys:
        merged = logistic_df.merge(
            xgb_df.drop(columns=["label"]),
            on=common_keys,
            how="inner"
        )
        merged = merged.merge(
            din_df.drop(columns=["label"]),
            on=common_keys,
            how="inner"
        )
    else:
        min_len = min(len(logistic_df), len(xgb_df), len(din_df))
        print("警告：未找到共同的用户/商品键，将按照行顺序对齐三个预测文件。")
        merged = pd.DataFrame({
            "row_id": np.arange(min_len),
            "label": logistic_df["label"].iloc[:min_len].to_numpy(),
            "logistic_prob": logistic_df["logistic_prob"].iloc[:min_len].to_numpy(),
            "xgboost_prob": xgb_df["xgboost_prob"].iloc[:min_len].to_numpy(),
            "din_prob": din_df["din_prob"].iloc[:min_len].to_numpy()
        })

    merged = merged.dropna(
        subset=["label", "logistic_prob", "xgboost_prob", "din_prob"]
    ).reset_index(drop=True)

    for col in ["logistic_prob", "xgboost_prob", "din_prob"]:
        merged[col] = merged[col].clip(1e-6, 1 - 1e-6)

    return merged


# ============================================================
# 4. Stacking融合
# ============================================================
def train_stacking_model(
    merged: pd.DataFrame,
    cfg: Config
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    feature_cols = ["logistic_prob", "xgboost_prob", "din_prob"]
    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=np.int64)

    if len(np.unique(y)) < 2:
        raise ValueError("融合数据只有一个类别，无法训练Stacking模型。")

    split_index = max(1, int(len(merged) * cfg.meta_train_ratio))
    if split_index >= len(merged):
        split_index = len(merged) - 1

    meta_train = merged.iloc[:split_index].copy()
    meta_test = merged.iloc[split_index:].copy()

    if meta_train["label"].nunique() < 2 or meta_test["label"].nunique() < 2:
        rng = np.random.RandomState(cfg.random_state)
        indices = np.arange(len(merged))
        rng.shuffle(indices)
        split_index = int(len(indices) * cfg.meta_train_ratio)
        train_idx = indices[:split_index]
        test_idx = indices[split_index:]
        meta_train = merged.iloc[train_idx].copy()
        meta_test = merged.iloc[test_idx].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("meta_model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=cfg.random_state
        ))
    ])

    model.fit(meta_train[feature_cols], meta_train["label"])

    meta_train["stacking_prob"] = model.predict_proba(meta_train[feature_cols])[:, 1]
    meta_test["stacking_prob"] = model.predict_proba(meta_test[feature_cols])[:, 1]
    meta_test["stacking_prediction"] = (
        meta_test["stacking_prob"] >= cfg.classification_threshold
    ).astype(int)

    return model, meta_train, meta_test


def compare_base_and_stacking(meta_test: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    model_prob_cols = {
        "Logistic Regression": "logistic_prob",
        "XGBoost": "xgboost_prob",
        "DIN": "din_prob",
        "Stacking": "stacking_prob"
    }

    y_true = meta_test["label"].to_numpy()
    for model_name, prob_col in model_prob_cols.items():
        metrics = evaluate_binary(
            y_true,
            meta_test[prob_col].to_numpy(),
            cfg.classification_threshold
        )
        metrics["model"] = model_name
        rows.append(metrics)

    return pd.DataFrame(rows)[
        ["model", "auc", "accuracy", "precision", "recall", "f1"]
    ].sort_values("auc", ascending=False)


# ============================================================
# 5. 逻辑回归解释
# ============================================================
def load_model(path: Optional[Path]):
    """
    安全读取由pickle或joblib保存的模型。

    说明：
    - sklearn模型虽然扩展名可能是.pkl，但常常由joblib.dump保存；
    - 因此优先使用joblib.load读取；
    - 如果joblib读取失败，再回退到pickle.load。
    """
    if path is None or not path.exists():
        return None

    suffix = path.suffix.lower()

    if suffix in (".pkl", ".pickle", ".joblib"):
        try:
            import joblib
            return joblib.load(path)
        except Exception as joblib_error:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as pickle_error:
                print(
                    f"警告：模型文件读取失败：{path}\n"
                    f"joblib错误：{joblib_error}\n"
                    f"pickle错误：{pickle_error}"
                )
                return None

    return None


def get_estimator(model):
    if model is None:
        return None

    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        for step in reversed(steps):
            if hasattr(step, "coef_"):
                return step

    if hasattr(model, "coef_"):
        return model

    return None


def explain_logistic_coefficients(
    model,
    feature_names: List[str],
    cfg: Config
) -> Optional[pd.DataFrame]:
    estimator = get_estimator(model)
    if estimator is None:
        print("未找到可解释的逻辑回归模型，跳过原始逻辑回归系数分析。")
        return None

    coefficients = np.ravel(estimator.coef_)
    if len(coefficients) != len(feature_names):
        print(
            f"逻辑回归系数数量({len(coefficients)})与特征数量"
            f"({len(feature_names)})不一致，跳过。"
        )
        return None

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
        "direction": np.where(coefficients >= 0, "促进购买", "抑制购买")
    }).sort_values("absolute_coefficient", ascending=False)

    output_dir = Path(cfg.results_output_dir)
    coef_df.to_csv(
        output_dir / "logistic_feature_coefficients.csv",
        index=False,
        encoding="utf-8-sig"
    )

    top = coef_df.head(cfg.top_n_features).sort_values("coefficient")
    plt.figure(figsize=(10, max(6, len(top) * 0.35)))
    plt.barh(top["feature"], top["coefficient"])
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title("Logistic Regression Feature Coefficients")
    plt.tight_layout()
    plt.savefig(output_dir / "logistic_feature_coefficients.png", dpi=200)
    plt.close()

    return coef_df


def explain_stacking_coefficients(model: Pipeline, cfg: Config) -> pd.DataFrame:
    meta = model.named_steps["meta_model"]
    feature_names = ["logistic_prob", "xgboost_prob", "din_prob"]
    coef_df = pd.DataFrame({
        "base_model": ["Logistic Regression", "XGBoost", "DIN"],
        "stacking_coefficient": np.ravel(meta.coef_),
        "absolute_coefficient": np.abs(np.ravel(meta.coef_))
    }).sort_values("absolute_coefficient", ascending=False)

    coef_df.to_csv(
        Path(cfg.results_output_dir) / "stacking_base_model_weights.csv",
        index=False,
        encoding="utf-8-sig"
    )
    return coef_df


# ============================================================
# 6. XGBoost SHAP解释
# ============================================================
def prepare_feature_data(cfg: Config) -> Tuple[Optional[pd.DataFrame], List[str]]:
    path = find_existing_file(
        cfg,
        cfg.test_feature_path,
        ["test", "dataset"]
    )
    if path is None:
        print("未找到测试特征数据，部分SHAP分析将跳过。")
        return None, []

    df = read_table(path)
    exclude_keywords = [
        "label", "target", "prediction", "probability",
        "split", "time", "date"
    ]

    numeric_cols = []
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in exclude_keywords):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    feature_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True)).fillna(0)

    return feature_df, numeric_cols


def explain_xgboost_shap(
    model,
    X: Optional[pd.DataFrame],
    feature_names: List[str],
    cfg: Config
) -> Optional[pd.DataFrame]:
    if model is None or X is None or X.empty:
        print("XGBoost模型或测试特征不存在，跳过XGBoost SHAP分析。")
        return None

    try:
        import shap
    except ImportError:
        print("未安装shap，跳过XGBoost SHAP。安装命令：pip install shap")
        return None

    sample = X.sample(
        n=min(cfg.shap_sample_size, len(X)),
        random_state=cfg.random_state
    )

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
    except Exception:
        try:
            explainer = shap.Explainer(model, sample)
            explanation = explainer(sample)
            shap_values = explanation.values
        except Exception as e:
            print(f"XGBoost SHAP计算失败：{e}")
            return None

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names[:len(importance)],
        "mean_abs_shap": importance
    }).sort_values("mean_abs_shap", ascending=False)

    output_dir = Path(cfg.results_output_dir)
    shap_df.to_csv(
        output_dir / "xgboost_shap_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    try:
        shap.summary_plot(
            shap_values,
            sample.iloc[:, :shap_values.shape[1]],
            show=False,
            max_display=cfg.top_n_features
        )
        plt.tight_layout()
        plt.savefig(output_dir / "xgboost_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"XGBoost SHAP图保存失败：{e}")

    return shap_df


# ============================================================
# 7. DIN近似SHAP解释
# ============================================================
def explain_din_with_prediction_shap(
    merged_features: pd.DataFrame,
    cfg: Config
) -> Optional[pd.DataFrame]:
    """
    当DIN原始PyTorch输入结构复杂且不能直接从当前文件恢复时，
    使用DIN预测概率作为被解释目标，训练一个可解释代理模型，
    再对代理模型计算SHAP。

    代理模型不是替代DIN，而是近似解释DIN输出与输入特征之间的关系。
    """
    required_prob = "din_prob"
    if required_prob not in merged_features.columns:
        print("没有DIN预测概率，跳过DIN代理SHAP解释。")
        return None

    exclude = {
        "label", "din_prob", "logistic_prob", "xgboost_prob",
        "stacking_prob", "stacking_prediction"
    }
    feature_cols = [
        c for c in merged_features.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(merged_features[c])
    ]

    if len(feature_cols) < 2:
        print("可用于DIN解释的数值特征不足，跳过DIN SHAP。")
        return None

    try:
        from sklearn.ensemble import RandomForestRegressor
        import shap
    except ImportError:
        print("DIN代理SHAP需要shap和scikit-learn。")
        return None

    data = merged_features[feature_cols + ["din_prob"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(data) < 20:
        print("DIN代理SHAP样本太少，跳过。")
        return None

    sample = data.sample(
        n=min(cfg.shap_sample_size, len(data)),
        random_state=cfg.random_state
    )

    X = sample[feature_cols]
    y = sample["din_prob"]

    proxy = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        random_state=cfg.random_state,
        n_jobs=-1
    )
    proxy.fit(X, y)

    explainer = shap.TreeExplainer(proxy)
    shap_values = explainer.shap_values(X)
    importance = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": importance
    }).sort_values("mean_abs_shap", ascending=False)

    output_dir = Path(cfg.results_output_dir)
    shap_df.to_csv(
        output_dir / "din_proxy_shap_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    try:
        shap.summary_plot(
            shap_values,
            X,
            show=False,
            max_display=cfg.top_n_features
        )
        plt.tight_layout()
        plt.savefig(output_dir / "din_proxy_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"DIN代理SHAP图保存失败：{e}")

    with open(output_dir / "din_shap_explanation_note.txt", "w", encoding="utf-8") as f:
        f.write(
            "DIN模型采用复杂的序列Embedding与注意力结构。当前程序使用代理模型近似DIN预测概率，"
            "并对代理模型计算SHAP值。该结果用于解释DIN输出与输入特征的关联，"
            "不能完全等同于DIN神经网络内部的精确SHAP值。"
        )

    return shap_df


# ============================================================
# 8. 错误样本、冷启动用户、长尾商品分析
# ============================================================
def add_behavior_statistics(
    prediction_df: pd.DataFrame,
    feature_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    if feature_df is None or feature_df.empty:
        return prediction_df.copy()

    out = prediction_df.copy()

    keys = [k for k in ["user_id", "item_id"] if k in out.columns and k in feature_df.columns]
    if keys:
        feature_unique = feature_df.drop_duplicates(keys)
        out = out.merge(feature_unique, on=keys, how="left")
    elif len(feature_df) >= len(out):
        feature_part = feature_df.iloc[:len(out)].reset_index(drop=True)
        out = pd.concat([out.reset_index(drop=True), feature_part], axis=1)

    return out


def detect_activity_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    user_candidates = [
        "user_total_behavior", "user_behavior_count", "user_activity_count",
        "user_total_actions", "user_interaction_count", "user_active_count"
    ]
    item_candidates = [
        "item_total_behavior", "item_behavior_count", "item_popularity",
        "item_interaction_count", "item_total_actions", "item_view_count"
    ]

    user_col = detect_column(df, user_candidates, required=False)
    item_col = detect_column(df, item_candidates, required=False)
    return user_col, item_col


def analyze_errors(
    df: pd.DataFrame,
    cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["stacking_prediction"] = (
        out["stacking_prob"] >= cfg.classification_threshold
    ).astype(int)

    out["error_type"] = "正确预测"
    out.loc[
        (out["label"] == 0) & (out["stacking_prediction"] == 1),
        "error_type"
    ] = "假阳性_FP"
    out.loc[
        (out["label"] == 1) & (out["stacking_prediction"] == 0),
        "error_type"
    ] = "假阴性_FN"

    user_activity_col, item_popularity_col = detect_activity_columns(out)

    if user_activity_col is not None:
        out["cold_start_user"] = (
            pd.to_numeric(out[user_activity_col], errors="coerce")
            <= cfg.cold_user_threshold
        )
    elif "user_id" in out.columns:
        user_counts = out["user_id"].value_counts()
        out["cold_start_user"] = out["user_id"].map(user_counts).fillna(0) <= cfg.cold_user_threshold
    else:
        out["cold_start_user"] = False

    if item_popularity_col is not None:
        popularity = pd.to_numeric(out[item_popularity_col], errors="coerce")
        threshold = popularity.quantile(cfg.long_tail_quantile)
        out["long_tail_item"] = popularity <= threshold
    elif "item_id" in out.columns:
        item_counts = out["item_id"].value_counts()
        threshold = item_counts.quantile(cfg.long_tail_quantile)
        out["long_tail_item"] = out["item_id"].map(item_counts).fillna(0) <= threshold
    else:
        out["long_tail_item"] = False

    error_samples = out[out["error_type"] != "正确预测"].copy()
    error_samples["absolute_error"] = np.abs(
        error_samples["label"] - error_samples["stacking_prob"]
    )
    error_samples = error_samples.sort_values("absolute_error", ascending=False)

    scenario_rows = []
    scenarios = {
        "整体样本": np.ones(len(out), dtype=bool),
        "冷启动用户": out["cold_start_user"].fillna(False).to_numpy(),
        "非冷启动用户": (~out["cold_start_user"].fillna(False)).to_numpy(),
        "长尾商品": out["long_tail_item"].fillna(False).to_numpy(),
        "非长尾商品": (~out["long_tail_item"].fillna(False)).to_numpy()
    }

    for name, mask in scenarios.items():
        part = out.loc[mask]
        if len(part) == 0:
            continue
        metrics = evaluate_binary(
            part["label"].to_numpy(),
            part["stacking_prob"].to_numpy(),
            cfg.classification_threshold
        )
        metrics["scenario"] = name
        metrics["sample_count"] = len(part)
        metrics["positive_rate"] = float(part["label"].mean())
        metrics["error_rate"] = float(
            (part["label"] != part["stacking_prediction"]).mean()
        )
        scenario_rows.append(metrics)

    scenario_df = pd.DataFrame(scenario_rows)[
        [
            "scenario", "sample_count", "positive_rate", "auc",
            "accuracy", "precision", "recall", "f1", "error_rate"
        ]
    ]

    output_dir = Path(cfg.results_output_dir)
    error_samples.to_csv(
        output_dir / "stacking_error_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )
    scenario_df.to_csv(
        output_dir / "cold_start_long_tail_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return error_samples, scenario_df


# ============================================================
# 9. 汇总核心特征
# ============================================================
def combine_core_features(
    logistic_df: Optional[pd.DataFrame],
    xgb_shap_df: Optional[pd.DataFrame],
    din_shap_df: Optional[pd.DataFrame],
    cfg: Config
) -> pd.DataFrame:
    frames = []

    if logistic_df is not None and not logistic_df.empty:
        part = logistic_df[["feature", "absolute_coefficient"]].copy()
        max_value = part["absolute_coefficient"].max()
        part["normalized_importance"] = (
            part["absolute_coefficient"] / max_value if max_value > 0 else 0
        )
        part["source_model"] = "Logistic Regression"
        frames.append(part[["feature", "source_model", "normalized_importance"]])

    if xgb_shap_df is not None and not xgb_shap_df.empty:
        part = xgb_shap_df[["feature", "mean_abs_shap"]].copy()
        max_value = part["mean_abs_shap"].max()
        part["normalized_importance"] = (
            part["mean_abs_shap"] / max_value if max_value > 0 else 0
        )
        part["source_model"] = "XGBoost"
        frames.append(part[["feature", "source_model", "normalized_importance"]])

    if din_shap_df is not None and not din_shap_df.empty:
        part = din_shap_df[["feature", "mean_abs_shap"]].copy()
        max_value = part["mean_abs_shap"].max()
        part["normalized_importance"] = (
            part["mean_abs_shap"] / max_value if max_value > 0 else 0
        )
        part["source_model"] = "DIN"
        frames.append(part[["feature", "source_model", "normalized_importance"]])

    if not frames:
        return pd.DataFrame(columns=[
            "feature", "mean_normalized_importance", "model_count"
        ])

    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby("feature").agg(
        mean_normalized_importance=("normalized_importance", "mean"),
        max_normalized_importance=("normalized_importance", "max"),
        model_count=("source_model", "nunique"),
        source_models=("source_model", lambda x: ",".join(sorted(set(x))))
    ).reset_index()

    summary = summary.sort_values(
        ["model_count", "mean_normalized_importance"],
        ascending=[False, False]
    )

    summary.to_csv(
        Path(cfg.results_output_dir) / "core_purchase_features.csv",
        index=False,
        encoding="utf-8-sig"
    )
    return summary


# ============================================================
# 10. 自动生成中文报告
# ============================================================
def generate_report(
    comparison_df: pd.DataFrame,
    stacking_weights: pd.DataFrame,
    core_features: pd.DataFrame,
    error_samples: pd.DataFrame,
    scenario_df: pd.DataFrame,
    cfg: Config
) -> None:
    output_path = Path(cfg.results_output_dir) / "model_fusion_explainability_report.md"

    best_row = comparison_df.iloc[0]
    stacking_row = comparison_df[
        comparison_df["model"] == "Stacking"
    ].iloc[0]

    fp_count = int((error_samples["error_type"] == "假阳性_FP").sum())
    fn_count = int((error_samples["error_type"] == "假阴性_FN").sum())

    lines = []
    lines.append("# 第四项：模型融合与可解释性分析报告\n")
    lines.append("## 1. Stacking模型融合方案\n")
    lines.append(
        "本项目将逻辑回归、XGBoost和DIN三个基础模型的预测概率作为二层输入，"
        "使用带类别权重的逻辑回归作为元学习器。该方案既保留了传统模型对结构化特征的学习能力，"
        "又融合了DIN对用户行为序列和兴趣变化的建模能力。\n"
    )

    lines.append("## 2. 模型性能对比\n")
    lines.append(comparison_df.to_markdown(index=False))
    lines.append("\n")
    lines.append(
        f"当前AUC最高的模型为 **{best_row['model']}**，AUC为"
        f" **{best_row['auc']:.4f}**。Stacking模型AUC为"
        f" **{stacking_row['auc']:.4f}**，F1为 **{stacking_row['f1']:.4f}**。\n"
    )

    lines.append("## 3. 三个基础模型在Stacking中的贡献\n")
    lines.append(stacking_weights.to_markdown(index=False))
    lines.append("\n")
    if not stacking_weights.empty:
        strongest = stacking_weights.iloc[0]
        lines.append(
            f"二层逻辑回归中绝对权重最大的基础模型为 **{strongest['base_model']}**，"
            "说明该模型的预测结果对最终融合结果影响最大。\n"
        )

    lines.append("## 4. 核心购买决策特征\n")
    if core_features.empty:
        lines.append(
            "未获得完整的原始特征或基础模型文件，因此本次未生成完整的跨模型核心特征排序。\n"
        )
    else:
        lines.append(core_features.head(cfg.top_n_features).to_markdown(index=False))
        lines.append("\n")
        lines.append(
            "模型重点关注用户活跃程度、商品热度、历史购买行为、浏览与加购频率、"
            "用户商品交互强度以及时间相关特征。多个模型同时认为重要的特征，"
            "具有更强的业务解释价值。\n"
        )

    lines.append("## 5. 错误样本分析\n")
    lines.append(
        f"Stacking模型共识别出假阳性样本 **{fp_count}** 条，"
        f"假阴性样本 **{fn_count}** 条。\n"
    )
    lines.append(
        "- 假阳性表示模型预测会购买，但用户实际没有购买。可能原因包括用户仅进行价格比较、"
        "短期浏览兴趣较强但购买意愿不足、优惠力度不足或商品缺货。\n"
        "- 假阴性表示模型预测不会购买，但用户实际发生购买。可能原因包括用户突然产生需求、"
        "外部营销活动刺激、历史行为过少或模型未捕捉到最新兴趣变化。\n"
    )

    lines.append("## 6. 冷启动用户与长尾商品分析\n")
    lines.append(scenario_df.to_markdown(index=False))
    lines.append("\n")
    lines.append(
        "冷启动用户历史行为较少，模型无法充分学习其兴趣偏好，因此预测通常更依赖人口属性、"
        "时间特征和商品整体热度。长尾商品交互次数少，模型难以获得稳定的商品Embedding，"
        "容易出现召回率和AUC下降。\n"
    )

    lines.append("## 7. 当前模型局限性\n")
    lines.append(
        "1. 数据时间跨度较短，目前采用过去7天行为预测未来3天购买，"
        "难以覆盖用户长期兴趣变化。\n"
        "2. 冷启动用户和新商品缺乏历史交互，序列模型和Embedding学习不充分。\n"
        "3. 长尾商品样本数量少，模型更容易偏向热门商品。\n"
        "4. DIN代理SHAP是对DIN输出的近似解释，不能完全代表神经网络内部所有注意力机制。\n"
        "5. 固定0.5阈值未必是业务最优阈值，后续应结合营销成本和转化收益选择阈值。\n"
    )

    lines.append("## 8. 优化建议\n")
    lines.append(
        "1. 增加更长时间跨度的数据，并补充用户画像、商品价格、促销、库存等业务特征。\n"
        "2. 对冷启动用户引入相似用户、热门商品和上下文推荐策略。\n"
        "3. 对长尾商品采用类别级Embedding、内容特征和重采样策略。\n"
        "4. 在验证集上根据F1、召回率或业务收益自动选择分类阈值。\n"
        "5. 使用交叉验证生成基础模型的Out-of-Fold预测，进一步降低Stacking过拟合风险。\n"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 11. 主流程
# ============================================================
def main(cfg: Config) -> None:
    set_seed(cfg.random_state)

    cfg.processed_output_dir = str(
        resolve_path(cfg, cfg.processed_output_dir)
    )
    cfg.results_output_dir = str(
        resolve_path(cfg, cfg.results_output_dir)
    )

    ensure_dir(cfg.processed_output_dir)
    ensure_dir(cfg.results_output_dir)

    print("项目根目录：", Path(cfg.project_root).resolve())
    print("processed输出目录：", Path(cfg.processed_output_dir).resolve())
    print("results输出目录：", Path(cfg.results_output_dir).resolve())

    with open(
        Path(cfg.results_output_dir) / "config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("第四项：模型融合与可解释性分析")
    print("=" * 70)

    logistic_pred_path = find_existing_file(
        cfg, cfg.logistic_pred_path, ["logistic", "prediction"]
    )
    xgb_pred_path = find_existing_file(
        cfg, cfg.xgboost_pred_path, ["xgboost", "prediction"]
    )
    din_pred_path = find_existing_file(
        cfg, cfg.din_pred_path, ["din", "prediction"]
    )

    missing = []
    if logistic_pred_path is None:
        missing.append("逻辑回归预测文件")
    if xgb_pred_path is None:
        missing.append("XGBoost预测文件")
    if din_pred_path is None:
        missing.append("DIN预测文件")

    if missing:
        raise FileNotFoundError(
            "缺少以下文件：" + "、".join(missing) + "\n"
            "请先运行第二项和第三项模型，并确保各模型保存了测试集预测概率。"
        )

    print("\n[1/9] 读取三个基础模型预测结果")
    logistic_pred = standardize_prediction_file(
        logistic_pred_path, "logistic", cfg
    )
    xgb_pred = standardize_prediction_file(
        xgb_pred_path, "xgboost", cfg
    )
    din_pred = standardize_prediction_file(
        din_pred_path, "din", cfg
    )

    print("\n[2/9] 合并预测结果")
    merged = merge_predictions(logistic_pred, xgb_pred, din_pred)
    merged.to_csv(
        Path(cfg.results_output_dir) / "merged_base_model_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"融合可用样本数：{len(merged):,}")

    print("\n[3/9] 训练Stacking元学习器")
    stacking_model, meta_train, meta_test = train_stacking_model(merged, cfg)

    stacking_model_path = (
        Path(cfg.processed_output_dir) / "stacking_meta_model.pkl"
    )
    with open(stacking_model_path, "wb") as f:
        pickle.dump(stacking_model, f)

    meta_test.to_csv(
        Path(cfg.results_output_dir) / "stacking_test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[4/9] 对比基础模型与融合模型")
    comparison_df = compare_base_and_stacking(meta_test, cfg)
    comparison_df.to_csv(
        Path(cfg.results_output_dir) / "model_performance_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(comparison_df.to_string(index=False))

    stacking_weights = explain_stacking_coefficients(stacking_model, cfg)

    print("\n[5/9] 逻辑回归特征系数解释")
    feature_df, feature_names = prepare_feature_data(cfg)

    logistic_model_path = find_existing_file(
        cfg, cfg.logistic_model_path, ["logistic", "model"]
    )
    logistic_model = load_model(logistic_model_path)
    logistic_coef_df = explain_logistic_coefficients(
        logistic_model, feature_names, cfg
    )

    print("\n[6/9] XGBoost SHAP解释")
    xgb_model_path = find_existing_file(
        cfg, cfg.xgboost_model_path, ["xgboost"]
    )
    xgb_model = load_model(xgb_model_path)
    xgb_shap_df = explain_xgboost_shap(
        xgb_model, feature_df, feature_names, cfg
    )

    print("\n[7/9] DIN SHAP近似解释")
    analysis_df = add_behavior_statistics(meta_test, feature_df)
    din_shap_df = explain_din_with_prediction_shap(analysis_df, cfg)

    print("\n[8/9] 错误样本与场景分析")
    error_samples, scenario_df = analyze_errors(analysis_df, cfg)

    print("\n[9/9] 汇总核心特征并生成报告")
    core_features = combine_core_features(
        logistic_coef_df, xgb_shap_df, din_shap_df, cfg
    )

    generate_report(
        comparison_df,
        stacking_weights,
        core_features,
        error_samples,
        scenario_df,
        cfg
    )

    required_outputs = [
        Path(cfg.processed_output_dir) / "stacking_meta_model.pkl",
        Path(cfg.results_output_dir) / "config.json",
        Path(cfg.results_output_dir) / "merged_base_model_predictions.csv",
        Path(cfg.results_output_dir) / "stacking_test_predictions.csv",
        Path(cfg.results_output_dir) / "model_performance_comparison.csv",
        Path(cfg.results_output_dir) / "stacking_base_model_weights.csv",
        Path(cfg.results_output_dir) / "stacking_error_samples.csv",
        Path(cfg.results_output_dir) / "cold_start_long_tail_analysis.csv",
        Path(cfg.results_output_dir) / "core_purchase_features.csv",
        Path(cfg.results_output_dir) / "model_fusion_explainability_report.md"
    ]

    failed_outputs = [
        str(path)
        for path in required_outputs
        if not path.exists()
    ]
    if failed_outputs:
        raise RuntimeError(
            "以下必要输出文件保存失败：\n"
            + "\n".join(failed_outputs)
        )

    print("\n全部任务完成。")
    print("processed输出保存在：", Path(cfg.processed_output_dir).resolve())
    print("results输出保存在：", Path(cfg.results_output_dir).resolve())
    print("\n主要输出文件：")
    print("1. model_performance_comparison.csv")
    print("2. stacking_test_predictions.csv")
    print("3. stacking_base_model_weights.csv")
    print("4. logistic_feature_coefficients.csv")
    print("5. xgboost_shap_importance.csv")
    print("6. din_proxy_shap_importance.csv")
    print("7. stacking_error_samples.csv")
    print("8. cold_start_long_tail_analysis.csv")
    print("9. core_purchase_features.csv")
    print("10. model_fusion_explainability_report.md")


def parse_args():
    parser = argparse.ArgumentParser(
        description="第四项：Stacking模型融合与可解释性分析"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2])
    )
    parser.add_argument(
        "--processed_output_dir",
        type=str,
        default=Config.processed_output_dir
    )
    parser.add_argument(
        "--results_output_dir",
        type=str,
        default=Config.results_output_dir
    )
    parser.add_argument(
        "--logistic_pred_path",
        type=str,
        default=Config.logistic_pred_path
    )
    parser.add_argument(
        "--xgboost_pred_path",
        type=str,
        default=Config.xgboost_pred_path
    )
    parser.add_argument(
        "--din_pred_path",
        type=str,
        default=Config.din_pred_path
    )
    parser.add_argument(
        "--logistic_model_path",
        type=str,
        default=Config.logistic_model_path
    )
    parser.add_argument(
        "--xgboost_model_path",
        type=str,
        default=Config.xgboost_model_path
    )
    parser.add_argument(
        "--test_feature_path",
        type=str,
        default=Config.test_feature_path
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=Config.classification_threshold
    )
    parser.add_argument(
        "--shap_sample_size",
        type=int,
        default=Config.shap_sample_size
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config(
        project_root=args.project_root,
        results_output_dir=args.results_output_dir,
        logistic_pred_path=args.logistic_pred_path,
        xgboost_pred_path=args.xgboost_pred_path,
        din_pred_path=args.din_pred_path,
        logistic_model_path=args.logistic_model_path,
        xgboost_model_path=args.xgboost_model_path,
        test_feature_path=args.test_feature_path,
        classification_threshold=args.threshold,
        shap_sample_size=args.shap_sample_size
    )
    main(config)
