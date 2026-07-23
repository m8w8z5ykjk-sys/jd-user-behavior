"""
文件名称：13_business_ab_test.py

功能：
1. 读取12_model_fusion_explainability.py生成的Stacking测试集预测结果；
2. 根据模型预测分数筛选高购买概率目标用户；
3. 按预测分数分层后随机划分实验组和对照组；
4. 明确A/B测试核心指标、护栏指标、样本量和实验周期；
5. 模拟精准营销对实验组购买转化率的提升；
6. 计算转化率、增量购买人数、增量收入、营销成本、增量利润和ROI；
7. 分析不同预测分层和商品类别的实验表现；
8. 生成精准营销、商品推荐和逐步上线建议；
9. 保存实验明细、指标汇总、业务分析报告和可视化图表；
10. 将本文件产生的所有results输出保存到以当前Python文件名命名的专属目录。

输入文件名：
1. results/12_model_fusion_explainability/stacking_test_predictions.csv
   默认输入文件，应至少包含真实标签和Stacking预测概率。

兼容输入：
- 当默认输入文件不存在时，程序会在results目录下递归搜索包含
  真实标签和预测概率的CSV文件；
- 也可以通过PREDICTION_FILE手动指定输入文件；
- 可通过LABEL_COL和SCORE_COL手动指定标签列和概率列。

输入字段要求：
- 必需字段：
  真实标签列、模型预测概率列；
- 推荐字段：
  user_id、item_id、item_category、price。

processed输出目录：
- 本文件不生成需要保存到data/processed的中间数据，因此不创建processed目录。

results输出目录：
- results/13_business_ab_test/

results输出文件名：
1. ab_test_user_level_detail.csv
   用户级实验分组、模型分数和模拟转化结果明细。

2. ab_test_group_metrics.csv
   实验组和对照组的样本量、购买数、转化率、收入、成本和毛利润。

3. ab_test_effect_summary.csv
   转化提升、显著性、样本量、增量收入、增量利润和ROI汇总。

4. score_segment_analysis.csv
   不同模型预测分层下的实验表现。

5. category_recommendation_analysis.csv
   不同商品类别的目标用户、预测分数、转化率和推荐优先级。

6. business_ab_test_report.md
   完整A/B测试设计、模拟结果、营销建议和上线决策报告。

7. 01_ab_conversion_rate.png
   实验组与对照组转化率对比图。

8. 02_ab_gross_profit.png
   实验组与对照组毛利润对比图。

9. 03_score_segment_conversion.png
   不同预测分层的转化率对比图。

目录规则：
- CSV、Markdown和PNG结果统一保存到：
  results/13_business_ab_test/
- 该目录由程序自动创建。

路径检查：
- 当前文件应位于项目根目录/src/models/；
- 项目根目录通过Path(__file__).resolve().parents[2]自动定位；
- 默认输入文件来自：
  results/12_model_fusion_explainability/stacking_test_predictions.csv。
"""


from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 1. 用户配置区
# ============================================================

RANDOM_SEED = 42

# 当前文件位于：项目根目录/src/models/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 全部results目录，用于默认输入和兼容搜索
RESULTS_DIR = PROJECT_ROOT / "results"

# 本脚本专属results输出目录
OUTPUT_DIR = RESULTS_DIR / "13_business_ab_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 默认读取12_model_fusion_explainability.py生成的Stacking预测结果
DEFAULT_PREDICTION_FILE = (
    RESULTS_DIR
    / "12_model_fusion_explainability"
    / "stacking_test_predictions.csv"
)

# 如自动识别失败，可填写具体文件路径。
# 设置为None时：优先读取DEFAULT_PREDICTION_FILE；
# 默认文件不存在时，再递归搜索results目录。
PREDICTION_FILE: Optional[Path] = None

# 如自动识别失败，可手动填写列名，例如：
# LABEL_COL = "label"
# SCORE_COL = "stacking_pred_proba"
LABEL_COL: Optional[str] = None
SCORE_COL: Optional[str] = None

# 用户、商品字段用于业务分层；没有这些字段也可以运行。
USER_COL_CANDIDATES = [
    "user_id", "userid", "user", "uid"
]
ITEM_COL_CANDIDATES = [
    "item_id", "itemid", "product_id", "sku_id", "item", "product"
]
CATEGORY_COL_CANDIDATES = [
    "item_category", "category_id", "category", "cate_id", "product_category"
]
PRICE_COL_CANDIDATES = [
    "price", "item_price", "product_price", "amount", "gmv", "order_amount"
]

# A/B 实验参数
TEST_RATIO = 0.50                  # 高潜用户中，50%进入实验组，50%进入对照组
TARGET_TOP_RATIO = 0.20            # 选择预测分数最高的20%用户作为目标人群
ALPHA = 0.05                       # 显著性水平
STATISTICAL_POWER = 0.80           # 统计功效
EXPECTED_RELATIVE_LIFT = 0.15      # 预计实验组相对提升15%
MAX_EXPERIMENT_DAYS = 28           # 最长实验周期
MIN_EXPERIMENT_DAYS = 14           # 最短实验周期
DAILY_REACH_RATE = 0.10            # 每天约可触达目标用户的10%

# 收益假设，可按真实业务修改
DEFAULT_AVERAGE_ORDER_VALUE = 100.0  # 平均客单价
INCENTIVE_COST_PER_USER = 2.0        # 每位实验组用户平均营销成本
VARIABLE_COST_RATE = 0.30             # 商品变动成本率
SIMULATE_TREATMENT_LIFT = True        # 是否模拟实验组效果
MAX_ABSOLUTE_LIFT = 0.08              # 模拟时最大绝对转化率提升8个百分点


# ============================================================
# 2. 通用工具函数
# ============================================================

def normalize_name(name: str) -> str:
    """统一列名格式，便于自动识别。"""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """从列名中查找候选字段，优先精确匹配，其次模糊匹配。"""
    original = list(columns)
    normalized = {normalize_name(c): c for c in original}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    for candidate in candidates:
        key = normalize_name(candidate)
        for normalized_col, original_col in normalized.items():
            if key in normalized_col:
                return original_col

    return None


def identify_label_column(df: pd.DataFrame) -> Optional[str]:
    """自动识别真实标签列。"""
    candidates = [
        "label", "target", "y_true", "true_label", "actual",
        "is_buy", "is_purchase", "purchase_label", "converted",
        "conversion", "buy_label", "future_7d_purchase",
        "purchase_in_next_7_days"
    ]
    return find_column(df.columns, candidates)


def identify_score_column(df: pd.DataFrame, label_col: Optional[str]) -> Optional[str]:
    """自动识别模型预测概率列，优先使用融合模型预测结果。"""
    priority_candidates = [
        "stacking_pred_proba", "stacking_probability", "stacking_score",
        "fusion_pred_proba", "fusion_probability", "fusion_score",
        "ensemble_pred_proba", "ensemble_probability", "ensemble_score",
        "din_pred_proba", "din_probability", "din_score",
        "xgboost_pred_proba", "xgb_pred_proba", "xgb_probability",
        "lightgbm_pred_proba", "lgbm_pred_proba",
        "logistic_pred_proba", "lr_pred_proba",
        "pred_proba", "predict_proba", "prediction_probability",
        "probability", "purchase_probability", "score", "y_score"
    ]
    found = find_column(df.columns, priority_candidates)
    if found:
        return found

    numeric_candidates = []
    for col in df.columns:
        if col == label_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(values) == 0:
                continue
            if values.between(0, 1).mean() >= 0.95:
                numeric_candidates.append(col)

    probability_like = [
        col for col in numeric_candidates
        if any(k in normalize_name(col) for k in ["pred", "prob", "score"])
    ]
    if probability_like:
        return probability_like[0]

    return numeric_candidates[0] if numeric_candidates else None


def locate_prediction_file(results_dir: Path) -> Path:
    """
    自动查找预测结果文件。
    文件必须至少包含：
    - 真实标签列；
    - 0~1之间的模型预测概率列。
    """
    if PREDICTION_FILE is not None:
        path = Path(PREDICTION_FILE)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"指定的预测文件不存在：{path}")
        return path

    if DEFAULT_PREDICTION_FILE.exists():
        return DEFAULT_PREDICTION_FILE

    preferred_keywords = [
        "stacking", "fusion", "ensemble", "prediction",
        "predictions", "explainability", "model_result"
    ]

    csv_files = list(results_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"在 {results_dir} 中没有找到CSV文件。"
            "请先运行模型融合脚本生成包含真实标签和预测概率的CSV结果。"
        )

    def file_priority(path: Path) -> tuple:
        name = normalize_name(path.stem)
        keyword_score = sum(keyword in name for keyword in preferred_keywords)
        return (-keyword_score, -path.stat().st_mtime)

    csv_files.sort(key=file_priority)

    for file_path in csv_files:
        try:
            sample = pd.read_csv(file_path, nrows=3000)
        except Exception:
            continue

        label_col = LABEL_COL or identify_label_column(sample)
        score_col = SCORE_COL or identify_score_column(sample, label_col)

        if label_col and score_col:
            label_values = pd.to_numeric(sample[label_col], errors="coerce").dropna()
            score_values = pd.to_numeric(sample[score_col], errors="coerce").dropna()
            if (
                len(label_values) > 0
                and len(score_values) > 0
                and set(label_values.unique()).issubset({0, 1})
                and score_values.between(0, 1).mean() >= 0.90
            ):
                return file_path

    checked = "\n".join(f"- {p}" for p in csv_files[:20])
    raise ValueError(
        "自动查找失败：没有找到同时包含真实标签和预测概率的CSV文件。\n"
        "请在脚本顶部手动设置 PREDICTION_FILE、LABEL_COL、SCORE_COL。\n"
        f"已检查的部分文件：\n{checked}"
    )


def load_prediction_data(file_path: Path) -> tuple[pd.DataFrame, str, str]:
    """读取并清洗模型预测结果。"""
    df = pd.read_csv(file_path)

    label_col = LABEL_COL or identify_label_column(df)
    if label_col is None:
        raise ValueError(
            f"无法识别真实标签列。当前列名：{list(df.columns)}\n"
            "请在脚本顶部设置 LABEL_COL。"
        )

    score_col = SCORE_COL or identify_score_column(df, label_col)
    if score_col is None:
        raise ValueError(
            f"无法识别预测概率列。当前列名：{list(df.columns)}\n"
            "请在脚本顶部设置 SCORE_COL。"
        )

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[label_col, score_col]).copy()
    df[label_col] = df[label_col].astype(int)
    df[score_col] = df[score_col].clip(0, 1)

    if not set(df[label_col].unique()).issubset({0, 1}):
        raise ValueError(f"标签列 {label_col} 必须只包含0和1。")

    if len(df) < 20:
        raise ValueError("有效预测样本少于20条，无法进行可靠的A/B测试模拟。")

    return df, label_col, score_col


def get_unit_level_data(
    df: pd.DataFrame,
    label_col: str,
    score_col: str
) -> tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    将数据整理为实验分流单位。
    如果存在 user_id，则按用户聚合，避免同一用户同时进入实验组和对照组。
    """
    user_col = find_column(df.columns, USER_COL_CANDIDATES)
    item_col = find_column(df.columns, ITEM_COL_CANDIDATES)
    category_col = find_column(df.columns, CATEGORY_COL_CANDIDATES)
    price_col = find_column(df.columns, PRICE_COL_CANDIDATES)

    if user_col:
        agg_dict = {
            label_col: "max",
            score_col: "max"
        }
        if item_col:
            agg_dict[item_col] = "first"
        if category_col:
            agg_dict[category_col] = "first"
        if price_col:
            agg_dict[price_col] = "mean"

        unit_df = df.groupby(user_col, as_index=False).agg(agg_dict)
        unit_df = unit_df.rename(columns={user_col: "experiment_unit_id"})
    else:
        unit_df = df.copy().reset_index(drop=True)
        unit_df.insert(0, "experiment_unit_id", np.arange(1, len(unit_df) + 1))

    return unit_df, user_col, item_col, category_col, price_col


def select_target_population(
    unit_df: pd.DataFrame,
    score_col: str,
    top_ratio: float
) -> tuple[pd.DataFrame, float]:
    """选择模型预测分数最高的目标人群。"""
    if not 0 < top_ratio <= 1:
        raise ValueError("TARGET_TOP_RATIO必须在0到1之间。")

    threshold = float(unit_df[score_col].quantile(1 - top_ratio))
    target_df = unit_df[unit_df[score_col] >= threshold].copy()

    if len(target_df) < 20:
        target_df = unit_df.nlargest(min(len(unit_df), 20), score_col).copy()
        threshold = float(target_df[score_col].min())

    return target_df, threshold


def stratified_ab_assignment(
    target_df: pd.DataFrame,
    score_col: str,
    test_ratio: float,
    seed: int
) -> pd.DataFrame:
    """
    按预测分数分层随机分组，保证实验组与对照组的高、中、低分用户比例接近。
    """
    rng = np.random.default_rng(seed)
    result = target_df.copy()

    unique_scores = result[score_col].nunique()
    q = min(10, max(2, unique_scores))
    try:
        result["score_stratum"] = pd.qcut(
            result[score_col],
            q=q,
            duplicates="drop",
            labels=False
        )
    except ValueError:
        result["score_stratum"] = 0

    result["ab_group"] = "control"

    for _, index in result.groupby("score_stratum").groups.items():
        index = np.array(list(index))
        rng.shuffle(index)
        treatment_count = int(round(len(index) * test_ratio))
        treatment_count = min(max(treatment_count, 1), max(len(index) - 1, 1))
        treatment_index = index[:treatment_count]
        result.loc[treatment_index, "ab_group"] = "treatment"

    return result


def calculate_required_sample_size(
    baseline_rate: float,
    relative_lift: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    使用两独立比例近似公式计算每组所需样本量。
    z值固定采用常见的双侧α=0.05、功效80%的近似值。
    """
    baseline_rate = float(np.clip(baseline_rate, 0.001, 0.999))
    treatment_rate = float(
        np.clip(baseline_rate * (1 + relative_lift), 0.001, 0.999)
    )

    z_alpha = 1.959963984540054
    z_beta = 0.8416212335729143

    pooled = (baseline_rate + treatment_rate) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_beta * math.sqrt(
            baseline_rate * (1 - baseline_rate)
            + treatment_rate * (1 - treatment_rate)
        )
    ) ** 2
    denominator = (treatment_rate - baseline_rate) ** 2

    if denominator <= 0:
        return len([])

    return int(math.ceil(numerator / denominator))


def normal_cdf(x: float) -> float:
    """标准正态分布累积分布函数。"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def two_proportion_z_test(
    treatment_success: int,
    treatment_n: int,
    control_success: int,
    control_n: int
) -> tuple[float, float]:
    """计算两比例z检验的z统计量和双侧p值。"""
    if treatment_n == 0 or control_n == 0:
        return float("nan"), float("nan")

    p_t = treatment_success / treatment_n
    p_c = control_success / control_n
    pooled = (treatment_success + control_success) / (treatment_n + control_n)
    standard_error = math.sqrt(
        pooled * (1 - pooled) * (1 / treatment_n + 1 / control_n)
    )

    if standard_error == 0:
        return 0.0, 1.0

    z_value = (p_t - p_c) / standard_error
    p_value = 2 * (1 - normal_cdf(abs(z_value)))
    return z_value, p_value


def simulate_ab_outcome(
    ab_df: pd.DataFrame,
    label_col: str,
    score_col: str,
    seed: int
) -> pd.DataFrame:
    """
    离线模拟实验结果：
    - 对照组使用原始真实购买标签；
    - 实验组在原标签基础上，根据预测分数模拟精准营销带来的增量转化；
    - 已经购买的用户保持购买状态；
    - 未购买用户的增量转化概率与模型分数正相关。
    """
    result = ab_df.copy()
    result["observed_conversion"] = result[label_col].astype(int)
    result["incremental_conversion"] = 0

    if not SIMULATE_TREATMENT_LIFT:
        return result

    rng = np.random.default_rng(seed)
    treatment_mask = result["ab_group"].eq("treatment")
    not_converted_mask = result[label_col].eq(0)

    score_rank = result[score_col].rank(pct=True)
    incremental_probability = (
        EXPECTED_RELATIVE_LIFT
        * np.maximum(result[score_col], 0.01)
        * (0.5 + score_rank)
    )
    incremental_probability = incremental_probability.clip(
        lower=0,
        upper=MAX_ABSOLUTE_LIFT
    )

    eligible_mask = treatment_mask & not_converted_mask
    random_draw = rng.random(len(result))
    incremental_mask = eligible_mask & (random_draw < incremental_probability)

    result.loc[incremental_mask, "observed_conversion"] = 1
    result.loc[incremental_mask, "incremental_conversion"] = 1

    return result


def calculate_ab_metrics(
    result_df: pd.DataFrame,
    price_col: Optional[str]
) -> pd.DataFrame:
    """计算实验组和对照组的核心业务指标。"""
    if price_col and price_col in result_df.columns:
        prices = pd.to_numeric(result_df[price_col], errors="coerce")
        average_order_value = float(prices[prices > 0].median())
        if not np.isfinite(average_order_value):
            average_order_value = DEFAULT_AVERAGE_ORDER_VALUE
    else:
        average_order_value = DEFAULT_AVERAGE_ORDER_VALUE

    rows = []
    for group_name, group_df in result_df.groupby("ab_group"):
        n = len(group_df)
        conversions = int(group_df["observed_conversion"].sum())
        conversion_rate = conversions / n if n else 0
        revenue = conversions * average_order_value
        marketing_cost = (
            n * INCENTIVE_COST_PER_USER
            if group_name == "treatment"
            else 0.0
        )
        variable_cost = revenue * VARIABLE_COST_RATE
        gross_profit = revenue - variable_cost - marketing_cost

        rows.append({
            "group": group_name,
            "sample_size": n,
            "conversions": conversions,
            "conversion_rate": conversion_rate,
            "average_order_value": average_order_value,
            "revenue": revenue,
            "marketing_cost": marketing_cost,
            "variable_cost": variable_cost,
            "gross_profit": gross_profit
        })

    metrics = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    return metrics


def build_summary(
    metrics: pd.DataFrame,
    required_sample_per_group: int,
    experiment_days: int
) -> pd.DataFrame:
    """生成实验效果汇总。"""
    treatment = metrics[metrics["group"] == "treatment"].iloc[0]
    control = metrics[metrics["group"] == "control"].iloc[0]

    treatment_rate = treatment["conversion_rate"]
    control_rate = control["conversion_rate"]
    absolute_lift = treatment_rate - control_rate
    relative_lift = (
        absolute_lift / control_rate
        if control_rate > 0
        else float("nan")
    )

    z_value, p_value = two_proportion_z_test(
        int(treatment["conversions"]),
        int(treatment["sample_size"]),
        int(control["conversions"]),
        int(control["sample_size"])
    )

    incremental_conversions = (
        treatment["conversions"]
        - treatment["sample_size"] * control_rate
    )
    incremental_revenue = (
        incremental_conversions * treatment["average_order_value"]
    )
    incremental_profit = (
        incremental_revenue * (1 - VARIABLE_COST_RATE)
        - treatment["marketing_cost"]
    )
    roi = (
        incremental_profit / treatment["marketing_cost"]
        if treatment["marketing_cost"] > 0
        else float("nan")
    )

    enough_sample = (
        treatment["sample_size"] >= required_sample_per_group
        and control["sample_size"] >= required_sample_per_group
    )

    return pd.DataFrame([{
        "control_conversion_rate": control_rate,
        "treatment_conversion_rate": treatment_rate,
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
        "z_value": z_value,
        "p_value": p_value,
        "statistically_significant": bool(p_value < ALPHA),
        "required_sample_per_group": required_sample_per_group,
        "actual_treatment_sample": int(treatment["sample_size"]),
        "actual_control_sample": int(control["sample_size"]),
        "sample_size_sufficient": bool(enough_sample),
        "recommended_experiment_days": experiment_days,
        "incremental_conversions": incremental_conversions,
        "incremental_revenue": incremental_revenue,
        "incremental_profit": incremental_profit,
        "marketing_roi": roi
    }])


def build_score_segment_analysis(
    result_df: pd.DataFrame,
    score_col: str
) -> pd.DataFrame:
    """按预测分数分层分析不同用户群的实验效果。"""
    result = result_df.copy()

    try:
        result["score_segment"] = pd.qcut(
            result[score_col],
            q=5,
            labels=["低潜", "较低潜", "中潜", "较高潜", "高潜"],
            duplicates="drop"
        )
    except ValueError:
        result["score_segment"] = "全部"

    rows = []
    for (segment, group), group_df in result.groupby(
        ["score_segment", "ab_group"],
        observed=False
    ):
        n = len(group_df)
        if n == 0:
            continue
        rows.append({
            "score_segment": str(segment),
            "group": group,
            "sample_size": n,
            "average_model_score": group_df[score_col].mean(),
            "conversion_rate": group_df["observed_conversion"].mean(),
            "incremental_conversions": group_df["incremental_conversion"].sum()
        })

    return pd.DataFrame(rows)


def build_category_recommendations(
    result_df: pd.DataFrame,
    category_col: Optional[str],
    score_col: str
) -> pd.DataFrame:
    """分析高潜商品类别并生成推荐优先级。"""
    if not category_col or category_col not in result_df.columns:
        return pd.DataFrame([{
            "category": "未提供商品类别字段",
            "target_users": len(result_df),
            "average_model_score": result_df[score_col].mean(),
            "conversion_rate": result_df["observed_conversion"].mean(),
            "recommendation_priority": "建议补充商品类别字段后进行个性化推荐"
        }])

    category_result = (
        result_df.groupby(category_col, dropna=False)
        .agg(
            target_users=("experiment_unit_id", "count"),
            average_model_score=(score_col, "mean"),
            conversion_rate=("observed_conversion", "mean"),
            incremental_conversions=("incremental_conversion", "sum")
        )
        .reset_index()
        .rename(columns={category_col: "category"})
    )

    category_result["priority_score"] = (
        category_result["average_model_score"].rank(pct=True) * 0.5
        + category_result["conversion_rate"].rank(pct=True) * 0.3
        + category_result["target_users"].rank(pct=True) * 0.2
    )

    category_result["recommendation_priority"] = pd.cut(
        category_result["priority_score"],
        bins=[-np.inf, 0.40, 0.70, np.inf],
        labels=["低", "中", "高"]
    )

    return category_result.sort_values(
        ["priority_score", "target_users"],
        ascending=False
    ).reset_index(drop=True)


def save_plots(
    metrics: pd.DataFrame,
    score_segments: pd.DataFrame,
    output_dir: Path
) -> None:
    """保存业务落地分析图。"""
    plt.figure(figsize=(8, 5))
    plt.bar(
        metrics["group"],
        metrics["conversion_rate"]
    )
    plt.title("A/B测试转化率对比")
    plt.xlabel("实验分组")
    plt.ylabel("转化率")
    plt.tight_layout()
    plt.savefig(output_dir / "01_ab_conversion_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        metrics["group"],
        metrics["gross_profit"]
    )
    plt.title("A/B测试毛利润对比")
    plt.xlabel("实验分组")
    plt.ylabel("毛利润")
    plt.tight_layout()
    plt.savefig(output_dir / "02_ab_gross_profit.png", dpi=200)
    plt.close()

    if not score_segments.empty:
        pivot = score_segments.pivot_table(
            index="score_segment",
            columns="group",
            values="conversion_rate",
            aggfunc="mean"
        )
        if not pivot.empty:
            pivot.plot(kind="bar", figsize=(9, 5))
            plt.title("不同预测分层的转化率")
            plt.xlabel("用户预测分层")
            plt.ylabel("转化率")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(
                output_dir / "03_score_segment_conversion.png",
                dpi=200
            )
            plt.close()


def write_markdown_report(
    source_file: Path,
    label_col: str,
    score_col: str,
    total_units: int,
    target_units: int,
    threshold: float,
    baseline_rate: float,
    required_sample_per_group: int,
    experiment_days: int,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_path: Path
) -> None:
    """生成第五项业务落地报告。"""
    s = summary.iloc[0]
    treatment = metrics[metrics["group"] == "treatment"].iloc[0]
    control = metrics[metrics["group"] == "control"].iloc[0]

    significance_text = (
        "达到统计显著"
        if s["statistically_significant"]
        else "尚未达到统计显著"
    )
    sample_text = (
        "当前样本量满足要求"
        if s["sample_size_sufficient"]
        else "当前样本量不足，正式实验应继续积累样本"
    )

    top_categories = recommendations.head(5)
    category_lines = []
    for _, row in top_categories.iterrows():
        category_lines.append(
            f"- 类别 `{row['category']}`："
            f"目标用户数 {int(row['target_users'])}，"
            f"平均模型分数 {row['average_model_score']:.4f}，"
            f"转化率 {row['conversion_rate']:.4f}，"
            f"推荐优先级 {row['recommendation_priority']}。"
        )

    report = f"""# 第五项：业务落地模拟设计报告

## 1. 数据与目标人群

- 预测结果来源：`{source_file}`
- 真实标签列：`{label_col}`
- 模型预测分数列：`{score_col}`
- 全部实验单位数：{total_units}
- 目标人群数：{target_units}
- 目标人群筛选规则：选择预测分数最高的 {TARGET_TOP_RATIO:.0%}
- 入选分数阈值：{threshold:.6f}
- 目标人群历史基准转化率：{baseline_rate:.4%}

本方案以用户为优先实验分流单位。同一用户只进入一个实验组，避免同一用户同时接受不同策略而产生实验污染。

## 2. 模拟A/B测试方案

### 2.1 对照组

对照组维持现有运营策略，不额外使用模型进行优惠券、消息推送或商品推荐干预。

### 2.2 实验组

实验组使用模型预测结果进行精准营销：

1. 优先触达购买概率较高的用户；
2. 推荐与用户近期行为和高潜商品类别匹配的商品；
3. 对临近决策用户提供适度优惠或提醒；
4. 控制触达频率，避免重复推送和用户打扰；
5. 对冷启动用户使用热门商品和相似人群偏好进行补充推荐。

### 2.3 随机分组方式

在目标人群内部，先按照模型预测分数进行分层，再在各层内随机分配：

- 实验组比例：{TEST_RATIO:.0%}
- 对照组比例：{1 - TEST_RATIO:.0%}
- 随机种子：{RANDOM_SEED}

这种方法能够使实验组和对照组的模型分数分布更加一致，减少分组偏差。

## 3. 核心评估指标

### 3.1 主指标

- 7日购买转化率；
- 实验组相对转化提升；
- 增量购买人数；
- 增量收入；
- 增量利润；
- 营销投入产出比ROI。

### 3.2 次级指标

- 点击率；
- 加购率；
- 收藏率；
- 客单价；
- 商品推荐覆盖率；
- 不同用户分层和商品类别的转化率。

### 3.3 护栏指标

正式线上实验还应监控：

- 退订率；
- 投诉率；
- 退款率；
- 推送频次；
- 页面响应时间；
- 用户次日和7日留存率。

如果主指标提升但护栏指标明显恶化，不应直接全量上线。

## 4. 样本量与实验周期

- 显著性水平：{ALPHA}
- 统计功效：{STATISTICAL_POWER}
- 预期相对提升：{EXPECTED_RELATIVE_LIFT:.0%}
- 每组建议最小样本量：{required_sample_per_group}
- 推荐实验周期：{experiment_days}天
- 周期范围：至少{MIN_EXPERIMENT_DAYS}天，最长{MAX_EXPERIMENT_DAYS}天

实验至少覆盖两个完整自然周，减少工作日、周末和促销日差异造成的影响。正式实验期间不得中途频繁修改人群、优惠力度或推荐规则。

## 5. 模拟实验结果

| 指标 | 对照组 | 实验组 |
|---|---:|---:|
| 样本量 | {int(control['sample_size'])} | {int(treatment['sample_size'])} |
| 购买人数 | {int(control['conversions'])} | {int(treatment['conversions'])} |
| 转化率 | {control['conversion_rate']:.4%} | {treatment['conversion_rate']:.4%} |
| 收入 | {control['revenue']:.2f} | {treatment['revenue']:.2f} |
| 营销成本 | {control['marketing_cost']:.2f} | {treatment['marketing_cost']:.2f} |
| 毛利润 | {control['gross_profit']:.2f} | {treatment['gross_profit']:.2f} |

- 绝对转化率提升：{s['absolute_lift']:.4%}
- 相对转化率提升：{s['relative_lift']:.4%}
- 双比例z检验p值：{s['p_value']:.6f}
- 显著性结论：{significance_text}
- 样本量结论：{sample_text}
- 预计增量购买人数：{s['incremental_conversions']:.2f}
- 预计增量收入：{s['incremental_revenue']:.2f}
- 预计增量利润：{s['incremental_profit']:.2f}
- 营销ROI：{s['marketing_roi']:.4f}

以上结果为离线模拟，不等同于真实线上因果效果。最终收益必须通过真实随机A/B测试验证。

## 6. 精准营销优化建议

1. **高分用户优先触达**  
   对模型分数最高的用户优先推送，减少对低意向用户的无效营销费用。

2. **设置差异化优惠力度**  
   高购买概率用户可以使用较小优惠或仅做提醒；中等概率用户可以使用适度优惠；低概率用户不建议高频发券。

3. **控制营销频率**  
   建议对同一用户设置每日和每周触达上限，防止过度推送影响退订率和长期留存。

4. **结合行为阶段设计内容**  
   浏览用户强调商品卖点；加购未购买用户强调库存、价格和时效；收藏用户强调降价提醒和相似商品。

5. **保留长期随机对照组**  
   即使模型上线，也应保留少量长期对照组，用于持续评估模型的真实增量价值。

## 7. 商品推荐优化建议

{chr(10).join(category_lines)}

推荐策略应优先使用模型分数、用户近期行为、商品类别偏好和商品热度进行组合排序。对于没有历史行为的冷启动用户，应使用热门商品、场景化榜单和相似人群偏好，同时限制过度个性化。

## 8. 上线决策规则

建议同时满足以下条件后再扩大流量：

1. 实验达到预先设定的最小样本量和实验周期；
2. 购买转化率提升达到统计显著；
3. 增量利润为正，营销ROI达到业务要求；
4. 退订率、投诉率、退款率等护栏指标没有明显恶化；
5. 在高潜用户、冷启动用户和长尾商品场景下分别检查效果；
6. 先从小流量逐步扩大到全量，避免一次性上线风险。

## 9. 本次输出文件

- `ab_test_user_level_detail.csv`：实验单位明细；
- `ab_test_group_metrics.csv`：实验组与对照组指标；
- `ab_test_effect_summary.csv`：实验效果汇总；
- `score_segment_analysis.csv`：不同模型分数分层结果；
- `category_recommendation_analysis.csv`：商品类别推荐分析；
- `business_ab_test_report.md`：业务落地报告；
- `01_ab_conversion_rate.png`：转化率对比图；
- `02_ab_gross_profit.png`：毛利润对比图；
- `03_score_segment_conversion.png`：分层转化率图。
"""
    output_path.write_text(report, encoding="utf-8")


# ============================================================
# 3. 主程序
# ============================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print("项目根目录：", PROJECT_ROOT)
    print("默认预测输入文件：", DEFAULT_PREDICTION_FILE)
    print("results输出目录：", OUTPUT_DIR)

    print("=" * 70)
    print("开始执行第五项：业务落地模拟设计")
    print("=" * 70)

    source_file = locate_prediction_file(RESULTS_DIR)
    print(f"读取预测结果：{source_file}")

    raw_df, label_col, score_col = load_prediction_data(source_file)
    print(f"真实标签列：{label_col}")
    print(f"模型分数列：{score_col}")
    print(f"有效样本数：{len(raw_df)}")

    (
        unit_df,
        user_col,
        item_col,
        category_col,
        price_col
    ) = get_unit_level_data(raw_df, label_col, score_col)

    print(f"实验分流单位数：{len(unit_df)}")
    if user_col:
        print(f"按用户字段 {user_col} 聚合，避免同一用户进入不同组。")
    else:
        print("未识别到用户字段，当前按每行样本作为实验单位。")

    target_df, threshold = select_target_population(
        unit_df,
        score_col,
        TARGET_TOP_RATIO
    )
    print(f"目标人群数：{len(target_df)}")
    print(f"目标人群分数阈值：{threshold:.6f}")

    ab_df = stratified_ab_assignment(
        target_df,
        score_col,
        TEST_RATIO,
        RANDOM_SEED
    )

    baseline_rate = float(
        ab_df.loc[ab_df["ab_group"] == "control", label_col].mean()
    )
    if not np.isfinite(baseline_rate) or baseline_rate <= 0:
        baseline_rate = float(ab_df[label_col].mean())
    if not np.isfinite(baseline_rate) or baseline_rate <= 0:
        baseline_rate = 0.01

    required_sample_per_group = calculate_required_sample_size(
        baseline_rate,
        EXPECTED_RELATIVE_LIFT,
        ALPHA,
        STATISTICAL_POWER
    )

    total_required = required_sample_per_group * 2
    estimated_daily_units = max(
        int(len(target_df) * DAILY_REACH_RATE),
        1
    )
    estimated_days = math.ceil(total_required / estimated_daily_units)
    experiment_days = int(
        np.clip(
            estimated_days,
            MIN_EXPERIMENT_DAYS,
            MAX_EXPERIMENT_DAYS
        )
    )

    result_df = simulate_ab_outcome(
        ab_df,
        label_col,
        score_col,
        RANDOM_SEED
    )

    metrics = calculate_ab_metrics(result_df, price_col)
    summary = build_summary(
        metrics,
        required_sample_per_group,
        experiment_days
    )
    score_segments = build_score_segment_analysis(
        result_df,
        score_col
    )
    recommendations = build_category_recommendations(
        result_df,
        category_col,
        score_col
    )

    detail_path = OUTPUT_DIR / "ab_test_user_level_detail.csv"
    metrics_path = OUTPUT_DIR / "ab_test_group_metrics.csv"
    summary_path = OUTPUT_DIR / "ab_test_effect_summary.csv"
    segment_path = OUTPUT_DIR / "score_segment_analysis.csv"
    recommendation_path = OUTPUT_DIR / "category_recommendation_analysis.csv"
    report_path = OUTPUT_DIR / "business_ab_test_report.md"

    result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    score_segments.to_csv(segment_path, index=False, encoding="utf-8-sig")
    recommendations.to_csv(
        recommendation_path,
        index=False,
        encoding="utf-8-sig"
    )

    save_plots(metrics, score_segments, OUTPUT_DIR)

    write_markdown_report(
        source_file=source_file,
        label_col=label_col,
        score_col=score_col,
        total_units=len(unit_df),
        target_units=len(target_df),
        threshold=threshold,
        baseline_rate=baseline_rate,
        required_sample_per_group=required_sample_per_group,
        experiment_days=experiment_days,
        metrics=metrics,
        summary=summary,
        recommendations=recommendations,
        output_path=report_path
    )

    print("\nA/B测试分组指标：")
    print(metrics.to_string(index=False))

    print("\n实验效果汇总：")
    print(summary.to_string(index=False))

    expected_outputs = [
        detail_path,
        metrics_path,
        summary_path,
        segment_path,
        recommendation_path,
        report_path,
        OUTPUT_DIR / "01_ab_conversion_rate.png",
        OUTPUT_DIR / "02_ab_gross_profit.png"
    ]

    # 当分层图存在可用数据时，程序会生成第三张图。
    score_segment_plot = OUTPUT_DIR / "03_score_segment_conversion.png"
    if not score_segments.empty:
        expected_outputs.append(score_segment_plot)

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

    print("\n输出目录：")
    print(OUTPUT_DIR)
    print("\n第五项任务已完成。")


if __name__ == "__main__":
    main()
