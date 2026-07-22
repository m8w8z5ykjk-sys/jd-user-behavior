"""
文件名称：02_model_comparison.py

功能说明：
1. 读取逻辑回归、XGBoost和LightGBM三个模型的评价指标文件。
2. 提取三个模型在测试集上的Accuracy、Precision、Recall、F1和ROC-AUC。
3. 将三种模型的测试集表现汇总为统一对比表。
4. 以ROC-AUC作为核心指标，对模型进行排序。
5. 绘制三种模型的综合指标对比图。
6. 分别绘制ROC-AUC和F1-score对比图。
7. 自动识别测试集ROC-AUC最高的模型。
8. 保存模型对比表、图表和文字分析报告。

输入文件：
- results/logistic_regression_metrics.csv
- results/xgboost_metrics.csv
- results/lightgbm_metrics.csv

输出文件：
- results/model_comparison.csv
- results/model_comparison_report.md
- results/figures/model_metrics_comparison.png
- results/figures/model_auc_comparison.png
- results/figures/model_f1_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==================================================
# 1. 设置项目路径
# ==================================================

# 当前文件位置：
# 京东/src/analysis/02_model_comparison.py
#
# parents[0] = analysis
# parents[1] = src
# parents[2] = 项目根目录“京东”
BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. 设置三个模型评价结果文件路径
# ==================================================

LOGISTIC_METRICS_PATH = (
    RESULTS_DIR /
    "logistic_regression_metrics.csv"
)

XGBOOST_METRICS_PATH = (
    RESULTS_DIR /
    "xgboost_metrics.csv"
)

LIGHTGBM_METRICS_PATH = (
    RESULTS_DIR /
    "lightgbm_metrics.csv"
)


# ==================================================
# 3. 设置输出文件路径
# ==================================================

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR /
    "model_comparison.csv"
)

REPORT_OUTPUT_PATH = (
    RESULTS_DIR /
    "model_comparison_report.md"
)

METRICS_FIGURE_PATH = (
    FIGURES_DIR /
    "model_metrics_comparison.png"
)

AUC_FIGURE_PATH = (
    FIGURES_DIR /
    "model_auc_comparison.png"
)

F1_FIGURE_PATH = (
    FIGURES_DIR /
    "model_f1_comparison.png"
)


# ==================================================
# 4. 检查三个模型结果文件是否存在
# ==================================================

required_files = {
    "Logistic Regression": LOGISTIC_METRICS_PATH,
    "XGBoost": XGBOOST_METRICS_PATH,
    "LightGBM": LIGHTGBM_METRICS_PATH
}

missing_files = []

for model_name, file_path in required_files.items():
    if not file_path.exists():
        missing_files.append(
            f"{model_name}：{file_path}"
        )

if missing_files:
    missing_text = "\n".join(missing_files)

    raise FileNotFoundError(
        "以下模型评价文件不存在：\n"
        f"{missing_text}\n\n"
        "请先分别运行逻辑回归、XGBoost和LightGBM模型代码，"
        "生成对应的metrics.csv文件。"
    )


# ==================================================
# 5. 定义读取测试集指标的函数
# ==================================================

def read_test_metrics(file_path, final_model_name):
    """
    读取单个模型的评价指标文件，
    并提取测试集Test对应的一行。

    参数：
    file_path：模型指标CSV文件路径。
    final_model_name：最终展示使用的模型名称。

    返回：
    一行测试集评价结果。
    """

    metrics_df = pd.read_csv(file_path)

    required_columns = [
        "Dataset",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metrics_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"文件 {file_path.name} 缺少以下字段："
            f"{missing_columns}"
        )

    test_rows = metrics_df[
        metrics_df["Dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("test")
    ].copy()

    if test_rows.empty:
        raise ValueError(
            f"文件 {file_path.name} 中没有找到"
            f"Dataset=Test的测试集结果。"
        )

    # 如果存在多行Test，默认取最后一行
    # 通常最后一行是Optuna调参后的最终模型结果
    test_row = test_rows.iloc[-1].copy()

    result = {
        "Model": final_model_name,
        "Accuracy": float(test_row["Accuracy"]),
        "Precision": float(test_row["Precision"]),
        "Recall": float(test_row["Recall"]),
        "F1": float(test_row["F1"]),
        "ROC_AUC": float(test_row["ROC_AUC"])
    }

    return result


# ==================================================
# 6. 读取三种模型测试集指标
# ==================================================

print("=" * 70)
print("开始读取三种传统机器学习模型的测试集指标")
print("=" * 70)

logistic_result = read_test_metrics(
    LOGISTIC_METRICS_PATH,
    "Logistic Regression"
)

xgboost_result = read_test_metrics(
    XGBOOST_METRICS_PATH,
    "XGBoost"
)

lightgbm_result = read_test_metrics(
    LIGHTGBM_METRICS_PATH,
    "LightGBM"
)


# ==================================================
# 7. 合并三个模型结果
# ==================================================

comparison_df = pd.DataFrame(
    [
        logistic_result,
        xgboost_result,
        lightgbm_result
    ]
)


# ==================================================
# 8. 检查指标范围
# ==================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]

for column in metric_columns:
    invalid_values = comparison_df[
        ~comparison_df[column].between(
            0,
            1,
            inclusive="both"
        )
    ]

    if not invalid_values.empty:
        raise ValueError(
            f"指标 {column} 中存在不在0到1范围内的数值。"
        )


# ==================================================
# 9. 按照ROC-AUC从高到低排序
# ==================================================

comparison_df = comparison_df.sort_values(
    by="ROC_AUC",
    ascending=False
).reset_index(drop=True)

comparison_df.insert(
    0,
    "Rank",
    range(1, len(comparison_df) + 1)
)


# ==================================================
# 10. 保存模型对比表
# ==================================================

comparison_df.to_csv(
    COMPARISON_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n三种模型测试集对比结果：")
print(
    comparison_df.to_string(
        index=False
    )
)

print("\n模型对比表已保存：")
print(COMPARISON_OUTPUT_PATH)


# ==================================================
# 11. 绘制综合指标对比图
# ==================================================

plot_df = comparison_df.set_index("Model")[
    metric_columns
]

ax = plot_df.plot(
    kind="bar",
    figsize=(12, 7)
)

ax.set_title(
    "Traditional Machine Learning Model Comparison"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1.05
)

ax.tick_params(
    axis="x",
    rotation=0
)

ax.legend(
    title="Metric",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    METRICS_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n综合指标对比图已保存：")
print(METRICS_FIGURE_PATH)


# ==================================================
# 12. 绘制ROC-AUC对比图
# ==================================================

auc_plot_df = comparison_df[
    [
        "Model",
        "ROC_AUC"
    ]
].copy()

ax = auc_plot_df.plot(
    x="Model",
    y="ROC_AUC",
    kind="bar",
    legend=False,
    figsize=(8, 6)
)

ax.set_title(
    "ROC-AUC Comparison"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_ylim(
    0,
    1.05
)

ax.tick_params(
    axis="x",
    rotation=0
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.4f"
    )

plt.tight_layout()

plt.savefig(
    AUC_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nROC-AUC对比图已保存：")
print(AUC_FIGURE_PATH)


# ==================================================
# 13. 绘制F1-score对比图
# ==================================================

f1_plot_df = comparison_df[
    [
        "Model",
        "F1"
    ]
].copy()

ax = f1_plot_df.plot(
    x="Model",
    y="F1",
    kind="bar",
    legend=False,
    figsize=(8, 6)
)

ax.set_title(
    "F1-score Comparison"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "F1-score"
)

ax.set_ylim(
    0,
    1.05
)

ax.tick_params(
    axis="x",
    rotation=0
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.4f"
    )

plt.tight_layout()

plt.savefig(
    F1_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nF1-score对比图已保存：")
print(F1_FIGURE_PATH)


# ==================================================
# 14. 提取最佳模型
# ==================================================

best_row = comparison_df.iloc[0]

best_model_name = best_row["Model"]
best_auc = best_row["ROC_AUC"]
best_accuracy = best_row["Accuracy"]
best_precision = best_row["Precision"]
best_recall = best_row["Recall"]
best_f1 = best_row["F1"]


# ==================================================
# 15. 生成Markdown表格
# ==================================================

markdown_table_lines = [
    "| 排名 | 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |",
    "|---:|---|---:|---:|---:|---:|---:|"
]

for _, row in comparison_df.iterrows():
    markdown_table_lines.append(
        f"| {int(row['Rank'])} "
        f"| {row['Model']} "
        f"| {row['Accuracy']:.6f} "
        f"| {row['Precision']:.6f} "
        f"| {row['Recall']:.6f} "
        f"| {row['F1']:.6f} "
        f"| {row['ROC_AUC']:.6f} |"
    )

markdown_table = "\n".join(
    markdown_table_lines
)


# ==================================================
# 16. 生成模型分析文字
# ==================================================

report_text = f"""# 三种传统机器学习模型性能对比报告

## 1. 对比目的

本报告对比以下三种二分类模型在测试集上的表现：

1. Logistic Regression
2. XGBoost
3. LightGBM

三个模型均完成基础训练，并采用Optuna进行自动超参数调优。
模型调优以验证集ROC-AUC为核心指标，最终使用独立测试集计算
Accuracy、Precision、Recall、F1和ROC-AUC。

---

## 2. 测试集性能对比

{markdown_table}

模型按照测试集ROC-AUC从高到低排序。

---

## 3. 最佳模型

本次测试中，ROC-AUC最高的模型为：

**{best_model_name}**

其测试集指标如下：

| 指标 | 数值 |
|---|---:|
| Accuracy | {best_accuracy:.6f} |
| Precision | {best_precision:.6f} |
| Recall | {best_recall:.6f} |
| F1-score | {best_f1:.6f} |
| ROC-AUC | {best_auc:.6f} |

由于本项目属于类别不平衡二分类问题，因此不能只依据Accuracy选择模型。
本项目将ROC-AUC作为核心评价指标，同时结合Recall和F1-score判断模型
识别未来购买用户的实际能力。

---

## 4. Logistic Regression分析

### 优势

1. 模型结构简单，训练速度较快；
2. 模型参数和特征系数具有较强可解释性；
3. 适合作为传统机器学习基线模型；
4. 在特征与购买概率近似线性相关时具有较稳定表现；
5. 使用类别权重后，可以缓解类别不平衡问题。

### 适用场景

1. 需要快速建立基线模型；
2. 业务方重视模型可解释性；
3. 数据特征数量适中；
4. 特征与目标变量的关系相对简单。

### 局限性

1. 难以自动学习复杂非线性关系；
2. 对复杂特征交互的捕捉能力有限；
3. 对异常值和特征尺度较敏感；
4. 面对复杂用户行为数据时，性能可能低于树模型。

---

## 5. XGBoost分析

### 优势

1. 能够学习复杂非线性关系；
2. 能够自动捕捉特征之间的交互；
3. 对表格型数据通常具有较好的预测表现；
4. 具有正则化机制，可以降低过拟合风险；
5. 可以通过scale_pos_weight处理类别不平衡问题。

### 适用场景

1. 用户行为特征关系复杂；
2. 对预测准确性和排序能力要求较高；
3. 数据规模中等或较大；
4. 可以接受较高的调参和训练成本。

### 局限性

1. 参数数量较多，调参过程较复杂；
2. 训练速度通常慢于逻辑回归；
3. 模型解释难度高于逻辑回归；
4. 参数设置不合理时可能出现过拟合。

---

## 6. LightGBM分析

### 优势

1. 训练速度较快；
2. 内存占用相对较低；
3. 适合大规模、高维表格数据；
4. 能够学习复杂非线性关系；
5. 在许多表格数据任务中具有较高的ROC-AUC。

### 适用场景

1. 样本量较大；
2. 特征数量较多；
3. 对训练速度和内存效率有要求；
4. 需要捕捉复杂用户行为模式。

### 局限性

1. 在小数据集上可能容易过拟合；
2. num_leaves和max_depth等参数需要合理控制；
3. 模型可解释性弱于逻辑回归；
4. 对参数组合较敏感，需要系统调参。

---

## 7. 综合结论

本项目以ROC-AUC作为核心模型选择指标，因为ROC-AUC能够衡量模型区分
购买用户和未购买用户的整体排序能力，并且不依赖单一分类阈值。

根据测试集结果，当前推荐模型为：

**{best_model_name}**

最终模型选择时还应综合考虑：

1. ROC-AUC；
2. Recall；
3. F1-score；
4. 模型训练速度；
5. 模型可解释性；
6. 业务应用成本。

若树模型与逻辑回归的ROC-AUC差异较小，但业务方更重视解释性，
可以选择Logistic Regression；若XGBoost或LightGBM明显提高ROC-AUC和F1，
则更适合作为最终购买预测模型。

---

## 8. 输出文件

本次模型对比生成以下文件：

- `results/model_comparison.csv`
- `results/model_comparison_report.md`
- `results/figures/model_metrics_comparison.png`
- `results/figures/model_auc_comparison.png`
- `results/figures/model_f1_comparison.png`
"""


# ==================================================
# 17. 保存模型对比报告
# ==================================================

REPORT_OUTPUT_PATH.write_text(
    report_text,
    encoding="utf-8"
)

print("\n模型对比报告已保存：")
print(REPORT_OUTPUT_PATH)


# ==================================================
# 18. 输出最终总结
# ==================================================

print("\n" + "=" * 70)
print("三种传统机器学习模型对比全部完成")
print("=" * 70)

print("\nROC-AUC排名：")

for _, row in comparison_df.iterrows():
    print(
        f"第{int(row['Rank'])}名："
        f"{row['Model']}，"
        f"ROC-AUC={row['ROC_AUC']:.6f}"
    )

print("\n当前最佳模型：")
print(best_model_name)

print("\n最佳模型测试集ROC-AUC：")
print(f"{best_auc:.6f}")

print("\n全部输出文件：")
print(COMPARISON_OUTPUT_PATH)
print(REPORT_OUTPUT_PATH)
print(METRICS_FIGURE_PATH)
print(AUC_FIGURE_PATH)
print(F1_FIGURE_PATH)