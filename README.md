京东用户行为预测项目（JD User Behavior Prediction）

一、项目简介

本项目基于京东用户行为数据，完成从数据加载、数据清洗、特征工程、模型训练、模型融合、模型解释到业务 A/B 测试的完整机器学习流程。

项目将主要处理流程封装为可复用的 Python 接口，便于按照统一方式运行不同的数据处理和模型模块。

二、项目目标

根据用户历史浏览、收藏、加购和购买行为，预测用户未来 7 天内是否会购买商品，并根据预测结果辅助制定精准营销策略。

三、项目主要功能

1. 大规模用户行为数据加载与清洗。
2. 用户、商品、时间和行为序列特征构建。
3. 类别型特征编码和数值型特征标准化。
4. 按时间窗口划分训练集、验证集和测试集。
5. 构建 Logistic Regression、XGBoost 和 LightGBM 模型。
6. 构建 LSTM、GRU 和 DIN 深度学习模型。
7. 使用 Stacking 进行模型融合。
8. 使用 SHAP 分析模型特征重要性。
9. 模拟业务 A/B 测试并评估营销效果。
10. 通过统一 API 调用项目中的主要功能。

四、项目目录结构

```text
jd-user-behavior/
├── data/
│   ├── raw/                         # 原始数据，不建议上传到 GitHub
│   └── processed/                   # 清洗后数据和中间特征数据
├── docs/                            # 项目说明和报告文档
├── results/                         # 图表、模型指标和分析结果
├── src/
│   ├── analysis/
│   │   ├── 01_exploratory_analysis.py
│   │   └── 02_model_comparison.py
│   ├── features/
│   │   ├── 01_time_features.py
│   │   ├── 01_user_lifecycle_features.py
│   │   ├── 02_item_lifecycle_features.py
│   │   ├── 03_behavior_sequence_features.py
│   │   ├── 04_implicit_features.py
│   │   ├── 05_business_features.py
│   │   ├── 06_feature_preprocessing.py
│   │   └── 07_build_model_dataset.py
│   ├── models/
│   │   ├── 08_logistic_regression.py
│   │   ├── 09_xgboost.py
│   │   ├── 10_lightgbm.py
│   │   ├── 11_deep_learning_models.py
│   │   ├── 12_model_fusion_explainability.py
│   │   └── 13_business_ab_test.py
│   └── jd_user_behavior/
│       ├── __init__.py              # 对外公开统一 API
│       └── pipeline.py              # 各模块运行接口
├── example_run.py                   # API 使用示例
├── requirements.txt                 # Python 依赖库
├── .gitignore                       # Git 忽略文件配置
└── README.md                        # 项目说明文档
```

> 注意：请根据 GitHub 仓库中的实际文件名称和目录结构进行调整。如果项目中没有某个文件，可以删除对应内容。

五、输入和输出文件

1. 输入文件

主要原始输入文件：

```text
data/raw/user_behavior.csv
```

常用字段包括：

|字段名          |含义    |
|-------------|------|
|user_id      |用户编号  |
|item_id      |商品编号  |
|behavior_type|用户行为类型|
|item_category|商品类别  |
|time         |行为发生时间|

行为类型通常包括浏览、收藏、加购和购买。实际编码方式应以原始数据说明为准。

2. 中间文件

数据清洗、特征计算和模型数据集构建过程中产生的文件保存在：

```text
data/processed/
```

例如：

```text
data/processed/user_behavior_cleaned.parquet
data/processed/time_features.parquet
data/processed/item_lifecycle_features.parquet
data/processed/behavior_sequence_features.parquet
data/processed/implicit_features.parquet
data/processed/business_features.parquet
data/processed/model_dataset.parquet
```

3. 输出文件

模型指标、图表、特征重要性和业务分析结果保存在：

```text
results/
```

输出内容主要包括：

• 模型评价指标。
• ROC 曲线和 PR 曲线。
• 混淆矩阵。
• 模型对比结果。
• SHAP 特征重要性图。
• 模型融合结果。
• A/B 测试模拟结果。

六、主要技术与依赖库

本项目主要使用以下 Python 库：

• Pandas
• NumPy
• PyArrow
• Scikit-learn
• XGBoost
• LightGBM
• PyTorch
• Optuna
• SHAP
• Matplotlib
• Joblib
• SQLite3

其中，SQLite3 通常随 Python 一起安装，不需要单独安装。

七、环境安装

1. 创建虚拟环境

```bash
python -m venv .venv
```

2. 激活虚拟环境

Windows：

```bash
.venv\Scripts\activate
```

macOS 或 Linux：

```bash
source .venv/bin/activate
```

3. 安装依赖库

```bash
pip install -r requirements.txt
```

八、项目运行顺序

建议按照以下顺序运行项目文件。

第一阶段：数据加载与清洗

```text
load_data.py
parallel_read.py
clean_data.py
build_feature_table.py
```

第二阶段：特征工程

```text
01_time_features.py
01_user_lifecycle_features.py
02_item_lifecycle_features.py
03_behavior_sequence_features.py
04_implicit_features.py
05_business_features.py
06_feature_preprocessing.py
07_build_model_dataset.py
```

第三阶段：模型训练与分析

```text
08_logistic_regression.py
09_xgboost.py
10_lightgbm.py
11_deep_learning_models.py
12_model_fusion_explainability.py
13_business_ab_test.py
```

第四阶段：结果分析

```text
01_exploratory_analysis.py
02_model_comparison.py
```

九、Python 工具包使用方法

项目将数据处理、特征工程、模型训练、模型融合和业务分析流程封装为统一 Python 接口。

1. 导入 API

```python
from src.jd_user_behavior import (
    run_time_features,
    run_item_lifecycle_features,
    run_behavior_sequence_features,
    run_implicit_features,
    run_business_features,
    run_feature_preprocessing,
    run_build_model_dataset,
    run_logistic_regression,
    run_xgboost,
    run_lightgbm,
    run_deep_learning,
    run_model_fusion,
    run_business_ab_test,
)
```

2. 运行单个模块

运行时间特征模块：

```python
from src.jd_user_behavior import run_time_features

run_time_features()
```

运行 XGBoost 模型：

```python
from src.jd_user_behavior import run_xgboost

run_xgboost()
```

运行模型融合与可解释性分析：

```python
from src.jd_user_behavior import run_model_fusion

run_model_fusion()
```

3. 使用示例文件

项目根目录中的 example_run.py 用于展示 API 的调用方式。

运行命令：

```bash
python example_run.py
```

十、API 说明

|API 名称                          |功能             |
|--------------------------------|---------------|
|run_time_features()             |构建时间特征         |
|run_item_lifecycle_features()   |构建商品生命周期特征     |
|run_behavior_sequence_features()|构建用户行为序列特征     |
|run_implicit_features()         |构建隐式交互特征       |
|run_business_features()         |构建业务特征         |
|run_feature_preprocessing()     |执行特征编码、填充和标准化  |
|run_build_model_dataset()       |构建模型训练数据集      |
|run_logistic_regression()       |训练逻辑回归模型       |
|run_xgboost()                   |训练 XGBoost 模型  |
|run_lightgbm()                  |训练 LightGBM 模型 |
|run_deep_learning()             |训练深度学习模型       |
|run_model_fusion()              |执行模型融合和 SHAP 分析|
|run_business_ab_test()          |执行业务 A/B 测试模拟  |

> API 名称必须与 `src/jd_user_behavior/__init__.py` 和 `pipeline.py` 中的实际函数名称保持一致。不存在的函数需要从表格和示例代码中删除。

十一、模型评价指标

项目主要使用以下指标评价模型：

• Accuracy：准确率。
• Precision：精确率。
• Recall：召回率。
• F1-score：精确率和召回率的综合指标。
• ROC-AUC：模型区分正负样本的能力。
• PR-AUC：适合评价类别不平衡数据。

由于购买样本通常远少于未购买样本，因此项目重点关注 Recall、F1-score、ROC-AUC 和 PR-AUC，而不是只关注 Accuracy。

十二、注意事项

1. 原始数据和较大的中间数据文件不建议上传到 GitHub。
2. 请先确认输入文件路径与代码中的路径一致。
3. 运行模型代码前，需要先完成数据清洗、特征工程和模型数据集构建。
4. 深度学习、Optuna 调参和 SHAP 分析可能需要较长运行时间。
5. 如果电脑配置较低，可以减少样本量、训练轮数和调参次数。
6. 所有输出文件应统一保存在 results/ 目录中。
7. 所有中间数据应统一保存在 data/processed/ 目录中。
8. 提交 GitHub 前，应检查 .gitignore，避免上传原始数据、模型文件和缓存文件。

十三、作者

Chen Yixuan

十四、项目状态

本项目已完成主要的数据处理、特征工程、模型训练、模型融合、模型解释和业务分析流程，后续可继续优化模型参数、运行效率和 API 封装方式。