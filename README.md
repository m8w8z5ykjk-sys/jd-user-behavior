# 京东用户行为预测项目（JD User Behavior Prediction）

## 项目简介
本项目基于京东用户行为数据，完成从数据清洗、特征工程、传统机器学习、深度学习、模型融合到业务A/B测试的完整机器学习流程。

## 项目目标
预测用户未来7天是否购买商品，并基于预测结果制定精准营销策略。

## 项目结构

```text
data/
 ├── raw/
 └── processed/

results/

src/
 ├── analysis/
 │   ├── 01_exploratory_analysis.py
 │   └── 02_model_comparison.py
 ├── features/
 │   ├── 01_time_features.py
 │   ├── 01_user_lifecycle_features.py
 │   ├── 02_item_lifecycle_features.py
 │   ├── 03_behavior_sequence_features.py
 │   ├── 04_implicit_features.py
 │   ├── 05_business_features.py
 │   ├── 06_feature_preprocessing.py
 │   └── 07_build_model_dataset.py
 └── models/
     ├── 08_logistic_regression.py
     ├── 09_xgboost.py
     ├── 10_lightgbm.py
     ├── 11_deep_learning_models.py
     ├── 12_model_fusion_explainability.py
     └── 13_business_ab_test.py
```

## 项目流程
1. 数据加载与清洗
2. 特征工程
3. 特征预处理
4. 构建训练/验证/测试集
5. Logistic Regression
6. XGBoost
7. LightGBM
8. LSTM / GRU / DIN
9. Stacking模型融合
10. SHAP可解释性分析
11. A/B测试模拟

## 输入数据
- data/raw/user_behavior.csv

## 中间数据
- data/processed/

## 输出结果
- results/

## 主要技术
- Pandas
- NumPy
- PyArrow
- Scikit-learn
- XGBoost
- LightGBM
- PyTorch
- Optuna
- SHAP
- Matplotlib
- SQLite

## 运行顺序

```text
load_data.py
parallel_read.py
clean_data.py
build_feature_table.py

01_time_features.py
01_user_lifecycle_features.py
02_item_lifecycle_features.py
03_behavior_sequence_features.py
04_implicit_features.py
05_business_features.py
06_feature_preprocessing.py
07_build_model_dataset.py

08_logistic_regression.py
09_xgboost.py
10_lightgbm.py
11_deep_learning_models.py
12_model_fusion_explainability.py
13_business_ab_test.py

01_exploratory_analysis.py
02_model_comparison.py
```

## 作者
Chen Yixuan
