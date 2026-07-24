"""
文件名称：example_run.py

功能：
演示如何调用京东用户行为预测工具包。
实际使用时，只保留需要运行的函数即可。
"""

from src.jd_user_behavior import (
    run_time_features,
    run_feature_pipeline,
    run_logistic_regression,
    run_traditional_models,
    run_model_fusion,
    run_business_ab_test,
)


# ============================================================
# 示例1：运行单个特征工程模块
# ============================================================

run_time_features()


# ============================================================
# 下面是其他调用示例
# 需要运行时，删除对应代码前面的#
# ============================================================

# 示例2：运行完整特征工程流程
# run_feature_pipeline()

# 示例3：只运行逻辑回归模型
# run_logistic_regression()

# 示例4：运行三个传统机器学习模型
# run_traditional_models()

# 示例5：运行模型融合与可解释性分析
# run_model_fusion()

# 示例6：运行业务A/B测试模拟
# run_business_ab_test()