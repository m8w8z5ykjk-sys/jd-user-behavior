"""
文件名称：pipeline.py

功能：
1. 对项目现有数据处理、特征工程、模型训练和业务分析脚本进行统一封装；
2. 提供可重复调用的Python API；
3. 保留原有脚本和目录，不对现有代码进行大规模修改；
4. 支持单独运行某个模块或按阶段运行完整流程。

使用示例：
from jd_user_behavior import run_data_pipeline

run_data_pipeline()
"""

from pathlib import Path
import runpy


# pipeline.py位于：项目根目录/src/jd_user_behavior/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"


def run_script(relative_path: str) -> None:
    """
    运行src目录中的指定Python脚本。

    参数：
        relative_path：
        相对于src目录的Python文件路径，
        例如 features/01_time_features.py。
    """
    script_path = SRC_DIR / relative_path

    if not script_path.exists():
        raise FileNotFoundError(
            f"找不到需要运行的文件：{script_path}"
        )

    print("=" * 70)
    print(f"开始运行：{relative_path}")
    print("=" * 70)

    runpy.run_path(
        str(script_path),
        run_name="__main__"
    )

    print(f"运行完成：{relative_path}")


# ============================================================
# 1. 数据处理
# ============================================================

def run_load_data() -> None:
    """运行数据加载模块。"""
    run_script("load_data.py")


def run_parallel_read() -> None:
    """运行分块并行读取模块。"""
    run_script("parallel_read.py")


def run_clean_data() -> None:
    """运行数据清洗模块。"""
    run_script("clean_data.py")


def run_build_feature_table() -> None:
    """运行基础特征表构建模块。"""
    run_script("build_feature_table.py")


def run_sqlite_analysis() -> None:
    """运行SQLite分析模块。"""
    run_script("sqlite_analysis.py")


def run_data_pipeline() -> None:
    """按顺序运行主要数据处理流程。"""
    run_parallel_read()
    run_clean_data()
    run_build_feature_table()


# ============================================================
# 2. 特征工程
# ============================================================

def run_time_features() -> None:
    """构建时间特征。"""
    run_script("features/01_time_features.py")


def run_user_lifecycle_features() -> None:
    """构建用户生命周期特征。"""
    run_script("features/01_user_lifecycle_features.py")


def run_item_lifecycle_features() -> None:
    """构建商品生命周期特征。"""
    run_script("features/02_item_lifecycle_features.py")


def run_behavior_sequence_features() -> None:
    """构建行为序列特征。"""
    run_script("features/03_behavior_sequence_features.py")


def run_implicit_features() -> None:
    """构建SVD隐式特征。"""
    run_script("features/04_implicit_features.py")


def run_business_features() -> None:
    """构建业务导向特征。"""
    run_script("features/05_business_features.py")


def run_feature_preprocessing() -> None:
    """运行特征预处理。"""
    run_script("features/06_feature_preprocessing.py")


def run_build_model_dataset() -> None:
    """构建模型训练数据集。"""
    run_script("features/07_build_model_dataset.py")


def run_feature_pipeline() -> None:
    """按顺序运行全部主要特征工程模块。"""
    run_time_features()
    run_item_lifecycle_features()
    run_behavior_sequence_features()
    run_implicit_features()
    run_business_features()
    run_feature_preprocessing()
    run_build_model_dataset()


# ============================================================
# 3. 传统机器学习模型
# ============================================================

def run_logistic_regression() -> None:
    """训练和调优逻辑回归模型。"""
    run_script("models/08_logistic_regression.py")


def run_xgboost() -> None:
    """训练和调优XGBoost模型。"""
    run_script("models/09_xgboost.py")


def run_lightgbm() -> None:
    """训练和调优LightGBM模型。"""
    run_script("models/10_lightgbm.py")


def run_traditional_models() -> None:
    """按顺序运行三个传统机器学习模型。"""
    run_logistic_regression()
    run_xgboost()
    run_lightgbm()


# ============================================================
# 4. 深度学习、模型融合与业务分析
# ============================================================

def run_deep_learning_models() -> None:
    """运行LSTM、GRU和DIN模型。"""
    run_script("models/11_deep_learning_models.py")


def run_model_fusion() -> None:
    """运行Stacking模型融合与可解释性分析。"""
    run_script("models/12_model_fusion_explainability.py")


def run_business_ab_test() -> None:
    """运行业务A/B测试模拟。"""
    run_script("models/13_business_ab_test.py")


def run_exploratory_analysis() -> None:
    """运行探索性数据分析。"""
    run_script("analysis/01_exploratory_analysis.py")


def run_model_comparison() -> None:
    """运行模型性能对比分析。"""
    run_script("analysis/02_model_comparison.py")


# ============================================================
# 5. 完整项目流程
# ============================================================

def run_full_pipeline() -> None:
    """
    运行完整项目流程。

    注意：
    深度学习和Optuna调参可能运行时间较长。
    """
    run_data_pipeline()
    run_feature_pipeline()
    run_traditional_models()
    run_deep_learning_models()
    run_model_fusion()
    run_business_ab_test()