"""京东用户行为预测项目的统一流水线接口。

本模块不重复实现数据处理或模型算法，而是将 ``src`` 目录中已有的
Python 脚本封装为可调用函数。使用者既可以单独运行某个模块，也可以
按照数据处理、特征工程、模型训练等阶段批量运行。

主要功能：
1. 校验目标脚本是否存在；
2. 使用 :mod:`runpy` 以主程序方式执行目标脚本；
3. 提供分阶段及完整项目流水线接口。

示例：
    from src.jd_user_behavior import run_time_features
    run_time_features()
"""

from pathlib import Path
import runpy

# pipeline.py 位于“项目根目录/src/jd_user_behavior/”，因此向上两级
# 可以得到项目根目录。所有脚本路径都以 PROJECT_ROOT/src 为基准。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"


def run_script(relative_path: str) -> None:
    """运行 ``src`` 目录中的指定 Python 脚本。

    Args:
        relative_path: 相对于 ``src`` 目录的脚本路径，例如
            ``"features/01_time_features.py"``。

    Raises:
        TypeError: ``relative_path`` 不是字符串时抛出。
        ValueError: 路径为空、不是 ``.py`` 文件或试图跳出 ``src`` 目录时抛出。
        FileNotFoundError: 目标脚本不存在时抛出。

    Returns:
        None。目标脚本产生的文件、日志和模型结果由脚本自身负责。
    """
    if not isinstance(relative_path, str):
        raise TypeError("relative_path必须是字符串。")

    relative_path = relative_path.strip()
    if not relative_path:
        raise ValueError("relative_path不能为空。")
    if Path(relative_path).suffix.lower() != ".py":
        raise ValueError("只能运行.py脚本。")

    # resolve() 后再次检查父目录，防止使用 ../ 访问 src 目录之外的文件。
    script_path = (SRC_DIR / relative_path).resolve()
    src_dir_resolved = SRC_DIR.resolve()
    if script_path != src_dir_resolved and src_dir_resolved not in script_path.parents:
        raise ValueError("脚本路径必须位于src目录中。")

    if not script_path.is_file():
        raise FileNotFoundError(f"找不到需要运行的文件：{script_path}")

    print("=" * 70)
    print(f"开始运行：{relative_path}")
    print("=" * 70)

    # run_name="__main__" 使被调用脚本的行为与直接执行该脚本一致。
    runpy.run_path(str(script_path), run_name="__main__")

    print(f"运行完成：{relative_path}")


# ============================================================
# 1. 数据处理接口
# ============================================================

def run_load_data() -> None:
    """运行原始数据加载模块。"""
    run_script("load_data.py")


def run_parallel_read() -> None:
    """运行大文件分块并行读取模块。"""
    run_script("parallel_read.py")


def run_clean_data() -> None:
    """运行缺失值、重复值及时间字段清洗模块。"""
    run_script("clean_data.py")


def run_build_feature_table() -> None:
    """运行基础特征表构建模块。"""
    run_script("build_feature_table.py")


def run_sqlite_analysis() -> None:
    """运行 SQLite 数据查询与分析模块。"""
    run_script("sqlite_analysis.py")


def run_data_pipeline() -> None:
    """依次运行并行读取、数据清洗和基础特征表构建。"""
    run_parallel_read()
    run_clean_data()
    run_build_feature_table()


# ============================================================
# 2. 特征工程接口
# ============================================================

def run_time_features() -> None:
    """构建小时、日期、星期等时间特征。"""
    run_script("features/01_time_features.py")


def run_user_lifecycle_features() -> None:
    """构建用户生命周期特征（仅在项目保留该模块时使用）。"""
    run_script("features/01_user_lifecycle_features.py")


def run_item_lifecycle_features() -> None:
    """构建商品生命周期及活跃周期特征。"""
    run_script("features/02_item_lifecycle_features.py")


def run_behavior_sequence_features() -> None:
    """构建用户行为顺序及历史序列特征。"""
    run_script("features/03_behavior_sequence_features.py")


def run_implicit_features() -> None:
    """使用矩阵分解方法构建 SVD 隐式特征。"""
    run_script("features/04_implicit_features.py")


def run_business_features() -> None:
    """构建转化、偏好和用户价值等业务特征。"""
    run_script("features/05_business_features.py")


def run_feature_preprocessing() -> None:
    """运行缺失值处理、编码和标准化等特征预处理。"""
    run_script("features/06_feature_preprocessing.py")


def run_build_model_dataset() -> None:
    """连接主要特征并构建最终模型训练数据集。"""
    run_script("features/07_build_model_dataset.py")


def run_feature_pipeline() -> None:
    """按照项目顺序运行全部主要特征工程模块。"""
    run_time_features()
    run_item_lifecycle_features()
    run_behavior_sequence_features()
    run_implicit_features()
    run_business_features()
    run_feature_preprocessing()
    run_build_model_dataset()


# ============================================================
# 3. 传统机器学习接口
# ============================================================

def run_logistic_regression() -> None:
    """训练、调优并评估逻辑回归模型。"""
    run_script("models/08_logistic_regression.py")


def run_xgboost() -> None:
    """训练、调优并评估 XGBoost 模型。"""
    run_script("models/09_xgboost.py")


def run_lightgbm() -> None:
    """训练、调优并评估 LightGBM 模型。"""
    run_script("models/10_lightgbm.py")


def run_traditional_models() -> None:
    """依次运行逻辑回归、XGBoost 和 LightGBM。"""
    run_logistic_regression()
    run_xgboost()
    run_lightgbm()


# ============================================================
# 4. 深度学习、模型融合与业务分析接口
# ============================================================

def run_deep_learning_models() -> None:
    """运行 LSTM、GRU 和 DIN 深度学习模型。"""
    run_script("models/11_deep_learning_models.py")


def run_model_fusion() -> None:
    """运行 Stacking 模型融合与可解释性分析。"""
    run_script("models/12_model_fusion_explainability.py")


def run_business_ab_test() -> None:
    """运行推荐策略业务 A/B 测试模拟。"""
    run_script("models/13_business_ab_test.py")


def run_exploratory_analysis() -> None:
    """运行探索性数据分析并输出统计图表。"""
    run_script("analysis/01_exploratory_analysis.py")


def run_model_comparison() -> None:
    """汇总并比较不同模型的评价指标。"""
    run_script("analysis/02_model_comparison.py")


# ============================================================
# 5. 完整项目流程
# ============================================================

def run_full_pipeline() -> None:
    """依次运行数据、特征、模型融合和业务分析的完整流程。

    提醒：该函数会执行多个模型，深度学习和 Optuna 调参可能耗时较长。
    探索性分析和模型对比属于结果展示模块，可根据需要单独调用。
    """
    run_data_pipeline()
    run_feature_pipeline()
    run_traditional_models()
    run_deep_learning_models()
    run_model_fusion()
    run_business_ab_test()
