"""example_run.py 的单元测试。"""

import importlib
from unittest.mock import patch

import example_run


def test_import_does_not_execute_pipeline() -> None:
    """重新导入示例文件时，不应自动运行时间特征模块。"""
    with patch("src.jd_user_behavior.run_time_features") as mocked_run:
        importlib.reload(example_run)
    mocked_run.assert_not_called()


def test_main_runs_default_example() -> None:
    """显式调用 main 时，应执行默认的时间特征示例。"""
    with patch.object(example_run, "run_time_features") as mocked_run:
        example_run.main()
    mocked_run.assert_called_once_with()
