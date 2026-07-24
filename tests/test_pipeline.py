"""pipeline.py 的单元测试。

测试使用 mock 替代真实脚本运行，因此不会读取大数据、训练模型或写入结果。
"""

import runpy
from pathlib import Path
from unittest.mock import DEFAULT, Mock, call, patch

import pytest

from src.jd_user_behavior import pipeline


def test_run_script_executes_existing_python_file(tmp_path: Path, monkeypatch, capsys) -> None:
    """存在的 Python 文件应通过 runpy.run_path 以 __main__ 方式执行。"""
    script = tmp_path / "demo.py"
    script.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "SRC_DIR", tmp_path)

    with patch.object(runpy, "run_path") as mocked_run_path:
        pipeline.run_script("demo.py")

    mocked_run_path.assert_called_once_with(str(script.resolve()), run_name="__main__")
    output = capsys.readouterr().out
    assert "开始运行：demo.py" in output
    assert "运行完成：demo.py" in output


@pytest.mark.parametrize("bad_path", ["", "   ", "data.csv", "README.md"])
def test_run_script_rejects_invalid_path(bad_path: str) -> None:
    """空路径和非 Python 文件应被拒绝。"""
    with pytest.raises(ValueError):
        pipeline.run_script(bad_path)


def test_run_script_rejects_non_string_path() -> None:
    """路径参数必须是字符串。"""
    with pytest.raises(TypeError):
        pipeline.run_script(123)  # type: ignore[arg-type]


def test_run_script_rejects_parent_directory_escape(tmp_path: Path, monkeypatch) -> None:
    """不允许通过 ../ 运行 src 目录外部的脚本。"""
    monkeypatch.setattr(pipeline, "SRC_DIR", tmp_path / "src")
    with pytest.raises(ValueError, match="必须位于src目录"):
        pipeline.run_script("../outside.py")


def test_run_script_raises_when_file_is_missing(tmp_path: Path, monkeypatch) -> None:
    """目标脚本不存在时应给出明确异常。"""
    monkeypatch.setattr(pipeline, "SRC_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="找不到需要运行的文件"):
        pipeline.run_script("missing.py")


def test_single_module_wrapper_uses_expected_path() -> None:
    """单模块接口应把正确的相对路径交给 run_script。"""
    with patch.object(pipeline, "run_script") as mocked_run_script:
        pipeline.run_time_features()
    mocked_run_script.assert_called_once_with("features/01_time_features.py")


def test_data_pipeline_order() -> None:
    """数据流水线必须按照读取、清洗、特征表的顺序运行。"""
    with (
        patch.object(pipeline, "run_parallel_read") as parallel,
        patch.object(pipeline, "run_clean_data") as clean,
        patch.object(pipeline, "run_build_feature_table") as build,
    ):
        # attach_mock 用于统一记录不同 mock 的调用顺序。
        ordered = Mock()
        ordered.attach_mock(parallel, "parallel")
        ordered.attach_mock(clean, "clean")
        ordered.attach_mock(build, "build")
        pipeline.run_data_pipeline()

    assert ordered.mock_calls == [call.parallel(), call.clean(), call.build()]


def test_feature_pipeline_order() -> None:
    """特征流水线应按预定顺序调用七个核心模块。"""
    names = [
        "run_time_features",
        "run_item_lifecycle_features",
        "run_behavior_sequence_features",
        "run_implicit_features",
        "run_business_features",
        "run_feature_preprocessing",
        "run_build_model_dataset",
    ]
    with patch.multiple(pipeline, **{name: DEFAULT for name in names}) as mocks:
        pipeline.run_feature_pipeline()

    for name in names:
        mocks[name].assert_called_once_with()


def test_full_pipeline_calls_each_major_stage_once() -> None:
    """完整流水线应调用六个主要阶段，且不真正训练模型。"""
    names = [
        "run_data_pipeline",
        "run_feature_pipeline",
        "run_traditional_models",
        "run_deep_learning_models",
        "run_model_fusion",
        "run_business_ab_test",
    ]
    with patch.multiple(pipeline, **{name: DEFAULT for name in names}) as mocks:
        pipeline.run_full_pipeline()

    for name in names:
        mocks[name].assert_called_once_with()
