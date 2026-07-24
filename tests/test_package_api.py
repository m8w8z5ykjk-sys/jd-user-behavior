"""验证工具包公开 API 的完整性。"""

import src.jd_user_behavior as package


def test_all_exports_exist() -> None:
    """__all__ 中列出的名称都必须可以从工具包中访问。"""
    assert package.__all__
    for name in package.__all__:
        assert hasattr(package, name), f"缺少公开接口：{name}"
        assert callable(getattr(package, name)), f"接口不可调用：{name}"


def test_package_version() -> None:
    """工具包应提供非空版本号。"""
    assert package.__version__ == "1.0.0"
