京东用户行为预测工具包 API 文档

工具包结构

```text
src/jd_user_behavior/
├── __init__.py
└── pipeline.py
```

• pipeline.py：负责定义如何运行各个项目脚本。
• __init__.py：负责公开工具包接口。
• example_run.py：负责演示如何使用工具包。

导入方式

```python
from src.jd_user_behavior import run_time_features
```

常用接口

运行单个脚本

```python
from src.jd_user_behavior import run_script

run_script("features/01_time_features.py")
```

运行时间特征

```python
from src.jd_user_behavior import run_time_features

run_time_features()
```

运行完整特征工程

```python
from src.jd_user_behavior import run_feature_pipeline

run_feature_pipeline()
```

运行三个传统模型

```python
from src.jd_user_behavior import run_traditional_models

run_traditional_models()
```

运行模型融合

```python
from src.jd_user_behavior import run_model_fusion

run_model_fusion()
```

运行A/B测试

```python
from src.jd_user_behavior import run_business_ab_test

run_business_ab_test()
```

运行完整项目

```python
from src.jd_user_behavior import run_full_pipeline

run_full_pipeline()
```

完整流程包含Optuna和深度学习，运行时间可能较长。

运行示例文件

在项目根目录执行：

```bash
python example_run.py
```