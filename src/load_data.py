"""
文件名称：load_data.py

任务内容：
1. 配置并检查 Pandas、NumPy、Dask、PyArrow 等大数据分析环境；
2. 使用 Pandas 的 chunksize 参数对原始用户行为 CSV 数据进行分块读取；
3. 展示第一个数据块，用于检查文件路径、字段结构和分块加载功能；
4. 为后续并行读取、数据清洗和特征工程提供原始数据加载基础。

输入文件：
- /Users/chenyixuan/Desktop/京东/data/raw/user_behavior_processed.csv
  原始用户行为数据，主要字段包括 user_id、item_id、behavior_type、
  item_category、time。

输出文件：
- 本脚本不生成新的数据文件。

控制台输出：
- 环境配置结果；
- 第一个数据块的前5行数据。
"""

import pandas as pd
import numpy as np
import dask
import pyarrow
print("环境配置成功")
import pandas as pd
for chunk in pd.read_csv("/Users/chenyixuan/Desktop/京东/data/raw/user_behavior_processed.csv",
                         chunksize=1000):
    print(chunk.head())
    break