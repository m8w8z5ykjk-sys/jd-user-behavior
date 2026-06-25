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