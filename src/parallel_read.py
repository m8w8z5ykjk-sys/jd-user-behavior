"""
文件名称：parallel_read.py

任务内容：
1. 使用 Dask 并行读取大规模用户行为 CSV 数据；
2. 通过分区计算降低一次性加载全部数据造成的内存压力；
3. 删除重复记录和包含缺失值的记录；
4. 将处理后的数据保存为 Parquet 格式，提高后续读写效率。

输入文件：
- /Users/chenyixuan/Desktop/京东/data/raw/user_behavior_processed.csv
  原始用户行为 CSV 数据。

输出文件：
- /Users/chenyixuan/Desktop/京东/data/processed/user_behavior_processed.parquet
  去重并处理缺失值后生成的 Parquet 数据集。
  Dask 通常会将该路径保存为包含多个分区文件的目录。

控制台输出：
- 数据前5行；
- 并行处理完成提示。
"""

import dask.dataframe as dd
df = dd.read_csv("/Users/chenyixuan/Desktop/京东/data/raw/user_behavior_processed.csv")
print(df.head())
df = df.drop_duplicates()
df = df.dropna()
df.to_parquet("/Users/chenyixuan/Desktop/京东/data/processed/user_behavior_processed.parquet",engine="pyarrow",overwrite=True)
print("并行处理完成！")
