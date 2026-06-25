import dask.dataframe as dd
df = dd.read_csv("/Users/chenyixuan/Desktop/京东/data/raw/user_behavior_processed.csv")
print(df.head())
df = df.drop_duplicates()
df = df.dropna()
df.to_parquet("/Users/chenyixuan/Desktop/京东/data/processed/user_behavior_processed.parquet",engine="pyarrow",overwrite=True)
print("并行处理完成！")
