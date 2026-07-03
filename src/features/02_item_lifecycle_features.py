import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
print(BASE_DIR)
INPUT_FILE = BASE_DIR / "data" / "processed" / "user_behavior_cleaned.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(INPUT_FILE)
print("数据读取成功")
print(df.head())
print(df.info())
print(df.columns)
df["date"] = df["time"].dt.date
df["date"] = pd.to_datetime(df["date"])
# 1. 每个商品总行为次数
item_total_behavior = (df.groupby("item_id").size().reset_index(name="item_total_behavior_count"))
print(item_total_behavior.head())
# 2. 最近7天开始日期
max_date = df["date"].max()
recent_7_days = max_date - pd.Timedelta(days=7)
print("数据最后一天：", max_date)
print("最近7天开始：", recent_7_days)
# 3. 最近7天商品行为
df_recent_7 = df[df["date"] >= recent_7_days]
print(df_recent_7.head())
# 4. 最近7天商品行为次数
item_recent_7_behavior = (df_recent_7.groupby("item_id").size().reset_index(name="item_recent_7_behavior_count"))
print(item_recent_7_behavior.head())
# 5. 合并商品总行为和最近7天行为
item_lifecycle_features = item_total_behavior.merge(item_recent_7_behavior,on="item_id",how="left")
print(item_lifecycle_features.head())
item_lifecycle_features["item_recent_7_behavior_count"] = item_lifecycle_features["item_recent_7_behavior_count"].fillna(0)
# 6. 计算热度衰减系数
item_lifecycle_features["heat_decay_score"] = (item_lifecycle_features["item_recent_7_behavior_count"]/ item_lifecycle_features["item_total_behavior_count"])
print(item_lifecycle_features.head())
# 7. 每个商品所属类目
item_category = (df.groupby("item_id")["item_category"].first().reset_index(name="item_category"))
print(item_category.head())
item_lifecycle_features = item_lifecycle_features.merge(item_category,on="item_id",how="left")
print(item_lifecycle_features.head())
# 8. 类目竞争度
category_item_count = (item_lifecycle_features.groupby("item_category")["item_id"].nunique().reset_index(name="category_item_count"))
print(category_item_count.head())
item_lifecycle_features = item_lifecycle_features.merge(category_item_count,on="item_category",how="left")
item_lifecycle_features["category_competition_degree"] = item_lifecycle_features["category_item_count"]
print(item_lifecycle_features.head())
# 9. 保存商品生命周期特征
ITEM_LIFECYCLE_OUTPUT = OUTPUT_DIR / "item_lifecycle_features.parquet"
item_lifecycle_features.to_parquet(ITEM_LIFECYCLE_OUTPUT,index=False)
print("商品生命周期特征已保存：")
print(ITEM_LIFECYCLE_OUTPUT)