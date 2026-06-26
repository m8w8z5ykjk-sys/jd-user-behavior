import pandas as pd
df = pd.read_csv("../data/processed/user_behavior_cleaned.csv")
print(df.head())
df.info()
#压缩减少数据内存
df["user_id"] = df["user_id"].astype("int32")
df["item_id"] = df["item_id"].astype("int32")
df["item_category"] = df["item_category"].astype("int32")
df["behavior_type"] = df["behavior_type"].astype("int8")
df["time"] = pd.to_datetime(df["time"])
df.info()
#将数据转换为parquet格式提升读写效率
df.to_parquet("../data/processed/user_behavior_cleaned.parquet",index=False)
print("保存成功！")
#验证是否保存成功
df2 = pd.read_parquet("../data/processed/user_behavior_cleaned.parquet")
print(df2.head())

#######################
######用户维度中间表######
#######################
# 先提取日期
df["date"] = df["time"].dt.date
user_table = (
    df.groupby("user_id")
      .agg(
          action_count=("behavior_type", "count"),      # 总行为次数
          browse_count=("behavior_type", lambda x: (x == 1).sum()),   # 浏览次数
          fav_count=("behavior_type", lambda x: (x == 2).sum()),      # 收藏次数
          cart_count=("behavior_type", lambda x: (x == 3).sum()),     # 加购次数
          buy_count=("behavior_type", lambda x: (x == 4).sum()),      # 购买次数
          active_days=("date", "nunique"),            # 活跃天数
          first_active_time=("time", "min"),           # 第一次活跃时间
          last_active_time=("time", "max"),            # 最后一次活跃时间
          unique_item_count=("item_id", "nunique"),    # 互动过的商品数
          unique_category_count=("item_category", "nunique") # 互动过的品类数
      )
      .reset_index()
)

# 转化率：购买次数 / 总行为次数
user_table["conversion_rate"] = user_table["buy_count"] / user_table["action_count"]
print(user_table.head())
user_table.to_parquet("../data/processed/user_table.parquet",index=False)
print("用户中间表保存完成")

#######################
######商品维度中间表######
#######################
df["is_browse"] = (df["behavior_type"] == 1).astype("int8")
df["is_fav"] = (df["behavior_type"] == 2).astype("int8")
df["is_cart"] = (df["behavior_type"] == 3).astype("int8")
df["is_buy"] = (df["behavior_type"] == 4).astype("int8")
item_table = (
    df.groupby("item_id")
      .agg(
          item_action_count=("behavior_type", "count"),          # 商品总行为次数
          item_browse_count=("is_browse","sum"),  # 被浏览次数
          item_fav_count=("is_fav","sum"),     # 被收藏次数
          item_cart_count=("is_cart","sum"),    # 被加购次数
          item_buy_count=("is_buy","sum"),     # 被购买次数
          buyer_count=("user_id", "nunique"),                 # 互动过这个商品的用户数
          item_active_days=("date", "nunique"),                # 商品被互动的天数
          item_category=("item_category", "first")              # 商品所属类别
      )
      .reset_index()
)

# 商品转化率：购买次数 / 总行为次数
item_table["item_conversion_rate"] = (
    item_table["item_buy_count"] / item_table["item_action_count"]
)
print(item_table.head())
item_table.to_parquet("../data/processed/item_table.parquet",index=False)
print("商品中间表保存完成")

#######################
######时间维度中间表######
#######################
time_table = (
    df.groupby("date")
      .agg(
          daily_action_count=("behavior_type", "count"),   # 每天总行为数
          daily_browse_count=("is_browse", "sum"),         # 每天浏览数
          daily_fav_count=("is_fav", "sum"),               # 每天收藏数
          daily_cart_count=("is_cart", "sum"),             # 每天加购数
          daily_buy_count=("is_buy", "sum"),               # 每天购买数
          active_user_count=("user_id", "nunique"),        # 每天活跃用户数
          active_item_count=("item_id", "nunique")         # 每天被互动商品数
      )
      .reset_index()
)
time_table["daily_conversion_rate"] = (
    time_table["daily_buy_count"] / time_table["daily_action_count"]
)
print(time_table.head())
time_table.to_parquet("../data/processed/time_table.parquet",index=False)
print("时间中间表保存完成")

