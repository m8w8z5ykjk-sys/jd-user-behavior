"""
文件名称：build_feature_table.py

任务内容：
1. 读取 clean_data.py 清洗后的用户行为 CSV 数据；
2. 压缩 user_id、item_id、item_category、behavior_type 等字段的数据类型；
3. 将清洗后的明细数据转换为 Parquet 格式；
4. 构建用户、商品、时间、商品类别和用户—商品五类中间表；
5. 对主要聚合字段进行抽样核验，验证中间表计算是否正确；
6. 将本脚本生成的所有 processed 文件统一保存到以本脚本命名的目录中。

明确输入文件名：
- data/processed/user_behavior_cleaned.csv
  由 clean_data.py 生成的清洗后用户行为明细数据。

明确输出目录：
- data/processed/build_feature_table/

明确输出文件名：
- data/processed/build_feature_table/user_behavior_cleaned.parquet
- data/processed/build_feature_table/user_table.parquet
- data/processed/build_feature_table/item_table.parquet
- data/processed/build_feature_table/time_table.parquet
- data/processed/build_feature_table/category_table.parquet
- data/processed/build_feature_table/user_item_table.parquet

目录规则：
- 如果输出属于 data/processed，则放入：
  data/processed/当前Python文件名/
- 本文件名为 build_feature_table.py，因此输出目录为：
  data/processed/build_feature_table/
"""

from pathlib import Path

import pandas as pd

# 自动定位项目根目录。当前文件位于项目根目录/src下。
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 本脚本专属processed输出目录
PROCESSED_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "build_feature_table"
PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输入文件仍从processed根目录读取
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "user_behavior_cleaned.csv"

df = pd.read_csv(INPUT_FILE)
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
df.to_parquet(PROCESSED_OUTPUT_DIR / "user_behavior_cleaned.parquet",index=False)
print("保存成功！")
#验证是否保存成功
df2 = pd.read_parquet(PROCESSED_OUTPUT_DIR / "user_behavior_cleaned.parquet")
print(df2.head())


########################################用户维度中间表########################################
# 先提取日期
df["date"] = df["time"].dt.date

df["is_browse"] = (df["behavior_type"] == 1).astype("int8")
df["is_fav"] = (df["behavior_type"] == 2).astype("int8")
df["is_cart"] = (df["behavior_type"] == 3).astype("int8")
df["is_buy"] = (df["behavior_type"] == 4).astype("int8")
user_table = (
    df.groupby("user_id")
      .agg(
          action_count=("behavior_type", "count"),      # 总行为次数
          browse_count=("is_browse","sum"),   # 浏览次数
          fav_count=("is_fav","sum"),      # 收藏次数
          cart_count=("is_cart","sum"),     # 加购次数
          buy_count=("is_buy","sum"),      # 购买次数
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
user_table.to_parquet(PROCESSED_OUTPUT_DIR / "user_table.parquet",index=False)
print("用户中间表保存完成")

#########################################抽检##############################################
sample = user_table.iloc[0]
user_data = df[df["user_id"] == sample["user_id"]]
#1
print(user_data.shape[0])
print(sample["action_count"])
#2
print(user_data["is_browse"].sum())
print(sample["browse_count"])
#3
print(user_data["is_fav"].sum())
print(sample["fav_count"])
#4
print(user_data["is_cart"].sum())
print(sample["cart_count"])
#5
print(user_data["is_buy"].sum())
print(sample["buy_count"])
#6
print(user_data["date"].nunique())
print(sample["active_days"])
#7
print(user_data["time"].min())
print(sample["first_active_time"])
#8
print(user_data["time"].max())
print(sample["last_active_time"])
#9
print(user_data["item_id"].nunique())
print(sample["unique_item_count"])
#10
print(user_data["item_category"].nunique())
print(sample["unique_category_count"])
#11
print(user_data["is_buy"].sum()/user_data.shape[0])
print(sample["conversion_rate"])


######################################商品维度中间表#####################################
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
item_table.to_parquet(PROCESSED_OUTPUT_DIR / "item_table.parquet",index=False)
print("商品中间表保存完成")

#########################################抽检##############################################
sample = item_table.iloc[0]
item_data = df[df["item_id"] == sample["item_id"]]

# 1. item_action_count 商品总行为次数
print("原始计算 item_action_count：", item_data.shape[0])
print("中间表 item_action_count：", sample["item_action_count"])

# 2. item_browse_count 商品浏览次数
print("原始计算 item_browse_count：", item_data["is_browse"].sum())
print("中间表 item_browse_count：", sample["item_browse_count"])

# 3. item_fav_count 商品收藏次数
print("原始计算 item_fav_count：", item_data["is_fav"].sum())
print("中间表 item_fav_count：", sample["item_fav_count"])

# 4. item_cart_count 商品加购次数
print("原始计算 item_cart_count：", item_data["is_cart"].sum())
print("中间表 item_cart_count：", sample["item_cart_count"])

# 5. item_buy_count 商品购买次数
print("原始计算 item_buy_count：", item_data["is_buy"].sum())
print("中间表 item_buy_count：", sample["item_buy_count"])

# 6. buyer_count 互动过该商品的用户数
print("原始计算 buyer_count：", item_data["user_id"].nunique())
print("中间表 buyer_count：", sample["buyer_count"])

# 7. item_active_days 商品活跃天数
print("原始计算 item_active_days：", item_data["date"].nunique())
print("中间表 item_active_days：", sample["item_active_days"])

# 8. item_category 商品所属类别
print("原始计算 item_category：", item_data["item_category"].iloc[0])
print("中间表 item_category：", sample["item_category"])

# 9. item_conversion_rate 商品转化率
manual_item_conversion_rate = item_data["is_buy"].sum() / item_data.shape[0]
print("原始计算 item_conversion_rate：", manual_item_conversion_rate)
print("中间表 item_conversion_rate：", sample["item_conversion_rate"])


##########################################时间维度中间表############################################
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
time_table.to_parquet(PROCESSED_OUTPUT_DIR / "time_table.parquet",index=False)
print("时间中间表保存完成")

#########################################抽检##############################################
sample = time_table.iloc[0]
time_data = df[df["date"] == sample["date"]]

# 1. daily_action_count 每天总行为数
print("原始计算 daily_action_count：", time_data.shape[0])
print("中间表 daily_action_count：", sample["daily_action_count"])

# 2. daily_browse_count 每天浏览数
print("原始计算 daily_browse_count：", time_data["is_browse"].sum())
print("中间表 daily_browse_count：", sample["daily_browse_count"])

# 3. daily_fav_count 每天收藏数
print("原始计算 daily_fav_count：", time_data["is_fav"].sum())
print("中间表 daily_fav_count：", sample["daily_fav_count"])

# 4. daily_cart_count 每天加购数
print("原始计算 daily_cart_count：", time_data["is_cart"].sum())
print("中间表 daily_cart_count：", sample["daily_cart_count"])

# 5. daily_buy_count 每天购买数
print("原始计算 daily_buy_count：", time_data["is_buy"].sum())
print("中间表 daily_buy_count：", sample["daily_buy_count"])

# 6. active_user_count 每天活跃用户数
print("原始计算 active_user_count：", time_data["user_id"].nunique())
print("中间表 active_user_count：", sample["active_user_count"])

# 7. active_item_count 每天被互动商品数
print("原始计算 active_item_count：", time_data["item_id"].nunique())
print("中间表 active_item_count：", sample["active_item_count"])

# 8. daily_conversion_rate 每日转化率
manual_daily_conversion_rate = time_data["is_buy"].sum() / time_data.shape[0]
print("原始计算 daily_conversion_rate：", manual_daily_conversion_rate)
print("中间表 daily_conversion_rate：", sample["daily_conversion_rate"])


###########################################商品类别中间表#############################################
category_table = (
    df.groupby("item_category")
    .agg(
        pv_count=("is_browse", "sum"),          # 浏览次数
        fav_count=("is_fav", "sum"),            # 收藏次数
        cart_count=("is_cart", "sum"),          # 加购次数
        buy_count=("is_buy", "sum"),            # 购买次数
        item_count=("item_id", "nunique"),      # 商品数量
        user_count=("user_id", "nunique"),      # 用户数量
    )
    .reset_index())

# 转化率
category_table["conversion_rate"] = (category_table["buy_count"] /category_table["pv_count"])
print(category_table.head())
category_table.to_parquet(PROCESSED_OUTPUT_DIR / "category_table.parquet",index=False)
print("商品类别中间表保存完成")

#########################################抽检##############################################
sample = category_table.iloc[0]
category_data = df[df["item_category"] == sample["item_category"]]

# 1. pv_count 浏览次数
print("原始计算 pv_count：", category_data["is_browse"].sum())
print("中间表 pv_count：", sample["pv_count"])

# 2. fav_count 收藏次数
print("原始计算 fav_count：", category_data["is_fav"].sum())
print("中间表 fav_count：", sample["fav_count"])

# 3. cart_count 加购次数
print("原始计算 cart_count：", category_data["is_cart"].sum())
print("中间表 cart_count：", sample["cart_count"])

# 4. buy_count 购买次数
print("原始计算 buy_count：", category_data["is_buy"].sum())
print("中间表 buy_count：", sample["buy_count"])

# 5. item_count 商品数量
print("原始计算 item_count：", category_data["item_id"].nunique())
print("中间表 item_count：", sample["item_count"])

# 6. user_count 用户数量
print("原始计算 user_count：", category_data["user_id"].nunique())
print("中间表 user_count：", sample["user_count"])

# 7. conversion_rate 转化率
manual_conversion_rate = (category_data["is_buy"].sum() /category_data["is_browse"].sum())
print("原始计算 conversion_rate：", manual_conversion_rate)
print("中间表 conversion_rate：", sample["conversion_rate"])


############################################用户商品中间表#########################################
user_item_table = (
    df.groupby(["user_id", "item_id"])
    .agg(
        ui_pv_count=("is_browse", "sum"),     # 浏览
        ui_fav_count=("is_fav", "sum"),       # 收藏
        ui_cart_count=("is_cart", "sum"),     # 加购
        ui_buy_count=("is_buy", "sum"),       # 购买
        last_behavior_time=("time", "max")    # 最近交互时间
    ).reset_index())
print(user_item_table.head())
user_item_table.to_parquet(PROCESSED_OUTPUT_DIR / "user_item_table.parquet",index=False)
print("用户商品中间表保存完成")

#########################################抽检##############################################
sample = user_item_table.iloc[0]
ui_data = df[(df["user_id"] == sample["user_id"]) &(df["item_id"] == sample["item_id"])]

# 1. 浏览次数 ui_pv_count
print("原始计算 ui_pv_count：", ui_data["is_browse"].sum())
print("中间表 ui_pv_count：", sample["ui_pv_count"])

# 2. 收藏次数 ui_fav_count
print("原始计算 ui_fav_count：", ui_data["is_fav"].sum())
print("中间表 ui_fav_count：", sample["ui_fav_count"])

# 3. 加购次数 ui_cart_count
print("原始计算 ui_cart_count：", ui_data["is_cart"].sum())
print("中间表 ui_cart_count：", sample["ui_cart_count"])

# 4. 购买次数 ui_buy_count
print("原始计算 ui_buy_count：", ui_data["is_buy"].sum())
print("中间表 ui_buy_count：", sample["ui_buy_count"])

# 5. 最后交互时间 last_behavior_time
print("原始计算 last_behavior_time：", ui_data["time"].max())
print("中间表 last_behavior_time：", sample["last_behavior_time"])
