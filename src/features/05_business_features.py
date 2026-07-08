import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "user_behavior_cleaned.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(INPUT_FILE)
df["datetime"] = pd.to_datetime(df["time"])
print("数据读取成功")
print(df.head())
print(df.columns)
df["date"] = df["time"].dt.date
df["date"] = pd.to_datetime(df["date"])
#行为类型说明
# 1 = 浏览 pv
# 2 = 收藏 fav
# 3 = 加购 cart
# 4 = 购买 buy
behavior_score_map = {1: 1,2: 2,3: 3,4: 5}# 购买，权重更高
df["behavior_score"] = df["behavior_type"].map(behavior_score_map)

#RFM 用户价值特征
max_date = df["date"].max()
user_last_behavior = (df.groupby("user_id")["date"].max().reset_index(name="last_behavior_date"))
user_last_behavior["recency_days"] = (max_date - user_last_behavior["last_behavior_date"]).dt.days
user_buy_frequency = (df[df["behavior_type"] == 4].groupby("user_id").size().reset_index(name="buy_frequency"))
user_behavior_value = (df.groupby("user_id")["behavior_score"].sum().reset_index(name="behavior_value_score"))
rfm_features = user_last_behavior.merge(user_buy_frequency,on="user_id",how="left")
rfm_features = rfm_features.merge(user_behavior_value,on="user_id",how="left")
rfm_features["buy_frequency"] = rfm_features["buy_frequency"].fillna(0)
rfm_features["R_score"] = 6 - pd.qcut(rfm_features["recency_days"].rank(method="first"),q=5,labels=False,duplicates="drop")
rfm_features["F_score"] = pd.qcut(rfm_features["buy_frequency"].rank(method="first"),q=5,labels=False,duplicates="drop") + 1
rfm_features["M_score"] = pd.qcut(rfm_features["behavior_value_score"].rank(method="first"),q=5,labels=False,duplicates="drop") + 1
rfm_features["rfm_value_score"] = (rfm_features["R_score"]+ rfm_features["F_score"]+ rfm_features["M_score"])

#全链路转化漏斗特征
user_behavior_count = (df.groupby(["user_id", "behavior_type"]).size().reset_index(name="count"))
user_funnel = user_behavior_count.pivot_table(index="user_id",columns="behavior_type",values="count",fill_value=0).reset_index()
user_funnel = user_funnel.rename(columns={1: "pv_count",2: "fav_count",3: "cart_count",4: "buy_count"})
for col in ["pv_count", "fav_count", "cart_count", "buy_count"]:
    if col not in user_funnel.columns:
        user_funnel[col] = 0
user_funnel["fav_to_pv_rate"] = (user_funnel["fav_count"] / user_funnel["pv_count"].replace(0, pd.NA))
user_funnel["cart_to_fav_rate"] = (user_funnel["cart_count"] / user_funnel["fav_count"].replace(0, pd.NA))
user_funnel["buy_to_cart_rate"] = (user_funnel["buy_count"] / user_funnel["cart_count"].replace(0, pd.NA))
user_funnel["buy_to_pv_rate"] = (user_funnel["buy_count"] / user_funnel["pv_count"].replace(0, pd.NA))
rate_cols = ["fav_to_pv_rate","cart_to_fav_rate","buy_to_cart_rate","buy_to_pv_rate"]
user_funnel[rate_cols] = user_funnel[rate_cols].fillna(0)

#用户偏好强度得分
user_preference_score = (df.groupby("user_id")["behavior_score"].sum().reset_index(name="user_preference_score"))

#合并业务导向特征
business_features = rfm_features.merge(user_funnel,on="user_id",how="left")
business_features = business_features.merge(user_preference_score,on="user_id",how="left")


business_features.to_parquet(OUTPUT_DIR / "business_features.parquet",index=False)

print("业务导向特征已保存")