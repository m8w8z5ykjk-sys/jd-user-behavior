import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULT_DIR = BASE_DIR / "results"
df = pd.read_parquet(DATA_DIR / "user_behavior_cleaned.parquet")
df["datetime"] = pd.to_datetime(df["time"])
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
print(df.head())

###基础行为分析
#用户行为类型分布
behavior_count = df["behavior_type"].value_counts()
print(behavior_count)
behavior_count.plot(kind="bar")
plt.title("Behavior Distribution")
plt.xlabel("Behavior Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(RESULT_DIR / "behavior_distribution.png")
plt.show()

#日活跃用户
dau = df.groupby("date")["user_id"].nunique()
print(dau.head())
dau.plot(figsize=(10,5))
plt.title("Daily Active Users")
plt.ylabel("Users")
plt.tight_layout()
plt.savefig(RESULT_DIR / "daily_active_users.png")
plt.show()

#周活跃用户
df["week"] = df["datetime"].dt.isocalendar().week
wau = df.groupby("week")["user_id"].nunique()
print(wau)
wau.plot(marker="o")
plt.title("Weekly Active Users")
plt.tight_layout()
plt.savefig(RESULT_DIR / "weekly_active_users.png")
plt.show()

#用户活跃时段分布
hour_count = df.groupby("hour").size()
print(hour_count)
hour_count.plot(kind="bar")
plt.title("Hourly Activity")
plt.xlabel("Hour")
plt.ylabel("Behavior Count")
plt.tight_layout()
plt.savefig(RESULT_DIR / "hour_distribution.png")
plt.show()

# 高转化时段分析
# 浏览行为（behavior_type == 1）
view = df[df["behavior_type"] == 1].groupby("hour").size()
# 购买行为（behavior_type == 4）
buy = df[df["behavior_type"] == 4].groupby("hour").size()
# 合并
conversion = pd.DataFrame({"view": view,"buy": buy}).fillna(0)
# 计算转化率
conversion["conversion_rate"] = conversion["buy"] / conversion["view"]
print(conversion)
# 保存结果
conversion.to_csv(RESULT_DIR / "hour_conversion_rate.csv",index=True)
# 画图
conversion["conversion_rate"].plot(kind="bar",figsize=(10,5))
plt.title("Hourly Conversion Rate")
plt.xlabel("Hour")
plt.ylabel("Conversion Rate")
plt.tight_layout()
plt.savefig(RESULT_DIR / "hour_conversion_rate.png")
plt.show()


###转化漏斗分析
view_users = df[df["behavior_type"] == 1]["user_id"].nunique()
favorite_users = df[df["behavior_type"] == 2]["user_id"].nunique()
cart_users = df[df["behavior_type"] == 3]["user_id"].nunique()
buy_users = df[df["behavior_type"] == 4]["user_id"].nunique()
print("浏览用户：", view_users)
print("收藏用户：", favorite_users)
print("加购用户：", cart_users)
print("购买用户：", buy_users)
#计算转化率
view_to_favorite = favorite_users / view_users
favorite_to_cart = cart_users / favorite_users
cart_to_buy = buy_users / cart_users
view_to_buy = buy_users / view_users
print("浏览→收藏：", round(view_to_favorite * 100, 2), "%")
print("收藏→加购：", round(favorite_to_cart * 100, 2), "%")
print("加购→购买：", round(cart_to_buy * 100, 2), "%")
print("浏览→购买：", round(view_to_buy * 100, 2), "%")
#画漏斗图
stages = ["View", "Favorite", "Cart", "Buy"]
values = [view_users, favorite_users, cart_users, buy_users]
plt.figure(figsize=(8,5))
plt.bar(stages, values)
plt.title("Conversion Funnel")
plt.ylabel("User Count")
plt.tight_layout()
plt.savefig(RESULT_DIR / "conversion_funnel.png")
plt.show()

#找最大的流失节点
drop_rate = {"View -> Favorite": 1 - view_to_favorite,"Favorite -> Cart": 1 - favorite_to_cart,"Cart -> Buy": 1 - cart_to_buy}
drop_rate = pd.Series(drop_rate)
print("各阶段流失率：")
print(drop_rate)
print("\n最大流失节点：")
print(drop_rate.idxmax())

funnel_result = pd.DataFrame({"Stage": stages,"Users": values})
funnel_result.to_csv(RESULT_DIR / "conversion_funnel.csv",index=False)
drop_rate.to_csv(RESULT_DIR / "drop_rate.csv",header=["Drop Rate"])

###用户分群分析
buy_df = df[df["behavior_type"] == 4].copy()
snapshot_date = buy_df["date"].max()
rfm = buy_df.groupby("user_id").agg(Recency=("date", lambda x: (snapshot_date - x.max()).days),Frequency=("behavior_type", "count")).reset_index()
rfm["Monetary"] = rfm["Frequency"]
rfm["R_score"] = pd.qcut(rfm["Recency"],4,labels=[4,3,2,1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"),4,labels=[1,2,3,4]).astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"),4,labels=[1,2,3,4]).astype(int)

def user_segment(row):
    if row["R_score"]>=3 and row["F_score"]>=3:
        return "High Value"

    elif row["R_score"]>=3:
        return "Potential"

    elif row["F_score"]>=3:
        return "At Risk"

    else:
        return "Dormant"
rfm["segment"] = rfm.apply(user_segment,axis=1)
segment_count = rfm["segment"].value_counts()
print(segment_count)
#绘制用户分群图
plt.figure(figsize=(7,5))
segment_count.plot(kind="bar")
plt.title("User Segments")
plt.ylabel("Users")
plt.tight_layout()
plt.savefig(RESULT_DIR/"user_segment.png")
plt.show()

#对比不同群体的行为特征与转化差异
# 把用户分群结果合并回原始行为数据
df_with_segment = df.merge(rfm[["user_id", "segment"]],on="user_id",how="inner")
# 统计每类用户的不同行为次数
segment_behavior = (df_with_segment.groupby(["segment", "behavior_type"]).size().unstack(fill_value=0))
# 重命名行为列
segment_behavior = segment_behavior.rename(
    columns={1: "view_count",2: "favorite_count",3: "cart_count",4: "buy_count"})
print("不同用户群体行为次数：")
print(segment_behavior)
# 计算每个群体的购买转化率：购买次数 / 浏览次数
segment_behavior["conversion_rate"] = (segment_behavior["buy_count"] / segment_behavior["view_count"])
print("不同用户群体转化率：")
print(segment_behavior["conversion_rate"])
# 保存结果
segment_behavior.to_csv(RESULT_DIR / "segment_behavior_comparison.csv")
# 画不同群体行为特征对比图
segment_behavior[["view_count", "favorite_count", "cart_count", "buy_count"]].plot(kind="bar",figsize=(10, 6))
plt.title("Behavior Comparison by User Segment")
plt.ylabel("Behavior Count")
plt.xlabel("User Segment")
plt.tight_layout()
plt.savefig(RESULT_DIR / "segment_behavior_comparison.png")
plt.show()
# 画不同群体转化率对比图
segment_behavior["conversion_rate"].plot(kind="bar",figsize=(8, 5))
plt.title("Conversion Rate by User Segment")
plt.ylabel("Conversion Rate")
plt.xlabel("User Segment")
plt.tight_layout()
plt.savefig(RESULT_DIR / "segment_conversion_rate.png")
plt.show()

###商品分析
# 每个商品出现次数
item_count = df["item_id"].value_counts()
print("热门商品TOP20：")
print(item_count.head(20))
#画图
plt.figure(figsize=(10,6))
item_count.head(20).plot(kind="bar")
plt.title("Top 20 Popular Items")
plt.xlabel("Item ID")
plt.ylabel("Behavior Count")
plt.tight_layout()
plt.savefig(RESULT_DIR / "top20_items.png")
plt.show()

# 热门商品类别分析
category_count = df["item_category"].value_counts()
print("热门商品类别TOP20：")
print(category_count.head(20))
#画图
plt.figure(figsize=(10,6))
category_count.head(20).plot(kind="bar")
plt.title("Top 20 Categories")
plt.xlabel("Category")
plt.ylabel("Behavior Count")
plt.tight_layout()
plt.savefig(RESULT_DIR / "top20_categories.png")
plt.show()

#分析不同类目商品的转化周期与复购率
# 不同商品类别购买转化率
view_category = (df[df["behavior_type"] == 1].groupby("item_category").size())
buy_category = (df[df["behavior_type"] == 4].groupby("item_category").size())
category_conversion = pd.DataFrame({"view_count": view_category,"buy_count": buy_category}).fillna(0)
category_conversion["conversion_rate"] = (category_conversion["buy_count"]/ category_conversion["view_count"])
category_conversion = category_conversion.sort_values(by="conversion_rate",ascending=False)
print("各商品类别购买转化率TOP20：")
print(category_conversion.head(20))
category_conversion.to_csv(RESULT_DIR / "category_conversion_rate.csv")
#画图
plt.figure(figsize=(10,6))
category_conversion.head(20)["conversion_rate"].plot(kind="bar")
plt.title("Top20 Category Conversion Rate")
plt.ylabel("Conversion Rate")
plt.tight_layout()
plt.savefig(RESULT_DIR / "category_conversion_rate.png")
plt.show()

#商品复购率分析
buy_df = df[df["behavior_type"] == 4].copy()
repeat_buy = (buy_df.groupby(["user_id", "item_id"]).size().reset_index(name="buy_times"))
repeat_users = repeat_buy[repeat_buy["buy_times"] > 1]
#计算复购率
repurchase_rate = len(repeat_users) / len(repeat_buy)
print("商品复购率：")
print(f"{repurchase_rate:.2%}")
#每个商品复购人数
item_repeat = (repeat_users.groupby("item_id").size().sort_values(ascending=False))
print("复购最多商品TOP20：")
print(item_repeat.head(20))
#画图
plt.figure(figsize=(10,6))
item_repeat.head(20).plot(kind="bar")
plt.title("Top20 Repurchased Items")
plt.ylabel("Repeat Users")
plt.tight_layout()
plt.savefig(RESULT_DIR / "top20_repurchase_items.png")
plt.show()

###时间维度分析
#区分工作日和周末
# 星期（0=周一，6=周日）
df["weekday"] = df["datetime"].dt.weekday
# 是否周末
df["day_type"] = df["weekday"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")
print(df[["datetime", "weekday", "day_type"]].head())
behavior_day = (
    df.groupby(["day_type", "behavior_type"])
      .size()
      .unstack(fill_value=0))
behavior_day = behavior_day.rename(columns={ 1: "View",2: "Favorite",3: "Cart",4: "Buy"})
print(behavior_day)
#画行为对比图
behavior_day.plot(kind="bar",figsize=(8,6))
plt.title("Weekday vs Weekend Behavior")
plt.ylabel("Behavior Count")
plt.tight_layout()
plt.savefig(RESULT_DIR / "weekday_weekend_behavior.png")
plt.show()
behavior_day["Conversion Rate"] = (behavior_day["Buy"] /behavior_day["View"])
print(behavior_day["Conversion Rate"])
#画转化率图
behavior_day["Conversion Rate"].plot(kind="bar",figsize=(6,5))
plt.title("Weekday vs Weekend Conversion Rate")
plt.ylabel("Conversion Rate")
plt.tight_layout()
plt.savefig(RESULT_DIR / "weekday_weekend_conversion_rate.png")
plt.show()
behavior_day.to_csv(RESULT_DIR / "weekday_weekend_analysis.csv")

#节假日与非节假日的用户行为差异
print(df["date"].min())
print(df["date"].max())
#没有春节，国庆，劳动节，端午节，中秋（这里对比不了）