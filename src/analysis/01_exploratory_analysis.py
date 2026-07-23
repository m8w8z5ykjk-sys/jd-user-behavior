"""
文件名称：01_exploratory_analysis.py

功能：
1. 读取清洗后的用户行为明细数据；
2. 分析用户行为类型分布；
3. 分析日活跃用户、周活跃用户和小时活跃分布；
4. 识别高转化时段并计算小时转化率；
5. 构建浏览、收藏、加购、购买转化漏斗；
6. 使用RFM方法进行用户分群；
7. 对比不同用户群体的行为特征和转化率；
8. 分析热门商品、热门类别、类别转化率和商品复购情况；
9. 对比工作日和周末的用户行为与转化率；
10. 将本文件产生的所有分析结果统一保存到以当前Python文件名命名的results目录。

输入文件名：
- data/processed/build_feature_table/user_behavior_cleaned.parquet
  由build_feature_table.py生成的清洗后用户行为Parquet明细数据。

输入字段要求：
- user_id：用户编号；
- item_id：商品编号；
- item_category：商品类别；
- behavior_type：行为类型，1浏览、2收藏、3加购、4购买；
- time：行为发生时间。

results输出目录：
- results/01_exploratory_analysis/

results输出文件名：
1. behavior_distribution.png
2. daily_active_users.png
3. weekly_active_users.png
4. hour_distribution.png
5. hour_conversion_rate.csv
6. hour_conversion_rate.png
7. conversion_funnel.png
8. conversion_funnel.csv
9. drop_rate.csv
10. user_segment.png
11. segment_behavior_comparison.csv
12. segment_behavior_comparison.png
13. segment_conversion_rate.png
14. top20_items.png
15. top20_categories.png
16. category_conversion_rate.csv
17. category_conversion_rate.png
18. top20_repurchase_items.png
19. weekday_weekend_behavior.png
20. weekday_weekend_conversion_rate.png
21. weekday_weekend_analysis.csv
22. exploratory_analysis_summary.txt

目录规则：
- 本文件没有需要保存到data/processed的中间数据；
- 所有CSV、TXT和PNG分析结果统一保存到：
  results/01_exploratory_analysis/
- 输出目录由程序自动创建。

路径检查：
- 当前文件应位于：项目根目录/src/analysis/；
- 项目根目录通过Path(__file__).resolve().parents[2]自动定位；
- 输入文件读取路径：
  项目根目录/data/processed/build_feature_table/user_behavior_cleaned.parquet；
- 输出目录：
  项目根目录/results/01_exploratory_analysis/。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "build_feature_table"
    / "user_behavior_cleaned.parquet"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "01_exploratory_analysis"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("项目根目录：", BASE_DIR)
print("输入文件：", INPUT_FILE)
print("输出目录：", RESULT_DIR)


# ============================================================
# 2. 输入文件与字段检查
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"未找到输入文件：{INPUT_FILE}\n"
        "请先运行src/build_feature_table.py，生成：\n"
        "data/processed/build_feature_table/user_behavior_cleaned.parquet"
    )

df = pd.read_parquet(INPUT_FILE)

required_columns = {
    "user_id",
    "item_id",
    "item_category",
    "behavior_type",
    "time",
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "输入文件缺少以下必要字段："
        + ", ".join(sorted(missing_columns))
    )

if df.empty:
    raise ValueError(f"输入文件为空：{INPUT_FILE}")

df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
invalid_time_count = int(df["datetime"].isna().sum())

if invalid_time_count > 0:
    print(f"警告：发现{invalid_time_count}条无效时间记录，已删除。")
    df = df.dropna(subset=["datetime"]).copy()

if df.empty:
    raise ValueError("删除无效时间记录后数据为空，无法进行探索性分析。")

df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

print("数据读取成功")
print("数据形状：", df.shape)
print(df.head())


def safe_divide(numerator, denominator):
    """避免分母为0造成无穷值。"""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def save_current_figure(filename: str) -> None:
    """统一保存并关闭当前图形。"""
    output_path = RESULT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("图形已保存：", output_path)


# ============================================================
# 3. 基础行为分析
# ============================================================

behavior_count = df["behavior_type"].value_counts().sort_index()
print("行为类型分布：")
print(behavior_count)

plt.figure(figsize=(8, 5))
behavior_count.plot(kind="bar")
plt.title("Behavior Distribution")
plt.xlabel("Behavior Type")
plt.ylabel("Count")
save_current_figure("behavior_distribution.png")

dau = df.groupby("date")["user_id"].nunique()
print("日活跃用户：")
print(dau.head())

plt.figure(figsize=(10, 5))
dau.plot()
plt.title("Daily Active Users")
plt.xlabel("Date")
plt.ylabel("Users")
save_current_figure("daily_active_users.png")

df["week"] = df["datetime"].dt.to_period("W").astype(str)
wau = df.groupby("week")["user_id"].nunique()
print("周活跃用户：")
print(wau)

plt.figure(figsize=(10, 5))
wau.plot(marker="o")
plt.title("Weekly Active Users")
plt.xlabel("Week")
plt.ylabel("Users")
plt.xticks(rotation=45)
save_current_figure("weekly_active_users.png")

hour_count = df.groupby("hour").size()
print("小时行为分布：")
print(hour_count)

plt.figure(figsize=(10, 5))
hour_count.plot(kind="bar")
plt.title("Hourly Activity")
plt.xlabel("Hour")
plt.ylabel("Behavior Count")
save_current_figure("hour_distribution.png")


# ============================================================
# 4. 高转化时段分析
# ============================================================

view = df[df["behavior_type"] == 1].groupby("hour").size()
buy = df[df["behavior_type"] == 4].groupby("hour").size()

conversion = pd.DataFrame(
    {
        "view": view,
        "buy": buy,
    }
).fillna(0)

conversion["conversion_rate"] = (
    conversion["buy"]
    .div(conversion["view"].replace(0, pd.NA))
    .fillna(0)
)

conversion.to_csv(
    RESULT_DIR / "hour_conversion_rate.csv",
    encoding="utf-8-sig"
)

print("小时转化率：")
print(conversion)

plt.figure(figsize=(10, 5))
conversion["conversion_rate"].plot(kind="bar")
plt.title("Hourly Conversion Rate")
plt.xlabel("Hour")
plt.ylabel("Conversion Rate")
save_current_figure("hour_conversion_rate.png")


# ============================================================
# 5. 转化漏斗分析
# ============================================================

view_users = df.loc[df["behavior_type"] == 1, "user_id"].nunique()
favorite_users = df.loc[df["behavior_type"] == 2, "user_id"].nunique()
cart_users = df.loc[df["behavior_type"] == 3, "user_id"].nunique()
buy_users = df.loc[df["behavior_type"] == 4, "user_id"].nunique()

view_to_favorite = safe_divide(favorite_users, view_users)
favorite_to_cart = safe_divide(cart_users, favorite_users)
cart_to_buy = safe_divide(buy_users, cart_users)
view_to_buy = safe_divide(buy_users, view_users)

print("浏览用户：", view_users)
print("收藏用户：", favorite_users)
print("加购用户：", cart_users)
print("购买用户：", buy_users)
print("浏览→收藏：", round(view_to_favorite * 100, 2), "%")
print("收藏→加购：", round(favorite_to_cart * 100, 2), "%")
print("加购→购买：", round(cart_to_buy * 100, 2), "%")
print("浏览→购买：", round(view_to_buy * 100, 2), "%")

stages = ["View", "Favorite", "Cart", "Buy"]
values = [view_users, favorite_users, cart_users, buy_users]

plt.figure(figsize=(8, 5))
plt.bar(stages, values)
plt.title("Conversion Funnel")
plt.ylabel("User Count")
save_current_figure("conversion_funnel.png")

drop_rate = pd.Series(
    {
        "View -> Favorite": 1 - view_to_favorite,
        "Favorite -> Cart": 1 - favorite_to_cart,
        "Cart -> Buy": 1 - cart_to_buy,
    },
    name="Drop Rate",
)

print("各阶段流失率：")
print(drop_rate)
print("最大流失节点：", drop_rate.idxmax())

funnel_result = pd.DataFrame(
    {
        "Stage": stages,
        "Users": values,
    }
)

funnel_result.to_csv(
    RESULT_DIR / "conversion_funnel.csv",
    index=False,
    encoding="utf-8-sig",
)

drop_rate.to_csv(
    RESULT_DIR / "drop_rate.csv",
    encoding="utf-8-sig",
)


# ============================================================
# 6. 用户分群分析
# ============================================================

buy_df = df[df["behavior_type"] == 4].copy()

if buy_df.empty:
    print("警告：没有购买行为，跳过RFM用户分群分析。")
    rfm = pd.DataFrame(columns=["user_id", "segment"])
else:
    snapshot_date = buy_df["date"].max()

    rfm = (
        buy_df.groupby("user_id")
        .agg(
            Recency=("date", lambda x: (snapshot_date - x.max()).days),
            Frequency=("behavior_type", "count"),
        )
        .reset_index()
    )

    rfm["Monetary"] = rfm["Frequency"]

    # rank后再qcut，减少重复边界导致报错
    rfm["R_score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        q=min(4, len(rfm)),
        labels=False,
        duplicates="drop",
    )

    rfm["F_score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        q=min(4, len(rfm)),
        labels=False,
        duplicates="drop",
    )

    rfm["M_score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        q=min(4, len(rfm)),
        labels=False,
        duplicates="drop",
    )

    # 转换成1~4分，Recency越小分数越高
    max_r = int(rfm["R_score"].max()) if not rfm["R_score"].isna().all() else 0
    rfm["R_score"] = max_r - rfm["R_score"].fillna(0) + 1
    rfm["F_score"] = rfm["F_score"].fillna(0) + 1
    rfm["M_score"] = rfm["M_score"].fillna(0) + 1

    def user_segment(row):
        if row["R_score"] >= 3 and row["F_score"] >= 3:
            return "High Value"
        if row["R_score"] >= 3:
            return "Potential"
        if row["F_score"] >= 3:
            return "At Risk"
        return "Dormant"

    rfm["segment"] = rfm.apply(user_segment, axis=1)
    segment_count = rfm["segment"].value_counts()

    print("用户分群数量：")
    print(segment_count)

    plt.figure(figsize=(7, 5))
    segment_count.plot(kind="bar")
    plt.title("User Segments")
    plt.ylabel("Users")
    save_current_figure("user_segment.png")

    df_with_segment = df.merge(
        rfm[["user_id", "segment"]],
        on="user_id",
        how="inner",
    )

    segment_behavior = (
        df_with_segment.groupby(["segment", "behavior_type"])
        .size()
        .unstack(fill_value=0)
    )

    for behavior_code in [1, 2, 3, 4]:
        if behavior_code not in segment_behavior.columns:
            segment_behavior[behavior_code] = 0

    segment_behavior = segment_behavior.rename(
        columns={
            1: "view_count",
            2: "favorite_count",
            3: "cart_count",
            4: "buy_count",
        }
    )

    segment_behavior["conversion_rate"] = (
        segment_behavior["buy_count"]
        .div(segment_behavior["view_count"].replace(0, pd.NA))
        .fillna(0)
    )

    segment_behavior.to_csv(
        RESULT_DIR / "segment_behavior_comparison.csv",
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 6))
    segment_behavior[
        [
            "view_count",
            "favorite_count",
            "cart_count",
            "buy_count",
        ]
    ].plot(kind="bar")
    plt.title("Behavior Comparison by User Segment")
    plt.ylabel("Behavior Count")
    plt.xlabel("User Segment")
    save_current_figure("segment_behavior_comparison.png")

    plt.figure(figsize=(8, 5))
    segment_behavior["conversion_rate"].plot(kind="bar")
    plt.title("Conversion Rate by User Segment")
    plt.ylabel("Conversion Rate")
    plt.xlabel("User Segment")
    save_current_figure("segment_conversion_rate.png")


# ============================================================
# 7. 商品分析
# ============================================================

item_count = df["item_id"].value_counts()
print("热门商品TOP20：")
print(item_count.head(20))

plt.figure(figsize=(10, 6))
item_count.head(20).plot(kind="bar")
plt.title("Top 20 Popular Items")
plt.xlabel("Item ID")
plt.ylabel("Behavior Count")
save_current_figure("top20_items.png")

category_count = df["item_category"].value_counts()
print("热门商品类别TOP20：")
print(category_count.head(20))

plt.figure(figsize=(10, 6))
category_count.head(20).plot(kind="bar")
plt.title("Top 20 Categories")
plt.xlabel("Category")
plt.ylabel("Behavior Count")
save_current_figure("top20_categories.png")

view_category = (
    df[df["behavior_type"] == 1]
    .groupby("item_category")
    .size()
)

buy_category = (
    df[df["behavior_type"] == 4]
    .groupby("item_category")
    .size()
)

category_conversion = pd.DataFrame(
    {
        "view_count": view_category,
        "buy_count": buy_category,
    }
).fillna(0)

category_conversion["conversion_rate"] = (
    category_conversion["buy_count"]
    .div(category_conversion["view_count"].replace(0, pd.NA))
    .fillna(0)
)

category_conversion = category_conversion.sort_values(
    by="conversion_rate",
    ascending=False,
)

category_conversion.to_csv(
    RESULT_DIR / "category_conversion_rate.csv",
    encoding="utf-8-sig",
)

print("各商品类别购买转化率TOP20：")
print(category_conversion.head(20))

plt.figure(figsize=(10, 6))
category_conversion.head(20)["conversion_rate"].plot(kind="bar")
plt.title("Top20 Category Conversion Rate")
plt.ylabel("Conversion Rate")
save_current_figure("category_conversion_rate.png")

repeat_buy = (
    buy_df.groupby(["user_id", "item_id"])
    .size()
    .reset_index(name="buy_times")
)

if repeat_buy.empty:
    repurchase_rate = 0.0
    item_repeat = pd.Series(dtype="int64")
else:
    repeat_users = repeat_buy[repeat_buy["buy_times"] > 1]
    repurchase_rate = safe_divide(len(repeat_users), len(repeat_buy))
    item_repeat = (
        repeat_users.groupby("item_id")
        .size()
        .sort_values(ascending=False)
    )

print("商品复购率：", f"{repurchase_rate:.2%}")

if not item_repeat.empty:
    print("复购最多商品TOP20：")
    print(item_repeat.head(20))

    plt.figure(figsize=(10, 6))
    item_repeat.head(20).plot(kind="bar")
    plt.title("Top20 Repurchased Items")
    plt.ylabel("Repeat Users")
    save_current_figure("top20_repurchase_items.png")


# ============================================================
# 8. 时间维度分析
# ============================================================

df["weekday"] = df["datetime"].dt.weekday
df["day_type"] = df["weekday"].apply(
    lambda x: "Weekend" if x >= 5 else "Weekday"
)

behavior_day = (
    df.groupby(["day_type", "behavior_type"])
    .size()
    .unstack(fill_value=0)
)

for behavior_code in [1, 2, 3, 4]:
    if behavior_code not in behavior_day.columns:
        behavior_day[behavior_code] = 0

behavior_day = behavior_day.rename(
    columns={
        1: "View",
        2: "Favorite",
        3: "Cart",
        4: "Buy",
    }
)

plt.figure(figsize=(8, 6))
behavior_day[["View", "Favorite", "Cart", "Buy"]].plot(kind="bar")
plt.title("Weekday vs Weekend Behavior")
plt.ylabel("Behavior Count")
save_current_figure("weekday_weekend_behavior.png")

behavior_day["Conversion Rate"] = (
    behavior_day["Buy"]
    .div(behavior_day["View"].replace(0, pd.NA))
    .fillna(0)
)

plt.figure(figsize=(6, 5))
behavior_day["Conversion Rate"].plot(kind="bar")
plt.title("Weekday vs Weekend Conversion Rate")
plt.ylabel("Conversion Rate")
save_current_figure("weekday_weekend_conversion_rate.png")

behavior_day.to_csv(
    RESULT_DIR / "weekday_weekend_analysis.csv",
    encoding="utf-8-sig",
)


# ============================================================
# 9. 保存分析摘要
# ============================================================

summary_text = f"""探索性数据分析摘要

数据行数：{len(df)}
用户数量：{df["user_id"].nunique()}
商品数量：{df["item_id"].nunique()}
商品类别数量：{df["item_category"].nunique()}
数据开始日期：{df["date"].min()}
数据结束日期：{df["date"].max()}

浏览用户数：{view_users}
收藏用户数：{favorite_users}
加购用户数：{cart_users}
购买用户数：{buy_users}

浏览到购买转化率：{view_to_buy:.4%}
商品复购率：{repurchase_rate:.4%}
最大流失节点：{drop_rate.idxmax()}
"""

summary_path = RESULT_DIR / "exploratory_analysis_summary.txt"
summary_path.write_text(summary_text, encoding="utf-8")

print("=" * 60)
print("探索性数据分析完成")
print("所有结果已保存到：", RESULT_DIR)
