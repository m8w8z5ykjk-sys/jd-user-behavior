"""
文件名称：sqlite_analysis.py

任务内容：
1. 读取清洗后的 Parquet 用户行为数据；
2. 检查 user_id、item_id、behavior_type、time、item_category 等必需字段；
3. 提取 date、hour、weekday、month 等时间维度；
4. 构建用户维度表、商品维度表、时间维度表和用户行为事实表；
5. 将四张表写入 SQLite 数据库并建立查询索引；
6. 封装 Python 与 SQLite 查询接口；
7. 完成行为总量、行为占比、日活、周活、热门商品和热门类目分析；
8. 完成行为序列、转化漏斗、用户分群和商品复购率分析；
9. 将本脚本产生的 processed 和 results 输出分别保存到
   以当前 Python 文件名命名的专属目录中。

明确输入文件名：
- data/processed/user_behavior_cleaned.parquet
  清洗后的用户行为 Parquet 明细数据。

明确 processed 输出目录：
- data/processed/sqlite_analysis/

明确 processed 输出文件名：
- data/processed/sqlite_analysis/jd_behavior.db
  SQLite 数据库文件，包含 dim_user、dim_item、dim_time、
  fact_user_behavior 四张表及相关索引。

明确 results 输出目录：
- results/sqlite_analysis/

明确 results 输出文件名：
- results/sqlite_analysis/behavior_count.csv
- results/sqlite_analysis/behavior_ratio.csv
- results/sqlite_analysis/daily_active_users.csv
- results/sqlite_analysis/weekly_active_users.csv
- results/sqlite_analysis/hot_items_top10.csv
- results/sqlite_analysis/hot_categories_top10.csv
- results/sqlite_analysis/behavior_sequence.csv
- results/sqlite_analysis/funnel.csv
- results/sqlite_analysis/user_group.csv
- results/sqlite_analysis/repurchase_rate_top10.csv

目录规则：
- 属于 data/processed 的输出统一保存到：
  data/processed/sqlite_analysis/
- 属于 results 的输出统一保存到：
  results/sqlite_analysis/
- 两个目录由程序自动创建，无需手动建立。
"""


from pathlib import Path

import pandas as pd
import os
import sqlite3

# 当前文件位于“项目根目录/src”中
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 输入文件仍从 data/processed 根目录读取
parquet_path = PROJECT_ROOT / "data" / "processed" /"build_feature_table"/ "user_behavior_cleaned.parquet"

# 当前脚本专属输出目录
processed_output_dir = PROJECT_ROOT / "data" / "processed" / "sqlite_analysis"
result_dir = PROJECT_ROOT / "results" / "sqlite_analysis"

# 自动创建输出目录
processed_output_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)

# SQLite数据库输出到专属processed目录
db_path = processed_output_dir / "jd_behavior.db"
df = pd.read_parquet(parquet_path)
print("数据读取成功")
print(df.head())
print(df.columns)
required_columns = ["user_id","item_id","behavior_type","time","item_category"]
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"缺少字段:{col}")
print("字段检查通过")
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date.astype(str)
df["hour"] = df["time"].dt.hour
df["weekday"] = df["time"].dt.dayofweek
df["month"] = df["time"].dt.month
print("时间字段处理完成")
print(df.head())
dim_user = df[["user_id"]].drop_duplicates().reset_index(drop=True)
dim_item = df [["item_id","item_category"]].drop_duplicates().reset_index(drop=True)
dim_time = df [["time","date","hour","weekday","month"]].drop_duplicates().reset_index(drop=True)
fact_user_behavior = df[["time","user_id","item_id","behavior_type"]].copy()
print("4张表创建完成")

conn = sqlite3.connect(db_path)
dim_user.to_sql("dim_user", conn, if_exists="replace", index=False)
dim_item.to_sql("dim_item", conn, if_exists="replace", index=False)
dim_time.to_sql("dim_time", conn, if_exists="replace", index=False)
fact_user_behavior.to_sql("fact_user_behavior", conn, if_exists="replace", index=False)

cursor = conn.cursor()
cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON fact_user_behavior(user_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_item_id ON fact_user_behavior(item_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_behavior_type ON fact_user_behavior(behavior_type);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON fact_user_behavior(time);")
conn.commit()
print("SQLite数据库和索引创建完成")

###掌握SQL核心语法，完成基础数据统计分析：
##用户行为总量统计
def query_sql(sql,db_path=db_path):
    conn = sqlite3.connect(db_path)
    result = pd.read_sql_query(sql, conn)
    conn.close()
    return result
sql_total = """SELECT COUNT(*) AS total_behavior_count
FROM fact_user_behavior;"""
result_total = query_sql(sql_total)
print(result_total)
##各行为类型占比计算
sql_behavior ="""SELECT behavior_type, COUNT(*) AS behavior_count
FROM fact_user_behavior
GROUP BY behavior_type;"""
result_behavior = query_sql(sql_behavior)
print(result_behavior)
result_behavior.to_csv(result_dir / "behavior_count.csv", index=False)
sql_behavior_ratio = """SELECT behavior_type, COUNT(*) AS behavior_count,
                               ROUND(COUNT(*)*1.0/(SELECT COUNT(*) FROM fact_user_behavior),4) AS ratio
FROM fact_user_behavior
GROUP BY behavior_type;"""
result_behavior_ratio = query_sql(sql_behavior_ratio)
print(result_behavior_ratio)
result_behavior_ratio.to_csv(result_dir / "behavior_ratio.csv", index=False)
##日活/周活趋势查询
#每一天有多少个不同用户活跃过
sql_daily_active = """SELECT DATE(time) AS date,
COUNT(DISTINCT user_id) AS daily_active_users
FROM fact_user_behavior
GROUP BY DATE(time)
ORDER BY date;"""
result_daily_active = query_sql(sql_daily_active)
print(result_daily_active)
result_daily_active.to_csv(result_dir / "daily_active_users.csv",index=False)
#每一周有多少个用户活跃过
sql_weekly_active = """SELECT strftime('%Y-%W', time) AS week,
COUNT(DISTINCT user_id) AS weekly_active_users
FROM fact_user_behavior
GROUP BY strftime('%Y-%W', time)
ORDER BY week;"""
result_weekly_active = query_sql(sql_weekly_active)
print(result_weekly_active)
result_weekly_active.to_csv(result_dir / "weekly_active_users.csv",index=False)
##热门商品与类目排名
#热门商品排名
sql_hot_items = """SELECT item_id,COUNT(*) AS behavior_count
FROM fact_user_behavior
GROUP BY item_id
ORDER BY behavior_count DESC
LIMIT 10;"""
result_hot_items = query_sql(sql_hot_items)
print(result_hot_items)
result_hot_items.to_csv(result_dir / "hot_items_top10.csv",index=False)
#热门类目排名
sql_hot_categories = """SELECT i.item_category,COUNT(*) AS behavior_count
FROM fact_user_behavior f
JOIN dim_item i
ON f.item_id = i.item_id
GROUP BY i.item_category
ORDER BY behavior_count DESC
LIMIT 10;"""
result_hot_categories = query_sql(sql_hot_categories)
print(result_hot_categories)
result_hot_categories.to_csv(result_dir / "hot_categories_top10.csv",index=False)
###实现复杂SQL查询
##用户行为序列提取
sql_behavior_sequence = """SELECT user_id,item_id,behavior_type,time
FROM fact_user_behavior
ORDER BY user_id, time
LIMIT 100;"""
result_behavior_sequence = query_sql(sql_behavior_sequence)
print(result_behavior_sequence)
result_behavior_sequence.to_csv(result_dir / "behavior_sequence.csv",index=False)
##转化漏斗各环节数据计算
sql_funnel = """SELECT behavior_type,COUNT(DISTINCT user_id) AS user_count
FROM fact_user_behavior
GROUP BY behavior_type
ORDER BY user_count DESC;"""
result_funnel = query_sql(sql_funnel)
print(result_funnel)
result_funnel.to_csv(result_dir / "funnel.csv",index=False)
##用户分群统计(按照活跃程度分组)
sql_user_group = """SELECT user_group,COUNT(*) AS user_count
FROM (SELECT user_id,CASE
            WHEN COUNT(*) < 10 THEN '低活跃用户'
            WHEN COUNT(*) BETWEEN 10 AND 50 THEN '中活跃用户'
            ELSE '高活跃用户'
        END AS user_group
    FROM fact_user_behavior
    GROUP BY user_id) t
GROUP BY user_group;"""
result_user_group = query_sql(sql_user_group)
print(result_user_group)
result_user_group.to_csv(result_dir / "user_group.csv",index=False)
##商品复购率分析
sql_repurchase_rate = """WITH user_item_purchase AS (SELECT user_id,item_id,COUNT(*) AS purchase_count
FROM fact_user_behavior
WHERE behavior_type = 4
GROUP BY user_id, item_id),
total_buyers AS (SELECT item_id,COUNT(DISTINCT user_id) AS total_buyers
FROM user_item_purchase
GROUP BY item_id),
repeat_buyers AS (SELECT item_id,COUNT(DISTINCT user_id) AS repeat_buyers
FROM user_item_purchase
WHERE purchase_count > 1
GROUP BY item_id)
SELECT t.item_id,t.total_buyers,COALESCE(r.repeat_buyers, 0) AS repeat_buyers,
ROUND(COALESCE(r.repeat_buyers, 0) * 1.0 / t.total_buyers,4) AS repurchase_rate
FROM total_buyers t
LEFT JOIN repeat_buyers r
ON t.item_id = r.item_id
ORDER BY repurchase_rate DESC
LIMIT 10;"""

result_repurchase_rate = query_sql(sql_repurchase_rate)
print(result_repurchase_rate)
result_repurchase_rate.to_csv(result_dir / "repurchase_rate_top10.csv",index=False)

print("全部 SQLite 数据库分析任务完成！")
