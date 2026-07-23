"""
文件名称：clean_data.py

任务内容：
1. 读取原始用户行为数据，检查数据规模、字段名称和缺失值；
2. 根据业务规则检查非法 user_id、item_id 和 item_category；
3. 按 user_id、item_id、behavior_type、time 四元组检查并删除重复记录；
4. 将 time 字段统一转换为日期时间格式，检查无效时间和时间范围；
5. 检查 behavior_type 是否属于合法取值 1、2、3、4；
6. 自动完成数据完整性、一致性、唯一性和时间有效性检测；
7. 生成数据质量报告；
8. 保存清洗后的用户行为数据。

输入文件：
- /Users/chenyixuan/Desktop/京东/data/raw/user_behavior_processed.csv
  原始用户行为数据。

输出文件：
- ../data/processed/user_behavior_cleaned.csv
  完成四元组去重、时间格式统一和基础质量检查后的清洗数据。

- ../results/quality_report.txt
  数据质量报告，包含缺失值、非法行为、重复记录和时间有效性等结果。

控制台输出：
- 数据前5行、数据形状、字段名称和缺失值；
- 非法字段数量、重复数量、时间范围及质量检测结果。
"""
import pandas as pd
df = pd.read_csv("/Users/chenyixuan/Desktop/京东/data/raw/user_behavior_processed.csv")
print(df.head())
print("数据行数和列数：")
print(df.shape)
print("字段名：")
print(df.columns)
print("各字段缺失值数量：")
print(df.isnull().sum())
#业务规则异常值处理
invalid_user = (df["user_id"] <= 0).sum()
invalid_item = (df["item_id"] <= 0).sum()
invalid_category = (df["item_category"] <= 0).sum()
print("非法 user_id:", invalid_user)
print("非法 item_id:", invalid_item)
print("非法 item_category：",invalid_category)
#IQR一般用于连续数值型，例如商品价格，用户年龄，订单金额和商品数量
#检查重复数据
duplicate_count = df.duplicated(subset=["user_id","item_id","behavior_type","time"]).sum()
print("重复数据数量：")
print(duplicate_count)
#删除重复数据
df = df.drop_duplicates(subset=["user_id","item_id","behavior_type","time"])
print("去重后的数据行数：")
print(df.shape)
#统一时间格式
df["time"] = pd.to_datetime(df["time"])
print("time 字段的数据类型：")
print(df["time"].dtype)
#校验时间有效性
invalid_time = df["time"].isna().sum()
print("无效时间数量：")
print(invalid_time)
#检查时间范围
print("最早时间:")
print(df["time"].min())
print("最晚时间:")
print(df["time"].max())
###自动化数据质量检测
#数据完整性检查
missing = df.isnull().sum()
print(missing)
#数据一致性检查
valid_behavior = [1,2,3,4]
invalid_behavior = df[ ~df["behavior_type"].isin(valid_behavior) ]
print("非法行为数量：")
print(len(invalid_behavior))
#数据唯一性检测
unique_duplicate = df.duplicated(subset=["user_id","item_id","behavior_type","time"]).sum()
print("四元组重复数量：",unique_duplicate)
# 生成数据质量报告
with open("../results/quality_report.txt", "w", encoding="utf-8") as f:

    f.write("========== 数据质量报告 ==========\n\n")

    f.write("【数据完整性】\n")
    f.write(str(missing))
    f.write("\n\n")

    f.write("【数据一致性】\n")
    f.write(f"非法行为数量：{len(invalid_behavior)}\n\n")

    f.write("【数据唯一性】\n")
    f.write(f"四元组重复数量：{unique_duplicate}\n\n")

    f.write("【时间有效性】\n")
    f.write(f"无效时间数量：{invalid_time}\n")
    f.write(f"最早时间：{df['time'].min()}\n")
    f.write(f"最晚时间：{df['time'].max()}\n")
print("质量报告生成成功！")
output_path = "../data/processed/user_behavior_cleaned.csv"
df.to_csv(output_path, index=False,encoding="utf-8-sig")