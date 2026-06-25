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