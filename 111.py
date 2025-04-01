from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, count, avg, sum, min, max, year, when, isnan, countDistinct, corr, round, udf
from pyspark.sql.types import IntegerType
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("VideoGameSalesAnalysis") \
    .getOrCreate()

# 1. 数据加载（假设数据存储在CSV文件中）
# 请根据你的实际文件路径修改
file_path = "videogame_sales.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# 显示数据集的基本信息
print("数据集基本信息:")
df.printSchema()
print(f"总行数: {df.count()}")

# 2. 数据清洗和预处理
# 检查缺失值
print("\n各列缺失值数量:")
for column in df.columns:
    null_count = df.filter(col(column).isNull() | isnan(col(column))).count()
    print(f"{column}: {null_count}")

# 处理Year列缺失值 - 用平均值填充
year_avg = df.select(avg("Year")).first()[0]
df = df.fillna({"Year": year_avg})

# 将Year转换为整数类型
df = df.withColumn("Year", col("Year").cast(IntegerType()))

# 处理Publisher列缺失值 - 用"Unknown"填充
df = df.fillna({"Publisher": "Unknown"})

# 3. 描述性统计分析
print("\n销售数据基本统计量:")
df.select("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales").summary().show()

# 全球销量最高的10款游戏
print("\n全球销量最高的10款游戏:")
df.select("Name", "Platform", "Year", "Genre", "Publisher", "Global_Sales") \
    .orderBy(col("Global_Sales").desc()) \
    .show(10, False)

# 游戏平台分布
print("\n各平台游戏数量:")
platform_counts = df.groupBy("Platform").count().orderBy(desc("count"))
platform_counts.show()

# 游戏类型分布
print("\n各类型游戏数量:")
genre_counts = df.groupBy("Genre").count().orderBy(desc("count"))
genre_counts.show()

# 4. 趋势分析
# 按年份统计游戏发行量和平均销售额
print("\n年度游戏发行量和平均销售额:")
yearly_stats = df.groupBy("Year") \
    .agg(
        count("Name").alias("Game_Count"),
        round(avg("Global_Sales"), 2).alias("Avg_Global_Sales")
    ) \
    .orderBy("Year")
yearly_stats.show()

# 5. 分类比较
# 各平台的总销售额
print("\n各平台的总销售额:")
platform_sales = df.groupBy("Platform") \
    .agg(
        round(sum("Global_Sales"), 2).alias("Total_Sales"),
        count("Name").alias("Game_Count")
    ) \
    .orderBy(desc("Total_Sales"))
platform_sales.show()

# 各游戏类型的总销售额
print("\n各游戏类型的总销售额:")
genre_sales = df.groupBy("Genre") \
    .agg(
        round(sum("Global_Sales"), 2).alias("Total_Sales"),
        count("Name").alias("Game_Count")
    ) \
    .orderBy(desc("Total_Sales"))
genre_sales.show()

# 各发行商的总销售额 (Top 15)
print("\n销售额最高的15家发行商:")
publisher_sales = df.groupBy("Publisher") \
    .agg(
        round(sum("Global_Sales"), 2).alias("Total_Sales"),
        count("Name").alias("Game_Count")
    ) \
    .orderBy(desc("Total_Sales"))
publisher_sales.show(15)

# 6. 地区差异分析
# 各地区销售总额
print("\n各地区销售总额:")
region_sales = df.select(
    round(sum("NA_Sales"), 2).alias("North_America"),
    round(sum("EU_Sales"), 2).alias("Europe"),
    round(sum("JP_Sales"), 2).alias("Japan"),
    round(sum("Other_Sales"), 2).alias("Other_Regions")
)
region_sales.show()

# 各地区最受欢迎的游戏类型
print("\n北美地区最受欢迎的游戏类型:")
na_genre = df.groupBy("Genre") \
    .agg(round(sum("NA_Sales"), 2).alias("NA_Total_Sales")) \
    .orderBy(desc("NA_Total_Sales"))
na_genre.show()

print("\n欧洲地区最受欢迎的游戏类型:")
eu_genre = df.groupBy("Genre") \
    .agg(round(sum("EU_Sales"), 2).alias("EU_Total_Sales")) \
    .orderBy(desc("EU_Total_Sales"))
eu_genre.show()

print("\n日本地区最受欢迎的游戏类型:")
jp_genre = df.groupBy("Genre") \
    .agg(round(sum("JP_Sales"), 2).alias("JP_Total_Sales")) \
    .orderBy(desc("JP_Total_Sales"))
jp_genre.show()

# 7. 相关性分析
# 计算发行年份与全球销量的相关性
print("\n发行年份与全球销量的相关性:")
year_sales_corr = df.select(corr("Year", "Global_Sales")).first()[0]
print(f"相关系数: {year_sales_corr}")

# 计算各地区销售额之间的相关性
print("\n各地区销售额之间的相关性:")
# NA与EU
na_eu_corr = df.select(corr("NA_Sales", "EU_Sales")).first()[0]
print(f"北美与欧洲销售额相关系数: {na_eu_corr}")
# NA与JP
na_jp_corr = df.select(corr("NA_Sales", "JP_Sales")).first()[0]
print(f"北美与日本销售额相关系数: {na_jp_corr}")
# EU与JP
eu_jp_corr = df.select(corr("EU_Sales", "JP_Sales")).first()[0]
print(f"欧洲与日本销售额相关系数: {eu_jp_corr}")

# 8. 可视化分析（需要将Spark DataFrame转换为Pandas DataFrame）
# 全球销量前10的游戏
top_10_games = df.select("Name", "Global_Sales") \
    .orderBy(desc("Global_Sales")) \
    .limit(10) \
    .toPandas()

# 将所有需要可视化的数据转换为Pandas DataFrame
platform_sales_pd = platform_sales.toPandas()
genre_sales_pd = genre_sales.toPandas()
yearly_stats_pd = yearly_stats.toPandas()
region_distribution = df.select(
    sum("NA_Sales").alias("North_America"),
    sum("EU_Sales").alias("Europe"),
    sum("JP_Sales").alias("Japan"),
    sum("Other_Sales").alias("Other_Regions")
).toPandas()

# 可视化代码（如果在Jupyter Notebook中运行，以下代码将显示图表）
def visualize_data():
    # 设置图表风格
    plt.style.use('ggplot')
    
    # 1. 全球销量前10的游戏
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Global_Sales', y='Name', data=top_10_games)
    plt.title('全球销量前10的游戏')
    plt.xlabel('全球销量 (百万)')
    plt.ylabel('游戏名称')
    plt.tight_layout()
    plt.savefig('top_10_games.png')
    
    # 2. 各平台销售额
    plt.figure(figsize=(12, 8))
    platform_sales_pd = platform_sales_pd.sort_values('Total_Sales', ascending=False).head(10)
    sns.barplot(x='Platform', y='Total_Sales', data=platform_sales_pd)
    plt.title('前10个平台的总销售额')
    plt.xlabel('平台')
    plt.ylabel('总销售额 (百万)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('platform_sales.png')
    
    # 3. 各游戏类型销售额
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Genre', y='Total_Sales', data=genre_sales_pd)
    plt.title('各游戏类型的总销售额')
    plt.xlabel('游戏类型')
    plt.ylabel('总销售额 (百万)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('genre_sales.png')
    
    # 4. 年度趋势分析
    plt.figure(figsize=(14, 6))
    
    # 过滤掉异常年份（如太早或太晚的年份）
    filtered_years = yearly_stats_pd[(yearly_stats_pd['Year'] >= 1980) & (yearly_stats_pd['Year'] <= 2020)]
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1.set_xlabel('年份')
    ax1.set_ylabel('游戏数量', color='tab:blue')
    line1 = ax1.plot(filtered_years['Year'], filtered_years['Game_Count'], color='tab:blue', marker='o', label='游戏数量')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('平均销售额 (百万)', color='tab:red')
    line2 = ax2.plot(filtered_years['Year'], filtered_years['Avg_Global_Sales'], color='tab:red', marker='x', label='平均销售额')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('年度游戏发行量和平均销售额趋势 (1980-2020)')
    plt.tight_layout()
    plt.savefig('yearly_trends.png')
    
    # 5. 地区销售额分布
    plt.figure(figsize=(8, 8))
    regions = region_distribution.iloc[0].index
    values = region_distribution.iloc[0].values
    plt.pie(values, labels=regions, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('全球各地区销售额分布')
    plt.tight_layout()
    plt.savefig('region_distribution.png')
    
    print("所有可视化图表已保存")

# 调用可视化函数（根据环境决定是否调用）
# visualize_data()

# 关闭SparkSession
spark.stop()