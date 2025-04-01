
#%%
import findspark
findspark.init('/usr/local/spark')  # 确保 Spark 环境正确
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.sql.functions import col, desc, sum as spark_sum, avg, count, year, lit, when
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
import logging
#%%
# 初始化Spark会话
spark = SparkSession.builder \
    .appName("VideoGameSalesAnalysis") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext
# 设置 PySpark 日志级别
sc.setLogLevel("ERROR")
# 获取 log4j 日志对象
log4j_logger = sc._jvm.org.apache.log4j
log4j_logger.LogManager.getRootLogger().setLevel(log4j_logger.Level.ERROR)
# 设置 Python 端的日志级别（避免 py4j 产生过多 INFO 日志）
logging.getLogger("py4j").setLevel(logging.ERROR)
# 加载数据集
# 注意：这里假设你已下载Kaggle数据集到本地
# 数据集链接: https://www.kaggle.com/datasets/gregorut/videogamesales
sdf = spark.read.csv('hdfs://localhost:9000/qimo/cleaned_data.csv', header=True, inferSchema=True)

# 显示数据结构
print("数据结构:")
sdf.printSchema()

# 显示前几行数据
print("\n数据预览:")
sdf.show(5)

# 基本数据探索
print("\n数据集统计信息:")
sdf.describe().show()

# 检查缺失值
print("\n缺失值统计:")
for col_name in sdf.columns:
    missing_count = sdf.filter(sdf[col_name].isNull()).count()
    print(f"列 '{col_name}' 的缺失值数量: {missing_count}")

# 数据清洗 - 处理缺失值
sdf_cleaned = sdf.na.drop()
print(f"\n清洗后的行数: {sdf_cleaned.count()} (原始行数: {sdf.count()})")

#%%
# 1. 计算Publisher与Genre的相关系数并作图
# 首先将分类变量转换为数值
publisher_indexer = StringIndexer(inputCol="Publisher", outputCol="Publisher_index", handleInvalid="skip")
genre_indexer = StringIndexer(inputCol="Genre", outputCol="Genre_index", handleInvalid="skip")

# 应用转换
sdf_indexed = publisher_indexer.fit(sdf_cleaned).transform(sdf_cleaned)
sdf_indexed = genre_indexer.fit(sdf_indexed).transform(sdf_indexed)

# 创建列联表（交叉表）进行卡方检验
# 将数据转换为Pandas进行处理
pandas_df = sdf_indexed.select("Publisher_index", "Genre_index").toPandas()

# 创建交叉表
cross_tab = pd.crosstab(pandas_df["Publisher_index"], pandas_df["Genre_index"])

# 执行卡方检验
chi2, p, dof, expected = chi2_contingency(cross_tab)
print(f"\n卡方检验结果: chi2={chi2}, p值={p}")

# 使用Cramer's V系数衡量关联强度
n = cross_tab.sum().sum()
cramer_v = np.sqrt(chi2 / (n * (min(cross_tab.shape) - 1)))
print(f"Publisher与Genre的Cramer's V系数: {cramer_v}")

# 生成热图以可视化关联
# 获取前15个发行商和所有类型进行可视化
top_publishers = sdf_cleaned.groupBy("Publisher") \
    .count() \
    .orderBy(desc("count")) \
    .limit(30) \
    .select("Publisher") \
    .rdd.flatMap(lambda x: x) \
    .collect()

# 创建过滤后的数据框
filtered_sdf = sdf_cleaned.filter(col("Publisher").isin(top_publishers))

# 转换为pandas进行可视化
publisher_genre_df = filtered_sdf.groupBy("Publisher", "Genre") \
    .count() \
    .toPandas()

# 创建透视表
pivot_table = publisher_genre_df.pivot(index="Publisher", columns="Genre", values="count").fillna(0)

# 创建热图
plt.rcParams['font.sans-serif'] = ['Noto Serif CJK JP']  
plt.figure(figsize=(15, 10))
sns.heatmap(pivot_table, annot=False, cmap="YlGnBu", linewidths=.5)
plt.title('Publisher与Genre的关联热图')
plt.savefig('publisher_genre_heatmap.png')
print("\n已保存Publisher与Genre关联热图到publisher_genre_heatmap.png")

#%%

# 2. 时间序列预测2017年全球销售额
# 转换Year列为数值类型
sdf_with_year = sdf_cleaned.withColumn("Year", col("Year").cast(DoubleType()))

# 按年份汇总全球销售额
yearly_sales = sdf_with_year.filter(col("Year").isNotNull()) \
    .groupBy("Year") \
    .agg(spark_sum("Global_Sales").alias("Total_Global_Sales")) \
    .orderBy("Year")

print("\n按年份汇总的全球销售额:")
yearly_sales.show(10)

# 转换为Pandas DataFrame进行时间序列分析
yearly_sales_pd = yearly_sales.toPandas()
yearly_sales_pd.set_index("Year", inplace=True)
yearly_sales_pd.sort_index(inplace=True)

# 使用statsmodels进行ARIMA时间序列预测
# 首先进行数据可视化
plt.figure(figsize=(12, 6))
plt.plot(yearly_sales_pd.index, yearly_sales_pd["Total_Global_Sales"], marker='o')
plt.title("视频游戏全球销售额历史趋势")
plt.xlabel("年份")
plt.ylabel("全球销售额 (百万美元)")
plt.grid(True, alpha=0.3)
plt.savefig("sales_history.png")

# 导入时间序列分析所需的库
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 检查时间序列的平稳性
result = adfuller(yearly_sales_pd["Total_Global_Sales"])
print(f"\nADF统计量: {result[0]}")
print(f"p值: {result[1]}")
print(f"临界值: {result[4]}")
if result[1] <= 0.05:
    print("时间序列是平稳的")
else:
    print("时间序列不是平稳的，需要差分")
    
# 绘制ACF和PACF图以确定ARIMA模型的p和q参数
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(211)
plot_acf(yearly_sales_pd["Total_Global_Sales"], ax=ax1, lags=10)
ax2 = plt.subplot(212)
plot_pacf(yearly_sales_pd["Total_Global_Sales"], ax=ax2, lags=10)
plt.tight_layout()
plt.savefig("acf_pacf_plots.png")

# 如果时间序列不平稳，进行差分
diff_order = 0
sales_data = yearly_sales_pd["Total_Global_Sales"]
if result[1] > 0.05:
    diff_order = 1
    sales_data = yearly_sales_pd["Total_Global_Sales"].diff().dropna()
    
    # 再次检查差分后的平稳性
    result_diff = adfuller(sales_data)
    print(f"\n差分后ADF统计量: {result_diff[0]}")
    print(f"差分后p值: {result_diff[1]}")
    
    # 如果仍不平稳，使用二阶差分
    if result_diff[1] > 0.05:
        diff_order = 2
        sales_data = sales_data.diff().dropna()

# 确定ARIMA模型的参数
# 基于ACF/PACF图形，或使用网格搜索确定最佳参数
# 例如使用简单的参数组合
best_aic = float("inf")
best_order = None
best_model = None

# 可能的p、d、q取值范围
p_values = range(0, 3)
d_values = [diff_order]  # 基于平稳性检验结果
q_values = range(0, 3)

# 网格搜索找到最佳ARIMA模型
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = ARIMA(yearly_sales_pd["Total_Global_Sales"], order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except:
                continue

print(f"\n最佳ARIMA模型参数: {best_order}")
print(f"AIC值: {best_aic}")

# 使用最佳模型进行预测
if best_model is not None:
    # 获取最大年份
    max_year = yearly_sales_pd.index.max()
    # 预测未来年份直到2017年
    forecast_steps = int(2017 - max_year)
    if forecast_steps > 0:
        forecast = best_model.forecast(steps=forecast_steps)
        forecast_years = np.arange(max_year + 1, max_year + forecast_steps + 1)
        
        # 2017年的预测值
        predicted_2017 = forecast.iloc[-1]
        print(f"\n预测2017年全球视频游戏销售额: {predicted_2017:.2f} 百万美元")
        
        # 创建预测可视化
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_sales_pd.index, yearly_sales_pd["Total_Global_Sales"], 
                 marker='o', label="历史销售额")
        plt.plot(forecast_years, forecast, marker='*', color='red', 
                 label="ARIMA预测", linestyle='--')
        plt.axvline(x=max_year, color='gray', linestyle='--', alpha=0.5)
        plt.scatter(2017, predicted_2017, color='green', s=100, 
                   label="2017年预测", zorder=5)
        plt.xlabel("年份")
        plt.ylabel("全球销售额 (百万美元)")
        plt.title(f"ARIMA({best_order[0]},{best_order[1]},{best_order[2]})模型预测2017年全球视频游戏销售额")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("arima_forecast.png")
        print("\n已保存ARIMA预测图表到arima_forecast.png")
    else:
        print("数据集已包含2017年，无需预测")
else:
    print("未能找到合适的ARIMA模型")
    
# 比较多种预测模型的性能
# 使用简单的线性回归作为基线
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 准备线性回归训练数据
X = yearly_sales_pd.index.values.reshape(-1, 1)
y = yearly_sales_pd["Total_Global_Sales"].values

# 拆分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 评估模型
lr_predictions = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
print(f"\n线性回归模型RMSE: {lr_rmse:.2f}")

# 预测2017年销售额
if 2017 > yearly_sales_pd.index.max():
    lr_prediction_2017 = lr_model.predict([[2017]])[0]
    print(f"线性回归预测2017年全球销售额: {lr_prediction_2017:.2f} 百万美元")
    
    # 绘制线性回归预测图
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='blue', label='历史数据')
    plt.plot(X, lr_model.predict(X), color='red', label='线性回归')
    plt.scatter(2017, lr_prediction_2017, color='green', s=100, 
               label="线性回归2017年预测")
    plt.xlabel("年份")
    plt.ylabel("全球销售额 (百万美元)")
    plt.title("线性回归预测2017年全球视频游戏销售额")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("linear_regression_forecast.png")
    
    # 对比不同预测模型
    if 'predicted_2017' in locals():
        print("\n不同模型2017年预测对比:")
        print(f"ARIMA模型: {predicted_2017:.2f} 百万美元")
        print(f"线性回归模型: {lr_prediction_2017:.2f} 百万美元")
        
        # 创建预测模型对比图
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_sales_pd.index, yearly_sales_pd["Total_Global_Sales"], 
                marker='o', label="历史销售额")
        
        # 添加线性回归预测线
        future_years = np.arange(min(yearly_sales_pd.index), 2018, 1).reshape(-1, 1)
        plt.plot(future_years, lr_model.predict(future_years), 
                color='red', linestyle='--', label="线性回归预测")
        
        # 添加ARIMA预测线
        if best_model is not None and forecast_steps > 0:
            plt.plot(forecast_years, forecast, marker='*', color='green', 
                    label="ARIMA预测", linestyle='-.')
        
        plt.axvline(x=max_year, color='gray', linestyle='--', alpha=0.5)
        plt.scatter(2017, lr_prediction_2017, color='red', s=80, marker='s',
                   label="线性回归2017年预测")
        plt.scatter(2017, predicted_2017, color='green', s=80, marker='*',
                   label="ARIMA 2017年预测")
        
        plt.xlabel("年份")
        plt.ylabel("全球销售额 (百万美元)")
        plt.title("不同模型预测2017年全球视频游戏销售额对比")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("model_comparison.png")
        print("\n已保存预测模型对比图表到model_comparison.png")

#%%
# 3. K-Means聚类 - 按销售特征分组游戏，并与实际游戏类别对比
# 准备聚类数据
kmeans_assembler = VectorAssembler(
    inputCols=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"],
    outputCol="features",
    handleInvalid="skip"
)

kmeans_data = kmeans_assembler.transform(sdf_cleaned)

# 训练KMeans模型，使用与Genre数量相同的聚类数
genre_count = sdf_cleaned.select("Genre").distinct().count()
print(f"\n数据集中独特的游戏类型数量: {genre_count}")

kmeans = KMeans(k=int(genre_count), seed=42)
kmeans_model = kmeans.fit(kmeans_data)

# 获取聚类结果
clustered_data = kmeans_model.transform(kmeans_data)
clustered_data = clustered_data.withColumnRenamed("prediction", "cluster")

# 查看每个聚类的中心
centers = kmeans_model.clusterCenters()
print("\nK-Means聚类中心 (NA_Sales, EU_Sales, JP_Sales, Other_Sales):")
for i, center in enumerate(centers):
    print(f"聚类 {i}: {center}")

# 分析每个聚类的游戏数量
cluster_counts = clustered_data.groupBy("cluster").count().orderBy("cluster")
print("\n每个聚类的游戏数量:")
cluster_counts.show()

# 将聚类结果与实际游戏类别进行比较
cluster_genre = clustered_data.groupBy("cluster", "Genre").count()
cluster_genre_total = cluster_genre.groupBy("cluster").agg(spark_sum("count").alias("total"))

# 计算每个聚类中各类型游戏的百分比
cluster_genre_pct = cluster_genre.join(
    cluster_genre_total, 
    on="cluster"
).withColumn(
    "percentage", 
    (col("count") / col("total") * 100).cast("double")
)

# 显示每个聚类的主要游戏类型
print("\n聚类与Genre的对比分析:")
for i in range(int(genre_count)):
    print(f"\n聚类 {i} 的游戏类型分布:")
    cluster_genre_pct.filter(col("cluster") == i) \
        .orderBy(desc("percentage")) \
        .select("Genre", "count", "percentage") \
        .show(5)

# 将聚类结果和原始数据转换为Pandas DataFrame
feature_pd = kmeans_data.select("Name", "Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales").toPandas()
cluster_pd = clustered_data.select("Name", "cluster").toPandas()
result_pd = pd.merge(feature_pd, cluster_pd, on="Name")

# 为更好的可视化效果，对超出正常范围的销售数据进行过滤
# 计算每列的百分位数
percentile_99 = result_pd[['NA_Sales', 'EU_Sales', 'JP_Sales']].quantile(0.99)
# 过滤异常值
plot_data = result_pd.copy()
for col in ['NA_Sales', 'EU_Sales', 'JP_Sales']:
    plot_data = plot_data[plot_data[col] <= percentile_99[col]]

# 创建一个字典将聚类ID映射到颜色
num_clusters = int(genre_count)
cluster_colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
cluster_color_map = {i: cluster_colors[i] for i in range(num_clusters)}

# 创建一个字典将Genre映射到颜色
unique_genres = plot_data['Genre'].unique()
genre_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))
genre_color_map = {genre: genre_colors[i] for i, genre in enumerate(unique_genres)}

# 1. 按聚类结果绘制3D散点图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 为每个聚类绘制点
for cluster_id in range(num_clusters):
    cluster_data = plot_data[plot_data['cluster'] == cluster_id]
    ax.scatter(
        cluster_data['NA_Sales'], 
        cluster_data['EU_Sales'], 
        cluster_data['JP_Sales'],
        color=cluster_color_map[cluster_id],
        label=f'聚类 {cluster_id}',
        alpha=0.7,
        s=30
    )

# 绘制聚类中心
centers_df = pd.DataFrame(centers, columns=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
for i, center in centers_df.iterrows():
    if center['NA_Sales'] <= percentile_99['NA_Sales'] and \
       center['EU_Sales'] <= percentile_99['EU_Sales'] and \
       center['JP_Sales'] <= percentile_99['JP_Sales']:
        ax.scatter(
            center['NA_Sales'], 
            center['EU_Sales'], 
            center['JP_Sales'],
            color=cluster_color_map[i],
            s=200,
            marker='*',
            edgecolors='black',
            linewidth=1.5
        )

# 添加轴标签和标题
ax.set_xlabel('北美销售额 (百万美元)')
ax.set_ylabel('欧洲销售额 (百万美元)')
ax.set_zlabel('日本销售额 (百万美元)')
ax.set_title('基于销售区域的K-Means聚类结果')

# 添加图例
# 由于图例项可能很多，创建一个单独的图例
legend1 = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="聚类")

# 保存图表
plt.tight_layout()
plt.savefig("kmeans_3d_clusters.png", bbox_inches='tight')
print("\n已保存聚类3D散点图到kmeans_3d_clusters.png")

# 2. 按原始游戏类型绘制3D散点图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 为每个游戏类型绘制点
for genre in unique_genres:
    genre_data = plot_data[plot_data['Genre'] == genre]
    ax.scatter(
        genre_data['NA_Sales'], 
        genre_data['EU_Sales'], 
        genre_data['JP_Sales'],
        color=genre_color_map[genre],
        label=genre,
        alpha=0.7,
        s=30
    )

# 添加轴标签和标题
ax.set_xlabel('北美销售额 (百万美元)')
ax.set_ylabel('欧洲销售额 (百万美元)')
ax.set_zlabel('日本销售额 (百万美元)')
ax.set_title('基于原始游戏类型的销售区域分布')

# 添加图例
# 分成两列显示图例
legend2 = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), 
                    title="游戏类型", ncol=1)

# 保存图表
plt.tight_layout()
plt.savefig("genre_3d_distribution.png", bbox_inches='tight')
print("\n已保存游戏类型3D散点图到genre_3d_distribution.png")

# 3. 创建交互式3D图表（使用plotly）
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 3.1 按聚类结果创建交互式3D图
fig_cluster = px.scatter_3d(
    plot_data, 
    x='NA_Sales', 
    y='EU_Sales', 
    z='JP_Sales',
    color='cluster',
    color_continuous_scale='rainbow',
    hover_name='Name',
    hover_data=['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales'],
    labels={'NA_Sales': '北美销售额', 'EU_Sales': '欧洲销售额', 'JP_Sales': '日本销售额'},
    title='基于销售区域的K-Means聚类结果 (交互式)'
)

# 添加聚类中心
centers_df = pd.DataFrame(centers, columns=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
centers_df['cluster'] = centers_df.index
filtered_centers = centers_df[
    (centers_df['NA_Sales'] <= percentile_99['NA_Sales']) &
    (centers_df['EU_Sales'] <= percentile_99['EU_Sales']) &
    (centers_df['JP_Sales'] <= percentile_99['JP_Sales'])
]

fig_cluster.add_trace(
    go.Scatter3d(
        x=filtered_centers['NA_Sales'],
        y=filtered_centers['EU_Sales'],
        z=filtered_centers['JP_Sales'],
        mode='markers',
        marker=dict(
            size=10,
            symbol='diamond',
            color=filtered_centers['cluster'],
            colorscale='rainbow',
            line=dict(color='black', width=2)
        ),
        name='聚类中心'
    )
)

# 保存为HTML文件以便交互式查看
fig_cluster.write_html("kmeans_clusters_interactive.html")
print("\n已保存交互式聚类3D图表到kmeans_clusters_interactive.html")

# 3.2 按原始游戏类型创建交互式3D图
fig_genre = px.scatter_3d(
    plot_data, 
    x='NA_Sales', 
    y='EU_Sales', 
    z='JP_Sales',
    color='Genre',
    hover_name='Name',
    hover_data=['cluster', 'NA_Sales', 'EU_Sales', 'JP_Sales'],
    labels={'NA_Sales': '北美销售额', 'EU_Sales': '欧洲销售额', 'JP_Sales': '日本销售额'},
    title='基于原始游戏类型的销售区域分布 (交互式)'
)

# 保存为HTML文件以便交互式查看
fig_genre.write_html("genre_distribution_interactive.html")
print("\n已保存交互式游戏类型3D图表到genre_distribution_interactive.html")

# 4. 创建对比分析图表 - 使用两个子图并排比较聚类与原始类型
# 创建一个2x2的子图网格
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=('基于K-Means聚类', '基于原始游戏类型')
)

# 子图1: 按聚类着色
for cluster_id in range(num_clusters):
    cluster_data = plot_data[plot_data['cluster'] == cluster_id]
    fig.add_trace(
        go.Scatter3d(
            x=cluster_data['NA_Sales'],
            y=cluster_data['EU_Sales'],
            z=cluster_data['JP_Sales'],
            mode='markers',
            marker=dict(
                size=5,
                color=cluster_id,
                colorscale='rainbow',
                opacity=0.7
            ),
            name=f'聚类 {cluster_id}',
            showlegend=(cluster_id < 5)  # 只显示前5个聚类的图例以避免过于拥挤
        ),
        row=1, col=1
    )

# 添加聚类中心到第一个子图
for _, center in filtered_centers.iterrows():
    fig.add_trace(
        go.Scatter3d(
            x=[center['NA_Sales']],
            y=[center['EU_Sales']],
            z=[center['JP_Sales']],
            mode='markers',
            marker=dict(
                size=10,
                symbol='diamond',
                color=center['cluster'],
                colorscale='rainbow',
                line=dict(color='black', width=1)
            ),
            name='聚类中心',
            showlegend=False
        ),
        row=1, col=1
    )

# 子图2: 按游戏类型着色
for i, genre in enumerate(unique_genres):
    if i < 10:  # 限制图例显示的类别数量，防止过于拥挤
        show_legend = True
    else:
        show_legend = False
        
    genre_data = plot_data[plot_data['Genre'] == genre]
    fig.add_trace(
        go.Scatter3d(
            x=genre_data['NA_Sales'],
            y=genre_data['EU_Sales'],
            z=genre_data['JP_Sales'],
            mode='markers',
            marker=dict(
                size=5,
                color=i,
                colorscale='hsv',
                opacity=0.7
            ),
            name=genre,
            showlegend=show_legend
        ),
        row=1, col=2
    )

# 更新布局
fig.update_layout(
    title_text="聚类结果与原始游戏类型的销售分布对比",
    height=800,
    width=1400,
    legend=dict(
        x=1.05,
        y=0.5
    ),
    scene=dict(
        xaxis_title='北美销售额',
        yaxis_title='欧洲销售额',
        zaxis_title='日本销售额'
    ),
    scene2=dict(
        xaxis_title='北美销售额',
        yaxis_title='欧洲销售额',
        zaxis_title='日本销售额'
    )
)

# 保存为HTML文件以便交互式查看
fig.write_html("comparison_interactive.html")
print("\n已保存聚类与游戏类型对比的3D图表到comparison_interactive.html")

# 5. 创建2D投影图表，帮助更清晰地理解聚类结果
# 创建三个2D投影视图：NA-EU、NA-JP、EU-JP
fig = plt.figure(figsize=(18, 6))

# NA-EU投影
ax1 = fig.add_subplot(131)
for cluster_id in range(num_clusters):
    cluster_data = plot_data[plot_data['cluster'] == cluster_id]
    ax1.scatter(
        cluster_data['NA_Sales'], 
        cluster_data['EU_Sales'],
        color=cluster_color_map[cluster_id],
        label=f'聚类 {cluster_id}' if cluster_id < 5 else "",  # 只显示前5个聚类的图例
        alpha=0.5,
        s=20
    )
    
# 添加聚类中心
for i, center in centers_df.iterrows():
    if center['NA_Sales'] <= percentile_99['NA_Sales'] and \
       center['EU_Sales'] <= percentile_99['EU_Sales']:
        ax1.scatter(
            center['NA_Sales'], 
            center['EU_Sales'],
            color=cluster_color_map[i],
            s=150,
            marker='*',
            edgecolors='black',
            linewidth=1.5
        )
        
ax1.set_xlabel('北美销售额 (百万美元)')
ax1.set_ylabel('欧洲销售额 (百万美元)')
ax1.set_title('北美-欧洲销售投影')
ax1.grid(True, alpha=0.3)
ax1.legend()

# NA-JP投影
ax2 = fig.add_subplot(132)
for cluster_id in range(num_clusters):
    cluster_data = plot_data[plot_data['cluster'] == cluster_id]
    ax2.scatter(
        cluster_data['NA_Sales'], 
        cluster_data['JP_Sales'],
        color=cluster_color_map[cluster_id],
        alpha=0.5,
        s=20
    )
    
# 添加聚类中心
for i, center in centers_df.iterrows():
    if center['NA_Sales'] <= percentile_99['NA_Sales'] and \
       center['JP_Sales'] <= percentile_99['JP_Sales']:
        ax2.scatter(
            center['NA_Sales'], 
            center['JP_Sales'],
            color=cluster_color_map[i],
            s=150,
            marker='*',
            edgecolors='black',
            linewidth=1.5
        )
        
ax2.set_xlabel('北美销售额 (百万美元)')
ax2.set_ylabel('日本销售额 (百万美元)')
ax2.set_title('北美-日本销售投影')
ax2.grid(True, alpha=0.3)

# EU-JP投影
ax3 = fig.add_subplot(133)
for cluster_id in range(num_clusters):
    cluster_data = plot_data[plot_data['cluster'] == cluster_id]
    ax3.scatter(
        cluster_data['EU_Sales'], 
        cluster_data['JP_Sales'],
        color=cluster_color_map[cluster_id],
        alpha=0.5,
        s=20
    )
    
# 添加聚类中心
for i, center in centers_df.iterrows():
    if center['EU_Sales'] <= percentile_99['EU_Sales'] and \
       center['JP_Sales'] <= percentile_99['JP_Sales']:
        ax3.scatter(
            center['EU_Sales'], 
            center['JP_Sales'],
            color=cluster_color_map[i],
            s=150,
            marker='*',
            edgecolors='black',
            linewidth=1.5
        )
        
ax3.set_xlabel('欧洲销售额 (百万美元)')
ax3.set_ylabel('日本销售额 (百万美元)')
ax3.set_title('欧洲-日本销售投影')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kmeans_2d_projections.png")
print("\n已保存2D投影图表到kmeans_2d_projections.png")

# 6. 分析不同聚类组与游戏类型的关联
# 创建一个热图，显示每个聚类中各游戏类型的分布
contingency_table = pd.crosstab(
    result_pd['cluster'], 
    result_pd['Genre'], 
    normalize='index'
) * 100  # 转换为百分比

plt.figure(figsize=(16, 10))
sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt=".1f")
plt.title('聚类与游戏类型的关联热图 (百分比)')
plt.xlabel('游戏类型')
plt.ylabel('聚类ID')
plt.xticks(rotation=45, ha='right')
plt.savefig("cluster_genre_heatmap.png", bbox_inches='tight')
print("\n已保存聚类与游戏类型关联热图到cluster_genre_heatmap.png")

# 7. 分析聚类的区域销售特征
# 计算每个聚类的平均区域销售额
cluster_sales = clustered_data.groupBy("cluster").agg(
    avg("NA_Sales").alias("Avg_NA_Sales"),
    avg("EU_Sales").alias("Avg_EU_Sales"),
    avg("JP_Sales").alias("Avg_JP_Sales"),
    avg("Other_Sales").alias("Avg_Other_Sales")
).orderBy("cluster")

# 打印每个聚类的平均销售情况
print("\n各聚类的平均区域销售额:")
cluster_sales.show()

# 将结果保存为CSV
cluster_sales.toPandas().to_csv("cluster_sales_analysis.csv", index=False)
print("\n已保存聚类销售分析到cluster_sales_analysis.csv")
# 关闭Spark会话
spark.stop()