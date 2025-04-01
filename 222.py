"""
视频游戏销售数据年份填充脚本
==============================

此脚本用于处理视频游戏销售数据集中的缺失年份数据，并使用多种策略进行填充。

功能：
1. 识别并提取年份为空的游戏记录
2. 使用多种策略填充缺失年份
3. 为每次填充提供可信度评分
4. 提供填充过程的详细统计信息
5. 生成可视化图表展示填充结果

作者：Claude AI
日期：2025-03-31
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Noto Serif CJK JP']  

def fill_missing_years(file_path):
    """
    填充视频游戏销售数据中缺失的年份
    
    参数:
    file_path -- CSV文件路径
    
    返回:
    filled_df -- 包含填充年份的DataFrame
    """
    print(f"正在处理文件: {file_path}")
    
    # 读取数据
    df = pd.read_csv(file_path, encoding='cp1252')
    
    # 找出年份为空的记录
    empty_year_games = df[df['Year'].isna() | (df['Year'] == 'N/A')]
    print(f"年份为空的游戏数量: {len(empty_year_games)}")
    
    # 确保Year列中的数值是数字类型（除了N/A）
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # 构建不同的映射用于填充
    # 1. 游戏名称+平台映射（最高优先级）
    name_platform_year_map = {}
    # 2. 游戏名称映射
    name_year_map = {}
    # 3. 平台+发行商映射
    platform_publisher_years = {}
    
    # 填充映射
    for _, row in df.iterrows():
        if pd.notna(row['Year']):
            # 1. 名称+平台映射
            if pd.notna(row['Name']) and pd.notna(row['Platform']):
                key = f"{row['Name']}_{row['Platform']}"
                name_platform_year_map[key] = row['Year']
            
            # 2. 名称映射
            if pd.notna(row['Name']):
                if row['Name'] not in name_year_map or row['Year'] < name_year_map[row['Name']]:
                    name_year_map[row['Name']] = row['Year']
            
            # 3. 平台+发行商映射
            if pd.notna(row['Platform']) and pd.notna(row['Publisher']):
                key = f"{row['Platform']}_{row['Publisher']}"
                if key not in platform_publisher_years:
                    platform_publisher_years[key] = []
                platform_publisher_years[key].append(row['Year'])
    
    # 计算平台+发行商的平均年份
    platform_publisher_avg_year = {}
    for key, years in platform_publisher_years.items():
        if years:
            platform_publisher_avg_year[key] = round(sum(years) / len(years))
    
    # 为平台创建年份范围和众数
    platform_stats = {}
    for platform, group in df[pd.notna(df['Year'])].groupby('Platform'):
        years = group['Year'].dropna().tolist()
        if years:
            counter = Counter(years)
            platform_stats[platform] = {
                'min': min(years),
                'max': max(years),
                'mode': counter.most_common(1)[0][0] if counter else None,
                'mid': round((min(years) + max(years)) / 2),
                'count': len(years)
            }
    
    # 从游戏名称中提取年份
    def extract_year_from_title(title):
        if pd.isna(title):
            return None
            
        # 匹配完整年份，如 FIFA 2004
        match = re.search(r'\b(19\d{2}|20\d{2})\b', title)
        if match:
            return int(match.group(1))
            
        # 匹配缩写年份，如 NFL '98
        match = re.search(r"\'(\d{2})\b", title)
        if match:
            year = int(match.group(1))
            return 1900 + year if year >= 50 else 2000 + year
            
        # 匹配2K系列，如 NBA 2K17
        match = re.search(r'\b2K(\d{1,2})\b', title)
        if match:
            return 2000 + int(match.group(1))
            
        return None
    
    # 填充年份
    results = []
    unfilled_count = 0
    
    for _, game in empty_year_games.iterrows():
        filled_year = None
        source = None
        confidence = 0
        
        # 1. 精确匹配（游戏名称+平台）
        if pd.notna(game['Name']) and pd.notna(game['Platform']):
            key = f"{game['Name']}_{game['Platform']}"
            if key in name_platform_year_map:
                filled_year = name_platform_year_map[key]
                source = "精确匹配（相同名称和平台）"
                confidence = 0.95
        
        # 2. 从游戏名称中提取年份
        if filled_year is None and pd.notna(game['Name']):
            extracted_year = extract_year_from_title(game['Name'])
            if extracted_year:
                filled_year = extracted_year
                source = "从游戏名称中提取"
                confidence = 0.9
        
        # 3. 同名游戏匹配
        if filled_year is None and pd.notna(game['Name']) and game['Name'] in name_year_map:
            filled_year = name_year_map[game['Name']]
            source = "同名游戏匹配"
            confidence = 0.8
        
        # 4. 平台+发行商匹配
        if filled_year is None and pd.notna(game['Platform']) and pd.notna(game['Publisher']):
            key = f"{game['Platform']}_{game['Publisher']}"
            if key in platform_publisher_avg_year:
                filled_year = platform_publisher_avg_year[key]
                source = "同平台同发行商游戏平均年份"
                confidence = 0.6
        
        # 5. 平台众数
        if filled_year is None and pd.notna(game['Platform']) and game['Platform'] in platform_stats:
            if platform_stats[game['Platform']]['mode'] is not None:
                filled_year = platform_stats[game['Platform']]['mode']
                source = "平台最常见发行年份"
                confidence = 0.4
        
        # 6. 平台中点
        if filled_year is None and pd.notna(game['Platform']) and game['Platform'] in platform_stats:
            filled_year = platform_stats[game['Platform']]['mid']
            source = "平台年份范围中点"
            confidence = 0.3
        
        # 记录结果
        if filled_year is not None:
            results.append({
                'Rank': game['Rank'],
                'Name': game['Name'],
                'Platform': game['Platform'],
                'Original_Year': 'N/A',
                'Filled_Year': filled_year,
                'Source': source,
                'Confidence': confidence,
                'Genre': game['Genre'],
                'Publisher': game['Publisher'],
                'Global_Sales': game['Global_Sales']
            })
        else:
            unfilled_count += 1
            results.append({
                'Rank': game['Rank'],
                'Name': game['Name'],
                'Platform': game['Platform'],
                'Original_Year': 'N/A',
                'Filled_Year': None,
                'Source': '无法填充',
                'Confidence': 0,
                'Genre': game['Genre'],
                'Publisher': game['Publisher'],
                'Global_Sales': game['Global_Sales']
            })
    
    # 创建填充结果DataFrame
    filled_df = pd.DataFrame(results)
    
    # 计算填充统计信息
    filled_count = len(filled_df[pd.notna(filled_df['Filled_Year'])])
    
    print(f"\n填充结果统计:")
    print(f"总缺失年份记录: {len(empty_year_games)}")
    print(f"成功填充记录: {filled_count} ({filled_count/len(empty_year_games)*100:.2f}%)")
    print(f"未能填充记录: {unfilled_count} ({unfilled_count/len(empty_year_games)*100:.2f}%)")
    
    # 按置信度分组
    high_confidence = filled_df[filled_df['Confidence'] >= 0.8]
    medium_confidence = filled_df[(filled_df['Confidence'] >= 0.5) & (filled_df['Confidence'] < 0.8)]
    low_confidence = filled_df[filled_df['Confidence'] < 0.5]
    
    print(f"\n按置信度分组:")
    print(f"高置信度 (≥80%): {len(high_confidence)} ({len(high_confidence)/filled_count*100:.2f}%)")
    print(f"中等置信度 (50-79%): {len(medium_confidence)} ({len(medium_confidence)/filled_count*100:.2f}%)")
    print(f"低置信度 (<50%): {len(low_confidence)} ({len(low_confidence)/filled_count*100:.2f}%)")
    
    # 统计各种填充方法的数量
    method_counts = filled_df['Source'].value_counts()
    
    print("\n填充方法统计:")
    for method, count in method_counts.items():
        if method != '无法填充':
            print(f"{method}: {count} ({count/filled_count*100:.2f}%)")
    
    # 显示高置信度填充示例
    print("\n高置信度填充示例 (前5个):")
    for _, row in high_confidence.head(5).iterrows():
        print(f"{row['Name']} ({row['Platform']}) - 填充年份: {row['Filled_Year']} - 来源: {row['Source']} - 置信度: {row['Confidence']*100:.0f}%")
        
    # 合并回原始数据
    df_filled = df.copy()
    for _, row in filled_df.iterrows():
        if pd.notna(row['Filled_Year']):
            idx = df_filled[df_filled['Rank'] == row['Rank']].index
            if len(idx) > 0:
                df_filled.loc[idx[0], 'Year'] = row['Filled_Year']
                df_filled.loc[idx[0], 'YearSource'] = row['Source']
                df_filled.loc[idx[0], 'YearConfidence'] = f"{row['Confidence']*100:.0f}%"
    
    return df_filled

def visualize_filled_data(file_path):
    """
    可视化填充后的游戏年份数据
    
    参数:
    file_path -- 填充后的CSV文件路径
    """
    # 读取填充后的数据
    df = pd.read_csv(file_path)
    
    # 设置风格
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 创建一个包含多个子图的图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 按置信度级别统计
    ax1 = fig.add_subplot(2, 2, 1)
    confidence_counts = df['YearConfidence'].fillna('原始数据').value_counts()
    
    # 根据置信度排序
    def extract_percentage(x):
        if x == '原始数据':
            return 100
        elif x == '0%':
            return 0
        else:
            return float(x.replace('%', ''))
    
    confidence_counts = confidence_counts.sort_index(key=lambda x: x.map(extract_percentage), ascending=False)
    
    colors = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c', '#cccccc']
    confidence_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('按填充置信度分组的记录数量')
    ax1.set_xlabel('置信度')
    ax1.set_ylabel('记录数量')
    
    # 2. 游戏发行年份分布 (填充前后对比)
    ax2 = fig.add_subplot(2, 2, 2)
    
    # 提取年份列为数值
    df['Year_Numeric'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # 按年份计数
    year_counts = df['Year_Numeric'].value_counts().sort_index()
    year_counts.plot(kind='line', ax=ax2, marker='o', linewidth=2, color='#2c7bb6')
    
    ax2.set_title('游戏发行年份分布 (填充后)')
    ax2.set_xlabel('年份')
    ax2.set_ylabel('游戏数量')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 填充方法分布
    ax3 = fig.add_subplot(2, 2, 3)
    
    method_counts = df['YearSource'].dropna().value_counts()
    method_colors = sns.color_palette('viridis', len(method_counts))
    method_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%', colors=method_colors, 
                      wedgeprops=dict(width=0.5), startangle=90)
    ax3.set_title('填充方法分布')
    ax3.set_ylabel('')
    
    # 4. 平台年份热图
    ax4 = fig.add_subplot(2, 2, 4)
    
    # 获取主要平台
    top_platforms = df['Platform'].value_counts().head(15).index.tolist()
    
    # 创建平台-年份透视表
    platform_year_pivot = pd.pivot_table(
        df[df['Platform'].isin(top_platforms)], 
        values='Rank',  # 使用Rank列只是为了计数
        index='Platform',
        columns='Year_Numeric',
        aggfunc='count',
        fill_value=0
    )
    
    # 按平台游戏总数排序
    platform_totals = platform_year_pivot.sum(axis=1).sort_values(ascending=False)
    platform_year_pivot = platform_year_pivot.loc[platform_totals.index]
    
    # 创建自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#abd9e9', '#2c7bb6'])
    
    # 绘制热图
    sns.heatmap(platform_year_pivot, cmap=cmap, ax=ax4, cbar_kws={'label': '游戏数量'})
    ax4.set_title('主要平台的游戏发行年份分布')
    ax4.set_xlabel('年份')
    ax4.set_ylabel('平台')
    
    # 添加总标题
    plt.suptitle('视频游戏年份数据分析 (含填充数据)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    output_file = 'vgsales_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到 {output_file}")
    
    # 额外创建一个填充记录置信度分析图
    plt.figure(figsize=(10, 8))
    
    # 准备填充记录数据
    filled_records = df[df['YearSource'].notna()]
    confidence_category = filled_records['YearConfidence'].apply(lambda x: 
                                                               '高 (≥80%)' if extract_percentage(x) >= 80 else
                                                               '中 (50-79%)' if extract_percentage(x) >= 50 else
                                                               '低 (<50%)')
    
    confidence_df = pd.DataFrame({
        'Source': filled_records['YearSource'],
        'Confidence': confidence_category
    })
    
    # 创建交叉表
    cross_tab = pd.crosstab(confidence_df['Source'], confidence_df['Confidence'])
    
    # 按高置信度排序
    if '高 (≥80%)' in cross_tab.columns:
        cross_tab = cross_tab.sort_values('高 (≥80%)', ascending=False)
    
    # 绘制堆叠条形图
    ax = cross_tab.plot(kind='barh', stacked=True, figsize=(10, 8), 
                       color=['#2c7bb6', '#abd9e9', '#fdae61'])
    
    plt.title('填充方法与置信度分布')
    plt.xlabel('记录数量')
    plt.ylabel('填充方法')
    plt.legend(title='置信度')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 在每个条形上添加数值标签
    for i, (index, row) in enumerate(cross_tab.iterrows()):
        cumulative = 0
        for j, col in enumerate(cross_tab.columns):
            if row[col] > 0:  # 只有当值大于0时才添加标签
                ax.text(cumulative + row[col]/2, i, str(row[col]), 
                        va='center', ha='center', color='white', fontweight='bold')
            cumulative += row[col]
    
    # 保存图表
    confidence_file = 'confidence_analysis.png'
    plt.savefig(confidence_file, dpi=300, bbox_inches='tight')
    print(f"置信度分析图表已保存到 {confidence_file}")
    
    return df

def export_filled_details(df, output_file="filled_year_details.csv"):
    """
    导出年份填充的详细信息
    
    参数:
    df -- 填充后的DataFrame
    output_file -- 输出文件名
    """
    # 提取填充记录
    filled_records = df[df['YearSource'].notna()].copy()
    
    # 选择需要的列
    export_df = filled_records[['Rank', 'Name', 'Platform', 'Year', 'YearSource', 'YearConfidence', 
                              'Genre', 'Publisher', 'Global_Sales']]
    
    # 重命名列以提高可读性
    export_df = export_df.rename(columns={
        'Year': '填充年份',
        'YearSource': '填充方法',
        'YearConfidence': '置信度'
    })
    
    # 导出到CSV
    export_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"填充详情已导出到 {output_file}")
    
    return export_df

def main():
    """
    主函数，执行完整的填充和可视化流程
    """
    print("=" * 60)
    print("视频游戏销售数据年份填充程序")
    print("=" * 60)
    
    input_file = "vgsales.csv"
    output_file = "vgsales_filled.csv"
    
    try:
        # 1. 填充缺失年份
        print("\n步骤1: 填充缺失年份...")
        df_filled = fill_missing_years(input_file)
        
        # 2. 保存填充后的数据
        df_filled.to_csv(output_file, index=False)
        print(f"\n步骤2: 填充后的数据已保存到 {output_file}")
        
        # 3. 可视化填充结果
        print("\n步骤3: 生成数据可视化...")
        visualize_filled_data(output_file)
        
        # 4. 导出填充详情
        print("\n步骤4: 导出填充详情...")
        export_filled_details(df_filled)
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"\n处理过程中出错: {e}")

if __name__ == "__main__":
    main()