import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def detect_and_handle_outliers(df, method='smart', percent_threshold=5.0, 
                              handling_method='auto', z_threshold=3.0):
    """
    集成化的离群值检测和处理函数
    
    参数:
    df - 输入的数据框
    method - 检测离群值的方法 ('iqr', 'zscore', 'smart')
    percent_threshold - 离群值百分比阈值，超过此阈值将触发处理
    handling_method - 处理离群值的方法 ('cap', 'remove', 'auto')
    z_threshold - Z-Score方法的阈值
    
    返回:
    结果字典，包含:
    - original_df: 原始数据框
    - outlier_stats: 离群值统计
    - cleaned_df: 处理后的数据框
    - treatment_log: 处理日志
    """
    # 复制输入数据框以避免修改原始数据
    df_copy = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # 存储离群值统计和处理日志
    outlier_stats = {}
    treatment_log = {}
    
    # 逐个特征检测离群值
    for col in numeric_cols:
        # IQR方法检测
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound_iqr = Q1 - 1.5 * IQR
        upper_bound_iqr = Q3 + 1.5 * IQR
        
        outliers_iqr = ((df[col] < lower_bound_iqr) | (df[col] > upper_bound_iqr))
        outliers_iqr_count = outliers_iqr.sum()
        outliers_iqr_percent = (outliers_iqr_count / len(df)) * 100
        
        # Z-Score方法检测
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers_z = (z_scores > z_threshold)
        outliers_z_count = outliers_z.sum()
        outliers_z_percent = (outliers_z_count / len(df[col].dropna())) * 100
        
        # 智能选择检测方法
        if method == 'smart':
            # 对于高偏斜数据，IQR方法更可靠
            skewness = abs(df[col].skew())
            if skewness > 1.0:
                selected_method = 'iqr'
                outliers_count = outliers_iqr_count
                outliers_percent = outliers_iqr_percent
                outliers_mask = outliers_iqr
                lower_bound = lower_bound_iqr
                upper_bound = upper_bound_iqr
            else:
                selected_method = 'zscore'
                outliers_count = outliers_z_count
                outliers_percent = outliers_z_percent
                outliers_mask = pd.Series(outliers_z, index=df[col].dropna().index)
                # 为Z-Score方法计算等效的边界
                mean, std = df[col].mean(), df[col].std()
                lower_bound = mean - z_threshold * std
                upper_bound = mean + z_threshold * std
        elif method == 'iqr':
            selected_method = 'iqr'
            outliers_count = outliers_iqr_count
            outliers_percent = outliers_iqr_percent
            outliers_mask = outliers_iqr
            lower_bound = lower_bound_iqr
            upper_bound = upper_bound_iqr
        elif method == 'zscore':
            selected_method = 'zscore'
            outliers_count = outliers_z_count
            outliers_percent = outliers_z_percent
            outliers_mask = pd.Series(outliers_z, index=df[col].dropna().index)
            mean, std = df[col].mean(), df[col].std()
            lower_bound = mean - z_threshold * std
            upper_bound = mean + z_threshold * std
        
        # 存储离群值统计
        outlier_stats[col] = {
            '检测方法': selected_method,
            '离群值数量': outliers_count,
            '离群值百分比': outliers_percent,
            '下界': lower_bound,
            '上界': upper_bound
        }
        
        # 根据离群值百分比决定处理方法
        if outliers_percent > percent_threshold:
            # 自动选择处理方法
            if handling_method == 'auto':
                # 如果离群值太多（>20%），可能数据本身就是偏斜的，用截断更保险
                if outliers_percent > 20.0:
                    actual_method = 'cap'
                else:
                    # 对于少量离群值（5-20%），如果主要集中在一侧，使用截断
                    # 如果两侧都有，并且不超过10%，可以考虑移除
                    lower_count = (df[col] < lower_bound).sum()
                    upper_count = (df[col] > upper_bound).sum()
                    
                    if outliers_percent <= 10.0 and min(lower_count, upper_count) > 0:
                        actual_method = 'remove'
                    else:
                        actual_method = 'cap'
            else:
                actual_method = handling_method
            
            # 应用选定的处理方法
            if actual_method == 'cap':
                # 截断离群值
                df_copy[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                treatment_applied = f"对特征'{col}'的离群值进行了截断处理（{outliers_percent:.2f}%的数据）"
            elif actual_method == 'remove':
                # 注意：这将移除整行数据，可能影响其他特征
                # 创建一个临时的DataFrame以便记录影响
                temp_df = df_copy.copy()
                df_copy = df_copy[~outliers_mask]
                rows_removed = len(temp_df) - len(df_copy)
                treatment_applied = f"从数据集中移除了含有特征'{col}'离群值的{rows_removed}行（{outliers_percent:.2f}%的数据）"
            
            treatment_log[col] = {
                '处理方法': actual_method,
                '处理描述': treatment_applied,
                '处理前离群值百分比': outliers_percent
            }
        else:
            treatment_log[col] = {
                '处理方法': 'none',
                '处理描述': f"特征'{col}'的离群值百分比（{outliers_percent:.2f}%）低于阈值（{percent_threshold}%），未处理",
                '处理前离群值百分比': outliers_percent
            }
    
    # 将统计和日志转换为DataFrame以便更好地展示
    outlier_stats_df = pd.DataFrame(outlier_stats).T
    outlier_stats_df = outlier_stats_df.sort_values('离群值百分比', ascending=False)
    
    treatment_log_df = pd.DataFrame(treatment_log).T
    
    return {
        'original_df': df,
        'outlier_stats': outlier_stats_df,
        'cleaned_df': df_copy,
        'treatment_log': treatment_log_df
    }

def visualize_outlier_treatment(original_df, cleaned_df, feature, figsize=(15, 6)):
    """
    可视化离群值处理前后的分布对比
    
    参数:
    original_df - 原始数据框
    cleaned_df - 处理后的数据框
    feature - 要可视化的特征
    figsize - 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 处理前的分布
    plt.subplot(2, 2, 1)
    sns.histplot(original_df[feature], kde=True, color='blue')
    plt.title(f'{feature} - 处理前的分布')
    
    # 处理前的箱线图
    plt.subplot(2, 2, 2)
    sns.boxplot(y=original_df[feature], color='blue')
    plt.title(f'{feature} - 处理前的箱线图')
    
    # 处理后的分布
    plt.subplot(2, 2, 3)
    sns.histplot(cleaned_df[feature], kde=True, color='green')
    plt.title(f'{feature} - 处理后的分布')
    
    # 处理后的箱线图
    plt.subplot(2, 2, 4)
    sns.boxplot(y=cleaned_df[feature], color='green')
    plt.title(f'{feature} - 处理后的箱线图')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息对比
    print("===== 处理前统计 =====")
    print(f"计数: {original_df[feature].count()}")
    print(f"平均值: {original_df[feature].mean():.4f}")
    print(f"标准差: {original_df[feature].std():.4f}")
    print(f"最小值: {original_df[feature].min():.4f}")
    print(f"Q1 (25%): {original_df[feature].quantile(0.25):.4f}")
    print(f"中位数: {original_df[feature].median():.4f}")
    print(f"Q3 (75%): {original_df[feature].quantile(0.75):.4f}")
    print(f"最大值: {original_df[feature].max():.4f}")
    print(f"偏度: {original_df[feature].skew():.4f}")
    print(f"峰度: {original_df[feature].kurtosis():.4f}")
    
    print("\n===== 处理后统计 =====")
    print(f"计数: {cleaned_df[feature].count()}")
    print(f"平均值: {cleaned_df[feature].mean():.4f}")
    print(f"标准差: {cleaned_df[feature].std():.4f}")
    print(f"最小值: {cleaned_df[feature].min():.4f}")
    print(f"Q1 (25%): {cleaned_df[feature].quantile(0.25):.4f}")
    print(f"中位数: {cleaned_df[feature].median():.4f}")
    print(f"Q3 (75%): {cleaned_df[feature].quantile(0.75):.4f}")
    print(f"最大值: {cleaned_df[feature].max():.4f}")
    print(f"偏度: {cleaned_df[feature].skew():.4f}")
    print(f"峰度: {cleaned_df[feature].kurtosis():.4f}")

def batch_process_features(df, features_list, percent_threshold=5.0, handling_method='auto'):
    """
    批量处理多个特征的离群值
    
    参数:
    df - 输入的数据框
    features_list - 要处理的特征列表
    percent_threshold - 离群值百分比阈值
    handling_method - 处理方法
    
    返回:
    处理后的数据框
    """
    result_df = df.copy()
    all_stats = {}
    all_logs = {}
    
    for feature in features_list:
        if feature not in df.columns:
            print(f"特征 {feature} 不存在，跳过")
            continue
            
        # 处理单个特征
        feature_df = result_df[[feature]].copy()
        result = detect_and_handle_outliers(
            feature_df, 
            percent_threshold=percent_threshold, 
            handling_method=handling_method
        )
        
        # 更新数据框
        if handling_method != 'remove':
            # 如果是截断，只更新该特征的值
            result_df[feature] = result['cleaned_df'][feature]
        else:
            # 如果是移除，需要保留索引一致性
            result_df = result_df.loc[result['cleaned_df'].index]
        
        # 收集统计和日志
        all_stats[feature] = result['outlier_stats'].loc[feature]
        all_logs[feature] = result['treatment_log'].loc[feature]
        
    # 汇总所有特征的统计和日志
    all_stats_df = pd.DataFrame(all_stats).T
    all_logs_df = pd.DataFrame(all_logs).T
    
    print("===== 离群值处理汇总 =====")
    print(f"原始数据行数: {len(df)}")
    print(f"处理后数据行数: {len(result_df)}")
    print(f"移除的行数: {len(df) - len(result_df)}")
    
    return {
        'original_df': df,
        'cleaned_df': result_df,
        'outlier_stats': all_stats_df,
        'treatment_log': all_logs_df
    }

# 使用示例
if __name__ == "__main__":
    # 加载数据
    # df = pd.read_csv("your_data_file.csv")
    
    # 示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'normal_feature': np.random.normal(0, 1, 1000),
        'skewed_feature': np.random.exponential(2, 1000),
        'outlier_feature': np.random.normal(0, 1, 1000)
    })
    
    # 添加一些离群值
    df.loc[0:9, 'outlier_feature'] = np.random.normal(10, 1, 10)
    
    # 使用离群值检测和处理工具
    result = detect_and_handle_outliers(df, percent_threshold=0.5)
    
    print("\n离群值统计:")
    print(result['outlier_stats'])
    
    print("\n处理日志:")
    print(result['treatment_log'])
    
    # 可视化处理效果
    visualize_outlier_treatment(
        result['original_df'], 
        result['cleaned_df'], 
        'outlier_feature'
    ) 