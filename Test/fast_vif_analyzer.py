import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import timedelta
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def fast_vif_calculation(X, use_parallel=True, n_jobs=-1, verbose=1):
    """
    高效计算VIF值
    
    参数:
    X - 特征数据框
    use_parallel - 是否使用并行计算
    n_jobs - 并行作业数，-1表示使用所有可用核心
    verbose - 显示详细程度 (0=无输出, 1=基本信息, 2=详细信息)
    
    返回:
    vif_df - 包含VIF值的数据框
    """
    if verbose > 0:
        print(f"计算 {X.shape[1]} 个特征的VIF值...")
        start_time = time.time()
    
    # 提前检查特征相关性，移除完全相关特征可加速计算
    if X.shape[1] > 10:  # 仅在特征较多时执行
        corr_matrix = X.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        highly_correlated = (corr_matrix > 0.999).any()
        if highly_correlated.any():
            problem_features = highly_correlated[highly_correlated].index.tolist()
            if verbose > 0:
                print(f"警告: 发现 {len(problem_features)} 个几乎完全相关的特征，可能导致VIF计算不稳定")
    
    # 高效并行计算VIF
    def _calculate_vif(i):
        try:
            return variance_inflation_factor(X.values, i)
        except:
            return np.nan
    
    if use_parallel and X.shape[1] > 10:
        if verbose > 0:
            print("使用并行计算VIF值...")
        vifs = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_vif)(i) for i in range(X.shape[1])
        )
    else:
        # 顺序计算，适用于特征数较少的情况
        vifs = [_calculate_vif(i) for i in range(X.shape[1])]
    
    # 创建结果数据框
    vif_df = pd.DataFrame({
        'feature': X.columns,
        'VIF': vifs
    }).sort_values('VIF', ascending=False)
    
    # 处理潜在的无穷大值
    vif_df.loc[np.isinf(vif_df['VIF']), 'VIF'] = float(1e9)
    vif_df.loc[np.isnan(vif_df['VIF']), 'VIF'] = float(1e9)
    
    if verbose > 0:
        elapsed = time.time() - start_time
        print(f"VIF计算完成，耗时: {elapsed:.2f}秒")
        if verbose > 1:
            print(f"\nVIF值摘要:\n{vif_df.head(10)}")
    
    return vif_df

def efficient_vif_elimination(X, thresh=10.0, max_iter=None, early_stop=True, verbose=1):
    """
    高效迭代移除高VIF特征
    
    参数:
    X - 特征数据框
    thresh - VIF阈值
    max_iter - 最大迭代次数，None表示不限制
    early_stop - 是否在连续3次迭代VIF减少不显著时提前停止
    verbose - 显示详细程度
    
    返回:
    X_reduced - 移除高VIF特征后的数据框
    eliminated_features - 被移除特征及其VIF值的列表
    """
    if max_iter is None:
        max_iter = X.shape[1] - 1  # 最多可以删除n-1个特征
    
    X_reduced = X.copy()
    eliminated_features = []
    
    if verbose > 0:
        print(f"开始VIF筛选过程 (阈值 = {thresh})...")
        start_time = time.time()
        print(f"初始特征数量: {X_reduced.shape[1]}")
    
    # 用于提前停止的变量
    previous_max_vif = float('inf')
    plateau_counter = 0
    
    for i in range(max_iter):
        # 仅计算一次VIF，避免重复计算
        vif_df = fast_vif_calculation(X_reduced, verbose=0)
        
        max_vif = vif_df['VIF'].max()
        
        # 提前停止检查
        if early_stop:
            vif_improvement = previous_max_vif - max_vif
            if max_vif > thresh and vif_improvement < thresh * 0.01:
                plateau_counter += 1
                if plateau_counter >= 3:
                    if verbose > 0:
                        print(f"\n提前停止: VIF改善不显著 ({vif_improvement:.2f})")
                    break
            else:
                plateau_counter = 0
            previous_max_vif = max_vif
        
        # 检查是否所有VIF值已经低于阈值
        if max_vif <= thresh:
            if verbose > 0:
                print(f"\n所有特征VIF值已低于阈值 {thresh}")
            break
        
        # 获取VIF最高的特征
        feature_to_drop = vif_df.iloc[0]['feature']
        vif_value = vif_df.iloc[0]['VIF']
        
        # 记录并移除
        eliminated_features.append((feature_to_drop, vif_value))
        X_reduced = X_reduced.drop(columns=[feature_to_drop])
        
        if verbose > 0:
            if i % max(1, max_iter//10) == 0 or verbose > 1:  # 减少进度输出频率
                elapsed = time.time() - start_time
                remaining = (elapsed / (i+1)) * (max_iter - (i+1)) if i > 0 else 0
                print(f"迭代 {i+1}: 移除 '{feature_to_drop}' (VIF={vif_value:.1f}), "
                      f"剩余特征: {X_reduced.shape[1]}, "
                      f"进度: {(i+1)/max_iter*100:.1f}%, "
                      f"已用时间: {timedelta(seconds=int(elapsed))}, "
                      f"预计剩余: {timedelta(seconds=int(remaining))}")
    
    if verbose > 0:
        total_time = time.time() - start_time
        print(f"\nVIF筛选完成，总耗时: {timedelta(seconds=int(total_time))}")
        print(f"移除特征数量: {len(eliminated_features)}/{X.shape[1]} "
              f"({len(eliminated_features)/X.shape[1]*100:.1f}%)")
        print(f"保留特征数量: {X_reduced.shape[1]}")
        
        # 计算最终VIF值
        final_vif = fast_vif_calculation(X_reduced, verbose=0)
        print(f"最终最大VIF值: {final_vif['VIF'].max():.2f}")
    
    return X_reduced, eliminated_features

def quick_vif_visualization(vif_df, thresh=10.0, max_features=30, show=True, save_path=None):
    """
    快速可视化VIF分析结果
    
    参数:
    vif_df - VIF数据框
    thresh - VIF阈值
    max_features - 最多显示的特征数量
    show - 是否显示图表
    save_path - 保存图表的路径，None表示不保存
    """
    plt.figure(figsize=(12, min(len(vif_df), max_features) * 0.3 + 2))
    
    # 选择要显示的数据
    if len(vif_df) > max_features:
        # 确保显示最高VIF值的特征
        display_df = vif_df.head(max_features)
        print(f"注: 仅显示VIF值最高的{max_features}个特征 (共{len(vif_df)}个)")
    else:
        display_df = vif_df
    
    # 按升序排列以便在图表中从上到下显示
    display_df = display_df.sort_values('VIF', ascending=True)
    
    # 创建条形图
    bars = plt.barh(display_df['feature'], display_df['VIF'])
    
    # 为超过阈值的条形图着色
    for i, bar in enumerate(bars):
        if display_df.iloc[i]['VIF'] > thresh:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    # 添加阈值线
    plt.axvline(x=thresh, color='red', linestyle='--', label=f'阈值 (VIF = {thresh})')
    
    # 美化图表
    plt.title('特征VIF分析')
    plt.xlabel('VIF值')
    plt.ylabel('特征')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

def optimized_vif_analysis(df, vif_threshold=10.0, max_iter=None, 
                          use_parallel=True, categorical_features=None,
                          target_col=None, verbose=1):
    """
    优化版VIF分析流程
    
    参数:
    df - 包含所有特征的数据框
    vif_threshold - VIF阈值
    max_iter - 最大迭代次数
    use_parallel - 是否使用并行计算
    categorical_features - 类别特征列表(如二值特征)
    target_col - 目标变量列名，可以为None
    verbose - 详细程度
    
    返回:
    result_dict - 结果字典
    """
    start_time = time.time()
    
    if verbose > 0:
        print("===== 优化版VIF分析 =====")
    
    # 1. 准备数据
    df = df.copy()
    # 确保列名为字符串类型
    df.columns = [str(col) for col in df.columns]
    
    # 分离目标变量
    if target_col is not None and target_col in df.columns:
        y = df[target_col].copy()
        X_all = df.drop(columns=[target_col])
    else:
        y = None
        X_all = df.copy()
    
    # 2. 智能特征过滤
    # 识别不同类型的列
    std_columns = [col for col in X_all.columns if col.endswith('_std')]
    
    # 获取指定的类别特征
    if categorical_features is None:
        categorical_features = []
    
    # 确保类别特征名为字符串
    categorical_features = [str(f) for f in categorical_features] 
    categorical_features = [f for f in categorical_features if f in X_all.columns]
    
    # 排除已经被归一化的特征的原始版本
    original_columns = []
    for col in X_all.columns:
        if col not in std_columns and col not in categorical_features:
            # 检查是否存在该列的标准化版本
            if f"{col}_std" not in std_columns:
                original_columns.append(col)
    
    # 组合所有要分析的列
    vif_columns = std_columns + categorical_features + original_columns
    
    if verbose > 0:
        print("特征组成:")
        print(f"- 标准化特征: {len(std_columns)}列")
        print(f"- 类别特征: {len(categorical_features)}列")
        print(f"- 原始特征: {len(original_columns)}列")
        print(f"- 总计: {len(vif_columns)}列")
    
    # 提取要分析的特征
    X_vif = X_all[vif_columns].copy()
    
    # 3. 设置最大迭代次数
    if max_iter is None:
        # 设置合理的最大迭代次数
        max_iter = min(100, X_vif.shape[1] - 1)
    
    # 4. 执行VIF分析
    if verbose > 0:
        print(f"\n开始VIF分析，数据维度: {X_vif.shape}")
    
    # 先计算一次VIF值
    vif_df = fast_vif_calculation(X_vif, use_parallel=use_parallel, verbose=verbose)
    
    # 迭代消除
    X_reduced, eliminated_features = efficient_vif_elimination(
        X_vif, thresh=vif_threshold, max_iter=max_iter, verbose=verbose
    )
    
    # 5. 可视化结果
    final_vif_df = fast_vif_calculation(X_reduced, verbose=0)
    quick_vif_visualization(final_vif_df, thresh=vif_threshold, 
                           save_path='vif_analysis_result.png')
    
    # 6. 记录总耗时
    total_time = time.time() - start_time
    if verbose > 0:
        print(f"\nVIF分析完成，总耗时: {timedelta(seconds=int(total_time))}")
    
    # 7. 返回结果
    return {
        'X_vif': X_vif,
        'X_reduced': X_reduced,
        'eliminated_features': eliminated_features,
        'final_vif': final_vif_df,
        'runtime_seconds': total_time
    }

# 兼容原始API
def run_vif_analysis(df, vif_threshold=5.0, iterative=True, max_iter=None):
    """
    与原始vif_analyzer兼容的函数
    
    参数:
    df - 数据框 
    vif_threshold - VIF阈值
    iterative - 是否使用迭代方法
    max_iter - 最大迭代次数
    
    返回:
    与原始run_vif_analysis相同的结果结构
    """
    # 确保列名为字符串
    df = df.copy()
    df.columns = [str(col) for col in df.columns]
    
    # 识别不同类型的列
    std_columns = [col for col in df.columns if col.endswith('_std')]
    binary_columns = ['56', '57', '58', '59', '60']
    binary_columns = [col for col in binary_columns if col in df.columns]
    
    original_columns = [col for col in df.columns 
                      if not col == 'label' 
                      and not col.endswith('_yeojohnson') 
                      and not col.endswith('_std')
                      and col not in binary_columns + std_columns]
    
    untransformed_to_keep = [col for col in original_columns 
                           if f"{col}_std" not in std_columns]
    
    vif_columns = std_columns + binary_columns + untransformed_to_keep
    
    print("VIF analysis feature composition:")
    print(f"- standardized features: {len(std_columns)} columns")
    print(f"- binary features: {len(binary_columns)} columns")
    print(f"- untransformed original features: {len(untransformed_to_keep)} columns")
    print(f"- total: {len(vif_columns)} columns")
    
    X_vif = df[vif_columns].copy()
    
    if max_iter is None:
        max_iter = len(vif_columns)
    
    print(f"\nstart VIF analysis, data dimension: {X_vif.shape}...")
    analysis_start = time.time()
    
    if iterative:
        # 使用优化的VIF消除，但保持输出格式与原函数一致
        X_reduced, eliminated_features = efficient_vif_elimination(
            X_vif, thresh=vif_threshold, max_iter=max_iter, verbose=1
        )
        
        print("\ncalculate the VIF value of the final feature set...")
        final_vif_df = fast_vif_calculation(X_reduced, verbose=0)
        
        # 转换为原始格式
        vif_data = final_vif_df.rename(columns={'feature': 'feature', 'VIF': 'VIF'})
        
        # 可视化
        quick_vif_visualization(final_vif_df, thresh=vif_threshold)
        
        analysis_time = time.time() - analysis_start
        print(f"\nVIF analysis completed, total time: {timedelta(seconds=int(analysis_time))}")
        
        return {
            'X_vif': X_vif,
            'X_reduced': X_reduced,
            'eliminated_features': eliminated_features,
            'final_vif': vif_data
        }
    else:
        # 非迭代模式，只计算VIF值
        vif_data, _ = fast_vif_calculation(X_vif), X_vif
        
        # 可视化
        quick_vif_visualization(vif_data, thresh=vif_threshold)
        
        analysis_time = time.time() - analysis_start
        print(f"\nVIF analysis completed, total time: {timedelta(seconds=int(analysis_time))}")
        
        return {
            'X_vif': X_vif,
            'vif_data': vif_data
        } 