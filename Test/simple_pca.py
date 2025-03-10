import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
import os
import matplotlib


def apply_pca_to_selected_features(
    df, 
    selected_features=None, 
    selected_features_file='selected_features.txt',
    target_col='label', 
    n_components=None, 
    variance_threshold=0.95,
    plot=True, 
    save_results=True
):
    """
    在选定的特征上应用PCA降维
    
    参数:
    df - 包含特征和目标变量的数据框
    selected_features - 选定特征列表，如果为None则从文件加载
    selected_features_file - 选定特征文件路径，默认为'selected_features.txt'
    target_col - 目标变量列名，默认为'label'
    n_components - PCA组件数量，None表示自动根据方差阈值确定
    variance_threshold - 方差阈值(0-1之间)，默认为0.95(保留95%的方差)
    plot - 是否绘制PCA相关图表，默认为True
    save_results - 是否保存PCA结果，默认为True
    
    返回:
    pca_df - PCA转换后的数据框
    pca - 训练好的PCA模型
    explained_variance_df - 解释方差相关信息
    """
    # 设置中文字体
    

    
    print("开始PCA降维过程...")
    
    # 创建进度条
    progress = tqdm(total=4, desc="PCA处理进度")
    
    # 如果没有提供选定特征，从文件加载
    if selected_features is None:
        if os.path.exists(selected_features_file):
            with open(selected_features_file, 'r') as f:
                selected_features = [line.strip() for line in f.readlines()]
            print(f"从文件加载了{len(selected_features)}个选定特征")
        else:
            raise FileNotFoundError(f"未找到选定特征文件: {selected_features_file}")
    
    # 检查目标列是否存在
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 在数据框中不存在")
    
    # 检查选定特征是否都在数据框中
    missing_features = [feat for feat in selected_features if feat not in df.columns]
    if missing_features:
        print(f"警告: 以下选定特征在数据框中不存在: {missing_features}")
        selected_features = [feat for feat in selected_features if feat in df.columns]
        print(f"继续使用剩余的{len(selected_features)}个特征")
    
    # 提取标签和选定特征
    y = df[target_col].copy()
    X = df[selected_features].copy()
    
    progress.update(1)
    progress.set_description("数据准备完成，标准化特征")
    
    # 标准化特征
    print("标准化特征...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    progress.update(1)
    progress.set_description("特征标准化完成，拟合PCA")
    
    # 拟合PCA以确定组件数量
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # 计算累积解释方差
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # 确定要保留的组件数量
    if n_components is None:
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # 确保至少有2个组件(用于可视化)，除非原始特征数量更少
    n_components = max(min(2, len(selected_features)), n_components)
    
    print(f"原始维度: {X.shape[1]}")
    print(f"PCA降维后的维度: {n_components}")
    print(f"维度降低: {X.shape[1] - n_components} ({(1 - n_components/X.shape[1])*100:.1f}%)")
    print(f"保留的方差: {cumulative_variance[n_components-1]*100:.2f}%")
    
    # 应用PCA降维
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    progress.update(1)
    progress.set_description("PCA拟合完成，准备可视化和结果")
    
    # 创建PCA数据框
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=X_pca, columns=pca_columns)
    
    # 添加目标变量
    pca_df[target_col] = y.values
    
    # 可视化解释方差
    if plot:
        print("生成PCA可视化...")
        plt.figure(figsize=(12, 8))
        
        # 累积解释方差曲线
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.axhline(y=variance_threshold, color='r', linestyle='--', 
                    label=f'{variance_threshold*100}% 累积方差')
        plt.axvline(x=n_components, color='g', linestyle='--', 
                    label=f'选定的组件数量 ({n_components})')
        plt.xlabel('组件数量')
        plt.ylabel('累积解释方差')
        plt.title('PCA累积解释方差')
        plt.legend()
        plt.grid(True)
        
        # 各组件的解释方差
        plt.subplot(2, 1, 2)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance)
        plt.axvline(x=n_components + 0.5, color='r', linestyle='--', 
                    label=f'选定的组件数量 ({n_components})')
        plt.xlabel('组件数量')
        plt.ylabel('解释方差')
        plt.title('各主成分的解释方差')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_results:
            plt.savefig('pca_variance_explained.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 如果主成分数量大于等于2，可视化前两个主成分
        if n_components >= 2:
            plt.figure(figsize=(10, 8))
            
            if target_col in pca_df.columns:
                # 如果是分类问题，使用不同颜色
                if len(pca_df[target_col].unique()) <= 10:  # 限制分类不超过10个
                    scatter = plt.scatter(
                        pca_df['PC1'], 
                        pca_df['PC2'], 
                        c=pca_df[target_col], 
                        cmap='viridis', 
                        alpha=0.6
                    )
                    plt.colorbar(scatter, label=target_col)
                else:
                    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
            else:
                plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
                
            plt.xlabel('第一主成分 (PC1)')
            plt.ylabel('第二主成分 (PC2)')
            plt.title('PCA: 前两个主成分散点图')
            plt.grid(True)
            if save_results:
                plt.savefig('pca_scatter.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        # 可视化主成分与原始特征的关系(载荷图)
        plt.figure(figsize=(12, 10))
        
        # 获取主成分系数(载荷)
        n_plot = min(n_components, 5)  # 最多显示5个主成分
        
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=X.columns
        )
        
        plt.subplot(2, 1, 1)
        sns.heatmap(
            loadings.iloc[:, :n_plot],  # 最多显示n_plot个主成分
            annot=False,
            cmap='coolwarm',
            center=0
        )
        plt.title('主成分载荷热图')
        plt.ylabel('原始特征')
        plt.xlabel('主成分')
        
        # 为了更清晰，也显示前3个主成分的最重要特征
        plt.subplot(2, 1, 2)
        
        # 每个主成分的最重要特征
        top_k = min(5, len(X.columns))  # 每个主成分最多显示5个特征
        
        # 绘制前3个主成分的主要载荷
        for i in range(min(3, n_components)):
            pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
            top_features = pc_loadings.index[:top_k].tolist()
            top_values = loadings.loc[top_features, f'PC{i+1}'].values
            
            plt.barh(
                [f"{feat}_PC{i+1}" for feat in top_features],
                top_values,
                label=f'PC{i+1}'
            )
        
        plt.title('主要主成分的重要特征载荷')
        plt.xlabel('载荷系数')
        plt.legend()
        plt.tight_layout()
        if save_results:
            plt.savefig('pca_loadings.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 创建解释方差数据框
    explained_variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Explained_Variance_Ratio': explained_variance,
        'Cumulative_Variance': cumulative_variance
    })
    
    # 保存PCA结果
    if save_results:
        print("保存PCA结果...")
        joblib.dump(pca, 'pca_model.pkl')
        joblib.dump(scaler, 'pca_scaler_model.pkl')
        pca_df.to_csv('pca_transformed_data.csv', index=False)
        explained_variance_df.to_csv('pca_explained_variance.csv', index=False)
        
        # 保存特征映射信息
        feature_mapping = pd.DataFrame({
            'Original_Feature': selected_features,
            'Selected_For_PCA': ['Yes'] * len(selected_features)
        })
        feature_mapping.to_csv('pca_feature_mapping.csv', index=False)
        
        print("PCA结果已保存!")
    
    progress.update(1)
    progress.set_description("PCA过程完成!")
    progress.close()
    
    return pca_df, pca, explained_variance_df

