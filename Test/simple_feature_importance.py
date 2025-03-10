import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm  
import matplotlib


def calculate_simple_feature_importance(
    df, 
    target_col='label', 
    importance_type='gain',
    categorical_features=None, 
    threshold=0.95,
    plot=True,
    save_results=True
):
    """
    计算数据框中特征的重要性并返回排序结果
    
    参数:
    df - 包含特征和目标变量的数据框
    target_col - 目标变量列名，默认为'label'
    importance_type - 特征重要性类型，可选'gain', 'split', 'weight'等，默认为'gain'
    categorical_features - 类别特征列表（如二值特征），默认为None
    threshold - 用于选择特征的累积重要性阈值（0-1之间），默认为0.95
    plot - 是否绘制特征重要性图，默认为True
    save_results - 是否保存结果为CSV文件，默认为True
    
    返回:
    feature_importance_df - 特征重要性数据框，按重要性排序
    selected_features - 选中的特征列表
    """
    
    print("开始计算特征重要性...")
    
    # 创建进度跟踪对象
    progress = tqdm(total=5, desc="特征重要性计算进度")
    
    # 检查目标列是否存在
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 在数据框中不存在")
    
    # 准备数据
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 确保列名为字符串
    X.columns = [str(col) for col in X.columns]
    
    progress.update(1)  
    progress.set_description("数据准备完成，进行数据集分割")
    
    # 分割训练集和测试集
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 处理类别特征
    if categorical_features is None:
        categorical_features = []
    
    # 确保类别特征为字符串
    categorical_features = [str(feat) for feat in categorical_features]
    
    # 只保留存在于X_train中的类别特征
    categorical_features = [feat for feat in categorical_features if feat in X_train.columns]
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=categorical_features
    )
    
    valid_data = lgb.Dataset(
        X_valid, 
        label=y_valid, 
        reference=train_data,
        categorical_feature=categorical_features
    )
    
    progress.update(1)  
    progress.set_description("数据集创建完成，配置LightGBM参数")
    
    # 目标变量是分类还是回归
    if len(np.unique(y)) <= 10:  # 假设小于等于10个不同值为分类问题
        if len(np.unique(y)) == 2:
            objective = 'binary'
            metric = 'auc'
            params_extra = {}
        else:
            objective = 'multiclass'
            metric = 'multi_logloss'
            params_extra = {'num_class': len(np.unique(y))}
    else:
        objective = 'regression'
        metric = 'rmse'
        params_extra = {}
    
    # 设置参数
    params = {
        'objective': objective,
        'metric': metric,
        'verbosity': 1,
        'seed': 42,
        **params_extra
    }
    
    progress.update(1)  
    progress.set_description("开始训练LightGBM模型")
    
    # 创建回调函数以在每次迭代时更新进度条
    class ProgressCallback:
        def __init__(self, total_iterations=100):
            self.pbar = tqdm(total=total_iterations, desc="模型训练进度", leave=False)
            self.latest_iteration = 0
            
        def __call__(self, env):
            iteration = env.iteration
            
            self.pbar.update(iteration - self.latest_iteration)
            self.latest_iteration = iteration
            
            if env.iteration == env.end_iteration - 1:
                self.pbar.close()
    
    # 训练模型
    print("计算特征重要性中...")
    callback = ProgressCallback(total_iterations=100)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100, 
        valid_sets=[valid_data],
        callbacks=[callback]
    )
    
    progress.update(1) 
    progress.set_description("模型训练完成，计算特征重要性")
    
    # 获取特征重要性
    importance = model.feature_importance(importance_type=importance_type)
    feature_names = model.feature_name()
    
    # 创建特征重要性数据框
    feature_importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importance
    })
    
    # 计算相对重要性百分比
    feature_importance_df['重要性百分比'] = (
        feature_importance_df['重要性'] / feature_importance_df['重要性'].sum() * 100
    )
    
    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values(
        by='重要性', ascending=False
    ).reset_index(drop=True)
    
    # 添加累计重要性百分比
    feature_importance_df['累计重要性'] = feature_importance_df['重要性百分比'].cumsum()
    
    # 选择重要特征（基于累计阈值）
    threshold_percent = threshold * 100
    selected_features = feature_importance_df[
        feature_importance_df['累计重要性'] <= threshold_percent
    ]['特征'].tolist()
    
    # 确保至少选择一些特征
    if len(selected_features) < 5:
        selected_features = feature_importance_df['特征'].head(
            min(5, len(feature_names))
        ).tolist()
    
    progress.update(1)  # 完成进度条
    progress.set_description("特征重要性计算完成！")
    progress.close()
    
    print(f"原始特征数量: {len(feature_names)}")
    print(f"选中的特征数量: {len(selected_features)}")
    print(f"选中特征解释了总方差的: {feature_importance_df.loc[len(selected_features)-1, '累计重要性']:.2f}%")
    
    # 绘制特征重要性图
    if plot:
        print("生成特征重要性可视化...")
        plt.figure(figsize=(12, 8))
        
        # 前20个特征的条形图
        plt.subplot(2, 1, 1)
        top_n = min(20, len(feature_importance_df))
        sns.barplot(
            x='重要性', 
            y='特征', 
            data=feature_importance_df.head(top_n)
        )
        plt.title(f'LightGBM特征重要性 (前{top_n}个特征)')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        
        # 累积重要性曲线
        plt.subplot(2, 1, 2)
        plt.plot(
            range(1, len(feature_importance_df) + 1), 
            feature_importance_df['累计重要性'], 
            marker='o'
        )
        plt.axhline(y=threshold_percent, color='r', linestyle='--', 
                    label=f'{threshold_percent}% 累计重要性')
        plt.axvline(x=len(selected_features), color='g', linestyle='--', 
                    label=f'所选特征数量 ({len(selected_features)})')
        plt.xlabel('特征数量')
        plt.ylabel('累计重要性 (%)')
        plt.title('累计重要性曲线')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_results:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 保存结果
    if save_results:
        print("保存结果到文件...")
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        joblib.dump(model, 'lightgbm_model.pkl')
        
        # 保存选中的特征
        with open('selected_features.txt', 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        print("结果保存完成！")
    
    return feature_importance_df, selected_features
