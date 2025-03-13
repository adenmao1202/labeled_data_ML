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
    Calculate the feature importance from the dataframe and return sorted results.
    
    Parameters:
    df - DataFrame containing features and the target variable.
    target_col - Name of the target variable column, default is 'label'.
    importance_type - Type of feature importance ('gain', 'split', 'weight', etc.), default is 'gain'.
    categorical_features - List of categorical features (e.g., binary features), default is None.
    threshold - Cumulative importance threshold (between 0 and 1) to select features, default is 0.95.
    plot - Whether to plot the feature importance charts, default is True.
    save_results - Whether to save results as CSV files, default is True.
    
    Returns:
    feature_importance_df - DataFrame of feature importance sorted by importance.
    selected_features - List of selected features.
    """
    
    print("Starting feature importance calculation...")
    
    # Initialize progress tracking
    progress = tqdm(total=5, desc="Feature importance calculation progress")
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist in the DataFrame")
    
    # Prepare the data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Ensure column names are strings
    X.columns = [str(col) for col in X.columns]
    
    progress.update(1)  
    progress.set_description("Data preparation completed, splitting dataset")
    
    # Split the dataset into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Process categorical features
    if categorical_features is None:
        categorical_features = []
    
    # Ensure categorical features are strings
    categorical_features = [str(feat) for feat in categorical_features]
    
    # Keep only categorical features that exist in X_train
    categorical_features = [feat for feat in categorical_features if feat in X_train.columns]
    
    # Create LightGBM datasets
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
    progress.set_description("Dataset creation completed, configuring LightGBM parameters")
    
    # Determine if the task is classification or regression
    if len(np.unique(y)) <= 10:  # Assume <=10 unique values indicate a classification task
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
    
    # Set parameters
    params = {
        'objective': objective,
        'metric': metric,
        'verbosity': 1,
        'seed': 42,
        **params_extra
    }
    
    progress.update(1)  
    progress.set_description("Starting LightGBM model training")
    
    # Create a callback to update progress during each iteration
    class ProgressCallback:
        def __init__(self, total_iterations=100):
            self.pbar = tqdm(total=total_iterations, desc="Model training progress", leave=False)
            self.latest_iteration = 0
            
        def __call__(self, env):
            iteration = env.iteration
            
            self.pbar.update(iteration - self.latest_iteration)
            self.latest_iteration = iteration
            
            if env.iteration == env.end_iteration - 1:
                self.pbar.close()
    
    # Train the model
    print("Calculating feature importance...")
    callback = ProgressCallback(total_iterations=100)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100, 
        valid_sets=[valid_data],
        callbacks=[callback]
    )
    
    progress.update(1) 
    progress.set_description("Model training completed, computing feature importance")
    
    # Get feature importance from the model
    importance = model.feature_importance(importance_type=importance_type)
    feature_names = model.feature_name()
    
    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Calculate relative importance percentage
    feature_importance_df['importance_percentage'] = (
        feature_importance_df['importance'] / feature_importance_df['importance'].sum() * 100
    )
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(
        by='importance', ascending=False
    ).reset_index(drop=True)
    
    # Add cumulative importance percentage
    feature_importance_df['cumulative_importance'] = feature_importance_df['importance_percentage'].cumsum()
    
    # Select important features based on cumulative threshold
    threshold_percent = threshold * 100
    selected_features = feature_importance_df[
        feature_importance_df['cumulative_importance'] <= threshold_percent
    ]['feature'].tolist()
    
    # Ensure at least some features are selected
    if len(selected_features) < 5:
        selected_features = feature_importance_df['feature'].head(
            min(5, len(feature_names))
        ).tolist()
    
    progress.update(1)  # Complete progress bar
    progress.set_description("Feature importance calculation complete!")
    progress.close()
    
    print(f"Total number of original features: {len(feature_names)}")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Selected features explain {feature_importance_df.loc[len(selected_features)-1, 'cumulative_importance']:.2f}% of total importance")
    
    # Plot feature importance visualization
    if plot:
        print("Generating feature importance visualization...")
        plt.figure(figsize=(12, 8))
        
        # Bar chart for the top 20 features
        plt.subplot(2, 1, 1)
        top_n = min(20, len(feature_importance_df))
        sns.barplot(
            x='importance', 
            y='feature', 
            data=feature_importance_df.head(top_n)
        )
        plt.title(f'LightGBM Feature Importance (Top {top_n} Features)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        # Cumulative importance curve
        plt.subplot(2, 1, 2)
        plt.plot(
            range(1, len(feature_importance_df) + 1), 
            feature_importance_df['cumulative_importance'], 
            marker='o'
        )
        plt.axhline(y=threshold_percent, color='r', linestyle='--', 
                    label=f'{threshold_percent}% Cumulative Importance')
        plt.axvline(x=len(selected_features), color='g', linestyle='--', 
                    label=f'Number of Selected Features ({len(selected_features)})')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance (%)')
        plt.title('Cumulative Importance Curve')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_results:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save results to files
    if save_results:
        print("Saving results to files...")
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        
        
        # Save the selected features
        with open('selected_features.txt', 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        print("Results saved successfully!")
    
    return feature_importance_df, selected_features
