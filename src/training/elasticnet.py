import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

def train_elasticnet_with_cv(X, y, feature_names, n_splits=5, random_state=42):
    if X.shape[1] == 0:
        print("Warning: No features found in X. Skipping training.")
        return {
            'fold_results': pd.DataFrame(columns=['fold', 'r2', 'rmse', 'mae']),
            'mean_r2': np.nan, 'std_r2': np.nan,
            'mean_rmse': np.nan, 'std_rmse': np.nan,
            'mean_mae': np.nan, 'std_mae': np.nan,
            'feature_importances': np.array([]),
            'feature_names': feature_names,
            'y_true': np.array([]), 'y_pred': np.array([])
        }

    en_params = {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'max_iter': 1000
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    all_coefs = []
    y_true_all, y_pred_all = [], []
    
    print(f"Training {n_splits}-fold CV (Imputation & Scaling inside folds)...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. Impute within fold
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)
        
        # 2. Scale data for ElasticNet
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        en = ElasticNet(**en_params, random_state=random_state)
        en.fit(X_train_scaled, y_train)
        y_pred = en.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        fold_results.append({'fold': fold_idx + 1, 'r2': r2, 'rmse': rmse, 'mae': mae})
        all_coefs.append(np.abs(en.coef_)) # Use absolute coefficients for importance
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        
        print(f"  Fold {fold_idx+1}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    
    fold_df = pd.DataFrame(fold_results)
    
    results = {
        'fold_results': fold_df,
        'mean_r2': fold_df['r2'].mean(),
        'std_r2': fold_df['r2'].std(),
        'mean_rmse': fold_df['rmse'].mean(),
        'std_rmse': fold_df['rmse'].std(),
        'mean_mae': fold_df['mae'].mean(),
        'std_mae': fold_df['mae'].std(),
        'feature_importances': np.mean(all_coefs, axis=0),
        'feature_names': feature_names,
        'y_true': np.array(y_true_all),
        'y_pred': np.array(y_pred_all)
    }
    
    print(f"\nR² = {results['mean_r2']:.3f} ± {results['std_r2']:.3f}")
    return results
