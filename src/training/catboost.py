import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

def train_catboost_with_cv(X, y, feature_names, n_splits=5, random_state=42, top_n=None):
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

    cb_params = {
        'iterations': 1000,
        'depth': 3,
        'learning_rate': 0.05,
        'loss_function': 'RMSE',
        'verbose': False,
        'random_seed': random_state,
        'thread_count': -1,
        'allow_writing_files': False
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    all_importances = []
    y_true_all, y_pred_all = [], []
    
    print(f"Training {n_splits}-fold CV (CatBoost with split isolation)...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if top_n is not None and top_n < X_train.shape[1]:
            # Impute for selection (CatBoost can handle NaNs, but selection is easier on dense)
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            
            # Quick CatBoost for selection
            cb_select = CatBoostRegressor(iterations=200, depth=3, verbose=False, random_seed=random_state, allow_writing_files=False)
            cb_select.fit(X_train_imp, y_train)
            indices = np.argsort(cb_select.get_feature_importance())[-top_n:]
            X_train = X_train[:, indices]
            X_test = X_test[:, indices]
        
        cb = CatBoostRegressor(**cb_params)
        cb.fit(X_train, y_train)
        y_pred = cb.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        fold_results.append({'fold': fold_idx + 1, 'r2': r2, 'rmse': rmse, 'mae': mae})
        
        if top_n is not None and top_n < len(feature_names):
            full_imp = np.zeros(len(feature_names))
            full_imp[indices] = cb.get_feature_importance()
            all_importances.append(full_imp)
        else:
            all_importances.append(cb.get_feature_importance())
            
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
        'feature_importances': np.mean(all_importances, axis=0),
        'feature_names': feature_names,
        'y_true': np.array(y_true_all),
        'y_pred': np.array(y_pred_all)
    }
    
    print(f"\nFinal R² = {results['mean_r2']:.3f} ± {results['std_r2']:.3f}")
    return results
