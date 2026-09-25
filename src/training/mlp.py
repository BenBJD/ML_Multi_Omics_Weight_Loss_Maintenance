import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from ..models.mlp import MLP

def train_mlp_with_cv(X, y, feature_names, n_splits=5, hidden_dims=None,
                      dropout=0.3, lr=0.001, epochs=200, batch_size=8, random_state=42):
    """
    Train MLP with k-fold cross-validation.
    """
    if X.shape[1] == 0:
        print("Warning: No features found in X. Skipping training.")
        return {
            'fold_results': pd.DataFrame(columns=['fold', 'r2', 'rmse', 'mae']),
            'mean_r2': np.nan, 'std_r2': np.nan,
            'mean_rmse': np.nan, 'std_rmse': np.nan,
            'mean_mae': np.nan, 'std_mae': np.nan,
            'y_true': np.array([]), 'y_pred': np.array([])
        }

    if hidden_dims is None:
        hidden_dims = [128, 64, 32]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    y_true_all, y_pred_all = [], []
    
    print(f"Training {n_splits}-fold CV (Imputation & Scaling inside folds)...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. Impute within fold
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)
        
        # 2. Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        
        # Create model
        model = MLP(input_dim=X_train_imp.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t).squeeze(-1)  # Squeeze for loss calculation
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_t).squeeze(-1).cpu().numpy()  # Squeeze for evaluation
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        fold_results.append({'fold': fold_idx + 1, 'r2': r2, 'rmse': rmse, 'mae': mae})
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
        'y_true': np.array(y_true_all),
        'y_pred': np.array(y_pred_all)
    }
    
    print(f"\nR² = {results['mean_r2']:.3f} ± {results['std_r2']:.3f}")
    return results
