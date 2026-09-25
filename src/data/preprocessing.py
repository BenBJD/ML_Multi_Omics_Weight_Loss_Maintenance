import pandas as pd
import numpy as np

def normalize_participant_id(value):
    if pd.isna(value):
        return np.nan
    return f"{int(value):03d}"

def normalize_timepoint(value):
    if pd.isna(value):
        return np.nan
    return int(value)

def make_sample_id(participant_id, timepoint):
    if pd.isna(participant_id) or pd.isna(timepoint):
        return np.nan
    return f"ELP_{participant_id}_{int(timepoint)}"

def extract_participant_timepoint(sample_id):
    parts = str(sample_id).split('_')
    if len(parts) >= 3 and parts[0] == 'ELP':
        return parts[1], int(parts[2])
    return np.nan, np.nan

def prepare_data(dataset: pd.DataFrame, dataset_name):
    """
    Standardize dataset preparation for RF and MLP models.
    """
    df = dataset.dropna(subset=['absolute weight change ']).copy()
    feature_cols = [c for c in df.columns if c not in ['participant_id', 'absolute weight change ']]
    
    # Drop known leaked features if present
    # These are features that are mathematically or logically identical to the target
    leak_features = [
        'WM_vel_weight', 'WM_vel_bmi', 'WM_vel_fatpercent',
        'WM_vel_fatmass', 'WM_vel_waistcircumference'
    ]
    feature_cols = [c for c in feature_cols if c not in leak_features]
    
    X = df[feature_cols].values.copy()
    y = df['absolute weight change '].values

    print(f"\n{dataset_name}: {len(X)} samples, {len(feature_cols)} features")
    return X, y, feature_cols

def prepare_lstm_data(dataset: pd.DataFrame, dataset_name):
    """
    Prepare data for LSTM: reshape velocity features into [n, 2, features] sequences.
    """
    df = dataset.dropna(subset=['absolute weight change ']).copy()
    y = df['absolute weight change '].values
    
    # Separate WL and WM velocity features
    wl_features = [c for c in df.columns if c.startswith('WL_vel_')]
    wm_features = [c for c in df.columns if c.startswith('WM_vel_')]
    
    # Extract feature names (without WL_vel_ / WM_vel_ prefix)
    wl_feature_names = [c.replace('WL_vel_', '') for c in wl_features]
    wm_feature_names = [c.replace('WM_vel_', '') for c in wm_features]
    
    # Find common features between WL and WM
    common_features = sorted(set(wl_feature_names) & set(wm_feature_names))
    
    # Drop known leaked features from common features
    leak_features = ['weight', 'bmi', 'fatpercent', 'fatmass', 'waistcircumference']
    common_features = [f for f in common_features if f not in leak_features]
    
    print(f"\n{dataset_name}:")
    print(f"  Samples: {len(df)}")
    print(f"  Common features: {len(common_features)}")
    
    # Create WL and WM matrices with aligned features
    wl_cols = ['WL_vel_' + f for f in common_features]
    wm_cols = ['WM_vel_' + f for f in common_features]
    
    X_wl = df[wl_cols].values.copy()
    X_wm = df[wm_cols].values.copy()
    
    # Stack into [n_samples, 2_timesteps, n_features] tensor
    X_temporal = np.stack([X_wl, X_wm], axis=1)
    
    print(f"  Temporal shape: {X_temporal.shape} (samples, timesteps, features)")
    
    return X_temporal, y, common_features
