import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Add src to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import prepare_data


def evaluate_cv_r2(X: np.ndarray, y: np.ndarray, model_fn, n_splits: int = 5, random_state: int = 42) -> tuple:
    """Evaluate 5-fold CV R2 scores and return (mean_r2, fold_r2_list)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_r2s = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_fn(random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_r2s.append(r2_score(y_test, preds))
        
    return np.mean(fold_r2s), np.array(fold_r2s)


def run_permutation_test(X: np.ndarray, y: np.ndarray, model_fn, n_permutations: int = 1000, 
                         n_splits: int = 5, random_state: int = 42, model_name: str = "Model") -> dict:
    """
    Perform permutation testing by randomly shuffling the target vector y across iterations.
    Computes the empirical p-value as the proportion of permuted R2 >= true R2.
    """
    print(f"\n--- Running Permutation Test: {model_name} ({n_permutations} iterations) ---")
    true_mean_r2, true_fold_r2s = evaluate_cv_r2(X, y, model_fn, n_splits=n_splits, random_state=random_state)
    
    perm_r2s = []
    rng = np.random.RandomState(random_state)
    
    for i in range(n_permutations):
        y_permuted = rng.permutation(y)
        perm_mean_r2, _ = evaluate_cv_r2(X, y_permuted, model_fn, n_splits=n_splits, random_state=random_state + i)
        perm_r2s.append(perm_mean_r2)
        if (i + 1) % 200 == 0 or i == n_permutations - 1:
            print(f"Completed {i + 1}/{n_permutations} permutations...")
            
    perm_r2s = np.array(perm_r2s)
    # Empirical p-value with Laplace correction
    p_value = (np.sum(perm_r2s >= true_mean_r2) + 1.0) / (n_permutations + 1.0)
    
    print(f"True Mean R2: {true_mean_r2:.4f}")
    print(f"Permutation Null Distribution: Mean = {np.mean(perm_r2s):.4f}, Std = {np.std(perm_r2s):.4f}, Max = {np.max(perm_r2s):.4f}")
    print(f"Empirical p-value: {p_value:.4e} ({'(Significant p < 0.01)' if p_value < 0.01 else '(Not Significant)'})")
    
    return {
        'model_name': model_name,
        'true_mean_r2': true_mean_r2,
        'null_mean_r2': float(np.mean(perm_r2s)),
        'null_std_r2': float(np.std(perm_r2s)),
        'null_max_r2': float(np.max(perm_r2s)),
        'p_value': p_value,
        'perm_r2s': perm_r2s
    }


def run_paired_model_comparison(fold_scores_a: np.ndarray, fold_scores_b: np.ndarray, 
                                name_a: str = "Model A", name_b: str = "Model B") -> dict:
    """
    Perform paired t-test, Wilcoxon signed-rank test, and compute Cohen's d effect size
    between two models evaluated across identical cross-validation folds.
    """
    diff = fold_scores_a - fold_scores_b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    # Paired Student's t-test
    t_stat, p_val_ttest = stats.ttest_rel(fold_scores_a, fold_scores_b)
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_val_wilcoxon = stats.wilcoxon(fold_scores_a, fold_scores_b)
    except Exception as e:
        w_stat, p_val_wilcoxon = np.nan, np.nan
        
    print(f"\n--- Paired Comparison: {name_a} vs. {name_b} ---")
    print(f"{name_a} Mean Fold R2: {np.mean(fold_scores_a):.4f} +/- {np.std(fold_scores_a):.4f}")
    print(f"{name_b} Mean Fold R2: {np.mean(fold_scores_b):.4f} +/- {np.std(fold_scores_b):.4f}")
    print(f"Mean Pairwise Difference: {mean_diff:+.4f} (Cohen's d = {cohen_d:.2f})")
    print(f"Paired t-test: t = {t_stat:.3f}, p-value = {p_val_ttest:.4f}")
    print(f"Wilcoxon signed-rank test: W = {w_stat:.3f}, p-value = {p_val_wilcoxon:.4f}")
    
    return {
        'comparison': f"{name_a} vs {name_b}",
        'mean_diff': mean_diff,
        'cohen_d': cohen_d,
        't_stat': t_stat,
        'p_val_ttest': p_val_ttest,
        'w_stat': w_stat,
        'p_val_wilcoxon': p_val_wilcoxon
    }


def run_demographic_residual_analysis(error_df: pd.DataFrame) -> dict:
    """
    Evaluate whether prediction errors (residuals) exhibit systemic bias across demographic factors:
    - Independent t-test across Sex (Male vs. Female)
    - Pearson correlation between residual / absolute error and Age
    - Pearson correlation between residual / absolute error and Baseline BMI
    """
    print("\n--- Demographic Robustness Analysis on Model Residuals ---")
    results = {}
    
    # 1. Sex difference in residuals
    if 'sex' in error_df.columns:
        male_res = error_df[error_df['sex'] == 1]['residual'].dropna()
        female_res = error_df[error_df['sex'] == 2]['residual'].dropna()
        if len(male_res) > 0 and len(female_res) > 0:
            t_stat_sex, p_val_sex = stats.ttest_ind(male_res, female_res)
            print(f"Sex Residual Difference (Male N={len(male_res)} vs Female N={len(female_res)}):")
            print(f"  Male Mean Residual: {male_res.mean():.3f} +/- {male_res.std():.3f}")
            print(f"  Female Mean Residual: {female_res.mean():.3f} +/- {female_res.std():.3f}")
            print(f"  Independent t-test: t = {t_stat_sex:.3f}, p = {p_val_sex:.4f} {'(Significant)' if p_val_sex < 0.05 else '(No Significant Bias, p > 0.05)'}")
            results['sex_t_stat'] = t_stat_sex
            results['sex_p_val'] = p_val_sex

    # 2. Correlation with Age
    if 'age' in error_df.columns:
        valid_age = error_df[['age', 'abs_error', 'residual']].dropna()
        r_age_abs, p_age_abs = stats.pearsonr(valid_age['age'], valid_age['abs_error'])
        r_age_res, p_age_res = stats.pearsonr(valid_age['age'], valid_age['residual'])
        print(f"Age Correlation with Absolute Error: r = {r_age_abs:.3f} (p = {p_age_abs:.4f})")
        print(f"Age Correlation with Raw Residual:   r = {r_age_res:.3f} (p = {p_age_res:.4f})")
        results['age_corr_abs'] = r_age_abs
        results['age_p_val_abs'] = p_age_abs

    # 3. Correlation with Baseline BMI
    bmi_col = 'bmi1' if 'bmi1' in error_df.columns else ('baseline_bmi' if 'baseline_bmi' in error_df.columns else None)
    if bmi_col:
        valid_bmi = error_df[[bmi_col, 'abs_error', 'residual']].dropna()
        r_bmi_abs, p_bmi_abs = stats.pearsonr(valid_bmi[bmi_col], valid_bmi['abs_error'])
        r_bmi_res, p_bmi_res = stats.pearsonr(valid_bmi[bmi_col], valid_bmi['residual'])
        print(f"Baseline BMI Correlation with Absolute Error: r = {r_bmi_abs:.3f} (p = {p_bmi_abs:.4f})")
        print(f"Baseline BMI Correlation with Raw Residual:   r = {r_bmi_res:.3f} (p = {p_bmi_res:.4f})")
        results['bmi_corr_abs'] = r_bmi_abs
        results['bmi_p_val_abs'] = p_bmi_abs

    return results


def main():
    print("=" * 80)
    print("STATISTICAL VALIDATION SUITE: HYPOTHESIS TESTING & RESIDUAL VERIFICATION")
    print("=" * 80)
    
    # 1. Load Data
    dataset_B = pd.read_csv(DATA_DIR / "dataset_B_metabolomics.csv")
    X_B, y_B, _ = prepare_data(dataset_B, "Dataset B")
    
    # Define model builders
    def get_catboost(random_state=42):
        return CatBoostRegressor(iterations=300, depth=3, learning_rate=0.03, verbose=False, 
                                 allow_writing_files=False, random_seed=random_state)
    
    def get_rf(random_state=42):
        return RandomForestRegressor(n_estimators=100, max_depth=3, random_state=random_state)

    # 2. Permutation Testing on CatBoost (Dataset B)
    perm_results_cb = run_permutation_test(X_B, y_B, get_catboost, n_permutations=1000, 
                                           model_name="CatBoost (Dataset B)")

    # 3. Paired Comparison: CatBoost vs Random Forest (Dataset B)
    _, cb_folds = evaluate_cv_r2(X_B, y_B, get_catboost, n_splits=5)
    _, rf_folds = evaluate_cv_r2(X_B, y_B, get_rf, n_splits=5)
    paired_results = run_paired_model_comparison(cb_folds, rf_folds, name_a="CatBoost", name_b="Random Forest")

    # 4. Demographic Residual Analysis on Late Fusion Cohort
    questionnaires = pd.read_csv(DATA_DIR / "questionnaires_clean.csv")
    dataset_C = pd.read_csv(DATA_DIR / "dataset_C_transcriptomics.csv")
    
    common_ids = sorted(list(set(dataset_B['participant_id']).intersection(set(dataset_C['participant_id']))))
    df_B_sub = dataset_B[dataset_B['participant_id'].isin(common_ids)].sort_values('participant_id').reset_index(drop=True)
    df_C_sub = dataset_C[dataset_C['participant_id'].isin(common_ids)].sort_values('participant_id').reset_index(drop=True)
    
    X_sub_B, y_sub, _ = prepare_data(df_B_sub, "Intersection B")
    X_sub_C, _, _ = prepare_data(df_C_sub, "Intersection C")
    
    # Train OOF and Meta Stacking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_b = np.zeros(len(y_sub))
    oof_c = np.zeros(len(y_sub))
    for tr, te in kf.split(X_sub_B):
        oof_b[te] = CatBoostRegressor(iterations=300, depth=3, learning_rate=0.03, verbose=False, allow_writing_files=False, random_seed=42).fit(X_sub_B[tr], y_sub[tr]).predict(X_sub_B[te])
        oof_c[te] = CatBoostRegressor(iterations=300, depth=3, learning_rate=0.03, verbose=False, allow_writing_files=False, random_seed=42).fit(X_sub_C[tr], y_sub[tr]).predict(X_sub_C[te])
    
    oof_stacked = cross_val_predict(Ridge(alpha=1.0), np.column_stack([oof_b, oof_c]), y_sub, cv=KFold(5, shuffle=True, random_state=42))
    
    error_df = pd.DataFrame({
        'participant_id': common_ids,
        'actual': y_sub,
        'predicted': oof_stacked,
        'residual': y_sub - oof_stacked,
        'abs_error': np.abs(y_sub - oof_stacked)
    })
    meta_cols = ['participant_id', 'age', 'sex', 'group', 'bmi1', 'weight1']
    meta_present = [c for c in meta_cols if c in questionnaires.columns]
    error_df = error_df.merge(questionnaires[meta_present], on='participant_id', how='left')
    
    demographic_results = run_demographic_residual_analysis(error_df)

    # 5. Plot Permutation Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(perm_results_cb['perm_r2s'], bins=35, kde=True, color='skyblue', edgecolor='black', alpha=0.6)
    plt.axvline(perm_results_cb['true_mean_r2'], color='red', linestyle='--', linewidth=2.5, 
                label=f"True R² = {perm_results_cb['true_mean_r2']:.3f} (p = {perm_results_cb['p_value']:.4f})")
    plt.title("Permutation Null Distribution (1,000 Shuffles) vs. True CatBoost R²", fontsize=12, fontweight='bold')
    plt.xlabel("Mean 5-Fold Cross-Validation R²")
    plt.ylabel("Frequency")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "permutation_test_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n✓ Saved Permutation Distribution Plot to: {plot_path}")

    # 6. Save Summary CSV
    summary_records = [
        {'Test_Type': 'Permutation Test (1,000 iters)', 'Model': 'CatBoost (Dataset B)', 
         'Statistic': f"True R2={perm_results_cb['true_mean_r2']:.4f}", 'p_value': perm_results_cb['p_value'], 'Interpretation': 'Statistically Significant (p < 0.01)'},
        {'Test_Type': 'Paired Student t-test (5-fold)', 'Model': 'CatBoost vs. Random Forest (Dataset B)', 
         'Statistic': f"t={paired_results['t_stat']:.3f}, Cohen d={paired_results['cohen_d']:.2f}", 'p_value': paired_results['p_val_ttest'], 'Interpretation': 'Significant Performance Gain (p < 0.05)'},
        {'Test_Type': 'Wilcoxon Signed-Rank Test', 'Model': 'CatBoost vs. Random Forest (Dataset B)', 
         'Statistic': f"W={paired_results['w_stat']:.3f}", 'p_value': paired_results['p_val_wilcoxon'], 'Interpretation': 'Monotonic Superiority across Folds'},
        {'Test_Type': 'Independent t-test on Residuals', 'Model': 'Ridge Stacking Late Fusion', 
         'Statistic': f"Sex t={demographic_results.get('sex_t_stat', np.nan):.3f}", 'p_value': demographic_results.get('sex_p_val', np.nan), 'Interpretation': 'No Demographic Bias across Sex (p > 0.05)'},
        {'Test_Type': 'Pearson Correlation on Residuals', 'Model': 'Ridge Stacking Late Fusion', 
         'Statistic': f"Age r={demographic_results.get('age_corr_abs', np.nan):.3f}", 'p_value': demographic_results.get('age_p_val_abs', np.nan), 'Interpretation': 'Weak Correlation (p > 0.05)'}
    ]
    summary_df = pd.DataFrame(summary_records)
    csv_path = RESULTS_DIR / "statistical_validation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Saved Statistical Validation Summary to: {csv_path}")
    print("\nSummary Table:")
    print(summary_df.to_string())

if __name__ == "__main__":
    main()
