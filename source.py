import pandas as pd
import numpy as np
import joblib
import json
import os
import shutil
import hashlib
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearnex import patch_sklearn

# Optimize scikit-learn operations
patch_sklearn()

# 1) Constants and contracts
RANDOM_SEED = 42
TARGET_COL = 'true_label'
FEATURE_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'LoanDuration', 'DependentCount']
ID_COLS = []
SENSITIVE_ATTRIBUTE = 'credit_score_band'

SCHEMA = {
    'Age': {'dtype': 'int', 'range': (20, 70)},
    'Income': {'dtype': 'float', 'range': (0, 200000)},
    'LoanAmount': {'dtype': 'float', 'range': (0, 100000)},
    'CreditScore': {'dtype': 'int', 'range': (300, 850)},
    'LoanDuration': {'dtype': 'int', 'range': (12, 60)},
    'DependentCount': {'dtype': 'int', 'range': (0, 5)},
    'credit_score_band': {'dtype': 'category', 'options': ['Poor', 'Fair', 'Good', 'Excellent']},
    'true_label': {'dtype': 'int', 'range': (0, 1)}
}

CRITICAL_THRESHOLDS = {
    'min_auc': 0.70,
    'max_degradation_auc_percent': 15.0,
    'max_brier_score': 0.25,
    'max_subgroup_delta_auc': 0.10
}

WARN_THRESHOLDS = {
    'min_auc': 0.75,
    'max_degradation_auc_percent': 10.0,
    'max_brier_score': 0.20,
    'max_subgroup_delta_auc': 0.05
}

# 2) Pure business-logic functions

def set_global_seed(seed: int) -> None:
    """Sets the seed for reproducibility across libraries."""
    np.random.seed(seed)

def initialize_synthetic_environment(out_dir: str, seed: int = RANDOM_SEED):
    """Generates synthetic data and a dummy model to simulate the environment."""
    set_global_seed(seed)
    num_samples = 1000
    data = {
        'Age': np.random.randint(20, 70, num_samples),
        'Income': np.random.normal(50000, 15000, num_samples),
        'LoanAmount': np.random.normal(15000, 5000, num_samples),
        'CreditScore': np.random.randint(300, 850, num_samples),
        'LoanDuration': np.random.randint(12, 60, num_samples),
        'DependentCount': np.random.randint(0, 5, num_samples)
    }
    df_train = pd.DataFrame(data)

    bins = [0, 580, 670, 740, 850]
    labels = ['Poor', 'Fair', 'Good', 'Excellent']
    df_train[SENSITIVE_ATTRIBUTE] = pd.cut(df_train['CreditScore'], bins=bins, labels=labels, right=False)
    
    df_train[TARGET_COL] = ((df_train['CreditScore'] < 600) & (df_train['LoanAmount'] > 20000) | (df_train['Income'] < 30000)).astype(int)

    X = df_train[FEATURE_COLS]
    y = df_train[TARGET_COL]

    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    X_processed = scaler.fit_transform(imputer.fit_transform(X))

    model = LogisticRegression(random_state=seed, solver='liblinear')
    model.fit(X_processed, y)

    model_path = os.path.join(out_dir, 'sample_model.pkl')
    data_path = os.path.join(out_dir, 'sample_credit_test.csv')
    
    joblib.dump(model, model_path)
    df_train.to_csv(data_path, index=False)
    
    return data_path, model_path

def load_assets(data_path: str, model_path: str, features: list, target: str, sensitive_col: str):
    """Loads dataset and model, performs demo-level preprocessing."""
    df_test = pd.read_csv(data_path)
    model = joblib.load(model_path)

    X_test_raw = df_test[features]
    
    # Note: notebook refits on test for demo purposes
    local_imputer = SimpleImputer(strategy='mean')
    X_imputed = local_imputer.fit_transform(X_test_raw)

    local_scaler = StandardScaler()
    X_scaled = local_scaler.fit_transform(X_imputed)

    X_test_aligned = pd.DataFrame(X_scaled, columns=features)
    y_test = df_test[target]
    sensitive_attr = df_test[sensitive_col]

    return X_test_aligned, y_test, sensitive_attr, model

def evaluate_model_performance(model, X, y, sensitive_attr=None, scenario_name="Baseline"):
    """Evaluates model performance metrics."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred_class = model.predict(X)

    metrics = {
        'scenario': scenario_name,
        'num_samples': len(y),
        'accuracy': accuracy_score(y, y_pred_class),
        'auc': roc_auc_score(y, y_pred_proba),
        'precision': precision_score(y, y_pred_class, zero_division=0),
        'recall': recall_score(y, y_pred_class, zero_division=0),
        'brier_score': brier_score_loss(y, y_pred_proba)
    }

    if sensitive_attr is not None and not sensitive_attr.empty:
        overall_auc = metrics['auc']
        subgroup_aucs = {}
        
        if pd.api.types.is_categorical_dtype(sensitive_attr.dtype):
            unique_groups = sensitive_attr.cat.categories.tolist()
        else:
            unique_groups = sorted(sensitive_attr.unique().tolist())

        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            if len(y[group_mask]) > 1 and len(np.unique(y[group_mask])) > 1:
                subgroup_auc = roc_auc_score(y[group_mask], y_pred_proba[group_mask])
                subgroup_aucs[f'auc_{group}'] = subgroup_auc
            else:
                subgroup_aucs[f'auc_{group}'] = np.nan

        subgroup_deltas = {f'delta_auc_{group}': abs(subgroup_auc - overall_auc) 
                          for group, subgroup_auc in subgroup_aucs.items() if not np.isnan(subgroup_auc)}

        metrics.update(subgroup_aucs)
        metrics['max_subgroup_delta_auc'] = max(subgroup_deltas.values()) if subgroup_deltas else np.nan
        metrics['subgroup_deltas'] = subgroup_deltas

    return metrics

def apply_gaussian_noise(df, features_to_noise, noise_std_multiplier, random_state=RANDOM_SEED):
    df_stressed = df.copy()
    for col in features_to_noise:
        if col in df_stressed.columns and pd.api.types.is_numeric_dtype(df_stressed[col]):
            std_dev = df_stressed[col].std()
            noise_level = std_dev * noise_std_multiplier
            noise = np.random.RandomState(random_state).normal(loc=0, scale=noise_level, size=len(df_stressed))
            df_stressed[col] = df_stressed[col] + noise
    return df_stressed

def apply_feature_scaling_shift(df, features_to_shift, shift_factor, random_state=RANDOM_SEED):
    df_stressed = df.copy()
    for col in features_to_shift:
        if col in df_stressed.columns and pd.api.types.is_numeric_dtype(df_stressed[col]):
            df_stressed[col] = df_stressed[col] * shift_factor
    return df_stressed

def apply_missingness_spike(df, features_to_spike, missing_rate, random_state=RANDOM_SEED):
    df_stressed = df.copy()
    for col in features_to_spike:
        if col in df_stressed.columns:
            mask = np.random.RandomState(random_state).rand(len(df_stressed)) < missing_rate
            df_stressed.loc[mask, col] = np.nan
    return df_stressed

def preprocess_stressed_data(X_stressed, X_baseline_cols_ref):
    X_processed = X_stressed.copy()
    local_imputer = SimpleImputer(strategy='mean')
    for col in X_processed.columns:
        if X_processed[col].isnull().any() and pd.api.types.is_numeric_dtype(X_processed[col]):
            X_processed[col] = local_imputer.fit_transform(X_processed[[col]])

    local_scaler = StandardScaler()
    X_scaled = local_scaler.fit_transform(X_processed)
    X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)

    missing_cols = set(X_baseline_cols_ref.columns) - set(X_processed.columns)
    for c in missing_cols:
        X_processed[c] = 0

    extra_cols = set(X_processed.columns) - set(X_baseline_cols_ref.columns)
    if extra_cols:
        X_processed = X_processed.drop(columns=list(extra_cols))

    X_processed = X_processed[X_baseline_cols_ref.columns]
    return X_processed

def run_and_evaluate_scenario(model, X_original, y_original, sensitive_attr_original,
                              scenario_name, transformation_func, transform_params,
                              X_baseline_cols, baseline_metrics):
    
    X_stressed_df = transformation_func(X_original.copy(), 
                                        transform_params['features'], 
                                        **{k:v for k,v in transform_params.items() if k not in ['features']})

    X_stressed_processed = preprocess_stressed_data(X_stressed_df, X_baseline_cols)
    stressed_metrics = evaluate_model_performance(model, X_stressed_processed, y_original, sensitive_attr_original, scenario_name)

    for metric_key in ['auc', 'accuracy', 'precision', 'recall', 'brier_score']:
        if metric_key in baseline_metrics and metric_key in stressed_metrics and not pd.isna(stressed_metrics[metric_key]):
            baseline_val = baseline_metrics[metric_key]
            stressed_val = stressed_metrics[metric_key]

            if metric_key == 'brier_score':
                deg = ((stressed_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else (np.inf if stressed_val > 0 else 0)
            else:
                deg = ((baseline_val - stressed_val) / baseline_val) * 100 if baseline_val != 0 else (np.inf if stressed_val < 0 else 0)

            stressed_metrics[f'degradation_{metric_key}_percent'] = deg

    if 'max_subgroup_delta_auc' in baseline_metrics and 'max_subgroup_delta_auc' in stressed_metrics and not pd.isna(stressed_metrics['max_subgroup_delta_auc']):
        b_delta = baseline_metrics['max_subgroup_delta_auc']
        s_delta = stressed_metrics['max_subgroup_delta_auc']
        stressed_metrics['degradation_max_subgroup_delta_auc_percent'] = ((s_delta - b_delta) / b_delta) * 100 if b_delta != 0 else (np.inf if s_delta > 0 else 0)

    return stressed_metrics

def evaluate_calibration_under_stress(model, X_original, y_original, sensitive_attr_original,
                                      scenario_name, transformation_func, transform_params,
                                      X_baseline_cols, baseline_metrics):
    
    X_stressed_df = transformation_func(X_original.copy(), transform_params['features'], **{k:v for k,v in transform_params.items() if k not in ['features']})
    X_stressed_processed = preprocess_stressed_data(X_stressed_df, X_baseline_cols)
    stressed_metrics = evaluate_model_performance(model, X_stressed_processed, y_original, sensitive_attr_original, scenario_name)

    baseline_brier = baseline_metrics['brier_score']
    stressed_brier = stressed_metrics['brier_score']
    stressed_metrics['degradation_brier_score_percent'] = ((stressed_brier - baseline_brier) / baseline_brier) * 100 if baseline_brier != 0 else np.inf
    
    return stressed_metrics

def evaluate_subgroup_stress(model, X_original, y_original, sensitive_attr_original,
                             scenario_name, target_group, baseline_metrics):
    
    group_mask = (sensitive_attr_original == target_group)
    if group_mask.sum() == 0:
        return None

    X_subgroup = X_original[group_mask]
    y_subgroup = y_original[group_mask]
    sensitive_attr_subgroup = sensitive_attr_original[group_mask]

    subgroup_metrics = evaluate_model_performance(model, X_subgroup, y_subgroup, sensitive_attr_subgroup, scenario_name)

    if 'auc' in subgroup_metrics and 'auc' in baseline_metrics and not pd.isna(subgroup_metrics['auc']):
        s_auc = subgroup_metrics['auc']
        b_auc = baseline_metrics['auc']
        subgroup_metrics['degradation_auc_percent'] = ((b_auc - s_auc) / b_auc) * 100 if b_auc != 0 else (np.inf if s_auc < 0 else 0)

    return subgroup_metrics

def evaluate_tail_slice_stress(model, X_original, y_original, sensitive_attr_original,
                               scenario_name, feature, percentile, slice_type, baseline_metrics):
    
    if feature not in X_original.columns or not pd.api.types.is_numeric_dtype(X_original[feature]):
        return None

    threshold_val = X_original[feature].quantile(percentile / 100.0)
    if slice_type == 'bottom':
        slice_mask = (X_original[feature] <= threshold_val)
    else:
        slice_mask = (X_original[feature] >= threshold_val)

    if slice_mask.sum() == 0:
        return None

    X_slice = X_original[slice_mask]
    y_slice = y_original[slice_mask]
    sensitive_attr_slice = sensitive_attr_original[slice_mask]

    slice_metrics = evaluate_model_performance(model, X_slice, y_slice, sensitive_attr_slice, scenario_name)

    if 'auc' in slice_metrics and 'auc' in baseline_metrics and not pd.isna(slice_metrics['auc']):
        s_auc = slice_metrics['auc']
        b_auc = baseline_metrics['auc']
        slice_metrics['degradation_auc_percent'] = ((b_auc - s_auc) / b_auc) * 100 if b_auc != 0 else (np.inf if s_auc < 0 else 0)

    return slice_metrics

def check_threshold_violations(scenario_results, critical_thresholds, warn_thresholds):
    critical_violations = []
    warn_violations = []

    for _, row in scenario_results.iterrows():
        scenario_name = row['scenario']

        if 'auc' in row and not pd.isna(row['auc']):
            if row['auc'] < critical_thresholds['min_auc']:
                critical_violations.append({'scenario': scenario_name, 'metric': 'AUC', 'value': row['auc'], 'threshold': critical_thresholds['min_auc'], 'type': 'CRITICAL_MIN_AUC'})
            elif row['auc'] < warn_thresholds['min_auc']:
                warn_violations.append({'scenario': scenario_name, 'metric': 'AUC', 'value': row['auc'], 'threshold': warn_thresholds['min_auc'], 'type': 'WARN_MIN_AUC'})

        if 'degradation_auc_percent' in row and not pd.isna(row['degradation_auc_percent']):
            if row['degradation_auc_percent'] > critical_thresholds['max_degradation_auc_percent']:
                critical_violations.append({'scenario': scenario_name, 'metric': 'AUC Degradation (%)', 'value': row['degradation_auc_percent'], 'threshold': critical_thresholds['max_degradation_auc_percent'], 'type': 'CRITICAL_AUC_DEGRADATION'})
            elif row['degradation_auc_percent'] > warn_thresholds['max_degradation_auc_percent']:
                warn_violations.append({'scenario': scenario_name, 'metric': 'AUC Degradation (%)', 'value': row['degradation_auc_percent'], 'threshold': warn_thresholds['max_degradation_auc_percent'], 'type': 'WARN_AUC_DEGRADATION'})

        if 'brier_score' in row and not pd.isna(row['brier_score']):
            if row['brier_score'] > critical_thresholds['max_brier_score']:
                critical_violations.append({'scenario': scenario_name, 'metric': 'Brier Score', 'value': row['brier_score'], 'threshold': critical_thresholds['max_brier_score'], 'type': 'CRITICAL_MAX_BRIER'})
            elif row['brier_score'] > warn_thresholds['max_brier_score']:
                warn_violations.append({'scenario': scenario_name, 'metric': 'Brier Score', 'value': row['brier_score'], 'threshold': warn_thresholds['max_brier_score'], 'type': 'WARN_MAX_BRIER'})

        if 'max_subgroup_delta_auc' in row and not pd.isna(row['max_subgroup_delta_auc']):
            if row['max_subgroup_delta_auc'] > critical_thresholds['max_subgroup_delta_auc']:
                critical_violations.append({'scenario': scenario_name, 'metric': 'Max Subgroup Delta AUC', 'value': row['max_subgroup_delta_auc'], 'threshold': critical_thresholds['max_subgroup_delta_auc'], 'type': 'CRITICAL_SUBGROUP_DELTA'})
            elif row['max_subgroup_delta_auc'] > warn_thresholds['max_subgroup_delta_auc']:
                warn_violations.append({'scenario': scenario_name, 'metric': 'Max Subgroup Delta AUC', 'value': row['max_subgroup_delta_auc'], 'threshold': warn_thresholds['max_subgroup_delta_auc'], 'type': 'WARN_SUBGROUP_DELTA'})

    return critical_violations, warn_violations

def make_go_no_go_decision(critical_violations, warn_violations):
    if critical_violations:
        decision = "NO GO"
        recommendation = "The model has critical performance degradations under stress. It is not approved for deployment."
    elif warn_violations:
        decision = "GO WITH MITIGATION"
        recommendation = "The model exhibits warning-level degradations. Deployment approved with mitigation strategies."
    else:
        decision = "GO"
        recommendation = "The model demonstrates sufficient robustness. Approved for deployment."
    return decision, recommendation

def plot_degradation_curves(scenario_results, baseline_metrics_dict):
    metrics_to_plot = ['auc', 'brier_score']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        baseline_value = baseline_metrics_dict[metric]
        scenario_values_series = scenario_results.set_index('scenario')[metric].drop('Baseline', errors='ignore')
        plot_values = [baseline_value] + scenario_values_series.tolist()
        plot_labels = ['Baseline'] + scenario_values_series.index.tolist()
        
        colors = ['skyblue'] + ['lightcoral'] * len(scenario_values_series)
        ax.bar(plot_labels, plot_values, color=colors)
        ax.axhline(y=baseline_value, color='gray', linestyle='--', label='Baseline')

        if metric == 'auc':
            ax.axhline(y=CRITICAL_THRESHOLDS['min_auc'], color='red', linestyle=':', label='Critical Min AUC')
            ax.axhline(y=WARN_THRESHOLDS['min_auc'], color='orange', linestyle=':', label='Warn Min AUC')
            ax.set_title('Model AUC Across Scenarios')
        else:
            ax.axhline(y=CRITICAL_THRESHOLDS['max_brier_score'], color='red', linestyle=':', label='Critical Max Brier')
            ax.axhline(y=WARN_THRESHOLDS['max_brier_score'], color='orange', linestyle=':', label='Warn Max Brier')
            ax.set_title('Model Brier Score Across Scenarios')

        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    except FileNotFoundError:
        return "FILE_NOT_FOUND"
    return sha256_hash.hexdigest()

def export_artifacts(artifacts: dict, out_dir: str, run_id: str) -> dict:
    manifest_data = {}
    zip_filename = os.path.join(os.path.dirname(out_dir), f'Session_06_{run_id}.zip')
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for name, path in artifacts.items():
            if os.path.exists(path):
                zipf.write(path, os.path.basename(path))
                manifest_data[name] = {
                    'path': os.path.basename(path),
                    'sha256': calculate_sha256(path)
                }
    
    manifest_path = os.path.join(out_dir, 'evidence_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=4)
        
    return manifest_data

# 3) Main pipeline function
def run_full_validation_pipeline(data_path, model_path, base_output_dir):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_output_dir = os.path.join(base_output_dir, f'session06_{run_id}')
    os.makedirs(session_output_dir, exist_ok=True)
    
    # 1. Load Assets
    X_baseline, y_baseline, sens_attr, model = load_assets(data_path, model_path, FEATURE_COLS, TARGET_COL, SENSITIVE_ATTRIBUTE)
    
    # 2. Baseline Evaluation
    baseline_metrics = evaluate_model_performance(model, X_baseline, y_baseline, sens_attr, "Baseline")
    results_list = [baseline_metrics]
    config_snapshot = {}

    # 3. Scenario Runs
    # Gaussian Noise
    noise_params = {'features': ['Age', 'Income', 'LoanAmount', 'CreditScore'], 'noise_std_multiplier': 0.5, 'random_state': RANDOM_SEED}
    config_snapshot['gaussian_noise_high'] = noise_params
    results_list.append(run_and_evaluate_scenario(model, X_baseline, y_baseline, sens_attr, "Gaussian Noise (High)", apply_gaussian_noise, noise_params, X_baseline, baseline_metrics))

    # Economic Shift
    shift_params = {'features': ['Income', 'LoanAmount'], 'shift_factor': 0.8, 'random_state': RANDOM_SEED}
    config_snapshot['economic_shift_income_loan_down'] = shift_params
    results_list.append(run_and_evaluate_scenario(model, X_baseline, y_baseline, sens_attr, "Economic Shift (Income/Loan -20%)", apply_feature_scaling_shift, shift_params, X_baseline, baseline_metrics))

    # Missingness
    miss_params = {'features': ['CreditScore', 'LoanDuration', 'Income'], 'missing_rate': 0.2, 'random_state': RANDOM_SEED}
    config_snapshot['missingness_spike'] = miss_params
    results_list.append(run_and_evaluate_scenario(model, X_baseline, y_baseline, sens_attr, "Missingness Spike (20%)", apply_missingness_spike, miss_params, X_baseline, baseline_metrics))

    # Calibration Stress
    cal_params = {'features': ['Age', 'Income', 'LoanAmount', 'CreditScore'], 'noise_std_multiplier': 0.7, 'random_state': RANDOM_SEED + 1}
    config_snapshot['calibration_noise_high'] = cal_params
    results_list.append(evaluate_calibration_under_stress(model, X_baseline, y_baseline, sens_attr, "Calibration Stress (Noise High)", apply_gaussian_noise, cal_params, X_baseline, baseline_metrics))

    # Subgroup Stress
    config_snapshot['subgroup_credit_poor'] = {'sensitive_attribute': SENSITIVE_ATTRIBUTE, 'target_group': 'Poor'}
    res_poor = evaluate_subgroup_stress(model, X_baseline, y_baseline, sens_attr, "Subgroup Stress (CreditScore Band: Poor)", 'Poor', baseline_metrics)
    if res_poor: results_list.append(res_poor)

    config_snapshot['subgroup_credit_excellent'] = {'sensitive_attribute': SENSITIVE_ATTRIBUTE, 'target_group': 'Excellent'}
    res_exc = evaluate_subgroup_stress(model, X_baseline, y_baseline, sens_attr, "Subgroup Stress (CreditScore Band: Excellent)", 'Excellent', baseline_metrics)
    if res_exc: results_list.append(res_exc)

    # Tail Slice
    config_snapshot['tail_income_low'] = {'feature': 'Income', 'percentile': 10, 'slice_type': 'bottom'}
    res_tail_inc = evaluate_tail_slice_stress(model, X_baseline, y_baseline, sens_attr, "Tail Stress (Low Income)", 'Income', 10, 'bottom', baseline_metrics)
    if res_tail_inc: results_list.append(res_tail_inc)

    # 4. Process Results
    results_df = pd.DataFrame(results_list)
    crit_v, warn_v = check_threshold_violations(results_df, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
    decision, recommendation = make_go_no_go_decision(crit_v, warn_v)
    
    # 5. Visuals
    fig = plot_degradation_curves(results_df, baseline_metrics)
    fig_path = os.path.join(session_output_dir, 'degradation_curves.png')
    fig.savefig(fig_path)
    plt.close(fig)

    # 6. Artifact Prep
    artifacts = {}
    
    baseline_path = os.path.join(session_output_dir, 'baseline_metrics.json')
    with open(baseline_path, 'w') as f: json.dump(baseline_metrics, f, indent=4)
    artifacts['baseline_metrics.json'] = baseline_path

    results_path = os.path.join(session_output_dir, 'scenario_results.json')
    results_df.to_json(results_path, orient='records', indent=4)
    artifacts['scenario_results.json'] = results_path

    viol_path = os.path.join(session_output_dir, 'violations_list.json')
    with open(viol_path, 'w') as f: json.dump({'critical': crit_v, 'warn': warn_v}, f, indent=4)
    artifacts['violations_list.json'] = viol_path
    
    artifacts['degradation_curves.png'] = fig_path

    config_path = os.path.join(session_output_dir, 'config_snapshot.json')
    full_config = {
        'run_id': run_id, 'random_seed': RANDOM_SEED, 'dataset_path': data_path,
        'model_path': model_path, 'features': FEATURE_COLS, 'target': TARGET_COL,
        'sensitive_attribute': SENSITIVE_ATTRIBUTE, 'critical_thresholds': CRITICAL_THRESHOLDS,
        'warn_thresholds': WARN_THRESHOLDS, 'scenario_configurations': config_snapshot
    }
    with open(config_path, 'w') as f: json.dump(full_config, f, indent=4)
    artifacts['config_snapshot.json'] = config_path

    summary_path = os.path.join(session_output_dir, f'session06_{run_id}_executive_summary.md')
    with open(summary_path, 'w') as f:
        f.write(f"# Executive Summary\n\nDecision: {decision}\n\n{recommendation}\n")
    artifacts['session06_executive_summary.md'] = summary_path

    manifest = export_artifacts(artifacts, session_output_dir, run_id)
    
    return {
        'results_df': results_df,
        'decision': decision,
        'recommendation': recommendation,
        'figure': fig,
        'manifest': manifest,
        'output_dir': session_output_dir
    }

if __name__ == "__main__":
    # Demonstration code
    temp_dir = 'temp_demo'
    os.makedirs(temp_dir, exist_ok=True)
    d_path, m_path = initialize_synthetic_environment(temp_dir)
    pipeline_results = run_full_validation_pipeline(d_path, m_path, 'reports')
    print(f"Pipeline Complete. Decision: {pipeline_results['decision']}")
