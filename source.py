import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import joblib
import json
import os
import shutil
import hashlib
import zipfile
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearnex import patch_sklearn

# Patch sklearn for performance
patch_sklearn()

# --- 1. Constants and Contracts ---
RANDOM_SEED = 42
TARGET_COL = 'true_label'
SENSITIVE_ATTRIBUTE = 'credit_score_band'
FEATURE_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'LoanDuration', 'DependentCount']
ID_COLS = []

SCHEMA = {
    'Age': {'dtype': 'int', 'range': (20, 70)},
    'Income': {'dtype': 'float', 'range': (0, 200000)},
    'LoanAmount': {'dtype': 'float', 'range': (0, 100000)},
    'CreditScore': {'dtype': 'int', 'range': (300, 850)},
    'LoanDuration': {'dtype': 'int', 'range': (12, 60)},
    'DependentCount': {'dtype': 'int', 'range': (0, 5)},
    'credit_score_band': {'dtype': 'category', 'options': ['Poor', 'Fair', 'Good', 'Excellent']},
    'true_label': {'dtype': 'int', 'options': [0, 1]}
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

# --- 2. Pure Business-Logic Functions ---

def set_global_seed(seed: int) -> None:
    """Sets the global seed for reproducibility."""
    np.random.seed(seed)

def generate_synthetic_data(seed: int, num_samples: int = 1000) -> tuple[pd.DataFrame, LogisticRegression]:
    """Generates synthetic credit data and a pre-trained model for demonstration."""
    np.random.seed(seed)
    data = {
        'Age': np.random.randint(20, 70, num_samples),
        'Income': np.random.normal(50000, 15000, num_samples),
        'LoanAmount': np.random.normal(15000, 5000, num_samples),
        'CreditScore': np.random.randint(300, 850, num_samples),
        'LoanDuration': np.random.randint(12, 60, num_samples),
        'DependentCount': np.random.randint(0, 5, num_samples)
    }
    df = pd.DataFrame(data)
    
    # Sensitive Attribute
    bins = [0, 580, 670, 740, 850]
    labels = ['Poor', 'Fair', 'Good', 'Excellent']
    df[SENSITIVE_ATTRIBUTE] = pd.cut(df['CreditScore'], bins=bins, labels=labels, right=False)
    
    # Target variable
    df[TARGET_COL] = ((df['CreditScore'] < 600) & (df['LoanAmount'] > 20000) | (df['Income'] < 30000)).astype(int)
    
    # Model Training
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(imputer.fit_transform(X))
    
    model = LogisticRegression(random_state=seed, solver='liblinear')
    model.fit(X_processed, y)
    
    return df, model

def validate_dataset(df: pd.DataFrame, expected_features: list[str], target: str) -> None:
    """Checks for schema drift and missing columns."""
    missing = [col for col in expected_features + [target] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

def load_assets(data_path: str, model_path: str, features: list[str], target: str, sensitive_col: str):
    """Loads dataset and model, and prepares baseline features."""
    df_test = pd.read_csv(data_path)
    model = joblib.load(model_path)
    
    X_test_raw = df_test[features]
    
    # Pre-processing consistent with notebook: refitting on test for demo
    local_imputer = SimpleImputer(strategy='mean')
    X_imputed = local_imputer.fit_transform(X_test_raw)
    local_scaler = StandardScaler()
    X_scaled = local_scaler.fit_transform(X_imputed)
    
    X_test_aligned = pd.DataFrame(X_scaled, columns=features)
    y_test = df_test[target]
    sensitive_attr = df_test[sensitive_col]
    
    return X_test_aligned, y_test, sensitive_attr, model

def evaluate_model_performance(model, X: pd.DataFrame, y: pd.Series, sensitive_attr: pd.Series = None, scenario_name: str = "Baseline") -> dict:
    """Computes performance metrics and subgroup disparities."""
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
            mask = (sensitive_attr == group)
            if mask.sum() > 1 and len(np.unique(y[mask])) > 1:
                subgroup_auc = roc_auc_score(y[mask], y_pred_proba[mask])
                subgroup_aucs[f'auc_{group}'] = subgroup_auc
            else:
                subgroup_aucs[f'auc_{group}'] = np.nan

        subgroup_deltas = {f'delta_auc_{g}': abs(v - overall_auc) for g, v in subgroup_aucs.items() if not np.isnan(v)}
        metrics.update(subgroup_aucs)
        metrics['max_subgroup_delta_auc'] = max(subgroup_deltas.values()) if subgroup_deltas else np.nan
        metrics['subgroup_deltas'] = subgroup_deltas

    return metrics

def apply_gaussian_noise(df: pd.DataFrame, features_to_noise: list[str], noise_std_multiplier: float, random_state: int) -> pd.DataFrame:
    df_stressed = df.copy()
    rng = np.random.RandomState(random_state)
    for col in features_to_noise:
        if col in df_stressed.columns and pd.api.types.is_numeric_dtype(df_stressed[col]):
            std_dev = df_stressed[col].std()
            noise = rng.normal(loc=0, scale=std_dev * noise_std_multiplier, size=len(df_stressed))
            df_stressed[col] = df_stressed[col] + noise
    return df_stressed

def apply_feature_scaling_shift(df: pd.DataFrame, features_to_shift: list[str], shift_factor: float, random_state: int) -> pd.DataFrame:
    df_stressed = df.copy()
    for col in features_to_shift:
        if col in df_stressed.columns and pd.api.types.is_numeric_dtype(df_stressed[col]):
            df_stressed[col] = df_stressed[col] * shift_factor
    return df_stressed

def apply_missingness_spike(df: pd.DataFrame, features_to_spike: list[str], missing_rate: float, random_state: int) -> pd.DataFrame:
    df_stressed = df.copy()
    rng = np.random.RandomState(random_state)
    for col in features_to_spike:
        if col in df_stressed.columns:
            mask = rng.rand(len(df_stressed)) < missing_rate
            df_stressed.loc[mask, col] = np.nan
    return df_stressed

def preprocess_stressed_data(X_stressed: pd.DataFrame, X_baseline_cols: list[str]) -> pd.DataFrame:
    """Ensures stressed data matches baseline schema and is preprocessed."""
    X_processed = X_stressed.copy()
    # Demo-consistent refitting
    local_imputer = SimpleImputer(strategy='mean')
    for col in X_processed.columns:
        if X_processed[col].isnull().any() and pd.api.types.is_numeric_dtype(X_processed[col]):
            X_processed[col] = local_imputer.fit_transform(X_processed[[col]])

    local_scaler = StandardScaler()
    X_scaled = local_scaler.fit_transform(X_processed)
    X_final = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)

    missing_cols = set(X_baseline_cols) - set(X_final.columns)
    for c in missing_cols: X_final[c] = 0
    
    extra_cols = set(X_final.columns) - set(X_baseline_cols)
    if extra_cols: X_final = X_final.drop(columns=list(extra_cols))
    
    return X_final[X_baseline_cols]

def run_and_evaluate_scenario(
    model, X_original, y_original, sensitive_attr_original,
    scenario_name, transformation_func, transform_params,
    X_baseline_cols, baseline_metrics, metrics_collector,
    config_snapshot, scenario_config_key
):
    """Executes a single stress scenario and calculates degradation."""
    config_snapshot[scenario_config_key] = transform_params
    
    X_stressed_raw = transformation_func(
        X_original.copy(), 
        transform_params['features'], 
        **{k: v for k, v in transform_params.items() if k not in ['features']}
    )
    
    X_stressed_processed = preprocess_stressed_data(X_stressed_raw, X_baseline_cols)
    stressed_metrics = evaluate_model_performance(model, X_stressed_processed, y_original, sensitive_attr_original, scenario_name)
    
    # Calculate Degradation
    for key in ['auc', 'accuracy', 'precision', 'recall', 'brier_score']:
        if key in baseline_metrics and key in stressed_metrics and not pd.isna(stressed_metrics[key]):
            b_val = baseline_metrics[key]
            s_val = stressed_metrics[key]
            if key == 'brier_score':
                deg = ((s_val - b_val) / b_val * 100) if b_val != 0 else 0
            else:
                deg = ((b_val - s_val) / b_val * 100) if b_val != 0 else 0
            stressed_metrics[f'degradation_{key}_percent'] = deg

    # Special case for subgroup delta
    if 'max_subgroup_delta_auc' in baseline_metrics and 'max_subgroup_delta_auc' in stressed_metrics:
        b_delta = baseline_metrics['max_subgroup_delta_auc']
        s_delta = stressed_metrics['max_subgroup_delta_auc']
        stressed_metrics['degradation_max_subgroup_delta_auc_percent'] = ((s_delta - b_delta) / b_delta * 100) if b_delta != 0 else 0

    metrics_collector.append(stressed_metrics)
    return stressed_metrics

def evaluate_subgroup_stress(model, X_original, y_original, sensitive_attr_original, scenario_name, target_group, X_baseline_cols, baseline_metrics, metrics_collector, config_snapshot, scenario_config_key):
    config_snapshot[scenario_config_key] = {'sensitive_attribute': SENSITIVE_ATTRIBUTE, 'target_group': target_group}
    mask = (sensitive_attr_original == target_group)
    if mask.sum() == 0: return None
    
    subgroup_metrics = evaluate_model_performance(model, X_original[mask], y_original[mask], sensitive_attr_original[mask], scenario_name)
    
    if 'auc' in subgroup_metrics and 'auc' in baseline_metrics:
        b_auc = baseline_metrics['auc']
        s_auc = subgroup_metrics['auc']
        subgroup_metrics['degradation_auc_percent'] = ((b_auc - s_auc) / b_auc * 100) if b_auc != 0 else 0
    
    metrics_collector.append(subgroup_metrics)
    return subgroup_metrics

def evaluate_tail_slice_stress(model, X_original, y_original, sensitive_attr_original, scenario_name, feature, percentile, slice_type, X_baseline_cols, baseline_metrics, metrics_collector, config_snapshot, scenario_config_key):
    config_snapshot[scenario_config_key] = {'feature': feature, 'percentile': percentile, 'slice_type': slice_type}
    threshold = X_original[feature].quantile(percentile / 100.0)
    mask = (X_original[feature] <= threshold) if slice_type == 'bottom' else (X_original[feature] >= threshold)
    
    if mask.sum() == 0: return None
    
    slice_metrics = evaluate_model_performance(model, X_original[mask], y_original[mask], sensitive_attr_original[mask], scenario_name)
    if 'auc' in slice_metrics and 'auc' in baseline_metrics:
        b_auc = baseline_metrics['auc']
        s_auc = slice_metrics['auc']
        slice_metrics['degradation_auc_percent'] = ((b_auc - s_auc) / b_auc * 100) if b_auc != 0 else 0
    
    metrics_collector.append(slice_metrics)
    return slice_metrics

def check_threshold_violations(scenario_results_df: pd.DataFrame, critical_thresholds: dict, warn_thresholds: dict) -> tuple[list, list]:
    critical_violations = []
    warn_violations = []

    for _, row in scenario_results_df.iterrows():
        s_name = row['scenario']
        # AUC
        if 'auc' in row and not pd.isna(row['auc']):
            if row['auc'] < critical_thresholds['min_auc']:
                critical_violations.append({'scenario': s_name, 'metric': 'AUC', 'value': row['auc'], 'threshold': critical_thresholds['min_auc'], 'type': 'CRITICAL_MIN_AUC'})
            elif row['auc'] < warn_thresholds['min_auc']:
                warn_violations.append({'scenario': s_name, 'metric': 'AUC', 'value': row['auc'], 'threshold': warn_thresholds['min_auc'], 'type': 'WARN_MIN_AUC'})
        # Degradation
        if 'degradation_auc_percent' in row and not pd.isna(row['degradation_auc_percent']):
            if row['degradation_auc_percent'] > critical_thresholds['max_degradation_auc_percent']:
                critical_violations.append({'scenario': s_name, 'metric': 'AUC Degradation (%)', 'value': row['degradation_auc_percent'], 'threshold': critical_thresholds['max_degradation_auc_percent'], 'type': 'CRITICAL_AUC_DEGRADATION'})
            elif row['degradation_auc_percent'] > warn_thresholds['max_degradation_auc_percent']:
                warn_violations.append({'scenario': s_name, 'metric': 'AUC Degradation (%)', 'value': row['degradation_auc_percent'], 'threshold': warn_thresholds['max_degradation_auc_percent'], 'type': 'WARN_AUC_DEGRADATION'})
        # Brier
        if 'brier_score' in row and not pd.isna(row['brier_score']):
            if row['brier_score'] > critical_thresholds['max_brier_score']:
                critical_violations.append({'scenario': s_name, 'metric': 'Brier Score', 'value': row['brier_score'], 'threshold': critical_thresholds['max_brier_score'], 'type': 'CRITICAL_MAX_BRIER'})
            elif row['brier_score'] > warn_thresholds['max_brier_score']:
                warn_violations.append({'scenario': s_name, 'metric': 'Brier Score', 'value': row['brier_score'], 'threshold': warn_thresholds['max_brier_score'], 'type': 'WARN_MAX_BRIER'})
        # Fairness
        if 'max_subgroup_delta_auc' in row and not pd.isna(row['max_subgroup_delta_auc']):
            if row['max_subgroup_delta_auc'] > critical_thresholds['max_subgroup_delta_auc']:
                critical_violations.append({'scenario': s_name, 'metric': 'Max Subgroup Delta AUC', 'value': row['max_subgroup_delta_auc'], 'threshold': critical_thresholds['max_subgroup_delta_auc'], 'type': 'CRITICAL_SUBGROUP_DELTA'})
            elif row['max_subgroup_delta_auc'] > warn_thresholds['max_subgroup_delta_auc']:
                warn_violations.append({'scenario': s_name, 'metric': 'Max Subgroup Delta AUC', 'value': row['max_subgroup_delta_auc'], 'threshold': warn_thresholds['max_subgroup_delta_auc'], 'type': 'WARN_SUBGROUP_DELTA'})

    return critical_violations, warn_violations

def make_go_no_go_decision(critical_violations: list, warn_violations: list) -> tuple[str, str]:
    if critical_violations:
        return "NO GO", "The model has critical performance degradations under stress."
    if warn_violations:
        return "GO WITH MITIGATION", "The model exhibits warning-level degradations under certain scenarios."
    return "GO", "The model demonstrates sufficient robustness under all tested stress scenarios."

def plot_degradation_curves(scenario_results: pd.DataFrame, baseline_metrics: dict, critical_thresholds: dict, warn_thresholds: dict):
    metrics_to_plot = ['auc', 'brier_score']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        b_val = baseline_metrics[metric]
        s_series = scenario_results.set_index('scenario')[metric].drop('Baseline', errors='ignore')
        plot_values = [b_val] + s_series.tolist()
        plot_labels = ['Baseline'] + s_series.index.tolist()
        colors = ['skyblue'] + ['lightcoral'] * len(s_series)

        ax.bar(plot_labels, plot_values, color=colors)
        ax.axhline(y=b_val, color='gray', linestyle='--', label='Baseline')

        if metric == 'auc':
            ax.axhline(y=critical_thresholds['min_auc'], color='red', linestyle=':', label='Critical Min AUC')
            ax.axhline(y=warn_thresholds['min_auc'], color='orange', linestyle=':', label='Warn Min AUC')
            ax.set_title('Model AUC Across Scenarios')
        else:
            ax.axhline(y=critical_thresholds['max_brier_score'], color='red', linestyle=':', label='Critical Max Brier')
            ax.axhline(y=warn_thresholds['max_brier_score'], color='orange', linestyle=':', label='Warn Max Brier')
            ax.set_title('Model Brier Score Across Scenarios')

        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def calculate_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    except FileNotFoundError:
        return "FILE_NOT_FOUND"
    return sha256_hash.hexdigest()

def export_artifacts(
    out_dir: str,
    baseline_metrics: dict,
    scenario_results_df: pd.DataFrame,
    critical_violations: list,
    warn_violations: list,
    scenario_config: dict,
    decision: str,
    recommendation: str,
    run_id: str
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}

    # JSON Exports
    paths = {
        'baseline_metrics.json': os.path.join(out_dir, 'baseline_metrics.json'),
        'scenario_results.json': os.path.join(out_dir, 'scenario_results.json'),
        'violations_list.json': os.path.join(out_dir, 'violations_list.json'),
        'config_snapshot.json': os.path.join(out_dir, 'config_snapshot.json')
    }

    with open(paths['baseline_metrics.json'], 'w') as f: json.dump(baseline_metrics, f, indent=4)
    scenario_results_df.to_json(paths['scenario_results.json'], orient='records', indent=4)
    with open(paths['violations_list.json'], 'w') as f: 
        json.dump({'critical': critical_violations, 'warn': warn_violations}, f, indent=4)
    
    full_config = {
        'run_id': run_id,
        'features': FEATURE_COLS,
        'target': TARGET_COL,
        'thresholds': {'critical': CRITICAL_THRESHOLDS, 'warn': WARN_THRESHOLDS},
        'scenarios': scenario_config
    }
    with open(paths['config_snapshot.json'], 'w') as f: json.dump(full_config, f, indent=4)

    # Summary MD
    summary_path = os.path.join(out_dir, f'executive_summary_{run_id}.md')
    with open(summary_path, 'w') as f:
        f.write(f"# Model Validation Summary - {run_id}\n\nDecision: {decision}\n\n{recommendation}")
    paths['executive_summary.md'] = summary_path

    # Hash manifest
    for name, path in paths.items():
        manifest[name] = {'path': os.path.basename(path), 'sha256': calculate_sha256(path)}
    
    manifest_path = os.path.join(out_dir, 'evidence_manifest.json')
    with open(manifest_path, 'w') as f: json.dump(manifest, f, indent=4)
    
    return manifest

# --- 3. Determinism ---
# Seed logic is encapsulated in pure functions and called in main.

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = f"reports/val_{RUN_ID}"
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # 1. Setup
    set_global_seed(RANDOM_SEED)
    df, model = generate_synthetic_data(RANDOM_SEED)
    
    data_path = os.path.join(BASE_DIR, "test_data.csv")
    model_path = os.path.join(BASE_DIR, "model.pkl")
    df.to_csv(data_path, index=False)
    joblib.dump(model, model_path)
    
    # 2. Baseline
    X_base, y_base, sens_base, model_loaded = load_assets(data_path, model_path, FEATURE_COLS, TARGET_COL, SENSITIVE_ATTRIBUTE)
    baseline_metrics = evaluate_model_performance(model_loaded, X_base, y_base, sens_base, "Baseline")
    
    # 3. Scenarios
    results_list = [baseline_metrics]
    config_snaps = {}
    
    run_and_evaluate_scenario(
        model_loaded, X_base, y_base, sens_base, 
        "Gaussian Noise", apply_gaussian_noise, 
        {'features': FEATURE_COLS[:4], 'noise_std_multiplier': 0.5, 'random_state': RANDOM_SEED},
        FEATURE_COLS, baseline_metrics, results_list, config_snaps, "noise_high"
    )
    
    run_and_evaluate_scenario(
        model_loaded, X_base, y_base, sens_base, 
        "Economic Shift", apply_feature_scaling_shift, 
        {'features': ['Income', 'LoanAmount'], 'shift_factor': 0.8, 'random_state': RANDOM_SEED},
        FEATURE_COLS, baseline_metrics, results_list, config_snaps, "econ_shift"
    )

    evaluate_subgroup_stress(
        model_loaded, X_base, y_base, sens_base, "Subgroup Stress (Poor)", "Poor", 
        FEATURE_COLS, baseline_metrics, results_list, config_snaps, "subgroup_poor"
    )

    # 4. Final Aggregation
    final_df = pd.DataFrame(results_list)
    crit, warn = check_threshold_violations(final_df, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
    decision, rec = make_go_no_go_decision(crit, warn)
    
    # 5. Export
    fig = plot_degradation_curves(final_df, baseline_metrics, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
    fig.savefig(os.path.join(BASE_DIR, "degradation_curves.png"))
    
    manifest = export_artifacts(BASE_DIR, baseline_metrics, final_df, crit, warn, config_snaps, decision, rec, RUN_ID)
    print(f"Validation complete. Decision: {decision}")