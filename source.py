import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import joblib
import json
import os
import hashlib
import zipfile
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearnex import patch_sklearn

# Patching sklearn for performance
patch_sklearn()

# --- 1. Constants and Contracts ---
RANDOM_SEED: int = 42
TARGET_COL: str = 'true_label'
FEATURE_COLS: list[str] = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'LoanDuration', 'DependentCount']
ID_COLS: list[str] = []
SENSITIVE_ATTRIBUTE: str = 'credit_score_band'

SCHEMA: dict = {
    'Age': {'dtype': 'int', 'range': (20, 70)},
    'Income': {'dtype': 'float', 'range': (0, np.inf)},
    'LoanAmount': {'dtype': 'float', 'range': (0, np.inf)},
    'CreditScore': {'dtype': 'int', 'range': (300, 850)},
    'LoanDuration': {'dtype': 'int', 'range': (12, 60)},
    'DependentCount': {'dtype': 'int', 'range': (0, 5)},
    'true_label': {'dtype': 'int', 'range': (0, 1)},
    'credit_score_band': {'dtype': 'category', 'categories': ['Poor', 'Fair', 'Good', 'Excellent']}
}

CRITICAL_THRESHOLDS: dict = {
    'min_auc': 0.70,
    'max_degradation_auc_percent': 15.0,
    'max_brier_score': 0.25,
    'max_subgroup_delta_auc': 0.10
}

WARN_THRESHOLDS: dict = {
    'min_auc': 0.75,
    'max_degradation_auc_percent': 10.0,
    'max_brier_score': 0.20,
    'max_subgroup_delta_auc': 0.05
}

# --- 2. Pure Business-Logic Functions ---

def set_global_seed(seed: int) -> None:
    """Sets the seed for reproducibility."""
    np.random.seed(seed)

def generate_synthetic_data(num_samples: int, seed: int) -> pd.DataFrame:
    """Generates synthetic credit data exactly as defined in the notebook."""
    rng = np.random.RandomState(seed)
    data = {
        'Age': rng.randint(20, 70, num_samples),
        'Income': rng.normal(50000, 15000, num_samples),
        'LoanAmount': rng.normal(15000, 5000, num_samples),
        'CreditScore': rng.randint(300, 850, num_samples),
        'LoanDuration': rng.randint(12, 60, num_samples),
        'DependentCount': rng.randint(0, 5, num_samples)
    }
    df = pd.DataFrame(data)
    
    # Create credit_score_band
    bins = [0, 580, 670, 740, 850]
    labels = ['Poor', 'Fair', 'Good', 'Excellent']
    df[SENSITIVE_ATTRIBUTE] = pd.cut(df['CreditScore'], bins=bins, labels=labels, right=False)
    
    # Create target variable
    df[TARGET_COL] = ((df['CreditScore'] < 600) & (df['LoanAmount'] > 20000) | (df['Income'] < 30000)).astype(int)
    return df

def train_and_save_model(df: pd.DataFrame, features: list[str], target: str, model_path: str, seed: int) -> None:
    """Trains a simple Logistic Regression model and saves it."""
    X = df[features]
    y = df[target]
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    model = LogisticRegression(random_state=seed, solver='liblinear')
    model.fit(X_scaled, y)
    joblib.dump(model, model_path)

def validate_dataset(df: pd.DataFrame, features: list[str], target: str) -> None:
    """Checks for presence of required columns and basic schema validation."""
    required_cols = features + [target, SENSITIVE_ATTRIBUTE]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def load_assets(data_path: str, model_path: str, features: list[str], target: str, sensitive_attr_col: str):
    """Loads the test dataset and the pre-trained model and aligns them."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    df_test = pd.read_csv(data_path)
    model = joblib.load(model_path)
    
    validate_dataset(df_test, features, target)

    X_test_raw = df_test[features]
    
    # Refit imputer and scaler for demonstration purposes as per notebook
    local_imputer = SimpleImputer(strategy='mean')
    X_imputed_local = local_imputer.fit_transform(X_test_raw)
    local_scaler = StandardScaler()
    X_scaled_local = local_scaler.fit_transform(X_imputed_local)
    
    X_test_aligned = pd.DataFrame(X_scaled_local, columns=features)
    y_test = df_test[target]
    sensitive_attr = df_test[sensitive_attr_col]

    return X_test_aligned, y_test, sensitive_attr, model

def evaluate_model_performance(model, X: pd.DataFrame, y: pd.Series, sensitive_attr: pd.Series = None, scenario_name: str = "Baseline"):
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
        
        # Handle categorical types for sorting
        if hasattr(sensitive_attr, 'cat'):
            unique_groups = sensitive_attr.cat.categories.tolist()
        else:
            unique_groups = sorted(sensitive_attr.unique().tolist())

        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            if group_mask.sum() > 1 and len(np.unique(y[group_mask])) > 1:
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

def apply_gaussian_noise(df: pd.DataFrame, features_to_noise: list[str], noise_std_multiplier: float, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    df_stressed = df.copy()
    rng = np.random.RandomState(random_state)
    for col in features_to_noise:
        if col in df_stressed.columns and pd.api.types.is_numeric_dtype(df_stressed[col]):
            std_dev = df_stressed[col].std()
            noise_level = std_dev * noise_std_multiplier
            noise = rng.normal(loc=0, scale=noise_level, size=len(df_stressed))
            df_stressed[col] = df_stressed[col] + noise
    return df_stressed

def apply_feature_scaling_shift(df: pd.DataFrame, features_to_shift: list[str], shift_factor: float, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    df_stressed = df.copy()
    for col in features_to_shift:
        if col in df_stressed.columns and pd.api.types.is_numeric_dtype(df_stressed[col]):
            df_stressed[col] = df_stressed[col] * shift_factor
    return df_stressed

def apply_missingness_spike(df: pd.DataFrame, features_to_spike: list[str], missing_rate: float, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    df_stressed = df.copy()
    rng = np.random.RandomState(random_state)
    for col in features_to_spike:
        if col in df_stressed.columns:
            mask = rng.rand(len(df_stressed)) < missing_rate
            df_stressed.loc[mask, col] = np.nan
    return df_stressed

def preprocess_stressed_data(X_stressed: pd.DataFrame, reference_cols: list[str]) -> pd.DataFrame:
    """Mocks the preprocessing pipeline as done in the notebook."""
    X_processed = X_stressed.copy()
    local_imputer = SimpleImputer(strategy='mean')
    for col in X_processed.columns:
        if X_processed[col].isnull().any() and pd.api.types.is_numeric_dtype(X_processed[col]):
            X_processed[col] = local_imputer.fit_transform(X_processed[[col]])

    local_scaler = StandardScaler()
    X_scaled = local_scaler.fit_transform(X_processed)
    X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)

    # Ensure column logic
    for c in set(reference_cols) - set(X_processed.columns):
        X_processed[c] = 0
    extra_cols = set(X_processed.columns) - set(reference_cols)
    if extra_cols:
        X_processed = X_processed.drop(columns=list(extra_cols))
    
    return X_processed[reference_cols]

def calculate_degradation(baseline_metrics: dict, stressed_metrics: dict) -> dict:
    """Calculates percentage degradation for metrics."""
    degradation = {}
    for key in ['auc', 'accuracy', 'precision', 'recall', 'brier_score']:
        if key in baseline_metrics and key in stressed_metrics and not pd.isna(stressed_metrics[key]):
            base = baseline_metrics[key]
            stressed = stressed_metrics[key]
            if key == 'brier_score':
                val = ((stressed - base) / base * 100) if base != 0 else (np.inf if stressed > 0 else 0)
            else:
                val = ((base - stressed) / base * 100) if base != 0 else (np.inf if stressed < 0 else 0)
            stressed_metrics[f'degradation_{key}_percent'] = val
            degradation[key] = val
    
    if 'max_subgroup_delta_auc' in baseline_metrics and 'max_subgroup_delta_auc' in stressed_metrics:
        b_delta = baseline_metrics['max_subgroup_delta_auc']
        s_delta = stressed_metrics['max_subgroup_delta_auc']
        if not pd.isna(s_delta):
            deg = ((s_delta - b_delta) / b_delta * 100) if b_delta != 0 else (np.inf if s_delta > 0 else 0)
            stressed_metrics['degradation_max_subgroup_delta_auc_percent'] = deg
    return stressed_metrics

def check_threshold_violations(scenario_results_df: pd.DataFrame, critical_thresholds: dict, warn_thresholds: dict):
    critical_violations = []
    warn_violations = []

    for _, row in scenario_results_df.iterrows():
        scenario_name = row['scenario']

        # AUC
        if 'auc' in row and not pd.isna(row['auc']):
            if row['auc'] < critical_thresholds['min_auc']:
                critical_violations.append({'scenario': scenario_name, 'metric': 'AUC', 'value': row['auc'], 'threshold': critical_thresholds['min_auc'], 'type': 'CRITICAL_MIN_AUC'})
            elif row['auc'] < warn_thresholds['min_auc']:
                warn_violations.append({'scenario': scenario_name, 'metric': 'AUC', 'value': row['auc'], 'threshold': warn_thresholds['min_auc'], 'type': 'WARN_MIN_AUC'})

        # Degradation
        if 'degradation_auc_percent' in row and not pd.isna(row['degradation_auc_percent']):
            if row['degradation_auc_percent'] > critical_thresholds['max_degradation_auc_percent']:
                critical_violations.append({'scenario': scenario_name, 'metric': 'AUC Degradation (%)', 'value': row['degradation_auc_percent'], 'threshold': critical_thresholds['max_degradation_auc_percent'], 'type': 'CRITICAL_AUC_DEGRADATION'})
            elif row['degradation_auc_percent'] > warn_thresholds['max_degradation_auc_percent']:
                warn_violations.append({'scenario': scenario_name, 'metric': 'AUC Degradation (%)', 'value': row['degradation_auc_percent'], 'threshold': warn_thresholds['max_degradation_auc_percent'], 'type': 'WARN_AUC_DEGRADATION'})

        # Brier
        if 'brier_score' in row and not pd.isna(row['brier_score']):
            if row['brier_score'] > critical_thresholds['max_brier_score']:
                critical_violations.append({'scenario': scenario_name, 'metric': 'Brier Score', 'value': row['brier_score'], 'threshold': critical_thresholds['max_brier_score'], 'type': 'CRITICAL_MAX_BRIER'})
            elif row['brier_score'] > warn_thresholds['max_brier_score']:
                warn_violations.append({'scenario': scenario_name, 'metric': 'Brier Score', 'value': row['brier_score'], 'threshold': warn_thresholds['max_brier_score'], 'type': 'WARN_MAX_BRIER'})

        # Fairness
        if 'max_subgroup_delta_auc' in row and not pd.isna(row['max_subgroup_delta_auc']):
            if row['max_subgroup_delta_auc'] > critical_thresholds['max_subgroup_delta_auc']:
                critical_violations.append({'scenario': scenario_name, 'metric': 'Max Subgroup Delta AUC', 'value': row['max_subgroup_delta_auc'], 'threshold': critical_thresholds['max_subgroup_delta_auc'], 'type': 'CRITICAL_SUBGROUP_DELTA'})
            elif row['max_subgroup_delta_auc'] > warn_thresholds['max_subgroup_delta_auc']:
                warn_violations.append({'scenario': scenario_name, 'metric': 'Max Subgroup Delta AUC', 'value': row['max_subgroup_delta_auc'], 'threshold': warn_thresholds['max_subgroup_delta_auc'], 'type': 'WARN_SUBGROUP_DELTA'})

    return critical_violations, warn_violations

def make_go_no_go_decision(critical_violations: list, warn_violations: list) -> tuple[str, str]:
    if critical_violations:
        return "NO GO", "The model has critical performance degradations under stress. It is not approved for deployment and requires significant re-evaluation and improvement."
    elif warn_violations:
        return "GO WITH MITIGATION", "The model exhibits warning-level degradations under certain stress scenarios. It can proceed to deployment, but specific mitigation strategies must be implemented."
    else:
        return "GO", "The model demonstrates sufficient robustness under all tested stress scenarios. It is approved for deployment."

def plot_degradation_curves(scenario_results: pd.DataFrame, baseline_metrics: dict):
    metrics_to_plot = ['auc', 'brier_score']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        baseline_value = baseline_metrics[metric]
        
        filtered_results = scenario_results[scenario_results['scenario'] != 'Baseline']
        plot_labels = ['Baseline'] + filtered_results['scenario'].tolist()
        plot_values = [baseline_value] + filtered_results[metric].tolist()
        
        colors = ['skyblue'] + ['lightcoral'] * (len(plot_values) - 1)
        ax.bar(plot_labels, plot_values, color=colors)
        ax.axhline(y=baseline_value, color='gray', linestyle='--', label='Baseline')

        if metric == 'auc':
            ax.axhline(y=CRITICAL_THRESHOLDS['min_auc'], color='red', linestyle=':', label='Critical Min AUC')
            ax.axhline(y=WARN_THRESHOLDS['min_auc'], color='orange', linestyle=':', label='Warn Min AUC')
            ax.set_title('Model AUC Across Scenarios')
            ax.set_ylabel('AUC')
        else:
            ax.axhline(y=CRITICAL_THRESHOLDS['max_brier_score'], color='red', linestyle=':', label='Critical Max Brier')
            ax.axhline(y=WARN_THRESHOLDS['max_brier_score'], color='orange', linestyle=':', label='Warn Max Brier')
            ax.set_title('Model Brier Score Across Scenarios')
            ax.set_ylabel('Brier Score')

        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
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
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    
    zip_filename = os.path.join(out_dir, f'Session_06_{run_id}.zip')
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for name, path in artifacts.items():
            if os.path.exists(path):
                zipf.write(path, os.path.basename(path))
                manifest[name] = {
                    'path': os.path.basename(path),
                    'sha256': calculate_sha256(path)
                }
    
    manifest_path = os.path.join(out_dir, 'evidence_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    return manifest

# --- 3. Orchestration Function ---

def run_validation_pipeline(data_path: str, model_path: str, output_dir: str):
    """Main function to run the entire stress testing suite."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_dir, f'session06_{run_id}')
    os.makedirs(session_dir, exist_ok=True)
    
    # 1. Load Assets
    X_baseline, y_baseline, sens_attr_baseline, model = load_assets(
        data_path, model_path, FEATURE_COLS, TARGET_COL, SENSITIVE_ATTRIBUTE
    )
    
    # 2. Baseline Evaluation
    baseline_metrics = evaluate_model_performance(model, X_baseline, y_baseline, sens_attr_baseline, "Baseline")
    all_results = [baseline_metrics]
    scenario_config = {}

    # 3. Stress Scenarios
    # Scenario 1: Gaussian Noise
    noise_params = {'features': ['Age', 'Income', 'LoanAmount', 'CreditScore'], 'noise_std_multiplier': 0.5}
    X_noise = apply_gaussian_noise(X_baseline, **noise_params, random_state=RANDOM_SEED)
    X_noise_prep = preprocess_stressed_data(X_noise, FEATURE_COLS)
    m_noise = evaluate_model_performance(model, X_noise_prep, y_baseline, sens_attr_baseline, "Gaussian Noise (High)")
    all_results.append(calculate_degradation(baseline_metrics, m_noise))
    scenario_config['gaussian_noise_high'] = noise_params

    # Scenario 2: Economic Shift
    shift_params = {'features': ['Income', 'LoanAmount'], 'shift_factor': 0.8}
    X_shift = apply_feature_scaling_shift(X_baseline, **shift_params)
    X_shift_prep = preprocess_stressed_data(X_shift, FEATURE_COLS)
    m_shift = evaluate_model_performance(model, X_shift_prep, y_baseline, sens_attr_baseline, "Economic Shift (Income/Loan -20%)")
    all_results.append(calculate_degradation(baseline_metrics, m_shift))
    scenario_config['economic_shift_income_loan_down'] = shift_params

    # Scenario 3: Missingness
    miss_params = {'features': ['CreditScore', 'LoanDuration', 'Income'], 'missing_rate': 0.2}
    X_miss = apply_missingness_spike(X_baseline, **miss_params, random_state=RANDOM_SEED)
    X_miss_prep = preprocess_stressed_data(X_miss, FEATURE_COLS)
    m_miss = evaluate_model_performance(model, X_miss_prep, y_baseline, sens_attr_baseline, "Missingness Spike (20%)")
    all_results.append(calculate_degradation(baseline_metrics, m_miss))
    scenario_config['missingness_spike'] = miss_params

    # Scenario 4: Calibration Noise
    cal_noise_params = {'features': ['Age', 'Income', 'LoanAmount', 'CreditScore'], 'noise_std_multiplier': 0.7}
    X_cal_noise = apply_gaussian_noise(X_baseline, **cal_noise_params, random_state=RANDOM_SEED+1)
    X_cal_prep = preprocess_stressed_data(X_cal_noise, FEATURE_COLS)
    m_cal = evaluate_model_performance(model, X_cal_prep, y_baseline, sens_attr_baseline, "Calibration Stress (Noise High)")
    all_results.append(calculate_degradation(baseline_metrics, m_cal))
    scenario_config['calibration_noise_high'] = cal_noise_params

    # Subgroup Stress: Poor
    mask_poor = (sens_attr_baseline == 'Poor')
    m_poor = evaluate_model_performance(model, X_baseline[mask_poor], y_baseline[mask_poor], sens_attr_baseline[mask_poor], "Subgroup Stress (CreditScore Band: Poor)")
    all_results.append(calculate_degradation(baseline_metrics, m_poor))
    
    # Tail Slice: Low Income
    threshold_low_inc = X_baseline['Income'].quantile(0.1)
    mask_low_inc = (X_baseline['Income'] <= threshold_low_inc)
    m_tail_inc = evaluate_model_performance(model, X_baseline[mask_low_inc], y_baseline[mask_low_inc], sens_attr_baseline[mask_low_inc], "Tail Stress (Low Income)")
    all_results.append(calculate_degradation(baseline_metrics, m_tail_inc))

    # 4. Results Consolidation
    results_df = pd.DataFrame(all_results)
    crit_v, warn_v = check_threshold_violations(results_df, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
    decision, recommendation = make_go_no_go_decision(crit_v, warn_v)
    
    # 5. Visuals and Artifacts
    fig = plot_degradation_curves(results_df, baseline_metrics)
    plot_path = os.path.join(session_dir, 'degradation_curves.png')
    fig.savefig(plot_path)
    plt.close(fig)

    # Markdown Summary
    summary_path = os.path.join(session_dir, 'executive_summary.md')
    with open(summary_path, 'w') as f:
        f.write(f"# Executive Summary\n\nDecision: {decision}\n\nRecommendation: {recommendation}\n")
    
    # JSON Artifacts
    artifacts = {
        'baseline_metrics.json': os.path.join(session_dir, 'baseline_metrics.json'),
        'scenario_results.json': os.path.join(session_dir, 'scenario_results.json'),
        'degradation_curves.png': plot_path,
        'executive_summary.md': summary_path,
        'config_snapshot.json': os.path.join(session_dir, 'config_snapshot.json')
    }
    
    with open(artifacts['baseline_metrics.json'], 'w') as f: json.dump(baseline_metrics, f, indent=4)
    results_df.to_json(artifacts['scenario_results.json'], orient='records', indent=4)
    
    full_config = {
        'run_id': run_id, 'random_seed': RANDOM_SEED, 'features': FEATURE_COLS,
        'critical_thresholds': CRITICAL_THRESHOLDS, 'warn_thresholds': WARN_THRESHOLDS,
        'scenario_configurations': scenario_config
    }
    with open(artifacts['config_snapshot.json'], 'w') as f: json.dump(full_config, f, indent=4)

    manifest = export_artifacts(artifacts, output_dir, run_id)

    return {
        'results_df': results_df,
        'decision': decision,
        'recommendation': recommendation,
        'figure': fig,
        'manifest': manifest,
        'session_dir': session_dir
    }

# --- 4. Execution Block ---
if __name__ == "__main__":
    # Setup paths
    TEST_DATA = "sample_credit_test.csv"
    TEST_MODEL = "sample_model.pkl"
    REPORTS_DIR = "reports"
    
    set_global_seed(RANDOM_SEED)
    
    # Simulate initial setup
    df_synth = generate_synthetic_data(1000, RANDOM_SEED)
    df_synth.to_csv(TEST_DATA, index=False)
    train_and_save_model(df_synth, FEATURE_COLS, TARGET_COL, TEST_MODEL, RANDOM_SEED)
    
    # Run pipeline
    results = run_validation_pipeline(TEST_DATA, TEST_MODEL, REPORTS_DIR)
    print(f"Decision: {results['decision']}")
    print(f"Recommendation: {results['recommendation']}")