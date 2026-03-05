import pandas as pd
import numpy as np
import joblib
import json
import os
import shutil
import hashlib
import zipfile
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
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

# =============================================================================
# 1. CONSTANTS AND CONTRACTS
# =============================================================================

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
    'DependentCount': {'dtype': 'int', 'range': (0, 10)},
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

# =============================================================================
# 2. MARKDOWN EXPLANATIONS
# =============================================================================

EXPLANATIONS = {
    "introduction": """
# Validating AI Model Robustness at NexusBank: A Stress-Testing Suite

## Introduction: The Challenge at NexusBank

Alex, a diligent Model Validator at NexusBank, is tasked with ensuring the reliability and safety of all AI models before they are deployed to production. Her current mission: validate the new Credit Risk Model. This model plays a critical role in determining loan approvals and setting interest rates, directly impacting NexusBank's financial stability and its customers' lives.

The ML team has developed a high-performing model, but Alex knows that real-world conditions are rarely perfect. Economic shifts, data quality issues, or unexpected changes in borrower behavior can significantly degrade a model's performance, leading to erroneous decisions and potential financial losses or regulatory penalties.

Her key question is: *Does this model remain within acceptable performance bounds when conditions shift, inputs degrade, or the system encounters edge cases?*

This notebook guides Alex through a rigorous functional validation process, focusing on stress-testing the Credit Risk Model under various simulated operational challenges. She will define stress scenarios, deterministically apply transformations, quantify performance degradation, evaluate model calibration, and ultimately provide a data-driven go/no-go recommendation for the model's deployment.
""",
    "setup": """
## 1. Setting Up the Validation Environment

Before Alex can begin her stress tests, she needs to prepare her environment. This involves installing the necessary libraries and importing them, ensuring all tools are ready for the complex analytical tasks ahead. She also defines global constants like random seeds for reproducibility and output directory structures for organized evidence generation.
""",
    "baseline_intro": """
## 2. Establishing the Model's Baseline Performance

Alex's first step in validating the Credit Risk Model is to understand its inherent performance under ideal, expected operational conditions. This baseline evaluation will serve as the crucial reference point against which all stressed scenarios will be compared. Without a solid baseline, she cannot quantify any degradation caused by stress.
""",
    "brier_score_formula": r"""
The **Brier Score** quantifies the accuracy of probabilistic predictions. A perfect model has a Brier score of 0, while a score of 0.25 is achieved by random guessing in a balanced binary classification task. It's calculated as the mean squared difference between the predicted probabilities and the actual outcomes:
$$
BS = \frac{{1}}{{N}} \sum_{{i=1}}^{{N}} (f_i - y_i)^2
$$
where $N$ is the number of samples, $f_i$ is the predicted probability for sample $i$, and $y_i$ is the actual outcome (0 or 1).
""",
    "stress_scenarios_intro": """
## 3. Defining Realistic Stress Scenarios for Credit Risk

Alex understands that real-world data is rarely pristine. Economic downturns can affect income and credit scores, data entry errors can introduce noise, and system migrations can lead to missing values. She defines several stress scenarios that reflect these potential operational shifts, preparing to deterministically transform the test data.
""",
    "degradation_formula": r"""
## 4. Applying Stress and Measuring Performance Degradation

With the scenario transformation functions ready, Alex now applies these stresses to the test dataset. For each scenario, she measures the model's performance again and quantifies the degradation from the baseline. This directly answers how robust the model is to these anticipated real-world changes.

The **degradation percentage** for a metric is calculated as:
$$
\text{{Degradation}} (\%) = \frac{{\text{{Baseline Metric}} - \text{{Stressed Metric}}}}{{\text{{Baseline Metric}}}} \times 100
$$
where a positive value indicates a drop in performance from baseline (for metrics where higher is better, like AUC) or an increase (for metrics where lower is better, like Brier Score). This formula helps Alex quickly grasp the impact of stress.
""",
    "calibration_intro": """
## 5. Assessing Model Calibration Under Stress

Beyond overall performance, Alex needs to verify if the model's predicted probabilities remain accurate and trustworthy under stress. A well-calibrated model means that its predicted probabilities reflect the true likelihood of an event. For NexusBank, this is critical for setting appropriate loan loss reserves, where a 10% predicted default rate should truly mean 10% of such loans default. Degradation in calibration can lead to significant financial misestimations.
""",
    "subgroup_intro": r"""
## 6. Pinpointing Vulnerabilities in Subgroups and Edge Cases

Alex knows that a model might perform well overall but fail catastrophically for specific, vulnerable customer segments or extreme data points. This could lead to unfair or discriminatory outcomes, posing significant reputational and regulatory risks for NexusBank. She performs targeted stress tests on subgroups and 'tail slices' of the data.

### 6.1 Subgroup Stress Analysis

Alex checks if the model's performance varies unacceptably across different `credit_score_band` groups (e.g., 'Poor', 'Fair', 'Good', 'Excellent'). This directly addresses fairness concerns.

The **Subgroup Delta** (for AUC) is defined as:
$$
\text{{Max Subgroup Delta AUC}} = \max_{{g \in \text{{Groups}}}} |\text{{AUC}}_g - \text{{AUC}}_{{\text{{Overall}}}}|
$$
where $\text{{AUC}}_g$ is the AUC for a specific subgroup $g$, and $\text{{AUC}}_{{\text{{Overall}}}}$ is the AUC for the entire dataset. A large delta indicates a disparity in performance.
""",
    "tail_slice_intro": """
### 6.2 Tail Slice Stress Analysis

Alex investigates how the model performs on extreme values of a feature, e.g., customers with very low income (bottom 10th percentile) or very high loan amounts (top 10th percentile). These are edge cases that often reveal model weaknesses.
""",
    "decision_intro": """
## 7. Consolidating Results and Making a Go/No-Go Decision

After running all stress scenarios, Alex compiles all the performance metrics, degradation values, and checks them against NexusBank's predefined critical and warning thresholds. This is the culmination of her validation work, leading to a definitive go/no-go recommendation for the model's deployment.
""",
    "archive_intro": """
## 8. Archiving Validation Evidence for Audit Trail

To ensure regulatory compliance, provide an audit trail, and support future ModelOps activities, Alex must meticulously document and export all aspects of her validation. This includes configurations, raw results, summaries, and visualizations, bundled into a secure, hashed archive.
"""
}

# =============================================================================
# 3. PURE BUSINESS-LOGIC FUNCTIONS
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Sets the global random seed for numpy and random."""
    np.random.seed(seed)

def generate_synthetic_data(num_samples: int, seed: int) -> pd.DataFrame:
    """Generates synthetic credit risk data for demonstration."""
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
    df['credit_score_band'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, right=False)
    
    # Create target
    df[TARGET_COL] = ((df['CreditScore'] < 600) & (df['LoanAmount'] > 20000) | (df['Income'] < 30000)).astype(int)
    return df

def validate_dataset(df: pd.DataFrame, schema: dict) -> None:
    """Validates the dataset against the expected schema."""
    for col, expected in schema.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        
        # Basic type checking (simplified)
        if expected['dtype'] == 'int' and not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column {col} expected as numeric, got {df[col].dtype}")
        
        if 'range' in expected:
            low, high = expected['range']
            # We don't raise error for stressed data exceeding range, but we log if baseline fails
            pass

def train_baseline_model(df: pd.DataFrame, features: list, target: str, seed: int):
    """Trains a simple Logistic Regression model for baseline."""
    X = df[features]
    y = df[target]
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    model = LogisticRegression(random_state=seed, solver='liblinear')
    model.fit(X_scaled, y)
    
    return model, imputer, scaler

def load_assets(data_path: str, model_path: str, features: list, target: str, sensitive_attr_col: str):
    """Loads test data and pre-trained model."""
    df_test = pd.read_csv(data_path)
    model = joblib.load(model_path)

    X_test_raw = df_test[features]
    
    # Notebook logic: Refitting imputer and scaler for demo purposes
    local_imputer = SimpleImputer(strategy='mean')
    X_imputed_local = local_imputer.fit_transform(X_test_raw)

    local_scaler = StandardScaler()
    X_scaled_local = local_scaler.fit_transform(X_imputed_local)

    X_test_aligned = pd.DataFrame(X_scaled_local, columns=features)
    y_test = df_test[target]
    sensitive_attr = df_test[sensitive_attr_col]

    return X_test_aligned, y_test, sensitive_attr, model

def evaluate_model_performance(model, X, y, sensitive_attr=None, scenario_name="Baseline"):
    """Evaluates model performance metrics including subgroup analysis."""
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
        
        # Handle categorical safely
        if hasattr(sensitive_attr, 'cat'):
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
    """Applies Gaussian noise to specified features."""
    df_stressed = df.copy()
    rng = np.random.RandomState(random_state)
    for col in features_to_noise:
        if col in df_stressed.columns and pd.api.types.is_numeric_dtype(df_stressed[col]):
            std_dev = df_stressed[col].std()
            noise_level = std_dev * noise_std_multiplier
            noise = rng.normal(loc=0, scale=noise_level, size=len(df_stressed))
            df_stressed[col] = df_stressed[col] + noise
    return df_stressed

def apply_feature_scaling_shift(df, features_to_shift, shift_factor, random_state=RANDOM_SEED):
    """Applies a scaling shift to specified features."""
    df_stressed = df.copy()
    for col in features_to_shift:
        if col in df_stressed.columns and pd.api.types.is_numeric_dtype(df_stressed[col]):
            df_stressed[col] = df_stressed[col] * shift_factor
    return df_stressed

def apply_missingness_spike(df, features_to_spike, missing_rate, random_state=RANDOM_SEED):
    """Introduces missing values (NaN) to specified features."""
    df_stressed = df.copy()
    rng = np.random.RandomState(random_state)
    for col in features_to_spike:
        if col in df_stressed.columns:
            mask = rng.rand(len(df_stressed)) < missing_rate
            df_stressed.loc[mask, col] = np.nan
    return df_stressed

def preprocess_stressed_data(X_stressed, X_baseline_cols):
    """Ensures stressed data matches baseline feature alignment."""
    X_processed = X_stressed.copy()
    local_imputer = SimpleImputer(strategy='mean')
    
    for col in X_processed.columns:
        if X_processed[col].isnull().any() and pd.api.types.is_numeric_dtype(X_processed[col]):
            X_processed[col] = local_imputer.fit_transform(X_processed[[col]])

    local_scaler = StandardScaler()
    X_scaled = local_scaler.fit_transform(X_processed)
    X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)

    missing_cols = set(X_baseline_cols.columns) - set(X_processed.columns)
    for c in missing_cols:
        X_processed[c] = 0

    extra_cols = set(X_processed.columns) - set(X_baseline_cols.columns)
    if extra_cols:
        X_processed = X_processed.drop(columns=list(extra_cols))

    X_processed = X_processed[X_baseline_cols.columns]
    return X_processed

def run_and_evaluate_scenario(model, X_original, y_original, sensitive_attr_original,
                              scenario_name, transformation_func, transform_params,
                              X_baseline_cols, baseline_metrics, metrics_collector, 
                              config_snapshot, scenario_config_key):
    """Logic to run a single stress scenario and record degradation."""
    config_snapshot[scenario_config_key] = transform_params
    X_stressed_df = transformation_func(X_original.copy(), transform_params['features'], 
                                        **{k:v for k,v in transform_params.items() if k not in ['features']})
    X_stressed_processed = preprocess_stressed_data(X_stressed_df, X_baseline_cols)
    stressed_metrics = evaluate_model_performance(model, X_stressed_processed, y_original, sensitive_attr_original, scenario_name)
    metrics_collector.append(stressed_metrics)

    for metric_key in ['auc', 'accuracy', 'precision', 'recall', 'brier_score']:
        if metric_key in baseline_metrics and metric_key in stressed_metrics and not pd.isna(stressed_metrics[metric_key]):
            baseline_val = baseline_metrics[metric_key]
            stressed_val = stressed_metrics[metric_key]
            if metric_key == 'brier_score':
                deg = ((stressed_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else 0
            else:
                deg = ((baseline_val - stressed_val) / baseline_val) * 100 if baseline_val != 0 else 0
            stressed_metrics[f'degradation_{metric_key}_percent'] = deg

    if 'max_subgroup_delta_auc' in baseline_metrics and 'max_subgroup_delta_auc' in stressed_metrics:
        b_delta = baseline_metrics['max_subgroup_delta_auc']
        s_delta = stressed_metrics['max_subgroup_delta_auc']
        if not pd.isna(s_delta) and not pd.isna(b_delta):
            deg_delta = ((s_delta - b_delta) / b_delta) * 100 if b_delta != 0 else 0
            stressed_metrics['degradation_max_subgroup_delta_auc_percent'] = deg_delta

    return stressed_metrics

def evaluate_calibration_under_stress(model, X_original, y_original, sensitive_attr_original,
                                      scenario_name, transformation_func, transform_params,
                                      X_baseline_cols, baseline_metrics, metrics_collector, 
                                      config_snapshot, scenario_config_key):
    """Specific check for calibration metrics under stress."""
    config_snapshot[scenario_config_key] = transform_params
    X_stressed_df = transformation_func(X_original.copy(), transform_params['features'], 
                                        **{k:v for k,v in transform_params.items() if k not in ['features']})
    X_stressed_processed = preprocess_stressed_data(X_stressed_df, X_baseline_cols)
    stressed_metrics = evaluate_model_performance(model, X_stressed_processed, y_original, sensitive_attr_original, scenario_name)

    found = False
    for i, res in enumerate(metrics_collector):
        if res['scenario'] == scenario_name:
            metrics_collector[i] = stressed_metrics
            found = True
            break
    if not found:
        metrics_collector.append(stressed_metrics)

    baseline_brier = baseline_metrics['brier_score']
    stressed_brier = stressed_metrics['brier_score']
    brier_degradation_percent = ((stressed_brier - baseline_brier) / baseline_brier) * 100 if baseline_brier != 0 else 0
    stressed_metrics[f'degradation_brier_score_percent'] = brier_degradation_percent
    return stressed_metrics

def evaluate_subgroup_stress(model, X_original, y_original, sensitive_attr_original,
                             scenario_name, target_group, baseline_metrics,
                             metrics_collector, config_snapshot, scenario_config_key):
    """Evaluates performance on a single sensitive subgroup."""
    config_snapshot[scenario_config_key] = {'sensitive_attribute': SENSITIVE_ATTRIBUTE, 'target_group': target_group}
    group_mask = (sensitive_attr_original == target_group)
    if group_mask.sum() == 0:
        return None

    X_subgroup = X_original[group_mask]
    y_subgroup = y_original[group_mask]
    sensitive_attr_subgroup = sensitive_attr_original[group_mask]

    subgroup_metrics = evaluate_model_performance(model, X_subgroup, y_subgroup, sensitive_attr_subgroup, scenario_name)
    metrics_collector.append(subgroup_metrics)

    if 'auc' in subgroup_metrics and 'auc' in baseline_metrics:
        sub_auc = subgroup_metrics['auc']
        base_auc = baseline_metrics['auc']
        if not pd.isna(sub_auc) and not pd.isna(base_auc):
            subgroup_metrics[f'degradation_auc_percent'] = ((base_auc - sub_auc) / base_auc) * 100 if base_auc != 0 else 0
    return subgroup_metrics

def evaluate_tail_slice_stress(model, X_original, y_original, sensitive_attr_original,
                               scenario_name, feature, percentile, slice_type,
                               baseline_metrics, metrics_collector, config_snapshot, scenario_config_key):
    """Evaluates performance on tail segments of feature distributions."""
    config_snapshot[scenario_config_key] = {'feature': feature, 'percentile': percentile, 'slice_type': slice_type}
    if feature not in X_original.columns:
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
    metrics_collector.append(slice_metrics)

    if 'auc' in slice_metrics and 'auc' in baseline_metrics:
        s_auc = slice_metrics['auc']
        b_auc = baseline_metrics['auc']
        if not pd.isna(s_auc) and not pd.isna(b_auc):
            slice_metrics[f'degradation_auc_percent'] = ((b_auc - s_auc) / b_auc) * 100 if b_auc != 0 else 0
    return slice_metrics

def check_threshold_violations(scenario_results, critical_thresholds, warn_thresholds):
    """Compares scenario results against Go/No-Go thresholds."""
    critical_violations = []
    warn_violations = []

    for _, row in scenario_results.iterrows():
        name = row['scenario']
        # AUC
        if 'auc' in row and not pd.isna(row['auc']):
            if row['auc'] < critical_thresholds['min_auc']:
                critical_violations.append({'scenario': name, 'metric': 'AUC', 'value': row['auc'], 'threshold': critical_thresholds['min_auc'], 'type': 'CRITICAL_MIN_AUC'})
            elif row['auc'] < warn_thresholds['min_auc']:
                warn_violations.append({'scenario': name, 'metric': 'AUC', 'value': row['auc'], 'threshold': warn_thresholds['min_auc'], 'type': 'WARN_MIN_AUC'})

        # AUC Deg
        if 'degradation_auc_percent' in row and not pd.isna(row['degradation_auc_percent']):
            if row['degradation_auc_percent'] > critical_thresholds['max_degradation_auc_percent']:
                critical_violations.append({'scenario': name, 'metric': 'AUC Degradation (%)', 'value': row['degradation_auc_percent'], 'threshold': critical_thresholds['max_degradation_auc_percent'], 'type': 'CRITICAL_AUC_DEGRADATION'})
            elif row['degradation_auc_percent'] > warn_thresholds['max_degradation_auc_percent']:
                warn_violations.append({'scenario': name, 'metric': 'AUC Degradation (%)', 'value': row['degradation_auc_percent'], 'threshold': warn_thresholds['max_degradation_auc_percent'], 'type': 'WARN_AUC_DEGRADATION'})

        # Brier
        if 'brier_score' in row and not pd.isna(row['brier_score']):
            if row['brier_score'] > critical_thresholds['max_brier_score']:
                critical_violations.append({'scenario': name, 'metric': 'Brier Score', 'value': row['brier_score'], 'threshold': critical_thresholds['max_brier_score'], 'type': 'CRITICAL_MAX_BRIER'})
            elif row['brier_score'] > warn_thresholds['max_brier_score']:
                warn_violations.append({'scenario': name, 'metric': 'Brier Score', 'value': row['brier_score'], 'threshold': warn_thresholds['max_brier_score'], 'type': 'WARN_MAX_BRIER'})

        # Max Delta AUC
        if 'max_subgroup_delta_auc' in row and not pd.isna(row['max_subgroup_delta_auc']):
            if row['max_subgroup_delta_auc'] > critical_thresholds['max_subgroup_delta_auc']:
                critical_violations.append({'scenario': name, 'metric': 'Max Subgroup Delta AUC', 'value': row['max_subgroup_delta_auc'], 'threshold': critical_thresholds['max_subgroup_delta_auc'], 'type': 'CRITICAL_SUBGROUP_DELTA'})
            elif row['max_subgroup_delta_auc'] > warn_thresholds['max_subgroup_delta_auc']:
                warn_violations.append({'scenario': name, 'metric': 'Max Subgroup Delta AUC', 'value': row['max_subgroup_delta_auc'], 'threshold': warn_thresholds['max_subgroup_delta_auc'], 'type': 'WARN_SUBGROUP_DELTA'})

    return critical_violations, warn_violations

def make_go_no_go_decision(critical_violations, warn_violations):
    """Final decision logic."""
    if critical_violations:
        return "NO GO", "The model has critical performance degradations under stress. It is not approved for deployment and requires significant re-evaluation and improvement."
    elif warn_violations:
        return "GO WITH MITIGATION", "The model exhibits warning-level degradations under certain stress scenarios. It can proceed to deployment, but specific mitigation strategies must be implemented."
    return "GO", "The model demonstrates sufficient robustness under all tested stress scenarios. It is approved for deployment."

def plot_degradation_curves(scenario_results, baseline_metrics):
    """Generates a figure with performance plots."""
    metrics_to_plot = ['auc', 'brier_score']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        baseline_value = baseline_metrics[metric]
        scenario_df = scenario_results.set_index('scenario')
        if 'Baseline' in scenario_df.index:
            scenario_df = scenario_df.drop('Baseline')
        
        plot_values = [baseline_value] + scenario_df[metric].tolist()
        plot_labels = ['Baseline'] + scenario_df.index.tolist()
        colors = ['skyblue'] + ['lightcoral'] * (len(plot_labels) - 1)

        ax.bar(plot_labels, plot_values, color=colors)
        ax.axhline(y=baseline_value, color='gray', linestyle='--', label='Baseline')

        if metric == 'auc':
            ax.axhline(y=CRITICAL_THRESHOLDS['min_auc'], color='red', linestyle=':', label='Critical Min AUC')
            ax.axhline(y=WARN_THRESHOLDS['min_auc'], color='orange', linestyle=':', label='Warn Min AUC')
            ax.set_title(f'Model {metric.upper()} Across Scenarios')
            ax.set_ylabel(metric.upper())
            ax.set_ylim(min(plot_values) * 0.9, max(plot_values) * 1.1)
        elif metric == 'brier_score':
            ax.axhline(y=CRITICAL_THRESHOLDS['max_brier_score'], color='red', linestyle=':', label='Critical Max Brier')
            ax.axhline(y=WARN_THRESHOLDS['max_brier_score'], color='orange', linestyle=':', label='Warn Max Brier')
            ax.set_title(f'Model Brier Score Across Scenarios')
            ax.set_ylabel('Brier Score')
            ax.set_ylim(0, max(plot_values) * 1.2)

        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    return fig

def calculate_sha256(file_path):
    """Calculates the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    except FileNotFoundError:
        return "FILE_NOT_FOUND"
    return sha256_hash.hexdigest()

def generate_evidence_artifacts(
    base_output_dir,
    run_id,
    baseline_metrics,
    scenario_results_df,
    critical_violations,
    warn_violations,
    scenario_config,
    final_decision,
    final_recommendation,
    display_df,
    fig
):
    """Archives all validation results and metadata."""
    os.makedirs(base_output_dir, exist_ok=True)
    artifacts = {}

    # 1. baseline_metrics.json
    path = os.path.join(base_output_dir, 'baseline_metrics.json')
    with open(path, 'w') as f:
        json.dump(baseline_metrics, f, indent=4)
    artifacts['baseline_metrics.json'] = path

    # 2. scenario_results.json
    path = os.path.join(base_output_dir, 'scenario_results.json')
    scenario_results_df.to_json(path, orient='records', indent=4)
    artifacts['scenario_results.json'] = path

    # 3. violations_list.json
    path = os.path.join(base_output_dir, 'violations_list.json')
    with open(path, 'w') as f:
        json.dump({'critical_violations': critical_violations, 'warn_violations': warn_violations}, f, indent=4)
    artifacts['violations_list.json'] = path

    # 4. degradation_curves.png
    path = os.path.join(base_output_dir, 'degradation_curves.png')
    fig.savefig(path)
    artifacts['degradation_curves.png'] = path

    # 5. executive_summary.md
    path = os.path.join(base_output_dir, f'session06_{run_id}_executive_summary.md')
    with open(path, 'w') as f:
        f.write(f"# Model Validation Executive Summary - Session {run_id}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"## Final Go/No-Go Decision: **{final_decision}**\n")
        f.write(f"{final_recommendation}\n\n")
        f.write(f"## Scenario Results:\n")
        f.write(display_df.to_markdown(index=False))
    artifacts['session06_executive_summary.md'] = path

    # 6. config_snapshot.json
    path = os.path.join(base_output_dir, 'config_snapshot.json')
    config = {
        'run_id': run_id,
        'random_seed': RANDOM_SEED,
        'features': FEATURE_COLS,
        'target': TARGET_COL,
        'scenario_configurations': scenario_config
    }
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    artifacts['config_snapshot.json'] = path

    # 7. manifest
    manifest_data = {}
    for name, a_path in artifacts.items():
        manifest_data[name] = {'path': os.path.basename(a_path), 'sha256': calculate_sha256(a_path)}
    
    m_path = os.path.join(base_output_dir, 'evidence_manifest.json')
    with open(m_path, 'w') as f:
        json.dump(manifest_data, f, indent=4)
    artifacts['evidence_manifest.json'] = m_path

    # Zip
    zip_filename = os.path.join('reports', f'Session_06_{run_id}.zip')
    os.makedirs('reports', exist_ok=True)
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for name, a_path in artifacts.items():
            zipf.write(a_path, os.path.basename(a_path))
    
    return artifacts, zip_filename

# =============================================================================
# 4. MAIN ORCHESTRATOR
# =============================================================================

def run_validation_pipeline(num_samples=1000):
    """Executes the full functional validation workflow."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'reports/session06_{run_id}'
    set_global_seed(RANDOM_SEED)

    # 1. Setup Data & Model
    df_train = generate_synthetic_data(num_samples, RANDOM_SEED)
    validate_dataset(df_train, SCHEMA)
    model, imputer, scaler = train_baseline_model(df_train, FEATURE_COLS, TARGET_COL, RANDOM_SEED)
    
    model_path = 'sample_model.pkl'
    data_path = 'sample_credit_test.csv'
    joblib.dump(model, model_path)
    df_train.to_csv(data_path, index=False)

    # 2. Baseline
    X_baseline, y_baseline, sens_baseline, trained_model = load_assets(data_path, model_path, FEATURE_COLS, TARGET_COL, SENSITIVE_ATTRIBUTE)
    baseline_metrics = evaluate_model_performance(trained_model, X_baseline, y_baseline, sens_baseline, "Baseline")
    
    results_list = [baseline_metrics]
    scenario_config = {}

    # 3. Scenarios
    # Noise
    run_and_evaluate_scenario(trained_model, X_baseline, y_baseline, sens_baseline, "Gaussian Noise (High)",
                              apply_gaussian_noise, {'features': ['Age', 'Income', 'LoanAmount', 'CreditScore'], 'noise_std_multiplier': 0.5},
                              X_baseline, baseline_metrics, results_list, scenario_config, "noise_high")
    # Shift
    run_and_evaluate_scenario(trained_model, X_baseline, y_baseline, sens_baseline, "Economic Shift (-20%)",
                              apply_feature_scaling_shift, {'features': ['Income', 'LoanAmount'], 'shift_factor': 0.8},
                              X_baseline, baseline_metrics, results_list, scenario_config, "econ_shift")
    # Missingness
    run_and_evaluate_scenario(trained_model, X_baseline, y_baseline, sens_baseline, "Missingness (20%)",
                              apply_missingness_spike, {'features': ['CreditScore', 'LoanDuration', 'Income'], 'missing_rate': 0.2},
                              X_baseline, baseline_metrics, results_list, scenario_config, "missingness")
    # Calibration
    evaluate_calibration_under_stress(trained_model, X_baseline, y_baseline, sens_baseline, "Calibration Stress",
                                      apply_gaussian_noise, {'features': ['Age', 'Income', 'LoanAmount', 'CreditScore'], 'noise_std_multiplier': 0.7},
                                      X_baseline, baseline_metrics, results_list, scenario_config, "calib_stress")
    # Subgroups
    evaluate_subgroup_stress(trained_model, X_baseline, y_baseline, sens_baseline, "Subgroup (Poor)", 'Poor', 
                             baseline_metrics, results_list, scenario_config, "sub_poor")
    # Tail Slice
    evaluate_tail_slice_stress(trained_model, X_baseline, y_baseline, sens_baseline, "Tail (Low Income)", 'Income', 10, 'bottom',
                               baseline_metrics, results_list, scenario_config, "tail_income")

    # 4. Decisions
    results_df = pd.DataFrame(results_list)
    crit, warn = check_threshold_violations(results_df, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
    decision, recommendation = make_go_no_go_decision(crit, warn)

    # 5. Format for Table
    display_df = results_df[['scenario', 'num_samples', 'auc', 'degradation_auc_percent', 'brier_score', 'max_subgroup_delta_auc']].copy()
    display_df['Status'] = 'PASS'
    for c in crit: display_df.loc[display_df['scenario'] == c['scenario'], 'Status'] = 'CRITICAL FAIL'
    for w in warn: 
        if display_df.loc[display_df['scenario'] == w['scenario'], 'Status'].iloc[0] != 'CRITICAL FAIL':
            display_df.loc[display_df['scenario'] == w['scenario'], 'Status'] = 'WARN'

    # 6. Visualization
    fig = plot_degradation_curves(results_df, baseline_metrics)

    # 7. Artifacts
    artifacts, zip_path = generate_evidence_artifacts(base_dir, run_id, baseline_metrics, results_df, crit, warn, scenario_config, decision, recommendation, display_df, fig)

    return {
        'run_id': run_id,
        'results_df': results_df,
        'display_df': display_df,
        'critical_violations': crit,
        'warn_violations': warn,
        'decision': decision,
        'recommendation': recommendation,
        'fig': fig,
        'zip_path': zip_path,
        'artifacts': artifacts
    }

if __name__ == "__main__":
    # Internal test run
    output = run_validation_pipeline()
    print(f"Pipeline complete. Decision: {output['decision']}")
    print(f"Artifacts saved in {output['zip_path']}")