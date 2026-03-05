

# Streamlit App Specification: NexusBank Model Robustness Stress-Testing Suite

## 0. One-paragraph summary
This application provides a rigorous stress-testing environment for the NexusBank Credit Risk Model to validate its robustness against operational shifts and data quality issues. Designed for Model Validators and ML Engineers, the app enables users to establish a performance baseline, apply deterministic stress scenarios (Gaussian noise, economic shifts, missingness spikes), and perform granular vulnerability analysis on sensitive subgroups. The workflow culminates in an automated "Go/No-Go" decision based on pre-defined regulatory thresholds and generates a complete, hashed evidence bundle for auditability.

## 1. Functional Equivalence Contract

### 1.1 What must stay identical to the notebook
- **Data Source**: Primary test dataset `sample_credit_test.csv` and the trained model artifact `.pkl`.
- **Feature/Target Definitions**: Target is `true_label`. Features are exactly `['Age', 'Income', 'LoanAmount', 'CreditScore', 'LoanDuration', 'DependentCount']`.
- **Computation Logic**: The exact transformation functions (`apply_gaussian_noise`, `apply_feature_scaling_shift`, `apply_missingness_spike`) and metric calculations (AUC, Brier Score, etc.).
- **Thresholds**: `CRITICAL_THRESHOLDS` and `WARN_THRESHOLDS` must be used for decision logic.
- **Randomness**: `RANDOM_SEED = 42` must be used for all stochastic transformations to ensure deterministic results.

### 1.2 Forbidden changes
- No dynamic re-binning of `credit_score_band`.
- No modification of `CRITICAL_THRESHOLDS` values in the UI (must remain read-only/hardcoded).
- No substitution of the Brier Score calculation with different calibration metrics.
- No dropping of features beyond the alignment logic provided in `preprocess_stressed_data`.

## 2. Data Contract & Validation

### 2.1 Canonical schema
Derived from `source.py` SCHEMA constant.

| Column Name | Role | dtype | unit | Allowed Range | Required |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Age | Feature | int | Years | [20, 70] | Y |
| Income | Feature | float | USD | [0, 200,000] | Y |
| LoanAmount | Feature | float | USD | [0, 100,000] | Y |
| CreditScore | Feature | int | Points | [300, 850] | Y |
| LoanDuration | Feature | int | Months | [12, 60] | Y |
| DependentCount | Feature | int | Count | [0, 10] | Y |
| credit_score_band | Sensitive Attribute | category | - | Poor, Fair, Good, Excellent | Y |
| true_label | Target | int | - | [0, 1] | Y |

### 2.2 Validation behavior
- **Missing Required Columns**: Hard fail with a red error box listing specific missing columns.
- **Extra Columns**: Warn the user and proceed; columns will be dropped during alignment in `preprocess_stressed_data`.
- **Dtype Coercion**: Numeric features must be numeric. If strings are passed for numeric columns, attempt coercion; if failed, HARD FAIL.
- **Missing Values**: Handled explicitly by `SimpleImputer` (mean strategy) inside `preprocess_stressed_data`.

### 2.3 Upload handling
- **Accepted Formats**: `.csv` (Data), `.pkl` or `.joblib` (Model).
- **Size Limits**: 50MB for CSV.
- **UI Report**: Post-upload, show a `st.dataframe` preview (first 5 rows) and a schema check status (Pass/Fail).

## 3. UX / IA: Pages, Layout, and State Machine

### 3.1 Information architecture
- **Sidebar**: Logo, App Title, Navigation (Selectbox), and Threshold Summary.
- **Page 1: Setup & Assets**: File uploaders for model and dataset. Story context introduction.
- **Page 2: Baseline Assessment**: Compute metrics on raw data. Establish the reference point.
- **Page 3: Stress Configuration**: Interactive sliders for noise levels, shift factors, and missingness rates.
- **Page 4: Robustness Evaluation**: Run scenarios and view real-time degradation metrics.
- **Page 5: Vulnerability Analysis**: Subgroup delta analysis and tail-slice performance.
- **Page 6: Final Decision & Archive**: Executive summary, Go/No-Go status, and ZIP download.

### 3.2 Workflow gates & resets
- **Baseline Lock**: Evaluation cannot proceed to Stress Evaluation until Baseline metrics are computed and stored.
- **Parameter Change**: If the user changes noise/shift parameters on Page 3, the "Results" on Page 4 and 5 must be cleared to prevent stale metrics.
- **Export Gate**: ZIP generation only enabled after a minimum of 3 stress scenarios have been run.

### 3.3 Loading states
- `with st.spinner("Calculating Baseline Metrics...")`
- `with st.spinner("Applying Stress Scenarios...")`
- `with st.spinner("Generating Evidence Bundle...")`

## 4. Application Overview
This application follows the workflow of **Alex, a Model Validator at NexusBank**. The story begins with Alex preparing a "Validation Gate" for a new Credit Risk Model. The user acts as Alex, ensuring that the model doesn't just work on historical data, but survives "Economic Shocks" (income shifts) or "Data Quality Failures" (noise/missingness). The goal is to produce a hashed evidence bundle that justifies a deployment decision to the risk committee.

## 5. App Architecture

### 5.1 Separation of concerns
- `source.py` manages all `numpy`, `sklearn`, and file-hashing logic.
- `app.py` manages `st.session_state`, widget layouts, and plot rendering.

### 5.2 Public functions imported from `source.py`
- `load_assets(data_path, model_path, features, target, sensitive_attr_col)`
- `evaluate_model_performance(model, X, y, sensitive_attr, scenario_name)`
- `run_and_evaluate_scenario(...)`
- `evaluate_calibration_under_stress(...)`
- `evaluate_subgroup_stress(...)`
- `evaluate_tail_slice_stress(...)`
- `check_threshold_violations(results_df, crit_thresh, warn_thresh)`
- `make_go_no_go_decision(crit, warn)`
- `plot_degradation_curves(scenario_results, baseline_metrics)`
- `generate_evidence_artifacts(...)`

### 5.3 Determinism & caching
- All `numpy` operations use `np.random.RandomState(42)`.
- `@st.cache_resource` used for the initial model/data load.
- `@st.cache_data` used for baseline metric computation.

## 6. Robustness, Fallbacks, and Auditability

### 6.1 Failure modes checklist
- **Incompatible Model**: If `.pkl` contains a non-sklearn model, show error: "Model must support predict_proba and be scikit-learn compatible."
- **Empty Subgroups**: If a subgroup (e.g., 'Poor') has 0 samples, skip and warn in the log.
- **Hashing Fail**: If an artifact cannot be written, provide a UI warning but do not crash.

### 6.2 Fallback policy
- If `SENSITIVE_ATTRIBUTE` is missing from the CSV, skip Page 5 (Vulnerability Analysis) and display a yellow banner: "Sensitive Attribute not found. Subgroup analysis disabled."

### 6.3 Export manifest
The manifest (JSON) will include:
- `input_schema_hash`
- `scenario_config_snapshot`
- `sha256_hashes` for all 7 required artifacts.

## 7. Code Requirements

### 7.1 Initialization
```python
from source import *
import streamlit as st

# Initialize session state
st.session_state.setdefault("baseline_metrics", None)
st.session_state.setdefault("results_list", [])
st.session_state.setdefault("data_loaded", False)
```

### 7.2 Page 1: Setup & Assets
- **Markdown**: Use `EXPLANATIONS["introduction"]` and `EXPLANATIONS["setup"]`.
- **Widgets**:
    - `st.file_uploader` for CSV.
    - `st.file_uploader` for PKL.
- **Action**: Call `load_assets()` and store `X_baseline`, `y_baseline`, `trained_model` in state.

### 7.3 Page 2: Baseline Assessment
- **Markdown**: Use `EXPLANATIONS["baseline_intro"]`.
- **Formula**:
```python
st.markdown(r"""$$
BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2
$$""")
st.markdown(r"where $N$ is the number of samples, $f_i$ is the predicted probability, and $y_i$ is the actual outcome.")
```
- **Action**: Call `evaluate_model_performance(...)`. Display results in `st.metric` columns.

### 7.4 Page 3: Stress Configuration
- **Markdown**: Use `EXPLANATIONS["stress_scenarios_intro"]`.
- **Widgets**:
    - `st.multiselect` for `features_to_noise`.
    - `st.slider` for `noise_std_multiplier` (0.0 to 2.0).
    - `st.slider` for `shift_factor` (0.5 to 1.5).
    - `st.slider` for `missing_rate` (0.0 to 0.5).

### 7.5 Page 4: Robustness Evaluation
- **Markdown**: Use `EXPLANATIONS["degradation_formula"]`.
- **Formula**:
```python
st.markdown(r"""$$
\text{Degradation} (\%) = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100
$$""")
```
- **Action**: Call `run_and_evaluate_scenario` for Noise, Shift, and Missingness.
- **Display**: Use `st.dataframe` with color-coded rows based on Status (PASS/WARN/CRITICAL).

### 7.6 Page 5: Vulnerability Analysis
- **Markdown**: Use `EXPLANATIONS["subgroup_intro"]`.
- **Formula**:
```python
st.markdown(r"""$$
\text{Max Subgroup Delta AUC} = \max_{g \in \text{Groups}} |\text{AUC}_g - \text{AUC}_{\text{Overall}}|
$$""")
```
- **Action**: Call `evaluate_subgroup_stress` and `evaluate_tail_slice_stress`.

### 7.7 Page 6: Final Decision & Archive
- **Markdown**: Use `EXPLANATIONS["decision_intro"]` and `EXPLANATIONS["archive_intro"]`.
- **Plotting**: Call `plot_degradation_curves` and display using `st.pyplot(fig)`.
- **Logic**: Display `st.success` or `st.error` based on `make_go_no_go_decision`.
- **Export**: Call `generate_evidence_artifacts` and provide `st.download_button` for the `.zip` file.

---
**Note**: All logic gates check `st.session_state` to ensure the user cannot skip to Page 6 without completing the data upload and baseline assessment.
