
# 0. One-paragraph summary
This application is a professional-grade **ML Stress-Testing Suite** designed for Model Validators, ML Engineers, and QA Leads to evaluate the robustness of predictive models (e.g., NexusBank Credit Risk) against operational shifts. Users upload a trained model and test dataset to establish a performance baseline, then interactively configure deterministic stress scenarios—including Gaussian noise, feature scaling shifts, and missingness spikes. The app calculates performance degradation across key metrics like AUC and Brier Score, evaluates subgroup fairness, and automates a **Go/No-Go deployment decision** based on rigorous threshold violations. All results are cryptographically hashed and bundled into an audit-ready evidence manifest.

# 1. Functional Equivalence Contract
## 1.1 What must stay identical to the notebook
- **Data Logic**: The exact synthetic data generation parameters (Seed 42, Mean/STD for Income and LoanAmount) must be preserved.
- **Metric Definitions**: AUC, Accuracy, Precision, Recall, and Brier Score calculated via `sklearn.metrics`.
- **Transformation Logic**: Deterministic `RandomState(42)` application for noise and missingness to ensure reproducibility.
- **Decision Logic**: The two-tier (Warning/Critical) thresholding system for AUC, AUC Degradation, and Brier Score.
- **Preprocessing Alignment**: Handling missing columns by zero-filling and extra columns by dropping, as defined in `preprocess_stressed_data`.

## 1.2 Forbidden changes
- No substitution of Brier Score with other calibration metrics (e.g., ECE) unless both are provided.
- No alteration of the `CRITICAL_THRESHOLDS` or `WARN_THRESHOLDS` constants.
- No renaming of the `true_label` target or the specific feature set used in the NexusBank model.

# 2. Data Contract & Validation
## 2.1 Canonical schema (NexusBank Model)
| Column Name | Role | dtype | unit | allowed range | required |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Age | Feature | int | Years | [20, 70] | Y |
| Income | Feature | float | USD | [0, inf) | Y |
| LoanAmount | Feature | float | USD | [0, inf) | Y |
| CreditScore | Feature | int | Points | [300, 850] | Y |
| LoanDuration | Feature | int | Months | [12, 60] | Y |
| DependentCount | Feature | int | Count | [0, 5] | Y |
| true_label | Target | int | Boolean | {0, 1} | Y |
| credit_score_band | Sensitive Attribute | category | Band | Poor, Fair, Good, Excellent | Y |

## 2.2 Validation behavior
- **Missing Required Columns**: Hard fail with a list of missing columns.
- **Extra Columns**: Automatically dropped during the `preprocess_stressed_data` step (Warn in UI).
- **Dtype Coercion**: Force numeric conversion for feature columns; categorical for `credit_score_band`.
- **Missing Values**: Handled via `SimpleImputer(strategy='mean')` within the logic functions.

## 2.3 Upload handling
- **Accepted formats**: `.csv` for datasets; `.pkl` or `.joblib` for models.
- **UI Report**: Post-upload, display a dataframe preview and a "Schema Check" status (Green/Red).

# 3. UX / IA: Pages, Layout, and State Machine
## 3.1 Information architecture
Navigation via Sidebar Selectbox:
1.  **Setup & Assets**: File upload for `.csv` and `.pkl`. Automated baseline calculation upon successful load.
2.  **Baseline Assessment**: Detailed breakdown of performance on clean data, including subgroup AUCs and Brier Scores.
3.  **Stress Configuration**: Interactive UI to select features for Noise, Scaling (Shift), and Missingness.
4.  **Robustness Evaluation**: Batch execution of all scenarios, rendering degradation charts and the main results table.
5.  **Vulnerability Drill-down**: Specific analysis of the "Poor" credit band and "Low Income" tail slices.
6.  **Audit & Export**: Go/No-Go status dashboard and generation of the SHA-256 evidence bundle.

## 3.2 Workflow gates & resets
- **Asset Reset**: Uploading a new model or dataset clears `st.session_state.results_df` and `st.session_state.baseline_metrics`.
- **Execution Gate**: Pages 4-6 are disabled until "Baseline" is successfully calculated on Page 2.
- **Export Gate**: Download only available after the "Run Evaluation" button has been triggered.

## 3.3 Loading states
- `st.spinner("Aligning features and calculating baseline...")` during data load.
- `st.spinner("Applying deterministic stress transformations...")` during scenario runs.
- `st.spinner("Hashing artifacts and creating ZIP archive...")` during export.

# 5. App Architecture
## 5.1 Separation of concerns
- `app.py` handles the navigation, parameter sliders, and rendering of `matplotlib` figures.
- `source.py` performs the actual `numpy` noise injection and `joblib` model inference.

## 5.2 Public functions imported from `source.py`
- `load_assets(data_path, model_path, features, target, sensitive_attr_col)`
- `evaluate_model_performance(model, X, y, sensitive_attr, scenario_name)`
- `apply_gaussian_noise(df, features, noise_std_multiplier, random_state)`
- `apply_feature_scaling_shift(df, features, shift_factor, random_state)`
- `apply_missingness_spike(df, features, missing_rate, random_state)`
- `preprocess_stressed_data(X_stressed, reference_cols)`
- `check_threshold_violations(results_df, crit_thresh, warn_thresh)`
- `export_artifacts(artifacts, out_dir, run_id)`

## 5.3 Determinism & caching
- **Seed**: Hardcoded `RANDOM_SEED = 42`.
- **Caching**: 
    - `@st.cache_resource` for `joblib.load(model_path)`.
    - `@st.cache_data` for the initial `pd.read_csv`.
- **Session State Keys**: 
    - `st.session_state.baseline_metrics`
    - `st.session_state.results_df`
    - `st.session_state.run_id` (initialized once with `datetime.now()`)

# 6. Robustness, Fallbacks, and Auditability
## 6.1 Failure modes checklist
- **Incompatible Model**: Model trained on 10 features, but CSV only provides 6. Logic must catch the `ValueError` and report feature mismatch.
- **Empty Subgroup**: If the uploaded data contains no "Poor" credit band members, the app must warn and skip subgroup metrics for that category rather than crashing.
- **File Permissions**: Catch errors when writing the `REPORTS_DIR`.

## 6.2 Fallback policy
- If `SENSITIVE_ATTRIBUTE` is missing, disable fairness/subgroup analysis but allow performance stress testing.
- Show `st.error` banners for schema violations.

## 6.3 Export manifest
The final `evidence_manifest.json` must include:
- `timestamp`: UTC execution time.
- `input_hash`: SHA-256 of the input CSV.
- `model_hash`: SHA-256 of the input PKL.
- `status`: Final Go/No-Go decision.
- `violations`: List of metrics that crossed critical/warning thresholds.

---

### Page-by-Page Storyline Specifications

#### Page 1: Setup & Assets
- **Content**: Detailed explanation of the "Functional Validation" gate. Explain that this is the primary mechanism for preventing "Production Surprises."
- **Interaction**: Two file uploaders. 
- **Validation**: Check columns against `SCHEMA`.

#### Page 2: Baseline Assessment
- **Content**: Introduction to the model performance baseline. Explain the Brier Score as a measure of calibration:
- **Formula**:
  st.markdown(r"$$ BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2 $$")
- **Logic**: Calls `evaluate_model_performance` on the raw data.

#### Page 3: Stress Configuration
- **Content**: Explain the three types of deterministic stress. 
  1. **Gaussian Noise**: Simulates sensor/input jitter.
  2. **Feature Scaling**: Simulates economic shifts (e.g., 20% drop in income).
  3. **Missingness**: Simulates data pipeline failures.
- **Interaction**: 
  - Sliders for `noise_std_multiplier`, `shift_factor`, `missing_rate`.
  - Multiselects for features to target per scenario.

#### Page 4: Robustness Evaluation
- **Content**: Comparison of Baseline vs. Stressed performance.
- **Visuals**: Replicate `plot_degradation_curves` showing the bar charts with Red (Critical) and Orange (Warn) horizontal dotted lines.
- **Table**: Display `results_df` with conditional formatting (Red text for violations).

#### Page 5: Vulnerability Analysis
- **Content**: Focus on "Subgroup Stress" and "Tail Stress."
- **Storyline**: Explain that global metrics often hide localized failures in sensitive populations or extreme cases (low income).

#### Page 6: Audit & Export
- **Content**: The final "Verdict."
- **Logic**: Uses `make_go_no_go_decision` to show a large "NO GO" (Red) or "GO" (Green) card.
- **Interaction**: "Generate Evidence Bundle" button triggers hashing and `export_artifacts`.
