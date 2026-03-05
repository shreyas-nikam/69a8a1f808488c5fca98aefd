
# Specification: Robustness & Functional Validation Stress-Testing Suite

## 0. One-paragraph summary
The **Robustness & Functional Validation Stress-Testing Suite** is an enterprise-grade application designed for Model Validators, ML Engineers, and QA Leads to stress-test predictive models (specifically the NexusBank Credit Risk Model) against operational shifts. The app enables users to upload models and datasets, establish a performance baseline, and then apply deterministic transformations—such as Gaussian noise, feature scaling shifts, and missingness spikes—to quantify model degradation. By evaluating calibration (Brier Score) and fairness (subgroup AUC deltas) against strict "Go/No-Go" thresholds, the suite automates the production readiness sign-off and generates a cryptographically hashed evidence bundle for auditability.

## 1. Functional Equivalence Contract
### 1.1 What must stay identical to the notebook
- **Data Source**: Primary support for `sample_credit_test.csv` (Credit Risk).
- **Target Column**: Must be `true_label`.
- **Feature Set**: `['Age', 'Income', 'LoanAmount', 'CreditScore', 'LoanDuration', 'DependentCount']`.
- **Logic**: All transformations (Noise, Shift, Missingness) must use the `RandomState` logic from `source.py` to ensure identical results to the notebook.
- **Metrics**: AUC, Accuracy, Precision, Recall, and Brier Score calculated using `sklearn` via `source.py`.
- **Calibration**: Brier Score loss is the primary metric for model confidence under stress.
- **Thresholds**: `CRITICAL_THRESHOLDS` and `WARN_THRESHOLDS` defined in `source.py` are the single source of truth for decisioning.

### 1.2 Forbidden changes
- Do not add new features or rename existing ones (e.g., do not rename `Income` to `Revenue`).
- Do not change the Brier Score calculation to ECE (Expected Calibration Error) unless the notebook explicitly provides the code.
- Do not change the `RANDOM_SEED = 42`.

## 2. Data Contract & Validation
### 2.1 Canonical schema (Credit Risk Context)
| Column Name | Role | dtype | Allowed Range | Required |
| :--- | :--- | :--- | :--- | :--- |
| `Age` | Feature | int | 20 - 70 | Y |
| `Income` | Feature | float | 0 - 200,000 | Y |
| `LoanAmount` | Feature | float | 0 - 100,000 | Y |
| `CreditScore` | Feature | int | 300 - 850 | Y |
| `LoanDuration` | Feature | int | 12 - 60 | Y |
| `DependentCount` | Feature | int | 0 - 5 | Y |
| `credit_score_band`| Sensitive | category | Poor, Fair, Good, Excellent | Y |
| `true_label` | Target | int | [0, 1] | Y |

### 2.2 Validation behavior
- **Missing Required Columns**: Hard fail with `st.error` listing the missing columns.
- **Extra Columns**: Allowed, but will be dropped during the alignment step in `preprocess_stressed_data`.
- **Dtype Coercion**: Numeric columns will be forced to `float64` for processing. `credit_score_band` must be categorical.
- **Missing Values**: Handled by `SimpleImputer(strategy='mean')` within the `source.py` logic.

### 2.3 Upload handling
- **Formats**: `.csv` for data, `.pkl` or `.joblib` for model.
- **Preview**: Show the first 5 rows of the uploaded dataframe and a schema summary (dtypes and missing counts).

## 3. UX / IA: Pages, Layout, and State Machine
### 3.1 Information architecture
Navigation via Sidebar Selectbox:
1.  **Step 1: Setup & Assets**: Model/Data upload and schema validation.
2.  **Step 2: Baseline Assessment**: Computation of metrics on clean data.
3.  **Step 3: Stress Testing**: Interactive configuration of noise, shifts, and missingness scenarios.
4.  **Step 4: Vulnerability Analysis**: Subgroup and Tail-slice drill-downs.
5.  **Step 5: Decision & Export**: Go/No-Go status and ZIP artifact generation.

### 3.2 Workflow gates & resets
- **Gate 1**: "Baseline Assessment" is locked until valid Data and Model are uploaded in Step 1.
- **Gate 2**: "Stress Testing" is locked until Baseline is computed.
- **Reset Logic**: Uploading a new CSV or Model clears all session state except for the `run_id`.

### 3.3 Loading states
- `st.spinner("Computing Baseline Metrics...")` during Step 2.
- `st.spinner("Applying Stress Scenarios and Evaluating...")` during Step 3.
- `st.spinner("Generating Audit Manifest and Zipping...")` during Step 5.

## 4. Formula Handling
- Model Calibration (Brier Score):
  ```python
  st.markdown(r"$$ BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2 $$")
  ```
- Performance Degradation:
  ```python
  st.markdown(r"$$ \text{Degradation \%} = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100 $$")
  ```

## 5. App Architecture
### 5.1 Separation of concerns
- `source.py`: Pure logic. Functions like `apply_gaussian_noise` and `evaluate_model_performance` take arguments and return dataframes/dicts.
- `app.py`: Handles file buffers, session state (`st.session_state.results_list`), and UI rendering.

### 5.2 Public functions imported from `source.py`
- `load_assets(data_path, model_path, ...)`
- `evaluate_model_performance(model, X, y, ...)`
- `run_and_evaluate_scenario(model, X_original, y_original, ...)`
- `check_threshold_violations(scenario_results_df, ...)`
- `make_go_no_go_decision(critical_violations, warn_violations)`
- `plot_degradation_curves(scenario_results, ...)`
- `export_artifacts(...)`

### 5.3 Determinism & caching
- **Seed**: `source.set_global_seed(42)` called on app startup.
- **Caching**:
    - `@st.cache_resource` for the loaded Model object.
    - `@st.cache_data` for the initial Baseline dataframe.
- **Session State Keys**:
    - `run_id`: Created once per session.
    - `baseline_metrics`: Stores dict of baseline performance.
    - `results_list`: List of dicts for all evaluated scenarios.
    - `data_loaded`: Boolean flag.

## 6. Robustness, Fallbacks, and Auditability
### 6.1 Failure modes checklist
- **Invalid Model**: Catch `joblib` loading errors (e.g., model trained with different sklearn version).
- **Empty Subgroups**: If a stress test targets a subgroup with 0 samples, show a warning and skip calculation for that group.
- **Hashing Failures**: Catch `FileNotFoundError` in `calculate_sha256` and report "HASH_FAILED" in manifest.

### 6.2 Fallback policy
- If the user does not provide a model, the app offers a button to **"Generate Synthetic Data & Model"** using `source.generate_synthetic_data()`.

### 6.3 Export manifest
The final `Session_06_<run_id>.zip` must include:
- `baseline_metrics.json`: Performance on clean data.
- `scenario_results.json`: Full metrics for all stress scenarios.
- `violations_list.json`: Any metric exceeding critical/warning levels.
- `config_snapshot.json`: Parameters used (e.g., `noise_std_multiplier=0.5`).
- `degradation_curves.png`: Visual evidence.
- `evidence_manifest.json`: SHA-256 hashes of all the above.

---

## Detailed Page Breakdown

### Page 1: Setup & Assets
- **Storyline**: Start the validation journey by providing the model artifact and the test dataset. This establishes the target system for the stress suite.
- **Interaction**:
    - Select Use Case: "NexusBank Credit Risk" (Default).
    - Upload `.csv` (Data) and `.pkl` (Model).
    - Validation Report: A table showing column alignment between the CSV and `FEATURE_COLS`.
- **Logic**: Calls `source.load_assets()`.

### Page 2: Baseline Assessment
- **Storyline**: Before breaking the model, we must know how it performs under optimal conditions. This page computes the "Ground Truth" metrics.
- **Math Context**: Explain the Brier Score as a measure of probability calibration.
- **Visualization**: Metric cards for AUC, Accuracy, and Brier Score.
- **Logic**: Calls `source.evaluate_model_performance()` with `scenario_name="Baseline"`.

### Page 3: Robustness Evaluation (Stress Testing)
- **Storyline**: Here we simulate "Real-World Drift." What happens if the sensors (data inputs) get noisy? Or if the economy shifts (scaling)?
- **Interaction**:
    - **Noise Scenario**: Multi-select features + Slider for `noise_std_multiplier`.
    - **Shift Scenario**: Multi-select features + Slider for `shift_factor`.
    - **Missingness Scenario**: Multi-select features + Slider for `missing_rate`.
    - Button: "Execute Scenarios".
- **Logic**: Loops through selected scenarios using `source.run_and_evaluate_scenario()`.

### Page 4: Vulnerability Analysis
- **Storyline**: Models often fail on specific slices or sensitive groups even if global metrics look fine.
- **Interaction**:
    - Select Subgroup (e.g., `credit_score_band == 'Poor'`).
    - Select Tail Slice (e.g., Bottom 10% of `Income`).
- **Logic**: Calls `source.evaluate_subgroup_stress()` and `source.evaluate_tail_slice_stress()`.

### Page 5: Final Decision & Archive
- **Storyline**: The final validation gate. The app compares all stressed results against the `CRITICAL_THRESHOLDS`.
- **UI Elements**:
    - **Decision Banner**: Large "GO" (Green), "GO WITH MITIGATION" (Orange), or "NO GO" (Red).
    - **Degradation Plot**: Bar charts comparing scenarios vs baseline.
    - **Download Button**: Triggers `source.export_artifacts()` and provides the ZIP.
- **Logic**: Calls `source.check_threshold_violations()` and `source.make_go_no_go_decision()`.

