
# Streamlit Application Specification: NexusBank Credit Risk Stress-Testing Suite

## 1. Application Overview
The application operationalizes functional validation for the NexusBank Credit Risk Model. It allows Model Validators (Alex) and ML Engineers to systematically evaluate how a trained model behaves under operational shifts, data quality issues, and edge-case scenarios. 

### Story Flow
1.  **Environment Setup**: User uploads the model artifact (`.pkl`) and the test dataset (`.csv`).
2.  **Baseline Assessment**: The app calculates and locks the "clean data" performance metrics to serve as the benchmark.
3.  **Stress Configuration**: User interactively configures deterministic stress scenarios (Gaussian noise, economic shifts, missingness spikes) using sliders.
4.  **Robustness Evaluation**: The model is evaluated against these scenarios, calculating performance and calibration degradation.
5.  **Vulnerability Drill-down**: The validator inspects performance specifically for sensitive subgroups (e.g., Credit Score Bands) and tail slices (e.g., lowest income percentile).
6.  **Audit & Recommendation**: The system automatically flags threshold violations and generates a Go/No-Go decision along with a cryptographic evidence bundle.

---

## 2. Code Requirements

### Import Statement
```python
from source import *
import streamlit as st
import pandas as pd
import os
import joblib
from datetime import datetime
```

### UI Interaction to Function Mapping

| UI Component | Function from `source.py` | Purpose |
| :--- | :--- | :--- |
| File Uploaders | `load_assets` | Load model and test CSV, align features. |
| Baseline Button | `evaluate_model_performance` | Calculate initial metrics on clean data. |
| Noise/Shift/Missing Sliders | `apply_gaussian_noise`, `apply_feature_scaling_shift`, `apply_missingness_spike` | Transform data based on user input. |
| Run Test Button | `preprocess_stressed_data`, `evaluate_model_performance` | Pipeline the transformations and evaluate. |
| Subgroup Select | `evaluate_subgroup_stress` | Targeted evaluation on sensitive attributes. |
| Tail Slice Select | `evaluate_tail_slice_stress` | Targeted evaluation on extreme data percentiles. |
| Threshold Check | `check_threshold_violations`, `make_go_no_go_decision` | Automated status determination. |
| Results Visualization | `display_results_table`, `plot_degradation_curves` | UI rendering of tables and charts. |
| Export Button | `generate_evidence_artifacts` | Package artifacts into a hashed ZIP. |

### Session State Design

| Key | Initial Value | Updated When |
| :--- | :--- | :--- |
| `data_loaded` | `False` | Successfully calling `load_assets`. |
| `baseline_metrics` | `None` | Clicking "Compute Baseline". |
| `all_scenario_results` | `[]` | Every time a stress scenario is executed. |
| `scenario_config` | `{}` | Parameters chosen in sliders are saved before run. |
| `model_assets` | `None` | Dictionary containing model, X_baseline, y_baseline, etc. |
| `run_id` | `RUN_ID` (from source) | App initialization. |
| `decision_status` | `None` | After running "Check Violations". |

---

## 3. Application Structure

### Sidebar Navigation
*   **Navigation**: Step 1: Data Setup, Step 2: Baseline, Step 3: Stress Testing, Step 4: Vulnerability Analysis, Step 5: Decision & Export.
*   **Constants Display**: Show `CRITICAL_THRESHOLDS` and `WARN_THRESHOLDS` in an expander.

### Tab 1: Data Setup & Assets
*   **Markdown**: "## 1. Validation Environment Setup"
*   **Widgets**:
    *   `st.file_uploader` for `sample_model.pkl`.
    *   `st.file_uploader` for `sample_credit_test.csv`.
*   **Logic**: 
    *   On upload: Save files locally to temporary paths.
    *   Call `load_assets(data_path, model_path, FEATURES, TARGET)`.
    *   Store results in `st.session_state.model_assets`.
    *   Display a preview of the loaded dataframe (`head`).

### Tab 2: Baseline Assessment
*   **Markdown**: "## 2. Model Baseline Performance"
*   **Formula**:
    ```python
    st.markdown(r"$$ BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2 $$")
    ```
    ```python
    st.markdown(r"where $N$ is the sample size, $f_i$ is the predicted probability, and $y_i$ is the actual outcome.")
    ```
*   **Logic**:
    *   Button: "Run Baseline Evaluation".
    *   Call `evaluate_model_performance` using baseline data from session state.
    *   Store in `st.session_state.baseline_metrics`.
    *   Display metrics (AUC, Accuracy, Brier Score) using `st.metric`.

### Tab 3: Stress Configuration & Evaluation
*   **Markdown**: "## 3. Robustness Evaluation under Stress"
*   **Interactive Inputs (Divided into columns/expanders)**:
    *   **Gaussian Noise**: Multi-select `FEATURES`, Slider for `noise_std_multiplier` (0.0 - 2.0).
    *   **Economic Shift**: Multi-select `FEATURES`, Slider for `shift_factor` (0.5 - 1.5).
    *   **Missingness**: Multi-select `FEATURES`, Slider for `missing_rate` (0.0 - 0.5).
*   **Logic**:
    *   Button: "Execute Stress Scenarios".
    *   For each active scenario, call the relevant `apply_*` function, then `preprocess_stressed_data`, then `evaluate_model_performance`.
    *   Calculate degradation using the formula:
    ```python
    st.markdown(r"$$ \text{Degradation} (\%) = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100 $$")
    ```
    ```python
    st.markdown(r"where a positive value represents performance loss for AUC and accuracy.")
    ```
    *   Append results to `st.session_state.all_scenario_results`.

### Tab 4: Vulnerability Analysis
*   **Subgroup Stress**:
    *   Dropdown: Select `SENSITIVE_ATTRIBUTE` (e.g., 'Poor', 'Fair').
    *   Call `evaluate_subgroup_stress`.
*   **Tail Slice Stress**:
    *   Dropdown: Select Feature from `FEATURES`.
    *   Slider: Percentile (1-99).
    *   Radio: 'top' vs 'bottom'.
    *   Call `evaluate_tail_slice_stress`.
*   **Logic**: Update `st.session_state.all_scenario_results` with these specific slices.

### Tab 5: Final Decision & Archive
*   **Markdown**: "## 4. Audit Trail and Go/No-Go Recommendation"
*   **Logic**:
    *   Button: "Generate Final Decision".
    *   Call `check_threshold_violations` with `all_scenario_results`.
    *   Call `make_go_no_go_decision`.
    *   Display decision in `st.success` (GO), `st.warning` (WARN), or `st.error` (NO GO).
    *   Call `display_results_table` to show the final audit dataframe.
    *   Call `plot_degradation_curves` and display the saved image using `st.image`.
*   **Export Section**:
    *   Button: "Bundle Evidence Package".
    *   Call `generate_evidence_artifacts`.
    *   Provide a `st.download_button` for the generated `.zip` file.

---

## 4. Determinism & Integrity
*   **Seed Control**: All operations must use `RANDOM_SEED = 42` to ensure the "Model Validator" persona can reproduce results across sessions.
*   **Hashing**: Use `calculate_sha256` to log the integrity of every CSV and PKL artifact included in the final report.

---

## 5. Typography & UI Style
*   Primary Font: Sans-serif (Inter).
*   Sidebar Width: 300px.
*   Status Indicators: Use colored tags for `PASS`, `WARN`, and `CRITICAL FAIL` in the results table.
*   Accessibility: Ensure charts generated by `plot_degradation_curves` are rendered with high resolution.

