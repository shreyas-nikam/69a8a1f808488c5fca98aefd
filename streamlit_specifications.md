
# Streamlit Application Specification: NexusBank Credit Risk Stress-Testing Suite

## 1. Application Overview
The **NexusBank Credit Risk Stress-Testing Suite** is a functional validation platform designed for Model Validators (Alex) and ML Engineers. It operationalizes the "Validation Gate" required before model deployment. The application allows users to simulate real-world operational shifts—such as economic shocks, data corruption, and subgroup disparities—to determine if a model remains robust and safe.

### High-Level Story Flow
1.  **Environment Setup**: User uploads the Credit Risk Model (`.pkl`) and the clean validation dataset (`.csv`).
2.  **Baseline Lockdown**: The app calculates performance metrics and calibration on the "clean" data to establish a benchmark.
3.  **Stress Configuration**: Users interactively define deterministic stress scenarios (Noise, Scaling Shifts, Missingness) using sliders and feature selectors.
4.  **Robustness Evaluation**: The system executes scenarios, calculating degradation relative to the baseline and checking against institutional risk thresholds.
5.  **Granular Vulnerability Analysis**: A deep dive into specific sensitive subgroups and tail slices (extreme values) to identify hidden risks.
6.  **Audit & Decision**: An automated Go/No-Go recommendation is generated based on violation counts, and a cryptographically signed evidence bundle is packaged for download.

---

## 2. Code Requirements

### Import Statement
```python
from source import *
import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
```

### Session State Management
The following keys must be initialized in `st.session_state`:
*   `run_id`: Unique timestamp for the session (initialized via `datetime.now().strftime("%Y%m%d_%H%M%S")`).
*   `assets_loaded`: Boolean (False).
*   `X_baseline`, `y_baseline`, `sensitive_attr_baseline`, `model`: Objects returned by `load_assets`.
*   `baseline_metrics`: Dict returned by `evaluate_model_performance`.
*   `scenario_results`: List (starts with `baseline_metrics`).
*   `scenario_config`: Dict for capturing transformation parameters.
*   `final_decision`: String.
*   `final_recommendation`: String.
*   `output_dir`: String path using `RUN_ID`.

### UI Interaction to Function Mapping

| UI Component | Interaction | Function Call |
| :--- | :--- | :--- |
| **File Uploader** | On Model/CSV Upload | `load_assets(data_path, model_path, FEATURES, TARGET)` |
| **Calculate Baseline** | Button Click | `evaluate_model_performance(...)` |
| **Noise Slider/Selector** | "Run Noise Test" Button | `run_and_evaluate_scenario(...)` |
| **Shift Slider/Selector** | "Run Shift Test" Button | `run_and_evaluate_scenario(...)` |
| **Missingness Slider** | "Run Missingness Test" Button | `run_and_evaluate_scenario(...)` |
| **Subgroup Selector** | "Analyze Subgroup" Button | `evaluate_subgroup_stress(...)` |
| **Tail Slice Selector** | "Analyze Tail Slice" Button | `evaluate_tail_slice_stress(...)` |
| **Final Evaluation** | Auto-run/Button | `check_threshold_violations(...)` and `make_go_no_go_decision(...)` |
| **Export Bundle** | Button Click | `generate_evidence_artifacts(...)` |

---

## 3. Application Structure and Flow

### Header & Persona Context
*   **Title**: `NexusBank Model Robustness Validation Suite`
*   **Persona Information**: Display a sidebar bio for "Alex, Model Validator" focusing on the quest to ensure model reliability under economic shifts.

### Tab 1: Setup & Baseline
1.  **File Uploaders**:
    *   Model file (`.pkl`)
    *   Dataset file (`.csv`)
2.  **Logic**: Once uploaded, call `load_assets`.
3.  **Baseline Performance**:
    *   Display a "Lock Baseline" button.
    *   On click, run `evaluate_model_performance`.
    *   Display metrics (AUC, Accuracy, Precision, Recall, Brier Score) in `st.metric` columns.
4.  **Formulas**:
    ```python
    st.markdown(r"$$ BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2 $$")
    st.markdown(r"where $N$ is the number of samples, $f_i$ is the predicted probability for sample $i$, and $y_i$ is the actual outcome (0 or 1).")
    ```

### Tab 2: Deterministic Stress Testing
1.  **Global Seed Display**: Show `RANDOM_SEED = 42` as a locked constant.
2.  **Scenario Builders**:
    *   **Gaussian Noise**: 
        *   Multi-select: Features (Default: `['Age', 'Income', 'LoanAmount', 'CreditScore']`)
        *   Slider: `noise_std_multiplier` (Range 0.0 to 2.0).
        *   Action: Button "Inject Noise & Evaluate".
    *   **Feature Scaling Shift (Economic Shock)**:
        *   Multi-select: Features (Default: `['Income', 'LoanAmount']`)
        *   Slider: `shift_factor` (Range 0.5 to 1.5).
        *   Action: Button "Apply Shift & Evaluate".
    *   **Missingness Spike**:
        *   Multi-select: Features (Default: `['CreditScore', 'LoanDuration']`)
        *   Slider: `missing_rate` (Range 0.0 to 0.5).
        *   Action: Button "Introduce Missingness & Evaluate".
3.  **Real-time Degradation Tracking**:
    *   Use the formula:
    ```python
    st.markdown(r"$$ \text{Degradation} (\%) = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100 $$")
    st.markdown(r"where a positive value indicates a drop in performance from baseline.")
    ```
4.  **Output**: Display the running results table using `st.dataframe` or `display_results_table`.

### Tab 3: Vulnerability Drill-Down
1.  **Subgroup Stress**:
    *   Dropdown: Select from unique values in `sensitive_attr_baseline` (e.g., 'Poor', 'Fair').
    *   Action: Button "Analyze Subgroup Fairness".
    *   Display: `Max Subgroup Delta AUC` using:
    ```python
    st.markdown(r"$$ \text{Max Subgroup Delta AUC} = \max_{g \in \text{Groups}} |\text{AUC}_g - \text{AUC}_{\text{Overall}}| $$")
    st.markdown(r"where $g$ represents specific customer subgroups.")
    ```
2.  **Tail Slice Analysis**:
    *   Dropdown: Select Numeric Feature.
    *   Slider: Percentile (1-99).
    *   Radio: Slice Type ("top" or "bottom").
    *   Action: Button "Check Edge Case Performance".

### Tab 4: Validation Decision & Export
1.  **Threshold Visuals**:
    *   Call `plot_degradation_curves` and display the saved image.
    *   Show critical thresholds (AUC < 0.70) and warning thresholds (AUC < 0.75).
2.  **Violation Summary**:
    *   Run `check_threshold_violations`.
    *   Display color-coded cards: **Red** for Critical, **Orange** for Warning.
3.  **Final Recommendation**:
    *   Run `make_go_no_go_decision`.
    *   Display large header: `STATUS: [Decision]`.
4.  **Evidence Bundling**:
    *   Button: "Generate Audit-Ready Archive".
    *   On click, run `generate_evidence_artifacts`.
    *   **Integrity Hashing**:
    ```python
    st.markdown(f"Evidence integrity verified via SHA-256 manifest. Session ID: {st.session_state.run_id}")
    ```
    *   Provide `st.download_button` for the resulting `.zip` file.

---

## 4. Typography & Styling
*   **Sidebar Navigation**: Clear section headers for "1. Baseline", "2. Stressors", "3. Vulnerabilities", "4. Report".
*   **Status Indicators**: 
    *   `st.success` for PASS
    *   `st.warning` for WARN
    *   `st.error` for CRITICAL FAIL
*   **Monospace Logs**: Display internal scenario outputs in `st.code` blocks to simulate a "Validator's Log".

---

## 5. Safety & Error Handling
*   **File Alignment**: If user-uploaded CSV features do not match `FEATURES` list, display a warning and use `preprocess_stressed_data` to align columns.
*   **Empty Subgroups**: If a selected subgroup has insufficient samples for AUC computation, display `st.warning("Insufficient subgroup sample size for metrics.")`.
*   **State Locking**: Disable "Run Tests" buttons until "Calculate Baseline" has been executed.

