import source
import pandas as pd
import numpy as np
import joblib
import io
import os
import pytest
from streamlit.testing.v1 import AppTest

# =============================================================================
# HELPERS
# =============================================================================

def create_valid_csv():
    """Creates a valid CSV based on source.SCHEMA."""
    data = {}
    for col, dtype in source.SCHEMA.items():
        if dtype == "float64":
            data[col] = np.random.randn(10)
        elif dtype == "int64":
            data[col] = np.random.randint(0, 100, 10)
        else:
            data[col] = ["A"] * 10
    
    # Ensure sensitive attribute and target are present as per source vars
    if source.TARGET_COL not in data:
        data[source.TARGET_COL] = np.random.randint(0, 2, 10)
    if source.SENSITIVE_ATTRIBUTE not in data:
        data[source.SENSITIVE_ATTRIBUTE] = np.random.choice(['Rich', 'Poor'], 10)
        
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf, df

class MockModel:
    """Mock model that implements predict_proba for AppTest compatibility."""
    def __init__(self):
        self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        # Return dummy probabilities
        return np.array([[0.5, 0.5]] * len(X))
    def predict(self, X):
        return np.zeros(len(X))

def create_mock_model_pkl():
    model = MockModel()
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf

# =============================================================================
# SMOKE TESTS
# =============================================================================

def test_smoke_app_loads():
    """App loads without error and displays the title."""
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    assert "QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone" in at.title[0].value

# =============================================================================
# SCHEMA DRIFT TESTS
# =============================================================================

def test_schema_drift_missing_column():
    """Upload a CSV missing a required feature column => app shows error."""
    at = AppTest.from_file("app.py").run()
    
    # Create CSV missing one required feature
    df = pd.DataFrame(np.random.randn(10, 2), columns=[source.FEATURE_COLS[0], "RandomCol"])
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_buf)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(create_mock_model_pkl())
    at.run()
    
    # Check for failure message in the specific error block
    assert any("Schema Validation Failed" in err.value for err in at.error)

def test_schema_drift_extra_column():
    """Upload a CSV with an extra column => app shows warning but allows progress."""
    at = AppTest.from_file("app.py").run()
    
    csv_buf, df = create_valid_csv()
    df["Unexpected_Column"] = 1.0
    csv_buf_extra = io.BytesIO()
    df.to_csv(csv_buf_extra, index=False)
    csv_buf_extra.seek(0)
    
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_buf_extra)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(create_mock_model_pkl())
    at.run()
    
    assert any("Extra columns detected" in warn.value for warn in at.warning)
    assert any("Schema Check: PASS" in succ.value for succ in at.success)

# =============================================================================
# EQUIVALENCE TESTS
# =============================================================================

def test_equivalence_data_processing():
    """Verify source.load_assets logic matches app-processed session state."""
    at = AppTest.from_file("app.py").run()
    
    csv_buf, df = create_valid_csv()
    model_buf = create_mock_model_pkl()
    
    # Setup files in app
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_buf)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(model_buf)
    at.run()
    
    # Trigger loading
    at.button(label="Load and Validate Assets").click().run()
    
    # Direct reference using source functions
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/test_ref_data.csv", "wb") as f: f.write(csv_buf.getvalue())
    with open("tmp/test_ref_model.pkl", "wb") as f: f.write(model_buf.getvalue())
    
    ref_X, ref_y, ref_sens, ref_model = source.load_assets(
        "tmp/test_ref_data.csv", "tmp/test_ref_model.pkl", 
        source.FEATURE_COLS, source.TARGET_COL, source.SENSITIVE_ATTRIBUTE
    )
    
    # Compare Session State vs Reference
    assert at.session_state.X_baseline.shape == ref_X.shape
    assert list(at.session_state.X_baseline.columns) == list(ref_X.columns)
    pd.testing.assert_frame_equal(at.session_state.X_baseline, ref_X)

# =============================================================================
# FALLBACK DISCLOSURE TESTS
# =============================================================================

def test_fallback_missing_sensitive_attribute():
    """Simulate missing sensitive attribute: App must show warning and disable subgroup analysis."""
    at = AppTest.from_file("app.py").run()
    
    # Create data without the specific sensitive attribute defined in source
    df = pd.DataFrame(np.random.randn(10, len(source.FEATURE_COLS)), columns=source.FEATURE_COLS)
    df[source.TARGET_COL] = np.random.randint(0, 2, 10)
    # Note: sensitive column missing
    
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_buf)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(create_mock_model_pkl())
    at.run()
    
    # Note: This might trigger schema validation error if SCHEMA requires it. 
    # But if it passes validation but fails detection in load_assets:
    at.button(label="Load and Validate Assets").click().run()
    
    # Navigate to Vulnerability Analysis
    at.sidebar.selectbox(label="Go to").select("5. Vulnerability Analysis").run()
    
    # App code: if st.session_state.sens_baseline is None or st.session_state.sens_baseline.empty:
    assert any("Sensitive Attribute not found. Subgroup analysis disabled." in warn.value for warn in at.warning)

# =============================================================================
# EXPORT MANIFEST & E2E FLOW
# =============================================================================

def test_full_pipeline_to_export():
    """Verify E2E flow from upload to Evidence Bundle generation."""
    at = AppTest.from_file("app.py").run()
    
    # 1. Setup
    csv_buf, _ = create_valid_csv()
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_buf)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(create_mock_model_pkl())
    at.run()
    at.button(label="Load and Validate Assets").click().run()
    
    # 2. Baseline
    at.sidebar.selectbox(label="Go to").select("2. Baseline Assessment").run()
    at.button(label="Calculate Baseline Metrics").click().run()
    assert at.session_state.baseline_metrics is not None
    
    # 3. Stress Configuration (Skip modification, use defaults)
    at.sidebar.selectbox(label="Go to").select("3. Stress Configuration").run()
    
    # 4. Robustness Evaluation
    at.sidebar.selectbox(label="Go to").select("4. Robustness Evaluation").run()
    at.button(label="Run Stress Scenarios").click().run()
    assert at.session_state.scenarios_run is True
    
    # 5. Vulnerability Analysis
    at.sidebar.selectbox(label="Go to").select("5. Vulnerability Analysis").run()
    at.button(label="Run Vulnerability Analysis").click().run()
    assert at.session_state.vulnerability_run is True
    
    # 6. Final Decision & Export
    at.sidebar.selectbox(label="Go to").select("6. Final Decision & Archive").run()
    at.button(label="Generate Final Decision & Evidence Bundle").click().run()
    
    # Check for decision text
    assert "Decision:" in at.session_state.final_decision or "final_decision" in at.session_state
    
    # Verify Download Button exists
    assert at.download_button(label="Download Evidence Bundle (ZIP)") is not None

def test_stress_param_reset_logic():
    """Verify that changing stress parameters resets downstream results as per reset_results()."""
    at = AppTest.from_file("app.py").run()
    
    # Load assets and baseline first
    csv_buf, _ = create_valid_csv()
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_buf)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(create_mock_model_pkl())
    at.button(label="Load and Validate Assets").click().run()
    
    at.sidebar.selectbox(label="Go to").select("2. Baseline Assessment").run()
    at.button(label="Calculate Baseline Metrics").click().run()
    
    # Go to configuration
    at.sidebar.selectbox(label="Go to").select("3. Stress Configuration").run()
    
    # Manually set scenarios_run to True to simulate previous execution
    at.session_state.scenarios_run = True
    
    # Change a slider (e.g., Noise STD Multiplier)
    at.slider(key="s_noise").set_value(1.2).run()
    
    # Verify scenarios_run was reset to False
    assert at.session_state.scenarios_run is False
    assert any("Settings updated. Downstream results cleared." in t.value for t in at.toast)

def test_critical_threshold_logic():
    """Verify threshold values match source ground truth in the sidebar."""
    at = AppTest.from_file("app.py").run()
    
    # Check sidebar threshold text matches source CRITICAL_THRESHOLDS
    sidebar_text = "".join([m.value for m in at.sidebar.markdown])
    assert str(source.CRITICAL_THRESHOLDS['min_auc']) in sidebar_text
    assert str(source.CRITICAL_THRESHOLDS['max_brier_score']) in sidebar_text
    assert "Critical Min AUC" in sidebar_text
    assert "Critical Max Brier" in sidebar_text