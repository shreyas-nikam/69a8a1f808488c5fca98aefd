import source
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import pytest
from streamlit.testing.v1 import AppTest
from sklearn.linear_model import LogisticRegression

# --- HELPERS ---

def create_valid_dataset(n_rows=20):
    """Creates a dataset matching source.FEATURE_COLS and requirements."""
    data = {}
    for col in source.FEATURE_COLS:
        data[col] = np.random.rand(n_rows) * 100
    data[source.TARGET_COL] = np.random.randint(0, 2, n_rows)
    # Ensure 'Poor' is present for drill-down tests
    data[source.SENSITIVE_ATTR] = ['Poor' if i % 5 == 0 else 'Good' for i in range(n_rows)]
    return pd.DataFrame(data)

def create_dummy_model(df):
    """Creates and saves a dummy logistic regression model."""
    model = LogisticRegression()
    X = df[source.FEATURE_COLS]
    y = df[source.TARGET_COL]
    model.fit(X, y)
    
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib").name
    joblib.dump(model, tmp_path)
    return tmp_path

def setup_app_with_assets(at):
    """Common setup to get past the file upload gate."""
    df = create_valid_dataset()
    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    df.to_csv(csv_path, index=False)
    
    model_path = create_dummy_model(df)
    
    at.sidebar.selectbox("Navigation").select("1. Setup & Assets").run()
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_path)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(model_path)
    at.button(label="Load and Validate").click().run()
    
    return df, model_path

# --- TESTS ---

def test_smoke_app_loads():
    """Requirement 1: App loads without error."""
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    assert at.title[0].value == "QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone"

def test_schema_drift_missing_column():
    """Requirement 2: Upload CSV missing a required feature column => app shows error."""
    at = AppTest.from_file("app.py").run()
    
    # Create data missing 'Age'
    df = create_valid_dataset()
    df_missing = df.drop(columns=[source.FEATURE_COLS[0]])
    
    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    df_missing.to_csv(csv_path, index=False)
    model_path = create_dummy_model(df) # Valid model anyway

    at.sidebar.selectbox("Navigation").select("1. Setup & Assets").run()
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_path)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(model_path)
    at.button(label="Load and Validate").click().run()

    # Check for error message defined in do_schema_validation
    assert any("Missing required columns" in err.value for err in at.error)
    assert at.session_state.data_loaded is False

def test_schema_drift_extra_column():
    """Requirement 2: Upload CSV with extra column => app shows warning but proceeds."""
    at = AppTest.from_file("app.py").run()
    
    df = create_valid_dataset()
    df['unnecessary_extra_column'] = 0
    
    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    df.to_csv(csv_path, index=False)
    model_path = create_dummy_model(df)

    at.sidebar.selectbox("Navigation").select("1. Setup & Assets").run()
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_path)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(model_path)
    at.button(label="Load and Validate").click().run()

    # App source says: st.warning(f"Extra columns detected and will be dropped: {extra}")
    assert any("Extra columns detected" in w.value for w in at.warning)
    assert at.session_state.data_loaded is True

def test_equivalence_baseline():
    """Requirement 3: Compare source path vs app path for baseline metrics."""
    at = AppTest.from_file("app.py").run()
    df_raw, model_path = setup_app_with_assets(at)
    
    # Navigate to baseline
    at.sidebar.selectbox("Navigation").select("2. Baseline Assessment").run()
    
    # Retrieve from app state
    app_metrics = at.session_state.baseline_metrics
    assert app_metrics is not None
    
    # Directly calculate via source
    model = joblib.load(model_path)
    ref_metrics = source.evaluate_model_performance(
        model, 
        df_raw[source.FEATURE_COLS], 
        df_raw[source.TARGET_COL], 
        df_raw[source.SENSITIVE_ATTR], 
        "Baseline"
    )
    
    # Compare (within float tolerance)
    assert pytest.approx(app_metrics['AUC']) == ref_metrics['AUC']
    assert app_metrics['Scenario'] == "Baseline"

def test_robustness_end_to_end():
    """Requirement 1: Default path runs end-to-end."""
    at = AppTest.from_file("app.py").run()
    setup_app_with_assets(at)
    
    # Step 2: Baseline
    at.sidebar.selectbox("Navigation").select("2. Baseline Assessment").run()
    assert "Baseline effectively calculated!" in at.success[0].value
    
    # Step 3: Stress Config (Keep defaults)
    at.sidebar.selectbox("Navigation").select("3. Stress Configuration").run()
    assert "Stress Configuration Saved." in at.success[0].value
    
    # Step 4: Robustness Evaluation
    at.sidebar.selectbox("Navigation").select("4. Robustness Evaluation").run()
    at.button(label="Run Evaluation").click().run()
    
    assert at.session_state.results_df is not None
    assert len(at.session_state.results_df) == 4 # Baseline + 3 Stress scenarios

def test_fallback_disclosure():
    """Requirement 4: App shows fallback logic when triggered."""
    at = AppTest.from_file("app.py").run()
    
    # In Step 6, if results_df is missing, we see a warning. 
    # But more specifically, if we can force an exception in evaluation.
    # Here we test the specific string for the manual evaluation fallback.
    at.sidebar.selectbox("Navigation").select("6. Audit & Export").run()
    assert "Robustness Evaluation has not been generated. Return to Step 4." in at.warning[0].value

def test_vulnerability_drilldown_poor_band():
    """Verify subgroup analysis for 'Poor' credit band."""
    at = AppTest.from_file("app.py").run()
    setup_app_with_assets(at)
    
    at.sidebar.selectbox("Navigation").select("5. Vulnerability Drill-down").run()
    
    # Look for the subgroup header
    assert any("Subgroup Vulnerability: 'Poor' Credit Band" in h.value for h in at.subheader)
    # Verification of evaluation logic rendering
    assert len(at.json) > 0

def test_export_manifest():
    """Requirement 5: Trigger export and verify manifest and audit bundle."""
    at = AppTest.from_file("app.py").run()
    setup_app_with_assets(at)
    
    # Must run robustness first to enable export
    at.sidebar.selectbox("Navigation").select("2. Baseline Assessment").run()
    at.sidebar.selectbox("Navigation").select("4. Robustness Evaluation").run()
    at.button(label="Run Evaluation").click().run()
    
    # Go to Audit & Export
    at.sidebar.selectbox("Navigation").select("6. Audit & Export").run()
    at.button(label="Generate Evidence Bundle").click().run()
    
    # Check for success message and download button
    assert any("Artifacts properly compiled" in s.value for s in at.success)
    
    # Verify the download button exists with the correct label
    dl_btn = at.download_button(label="Download Cryptographic Evidence Bundle")
    assert dl_btn is not None
    
    # Verify run_summary content in session state
    summary = at.session_state.run_summary
    assert "Baseline" in summary
    assert "Robustness" in summary

def test_missing_target_fails():
    """Requirement 2: Missing target column should hard fail."""
    at = AppTest.from_file("app.py").run()
    
    df = create_valid_dataset()
    df_no_target = df.drop(columns=[source.TARGET_COL])
    
    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    df_no_target.to_csv(csv_path, index=False)
    model_path = create_dummy_model(df)

    at.sidebar.selectbox("Navigation").select("1. Setup & Assets").run()
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_path)
    at.file_uploader(label="Upload Trained Model (PKL/JOBLIB)").upload(model_path)
    at.button(label="Load and Validate").click().run()

    # Based on app.py: missing = [col for col in expected if col not in df.columns]
    assert any("Missing required columns" in err.value for err in at.error)
    assert source.TARGET_COL in at.error[0].value
    assert at.session_state.data_loaded is False