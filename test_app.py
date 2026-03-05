import pytest
import pandas as pd
import numpy as np
import os
import joblib
import tempfile
import source
from streamlit.testing.v1 import AppTest
from sklearn.ensemble import RandomForestClassifier

# HELPER FUNCTIONS

def create_dummy_dataset(include_target=True, missing_col=None, extra_col=None):
    """Creates a dummy CSV matching the source.FEATURES schema."""
    cols = list(source.FEATURES)
    if include_target and hasattr(source, 'TARGET'):
        cols.append(source.TARGET)
    
    data = np.random.rand(10, len(cols))
    df = pd.DataFrame(data, columns=cols)
    
    if missing_col and missing_col in df.columns:
        df = df.drop(columns=[missing_col])
    
    if extra_col:
        df[extra_col] = 0
        
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name

def create_dummy_model():
    """Creates a simple sklearn model and saves to pkl."""
    X = np.random.rand(20, len(source.FEATURES))
    y = np.random.randint(0, 2, 20)
    model = RandomForestClassifier(n_estimators=2)
    model.fit(X, y)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    joblib.dump(model, tmp.name)
    return tmp.name

# TESTS

def test_smoke_app_loads():
    """Requirement 1: App loads without error."""
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    assert at.title[0].value == "QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone"

def test_smoke_e2e_path():
    """Requirement 1: Default path can run end-to-end using sample data."""
    at = AppTest.from_file("app.py").run()
    
    model_path = create_dummy_model()
    data_path = create_dummy_dataset()
    
    # Step 1: Data Setup
    with open(model_path, "rb") as f:
        at.file_uploader(label="Upload Model Artifact (.pkl)").upload(f)
    with open(data_path, "rb") as f:
        at.file_uploader(label="Upload Test Dataset (.csv)").upload(f)
    at.run()
    
    assert "Assets successfully loaded and validated." in at.success[0].value
    
    # Step 2: Baseline
    at.sidebar.selectbox("Navigation").select("Step 2: Baseline").run()
    at.button(label="Run Baseline Evaluation").click().run()
    assert "Baseline evaluation complete!" in at.success[0].value
    assert len(at.metric) > 0
    
    # Step 3: Stress Testing
    at.sidebar.selectbox("Navigation").select("Step 3: Stress Testing").run()
    # Select first available feature for Gaussian Noise
    at.multiselect(key="gn_f").select(source.FEATURES[0])
    at.button(label="Execute Stress Scenarios").click().run()
    assert "Stress scenarios executed successfully!" in at.success[0].value
    
    # Step 5: Decision
    at.sidebar.selectbox("Navigation").select("Step 5: Decision & Export").run()
    at.button(label="Generate Final Decision").click().run()
    assert any(x in at.session_state.decision_status for x in ["GO", "WARN", "NO GO"])

def test_schema_drift_missing_column():
    """Requirement 2: Missing feature column => app shows error."""
    at = AppTest.from_file("app.py").run()
    
    model_path = create_dummy_model()
    # Remove the first required feature
    missing_col = source.FEATURES[0]
    data_path = create_dummy_dataset(missing_col=missing_col)
    
    with open(model_path, "rb") as f:
        at.file_uploader(label="Upload Model Artifact (.pkl)").upload(f)
    with open(data_path, "rb") as f:
        at.file_uploader(label="Upload Test Dataset (.csv)").upload(f)
    at.run()
    
    assert "Schema Error: Missing required columns" in at.error[0].value
    assert missing_col in at.error[0].value

def test_schema_drift_extra_column():
    """Requirement 2: Extra column => app shows error."""
    at = AppTest.from_file("app.py").run()
    
    model_path = create_dummy_model()
    data_path = create_dummy_dataset(extra_col="UNWANTED_COL")
    
    with open(model_path, "rb") as f:
        at.file_uploader(label="Upload Model Artifact (.pkl)").upload(f)
    with open(data_path, "rb") as f:
        at.file_uploader(label="Upload Test Dataset (.csv)").upload(f)
    at.run()
    
    assert "Schema Error: Extra columns detected" in at.error[0].value
    assert "UNWANTED_COL" in at.error[0].value

def test_equivalence_baseline_metrics():
    """Requirement 3: Compare model predictions/metrics from source vs app path."""
    model_path = create_dummy_model()
    data_path = create_dummy_dataset()
    
    # 1. Direct Reference from Source
    ref_assets = source.load_assets(data_path, model_path)
    ref_metrics = source.evaluate_model_performance(
        ref_assets['model'], 
        ref_assets['X_baseline'], 
        ref_assets['y_baseline']
    )
    
    # 2. App Path
    at = AppTest.from_file("app.py").run()
    with open(model_path, "rb") as f:
        at.file_uploader(label="Upload Model Artifact (.pkl)").upload(f)
    with open(data_path, "rb") as f:
        at.file_uploader(label="Upload Test Dataset (.csv)").upload(f)
    at.run()
    
    at.sidebar.selectbox("Navigation").select("Step 2: Baseline").run()
    at.button(label="Run Baseline Evaluation").click().run()
    
    # Compare metrics displayed in app to reference
    # App metrics are rounded to 4 decimals in the code
    app_metrics_state = at.session_state.baseline_metrics
    for key in ref_metrics:
        assert np.isclose(ref_metrics[key], app_metrics_state[key], atol=1e-4)

def test_fallback_disclosure_api_key():
    """Requirement 4: App must show a warning banner if API key is missing."""
    at = AppTest.from_file("app.py").run()
    # The app shows a warning if api_key is empty
    at.sidebar.text_input(label="API Key (OpenAI/Gemini)").set_value("").run()
    assert "Please provide an API key if required by upstream services." in at.sidebar.warning[0].value

def test_export_manifest_contents():
    """Requirement 5: Verify export manifest contains required metadata."""
    at = AppTest.from_file("app.py").run()
    
    model_path = create_dummy_model()
    data_path = create_dummy_dataset()
    
    # Setup
    with open(model_path, "rb") as f:
        at.file_uploader(label="Upload Model Artifact (.pkl)").upload(f)
    with open(data_path, "rb") as f:
        at.file_uploader(label="Upload Test Dataset (.csv)").upload(f)
    at.run()
    
    # Populate data for export
    at.sidebar.selectbox("Navigation").select("Step 2: Baseline").run()
    at.button(label="Run Baseline Evaluation").click().run()
    
    at.sidebar.selectbox("Navigation").select("Step 3: Stress Testing").run()
    at.multiselect(key="gn_f").select(source.FEATURES[0])
    at.button(label="Execute Stress Scenarios").click().run()
    
    # Export
    at.sidebar.selectbox("Navigation").select("Step 5: Decision & Export").run()
    at.button(label="Bundle Evidence Package").click().run()
    
    # Verify manifest JSON display
    manifest = at.json[0].value
    assert "run_id" in manifest
    # If source.export_artifacts is used, it typically returns timestamps/hashes
    # Based on app.py, if 'export_artifacts' exists, it displays the returned manifest
    # Otherwise it displays a default dict with run_id and status
    if "status" in manifest:
        assert manifest["status"] in ["bundled", "success", "complete"]
    
    # Verify Download Button exists
    assert at.download_button[0].label == "Download Evidence Package"

def test_vulnerability_analysis_triggers():
    """Verifies that Vulnerability Analysis steps can be triggered."""
    at = AppTest.from_file("app.py").run()
    
    # Setup prerequisite data
    at.session_state.data_loaded = True
    at.session_state.baseline_metrics = {"auc": 0.8}
    at.session_state.model_assets = {
        'model': joblib.load(create_dummy_model()),
        'X_baseline': pd.DataFrame(np.random.rand(10, len(source.FEATURES)), columns=source.FEATURES),
        'y_baseline': pd.Series(np.random.randint(0, 2, 10))
    }
    
    at.sidebar.selectbox("Navigation").select("Step 4: Vulnerability Analysis").run()
    
    # Subgroup Stress
    at.button(label="Evaluate Subgroup Stress").click().run()
    assert "completed!" in at.success[0].value
    
    # Tail Slice Stress
    at.button(label="Evaluate Tail Slice Stress").click().run()
    assert "completed!" in at.success[1].value

def test_step_dependency_warnings():
    """Verifies that later steps warn if earlier steps aren't completed."""
    at = AppTest.from_file("app.py").run()
    
    # Step 2 without data
    at.sidebar.selectbox("Navigation").select("Step 2: Baseline").run()
    assert "Please complete Step 1: Data Setup first." in at.warning[0].value
    
    # Step 3 without baseline
    at.sidebar.selectbox("Navigation").select("Step 3: Stress Testing").run()
    assert "Please compute Baseline performance first (Step 2)." in at.warning[0].value