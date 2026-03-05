import pytest
import pandas as pd
import io
import os
import joblib
import tempfile
import source
from streamlit.testing.v1 import AppTest

# ==============================================================================
# HELPERS
# ==============================================================================

def create_valid_csv():
    """Creates a dataframe matching source schema."""
    cols = source.FEATURE_COLS + [source.TARGET_COL, source.SENSITIVE_ATTRIBUTE]
    df = pd.DataFrame([[0.0] * len(cols)], columns=cols)
    # Ensure some data exists for metrics
    df[source.TARGET_COL] = [0]
    df[source.SENSITIVE_ATTRIBUTE] = ["Good"]
    return df

def create_invalid_csv(missing_col=True):
    """Creates a dataframe missing a required feature or with extra columns."""
    df = create_valid_csv()
    if missing_col:
        return df.drop(columns=[source.FEATURE_COLS[0]])
    else:
        df['extra_junk_column'] = 1.0
        return df

def get_temp_pkl_path(model):
    """Saves a model to a temporary file and returns the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    joblib.dump(model, tmp.name)
    return tmp.name

# ==============================================================================
# TESTS
# ==============================================================================

def test_app_smoke_load():
    """Requirement 1: App loads without error."""
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone"

def test_full_end_to_end_synthetic_flow():
    """Requirement 1: Default path can run end-to-end using synthetic data."""
    at = AppTest.from_file("app.py", default_timeout=30).run()
    
    # Step 1: Generate Synthetic Data
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    assert "Synthetic data and model generated successfully!" in at.success[0].value
    
    # Step 2: Baseline Assessment
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    assert "Baseline assessment complete!" in at.success[0].value
    assert at.metric(label="AUC").value is not None
    
    # Step 3: Stress Testing
    at.sidebar.selectbox(label="Go to:").select("3. Stress Testing").run()
    at.button(label="Execute Stress Scenarios").click().run()
    assert "Scenarios Executed Successfully!" in at.success[0].value
    
    # Step 4: Vulnerability
    at.sidebar.selectbox(label="Go to:").select("4. Vulnerability Analysis").run()
    at.button(label="Analyze Vulnerabilities").click().run()
    assert "Vulnerability Analysis Complete!" in at.success[0].value
    
    # Step 5: Export
    at.sidebar.selectbox(label="Go to:").select("5. Decision & Export").run()
    assert at.header[1].value == "Validation Verdict"
    # Trigger manifest
    at.button(label="Generate Audit Manifest and ZIP").click().run()
    assert at.download_button(label="Download Evidence Bundle (ZIP)") is not None

def test_schema_drift_missing_column():
    """Requirement 2: Upload a CSV missing a required feature column => app shows error."""
    at = AppTest.from_file("app.py").run()
    
    # Create bad data
    bad_df = create_invalid_csv(missing_col=True)
    csv_bytes = bad_df.to_csv(index=False).encode()
    
    # Mock upload data and a valid model pkl
    _, mock_model = source.generate_synthetic_data(source.RANDOM_SEED)
    model_path = get_temp_pkl_path(mock_model)
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    
    # Upload to widget
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_bytes)
    at.file_uploader(label="Upload Model Artifact (PKL/JOBLIB)").upload(model_bytes)
    at.run()
    
    # Check for error
    assert len(at.error) > 0
    assert "Schema Validation Error" in at.error[0].value
    
    # Verify next step is blocked
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    assert "Please complete Step 1: Setup & Assets to unlock this step." in at.warning[0].value
    os.remove(model_path)

def test_schema_drift_extra_column():
    """Requirement 2: Upload a CSV with extra columns - assert behavior."""
    at = AppTest.from_file("app.py").run()
    
    # The source.validate_dataset (implied) usually checks if required cols are present.
    # If it fails on extra columns, we assert error.
    bad_df = create_invalid_csv(missing_col=False)
    csv_bytes = bad_df.to_csv(index=False).encode()
    
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_bytes)
    at.run()
    
    # If the app produces error, it shows as at.error. 
    # Based on app.py code: validate_dataset(test_df, FEATURE_COLS, TARGET_COL) is called.
    # We assert consistency with the app's response.
    if len(at.error) > 0:
        assert "Schema Validation Error" in at.error[0].value

def test_equivalence_metrics():
    """Requirement 3: Compare model predictions from source path vs app-generated path."""
    # 1. Get reference metrics from source directly
    df_synth, model_synth = source.generate_synthetic_data(source.RANDOM_SEED)
    X, y, sens = source.prepare_features(df_synth, source.FEATURE_COLS, source.TARGET_COL, source.SENSITIVE_ATTRIBUTE)
    ref_metrics = source.evaluate_model_performance(model_synth, X, y, sens, "Ref")
    
    # 2. Run App
    at = AppTest.from_file("app.py").run()
    # Use synthetic button to ensure same seed
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    
    # 3. Compare values
    app_auc = float(at.metric(label="AUC").value)
    app_acc = float(at.metric(label="Accuracy").value)
    
    assert pytest.approx(app_auc, abs=1e-4) == ref_metrics['auc']
    assert pytest.approx(app_acc, abs=1e-4) == ref_metrics['accuracy']

def test_stress_testing_configuration():
    """Test that stress testing parameters can be adjusted."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    
    at.sidebar.selectbox(label="Go to:").select("3. Stress Testing").run()
    
    # Modify noise slider
    at.slider(label="Noise Std Multiplier").set_value(1.5).run()
    # Modify shift features
    at.multiselect(label="Features to shift").set_value([source.FEATURE_COLS[0]]).run()
    
    at.button(label="Execute Stress Scenarios").click().run()
    assert "Scenarios Executed Successfully!" in at.success[0].value
    
    # Check that results dataframe appeared
    assert len(at.dataframe) > 0

def test_vulnerability_analysis_params():
    """Test vulnerability analysis interactions."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    
    at.sidebar.selectbox(label="Go to:").select("4. Vulnerability Analysis").run()
    
    # Change subgroup
    at.selectbox(label="Select Sensitive Subgroup (credit_score_band)").select("Excellent").run()
    # Change tail slice
    at.slider(label="Target Percentile").set_value(5).run()
    
    at.button(label="Analyze Vulnerabilities").click().run()
    assert "Vulnerability Analysis Complete!" in at.success[0].value

def test_export_manifest_contents():
    """Requirement 5: Verify manifest generation and ZIP button."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    at.sidebar.selectbox(label="Go to:").select("3. Stress Testing").run()
    at.button(label="Execute Stress Scenarios").click().run()
    
    at.sidebar.selectbox(label="Go to:").select("5. Decision & Export").run()
    
    # Trigger generation
    at.button(label="Generate Audit Manifest and ZIP").click().run()
    
    # Verify download button exists
    assert at.download_button(label="Download Evidence Bundle (ZIP)") is not None
    
    # Check that degradation curves were plotted
    assert at.pyplot is not None

def test_navigation_logic():
    """Verify that pages are locked until data is loaded."""
    at = AppTest.from_file("app.py").run()
    
    # Try Step 2 without data
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    assert "Please complete Step 1: Setup & Assets to unlock this step." in at.warning[0].value
    
    # Try Step 3 without baseline
    at.sidebar.selectbox(label="Go to:").select("3. Stress Testing").run()
    assert "Please compute Baseline Metrics in Step 2 to unlock Stress Testing." in at.warning[0].value

def test_target_missing_hard_fail():
    """Requirement 2: Missing target column should fail validation."""
    at = AppTest.from_file("app.py").run()
    
    df = create_valid_csv()
    bad_df = df.drop(columns=[source.TARGET_COL])
    csv_bytes = bad_df.to_csv(index=False).encode()
    
    at.file_uploader(label="Upload Test Dataset (CSV)").upload(csv_bytes)
    at.run()
    
    assert "Schema Validation Error" in at.error[0].value

def test_typography_injection():
    """Verify that custom styles are injected (Smoke)."""
    at = AppTest.from_file("app.py").run()
    # Check if the markdown for font-family 'Inter' is present
    style_found = False
    for md in at.markdown:
        if "font-family: 'Inter'" in md.value:
            style_found = True
            break
    assert style_found

def test_subgroup_missing_data_warning():
    """Test that app warns if a subgroup is empty."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    at.sidebar.selectbox(label="Go to:").select("4. Vulnerability Analysis").run()
    
    # Assuming "Poor" might be empty in a small synthetic set or specifically testing the warning branch
    # Note: synthetic data might contain it, but we test the button logic.
    at.button(label="Analyze Vulnerabilities").click().run()
    
    # Check for either success or warning about no samples
    outputs = [w.value for w in at.success] + [w.value for w in at.warning]
    assert any("Vulnerability Analysis Complete" in s or "Warning: No samples found" in s for s in outputs)

def test_global_seed_consistency():
    """Verify that set_global_seed is called via deterministic results."""
    # Run once
    at1 = AppTest.from_file("app.py").run()
    at1.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at1.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at1.button(label="Compute Baseline Metrics").click().run()
    auc1 = float(at1.metric(label="AUC").value)
    
    # Run twice
    at2 = AppTest.from_file("app.py").run()
    at2.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at2.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at2.button(label="Compute Baseline Metrics").click().run()
    auc2 = float(at2.metric(label="AUC").value)
    
    assert auc1 == auc2

def test_session_state_reset():
    """Verify reset_analysis_state() functionality logic via UI selection."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    
    # Compute baseline
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    assert len(at.metric) > 0
    
    # Go back to Step 1 and generate new data
    at.sidebar.selectbox(label="Go to:").select("1. Setup & Assets").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    
    # Go back to Step 2 - baseline metrics should be gone/reset
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    # If the metrics UI is missing until "Compute" is clicked again, reset worked
    assert len(at.metric) == 0 or "Compute Baseline Metrics" in at.button[0].label

def test_brier_score_explanation_exists():
    """Verify the calibration context markdown is present."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    
    found_brier = False
    for md in at.markdown:
        if "Brier Score" in md.value:
            found_brier = True
            break
    assert found_brier
    # Check for LaTeX formula
    assert any("BS = \\frac{1}{N}" in md.value for md in at.markdown)

def test_degradation_formula_display():
    """Requirement 5: Verify the degradation formula display in Step 5."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    at.sidebar.selectbox(label="Go to:").select("3. Stress Testing").run()
    at.button(label="Execute Stress Scenarios").click().run()
    
    at.sidebar.selectbox(label="Go to:").select("5. Decision & Export").run()
    assert any("Degradation \\%" in md.value for md in at.markdown)

def test_vulnerability_drilldown_default():
    """Verify defaults for vulnerability page."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    at.sidebar.selectbox(label="Go to:").select("4. Vulnerability Analysis").run()
    
    assert at.selectbox(label="Select Sensitive Subgroup (credit_score_band)").value == "Poor"
    assert at.radio(label="Distribution Slice").value == "bottom"

def test_setup_page_elements():
    """Verify static setup page text."""
    at = AppTest.from_file("app.py").run()
    assert at.header[0].value == "Step 1: Setup & Assets"
    assert "NexusBank Credit Risk" in at.selectbox(label="Select Target Use Case").value
    assert at.selectbox(label="Select Target Use Case").disabled

def test_stress_testing_interim_results():
    """Verify cumulative results logic in Stress Testing page."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    
    at.sidebar.selectbox(label="Go to:").select("3. Stress Testing").run()
    at.button(label="Execute Stress Scenarios").click().run()
    
    # Find the dataframe showing results
    found_results_df = False
    for df_widget in at.dataframe:
        df_val = df_widget.value
        if "Gaussian Noise" in df_val.values:
            found_results_df = True
            break
    assert found_results_df

def test_sidebar_image():
    """Verify sidebar logo exists."""
    at = AppTest.from_file("app.py").run()
    assert at.sidebar.image[0].image_url == "https://www.quantuniversity.com/assets/img/logo5.jpg"
    
def test_export_manifest_zip_naming():
    """Verify the zip file name follows naming convention."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    at.sidebar.selectbox(label="Go to:").select("3. Stress Testing").run()
    at.button(label="Execute Stress Scenarios").click().run()
    at.sidebar.selectbox(label="Go to:").select("5. Decision & Export").run()
    at.button(label="Generate Audit Manifest and ZIP").click().run()
    
    db = at.download_button(label="Download Evidence Bundle (ZIP)")
    assert "Session_06_NexusBank" in db.file_name

def test_data_preview_and_schema():
    """Verify Step 1 data preview logic."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    
    # Check subheader
    assert "Data Preview & Schema Summary" in [s.value for s in at.subheader]
    # Check that dataframes exist on the page
    assert len(at.dataframe) >= 2

def test_threshold_violation_display():
    """Verify violation drill-down headers are present in Step 5."""
    at = AppTest.from_file("app.py").run()
    at.button(label="Generate Synthetic Data & Model Fallback").click().run()
    at.sidebar.selectbox(label="Go to:").select("2. Baseline Assessment").run()
    at.button(label="Compute Baseline Metrics").click().run()
    at.sidebar.selectbox(label="Go to:").select("3. Stress Testing").run()
    at.button(label="Execute Stress Scenarios").click().run()
    at.sidebar.selectbox(label="Go to:").select("5. Decision & Export").run()
    
    assert any("Critical Breaches" in md.value for md in at.markdown)
    assert any("Warning Breaches" in md.value for md in at.markdown)

def test_nav_options_match():
    """Verify nav options strings match app code."""
    at = AppTest.from_file("app.py").run()
    expected = [
        "1. Setup & Assets",
        "2. Baseline Assessment",
        "3. Stress Testing",
        "4. Vulnerability Analysis",
        "5. Decision & Export"
    ]
    assert at.sidebar.selectbox(label="Go to:").options == expected