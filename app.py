import matplotlib
matplotlib.use('Agg')
import streamlit as st
from source import *
import os
import tempfile
import uuid
import zipfile
import json
import pandas as pd

st.set_page_config(page_title="QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone")
st.divider()

# Set the global deterministic seed
set_global_seed(RANDOM_SEED)

# Initialize Typography
def apply_typography():
    st.markdown("""
    <style>
    /* Configured Typography for Inter Font mapping */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
apply_typography()

# Initialize Session State Variables
st.session_state.setdefault('run_id', str(uuid.uuid4())[:8])
st.session_state.setdefault('data_loaded', False)
st.session_state.setdefault('baseline_metrics', {})
st.session_state.setdefault('results_list', [])
st.session_state.setdefault('config_snapshot', {})
st.session_state.setdefault('data_path', None)
st.session_state.setdefault('model_path', None)

TEMP_DIR = os.path.join(tempfile.gettempdir(), "nexus_stress_suite")
os.makedirs(TEMP_DIR, exist_ok=True)

def reset_analysis_state():
    """Resets downstream analysis when new data/models are loaded."""
    st.session_state.baseline_metrics = {}
    st.session_state.results_list = []
    st.session_state.config_snapshot = {}

@st.cache_resource
def get_cached_assets(data_path, model_path):
    """Loads model and data. Cached by file path to prevent redundant reads."""
    return load_assets(data_path, model_path, FEATURE_COLS, TARGET_COL, SENSITIVE_ATTRIBUTE)

# Sidebar Navigation
st.sidebar.header("Navigation")
nav_options = [
    "1. Setup & Assets",
    "2. Baseline Assessment",
    "3. Stress Testing",
    "4. Vulnerability Analysis",
    "5. Decision & Export"
]
page_selection = st.sidebar.selectbox("Go to:", nav_options, index=0)

# ==========================================
# PAGE 1: SETUP & ASSETS
# ==========================================
if page_selection == nav_options[0]:
    st.header("Step 1: Setup & Assets")
    st.markdown("Start the validation journey by providing the model artifact and the test dataset. This establishes the target system for the stress suite. For this demonstration, we focus on the NexusBank Credit Risk model, which predicts default probability.")
    
    st.selectbox("Select Target Use Case", ["NexusBank Credit Risk"], index=0, disabled=True)
    
    st.subheader("Upload Artifacts")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_data = st.file_uploader("Upload Test Dataset (CSV)", type=['csv'])
    with col2:
        uploaded_model = st.file_uploader("Upload Model Artifact (PKL/JOBLIB)", type=['pkl', 'joblib'])
    
    if st.button("Generate Synthetic Data & Model Fallback"):
        with st.spinner("Generating synthetic data and training fallback model..."):
            df_synth, model_synth = generate_synthetic_data(RANDOM_SEED)
            
            new_file_id = str(uuid.uuid4())[:8]
            dp = os.path.join(TEMP_DIR, f"synth_data_{new_file_id}.csv")
            mp = os.path.join(TEMP_DIR, f"synth_model_{new_file_id}.pkl")
            
            df_synth.to_csv(dp, index=False)
            import joblib
            joblib.dump(model_synth, mp)
            
            st.session_state.data_path = dp
            st.session_state.model_path = mp
            st.session_state.data_loaded = True
            reset_analysis_state()
            st.success("Synthetic data and model generated successfully!")

    if uploaded_data and uploaded_model:
        new_file_id = str(uuid.uuid4())[:8]
        dp = os.path.join(TEMP_DIR, f"uploaded_data_{new_file_id}.csv")
        mp = os.path.join(TEMP_DIR, f"uploaded_model_{new_file_id}.pkl")
        
        with open(dp, "wb") as f:
            f.write(uploaded_data.getbuffer())
        with open(mp, "wb") as f:
            f.write(uploaded_model.getbuffer())

        # Validate Dataset against Schema
        try:
            test_df = pd.read_csv(dp)
            validate_dataset(test_df, FEATURE_COLS, TARGET_COL)
            
            # Update paths if valid
            if st.session_state.data_path != dp or st.session_state.model_path != mp:
                st.session_state.data_path = dp
                st.session_state.model_path = mp
                st.session_state.data_loaded = True
                reset_analysis_state()
                st.success("Assets uploaded and validated against NexusBank Schema successfully!")
        except Exception as e:
            st.error(f"Schema Validation Error: {str(e)}")
            st.session_state.data_loaded = False

    if st.session_state.data_loaded:
        st.divider()
        st.subheader("Data Preview & Schema Summary")
        st.markdown("Review the first few records and the data type alignment of the uploaded dataset.")
        
        df_preview = pd.read_csv(st.session_state.data_path)
        st.dataframe(df_preview.head(), use_container_width=True)
        
        schema_summary = pd.DataFrame({
            "Column": df_preview.columns,
            "Dtype": df_preview.dtypes.astype(str),
            "Missing Values": df_preview.isnull().sum()
        })
        st.dataframe(schema_summary, use_container_width=True)

# ==========================================
# PAGE 2: BASELINE ASSESSMENT
# ==========================================
elif page_selection == nav_options[1]:
    st.header("Step 2: Baseline Assessment")
    if not st.session_state.data_loaded:
        st.warning("Please complete Step 1: Setup & Assets to unlock this step.")
    else:
        st.markdown("Before breaking the model, we must know how it performs under optimal conditions. This page computes the 'Ground Truth' metrics against which all stress scenarios will be measured.")
        
        st.markdown("### Calibration Context: Brier Score")
        st.markdown("The Brier Score evaluates the accuracy of probabilistic predictions. It represents the mean squared difference between predicted probability assigned to the possible outcomes and the actual outcome. Lower scores indicate better model calibration and confidence.")
        st.markdown(r"$$ BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2 $$")

        if st.button("Compute Baseline Metrics"):
            with st.spinner("Computing Baseline Metrics..."):
                X_base, y_base, sens_base, model = get_cached_assets(st.session_state.data_path, st.session_state.model_path)
                baseline_metrics = evaluate_model_performance(model, X_base, y_base, sens_base, "Baseline")
                
                st.session_state.baseline_metrics = baseline_metrics
                st.session_state.results_list = [baseline_metrics]
            st.success("Baseline assessment complete! You can now proceed to Stress Testing.")

        if st.session_state.baseline_metrics:
            st.divider()
            st.subheader("Baseline Performance Results")
            bm = st.session_state.baseline_metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("AUC", f"{bm.get('auc', 0):.4f}")
            col2.metric("Accuracy", f"{bm.get('accuracy', 0):.4f}")
            col3.metric("Precision", f"{bm.get('precision', 0):.4f}")
            col4.metric("Brier Score", f"{bm.get('brier_score', 0):.4f}")

# ==========================================
# PAGE 3: STRESS TESTING
# ==========================================
elif page_selection == nav_options[2]:
    st.header("Step 3: Robustness Evaluation (Stress Testing)")
    if not st.session_state.baseline_metrics:
        st.warning("Please compute Baseline Metrics in Step 2 to unlock Stress Testing.")
    else:
        st.markdown("Models must withstand real-world drift. What happens if the sensors (data inputs) get noisy? Or if the economy shifts, drastically altering scaling bounds? Configure deterministic transformations below to measure model degradation under duress.")
        
        st.subheader("Scenario Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**1. Gaussian Noise**")
            st.markdown("Simulate sensor noise or input jitter.")
            noise_features = st.multiselect("Features to inject noise", FEATURE_COLS, default=FEATURE_COLS[:4])
            noise_std = st.slider("Noise Std Multiplier", 0.1, 2.0, 0.5)
            
        with col2:
            st.markdown("**2. Economic Feature Shift**")
            st.markdown("Simulate macro-economic shifts (e.g., inflation).")
            shift_features = st.multiselect("Features to shift", FEATURE_COLS, default=['Income', 'LoanAmount'])
            shift_factor = st.slider("Shift Factor", 0.5, 1.5, 0.8)
            
        with col3:
            st.markdown("**3. Missingness Spike**")
            st.markdown("Simulate intermittent systemic data drops.")
            miss_features = st.multiselect("Features to spike missingness", FEATURE_COLS, default=FEATURE_COLS)
            missing_rate = st.slider("Missing Rate", 0.05, 0.5, 0.1)

        if st.button("Execute Stress Scenarios"):
            with st.spinner("Applying Stress Scenarios and Evaluating Degradation..."):
                X_base, y_base, sens_base, model = get_cached_assets(st.session_state.data_path, st.session_state.model_path)
                
                # Reset scenario lists but keep baseline
                st.session_state.results_list = [st.session_state.baseline_metrics]
                
                # 1. Noise
                run_and_evaluate_scenario(
                    model, X_base, y_base, sens_base,
                    "Gaussian Noise", apply_gaussian_noise,
                    {'features': noise_features, 'noise_std_multiplier': noise_std, 'random_state': RANDOM_SEED},
                    FEATURE_COLS, st.session_state.baseline_metrics, st.session_state.results_list,
                    st.session_state.config_snapshot, "scenario_noise"
                )
                
                # 2. Shift
                run_and_evaluate_scenario(
                    model, X_base, y_base, sens_base,
                    "Economic Shift", apply_feature_scaling_shift,
                    {'features': shift_features, 'shift_factor': shift_factor, 'random_state': RANDOM_SEED},
                    FEATURE_COLS, st.session_state.baseline_metrics, st.session_state.results_list,
                    st.session_state.config_snapshot, "scenario_shift"
                )
                
                # 3. Missingness
                run_and_evaluate_scenario(
                    model, X_base, y_base, sens_base,
                    "Missingness Spike", apply_missingness_spike,
                    {'features': miss_features, 'missing_rate': missing_rate, 'random_state': RANDOM_SEED},
                    FEATURE_COLS, st.session_state.baseline_metrics, st.session_state.results_list,
                    st.session_state.config_snapshot, "scenario_missingness"
                )
                
            st.success("Scenarios Executed Successfully! View interim results below or proceed to Vulnerability Analysis.")
            
        if len(st.session_state.results_list) > 1:
            st.divider()
            st.subheader("Interim Stress Results")
            st.dataframe(pd.DataFrame(st.session_state.results_list), use_container_width=True)

# ==========================================
# PAGE 4: VULNERABILITY ANALYSIS
# ==========================================
elif page_selection == nav_options[3]:
    st.header("Step 4: Vulnerability Analysis")
    if not st.session_state.baseline_metrics:
        st.warning("Please compute Baseline Metrics in Step 2 first.")
    else:
        st.markdown("Models often fail on specific population slices or sensitive groups even if global metrics look fine. Use this section to drill down into fairness constraints and tail distributions.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Subgroup Drill-Down")
            target_group = st.selectbox("Select Sensitive Subgroup (credit_score_band)", ['Poor', 'Fair', 'Good', 'Excellent'], index=0)
            
        with col2:
            st.subheader("Tail Slice Analytics")
            tail_feature = st.selectbox("Select Feature", FEATURE_COLS, index=1) # Income default
            tail_percentile = st.slider("Target Percentile", 1, 50, 10)
            slice_type = st.radio("Distribution Slice", ['bottom', 'top'], index=0)
            
        if st.button("Analyze Vulnerabilities"):
            with st.spinner("Computing subset distributions and slicing performance..."):
                X_base, y_base, sens_base, model = get_cached_assets(st.session_state.data_path, st.session_state.model_path)
                
                # Execute Subgroup Stress
                res_sub = evaluate_subgroup_stress(
                    model, X_base, y_base, sens_base, 
                    f"Subgroup Stress ({target_group})", target_group,
                    FEATURE_COLS, st.session_state.baseline_metrics, st.session_state.results_list,
                    st.session_state.config_snapshot, f"subgroup_{target_group}"
                )
                if res_sub is None:
                    st.warning(f"Warning: No samples found for subgroup {target_group}. Metrics skipped.")
                
                # Execute Tail Slice Stress
                res_tail = evaluate_tail_slice_stress(
                    model, X_base, y_base, sens_base, 
                    f"Tail Slice ({slice_type} {tail_percentile}% {tail_feature})", 
                    tail_feature, tail_percentile, slice_type,
                    FEATURE_COLS, st.session_state.baseline_metrics, st.session_state.results_list,
                    st.session_state.config_snapshot, f"tail_{tail_feature}_{slice_type}"
                )
                if res_tail is None:
                    st.warning("Warning: No samples found for tail slice criteria. Metrics skipped.")
                    
            st.success("Vulnerability Analysis Complete! Check Step 5 for the final Go/No-Go Decision.")
            
        if len(st.session_state.results_list) > 1:
            st.divider()
            st.subheader("Cumulative Analysis Log")
            st.dataframe(pd.DataFrame(st.session_state.results_list), use_container_width=True)

# ==========================================
# PAGE 5: DECISION & EXPORT
# ==========================================
elif page_selection == nav_options[4]:
    st.header("Step 5: Final Decision & Archive")
    if len(st.session_state.results_list) <= 1:
        st.warning("Insufficient evidence. Please execute stress scenarios in Step 3 or 4 to unlock the final decision.")
    else:
        st.markdown("The final validation gate compares all cumulative stress results against strict `CRITICAL_THRESHOLDS` and `WARN_THRESHOLDS`. This determines the production readiness (Go/No-Go) and computes cryptographic evidence for audit.")
        
        st.markdown("### Degradation Formulation")
        st.markdown(r"$$ \text{Degradation \%} = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100 $$")
        
        results_df = pd.DataFrame(st.session_state.results_list)
        
        critical_violations, warn_violations = check_threshold_violations(results_df, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
        decision, recommendation = make_go_no_go_decision(critical_violations, warn_violations)
        
        st.divider()
        st.subheader("Validation Verdict")
        
        if decision == "NO GO":
            st.error(f"**{decision}**\n\n{recommendation}")
        elif decision == "GO WITH MITIGATION":
            st.warning(f"**{decision}**\n\n{recommendation}")
        else:
            st.success(f"**{decision}**\n\n{recommendation}")
            
        st.divider()
        st.subheader("Violation Drill-Down")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Critical Breaches**")
            if critical_violations:
                st.dataframe(pd.DataFrame(critical_violations), use_container_width=True)
            else:
                st.info("No critical breaches detected.")
        with col2:
            st.markdown("**Warning Breaches**")
            if warn_violations:
                st.dataframe(pd.DataFrame(warn_violations), use_container_width=True)
            else:
                st.info("No warning breaches detected.")
                
        st.divider()
        st.subheader("Degradation Visual Evidence")
        fig = plot_degradation_curves(results_df, st.session_state.baseline_metrics, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
        st.pyplot(fig)
        
        st.divider()
        st.subheader("Evidence Bundle Manifest Generation")
        if st.button("Generate Audit Manifest and ZIP"):
            with st.spinner("Generating cryptographically hashed Audit Manifest and Zipping artifacts..."):
                export_dir = os.path.join(TEMP_DIR, f"export_{st.session_state.run_id}")
                manifest = export_artifacts(
                    out_dir=export_dir, 
                    baseline_metrics=st.session_state.baseline_metrics, 
                    scenario_results_df=results_df, 
                    critical_violations=critical_violations, 
                    warn_violations=warn_violations, 
                    scenario_config=st.session_state.config_snapshot, 
                    decision=decision, 
                    recommendation=recommendation, 
                    run_id=st.session_state.run_id
                )
                
                # Save plot locally to the export dir
                fig.savefig(os.path.join(export_dir, "degradation_curves.png"))
                
                # Zip Directory
                zip_path = os.path.join(TEMP_DIR, f"Session_06_NexusBank_{st.session_state.run_id}.zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(export_dir):
                        for file in files:
                            zipf.write(os.path.join(root, file), arcname=file)
                            
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="Download Evidence Bundle (ZIP)",
                        data=f,
                        file_name=f"Session_06_NexusBank_{st.session_state.run_id}.zip",
                        mime="application/zip"
                    )


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@quantuniversity.com](mailto:info@quantuniversity.com)
''')
