import matplotlib
matplotlib.use('Agg')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import json
import hashlib
from datetime import datetime

from source import *

# Configure page with deterministic typography system
st.set_page_config(page_title="QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone", layout="wide")

def apply_typography():
    # The configuration for typography and theming typically resides in .streamlit/config.toml.
    st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

apply_typography()

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone")
st.divider()

# Constants strictly reflecting the NexusBank schema and specifications
FEATURE_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'LoanDuration', 'DependentCount']
TARGET_COL = 'true_label'
SENSITIVE_ATTR = 'credit_score_band'
CRITICAL_THRESHOLDS = {'AUC_Degradation': 0.1, 'Brier_Score': 0.25}
WARN_THRESHOLDS = {'AUC_Degradation': 0.05, 'Brier_Score': 0.2}

# Workflow State Machine via session_state
st.session_state.setdefault('run_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
st.session_state.setdefault('created_utc', datetime.utcnow().isoformat())
st.session_state.setdefault('seed', 42)
st.session_state.setdefault('data_loaded', False)
st.session_state.setdefault('model_loaded', False)
st.session_state.setdefault('baseline_metrics', None)
st.session_state.setdefault('results_df', None)
st.session_state.setdefault('df_raw', None)
st.session_state.setdefault('model', None)
st.session_state.setdefault('run_summary', {})
st.session_state.setdefault('selected_instance_idx', 0)
st.session_state.setdefault('stress_config', {
    'noise_features': FEATURE_COLS[:2], 'noise_std': 1.0,
    'shift_features': ['Income'], 'shift_factor': 0.8,
    'missing_features': FEATURE_COLS[-2:], 'missing_rate': 0.1
})

# Sidebar Navigation
pages = [
    "1. Setup & Assets",
    "2. Baseline Assessment",
    "3. Stress Configuration",
    "4. Robustness Evaluation",
    "5. Vulnerability Drill-down",
    "6. Audit & Export"
]
selected_page = st.sidebar.selectbox("Navigation", pages)

@st.cache_data
def cache_load_and_validate_data(data_path):
    df = pd.read_csv(data_path)
    # Explicit coercion according to data contract requirements
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if SENSITIVE_ATTR in df.columns:
        df[SENSITIVE_ATTR] = df[SENSITIVE_ATTR].astype('category')
    return df

@st.cache_resource
def cache_load_model(model_path):
    import joblib
    return joblib.load(model_path)

def do_schema_validation(df):
    # Enforce schema contract by calling validate_dataset
    try:
        # Expecting validate_dataset from source.py
        valid, msg = validate_dataset(df)
        if not valid:
            st.error(f"Schema Validation Failed: {msg}")
            return False, df
    except Exception:
        # Strict fallback in case validate_dataset signature differs
        expected = FEATURE_COLS + [TARGET_COL, SENSITIVE_ATTR]
        missing = [col for col in expected if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return False, df
        extra = [col for col in df.columns if col not in expected]
        if extra:
            st.warning(f"Extra columns detected and will be dropped: {extra}")

    # Always strictly reorder columns to FEATURE_COLS for modeling
    valid_cols = [c for c in (FEATURE_COLS + [TARGET_COL, SENSITIVE_ATTR]) if c in df.columns]
    df_reordered = df[valid_cols].copy()
    return True, df_reordered

if selected_page == "1. Setup & Assets":
    st.header("1. Setup & Assets")
    st.markdown("Functional Validation gate for preventing Production Surprises. Upload your datasets and trained models here.")
    
    col1, col2 = st.columns(2)
    with col1:
        data_file = st.file_uploader("Upload Test Dataset (CSV)", type=['csv'])
    with col2:
        model_file = st.file_uploader("Upload Trained Model (PKL/JOBLIB)", type=['pkl', 'joblib'])

    if data_file and model_file:
        if st.button("Load and Validate"):
            with st.spinner("Aligning features and calculating baseline..."):
                # Reset states for fresh load
                st.session_state.results_df = None
                st.session_state.baseline_metrics = None
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as td:
                    td.write(data_file.getbuffer())
                    dp = td.name
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tm:
                    tm.write(model_file.getbuffer())
                    mp = tm.name

                df = cache_load_and_validate_data(dp)
                model = cache_load_model(mp)
                
                is_valid, df_clean = do_schema_validation(df)
                if is_valid:
                    try:
                        # Ensure usage of load_assets from source.py
                        df_assets, model_assets = load_assets(dp, mp, FEATURE_COLS, TARGET_COL, SENSITIVE_ATTR)
                        st.session_state.df_raw = df_assets
                        st.session_state.model = model_assets
                    except Exception as e:
                        st.toast(f"Using fallback loading procedure due to load_assets exception: {e}")
                        st.session_state.df_raw = df_clean
                        st.session_state.model = model
                    
                    st.session_state.data_loaded = True
                    st.session_state.model_loaded = True
                    
                    st.success("Assets successfully loaded and validated. Schema Check: Green.")
                    st.dataframe(st.session_state.df_raw.head(), use_container_width=True)

elif selected_page == "2. Baseline Assessment":
    st.header("2. Baseline Assessment")
    st.markdown("Detailed breakdown of performance on clean data, identifying the NexusBank default baseline.")
    st.markdown("**Brier Score** representing model calibration as a standard metric:")
    st.markdown(r"$$ BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2 $$")
    
    if not st.session_state.data_loaded or not st.session_state.model_loaded:
        st.warning("Please upload and validate assets on the 'Setup & Assets' page first.")
    else:
        if st.session_state.baseline_metrics is None:
            with st.spinner("Calculating Baseline Metrics..."):
                df = st.session_state.df_raw
                model = st.session_state.model
                
                try:
                    baseline = evaluate_model_performance(model, df[FEATURE_COLS], df[TARGET_COL], df[SENSITIVE_ATTR], "Baseline")
                    st.session_state.baseline_metrics = baseline
                    st.session_state.run_summary['Baseline'] = baseline
                except Exception as e:
                    st.error(f"Error calculating baseline: {str(e)}")
        
        if st.session_state.baseline_metrics is not None:
            st.success("Baseline effectively calculated!")
            st.json(st.session_state.baseline_metrics)

elif selected_page == "3. Stress Configuration":
    st.header("3. Stress Configuration")
    st.markdown("Interactive UI to configure deterministic stress scenarios. Parameters apply seed 42 internally.")
    
    config = st.session_state.stress_config
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Gaussian Noise")
        st.markdown("Simulates sensor/input jitter.")
        noise_features = st.multiselect("Noise Features", FEATURE_COLS, default=config['noise_features'])
        noise_std = st.slider("Noise STD Multiplier", 0.0, 5.0, config['noise_std'], 0.1)
    with col2:
        st.subheader("Feature Scaling")
        st.markdown("Simulates economic shifts (e.g., 20% drop in income).")
        shift_features = st.multiselect("Shift Features", FEATURE_COLS, default=config['shift_features'])
        shift_factor = st.slider("Scaling Factor", 0.0, 2.0, config['shift_factor'], 0.1)
    with col3:
        st.subheader("Missingness")
        st.markdown("Simulates data pipeline failures.")
        missing_features = st.multiselect("Missingness Features", FEATURE_COLS, default=config['missing_features'])
        missing_rate = st.slider("Missing Rate", 0.0, 1.0, config['missing_rate'], 0.05)
        
    st.session_state.stress_config = {
        'noise_features': noise_features, 'noise_std': noise_std,
        'shift_features': shift_features, 'shift_factor': shift_factor,
        'missing_features': missing_features, 'missing_rate': missing_rate
    }
    st.success("Stress Configuration Saved.")

elif selected_page == "4. Robustness Evaluation":
    st.header("4. Robustness Evaluation")
    if not st.session_state.get('baseline_metrics'):
        st.warning("Baseline not computed. Please visit Step 2 before evaluation.")
    else:
        if st.button("Run Evaluation"):
            with st.spinner("Applying deterministic stress transformations..."):
                df = st.session_state.df_raw
                model = st.session_state.model
                config = st.session_state.stress_config
                seed = st.session_state.seed
                
                results = [st.session_state.baseline_metrics]
                
                try:
                    # 1. Gaussian Noise
                    df_noise = apply_gaussian_noise(df.copy(), config['noise_features'], config['noise_std'], seed)
                    X_noise = preprocess_stressed_data(df_noise[FEATURE_COLS], FEATURE_COLS)
                    res_noise = evaluate_model_performance(model, X_noise, df_noise[TARGET_COL], df_noise[SENSITIVE_ATTR], "Gaussian Noise")
                    results.append(res_noise)
                    
                    # 2. Scaling Shift
                    df_shift = apply_feature_scaling_shift(df.copy(), config['shift_features'], config['shift_factor'], seed)
                    X_shift = preprocess_stressed_data(df_shift[FEATURE_COLS], FEATURE_COLS)
                    res_shift = evaluate_model_performance(model, X_shift, df_shift[TARGET_COL], df_shift[SENSITIVE_ATTR], "Feature Shift")
                    results.append(res_shift)
                    
                    # 3. Missingness
                    df_miss = apply_missingness_spike(df.copy(), config['missing_features'], config['missing_rate'], seed)
                    X_miss = preprocess_stressed_data(df_miss[FEATURE_COLS], FEATURE_COLS)
                    res_miss = evaluate_model_performance(model, X_miss, df_miss[TARGET_COL], df_miss[SENSITIVE_ATTR], "Missingness Spike")
                    results.append(res_miss)
                    
                    results_df = pd.DataFrame(results)
                    st.session_state.results_df = results_df
                    
                    # Store detailed metrics logically in summary state
                    st.session_state.run_summary['Robustness'] = results_df.to_dict(orient='records')
                    
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
                    
        if st.session_state.results_df is not None:
            st.subheader("Evaluation Results")
            st.dataframe(st.session_state.results_df, use_container_width=True)
            
            try:
                violations = check_threshold_violations(st.session_state.results_df, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
                if violations:
                    st.warning(f"Threshold violations detected: {violations}")
                else:
                    st.success("No threshold violations detected!")
            except Exception as e:
                st.info(f"Threshold checking logic rendered fallback: {e}")
                
            # Rendering of the specific Performance Degradation charts 
            st.subheader("Performance Degradation Analysis")
            try:
                fig, ax = plt.subplots(figsize=(10, 4))
                res_df = st.session_state.results_df
                scenarios = res_df.get('Scenario', res_df.index)
                
                if 'AUC' in res_df.columns:
                    baseline_val = res_df['AUC'].iloc[0]
                    ax.bar(scenarios, res_df['AUC'], color='#1f77b4')
                    ax.axhline(y=baseline_val - CRITICAL_THRESHOLDS['AUC_Degradation'], color='red', linestyle='dotted', label='Critical Threshold')
                    ax.axhline(y=baseline_val - WARN_THRESHOLDS['AUC_Degradation'], color='orange', linestyle='dotted', label='Warning Threshold')
                    ax.set_title("AUC Degradation across Scenarios")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visual degradation charting encountered an error: {e}")

elif selected_page == "5. Vulnerability Drill-down":
    st.header("5. Vulnerability Drill-down")
    st.markdown("Focus specifically on 'Poor' Credit Band representations and instances of severe financial tail stress (Low Income).")
    
    if not st.session_state.data_loaded:
        st.warning("Please process datasets via Step 1 properly.")
    else:
        df = st.session_state.df_raw
        model = st.session_state.model
        
        if 'Poor' not in df[SENSITIVE_ATTR].values:
            st.warning("No 'Poor' credit band members identified. Skipping explicit subgroup breakdown to preserve session integrity.")
        else:
            with st.spinner("Executing detailed vulnerability analysis..."):
                df_poor = df[df[SENSITIVE_ATTR] == 'Poor'].copy()
                df_tail = df[df['Income'] < df['Income'].quantile(0.1)].copy()
                
                st.subheader("Subgroup Vulnerability: 'Poor' Credit Band")
                try:
                    res_poor = evaluate_model_performance(model, df_poor[FEATURE_COLS], df_poor[TARGET_COL], df_poor[SENSITIVE_ATTR], "Subgroup Analysis: Poor")
                    st.json(res_poor)
                except Exception as e:
                    st.error(f"Subgroup computation failed: {e}")
                
                st.subheader("Tail Stress: Base Decile Income Limits")
                try:
                    res_tail = evaluate_model_performance(model, df_tail[FEATURE_COLS], df_tail[TARGET_COL], df_tail[SENSITIVE_ATTR], "Tail Slice: Low Income")
                    st.json(res_tail)
                except Exception as e:
                    st.error(f"Tail segment evaluation flagged an error: {e}")

elif selected_page == "6. Audit & Export":
    st.header("6. Audit & Export")
    st.markdown("Review Go/No-Go final judgment and synthesize cryptographic evaluation artifacts into an audit bundle.")
    
    # Strictly mandated instance index binding
    idx = st.session_state.get("selected_instance_idx", 0)
    
    if st.session_state.results_df is None:
        st.warning("Robustness Evaluation has not been generated. Return to Step 4.")
    else:
        try:
            verdict = make_go_no_go_decision(st.session_state.results_df)
            if 'NO GO' in str(verdict).upper() or 'NO-GO' in str(verdict).upper():
                st.error(f"FINAL VALIDATION VERDICT: {verdict}", icon="\ud83d\uded1")
            else:
                st.success(f"FINAL VALIDATION VERDICT: {verdict}", icon="\u2705")
        except Exception as e:
            st.status("Automated Go/No-Go judgment is unavailable, manual evaluation fallback initialized.", state="error")
            verdict = "MANUAL_REVIEW_REQUIRED"
            
        if st.button("Generate Evidence Bundle"):
            with st.spinner("Hashing artifacts and creating ZIP archive..."):
                out_dir = tempfile.mkdtemp()
                run_id = st.session_state.run_id
                
                # Standardize evidence definitions via an explicit manifest
                manifest_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "run_id": run_id,
                    "status": verdict,
                    "instance_index_used": idx,
                    "run_summary": st.session_state.run_summary
                }
                
                try:
                    manifest_path = os.path.join(out_dir, f"evidence_manifest_{run_id}.json")
                    with open(manifest_path, 'w') as f:
                        json.dump(manifest_data, f, indent=2)
                    
                    # Execute core export function 
                    artifacts_dict = {"manifest": manifest_path}
                    zip_path = export_artifacts(artifacts_dict, out_dir, run_id)
                    
                    with open(zip_path, "rb") as fp:
                        st.download_button(
                            label="Download Cryptographic Evidence Bundle",
                            data=fp,
                            file_name=f"nexusbank_audit_bundle_{run_id}.zip",
                            mime="application/zip"
                        )
                    st.success("Artifacts properly compiled and securely exported to bundle.")
                except Exception as e:
                    st.error(f"Export construction failed. Audit trace: {e}")
                    
        st.subheader("Run Summary Validation Record")
        st.json(st.session_state.run_summary)


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
