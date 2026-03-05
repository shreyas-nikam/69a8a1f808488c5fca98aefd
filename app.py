import matplotlib
matplotlib.use("Agg")
import streamlit as st
from source import *
import pandas as pd
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone")
st.divider()

def apply_typography():
    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    </style>
    """, unsafe_allow_html=True)
apply_typography()

# Initialize State
st.session_state.setdefault("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
st.session_state.setdefault("data_loaded", False)
st.session_state.setdefault("baseline_metrics", None)
st.session_state.setdefault("results_list", [])
st.session_state.setdefault("scenario_config", {})
st.session_state.setdefault("run_summary", {})
st.session_state.setdefault("scenarios_run", False)
st.session_state.setdefault("vulnerability_run", False)
st.session_state.setdefault("X_baseline", None)
st.session_state.setdefault("y_baseline", None)
st.session_state.setdefault("sens_baseline", None)
st.session_state.setdefault("trained_model", None)
st.session_state.setdefault("stress_params", {
    'noise': {'features': ['Age', 'Income', 'LoanAmount', 'CreditScore'], 'noise_std_multiplier': 0.5},
    'shift': {'features': ['Income', 'LoanAmount'], 'shift_factor': 0.8},
    'missing': {'features': ['CreditScore', 'LoanDuration', 'Income'], 'missing_rate': 0.2}
})

@st.cache_resource(show_spinner="Loading Assets...")
def load_assets_cached(data_path, model_path, features, target, sensitive_attr_col):
    return load_assets(data_path, model_path, features, target, sensitive_attr_col)

@st.cache_data(show_spinner="Calculating Baseline...")
def compute_baseline_cached(X, y, sens, _model):
    return evaluate_model_performance(_model, X, y, sens, "Baseline")

def render_markdown_safely(text):
    parts = text.split("$$")
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(f"{part.strip()}")
        else:
            st.markdown(r"$$" + part.strip() + r"$$")

# Sidebar Navigation
st.sidebar.header("Navigation")
pages = [
    "1. Setup & Assets",
    "2. Baseline Assessment",
    "3. Stress Configuration",
    "4. Robustness Evaluation",
    "5. Vulnerability Analysis",
    "6. Final Decision & Archive"
]
selected_page = st.sidebar.selectbox("Go to", options=pages, index=0)

st.sidebar.divider()
st.sidebar.subheader("Threshold Summary")
st.sidebar.markdown(f"**Critical Min AUC:** {CRITICAL_THRESHOLDS['min_auc']}")
st.sidebar.markdown(f"**Warn Min AUC:** {WARN_THRESHOLDS['min_auc']}")
st.sidebar.markdown(f"**Critical Max Brier:** {CRITICAL_THRESHOLDS['max_brier_score']}")
st.sidebar.markdown(f"**Warn Max Brier:** {WARN_THRESHOLDS['max_brier_score']}")

# =============================================================================
# PAGE 1: SETUP & ASSETS
# =============================================================================
if selected_page == pages[0]:
    render_markdown_safely(EXPLANATIONS["introduction"])
    render_markdown_safely(EXPLANATIONS["setup"])
    
    st.subheader("Asset Upload")
    col1, col2 = st.columns(2)
    with col1:
        csv_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
    with col2:
        pkl_file = st.file_uploader("Upload Trained Model (PKL/JOBLIB)", type=["pkl", "joblib"])
        
    if csv_file and pkl_file:
        try:
            df = pd.read_csv(csv_file)
            st.write("Data Preview:")
            st.dataframe(df.head(), use_container_width=True)
            
            validation_passed = False
            try:
                validate_dataset(df, SCHEMA)
                validation_passed = True
            except Exception as e:
                st.error(f"Schema Validation Failed: {e}")
            
            if validation_passed:
                extra_cols = set(df.columns) - set(SCHEMA.keys())
                if extra_cols:
                    st.warning(f"Extra columns detected and will be dropped during alignment: {', '.join(extra_cols)}")
                st.success("Schema Check: PASS")
                
                if st.button("Load and Validate Assets"):
                    with st.spinner("Validating Data and Loading Model..."):
                        os.makedirs("tmp", exist_ok=True)
                        data_path = "tmp/uploaded_data.csv"
                        model_path = "tmp/uploaded_model.pkl"
                        df.to_csv(data_path, index=False)
                        with open(model_path, "wb") as f:
                            f.write(pkl_file.getbuffer())
                            
                        X_base, y_base, sens_base, model = load_assets_cached(data_path, model_path, FEATURE_COLS, TARGET_COL, SENSITIVE_ATTRIBUTE)
                        
                        if not hasattr(model, "predict_proba"):
                            st.error("Model must support predict_proba and be scikit-learn compatible.")
                        else:
                            st.session_state.X_baseline = X_base
                            st.session_state.y_baseline = y_base
                            st.session_state.sens_baseline = sens_base
                            st.session_state.trained_model = model
                            st.session_state.data_loaded = True
                            st.session_state.baseline_metrics = None
                            st.session_state.results_list = []
                            st.session_state.run_summary['data_shape'] = X_base.shape
                            st.success("Assets successfully validated and loaded! Proceed to Baseline Assessment.")
        except Exception as e:
            st.error(f"Failed to read or process files: {e}")

# =============================================================================
# PAGE 2: BASELINE ASSESSMENT
# =============================================================================
elif selected_page == pages[1]:
    render_markdown_safely(EXPLANATIONS["baseline_intro"])
    st.markdown(r"$$BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2$$")
    st.markdown(f"where $N$ is the number of samples, $f_i$ is the predicted probability, and $y_i$ is the actual outcome.")
    
    if not st.session_state.data_loaded:
        st.error("Please load data and model on the Setup page first.")
    else:
        if st.button("Calculate Baseline Metrics"):
            with st.spinner("Calculating Baseline Metrics..."):
                bm = compute_baseline_cached(
                    st.session_state.X_baseline,
                    st.session_state.y_baseline,
                    st.session_state.sens_baseline,
                    st.session_state.trained_model
                )
                st.session_state.baseline_metrics = bm
                st.session_state.results_list = [bm]
                st.session_state.run_summary['baseline_computed'] = True
                st.success("Baseline metrics calculated.")
        
        if st.session_state.baseline_metrics:
            bm = st.session_state.baseline_metrics
            st.subheader("Baseline Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("AUC", f"{bm.get('auc', 0):.4f}")
            col2.metric("Accuracy", f"{bm.get('accuracy', 0):.4f}")
            col3.metric("Brier Score", f"{bm.get('brier_score', 0):.4f}")
            if 'max_subgroup_delta_auc' in bm and not pd.isna(bm['max_subgroup_delta_auc']):
                col4.metric("Max Subgroup Delta AUC", f"{bm['max_subgroup_delta_auc']:.4f}")
            else:
                col4.metric("Max Subgroup Delta AUC", "N/A")

# =============================================================================
# PAGE 3: STRESS CONFIGURATION
# =============================================================================
elif selected_page == pages[2]:
    render_markdown_safely(EXPLANATIONS["stress_scenarios_intro"])
    
    if not st.session_state.baseline_metrics:
        st.error("Please establish Baseline Metrics first.")
    else:
        def reset_results():
            if st.session_state.baseline_metrics:
                st.session_state.results_list = [st.session_state.baseline_metrics]
            st.session_state.scenarios_run = False
            st.session_state.vulnerability_run = False
            st.toast("Settings updated. Downstream results cleared.")
        
        st.subheader("1. Gaussian Noise Injection")
        noise_feats = st.multiselect("Features for Gaussian Noise", FEATURE_COLS, default=st.session_state.stress_params['noise']['features'], on_change=reset_results, key="m_noise")
        noise_std = st.slider("Noise STD Multiplier", 0.0, 2.0, float(st.session_state.stress_params['noise']['noise_std_multiplier']), on_change=reset_results, key="s_noise")
        
        st.subheader("2. Economic Scaling Shift")
        shift_feats = st.multiselect("Features for Scaling Shift", FEATURE_COLS, default=st.session_state.stress_params['shift']['features'], on_change=reset_results, key="m_shift")
        shift_factor = st.slider("Shift Factor", 0.5, 1.5, float(st.session_state.stress_params['shift']['shift_factor']), on_change=reset_results, key="s_shift")
        
        st.subheader("3. Missingness Spike")
        missing_feats = st.multiselect("Features for Missingness", FEATURE_COLS, default=st.session_state.stress_params['missing']['features'], on_change=reset_results, key="m_missing")
        missing_rate = st.slider("Missingness Rate", 0.0, 0.5, float(st.session_state.stress_params['missing']['missing_rate']), on_change=reset_results, key="s_missing")
        
        st.session_state.stress_params = {
            'noise': {'features': noise_feats, 'noise_std_multiplier': noise_std},
            'shift': {'features': shift_feats, 'shift_factor': shift_factor},
            'missing': {'features': missing_feats, 'missing_rate': missing_rate}
        }

# =============================================================================
# PAGE 4: ROBUSTNESS EVALUATION
# =============================================================================
elif selected_page == pages[3]:
    render_markdown_safely(EXPLANATIONS["degradation_formula"])
    
    if not st.session_state.baseline_metrics:
        st.error("Please establish Baseline Metrics first.")
    else:
        if st.button("Run Stress Scenarios"):
            with st.spinner("Applying Stress Scenarios..."):
                st.session_state.results_list = [st.session_state.baseline_metrics]
                
                run_and_evaluate_scenario(
                    st.session_state.trained_model, st.session_state.X_baseline, st.session_state.y_baseline, st.session_state.sens_baseline,
                    "Gaussian Noise", apply_gaussian_noise, st.session_state.stress_params['noise'],
                    st.session_state.X_baseline, st.session_state.baseline_metrics, st.session_state.results_list, st.session_state.scenario_config, "noise_custom"
                )
                
                run_and_evaluate_scenario(
                    st.session_state.trained_model, st.session_state.X_baseline, st.session_state.y_baseline, st.session_state.sens_baseline,
                    "Economic Shift", apply_feature_scaling_shift, st.session_state.stress_params['shift'],
                    st.session_state.X_baseline, st.session_state.baseline_metrics, st.session_state.results_list, st.session_state.scenario_config, "shift_custom"
                )
                
                run_and_evaluate_scenario(
                    st.session_state.trained_model, st.session_state.X_baseline, st.session_state.y_baseline, st.session_state.sens_baseline,
                    "Missingness Spike", apply_missingness_spike, st.session_state.stress_params['missing'],
                    st.session_state.X_baseline, st.session_state.baseline_metrics, st.session_state.results_list, st.session_state.scenario_config, "missing_custom"
                )
                
                st.session_state.scenarios_run = True
                st.session_state.run_summary['scenarios_count'] = len(st.session_state.results_list)
                st.success("Stress scenarios evaluated successfully.")
        
        if st.session_state.scenarios_run:
            res_df = pd.DataFrame(st.session_state.results_list)
            display_cols = ['scenario', 'auc', 'degradation_auc_percent', 'brier_score']
            cols_to_show = [c for c in display_cols if c in res_df.columns]
            st.dataframe(res_df[cols_to_show], use_container_width=True)

# =============================================================================
# PAGE 5: VULNERABILITY ANALYSIS
# =============================================================================
elif selected_page == pages[4]:
    render_markdown_safely(EXPLANATIONS["subgroup_intro"])
    render_markdown_safely(EXPLANATIONS["tail_slice_intro"])
    
    if not st.session_state.scenarios_run:
        st.error("Please run Robustness Evaluation scenarios first.")
    else:
        if st.session_state.sens_baseline is None or st.session_state.sens_baseline.empty:
            st.warning("Sensitive Attribute not found. Subgroup analysis disabled.")
        else:
            if st.button("Run Vulnerability Analysis"):
                with st.spinner("Applying Vulnerability Scenarios..."):
                    evaluate_subgroup_stress(
                        st.session_state.trained_model, st.session_state.X_baseline, st.session_state.y_baseline, st.session_state.sens_baseline,
                        "Subgroup (Poor)", 'Poor', st.session_state.baseline_metrics, st.session_state.results_list, st.session_state.scenario_config, "sub_poor"
                    )
                    
                    evaluate_tail_slice_stress(
                        st.session_state.trained_model, st.session_state.X_baseline, st.session_state.y_baseline, st.session_state.sens_baseline,
                        "Tail (Low Income)", 'Income', 10, 'bottom',
                        st.session_state.baseline_metrics, st.session_state.results_list, st.session_state.scenario_config, "tail_income"
                    )
                    st.session_state.vulnerability_run = True
                    st.session_state.run_summary['vulnerabilities_checked'] = True
                    st.success("Vulnerability analysis completed.")
                    
            if st.session_state.vulnerability_run:
                res_df = pd.DataFrame(st.session_state.results_list)
                st.dataframe(res_df, use_container_width=True)

# =============================================================================
# PAGE 6: FINAL DECISION & ARCHIVE
# =============================================================================
elif selected_page == pages[5]:
    render_markdown_safely(EXPLANATIONS["decision_intro"])
    render_markdown_safely(EXPLANATIONS["archive_intro"])
    
    if len(st.session_state.results_list) < 4:  # Baseline + 3 Scenarios minimum
        st.error("Insufficient scenarios run. Please complete the Robustness Evaluation (minimum 3 stress scenarios required).")
    else:
        if st.button("Generate Final Decision & Evidence Bundle"):
            with st.spinner("Generating Evidence Bundle..."):
                results_df = pd.DataFrame(st.session_state.results_list)
                crit, warn = check_threshold_violations(results_df, CRITICAL_THRESHOLDS, WARN_THRESHOLDS)
                decision, recommendation = make_go_no_go_decision(crit, warn)
                
                display_df = results_df[['scenario', 'num_samples', 'auc']].copy()
                if 'degradation_auc_percent' in results_df.columns:
                    display_df['degradation_auc_percent'] = results_df['degradation_auc_percent']
                if 'brier_score' in results_df.columns:
                    display_df['brier_score'] = results_df['brier_score']
                if 'max_subgroup_delta_auc' in results_df.columns:
                    display_df['max_subgroup_delta_auc'] = results_df['max_subgroup_delta_auc']
                    
                display_df['Status'] = 'PASS'
                for c in crit: display_df.loc[display_df['scenario'] == c['scenario'], 'Status'] = 'CRITICAL FAIL'
                for w in warn: 
                    if display_df.loc[display_df['scenario'] == w['scenario'], 'Status'].iloc[0] != 'CRITICAL FAIL':
                        display_df.loc[display_df['scenario'] == w['scenario'], 'Status'] = 'WARN'
                
                fig = plot_degradation_curves(results_df, st.session_state.baseline_metrics)
                
                base_dir = f"reports/session06_{st.session_state.run_id}"
                artifacts, zip_path = generate_evidence_artifacts(
                    base_dir, st.session_state.run_id, st.session_state.baseline_metrics,
                    results_df, crit, warn, st.session_state.scenario_config, decision, recommendation,
                    display_df, fig
                )
                
                st.session_state.final_decision = decision
                st.session_state.final_recommendation = recommendation
                st.session_state.display_df = display_df
                st.session_state.degradation_fig = fig
                st.session_state.zip_path = zip_path
                
        if "final_decision" in st.session_state:
            st.subheader("Executive Summary")
            if st.session_state.final_decision == "GO":
                st.success(f"**Decision: {st.session_state.final_decision}**\n\n{st.session_state.final_recommendation}")
            elif st.session_state.final_decision == "GO WITH MITIGATION":
                st.warning(f"**Decision: {st.session_state.final_decision}**\n\n{st.session_state.final_recommendation}")
            else:
                st.error(f"**Decision: {st.session_state.final_decision}**\n\n{st.session_state.final_recommendation}")
                
            st.dataframe(st.session_state.display_df, use_container_width=True)
            st.pyplot(st.session_state.degradation_fig)
            
            try:
                with open(st.session_state.zip_path, "rb") as f:
                    st.download_button("Download Evidence Bundle (ZIP)", data=f, file_name=os.path.basename(st.session_state.zip_path), mime="application/zip")
            except Exception as e:
                st.error(f"Could not load generated zip bundle: {e}")


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
