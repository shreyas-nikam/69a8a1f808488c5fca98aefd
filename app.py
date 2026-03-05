import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import os
import sys
import joblib
import tempfile
import re
from datetime import datetime

from source import *

st.set_page_config(page_title="QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone")
st.divider()

def apply_typography():
    st.markdown(
        """
        <style>
        /* Typography configuration. See .streamlit/config.toml for base font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True
    )

apply_typography()

# Workflow State Machine Initialization
st.session_state.setdefault('run_id', RUN_ID if 'RUN_ID' in globals() else f"RUN_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
st.session_state.setdefault('created_utc', datetime.utcnow().isoformat())
st.session_state.setdefault('seed', 42)
st.session_state.setdefault('data_loaded', False)
st.session_state.setdefault('baseline_metrics', None)
st.session_state.setdefault('all_scenario_results', [])
st.session_state.setdefault('scenario_config', {})
st.session_state.setdefault('model_assets', None)
st.session_state.setdefault('decision_status', None)
st.session_state.setdefault('run_summary', {})
st.session_state.setdefault('selected_instance_idx', 0)

# Sidebar Navigation
nav_options = [
    "Step 1: Data Setup",
    "Step 2: Baseline",
    "Step 3: Stress Testing",
    "Step 4: Vulnerability Analysis",
    "Step 5: Decision & Export"
]
selected_step = st.sidebar.selectbox("Navigation", nav_options)

# User Keys (API Keys)
st.sidebar.markdown("### Configuration")
api_key = st.sidebar.text_input("API Key (OpenAI/Gemini)", type="password")
if not api_key:
    st.sidebar.warning("Please provide an API key if required by upstream services.")
st.session_state['api_key'] = api_key

# Constants Display
crit_thresh = globals().get('CRITICAL_THRESHOLDS', {"auc_drop": 0.05})
warn_thresh = globals().get('WARN_THRESHOLDS', {"auc_drop": 0.02})

with st.sidebar.expander("Validation Thresholds"):
    st.markdown("**Critical Thresholds**")
    st.json(crit_thresh)
    st.markdown("**Warning Thresholds**")
    st.json(warn_thresh)

@st.cache_resource(show_spinner=False)
def get_model_assets(d_path, m_path):
    return load_assets(d_path, m_path)

if selected_step == "Step 1: Data Setup":
    st.markdown("## 1. Validation Environment Setup")
    
    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("Upload Model Artifact (.pkl)", type=["pkl"])
    with col2:
        data_file = st.file_uploader("Upload Test Dataset (.csv)", type=["csv"])
        
    if model_file and data_file:
        with st.spinner("Loading and validating assets..."):
            tmp_dir = tempfile.mkdtemp()
            model_path = os.path.join(tmp_dir, re.sub(r'[^a-zA-Z0-9_\-\.]', '_', model_file.name))
            data_path = os.path.join(tmp_dir, re.sub(r'[^a-zA-Z0-9_\-\.]', '_', data_file.name))
            
            with open(model_path, "wb") as f: f.write(model_file.getbuffer())
            with open(data_path, "wb") as f: f.write(data_file.getbuffer())
            
            # Enforce schema contract
            try:
                df = pd.read_csv(data_path)
                
                if 'FEATURES' in globals():
                    missing_cols = [c for c in FEATURES if c not in df.columns]
                    if missing_cols:
                        st.error(f"Schema Error: Missing required columns: {missing_cols}")
                        st.stop()
                        
                    extra_cols = [c for c in df.columns if c not in FEATURES and ('TARGET' in globals() and c != TARGET)]
                    if extra_cols:
                        st.error(f"Schema Error: Extra columns detected: {extra_cols}")
                        st.stop()
                        
                    # Reorder columns
                    ordered = FEATURES + ([TARGET] if 'TARGET' in globals() and TARGET in df.columns else [])
                    df = df[ordered]
                    df.to_csv(data_path, index=False)
                    
                # Call validate_dataset from source.py if it exists
                if 'validate_dataset' in globals():
                    df = validate_dataset(df)
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(data_path, index=False)
                        
            except Exception as e:
                st.error(f"Schema validation failed: {str(e)}")
                st.stop()
                
            try:
                assets = get_model_assets(data_path, model_path)
                st.session_state.model_assets = assets
                st.session_state.data_loaded = True
                st.success("Assets successfully loaded and validated.")
                
                if isinstance(assets, dict) and 'X_baseline' in assets:
                    st.dataframe(assets['X_baseline'].head(), use_container_width=True)
                else:
                    st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading assets: {str(e)}")

elif selected_step == "Step 2: Baseline":
    st.markdown("## 2. Model Baseline Performance")
    st.markdown(r"$$ BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2 $$")
    st.markdown(r"where $N$ is the sample size, $f_i$ is the predicted probability, and $y_i$ is the actual outcome.")
    
    if not st.session_state.data_loaded:
        st.warning("Please complete Step 1: Data Setup first.")
    else:
        if st.button("Run Baseline Evaluation"):
            with st.spinner("Computing baseline metrics..."):
                try:
                    assets = st.session_state.model_assets
                    baseline = evaluate_model_performance(assets['model'], assets['X_baseline'], assets['y_baseline'])
                    st.session_state.baseline_metrics = baseline
                    st.session_state.run_summary['baseline'] = baseline
                    st.success("Baseline evaluation complete!")
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    
        if st.session_state.baseline_metrics:
            st.markdown("### Baseline Metrics")
            cols = st.columns(len(st.session_state.baseline_metrics))
            for i, (k, v) in enumerate(st.session_state.baseline_metrics.items()):
                cols[i].metric(label=str(k).upper(), value=round(float(v), 4))

elif selected_step == "Step 3: Stress Testing":
    st.markdown("## 3. Robustness Evaluation under Stress")
    
    if not st.session_state.baseline_metrics:
        st.warning("Please compute Baseline performance first (Step 2).")
    else:
        st.markdown(r"$$ \text{Degradation} (\%) = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100 $$")
        st.markdown("where a positive value represents performance loss for AUC and accuracy.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.expander("Gaussian Noise", expanded=True):
                gn_features = st.multiselect("Select Features for Noise", globals().get('FEATURES', []), key='gn_f')
                gn_std = st.slider("Noise Std Multiplier", 0.0, 2.0, 0.5, step=0.1, key='gn_s')
        with c2:
            with st.expander("Economic Shift", expanded=True):
                es_features = st.multiselect("Select Features for Shift", globals().get('FEATURES', []), key='es_f')
                es_factor = st.slider("Shift Factor", 0.5, 1.5, 1.0, step=0.1, key='es_s')
        with c3:
            with st.expander("Missingness", expanded=True):
                ms_features = st.multiselect("Select Features for Missingness", globals().get('FEATURES', []), key='ms_f')
                ms_rate = st.slider("Missing Rate", 0.0, 0.5, 0.1, step=0.05, key='ms_s')
                
        if st.button("Execute Stress Scenarios", use_container_width=True):
            with st.spinner("Running configured stress scenarios..."):
                assets = st.session_state.model_assets
                X, y = assets['X_baseline'], assets['y_baseline']
                model = assets['model']
                
                try:
                    if gn_features:
                        X_gn = apply_gaussian_noise(X.copy(), gn_features, gn_std)
                        X_gn_pre = preprocess_stressed_data(X_gn)
                        res_gn = evaluate_model_performance(model, X_gn_pre, y)
                        res_gn['Scenario'] = f"Gaussian Noise (std={gn_std})"
                        st.session_state.all_scenario_results.append(res_gn)
                        
                    if es_features:
                        X_es = apply_feature_scaling_shift(X.copy(), es_features, es_factor)
                        X_es_pre = preprocess_stressed_data(X_es)
                        res_es = evaluate_model_performance(model, X_es_pre, y)
                        res_es['Scenario'] = f"Economic Shift (factor={es_factor})"
                        st.session_state.all_scenario_results.append(res_es)
                        
                    if ms_features:
                        X_ms = apply_missingness_spike(X.copy(), ms_features, ms_rate)
                        X_ms_pre = preprocess_stressed_data(X_ms)
                        res_ms = evaluate_model_performance(model, X_ms_pre, y)
                        res_ms['Scenario'] = f"Missingness Spike (rate={ms_rate})"
                        st.session_state.all_scenario_results.append(res_ms)
                        
                    st.success("Stress scenarios executed successfully!")
                    st.session_state.run_summary['scenarios_run'] = len(st.session_state.all_scenario_results)
                except Exception as e:
                    st.error(f"Error during stress execution: {str(e)}")

elif selected_step == "Step 4: Vulnerability Analysis":
    st.markdown("## Vulnerability Analysis")
    if not st.session_state.baseline_metrics:
        st.warning("Please compute Baseline performance first (Step 2).")
    else:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Subgroup Stress")
            sens_attr = st.selectbox("Sensitive Attribute", globals().get('SENSITIVE_ATTRIBUTES', ['Credit_Score_Band', 'Income_Level']))
            if st.button("Evaluate Subgroup Stress", use_container_width=True):
                with st.spinner("Evaluating performance on sensitive subgroups..."):
                    try:
                        assets = st.session_state.model_assets
                        res = evaluate_subgroup_stress(assets['model'], assets['X_baseline'], assets['y_baseline'], sens_attr)
                        if isinstance(res, list):
                            st.session_state.all_scenario_results.extend(res)
                        else:
                            st.session_state.all_scenario_results.append(res)
                        st.success(f"Subgroup analysis for {sens_attr} completed!")
                    except Exception as e:
                        st.error(f"Subgroup evaluation failed: {str(e)}")
                        
        with c2:
            st.subheader("Tail Slice Stress")
            tail_feat = st.selectbox("Feature for Tail Slice", globals().get('FEATURES', []))
            col_a, col_b = st.columns(2)
            with col_a:
                tail_pct = st.slider("Percentile", 1, 99, 5)
            with col_b:
                tail_dir = st.radio("Direction", ['bottom', 'top'])
                
            if st.button("Evaluate Tail Slice Stress", use_container_width=True):
                with st.spinner("Evaluating performance on tail slice..."):
                    try:
                        assets = st.session_state.model_assets
                        res = evaluate_tail_slice_stress(assets['model'], assets['X_baseline'], assets['y_baseline'], tail_feat, tail_pct, tail_dir)
                        if isinstance(res, list):
                            st.session_state.all_scenario_results.extend(res)
                        else:
                            st.session_state.all_scenario_results.append(res)
                        st.success(f"Tail slice analysis for {tail_dir} {tail_pct}% of {tail_feat} completed!")
                    except Exception as e:
                        st.error(f"Tail slice evaluation failed: {str(e)}")

elif selected_step == "Step 5: Decision & Export":
    st.markdown("## 4. Audit Trail and Go/No-Go Recommendation")
    
    # Required by constraints
    idx = st.session_state.get("selected_instance_idx", 0)
    
    if not st.session_state.all_scenario_results:
        st.warning("No stress scenarios or vulnerability analyses have been executed yet.")
    else:
        if st.button("Generate Final Decision", use_container_width=True):
            with st.spinner("Checking thresholds and computing decision..."):
                try:
                    violations = check_threshold_violations(st.session_state.all_scenario_results)
                    decision = make_go_no_go_decision(violations)
                    st.session_state.decision_status = decision
                    
                    st.markdown("### Recommendation")
                    if decision == "GO":
                        st.success("✅ GO: Model passed all critical robustness thresholds.")
                    elif decision == "WARN":
                        st.warning("⚠️ WARN: Model showed degradation but passed critical thresholds. Proceed with caution.")
                    else:
                        st.error("🛑 NO GO: Model failed critical robustness thresholds. Remediation required.")
                        
                    st.markdown("### Audit Report")
                    res_table = display_results_table(st.session_state.all_scenario_results)
                    if res_table is not None:
                        st.dataframe(res_table, use_container_width=True)
                    else:
                        st.dataframe(pd.DataFrame(st.session_state.all_scenario_results), use_container_width=True)
                        
                    st.markdown("### Degradation Curves")
                    fig_path = plot_degradation_curves(st.session_state.all_scenario_results)
                    if fig_path and os.path.exists(fig_path):
                        st.image(fig_path, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error generating decision: {str(e)}")
                    
        st.divider()
        st.markdown("### Export Artifacts")
        
        if st.button("Bundle Evidence Package", use_container_width=True):
            with st.spinner("Generating and cryptographically hashing evidence artifacts..."):
                try:
                    if 'export_artifacts' in globals():
                        zip_path, manifest = export_artifacts(st.session_state.run_id, st.session_state.model_assets, st.session_state.all_scenario_results)
                        st.json(manifest)
                    else:
                        zip_path = generate_evidence_artifacts(st.session_state.run_id, st.session_state.model_assets, st.session_state.all_scenario_results)
                        st.json({"run_id": st.session_state.run_id, "status": "bundled", "path": zip_path})
                        
                    if zip_path and os.path.exists(zip_path):
                        with open(zip_path, "rb") as f:
                            safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', os.path.basename(zip_path))
                            if not safe_name.endswith('.zip'): safe_name += '.zip'
                            st.download_button(
                                label="Download Evidence Package",
                                data=f,
                                file_name=safe_name,
                                mime="application/zip",
                                use_container_width=True
                            )
                            st.toast("Artifacts bundled successfully!", icon="✅")
                    else:
                        st.error("Evidence package path is invalid or missing.")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
