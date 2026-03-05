This is a comprehensive `README.md` file tailored for the Streamlit application provided.

---

# QuLab: Robustness & Functional Validation Stress-Testing Suite

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Validation-blue?style=for-the-badge)

A specialized diagnostic framework designed to evaluate the resilience, fairness, and reliability of Machine Learning models. This application specifically targets the **NexusBank Credit Risk** model, providing a rigorous environment to simulate real-world data degradation and macro-economic shifts before production deployment.

## 🚀 Overview

The **Robustness & Functional Validation Suite** allows data scientists and risk officers to move beyond static test sets. By applying deterministic stressors—such as Gaussian noise, feature scaling shifts, and missingness spikes—the tool quantifies exactly when and how a model fails.

### Key Validation Pillars:
*   **Performance Baseline:** Establishing "Ground Truth" using metrics like AUC, Accuracy, and Brier Score.
*   **Stress Resilience:** Testing model stability against noisy or drifted input data.
*   **Vulnerability Mapping:** Identifying "blind spots" in specific demographic subgroups or distribution tails.
*   **Automated Governance:** Generating a Go/No-Go decision based on pre-defined critical thresholds.

---

## ✨ Features

### 1. Asset & Schema Management
*   Upload custom `.csv` datasets and `.pkl`/`.joblib` model artifacts.
*   Automatic schema validation against NexusBank standards.
*   Built-in synthetic data and model generator for immediate demonstration.

### 2. Baseline Performance Analytics
*   Calculates probabilistic calibration using the **Brier Score**.
*   Standard classification metrics (AUC, Precision, Accuracy).

### 3. Resilience Stress Testing
*   **Gaussian Noise:** Simulates sensor jitter or data entry errors.
*   **Economic Feature Shift:** Simulates inflation or shifts in purchasing power (e.g., Income/Loan scaling).
*   **Missingness Spike:** Simulates systemic data drops or API failures.

### 4. Vulnerability & Fairness Drill-down
*   **Subgroup Analysis:** Isolated testing on sensitive attributes (e.g., `credit_score_band`).
*   **Tail Slice Analytics:** Evaluates model behavior on extreme outliers (e.g., bottom 10% of income earners).

### 5. Audit & Export
*   **Degradation Visualizations:** Comparative plots of baseline vs. stressed performance.
*   **Evidence Bundle:** Generates a cryptographically hashed ZIP manifest containing results, configurations, and plots for regulatory compliance.

---

## 🛠️ Technology Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Data Handling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Visualization:** [Matplotlib](https://matplotlib.org/)
*   **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)
*   **Utilities:** `uuid` (Run tracking), `tempfile` (Session data), `zipfile` (Archiving)

---

## 📦 Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   pip (Python package installer)

### Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/qu-lab-stress-suite.git
   cd qu-lab-stress-suite
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure `streamlit`, `pandas`, `matplotlib`, and `joblib` are included.*

3. **Required Project Files:**
   Ensure the `source.py` file is present in the root directory. This file must contain the core logic for:
   *   `evaluate_model_performance`
   *   `apply_gaussian_noise`, `apply_feature_scaling_shift`, `apply_missingness_spike`
   *   `check_threshold_violations` and `export_artifacts`

---

## 💻 Usage

1. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```

2. **The 5-Step Workflow:**
    *   **Step 1 (Setup):** Upload your model and dataset. Use the "Generate Synthetic" button if you don't have files ready.
    *   **Step 2 (Baseline):** Click "Compute Baseline Metrics" to set the standard for comparison.
    *   **Step 3 (Stress):** Configure the sliders for noise and shifts, then "Execute Stress Scenarios."
    *   **Step 4 (Vulnerability):** Select specific subgroups (e.g., 'Poor' credit band) to see if the model is biased or weak in that slice.
    *   **Step 5 (Export):** Review the automated Go/No-Go verdict and download the **Evidence Bundle** for your audit trail.

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit application (UI & Orchestration)
├── source.py             # Core engine (Transformations, Metrics, Logic)
├── requirements.txt      # Project dependencies
├── data/                 # (Optional) Sample datasets
└── README.md             # Project documentation
```

---

## 📐 Mathematical Formulation: Degradation

The suite calculates model failure based on the percentage of degradation:

$$ \text{Degradation \%} = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100 $$

If the degradation exceeds `CRITICAL_THRESHOLDS`, the system automatically triggers a **NO GO** status, preventing the model from proceeding to production.

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contact

**QuantUniversity**  
Email: [info@quantuniversity.com](mailto:info@quantuniversity.com)  
Website: [www.quantuniversity.com](https://www.quantuniversity.com)

*Disclaimer: This tool is intended for validation and educational purposes within the QuLab environment.*

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@quantuniversity.com](mailto:info@quantuniversity.com)
