# QuLab: Robustness & Functional Validation Stress-Testing Suite

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 📋 Project Overview
The **QuLab: Robustness & Functional Validation Stress-Testing Suite** is an enterprise-grade diagnostic tool designed for the **NexusBank** ecosystem. It serves as a functional validation gate to prevent production surprises by subjecting machine learning models to deterministic stress scenarios. 

This application evaluates how model performance (specifically AUC and Brier Score) degrades under various "real-world" data corruptions, such as sensor noise, economic shifts, and data pipeline failures.

## ✨ Key Features
*   **Asset Validation:** Upload and validate CSV datasets and Joblib/PKL models against the NexusBank data contract.
*   **Baseline Assessment:** Calculate standard performance metrics on clean data, including a detailed Brier Score analysis for model calibration.
*   **Deterministic Stress Testing:**
    *   **Gaussian Noise:** Simulates input jitter.
    *   **Feature Scaling:** Simulates economic shifts (e.g., income drops).
    *   **Missingness Spikes:** Simulates data pipeline or API failures.
*   **Vulnerability Drill-down:** Targeted analysis on sensitive subgroups (e.g., 'Poor' credit bands) and financial tail risks (low-income deciles).
*   **Automated Governance:** 
    *   Threshold violation detection (AUC degradation & Brier Score limits).
    *   Go/No-Go decision logic for model deployment.
*   **Audit-Ready Export:** Generation of a cryptographic evidence bundle containing execution manifests and validation records.

## 🛠️ Technology Stack
*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Visualization:** [Matplotlib](https://matplotlib.org/)
*   **Model Handling:** [Joblib](https://joblib.readthedocs.io/)
*   **Typography:** Inter UI System

## 📂 Project Structure
```text
.
├── app.py                # Main Streamlit application script
├── source.py             # Logic for stress transformations & metrics (Required)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites
* Python 3.8 or higher
* `pip` package manager

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/qu-lab-stress-testing.git
   cd qu-lab-stress-testing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure `source.py` is present:**
   The application relies on a `source.py` file containing the following core functions:
   - `validate_dataset`, `load_assets`
   - `evaluate_model_performance`
   - `apply_gaussian_noise`, `apply_feature_scaling_shift`, `apply_missingness_spike`
   - `preprocess_stressed_data`, `check_threshold_violations`
   - `make_go_no_go_decision`, `export_artifacts`

### Running the Application
```bash
streamlit run app.py
```

## 📖 Usage Guide

1.  **Setup & Assets:** Upload your `test_data.csv` and `model.joblib`. The app will validate the schema against the NexusBank requirements (Age, Income, LoanAmount, etc.).
2.  **Baseline Assessment:** View the model's performance on undisturbed data to establish a benchmark.
3.  **Stress Configuration:** Use the sliders in the sidebar to define the intensity of noise, scaling factors, and missing data rates.
4.  **Robustness Evaluation:** Run the suite to see how the model responds to the configured stresses. Review the "AUC Degradation" charts.
5.  **Vulnerability Drill-down:** Inspect how specific demographics, like the 'Poor' credit score band, are disproportionately affected.
6.  **Audit & Export:** Review the final **Go/No-Go** verdict and download the `.zip` evidence bundle for compliance records.

## 📉 Data Specifications
The application expects a specific schema:
*   **Features:** `Age`, `Income`, `LoanAmount`, `CreditScore`, `LoanDuration`, `DependentCount`.
*   **Target:** `true_label`.
*   **Sensitive Attribute:** `credit_score_band`.

## 🛡️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✉️ Contact
**QuantUniversity**
Website: [www.quantuniversity.com](https://www.quantuniversity.com)

---
*Developed for the Robustness & Functional Validation Lab Suite.*

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@quantuniversity.com](mailto:info@quantuniversity.com)
