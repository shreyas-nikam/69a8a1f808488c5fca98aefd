This is a comprehensive `README.md` file tailored for the Streamlit application code you provided.

***

# QuLab: Robustness & Functional Validation Stress-Testing Suite

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## 📌 Project Overview
**QuLab: Lab 6** is a professional-grade stress-testing suite designed to evaluate the robustness and functional validity of machine learning models. The application allows users to upload trained models and datasets to perform rigorous "what-if" analysis, simulating real-world data degradations such as Gaussian noise, economic shifts, and data missingness.

The goal of this suite is to provide a **Go/No-Go decision framework** based on model performance degradation, ensuring that models are resilient before they are deployed into production environments.

---

## ✨ Key Features

### 1. **Asset Validation & Setup**
*   **Data Upload:** Support for CSV test datasets with automated schema validation.
*   **Model Integration:** Compatible with Scikit-Learn models (`.pkl` or `.joblib`) that support the `predict_proba` method.
*   **Schema Alignment:** Automatically aligns input data with the required feature set.

### 2. **Baseline Assessment**
*   Establishes "Ground Truth" performance metrics including **AUC-ROC**, **Accuracy**, and **Brier Score**.
*   Calculates **Max Subgroup Delta AUC** to detect initial bias in the baseline data.

### 3. **Configurable Stress Scenarios**
*   **Gaussian Noise Injection:** Add random variance to continuous features to simulate sensor or reporting errors.
*   **Economic Scaling Shift:** Simulate systemic shifts (e.g., inflation or market downturns) by scaling specific features.
*   **Missingness Spike:** Simulate data pipeline failures by introducing artificial null values.

### 4. **Vulnerability Analysis**
*   **Subgroup Stress:** Isolate specific demographic or economic groups (e.g., "Poor") to see if the model fails disproportionately.
*   **Tail-Slice Testing:** Analyze model performance on the "bottom 10%" of specific features (e.g., Low Income) to identify edge-case vulnerabilities.

### 5. **Reporting & Governance**
*   **Degradation Curves:** Visual representation of how performance drops as stress increases.
*   **Go/No-Go Decision Engine:** Automated recommendations based on pre-defined critical and warning thresholds.
*   **Evidence Bundle:** Downloadable ZIP archive containing metrics, plots, and configuration logs for audit purposes.

---

## 🛠 Technology Stack
*   **Framework:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Visualization:** [Matplotlib](https://matplotlib.org/)
*   **Machine Learning:** Scikit-Learn (Model handling and metrics)
*   **Styling:** Custom CSS/Typography integration (Inter font)

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.8 or higher
*   A trained Scikit-Learn model file (`.pkl` or `.joblib`)
*   A test dataset in `.csv` format matching the expected schema

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/qu-lab-stress-testing.git
    cd qu-lab-stress-testing
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `streamlit`, `pandas`, `matplotlib`, and `scikit-learn` are in your requirements file.)*

### Running the App
```bash
streamlit run app.py
```

---

## 📖 Usage Guide

1.  **Step 1: Setup:** Upload your CSV data and Model file. The system will validate them against the `SCHEMA` defined in the source files.
2.  **Step 2: Baseline:** Run the baseline assessment to see how your model performs under ideal conditions.
3.  **Step 3: Configuration:** Use the sidebar and sliders to define how much "stress" you want to apply (e.g., 20% missingness).
4.  **Step 4: Robustness:** Execute the stress tests. The app will generate comparison tables.
5.  **Step 5: Vulnerability:** Check if the model is particularly weak for certain subgroups.
6.  **Step 6: Archive:** Review the final **Go/No-Go** recommendation and download your PDF/ZIP evidence bundle for documentation.

---

## 📂 Project Structure
```text
├── app.py                # Main Streamlit application entry point
├── source.py             # Core logic for stress functions and metrics (imported as source)
├── reports/              # Directory where generated evidence bundles are stored
├── tmp/                  # Temporary storage for uploaded assets
└── requirements.txt      # Python dependencies
```

---

## ⚖️ Governance Thresholds
The application evaluates results against the following default logic:
*   **Critical Threshold:** AUC < 0.60 or Brier Score > 0.20 (Results in a **NO-GO**).
*   **Warning Threshold:** AUC < 0.70 or Brier Score > 0.15 (Results in **GO WITH MITIGATION**).

---

## 🤝 Contributing
1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---

## ✉️ Contact
**QuantUniversity**
Email: info@quantuniversity.com
Website: [www.quantuniversity.com](https://www.quantuniversity.com)

---
*Disclaimer: This tool is intended for model validation and stress testing purposes. It does not guarantee model performance in live production environments.*