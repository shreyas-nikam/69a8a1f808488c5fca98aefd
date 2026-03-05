This README provides a comprehensive guide to the **QuLab: Robustness & Functional Validation Stress-Testing Suite**, a Streamlit-based application designed for rigorous model validation and reliability assessment.

---

# 🛡️ QuLab: Robustness & Functional Validation Stress-Testing Suite

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

An automated framework for data scientists and model validators to stress-test machine learning models against synthetic noise, distribution shifts, and edge cases. This tool generates a cryptographic audit trail and a "Go/No-Go" recommendation based on pre-defined performance thresholds.

## 📖 Table of Contents
- [Features](#features)
- [Workflow](#workflow)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Validation Metrics](#validation-metrics)
- [License](#license)

---

## ✨ Features

### 1. **Automated Data & Model Validation**
- Upload model artifacts (`.pkl`) and test datasets (`.csv`).
- Automated schema enforcement to ensure feature consistency.
- Environment setup with cryptographic run IDs for auditability.

### 2. **Baseline Performance Benchmarking**
- Calculation of baseline metrics (AUC, Accuracy, Brier Score).
- LaTeX-formatted mathematical transparency for scoring methodologies.

### 3. **Stress Testing Suite**
Evaluate model degradation using three primary scenarios:
- **Gaussian Noise:** Introducing random variance to feature sets.
- **Economic Shift:** Simulating feature scaling shifts (e.g., inflation or market volatility).
- **Missingness Spike:** Testing model resilience against sudden data dropouts/null values.

### 4. **Vulnerability Analysis**
- **Subgroup Stress:** Evaluate performance across sensitive attributes (e.g., Credit Score Band, Income Level).
- **Tail Slice Stress:** Analyze model behavior on edge cases (bottom/top percentiles of specific features).

### 5. **Audit & Decision Support**
- **Threshold Monitoring:** Automatic flagging of "Critical" vs "Warning" performance drops.
- **Go/No-Go Recommendation:** AI-assisted or logic-based decision on model readiness.
- **Evidence Bundling:** Export a ZIP package containing the manifest, degradation curves, and performance logs.

---

## 🚀 Workflow

1.  **Step 1: Data Setup** – Upload `.pkl` model and `.csv` data.
2.  **Step 2: Baseline** – Establish the "ground truth" performance of the model.
3.  **Step 3: Stress Testing** – Configure parameters for noise and shifts.
4.  **Step 4: Vulnerability Analysis** – Check for bias or failure in subgroups and tail ends.
5.  **Step 5: Decision & Export** – Review the audit trail and download the evidence package.

---

## 🛠 Technology Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Handling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning Utilities:** [Joblib](https://joblib.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
- **Visualization:** [Matplotlib](https://matplotlib.org/)
- **Security/Audit:** Cryptographic hashing and tempfile management.

---

## 📁 Project Structure

```text
├── app.py                # Main Streamlit application
├── source.py             # Logic for stress functions, evaluation, and plotting
├── requirements.txt      # Python dependencies
├── data/                 # (Optional) Sample datasets
├── models/               # (Optional) Pre-trained model artifacts
└── README.md             # Project documentation
```

*Note: The `source.py` file must contain core functions like `load_assets()`, `apply_gaussian_noise()`, and `evaluate_model_performance()` for the app to function.*

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.8 or higher.
- A virtual environment (recommended).

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

3. **Set up API Keys (Optional):**
   If using OpenAI or Gemini for LLM-based insights, ensure your keys are ready for input within the application sidebar.

---

## 🖥 Usage

To launch the application, run the following command in your terminal:

```bash
streamlit run app.py
```

### Configuration
- **API Keys:** Provide keys in the sidebar if your `source.py` logic requires upstream LLM services.
- **Thresholds:** You can view current validation thresholds (e.g., `AUC Drop > 0.05` is Critical) in the sidebar expander.

---

## 📊 Validation Metrics

The suite primarily monitors **Degradation (%)**, calculated as:

$$ \text{Degradation} (\%) = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100 $$

A positive value indicates performance loss. The "Go/No-Go" logic flags any scenario where degradation exceeds the **Critical Threshold**.

---

## 🤝 Contributing
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---

## ✉️ Contact
**QuantUniversity**  
Website: [www.quantuniversity.com](https://www.quantuniversity.com)  
*Lab 6: Robustness & Functional Validation Suite*