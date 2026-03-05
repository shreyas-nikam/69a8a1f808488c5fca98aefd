id: 69a8a1f808488c5fca98aefd_documentation
summary: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Lab 6: Robustness & Functional Validation Stress-Testing Suite

## Overview
Duration: 5:00

In this codelab, you will explore the **Robustness & Functional Validation Stress-Testing Suite**, a comprehensive tool designed to evaluate the reliability and resilience of machine learning models. Using a credit risk use case (NexusBank), this application demonstrates how to systematically "break" a model to understand its failure modes before it reaches production.

### Why is this important?
Machine learning models often perform exceptionally well on static test sets but fail in production due to data drift, sensor noise, or changing economic conditions. Robustness testing ensures that:
1.  **Model Reliability**: The model maintains performance even when inputs are slightly perturbed.
2.  **Fairness & Safety**: The model doesn't disproportionately fail on specific demographic subgroups.
3.  **Operational Readiness**: Stakeholders have a "Go/No-Go" framework based on quantitative degradation thresholds.

### Concepts Covered
- **Baseline Assessment**: Establishing ground truth performance.
- **Probabilistic Calibration**: Understanding the Brier Score for risk models.
- **Stress Scenarios**: Applying Gaussian noise, feature scaling shifts, and missingness spikes.
- **Vulnerability Analysis**: Slicing data by subgroups and tail distributions to find hidden weaknesses.
- **Validation Governance**: Using automated thresholds and cryptographic manifests for auditing.

<aside class="positive">
<b>Pro Tip:</b> Always establish a baseline before applying stress. Without a baseline, you cannot quantify the "Degradation %" which is critical for the final decision.
</aside>

## Setup & Asset Management
Duration: 7:00

The first step in the validation journey is providing the model artifact and the test dataset. This establishes the target system for the stress suite.

### Functionalities
1.  **Artifact Upload**: Users can upload a CSV dataset and a PKL/Joblib model file.
2.  **Synthetic Fallback**: If you don't have a model ready, the app provides a `generate_synthetic_data` function that creates a NexusBank-compliant dataset and a trained model on the fly.
3.  **Schema Validation**: The application strictly validates the uploaded CSV against the NexusBank schema (specific feature columns and target variables) to ensure compatibility.

### Application Logic Flow
The application follows a linear state-driven flow:
- **Input**: CSV + Model Artifact.
- **Processing**: UUID generation for the run, temporary directory storage.
- **Output**: Data preview and schema summary.

```python
# Validation logic snippet
try:
    test_df = pd.read_csv(dp)
    validate_dataset(test_df, FEATURE_COLS, TARGET_COL)
    st.success("Assets uploaded and validated successfully!")
except Exception as e:
    st.error(f"Schema Validation Error: {str(e)}")
```

## Baseline Assessment
Duration: 5:00

Before stressing the model, we must know how it performs under optimal conditions. This step computes the "Ground Truth" metrics.

### Key Metrics
- **AUC (Area Under the Curve)**: Measures the model's ability to distinguish between classes.
- **Accuracy & Precision**: Standard classification performance metrics.
- **Brier Score**: A critical metric for probabilistic models.

### Brier Score Formula
The Brier Score evaluates the accuracy of probabilistic predictions. It represents the mean squared difference between the predicted probability assigned to the possible outcomes and the actual outcome. Lower scores indicate better model calibration.

$$ BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2 $$

Where:
- $N$ is the number of samples.
- $f_i$ is the predicted probability of the outcome.
- $y_i$ is the actual outcome (0 or 1).

<aside class="negative">
<b>Warning:</b> If your model's Brier Score is high at the baseline, it is poorly calibrated and any stress testing results might be misleading.
</aside>

## Robustness Evaluation (Stress Testing)
Duration: 10:00

This is the core of the application. Models must withstand real-world drift. You will configure deterministic transformations to measure model degradation under duress.

### 1. Gaussian Noise
Simulates sensor noise or input jitter. This adds random values from a normal distribution to numerical features.
- **Parameter**: Noise Std Multiplier (controls the intensity).

### 2. Economic Feature Shift
Simulates macro-economic shifts (e.g., sudden inflation or interest rate hikes). It scales feature values by a specific factor.
- **Parameter**: Shift Factor (e.g., 0.8 to simulate a 20% decrease in income).

### 3. Missingness Spike
Simulates intermittent systemic data drops or API failures where certain features become `NaN`.
- **Parameter**: Missing Rate (percentage of data points to drop).

### Execution Flowchart
1.  **Baseline Metrics** $\rightarrow$ **Apply Transformation** (Noise/Shift/Drop) $\rightarrow$ **Re-run Model Inference** $\rightarrow$ **Calculate Delta**.

```python
# Applying noise snippet
def apply_gaussian_noise(X, features, noise_std_multiplier, random_state):
    # Transformation logic applied to specific features
    ...
```

## Vulnerability Analysis
Duration: 8:00

Models often fail on specific population slices or sensitive groups even if global metrics look fine. This page allows for granular drill-downs.

### Subgroup Drill-Down
Users can select a specific sensitive attribute (e.g., `credit_score_band` like 'Poor' or 'Fair') to see if the model's performance collapses specifically for that group compared to the baseline.

### Tail Slice Analytics
In credit risk, the most vulnerable populations are often at the "tails" of the distribution (e.g., the bottom 10% of income earners).
- **Functionality**: Users select a feature and a percentile (top or bottom) to create a specific "slice" for evaluation.

<aside class="positive">
<b>Insight:</b> A model might have an overall AUC of 0.85, but a Tail Slice AUC of only 0.55. This indicates the model is essentially guessing for high-risk or low-income populations.
</aside>

## Final Decision & Archive
Duration: 5:00

The final validation gate compares all cumulative stress results against strict thresholds.

### Degradation Formulation
Degradation is calculated as the percentage drop from the baseline:

$$ \text{Degradation \%} = \frac{\text{Baseline Metric} - \text{Stressed Metric}}{\text{Baseline Metric}} \times 100 $$

### Decision Logic
The application uses two levels of thresholds:
- **WARN_THRESHOLDS**: Triggers a "GO WITH MITIGATION" recommendation.
- **CRITICAL_THRESHOLDS**: Triggers a "NO GO" decision.

### Exporting the Evidence Bundle
The app generates a cryptographically hashed **Audit Manifest**. This includes:
1.  A JSON file containing all configuration parameters and results.
2.  A PNG plot showing degradation curves across all scenarios.
3.  A ZIP file containing all artifacts for compliance and regulatory reporting.

<button>
  [Download Evidence Bundle](https://www.quantuniversity.com)
</button>

<aside class="positive">
<b>Summary:</b> You have now completed a full robustness validation cycle, moving from asset setup to a final production-readiness decision based on empirical stress testing.
</aside>
