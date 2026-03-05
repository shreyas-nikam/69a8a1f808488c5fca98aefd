id: 69a8a1f808488c5fca98aefd_documentation
summary: Lab 6: Robustness & Functional Validation Stress-Testing Suite - Clone Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Robustness & Functional Validation Stress-Testing Suite

## Overview
Duration: 0:05:00

In the world of Model Risk Management (MRM), evaluating a model's performance on a static test set is insufficient. Real-world data is messy, prone to distribution shifts, and often contains noise or missing values. This codelab explores the **QuLab Stress-Testing Suite**, a comprehensive tool designed to evaluate the robustness and functional validity of machine learning models.

### Importance of the Application
The suite allows developers and risk managers to:
1.  **Quantify Robustness**: Measure how much performance degrades when data quality drops.
2.  **Identify Vulnerabilities**: Pinpoint specific subgroups or data "tails" where the model fails.
3.  **Automate Governance**: Establish standardized "Go/No-Go" decisions based on predefined critical thresholds.
4.  **Evidence Generation**: Produce an archive of stress-test results for regulatory compliance.

### Key Concepts
*   **Baseline Assessment**: The performance of the model under ideal (test set) conditions.
*   **Gaussian Noise Injection**: Simulating sensor errors or data entry jitters.
*   **Economic Scaling Shift**: Simulating external shocks like inflation or market downturns.
*   **Missingness Spike**: Simulating system failures where specific features are no longer captured.
*   **Vulnerability Analysis**: Focusing stress on sensitive subgroups (e.g., specific demographic or socio-economic classes).

## Architecture & Workflow
Duration: 0:03:00

The application follows a linear validation pipeline. Below is the conceptual flow:

1.  **Asset Loading**: Ingesting the trained model (`.pkl`) and the reference dataset (`.csv`).
2.  **Schema Validation**: Ensuring the dataset matches the expected features and target required by the model.
3.  **Baseline Calculation**: Establishing the benchmark metrics.
4.  **Stress Perturbation**: Applying mathematical transformations to the baseline data.
5.  **Performance Comparison**: Calculating the delta (degradation) between baseline and stressed scenarios.
6.  **Threshold Logic**: Evaluating failures against Critical and Warning limits.
7.  **Reporting**: Exporting results into a ZIP evidence bundle.

<aside class="positive">
<b>Tip:</b> Always ensure your model implements the <code>predict_proba</code> method, as many robustness metrics (like Brier Score and AUC) rely on probability estimates rather than hard labels.
</aside>

## Step 1: Setup & Assets
Duration: 0:07:00

The first step in any validation exercise is ensuring the integrity of the input assets.

### Asset Upload
You must provide two primary files:
*   **Test Dataset (CSV)**: This should contain the features the model was trained on, the ground truth target, and any sensitive attributes for subgroup analysis.
*   **Trained Model (PKL/JOBLIB)**: A scikit-learn compatible model object.

### Schema Validation
The application automatically validates the uploaded CSV against a predefined schema. It checks for:
*   Presence of required feature columns (e.g., `Age`, `Income`, `LoanAmount`).
*   Presence of the target column.
*   Presence of the sensitive attribute (e.g., `SocioEconomicStatus`).

```python
# Internal logic snippet for validation
def validate_dataset(df, schema):
    for col, dtype in schema.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
```

If extra columns are detected, the application will warn you and drop them during the alignment phase to prevent model input errors.

## Step 2: Baseline Assessment
Duration: 0:05:00

Before stressing the model, we must know how it performs under normal conditions. This step calculates the **Baseline Metrics**.

### Key Metrics
The suite focuses on three primary metrics:
1.  **AUC (Area Under the ROC Curve)**: Measures the model's ability to discriminate between classes.
2.  **Accuracy**: The percentage of correct predictions.
3.  **Brier Score**: Measures the accuracy of probabilistic predictions. It is calculated as:
    $$BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2$$
    where $N$ is the number of samples, $f_i$ is the predicted probability, and $y_i$ is the actual outcome.

<aside class="negative">
<b>Warning:</b> A high AUC does not always mean a robust model. A model can be well-discriminating but poorly calibrated (reflected in a high Brier Score).
</aside>

## Step 3: Stress Configuration
Duration: 0:08:00

This page allows the developer to configure the "intensity" of the stress tests. You can customize three types of perturbations:

### 1. Gaussian Noise Injection
Adds random noise to numerical features. This simulates data corruption.
*   **Parameters**: Feature selection and the `Noise STD Multiplier`.
*   Formula: $x_{stressed} = x_{original} + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma \cdot \text{multiplier})$.

### 2. Economic Scaling Shift
Simulates a systematic shift in data (e.g., everyone's income dropping by 20%).
*   **Parameters**: Feature selection and `Shift Factor` (e.g., 0.8 for a 20% reduction).
*   Formula: $x_{stressed} = x_{original} \cdot \text{factor}$.

### 3. Missingness Spike
Simulates data loss where values are replaced with defaults or indicators (e.g., 0 or mean).
*   **Parameters**: `Missingness Rate` (0.0 to 0.5).

<aside class="positive">
<b>Note:</b> Changing these configurations will clear any previous results to ensure the evaluation reflects the current settings.
</aside>

## Step 4: Robustness Evaluation
Duration: 0:10:00

Once configured, the application runs the stress scenarios. For each scenario, the model makes predictions on the perturbed data, and the application calculates the **Degradation**.

### Degradation Formula
Degradation is measured as the percentage change from the baseline AUC:
$$\text{Degradation \%} = \frac{AUC_{baseline} - AUC_{stressed}}{AUC_{baseline}} \cdot 100$$

### Interpretation
The application displays a results table comparing the scenarios:
*   **Scenario**: The name of the stress test.
*   **AUC**: The performance under stress.
*   **Brier Score**: The calibration under stress.
*   **Status**: Marked as PASS, WARN, or CRITICAL FAIL based on thresholds.

## Step 5: Vulnerability Analysis
Duration: 0:07:00

General stress testing might hide failures that occur only in specific pockets of data. Vulnerability analysis focuses on "Slices".

### Subgroup Stress
This targets specific categorical groups (e.g., customers in the "Poor" socio-economic subgroup). The application filters the data to this subgroup and then applies stress to see if the model is disproportionately fragile for them.

### Tail-Slice Stress
This focuses on the extremes of the distribution (the "tails"). For example:
*   **Tail (Low Income)**: Evaluating only the bottom 10% of earners.
*   Logic: Models often perform well on the "average" case but fail significantly on outliers or extreme values.

## Step 6: Final Decision & Archive
Duration: 0:05:00

The final step consolidates all findings into a regulatory-grade decision and evidence bundle.

### Decision Logic
The suite applies a "Go/No-Go" logic based on thresholds:
*   **GO**: No critical violations and no more than two warnings.
*   **GO WITH MITIGATION**: One critical violation or multiple warnings. Requires a remediation plan.
*   **NO-GO**: Multiple critical violations. The model is deemed unfit for production.

### Thresholds Used
| Metric | Warning Threshold | Critical Threshold |
| : | : | : |
| Min AUC | 0.65 | 0.60 |
| Max Brier Score | 0.20 | 0.25 |

### Evidence Bundle
Upon completion, the application generates a ZIP file containing:
*   `summary_report.md`: A markdown summary of the run.
*   `all_results.csv`: Detailed metrics for every scenario.
*   `degradation_curves.png`: A visual plot showing performance drops.
*   `metadata.json`: Configuration settings for reproducibility.

<button>
  [Download Evidence Bundle (Sample Logic)](https://github.com/QuantUniversity/QuLab)
</button>

<aside class="positive">
<b>Success:</b> You have successfully navigated the robustness validation suite. This process ensures that your model is not just accurate, but resilient!
</aside>
