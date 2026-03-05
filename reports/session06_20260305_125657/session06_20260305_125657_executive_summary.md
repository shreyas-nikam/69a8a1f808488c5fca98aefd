# Model Validation Executive Summary - Session 20260305_125657

**Date:** 2026-03-05 12:57:21
## Final Go/No-Go Decision: **NO GO**
The model has critical performance degradations under stress. It is not approved for deployment and requires significant re-evaluation and improvement.

## Scenario Results:
| scenario          |   num_samples |      auc |   degradation_auc_percent |   brier_score |   max_subgroup_delta_auc | Status        |
|:------------------|--------------:|---------:|--------------------------:|--------------:|-------------------------:|:--------------|
| Baseline          |          1000 | 0.900734 |                nan        |     0.0863046 |                0.021799  | PASS          |
| Gaussian Noise    |          1000 | 0.861547 |                  4.35052  |     0.101262  |                0.0812807 | WARN          |
| Economic Shift    |          1000 | 0.900734 |                  0        |     0.0863046 |                0.021799  | PASS          |
| Missingness Spike |          1000 | 0.872    |                  3.18997  |     0.0936168 |                0.0427027 | PASS          |
| Subgroup (Poor)   |           506 | 0.896389 |                  0.482323 |     0.103188  |                0         | PASS          |
| Tail (Low Income) |           100 | 0.705263 |                 21.7013   |     0.272255  |                0.250292  | CRITICAL FAIL |