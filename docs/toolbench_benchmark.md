# ToolBench Benchmark Results

This document contains the evaluation results of **TabAgent** on the [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) dataset.

## 1. Overall Performance Metrics

The table below summarizes the core retrieval and classification metrics across the three ToolBench difficulty levels (G1, G2, G3) evaluated against a catalog of 46,966 APIs.

| Metric | G1 (Single-tool) | G2 (Same-category) | G3 (Cross-category) |
|---|---|---|---|
| **R-Precision** | **0.830** | **0.710** | **0.708** |
| **Pass Rate@1** | 0.830 | 0.710 | 0.708 |
| **Pass Rate@3** | **1.000** | **0.977** | **0.955** |
| **Pass Rate@5** | 1.000 | 0.997 | 0.993 |
| **Pass Rate@7** | 1.000 | 1.000 | 1.000 |
| **AUC** | 0.991 | 0.953 | 0.942 |
| **F1 (at threshold=0.5)** | 0.898 | 0.592 | 0.587 |
| **F1 (optimal threshold)**| — | — | **0.659** (t=0.35) |
| **Precision** | 0.915 | 0.756 | 0.804 |
| **Recall** | 0.881 | 0.487 | 0.463 (0.683 @ t=0.35) |

---

## 2. Step-Wise Degradation in Multi-Tool Scenarios

The **Pass Rate@3** at each step of the trajectory in G2 and G3 is shown below:

| Step Index | G2 Pass Rate@3 | G3 Pass Rate@3 |
|---|---|---|
| **Step 0** | 0.956 | 0.968 |
| **Step 1** | 0.990 | 0.946 |
| **Step 2** | 0.984 | 0.977 |
| **Step 3** | 1.000 | 0.912 |
| **Step 4** | 0.900 | 0.833 |

---

## 3. Threshold Tuning (G3 Scenario)

Threshold sweep performed on the G3 validation data to find the optimal balance between precision and recall:

| Threshold | F1 Score | Precision | Recall |
|---|---|---|---|
| 0.15 | 0.605 | 0.438 | 0.975 |
| 0.20 | 0.626 | 0.469 | 0.940 |
| 0.25 | 0.635 | 0.501 | 0.865 |
| 0.30 | 0.645 | 0.557 | 0.765 |
| **0.35** | **0.659** | **0.638** | **0.683** |
| 0.40 | 0.642 | 0.690 | 0.600 |
| 0.45 | 0.626 | 0.780 | 0.523 |
| 0.50 | 0.587 | 0.804 | 0.463 |
