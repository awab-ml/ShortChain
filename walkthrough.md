# TabAgent Phase 1 MVP — Walkthrough & Status

## Project Status: MVP Complete 🚀

Phase 1 of TabAgent is **100% complete**. We successfully built the MVP pipeline that translates raw agent execution trajectories into a lightweight, fast classification engine (using XGBoost/RandomForest) to replace generative LLM decision layers. 

All 44 unit tests are passing, and the end-to-end pipeline (from log ingestion to test-set evaluation) runs flawlessly.

## Pipeline Architecture

The system flows through four main stages. You can execute these via the `scripts/` directory:

### 1. Data Ingestion (`tabagent/ingest`)
*   **What it does:** Reads raw agent execution logs (`.json` or `.jsonl`). 
*   **How it works:** We implemented a `JSONLTrajectoryLoader` that uses a **field mapping configuration**. This allows the tool to parse trajectories from various sources (like LangChain, IBM CUGA, or custom internal formats) without altering the Python code. You simply update `configs/default.yaml`.
*   **Output:** Validated Pydantic `Trajectory` objects containing intent, steps, and automatically derived `tools_used`.

### 2. Dataset Construction (`tabagent/dataset`)
*   **What it does:** Converts trajectories into machine learning ready rows.
*   **How it works:** Following the "pointwise reduction" approach from the paper, the `DatasetBuilder` creates positive pairs `(context, tool)` for tools the agent actually called (Label=1). It then dynamically samples negative tools from the catalog that weren't used (Label=0).
*   **Splitting:** We implemented a `GroupStratifiedSplitter` that splits the data using `GroupKFold` on the `task_id`. This strictly prevents data leakage, guaranteeing that rows from the same agent task don't appear in both the training and test sets.

### 3. Classifier Head (`tabagent/head`)
*   **What it does:** The discriminative layer that learns tool routing rules.
*   **How it works:** The `TabAgentClassifier` is a unified wrapper supporting XGBoost, Random Forest, and Logistic Regression. 
*   **Encoding:** It automatically applies TF-IDF encoding for text fields (handling edge cases where text might be blank gracefully) and label encoding for categoricals (like `app_name`). 
*   **Training & Inference:** `trainer.py` orchestrates cross-validation, while `inference.py` provides the millisecond-latency endpoint used to score live candidates.

### 4. Evaluation Metrics (`tabagent/evaluation`)
*   **What it does:** Benchmarks the classifier's performance realistically.
*   **How it works:** It measures standard classification metrics (F1, Accuracy), but more importantly, it implements the paper's custom ranking metrics: **R-Precision (P@R)** and **Recall@K**. These metrics do proper macro-averaging grouped by the specific task.

---

## Verifying the Build

To see the system in action using the 15 synthetic example trajectories provided:

**1. Build the Dataset**
```bash
python scripts/build_dataset.py --trajectories data/example/ --output data/datasets/
```

**2. Train the Model (Cross-Validated)**
```bash
python scripts/train.py --dataset data/datasets/ --output models/tabagent.pkl --folds 3
```

**3. Evaluate the Model**
```bash
python scripts/evaluate.py --model models/tabagent.pkl --dataset data/datasets/test.csv
```

## Next Steps (Phase 2 & Beyond)
Now that the core machine-learning infrastructure is solid and tested:
1.  **Phase 2:** We will replace the simple TF-IDF vectors with Dense Semantic Retrieval embeddings (e.g., E5-small) and add BM25/DSR baseline comparisons.
2.  **Phase 3 & 4:** We will tackle the heavy multi-agent LLM systems (**TabSchema** for automated feature extraction, and **TabSynth** for generative data augmentation). 

The foundation is ready for real agent data!
