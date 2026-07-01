<img width="1040" height="1131" alt="pitchguard_shap" src="https://github.com/user-attachments/assets/30f00f62-41cb-47ed-8c62-a4f2cf35f18d" />
# PitchGuard

**Predicting MLB pitcher elbow-injury (Tommy John) risk — and explaining every prediction with cited sports-science research.**

PitchGuard is a two-part system. A machine-learning model estimates a pitcher's risk of
needing ulnar collateral ligament (UCL) reconstruction — "Tommy John" surgery — from their
pitching data. A retrieval-augmented generation (RAG) layer then explains *why* a pitcher is
flagged, in plain language, grounded in and citing real research. The two halves connect
through SHAP: the model surfaces *which* factors drove a prediction, and those factors become
the query that retrieves the relevant literature.

The result isn't just a risk score — it's a risk score a coach or clinician could actually
interpret and trust.

---

## Architecture

```
Statcast pitch data ──► feature engineering ──► risk model ──► SHAP
(via pybaseball)        (pitcher-season           (XGBoost)      │ top risk factors
                         workload + mechanics)                    ▼
                                                         retrieval query
                                                                  │
 sports-science corpus ──► vector DB (Chroma) ──► nearest relevant chunks
                                                                  │
                                                                  ▼
                                                    grounded, cited explanation
                                                       (local LLM via Ollama)
                                                                  │
                                                                  ▼
                                                        Streamlit interface
```

The **SHAP → retrieval bridge** is the core idea: model interpretability output becomes the
input to the explanation layer, making the predictor and the explainer one system.

---

## Results

Evaluated with **time-based cross-validation** (train on past seasons, test on the next,
repeated 2019–2023) so performance reflects real forward-in-time prediction.

| Metric | Score |
|---|---|
| PR-AUC (mean ± std across 5 folds) | **0.113 ± 0.019** |
| ROC-AUC (mean ± std) | 0.645 ± 0.035 |
| Random baseline (PR) | ~0.05 |
<img width="2080" height="598" alt="pitchguard_metrics" src="https://github.com/user-attachments/assets/2a1523ed-3df8-4da2-8558-403a23e30075" />

The model scores roughly **2× the random base rate**, consistently across folds — modest but
real. The small spread is the point: it shows the signal is genuine, not a lucky split.

**What the model learned (stable across all folds):** higher pitch workload and higher fastball
velocity raise predicted risk; more rest lowers it — consistent with the injury literature.
Age and prior-surgery history lower predicted risk, which reflects **survivorship** (pitchers
still throwing after years or after a prior surgery are the durable ones), not a protective
effect. SHAP shows what the model *used*, not what *causes* injury.

<img width="1040" height="1131" alt="pitchguard_shap" src="https://github.com/user-attachments/assets/90085508-76da-4430-86e1-bffe8e16da9d" />

---

## How it works

**Data** (`data.py`) — pulls pitch-by-pitch Statcast data (2015–2023) one season at a time via
`pybaseball`, cached and saved to parquet.

**Features** (`features.py`) — aggregates raw pitches into pitcher-*season* rows. Engineered,
time-aware features include acute-to-chronic workload ratio (ACWR), velocity trend, release-point
drift, pitch mix, and rest. All rolling windows look strictly backward to prevent data leakage.
Labels come from the public Tommy John Surgery List (surgery within ~18 months of a season).

**Model** (`model.py`) — a logistic-regression baseline and a gradient-boosted tree model
(XGBoost) with class-imbalance weighting; the decision threshold is tuned on a validation set,
never the test set. SHAP explains each prediction.

**Validation** (`crossval.py`) — time-based cross-validation for a trustworthy averaged score,
plus a check on whether SHAP feature directions are stable across folds.

**Explanation** (`rag/engine.py`) — a RAG pipeline: a sports-science corpus is chunked, embedded
(`sentence-transformers`), and stored in a vector database (`chromadb`). For a prediction, the
top SHAP features become a query, the nearest chunks are retrieved by cosine similarity, and a
local LLM (via `Ollama`) writes a grounded, cited answer — refusing when retrieved evidence is
too weak.

**Interface** (`rag/app.py`) — a Streamlit app to enter a pitcher's risk factors and see the
explanation with sources.

---

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · ChromaDB · sentence-transformers · Ollama · Streamlit · pybaseball

---

## Project structure

```
pitchguard/
├── config.py        # shared constants and paths
├── data.py          # pull Statcast -> parquet
├── features.py      # pitches -> pitcher-season features + labels
├── model.py         # baseline + XGBoost + SHAP
├── crossval.py      # time-based cross-validation
├── visualize.py     # PR / ROC / confusion matrix / SHAP plots
├── run.py           # runs the full pipeline
└── rag/
    ├── engine.py    # retrieval + grounded generation
    ├── app.py       # Streamlit interface
    └── corpus/      # paraphrased, cited sports-science sources
```

---

## Running it

```bash
pip install -r requirements.txt

# 1. pull the data (slow — millions of pitches; runs once)
python data.py

# 2. train, evaluate, and cross-validate
python run.py

# 3. (optional) the explanation layer
#    install Ollama from https://ollama.com, then:
ollama pull llama3.2
streamlit run rag/app.py
```

---

## Limitations

This is an honest model on hard, public data — the limitations are part of the result:

- **Few positive cases.** Only ~50–80 documented surgeries across the dataset, which caps how
  well any model can do and makes metrics inherently noisy.
- **Public-data ceiling.** Statcast pitch data plus surgery dates can only go so far; the
  strongest published injury models use biomechanics-lab measurements not publicly available.
- **Correlation, not causation.** SHAP explanations reflect patterns the model exploited
  (including survivorship effects), not proven causes of injury.
- **Not a medical device.** This is a research and portfolio project, not a clinical tool.

A modest, well-validated result is the correct outcome here — a high accuracy number on this
data would more likely indicate a data-leakage bug than a genuinely strong model.

---

## What this project is really about

The methodology generalizes beyond baseball: swap pitch data for clinical or wearable signals
and the sports-science corpus for medical literature, and the same architecture becomes a
disease-risk predictor with grounded, cited explanations. The pipeline doesn't care whether the
signal is a fastball or a heart rate.
