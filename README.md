# 🫀 Disease Phenotyping on MIMIC-II

> **EHR Analysis coding exercise — Exercise 2.** Unsupervised discovery of ICU patient
> subgroups, reimplementing the [ehrapy MIMIC-II tutorial](https://github.com/theislab/ehrapy)
> story with scikit-learn — plus an interactive **Streamlit dashboard**.

[![CI](https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/REPO/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/REPO/blob/main/notebooks/mimic_phenotyping.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this is

Given a 1776-patient, 31-feature ICU table (the MIMIC-II IAC cohort), this project runs a
complete **unsupervised phenotyping pipeline** to find clinically interpretable patient
subgroups — without ever using the mortality label to *define* them.

```
preprocess → quality control → KNN impute → log-normalize → PCA → UMAP → KMeans → annotate
```

It ships in two forms:

| Deliverable | File | Use it for |
|---|---|---|
| 📓 **Colab notebook** | [`notebooks/mimic_phenotyping.ipynb`](notebooks/mimic_phenotyping.ipynb) | The exercise itself — runs top-to-bottom in Colab |
| 📊 **Streamlit dashboard** | [`dashboard/app.py`](dashboard/app.py) | Interactive exploration with live controls |
| 🧠 **Shared pipeline** | [`src/phenotyping.py`](src/phenotyping.py) | Single source of truth for both |

### Why it always runs

Real MIMIC-II needs credentialed PhysioNet access. So the code **tries `ehrapy`/`ehrdata`
first, then falls back to a built-in synthetic generator** that produces a MIMIC-like cohort
with realistic distributions, injected missingness, and **six planted latent phenotypes**.
On the synthetic data we validate recovery with the Adjusted Rand Index (≈0.39 — strong for
fully unsupervised clustering of overlapping clinical groups).

---

## Quickstart

### Option A — Google Colab (zero setup)
Click the **Open in Colab** badge above and run all cells. *(Update `USER/REPO` in the badge
URLs after you push.)*

### Option B — Run the dashboard locally

```bash
git clone https://github.com/USER/REPO.git
cd REPO
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Then open http://localhost:8501.

### Option C — Just the pipeline

```python
from src.phenotyping import load_synthetic_mimic, run_pipeline

df  = load_synthetic_mimic(n_patients=1776)
res = run_pipeline(df, n_clusters=6)
print(res.silhouette, res.annotations)
```

---

## The dashboard

A dark, clinical-themed Streamlit app with live pipeline controls in the sidebar
(cohort size, *k*, PCA components, KNN neighbors, UMAP toggle, seed) and five tabs:

- **🗺️ Phenotype Map** — UMAP/PCA embedding coloured by cluster, with a clinical-variable overlay
- **🧬 Cluster Profiles** — phenotype cards (size, mean age, 28-day mortality) + a z-scored heatmap
- **📊 Feature Drivers** — what most defines each cluster, plus the PCA scree
- **🩺 Data Quality** — missingness and feature distributions
- **🗃️ Cohort Table** — the labelled cohort, downloadable as CSV

---

## How it maps to the ehrapy tutorial

| ehrapy tutorial step | This project |
|---|---|
| `ed.dt.mimic_2()` | `load_real_mimic()` → synthetic fallback |
| `ep.pp.qc_metrics` | missingness + distribution QC |
| `ep.pp.knn_impute` | `KNNImputer(n_neighbors=5)` |
| `ep.pp.log_norm` on `iv_day_1`, `po2_first` | `np.log1p` on the same two features |
| `ep.pp.pca` | `sklearn.decomposition.PCA` |
| `ep.tl.umap` | `umap-learn` (PCA-2D fallback) |
| `ep.tl.leiden` | `KMeans` (deterministic, dependency-free) |
| `ep.tl.rank_features_groups` | z-score-based auto-annotation |

KMeans stands in for Leiden so the project has no graph-clustering dependency and is
fully reproducible across machines; swap it back to Leiden trivially if `ehrapy` is installed.

---

## Repository layout

```
.
├── notebooks/
│   └── mimic_phenotyping.ipynb     # the Colab exercise
├── dashboard/
│   └── app.py                      # Streamlit UI
├── src/
│   ├── __init__.py
│   └── phenotyping.py              # shared pipeline + synthetic data
├── data/                           # (gitignored CSV outputs)
├── .streamlit/config.toml          # dashboard theme
├── .github/workflows/ci.yml        # smoke-tests the pipeline
├── build_notebook.py               # regenerates the notebook
├── requirements.txt
└── README.md
```

---

## Using the real MIMIC-II data

1. Get [PhysioNet](https://physionet.org/content/mimic2-iaccd/1.0/) access to the IAC dataset.
2. `pip install ehrapy ehrdata`
3. In the dashboard sidebar pick **Real MIMIC-II (ehrapy)**, or in the notebook the loader
   picks it up automatically.

---

## License

MIT — see [LICENSE](LICENSE). The synthetic generator produces fully artificial data and
contains no real patient records.
