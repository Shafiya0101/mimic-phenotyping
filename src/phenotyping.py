"""
phenotyping.py
==============
Reusable disease-phenotyping pipeline for the MIMIC-II IAC dataset.

This module is the single source of truth shared by:
  * notebooks/mimic_phenotyping.ipynb   (the Colab exercise)
  * dashboard/app.py                    (the Streamlit UI)

The pipeline mirrors the official ehrapy MIMIC-II tutorial:
    preprocess -> QC -> impute -> normalize -> PCA -> UMAP/neighbors
    -> KMeans clustering -> cluster profiling -> annotation.

It is deliberately written with scikit-learn (not ehrapy) so it runs
anywhere with no credentialed-data access and no heavy dependencies,
while reproducing the same analytical story. A synthetic MIMIC-like
generator is provided so the whole thing runs instantly offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import umap  # umap-learn

    _HAS_UMAP = True
except Exception:  # pragma: no cover - umap optional
    _HAS_UMAP = False


# --------------------------------------------------------------------------- #
# 1. Data loading                                                             #
# --------------------------------------------------------------------------- #

# These are the real numeric/flag feature names from the MIMIC-II IAC dataset.
NUMERIC_FEATURES = [
    "age", "weight_first", "bmi", "sapsi_first", "sofa_first",
    "hr_1st", "temp_1st", "map_1st", "spo2_1st",
    "wbc_first", "hgb_first", "platelet_first",
    "sodium_first", "potassium_first", "creatinine_first",
    "po2_first", "pco2_first", "iv_day_1", "icu_los_day",
]

FLAG_FEATURES = [
    "gender_num", "stroke_flg", "liver_flg", "copd_flg",
    "chf_flg", "renal_flg", "mal_flg", "resp_flg",
    "sepsis_flg", "day_28_flg", "hosp_exp_flg", "censor_flg",
]

ALL_FEATURES = NUMERIC_FEATURES + FLAG_FEATURES


def load_synthetic_mimic(n_patients: int = 1776, seed: int = 42) -> pd.DataFrame:
    """Generate a MIMIC-II-like cohort with realistic feature distributions
    and latent patient subgroups, so clustering recovers meaningful structure.

    Returns a DataFrame of shape (n_patients, len(ALL_FEATURES)) with injected
    missingness, matching the spirit of the real 1776x46 IAC table.
    """
    rng = np.random.default_rng(seed)

    # Latent archetypes -> the "ground-truth" phenotypes we hope to recover.
    # weights, means roughly chosen to echo the tutorial's annotated clusters.
    archetypes = [
        # name, prob, age, saps, sofa, comorbid_boost, mortality
        ("young_stable",        0.28, 42, 12, 3, 0.05, 0.04),
        ("elderly_cardiac",     0.20, 74, 20, 7, 0.45, 0.22),
        ("septic_critical",     0.16, 63, 28, 11, 0.55, 0.38),
        ("liver_sofa_high",     0.12, 55, 22, 9, 0.40, 0.25),
        ("respiratory_copd",    0.14, 68, 18, 6, 0.50, 0.20),
        ("metabolic_obese",     0.10, 58, 16, 5, 0.30, 0.12),
    ]
    names = [a[0] for a in archetypes]
    probs = np.array([a[1] for a in archetypes])
    probs = probs / probs.sum()
    labels = rng.choice(len(archetypes), size=n_patients, p=probs)

    rows = []
    for li in labels:
        _, _, age_m, saps_m, sofa_m, comorb, mort = archetypes[li]
        age = np.clip(rng.normal(age_m, 12), 18, 95)
        weight = np.clip(rng.normal(80 if li == 5 else 72, 18), 40, 160)
        height_m = rng.normal(1.70, 0.09)
        bmi = np.clip(weight / (height_m ** 2), 14, 55)
        sapsi = np.clip(rng.normal(saps_m, 5), 0, 50)
        sofa = np.clip(rng.normal(sofa_m, 2.5), 0, 24)

        row = {
            "age": age,
            "weight_first": weight,
            "bmi": bmi,
            "sapsi_first": sapsi,
            "sofa_first": sofa,
            "hr_1st": np.clip(rng.normal(88 + comorb * 20, 18), 40, 180),
            "temp_1st": np.clip(rng.normal(36.8 + (0.8 if li == 2 else 0), 0.8), 33, 41),
            "map_1st": np.clip(rng.normal(82 - comorb * 12, 14), 40, 130),
            "spo2_1st": np.clip(rng.normal(97 - comorb * 5, 3), 70, 100),
            "wbc_first": np.clip(rng.normal(11 + comorb * 6, 5), 1, 50),
            "hgb_first": np.clip(rng.normal(12.5 - comorb * 1.5, 2), 5, 19),
            "platelet_first": np.clip(rng.normal(240 - comorb * 60, 90), 10, 700),
            "sodium_first": np.clip(rng.normal(139, 4), 120, 160),
            "potassium_first": np.clip(rng.normal(4.1, 0.6), 2.5, 7),
            "creatinine_first": np.clip(rng.lognormal(0.0 + comorb, 0.5), 0.3, 12),
            "po2_first": np.clip(rng.normal(110 - comorb * 30, 45), 30, 500),
            "pco2_first": np.clip(rng.normal(40 + comorb * 8, 10), 20, 90),
            "iv_day_1": np.clip(rng.lognormal(7.5 + comorb, 0.7), 50, 20000),
            "icu_los_day": np.clip(rng.lognormal(0.9 + comorb, 0.7), 0.2, 60),
            # flags
            "gender_num": int(rng.random() < 0.55),
            "stroke_flg": int(rng.random() < (0.30 if li == 1 else 0.06)),
            "liver_flg": int(rng.random() < (0.45 if li == 3 else 0.05)),
            "copd_flg": int(rng.random() < (0.40 if li == 4 else 0.07)),
            "chf_flg": int(rng.random() < (0.45 if li == 1 else 0.10)),
            "renal_flg": int(rng.random() < (0.35 if li in (2, 3) else 0.08)),
            "mal_flg": int(rng.random() < (0.25 if li == 1 else 0.06)),
            "resp_flg": int(rng.random() < (0.55 if li == 4 else 0.15)),
            "sepsis_flg": int(rng.random() < (0.70 if li == 2 else 0.10)),
            "day_28_flg": int(rng.random() < mort),
            "hosp_exp_flg": int(rng.random() < mort * 0.9),
            "censor_flg": int(rng.random() < 0.5),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=ALL_FEATURES)

    # Inject realistic missingness (BMI hit hardest, like the tutorial's ~27%).
    miss_rates = {"bmi": 0.27, "po2_first": 0.18, "pco2_first": 0.18,
                  "sofa_first": 0.10, "sapsi_first": 0.08}
    for col, rate in miss_rates.items():
        mask = rng.random(n_patients) < rate
        df.loc[mask, col] = np.nan

    # Keep a hidden ground-truth column for optional validation (not used in
    # clustering). Notebook/dashboard ignore it for the unsupervised task.
    df["_true_phenotype"] = [names[i] for i in labels]
    return df


def load_real_mimic() -> pd.DataFrame | None:
    """Attempt to load the real MIMIC-II IAC dataset via ehrapy/ehrdata.

    Returns a feature DataFrame on success, or None if the libraries or
    credentialed data are unavailable (so callers can fall back to synthetic).
    """
    try:
        import ehrapy as ep  # noqa: F401
        import ehrdata as ed

        edata = ed.dt.mimic_2()
        df = edata.to_df() if hasattr(edata, "to_df") else pd.DataFrame(
            edata.X, columns=list(edata.var_names)
        )
        return df
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 2. Pipeline                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class PhenotypeResult:
    """Everything the dashboard and notebook need to render results."""
    df_raw: pd.DataFrame
    feature_matrix: pd.DataFrame          # imputed + normalized features used
    pca_coords: np.ndarray                # (n, n_pcs)
    embedding: np.ndarray                 # (n, 2) UMAP or PCA-2D
    embedding_name: str                   # "UMAP" or "PCA"
    labels: np.ndarray                    # cluster assignment per patient
    silhouette: float
    cluster_profile: pd.DataFrame         # mean per feature per cluster
    annotations: dict                     # cluster_id -> human label
    missing_pct: pd.Series                # per-feature missing % (pre-impute)
    explained_var: np.ndarray             # PCA explained variance ratio
    extra: dict = field(default_factory=dict)


def compute_missingness(df: pd.DataFrame, features: list[str]) -> pd.Series:
    return (df[features].isna().mean() * 100).sort_values(ascending=False)


def run_pipeline(
    df: pd.DataFrame,
    n_clusters: int = 6,
    n_pcs: int = 10,
    knn_neighbors: int = 5,
    use_umap: bool = True,
    random_state: int = 42,
) -> PhenotypeResult:
    """Run the full phenotyping pipeline and return a PhenotypeResult.

    Steps mirror the ehrapy tutorial: impute -> normalize -> PCA -> embed
    -> cluster -> profile -> annotate.
    """
    feats = [c for c in ALL_FEATURES if c in df.columns]
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]

    missing_pct = compute_missingness(df, feats)

    # --- KNN imputation (numeric only; flags treated as already-known 0/1) ---
    X = df[feats].copy()
    imputer = KNNImputer(n_neighbors=knn_neighbors)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feats, index=df.index)

    # Log-normalize the high-spread features, echoing iv_day_1 / po2_first.
    for col in ("iv_day_1", "po2_first"):
        if col in X_imputed.columns:
            X_imputed[col] = np.log1p(X_imputed[col].clip(lower=0))

    # --- Standardize ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed[feats])

    # --- PCA ---
    n_pcs = int(min(n_pcs, X_scaled.shape[1]))
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)

    # --- Embedding for visualization ---
    if use_umap and _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=random_state)
        embedding = reducer.fit_transform(pca_coords)
        emb_name = "UMAP"
    else:
        embedding = pca_coords[:, :2]
        emb_name = "PCA"

    # --- Clustering (KMeans stands in for Leiden; deterministic & dependency-free) ---
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(pca_coords)
    sil = float(silhouette_score(pca_coords, labels)) if n_clusters > 1 else float("nan")

    # --- Cluster profiling (mean of original-scale features per cluster) ---
    profile_src = X_imputed.copy()
    profile_src["cluster"] = labels
    cluster_profile = profile_src.groupby("cluster")[feats].mean()

    # --- Auto-annotation from the most distinctive features per cluster ---
    annotations = _auto_annotate(cluster_profile, numeric, FLAG_FEATURES)

    return PhenotypeResult(
        df_raw=df,
        feature_matrix=X_imputed,
        pca_coords=pca_coords,
        embedding=np.asarray(embedding),
        embedding_name=emb_name,
        labels=labels,
        silhouette=sil,
        cluster_profile=cluster_profile,
        annotations=annotations,
        missing_pct=missing_pct,
        explained_var=pca.explained_variance_ratio_,
    )


def _auto_annotate(profile: pd.DataFrame, numeric: list[str],
                   flags: list[str]) -> dict:
    """Label each cluster by its most distinctive features vs. the cohort mean."""
    z = (profile - profile.mean()) / (profile.std(ddof=0) + 1e-9)
    pretty = {
        "age": "elderly", "sofa_first": "SOFA+", "sapsi_first": "SAPS+",
        "creatinine_first": "renal+", "bmi": "obese", "weight_first": "obese",
        "platelet_first": "platelet", "po2_first": "hypoxic",
        "liver_flg": "liver", "copd_flg": "COPD", "stroke_flg": "stroke",
        "sepsis_flg": "septic", "chf_flg": "cardiac", "day_28_flg": "high-mortality",
        "hosp_exp_flg": "deceased", "resp_flg": "respiratory", "mal_flg": "malignancy",
        "hr_1st": "tachycardic", "iv_day_1": "high-fluids",
    }
    annotations = {}
    for cid in profile.index:
        scores = z.loc[cid].drop(labels=[c for c in z.columns if c not in pretty],
                                 errors="ignore")
        top = scores.sort_values(ascending=False).head(4)
        tags = []
        for f in top.index:
            if top[f] > 0.4 and f in pretty and pretty[f] not in tags:
                tags.append(pretty[f])
            if len(tags) == 3:
                break
        annotations[int(cid)] = "/".join(tags) if tags else f"cluster {cid}"
    return annotations


if __name__ == "__main__":
    # Smoke test
    d = load_synthetic_mimic(n_patients=600)
    res = run_pipeline(d, n_clusters=6, use_umap=False)
    print("Silhouette:", round(res.silhouette, 3))
    print("Annotations:", res.annotations)
    print("Embedding:", res.embedding.shape, res.embedding_name)
