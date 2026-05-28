"""
Streamlit dashboard for MIMIC-II disease phenotyping.

Run locally:
    pip install -r requirements.txt
    streamlit run dashboard/app.py
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make src/ importable whether launched from repo root or dashboard/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.phenotyping import (  # noqa: E402
    load_synthetic_mimic, load_real_mimic, run_pipeline,
    NUMERIC_FEATURES, FLAG_FEATURES,
)

# --------------------------------------------------------------------------- #
# Page config + theme                                                         #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MIMIC-II Phenotyping",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#1FA6C9", "#54C285", "#F4CC47", "#E2725B",
           "#9B5DE5", "#57C8B9", "#FF7F50", "#8D99AE"]

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
      html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
      .main { background: radial-gradient(circle at 15% 0%, #0d2030 0%, #0a1822 55%, #081019 100%); }
      h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; color: #eaf4f7; }
      .hero-title { font-family:'Fraunces',serif; font-size:2.6rem; font-weight:600;
                    background:linear-gradient(92deg,#54C285,#1FA6C9 55%,#57C8B9);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    margin-bottom:0.1rem; }
      .hero-sub { color:#8fb2c0; font-size:1.02rem; margin-top:0; font-family:'IBM Plex Mono',monospace; }
      .metric-card { background:rgba(31,166,201,0.07); border:1px solid rgba(84,194,133,0.22);
                     border-radius:14px; padding:1rem 1.2rem; }
      .metric-val { font-family:'Fraunces',serif; font-size:2rem; color:#54C285; font-weight:600; }
      .metric-lbl { color:#8fb2c0; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; }
      .chip { display:inline-block; padding:3px 11px; margin:2px; border-radius:999px;
              font-family:'IBM Plex Mono',monospace; font-size:0.78rem; border:1px solid; }
      .stTabs [data-baseweb="tab-list"] { gap:4px; }
      .stTabs [data-baseweb="tab"] { background:rgba(255,255,255,0.03); border-radius:10px 10px 0 0;
                                     padding:8px 18px; }
      [data-testid="stSidebar"] { background:#081019; border-right:1px solid rgba(84,194,133,0.15); }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data + pipeline (cached)                                                    #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_data(source: str, n_patients: int, seed: int) -> pd.DataFrame:
    if source == "Real MIMIC-II (ehrapy)":
        df = load_real_mimic()
        if df is not None:
            return df
        st.sidebar.warning("ehrapy/MIMIC-II unavailable — using synthetic data.")
    return load_synthetic_mimic(n_patients=n_patients, seed=seed)


@st.cache_data(show_spinner=False)
def get_result(df_hash_key: str, df: pd.DataFrame, k: int, n_pcs: int,
               knn: int, use_umap: bool, seed: int):
    return run_pipeline(df, n_clusters=k, n_pcs=n_pcs,
                        knn_neighbors=knn, use_umap=use_umap, random_state=seed)


# --------------------------------------------------------------------------- #
# Sidebar controls                                                            #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Pipeline controls")
    source = st.radio("Data source",
                      ["Synthetic MIMIC-like", "Real MIMIC-II (ehrapy)"],
                      help="Real data needs credentialed PhysioNet access + ehrapy installed.")
    n_patients = st.slider("Cohort size (synthetic)", 300, 3000, 1776, step=100)
    k = st.slider("Number of clusters (k)", 2, 8, 6)
    n_pcs = st.slider("PCA components", 2, 18, 10)
    knn = st.slider("KNN imputation neighbors", 1, 15, 5)
    use_umap = st.toggle("Use UMAP embedding (if installed)", value=True)
    seed = st.number_input("Random seed", 0, 9999, 42)
    st.markdown("---")
    st.caption("Built for the EHR analysis coding exercise · ehrapy-style "
               "phenotyping reimplemented with scikit-learn.")

df = get_data(source, n_patients, seed)
res = get_result(f"{source}-{len(df)}-{k}-{n_pcs}-{knn}-{use_umap}-{seed}",
                 df, k, n_pcs, knn, use_umap, seed)

# --------------------------------------------------------------------------- #
# Header                                                                      #
# --------------------------------------------------------------------------- #
st.markdown('<div class="hero-title">Disease Phenotyping · MIMIC-II</div>',
            unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Unsupervised discovery of ICU patient subgroups '
            '— preprocess → impute → PCA → embed → cluster → annotate</p>',
            unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{len(df):,}", "Patients"),
    (c2, f"{len(NUMERIC_FEATURES)+len(FLAG_FEATURES)}", "Features"),
    (c3, f"{k}", "Phenotypes"),
    (c4, f"{res.silhouette:.3f}", "Silhouette"),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                 f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.write("")
tab_map, tab_clusters, tab_features, tab_qc, tab_data = st.tabs(
    ["🗺️ Phenotype Map", "🧬 Cluster Profiles", "📊 Feature Drivers",
     "🩺 Data Quality", "🗃️ Cohort Table"])

label_names = {c: f"{c}: {res.annotations[c]}" for c in sorted(res.annotations)}
plot_df = pd.DataFrame({
    "x": res.embedding[:, 0], "y": res.embedding[:, 1],
    "cluster": [label_names[c] for c in res.labels],
    "age": df["age"].values,
    "mortality": df["day_28_flg"].values if "day_28_flg" in df else 0,
})

# --- Tab 1: embedding map -------------------------------------------------- #
with tab_map:
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        fig = px.scatter(
            plot_df, x="x", y="y", color="cluster",
            color_discrete_sequence=PALETTE,
            title=f"{res.embedding_name} embedding coloured by phenotype",
            labels={"x": f"{res.embedding_name} 1", "y": f"{res.embedding_name} 2"},
        )
        fig.update_traces(marker=dict(size=6, opacity=0.78,
                                      line=dict(width=0.4, color="#0a1822")))
        fig.update_layout(template="plotly_dark", height=560,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        color_by = st.selectbox("Overlay a clinical variable",
                                ["age", "mortality"] +
                                [c for c in NUMERIC_FEATURES if c in df.columns])
        vals = df[color_by].values if color_by in df else plot_df[color_by].values
        fig2 = px.scatter(x=res.embedding[:, 0], y=res.embedding[:, 1],
                          color=vals, color_continuous_scale="Tealgrn",
                          labels={"x": "", "y": "", "color": color_by})
        fig2.update_traces(marker=dict(size=6, opacity=0.8))
        fig2.update_layout(template="plotly_dark", height=560,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           title=f"Coloured by {color_by}")
        st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: cluster profiles ---------------------------------------------- #
with tab_clusters:
    st.markdown("#### Phenotype cards")
    sizes = pd.Series(res.labels).value_counts().sort_index()
    cols = st.columns(min(3, k))
    for i, cid in enumerate(sorted(res.annotations)):
        with cols[i % len(cols)]:
            color = PALETTE[cid % len(PALETTE)]
            tags = res.annotations[cid].split("/")
            chips = "".join(
                f'<span class="chip" style="color:{color};border-color:{color}">{t}</span>'
                for t in tags)
            mort = df.loc[res.labels == cid, "day_28_flg"].mean() * 100 if "day_28_flg" in df else 0
            age = df.loc[res.labels == cid, "age"].mean()
            st.markdown(
                f'<div class="metric-card" style="border-color:{color}55">'
                f'<div style="font-family:Fraunces;font-size:1.3rem;color:{color}">'
                f'Cluster {cid}</div>{chips}'
                f'<div style="margin-top:8px;color:#8fb2c0;font-size:0.85rem">'
                f'n = {sizes.get(cid,0)} · mean age {age:.0f} · '
                f'28-day mortality {mort:.0f}%</div></div>',
                unsafe_allow_html=True)
            st.write("")

    st.markdown("#### Cluster × feature heatmap (z-scored)")
    prof = res.cluster_profile.copy()
    z = (prof - prof.mean()) / (prof.std(ddof=0) + 1e-9)
    z.index = [f"{c}: {res.annotations[c]}" for c in z.index]
    heat = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index,
                                colorscale="RdBu_r", zmid=0,
                                colorbar=dict(title="z")))
    heat.update_layout(template="plotly_dark", height=420,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis=dict(tickangle=-45))
    st.plotly_chart(heat, use_container_width=True)

# --- Tab 3: feature drivers ------------------------------------------------ #
with tab_features:
    sel = st.selectbox("Inspect a cluster",
                       sorted(res.annotations),
                       format_func=lambda c: f"Cluster {c}: {res.annotations[c]}")
    prof = res.cluster_profile
    z = (prof - prof.mean()) / (prof.std(ddof=0) + 1e-9)
    row = z.loc[sel].sort_values()
    drivers = pd.concat([row.head(8), row.tail(8)])
    bar = px.bar(x=drivers.values, y=drivers.index, orientation="h",
                 color=drivers.values, color_continuous_scale="RdBu_r",
                 labels={"x": "z-score vs cohort", "y": ""})
    bar.update_layout(template="plotly_dark", height=520,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=f"What defines Cluster {sel}",
                      coloraxis_showscale=False)
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("#### PCA explained variance")
    ev = res.explained_var
    evfig = px.bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev * 100,
                   labels={"x": "", "y": "% variance"})
    evfig.update_traces(marker_color="#54C285")
    evfig.update_layout(template="plotly_dark", height=300,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(evfig, use_container_width=True)

# --- Tab 4: data quality --------------------------------------------------- #
with tab_qc:
    st.markdown("#### Missing values before imputation")
    miss = res.missing_pct[res.missing_pct > 0]
    if len(miss):
        mfig = px.bar(x=miss.values, y=miss.index, orientation="h",
                      labels={"x": "% missing", "y": ""})
        mfig.update_traces(marker_color="#E2725B")
        mfig.update_layout(template="plotly_dark", height=380,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(mfig, use_container_width=True)
        st.caption(f"Imputed with KNN (k={knn}). Highest: {miss.index[0]} "
                   f"at {miss.iloc[0]:.1f}%.")
    else:
        st.success("No missing values detected in this cohort.")

    st.markdown("#### Feature distributions")
    feat = st.selectbox("Feature", [c for c in NUMERIC_FEATURES if c in df.columns])
    dist = px.histogram(df, x=feat, nbins=40, color_discrete_sequence=["#1FA6C9"])
    dist.update_layout(template="plotly_dark", height=320,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(dist, use_container_width=True)

# --- Tab 5: data table ----------------------------------------------------- #
with tab_data:
    show = df.copy()
    show.insert(0, "cluster", res.labels)
    show.insert(1, "phenotype", [res.annotations[c] for c in res.labels])
    st.dataframe(show, use_container_width=True, height=560)
    st.download_button("⬇️ Download labelled cohort (CSV)",
                       show.to_csv(index=False).encode(),
                       "mimic_phenotyped.csv", "text/csv")
