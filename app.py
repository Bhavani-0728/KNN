import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from datetime import datetime

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, r2_score, mean_squared_error
)

st.set_page_config(page_title="KNN Dashboard", page_icon="🤖", layout="wide")

# ── UI Theme ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Container */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1100px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stSlider > div { padding: 0 4px; }

/* Step headers */
.step-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 2.2rem 0 1.2rem 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #21262d;
}
.step-num {
    background: #1f6feb;
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
}
.step-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 0;
}

/* Cards */
.knn-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 14px; margin: 1rem 0; }
.metric-box {
    flex: 1;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    text-align: center;
}
.metric-box .m-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.metric-box .m-value {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1;
}

/* Buttons */
.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #388bfd !important;
}

/* Inputs */
.stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
.stRadio > div { gap: 12px; }
.stRadio label { color: #c9d1d9 !important; font-size: 0.9rem !important; }
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 10px;
    padding: 0.5rem;
}

/* Alerts */
.stSuccess {
    background: rgba(35,134,54,0.15) !important;
    border: 1px solid rgba(35,134,54,0.4) !important;
    border-radius: 8px !important;
    color: #3fb950 !important;
    font-size: 0.88rem !important;
}
.stError {
    background: rgba(248,81,73,0.12) !important;
    border: 1px solid rgba(248,81,73,0.35) !important;
    border-radius: 8px !important;
    color: #f85149 !important;
    font-size: 0.88rem !important;
}
.stInfo {
    background: rgba(31,111,235,0.12) !important;
    border: 1px solid rgba(31,111,235,0.35) !important;
    border-radius: 8px !important;
    color: #58a6ff !important;
    font-size: 0.88rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Labels */
label, p { color: #c9d1d9 !important; }
h1, h2, h3 { color: #e6edf3 !important; }

/* Subheader */
.stMarkdown h3 {
    font-size: 0.82rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #8b949e !important;
    margin-top: 1.4rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0; border-bottom: 1px solid #21262d; margin-bottom: 0.5rem;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                color:#1f6feb; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:10px;">
        ● Machine Learning
    </div>
    <h1 style="font-size:2.2rem; font-weight:700; color:#e6edf3; margin:0; line-height:1.2;">
        KNN Dashboard
    </h1>
    <p style="color:#8b949e; font-size:0.95rem; margin-top:8px; margin-bottom:0;">
        End-to-end K-Nearest Neighbours · Classification & Regression
    </p>
</div>
""", unsafe_allow_html=True)

def step(num, title):
    st.markdown(f"""
    <div class="step-header">
        <span class="step-num">STEP {num:02d}</span>
        <span class="step-title">{title}</span>
    </div>""", unsafe_allow_html=True)

def metric_card(label, value):
    return f"""
    <div class="metric-box">
        <div class="m-label">{label}</div>
        <div class="m-value">{value}</div>
    </div>"""

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#161b22",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
})

# ── Session state ─────────────────────────────────────────────────────────────
for k in ["df", "df_clean"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Dirs ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
RAW_DIR       = os.path.join(BASE, "data", "raw")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                color:#1f6feb; text-transform:uppercase; letter-spacing:0.12em;
                padding: 1rem 0 1.2rem 0; border-bottom:1px solid #21262d; margin-bottom:1rem;">
        ⚙ Model Settings
    </div>""", unsafe_allow_html=True)

    task_type   = st.selectbox("Task Type",       ["Classification", "Regression"])
    n_neighbors = st.slider("K Neighbors",         1, 20, 5)
    weights     = st.selectbox("Weights",          ["uniform", "distance"])
    algorithm   = st.selectbox("Algorithm",        ["auto", "ball_tree", "kd_tree", "brute"])
    metric      = st.selectbox("Distance Metric",  ["minkowski", "euclidean", "manhattan"])
    p           = st.slider("P Value",             1, 5, 2)
    leaf_size   = st.slider("Leaf Size",           10, 100, 30)

df = st.session_state.df

# ── Step 1 ────────────────────────────────────────────────────────────────────
step(1, "Data Ingestion")
option = st.radio("Source", ["Download Dataset", "Upload CSV"], horizontal=True)

if option == "Download Dataset":
    label = "Download Iris Dataset" if task_type == "Classification" else "Download Tips Dataset"
    url   = ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
             if task_type == "Classification" else
             "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
    fname = "iris.csv" if task_type == "Classification" else "tips.csv"
    if st.button(label):
        path = os.path.join(RAW_DIR, fname)
        with open(path, "wb") as f:
            f.write(requests.get(url).content)
        df = pd.read_csv(path)
        st.session_state.df = df
        st.success(f"✓ {fname} downloaded")

if option == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        path = os.path.join(RAW_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        df = pd.read_csv(path)
        st.session_state.df = df
        st.success(f"✓ {uploaded.name} loaded")

# ── Step 2 ────────────────────────────────────────────────────────────────────
if df is not None:
    step(2, "Exploratory Data Analysis")

    missing = int(df.isnull().sum().sum())
    st.markdown(f"""
    <div class="metric-row">
        {metric_card("Rows",    str(df.shape[0]))}
        {metric_card("Columns", str(df.shape[1]))}
        {metric_card("Missing", str(missing))}
        {metric_card("Numeric", str(len(df.select_dtypes(include=np.number).columns)))}
    </div>""", unsafe_allow_html=True)

    st.dataframe(df.head(6), use_container_width=True)

    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) > 1:
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
                    cmap="Blues", linewidths=0.5, linecolor="#21262d",
                    ax=ax, cbar_kws={"shrink": 0.75})
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Step 3 ────────────────────────────────────────────────────────────────────
if df is not None:
    step(3, "Data Cleaning")
    strategy = st.selectbox("Missing Value Strategy", ["Mean", "Median", "Drop Rows"])
    df_clean = df.copy()
    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number):
            fill = df_clean[col].mean() if strategy == "Mean" else df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(fill)
    st.session_state.df_clean = df_clean
    st.success(f"✓ Cleaned — {len(df_clean)} rows retained")

# ── Step 4 ────────────────────────────────────────────────────────────────────
step(4, "Save Cleaned Dataset")
if st.button("Save Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data found. Complete Step 3 first.")
    else:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(PROCESSED_DIR, f"cleaned_{ts}.csv")
        st.session_state.df_clean.to_csv(path, index=False)
        st.success(f"✓ Saved → data/processed/cleaned_{ts}.csv")

# ── Step 5 ────────────────────────────────────────────────────────────────────
step(5, "Load Dataset for Modelling")
files    = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("cleaned_")])
df_model = None
if files:
    selected = st.selectbox("Choose file", files)
    df_model = pd.read_csv(os.path.join(PROCESSED_DIR, selected))
    st.dataframe(df_model.head(5), use_container_width=True)
else:
    st.info("No cleaned files yet — complete Steps 3 & 4 first.")

# ── Step 6 ────────────────────────────────────────────────────────────────────
if df_model is not None:
    step(6, "Train KNN Model")
    target = st.selectbox("Target Column", df_model.columns)

    X = df_model.drop(columns=[target]).copy()
    y = df_model[target].copy()

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if task_type == "Classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X = X.select_dtypes(include=np.number)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    params = dict(n_neighbors=n_neighbors, weights=weights,
                  algorithm=algorithm, metric=metric, p=p, leaf_size=leaf_size)

    if st.button("▶  Train Model"):

        if task_type == "Classification":
            model  = KNeighborsClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)

            st.markdown(f"""
            <div class="metric-row">
                {metric_card("Accuracy",     f"{acc:.2%}")}
                {metric_card("Test Samples", str(len(y_test)))}
                {metric_card("K",            str(n_neighbors))}
            </div>""", unsafe_allow_html=True)

            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(confusion_matrix(y_test, y_pred),
                        annot=True, fmt="d", cmap="Blues",
                        linewidths=0.5, linecolor="#0d1117", ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

        else:
            model  = KNeighborsRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2     = r2_score(y_test, y_pred)
            mse    = mean_squared_error(y_test, y_pred)

            st.markdown(f"""
            <div class="metric-row">
                {metric_card("R² Score", f"{r2:.3f}")}
                {metric_card("MSE",      f"{mse:.3f}")}
                {metric_card("RMSE",     f"{np.sqrt(mse):.3f}")}
            </div>""", unsafe_allow_html=True)

            st.markdown("### Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.75, color="#58a6ff",
                       edgecolors="#0d1117", linewidths=0.4, s=55)
            mn = min(float(np.min(y_test)), float(np.min(y_pred)))
            mx = max(float(np.max(y_test)), float(np.max(y_pred)))
            ax.plot([mn, mx], [mn, mx], "--", color="#8b949e", lw=1.2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1.5rem; border-top:1px solid #21262d;
            font-family:'JetBrains Mono',monospace; font-size:0.68rem;
            color:#30363d; display:flex; justify-content:space-between;">
    <span>KNN Dashboard</span>
    <span>Classification · Regression</span>
</div>
""", unsafe_allow_html=True)