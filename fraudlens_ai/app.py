from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="FraudLens AI | Transaction Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Design system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e8edf7; --muted:#9ba9c2; --panel:#111c2e; --panel2:#16243a; --line:#263752; --teal:#5eead4; --red:#fb7185; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(ellipse at 10% 0%, #152c45 0%, #0b1220 42%, #080d17 100%); color:var(--ink); }
.block-container { padding-top:1.8rem; padding-bottom:3rem; max-width: 1500px; }
[data-testid="stSidebar"] { background:#0b1422; border-right:1px solid #263752; }
[data-testid="stSidebar"] * { color:#dce6f7; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; }
h1 { font-size:2.35rem !important; }
.hero { padding: 1.25rem 1.4rem; border:1px solid #29415d; border-radius:18px; background:linear-gradient(120deg,rgba(23,44,68,.96),rgba(15,26,43,.88)); margin-bottom:1.15rem; }
.eyebrow { color:#5eead4; font-size:.74rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; line-height:1.15; margin:.35rem 0 .5rem; color:#f4f7ff; }
.hero-copy { color:#aab9d0; font-size:.98rem; max-width:850px; }
.metric-card { background:linear-gradient(145deg,#15243a,#101a2a); border:1px solid #263a55; border-radius:15px; padding:1rem 1.05rem; min-height:115px; }
.metric-label { color:#9ba9c2; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
.metric-value { color:#f4f7ff; font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; margin:.35rem 0 .2rem; }
.metric-note { color:#92a4bf; font-size:.78rem; }
.section-label { color:#5eead4; font-size:.75rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin:.3rem 0 .35rem; }
.info-card { background:rgba(17,28,46,.82); border:1px solid #263752; border-radius:14px; padding:1rem 1.1rem; height:100%; }
.info-title { color:#f4f7ff; font-weight:700; font-size:1rem; margin-bottom:.35rem; }
.info-copy { color:#a9b7ce; font-size:.88rem; line-height:1.5; }
div[data-testid="stMetric"] { background:#111c2e; border:1px solid #263752; border-radius:13px; padding:14px 16px; }
div[data-testid="stMetricLabel"] { color:#9ba9c2; }
div[data-testid="stMetricValue"] { color:#f4f7ff; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid #2d5870; background:#15364a; color:#e8ffff; font-weight:700; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#5eead4; color:#fff; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#111c2e; border-radius:8px; padding:8px 14px; }
.small-muted { color:#9ba9c2; font-size:.82rem; }
.pill { display:inline-block; border:1px solid #31516c; border-radius:999px; padding:4px 9px; font-size:.72rem; color:#b8c9df; margin-right:5px; }
hr { border-color:#263752; }
</style>
""", unsafe_allow_html=True)

FEATURES = ["amount", "hour", "account_age_days", "transactions_24h", "distance_km", "device_risk", "merchant_risk", "international", "card_present"]
FEATURE_LABELS = {
    "amount": "Transaction amount ($)",
    "hour": "Hour of day",
    "account_age_days": "Account age (days)",
    "transactions_24h": "Transactions in last 24h",
    "distance_km": "Distance from usual location (km)",
    "device_risk": "Device risk score",
    "merchant_risk": "Merchant risk score",
    "international": "International transaction",
    "card_present": "Card present",
}


def make_demo_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate explicitly simulated transactions for a no-upload demo."""
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(mean=3.35, sigma=1.0, size=n), 2, 3500)
    hour = rng.integers(0, 24, n)
    account_age = np.clip(rng.gamma(shape=3.2, scale=240, size=n), 1, 4500).astype(int)
    tx24 = np.clip(rng.poisson(lam=2.3, size=n), 0, 22)
    distance = np.clip(rng.exponential(scale=18, size=n), 0, 900)
    device = rng.beta(1.8, 7.0, n)
    merchant = rng.beta(2.0, 5.5, n)
    international = rng.binomial(1, .12, n)
    card_present = rng.binomial(1, .34, n)
    # Synthetic label logic: intentionally simulated, not a real fraud process.
    logit = (
        -5.05 + .00085 * amount + .075 * tx24 + .0023 * distance
        + 2.0 * device + 1.65 * merchant + .85 * international
        + .9 * (hour <= 4) + .8 * (account_age < 30)
        + .45 * (1 - card_present) + .0000005 * amount * distance
    )
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)
    df = pd.DataFrame({
        "amount": amount.round(2), "hour": hour, "account_age_days": account_age,
        "transactions_24h": tx24, "distance_km": distance.round(2),
        "device_risk": device.round(3), "merchant_risk": merchant.round(3),
        "international": international, "card_present": card_present, "is_fraud": fraud,
    })
    return df


def normalize_uploaded_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Map common IEEE-CIS columns or simple custom CSV columns to model features."""
    df = df.copy()
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    target_candidates = ["is_fraud", "isfraud", "fraud", "target", "label"]
    target_col = next((lower_to_original[c] for c in target_candidates if c in lower_to_original), None)
    if target_col is None:
        raise ValueError("Could not find a fraud label column. Expected isFraud, is_fraud, fraud, target, or label.")

    mapping = {
        "amount": ["amount", "transactionamt", "transaction_amt"],
        "hour": ["hour", "transaction_hour"],
        "account_age_days": ["account_age_days"],
        "transactions_24h": ["transactions_24h"],
        "distance_km": ["distance_km"],
        "device_risk": ["device_risk"],
        "merchant_risk": ["merchant_risk"],
        "international": ["international"],
        "card_present": ["card_present"],
    }
    selected: dict[str, Any] = {}
    for feature, candidates in mapping.items():
        found = next((lower_to_original[c] for c in candidates if c in lower_to_original), None)
        if found is not None:
            selected[feature] = pd.to_numeric(df[found], errors="coerce")
    # IEEE-CIS-compatible feature selection; no TransactionDT is treated as a real timestamp.
    ieee_map = {
        "amount": "TransactionAmt", "hour": "TransactionDT", "device_risk": "DeviceType",
        "merchant_risk": "ProductCD", "international": "P_emaildomain", "card_present": "card4",
    }
    for feature, original_name in ieee_map.items():
        if feature in selected:
            continue
        col = lower_to_original.get(original_name.lower())
        if col is not None:
            if feature in ["device_risk", "merchant_risk", "international", "card_present"]:
                selected[feature] = df[col].astype("string").fillna("Unknown")
            else:
                selected[feature] = pd.to_numeric(df[col], errors="coerce")
    for col in ["card1", "card2", "card3", "card5", "addr1", "dist1", "C1", "C2", "D1", "V1", "V2"]:
        source = lower_to_original.get(col.lower())
        if source is not None and col.lower() not in selected:
            selected[col.lower()] = pd.to_numeric(df[source], errors="coerce")
    if "amount" not in selected:
        raise ValueError("Could not identify an amount field. Expected amount or TransactionAmt.")
    features = pd.DataFrame(selected)
    features = features.replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df[target_col], errors="coerce")
    keep = y.isin([0, 1])
    features = features.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).reset_index(drop=True)
    features["is_fraud"] = y
    features = features.dropna(subset=["amount"])
    if len(features) < 100:
        raise ValueError("Please upload at least 100 labeled rows for a meaningful demo.")
    if features["is_fraud"].nunique() < 2:
        raise ValueError("The uploaded sample must contain both legitimate and fraudulent labels.")
    return features, f"Uploaded labeled data ({len(features):,} rows)"


def fit_model(df: pd.DataFrame):
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    # Keep a reasonable runtime on free cloud hosting.
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42).reset_index(drop=True)
        X = df.drop(columns=["is_fraud"])
        y = df["is_fraud"].astype(int)
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    # Balanced class weights help with the naturally imbalanced target.
    model = RandomForestClassifier(
        n_estimators=140, max_depth=12, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.25, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    pred = (probs >= .5).astype(int)
    metrics = {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, probs) if y_test.nunique() == 2 else np.nan,
        "roc_auc": roc_auc_score(y_test, probs) if y_test.nunique() == 2 else np.nan,
        "cm": confusion_matrix(y_test, pred, labels=[0, 1]),
        "y_test": y_test.reset_index(drop=True),
        "probs": probs,
        "pred": pred,
        "test_rows": len(y_test),
        "train_rows": len(y_train),
    }
    return pipe, metrics, list(X.columns)


def score_label(p: float) -> tuple[str, str]:
    if p >= .70:
        return "HIGH RISK", "#fb7185"
    if p >= .35:
        return "REVIEW", "#fbbf24"
    return "LOW RISK", "#5eead4"


def metric_card(label: str, value: str, note: str) -> str:
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>'


def make_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------- Sidebar / data mode ----------
st.sidebar.markdown("<div class='eyebrow'>RISK INTELLIGENCE</div><h2 style='margin-top:.2rem'>FraudLens AI</h2><div class='small-muted'>Transaction risk screening prototype</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Workspace", ["Overview", "Transaction Scoring", "Model Performance", "Data Explorer", "Methodology"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**Data source**")
upload = st.sidebar.file_uploader("Upload a labeled transaction CSV (optional)", type=["csv"], help="Supports a compact custom CSV or an IEEE-CIS-style sample with TransactionAmt and isFraud columns. For large Kaggle files, create a smaller sample first.")

@st.cache_data(show_spinner=False)
def get_demo() -> pd.DataFrame:
    return make_demo_data()

@st.cache_data(show_spinner=False)
def read_upload(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    normalized, _ = normalize_uploaded_data(raw)
    return normalized

@st.cache_resource(show_spinner="Preparing the risk model…")
def cached_model(data_fingerprint: str, data_json: str):
    df = pd.read_json(io.StringIO(data_json), orient="split")
    return fit_model(df)

if upload is not None:
    try:
        source_df = read_upload(upload.getvalue())
        source_name = f"Uploaded dataset · {len(source_df):,} rows"
        is_simulated = False
    except Exception as exc:
        st.sidebar.error(f"Upload issue: {exc}")
        source_df = get_demo()
        source_name = "Simulated demo data (fallback)"
        is_simulated = True
else:
    source_df = get_demo()
    source_name = "Simulated demo data"
    is_simulated = True

# Fit once per dataset version; serialize to make the cached input deterministic.
serialized = source_df.to_json(orient="split", date_format="iso")
fingerprint = str(pd.util.hash_pandas_object(source_df, index=True).sum())
model, metrics, feature_names = cached_model(fingerprint, serialized)

# Use a unified feature frame for the scorer.
X_all = source_df.drop(columns=["is_fraud"])
all_probs = model.predict_proba(X_all)[:, 1]
scored_df = source_df.copy()
scored_df["risk_probability"] = all_probs
scored_df["risk_band"] = pd.cut(all_probs, bins=[-0.001, .35, .70, 1.001], labels=["Low", "Review", "High"])

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">FINANCIAL CRIME ANALYTICS · ML WORKBENCH</div>
  <div class="hero-title">See risk before it becomes loss.</div>
  <div class="hero-copy">An explainable transaction-risk screening prototype combining data quality checks, imbalanced classification, risk prioritization, and model evaluation in one workspace.</div>
  <div style="margin-top:1rem"><span class="pill">Python</span><span class="pill">Scikit-learn</span><span class="pill">Plotly</span><span class="pill">Streamlit</span></div>
</div>
""", unsafe_allow_html=True)

if is_simulated:
    st.warning("**Demo mode:** this dashboard currently uses simulated transactions. Results demonstrate the workflow only; they are not real-world fraud rates or evidence of production model performance. Upload a labeled CSV to explore your own data.", icon="⚠️")
else:
    st.success(f"Active data source: {source_name}. Metrics below are evaluated on a random holdout split; for financial deployment, replace this with time-based validation.", icon="✓")

if page == "Overview":
    st.markdown("<div class='section-label'>Executive overview</div>", unsafe_allow_html=True)
    total = len(scored_df)
    fraud_rate = source_df["is_fraud"].mean() * 100
    high_count = int((all_probs >= .70).sum())
    avg_amount = float(source_df["amount"].mean())
    cols = st.columns(4)
    with cols[0]: st.markdown(metric_card("Transactions analyzed", f"{total:,}", "Rows in the active dataset"), unsafe_allow_html=True)
    with cols[1]: st.markdown(metric_card("Labeled fraud rate", f"{fraud_rate:.2f}%", "Observed label share in dataset"), unsafe_allow_html=True)
    with cols[2]: st.markdown(metric_card("High-risk flagged", f"{high_count:,}", "Model score ≥ 70%"), unsafe_allow_html=True)
    with cols[3]: st.markdown(metric_card("Average transaction", f"${avg_amount:,.2f}", "Across analyzed transactions"), unsafe_allow_html=True)

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("### Risk distribution")
        risk_counts = scored_df["risk_band"].value_counts().reindex(["Low", "Review", "High"], fill_value=0).rename_axis("Risk band").reset_index(name="Transactions")
        fig = px.bar(risk_counts, x="Risk band", y="Transactions", color="Risk band", color_discrete_map={"Low":"#5eead4", "Review":"#fbbf24", "High":"#fb7185"}, text_auto=True)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8,r=8,t=12,b=8), height=330, showlegend=False, xaxis_title="", yaxis_title="Transactions")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Model snapshot")
        m1, m2 = st.columns(2)
        m1.metric("PR-AUC", f"{metrics['pr_auc']:.3f}" if np.isfinite(metrics["pr_auc"]) else "N/A")
        m2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if np.isfinite(metrics["roc_auc"]) else "N/A")
        st.markdown(f"<div class='info-card'><div class='info-title'>Evaluation protocol</div><div class='info-copy'>Stratified holdout · {metrics['train_rows']:,} train rows · {metrics['test_rows']:,} test rows. Threshold = 0.50 for reported metrics. Adjust operational thresholds based on investigation capacity and false-positive cost.</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='info-card'><div class='info-title'>Decision support, not an automatic block</div><div class='info-copy'>Use model scores to prioritize human review. A probability score is not proof of fraud and should not be used as the sole basis for adverse decisions.</div></div>", unsafe_allow_html=True)

    st.markdown("### Latest high-risk cases")
    display_cols = [c for c in ["amount", "hour", "transactions_24h", "distance_km", "device_risk", "merchant_risk", "international", "card_present", "risk_probability", "risk_band", "is_fraud"] if c in scored_df.columns]
    top = scored_df.sort_values("risk_probability", ascending=False).head(10)[display_cols].copy()
    top["risk_probability"] = (top["risk_probability"] * 100).round(1).astype(str) + "%"
    if "is_fraud" in top: top["is_fraud"] = top["is_fraud"].map({0:"Legitimate",1:"Fraud"})
    st.dataframe(top, use_container_width=True, hide_index=True)

elif page == "Transaction Scoring":
    st.markdown("<div class='section-label'>Interactive case assessment</div>", unsafe_allow_html=True)
    st.header("Score a transaction")
    st.caption("Adjust transaction signals to see how the trained model changes its risk estimate. Inputs are a demonstration and should not be treated as a real payment authorization decision.")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Transaction amount ($)", min_value=1.0, max_value=10000.0, value=245.0, step=10.0)
        hour = st.slider("Hour of day", 0, 23, 2)
        account_age = st.number_input("Account age (days)", 1, 5000, 20)
    with col2:
        tx24 = st.slider("Transactions in last 24 hours", 0, 30, 8)
        distance = st.slider("Distance from usual location (km)", 0.0, 1000.0, 180.0, 5.0)
        device = st.slider("Device risk score", 0.0, 1.0, .65, .01)
    with col3:
        merchant = st.slider("Merchant risk score", 0.0, 1.0, .55, .01)
        international = st.selectbox("International transaction", [0, 1], format_func=lambda x: "Yes" if x else "No")
        card_present = st.selectbox("Card present", [1, 0], format_func=lambda x: "Yes" if x else "No")
    sample = {"amount": amount, "hour": hour, "account_age_days": account_age, "transactions_24h": tx24, "distance_km": distance, "device_risk": device, "merchant_risk": merchant, "international": international, "card_present": card_present}
    # Align to whichever feature schema the current model was trained on.
    scoring_values = {}
    for c in feature_names:
        # Uploaded datasets may contain categorical versions of a signal (e.g. DeviceType).
        # Use the training-set mode for those columns instead of passing a numeric slider value
        # into a string OneHotEncoder. This keeps the generic scorer robust across schemas.
        if c in X_all.columns and not pd.api.types.is_numeric_dtype(X_all[c]):
            mode = X_all[c].mode(dropna=True)
            scoring_values[c] = mode.iloc[0] if not mode.empty else "Unknown"
        elif c in sample:
            scoring_values[c] = sample[c]
        elif c in X_all.columns:
            scoring_values[c] = X_all[c].median()
        else:
            scoring_values[c] = 0
    scoring = pd.DataFrame([scoring_values], columns=feature_names)
    probability = float(model.predict_proba(scoring[feature_names])[:, 1][0])
    label, color = score_label(probability)
    r1, r2 = st.columns([1, 1.6])
    with r1:
        st.markdown(f"<div class='info-card' style='border-color:{color};'><div class='metric-label'>Estimated risk score</div><div class='metric-value' style='color:{color};font-size:2.6rem'>{probability:.1%}</div><div style='font-weight:700;color:{color};margin-top:.2rem'>{label}</div><div class='info-copy' style='margin-top:.8rem'>Illustrative model output. Thresholds are configurable and require calibration against operational costs.</div></div>", unsafe_allow_html=True)
    with r2:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=probability*100, number={"suffix":"%", "font":{"color":"#f4f7ff","size":28}}, gauge={"axis":{"range":[0,100],"tickcolor":"#9ba9c2"},"bar":{"color":color},"bgcolor":"#111c2e","bordercolor":"#263752","steps":[{"range":[0,35],"color":"#153c3b"},{"range":[35,70],"color":"#4b3c20"},{"range":[70,100],"color":"#4b2430"}]}, title={"text":"Risk probability · model estimate","font":{"color":"#a9b7ce","size":14}}))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=230, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(gauge, use_container_width=True)
    st.info("For uploaded datasets with a different feature schema, this form uses the available supported features and fills unavailable features with training-data defaults. Interpret those scores cautiously.")

elif page == "Model Performance":
    st.markdown("<div class='section-label'>Validation & model quality</div>", unsafe_allow_html=True)
    st.header("Model performance")
    st.caption("The default demonstration uses a random stratified holdout. For an actual fraud-detection study, prefer time-aware validation and tune the decision threshold against operational costs.")
    cols = st.columns(4)
    cols[0].metric("Precision @ 0.50", f"{metrics['precision']:.3f}")
    cols[1].metric("Recall @ 0.50", f"{metrics['recall']:.3f}")
    cols[2].metric("PR-AUC", f"{metrics['pr_auc']:.3f}" if np.isfinite(metrics["pr_auc"]) else "N/A")
    cols[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if np.isfinite(metrics["roc_auc"]) else "N/A")
    cm = metrics["cm"]
    cm_fig = px.imshow(cm, text_auto=True, x=["Predicted legitimate", "Predicted fraud"], y=["Actual legitimate", "Actual fraud"], color_continuous_scale=[[0,"#15243a"],[1,"#5eead4"]], aspect="auto")
    cm_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360, coloraxis_showscale=False, margin=dict(l=10,r=10,t=20,b=10))
    st.markdown("### Confusion matrix")
    st.plotly_chart(cm_fig, use_container_width=True)
    st.markdown("### Threshold trade-off")
    y_true = np.asarray(metrics["y_test"])
    probs = np.asarray(metrics["probs"])
    rows = []
    for threshold in np.arange(.1, .91, .05):
        pred = (probs >= threshold).astype(int)
        rows.append({"Threshold": round(float(threshold),2), "Precision": precision_score(y_true,pred,zero_division=0), "Recall": recall_score(y_true,pred,zero_division=0), "Flagged share": float(pred.mean())})
    curve_df = pd.DataFrame(rows)
    curve_fig = px.line(curve_df, x="Threshold", y=["Precision", "Recall", "Flagged share"], markers=True, color_discrete_sequence=["#5eead4", "#fb7185", "#fbbf24"])
    curve_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360, yaxis=dict(range=[0,1]), margin=dict(l=8,r=8,t=20,b=8), legend_title="Metric")
    st.plotly_chart(curve_fig, use_container_width=True)
    st.markdown("<div class='info-card'><div class='info-title'>How to read these metrics</div><div class='info-copy'><b>Precision</b> asks how many flagged transactions are actually fraud. <b>Recall</b> asks how much of the fraud the model catches. <b>PR-AUC</b> is especially informative for imbalanced classification. No single threshold is universally best; the choice depends on investigation capacity and the relative costs of missed fraud versus false alarms.</div></div>", unsafe_allow_html=True)

elif page == "Data Explorer":
    st.markdown("<div class='section-label'>Dataset quality & patterns</div>", unsafe_allow_html=True)
    st.header("Explore transactions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(source_df):,}")
    c2.metric("Features", str(len(source_df.columns)-1))
    c3.metric("Missing cells", f"{int(source_df.isna().sum().sum()):,}")
    st.markdown("### Fraud rate by transaction amount")
    temp = source_df.copy()
    temp["amount_bucket"] = pd.qcut(temp["amount"], q=min(8, temp["amount"].nunique()), duplicates="drop")
    amount_rates = temp.groupby("amount_bucket", observed=True)["is_fraud"].agg(["mean", "count"]).reset_index()
    amount_rates["amount_bucket"] = amount_rates["amount_bucket"].astype(str)
    amount_rates["fraud_pct"] = amount_rates["mean"]*100
    fig = px.bar(amount_rates, x="amount_bucket", y="fraud_pct", hover_data={"count":True,"mean":False}, color="fraud_pct", color_continuous_scale=[[0,"#1f5664"],[1,"#fb7185"]], labels={"amount_bucket":"Transaction amount bucket","fraud_pct":"Labeled fraud (%)"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340, margin=dict(l=8,r=8,t=15,b=8), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Sample records")
    st.dataframe(scored_df.head(250), use_container_width=True, hide_index=True)
    st.download_button("Download scored transactions CSV", make_download(scored_df), file_name="fraudlens_scored_transactions.csv", mime="text/csv")
    if is_simulated:
        st.download_button("Download demo training data CSV", make_download(source_df), file_name="fraudlens_simulated_transactions.csv", mime="text/csv")

else:
    st.markdown("<div class='section-label'>Project documentation</div>", unsafe_allow_html=True)
    st.header("Methodology & responsible use")
    st.markdown("""
    <div class='info-card'>
      <div class='info-title'>1 · Data contract</div>
      <div class='info-copy'>The app expects a binary target named <code>isFraud</code>, <code>is_fraud</code>, <code>fraud</code>, <code>target</code>, or <code>label</code>, plus a transaction amount field named <code>TransactionAmt</code> or <code>amount</code>. It supports a compact feature set from custom CSVs and a subset of IEEE-CIS columns.</div>
    </div><br>
    <div class='info-card'>
      <div class='info-title'>2 · Modelling approach</div>
      <div class='info-copy'>A scikit-learn preprocessing pipeline imputes missing values, encodes categorical variables, and trains a class-weighted Random Forest. A held-out split is used to calculate precision, recall, PR-AUC, ROC-AUC, and a confusion matrix. The threshold can be explored independently of the model.</div>
    </div><br>
    <div class='info-card'>
      <div class='info-title'>3 · Limitations</div>
      <div class='info-copy'>The default dataset is synthetic and must never be presented as real financial data. Uploaded data uses a random holdout split, which may not reflect future deployment. Production-grade work should use time-based validation, calibration, leakage checks, subgroup/fairness analysis, drift monitoring, and cost-sensitive threshold selection.</div>
    </div><br>
    <div class='info-card'>
      <div class='info-title'>4 · Responsible use</div>
      <div class='info-copy'>A risk score is a prioritization signal, not proof of misconduct. Keep a human review process, protect personal information, document the model's limitations, and do not connect this prototype to a live payment authorization system.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Recommended next steps for a research-grade version")
    st.write("1. Use a chronological train/validation/test split when reliable timestamps are available.")
    st.write("2. Compare Logistic Regression, Random Forest, and gradient boosting against a simple baseline.")
    st.write("3. Add probability calibration and choose a threshold based on a defined review budget.")
    st.write("4. Add time-based drift monitoring and a reproducible experiment log.")
    st.write("5. Document the Kaggle competition rules and the dataset's permitted uses before redistribution.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"<div class='small-muted'>Active source<br><b>{source_name}</b></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='small-muted' style='margin-top:1rem'>Prototype for educational use · Not financial advice or a production fraud-control system.</div>", unsafe_allow_html=True)
