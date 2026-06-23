"""
IEEE Fraud Detection – Streamlit Demo App
=========================================
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import lightgbm as lgb
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IEEE Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path(__file__).parent / "model"
CHART_DIR = MODEL_DIR / "charts"

# ── Load model artefacts (cached) ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    booster      = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_fraud_model.txt"))
    feature_info = joblib.load(MODEL_DIR / "feature_info.pkl")
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return booster, feature_info, metrics


booster, feature_info, metrics = load_model()
FEAT_NAMES   = feature_info["feature_names"]
CAT_COLS     = feature_info["cat_cols"]
ENCODERS     = feature_info["encoders"]
TOP_FEATURES = feature_info["top_features"][:15]

# ── Sidebar – nav ──────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/fraud.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🔎 Single Transaction", "📦 Batch Prediction", "📊 Model Evaluation"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.metric("ROC-AUC",       f"{metrics['roc_auc']:.4f}")
st.sidebar.metric("F1 (fraud)",    f"{metrics['f1_fraud']:.4f}")
st.sidebar.metric("Avg Precision", f"{metrics['avg_precision']:.4f}")
st.sidebar.metric("Features",      metrics["n_features"])
st.sidebar.caption(f"Best iteration: {metrics['best_iteration']}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Decision Threshold")
st.sidebar.caption(
    "Lower → catch more fraud (more false alarms).  \n"
    "Higher → fewer false alarms (may miss fraud)."
)
THRESHOLD = st.sidebar.slider(
    "Fraud threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(round(feature_info["threshold"], 2)),
    step=0.01,
    format="%.2f",
)
st.sidebar.caption(f"Model-optimal threshold: **{metrics['threshold']:.4f}**")

# ══════════════════════════════════════════════════════════════════════════════
# Page helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict_row(row_dict: dict) -> tuple[float, int]:
    """Given a dict of raw feature values, return (fraud_prob, label)."""
    df = pd.DataFrame([row_dict])

    # fill missing features with NaN
    for col in FEAT_NAMES:
        if col not in df.columns:
            df[col] = np.nan

    # encode categoricals
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        le       = ENCODERS[col]
        known    = set(le.classes_)
        fallback = le.classes_[0]
        df[col] = df[col].apply(lambda x: x if x in known else fallback)
        df[col] = le.transform(df[col])

    # ensure correct column order
    df = df[FEAT_NAMES]
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")

    prob  = float(booster.predict(df, num_iteration=booster.best_iteration)[0])
    label = int(prob >= THRESHOLD)
    return prob, label


def fraud_gauge(prob: float):
    """Draw a simple gauge chart for fraud probability."""
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background bar
    ax.barh(0.5, 1.0, height=0.35, color="#e0e0e0", left=0, align="center")
    # Colour zones
    ax.barh(0.5, 0.33,   height=0.35, color="#4caf50", left=0.00, align="center", alpha=0.6)
    ax.barh(0.5, 0.34,   height=0.35, color="#ff9800", left=0.33, align="center", alpha=0.6)
    ax.barh(0.5, 0.33,   height=0.35, color="#f44336", left=0.67, align="center", alpha=0.6)
    # Probability marker
    ax.scatter([prob], [0.5], s=300, color="black", zorder=5)
    ax.axvline(THRESHOLD, color="black", lw=1.5, ls="--", alpha=0.6)

    ax.text(0.165, 0.12, "Low",    ha="center", fontsize=10, color="#4caf50", fontweight="bold")
    ax.text(0.50,  0.12, "Medium", ha="center", fontsize=10, color="#ff9800", fontweight="bold")
    ax.text(0.835, 0.12, "High",   ha="center", fontsize=10, color="#f44336", fontweight="bold")
    ax.text(THRESHOLD, 0.92, f"Threshold\n{THRESHOLD:.3f}", ha="center", fontsize=8, color="black")
    ax.set_title(f"Fraud Probability: {prob:.1%}", fontsize=14, fontweight="bold", pad=8)
    fig.tight_layout()
    return fig


def feature_contribution_chart(row_dict: dict):
    """Simple bar of top-feature values vs. their importance."""
    importances = feature_info["feature_importances"]
    top_feats   = [f for f in TOP_FEATURES if f in row_dict][:10]
    if not top_feats:
        return None

    vals   = [row_dict.get(f, np.nan) for f in top_feats]
    imps   = [importances.get(f, 0) for f in top_feats]
    max_imp = max(imps) or 1
    norm_imps = [v / max_imp for v in imps]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#e05c5c" if ni > 0.5 else "#4c9be8" for ni in norm_imps]
    ax.barh(top_feats, norm_imps, color=colors)
    ax.set_xlabel("Normalised Feature Importance", fontsize=11)
    ax.set_title("Top Features Used in This Prediction", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    high_patch = mpatches.Patch(color="#e05c5c", label="High importance")
    low_patch  = mpatches.Patch(color="#4c9be8", label="Moderate importance")
    ax.legend(handles=[high_patch, low_patch], fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Page 1: Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🔍 IEEE-CIS Fraud Detection")
    st.markdown(
        """
        This app demonstrates a **LightGBM** classifier trained on the
        [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
        dataset (~590 k transactions). The model predicts whether an online
        transaction is **fraudulent** (`isFraud = 1`).
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC",          f"{metrics['roc_auc']:.4f}",  help="Area under the ROC curve on 20% hold-out")
    col2.metric("Avg Precision",     f"{metrics['avg_precision']:.4f}", help="Area under the Precision-Recall curve")
    col3.metric("F1 Score (fraud)",  f"{metrics['f1_fraud']:.4f}", help="F1 at the optimal threshold")
    col4.metric("Training samples",  f"{metrics['train_rows']:,}")

    st.markdown("---")
    st.subheader("How It Works")
    steps = [
        ("1️⃣ Data Merge",         "Transaction + Identity tables joined on `TransactionID` (left join)."),
        ("2️⃣ Feature Selection",  "V-columns with <30% missing kept; high-missing columns dropped. 170 features total."),
        ("3️⃣ Encoding",           "Categorical features label-encoded; LightGBM handles NaN natively."),
        ("4️⃣ Class Imbalance",    f"Only {metrics['fraud_rate']:.1%} of transactions are fraud. `scale_pos_weight={1/metrics['fraud_rate']:.0f}` applied."),
        ("5️⃣ Training",           f"LightGBM with early stopping (AUC). Best iteration: {metrics['best_iteration']}."),
        ("6️⃣ Threshold",          f"Optimal threshold {metrics['threshold']:.4f} selected to maximise F1 on the validation set."),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

    st.markdown("---")
    st.subheader("Dataset Class Distribution (Train)")
    fraud_n = int(metrics["train_rows"] * metrics["fraud_rate"])
    legit_n = metrics["train_rows"] - fraud_n
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Legit", "Fraud"], [legit_n, fraud_n], color=["#4c9be8", "#e05c5c"])
    ax.set_ylim(0, max(legit_n, fraud_n) * 1.25)   # 25% headroom above tallest bar
    for bar, v in zip(bars, [legit_n, fraud_n]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(legit_n, fraud_n) * 0.01,
            f"{v:,}\n({v / metrics['train_rows']:.1%})",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Count")
    ax.set_title("Train Set Class Balance", fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Page 2: Single Transaction
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Single Transaction":
    st.title("🔎 Single Transaction Prediction")
    st.markdown("Fill in the transaction details below to get an instant fraud risk score.")

    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        c1, c2, c3 = st.columns(3)
        transaction_amt = c1.number_input("TransactionAmt ($)", min_value=0.0, value=150.0, step=1.0)
        product_cd      = c2.selectbox("ProductCD", ["W","H","C","S","R"], index=0)
        transaction_dt  = c3.number_input("TransactionDT (seconds)", min_value=0, value=86400, step=3600)

        st.subheader("Card Information")
        c1, c2, c3, c4 = st.columns(4)
        card1 = c1.number_input("card1", value=4000, step=1)
        card2 = c2.number_input("card2", value=321.0, step=1.0)
        card3 = c3.number_input("card3", value=150.0, step=1.0)
        card4 = c4.selectbox("card4 (network)", ["visa","mastercard","discover","american express"], index=0)
        c1, c2, c3 = st.columns(3)
        card5 = c1.number_input("card5", value=142.0, step=1.0)
        card6 = c2.selectbox("card6 (type)", ["credit","debit","debit or credit","charge card"], index=0)

        st.subheader("Address & Email")
        c1, c2, c3, c4 = st.columns(4)
        addr1        = c1.number_input("addr1 (billing region)", value=315.0, step=1.0)
        addr2        = c2.number_input("addr2 (billing country)", value=87.0,  step=1.0)
        p_emaildomain = c3.selectbox("P_emaildomain",
            ["gmail.com","yahoo.com","hotmail.com","outlook.com","anonymous.com","other",""], index=0)
        r_emaildomain = c4.selectbox("R_emaildomain",
            ["","gmail.com","yahoo.com","hotmail.com","anonymous.com","other"], index=0)

        st.subheader("Count Features (C1–C14)")
        c_vals = {}
        cols = st.columns(7)
        for i, cidx in enumerate(range(1, 15)):
            c_vals[f"C{cidx}"] = cols[i % 7].number_input(f"C{cidx}", value=1.0, step=1.0, key=f"c{cidx}")

        st.subheader("Match Features (M1–M6)")
        m_choices = ["T","F","M0","M1","M2",""]
        mc = st.columns(6)
        m_vals = {}
        for i in range(1, 7):
            m_vals[f"M{i}"] = mc[i-1].selectbox(f"M{i}", m_choices, index=0, key=f"m{i}")

        submitted = st.form_submit_button("🔮 Predict", use_container_width=False)

    if submitted:
        row = {
            "TransactionAmt": transaction_amt,
            "ProductCD":      product_cd,
            "TransactionDT":  transaction_dt,
            "card1": card1, "card2": card2, "card3": card3,
            "card4": card4, "card5": card5, "card6": card6,
            "addr1": addr1, "addr2": addr2,
            "P_emaildomain": p_emaildomain,
            "R_emaildomain": r_emaildomain if r_emaildomain else "__nan__",
        }
        row.update(c_vals)
        row.update(m_vals)

        prob, label = predict_row(row)

        st.markdown("---")
        verdict_col, gauge_col = st.columns([1, 2])

        with verdict_col:
            if label == 1:
                st.error(f"### 🚨 FRAUD DETECTED\nFraud probability: **{prob:.1%}**")
            else:
                st.success(f"### ✅ LEGITIMATE\nFraud probability: **{prob:.1%}**")
            st.metric("Fraud Probability", f"{prob:.4f}")
            st.metric("Decision Threshold", f"{THRESHOLD:.4f}")
            st.metric("Verdict", "FRAUD" if label else "LEGIT")

        with gauge_col:
            st.pyplot(fraud_gauge(prob))

        st.markdown("---")
        contrib_fig = feature_contribution_chart(row)
        if contrib_fig:
            st.subheader("Feature Importances (model-wide)")
            st.pyplot(contrib_fig)


# ══════════════════════════════════════════════════════════════════════════════
# Page 3: Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📦 Batch Prediction":
    st.title("📦 Batch Prediction")
    st.markdown(
        """
        Upload a CSV file with transaction data (same columns as the training set)
        to get predictions for multiple transactions at once.
        """
    )

    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])

    if uploaded is not None:
        with st.spinner("Loading and predicting …"):
            df_upload = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(df_upload):,}** rows, **{df_upload.shape[1]}** columns")

            # Encode
            df_pred = df_upload.copy()
            for col in CAT_COLS:
                if col not in df_pred.columns:
                    continue
                df_pred[col] = df_pred[col].astype(str)
                le       = ENCODERS[col]
                known    = set(le.classes_)
                fallback = le.classes_[0]
                df_pred[col] = df_pred[col].apply(lambda x: x if x in known else fallback)
                df_pred[col] = le.transform(df_pred[col])

            # Align columns – reindex adds all missing cols as NaN in one shot
            df_input = df_pred.reindex(columns=FEAT_NAMES).astype({
                c: "float32" for c in FEAT_NAMES
                if c not in df_pred.columns or df_pred[c].dtype == "float64"
            })

            probs  = booster.predict(df_input, num_iteration=booster.best_iteration)
            labels = (probs >= THRESHOLD).astype(int)

        df_upload["fraud_probability"] = probs.round(4)
        df_upload["predicted_fraud"]   = labels

        fraud_count = labels.sum()
        st.metric("Transactions analysed", f"{len(df_upload):,}")
        col1, col2 = st.columns(2)
        col1.metric("Predicted fraud",    f"{fraud_count:,}  ({fraud_count/len(labels):.1%})")
        col2.metric("Predicted legit",    f"{len(labels)-fraud_count:,}")

        st.subheader("Results Preview")
        display_cols = ["fraud_probability","predicted_fraud"] + \
                       [c for c in ["TransactionAmt","ProductCD","card4","card6","P_emaildomain"] if c in df_upload.columns]
        st.dataframe(
            df_upload[display_cols].sort_values("fraud_probability", ascending=False).head(200),
            width="stretch",
        )

        # Score distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probs, bins=50, color="#4c9be8", edgecolor="white")
        ax.axvline(THRESHOLD, color="red", ls="--", lw=1.5, label=f"Threshold ({THRESHOLD:.3f})")
        ax.set_xlabel("Predicted Fraud Probability")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution – Uploaded Batch")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

        csv_out = df_upload.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions CSV",
            csv_out,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("👆 Upload a CSV to get started. You can use the **test_transaction.csv** file from the project folder.")

        # Show sample input format
        st.subheader("Expected columns (sample)")
        sample_cols = ["TransactionID","TransactionDT","TransactionAmt","ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain","R_emaildomain","C1","C2","C3","C4","C5"]
        st.code(", ".join(sample_cols))


# ══════════════════════════════════════════════════════════════════════════════
# Page 4: Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation")
    st.markdown("All charts below are computed on the **20% hold-out validation set** (≈118 k transactions).")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC",         f"{metrics['roc_auc']:.4f}")
    col2.metric("Avg Precision",   f"{metrics['avg_precision']:.4f}")
    col3.metric("F1 (fraud)",      f"{metrics['f1_fraud']:.4f}")
    col4.metric("Best Iteration",  metrics["best_iteration"])

    st.markdown("---")

    chart_files = {
        "ROC Curve":              CHART_DIR / "roc_curve.png",
        "Precision-Recall Curve": CHART_DIR / "pr_curve.png",
        "Confusion Matrix":       CHART_DIR / "confusion_matrix.png",
        "Score Distribution":     CHART_DIR / "score_distribution.png",
        "Feature Importance":     CHART_DIR / "feature_importance.png",
    }

    # Display charts in a 2-column grid
    items = list(chart_files.items())
    for i in range(0, len(items), 2):
        c1, c2 = st.columns(2)
        name1, path1 = items[i]
        c1.subheader(name1)
        if path1.exists():
            c1.image(str(path1), width="stretch")
        else:
            c1.warning("Chart not found – run train_model.py first.")

        if i + 1 < len(items):
            name2, path2 = items[i + 1]
            c2.subheader(name2)
            if path2.exists():
                c2.image(str(path2), width="stretch")
            else:
                c2.warning("Chart not found – run train_model.py first.")

    st.markdown("---")
    st.subheader("Training Progress (AUC vs. Boosting Round)")
    st.markdown(
        f"""
        The model trained for **{metrics['best_iteration']} boosting rounds** (from a max of 2,000) before early stopping.
        AUC improved from ~0.95 at round 100 to **{metrics['roc_auc']}** at round {metrics['best_iteration']}.
        """
    )

    st.subheader("How to Interpret the Threshold")
    st.markdown(
        f"""
        At the **current threshold of {THRESHOLD:.4f}** (adjustable in the sidebar):

        - Predicted probability ≥ **{THRESHOLD:.4f}** → 🚨 **FRAUD**
        - Predicted probability < **{THRESHOLD:.4f}** → ✅ **LEGIT**

        The model-optimal threshold of **{metrics['threshold']:.4f}** was chosen to maximise F1 on the validation set.
        Use the **Decision Threshold** slider in the sidebar to explore the trade-off:
        lowering it catches more fraud at the cost of more false alarms; raising it reduces false alarms but risks missing fraud.
        The charts above were generated at the model-optimal threshold.
        """
    )
