"""
IEEE Fraud Detection – Training Script
=======================================
Trains a LightGBM binary classifier on the merged transaction + identity data.
Saves:
  model/lgbm_fraud_model.txt   – trained booster
  model/feature_info.pkl       – encoders, feature names, threshold
  model/metrics.json           – validation metrics
  model/charts/                – evaluation PNG charts
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score, precision_recall_curve, f1_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR  = Path(__file__).parent
MODEL_DIR = DATA_DIR / "model"
CHART_DIR = MODEL_DIR / "charts"
MODEL_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
V_THRESH     = 0.30   # keep V-cols with <30% missing in sample

CAT_TX = ["ProductCD","card1","card2","card3","card4","card5","card6",
          "addr1","addr2","P_emaildomain","R_emaildomain",
          "M1","M2","M3","M4","M5","M6","M7","M8","M9"]
CAT_ID = ["DeviceType","DeviceInfo"] + [f"id_{i:02d}" for i in range(12, 39)]


# ── helpers ────────────────────────────────────────────────────────────────────

def select_columns():
    print("Scanning columns for missingness …")
    tx_s = pd.read_csv(DATA_DIR / "train_transaction.csv", nrows=100_000)
    id_s = pd.read_csv(DATA_DIR / "train_identity.csv",  nrows=100_000)

    non_v  = [c for c in tx_s.columns if not c.startswith("V")]
    v_miss = tx_s.filter(like="V").isnull().mean()
    v_keep = v_miss[v_miss < V_THRESH].index.tolist()
    tx_cols = non_v + v_keep

    id_miss = id_s.isnull().mean()
    id_cols = ["TransactionID"] + [
        c for c in id_miss[id_miss < 0.80].index if c != "TransactionID"
    ]
    print(f"  TX cols: {len(tx_cols)}  |  ID cols: {len(id_cols)}")
    return tx_cols, id_cols


def load_data(tx_cols, id_cols):
    print("Loading full training data …")
    tx  = pd.read_csv(DATA_DIR / "train_transaction.csv", usecols=tx_cols)
    id_ = pd.read_csv(DATA_DIR / "train_identity.csv",  usecols=id_cols)
    df  = tx.merge(id_, on="TransactionID", how="left")
    del tx, id_

    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        df[c] = df[c].astype("int32")

    print(f"  Merged shape : {df.shape}")
    print(f"  RAM          : {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"  Fraud rate   : {df['isFraud'].mean():.4%}")
    return df


def encode_cats(df, cat_cols, encoders=None, fit=True):
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("__nan__")
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]).astype("int32")
            encoders[col] = le
        else:
            le    = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__nan__")
            df[col] = le.transform(df[col]).astype("int32")
    return df, encoders


def best_threshold(y_true, y_prob):
    """Threshold that maximises F1 on the validation set."""
    prec, rec, thrs = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thrs[np.argmax(f1s)])


# ── charts ─────────────────────────────────────────────────────────────────────

def plot_roc(y_val, y_prob):
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#e05c5c", label=f"LightGBM (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve – IEEE Fraud Detection", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    print("  Saved roc_curve.png")


def plot_pr(y_val, y_prob):
    prec, rec, _ = precision_recall_curve(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, color="#4c9be8", label=f"LightGBM (AP = {ap:.4f})")
    baseline = y_val.mean()
    ax.axhline(baseline, color="k", ls="--", lw=1, label=f"Baseline (fraud rate {baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve – IEEE Fraud Detection", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)
    print("  Saved pr_curve.png")


def plot_confusion(y_val, y_pred, thr):
    cm   = confusion_matrix(y_val, y_pred)
    cmn  = cm.astype("float") / cm.sum(axis=1, keepdims=True)  # row-normalised
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm,  cmn],
        ["d", ".2%"],
        ["Confusion Matrix (counts)", "Confusion Matrix (row-normalised)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual",    fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

    fig.suptitle(f"Threshold = {thr:.4f}", fontsize=12, color="grey")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("  Saved confusion_matrix.png")


def plot_feature_importance(booster, feat_names, top_n=25):
    imp  = pd.Series(booster.feature_importance("gain"), index=feat_names)
    imp  = imp.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#4c9be8"] * top_n
    colors[-1] = "#e05c5c"   # highlight top feature
    ax.barh(imp.index, imp.values, color=colors[::-1])
    ax.set_xlabel("Gain (feature importance)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances (Gain)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved feature_importance.png")


def plot_score_distribution(y_val, y_prob, thr):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 60)
    ax.hist(y_prob[y_val == 0], bins=bins, alpha=0.6, color="#4c9be8", label="Legit",  density=True)
    ax.hist(y_prob[y_val == 1], bins=bins, alpha=0.6, color="#e05c5c", label="Fraud", density=True)
    ax.axvline(thr, color="black", ls="--", lw=1.5, label=f"Threshold ({thr:.3f})")
    ax.set_xlabel("Predicted Fraud Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution by True Class", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved score_distribution.png")


# ── main ───────────────────────────────────────────────────────────────────────

def train():
    tx_cols, id_cols = select_columns()
    with open(MODEL_DIR / "selected_cols.json", "w") as f:
        json.dump({"tx": tx_cols, "id": id_cols}, f)

    df = load_data(tx_cols, id_cols)

    all_cat = [c for c in CAT_TX + CAT_ID if c in df.columns]
    df, encoders = encode_cats(df, all_cat, fit=True)
    df.drop(columns=["TransactionID"], errors="ignore", inplace=True)

    y = df.pop("isFraud").astype("int32")
    X = df
    feat_names = X.columns.tolist()
    print(f"  Features: {len(feat_names)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / pos
    print(f"  scale_pos_weight = {spw:.1f}  |  fraud: {pos:,}")

    lgb_tr  = lgb.Dataset(X_tr,  label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)

    params = {
        "objective":          "binary",
        "metric":             "auc",          # AUC only → early stopping tracks AUC
        "learning_rate":      0.05,
        "num_leaves":         255,
        "min_child_samples":  20,
        "feature_fraction":   0.8,
        "bagging_fraction":   0.8,
        "bagging_freq":       5,
        "scale_pos_weight":   spw,
        "n_jobs":             -1,
        "seed":               RANDOM_STATE,
        "verbosity":          -1,
    }

    print("\nTraining LightGBM …")
    booster = lgb.train(
        params,
        lgb_tr,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"  Best iteration: {booster.best_iteration}")

    # ── evaluate ───────────────────────────────────────────────────────────────
    y_prob = booster.predict(X_val, num_iteration=booster.best_iteration)
    thr    = best_threshold(y_val.values, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    ap  = average_precision_score(y_val, y_prob)
    f1  = f1_score(y_val, y_pred, pos_label=1)

    print(f"\n{'='*50}")
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  Avg Precision    : {ap:.4f}")
    print(f"  Optimal threshold: {thr:.4f}")
    print(f"  F1 (fraud)       : {f1:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["legit","fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # ── charts ─────────────────────────────────────────────────────────────────
    print("\nGenerating charts …")
    plot_roc(y_val.values, y_prob)
    plot_pr(y_val.values, y_prob)
    plot_confusion(y_val.values, y_pred, thr)
    plot_feature_importance(booster, feat_names, top_n=25)
    plot_score_distribution(y_val.values, y_prob, thr)

    # ── save artefacts ─────────────────────────────────────────────────────────
    print("\nSaving artefacts …")
    booster.save_model(str(MODEL_DIR / "lgbm_fraud_model.txt"))

    imp = pd.Series(booster.feature_importance("gain"), index=feat_names)
    feature_info = {
        "feature_names":       feat_names,
        "cat_cols":            [c for c in all_cat if c in feat_names],
        "encoders":            encoders,
        "top_features":        imp.sort_values(ascending=False).head(30).index.tolist(),
        "feature_importances": imp.sort_values(ascending=False).to_dict(),
        "threshold":           thr,
    }
    joblib.dump(feature_info, MODEL_DIR / "feature_info.pkl")

    metrics = {
        "roc_auc":        round(auc, 4),
        "avg_precision":  round(ap,  4),
        "threshold":      round(thr, 4),
        "f1_fraud":       round(f1,  4),
        "best_iteration": booster.best_iteration,
        "n_features":     len(feat_names),
        "train_rows":     int(len(X_tr)),
        "val_rows":       int(len(X_val)),
        "fraud_rate":     round(float(y_tr.mean()), 4),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll artefacts saved:")
    print(f"  model/lgbm_fraud_model.txt")
    print(f"  model/feature_info.pkl")
    print(f"  model/metrics.json")
    print(f"  model/charts/  (5 PNG files)")
    print("\n✓ Training complete!")


if __name__ == "__main__":
    train()
