"""Compare TabICLv2 (a modern tabular foundation model) with multinomial
Logistic Regression on a stratified subsample of the CDC Diabetes Health
Indicators data (BRFSS 2015, UCI #891).

Run:  python src/foundation_comparison.py
It prints the measured metrics for both models and writes them into
metrics.json under "foundation_model_comparison".

Subsample: 6,000 rows drawn stratified from the full 253,680-row file,
then split stratified 5,000 train / 1,000 test (random_state=42).
The 10,000/2,000 subsample was tried first and the run was killed by
the Linux OOM killer during TabICL prediction on this 1.5-CPU / 6 GB
machine (host memory also fluctuates), so the comparison uses the
smaller subsample — the numbers below are the ones actually measured.
TabICL is an in-context-learning transformer: it does NOT train on the
data in the usual sense — it conditions on the 5,000 training rows in a
single forward pass. That is why the same split is used for both models.

TabICLv2 model: TabICLClassifier from the `tabicl` package (v2.2.0),
default checkpoint tabicl-classifier-v2-20260212.ckpt.
License: BSD 3-Clause, Copyright (c) 2025, Soda team @ Inria —
permissive, no use-case restrictions. Source:
https://github.com/soda-inria/tabicl
The checkpoint weights are downloaded automatically by the package on
first use (from Hugging Face) and are NOT bundled in this repository.

EDUCATIONAL PROJECT ONLY: self-reported survey answers, not lab
results, not a medical diagnosis tool.
"""

import json
import resource
import time
import urllib.request
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tabicl
from tabicl import TabICLClassifier

BASE = Path(__file__).resolve().parent.parent
FULL_URL = (
    "https://raw.githubusercontent.com/ucb-stat-159-s23/project-group25/"
    "main/data/diabetes_012_health_indicators_BRFSS2015.csv"
)
FULL_PATH = BASE / "data" / "diabetes_012_full.csv"
SAMPLE_PATH = BASE / "data" / "diabetes_sample.csv"
METRICS_PATH = BASE / "metrics.json"

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
    "Income",
]
TARGET = "Diabetes_012"
RANDOM_STATE = 42
SUBSAMPLE_ROWS = 6_000
TRAIN_ROWS = 5_000
TEST_ROWS = 1_000
# Checkpoint downloaded once with src/download_checkpoint.py (plain
# streaming HTTP, no torch in memory) and stored OUTSIDE the repo at
# ~/.cache/tabicl/ — weights are never bundled in the project.
# model_path avoids a second download through huggingface_hub at runtime.
CHECKPOINT = Path.home() / ".cache" / "tabicl" / "tabicl-classifier-v2-20260212.ckpt"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_data() -> pd.DataFrame:
    if FULL_PATH.exists():
        print(f"Loading full file: {FULL_PATH}")
        return pd.read_csv(FULL_PATH)
    if SAMPLE_PATH.exists():
        print(f"Full file missing; falling back to bundled sample: {SAMPLE_PATH}")
        return pd.read_csv(SAMPLE_PATH)
    print("No local file; downloading full file (23 MB)...")
    urllib.request.urlretrieve(FULL_URL, FULL_PATH)
    return pd.read_csv(FULL_PATH)


def evaluate(name, y_true, y_pred, y_proba):
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    per_class = {
        str(k): {
            "precision": round(v["precision"], 4),
            "recall": round(v["recall"], 4),
            "f1": round(v["f1-score"], 4),
            "support": int(v["support"]),
        }
        for k, v in report.items()
        if k in ("0", "1", "2")
    }
    ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None)
    brier = float(
        np.mean(
            [
                brier_score_loss((y_true == c).astype(int), y_proba[:, c])
                for c in range(3)
            ]
        )
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": per_class,
        "roc_auc_ovr": {str(c): round(float(ovr[c]), 4) for c in range(3)},
        "roc_auc_ovr_macro": round(float(np.mean(ovr)), 4),
        "brier_multiclass": round(brier, 4),
    }


def main():
    print(f"[rss {rss_gb():.2f} GB] starting", flush=True)
    df = load_data()
    print(f"[rss {rss_gb():.2f} GB] data loaded", flush=True)
    print(f"Loaded {len(df):,} rows; using stratified subsample of "
          f"{SUBSAMPLE_ROWS:,} rows.")
    sub, _ = train_test_split(
        df, train_size=SUBSAMPLE_ROWS, stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    X = sub[FEATURES].to_numpy(dtype=np.float32)
    y = sub[TARGET].to_numpy(dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_ROWS, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"Train: {len(y_train):,} rows, test: {len(y_test):,} rows")
    print("Train class counts:",
          {int(c): int((y_train == c).sum()) for c in range(3)})
    print("Test class counts:",
          {int(c): int((y_test == c).sum()) for c in range(3)})

    # --- TabICLv2 (modern tabular foundation model) ---
    print("\n--- TabICLv2 ---")
    tabicl_version = pkg_version("tabicl")
    print(f"tabicl version: {tabicl_version}")
    t0 = time.time()
    clf = TabICLClassifier(
        random_state=RANDOM_STATE,
        device="cpu",
        model_path=str(CHECKPOINT) if CHECKPOINT.exists() else None,
        allow_auto_download=not CHECKPOINT.exists(),
        # batch_size 4 (default 8) halves activation memory on this
        # 1.5-CPU / 6 GB machine; n_estimators stays at the default 8.
        batch_size=4,
    )
    print(f"[rss {rss_gb():.2f} GB] TabICLClassifier created", flush=True)
    clf.fit(X_train, y_train)
    fit_s = time.time() - t0
    print(f"[rss {rss_gb():.2f} GB] TabICL fit done in {fit_s:.1f}s", flush=True)
    t0 = time.time()
    # Predict in chunks: one predict_proba call over all 2,000 test rows
    # spikes memory on this small machine, so stream the test set in
    # 200-row chunks and concatenate (same model, same predictions).
    proba_chunks = []
    for start in range(0, len(X_test), 200):
        proba_chunks.append(
            clf.predict_proba(X_test[start:start + 200])
        )
        print(f"[rss {rss_gb():.2f} GB] TabICL predicted "
              f"{min(start + 200, len(X_test))}/{len(X_test)} rows",
              flush=True)
    proba_tabicl = np.concatenate(proba_chunks, axis=0)
    pred_tabicl = proba_tabicl.argmax(axis=1)
    predict_s = time.time() - t0
    print(f"fit: {fit_s:.1f}s, predict: {predict_s:.1f}s")
    tabicl_res = evaluate("TabICLv2", y_test, pred_tabicl, proba_tabicl)
    tabicl_res["fit_seconds"] = round(fit_s, 1)
    tabicl_res["predict_seconds"] = round(predict_s, 1)
    print(json.dumps(tabicl_res, indent=2))

    # --- Multinomial Logistic Regression (baseline, same splits) ---
    print("\n--- Multinomial Logistic Regression (baseline) ---")
    t0 = time.time()
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )
    logreg.fit(X_train, y_train)
    fit_s = time.time() - t0
    t0 = time.time()
    proba_lr = logreg.predict_proba(X_test)
    pred_lr = proba_lr.argmax(axis=1)
    predict_s = time.time() - t0
    print(f"fit: {fit_s:.1f}s, predict: {predict_s:.1f}s")
    lr_res = evaluate("Logistic Regression", y_test, pred_lr, proba_lr)
    lr_res["fit_seconds"] = round(fit_s, 1)
    lr_res["predict_seconds"] = round(predict_s, 1)
    print(json.dumps(lr_res, indent=2))

    comparison = {
        "model": "TabICLv2 (tabicl.TabICLClassifier, default checkpoint "
                 "tabicl-classifier-v2-20260212.ckpt)",
        "package": f"tabicl {tabicl_version}",
        "license": "BSD 3-Clause License, Copyright (c) 2025, Soda team @ Inria",
        "license_url": "https://github.com/soda-inria/tabicl",
        "license_note": "Permissive open-source license, no use-case "
                        "restrictions (verified in the installed tabicl "
                        "package LICENSE file). Weights are downloaded at "
                        "runtime by the package and are not bundled here. "
                        "TabPFN-3/3.5 was deliberately NOT used: its "
                        "weights carry a non-commercial license.",
        "subsample": {
            "rows_drawn": SUBSAMPLE_ROWS,
            "train_rows": TRAIN_ROWS,
            "test_rows": TEST_ROWS,
            "random_state": RANDOM_STATE,
            "note": "Stratified subsample of the full 253,680-row CDC "
                    "dataset; same split and same test rows for both "
                    "models. A 10,000/2,000 subsample was tried first "
                    "and the run was killed by the Linux OOM killer "
                    "during TabICL prediction on this 1.5-CPU / 6 GB "
                    "machine, so the comparison uses 5,000 train / "
                    "1,000 test. TabICL conditions on the training "
                    "rows in one forward pass (in-context learning) "
                    "rather than training weights.",
        },
        "tabiclv2": tabicl_res,
        "logistic_regression_baseline": lr_res,
    }

    metrics = json.loads(METRICS_PATH.read_text())
    metrics["foundation_model_comparison"] = comparison
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote comparison to {METRICS_PATH}")


if __name__ == "__main__":
    main()
