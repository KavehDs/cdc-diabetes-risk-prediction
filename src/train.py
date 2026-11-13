"""Train and evaluate diabetes-risk models on the CDC Diabetes Health
Indicators dataset (BRFSS 2015, UCI #891).

Run:  python src/train.py
It prints the measured test-set metrics and writes metrics.json plus the
plots in plots/.

Data: 253,680 survey responses, 21 features, target Diabetes_012
(0 = no diabetes, 1 = prediabetes, 2 = diabetes). The full CSV
(data/diabetes_012_full.csv, ~23 MB) is gitignored and downloaded at
runtime if missing; the bundled data/diabetes_sample.csv (20,000 rows,
stratified) is used as a fallback with a printed warning.

EDUCATIONAL PROJECT ONLY: this uses self-reported survey answers, not
lab results, and is not a medical diagnosis tool.
"""
import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FULL = DATA_DIR / "diabetes_012_full.csv"
SAMPLE = DATA_DIR / "diabetes_sample.csv"
MIRROR_URL = ("https://raw.githubusercontent.com/ucb-stat-159-s23/"
              "project-group25/main/data/"
              "diabetes_012_health_indicators_BRFSS2015.csv")
CANONICAL = "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators"

FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
            "Income"]
TARGET = "Diabetes_012"
CLASSES = [0, 1, 2]
CLASS_NAMES = ["No diabetes", "Prediabetes", "Diabetes"]


def load_data():
    if FULL.exists():
        print(f"Using full file: {FULL}")
        return pd.read_csv(FULL), "full"
    print(f"Full file not found; downloading from {MIRROR_URL}")
    try:
        urllib.request.urlretrieve(MIRROR_URL, FULL)
        print("Download complete.")
        return pd.read_csv(FULL), "full"
    except Exception as exc:  # noqa: BLE001 - fall back to the sample
        print(f"Download failed ({exc}); using bundled sample {SAMPLE}")
        return pd.read_csv(SAMPLE), "sample"


def evaluate(name, y_true, y_pred, y_proba):
    rep = classification_report(y_true, y_pred, output_dict=True,
                                zero_division=0)
    per_class = {
        str(c): {"precision": round(rep[str(c)]["precision"], 4),
                 "recall": round(rep[str(c)]["recall"], 4),
                 "f1": round(rep[str(c)]["f1-score"], 4),
                 "support": int(rep[str(c)]["support"])}
        for c in CLASSES
    }
    # One-vs-rest ROC-AUC per class + macro mean.
    aucs = {}
    for c in CLASSES:
        aucs[str(c)] = round(
            roc_auc_score((y_true == c).astype(int), y_proba[:, c]), 4)
    # Multiclass Brier score (mean squared error of predicted probabilities).
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = round(float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1))), 4)
    # Binary view, derived from the 3-class prediction: diabetes (class 2)
    # vs everyone else, and any-diabetes (class 1 or 2) vs none.
    y_bin = (y_true > 0).astype(int)
    pred_bin = (y_pred > 0).astype(int)
    bin_rep = classification_report(y_bin, pred_bin, output_dict=True,
                                    zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "per_class": per_class,
        "roc_auc_ovr": aucs,
        "roc_auc_ovr_macro": round(float(np.mean(list(aucs.values()))), 4),
        "brier_multiclass": brier,
        "binary_any_diabetes": {
            "accuracy": round(bin_rep["accuracy"], 4),
            "precision_positive": round(bin_rep["1"]["precision"], 4),
            "recall_positive": round(bin_rep["1"]["recall"], 4),
            "f1_positive": round(bin_rep["1"]["f1-score"], 4),
            "roc_auc": round(roc_auc_score(y_bin, y_proba[:, 1]
                                           + y_proba[:, 2]), 4),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    df, source = load_data()
    assert list(df.columns)[:1] == [TARGET], df.columns.tolist()
    print(f"Rows: {len(df):,}  Columns: {df.shape[1]}  Source: {source}")
    print("Class balance:",
          df[TARGET].value_counts(normalize=True).sort_index().round(4)
          .to_dict())
    assert df.isna().sum().sum() == 0, "unexpected missing values"

    X = df[FEATURES]
    y = df[TARGET].astype(int).to_numpy()
    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.20, stratify=y, random_state=42)
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Balanced class weights (prediabetes is only ~1.8% of rows).
    counts = np.bincount(y_train, minlength=3)
    cw = {c: float(len(y_train) / (3 * counts[c])) for c in CLASSES}
    print("Class weights:", {k: round(v, 2) for k, v in cw.items()})

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    models = {
        "Multinomial Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=300,
                                       class_weight="balanced",
                                       random_state=42)),
        ]),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6,
                                 learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=2,
                                 random_state=42, eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(n_estimators=300, num_leaves=31,
                                   learning_rate=0.05, class_weight="balanced",
                                   random_state=42, n_jobs=2, verbose=-1),
    }
    sample_weight = np.array([cw[c] for c in y_train])

    results = {}
    probas = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        if name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        probas[name] = proba
        results[name] = evaluate(name, y_test, pred, proba)
        r = results[name]
        print(f"  accuracy={r['accuracy']} macro-F1={r['macro_f1']} "
              f"OvR AUC={r['roc_auc_ovr_macro']} Brier={r['brier_multiclass']}")
        for c in CLASSES:
            pc = r["per_class"][str(c)]
            print(f"  class {c} ({CLASS_NAMES[c]}): P={pc['precision']} "
                  f"R={pc['recall']} F1={pc['f1']}")

    # Majority-class baseline (always predict "No diabetes"): accuracy
    # alone would make this look best (84.2%), which is exactly why
    # accuracy is the wrong headline metric for this dataset.
    print("\nMajority-class baseline (always predict 0) ...")
    base_pred = np.zeros_like(y_test)
    base_proba = np.zeros((len(y_test), 3))
    base_proba[:, 0] = 1.0
    results["Majority class baseline"] = evaluate(
        "Majority class baseline", y_test, base_pred, base_proba)
    r = results["Majority class baseline"]
    print(f"  accuracy={r['accuracy']} macro-F1={r['macro_f1']} "
          f"(OvR AUC undefined for a constant predictor)")

    # Best model by macro-F1 (the fair summary under this imbalance).
    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    print(f"\nBest by macro-F1: {best_name}")
    best_proba = probas[best_name]
    best_pred = best_proba.argmax(axis=1)

    # --- Fairness check: recall for diabetes (class 2) and prediabetes
    # (class 1) by age group and by sex, for the best model. Age coding:
    # 1=18-24 ... 5=40-44, 6=45-49 ... 9=60-64, 10=65-69 ... 13=80+.
    test_df = df.loc[idx_te].copy()
    test_df["pred"] = best_pred
    age_group = pd.cut(test_df["Age"], bins=[0, 5, 9, 13],
                       labels=["18-44", "45-64", "65+"])
    fairness = {}
    for group_name, group in [("age_group", age_group),
                              ("sex", test_df["Sex"].map({0: "Female",
                                                          1: "Male"}))]:
        fairness[group_name] = {}
        for label in sorted(pd.Series(group).dropna().unique()):
            mask = (group == label).to_numpy()
            sub = test_df[mask]
            fairness[group_name][str(label)] = {
                "n_test": int(mask.sum()),
                "recall_diabetes_class2": round(
                    float(((sub["pred"] == 2) & (sub[TARGET] == 2)).sum()
                          / max((sub[TARGET] == 2).sum(), 1)), 4),
                "recall_prediabetes_class1": round(
                    float(((sub["pred"] == 1) & (sub[TARGET] == 1)).sum()
                          / max((sub[TARGET] == 1).sum(), 1)), 4),
                "share_diabetes_true": round(
                    float((sub[TARGET] == 2).mean()), 4),
            }
    print("Fairness (best model):", json.dumps(fairness, indent=1))

    # --- SHAP for the best tree model (LightGBM preferred; fall back to
    # XGBoost if LightGBM is not the best model).
    import shap
    shap_model_name = best_name if best_name in ("LightGBM", "XGBoost") \
        else "LightGBM"
    shap_model = models[shap_model_name]
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test), size=min(300, len(X_test)),
                            replace=False)
    X_shap = X_test.iloc[sample_idx]
    explainer = shap.TreeExplainer(shap_model)
    sv = explainer.shap_values(X_shap)
    # shap returns a list (one array per class) for multiclass.
    if isinstance(sv, list):
        mean_abs = np.mean([np.abs(a).mean(axis=0) for a in sv], axis=0)
    else:
        mean_abs = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 \
            else np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    shap_top = [{"feature": FEATURES[i],
                 "mean_abs_shap": round(float(mean_abs[i]), 4)}
                for i in order[:10]]
    print("SHAP top 10:", shap_top)

    metrics = {
        "dataset": {
            "name": "CDC Diabetes Health Indicators (BRFSS 2015, UCI #891)",
            "canonical_source": CANONICAL,
            "mirror_used": MIRROR_URL,
            "kaggle_mirror": "https://www.kaggle.com/datasets/"
                            "alexteboul/diabetes-health-indicators-dataset",
            "rows_used": int(len(df)),
            "columns": int(df.shape[1]),
            "features": FEATURES,
            "target": TARGET,
            "class_labels": {str(i): CLASS_NAMES[i] for i in CLASSES},
            "class_balance": {str(k): round(float(v), 4) for k, v in
                              df[TARGET].value_counts(normalize=True)
                              .sort_index().items()},
            "class_counts": {str(k): int(v) for k, v in
                             df[TARGET].value_counts().sort_index().items()},
            "split": "stratified 80/20, random_state=42",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "source_used": source,
            "note": "Self-reported telephone-survey responses, not lab "
                    "results. No missing values.",
        },
        "results": results,
        "best_model_by_macro_f1": best_name,
        "fairness_best_model": fairness,
        "shap_top10": shap_top,
        "shap_model": shap_model_name,
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nWrote metrics.json")

    # --- Plots (best model) ---
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)

    cm = np.array(results[best_name]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion matrix — {best_name}\n(test set, n={len(y_test):,})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plots / "confusion_matrix.svg")
    plt.close(fig)

    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in CLASSES:
        fpr, tpr, _ = roc_curve((y_test == c).astype(int),
                                best_proba[:, c])
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} "
                f"(AUC {results[best_name]['roc_auc_ovr'][str(c)]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title(f"One-vs-rest ROC curves — {best_name}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "roc_curves.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    feats = [s["feature"] for s in shap_top][::-1]
    vals = [s["mean_abs_shap"] for s in shap_top][::-1]
    ax.barh(feats, vals)
    ax.set_title(f"Top 10 drivers — mean |SHAP| ({shap_model_name}, "
                 f"300-row test sample)")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(plots / "shap_top_drivers.svg")
    plt.close(fig)
    print("Wrote plots:", sorted(p.name for p in plots.glob("*.svg")))


if __name__ == "__main__":
    main()
