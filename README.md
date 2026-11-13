# CDC Diabetes Risk Prediction

Predict diabetes status from self-reported health indicators — no
diabetes, prediabetes, or diabetes — and be honest about how hard the
rare class is to catch.

> ⚠️ **This is an educational portfolio project using self-reported
> survey data, not a medical diagnosis tool. It must not be used for
> clinical decisions.** The data are telephone-survey answers, not lab
> results, and the model is far from good enough to screen anyone.

## Results (test set, n = 50,736)

Stratified 80/20 train/test split (`random_state=42`) on the full
253,680-row dataset. The honest headline is **macro-F1, not accuracy**:
a classifier that always predicts "No diabetes" scores 84.2% accuracy
and finds zero cases. Balanced class weights trade accuracy for recall
of the minority classes.

| Model | Accuracy | Macro-F1 | OvR ROC-AUC (macro) | Brier |
|---|---|---|---|---|
| **Multinomial Logistic Regression** | 0.6458 | **0.4269** | **0.7728** | 0.4896 |
| XGBoost | 0.6431 | 0.4245 | 0.7643 | 0.4626 |
| LightGBM | 0.6339 | 0.4215 | 0.7646 | 0.4671 |
| Majority-class baseline | 0.8424 | 0.3048 | 0.5000 | 0.3152 |

Per-class results (Logistic Regression, the best model by macro-F1):

| Class | Precision | Recall | F1 | OvR ROC-AUC |
|---|---|---|---|---|
| No diabetes (84.2%) | 0.951 | 0.663 | 0.781 | 0.815 |
| Prediabetes (1.8%) | 0.033 | 0.326 | 0.059 | 0.685 |
| Diabetes (13.9%) | 0.353 | 0.583 | 0.440 | 0.818 |

### The honest finding: prediabetes is nearly undetectable here

Prediabetes is only 1.8% of respondents, and even with a class weight
of 18× the best model finds only a third of cases — and 97% of its
prediabetes predictions are wrong (precision 0.033). Diabetes itself
is caught better (recall 0.58–0.64, OvR AUC ~0.82), and collapsing
the problem to binary (any diabetes vs none) gives ROC-AUC 0.81–0.82
with recall ~0.82–0.84 at precision ~0.31. Self-reported survey
answers (general health, BMI, age, blood pressure) simply do not
separate prediabetes from healthy respondents. Reporting that is the
point of this project — a model that quietly predicted "no diabetes"
for everyone would look 84% accurate and be useless.

Reproduce: `python src/train.py` — it prints these numbers and writes
`metrics.json` plus the plots in `plots/`.

![ROC curves](plots/roc_curves.svg)
![Confusion matrix](plots/confusion_matrix.svg)
![Top drivers](plots/shap_top_drivers.svg)

## What drives the prediction? (SHAP, LightGBM, 300-row test sample)

Mean |SHAP value| — top 5:

1. **GenHlth** (0.248) — self-rated general health (1 = excellent … 5 = poor)
2. **Age** (0.222)
3. **BMI** (0.193)
4. **HighBP** (0.187) — told by a doctor they have high blood pressure
5. **HighChol** (0.137)

## Fairness check (best model, test set)

Diabetes recall is much lower for younger respondents — the model
misses most cases where diabetes is rare:

| Group | Diabetes recall | Prediabetes recall | True diabetes share |
|---|---|---|---|
| Age 18–44 | 0.30 | 0.15 | 4.0% |
| Age 45–64 | 0.55 | 0.33 | 13.4% |
| Age 65+ | 0.64 | 0.36 | 20.8% |
| Female | 0.53 | 0.36 | 13.0% |
| Male | 0.64 | 0.28 | 15.1% |

A screening tool built on this data would under-detect diabetes in
younger adults and in women. That is a reason not to deploy it, and
the reason the fairness check is in this README.

## Method

- **Data:** CDC Diabetes Health Indicators, BRFSS 2015 (UCI #891),
  253,680 rows, 21 features, no missing values. Target `Diabetes_012`:
  0 = no diabetes (213,703 / 84.2%), 1 = prediabetes (4,631 / 1.8%),
  2 = diabetes (35,346 / 13.9%).
- **Split:** stratified 80/20, so train and test keep the same class mix.
- **Imbalance:** class weights `balanced` (Logistic Regression,
  LightGBM) and per-sample weights (XGBoost); prediabetes weight ≈ 18×.
- **Models:** multinomial Logistic Regression (standardised features),
  XGBoost (300 trees, depth 6) and LightGBM (300 trees, 31 leaves).
- **Evaluation:** per-class Precision / Recall / F1, macro-F1, one-vs-rest
  ROC-AUC and multiclass Brier score. Accuracy is reported only to show
  why it misleads here.
- **Calibration:** Brier scores are 0.46–0.49 for the real models — the
  predicted probabilities are poorly calibrated, another reason this
  model must not be used for individual decisions.

## Dataset

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, cleaned
to 21 health indicators by the UCI Diabetes Health Indicators dataset
(#891). The file used here is `diabetes_012_health_indicators_BRFSS2015.csv`
(253,680 rows). Sources:

- UCI dataset page: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Kaggle mirror: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- Download mirror used by `src/train.py`: https://raw.githubusercontent.com/ucb-stat-159-s23/project-group25/main/data/diabetes_012_health_indicators_BRFSS2015.csv

The bundled `data/diabetes_sample.csv` is a stratified 20,000-row
sample (16,848 / 365 / 2,787). The full 23 MB file is gitignored and
downloaded by `src/train.py` at runtime if missing.

> **Honest note:** these are **self-reported survey answers** from a
> telephone survey — general health, smoking, fruit/vegetable intake,
> income, education — not lab tests, not a clinical cohort, and not a
> representative sample of any other country's population. The value
> here is the method (stratified evaluation under extreme imbalance,
> per-class and fairness reporting, calibration, explainability) applied
> transparently, not a claim that survey answers can diagnose diabetes.

## Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

The demo trains the LightGBM model in memory on the bundled 20,000-row
sample and shows predicted probabilities for the three classes from
health indicators you enter — with the same not-a-diagnosis disclaimer.

## Project structure

```
├── app.py                  # Streamlit demo (trains on the bundled sample)
├── data/
│   ├── diabetes_sample.csv # stratified 20,000-row sample (bundled)
│   └── diabetes_012_full.csv  # full 253,680-row file (gitignored, downloaded)
├── metrics.json            # measured test-set results (from src/train.py)
├── plots/
│   ├── confusion_matrix.svg
│   ├── roc_curves.svg
│   └── shap_top_drivers.svg
├── requirements.txt
└── src/
    └── train.py            # loading, training, evaluation, fairness, SHAP
```

## Limitations

- Prediabetes detection is poor (F1 ≈ 0.06) — the headline result,
  not a footnote.
- Probabilities are poorly calibrated (Brier ≈ 0.46–0.49).
- Recall varies sharply by age and sex; the model under-detects
  diabetes in younger adults.
- Self-reported data: BMI, blood pressure and cholesterol are what
  respondents *said*, not what was measured.
- Single dataset, single split — no external validation on another
  population or year.
- **Not a medical device, not a screening tool, not a diagnosis.**
