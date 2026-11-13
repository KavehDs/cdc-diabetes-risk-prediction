"""Streamlit demo: CDC Diabetes Risk Prediction.

Run locally:  streamlit run app.py

Trains a LightGBM model in memory on the bundled 20,000-row stratified
sample of the CDC Diabetes Health Indicators dataset (BRFSS 2015) and
shows predicted probabilities for No diabetes / Prediabetes / Diabetes
from health indicators you enter.

EDUCATIONAL DEMO ONLY — not a medical diagnosis tool. The model uses
self-reported survey answers, not lab results, and its prediabetes
detection is poor (see README.md). Do not use it for clinical decisions.
"""
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier

DATA = "data/diabetes_sample.csv"
FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
            "Income"]
CLASSES = ["No diabetes", "Prediabetes", "Diabetes"]


@st.cache_resource
def load_model():
    df = pd.read_csv(DATA)
    y = df["Diabetes_012"].astype(int)
    X = df[FEATURES]
    model = LGBMClassifier(n_estimators=300, num_leaves=31,
                           learning_rate=0.05, class_weight="balanced",
                           random_state=42, n_jobs=2, verbose=-1)
    model.fit(X, y)
    return model


st.set_page_config(page_title="Diabetes Risk — CDC BRFSS 2015",
                   layout="centered")
st.title("Diabetes Risk Prediction (CDC BRFSS 2015)")
# Sliders must stay left-to-right even in RTL browsers (e.g. Persian):
# without this fix, dragging the thumb right decreases the value.
st.markdown(
    """<style>
    [data-testid="stSlider"] { direction: ltr !important; }
    </style>""",
    unsafe_allow_html=True,
)
st.warning("⚠️ Educational demo only — not a medical diagnosis tool. "
           "This model uses self-reported survey answers, not lab results, "
           "and its prediabetes detection is poor. See README.md.")
st.write("LightGBM model trained on a 20,000-row sample of the CDC "
         "Diabetes Health Indicators dataset (253,680 survey responses). "
         "Enter health indicators to see predicted class probabilities.")

model = load_model()

with st.form("person"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox(
            "Age group", list(range(1, 14)), index=8,
            format_func=lambda a: {1: "18-24", 2: "25-29", 3: "30-34",
                                   4: "35-39", 5: "40-44", 6: "45-49",
                                   7: "50-54", 8: "55-59", 9: "60-64",
                                   10: "65-69", 11: "70-74", 12: "75-79",
                                   13: "80+"}[a])
        sex = st.selectbox("Sex", [0, 1], format_func=lambda s: "Female"
                           if s == 0 else "Male")
        bmi = st.number_input("BMI", 12.0, 60.0, 28.0)
        gen_hlth = st.slider("General health (1=excellent … 5=poor)",
                             1, 5, 3)
        high_bp = st.selectbox("High blood pressure?", [0, 1],
                               format_func=lambda v: "Yes" if v else "No")
        high_chol = st.selectbox("High cholesterol?", [0, 1],
                                 format_func=lambda v: "Yes" if v else "No")
    with col2:
        smoker = st.selectbox("Smoked ≥100 cigarettes?", [0, 1],
                              format_func=lambda v: "Yes" if v else "No")
        phys_activity = st.selectbox("Physical activity?", [0, 1],
                                     format_func=lambda v: "Yes"
                                     if v else "No")
        fruits = st.selectbox("Eats fruit daily?", [0, 1],
                              format_func=lambda v: "Yes" if v else "No")
        veggies = st.selectbox("Eats vegetables daily?", [0, 1],
                               format_func=lambda v: "Yes" if v else "No")
        stroke = st.selectbox("Ever had a stroke?", [0, 1],
                              format_func=lambda v: "Yes" if v else "No")
        heart = st.selectbox("Heart disease / heart attack?", [0, 1],
                             format_func=lambda v: "Yes" if v else "No")
    submitted = st.form_submit_button("Predict probabilities")

if submitted:
    row = pd.DataFrame([{
        "HighBP": high_bp, "HighChol": high_chol, "CholCheck": 1,
        "BMI": bmi, "Smoker": smoker, "Stroke": stroke,
        "HeartDiseaseorAttack": heart, "PhysActivity": phys_activity,
        "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": gen_hlth,
        "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": sex,
        "Age": age, "Education": 5, "Income": 5,
    }])[FEATURES]
    proba = model.predict_proba(row)[0]
    pred = int(proba.argmax())
    st.subheader(f"Most likely class: {CLASSES[pred]} "
                 f"({proba[pred]:.1%})")
    for name, p in zip(CLASSES, proba):
        st.write(f"{name}: {p:.1%}")
        st.progress(float(p))
    st.caption("Demo defaults: cholesterol checked, has healthcare, no "
               "cost barrier, no difficulty walking, 0 poor-mental/physical-"
               "health days, no heavy drinking, education/income at "
               "mid-levels. Full test-set metrics: Logistic Regression "
               "macro-F1 0.427, OvR ROC-AUC 0.773; prediabetes F1 only 0.06. "
               "See README.md — and remember, this is not a diagnosis.")
