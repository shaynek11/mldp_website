import streamlit as st
import pandas as pd
import joblib

# ---- Custom Style: Beige, Large Font, Black Text, Button ----
st.markdown("""
    <style>
    html, body, [class*="stApp"] {
        background: linear-gradient(135deg, #f7ecd6 0%, #ded6b8 100%) !important;
        color: #1d1a13 !important;
    }
    .block-container {padding-top: 2.5rem; padding-bottom: 2.5rem; max-width: 100vw !important;}
    /* Input label and text style */
    label, .stMarkdown, .stTitle, .stHeader {
        color: #1d1a13 !important;
        font-size: 1.19rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }
    /* Make all selectbox/slider input text bigger */
    .stSelectbox, .stSlider, .stButton {font-size: 1.13rem !important;}
    /* Coloured button */
    .stButton > button {
        background-color: #be9d4a;
        color: #181200;
        font-size: 1.16rem;
        border-radius: 13px;
        border: none;
        font-weight: 700;
        letter-spacing: 0.01em;
        padding: 0.87em 2.3em;
        margin: 1em 0 0.5em 0;
        transition: 0.14s;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        display: block;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .stButton > button:hover {
        background: #8b7a36;
        color: #fff;
        transform: scale(1.04);
    }
    /* Black text for slider values */
    .stSlider > div {color: #1d1a13 !important;}
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
    <div style='text-align:center;'>
        <h1 style='font-size:2.8rem; font-weight:900; margin-bottom:0.33em; letter-spacing:0.03em; color:#181200'>
            🧑‍⚕️ Stroke Risk Predictor
        </h1>
        <div style='font-size:1.15rem; margin-bottom:1em;'>
            Enter your health details below.<br>
            <span style='color:#be9d4a;font-weight:600'>Our AI model will estimate your risk of stroke instantly!</span>
            <div style='color:#444; font-size:0.99rem; margin-top:0.38em;'>Your info is not stored.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---- INPUTS (no card) ----
def label(text):
    st.markdown(f"<span style='font-size:1.21rem;font-weight:600;color:#181200;'>{text}</span>", unsafe_allow_html=True)

label("👤 Gender")
gender = st.selectbox("", ['Male', 'Female'], key="gender_input")

label("🎂 Age")
age = st.slider('', min_value=0, max_value=120, value=30, key="age_input")

label("💓 Hypertension")
hypertension = st.selectbox('', ['No', 'Yes'], key="hyper_input")
hypertension = 1 if hypertension == 'Yes' else 0

label("🫀 Heart Disease")
heart_disease = st.selectbox('', ['No', 'Yes'], key="heart_input")
heart_disease = 1 if heart_disease == 'Yes' else 0

label("💍 Ever Married")
ever_married = st.selectbox("", ['Yes', 'No'], key="married_input")

label("💼 Work Type")
work_type = st.selectbox("", ['Private', 'Self-employed', 'Govt_job', 'Other'], key="work_input")

label("🏠 Residence Type")
Residence_type = st.selectbox("", ['Urban', 'Rural'], key="residence_input")

label("🩸 Average Glucose Level")
avg_glucose_level = st.slider('', min_value=0.0, max_value=300.0, value=100.0, step=0.1, key="glucose_input")

label("⚖️ BMI")
bmi = st.slider('', min_value=10.0, max_value=70.0, value=25.0, step=0.1, key="bmi_input")

label("🚬 Smoking Status")
smoking_status = st.selectbox("", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'], key="smoke_input")

predict_clicked = st.button('✨ Predict Stroke Risk')

# ---- MODEL LOAD & PREPROCESS ----
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
model_lr = joblib.load('lr_model.pkl')
model_gb = joblib.load('gb_model.pkl')

input_dict = {
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [Residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
}
input_df = pd.DataFrame(input_dict)
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
num_cols = ['age', 'avg_glucose_level', 'bmi']
bool_cols = ['hypertension', 'heart_disease']

# 1. Encode categoricals
encoded_cat = encoder.transform(input_df[cat_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))
# 2. Scale numericals
scaled_num = scaler.transform(input_df[num_cols])
scaled_num_df = pd.DataFrame(scaled_num, columns=num_cols)
# 3. Concatenate all features in model's input order
X_input = pd.concat([scaled_num_df, input_df[bool_cols].astype(int).reset_index(drop=True), encoded_cat_df], axis=1)
expected_cols = model_lr.feature_names_in_
X_input = X_input.reindex(columns=expected_cols, fill_value=0)

# ---- RESULT CARD: Always visible ----
result_placeholder = st.empty()
default_result = """
    <div style='background:#ede6d5;padding:2em 1.3em 1.3em 1.3em;border-radius:18px;margin-top:2em;text-align:center;
        box-shadow:0 4px 24px rgba(30,90,180,0.06);border:1px solid #e3d9be;min-height:105px;'>
        <span style='font-size:2.1rem;color:#302914'>🧠 Prediction result will appear here</span><br>
        <span style='color:#666;font-size:1.07rem;'>Please fill in your info and press <b>Predict Stroke Risk</b>.</span>
    </div>
"""
result_placeholder.markdown(default_result, unsafe_allow_html=True)

if predict_clicked:
    prob_lr = model_lr.predict_proba(X_input)[:, 1]
    prob_gb = model_gb.predict_proba(X_input)[:, 1]
    ensemble_prob = (prob_lr + prob_gb) / 2

    result_placeholder.markdown(
        f"<div style='background:#ede6d5;padding:2em 1.3em 1.3em 1.3em;border-radius:18px;margin-top:2em;text-align:center;"
        f"box-shadow:0 4px 24px rgba(30,90,180,0.06);border:1px solid #e3d9be;'>"
        f"<span style='font-size:2.35rem;'>🧠 Stroke Risk: <b style='color:#be9d4a'>{ensemble_prob[0]:.2%}</b></span><br>"
        f"<span style='color:#444;font-size:1.08rem'>Please consult a healthcare professional for more advice.</span>"
        f"</div>", unsafe_allow_html=True
    )
