import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")

@st.cache_resource
def load_artifacts():
    model = load_model("model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)
    return model, scaler, training_columns

model, scaler, training_columns = load_artifacts()

st.title("💰 Loan Approval Prediction")
st.write("Enter applicant details below:")

# --- Numerical Inputs ---
person_age                 = st.number_input("Person Age", min_value=18, max_value=100, value=25)
person_income              = st.number_input("Person Income ($)", min_value=0.0, value=50000.0)
person_emp_exp             = st.number_input("Employment Experience (years)", min_value=0, value=2)
loan_amnt                  = st.number_input("Loan Amount ($)", min_value=0.0, value=5000.0)
loan_int_rate              = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=10.0)
loan_percent_income        = st.number_input("Loan % of Income (e.g. 0.3 = 30%)", min_value=0.0, max_value=1.0, value=0.1)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=3)
credit_score               = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

# --- Categorical Inputs ---
gender      = st.selectbox("Gender", ["Female", "Male"])
education   = st.selectbox("Education Level", ["Associate", "Bachelor", "Doctorate", "High School", "Master"])
home_owner  = st.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"])
loan_intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
prev_default = st.selectbox("Previous Loan Default on File", ["No", "Yes"])

if st.button("Predict"):

    data = {
        # Numerical
        "person_age":                    person_age,
        "person_income":                 person_income,
        "person_emp_exp":                person_emp_exp,
        "loan_amnt":                     loan_amnt,
        "loan_int_rate":                 loan_int_rate,
        "loan_percent_income":           loan_percent_income,
        "cb_person_cred_hist_length":    cb_person_cred_hist_length,
        "credit_score":                  credit_score,

        # Gender (binary)
        "person_gender_male":            1 if gender == "Male" else 0,

        # Education (one-hot — base category: Associate)
        "person_education_Bachelor":     1 if education == "Bachelor" else 0,
        "person_education_Doctorate":    1 if education == "Doctorate" else 0,
        "person_education_High School":  1 if education == "High School" else 0,
        "person_education_Master":       1 if education == "Master" else 0,

        # Home Ownership (one-hot — base category: MORTGAGE)
        "person_home_ownership_OTHER":   1 if home_owner == "OTHER" else 0,
        "person_home_ownership_OWN":     1 if home_owner == "OWN" else 0,
        "person_home_ownership_RENT":    1 if home_owner == "RENT" else 0,

        # Loan Intent (one-hot — base category: DEBTCONSOLIDATION)
        "loan_intent_EDUCATION":         1 if loan_intent == "EDUCATION" else 0,
        "loan_intent_HOMEIMPROVEMENT":   1 if loan_intent == "HOMEIMPROVEMENT" else 0,
        "loan_intent_MEDICAL":           1 if loan_intent == "MEDICAL" else 0,
        "loan_intent_PERSONAL":          1 if loan_intent == "PERSONAL" else 0,
        "loan_intent_VENTURE":           1 if loan_intent == "VENTURE" else 0,

        # Previous Default (binary)
        "previous_loan_defaults_on_file_Yes": 1 if prev_default == "Yes" else 0,
    }

    input_df = pd.DataFrame([data])

    # Align to exact training column order
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    x    = scaler.transform(input_df)
    prob = float(model.predict(x, verbose=0)[0][0])
    pred = int(prob >= 0.5)

    st.subheader("Prediction Result")
    if pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
    st.metric("Approval Probability", f"{prob*100:.2f}%")