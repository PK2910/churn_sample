import json
from datetime import datetime, timezone

import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"
LOG_FILE = "prediction_logs.jsonl"

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("Customer Churn Predictor")
st.caption("Fills out a form, sends it to the Dockerized FastAPI model, shows the prediction.")

with st.form("customer_form"):
    st.subheader("Account")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    with col2:
        marital_status = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    with col3:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=9)

    st.subheader("Services")
    col1, col2, col3 = st.columns(3)
    with col1:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
    with col2:
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        voicemail_plan = st.selectbox("Voicemail Plan", ["No", "Yes"])

    st.subheader("Contract & Billing")
    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col3:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    international_plan = st.selectbox("International Plan", ["No", "Yes"])
    num_voicemail_messages = st.number_input("Number of Voicemail Messages", min_value=0, value=0)

    st.subheader("Usage")
    col1, col2 = st.columns(2)
    with col1:
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=168.8)
        total_day_calls = st.number_input("Total Day Calls", min_value=0, value=137)
        total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=241.4)
        total_eve_calls = st.number_input("Total Evening Calls", min_value=0, value=107)
    with col2:
        total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, value=204.8)
        total_night_calls = st.number_input("Total Night Calls", min_value=0, value=106)
        total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=15.5)
        total_intl_calls = st.number_input("Total International Calls", min_value=0, value=4)

    customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=0)
    total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, value=593.3)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # customerID, PhoneService, and TotalCall are dropped by the pipeline
    # before the model ever sees them (confirmed back in the notebook), so
    # they're filled with harmless defaults here instead of cluttering the
    # form with fields that can't actually change the prediction.
    payload = {
        "customerID": "STREAMLIT-USER",
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "MaritalStatus": marital_status,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "InternationalPlan": international_plan,
        "VoiceMailPlan": voicemail_plan,
        "NumbervMailMessages": num_voicemail_messages,
        "TotalDayMinutes": total_day_minutes,
        "TotalDayCalls": total_day_calls,
        "TotalEveMinutes": total_eve_minutes,
        "TotalEveCalls": total_eve_calls,
        "TotalNightMinutes": total_night_minutes,
        "TotalNightCalls": total_night_calls,
        "TotalIntlMinutes": total_intl_minutes,
        "TotalIntlCalls": total_intl_calls,
        "CustomerServiceCalls": customer_service_calls,
        "TotalCall": total_day_calls + total_eve_calls + total_night_calls + total_intl_calls,
        "TotalRevenue": str(total_revenue),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        probability = result["churn_probability"]
        will_churn = result["churn_prediction"]

        st.metric("Churn Probability", f"{probability:.1%}")
        if will_churn:
            st.error("Prediction: likely to churn")
        else:
            st.success("Prediction: not likely to churn")

        with st.expander("Show submitted values"):
            st.json(payload)

        # Append-only log: one JSON object per line, so each prediction is
        # a self-contained record. Opening in "a" (append) mode means every
        # write just adds to the end of the file without touching or
        # re-reading what's already there — no risk of overwriting history.
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": payload,
            "churn_probability": probability,
            "churn_prediction": will_churn,
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    except requests.exceptions.ConnectionError:
        st.error(
            "Couldn't reach the API. Is the Docker container running? "
            "(docker ps should show 'churn-api-test' as Up)"
        )
    except requests.exceptions.HTTPError:
        st.error(f"API returned an error: {response.status_code} — {response.text}")
