
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

try:
    bundle = load_model()
    model = bundle["model"]
    model_columns = bundle["columns"]
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

st.title("📊 Customer Churn Prediction")
st.write(
    "Customer details enter karein aur predict karein "
    "ki customer service chhod sakta hai ya nahi."
)

st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=12
    )
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, value=840.0
    )

if st.button("Predict Churn", type="primary"):
    raw = pd.DataFrame([{
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment
    }])

    # Match the same one-hot encoding used during training
    encoded = pd.get_dummies(raw, drop_first=True, dtype=int)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)

    try:
        prediction = model.predict(encoded)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(encoded)[0]
            classes = list(model.classes_)
            churn_index = classes.index("Yes") if "Yes" in classes else (
                classes.index(1) if 1 in classes else None
            )

            if churn_index is not None:
                churn_probability = probabilities[churn_index] * 100
            else:
                churn_probability = None
        else:
            churn_probability = None

        st.subheader("Prediction Result")

        if prediction == "Yes" or prediction == 1:
            st.error("⚠️ Customer may churn (leave the service).")
        else:
            st.success("✅ Customer is predicted not to churn.")

        if churn_probability is not None:
            st.metric("Churn Probability", f"{churn_probability:.1f}%")

    except Exception as e:
        st.error(f"Prediction error: {e}")

