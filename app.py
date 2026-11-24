
import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "churn_scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "churn_threshold.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, scaler, features, threshold

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["customerID", "Churn"], errors="ignore")
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["TotalCharges"].median()
    )
    return data

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.write("Predict customer churn risk using Machine Learning.")

try:
    model, scaler, features, threshold = load_model()
    raw_data = load_data()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

st.sidebar.header("Customer Information")

with st.sidebar.form("customer_form"):

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

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

    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
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

    monthly = st.number_input(
        "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
    )
    total = st.number_input(
        "Total Charges", min_value=0.0, max_value=10000.0, value=840.0
    )

    submitted = st.form_submit_button("Predict Churn")

if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
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
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    combined = pd.concat(
        [raw_data, customer], ignore_index=True
    )

    categorical_columns = combined.select_dtypes(
        include="object"
    ).columns

    encoded = pd.get_dummies(
        combined,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    customer_encoded = encoded.iloc[[-1]].reindex(
        columns=features, fill_value=0
    )

    customer_scaled = scaler.transform(customer_encoded)

    probability = model.predict_proba(customer_scaled)[0][1]
    prediction = int(probability >= threshold)

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Model Threshold",
            f"{threshold * 100:.2f}%"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ High Churn Risk: Customer may leave.")

        st.write("**Suggested Retention Actions:**")
        st.write("- Offer a personalized discount.")
        st.write("- Provide a suitable long-term contract.")
        st.write("- Contact the customer to understand concerns.")

    else:
        st.success("✅ Low Churn Risk: Customer is likely to stay.")

        st.write("**Suggested Actions:**")
        st.write("- Maintain good customer service.")
        st.write("- Offer loyalty benefits.")
        st.write("- Continue monitoring customer satisfaction.")

st.divider()
st.caption("Customer Churn Prediction | Machine Learning Project")
