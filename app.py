import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load models
# -----------------------------
log_model = joblib.load("logistic_regression_churn_model.pkl")
tree_model = joblib.load("decision_tree_churn_model.pkl")
scaler = joblib.load("scaler.pkl")


df = pd.read_csv("Telco-Customer-Churn.csv", sep=";")
df.columns = df.columns.str.strip()
#st.write(df.columns.tolist())

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Dashboard Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree"]
)

st.title("Customer Churn Prediction Dashboard")
st.write("Interactive dashboard for customer churn analysis and prediction.")

# -----------------------------
# Top KPI cards
# -----------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Selected Model", model_choice)

with kpi2:
    st.metric("Dataset Rows", f"{len(df)}")

with kpi3:
    churn_rate = (df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100)
    st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")

with kpi4:
    st.metric("App Status", "Ready")

st.divider()

# -----------------------------
# Main layout
# -----------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Customer Input Form")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", [0, 1])
    dependents = st.selectbox("Dependents", [0, 1])
    tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", [0, 1])
    multiple_lines = st.selectbox("Multiple Lines", [0, 1])
    internet_service = st.selectbox("Internet Service", [0, 1])
    online_security = st.selectbox("Online Security", [0, 1])
    online_backup = st.selectbox("Online Backup", [0, 1])
    device_protection = st.selectbox("Device Protection", [0, 1])
    tech_support = st.selectbox("Tech Support", [0, 1])
    streaming_tv = st.selectbox("Streaming TV", [0, 1])
    streaming_movies = st.selectbox("Streaming Movies", [0, 1])
    paperless_billing = st.selectbox("Paperless Billing", [0, 1])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

    predict_btn = st.button("Predict Churn")

with right_col:
    st.subheader("Prediction Result")

    if predict_btn:
        gender_value = 1 if gender == "Male" else 0

        contract_one_year = 1 if contract == "One year" else 0
        contract_two_year = 1 if contract == "Two year" else 0

        payment_credit_card = 1 if payment_method == "Credit card (automatic)" else 0
        payment_electronic_check = 1 if payment_method == "Electronic check" else 0
        payment_mailed_check = 1 if payment_method == "Mailed check" else 0

        input_data = pd.DataFrame([{
            "gender": gender_value,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "PaperlessBilling": paperless_billing,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract_One year": contract_one_year,
            "Contract_Two year": contract_two_year,
            "PaymentMethod_Credit card (automatic)": payment_credit_card,
            "PaymentMethod_Electronic check": payment_electronic_check,
            "PaymentMethod_Mailed check": payment_mailed_check
        }])

        if model_choice == "Logistic Regression":
            input_scaled = scaler.transform(input_data)
            prediction = log_model.predict(input_scaled)[0]
            probability = log_model.predict_proba(input_scaled)[0][1]
        else:
            prediction = tree_model.predict(input_data)[0]
            probability = tree_model.predict_proba(input_data)[0][1]

        result_cols = st.columns(3)
        with result_cols[0]:
            st.metric("Prediction", "Yes" if prediction == 1 else "No")
        with result_cols[1]:
            st.metric("Probability", f"{probability:.2%}")
        with result_cols[2]:
            if probability < 0.40:
                risk = "Low"
            elif probability < 0.70:
                risk = "Medium"
            else:
                risk = "High"
            st.metric("Risk Level", risk)

        if probability < 0.40:
            st.success("Low churn risk")
        elif probability < 0.70:
            st.warning("Medium churn risk")
        else:
            st.error("High churn risk")

        st.progress(float(probability))

        st.write("### Input Summary")
        st.dataframe(input_data)

st.divider()

# -----------------------------
# Charts
# -----------------------------
st.subheader("Churn Insights")

chart1, chart2 = st.columns(2)

with chart1:
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.countplot(x="Contract", hue="Churn", data=df, ax=ax1)
    ax1.set_title("Contract Type vs Customer Churn")
    st.pyplot(fig1)

with chart2:
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax2)
    ax2.set_title("Monthly Charges vs Churn")
    st.pyplot(fig2)