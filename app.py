import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
# This requires the 'churn_model.pkl' file to be in the repository.
# We will need to upload that file as well.
try:
    model = joblib.load('churn_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'churn_model.pkl' is uploaded to the GitHub repository.")
    st.stop()

# Set up the Streamlit page
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Prediction 🔮")

# Create columns for inputs for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    tenure = st.slider("Tenure (months)", 0, 72, 24)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col3:
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.slider("Total Charges ($)", 18.0, 9000.0, 1400.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Create a dictionary of the inputs to pass to the model
input_data = {
    'gender': gender, 'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
    'PhoneService': phone_service, 'MultipleLines': multiple_lines,
    'InternetService': internet_service, 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': 'Yes',
    'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
    'TotalCharges': float(total_charges)
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Encode the categorical variables in the input DataFrame
for column in input_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    input_df[column] = le.fit_transform(input_df[column])

# Prediction button
if st.button("Predict Churn"):
    prediction_proba = model.predict_proba(input_df)[:, 1]
    churn_risk = prediction_proba[0]

    st.subheader("Prediction Result")
    if churn_risk > 0.5:
        st.error(f"High Churn Risk! Probability: {churn_risk:.2f}")
    else:
        st.success(f"Low Churn Risk. Probability: {churn_risk:.2f}")
