import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Load the saved model and preprocessor
model = keras.models.load_model('customer_churn_model.keras')
preprocessor = joblib.load('preprocessor.pkl')

# Streamlit UI
st.title("Customer Churn Prediction 🏦")
st.markdown("Predict if a customer will churn using a Neural Network.")

# Input form
st.sidebar.header("Customer Details")

# Numerical Inputs
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (Years with Bank)", 0, 15, 5)
balance = st.sidebar.number_input("Account Balance", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", [1, 0])
is_active_member = st.sidebar.selectbox("Is Active Member?", [1, 0])
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Categorical Inputs
geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
    "Geography": [geography],
    "Gender": [gender]
})

# Preprocess input data
processed_input = preprocessor.transform(input_data)

# Predict churn probability
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(processed_input)
    churn_prob = prediction[0][0]
    
    st.subheader("Prediction Result")
    if churn_prob > 0.5:
        st.error(f"🚨 High Risk of Churn ({churn_prob:.2%} Probability)")
    else:
        st.success(f"✅ Low Risk of Churn ({churn_prob:.2%} Probability)")

# Display input data
st.write("### Input Data")
st.dataframe(input_data)