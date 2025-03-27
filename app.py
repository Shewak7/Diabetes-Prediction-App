import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
data = pd.read_csv("diabetes.csv")

# Remove 'Pregnancies' and 'SkinThickness' to make it unisex
data = data.drop(columns=['Pregnancies', 'SkinThickness'])

# Define features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Ensure consistent column order
FEATURE_ORDER = X.columns.tolist()  # Store feature order

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the feature names for later use
pickle.dump(FEATURE_ORDER, open("feature_order.pkl", "wb"))

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Load trained model, scaler, and feature order
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
FEATURE_ORDER = pickle.load(open("feature_order.pkl", "rb"))

# Streamlit UI
st.title("Diabetes Prediction App")

# User inputs
glucose = st.number_input("Glucose Level (mg/dL)", 0, 200)
bmi = st.number_input("BMI", 0.0, 50.0)
blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 150)
insulin = st.number_input("Insulin Level (pmol/L)", 0, 300)
age = st.number_input("Age", 1, 100)

# Ask family history questions to calculate DPF
st.subheader("Family History Questions")
parents = st.radio("Do your parents have diabetes?", ["No", "Yes"])
siblings = st.radio("Do your siblings have diabetes?", ["No", "Yes"])
grandparents = st.radio("Do your grandparents have diabetes?", ["No", "Yes"])

# Calculate DPF Score
dpf_score = 0.0
if parents == "Yes":
    dpf_score += 0.3
if siblings == "Yes":
    dpf_score += 0.2
if grandparents == "Yes":
    dpf_score += 0.2

# Predict button
if st.button("Predict"):
    # Prepare input data as a DataFrame in the correct order
    input_data = pd.DataFrame([[glucose, bmi, blood_pressure, insulin, dpf_score, age]], 
                              columns=["Glucose", "BMI", "BloodPressure", "Insulin", "DiabetesPedigreeFunction", "Age"])

    # Reorder columns to match training data
    input_data = input_data[FEATURE_ORDER]

    # Apply the same scaling as in training
    input_data = scaler.transform(input_data)

    # Make prediction
    result = model.predict(input_data)

    # Display result
    st.write("Diabetes Prediction:", "Positive" if result[0] == 1 else "Negative")
