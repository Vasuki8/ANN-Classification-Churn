import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

model = tf.keras.models.load_model('model.keras')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 0, 100, 15)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_product = st.slider("Number of products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = {
    "CreditScore": credit_score,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_product,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
}

# Create DataFrame with scalar values
input_data_df = pd.DataFrame([input_data])

# Encode Gender
input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]])

geo_columns = onehot_encoder_geo.get_feature_names_out()
input_data_df[geo_columns] = geo_encoded

# Scale features
input_data_scaled = scaler.transform(input_data_df)

# Make prediction
prediction = model.predict(input_data_scaled)
st.write(f"Prediction: {prediction[0][0]}")