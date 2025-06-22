import streamlit as st
import pandas as pd
import joblib

# Load saved model and label encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🌾 Smart Crop Recommendation")

# Input sliders for farmer data
n = st.slider("Nitrogen (N)", 0, 140)
p = st.slider("Phosphorus (P)", 5, 145)
k = st.slider("Potassium (K)", 5, 205)
temp = st.slider("Temperature (°C)", 8, 45)
humidity = st.slider("Humidity (%)", 10, 100)
ph = st.slider("Soil pH", 3, 10)
rainfall = st.slider("Rainfall (mm)", 20, 300)

# Predict
if st.button("Get Recommendation"):
    input_df = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_df)[0]
    crop = label_encoder.inverse_transform([prediction])[0]
    st.success(f"✅ Recommended Crop: **{crop.upper()}**")
