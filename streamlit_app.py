import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Page config (MUST be first)
# ---------------------------
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather Prediction System")
st.write("Predict weather using machine learning model")

# ---------------------------
# Load files safely
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/weather_model.pkl")

@st.cache_resource
def load_encoders():
    city_enc = joblib.load("model/city_encoder.pkl")
    weather_enc = joblib.load("model/weather_encoder.pkl")
    return city_enc, weather_enc

@st.cache_data
def load_data():
    return pd.read_csv("data/weather_dataset.csv")

try:
    model = load_model()
    city_encoder, weather_encoder = load_encoders()
    df = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------------------------
# Dataset preview
# ---------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Weather Inputs")

temperature = st.sidebar.slider("Temperature", 15.0, 45.0, 30.0)
humidity = st.sidebar.slider("Humidity", 20.0, 100.0, 60.0)
pressure = st.sidebar.slider("Pressure", 980.0, 1050.0, 1000.0)
wind_speed = st.sidebar.slider("Wind Speed", 0.0, 25.0, 10.0)
rainfall = st.sidebar.slider("Rainfall", 0.0, 200.0, 50.0)

city = st.sidebar.selectbox(
    "Select City",
    ["Nagpur","Mumbai","Pune","Delhi","Bangalore",
     "Chennai","Hyderabad","Kolkata","Jaipur","Ahmedabad"]
)

# ---------------------------
# Prediction
# ---------------------------
if st.sidebar.button("Predict Weather"):

    try:
        city_encoded = city_encoder.transform([city])[0]

        features = np.array([[
            temperature,
            humidity,
            pressure,
            wind_speed,
            rainfall,
            city_encoded
        ]])

        prediction = model.predict(features)
        predicted_weather = weather_encoder.inverse_transform(prediction)[0]

        st.success(f"🌤️ Predicted Weather: {predicted_weather}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------------------
# Chart
# ---------------------------
st.subheader("Temperature Distribution")

fig, ax = plt.subplots()
ax.hist(df["temperature"], bins=20)
st.pyplot(fig)