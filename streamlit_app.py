import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and encoders
model = joblib.load("model/weather_model.pkl")

city_encoder = joblib.load("model/city_encoder.pkl")

weather_encoder = joblib.load("model/weather_encoder.pkl")

# Load dataset
df = pd.read_csv("data/weather_dataset.csv")

# Page title
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌦️",
    layout="centered"
)

# Title
st.title("🌦️ Weather Prediction System")

st.write("Enter weather details below")

st.subheader("Dataset Preview")

st.dataframe(df.head())

# Sidebar
st.sidebar.header("Weather Inputs")

# Inputs
temperature = st.slider(
    "Temperature",
    15.0, 45.0, 30.0
)

humidity = st.slider(
    "Humidity",
    20.0, 100.0, 60.0
)

pressure = st.slider(
    "Pressure",
    980.0, 1050.0, 1000.0
)

wind_speed = st.slider(
    "Wind Speed",
    0.0, 25.0, 10.0
)

rainfall = st.slider(
    "Rainfall",
    0.0, 200.0, 50.0
)

city = st.selectbox(
    "Select City",
    [
        "Nagpur",
        "Mumbai",
        "Pune",
        "Delhi",
        "Bangalore",
        "Chennai",
        "Hyderabad",
        "Kolkata",
        "Jaipur",
        "Ahmedabad"
    ]
)

# Prediction button
if st.button("Predict Weather"):

    # Encode city
    city_encoded = city_encoder.transform([city])[0]

    # Input array
    features = np.array([[
        temperature,
        humidity,
        pressure,
        wind_speed,
        rainfall,
        city_encoded
    ]])

    # Prediction
    prediction = model.predict(features)

    # Decode output
    predicted_weather = weather_encoder.inverse_transform(prediction)[0]

    # Show output
    st.success(f"Predicted Weather: {predicted_weather}")

# Temperature Chart
st.subheader("Temperature Distribution")

fig, ax = plt.subplots()

ax.hist(df["temperature"], bins=20)

st.pyplot(fig)