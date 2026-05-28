from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/weather_model.pkl")
city_encoder = joblib.load("model/city_encoder.pkl")
weather_encoder = joblib.load("model/weather_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get form values
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        pressure = float(request.form["pressure"])
        wind_speed = float(request.form["wind_speed"])
        rainfall = float(request.form["rainfall"])
        city = request.form["city"]

        # Encode city
        city_encoded = city_encoder.transform([city])[0]

        # Prepare input
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

        return render_template(
            "index.html",
            prediction_text=f"Predicted Weather: {predicted_weather}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)