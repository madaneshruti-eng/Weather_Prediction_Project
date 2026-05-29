import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/weather_dataset.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Encode categorical columns
city_encoder = LabelEncoder()
weather_encoder = LabelEncoder()

df["city"] = city_encoder.fit_transform(df["city"])
df["weather_condition"] = weather_encoder.fit_transform(df["weather_condition"])

# Features and target
X = df[[
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rainfall",
    "city"
]]

y = df["weather_condition"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Optimized smaller model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save compressed model
joblib.dump(model, "model/weather_model.pkl", compress=3)

# Save encoders
joblib.dump(city_encoder, "model/city_encoder.pkl")
joblib.dump(weather_encoder, "model/weather_encoder.pkl")

print("Model Saved Successfully!")