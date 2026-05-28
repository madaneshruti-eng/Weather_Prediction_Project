import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

num_rows = 30000

cities = [
    "Nagpur", "Mumbai", "Pune", "Delhi",
    "Bangalore", "Chennai", "Hyderabad",
    "Kolkata", "Jaipur", "Ahmedabad"
]

weather_conditions = [
    "Sunny", "Cloudy", "Rainy",
    "Stormy", "Foggy"
]

start_date = datetime(2023, 1, 1)

data = []

for i in range(num_rows):

    date = start_date + timedelta(days=i % 365)

    temperature = round(random.uniform(15, 45), 1)

    humidity = random.randint(20, 100)

    pressure = random.randint(980, 1050)

    wind_speed = round(random.uniform(0, 25), 1)

    rainfall = round(random.uniform(0, 200), 1)

    city = random.choice(cities)

    weather_condition = random.choice(weather_conditions)

    data.append([
        date.strftime("%Y-%m-%d"),
        city,
        temperature,
        humidity,
        pressure,
        wind_speed,
        rainfall,
        weather_condition
    ])

df = pd.DataFrame(data, columns=[
    "date",
    "city",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rainfall",
    "weather_condition"
])

df.to_csv("data/weather_dataset.csv", index=False)

print("Dataset Generated Successfully!")
print(df.head())