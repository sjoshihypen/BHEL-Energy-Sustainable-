import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points
num_samples = 1000

# Generate synthetic weather data
wind_speed = np.random.uniform(0, 150, num_samples)  # Wind speed in km/h
precipitation = np.random.uniform(0, 200, num_samples)  # Precipitation in mm
temperature = np.random.uniform(-20, 50, num_samples)  # Temperature in °C

# Generate synthetic damage data
# Assume damage occurs if wind speed > 100 km/h or precipitation > 150 mm
damage_occurred = (wind_speed > 100) | (precipitation > 150)
damage_cost = damage_occurred * np.random.uniform(1000, 50000, num_samples)  # Damage cost in USD

# Create a DataFrame
data = pd.DataFrame({
    'wind_speed': wind_speed,
    'precipitation': precipitation,
    'temperature': temperature,
    'damage_occurred': damage_occurred.astype(int),  # Convert boolean to int (1 for True, 0 for False)
    'damage_cost': damage_cost
})

# Save to CSV
data.to_csv('historical_weather_damage_data.csv', index=False)
