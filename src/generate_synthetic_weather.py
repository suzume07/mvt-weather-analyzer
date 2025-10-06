"""
generate_synthetic_weather.py
Tạo dataset giả hourly để test pipeline.
"""

import numpy as np
import pandas as pd

def generate_synthetic_weather(hours=24*30, start='2025-09-01'):
    """
    Sinh dữ liệu hourly trong 'hours' giờ.
    Trả về DataFrame gồm: time, temperature, humidity, wind_direction (degrees), rainfall
    """
    np.random.seed(42)
    t = pd.date_range(start=start, periods=hours, freq='H')
    # daily sinusoidal + small trend + gaussian noise
    temp = 15 + 8 * np.sin(2 * np.pi * (np.arange(hours) % 24) / 24) + 0.01 * np.arange(hours) + np.random.normal(0, 0.8, hours)
    humidity = 60 - 10 * np.sin(2 * np.pi * (np.arange(hours) % 24) / 24) + np.random.normal(0, 3, hours)
    # wind direction slowly varying
    wind_dir = (np.cumsum(np.random.normal(0, 2, hours)) + 180) % 360
    # sparse rainfall spikes
    rainfall = np.random.poisson(0.02, hours) * np.random.exponential(5, hours)
    df = pd.DataFrame({
        'time': t,
        'temperature': temp,
        'humidity': humidity,
        'wind_direction': wind_dir,
        'rainfall': rainfall
    })
    return df

if __name__ == "__main__":
    df = generate_synthetic_weather()
    df.to_csv("data/raw_weather_synthetic.csv", index=False)
    print("Saved synthetic data to data/raw_weather_synthetic.csv")
