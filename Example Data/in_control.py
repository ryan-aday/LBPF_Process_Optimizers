import pandas as pd
import numpy as np

# Generating in control data
np.random.seed(42)

temperature_in_control = np.random.normal(loc=20, scale=1, size=100)  # Mean 20, SD 1
humidity_in_control = np.random.normal(loc=50, scale=5, size=100)  # Mean 50, SD 5

in_control_data = pd.DataFrame({
    'Temperature': temperature_in_control,
    'Humidity': humidity_in_control
})

in_control_data.to_csv('in_control.csv', index=False)