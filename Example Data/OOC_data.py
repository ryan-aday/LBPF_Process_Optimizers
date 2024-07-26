import pandas as pd
import numpy as np

# Generating out of control data
np.random.seed(42)

temperature_out_of_control = np.random.normal(loc=20, scale=1, size=95).tolist() + [25, 26, 27, 28, 29]  # Mean 20, SD 1 with outliers
humidity_out_of_control = np.random.normal(loc=50, scale=5, size=95).tolist() + [30, 32, 34, 36, 38]  # Mean 50, SD 5 with outliers

out_of_control_data = pd.DataFrame({
    'Temperature': temperature_out_of_control,
    'Humidity': humidity_out_of_control
})

out_of_control_data.to_csv('out_of_control.csv', index=False)
