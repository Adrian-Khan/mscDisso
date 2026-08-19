import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv("outputs\data_collection_outputs\session_008_clean.csv") # path for windows 
sensor_signal = df.iloc[:, 0].values  # first sensor column

# plot autocorrelation up to 500 lags
plt.figure(figsize=(10, 4))
plot_acf(sensor_signal, lags=500)
plt.axhline(y=0.2, color='r', linestyle='--', label='Correlation Cutoff (0.2)')
plt.xlabel("Lag (steps)")
plt.ylabel("Autocorrelation")
plt.title("Sensor Autocorrelation Analysis")
plt.legend()
plt.show()