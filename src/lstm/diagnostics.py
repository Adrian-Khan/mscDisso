"""
diagnostics.py

Analyses raw vs. processed strain gauge & ADC data, detects alignment lag from cleaned dataset,
and measures baseline linear correlation for train.py data pre-processing.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# config 
DATA_PATH = Path('outputs/testing_01/testing_01_clean.csv') # using the cleaned dataset from data_inspection.py

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Target CSV not found at: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)

adc_raw = df['adc_raw'].values.astype(np.float32)
strain_raw = df['strain_ratio'].values.astype(np.float32)


# run linear regression cause there should be at least some type of R2 if the data was collected correctly (this will differ depending on length of data collection)
X_raw = adc_raw.reshape(-1, 1)
y_raw = strain_raw

model_raw = LinearRegression()
model_raw.fit(X_raw, y_raw)
y_pred_raw = model_raw.predict(X_raw)

mse_raw = mean_squared_error(y_raw, y_pred_raw)
rmse_raw = np.sqrt(mse_raw)
mae_raw = mean_absolute_error(y_raw, y_pred_raw)
r2_raw = r2_score(y_raw, y_pred_raw)

print("\n===============================================================================")
print(f"Baseline Linear Regression Metrics")
print("\n===============================================================================")
print(f"Slope (Coefficient): {model_raw.coef_[0]:.6f}")
print(f"Intercept: {model_raw.intercept_:.6f}")
print(f"Mean Squared Error: {mse_raw:.6f}")
print(f"Root Mean Sq Error: {rmse_raw:.6f}")
print(f"Mean Absolute Error: {mae_raw:.6f}")
print(f"R2 Score: {r2_raw:.4f}\n")

# cross correlation and lag detection
# standardize signals (zero mean, unit variance) for normalized correlation
adc_std = (adc_raw - np.mean(adc_raw)) / (np.std(adc_raw) + 1e-8)
strain_std = (strain_raw - np.mean(strain_raw)) / (np.std(strain_raw) + 1e-8)

corr = np.correlate(adc_std, strain_std, mode='full') / len(adc_raw)
lags = np.arange(-len(adc_raw) + 1, len(adc_raw))
best_lag = lags[np.argmax(corr)]

print("\n===============================================================================")
print("Cross-Correlation Alignment")
print("\n===============================================================================")
print(f"Peak Correlation Lag: {best_lag} samples")
print(f"Max Correlation Value: {corr.max():.4f}\n")

# lag corrected regression and metrics
if best_lag > 0:
    adc_aligned = adc_raw[:-best_lag]
    strain_aligned = strain_raw[best_lag:]
elif best_lag < 0:
    abs_lag = abs(best_lag)
    adc_aligned = adc_raw[abs_lag:]
    strain_aligned = strain_raw[:-abs_lag]
else:
    adc_aligned = adc_raw
    strain_aligned = strain_raw

X_aligned = adc_aligned.reshape(-1, 1)
y_aligned = strain_aligned

model_aligned = LinearRegression()
model_aligned.fit(X_aligned, y_aligned)
y_pred_aligned = model_aligned.predict(X_aligned)

r2_aligned = r2_score(y_aligned, y_pred_aligned)
corr_aligned = np.corrcoef(adc_aligned, strain_aligned)[0,1]

print("\n===============================================================================")
print(f"Post-Alignment Linear Regression Metrics")
print("\n===============================================================================")
print(f"Aligned R2 Score (Lag={best_lag}): {r2_aligned:.4f}\n")
print(f"Aligned correlation metrics (Lag={best_lag}): {corr_aligned:.4f}\n ")

# plots 
fig, axes = plt.subplots(3, 1, figsize=(14, 11))

# 1: make a dual axis raw time series 
color_adc = 'blue'
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('ADC Raw', color=color_adc)
line1 = axes[0].plot(adc_raw, color=color_adc, linewidth=1, label='ADC Raw')
axes[0].tick_params(axis='y', labelcolor=color_adc)
axes[0].grid(True, alpha=0.3)

ax0_twin = axes[0].twinx()
color_strain = 'orange'
ax0_twin.set_ylabel('Strain Ratio Raw', color=color_strain)
line2 = ax0_twin.plot(strain_raw, color=color_strain, linewidth=1, label='Strain Ratio Raw')
ax0_twin.tick_params(axis='y', labelcolor=color_strain)

lines = line1 + line2
labels = [l.get_label() for l in lines]
axes[0].legend(lines, labels, loc='upper right')
axes[0].set_title(f'Raw ADC vs Raw Strain Ratio Over Time')

# 2: cross correlation lag peak 
axes[1].plot(lags, corr, color='purple', linewidth=0.8)
axes[1].set_title('Cross-Correlation (ADC vs Strain)')
axes[1].set_xlabel('Lag (samples)')
axes[1].set_ylabel('Correlation')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero Lag')
axes[1].axvline(x=best_lag, color='green', linestyle='--', linewidth=1, label=f'Peak Lag: {best_lag}')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# 3: scatter and regressions 
axes[2].scatter(adc_raw, strain_raw, alpha=0.15, color='gray', s=8, label='Raw Data Points')
axes[2].plot(adc_raw, y_pred_raw, color='red', linewidth=2, label=f'Raw Fit (R2 = {r2_raw:.3f})')
axes[2].plot(adc_aligned, y_pred_aligned, color='green', linewidth=2, linestyle='--', label=f'Aligned Fit (R2 = {r2_aligned:.3f})')
axes[2].set_title('ADC Raw → Strain Ratio Raw Regression Fit')
axes[2].set_xlabel('ADC Raw')
axes[2].set_ylabel('Strain Ratio Raw')
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
output_img = f'diagnostics_raw_vs_strain_test02.png'
plt.savefig(output_img, dpi=150)
plt.show()

print(f"Diagnostics complete. Visualisation saved to {output_img}")