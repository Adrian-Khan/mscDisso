"""
data_inspection.py
inspect the data collected in the csv files from capture.py 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#  change global var for session 
SESSION = 'session_005.csv'

df = pd.read_csv(SESSION)

# filter implausible electrical values before missingness analysis
df = df[(df['adc_raw'] > 10) & (df['adc_raw'] < 4085) | df['adc_raw'].isna()]
print(f"INSPECTING: {SESSION}")
print(f"Columns: {list(df.columns)}")

# missingness 
df = df.reset_index(drop=True)

# find how many consecutive NaN rows exist in strain_ratio
df['strain_nan'] = df['strain_ratio'].isna()
df['gap_group'] = (df['strain_nan'] != df['strain_nan'].shift()).cumsum()

gap_sizes = df[df['strain_nan']].groupby('gap_group').size()
if len(gap_sizes) > 0:
    print(f"\nMISSING VALUE ANALYSIS")
    print(f"Total missing strain rows: {df['strain_nan'].sum()}")
    print(f"Number of separate gaps: {len(gap_sizes)}")
    print(f"Gap size distribution:")
    print(f"  1-5 frames (interpolatable): {(gap_sizes <= 5).sum()} gaps")
    print(f"  6-30 frames (borderline): {((gap_sizes > 5) & (gap_sizes <= 30)).sum()} gaps")
    print(f"  >30 frames (drop): {(gap_sizes > 30).sum()} gaps")
    print(f"  Largest gap: {gap_sizes.max()} frames")
else:
    print("No missing strain values found.")


MAX_INTERP_GAP = 5

df['strain_ratio'] = df['strain_ratio'].interpolate(
    method='linear',
    limit=MAX_INTERP_GAP,
    limit_direction='forward'
)
df['adc_raw'] = df['adc_raw'].interpolate(
    method='linear',
    limit=MAX_INTERP_GAP,
    limit_direction='forward'
)

# drop remaining rows where strain still missing (long gaps)
df = df.dropna(subset=['adc_raw', 'strain_ratio'])
print(f"\nAfter interpolation and dropping long gaps: {len(df)}")

# mark sequence boundaries at long gaps
# any timestamp jump > 200ms is a sequence boundary   ---------> 200 was arbitrary, can change to 500 maybe so half a second? 
# sequences should not cross these boundaries during training
df['time_gap'] = df['timestamp_ms'].diff()
df['sequence_break'] = df['time_gap'] > 200

n_breaks = df['sequence_break'].sum()
print(f"Sequence breaks (gaps > 200ms): {n_breaks}")
print(f"This gives approximately {n_breaks + 1} continuous segments")


# filter 
df = df[(df['strain_ratio'] >= 0.5) & (df['strain_ratio'] <= 2.5)]
print(f"Clean rows after filtering: {len(df)}")

# summary stats 
print(f"\nADC RAW")
print(f"Min: {df['adc_raw'].min()}")
print(f"Max: {df['adc_raw'].max()}")
print(f"Mean: {df['adc_raw'].mean():.1f}")
print(f"Std: {df['adc_raw'].std():.1f}")
print(f"Range (max-min): {df['adc_raw'].max() - df['adc_raw'].min()}")

print(f"\nSTRAIN RATIO")
print(f"Min: {df['strain_ratio'].min():.4f}")
print(f"Max: {df['strain_ratio'].max():.4f}")
print(f"Mean: {df['strain_ratio'].mean():.4f}")
print(f"Std: {df['strain_ratio'].std():.4f}")

print(f"\nSYNC CHECK")
gaps = df['timestamp_ms'].diff().dropna()
print(f"Median gap: {gaps.median():.1f} ms")
print(f"Max gap: {gaps.max():.1f} ms")
print(f"Gaps > 100ms: {(gaps > 100).sum()}")

# plots 
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

t_min = df['timestamp_ms'] / 1000 / 60

# full session ADC
axes[0].plot(t_min, df['adc_raw'], color='steelblue', linewidth=0.4)
axes[0].set_ylabel('ADC Raw')
axes[0].set_xlabel('Time (minutes)')
axes[0].set_title(f'{SESSION} - ADC signal full session')
axes[0].grid(True, alpha=0.3)

# full session strain
axes[1].plot(t_min, df['strain_ratio'], color='orange', linewidth=0.4)
axes[1].set_ylabel('Strain Ratio')
axes[1].set_xlabel('Time (minutes)')
axes[1].set_title(f'{SESSION} - Vision ground truth full session')
axes[1].grid(True, alpha=0.3)

# first 2 minutes zoomed
sample = df[df['timestamp_ms'] < df['timestamp_ms'].min() + 120000].copy()
t_sec = sample['timestamp_ms'] / 1000
axes[2].plot(t_sec, sample['adc_raw'],
             color='steelblue', linewidth=0.8, label='ADC')
ax2b = axes[2].twinx()
ax2b.plot(t_sec, sample['strain_ratio'],
          color='orange', linewidth=0.8, label='Strain')
axes[2].set_ylabel('ADC Raw', color='steelblue')
ax2b.set_ylabel('Strain Ratio', color='orange')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_title('First 2 minutes - ADC and strain overlaid')
axes[2].grid(True, alpha=0.3)

# hysteresis scatter
axes[3].scatter(df['adc_raw'], df['strain_ratio'],
                alpha=0.05, s=1, color='purple')
axes[3].set_xlabel('ADC Raw')
axes[3].set_ylabel('Strain Ratio')
axes[3].set_title('Hysteresis plot - ADC vs Strain')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
outname = SESSION.replace('.csv', '_inspection.png')
plt.savefig(outname, dpi=150)
plt.show()
print(f"\nPlot saved as {outname}")



clean_name = SESSION.replace('.csv', '_clean.csv')
df.to_csv(clean_name, index=False)
print(f"Clean data saved: {clean_name}")