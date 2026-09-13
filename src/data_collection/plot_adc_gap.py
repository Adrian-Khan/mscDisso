"""
plot_adc_gap.py
session_008 ADC histogram with evaluation session peak ADCs marked.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_CSV = "outputs/testing_08/session_008_clean.csv"

EVAL_SESSIONS = [
    ("evaluation_01", "outputs/testing_01/testing_01_clean.csv"),
    ("evaluation_02", "outputs/testing_02/testing_02_clean.csv"),
    ("evaluation_03", "outputs/testing_03/testing_03_clean.csv"),
]

ADC_COLUMN = "adc_raw"
HIST_BINS = 100          

HIST_COLOR = "#1f77b4" 
LINE_COLOR = "black"        
TEXT_COLOR = "black"         

FIGSIZE = (9, 5.5)
DPI = 200

OUT_PNG = "figures/adc_gap.png"

Path("figures").mkdir(parents=True, exist_ok=True)

train = pd.read_csv(TRAIN_CSV)
train_adc = train[ADC_COLUMN].dropna()

eval_data = []
for label, path in EVAL_SESSIONS:
    df = pd.read_csv(path)
    peak_adc = df[ADC_COLUMN].max()
    eval_data.append((label, peak_adc))

sns.set_theme(style="ticks")  
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

bin_edges = range(
    int(train_adc.min() // HIST_BINS) * HIST_BINS,
    int(train_adc.max() // HIST_BINS + 2) * HIST_BINS,
    HIST_BINS,
)
ax.hist(train_adc, bins=list(bin_edges), color=HIST_COLOR, edgecolor="white",
        linewidth=0.3, label="session_008 (training) ADC distribution")

ymax = ax.get_ylim()[1]

eval_data_sorted = sorted(eval_data, key=lambda x: x[1])
y_fractions = [0.67, 0.80, 0.93] 

for (label, peak_adc), y_frac in zip(eval_data_sorted, y_fractions):
    ax.axvline(peak_adc, color=LINE_COLOR, linestyle="--", linewidth=1.4)
    ax.annotate(
        f"{label}\npeak = {peak_adc:,.0f}",
        xy=(peak_adc, ymax * y_frac),
        xytext=(10, 0),
        textcoords="offset points",
        color=TEXT_COLOR,
        fontsize=9,
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85)
    )

ax.set_xlabel("Raw ADC value", color=TEXT_COLOR)
ax.set_ylabel("Sample count (session_008)", color=TEXT_COLOR)
ax.set_title("session_008 ADC distribution and evaluation-session peak ADC values", color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR, which="both")

legend = ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

ax.margins(x=0.02)
sns.despine(fig)

fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"Saved {OUT_PNG}")