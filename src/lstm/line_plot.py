"""
ADC to strain calibration curve for all four sessions, binned,
overlaid, with the same training-coverage gap as identified before 
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# config 
TRAIN_LABEL = "session_008 (train)"
TRAIN_CSV = "outputs/testing_08/session_008_clean.csv"

EVAL_SESSIONS = [
    ("evaluation_01", "outputs/testing_01/testing_01_clean.csv"),
    ("evaluation_02", "outputs/testing_02/testing_02_clean.csv"),
    ("evaluation_03", "outputs/testing_03/testing_03_clean.csv"),
]

ADC_COLUMN = "adc_raw"
STRAIN_COLUMN = "strain_ratio"

BIN_WIDTH = 150             
MIN_SAMPLES_PER_BIN = 8    

GAP_BAND = (2500, 3800)
GAP_SHADE_COLOR = "#fff2ae" # yellow
GAP_SHADE_ALPHA = 0.65

TEXT_COLOR = "black"

FIGSIZE = (9, 6)
DPI = 200

OUT_PNG = "figures/fig_curve_shift.png"
Path("figures").mkdir(parents=True, exist_ok=True)

# helper funcs 
def binned_calibration_curve(df, bin_width=BIN_WIDTH, min_samples=MIN_SAMPLES_PER_BIN):
    df = df.dropna(subset=[ADC_COLUMN, STRAIN_COLUMN]).copy()
    lo = int(df[ADC_COLUMN].min() // bin_width) * bin_width
    hi = int(df[ADC_COLUMN].max() // bin_width + 2) * bin_width
    edges = range(lo, hi, bin_width)
    df["bin"] = pd.cut(df[ADC_COLUMN], bins=list(edges), right=False)

    grp = df.groupby("bin", observed=True).agg(
        count=(ADC_COLUMN, "size"),
        mean_strain=(STRAIN_COLUMN, "mean"),
    )
    grp = grp[grp["count"] >= min_samples]
    grp["bin_center"] = grp.index.map(lambda iv: iv.mid).astype(float)
    return grp.sort_values("bin_center")[["bin_center", "mean_strain", "count"]].reset_index(drop=True)


# load and compute 
n_sessions = 1 + len(EVAL_SESSIONS)
line_colors = sns.color_palette("colorblind", n_colors=n_sessions)

sessions = [(TRAIN_LABEL, TRAIN_CSV, line_colors[0])]
sessions += [(label, path, color) for (label, path), color in zip(EVAL_SESSIONS, line_colors[1:])]

curves = []
for label, path, color in sessions:
    df = pd.read_csv(path)
    curve = binned_calibration_curve(df)
    curves.append((label, curve, color))

# plot
sns.set_theme(style="ticks")
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

ax.axvspan(*GAP_BAND, color=GAP_SHADE_COLOR, alpha=GAP_SHADE_ALPHA,
           label=f"training-coverage gap ({GAP_BAND[0]}-{GAP_BAND[1]} ADC)", zorder=0)

for label, curve, color in curves:
    ax.plot(curve["bin_center"], curve["mean_strain"], marker="o", markersize=4,
            linewidth=1.8, color=color, label=label)

ax.set_xlabel(f"ADC raw (binned, {BIN_WIDTH}-count bins, >={MIN_SAMPLES_PER_BIN} samples/bin)", color=TEXT_COLOR)
ax.set_ylabel("Mean strain ratio", color=TEXT_COLOR)
ax.set_title("ADC to strain calibration curve across sessions", color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR, which="both")

legend = ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

ax.margins(x=0.02)
sns.despine(fig)

fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"Saved {OUT_PNG}")