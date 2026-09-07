import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# config 
DATA_FILE = "/scratch0/adrikhan/sensor_project/session_008_clean.csv"
NORM_FILE = "norm_params.json"
TARGET_COL = "strain_norm"
BREAK_COL = "sequence_break"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE_HZ = 30    
EVAL_BATCH_SIZE = 64   # batching for evaluation only

KNOWN_CONFIGS = {
    "adc_grad": {
        "weights_file": "CNN_LSTM_Attention_adc_grad_best.pth",
        "feature_cols": ["adc_norm", "adc_grad_norm"],
        "window_size": 120,
        "hidden_size": 64,
        "conv1_filters": 64,
        "kernel_size": 5,
        "dropout": 0.2,
    },
    "adc_diff": {
        "weights_file": "CNN_LSTM_Attention_adc_diff_best.pth",
        "feature_cols": ["adc_norm", "adc_diff_norm"],
        "window_size": 60,
        "hidden_size": 64,
        "conv1_filters": 64,
        "kernel_size": 7,
        "dropout": 0.35,
    },
}

# define models 
class Attention(nn.Module):
    def __init__(self, hidden_size, attn_dim=32):
        super().__init__()
        self.attn = nn.Linear(hidden_size, attn_dim)
        self.context = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, lstm_out):
        score = torch.tanh(self.attn(lstm_out))
        weight = torch.softmax(self.context(score), dim=1)
        out = (weight * lstm_out).sum(dim=1)
        return out, weight.squeeze(-1)


class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, conv1_filters=32, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=conv1_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv1_filters, out_channels=conv1_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv1_filters * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention=False):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out, weights = self.attention(lstm_out)
        out = self.dropout(out)
        out = self.fc(out)
        if return_attention:
            return out, weights
        return out

# daa prep and norm 
def load_and_preprocess(feature_cols, window_size):
    df = pd.read_csv(DATA_FILE)

    # derivatives if present in features
    adc_raw = df["adc_raw"].values.astype(np.float32)
    df["adc_grad_raw"] = np.gradient(adc_raw).astype(np.float32)
    df["adc_diff_raw"] = np.diff(adc_raw, prepend=adc_raw[0]).astype(np.float32)
    df["adc_grad2_raw"] = np.gradient(df["adc_grad_raw"].values).astype(np.float32)

    for lag in [1, 3, 5]:
        df[f"adc_lag_{lag}_raw"] = df["adc_raw"].shift(lag).bfill().astype(np.float32)

    #  norm params 
    with open(NORM_FILE, "r") as f:
        norm_params = json.load(f)

    for col in ["adc_raw", "strain_ratio", "adc_grad_raw", "adc_diff_raw", "adc_grad2_raw", "adc_lag_1_raw", "adc_lag_3_raw", "adc_lag_5_raw"]:
        if f"{col}_min" in norm_params:
            c_min = norm_params[f"{col}_min"]
            c_max = norm_params[f"{col}_max"]
            out_col = "strain_norm" if col == "strain_ratio" else col.replace("_raw", "") + "_norm"
            df[out_col] = (df[col] - c_min) / (c_max - c_min + 1e-8)

    # rolling statistics if applicable
    if any("adc_roll_" in c for c in feature_cols):
        w_short = max(3, int(window_size * 0.10))
        w_long = max(5, int(window_size * 0.25))
        df["adc_roll_mean_short_norm"] = df["adc_norm"].rolling(window=w_short, min_periods=1).mean().astype(np.float32)
        df["adc_roll_std_short_norm"] = df["adc_norm"].rolling(window=w_short, min_periods=1).std().fillna(0).astype(np.float32)
        df["adc_roll_mean_long_norm"] = df["adc_norm"].rolling(window=w_long, min_periods=1).mean().astype(np.float32)

    # extract test split (last 15% of sequence data)
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    df_test = df.iloc[n_train + n_val:]

    # create sequences
    features = df_test[feature_cols].values
    targets = df_test[TARGET_COL].values
    breaks = df_test[BREAK_COL].values.astype(bool)

    X, y = [], []
    start_indices = []
    for i in range(len(df_test) - window_size):
        if breaks[i + 1: i + window_size + 1].any():
            continue
        X.append(features[i: i + window_size])
        y.append([targets[i + window_size - 1]])
        start_indices.append(i)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    start_indices = np.array(start_indices, dtype=np.int64)

    test_dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    return test_loader, norm_params, X, start_indices

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=sorted(KNOWN_CONFIGS.keys()), required=True,
                         help="Which Phase 1 feature set's CNN-LSTM-Attention model to plot.")
    parser.add_argument("--weights", default=None,
                         help="Override the checkpoint path if it differs from KNOWN_CONFIGS.")
    args = parser.parse_args()

    config = KNOWN_CONFIGS[args.feature_set]
    feature_cols = config["feature_cols"]
    window_size = config["window_size"]
    hidden_size = config["hidden_size"]
    conv1_filters = config["conv1_filters"]
    kernel_size = config["kernel_size"]
    dropout = config["dropout"]
    weights_path = args.weights or config["weights_file"]
    model_tag = args.feature_set

    print(f"Feature set: {args.feature_set}  (columns: {feature_cols})")
    print(f"Loading model weights from: {weights_path}")
    print(f"Architecture: window={window_size}, hidden={hidden_size}, "
          f"conv1_filters={conv1_filters}, kernel={kernel_size}, dropout={dropout}")

    test_loader, norm_params, X_all, start_indices = load_and_preprocess(feature_cols, window_size)

    # load checkpoint
    model = CNNLSTMAttentionModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=2,
        dropout=dropout,
        conv1_filters=conv1_filters,
        kernel_size=kernel_size
    ).to(DEVICE)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_targets = []
    all_attention_weights = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds, weights = model(X_batch, return_attention=True)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())
            all_attention_weights.append(weights.cpu().numpy())

    all_attention_weights = np.vstack(all_attention_weights) 
    all_targets = np.array(all_targets)
    time_steps = np.arange(-window_size + 1, 1)

    print(f"Extracted attention map matrix shape: {all_attention_weights.shape}")

    # FIGURE 1: Mean Temporal Attention Profile (1D Line Plot)
    avg_attention = np.mean(all_attention_weights, axis=0)

    plt.figure(figsize=(9, 4.5))
    plt.plot(time_steps, avg_attention, color="#6a0dad", linewidth=2.0, marker="o", markersize=3)
    plt.fill_between(time_steps, avg_attention, color="#6a0dad", alpha=0.15)
    plt.title(f"Mean temporal attention profile - {model_tag}", fontsize=12, fontweight="bold")
    plt.xlabel("Lookback step relative to prediction target (t = 0)", fontsize=10)
    plt.ylabel("Mean attention weight (α)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"attention_mean_profile_{model_tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: attention_mean_profile_{model_tag}.png")

    # FIGURE 2: Representative window - input signal vs attention
    adc_channel_idx = feature_cols.index("adc_norm")
    window_ranges = X_all[:, :, adc_channel_idx].max(axis=1) - X_all[:, :, adc_channel_idx].min(axis=1)
    example_idx = int(np.argmax(window_ranges))
    example_window = X_all[example_idx]
    example_weights = all_attention_weights[example_idx]

    fig, (ax_sig, ax_attn) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ch_idx, ch_name in enumerate(feature_cols):
        ax_sig.plot(time_steps, example_window[:, ch_idx], linewidth=1.4, label=ch_name)
    ax_sig.set_ylabel("Normalised value", fontsize=10)
    ax_sig.set_title(f"Representative test window - input signal vs. attention weight ({model_tag})",
                      fontsize=12, fontweight="bold")
    # position legend outside right
    ax_sig.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=9)
    ax_sig.grid(True, linestyle="--", alpha=0.5)

    ax_attn.plot(time_steps, example_weights, color="#6a0dad", linewidth=1.8)
    ax_attn.fill_between(time_steps, example_weights, color="#6a0dad", alpha=0.15)
    peak_step_here = time_steps[np.argmax(example_weights)]
    ax_attn.axvline(peak_step_here, color="black", linestyle=":", linewidth=1.0,
                     label=f"Peak attention\n(t = {peak_step_here})")
    ax_attn.set_xlabel("Lookback step relative to prediction target (t = 0)", fontsize=10)
    ax_attn.set_ylabel("Attention weight (α)", fontsize=10)
    # position legend outside right
    ax_attn.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=9)
    ax_attn.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"attention_example_window_{model_tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: attention_example_window_{model_tag}.png")

    # FIGURE 3: Sample-wise Temporal Attention Heatmap
    num_display_samples = min(250, len(all_attention_weights))

    plt.figure(figsize=(11, 6))
    im = plt.imshow(
        all_attention_weights[:num_display_samples],
        aspect="auto",
        cmap="magma",
        extent=[-window_size + 1, 0, num_display_samples, 0]
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Attention weight (α)", rotation=270, labelpad=15)
    plt.title(f"Temporal attention heatmap - first {num_display_samples} test sequences ({model_tag})",
              fontsize=12, fontweight="bold")
    plt.xlabel("Lookback step (t - k)", fontsize=10)
    plt.ylabel("Test sample index", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"attention_sample_heatmap_{model_tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: attention_sample_heatmap_{model_tag}.png")

    # FIGURE 4: Synchronised Strain vs. Attention Focus
    s_min = norm_params["strain_ratio_min"]
    s_max = norm_params["strain_ratio_max"]
    targets_physical = all_targets * (s_max - s_min) + s_min

    gaps = np.diff(start_indices)
    run_breaks = np.where(gaps != 1)[0] + 1
    run_bounds = np.concatenate(([0], run_breaks, [len(start_indices)]))
    run_lengths = np.diff(run_bounds)
    longest_run_pos = int(np.argmax(run_lengths))
    run_start = run_bounds[longest_run_pos]
    run_end = min(run_bounds[longest_run_pos + 1], run_start + 300)

    if run_end - run_start < 300:
        print(f"Note: longest contiguous run available is only {run_end - run_start} "
              f"samples (requested up to 300); plotting what's available.")

    sl = slice(run_start, run_end)
    display_targets = targets_physical[sl]
    display_weights = all_attention_weights[sl]
    display_time_s = (start_indices[sl] - start_indices[run_start]) / SAMPLE_RATE_HZ

    peak_attention_step = time_steps[np.argmax(display_weights, axis=1)]
    mean_attention_step = (display_weights * time_steps).sum(axis=1)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    color_strain = "tab:orange"
    ax1.set_xlabel("Elapsed time within contiguous test segment (s)")
    ax1.set_ylabel("Ground truth strain ratio", color=color_strain, fontweight="bold")
    line1, = ax1.plot(display_time_s, display_targets, color=color_strain, linewidth=1.2, label="Strain")
    ax1.tick_params(axis="y", labelcolor=color_strain)

    ax2 = ax1.twinx()
    color_peak = "tab:blue"
    color_mean = "tab:green"
    ax2.set_ylabel("Lookback step (attention focus)", fontweight="bold")
    sc1 = ax2.scatter(display_time_s, peak_attention_step, color=color_peak, alpha=0.5, s=8, label="Peak step (argmax)")
    line2, = ax2.plot(display_time_s, mean_attention_step, color=color_mean, alpha=0.7, linewidth=1.2, label="Weighted mean step")
    ax2.tick_params(axis="y")

    # combine handles and place legend outside to the right
    handles = [line1, sc1, line2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, bbox_to_anchor=(1.08, 1), loc="upper left", borderaxespad=0, fontsize=9)

    plt.title(f"Physical strain dynamics vs. temporal attention focus ({model_tag})",
              fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(f"attention_strain_sync_{model_tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: attention_strain_sync_{model_tag}.png")


if __name__ == "__main__":
    main()