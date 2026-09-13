"""
evaluation.py

Cross-session evaluation of trained LSTM and Causal TCN models
on held-out testing sessions (testing_01, testing_02, testing_03).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

NORM_PARAMS_FILE = 'norm_params.json'
WINDOW_SIZE  = 60    
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model weight files
LSTM_WEIGHTS = 'LSTM_adc_grad_best.pth'
TCN_WEIGHTS = 'Causal_TCN_adc_grad_best.pth'

# Evaluation sessions - add or remove as needed
EVAL_SESSIONS = {
    'testing_01': 'testing_01_clean.csv',
    'testing_02': 'testing_02_clean.csv',
    'testing_03': 'testing_03_clean.csv',
}

print(f"Using device: {DEVICE}")

# params 
with open(NORM_PARAMS_FILE, 'r') as f:
    norm = json.load(f)

ADC_MIN = norm['adc_raw_min']
ADC_MAX= norm['adc_raw_max']
STRAIN_MIN = norm['strain_ratio_min']
STRAIN_MAX = norm['strain_ratio_max']
GRAD_MIN= norm['adc_grad_raw_min']
GRAD_MAX= norm['adc_grad_raw_max']

print(f"Loaded norm params from {NORM_PARAMS_FILE}")
print(f"ADC: [{ADC_MIN:.1f}, {ADC_MAX:.1f}]")
print(f"Strain: [{STRAIN_MIN:.4f}, {STRAIN_MAX:.4f}]")
print(f"Grad: [{GRAD_MIN:.1f}, {GRAD_MAX:.1f}]")

EPS = 1e-8

# pre process
def preprocess_session(csv_path):
    """
    Load a clean CSV session, compute features using session_008
    normalisation parameters, and return arrays ready for sequence
    construction.
    """
    df = pd.read_csv(csv_path)
    print(f"\n  Raw rows: {len(df)}")

    # drop rows missing either signal
    df = df.dropna(subset=['adc_raw', 'strain_ratio'])
    print(f" After dropna: {len(df)}")

    # apply same filters as training cleaning pipeline
    df = df[(df['adc_raw'] > 10) & (df['adc_raw'] < 4085)]
    df = df[(df['strain_ratio'] >= 0.5) & (df['strain_ratio'] <= 2.5)]
    print(f" After plausibility filter: {len(df)}")

    df = df.reset_index(drop=True)

    # compute gradient using same method as training (np.gradient)
    adc_raw  = df['adc_raw'].values.astype(np.float32)
    strain_raw = df['strain_ratio'].values.astype(np.float32)
    adc_grad_raw   = np.gradient(adc_raw).astype(np.float32)

    # normalise using session_008 training bounds - do NOT recompute
    adc_norm = (adc_raw - ADC_MIN) / (ADC_MAX - ADC_MIN + EPS)
    adc_grad_norm  = (adc_grad_raw - GRAD_MIN) / (GRAD_MAX - GRAD_MIN   + EPS)
    strain_norm   = (strain_raw - STRAIN_MIN) / (STRAIN_MAX - STRAIN_MIN + EPS)

    # sequence break flags 
    breaks = np.zeros(len(df), dtype=bool)
    if 'sequence_break' in df.columns:
        breaks = df['sequence_break'].values.astype(bool)
    else:
        # recompute from timestamps if column missing
        timestamps = df['timestamp_ms'].values
        gaps = np.diff(timestamps, prepend=timestamps[0])
        breaks = gaps > 200
        print(f" Recomputed sequence_break from timestamps")

    n_breaks = breaks.sum()
    print(f"Sequence breaks: {n_breaks} -> ~{n_breaks+1} segments")

    return adc_norm, adc_grad_norm, strain_norm, breaks, strain_raw


def make_sequences(adc_norm, adc_grad_norm, strain_norm, breaks, window_size):
    """
    Build sliding window sequences that do not cross sequence breaks.
    Target is strain at the final timestep of each window.
    """
    X, y = [], []
    n = len(adc_norm)
    for i in range(n - window_size):
        if breaks[i+1 : i+window_size+1].any():
            continue
        seq = np.stack([
            adc_norm[i : i+window_size],
            adc_grad_norm[i : i+window_size]
        ], axis=-1) 
        X.append(seq)
        y.append(strain_norm[i + window_size - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# dataset 
class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(-1) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# define models 
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.parametrize.register_parametrization if False else \
                     nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
                          if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CausalTCN(nn.Module):
    def __init__(self, input_size=2, num_channels=None,
                 kernel_size=5, dropout=0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 128]
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation  = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        stride=1, dilation=dilation,
                                        padding=padding, dropout=dropout))
        self.tcn = nn.ModuleList(layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.tcn:
            x = layer(x)
        # take last timestep
        out = x[:, :, -1]
        return self.fc(out)

# now eval 
def evaluate(model, loader, strain_min, strain_max):
    model.eval()
    preds_norm, targets_norm = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            preds_norm.extend(pred.cpu().numpy().flatten())
            targets_norm.extend(y_batch.numpy().flatten())

    preds_norm = np.array(preds_norm)
    targets_norm = np.array(targets_norm)

    # denormalise
    preds = preds_norm * (strain_max - strain_min) + strain_min
    targets = targets_norm * (strain_max - strain_min) + strain_min

    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    return preds, targets, mae, rmse, r2


# plot 
def plot_predictions(preds, targets, model_name, session_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    n = min(500, len(preds))
    ax1.plot(targets[:n], color='orange', linewidth=0.8, label='Ground truth')
    ax1.plot(preds[:n], color='steelblue', linewidth=0.8,
             alpha=0.8, label='Predicted')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Strain Ratio')
    ax1.set_title(f'{model_name}, {session_name} - predictions vs ground truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(targets, preds, alpha=0.05, s=1, color='purple')
    lims = [min(targets.min(), preds.min()),
            max(targets.max(), preds.max())]
    ax2.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
    ax2.set_xlabel('Ground truth strain ratio')
    ax2.set_ylabel('Predicted strain ratio')
    ax2.set_title(f'{model_name}, {session_name} - scatter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'crosssession_{model_name}_{session_name}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Plot saved: {fname}")


# load 
print(f"\nLoading models")

lstm_model = LSTMModel(input_size=2, hidden_size=HIDDEN_SIZE,
                       num_layers=NUM_LAYERS, dropout=DROPOUT)
lstm_model.load_state_dict(torch.load(LSTM_WEIGHTS,
                           map_location=DEVICE))
lstm_model = lstm_model.to(DEVICE)
lstm_model.eval()
print(f"  LSTM loaded from {LSTM_WEIGHTS}")

tcn_model = CausalTCN(input_size=2, num_channels=[64, 128, 128],
                      kernel_size=5, dropout=0.2)
tcn_model.load_state_dict(torch.load(TCN_WEIGHTS,
                          map_location=DEVICE))
tcn_model = tcn_model.to(DEVICE)
tcn_model.eval()
print(f"  TCN loaded from {TCN_WEIGHTS}")


# main loop
all_results = []

for session_name, csv_file in EVAL_SESSIONS.items():
    print(f"\n{'='*60}")
    print(f"Evaluating on: {session_name} ({csv_file})")
    print(f"{'='*60}")

    if not os.path.exists(csv_file):
        print(f"  WARNING: {csv_file} not found - skipping")
        continue

    # preprocess
    adc_norm, adc_grad_norm, strain_norm, breaks, strain_raw = \
        preprocess_session(csv_file)

    # build sequences
    X, y = make_sequences(adc_norm, adc_grad_norm,
                          strain_norm, breaks, WINDOW_SIZE)
    print(f"  Sequences: {len(X)}")

    if len(X) == 0:
        print(f"  WARNING: no valid sequences - skipping")
        continue

    loader = DataLoader(SensorDataset(X, y),
                        batch_size=BATCH_SIZE, shuffle=False)

    # evaluate LSTM
    lstm_preds, lstm_targets, lstm_mae, lstm_rmse, lstm_r2 = \
        evaluate(lstm_model, loader, STRAIN_MIN, STRAIN_MAX)
    print(f"\n  LSTM results:")
    print(f" MAE:  {lstm_mae:.4f}")
    print(f" RMSE: {lstm_rmse:.4f}")
    print(f" R2:   {lstm_r2:.4f}")
    plot_predictions(lstm_preds, lstm_targets, 'LSTM', session_name)

    # evaluate TCN
    tcn_preds, tcn_targets, tcn_mae, tcn_rmse, tcn_r2 = \
        evaluate(tcn_model, loader, STRAIN_MIN, STRAIN_MAX)
    print(f"\n  Causal TCN results:")
    print(f"    MAE:  {tcn_mae:.4f}")
    print(f"    RMSE: {tcn_rmse:.4f}")
    print(f"    R2:   {tcn_r2:.4f}")
    plot_predictions(tcn_preds, tcn_targets, 'Causal_TCN', session_name)

    all_results.append({
        'session':   session_name,
        'model':     'LSTM',
        'MAE':       lstm_mae,
        'RMSE':      lstm_rmse,
        'R2':        lstm_r2,
        'n_seqs':    len(X)
    })
    all_results.append({
        'session':   session_name,
        'model':     'Causal_TCN',
        'MAE':       tcn_mae,
        'RMSE':      tcn_rmse,
        'R2':        tcn_r2,
        'n_seqs':    len(X)
    })


# summary 
print(f"\n{'='*70}")
print("CROSS-SESSION EVALUATION SUMMARY")
print(f"{'='*70}")
print(f"{'Session':<15} {'Model':<15} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'N seqs':>8}")
print("-" * 70)

results_df = pd.DataFrame(all_results)
for _, row in results_df.iterrows():
    print(f"{row['session']:<15} {row['model']:<15} "
          f"{row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
          f"{row['R2']:>8.4f} {int(row['n_seqs']):>8}")

# per-model summary statistics across sessions
print(f"\n{'='*70}")
print("PER-MODEL SUMMARY ACROSS ALL SESSIONS")
print(f"{'='*70}")
for model_name in ['LSTM', 'Causal_TCN']:
    subset = results_df[results_df['model'] == model_name]
    print(f"\n{model_name}:")
    print(f"  MAE  - mean: {subset['MAE'].mean():.4f}  "
          f"std: {subset['MAE'].std():.4f}  "
          f"range: [{subset['MAE'].min():.4f}, {subset['MAE'].max():.4f}]")
    print(f"  RMSE - mean: {subset['RMSE'].mean():.4f}  "
          f"std: {subset['RMSE'].std():.4f}  "
          f"range: [{subset['RMSE'].min():.4f}, {subset['RMSE'].max():.4f}]")
    print(f"  R2   - mean: {subset['R2'].mean():.4f}  "
          f"std: {subset['R2'].std():.4f}  "
          f"range: [{subset['R2'].min():.4f}, {subset['R2'].max():.4f}]")

# save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv   = f'cross_session_results_{timestamp}.csv'
results_df.to_csv(out_csv, index=False)
print(f"\nResults saved to: {out_csv}")
print(f"Completed at: {datetime.now().strftime('%H:%M:%S')}")