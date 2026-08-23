"""
train.py - Real-Time Benchmark: Champion LSTM vs. Causal TCN
Feature Baseline: adc_grad ('adc_norm', 'adc_grad_norm')
"""

from datetime import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# config 
DATA_FILE = "/scratch0/adrikhan/sensor_project/session_008_clean.csv"
EPOCHS = 150
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15  # test split is remaining 0.15
EARLY_STOP_PATIENCE = 40
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")
print(f"Loading data from: {DATA_FILE}")

# data prep 
df = pd.read_csv(DATA_FILE)

# raw grad 
adc_raw = df["adc_raw"].values.astype(np.float32)
df["adc_grad_raw"] = np.gradient(adc_raw).astype(np.float32)

# min max on train only 
n = len(df)
n_train = int(n * TRAIN_SPLIT)
df_train_raw = df.iloc[:n_train]

raw_cols_to_normalize = ["adc_raw", "strain_ratio", "adc_grad_raw"]

norm_params = {}
for col in raw_cols_to_normalize:
    col_min, col_max = df_train_raw[col].min(), df_train_raw[col].max()
    norm_params[f"{col}_min"] = float(col_min)
    norm_params[f"{col}_max"] = float(col_max)
    
    out_col = col.replace("_raw", "") + "_norm"
    if col == "strain_ratio":
        out_col = "strain_norm"
    df[out_col] = (df[col] - col_min) / (col_max - col_min + 1e-8)

with open("norm_params.json", "w") as f:
    json.dump(norm_params, f, indent=2)
print("Saved non-leaked normalisation parameters to norm_params.json")

# feature set params 
FEATURE_COLS = ["adc_norm", "adc_grad_norm"]
TARGET_COL = "strain_norm"
BREAK_COL = "sequence_break"
N_TRIALS = 100
strain_min = norm_params["strain_ratio_min"]
strain_max = norm_params["strain_ratio_max"]


# define models 
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# causal tcn components 
class Chomp1d(nn.Module):
    """Trims padding at the right boundary to enforce causality (no future information leakage)."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CausalTCNModel(nn.Module):
    def __init__(self, input_size=2, num_channels=[32, 64, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # exponentially increasing dilation (1, 2, 4, 8...)
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation_size, padding=padding, dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # input shape: (batch_size, sequence_length, channels) -> Permute for 1D Conv
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = self.fc(out[:, :, -1])  # take the last time step prediction
        return out


# pipeline and training helpers 
def create_sequences_from_raw(raw_data, feature_cols, target_col, break_col, window_size):
    features = raw_data[feature_cols].values
    targets = raw_data[target_col].values
    breaks = raw_data[break_col].values.astype(bool)

    X, y = [], []
    n = len(raw_data)
    for i in range(n - window_size):
        if breaks[i + 1 : i + window_size + 1].any():
            continue
        X.append(features[i : i + window_size])
        y.append([targets[i + window_size - 1]])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_dataloaders(df, feature_cols, target_col, break_col, window_size, batch_size, train_split=0.7, val_split=0.15):
    n = len(df)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train : n_train + n_val]
    df_test = df.iloc[n_train + n_val :]

    X_train, y_train = create_sequences_from_raw(df_train, feature_cols, target_col, break_col, window_size)
    X_val, y_val = create_sequences_from_raw(df_val, feature_cols, target_col, break_col, window_size)
    X_test, y_test = create_sequences_from_raw(df_test, feature_cols, target_col, break_col, window_size)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, (X_test, y_test)


def train_model(model, train_loader, val_loader, epochs, lr, model_name, weight_decay=1e-4, verbose=True):
    if verbose:
        print("\n" + "=" * 80)
        print(f"Training: {model_name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("=" * 80)

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimiser.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_batch_losses.append(loss.item())
        val_loss = np.mean(val_batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | Best val: {best_val_loss:.6f}")

        if patience_count >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, model_name, s_min, s_max):
    model.eval()
    preds_norm, targets_norm = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            preds_norm.extend(pred.cpu().numpy().flatten())
            targets_norm.extend(y_batch.numpy().flatten())

    preds_norm = np.array(preds_norm)
    targets_norm = np.array(targets_norm)

    preds = preds_norm * (s_max - s_min) + s_min
    targets = targets_norm * (s_max - s_min) + s_min

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n{model_name} TEST RESULTS:")
    print(f"MAE: {mae:.4f} strain ratio units | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    return preds, targets, mae, rmse, r2


def plot_losses(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - training curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_name}_loss_curve.png", dpi=150)
    plt.close()


def plot_predictions(preds, targets, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    n = min(500, len(preds))
    ax1.plot(targets[:n], label="Ground truth", color="orange", linewidth=1.0)
    ax1.plot(preds[:n], label="Predicted", color="blue", linewidth=1.0, alpha=0.8)
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Strain Ratio")
    ax1.set_title(f"{model_name} - predictions vs ground truth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(targets, preds, alpha=0.1, s=1, color="purple")
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax2.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax2.set_xlabel("Ground truth strain ratio")
    ax2.set_ylabel("Predicted strain ratio")
    ax2.set_title(f"{model_name} - scatter plot")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{model_name}_predictions.png", dpi=150)
    plt.close()


def objective(trial, model_type, feature_cols, target_col, break_col, df_clean):
    lr = trial.suggest_float(f"{model_type}_lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float(f"{model_type}_weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float(f"{model_type}_dropout", 0.2, 0.5, step=0.05)
    batch_size = trial.suggest_categorical(f"{model_type}_batch_size", [16, 32, 64])
    window_size = trial.suggest_categorical(f"{model_type}_window_size", [30, 60, 120, 240, 480])

    train_loader, val_loader, _, _ = prepare_dataloaders(
        df=df_clean, feature_cols=feature_cols, target_col=target_col, break_col=break_col,
        window_size=window_size, batch_size=batch_size, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT
    )

    input_dim = len(feature_cols)

    if model_type == "LSTM":
        hidden_size = trial.suggest_categorical(f"{model_type}_hidden_size", [64, 128])
        model = LSTMModel(input_size=input_dim, hidden_size=hidden_size, dropout=dropout).to(DEVICE)

    elif model_type == "Causal_TCN":
        kernel_size = trial.suggest_categorical(f"{model_type}_kernel_size", [3, 5])
        channels_str = trial.suggest_categorical(f"{model_type}_channels", ["32_64_64", "64_128_128"])
        num_channels = [int(c) for c in channels_str.split("_")]
        model = CausalTCNModel(input_size=input_dim, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(DEVICE)

    model, _, val_losses = train_model(
        model, train_loader, val_loader, epochs=15, lr=lr, weight_decay=weight_decay,
        model_name=f"Trial_{trial.number}", verbose=False
    )
    return min(val_losses)


# main pipeline
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    models_to_run = ["LSTM", "Causal_TCN"]
    feat_name = "adc_grad"

    print("\n" + "=" * 80)
    print(f"BENCHMARKING ON FEATURE SET: {feat_name}")
    print("=" * 80)

    best_hparams = {}
    results = {}

    # optuna loop for optimisation 
    for model_name in models_to_run:
        trial_csv = f"optuna_trials_{model_name}_{feat_name}.csv"

        if os.path.exists(trial_csv):
            print(f"\nSkipping Optuna for {model_name} - loading best params from {trial_csv}.")
            trials_df = pd.read_csv(trial_csv)
            best_trial = trials_df.loc[trials_df["value"].idxmin()]
            param_cols = [c for c in trials_df.columns if c.startswith(f"params_{model_name}_")]
            clean_params = {c.replace(f"params_{model_name}_", ""): best_trial[c] for c in param_cols}

            for int_param in ["hidden_size", "batch_size", "window_size", "kernel_size"]:
                if int_param in clean_params:
                    clean_params[int_param] = int(clean_params[int_param])
            best_hparams[model_name] = clean_params

        else:
            print(f"\nStarting Optuna Hyperparameter Optimisation for {model_name}...", flush=True)
            study = optuna.create_study(
                study_name=f"study_{model_name}_{feat_name}",
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
            )
            study.optimize(
                lambda trial: objective(trial, model_name, FEATURE_COLS, TARGET_COL, BREAK_COL, df),
                n_trials=N_TRIALS,
            )
            clean_params = {k.replace(f"{model_name}_", ""): v for k, v in study.best_params.items()}
            best_hparams[model_name] = clean_params
            study.trials_dataframe().to_csv(trial_csv, index=False)

    # retraining and eval loop 
    for model_name in models_to_run:
        params = best_hparams[model_name]
        train_loader, val_loader, test_loader, _ = prepare_dataloaders(
            df=df, feature_cols=FEATURE_COLS, target_col=TARGET_COL, break_col=BREAK_COL,
            window_size=params["window_size"], batch_size=params["batch_size"],
            train_split=TRAIN_SPLIT, val_split=VAL_SPLIT
        )

        if model_name == "LSTM":
            final_model = LSTMModel(len(FEATURE_COLS), params["hidden_size"], dropout=params["dropout"])
        elif model_name == "Causal_TCN":
            num_channels = [int(c) for c in params["channels"].split("_")]
            final_model = CausalTCNModel(len(FEATURE_COLS), num_channels=num_channels, kernel_size=params["kernel_size"], dropout=params["dropout"])

        weights_file = f"{model_name}_{feat_name}_best.pth"

        if os.path.exists(weights_file):
            print(f"\nFound existing weights file '{weights_file}'. Skipping {EPOCHS}-epoch retraining.")
            final_model.load_state_dict(torch.load(weights_file, map_location=DEVICE))
            final_model = final_model.to(DEVICE)
        else:
            final_model, train_losses, val_losses = train_model(
                final_model, train_loader, val_loader, epochs=EPOCHS, lr=params["lr"],
                weight_decay=params["weight_decay"], model_name=f"{model_name}_{feat_name}_Benchmark"
            )
            torch.save(final_model.state_dict(), weights_file)
            print(f"Saved best model weights to: {weights_file}")
            plot_losses(train_losses, val_losses, f"{model_name}_{feat_name}_Benchmark")

        preds, targets, mae, rmse, r2 = evaluate_model(final_model, test_loader, f"{model_name}_{feat_name}_Benchmark", strain_min, strain_max)
        plot_predictions(preds, targets, f"{model_name}_{feat_name}_Benchmark")
        results[model_name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    # save results and metadata summary 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results_tcn_vs_lstm_{timestamp}.json", "w") as f:
        json.dump({"metrics": results, "hyperparameters": best_hparams}, f, indent=2)

    print("\n==================================================")
    print("REAL-TIME BENCHMARK SUMMARY (adc_grad BASELINE)")
    print(f"{'Model Architecture':<25} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    print("\n==================================================")
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f} {metrics['R2']:>10.4f}")
    print("\n==================================================")
    print("Benchmark Complete.")