"""
train.py (and tuning)

Trains LSTM, CNN-LSTM, and CNN-LSTM-Attention models with Bayesian Optimization (Optuna)
Input: adc_raw (normalised) + gradient (normalised) + velocity/diff (normalised)
Output: strain_ratio (normalised)
"""

from datetime import datetime
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

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

# prep the data
df = pd.read_csv(DATA_FILE)

# compute raw gradient and 1st order difference before splitting
adc_raw = df["adc_raw"].values.astype(np.float32)
df["adc_grad_raw"] = np.gradient(adc_raw).astype(np.float32)
df["adc_diff_raw"] = np.diff(adc_raw, prepend=adc_raw[0]).astype(np.float32)

# PREVENT DATA LEAKAGE: CALCULATE MIN/MAX ONLY ON TRAIN SPLIT!!!!!!
n = len(df)
n_train = int(n * TRAIN_SPLIT)
df_train_raw = df.iloc[:n_train]

adc_min, adc_max = df_train_raw["adc_raw"].min(), df_train_raw["adc_raw"].max()
strain_min, strain_max = df_train_raw["strain_ratio"].min(), df_train_raw["strain_ratio"].max()
grad_min, grad_max = df_train_raw["adc_grad_raw"].min(), df_train_raw["adc_grad_raw"].max()
diff_min, diff_max = df_train_raw["adc_diff_raw"].min(), df_train_raw["adc_diff_raw"].max()

# apply normalisation across full dataframe using TRAIN statistics
df["adc_norm"] = (df["adc_raw"] - adc_min) / (adc_max - adc_min + 1e-8)
df["strain_norm"] = (df["strain_ratio"] - strain_min) / (strain_max - strain_min + 1e-8)
df["adc_grad_norm"] = (df["adc_grad_raw"] - grad_min) / (grad_max - grad_min + 1e-8)
df["adc_diff_norm"] = (df["adc_diff_raw"] - diff_min) / (diff_max - diff_min + 1e-8)

# save normalisation parameters (derived strictly from training set)
norm_params = {
    "adc_min": float(adc_min),
    "adc_max": float(adc_max),
    "strain_min": float(strain_min),
    "strain_max": float(strain_max),
    "grad_min": float(grad_min),
    "grad_max": float(grad_max),
    "diff_min": float(diff_min),
    "diff_max": float(diff_max),
}

with open("norm_params.json", "w") as f:
    json.dump(norm_params, f, indent=2)
print("Saved non-leaked normalisation parameters to norm_params.json")

FEATURE_COLS = ["adc_norm", "adc_grad_norm", "adc_diff_norm"]
FEATURE_SETS = {
    "adc_only": ["adc_norm"],
    "adc_grad": ["adc_norm", "adc_grad_norm"],
    "adc_diff": ["adc_norm", "adc_diff_norm"],
    "adc_grad_diff": ["adc_norm", "adc_grad_norm", "adc_diff_norm"],
}
TARGET_COL = "strain_norm"
BREAK_COL = "sequence_break"
N_TRIALS = 100


# model definitions
class LSTMModel(nn.Module):

    def __init__(
        self, input_size=3, hidden_size=64, num_layers=2, dropout=0.3
    ):
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


class CNNLSTMModel(nn.Module):

    def __init__(
        self,
        input_size=3,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        conv1_filters=32,
        kernel_size=5,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=conv1_filters,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=conv1_filters,
                out_channels=conv1_filters * 2,
                kernel_size=3,
                padding=1,
            ),
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
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


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

    def __init__(
        self,
        input_size=3,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        conv1_filters=32,
        kernel_size=5,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=conv1_filters,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=conv1_filters,
                out_channels=conv1_filters * 2,
                kernel_size=3,
                padding=1,
            ),
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


# sequence generation and data pipeline
def create_sequences_from_raw(
    raw_data, feature_cols, target_col, break_col, window_size
):
    """Generates temporal sequence windows, taking sequence breaks into account."""
    features = raw_data[feature_cols].values
    targets = raw_data[target_col].values
    breaks = raw_data[break_col].values.astype(bool)

    X, y = [], []
    n = len(raw_data)
    for i in range(n - window_size):
        # skip sequence if it crosses a break boundary
        if breaks[i + 1 : i + window_size + 1].any():
            continue
        X.append(features[i : i + window_size])
        y.append([targets[i + window_size - 1]])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_dataloaders(
    df,
    feature_cols,
    target_col,
    break_col,
    window_size,
    batch_size,
    train_split=0.7,
    val_split=0.15,
):
    n = len(df)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train : n_train + n_val]
    df_test = df.iloc[n_train + n_val :]

    X_train, y_train = create_sequences_from_raw(
        df_train, feature_cols, target_col, break_col, window_size
    )
    X_val, y_val = create_sequences_from_raw(
        df_val, feature_cols, target_col, break_col, window_size
    )
    X_test, y_test = create_sequences_from_raw(
        df_test, feature_cols, target_col, break_col, window_size
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, (X_test, y_test)


# training and eval functions
def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    model_name,
    weight_decay=1e-4,
    verbose=True,
):
    if verbose:
        print(
            "\n==============================================================================="
        )
        print(f"Training: {model_name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            "==============================================================================="
        )

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )

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
            print(
                f"Epoch {epoch:3d}/{epochs} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | Best val: {best_val_loss:.6f}"
            )

        if patience_count >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, model_name, strain_min, strain_max):
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

    preds = preds_norm * (strain_max - strain_min) + strain_min
    targets = targets_norm * (strain_max - strain_min) + strain_min

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n{model_name} TEST RESULTS:")
    print(f"MAE: {mae:.4f} strain ratio units")
    print(f"RMSE: {rmse:.4f} strain ratio units")
    print(f"R2: {r2:.4f}")

    return preds, targets, mae, rmse, r2


# plotting functions and helpers
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
    ax1.plot(
        preds[:n], label="Predicted", color="blue", linewidth=1.0, alpha=0.8
    )
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


def plot_attention(model, X_test, n_examples=3):
    model.eval()
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4 * n_examples))
    if n_examples == 1:
        axes = [axes]

    indices = np.random.choice(len(X_test), n_examples, replace=False)

    for i, idx in enumerate(indices):
        x = (
            torch.tensor(X_test[idx], dtype=torch.float32)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            _, weights = model(x, return_attention=True)

        weights = weights.cpu().numpy().flatten()
        adc_seq = X_test[idx][:, 0]

        ax = axes[i]
        ax2 = ax.twinx()
        ax.bar(
            range(len(weights)),
            weights,
            alpha=0.4,
            color="purple",
            label="Attention weight",
        )
        ax2.plot(adc_seq, color="steelblue", linewidth=1.5, label="ADC signal")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Attention weight", color="purple")
        ax2.set_ylabel("ADC (normalised)", color="steelblue")
        ax.set_title(f"Attention map - example {i+1}")

    plt.tight_layout()
    plt.savefig("CNN_LSTM_Attention_attention_maps.png", dpi=150)
    plt.close()


# optuna optimisation objective
def objective(trial, model_type, feature_cols, target_col, break_col, df_clean):
    lr = trial.suggest_float(f"{model_type}_lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float(
        f"{model_type}_weight_decay", 1e-5, 1e-2, log=True
    )
    hidden_size = trial.suggest_categorical(
        f"{model_type}_hidden_size", [64, 128]
    )
    dropout = trial.suggest_float(f"{model_type}_dropout", 0.2, 0.5, step=0.05)
    batch_size = trial.suggest_categorical(
        f"{model_type}_batch_size", [16, 32, 64]
    )
    window_size = trial.suggest_categorical(
        f"{model_type}_window_size", [30, 60, 120, 240, 480]
    )

    train_loader, val_loader, _, _ = prepare_dataloaders(
        df=df_clean,
        feature_cols=feature_cols,
        target_col=target_col,
        break_col=break_col,
        window_size=window_size,
        batch_size=batch_size,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
    )

    input_dim = len(feature_cols)

    if model_type == "LSTM":
        model = LSTMModel(
            input_size=input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        ).to(DEVICE)

    elif model_type == "CNN_LSTM":
        conv1_filters = trial.suggest_categorical(
            f"{model_type}_conv1_filters", [32, 64]
        )
        kernel_size = trial.suggest_categorical(
            f"{model_type}_kernel_size", [3, 5, 7]
        )
        model = CNNLSTMModel(
            input_size=input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            conv1_filters=conv1_filters,
            kernel_size=kernel_size,
        ).to(DEVICE)

    elif model_type == "CNN_LSTM_Attention":
        conv1_filters = trial.suggest_categorical(
            f"{model_type}_conv1_filters", [32, 64]
        )
        kernel_size = trial.suggest_categorical(
            f"{model_type}_kernel_size", [3, 5, 7]
        )
        model = CNNLSTMAttentionModel(
            input_size=input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            conv1_filters=conv1_filters,
            kernel_size=kernel_size,
        ).to(DEVICE)

    model, _, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs=15,
        lr=lr,
        weight_decay=weight_decay,
        model_name=f"Trial_{trial.number}",
        verbose=False,
    )

    return min(val_losses)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    models_to_run = ["LSTM", "CNN_LSTM", "CNN_LSTM_Attention"]
    all_results = {}
    all_best_hparams = {}

    for feat_name, feat_cols in FEATURE_SETS.items():

        print(f"\n==============================")
        print(f"FEATURE SET: {feat_name} -> {feat_cols}")
        print(f"==============================")

        best_hyperparams = {}
        results = {}

        # STEP 1: optuna optimisation (or recover best params from CSV)
        for model_name in models_to_run:
            trial_csv = f"optuna_trials_{model_name}_{feat_name}.csv"

            if os.path.exists(trial_csv):
                print(
                    f"Skipping Optuna for {model_name} on {feat_name} - loading best params from {trial_csv}."
                )

                trials_df = pd.read_csv(trial_csv)
                best_trial = trials_df.loc[trials_df["value"].idxmin()]

                param_cols = [
                    c
                    for c in trials_df.columns
                    if c.startswith(f"params_{model_name}_")
                ]
                clean_params = {
                    c.replace(f"params_{model_name}_", ""): best_trial[c]
                    for c in param_cols
                }

                for int_param in [
                    "hidden_size",
                    "batch_size",
                    "window_size",
                    "conv1_filters",
                    "kernel_size",
                ]:
                    if int_param in clean_params:
                        clean_params[int_param] = int(clean_params[int_param])

                best_hyperparams[model_name] = clean_params

            else:
                print(f"\nStarting Optimisation for {model_name}", flush=True)

                study = optuna.create_study(
                    study_name=f"study_{model_name}_{feat_name}",
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=SEED),
                )

                def trial_callback(study, trial):
                    best_val = (
                        f"{study.best_value:.6f}"
                        if len(study.trials) > 0
                        else "N/A"
                    )
                    trial_val = (
                        f"{trial.value:.6f}"
                        if trial.value is not None
                        else "N/A"
                    )
                    clean_p = {
                        k.replace(f"{model_name}_", ""): v
                        for k, v in trial.params.items()
                    }
                    print(
                        f"  Trial {len(study.trials)}/{N_TRIALS} | Val loss: {trial_val} | Best: {best_val} | Params: {clean_p}"
                    )

                study.optimize(
                    lambda trial: objective(
                        trial,
                        model_type=model_name,
                        feature_cols=feat_cols,
                        target_col=TARGET_COL,
                        break_col=BREAK_COL,
                        df_clean=df,
                    ),
                    n_trials=N_TRIALS,
                    callbacks=[trial_callback],
                )

                clean_params = {
                    k.replace(f"{model_name}_", ""): v
                    for k, v in study.best_params.items()
                }
                best_hyperparams[model_name] = clean_params

                study.trials_dataframe().to_csv(trial_csv, index=False)

        # STEP 2: final retraining, testing, and saving weights
        print("\nRETRAINING FINAL MODELS WITH OPTIMAL HYPERPARAMETERS")

        for model_name in models_to_run:
            params = best_hyperparams[model_name]

            batch_size = params["batch_size"]
            lr = params["lr"]
            weight_decay = params["weight_decay"]
            window_size = params["window_size"]

            train_loader, val_loader, test_loader, (X_test, _) = (
                prepare_dataloaders(
                    df=df,
                    feature_cols=feat_cols,
                    target_col=TARGET_COL,
                    break_col=BREAK_COL,
                    window_size=window_size,
                    batch_size=batch_size,
                    train_split=TRAIN_SPLIT,
                    val_split=VAL_SPLIT,
                )
            )

            if model_name == "LSTM":
                final_model = LSTMModel(
                    input_size=len(feat_cols),
                    hidden_size=params["hidden_size"],
                    dropout=params["dropout"],
                )
            elif model_name == "CNN_LSTM":
                final_model = CNNLSTMModel(
                    input_size=len(feat_cols),
                    hidden_size=params["hidden_size"],
                    dropout=params["dropout"],
                    conv1_filters=params["conv1_filters"],
                    kernel_size=params["kernel_size"],
                )
            elif model_name == "CNN_LSTM_Attention":
                final_model = CNNLSTMAttentionModel(
                    input_size=len(feat_cols),
                    hidden_size=params["hidden_size"],
                    dropout=params["dropout"],
                    conv1_filters=params["conv1_filters"],
                    kernel_size=params["kernel_size"],
                )

            final_model, train_losses, val_losses = train_model(
                final_model,
                train_loader,
                val_loader,
                epochs=EPOCHS,
                lr=lr,
                weight_decay=weight_decay,
                model_name=f"{model_name}_{feat_name}_Optimized",
            )

            # SAVE WEIGHTS!!
            weights_file = f"{model_name}_{feat_name}_best.pth"
            torch.save(final_model.state_dict(), weights_file)
            print(f"Saved model weights to: {weights_file}")

            plot_losses(
                train_losses, val_losses, f"{model_name}_{feat_name}_Optimized"
            )

            preds, targets, mae, rmse, r2 = evaluate_model(
                final_model,
                test_loader,
                f"{model_name}_{feat_name}_Optimized",
                strain_min,
                strain_max,
            )
            plot_predictions(
                preds, targets, f"{model_name}_{feat_name}_Optimized"
            )

            results[model_name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

            if model_name == "CNN_LSTM_Attention":
                plot_attention(final_model, X_test, n_examples=3)

        all_results[feat_name] = results
        all_best_hparams[feat_name] = best_hyperparams

    # STEP 3: save final experiment results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_optimized_feature_ablation_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "metrics": {
                    feat: {
                        model: {m: float(v) for m, v in metrics.items()}
                        for model, metrics in models.items()
                    }
                    for feat, models in all_results.items()
                },
                "best_hyperparameters": all_best_hparams,
            },
            f,
            indent=2,
        )

    print("\nFINAL BAYESIAN OPTIMISED RESULTS (FEATURE ABLATION)")
    print(
        f"{'FeatureSet':<15} {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}"
    )
    print(
        "===================================================================================="
    )
    for feat_name, models in all_results.items():
        for model_name, metrics in models.items():
            print(
                f"{feat_name:<15} {model_name:<25} {metrics['MAE']:>8.4f} {metrics['RMSE']:>8.4f} {metrics['R2']:>8.4f}"
            )