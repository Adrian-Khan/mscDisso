"""
train.py

Trains LSTM, CNN-LSTM, and CNN-LSTM-Attention models with Bayesian Optimization (Optuna)
Input: adc_raw (normalised) + gradient (normalised)
Output: strain_ratio (normalised)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna

# config 
#DATA_FILE = "outputs\data_collection_outputs\session_008_clean.csv" # windows
DATA_FILE = "/scratch0/adrikhan/sensor_project/session_008_clean.csv" # linux
SEQ_LEN = 120    # timesteps per sequence
EPOCHS = 150     # final evaluation epochs
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15  # test split is remainder: 0.15
EARLY_STOP_PATIENCE = 40
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")
print(f"Loading data from: {DATA_FILE}")

# load and process data 
df = pd.read_csv(DATA_FILE)

adc = df['adc_raw'].values.astype(np.float32)
strain = df['strain_ratio'].values.astype(np.float32)
breaks = df['sequence_break'].values.astype(bool)

# normalise signals
adc_min, adc_max = adc.min(), adc.max()
strain_min, strain_max = strain.min(), strain.max()

adc_norm = (adc - adc_min) / (adc_max - adc_min)
strain_norm = (strain - strain_min) / (strain_max - strain_min)

# compute gradient (rate of change)
adc_grad = np.gradient(adc_norm).astype(np.float32)
grad_min, grad_max = adc_grad.min(), adc_grad.max()
adc_grad_norm = (adc_grad - grad_min) / (grad_max - grad_min)

# save normalization parameters
norm_params = {
    'adc_min': float(adc_min),
    'adc_max': float(adc_max),
    'strain_min': float(strain_min),
    'strain_max': float(strain_max),
    'grad_min': float(grad_min),
    'grad_max': float(grad_max)
}

with open('norm_params.json', 'w') as f:
    json.dump(norm_params, f, indent=2)
print("Saved normalisation parameters to norm_params.json")


def make_sequences(adc_norm, adc_grad_norm, strain_norm, breaks, seq_len):
    X, y = [], []
    n = len(adc_norm)
    for i in range(n - seq_len):
        if breaks[i+1 : i+seq_len+1].any():
            continue
        seq = np.stack([adc_norm[i : i+seq_len], adc_grad_norm[i : i+seq_len]], axis=-1)
        X.append(seq)
        y.append(strain_norm[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_sequences(adc_norm, adc_grad_norm, strain_norm, breaks, SEQ_LEN)

print(f"Total sequences: {len(X)}")
print(f"Input shape: {X.shape} Output shape: {y.shape}")

# chronological train / val / test split
n = len(X)
n_train = int(n * TRAIN_SPLIT)
n_val = int(n * VAL_SPLIT)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

print(f"\nTrain: {len(X_train)} sequences")
print(f"Validation: {len(X_val)} sequences")
print(f"Test: {len(X_test)} sequences")


class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# dynamic model definitions
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3):
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


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3, conv1_filters=32, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=conv1_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv1_filters, out_channels=conv1_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=conv1_filters * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
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
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        score = torch.tanh(self.attn(lstm_out))
        weight = torch.softmax(self.context(score), dim=1)
        out = (weight * lstm_out).sum(dim=1)
        return out, weight.squeeze(-1)


class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3, conv1_filters=32, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=conv1_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv1_filters, out_channels=conv1_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=conv1_filters * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
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

# train and eval funcs
def train_model(model, train_loader, val_loader, epochs, lr, model_name, verbose=True):
    if verbose:
        print("\n===============================================================================")
        print(f"Training: {model_name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("===============================================================================")

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
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

# plotting helpers 
def plot_losses(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{model_name} - training curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name}_loss_curve.png', dpi=150)
    plt.close()


def plot_predictions(preds, targets, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    n = min(500, len(preds))
    ax1.plot(targets[:n], label='Ground truth', color='orange', linewidth=1.0)
    ax1.plot(preds[:n], label='Predicted', color='blue', linewidth=1.0, alpha=0.8)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Strain Ratio')
    ax1.set_title(f'{model_name} - predictions vs ground truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(targets, preds, alpha=0.1, s=1, color='purple')
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax2.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
    ax2.set_xlabel('Ground truth strain ratio')
    ax2.set_ylabel('Predicted strain ratio')
    ax2.set_title(f'{model_name} - scatter plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions.png', dpi=150)
    plt.close()


def plot_attention(model, X_test, n_examples=3):
    model.eval()
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4 * n_examples))
    if n_examples == 1:
        axes = [axes]

    indices = np.random.choice(len(X_test), n_examples, replace=False)

    for i, idx in enumerate(indices):
        x = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            _, weights = model(x, return_attention=True)
            
        weights = weights.cpu().numpy().flatten()
        adc_seq = X_test[idx][:, 0]  # Plot ADC feature sequence

        ax = axes[i]
        ax2 = ax.twinx()
        ax.bar(range(len(weights)), weights, alpha=0.4, color='purple', label='Attention weight')
        ax2.plot(adc_seq, color='steelblue', linewidth=1.5, label='ADC signal')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Attention weight', color='purple')
        ax2.set_ylabel('ADC (normalised)', color='steelblue')
        ax.set_title(f'Attention map - example {i+1}')

    plt.tight_layout()
    plt.savefig('CNN_LSTM_Attention_attention_maps.png', dpi=150)
    plt.close()
    print("Saved attention maps to CNN_LSTM_Attention_attention_maps.png")

# optuna bayesian optimization objective (fixed namespace)
def objective(trial, model_type):
    # Prefix parameter names with model_type to prevent distribution collisions
    lr = trial.suggest_float(f"{model_type}_lr", 1e-4, 3e-3, log=True)
    hidden_size = trial.suggest_categorical(f"{model_type}_hidden_size", [64, 128])
    dropout = trial.suggest_float(f"{model_type}_dropout", 0.2, 0.5, step=0.05)
    batch_size = trial.suggest_categorical(f"{model_type}_batch_size", [16, 32, 64])

    train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    if model_type == "LSTM":
        model = LSTMModel(
            hidden_size=hidden_size, 
            dropout=dropout
        ).to(DEVICE)

    elif model_type == "CNN_LSTM":
        conv1_filters = trial.suggest_categorical(f"{model_type}_conv1_filters", [32, 64])
        kernel_size = trial.suggest_categorical(f"{model_type}_kernel_size", [3, 5, 7])
        model = CNNLSTMModel(
            hidden_size=hidden_size, 
            dropout=dropout, 
            conv1_filters=conv1_filters, 
            kernel_size=kernel_size
        ).to(DEVICE)

    elif model_type == "CNN_LSTM_Attention":
        conv1_filters = trial.suggest_categorical(f"{model_type}_conv1_filters", [32, 64])
        kernel_size = trial.suggest_categorical(f"{model_type}_kernel_size", [3, 5, 7])
        model = CNNLSTMAttentionModel(
            hidden_size=hidden_size, 
            dropout=dropout, 
            conv1_filters=conv1_filters, 
            kernel_size=kernel_size
        ).to(DEVICE)

    # 10 epochs like the paper 
    model, _, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=10, lr=lr, model_name=f"Trial_{trial.number}", verbose=False
    )

    return min(val_losses)


#  main execution pipeline in main 
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    models_to_run = ["LSTM", "CNN_LSTM", "CNN_LSTM_Attention"]
    best_hyperparams = {}
    results = {}

    # step 1: run bayesian optimization across all models
    for model_name in models_to_run:
        print(f"\n--- Starting Bayesian Optimization Study for {model_name} ---")
        
        # explicit fresh study creation per model
        study = optuna.create_study(
            study_name=f"study_{model_name}",
            direction="minimize", 
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        
        def trial_callback(study, trial):
            best_val = f"{study.best_value:.6f}" if len(study.completed_trials) > 0 else "N/A"
            trial_val = f"{trial.value:.6f}" if trial.value is not None else "N/A"
            clean_p = {k.replace(f"{model_name}_", ""): v for k, v in trial.params.items()}
            
            print(f"  Trial {trial.number + 1}/50 | Val loss: {trial_val} | "
                f"Best so far: {best_val} | "
                f"Params: {clean_p}")

        study.optimize(
            lambda trial: objective(trial, model_type=model_name),
            n_trials=50,
            callbacks=[trial_callback])
        
        # clean parameter names (strip the prefix for final retraining)
        clean_params = {k.replace(f"{model_name}_", ""): v for k, v in study.best_params.items()}
        best_hyperparams[model_name] = clean_params

        print(f"\n{'='*60}")
        print(f"Optuna complete for {model_name}")
        print(f"  Best validation loss: {study.best_value:.6f}")
        print(f"  Best hyperparameters:")
        for k, v in clean_params.items():
            print(f"    {k}: {v}")
        print(f"  Trials completed: {len(study.trials)}")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # save trial history for this model to CSV
        trials_df = study.trials_dataframe()
        trials_df.to_csv(f'optuna_trials_{model_name}.csv', index=False)
        print(f"  Trial history saved to optuna_trials_{model_name}.csv")

    # step 2: full retraining and testing using best hyperparameters
    print("\n" + "="*80)
    print("RETRAIN FINAL MODELS WITH OPTIMAL HYPERPARAMETERS")
    print("="*80)

    for model_name, params in best_hyperparams.items():
        batch_size = params['batch_size']
        lr = params['lr']

        train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(SensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(SensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        if model_name == "LSTM":
            final_model = LSTMModel(
                hidden_size=params['hidden_size'], 
                dropout=params['dropout']
            )
        elif model_name == "CNN_LSTM":
            final_model = CNNLSTMModel(
                hidden_size=params['hidden_size'], 
                dropout=params['dropout'],
                conv1_filters=params['conv1_filters'],
                kernel_size=params['kernel_size']
            )
        elif model_name == "CNN_LSTM_Attention":
            final_model = CNNLSTMAttentionModel(
                hidden_size=params['hidden_size'], 
                dropout=params['dropout'],
                conv1_filters=params['conv1_filters'],
                kernel_size=params['kernel_size']
            )

        print(f"\nRetraining {model_name} with optimal hyperparameters")
        print(f"lr={lr:.6f} | hidden={params['hidden_size']} | "
            f"dropout={params['dropout']:.2f} | batch={batch_size}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

        # full training run
        final_model, train_losses, val_losses = train_model(
            final_model, train_loader, val_loader, 
            epochs=EPOCHS, lr=lr, model_name=f"{model_name}_Optimized"
        )

        print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")

        plot_losses(train_losses, val_losses, f"{model_name}_Optimized")
        
        preds, targets, mae, rmse, r2 = evaluate_model(
            final_model, test_loader, f"{model_name}_Optimized", strain_min, strain_max
        )
        plot_predictions(preds, targets, f"{model_name}_Optimized")
        
        results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        if model_name == "CNN_LSTM_Attention":
            plot_attention(final_model, X_test, n_examples=3)

    # step 3: print and save final results comparison
    print("\n===============================================================================")
    print("FINAL BAYESIAN OPTIMIZED RESULTS COMPARISON")
    print("===============================================================================")
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-------------------------------------------------------------------------------")
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['MAE']:>8.4f} {metrics['RMSE']:>8.4f} {metrics['R2']:>8.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save JSON as before
    results_file = f'results_optimized_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            "metrics": {k: {m: float(v) for m, v in metrics.items()}
                        for k, metrics in results.items()},
            "best_hyperparameters": best_hyperparams
        }, f, indent=2)

    # save metrics to CSV
    metrics_rows = []
    for model_name, metrics in results.items():
        row = {
            'model': model_name,
            'MAE': float(metrics['MAE']),
            'RMSE': float(metrics['RMSE']),
            'R2': float(metrics['R2']),
            'session': DATA_FILE.split('/')[-1].replace('_clean.csv', ''),
            'seq_len': SEQ_LEN,
            'epochs': EPOCHS,
            'timestamp': timestamp
        }
        # add best hyperparams as columns
        for k, v in best_hyperparams[model_name].items():
            row[f'hp_{k}'] = v
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = f'results_optimized_{timestamp}.csv'
    metrics_df.to_csv(metrics_csv, index=False)

    # print final table
    print("\n===============================================================================")
    print("FINAL BAYESIAN OPTIMIZED RESULTS")
    print("===============================================================================")
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-"*50)
    for _, row in metrics_df.iterrows():
        print(f"{row['model']:<25} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} {row['R2']:>8.4f}")

    print(f"\nJSON results saved to:    {results_file}")
    print(f"CSV results saved to:     {metrics_csv}")
    print(f"Optuna trial CSVs saved:  optuna_trials_[model].csv for each model")
    print(f"\nAll done. Completed at: {datetime.now().strftime('%H:%M:%S')}")