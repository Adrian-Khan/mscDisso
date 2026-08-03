"""
train.py

i will train all three models here on session_08_clean.csv (my best dataset)

input: adc_raw (normalised)
output: strain_ratio (normalised)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


# config - experiment with later and are there any papers with some better settings after we do some typical ones 
DATA_FILE = "/scratch0/adrikhan/sensor_project/session_008_clean.csv"
SEQ_LEN = 120    # timesteps per sequence should be around 120 ish
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# test split is remainder: 0.15
EARLY_STOP_PATIENCE = 40
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Using device: {DEVICE}")
print(f"Loading data from: {DATA_FILE}")


# load and process data 
df = pd.read_csv(DATA_FILE)

# use adc_raw as input and strain_ratio as output
adc = df['adc_raw'].values.astype(np.float32)
strain = df['strain_ratio'].values.astype(np.float32)
breaks = df['sequence_break'].values.astype(bool)

# normalise adc to 0-1
adc_min, adc_max = adc.min(), adc.max()
strain_min, strain_max = strain.min(), strain.max()

adc_norm = (adc - adc_min) / (adc_max - adc_min)
strain_norm = (strain - strain_min) / (strain_max - strain_min)

# compute gradient (rate of change) of normalised ADC
adc_grad = np.gradient(adc_norm).astype(np.float32)

# normalise gradient to 0-1
grad_min, grad_max = adc_grad.min(), adc_grad.max()
adc_grad_norm = (adc_grad - grad_min) / (grad_max - grad_min)

# save normalisation params including gradient
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

# create sequences 
# using sliding window sequences that do not cross sequence breaks 
def make_sequences(adc_norm, adc_grad_norm, strain_norm, breaks, seq_len):
    X, y = [], []
    n = len(adc_norm)
    for i in range(n - seq_len):
        if breaks[i+1 : i+seq_len+1].any():
            continue
        # stack adc and gradient as two features
        # seq = adc_norm[i:i+seq_len][:, None] # replace w raw values for testing but then REREPLACE WITH NORMALISED!
        seq = np.stack([adc_norm[i : i+seq_len], adc_grad_norm[i : i+seq_len]], axis=-1)  # shape: (seq_len, 2)
        X.append(seq)
        y.append(strain_norm[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_sequences(adc_norm, adc_grad_norm, strain_norm, breaks, SEQ_LEN)

print(f"Total sequences: {len(X)}")
print(f"Input shape: {X.shape} Output shape: {y.shape}")


# train / val / test split
# split chronologically - do not shuffle time series
n = len(X)
n_train = int(n * TRAIN_SPLIT)
n_val = int(n * VAL_SPLIT)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

print(f"\nTrain: {len(X_train)} sequences")
print(f"Validation: {len(X_val)} sequences")
print(f"Test: {len(X_test)} sequences")


# dataset and dataloader
class SensorDataset(Dataset):
    def __init__(self, X, y):
        # X is already (N, seq_len, 2) - no unsqueeze needed
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(SensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# model definitions 
"""
LSTM MODEL - JUST LSTM
"""
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
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
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # take last timestep
        return self.fc(out)


"""
CNN-LSTM 
"""
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # back to (batch, seq_len, channels) for LSTM
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

"""
CNN LSTM WITH ATTENTION
"""
# attention module 
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_size)
        score = torch.tanh(self.attn(lstm_out))
        weight = torch.softmax(self.context(score), dim=1)
        # weighted sum over timesteps
        out = (weight * lstm_out).sum(dim=1)
        return out, weight.squeeze(-1)

# model 
class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=64,
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


"""
train func
"""
def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    print("\n===============================================================================")
    print(f"Training: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n===============================================================================")

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    patience_count = 0
    early_stop_patience = EARLY_STOP_PATIENCE

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimiser.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)

        # validate
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_batch_losses.append(loss.item())
        val_loss = np.mean(val_batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train loss: {train_loss:.6f} | "
                  f"Val loss: {val_loss:.6f} | "
                  f"Best val: {best_val_loss:.6f}")

        # early stopping
        if patience_count >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # load best weights
    model.load_state_dict(best_state)

    # save model
    save_path = f'{model_name}_best.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")

    return model, train_losses, val_losses

"""
eval
"""
# evaluation
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

    # denormalise back to strain ratio
    preds = preds_norm * (strain_max - strain_min) + strain_min
    targets = targets_norm * (strain_max - strain_min) + strain_min

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    # R squared
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n{model_name} TEST RESULTS:")
    print(f"MAE: {mae:.4f} strain ratio units")
    print(f"RMSE: {rmse:.4f} strain ratio units")
    print(f"R2: {r2:.4f}")

    return preds, targets, mae, rmse, r2


"""
plot
"""
# funcs 
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

    # time series - first 500 predictions
    n = min(500, len(preds))
    ax1.plot(targets[:n], label='Ground truth',
             color='orange', linewidth=1.0)
    ax1.plot(preds[:n], label='Predicted', color='blue', linewidth=1.0, alpha=0.8)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Strain Ratio')
    ax1.set_title(f'{model_name} - predictions vs ground truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # scatter
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
        # FIX: Keep input as 3D (batch_size=1, seq_len, features)
        x = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            _, weights = model(x, return_attention=True)
            
        weights = weights.cpu().numpy().flatten()
        # flatten adc_seq so it plots cleanly as a 1D sequence against weights
        adc_seq = X_test[idx].squeeze()

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


"""
main training loop
"""
results = {}

# model 1: lstm 
lstm_model = LSTMModel()
lstm_model, lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader, EPOCHS, LR, 'LSTM')
plot_losses(lstm_train_losses, lstm_val_losses, 'LSTM')
lstm_preds, lstm_targets, lstm_mae, lstm_rmse, lstm_r2 = evaluate_model(lstm_model, test_loader, 'LSTM', strain_min, strain_max)
plot_predictions(lstm_preds, lstm_targets, 'LSTM')
results['LSTM'] = {'MAE': lstm_mae, 'RMSE': lstm_rmse, 'R2': lstm_r2}

# model 2: cnn-lstm
cnn_lstm_model = CNNLSTMModel()
cnn_lstm_model, cnn_lstm_train_losses, cnn_lstm_val_losses = train_model(cnn_lstm_model, train_loader, val_loader, EPOCHS, LR, 'CNN_LSTM')
plot_losses(cnn_lstm_train_losses, cnn_lstm_val_losses, 'CNN_LSTM')
cnn_lstm_preds, cnn_lstm_targets, cnn_lstm_mae, cnn_lstm_rmse, cnn_lstm_r2 = evaluate_model(cnn_lstm_model, test_loader, 'CNN_LSTM', strain_min, strain_max)
plot_predictions(cnn_lstm_preds, cnn_lstm_targets, 'CNN_LSTM')
results['CNN_LSTM'] = {'MAE': cnn_lstm_mae,'RMSE': cnn_lstm_rmse, 'R2': cnn_lstm_r2}

# model 3: cnn-lstm with attention
attn_model = CNNLSTMAttentionModel()
attn_model, attn_train_losses, attn_val_losses = train_model(attn_model, train_loader, val_loader, EPOCHS, LR, 'CNN_LSTM_Attention')
plot_losses(attn_train_losses, attn_val_losses, 'CNN_LSTM_Attention')
attn_preds, attn_targets, attn_mae, attn_rmse, attn_r2 = evaluate_model(attn_model, test_loader, 'CNN_LSTM_Attention', strain_min, strain_max)
plot_predictions(attn_preds, attn_targets, 'CNN_LSTM_Attention')
results['CNN_LSTM_Attention'] = {'MAE': attn_mae, 'RMSE': attn_rmse, 'R2': attn_r2}

# attention maps on test set
plot_attention(attn_model, X_test, n_examples=3)

# final comparison of all 
print("\n===============================================================================")
print("FINAL RESULTS COMPARISON")
print("\n===============================================================================")
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("\n===============================================================================")
for name, metrics in results.items():
    print(f"{name:<25} "
          f"{metrics['MAE']:>8.4f} "
          f"{metrics['RMSE']:>8.4f} "
          f"{metrics['R2']:>8.4f}")

# save results to json
results_file = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

with open(results_file, 'w') as f:
    json.dump({k: {m: float(v) for m, v in metrics.items()} 
               for k, metrics in results.items()}, f, indent=2)
    
print(f"\nResults saved to {results_file}")
print("\nAll done.")