import os
import json
import hashlib
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime
from dataclasses import dataclass, asdict
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------------------------------: experiment config
@dataclass
class ExperimentConfig:
    n_steps: int = 11000
    dt: float = 0.01
    sigma: float = 12.0
    rho: float = 99.0
    beta: float = 8/3
    warmup: int = 1000
    noise_level: float = 0.05
    input_window: int = 100
    n_epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42

    def get_hash(self):
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class ExperimentLogger:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.exp_id = f"{self.timestamp}_{self.config.get_hash()}"
        self.path = f"experiments/{self.exp_id}"
        
        os.makedirs(f"{self.path}/plots", exist_ok=True)
        
        with open(f"{self.path}/params.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=4)
        print(f"experiment registered em: {self.path}")

    def save_results(self, exp_object):
        exp_object.summary(eval_set="test").loc[:, :"z_RMSE"].to_csv(f"{self.path}/test_metrics.csv", index=False)
        exp_object.summary(eval_set="val").loc[:, :"z_RMSE"].to_csv(f"{self.path}/val_metrics.csv", index=False)


# ---------------------------------------------------------------------------------------------------: generate database
def get_lorentz_63_dataframe(n_steps=31000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3, warmup=1000):
    def lorenz_derivs(state, s, r, b):
        x, y, z = state
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        return np.array([dx, dy, dz])
        
    def rk4_step(f, x, dt, s, r, b):
        k1 = f(x, s, r, b)
        k2 = f(x + 0.5 * dt * k1, s, r, b)
        k3 = f(x + 0.5 * dt * k2, s, r, b)
        k4 = f(x + dt * k3, s, r, b)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    states = np.zeros((n_steps, 3))
    states[0] = np.array([1.0, 1.0, 1.0])

    for t in range(1, n_steps):
        states[t] = rk4_step(lorenz_derivs, states[t-1], dt, sigma, rho, beta)

    states = states[warmup:]
    df_lorenz = pd.DataFrame(states, columns=["X", "Y", "Z"])
    
    return df_lorenz

def add_noise_to_dataframe(df, noise_level=0.15, seed=42):
    np.random.seed(seed)
    df_noisy = df.copy()
    for col in df_noisy.columns:
        std = df_noisy[col].std()
        noise = np.random.normal(0, std * noise_level, size=len(df_noisy))
        df_noisy[col] = df_noisy[col] + noise
    return df_noisy

# ---------------------------------------------------------------------------------------------------: database visualization
def plot_lorenz_3d(df, steps=2000, title="3D Trajectory Lorentz 63", save_path=None):
    p = df.iloc[:steps]

    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p["X"], p["Y"], p["Z"], color='black', linewidth=0.8, alpha=0.8)

    for axis, col in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], ["X", "Y", "Z"]):
        margin = (p[col].max() - p[col].min()) * 0.1
        axis(p[col].min() - margin, p[col].max() + margin)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_box_aspect(None, zoom=0.85) 
    plt.suptitle(title, fontsize=14, y=0.92) 
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        
    plt.show()


def plot_axes_timeseries(df, steps=2000, title="Lorenz 63 - Time Series por Eixo", save_path=None):
    p = df.iloc[:steps]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=120)
    
    cols = ['X', 'Y', 'Z']
    
    for i, col in enumerate(cols):
        axes[i].plot(p.index, p[col], color='black', linewidth=1)
        axes[i].set_ylabel(f'{col.upper()}')
        axes[i].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Steps')
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

    plt.show()


# ---------------------------------------------------------------------------------------------------: x and y split
def generate_XY(data, input_window=50, output_window=1):
    windows = {}
    for i in range(len(data) - input_window - output_window):
        x = data[i:i+input_window]
        y = data[i+input_window:i+input_window+output_window]

        windows[f"window_{i}"] = {"X": x, "Y": y}

    return windows

# ---------------------------------------------------------------------------------------------------: modeling
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]   
        y = self.fc(last)
        return y


class GRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h = self.gru(x)   
        last = out[:, -1, :]
        y = self.fc(last)

        return y


class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last = out[:, -1, :]   
        y = self.fc(last)

        return y

        
# ---------------------------------------------------------------------------------------------------: results
def compute_metrics(y_true, y_pred, cols=("x", "y", "z")):
    results = {}

    for i, col in enumerate(cols):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        results[f"{col}_RMSE"] = round(rmse, 6)

    return results


class ModelingExperimentRegression:
    def __init__(self):
        self.records = []

    def save(self, name,
             y_train=None, y_pred_train=None,
             y_val=None, y_pred_val=None,
             y_test=None, y_pred_test=None,
             loss_history=None, training_time=None):

        record = {"Model": name}

        def add_split(prefix, y_true, y_pred):
            if y_true is not None and y_pred is not None:

                m = compute_metrics(y_true, y_pred)
                record.update({f"{prefix}_{k}": v for k, v in m.items()})

                record[f"{prefix}_y_true"] = y_true
                record[f"{prefix}_y_pred"] = y_pred

        add_split("train", y_train, y_pred_train)
        add_split("val", y_val, y_pred_val)
        add_split("test", y_test, y_pred_test)

        if loss_history is not None:
            record["loss_history"] = loss_history
            record["training_time"] = training_time

        self.records.append(record)

    def summary(self, eval_set=None):
        df = pd.DataFrame(self.records)

        if eval_set is None:
            return df

        prefix = f"{eval_set}_"
        cols = ["Model"] + [c for c in df.columns if c.startswith(prefix)]

        df = df[cols].copy()

        df.columns = [
            col.replace(prefix, "") if col != "Model" else col
            for col in df.columns
        ]

        return df


# ---------------------------------------------------------------------------------------------------: evaluation plots
def plot_training_time(exp_records, save_path="training_efficiency.png"):
    models = [r["Model"] for r in exp_records]
    times = [r["training_time"] for r in exp_records]
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)

    color_time = '#BDC3C7'
    bars = ax1.bar(models, times, color=color_time, alpha=0.6, label='Training Time (s)', width=0.5)
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Time (seconds)', color='#7F8C8D', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#7F8C8D')

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#7F8C8D')


    plt.title('Training Time (s)', fontsize=15, fontweight='bold', pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()


def plot_loss_curves(exp_records, save_path="loss_curve.png"):
    """
    Plots learning curves (Train vs Val) for all models in English.
    """
    plt.figure(figsize=(10, 6), dpi=120)
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, r in enumerate(exp_records):
        name = r["Model"]
        lh = r.get("loss_history", {})
        color = colors[i % len(colors)]

        train_loss = lh.get("train", None)
        val_loss = lh.get("val", None)

        if train_loss is not None:
            plt.plot(train_loss, label=f"{name} (Train)", color=color, linewidth=1.5)

        if val_loss is not None:
            plt.plot(val_loss, label=f"{name} (Val)", color=color, 
                     linestyle='--', alpha=0.8, linewidth=1.5)

    plt.title("Learning Curves: Training vs Validation Loss", fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("MSE Loss (Log Scale)", fontsize=12)
    
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.legend(loc='best', frameon=True, fontsize='small')

    max_len = 0
    for r in exp_records:
        lh = r.get("loss_history", {})
        if "train" in lh:
            max_len = max(max_len, len(lh["train"]))

    plt.xticks(np.arange(0, max_len, 1))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    
    plt.show()


def plot_models_comparison_by_axis(exp_records, steps=500, save_path="comparison_by_axis.png"):
    models = exp_records
    n_models = len(models)
    fig, axes = plt.subplots(3, n_models, figsize=(15, 10), sharex=True, dpi=120)
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    axes_names = ['X', 'Y', 'Z']
    colors_real = 'black'
    colors_pred = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    for col_idx, r in enumerate(models):
        model_name = r["Model"]
        
        y_true = r["test_y_true"][:steps]
        y_pred = r["test_y_pred"][:steps]
        
        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            
            ax.plot(y_true[:, row_idx], color=colors_real, label='Real', linewidth=1)
            ax.plot(y_pred[:, row_idx], color=colors_pred[col_idx], label=f'Pred ({model_name})', linewidth=1.2)
            
            if row_idx == 0:
                ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(f"Axis {axes_names[row_idx]}", fontsize=10, fontweight='bold')
    
            if row_idx == 2:
                ax.set_xlabel('Steps', fontsize=10, fontweight='bold')
                
            ax.grid(True, alpha=0.2)
            
            ax.legend(loc='upper right', fontsize='x-small', frameon=True, framealpha=0.8)

    plt.suptitle(f"Comparison by Axis (First {steps} steps of Test Set)", fontsize=14, y=1.02)    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_3d_comparison_one(exp_records, steps=500, save_path="trajectory_3d.png"):

    models = exp_records
    n_models = len(models)

    fig = plt.figure(figsize=(15, 6), dpi=120)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_pred = prop_cycle.by_key()['color']

    for i, r in enumerate(models):
        model_name = r["Model"]
        test_true = r["test_y_true"]
        test_pred = r["test_y_pred"]

        color = colors_pred[i % len(colors_pred)]

        # ---- converter pra DataFrame ----
        true_df = pd.DataFrame(test_true, columns=["x", "y", "z"]).iloc[:steps]
        pred_df = pd.DataFrame(test_pred, columns=["x", "y", "z"]).iloc[:steps]

        ax = fig.add_subplot(1, n_models, i + 1, projection='3d')

        ax.plot(true_df["x"], true_df["y"], true_df["z"],
                color='black', linewidth=1.0, label="Real")

        ax.plot(pred_df["x"], pred_df["y"], pred_df["z"],
                color=color, linewidth=1.2, label=f"Prediction ({model_name})")

        ax.set_title(f"{model_name}", fontsize=13, fontweight='bold')

        ax.set_xlabel('X', labelpad=10)
        ax.set_ylabel('Y', labelpad=10)
        ax.set_zlabel('Z', labelpad=10)

        ax.legend(loc='upper right', fontsize='x-small', framealpha=0.7)

    plt.suptitle("3D Phase Space Trajectory: Ground Truth vs Prediction", fontsize=14)

    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

    plt.show()


import scipy.stats as stats
def plot_qq_matrix(exp_records, save_path="qq_plot_matrix.png"):

    n_models = len(exp_records)
    axes_letters = ['x', 'y', 'z']
    
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 10), dpi=120)
    
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, r in enumerate(exp_records):
        model_name = r["Model"]
        
        for col_idx, letter in enumerate(axes_letters):
            ax = axes[row_idx, col_idx]
            
            y_true = r["test_y_true"][:, col_idx]
            y_pred = r["test_y_pred"][:, col_idx]
            residuals = y_true - y_pred
            
            res_norm = (residuals - np.mean(residuals)) / np.std(residuals)
            
            stats.probplot(res_norm, dist="norm", plot=ax)
            
            ax.get_lines()[0].set_markerfacecolor('#1f77b4')
            ax.get_lines()[0].set_alpha(0.4)
            ax.get_lines()[1].set_color('red')
            ax.get_lines()[1].set_linewidth(2)
            
            if row_idx == 0:
                ax.set_title(f"{letter.upper()}", fontsize=14, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\nTheoretical Quantiles", fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("")
                
            ax.grid(True, alpha=0.2, linestyle='--')

    plt.suptitle("Residuals Analysis: Q-Q Plots (Normal Distribution Consistency)", 
                 fontsize=18, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
