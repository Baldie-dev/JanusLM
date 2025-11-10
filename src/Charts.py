import os, csv
import matplotlib.pyplot as plt
import scienceplots, sqlite3
import pandas as pd
import matplotlib as mpl
from utils import Utils
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


mpl.rcParams['text.usetex'] = False
plt.rc('text', usetex=False)

def find_logs():
    current_dir = os.getcwd()
    matching_folders = []
    logs = []
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path):
            log_file_path = os.path.join(item_path, "log_history.csv")
            if os.path.isfile(log_file_path):
                matching_folders.append(item)
    if matching_folders:
        for folder in matching_folders:
            logs.append(os.path.join(folder, "log_history.csv"))
    else:
        print("No folders contain 'log_history.csv'")
    return logs

def extract_epoch_loss(csv_path):
    result = {"epoch": [], "loss": []}
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)
            try:
                epoch_idx = headers.index("epoch")
                loss_idx = headers.index("loss")
            except ValueError:
                print("Required columns 'epoch' or 'loss' not found.")
                return result
            for row in reader:
                try:
                    epoch_val = float(row[epoch_idx])
                    loss_val = float(row[loss_idx])
                    result["epoch"].append(epoch_val)
                    result["loss"].append(loss_val)
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return result

def moving_average(x, y, window_size=5):
    smoothed_x = []
    smoothed_y = []
    for i in range(len(x) - window_size + 1):
        window_x = x[i:i+window_size]
        window_y = y[i:i+window_size]
        smoothed_x.append(np.mean(window_x))
        smoothed_y.append(np.mean(window_y))
    return smoothed_x, smoothed_y

# Used to visualize training loss over epochs of single training run
def plot_epoch_loss(data, title="Epoch vs Loss"):
    mpl.style.use("science")
    epochs = data["epoch"]
    losses = data["loss"]
    coeffs = np.polyfit(epochs, losses, deg=2)
    poly = np.poly1d(coeffs)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, 'o-',color='grey', label="Loss", markersize=3)

    smoothed = lowess(losses, epochs, frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], 'r--', linewidth=1.5, label="Approximation")

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_epoch_loss_comparison(datas):
    mpl.style.use("science")
    plt.figure(figsize=(8, 5))
    for idx, csv_path in enumerate(logs):
        data = extract_epoch_loss(csv_path)
        name = os.path.dirname(csv_path)
        epochs = data["epoch"]
        losses = data["loss"]
        smoothed = lowess(losses, epochs, frac=0.2)
        plt.plot(smoothed[:, 0], smoothed[:, 1], 'r--', linewidth=1.5, label=name)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Epoch vs Loss Comparison", fontsize=15)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

logs = find_logs()
plot_epoch_loss(extract_epoch_loss(logs[0]))
#plot_epoch_loss_comparison(logs)