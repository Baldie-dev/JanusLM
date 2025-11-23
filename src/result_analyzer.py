import matplotlib.pyplot as plt
import scienceplots, sqlite3
import pandas as pd
import matplotlib as mpl
from utils import Utils
import numpy as np

mpl.rcParams['text.usetex'] = False
plt.rc('text', usetex=False)

def get_benchmark_accuracy():
    conn = sqlite3.connect('datasets/data.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            b.id AS benchmark_id,
            b.label,
            COUNT(r.id) AS total_tasks,
            SUM(CASE WHEN r.result = t.is_vulnerable THEN 1 ELSE 0 END) AS correct_matches,
            SUM(CASE WHEN r.result != t.is_vulnerable THEN 1 ELSE 0 END) AS incorrect_matches,
            SUM(CASE WHEN t.is_vulnerable = 0 AND r.result = 1 THEN 1 ELSE 0 END) AS false_positives,
            ROUND(
                (SUM(CASE WHEN r.result = t.is_vulnerable THEN 1 ELSE 0 END) * 100.0) / 
                COUNT(r.id), 2
            ) AS accuracy_rate,
            ROUND(
                (SUM(CASE WHEN t.is_vulnerable = 0 AND r.result = 1 THEN 1 ELSE 0 END) * 100.0) / 
                COUNT(r.id), 2
            ) AS false_positive_rate
        FROM benchmarks b
        LEFT JOIN benchmark_results r 
            ON b.id = r.benchmark_id
        LEFT JOIN training_data t
            ON r.task_id = t.id
        GROUP BY b.id, b.label
        ORDER BY b.id
    """)
    
    stats = cursor.fetchall()
    return [
        {
            "benchmark_id": row[0],
            "label": row[1],
            "total_tasks": row[2],
            "correct_matches": row[3],
            "incorrect_matches": row[4],
            "accuracy_rate": row[6],
            "false_positive_rate": row[7]
        }
        for row in stats
    ]


def get_model_stats(data, model):
    TP = FP = TN = FN = 0
    for result in data:
        if result['model'] == model:
            if result['is_vulnerable']:
                if result['result']:
                    TP += 1
                else:
                    FN += 1
            else:
                if not result['result']:
                    TN += 1
                else:
                    FP += 1
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    accuracy = (TP+FP)/(TP+FP+TN+FN)
    return precision, accuracy

def plot_chart(categories, values, yaxis, title, filename):
    plt.rcParams.update({'font.size': 14})
    plt.style.use('science')
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=Utils.colors)
    plt.xlabel('Model')
    plt.ylabel(yaxis)
    plt.title(title)
    plt.tight_layout()
    output_path = 'imgs/'+filename+'-benchmark.png'
    plt.savefig(output_path)
    plt.close()

def plot_lines(lines, xaxis, yaxis, title, filename):
    mpl.style.use("science")
    plt.figure(figsize=(8, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    i = 0
    for line in lines:
        color = colors[i % len(colors)]
        i += 1
        plt.plot(line["x"], line["y"], linestyle='--',color=color, linewidth=0.8, marker='x', markersize=12, label=line["label"])
    plt.xlabel(xaxis, fontsize=14)
    plt.ylabel(yaxis, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    output_path = 'imgs/'+filename+'-benchmark.png'
    plt.savefig(output_path)

def plot_benchmark_accuracy(stats):
    lines = []
    lines_fp = []
    models = ["Qwen3-1.7B-lora", "Qwen3-4B-lora", "gpt-5-mini", "deepseek-chat"]
    for model in models:
        x = []
        y = []
        y_fp = []
        for stat in stats:
            modelname = "-".join(stat['label'].split("-")[:-1])
            if model == modelname:  
                length = int(stat['label'].split('-')[-1].replace('w',''))
                x.append(length)
                y.append(stat['accuracy_rate'])
                y_fp.append(stat['false_positive_rate'])
        lines.append({"label": model, "x": x, "y": y})
        lines_fp.append({"label": model, "x": x, "y": y_fp})
    plot_lines(lines, "Analysis Length", "Accuracy Rate (\%)", "Accuracy Rate by Analysis Length", "accuracy-lora-by-length")
    plot_lines(lines_fp, "Analysis Length", "False-Positive Rate (\%)", "False-Positive Rate by Analysis Length", "false-positive-lora-by-length")


def plot_benchmark_lora_improvment(stats):
    groups = [
        ("Qwen3-1.7B-0", "Qwen3-1.7B-lora-0"),
        ("Qwen3-1.7B-500", "Qwen3-1.7B-lora-300"),
        ("Qwen3-4B-0", "Qwen3-4B-lora-0"),
        ("Qwen3-4B-500", "Qwen3-4B-lora-300"),
    ]
    bars = []
    for baseline_label, lora_label in groups:
        base_acc = next((s['accuracy_rate'] for s in stats if s['label'] == baseline_label), 0)
        lora_acc = next((s['accuracy_rate'] for s in stats if s['label'] == lora_label), 0)

        bars.append({
            "group": baseline_label, 
            "baseline": base_acc,
            "lora": lora_acc
        })
    bars_fp = []
    for baseline_label, lora_label in groups:
        base_acc = next((s['false_positive_rate'] for s in stats if s['label'] == baseline_label), 0)
        lora_acc = next((s['false_positive_rate'] for s in stats if s['label'] == lora_label), 0)

        bars_fp.append({
            "group": baseline_label, 
            "baseline": base_acc,
            "lora": lora_acc
        })
    x = np.arange(len(bars))
    x_fp = np.arange(len(bars_fp))
    # Accuracy Rate Plot
    width = 0.35
    mpl.rcParams['text.usetex'] = False
    plt.rc('text', usetex=False)
    mpl.style.use("science")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(True, zorder=0, linewidth=0.4) 
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': 14})
    plt.style.use('science')
    baseline_vals = [b["baseline"] for b in bars]
    lora_vals = [b["lora"] for b in bars] 
    ax.bar(x - width/2, baseline_vals, width, label="Base", color=Utils.colors[0], zorder=2)
    ax.bar(x + width/2, lora_vals, width, label="LoRA", color=Utils.colors[1], zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([b["group"] for b in bars], rotation=45, ha="right")
    ax.set_ylabel("Accuracy Rate (\%)", fontsize=14)
    ax.set_title("LoRA Accuracy Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig('imgs/lora-accuracy-improvment-benchmark.png')
    # False Positive Rate Plot
    mpl.rcParams['text.usetex'] = False
    plt.rc('text', usetex=False)
    mpl.style.use("science")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(True, zorder=0, linewidth=0.4) 
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': 14})
    plt.style.use('science')
    baseline_vals = [b["baseline"] for b in bars_fp]
    lora_vals = [b["lora"] for b in bars_fp]
    ax.bar(x - width/2, baseline_vals, width, label="Base", color=Utils.colors[0], zorder=2)
    ax.bar(x + width/2, lora_vals, width, label="LoRA", color=Utils.colors[1], zorder=2)
    ax.set_xticks(x_fp)
    ax.set_xticklabels([b["group"] for b in bars_fp], rotation=45, ha="right")
    ax.set_ylabel("False-Positive Rate (\%)", fontsize=14)
    ax.set_title("LoRA False-Positive Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig('imgs/lora-false-positive-improvment-benchmark.png')

def print_benchmark_stats(stats):
    print(f"{'Benchmark':<25} {'Total Tasks':<15} {'Correct':<10} {'Incorrect':<12} {'Accuracy (%)':<15} {'False-Positive (%)':<15}")
    print("-" * 80)
    for stat in stats:
        print(f"{stat['label']:<25} {stat['total_tasks']:<15} {stat['correct_matches']:<10} {stat['incorrect_matches']:<12} {stat['accuracy_rate']:<15} {stat['false_positive_rate']:<15}")

stats = get_benchmark_accuracy()
print_benchmark_stats(stats)
plot_benchmark_accuracy(stats)
#plot_benchmark_lora_improvment(stats)
