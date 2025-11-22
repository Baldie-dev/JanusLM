import matplotlib.pyplot as plt
import scienceplots, sqlite3
import pandas as pd
import matplotlib as mpl
from utils import Utils

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
            ROUND(
                (SUM(CASE WHEN r.result = t.is_vulnerable THEN 1 ELSE 0 END) * 100.0) / 
                COUNT(r.id), 2
            ) AS accuracy_rate
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
            "accuracy_rate": row[5]
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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    plt.bar(categories, values, color=colors)
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
    plt.show()
    output_path = 'imgs/'+filename+'-benchmark.png'
    plt.savefig(output_path)

def plot_benchmark_accuracy(stats):
    lines = []
    models = ["Qwen3-1.7B-lora", "Qwen3-4B-lora"]
    for model in models:
        x = []
        y = []
        for stat in stats:
            if model in stat['label']:
                length = int(stat['label'].split('-')[-1])
                x.append(length)
                y.append(stat['accuracy_rate'])
        lines.append({"label": model, "x": x, "y": y})
    plot_lines(lines, "Analysis Length", "Accuracy Rate (\%)", "Accuracy Rate by Analysis Length", "accuracy-lora-by-length")


stats = get_benchmark_accuracy()
plot_benchmark_accuracy(stats)