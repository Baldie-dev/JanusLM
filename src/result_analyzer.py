import matplotlib.pyplot as plt
import scienceplots, sqlite3
import pandas as pd
import matplotlib as mpl
from utils import Utils

mpl.rcParams['text.usetex'] = False
plt.rc('text', usetex=False)

def load_data():
    conn = sqlite3.connect('datasets/data.db')
    results = []
    df = pd.read_sql_query("SELECT * FROM ext_benchmark LEFT JOIN training_data ON ext_benchmark.task_id == training_data.id;", conn)
    df2 = pd.read_sql_query("SELECT * FROM int_benchmark LEFT JOIN training_data ON int_benchmark.task_id == training_data.id;", conn)
    for i in range(len(df['id'])):
        results.append({
            'model': df['model'][i],
            'is_vulnerable': bool(df['is_vulnerable'][i]),
            'result': bool(df['result'][i]),
            'vuln_category': int(df['vuln_category'][i])
        })
    for i in range(len(df2['id'])):
        results.append({
            'model': df2['model'][i],
            'is_vulnerable': bool(df2['is_vulnerable'][i]),
            'result': bool(df2['result'][i]),
            'vuln_category': int(df['vuln_category'][i])
        })
    return results

def get_unique_models(data):
    models = []
    for result in data:
        if result['model'] not in models:
            models.append(result['model'])
    return models

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

data = load_data()
models = get_unique_models(data)
accuracies = []
precisions = []
for model in models:
    stats = get_model_stats(data, model)
    precisions.append(stats[0])
    accuracies.append(stats[1])

# Plot Precision
plot_chart(models, precisions, "Precision (\%)", "Precision by model", "precision")

# Plot Accuracy
plot_chart(models, accuracies, "Accuracy (\%)", "Accuracy by model", "accuracy")
