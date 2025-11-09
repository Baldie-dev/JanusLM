import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
from utils import Utils

mpl.rcParams['text.usetex'] = False
plt.rc('text', usetex=False)

# TODO: load results from SQLlite and files
categories = [args.model+"-SC",args.model+"-FT-SC",args.model+"-FT-CH","GPT4","GPT4-Prompt-Engineering"]
gpt4 = gpt4pe = total = 0
with open("datasets/reasoning.jsonl", "r", encoding="utf-8") as f:
    content = ''.join(f.readlines())
    items = json.loads(content)
    for item in items:
        if item['classification'] == item['gpt5_classification']:
            gpt4 += 1
        if item['classification'] == item['gpt5_classification_prompt_engineering']:
            gpt4pe += 1
        total += 1
results = [default_self, finetuned_self, 0, gpt4/total, gpt4pe/total]

# Plotting
plt.rcParams.update({'font.size': 14})
plt.style.use('science')
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
plt.bar(categories, results, color=colors)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Benchmark')
plt.tight_layout()
    
# Save the plot
output_path = 'imgs/'+args.model+'-benchmark.png'
plt.savefig(output_path)
plt.close()