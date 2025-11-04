import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

load_dotenv()
model_path = os.getenv("MODEL_PATH")

# Load benchmark data
eval_data = []
with open("datasets/reasoning.jsonl", "r", encoding="utf-8") as f:
    content = ''.join(f.readlines())
    items = json.loads(content)
    for item in items:
        eval_data.append({
            "prompt": item["prompt"],
            "true_class": item["classification"]
        })

def run_benchmark_default_self_classification():
    print("Running default self classification benchmark")
    # Run inference
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    model.config.is_decoder = True
    model.config.is_encoder_decoder = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    print(f"Model type: {model.config.model_type}")
    print(f"Is Decoder: {model.config.is_decoder}")

    def classify_prompt(prompt):
        prompt += """Analyze following pair of request/response from cybersecurity perspective.
Evaluate HTTP response headers for missing or misconfgired any security headers.

Answer with only single digit 1 or 0 when:
1 - HTTP response contains misconfigured headers and requires attention.
0 - HTTP response is properly configured.\n\n### Response:"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        if "1" in response[-1]:
            return 1
        elif "0" in response[-1]:
            return 0
        else:
            return None

    correct = 0
    total = 0

    for item in eval_data:
        pred = classify_prompt(item["prompt"])
        total += 1
        if pred is not None:
            if pred == item["true_class"]:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Model Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

default_self = run_benchmark_default_self_classification()

model = 'ollama-3.1-1B'
categories = [model+"-SC",model+"-FT-SC",model+"-FT-CH","GPT4","GPT4-Prompt-Engineering"]
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
sales = [default_self, 0, 0, gpt4/total, gpt4pe/total]

# Plotting
plt.style.use('science')
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
plt.bar(categories, sales, color=colors)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Benchmark')
plt.tight_layout()

# Save the plot
output_path = 'imgs/'+model+'-benchmark.png'
plt.savefig(output_path)
plt.close()