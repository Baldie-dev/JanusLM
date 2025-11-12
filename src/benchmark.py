import json, torch, argparse, logging, sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
from peft import PeftModel
from utils import Utils
from JanusLModel import JanusClassification

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", required=False, help="Safe and slow benchmark on CPU, for compatibility reasons")
parser.add_argument("--output", default="benchmark_out", required=False, help="Output folder for benchmark results")
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during benchmark")
parser.add_argument("--lora-adapter", required=False, help="Folder name with LoRA adapter")
parser.add_argument("--vuln", required=True, choices=Utils.get_vuln_choices(), help="Select category of vulnerability")
parser.add_argument("--max-tokens", default=100, required=False, help="Maximum number of tokens for analysis")
parser.add_argument("--model-path", required=True)
parser.add_argument("--label", required=True, help="Label for benchmark")
args = parser.parse_args()

# Init environment
load_dotenv()
conn = sqlite3.connect('datasets/data.db')
cursor = conn.cursor()
documents = Utils.load_documents()

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load benchmark data
vuln_id = Utils.get_vuln_id(args.vuln)

def run_benchmark():
    global documents, eval_data
    logger.info("Loading models and prompts...")
    eval_dataset = Utils.load_data()
    FP = FN = TP = TN = 0
    for task in eval_dataset:
        janus = JanusClassification(args.model_path, args.lora_adapter, is_cpu=True)
        documents = Utils.load_documents()
        prompt_class_pe = Utils.load_prompt("task_self_class_pe.txt", documents)
        prompt_class_pe_class = Utils.load_prompt("task_self_class_pe_class.txt", documents)

        # 1 Create initial task
        is_vuln = int(task["is_vulnerable"])
        prompt = prompt_class_pe.replace("{request}",task["request"]).replace("{response}",task["response"])

        # 2 Create a reasoning
        analysis = janus.complete(prompt.replace("{analysis}",""), args.max_tokens)
        analysis = analysis.split("### Analysis")[1].strip()

        # 3 Perform Assesment
        prompt2 = prompt_class_pe_class.replace("{request}",task["request"]).replace("{response}",task["response"])
        assesment = janus.complete(prompt2.replace("{analysis}", analysis), 1)
        try:
            assesment = int(assesment.split("### Result: ")[1].strip())
        except:
            if is_vuln:
                FN += 1
            else:
                FP += 1
            continue
        print(f"expected: {is_vuln}; got: {assesment}")

        if is_vuln:
            if assesment:
                TP += 1
            else:
                FN += 1
        else:
            if assesment:
                FP += 1
            else:
                TN += 1
    if TP+FP == 0:
        precision = 0
    else:
        precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+FN+FP+TN)
    with open("benchmarks.txt", "a") as f:
        f.write(f"\n{args.label},{precision:.4f},{accuracy:.4f}")


# Start benchmark
run_benchmark()
