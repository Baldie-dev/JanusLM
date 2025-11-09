import json, torch, argparse, logging, sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
from peft import PeftModel
from utils import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", required=False, help="Safe and slow benchmark on CPU, for compatibility reasons")
parser.add_argument("--output", default="benchmark_out", required=False, help="Output folder for benchmark results")
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during benchmark")
parser.add_argument("--model", required=True, help="Name of the base model")
parser.add_argument("--lora-adadpter", required=False, help="Folder name with LoRA adapter")
parser.add_argument("--vuln", required=True, choices=Utils.get_vuln_choices(), help="Select category of vulnerability")
parser.add_argument("--max-tokens", default=100, required=False, help="Maximum number of tokens for analysis")
args = parser.parse_args()

# Init environment
load_dotenv()
model_path = os.getenv("MODEL_PATH")
conn = sqlite3.connect('datasets/data.db')
cursor = conn.cursor()
documents = Utils.load_documents()

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load benchmark data
eval_data = Utils.load_data()
vuln_id = Utils.get_vuln_id()

def init_db():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS int_benchmark (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            vuln_id INTEGER NOT NULL,
            result BOOL NOT NULL
        )
        ''')
    conn.commit()

def store_result(result):
    global vuln_id
    model = args.model
    cursor.execute('''
        INSERT INTO int_benchmark (model, vuln_id, result)
        VALUES (?, ?, ?)
    ''', (model, vuln_id, result))
    conn.commit()

def run_benchmark():
    global documents, eval_data
    logger.info("Loading models and prompts...")
    # Run inference
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if args.lora_adapter:
        model = PeftModel.from_pretrained(model, args.lora_adapter)
    prompt_direct = Utils.load_prompt("benchmark_direct_self_class.txt", documents)
    prompt_class = Utils.load_prompt("benchmark_self_class.txt", documents)
    prompt_class_pe = Utils.load_prompt("benchmark_self_class_pe.txt", documents)
    
    model.eval()
    model.config.is_decoder = True
    model.config.is_encoder_decoder = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Model type: {model.config.model_type}")
    logger.info(f"Is Decoder: {model.config.is_decoder}")

    eval_dataset = eval_data[int(len(eval_data)*0.8):]

    def reason(prompt):
        new_prompt = prompt.replace("{analysis}", "")
        analysis = ""
        inputs = tokenizer(new_prompt, return_tensors="pt")
        if args.cpu:
            inputs = inputs.to("cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # TODO: Check minimum number of tokens
        analysis = analysis.split("### Result")[0]
        return prompt.replace("{analysis}", analysis)

    def classify(prompt):
        if "{analysis}" in prompt:
            prompt = reason(prompt)
        prompt += "\n### Result: "
        inputs = tokenizer(prompt, return_tensors="pt")
        if args.cpu:
            inputs = inputs.to("cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "1" in response[-1]:
            return 1
        elif "0" in response[-1]:
            return 0
        else:
            return None

    for item in eval_dataset:
        pred_direct = classify(prompt_direct.replace("{request}",item["request"]).replace("{response}",item["response"]))
        pred_class = classify(prompt_class.replace("{request}",item["request"]).replace("{response}",item["response"]))
        pred_class_pe = classify(prompt_class_pe.replace("{request}",item["request"]).replace("{response}",item["response"]))
        store_result()

# Start benchmark
init_db()
run_benchmark()
