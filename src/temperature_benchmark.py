import json, torch, argparse, logging, sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os, time
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
parser.add_argument("--rtemperature", required=True, help="Temperature for reasoning")
parser.add_argument("--ctemperature", required=True, help="Temperature for classification")
parser.add_argument("--iterations", required=False, help="Number of iterations per task, default 1", default=1)
parser.add_argument("--task-id", required=False, help="task id", default=1)
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

def init_db():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_t_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            benchmark_id INTEGER NOT NULL,
            task_id INTEGER NOT NULL,
            result BOOL NOT NULL
        )
        ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmarks_t (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL
        )
        ''')
    conn.commit()

def store_result(benchmark_id, task_id, result):
    cursor.execute('''
        INSERT INTO benchmark_t_results (benchmark_id, task_id, result)
        VALUES (?, ?, ?)
    ''', (benchmark_id, task_id, result))
    conn.commit()

def get_or_create_benchmark_id(label: str) -> int:
    cursor.execute("SELECT id FROM benchmarks_t WHERE label = ?", (label,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("INSERT INTO benchmarks_t (label) VALUES (?)", (label,))
    return cursor.lastrowid

def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def run_benchmark(benchmark_id: int):
    global documents
    logger.info("Loading models and prompts...")
    eval_dataset = Utils.load_data()
    janus = JanusClassification(args.model_path, args.lora_adapter, is_cpu=True)
    documents = Utils.load_documents()
    prompt_class_pe = Utils.load_prompt("task_self_class_pe.txt", documents)
    prompt_class_pe_class = Utils.load_prompt("task_self_class_pe_class.txt", documents)

    total = int(args.iterations)
    logger.info(f"Total tasks to run: {total}")
    start_time = time.time()
    task = eval_dataset[0]

    for idx in range(1, total + 1):
        iter_start = time.time()
        # 1 Create initial task
        is_vuln = int(task["is_vulnerable"])
        prompt = prompt_class_pe.replace("{request}",task["request"]).replace("{response}",task["response"])

        # 2 Create a reasoning
        analysis = ""
        if int(args.max_tokens) > 0:
            analysis = janus.complete(prompt.replace("{analysis}",""), int(args.max_tokens), temperature=int(args.rtemperature))
            analysis = analysis.split("### Analysis")[1].strip()

        # 3 Perform Assesment
        prompt2 = prompt_class_pe_class.replace("{request}",task["request"]).replace("{response}",task["response"])
        assesment = janus.complete(prompt2.replace("{analysis}", analysis), 1, temperature=int(args.ctemperature))
        try:
            assesment = int(assesment.split("### Result: ")[1].strip())
        except:
            continue

        # Timing stats
        iter_end = time.time()
        iter_duration = iter_end - iter_start
        elapsed = iter_end - start_time
        avg_time = elapsed / idx
        remaining = total - idx
        eta = remaining * avg_time
        eta_str = format_eta(eta)

        logger.info(
            f"[{idx}/{total}] expected: {is_vuln}; got: {assesment} | "
            f"pass_time={iter_duration:.2f}s | avg={avg_time:.2f}s | "
            f"elapsed={elapsed:.2f}s | remaining={remaining} | ETA={eta_str}"
        )

        store_result(benchmark_id, task["id"], assesment)


# Start benchmark
init_db()
benchmark_id = get_or_create_benchmark_id(args.label)
run_benchmark(benchmark_id)