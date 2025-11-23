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
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            benchmark_id INTEGER NOT NULL,
            task_id INTEGER NOT NULL,
            result BOOL NOT NULL
        )
        ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL
        )
        ''')
    conn.commit()

def store_result(benchmark_id, task_id, result):
    cursor.execute('''
        INSERT INTO benchmark_results (benchmark_id, task_id, result)
        VALUES (?, ?, ?)
    ''', (benchmark_id, task_id, result))
    conn.commit()

def get_or_create_benchmark_id(label: str) -> int:
    cursor.execute("SELECT id FROM benchmarks WHERE label = ?", (label,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("INSERT INTO benchmarks (label) VALUES (?)", (label,))
    return cursor.lastrowid

def get_task_ids_by_benchmark_id(benchmark_id: int) -> list[int]:
    cursor.execute(
        "SELECT task_id FROM benchmark_results WHERE benchmark_id = ?",
        (benchmark_id,)
    )
    rows = cursor.fetchall()
    return [row[0] for row in rows]

def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def run_benchmark(benchmark_id: int, label_ids: list[int]):
    global documents
    logger.info("Loading models and prompts...")
    eval_dataset = Utils.load_data()
    janus = JanusClassification(args.model_path, args.lora_adapter, is_cpu=True)
    documents = Utils.load_documents()
    prompt_class_pe = Utils.load_prompt("task_self_class_pe.txt", documents)
    prompt_class_pe_class = Utils.load_prompt("task_self_class_pe_class.txt", documents)

    tasks_to_run = [task for task in eval_dataset if task["id"] not in label_ids]
    total = len(tasks_to_run)
    logger.info(f"Total tasks to run: {total}")
    start_time = time.time()

    for idx, task in enumerate(tasks_to_run, start=1):
        iter_start = time.time()
        # 1 Create initial task
        is_vuln = int(task["is_vulnerable"])
        prompt = prompt_class_pe.replace("{request}",task["request"]).replace("{response}",task["response"])

        # 2 Create a reasoning
        analysis = ""
        if int(args.max_tokens) > 0:
            analysis = janus.complete(prompt.replace("{analysis}",""), int(args.max_tokens))
            analysis = analysis.split("### Analysis")[1].strip()

        # 3 Perform Assesment
        prompt2 = prompt_class_pe_class.replace("{request}",task["request"]).replace("{response}",task["response"])
        assesment = janus.complete(prompt2.replace("{analysis}", analysis), 1)
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
label_ids = get_task_ids_by_benchmark_id(benchmark_id)
run_benchmark(benchmark_id, label_ids)
