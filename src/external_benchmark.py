import argparse, sqlite3, random, os, logging
from dotenv import load_dotenv
from openai import OpenAI
from utils import Utils
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during benchmark")
parser.add_argument("--vuln", required=True, choices=Utils.get_vuln_choices(), help="Select category of vulnerability")
parser.add_argument("--max-tokens", default=100, required=False, help="Maximum number of tokens for analysis")
parser.add_argument("--label", required=True, help="Label for benchmark")
args = parser.parse_args()

# Init global vars
conn = sqlite3.connect('datasets/data.db')
cursor = conn.cursor()
documents = Utils.load_documents()
load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_URL")
)

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load benchmark data
eval_data = Utils.load_data()

# --------- Database functions ---------
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

def store_result(benchmark_id, task_id, result):
    cursor.execute('''
        INSERT INTO benchmark_results (benchmark_id, task_id, result)
        VALUES (?, ?, ?)
    ''', (benchmark_id, task_id, result))
    conn.commit()

def get_response(client: OpenAI, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_API_MODEL"),
            messages=[{"role": "user", "content": prompt}]
        )
        output_text = response.choices[0].message.content.strip()
        return output_text[-2:].strip()
    
    except Exception as e:
        print(f"Error while fetching response: {e}")
        return ""


def run_benchmark(benchmark_id: int, label_ids: list[int]):
    global documents, client
    logger.info("Loading models and prompts...")
    eval_dataset = Utils.load_data()
    prompt_class = Utils.load_prompt("benchmark_self_class_ext.txt", documents)
    max_words = int(args.max_tokens)
    prompt_class = prompt_class.replace("{word_count}", str(max_words))

    for task in eval_dataset:
        if (task["id"] in label_ids):
            continue
        is_vuln = int(task["is_vulnerable"])

        prompt = prompt_class.replace("{request}",task["request"]).replace("{response}",task["response"])
        response = int(get_response(client, prompt))
        print(f"expected: {is_vuln}; got: {response}")
        #store_result(benchmark_id, task["id"], response)
        exit(0)


init_db()

# --------- Start benchmark ---------

benchmark_id = get_or_create_benchmark_id(args.label)
label_ids = get_task_ids_by_benchmark_id(benchmark_id)
run_benchmark(benchmark_id, label_ids)