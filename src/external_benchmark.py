import argparse, sqlite3, random, os, logging
from dotenv import load_dotenv
from openai import OpenAI
from utils import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="benchmark_out", required=False, help="Output folder for benchmark results")
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during benchmark")
args = parser.parse_args()

# Init global vars
conn = sqlite3.connect('datasets/data.db')
cursor = conn.cursor()
documents = Utils.load_documents()
load_dotenv()

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load benchmark data
eval_data = Utils.load_data()

def init_db():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ext_benchmark (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            task_id INTEGER NOT NULL,
            result BOOL NOT NULL
        )
        ''')
    conn.commit()

def store_result(model, vuln_id, result):
    cursor.execute('''
        INSERT INTO ext_benchmark (model, task_id, result)
        VALUES (?, ?, ?)
    ''', (model, vuln_id, result))
    conn.commit()

def run_benchmark():
    pass

init_db()