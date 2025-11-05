import argparse, sqlite3, random, os, logging
from dotenv import load_dotenv
from openai import OpenAI

conn = sqlite3.connect('datasets/data.db')
cursor = conn.cursor()

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_URL")
    )

def init():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request TEXT NOT NULL,
            response TEXT NOT NULL,
            analysis TEXT NOT NULL,
            is_vulnerable BOOL NOT NULL,
            vuln_category INTEGER NOT NULL
        )
        ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vuln_category (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        )''')
    cursor.execute('''INSERT INTO vuln_category VALUES (1, 'HTTP_HEADERS');''')
    cursor.execute('''INSERT INTO vuln_category VALUES (2, 'XSS');''')
    conn.commit()

def store_data(request, response, is_vulnerable):
    pass

def submit_prompt(prompt):
    completion = client.chat.completions.create(
        model=os.getenv("LLM_API_MODEL"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content[-1]

def generate_data():
    # Load templates
    logger.info("Loading request/response templates...")
    templates = []
    with open(args.templates, "r", encoding="utf-8") as f:
        samples = (''.join(f.readlines())).split("\n<<<<>>>>\n")
        for i in range(0, len(samples), 2):
            templates.append({'request': samples[i], 'response': samples[i+1]})
    logger.info("Loading prompts...")
    with open("prompts/data_generation_request_enhancer.txt", "r", encoding="utf-8") as f: prompt_tmp_req_enhance = f.read()
    with open("prompts/data_generation_response_enhancer.txt", "r", encoding="utf-8") as f: prompt_tmp_res_enhance = f.read()
    for i in range(args.num):
        # Randomly choose template
        chosen = random.choice(templates)
        # Define data enhancment agent prompt
        prompt_req_enhance = prompt_tmp_req_enhance.replace("{template}",f"{chosen['request']}")
        # Enhance request
        request = submit_prompt(prompt_req_enhance)
        prompt_res_enhance = prompt_tmp_res_enhance.replace("{request}", request).replace("{template}",f"{chosen['response']}")
        # Enhance response
        response = submit_prompt(prompt_res_enhance)

init()
parser = argparse.ArgumentParser()
parser.add_argument("--num", default=1, type=int, required=False, help="number of generated request/response pairs")
parser.add_argument("--templates", default="datasets/req_res_templates.txt", help="templates for request/response pairs.")
parser.add_argument("--vuln", required=True, choices=['HTTP_HEADERS','XSS'], help="Select category of vulnerability")
parser.add_argument("--verbose", action="store_true", help="Activates detailed log output")
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


generate_data()