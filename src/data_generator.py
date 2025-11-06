import argparse, sqlite3, random, os, logging
from dotenv import load_dotenv
from openai import OpenAI

conn = sqlite3.connect('datasets/data.db')
cursor = conn.cursor()
total_in_tokens = 0
total_out_tokens = 0

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_URL")
    )

VULNERABILITIES = [
    [1, 'HTTP_HEADERS', 'Misconfigured HTTP Headers'],
    [2, 'XSS', 'Cross-Site Scripting']
]

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
    
    for vuln in VULNERABILITIES:
        try:
            cursor.execute("INSERT INTO vuln_category VALUES ("+str(vuln[0])+", '"+vuln[1]+"');")
        except:
            pass
    conn.commit()

def store_data(request, response, is_vulnerable, reasoning):
    vuln_id = None
    for vuln in VULNERABILITIES:
        if vuln[1] == args.vuln:
            vuln_id = vuln[0]
            break

    cursor.execute('''
        INSERT INTO training_data (request, response, analysis, is_vulnerable, vuln_category)
        VALUES (?, ?, ?, ?, ?)
    ''', (request, response, reasoning, is_vulnerable, vuln_id))

    conn.commit()


def submit_prompt(prompt):
    global total_in_tokens, total_out_tokens
    logger.info("------LLM CALL-------")
    logger.info(f"prompt:\n{prompt}")
    completion = client.chat.completions.create(
        model=os.getenv("LLM_API_MODEL"),
        messages=[
            {"role": "system", "content": "You are a machine learning engineer and your goal is to produce high quality training data."},
            {"role": "user", "content": prompt}
        ]
    )
    logger.info(f"response:\n{completion.choices[0].message.content}")
    if hasattr(completion, "usage"):
        logger.info(f"Tokens used: prompt={completion.usage.prompt_tokens}, "
                    f"completion={completion.usage.completion_tokens}, "
                    f"total={completion.usage.total_tokens}")
        total_in_tokens += completion.usage.prompt_tokens
        total_out_tokens += completion.usage.completion_tokens
    return completion.choices[0].message.content

def get_vulnerability_name():
    for vuln in VULNERABILITIES:
        if vuln[1] == args.vuln:
            return vuln[2]

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
    with open("prompts/data_generation_vuln_clone.txt", "r", encoding="utf-8") as f: prompt_tmp_vuln = f.read()
    with open("prompts/data_generation_qa.txt", "r", encoding="utf-8") as f: prompt_tmp_qa = f.read()
    with open("prompts/data_generation_headers_reasoning.txt", "r", encoding="utf-8") as f: prompt_tmp_reasoning = f.read()
    vulnerability = get_vulnerability_name()
    for i in range(args.num):
        # Randomly choose template
        chosen = random.choice(templates)
        # Enhance request
        prompt_req_enhance = prompt_tmp_req_enhance.replace("{request}",f"{chosen['request']}")
        request = submit_prompt(prompt_req_enhance)
        # Enhance response
        prompt_res_enhance = prompt_tmp_res_enhance.replace("{request}", request).replace("{response}",f"{chosen['response']}")
        response = submit_prompt(prompt_res_enhance)
        # Pass through QA Agent
        prompt_res_qa = prompt_tmp_qa.replace("{request}", request).replace("{response}", response)
        vuln_pair = submit_prompt(prompt_res_qa)
        request = vuln_pair.split("<request>")[1].split("</request")[0]
        response = vuln_pair.split("<response>")[1].split("</response")[0]
        # Create a vulnerable clone
        prompt_res_vuln = prompt_tmp_vuln.replace("{request}", request).replace("{response}", response).replace("{vulnerability}", vulnerability)
        vuln_pair = submit_prompt(prompt_res_vuln)
        vuln_request = vuln_pair.split("<request>")[1].split("</request")[0]
        vuln_response = vuln_pair.split("<response>")[1].split("</response")[0]
        # Create a reasoning / analysis
        reasoning = submit_prompt(prompt_tmp_reasoning.replace("{request}", request).replace("{response}", response))
        reasoning_vuln = submit_prompt(prompt_tmp_reasoning.replace("{request}", vuln_request).replace("{response}", vuln_response))
        # Store data in DB
        store_data(request.strip(), response.strip(), False, reasoning.strip())
        store_data(vuln_request.strip(), vuln_response.strip(), True, reasoning_vuln.strip())

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

print("Total in tokens:",total_in_tokens)
print("Total out tokens:",total_out_tokens)