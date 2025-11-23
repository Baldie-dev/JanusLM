from torchrl.objectives.llm.grpo import GRPOLoss
from torchrl.modules.llm import LLMWrapperBase
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from tensordict import TensorDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, os, logging, argparse, time
from dotenv import load_dotenv
from torch.distributions import Categorical
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from JanusLModel import JanusClassification
from utils import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", required=False, help="Verbose output during training")
args = parser.parse_args()

model_path = "C:/Users/micha/Downloads/ollama-3.1-1B"
lora_adapter = "test-lora2"
device = "cpu"
batch_size = 1

def PrintSection(name):
    print("\n#"+"-"*15+name+"-"*15+"#")

if args.verbose:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

janus = JanusClassification(model_path, lora_adapter, is_cpu=True)
tasks = Utils.load_data()
documents = Utils.load_documents()
prompt_class_pe = Utils.load_prompt("task_self_class_pe.txt", documents)
prompt_class_pe_class = Utils.load_prompt("task_self_class_pe_class.txt", documents)
prompt_grpo = Utils.load_prompt("grpo_self_reflection.txt", documents)
prompt_grpo_reward = Utils.load_prompt("grpo_reward_function.txt", documents)

# --- LOOP START HERE ---

system = "You are an assistant who generates 1 paragraph long analysis from cyber security perspective of following reflected parameter."
prompt = '''<p>Search results for: <p>Search results for: <span>"><script>alert('document.domain')</script></span></p>'''

# Measure janus.complete
start = time.perf_counter()
resp_complete = janus.complete_template(system, prompt, max_tokens=200)
end = time.perf_counter()
print(f"janus.complete took {end - start:.6f} seconds")
print("Response (complete):", resp_complete)

exit(0)
# 1 Create initial task
task = tasks[0]
is_vuln = int(task["is_vulnerable"])
prompt = prompt_class_pe.replace("{request}",task["request"]).replace("{response}",task["response"])

# 2 Create a reasoning
PrintSection("Reasoning")
analysis = janus.complete(prompt.replace("{analysis}",""), stream=True)
analysis = analysis.split("### Analysis")[1].strip()
print(analysis)

# 3 Perform Assesment
PrintSection("Creating Assesment")
prompt2 = prompt_class_pe_class.replace("{request}",task["request"]).replace("{response}",task["response"])
assesment = janus.complete(prompt2.replace("{analysis}", analysis), 2, stream=True)
assesment = assesment.split("### Result: ")[1].strip()[0]
print(assesment)

# Check if it is number
try:
    assesment = int(assesment)
except Exception:
    assesment = 0

# 4 Evaluate Assesment
print(f"{is_vuln} == {assesment}")

if is_vuln != assesment:
    # Generate analysis why it failed
    data = {
        "prompt": [],
        "completion": []
    }
    self_reflection_prompt = prompt_grpo.replace("{correct}",str(is_vuln)).replace("{incorrect}",str(assesment))
    self_reflection_prompt = self_reflection_prompt.replace("{request}",task["request"]).replace("{response}",task["response"])
    self_reflection_prompt = self_reflection_prompt.replace("{analysis}", analysis)
    self_reflection_prompt = self_reflection_prompt.replace("{result}", str(assesment))
    self_reflection_prompt += "\n\n# Self-Reflection\n"
    for i in range(5):
        PrintSection(f"Self Reflection number {i}")
        self_relection = janus.complete(self_reflection_prompt, 10000, stream=True)
        print(self_relection)
        data["prompt"].append(self_reflection_prompt)
        data["completion"].append(self_relection)
    
out_path = model_path.replace('\\',"/").split("/")[-1]
training_args = GRPOConfig(
    bf16=False,
    output_dir=out_path+"-GRPO",
    per_device_train_batch_size=batch_size,
    num_generations=batch_size,
    gradient_checkpointing=True
)

dataset = Dataset.from_dict(data)

# Reward classification
def reward_based_on_class(completions, **kwargs):
    global prompt_grpo
    # Perform classification
    return [len(set(c)) for c in completions]

prompt_grpo = prompt_grpo_reward

trainer = GRPOTrainer(
    model=janus.model, 
    args=training_args, 
    reward_funcs=reward_based_on_class, 
    use_cpu=True,
    bf16=False,
    train_dataset=dataset)

train_result = trainer.train()
print(train_result)