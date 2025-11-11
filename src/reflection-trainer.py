from torchrl.objectives.llm.grpo import GRPOLoss
from torchrl.modules.llm import LLMWrapperBase
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from tensordict import TensorDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, os
from dotenv import load_dotenv
from torch.distributions import Categorical
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from JanusLModel import JanusClassification
from utils import Utils

model_path = "C:/Users/micha/Downloads/ollama-3.1-1B"
lora_adapter = "lora_adapter_direct_class"
device = "cpu"
batch_size = 1

janus = JanusClassification(model_path, lora_adapter, is_cpu=True)
tasks = Utils.load_data()
documents = Utils.load_documents()
prompt_class_pe = Utils.load_prompt("task_self_class_pe.txt", documents)
prompt_class_pe_class = Utils.load_prompt("task_self_class_pe_class.txt", documents)

# 1 Create initial task
task = tasks[0]
is_vuln = task["is_vulnerable"]
prompt = prompt_class_pe.replace("{request}",task["request"]).replace("{response}",task["response"])

# 2 Create a reasoning
analysis = janus.complete(prompt.replace("{analysis}",""))
print(analysis)
analysis = analysis.split("### Analysis")[1].strip()

# 3 Perform Assesment
prompt2 = prompt_class_pe_class.replace("{request}",task["request"]).replace("{response}",task["response"])
assesment = janus.complete(prompt2.replace("{analysis}", analysis), 2)
assesment = assesment.split("### Result: ")[1].strip()
print(assesment)

# 4 Evaluate Assesment
print("-----")
print(f"{is_vuln} == {assesment}")


exit(1)

# Step 1, prepare the group for training

out_path = model_path.replace('\\',"/").split("/")[-1]
training_args = GRPOConfig(
    bf16=False,
    output_dir=out_path+"-GRPO",
    per_device_train_batch_size=batch_size,
    num_generations=batch_size,
    gradient_checkpointing=True
)

data = {
    "prompt": ["What is the summary of this article?", "What is the summary of this article?"],
    "completion": [
        "This article discusses the impact of climate change on Arctic wildlife.",
        "The article explores how rising temperatures affect Arctic ecosystems and animal populations."
    ]
}

dataset = Dataset.from_dict(data)

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

trainer = GRPOTrainer(
    model=model, 
    args=training_args, 
    reward_funcs=reward_num_unique_chars, 
    use_cpu=True,
    bf16=False,
    train_dataset=dataset)
trainer.train()