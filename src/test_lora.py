from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForCausalLM
from dotenv import load_dotenv
import os, torch, logging, datasets, argparse
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW 
import csv
from utils import Utils
from transformers import AutoTokenizer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = "C:/Users/micha/Downloads/ollama-3.1-1B"
model_path = "E:/models/Qwen3-4B"
lora_adapter = "lora_adapter_direct_class"
device = "cpu"
batch_size = 1
threads = 1

tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
base_model.to("cpu")
torch.set_num_threads(int(threads))
if int(threads) > 2:
    torch.set_num_interop_threads(int(int(threads)/2))
if hasattr(base_model.config, "use_cache"):
    base_model.config.use_cache = False

lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

model = get_peft_model(base_model, lora_config)
model.to("cpu")
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None)

model.train()
optimizer = AdamW(model.parameters(), lr=2e-4)

max_trainning_steps = 10
training_args = TrainingArguments(
            output_dir="./lora-out",
            per_device_train_batch_size=1,
            max_steps=max_trainning_steps,
            learning_rate=2e-4,
            logging_steps=1,
            report_to=None,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            use_cpu=True,
            save_strategy="no",
            logging_strategy="steps",
        )

print("Tokenizing dataset...")
train_dataset = Utils.tokenize_datasets_lora(tokenizer, ["Hi, how are you?"], ["I am fine."])

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

print("Starting Training...")
trainer.train()
print("Training completed!")
log_history = trainer.state.log_history
model.save_pretrained("test-lora2")