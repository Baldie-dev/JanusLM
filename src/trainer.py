from transformers import AutoTokenizer, TrainingArguments, Trainer
from dotenv import load_dotenv
import os
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from JanusLModel import JanusSequenceClassification

load_dotenv()
model_path = os.getenv("MODEL_PATH")

def StartTraining():
    # Prepare the training data
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset("json", data_files="lora_dataset/dataset.jsonl")
    def tokenize_fn(example):
        tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
        tokens["labels"] = example["label"]
        return tokens
    tokenized = dataset.map(tokenize_fn)

    # Load JanusLM model and define training
    model = JanusSequenceClassification(num_labels=2)
    training_args = TrainingArguments(
        output_dir="./classifier_output",
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_total_limit=1
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
    )
    trainer.train()
    model.save_pretrained("./lora_adapter")

def LoadTrained():
    #Loading
    #from peft import PeftModel
    #base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    #model = PeftModel.from_pretrained(base, "./lora_adapter")
    pass