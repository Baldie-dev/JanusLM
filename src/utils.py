import os, sqlite3, torch, logging
from transformers import AutoTokenizer
import pandas as pd

class Utils:
    @staticmethod
    def load_documents():
        folder_path = 'documents'
        text_data = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_data[filename.replace(".txt","")] = file.read()
        return text_data
    
    @staticmethod
    def load_data():
        conn = sqlite3.connect('datasets/data.db')
        df = pd.read_sql_query("SELECT * FROM training_data", conn)
        from datasets import Dataset
        dataset = Dataset.from_pandas(df)
        return dataset
    
    @staticmethod
    def load_dataset(model_path, is_cpu, logger=None):
        if logger:
            logger.info("Initializing tokenizer and loading dataset...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, is_cpu)
        if is_cpu:
            torch.set_num_threads(1)
        dataset = Utils.load_data()
        tokenizer.pad_token = tokenizer.eos_token
        with open("prompts/self_classification_input.txt", "r", encoding="utf-8") as f: prompt_input_tmp = f.read()
        with open("prompts/self_classification_output.txt", "r", encoding="utf-8") as f: prompt_output_tmp = f.read()

        def tokenize_fn(sample):
            input_text = prompt_input_tmp.replace("{request}", sample["request"]).replace("{response}", sample["response"]).strip()
            output_text = prompt_output_tmp.replace("{reasoning}", sample["analysis"]).replace("{result}", "1" if sample["is_vulnerable"] else "0").strip()
            full_text = input_text + output_text

            input_tokens = tokenizer(input_text, truncation=True, max_length=1024, add_special_tokens=False)
            output_tokens = tokenizer(output_text, truncation=True, max_length=1024, add_special_tokens=False)
            input_ids = input_tokens["input_ids"] + output_tokens["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(input_tokens["input_ids"]) + output_tokens["input_ids"]
            max_len = 2048
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids += [tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len
            else:
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        if logger:
            logger.info("Tokenizing dataset...")
        if is_cpu:
            tokenized = dataset.map(
                tokenize_fn, 
                batched=False, 
                remove_columns=dataset.column_names,
                num_proc=None)
        else:
            tokenized = dataset.map(
                tokenize_fn,
                remove_columns=dataset.column_names
                )
        return tokenizer, tokenized
