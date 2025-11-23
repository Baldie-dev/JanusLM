import os, sqlite3, torch, logging
from transformers import AutoTokenizer
import pandas as pd

class Utils:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    
    @staticmethod
    def get_vulnerabilities():
        return [
            [1, 'HTTP_HEADERS', 'Misconfigured HTTP Headers', 'headers_self_classification_input'],
            [2, 'XSS', 'Cross-Site Scripting'],
            [3, 'SQLI', "SQL Injection"],
            [4, 'INFO', "Information Disclosure"]
        ]
    
    @staticmethod
    def get_vuln_choices():
        vulns = Utils.get_vulnerabilities()
        res = []
        for vuln in vulns:
            res.append(vuln[1])
        return res
    
    @staticmethod
    def get_vuln_output_prompt(vuln_title):
        for vuln in Utils.get_vulnerabilities():
            if vuln[1] == vuln_title:
                return vuln[3]
    
    @staticmethod
    def get_vuln_id(vuln_title):
        for vuln in Utils.get_vulnerabilities():
            if vuln[1] == vuln_title:
                return vuln[0]
    
    @staticmethod
    def load_prompt(prompt_name, documents):
        with open("prompts/"+prompt_name, "r", encoding="utf-8") as f: template = f.read()
        # Automatically inject documents
        for filename, content in documents.items():
            if filename in template:
                template = template.replace("{"+filename+"}",content)
        return template
    
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
    def load_training_dataset(model_path, vuln, is_cpu, logger=None):
        device = "cpu" if is_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        if logger:
            logger.info("Initializing tokenizer and loading dataset...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
        dataset = Utils.load_data()
        tokenizer.pad_token = tokenizer.eos_token
        with open("prompts/"+Utils.get_vuln_output_prompt(vuln)+".txt", "r", encoding="utf-8") as f: prompt_input_tmp = f.read()
        with open("prompts/self_classification_output.txt", "r", encoding="utf-8") as f: prompt_output_tmp = f.read()
        with open("prompts/output_classification.txt", "r", encoding="utf-8") as f: prompt_output_format = f.read()
        prompt_input_tmp = prompt_input_tmp.replace("{Expected_Output}", prompt_output_format)

        def tokenize_fn(sample):
            input_text = prompt_input_tmp.replace("{request}", sample["request"]).replace("{response}", sample["response"]).strip()
            output_text = prompt_output_tmp.replace("{reasoning}", sample["analysis"]).replace("{result}", "1" if sample["is_vulnerable"] else "0").strip()

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
