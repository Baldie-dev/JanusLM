import json, torch, argparse, logging, sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
from peft import PeftModel
from utils import Utils

#class JanusSequenceClassification(nn.Module):
    #def __init__(self, model, num_labels):
    #    super().__init__()
    #    self.base = model
    #    self.classifier = nn.Linear(model.config.hidden_size, num_labels)
    #
    ## Definition of classification head
    #def forward(self, input_ids, attention_mask=None, labels=None):
    #    # Get the output of last hidden layer
    #    outputs = self.base.model(
    #        input_ids=input_ids,
    #        attention_mask=attention_mask,
    #        output_hidden_states=True
    #    )
    #    last_hidden = outputs.hidden_states[-1]
    #    
    #    # First approach would be to take a mean
    #    # But I would like to add additinal trained dense network here
    #    pooled = last_hidden.mean(dim=1)
    #    logits = self.classifier(pooled)
    #
    #    # Loss function via simple softmax
    #    loss = None
    #    if labels is not None:
    #        loss_fct = nn.CrossEntropyLoss()
    #        loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
    #
    #    return {"loss": loss, "logits": logits}

class JanusClassification():
    def __init__(self, model_path, lora_adapter, is_cpu=True):
        self.is_cpu = is_cpu
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if lora_adapter:
            self.model = PeftModel.from_pretrained(self.model, lora_adapter)
        self.model.eval()
        self.model.config.is_decoder = True
        self.model.config.is_encoder_decoder = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt, max_tokens=100, temperature=1):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.is_cpu:
            inputs = inputs.to("cpu")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response