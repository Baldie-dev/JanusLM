from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn

class JanusSequenceClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.base = base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype="auto", device_map="auto")
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get the output of last hidden layer
        outputs = self.base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        
        # First approach would be to take a mean
        # But I would like to add additinal trained dense network here
        pooled = last_hidden.mean(dim=1)
        logits = self.classifier(pooled)

        # Loss function via simple softmax
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {"loss": loss, "logits": logits}
