# JanusLM
Reasoning Classification Dual-Head Transformer with LoRA-Based Fine-Tuning for Web Security assesments.

## Description

This project proposes and evalutes a Dual-Head Large Language Model (LLM) Architecture with Parameter-Efficient Fine-Tuning (LoRA - Low-Rank Adaptation) designed for joint generative reasoning and cyber-security classification tasks.

By using LoRA (Low-Rank Adaptation), the model can efficiently adapt to perform security analysis tasks without changing the original base weights, keeping the reasoning power of the pretrained LLM minimally impacted while adding security domain specific intelligence.

Dual-head design provides a multi-purpose inference, where the pre-trained and fine-tuned generative head creates detailed analysis and the manually trained classification head delivers prediction.

## Goal

Locally served customized LLM that would be capable of analysing HTTP request / response pair and clasify it from the cyber-security perspective.

## Architecture

### Proposal 1: Classification based on Decoder's hidden state
![JanusLM Architecture](imgs/architecture.png)

### Proposal 2: Classification based on embedings
![JanusLM Architecture](imgs/architecture2.png)

## Key Features

- This framework can be applied on any model.
- PEFT (LoRA) for analysis process on request / response pairs.
- Posibility to swap LoRA matrices for different analysis types.
- Fully trained Classification head (Multi-Layer Perceptron) for classification tasks.
- Designed to be locally hosted on user device.

## Training

*Note: Final Training data has been omited from the repository.*

Training data has been prepared by manually crafting pairs of insecure HTTP responses (misconfigured HTTP headers) and their clones with properly configures headers.

Data has been prepared in following format:
```json
  {
    "prompt": "### Instruction:\n \n Evaluate HTTP response headers in a single paragraph and...\n \n <request>RAW Request</request>\n<response>RAW Response</response>",
    "reasoning": "The HTTP response for **myawesome.shop** lacks several key ... ",
    "classification": 1,
    "gpt5_classification": 0, // used for benchmarking
    "gpt5_classification_prompt_engineering": 1, // used for benchmarking
  },
```

*Note: Section to be added on syntetic data generation using larger models and multiple agents*

### 1. Phase:
LoRA fine-tuning for improved analysis reasoning on request / response pair. This was performed via pre-defined high-quality `x` examples of what kind of analysis/reasoning should be performed and what should be considered during evaluation.

*Note: following training stats were collected on smaller model ollama-3.1-1B, with small subset of training data. This will be updated later...*
Evolution of loss function during fine-tuning of LoRA adapter for analysing HTTP headers in relation to the content of the page:
![FineTuning-Training-Loss](imgs/fine-tuning-training-loss.png)

### 2. Phase:
*Still in design process...*

Full MLP (Multi-layered perceptron) training for data classification performed on the output of the last hidden state. (probably mean of all outputs)

## Evaluation

### Metrics
Definition of terms:
- $TP$ - True Positive, correctly marked finding.
- $FP$ - False Positive, incorrectly marked finding.
- $TN$ - True Negative, correctly marked input as safe.
- $FN$ - False Negative, missed finding.

#### Precision

Calculation of how many marked findings were actually correct:

$$Precision = \frac{TP}{TP+FP}$$

#### Accuracy

Overall correctness:

$$Accuracy = \frac{TP+TN}{TP+FP+TN+FN}$$

### Results
To be done:

1. Comparison of proposal 1 and proposal 2 vs baseline (LLM that is doing natively classification)
2. Evaluation how big impact does fine-tuning has
3. Evaluation of different models on accuracy.
4. Evaluation of different size/shape of classification head on accuracy.
5. Evaluation of reasoning length on accuracy.
6. Evaluation of fine-tuning approaches / effect of sample size on accuracy.

## Pre-Requisities

- llama-cpp-python:
    pre-built wheel with basic CPU support:
    ```bash
    pip install -r requirements
    ```
- default model `llama-2-13b`

## Getting Started

1. Create `.env` file with following:
```python
MODEL_PATH="<path_to_llama-3.2-model>"
```

## Execution

```bash
...to be done...
```

