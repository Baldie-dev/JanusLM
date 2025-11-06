# JanusLM
Reasoning Classification Dual-Head Transformer with LoRA-Based Fine-Tuning for a Web Security false-positive evaluations.

## Description

This project proposes and evaluates a Dual-Head Large Language Model (LLM) architecture utilizing Parameter-Efficient Fine-Tuning (Low-Rank Adaptation), specifically designed for joint generative reasoning and cybersecurity-related classification tasks

By leveraging multiple tailored LoRA adapters, the model can efficiently adapt to perform security analysis tasks without changing the original base weights, keeping the reasoning power of the pretrained LLM minimally impacted while adding security domain specific intelligence.

Dual-head design provides a multi-purpose inference, where the pre-trained and fine-tuned generative head creates detailed analysis and the trained classification head consisting of dense neural network delivers prediction.

## Goal

The primary goal is to explore the possibility of using a small, locally hosted, tailored model that delivers performance comparable to the largest commercial models. The second goal focuses on generating synthetic, high-quality training data and evaluating various approaches to training, fine-tuning, base model selection, and classification methods

## Architecture

### Proposal 1: Classification based on Decoder's hidden state
![JanusLM Architecture](imgs/architecture.png)

### Proposal 2: Classification based on embedings
![JanusLM Architecture](imgs/architecture2.png)

## Key Features

- Compatible with any open-source model.
- PEFT (LoRA) fine-tuned for analysis creation of request/response pairs.
- Supports swapping LoRA matrices to target different analysis types.
- Includes a fully trained classification head (Multi-Layer Perceptron) for false-positive evaluation.
- Optimized for local deployment on the user's device.

## Training

*Note: Training data has been omited from the repository.*

The training data was manually prepared by crafting pairs of insecure HTTP responses featuring misconfigured headers and their counterparts with properly configured headers. This should help model understand the main differences between false and true positive.

Data has been prepared in following format:
```json
  {
    "prompt": "### Instruction:\n \n Evaluate HTTP response headers in a single paragraph and...\n \n <request>RAW Request</request>\n<response>RAW Response</response>",
    "reasoning": "The HTTP response for **myawesome.shop** lacks several key ... ",
    "classification": 1,
    "gpt4_classification": 0, // used for benchmarking
    "gpt4_classification_prompt_engineering": 1, // used for benchmarking
  },
```

### Synthetic Data Generation

*Note: This section is still in design phase...*

High-quality of manually crafted templates of realistic HTTP request/response pairs were passed through multiple agents that used larger models to generate synthetic training data. Each agent performed small mutations to the templates to expand 

![synthentic-data-generation](imgs/synthetic-data-generation.png)

Manual review was required to fine-tune the prompts for each agent.

For training data generation, please see `src/data_generator.py`:
```console
usage: data_generator.py [-h] [--num NUM] --templates TEMPLATES --vuln {HTTP_HEADERS,XSS}

options:
  -h, --help            show this help message and exit
  --num NUM             number of generated request/response pairs
  --templates PATH      templates for request/response pairs.
  --vuln {HTTP_HEADERS,XSS} Select category of vulnerability
```

for example:
```console
python src\data_generator.py --verbose --vuln HTTP_HEADERS --num 5 --instruction "Do not introduce CORS misconfiguration, but introduce HSTS misconfiguration"
```

To generate one pair of high-quality training data, it costs on average:
- Input tokens: ~4000
- Output tokens: ~10000

Current token distribution in training data (*Larger dataset to be prepared*):

![Token-Data-Distribution](imgs/training-data-tokens-distribution.png)

### 1. Phase:
LoRA fine-tuning was applied to enhance analytical reasoning over request/response pairs. This was achieved using predefined, high-quality `x` examples that illustrated the desired type of analysis and the evaluation criteria to be considered.

*Note: following training stats were collected on smaller model ollama-3.1-1B, with small subset of training data. This will be updated later...*

Evolution of the loss function during fine-tuning of a LoRA adapter for creation of analysis for HTTP responses, showing that model is improving ability to perform reasoning.
![FineTuning-Training-Loss](imgs/fine-tuning-training-loss.png)

### 2. Phase:
*Still in design process...*

Full MLP (Multi-layered perceptron) training for data classification performed on the output of the last hidden state. (probably mean of all outputs)

## Evaluation

Evaluation was performed using a cross-validation technique by splitting the training data into two folds, where one was used for training and the other for validation.

### Metrics
Definition of terms:
- $TP$ - True Positive, correctly marked finding.
- $FP$ - False Positive, incorrectly marked finding.
- $TN$ - True Negative, correctly marked HTTP response as secure.
- $FN$ - False Negative, missed finding.

#### Precision

Calculation of how many marked findings were actually correct (how many false-positives were generated):

$$Precision = \frac{TP}{TP+FP}$$

#### Accuracy

Overall correctness:

$$Accuracy = \frac{TP+TN}{TP+FP+TN+FN}$$

### Results

Explanation of model types:
- `<base_model>-SC`: Base model that performs false-positive analysis by generating token (1 or 0).
- `<base_model>-PE-SC`: Base model with prompt engineering that performs false-positive analysis by generating token (1 or 0).
- `<base_model>-FT-SC`: Fine-Tuned model that performs false-positive analysis by generating token (1 or 0).
- `<base_model>-FT-PE-SC`: Fine-Tuned model with prompt engineering that performs false-positive analysis by generating token (1 or 0).
- `<base_model>-FT-CH`: Fine-Tuned model that performs false-positive analysis by leveraging classification head.
- `<base_model>-FT-PE-CH`: Fine-Tuned model with prompt engineering that performs false-positive analysis by leveraging classification head.

#### Baseline vs Proposal 1 and Proposal 2
Comparison of accuracy between different models:
![model-accuracy-benchmark](imgs/ollama-3.1-1B-benchmark.png)

1. Evaluation of different size/shape of classification head on accuracy.
2. Evaluation of reasoning length on accuracy.
3. Evaluation of fine-tuning approaches / effect of sample size on accuracy.

## Pre-Requisities

- llama-cpp-python
- for CPU support: pre-built wheel with basic CPU support
- install all requirements:
    ```bash
    pip install -r requirements
    ```
- download open source base model, for example `llama-2-7b`

## Getting Started

1. Create `.env` file with following:
```python
MODEL_PATH="<path_to_base_model>"
LLM_API_URL=http://127.0.0.1:8000/v1
LLM_API_KEY=dummy_key
```

2. Update `datasets/reasoning.jsonl` with training data for fine-tuning.

3. Start LoRA training first via command:
```console
usage: trainer.py [-h] --mode {lora,class} [--cpu] [--output OUTPUT] [--verbose] [--charts]

options:
  -h, --help           show this help message and exit
  --mode {lora,class}  Triggers training either for LoRA or Classification head
  --cpu                Safe and slow training on CPU, for compatibility reasons
  --output OUTPUT      output folder for trained model
  --verbose            Verbose output during training
  --charts             If sets, training charts are generated.
```
for example:
```console
python src/trainer.py --mode lora --cpu --output lora-adapter --charts
```

4. Start Classification head training with previously trained LoRA adapters:

```console
python src/trainer.py --mode class --cpu --output class-model --input lora-adapter --charts
```

## Execution

To execute benchmarks:
```bash
python src/benchmark.py
```

## Notes to self

- Can I fine-tune / train it purely on what I consider secure/insecure (true/false input) and let it figure it out what does it make secure/insecure?