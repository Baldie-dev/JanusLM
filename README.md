# JanusLM
Reasoning Classification Dual-Head Transformer with LoRA-Based Fine-Tuning for Web Security assesments.

## Description

This project proposes and evalutes a Dual-Head Large Language Model (LLM) Architecture with Parameter-Efficient Fine-Tuning (LoRA - Low-Rank Adaptation) designed for joint generative reasoning and cyber-security classification tasks.

By using LoRA (Low-Rank Adaptation), the model can efficiently adapt to security assesment tasks without changing the original base weights, keeping the reasoning power of the pretrained LLM minimally impacted while adding security domain specific intelligence.

Dual-head design provides a multi-purpose inference, where the pre-trained and fine-tuned generative head creates detailed reasoning and the manually trained classification head delivers predictions.

## Goal

Locally served LLM that would be capable of analysing HTTP request / response pair and clasify it from the cyber-security perspective.

## Architecture

![JanusLM Architecture](imgs/architecture.png)


## Key Features

- Based on llama-cpp-python (initial model `Llama-2 13B`)
- PEFT (LoRA) based on classified request/response pairs.
- 2 heads - reasoning and classification head
- Locally hosted via OpenAI API

## Training Data

## Evaluation

## Pre-Requisities

- llama-cpp-python:
    pre-built wheel with basic CPU support:
    ```bash
    pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    ```
- default model `llama-2-13b`

## Execution

```bash
...
```

