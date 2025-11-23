# ------------------------------------------------------------------------ #
# Lora benchmarks
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-1.7B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-1.7B --label Qwen3-1.7B-lora-0 --max-tokens 0 
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-1.7B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-1.7B --label Qwen3-1.7B-lora-100 --max-tokens 100 
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-1.7B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-1.7B --label Qwen3-1.7B-lora-300 --max-tokens 300 
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-1.7B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-1.7B --label Qwen3-1.7B-lora-500 --max-tokens 500

python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-4B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-4B --label Qwen3-4B-lora-0 --max-tokens 0 
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-4B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-4B --label Qwen3-4B-lora-100 --max-tokens 100 
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-4B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-4B --label Qwen3-4B-lora-300 --max-tokens 300 
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-4B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-4B --label Qwen3-4B-lora-500 --max-tokens 500

python src\benchmark.py --cpu --verbose --lora-adapter DeepSeek-R1-Distrill-Qwen-7B-lora --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-lora-0 --max-tokens 0 
python src\benchmark.py --cpu --verbose --lora-adapter DeepSeek-R1-Distrill-Qwen-7B-lora --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-lora-100 --max-tokens 100 
python src\benchmark.py --cpu --verbose --lora-adapter DeepSeek-R1-Distrill-Qwen-7B-lora --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-lora-300 --max-tokens 300 
python src\benchmark.py --cpu --verbose --lora-adapter DeepSeek-R1-Distrill-Qwen-7B-lora --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-lora-500 --max-tokens 500
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
# External benchmarks
python src\external_benchmark.py --verbose --vuln HTTP_HEADERS --label gpt-5-mini-w0 --max-tokens 0
python src\external_benchmark.py --verbose --vuln HTTP_HEADERS --label gpt-5-mini-w100 --max-tokens 100
python src\external_benchmark.py --verbose --vuln HTTP_HEADERS --label gpt-5-mini-w300 --max-tokens 300
python src\external_benchmark.py --verbose --vuln HTTP_HEADERS --label gpt-5-mini-w500 --max-tokens 500

python src\external_benchmark.py --verbose --vuln HTTP_HEADERS --label deepseek-chat-w0 --max-tokens 0
python src\external_benchmark.py --verbose --vuln HTTP_HEADERS --label deepseek-chat-w100 --max-tokens 100
python src\external_benchmark.py --verbose --vuln HTTP_HEADERS --label deepseek-chat-w300 --max-tokens 300
python src\external_benchmark.py --verbose --vuln HTTP_HEADERS --label deepseek-chat-w500 --max-tokens 500
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
# Baseline benchmarks
python src\benchmark.py --cpu --verbose --vuln HTTP_HEADERS --model-path E:\models\Qwen3-1.7B --label Qwen3-1.7B-0 --max-tokens 0 
python src\benchmark.py --cpu --verbose --vuln HTTP_HEADERS --model-path E:\models\Qwen3-4B --label Qwen3-4B-0 --max-tokens 0 
python src\benchmark.py --cpu --verbose --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-0 --max-tokens 0 
python src\benchmark.py --cpu --verbose --vuln HTTP_HEADERS --model-path E:\models\Qwen3-1.7B --label Qwen3-1.7B-500 --max-tokens 500
python src\benchmark.py --cpu --verbose --vuln HTTP_HEADERS --model-path E:\models\Qwen3-4B --label Qwen3-4B-500 --max-tokens 500 
python src\benchmark.py --cpu --verbose --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-500 --max-tokens 500
# ------------------------------------------------------------------------ #

# TO-DO
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-4B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-4B --label Qwen3-4B-lora-300 --max-tokens 300
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-4B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-4B --label Qwen3-4B-lora-500 --max-tokens 500
python src\benchmark.py --cpu --verbose --lora-adapter Qwen3-1.7B-lora --vuln HTTP_HEADERS --model-path E:\models\Qwen3-1.7B --label Qwen3-1.7B-lora-500 --max-tokens 500
python src\benchmark.py --cpu --verbose --lora-adapter DeepSeek-R1-Distrill-Qwen-7B-lora --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-lora-0 --max-tokens 0 
python src\benchmark.py --cpu --verbose --lora-adapter DeepSeek-R1-Distrill-Qwen-7B-lora --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-lora-100 --max-tokens 100 
python src\benchmark.py --cpu --verbose --lora-adapter DeepSeek-R1-Distrill-Qwen-7B-lora --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-lora-300 --max-tokens 300 
python src\benchmark.py --cpu --verbose --lora-adapter DeepSeek-R1-Distrill-Qwen-7B-lora --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-lora-500 --max-tokens 500
python src\benchmark.py --cpu --verbose --vuln HTTP_HEADERS --model-path E:\models\DeepSeek-R1-Distrill-Qwen-7B --label DeepSeek-R1-Distrill-Qwen-7B-500 --max-tokens 500
