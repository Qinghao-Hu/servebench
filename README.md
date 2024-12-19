## Installation

```bash
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/
```


## Run the servers

```bash
# Llama 3.1 8B Instruct on single GPU
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile --disable-radix-cache
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --num-scheduler-steps 10 --max_model_len 4096
```

## 1. Online benchmarks

```bash
# bench serving
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 1200 --request-rate 4
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 2400 --request-rate 8
python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompts 1200 --request-rate 4
python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompts 2400 --request-rate 8
```

## 2. Offline benchmarks

```bash
# bench serving
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 5000
python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompts 5000
```
