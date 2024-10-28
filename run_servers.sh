# # vllm Llama 3.1 8B
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model /local/public/qhhu/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 --disable-log-requests  --max_model_len 550000 --port 8081 --gpu_memory_utilization 0.99

python3 -m sglang.bench_serving --backend vllm --dataset-name random --random-input-len 16000 --random-output-len 128 --random-range-ratio 1 --num-prompts 1 --port 8081

# --num-scheduler-steps 10
# VLLM_ATTENTION_BACKEND=FLASHINFER
# nsys profile --stats=true 


## Sglang

/local/model/llama3.1/Llama-3.1-8B-Instruct
/local/public/qhhu/Meta-Llama-3.1-8B-Instruct-quantized.w8a8

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path /local/public/qhhu/Meta-Llama-3.1-8B-Instruct-quantized.w8a8  --disable-radix-cache --torchao-config int8dq

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input-len 128000 --random-output-len 128 --random-range-ratio 1 --num-prompts 1


# --enable-torch-compile

# --chunked-prefill-size 4096

============

#!/bin/bash
#SBATCH -J sglang
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

source ~/.bashrc
. ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang
which python

cd ~/workdir/sglang



# # SGlang Llama 3.1 8B Instruct on 1 x H100 
# python -m sglang.launch_server --model-path /local/model/llama3.1/Meta-Llama-3.1-8B-Instruct --enable-torch-compile --disable-radix-cache & 

# python -m sglang.launch_server --model-path /local/model/llama3.1/Llama-3.1-70B-Instruct --disable-radix-cache --tp 4 --mem-frac 0.88 & 

# sleep 300

# Online
# python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1200 --request-rate 4

# Offline
# python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 5000

# wait






# sleep 60

# # Online
# python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --dataset-path /nobackup/qinghao/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1200 --request-rate 4


# # Offline
# python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --dataset-path /nobackup/qinghao/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 5000


# wait

# srun -J sglang -N 1 --gpus-per-node 8 --exclusive bash run.sh