#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

model="Llama-3.1-8B-int8"
framework="vllm"
# INPUT_SIZES=(4 8 16 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512)
INPUT_SIZES=(416 448 480 512)

for size in "${INPUT_SIZES[@]}"; do

    value=$((size * 1000))
    
    echo "Testing with input length: ${size}K"
    
    log_file="${LOG_DIR}/${framework}_${model}_${value}.log"
    
    python3 -m sglang.bench_serving --backend vllm --dataset-name random --random-input-len $value --random-output-len 128 --random-range-ratio 1 --num-prompts 1 --port 8081 2>&1 | tee "$log_file"
    
    echo "Results saved to: $log_file"
    echo "----------------------------------------"
done