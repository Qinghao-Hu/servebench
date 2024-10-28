#!/bin/bash

LOG_DIR="logs"
OUTPUT_FILE="benchmark_results.csv"
# expriement_name="sglang_Llama-3.1-8B"
expriement_name="vllm_Llama-3.1-8B-int8"

# 写入CSV头
echo "input_size,mean_ttft_ms,mean_itl_ms" > $OUTPUT_FILE

for size in 4 8 16 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512; do
    value=$((size * 1000))
    log_file="${LOG_DIR}/${expriement_name}_${value}.log"
    
    if [ -f "$log_file" ]; then
        mean_ttft=$(grep "Mean TTFT (ms):" "$log_file" | awk '{print $4}')
        mean_itl=$(grep "Mean ITL (ms):" "$log_file" | awk '{print $4}')
        
        echo "${value},${mean_ttft},${mean_itl}" >> $OUTPUT_FILE
    fi
done

echo "Results extracted to $OUTPUT_FILE"