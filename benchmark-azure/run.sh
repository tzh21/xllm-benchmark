#!/bin/bash

# Azure benchmark script using utils/benchmark.py
# Based on local/archive/xllm-test.sh

CURRENT_TIME=$(date +"%m%d-%H%M%S")
echo "Running Azure benchmark at ${CURRENT_TIME}"

# Configuration
num_prompts=19000
sampling_ratio=0.99
RUNTIME_LOG="benchmark-azure/log/runtime/runtime-${CURRENT_TIME}.log"
RESULT_FILE="benchmark-azure/log/result/result-${CURRENT_TIME}-${num_prompts}-${sampling_ratio}.jsonl"

# Model configuration
model=/export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct

# Dataset paths
dataset_path=/export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json
trace_path=/export/home/tangzihan/xllm-base/datasets/online-datasets/azure_conv.jsonl

# Create log directories if they don't exist
mkdir -p benchmark-azure/log/runtime
mkdir -p benchmark-azure/log/result

# Run benchmark using utils/benchmark.py
python utils/benchmark.py \
    --base-url http://127.0.0.1:27712 \
    --dataset-name azure_conv \
    --dataset-path $dataset_path \
    --trace-path $trace_path \
    --model $model \
    --num-prompts $num_prompts \
    --sampling-ratio $sampling_ratio \
    --trace-scale 1.0 \
    --slo-ttft 1000 \
    --slo-tpot 30 \
    --output-file $RESULT_FILE \
    | tee "${RUNTIME_LOG}"

echo 'Azure benchmark finished'