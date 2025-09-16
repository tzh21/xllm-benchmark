#!/bin/bash

# Azure benchmark script using utils/benchmark.py
# Based on local/archive/xllm-test.sh

CURRENT_TIME=$(date +"%m%d-%H%M%S")
echo "Running benchmark at ${CURRENT_TIME}"
base_dir=benchmark-jd

# Configuration
# num_prompts=19000
mkdir -p $base_dir/log/runtime
mkdir -p $base_dir/log/result
RUNTIME_LOG="$base_dir/log/runtime/runtime-${CURRENT_TIME}.log"
RESULT_FILE="$base_dir/log/result/result-${CURRENT_TIME}.jsonl"

# Run benchmark using utils/benchmark.py
python utils/benchmark.py \
    --base-url http://127.0.0.1:27712 \
    --dataset-name trace \
    --dataset-path /export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --trace-path /export/home/tangzihan/xllm-base/datasets/online-datasets/jd-online.jsonl \
    --model /export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct \
    --trace-start-time 0 \
    --trace-end-time 600 \
    --sampling-ratio 100 \
    --slo-ttft 1000 \
    --slo-tpot 30 \
    --output-file $RESULT_FILE \
    | tee "${RUNTIME_LOG}"

echo 'Benchmark finished'