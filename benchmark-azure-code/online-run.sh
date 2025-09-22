#!/bin/bash

CURRENT_TIME=$(date +"%m%d-%H%M%S")
echo "Running benchmark at ${CURRENT_TIME}"

xservice_port=${1:?}
sampling_ratio=${2:?}

base_dir=benchmark-azure-code
mkdir -p $base_dir/log/runtime
mkdir -p $base_dir/log/result
RUNTIME_LOG="$base_dir/log/runtime/runtime-${CURRENT_TIME}-sr-${sampling_ratio}.log"
RESULT_FILE="$base_dir/log/result/result-${CURRENT_TIME}-sr-${sampling_ratio}.json"

# Preheat
./simple-online/run.sh $xservice_port

# Run benchmark using utils/benchmark.py
python utils/benchmark.py \
    --base-url http://127.0.0.1:$xservice_port \
    --dataset-name trace \
    --dataset-path /export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --trace-path /export/home/tangzihan/xllm-base/datasets/online-datasets/AzureLLMInferenceTrace_code_extracted_136800000_140400000.jsonl \
    --model /export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct \
    --trace-start-time 136800000 \
    --trace-end-time 140400000 \
    --sampling-ratio $sampling_ratio \
    --slo-ttft 5000 \
    --slo-tpot 40 \
    --output-file $RESULT_FILE \
    | tee "${RUNTIME_LOG}"

echo 'Benchmark finished'