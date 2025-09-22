#!/bin/bash

set -e

current_time=$(date +"%m%d-%H%M%S")
echo "Running benchmark at ${current_time}"

xservice_port=${1:?}; shift
sampling_ratio=${1:?}; shift
qps=${1:?}; shift

base_dir=benchmark-jd
mkdir -p $base_dir/log/runtime
mkdir -p $base_dir/log/result

# Preheat
./simple-online/run.sh $xservice_port

log_base=${current_time}-sr-${sampling_ratio}-qps-${qps}

python utils/benchmark.py \
    --base-url http://127.0.0.1:$xservice_port \
    --traffic-mode trace \
    --prompt-path /export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --trace-path /export/home/tangzihan/xllm-base/datasets/online-datasets/jd-online.jsonl \
    --model /export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct \
    --trace-start-time 50400000 \
    --trace-end-time 54000000 \
    --sampling-ratio $sampling_ratio \
    --slo-ttft 5000 \
    --slo-tpot 40 \
    --output-file "$base_dir/log/result/$log_base-online.json" \
    | tee "$base_dir/log/runtime/$log_base-online.log" &

python utils/benchmark.py \
    --base-url http://127.0.0.1:$xservice_port \
    --traffic-mode constant \
    --prompt-path /export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --trace-path /export/home/tangzihan/xllm-base/datasets/offline-datasets/jd-offline.jsonl \
    --model /export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct \
    --trace-start-time 50400000 \
    --trace-end-time 54000000 \
    --slo-ttft 100000000 \
    --slo-tpot 100000000 \
    --offline \
    --constant-rate $qps \
    --constant-duration 3600 \
    --output-file "$base_dir/log/result/$log_base-offline.json" \
    | tee "$base_dir/log/runtime/$log_base-offline.log" &

wait

echo 'Benchmark finished'