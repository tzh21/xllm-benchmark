#!/bin/bash

set -e

current_time=$(date +"%m%d-%H%M%S")
echo "Running benchmark at ${current_time}"

version=${1:?}; shift
nodes=${1:?}; shift
dataset=${1:?}; shift
qps=${1:?}; shift
xservice_port=${1:?}; shift

# Preheat
./simple-online/run.sh $nodes $xservice_port

trace_options=()
const_options=()

if [ $version == "v0" ]; then
    echo "Treating offline requests as online"
elif [[ $version == "v1" || $version == "v2" ]]; then
    const_options+=(--offline)
else
    echo "Unknown version"; exit 1
fi

if [ $nodes == "s" ]; then
    model_path=/export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct
    slo_ttft=5000
    slo_tpot=40
elif [ $nodes == "m" ]; then
    model_path=/export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-72B-Instruct
    slo_ttft=10000
    slo_tpot=80
else
    echo "Unknown nodes"; exit 1
fi

if [ $dataset == "jd" ]; then
    if [ $nodes == "s" ]; then
        sampling_ratio=10
    elif [ $nodes == "m" ]; then
        sampling_ratio=3
    fi
    trace_options+=(
        --trace-path /export/home/tangzihan/xllm-base/datasets/online-datasets/jd-online.jsonl
        --trace-start-time 50400000
        --trace-end-time 52200000
        --sampling-ratio $sampling_ratio
    )
elif [ $dataset == "code" ]; then
    sampling_ratio=0.25
    trace_options+=(
        --trace-path /export/home/tangzihan/xllm-base/datasets/online-datasets/AzureLLMInferenceTrace_code_extracted_136800000_140400000.jsonl
        --trace-start-time 136800000
        --trace-end-time 138600000
        --sampling-ratio $sampling_ratio
    )
elif [ $dataset == "conv" ]; then
    sampling_ratio=0.05
    trace_options+=(
        --trace-path /export/home/tangzihan/xllm-base/datasets/online-datasets/AzureLLMInferenceTrace_conv_extracted_144000000_147600000.jsonl
        --trace-start-time 144000000
        --trace-end-time 145800000
        --sampling-ratio $sampling_ratio
    )
else
    echo "Unknown dataset"; exit 1
fi
const_options+=(
    --trace-path /export/home/tangzihan/xllm-base/datasets/offline-datasets/jd-offline.jsonl
    --trace-start-time 50400000 
    --constant-rate $qps 
    --constant-duration 1800 
)

result_dir="./benchmark-jd/$dataset/log/result/$nodes-nodes"; mkdir -p $result_dir
runtime_dir="./benchmark-jd/$dataset/log/runtime/$nodes-nodes"; mkdir -p $runtime_dir
log_base=${current_time}-sr-${sampling_ratio}-qps-${qps}-$version

python utils/benchmark.py \
    --base-url http://127.0.0.1:$xservice_port \
    --traffic-mode trace \
    --prompt-path /export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model $model_path \
    --slo-ttft $slo_ttft \
    --slo-tpot $slo_tpot \
    --output-file "$result_dir/$log_base-online.json" \
    "${trace_options[@]}" \
    | tee "$runtime_dir/$log_base-online.log" &

python utils/benchmark.py \
    --base-url http://127.0.0.1:$xservice_port \
    --traffic-mode constant \
    --prompt-path /export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model $model_path \
    --slo-ttft 100000000 \
    --slo-tpot 100000000 \
    --output-file "$result_dir/$log_base-offline.json" \
    "${const_options[@]}" \
    | tee "$runtime_dir/$log_base-offline.log" &

wait

echo 'Benchmark finished'