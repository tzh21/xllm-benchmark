version=${1:?}; shift
nodes=${1:?}; shift
dataset=${1:?}; shift
qps=${1:?}; shift
p=${1:?}; shift
d=${1:?}; shift

source "benchmark-jd/common-cleanup.sh"

current_time=$(date +"%m%d-%H%M%S")
echo "Running benchmark at ${current_time}"

# Extract xservice_port from cluster-info filename: {nodes}-p-{p}-d-{d}-x-{xservice_port}-{timestamp}.log
cluster_info_file=$(ls /export/home/tangzihan/xllm-base/scripts/cluster/info/${nodes}-p-${p}-d-${d}-x-*.log | head -1)
xservice_port=$(basename "$cluster_info_file" | sed -n 's/.*-x-\([0-9]*\)-.*/\1/p')

# Preheat
echo "Running online simple test"
bash ./simple-test/run.sh false $nodes $xservice_port
echo "Running offline simple test"
bash ./simple-test/run.sh true $nodes $xservice_port

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
    sampling_ratio=0.1
    trace_options+=(
        --trace-path /export/home/tangzihan/xllm-base/datasets/online-datasets/AzureLLMInferenceTrace_conv_extracted_144000000_147600000.jsonl
        --trace-start-time 144000000
        --trace-end-time 145800000
        --sampling-ratio $sampling_ratio
        --seed $RANDOM
    )
    const_options+=(
        --seed $RANDOM
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
log_base=${dataset}-${current_time}-sr-${sampling_ratio}-qps-${qps}-$version

# Sending requests based on a trace file
python utils/benchmark.py \
    --base-url http://127.0.0.1:$xservice_port \
    --traffic-mode trace \
    --prompt-path /export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model $model_path \
    --slo-ttft $slo_ttft \
    --slo-tpot $slo_tpot \
    --output-file "$result_dir/$log_base-online.json" \
    "${trace_options[@]}" \
    2>&1 | tee "$runtime_dir/$log_base-online.log" &
online_pid=$(($! - 1))
echo "Online benchmark: $online_pid"

# Sending requests at a const rate
python utils/benchmark.py \
    --base-url http://127.0.0.1:$xservice_port \
    --traffic-mode constant \
    --prompt-path /export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model $model_path \
    --slo-ttft 100000000 \
    --slo-tpot 100000000 \
    --output-file "$result_dir/$log_base-offline.json" \
    "${const_options[@]}" \
    2>&1 | tee "$runtime_dir/$log_base-offline.log" &
offline_pid=$(($! - 1))
echo "Offline benchmark: $offline_pid"

# Wait for online/trace mode to finish
wait $online_pid
echo "Online benchmark finished, interrupting offline benchmark"

# Send SIGINT to offline/const mode
kill -INT $offline_pid

# Wait for offline mode to finish
wait $offline_pid

echo 'Benchmark finished'