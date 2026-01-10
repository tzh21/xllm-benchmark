etcd_port=${1:?}; shift
pd=${1:?}; shift
npu=${1:?}; shift

source scripts/utils.sh

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_IF_BASE_PORT=$(randport)  # HCCL 通信基础端口

MODEL_PATH="/export/home/tangzihan.15/models/Qwen2.5-7B-Instruct"               # 模型路径

args=(
    --model $MODEL_PATH
    --port $(randport)
    --devices npu:$npu
    --master_node_addr 127.0.0.1:$(randport)
    --enable_disagg_pd
    --noenable_prefix_cache
    --noenable_chunked_prefill
    --disable_ttft_profiling
    --etcd_addr 127.0.0.1:$etcd_port
    --kv_cache_transfer_type HCCL
    --transfer_listen_port $(randport)
    --disagg_pd_port $(randport)
    --node_rank 0
    --nnodes 1

    --enable_pd_ooc
)

if [ $pd = "p" ]; then
    args+=(
        --instance_role PREFILL
    )
elif [ $pd = "d" ]; then
    args+=(
        --instance_role DECODE
    )
else
    echo "no pd"; exit 1
fi

# v0, v1
    # --enable_latency_aware_schedule
    # --max_global_ttft_ms 5000
    # --max_global_tpot_ms 40

# v2
    # --max_global_tpot_ms 35

/export/home/tangzihan.15/xllm/build/xllm/core/server/xllm "${args[@]}"