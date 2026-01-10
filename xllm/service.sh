etcd_port=${1:?}; shift
http_server_port=${1:?}; shift

source xllm/utils.sh

ENABLE_DECODE_RESPONSE_TO_SERVICE=false \
$XLLM_SERVICE_BIN \
    --etcd_addr "127.0.0.1:$etcd_port" \
    --http_server_port $http_server_port \
    --rpc_server_port $((http_server_port + 1)) \
    --tokenizer_path $MODEL_PATH \
    > $XLLM_SERVICE_LOG/$(date +%m%d-%H%M%S).log
