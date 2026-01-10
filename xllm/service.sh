etcd_port=${1:?}; shift
http_server_port=${1:?}; shift

ENABLE_DECODE_RESPONSE_TO_SERVICE=false \
/export/home/tangzihan.15/xllm-service/build/xllm_service/xllm_master_serving \
    --etcd_addr "127.0.0.1:$etcd_port" \
    --http_server_port $http_server_port \
    --rpc_server_port $((http_server_port + 1)) \
    --tokenizer_path /export/home/tangzihan.15/models/Qwen2.5-7B-Instruct