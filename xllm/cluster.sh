version=${1:?}; shift
prefill_npu=${1:?}; shift
decode_npu=${1:?}; shift

source xllm/utils.sh
etcd_client_port=$(randport)
xllm_service_http_port=$(randport)

bash xllm/etcd.sh $etcd_client_port &
sleep 2
bash xllm/service.sh $etcd_client_port $xllm_service_http_port &
sleep 2
bash xllm/pd-xllm.sh $etcd_client_port $version p $prefill_npu &
bash xllm/pd-xllm.sh $etcd_client_port $version d $decode_npu &

mkdir -p xllm/info
rm -f xllm/info/p-${prefill_npu}_d-${decode_npu}*
touch xllm/info/p-${prefill_npu}_d-${decode_npu}_srv-${xllm_service_http_port}_etcd-${etcd_client_port}
