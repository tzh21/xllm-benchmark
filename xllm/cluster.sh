prefill_npu=${1:?}; shift
decode_npu=${1:?}; shift

source scripts/utils.sh
etcd_client_port=$(randport)
xllm_service_http_port=$(randport)

bash scripts/etcd.sh $etcd_client_port &
sleep 2
bash scripts/service.sh $etcd_client_port $xllm_service_http_port &
sleep 2
bash scripts/pd-xllm.sh $etcd_client_port p $prefill_npu &
bash scripts/pd-xllm.sh $etcd_client_port d $decode_npu &

mkdir -p scripts/clusters-log
rm scripts/clusters-log/p${prefill_npu}_d${decode_npu}*
touch scripts/clusters-log/p${prefill_npu}_d${decode_npu}_srv-${xllm_service_http_port}_etcd-${etcd_client_port}