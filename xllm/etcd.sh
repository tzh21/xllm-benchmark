etcd_client_port=${1:?}; shift

source xllm/utils.sh

$ETCD_BIN \
    --listen-client-urls "http://127.0.0.1:$etcd_client_port" \
    --listen-peer-urls "http://127.0.0.1:$((etcd_client_port + 1))"  \
    --advertise-client-urls  "http://127.0.0.1:$((etcd_client_port + 2))" \
    > $ETCD_LOG/$(date +%m%d-%H%M%S).log
