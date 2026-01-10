etcd_client_port=${1:?}; shift

/export/home/tangzihan.15/etcd/etcd \
    --listen-client-urls "http://127.0.0.1:$etcd_client_port" \
    --listen-peer-urls "http://127.0.0.1:$((etcd_client_port + 1))"  \
    --advertise-client-urls  "http://127.0.0.1:$((etcd_client_port + 2))"