#!/bin/bash
# 根据 xllm-service 端口终止整个 cluster

xservice_port=${1:?'Usage: cleanup.sh <xllm_service_http_port>'}

# 从 info 文件中提取 etcd 端口
info_file=$(ls xllm/info/*_srv-${xservice_port}_etcd-* 2>/dev/null | head -1)
if [ -z "$info_file" ]; then
    echo "Warning: No info file found for service port $xservice_port"
else
    etcd_port=$(basename "$info_file" | sed -n 's/.*_etcd-\([0-9]*\)$/\1/p')
    echo "Found cluster: service=$xservice_port, etcd=$etcd_port"
fi

echo "Terminating cluster processes..."

# 终止 xllm-service (通过端口匹配)
pkill -f "http_server_port $xservice_port" 2>/dev/null && echo "Killed xllm-service"

# 终止 xllm (prefill/decode) - 通过 etcd 地址匹配
if [ -n "$etcd_port" ]; then
    pkill -f "etcd_addr 127.0.0.1:$etcd_port" 2>/dev/null && echo "Killed xllm instances"
    
    # 终止 etcd
    pkill -f "listen-client-urls http://127.0.0.1:$etcd_port" 2>/dev/null && echo "Killed etcd"
fi

# 删除 info 文件
if [ -n "$info_file" ] && [ -f "$info_file" ]; then
    rm -f "$info_file"
    echo "Removed info file: $info_file"
fi

echo "Cluster cleanup complete"
