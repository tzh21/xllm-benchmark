set -e

# 外部依赖路径配置
XLLM_HOME=/export/home/tangzihan.15/xllm
XLLM_SERVICE_HOME=/export/home/tangzihan.15/xllm-service
ETCD_HOME=/export/home/tangzihan.15/etcd
MODEL_PATH=/export/home/tangzihan.15/models/Qwen2.5-7B-Instruct

# 可执行文件路径
XLLM_BIN=$XLLM_HOME/build/xllm/core/server/xllm
XLLM_SERVICE_BIN=$XLLM_SERVICE_HOME/build/xllm_service/xllm_master_serving
ETCD_BIN=$ETCD_HOME/etcd

# 日志路径
XLLM_LOG=$XLLM_HOME/local/log
XLLM_SERVICE_LOG=$XLLM_SERVICE_HOME/local/log
ETCD_LOG=$ETCD_HOME/log

randport() {
  echo $(( RANDOM % 40001 + 20000 ))
}
