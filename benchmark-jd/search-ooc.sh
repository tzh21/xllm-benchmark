set -e

trap "echo 'Exiting, killing child processes...'; kill 0" EXIT INT TERM

xservice_port=${1:?}; shift
mode=${1:?}; shift

if [ $mode == "v1" ]; then
    script=./benchmark-jd/v1-run.sh
elif [ $mode == "v2" ]; then
    script=./benchmark-jd/v2-run.sh
else
    echo "Unknown mode $mode"
    exit 1
fi

# ratios=(12)
# all_qps=(0.1 0.15 0.2)
ratios=(12)
all_qps=(0.3 0.4 0.5)
# ratios=(12)
# all_qps=(0.1 0.15 0.2)
# ratios=(12)
# all_qps=(0.1 0.15 0.2)
# ratios=(12)
# all_qps=(0.1 0.15 0.2)
# ratios=(12)
# all_qps=(0.1 0.15 0.2)

for qps in ${all_qps[@]}; do
    for ratio in "${ratios[@]}"; do
        echo "Sampling ratio: $ratio; Offline QPS: $qps"
        $script $xservice_port $ratio $qps
    done
done