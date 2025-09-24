set -e

trap "echo 'Exiting, killing child processes...'; kill 0" EXIT INT TERM

xservice_port=${1:?}; shift
version=${1:?}; shift
all_qps=($@)

if [ $version == "v0" ]; then
    script=./benchmark-jd/v0-run.sh
elif [ $version == "v1" ]; then
    script=./benchmark-jd/v1-run.sh
elif [ $version == "v2" ]; then
    script=./benchmark-jd/v2-run.sh
else
    echo "Unknown mode $version"
    exit 1
fi

ratios=(13)
# all_qps=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# all_qps=(1.2 1.4)
# all_qps=(1.6 1.8)
# all_qps=(1.8)
# all_qps=(2.0)
# all_qps=(0.5)
# all_qps=(1.0)

for qps in ${all_qps[@]}; do
    for ratio in "${ratios[@]}"; do
        echo "Sampling ratio: $ratio; Offline QPS: $qps"
        $script $xservice_port $ratio $qps
    done
done