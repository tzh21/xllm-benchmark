set -e

trap "echo 'Exiting, killing child processes...'; kill 0" EXIT INT TERM

version=${1:?}; shift
nodes=${1:?}; shift
dataset=${1:?}; shift
xservice_port=${1:?}; shift
all_qps=($@)

# ratios=(10)
# all_qps=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# all_qps=(1.2 1.4)
# all_qps=(1.6 1.8)
# all_qps=(1.8)
# all_qps=(2.0)
# all_qps=(0.5)
# all_qps=(1.0)

for qps in ${all_qps[@]}; do
    ./benchmark-jd/run.sh $version $nodes $dataset $qps $xservice_port
done