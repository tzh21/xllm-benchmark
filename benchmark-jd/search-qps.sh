version=${1:?}; shift
nodes=${1:?}; shift
dataset=${1:?}; shift
p=${1:?}; shift
d=${1:?}; shift
all_qps=($@)

source "benchmark-jd/common-cleanup.sh"

for qps in ${all_qps[@]}; do
    ./benchmark-jd/run.sh $version $nodes $dataset $qps $p $d
done