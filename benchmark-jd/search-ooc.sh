xservice_port=${1:?}; shift

ratios=(11 13 15)
all_qps=(0.05 0.4 0.5)
# ratios=(11 13 15)
# all_qps=(0.1 0.2 0.3)
# ratios=(6 8 10)
# all_qps=(0.1 0.2 0.3)
# ratios=(6 8 10)
# all_qps=(0.1 0.2 0.3)
# ratios=(5 7 9)
# all_qps=(0.1 0.2 0.3)
echo "Sampling ratios: ${ratios[@]}"

for qps in ${all_qps[@]}; do
    for ratio in "${ratios[@]}"; do
        echo "Sampling ratio: $ratio; Offline QPS: $qps"
        ./benchmark-jd/ooc-run.sh $xservice_port $ratio $qps
    done
done