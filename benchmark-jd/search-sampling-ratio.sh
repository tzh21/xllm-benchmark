set -e

xservice_port=${1:?}; shift

ratios=("$@")
echo "Sampling ratios: ${ratios[@]}"

for ratio in "${ratios[@]}"; do
    echo "Sampling ratio: $ratio"
    ./benchmark-jd/run.sh $xservice_port $ratio
done
