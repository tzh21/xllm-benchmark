# 1 32
# 1-8
# 9-16
# 17-24
# 25-32

# original_qps=60.0
# start=${1:?}
# end=${2:?}
# step=${3:?}
# xservice_port=${4:?}

# for ((ratio=$start; ratio<=$end; ratio+=$step))
# do
#     echo "Sampling ratio: $ratio"
#     ./benchmark-jd/run.sh $xservice_port $ratio 
# done

# sleep 10

xservice_port=${1:?}