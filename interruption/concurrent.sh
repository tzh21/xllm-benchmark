xservice_port=${1:?}
./interruption/long-offline.sh $xservice_port &
sleep 0.3
./interruption/long-online.sh $xservice_port &
wait