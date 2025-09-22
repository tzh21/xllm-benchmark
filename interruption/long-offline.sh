xservice_port=${1:?}
python ./interruption/long-prompt.py \
    --offline \
    --prompt ./interruption/long-prompt.txt \
    --max_tokens 5 \
    --port $xservice_port