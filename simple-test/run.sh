offline=${1:?}; shift
nodes=${1:?}; shift
xservice_port=${1:?}; shift

if [ $nodes == "m" ]; then
    model=Qwen2.5-72B-Instruct
elif [ $nodes == "s" ]; then
    model=Qwen2.5-7B-Instruct
else
    echo "Unknown nodes"
fi

curl http://127.0.0.1:$xservice_port/v1/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "$model",
  "prompt": "William Hanna",
  "max_tokens": 10,
  "temperature": 0,
  "stream": true,
  "ignore_eos": true,
  "offline": $offline
}
EOF
