xservice_port=${1:?}
curl http://127.0.0.1:$xservice_port/v1/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "Qwen2.5-7B-Instruct",
  "prompt": "William Hanna",
  "max_tokens": 5,
  "temperature": 0,
  "stream": true,
  "offline": false
}
EOF
