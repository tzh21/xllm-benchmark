# model=Llama-3.3-70B-Instruct
model=Qwen2.5-7B-Instruct
curl http://127.0.0.1:27712/v1/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "$model",
  "prompt": "Tom and Jerry",
  "max_tokens": 10,
  "temperature": 0,
  "stream": true,
  "offline": true
}
EOF
