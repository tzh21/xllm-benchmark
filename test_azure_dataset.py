#!/usr/bin/env python3

import json
import random
from transformers import AutoTokenizer

# Test the specific azure_conv dataset processing
model_name = "/export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct"
dataset_path = "/export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
trace_path = "/export/home/tangzihan/xllm-base/datasets/online-datasets/azure_conv.jsonl"

print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading trace file: {trace_path}")
with open(trace_path) as f:
    mooncake_data = [json.loads(line) for line in f]

print(f"Trace file loaded with {len(mooncake_data)} entries")

# Apply sampling ratio
sampling_ratio = 0.99
original_count = len(mooncake_data)
sample_size = max(1, int(original_count * sampling_ratio))
mooncake_data = random.sample(mooncake_data, sample_size)
mooncake_data.sort(key=lambda x: x["timestamp"])
print(f"Sampled {sample_size} requests from {original_count} (ratio: {sampling_ratio})")

print(f"Loading dataset: {dataset_path}")
with open(dataset_path) as f:
    dataset = json.load(f)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
random.shuffle(dataset)
print(f"Dataset loaded with {len(dataset)} valid entries")

# Test the problematic loop
num_prompts = 19000
if len(mooncake_data) < num_prompts:
    num_prompts = len(mooncake_data)

print(f"Processing {num_prompts} prompts...")

input_requests = []
problematic_entries = []

for data in dataset:
    i = len(input_requests)
    if i == num_prompts:
        break

    if i % 1000 == 0:
        print(f"Processing entry {i}...")

    # Tokenize the prompts and completions.
    prompt = data[0]
    prompt_token_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_token_ids)

    # Skip empty prompt
    if prompt_len == 0:
        continue

    input_len = int(mooncake_data[i]["input_length"])
    output_len = int(mooncake_data[i]["output_length"])

    if prompt_len > input_len:
        input_ids = prompt_token_ids[: input_len]
    else:
        ratio = (input_len + prompt_len - 1) // prompt_len
        input_ids = (prompt_token_ids * ratio)[: input_len]

    # Test decode with a timeout or size check
    if len(input_ids) > 10000:  # Large token sequence check
        print(f"Warning: Large token sequence at entry {i}: {len(input_ids)} tokens")
        problematic_entries.append(i)
        # Skip this entry or truncate further
        input_ids = input_ids[:10000]

    try:
        decoded_prompt = tokenizer.decode(input_ids)
        input_requests.append((decoded_prompt, input_len, output_len, mooncake_data[i]["timestamp"]))
    except Exception as e:
        print(f"Error decoding entry {i}: {e}")
        problematic_entries.append(i)
        continue

print(f"Completed processing. Created {len(input_requests)} requests.")
if problematic_entries:
    print(f"Found {len(problematic_entries)} problematic entries: {problematic_entries[:10]}...")  # Show first 10

print("Azure dataset test completed successfully!")