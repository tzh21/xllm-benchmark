#!/usr/bin/env python3

import json
import random
from transformers import AutoTokenizer

# Minimal test to find the hanging issue
model_name = "/export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct"
dataset_path = "/export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
trace_path = "/export/home/tangzihan/xllm-base/datasets/online-datasets/azure_conv.jsonl"

print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading trace file: {trace_path}")
with open(trace_path) as f:
    mooncake_data = [json.loads(line) for line in f]

print(f"Loading dataset: {dataset_path}")
with open(dataset_path) as f:
    dataset = json.load(f)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
random.shuffle(dataset)

print(f"Testing first 10 entries individually...")

for i in range(min(10, len(dataset))):
    print(f"\n--- Testing entry {i} ---")

    prompt = dataset[i][0]
    print(f"Prompt length: {len(prompt)} characters")

    prompt_token_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_token_ids)
    print(f"Tokenized to {prompt_len} tokens")

    input_len = int(mooncake_data[i]["input_length"])
    output_len = int(mooncake_data[i]["output_length"])
    print(f"Target input_len: {input_len}, output_len: {output_len}")

    if prompt_len > input_len:
        input_ids = prompt_token_ids[: input_len]
    else:
        ratio = (input_len + prompt_len - 1) // prompt_len
        input_ids = (prompt_token_ids * ratio)[: input_len]

    print(f"Final input_ids length: {len(input_ids)}")

    if len(input_ids) > 5000:
        print(f"WARNING: Large token sequence detected!")
        # Try to decode just first 100 tokens
        test_decode = tokenizer.decode(input_ids[:100])
        print(f"First 100 tokens decode to: {len(test_decode)} chars")
    else:
        print("Decoding...")
        decoded = tokenizer.decode(input_ids)
        print(f"Decoded to {len(decoded)} characters")

print("\nFirst 10 entries completed successfully!")

# Now test the problematic case
print(f"\nTesting entry with very large input_length...")

# Find entries with large input_length
large_input_entries = []
for i, data in enumerate(mooncake_data[:1000]):  # Check first 1000
    if int(data["input_length"]) > 5000:
        large_input_entries.append(i)

print(f"Found {len(large_input_entries)} entries with input_length > 5000 in first 1000")

if large_input_entries:
    test_idx = large_input_entries[0]
    print(f"Testing entry {test_idx} with large input_length...")

    prompt = dataset[test_idx][0]
    prompt_token_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_token_ids)

    input_len = int(mooncake_data[test_idx]["input_length"])
    print(f"Entry {test_idx}: prompt_len={prompt_len}, target input_len={input_len}")

    if prompt_len > input_len:
        input_ids = prompt_token_ids[: input_len]
    else:
        ratio = (input_len + prompt_len - 1) // prompt_len
        input_ids = (prompt_token_ids * ratio)[: input_len]

    print(f"Final input_ids length: {len(input_ids)}")

    if len(input_ids) > 10000:
        print(f"Very large token sequence: {len(input_ids)} tokens!")
        print("Trying incremental decode...")

        for chunk_size in [100, 1000, 5000]:
            if len(input_ids) >= chunk_size:
                print(f"  Testing decode of first {chunk_size} tokens...")
                try:
                    chunk_decode = tokenizer.decode(input_ids[:chunk_size])
                    print(f"  Success: decoded to {len(chunk_decode)} characters")
                except Exception as e:
                    print(f"  Failed: {e}")
                    break

print("\nDebug completed!")