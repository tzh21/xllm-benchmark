#!/usr/bin/env python3

import json
from transformers import AutoTokenizer

# Test tokenizer with the dataset
model_name = "/export/home/tangzihan/modelscope/models/Qwen/Qwen2.5-7B-Instruct"
dataset_path = "/export/home/tangzihan/xllm-base/datasets/online-datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading dataset: {dataset_path}")
with open(dataset_path) as f:
    dataset = json.load(f)

print(f"Dataset loaded with {len(dataset)} entries")

# Test first few entries
for i, data in enumerate(dataset[:5]):
    print(f"\nTesting entry {i}:")
    prompt = data["conversations"][0]["value"]
    print(f"Prompt length: {len(prompt)} characters")

    try:
        token_ids = tokenizer.encode(prompt)
        print(f"Tokenized to {len(token_ids)} tokens")

        # Test decoding
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded back to {len(decoded)} characters")

        # Test with different input lengths
        for input_len in [100, 200, 500]:
            if len(token_ids) > input_len:
                truncated_ids = token_ids[:input_len]
            else:
                # Repeat tokens to reach desired length
                ratio = (input_len + len(token_ids) - 1) // len(token_ids)
                truncated_ids = (token_ids * ratio)[:input_len]

            print(f"Testing decode with input_len={input_len} (actual tokens: {len(truncated_ids)})")
            decoded_truncated = tokenizer.decode(truncated_ids)
            print(f"  Decoded length: {len(decoded_truncated)} characters")

    except Exception as e:
        print(f"Error: {e}")
        break

print("\nTokenizer test completed successfully!")