import argparse
import asyncio
import json
import os
import pickle
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
import csv
import pandas as pd

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: str
    extra_request_body: Dict[str, Any]
    timestamp: float = 0.0


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    timestamp: float = 0.0


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    p90_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float
    slo_ttft_violation_rate: Optional[float] = None
    slo_tpot_violation_rate: Optional[float] = None
    slo_ttft_tpot_violation_rate: Optional[float] = None
    slo_ttft_or_tpot_violation_rate: Optional[float] = None


SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def interpolate_timestamp(start, end):
    return random.uniform(start, end)


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    apply_chat_template=False,
) -> List[Tuple[str, int, int, float]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append((prompt, prompt_len, output_len, -1.0))

    print(f"#Input tokens: {np.sum([x[1] for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x[2] for x in filtered_dataset])}")
    return filtered_dataset


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
) -> List[Tuple[str, int, int, float]]:

    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    if True:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        # Download sharegpt if necessary
        if not os.path.isfile(dataset_path):
            dataset_path = download_and_cache_file(SHAREGPT_URL)

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[Tuple[str, int, int]] = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            prompt = tokenizer.decode(input_ids)
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i]), -1))
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            prompt = tokenizer.decode(
                [
                    (offsets[i] + i + j) % tokenizer.vocab_size
                    for j in range(input_lens[i])
                ]
            )
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i], -1.0)))

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def get_gen_prefix_cache_path(args, tokenizer):
    """Create cache directory under ~/.cache/sglang/benchmark"""
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    # Create a unique cache filename based on the generation parameters
    cache_key = (
        f"gen_shared_prefix_{args.gsp_num_groups}_{args.gsp_prompts_per_group}_"
        f"{args.gsp_system_prompt_len}_{args.gsp_question_len}_{args.gsp_output_len}_"
        f"{tokenizer.__class__.__name__}.pkl"
    )
    return cache_dir / cache_key

def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
    args
) -> List[Tuple[str, int, int, float]]:
    """Generate benchmark requests with shared system prompts using random tokens and caching."""
    cache_path = get_gen_prefix_cache_path(args, tokenizer)

    # Try to load from cache first
    if cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("\nGenerating new input data...")

    # Generate system prompts for each group
    system_prompts = []
    for _ in range(num_groups):
        system_prompt = gen_prompt(tokenizer, system_prompt_len)
        system_prompts.append(system_prompt)

    # Generate questions
    questions = []
    for _ in range(num_groups * prompts_per_group):
        question = gen_prompt(tokenizer, question_len)
        questions.append(question)

    # Combine system prompts with questions
    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0

    for group_idx in tqdm(range(num_groups), desc="Generating system prompt"):
        system_prompt = system_prompts[group_idx]
        for prompt_idx in tqdm(
            range(prompts_per_group), desc="Generating questions", leave=False
        ):
            question = questions[group_idx * prompts_per_group + prompt_idx]
            full_prompt = f"{system_prompt}\n\n{question}"
            prompt_len = len(tokenizer.encode(full_prompt))

            input_requests.append((full_prompt, prompt_len, output_len, -1.0))
            total_input_tokens += prompt_len
            total_output_tokens += output_len

    # Shuffle questions
    random.shuffle(input_requests)

    # Print statistics
    print(f"\nGenerated shared prefix dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Total prompts: {len(input_requests)}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(
        f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
    )
    print(
        f"Average question length: {sum(len(tokenizer.encode(q)) for q in questions) / len(questions):.1f} tokens\n"
    )

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Caching generated input data to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(input_requests, f)

    return input_requests

def sample_trace_requests(
    dataset_path: str,
    trace_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_prompts: Optional[int] = None,
    trace_scale: float = 1.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    sampling_ratio: float = 1.0,
) -> List[Tuple[str, int, int, float]]:

    # Load the dataset from trace file, num_prompts controls how many entries to read
    mooncake_data = []
    with open(trace_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            timestamp = float(data["timestamp"]) / 1000
            if start_time is not None and end_time is not None:
                if timestamp < start_time:
                    continue
                if timestamp > end_time:
                    break
            data["timestamp"] = timestamp / trace_scale
            mooncake_data.append(data)
            if num_prompts is not None and len(mooncake_data) == num_prompts:
                break

    # Apply sampling ratio to the loaded trace data
    if sampling_ratio < 1.0:
        original_count = len(mooncake_data)
        sample_size = max(1, int(original_count * sampling_ratio))
        mooncake_data = random.sample(mooncake_data, sample_size)
        mooncake_data.sort(key=lambda x: x["timestamp"])
        print(f"Sampled {sample_size} requests from {original_count} (ratio: {sampling_ratio})")
    elif sampling_ratio > 1.0:
        original_count = len(mooncake_data)
        additional_request = int((sampling_ratio - 1.0) * original_count)

        for _ in range(additional_request):
            # Avoid inserting at the last index to prevent edge case
            # Use len(mooncake_data) - 2 to ensure we can always find a valid interpolation range
            insert_index = random.randint(0, len(mooncake_data) - 2)
            new_request = mooncake_data[insert_index].copy()
            new_request['timestamp'] = interpolate_timestamp(mooncake_data[insert_index]['timestamp'], mooncake_data[insert_index+1]['timestamp'])
            mooncake_data.insert(insert_index + 1, new_request)
        mooncake_data.sort(key=lambda x: x["timestamp"])
        print(f"Sampled {len(mooncake_data)} requests from {original_count} (ratio: {sampling_ratio})")

    # Use all mooncake_data entries (after sampling) to generate prompts
    final_prompt_count = len(mooncake_data)
    if final_prompt_count == 0:
        print("Warning: No trace data available to generate prompts")
        return []

    print(f"Generating {final_prompt_count} prompts from trace data")

    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    with open(dataset_path) as f:
        dataset = json.load(f)
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]
    random.shuffle(dataset)

    input_requests: List[Tuple[str, int, int, float]] = []
    max_token_limit = 8192  # Safety limit to prevent excessive token sequences

    for i, trace_entry in enumerate(mooncake_data):
        # Find a suitable prompt from the dataset
        if i >= len(dataset):
            # If we've exhausted the dataset, cycle back to the beginning
            data = dataset[i % len(dataset)]
        else:
            data = dataset[i]

        # Tokenize the prompts and completions.
        prompt = data[0]
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)

        # Skip empty prompt and try next one
        while prompt_len == 0 and i < len(dataset):
            i += 1
            if i >= len(dataset):
                data = dataset[i % len(dataset)]
            else:
                data = dataset[i]
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

        # If still empty, use a default prompt
        if prompt_len == 0:
            prompt = "Hello"
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

        input_len = int(trace_entry["input_length"])
        output_len = int(trace_entry["output_length"])

        # Apply safety limits to prevent excessive token sequences
        if input_len > max_token_limit:
            # print(f"Warning: Clamping input_len from {input_len} to {max_token_limit} for entry {i}")
            input_len = max_token_limit

        if prompt_len > input_len:
            input_ids = prompt_token_ids[: input_len]
        else:
            ratio = (input_len + prompt_len - 1) // prompt_len
            input_ids = (prompt_token_ids * ratio)[: input_len]

        # Additional safety check for token sequence length
        # if len(input_ids) > max_token_limit:
            # print(f"Warning: Truncating token sequence from {len(input_ids)} to {max_token_limit} for entry {i}")
            # input_ids = input_ids[:max_token_limit]

        # Decode with progress indicator for large sequences
        # if len(input_ids) > 4000:
        #     print(f"Decoding large sequence: {len(input_ids)} tokens for entry {i}")

        prompt = tokenizer.decode(input_ids)
        input_requests.append((prompt, input_len, output_len, trace_entry["timestamp"]))

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{final_prompt_count} requests...")

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int, float]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    
    if input_requests[0][3] == -1.0:
        input_requests = iter(input_requests)
        for request in input_requests:
            yield request

            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
            # The next request will be sent after the interval.
            await asyncio.sleep(interval)
    else:
        start_time = time.perf_counter() + input_requests[0][3]
        # If the input_requests has timestamp, then we need to wait until the timestamp.
        input_requests = iter(input_requests)
        for request in input_requests:

            # Wait until the timestamp.
            await asyncio.sleep(max(0, request[3] + start_time - time.perf_counter()))
            yield request


def calculate_metrics(
    input_requests: List[Tuple[str, int, int, float]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
    slo_ttft_ms: Optional[float] = None,
    slo_tpot_ms: Optional[float] = None,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += outputs[i].prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    # Calculate SLO violation rates
    slo_ttft_violation_rate = None
    slo_tpot_violation_rate = None
    slo_ttft_tpot_violation_rate = None
    slo_ttft_or_tpot_violation_rate = None
    
    if slo_ttft_ms is not None or slo_tpot_ms is not None:
        if completed > 0:
            ttft_violations = 0
            tpot_violations = 0
            both_violations = 0
            either_violations = 0
            
            for i in range(len(outputs)):
                if outputs[i].success:
                    ttft_ms = outputs[i].ttft * 1000
                    output_len = outputs[i].output_len
                    
                    ttft_violates = slo_ttft_ms is not None and ttft_ms > slo_ttft_ms
                    tpot_violates = False
                    
                    if slo_tpot_ms is not None and output_len > 1:
                        tpot_ms = ((outputs[i].latency - outputs[i].ttft) / (output_len - 1)) * 1000
                        tpot_violates = tpot_ms > slo_tpot_ms
                    
                    if ttft_violates:
                        ttft_violations += 1
                    if tpot_violates:
                        tpot_violations += 1
                    if ttft_violates and tpot_violates:
                        both_violations += 1
                    if ttft_violates or tpot_violates:
                        either_violations += 1
            
            if slo_ttft_ms is not None:
                slo_ttft_violation_rate = ttft_violations / completed
            if slo_tpot_ms is not None:
                slo_tpot_violation_rate = tpot_violations / completed
            if slo_ttft_ms is not None and slo_tpot_ms is not None:
                slo_ttft_tpot_violation_rate = both_violations / completed
            if slo_ttft_ms is not None or slo_tpot_ms is not None:
                slo_ttft_or_tpot_violation_rate = either_violations / completed

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens))
        / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        p90_tpot_ms=np.percentile(tpots or 0, 90) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
        slo_ttft_violation_rate=slo_ttft_violation_rate,
        slo_tpot_violation_rate=slo_tpot_violation_rate,
        slo_ttft_tpot_violation_rate=slo_ttft_tpot_violation_rate,
        slo_ttft_or_tpot_violation_rate=slo_ttft_or_tpot_violation_rate,
    )

    return metrics, output_lens

