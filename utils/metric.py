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
    tpot: float = 0.0
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
    p90_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    p90_tpot_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    concurrency: float
    slo_ttft_violation_rate: Optional[float] = None
    slo_tpot_violation_rate: Optional[float] = None
    slo_ttft_tpot_violation_rate: Optional[float] = None
    slo_ttft_or_tpot_violation_rate: Optional[float] = None



def interpolate_timestamp(start, end):
    return random.uniform(start, end)


def sample_trace_requests(
    prompt_path: str,
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
            timestamp_ms = float(data["timestamp"])  # Now in milliseconds
            if start_time is not None and end_time is not None:
                if timestamp_ms < start_time:
                    continue
                if timestamp_ms > end_time:
                    break
            data["timestamp"] = timestamp_ms / trace_scale  # Apply scale (in milliseconds)
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

    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path) as f:
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
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_token_ids)

        # Skip empty prompt and try next one
        while prompt_len == 0 and i < len(dataset):
            i += 1
            if i >= len(dataset):
                data = dataset[i % len(dataset)]
            else:
                data = dataset[i]
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_len = len(prompt_token_ids)

        # If still empty, use a default prompt
        if prompt_len == 0:
            prompt = "Hello"
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_len = len(prompt_token_ids)

        input_len = int(trace_entry["input_length"])
        output_len = int(trace_entry["output_length"])

        # Apply safety limits to prevent excessive token sequences
        if input_len > max_token_limit:
            input_len = max_token_limit

        if prompt_len > input_len:
            input_ids = prompt_token_ids[: input_len]
        else:
            ratio = (input_len + prompt_len - 1) // prompt_len
            input_ids = (prompt_token_ids * ratio)[: input_len]

        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        input_requests.append((prompt, input_len, output_len, trace_entry["timestamp"]))

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{final_prompt_count} requests...")

    return input_requests


def sample_constant_requests(
    prompt_path: str,
    trace_path: str,
    tokenizer: PreTrainedTokenizerBase,
    trace_scale: float = 1.0,
    start_time: Optional[float] = None,
    constant_rate: float = 1.0,
    constant_duration: Optional[float] = None,
) -> List[Tuple[str, int, int, float]]:
    """Sample requests from trace file but use constant timestamps for constant rate sending"""

    # Calculate the target number of requests based on constant_rate and constant_duration
    target_request_count = None
    if constant_duration is not None and constant_rate > 0:
        target_request_count = int(constant_rate * constant_duration)

    # Load the dataset from trace file and determine end_time based on target request count
    mooncake_data = []
    end_time = None

    with open(trace_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            timestamp_ms = float(data["timestamp"])  # Now in milliseconds

            # Skip requests before start_time
            if start_time is not None and timestamp_ms < start_time:
                continue

            mooncake_data.append(data)

            # If we have a target request count, find the end_time for exactly that many requests
            if target_request_count is not None and len(mooncake_data) >= target_request_count:
                end_time = timestamp_ms
                break

    # Use all mooncake_data entries to generate prompts
    final_prompt_count = len(mooncake_data)
    if final_prompt_count == 0:
        print("Warning: No trace data available to generate prompts")
        return []

    # Limit requests based on duration if specified
    if constant_duration is not None:
        max_requests_by_duration = int(constant_rate * constant_duration)
        if max_requests_by_duration < final_prompt_count:
            final_prompt_count = max_requests_by_duration
            mooncake_data = mooncake_data[:final_prompt_count]
            print(f"Limited to {final_prompt_count} requests based on duration {constant_duration}s at rate {constant_rate} req/s")

    print(f"Generating {final_prompt_count} constant rate requests from trace data")

    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path) as f:
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
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_token_ids)

        # Skip empty prompt and try next one
        while prompt_len == 0 and i < len(dataset):
            i += 1
            if i >= len(dataset):
                data = dataset[i % len(dataset)]
            else:
                data = dataset[i]
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_len = len(prompt_token_ids)

        # If still empty, use a default prompt
        if prompt_len == 0:
            prompt = "Hello"
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_len = len(prompt_token_ids)

        input_len = int(trace_entry["input_length"])
        output_len = int(trace_entry["output_length"])

        # Apply safety limits to prevent excessive token sequences
        if input_len > max_token_limit:
            input_len = max_token_limit

        if prompt_len > input_len:
            input_ids = prompt_token_ids[: input_len]
        else:
            ratio = (input_len + prompt_len - 1) // prompt_len
            input_ids = (prompt_token_ids * ratio)[: input_len]

        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        # Generate timestamp based on constant rate (in milliseconds)
        # Interval between requests in seconds = 1.0 / constant_rate
        # Convert to milliseconds for consistency with trace mode
        timestamp_ms = i * (1000.0 / constant_rate)
        input_requests.append((prompt, input_len, output_len, timestamp_ms))

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{final_prompt_count} requests...")

    return input_requests


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
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=True)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += outputs[i].prompt_len
            tpots.append(outputs[i].tpot)
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

    # Calculate SLO violation rates including timeouts
    slo_ttft_violation_rate = None
    slo_tpot_violation_rate = None
    slo_ttft_tpot_violation_rate = None
    slo_ttft_or_tpot_violation_rate = None

    if slo_ttft_ms is not None or slo_tpot_ms is not None:
        total_requests = len(outputs)
        if total_requests > 0:
            ttft_violations = 0
            tpot_violations = 0
            both_violations = 0
            either_violations = 0
            timeout_requests = 0

            for i in range(len(outputs)):
                # Count timeout requests as violations
                is_timeout = not outputs[i].success and outputs[i].error == "Request timeout"
                if is_timeout:
                    timeout_requests += 1

                if outputs[i].success:
                    ttft_ms = outputs[i].ttft * 1000
                    output_len = outputs[i].output_len

                    ttft_violates = slo_ttft_ms is not None and ttft_ms > slo_ttft_ms
                    tpot_violates = False

                    if slo_tpot_ms is not None and output_len > 2:
                        tpot_ms = ((outputs[i].latency - outputs[i].ttft) / (output_len - 2)) * 1000
                        tpot_violates = tpot_ms > slo_tpot_ms

                    if ttft_violates:
                        ttft_violations += 1
                    if tpot_violates:
                        tpot_violations += 1
                    if ttft_violates and tpot_violates:
                        both_violations += 1
                    if ttft_violates or tpot_violates:
                        either_violations += 1
                elif is_timeout:
                    # Timeout requests count as both TTFT and TPOT violations
                    if slo_ttft_ms is not None:
                        ttft_violations += 1
                    if slo_tpot_ms is not None:
                        tpot_violations += 1
                    if slo_ttft_ms is not None and slo_tpot_ms is not None:
                        both_violations += 1
                    if slo_ttft_ms is not None or slo_tpot_ms is not None:
                        either_violations += 1

            if slo_ttft_ms is not None:
                slo_ttft_violation_rate = ttft_violations / total_requests
            if slo_tpot_ms is not None:
                slo_tpot_violation_rate = tpot_violations / total_requests
            if slo_ttft_ms is not None and slo_tpot_ms is not None:
                slo_ttft_tpot_violation_rate = both_violations / total_requests
            if slo_ttft_ms is not None or slo_tpot_ms is not None:
                slo_ttft_or_tpot_violation_rate = either_violations / total_requests

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
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        p90_tpot_ms=np.percentile(tpots or 0, 90) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        p90_e2e_latency_ms=np.percentile(e2e_latencies, 90) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
        slo_ttft_violation_rate=slo_ttft_violation_rate,
        slo_tpot_violation_rate=slo_tpot_violation_rate,
        slo_ttft_tpot_violation_rate=slo_ttft_tpot_violation_rate,
        slo_ttft_or_tpot_violation_rate=slo_ttft_or_tpot_violation_rate,
    )

    return metrics, output_lens

