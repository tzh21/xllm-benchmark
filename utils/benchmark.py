"""
Unified benchmarking script for xLLM inference engine.
Supports multiple dataset types including ShareGPT, random, trace-based, and offline datasets.
"""

import argparse
import asyncio
import json
import os
import random
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from metric import (
    RequestFuncInput,
    RequestFuncOutput,
    calculate_metrics,
    get_request,
    sample_sharegpt_requests,
    sample_random_requests,
    sample_generated_shared_prefix_requests,
    sample_trace_requests,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=90)


def set_ulimit(target_soft_limit=65535):
    """Set system resource limits for file descriptors"""
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Failed to set RLIMIT_NOFILE: {e}")


def get_tokenizer(pretrained_model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Load tokenizer from model path or name"""
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


def parse_request_rate_range(request_rate_range: str) -> List[int]:
    """Parse request rate range string into list of rates"""
    parts = request_rate_range.split(",")
    if len(parts) == 3:
        start, stop, step = map(int, parts)
        return list(range(start, stop, step))
    else:
        return list(map(int, parts))


async def async_request_xllm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    disable_stream: bool = False,
    disable_ignore_eos: bool = False,
) -> RequestFuncOutput:
    """Send async request to xLLM backend"""
    api_url = request_func_input.api_url
    assert api_url.endswith("completions"), "API URL must end with 'completions'."

    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model.split('/')[-1],
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not disable_stream,
            "ignore_eos": not disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st

        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    if not disable_stream:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = chunk_bytes.decode("utf-8")
                            if chunk.startswith("data: "):
                                chunk = chunk[6:]

                            latency = time.perf_counter() - st
                            if chunk == "[DONE]":
                                pass
                            else:
                                data = json.loads(chunk)

                                if data["choices"][0].get("text"):
                                    timestamp = time.perf_counter()
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.ttft = ttft
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)

                                    most_recent_timestamp = timestamp
                                    generated_text += data["choices"][0]["text"]
                    else:
                        # Non-streaming response
                        data = await response.json()
                        generated_text = data["choices"][0]["text"]
                        latency = time.perf_counter() - st
                        output.ttft = latency

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                    output.timestamp = request_func_input.timestamp
                else:
                    output.success = False
                    output.error = f"HTTP {response.status}: {await response.text()}"
        except asyncio.TimeoutError:
            output.success = False
            output.error = "Request timeout"
        except Exception as e:
            output.success = False
            output.error = str(e)

    if pbar:
        pbar.update(1)
    return output


def get_dataset(args, tokenizer):
    """Load dataset based on specified type"""
    if args.dataset_name == "sharegpt":
        return sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            apply_chat_template=args.apply_chat_template,
        )
    elif args.dataset_name == "random":
        return sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
        )
    elif args.dataset_name == "generated-shared-prefix":
        return sample_generated_shared_prefix_requests(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            tokenizer=tokenizer,
            args=args
        )
    elif args.dataset_name == "trace":
        return sample_trace_requests(
            dataset_path=args.dataset_path,
            trace_path=args.trace_path,
            tokenizer=tokenizer,
            num_prompts=args.num_prompts,
            trace_scale=args.trace_scale,
            start_time=args.trace_start_time,
            end_time=args.trace_end_time,
            sampling_ratio=args.sampling_ratio,
        )
    elif args.dataset_name == "offline":
        return load_offline_dataset(args, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")


def load_offline_dataset(args, tokenizer):
    """Load offline dataset from parquet files"""
    data_path = Path(args.offline_data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {args.offline_data_dir}")

    parquet_files = list(data_path.glob("*.parquet"))
    if args.offline_max_files:
        parquet_files = parquet_files[:args.offline_max_files]

    print(f"Loading {len(parquet_files)} parquet files from {args.offline_data_dir}")

    all_texts = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)

        # Find text column
        text_columns = ['text', 'article', 'content']
        for col in text_columns:
            if col in df.columns:
                all_texts.extend(df[col].tolist())
                break
        else:
            # Use first string column
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            if text_cols:
                all_texts.extend(df[text_cols[0]].tolist())

    print(f"Loaded {len(all_texts)} text samples")

    # Create prompts by randomly truncating texts
    num_prompts = min(args.num_prompts, len(all_texts))
    selected_texts = random.sample(all_texts, num_prompts)

    prompts = []
    for text in selected_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) < args.offline_min_length:
            continue

        if args.offline_max_length and len(tokens) > args.offline_max_length:
            start_idx = random.randint(0, len(tokens) - args.offline_max_length)
            tokens = tokens[start_idx:start_idx + args.offline_max_length]

        prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)
        prompts.append((prompt_text, len(tokens), args.offline_output_length, -1.0))

    return prompts


async def benchmark(
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int, float]],
    request_rate: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    disable_stream: bool,
    disable_ignore_eos: bool,
    slo_ttft: float,
    slo_tpot: float,
    extra_request_body: Dict[str, Any],
    profile: bool = False,
):
    """Run benchmark with specified configuration"""

    # Limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await async_request_xllm(
                request_func_input=request_func_input,
                pbar=pbar,
                disable_stream=disable_stream,
                disable_ignore_eos=disable_ignore_eos,
            )
        async with semaphore:
            return await async_request_xllm(
                request_func_input=request_func_input,
                pbar=pbar,
                disable_stream=disable_stream,
                disable_ignore_eos=disable_ignore_eos,
            )

    # Warmup test
    test_prompt, test_prompt_len, test_output_len, test_timestamp = input_requests[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=min(test_output_len, 32),
        lora_name="",
        extra_request_body=extra_request_body,
        timestamp=test_timestamp,
    )

    test_output = await async_request_xllm(
        request_func_input=test_input,
        disable_stream=disable_stream,
        disable_ignore_eos=disable_ignore_eos,
    )

    if not test_output.success:
        raise ValueError(f"Initial test run failed: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark...")

    time.sleep(1.0)

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []

    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len, timestamp = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            lora_name="",
            extra_request_body=extra_request_body,
            timestamp=timestamp,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    # Calculate metrics
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend="xllm",
        slo_ttft_ms=slo_ttft,
        slo_tpot_ms=slo_tpot,
    )

    # Print results
    print("\n{s:{c}^{n}}".format(s=" Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", "xllm"))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print("{:<40} {:<10}".format("Max request concurrency:",
                                 max_concurrency if max_concurrency else "not set"))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total token throughput (tok/s):", metrics.total_throughput))
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))

    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median E2E Latency (ms):", metrics.median_e2e_latency_ms))

    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))

    print("{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", metrics.p90_tpot_ms))

    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))

    if metrics.slo_ttft_violation_rate is not None or metrics.slo_tpot_violation_rate is not None:
        print("{s:{c}^{n}}".format(s="SLO Violation Rates", n=50, c="-"))
        if metrics.slo_ttft_violation_rate is not None:
            print("{:<40} {:<10.4f}".format("TTFT SLO violation rate:", metrics.slo_ttft_violation_rate))
        if metrics.slo_tpot_violation_rate is not None:
            print("{:<40} {:<10.4f}".format("TPOT SLO violation rate:", metrics.slo_tpot_violation_rate))
        if metrics.slo_ttft_or_tpot_violation_rate is not None:
            print("{:<40} {:<10.4f}".format("SLO TTFT or TPOT violation rate:", metrics.slo_ttft_or_tpot_violation_rate))

    print("=" * 50)

    return {
        "backend": "xllm",
        "request_rate": request_rate,
        "max_concurrency": max_concurrency,
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "concurrency": metrics.concurrency,
        "slo_ttft_violation_rate": metrics.slo_ttft_violation_rate,
        "slo_tpot_violation_rate": metrics.slo_tpot_violation_rate,
        "slo_ttft_or_tpot_violation_rate": metrics.slo_ttft_or_tpot_violation_rate,
    }


def run_benchmark(args):
    """Main function to run benchmark"""

    # Set environment
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse extra request body
    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    # Set xLLM URL
    model_url = f"{args.base_url}/v1/models"
    api_url = f"{args.base_url}/v1/completions"
    base_url = args.base_url

    # Get model name
    if args.model is None:
        try:
            response = requests.get(model_url)
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    print(f"Arguments: {args}\n")

    # Get tokenizer
    tokenizer_id = args.tokenizer if args.tokenizer else args.model
    tokenizer = get_tokenizer(tokenizer_id)

    # Load dataset
    input_requests = get_dataset(args, tokenizer)
    args.num_prompts = len(input_requests)

    print(f"Number of input requests: {len(input_requests)}")

    # Run benchmark
    if not args.multi:
        result = asyncio.run(
            benchmark(
                api_url=api_url,
                base_url=base_url,
                model_id=args.model,
                tokenizer=tokenizer,
                input_requests=input_requests,
                request_rate=args.request_rate,
                max_concurrency=args.max_concurrency,
                disable_tqdm=args.disable_tqdm,
                disable_stream=args.disable_stream,
                disable_ignore_eos=args.disable_ignore_eos,
                slo_ttft=args.slo_ttft,
                slo_tpot=args.slo_tpot,
                extra_request_body=extra_request_body,
                profile=args.profile,
            )
        )
        save_results(args, result)
    else:
        # Multiple request rates
        request_rates = parse_request_rate_range(args.request_rate_range)
        for rate in request_rates:
            result = asyncio.run(
                benchmark(
                    api_url=api_url,
                    base_url=base_url,
                    model_id=args.model,
                    tokenizer=tokenizer,
                    input_requests=input_requests,
                    request_rate=rate,
                    max_concurrency=args.max_concurrency,
                    disable_tqdm=args.disable_tqdm,
                    disable_stream=args.disable_stream,
                    disable_ignore_eos=args.disable_ignore_eos,
                    slo_ttft=args.slo_ttft,
                    slo_tpot=args.slo_tpot,
                    extra_request_body=extra_request_body,
                    profile=args.profile,
                )
            )
            result["request_rate"] = rate
            save_results(args, result)


def save_results(args, result):
    """Save benchmark results to file"""
    if args.output_file:
        output_file = args.output_file
    else:
        now = datetime.now().strftime("%m%d%H%M%S")
        os.makedirs("results", exist_ok=True)

        # Handle num_prompts=None for file naming
        prompt_count = args.num_prompts if args.num_prompts is not None else "all"

        if args.dataset_name == "random":
            output_file = f"results/xllm_{now}_{prompt_count}_{args.dataset_name}_{args.random_input_len}_{args.random_output_len}.jsonl"
        elif args.dataset_name == "trace":
            output_file = f"results/xllm_{now}_{prompt_count}_{args.dataset_name}_{args.trace_scale}.jsonl"
        elif args.dataset_name == "offline":
            output_file = f"results/xllm_{now}_{prompt_count}_offline.jsonl"
        else:
            output_file = f"results/xllm_{now}_{prompt_count}_sharegpt.jsonl"

    # Add dataset info to result
    result["dataset_name"] = args.dataset_name
    result["num_prompts"] = args.num_prompts

    with open(output_file, "a") as f:
        f.write(json.dumps(result, indent=4) + "\n")

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark xLLM inference engine")

    # Basic arguments
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Server or API base url (e.g., http://127.0.0.1:9811).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, will request /v1/models.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, uses the model.",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "random", "generated-shared-prefix", "trace", "offline"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to process.",
    )

    # Request configuration
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. Use 'inf' for batch mode.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests.",
    )
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )

    # Output configuration
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")

    # SLO arguments
    parser.add_argument(
        "--slo-ttft",
        type=float,
        default=1000,
        help="Service Level Objective for TTFT in milliseconds.",
    )
    parser.add_argument(
        "--slo-tpot",
        type=float,
        default=30,
        help="Service Level Objective for TPOT in milliseconds.",
    )

    # ShareGPT dataset arguments
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for ShareGPT dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="Context length limit for ShareGPT dataset.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template to prompts.",
    )

    # Random dataset arguments
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Input length for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=1024,
        help="Output length for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for random dataset.",
    )

    # Generated shared prefix arguments
    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument("--gsp-num-groups", type=int, default=64)
    group.add_argument("--gsp-prompts-per-group", type=int, default=16)
    group.add_argument("--gsp-system-prompt-len", type=int, default=2048)
    group.add_argument("--gsp-question-len", type=int, default=128)
    group.add_argument("--gsp-output-len", type=int, default=256)

    # Trace dataset arguments
    parser.add_argument("--trace-path", type=str, default="", help="Trace file path.")
    parser.add_argument("--trace-start-time", type=float, default=None)
    parser.add_argument("--trace-end-time", type=float, default=None)
    parser.add_argument("--trace-scale", type=float, default=1)
    parser.add_argument("--sampling-ratio", type=float, default=1.0)

    # Offline dataset arguments
    parser.add_argument(
        "--offline-data-dir",
        type=str,
        default="/export/home/tangzihan/xllm/offline-dataset/arxiv-summarization/section/",
        help="Directory containing offline dataset parquet files.",
    )
    parser.add_argument(
        "--offline-max-files",
        type=int,
        default=None,
        help="Maximum number of parquet files to load.",
    )
    parser.add_argument(
        "--offline-min-length",
        type=int,
        default=100,
        help="Minimum prompt length in tokens.",
    )
    parser.add_argument(
        "--offline-max-length",
        type=int,
        default=2048,
        help="Maximum prompt length in tokens.",
    )
    parser.add_argument(
        "--offline-output-length",
        type=int,
        default=512,
        help="Output length for offline dataset.",
    )

    # Multi-rate testing
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Test multiple request rates.",
    )
    parser.add_argument(
        "--request-rate-range",
        type=str,
        default="2,34,2",
        help="Range of request rates (start,stop,step) or comma-separated list.",
    )

    # Advanced options
    parser.add_argument(
        "--extra-request-body",
        type=str,
        help="Extra JSON to append to request payload.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling (requires server support).",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()