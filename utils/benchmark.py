"""
Unified benchmarking script for xLLM inference engine.
Supports trace-based datasets.
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
    sample_trace_requests,
    sample_constant_requests,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=600)


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
    tokenizer: PreTrainedTokenizerBase,
    pbar: Optional[tqdm] = None,
    disable_stream: bool = False,
    disable_ignore_eos: bool = False,
    offline: bool = False,
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
            "offline": offline,
            **request_func_input.extra_request_body,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        token_count = 0
        st = time.perf_counter()

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
                                    if token_count < 2:
                                        output.ttft = timestamp - st
                                    token_count += 1
                                    chunk_text = data["choices"][0]["text"]
                                    generated_text += chunk_text
                    else:
                        assert False # Non-streaming response not supported
                        # data = await response.json()
                        # generated_text = data["choices"][0]["text"]
                        # latency = time.perf_counter() - st
                        # output.ttft = latency

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len  # Use actual token count instead of expected length
                    output.timestamp = request_func_input.timestamp
                    if token_count > 2:
                        output.tpot = (output.latency - output.ttft) / (output.output_len - 2)
                    else:
                        output.tpot = 0.0
                else:
                    output.success = False
                    output.error = f"HTTP {response.status}: {await response.text()}"
        except asyncio.TimeoutError:
            print(f"Request timeout - API URL: {request_func_input.api_url}, Model: {request_func_input.model}, "
                  f"Prompt length: {request_func_input.prompt_len}, Output length: {request_func_input.output_len}, "
                  f"LoRA: {request_func_input.lora_name}, Timestamp: {request_func_input.timestamp}, "
                  f"Extra request body: {request_func_input.extra_request_body}")
            output.success = False
            output.error = "Request timeout"
            # sys.exit(1)
        except Exception as e:
            output.success = False
            output.error = str(e)

    if pbar:
        pbar.update(1)
    return output


def get_dataset(args, tokenizer):
    """Load dataset based on specified type"""
    if args.traffic_mode == "trace":
        return sample_trace_requests(
            prompt_path=args.prompt_path,
            trace_path=args.trace_path,
            tokenizer=tokenizer,
            num_prompts=args.num_prompts,
            trace_scale=args.trace_scale,
            start_time=args.trace_start_time,
            end_time=args.trace_end_time,
            sampling_ratio=args.sampling_ratio,
        )
    elif args.traffic_mode == "constant":
        return sample_constant_requests(
            prompt_path=args.prompt_path,
            trace_path=args.trace_path,
            tokenizer=tokenizer,
            trace_scale=args.trace_scale,
            start_time=args.trace_start_time,
            constant_rate=args.constant_rate,
            constant_duration=args.constant_duration,
        )
    else:
        raise ValueError(f"Unknown traffic mode: {args.traffic_mode}")


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
    args: argparse.Namespace,
    profile: bool = False,
    offline: bool = False,
):
    """Run benchmark with specified configuration"""

    # Limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await async_request_xllm(
                request_func_input=request_func_input,
                tokenizer=tokenizer,
                pbar=pbar,
                disable_stream=disable_stream,
                disable_ignore_eos=disable_ignore_eos,
                offline=offline,
            )
        async with semaphore:
            return await async_request_xllm(
                request_func_input=request_func_input,
                tokenizer=tokenizer,
                pbar=pbar,
                disable_stream=disable_stream,
                disable_ignore_eos=disable_ignore_eos,
                offline=offline,
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
        tokenizer=tokenizer,
        disable_stream=disable_stream,
        disable_ignore_eos=disable_ignore_eos,
        offline=offline,
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

    # Collect per-request details
    request_details = []
    for i, (input_req, output) in enumerate(zip(input_requests, outputs)):
        prompt, input_len, max_output_len, timestamp = input_req

        request_detail = {
            "request_id": i,
            "timestamp": timestamp,
            "latency_ms": output.latency * 1000 if output.success else None,
            "input_length": output.prompt_len,
            "output_length": output.output_len,
            "success": output.success,
            "error": output.error if not output.success else None,
            "ttft_ms": output.ttft * 1000 if output.success and output.ttft >= 0 else None,
            "tpot_ms": output.tpot * 1000 if output.success and output.tpot >= 0 else None, 
        }
        request_details.append(request_detail)

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
    print("{:<40} {:<10.2f}".format("P99 E2E Latency (ms):", metrics.p99_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("P90 E2E Latency (ms):", metrics.p90_e2e_latency_ms))

    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", metrics.p90_ttft_ms))

    print("{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", metrics.p90_tpot_ms))

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
        "sampling_ratio": args.sampling_ratio,
        "sent_requests": len(input_requests),
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
        "p90_e2e_latency_ms": metrics.p90_e2e_latency_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "p90_ttft_ms": metrics.p90_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "concurrency": metrics.concurrency,
        "slo_ttft_violation_rate": metrics.slo_ttft_violation_rate,
        "slo_tpot_violation_rate": metrics.slo_tpot_violation_rate,
        "slo_ttft_or_tpot_violation_rate": metrics.slo_ttft_or_tpot_violation_rate,
        "args": str(args),
        "requests": request_details,
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
                args=args,
                profile=args.profile,
                offline=args.offline,
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
                    args=args,
                    profile=args.profile,
                    offline=args.offline,
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

        if args.traffic_mode == "trace":
            output_file = f"results/xllm_{now}_{prompt_count}_{args.traffic_mode}_{args.trace_scale}.jsonl"
        else:
            output_file = f"results/xllm_{now}_{prompt_count}_{args.traffic_mode}.jsonl"

    # Add dataset info to result
    result["traffic_mode"] = args.traffic_mode
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

    # Traffic mode arguments
    parser.add_argument(
        "--traffic-mode",
        type=str,
        default="trace",
        choices=["trace", "constant"],
        help="Traffic mode to use for benchmarking. 'trace' uses timestamps from trace file, 'constant' sends requests at constant rate.",
    )
    parser.add_argument(
        "--prompt-path", type=str, default="", help="Path to the prompt dataset."
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
        default=5000,
        help="Service Level Objective for TTFT in milliseconds.",
    )
    parser.add_argument(
        "--slo-tpot",
        type=float,
        default=40,
        help="Service Level Objective for TPOT in milliseconds.",
    )


    # Trace dataset arguments
    parser.add_argument("--trace-path", type=str, default="", help="Trace file path.")
    parser.add_argument("--trace-start-time", type=float, default=None, help="Start time in milliseconds")
    parser.add_argument("--trace-end-time", type=float, default=None, help="End time in milliseconds")
    parser.add_argument("--trace-scale", type=float, default=1)
    parser.add_argument("--sampling-ratio", type=float, default=1.0)

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
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set offline field to true in requests.",
    )
    parser.add_argument(
        "--constant-rate",
        type=float,
        default=None,
        help="Request rate (requests per second) for constant traffic mode.",
    )
    parser.add_argument(
        "--constant-duration",
        type=float,
        default=None,
        help="Total duration (in seconds) for sending requests in constant mode. If not set, uses all available requests.",
    )

    args = parser.parse_args()
    if args.constant_rate is not None and args.constant_rate == 0.0:
        print("Constant rate equals 0. Exit")
        sys.exit(0)
    run_benchmark(args)


if __name__ == "__main__":
    main()