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
import signal
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
    sample_trace_requests,
    sample_constant_requests,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=3600) # in seconds


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
        token_count = 0
        start_time = time.perf_counter()

        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:

                    async for chunk_bytes in response.content:
                        curr_time = time.perf_counter()
                        latency = curr_time - start_time

                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk = chunk_bytes.decode("utf-8")
                        if chunk.startswith("data: "):
                            chunk = chunk[6:]
                        if chunk == "[DONE]":
                            break

                        try:
                            data = json.loads(chunk)
                            if data["choices"][0].get("text"):
                                if token_count < 2:
                                    output.ttft = curr_time - start_time
                                token_count += 1
                                generated_text = data["choices"][0]["text"]
                        except json.JSONDecodeError as e:
                            output.success = False
                            output.error = f"JSON decode error: {e}. Raw response: {chunk}"
                            break

                    output.success = True
                    output.generated_text = generated_text
                    output.latency = latency
                    output.output_len = request_func_input.output_len  # Use actual token count instead of expected length
                    output.timestamp = request_func_input.timestamp
                    output.tpot = (output.latency - output.ttft) / (output.output_len - 2) if token_count > 2 else 0.0
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
        # except Exception as e:
        #     output.success = False
        #     output.error = str(e)

    if pbar is not None:
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
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int, float]],
    disable_tqdm: bool,
    disable_stream: bool,
    disable_ignore_eos: bool,
    slo_ttft: float,
    slo_tpot: float,
    extra_request_body: Dict[str, Any],
    args: argparse.Namespace,
    offline: bool = False,
):
    """Run benchmark with specified configuration"""

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

    # 添加信号处理标志
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal. Shutting down gracefully...")
        shutdown_event.set()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[Tuple[int, asyncio.Task]] = []

    try:
        for idx, request in enumerate(input_requests):
            # 检查是否收到中断信号
            if shutdown_event.is_set():
                print("Interrupt detected, stopping request submission...")
                raise KeyboardInterrupt

            if idx != 0:
                last_timestamp = input_requests[idx - 1][3]
                curr_timestamp = input_requests[idx][3]
                sleep_duration = (curr_timestamp - last_timestamp) / 1000
                await asyncio.sleep(sleep_duration)
                
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
            task = asyncio.create_task(
                async_request_xllm(
                    request_func_input=request_func_input,
                    tokenizer=tokenizer,
                    pbar=pbar,
                    disable_stream=disable_stream,
                    disable_ignore_eos=disable_ignore_eos,
                    offline=offline,
                )
            )
            tasks.append((idx, task))

            outputs: List[RequestFuncOutput] = await asyncio.gather(*[task for _, task in tasks])

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt received. Cancelling remaining requests and processing partial results...")
        # Cancel all pending tasks
        for _, task in tasks:
            if not task.done():
                task.cancel()

        # Gather completed results with their indices
        completed_pairs = []
        for idx, task in tasks:
            if task.done() and not task.cancelled():
                try:
                    result = task.result()
                    if result.success:
                        completed_pairs.append((idx, result))
                except Exception:
                    pass

        print(f"Processed {len(completed_pairs)} completed requests out of {len(tasks)} total.")

        # Sort by index and extract outputs
        completed_pairs.sort(key=lambda x: x[0])
        outputs = [result for _, result in completed_pairs]
        # Update input_requests to only include completed ones
        input_requests = [input_requests[idx] for idx, _ in completed_pairs]

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

    result = {
        "backend": "xllm",
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
    return result


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
    result = asyncio.run(
        benchmark(
            api_url=api_url,
            model_id=args.model,
            tokenizer=tokenizer,
            input_requests=input_requests,
            disable_tqdm=args.disable_tqdm,
            disable_stream=args.disable_stream,
            disable_ignore_eos=args.disable_ignore_eos,
            slo_ttft=args.slo_ttft,
            slo_tpot=args.slo_tpot,
            extra_request_body=extra_request_body,
            args=args,
            offline=args.offline,
        )
    )
    save_results(args, result)


def save_results(args, result):
    """Save benchmark results to file"""
    output_file = args.output_file

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
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL file name."
    )
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
    assert not args.disable_stream
    assert args.output_file is not None
    if args.constant_rate is not None and args.constant_rate == 0.0:
        print("Constant rate equals 0. Exit")
        sys.exit(0)
    run_benchmark(args)


if __name__ == "__main__":
    main()