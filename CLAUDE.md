# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a benchmarking and testing framework for various LLM inference engines, specifically designed for performance analysis and comparison. The repository focuses on testing xLLM, vLLM, and other inference backends using different datasets and trace patterns.

## Core Architecture

### Main Components

- **test.py**: Primary benchmarking script that orchestrates async requests to LLM inference endpoints
- **metric.py**: Contains data structures and utilities for measuring performance metrics (TTFT, TPOT, ITL, latency)
- **traffic.py**: Visualizes request traffic patterns and generates plots for analysis
- **utils.py**: Contains utility functions (currently minimal)

### Key Data Structures

- `RequestFuncInput`: Contains prompt, API URL, lengths, model info, and request metadata
- `RequestFuncOutput`: Stores response data including generated text, latency metrics, and success status
- `BenchmarkMetrics`: Comprehensive performance metrics including throughput, latency percentiles, and timing data

### Dataset Support

The framework supports multiple dataset types:
- **sharegpt**: Standard ShareGPT conversation dataset
- **random**: Randomly generated prompts with configurable lengths
- **generated-shared-prefix**: Shared system prompts with unique questions
- **burstgpt**: Trace-based dataset from BurstGPT research
- **mooncake**: Trace data from Mooncake project
- **azure_code/azure_conv**: Azure LLM inference traces for code and conversation workloads

## Common Commands

### Running Benchmarks

**xLLM Backend Testing:**
```bash
./xllm-test.sh [trace_start_time]
```

**vLLM Backend Testing:**
```bash
./vllm-test.sh
```

**Manual Test Execution:**
```bash
python test.py \
    --backend xllm \
    --base-url http://127.0.0.1:27712 \
    --dataset-name azure_conv \
    --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --trace-path /path/to/azure_conv.jsonl \
    --model /path/to/model \
    --num-prompts 7000
```

### System Monitoring

**Start NPU monitoring:**
```bash
./aicore.sh
```

**Simple NPU monitoring:**
```bash
./npu.sh
```

### Data Preparation

**Download and preprocess datasets:**
```bash
./download_datasets.sh
```

**API Health Check:**
```bash
./check.sh
```

### Results Analysis

**Generate traffic visualization:**
```bash
python traffic.py
```

## Dataset Configuration

### Trace Datasets
When using trace-based datasets (burstgpt, mooncake, azure_*), specify:
- `--trace-path`: Path to trace JSONL file
- `--trace-scale`: Time scaling factor (e.g., 100 = 100x faster replay)
- `--trace-start-time`/`--trace-end-time`: Time window selection

### Request Rate Control
- `--request-rate`: Requests per second (use `inf` for batch mode)
- `--max-concurrency`: Limit concurrent requests
- `--multi` + `--request-rate-range`: Test multiple request rates

## Backend-Specific Notes

### xLLM
- Default port: 9811
- Endpoint: `/v1/completions`
- Supports streaming and ignore_eos options

### vLLM  
- Default port: 8000
- Endpoint: `/v1/completions`
- May require `--extra-request-body` to override model name

### Request Parameters
- All backends use OpenAI-compatible completion API
- Streaming enabled by default (disable with `--disable-stream`)
- EOS token ignoring enabled by default (disable with `--disable-ignore-eos`)

## Results Structure

Benchmark results are saved as JSONL files in `results/` directory with format:
`{backend}_{timestamp}_{num_prompts}_{dataset}_{additional_params}.jsonl`

Each result contains comprehensive metrics including:
- Throughput metrics (requests/sec, tokens/sec)
- Latency percentiles (mean, median, P90, P99)
- Timing breakdowns (TTFT, TPOT, ITL)
- Request-level data (timestamps, latencies)

## Log Structure

- `log/test/`: Test execution logs with timestamps
- `log/aicore/`: NPU monitoring data (CSV format)
- Logs use format: `MMDD-HHMMSS.log` or `aicore_usage_MMDD_HHMMSS_npuX.csv`

## Development Notes

- Scripts expect models and datasets in `/export/home/tangzihan/` hierarchy
- NPU monitoring requires sudo privileges for `npu-smi` commands
- Traffic visualization uses Chinese font configuration (Source Han Sans SC)
- Exception handling exits immediately on request failures to prevent hanging