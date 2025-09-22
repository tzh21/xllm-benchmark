#!/usr/bin/env python3
"""
Script to compute statistics from request trace datasets.
Analyzes timestamp, input_length, and output_length distributions.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta


def load_trace_data(
    file_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    num_requests: Optional[int] = None
) -> List[Dict]:
    """
    Load trace data from JSONL file with optional filtering.

    Args:
        file_path: Path to the JSONL file
        start_time: Filter requests after this timestamp (milliseconds)
        end_time: Filter requests before this timestamp (milliseconds)
        num_requests: Maximum number of requests to load

    Returns:
        List of request dictionaries
    """
    data = []

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())

                # Check timestamp filters
                timestamp = entry.get('timestamp', 0)

                if start_time is not None and timestamp < start_time:
                    continue

                if end_time is not None and timestamp > end_time:
                    # If we've reached the end time, stop reading
                    break

                data.append(entry)

                # Check if we've reached the requested number of requests
                if num_requests is not None and len(data) >= num_requests:
                    break

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
                continue

    # Apply the constraint: if end_time and num_requests don't match, use the smaller
    if end_time is not None and num_requests is not None:
        # The data is already filtered by both constraints above
        pass

    return data


def compute_statistics(values: List[float], name: str) -> Dict:
    """
    Compute comprehensive statistics for a list of values.

    Args:
        values: List of numeric values
        name: Name of the metric

    Returns:
        Dictionary of statistics
    """
    if not values:
        return {
            'name': name,
            'count': 0,
            'statistics': 'No data available'
        }

    values_array = np.array(values)

    stats = {
        'name': name,
        'count': len(values),
        'mean': np.mean(values_array),
        'median': np.median(values_array),
        'std': np.std(values_array),
        'min': np.min(values_array),
        'max': np.max(values_array),
        'p25': np.percentile(values_array, 25),
        'p50': np.percentile(values_array, 50),  # Same as median
        'p75': np.percentile(values_array, 75),
        'p90': np.percentile(values_array, 90),
        'p95': np.percentile(values_array, 95),
        'p99': np.percentile(values_array, 99),
        'sum': np.sum(values_array),
    }

    return stats


def analyze_timestamps(timestamps: List[float]) -> Dict:
    """
    Analyze timestamp patterns including request rate and inter-arrival times.

    Args:
        timestamps: List of timestamps in milliseconds

    Returns:
        Dictionary of timestamp-specific statistics
    """
    if len(timestamps) < 2:
        return {
            'duration_seconds': 0,
            'request_rate': 0,
            'inter_arrival_stats': None
        }

    sorted_timestamps = sorted(timestamps)
    duration_ms = sorted_timestamps[-1] - sorted_timestamps[0]
    duration_seconds = duration_ms / 1000.0

    # Calculate inter-arrival times
    inter_arrivals = []
    for i in range(1, len(sorted_timestamps)):
        inter_arrivals.append(sorted_timestamps[i] - sorted_timestamps[i-1])

    stats = {
        'duration_seconds': duration_seconds,
        'start_time_ms': sorted_timestamps[0],
        'end_time_ms': sorted_timestamps[-1],
        'start_time_readable': datetime.fromtimestamp(sorted_timestamps[0]/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        'end_time_readable': datetime.fromtimestamp(sorted_timestamps[-1]/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        'request_rate': len(timestamps) / duration_seconds if duration_seconds > 0 else 0,
        'inter_arrival_stats': compute_statistics(inter_arrivals, 'Inter-arrival Time (ms)')
    }

    return stats


def print_statistics(stats: Dict, indent: int = 0):
    """
    Pretty print statistics dictionary.

    Args:
        stats: Statistics dictionary
        indent: Indentation level
    """
    prefix = "  " * indent

    for key, value in stats.items():
        if key == 'name':
            continue
        elif key == 'inter_arrival_stats' and value:
            print(f"{prefix}Inter-arrival Time Statistics:")
            print_statistics(value, indent + 1)
        elif isinstance(value, float):
            # Format based on the magnitude of the value
            if abs(value) < 0.01 or abs(value) > 1000000:
                print(f"{prefix}{key:20s}: {value:.2e}")
            else:
                print(f"{prefix}{key:20s}: {value:.2f}")
        elif isinstance(value, str):
            print(f"{prefix}{key:20s}: {value}")
        else:
            print(f"{prefix}{key:20s}: {value}")


def generate_output_filename(args) -> str:
    """
    Generate output filename based on input arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Filename string based on input parameters
    """
    # Extract base filename from trace file path
    trace_filename = Path(args.trace_file).stem

    # Build filename components
    filename_parts = [trace_filename]

    if args.start_time is not None:
        filename_parts.append(f"start{int(args.start_time)}ms")

    if args.end_time is not None:
        filename_parts.append(f"end{int(args.end_time)}ms")

    if args.num_requests is not None:
        filename_parts.append(f"n{args.num_requests}")

    # Add timestamp to make filename unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts.append(timestamp)

    return "_".join(filename_parts) + ".txt"


def save_statistics_to_file(filename: str, content: str):
    """
    Save statistics content to a file in local/dataset-stats directory.

    Args:
        filename: Name of the output file
        content: Statistics content to save
    """
    output_dir = Path("local/dataset-stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"\nStatistics saved to: {output_path}")


class StatsCapture:
    """
    Context manager to capture print output for saving to file.
    """
    def __init__(self):
        self.content = []
        self.original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout

    def write(self, text):
        self.content.append(text)
        self.original_stdout.write(text)

    def flush(self):
        self.original_stdout.flush()

    def get_content(self):
        return ''.join(self.content)


def main():
    parser = argparse.ArgumentParser(
        description='Compute statistics from request trace datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'trace_file',
        type=str,
        help='Path to the trace JSONL file'
    )

    parser.add_argument(
        '--start-time',
        type=float,
        default=None,
        help='Start time filter (milliseconds since epoch)'
    )

    parser.add_argument(
        '--end-time',
        type=float,
        default=None,
        help='End time filter (milliseconds since epoch)'
    )

    parser.add_argument(
        '--num-requests',
        type=int,
        default=None,
        help='Maximum number of requests to analyze'
    )

    parser.add_argument(
        '--output-format',
        choices=['text', 'json'],
        default='text',
        help='Output format for statistics'
    )

    parser.add_argument(
        '--save-to-file',
        action='store_true',
        default=True,
        help='Save statistics to file in local/dataset-stats directory (default: True)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        default=False,
        help='Do not save statistics to file, only display on screen'
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.trace_file).exists():
        print(f"Error: File '{args.trace_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Load data with filters
    print(f"Loading trace data from: {args.trace_file}")

    if args.start_time:
        print(f"  Filter: start_time >= {args.start_time} milliseconds")
    if args.end_time:
        print(f"  Filter: end_time <= {args.end_time} milliseconds")
    if args.num_requests:
        print(f"  Filter: max {args.num_requests} requests")

    # Use milliseconds directly (no conversion needed)
    data = load_trace_data(
        args.trace_file,
        start_time=args.start_time,
        end_time=args.end_time,
        num_requests=args.num_requests
    )

    if not data:
        print("No data found matching the filters", file=sys.stderr)
        sys.exit(1)

    # Determine if we should save to file
    should_save = args.save_to_file and not args.no_save

    def run_analysis():
        """Run the analysis and generate output"""
        print(f"\nLoaded {len(data)} requests\n")
        print("=" * 60)

        # Extract metrics
        timestamps = []
        input_lengths = []
        output_lengths = []

        for entry in data:
            if 'timestamp' in entry:
                timestamps.append(float(entry['timestamp']))
            if 'input_length' in entry:
                input_lengths.append(int(entry['input_length']))
            if 'output_length' in entry:
                output_lengths.append(int(entry['output_length']))

        # Compute and display timestamp statistics
        if timestamps:
            print("\nTIMESTAMP ANALYSIS")
            print("-" * 40)
            timestamp_stats = analyze_timestamps(timestamps)
            print_statistics(timestamp_stats)

            print("\n  Timestamp Distribution:")
            ts_stats = compute_statistics(timestamps, 'Timestamp')
            print_statistics(ts_stats, indent=1)

        # Compute and display input_length statistics
        if input_lengths:
            print("\nINPUT LENGTH STATISTICS")
            print("-" * 40)
            input_stats = compute_statistics(input_lengths, 'Input Length')
            print_statistics(input_stats)

        # Compute and display output_length statistics
        if output_lengths:
            print("\nOUTPUT LENGTH STATISTICS")
            print("-" * 40)
            output_stats = compute_statistics(output_lengths, 'Output Length')
            print_statistics(output_stats)

        # Combined token statistics
        if input_lengths and output_lengths:
            print("\nCOMBINED TOKEN STATISTICS")
            print("-" * 40)

            total_tokens = [i + o for i, o in zip(input_lengths, output_lengths)]
            combined_stats = compute_statistics(total_tokens, 'Total Tokens per Request')
            print_statistics(combined_stats)

            # Ratio statistics
            ratios = []
            for i, o in zip(input_lengths, output_lengths):
                if i > 0:
                    ratios.append(o / i)

            if ratios:
                print("\n  Output/Input Ratio:")
                ratio_stats = compute_statistics(ratios, 'Output/Input Ratio')
                print_statistics(ratio_stats, indent=1)

        # Overall summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("-" * 40)
        print(f"Total requests analyzed: {len(data)}")

        if timestamps:
            print(f"Time span: {timestamp_stats['duration_seconds']:.2f} seconds")
            print(f"Average request rate: {timestamp_stats['request_rate']:.2f} req/s")

        if input_lengths:
            print(f"Total input tokens: {sum(input_lengths):,}")

        if output_lengths:
            print(f"Total output tokens: {sum(output_lengths):,}")

        if input_lengths and output_lengths:
            print(f"Total tokens: {sum(input_lengths) + sum(output_lengths):,}")

        print("=" * 60)

    # Run analysis with or without capturing
    if should_save:
        with StatsCapture() as capture:
            run_analysis()
        # Save to file
        filename = generate_output_filename(args)
        content = capture.get_content()
        save_statistics_to_file(filename, content)
    else:
        run_analysis()


if __name__ == '__main__':
    main()