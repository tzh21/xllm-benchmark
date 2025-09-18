#!/usr/bin/env python3
"""
Script to select the hour with most significant fluctuation in request trace data.
Significant fluctuation is defined as large amplitude but low frequency.
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
import sys


def load_trace_data(file_path: str) -> List[Dict]:
    """Load trace data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}", file=sys.stderr)
    return data


def group_by_hour(data: List[Dict]) -> Dict[int, List[Dict]]:
    """Group trace data by hour based on timestamp (in milliseconds)."""
    hourly_data = defaultdict(list)

    for record in data:
        timestamp_ms = record['timestamp']
        # Convert milliseconds to hour bucket
        hour_bucket = int(timestamp_ms // (1000 * 3600))  # Convert to hours
        hourly_data[hour_bucket].append(record)

    return dict(hourly_data)


def calculate_request_rate_series(hour_data: List[Dict], window_size_ms: int = 60000) -> np.ndarray:
    """
    Calculate request rate time series for an hour.

    Args:
        hour_data: List of trace records for a specific hour
        window_size_ms: Window size in milliseconds for calculating request rate (default: 1 minute)

    Returns:
        Array of request counts per window
    """
    if not hour_data:
        return np.array([])

    # Sort by timestamp
    sorted_data = sorted(hour_data, key=lambda x: x['timestamp'])

    # Get time range
    min_time = sorted_data[0]['timestamp']
    max_time = sorted_data[-1]['timestamp']

    # Create time windows
    num_windows = int((max_time - min_time) / window_size_ms) + 1
    request_counts = np.zeros(num_windows)

    # Count requests per window
    for record in sorted_data:
        window_idx = int((record['timestamp'] - min_time) / window_size_ms)
        if 0 <= window_idx < num_windows:
            request_counts[window_idx] += 1

    return request_counts


def calculate_fluctuation_metrics(time_series: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate fluctuation metrics for a time series.

    Returns:
        Tuple of (amplitude, frequency, significance_score)
    """
    if len(time_series) < 2:
        return 0.0, 0.0, 0.0

    # Amplitude: range or standard deviation
    amplitude = np.std(time_series) if len(time_series) > 0 else 0

    # Alternative amplitude measure: peak-to-peak
    peak_to_peak = np.max(time_series) - np.min(time_series) if len(time_series) > 0 else 0

    # Frequency: count zero-crossings of detrended series
    detrended = time_series - np.mean(time_series)
    zero_crossings = 0
    for i in range(1, len(detrended)):
        if detrended[i-1] * detrended[i] < 0:
            zero_crossings += 1

    # Normalize frequency (lower is better for our use case)
    frequency = zero_crossings / len(time_series) if len(time_series) > 0 else 0

    # Calculate number of significant peaks/valleys
    if len(time_series) >= 3:
        peaks = 0
        valleys = 0
        for i in range(1, len(time_series) - 1):
            if time_series[i] > time_series[i-1] and time_series[i] > time_series[i+1]:
                peaks += 1
            elif time_series[i] < time_series[i-1] and time_series[i] < time_series[i+1]:
                valleys += 1

        # Use total direction changes as frequency metric
        direction_changes = peaks + valleys
        normalized_frequency = direction_changes / len(time_series)
    else:
        normalized_frequency = 0

    # Significance score: high amplitude with low frequency
    # We want large amplitude but low frequency (few direction changes)
    if normalized_frequency > 0:
        significance_score = amplitude / normalized_frequency
    else:
        significance_score = amplitude * 10  # High score if no frequency (constant high variance)

    # Alternative scoring using peak-to-peak
    if normalized_frequency > 0:
        alt_significance = peak_to_peak / normalized_frequency
    else:
        alt_significance = peak_to_peak * 10

    # Use the maximum of both scoring methods
    final_score = max(significance_score, alt_significance * 0.5)

    return amplitude, normalized_frequency, final_score


def find_most_significant_hour(hourly_data: Dict[int, List[Dict]], window_size_ms: int = 60000) -> Tuple[int, float, Dict]:
    """
    Find the hour with the most significant fluctuation.

    Returns:
        Tuple of (hour_bucket, significance_score, metrics_dict)
    """
    best_hour = -1
    best_score = -1
    best_metrics = {}

    for hour_bucket, hour_data in hourly_data.items():
        # Calculate request rate time series
        time_series = calculate_request_rate_series(hour_data, window_size_ms)

        if len(time_series) < 2:
            continue

        # Calculate fluctuation metrics
        amplitude, frequency, significance = calculate_fluctuation_metrics(time_series)

        # Store metrics
        metrics = {
            'hour_bucket': hour_bucket,
            'hour_start_ms': hour_bucket * 3600 * 1000,
            'num_requests': len(hour_data),
            'num_windows': len(time_series),
            'amplitude': amplitude,
            'frequency': frequency,
            'significance_score': significance,
            'mean_request_rate': np.mean(time_series),
            'max_request_rate': np.max(time_series),
            'min_request_rate': np.min(time_series),
            'std_request_rate': np.std(time_series)
        }

        if significance > best_score:
            best_score = significance
            best_hour = hour_bucket
            best_metrics = metrics

    return best_hour, best_score, best_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Select the hour with most significant fluctuation in request trace data'
    )
    parser.add_argument(
        'input_file',
        help='Path to the JSONL file containing trace data'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=60000,
        help='Window size in milliseconds for calculating request rate (default: 60000ms = 1 minute)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed metrics for each hour'
    )

    args = parser.parse_args()

    # Load trace data
    print(f"Loading trace data from {args.input_file}...")
    data = load_trace_data(args.input_file)
    print(f"Loaded {len(data)} trace records")

    # Group by hour
    hourly_data = group_by_hour(data)
    print(f"Data spans {len(hourly_data)} hours")

    # Find most significant hour
    best_hour, best_score, best_metrics = find_most_significant_hour(
        hourly_data, args.window_size
    )

    if best_hour == -1:
        print("No valid hour found with sufficient data")
        return

    # Print results
    print("\n" + "="*60)
    print("MOST SIGNIFICANT FLUCTUATION HOUR")
    print("="*60)

    # Convert hour bucket to readable format
    hour_start_ms = best_metrics['hour_start_ms']
    hour_start_dt = datetime.fromtimestamp(hour_start_ms / 1000)
    hour_end_dt = datetime.fromtimestamp((hour_start_ms + 3600000) / 1000)

    print(f"Hour: {hour_start_dt.strftime('%Y-%m-%d %H:%M:%S')} - {hour_end_dt.strftime('%H:%M:%S')}")
    print(f"Hour Bucket ID: {best_hour}")
    print(f"Significance Score: {best_score:.2f}")
    print(f"\nMetrics:")
    print(f"  - Total Requests: {best_metrics['num_requests']}")
    print(f"  - Amplitude (std dev): {best_metrics['amplitude']:.2f}")
    print(f"  - Frequency (direction changes): {best_metrics['frequency']:.4f}")
    print(f"  - Mean Request Rate: {best_metrics['mean_request_rate']:.2f} req/window")
    print(f"  - Max Request Rate: {best_metrics['max_request_rate']:.0f} req/window")
    print(f"  - Min Request Rate: {best_metrics['min_request_rate']:.0f} req/window")
    print(f"  - Std Dev Request Rate: {best_metrics['std_request_rate']:.2f}")

    if args.verbose:
        print("\n" + "="*60)
        print("ALL HOURS METRICS (sorted by significance)")
        print("="*60)

        all_metrics = []
        for hour_bucket, hour_data in hourly_data.items():
            time_series = calculate_request_rate_series(hour_data, args.window_size)
            if len(time_series) >= 2:
                amplitude, frequency, significance = calculate_fluctuation_metrics(time_series)
                all_metrics.append({
                    'hour': hour_bucket,
                    'requests': len(hour_data),
                    'amplitude': amplitude,
                    'frequency': frequency,
                    'significance': significance
                })

        # Sort by significance
        all_metrics.sort(key=lambda x: x['significance'], reverse=True)

        for i, metrics in enumerate(all_metrics[:10], 1):
            hour_ms = metrics['hour'] * 3600 * 1000
            hour_dt = datetime.fromtimestamp(hour_ms / 1000)
            print(f"{i}. Hour {hour_dt.strftime('%Y-%m-%d %H:%M')}: "
                  f"Score={metrics['significance']:.2f}, "
                  f"Amp={metrics['amplitude']:.2f}, "
                  f"Freq={metrics['frequency']:.4f}, "
                  f"Requests={metrics['requests']}")


if __name__ == '__main__':
    main()