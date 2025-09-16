#!/usr/bin/env python3
"""
Request Flow Chart Visualization Script
Reads JSONL trace data and generates request flow charts showing:
- Request arrival patterns over time
- Input/output token distributions
- Request rate over time
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns


def load_trace_data(
    file_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    max_requests: Optional[int] = None
) -> List[Dict]:
    """
    Load trace data from JSONL file with optional filtering.

    Args:
        file_path: Path to the JSONL file
        start_time: Filter requests after this timestamp (ms)
        end_time: Filter requests before this timestamp (ms)
        max_requests: Maximum number of requests to load

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
                    break

                data.append(entry)

                # Check if we've reached the requested number of requests
                if max_requests is not None and len(data) >= max_requests:
                    break

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
                continue

    return data


def extract_metrics(data: List[Dict]) -> Tuple[List[float], List[int], List[int]]:
    """
    Extract timestamps, input lengths, and output lengths from trace data.

    Args:
        data: List of trace entries

    Returns:
        Tuple of (timestamps, input_lengths, output_lengths)
    """
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

    return timestamps, input_lengths, output_lengths


def generate_filename(file_path: str, num_requests: int, start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> str:
    """
    Generate filename for the flow chart based on dataset characteristics.

    Args:
        file_path: Original dataset file path
        num_requests: Number of requests in the dataset
        start_time: Start time filter if applied
        end_time: End time filter if applied

    Returns:
        Generated filename
    """
    # Extract base filename
    base_name = Path(file_path).stem

    # Build filename components
    components = [base_name]
    components.append(f"n{num_requests}")

    if start_time is not None:
        components.append(f"start{int(start_time)}")
    if end_time is not None:
        components.append(f"end{int(end_time)}")

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    components.append(timestamp)

    return "_".join(components) + "_flow.png"


def create_request_flow_chart(timestamps: List[float], input_lengths: List[int],
                            output_lengths: List[int], title: str) -> plt.Figure:
    """
    Create a comprehensive request flow chart.

    Args:
        timestamps: List of request timestamps (ms)
        input_lengths: List of input token lengths
        output_lengths: List of output token lengths
        title: Chart title

    Returns:
        matplotlib Figure object
    """
    # Convert timestamps to datetime objects for better plotting
    if timestamps:
        # Convert from milliseconds to seconds, then to datetime
        start_timestamp = min(timestamps) / 1000
        datetime_timestamps = [datetime.fromtimestamp(start_timestamp + (ts - min(timestamps)) / 1000)
                              for ts in timestamps]
    else:
        datetime_timestamps = []

    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Subplot 1: Request arrival pattern over time
    ax1 = axes[0, 0]
    if datetime_timestamps:
        # Create request arrival timeline
        y_pos = np.ones(len(datetime_timestamps))  # All requests at same y-level
        colors = plt.cm.viridis(np.linspace(0, 1, len(datetime_timestamps)))

        ax1.scatter(datetime_timestamps, y_pos, c=colors, alpha=0.6, s=20)
        ax1.set_ylabel('Requests')
        ax1.set_xlabel('Time')
        ax1.set_title('Request Arrival Pattern')
        ax1.set_ylim(0.5, 1.5)
        ax1.set_yticks([1])
        ax1.set_yticklabels(['Arrivals'])

        # Format x-axis for time
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(datetime_timestamps)//10)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax1.text(0.5, 0.5, 'No timestamp data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Request Arrival Pattern')

    # Subplot 2: Request rate over time (requests per time window)
    ax2 = axes[0, 1]
    if timestamps and len(timestamps) > 1:
        # Calculate request rate in time windows
        time_span = (max(timestamps) - min(timestamps)) / 1000  # in seconds
        num_windows = min(50, len(timestamps) // 10)  # Reasonable number of windows

        if num_windows > 1:
            window_size = time_span / num_windows
            window_centers = []
            request_rates = []

            for i in range(num_windows):
                window_start = min(timestamps) + i * window_size * 1000
                window_end = window_start + window_size * 1000

                # Count requests in this window
                requests_in_window = sum(1 for ts in timestamps if window_start <= ts < window_end)
                rate = requests_in_window / window_size  # requests per second

                window_centers.append(datetime.fromtimestamp((window_start + window_end) / 2000))
                request_rates.append(rate)

            ax2.plot(window_centers, request_rates, 'b-', linewidth=2, marker='o', markersize=4)
            ax2.set_ylabel('Requests/second')
            ax2.set_xlabel('Time')
            ax2.set_title('Request Rate Over Time')
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for rate calculation', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'Insufficient timestamp data', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Request Rate Over Time')

    # Subplot 3: Token length distributions
    ax3 = axes[1, 0]
    if input_lengths and output_lengths:
        # Create violin plots for token distributions
        data_to_plot = []
        labels = []

        if input_lengths:
            data_to_plot.append(input_lengths)
            labels.append('Input Tokens')
        if output_lengths:
            data_to_plot.append(output_lengths)
            labels.append('Output Tokens')

        violin_parts = ax3.violinplot(data_to_plot, positions=range(1, len(data_to_plot) + 1),
                                     showmeans=True, showmedians=True)

        # Customize violin plot colors
        colors = ['lightblue', 'lightcoral']
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)

        ax3.set_xticks(range(1, len(labels) + 1))
        ax3.set_xticklabels(labels)
        ax3.set_ylabel('Token Count')
        ax3.set_title('Token Length Distributions')
        ax3.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = []
        if input_lengths:
            stats_text.append(f"Input: μ={np.mean(input_lengths):.0f}, σ={np.std(input_lengths):.0f}")
        if output_lengths:
            stats_text.append(f"Output: μ={np.mean(output_lengths):.0f}, σ={np.std(output_lengths):.0f}")

        ax3.text(0.02, 0.98, '\n'.join(stats_text), transform=ax3.transAxes,
                verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No token length data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Token Length Distributions')

    # Subplot 4: Request flow timeline with token sizes
    ax4 = axes[1, 1]
    if datetime_timestamps and input_lengths and output_lengths:
        # Create a timeline showing request flows with token sizes
        for i, (dt, input_len, output_len) in enumerate(zip(datetime_timestamps, input_lengths, output_lengths)):
            # Normalize token lengths for visualization
            max_input = max(input_lengths) if input_lengths else 1
            max_output = max(output_lengths) if output_lengths else 1

            input_height = (input_len / max_input) * 0.4  # Scale to 0.4 max height
            output_height = (output_len / max_output) * 0.4

            # Draw input and output bars
            ax4.bar(dt, input_height, bottom=0.5, width=timedelta(seconds=10),
                   color='lightblue', alpha=0.7, label='Input' if i == 0 else "")
            ax4.bar(dt, -output_height, bottom=0.5, width=timedelta(seconds=10),
                   color='lightcoral', alpha=0.7, label='Output' if i == 0 else "")

        ax4.axhline(y=0.5, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax4.set_ylabel('Token Size (normalized)')
        ax4.set_xlabel('Time')
        ax4.set_title('Request Flow with Token Sizes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Format x-axis
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for flow timeline', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Request Flow with Token Sizes')

    # Adjust layout
    plt.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Generate request flow charts from trace datasets',
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
        '--max-requests',
        type=int,
        default=None,
        help='Maximum number of requests to visualize'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='local/flow',
        help='Output directory for flow charts'
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.trace_file).exists():
        print(f"Error: File '{args.trace_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Load data
    print(f"Loading trace data from: {args.trace_file}")
    data = load_trace_data(
        args.trace_file,
        start_time=args.start_time,
        end_time=args.end_time,
        max_requests=args.max_requests
    )

    if not data:
        print("No data found matching the filters", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data)} requests")

    # Extract metrics
    timestamps, input_lengths, output_lengths = extract_metrics(data)

    # Create chart title with dataset info
    dataset_name = Path(args.trace_file).stem
    title = f"Request Flow Analysis: {dataset_name}\n"
    title += f"Requests: {len(data):,}"

    if timestamps:
        duration = (max(timestamps) - min(timestamps)) / 1000  # seconds
        title += f" | Duration: {duration:.1f}s"
        title += f" | Rate: {len(data)/duration:.2f} req/s"

    if input_lengths and output_lengths:
        title += f"\nTokens - Input: {sum(input_lengths):,}, Output: {sum(output_lengths):,}"

    # Create the flow chart
    print("Generating request flow chart...")
    fig = create_request_flow_chart(timestamps, input_lengths, output_lengths, title)

    # Save the chart
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = generate_filename(args.trace_file, len(data), args.start_time, args.end_time)
    output_path = output_dir / filename

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Flow chart saved to: {output_path}")

    # Clean up
    plt.close(fig)


if __name__ == '__main__':
    main()