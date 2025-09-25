#!/usr/bin/env python3
import json
import argparse


def calculate_violation_rates(json_file_path, slo_ttft, slo_tpot):
    """Calculate TTFT and TPOT violation rates from a benchmark JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    requests = data.get('requests', [])
    if not requests:
        return 0.0, 0.0, 0.0

    ttft_violations = 0
    tpot_violations = 0
    ttft_or_tpot_violations = 0
    total_requests = len(requests)

    for request in requests:
        if not request.get('success', False):
            continue

        ttft_ms = request.get('ttft_ms', 0)
        tpot_ms = request.get('tpot_ms', 0)

        ttft_violated = ttft_ms > slo_ttft
        tpot_violated = tpot_ms > slo_tpot

        if ttft_violated:
            ttft_violations += 1
        if tpot_violated:
            tpot_violations += 1
        if ttft_violated or tpot_violated:
            ttft_or_tpot_violations += 1

    ttft_violation_rate = ttft_violations / total_requests
    tpot_violation_rate = tpot_violations / total_requests
    ttft_or_tpot_violation_rate = ttft_or_tpot_violations / total_requests

    return ttft_violation_rate, tpot_violation_rate, ttft_or_tpot_violation_rate


def main():
    parser = argparse.ArgumentParser(description='Calculate violation rates from benchmark JSON file')
    parser.add_argument('json_file', help='Path to the JSON file')
    parser.add_argument('--slo-ttft', type=float, default=10000.0, help='TTFT SLO threshold in ms (default: 10000.0)')
    parser.add_argument('--slo-tpot', type=float, default=75.0, help='TPOT SLO threshold in ms (default: 75.0)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    ttft_rate, tpot_rate, combined_rate = calculate_violation_rates(args.json_file, args.slo_ttft, args.slo_tpot)

    if args.verbose:
        print(f"TTFT violation rate: {ttft_rate:.6f}")
        print(f"TPOT violation rate: {tpot_rate:.6f}")
        print(f"TTFT or TPOT violation rate: {combined_rate:.6f}")
    else:
        print(f"{ttft_rate:.6f},{tpot_rate:.6f},{combined_rate:.6f}")


if __name__ == '__main__':
    main()