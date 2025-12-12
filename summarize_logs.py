#!/usr/bin/env python3
"""
Script to summarize evaluation results from log files into a CSV format.

Usage:
    python summarize_logs.py <folder_name>

Example:
    python summarize_logs.py resnet18_cifar10
"""

import os
import re
import csv
import argparse
import glob
from pathlib import Path


def find_latest_evaluation_file(run_folder, pattern="evaluation_20251113_*"):
    """Find the latest evaluation file matching the pattern in the run folder."""
    files = glob.glob(os.path.join(run_folder, f"{pattern}.txt"))
    if not files:
        return None
    # Sort by modification time and return the latest
    return max(files, key=os.path.getmtime)


def parse_evaluation_file(file_path):
    """
    Parse the evaluation file and extract the key metrics.

    Returns:
        dict: Dictionary with metrics as keys and values as floats
    """
    metrics = {
        'Baseline Uniform PR': None,
        'Baseline Gaussian PR': None,
        'PGD Robust Accuracy': None,
        'CW Robust Accuracy': None
    }

    if not os.path.exists(file_path):
        return metrics

    with open(file_path, 'r') as f:
        content = f.read()

    # Extract metrics from the Summary section using regex
    patterns = {
        'Baseline Uniform PR': r'Baseline Uniform PR:\s+([\d.]+)%',
        'Baseline Gaussian PR': r'Baseline Gaussian PR:\s+([\d.]+)%',
        'PGD Robust Accuracy': r'PGD Robust Accuracy:\s+([\d.]+)%',
        'CW Robust Accuracy': r'CW Robust Accuracy:\s+([\d.]+)%'
    }

    for metric_name, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[metric_name] = float(match.group(1))

    return metrics


def summarize_logs(folder_name, logs_dir="logs", pattern="evaluation_20251113_*"):
    """
    Summarize evaluation logs from multiple runs into a CSV file.

    Args:
        folder_name (str): Name of the folder containing run subfolders
        logs_dir (str): Base logs directory
        pattern (str): Pattern to match evaluation files
    """
    base_path = Path(logs_dir) / folder_name

    if not base_path.exists():
        print(f"Error: Folder {base_path} does not exist!")
        return

    # Find all run folders
    run_folders = sorted([d for d in base_path.iterdir()
                         if d.is_dir() and d.name.startswith('run_')],
                        key=lambda x: int(x.name.split('_')[1]))

    if not run_folders:
        print(f"Error: No run_* folders found in {base_path}")
        return

    print(f"Found {len(run_folders)} run folders: {[f.name for f in run_folders]}")

    # Collect data from all runs
    all_metrics = []
    run_names = []

    for run_folder in run_folders:
        run_name = run_folder.name
        eval_file = find_latest_evaluation_file(run_folder, pattern)

        if eval_file:
            print(f"Processing {run_name}: {os.path.basename(eval_file)}")
            metrics = parse_evaluation_file(eval_file)
            all_metrics.append(metrics)
            run_names.append(run_name)
        else:
            print(f"Warning: No evaluation file found in {run_name}")

    if not all_metrics:
        print("Error: No data to write!")
        return

    # Create CSV file
    csv_path = base_path / f"{folder_name}_summary.csv"

    # Metric names (rows in CSV)
    metric_names = [
        'Baseline Uniform PR',
        'Baseline Gaussian PR',
        'PGD Robust Accuracy',
        'CW Robust Accuracy'
    ]

    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header (columns: Metric, run_1, run_2, ...)
        header = ['Metric'] + run_names
        writer.writerow(header)

        # Write each metric as a row
        for metric_name in metric_names:
            row = [metric_name]
            for metrics_dict in all_metrics:
                value = metrics_dict.get(metric_name)
                if value is not None:
                    row.append(f"{value:.2f}")
                else:
                    row.append("N/A")
            writer.writerow(row)

    print(f"\nCSV file created: {csv_path}")
    print(f"\nSummary:")
    print(f"  Total runs processed: {len(run_names)}")
    print(f"  Metrics extracted: {len(metric_names)}")

    # Display the CSV content
    print(f"\nCSV content:")
    with open(csv_path, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(
        description='Summarize evaluation results from log files into CSV format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python summarize_logs.py resnet18_cifar10
  python summarize_logs.py resnet18_cifar100
  python summarize_logs.py resnet18_tinyimagenet
        '''
    )

    parser.add_argument(
        'folder_name',
        type=str,
        help='Name of the folder containing run subfolders (e.g., resnet18_cifar10)'
    )

    parser.add_argument(
        '--logs-dir',
        type=str,
        default='logs',
        help='Base logs directory (default: logs)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='evaluation_20251113_*',
        help='Pattern to match evaluation files (default: evaluation_20251113_*)'
    )

    args = parser.parse_args()

    summarize_logs(args.folder_name, args.logs_dir, args.pattern)


if __name__ == '__main__':
    main()
