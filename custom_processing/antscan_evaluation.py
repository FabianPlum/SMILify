import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_mesh_metrics(input_folder):
    """
    Analyze mesh metrics from JSON files in the input folder and its subfolders.

    Args:
        input_folder (str): Path to the input folder containing JSON files.

    Returns:
        tuple: A tuple containing two dictionaries:
            - metrics (dict): Mesh metrics data.
            - file_paths (dict): File paths for each metric.
    """
    metrics = defaultdict(list)
    file_paths = defaultdict(list)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    for metric in [
                        "processed_hole_count",
                        "processed_face_size_cov",
                        "processed_mesh_smoothness",
                    ]:
                        if metric in data:
                            metrics[metric].append(data[metric])
                            file_paths[metric].append(file_path)

    return metrics, file_paths


def plot_histograms(metrics, output_dir):
    """
    Plot histograms for each metric, highlighting mean, standard deviation, and 80th percentile.

    Args:
        metrics (dict): Dictionary containing metric data.
        output_dir (str): Directory to save the histogram plots.
    """
    for metric, values in metrics.items():
        plt.figure(figsize=(12, 7))
        
        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values)
        percentile_80 = np.percentile(values, 80)
        
        # Plot histogram
        plt.hist(values, bins=30, edgecolor="black", alpha=0.7)
        
        # Add vertical lines for mean, std, and 80th percentile
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
        plt.axvline(mean + std, color='g', linestyle='dashed', linewidth=2, label=f'Mean + Std: {mean+std:.2f}')
        plt.axvline(mean - std, color='g', linestyle='dashed', linewidth=2, label=f'Mean - Std: {mean-std:.2f}')
        plt.axvline(percentile_80, color='b', linestyle='dashed', linewidth=2, label=f'80th Percentile: {percentile_80:.2f}')
        
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.legend() 
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_histogram.png"))
        plt.close()


def calculate_statistics(metrics):
    """
    Calculate statistics for each metric.

    Args:
        metrics (dict): Dictionary containing metric data.

    Returns:
        dict: Dictionary containing statistics for each metric.
    """
    stats = {}
    for metric, values in metrics.items():
        stats[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }
    return stats


def get_high_quality_scans(metrics, file_paths, thresholds):
    """
    Get high-quality scans based on the given thresholds.

    Args:
        metrics (dict): Dictionary containing metric data.
        file_paths (dict): Dictionary containing file paths for each metric.
        thresholds (dict): Dictionary containing threshold values for each metric.

    Returns:
        list: List of file paths for high-quality scans.
    """
    high_quality_scans = set(file_paths["processed_hole_count"])

    for metric, threshold in thresholds.items():
        high_quality_scans &= set(
            [
                path
                for path, value in zip(file_paths[metric], metrics[metric])
                if value <= threshold
            ]
        )

    return list(high_quality_scans)


def main():
    """
    Main function to run the mesh quality analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze mesh quality metrics from JSON files."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="custom_processing/antscan_data",
        help="Input folder containing subfolders with JSON files (default: custom_processing/antscan_data)",
    )
    parser.add_argument(
        "--hole_count_threshold",
        type=float,
        default=5000,
        help="Maximum allowed hole count (default: infinity)",
    )
    parser.add_argument(
        "--face_size_cov_threshold",
        type=float,
        default=2,
        help="Maximum allowed face size coefficient of variation (default: infinity)",
    )
    parser.add_argument(
        "--mesh_smoothness_threshold",
        type=float,
        default=40,
        help="Maximum allowed mesh smoothness value (default: infinity)",
    )
    args = parser.parse_args()

    # Create the output directory
    output_dir = os.path.join("custom_processing", "antscan_stats")
    os.makedirs(output_dir, exist_ok=True)

    # Print the absolute path of the current working directory
    current_dir = os.path.abspath(os.getcwd())
    print(f"Current working directory: {current_dir}")

    # Get the absolute path of the input folder
    input_folder_abs = os.path.abspath(os.path.join(current_dir, args.input_folder))
    print(f"Input folder: {input_folder_abs}")

    # Check if the input folder exists
    if not os.path.exists(input_folder_abs):
        print(f"Error: The input folder '{input_folder_abs}' does not exist.")
        return

    # Count the number of scans (subfolders)
    scan_count = sum(1 for entry in os.scandir(input_folder_abs) if entry.is_dir())
    print(f"Number of scans found: {scan_count}")

    metrics, file_paths = analyze_mesh_metrics(input_folder_abs)
    plot_histograms(metrics, output_dir)
    stats = calculate_statistics(metrics)

    # Use user-defined thresholds
    thresholds = {
        "processed_hole_count": args.hole_count_threshold,
        "processed_face_size_cov": args.face_size_cov_threshold,
        "processed_mesh_smoothness": args.mesh_smoothness_threshold,
    }

    high_quality_scans = get_high_quality_scans(metrics, file_paths, thresholds)

    # Prepare data for JSON output
    output_data = {
        "statistics": stats,
        "thresholds": thresholds,
        "total_scans": scan_count,
        "high_quality_scans_count": len(high_quality_scans),
        "high_quality_scan_paths": high_quality_scans,
    }

    # Write to JSON file
    json_output_path = os.path.join(output_dir, "antscan_processing_stats.json")
    with open(json_output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Statistics and high-quality scan paths written to {json_output_path}")
    print(f"Number of high-quality scans: {len(high_quality_scans)}")


if __name__ == "__main__":
    main()