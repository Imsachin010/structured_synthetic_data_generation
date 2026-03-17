import json
import os
import argparse

def evaluate_metrics(filepath):
    """
    Evaluate validation metrics from a generated JSON file.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, "r") as f:
        data = json.load(f)

    total_queries = len(data)
    if total_queries == 0:
        print(f"No queries found in {filepath}")
        return

    valid_count = 0
    total_structural_score = 0
    total_alignment_score = 0
    total_repair_attempts = 0
    
    # Track failure modes for baseline
    failure_modes_summary = {}

    for item in data:
        # Check validity (usually stored in info error key if invalid)
        info = item.get("info", {})
        
        # Valid means no 'error' key or 'is_valid' was true somewhere
        # The schema enforcement marks error='validation_error'
        if "error" not in info:
            valid_count += 1
            
        # Collect failure modes if any
        modes = info.get("failure_modes", [])
        for mode in modes:
            failure_modes_summary[mode] = failure_modes_summary.get(mode, 0) + 1

        # Collect Scores
        total_structural_score += info.get("score", 0)
        total_alignment_score += info.get("alignment_score", 0)
        
        # Collect total repairs
        repairs = info.get("repair_attempts", {})
        if isinstance(repairs, dict):
            total_repair_attempts += repairs.get("json_repair_attempts", 0)
            total_repair_attempts += repairs.get("schema_repair_attempts", 0)

    # Compute Averages
    validity_rate = (valid_count / total_queries) * 100
    mean_structural_score = total_structural_score / total_queries
    mean_alignment_score = total_alignment_score / total_queries

    print(f"Metrics for: {os.path.basename(filepath)}")
    print(f"----------------------------------------")
    print(f"Total Queries:         {total_queries}")
    print(f"Valid Count:           {valid_count}")
    print(f"Validity Rate:         {validity_rate:.2f}%")
    
    if failure_modes_summary:
        print("\nFailure Mode Breakdown:")
        for mode, count in failure_modes_summary.items():
            print(f"  - {mode}: {count}")
    else:
        print(f"\nMean Structural Score: {mean_structural_score:.2f}")
        print(f"Mean Alignment Score:  {mean_alignment_score:.2f}")
        print(f"Total Repair Attempts: {total_repair_attempts}")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pipeline metrics.")
    parser.add_argument("filepath", type=str, help="Path to the JSON output file to analyze.")
    args = parser.parse_args()
    
    evaluate_metrics(args.filepath)
