import json
import os
from statistics import mean
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

INPUT_DIR = "experiments"
OUT_JSON = "experiments/multimodal_comparison_summary.json"
OUT_MD = "experiments/multimodal_comparison.md"


def summarize_stage(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    total = len(data)
    valid = 0
    structural_scores = []
    alignment_scores = []

    for entry in data:
        info = entry.get("info", {})
        if "error" not in info:
            valid += 1
            structural_scores.append(info.get("score", 0))
            alignment_scores.append(info.get("alignment_score", 0))

    return {
        "validity_rate": round(valid / total, 3) if total > 0 else 0,
        "avg_structural_score": round(mean(structural_scores), 3) if structural_scores else 0,
        "avg_alignment_score": round(mean(alignment_scores), 3) if alignment_scores else 0,
    }


def summarize():
    summary = {}

    for filename in os.listdir(INPUT_DIR):
        if filename.startswith("stage") and filename.endswith(".json"):
            stage_name = filename.replace(".json", "")
            filepath = os.path.join(INPUT_DIR, filename)
            summary[stage_name] = summarize_stage(filepath)

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_MD, "w") as f:
        f.write("# Multimodal Retrieval Comparison\n\n")
        f.write("| Stage | Validity | Avg Structural | Avg Alignment |\n")
        f.write("|-------|----------|----------------|---------------|\n")

        for stage, stats in summary.items():
            f.write(
                f"| {stage} | "
                f"{stats['validity_rate']} | "
                f"{stats['avg_structural_score']} | "
                f"{stats['avg_alignment_score']} |\n"
            )

    print("Multimodal comparison summary generated.")


if __name__ == "__main__":
    summarize()
