#!/usr/bin/env python3
"""
汇总多个模型的评估报告，生成 Markdown 对比表。

Usage:
    python compare_results.py --results-dir results/ --output results/comparison_summary.md
"""

import argparse
import json
import os
from typing import Dict, List


def load_reports(results_dir: str) -> Dict[str, Dict]:
    reports = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_report.json"):
            model_name = fname.replace("_report.json", "")
            path = os.path.join(results_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                reports[model_name] = json.load(f)
    return reports


def generate_markdown(reports: Dict[str, Dict]) -> str:
    models = list(reports.keys())
    if not models:
        return "No reports found.\n"

    lines = []
    lines.append("# MemGUI-3K Offline Evaluation: Model Comparison\n")
    lines.append(f"Models: {', '.join(models)}\n")
    lines.append(f"Steps: {reports[models[0]].get('total_steps', '?')}  |  "
                 f"Trajectories: {reports[models[0]].get('total_trajectories', '?')}\n")

    # ── Overall ──
    lines.append("\n## Overall Action Matching\n")
    header = f"| Metric | " + " | ".join(models) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(models)) + "|"
    lines.append(header)
    lines.append(sep)

    for metric, key in [("Type Accuracy", "type_acc"), ("Match Accuracy", "match_acc")]:
        row = f"| {metric} |"
        for m in models:
            val = reports[m].get("overall", {}).get(key, 0)
            row += f" {val:.2f}% |"
        lines.append(row)

    # ── UI Actions ──
    lines.append("\n## UI Actions\n")
    lines.append(header)
    lines.append(sep)
    for metric, key in [("Type Accuracy", "type_acc"), ("Match Accuracy", "match_acc")]:
        row = f"| UI {metric} |"
        for m in models:
            val = reports[m].get("ui_actions", {}).get(key, 0)
            row += f" {val:.2f}% |"
        lines.append(row)

    # Per-action breakdown
    all_actions = set()
    for m in models:
        all_actions.update(reports[m].get("ui_actions", {}).get("by_action", {}).keys())

    if all_actions:
        lines.append("\n### UI Actions by Type\n")
        lines.append(f"| Action | " + " | ".join(f"{m} (match%)" for m in models) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(models)) + "|")
        for action in sorted(all_actions):
            row = f"| {action} |"
            for m in models:
                s = reports[m].get("ui_actions", {}).get("by_action", {}).get(action, {})
                val = s.get("match_acc", 0)
                total = s.get("total", 0)
                row += f" {val:.1f}% ({total}) |"
            lines.append(row)

    # ── Memory Actions ──
    lines.append("\n## Memory Actions\n")
    lines.append(f"| Metric | " + " | ".join(models) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(models)) + "|")
    for metric, key in [
        ("Type Accuracy", "type_acc"),
        ("Match Accuracy", "match_acc"),
        ("Content F1", "content_f1_mean"),
    ]:
        row = f"| {metric} |"
        for m in models:
            val = reports[m].get("memory_actions", {}).get(key, 0)
            if "f1" in key.lower():
                row += f" {val:.4f} |"
            else:
                row += f" {val:.2f}% |"
        lines.append(row)

    lines.append("\n### Memory Trigger\n")
    lines.append(f"| Metric | " + " | ".join(models) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(models)) + "|")
    for metric, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
        row = f"| {metric} |"
        for m in models:
            val = reports[m].get("memory_actions", {}).get("trigger", {}).get(key, 0)
            row += f" {val:.1f}% |"
        lines.append(row)

    # ── Folding ──
    lines.append("\n## Folding\n")
    lines.append(f"| Metric | " + " | ".join(models) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(models)) + "|")
    for metric, key in [
        ("Presence Acc", "presence_acc"),
        ("Range Acc (all)", "range_acc"),
        ("Range Acc (shallow)", "range_acc_shallow"),
        ("Range Acc (deep)", "range_acc_deep"),
        ("Depth Type Acc", "depth_type_acc"),
        ("Pred Deep Ratio", "pred_deep_ratio"),
    ]:
        row = f"| {metric} |"
        for m in models:
            val = reports[m].get("folding", {}).get(key, 0)
            row += f" {val:.2f}% |"
        lines.append(row)

    # ── Format Compliance ──
    lines.append("\n## Format Compliance\n")
    lines.append(f"| Metric | " + " | ".join(models) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(models)) + "|")
    row = "| All Tags Correct |"
    for m in models:
        val = reports[m].get("format_compliance", {}).get("all_correct", 0)
        row += f" {val:.2f}% |"
    lines.append(row)

    tags = reports[models[0]].get("format_compliance", {}).get("by_tag", {}).keys()
    for tag in tags:
        row = f"| `<{tag}>` |"
        for m in models:
            val = reports[m].get("format_compliance", {}).get("by_tag", {}).get(tag, 0)
            row += f" {val:.1f}% |"
        lines.append(row)

    # ── Trajectory-Level ──
    lines.append("\n## Trajectory-Level\n")
    lines.append(f"| Metric | " + " | ".join(models) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(models)) + "|")
    row = "| Full Match Rate |"
    for m in models:
        val = reports[m].get("trajectory_level", {}).get("full_match_rate", 0)
        row += f" {val:.2f}% |"
    lines.append(row)

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation reports")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = load_reports(args.results_dir)
    if not reports:
        print("No *_report.json files found!")
        return

    md = generate_markdown(reports)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
