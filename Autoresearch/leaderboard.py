#!/usr/bin/env python3
"""AutoResearch Leaderboard.

Reads the experiment registry and produces a ranked table
by validation Top-1 accuracy.

Usage:
    python Autoresearch/leaderboard.py
    python Autoresearch/leaderboard.py --output Autoresearch/LEADERBOARD.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "Autoresearch" / "experiment_registry.json"


def load_registry() -> list[dict]:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return []


def format_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def format_float(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def format_delta(val: float | None) -> str:
    if val is None:
        return ""
    sign = "+" if val > 0 else ""
    return f"({sign}{val * 100:.2f}%)"


def generate_leaderboard(registry: list[dict]) -> str:
    """Generate markdown leaderboard from registry."""
    # Sort by Top-1 accuracy (descending), then by wall time (ascending)
    scored = []
    for exp in registry:
        metrics = exp.get("metrics", {})
        acc = metrics.get("test_top1_accuracy")
        scored.append((acc if acc is not None else -1, exp))

    scored.sort(key=lambda x: (-x[0], x[1].get("metrics", {}).get("wall_time_min", 999)))

    lines = [
        "# AutoResearch Leaderboard",
        "",
        f"_Generated from {len(registry)} experiments._",
        "",
        "## Rankings (by Top-1 Accuracy)",
        "",
        "| Rank | ID | Name | Top-1 | Top-3 | NLL | Move Acc | Switch Acc | ECE | Wall (min) | Tier | Decision |",
        "|------|----|------|-------|-------|-----|----------|------------|-----|------------|------|----------|",
    ]

    for rank, (acc, exp) in enumerate(scored, 1):
        m = exp.get("metrics", {})
        delta = exp.get("delta_from_parent", {})

        # Try to get move/switch from per-action breakdown
        per_action = m.get("per_action_accuracy", {})
        move_acc = None
        switch_acc = None
        if per_action:
            move_counts = [(v["accuracy"], v["count"]) for k, v in per_action.items() if k.startswith("move")]
            switch_counts = [(v["accuracy"], v["count"]) for k, v in per_action.items() if k.startswith("switch")]
            if move_counts:
                total_mc = sum(c for _, c in move_counts)
                move_acc = sum(a * c for a, c in move_counts) / max(total_mc, 1)
            if switch_counts:
                total_sc = sum(c for _, c in switch_counts)
                switch_acc = sum(a * c for a, c in switch_counts) / max(total_sc, 1)

        # Fall back to direct metrics if available
        move_acc = m.get("move_accuracy", move_acc)
        switch_acc = m.get("switch_accuracy", switch_acc)

        top1_delta = format_delta(delta.get("test_top1_accuracy"))
        decision = exp.get("decision") or "—"

        line = (
            f"| {rank} "
            f"| {exp.get('experiment_id', '?')} "
            f"| {exp.get('name', '?')} "
            f"| {format_pct(m.get('test_top1_accuracy'))} {top1_delta} "
            f"| {format_pct(m.get('test_top3_accuracy'))} "
            f"| {format_float(m.get('test_nll'))} "
            f"| {format_pct(move_acc)} "
            f"| {format_pct(switch_acc)} "
            f"| {format_float(m.get('test_ece'))} "
            f"| {format_float(m.get('wall_time_min'), 1)} "
            f"| T{exp.get('tier', '?')} "
            f"| {decision} |"
        )
        lines.append(line)

    lines.extend([
        "",
        "## Champion",
        "",
    ])

    if scored and scored[0][0] > 0:
        champion = scored[0][1]
        m = champion.get("metrics", {})
        lines.extend([
            f"**{champion.get('name', '?')}** ({champion.get('experiment_id', '?')})",
            f"- Top-1: {format_pct(m.get('test_top1_accuracy'))}",
            f"- Top-3: {format_pct(m.get('test_top3_accuracy'))}",
            f"- NLL: {format_float(m.get('test_nll'))}",
            f"- Config changes: {json.dumps(champion.get('config_changes', {}), indent=2)}",
        ])
    else:
        lines.append("No experiments with results yet.")

    lines.extend([
        "",
        "## Experiment Summary",
        "",
        f"- Total experiments: {len(registry)}",
        f"- With results: {sum(1 for e in registry if e.get('metrics', {}).get('test_top1_accuracy') is not None)}",
        f"- Promoted: {sum(1 for e in registry if e.get('decision') == 'PROMOTE')}",
        f"- Killed: {sum(1 for e in registry if e.get('decision') == 'KILL')}",
        f"- Pending: {sum(1 for e in registry if e.get('decision') is None)}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AutoResearch Leaderboard")
    parser.add_argument("--output", help="Save leaderboard as markdown file")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead")
    args = parser.parse_args()

    registry = load_registry()

    if not registry:
        print("No experiments in registry. Run experiments first.")
        return 0

    if args.json:
        print(json.dumps(registry, indent=2))
    else:
        leaderboard = generate_leaderboard(registry)
        print(leaderboard)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(leaderboard)
            print(f"\nSaved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
