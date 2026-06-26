#!/usr/bin/env python3
"""Train the Phase 4 P4 model on a 25,000-battle Gen 3 OU replay cohort.

P4 Gen 3 configuration:
- 6 layers / 384 hidden dim / 6 heads
- max window 20
- auxiliary head enabled (aux_weight=0.2)
- value head enabled
- dropout 0.1

This wrapper launches scripts/train_phase4.py for one or more seeds, then
creates an aggregated benchmark summary JSON suitable for cross-model
comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_phase4.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P4 model on 25,000 replay battles")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--num-battles", type=int, default=25000)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="One or more seeds (e.g. --seeds 42 43 44)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--non-blocking-transfer", action="store_true")
    parser.add_argument("--blocking-transfer", action="store_true")
    parser.add_argument("--output-root", type=str, default="checkpoints/phase4_gen3_p4_25k")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    cmd_str = " ".join(cmd)
    print(f"\n>>> {cmd_str}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_seed(report: dict[str, Any], seed: int) -> dict[str, Any]:
    training_cfg = report.get("training_config", {})
    test = report.get("test_results", {})
    final = report.get("final_results", {})
    return {
        "seed": seed,
        "parameter_count": training_cfg.get("parameter_count"),
        "best_epoch": final.get("best_epoch"),
        "best_val_loss": final.get("best_val_loss"),
        "test_top1_accuracy": test.get("test_accuracy"),
        "test_top3_accuracy": test.get("test_top3_accuracy"),
        "test_nll": test.get("test_nll"),
        "test_ece": test.get("expected_calibration_error"),
        "test_loss": test.get("test_loss"),
        "wall_time_min": final.get("total_wall_time_min"),
        "train_time_min": final.get("total_training_time_min"),
    }


def mean_std(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}


def aggregate(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    top1 = [r["test_top1_accuracy"] for r in results if r.get("test_top1_accuracy") is not None]
    top3 = [r["test_top3_accuracy"] for r in results if r.get("test_top3_accuracy") is not None]
    nll = [r["test_nll"] for r in results if r.get("test_nll") is not None]
    ece = [r["test_ece"] for r in results if r.get("test_ece") is not None]
    wall = [r["wall_time_min"] for r in results if r.get("wall_time_min") is not None]

    return {
        "experiment": "phase4_gen3_p4_25k",
        "created_at": datetime.now(UTC).isoformat(),
        "p4_config": {
            "num_layers": 6,
            "hidden_dim": 384,
            "num_heads": 6,
            "max_window": 20,
            "aux_weight": 0.2,
            "use_value_head": True,
        },
        "train_hparams": {
            "num_battles": args.num_battles,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "grad_accum": args.grad_accum,
            "dropout": args.dropout,
            "seeds": args.seeds,
        },
        "seed_results": results,
        "aggregate": {
            "test_top1_accuracy": mean_std(top1),
            "test_top3_accuracy": mean_std(top3),
            "test_nll": mean_std(nll),
            "test_ece": mean_std(ece),
            "wall_time_min": mean_std(wall),
        },
    }


def main() -> int:
    args = parse_args()

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing training script: {TRAIN_SCRIPT}")

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    seed_summaries: list[dict[str, Any]] = []

    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        report_path = seed_dir / "training_report.json"

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--mode",
            "full",
            "--data-dir",
            args.data_dir,
            "--num-battles",
            str(args.num_battles),
            "--num-layers",
            "6",
            "--hidden-dim",
            "384",
            "--num-heads",
            "6",
            "--max-window",
            "20",
            "--aux-weight",
            "0.2",
            "--dropout",
            str(args.dropout),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--warmup-steps",
            str(args.warmup_steps),
            "--patience",
            str(args.patience),
            "--grad-accum",
            str(args.grad_accum),
            "--seed",
            str(seed),
            "--checkpoint-dir",
            str(seed_dir),
            "--report-path",
            str(report_path),
        ]

        if args.num_workers is not None:
            cmd.extend(["--num-workers", str(args.num_workers)])
        cmd.extend(["--prefetch-factor", str(args.prefetch_factor)])
        if args.persistent_workers:
            cmd.append("--persistent-workers")
        if args.no_persistent_workers:
            cmd.append("--no-persistent-workers")
        if args.pin_memory:
            cmd.append("--pin-memory")
        if args.no_pin_memory:
            cmd.append("--no-pin-memory")
        if args.non_blocking_transfer:
            cmd.append("--non-blocking-transfer")
        if args.blocking_transfer:
            cmd.append("--blocking-transfer")

        run_command(cmd, dry_run=args.dry_run)

        if not args.dry_run:
            report = load_json(report_path)
            seed_summaries.append(summarize_seed(report, seed=seed))

    if args.dry_run:
        print("\nDry-run complete. No training executed.")
        return 0

    aggregate_report = aggregate(seed_summaries, args)
    aggregate_path = output_root / "p4_benchmark_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate_report, f, indent=2)

    print(f"\nSaved aggregate benchmark summary: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
