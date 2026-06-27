#!/usr/bin/env python3
"""A5 calibration track: post-hoc temperature scaling + game-phase ECE.

Fits a single temperature ``T`` on the validation split (minimising policy NLL),
applies it to the test split, and reports ECE overall and broken down by game
phase (early / mid / late, by relative turn position within each battle). The
fitted temperature is written next to the checkpoint as ``temperature.json`` so
inference can load it (A5: "temperature constant stored with checkpoint").

Usage:
    python Autoresearch/calibration.py \
        --checkpoint checkpoints/autoresearch_ar-024_.../best_model.pt \
        --data-dir data/processed \
        --output Autoresearch/results/ar-024_calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.environment.action_space import NUM_ACTIONS
from scripts.train_phase4 import (
    WindowedTurnDataset,
    collate_windowed,
    load_all_battles,
    add_auxiliary_labels,
    split_data,
    forward_step,
)
from Autoresearch.eval_harness import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PHASE_NAMES = ["early", "mid", "late"]


def _amp_ctx(device: torch.device):
    """Match eval_harness: bf16 autocast only where supported (Turing -> off)."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.no_grad()


@torch.no_grad()
def collect_logits(
    model,
    data: list[dict],
    config,
    device: torch.device,
    batch_size: int,
    max_window: int,
    num_workers: int = 0,
    with_phase: bool = False,
) -> dict[str, torch.Tensor]:
    """Run the model in dataset order and gather per-example logits/targets.

    Returns CPU tensors: ``logits`` (N, NUM_ACTIONS), ``targets`` (N,), and,
    when ``with_phase``, ``phase`` (N,) in {0:early, 1:mid, 2:late}.
    """
    dataset = WindowedTurnDataset(data, max_window=max_window, shuffle_moves=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windowed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # Per-example game phase from absolute turn position within its battle.
    phase_lookup = None
    if with_phase:
        phase_lookup = np.empty(len(dataset.examples), dtype=np.int64)
        for k, (b_idx, t_idx) in enumerate(dataset.examples):
            total = int(dataset.battles[b_idx]["action"].shape[0])
            frac = t_idx / max(total - 1, 1)
            phase_lookup[k] = 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)

    logits_chunks, target_chunks, phase_chunks = [], [], []
    cursor = 0
    for batch in loader:
        bsz = batch["action"].shape[0]
        batch = {k: v.to(device) for k, v in batch.items()}
        with _amp_ctx(device):
            _, _, logits, _ = forward_step(model, batch, config)
        logits_chunks.append(logits.float().cpu())
        target_chunks.append(batch["action"].cpu())
        if with_phase:
            phase_chunks.append(torch.from_numpy(phase_lookup[cursor:cursor + bsz]))
        cursor += bsz

    out = {
        "logits": torch.cat(logits_chunks),
        "targets": torch.cat(target_chunks),
    }
    if with_phase:
        out["phase"] = torch.cat(phase_chunks)
    return out


def nll_at_temperature(logits: torch.Tensor, targets: torch.Tensor, temp: float) -> float:
    """Mean policy NLL over valid (target >= 0) examples at temperature ``temp``."""
    valid = targets >= 0
    lp = F.log_softmax(logits[valid] / temp, dim=-1)
    return F.nll_loss(lp, targets[valid], reduction="mean").item()


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Find T>0 minimising NLL via coarse grid + golden-section refinement."""
    grid = np.linspace(0.5, 5.0, 46)
    best_t = min(grid, key=lambda t: nll_at_temperature(logits, targets, float(t)))

    # Golden-section search in a bracket around the grid winner.
    lo, hi = max(0.25, best_t - 0.5), best_t + 0.5
    gr = (np.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    for _ in range(40):
        if nll_at_temperature(logits, targets, c) < nll_at_temperature(logits, targets, d):
            hi = d
        else:
            lo = c
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
    return round((lo + hi) / 2, 4)


def ece_and_acc(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 5) -> dict:
    """Expected calibration error + top-1 over valid examples (5 equal-width bins)."""
    valid = targets >= 0
    logits, targets = logits[valid], targets[valid]
    if logits.shape[0] == 0:
        return {"ece": None, "accuracy": None, "count": 0, "bins": {}}

    probs = F.softmax(logits, dim=-1)
    conf, preds = probs.max(dim=-1)
    correct = (preds == targets).float()
    conf_np, correct_np = conf.numpy(), correct.numpy()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf_np >= lo) & (conf_np < hi) if i < n_bins - 1 else (conf_np >= lo) & (conf_np <= hi)
        if mask.any():
            bin_conf = float(conf_np[mask].mean())
            bin_acc = float(correct_np[mask].mean())
            cnt = int(mask.sum())
            bins[f"{lo:.1f}-{hi:.1f}"] = {
                "mean_confidence": round(bin_conf, 4),
                "mean_accuracy": round(bin_acc, 4),
                "count": cnt,
            }
            ece += abs(bin_conf - bin_acc) * cnt
    ece /= conf_np.shape[0]
    return {
        "ece": round(ece, 4),
        "accuracy": round(float(correct_np.mean()), 4),
        "count": int(conf_np.shape[0]),
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A5 temperature scaling + game-phase ECE")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-battles", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-window", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-temperature", action="store_true",
                        help="Do not write temperature.json next to the checkpoint.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, config = load_checkpoint(args.checkpoint, device)

    sequences, vocabs = load_all_battles(args.data_dir, max_battles=args.num_battles)
    sequences = add_auxiliary_labels(sequences, vocabs)
    _, val_data, test_data = split_data(sequences, seed=args.seed)
    logger.info("val=%d test=%d battles", len(val_data), len(test_data))

    start = time.time()
    val = collect_logits(model, val_data, config, device, args.batch_size,
                         args.max_window, args.num_workers, with_phase=False)
    test = collect_logits(model, test_data, config, device, args.batch_size,
                          args.max_window, args.num_workers, with_phase=True)

    temperature = fit_temperature(val["logits"], val["targets"])
    val_nll_raw = nll_at_temperature(val["logits"], val["targets"], 1.0)
    val_nll_temp = nll_at_temperature(val["logits"], val["targets"], temperature)
    logger.info("Fitted T=%.4f (val NLL %.4f -> %.4f)", temperature, val_nll_raw, val_nll_temp)

    overall_raw = ece_and_acc(test["logits"], test["targets"])
    overall_temp = ece_and_acc(test["logits"] / temperature, test["targets"])

    by_phase = {}
    for p, name in enumerate(PHASE_NAMES):
        pmask = test["phase"] == p
        by_phase[name] = {
            "raw": ece_and_acc(test["logits"][pmask], test["targets"][pmask]),
            "temperature_scaled": ece_and_acc(test["logits"][pmask] / temperature, test["targets"][pmask]),
        }

    results = {
        "checkpoint": args.checkpoint,
        "temperature": temperature,
        "val_nll_raw": round(val_nll_raw, 4),
        "val_nll_temperature_scaled": round(val_nll_temp, 4),
        "test_overall": {"raw": overall_raw, "temperature_scaled": overall_temp},
        "test_by_game_phase": by_phase,
        "ece_reduction": round((overall_raw["ece"] or 0) - (overall_temp["ece"] or 0), 4),
        "seed": args.seed,
        "max_window": args.max_window,
        "wall_time_sec": round(time.time() - start, 1),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    if not args.no_save_temperature:
        temp_path = Path(args.checkpoint).parent / "temperature.json"
        temp_path.write_text(json.dumps({"temperature": temperature, "fit_seed": args.seed}, indent=2))
        logger.info("Saved temperature to %s", temp_path)

    print("\n" + "=" * 60)
    print("A5 CALIBRATION")
    print("=" * 60)
    print(f"  Temperature T:        {temperature:.4f}")
    print(f"  Test ECE raw:         {overall_raw['ece']:.4f}  (top-1 {overall_raw['accuracy']:.4f})")
    print(f"  Test ECE T-scaled:    {overall_temp['ece']:.4f}  (top-1 {overall_temp['accuracy']:.4f})")
    print("  By phase (raw -> T-scaled ECE):")
    for name in PHASE_NAMES:
        r = by_phase[name]["raw"]["ece"]
        t = by_phase[name]["temperature_scaled"]["ece"]
        print(f"    {name:5s}: {r:.4f} -> {t:.4f}")
    print("=" * 60)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
