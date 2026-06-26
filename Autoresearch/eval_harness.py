#!/usr/bin/env python3
"""AutoResearch Evaluation Harness.

Unified evaluation command for any checkpoint against the test split.
Reports: Top-1, Top-3, move/switch accuracy, per-action breakdown,
policy NLL, ECE calibration, auxiliary head accuracies, throughput.

Usage:
    python Autoresearch/eval_harness.py \
        --checkpoint checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt \
        --data-dir data/processed \
        --output Autoresearch/results/anchor.json
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

from src.data.tensorizer import BattleVocabularies
from src.data.auxiliary_labels import NUM_ITEM_CLASSES
from src.models.battle_transformer import (
    BattleTransformer,
    TransformerConfig,
    compute_total_loss,
    TOKENS_PER_STEP,
)
from src.environment.action_space import NUM_ACTIONS

# Reuse training infrastructure
from scripts.train_phase4 import (
    WindowedTurnDataset,
    collate_windowed,
    load_all_battles,
    add_auxiliary_labels,
    split_data,
    forward_step,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ACTION_NAMES = [
    "move1", "move2", "move3", "move4",
    "switch2", "switch3", "switch4", "switch5", "switch6",
]


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[BattleTransformer, TransformerConfig]:
    """Load a model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]

    config = TransformerConfig(
        num_layers=cfg_dict["num_layers"],
        hidden_dim=cfg_dict["hidden_dim"],
        num_heads=cfg_dict["num_heads"],
        dropout=cfg_dict.get("dropout", 0.1),
        ffn_multiplier=cfg_dict.get("ffn_multiplier", 4),
        species_vocab_size=cfg_dict["species_vocab_size"],
        moves_vocab_size=cfg_dict["moves_vocab_size"],
        items_vocab_size=cfg_dict["items_vocab_size"],
        abilities_vocab_size=cfg_dict["abilities_vocab_size"],
        types_vocab_size=cfg_dict["types_vocab_size"],
        status_vocab_size=cfg_dict["status_vocab_size"],
        weather_vocab_size=cfg_dict["weather_vocab_size"],
        terrain_vocab_size=cfg_dict.get("terrain_vocab_size", 5),
        species_embedding_dim=cfg_dict.get("species_embedding_dim", 64),
        move_embedding_dim=cfg_dict.get("move_embedding_dim", 32),
        item_embedding_dim=cfg_dict.get("item_embedding_dim", 32),
        ability_embedding_dim=cfg_dict.get("ability_embedding_dim", 32),
        type_embedding_dim=cfg_dict.get("type_embedding_dim", 16),
        auxiliary_loss_weight=cfg_dict.get("auxiliary_loss_weight", 0.2),
        use_value_head=cfg_dict.get("use_value_head", False),
        value_loss_weight=cfg_dict.get("value_loss_weight", 0.1),
    )

    model = BattleTransformer(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model: {param_count:,} parameters from {checkpoint_path}")
    return model, config


@torch.no_grad()
def evaluate_full(
    model: BattleTransformer,
    test_data: list[dict],
    config: TransformerConfig,
    device: torch.device,
    batch_size: int = 64,
    max_window: int = 20,
    num_workers: int = 0,
) -> dict:
    """Full evaluation with detailed metrics."""
    dataset = WindowedTurnDataset(test_data, max_window=max_window)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windowed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    total_loss = total_policy = total_aux = total_value = 0.0
    total_correct = total_top3 = total_examples = n_batches = 0
    action_type_correct = np.zeros(NUM_ACTIONS)
    action_type_total = np.zeros(NUM_ACTIONS)

    # Move vs switch breakdown
    move_correct = move_total = 0
    switch_correct = switch_total = 0

    # Calibration
    all_probs, all_correct_list = [], []

    # NLL
    total_nll = 0.0
    nll_count = 0

    # Auxiliary
    aux_correct = {"item": 0, "speed": 0, "role": 0}
    aux_total = {"item": 0, "speed": 0, "role": 0}

    # Throughput
    start_time = time.time()

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else torch.no_grad()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with amp_ctx:
            loss, loss_dict, logits, aux_preds = forward_step(model, batch, config)

        total_loss += loss_dict.get("total", 0.0)
        total_policy += loss_dict.get("policy", 0.0)
        total_aux += loss_dict.get("auxiliary", 0.0)
        total_value += loss_dict.get("value", 0.0)
        n_batches += 1

        action = batch["action"]
        valid = action >= 0
        if valid.any():
            valid_logits = logits[valid]
            targets = action[valid]
            preds = valid_logits.argmax(dim=-1)

            total_correct += (preds == targets).sum().item()
            top3 = valid_logits.topk(min(3, NUM_ACTIONS), dim=-1).indices
            total_top3 += (top3 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
            total_examples += valid.sum().item()

            # Per-action accuracy
            for a in range(NUM_ACTIONS):
                mask_a = targets == a
                if mask_a.any():
                    action_type_total[a] += mask_a.sum().item()
                    action_type_correct[a] += ((preds == targets) & mask_a).sum().item()

            # Move vs switch
            is_move = targets < 4
            is_switch = targets >= 4
            if is_move.any():
                move_correct += ((preds == targets) & is_move).sum().item()
                move_total += is_move.sum().item()
            if is_switch.any():
                switch_correct += ((preds == targets) & is_switch).sum().item()
                switch_total += is_switch.sum().item()

            # NLL
            log_probs = F.log_softmax(valid_logits, dim=-1)
            nll = F.nll_loss(log_probs, targets, reduction="sum")
            total_nll += nll.item()
            nll_count += targets.shape[0]

            # Calibration
            probs = F.softmax(valid_logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            is_correct = (preds == targets).float()
            all_probs.extend(max_probs.float().cpu().tolist())
            all_correct_list.extend(is_correct.float().cpu().tolist())

        # Auxiliary accuracy
        if aux_preds is not None:
            for head, tgt_key, short in [
                ("item_logits", "item_targets", "item"),
                ("speed_logits", "speed_targets", "speed"),
                ("role_logits", "role_targets", "role"),
            ]:
                if head in aux_preds and tgt_key in batch:
                    pred = aux_preds[head]
                    target = batch[tgt_key]
                    flat_pred = pred.reshape(-1, pred.shape[-1])
                    flat_target = target.reshape(-1)
                    valid_aux = flat_target >= 0
                    if valid_aux.any():
                        aux_pred_classes = flat_pred[valid_aux].argmax(dim=-1)
                        aux_correct[short] += (aux_pred_classes == flat_target[valid_aux]).sum().item()
                        aux_total[short] += valid_aux.sum().item()

    elapsed = time.time() - start_time
    n = max(n_batches, 1)
    ne = max(total_examples, 1)

    # Per-action breakdown
    per_action_acc = {}
    for i in range(NUM_ACTIONS):
        if action_type_total[i] > 0:
            per_action_acc[ACTION_NAMES[i]] = {
                "accuracy": round(action_type_correct[i] / action_type_total[i], 4),
                "count": int(action_type_total[i]),
            }

    # Calibration (5 bins)
    calibration = {}
    ece = 0.0
    if all_probs:
        probs_arr = np.array(all_probs)
        correct_arr = np.array(all_correct_list)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins) - 1):
            mask = (probs_arr >= bins[i]) & (probs_arr < bins[i + 1])
            if mask.any():
                bin_conf = float(probs_arr[mask].mean())
                bin_acc = float(correct_arr[mask].mean())
                bin_count = int(mask.sum())
                calibration[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] = {
                    "mean_confidence": round(bin_conf, 4),
                    "mean_accuracy": round(bin_acc, 4),
                    "count": bin_count,
                }
                ece += abs(bin_conf - bin_acc) * bin_count
        ece /= len(probs_arr)

    results = {
        "test_loss": round(total_loss / n, 4),
        "test_policy_loss": round(total_policy / n, 4),
        "test_aux_loss": round(total_aux / n, 4),
        "test_value_loss": round(total_value / n, 4),
        "test_accuracy": round(total_correct / ne, 4),
        "test_top3_accuracy": round(total_top3 / ne, 4),
        "test_nll": round(total_nll / max(nll_count, 1), 4),
        "test_examples": total_examples,
        "move_accuracy": round(move_correct / max(move_total, 1), 4),
        "move_count": move_total,
        "switch_accuracy": round(switch_correct / max(switch_total, 1), 4),
        "switch_count": switch_total,
        "expected_calibration_error": round(ece, 4),
        "per_action_accuracy": per_action_acc,
        "calibration": calibration,
        "auxiliary_accuracy": {
            name: round(aux_correct[name] / max(aux_total[name], 1), 4)
            for name in ["item", "speed", "role"]
        },
        "auxiliary_counts": {name: aux_total[name] for name in ["item", "speed", "role"]},
        "throughput_examples_per_sec": round(total_examples / max(elapsed, 0.001), 1),
        "wall_time_sec": round(elapsed, 2),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="AutoResearch Evaluation Harness")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--num-battles", type=int, default=None, help="Max battles to load")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-window", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model, config = load_checkpoint(args.checkpoint, device)

    # Load and split data
    sequences, vocabs = load_all_battles(args.data_dir, max_battles=args.num_battles)
    sequences = add_auxiliary_labels(sequences)
    _, _, test_data = split_data(sequences, seed=args.seed)
    logger.info(f"Test set: {len(test_data)} battles")

    # Evaluate
    results = evaluate_full(
        model, test_data, config, device,
        batch_size=args.batch_size,
        max_window=args.max_window,
        num_workers=args.num_workers,
    )

    # Add metadata
    results["checkpoint"] = args.checkpoint
    results["data_dir"] = args.data_dir
    results["num_battles_loaded"] = len(sequences)
    results["num_test_battles"] = len(test_data)
    results["seed"] = args.seed
    results["max_window"] = args.max_window

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Top-1 accuracy:   {results['test_accuracy'] * 100:.2f}%")
    print(f"  Top-3 accuracy:   {results['test_top3_accuracy'] * 100:.2f}%")
    print(f"  Move accuracy:    {results['move_accuracy'] * 100:.2f}% ({results['move_count']} examples)")
    print(f"  Switch accuracy:  {results['switch_accuracy'] * 100:.2f}% ({results['switch_count']} examples)")
    print(f"  Policy NLL:       {results['test_nll']:.4f}")
    print(f"  ECE:              {results['expected_calibration_error']:.4f}")
    print(f"  Throughput:       {results['throughput_examples_per_sec']:.0f} ex/sec")
    print(f"  Wall time:        {results['wall_time_sec']:.1f}s")
    print(f"\n  Aux item acc:     {results['auxiliary_accuracy']['item'] * 100:.2f}%")
    print(f"  Aux speed acc:    {results['auxiliary_accuracy']['speed'] * 100:.2f}%")
    print(f"  Aux role acc:     {results['auxiliary_accuracy']['role'] * 100:.2f}%")
    print("=" * 60)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
