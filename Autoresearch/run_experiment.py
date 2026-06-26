#!/usr/bin/env python3
"""AutoResearch experiment launcher and registry manager.

This script wraps ``scripts/train_phase4.py`` with a richer experiment-management layer so
agents can safely explore training-process and BattleTransformer architecture changes while
keeping the project inside the Phase 4 transformer family.

Key features:
- Registry-first experiment tracking
- Parent inheritance from the full effective config
- Tier budget enforcement with optional tighter caps
- Rich metadata for hypotheses, experiment families, datasets, and tags
- Broad support for train_phase4.py knobs (throughput, precision, architecture, regularization)
- Config validation to preserve the essential transformer architecture
- Optional note template generation for agent handoff / analysis
- Dry-run and register-only modes for planning before compute is spent
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def discover_project_root(start: Path) -> Path:
    """Find the repository root by locating the training script."""
    for candidate in [start, *start.parents]:
        if (candidate / "scripts" / "train_phase4.py").exists():
            return candidate
    raise FileNotFoundError("Could not find scripts/train_phase4.py from run_experiment.py")


RUN_EXPERIMENT_PATH = Path(__file__).resolve()
PROJECT_ROOT = discover_project_root(RUN_EXPERIMENT_PATH.parent)
AUTORESEARCH_DIR = PROJECT_ROOT / "Autoresearch"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_phase4.py"
REGISTRY_PATH = AUTORESEARCH_DIR / "experiment_registry.json"
RESULTS_DIR = AUTORESEARCH_DIR / "results"
NOTES_DIR = AUTORESEARCH_DIR / "notes"

DEFAULT_CONFIG: dict[str, Any] = {
    # Architecture: anchor P8-Lean 50K
    "num_layers": 3,
    "hidden_dim": 224,
    "num_heads": 4,
    "ffn_multiplier": 3,
    "species_embedding_dim": 48,
    "move_embedding_dim": 24,
    "item_embedding_dim": 16,
    "ability_embedding_dim": 16,
    "type_embedding_dim": 12,
    "max_window": 2,
    "dropout": 0.1,
    # Objective / optimization
    "aux_weight": 0.2,
    "value_weight": 0.1,
    "switch_weight": 1.0,
    "label_smoothing": 0.0,
    "no_value_head": True,
    "candidate_head": False,
    "split_head": False,
    "move_identity": False,
    "shuffle_moves": False,
    "cosine_epochs": None,
    "policy_head_layers": 1,
    "action_self_attention": False,
    "prune_dead_features": True,
    "batch_size": 64,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 300,
    "patience": 7,
    "grad_accum": 1,
    # Data / execution
    "num_battles": 50_000,
    "seed": 42,
    "data_dir": "data/processed",
    "num_workers": None,
    "prefetch_factor": 4,
    "persistent_workers": None,
    "pin_memory": None,
    "non_blocking_transfer": None,
    "amp": "auto",
    "torch_compile": False,
    "battle_manifest": None,
    "resume_from": None,
}

ALLOWED_CONFIG_KEYS = set(DEFAULT_CONFIG)
BOOL_KEYS = {
    "no_value_head",
    "candidate_head",
    "split_head",
    "move_identity",
    "shuffle_moves",
    "action_self_attention",
    "prune_dead_features",
    "persistent_workers",
    "pin_memory",
    "non_blocking_transfer",
    "torch_compile",
}
INT_KEYS = {
    "num_layers",
    "hidden_dim",
    "num_heads",
    "ffn_multiplier",
    "species_embedding_dim",
    "move_embedding_dim",
    "item_embedding_dim",
    "ability_embedding_dim",
    "type_embedding_dim",
    "max_window",
    "batch_size",
    "epochs",
    "warmup_steps",
    "patience",
    "grad_accum",
    "num_battles",
    "seed",
    "num_workers",
    "prefetch_factor",
    "cosine_epochs",
    "policy_head_layers",
}
FLOAT_KEYS = {"dropout", "aux_weight", "value_weight", "lr", "weight_decay", "switch_weight", "label_smoothing"}
STRING_KEYS = {"data_dir", "amp", "battle_manifest", "resume_from"}

TIER_BUDGETS = {
    1: {"max_epochs": 5, "max_minutes": 15},
    2: {"max_epochs": 30, "max_minutes": 120},
    3: {"max_epochs": 50, "max_minutes": 240},
}

PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "anchor": {},
    "throughput": {"batch_size": 256, "num_workers": 8, "prefetch_factor": 4, "amp": "bf16"},
    "window": {"max_window": 5},
    "full_data": {"num_battles": 100_000},
    "regularization": {"dropout": 0.15},
    "architecture_p8": {"num_layers": 4, "hidden_dim": 256, "num_heads": 4, "ffn_multiplier": 3},
    "architecture_p4": {"num_layers": 6, "hidden_dim": 384, "num_heads": 6, "ffn_multiplier": 3},
    "compile": {"torch_compile": True},
    # Champion-class recipe retuned to fit a single NVIDIA GTX 1650 (4 GB, Turing)
    # with ~15 GB system RAM. Same 5L/256d split-head + move-identity architecture as
    # AR-041 (~4.6M params), but: micro-batch 256 x grad_accum 4 == effective batch
    # 1024 (matches the A40 champion's optimizer batch). The model is small enough that
    # activations dominate VRAM (~2.5-3.8 MB/example at window 5); batch 256 is ~1.5 GB,
    # leaving comfortable headroom under 4 GB while keeping the GPU well utilised. Drop
    # to 128 (grad_accum 8) if a display is attached and VRAM is tight; bump to 512
    # (grad_accum 2) if running headless. fp16 not bf16 (Turing has 2x packed FP16 but
    # no bf16 hardware); 25K battles so a run finishes inside the local ~8 h budget.
    # Pair with an explicit, generous --budget-minutes (e.g. 480); the tier wall-clock
    # caps assume an A40, not a 1650.
    "gtx1650": {
        "num_layers": 5, "hidden_dim": 256, "num_heads": 4, "ffn_multiplier": 3,
        "max_window": 5,
        "split_head": True, "move_identity": True, "shuffle_moves": True,
        "no_value_head": True,
        "batch_size": 256, "grad_accum": 4, "amp": "fp16",
        "lr": 4e-4, "warmup_steps": 300, "patience": 20,
        "num_battles": 25_000,
        "num_workers": 6, "prefetch_factor": 4,
    },
}
# `local` is an alias of `gtx1650` for ergonomics.
PROFILE_OVERRIDES["local"] = dict(PROFILE_OVERRIDES["gtx1650"])

PROFILE_DESCRIPTIONS = {
    "anchor": "Reproduce the frozen P8-Lean anchor defaults.",
    "throughput": "Throughput-oriented run for batch size / loader / precision tuning.",
    "window": "History-length sweep centered on the max_window knob.",
    "full_data": "Scale from 50K battles toward the full processed corpus.",
    "regularization": "Regularization and calibration-focused run.",
    "architecture_p8": "Moderate transformer scale-up for longer-context experiments.",
    "architecture_p4": "Larger transformer scale-up intended for high-memory GPUs.",
    "compile": "Test torch.compile() as a performance optimization.",
    "gtx1650": "Champion-class 5L/256d recipe retuned for a 4 GB GTX 1650 (fp16, "
               "micro-batch + grad-accum, 25K battles, ~8 h budget).",
    "local": "Alias of gtx1650: champion-class recipe sized for a local 4 GB GPU.",
}

METRIC_KEYS_FOR_DELTA = [
    "test_top1_accuracy",
    "test_top3_accuracy",
    "test_loss",
    "test_nll",
    "test_ece",
    "move_accuracy",
    "switch_accuracy",
    "throughput_examples_sec",
    "wall_time_min",
]


def slugify(value: str) -> str:
    """Create a filesystem-safe experiment slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "experiment"


def load_registry() -> list[dict[str, Any]]:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return []


def save_registry(registry: list[dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def find_experiment(registry: list[dict[str, Any]], name_or_id: str) -> dict[str, Any] | None:
    for exp in registry:
        if exp.get("name") == name_or_id or exp.get("experiment_id") == name_or_id:
            return exp
    return None


def next_experiment_id(registry: list[dict[str, Any]]) -> str:
    max_num = 0
    for exp in registry:
        experiment_id = str(exp.get("experiment_id", ""))
        match = re.search(r"(\d+)$", experiment_id)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"AR-{max_num + 1:03d}"


def parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_key_value_pairs(items: list[str] | None, *, option_name: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not items:
        return overrides

    for item in items:
        if "=" not in item:
            raise ValueError(f"Malformed {option_name} value '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in ALLOWED_CONFIG_KEYS:
            allowed = ", ".join(sorted(ALLOWED_CONFIG_KEYS))
            raise ValueError(f"Unknown config key '{key}'. Allowed keys: {allowed}")
        overrides[key] = parse_scalar(raw_value.strip())
    return overrides


def ensure_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [ensure_serializable(v) for v in value]
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Guardrails: allow broad exploration without leaving the transformer family."""
    for key in config:
        if key not in ALLOWED_CONFIG_KEYS:
            raise ValueError(f"Unsupported config key '{key}'")

    for key in INT_KEYS:
        value = config.get(key)
        if value is not None and not isinstance(value, int):
            raise ValueError(f"Config '{key}' must be an int or None, got {type(value).__name__}")
    for key in FLOAT_KEYS:
        value = config.get(key)
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"Config '{key}' must be numeric, got {type(value).__name__}")
    for key in BOOL_KEYS:
        value = config.get(key)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"Config '{key}' must be bool or None, got {type(value).__name__}")
    for key in STRING_KEYS:
        value = config.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Config '{key}' must be a string or None, got {type(value).__name__}")

    if config["num_layers"] < 1:
        raise ValueError("num_layers must be >= 1")
    if config["hidden_dim"] < 32:
        raise ValueError("hidden_dim must be >= 32 to remain a meaningful transformer hidden size")
    if config["num_heads"] < 1:
        raise ValueError("num_heads must be >= 1")
    if config["hidden_dim"] % config["num_heads"] != 0:
        raise ValueError("hidden_dim must be divisible by num_heads for multi-head attention")
    if config["ffn_multiplier"] < 1:
        raise ValueError("ffn_multiplier must be >= 1")
    if config["max_window"] < 1:
        raise ValueError("max_window must be >= 1")
    if not 0.0 <= float(config["dropout"]) < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")
    if float(config["lr"]) <= 0:
        raise ValueError("lr must be > 0")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must be >= 0")
    if config["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    if config["epochs"] < 1:
        raise ValueError("epochs must be >= 1")
    if config["grad_accum"] < 1:
        raise ValueError("grad_accum must be >= 1")
    if config["num_battles"] is not None and config["num_battles"] < 1:
        raise ValueError("num_battles must be >= 1 when provided")
    if config["num_workers"] is not None and config["num_workers"] < 0:
        raise ValueError("num_workers must be >= 0")
    if config["prefetch_factor"] < 1:
        raise ValueError("prefetch_factor must be >= 1")
    if config["amp"] not in {"off", "fp16", "bf16", "auto"}:
        raise ValueError("amp must be one of: off, fp16, bf16, auto")


def normalize_parent_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Translate legacy registry fields into the current launcher schema."""
    if not config:
        return {}

    normalized = dict(config)
    if "use_value_head" in normalized and "no_value_head" not in normalized:
        normalized["no_value_head"] = not bool(normalized.pop("use_value_head"))
    else:
        normalized.pop("use_value_head", None)

    return {key: value for key, value in normalized.items() if key in ALLOWED_CONFIG_KEYS}


def compute_config_diff(base: dict[str, Any], updated: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key in sorted(ALLOWED_CONFIG_KEYS):
        if base.get(key) != updated.get(key):
            diff[key] = updated.get(key)
    return diff


def build_train_command(config: dict[str, Any], checkpoint_dir: Path, report_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--mode", "full",
        "--data-dir", str(config["data_dir"]),
        "--hidden-dim", str(config["hidden_dim"]),
        "--num-layers", str(config["num_layers"]),
        "--num-heads", str(config["num_heads"]),
        "--ffn-multiplier", str(config["ffn_multiplier"]),
        "--dropout", str(config["dropout"]),
        "--species-embedding-dim", str(config["species_embedding_dim"]),
        "--move-embedding-dim", str(config["move_embedding_dim"]),
        "--item-embedding-dim", str(config["item_embedding_dim"]),
        "--ability-embedding-dim", str(config["ability_embedding_dim"]),
        "--type-embedding-dim", str(config["type_embedding_dim"]),
        "--batch-size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--warmup-steps", str(config["warmup_steps"]),
        "--patience", str(config["patience"]),
        "--grad-accum", str(config["grad_accum"]),
        "--aux-weight", str(config["aux_weight"]),
        "--value-weight", str(config["value_weight"]),
        "--switch-weight", str(config["switch_weight"]),
        "--label-smoothing", str(config["label_smoothing"]),
        "--max-window", str(config["max_window"]),
        "--seed", str(config["seed"]),
        "--amp", str(config["amp"]),
        "--prefetch-factor", str(config["prefetch_factor"]),
        "--checkpoint-dir", str(checkpoint_dir),
        "--report-path", str(report_path),
    ]

    if config.get("num_battles") is not None:
        cmd.extend(["--num-battles", str(config["num_battles"])])
    if config.get("no_value_head"):
        cmd.append("--no-value-head")
    if config.get("candidate_head"):
        cmd.append("--candidate-head")
    if config.get("split_head"):
        cmd.append("--split-head")
    if config.get("move_identity"):
        cmd.append("--move-identity")
    if config.get("shuffle_moves"):
        cmd.append("--shuffle-moves")
    if config.get("policy_head_layers", 1) > 1:
        cmd.extend(["--policy-head-layers", str(config["policy_head_layers"])])
    if config.get("action_self_attention"):
        cmd.append("--action-self-attention")
    if config.get("cosine_epochs") is not None:
        cmd.extend(["--cosine-epochs", str(config["cosine_epochs"])])
    if config.get("prune_dead_features"):
        cmd.append("--prune-dead-features")
    if config.get("torch_compile"):
        cmd.append("--torch-compile")
    if config.get("num_workers") is not None:
        cmd.extend(["--num-workers", str(config["num_workers"])])
    if config.get("persistent_workers") is True:
        cmd.append("--persistent-workers")
    elif config.get("persistent_workers") is False:
        cmd.append("--no-persistent-workers")
    if config.get("pin_memory") is True:
        cmd.append("--pin-memory")
    elif config.get("pin_memory") is False:
        cmd.append("--no-pin-memory")
    if config.get("non_blocking_transfer") is True:
        cmd.append("--non-blocking-transfer")
    elif config.get("non_blocking_transfer") is False:
        cmd.append("--blocking-transfer")
    if config.get("battle_manifest") is not None:
        cmd.extend(["--battle-manifest", str(config["battle_manifest"])])
    if config.get("resume_from") is not None:
        cmd.extend(["--resume-from", str(config["resume_from"])])

    return cmd


def extract_results(report_path: Path) -> dict[str, Any] | None:
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    test = report.get("test_results", {})
    final = report.get("final_results", {})
    resources = report.get("resource_summary", {})
    epoch_metrics = report.get("epoch_metrics") or []
    best_examples_sec = max((epoch.get("examples_per_sec", 0.0) for epoch in epoch_metrics), default=0.0)

    per_action = test.get("per_action_accuracy") or {}
    move_stats = per_action.get("move") or {}
    switch_stats = per_action.get("switch") or {}

    return {
        "test_top1_accuracy": test.get("test_accuracy"),
        "test_top3_accuracy": test.get("test_top3_accuracy"),
        "test_loss": test.get("test_loss"),
        "test_nll": test.get("test_nll") or test.get("test_policy_loss"),
        "test_ece": test.get("expected_calibration_error"),
        "per_action_accuracy": per_action,
        "move_accuracy": move_stats.get("accuracy"),
        "switch_accuracy": switch_stats.get("accuracy"),
        "move_count": move_stats.get("count"),
        "switch_count": switch_stats.get("count"),
        "best_epoch": final.get("best_epoch"),
        "best_val_loss": final.get("best_val_loss"),
        "wall_time_min": final.get("total_wall_time_min"),
        "parameter_count": report.get("training_config", {}).get("parameter_count"),
        "throughput_examples_sec": best_examples_sec,
        "gpu_memory_used_gb": resources.get("gpu_memory_used_gb"),
        "peak_ram_gb": resources.get("peak_ram_gb"),
    }


def compare_to_parent(results: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    parent_metrics = parent.get("metrics", {})
    for key in METRIC_KEYS_FOR_DELTA:
        current = results.get(key)
        previous = parent_metrics.get(key)
        if current is not None and previous is not None:
            delta[key] = round(float(current) - float(previous), 4)
    return delta


def write_note_template(experiment: dict[str, Any]) -> str:
    note_name = f"{experiment['experiment_id'].lower()}_{slugify(experiment['name'])}.md"
    note_path = NOTES_DIR / note_name
    if note_path.exists():
        return str(note_path)

    parent_id = experiment.get("parent_id") or "none"
    config_changes = experiment.get("config_changes") or {}
    metrics = experiment.get("metrics") or {}
    delta = experiment.get("delta_from_parent") or {}

    note_path.parent.mkdir(parents=True, exist_ok=True)
    with open(note_path, "w") as f:
        f.write(
            f"# Experiment {experiment['experiment_id']}: {experiment['name']}\n\n"
            f"## Hypothesis\n{experiment.get('hypothesis') or 'TODO'}\n\n"
            f"## Change Made\n"
            f"Parent: {parent_id}\n\n"
            "```json\n"
            f"{json.dumps(config_changes, indent=2)}\n"
            "```\n\n"
            f"## Expected Impact\n{experiment.get('expected_impact') or 'TODO'}\n\n"
            "## Results\n"
            "```json\n"
            f"{json.dumps(metrics, indent=2)}\n"
            "```\n\n"
            "## Delta From Parent\n"
            "```json\n"
            f"{json.dumps(delta, indent=2)}\n"
            "```\n\n"
            "## Analysis\nTODO\n\n"
            "## Decision\n"
            "- [ ] KILL\n"
            "- [ ] RETRY\n"
            "- [ ] PROMOTE\n"
        )
    return str(note_path)


def summarize_profile(profile: str | None) -> str | None:
    if not profile:
        return None
    return PROFILE_DESCRIPTIONS.get(profile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoResearch experiment launcher for BattleTransformer training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="Human-readable experiment name")
    parser.add_argument("--parent", default="anchor", help="Parent experiment name or ID")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=1, help="Compute budget tier")
    parser.add_argument("--profile", choices=sorted(PROFILE_OVERRIDES), help="Convenience preset for common experiment families")
    parser.add_argument("--description", default="", help="Short experiment description")
    parser.add_argument("--hypothesis", default="", help="Why this experiment might work")
    parser.add_argument("--expected-impact", default="", help="Expected metric movement or operational gain")
    parser.add_argument("--phase", default="AR-unassigned", help="AutoResearch phase label, e.g. AR-3")
    parser.add_argument("--family", default="custom", help="Experiment family, e.g. window, throughput, architecture")
    parser.add_argument("--dataset-version", default="gen3ou_50k_v1", help="Dataset identifier for reproducibility")
    parser.add_argument("--tags", nargs="*", default=[], help="Free-form tags for filtering experiments later")
    parser.add_argument("--config-override", nargs="*", help="Config overrides as key=value pairs")
    parser.add_argument("--budget-epochs", type=int, default=None, help="Optional stricter epoch cap than the tier limit")
    parser.add_argument("--budget-minutes", type=int, default=None, help="Optional stricter wall-time cap than the tier limit")
    parser.add_argument("--data-dir", default=None, help="Override the dataset path used by training")
    parser.add_argument("--seed", type=int, default=None, help="Override the training seed")
    parser.add_argument("--register-only", action="store_true", help="Register the experiment and create notes, but do not launch training")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command/config without launching training")
    parser.add_argument("--skip-note-template", action="store_true", help="Do not create a note template file")
    parser.add_argument("--print-json", action="store_true", help="Print the full resolved experiment record as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry()

    if find_experiment(registry, args.name):
        logger.error("Experiment '%s' already exists in registry", args.name)
        return 1

    parent = find_experiment(registry, args.parent)
    if parent is None and args.parent != "anchor":
        logger.warning("Parent '%s' not found; falling back to anchor defaults", args.parent)

    base_config = deepcopy(DEFAULT_CONFIG)
    if parent and parent.get("full_config"):
        base_config.update(normalize_parent_config(parent["full_config"]))

    profile_overrides = deepcopy(PROFILE_OVERRIDES.get(args.profile, {}))
    cli_overrides = parse_key_value_pairs(args.config_override, option_name="--config-override")

    config = deepcopy(base_config)
    config.update(profile_overrides)
    config.update(cli_overrides)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.seed is not None:
        config["seed"] = args.seed

    tier_budget = TIER_BUDGETS[args.tier]
    epoch_cap = args.budget_epochs if args.budget_epochs is not None else tier_budget["max_epochs"]
    max_minutes = args.budget_minutes if args.budget_minutes is not None else tier_budget["max_minutes"]
    config["epochs"] = min(int(config["epochs"]), int(epoch_cap))

    validate_config(config)

    experiment_id = next_experiment_id(registry)
    slug = slugify(args.name)
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / f"autoresearch_{experiment_id.lower()}_{slug}"
    report_path = RESULTS_DIR / f"{experiment_id.lower()}_{slug}_report.json"

    config_changes = compute_config_diff(base_config, config)

    experiment: dict[str, Any] = {
        "experiment_id": experiment_id,
        "parent_id": parent.get("experiment_id") if parent else None,
        "name": args.name,
        "description": args.description,
        "hypothesis": args.hypothesis,
        "expected_impact": args.expected_impact,
        "phase": args.phase,
        "family": args.family,
        "profile": args.profile,
        "profile_description": summarize_profile(args.profile),
        "tier": args.tier,
        "dataset_version": args.dataset_version,
        "tags": args.tags,
        "status": "registered",
        "config_changes": ensure_serializable(config_changes),
        "full_config": ensure_serializable(config),
        "metrics": {},
        "delta_from_parent": {},
        "decision": None,
        "notes": "",
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": config["seed"],
        "checkpoint_path": str(checkpoint_dir),
        "report_path": str(report_path),
        "budget": {
            "max_epochs": config["epochs"],
            "max_minutes": max_minutes,
        },
        "lineage": {
            "parent_name": parent.get("name") if parent else None,
            "base_config_source": parent.get("experiment_id") if parent else "DEFAULT_CONFIG",
        },
    }

    if not args.skip_note_template:
        experiment["notes"] = write_note_template(experiment)

    registry.append(experiment)
    save_registry(registry)
    logger.info("Registered experiment %s: %s", experiment_id, args.name)

    cmd = build_train_command(config, checkpoint_dir, report_path)
    logger.info("Resolved command: %s", " ".join(cmd))
    logger.info("Resolved config diff vs parent/defaults: %s", json.dumps(config_changes, indent=2, sort_keys=True))

    if args.print_json:
        print(json.dumps(experiment, indent=2))

    if args.register_only:
        print(f"Registered {experiment_id} without launching training.")
        return 0

    if args.dry_run:
        print("\nDry run complete — experiment registered and command printed above.")
        return 0

    logger.info(
        "Starting training for %s (tier=%s, epoch cap=%s, wall cap=%s min)",
        experiment_id,
        args.tier,
        config["epochs"],
        max_minutes,
    )

    experiment["status"] = "running"
    experiment["start_timestamp"] = datetime.now(UTC).isoformat()
    save_registry(registry)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    success = False
    timeout_hit = False
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=max_minutes * 60, check=False)
        success = result.returncode == 0
        experiment["return_code"] = result.returncode
    except subprocess.TimeoutExpired:
        timeout_hit = True
        experiment["return_code"] = None
        logger.warning("Training exceeded wall-time budget of %s minutes", max_minutes)

    elapsed_min = round((time.time() - start) / 60, 2)
    results = extract_results(report_path)
    if results:
        results["wall_time_min"] = elapsed_min
        experiment["metrics"] = ensure_serializable(results)
        if parent:
            experiment["delta_from_parent"] = ensure_serializable(compare_to_parent(results, parent))

    experiment["end_timestamp"] = datetime.now(UTC).isoformat()
    experiment["status"] = "timeout" if timeout_hit else ("completed" if success else "failed")

    if not args.skip_note_template and experiment.get("notes"):
        write_note_template(experiment)

    for index, existing in enumerate(registry):
        if existing["experiment_id"] == experiment_id:
            registry[index] = experiment
            break
    save_registry(registry)

    print("\n" + "=" * 72)
    print(f"EXPERIMENT {experiment_id}: {args.name}")
    print("=" * 72)
    print(f"Phase/family: {args.phase} / {args.family}")
    print(f"Profile: {args.profile or 'custom'}")
    print(f"Status: {experiment['status'].upper()}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Report path: {report_path}")
    if results:
        top1 = results.get("test_top1_accuracy")
        top3 = results.get("test_top3_accuracy")
        move_acc = results.get("move_accuracy")
        switch_acc = results.get("switch_accuracy")
        throughput = results.get("throughput_examples_sec")
        if top1 is not None:
            print(f"Top-1 accuracy: {top1 * 100:.2f}%")
        if top3 is not None:
            print(f"Top-3 accuracy: {top3 * 100:.2f}%")
        if move_acc is not None:
            print(f"Move accuracy: {move_acc * 100:.2f}%")
        if switch_acc is not None:
            print(f"Switch accuracy: {switch_acc * 100:.2f}%")
        if throughput is not None:
            print(f"Best throughput: {throughput:.1f} examples/sec")
        if experiment.get("delta_from_parent"):
            print("Delta from parent:")
            for key, value in experiment["delta_from_parent"].items():
                sign = "+" if value > 0 else ""
                print(f"  {key}: {sign}{value:.4f}")
    else:
        print("No results extracted (training may have failed before writing a report).")
    print(f"Wall time: {elapsed_min:.2f} min")
    if experiment.get("notes"):
        print(f"Note template: {experiment['notes']}")
    print("=" * 72)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
