#!/usr/bin/env python3
"""Process raw replay files into tensorized training data.

Reads .json.lz4 files from data/raw/, parses them into observations,
tensorizes, and saves processed data with train/val/test splits.

Uses streaming processing to avoid loading all battles into memory at once.
Supports multiple generations via --generation flag. Vocabularies are
saved to generation-specific subdirectories (e.g., vocabs/gen3/, vocabs/gen9/).

Both player perspectives of a battle are kept as separate training examples
(using perspective_id for filenames). Splits are done by base battle_id to
prevent data leakage — both perspectives always land in the same split.

Usage:
    python scripts/process_dataset.py
    python scripts/process_dataset.py --generation gen3ou
    python scripts/process_dataset.py --generation gen9ou --input-dir data/raw --output-dir data/processed
    python scripts/process_dataset.py --max-battles 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_leakage_safe_splits(
    perspective_ids: list[str],
    battle_id_map: dict[str, str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """Create train/val/test splits by base battle ID, then expand to perspective IDs.

    Both perspectives of the same battle always end up in the same split.

    Args:
        perspective_ids: All perspective IDs (one per processed file).
        battle_id_map: Maps perspective_id -> base battle_id.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        output_dir: Optional directory to save split manifests.

    Returns:
        Dict with "train", "val", "test" keys mapping to lists of perspective IDs.
    """
    # Group perspective IDs by base battle ID
    battle_groups: dict[str, list[str]] = defaultdict(list)
    for pid in perspective_ids:
        bid = battle_id_map[pid]
        battle_groups[bid].append(pid)

    # Shuffle and split by base battle ID
    rng = random.Random(seed)
    base_ids = list(battle_groups.keys())
    rng.shuffle(base_ids)

    n = len(base_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_base = base_ids[:n_train]
    val_base = base_ids[n_train : n_train + n_val]
    test_base = base_ids[n_train + n_val :]

    # Expand to perspective IDs
    splits: dict[str, list[str]] = {
        "train": [pid for bid in train_base for pid in battle_groups[bid]],
        "val": [pid for bid in val_base for pid in battle_groups[bid]],
        "test": [pid for bid in test_base for pid in battle_groups[bid]],
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, ids in splits.items():
            with open(output_dir / f"{split_name}.json", "w") as f:
                json.dump(ids, f)

    logger.info(
        f"Splits (by {n} unique battles): "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw replays into training data")
    parser.add_argument(
        "--generation", type=str, default="gen3ou",
        help="Generation format being processed, e.g. gen3ou, gen9ou (default: gen3ou)",
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/raw",
        help="Directory containing raw .json.lz4 files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--splits-dir", type=str, default="data/splits",
        help="Output directory for split manifests",
    )
    parser.add_argument(
        "--max-battles", type=int, default=None,
        help="Maximum number of battles to process",
    )
    parser.add_argument(
        "--max-turns", type=int, default=20,
        help="Maximum turns per battle sequence",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits",
    )
    args = parser.parse_args()

    import numpy as np

    from src.data.observation import build_observations
    from src.data.priors import MetagamePriors
    from src.data.replay_parser import iter_battles_from_directory
    from src.data.tensorizer import BattleVocabularies, tensorize_battle

    # Set up output directories
    output_dir = Path(args.output_dir)
    battles_dir = output_dir / "battles"
    battles_dir.mkdir(parents=True, exist_ok=True)

    vocabs = BattleVocabularies()
    priors = MetagamePriors()

    generation = args.generation
    logger.info(f"Processing generation: {generation}")

    metadata: dict = {
        "generation": generation,
        "num_perspectives": 0,
        "num_unique_battles": 0,
        "num_turns": 0,
        "num_wins": 0,
        "num_losses": 0,
        "avg_turns": 0.0,
        "elo_distribution": {},
        "perspective_ids": [],
    }
    elo_counts: dict[str, int] = {}

    # Track perspective_id -> battle_id mapping for leakage-safe splits
    perspective_to_battle: dict[str, str] = {}
    seen_battle_ids: set[str] = set()

    # Stream battles one at a time to avoid OOM
    logger.info(f"Processing battles from {args.input_dir} (streaming)...")
    for battle in iter_battles_from_directory(
        args.input_dir,
        max_battles=args.max_battles,
    ):
        # Build observations
        observations = build_observations(battle)
        if not observations:
            continue

        # Tensorize
        seq = tensorize_battle(
            observations, vocabs, build_vocab=True, max_turns=args.max_turns
        )
        if not seq:
            continue

        # Use perspective_id for unique filenames (falls back to battle_id)
        file_id = battle.perspective_id or battle.battle_id

        # Save as npz
        battle_file = battles_dir / f"{file_id}.npz"
        np.savez_compressed(str(battle_file), **seq)

        # Track mapping
        perspective_to_battle[file_id] = battle.battle_id
        seen_battle_ids.add(battle.battle_id)

        # Update priors
        priors.update_from_battle(battle)

        # Update metadata
        metadata["num_perspectives"] += 1
        metadata["num_turns"] += len(observations)
        metadata["perspective_ids"].append(file_id)

        if battle.won:
            metadata["num_wins"] += 1
        else:
            metadata["num_losses"] += 1

        elo_bucket = str((battle.player_elo // 100) * 100)
        elo_counts[elo_bucket] = elo_counts.get(elo_bucket, 0) + 1

        if metadata["num_perspectives"] % 1000 == 0:
            logger.info(
                f"  Processed {metadata['num_perspectives']} perspectives "
                f"({metadata['num_turns']} turns)..."
            )

    if metadata["num_perspectives"] == 0:
        logger.error("No valid battles processed!")
        sys.exit(1)

    metadata["num_unique_battles"] = len(seen_battle_ids)
    metadata["elo_distribution"] = elo_counts
    metadata["avg_turns"] = metadata["num_turns"] / metadata["num_perspectives"]

    # Save vocabularies
    vocabs.freeze_all()
    vocabs.save(output_dir / "vocabs")

    # Save perspective -> battle_id mapping for reproducibility
    with open(output_dir / "perspective_map.json", "w") as f:
        json.dump(perspective_to_battle, f)

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Saved {metadata['num_perspectives']} perspectives "
        f"({metadata['num_unique_battles']} unique battles) to {output_dir} "
        f"({metadata['num_turns']} total turns)"
    )

    # Save priors
    logger.info("Saving metagame priors...")
    priors.save(output_dir / "priors.json")

    # Create leakage-safe splits (group by base battle ID)
    logger.info("Creating train/val/test splits (leakage-safe by battle ID)...")
    splits = create_leakage_safe_splits(
        perspective_ids=metadata["perspective_ids"],
        battle_id_map=perspective_to_battle,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed,
        output_dir=args.splits_dir,
    )

    # Print summary
    logger.info("\n=== Processing Summary ===")
    logger.info(f"Total perspectives: {metadata['num_perspectives']}")
    logger.info(f"Unique battles: {metadata['num_unique_battles']}")
    logger.info(f"Total turns: {metadata['num_turns']}")
    logger.info(f"Average turns per perspective: {metadata['avg_turns']:.1f}")
    logger.info(f"Wins: {metadata['num_wins']}, Losses: {metadata['num_losses']}")
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    logger.info(f"Vocabulary sizes:")
    logger.info(f"  Species: {vocabs.species.size}")
    logger.info(f"  Moves: {vocabs.moves.size}")
    logger.info(f"  Items: {vocabs.items.size}")
    logger.info(f"  Abilities: {vocabs.abilities.size}")

    # Print Elo distribution
    logger.info("Elo distribution:")
    for bucket, count in sorted(metadata.get("elo_distribution", {}).items()):
        logger.info(f"  {bucket}+: {count}")

    # Print top species
    top_species = priors.get_top_species(20)
    logger.info("Top 20 species:")
    for species, count in top_species:
        logger.info(f"  {species}: {count}")


if __name__ == "__main__":
    main()
