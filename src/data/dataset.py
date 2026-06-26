"""PyTorch dataset for Pokemon battle training data.

Provides BattleDataset and BattleSequenceDataset classes that load
tensorized battle data from processed Parquet/binary files or directly
from parsed replays.

Split by battle (not by turn) to prevent data leakage.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data.observation import TurnObservation, build_observations
from src.data.replay_parser import ParsedBattle
from src.data.tensorizer import (
    BattleVocabularies,
    HISTORY_LENGTH,
    tensorize_battle,
    tensorize_turn,
)

logger = logging.getLogger(__name__)


class BattleTurnDataset:
    """Dataset of individual turns from battles.

    Each item is a single turn observation tensorized for training.
    Suitable for behavior cloning with per-turn cross-entropy loss.
    """

    def __init__(
        self,
        battles: list[ParsedBattle],
        vocabs: BattleVocabularies,
        build_vocab: bool = False,
    ) -> None:
        self.vocabs = vocabs
        self._turns: list[dict[str, np.ndarray]] = []
        self._battle_ids: list[str] = []

        for battle in battles:
            observations = build_observations(battle)
            for obs in observations:
                # Skip turns without actions (can't train on them)
                if not obs.action_taken:
                    continue
                tensor_dict = tensorize_turn(obs, vocabs, build_vocab=build_vocab)
                self._turns.append(tensor_dict)
                self._battle_ids.append(battle.battle_id)

        logger.info(
            f"BattleTurnDataset: {len(self._turns)} turns from "
            f"{len(set(self._battle_ids))} battles"
        )

    def __len__(self) -> int:
        return len(self._turns)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self._turns[idx]


class BattleSequenceDataset:
    """Dataset of full battle sequences.

    Each item is a full battle trajectory tensorized as a sequence.
    Suitable for transformer training with sequence-level context.
    """

    def __init__(
        self,
        battles: list[ParsedBattle],
        vocabs: BattleVocabularies,
        build_vocab: bool = False,
        max_turns: int = HISTORY_LENGTH,
    ) -> None:
        self.vocabs = vocabs
        self.max_turns = max_turns
        self._sequences: list[dict[str, np.ndarray]] = []
        self._battle_ids: list[str] = []

        for battle in battles:
            observations = build_observations(battle)
            if not observations:
                continue
            seq = tensorize_battle(
                observations, vocabs, build_vocab=build_vocab, max_turns=max_turns
            )
            if seq:
                self._sequences.append(seq)
                self._battle_ids.append(battle.battle_id)

        logger.info(
            f"BattleSequenceDataset: {len(self._sequences)} battles, "
            f"max_turns={max_turns}"
        )

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self._sequences[idx]


# ── Saving / loading processed data ──────────────────────────────────────


def save_processed_battles(
    battles: list[ParsedBattle],
    output_dir: str | Path,
    vocabs: BattleVocabularies,
    max_turns: int = HISTORY_LENGTH,
) -> dict[str, Any]:
    """Process and save battles as numpy arrays.

    Saves:
        - Individual .npz files per battle in output_dir/battles/
        - Vocabulary files in output_dir/vocabs/
        - Metadata JSON in output_dir/metadata.json

    Returns:
        Metadata dict with statistics.
    """
    output_dir = Path(output_dir)
    battles_dir = output_dir / "battles"
    battles_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "num_battles": 0,
        "num_turns": 0,
        "num_wins": 0,
        "num_losses": 0,
        "avg_turns": 0.0,
        "elo_distribution": {},
        "battle_ids": [],
    }

    elo_counts: dict[str, int] = {}

    for battle in battles:
        observations = build_observations(battle)
        if not observations:
            continue

        seq = tensorize_battle(
            observations, vocabs, build_vocab=True, max_turns=max_turns
        )
        if not seq:
            continue

        # Save as npz
        battle_file = battles_dir / f"{battle.battle_id}.npz"
        np.savez_compressed(str(battle_file), **seq)

        metadata["num_battles"] += 1
        metadata["num_turns"] += len(observations)
        metadata["battle_ids"].append(battle.battle_id)

        if battle.won:
            metadata["num_wins"] += 1
        else:
            metadata["num_losses"] += 1

        # Track Elo distribution
        elo_bucket = str((battle.player_elo // 100) * 100)
        elo_counts[elo_bucket] = elo_counts.get(elo_bucket, 0) + 1

    metadata["elo_distribution"] = elo_counts
    if metadata["num_battles"] > 0:
        metadata["avg_turns"] = metadata["num_turns"] / metadata["num_battles"]

    # Save vocabularies
    vocabs.freeze_all()
    vocabs.save(output_dir / "vocabs")

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Saved {metadata['num_battles']} battles to {output_dir} "
        f"({metadata['num_turns']} total turns)"
    )

    return metadata


def load_processed_dataset(
    data_dir: str | Path,
    split_file: str | Path | None = None,
) -> tuple[list[dict[str, np.ndarray]], BattleVocabularies]:
    """Load a processed dataset from disk.

    Args:
        data_dir: Directory containing battles/ and vocabs/ subdirectories.
        split_file: Optional path to a split manifest (JSON list of battle IDs).

    Returns:
        Tuple of (list of tensor dicts, vocabularies).
    """
    data_dir = Path(data_dir)
    battles_dir = data_dir / "battles"

    # Load vocabularies
    vocabs = BattleVocabularies.load(data_dir / "vocabs")

    # Load split manifest if provided
    battle_ids: set[str] | None = None
    if split_file is not None:
        with open(split_file) as f:
            battle_ids = set(json.load(f))

    # Load battle tensors
    sequences: list[dict[str, np.ndarray]] = []
    for npz_file in sorted(battles_dir.glob("*.npz")):
        battle_id = npz_file.stem
        if battle_ids is not None and battle_id not in battle_ids:
            continue
        data = dict(np.load(str(npz_file)))
        sequences.append(data)

    logger.info(f"Loaded {len(sequences)} battles from {data_dir}")
    return sequences, vocabs


# ── Train/val/test split ─────────────────────────────────────────────────


def create_splits(
    battle_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """Create train/val/test splits by battle ID.

    Splits by battle (not by turn) to prevent data leakage.

    Args:
        battle_ids: List of all battle IDs.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        output_dir: Optional directory to save split manifests.

    Returns:
        Dict with "train", "val", "test" keys mapping to lists of battle IDs.
    """
    rng = random.Random(seed)
    ids = list(battle_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }

    logger.info(
        f"Split {n} battles: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_ids in splits.items():
            with open(output_dir / f"{split_name}.json", "w") as f:
                json.dump(split_ids, f)

    return splits
