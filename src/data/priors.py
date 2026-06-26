"""Metagame usage priors for soft opponent prediction.

Computes and stores usage statistics from the training data:
- Species frequency
- Item distribution per species
- Ability distribution per species
- Move frequency per species
- Common teammates

These are used as soft prior features in the observation, NOT as
leaked ground truth. The Hidden Information Doctrine requires that
priors inform probability channels, not ground-truth labels.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.data.observation import TurnObservation
from src.data.replay_parser import ParsedBattle

logger = logging.getLogger(__name__)


class MetagamePriors:
    """Usage statistics computed from training data.

    Stores normalized frequency distributions for pokemon attributes.
    """

    def __init__(self) -> None:
        # species -> count
        self.species_counts: Counter[str] = Counter()
        # species -> {item: count}
        self.item_counts: dict[str, Counter[str]] = defaultdict(Counter)
        # species -> {ability: count}
        self.ability_counts: dict[str, Counter[str]] = defaultdict(Counter)
        # species -> {move: count}
        self.move_counts: dict[str, Counter[str]] = defaultdict(Counter)
        # Total battles processed
        self.total_battles: int = 0

    def update_from_battle(self, battle: ParsedBattle) -> None:
        """Update priors from a single battle."""
        self.total_battles += 1
        seen_species: set[str] = set()

        for turn in battle.turns:
            # Track player's own pokemon (we know full info for these)
            if turn.player_active:
                species = turn.player_active.name or turn.player_active.base_species
                if species and species not in seen_species:
                    seen_species.add(species)
                    self.species_counts[species] += 1

                    if turn.player_active.item:
                        self.item_counts[species][turn.player_active.item] += 1
                    if turn.player_active.ability:
                        self.ability_counts[species][turn.player_active.ability] += 1

                # Track moves used
                if species and turn.player_prev_move and turn.player_prev_move.name:
                    self.move_counts[species][turn.player_prev_move.name] += 1

            # Also track bench pokemon moves from switch data
            for poke in turn.available_switches:
                poke_species = poke.name or poke.base_species
                if poke_species:
                    for move in poke.moves:
                        if move.name:
                            self.move_counts[poke_species][move.name] += 1

    def get_item_distribution(
        self, species: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Get normalized item distribution for a species."""
        counts = self.item_counts.get(species, Counter())
        if not counts:
            return []
        total = sum(counts.values())
        return [
            (item, count / total) for item, count in counts.most_common(top_k)
        ]

    def get_ability_distribution(
        self, species: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Get normalized ability distribution for a species."""
        counts = self.ability_counts.get(species, Counter())
        if not counts:
            return []
        total = sum(counts.values())
        return [
            (ability, count / total) for ability, count in counts.most_common(top_k)
        ]

    def get_move_distribution(
        self, species: str, top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Get normalized move distribution for a species."""
        counts = self.move_counts.get(species, Counter())
        if not counts:
            return []
        total = sum(counts.values())
        return [
            (move, count / total) for move, count in counts.most_common(top_k)
        ]

    def get_top_species(self, top_k: int = 50) -> list[tuple[str, int]]:
        """Get most common species."""
        return self.species_counts.most_common(top_k)

    def get_item_prior_vector(
        self, species: str, item_vocab: dict[str, int], top_k: int = 10
    ) -> np.ndarray:
        """Get a soft prior vector over items for a species.

        Returns a probability vector indexed by the item vocabulary.
        """
        vec = np.zeros(len(item_vocab), dtype=np.float32)
        dist = self.get_item_distribution(species, top_k)
        for item, prob in dist:
            if item in item_vocab:
                vec[item_vocab[item]] = prob
        # Normalize
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    def save(self, path: str | Path) -> None:
        """Save priors to JSON."""
        data = {
            "total_battles": self.total_battles,
            "species_counts": dict(self.species_counts),
            "item_counts": {
                sp: dict(counts) for sp, counts in self.item_counts.items()
            },
            "ability_counts": {
                sp: dict(counts) for sp, counts in self.ability_counts.items()
            },
            "move_counts": {
                sp: dict(counts) for sp, counts in self.move_counts.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Saved priors: {len(self.species_counts)} species, "
            f"{self.total_battles} battles"
        )

    @classmethod
    def load(cls, path: str | Path) -> MetagamePriors:
        """Load priors from JSON."""
        with open(path) as f:
            data = json.load(f)

        priors = cls()
        priors.total_battles = data.get("total_battles", 0)
        priors.species_counts = Counter(data.get("species_counts", {}))
        priors.item_counts = defaultdict(
            Counter,
            {sp: Counter(counts) for sp, counts in data.get("item_counts", {}).items()},
        )
        priors.ability_counts = defaultdict(
            Counter,
            {
                sp: Counter(counts)
                for sp, counts in data.get("ability_counts", {}).items()
            },
        )
        priors.move_counts = defaultdict(
            Counter,
            {sp: Counter(counts) for sp, counts in data.get("move_counts", {}).items()},
        )
        return priors


def build_priors_from_battles(battles: list[ParsedBattle]) -> MetagamePriors:
    """Build metagame priors from a list of parsed battles."""
    priors = MetagamePriors()
    for battle in battles:
        priors.update_from_battle(battle)
    logger.info(
        f"Built priors from {priors.total_battles} battles, "
        f"{len(priors.species_counts)} unique species"
    )
    return priors
