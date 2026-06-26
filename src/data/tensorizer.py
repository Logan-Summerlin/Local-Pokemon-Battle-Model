"""Tensorization pipeline for battle observations.

Converts TurnObservation objects into fixed-size tensor representations
for transformer input. Uses:
- Categorical indices for species, moves, items, abilities, types
- Continuous features for HP fractions, stat boosts, turn numbers
- Binary flags for status conditions, reveals, unknowns

Output format: dict of named tensors suitable for batching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.data.observation import (
    HISTORY_LENGTH,
    MAX_MOVES,
    MAX_TEAM_SIZE,
    UNKNOWN,
    FieldObservation,
    PokemonObservation,
    TurnObservation,
)
from src.environment.action_space import NUM_ACTIONS

logger = logging.getLogger(__name__)


# ── Vocabulary management ─────────────────────────────────────────────────


class Vocabulary:
    """Maps string tokens to integer indices with an explicit unknown token.

    Index 0 is reserved for padding/empty.
    Index 1 is reserved for UNKNOWN.
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, name: str = "vocab") -> None:
        self.name = name
        self._token_to_idx: dict[str, int] = {"": self.PAD_IDX, UNKNOWN: self.UNK_IDX}
        self._idx_to_token: dict[int, str] = {self.PAD_IDX: "", self.UNK_IDX: UNKNOWN}
        self._next_idx = 2
        self._frozen = False

    def add(self, token: str) -> int:
        """Add a token and return its index."""
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        if self._frozen:
            return self.UNK_IDX
        idx = self._next_idx
        self._token_to_idx[token] = idx
        self._idx_to_token[idx] = token
        self._next_idx += 1
        return idx

    def encode(self, token: str) -> int:
        """Encode a token to its index. Returns UNK_IDX for unknown tokens."""
        if not token:
            return self.PAD_IDX
        return self._token_to_idx.get(token, self.UNK_IDX)

    def decode(self, idx: int) -> str:
        """Decode an index back to its token."""
        return self._idx_to_token.get(idx, UNKNOWN)

    def freeze(self) -> None:
        """Freeze the vocabulary - no new tokens can be added."""
        self._frozen = True

    @property
    def size(self) -> int:
        return self._next_idx

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON."""
        data = {
            "name": self.name,
            "tokens": self._token_to_idx,
            "frozen": self._frozen,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        """Load vocabulary from JSON."""
        with open(path) as f:
            data = json.load(f)
        vocab = cls(name=data.get("name", "vocab"))
        for token, idx in data["tokens"].items():
            vocab._token_to_idx[token] = idx
            vocab._idx_to_token[idx] = token
            vocab._next_idx = max(vocab._next_idx, idx + 1)
        if data.get("frozen"):
            vocab.freeze()
        return vocab


class BattleVocabularies:
    """Collection of all vocabularies needed for tensorization."""

    def __init__(self) -> None:
        self.species = Vocabulary("species")
        self.moves = Vocabulary("moves")
        self.items = Vocabulary("items")
        self.abilities = Vocabulary("abilities")
        self.types = Vocabulary("types")
        self.status = Vocabulary("status")
        self.weather = Vocabulary("weather")
        self.terrain = Vocabulary("terrain")
        self.actions = Vocabulary("actions")

    def freeze_all(self) -> None:
        """Freeze all vocabularies."""
        for v in [
            self.species, self.moves, self.items, self.abilities,
            self.types, self.status, self.weather, self.terrain,
            self.actions,
        ]:
            v.freeze()

    def save(self, directory: str | Path) -> None:
        """Save all vocabularies to a directory."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name, vocab in self._all_vocabs():
            vocab.save(d / f"{name}.json")

    @classmethod
    def load(cls, directory: str | Path) -> BattleVocabularies:
        """Load all vocabularies from a directory."""
        bv = cls()
        d = Path(directory)
        for name, _ in bv._all_vocabs():
            path = d / f"{name}.json"
            if path.exists():
                setattr(bv, name, Vocabulary.load(path))
        return bv

    def _all_vocabs(self) -> list[tuple[str, Vocabulary]]:
        return [
            ("species", self.species),
            ("moves", self.moves),
            ("items", self.items),
            ("abilities", self.abilities),
            ("types", self.types),
            ("status", self.status),
            ("weather", self.weather),
            ("terrain", self.terrain),
            ("actions", self.actions),
        ]


# ── Feature dimensions ────────────────────────────────────────────────────

# Per-pokemon feature counts
# Categorical: species(1) + moves(4) + item(1) + ability(1) + types(1) + status(1) = 9
POKEMON_CATEGORICAL_DIM = 9
# Continuous: hp_frac(1) + boosts(7) + base_stats(6) = 14
POKEMON_CONTINUOUS_DIM = 14
# Binary: is_active(1) + is_fainted(1) + is_own(1) + is_unknown_item(1)
#         + is_unknown_ability(1) = 5
POKEMON_BINARY_DIM = 5
POKEMON_FEATURE_DIM = POKEMON_CATEGORICAL_DIM + POKEMON_CONTINUOUS_DIM + POKEMON_BINARY_DIM

# Field features
# Categorical: weather(1) + terrain(1) = 2
# Binary: weather_permanent(1) + own side (8) + opp side (8) = 17
FIELD_FEATURE_DIM = 2 + 17

# Turn context (Gen 3)
# Continuous: turn_number(1) + opponents_remaining(1) + num_opponent_revealed(1) = 3
# Binary: forced_switch(1) + is_lead_turn(1) = 2
# Categorical: prev_player_move(1) + prev_opponent_move(1) = 2
CONTEXT_FEATURE_DIM = 7


# ── Tensorization functions ───────────────────────────────────────────────


def tensorize_pokemon(
    poke: PokemonObservation,
    vocabs: BattleVocabularies,
    build_vocab: bool = False,
) -> np.ndarray:
    """Convert a PokemonObservation to a feature vector.

    Returns:
        np.ndarray of shape (POKEMON_FEATURE_DIM,) with float32 dtype.
    """
    features = np.zeros(POKEMON_FEATURE_DIM, dtype=np.float32)
    idx = 0

    # Categorical features (stored as float indices for embedding lookup)
    if build_vocab:
        features[idx] = vocabs.species.add(poke.species)
    else:
        features[idx] = vocabs.species.encode(poke.species)
    idx += 1

    # Moves (4 slots)
    for i in range(MAX_MOVES):
        move_name = poke.moves[i] if i < len(poke.moves) else ""
        if build_vocab:
            features[idx] = vocabs.moves.add(move_name)
        else:
            features[idx] = vocabs.moves.encode(move_name)
        idx += 1

    # Item
    if build_vocab:
        features[idx] = vocabs.items.add(poke.item)
    else:
        features[idx] = vocabs.items.encode(poke.item)
    idx += 1

    # Ability
    if build_vocab:
        features[idx] = vocabs.abilities.add(poke.ability)
    else:
        features[idx] = vocabs.abilities.encode(poke.ability)
    idx += 1

    # Types
    if build_vocab:
        features[idx] = vocabs.types.add(poke.types)
    else:
        features[idx] = vocabs.types.encode(poke.types)
    idx += 1

    # Status
    if build_vocab:
        features[idx] = vocabs.status.add(poke.status)
    else:
        features[idx] = vocabs.status.encode(poke.status)
    idx += 1

    # Continuous features
    features[idx] = poke.hp_fraction
    idx += 1

    # Boosts (7 stats)
    for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
        features[idx] = poke.boosts.get(stat, 0) / 6.0  # Normalize to [-1, 1]
        idx += 1

    # Base stats (6) - normalized
    for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
        features[idx] = poke.base_stats.get(stat, 0) / 255.0  # Max base stat ~255
        idx += 1

    # Binary features
    features[idx] = float(poke.is_active)
    idx += 1
    features[idx] = float(poke.is_fainted)
    idx += 1
    features[idx] = float(poke.is_own)
    idx += 1
    features[idx] = float(poke.item == UNKNOWN)
    idx += 1
    features[idx] = float(poke.ability == UNKNOWN)
    idx += 1

    return features


def tensorize_field(
    field: FieldObservation,
    vocabs: BattleVocabularies,
    build_vocab: bool = False,
) -> np.ndarray:
    """Convert a FieldObservation to a feature vector.

    Returns:
        np.ndarray of shape (FIELD_FEATURE_DIM,) with float32 dtype.
    """
    features = np.zeros(FIELD_FEATURE_DIM, dtype=np.float32)
    idx = 0

    # Weather (categorical)
    if build_vocab:
        features[idx] = vocabs.weather.add(field.weather)
    else:
        features[idx] = vocabs.weather.encode(field.weather)
    idx += 1

    # Terrain (categorical)
    if build_vocab:
        features[idx] = vocabs.terrain.add(field.terrain)
    else:
        features[idx] = vocabs.terrain.encode(field.terrain)
    idx += 1

    # Weather permanent flag (Gen 3: ability-set weather is permanent)
    features[idx] = float(field.weather_permanent); idx += 1

    # Own side conditions (binary)
    features[idx] = float(field.own_stealth_rock); idx += 1
    features[idx] = field.own_spikes / 3.0; idx += 1
    features[idx] = field.own_toxic_spikes / 2.0; idx += 1
    features[idx] = float(field.own_sticky_web); idx += 1
    features[idx] = float(field.own_reflect); idx += 1
    features[idx] = float(field.own_light_screen); idx += 1
    features[idx] = float(field.own_aurora_veil); idx += 1
    features[idx] = float(field.own_tailwind); idx += 1

    # Opponent side conditions
    features[idx] = float(field.opp_stealth_rock); idx += 1
    features[idx] = field.opp_spikes / 3.0; idx += 1
    features[idx] = field.opp_toxic_spikes / 2.0; idx += 1
    features[idx] = float(field.opp_sticky_web); idx += 1
    features[idx] = float(field.opp_reflect); idx += 1
    features[idx] = float(field.opp_light_screen); idx += 1
    features[idx] = float(field.opp_aurora_veil); idx += 1
    features[idx] = float(field.opp_tailwind); idx += 1

    return features


def tensorize_turn(
    obs: TurnObservation,
    vocabs: BattleVocabularies,
    build_vocab: bool = False,
) -> dict[str, np.ndarray]:
    """Convert a TurnObservation to a dict of tensors.

    Returns dict with keys:
        - own_team: (MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        - opponent_team: (MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        - field: (FIELD_FEATURE_DIM,)
        - context: (CONTEXT_FEATURE_DIM,)
        - legal_mask: (NUM_ACTIONS,)
        - action: scalar int (action index, or -1 if unavailable)
        - game_result: scalar float (1.0 = win, 0.0 = loss, 0.5 = unknown)
    """
    # Own team
    own_team = np.zeros((MAX_TEAM_SIZE, POKEMON_FEATURE_DIM), dtype=np.float32)
    for i, poke in enumerate(obs.own_team[:MAX_TEAM_SIZE]):
        own_team[i] = tensorize_pokemon(poke, vocabs, build_vocab)

    # Opponent team
    opp_team = np.zeros((MAX_TEAM_SIZE, POKEMON_FEATURE_DIM), dtype=np.float32)
    for i, poke in enumerate(obs.opponent_team[:MAX_TEAM_SIZE]):
        opp_team[i] = tensorize_pokemon(poke, vocabs, build_vocab)

    # Field
    field = tensorize_field(obs.field, vocabs, build_vocab)

    # Context features
    context = np.zeros(CONTEXT_FEATURE_DIM, dtype=np.float32)
    context[0] = obs.turn_number / 100.0  # Normalize turn number
    context[1] = obs.opponents_remaining / 6.0
    context[2] = obs.num_opponent_revealed / 6.0  # How many opponent Pokemon revealed
    context[3] = float(obs.forced_switch)
    context[4] = float(obs.is_lead_turn)

    # Previous moves (categorical)
    if build_vocab:
        context[5] = vocabs.moves.add(obs.prev_player_move)
        context[6] = vocabs.moves.add(obs.prev_opponent_move)
    else:
        context[5] = vocabs.moves.encode(obs.prev_player_move)
        context[6] = vocabs.moves.encode(obs.prev_opponent_move)

    # Legal mask
    legal_mask = np.array(obs.legal_action_mask[:NUM_ACTIONS], dtype=np.float32)
    if len(legal_mask) < NUM_ACTIONS:
        legal_mask = np.pad(legal_mask, (0, NUM_ACTIONS - len(legal_mask)))

    # Action taken — remap from Metamon encoding to our canonical action space.
    # Metamon Gen 3: 0-3 = moves, 4-8 = switches (bench 0-4)
    # Ours Gen 3:    0-3 = moves, 4-8 = switches (slot 2-6)
    _METAMON_TO_CANONICAL = {
        0: 0, 1: 1, 2: 2, 3: 3,           # moves 1-4 → moves 1-4
        4: 4, 5: 5, 6: 6, 7: 7, 8: 8,     # switch bench 0-4 → switch 2-6
    }
    action_idx = -1
    if obs.action_taken:
        try:
            parsed = int(obs.action_taken)
            action_idx = _METAMON_TO_CANONICAL.get(parsed, -1)
        except ValueError:
            pass  # Non-integer action string — leave as -1

    # Ensure the target action is always marked legal (if it was taken, it was legal)
    if 0 <= action_idx < NUM_ACTIONS:
        legal_mask[action_idx] = 1.0

    # Game result
    if obs.game_won is True:
        game_result = 1.0
    elif obs.game_won is False:
        game_result = 0.0
    else:
        game_result = 0.5

    return {
        "own_team": own_team,
        "opponent_team": opp_team,
        "field": field,
        "context": context,
        "legal_mask": legal_mask,
        "action": np.int64(action_idx),
        "game_result": np.float32(game_result),
    }


def tensorize_battle(
    observations: list[TurnObservation],
    vocabs: BattleVocabularies,
    build_vocab: bool = False,
    max_turns: int = HISTORY_LENGTH,
) -> dict[str, np.ndarray]:
    """Tensorize a full battle sequence.

    Returns dict with keys matching tensorize_turn but with an extra
    sequence dimension: (num_turns, ...) for each tensor.
    """
    if not observations:
        return {}

    # Limit to max_turns
    obs_list = observations[:max_turns]
    n_turns = len(obs_list)

    # Tensorize each turn
    turn_tensors = [tensorize_turn(o, vocabs, build_vocab) for o in obs_list]

    # Stack into sequences
    result: dict[str, np.ndarray] = {}
    for key in turn_tensors[0]:
        arrays = [t[key] for t in turn_tensors]
        result[key] = np.stack(arrays, axis=0)

    # Add sequence length
    result["seq_len"] = np.int64(n_turns)

    return result
