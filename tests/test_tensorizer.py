"""Tests for the tensorization pipeline.

Covers:
- Fixed-size tensor output from variable observations
- Vocabulary building and encoding
- Pokemon feature vectorization
- Field feature vectorization
- Full turn and battle tensorization
- Shape validation
"""

import numpy as np
import pytest

from src.data.observation import (
    UNKNOWN,
    MAX_TEAM_SIZE,
    FieldObservation,
    PokemonObservation,
    TurnObservation,
    build_observations,
)
from src.data.replay_parser import (
    ParsedBattle,
    ParsedMove,
    ParsedPokemon,
    ParsedTurnState,
)
from src.data.tensorizer import (
    CONTEXT_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    POKEMON_FEATURE_DIM,
    BattleVocabularies,
    Vocabulary,
    tensorize_battle,
    tensorize_field,
    tensorize_pokemon,
    tensorize_turn,
)
from src.environment.action_space import NUM_ACTIONS


# ── Helpers ───────────────────────────────────────────────────────────────


def make_pokemon_obs(
    species: str = "Pikachu",
    hp: float = 1.0,
    moves: list[str] | None = None,
    item: str = "Light Ball",
    ability: str = "Static",
    is_own: bool = True,
    is_active: bool = True,
    status: str = "",
) -> PokemonObservation:
    return PokemonObservation(
        species=species,
        hp_fraction=hp,
        status=status,
        is_active=is_active,
        is_fainted=hp <= 0,
        moves=moves or ["Thunderbolt", "Quick Attack"],
        item=item,
        ability=ability,
        boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0},
        base_stats={"hp": 35, "atk": 55, "def": 30, "spa": 50, "spd": 40, "spe": 90},
        types="Electric",
        level=100,
        is_own=is_own,
    )


def make_turn_obs(
    turn_number: int = 0,
    action: str = "move0",
    game_won: bool | None = True,
) -> TurnObservation:
    own = [make_pokemon_obs(is_active=True)] + [
        make_pokemon_obs(f"Mon{i}", is_active=False) for i in range(5)
    ]
    opp = [make_pokemon_obs("Charizard", is_own=False, is_active=True)] + [
        make_pokemon_obs(f"OppMon{i}", is_own=False, is_active=False) for i in range(5)
    ]
    return TurnObservation(
        turn_number=turn_number,
        own_team=own,
        opponent_team=opp,
        field=FieldObservation(),
        action_taken=action,
        game_won=game_won,
    )


# ── Tests: Vocabulary ────────────────────────────────────────────────────


class TestVocabulary:
    def test_pad_and_unk(self) -> None:
        vocab = Vocabulary("test")
        assert vocab.encode("") == 0  # PAD
        assert vocab.encode(UNKNOWN) == 1  # UNK

    def test_add_and_encode(self) -> None:
        vocab = Vocabulary("test")
        idx = vocab.add("Pikachu")
        assert idx == 2
        assert vocab.encode("Pikachu") == 2

    def test_unknown_token(self) -> None:
        vocab = Vocabulary("test")
        vocab.add("Pikachu")
        vocab.freeze()
        assert vocab.encode("Charizard") == 1  # UNK

    def test_decode(self) -> None:
        vocab = Vocabulary("test")
        vocab.add("Pikachu")
        assert vocab.decode(2) == "Pikachu"
        assert vocab.decode(999) == UNKNOWN

    def test_freeze_prevents_add(self) -> None:
        vocab = Vocabulary("test")
        vocab.add("Pikachu")
        vocab.freeze()
        idx = vocab.add("Charizard")
        assert idx == 1  # UNK, not added

    def test_size(self) -> None:
        vocab = Vocabulary("test")
        vocab.add("A")
        vocab.add("B")
        vocab.add("C")
        assert vocab.size == 5  # PAD + UNK + 3

    def test_save_load(self, tmp_path) -> None:
        vocab = Vocabulary("test")
        vocab.add("Pikachu")
        vocab.add("Charizard")
        vocab.freeze()

        path = tmp_path / "vocab.json"
        vocab.save(path)

        loaded = Vocabulary.load(path)
        assert loaded.encode("Pikachu") == vocab.encode("Pikachu")
        assert loaded.encode("Charizard") == vocab.encode("Charizard")
        assert loaded.size == vocab.size


# ── Tests: Pokemon tensorization ──────────────────────────────────────────


class TestPokemonTensorization:
    def test_output_shape(self) -> None:
        vocabs = BattleVocabularies()
        poke = make_pokemon_obs()
        tensor = tensorize_pokemon(poke, vocabs, build_vocab=True)
        assert tensor.shape == (POKEMON_FEATURE_DIM,)
        assert tensor.dtype == np.float32

    def test_species_encoded(self) -> None:
        vocabs = BattleVocabularies()
        poke = make_pokemon_obs(species="Pikachu")
        tensor = tensorize_pokemon(poke, vocabs, build_vocab=True)
        # First feature should be the species index
        assert tensor[0] == vocabs.species.encode("Pikachu")

    def test_hp_fraction(self) -> None:
        vocabs = BattleVocabularies()
        poke = make_pokemon_obs(hp=0.75)
        tensor = tensorize_pokemon(poke, vocabs, build_vocab=True)
        # HP fraction is the first continuous feature (after 9 categorical)
        assert tensor[9] == pytest.approx(0.75)

    def test_fainted_flag(self) -> None:
        vocabs = BattleVocabularies()
        poke = make_pokemon_obs(hp=0.0)
        tensor = tensorize_pokemon(poke, vocabs, build_vocab=True)
        # is_fainted binary flag
        fainted_idx = 9 + 14 + 1  # categorical + continuous + is_active
        assert tensor[fainted_idx] == 1.0

    def test_unknown_item_flag(self) -> None:
        vocabs = BattleVocabularies()
        poke = make_pokemon_obs(item=UNKNOWN)
        tensor = tensorize_pokemon(poke, vocabs, build_vocab=True)
        # is_unknown_item flag
        unknown_item_idx = 9 + 14 + 3  # after categorical, continuous, is_active, is_fainted, is_own
        assert tensor[unknown_item_idx] == 1.0

    def test_opponent_base_stats_from_crosswalk(self) -> None:
        """Opponent pokemon now get base stats from crosswalk (public knowledge)."""
        vocabs = BattleVocabularies()
        # Create opponent with base_stats populated (as crosswalk now provides)
        poke = PokemonObservation(
            species="Charizard",
            hp_fraction=1.0,
            is_active=True,
            is_own=False,
            moves=[],
            item=UNKNOWN,
            ability=UNKNOWN,
            boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0},
            base_stats={"hp": 78, "atk": 84, "def": 78, "spa": 109, "spd": 85, "spe": 100},
            types="Fire/Flying",
        )
        tensor = tensorize_pokemon(poke, vocabs, build_vocab=True)
        # Base stats start at index 9 + 1 + 7 = 17
        base_stat_start = 9 + 1 + 7  # after categorical, hp_frac, boosts
        # HP base stat should be 78/255.0
        assert tensor[base_stat_start] == pytest.approx(78 / 255.0)
        # Speed base stat should be 100/255.0
        assert tensor[base_stat_start + 5] == pytest.approx(100 / 255.0)


# ── Tests: Field tensorization ────────────────────────────────────────────


class TestFieldTensorization:
    def test_output_shape(self) -> None:
        vocabs = BattleVocabularies()
        field = FieldObservation()
        tensor = tensorize_field(field, vocabs, build_vocab=True)
        assert tensor.shape == (FIELD_FEATURE_DIM,)

    def test_weather_encoded(self) -> None:
        vocabs = BattleVocabularies()
        field = FieldObservation(weather="RainDance")
        tensor = tensorize_field(field, vocabs, build_vocab=True)
        assert tensor[0] == vocabs.weather.encode("RainDance")

    def test_weather_permanent(self) -> None:
        vocabs = BattleVocabularies()
        field = FieldObservation(weather_permanent=True)
        tensor = tensorize_field(field, vocabs, build_vocab=True)
        assert tensor[2] == 1.0  # After weather(1) + terrain(1)

    def test_stealth_rock(self) -> None:
        vocabs = BattleVocabularies()
        field = FieldObservation(own_stealth_rock=True)
        tensor = tensorize_field(field, vocabs, build_vocab=True)
        assert tensor[3] == 1.0  # After weather(1) + terrain(1) + weather_permanent(1)

    def test_spikes_normalized(self) -> None:
        vocabs = BattleVocabularies()
        field = FieldObservation(own_spikes=3)
        tensor = tensorize_field(field, vocabs, build_vocab=True)
        assert tensor[4] == pytest.approx(1.0)  # 3/3 = 1.0


# ── Tests: Turn tensorization ─────────────────────────────────────────────


class TestTurnTensorization:
    def test_output_keys(self) -> None:
        vocabs = BattleVocabularies()
        obs = make_turn_obs()
        result = tensorize_turn(obs, vocabs, build_vocab=True)
        expected_keys = {"own_team", "opponent_team", "field", "context",
                         "legal_mask", "action", "game_result"}
        assert set(result.keys()) == expected_keys

    def test_own_team_shape(self) -> None:
        vocabs = BattleVocabularies()
        obs = make_turn_obs()
        result = tensorize_turn(obs, vocabs, build_vocab=True)
        assert result["own_team"].shape == (MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)

    def test_opponent_team_shape(self) -> None:
        vocabs = BattleVocabularies()
        obs = make_turn_obs()
        result = tensorize_turn(obs, vocabs, build_vocab=True)
        assert result["opponent_team"].shape == (MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)

    def test_field_shape(self) -> None:
        vocabs = BattleVocabularies()
        obs = make_turn_obs()
        result = tensorize_turn(obs, vocabs, build_vocab=True)
        assert result["field"].shape == (FIELD_FEATURE_DIM,)

    def test_context_shape(self) -> None:
        vocabs = BattleVocabularies()
        obs = make_turn_obs()
        result = tensorize_turn(obs, vocabs, build_vocab=True)
        assert result["context"].shape == (CONTEXT_FEATURE_DIM,)

    def test_legal_mask_shape(self) -> None:
        vocabs = BattleVocabularies()
        obs = make_turn_obs()
        result = tensorize_turn(obs, vocabs, build_vocab=True)
        assert result["legal_mask"].shape == (NUM_ACTIONS,)

    def test_game_result_win(self) -> None:
        vocabs = BattleVocabularies()
        obs = make_turn_obs(game_won=True)
        result = tensorize_turn(obs, vocabs, build_vocab=True)
        assert result["game_result"] == pytest.approx(1.0)

    def test_game_result_loss(self) -> None:
        vocabs = BattleVocabularies()
        obs = make_turn_obs(game_won=False)
        result = tensorize_turn(obs, vocabs, build_vocab=True)
        assert result["game_result"] == pytest.approx(0.0)


# ── Tests: Battle tensorization ───────────────────────────────────────────


class TestBattleTensorization:
    def test_sequence_output(self) -> None:
        vocabs = BattleVocabularies()
        obs_list = [make_turn_obs(turn_number=t) for t in range(5)]
        result = tensorize_battle(obs_list, vocabs, build_vocab=True)

        assert result["own_team"].shape[0] == 5
        assert result["own_team"].shape[1] == MAX_TEAM_SIZE
        assert result["own_team"].shape[2] == POKEMON_FEATURE_DIM
        assert result["seq_len"] == 5

    def test_max_turns_truncation(self) -> None:
        vocabs = BattleVocabularies()
        obs_list = [make_turn_obs(turn_number=t) for t in range(30)]
        result = tensorize_battle(obs_list, vocabs, build_vocab=True, max_turns=10)
        assert result["own_team"].shape[0] == 10
        assert result["seq_len"] == 10

    def test_empty_observations(self) -> None:
        vocabs = BattleVocabularies()
        result = tensorize_battle([], vocabs, build_vocab=True)
        assert result == {}


# ── Tests: Vocabulary persistence ─────────────────────────────────────────


class TestBattleVocabularies:
    def test_save_and_load(self, tmp_path) -> None:
        vocabs = BattleVocabularies()
        vocabs.species.add("Pikachu")
        vocabs.species.add("Charizard")
        vocabs.moves.add("Thunderbolt")
        vocabs.items.add("Light Ball")
        vocabs.freeze_all()

        vocabs.save(tmp_path)
        loaded = BattleVocabularies.load(tmp_path)

        assert loaded.species.encode("Pikachu") == vocabs.species.encode("Pikachu")
        assert loaded.moves.encode("Thunderbolt") == vocabs.moves.encode("Thunderbolt")
        assert loaded.items.encode("Light Ball") == vocabs.items.encode("Light Ball")
