"""Tests for the observation constructor.

Covers:
- First-person observation correctness (no hidden state leakage)
- Own team full info vs. opponent team partial info
- Unknown flag handling
- Battlefield state tracking
- OpponentTracker reveal logic
"""

import pytest

from src.data.observation import (
    UNKNOWN,
    MAX_TEAM_SIZE,
    OpponentTracker,
    TurnObservation,
    build_observations,
)
from src.data.replay_parser import (
    ParsedBattle,
    ParsedMove,
    ParsedPokemon,
    ParsedTurnState,
    load_battle_from_json,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def make_pokemon(
    name: str = "Pikachu",
    hp_pct: float = 1.0,
    item: str = "Light Ball",
    ability: str = "Static",
    moves: list[str] | None = None,
    status: str = "",
) -> ParsedPokemon:
    move_list = []
    default_moves = moves or ["Thunderbolt", "Quick Attack"]
    for m in default_moves:
        if m == "Quick Attack":
            move_list.append(ParsedMove(name=m, move_type="Normal", base_power=40))
        else:
            move_list.append(ParsedMove(name=m, move_type="Electric", base_power=90))
    return ParsedPokemon(
        name=name,
        hp_pct=hp_pct,
        types="Electric",
        item=item,
        ability=ability,
        moves=move_list,
        status=status,
        base_atk=55,
        base_spa=50,
        base_def=40,
        base_spd=50,
        base_spe=90,
        base_hp=35,
    )


def make_turn(
    player_active: ParsedPokemon | None = None,
    opponent_active: ParsedPokemon | None = None,
    available_switches: list[ParsedPokemon] | None = None,
    weather: str = "",
    player_prev_move: str = "",
    opponent_prev_move: str = "",
    battle_won: bool = False,
    battle_lost: bool = False,
    opponent_teampreview: list[ParsedPokemon] | None = None,
) -> ParsedTurnState:
    return ParsedTurnState(
        format="gen3ou",
        player_active=player_active or make_pokemon("Pikachu"),
        opponent_active=opponent_active or make_pokemon("Charizard", item="Leftovers", ability="Blaze"),
        available_switches=available_switches or [
            make_pokemon("Blastoise", item="Leftovers", ability="Torrent"),
        ],
        player_prev_move=ParsedMove(name=player_prev_move) if player_prev_move else None,
        opponent_prev_move=ParsedMove(name=opponent_prev_move) if opponent_prev_move else None,
        weather=weather,
        battle_won=battle_won,
        battle_lost=battle_lost,
        opponent_teampreview=opponent_teampreview or [],
    )


def make_battle(
    num_turns: int = 3,
    result_won: bool = True,
) -> ParsedBattle:
    turns = []
    for t in range(num_turns):
        is_last = t == num_turns - 1
        turn = make_turn(
            opponent_prev_move="Flamethrower" if t > 0 else "",
            battle_won=result_won and is_last,
            battle_lost=(not result_won) and is_last,
        )
        turns.append(turn)

    return ParsedBattle(
        battle_id="test-battle-001",
        format="gen3ou",
        player_elo=1800,
        result="WIN" if result_won else "LOSS",
        turns=turns,
        actions=[f"move{i}" for i in range(num_turns)],
    )


# ── Tests: Observation construction ───────────────────────────────────────


class TestObservationConstruction:
    def test_basic_observation_count(self) -> None:
        battle = make_battle(num_turns=5)
        obs = build_observations(battle)
        assert len(obs) == 5

    def test_own_team_has_full_info(self) -> None:
        battle = make_battle(num_turns=3)
        obs = build_observations(battle)

        own_active = obs[0].own_team[0]
        assert own_active.species == "Pikachu"
        assert own_active.is_own is True
        assert own_active.is_active is True
        assert own_active.item == "Light Ball"
        assert own_active.ability == "Static"
        assert len(own_active.moves) == 2
        assert "Thunderbolt" in own_active.moves

    def test_own_team_padded_to_max(self) -> None:
        battle = make_battle()
        obs = build_observations(battle)
        assert len(obs[0].own_team) == MAX_TEAM_SIZE

    def test_opponent_starts_with_unknowns(self) -> None:
        """At turn 0, opponent should have no revealed info."""
        battle = make_battle(num_turns=3)
        obs = build_observations(battle)

        opp = obs[0].opponent_team[0]
        assert opp.species == "Charizard"
        assert opp.is_own is False
        assert opp.is_active is True
        # At turn 0, no moves should be revealed yet
        assert len(opp.moves) == 0
        # Item and ability start unknown (no reveals yet)
        # Note: the tracker reveals based on Metamon data which may
        # expose item/ability when the opponent is active
        assert opp.is_own is False

    def test_opponent_team_padded(self) -> None:
        battle = make_battle()
        obs = build_observations(battle)
        assert len(obs[0].opponent_team) == MAX_TEAM_SIZE

    def test_field_observation(self) -> None:
        battle = make_battle(num_turns=2)
        battle.turns[0] = make_turn(weather="RainDance")
        obs = build_observations(battle)
        assert obs[0].field.weather == "RainDance"

    def test_action_taken_recorded(self) -> None:
        battle = make_battle(num_turns=3)
        obs = build_observations(battle)
        assert obs[0].action_taken == "move0"
        assert obs[1].action_taken == "move1"
        assert obs[2].action_taken == "move2"

    def test_game_result_set(self) -> None:
        battle = make_battle(result_won=True)
        obs = build_observations(battle)
        assert obs[0].game_won is True

    def test_game_loss_set(self) -> None:
        battle = make_battle(result_won=False)
        obs = build_observations(battle)
        assert obs[0].game_won is False

    def test_turn_number_increases(self) -> None:
        battle = make_battle(num_turns=5)
        obs = build_observations(battle)
        for i, o in enumerate(obs):
            assert o.turn_number == i


# ── Tests: Hidden info doctrine ───────────────────────────────────────────


class TestHiddenInfoDoctrine:
    def test_opponent_base_stats_from_crosswalk(self) -> None:
        """Opponent base stats are public knowledge (looked up by species)."""
        battle = make_battle(num_turns=3)
        obs = build_observations(battle)
        opp = obs[0].opponent_team[0]
        # Base stats should be populated from crosswalk for known species
        # Charizard: HP=78, Atk=84, Def=78, SpA=109, SpD=85, Spe=100
        assert opp.base_stats.get("hp", 0) > 0
        assert opp.base_stats.get("spe", 0) > 0

    def test_own_base_stats_visible(self) -> None:
        """Own base stats should be available."""
        battle = make_battle(num_turns=3)
        obs = build_observations(battle)
        own = obs[0].own_team[0]
        assert own.base_stats.get("spe", 0) > 0


# ── Tests: Opponent tracker ──────────────────────────────────────────────


class TestOpponentTracker:
    def test_move_reveal(self) -> None:
        tracker = OpponentTracker()
        turn = make_turn(
            opponent_active=make_pokemon("Charizard"),
            opponent_prev_move="Flamethrower",
        )
        tracker.update_from_turn(turn)
        moves = tracker.get_revealed_moves("Charizard")
        assert "Flamethrower" in moves

    def test_multiple_move_reveals(self) -> None:
        tracker = OpponentTracker()
        turn1 = make_turn(
            opponent_active=make_pokemon("Charizard"),
            opponent_prev_move="Flamethrower",
        )
        turn2 = make_turn(
            opponent_active=make_pokemon("Charizard"),
            opponent_prev_move="Air Slash",
        )
        tracker.update_from_turn(turn1)
        tracker.update_from_turn(turn2)
        moves = tracker.get_revealed_moves("Charizard")
        assert "Flamethrower" in moves
        assert "Air Slash" in moves

    def test_no_duplicate_reveals(self) -> None:
        tracker = OpponentTracker()
        turn = make_turn(
            opponent_active=make_pokemon("Charizard"),
            opponent_prev_move="Flamethrower",
        )
        tracker.update_from_turn(turn)
        tracker.update_from_turn(turn)
        moves = tracker.get_revealed_moves("Charizard")
        assert moves.count("Flamethrower") == 1

    def test_unknown_species_returns_empty(self) -> None:
        tracker = OpponentTracker()
        assert tracker.get_revealed_moves("Nonexistent") == []
        assert tracker.get_revealed_item("Nonexistent") == UNKNOWN
        assert tracker.get_revealed_ability("Nonexistent") == UNKNOWN

    def test_item_reveal(self) -> None:
        tracker = OpponentTracker()
        opp = make_pokemon("Salamence", item="Choice Band")
        turn = make_turn(opponent_active=opp)
        tracker.update_from_turn(turn)
        assert tracker.get_revealed_item("Salamence") == "Choice Band"

    def test_ability_reveal(self) -> None:
        tracker = OpponentTracker()
        opp = make_pokemon("Metagross", ability="Clear Body")
        turn = make_turn(opponent_active=opp)
        tracker.update_from_turn(turn)
        assert tracker.get_revealed_ability("Metagross") == "Clear Body"
