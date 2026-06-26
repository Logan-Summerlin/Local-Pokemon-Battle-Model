"""First-person observation constructor for training data.

Converts ParsedBattle trajectories into per-turn observation dicts
suitable for tensorization. Enforces the Hidden Information Doctrine:
- Own team has full information
- Opponent team has only what has been revealed up to this turn
- Unknown values use explicit "unknown" markers

Supports both Gen 9 (team preview) and Gen 3 (no team preview,
no terrain, permanent weather) formats.

The observation at turn t contains:
- Own team: species, HP fraction, status, boosts, moves, item, ability
- Opponent team: species, HP fraction, status, boosts, revealed moves only
- Field state: weather, terrain, hazards, screens
- Turn context: turn number, previous actions, opponent revealed count
- Legal action mask
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

from src.data.base_stats import lookup_base_stats
from src.data.replay_parser import ParsedBattle, ParsedPokemon, ParsedTurnState
from src.environment.action_space import NUM_ACTIONS

UNKNOWN = "unknown"
MAX_TEAM_SIZE = 6
MAX_MOVES = 4
HISTORY_LENGTH = 20


@dataclass
class PokemonObservation:
    """Observation of a single Pokemon."""

    species: str = ""
    hp_fraction: float = 1.0
    status: str = ""
    is_active: bool = False
    is_fainted: bool = False
    # Moves (up to 4)
    moves: list[str] = dc_field(default_factory=list)
    # Item / ability
    item: str = UNKNOWN
    ability: str = UNKNOWN
    # Stat boosts
    boosts: dict[str, int] = dc_field(default_factory=dict)
    # Base stats (own pokemon only; 0 for opponent = unknown)
    base_stats: dict[str, int] = dc_field(default_factory=dict)
    # Type info
    types: str = ""
    # Level
    level: int = 100
    # Whether this is our pokemon (full info) or opponent (partial)
    is_own: bool = True


@dataclass
class FieldObservation:
    """Observation of the battlefield.

    Fields that don't exist in certain generations (e.g., terrain in Gen 3)
    are kept for tensor-dimension compatibility but set to their default
    (empty/zero) values. The model learns to ignore dead features via
    prune_dead_features or naturally.
    """

    weather: str = ""
    terrain: str = ""
    # Whether ability-set weather is permanent (Gen 3: True, Gen 5+: False)
    weather_permanent: bool = False
    # Own side
    own_stealth_rock: bool = False
    own_spikes: int = 0
    own_toxic_spikes: int = 0
    own_sticky_web: bool = False
    own_reflect: bool = False
    own_light_screen: bool = False
    own_aurora_veil: bool = False
    own_tailwind: bool = False
    # Opponent side
    opp_stealth_rock: bool = False
    opp_spikes: int = 0
    opp_toxic_spikes: int = 0
    opp_sticky_web: bool = False
    opp_reflect: bool = False
    opp_light_screen: bool = False
    opp_aurora_veil: bool = False
    opp_tailwind: bool = False


@dataclass
class TurnObservation:
    """Complete observation at a single decision point."""

    turn_number: int = 0
    # Team observations
    own_team: list[PokemonObservation] = dc_field(default_factory=list)
    opponent_team: list[PokemonObservation] = dc_field(default_factory=list)
    # Field
    field: FieldObservation = dc_field(default_factory=FieldObservation)
    # Action taken (label for training)
    action_taken: str = ""
    # Legal action mask
    legal_action_mask: list[bool] = dc_field(default_factory=lambda: [True] * NUM_ACTIONS)
    # Previous actions for context
    prev_player_move: str = ""
    prev_opponent_move: str = ""
    # Forced switch
    forced_switch: bool = False
    # Number of opponent Pokemon remaining
    opponents_remaining: int = 6
    # Number of distinct opponent Pokemon revealed so far (for no-team-preview gens)
    num_opponent_revealed: int = 0
    # Whether this is the lead turn (turn 0 — significant for Gen 3 lead metagame)
    is_lead_turn: bool = False
    # Game result (for value head training)
    game_won: bool | None = None
    # Game phase info
    game_phase: str = "battle"  # "battle", "finished"


def _parse_conditions(conditions_str: str) -> dict[str, Any]:
    """Parse a conditions string into structured data.

    Conditions are comma-separated, e.g., "Stealth Rock,Spikes:2,Reflect"
    """
    result: dict[str, Any] = {}
    if not conditions_str:
        return result

    for cond in conditions_str.split(","):
        cond = cond.strip()
        if not cond:
            continue
        if ":" in cond:
            name, val = cond.rsplit(":", 1)
            try:
                result[name.strip()] = int(val.strip())
            except ValueError:
                result[name.strip()] = val.strip()
        else:
            result[cond] = True

    return result


def _pokemon_to_own_observation(
    poke: ParsedPokemon | None, is_active: bool = False
) -> PokemonObservation:
    """Convert a ParsedPokemon to an own-team observation (full info)."""
    if poke is None:
        return PokemonObservation(is_own=True)

    return PokemonObservation(
        species=poke.name or poke.base_species,
        hp_fraction=poke.hp_pct,
        status=poke.status,
        is_active=is_active,
        is_fainted=poke.hp_pct <= 0,
        moves=[m.name for m in poke.moves if m.name],
        item=poke.item or UNKNOWN,
        ability=poke.ability or UNKNOWN,
        boosts={
            "atk": poke.atk_boost,
            "def": poke.def_boost,
            "spa": poke.spa_boost,
            "spd": poke.spd_boost,
            "spe": poke.spe_boost,
            "accuracy": poke.accuracy_boost,
            "evasion": poke.evasion_boost,
        },
        base_stats={
            "hp": poke.base_hp,
            "atk": poke.base_atk,
            "def": poke.base_def,
            "spa": poke.base_spa,
            "spd": poke.base_spd,
            "spe": poke.base_spe,
        },
        types=poke.types,
        level=poke.level,
        is_own=True,
    )


def _pokemon_to_opponent_observation(
    poke: ParsedPokemon | None,
    is_active: bool = False,
    revealed_moves: list[str] | None = None,
    revealed_item: str = UNKNOWN,
    revealed_ability: str = UNKNOWN,
) -> PokemonObservation:
    """Convert a ParsedPokemon to an opponent observation (partial info).

    Hidden Information Doctrine: we only include what has been revealed.
    The Metamon dataset provides the opponent's actual info, so we must
    carefully filter to only revealed information.

    Base stats are PUBLIC knowledge — once a species is visible (switched in),
    any player can look up its base stats. This is not hidden information.
    """
    if poke is None:
        return PokemonObservation(is_own=False)

    species = poke.name or poke.base_species

    # Base stats are public knowledge: look up from the crosswalk by species name
    base_stats = lookup_base_stats(species)

    return PokemonObservation(
        species=species,
        hp_fraction=poke.hp_pct,
        status=poke.status,
        is_active=is_active,
        is_fainted=poke.hp_pct <= 0,
        # Only include moves that have been revealed
        moves=revealed_moves if revealed_moves is not None else [],
        # Item/ability: only if revealed
        item=revealed_item,
        ability=revealed_ability,
        boosts={
            "atk": poke.atk_boost,
            "def": poke.def_boost,
            "spa": poke.spa_boost,
            "spd": poke.spd_boost,
            "spe": poke.spe_boost,
            "accuracy": poke.accuracy_boost,
            "evasion": poke.evasion_boost,
        },
        base_stats=base_stats,
        types=poke.types,
        level=poke.level,
        is_own=False,
    )


def _build_field_observation(
    player_conditions: str,
    opponent_conditions: str,
    weather: str,
    battle_field: str,
    generation: int = 9,
) -> FieldObservation:
    """Build field observation from condition strings.

    For Gen 3 (generation <= 3):
    - Terrain is always empty (introduced in Gen 6)
    - Stealth Rock, Toxic Spikes, Sticky Web, Aurora Veil, Tailwind don't exist
    - Weather from abilities is permanent
    - Only Spikes (up to 3 layers), Reflect, and Light Screen exist as side conditions
    """
    own_conds = _parse_conditions(player_conditions)
    opp_conds = _parse_conditions(opponent_conditions)

    is_gen3 = generation <= 3

    # Map terrain from battle_field (always empty for Gen 3)
    terrain = ""
    if battle_field and not is_gen3:
        terrain = battle_field

    return FieldObservation(
        weather=weather,
        terrain=terrain,
        # In Gen 3, ability-set weather (Sand Stream, Drizzle, Drought) is permanent
        weather_permanent=is_gen3 and bool(weather),
        # Stealth Rock: Gen 4+ only
        own_stealth_rock=bool(own_conds.get("Stealth Rock", False)) if not is_gen3 else False,
        # Spikes: Gen 2+ (available in Gen 3)
        own_spikes=int(own_conds.get("Spikes", 0)) if isinstance(own_conds.get("Spikes"), int) else (1 if own_conds.get("Spikes") else 0),
        # Toxic Spikes: Gen 4+ only
        own_toxic_spikes=(int(own_conds.get("Toxic Spikes", 0)) if isinstance(own_conds.get("Toxic Spikes"), int) else (1 if own_conds.get("Toxic Spikes") else 0)) if not is_gen3 else 0,
        # Sticky Web: Gen 6+ only
        own_sticky_web=bool(own_conds.get("Sticky Web", False)) if not is_gen3 else False,
        own_reflect=bool(own_conds.get("Reflect", False)),
        own_light_screen=bool(own_conds.get("Light Screen", False)),
        # Aurora Veil: Gen 7+ only
        own_aurora_veil=bool(own_conds.get("Aurora Veil", False)) if not is_gen3 else False,
        # Tailwind: Gen 4+ only
        own_tailwind=bool(own_conds.get("Tailwind", False)) if not is_gen3 else False,
        # Opponent side — same rules
        opp_stealth_rock=bool(opp_conds.get("Stealth Rock", False)) if not is_gen3 else False,
        opp_spikes=int(opp_conds.get("Spikes", 0)) if isinstance(opp_conds.get("Spikes"), int) else (1 if opp_conds.get("Spikes") else 0),
        opp_toxic_spikes=(int(opp_conds.get("Toxic Spikes", 0)) if isinstance(opp_conds.get("Toxic Spikes"), int) else (1 if opp_conds.get("Toxic Spikes") else 0)) if not is_gen3 else 0,
        opp_sticky_web=bool(opp_conds.get("Sticky Web", False)) if not is_gen3 else False,
        opp_reflect=bool(opp_conds.get("Reflect", False)),
        opp_light_screen=bool(opp_conds.get("Light Screen", False)),
        opp_aurora_veil=bool(opp_conds.get("Aurora Veil", False)) if not is_gen3 else False,
        opp_tailwind=bool(opp_conds.get("Tailwind", False)) if not is_gen3 else False,
    )


class OpponentTracker:
    """Tracks revealed information about opponent Pokemon across turns.

    Enforces the Hidden Information Doctrine by only recording what
    has been explicitly shown during the battle.

    For Gen 3 (no team preview), also tracks which opponent Pokemon
    have been revealed by switching in, ordered by first appearance.
    """

    def __init__(self) -> None:
        # species -> set of revealed move names
        self.revealed_moves: dict[str, list[str]] = {}
        # species -> revealed item
        self.revealed_items: dict[str, str] = {}
        # species -> revealed ability
        self.revealed_abilities: dict[str, str] = {}
        # Ordered list of opponent species revealed (by switch-in), for no-team-preview gens
        self.revealed_species: list[str] = []
        # species -> last known ParsedPokemon state (for building bench observations)
        self._last_known_state: dict[str, ParsedPokemon] = {}

    def update_from_turn(self, turn: ParsedTurnState) -> None:
        """Update revealed information from this turn's state.

        We infer reveals from opponent's active pokemon data:
        - Moves are revealed when used (opponent_prev_move)
        - Items may be shown (e.g., Leftovers recovery, choice lock)
        - Abilities may be shown (e.g., Intimidate on switch-in)
        - Species are revealed when switched in (tracked for no-team-preview)
        """
        opp = turn.opponent_active
        if opp is None:
            return

        species = opp.name or opp.base_species
        if not species:
            return

        # Track newly revealed opponent Pokemon (for no-team-preview gens)
        if species not in self.revealed_species:
            self.revealed_species.append(species)

        # Save last known state for this species
        self._last_known_state[species] = opp

        # Initialize tracking for this species
        if species not in self.revealed_moves:
            self.revealed_moves[species] = []

        # Reveal move if opponent used one
        if turn.opponent_prev_move and turn.opponent_prev_move.name:
            move_name = turn.opponent_prev_move.name
            if move_name not in self.revealed_moves.get(species, []):
                self.revealed_moves.setdefault(species, []).append(move_name)

        # Reveal item if the opponent has a non-empty, non-unknown item
        # In Metamon data, unrevealed items are "unknownitem"
        if opp.item and opp.item.lower() not in (
            "", "unknown", "none", "unknownitem",
        ):
            self.revealed_items[species] = opp.item

        # Reveal ability similarly
        # In Metamon data, unrevealed abilities are "unknownability"
        if opp.ability and opp.ability.lower() not in (
            "", "unknown", "none", "unknownability",
        ):
            self.revealed_abilities[species] = opp.ability

    def get_revealed_moves(self, species: str) -> list[str]:
        return self.revealed_moves.get(species, [])

    def get_revealed_item(self, species: str) -> str:
        return self.revealed_items.get(species, UNKNOWN)

    def get_revealed_ability(self, species: str) -> str:
        return self.revealed_abilities.get(species, UNKNOWN)

    def get_last_known_state(self, species: str) -> ParsedPokemon | None:
        """Get the last known state for a previously revealed opponent Pokemon."""
        return self._last_known_state.get(species)

    @property
    def num_revealed(self) -> int:
        """Number of distinct opponent Pokemon revealed so far."""
        return len(self.revealed_species)


def _build_legal_mask(turn: ParsedTurnState) -> list[bool]:
    """Build a legal action mask from the turn state.

    Uses our canonical action space (Gen 3):
      0-3: move 1-4, 4-8: switch to bench 0-4
    """
    from src.environment.action_space import NUM_ACTIONS

    mask = [False] * NUM_ACTIONS

    if turn.forced_switch:
        # Only switch actions are legal during a forced switch
        num_switches = len(turn.available_switches)
        for i in range(min(num_switches, 5)):
            # Filter out fainted pokemon
            poke = turn.available_switches[i]
            if poke.hp_pct > 0:
                mask[4 + i] = True  # SWITCH_2 + i
    else:
        # Moves are legal if the active pokemon has them
        if turn.player_active:
            num_moves = len(turn.player_active.moves)
            for i in range(min(num_moves, 4)):
                if turn.player_active.moves[i].name:
                    mask[i] = True  # MOVE_1 + i

        # Switch actions
        num_switches = len(turn.available_switches)
        for i in range(min(num_switches, 5)):
            poke = turn.available_switches[i]
            if poke.hp_pct > 0:
                mask[4 + i] = True  # SWITCH_2 + i

    # Ensure at least one action is legal (fallback)
    if not any(mask):
        mask[0] = True

    return mask


def build_observations(battle: ParsedBattle) -> list[TurnObservation]:
    """Convert a ParsedBattle into a list of TurnObservations.

    Each observation represents the first-person state at a decision
    point (before the action is taken). The Hidden Information Doctrine
    is enforced: opponent info only includes what has been revealed.

    Handles both team-preview (Gen 5+) and no-team-preview (Gen 1-4) formats:
    - Gen 5+: Opponent team slots populated from team preview species
    - Gen 1-4: Opponent team built incrementally from revealed switch-ins

    Args:
        battle: A parsed battle trajectory.

    Returns:
        List of TurnObservation, one per decision point.
    """
    observations: list[TurnObservation] = []
    tracker = OpponentTracker()
    game_won = battle.won
    has_team_preview = battle.has_team_preview
    generation = battle.generation

    for t, turn in enumerate(battle.turns):
        # Update opponent tracker BEFORE building observation
        # (reveals from previous turns are visible)
        if t > 0:
            tracker.update_from_turn(battle.turns[t - 1])

        # Also track the current turn's active opponent (for revealed_species)
        # This ensures the current active opponent is in revealed_species
        # even on the first turn they appear
        if turn.opponent_active:
            cur_opp_species = turn.opponent_active.name or turn.opponent_active.base_species
            if cur_opp_species and cur_opp_species not in tracker.revealed_species:
                tracker.revealed_species.append(cur_opp_species)
            if cur_opp_species:
                tracker._last_known_state[cur_opp_species] = turn.opponent_active

        # Build own team observation
        own_team_obs: list[PokemonObservation] = []

        # Active pokemon
        if turn.player_active:
            own_team_obs.append(
                _pokemon_to_own_observation(turn.player_active, is_active=True)
            )

        # Bench pokemon (available switches)
        for switch_poke in turn.available_switches:
            own_team_obs.append(
                _pokemon_to_own_observation(switch_poke, is_active=False)
            )

        # Pad to MAX_TEAM_SIZE
        while len(own_team_obs) < MAX_TEAM_SIZE:
            obs = PokemonObservation(is_own=True, is_fainted=True, hp_fraction=0.0)
            own_team_obs.append(obs)

        # Build opponent team observation (partial info only!)
        opponent_team_obs: list[PokemonObservation] = []

        # Current active opponent (always slot 0 if present)
        active_opp_species = ""
        if turn.opponent_active:
            active_opp_species = turn.opponent_active.name or turn.opponent_active.base_species
            opponent_team_obs.append(
                _pokemon_to_opponent_observation(
                    turn.opponent_active,
                    is_active=True,
                    revealed_moves=tracker.get_revealed_moves(active_opp_species),
                    revealed_item=tracker.get_revealed_item(active_opp_species),
                    revealed_ability=tracker.get_revealed_ability(active_opp_species),
                )
            )

        if has_team_preview:
            # Gen 5+: Fill remaining slots from team preview
            for preview_poke in turn.opponent_teampreview:
                preview_species = preview_poke.name or preview_poke.base_species
                # Skip the active pokemon (already added)
                if preview_species == active_opp_species:
                    continue
                opponent_team_obs.append(
                    _pokemon_to_opponent_observation(
                        preview_poke,
                        is_active=False,
                        revealed_moves=tracker.get_revealed_moves(preview_species),
                        revealed_item=tracker.get_revealed_item(preview_species),
                        revealed_ability=tracker.get_revealed_ability(preview_species),
                    )
                )
        else:
            # Gen 1-4 (no team preview): Fill from revealed species only
            for species in tracker.revealed_species:
                if species == active_opp_species:
                    continue  # Already added as active
                last_state = tracker.get_last_known_state(species)
                if last_state is not None:
                    opponent_team_obs.append(
                        _pokemon_to_opponent_observation(
                            last_state,
                            is_active=False,
                            revealed_moves=tracker.get_revealed_moves(species),
                            revealed_item=tracker.get_revealed_item(species),
                            revealed_ability=tracker.get_revealed_ability(species),
                        )
                    )

        # Pad to MAX_TEAM_SIZE (unrevealed opponent slots stay empty)
        while len(opponent_team_obs) < MAX_TEAM_SIZE:
            obs = PokemonObservation(is_own=False)
            opponent_team_obs.append(obs)

        # Build field observation
        field_obs = _build_field_observation(
            turn.player_conditions,
            turn.opponent_conditions,
            turn.weather,
            turn.battle_field,
            generation=generation,
        )

        # Get action taken (if available)
        action_taken = ""
        if t < len(battle.actions):
            action_taken = str(battle.actions[t])

        # Previous moves for context
        prev_player_move = ""
        prev_opponent_move = ""
        if turn.player_prev_move and turn.player_prev_move.name:
            prev_player_move = turn.player_prev_move.name
        if turn.opponent_prev_move and turn.opponent_prev_move.name:
            prev_opponent_move = turn.opponent_prev_move.name

        # Build legal action mask based on available actions
        legal_mask = _build_legal_mask(turn)

        obs = TurnObservation(
            turn_number=t,
            own_team=own_team_obs[:MAX_TEAM_SIZE],
            opponent_team=opponent_team_obs[:MAX_TEAM_SIZE],
            field=field_obs,
            action_taken=action_taken,
            legal_action_mask=legal_mask,
            forced_switch=turn.forced_switch,
            opponents_remaining=turn.opponents_remaining,
            num_opponent_revealed=tracker.num_revealed,
            is_lead_turn=(t == 0),
            game_won=game_won,
            prev_player_move=prev_player_move,
            prev_opponent_move=prev_opponent_move,
        )

        observations.append(obs)

    return observations
