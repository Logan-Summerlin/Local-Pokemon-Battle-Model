"""Replay parser for Metamon-format battle data.

Parses LZ4-compressed JSON replay files from the Metamon dataset
(jakegrigsby/metamon-parsed-replays) into our internal BattleState
representation. Each file contains one battle trajectory as seen
from one player's perspective.

Supports both Gen 9 and Gen 3 replay formats. Key differences:
- Gen 3: No team preview (opponent_teampreview is empty)
- Gen 3: can_tera is always False, tera_type fields are empty
- Gen 3: No terrain data (terrains introduced in Gen 6)
- Gen 3: Weather from abilities is permanent (e.g., Sand Stream)

Metamon JSON structure:
    {
        "states": [<UniversalState dict>, ...],   # per-turn game states
        "actions": [<action string>, ...]          # actions taken each turn
    }

UniversalState fields (relevant subset):
    - format: str (e.g., "gen9ou", "gen3ou")
    - player_active_pokemon: dict with name, hp_pct, types, item, ability, ...
    - opponent_active_pokemon: dict with same fields
    - available_switches: list of pokemon dicts
    - player_prev_move / opponent_prev_move: move dicts
    - opponents_remaining: int
    - player_conditions / opponent_conditions: str
    - weather / battle_field: str
    - forced_switch: bool
    - can_tera: bool (always False for Gen 3)
    - battle_won / battle_lost: bool
    - opponent_teampreview: list of pokemon dicts (Gen 9 only; empty for Gen 3)
"""

from __future__ import annotations

import json
import logging
import os
import re
import tarfile
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Optional lz4 import
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


# ── Data structures for parsed replays ────────────────────────────────────


@dataclass
class ParsedMove:
    """A move as parsed from replay data."""

    name: str = ""
    move_type: str = ""
    category: str = ""  # Physical, Special, Status
    base_power: int = 0
    accuracy: float = 100.0
    priority: int = 0
    current_pp: int = -1
    max_pp: int = -1


@dataclass
class ParsedPokemon:
    """A Pokemon as parsed from replay data."""

    name: str = ""
    hp_pct: float = 1.0
    types: str = ""
    item: str = ""
    ability: str = ""
    level: int = 100
    status: str = ""
    effect: str = ""
    moves: list[ParsedMove] = dc_field(default_factory=list)
    # Stat boosts
    atk_boost: int = 0
    spa_boost: int = 0
    def_boost: int = 0
    spd_boost: int = 0
    spe_boost: int = 0
    accuracy_boost: int = 0
    evasion_boost: int = 0
    # Base stats
    base_atk: int = 0
    base_spa: int = 0
    base_def: int = 0
    base_spd: int = 0
    base_spe: int = 0
    base_hp: int = 0
    # Gen 9 features
    tera_type: str = ""
    base_species: str = ""


@dataclass
class ParsedTurnState:
    """A single turn state from a replay."""

    format: str = ""
    player_active: ParsedPokemon | None = None
    opponent_active: ParsedPokemon | None = None
    available_switches: list[ParsedPokemon] = dc_field(default_factory=list)
    player_prev_move: ParsedMove | None = None
    opponent_prev_move: ParsedMove | None = None
    opponents_remaining: int = 0
    player_conditions: str = ""
    opponent_conditions: str = ""
    weather: str = ""
    battle_field: str = ""
    forced_switch: bool = False
    can_tera: bool = False
    battle_won: bool = False
    battle_lost: bool = False
    opponent_teampreview: list[ParsedPokemon] = dc_field(default_factory=list)


@dataclass
class ParsedBattle:
    """A fully parsed battle trajectory."""

    battle_id: str = ""
    perspective_id: str = ""  # unique per player POV (battle_id + player)
    format: str = ""
    player_elo: int = 0
    opponent_name: str = ""
    date: str = ""
    result: str = ""  # "WIN" or "LOSS"
    turns: list[ParsedTurnState] = dc_field(default_factory=list)
    actions: list[str] = dc_field(default_factory=list)
    # Metadata parsed from filename
    filename: str = ""

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def won(self) -> bool:
        return self.result == "WIN"

    @property
    def generation(self) -> int:
        """Extract the generation number from the format string.

        Returns 0 if the generation cannot be determined.
        """
        return _parse_generation(self.format)

    @property
    def has_team_preview(self) -> bool:
        """Whether this battle format has team preview.

        Team preview was introduced in Gen 5. Gens 1-4 have no team preview.
        """
        gen = self.generation
        return gen >= 5

    def is_valid(self) -> bool:
        """Check if this battle has valid data for training."""
        if not self.turns or not self.actions:
            return False
        if len(self.actions) > len(self.turns):
            return False
        if self.num_turns < 2:
            return False
        return True


# ── Parsing functions ─────────────────────────────────────────────────────


def _parse_generation(format_str: str) -> int:
    """Extract generation number from a format string like 'gen3ou' or 'gen9ou'.

    Returns 0 if the generation cannot be determined.
    """
    if not format_str:
        return 0
    match = re.match(r"gen(\d+)", format_str.lower())
    if match:
        return int(match.group(1))
    return 0


def _parse_move_dict(d: dict[str, Any] | None) -> ParsedMove:
    """Parse a move dict from Metamon format."""
    if d is None:
        return ParsedMove()
    return ParsedMove(
        name=d.get("name", ""),
        move_type=d.get("move_type", d.get("type", "")),
        category=d.get("category", ""),
        base_power=int(d.get("base_power", 0)),
        accuracy=float(d.get("accuracy", 100.0)),
        priority=int(d.get("priority", 0)),
        current_pp=int(d.get("current_pp", -1)),
        max_pp=int(d.get("max_pp", -1)),
    )


def _parse_pokemon_dict(d: dict[str, Any] | None) -> ParsedPokemon | None:
    """Parse a Pokemon dict from Metamon format."""
    if d is None:
        return None
    return ParsedPokemon(
        name=d.get("name", ""),
        hp_pct=float(d.get("hp_pct", 1.0)),
        types=d.get("types", ""),
        item=d.get("item", ""),
        ability=d.get("ability", ""),
        level=int(d.get("lvl", d.get("level", 100))),
        status=d.get("status", ""),
        effect=d.get("effect", ""),
        moves=[_parse_move_dict(m) for m in d.get("moves", [])],
        atk_boost=int(d.get("atk_boost", 0)),
        spa_boost=int(d.get("spa_boost", 0)),
        def_boost=int(d.get("def_boost", 0)),
        spd_boost=int(d.get("spd_boost", 0)),
        spe_boost=int(d.get("spe_boost", 0)),
        accuracy_boost=int(d.get("accuracy_boost", 0)),
        evasion_boost=int(d.get("evasion_boost", 0)),
        base_atk=int(d.get("base_atk", 0)),
        base_spa=int(d.get("base_spa", 0)),
        base_def=int(d.get("base_def", 0)),
        base_spd=int(d.get("base_spd", 0)),
        base_spe=int(d.get("base_spe", 0)),
        base_hp=int(d.get("base_hp", 0)),
        tera_type=d.get("tera_type", ""),
        base_species=d.get("base_species", ""),
    )


def _parse_teampreview(raw: list[Any]) -> list[ParsedPokemon]:
    """Parse opponent_teampreview, which may be a list of strings or dicts.

    In the real Metamon dataset, teampreview is a list of species name
    strings (e.g., ["ogerpon", "dragapult", ...]). In synthetic/test data
    it may be a list of full pokemon dicts.
    """
    result: list[ParsedPokemon] = []
    for entry in raw:
        if isinstance(entry, str):
            # Real Metamon format: just species names
            if entry:
                result.append(ParsedPokemon(name=entry))
        elif isinstance(entry, dict):
            p = _parse_pokemon_dict(entry)
            if p is not None:
                result.append(p)
    return result


def _parse_turn_state(state_dict: dict[str, Any]) -> ParsedTurnState:
    """Parse a single UniversalState dict into a ParsedTurnState."""
    return ParsedTurnState(
        format=state_dict.get("format", ""),
        player_active=_parse_pokemon_dict(state_dict.get("player_active_pokemon")),
        opponent_active=_parse_pokemon_dict(state_dict.get("opponent_active_pokemon")),
        available_switches=[
            p
            for d in state_dict.get("available_switches", [])
            if (p := _parse_pokemon_dict(d)) is not None
        ],
        player_prev_move=_parse_move_dict(state_dict.get("player_prev_move")),
        opponent_prev_move=_parse_move_dict(state_dict.get("opponent_prev_move")),
        opponents_remaining=int(state_dict.get("opponents_remaining", 0)),
        player_conditions=state_dict.get("player_conditions", ""),
        opponent_conditions=state_dict.get("opponent_conditions", ""),
        weather=state_dict.get("weather", ""),
        battle_field=state_dict.get("battle_field", ""),
        forced_switch=bool(state_dict.get("forced_switch", False)),
        can_tera=bool(state_dict.get("can_tera", False)),
        battle_won=bool(state_dict.get("battle_won", False)),
        battle_lost=bool(state_dict.get("battle_lost", False)),
        opponent_teampreview=_parse_teampreview(state_dict.get("opponent_teampreview", [])),
    )


def parse_filename_metadata(filename: str) -> dict[str, str]:
    """Extract metadata from Metamon replay filename.

    Format: {battleid}_{ELO}_{player}_vs_{opponent}_{DD-MM-YYYY}_{WIN/LOSS}.json.lz4

    Returns dict with keys: battle_id, perspective_id, elo, player, opponent, date, result.
    ``battle_id`` is the shared match identifier (same for both players).
    ``perspective_id`` is ``{battle_id}_{player}`` and uniquely identifies a
    single player's view of the battle so both POVs can be stored separately.
    """
    # Strip extensions
    basename = os.path.basename(filename)
    basename = basename.replace(".json.lz4", "").replace(".json", "")

    parts = basename.split("_")
    result: dict[str, str] = {
        "battle_id": "",
        "perspective_id": "",
        "elo": "0",
        "player": "",
        "opponent": "",
        "date": "",
        "result": "",
    }

    if len(parts) < 6:
        result["battle_id"] = basename
        result["perspective_id"] = basename
        return result

    # The result is the last part (WIN/LOSS)
    result["result"] = parts[-1]

    # The date is second-to-last (DD-MM-YYYY)
    result["date"] = parts[-2]

    # Find "vs" to split player and opponent
    try:
        vs_idx = parts.index("vs")
        result["battle_id"] = parts[0]
        result["elo"] = parts[1] if len(parts) > 1 else "0"
        result["player"] = "_".join(parts[2:vs_idx])
        result["opponent"] = "_".join(parts[vs_idx + 1 : -2])
        result["perspective_id"] = f"{parts[0]}_{result['player']}"
    except ValueError:
        # No "vs" found, use best-effort parsing
        result["battle_id"] = parts[0]
        result["elo"] = parts[1] if len(parts) > 1 else "0"
        result["perspective_id"] = parts[0]

    return result


def _safe_int(value: str | int, default: int = 0) -> int:
    """Safely convert a value to int, returning default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def load_battle_from_json(data: dict[str, Any], filename: str = "") -> ParsedBattle:
    """Load a ParsedBattle from a Metamon JSON dict.

    Args:
        data: Dict with "states" and "actions" keys.
        filename: Original filename for metadata extraction.
    """
    states = data.get("states", [])
    actions = data.get("actions", [])

    # Parse metadata from filename
    meta = parse_filename_metadata(filename)

    # Parse turn states
    turns = [_parse_turn_state(s) for s in states]

    # Determine result from final state
    result = meta.get("result", "")
    if not result and turns:
        last = turns[-1]
        if last.battle_won:
            result = "WIN"
        elif last.battle_lost:
            result = "LOSS"

    # Determine format
    fmt = ""
    if turns:
        fmt = turns[0].format

    return ParsedBattle(
        battle_id=meta.get("battle_id", ""),
        perspective_id=meta.get("perspective_id", meta.get("battle_id", "")),
        format=fmt,
        player_elo=_safe_int(meta.get("elo", "0")),
        opponent_name=meta.get("opponent", ""),
        date=meta.get("date", ""),
        result=result,
        turns=turns,
        actions=[str(a) for a in actions],
        filename=filename,
    )


def load_battle_from_file(filepath: str | Path) -> ParsedBattle:
    """Load a single battle from a .json or .json.lz4 file."""
    filepath = str(filepath)

    if filepath.endswith(".json.lz4"):
        if not HAS_LZ4:
            raise ImportError("lz4 package required to read .json.lz4 files")
        with lz4.frame.open(filepath, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    return load_battle_from_json(data, filename=os.path.basename(filepath))


def iter_battles_from_tar(
    tar_path: str | Path,
    max_battles: int | None = None,
    elo_threshold: int = 0,
    format_filter: str = "",
) -> Iterator[ParsedBattle]:
    """Iterate over battles from a tar.gz archive.

    Args:
        tar_path: Path to gen3ou.tar.gz, gen9ou.tar.gz, or similar archive.
        max_battles: Maximum number of battles to yield.
        elo_threshold: Minimum Elo to include (from filename).
        format_filter: Only include battles matching this format string.

    Yields:
        ParsedBattle objects for each valid battle in the archive.
    """
    count = 0
    errors = 0

    with tarfile.open(str(tar_path), "r:gz") as tf:
        for member in tf:
            if max_battles is not None and count >= max_battles:
                break

            if not member.isfile():
                continue

            name = member.name
            if not (name.endswith(".json.lz4") or name.endswith(".json")):
                continue

            # Quick Elo filter from filename before decompressing
            if elo_threshold > 0:
                meta = parse_filename_metadata(name)
                try:
                    elo = int(meta.get("elo", "0") or "0")
                    if elo < elo_threshold:
                        continue
                except ValueError:
                    continue

            try:
                f = tf.extractfile(member)
                if f is None:
                    continue

                raw_bytes = f.read()

                if name.endswith(".json.lz4"):
                    if not HAS_LZ4:
                        raise ImportError("lz4 required for .json.lz4 files")
                    raw_bytes = lz4.frame.decompress(raw_bytes)

                data = json.loads(raw_bytes.decode("utf-8"))
                battle = load_battle_from_json(data, filename=os.path.basename(name))

                # Format filter
                if format_filter and battle.format and format_filter not in battle.format:
                    continue

                if battle.is_valid():
                    count += 1
                    yield battle
                else:
                    errors += 1

            except Exception as e:
                errors += 1
                if errors <= 10:
                    logger.warning(f"Error parsing {name}: {e}")
                elif errors == 11:
                    logger.warning("Suppressing further parse error messages...")
                continue

    logger.info(
        f"Parsed {count} valid battles from {tar_path} ({errors} errors/skipped)"
    )


def load_battles_from_directory(
    directory: str | Path,
    max_battles: int | None = None,
    elo_threshold: int = 0,
) -> list[ParsedBattle]:
    """Load battles from a directory of .json or .json.lz4 files.

    Args:
        directory: Path to directory containing battle files.
        max_battles: Maximum number of battles to load.
        elo_threshold: Minimum Elo to include.

    Returns:
        List of ParsedBattle objects.
    """
    directory = Path(directory)
    battles = []

    files = sorted(directory.glob("*.json*"))
    for filepath in files:
        if max_battles is not None and len(battles) >= max_battles:
            break

        if not (filepath.suffix == ".json" or filepath.name.endswith(".json.lz4")):
            continue

        # Quick Elo filter
        if elo_threshold > 0:
            meta = parse_filename_metadata(filepath.name)
            try:
                elo = int(meta.get("elo", "0") or "0")
                if elo < elo_threshold:
                    continue
            except ValueError:
                continue

        try:
            battle = load_battle_from_file(filepath)
            if battle.is_valid():
                battles.append(battle)
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")
            continue

    return battles


def iter_battles_from_directory(
    directory: str | Path,
    max_battles: int | None = None,
    elo_threshold: int = 0,
) -> Iterator[ParsedBattle]:
    """Yield battles one at a time from a directory of .json or .json.lz4 files.

    Memory-efficient alternative to load_battles_from_directory for large datasets.

    Args:
        directory: Path to directory containing battle files.
        max_battles: Maximum number of battles to yield.
        elo_threshold: Minimum Elo to include.

    Yields:
        ParsedBattle objects.
    """
    directory = Path(directory)
    count = 0

    files = sorted(directory.glob("*.json*"))
    for filepath in files:
        if max_battles is not None and count >= max_battles:
            break

        if not (filepath.suffix == ".json" or filepath.name.endswith(".json.lz4")):
            continue

        # Quick Elo filter
        if elo_threshold > 0:
            meta = parse_filename_metadata(filepath.name)
            try:
                elo = int(meta.get("elo", "0") or "0")
                if elo < elo_threshold:
                    continue
            except ValueError:
                continue

        try:
            battle = load_battle_from_file(filepath)
            if battle.is_valid():
                count += 1
                yield battle
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")
            continue
