"""Auxiliary label extraction for hidden-info prediction head.

Extracts training labels for the auxiliary head from replay data.
Labels are derived from information that was EVENTUALLY revealed
during the battle — never from omniscient data.

Label types (Gen 3 OU):
    - Item class: Categorizes opponent items into ~25 classes
    - Speed bucket: Ordinal speed category (very fast -> very slow)
    - Role archetype: Sweeper, wall, pivot, trapper, etc. (inferred from moves/stats)
    - Move family presence: Priority, recovery, hazards, status, etc.

Key constraint (Hidden Information Doctrine):
    If an item was revealed on turn 15, we use that label for turns 1-14
    of THAT game. If it was never revealed, we use -1 (unknown/masked).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.data.observation import MAX_TEAM_SIZE, UNKNOWN
from src.data.replay_parser import ParsedBattle, ParsedPokemon, ParsedTurnState

logger = logging.getLogger(__name__)


# ── Item classification ──────────────────────────────────────────────────

# Map individual items to broader item classes for the auxiliary head.
# These classes group items by competitive function (Gen 3 OU).
ITEM_CLASS_MAP: dict[str, int] = {}
ITEM_CLASSES: list[str] = [
    "leftovers",            # 0 - by far the most common Gen 3 item
    "choiceband",           # 1
    "lumberry",             # 2
    "liechiberry",          # 3 (pinch berries - Atk boost)
    "petayaberry",          # 4 (pinch berries - SpA boost)
    "salacberry",           # 5 (pinch berries - Spe boost)
    "sitrusberry",          # 6
    "focusband",            # 7
    "shellbell",            # 8
    "whiteherb",            # 9
    "mentalherb",           # 10
    "leppaberry",           # 11
    "chestoberry",          # 12 (status cure - Rest/Sleep)
    "lumberry_cure",        # 13 (status cure berries grouped)
    "soothebell",           # 14
    "typeboost",            # 15 (type-boosting items: Charcoal, Mystic Water, etc.)
    "scopelens",            # 16
    "kingsrock",            # 17
    "brightpowder",         # 18
    "quickclaw",            # 19
    "magoberry",            # 20 (pinch berries - HP recovery)
    "aguavberry",           # 21 (pinch berries - HP recovery)
    "lightclay",            # 22
    "other",                # 23
    "noitem",               # 24
]

# Build reverse mapping
for i, item_name in enumerate(ITEM_CLASSES):
    ITEM_CLASS_MAP[item_name] = i

# Type-boosting items map to the typeboost class
_TYPE_BOOST_ITEMS = [
    "charcoal", "mysticwater", "magnet", "miracleseed",
    "nevermeltice", "blackbelt", "poisonbarb", "sharpbeak", "silkscarf",
    "silverpowder", "softsand", "spelltag", "twistedspoon", "dragonfang",
    "blackglasses", "hardstone", "metalcoat",
]
for item in _TYPE_BOOST_ITEMS:
    if item not in ITEM_CLASS_MAP:
        ITEM_CLASS_MAP[item] = ITEM_CLASS_MAP["typeboost"]

# Pinch berries (HP recovery) map to magoberry/aguavberry class
_PINCH_HP_BERRIES = [
    "magoberry", "aguavberry", "figyberry", "iapapaberry", "wikiberry",
]
for berry in _PINCH_HP_BERRIES:
    if berry not in ITEM_CLASS_MAP:
        ITEM_CLASS_MAP[berry] = ITEM_CLASS_MAP["magoberry"]

# Status cure berries map to lumberry_cure class
_STATUS_CURE_BERRIES = [
    "cheriberry", "rawstberry", "aspearberry", "pechaberry", "persimberry",
]
for berry in _STATUS_CURE_BERRIES:
    if berry not in ITEM_CLASS_MAP:
        ITEM_CLASS_MAP[berry] = ITEM_CLASS_MAP["lumberry_cure"]

NUM_ITEM_CLASSES = len(ITEM_CLASSES)  # 25


def classify_item(item_name: str) -> int:
    """Map an item name to its class index."""
    if not item_name or item_name.lower() in ("", "none", "unknown", "unknownitem"):
        return ITEM_CLASS_MAP["noitem"]
    normalized = item_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    return ITEM_CLASS_MAP.get(normalized, ITEM_CLASS_MAP["other"])


# ── Speed bucket classification ──────────────────────────────────────────

# Speed buckets based on base speed stat thresholds (Gen 3 OU meta)
# Very fast (>=110): Aerodactyl, Starmie, Jolteon, Dugtrio
# Fast (90-109): Gengar, Salamence, Celebi, Jirachi
# Medium (65-89): Suicune, Metagross, Heracross
# Slow (40-64): Swampert, Skarmory, Blissey
# Very slow (<40): Snorlax, Dusclops
SPEED_THRESHOLDS = [110, 90, 65, 40]  # Boundaries between buckets
NUM_SPEED_BUCKETS = 5

SPEED_BUCKET_NAMES = ["very_fast", "fast", "medium", "slow", "very_slow"]


def classify_speed(base_spe: int) -> int:
    """Map base speed stat to a speed bucket index."""
    if base_spe >= SPEED_THRESHOLDS[0]:
        return 0  # very fast
    elif base_spe >= SPEED_THRESHOLDS[1]:
        return 1  # fast
    elif base_spe >= SPEED_THRESHOLDS[2]:
        return 2  # medium
    elif base_spe >= SPEED_THRESHOLDS[3]:
        return 3  # slow
    else:
        return 4  # very slow


# ── Role archetype classification ────────────────────────────────────────

NUM_ROLE_ARCHETYPES = 8
ROLE_NAMES = [
    "physical_sweeper",    # 0
    "special_sweeper",     # 1
    "mixed_attacker",      # 2
    "physical_wall",       # 3
    "special_wall",        # 4
    "pivot",               # 5
    "hazard_setter",       # 6
    "support",             # 7
]

# Known pivoting moves (Gen 3: only Baton Pass; U-turn/Volt Switch don't exist)
_PIVOT_MOVES = {"batonpass"}

# Known hazard-setting moves (Gen 3: only Spikes; no Stealth Rock, Toxic Spikes, Sticky Web)
_HAZARD_MOVES = {"spikes"}

# Recovery moves (Gen 3: no Roost, Shore Up, Strength Sap)
_RECOVERY_MOVES = {
    "recover", "softboiled", "moonlight", "morningsun", "synthesis",
    "slackoff", "milkdrink", "wish", "rest",
}

# Status/support moves (Gen 3: no Defog, no Aurora Veil)
_SUPPORT_MOVES = {
    "willowisp", "thunderwave", "toxic", "aromatherapy", "healbell",
    "rapidspin", "haze", "whirlwind", "roar", "yawn",
    "encore", "taunt", "trick", "knockoff", "reflect", "lightscreen",
}

# Setup moves (Gen 3: no Nasty Plot, Shell Smash, Quiver Dance, Tidy Up, Victory Dance, etc.)
_SETUP_MOVES = {
    "swordsdance", "calmmind", "dragondance", "bulkup",
    "irondefense", "amnesia", "cosmicpower",
    "curse", "agility", "bellydrum", "meditate",
}


def _normalize_move(name: str) -> str:
    """Normalize move name for matching."""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def classify_role(
    poke: ParsedPokemon,
    moves_used: list[str] | None = None,
) -> int:
    """Classify a pokemon's competitive role based on stats and moves.

    Uses base stats + known move pool to infer role archetype.
    """
    if poke is None:
        return 7  # support as default

    # Normalize moves
    all_moves: set[str] = set()
    if moves_used:
        all_moves.update(_normalize_move(m) for m in moves_used)
    for m in poke.moves:
        if m.name:
            all_moves.update([_normalize_move(m.name)])

    # Check for pivot moves
    if all_moves & _PIVOT_MOVES:
        return 5  # pivot

    # Check for hazard setter
    if all_moves & _HAZARD_MOVES:
        return 6  # hazard_setter

    # Check attack stats
    atk = poke.base_atk
    spa = poke.base_spa
    def_stat = poke.base_def
    spd = poke.base_spd
    hp = poke.base_hp

    # Bulk score
    phys_bulk = hp * def_stat
    spec_bulk = hp * spd
    total_bulk = phys_bulk + spec_bulk

    # Check for heavy recovery/support
    support_count = len(all_moves & (_RECOVERY_MOVES | _SUPPORT_MOVES))

    # If many support moves and low offensive stats -> wall/support
    if support_count >= 2:
        if total_bulk > 50000 or (hp > 80 and (def_stat > 100 or spd > 100)):
            if def_stat > spd:
                return 3  # physical_wall
            else:
                return 4  # special_wall

    if support_count >= 3:
        return 7  # support

    # Check for setup sweepers
    has_setup = bool(all_moves & _SETUP_MOVES)

    # Offensive classification
    if atk > spa + 20:
        if has_setup or atk >= 100:
            return 0  # physical_sweeper
    elif spa > atk + 20:
        if has_setup or spa >= 100:
            return 1  # special_sweeper
    else:
        if (atk >= 80 and spa >= 80) or has_setup:
            return 2  # mixed_attacker

    # Fallback: use bulk vs offense
    offense = max(atk, spa)
    if offense > 90:
        return 0 if atk >= spa else 1
    elif total_bulk > 50000:
        return 3 if def_stat > spd else 4
    else:
        return 7  # support


# ── Move family classification ───────────────────────────────────────────

NUM_MOVE_FAMILIES = 10
MOVE_FAMILY_NAMES = [
    "priority",         # 0: Quick Attack, Mach Punch, ExtremeSpeed, Fake Out
    "recovery",         # 1: Recover, Softboiled, Rest, Wish, etc.
    "hazard_setup",     # 2: Spikes only in Gen 3
    "hazard_removal",   # 3: Rapid Spin only in Gen 3
    "status_move",      # 4: Will-O-Wisp, Thunder Wave, Toxic, etc.
    "setup_boost",      # 5: Swords Dance, Dragon Dance, Calm Mind, etc.
    "pivot_move",       # 6: Baton Pass only in Gen 3
    "screen_move",      # 7: Reflect, Light Screen
    "phazing_move",     # 8: Whirlwind, Roar, Haze
    "trick_move",       # 9: Trick, Knock Off, Thief
]

# Gen 3 priority moves (no Shadow Sneak, Sucker Punch, Bullet Punch, Aqua Jet, Ice Shard)
_PRIORITY_MOVES = {
    "extremespeed", "machpunch", "quickattack", "fakeout",
}

# Gen 3 hazard removal (no Defog, Court Change, Tidy Up, Mortal Spin)
_HAZARD_REMOVAL_MOVES = {"rapidspin"}

# Gen 3 screens (no Aurora Veil)
_SCREEN_MOVES = {"reflect", "lightscreen"}

# Gen 3 phazing (no Dragon Tail, Circle Throw)
_PHAZING_MOVES = {"whirlwind", "roar", "haze"}

# Gen 3 trick/disruption (no Switcheroo)
_TRICK_MOVES = {"trick", "knockoff", "thief", "covet"}


def classify_move_families(moves: list[str]) -> list[int]:
    """Classify which move families are present in a moveset.

    Returns:
        List of 10 binary values (0/1) indicating family presence.
    """
    families = [0] * NUM_MOVE_FAMILIES
    if not moves:
        return families

    normalized = {_normalize_move(m) for m in moves if m}

    if normalized & _PRIORITY_MOVES:
        families[0] = 1
    if normalized & _RECOVERY_MOVES:
        families[1] = 1
    if normalized & _HAZARD_MOVES:
        families[2] = 1
    if normalized & _HAZARD_REMOVAL_MOVES:
        families[3] = 1
    if normalized & _SUPPORT_MOVES:
        families[4] = 1
    if normalized & _SETUP_MOVES:
        families[5] = 1
    if normalized & _PIVOT_MOVES:
        families[6] = 1
    if normalized & _SCREEN_MOVES:
        families[7] = 1
    if normalized & _PHAZING_MOVES:
        families[8] = 1
    if normalized & _TRICK_MOVES:
        families[9] = 1

    return families


# ── End-of-game label extraction ─────────────────────────────────────────


def extract_opponent_labels(
    battle: ParsedBattle,
) -> dict[str, dict[str, Any]]:
    """Extract auxiliary labels for each opponent pokemon from a battle.

    Scans the full battle to find eventually-revealed information about
    opponent pokemon. Only uses information that was visible at some
    point during the battle (items shown via Knock Off, abilities
    activated, moves used, etc.)

    Returns:
        Dict mapping opponent species -> label dict with:
            item_class: int (0-24) or -1
            speed_bucket: int (0-4) or -1
            role: int (0-7) or -1
            move_families: list[int] of length 10, or all -1 if unknown
    """
    labels: dict[str, dict[str, Any]] = {}
    revealed_moves: dict[str, list[str]] = {}
    revealed_items: dict[str, str] = {}
    revealed_abilities: dict[str, str] = {}
    pokemon_data: dict[str, ParsedPokemon] = {}  # Store last seen data

    for turn in battle.turns:
        opp = turn.opponent_active
        if opp is None:
            continue
        species = opp.name or opp.base_species
        if not species:
            continue

        # Store pokemon data (may contain base stats in Metamon format)
        pokemon_data[species] = opp

        # Track revealed moves
        if species not in revealed_moves:
            revealed_moves[species] = []
        for m in opp.moves:
            if m.name and m.name not in revealed_moves[species]:
                revealed_moves[species].append(m.name)
        if turn.opponent_prev_move and turn.opponent_prev_move.name:
            mn = turn.opponent_prev_move.name
            if mn not in revealed_moves.get(species, []):
                revealed_moves.setdefault(species, []).append(mn)

        # Track revealed items
        if opp.item and opp.item.lower() not in ("", "unknown", "none", "unknownitem"):
            revealed_items[species] = opp.item

        # Track revealed abilities
        if opp.ability and opp.ability.lower() not in ("", "unknown", "none", "unknownability"):
            revealed_abilities[species] = opp.ability

    # Also check team preview pokemon
    for turn in battle.turns:
        for preview_poke in turn.opponent_teampreview:
            species = preview_poke.name or preview_poke.base_species
            if species and species not in pokemon_data:
                pokemon_data[species] = preview_poke

    # Build labels for each opponent pokemon
    for species, poke in pokemon_data.items():
        label: dict[str, Any] = {
            "item_class": -1,
            "speed_bucket": -1,
            "role": -1,
            "move_families": [-1] * NUM_MOVE_FAMILIES,
        }

        # Item label: only if revealed during battle
        if species in revealed_items:
            label["item_class"] = classify_item(revealed_items[species])

        # Speed bucket: from base stats (available in Metamon data)
        if poke.base_spe > 0:
            label["speed_bucket"] = classify_speed(poke.base_spe)

        # Role: infer from stats + moves
        moves = revealed_moves.get(species, [])
        # Also include moves from pokemon data
        for m in poke.moves:
            if m.name and m.name not in moves:
                moves.append(m.name)
        if poke.base_atk > 0 or poke.base_spa > 0 or moves:
            label["role"] = classify_role(poke, moves)

        # Move families: from all known moves
        if moves:
            label["move_families"] = classify_move_families(moves)

        labels[species] = label

    return labels


def build_auxiliary_targets(
    battle: ParsedBattle,
    max_team_size: int = MAX_TEAM_SIZE,
) -> dict[str, np.ndarray]:
    """Build auxiliary target tensors for a full battle.

    Returns arrays shaped for per-turn training:
        item_targets: (num_turns, 6) int64 - item class per opponent slot
        speed_targets: (num_turns, 6) int64
        role_targets: (num_turns, 6) int64
        move_family_targets: (num_turns, 6, 10) int64

    Labels are constant across all turns (we use end-of-battle knowledge
    applied retroactively, as specified by the Hidden Information Doctrine
    for label sourcing).
    """
    # Extract labels from the full battle
    opp_labels = extract_opponent_labels(battle)

    num_turns = len(battle.turns)

    item_targets = np.full((num_turns, max_team_size), -1, dtype=np.int64)
    speed_targets = np.full((num_turns, max_team_size), -1, dtype=np.int64)
    role_targets = np.full((num_turns, max_team_size), -1, dtype=np.int64)
    move_family_targets = np.full(
        (num_turns, max_team_size, NUM_MOVE_FAMILIES), -1, dtype=np.int64
    )

    for t, turn in enumerate(battle.turns):
        # Build opponent ordering consistent with observation builder
        # Gen 3: no team preview, so only use active + revealed species
        opp_species_order: list[str] = []

        if turn.opponent_active:
            sp = turn.opponent_active.name or turn.opponent_active.base_species
            if sp:
                opp_species_order.append(sp)

        for preview_poke in turn.opponent_teampreview:
            sp = preview_poke.name or preview_poke.base_species
            if sp and sp not in opp_species_order:
                opp_species_order.append(sp)

        # Fill targets for each slot
        for slot_idx, species in enumerate(opp_species_order[:max_team_size]):
            if species in opp_labels:
                lab = opp_labels[species]
                item_targets[t, slot_idx] = lab["item_class"]
                speed_targets[t, slot_idx] = lab["speed_bucket"]
                role_targets[t, slot_idx] = lab["role"]
                families = lab["move_families"]
                move_family_targets[t, slot_idx] = np.array(families, dtype=np.int64)

    return {
        "item_targets": item_targets,
        "speed_targets": speed_targets,
        "role_targets": role_targets,
        "move_family_targets": move_family_targets,
    }


def build_turn_auxiliary_targets(
    battle: ParsedBattle,
    turn_index: int,
    max_team_size: int = MAX_TEAM_SIZE,
) -> dict[str, np.ndarray]:
    """Build auxiliary targets for a single turn.

    Returns:
        Dict with per-slot target arrays (without turn dimension).
    """
    full = build_auxiliary_targets(battle, max_team_size)
    return {
        "item_targets": full["item_targets"][turn_index],
        "speed_targets": full["speed_targets"][turn_index],
        "role_targets": full["role_targets"][turn_index],
        "move_family_targets": full["move_family_targets"][turn_index],
    }
