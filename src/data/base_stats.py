"""Pokemon base stats crosswalk.

Loads base stats from the CSV file and provides a lookup by species name
using the same normalized naming convention as the species vocabulary
(lowercase, no spaces/hyphens/punctuation, form suffixes appended).

Base stats are PUBLIC knowledge in Pokemon — knowing the opponent's species
(which is revealed on switch-in) is sufficient to know their base stats.
This does NOT violate the Hidden Information Doctrine.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the CSV relative to the project root
_DEFAULT_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "pokedex" / "pokemon_base_stats.csv"

# Base stat keys in the order used by the observation/tensorizer pipeline
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

# Forms that map to the base species name (no suffix appended)
_DEFAULT_FORMS = frozenset({
    "",
    "normal forme",  # Deoxys Normal Forme
    "normal form",
    "male",
})

# Manual overrides for forms with non-obvious normalization.
# Maps (lowercase_name, lowercase_form) -> showdown_id
_FORM_OVERRIDES: dict[tuple[str, str], str] = {
    # Gendered species with distinct forms
    ("nidoran", "female"): "nidoranf",
    ("nidoran", "male"): "nidoranm",
    # Deoxys special formes
    ("deoxys", "attack forme"): "deoxysattack",
    ("deoxys", "defense forme"): "deoxysdefense",
    ("deoxys", "speed forme"): "deoxysspeed",
    # Unown (all forms share base stats)
    ("unown", "a"): "unown",
}


def _normalize_species_name(name: str, form: str) -> str:
    """Convert CSV (Name, Form) pair to the showdown-style species ID.

    Examples:
        ("Bulbasaur", " ") -> "bulbasaur"
        ("Deoxys", "Attack Forme") -> "deoxysattack"
        ("Nidoran", "Female") -> "nidoranf"
    """
    name_lower = name.strip().lower()
    form_lower = form.strip().lower()

    # Check manual overrides first
    override = _FORM_OVERRIDES.get((name_lower, form_lower))
    if override is not None:
        return override

    # Base name: remove non-alphanumeric
    base = re.sub(r"[^a-z0-9]", "", name_lower)

    # Default forms → base name only
    if form_lower in _DEFAULT_FORMS:
        return base

    # Generic: extract form keyword by removing "Forme"/"Form"/"Mode" suffixes
    form_suffix = re.sub(r"\s*(forme?|mode|style|cloak|breed)\s*", "", form_lower).strip()
    form_suffix = re.sub(r"[^a-z0-9]", "", form_suffix)

    return base + form_suffix


class BaseStatsCrosswalk:
    """Lookup table mapping normalized species names to base stats.

    Usage:
        crosswalk = BaseStatsCrosswalk.load()
        stats = crosswalk.get("tyranitar")
        # stats = {"hp": 100, "atk": 134, "def": 110, "spa": 95, "spd": 100, "spe": 61}
    """

    def __init__(self, stats_by_species: dict[str, dict[str, int]]) -> None:
        self._stats = stats_by_species

    def get(self, species: str) -> dict[str, int]:
        """Look up base stats for a species. Returns empty dict if not found."""
        if not species:
            return {}
        # Normalize the query: lowercase, remove non-alphanumeric
        key = re.sub(r"[^a-z0-9]", "", species.lower())
        return dict(self._stats.get(key, {}))

    def __contains__(self, species: str) -> bool:
        key = re.sub(r"[^a-z0-9]", "", species.lower())
        return key in self._stats

    def __len__(self) -> int:
        return len(self._stats)

    @classmethod
    def load(cls, csv_path: str | Path | None = None) -> BaseStatsCrosswalk:
        """Load base stats from the CSV file.

        Args:
            csv_path: Path to pokemon_base_stats.csv. Uses default if None.

        Returns:
            BaseStatsCrosswalk instance.
        """
        path = Path(csv_path) if csv_path else _DEFAULT_CSV_PATH
        if not path.exists():
            logger.warning(f"Base stats CSV not found at {path}, returning empty crosswalk")
            return cls({})

        stats_by_species: dict[str, dict[str, int]] = {}
        duplicates = 0

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name", "").strip().strip('"')
                form = row.get("Form", "").strip().strip('"')
                species_id = _normalize_species_name(name, form)

                stats = {
                    "hp": int(row.get("HP", 0)),
                    "atk": int(row.get("Attack", 0)),
                    "def": int(row.get("Defense", 0)),
                    "spa": int(row.get("Sp. Atk", 0)),
                    "spd": int(row.get("Sp. Def", 0)),
                    "spe": int(row.get("Speed", 0)),
                }

                if species_id in stats_by_species:
                    duplicates += 1
                stats_by_species[species_id] = stats

        # Gen 3: Castform weather forms share base stats
        _STAT_ALIASES: dict[str, str] = {
            "castformsunny": "castform",
            "castformrainy": "castform",
            "castformsnowy": "castform",
        }
        for alias, source in _STAT_ALIASES.items():
            if alias not in stats_by_species and source in stats_by_species:
                stats_by_species[alias] = stats_by_species[source]

        logger.info(
            f"Loaded base stats for {len(stats_by_species)} species from {path}"
            f" ({duplicates} form duplicates resolved by last-wins)"
        )
        return cls(stats_by_species)


# Module-level singleton, lazily loaded
_crosswalk: BaseStatsCrosswalk | None = None


def get_base_stats_crosswalk() -> BaseStatsCrosswalk:
    """Get the module-level base stats crosswalk (lazy singleton)."""
    global _crosswalk
    if _crosswalk is None:
        _crosswalk = BaseStatsCrosswalk.load()
    return _crosswalk


def lookup_base_stats(species: str) -> dict[str, int]:
    """Convenience function: look up base stats for a species name.

    Args:
        species: Species name in any format (will be normalized).

    Returns:
        Dict with keys hp, atk, def, spa, spd, spe. Empty dict if unknown.
    """
    return get_base_stats_crosswalk().get(species)
