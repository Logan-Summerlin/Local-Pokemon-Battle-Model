"""Tests for the base stats crosswalk module.

Covers:
- CSV loading and species name normalization
- Crosswalk lookup for standard and form-variant species
- Coverage of all species in the vocabulary
- Edge cases (empty species, unknown species)
"""

import json
from pathlib import Path

import pytest

from src.data.base_stats import (
    BaseStatsCrosswalk,
    _normalize_species_name,
    lookup_base_stats,
)


# ── Tests: Name normalization ───────────────────────────────────────────


class TestNormalization:
    def test_standard_species(self) -> None:
        assert _normalize_species_name("Bulbasaur", " ") == "bulbasaur"
        assert _normalize_species_name("Charizard", " ") == "charizard"
        assert _normalize_species_name("Pikachu", " ") == "pikachu"

    def test_multi_word_species(self) -> None:
        assert _normalize_species_name("Mr. Mime", " ") == "mrmime"
        assert _normalize_species_name("Ho-Oh", " ") == "hooh"

    def test_deoxys_form_variants(self) -> None:
        assert _normalize_species_name("Deoxys", "Attack Forme") == "deoxysattack"
        assert _normalize_species_name("Deoxys", "Defense Forme") == "deoxysdefense"
        assert _normalize_species_name("Deoxys", "Speed Forme") == "deoxysspeed"

    def test_deoxys_forms(self) -> None:
        assert _normalize_species_name("Deoxys", "Normal Forme") == "deoxys"
        assert _normalize_species_name("Deoxys", "Speed Forme") == "deoxysspeed"
        assert _normalize_species_name("Deoxys", "Defense Forme") == "deoxysdefense"

    def test_special_characters(self) -> None:
        assert _normalize_species_name("Mr. Mime", " ") == "mrmime"
        assert _normalize_species_name("Farfetch'd", " ") == "farfetchd"
        assert _normalize_species_name("Ho-Oh", " ") == "hooh"


# ── Tests: Crosswalk loading ────────────────────────────────────────────


class TestCrosswalkLoading:
    def test_load_from_default_csv(self) -> None:
        cw = BaseStatsCrosswalk.load()
        assert len(cw) > 380  # Gen 1-3: ~392 entries + form aliases

    def test_lookup_standard_species(self) -> None:
        cw = BaseStatsCrosswalk.load()
        stats = cw.get("salamence")
        assert stats["hp"] == 95
        assert stats["atk"] == 135
        assert stats["spe"] == 100

    def test_lookup_form_variant(self) -> None:
        cw = BaseStatsCrosswalk.load()
        stats = cw.get("deoxysspeed")
        assert stats["atk"] == 95
        assert stats["spe"] == 180

    def test_lookup_gen3_pokemon(self) -> None:
        cw = BaseStatsCrosswalk.load()
        stats = cw.get("metagross")
        assert stats["atk"] == 135
        assert stats["spe"] == 70

    def test_lookup_returns_copy(self) -> None:
        """Modifying returned dict shouldn't affect internal state."""
        cw = BaseStatsCrosswalk.load()
        stats = cw.get("pikachu")
        stats["hp"] = 999
        assert cw.get("pikachu")["hp"] == 35

    def test_unknown_species_returns_empty(self) -> None:
        cw = BaseStatsCrosswalk.load()
        assert cw.get("nonexistentmon") == {}
        assert cw.get("") == {}

    def test_contains(self) -> None:
        cw = BaseStatsCrosswalk.load()
        assert "pikachu" in cw
        assert "nonexistent" not in cw

    def test_alias_coverage(self) -> None:
        """Aliases for forms not in CSV should be populated."""
        cw = BaseStatsCrosswalk.load()
        # Deoxys form aliases should all resolve
        assert cw.get("deoxysattack") != {}
        assert cw.get("deoxysdefense") != {}
        # Castform forms share base stats with base Castform
        assert cw.get("castformsunny") == cw.get("castform")
        assert cw.get("castformrainy") == cw.get("castform")


# ── Tests: Vocab coverage ──────────────────────────────────────────────


class TestVocabCoverage:
    def test_all_vocab_species_have_base_stats(self) -> None:
        """Every species in the frozen vocabulary should have base stats."""
        vocab_path = Path("data/processed/vocabs/species.json")
        if not vocab_path.exists():
            pytest.skip("Species vocabulary not found")

        with open(vocab_path) as f:
            vocab = json.load(f)

        cw = BaseStatsCrosswalk.load()
        missing = []
        for species in vocab["tokens"]:
            if not species or species == "unknown":
                continue
            if not cw.get(species):
                missing.append(species)

        assert missing == [], f"Species missing from crosswalk: {missing}"


# ── Tests: Convenience function ─────────────────────────────────────────


class TestLookupFunction:
    def test_lookup_base_stats(self) -> None:
        stats = lookup_base_stats("charizard")
        assert stats["hp"] == 78
        assert stats["spe"] == 100

    def test_lookup_normalizes_input(self) -> None:
        stats = lookup_base_stats("Charizard")
        assert stats["hp"] == 78
