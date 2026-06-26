#!/usr/bin/env python3
"""Prepare curriculum training data from the Metamon Gen 3 OU dataset.

Downloads (or reuses cached) gen3ou.tar.gz, scans all battles by Elo,
identifies which battles are already processed, extracts only new ones,
processes them, and creates stage manifests for curriculum training.

Stage 1 (Foundation): 50K battles from 1100-1300 Elo
Stage 2 (Specialization): All 1300+ Elo battles (~108K)

Usage:
    python scripts/prepare_curriculum_data.py
    python scripts/prepare_curriculum_data.py --tar-path /tmp/gen3ou.tar.gz
    python scripts/prepare_curriculum_data.py --skip-download  # if tar already cached
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BATTLES_DIR = PROCESSED_DIR / "battles"
CURRICULUM_DIR = DATA_DIR / "curriculum"


def download_tar(cache_dir: str = "/tmp/metamon_cache") -> str:
    """Download gen3ou.tar.gz from HuggingFace."""
    from huggingface_hub import hf_hub_download

    logger.info("Downloading gen3ou.tar.gz from jakegrigsby/metamon-parsed-replays...")
    path = hf_hub_download(
        repo_id="jakegrigsby/metamon-parsed-replays",
        filename="gen3ou.tar.gz",
        repo_type="dataset",
        revision="main",
        cache_dir=cache_dir,
    )
    size_gb = os.path.getsize(path) / 1e9
    logger.info(f"Downloaded to: {path} ({size_gb:.2f} GB)")
    return path


def parse_elo_from_tar_name(basename: str) -> int | None:
    """Extract Elo from a Metamon raw replay filename.

    Format: {showdown_id}_{ELO}_{player}_vs_{opponent}_{date}_{result}.json.lz4
    """
    parts = basename.replace(".json.lz4", "").replace(".json", "").split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def parse_battle_id_from_tar_name(basename: str) -> str | None:
    """Extract the numeric battle ID from a raw filename.

    Format: gen3ou-{BATTLEID}_{ELO}_{player}_vs_...
    Returns just the numeric BATTLEID to match npz filenames.
    """
    m = re.search(r"gen3ou-(\d+)_", basename)
    return m.group(1) if m else None


def parse_battle_id_from_npz(npz_name: str) -> str | None:
    """Extract the Showdown battle ID from a processed npz filename.

    Format: [prefix-]gen3ou-{BATTLEID}_{username}.npz
    """
    m = re.search(r"gen3ou-(\d+)_", npz_name)
    return m.group(1) if m else None


def scan_tar(tar_path: str) -> dict[str, list[tuple[str, int]]]:
    """Scan the tar archive and categorize all battles by Elo bin.

    Returns dict mapping Elo bin label -> list of (member_name, elo).
    Also builds a battle_id -> elo mapping for all entries.
    """
    logger.info("Scanning archive for all battles...")
    bins: dict[str, list[tuple[str, int]]] = defaultdict(list)
    battle_id_to_elo: dict[str, int] = {}
    total = 0
    skipped = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if not (name.endswith(".json.lz4") or name.endswith(".json")):
                continue

            total += 1
            basename = os.path.basename(name)
            elo = parse_elo_from_tar_name(basename)
            battle_id = parse_battle_id_from_tar_name(basename)

            if elo is None or battle_id is None:
                skipped += 1
                continue

            # Store elo for this battle (may see same battle_id from different years/dates)
            battle_id_to_elo[battle_id] = elo

            if elo < 1100:
                bins["below_1100"].append((name, elo))
            elif elo < 1200:
                bins["1100-1200"].append((name, elo))
            elif elo < 1300:
                bins["1200-1300"].append((name, elo))
            elif elo < 1400:
                bins["1300-1400"].append((name, elo))
            elif elo < 1500:
                bins["1400-1500"].append((name, elo))
            else:
                bins["1500+"].append((name, elo))

            if total % 50000 == 0:
                logger.info(f"  Scanned {total} files...")

    logger.info(f"Scan complete: {total} total files ({skipped} skipped)")
    for label in ["below_1100", "1100-1200", "1200-1300", "1300-1400", "1400-1500", "1500+"]:
        logger.info(f"  {label}: {len(bins.get(label, []))} replays")

    return bins, battle_id_to_elo


def get_already_processed_battle_ids() -> set[str]:
    """Get the set of battle IDs already processed in data/processed/battles/."""
    processed = set()
    if not BATTLES_DIR.exists():
        return processed

    for fname in os.listdir(BATTLES_DIR):
        if fname.endswith(".npz"):
            bid = parse_battle_id_from_npz(fname)
            if bid:
                processed.add(bid)

    logger.info(f"Found {len(processed)} unique battle IDs already processed")
    return processed


def select_battles_for_stages(
    bins: dict[str, list[tuple[str, int]]],
    stage1_size: int = 50000,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Select battles for Stage 1 and Stage 2.

    Stage 1: stage1_size battles from 1100-1300 Elo (foundation)
    Stage 2: ALL 1300+ Elo battles (specialization)

    Returns (stage1_members, stage2_members) - lists of tar member names.
    """
    rng = random.Random(seed)

    # Stage 2: all 1300+ (deterministic - take everything)
    stage2_members = []
    for label in ["1300-1400", "1400-1500", "1500+"]:
        stage2_members.extend([name for name, _elo in bins.get(label, [])])

    logger.info(f"Stage 2: {len(stage2_members)} replays (all 1300+ Elo)")

    # Stage 1: sample from 1100-1300
    pool_1100_1300 = []
    for label in ["1100-1200", "1200-1300"]:
        pool_1100_1300.extend([name for name, _elo in bins.get(label, [])])

    if len(pool_1100_1300) <= stage1_size:
        stage1_members = pool_1100_1300
        logger.info(f"Stage 1: taking all {len(stage1_members)} replays from 1100-1300")
    else:
        stage1_members = rng.sample(pool_1100_1300, stage1_size)
        logger.info(f"Stage 1: sampled {len(stage1_members)} from {len(pool_1100_1300)} (1100-1300)")

    return stage1_members, stage2_members


def extract_new_battles(
    tar_path: str,
    members_to_extract: list[str],
    already_processed: set[str],
    output_dir: Path,
) -> list[str]:
    """Extract only battles not already processed.

    Returns list of extracted filenames.
    """
    # Figure out which members correspond to new (unprocessed) battles
    new_members = []
    skipped = 0
    for member_name in members_to_extract:
        basename = os.path.basename(member_name)
        battle_id = parse_battle_id_from_tar_name(basename)
        if battle_id and battle_id in already_processed:
            skipped += 1
            continue
        new_members.append(member_name)

    logger.info(
        f"Need to extract {len(new_members)} new replays "
        f"({skipped} already processed, skipped)"
    )

    if not new_members:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    new_set = set(new_members)
    extracted = []
    count = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if member.name not in new_set:
                continue

            f = tf.extractfile(member)
            if f is None:
                continue

            raw_bytes = f.read()
            basename = os.path.basename(member.name)
            output_path = output_dir / basename

            with open(output_path, "wb") as out:
                out.write(raw_bytes)

            extracted.append(basename)
            count += 1

            if count % 5000 == 0:
                logger.info(f"  Extracted {count}/{len(new_members)} new replays...")

    logger.info(f"Extracted {count} new replays to {output_dir}")
    return extracted


def process_new_battles(raw_dir: Path, max_battles: int | None = None) -> int:
    """Process raw .json.lz4 files into tensorized .npz files.

    Only processes files that don't already have a corresponding .npz.
    Returns count of newly processed battles.
    """
    import numpy as np

    from src.data.observation import build_observations
    from src.data.priors import MetagamePriors
    from src.data.replay_parser import iter_battles_from_directory
    from src.data.tensorizer import BattleVocabularies, tensorize_battle

    # Load existing vocabs if available
    vocabs = BattleVocabularies()
    priors = MetagamePriors()

    BATTLES_DIR.mkdir(parents=True, exist_ok=True)

    # Get already-processed perspective IDs to skip
    existing_npz = {f.stem for f in BATTLES_DIR.glob("*.npz")}

    count = 0
    skipped = 0

    logger.info(f"Processing new battles from {raw_dir}...")
    for battle in iter_battles_from_directory(
        str(raw_dir),
        max_battles=max_battles,
    ):
        file_id = battle.perspective_id or battle.battle_id
        if file_id in existing_npz:
            skipped += 1
            continue

        observations = build_observations(battle)
        if not observations:
            continue

        seq = tensorize_battle(
            observations, vocabs, build_vocab=True, max_turns=20
        )
        if not seq:
            continue

        battle_file = BATTLES_DIR / f"{file_id}.npz"
        np.savez_compressed(str(battle_file), **seq)

        count += 1
        if count % 2000 == 0:
            logger.info(f"  Processed {count} new battles ({skipped} skipped)...")

    logger.info(f"Processing complete: {count} new battles ({skipped} skipped)")

    # Save updated vocabs
    vocabs_dir = PROCESSED_DIR / "vocabs" / "gen3"
    vocabs_dir.mkdir(parents=True, exist_ok=True)
    vocabs.save(str(vocabs_dir))
    logger.info(f"Vocabs saved to {vocabs_dir}")

    return count


def build_curriculum_manifests(
    stage1_tar_members: list[str],
    stage2_tar_members: list[str],
    battle_id_to_elo: dict[str, int],
) -> None:
    """Build manifest JSON files mapping stage -> list of processed npz filenames.

    Each manifest maps battle IDs selected for that stage to their actual
    npz filenames on disk (handling the perspective_id naming).
    """
    CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)

    # Build battle_id sets for each stage from tar member names
    stage1_battle_ids = set()
    for name in stage1_tar_members:
        bid = parse_battle_id_from_tar_name(os.path.basename(name))
        if bid:
            stage1_battle_ids.add(bid)

    stage2_battle_ids = set()
    for name in stage2_tar_members:
        bid = parse_battle_id_from_tar_name(os.path.basename(name))
        if bid:
            stage2_battle_ids.add(bid)

    # Map battle IDs to actual npz files on disk
    all_npz = list(BATTLES_DIR.glob("*.npz"))
    logger.info(f"Mapping {len(all_npz)} npz files to stage manifests...")

    stage1_files = []
    stage2_files = []

    for npz_path in all_npz:
        bid = parse_battle_id_from_npz(npz_path.name)
        if not bid:
            continue
        if bid in stage1_battle_ids:
            stage1_files.append(npz_path.name)
        if bid in stage2_battle_ids:
            stage2_files.append(npz_path.name)

    # Sort for reproducibility
    stage1_files.sort()
    stage2_files.sort()

    # Save manifests
    stage1_manifest = {
        "stage": "foundation",
        "description": "1100-1300 Elo battles for learning basic mechanics",
        "elo_range": "1100-1300",
        "num_battles": len(stage1_battle_ids),
        "num_files": len(stage1_files),
        "files": stage1_files,
    }

    stage2_manifest = {
        "stage": "specialization",
        "description": "1300+ Elo battles for high-quality decision learning",
        "elo_range": "1300+",
        "num_battles": len(stage2_battle_ids),
        "num_files": len(stage2_files),
        "files": stage2_files,
    }

    with open(CURRICULUM_DIR / "stage1.json", "w") as f:
        json.dump(stage1_manifest, f, indent=2)
    logger.info(
        f"Stage 1 manifest: {len(stage1_files)} files "
        f"({len(stage1_battle_ids)} unique battles)"
    )

    with open(CURRICULUM_DIR / "stage2.json", "w") as f:
        json.dump(stage2_manifest, f, indent=2)
    logger.info(
        f"Stage 2 manifest: {len(stage2_files)} files "
        f"({len(stage2_battle_ids)} unique battles)"
    )

    # Save Elo mapping for future reference
    elo_map_for_processed = {}
    for npz_path in all_npz:
        bid = parse_battle_id_from_npz(npz_path.name)
        if bid and bid in battle_id_to_elo:
            elo_map_for_processed[npz_path.name] = battle_id_to_elo[bid]

    with open(CURRICULUM_DIR / "elo_mapping.json", "w") as f:
        json.dump(elo_map_for_processed, f)
    logger.info(f"Elo mapping saved for {len(elo_map_for_processed)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare curriculum training data")
    parser.add_argument(
        "--tar-path", type=str, default=None,
        help="Path to already-downloaded gen3ou.tar.gz (skip download)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/tmp/metamon_cache",
        help="Cache directory for HF downloads",
    )
    parser.add_argument(
        "--stage1-size", type=int, default=50000,
        help="Number of battles for Stage 1 (default: 50000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--skip-processing", action="store_true",
        help="Skip processing step (only download, extract, build manifests)",
    )
    args = parser.parse_args()

    # Step 1: Download or locate tar
    if args.tar_path:
        tar_path = args.tar_path
        logger.info(f"Using existing tar: {tar_path}")
    else:
        tar_path = download_tar(args.cache_dir)

    # Step 2: Scan tar for all battles by Elo
    bins, battle_id_to_elo = scan_tar(tar_path)

    # Step 3: Select battles for each stage
    stage1_members, stage2_members = select_battles_for_stages(
        bins, stage1_size=args.stage1_size, seed=args.seed
    )

    # Step 4: Find what's already processed
    already_processed = get_already_processed_battle_ids()

    # Step 5: Extract only new battles
    all_needed = list(set(stage1_members + stage2_members))
    logger.info(f"Total unique replays needed: {len(all_needed)}")

    extracted = extract_new_battles(
        tar_path, all_needed, already_processed, RAW_DIR
    )

    # Step 6: Process new raw files
    if not args.skip_processing and extracted:
        process_new_battles(RAW_DIR)
    elif not extracted:
        logger.info("No new battles to process — all already on disk!")

    # Step 7: Build curriculum manifests
    build_curriculum_manifests(stage1_members, stage2_members, battle_id_to_elo)

    logger.info("\n" + "=" * 60)
    logger.info("CURRICULUM DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Stage 1 manifest: {CURRICULUM_DIR / 'stage1.json'}")
    logger.info(f"Stage 2 manifest: {CURRICULUM_DIR / 'stage2.json'}")
    logger.info(f"Elo mapping: {CURRICULUM_DIR / 'elo_mapping.json'}")


if __name__ == "__main__":
    main()
