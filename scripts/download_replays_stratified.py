#!/usr/bin/env python3
"""Download Gen 3 OU replays with stratified Elo sampling.

Downloads ALL battles above a high-Elo threshold, then fills the remaining
quota with equal shares from lower Elo bins.

Strategy for 100K target:
  1. Take ALL battles >= 1500 Elo
  2. Remaining quota split equally across 5 bins:
     [1000-1100), [1100-1200), [1200-1300), [1300-1400), [1400-1500)

Usage:
    python scripts/download_replays_stratified.py
    python scripts/download_replays_stratified.py --target-size 100000
    python scripts/download_replays_stratified.py --tar-path /tmp/gen3ou.tar.gz
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def download_tar(
    generation: str = "gen3ou",
    cache_dir: str = "/tmp/metamon_cache",
) -> str:
    """Download the generation's replay archive from Hugging Face."""
    from huggingface_hub import hf_hub_download

    filename = f"{generation}.tar.gz"
    logger.info(f"Downloading {filename} from jakegrigsby/metamon-parsed-replays...")
    path = hf_hub_download(
        repo_id="jakegrigsby/metamon-parsed-replays",
        filename=filename,
        repo_type="dataset",
        revision="main",
        cache_dir=cache_dir,
    )
    size_gb = os.path.getsize(path) / 1e9
    logger.info(f"Downloaded to: {path} ({size_gb:.2f} GB)")
    return path


def parse_elo_from_filename(basename: str) -> int | None:
    """Extract Elo rating from a Metamon replay filename.

    Filename format: {id}_{ELO}_{player}_vs_{opponent}_{date}_{result}.json.lz4
    Returns None for unrated battles.
    """
    parts = basename.replace(".json.lz4", "").replace(".json", "").split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def scan_and_categorize(tar_path: str) -> dict[str, list[str]]:
    """Scan the archive and categorize members by Elo bin.

    Returns dict mapping bin labels to lists of member names.
    Bins: "1000-1100", "1100-1200", ..., "1400-1500", "1500+"
    """
    import tarfile

    logger.info("Scanning archive to categorize battles by Elo...")
    bins: dict[str, list[str]] = defaultdict(list)
    total_scanned = 0
    skipped_unrated = 0
    skipped_low = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if not (name.endswith(".json.lz4") or name.endswith(".json")):
                continue

            total_scanned += 1
            basename = os.path.basename(name)
            elo = parse_elo_from_filename(basename)

            if elo is None:
                skipped_unrated += 1
                continue
            if elo < 1000:
                skipped_low += 1
                continue

            if elo >= 1500:
                bins["1500+"].append(name)
            elif elo >= 1400:
                bins["1400-1500"].append(name)
            elif elo >= 1300:
                bins["1300-1400"].append(name)
            elif elo >= 1200:
                bins["1200-1300"].append(name)
            elif elo >= 1100:
                bins["1100-1200"].append(name)
            else:
                bins["1000-1100"].append(name)

            if total_scanned % 50000 == 0:
                logger.info(f"  Scanned {total_scanned} files...")

    logger.info(f"Scan complete: {total_scanned} total files")
    logger.info(f"  Skipped {skipped_unrated} unrated, {skipped_low} below 1000 Elo")
    for bin_label in ["1000-1100", "1100-1200", "1200-1300", "1300-1400", "1400-1500", "1500+"]:
        count = len(bins.get(bin_label, []))
        logger.info(f"  {bin_label}: {count} battles")

    return dict(bins)


def stratified_sample(
    bins: dict[str, list[str]],
    target_size: int,
    seed: int = 42,
) -> list[str]:
    """Select battles using stratified sampling.

    1. Take ALL battles >= 1500 Elo
    2. Divide remaining quota equally among 5 lower bins
    """
    rng = random.Random(seed)

    # Take all high-Elo battles
    high_elo = list(bins.get("1500+", []))
    logger.info(f"High Elo (>=1500): taking all {len(high_elo)} battles")

    remaining_quota = target_size - len(high_elo)
    if remaining_quota <= 0:
        logger.info(f"High-Elo battles alone exceed target ({len(high_elo)} >= {target_size})")
        return high_elo[:target_size] if len(high_elo) > target_size else high_elo

    # Divide remaining quota equally among 5 lower bins
    lower_bins = ["1000-1100", "1100-1200", "1200-1300", "1300-1400", "1400-1500"]
    per_bin_quota = remaining_quota // len(lower_bins)
    # Distribute remainder to lower bins first (they need more representation)
    extra = remaining_quota % len(lower_bins)

    sampled = list(high_elo)
    for i, bin_label in enumerate(lower_bins):
        available = bins.get(bin_label, [])
        quota = per_bin_quota + (1 if i < extra else 0)

        if len(available) <= quota:
            selected = available
            logger.info(
                f"  {bin_label}: taking all {len(available)} "
                f"(wanted {quota}, only {len(available)} available)"
            )
        else:
            selected = rng.sample(available, quota)
            logger.info(f"  {bin_label}: sampled {quota} from {len(available)} available")

        sampled.extend(selected)

    logger.info(f"Total selected: {len(sampled)} battles")
    return sampled


def extract_battles(
    tar_path: str,
    selected_members: list[str],
    output_dir: str,
) -> list[str]:
    """Extract selected battles from the archive."""
    import tarfile

    os.makedirs(output_dir, exist_ok=True)
    selected_set = set(selected_members)
    output_files: list[str] = []
    count = 0

    logger.info(f"Extracting {len(selected_set)} battles to {output_dir}...")

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if member.name not in selected_set:
                continue

            f = tf.extractfile(member)
            if f is None:
                continue

            raw_bytes = f.read()
            basename = os.path.basename(member.name)
            output_path = os.path.join(output_dir, basename)

            with open(output_path, "wb") as out:
                out.write(raw_bytes)

            output_files.append(basename)
            count += 1

            if count % 5000 == 0:
                logger.info(f"  Extracted {count}/{len(selected_set)} battles...")

    logger.info(f"Extracted {count} battles to {output_dir}")
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Gen 3 OU replays with stratified Elo sampling"
    )
    parser.add_argument(
        "--generation", type=str, default="gen3ou",
        help="Generation format (default: gen3ou)",
    )
    parser.add_argument(
        "--target-size", type=int, default=100000,
        help="Target number of battles to download (default: 100000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/tmp/metamon_cache",
        help="Cache directory for HF downloads (default: /tmp/metamon_cache)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--tar-path", type=str, default=None,
        help="Path to already-downloaded .tar.gz (skip download)",
    )
    args = parser.parse_args()

    # Download or use existing tar
    if args.tar_path:
        tar_path = args.tar_path
        logger.info(f"Using existing tar: {tar_path}")
    else:
        tar_path = download_tar(args.generation, args.cache_dir)

    # Scan and categorize
    bins = scan_and_categorize(tar_path)

    # Stratified sample
    selected = stratified_sample(bins, args.target_size, args.seed)

    # Extract
    output_files = extract_battles(tar_path, selected, args.output_dir)

    # Save sampling manifest
    manifest = {
        "generation": args.generation,
        "target_size": args.target_size,
        "actual_size": len(output_files),
        "seed": args.seed,
        "bin_counts": {k: len(v) for k, v in bins.items()},
        "sampled_counts": {},
    }
    # Count sampled per bin
    sampled_bins: dict[str, int] = defaultdict(int)
    for fname in output_files:
        elo = parse_elo_from_filename(fname)
        if elo is not None:
            if elo >= 1500:
                sampled_bins["1500+"] += 1
            elif elo >= 1400:
                sampled_bins["1400-1500"] += 1
            elif elo >= 1300:
                sampled_bins["1300-1400"] += 1
            elif elo >= 1200:
                sampled_bins["1200-1300"] += 1
            elif elo >= 1100:
                sampled_bins["1100-1200"] += 1
            else:
                sampled_bins["1000-1100"] += 1
    manifest["sampled_counts"] = dict(sampled_bins)

    manifest_path = os.path.join(args.output_dir, "sampling_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nDone! {len(output_files)} battles saved to {args.output_dir}")
    logger.info(f"Sampling manifest saved to {manifest_path}")
    logger.info("\nFinal distribution:")
    for bin_label in ["1000-1100", "1100-1200", "1200-1300", "1300-1400", "1400-1500", "1500+"]:
        logger.info(f"  {bin_label}: {sampled_bins.get(bin_label, 0)}")


if __name__ == "__main__":
    main()
