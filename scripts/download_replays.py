#!/usr/bin/env python3
"""Download and sample replays from the Metamon dataset.

Downloads replay archives from jakegrigsby/metamon-parsed-replays on
Hugging Face, then samples a specified number of battles and saves
them to data/raw/ as individual .json.lz4 files.

Supports multiple generations via the --generation flag (default: gen3ou).

Usage:
    python scripts/download_replays.py --sample-size 10000 --elo-threshold 1300
    python scripts/download_replays.py --generation gen9ou --elo-threshold 1500
    python scripts/download_replays.py --sample-size 10000 --output-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default Elo thresholds per generation.
# Gen 3 has a smaller ladder population, so we use a lower default.
DEFAULT_ELO_THRESHOLDS: dict[str, int] = {
    "gen3ou": 1300,
    "gen9ou": 1500,
}


def download_replays_tar(
    generation: str = "gen3ou",
    cache_dir: str = "/tmp/metamon_cache",
) -> str:
    """Download a generation's replay archive from Hugging Face.

    Args:
        generation: Format string, e.g. "gen3ou", "gen9ou".
        cache_dir: Local cache directory for HF downloads.

    Returns:
        Path to the downloaded .tar.gz file.
    """
    from huggingface_hub import hf_hub_download

    filename = f"{generation}.tar.gz"
    logger.info(
        f"Downloading {filename} from jakegrigsby/metamon-parsed-replays..."
    )
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


def sample_and_extract(
    tar_path: str,
    output_dir: str,
    sample_size: int = 10000,
    elo_threshold: int = 1300,
    seed: int = 42,
) -> list[str]:
    """Sample battles from tar and save to output directory.

    Uses reservoir sampling to efficiently sample from a large archive
    without loading everything into memory.

    Returns list of output filenames.
    """
    import tarfile

    try:
        import lz4.frame
    except ImportError:
        logger.error("lz4 package required. Install with: pip install lz4")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    # First pass: collect eligible member names
    logger.info(f"Scanning archive for eligible battles (Elo >= {elo_threshold})...")
    eligible_members: list[str] = []

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if not (name.endswith(".json.lz4") or name.endswith(".json")):
                continue

            # Quick Elo check from filename
            basename = os.path.basename(name)
            parts = basename.replace(".json.lz4", "").replace(".json", "").split("_")
            if len(parts) >= 2:
                try:
                    elo = int(parts[1])
                    if elo < elo_threshold:
                        continue
                except ValueError:
                    pass  # Can't parse Elo, include anyway

            eligible_members.append(name)

    logger.info(f"Found {len(eligible_members)} eligible battles")

    # Sample
    if len(eligible_members) <= sample_size:
        sampled = eligible_members
        logger.info(f"Taking all {len(sampled)} eligible battles (less than sample_size)")
    else:
        sampled = rng.sample(eligible_members, sample_size)
        logger.info(f"Sampled {len(sampled)} battles from {len(eligible_members)} eligible")

    sampled_set = set(sampled)

    # Second pass: extract sampled battles
    logger.info("Extracting sampled battles...")
    output_files: list[str] = []
    count = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if member.name not in sampled_set:
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

            if count % 1000 == 0:
                logger.info(f"  Extracted {count}/{len(sampled)} battles...")

    logger.info(f"Extracted {count} battles to {output_dir}")
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and sample replays from the Metamon dataset"
    )
    parser.add_argument(
        "--generation",
        type=str,
        default="gen3ou",
        help="Generation format to download, e.g. gen3ou, gen9ou (default: gen3ou)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of battles to sample (default: 10000)",
    )
    parser.add_argument(
        "--elo-threshold",
        type=int,
        default=None,
        help="Minimum Elo for both players (default: 1300 for gen3ou, 1500 for gen9ou)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for sampled battles (default: data/raw)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/metamon_cache",
        help="Cache directory for HF downloads (default: /tmp/metamon_cache)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--tar-path",
        type=str,
        default=None,
        help="Path to already-downloaded .tar.gz (skip download)",
    )
    args = parser.parse_args()

    # Resolve Elo threshold: use explicit value or generation-specific default
    elo_threshold = args.elo_threshold
    if elo_threshold is None:
        elo_threshold = DEFAULT_ELO_THRESHOLDS.get(args.generation, 1300)

    # Download or use existing tar
    if args.tar_path:
        tar_path = args.tar_path
        logger.info(f"Using existing tar: {tar_path}")
    else:
        tar_path = download_replays_tar(args.generation, args.cache_dir)

    # Sample and extract
    output_files = sample_and_extract(
        tar_path=tar_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        elo_threshold=elo_threshold,
        seed=args.seed,
    )

    logger.info(
        f"Done! {len(output_files)} {args.generation} battles saved to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
