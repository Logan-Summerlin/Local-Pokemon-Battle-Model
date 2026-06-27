#!/usr/bin/env python3
"""Phase 4: Train the BattleTransformer on processed replay battles.

Unrolls each battle into per-turn training examples with causal history
windows, so every valid decision point produces training signal.

Runs the full training pipeline with detailed resource monitoring:
- CPU/RAM usage tracked per epoch
- Wall-clock time per epoch and total
- Saves comprehensive training report as JSON

Usage:
    # Smoke test (validate pipeline)
    python scripts/train_phase4.py --mode smoke

    # Full training on all data
    python scripts/train_phase4.py --mode full

    # Custom
    python scripts/train_phase4.py --mode full --num-battles 100 --epochs 30
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import platform
import sys
from contextlib import nullcontext

if sys.platform == "win32":
    resource_mod = None
else:
    import resource as resource_mod
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# On small-VRAM GPUs (e.g. GTX 1650, 4 GB) memory fragmentation is the most
# common cause of spurious CUDA OOMs. expandable_segments lets the caching
# allocator grow/shrink segments instead of holding fixed reserved blocks,
# which materially lowers peak reserved memory. Set before torch is imported,
# and only if the user has not already exported a value.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.tensorizer import BattleVocabularies
from src.data.auxiliary_labels import (
    NUM_ITEM_CLASSES,
    NUM_MOVE_FAMILIES,
    classify_speed,
    classify_role_from_stats,
    classify_move_families,
)
from src.models.battle_transformer import (
    BattleTransformer,
    TransformerConfig,
    compute_total_loss,
    TOKENS_PER_STEP,
)
from src.environment.action_space import NUM_ACTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Resource monitoring ──────────────────────────────────────────────────


def get_system_info() -> dict:
    """Collect static system information."""
    info = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except Exception:
        info["cpu_model"] = platform.processor() or "unknown"

    info["cpu_count"] = os.cpu_count() or 0

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["ram_total_gb"] = round(kb / 1024 / 1024, 2)
                    break
    except Exception:
        info["ram_total_gb"] = 0.0

    info["gpu_name"] = "None"
    info["gpu_memory_gb"] = 0.0
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )

    info["python_version"] = sys.version.split()[0]
    info["pytorch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["os_info"] = f"{platform.system()} {platform.release()}"
    return info


def get_resource_snapshot() -> dict:
    """Get current resource usage."""
    snapshot = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    snapshot["ram_used_gb"] = round(int(line.split()[1]) / 1024 / 1024, 3)
                elif line.startswith("VmPeak:"):
                    snapshot["ram_peak_gb"] = round(int(line.split()[1]) / 1024 / 1024, 3)
    except Exception:
        snapshot["ram_used_gb"] = 0.0
        snapshot["ram_peak_gb"] = 0.0

    if resource_mod is not None:
        try:
            ru = resource_mod.getrusage(resource_mod.RUSAGE_SELF)
            snapshot["cpu_user_time_sec"] = round(ru.ru_utime, 2)
            snapshot["cpu_sys_time_sec"] = round(ru.ru_stime, 2)
        except Exception:
            pass

    if torch.cuda.is_available():
        snapshot["gpu_memory_used_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        snapshot["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
    else:
        snapshot["gpu_memory_used_gb"] = 0.0

    return snapshot


# ── Windowed turn dataset ────────────────────────────────────────────────


class WindowedTurnDataset(Dataset):
    """Dataset that unrolls battles into per-turn examples with causal history.

    Each example is a window of consecutive turns [t-W+1 .. t] from a battle,
    where the target is the action at turn t. This ensures every valid decision
    point contributes to training, not just the last turn per battle.

    For a battle with L turns and window size W, this produces L examples
    (with shorter windows for early turns).

    When shuffle_moves=True, each battle gets a random permutation of its 4
    move slots. This permutation is applied consistently across all turns in
    the battle, preventing the model from memorizing slot positions instead
    of learning move identities.
    """

    # Move slot indices in the pokemon feature vector (categorical features 1-4)
    _MOVE_FEAT_START = 1
    _MOVE_FEAT_END = 5  # exclusive
    _NUM_MOVE_SLOTS = 4

    def __init__(
        self,
        battles: list[dict[str, np.ndarray]],
        max_window: int = 20,
        shuffle_moves: bool = False,
    ) -> None:
        self.max_window = max_window
        self.shuffle_moves = shuffle_moves
        self.examples: list[tuple[int, int]] = []
        self.battles: list[dict[str, torch.Tensor]] = []
        self._move_perms: list[torch.Tensor | None] = []

        for b_idx, battle in enumerate(battles):
            tensor_battle = {
                "own_team": torch.from_numpy(battle["own_team"]).float(),
                "opponent_team": torch.from_numpy(battle["opponent_team"]).float(),
                "field": torch.from_numpy(battle["field"]).float(),
                "context": torch.from_numpy(battle["context"]).float(),
                "legal_mask": torch.from_numpy(battle["legal_mask"]).float(),
                "action": torch.from_numpy(battle["action"]).long(),
                "game_result": torch.from_numpy(battle["game_result"]).float(),
                "item_targets": torch.from_numpy(battle["item_targets"]).long(),
                "speed_targets": torch.from_numpy(battle["speed_targets"]).long(),
                "role_targets": torch.from_numpy(battle["role_targets"]).long(),
                "move_family_targets": torch.from_numpy(battle["move_family_targets"]).long(),
            }
            self.battles.append(tensor_battle)

            # Generate one random move permutation per battle
            if shuffle_moves:
                self._move_perms.append(torch.randperm(self._NUM_MOVE_SLOTS))
            else:
                self._move_perms.append(None)

            seq_len = int(battle.get("seq_len", battle["action"].shape[0]))
            for t in range(seq_len):
                action = int(tensor_battle["action"][t])
                if action >= 0:
                    self.examples.append((b_idx, t))

    def __len__(self) -> int:
        return len(self.examples)

    def _apply_move_shuffle(
        self,
        own_team: torch.Tensor,
        legal_mask: torch.Tensor,
        action: torch.Tensor,
        perm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply move slot permutation to a window of turns.

        Args:
            own_team: (seq, 6, 28) own team features
            legal_mask: (seq, 9) legal action mask
            action: scalar action label
            perm: (4,) permutation of [0,1,2,3]

        Returns:
            Shuffled (own_team, legal_mask, action)
        """
        own_team = own_team.clone()
        legal_mask = legal_mask.clone()
        action = action.clone()

        # Shuffle move IDs in the active pokemon (slot 0) across all turns
        # Feature indices 1-4 are the 4 move IDs
        s, e = self._MOVE_FEAT_START, self._MOVE_FEAT_END
        own_team[:, 0, s:e] = own_team[:, 0, s:e][:, perm]

        # Shuffle the move entries (indices 0-3) in the legal mask
        legal_mask[:, :self._NUM_MOVE_SLOTS] = legal_mask[:, :self._NUM_MOVE_SLOTS][:, perm]

        # Remap the action label if it's a move action (0-3)
        act_val = action.item()
        if 0 <= act_val < self._NUM_MOVE_SLOTS:
            # Find where the original slot ended up after permutation
            # perm[new_pos] = old_pos, so we need: new_pos where perm[new_pos] == act_val
            new_pos = (perm == act_val).nonzero(as_tuple=True)[0].item()
            action = torch.tensor(new_pos, dtype=action.dtype)

        return own_team, legal_mask, action

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        b_idx, t_idx = self.examples[idx]
        battle = self.battles[b_idx]

        start = max(0, t_idx - self.max_window + 1)
        end = t_idx + 1
        actual_len = end - start

        own_team = battle["own_team"][start:end]
        legal_mask = battle["legal_mask"][start:end]
        action = battle["action"][t_idx]

        perm = self._move_perms[b_idx]
        if perm is not None:
            own_team, legal_mask, action = self._apply_move_shuffle(
                own_team, legal_mask, action, perm,
            )

        return {
            "own_team": own_team,
            "opponent_team": battle["opponent_team"][start:end],
            "field": battle["field"][start:end],
            "context": battle["context"][start:end],
            "legal_mask": legal_mask,
            "action": action,
            "game_result": battle["game_result"][t_idx],
            "seq_len": torch.tensor(actual_len, dtype=torch.long),
            "item_targets": battle["item_targets"][t_idx],
            "speed_targets": battle["speed_targets"][t_idx],
            "role_targets": battle["role_targets"][t_idx],
            "move_family_targets": battle["move_family_targets"][t_idx],
        }


def collate_windowed(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate windowed turn examples with right-padding to max window in batch."""
    max_len = max(item["seq_len"].item() for item in batch)

    result: dict[str, torch.Tensor] = {}
    seq_lens = torch.tensor([item["seq_len"].item() for item in batch], dtype=torch.long)
    result["seq_len"] = seq_lens

    result["action"] = torch.stack([item["action"] for item in batch])
    result["game_result"] = torch.stack([item["game_result"] for item in batch])

    result["item_targets"] = torch.stack([item["item_targets"] for item in batch])
    result["speed_targets"] = torch.stack([item["speed_targets"] for item in batch])
    result["role_targets"] = torch.stack([item["role_targets"] for item in batch])
    result["move_family_targets"] = torch.stack([item["move_family_targets"] for item in batch])

    # Sequence tensors: right-pad to max_len (model masks padding via seq_len)
    for key in ["own_team", "opponent_team", "field", "context", "legal_mask"]:
        tensors = []
        for item in batch:
            t = item[key]
            cur_len = t.shape[0]
            if cur_len < max_len:
                pad_shape = list(t.shape)
                pad_shape[0] = max_len - cur_len
                padding = torch.zeros(pad_shape, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            tensors.append(t)
        result[key] = torch.stack(tensors)

    return result


# ── Learning rate scheduler ──────────────────────────────────────────────


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self):
        if self._step < self.warmup_steps:
            return self.base_lrs[0] * self._step / max(self.warmup_steps, 1)
        progress = (self._step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ── Training loop ────────────────────────────────────────────────────────


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_policy_loss: float = 0.0
    val_policy_loss: float = 0.0
    train_aux_loss: float = 0.0
    val_aux_loss: float = 0.0
    train_value_loss: float = 0.0
    val_value_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_top3_accuracy: float = 0.0
    val_top3_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch_time_sec: float = 0.0
    train_examples: int = 0
    val_examples: int = 0
    examples_per_sec: float = 0.0
    aux_item_accuracy: float = 0.0
    aux_speed_accuracy: float = 0.0
    aux_role_accuracy: float = 0.0
    ram_used_gb: float = 0.0
    ram_peak_gb: float = 0.0
    cpu_user_time_sec: float = 0.0


def _amp_context(amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def forward_step(
    model: BattleTransformer,
    batch: dict[str, torch.Tensor],
    config: TransformerConfig,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, dict[str, torch.Tensor] | None]:
    """Forward pass for windowed turn data.

    Returns: (loss, loss_dict, policy_logits, auxiliary_preds)
    """
    own_team = batch["own_team"]
    opp_team = batch["opponent_team"]
    field_feat = batch["field"]
    context = batch["context"]
    legal_mask_seq = batch["legal_mask"]
    action = batch["action"]
    game_result = batch["game_result"]
    seq_len = batch["seq_len"]

    output = model(
        own_team, opp_team, field_feat, context,
        legal_mask=legal_mask_seq,
        seq_len=seq_len,
        return_auxiliary=True,
        return_value=config.use_value_head,
    )

    logits = output.policy_logits

    last_idx = (seq_len - 1).clamp(min=0)
    legal_last = torch.gather(
        legal_mask_seq, dim=1,
        index=last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, NUM_ACTIONS),
    ).squeeze(1)

    aux_targets = {
        "item_targets": batch["item_targets"],
        "speed_targets": batch["speed_targets"],
        "role_targets": batch["role_targets"],
        "move_family_targets": batch["move_family_targets"],
    }

    loss, loss_dict = compute_total_loss(
        output, action, legal_last,
        aux_targets=aux_targets,
        game_result=game_result,
        config=config,
    )

    return loss, loss_dict, logits, output.auxiliary_preds


def train_epoch(
    model, loader, optimizer, scheduler, config, device,
    grad_accum=1, max_grad_norm=1.0,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
    non_blocking_transfer: bool = False,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    total_loss = total_policy = total_aux = total_value = 0.0
    total_correct = total_top3 = total_examples = n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=non_blocking_transfer) for k, v in batch.items()}
        with _amp_context(amp_dtype):
            loss, loss_dict, logits, _ = forward_step(model, batch, config)

        if scaler is not None:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()

        if (batch_idx + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss_dict.get("total", 0.0)
        total_policy += loss_dict.get("policy", 0.0)
        total_aux += loss_dict.get("auxiliary", 0.0)
        total_value += loss_dict.get("value", 0.0)
        n_batches += 1

        action = batch["action"]
        with torch.no_grad():
            valid = action >= 0
            if valid.any():
                preds = logits[valid].argmax(dim=-1)
                total_correct += (preds == action[valid]).sum().item()
                top3 = logits[valid].topk(min(3, NUM_ACTIONS), dim=-1).indices
                total_top3 += (top3 == action[valid].unsqueeze(-1)).any(dim=-1).sum().item()
                total_examples += valid.sum().item()

    n = max(n_batches, 1)
    ne = max(total_examples, 1)
    return {
        "loss": total_loss / n, "policy_loss": total_policy / n,
        "aux_loss": total_aux / n, "value_loss": total_value / n,
        "accuracy": total_correct / ne, "top3_accuracy": total_top3 / ne,
        "total_examples": total_examples,
    }


@torch.no_grad()
def validate(
    model,
    loader,
    config,
    device,
    amp_dtype: torch.dtype | None = None,
    non_blocking_transfer: bool = False,
) -> dict[str, float]:
    """Run validation."""
    model.eval()
    total_loss = total_policy = total_aux = total_value = 0.0
    total_correct = total_top3 = total_examples = n_batches = 0
    aux_correct = {"item": 0, "speed": 0, "role": 0}
    aux_total = {"item": 0, "speed": 0, "role": 0}

    for batch in loader:
        batch = {k: v.to(device, non_blocking=non_blocking_transfer) for k, v in batch.items()}
        with _amp_context(amp_dtype):
            loss, loss_dict, logits, aux_preds = forward_step(model, batch, config)

        total_loss += loss_dict.get("total", 0.0)
        total_policy += loss_dict.get("policy", 0.0)
        total_aux += loss_dict.get("auxiliary", 0.0)
        total_value += loss_dict.get("value", 0.0)
        n_batches += 1

        action = batch["action"]
        valid = action >= 0
        if valid.any():
            preds = logits[valid].argmax(dim=-1)
            total_correct += (preds == action[valid]).sum().item()
            top3 = logits[valid].topk(min(3, NUM_ACTIONS), dim=-1).indices
            total_top3 += (top3 == action[valid].unsqueeze(-1)).any(dim=-1).sum().item()
            total_examples += valid.sum().item()

        # Auxiliary accuracy
        if aux_preds is not None:
            for head, tgt_key, short in [
                ("item_logits", "item_targets", "item"),
                ("speed_logits", "speed_targets", "speed"),
                ("role_logits", "role_targets", "role"),
            ]:
                if head in aux_preds and tgt_key in batch:
                    pred = aux_preds[head]
                    target = batch[tgt_key]
                    flat_pred = pred.reshape(-1, pred.shape[-1])
                    flat_target = target.reshape(-1)
                    valid_aux = flat_target >= 0
                    if valid_aux.any():
                        aux_pred_classes = flat_pred[valid_aux].argmax(dim=-1)
                        aux_correct[short] += (aux_pred_classes == flat_target[valid_aux]).sum().item()
                        aux_total[short] += valid_aux.sum().item()

    n = max(n_batches, 1)
    ne = max(total_examples, 1)
    result = {
        "loss": total_loss / n, "policy_loss": total_policy / n,
        "aux_loss": total_aux / n, "value_loss": total_value / n,
        "accuracy": total_correct / ne, "top3_accuracy": total_top3 / ne,
        "total_examples": total_examples,
    }
    for name in ["item", "speed", "role"]:
        result[f"aux_{name}_accuracy"] = aux_correct[name] / aux_total[name] if aux_total[name] > 0 else 0.0
    return result


# ── Data loading ─────────────────────────────────────────────────────────


def load_all_battles(data_dir, max_battles=None, battle_manifest=None):
    """Load all valid battles from processed data directory.

    Args:
        data_dir: Path to processed data directory (contains battles/ and vocabs/).
        max_battles: Maximum number of battles to load (ignored if manifest provided).
        battle_manifest: Path to a JSON manifest file listing specific battle files to load.
            If provided, only battles listed in the manifest are loaded, and max_battles
            is ignored. The manifest should have a "files" key with a list of npz filenames.
    """
    battles_dir = Path(data_dir) / "battles"
    vocabs = BattleVocabularies.load(Path(data_dir) / "vocabs")

    if battle_manifest is not None:
        import json as _json
        with open(battle_manifest) as f:
            manifest = _json.load(f)
        manifest_files = manifest["files"]
        npz_files = [battles_dir / fname for fname in manifest_files]
        logger.info(
            f"Loading from manifest: {battle_manifest} "
            f"({len(npz_files)} files, stage={manifest.get('stage', '?')})"
        )
    else:
        npz_files = sorted(battles_dir.glob("*.npz"))
        if max_battles is not None:
            npz_files = npz_files[:max_battles]

    sequences, skipped = [], 0
    for npz_file in npz_files:
        try:
            data = dict(np.load(str(npz_file)))
            if "own_team" in data and "action" in data:
                sequences.append(data)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    logger.info(f"Loaded {len(sequences)} battles ({skipped} skipped)")
    return sequences, vocabs


# Per-pokemon feature indices in the 28-dim tensor vector (see tensorizer.py):
# categorical species(0) moves(1-4) item(5) ...; base stats are the 6 continuous
# values at 17-22 (hp, atk, def, spa, spd, spe), each normalized by /255.
_AUX_FEAT_HP, _AUX_FEAT_ATK, _AUX_FEAT_DEF = 17, 18, 19
_AUX_FEAT_SPA, _AUX_FEAT_SPD, _AUX_FEAT_SPE = 20, 21, 22


def add_auxiliary_labels(sequences, vocabs=None):
    """Add auxiliary hidden-info targets from tensorized opponent features.

    Targets (one per opponent slot, ``-1`` = unknown/masked):
      - item_targets:        opponent item vocab index hashed into item classes.
      - speed_targets:       speed bucket from revealed base speed (idx 22 * 255).
      - role_targets:        role archetype from revealed base stats + move names.
      - move_family_targets: multi-hot move families from revealed move identities.

    Speed/role/move-family require the moves vocab (to decode move identity
    indices back to names); pass ``vocabs`` to enable them. All targets respect
    the hidden-information doctrine: only non-empty opponent slots get labels, and
    attributes that were never revealed stay ``-1``.
    """
    moves_vocab = getattr(vocabs, "moves", None)

    augmented = []
    for seq in sequences:
        new_seq = dict(seq)
        opp_team = seq["opponent_team"]
        is_seq = opp_team.ndim == 3
        if not is_seq:
            opp_team = opp_team[np.newaxis]
        seq_len, n_slots = opp_team.shape[0], opp_team.shape[1]

        item_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        speed_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        role_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        move_family_targets = np.full(
            (seq_len, n_slots, NUM_MOVE_FAMILIES), -1, dtype=np.int64
        )

        for t in range(seq_len):
            for s in range(n_slots):
                feat = opp_team[t, s]
                if int(feat[0]) == 0:
                    continue  # empty/padding slot

                item_idx = int(feat[5])
                if item_idx > 1:
                    item_targets[t, s] = min(item_idx % NUM_ITEM_CLASSES, NUM_ITEM_CLASSES - 1)

                # De-normalize base stats (stored as stat / 255.0).
                base_hp = int(round(float(feat[_AUX_FEAT_HP]) * 255))
                base_atk = int(round(float(feat[_AUX_FEAT_ATK]) * 255))
                base_def = int(round(float(feat[_AUX_FEAT_DEF]) * 255))
                base_spa = int(round(float(feat[_AUX_FEAT_SPA]) * 255))
                base_spd = int(round(float(feat[_AUX_FEAT_SPD]) * 255))
                base_spe = int(round(float(feat[_AUX_FEAT_SPE]) * 255))

                if base_spe > 0:
                    speed_targets[t, s] = classify_speed(base_spe)

                # Decode revealed move identities (skip PAD=0 / UNK=1).
                move_names: list[str] = []
                if moves_vocab is not None:
                    for mi in range(1, 5):
                        midx = int(feat[mi])
                        if midx > 1:
                            move_names.append(moves_vocab.decode(midx))

                if base_atk > 0 or base_spa > 0 or move_names:
                    role_targets[t, s] = classify_role_from_stats(
                        base_atk, base_spa, base_def, base_spd, base_hp, move_names,
                    )

                if move_names:
                    move_family_targets[t, s] = np.array(
                        classify_move_families(move_names), dtype=np.int64
                    )

        if not is_seq:
            item_targets = item_targets[0]
            speed_targets = speed_targets[0]
            role_targets = role_targets[0]
            move_family_targets = move_family_targets[0]

        new_seq["item_targets"] = item_targets
        new_seq["speed_targets"] = speed_targets
        new_seq["role_targets"] = role_targets
        new_seq["move_family_targets"] = move_family_targets
        augmented.append(new_seq)
    return augmented


def split_data(sequences, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data into train/val/test by battle (no leakage)."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    n = len(indices)
    n_val = int(n * val_ratio)
    n_test = n - int(n * train_ratio) - n_val
    n_train = n - n_val - n_test
    return (
        [sequences[i] for i in indices[:n_train]],
        [sequences[i] for i in indices[n_train:n_train + n_val]],
        [sequences[i] for i in indices[n_train + n_val:]],
    )


# ── Test set evaluation ──────────────────────────────────────────────────


@torch.no_grad()
def evaluate_on_test(
    model, test_data, config, device, batch_size=32, max_window=20,
    amp_dtype: torch.dtype | None = None,
    loader_kwargs: dict | None = None,
    non_blocking_transfer: bool = False,
):
    """Full evaluation on test set with per-turn windowed examples."""
    model.eval()
    dataset = WindowedTurnDataset(test_data, max_window=max_window)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windowed,
        **(loader_kwargs or {}),
    )

    total_loss = total_policy = total_aux = total_value = 0.0
    total_correct = total_top3 = total_examples = n_batches = 0
    action_type_correct = np.zeros(NUM_ACTIONS)
    action_type_total = np.zeros(NUM_ACTIONS)
    all_probs, all_correct_list = [], []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=non_blocking_transfer) for k, v in batch.items()}
        with _amp_context(amp_dtype):
            loss, loss_dict, logits, _ = forward_step(model, batch, config)

        total_loss += loss_dict.get("total", 0.0)
        total_policy += loss_dict.get("policy", 0.0)
        total_aux += loss_dict.get("auxiliary", 0.0)
        total_value += loss_dict.get("value", 0.0)
        n_batches += 1

        action = batch["action"]
        valid = action >= 0
        if valid.any():
            preds = logits[valid].argmax(dim=-1)
            targets = action[valid]
            total_correct += (preds == targets).sum().item()
            top3 = logits[valid].topk(min(3, NUM_ACTIONS), dim=-1).indices
            total_top3 += (top3 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
            total_examples += valid.sum().item()

            for a in range(NUM_ACTIONS):
                mask_a = targets == a
                if mask_a.any():
                    action_type_total[a] += mask_a.sum().item()
                    action_type_correct[a] += ((preds == targets) & mask_a).sum().item()

            probs = F.softmax(logits[valid], dim=-1)
            max_probs = probs.max(dim=-1).values
            is_correct = (preds == targets).float()
            all_probs.extend(max_probs.float().cpu().tolist())
            all_correct_list.extend(is_correct.float().cpu().tolist())

    n = max(n_batches, 1)
    ne = max(total_examples, 1)

    action_names = [
        "move1", "move2", "move3", "move4",
        "switch2", "switch3", "switch4", "switch5", "switch6",
    ]
    per_action_acc = {}
    for i in range(NUM_ACTIONS):
        if action_type_total[i] > 0:
            per_action_acc[action_names[i]] = {
                "accuracy": round(action_type_correct[i] / action_type_total[i], 4),
                "count": int(action_type_total[i]),
            }

    calibration = {}
    if all_probs:
        probs_arr = np.array(all_probs)
        correct_arr = np.array(all_correct_list)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins) - 1):
            mask = (probs_arr >= bins[i]) & (probs_arr < bins[i + 1])
            if mask.any():
                calibration[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] = {
                    "mean_confidence": round(float(probs_arr[mask].mean()), 4),
                    "mean_accuracy": round(float(correct_arr[mask].mean()), 4),
                    "count": int(mask.sum()),
                }

    return {
        "test_loss": round(total_loss / n, 4),
        "test_policy_loss": round(total_policy / n, 4),
        "test_aux_loss": round(total_aux / n, 4),
        "test_value_loss": round(total_value / n, 4),
        "test_accuracy": round(total_correct / ne, 4),
        "test_top3_accuracy": round(total_top3 / ne, 4),
        "test_examples": total_examples,
        "per_action_accuracy": per_action_acc,
        "calibration": calibration,
    }


# ── Checkpointing ───────────────────────────────────────────────────────


def save_checkpoint(model, optimizer, config, epoch, val_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_class": "BattleTransformer",
        "config": {
            "num_layers": config.num_layers,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "ffn_multiplier": config.ffn_multiplier,
            "species_embedding_dim": config.species_embedding_dim,
            "move_embedding_dim": config.move_embedding_dim,
            "item_embedding_dim": config.item_embedding_dim,
            "ability_embedding_dim": config.ability_embedding_dim,
            "type_embedding_dim": config.type_embedding_dim,
            "species_vocab_size": config.species_vocab_size,
            "moves_vocab_size": config.moves_vocab_size,
            "items_vocab_size": config.items_vocab_size,
            "abilities_vocab_size": config.abilities_vocab_size,
            "types_vocab_size": config.types_vocab_size,
            "status_vocab_size": config.status_vocab_size,
            "weather_vocab_size": config.weather_vocab_size,
            "terrain_vocab_size": config.terrain_vocab_size,
            "auxiliary_loss_weight": config.auxiliary_loss_weight,
            "use_value_head": config.use_value_head,
            "value_loss_weight": config.value_loss_weight,
            # A1 (fix 002): persist policy-head, loss-tuning, and sequence config
            # so checkpoints can be reconstructed outside their training run.
            "use_candidate_head": config.use_candidate_head,
            "use_split_head": config.use_split_head,
            "move_identity_candidates": config.move_identity_candidates,
            "policy_head_layers": config.policy_head_layers,
            "action_self_attention": config.action_self_attention,
            "switch_weight": config.switch_weight,
            "label_smoothing": config.label_smoothing,
            "max_seq_len": config.max_seq_len,
        },
    }
    checkpoint_dir = Path(checkpoint_dir)
    if is_best:
        torch.save(state, checkpoint_dir / "best_model.pt")
        logger.info(f"  Saved best checkpoint (val_loss={val_loss:.4f})")

    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, path)

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > 3:
        for old in checkpoints[:-3]:
            old.unlink()


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 BattleTransformer Training")
    parser.add_argument("--mode", choices=["smoke", "small", "full", "local"], default="full")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--num-battles", type=int, default=None)
    parser.add_argument("--battle-manifest", type=str, default=None,
                        help="Path to curriculum manifest JSON (overrides --num-battles)")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--species-embedding-dim", type=int, default=64)
    parser.add_argument("--move-embedding-dim", type=int, default=32)
    parser.add_argument("--item-embedding-dim", type=int, default=32)
    parser.add_argument("--ability-embedding-dim", type=int, default=32)
    parser.add_argument("--type-embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps (default: filled by --mode; 1 for smoke/small/full, 16 for local).")
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--no-value-head", action="store_true")
    parser.add_argument("--candidate-head", action="store_true",
                       help="Use candidate-conditioned policy head instead of pooled MLP")
    parser.add_argument("--split-head", action="store_true",
                       help="Use separate move/switch scoring pathways")
    parser.add_argument("--move-identity", action="store_true",
                       help="Use actual move ID embeddings instead of slot embeddings for move candidates")
    parser.add_argument("--shuffle-moves", action="store_true",
                       help="Randomly shuffle move slot order per battle to prevent slot memorization")
    parser.add_argument("--policy-head-layers", type=int, default=1,
                       help="Number of cross-attention layers in split policy head (default: 1)")
    parser.add_argument("--action-self-attention", action="store_true",
                       help="Enable self-attention over action candidates before scoring")
    parser.add_argument("--cosine-epochs", type=int, default=None,
                       help="Epoch count for cosine LR schedule (default: same as --epochs). Set higher to slow LR decay.")
    parser.add_argument("--switch-weight", type=float, default=1.0,
                       help="Upweight switch actions in policy loss (1.0 = uniform, 2.0 = 2x switch weight)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                       help="Label smoothing for policy loss (0.0 = none)")
    parser.add_argument("--max-window", type=int, default=None,
                        help="History window length (default: filled by --mode; 20 for smoke/small/full, 5 for local).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from (loads model weights only, resets optimizer/scheduler)")
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (default: auto, prefers >0 on CUDA).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches prefetched per worker when num_workers > 0.",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs (default: enabled when num_workers > 0).",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    parser.set_defaults(persistent_workers=None)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned host memory for faster host->GPU transfer (default on CUDA).",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned host memory in DataLoader.",
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument(
        "--non-blocking-transfer",
        dest="non_blocking_transfer",
        action="store_true",
        help="Use non_blocking=True for tensor transfers to GPU (default when pin_memory is enabled).",
    )
    parser.add_argument(
        "--blocking-transfer",
        dest="non_blocking_transfer",
        action="store_false",
        help="Use blocking host->device transfers.",
    )
    parser.set_defaults(non_blocking_transfer=None)
    parser.add_argument(
        "--amp",
        choices=["off", "fp16", "bf16", "auto"],
        default=None,
        help="Mixed precision mode on CUDA (default: filled by --mode; 'auto' normally, "
             "'fp16' for local. auto picks bf16 if supported, else fp16. On Turing GPUs "
             "(GTX 1650/1660) there is no bf16 hardware, so fp16 is both faster and lighter.",
    )
    parser.add_argument(
        "--prune-dead-features",
        action="store_true",
        help="Drop known dead input channels (e.g., field binary block for Gen 4+ features) in embeddings.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile(model) for potentially better steady-state throughput.",
    )
    args = parser.parse_args()

    # Apply mode presets
    if args.mode == "smoke":
        defaults = dict(
            num_battles=50, hidden_dim=128, num_layers=2, num_heads=4,
            batch_size=16, epochs=3, warmup_steps=20,
            grad_accum=1, max_window=20, amp="auto",
            checkpoint_dir="checkpoints/phase4_smoke",
            report_path="checkpoints/phase4_smoke/training_report.json",
        )
    elif args.mode == "small":
        defaults = dict(
            num_battles=500, hidden_dim=256, num_layers=4, num_heads=4,
            batch_size=32, epochs=15, warmup_steps=100,
            grad_accum=1, max_window=20, amp="auto",
            checkpoint_dir="checkpoints/phase4_small",
            report_path="checkpoints/phase4_small/training_report.json",
        )
    elif args.mode == "local":
        # Tuned for a single NVIDIA GTX 1650 (4 GB, Turing) + ~15 GB system RAM.
        #   * 5L/256d champion-class encoder (~4.6M params), history window 5.
        #   * batch_size 256 x grad_accum 4 == effective batch 1024 (matches the A40
        #     champion's optimizer batch). The model is small, so activations dominate
        #     VRAM (~2.5-3.8 MB/example at window 5); batch 256 is ~1.5 GB, leaving room
        #     under 4 GB. If a display is attached and VRAM is tight, drop to
        #     batch_size 128 / grad_accum 8; headless, batch 512 / grad_accum 2 is fine.
        #   * amp=fp16: Turing has 2x-rate packed FP16 on its CUDA cores but NO bf16
        #     hardware, so fp16 is the fast/light precision here.
        #   * 25K battles keeps the in-RAM dataset ~0.6 GB and the run inside ~8 h.
        # Architecture *flags* (--split-head --move-identity --shuffle-moves
        # --no-value-head) are intentionally NOT forced here — pass them explicitly,
        # or use the `gtx1650` profile in Autoresearch/run_experiment.py / the local
        # curriculum script for the full champion recipe.
        defaults = dict(
            num_battles=25_000, hidden_dim=256, num_layers=5, num_heads=4,
            batch_size=256, epochs=12, warmup_steps=200,
            grad_accum=4, max_window=5, amp="fp16",
            checkpoint_dir="checkpoints/phase4_local",
            report_path="checkpoints/phase4_local/training_report.json",
        )
    else:  # full
        defaults = dict(
            num_battles=None, hidden_dim=384, num_layers=6, num_heads=6,
            batch_size=32, epochs=30, warmup_steps=300,
            grad_accum=1, max_window=20, amp="auto",
            checkpoint_dir="checkpoints/phase4_full",
            report_path="checkpoints/phase4_full/training_report.json",
        )

    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("PHASE 4: BattleTransformer Training (Windowed Per-Turn)")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}, Device: {device}")

    sys_info = get_system_info()
    logger.info(f"CPU: {sys_info['cpu_model']} ({sys_info['cpu_count']} cores)")
    logger.info(f"RAM: {sys_info['ram_total_gb']} GB")
    logger.info(f"GPU: {sys_info['gpu_name']}")
    logger.info(f"PyTorch: {sys_info['pytorch_version']}")
    logger.info(f"PyTorch CUDA build: {torch.version.cuda or 'None (CPU-only build)'}")

    if device == "cpu":
        logger.warning(
            "CUDA is not available. Training will run on CPU. "
            "On Windows + NVIDIA GPU, install a CUDA-enabled PyTorch build."
        )
        logger.warning(
            "Example: pip uninstall -y torch torchvision torchaudio && "
            "pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1"
        )

    # Avoid noisy warning on CUDA builds without flash-attention kernels.
    if device == "cuda" and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        flash_available = False
        if hasattr(torch.backends.cuda, "is_flash_attention_available"):
            flash_available = torch.backends.cuda.is_flash_attention_available()
        if not flash_available:
            torch.backends.cuda.enable_flash_sdp(False)
            logger.info("Disabled flash SDP backend (not available in this PyTorch build).")

    amp_dtype: torch.dtype | None = None
    if device == "cuda" and args.amp != "off":
        if args.amp == "fp16":
            amp_dtype = torch.float16
        elif args.amp == "bf16":
            amp_dtype = torch.bfloat16
        else:  # auto
            bf16_ok = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    amp_name = "off" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")
    logger.info(f"AMP mode: {amp_name}")

    cpu_count = os.cpu_count() or 1
    if args.num_workers is None:
        args.num_workers = min(8, max(1, cpu_count // 2)) if device == "cuda" else 0
    if args.pin_memory is None:
        args.pin_memory = device == "cuda"
    if args.persistent_workers is None:
        args.persistent_workers = args.num_workers > 0
    if args.non_blocking_transfer is None:
        args.non_blocking_transfer = device == "cuda" and args.pin_memory

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    logger.info(
        "DataLoader config: num_workers=%d pin_memory=%s prefetch_factor=%s persistent_workers=%s non_blocking_transfer=%s",
        args.num_workers,
        args.pin_memory,
        loader_kwargs.get("prefetch_factor", "n/a"),
        loader_kwargs.get("persistent_workers", "n/a"),
        args.non_blocking_transfer,
    )

    # Load data
    data_dir = Path(args.data_dir)
    logger.info(f"Loading data from {data_dir}...")
    load_start = time.time()
    sequences, vocabs = load_all_battles(
        data_dir, max_battles=args.num_battles,
        battle_manifest=args.battle_manifest,
    )
    load_time = time.time() - load_start
    snap = get_resource_snapshot()
    logger.info(f"Data loaded in {load_time:.1f}s, RAM: {snap['ram_used_gb']:.2f} GB")

    # Split
    train_seqs, val_seqs, test_seqs = split_data(sequences, seed=args.seed)
    logger.info(f"Split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")
    del sequences
    gc.collect()

    # Auxiliary labels
    train_seqs = add_auxiliary_labels(train_seqs, vocabs)
    val_seqs = add_auxiliary_labels(val_seqs, vocabs)
    test_seqs = add_auxiliary_labels(test_seqs, vocabs)

    # Create windowed datasets (shuffle moves only for training, not val)
    train_dataset = WindowedTurnDataset(train_seqs, max_window=args.max_window, shuffle_moves=args.shuffle_moves)
    val_dataset = WindowedTurnDataset(val_seqs, max_window=args.max_window)
    logger.info(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

    data_stats = {
        "total_battles": len(train_seqs) + len(val_seqs) + len(test_seqs),
        "train_battles": len(train_seqs), "val_battles": len(val_seqs), "test_battles": len(test_seqs),
        "train_examples": len(train_dataset), "val_examples": len(val_dataset),
        "max_window": args.max_window,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_windowed,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_windowed,
        **loader_kwargs,
    )

    # Create model
    config = TransformerConfig.from_vocabs(
        vocabs, num_layers=args.num_layers, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, dropout=args.dropout,
        ffn_multiplier=args.ffn_multiplier,
        species_embedding_dim=args.species_embedding_dim,
        move_embedding_dim=args.move_embedding_dim,
        item_embedding_dim=args.item_embedding_dim,
        ability_embedding_dim=args.ability_embedding_dim,
        type_embedding_dim=args.type_embedding_dim,
        auxiliary_loss_weight=args.aux_weight,
        use_value_head=not args.no_value_head,
        value_loss_weight=args.value_weight,
        prune_dead_features=args.prune_dead_features,
        switch_weight=args.switch_weight,
        label_smoothing=args.label_smoothing,
        use_candidate_head=args.candidate_head,
        use_split_head=args.split_head,
        move_identity_candidates=args.move_identity,
        policy_head_layers=args.policy_head_layers,
        action_self_attention=args.action_self_attention,
    )
    model = BattleTransformer(config).to(device)

    # Resume from checkpoint (model weights only — optimizer/scheduler reset for new stage)
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "best_model.pt"
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed model weights from {resume_path} (epoch {ckpt.get('epoch', '?')})")

    param_count = model.count_parameters()
    if args.torch_compile:
        model = torch.compile(model)
        logger.info("Enabled torch.compile for model")
    logger.info(f"Model: {config.num_layers}L/{config.hidden_dim}d/{config.num_heads}H, "
                f"FFN x{config.ffn_multiplier}, {param_count:,} params")

    training_config = {
        "mode": args.mode, "num_layers": config.num_layers,
        "hidden_dim": config.hidden_dim, "num_heads": config.num_heads,
        "ffn_multiplier": config.ffn_multiplier,
        "dropout": config.dropout, "parameter_count": param_count,
        "species_embedding_dim": config.species_embedding_dim,
        "move_embedding_dim": config.move_embedding_dim,
        "item_embedding_dim": config.item_embedding_dim,
        "ability_embedding_dim": config.ability_embedding_dim,
        "type_embedding_dim": config.type_embedding_dim,
        "batch_size": args.batch_size, "max_epochs": args.epochs,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps, "patience": args.patience,
        "grad_accumulation": args.grad_accum,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": loader_kwargs.get("prefetch_factor"),
        "persistent_workers": loader_kwargs.get("persistent_workers"),
        "non_blocking_transfer": args.non_blocking_transfer,
        "aux_loss_weight": args.aux_weight, "value_loss_weight": args.value_weight,
        "switch_weight": args.switch_weight, "label_smoothing": args.label_smoothing,
        "use_value_head": not args.no_value_head,
        "max_window": args.max_window, "device": device, "seed": args.seed,
        "amp": amp_name,
        "prune_dead_features": args.prune_dead_features,
        "torch_compile": args.torch_compile,
        "candidate_head": args.candidate_head,
        "split_head": args.split_head,
        "move_identity": args.move_identity,
        "shuffle_moves": args.shuffle_moves,
        "cosine_epochs": args.cosine_epochs if args.cosine_epochs is not None else args.epochs,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.999))
    cosine_epochs = args.cosine_epochs if args.cosine_epochs is not None else args.epochs
    total_steps = cosine_epochs * len(train_loader) // args.grad_accum
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=args.warmup_steps,
                                       total_steps=total_steps)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("=" * 70)
    logger.info(f"STARTING TRAINING ({len(train_loader)} batches/epoch)")
    logger.info("=" * 70)

    train_start = time.time()
    best_val_loss = float("inf")
    patience_counter = 0
    epoch_metrics_list = []
    epoch_resources_list = []
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            config,
            device,
            grad_accum=args.grad_accum,
            amp_dtype=amp_dtype,
            scaler=scaler,
            non_blocking_transfer=args.non_blocking_transfer,
        )
        val_metrics = validate(
            model,
            val_loader,
            config,
            device,
            amp_dtype=amp_dtype,
            non_blocking_transfer=args.non_blocking_transfer,
        )
        epoch_time = time.time() - epoch_start
        snap = get_resource_snapshot()
        lr = scheduler.get_lr()

        em = EpochMetrics(
            epoch=epoch, train_loss=train_metrics["loss"], val_loss=val_metrics["loss"],
            train_policy_loss=train_metrics["policy_loss"], val_policy_loss=val_metrics["policy_loss"],
            train_aux_loss=train_metrics["aux_loss"], val_aux_loss=val_metrics["aux_loss"],
            train_value_loss=train_metrics["value_loss"], val_value_loss=val_metrics["value_loss"],
            train_accuracy=train_metrics["accuracy"], val_accuracy=val_metrics["accuracy"],
            train_top3_accuracy=train_metrics["top3_accuracy"], val_top3_accuracy=val_metrics["top3_accuracy"],
            learning_rate=lr, epoch_time_sec=epoch_time,
            train_examples=train_metrics["total_examples"], val_examples=val_metrics["total_examples"],
            examples_per_sec=train_metrics["total_examples"] / max(epoch_time, 0.001),
            aux_item_accuracy=val_metrics.get("aux_item_accuracy", 0.0),
            aux_speed_accuracy=val_metrics.get("aux_speed_accuracy", 0.0),
            aux_role_accuracy=val_metrics.get("aux_role_accuracy", 0.0),
            ram_used_gb=snap.get("ram_used_gb", 0.0),
            ram_peak_gb=snap.get("ram_peak_gb", 0.0),
            cpu_user_time_sec=snap.get("cpu_user_time_sec", 0.0),
        )

        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"loss={em.train_loss:.4f} val_loss={em.val_loss:.4f} "
            f"acc={em.train_accuracy:.3f} val_acc={em.val_accuracy:.3f} "
            f"top3={em.val_top3_accuracy:.3f} "
            f"lr={lr:.2e} time={epoch_time:.1f}s "
            f"RAM={em.ram_used_gb:.2f}GB"
        )
        if em.aux_item_accuracy > 0:
            logger.info(f"  Aux: item={em.aux_item_accuracy:.3f} speed={em.aux_speed_accuracy:.3f} role={em.aux_role_accuracy:.3f}")

        epoch_metrics_list.append(em)
        epoch_resources_list.append({
            "epoch": epoch, "wall_time_sec": round(epoch_time, 2),
            "examples_per_sec": round(em.examples_per_sec, 1),
            "ram_used_gb": em.ram_used_gb, "ram_peak_gb": em.ram_peak_gb,
            "cpu_user_time_sec": em.cpu_user_time_sec,
        })

        save_checkpoint(model, optimizer, config, epoch, em.val_loss, checkpoint_dir)
        if em.val_loss < best_val_loss:
            best_val_loss = em.val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, config, epoch, em.val_loss, checkpoint_dir, is_best=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    total_train_time = time.time() - train_start
    best_epoch = min(epoch_metrics_list, key=lambda m: m.val_loss).epoch
    logger.info("=" * 70)
    logger.info(f"TRAINING COMPLETE in {total_train_time:.1f}s ({total_train_time/60:.1f} min)")
    logger.info(f"Best epoch: {best_epoch}, Best val_loss: {best_val_loss:.4f}")

    # Load best model
    best_path = checkpoint_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model from epoch {ckpt['epoch']}")

    # Test evaluation
    logger.info("=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)
    eval_start = time.time()
    test_results = evaluate_on_test(
        model,
        test_seqs,
        config,
        device,
        batch_size=args.batch_size,
        max_window=args.max_window,
        amp_dtype=amp_dtype,
        loader_kwargs=loader_kwargs,
        non_blocking_transfer=args.non_blocking_transfer,
    )
    eval_time = time.time() - eval_start

    logger.info(f"Test accuracy: {test_results['test_accuracy']:.4f}")
    logger.info(f"Test top-3 accuracy: {test_results['test_top3_accuracy']:.4f}")
    logger.info(f"Test loss: {test_results['test_loss']:.4f}")
    logger.info(f"Test examples: {test_results['test_examples']}")

    if test_results.get("per_action_accuracy"):
        logger.info("Per-action accuracy:")
        for name, stats in test_results["per_action_accuracy"].items():
            logger.info(f"  {name}: {stats['accuracy']:.3f} (n={stats['count']})")

    if test_results.get("calibration"):
        logger.info("Calibration:")
        for bin_name, stats in test_results["calibration"].items():
            logger.info(f"  [{bin_name}]: conf={stats['mean_confidence']:.3f} acc={stats['mean_accuracy']:.3f} (n={stats['count']})")

    # Build report
    final_snap = get_resource_snapshot()
    total_wall_time = time.time() - train_start

    report = {
        "phase": "Phase 4 - BattleTransformer Training",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "system_info": sys_info,
        "training_config": training_config,
        "data_stats": data_stats,
        "epoch_metrics": [
            {
                "epoch": em.epoch,
                "train_loss": round(em.train_loss, 4), "val_loss": round(em.val_loss, 4),
                "train_policy_loss": round(em.train_policy_loss, 4), "val_policy_loss": round(em.val_policy_loss, 4),
                "train_aux_loss": round(em.train_aux_loss, 4), "val_aux_loss": round(em.val_aux_loss, 4),
                "train_value_loss": round(em.train_value_loss, 4), "val_value_loss": round(em.val_value_loss, 4),
                "train_accuracy": round(em.train_accuracy, 4), "val_accuracy": round(em.val_accuracy, 4),
                "train_top3_accuracy": round(em.train_top3_accuracy, 4), "val_top3_accuracy": round(em.val_top3_accuracy, 4),
                "learning_rate": em.learning_rate, "epoch_time_sec": round(em.epoch_time_sec, 2),
                "examples_per_sec": round(em.examples_per_sec, 1),
                "aux_item_accuracy": round(em.aux_item_accuracy, 4),
                "aux_speed_accuracy": round(em.aux_speed_accuracy, 4),
                "aux_role_accuracy": round(em.aux_role_accuracy, 4),
            }
            for em in epoch_metrics_list
        ],
        "epoch_resources": epoch_resources_list,
        "test_results": test_results,
        "final_results": {
            "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 4),
            "total_epochs_trained": len(epoch_metrics_list),
            "best_checkpoint": str(best_path),
            "total_training_time_sec": round(total_train_time, 2),
            "total_training_time_min": round(total_train_time / 60, 2),
            "test_evaluation_time_sec": round(eval_time, 2),
            "total_wall_time_sec": round(total_wall_time, 2),
            "total_wall_time_min": round(total_wall_time / 60, 2),
        },
        "resource_summary": {
            "peak_ram_gb": final_snap.get("ram_peak_gb", 0.0),
            "final_ram_gb": final_snap.get("ram_used_gb", 0.0),
            "cpu_user_time_sec": final_snap.get("cpu_user_time_sec", 0.0),
            "cpu_sys_time_sec": final_snap.get("cpu_sys_time_sec", 0.0),
            "gpu_used": torch.cuda.is_available(),
            "gpu_memory_used_gb": final_snap.get("gpu_memory_used_gb", 0.0),
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    logger.info("=" * 70)
    logger.info("PHASE 4 TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model: {config.num_layers}L/{config.hidden_dim}d/{config.num_heads}H ({param_count:,} params)")
    logger.info(f"Data: {data_stats['total_battles']} battles, {data_stats['train_examples']} train examples")
    logger.info(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    logger.info(f"Test accuracy: {test_results['test_accuracy']:.4f}")
    logger.info(f"Test top-3 accuracy: {test_results['test_top3_accuracy']:.4f}")
    logger.info(f"Total time: {total_wall_time/60:.1f} min")
    logger.info(f"Peak RAM: {final_snap.get('ram_peak_gb', 0.0):.2f} GB")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
