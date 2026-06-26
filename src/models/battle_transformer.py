"""Structured transformer model for Pokemon battle action prediction.

Phase 4 core model: transformer encoder with structured token input,
policy head for action selection, auxiliary hidden-info prediction heads,
and optional value head for win probability estimation.

Architecture:
    1. Embedding layer converts raw features into token embeddings
    2. Transformer encoder processes tokens with self-attention
    3. Policy head scores candidate actions via cross-attention
    4. Auxiliary head predicts hidden opponent info (item, speed, role, etc.)
    5. Value head estimates win probability (optional)

Input tokens (Gen 3 OU):
    - 6 own-team slot tokens (full info)
    - 6 opponent-team slot tokens (partial/unknown info, no team preview)
    - 1 field token (weather, hazards, screens)
    - 1 context token (turn number, opponent revealed count, etc.)
    Total: 14 tokens per turn step
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tensorizer import (
    CONTEXT_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    POKEMON_CATEGORICAL_DIM,
    POKEMON_CONTINUOUS_DIM,
    POKEMON_BINARY_DIM,
    POKEMON_FEATURE_DIM,
)
from src.data.observation import MAX_TEAM_SIZE
from src.environment.action_space import NUM_ACTIONS


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class TransformerConfig:
    """Configuration for the battle transformer model."""

    # Encoder
    num_layers: int = 6
    hidden_dim: int = 384
    num_heads: int = 6
    dropout: float = 0.1
    activation: str = "gelu"
    ffn_multiplier: int = 4

    # Vocabulary sizes (set from actual vocabs, Gen 3 defaults)
    species_vocab_size: int = 200
    moves_vocab_size: int = 400
    items_vocab_size: int = 80
    abilities_vocab_size: int = 150
    types_vocab_size: int = 150
    status_vocab_size: int = 20
    weather_vocab_size: int = 20
    terrain_vocab_size: int = 5

    # Embedding dimensions
    species_embedding_dim: int = 64
    move_embedding_dim: int = 32
    item_embedding_dim: int = 32
    ability_embedding_dim: int = 32
    type_embedding_dim: int = 16
    status_embedding_dim: int = 8
    weather_embedding_dim: int = 8
    terrain_embedding_dim: int = 8

    # Auxiliary hidden-info head
    auxiliary_loss_weight: float = 0.2
    num_item_classes: int = 25  # Gen 3 item taxonomy (~25 classes)
    num_speed_buckets: int = 5
    num_role_archetypes: int = 8
    num_move_families: int = 10

    # Policy loss tuning
    switch_weight: float = 1.0  # Upweight switch actions in policy loss (1.0 = no upweighting)
    label_smoothing: float = 0.0  # Label smoothing for policy loss

    # Policy head type
    use_candidate_head: bool = False  # Use candidate-conditioned policy head instead of pooled MLP
    use_split_head: bool = False  # Use separate move/switch scoring pathways
    move_identity_candidates: bool = False  # Use actual move IDs instead of slot embeddings for move candidates
    policy_head_layers: int = 1  # Number of cross-attention layers in split policy head
    action_self_attention: bool = False  # Self-attention over action candidates before scoring

    # Value head
    use_value_head: bool = True
    value_loss_weight: float = 0.1

    # Sequence
    max_seq_len: int = 50  # Max turns in a battle

    # Efficiency toggles
    prune_dead_features: bool = True

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> TransformerConfig:
        """Create config from a Hydra/YAML dict."""
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in cfg.items() if k in valid_fields})

    @classmethod
    def from_vocabs(cls, vocabs: Any, **kwargs: Any) -> TransformerConfig:
        """Create config with vocabulary sizes from BattleVocabularies."""
        return cls(
            species_vocab_size=vocabs.species.size,
            moves_vocab_size=vocabs.moves.size,
            items_vocab_size=vocabs.items.size,
            abilities_vocab_size=vocabs.abilities.size,
            types_vocab_size=vocabs.types.size,
            status_vocab_size=vocabs.status.size,
            weather_vocab_size=vocabs.weather.size,
            terrain_vocab_size=vocabs.terrain.size,
            **kwargs,
        )

    @classmethod
    def smoke_test(cls, vocabs: Any | None = None, **kwargs: Any) -> TransformerConfig:
        """Tiny config for smoke testing (2 layers, 128 dim)."""
        base = dict(
            num_layers=2,
            hidden_dim=128,
            num_heads=4,
            dropout=0.1,
            species_embedding_dim=32,
            move_embedding_dim=16,
            item_embedding_dim=16,
            ability_embedding_dim=16,
            type_embedding_dim=8,
            status_embedding_dim=8,
            weather_embedding_dim=8,
            terrain_embedding_dim=8,
        )
        base.update(kwargs)
        if vocabs is not None:
            base.update(
                species_vocab_size=vocabs.species.size,
                moves_vocab_size=vocabs.moves.size,
                items_vocab_size=vocabs.items.size,
                abilities_vocab_size=vocabs.abilities.size,
                types_vocab_size=vocabs.types.size,
                status_vocab_size=vocabs.status.size,
                weather_vocab_size=vocabs.weather.size,
                terrain_vocab_size=vocabs.terrain.size,
            )
        return cls(**base)

    @classmethod
    def p8_gen3(cls, vocabs: Any | None = None, **kwargs: Any) -> TransformerConfig:
        """P8 profile tuned for Gen 3 OU (4L/256d/4H, ~2.5M params).

        Standard P8 architecture scaled for Gen 3's smaller metagame:
        smaller embedding tables, higher aux weight (hidden info is more
        critical without team preview), and Gen 3 item taxonomy.
        """
        base = dict(
            num_layers=4,
            hidden_dim=256,
            num_heads=4,
            ffn_multiplier=4,
            use_value_head=True,
            prune_dead_features=True,
            species_embedding_dim=48,   # Smaller (~100 species vs ~600)
            move_embedding_dim=24,      # Smaller (~200 moves vs ~600)
            item_embedding_dim=12,      # Much smaller (~30 items vs ~200)
            ability_embedding_dim=12,   # Smaller (~80 abilities vs ~300)
            type_embedding_dim=12,      # Slightly smaller (no Fairy combos)
            max_seq_len=20,
            num_item_classes=25,        # Gen 3 item taxonomy
            auxiliary_loss_weight=0.2,
        )
        base.update(kwargs)
        if vocabs is not None:
            base.update(
                species_vocab_size=vocabs.species.size,
                moves_vocab_size=vocabs.moves.size,
                items_vocab_size=vocabs.items.size,
                abilities_vocab_size=vocabs.abilities.size,
                types_vocab_size=vocabs.types.size,
                status_vocab_size=vocabs.status.size,
                weather_vocab_size=vocabs.weather.size,
                terrain_vocab_size=vocabs.terrain.size,
            )
        return cls(**base)

    @classmethod
    def p8_lean(cls, vocabs: Any | None = None, **kwargs: Any) -> TransformerConfig:
        """P8-Lean profile tuned for Gen 3 OU (~1.2-1.5M params).

        Compressed architecture for fast iteration on Gen 3's compact metagame:
        3 layers, smaller hidden dim, reduced embeddings.
        """
        base = dict(
            num_layers=3,
            hidden_dim=192,  # Smaller (fewer entities in Gen 3)
            num_heads=4,
            ffn_multiplier=3,
            use_value_head=False,
            prune_dead_features=True,
            species_embedding_dim=48,
            move_embedding_dim=24,
            item_embedding_dim=12,  # Fewer items in Gen 3
            ability_embedding_dim=12,  # Fewer abilities in Gen 3
            type_embedding_dim=12,
            max_seq_len=5,
            num_item_classes=25,  # Gen 3 item taxonomy
            auxiliary_loss_weight=0.2,
        )
        base.update(kwargs)
        if vocabs is not None:
            base.update(
                species_vocab_size=vocabs.species.size,
                moves_vocab_size=vocabs.moves.size,
                items_vocab_size=vocabs.items.size,
                abilities_vocab_size=vocabs.abilities.size,
                types_vocab_size=vocabs.types.size,
                status_vocab_size=vocabs.status.size,
                weather_vocab_size=vocabs.weather.size,
                terrain_vocab_size=vocabs.terrain.size,
            )
        return cls(**base)


# ── Token number constants ───────────────────────────────────────────────

NUM_OWN_SLOTS = MAX_TEAM_SIZE       # 6
NUM_OPP_SLOTS = MAX_TEAM_SIZE       # 6
NUM_FIELD_TOKENS = 1
NUM_CONTEXT_TOKENS = 1
TOKENS_PER_STEP = NUM_OWN_SLOTS + NUM_OPP_SLOTS + NUM_FIELD_TOKENS + NUM_CONTEXT_TOKENS  # 14


# ── Embedding layers ────────────────────────────────────────────────────


class PokemonEmbedding(nn.Module):
    """Converts raw pokemon feature vector into a dense embedding.

    Takes the 28-dim raw feature vector (9 categorical + 14 continuous + 5 binary)
    and produces a hidden_dim embedding via learned categorical embeddings +
    linear projection of continuous/binary features.
    """

    def __init__(self, config: TransformerConfig, move_embedding: nn.Embedding | None = None) -> None:
        super().__init__()
        c = config
        self.config = config

        # Categorical embeddings
        # Feature layout: [species, move1, move2, move3, move4, item, ability, types, status]
        self.species_emb = nn.Embedding(c.species_vocab_size, c.species_embedding_dim, padding_idx=0)
        self.move_emb = move_embedding or nn.Embedding(c.moves_vocab_size, c.move_embedding_dim, padding_idx=0)
        self.item_emb = nn.Embedding(c.items_vocab_size, c.item_embedding_dim, padding_idx=0)
        self.ability_emb = nn.Embedding(c.abilities_vocab_size, c.ability_embedding_dim, padding_idx=0)
        self.type_emb = nn.Embedding(c.types_vocab_size, c.type_embedding_dim, padding_idx=0)
        self.status_emb = nn.Embedding(c.status_vocab_size, c.status_embedding_dim, padding_idx=0)

        # Total embedding dim from categoricals
        cat_dim = (
            c.species_embedding_dim
            + 4 * c.move_embedding_dim  # 4 moves
            + c.item_embedding_dim
            + c.ability_embedding_dim
            + c.type_embedding_dim
            + c.status_embedding_dim
        )

        # Continuous + binary features projection
        cont_binary_dim = POKEMON_CONTINUOUS_DIM + POKEMON_BINARY_DIM

        # Project to hidden_dim
        self.proj = nn.Linear(cat_dim + cont_binary_dim, c.hidden_dim)
        self.norm = nn.LayerNorm(c.hidden_dim)

    def forward(self, pokemon_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pokemon_features: (..., POKEMON_FEATURE_DIM) raw features

        Returns:
            (..., hidden_dim) pokemon token embedding
        """
        # Split into categorical, continuous, binary
        # Categorical indices: [0:9] -> species(0), moves(1:5), item(5), ability(6), types(7), status(8)
        cat_indices = pokemon_features[..., :POKEMON_CATEGORICAL_DIM].long().clamp(min=0)

        species = self.species_emb(cat_indices[..., 0])
        move1 = self.move_emb(cat_indices[..., 1])
        move2 = self.move_emb(cat_indices[..., 2])
        move3 = self.move_emb(cat_indices[..., 3])
        move4 = self.move_emb(cat_indices[..., 4])
        item = self.item_emb(cat_indices[..., 5])
        ability = self.ability_emb(cat_indices[..., 6])
        types = self.type_emb(cat_indices[..., 7])
        status = self.status_emb(cat_indices[..., 8])

        cat_emb = torch.cat([species, move1, move2, move3, move4, item, ability, types, status], dim=-1)

        # Continuous + binary features: [9:28]
        cont_binary = pokemon_features[..., POKEMON_CATEGORICAL_DIM:]

        # Combine and project
        combined = torch.cat([cat_emb, cont_binary], dim=-1)
        return self.norm(self.proj(combined))


class FieldEmbedding(nn.Module):
    """Converts raw field features into a dense embedding."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        c = config
        self.config = config

        # Categorical: weather(0), terrain(1)
        self.weather_emb = nn.Embedding(c.weather_vocab_size, c.weather_embedding_dim, padding_idx=0)
        self.terrain_emb = nn.Embedding(c.terrain_vocab_size, c.terrain_embedding_dim, padding_idx=0)

        # Binary features: side conditions (often all-zero in processed P8 data).
        binary_dim = FIELD_FEATURE_DIM - 2
        self.prune_dead_features = c.prune_dead_features
        if self.prune_dead_features:
            binary_dim = 0

        total_in = c.weather_embedding_dim + c.terrain_embedding_dim + binary_dim
        self.proj = nn.Linear(total_in, c.hidden_dim)
        self.norm = nn.LayerNorm(c.hidden_dim)

    def forward(self, field_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field_features: (..., FIELD_FEATURE_DIM) raw features

        Returns:
            (..., hidden_dim) field token embedding
        """
        cat_indices = field_features[..., :2].long().clamp(min=0)
        weather = self.weather_emb(cat_indices[..., 0])
        terrain = self.terrain_emb(cat_indices[..., 1])
        pieces = [weather, terrain]
        if not self.prune_dead_features:
            pieces.append(field_features[..., 2:])

        combined = torch.cat(pieces, dim=-1)
        return self.norm(self.proj(combined))


class ContextEmbedding(nn.Module):
    """Converts raw context features into a dense embedding."""

    def __init__(self, config: TransformerConfig, move_embedding: nn.Embedding | None = None) -> None:
        super().__init__()
        c = config
        self.config = config

        # Context (Gen 3): turn_num(0), opp_remaining(1), num_opp_revealed(2),
        #                   forced_switch(3), is_lead_turn(4),
        #                   prev_player_move(5), prev_opponent_move(6)
        self.prev_move_emb = move_embedding or nn.Embedding(c.moves_vocab_size, c.move_embedding_dim, padding_idx=0)

        cont_dim = 5  # turn_num, opp_remaining, num_opp_revealed, forced_switch, is_lead_turn
        total_in = cont_dim + 2 * c.move_embedding_dim

        self.proj = nn.Linear(total_in, c.hidden_dim)
        self.norm = nn.LayerNorm(c.hidden_dim)

    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_features: (..., CONTEXT_FEATURE_DIM) raw features

        Returns:
            (..., hidden_dim) context token embedding
        """
        cont = context_features[..., :5]
        move_indices = context_features[..., 5:7].long().clamp(min=0)

        prev_player = self.prev_move_emb(move_indices[..., 0])
        prev_opp = self.prev_move_emb(move_indices[..., 1])

        combined = torch.cat([cont, prev_player, prev_opp], dim=-1)
        return self.norm(self.proj(combined))


# ── Positional / token type embeddings ──────────────────────────────────


class TokenTypeEmbedding(nn.Module):
    """Learned embeddings for different token types (own/opp/field/context)."""

    NUM_TOKEN_TYPES = 4  # own_pokemon, opp_pokemon, field, context

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(self.NUM_TOKEN_TYPES, hidden_dim)

    def forward(self, num_own: int = 6, num_opp: int = 6) -> torch.Tensor:
        """Returns (TOKENS_PER_STEP, hidden_dim) type embeddings."""
        type_ids = torch.cat([
            torch.full((num_own,), 0, dtype=torch.long),    # own team
            torch.full((num_opp,), 1, dtype=torch.long),    # opp team
            torch.full((NUM_FIELD_TOKENS,), 2, dtype=torch.long),   # field
            torch.full((NUM_CONTEXT_TOKENS,), 3, dtype=torch.long), # context
        ])
        return self.embedding(type_ids.to(self.embedding.weight.device))


class SlotPositionEmbedding(nn.Module):
    """Position embedding for slot position within each token type group."""

    def __init__(self, hidden_dim: int, max_slots: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_slots, hidden_dim)
        self.max_slots = max_slots

    def forward(self, num_own: int = 6, num_opp: int = 6) -> torch.Tensor:
        """Returns (TOKENS_PER_STEP, hidden_dim) positional embeddings."""
        dev = self.embedding.weight.device
        own_pos = torch.arange(num_own, device=dev)
        opp_pos = torch.arange(num_opp, device=dev)
        field_pos = torch.zeros(NUM_FIELD_TOKENS, dtype=torch.long, device=dev)
        context_pos = torch.zeros(NUM_CONTEXT_TOKENS, dtype=torch.long, device=dev)
        positions = torch.cat([own_pos, opp_pos, field_pos, context_pos])
        return self.embedding(positions)


class TurnPositionEmbedding(nn.Module):
    """Sinusoidal or learned positional embedding for turn position in sequence."""

    def __init__(self, hidden_dim: int, max_len: int = 100) -> None:
        super().__init__()
        # Use sinusoidal encoding
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:hidden_dim // 2]) if hidden_dim % 2 else torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns (seq_len, hidden_dim) positional embeddings."""
        return self.pe[:seq_len]


# ── Transformer encoder ────────────────────────────────────────────────


class BattleTransformerEncoder(nn.Module):
    """Transformer encoder over structured battle tokens."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings
        self.shared_move_emb = nn.Embedding(config.moves_vocab_size, config.move_embedding_dim, padding_idx=0)
        self.pokemon_emb = PokemonEmbedding(config, move_embedding=self.shared_move_emb)
        self.field_emb = FieldEmbedding(config)
        self.context_emb = ContextEmbedding(config, move_embedding=self.shared_move_emb)

        # Positional/type embeddings
        self.token_type_emb = TokenTypeEmbedding(config.hidden_dim)
        self.slot_pos_emb = SlotPositionEmbedding(config.hidden_dim)
        self.turn_pos_emb = TurnPositionEmbedding(config.hidden_dim, config.max_seq_len)

        # Transformer layers
        act = nn.GELU() if config.activation == "gelu" else nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * config.ffn_multiplier,
            dropout=config.dropout,
            activation=act,
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        self.embed_dropout = nn.Dropout(config.dropout)

    def embed_step(
        self,
        own_team: torch.Tensor,
        opponent_team: torch.Tensor,
        field: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a single time-step into token embeddings.

        Args:
            own_team: (..., 6, POKEMON_FEATURE_DIM)
            opponent_team: (..., 6, POKEMON_FEATURE_DIM)
            field: (..., FIELD_FEATURE_DIM)
            context: (..., CONTEXT_FEATURE_DIM)

        Returns:
            (..., TOKENS_PER_STEP, hidden_dim)
        """
        # Embed pokemon tokens: (..., 6, hidden_dim) each
        own_embs = self.pokemon_emb(own_team)
        opp_embs = self.pokemon_emb(opponent_team)

        # Embed field and context: (..., hidden_dim) -> (..., 1, hidden_dim)
        field_emb = self.field_emb(field).unsqueeze(-2)
        ctx_emb = self.context_emb(context).unsqueeze(-2)

        # Concatenate: (..., 14, hidden_dim)
        tokens = torch.cat([own_embs, opp_embs, field_emb, ctx_emb], dim=-2)

        # Add token type and slot position embeddings
        type_emb = self.token_type_emb()
        slot_emb = self.slot_pos_emb()
        tokens = tokens + type_emb + slot_emb

        return tokens

    def forward(
        self,
        own_team: torch.Tensor,
        opponent_team: torch.Tensor,
        field: torch.Tensor,
        context: torch.Tensor,
        seq_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the encoder.

        For sequence input (batch, seq, ...):
            Flattens sequence of step-tokens into a single token sequence,
            adds turn positional encoding, runs transformer.

        For single-step input (batch, ...):
            Wraps as seq_len=1 and processes.

        Args:
            own_team: (batch, [seq,] 6, POKEMON_FEATURE_DIM)
            opponent_team: same
            field: (batch, [seq,] FIELD_FEATURE_DIM)
            context: (batch, [seq,] CONTEXT_FEATURE_DIM)
            seq_len: (batch,) actual sequence lengths for masking

        Returns:
            (batch, total_tokens, hidden_dim) encoder output
        """
        is_sequence = own_team.dim() == 4  # (batch, seq, 6, feat)

        if is_sequence:
            batch_size, seq, _, _ = own_team.shape

            # Embed each step: (batch, seq, TOKENS_PER_STEP, hidden_dim)
            step_tokens = self.embed_step(own_team, opponent_team, field, context)

            # Add turn-level positional encoding
            # (seq, hidden_dim) -> broadcast to (1, seq, 1, hidden_dim)
            turn_pe = self.turn_pos_emb(seq).unsqueeze(0).unsqueeze(2)
            step_tokens = step_tokens + turn_pe

            # Flatten to (batch, seq * TOKENS_PER_STEP, hidden_dim)
            total_tokens = seq * TOKENS_PER_STEP
            tokens = step_tokens.reshape(batch_size, total_tokens, -1)

            # Build attention mask for padding
            attn_mask = None
            if seq_len is not None:
                # Create mask: True means position is padded (should be masked)
                token_positions = torch.arange(total_tokens, device=tokens.device)
                # Each step has TOKENS_PER_STEP tokens
                turn_for_token = token_positions // TOKENS_PER_STEP  # which turn each token belongs to
                # Mask tokens whose turn >= seq_len for each batch item
                # (batch, total_tokens)
                attn_mask = turn_for_token.unsqueeze(0) >= seq_len.unsqueeze(1)
        else:
            # Single step: (batch, 6, feat) -> treat as seq_len=1
            batch_size = own_team.shape[0]

            tokens = self.embed_step(own_team, opponent_team, field, context)
            # tokens: (batch, TOKENS_PER_STEP, hidden_dim)
            attn_mask = None

        tokens = self.embed_dropout(tokens)

        # Run transformer
        output = self.transformer(tokens, src_key_padding_mask=attn_mask)

        return output


# ── Output heads ────────────────────────────────────────────────────────


class PolicyHead(nn.Module):
    """Action prediction head using pooled encoder output.

    Pools encoder tokens (mean over non-padded tokens) and projects
    to action logits. Legal action masking is applied externally.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, NUM_ACTIONS),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, num_tokens, hidden_dim)
            attn_mask: (batch, num_tokens) True = padded/masked

        Returns:
            (batch, NUM_ACTIONS) logits
        """
        if attn_mask is not None:
            # Mean pool over non-padded tokens
            mask = (~attn_mask).float().unsqueeze(-1)  # (batch, tokens, 1)
            pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = encoder_output.mean(dim=1)

        return self.mlp(pooled)


class CandidateConditionedPolicyHead(nn.Module):
    """Candidate-conditioned action scorer using cross-attention.

    Instead of mean-pooling to 9 fixed logits, this head:
    1. Builds candidate embeddings for each legal action
       - Move candidates: active pokemon token + move slot embedding
       - Switch candidates: bench pokemon token (the actual switch target)
    2. Each candidate queries the encoder memory via cross-attention
    3. A small MLP scores each attended representation to produce one logit

    This lets the model reason about *actual* move/switch targets rather
    than abstract slot IDs, which is especially important for switches
    where slot 3 can contain completely different pokemon across battles.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        h = config.hidden_dim

        # Move candidate: combine active pokemon token with move slot embedding
        # We use a learned embedding per move slot (0-3) to distinguish which move
        self.move_slot_emb = nn.Embedding(4, h)
        self.move_combine = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )

        # Switch candidate: bench pokemon token is already a good representation
        # but we add a learned "switch intent" projection to distinguish from encoding
        self.switch_proj = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )

        # Cross-attention: candidate queries attend over encoder memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(h)

        # Score MLP: attended representation -> scalar logit
        self.score_mlp = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, 1),
        )

    def forward(
        self,
        last_step_tokens: torch.Tensor,
        encoder_memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidate actions against encoded battle state.

        Args:
            last_step_tokens: (batch, TOKENS_PER_STEP, hidden_dim)
                Token layout: [own0..own5, opp0..opp5, field, context]
                own0 = active pokemon, own1..own5 = bench slots
            encoder_memory: (batch, total_tokens, hidden_dim)
                Full encoder output (all turns) for cross-attention
            memory_mask: (batch, total_tokens) True = padded/masked

        Returns:
            (batch, NUM_ACTIONS) raw logits (before legal masking)
        """
        batch_size = last_step_tokens.shape[0]
        h = last_step_tokens.shape[-1]
        device = last_step_tokens.device

        # Extract key tokens from last step
        active_token = last_step_tokens[:, 0, :]  # (batch, h) - active pokemon
        bench_tokens = last_step_tokens[:, 1:6, :]  # (batch, 5, h) - bench slots 1-5

        # --- Build move candidates (actions 0-3) ---
        # Each move candidate = combine(active_pokemon, move_slot_embedding)
        move_slot_ids = torch.arange(4, device=device)  # [0, 1, 2, 3]
        move_slot_embs = self.move_slot_emb(move_slot_ids)  # (4, h)
        move_slot_embs = move_slot_embs.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 4, h)

        active_expanded = active_token.unsqueeze(1).expand(-1, 4, -1)  # (batch, 4, h)
        move_candidates = self.move_combine(
            torch.cat([active_expanded, move_slot_embs], dim=-1)
        )  # (batch, 4, h)

        # --- Build switch candidates (actions 4-8) ---
        # Each switch candidate = project(bench_pokemon_token)
        switch_candidates = self.switch_proj(bench_tokens)  # (batch, 5, h)

        # --- Combine all candidates ---
        all_candidates = torch.cat([move_candidates, switch_candidates], dim=1)  # (batch, 9, h)

        # --- Cross-attention: candidates query the encoder memory ---
        # Convert memory_mask for MultiheadAttention (True = ignore)
        attn_out, _ = self.cross_attn(
            query=all_candidates,
            key=encoder_memory,
            value=encoder_memory,
            key_padding_mask=memory_mask,
        )  # (batch, 9, h)

        # Residual + norm
        scored = self.cross_norm(all_candidates + attn_out)  # (batch, 9, h)

        # --- Score each candidate ---
        logits = self.score_mlp(scored).squeeze(-1)  # (batch, 9)

        return logits


class _CrossAttentionBlock(nn.Module):
    """Single cross-attention + FFN block with pre-norm residuals."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm cross-attention with residual
        normed = self.cross_norm(query)
        attn_out, _ = self.cross_attn(
            query=normed, key=memory, value=memory,
            key_padding_mask=memory_mask,
        )
        x = query + attn_out
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SplitMoveSwitchPolicyHead(nn.Module):
    """Separate move and switch scoring pathways with shared encoder.

    Splits the policy into two specialized branches:
    - MoveHead: scores up to 4 move candidates using active pokemon + move embeddings
    - SwitchHead: scores up to 5 switch candidates using bench pokemon tokens

    Each head uses its own cross-attention stack over the shared encoder memory,
    allowing moves and switches to attend to different battle-state features.
    Final output concatenates move scores (0-3) and switch scores (4-8) into
    the standard 9-action logit vector.

    When move_identity_candidates=True, move candidates use the actual move ID
    embedding (e.g. "Recover", "Ice Punch") instead of abstract slot position
    embeddings (move1, move2, move3, move4).

    When policy_head_layers > 1, each pathway uses a stack of cross-attention
    blocks for deeper strategic reasoning.

    When action_self_attention=True, all 9 candidates attend to each other
    before scoring, enabling relative action reasoning.
    """

    def __init__(
        self,
        config: TransformerConfig,
        move_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        h = config.hidden_dim
        self.use_move_identity = config.move_identity_candidates
        self.use_action_self_attn = config.action_self_attention
        n_policy_layers = config.policy_head_layers

        # ── Move pathway ──
        if self.use_move_identity:
            self.move_id_proj = nn.Linear(config.move_embedding_dim, h)
            self.move_emb = move_embedding  # shared with encoder
        else:
            self.move_slot_emb = nn.Embedding(4, h)
        self.move_combine = nn.Sequential(
            nn.Linear(2 * h, h), nn.GELU(), nn.LayerNorm(h),
        )
        self.move_cross_blocks = nn.ModuleList([
            _CrossAttentionBlock(h, config.num_heads, config.dropout)
            for _ in range(n_policy_layers)
        ])
        self.move_scorer = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, 1),
        )

        # ── Switch pathway ──
        self.switch_proj = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
        )
        self.switch_cross_blocks = nn.ModuleList([
            _CrossAttentionBlock(h, config.num_heads, config.dropout)
            for _ in range(n_policy_layers)
        ])
        self.switch_scorer = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, 1),
        )

        # ── Action self-attention (optional) ──
        if self.use_action_self_attn:
            self.action_self_attn = nn.MultiheadAttention(
                embed_dim=h, num_heads=config.num_heads,
                dropout=config.dropout, batch_first=True,
            )
            self.action_self_norm = nn.LayerNorm(h)

    def forward(
        self,
        last_step_tokens: torch.Tensor,
        encoder_memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        active_move_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score move and switch candidates through separate pathways.

        Args:
            last_step_tokens: (batch, TOKENS_PER_STEP, hidden_dim)
                Token layout: [own0..own5, opp0..opp5, field, context]
                own0 = active pokemon, own1..own5 = bench slots
            encoder_memory: (batch, total_tokens, hidden_dim)
            memory_mask: (batch, total_tokens) True = padded/masked
            active_move_ids: (batch, 4) actual move IDs for the active pokemon
                Only used when move_identity_candidates=True

        Returns:
            (batch, NUM_ACTIONS) raw logits [move0..move3, switch2..switch6]
        """
        batch_size = last_step_tokens.shape[0]
        device = last_step_tokens.device

        active_token = last_step_tokens[:, 0, :]   # (batch, h)
        bench_tokens = last_step_tokens[:, 1:6, :]  # (batch, 5, h)

        # ── Move scoring ──
        active_expanded = active_token.unsqueeze(1).expand(-1, 4, -1)  # (batch, 4, h)
        if self.use_move_identity and active_move_ids is not None:
            move_embs = self.move_emb(active_move_ids)  # (batch, 4, move_emb_dim)
            move_embs = self.move_id_proj(move_embs)  # (batch, 4, h)
        else:
            move_slot_ids = torch.arange(4, device=device)
            move_embs = self.move_slot_emb(move_slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
        move_candidates = self.move_combine(
            torch.cat([active_expanded, move_embs], dim=-1)
        )  # (batch, 4, h)

        for block in self.move_cross_blocks:
            move_candidates = block(move_candidates, encoder_memory, memory_mask)

        # ── Switch scoring ──
        switch_candidates = self.switch_proj(bench_tokens)  # (batch, 5, h)

        for block in self.switch_cross_blocks:
            switch_candidates = block(switch_candidates, encoder_memory, memory_mask)

        # ── Action self-attention (optional) ──
        if self.use_action_self_attn:
            all_candidates = torch.cat([move_candidates, switch_candidates], dim=1)  # (batch, 9, h)
            normed = self.action_self_norm(all_candidates)
            self_attn_out, _ = self.action_self_attn(
                query=normed, key=normed, value=normed,
            )
            all_candidates = all_candidates + self_attn_out
            move_candidates = all_candidates[:, :4, :]
            switch_candidates = all_candidates[:, 4:, :]

        # ── Score ──
        move_logits = self.move_scorer(move_candidates).squeeze(-1)  # (batch, 4)
        switch_logits = self.switch_scorer(switch_candidates).squeeze(-1)  # (batch, 5)

        # ── Combine into 9-action logits ──
        return torch.cat([move_logits, switch_logits], dim=1)  # (batch, 9)


class AuxiliaryHead(nn.Module):
    """Predicts hidden opponent information from encoder output.

    For each opponent pokemon slot, predicts:
    - Item class (categorical, 25 classes for Gen 3)
    - Speed bucket (ordinal, 5 classes)
    - Role archetype (categorical, 8 classes)
    - Threat profile (joint speed-bucket x role-archetype)
    - Move family presence (multi-label, 10 families)
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        h = config.hidden_dim

        # Per-opponent-slot predictions from opponent slot tokens
        self.item_head = nn.Sequential(
            nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(h // 2, config.num_item_classes),
        )
        threat_classes = config.num_speed_buckets * config.num_role_archetypes
        self.num_speed_buckets = config.num_speed_buckets
        self.num_role_archetypes = config.num_role_archetypes
        self.threat_profile_head = nn.Sequential(
            nn.Linear(h, h // 4), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(h // 4, threat_classes),
        )
        self.move_family_head = nn.Sequential(
            nn.Linear(h, h // 4), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(h // 4, config.num_move_families),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        opp_token_indices: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_output: (batch, num_tokens, hidden_dim)
            opp_token_indices: (start, end) indices of opponent slot tokens
                              in the last step. Default: (6, 12) for single-step.

        Returns:
            Dict of predictions, each (batch, 6, num_classes)
        """
        if opp_token_indices is None:
            opp_token_indices = (NUM_OWN_SLOTS, NUM_OWN_SLOTS + NUM_OPP_SLOTS)

        start, end = opp_token_indices
        # Get opponent slot token embeddings: (batch, 6, hidden_dim)
        opp_tokens = encoder_output[:, start:end]

        threat_flat = self.threat_profile_head(opp_tokens)
        threat_joint = threat_flat.view(*threat_flat.shape[:2], self.num_speed_buckets, self.num_role_archetypes)

        return {
            "item_logits": self.item_head(opp_tokens),        # (batch, 6, num_item_classes)
            "speed_logits": torch.logsumexp(threat_joint, dim=-1),  # (batch, 6, num_speed_buckets)
            "role_logits": torch.logsumexp(threat_joint, dim=-2),   # (batch, 6, num_role_archetypes)
            "threat_profile_logits": threat_flat,
            "move_family_logits": self.move_family_head(opp_tokens),  # (batch, 6, num_move_families)
        }


class ValueHead(nn.Module):
    """Win probability prediction head."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, num_tokens, hidden_dim)
            attn_mask: (batch, num_tokens) True = masked

        Returns:
            (batch,) win probability logits (pre-sigmoid)
        """
        if attn_mask is not None:
            mask = (~attn_mask).float().unsqueeze(-1)
            pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = encoder_output.mean(dim=1)

        return self.mlp(pooled).squeeze(-1)


# ── Full model ──────────────────────────────────────────────────────────


@dataclass
class TransformerOutput:
    """Output from the BattleTransformer model."""

    policy_logits: torch.Tensor           # (batch, NUM_ACTIONS)
    auxiliary_preds: dict[str, torch.Tensor] | None = None
    value_logits: torch.Tensor | None = None
    encoder_output: torch.Tensor | None = None  # For analysis


class BattleTransformer(nn.Module):
    """Structured transformer for Pokemon battle action prediction.

    Combines encoder + policy head + auxiliary head + optional value head.
    Compatible with the existing training pipeline interface (same forward
    signature as BaselineMLP/BaselineGRU).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = BattleTransformerEncoder(config)
        if config.use_split_head:
            self.policy_head = SplitMoveSwitchPolicyHead(
                config, move_embedding=self.encoder.shared_move_emb,
            )
        elif config.use_candidate_head:
            self.policy_head = CandidateConditionedPolicyHead(config)
        else:
            self.policy_head = PolicyHead(config)
        self.auxiliary_head = AuxiliaryHead(config)

        self.value_head: ValueHead | None = None
        if config.use_value_head:
            self.value_head = ValueHead(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small standard deviations."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        own_team: torch.Tensor,
        opponent_team: torch.Tensor,
        field: torch.Tensor,
        context: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        seq_len: torch.Tensor | None = None,
        return_auxiliary: bool = True,
        return_value: bool = True,
    ) -> torch.Tensor | TransformerOutput:
        """Forward pass.

        When return_auxiliary=False and return_value=False, returns just logits
        (compatible with BaselineMLP interface). Otherwise returns TransformerOutput.

        Args:
            own_team: (batch, [seq,] 6, POKEMON_FEATURE_DIM)
            opponent_team: same
            field: (batch, [seq,] FIELD_FEATURE_DIM)
            context: (batch, [seq,] CONTEXT_FEATURE_DIM)
            legal_mask: (batch, NUM_ACTIONS) or (batch, seq, NUM_ACTIONS)
            seq_len: (batch,) actual sequence lengths
            return_auxiliary: Whether to compute auxiliary predictions
            return_value: Whether to compute value prediction

        Returns:
            logits (batch, NUM_ACTIONS) if return_auxiliary=False and return_value=False,
            else TransformerOutput with all heads.
        """
        is_sequence = own_team.dim() == 4
        batch_size = own_team.shape[0]

        # Run encoder
        encoder_out = self.encoder(
            own_team, opponent_team, field, context, seq_len=seq_len
        )

        # For sequence models, extract the LAST valid step's tokens for heads
        if is_sequence:
            seq = own_team.shape[1]
            total_tokens = seq * TOKENS_PER_STEP

            # Build padding mask
            attn_mask = None
            if seq_len is not None:
                token_positions = torch.arange(total_tokens, device=encoder_out.device)
                turn_for_token = token_positions // TOKENS_PER_STEP
                attn_mask = turn_for_token.unsqueeze(0) >= seq_len.unsqueeze(1)

            # For policy: use the LAST valid step's tokens
            # We extract the last valid step's range of tokens
            if seq_len is not None:
                last_step = (seq_len - 1).clamp(min=0)
            else:
                last_step = torch.full((batch_size,), seq - 1, device=encoder_out.device)

            # Extract last-step token range for each batch item
            # last_step_start[i] = last_step[i] * TOKENS_PER_STEP
            last_step_start = last_step * TOKENS_PER_STEP

            # Gather last step's tokens: (batch, TOKENS_PER_STEP, hidden_dim)
            indices = last_step_start.unsqueeze(1) + torch.arange(
                TOKENS_PER_STEP, device=encoder_out.device
            ).unsqueeze(0)
            # Clamp indices to valid range
            indices = indices.clamp(max=total_tokens - 1)
            last_step_tokens = torch.gather(
                encoder_out,
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_dim),
            )
        else:
            last_step_tokens = encoder_out
            attn_mask = None

        # Extract active pokemon's move IDs for move-identity candidates
        active_move_ids = None
        if self.config.move_identity_candidates and self.config.use_split_head:
            # Active pokemon raw features: own_team slot 0, categorical indices 1-4
            if is_sequence:
                # Get last valid step's own_team
                # last_step: (batch,) index of last valid step
                active_raw = torch.gather(
                    own_team[:, :, 0, :],  # (batch, seq, feat_dim) for slot 0
                    dim=1,
                    index=last_step.unsqueeze(1).unsqueeze(2).expand(-1, 1, own_team.shape[-1]),
                ).squeeze(1)  # (batch, feat_dim)
            else:
                active_raw = own_team[:, 0, :]  # (batch, feat_dim)
            # Move IDs are categorical indices 1-4
            active_move_ids = active_raw[:, 1:5].long().clamp(min=0)  # (batch, 4)

        # Policy head on last-step tokens
        if self.config.use_split_head or self.config.use_candidate_head:
            policy_logits = self.policy_head(
                last_step_tokens, encoder_out, memory_mask=attn_mask,
                **({"active_move_ids": active_move_ids} if self.config.use_split_head else {}),
            )
        else:
            policy_logits = self.policy_head(last_step_tokens)

        # Apply legal mask
        if legal_mask is not None:
            # Handle sequence legal mask: use last step's mask
            if legal_mask.dim() == 3:
                if seq_len is not None:
                    last_idx = (seq_len - 1).clamp(min=0)
                else:
                    last_idx = torch.full((batch_size,), legal_mask.shape[1] - 1, device=legal_mask.device)
                # Gather last step's legal mask
                last_mask = torch.gather(
                    legal_mask,
                    dim=1,
                    index=last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, NUM_ACTIONS),
                ).squeeze(1)
            else:
                last_mask = legal_mask
            policy_logits = policy_logits.masked_fill(last_mask == 0, float("-inf"))

        # Early return for inference (baseline-compatible)
        if not return_auxiliary and not return_value:
            return policy_logits

        # Auxiliary head on last-step opponent tokens
        aux_preds = None
        if return_auxiliary:
            opp_start = NUM_OWN_SLOTS
            opp_end = NUM_OWN_SLOTS + NUM_OPP_SLOTS
            aux_preds = self.auxiliary_head(
                last_step_tokens, opp_token_indices=(opp_start, opp_end)
            )

        # Value head
        value_logits = None
        if return_value and self.value_head is not None:
            value_logits = self.value_head(last_step_tokens)

        return TransformerOutput(
            policy_logits=policy_logits,
            auxiliary_preds=aux_preds,
            value_logits=value_logits,
            encoder_output=encoder_out,
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss functions ──────────────────────────────────────────────────────


def compute_policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    ignore_index: int = -1,
    switch_weight: float = 1.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Masked cross-entropy loss for action prediction.

    Same as baseline masked_cross_entropy but standalone for clarity.
    Supports per-class weighting for switch actions and label smoothing.
    """
    masked_logits = logits.masked_fill(legal_mask == 0, float("-inf"))

    valid = targets != ignore_index
    if valid.any():
        weight = None
        if switch_weight != 1.0:
            num_actions = logits.shape[-1]
            weight = torch.ones(num_actions, device=logits.device)
            # Actions 4-8 are switches (switch2-switch6)
            weight[4:] = switch_weight
        return F.cross_entropy(
            masked_logits[valid], targets[valid],
            weight=weight, label_smoothing=label_smoothing,
        )
    return torch.tensor(0.0, device=logits.device, requires_grad=True)


def compute_auxiliary_loss(
    aux_preds: dict[str, torch.Tensor],
    aux_targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute multi-task auxiliary loss.

    Args:
        aux_preds: Dict with keys item_logits, speed_logits, role_logits,
                  move_family_logits. Each (batch, 6, num_classes).
        aux_targets: Dict with matching keys, targets as class indices or
                    multi-hot for move families. Use -1 for unknown/masked.

    Returns:
        (total_loss, component_losses_dict)
    """
    total_loss = torch.tensor(0.0, device=next(iter(aux_preds.values())).device, requires_grad=True)
    losses: dict[str, float] = {}
    n_tasks = 0

    for key in ["item_logits", "speed_logits", "role_logits"]:
        if key not in aux_preds or key.replace("_logits", "_targets") not in aux_targets:
            continue
        target_key = key.replace("_logits", "_targets")
        logits = aux_preds[key]  # (batch, 6, num_classes)
        targets = aux_targets[target_key]  # (batch, 6)

        # Flatten
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1).long()

        # Filter valid targets (not -1)
        valid = flat_targets >= 0
        if valid.any():
            loss = F.cross_entropy(flat_logits[valid], flat_targets[valid])
            total_loss = total_loss + loss
            losses[key.replace("_logits", "")] = loss.item()
            n_tasks += 1

    # Move family: multi-label binary cross-entropy
    if "move_family_logits" in aux_preds and "move_family_targets" in aux_targets:
        logits = aux_preds["move_family_logits"]  # (batch, 6, num_families)
        targets = aux_targets["move_family_targets"]  # (batch, 6, num_families)

        # Mask: only compute where we have labels (any family > -1)
        # If all values are -1 for a slot, skip it
        valid_mask = (targets >= 0).any(dim=-1)  # (batch, 6)
        if valid_mask.any():
            valid_logits = logits[valid_mask]
            valid_targets = targets[valid_mask].clamp(min=0).float()
            loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets)
            total_loss = total_loss + loss
            losses["move_family"] = loss.item()
            n_tasks += 1

    # Average over active tasks
    if n_tasks > 0:
        total_loss = total_loss / n_tasks

    return total_loss, losses


def compute_value_loss(
    value_logits: torch.Tensor,
    game_result: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy loss for win probability.

    Args:
        value_logits: (batch,) pre-sigmoid logits
        game_result: (batch,) with 1.0=win, 0.0=loss, 0.5=unknown

    Returns:
        Scalar loss (excludes unknown results with value 0.5).
    """
    # Only train on known outcomes
    valid = (game_result != 0.5)
    if valid.any():
        return F.binary_cross_entropy_with_logits(
            value_logits[valid], game_result[valid]
        )
    return torch.tensor(0.0, device=value_logits.device, requires_grad=True)


def compute_total_loss(
    output: TransformerOutput,
    action_targets: torch.Tensor,
    legal_mask: torch.Tensor,
    aux_targets: dict[str, torch.Tensor] | None = None,
    game_result: torch.Tensor | None = None,
    config: TransformerConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute composite loss: policy + auxiliary + value.

    Returns:
        (total_loss, loss_components_dict)
    """
    aux_weight = config.auxiliary_loss_weight if config else 0.2
    val_weight = config.value_loss_weight if config else 0.1

    loss_dict: dict[str, float] = {}

    # Policy loss
    sw = config.switch_weight if config else 1.0
    ls = config.label_smoothing if config else 0.0
    policy_loss = compute_policy_loss(
        output.policy_logits, action_targets, legal_mask,
        switch_weight=sw, label_smoothing=ls,
    )
    total = policy_loss
    loss_dict["policy"] = policy_loss.item()

    # Auxiliary loss
    if output.auxiliary_preds is not None and aux_targets is not None:
        aux_loss, aux_components = compute_auxiliary_loss(output.auxiliary_preds, aux_targets)
        total = total + aux_weight * aux_loss
        loss_dict["auxiliary"] = aux_loss.item()
        for k, v in aux_components.items():
            loss_dict[f"aux/{k}"] = v

    # Value loss
    if output.value_logits is not None and game_result is not None:
        val_loss = compute_value_loss(output.value_logits, game_result)
        total = total + val_weight * val_loss
        loss_dict["value"] = val_loss.item()

    loss_dict["total"] = total.item()
    return total, loss_dict


# ── Factory function ────────────────────────────────────────────────────


def create_battle_transformer(
    vocabs: Any | None = None,
    config: TransformerConfig | None = None,
    **kwargs: Any,
) -> BattleTransformer:
    """Create a BattleTransformer model.

    Args:
        vocabs: BattleVocabularies for setting vocab sizes.
        config: Optional pre-built config. If None, creates from vocabs + kwargs.
        **kwargs: Additional config overrides.

    Returns:
        BattleTransformer model.
    """
    if config is None:
        if vocabs is not None:
            config = TransformerConfig.from_vocabs(vocabs, **kwargs)
        else:
            config = TransformerConfig(**kwargs)

    return BattleTransformer(config)
