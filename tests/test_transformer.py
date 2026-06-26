"""Tests for the BattleTransformer model (Phase 4).

Tests cover:
- Model instantiation and parameter counting
- Forward pass shapes (single-step and sequence)
- Loss computation (policy, auxiliary, value, total)
- Auxiliary label extraction
- Config creation from vocabs
- Integration with existing tensorizer pipeline
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.observation import MAX_TEAM_SIZE
from src.data.tensorizer import (
    CONTEXT_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    POKEMON_FEATURE_DIM,
)
from src.environment.action_space import NUM_ACTIONS
from src.models.battle_transformer import (
    BattleTransformer,
    BattleTransformerEncoder,
    TransformerConfig,
    TransformerOutput,
    PolicyHead,
    AuxiliaryHead,
    ValueHead,
    PokemonEmbedding,
    FieldEmbedding,
    ContextEmbedding,
    compute_policy_loss,
    compute_auxiliary_loss,
    compute_value_loss,
    compute_total_loss,
    create_battle_transformer,
    TOKENS_PER_STEP,
)


@pytest.fixture
def config():
    """Tiny config for testing."""
    return TransformerConfig(
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        dropout=0.0,
        species_vocab_size=100,
        moves_vocab_size=100,
        items_vocab_size=50,
        abilities_vocab_size=50,
        types_vocab_size=20,
        status_vocab_size=10,
        weather_vocab_size=10,
        terrain_vocab_size=10,
        species_embedding_dim=16,
        move_embedding_dim=8,
        item_embedding_dim=8,
        ability_embedding_dim=8,
        type_embedding_dim=8,
        status_embedding_dim=4,
        weather_embedding_dim=4,
        terrain_embedding_dim=4,
        num_item_classes=10,
        num_speed_buckets=5,
        num_role_archetypes=8,
        num_move_families=10,
        use_value_head=True,
        value_loss_weight=0.1,
        auxiliary_loss_weight=0.3,
    )


@pytest.fixture
def model(config):
    return BattleTransformer(config)


@pytest.fixture
def batch_data():
    """Create a synthetic batch for testing."""
    batch_size = 4
    return {
        "own_team": torch.randn(batch_size, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM),
        "opponent_team": torch.randn(batch_size, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM),
        "field": torch.randn(batch_size, FIELD_FEATURE_DIM),
        "context": torch.randn(batch_size, CONTEXT_FEATURE_DIM),
        "legal_mask": torch.ones(batch_size, NUM_ACTIONS),
        "action": torch.randint(0, NUM_ACTIONS, (batch_size,)),
        "game_result": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }


@pytest.fixture
def seq_batch_data():
    """Create a synthetic sequence batch for testing."""
    batch_size = 3
    seq_len = 5
    return {
        "own_team": torch.randn(batch_size, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM),
        "opponent_team": torch.randn(batch_size, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM),
        "field": torch.randn(batch_size, seq_len, FIELD_FEATURE_DIM),
        "context": torch.randn(batch_size, seq_len, CONTEXT_FEATURE_DIM),
        "legal_mask": torch.ones(batch_size, seq_len, NUM_ACTIONS),
        "action": torch.randint(0, NUM_ACTIONS, (batch_size, seq_len)),
        "game_result": torch.tensor([[1.0]*seq_len, [0.0]*seq_len, [1.0]*seq_len]),
        "seq_len": torch.tensor([5, 3, 4]),
    }


# ── Config tests ────────────────────────────────────────────────────────


class TestTransformerConfig:

    def test_default_config(self):
        config = TransformerConfig()
        assert config.num_layers == 6
        assert config.hidden_dim == 384
        assert config.num_heads == 6

    def test_smoke_test_config(self):
        config = TransformerConfig.smoke_test()
        assert config.num_layers == 2
        assert config.hidden_dim == 128
        assert config.num_heads == 4


# ── Embedding tests ─────────────────────────────────────────────────────


class TestEmbeddings:

    def test_pokemon_embedding_shape(self, config):
        emb = PokemonEmbedding(config)
        x = torch.randn(4, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        out = emb(x)
        assert out.shape == (4, MAX_TEAM_SIZE, config.hidden_dim)

    def test_pokemon_embedding_batch_shapes(self, config):
        emb = PokemonEmbedding(config)
        # Single pokemon
        x = torch.randn(POKEMON_FEATURE_DIM)
        out = emb(x)
        assert out.shape == (config.hidden_dim,)

        # Batch of teams
        x = torch.randn(2, 3, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        out = emb(x)
        assert out.shape == (2, 3, MAX_TEAM_SIZE, config.hidden_dim)

    def test_field_embedding_shape(self, config):
        emb = FieldEmbedding(config)
        x = torch.randn(4, FIELD_FEATURE_DIM)
        out = emb(x)
        assert out.shape == (4, config.hidden_dim)

    def test_context_embedding_shape(self, config):
        emb = ContextEmbedding(config)
        x = torch.randn(4, CONTEXT_FEATURE_DIM)
        out = emb(x)
        assert out.shape == (4, config.hidden_dim)


# ── Encoder tests ────────────────────────────────────────────────────────


class TestEncoder:

    def test_single_step_shape(self, config):
        encoder = BattleTransformerEncoder(config)
        own = torch.randn(2, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        opp = torch.randn(2, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        field = torch.randn(2, FIELD_FEATURE_DIM)
        ctx = torch.randn(2, CONTEXT_FEATURE_DIM)

        out = encoder(own, opp, field, ctx)
        assert out.shape == (2, TOKENS_PER_STEP, config.hidden_dim)

    def test_sequence_shape(self, config):
        encoder = BattleTransformerEncoder(config)
        seq_len = 5
        own = torch.randn(2, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        opp = torch.randn(2, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        field = torch.randn(2, seq_len, FIELD_FEATURE_DIM)
        ctx = torch.randn(2, seq_len, CONTEXT_FEATURE_DIM)
        sl = torch.tensor([5, 3])

        out = encoder(own, opp, field, ctx, seq_len=sl)
        expected_tokens = seq_len * TOKENS_PER_STEP
        assert out.shape == (2, expected_tokens, config.hidden_dim)


# ── Head tests ───────────────────────────────────────────────────────────


class TestHeads:

    def test_policy_head(self, config):
        head = PolicyHead(config)
        encoder_out = torch.randn(4, TOKENS_PER_STEP, config.hidden_dim)
        logits = head(encoder_out)
        assert logits.shape == (4, NUM_ACTIONS)

    def test_auxiliary_head(self, config):
        head = AuxiliaryHead(config)
        encoder_out = torch.randn(4, TOKENS_PER_STEP, config.hidden_dim)
        preds = head(encoder_out)
        assert preds["item_logits"].shape == (4, MAX_TEAM_SIZE, config.num_item_classes)
        assert preds["speed_logits"].shape == (4, MAX_TEAM_SIZE, config.num_speed_buckets)
        assert preds["role_logits"].shape == (4, MAX_TEAM_SIZE, config.num_role_archetypes)
        assert "tera_logits" not in preds
        assert preds["threat_profile_logits"].shape == (
            4, MAX_TEAM_SIZE, config.num_speed_buckets * config.num_role_archetypes
        )
        assert preds["move_family_logits"].shape == (4, MAX_TEAM_SIZE, config.num_move_families)

    def test_value_head(self, config):
        head = ValueHead(config)
        encoder_out = torch.randn(4, TOKENS_PER_STEP, config.hidden_dim)
        value = head(encoder_out)
        assert value.shape == (4,)


# ── Full model tests ────────────────────────────────────────────────────


class TestBattleTransformer:

    def test_single_step_logits_only(self, model, batch_data):
        """Baseline-compatible: returns just logits when not requesting aux/value."""
        logits = model(
            batch_data["own_team"],
            batch_data["opponent_team"],
            batch_data["field"],
            batch_data["context"],
            legal_mask=batch_data["legal_mask"],
            return_auxiliary=False,
            return_value=False,
        )
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (4, NUM_ACTIONS)

    def test_single_step_full_output(self, model, batch_data):
        """Full output with all heads."""
        output = model(
            batch_data["own_team"],
            batch_data["opponent_team"],
            batch_data["field"],
            batch_data["context"],
            legal_mask=batch_data["legal_mask"],
        )
        assert isinstance(output, TransformerOutput)
        assert output.policy_logits.shape == (4, NUM_ACTIONS)
        assert output.auxiliary_preds is not None
        assert output.value_logits is not None
        assert output.value_logits.shape == (4,)

    def test_sequence_forward(self, model, seq_batch_data):
        """Test sequence input with variable lengths."""
        output = model(
            seq_batch_data["own_team"],
            seq_batch_data["opponent_team"],
            seq_batch_data["field"],
            seq_batch_data["context"],
            legal_mask=seq_batch_data["legal_mask"],
            seq_len=seq_batch_data["seq_len"],
        )
        assert isinstance(output, TransformerOutput)
        assert output.policy_logits.shape == (3, NUM_ACTIONS)

    def test_legal_mask_applied(self, model, batch_data):
        """Illegal actions should be masked to -inf."""
        # Make only action 0 and 1 legal
        mask = torch.zeros(4, NUM_ACTIONS)
        mask[:, 0] = 1
        mask[:, 1] = 1

        logits = model(
            batch_data["own_team"],
            batch_data["opponent_team"],
            batch_data["field"],
            batch_data["context"],
            legal_mask=mask,
            return_auxiliary=False,
            return_value=False,
        )

        # Check illegal actions are -inf
        assert (logits[:, 2:] == float("-inf")).all()
        # Legal actions should be finite
        assert torch.isfinite(logits[:, :2]).all()

    def test_parameter_count(self, model):
        count = model.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_gradient_flow(self, model, batch_data, config):
        """Verify gradients flow through all components via total loss."""
        output = model(
            batch_data["own_team"],
            batch_data["opponent_team"],
            batch_data["field"],
            batch_data["context"],
            legal_mask=batch_data["legal_mask"],
        )

        # Use total loss so gradients flow through all heads
        aux_targets = {
            "item_targets": torch.randint(0, config.num_item_classes, (4, 6)),
            "speed_targets": torch.randint(0, config.num_speed_buckets, (4, 6)),
            "role_targets": torch.randint(0, config.num_role_archetypes, (4, 6)),
            "move_family_targets": torch.randint(0, 2, (4, 6, config.num_move_families)),
        }
        loss, _ = compute_total_loss(
            output, batch_data["action"], batch_data["legal_mask"],
            aux_targets=aux_targets, game_result=batch_data["game_result"],
            config=config,
        )
        loss.backward()

        # Check gradients exist for key components
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ── Loss function tests ─────────────────────────────────────────────────


class TestLossFunctions:

    def test_policy_loss(self):
        logits = torch.randn(4, NUM_ACTIONS)
        targets = torch.randint(0, NUM_ACTIONS, (4,))
        mask = torch.ones(4, NUM_ACTIONS)
        loss = compute_policy_loss(logits, targets, mask)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_policy_loss_ignores_padding(self):
        logits = torch.randn(4, NUM_ACTIONS)
        targets = torch.tensor([0, 1, -1, -1])  # Last two are padding
        mask = torch.ones(4, NUM_ACTIONS)
        loss = compute_policy_loss(logits, targets, mask, ignore_index=-1)
        assert torch.isfinite(loss)

    def test_auxiliary_loss(self, config):
        batch = 4
        preds = {
            "item_logits": torch.randn(batch, 6, config.num_item_classes),
            "speed_logits": torch.randn(batch, 6, config.num_speed_buckets),
            "role_logits": torch.randn(batch, 6, config.num_role_archetypes),
            "move_family_logits": torch.randn(batch, 6, config.num_move_families),
        }
        targets = {
            "item_targets": torch.randint(0, config.num_item_classes, (batch, 6)),
            "speed_targets": torch.randint(0, config.num_speed_buckets, (batch, 6)),
            "role_targets": torch.randint(0, config.num_role_archetypes, (batch, 6)),
            "move_family_targets": torch.randint(0, 2, (batch, 6, config.num_move_families)),
        }
        loss, components = compute_auxiliary_loss(preds, targets)
        assert torch.isfinite(loss)
        assert "item" in components
        assert "speed" in components

    def test_auxiliary_loss_with_unknown_targets(self, config):
        batch = 4
        preds = {
            "item_logits": torch.randn(batch, 6, config.num_item_classes),
        }
        targets = {
            "item_targets": torch.full((batch, 6), -1, dtype=torch.long),
        }
        # All targets unknown -> loss should still be valid
        loss, components = compute_auxiliary_loss(preds, targets)
        assert torch.isfinite(loss)

    def test_value_loss(self):
        logits = torch.randn(4)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = compute_value_loss(logits, targets)
        assert torch.isfinite(loss)

    def test_value_loss_ignores_unknown(self):
        logits = torch.randn(4)
        targets = torch.tensor([1.0, 0.5, 0.5, 0.0])  # Middle two unknown
        loss = compute_value_loss(logits, targets)
        assert torch.isfinite(loss)

    def test_total_loss(self, model, batch_data, config):
        output = model(
            batch_data["own_team"],
            batch_data["opponent_team"],
            batch_data["field"],
            batch_data["context"],
            legal_mask=batch_data["legal_mask"],
        )

        aux_targets = {
            "item_targets": torch.randint(0, config.num_item_classes, (4, 6)),
            "speed_targets": torch.randint(0, config.num_speed_buckets, (4, 6)),
            "role_targets": torch.randint(0, config.num_role_archetypes, (4, 6)),
            "move_family_targets": torch.randint(0, 2, (4, 6, config.num_move_families)),
        }

        total_loss, loss_dict = compute_total_loss(
            output,
            batch_data["action"],
            batch_data["legal_mask"],
            aux_targets=aux_targets,
            game_result=batch_data["game_result"],
            config=config,
        )

        assert torch.isfinite(total_loss)
        assert "policy" in loss_dict
        assert "auxiliary" in loss_dict
        assert "value" in loss_dict
        assert "total" in loss_dict


# ── Auxiliary label tests ────────────────────────────────────────────────


class TestAuxiliaryLabels:

    def test_item_classification(self):
        from src.data.auxiliary_labels import classify_item, NUM_ITEM_CLASSES
        assert 0 <= classify_item("leftovers") < NUM_ITEM_CLASSES
        assert 0 <= classify_item("choiceband") < NUM_ITEM_CLASSES
        assert classify_item("") == classify_item("noitem")
        assert classify_item("unknownitem") == classify_item("noitem")

    def test_speed_classification(self):
        from src.data.auxiliary_labels import classify_speed
        # Gen 3 thresholds: [110, 90, 65, 40]
        assert classify_speed(130) == 0  # very fast (Jolteon)
        assert classify_speed(100) == 1  # fast (Salamence)
        assert classify_speed(80) == 2   # medium (Suicune)
        assert classify_speed(50) == 3   # slow (Blissey)
        assert classify_speed(30) == 4   # very slow (Snorlax)

    def test_move_families(self):
        from src.data.auxiliary_labels import classify_move_families, NUM_MOVE_FAMILIES
        # Gen 3 priority: ExtremeSpeed, Mach Punch, Quick Attack, Fake Out
        families = classify_move_families(["ExtremeSpeed", "Swords Dance", "Meteor Mash"])
        assert len(families) == NUM_MOVE_FAMILIES
        assert families[0] == 1  # priority (ExtremeSpeed)
        assert families[5] == 1  # setup (Swords Dance)

    def test_empty_move_families(self):
        from src.data.auxiliary_labels import classify_move_families
        families = classify_move_families([])
        assert all(f == 0 for f in families)


# ── Factory function tests ──────────────────────────────────────────────


class TestFactory:

    def test_create_from_config(self, config):
        model = create_battle_transformer(config=config)
        assert isinstance(model, BattleTransformer)
        assert model.config.hidden_dim == 64

    def test_create_with_kwargs(self):
        model = create_battle_transformer(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )
        assert model.config.hidden_dim == 128
        assert model.config.num_layers == 2
