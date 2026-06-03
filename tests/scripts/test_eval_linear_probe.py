"""Tests for eval_linear_probe.py v2 compatibility."""
import os
import tempfile

import torch
import pytest

from prostrencoder.models.encoder import ProStrEncoder


def _make_v2_checkpoint(tmp_dir: str, phase: int = 2) -> tuple:
    """
    Write a minimal v2-format checkpoint (contains a config dict,
    which makes weights_only=True fail).
    Returns (path, config).
    """
    config = {
        "model": {
            "num_layers": 1, "node_scalar_in": 91, "node_vector_in": 3,
            "edge_scalar_in": 16, "edge_vector_in": 1,
            "hidden_scalar": 32, "hidden_vector": 4, "output_dim": 64,
            "dropout": 0.0, "num_walks": 4, "walk_length": 5,
            "walk_embed_dim": 64, "walk_layers": 2, "walk_heads": 4,
            "max_deg_cap": 30,
            "tx_layers": 2, "tx_d_model": 64, "tx_num_heads": 4,
            "tx_ff_mult": 2, "tx_dropout": 0.0, "attn_window_size": -1,
            "gradient_checkpointing": False,
        }
    }
    from prostrencoder.models.gnn import ResidueHead
    encoder = ProStrEncoder(config["model"])
    hidden_dim = encoder.hidden_dim
    masked_head = ResidueHead(hidden_dim)
    inv_fold_head = ResidueHead(hidden_dim)

    path = os.path.join(tmp_dir, f"ckpt_phase{phase}.pt")
    torch.save({
        "epoch": 1,
        "step": 100,
        "phase": phase,
        "encoder": encoder.state_dict(),
        "masked_head": masked_head.state_dict(),
        "inv_fold_head": inv_fold_head.state_dict(),
        "optimizer": {},
        "config": config,  # <-- causes weights_only=True to fail
    }, path)
    return path, config


def test_weights_only_false_loads_v2_checkpoint():
    """weights_only=False must succeed on checkpoints containing Python dicts."""
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _make_v2_checkpoint(tmp, phase=2)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "encoder" in ckpt
        assert "config" in ckpt
        assert isinstance(ckpt["config"], dict)


def test_phase2_checkpoint_sets_use_transformer():
    """
    When loading a phase=2 checkpoint, encoder.use_transformer must be True
    so the full transformer runs during inference.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path, config = _make_v2_checkpoint(tmp, phase=2)
        encoder = ProStrEncoder(config["model"])
        assert encoder.use_transformer is False  # default

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        if ckpt.get("phase", 1) == 2:
            encoder.use_transformer = True

        assert encoder.use_transformer is True


def test_phase1_checkpoint_leaves_use_transformer_false():
    """
    A phase=1 checkpoint must leave use_transformer=False.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path, config = _make_v2_checkpoint(tmp, phase=1)
        encoder = ProStrEncoder(config["model"])

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        if ckpt.get("phase", 1) == 2:
            encoder.use_transformer = True

        assert encoder.use_transformer is False
