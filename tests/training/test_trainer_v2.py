# tests/training/test_trainer_v2.py
import torch
import pytest
from torch_geometric.data import Data, Batch


def _toy_dataset(n_proteins=6, n_residues=15, n_edges=30):
    items = []
    for _ in range(n_proteins):
        items.append(Data(
            seq_idx=torch.randint(0, 20, (n_residues,)),
            x_scalar=torch.randn(n_residues, 27),
            x_vec=torch.randn(n_residues, 3, 3),
            edge_index=torch.randint(0, n_residues, (2, n_edges)),
            edge_scalar=torch.randn(n_edges, 16),
            edge_vec=torch.randn(n_edges, 1, 3),
        ))
    return items


V2_CONFIG = {
    "model": {
        "num_layers": 2, "node_scalar_in": 91, "node_vector_in": 3,
        "edge_scalar_in": 16, "edge_vector_in": 1,
        "hidden_scalar": 32, "hidden_vector": 4, "output_dim": 64,
        "dropout": 0.0, "num_walks": 4, "walk_length": 5,
        "walk_embed_dim": 64, "walk_layers": 2, "walk_heads": 4,
        "max_deg_cap": 30,
        "tx_layers": 2, "tx_d_model": 64, "tx_num_heads": 4,
        "tx_ff_mult": 2, "tx_dropout": 0.0, "attn_window_size": -1,
        "gradient_checkpointing": False,
    },
    "training": {
        "batch_size": 2, "max_epochs": 2, "lr": 1e-3,
        "weight_decay": 0.01, "mask_rate": 0.15, "grad_clip": 1.0,
        "checkpoint_dir": "/tmp/test_ckpt_v2", "log_every": 100,
        "num_workers": 0, "precision": "fp32",
        "gvp_lr_multiplier": 0.1, "warmup_phase_fraction": 0.5,
        "inverse_fold_prob": 0.5, "inverse_fold_weight": 1.0,
        "save_every_steps": 999999,
        "device_strategy": "single_gpu",
    },
}


def test_trainer_v2_two_epochs():
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset()
    trainer = Trainer(V2_CONFIG, dataset, device="cpu")
    trainer.fit()   # must complete without error


def test_trainer_v2_phase1_use_transformer_false():
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset()
    trainer = Trainer(V2_CONFIG, dataset, device="cpu")
    # Before fit(), still in Phase 1 setup
    assert trainer.encoder.use_transformer is False


def test_trainer_v2_phase2_use_transformer_true():
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset()
    # warmup_phase_fraction=0 -> Phase 2 immediately
    cfg = {**V2_CONFIG}
    cfg["training"] = {**V2_CONFIG["training"], "warmup_phase_fraction": 0.0}
    trainer = Trainer(cfg, dataset, device="cpu")
    trainer._transition_to_phase2()
    assert trainer.encoder.use_transformer is True


def test_trainer_v2_two_param_groups_in_phase2():
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset()
    cfg = {**V2_CONFIG}
    cfg["training"] = {**V2_CONFIG["training"], "warmup_phase_fraction": 0.0}
    trainer = Trainer(cfg, dataset, device="cpu")
    trainer._transition_to_phase2()
    assert len(trainer.optimizer.param_groups) == 2
    lr0 = trainer.optimizer.param_groups[0]["lr"]
    lr1 = trainer.optimizer.param_groups[1]["lr"]
    assert abs(lr0 / lr1 - 0.1) < 1e-6, "Group 0 must be 0.1x group 1 LR"


def test_trainer_v2_scheduler_decays_in_phase2():
    """LR must decrease in Phase 2 (scheduler must follow new optimizer)."""
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset(n_proteins=20)
    cfg = {**V2_CONFIG}
    cfg["training"] = {**V2_CONFIG["training"],
                       "warmup_phase_fraction": 0.0,
                       "max_epochs": 10,
                       "inverse_fold_prob": 0.0}
    trainer = Trainer(cfg, dataset, device="cpu")
    trainer._transition_to_phase2()
    lr_before = trainer.optimizer.param_groups[-1]["lr"]
    # Run several scheduler steps
    for _ in range(20):
        trainer.scheduler.step()
    lr_after = trainer.optimizer.param_groups[-1]["lr"]
    assert lr_after < lr_before, (
        f"LR did not decay after phase transition: {lr_before} -> {lr_after}. "
        "Scheduler may be orphaned."
    )


def test_trainer_v2_loss_decreases():
    from prostrencoder.training.trainer import Trainer
    torch.manual_seed(42)
    # Use 20 proteins (40 total steps) so the cosine LR schedule doesn't
    # collapse to near-zero before the final epoch on this toy dataset.
    dataset = _toy_dataset(n_proteins=20)
    cfg = {**V2_CONFIG}
    cfg["training"] = {**V2_CONFIG["training"], "max_epochs": 4,
                       "warmup_phase_fraction": 0.0, "inverse_fold_prob": 0.0}
    trainer = Trainer(cfg, dataset, device="cpu")
    losses = []
    for epoch in range(1, 5):
        losses.append(trainer.train_epoch(epoch))
    # Loss should trend downward over 4 epochs on tiny dataset
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
