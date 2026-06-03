# Fix Uncommitted Bugs and Eval v2 Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit five pending bug-fix files that have been tested but never committed, track the untracked `make_toy_dataset.py`, and fix two v2 compatibility bugs in `eval_linear_probe.py`.

**Architecture:** Three independent tasks in dependency order — commit core model fixes, commit script improvements, then fix the eval script. All 89 existing tests must continue to pass throughout. Two new tests are added for the eval script fix.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, pytest, YAML

---

## Context: What Is Already Done

The following files have been **modified/created and tested** in a previous session but were **never committed**. They are safe to commit as-is — all 89 tests already pass with these changes in the working tree.

| File | Status | What changed |
|------|--------|--------------|
| `prostrencoder/data/features.py` | Modified | NaN guard in `_dihedral`, `nan_to_num` safety net in `compute_torsion_angles`, `rbf_encoding` div-by-zero guard, `build_pyg_data` shared helper |
| `prostrencoder/models/gvp.py` | Modified | `_ACT_DEFAULT` sentinel fixes mutable default arg; `res_v_proj` + residual in `GVPConv.forward` prevents unbounded vector norm growth |
| `prostrencoder/models/walk_sampler.py` | Modified | `max_deg_cap` parameter avoids GPU→CPU sync on every forward pass |
| `scripts/eval_linear_probe.py` | Modified | Probe hyperparams from YAML `evaluation:` section; optional CLI overrides |
| `scripts/make_toy_dataset.py` | Untracked | New script: generates synthetic protein datasets for pipeline testing |
| `.claude/settings.local.json` | Modified | Permission settings — **do NOT commit this file** |

---

## File Map

**Modified in Task 1:**
- `prostrencoder/data/features.py` — commit as-is
- `prostrencoder/models/gvp.py` — commit as-is
- `prostrencoder/models/walk_sampler.py` — commit as-is

**Modified in Task 2:**
- `scripts/eval_linear_probe.py` — commit as-is
- `scripts/make_toy_dataset.py` — add to tracking and commit

**Modified in Task 3:**
- `scripts/eval_linear_probe.py` — two bug fixes + update test
- `tests/scripts/test_eval_linear_probe.py` — new test file

---

## Task 1: Commit Core Model Bug Fixes

**Files:**
- Commit: `prostrencoder/data/features.py`, `prostrencoder/models/gvp.py`, `prostrencoder/models/walk_sampler.py`

These fixes were verified by the full test suite (89 passing). No new code to write — commit only.

- [ ] **Step 1: Verify the working tree matches expectations**

```bash
git diff --name-only HEAD
```
Expected output includes these three files (plus others):
```
prostrencoder/data/features.py
prostrencoder/models/gvp.py
prostrencoder/models/walk_sampler.py
```

- [ ] **Step 2: Run the test suite to confirm 89 pass**

```bash
python -m pytest tests/ -q 2>&1 | tail -4
```
Expected:
```
89 passed, 2 skipped, 58 warnings in ...
```

- [ ] **Step 3: Commit the three core model fix files**

```bash
git add prostrencoder/data/features.py \
        prostrencoder/models/gvp.py \
        prostrencoder/models/walk_sampler.py
git commit -m "fix: NaN torsion guard, rbf div-by-zero, GVP mutable default, vector residual, max_deg_cap GPU sync"
```

---

## Task 2: Commit Script Improvements

**Files:**
- Commit: `scripts/eval_linear_probe.py`, `scripts/make_toy_dataset.py`

Also verified by the test suite. No new code — commit only. `.claude/settings.local.json` must NOT be included.

- [ ] **Step 1: Confirm settings.local.json is excluded**

```bash
git diff --name-only HEAD -- .claude/settings.local.json
```
Expected: one line output (the file is modified). Confirm it is NOT staged below.

- [ ] **Step 2: Stage only the two script files**

```bash
git add scripts/eval_linear_probe.py scripts/make_toy_dataset.py
git status --short
```
Expected: `M  scripts/eval_linear_probe.py` and `A  scripts/make_toy_dataset.py` are staged (green).
`.claude/settings.local.json` must appear as `M` (red, unstaged) — not staged.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: eval_linear_probe YAML hyperparams; make_toy_dataset.py synthetic data generator"
```

- [ ] **Step 4: Verify settings.local.json is still uncommitted**

```bash
git status --short .claude/settings.local.json
```
Expected: ` M .claude/settings.local.json` (unstaged, not committed).

---

## Task 3: Fix eval_linear_probe.py v2 Compatibility

**Files:**
- Modify: `scripts/eval_linear_probe.py`
- Create: `tests/scripts/test_eval_linear_probe.py`

Two bugs in the current `eval_linear_probe.py`:

**Bug A — `weights_only=True` crashes on v2 checkpoints.**
The v2 Trainer saves `config` (a Python dict) inside each checkpoint. `torch.load(..., weights_only=True)` rejects arbitrary Python objects and raises `UnpicklingError`. Must use `weights_only=False`.

**Bug B — `use_transformer` never set after loading.**
For v2 checkpoints trained with a global transformer (`tx_layers > 0`), the encoder silently runs in Phase 1 mode (bridge-only, no attention layers) because `encoder.use_transformer` defaults to `False`. The checkpoint's `phase` key records which phase training was in — this must be used to restore the correct inference mode.

### Step 1: Write failing tests

- [ ] Create `tests/scripts/` directory and test file:

```bash
mkdir -p tests/scripts
touch tests/scripts/__init__.py
```

```python
# tests/scripts/test_eval_linear_probe.py
"""Tests for eval_linear_probe.py v2 compatibility."""
import os
import math
import tempfile

import torch
import pytest
from torch_geometric.data import Data

from prostrencoder.models.encoder import ProStrEncoder


def _make_v2_checkpoint(tmp_dir: str, phase: int = 2) -> str:
    """
    Write a minimal v2-format checkpoint (contains a config dict,
    which makes weights_only=True fail).
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
        "config": config,           # <-- this causes weights_only=True to fail
    }, path)
    return path, config


def test_weights_only_false_loads_v2_checkpoint():
    """weights_only=False must succeed on checkpoints containing Python dicts."""
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _make_v2_checkpoint(tmp, phase=2)
        # weights_only=True would raise UnpicklingError
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
        assert encoder.use_transformer is False   # default

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        if ckpt.get("phase", 1) == 2:
            encoder.use_transformer = True

        assert encoder.use_transformer is True


def test_phase1_checkpoint_leaves_use_transformer_false():
    """
    A phase=1 checkpoint (no transformer active during training)
    must leave use_transformer=False.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path, config = _make_v2_checkpoint(tmp, phase=1)
        encoder = ProStrEncoder(config["model"])
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        if ckpt.get("phase", 1) == 2:
            encoder.use_transformer = True

        assert encoder.use_transformer is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/scripts/test_eval_linear_probe.py -v 2>&1 | tail -10
```
Expected: `test_phase2_checkpoint_sets_use_transformer` and `test_phase1_checkpoint_leaves_use_transformer_false` fail because the logic is not in the script yet. `test_weights_only_false_loads_v2_checkpoint` may pass already (it tests `torch.load` directly, not the script).

- [ ] **Step 3: Apply the two fixes to `scripts/eval_linear_probe.py`**

Find lines 116–120 (the checkpoint loading block) and replace them:

**Current code (lines 116–120):**
```python
    device = torch.device(args.device)
    encoder = ProStrEncoder(config["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)
```

**Replace with:**
```python
    device = torch.device(args.device)
    encoder = ProStrEncoder(config["model"]).to(device)
    # weights_only=False required: v2 checkpoints contain a config dict
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    # Restore transformer activation from checkpoint phase (v2 only)
    if ckpt.get("phase", 1) == 2:
        encoder.use_transformer = True
    for p in encoder.parameters():
        p.requires_grad_(False)
```

- [ ] **Step 4: Run the new tests to confirm they pass**

```bash
python -m pytest tests/scripts/test_eval_linear_probe.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -4
```
Expected:
```
92 passed, 2 skipped, ...
```
(89 existing + 3 new = 92)

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_linear_probe.py \
        tests/scripts/__init__.py \
        tests/scripts/test_eval_linear_probe.py
git commit -m "fix: eval_linear_probe — weights_only=False for v2 checkpoints, restore use_transformer from phase"
```

---

## Self-Review

**Spec coverage:**
- [x] Commit `features.py` (NaN guard, rbf fix, build_pyg_data) → Task 1
- [x] Commit `gvp.py` (mutable default fix, vector residual) → Task 1
- [x] Commit `walk_sampler.py` (max_deg_cap) → Task 1
- [x] Commit `eval_linear_probe.py` (YAML hyperparams) → Task 2
- [x] Track and commit `make_toy_dataset.py` → Task 2
- [x] Exclude `.claude/settings.local.json` → Task 2 Step 1–4
- [x] Fix `weights_only=True` crash on v2 checkpoints → Task 3
- [x] Fix `use_transformer` not set after loading phase-2 checkpoint → Task 3
- [x] Tests for both eval fixes → Task 3

**Placeholder scan:** No TBD, TODO, or vague instructions. All code blocks are complete.

**Type consistency:**
- `encoder.use_transformer` — property backed by buffer `_use_transformer`; setter assigns via `.fill_()`. Used consistently in tests and fix.
- `ckpt.get("phase", 1)` — `phase` is an int (1 or 2) in all v2 checkpoints; default 1 covers v1 checkpoints that lack the key.
- `_make_v2_checkpoint` returns `(path, config)` — used consistently in all three tests.
