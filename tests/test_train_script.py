# tests/test_train_script.py
import os
import subprocess
import sys
import pytest


def _run(args, env=None):
    result = subprocess.run(
        [sys.executable, "scripts/train.py"] + args,
        capture_output=True, text=True, env=env,
        cwd="/home/qshao/ProStrEncoder",
    )
    return result


def test_missing_train_dir_raises():
    """Script must error clearly when train_dir is not set."""
    import tempfile, yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        cfg = yaml.safe_load(open("configs/default.yaml"))
        # Remove train_dir from config so the script has no path to use
        cfg.get("data", {}).pop("train_dir", None)
        yaml.dump(cfg, f)
        tmp_cfg = f.name
    r = _run(["--config", tmp_cfg])
    os.unlink(tmp_cfg)
    assert r.returncode != 0
    assert "train_dir" in r.stderr or "train_dir" in r.stdout


def test_single_gpu_with_local_rank_raises():
    """device_strategy=single_gpu + LOCAL_RANK set must raise clear error."""
    env = {**os.environ, "LOCAL_RANK": "0"}
    r = _run(["--config", "configs/default.yaml",
              "--train_dir", "data/toy/train"], env=env)
    assert r.returncode != 0
    err = r.stderr.lower() + r.stdout.lower()
    assert "torchrun" in err or "single_gpu" in err


def test_ddp_without_local_rank_raises():
    """device_strategy=ddp without LOCAL_RANK must raise clear error."""
    import tempfile, yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        cfg = yaml.safe_load(open("configs/default.yaml"))
        cfg["training"]["device_strategy"] = "ddp"
        yaml.dump(cfg, f)
        tmp_cfg = f.name
    env = {k: v for k, v in os.environ.items() if k != "LOCAL_RANK"}
    r = _run(["--config", tmp_cfg, "--train_dir", "data/toy/train"], env=env)
    os.unlink(tmp_cfg)
    assert r.returncode != 0
    err = r.stderr.lower() + r.stdout.lower()
    assert "torchrun" in err or "local_rank" in err.lower()
