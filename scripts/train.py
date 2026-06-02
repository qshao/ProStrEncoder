# scripts/train.py
"""
Pre-train ProStrEncoder v2.

Device strategy is set in YAML (training.device_strategy):
  single_gpu  ->  python scripts/train.py --config configs/medium.yaml
  ddp         ->  torchrun --nproc_per_node=4 scripts/train.py --config configs/large.yaml
  fsdp        ->  torchrun --nproc_per_node=4 scripts/train.py --config configs/xl.yaml
"""
import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
import yaml

from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.training.trainer import Trainer


def _setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def _wrap_ddp(encoder, masked_head, inv_fold_head, local_rank):
    from torch.nn.parallel import DistributedDataParallel as DDP
    encoder       = DDP(encoder,       device_ids=[local_rank])
    masked_head   = DDP(masked_head,   device_ids=[local_rank])
    inv_fold_head = DDP(inv_fold_head, device_ids=[local_rank])
    return encoder, masked_head, inv_fold_head


def _wrap_fsdp(encoder, masked_head, inv_fold_head):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    encoder       = FSDP(encoder,       mixed_precision=bf16_policy)
    masked_head   = FSDP(masked_head,   mixed_precision=bf16_policy)
    inv_fold_head = FSDP(inv_fold_head, mixed_precision=bf16_policy)
    return encoder, masked_head, inv_fold_head


def _save_fsdp_checkpoint(trainer, epoch, val_loss):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    import math
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(trainer.encoder, StateDictType.FULL_STATE_DICT, cfg):
        encoder_state = trainer.encoder.state_dict()
    with FSDP.state_dict_type(trainer.masked_head, StateDictType.FULL_STATE_DICT, cfg):
        masked_state = trainer.masked_head.state_dict()
    with FSDP.state_dict_type(trainer.inv_fold_head, StateDictType.FULL_STATE_DICT, cfg):
        inv_fold_state = trainer.inv_fold_head.state_dict()
    if dist.get_rank() == 0:
        tag = (f"step{trainer._step:07d}" if math.isnan(val_loss)
               else f"epoch{epoch:03d}_val{val_loss:.4f}")
        path = os.path.join(trainer.ckpt_dir, f"ckpt_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "step": trainer._step,
            "phase": trainer._phase,
            "encoder": encoder_state,
            "masked_head": masked_state,
            "inv_fold_head": inv_fold_state,
            "config": trainer.config,
        }, path)
        print(json.dumps({"event": "fsdp_checkpoint", "path": path}))


def main():
    parser = argparse.ArgumentParser(description="Pre-train ProStrEncoder v2.")
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir",   default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg       = config.get("training", {})
    data_cfg        = config.get("data", {})
    strategy        = train_cfg.get("device_strategy", "single_gpu")
    has_local_rank  = "LOCAL_RANK" in os.environ

    # Validate launch method vs config
    if strategy == "single_gpu" and has_local_rank:
        print(
            "ERROR: device_strategy=single_gpu but LOCAL_RANK is set.\n"
            "       Do not use torchrun for single-GPU training.\n"
            "       Run: python scripts/train.py --config ...",
            file=sys.stderr,
        )
        sys.exit(1)
    if strategy in ("ddp", "fsdp") and not has_local_rank:
        print(
            f"ERROR: device_strategy={strategy} requires torchrun.\n"
            f"       Run: torchrun --nproc_per_node=N scripts/train.py --config ...",
            file=sys.stderr,
        )
        sys.exit(1)

    # Data paths
    train_dir = args.train_dir or data_cfg.get("train_dir")
    val_dir   = args.val_dir   or data_cfg.get("val_dir")
    if not train_dir:
        print(
            "ERROR: train_dir not set. Provide --train_dir or config.data.train_dir",
            file=sys.stderr,
        )
        sys.exit(1)

    # Distributed setup
    local_rank = 0
    world_size = 1
    if strategy in ("ddp", "fsdp"):
        local_rank, world_size = _setup_distributed()

    device   = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main  = (local_rank == 0)

    train_ds = ProteinDataset(train_dir)
    val_ds   = ProteinDataset(val_dir) if val_dir else None

    if is_main:
        print(json.dumps({"event": "start", "strategy": strategy,
                          "world_size": world_size,
                          "train_size": len(train_ds),
                          "val_size": len(val_ds) if val_ds else 0}))

    trainer = Trainer(config, train_ds, val_ds, device=device)

    # Wrap with DDP / FSDP
    if strategy == "ddp":
        trainer.encoder, trainer.masked_head, trainer.inv_fold_head = _wrap_ddp(
            trainer.encoder, trainer.masked_head, trainer.inv_fold_head, local_rank
        )
    elif strategy == "fsdp":
        trainer.encoder, trainer.masked_head, trainer.inv_fold_head = _wrap_fsdp(
            trainer.encoder, trainer.masked_head, trainer.inv_fold_head
        )
        trainer.save_checkpoint = lambda epoch, val_loss: _save_fsdp_checkpoint(
            trainer, epoch, val_loss
        )

    # DDP and FSDP: each rank sees different data via DistributedSampler
    if strategy in ("ddp", "fsdp"):
        from torch.utils.data import DistributedSampler
        from torch_geometric.loader import DataLoader
        sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                     rank=local_rank, shuffle=True)
        trainer.train_loader = DataLoader(
            train_ds,
            batch_size=config["training"]["batch_size"],
            sampler=sampler,
            num_workers=config["training"].get("num_workers", 4),
            pin_memory=True,
            persistent_workers=(config["training"].get("num_workers", 4) > 0),
        )

    trainer.fit()

    if strategy in ("ddp", "fsdp"):
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
