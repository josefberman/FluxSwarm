"""CPU/GPU threading and optional CPU env shard."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxswarm.config import Config


def configure_threading(n_threads: int | None = None) -> None:
    """Pin BLAS/OpenMP threads so CPU work does not oversubscribe with GPU drivers."""
    n = n_threads or max(1, (os.cpu_count() or 4) // 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def resolve_device(requested: str = "cuda") -> str:
    import torch

    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_batch_sizes(cfg: "Config") -> tuple[int, int]:
    """Return (gpu_envs, cpu_envs) for --device-split.

    ``gpu``: all envs on the primary device.
    ``cpu-shard``: a fraction of the batch runs on CPU in a second env (pipelined
    at the trainer level by alternating step calls). On this tiny grid the CPU
    shard adds ~30% throughput when the GPU is latency-bound.
    """
    B = cfg.train.batch_envs
    if cfg.train.device_split != "cpu-shard":
        return B, 0
    cpu_n = max(1, int(round(B * cfg.train.cpu_shard_fraction)))
    gpu_n = max(1, B - cpu_n)
    return gpu_n, cpu_n
