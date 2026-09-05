"""Training infrastructure helpers (Phase 2B).

Reproducibility (:func:`set_seed`), device selection (:func:`get_device`),
parameter counting, and per-run output directories with logging
(:class:`RunDirectory`). These are shared by every training phase from Phase 4
onwards via :class:`src.training.trainer.Trainer`.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

__all__ = [
    "set_seed",
    "get_device",
    "count_parameters",
    "config_to_dict",
    "RunDirectory",
]


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    With ``deterministic=True`` PyTorch is also asked to use deterministic
    algorithms and cuDNN autotuning is disabled - reproducible but slower, and it
    raises if an op has no deterministic implementation.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or the requested one if given."""

    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = True) -> int:
    """Total number of (trainable) parameters in ``module``."""

    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def config_to_dict(config: Any) -> Any:
    """Recursively convert a (possibly nested) dataclass config to plain dict/JSON."""

    if is_dataclass(config) and not isinstance(config, type):
        return {k: config_to_dict(v) for k, v in asdict(config).items()}
    if isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [config_to_dict(v) for v in config]
    return config


class RunDirectory:
    """A timestamped output directory for one experiment run.

    Layout::

        <output_root>/<name>-<YYYYmmdd-HHMMSS>/
            config.yaml         # resolved config snapshot
            metrics.csv         # one row per logged step (written by Trainer)
            run.log             # mirror of the console log
            checkpoints/        # best.pt, last.pt, ...
    """

    def __init__(
        self,
        output_root: str | Path,
        name: str,
        *,
        timestamp: str | None = None,
        create: bool = True,
    ) -> None:
        stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(output_root) / f"{name}-{stamp}"
        self.checkpoints = self.path / "checkpoints"
        if create:
            self.checkpoints.mkdir(parents=True, exist_ok=True)

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.csv"

    @property
    def log_path(self) -> Path:
        return self.path / "run.log"

    def checkpoint_path(self, tag: str = "last") -> Path:
        return self.checkpoints / f"{tag}.pt"

    def save_config(self, config: Any) -> Path:
        payload = config_to_dict(config)
        with self.config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
        return self.config_path

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.path / name
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return target

    def get_logger(self, logger_name: str = "syncguard") -> logging.Logger:
        """A logger that writes to both the console and ``run.log``."""

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        already = {getattr(h, "_syncguard_tag", None) for h in logger.handlers}
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

        if "console" not in already:
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            console._syncguard_tag = "console"  # type: ignore[attr-defined]
            logger.addHandler(console)

        file_tag = f"file:{self.log_path}"
        if file_tag not in already:
            file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
            file_handler.setFormatter(fmt)
            file_handler._syncguard_tag = file_tag  # type: ignore[attr-defined]
            logger.addHandler(file_handler)

        return logger
