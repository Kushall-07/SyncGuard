"""Checkpoint save / load helpers (Phase 2B).

A checkpoint is a plain ``dict`` written with :func:`torch.save` containing the
model weights plus enough state to resume: optimizer, scheduler, epoch, the
monitored metric value, the RNG states, and an arbitrary ``extra`` payload
(e.g. the resolved config).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    extra: dict[str, Any] | None = None,
    include_rng_state: bool = True,
) -> Path:
    """Write a resumable checkpoint. Returns the path written."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    if include_rng_state:
        payload["rng_state"] = {
            "python": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng_state: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore module/optimizer/scheduler state.

    Returns the raw checkpoint dict so callers can read ``epoch`` / ``best_metric``
    / ``extra``.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and ckpt.get("model") is not None:
        model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if restore_rng_state and ckpt.get("rng_state"):
        rng = ckpt["rng_state"]
        if rng.get("python") is not None:
            np.random.set_state(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(_as_byte_tensor(rng["torch"]))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in rng["cuda"]])

    return ckpt


def _as_byte_tensor(state: Any) -> torch.Tensor:
    """RNG states round-trip through torch.save as tensors already; be defensive."""

    if isinstance(state, torch.Tensor):
        return state.to(torch.uint8).cpu()
    return torch.as_tensor(state, dtype=torch.uint8)
