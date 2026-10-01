"""Explicit checkpoint-compatibility validation for production inference
(Phase 2 of the production AV-inference correctness audit).

`nn.Module.load_state_dict(..., strict=False)` collapses two very different
situations into one silent outcome: (a) a handful of keys that differ only
because of a known, understood wrapper/prefix naming difference, and (b) a
genuinely incompatible checkpoint that leaves some submodule partially at its
random initialization. `strict=False` alone cannot tell these apart, and (b)
must never happen silently in a production inference path.

`load_state_dict_validated` still uses `strict=False` internally - so a
checkpoint whose ONLY difference is an explicitly documented, allow-listed key
name can still load - but then checks the resulting `missing_keys` /
`unexpected_keys` against that explicit allow-list and raises
`CheckpointCompatibilityError` for anything outside it, instead of silently
running with an incompletely-loaded model.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch
from torch import nn

__all__ = ["CheckpointCompatibilityError", "load_state_dict_validated"]


class CheckpointCompatibilityError(RuntimeError):
    """A checkpoint's keys/shapes don't match the target module beyond an
    explicitly allowed, documented difference. Raised instead of allowing a
    partially-loaded model to run inference."""


def load_state_dict_validated(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    context: str,
    allowed_missing: Iterable[str] = (),
    allowed_unexpected_prefixes: Iterable[str] = (),
) -> None:
    """Load ``state_dict`` into ``module``, raising ``CheckpointCompatibilityError``
    for any missing/unexpected key not explicitly allowed for.

    Args:
        module: the target module to load into.
        state_dict: the (already key-normalized, e.g. prefix-stripped) source
            state dict.
        context: human-readable description of what's being loaded, included
            in the error message (e.g. "cross_attention from sync_model_path").
        allowed_missing: exact target-module parameter names allowed to be
            absent from ``state_dict``. Should normally be empty - a real
            checkpoint for this submodule is expected to cover every
            parameter; a non-empty value here documents a specific, known
            exception, not a general escape hatch.
        allowed_unexpected_prefixes: checkpoint key prefixes allowed to be
            present in ``state_dict`` but absent from ``module`` (e.g. a
            retired adapter block from an older checkpoint schema).

    Note: shape mismatches for keys present in both are NOT affected by
    ``strict`` - PyTorch always raises on those regardless, so this function
    does not need to check shapes separately.
    """

    result = module.load_state_dict(state_dict, strict=False)
    allowed_missing_set = set(allowed_missing)
    missing = [k for k in result.missing_keys if k not in allowed_missing_set]
    unexpected_prefixes = tuple(allowed_unexpected_prefixes)
    unexpected = [k for k in result.unexpected_keys if not k.startswith(unexpected_prefixes)]

    if missing or unexpected:
        raise CheckpointCompatibilityError(
            f"{context}: checkpoint is architecturally incompatible with "
            f"{type(module).__name__} - refusing to run inference with a "
            f"partially loaded model. missing_keys={missing!r} "
            f"unexpected_keys={unexpected!r}"
        )
