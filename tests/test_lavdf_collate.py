"""Regression tests for the LAV-DF collate function's multiprocessing safety.

Windows' default ``multiprocessing`` start method is ``spawn``, which pickles
the target callable (including ``collate_fn``) to hand off to worker
processes. A collate function defined as a local closure inside another
function (e.g. inside ``main()`` in a training script) is not picklable under
``spawn`` and fails with::

    AttributeError: Can't pickle local object 'main.<locals>.collate_fn'

``lavdf_collate_fn`` must stay a module-level function in
``src.data.lavdf_dataset`` so it can be pickled and shipped to DataLoader
worker processes regardless of platform.
"""

from __future__ import annotations

import pickle

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.lavdf_dataset import lavdf_collate_fn


def _make_item(mel_len: int, n_mels: int = 8, n_landmarks: int = 4) -> dict:
    return {
        "mel_window": torch.arange(n_mels * mel_len, dtype=torch.float32).reshape(n_mels, mel_len),
        "landmarks": torch.zeros(n_landmarks, 3),
        "fps": 25.0,
        "window_seconds": 1.28,
        "shift_seconds": 0.0,
        "is_positive": True,
    }


class _VariableLengthMelDataset(Dataset):
    """Minimal module-level dataset producing variable-length mel windows.

    Defined at module scope (not inside a test function) so that it, like
    ``lavdf_collate_fn``, is picklable under Windows' ``spawn`` start method.
    """

    def __init__(self, mel_lengths: list[int]) -> None:
        self.mel_lengths = mel_lengths

    def __len__(self) -> int:
        return len(self.mel_lengths)

    def __getitem__(self, index: int) -> dict:
        return _make_item(self.mel_lengths[index])


def test_collate_fn_is_module_level_and_picklable():
    """The bug: a local collate_fn cannot be pickled. Guard the fix directly."""
    assert lavdf_collate_fn.__qualname__ == "lavdf_collate_fn"
    # This is exactly what a multiprocessing spawn worker does when the
    # DataLoader ships collate_fn to it; a local closure raises here.
    pickled = pickle.dumps(lavdf_collate_fn)
    restored = pickle.loads(pickled)
    assert restored is lavdf_collate_fn


def test_collate_fn_pads_variable_length_mel_windows():
    batch = [_make_item(5), _make_item(3), _make_item(5)]
    out = lavdf_collate_fn(batch)

    assert out["mel_window"].shape == (3, 8, 5)
    # Shorter window (index 1) must be zero-padded on the right, original
    # content preserved for the unpadded region.
    assert torch.equal(out["mel_window"][1, :, :3], batch[1]["mel_window"])
    assert torch.equal(out["mel_window"][1, :, 3:], torch.zeros(8, 2))

    assert out["landmarks"].shape == (3, 4, 3)
    assert out["fps"] == [25.0, 25.0, 25.0]
    assert out["window_seconds"] == [1.28, 1.28, 1.28]
    assert out["shift_seconds"] == [0.0, 0.0, 0.0]
    assert out["is_positive"] == [True, True, True]


def test_collate_fn_usable_with_spawn_multiprocessing_dataloader():
    """End-to-end regression: a DataLoader with workers under 'spawn' must
    not raise when pickling collate_fn (or the dataset) to worker processes.

    This mirrors the Windows default start method regardless of the host OS
    running the test, so the regression is caught in CI even off-Windows.
    """
    dataset = _VariableLengthMelDataset([5, 3, 5, 4])
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=lavdf_collate_fn,
        multiprocessing_context="spawn",
    )

    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert batch["mel_window"].shape[0] == 2
        assert batch["landmarks"].shape == (2, 4, 3)
