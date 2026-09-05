"""Manifest-backed audio ``Dataset`` (Phase 3).

:class:`ManifestAudioDataset` turns any :class:`~src.data.manifests.Manifest`
into a PyTorch dataset yielding ``(feature, label)`` pairs. It reuses the Phase 2A
preprocessing (:func:`~src.preprocessing.audio.preprocess_audio`) and mel feature
extractor, so ASVspoof ``.flac`` and the synthetic WAVs flow through identical
code.

Every item is cropped or zero-padded to ``fixed_seconds`` so a plain default
collate produces rectangular batches. Training datasets should pass
``random_crop=True`` (random offset); evaluation datasets keep the default
center crop for determinism.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import AudioConfig, TrainingConfig
from src.data.manifests import Manifest, ManifestRow
from src.features.mel_spectrogram import MelSpectrogramExtractor
from src.preprocessing.audio import preprocess_audio

__all__ = ["ManifestAudioDataset", "pad_or_crop", "build_dataloader"]

Feature = Literal["waveform", "logmel"]


def pad_or_crop(
    waveform: torch.Tensor,
    target_len: int,
    *,
    random_crop: bool = False,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Force ``waveform`` ``[C, N]`` to ``[C, target_len]`` by cropping or right-padding."""

    n = waveform.shape[-1]
    if n == target_len:
        return waveform
    if n > target_len:
        if random_crop:
            start = (rng or random).randint(0, n - target_len)
        else:
            start = (n - target_len) // 2
        return waveform[..., start : start + target_len]
    pad = waveform.new_zeros(*waveform.shape[:-1], target_len - n)
    return torch.cat([waveform, pad], dim=-1)


class ManifestAudioDataset(Dataset):
    def __init__(
        self,
        manifest: Manifest,
        audio_cfg: AudioConfig,
        *,
        feature: Feature = "logmel",
        fixed_seconds: float = 4.0,
        random_crop: bool = False,
        seed: int = 0,
    ) -> None:
        if len(manifest) == 0:
            raise ValueError("manifest is empty")
        if feature not in ("waveform", "logmel"):
            raise ValueError("feature must be 'waveform' or 'logmel'")
        if fixed_seconds <= 0:
            raise ValueError("fixed_seconds must be positive")

        self.manifest = manifest
        self.audio_cfg = audio_cfg
        self.feature = feature
        self.random_crop = random_crop
        self.target_len = int(round(fixed_seconds * audio_cfg.sample_rate))
        self._base_seed = seed
        # One extractor, kept on CPU; cloned state is unnecessary (no learnable params).
        self._mel = MelSpectrogramExtractor(audio_cfg) if feature == "logmel" else None
        if self._mel is not None:
            self._mel.eval()

    def __len__(self) -> int:
        return len(self.manifest)

    def row(self, index: int) -> ManifestRow:
        return self.manifest[index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.manifest[index]
        if not Path(row.path).is_file():
            raise FileNotFoundError(f"sample {row.sample_id!r}: audio file missing at {row.path}")

        waveform = preprocess_audio(row.path, self.audio_cfg)  # [1, N], 16 kHz, normalized
        rng = random.Random(self._base_seed * 1_000_003 + index) if self.random_crop else None
        waveform = pad_or_crop(waveform, self.target_len, random_crop=self.random_crop, rng=rng)

        if self.feature == "waveform":
            feat = waveform
        else:
            with torch.no_grad():
                feat = self._mel(waveform)  # [n_mels, T]

        return feat.contiguous(), int(row.label)

    def label_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights ``[w_spoof, w_bonafide]`` for a weighted loss."""

        counts = torch.zeros(2)
        for r in self.manifest:
            counts[r.label] += 1
        counts = counts.clamp(min=1.0)
        weights = counts.sum() / (2.0 * counts)
        return weights


def build_dataloader(
    dataset: ManifestAudioDataset,
    training_cfg: TrainingConfig,
    *,
    shuffle: bool,
    drop_last: bool | None = None,
) -> DataLoader:
    """A :class:`DataLoader` configured from :class:`~src.config.TrainingConfig`."""

    return DataLoader(
        dataset,
        batch_size=training_cfg.batch_size,
        shuffle=shuffle,
        num_workers=training_cfg.num_workers,
        drop_last=shuffle if drop_last is None else drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=training_cfg.num_workers > 0,
    )
