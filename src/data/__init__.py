"""Dataset manifests and manifest-backed audio datasets."""

from src.data.asvspoof_dataset import (
    build_asvspoof_manifest,
    make_asvspoof_datasets,
    parse_protocol_file,
)
from src.data.audio_dataset import ManifestAudioDataset, build_dataloader, pad_or_crop
from src.data.manifests import (
    BONAFIDE,
    SPOOF,
    LeakageError,
    Manifest,
    ManifestRow,
    assert_speaker_disjoint,
    synthetic_manifest,
)

__all__ = [
    "Manifest",
    "ManifestRow",
    "LeakageError",
    "assert_speaker_disjoint",
    "synthetic_manifest",
    "BONAFIDE",
    "SPOOF",
    "ManifestAudioDataset",
    "pad_or_crop",
    "build_dataloader",
    "build_asvspoof_manifest",
    "make_asvspoof_datasets",
    "parse_protocol_file",
]
