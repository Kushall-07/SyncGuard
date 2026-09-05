"""ASVspoof 2019 LA protocol parsing and dataset construction (Phase 3).

The dataset itself is not bundled; :data:`DOWNLOAD_HINT` explains how to obtain it.
Once the ``LA`` directory is present, :func:`build_asvspoof_manifest` converts the
official countermeasure (CM) protocol files into a single
:class:`~src.data.manifests.Manifest`, and :func:`make_asvspoof_datasets` wraps
each split in a :class:`~src.data.audio_dataset.ManifestAudioDataset`.

Expected on-disk layout (``root`` points at ``LA``)::

    LA/
      ASVspoof2019_LA_train/flac/*.flac
      ASVspoof2019_LA_dev/flac/*.flac
      ASVspoof2019_LA_eval/flac/*.flac
      ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

Protocol line: ``SPEAKER_ID  UTTERANCE_ID  -  SYSTEM_ID  KEY`` where ``KEY`` is
``bonafide`` or ``spoof`` and ``SYSTEM_ID`` is ``-`` for bonafide or ``A01``..``A19``.
The three protocols use disjoint speaker pools by design; the builder verifies it.
"""

from __future__ import annotations

from pathlib import Path

from src.config import AudioConfig
from src.data.audio_dataset import ManifestAudioDataset
from src.data.manifests import (
    BONAFIDE,
    SPOOF,
    Manifest,
    ManifestRow,
    assert_speaker_disjoint,
)

__all__ = [
    "DOWNLOAD_HINT",
    "SOURCE_TAG",
    "PROTOCOL_FILES",
    "parse_protocol_file",
    "build_asvspoof_manifest",
    "make_asvspoof_datasets",
]

SOURCE_TAG = "asvspoof2019-la"

DOWNLOAD_HINT = (
    "ASVspoof 2019 LA was not found. Download it from the Edinburgh DataShare "
    "record 'ASVspoof 2019' (https://doi.org/10.7488/ds/2555), extract "
    "'LA.zip', and point --root at the resulting 'LA' directory (the one "
    "containing 'ASVspoof2019_LA_cm_protocols')."
)

PROTOCOL_FILES: dict[str, str] = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

_PROTOCOL_DIR = "ASVspoof2019_LA_cm_protocols"
_KEY_TO_LABEL = {"bonafide": BONAFIDE, "spoof": SPOOF}


def _flac_dir(root: Path, split: str) -> Path:
    return root / f"ASVspoof2019_LA_{split}" / "flac"


def parse_protocol_file(path: str | Path, split: str) -> list[dict[str, str]]:
    """Parse one CM protocol file into a list of ``{speaker, utterance, system_id, key}``."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"protocol file not found: {path}")

    entries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{path}:{lineno}: expected >=5 fields, got {parts!r}")
            speaker, utterance = parts[0], parts[1]
            system_id, key = parts[-2], parts[-1].lower()
            if key not in _KEY_TO_LABEL:
                raise ValueError(f"{path}:{lineno}: unexpected key {key!r}")
            entries.append(
                {"speaker": speaker, "utterance": utterance, "system_id": system_id, "key": key}
            )
    if not entries:
        raise ValueError(f"{path}: no entries parsed")
    return entries


def build_asvspoof_manifest(
    root: str | Path,
    *,
    out_csv: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "dev", "eval"),
    verify_audio: bool = True,
) -> Manifest:
    """Build a combined manifest for the requested splits.

    ``verify_audio`` checks that every referenced ``.flac`` exists (raises listing
    the first few missing). Set it False for a fast protocol-only pass.
    """

    root = Path(root)
    protocol_dir = root / _PROTOCOL_DIR
    if not protocol_dir.is_dir():
        raise FileNotFoundError(f"{DOWNLOAD_HINT}\n(looked for {protocol_dir})")

    rows: list[ManifestRow] = []
    missing: list[str] = []

    for split in splits:
        protocol_path = protocol_dir / PROTOCOL_FILES[split]
        flac_dir = _flac_dir(root, split)
        for entry in parse_protocol_file(protocol_path, split):
            flac_path = flac_dir / f"{entry['utterance']}.flac"
            if verify_audio and not flac_path.is_file():
                missing.append(str(flac_path))
                if len(missing) > 20:
                    break
            label = _KEY_TO_LABEL[entry["key"]]
            rows.append(
                ManifestRow.make(
                    sample_id=entry["utterance"],
                    path=flac_path,
                    label=label,
                    speaker=entry["speaker"],
                    split=split,
                    source=SOURCE_TAG,
                    attack=entry["system_id"] if label == SPOOF else "-",
                )
            )
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}+ audio files referenced by {protocol_path.name} are missing, "
                f"e.g.:\n  " + "\n  ".join(missing[:5])
            )

    manifest = Manifest(rows)
    manifest.assert_splits_speaker_disjoint()  # ASVspoof LA guarantees this

    if out_csv is not None:
        manifest.write_csv(out_csv)
    return manifest


def make_asvspoof_datasets(
    manifest: Manifest | str | Path,
    audio_cfg: AudioConfig,
    *,
    feature: str = "logmel",
    fixed_seconds: float = 4.0,
    splits: tuple[str, ...] = ("train", "dev", "eval"),
    seed: int = 0,
) -> dict[str, ManifestAudioDataset]:
    """Wrap each split in a :class:`ManifestAudioDataset` (random crop for ``train`` only)."""

    if not isinstance(manifest, Manifest):
        manifest = Manifest.read_csv(manifest)

    present = {r.split for r in manifest}
    wanted = [s for s in splits if s in present]
    if not wanted:
        raise ValueError(f"manifest has splits {sorted(present)}, none of {splits}")

    assert_speaker_disjoint(**{s: manifest.split(s) for s in wanted})

    datasets: dict[str, ManifestAudioDataset] = {}
    for split in wanted:
        datasets[split] = ManifestAudioDataset(
            manifest.split(split),
            audio_cfg,
            feature=feature,  # type: ignore[arg-type]
            fixed_seconds=fixed_seconds,
            random_crop=(split == "train"),
            seed=seed,
        )
    return datasets
