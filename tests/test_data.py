"""Phase 3 unit tests: manifests, leakage guards, ASVspoof protocol, audio dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import load_audio_config
from src.data.asvspoof_dataset import (
    build_asvspoof_manifest,
    make_asvspoof_datasets,
    parse_protocol_file,
)
from src.data.audio_dataset import ManifestAudioDataset, pad_or_crop
from src.data.manifests import (
    BONAFIDE,
    SPOOF,
    LeakageError,
    Manifest,
    ManifestRow,
    assert_speaker_disjoint,
    synthetic_manifest,
)

AUDIO_CFG = load_audio_config("configs/audio.yaml")


@pytest.fixture(scope="module")
def syn_manifest(tmp_path_factory) -> Manifest:
    out = tmp_path_factory.mktemp("syn_audio")
    return synthetic_manifest(
        out, n_per_split={"train": 20, "dev": 8, "eval": 8}, n_speakers_per_split=4,
        seed=7, sample_rate=AUDIO_CFG.sample_rate, duration_s=1.0,
    )


# --------------------------------------------------------------------- manifests


def test_manifest_row_label_name_is_derived() -> None:
    row = ManifestRow.make(sample_id="x", path="x.wav", label=1, speaker="s", split="train")
    assert row.label_name == "bonafide"
    with pytest.raises(ValueError):
        ManifestRow(sample_id="x", path="x.wav", label=1, label_name="spoof",
                    speaker="s", split="train")


def test_synthetic_manifest_is_balanced_and_disjoint(syn_manifest: Manifest) -> None:
    assert len(syn_manifest) == 36
    assert syn_manifest.split_sizes() == {"train": 20, "dev": 8, "eval": 8}
    dist = syn_manifest.label_distribution()
    assert dist["bonafide"] == dist["spoof"] == 18
    syn_manifest.assert_splits_speaker_disjoint()  # must not raise


def test_manifest_csv_round_trip(tmp_path, syn_manifest: Manifest) -> None:
    path = syn_manifest.write_csv(tmp_path / "m.csv")
    back = Manifest.read_csv(path)
    assert len(back) == len(syn_manifest)
    assert back.rows == syn_manifest.rows


def test_manifest_read_csv_rejects_missing_columns(tmp_path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("sample_id,path\nx,x.wav\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        Manifest.read_csv(bad)


def test_filter_accepts_scalar_and_collection(syn_manifest: Manifest) -> None:
    assert len(syn_manifest.filter(split="dev")) == 8
    assert len(syn_manifest.filter(split=["dev", "eval"])) == 16
    assert len(syn_manifest.filter(label=BONAFIDE, split="train")) == 10


def test_subset_is_deterministic_and_stratified(syn_manifest: Manifest) -> None:
    train = syn_manifest.split("train")
    a = train.subset(8, seed=3, stratify_by="label")
    b = train.subset(8, seed=3, stratify_by="label")
    assert [r.sample_id for r in a] == [r.sample_id for r in b]
    assert len(a) == 8
    labels = [r.label for r in a]
    assert labels.count(BONAFIDE) == labels.count(SPOOF) == 4


# ----------------------------------------------------------------- leakage guard


def test_assert_speaker_disjoint_raises_on_overlap() -> None:
    with pytest.raises(LeakageError, match="both"):
        assert_speaker_disjoint(train={"spk1", "spk2"}, eval={"spk2", "spk3"})


def test_assert_speaker_disjoint_accepts_manifests(syn_manifest: Manifest) -> None:
    assert_speaker_disjoint(
        train=syn_manifest.split("train"),
        dev=syn_manifest.split("dev"),
        eval=syn_manifest.split("eval"),
    )


# ----------------------------------------------------------------------- dataset


def test_pad_or_crop_pads_and_center_crops() -> None:
    short = torch.ones(1, 100)
    assert pad_or_crop(short, 160).shape == (1, 160)
    assert pad_or_crop(short, 160)[0, 120:].eq(0).all()
    long = torch.arange(200).float().view(1, 200)
    cropped = pad_or_crop(long, 100)  # center crop -> starts at 50
    assert cropped.shape == (1, 100) and cropped[0, 0].item() == 50.0


def test_dataset_yields_logmel_and_waveform(syn_manifest: Manifest) -> None:
    ds_mel = ManifestAudioDataset(syn_manifest.split("dev"), AUDIO_CFG,
                                  feature="logmel", fixed_seconds=1.0)
    feat, label = ds_mel[0]
    assert feat.shape == (AUDIO_CFG.mel.n_mels, 101)
    assert label in (0, 1)

    ds_wav = ManifestAudioDataset(syn_manifest.split("dev"), AUDIO_CFG,
                                  feature="waveform", fixed_seconds=1.0)
    wav, _ = ds_wav[0]
    assert wav.shape == (1, AUDIO_CFG.sample_rate)


def test_dataset_random_crop_is_seed_stable(syn_manifest: Manifest) -> None:
    kw = dict(feature="waveform", fixed_seconds=0.5, random_crop=True, seed=11)
    a = ManifestAudioDataset(syn_manifest.split("train"), AUDIO_CFG, **kw)[3][0]
    b = ManifestAudioDataset(syn_manifest.split("train"), AUDIO_CFG, **kw)[3][0]
    assert torch.equal(a, b)


def test_label_weights_sum_makes_sense(syn_manifest: Manifest) -> None:
    ds = ManifestAudioDataset(syn_manifest.split("train"), AUDIO_CFG, fixed_seconds=1.0)
    w = ds.label_weights()
    assert w.shape == (2,)
    assert torch.allclose(w, torch.tensor([1.0, 1.0]))  # balanced synthetic data


def test_make_asvspoof_datasets_random_crop_only_for_train(syn_manifest: Manifest) -> None:
    ds = make_asvspoof_datasets(syn_manifest, AUDIO_CFG, fixed_seconds=1.0,
                                splits=("train", "dev", "eval"))
    assert set(ds) == {"train", "dev", "eval"}
    assert ds["train"].random_crop is True
    assert ds["dev"].random_crop is False


# ---------------------------------------------------------------- asvspoof parse


_PROTOCOL_SAMPLE = """\
LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1272637 - A01 spoof
LA_0080 LA_T_1000001 - A02 spoof
"""


def test_parse_protocol_file(tmp_path) -> None:
    p = tmp_path / "ASVspoof2019.LA.cm.train.trn.txt"
    p.write_text(_PROTOCOL_SAMPLE, encoding="utf-8")
    entries = parse_protocol_file(p, "train")
    assert len(entries) == 3
    assert entries[0] == {"speaker": "LA_0079", "utterance": "LA_T_1138215",
                          "system_id": "-", "key": "bonafide"}
    assert entries[1]["system_id"] == "A01" and entries[1]["key"] == "spoof"


def test_parse_protocol_file_rejects_short_lines(tmp_path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("LA_0079 LA_T_1 bonafide\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected >=5 fields"):
        parse_protocol_file(p, "train")


def test_build_asvspoof_manifest_missing_root(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="ASVspoof 2019 LA was not found"):
        build_asvspoof_manifest(tmp_path / "nope")


def test_build_asvspoof_manifest_from_fake_tree(tmp_path) -> None:
    root = tmp_path / "LA"
    (root / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
    (root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt").write_text(
        _PROTOCOL_SAMPLE, encoding="utf-8"
    )
    flac_dir = root / "ASVspoof2019_LA_train" / "flac"
    flac_dir.mkdir(parents=True)
    for utt in ("LA_T_1138215", "LA_T_1272637", "LA_T_1000001"):
        (flac_dir / f"{utt}.flac").write_bytes(b"")  # existence check only

    manifest = build_asvspoof_manifest(root, splits=("train",))
    assert len(manifest) == 3
    assert manifest.label_distribution() == {"bonafide": 1, "spoof": 2}
    bona = manifest.filter(label=BONAFIDE).rows[0]
    assert bona.attack == "-" and bona.source == "asvspoof2019-la"
    assert manifest.filter(label=SPOOF).rows[0].attack == "A01"


def test_build_asvspoof_manifest_flags_missing_audio(tmp_path) -> None:
    root = tmp_path / "LA"
    (root / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
    (root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt").write_text(
        _PROTOCOL_SAMPLE, encoding="utf-8"
    )
    (root / "ASVspoof2019_LA_dev" / "flac").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="are missing"):
        build_asvspoof_manifest(root, splits=("dev",))
