# 5. LAV-DF dataset integration — Phase 11 decisions

- Status: Accepted
- Date: 2026-09-09
- Scope: LAV-DF dataset preparation for audio-visual synchronization
- Supersedes: none

This note records decisions made during the dataset preparation stage of Phase 11,
specifically the creation of a reusable LAV-DF subset manifest generator.

---

## Context

Phase 11 introduces audio-visual synchronization detection using the LAV-DF
(Large-scale Audio-Video DeepFake) dataset. LAV-DF contains 136,304 samples with
manipulations across audio, video, and both modalities. The dataset is distributed
as a 25.6 GB TAR archive containing video files and metadata.

This decision document covers ONLY the dataset-preparation stage. Model architecture,
training, and inference decisions will be documented in future notes.

Dataset structure:
- Total: 136,304 samples
- Train: 78,703 samples
- Dev: 31,501 samples
- Test: 26,100 samples (never used for development/training)

Metadata fields:
- file, n_fakes, fake_periods, duration, original
- modify_video, modify_audio, split
- video_frames, audio_frames

---

## Decisions

### D1 — TAR-based metadata access
Read `metadata.min.json` directly from inside the TAR archive using Python's
`tarfile` module. Do NOT extract the 25.6 GB archive. The script supports both
`LAV-DF/metadata.min.json` and `metadata.min.json` paths within the TAR.
**Status: Accepted.**

### D2 — Official split preservation
Preserve the official train/dev/test split exactly as provided in the metadata.
Never sample across splits. The test split must remain completely untouched and
must never be used to construct development/training subsets.
**Status: Accepted.**

### D3 — Manifest schema
Generate CSV manifests with the following columns:
- sample_id, path, label, label_name, split
- modify_audio, modify_video, n_fakes, fake_periods
- duration, original, video_frames, audio_frames

Label convention:
- `n_fakes == 0` → label 1, label_name "real"
- `n_fakes > 0` → label 0, label_name "manipulated"
**Status: Accepted.**

### D4 — Category classification for development subsets
Support selecting four categories for Phase 11 development data:
- **real**: n_fakes == 0
- **audio_only**: n_fakes > 0, modify_audio == true, modify_video == false
- **video_only**: n_fakes > 0, modify_audio == false, modify_video == true
- **audio_video**: n_fakes > 0, modify_audio == true, modify_video == true
**Status: Accepted.**

### D5 — Temporal information preservation
Keep `fake_periods` exactly represented in the manifest as a JSON string for
future temporal synchronization evaluation. This enables frame-level analysis
of manipulation boundaries.
**Status: Accepted.**

### D6 — Deterministic sampling
Sampling must be deterministic with a configurable random seed (default 42).
This ensures reproducible subset generation across runs.
**Status: Accepted.**

### D7 — Configurable subset sizes
Category counts and requested subset sizes must be configurable through CLI
arguments (--real, --audio-only, --video-only, --audio-video). Zero counts are
skipped. At least one category must have a positive count.
**Status: Accepted.**

### D8 — TAR member validation
Before writing a manifest, verify that every selected path exists as a member
of the TAR. Do not extract the files. This catches metadata/disk mismatches
early. Can be disabled with --no-verify-members.
**Status: Accepted.**

### D9 — Statistics reporting
Print comprehensive statistics before writing:
- Total selected
- Train/dev/test counts
- Real count
- Manipulated count
- Audio-only count
- Video-only count
- Audio+video count
- Duration min/max/mean
**Status: Accepted.**

### D10 — Dry-run mode
Support a --dry-run flag that prints statistics without writing any output file.
This enables exploration of subset configurations without side effects.
**Status: Accepted.**

### D11 — Test split protection
The script explicitly rejects attempts to include the test split in development
subsets (raises ValueError). A warning is printed if --splits includes "test".
**Status: Accepted.**

### D12 — Implementation approach
Create a standalone LAV-DF manifest builder (`scripts/build_lavdf_manifest.py`)
with its own `LAVDFRow` and `LAVDFManifest` classes. Do not duplicate the existing
generic `Manifest` class from `src.data.manifests` unless necessary. The LAV-DF
schema differs significantly (audio/video-specific fields, fake_periods, etc.).
**Status: Accepted.**

---

## Implementation details

### Script location
`scripts/build_lavdf_manifest.py`

### Test coverage
`tests/test_lavdf_manifest.py` covers:
- Metadata parsing from TAR
- Category classification (real, audio_only, video_only, audio_video)
- Label assignment (1=real, 0=manipulated)
- Deterministic sampling with seed
- Split isolation (no cross-split sampling)
- Manifest serialization (CSV round-trip)
- TAR member validation
- Statistics computation
- Integration tests for full workflow

### CLI interface
```bash
python scripts/build_lavdf_manifest.py --tar data/lavdf/LAV-DF.tar \
    --out data/lavdf/manifest.csv \
    --real 1000 \
    --audio-only 500 \
    --video-only 500 \
    --audio-video 500 \
    --seed 42 \
    --splits train dev \
    [--dry-run] \
    [--no-verify-members]
```

### Compatibility
- Python 3.11+
- Follows existing SyncGuard project style
- Reuses existing project utilities where appropriate
- No new dependencies beyond standard library

---

## Consequences

- Development subsets can be generated reproducibly without extracting the 25.6 GB TAR
- The test split is protected from accidental inclusion
- Temporal manipulation boundaries are preserved for future evaluation
- Category-based selection enables controlled experimentation with different
  manipulation types
- Statistics reporting provides immediate feedback on subset composition

---

## Open threads (non-blocking)

1. **Dataset exploration** — Once the manifest builder is tested, explore the actual
   LAV-DF metadata to understand category distributions and determine appropriate
   subset sizes for Phase 11 development.
2. **Temporal evaluation** — Design evaluation metrics that leverage the preserved
   `fake_periods` for frame-level synchronization assessment.
3. **Dataset integration** — Future work will integrate this manifest with the
   audio-visual dataset loader (not implemented in this stage).

---

## Evidence

- Script: `scripts/build_lavdf_manifest.py`
- Tests: `tests/test_lavdf_manifest.py`
- Decision document: `docs/decisions/0005-lavdf-dataset.md`
