# SyncGuard — Datasets

None of these datasets are copied into this repository or redistributed.
`data/` on disk holds local copies obtained directly from each dataset's
official source under its own license/access terms; only manifests (CSV files
listing sample IDs, labels, and metadata — no media) are ever meant to be
checked into version control, and even those should be reviewed against
`.gitignore` before committing.

## Datasets in current use

### ASVspoof 2019 LA

| Field | Value |
|---|---|
| Purpose | Audio spoof / synthetic-speech detection (training + evaluation) |
| Modality | Audio only |
| Role in SyncGuard | Primary dataset for the audio-only branch (CNN + Transformer + ensemble); provides the frozen audio encoder later reused, unmodified, in the AV pipeline |
| Official source | https://www.asvspoof.org/ (ASVspoof 2019 challenge); dataset hosted via the official ASVspoof/Edinburgh DataShare mirror |
| License / access | Available for research use under the ASVspoof organizers' terms; requires accepting the dataset's usage agreement on the official download page |
| Redistribution | Not redistributed by this project. Only a manifest CSV (protocol-derived sample IDs, labels, attack IDs) is present under `data/asvspoof/manifest.csv`; audio itself is not committed |
| Included in Git | Manifest only, subject to `.gitignore` — verify before committing; the audio archive (`data/asvspoof/LA.zip`) and extracted audio are excluded |

### Celeb-DF v2

| Field | Value |
|---|---|
| Purpose | Development of the standalone visual landmark representation and its honesty-checked standalone evaluation |
| Modality | Video (no audio track used by this project) |
| Role in SyncGuard | Source of face video for MediaPipe landmark extraction and the Phase 7/8 visual Transformer; **not** used for AV synchronization, since it does not provide the audio this project's AV pipeline requires |
| Official source | https://github.com/yuezunli/celeb-deepfakeforensics (request form for the official release) |
| License / access | Research-only license; requires submitting the official request form and agreeing to the dataset's terms of use before download |
| Redistribution | Not redistributed. Manifests (`data/celebdf/manifest*.csv`) and derived landmark `.npz` files are local build artifacts, not the original media |
| Included in Git | Manifests/landmarks only, subject to `.gitignore`; the video archive (`data/celebdf/Celeb-DF-v2.zip`) and raw video directories are excluded |

### LAV-DF (Localized Audio-Visual DeepFake)

| Field | Value |
|---|---|
| Purpose | Audio-visual synchronization training/evaluation, and the manipulation-category native-timing evaluation in `docs/experiments.md` Section 3.3 |
| Modality | Video + audio, with manipulation metadata |
| Role in SyncGuard | Only dataset in this project with real (video, audio, manipulation-label) triples; used for the controlled temporal-shift experiment and the independent native-timing category evaluation. Its `fake_periods` field is preserved in manifests but is never used as a synchronization label (see `docs/decisions/0005-lavdf-dataset.md`) |
| Official source | https://github.com/ControlNet/LAV-DF |
| License / access | Released for non-commercial research use; see the official repository for the exact license terms and access instructions |
| Redistribution | Not redistributed. The 25.6 GB official TAR archive is read directly for metadata and selective extraction; only manifests and derived audio/landmark artifacts for the sampled subset are stored locally |
| Included in Git | Manifests only, subject to `.gitignore`; `data/lavdf/LAV-DF.tar`, extracted videos, and processed audio/landmarks are excluded |
| Splits | Official train (78,703) / dev (31,501) / test (26,100), preserved exactly. **The test split has never been extracted, preprocessed, or evaluated in this project** — see `docs/decisions/0005-lavdf-dataset.md` D2 and `docs/limitations.md` item 14 |

---

## Candidate datasets investigated for independent AV-sync validation (not downloaded)

These were evaluated as candidates for a genuinely independent (non-LAV-DF)
audio-visual evaluation source, per the Section 6 investigation in this
project's research-validation phase. None were downloaded automatically —
each requires tens of GB and/or an access request, which this project does
not do without explicit approval. `scripts/evaluate_av_independent.py` can be
pointed at any of these once converted into the LAV-DF-style manifest schema
(`sample_id, path, label, label_name, split, modify_audio, modify_video,
n_fakes, fake_periods, duration, original, video_frames, audio_frames`) plus
extracted video / audio / MediaPipe-landmark `.npz` files, following the same
layout `scripts/extract_lavdf_subset.py` and `scripts/preprocess_lavdf.py`
already produce for LAV-DF.

### FakeAVCeleb

| Field | Value |
|---|---|
| Modality | Video + audio |
| Manipulation metadata | Yes — real, fake-audio-only, fake-video-only, fake-both categories, comparable in spirit to LAV-DF's `modify_audio`/`modify_video` flags |
| Explicit synchronization ground truth | Not documented — categories are manipulation-content labels, the same conflation risk as LAV-DF |
| Official source | https://github.com/DASH-Lab/FakeAVCeleb |
| License / access | Research-only; requires filling out the official Google Form request on the repository before a download link is provided |
| Redistribution | N/A — not present locally |
| Required to use here | Full dataset or a category-balanced subset, converted to the LAV-DF manifest schema; audio and MediaPipe landmarks extracted per-clip the same way as LAV-DF |

### DFDC (Deepfake Detection Challenge dataset)

| Field | Value |
|---|---|
| Modality | Video + audio (original audio is largely preserved even in manipulated clips) |
| Manipulation metadata | Real/fake video label only, no audio-manipulation flag and no timing metadata |
| Explicit synchronization ground truth | None |
| Official source | https://ai.meta.com/datasets/dfdc/ (Meta AI / the original Kaggle DFDC competition) |
| License / access | Research-only license; requires registering and agreeing to the dataset terms |
| Redistribution | N/A — `data/dfdc/` in this repository is currently an empty placeholder directory; nothing has been downloaded |
| Required to use here | Because DFDC does not flag audio-only vs. video-only vs. both-modality manipulation, it can only support a REAL-vs-FAKE grouping (not the four-way LAV-DF-style category breakdown) if used with `scripts/evaluate_av_independent.py` |

### KoDF (Korean DeepFake Detection Dataset)

| Field | Value |
|---|---|
| Modality | Video + audio |
| Manipulation metadata | Real/fake label only |
| Explicit synchronization ground truth | None |
| Official source | https://deepbrainai-research.github.io/kodf/ |
| License / access | Research-only; requires an access request to the dataset authors |
| Redistribution | N/A — not present locally |
| Required to use here | Same constraints as DFDC — no audio/video-only manipulation split, so only a REAL-vs-FAKE grouping is possible |

## Conclusion of the dataset investigation

No dataset — local or among the above candidates — carries an explicit,
annotated audio-visual desynchronization label independent of content
manipulation. See `docs/experiments.md` Section 3.4 for the full reasoning and
`docs/limitations.md` item 7. This project therefore does not claim validation
against real, ground-truth-labeled desynchronization; it reports controlled
temporal-shift sensitivity (Section 3.2) and a manipulation-category
descriptive check at native timing (Section 3.3) as two separate, honestly
labeled pieces of evidence instead.
