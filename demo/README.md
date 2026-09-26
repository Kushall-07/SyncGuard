# SyncGuard Demo Samples

Small, curated set of local audio and audio-visual samples for live exhibition
demos of SyncGuard. **These are demonstration samples only, not a benchmark
dataset**, and no source dataset is redistributed here beyond this handful of
files — the full ASVspoof2019-LA and LAV-DF datasets remain wherever they are
already stored locally for this project and are unmodified.

## Audio

`audio/spoof/`
- Curated ASVspoof2019-LA **spoof** examples (eval split), spread across the
  13 attack types present in the eval partition (A07–A19) for variety.

`audio/bonafide/`
- Curated ASVspoof2019-LA **bona fide** examples (eval split), one per
  speaker across 25 distinct speakers.

Files are kept in their original FLAC format — no re-encoding was performed,
since the SyncGuard audio predictor already supports FLAC input directly.

## Audio-Visual (Synchronization)

**Important:** SyncGuard's AV Sync Head detects *temporal audio-video
synchronization*, not deepfake/manipulation status. A manipulated clip can
still be perfectly synchronized, and a genuine clip can be desynchronized by
a simple timing shift. For that reason these folders are named by
synchronization status (`sync` / `desync`), not by real/fake labels, and both
groups are built from **genuine, unmanipulated** LAV-DF video/audio pairs
(`label_name == "real"` in the LAV-DF manifest) — manipulation category is
never used as a stand-in for a synchronization label.

`video/sync/`
- Native, untouched video/audio pairs extracted from genuine LAV-DF clips.
  Audio and video are at their original relative timing.

`video/desync/`
- The **same kind** of genuine LAV-DF video/audio pairs, but with a
  controlled **+0.5 second** audio shift applied to a copy of the audio only.
  The video file is byte-for-byte the original extracted clip; only the demo
  audio copy is shifted. This is a controlled temporal-shift demonstration,
  not a re-labeled manipulated sample.

### File naming

Each sample is a numbered pair inside its folder:

```
demo/video/sync/0001_video.mp4
demo/video/sync/0001_audio.wav
demo/video/sync/0002_video.mp4
demo/video/sync/0002_audio.wav
...
```

The `_video` / `_audio` suffix makes it unambiguous which files belong
together, matching the video+audio input the SyncGuard AV predictor expects.

### How the desync audio was generated

For each `desync` sample:
1. The original LAV-DF video (`extracted/dev/<id>.mp4`) is copied unchanged.
2. The corresponding pre-extracted native audio
   (`processed/audio/<id>.wav`) is read.
3. A new audio file is created by prepending 0.5 seconds of silence and
   trimming the same amount off the end, so the audio content begins 0.5s
   later relative to the unchanged video timeline while total duration stays
   identical to the original.
4. The result is written only to `demo/video/desync/`; the original
   processed audio file is never overwritten.

The shift was verified per-sample by cross-correlating the shifted audio
against the original audio and confirming the measured lag equals +0.5s
(within one audio sample). All 25/25 desync samples verified correctly.

## Source table

| Demo category        | Source dataset      | Intended label/condition                  | Approx. count | Temporal shift |
|-----------------------|---------------------|--------------------------------------------|:---:|:---:|
| `audio/spoof/`         | ASVspoof2019-LA (eval) | spoof                                    | 25 | – |
| `audio/bonafide/`      | ASVspoof2019-LA (eval) | bonafide                                 | 25 | – |
| `video/sync/`          | LAV-DF (dev, `label_name == real`) | native synchronized A/V pair | 25 | 0s |
| `video/desync/`        | LAV-DF (dev, `label_name == real`) | controlled desynchronized A/V pair | 25 | +0.5s |

## Notes

- Selection was made using the existing dataset manifests
  (`data/asvspoof/manifest.csv`, `data/lavdf/manifest_dev.csv`) — labels were
  read from the manifest, never guessed from filenames.
- These demo files are a small, hand-picked illustration set for exhibition
  purposes. They are **not** statistically representative of the full
  datasets and must not be used for benchmarking or reporting model metrics.
- Original dataset files under `data/asvspoof/` and `data/lavdf/` were only
  read from, never modified, moved, or deleted.
