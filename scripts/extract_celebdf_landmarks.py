"""Phase 7/8: cache MediaPipe FaceMesh landmarks for Celeb-DF-v2 videos.

For each video it runs the MediaPipe Tasks ``FaceLandmarker`` (478-point mesh) on
a set of selected frames and writes

    <landmarks-dir>/<sample_id>.npz        points[W,478,3] float32, valid[W] bool,
                                           fps float32, frame_idx[W] int64
    <landmarks-dir>/<sample_id>.meta.json  detection stats + source path

Frame selection has two modes (the MediaPipe model, the 478-point representation
and the ``.npz`` schema are identical in both):

* **sparse** (default, Phase 7): ``--num-frames`` (16) indices spread across the
  whole clip. Output dir defaults to ``data/celebdf/landmarks/``.
* **dense** (Phase 8, ``--window`` given): one *contiguous* window of ``--window``
  frames at ``--stride``, positioned by ``--window-start`` (``center`` by default,
  deterministic). Output dir defaults to ``data/celebdf/landmarks_dense/``. Clips
  with fewer than ``(window-1)*stride+1`` decoded frames are recorded as
  ``insufficient_decoded_frames`` (no ``.npz`` written) instead of crashing.

The model file ``face_landmarker.task`` is downloaded once (to
``checkpoints/mediapipe/`` by default) if not present. Runs are idempotent: a
video whose ``.npz`` already exists is skipped unless ``--overwrite``.

``extraction_audit.json`` is rebuilt from scratch on every run as a pure function
of the *current* landmark cache and the *current* ``--min-valid-frames``: no
merge with a previous audit, so a stale threshold can never contaminate it. Use
``--rescan-only`` to regenerate it (and per-video ``meta.json`` exclusion flags)
without touching any ``.npz`` or re-running MediaPipe.

Examples
--------
    # sparse 5-video dry run (sanity check)
    python scripts/extract_celebdf_landmarks.py --limit 5

    # dense Phase 8 default: 32 contiguous centre frames -> landmarks_dense/
    python scripts/extract_celebdf_landmarks.py --manifest data/celebdf/manifest.csv \
        --window 32 --stride 1 --window-start center

    # rebuild the audit from the existing cache at a new threshold, no extraction
    python scripts/extract_celebdf_landmarks.py --manifest data/celebdf/manifest.csv \
        --rescan-only --min-valid-frames 4
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.celebdf_dataset import (  # noqa: E402
    VIDEO_DIRS,
    video_sample_id,
)
from src.data.manifests import Manifest  # noqa: E402
from src.preprocessing.landmarks import N_FACEMESH_POINTS  # noqa: E402

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL = REPO_ROOT / "checkpoints" / "mediapipe" / "face_landmarker.task"


def ensure_model(path: Path) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading face_landmarker.task -> {path}")
    urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310 - fixed Google URL
    return path


def make_landmarker(model_path: Path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def read_frames(video_path: Path, max_frames: int = 600) -> tuple[list[np.ndarray], float]:
    """Decode a video to a list of RGB uint8 ``[H, W, 3]`` frames (capped)."""

    import av

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    return frames, fps


def sample_indices(n_frames: int, n_want: int) -> np.ndarray:
    """Sparse (legacy) selection: ``n_want`` indices spread across the whole clip."""

    if n_frames <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, n_frames - 1, n_want).round().astype(np.int64))


def dense_window_indices(
    n_frames: int, window: int, stride: int, window_start: str
) -> np.ndarray | None:
    """A single *contiguous* window of ``window`` frame indices at ``stride``.

    Indices are ``start, start+stride, ..., start+(window-1)*stride`` and span
    ``span = (window-1)*stride + 1`` source frames. ``window_start`` places the
    span deterministically: ``"start"`` -> 0, ``"end"`` -> flush to the last
    frame, ``"center"`` -> centred (integer floor). Returns ``None`` when the
    clip has fewer than ``span`` decoded frames (too short for a full window).
    """

    if n_frames <= 0 or window <= 0 or stride <= 0:
        return None
    span = (window - 1) * stride + 1
    if n_frames < span:
        return None
    if window_start == "start":
        start = 0
    elif window_start == "end":
        start = n_frames - span
    elif window_start == "center":
        start = (n_frames - span) // 2
    else:
        raise ValueError(f"window_start must be start|center|end, got {window_start!r}")
    return np.arange(window, dtype=np.int64) * stride + start


def extract_one(
    landmarker,
    video_path: Path,
    num_frames: int,
    *,
    window: int | None = None,
    stride: int = 1,
    window_start: str = "center",
) -> dict:
    """Landmark selected frames of one video.

    Sparse mode (``window is None``): ``num_frames`` indices from
    :func:`sample_indices`. Dense mode (``window`` set): one contiguous window
    from :func:`dense_window_indices`; if the clip is too short the returned dict
    has ``short=True`` and no ``points`` / ``frame_idx``.
    """

    import mediapipe as mp

    frames, fps = read_frames(video_path)
    n_decoded = len(frames)

    if window is not None:
        idx = dense_window_indices(n_decoded, window, stride, window_start)
        if idx is None:
            return {
                "points": None, "valid": None, "fps": np.float32(fps),
                "frame_idx": None, "n_frames_decoded": n_decoded, "short": True,
            }
    else:
        idx = sample_indices(n_decoded, num_frames)

    points = np.zeros((len(idx), N_FACEMESH_POINTS, 3), dtype=np.float32)
    valid = np.zeros(len(idx), dtype=bool)
    for k, fi in enumerate(idx):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frames[fi]))
        result = landmarker.detect(image)
        if result.face_landmarks:
            lms = result.face_landmarks[0]
            points[k] = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            valid[k] = True

    return {
        "points": points,
        "valid": valid,
        "fps": np.float32(fps),
        "frame_idx": idx.astype(np.int64),
        "n_frames_decoded": n_decoded,
        "short": False,
    }


def _read_meta(landmarks_dir: Path, sample_id: str) -> dict | None:
    p = landmarks_dir / f"{sample_id}.meta.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _npz_valid(landmarks_dir: Path, sample_id: str) -> tuple[int, int] | None:
    """``(n_valid, n_frames_in_npz)`` from the cached ``.npz``; ``None`` if absent/unreadable."""

    p = landmarks_dir / f"{sample_id}.npz"
    if not p.is_file():
        return None
    try:
        with np.load(p) as data:
            valid = np.asarray(data["valid"], dtype=bool)
        return int(valid.sum()), int(valid.size)
    except (OSError, ValueError, KeyError, EOFError):
        return None


def build_cache_audit(
    landmarks_dir: Path,
    sample_ids: list[str],
    min_valid_frames: int,
    *,
    refresh_meta: bool = True,
    window: int | None = None,
) -> dict:
    """Audit the *current* landmark cache against the *current* threshold.

    Pure function of what is on disk now - no merge with any previous audit. A
    video is excluded when its cached ``valid`` count is ``< min_valid_frames``;
    the reason is ``corrupt_or_truncated_source`` when the source decoded to fewer
    frames than the threshold (it could never pass), else ``face_not_detected``.
    Missing ``.npz`` -> ``missing_landmarks``. With ``refresh_meta`` each cached
    video's ``meta.json`` exclusion fields are rewritten to the current threshold
    (the ``.npz`` itself is never touched).
    """

    excluded: list[dict] = []
    rates: list[float] = []
    n_cached = n_uncached = 0

    for sid in sample_ids:
        stats = _npz_valid(landmarks_dir, sid)
        meta = _read_meta(landmarks_dir, sid)
        n_decoded = meta.get("n_frames_decoded") if meta else None

        if stats is None:
            n_uncached += 1
            # a dense short-video stub carries its own reason (no .npz written)
            reason = "missing_landmarks"
            if meta and meta.get("exclude_reason") == "insufficient_decoded_frames":
                reason = "insufficient_decoded_frames"
            excluded.append({
                "sample_id": sid,
                "n_detected": (meta or {}).get("n_detected"),
                "n_requested": (meta or {}).get("n_requested"),
                "n_frames_decoded": n_decoded,
                "reason": reason,
            })
            continue

        n_valid, n_in_npz = stats
        n_cached += 1
        rates.append(n_valid / n_in_npz if n_in_npz else 0.0)
        below = n_valid < min_valid_frames
        reason = None
        if below:
            truncated = (n_decoded is not None and n_decoded < min_valid_frames) \
                or n_in_npz < min_valid_frames
            reason = "corrupt_or_truncated_source" if truncated else "face_not_detected"
            excluded.append({
                "sample_id": sid, "n_detected": n_valid, "n_requested": n_in_npz,
                "n_frames_decoded": n_decoded, "reason": reason,
            })

        if refresh_meta and meta is not None:
            meta.update({
                "min_valid_frames": min_valid_frames,
                "excluded": below,
                "exclude_reason": reason,
            })
            try:
                (landmarks_dir / f"{sid}.meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8")
            except OSError:
                pass

    by_reason: dict[str, int] = {}
    for e in excluded:
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1

    return {
        "min_valid_frames": min_valid_frames,
        "window": window,
        "n_audited": len(sample_ids),
        "n_cached": n_cached,
        "n_uncached": n_uncached,
        "n_excluded": len(excluded),
        "excluded_by_reason": by_reason,
        "detection_rate_mean": float(np.mean(rates)) if rates else None,
        "detection_rate_min": float(np.min(rates)) if rates else None,
        "excluded": sorted(excluded, key=lambda e: e["sample_id"]),
    }


def iter_targets(args) -> list[tuple[str, Path]]:
    """Return ``[(sample_id, abs_video_path)]`` for the requested set."""

    root = Path(args.root)
    if args.manifest:
        manifest = Manifest.read_csv(args.manifest)
        return [(r.sample_id, Path(r.path)) for r in manifest]
    if args.videos:
        out = []
        for v in args.videos:
            p = (root / v) if not Path(v).is_absolute() else Path(v)
            rel = p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name
            out.append((video_sample_id(rel), p))
        return out
    targets: list[tuple[str, Path]] = []
    for folder in VIDEO_DIRS:
        for p in sorted((root / folder).glob("*.mp4")):
            targets.append((f"{folder}__{p.stem}", p))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "celebdf")
    parser.add_argument("--landmarks-dir", type=Path, default=None,
                        help="output dir; defaults to data/celebdf/landmarks (sparse) or "
                             "data/celebdf/landmarks_dense (when --window is given)")
    parser.add_argument("--manifest", type=Path, help="drive the set from a manifest CSV")
    parser.add_argument("--videos", nargs="+", help="explicit video paths (rel to --root or abs)")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="sparse mode: frames spread across the whole clip")
    parser.add_argument("--window", type=int, default=None,
                        help="dense mode: length of a single CONTIGUOUS frame window "
                             "(Phase 8 default 32)")
    parser.add_argument("--stride", type=int, default=1,
                        help="dense mode: frame step within the window (default 1)")
    parser.add_argument("--window-start", choices=["start", "center", "end"], default="center",
                        help="dense mode: where the window sits (deterministic; default center)")
    parser.add_argument("--limit", type=int, help="process at most N videos (dry run)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--min-valid-frames", type=int, default=4)
    parser.add_argument("--rescan-only", action="store_true",
                        help="rebuild extraction_audit.json (and meta.json exclusion flags) "
                             "from the existing cache at --min-valid-frames; no extraction")
    parser.add_argument("--no-refresh-meta", action="store_true",
                        help="with --rescan-only / after a run, do not rewrite per-video "
                             "meta.json exclusion flags")
    args = parser.parse_args()

    dense = args.window is not None
    if dense and (args.window < 1 or args.stride < 1):
        raise SystemExit("--window and --stride must be >= 1")
    if args.landmarks_dir is None:
        args.landmarks_dir = REPO_ROOT / "data" / "celebdf" / (
            "landmarks_dense" if dense else "landmarks")

    targets = iter_targets(args)
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit("no videos to process")

    args.landmarks_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = [sid for sid, _ in targets]
    audit_path = args.landmarks_dir / "extraction_audit.json"
    span = (args.window - 1) * args.stride + 1 if dense else None

    done = skipped = missing = 0
    new_exclusions = 0

    if not args.rescan_only:
        model_path = ensure_model(args.model_path)
        landmarker = make_landmarker(model_path)
        for i, (sample_id, video_path) in enumerate(targets, 1):
            npz_path = args.landmarks_dir / f"{sample_id}.npz"
            if npz_path.is_file() and not args.overwrite:
                skipped += 1
                continue
            if not video_path.is_file():
                print(f"  MISSING  {video_path}")
                missing += 1
                continue

            result = extract_one(
                landmarker, video_path, args.num_frames,
                window=args.window, stride=args.stride, window_start=args.window_start,
            )

            meta = {
                "sample_id": sample_id,
                "source": str(video_path),
                "n_frames_decoded": result["n_frames_decoded"],
                "fps": float(result["fps"]),
                "min_valid_frames": args.min_valid_frames,
            }
            if dense:
                meta.update({"mode": "dense", "window_requested": args.window,
                             "stride": args.stride, "window_start": args.window_start,
                             "window_span": span})

            # dense: clip too short for a full contiguous window -> record, no .npz
            if result.get("short"):
                meta.update({"n_requested": 0, "n_detected": 0, "detection_rate": 0.0,
                             "excluded": True, "exclude_reason": "insufficient_decoded_frames"})
                (args.landmarks_dir / f"{sample_id}.meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8")
                new_exclusions += 1
                done += 1
                continue

            rate = float(result["valid"].mean()) if result["valid"].size else 0.0
            n_valid = int(result["valid"].sum())
            below = n_valid < args.min_valid_frames
            new_exclusions += int(below)

            np.savez_compressed(
                npz_path,
                points=result["points"], valid=result["valid"],
                fps=result["fps"], frame_idx=result["frame_idx"],
            )
            meta.update({
                "n_requested": len(result["frame_idx"]),
                "n_detected": n_valid,
                "detection_rate": rate,
                "excluded": below,
                "exclude_reason": None,  # filled by build_cache_audit below
            })
            (args.landmarks_dir / f"{sample_id}.meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")
            done += 1
            if i % 50 == 0 or i == len(targets):
                print(f"  [{i}/{len(targets)}] done={done} skipped={skipped} "
                      f"new_exclusions={new_exclusions}")

    # extraction_audit.json is rebuilt from scratch: a pure function of the
    # current cache + current --min-valid-frames. No merge with any prior audit.
    audit = build_cache_audit(
        args.landmarks_dir, sample_ids, args.min_valid_frames,
        refresh_meta=not args.no_refresh_meta, window=args.window,
    )
    audit["mode"] = "dense" if dense else "sparse"
    audit["num_frames_requested"] = args.num_frames
    if dense:
        audit["window"] = args.window
        audit["stride"] = args.stride
        audit["window_start"] = args.window_start
        audit["window_span"] = span
    audit["run"] = {
        "rescan_only": args.rescan_only,
        "n_processed": done,
        "n_skipped_cached": skipped,
        "n_missing_on_disk": missing,
        "n_new_exclusions": new_exclusions,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if not args.rescan_only:
        print(f"\n{'dense' if dense else 'sparse'} mode"
              + (f" (window={args.window} stride={args.stride} start={args.window_start} "
                 f"span={span})" if dense else f" (num_frames={args.num_frames})"))
        print(f"processed {done}, skipped {skipped} (already cached), missing {missing}")
    print(f"cache-wide exclusions (< --min-valid-frames={args.min_valid_frames}): "
          f"{audit['n_excluded']}  -> {audit_path}")
    print(f"  by reason: {audit['excluded_by_reason']}")
    print(f"  cached {audit['n_cached']} / uncached {audit['n_uncached']} of {audit['n_audited']}")
    if audit["detection_rate_mean"] is not None:
        print(f"detection rate: mean {audit['detection_rate_mean']:.3f}  "
              f"min {audit['detection_rate_min']:.3f}")
    print(f"landmarks -> {args.landmarks_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
