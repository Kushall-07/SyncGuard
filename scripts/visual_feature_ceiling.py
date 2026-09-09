"""Phase 8 experiment 1: feature-ceiling probe for the visual landmark stream.

No neural training. Fits a Logistic Regression and a Gradient-Boosted-Trees
classifier on hand-pooled features of the *dense* normalised landmark sequences
and reports eval / dev AUC + EER for three feature sets:

    A  pooled coordinates          mean & std over time of the face_mouth coords
    B  A + motion                  per-coord |velocity| / |acceleration| stats
                                   + global frame-to-frame landmark jitter
    C  B + articulation geometry   per-frame mouth-aspect-ratio, lip thickness,
                                   eye-aspect-ratio (L/R), jaw opening, brow
                                   raise (L/R) -> time-series mean/std/range/|d|

If even set C caps near the level the previous investigation's linear model hit
(~0.68 eval AUC), the landmark representation itself carries little Celeb-DF
signal (hypothesis A) - independent of model or temporal architecture.

    python scripts/visual_feature_ceiling.py \
        --manifest data/celebdf/manifest_dense.csv \
        --landmarks-dir data/celebdf/landmarks_dense

Uses the exact preprocessing the model would see (interpolate_invalid ->
normalize_landmarks(interocular, align_rotation)); it consumes *all* cached
frames (32 for the dense cache), no random subsampling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.manifests import Manifest  # noqa: E402
from src.evaluation.metrics import equal_error_rate, roc_auc  # noqa: E402
from src.preprocessing.landmarks import (  # noqa: E402
    LEFT_BROW,
    LEFT_EYE,
    LIPS_INNER,
    LIPS_OUTER,
    NOSE_TIP,
    RIGHT_BROW,
    RIGHT_EYE,
    interpolate_invalid,
    normalize_landmarks,
    region_indices,
)

_EPS = 1e-6
_CHIN = 152  # MediaPipe FaceMesh chin apex (in FACE_OVAL)


# --------------------------------------------------------------------- features

def _clip_normalized(npz_path: Path, coords: int) -> np.ndarray:
    with np.load(npz_path) as data:
        points = np.asarray(data["points"], dtype=np.float32)   # [T, 478, 3]
        valid = np.asarray(data["valid"], dtype=bool)
    points = interpolate_invalid(points, valid)
    return normalize_landmarks(points, method="interocular", align_rotation=True, coords=coords)


def _pooled(seq: np.ndarray) -> np.ndarray:
    """mean & std over time of a flattened [T, D] view -> [2D]."""

    f = seq.reshape(seq.shape[0], -1)
    return np.concatenate([f.mean(axis=0), f.std(axis=0)])


def _motion(region_seq: np.ndarray) -> np.ndarray:
    f = region_seq.reshape(region_seq.shape[0], -1)          # [T, N*C]
    v = np.abs(np.diff(f, axis=0))
    a = np.abs(np.diff(f, axis=0, n=2)) if f.shape[0] > 2 else np.zeros((1, f.shape[1]), np.float32)
    per_coord = np.concatenate([
        v.mean(0), v.std(0), v.max(0),
        a.mean(0), a.std(0),
    ])
    # global per-frame landmark displacement magnitude
    d = np.sqrt((np.diff(region_seq, axis=0) ** 2).sum(axis=(1, 2)))   # [T-1]
    glob = np.array([d.mean(), d.std(), d.max()], dtype=np.float32) if d.size else np.zeros(3, np.float32)
    return np.concatenate([per_coord, glob])


def _articulation(norm_full: np.ndarray) -> np.ndarray:
    """Per-frame geometric scalars from the normalised full mesh -> pooled stats."""

    xy = norm_full[:, :, :2]                                  # [T, 478, 2]
    x, y = xy[..., 0], xy[..., 1]

    def _h(idx):  # vertical extent of an index set, per frame
        return y[:, list(idx)].max(1) - y[:, list(idx)].min(1)

    def _w(idx):
        return x[:, list(idx)].max(1) - x[:, list(idx)].min(1)

    mouth_h, mouth_w = _h(LIPS_INNER), _w(LIPS_OUTER)
    series = np.stack([
        mouth_h / (mouth_w + _EPS),                           # mouth aspect ratio
        _h(LIPS_OUTER) - mouth_h,                             # lip thickness
        _h(LEFT_EYE) / (_w(LEFT_EYE) + _EPS),                 # eye aspect ratio L
        _h(RIGHT_EYE) / (_w(RIGHT_EYE) + _EPS),               # eye aspect ratio R
        np.abs(y[:, _CHIN] - y[:, NOSE_TIP]),                 # jaw opening
        y[:, list(LEFT_BROW)].mean(1) - y[:, list(LEFT_EYE)].mean(1),   # brow raise L
        y[:, list(RIGHT_BROW)].mean(1) - y[:, list(RIGHT_EYE)].mean(1),  # brow raise R
    ], axis=1)                                                # [T, 7]

    dv = np.abs(np.diff(series, axis=0)) if series.shape[0] > 1 else np.zeros((1, 7), np.float32)
    return np.concatenate([
        series.mean(0), series.std(0), series.min(0), series.max(0),
        series.max(0) - series.min(0), dv.mean(0), dv.std(0),
    ])


def clip_feature_sets(npz_path: Path, region_idx: np.ndarray, coords: int) -> dict[str, np.ndarray]:
    norm = _clip_normalized(npz_path, coords)                 # [T, 478, coords]
    region_seq = norm[:, region_idx, :]                       # [T, N, coords]
    a = _pooled(region_seq)
    b = np.concatenate([a, _motion(region_seq)])
    c = np.concatenate([b, _articulation(norm)])
    return {"A": a, "B": b, "C": c}


# --------------------------------------------------------------------- driving

def build_split(manifest: Manifest, split: str, landmarks_dir: Path, region_idx: np.ndarray,
                coords: int, limit: int | None, seed: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows = list(manifest.split(split))
    if limit and limit < len(rows):
        # deterministic shuffled subsample (the manifest is folder-ordered, so a
        # head slice would be single-class)
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in rng.permutation(len(rows))[:limit]]
    feats: dict[str, list] = {"A": [], "B": [], "C": []}
    labels: list[int] = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        npz = landmarks_dir / f"{r.sample_id}.npz"
        if not npz.is_file():
            continue
        fs = clip_feature_sets(npz, region_idx, coords)
        for k in feats:
            feats[k].append(fs[k])
        labels.append(int(r.label))
        if i % 500 == 0:
            print(f"    {split}: {i}/{len(rows)}  ({time.time() - t0:.0f}s)")
    X = {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}
    return X, np.asarray(labels, dtype=np.int64)


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    w = np.empty(n, dtype=np.float64)
    for c in (0, 1):
        m = y == c
        w[m] = n / (2.0 * max(m.sum(), 1))
    return w


def fit_eval(Xtr, ytr, Xev, yev, Xdv, ydv, seed: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    sw = _balanced_weights(ytr)
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed),
        ),
        "hist_gbt": HistGradientBoostingClassifier(random_state=seed, early_stopping=True),
    }
    for name, model in models.items():
        if name == "hist_gbt":
            model.fit(Xtr, ytr, sample_weight=sw)
        else:
            model.fit(Xtr, ytr)
        for split_name, Xs, ys in (("eval", Xev, yev), ("dev", Xdv, ydv)):
            p = model.predict_proba(Xs)[:, 1]
            eer, thr = equal_error_rate(ys, p)
            out[f"{name}/{split_name}"] = {
                "auc": round(float(roc_auc(ys, p)), 4),
                "eer": round(float(eer), 4),
                "eer_threshold": round(float(thr), 4),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data" / "celebdf" / "manifest_dense.csv")
    parser.add_argument("--landmarks-dir", type=Path,
                        default=REPO_ROOT / "data" / "celebdf" / "landmarks_dense")
    parser.add_argument("--regions", default="face_mouth", choices=["face", "mouth", "face_mouth"])
    parser.add_argument("--coords", type=int, default=3, choices=[2, 3])
    parser.add_argument("--train-subsample", type=int, default=4000,
                        help="cap train clips (0 = all); deterministic head of the split")
    parser.add_argument("--limit", type=int, help="per-split cap (smoke test)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "outputs" / "analysis" / "visual_feature_ceiling" / "ceiling.json")
    args = parser.parse_args()

    manifest = Manifest.read_csv(args.manifest)
    region_idx = region_indices(args.regions)
    print(f"manifest {args.manifest.name}  splits {manifest.split_sizes()}")
    print(f"landmarks {args.landmarks_dir}  regions={args.regions} coords={args.coords}")

    tr_limit = args.limit or (args.train_subsample or None)
    Xtr, ytr = build_split(manifest, "train", args.landmarks_dir, region_idx, args.coords,
                           tr_limit, seed=args.seed)
    Xev, yev = build_split(manifest, "eval", args.landmarks_dir, region_idx, args.coords,
                           args.limit, seed=args.seed)
    Xdv, ydv = build_split(manifest, "dev", args.landmarks_dir, region_idx, args.coords,
                           args.limit, seed=args.seed)
    print(f"\ntrain {len(ytr)} (real {int(ytr.sum())})  eval {len(yev)} (real {int(yev.sum())})  "
          f"dev {len(ydv)} (real {int(ydv.sum())})")

    results: dict[str, dict] = {}
    for fs in ("A", "B", "C"):
        r = fit_eval(Xtr[fs], ytr, Xev[fs], yev, Xdv[fs], ydv, args.seed)
        r["_n_features"] = int(Xtr[fs].shape[1])
        results[fs] = r

    payload = {
        "manifest": str(args.manifest), "landmarks_dir": str(args.landmarks_dir),
        "regions": args.regions, "coords": args.coords, "seed": args.seed,
        "n_train": len(ytr), "n_eval": len(yev), "n_dev": len(ydv),
        "feature_sets": {
            "A": "pooled coords (mean+std over time)",
            "B": "A + per-coord |velocity|/|accel| stats + global landmark jitter",
            "C": "B + articulation geometry time-series (MAR, lip thickness, EAR L/R, jaw, brow L/R)",
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'set':<4}{'nfeat':>7}  {'model':<9}{'eval AUC':>10}{'eval EER':>10}{'dev AUC':>9}{'dev EER':>9}")
    print("-" * 60)
    for fs in ("A", "B", "C"):
        r = results[fs]
        for m in ("logreg", "hist_gbt"):
            e, d = r[f"{m}/eval"], r[f"{m}/dev"]
            print(f"{fs:<4}{r['_n_features']:>7}  {m:<9}{e['auc']:>10.3f}{e['eer']:>10.3f}"
                  f"{d['auc']:>9.3f}{d['eer']:>9.3f}")
    print(f"\nceiling.json -> {args.out}")
    best_eval_auc = max(results[fs][f"{m}/eval"]["auc"]
                        for fs in ("A", "B", "C") for m in ("logreg", "hist_gbt"))
    print(f"best eval AUC across all feature sets / models: {best_eval_auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
