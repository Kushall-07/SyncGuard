"""Score-level ensemble experiment (Phase 5): CNN + Transformer, no retraining.

Loads the CNN's saved eval scores, re-scores the Transformer on the *identical*
ASVspoof 2019 LA eval samples, then compares CNN-only / Transformer-only /
mean-ensemble on pooled EER+AUC and per-attack EER. Writes
outputs/analysis/ensemble/{ensemble.json, ensemble.txt, ensemble_scores.npz}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data.asvspoof_dataset import make_asvspoof_datasets
from src.data.audio_dataset import build_dataloader
from src.data.manifests import Manifest
from src.evaluation.metrics import equal_error_rate, roc_auc
from src.evaluation.spoof_eval import evaluate_spoof_model
from src.models.audio import SpoofClassifier
from src.training.checkpoint import load_checkpoint
from src.training.utils import get_device, set_seed

CNN_RUN = REPO_ROOT / "outputs" / "runs" / "spoof-cnn-baseline-20260906-104508"
TF_RUN = REPO_ROOT / "outputs" / "runs" / "spoof-transformer-20260906-123646"
CNN_SCORES = REPO_ROOT / "outputs" / "analysis" / "spoof-cnn-baseline" / "scores.npz"
MANIFEST = REPO_ROOT / "data" / "asvspoof" / "manifest.csv"
OUT = REPO_ROOT / "outputs" / "analysis" / "ensemble"


def pooled(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    eer, thr = equal_error_rate(y_true, y_score)
    return {"eer": float(eer), "eer_threshold": float(thr), "auc": float(roc_auc(y_true, y_score))}


def per_attack(y_true: np.ndarray, y_score: np.ndarray, attacks: np.ndarray) -> dict:
    bona = y_score[y_true == 1]
    out = {}
    for a in sorted(set(attacks[y_true == 0])):
        ss = y_score[(y_true == 0) & (attacks == a)]
        st = np.concatenate([np.ones_like(bona), np.zeros_like(ss)])
        sc = np.concatenate([bona, ss])
        eer, _ = equal_error_rate(st, sc)
        out[a] = float(eer)
    return out


@torch.no_grad()
def score_transformer() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = load_config(TF_RUN / "config.yaml")
    set_seed(cfg.experiment.seed)
    device = get_device()
    manifest = Manifest.read_csv(MANIFEST)
    ds = make_asvspoof_datasets(
        manifest, cfg.audio, feature=cfg.data.feature,
        fixed_seconds=cfg.data.fixed_seconds, splits=("eval",), seed=cfg.experiment.seed,
    )["eval"]
    model = SpoofClassifier(cfg.model, n_mels=cfg.audio.mel.n_mels).to(device)
    load_checkpoint(TF_RUN / "checkpoints" / "best.pt", model=model, map_location=device)
    model.eval()
    loader = build_dataloader(ds, cfg.training, shuffle=False, drop_last=False)
    res = evaluate_spoof_model(model, loader, device, attacks=[r.attack for r in ds.manifest])
    return res.y_true, res.y_score, np.array([r.attack for r in ds.manifest], dtype=object)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load(CNN_SCORES, allow_pickle=True)
    cnn_yt = z["eval_y_true"].astype(int)
    cnn_ys = z["eval_y_score"].astype(float)
    cnn_atk = z["eval_attacks"].astype(object)

    tf_yt, tf_ys, tf_atk = score_transformer()
    tf_yt = tf_yt.astype(int)
    tf_ys = tf_ys.astype(float)

    # --- alignment guarantees ---
    assert cnn_yt.shape == tf_yt.shape == (71237,), (cnn_yt.shape, tf_yt.shape)
    assert np.array_equal(cnn_yt, tf_yt), "y_true misaligned between CNN and Transformer scorings"
    assert np.array_equal(cnn_atk.astype(str), tf_atk.astype(str)), "attack ids misaligned"
    y_true, attacks = cnn_yt, cnn_atk
    print(f"alignment OK: {y_true.size} eval samples, "
          f"{int((y_true == 1).sum())} bonafide / {int((y_true == 0).sum())} spoof, "
          f"{len(set(attacks[y_true == 0]))} attacks")

    ens_ys = 0.5 * cnn_ys + 0.5 * tf_ys

    models = {"CNN": cnn_ys, "Transformer": tf_ys, "Ensemble(mean)": ens_ys}
    result = {"n_eval": int(y_true.size), "pooled": {}, "per_attack": {}}
    for name, ys in models.items():
        result["pooled"][name] = pooled(y_true, ys)
        result["per_attack"][name] = per_attack(y_true, ys, attacks)

    # small weight sweep for context (analysis only; primary result is the 0.5/0.5 mean)
    sweep = {}
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        s = w * tf_ys + (1 - w) * cnn_ys
        e, _ = equal_error_rate(y_true, s)
        sweep[f"w_tf={w:.1f}"] = {"eer": float(e), "auc": float(roc_auc(y_true, s))}
    result["weight_sweep_analysis_only"] = sweep

    # selection rule
    tf_eer = result["pooled"]["Transformer"]["eer"]
    ens_eer = result["pooled"]["Ensemble(mean)"]["eer"]
    tf_auc = result["pooled"]["Transformer"]["auc"]
    ens_auc = result["pooled"]["Ensemble(mean)"]["auc"]
    improves = (ens_eer < tf_eer - 1e-6) and (ens_auc >= tf_auc - 1e-6)
    result["selected_audio_model"] = "Ensemble(mean)" if improves else "Transformer"
    result["selection_reason"] = (
        f"ensemble pooled EER {ens_eer*100:.2f}% vs Transformer {tf_eer*100:.2f}%, "
        f"AUC {ens_auc:.4f} vs {tf_auc:.4f} -> "
        + ("ensemble selected" if improves else "no improvement, Transformer kept")
    )

    np.savez_compressed(OUT / "ensemble_scores.npz",
                        y_true=y_true, attacks=attacks.astype(str),
                        cnn=cnn_ys, transformer=tf_ys, ensemble=ens_ys)
    (OUT / "ensemble.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- text report ----
    L = []
    L.append("SCORE-LEVEL ENSEMBLE EXPERIMENT  (ASVspoof 2019 LA eval, n=71,237)")
    L.append("no retraining; P(bonafide) averaged over identical samples\n")
    L.append(f"{'model':<18}{'pooled EER':>12}{'pooled AUC':>12}{'EER thr':>10}")
    for name in models:
        p = result["pooled"][name]
        L.append(f"{name:<18}{p['eer']*100:>11.2f}%{p['auc']:>12.4f}{p['eer_threshold']:>10.4f}")
    L.append("")
    L.append("weight sweep (w_tf * Transformer + (1-w_tf) * CNN)  [analysis only]")
    for k, v in sweep.items():
        L.append(f"  {k:<10} EER={v['eer']*100:.2f}%  AUC={v['auc']:.4f}")
    L.append("")
    L.append(f"{'attack':<8}{'CNN':>9}{'Transf':>9}{'Ensemble':>10}{'best':>12}")
    pa = result["per_attack"]
    for a in sorted(pa["CNN"]):
        row = {n: pa[n][a] * 100 for n in models}
        best = min(row, key=row.get)
        L.append(f"{a:<8}{row['CNN']:>8.2f}%{row['Transformer']:>8.2f}%{row['Ensemble(mean)']:>9.2f}%"
                 f"{best:>12}")
    L.append("")
    L.append(f"SELECTED AUDIO MODEL: {result['selected_audio_model']}")
    L.append(f"  {result['selection_reason']}")
    txt = "\n".join(L)
    (OUT / "ensemble.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\nwrote {OUT/'ensemble.json'}, {OUT/'ensemble.txt'}, {OUT/'ensemble_scores.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
