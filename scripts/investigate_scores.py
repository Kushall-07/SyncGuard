"""Phase 4 baseline investigation, model side: score dev + eval with best.pt,
then do threshold / calibration / per-attack / error analysis.

Writes outputs/analysis/spoof-cnn-baseline/{scores.npz, analysis.json} and prints
a full report. Uses the project's own evaluate_spoof_model + metrics where possible.
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
from src.evaluation.metrics import (
    binary_classification_report,
    equal_error_rate,
    roc_auc,
)
from src.evaluation.spoof_eval import evaluate_spoof_model
from src.models.audio import SpoofClassifier
from src.training.checkpoint import load_checkpoint
from src.training.utils import get_device, set_seed

RUN = REPO_ROOT / "outputs" / "runs" / "spoof-cnn-baseline-20260906-104508"
MANIFEST = REPO_ROOT / "data" / "asvspoof" / "manifest.csv"
OUT = REPO_ROOT / "outputs" / "analysis" / "spoof-cnn-baseline"


def far_frr_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> tuple[float, float, float, float]:
    """Return (accuracy, f1_bonafide, FAR=spoof->bonafide rate, FRR=bonafide->spoof rate) at threshold thr."""
    pred = (y_score >= thr).astype(int)
    acc = float((pred == y_true).mean())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else float("nan")   # spoof accepted as bonafide
    frr = fn / (fn + tp) if (fn + tp) else float("nan")   # bonafide rejected as spoof
    return acc, f1, far, frr


def sweep_best(y_true: np.ndarray, y_score: np.ndarray):
    grid = np.unique(np.round(np.concatenate([np.linspace(0, 1, 2001), y_score]), 6))
    best_f1 = (-1.0, 0.5)
    best_acc = (-1.0, 0.5)
    for t in grid:
        acc, f1, _, _ = far_frr_at(y_true, y_score, t)
        if f1 > best_f1[0]:
            best_f1 = (f1, float(t))
        if acc > best_acc[0]:
            best_acc = (acc, float(t))
    return {"f1_optimal": {"f1": best_f1[0], "threshold": best_f1[1]},
            "acc_optimal": {"accuracy": best_acc[0], "threshold": best_acc[1]}}


@torch.no_grad()
def score_split(model, ds, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    loader = build_dataloader(ds, CFG.training, shuffle=False, drop_last=False)
    res = evaluate_spoof_model(model, loader, device,
                               attacks=[r.attack for r in ds.manifest])
    attacks = [r.attack for r in ds.manifest]
    return res.y_true, res.y_pred, res.y_score, attacks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    global CFG
    CFG = load_config(RUN / "config.yaml")
    set_seed(CFG.experiment.seed)
    device = get_device()

    manifest = Manifest.read_csv(MANIFEST)
    datasets = make_asvspoof_datasets(
        manifest, CFG.audio, feature=CFG.data.feature,
        fixed_seconds=CFG.data.fixed_seconds, splits=("dev", "eval"), seed=CFG.experiment.seed,
    )
    model = SpoofClassifier(CFG.model, n_mels=CFG.audio.mel.n_mels).to(device)
    load_checkpoint(RUN / "checkpoints" / "best.pt", model=model, map_location=device)
    model.eval()

    out: dict = {}
    store: dict = {}
    for split in ("dev", "eval"):
        yt, yp, ys, atk = score_split(model, datasets[split], device)
        store[f"{split}_y_true"] = yt
        store[f"{split}_y_score"] = ys
        store[f"{split}_attacks"] = np.array(atk, dtype=object)

        rep_argmax = binary_classification_report(yt, yp, ys)
        eer, eer_thr = equal_error_rate(yt, ys)
        acc_05, f1_05, far_05, frr_05 = far_frr_at(yt, ys, 0.5)
        acc_eer, f1_eer, far_eer, frr_eer = far_frr_at(yt, ys, eer_thr)
        sweeps = sweep_best(yt, ys)

        bona = ys[yt == 1]
        spoof = ys[yt == 0]
        out[split] = {
            "n": int(yt.size), "n_bonafide": int((yt == 1).sum()), "n_spoof": int((yt == 0).sum()),
            "argmax_0.5": {"accuracy": rep_argmax["accuracy"], "precision": rep_argmax["precision"],
                            "recall": rep_argmax["recall"], "f1": rep_argmax["f1"],
                            "confusion_matrix_[[TN,FP],[FN,TP]]": rep_argmax["confusion_matrix"],
                            "FAR_spoof_as_bonafide": far_05, "FRR_bonafide_as_spoof": frr_05},
            "roc_auc": rep_argmax["roc_auc"],
            "eer": eer, "eer_threshold": eer_thr,
            "at_eer_threshold": {"accuracy": acc_eer, "f1": f1_eer,
                                  "FAR": far_eer, "FRR": frr_eer},
            "f1_optimal_threshold_ANALYSIS_ONLY": sweeps["f1_optimal"],
            "acc_optimal_threshold_ANALYSIS_ONLY": sweeps["acc_optimal"],
            "score_dist": {
                "bonafide": {"mean": float(bona.mean()), "std": float(bona.std()),
                              "p5": float(np.percentile(bona, 5)), "p50": float(np.percentile(bona, 50)),
                              "p95": float(np.percentile(bona, 95))},
                "spoof": {"mean": float(spoof.mean()), "std": float(spoof.std()),
                           "p5": float(np.percentile(spoof, 5)), "p50": float(np.percentile(spoof, 50)),
                           "p95": float(np.percentile(spoof, 95)),
                           "frac_above_0.5": float((spoof >= 0.5).mean()),
                           "frac_above_0.9": float((spoof >= 0.9).mean())},
            },
        }

    # ---- per-attack table on eval (EER vs pooled bonafide, + AUC + argmax miss rate)
    yt = store["eval_y_true"]; ys = store["eval_y_score"]; atk = store["eval_attacks"]
    bona_scores = ys[yt == 1]
    per_attack = {}
    for a in sorted(set(atk[yt == 0])):
        sm = (yt == 0) & (atk == a)
        ss = ys[sm]
        sub_true = np.concatenate([np.ones_like(bona_scores), np.zeros_like(ss)])
        sub_score = np.concatenate([bona_scores, ss])
        a_eer, a_thr = equal_error_rate(sub_true, sub_score)
        a_auc = roc_auc(sub_true, sub_score)
        per_attack[a] = {
            "n_spoof": int(ss.size),
            "eer_vs_pooled_bonafide": float(a_eer),
            "auc_vs_pooled_bonafide": float(a_auc),
            "mean_score_Pbonafide": float(ss.mean()),
            "miss_rate_at_0.5": float((ss >= 0.5).mean()),      # spoof classified bonafide
            "miss_rate_at_eer_thr_global": float((ss >= out["eval"]["eer_threshold"]).mean()),
        }
    out["per_attack_eval"] = per_attack

    # ---- error analysis at argmax 0.5 on eval
    pred = (ys >= 0.5).astype(int)
    fp_mask = (pred == 1) & (yt == 0)   # spoof -> bonafide
    fn_mask = (pred == 0) & (yt == 1)   # bonafide -> spoof
    fp_by_attack = {a: int(((atk == a) & fp_mask).sum()) for a in sorted(set(atk[yt == 0]))}
    total_fp = int(fp_mask.sum())
    out["error_analysis_eval_argmax0.5"] = {
        "FP_spoof_as_bonafide": total_fp,
        "FN_bonafide_as_spoof": int(fn_mask.sum()),
        "FP_rate_over_spoof": float(total_fp / (yt == 0).sum()),
        "FN_rate_over_bonafide": float(fn_mask.sum() / (yt == 1).sum()),
        "FP_by_attack": fp_by_attack,
        "FP_by_attack_pct_of_all_FP": {a: round(100 * c / total_fp, 1) for a, c in fp_by_attack.items()},
        "correct_spoof_mean_score": float(ys[(pred == 0) & (yt == 0)].mean()),
        "wrong_spoof_mean_score": float(ys[fp_mask].mean()),
    }

    np.savez_compressed(OUT / "scores.npz", **store)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------- pretty print ----------
    def block(title): print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

    block("DEV vs EVAL  —  headline")
    print(f"{'metric':<22}{'dev':>14}{'eval':>14}{'diff':>14}")
    for k, path in [("accuracy@0.5", ("argmax_0.5", "accuracy")),
                    ("f1@0.5", ("argmax_0.5", "f1")),
                    ("roc_auc", ("roc_auc",)),
                    ("eer", ("eer",)),
                    ("eer_threshold", ("eer_threshold",))]:
        dv = out["dev"]; ev = out["eval"]
        for p in path[:-1]:
            dv = dv[p]; ev = ev[p]
        dv = dv[path[-1]] if isinstance(dv, dict) else dv
        ev = ev[path[-1]] if isinstance(ev, dict) else ev
        print(f"{k:<22}{dv:>14.4f}{ev:>14.4f}{ev - dv:>14.4f}")

    block("SCORE DISTRIBUTION  P(bonafide)  —  dev vs eval")
    for split in ("dev", "eval"):
        sd = out[split]["score_dist"]
        print(f"\n{split}:")
        print(f"  bonafide  mean={sd['bonafide']['mean']:.3f} std={sd['bonafide']['std']:.3f} "
              f"p5={sd['bonafide']['p5']:.3f} p50={sd['bonafide']['p50']:.3f} p95={sd['bonafide']['p95']:.3f}")
        print(f"  spoof     mean={sd['spoof']['mean']:.3f} std={sd['spoof']['std']:.3f} "
              f"p5={sd['spoof']['p5']:.3f} p50={sd['spoof']['p50']:.3f} p95={sd['spoof']['p95']:.3f}")
        print(f"  spoof frac scored >=0.5 : {sd['spoof']['frac_above_0.5']:.3f}   "
              f">=0.9 : {sd['spoof']['frac_above_0.9']:.3f}")

    block("THRESHOLD BEHAVIOUR on EVAL")
    ev = out["eval"]
    print(f"  argmax (thr=0.5)        acc={ev['argmax_0.5']['accuracy']:.4f}  f1={ev['argmax_0.5']['f1']:.4f}  "
          f"FAR={ev['argmax_0.5']['FAR_spoof_as_bonafide']:.4f}  FRR={ev['argmax_0.5']['FRR_bonafide_as_spoof']:.4f}")
    print(f"  EER thr (={ev['eer_threshold']:.4f})   acc={ev['at_eer_threshold']['accuracy']:.4f}  "
          f"f1={ev['at_eer_threshold']['f1']:.4f}  FAR={ev['at_eer_threshold']['FAR']:.4f}  FRR={ev['at_eer_threshold']['FRR']:.4f}")
    f1o = ev["f1_optimal_threshold_ANALYSIS_ONLY"]; aco = ev["acc_optimal_threshold_ANALYSIS_ONLY"]
    print(f"  F1-opt thr (={f1o['threshold']:.4f}) [ANALYSIS ONLY]  f1={f1o['f1']:.4f}")
    print(f"  Acc-opt thr (={aco['threshold']:.4f}) [ANALYSIS ONLY] acc={aco['accuracy']:.4f}")
    # dev EER threshold applied to eval (legit: threshold picked on dev only)
    dev_eer_thr = out["dev"]["eer_threshold"]
    acc_d, f1_d, far_d, frr_d = far_frr_at(store["eval_y_true"], store["eval_y_score"], dev_eer_thr)
    out["eval"]["at_DEV_eer_threshold"] = {"threshold": dev_eer_thr, "accuracy": acc_d, "f1": f1_d,
                                            "FAR": far_d, "FRR": frr_d}
    print(f"  DEV EER thr (={dev_eer_thr:.4f}) applied to eval [LEGIT]  acc={acc_d:.4f}  f1={f1_d:.4f}  "
          f"FAR={far_d:.4f}  FRR={frr_d:.4f}")

    block("PER-ATTACK  (eval, spoof system vs pooled bonafide)")
    print(f"{'atk':<6}{'n':>8}{'EER%':>9}{'AUC':>9}{'meanP(bona)':>13}{'miss@0.5':>11}")
    for a, d in sorted(per_attack.items(), key=lambda kv: kv[1]['eer_vs_pooled_bonafide']):
        print(f"{a:<6}{d['n_spoof']:>8}{d['eer_vs_pooled_bonafide']*100:>9.2f}{d['auc_vs_pooled_bonafide']:>9.4f}"
              f"{d['mean_score_Pbonafide']:>13.3f}{d['miss_rate_at_0.5']*100:>10.1f}%")

    block("ERROR ANALYSIS  (eval, argmax thr=0.5)")
    ea = out["error_analysis_eval_argmax0.5"]
    print(f"  FP (spoof->bonafide) : {ea['FP_spoof_as_bonafide']}   rate over spoof   = {ea['FP_rate_over_spoof']:.4f}")
    print(f"  FN (bonafide->spoof) : {ea['FN_bonafide_as_spoof']}   rate over bonafide = {ea['FN_rate_over_bonafide']:.4f}")
    print(f"  correct-spoof mean P(bona) = {ea['correct_spoof_mean_score']:.3f}   "
          f"wrong-spoof mean P(bona) = {ea['wrong_spoof_mean_score']:.3f}")
    print("  FP share by attack:")
    for a, pct in sorted(ea["FP_by_attack_pct_of_all_FP"].items(), key=lambda kv: -kv[1]):
        print(f"    {a:<5} {ea['FP_by_attack'][a]:>6}  ({pct}% of all FP)")

    (OUT / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'analysis.json'} and {OUT/'scores.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
