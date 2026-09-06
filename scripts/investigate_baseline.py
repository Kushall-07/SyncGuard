"""One-off Phase 4 baseline investigation: leakage, protocol, duplication checks.

Pure manifest + filesystem analysis (no model). Writes findings to stdout and
a JSON blob under outputs/analysis/spoof-cnn-baseline/leakage_protocol.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.asvspoof_dataset import PROTOCOL_FILES, parse_protocol_file
from src.data.manifests import Manifest

MANIFEST = REPO_ROOT / "data" / "asvspoof" / "manifest.csv"
LA_ROOT = REPO_ROOT / "data" / "asvspoof" / "LA"
OUT = REPO_ROOT / "outputs" / "analysis" / "spoof-cnn-baseline"

# Official ASVspoof 2019 LA CM statistics (Wang et al. 2020, Table 2 / dataset docs).
OFFICIAL = {
    "train": {"total": 25380, "bonafide": 2580, "spoof": 22800, "attacks": {"A01", "A02", "A03", "A04", "A05", "A06"}},
    "dev":   {"total": 24844, "bonafide": 2548, "spoof": 22296, "attacks": {"A01", "A02", "A03", "A04", "A05", "A06"}},
    "eval":  {"total": 71237, "bonafide": 7355, "spoof": 63882,
              "attacks": {f"A{i:02d}" for i in range(7, 20)}},  # A07..A19
}


def sha1_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m = Manifest.read_csv(MANIFEST)
    by_split = {s: m.split(s) for s in ("train", "dev", "eval")}
    findings: dict = {}

    print("=" * 70)
    print("1 + 3.  SPLIT COUNTS  (manifest vs official ASVspoof 2019 LA)")
    print("=" * 70)
    counts = {}
    for s, sub in by_split.items():
        lab = Counter(r.label_name for r in sub)
        atk = {r.attack for r in sub if r.label_name == "spoof"}
        spk = {r.speaker for r in sub}
        counts[s] = {
            "total": len(sub), "bonafide": lab["bonafide"], "spoof": lab["spoof"],
            "n_speakers": len(spk), "attacks": sorted(atk),
        }
        o = OFFICIAL[s]
        ok = (len(sub) == o["total"] and lab["bonafide"] == o["bonafide"] and lab["spoof"] == o["spoof"])
        print(f"\n{s.upper()}")
        print(f"  manifest : total={len(sub):>6}  bonafide={lab['bonafide']:>5}  spoof={lab['spoof']:>6}  speakers={len(spk)}")
        print(f"  official : total={o['total']:>6}  bonafide={o['bonafide']:>5}  spoof={o['spoof']:>6}")
        print(f"  MATCH    : {'YES' if ok else 'NO <-- DISCREPANCY'}")
        print(f"  attacks  : {sorted(atk)}")
        print(f"  official attacks: {sorted(o['attacks'])}  "
              f"{'(match)' if atk == o['attacks'] else '(DIFF: ' + str(sorted(atk ^ o['attacks'])) + ')'}")
    findings["counts"] = counts

    print("\n" + "=" * 70)
    print("1.  SPEAKER OVERLAP BETWEEN SPLITS")
    print("=" * 70)
    spk = {s: {r.speaker for r in sub} for s, sub in by_split.items()}
    spk_overlap = {}
    for a, b in combinations(("train", "dev", "eval"), 2):
        ov = spk[a] & spk[b]
        spk_overlap[f"{a}_INT_{b}"] = sorted(ov)
        print(f"  {a}_INT_{b} = {len(ov)}   {sorted(ov) if ov else ''}")
    findings["speaker_overlap"] = spk_overlap

    print("\n" + "=" * 70)
    print("1.  FILE-PATH OVERLAP BETWEEN SPLITS")
    print("=" * 70)
    paths = {s: [r.path for r in sub] for s, sub in by_split.items()}
    path_overlap = {}
    for a, b in combinations(("train", "dev", "eval"), 2):
        ov = set(paths[a]) & set(paths[b])
        path_overlap[f"{a}_INT_{b}"] = len(ov)
        print(f"  {a}_INT_{b} = {len(ov)}")
    findings["path_overlap"] = path_overlap

    print("\n" + "=" * 70)
    print("1.  SAMPLE-ID OVERLAP BETWEEN SPLITS")
    print("=" * 70)
    sids = {s: [r.sample_id for r in sub] for s, sub in by_split.items()}
    sid_overlap = {}
    for a, b in combinations(("train", "dev", "eval"), 2):
        ov = set(sids[a]) & set(sids[b])
        sid_overlap[f"{a}_INT_{b}"] = sorted(list(ov))[:10]
        print(f"  {a}_INT_{b} = {len(ov)}")
    findings["sample_id_overlap"] = sid_overlap

    print("\n" + "=" * 70)
    print("1.  DUPLICATE sample_id / path WITHIN THE WHOLE MANIFEST")
    print("=" * 70)
    all_sid = Counter(r.sample_id for r in m)
    all_path = Counter(r.path for r in m)
    dup_sid = {k: c for k, c in all_sid.items() if c > 1}
    dup_path = {k: c for k, c in all_path.items() if c > 1}
    print(f"  duplicate sample_id : {len(dup_sid)}  {list(dup_sid.items())[:5]}")
    print(f"  duplicate path      : {len(dup_path)}  {list(dup_path.items())[:5]}")
    findings["duplicate_sample_id"] = len(dup_sid)
    findings["duplicate_path"] = len(dup_path)

    print("\n" + "=" * 70)
    print("2.  MANIFEST vs OFFICIAL PROTOCOL FILES (row-for-row)")
    print("=" * 70)
    proto_check = {}
    for s in ("train", "dev", "eval"):
        entries = parse_protocol_file(LA_ROOT / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILES[s], s)
        proto_ids = {e["utterance"] for e in entries}
        man_ids = {r.sample_id for r in by_split[s]}
        proto_labels = {e["utterance"]: e["key"] for e in entries}
        man_labels = {r.sample_id: r.label_name for r in by_split[s]}
        mismatch = [u for u in (proto_ids & man_ids) if proto_labels[u] != man_labels[u]]
        proto_check[s] = {
            "protocol_rows": len(entries),
            "manifest_rows": len(man_ids),
            "in_protocol_not_manifest": len(proto_ids - man_ids),
            "in_manifest_not_protocol": len(man_ids - proto_ids),
            "label_mismatches": len(mismatch),
        }
        print(f"  {s}: protocol={len(entries)}  manifest={len(man_ids)}  "
              f"proto-only={len(proto_ids - man_ids)}  man-only={len(man_ids - proto_ids)}  "
              f"label_mismatch={len(mismatch)}")
    findings["protocol_check"] = proto_check

    print("\n" + "=" * 70)
    print("3.  DUPLICATE AUDIO CONTENT ACROSS SPLITS  (sha1 of raw .flac bytes)")
    print("=" * 70)
    print("  hashing all files (this takes a few minutes)...")
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    n = 0
    for s in ("train", "dev", "eval"):
        for r in by_split[s]:
            p = Path(r.path)
            if not p.is_file():
                continue
            hashes[sha1_of(p)].append((s, r.sample_id))
            n += 1
            if n % 20000 == 0:
                print(f"    hashed {n} files...")
    cross = {h: v for h, v in hashes.items() if len({s for s, _ in v}) > 1}
    within = {h: v for h, v in hashes.items() if len(v) > 1 and len({s for s, _ in v}) == 1}
    print(f"  files hashed                  : {n}")
    print(f"  identical content ACROSS splits: {len(cross)} hash groups")
    for h, v in list(cross.items())[:10]:
        print(f"      {h[:12]}  {v}")
    print(f"  identical content WITHIN a split: {len(within)} hash groups")
    for h, v in list(within.items())[:10]:
        print(f"      {h[:12]}  {v[:6]}")
    findings["audio_hash"] = {
        "files_hashed": n,
        "cross_split_dup_groups": len(cross),
        "cross_split_examples": {h[:12]: v for h, v in list(cross.items())[:20]},
        "within_split_dup_groups": len(within),
        "within_split_examples": {h[:12]: v[:6] for h, v in list(within.items())[:20]},
    }

    (OUT / "leakage_protocol.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'leakage_protocol.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
