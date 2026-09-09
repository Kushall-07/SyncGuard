from pathlib import Path
import csv
import json
import numpy as np

root = Path("data/celebdf")
manifest = root / "manifest_dense.csv"
landmarks = root / "landmarks_dense"

rows = list(csv.DictReader(manifest.open(encoding="utf-8")))

print("=" * 60)
print("1. MANIFEST")
print("=" * 60)
print("Rows:", len(rows))

splits = {}
for r in rows:
    splits.setdefault(r["split"], []).append(r)

for s in ["train", "dev", "eval"]:
    rs = splits.get(s, [])
    real = sum(r["label"] == "1" for r in rs)
    fake = sum(r["label"] == "0" for r in rs)
    print(f"{s:5s}: {len(rs):4d} | real={real:4d} fake={fake:4d}")

print()
print("=" * 60)
print("2. LANDMARK CACHE")
print("=" * 60)

missing = []
bad_shape = []
bad_valid = []
bad_frame_idx = []

for r in rows:
    sid = r["sample_id"]
    p = landmarks / f"{sid}.npz"

    if not p.exists():
        missing.append(sid)
        continue

    try:
        with np.load(p) as d:
            points = d["points"]
            valid = d["valid"]
            frame_idx = d["frame_idx"]

            if points.shape != (32, 478, 3):
                bad_shape.append((sid, "points", points.shape))

            if valid.shape != (32,):
                bad_shape.append((sid, "valid", valid.shape))

            if frame_idx.shape != (32,):
                bad_shape.append((sid, "frame_idx", frame_idx.shape))

            if int(valid.sum()) < 4:
                bad_valid.append((sid, int(valid.sum())))

            if len(frame_idx) != 32 or not np.all(np.diff(frame_idx) == 1):
                bad_frame_idx.append(sid)

    except Exception as e:
        bad_shape.append((sid, "load_error", str(e)))

print("Missing .npz:", len(missing))
print("Bad shapes:", len(bad_shape))
print("Valid-frame violations:", len(bad_valid))
print("Non-contiguous frame indices:", len(bad_frame_idx))

if missing:
    print("  First missing:", missing[:5])
if bad_shape:
    print("  First bad:", bad_shape[:5])
if bad_valid:
    print("  First invalid:", bad_valid[:5])
if bad_frame_idx:
    print("  First non-contiguous:", bad_frame_idx[:5])

print()
print("=" * 60)
print("3. IDENTITY LEAKAGE")
print("=" * 60)

train_ids = {r["speaker"] for r in splits.get("train", [])}
dev_ids = {r["speaker"] for r in splits.get("dev", [])}
overlap = train_ids & dev_ids

print("Train identities:", len(train_ids))
print("Dev identities:", len(dev_ids))
print("Overlap:", len(overlap))

if overlap:
    print("OVERLAPPING IDS:", sorted(overlap))

print()
print("=" * 60)
print("4. AUDIT")
print("=" * 60)

audit = json.loads(
    (root / "manifest_dense.csv.audit.json").read_text(encoding="utf-8")
)

print("Excluded:", audit.get("n_excluded"))
print("Excluded by split:", audit.get("excluded_by_split"))
print("Excluded by reason:", audit.get("excluded_by_reason"))

print()
print("=" * 60)
print("5. LABEL CHECK")
print("=" * 60)

labels = sorted(set(r["label"] for r in rows))
label_names = sorted(set(r["label_name"] for r in rows))

print("Labels:", labels)
print("Label names:", label_names)

print()
print("=" * 60)
print("FINAL CHECK")
print("=" * 60)

ok = (
    len(rows) == 6429
    and not missing
    and not bad_shape
    and not bad_valid
    and not bad_frame_idx
    and not overlap
    and set(labels) == {"0", "1"}
)

print("PASS" if ok else "FAIL")
