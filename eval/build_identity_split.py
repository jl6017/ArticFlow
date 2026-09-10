"""Construct a genuine HELD-OUT-IDENTITY split, and refuse to emit a bad one.

Every split shipped in this tree is a POSE split: 100% of object identities appear on both
sides, in different articulations. So an identity split cannot be made by slicing shards or
files — it must filter on `anno_id`, and it must be VERIFIED, because the failure mode is
silent: a split built by slicing is just another pose split, the two reference sets then
measure the same thing, the generalization gap comes out near zero, and that reads as
"generalizes to unseen identities" when nothing was held out at all.

Emits three index files (no cloud data is copied):
  train_ids        the identities the model may train on
  ref_heldout      unseen IDENTITIES, unseen poses   <- the number the AC asked for
  ref_seen         seen identities, unseen POSES     <- the within-model control

The two references are MATCHED in identity count and cloud count. COV over S generated vs R
references is bounded by min(1, S/R), so unequal reference sizes produce a difference that is
an artefact of the protocol rather than of generalization.

  python analysis/build_identity_split.py --cat Eyeglasses --hold 10 --seed 0
"""
import argparse, glob, json, os
import numpy as np

RSS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = {
    "Eyeglasses": os.path.join(RSS, "dataset/Eyeglasses"),
    "Pliers":     os.path.join(RSS, "dataset/Pliers"),
    "Scissors":   "/media/jiong/T7/dataset-articflow/Scissors_POSE_SPLIT/Scissors",
}


def load(root, split):
    import h5py
    ids, rows = [], []
    for f in sorted(glob.glob(os.path.join(root, split, "*.h5"))):
        with h5py.File(f, "r") as h:
            a = h["anno_id"][:]
            for i, x in enumerate(a):
                ids.append(x.decode() if isinstance(x, bytes) else str(x))
                rows.append((os.path.relpath(f, root), i))
    return np.array(ids), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="Eyeglasses")
    ap.add_argument("--hold", type=int, default=10, help="identities to hold out (~15%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = ROOTS[a.cat]
    out = a.out or os.path.join(RSS, f"rebuttal/idsplit_{a.cat.lower()}")
    os.makedirs(out, exist_ok=True)

    tr_ids, tr_rows = load(root, "train")
    te_ids, te_rows = load(root, "test")
    uniq = sorted(set(tr_ids.tolist()) | set(te_ids.tolist()))
    print(f"[split] {a.cat}: {len(uniq)} identities, {len(tr_ids)} train clouds, {len(te_ids)} test clouds")
    print(f"[split] shipped split shares {len(set(tr_ids.tolist()) & set(te_ids.tolist()))}"
          f"/{len(uniq)} identities -> it is a POSE split, as expected")

    rng = np.random.default_rng(a.seed)
    held = sorted(rng.choice(uniq, size=a.hold, replace=False).tolist())
    train_ids = [u for u in uniq if u not in held]
    # control identities: a matched-size sample OF THE TRAINING identities
    seen_ref_ids = sorted(rng.choice(train_ids, size=a.hold, replace=False).tolist())

    # ---- the assertions that make this a real experiment ---------------------------------
    assert not (set(train_ids) & set(held)), "FATAL: train and held-out identities overlap"
    assert len(held) == len(seen_ref_ids), "FATAL: reference sets differ in identity count"
    assert set(seen_ref_ids) <= set(train_ids), "FATAL: control ids must be training identities"

    # reference clouds come from the TEST poses (unseen articulations) in both cases,
    # so the only difference between the two references is identity seen vs unseen.
    def pick(idlist):
        return [(f, i) for (f, i), t in zip(te_rows, te_ids) if t in set(idlist)]
    ref_heldout = pick(held)
    ref_seen = pick(seen_ref_ids)
    n = min(len(ref_heldout), len(ref_seen))
    idx_h = rng.permutation(len(ref_heldout))[:n]
    idx_s = rng.permutation(len(ref_seen))[:n]
    ref_heldout = [ref_heldout[i] for i in sorted(idx_h)]
    ref_seen = [ref_seen[i] for i in sorted(idx_s)]
    assert len(ref_heldout) == len(ref_seen) == n, "FATAL: reference sets differ in cloud count"

    train_rows = [(f, i) for (f, i), t in zip(tr_rows, tr_ids) if t in set(train_ids)]
    rec = dict(category=a.cat, root=root, seed=a.seed,
               n_identities=len(uniq), held_out_ids=held, train_ids=train_ids,
               control_ids=seen_ref_ids,
               n_train_clouds=len(train_rows), n_ref_each=n,
               ref_heldout=ref_heldout, ref_seen=ref_seen, train_rows=train_rows)
    json.dump(rec, open(os.path.join(out, "split.json"), "w"), indent=2)

    print(f"[split] held out {len(held)} identities: {', '.join(held[:6])}{' ...' if len(held)>6 else ''}")
    print(f"[split] train on {len(train_ids)} identities / {len(train_rows)} clouds")
    print(f"[split] ref_heldout {n} clouds from {len(held)} UNSEEN identities")
    print(f"[split] ref_seen    {n} clouds from {len(seen_ref_ids)} SEEN identities (unseen poses)")
    print(f"[split] VERIFIED: identity overlap 0, reference sets matched at {n} clouds each")
    print(f"[split] -> {out}/split.json")


if __name__ == "__main__":
    main()
