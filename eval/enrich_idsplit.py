#!/usr/bin/env python3
"""Attach the paired analysis, interpretation limits and caveats to an idsplit result.

Separate from eval_idsplit.py on purpose: recomputing the metrics must not strip the
constraints that travel with them. Re-run this after any recomputation.

    python -u analysis/enrich_idsplit.py --category scissors --category eyeglasses
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORSE = {"mmd_cd": "higher", "cov": "lower", "one_nna": "higher"}
SIGN_TEST = {3: 0.125, 4: 0.0625, 5: 0.031, 6: 0.016}


def enrich(cat):
    p = ROOT / "rebuttal" / f"idsplit_{cat}" / "idsplit_result.json"
    d = json.loads(p.read_text())
    rows = d["per_seed"]
    n = len(rows)

    paired = {}
    for key, worse in WORSE.items():
        diffs = [r["heldout"][key] - r["seen"][key] for r in rows]
        n_worse = sum((x > 0) if worse == "higher" else (x < 0) for x in diffs)
        hm = sum(r["heldout"][key] for r in rows) / n
        sm = sum(r["seen"][key] for r in rows) / n
        paired[key] = {"per_seed_diff_heldout_minus_seen": diffs,
                       "mean_diff": sum(diffs) / n,
                       "heldout_mean": hm, "seen_mean": sm,
                       "ratio_heldout_over_seen": hm / sm if sm else None,
                       "seeds_where_heldout_is_worse": n_worse, "n_seeds": n,
                       "consistent": n_worse == n}
    paired["_why_paired"] = (
        "The SAME generated set is scored against both references, so each seed is a matched "
        "pair and the per-seed difference is the estimator. Comparing two independent means "
        "with their spreads can hide a sign reversal (it did, for scissors' COV).")
    p_val = SIGN_TEST.get(n)
    paired["_significance"] = (
        f"Sign test, one-tailed, {n}/{n} in one direction: p = {p_val}. "
        + ("Crosses 0.05." if p_val and p_val < 0.05 else
           f"CANNOT reach p<0.05 at n={n}; suggestive only."))

    d["paired_analysis"] = paired
    d["caveats"] = [
        f"{d['n_ref']} reference clouds per side; 120 is used elsewhere in this project.",
        f"{n} seeds.",
        "References drawn from the same pose pool; the only difference between the two sides "
        "is whether the identity was in training.",
        "Architecture asserted at eval time (paper-matched Uncond, 6.97M pf params).",
    ]
    d["FRAMING_REQUIRED"] = (
        "Report BOTH columns, never the gap alone: 1-NNA sits near 0.84-0.89 where 0.5 is "
        "ideal, so quoting only a delta when a reviewer can compute the absolute reads as "
        "concealment. Claim only that the seen-vs-unseen GAP is small/large; never that "
        "generation quality is high. Label every cell with its EPOCH -- three checkpoints "
        "is a trend if labelled and a mess if not.")
    p.write_text(json.dumps(d, indent=2))

    ep = (d.get("architecture") or {}).get("epoch", "?")
    print(f"{cat} (ep{ep}, {n} seeds, {d['n_ref']} refs): "
          + ", ".join(f"{k} {'consistent' if paired[k]['consistent'] else 'FLIPS'}"
                      for k in WORSE) + f" | {paired['_significance'].split(':')[1].strip()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", action="append", required=True)
    for c in ap.parse_args().category:
        enrich(c)
