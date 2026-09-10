#!/usr/bin/env python3
"""Held-out-identity evaluation: score ONE generated set against TWO matched references.

The whole experiment is the GAP between two columns, so the protocol matters more than
either column:

* ONE sample set per seed, scored against both references. Sampling twice would inject
  sampling noise into precisely the quantity being measured.
* S = R. `one_nna` accepts unequal sizes, but then chance level is max(S,R)/(S+R) rather
  than 0.5 -- and "0.5 means indistinguishable" is the only reason to quote 1-NNA.
* `--max_ref` passed explicitly. eval_k3's default is 120, which would silently subsample
  a 214-cloud reference to a random 120 and break both the matching and the chance level.
* Three seeds, reported as mean +/- spread: at 65 clouds a single seed's 1-NNA moves more
  than the effect we are looking for.

    python -u analysis/eval_idsplit.py --category scissors --seeds 0 1 2
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable


def run_eval(ref_dir, sample_dir, label, max_ref, out_json, device="cuda", batch=4):
    """One eval_k3 invocation. Small batch: never evict a training run sharing the GPU."""
    cmd = [PY, "-u", str(ROOT / "analysis" / "eval_k3.py"),
           "--ref", str(ref_dir), "--samples", str(sample_dir), "--labels", label,
           "--max_ref", str(max_ref), "--out", str(out_json),
           "--device", device, "--batch", str(batch)]
    import os
    e = dict(os.environ)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=e)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"eval_k3 failed for {label} vs {ref_dir}")
    return json.loads(pathlib.Path(out_json).read_text())[label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--tag", default="", help="epoch tag, e.g. ep675; keeps runs from different checkpoints in separate files so one cannot overwrite another")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max_ref", type=int, default=0, help="0 = infer from the reference shard")
    ap.add_argument("--ckpt", default="", help="checkpoint that produced the samples; its "
                    "architecture is ASSERTED to be paper-matched and recorded in the output")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4, help="pairs per cdist call; keep small so "
                    "the eval can never evict a training run sharing the GPU")
    a = ap.parse_args()

    # A stale mis-configured checkpoint (ctx_dim 64, [128,256,256], 2048 pts, 20.81M params)
    # exists in this tree under a plausible name.  Evaluating it would produce perfectly
    # normal-looking MMD/COV/1-NNA with no indication the architecture was wrong -- the eval
    # output says nothing about the model.  So assert the architecture here rather than
    # trusting the path.
    arch = None
    if a.ckpt:
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        ca = ck["args"]; ca = vars(ca) if not isinstance(ca, dict) else ca
        arch = {"epoch": int(ck.get("epoch", -1)),
                "ctx_dim": ca["ctx_dim"], "ctx_stage_channels": list(ca["ctx_stage_channels"]),
                "tr_max_sample_points": ca["tr_max_sample_points"],
                "lf_use_joint_cond": bool(ca["lf_use_joint_cond"]),
                "adv_enable": bool(ca.get("adv_enable", False)),
                "pf_params_M": round(sum(v.numel() for v in ck["pf"].values()) / 1e6, 3)}
        bad = []
        if arch["ctx_dim"] != 16: bad.append(f"ctx_dim={arch['ctx_dim']} (paper: 16)")
        if arch["ctx_stage_channels"] != [80, 112, 112]:
            bad.append(f"ctx_stage_channels={arch['ctx_stage_channels']} (paper: [80,112,112])")
        if arch["tr_max_sample_points"] != 20000:
            bad.append(f"tr_max_sample_points={arch['tr_max_sample_points']} (paper: 20000)")
        if arch["lf_use_joint_cond"]:
            bad.append("lf_use_joint_cond=True (this experiment requires Uncond)")
        if bad:
            raise SystemExit("ABORT — checkpoint is not the paper-matched Uncond model:\n  "
                             + "\n  ".join(bad) + f"\n  ({a.ckpt})")
        print(f"[arch] OK: ep{arch['epoch']} ctx_dim=16 [80,112,112] 20000pts "
              f"Uncond {arch['pf_params_M']}M params")

    base = ROOT / "rebuttal" / f"idsplit_{a.category}"
    refs = {"heldout": base / "ref_heldout", "seen": base / "ref_seen"}

    if a.max_ref:
        max_ref = a.max_ref
    else:  # infer, so S=R can never drift from the actual shard
        import h5py, glob
        f = sorted(glob.glob(str(refs["heldout"] / "*.h5")))[0]
        with h5py.File(f, "r") as h:
            max_ref = (h["data_norm"] if "data_norm" in h else h["data"]).shape[0]
        n2 = None
        f2 = sorted(glob.glob(str(refs["seen"] / "*.h5")))[0]
        with h5py.File(f2, "r") as h:
            n2 = (h["data_norm"] if "data_norm" in h else h["data"]).shape[0]
        if n2 != max_ref:
            raise SystemExit(f"reference sizes differ ({max_ref} vs {n2}); S=R impossible")

    rows = []
    for s in a.seeds:
        gen = base / f"gen{a.tag}_s{s}"
        n_gen = len(list(gen.glob("*.ply")))
        if n_gen != max_ref:
            raise SystemExit(f"seed {s}: {n_gen} samples vs {max_ref} references -- S must equal R")
        rec = {"seed": s, "n_gen": n_gen, "max_ref": max_ref}
        for name, rd in refs.items():
            m = run_eval(rd, gen, f"{a.category}_s{s}_{name}", max_ref,
                         base / f"eval{a.tag}_s{s}_{name}.json", a.device, a.batch)
            rec[name] = {k: m[k] for k in ("mmd_cd", "cov", "one_nna")}
            print(f"  seed {s} vs {name:8s}: MMD {m['mmd_cd']:.6f}  COV {m['cov']:.4f}  1-NNA {m['one_nna']:.4f}")
        rows.append(rec)

    def agg(side, key):
        v = [r[side][key] for r in rows]
        return sum(v) / len(v), (max(v) - min(v))

    summary = {"category": a.category, "seeds": a.seeds, "n_gen": max_ref, "n_ref": max_ref,
               "action_file": sorted(str(x.relative_to(ROOT)) for x in base.glob("actions_*.npy")),
               "note": "one sample set per seed scored against both references; S=R",
               "per_seed": rows, "summary": {}, "architecture": arch,
               "checkpoint": a.ckpt or "(not asserted)"}
    print(f"\n{'metric':10s} {'held-out (unseen ids)':>24s} {'seen ids':>20s} {'gap':>14s}")
    for key, lbl in (("mmd_cd", "MMD-CD"), ("cov", "COV"), ("one_nna", "1-NNA")):
        hm, hs = agg("heldout", key)
        sm, ss = agg("seen", key)
        summary["summary"][key] = {"heldout_mean": hm, "heldout_spread": hs,
                                   "seen_mean": sm, "seen_spread": ss, "gap": hm - sm}
        print(f"{lbl:10s} {hm:>15.6f} +/-{hs:<7.6f} {sm:>11.6f} +/-{ss:<7.6f} {hm - sm:>+13.6f}")

    out = base / f"idsplit_result{a.tag}.json"
    # Anti-clobber: a single-seed smoke run must not silently replace a full multi-seed
    # result. This happened once -- a `--seeds 0` guard test overwrote a completed 3-seed
    # scissors result, including its paired analysis and caveats. The per-seed eval JSONs
    # survived so nothing was lost, but only by luck.
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            if len(prev.get("seeds", [])) > len(a.seeds):
                raise SystemExit(
                    f"REFUSING TO OVERWRITE {out.name}: it holds {len(prev['seeds'])} seeds "
                    f"{prev['seeds']} and this run has only {len(a.seeds)} {a.seeds}.\n"
                    f"  Re-run with the full seed set, or pass --out_suffix to write elsewhere.")
        except (json.JSONDecodeError, KeyError):
            pass
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
