"""Assembly rate over a BATCH of generated quadrupeds — turns one anecdote into a rate.

For every generated shape: recover joints, export an MJX-ready MJCF, and check it is a
well-formed articulated model. Reports sampled-vs-interpolated separately, because
"interpolated shapes are as good as sampled ones" is the answer to the memorisation
objection, and it has to be a rate over many shapes to mean anything.

Per shape we record:
  assembles      all 12 joints recovered and the MJCF compiles in MuJoCo
  ordering       legs with hip above knee above ankle (4 = all)
  foot_spread    max/min foot mass — 1.3x on ground truth, high values flag bad segmentation
  mass, links    sanity numbers

  python analysis/batch_assembly_stats.py --dir rebuttal/kinprobe5/dog --out rebuttal/k1_batch.json
"""
import os, sys, json, glob, argparse, subprocess, tempfile
import numpy as np

RSS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_AU = "/mnt/t7env/conda-envs/autourdf/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="rebuttal/k1_batch.json")
    ap.add_argument("--ref_mass", type=float, default=2.339)
    ap.add_argument("--ref_diag", type=float, default=0.796)
    ap.add_argument("--work", default=None, help="where to put per-shape joints/mjcf")
    a = ap.parse_args()
    import mujoco

    P = lambda p: p if os.path.isabs(p) else os.path.join(RSS, p)
    work = P(a.work) if a.work else os.path.join(RSS, "rebuttal/k1_work")
    os.makedirs(work, exist_ok=True)
    shapes = sorted(glob.glob(os.path.join(P(a.dir), "*.npz")))
    print(f"[k1] {len(shapes)} shapes from {a.dir}")

    rows = []
    for npz in shapes:
        tag = os.path.basename(npz).replace(".npz", "")
        kind = "interpolated" if tag.startswith("interp") else "sampled"
        jd = os.path.join(work, tag, "joints")
        md = os.path.join(work, tag, "mjcf")
        rec = dict(tag=tag, kind=kind, assembles=False, ordering=None,
                   foot_spread=None, mass=None, error=None)
        try:
            r = subprocess.run(
                [PY_AU, "-u", os.path.join(RSS, "autourdf_joints/geometric_joints.py"),
                 "--npz", npz, "--out", jd, "--template", "dog"],
                capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr else "joints failed")
            ordering = None
            for line in r.stdout.splitlines():
                if "legs correctly ordered" in line:
                    ordering = int(line.strip().split("/")[0].split()[-1])
            rec["ordering"] = ordering
            njoints = len(glob.glob(os.path.join(jd, "joint_*.json")))
            if njoints < 12:
                raise RuntimeError(f"only {njoints}/12 joints recovered")

            r = subprocess.run(
                [PY_AU, "-u", os.path.join(RSS, "autourdf_joints/export_icos_mjcf.py"),
                 "--npz", npz, "--joints", jd, "--out_dir", md,
                 "--ref_mass", str(a.ref_mass), "--ref_diag", str(a.ref_diag)],
                capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr else "export failed")

            m = mujoco.MjModel.from_xml_path(
                os.path.join(md, "scene_mjx_quadruped_flat_terrain.xml"))
            d = mujoco.MjData(m); mujoco.mj_forward(m, d)
            if m.nu != 12 or m.nq != 19:
                raise RuntimeError(f"bad model nu={m.nu} nq={m.nq}")
            feet = [m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"seg_{f}_2")]
                    for f in (0, 3, 10, 14)]
            rec.update(assembles=True, mass=float(sum(m.body_mass)),
                       foot_spread=float(max(feet) / max(min(feet), 1e-9)),
                       feet=[float(x) for x in feet])
        except Exception as e:                       # noqa: BLE001 — record, never abort the batch
            rec["error"] = str(e)[:200]
        rows.append(rec)
        print(f"  {tag:12s} {kind:12s} assembles={rec['assembles']} "
              f"order={rec['ordering']} spread="
              + (f"{rec['foot_spread']:.1f}x" if rec["foot_spread"] else "-")
              + (f"  ERR {rec['error']}" if rec["error"] else ""))

    def summarize(sel, name):
        n = len(sel)
        if not n:
            return None
        ok = [r for r in sel if r["assembles"]]
        sp = [r["foot_spread"] for r in ok if r["foot_spread"]]
        od = [r["ordering"] for r in sel if r["ordering"] is not None]
        s = dict(name=name, n=n, assembled=len(ok), rate=len(ok) / n,
                 mean_ordering=float(np.mean(od)) if od else None,
                 median_foot_spread=float(np.median(sp)) if sp else None)
        print(f"[k1] {name:14s} {len(ok)}/{n} assemble ({100*len(ok)/n:.0f}%), "
              f"mean legs ordered {s['mean_ordering']:.2f}/4, "
              f"median foot spread {s['median_foot_spread']:.1f}x"
              if sp else f"[k1] {name}: {len(ok)}/{n}")
        return s

    out = dict(rows=rows,
               all=summarize(rows, "all"),
               sampled=summarize([r for r in rows if r["kind"] == "sampled"], "sampled"),
               interpolated=summarize([r for r in rows if r["kind"] == "interpolated"],
                                      "interpolated"))
    op = P(a.out); os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=2)
    print(f"[k1] -> {op}")


if __name__ == "__main__":
    main()
