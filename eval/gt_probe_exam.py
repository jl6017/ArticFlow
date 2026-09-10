"""GT exam for the kinematic probe: generate point-cloud sweeps from a REAL URDF
via forward kinematics, run the identical probe algorithms, score against the
URDF's analytic ground truth. The probe must pass here before any conclusion is
drawn about generated clouds.

generate: sample each link's collision mesh once (canonical, link frame), then for
each revolute joint sweep the joint one-at-a-time (others held at base config) and
assemble clouds by FK -> perfect point correspondence (like ArticFlow's shared-x0
sampling). Optional --noise adds per-frame Gaussian jitter to emulate transport
noise; --resample destroys correspondence (worst case). Saves probe-format npz +
gt.json (axes, anchors, per-point link labels, template).

evaluate: run kinematic_tree_eval.derive + kinematic_viz.screw_axis_anchor on the
npz and score: axis angular error, anchor-to-axis distance, moving-set IoU vs GT
distal subtree, cross-branch coupling floor, gain error. PASS thresholds printed.

Run with the autourdf env python (pybullet + open3d):
  python -u analysis/gt_probe_exam.py generate --urdf AutoURDF/Robot/franka/franka_panda.urdf --out rebuttal/gt_exam/franka
  python -u analysis/gt_probe_exam.py evaluate --dir rebuttal/gt_exam/franka
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------- generation -----------------------------

def rot_from_quat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def sample_link_meshes(p, body, n_total, urdf_dir):
    """Canonical per-link surface samples (link frame). Returns dict link->pts."""
    import open3d as o3d
    shapes = {}
    all_entries = list(p.getCollisionShapeData(body, -1))
    for li in range(p.getNumJoints(body)):
        all_entries.extend(p.getCollisionShapeData(body, li))
    for entry in all_entries:
        link = entry[1]
        gtype, fname = entry[2], entry[4].decode() if isinstance(entry[4], bytes) else entry[4]
        dims, lpos, lorn = entry[3], np.array(entry[5]), entry[6]
        shapes.setdefault(link, []).append((gtype, fname, dims, lpos, rot_from_quat(lorn)))

    meshes = {}
    areas = {}
    for link, items in shapes.items():
        tri = []
        for gtype, fname, dims, lpos, lR in items:
            m = None
            if fname:
                path = fname if os.path.isabs(fname) else os.path.join(urdf_dir, fname)
                if os.path.exists(path):
                    m = o3d.io.read_triangle_mesh(path)
                    if len(m.vertices) == 0:
                        m = None
                if m is not None and gtype == 5:  # GEOM_MESH: apply scale
                    m.scale(dims[0] if isinstance(dims, (list, tuple)) else 1.0,
                            center=np.zeros(3))
            if m is None and gtype == 2:  # sphere
                m = o3d.geometry.TriangleMesh.create_sphere(dims[0])
            if m is None and gtype == 3:  # box (dims = full extents)
                m = o3d.geometry.TriangleMesh.create_box(*dims)
                m.translate(-np.array(dims) / 2)
            if m is None and gtype == 4:  # cylinder dims=[len, radius]
                m = o3d.geometry.TriangleMesh.create_cylinder(dims[1], dims[0])
            if m is None:
                continue
            V = np.asarray(m.vertices) @ lR.T + lpos
            m2 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(V), m.triangles)
            tri.append(m2)
        if tri:
            merged = tri[0]
            for t in tri[1:]:
                merged += t
            meshes[link] = merged
            areas[link] = max(merged.get_surface_area(), 1e-9)

    total_area = sum(areas.values())
    pts = {}
    import open3d as o3d  # noqa
    for link, m in meshes.items():
        n = max(int(round(n_total * areas[link] / total_area)), 50)
        pc = m.sample_points_uniformly(number_of_points=n)
        pts[link] = np.asarray(pc.points)
    return pts


def generate(args):
    import pybullet as p
    p.connect(p.DIRECT)
    urdf = os.path.abspath(args.urdf)
    body = p.loadURDF(urdf, useFixedBase=True,
                      flags=p.URDF_MERGE_FIXED_LINKS if args.merge_fixed else 0)
    nj = p.getNumJoints(body)
    movable, template_parent, limits, axes_local = [], [], [], []
    link_parent = {}
    for ji in range(nj):
        info = p.getJointInfo(body, ji)
        link_parent[ji] = info[16]  # parent link index (-1 base)
        if info[2] == p.JOINT_REVOLUTE:
            movable.append(ji)
            lo, hi = info[8], info[9]
            if lo > hi:
                lo, hi = -1.0, 1.0
            limits.append((lo, hi))
            axes_local.append(np.array(info[13]))
    J = len(movable)
    # template: for each movable joint, its nearest movable ancestor
    def movable_ancestor(ji):
        pl = link_parent[ji]
        while pl >= 0:
            if pl in movable:
                return movable.index(pl)
            pl = link_parent[pl]
        return -1
    template = [movable_ancestor(ji) for ji in movable]

    base_q = [np.clip(0.5 * (lo + hi), lo, hi) for lo, hi in limits]
    span = []
    for k, (lo, hi) in enumerate(limits):
        half = min(args.range, (hi - lo) / 2 * 0.9)
        span.append((max(lo, base_q[k] - half), min(hi, base_q[k] + half)))

    canon = sample_link_meshes(p, body, args.n_points, os.path.dirname(urdf))
    links = sorted(canon.keys())
    labels = np.concatenate([[l] * len(canon[l]) for l in links])
    N = len(labels)

    nn_cache = {}

    def fk_cloud(q, rng=None):
        for k, ji in enumerate(movable):
            p.resetJointState(body, ji, q[k])
        out = np.zeros((N, 3))
        i = 0
        for l in links:
            if l == -1:
                pos, orn = p.getBasePositionAndOrientation(body)
            else:
                st = p.getLinkState(body, l, computeForwardKinematics=True)
                pos, orn = st[4], st[5]
            R = rot_from_quat(orn)
            k = len(canon[l])
            src = canon[l]
            if rng is not None and args.resample:
                src = src[rng.integers(0, k, k)]  # crude resample proxy
            if rng is not None and args.migrate > 0:
                # correspondence MIGRATION. Two models:
                # - local drift (default, k-NN swap): points slide to NEARBY
                #   surface positions — matches ArticFlow's small static-set
                #   residuals; severity = migrate fraction x neighborhood size
                # - teleport (--migrate_teleport): swap anywhere on the link
                #   (worst case; unrealistically harsh vs measured residuals)
                src = src.copy()
                m = max(int(args.migrate * k), 1)
                sel = rng.choice(k, m, replace=False)
                if args.migrate_teleport:
                    src[sel] = src[rng.permutation(sel)]
                else:
                    nn = nn_cache.get(l)
                    if nn is None:
                        from scipy.spatial import cKDTree
                        kq = min(args.migrate_knn + 1, k)
                        _, nn = cKDTree(canon[l]).query(canon[l], k=kq)
                        nn_cache[l] = nn
                    pick = nn[sel, rng.integers(1, nn.shape[1], size=len(sel))]
                    src[sel] = canon[l][pick]
            out[i:i + k] = src @ R.T + np.array(pos)
            i += k
        if rng is not None and args.noise > 0:
            out += rng.normal(0, args.noise, out.shape)
        return out

    S = args.sweep_steps
    rng = np.random.default_rng(0)
    clouds = np.zeros((J, S, N, 3), dtype=np.float16)
    sweep_vals = np.zeros((J, S), dtype=np.float32)
    for jdx in range(J):
        vals = np.linspace(*span[jdx], S)
        sweep_vals[jdx] = vals
        for s, v in enumerate(vals):
            q = list(base_q)
            q[jdx] = v
            clouds[jdx, s] = fk_cloud(q, rng)
    base_cloud = fk_cloud(base_q).astype(np.float16)

    # GT axes/anchors in world at base config
    for k, ji in enumerate(movable):
        p.resetJointState(body, ji, base_q[k])
    gt_axes, gt_anchor = [], []
    for k, ji in enumerate(movable):
        st = p.getLinkState(body, ji, computeForwardKinematics=True)
        pos, orn = np.array(st[4]), rot_from_quat(st[5])
        gt_axes.append((orn @ axes_local[k]).tolist())
        gt_anchor.append(pos.tolist())

    # distal subtree points per joint (GT moving sets)
    subtree = {}
    for k, ji in enumerate(movable):
        desc = set()
        frontier = {ji}
        while frontier:
            nxt = set()
            for l in links:
                if l >= 0 and link_parent.get(l) in frontier | desc and l not in desc:
                    nxt.add(l)
            desc |= frontier
            frontier = nxt - desc
            if not frontier:
                break
        desc.add(ji)
        subtree[k] = sorted(int(x) for x in desc)

    os.makedirs(args.out, exist_ok=True)
    # per-sweep npz uses that sweep's own values normalized 0..1 for eval cmd;
    # store radians separately in gt.json
    np.savez_compressed(os.path.join(args.out, "gt_model.npz"),
                        clouds=clouds, base=base_cloud,
                        sweep_vals=np.linspace(0, 1, S).astype(np.float32))
    gt = dict(urdf=urdf, n_joints=J, template=template,
              axes=gt_axes, anchors=gt_anchor,
              sweep_radians=sweep_vals.tolist(),
              labels=labels.tolist(), links=[int(l) for l in links],
              subtree={str(k): v for k, v in subtree.items()},
              base_q=list(map(float, base_q)), noise=args.noise, migrate=args.migrate,
              resample=bool(args.resample))
    with open(os.path.join(args.out, "gt.json"), "w") as f:
        json.dump(gt, f)
    print(f"[gt-exam] {os.path.basename(urdf)}: J={J} N={N} S={S} "
          f"noise={args.noise} -> {args.out}")
    p.disconnect()


# ----------------------------- evaluation -----------------------------

def evaluate(args):
    from kinematic_tree_eval import derive, kabsch, rot_axis_angle
    from kinematic_viz import screw_axis_anchor
    gt = json.load(open(os.path.join(args.dir, "gt.json")))
    npz_path = os.path.join(args.dir, "gt_model.npz")
    rep = derive(npz_path, template=gt["template"])
    z = np.load(npz_path)
    clouds = z["clouds"].astype(np.float32)
    base = z["base"].astype(np.float32)
    diag = float(np.linalg.norm(base.max(0) - base.min(0)))
    disp = np.linalg.norm(clouds - clouds[:, :1], axis=-1).max(axis=1)
    labels = np.array(gt["labels"])

    J = gt["n_joints"]
    print(f"\n=== GT EXAM: {os.path.basename(gt['urdf'])} | J={J} "
          f"noise={gt['noise']} resample={gt['resample']} ===")
    print(f"active detected: {rep['n_active']}/{J} "
          f"(missed: {sorted(set(range(J)) - set(rep['active_joints']))})")
    rows, fails = [], []
    for j in range(J):
        if j not in rep["active_joints"]:
            fails.append(f"joint {j} NOT DETECTED")
            continue
        core = disp[j] > max(rep["tau"] * diag, 0.5 * np.percentile(disp[j], 95))
        ax, anchor = screw_axis_anchor(clouds[j], disp[j], core, diag)
        gax = np.array(gt["axes"][j]); gax /= np.linalg.norm(gax)
        ganch = np.array(gt["anchors"][j])
        if ax is None:
            fails.append(f"joint {j}: no axis recovered")
            continue
        axis_err = np.degrees(np.arccos(np.clip(abs(float(ax @ gax)), 0, 1)))
        # anchor distance to GT axis LINE
        d = anchor - ganch
        anch_err = float(np.linalg.norm(d - gax * (d @ gax))) / diag
        # moving set IoU vs GT subtree (membership floor: 0.5% diag or 3x noise)
        gtM = np.isin(labels, gt["subtree"][str(j)])
        floor = max(0.005, 3.0 * gt["noise"] / diag)
        M = disp[j] > floor * diag
        iou = float(np.logical_and(M, gtM).sum() / max(np.logical_or(M, gtM).sum(), 1))
        # gain: recovered total angle vs commanded radians
        rads = np.array(gt["sweep_radians"][j])
        cmd_total = float(abs(rads[-1] - rads[0]))
        rec_total = np.radians(rep["per_joint"][j]["total_angle_deg"])
        gain_err = abs(rec_total - cmd_total) / max(cmd_total, 1e-9)
        rows.append((j, axis_err, anch_err, iou, gain_err))
        ok = axis_err < args.axis_deg and anch_err < args.anchor_frac \
            and iou > args.iou and gain_err < args.gain
        if not ok:
            fails.append(f"joint {j}: axis={axis_err:.1f}deg anchor={anch_err:.3f} "
                         f"IoU={iou:.2f} gainErr={gain_err:.2f}")
        print(f"  j{j}: axisErr={axis_err:5.1f}deg anchorDist={anch_err:.3f}diag "
              f"IoU={iou:.3f} gainErr={gain_err:.2%} "
              f"[gt axis {np.round(gax,2)} rec {np.round(ax,2)}]")
    xb = rep.get("template", {}) or {}
    if xb.get("cross_branch_overlap_max") is not None:
        print(f"  cross-branch overlap on GT (floor): {xb['cross_branch_overlap_max']:.3f}")
    if rows:
        a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
        print(f"  MEAN: axisErr={a[:,0].mean():.1f}deg anchor={a[:,1].mean():.3f} "
              f"IoU={a[:,2].mean():.3f} gainErr={a[:,3].mean():.2%}")
    verdict = "PASS" if not fails else "FAIL"
    print(f"  ==> {verdict}" + (f" ({len(fails)} issues)" if fails else ""))
    for f_ in fails:
        print("   -", f_)
    # Persist. These are the instrument-validation numbers the rebuttal quotes; printing them
    # only means a claim rests on scrollback, and a table cell cannot cite a terminal.
    out_json = getattr(args, "out_json", None) or os.path.join(args.dir, "gt_eval.json")
    rec = dict(verdict=verdict, dir=args.dir, n_joints=len(rows), fails=fails,
               per_joint=[dict(joint=int(r[0]), axis_err_deg=float(r[1]),
                               anchor_dist_diag=float(r[2]), iou=float(r[3]),
                               gain_err=float(r[4])) for r in rows])
    if rows:
        a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
        rec["mean"] = dict(axis_err_deg=float(a[:, 0].mean()),
                           anchor_dist_diag=float(a[:, 1].mean()),
                           iou=float(a[:, 2].mean()), gain_err=float(a[:, 3].mean()))
        rec["max"] = dict(axis_err_deg=float(a[:, 0].max()),
                          anchor_dist_diag=float(a[:, 1].max()),
                          gain_err=float(a[:, 3].max()))
    with open(out_json, "w") as f_:
        json.dump(rec, f_, indent=2)
    print(f"  [persisted] -> {out_json}")
    return verdict


def plot(args):
    """Render the SAME tree figure the ArticFlow probe uses, on GT data."""
    from kinematic_tree_eval import derive
    from kinematic_viz import tree_figure
    gt = json.load(open(os.path.join(args.dir, "gt.json")))
    npz_path = os.path.join(args.dir, "gt_model.npz")
    rep = derive(npz_path, template=gt["template"])
    z = np.load(npz_path)
    clouds = z["clouds"].astype(np.float32)
    base = z["base"].astype(np.float32)
    out = os.path.join(args.dir, "gt_tree.png")
    tree_figure(clouds, base, rep, gt["template"], out)
    print(f"[gt-exam] tree figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    pl = sub.add_parser("plot")
    pl.add_argument("--dir", required=True)
    pl.add_argument("--out_json", default=None,
                    help="where to persist the evaluation (default <dir>/gt_eval.json)")
    g = sub.add_parser("generate")
    g.add_argument("--urdf", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n_points", type=int, default=8192)
    g.add_argument("--sweep_steps", type=int, default=9)
    g.add_argument("--range", type=float, default=0.6, help="max half-span (rad)")
    g.add_argument("--noise", type=float, default=0.0, help="per-frame jitter (m)")
    g.add_argument("--migrate", type=float, default=0.0,
                   help="fraction of each link's points re-positioned on the same "
                        "link per frame (correspondence migration)")
    g.add_argument("--migrate_knn", type=int, default=20,
                   help="neighborhood size for local-drift migration")
    g.add_argument("--migrate_teleport", action="store_true",
                   help="teleport model (anywhere on link) instead of local drift")
    g.add_argument("--resample", action="store_true")
    g.add_argument("--merge_fixed", action="store_true", default=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--dir", required=True)
    e.add_argument("--axis_deg", type=float, default=0.1)
    e.add_argument("--anchor_frac", type=float, default=0.03)
    e.add_argument("--iou", type=float, default=0.9)
    e.add_argument("--gain", type=float, default=0.1)
    args = ap.parse_args()
    if args.mode == "generate":
        generate(args)
    elif args.mode == "plot":
        plot(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
