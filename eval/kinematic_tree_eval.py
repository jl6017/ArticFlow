"""Derive the kinematic tree + kinematic-consistency metrics from probe npz files.

Input: <probe_dir>/<shape>.npz from kinematic_probe.py
(clouds [J, S, N, 3] index-corresponded across the whole probe, base [N, 3]).

Key facts exploited:
  - One-joint-at-a-time sweep => the moving set M_j is ONE rigid body (all joints
    distal to j are frozen), so no fine segmentation is needed for metrics.
  - Correspondence jitter accumulates over large action jumps => fit rigid motion
    INCREMENTALLY between consecutive sweep steps and compose.
  - The static complement of M_j is motionless by construction => its apparent
    displacement/Kabsch residual is the per-point transport-jitter noise floor.

Per shape:
  1. Moving set M_j per dim (threshold adaptive to the measured noise floor).
  2. Active dims -> derived joint count; inactive dims -> padding inertness.
  3. Tree from asymmetric coverage: j descends from k if M_j is covered by M_k and
     M_k is larger; near-equal "moves-everything" pairs are flagged ambiguous and
     ordered by static-set size (more proximal = smaller static set).
  4. Per-joint metrics: composed rotation angle, per-step axis spread, rigidity
     excess over the noise floor, response curve (cum. angle vs commanded, R^2).
  5. Topology classification vs expected category topology.
"""
import os, sys, json, glob, argparse
import numpy as np

EXPECTED = {
    # components = expected # of connected components in the moving-set overlap
    # graph over ACTIVE dims (robust to proximal-pair ordering ambiguity)
    "arm":        dict(components=1),
    "dog":        dict(components=4, comp_size=3),
    "quadruped":  dict(components=4, comp_size=3),
    "eyeglasses": dict(components=2),
    "pliers":     dict(components=1),
    "scissors":   dict(components=1),
}

# The dim->tree assignment is FIXED by the dataset convention (zero-padding order,
# DH normalization), so evaluation places measurements onto the KNOWN template and
# scores consistency — no structure discovery needed. parent[i] = parent dim of
# dim i (-1 = base link).
TEMPLATE = {
    "arm":        [-1, 0, 1, 2, 3, 4, 5],                    # serial base->tip
    "dog":        [-1, 0, 1, -1, 3, 4, -1, 6, 7, -1, 9, 10],  # 4 legs x (hip->knee->ankle)
    "quadruped":  [-1, 0, 1, -1, 3, 4, -1, 6, 7, -1, 9, 10],
    "eyeglasses": [-1, -1],                                   # two independent hinges
    "pliers":     [-1],
    "scissors":   [-1],
}


def template_scores(M, disp, active, template, tau):
    """Score consistency of measured moving sets with the known category template.
    - edge_containment: per template edge parent->child, |M_c & M_p| / |M_c|
      (child's movers must be a subset of parent's movers)  -> mean over edges
    - branch_separation: max core-overlap between dims of DIFFERENT template
      branches (0 = perfectly decoupled limbs)"""
    n = len(template)
    edges = [(p, c) for c, p in enumerate(template) if p >= 0
             and p in active and c in active and p < n]
    edge_scores = {}
    for p, c in edges:
        edge_scores[f"{p}->{c}"] = float(
            np.logical_and(M[c], M[p]).sum() / max(M[c].sum(), 1))
    # branch id = walk template up to a root
    def root_of(i):
        while template[i] >= 0:
            i = template[i]
        return i
    branch = {j: root_of(j) for j in active}
    cross = 0.0
    for i in active:
        for j in active:
            if i < j and branch[i] != branch[j]:
                core_i = disp[i] > max(tau, 0.5 * np.percentile(disp[i], 95))
                core_j = disp[j] > max(tau, 0.5 * np.percentile(disp[j], 95))
                ov = np.logical_and(core_i, core_j).sum() / max(
                    min(core_i.sum(), core_j.sum()), 1)
                cross = max(cross, float(ov))
    return dict(
        edge_containment=edge_scores,
        edge_containment_mean=(float(np.mean(list(edge_scores.values())))
                               if edge_scores else None),
        cross_branch_overlap_max=cross if len(set(branch.values())) > 1 else None,
    )


def kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cq - R @ cp
    res = np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1)))
    return R, t, res


def rot_axis_angle(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-6:
        return np.zeros(3), 0.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    return (ax / n if n > 1e-9 else np.zeros(3)), ang


def trimmed_kabsch(P, Q, trim=0.9, iters=2):
    """Kabsch with residual trimming: refit on the best `trim` fraction —
    suppresses transport-jitter outliers in the correspondence."""
    idx = np.arange(P.shape[0])
    R, t, res = kabsch(P, Q)
    for _ in range(iters):
        r = np.linalg.norm(P[idx] @ R.T + t - Q[idx], axis=1)
        keep = idx[r <= np.quantile(r, trim)]
        if len(keep) < 20:
            break
        idx = keep
        R, t, res = kabsch(P[idx], Q[idx])
    return R, t, res


def stabilize_sweep(clouds_j, refine_frac=0.6, iters=2):
    """Remove per-step GLOBAL rigid drift (from per-frame centering in the training
    normalization) by aligning each step into the step-0 frame using a robust
    static-consensus rigid fit. Returns stabilized clouds and mean drift/step."""
    S = clouds_j.shape[0]
    out = clouds_j.copy()
    drifts = []
    for s in range(1, S):
        P, Q = out[s], out[s - 1]
        idx = np.arange(P.shape[0])
        for _ in range(iters + 1):
            R, t, _ = kabsch(P[idx], Q[idx])
            res = np.linalg.norm(P @ R.T + t - Q, axis=1)
            keep = res <= np.quantile(res, refine_frac)
            idx = np.where(keep)[0]
            if len(idx) < 100:
                break
        drifts.append(np.linalg.norm(t) + 2 * np.sin(rot_axis_angle(R)[1] / 2))
        out[s:] = out[s:] @ R.T + t  # carry the correction through the rest
    return out, float(np.mean(drifts)) if drifts else 0.0


def sweep_rotation_fits(clouds_j, idx, fold_deg=60.0):
    """Rotation estimation along a sweep, GT-validated across 9-172 deg totals.
    - If the estimated total rotation stays under fold_deg: DIRECT trimmed fits
      (0 -> s) at EVERY step (max SNR, full-resolution response curve).
    - Otherwise: chained checkpoints with ~15 deg segments (direct fits fold
      past ~90 deg).
    Returns (steps, composed [(R0s, t0s)] per step, axes_w [(axis, weight)],
    cum_rad signed cumulative rotation per step)."""
    S = clouds_j.shape[0]
    R1, _, _ = trimmed_kabsch(clouds_j[0][idx], clouds_j[1][idx])
    _, a1 = rot_axis_angle(R1)
    est_total = a1 * (S - 1)

    if est_total < np.radians(fold_deg):
        steps = list(range(1, S))
        fits = []
        for s in steps:
            R, t, _ = trimmed_kabsch(clouds_j[0][idx], clouds_j[s][idx])
            ax, ang = rot_axis_angle(R)
            fits.append((R, t, ax, ang))
        ref = fits[int(np.argmax([f[3] for f in fits]))][2]
        composed, axes_w, cum = [], [], []
        for R, t, ax, ang in fits:
            flip = ang > np.radians(2) and np.dot(ax, ref) < 0
            cum.append(-ang if flip else ang)
            if ang > np.radians(2):
                axes_w.append((-ax if flip else ax, ang))
            composed.append((R, t))
        return steps, composed, axes_w, np.array(cum)

    k = int(np.clip(round(np.radians(15.0) / max(a1, np.radians(0.75))), 1, S - 1))
    steps = list(range(k, S, k))
    if steps[-1] != S - 1:
        steps.append(S - 1)
    Rc, tc = np.eye(3), np.zeros(3)
    prev = 0
    composed, segs = [], []
    for s in steps:
        R, t, _ = trimmed_kabsch(clouds_j[prev][idx], clouds_j[s][idx])
        Rc, tc = R @ Rc, R @ tc + t
        composed.append((Rc.copy(), tc.copy()))
        segs.append(rot_axis_angle(R))
        prev = s
    ref = segs[int(np.argmax([abs(a) for _, a in segs]))][0]
    axes_w, sgn = [], []
    for ax, ang in segs:
        flip = ang > np.radians(2) and np.dot(ax, ref) < 0
        sgn.append(-ang if flip else ang)
        if ang > np.radians(2):
            axes_w.append((-ax if flip else ax, ang))
    return steps, composed, axes_w, np.cumsum(sgn)


def sweep_metrics(clouds_j, core, M, diag, sweep_vals):
    """Adaptive-stride chained fits on the CORE set (strong movers only — the
    soft fringe dilutes rotation). Cumulative angle = sum of signed SEGMENT
    angles (unwrapped, can exceed 180 deg); sign reference = largest segment."""
    S = clouds_j.shape[0]
    static = ~M
    idx = np.where(core)[0]
    if len(idx) < 30:
        idx = np.where(M)[0]
    steps, composed, axes_w, cum_rad = sweep_rotation_fits(clouds_j, idx)
    mov_res, sta_res = [], []
    for s in steps:
        _, _, res = trimmed_kabsch(clouds_j[0][idx], clouds_j[s][idx])
        mov_res.append(res / diag)
        if static.sum() >= 30:
            _, _, rs = trimmed_kabsch(clouds_j[0][static], clouds_j[s][static])
            sta_res.append(rs / diag)
    if not len(cum_rad):
        return dict(note="no fits")
    axes = [a for a, _ in axes_w]
    wts = [w for _, w in axes_w]
    cum = np.degrees(cum_rad)
    cmd = (sweep_vals[np.array(steps)] - sweep_vals[0]).astype(np.float64)
    g = float((cum @ cmd) / (cmd @ cmd)) if (cmd @ cmd) > 0 else 0.0
    ss_res = float(((cum - g * cmd) ** 2).sum())
    ss_tot = float(((cum - cum.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    ax_spread = 0.0
    mean_axis = np.zeros(3)
    if axes:
        A = np.stack(axes)
        w = np.array(wts)[:, None]
        mean_axis = (A * w).sum(0)
        mean_axis /= (np.linalg.norm(mean_axis) + 1e-12)
        if len(axes) >= 2:
            ax_spread = float(np.average(
                np.degrees(np.arccos(np.clip(A @ mean_axis, -1, 1))),
                weights=np.array(wts)))
    noise = float(np.mean(sta_res)) if sta_res else float("nan")
    mov = float(np.mean(mov_res))
    return dict(
        total_angle_deg=float(np.abs(cum).max()) if len(cum) else 0.0,
        end_angle_deg=float(cum[-1]) if len(cum) else 0.0,
        response_curve_deg=np.round(cum, 2).tolist(),
        axis=np.round(mean_axis, 3).tolist(),
        axis_spread_deg=ax_spread,
        rigidity_resid=mov,
        noise_floor=noise,
        rigidity_excess=(mov - noise) if noise == noise else mov,
        gain_deg=g, linearity_r2=float(r2),
    )


def derive(npz_path, move_thresh=0.03, active_frac=0.02, contain_frac=0.85,
           template=None):
    z = np.load(npz_path)
    clouds = z["clouds"].astype(np.float32)
    base = z["base"].astype(np.float32)
    sweep_vals = z["sweep_vals"].astype(np.float32)
    J, S, N, _ = clouds.shape
    diag = float(np.linalg.norm(base.max(0) - base.min(0)))

    disp = np.linalg.norm(clouds - clouds[:, :1], axis=-1).max(axis=1)  # [J, N]
    # adaptive threshold: at least move_thresh*diag, at least 3x the jitter floor
    # (jitter floor estimated from the least-moving dim's lower displacement mass)
    floor = float(np.median(np.sort(disp, axis=1)[:, : max(N // 20, 10)].mean(axis=1).min()))
    tau = max(move_thresh * diag, 3.0 * floor)
    M = disp > tau
    frac = M.mean(axis=1)
    active = [j for j in range(J) if frac[j] > active_frac]
    inert = {int(j): float(disp[j].max() / diag) for j in range(J) if j not in active}

    # --- tree from asymmetric coverage ---
    parent, ambiguous = {}, []
    sizes = {j: int(M[j].sum()) for j in active}
    for j in active:
        containers = []
        for k in active:
            if k == j:
                continue
            cov = np.logical_and(M[j], M[k]).sum() / max(sizes[j], 1)
            if cov >= contain_frac and sizes[k] >= sizes[j]:
                containers.append((sizes[k], k))
        if containers:
            psize, p = min(containers)
            parent[j] = p
            if psize < 1.05 * sizes[j]:
                ambiguous.append((j, p))
        else:
            parent[j] = None
    # near-equal pairs: order by static-set size (proximal joint has fewer static pts)
    for j, p in ambiguous:
        if parent.get(p) == j:  # mutual containment -> break cycle
            if sizes[j] >= sizes[p]:
                parent[j] = None
            else:
                parent[p] = None

    children = {j: [c for c in active if parent[c] == j] for j in active}
    roots = [j for j in active if parent[j] is None]

    # --- motion cores: points above a fraction of the sweep's own peak motion.
    # The 3%-of-bbox set M captures core + soft non-local response; topology comes
    # from the cores, and the soft spread is reported as "leakage".
    core = {}
    leakage = {}
    for j in active:
        p95 = np.percentile(disp[j], 95)
        cthr = max(tau, 0.35 * p95)
        core[j] = disp[j] > cthr
        leakage[j] = float(1.0 - core[j].sum() / max(M[j].sum(), 1))

    fit_core = {j: disp[j] > max(tau, 0.5 * np.percentile(disp[j], 95))
                for j in active}
    metrics = {int(j): sweep_metrics(clouds[j], fit_core[j], M[j], diag, sweep_vals)
               for j in active}

    # --- connected components of the CORE overlap graph (robust topology) ---
    def components_at(factor):
        cores = {j: disp[j] > max(tau, factor * np.percentile(disp[j], 95))
                 for j in active}
        par = {j: j for j in active}
        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for i, j in [(a, b) for a in active for b in active if a < b]:
            ov = np.logical_and(cores[i], cores[j]).sum() / max(
                min(cores[i].sum(), cores[j].sum()), 1)
            if ov > 0.5:
                par[find(i)] = find(j)
        comps = {}
        for j in active:
            comps.setdefault(find(j), []).append(j)
        return sorted(comps.values(), key=lambda c: (-len(c), c))

    components = components_at(0.35)
    component_curve = {round(f, 2): len(components_at(f))
                       for f in [0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.8]}

    if not active:
        topo = "none"
    elif all(len(children[j]) == 0 for j in active):
        topo = "leaves"
    elif len(components) == 1 and all(len(children[j]) <= 1 for j in active):
        topo = "chain"
    elif len(components) >= 2:
        topo = f"forest({len(components)})"
    else:
        topo = "tree"

    tmpl = (template_scores(M, disp, active, template, tau)
            if template and len(template) == J else None)

    return dict(
        n_dims=J, n_points=N, sweep_steps=S, template=tmpl,
        active_joints=[int(j) for j in active], n_active=len(active),
        parent={str(j): (int(p) if p is not None else None) for j, p in parent.items()},
        roots=[int(r) for r in roots], ambiguous_pairs=[[int(a), int(b)] for a, b in ambiguous],
        topology=topo, components=[[int(j) for j in c] for c in components],
        component_curve={str(k): v for k, v in component_curve.items()},
        tau=float(tau / diag), jitter_floor=float(floor / diag),
        motion_leakage={int(j): round(leakage[j], 3) for j in active},
        padding_inert_maxdisp=inert, per_joint=metrics,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--category", default=None)
    ap.add_argument("--move_thresh", type=float, default=0.03)
    args = ap.parse_args()

    for path in sorted(glob.glob(os.path.join(args.probe_dir, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        rep = derive(path, move_thresh=args.move_thresh,
                     template=TEMPLATE.get(args.category))
        if args.category and args.category in EXPECTED:
            exp = EXPECTED[args.category]
            rep["expected_components"] = exp["components"]
            ok = len(rep["components"]) == exp["components"]
            if ok and "comp_size" in exp:
                ok = all(len(c) == exp["comp_size"] for c in rep["components"])
            rep["topology_match"] = bool(ok)
            # separation threshold: min core factor whose component count matches
            sep = [float(f) for f, n in rep["component_curve"].items()
                   if n == exp["components"]]
            rep["separation_threshold"] = min(sep) if sep else None
        with open(path.replace(".npz", "_tree.json"), "w") as f:
            json.dump(rep, f, indent=2)
        pj = rep["per_joint"]
        def avg(k):
            v = [m[k] for m in pj.values() if k in m and m[k] == m[k]]
            return float(np.mean(v)) if v else float("nan")
        t = rep.get("template") or {}
        tstr = ""
        if t:
            ec = t.get("edge_containment_mean")
            xb = t.get("cross_branch_overlap_max")
            tstr = (f" | tmplEdge={ec:.3f}" if ec is not None else "") + \
                   (f" xBranch={xb:.3f}" if xb is not None else "")
        print(f"{name}: active={rep['n_active']}/{rep['n_dims']} topo={rep['topology']}{tstr}"
              + (f" match={rep.get('topology_match')}" if "topology_match" in rep else "")
              + f" | angle={avg('total_angle_deg'):.0f}deg axisSpread={avg('axis_spread_deg'):.1f}deg"
              + f" rigidExcess={avg('rigidity_excess'):.4f} R2={avg('linearity_r2'):.3f}"
              + f" jitter={rep['jitter_floor']:.4f}"
              + (f" ambiguous={rep['ambiguous_pairs']}" if rep["ambiguous_pairs"] else ""))


if __name__ == "__main__":
    main()
