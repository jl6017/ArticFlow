"""Nearest-neighbour visualizations: generated sample beside its closest training clouds.

Promised to the AC, R1 (W2) and R2 (W4): show what the model's outputs look like NEXT TO the
most similar real data, so novelty-vs-memorisation is inspectable rather than asserted.

For each selected generated sample: chamfer-nearest cloud in the FULL TRAINING SPLIT and in
ref_heldout (identities never trained on), rendered side by side with distances printed.
NOT ref_seen: that is a 10-of-55-identity subset built as a size-matched control, and NN
distance into a smaller candidate pool is systematically larger -- generated samples would
look more novel than they are, with the bias flattering us (tables agent's catch, #11).
Metric: the project's standard two-sided squared chamfer (same as eval_k3). The figure
caption must state the candidate count; the train-train NN baseline (--baseline_n) is what
makes any NN distance readable at all, and matches the paper's own Fig 6b red-dashed line. Selection is BY RANK over the whole sample set (best /
median / worst NN distance to ref_seen), not hand-picked -- a hand-picked panel is the
first thing a reviewer discounts.

CPU-only by design (torch cdist on 2048-pt subsets): must not touch the GPU while the
eyeglasses training holds it.

  python analysis/nn_viz.py --gen rebuttal/idsplit_eyeglasses/genep675_s0 \
      --refs rebuttal/idsplit_eyeglasses/ref_seen rebuttal/idsplit_eyeglasses/ref_heldout \
      --labels seen held-out --out rebuttal/nn_viz/eyeglasses_ep675.png
"""
import argparse, glob, os
import numpy as np

RSS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        n = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        if b"format ascii" in head:
            pts = np.loadtxt(path, skiprows=head.decode().count("\n"), max_rows=n)[:, :3]
        else:
            props = [l.split()[1:] for l in head.decode().splitlines() if l.startswith("property")]
            dt = np.dtype([(f"f{i}", {"double": "<f8", "float": "<f4", "uchar": "u1"}[t])
                           for i, (t, _) in enumerate(props)])
            rec = np.frombuffer(f.read(), dtype=dt, count=n)
            pts = np.stack([rec["f0"], rec["f1"], rec["f2"]], 1).astype(np.float64)
    return pts


def load_ref_h5(d, n_points, seed=0):
    import h5py
    rng = np.random.default_rng(seed)
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.h5"))):
        with h5py.File(f, "r") as h:
            key = "data_norm" if "data_norm" in h else "data"
            for c in np.asarray(h[key]):
                idx = rng.choice(len(c), n_points, replace=len(c) < n_points)
                out.append(c[idx][:, :3])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--n_points", type=int, default=1024,
                    help="1024 keeps 214x4407 CPU chamfers to ~10 min; GPU is off-limits "
                         "while the training holds it")
    ap.add_argument("--baseline_n", type=int, default=300,
                    help="train clouds sampled as queries for the train-train NN baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(max(os.cpu_count() - 4, 2))   # leave cores for the training loader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[nn] device {dev}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(a.seed)
    gfiles = sorted(glob.glob(os.path.join(RSS, a.gen, "*.ply")))
    G = []
    for f in gfiles:
        p = read_ply(f)
        idx = rng.choice(len(p), a.n_points, replace=len(p) < a.n_points)
        G.append(p[idx])
    G = torch.from_numpy(np.stack(G)).float()
    refs = {lab: torch.from_numpy(load_ref_h5(os.path.join(RSS, d), a.n_points)).float()
            for lab, d in zip(a.labels, a.refs)}
    print(f"[nn] {len(G)} generated vs " +
          ", ".join(f"{lab}:{len(r)}" for lab, r in refs.items()))

    def chamfer_row(g, R, bs=16):
        out = torch.zeros(len(R))
        g = g.to(dev)
        for i in range(0, len(R), bs):
            r = R[i:i + bs].to(dev)
            d = torch.cdist(g.unsqueeze(0).expand(len(r), -1, -1), r) ** 2
            out[i:i + bs] = (d.min(2)[0].mean(1) + d.min(1)[0].mean(1)).cpu()
        return out

    # rank all samples by NN distance into the SEEN reference; pick best / median / worst
    prim = a.labels[0]
    nn_d = torch.zeros(len(G))
    nn_i = torch.zeros(len(G), dtype=torch.long)
    for k in range(len(G)):
        row = chamfer_row(G[k], refs[prim])
        nn_d[k], nn_i[k] = row.min(0)
    # train-train NN baseline: each (sampled) training cloud vs its nearest OTHER one
    R0 = refs[prim]
    bidx = rng.choice(len(R0), min(a.baseline_n, len(R0)), replace=False)
    bl = torch.zeros(len(bidx))
    for t, k in enumerate(bidx):
        row = chamfer_row(R0[int(k)], R0)
        row[int(k)] = float("inf")
        bl[t] = row.min()
    print(f"[nn] train-train NN-CD baseline over {len(bidx)} queries x {len(R0)} candidates: "
          f"median {bl.median():.5f}, p10 {np.percentile(bl.numpy(),10):.5f}, "
          f"p90 {np.percentile(bl.numpy(),90):.5f}")
    order = nn_d.argsort()
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    names = ["best", "median", "worst"]
    print(f"[nn] rank by NN into '{prim}': best {nn_d[picks[0]]:.5f}, "
          f"median {nn_d[picks[1]]:.5f}, worst {nn_d[picks[2]]:.5f}")

    ncol = 1 + len(refs)
    fig = plt.figure(figsize=(3.1 * ncol, 3.1 * len(picks)), dpi=130)
    for r, (k, nm) in enumerate(zip(picks, names)):
        ax = fig.add_subplot(len(picks), ncol, r * ncol + 1, projection="3d")
        P = G[k].numpy()
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.6, c="#1f4e79")
        ax.set_title(f"generated ({nm} NN dist)\n{os.path.basename(gfiles[k])}", fontsize=8)
        ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
        for c, (lab, R) in enumerate(refs.items()):
            row = chamfer_row(G[k], R)
            d, i = row.min(0)
            ax = fig.add_subplot(len(picks), ncol, r * ncol + 2 + c, projection="3d")
            Q = R[int(i)].numpy()
            ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=0.6, c="#b34700")
            ax.set_title(f"NN in {lab}\nCD {float(d):.5f}", fontsize=8)
            ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    fig.suptitle(f"generated vs chamfer-nearest real (rank-selected best/median/worst, "
                 f"{len(refs[prim])} candidates searched; train-train NN-CD median "
                 f"{bl.median():.5f})", fontsize=9)
    fig.tight_layout()
    op = a.out if os.path.isabs(a.out) else os.path.join(RSS, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    fig.savefig(op); plt.close(fig)
    print("[nn] ->", op)


if __name__ == "__main__":
    main()
