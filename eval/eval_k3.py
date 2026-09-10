"""K3 evaluation: set-level comparison of ArticFlow vs its controls.

The controls (one-stage FM, PVD-objective diffusion) carry no shape latent, so
per-instance reconstruction is undefined for them; the fair comparison is
distribution-level, exactly as the paper's PointFlow-Action baseline is scored:
MMD-CD, COV-CD and 1-NNA against the held-out split.

  python analysis/eval_k3.py --ref dataset/Pliers/test \
      --samples runs/cloud_k3/pliers_pvd/samples_eval runs/cloud_k3/pliers_fm/samples \
      --labels PVD FM --n_points 2048 --out rebuttal/k3_pliers.json

Self-contained chamfer (torch cdist) — no CUDA extension needed; runs on CPU if
no GPU is free.
"""
import os, sys, json, glob, argparse
import numpy as np
import torch


def load_ref(ref_dir, n_points, max_n=None, seed=0):
    import h5py
    files = sorted(glob.glob(os.path.join(ref_dir, "*.h5")))
    if not files:
        raise SystemExit(f"no h5 in {ref_dir}")
    out = []
    rng = np.random.default_rng(seed)
    for f in files:
        with h5py.File(f, "r") as h:
            key = "data_norm" if "data_norm" in h else "data"
            d = np.asarray(h[key])                      # (K, N, 3)
            for c in d:
                idx = rng.choice(len(c), n_points, replace=len(c) < n_points)
                out.append(c[idx])
    out = np.stack(out)
    if max_n and len(out) > max_n:
        out = out[rng.choice(len(out), max_n, replace=False)]
    return torch.from_numpy(out).float()


def load_samples(sample_dir, n_points, seed=0):
    from plyfile import PlyData
    files = sorted(glob.glob(os.path.join(sample_dir, "**", "*.ply"), recursive=True))
    if not files:
        raise SystemExit(f"no plys in {sample_dir}")
    rng = np.random.default_rng(seed)
    out = []
    for f in files:
        v = PlyData.read(f)["vertex"]
        c = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
        idx = rng.choice(len(c), n_points, replace=len(c) < n_points)
        out.append(c[idx])
    return torch.from_numpy(np.stack(out)).float()


def chamfer(a, b):
    """a (B,N,3), b (B,N,3) -> (B,) mean-squared bidirectional chamfer."""
    d = torch.cdist(a, b) ** 2
    return d.min(2)[0].mean(1) + d.min(1)[0].mean(1)


def pairwise_cd(S, R, dev, bs=8):
    M = torch.zeros(len(S), len(R))
    for i in range(0, len(S), bs):
        s = S[i:i + bs].to(dev)
        for j in range(0, len(R), bs):
            r = R[j:j + bs].to(dev)
            a = s.unsqueeze(1).expand(-1, len(r), -1, -1).reshape(-1, s.shape[1], 3)
            b = r.unsqueeze(0).expand(len(s), -1, -1, -1).reshape(-1, r.shape[1], 3)
            M[i:i + len(s), j:j + len(r)] = chamfer(a, b).view(len(s), len(r)).cpu()
    return M


def metrics(M):
    """MMD (ref-side min), COV (fraction of refs matched), 1-NNA."""
    mmd = M.min(0)[0].mean().item()
    cov = len(set(M.min(1)[1].tolist())) / M.shape[1]
    S, R = M.shape
    big = torch.tensor(float("inf"))
    Mss = pairwise_self = None
    return mmd, cov


def one_nna(M, Mss, Mrr):
    S, R = M.shape
    lab = torch.cat([torch.ones(S), torch.zeros(R)])
    D = torch.zeros(S + R, S + R)
    D[:S, :S] = Mss; D[:S, S:] = M; D[S:, :S] = M.t(); D[S:, S:] = Mrr
    D.fill_diagonal_(float("inf"))
    nn = D.argmin(1)
    return (lab[nn] == lab).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--max_ref", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="default: cuda when present. The pairwise CD is the whole cost and "
                         "it is O(N^2) in reference count, so on 214 references CPU is hours "
                         "and GPU is minutes -- prefer cuda even when a training job holds "
                         "most of the card, and drop --batch instead of falling back to CPU.")
    ap.add_argument("--batch", type=int, default=8,
                    help="pairs per cdist call. Peak memory is ~batch^2 * n_points^2 * 4B: "
                         "at n_points=2048 that is ~1.1GB for batch 8 and ~270MB for batch 4.")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    R = load_ref(a.ref, a.n_points, a.max_ref)
    print(f"[eval] reference: {tuple(R.shape)} from {a.ref} | device {dev}")
    Mrr = pairwise_cd(R, R, dev, bs=a.batch)
    res = {}
    for lab, sd in zip(a.labels, a.samples):
        S = load_samples(sd, a.n_points)
        M = pairwise_cd(S, R, dev, bs=a.batch)
        Mss = pairwise_cd(S, S, dev, bs=a.batch)
        mmd, cov = metrics(M)
        nna = one_nna(M, Mss, Mrr)
        res[lab] = dict(n_samples=len(S), mmd_cd=mmd, cov=cov, one_nna=nna)
        print(f"  {lab:>28s}: MMD-CD {mmd:.5f} | COV {cov:.3f} | 1-NNA {nna:.3f} "
              f"(n={len(S)})")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
        print("[eval] ->", a.out)


if __name__ == "__main__":
    main()
