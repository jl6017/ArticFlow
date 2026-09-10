"""Kinematic cross-validation, piece A: held-out-frame selection (GT-free).

Fit the screw (axis via registration or given candidate) on frames 0..k, extrapolate the
per-frame angle ramp one step, PREDICT frame k+1 by rotating frame 0's moving set, score by
symmetric chamfer to the actual frame k+1. Selects among candidates (direction, mask,
num_seg) by generalization instead of clearance or GT.

  python analysis/heldout_frame_select.py --npz <probe> --dim D --mask <npy> \
      [--axis ax ay az --angles a1,a2,...] ...
Used as a library by the CV driver.
"""
import numpy as np, torch

def rotm(a, th):
    a = a/ (np.linalg.norm(a)+1e-12)
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)

def chamfer(A, B, dev='cuda'):
    a = torch.as_tensor(A, device=dev, dtype=torch.float32).unsqueeze(0)
    b = torch.as_tensor(B, device=dev, dtype=torch.float32).unsqueeze(0)
    d = torch.cdist(a, b)**2
    return float((d.min(2)[0].mean()+d.min(1)[0].mean()).item())

def fit_angle(X0, Xs, axis, pos, span=90.0, coarse=2.0, fine=0.25, dev='cuda'):
    best, bth = 1e18, 0.0
    P0 = X0 - pos
    for th in np.arange(-span, span+1e-9, coarse):
        v = chamfer(P0 @ rotm(axis, np.radians(th)).T + pos, Xs, dev)
        if v < best: best, bth = v, th
    for th in np.arange(bth-coarse, bth+coarse+1e-9, fine):
        v = chamfer(P0 @ rotm(axis, np.radians(th)).T + pos, Xs, dev)
        if v < best: best, bth = v, th
    return bth, best

def heldout_score(C, mask, axis, pos, dev='cuda', n_max=1200):
    """Fit angles on frames 1..S-2, linear-extrapolate to S-1, score prediction.
    Returns (heldout_chamfer/diag, fit_residual/diag, angles)."""
    X = C[:, mask, :].astype(np.float32)
    if X.shape[1] > n_max:
        idx = np.random.default_rng(0).choice(X.shape[1], n_max, replace=False)
        X = X[:, idx, :]
    S = X.shape[0]
    diag = float(np.linalg.norm(C[0].max(0)-C[0].min(0)))
    angs, fits = [], []
    for s in range(1, S-1):
        th, r = fit_angle(X[0], X[s], axis, pos, dev=dev)
        angs.append(th); fits.append(r)
    # linear ramp extrapolation to the held-out last frame
    t = np.arange(1, S-1)
    A = np.polyfit(t, angs, 1)
    th_pred = float(np.polyval(A, S-1))
    pred = (X[0]-pos) @ rotm(axis, np.radians(th_pred)).T + pos
    ho = chamfer(pred, X[S-1], dev)
    return np.sqrt(ho)/diag, float(np.mean([np.sqrt(f) for f in fits]))/diag, angs
