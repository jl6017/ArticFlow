import os, argparse, glob, math
from typing import List, Optional, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

# ==== Models from this repo ====
# Keep imports aligned with the training script (models.py definitions).
from models import VelocityNet, HybridMLP, ConditionalLatentVelocityNet  # PF & LF  # noqa

# =========================
# EMA weight context manager (temporarily swaps in EMA parameters for eval/sampling)
# =========================
class use_ema_weights:
    def __init__(self, module: torch.nn.Module, ema_shadow: Optional[dict], enabled: bool = True):
        self.module, self.ema, self.enabled = module, ema_shadow, enabled
        self.saved_params, self.saved_bufs = {}, {}
    def __enter__(self):
        if (not self.enabled) or (self.ema is None): return self.module
        dev = next(self.module.parameters()).device
        for n, p in self.module.named_parameters(recurse=True):
            if p.dtype.is_floating_point and (n in self.ema):
                self.saved_params[n] = p.data.detach().clone()
                p.data.copy_(self.ema[n].to(device=dev, dtype=p.dtype))
        for n, b in self.module.named_buffers(recurse=True):
            if torch.is_tensor(b) and b.dtype.is_floating_point and (n in self.ema):
                self.saved_bufs[n] = b.data.detach().clone()
                b.data.copy_(self.ema[n].to(device=dev, dtype=b.dtype))
        return self.module
    def __exit__(self, exc_type, exc, tb):
        if (not self.enabled) or (self.ema is None): return
        for n, p in self.module.named_parameters(recurse=True):
            if n in self.saved_params: p.data.copy_(self.saved_params[n])
        for n, b in self.module.named_buffers(recurse=True):
            if n in self.saved_bufs: b.data.copy_(self.saved_bufs[n])

# =========================
# Rebuild PF / LF from a checkpoint (supports ckpts without LF / EMA)
# =========================
def _safe_load(module: torch.nn.Module, sd: dict, name="module"):
    """Load only keys that exist and have identical shapes; skip the rest and print a brief summary."""
    model_sd = module.state_dict()
    ok, skipped_shape, skipped_missing = {}, [], []
    for k, v in sd.items():
        if k in model_sd:
            if tuple(model_sd[k].shape) == tuple(v.shape):
                ok[k] = v.to(dtype=model_sd[k].dtype)
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            skipped_missing.append(k)
    module.load_state_dict(ok, strict=False)
    print(f"[{name}] loaded {len(ok)} / {len(model_sd)} tensors; "
          f"skip shape-mismatch: {len(skipped_shape)}, skip unknown keys: {len(skipped_missing)}")
    for k, shp_ckpt, shp_cur in skipped_shape[:6]:
        print(f"  - shape mismatch: {k}: ckpt {shp_ckpt} vs cur {shp_cur}")
    if len(skipped_shape) > 6:
        print("  ...")

def _infer_lf_cond_dim_from_sd(lf_sd: dict | None) -> int:
    """Infer LF cond dimension from c_proj.weight; treat in_features==1 as an unconditional placeholder."""
    if not isinstance(lf_sd, dict): return 0
    for k in ("c_proj.weight", "module.c_proj.weight"):
        if k in lf_sd and torch.is_tensor(lf_sd[k]):
            in_feat = lf_sd[k].shape[1]
            # Convention: unconditional LF uses Linear(in_features=1) as a placeholder.
            return 0 if in_feat == 1 else int(in_feat)
    return 0

def _infer_pf_joint_dim_from_sd(pf_sd: dict | None, latent_dim: int) -> int:
    """Infer PF joint dimension from c_proj.weight: in_features = latent_dim + joint_dim (or 1 as placeholder)."""
    if not isinstance(pf_sd, dict): return 0
    for k in ("c_proj.weight", "head.c_proj.weight", "module.c_proj.weight", "module.head.c_proj.weight"):
        if k in pf_sd and torch.is_tensor(pf_sd[k]):
            in_feat = pf_sd[k].shape[1]
            if in_feat == 1:    # unconditional placeholder
                return 0
            if in_feat >= latent_dim:
                return int(in_feat - latent_dim)
    return 0

def build_from_ckpt(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {}) or {}

    latent_dim   = int(a.get("latent_dim", 256))
    pf_point_dim = int(a.get("pf_point_dim", 3))
    # Recorded joint/cond dim from dataset/ckpt metadata (fallback reference).
    cond_dim_data = int(ckpt.get("cond_dim", a.get("cond_dim", 0)))

    # ===== Infer the effective dims from checkpoint weights first =====
    lf_cond_dim_w = _infer_lf_cond_dim_from_sd(ckpt.get("lf", None))
    pf_joint_dim_w = _infer_pf_joint_dim_from_sd(ckpt.get("pf", None), latent_dim=latent_dim)

    # Prefer weight-inferred dims; fall back to recorded cond_dim when unavailable.
    lf_cond_dim = lf_cond_dim_w if lf_cond_dim_w is not None else cond_dim_data
    pf_joint_dim = pf_joint_dim_w if pf_joint_dim_w is not None else cond_dim_data

    # ===== PF =====
    pf_backbone = str(a.get("pf_backbone", "mlp"))
    pf_width    = int(a.get("pf_width", 512))
    pf_depth    = int(a.get("pf_depth", 6))
    pf_emb_dim  = int(a.get("pf_emb_dim", 256))
    cfg_drop_p  = float(a.get("cfg_drop_p", 0.0))

    pf_cond_dim = latent_dim + max(0, int(pf_joint_dim))

    if pf_backbone == "mlp":
        pf = VelocityNet(cond_dim=pf_cond_dim, width=pf_width, depth=pf_depth,
                         emb_dim=pf_emb_dim, cfg_dropout_p=cfg_drop_p,
                         point_dim=pf_point_dim).to(device)
    else:
        ctx_dim         = int(a.get("ctx_dim", 64))
        ctx_emb_dim     = int(a.get("ctx_emb_dim", 256))
        stage_channels  = list(a.get("ctx_stage_channels", [128,256,256]))
        stage_blocks    = list(a.get("ctx_stage_blocks", [2,2,2]))
        stage_res       = list(a.get("ctx_stage_res", [32,16,8]))
        with_se         = bool(a.get("ctx_with_se", True))
        norm_type       = a.get("ctx_norm", "group")
        gn_groups       = int(a.get("ctx_gn_groups", 32))
        with_global     = bool(a.get("ctx_with_global", True))
        voxel_normalize = bool(a.get("ctx_voxel_normalize", True))
        t_gate_tau      = float(a.get("ctx_t_gate_tau", a.get("t_gate_tau", 0.8)))
        t_gate_k        = float(a.get("ctx_t_gate_k",   a.get("t_gate_k", 10.0)))
        pf = HybridMLP(
            cond_dim=pf_cond_dim, point_dim=pf_point_dim,
            ctx_dim=ctx_dim, ctx_emb_dim=ctx_emb_dim,
            stage_channels=stage_channels, stage_blocks=stage_blocks, stage_res=stage_res,
            with_se=with_se, norm_type=norm_type, gn_groups=gn_groups,
            with_global=with_global, voxel_normalize=voxel_normalize,
            use_t_gate=True, t_gate_k=t_gate_k, t_gate_tau=t_gate_tau,
            pf_width=pf_width, pf_depth=pf_depth, pf_emb_dim=pf_emb_dim,
            cfg_dropout_p=cfg_drop_p
        ).to(device)

    # Expose z_dim so guided_velocity() can zero out only the joint part (keep z).
    try: pf.z_dim = latent_dim
    except: pass

    # ===== LF =====
    lf_width   = int(a.get("lf_width", 512))
    lf_depth   = int(a.get("lf_depth", 6))
    lf_emb_dim = int(a.get("lf_emb_dim", 256))
    lf = ConditionalLatentVelocityNet(latent_dim, cond_dim=int(lf_cond_dim),
                                      width=lf_width, depth=lf_depth, emb_dim=lf_emb_dim).to(device)

    # ===== Safely load weights =====
    if "pf" in ckpt:      _safe_load(pf, ckpt["pf"], name="pf")
    elif "model" in ckpt: _safe_load(pf, ckpt["model"], name="pf(model)")
    lf_loaded = False
    if "lf" in ckpt and isinstance(ckpt["lf"], dict) and len(ckpt["lf"]) > 0:
        _safe_load(lf, ckpt["lf"], name="lf"); lf_loaded = True

    info = dict(
        latent_dim=latent_dim, pf_point_dim=pf_point_dim,
        # Return the effective joint/cond dim for downstream printing and logic.
        cond_dim=int(max(lf_cond_dim, pf_joint_dim, cond_dim_data)),
        point_prior_std=float(a.get("point_prior_std", 1.0)),
        color_prior=str(a.get("color_prior", "gauss")),
        color_prior_std=float(a.get("color_prior_std", 1.0)),
        latent_prior_std=float(a.get("latent_prior_std", 1.0)),
        guidance_scale=float(a.get("guidance_scale", 0.0)),
        pf_joint_dim=int(pf_joint_dim),
        lf_cond_dim=int(lf_cond_dim),
    )
    ema_pf = ckpt.get("ema_pf", None) if isinstance(ckpt.get("ema_pf", None), dict) else None
    ema_lf = ckpt.get("ema_lf", None) if isinstance(ckpt.get("ema_lf", None), dict) else None
    return pf, lf, lf_loaded, info, ema_pf, ema_lf

# =========================
# Heun (RK2) integrators (PF and LF)
# =========================
@torch.no_grad()
def heun_integrate_pf(pf, x0, cond_full, steps: int, guidance_scale: float):
    x = x0; B = x.shape[0]; dt = 1.0 / max(1, int(steps))
    for k in range(max(1, int(steps))):
        t0 = torch.full((B,), k * dt,       device=x.device, dtype=x.dtype)
        v1 = pf.guided_velocity(x, t0, cond_full, guidance_scale=guidance_scale)
        x_hat = x + v1 * dt
        t1 = torch.full((B,), (k + 1) * dt, device=x.device, dtype=x.dtype)
        v2 = pf.guided_velocity(x_hat, t1, cond_full, guidance_scale=guidance_scale)
        x = x + 0.5 * dt * (v1 + v2)
    return x

@torch.no_grad()
def heun_integrate_lf(lf, y0, steps: int, cond=None):
    z = y0; B = z.shape[0]; dt = 1.0 / max(1, int(steps))
    for k in range(max(1, int(steps))):
        t0 = torch.full((B,), k * dt,       device=z.device, dtype=z.dtype)
        v1 = lf(z, t0, cond=cond)
        z_hat = z + v1 * dt
        t1 = torch.full((B,), (k + 1) * dt, device=z.device, dtype=z.dtype)
        v2 = lf(z_hat, t1, cond=cond)
        z = z + 0.5 * dt * (v1 + v2)
    return z

# =========================
# PF priors (3D / 6D)
# =========================
def make_pf_prior_like(B, N, D, point_prior_std: float, color_prior: str = "gauss",
                       color_prior_std: float = 1.0, device=None, dtype=None):
    if D == 3:
        x0 = torch.randn(B, N, 3, device=device, dtype=dtype) * point_prior_std
    else:
        x0 = torch.empty(B, N, D, device=device, dtype=dtype)
        x0[..., :3] = torch.randn(B, N, 3, device=device, dtype=dtype) * point_prior_std
        if color_prior == "gauss":
            x0[..., 3:] = torch.randn(B, N, D-3, device=device, dtype=dtype) * color_prior_std
        elif color_prior == "uniform":
            x0[..., 3:] = torch.rand(B, N, D-3, device=device, dtype=dtype)
        else:
            x0[..., 3:] = 0.0
    return x0

# =========================
# Spherical linear interpolation (SLERP)
# =========================
def slerp_path(z0: torch.Tensor, z1: torch.Tensor, num_points: int) -> List[torch.Tensor]:
    """
    Return `num_points` latents including both endpoints (t in [0, 1], uniform).
    Falls back to LERP when the angle is extremely small.
    """
    assert z0.shape == z1.shape and z0.ndim == 2 and z0.shape[0] == 1
    z0n = z0 / (z0.norm(dim=1, keepdim=True) + 1e-8)
    z1n = z1 / (z1.norm(dim=1, keepdim=True) + 1e-8)
    dot = (z0n * z1n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    zs = []
    for i, t in enumerate(np.linspace(0.0, 1.0, num_points, dtype=np.float32)):
        tt = torch.as_tensor([t], device=z0.device, dtype=z0.dtype).view(1, 1)
        if float(sin_theta.item()) < 1e-6:
            z = (1.0 - tt) * z0 + tt * z1
        else:
            s0 = torch.sin((1.0 - tt) * theta) / sin_theta
            s1 = torch.sin(tt * theta) / sin_theta
            # Also interpolate the radius linearly to avoid sudden norm changes.
            r = (1.0 - tt) * z0.norm(dim=1, keepdim=True) + tt * z1.norm(dim=1, keepdim=True)
            z = (s0 * z0n + s1 * z1n) * r
        zs.append(z)
    return zs  # list of (1, Dz)

# =========================
# Fixed-canvas rendering (optional RGB; stable frame size for mosaics)
# =========================
def render_topview_fixed(xyz: np.ndarray, outfile: str, plane: str = "xy",
                         rgb: Optional[np.ndarray] = None,
                         marker_size=0.35, bg="#0f172a",
                         fig_px: int = 800, dpi: int = 200):
    assert plane in ("xy", "xz", "yz")
    idx = {"xy": [0,1], "xz": [0,2], "yz": [1,2]}[plane]
    xy = xyz[:, idx].astype(np.float32)
    # Normalize to [-1, 1] based on the current span; keep fixed display limits for consistent frames.
    mn = xy.min(axis=0); mx = xy.max(axis=0)
    span = float(np.max(mx - mn));  span = 1.0 if span < 1e-8 else span
    xy = (xy - (mn + mx) / 2.0) / span

    fig_inches = fig_px / float(dpi)
    fig = plt.figure(figsize=(fig_inches, fig_inches), dpi=dpi)
    ax = plt.gca()
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.axis("off"); ax.set_aspect("equal", "box")
    ax.set_xlim([-1.05, 1.05]); ax.set_ylim([-1.05, 1.05])
    if rgb is None:
        ax.scatter(xy[:,0], xy[:,1], s=marker_size, c="w", linewidths=0.0, zorder=2)
    else:
        ax.scatter(xy[:,0], xy[:,1], s=marker_size, c=rgb, linewidths=0.0, zorder=2)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

# =========================
# Compose an N x M PNG grid
# =========================
def compose_grid(image_matrix: List[List[str]], out_png: str):
    assert len(image_matrix) > 0 and len(image_matrix[0]) > 0
    rows, cols = len(image_matrix), len(image_matrix[0])
    # Load all images and assert consistent sizes.
    imgs = [[Image.open(pth).convert("RGB") for pth in row] for row in image_matrix]
    H, W = imgs[0][0].size[1], imgs[0][0].size[0]
    for r in imgs:
        for im in r:
            assert im.size == (W, H), "Frame size mismatch; please ensure fixed renderer canvas."
    canvas = Image.new("RGB", (W*cols, H*rows), (16, 23, 42))  # background close to #0f172a
    for i in range(rows):
        for j in range(cols):
            canvas.paste(imgs[i][j], (j*W, i*H))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)
    print(f"[OK] wrote grid: {out_png}")

# =========================
# Write PLY (XYZ or XYZ+RGB)
# =========================
def write_ply(path: str, xyz: np.ndarray, rgb: Optional[np.ndarray] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = xyz.astype(np.float32)
    header = [
        "ply","format ascii 1.0",f"element vertex {xyz.shape[0]}",
        "property float x","property float y","property float z"
    ]
    if rgb is not None:
        header += ["property uchar red","property uchar green","property uchar blue"]
    header += ["end_header\n"]
    with open(path, "w") as f:
        f.write("\n".join(header))
        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            rgb255 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            for p, c in zip(xyz, rgb255):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser("Grid demo: SLERP in latent x linear joint (supports conditional/unconditional LF)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="grid_demo_out")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_points", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=200, help="PF Heun steps")
    ap.add_argument("--lf_steps", type=int, default=200, help="LF Heun steps")
    ap.add_argument("--latent_temp", type=float, default=1.0)
    ap.add_argument("--latent_mids", type=int, default=5, help="Number of intermediate SLERP steps (excluding endpoints); total cols = latent_mids + 2")
    ap.add_argument("--joint_mids",  type=int, default=5, help="Number of intermediate joint steps (excluding endpoints); total rows = joint_mids + 2")
    ap.add_argument("--joint_min", type=float, default=0.0)
    ap.add_argument("--joint_max", type=float, default=1.0)

    ap.add_argument("--views", nargs="+", default=["xy"], choices=["xy","xz","yz"])
    ap.add_argument("--frame_px", type=int, default=800)
    ap.add_argument("--ema_eval", action="store_true", default=True)
    ap.add_argument("--fps", type=int, default=12)

    # Choose whether latent-flow uses joint conditioning
    ap.add_argument("--lf_cond", type=str, default="auto", choices=["auto","joint","none"],
                    help="latent-flow conditioning: auto follows ckpt; joint forces joint cond; none forces unconditional")
    # Demo mode controls how the grid is constructed
    ap.add_argument("--demo", type=str, default="auto", choices=["auto","uncond","cond"],
                    help="auto inferred from lf_cond; uncond=unconditional latent; cond=joint-conditioned latent")

    # Whether to use per-point color when rendering PNG (PF must be 6D)
    ap.add_argument("--png_use_rgb", action="store_true", help="Render PNGs with per-point RGB (only when PF outputs RGB)")

    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load/build
    pf, lf, lf_loaded, shape, ema_pf, ema_lf = build_from_ckpt(args.ckpt, device=device)
    pf.eval(); lf.eval()
    print(f"[ckpt] latent_dim={shape['latent_dim']} pf_point_dim={shape['pf_point_dim']} cond_dim={shape['cond_dim']} lf_loaded={lf_loaded}")

    latent_dim    = int(shape["latent_dim"])
    pf_point_dim  = int(shape["pf_point_dim"])
    cond_dim      = int(shape["cond_dim"])
    point_std     = float(shape.get("point_prior_std", 1.0))
    color_prior   = str(shape.get("color_prior", "gauss"))
    color_std     = float(shape.get("color_prior_std", 1.0))
    latent_std    = float(shape.get("latent_prior_std", 1.0))
    gscale        = float(shape.get("guidance_scale", 0.0))

    # Decide whether LF should consume joint condition
    lf_has_cond = (cond_dim > 0) and lf_loaded and (getattr(lf, "cond_dim", cond_dim) > 0)
    if args.lf_cond == "joint": use_cond_lf = True
    elif args.lf_cond == "none": use_cond_lf = False
    else: use_cond_lf = bool(lf_has_cond)  # auto

    # Decide demo strategy
    if args.demo == "auto":
        demo_mode = "cond" if use_cond_lf else "uncond"
    else:
        demo_mode = args.demo
        if demo_mode == "cond" and not lf_has_cond:
            print("[WARN] The checkpoint latent-flow has no conditioning dimension; switching to 'uncond' mode.")
            demo_mode = "uncond"
    print(f"[Demo] mode={demo_mode}, lf_cond={'joint' if use_cond_lf else 'none'}")

    # Fixed noise endpoints
    B, N, D = 1, int(args.n_points), pf_point_dim
    y0_a = torch.randn(1, latent_dim, device=device) * (latent_std * float(args.latent_temp))
    y0_b = torch.randn(1, latent_dim, device=device) * (latent_std * float(args.latent_temp))
    x0   = make_pf_prior_like(B, N, D, point_std, color_prior, color_std, device=device, dtype=torch.float32)

    # Joint values for rows (including endpoints)
    J = int(args.joint_mids) + 2
    joint_vals = np.linspace(float(args.joint_min), float(args.joint_max), J, dtype=np.float32)

    # Latent columns: uncond uses one global SLERP; cond regenerates SLERP per row
    M = int(args.latent_mids) + 2

    out_root = os.path.join(args.out_dir, f"grid_{demo_mode}")
    os.makedirs(out_root, exist_ok=True)

    # ===== Prepare global latent columns (uncond) =====
    latent_cols_global: Optional[List[torch.Tensor]] = None
    if demo_mode == "uncond":
        with torch.no_grad():
            if lf_loaded:
                z_a = heun_integrate_lf(lf, y0_a.clone(), steps=int(args.lf_steps), cond=None)
                z_b = heun_integrate_lf(lf, y0_b.clone(), steps=int(args.lf_steps), cond=None)
            else:
                z_a, z_b = y0_a.clone(), y0_b.clone()
            latent_cols_global = slerp_path(z_a, z_b, M)

    # ===== Main loop: for each joint row, generate M columns =====
    # Collect per-view frames into a J x M grid, then compose mosaics
    view_grids = {v: [[None for _ in range(M)] for __ in range(J)] for v in args.views}

    with torch.no_grad():
        for i_row, jv in enumerate(joint_vals):
            # Joint conditioning vector for this row
            joint_vec = torch.full((1, cond_dim), float(jv), device=device, dtype=torch.float32) if cond_dim > 0 else None

            # Prepare latent columns for this row
            if demo_mode == "uncond":
                latent_cols = latent_cols_global
            else:
                # cond: run LF once per endpoint under this joint, then SLERP between the results
                if lf_loaded and use_cond_lf and (cond_dim > 0):
                    z_a = heun_integrate_lf(lf, y0_a.clone(), steps=int(args.lf_steps), cond=joint_vec)
                    z_b = heun_integrate_lf(lf, y0_b.clone(), steps=int(args.lf_steps), cond=joint_vec)
                elif lf_loaded and not use_cond_lf:
                    # Force unconditional LF (optional): ignore joint condition
                    z_a = heun_integrate_lf(lf, y0_a.clone(), steps=int(args.lf_steps), cond=None)
                    z_b = heun_integrate_lf(lf, y0_b.clone(), steps=int(args.lf_steps), cond=None)
                else:
                    z_a, z_b = y0_a.clone(), y0_b.clone()
                latent_cols = slerp_path(z_a, z_b, M)

            # Iterate over latent columns
            for j_col, z in enumerate(latent_cols):
                if cond_dim > 0:
                    cond_full = torch.cat([z, joint_vec], dim=1)
                else:
                    cond_full = z

                # PF sampling: start from the same x0 so differences come only from (z, joint)
                with use_ema_weights(pf, ema_pf, enabled=bool(args.ema_eval)):
                    x_gen = heun_integrate_pf(pf, x0.clone(), cond_full, steps=int(args.steps), guidance_scale=gscale)
                xyz = (x_gen[0, :, :3]).detach().cpu().numpy()
                rgb = None
                if D == 6 and args.png_use_rgb:
                    rgb = x_gen[0, :, 3:].detach().cpu().numpy()
                    rgb = np.clip(rgb, 0.0, 1.0)

                # Save PLY
                ply_dir = os.path.join(out_root, "ply")
                write_ply(os.path.join(ply_dir, f"r{i_row:02d}_c{j_col:02d}.ply"), xyz, rgb=rgb)

                # Save per-view PNG frames
                for v in args.views:
                    frame_dir = os.path.join(out_root, f"frames_{v}")
                    os.makedirs(frame_dir, exist_ok=True)
                    png_path = os.path.join(frame_dir, f"r{i_row:02d}_c{j_col:02d}.png")
                    render_topview_fixed(xyz, png_path, plane=v, rgb=rgb if args.png_use_rgb else None,
                                         fig_px=int(args.frame_px))
                    view_grids[v][i_row][j_col] = png_path

    # Compose the J x M grid for each view
    for v in args.views:
        grid_png = os.path.join(out_root, f"grid_{v}.png")
        compose_grid(view_grids[v], grid_png)

    print(f"[DONE] wrote grid demo to: {out_root}")

if __name__ == "__main__":
    main()