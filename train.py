from __future__ import annotations
import os, argparse
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch import distributed as dist
# Additional imports
from torch.utils.tensorboard import SummaryWriter
import json, random
import numpy as np

# ---- datasets / models / utils ----
from datasets import get_datasets, init_np_seed
from models import VelocityNet, HybridMLP, ConditionalLatentVelocityNet, ShapeEncoder, CondAdversary, GRL

from util import EMA, seed_all, init_distributed, cleanup_distributed, cosine_lr, \
                         save_point_cloud_ply, save_point_cloud_xyz, count_parameters
from util import save_point_cloud_ply_rgb

# ---- VAE encoder wrapper----
class VAEEncoder(nn.Module):
    """
    Wrap ShapeEncoder to output mu, logvar (size D each) by setting base latent_dim=2D.
    forward(x) -> (z_sample, {"mu":mu, "logvar":logvar, "kl":kl})
    """
    def __init__(self, latent_dim: int, width: int, depth: int, in_channels: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        # base encoder outputs 2*D
        self.base = ShapeEncoder(self.latent_dim * 2, width=width, depth=depth, in_channels=in_channels)
        self.is_vae = True

    def forward(self, x):
        h, _ = self.base(x)                 # (B, 2D)
        mu, logvar = torch.chunk(h, 2, dim=1)
        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        # KL per-sample, then mean over batch
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()
        return z, {"mu": mu, "logvar": logvar, "kl": kl}


# ========== EMA eval helper ==========
@contextmanager
def use_ema_weights(module: nn.Module, ema_shadow: dict | None, enabled: bool = True):
    """
    Temporarily override a module's floating-point params/buffers with EMA weights.
    Usage:
        with use_ema_weights(net_pf, ema_pf.shadow, enabled=True):
            # eval forward ...
    """
    if (not enabled) or (ema_shadow is None):
        yield module
        return

    device = next(module.parameters()).device
    saved_params, saved_bufs = {}, {}

    # Overwrite trainable parameters
    for name, p in module.named_parameters(recurse=True):
        if p.dtype.is_floating_point and (name in ema_shadow):
            saved_params[name] = p.data.detach().clone()
            p.data.copy_(ema_shadow[name].to(device=device, dtype=p.dtype))

    # Overwrite floating-point buffers (e.g., BN running_mean/var)
    for name, b in module.named_buffers(recurse=True):
        if torch.is_tensor(b) and b.dtype.is_floating_point and (name in ema_shadow):
            saved_bufs[name] = b.data.detach().clone()
            b.data.copy_(ema_shadow[name].to(device=device, dtype=b.dtype))

    try:
        yield module
    finally:
        # Restore
        for name, p in module.named_parameters(recurse=True):
            if name in saved_params:
                p.data.copy_(saved_params[name])
        for name, b in module.named_buffers(recurse=True):
            if name in saved_bufs:
                b.data.copy_(saved_bufs[name])


# ---- AMP helpers ----
_USE_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
def make_autocast(enabled: bool, use_bf16: bool):
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    if _USE_NEW_AMP:
        return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
    else:
        from torch.cuda.amp import autocast as _autocast
        return _autocast(enabled=enabled, dtype=dtype)
def make_scaler(enabled: bool):
    if _USE_NEW_AMP:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    else:
        from torch.cuda.amp import GradScaler as _GradScaler
        return _GradScaler(enabled=enabled)

# ---- metrics ----
@torch.no_grad()
def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Sum of bidirectional nearest-neighbor L2^2 (mean over batch)
    d2 = torch.cdist(pred, target, p=2).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)

def main():
    p = argparse.ArgumentParser("FM training (MLP / HybridMLP point-flow)")
    # ========== Data ==========
    p.add_argument("--dataset_type", type=str, default="partnet_h5", choices=["tdcr_h5","partnet_h5"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--tr_max_sample_points", type=int, default=2048)
    p.add_argument("--te_max_sample_points", type=int, default=2048)
    p.add_argument("--tdcr_use_norm", action="store_true", default=True)
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--train_subset_seed", type=int, default=0)

    # ========== Backbone & Models ==========
    # Point-flow backbone: mlp (baseline) or hybrid (ContextNet + per-point MLP)
    p.add_argument("--pf_backbone", type=str, default="mlp", choices=["mlp","hybrid"])

    # Shared hyperparameters (encoder / point-flow / latent-flow)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--enc_width", type=int, default=128)
    p.add_argument("--enc_depth", type=int, default=4)

    p.add_argument("--pf_width", type=int, default=512)
    p.add_argument("--pf_depth", type=int, default=6)
    p.add_argument("--pf_emb_dim", type=int, default=256)
    p.add_argument("--cfg_drop_p", type=float, default=0.1)

    p.add_argument("--lf_width", type=int, default=512)
    p.add_argument("--lf_depth", type=int, default=6)
    p.add_argument("--lf_emb_dim", type=int, default=256)
    # ========== VAE Switch ==========
    p.add_argument("--use_vae", action="store_true", default=False,
                help="Use VAE encoder (mu, logvar) + reparameterization; disable latent-flow")
    p.add_argument("--beta_kl", type=float, default=1e-3,
                help="Weight for KL(q(z|x)||N(0,I)) when --use_vae")
    p.add_argument("--kl_warmup_epochs", type=int, default=0,
                help="Linearly ramp beta from 0->beta_kl over these epochs (0 = no warmup)")
    # ========== One-stage baseline (K3): no shape latent ==========
    p.add_argument("--no_shape_latent", action="store_true", default=False,
                help="One-stage baseline: point-flow conditioned on the raw action vector only. "
                     "No shape encoder, no latent flow, no adversary; loss reduces to L_point (+color).")

    # ContextNet (hybrid backbone) hyperparameters
    p.add_argument("--ctx_dim", type=int, default=64)
    p.add_argument("--ctx_emb_dim", type=int, default=256)
    p.add_argument("--ctx_stage_channels", type=int, nargs="+", default=[128, 256, 256])
    p.add_argument("--ctx_stage_blocks", type=int, nargs="+", default=[2, 2, 2])
    p.add_argument("--ctx_stage_res", type=int, nargs="+", default=[32, 16, 8])
    p.add_argument("--ctx_with_se", action="store_true", default=True)
    p.add_argument("--ctx_norm", type=str, default="group", choices=["group","batch","syncbn","none"])
    p.add_argument("--ctx_gn_groups", type=int, default=32)
    p.add_argument("--ctx_with_global", action="store_true", default=True)
    p.add_argument("--ctx_voxel_normalize", action="store_true", default=True)  # strongly recommended: True

    # RGB options
    p.add_argument("--use_rgb_in_latent", action="store_true", default=True, help="Concatenate RGB into encoder input (if available).")
    p.add_argument("--pointflow_rgb", action="store_true", default=True, help="Learn/sample point-flow in 6D (xyz+rgb) if RGB is available.")

    # ========== Training ==========
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr_enc", type=float, default=3e-4)
    p.add_argument("--lr_pf", type=float, default=3e-4)
    p.add_argument("--lr_lf", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--use_cosine_lr", action="store_true", default=True)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--t_beta_a", type=float, default=2.0, help="t ~ Beta(a, 1). Larger a concentrates samples closer to 1 (default: 2.0).")
    p.add_argument("--geom_warmup_epochs", type=int, default=200, help="Warmup epochs for geometry-only training (RGB dims set to zero and excluded from loss).")
    # ========== FM priors ==========
    p.add_argument("--point_prior_std", type=float, default=1.0, help="Std of the Gaussian prior for XYZ.")
    p.add_argument("--latent_prior_std", type=float, default=1.0)
    p.add_argument("--color_prior", type=str, choices=["gauss","uniform","zeros"], default="gauss",
                   help="Initial RGB prior used in point-flow.")
    p.add_argument("--color_prior_std", type=float, default=1.0, help="Used when color_prior=gauss.")
 
    p.add_argument("--ctx_t_gate_tau", type=float, default=0.95, help="t-gate threshold (larger = enable PV context later).")
    p.add_argument("--ctx_t_gate_k", type=float, default=5.0, help="t-gate sharpness (sigmoid slope).")
    p.add_argument("--cfg_drop_warmup_epochs", type=int, default=3000, help="Warmup epochs for CFG cond-dropout (ramp 0 -> cfg_drop_p).")
    # ========== Latent-flow condition switch ==========
    g = p.add_mutually_exclusive_group()
    g.add_argument("--lf_use_joint_cond", dest="lf_use_joint_cond",
                action="store_true", default=True,
                help="Use joint condition in latent-flow (default: True).")
    g.add_argument("--lf_uncond", dest="lf_use_joint_cond",
                action="store_false",
                help="Disable joint condition in latent-flow (run unconditional LF).")

    # ========== Sampling / CFG / EMA ==========
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_eval", action="store_true", default=True)

    # ========== Loss ==========
    p.add_argument("--lambda_point", type=float, default=1.0)
    p.add_argument("--lambda_latent", type=float, default=1.0)
    p.add_argument("--lambda_color", type=float, default=1.0)

    # ========== System / I/O ==========
    p.add_argument("--out_dir", type=str, default="./runs/hybrid")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--vis_count", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--use_bf16", action="store_true", default=True)
    # TensorBoard options
    p.add_argument("--no_tb", action="store_true", help="Disable TensorBoard logging (enabled by default).")
    p.add_argument("--tb_log_dir", type=str, default=None, help="TensorBoard log dir (default: out_dir/tb).")
    p.add_argument("--tb_log_every", type=int, default=50, help="Log training scalars every N steps.")
    p.add_argument("--val_every", type=int, default=None,
              help="Run validation every N epochs (default: --save_every).")


    # ------- Adversarial invariance (z ⟂ joint) -------
    p.add_argument("--adv_enable", action="store_true",
                help="Enable adversarial head + GRL to remove joint info from z")
    p.add_argument("--adv_width", type=int, default=256)
    p.add_argument("--adv_depth", type=int, default=3)
    p.add_argument("--adv_dropout", type=float, default=0.1)
    p.add_argument("--lr_adv", type=float, default=3e-4)

    # Strength and schedule
    p.add_argument("--adv_lambda", type=float, default=0.5,
                help="Weight on adversarial loss added to total loss")
    p.add_argument("--adv_grl_max", type=float, default=1.0,
                help="Max GRL scale (gradient reversal factor)")
    p.add_argument("--adv_warmup_epochs", type=int, default=1000,
                help="Warm up GRL scale linearly from 0 to adv_grl_max over these epochs")

    # Optionally apply adversarial loss to z sampled by latent-flow (off by default)
    p.add_argument("--adv_on_lf", action="store_true",
                help="Also adversarially strip joint from z sampled by latent-flow")



    args = p.parse_args()
    if args.val_every is None:
        args.val_every = args.save_every
    if args.no_shape_latent and args.use_vae:
        p.error("--no_shape_latent is incompatible with --use_vae (there is no encoder at all)")

    # ---- DDP / device / seed ----
    is_dist, rank, world_size, local_rank = init_distributed()
    args.is_distributed = is_dist; args.rank=rank; args.world_size=world_size; args.local_rank=local_rank
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if rank == 0: os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed + rank)

    if not torch.cuda.is_available():
        args.amp = False
        args.use_bf16 = False
    elif args.use_bf16 and hasattr(torch.cuda, "is_bf16_supported") and (not torch.cuda.is_bf16_supported()):
        args.use_bf16 = False

    # ===== TensorBoard =====
    use_tb = (not args.no_tb) and (rank == 0)
    tb_dir = args.tb_log_dir or os.path.join(args.out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir) if use_tb else None
    if writer is not None:
        # Write hyperparameters as JSON for easier experiment tracking
        hp = {k: (v if isinstance(v, (bool, int, float, str)) else str(v)) for k, v in vars(args).items()}
        writer.add_text("hparams/json", json.dumps(hp, ensure_ascii=False, indent=2), 0)
        writer.flush()

    # ---- datasets (get_datasets sets args.cond_dim & args.has_rgb) ----
    tr_ds, te_ds = get_datasets(args)
    args.has_rgb = bool(getattr(args, "has_rgb", False))

    # ---- loaders ----
    if is_dist:
        tr_sampler = DistributedSampler(tr_ds, shuffle=True, drop_last=True)
        te_sampler = DistributedSampler(te_ds, shuffle=False, drop_last=False)
    else:
        tr_sampler = None; te_sampler = None
    train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=(tr_sampler is None),
                              sampler=tr_sampler, num_workers=args.num_workers, drop_last=True,
                              pin_memory=True, worker_init_fn=init_np_seed)
    val_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                            sampler=te_sampler, num_workers=max(1, args.num_workers//2),
                            drop_last=False, pin_memory=True, worker_init_fn=init_np_seed)

    # ---- models ----
    # enc_in_ch = 6 if (args.use_rgb_in_latent and args.has_rgb) else 3
    # enc = ShapeEncoder(args.latent_dim, width=args.enc_width, depth=args.enc_depth, in_channels=enc_in_ch).to(args.device)

    enc_in_ch = 6 if (args.use_rgb_in_latent and args.has_rgb) else 3
    if args.no_shape_latent:
        enc = None  # one-stage baseline: no shape encoder / no z anywhere
    elif args.use_vae:
        enc = VAEEncoder(args.latent_dim, width=args.enc_width, depth=args.enc_depth, in_channels=enc_in_ch).to(args.device)
    else:
        enc = ShapeEncoder(args.latent_dim, width=args.enc_width, depth=args.enc_depth, in_channels=enc_in_ch).to(args.device)


    pf_point_dim = 6 if (args.pointflow_rgb and args.has_rgb) else 3
    pf_cond_dim  = (0 if args.no_shape_latent else args.latent_dim) + args.cond_dim

    if args.pf_backbone == "mlp":
        pf = VelocityNet(cond_dim=pf_cond_dim, width=args.pf_width, depth=args.pf_depth,
                         emb_dim=args.pf_emb_dim, cfg_dropout_p=args.cfg_drop_p,
                         point_dim=pf_point_dim).to(args.device)
    else:
        # HybridMLP: ContextNet (SA/FP) + VelocityNetWithContext
        pf = HybridMLP(
            cond_dim=pf_cond_dim,
            point_dim=pf_point_dim,
            # ContextNet
            ctx_dim=args.ctx_dim, ctx_emb_dim=args.ctx_emb_dim,
            stage_channels=args.ctx_stage_channels, stage_blocks=args.ctx_stage_blocks, stage_res=args.ctx_stage_res,
            with_se=args.ctx_with_se, norm_type=args.ctx_norm, gn_groups=args.ctx_gn_groups,
            with_global=args.ctx_with_global, voxel_normalize=args.ctx_voxel_normalize,
            # t-gate
            use_t_gate=True, t_gate_k=args.ctx_t_gate_k, t_gate_tau=args.ctx_t_gate_tau,
            # Head (per-point MLP)
            pf_width=args.pf_width, pf_depth=args.pf_depth, pf_emb_dim=args.pf_emb_dim,
            cfg_dropout_p=args.cfg_drop_p,
        ).to(args.device)

    # lf = ConditionalLatentVelocityNet(args.latent_dim, cond_dim=0, width=args.lf_width,
    #                                   depth=args.lf_depth, emb_dim=args.lf_emb_dim).to(args.device)

    # --- Latent-flow (enabled only in AE mode; never in one-stage mode) ---
    lf_cond_dim = args.cond_dim if getattr(args, "lf_use_joint_cond", True) else 0
    if (not args.use_vae) and (not args.no_shape_latent):
        lf = ConditionalLatentVelocityNet(args.latent_dim, cond_dim=lf_cond_dim,
                                        width=args.lf_width, depth=args.lf_depth, emb_dim=args.lf_emb_dim).to(args.device)
    else:
        lf = None
    if args.no_shape_latent:
        print("[Model Status] ONE-STAGE baseline: no shape latent (enc=None, lf=None); PF conditioned on action only.")
        print(f"[Model Status] Point-flow point dim: {pf_point_dim}, Point-flow cond dim: {pf_cond_dim} (= action dim).")
    else:
        print(f"[Model Status] Using {'VAE (no latent-flow)' if args.use_vae else 'AE + latent-flow'} mode.")
        print(f"[Model Status] Latent-flow is {'conditional' if args.lf_use_joint_cond else 'unconditional'}.")
        print(f"[Model Status] Encoder input channels: {enc_in_ch}, Point-flow point dim: {pf_point_dim}, Point-flow cond dim: {pf_cond_dim}, Latent-flow cond dim: {lf_cond_dim}.")

    adv = None
    grl = None
    if args.adv_enable and args.no_shape_latent and rank == 0:
        print("[WARN] --adv_enable is ignored with --no_shape_latent (no z to adversarially strip).")
    if args.adv_enable and args.cond_dim > 0 and (not args.no_shape_latent):
        adv = CondAdversary(args.latent_dim, args.cond_dim,
                            width=args.adv_width, depth=args.adv_depth,
                            dropout=args.adv_dropout).to(args.device)
        grl = GRL()


    # --- EMA ---
    ema_pf = EMA(pf, decay=args.ema_decay)
    pf.ema_shadow = ema_pf.shadow
    if lf is not None:
        ema_lf = EMA(lf, decay=args.ema_decay)
        lf.ema_shadow = ema_lf.shadow
    else:
        ema_lf = None

    if rank == 0:
        n_enc = count_parameters(enc) if enc is not None else 0
        n_pf = count_parameters(pf)
        if lf is not None:
            print(f"[Models] enc: {n_enc/1e6:.2f}M  pf: {n_pf/1e6:.2f}M  lf: {count_parameters(lf)/1e6:.2f}M")
        else:
            print(f"[Models] enc(VAE={args.use_vae}): {n_enc/1e6:.2f}M  pf: {n_pf/1e6:.2f}M  lf: -")
        print(f"[Dims] cond_dim(joint)={args.cond_dim} latent_dim={args.latent_dim} pf_cond_dim={pf_cond_dim} enc_in={enc_in_ch} pf_point_dim={pf_point_dim}")

    model_pf = pf
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if enc is not None:
            enc = DDP(enc, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        model_pf = DDP(pf, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        if lf is not None:
            lf = DDP(lf, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        if adv is not None:
            adv = DDP(adv, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    # z_dim=0 in one-stage mode -> guided_velocity zeroes the WHOLE cond for the uncond CFG branch
    pf_z_dim = 0 if args.no_shape_latent else args.latent_dim
    if hasattr(pf, "module"):  # DDP
        pf.module.z_dim = pf_z_dim
    else:
        pf.z_dim = pf_z_dim
    # ---- optim / scaler ----
    # opt = torch.optim.AdamW(list(enc.parameters()) + list(pf.parameters()) + list(lf.parameters()),
    #                         lr=args.lr_pf, weight_decay=args.weight_decay)
    # Named group index map (order matches the historical positional layout: enc / pf / lf / adv)
    param_groups = []
    pg_idx = {}
    if enc is not None:
        pg_idx["enc"] = len(param_groups)
        param_groups.append({"params": enc.parameters(), "lr": args.lr_enc})
    pg_idx["pf"] = len(param_groups)
    param_groups.append({"params": pf.parameters(),  "lr": args.lr_pf})
    if lf is not None:
        pg_idx["lf"] = len(param_groups)
        param_groups.append({"params": lf.parameters(), "lr": args.lr_lf})
    if adv is not None:
        pg_idx["adv"] = len(param_groups)
        param_groups.append({"params": adv.parameters(), "lr": args.lr_adv})

    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # BF16 does not need GradScaler (saves overhead)
    scaler = make_scaler(enabled=(args.amp and (not args.use_bf16)))



    args.total_steps = args.epochs * max(1, len(train_loader))
    args.global_step = 0

    # ---- Use a fixed validation batch for consistent comparisons ----
    val_iter = iter(val_loader)
    try:    val_batch = next(val_iter)
    except: val_batch = next(iter(val_loader))

    # ---- Prior sampler (supports RGB priors) ----
    def make_pf_prior_like(data_pf: torch.Tensor) -> torch.Tensor:
        B, N, D = data_pf.shape
        if D == 3:
            return torch.randn_like(data_pf) * args.point_prior_std
        else:
            z = data_pf.new_empty(B, N, 6)
            z[..., :3] = torch.randn(B, N, 3, device=data_pf.device, dtype=data_pf.dtype) * args.point_prior_std
            if args.color_prior == "gauss":
                z[..., 3:] = torch.randn(B, N, 3, device=data_pf.device, dtype=data_pf.dtype) * args.color_prior_std
            elif args.color_prior == "uniform":
                z[..., 3:] = torch.rand(B, N, 3, device=data_pf.device, dtype=data_pf.dtype)  # U[0,1]
            else:
                z[..., 3:] = 0.0
            return z

    # ---- Visualization: reconstruction from encoder z (deterministic) ----
    @torch.no_grad()
    def save_val_recon(ep: int):
        """
        Reconstruction visualization using EMA weights + Heun (RK2) integration.
        - latent is taken directly from the encoder (not sampled)
        - point-flow uses Heun (predictor-corrector)
        """
        if enc is None:
            return  # one-stage baseline: no encoder-z reconstruction; save_val_samples covers action-cond sampling
        # Unwrap DDP modules (no-op if not using DDP)
        net_pf  = pf.module  if hasattr(pf,  "module") else pf
        net_enc = enc.module if hasattr(enc, "module") else enc
        net_pf.eval(); net_enc.eval()

        # Use a fixed validation batch
        pts = val_batch["test_points"].to(args.device).float()  # (B,N,3)
        rgb = val_batch.get("test_rgb", None)
        if rgb is not None:
            rgb = rgb.to(args.device).float()

        # Encoder input (optionally concat RGB depending on enc_in_ch)
        enc_in = pts if (enc_in_ch == 3 or rgb is None) else torch.cat([pts, rgb], dim=-1)

        # Optionally evaluate with EMA weights
        use_ema = bool(getattr(args, "ema_eval", True))
        with use_ema_weights(net_pf, ema_pf.shadow, enabled=use_ema):
            z_out, aux = net_enc(enc_in)  # AE: aux may be None; VAE: aux contains mu/logvar/kl
            # In VAE mode, using mu for deterministic recon is usually more stable
            if isinstance(aux, dict) and ("mu" in aux):
                z_gt = aux["mu"]
            else:
                z_gt = z_out

            # 2) Build cond_full = [z_gt | cond] to match the point-flow conditioning dimension
            B = z_gt.shape[0]
            cond_j = val_batch.get("cond", None)
            if cond_j is not None:
                cond_j = cond_j.to(args.device).to(z_gt.dtype)
                cond_full = torch.cat([z_gt, cond_j], dim=1)
            else:
                # If the model expects extra cond dims but the batch has none, pad with zeros
                need = int(getattr(args, "cond_dim", 0))
                if need > 0:
                    pad = torch.zeros((B, need), device=args.device, dtype=z_gt.dtype)
                    cond_full = torch.cat([z_gt, pad], dim=1)
                else:
                    cond_full = z_gt

            # 3) Initial prior x0 for point-flow
            data_pf = torch.cat([pts, rgb], dim=-1) if (pf_point_dim == 6 and rgb is not None) else pts
            x = make_pf_prior_like(data_pf)  # (B,N,3/6)

            # 4) Heun (RK2) predictor-corrector integration: t0=k/steps -> t1=(k+1)/steps
            steps = max(1, int(args.sample_steps))
            dt = 1.0 / steps
            for k in range(steps):
                t0 = torch.full((B,), k * dt,          device=x.device, dtype=x.dtype)
                v1 = net_pf.guided_velocity(x, t0, cond_full, guidance_scale=args.guidance_scale)
                x_hat = x + v1 * dt
                t1 = torch.full((B,), (k + 1) * dt,    device=x.device, dtype=x.dtype)
                v2 = net_pf.guided_velocity(x_hat, t1, cond_full, guidance_scale=args.guidance_scale)
                x = x + 0.5 * dt * (v1 + v2)

        # 5) Save and evaluate
        out_dir = os.path.join(args.out_dir, f"samples_recon_ep{ep:04d}")
        if args.rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(args.vis_count, x.shape[0])):
                if x.shape[-1] == 6 and (rgb is not None) and ("save_point_cloud_ply_rgb" in globals() and save_point_cloud_ply_rgb is not None):
                    save_point_cloud_ply_rgb(x[i, :, :3], x[i, :, 3:].clamp(0,1), os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply_rgb(pts[i],       rgb[i].clamp(0,1),    os.path.join(out_dir, f"gt_{i}.ply"))
                else:
                    save_point_cloud_ply(x[i, :, :3] if x.shape[-1] == 6 else x[i], os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply(pts[i],                                           os.path.join(out_dir, f"gt_{i}.ply"))
            cd = chamfer_l2(x[:, :, :3] if x.shape[-1] == 6 else x, pts).mean().item()
            print(f"[Val-Recon ep{ep:04d}] CD = {cd:.6f} (EMA={use_ema}, Heun)")
            if writer is not None and args.rank == 0:
                writer.add_scalar("val/recon_cd", cd, ep)


    # ---- Visualization: random z sampling ----
    @torch.no_grad()
    def save_val_samples(ep: int):
        """
        Random sampling visualization using EMA weights + Heun (RK2):
        - latent-flow: sample z with Heun
        - point-flow: integrate from x0 to data space with Heun
        """
        from contextlib import nullcontext
        net_pf = pf.module if hasattr(pf, "module") else pf
        net_pf.eval()
        net_lf = None
        if lf is not None:
            net_lf = lf.module if hasattr(lf, "module") else lf
            net_lf.eval()

        pts = val_batch["test_points"].to(args.device).float()
        rgb = val_batch.get("test_rgb", None)
        if rgb is not None:
            rgb = rgb.to(args.device).float()
        # NOTE: fetch joint condition early so it is available for latent-flow FM
        cond_j = val_batch.get("cond", None)
        if cond_j is not None:
            cond_j = cond_j.to(args.device).to(pts.dtype)
        use_ema = bool(getattr(args, "ema_eval", True))
        ctx_pf = use_ema_weights(net_pf, ema_pf.shadow, enabled=use_ema)
        ctx_lf = use_ema_weights(net_lf, ema_lf.shadow, enabled=use_ema) if (net_lf is not None) else nullcontext()
        with ctx_pf, ctx_lf:
            steps = max(1, int(args.sample_steps))
            dt = 1.0 / steps
            B = pts.shape[0]

            # # 1) latent-flow (unconditional) Heun sampling: y0 ~ N(0, sigma^2 I) -> z
            # z = torch.randn((B, args.latent_dim), device=args.device, dtype=pts.dtype) * args.latent_prior_std
            # steps = max(1, int(args.sample_steps))
            # dt = 1.0 / steps
            # for k in range(steps):
            #     t0 = torch.full((B,), k * dt,       device=z.device, dtype=z.dtype)
            #     v1 = net_lf(z, t0, cond=None)
            #     z_hat = z + v1 * dt
            #     t1 = torch.full((B,), (k + 1) * dt, device=z.device, dtype=z.dtype)
            #     v2 = net_lf(z_hat, t1, cond=None)
            #     z = z + 0.5 * dt * (v1 + v2)
            # VAE: sample directly from the prior; AE+LF: keep Heun sampling

            if args.no_shape_latent:
                # One-stage baseline: no shape latent at all
                z = None
            elif lf is None:
                # VAE: sample directly from the prior
                z = torch.randn((B, args.latent_dim), device=args.device, dtype=pts.dtype) * args.latent_prior_std
            else:
                # AE+LF: keep the original Heun sampler
                z = torch.randn((B, args.latent_dim), device=args.device, dtype=pts.dtype) * args.latent_prior_std
                steps = max(1, int(args.sample_steps)); dt = 1.0 / steps
                cond_for_lf = cond_j if args.lf_use_joint_cond else None
                for k in range(steps):
                    t0 = torch.full((B,), k * dt,       device=z.device, dtype=z.dtype)
                    v1 = net_lf(z, t0, cond=cond_for_lf)
                    z_hat = z + v1 * dt
                    t1 = torch.full((B,), (k + 1) * dt, device=z.device, dtype=z.dtype)
                    v2 = net_lf(z_hat, t1, cond=cond_for_lf)
                    z = z + 0.5 * dt * (v1 + v2)


            # 2) Build cond_full = [z | cond] (aligned with training)
            if args.no_shape_latent:
                # One-stage: condition on the action vector only (aligned with training)
                need = int(getattr(args, "cond_dim", 0))
                if cond_j is not None:
                    cond_full = cond_j
                elif need > 0:
                    cond_full = torch.zeros((B, need), device=args.device, dtype=pts.dtype)
                else:
                    cond_full = None
            elif cond_j is not None:
                cond_full = torch.cat([z, cond_j], dim=1)
            else:
                need = int(getattr(args, "cond_dim", 0))
                if need > 0:
                    pad = torch.zeros((B, need), device=args.device, dtype=z.dtype)
                    cond_full = torch.cat([z, pad], dim=1)
                else:
                    cond_full = z

            # 3) point-flow: x0 ~ prior -> Heun integration
            target_pf = torch.cat([pts, rgb], dim=-1) if (pf_point_dim == 6 and rgb is not None) else pts
            x = make_pf_prior_like(target_pf)
            for k in range(steps):
                t0 = torch.full((B,), k * dt,       device=x.device, dtype=x.dtype)
                v1 = net_pf.guided_velocity(x, t0, cond_full, guidance_scale=args.guidance_scale)
                x_hat = x + v1 * dt
                t1 = torch.full((B,), (k + 1) * dt, device=x.device, dtype=x.dtype)
                v2 = net_pf.guided_velocity(x_hat, t1, cond_full, guidance_scale=args.guidance_scale)
                x = x + 0.5 * dt * (v1 + v2)

        # 4) Save and evaluate
        if args.rank == 0:
            out_dir = os.path.join(args.out_dir, f"samples_ep{ep:04d}")
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(args.vis_count, x.shape[0])):
                if x.shape[-1] == 6 and (rgb is not None) and ("save_point_cloud_ply_rgb" in globals() and save_point_cloud_ply_rgb is not None):
                    save_point_cloud_ply_rgb(x[i, :, :3], x[i, :, 3:].clamp(0,1), os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply_rgb(pts[i],       rgb[i].clamp(0,1),    os.path.join(out_dir, f"gt_{i}.ply"))
                else:
                    save_point_cloud_ply(x[i, :, :3] if x.shape[-1] == 6 else x[i], os.path.join(out_dir, f"pred_{i}.ply"))
                    save_point_cloud_ply(pts[i],                                           os.path.join(out_dir, f"gt_{i}.ply"))
            cd = chamfer_l2(x[:, :, :3] if x.shape[-1] == 6 else x, pts).mean().item()
            print(f"[Val ep{ep:04d}] random-z CD = {cd:.4f} (EMA={use_ema}, Heun)")
            if writer is not None and args.rank == 0:
                writer.add_scalar("val/randomz_cd", cd, ep)


    # =========================
    # [Auto-Resume] checkpoint resume + device/state fixes
    # =========================
    import re

    def _find_latest_ckpt(ckpt_dir: str):
        """Return (path, epoch); if not found, return (None, 0)."""
        if not os.path.isdir(ckpt_dir):
            return None, 0
        best_ep, best_path = 0, None
        for fn in os.listdir(ckpt_dir):
            m = re.match(r"hybrid_ep(\d+)\.pt$", fn)
            if m:
                ep = int(m.group(1))
                if ep > best_ep:
                    best_ep = ep
                    best_path = os.path.join(ckpt_dir, fn)
        return best_path, best_ep

    def _find_resume_ckpt(ckpt_dir: str):
        latest = os.path.join(ckpt_dir, "latest.pt")
        if os.path.isfile(latest):
            return latest, None  # epoch will be read from the checkpoint
        return _find_latest_ckpt(ckpt_dir)


    def _move_opt_state_to_device(opt: torch.optim.Optimizer, device: torch.device):
        """Move optimizer state tensors to target device to avoid device mismatch after resume."""
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(device)

    def _safe_load_ema(ema_obj: EMA, state_dict: dict, ref_model: nn.Module, device: torch.device):
        """
        Use the current ema_obj.shadow as the full key set; overwrite overlapping keys from ckpt and move to device.
        This avoids KeyError and fixes CPU/GPU device mismatches.
        """
        cur = ema_obj.shadow  # contains the full key set
        ref_sd = ref_model.state_dict()
        for k in cur.keys():
            if k in state_dict:
                v = state_dict[k]
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    cur[k] = v.to(device=device, dtype=ref_sd[k].dtype)
        ema_obj.shadow = cur

    def atomic_torch_save(obj, path: str):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic replace to avoid partial writes

    def _capture_rng_states():
        np_state = np.random.get_state()  # (alg, ndarray(uint32, 624), pos, has_gauss, cached_gaussian)
        # Convert the second item (ndarray) to list to avoid safe-unpickling issues in newer PyTorch
        if isinstance(np_state, tuple) and len(np_state) >= 2 and hasattr(np_state[1], "tolist"):
            np_state = (np_state[0], np_state[1].tolist(), *np_state[2:])
        return {
            "torch": torch.get_rng_state(),
            "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np_state,
            "python": random.getstate(),
        }


    def _restore_rng_states(rng):
        try:
            if rng is None: return
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"])
            if "cuda_all" in rng and rng["cuda_all"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda_all"])
            if "numpy" in rng and rng["numpy"] is not None:
                np_state = rng["numpy"]
                # Support both the newer list-based format and the older ndarray-based format
                if isinstance(np_state, tuple) and len(np_state) >= 2 and isinstance(np_state[1], list):
                    arr = np.array(np_state[1], dtype=np.uint32)
                    np_state = (np_state[0], arr, *np_state[2:])
                np.random.set_state(np_state)
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])
        except Exception as e:
            if args.rank == 0:
                print(f"[WARN] Restore RNG states failed: {e}")


    start_epoch = 1
    ckpt_path, ckpt_ep = _find_resume_ckpt(os.path.join(args.out_dir, "ckpts"))

    if ckpt_path is not None:
        if rank == 0:
            print(f"[Auto-Resume] Found latest ckpt: {ckpt_path} (ep={ckpt_ep})")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Optionally sanitize RNG state for compatibility
        try:
            rng = ckpt.get("rng_state", None)
            if isinstance(rng, dict) and isinstance(rng.get("numpy", None), tuple):
                np_state = list(rng["numpy"])
                if len(np_state) >= 2 and hasattr(np_state[1], "tolist"):  # old format: ndarray
                    np_state[1] = np_state[1].tolist()
                    rng["numpy"] = tuple(np_state)
                    atomic_torch_save(ckpt, ckpt_path)
                    if rank == 0:
                        print("[Auto-Resume] Re-saved latest.pt in safe RNG format.")
        except Exception as e:
            if rank == 0: print(f"[Auto-Resume][WARN] sanitize latest.pt failed: {e}")


        _restore_rng_states(ckpt.get("rng_state", None))


        # Restore model weights (unwrap .module under DDP)
        enc_t = enc.module if (enc is not None and hasattr(enc, "module")) else enc
        pf_t  = pf.module  if hasattr(pf,  "module") else pf
        lf_t  = lf.module  if (lf is not None and hasattr(lf,  "module")) else lf

        if ("encoder" in ckpt) and (enc_t is not None): enc_t.load_state_dict(ckpt["encoder"], strict=True)
        if "pf" in ckpt:      pf_t.load_state_dict(ckpt["pf"], strict=False)
        elif "model" in ckpt: pf_t.load_state_dict(ckpt["model"], strict=False)  # backward-compat key
        if lf_t is not None and ("lf" in ckpt):
            lf_t.load_state_dict(ckpt["lf"], strict=False)
        if adv is not None and ("adv" in ckpt) and isinstance(ckpt["adv"], dict) and len(ckpt["adv"]) > 0:
            (adv.module if hasattr(adv, "module") else adv).load_state_dict(ckpt["adv"], strict=False)

        # Restore EMA (move to correct device + align keys)
        if "ema_pf" in ckpt and isinstance(ckpt["ema_pf"], dict):
            _safe_load_ema(ema_pf, ckpt["ema_pf"], pf_t, device=torch.device(args.device))
            pf.ema_shadow = ema_pf.shadow
        if (lf is not None) and ("ema_lf" in ckpt) and isinstance(ckpt["ema_lf"], dict):
            _safe_load_ema(ema_lf, ckpt["ema_lf"], lf_t, device=torch.device(args.device))
            lf.ema_shadow = ema_lf.shadow

        # Restore optimizer / AMP scaler (if any) and move optimizer state to current device
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
                _move_opt_state_to_device(opt, torch.device(args.device))
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] opt state load failed: {e}")
        elif "opt_main" in ckpt:  # backward-compat key
            try:
                opt.load_state_dict(ckpt["opt_main"])
                _move_opt_state_to_device(opt, torch.device(args.device))
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] opt_main->opt load failed: {e}")

        if args.amp and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                if rank == 0: print(f"[Auto-Resume][WARN] scaler state load failed: {e}")


        # Restore epoch and global step (continue training)
        last_epoch = int(ckpt.get("epoch", ckpt_ep if ckpt_ep is not None else 0))
        approx_gs  = last_epoch * max(1, len(train_loader))
        args.global_step = int(ckpt.get("global_step", approx_gs))
        start_epoch = last_epoch + 1

        if rank == 0:
            remain = max(0, args.epochs - last_epoch)
            print(f"[Auto-Resume] Resume from epoch {last_epoch}. "
                f"Target total epochs = {args.epochs}. Will run {remain} more epoch(s).")

        # If we have already reached the requested total epochs, exit early
        if start_epoch > args.epochs:
            if rank == 0:
                print("[Auto-Resume] Training already completed for the requested total epochs. Nothing to do.")
            if writer is not None:
                writer.close()
            cleanup_distributed()
            return
    else:
        if rank == 0:
            print("[Auto-Resume] No checkpoint found. Start training from scratch.")


        

    # ================= Training =================
    for ep in range(start_epoch, args.epochs + 1):
        # Two-stage training: during warmup, train geometry only (RGB disabled)
        use_rgb_this_epoch = (ep > args.geom_warmup_epochs) and (args.pointflow_rgb and args.has_rgb)

        if is_dist and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(ep)
        # enc.train(); pf.train(); lf.train()
        if enc is not None:
            enc.train()
        pf.train()
        if lf is not None:
            lf.train()
        pbar = tqdm(total=len(train_loader), desc=f"Ep{ep}") if rank == 0 else None

        for batch in train_loader:
            pts = batch["train_points"].to(args.device).float()    # (B,N,3)
            rgb = batch.get("train_rgb", None)
            if rgb is not None:
                rgb = rgb.to(args.device).float()
            cond_j = batch.get("cond", None)
            if cond_j is not None:
                cond_j = cond_j.to(args.device).float()

            # ---- Encoder ----
            # During warmup, let the encoder see geometry only (avoid relying on RGB)
            # If enc_in_ch == 6, keep encoder input 6D in both warmup and full training
            # In warmup, zero-out the RGB channels to keep dims unchanged while removing color signal
            if enc is None:
                # One-stage baseline: no encoder / no z
                z, aux, kl = None, None, None
            else:
                if enc_in_ch == 6:
                    if rgb is not None:
                        if use_rgb_this_epoch:
                            enc_in = torch.cat([pts, rgb], dim=-1)          # (B,N,6)
                        else:
                            zeros_rgb = torch.zeros_like(pts)                # (B,N,3)
                            enc_in = torch.cat([pts, zeros_rgb], dim=-1)     # (B,N,6) - warmup: rgb=0
                    else:
                        # Fallback: if RGB is expected but missing, pad zeros
                        zeros_rgb = torch.zeros_like(pts)
                        enc_in = torch.cat([pts, zeros_rgb], dim=-1)
                else:
                    enc_in = pts                                             # (B,N,3)

                # with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                #     z, _ = enc(enc_in)   # (B, Dz)
                # In VAE mode, aux may contain mu/logvar/kl
                with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                    z, aux = enc(enc_in)   # AE: aux may be None; VAE: aux is a dict
                    kl = None
                    if args.use_vae and isinstance(aux, dict):
                        kl = aux.get("kl", None)


            # ---- Point-flow FM (3D or 6D) ----
            # For 6D: during warmup, keep RGB dims at 0 (inputs/targets are 0) so geometry is learned first
            if pf_point_dim == 6:
                if (rgb is not None) and use_rgb_this_epoch:
                    data_pf = torch.cat([pts, rgb], dim=-1)                     # (B,N,6)
                    # RGB prior as configured (gauss / uniform / zeros)
                    z_pts = make_pf_prior_like(data_pf)                          # (B,N,6)
                else:
                    # Warmup: keep RGB dims at 0; also set prior RGB to 0 to avoid noise affecting geometry
                    zeros_rgb = torch.zeros_like(pts)
                    data_pf = torch.cat([pts, zeros_rgb], dim=-1)                # (B,N,6)
                    z_pts = torch.empty_like(data_pf)
                    z_pts[..., :3] = torch.randn_like(pts) * args.point_prior_std
                    z_pts[..., 3:] = 0.0
            else:
                data_pf = pts
                z_pts  = torch.randn_like(data_pf) * args.point_prior_std

            B, N, D = data_pf.shape

            # ---- Sample t biased towards 1 (Beta) ----
            beta = torch.distributions.Beta(concentration1=args.t_beta_a, concentration0=1.0)
            t_pts = beta.sample((B,)).to(device=args.device, dtype=data_pf.dtype)  # (B,)
            x_t   = (1.0 - t_pts)[:, None, None] * z_pts + t_pts[:, None, None] * data_pf
            target_v = (data_pf - z_pts)

            # ---- Build conditioning vector (aligned with training) ----
            if args.no_shape_latent:
                # One-stage: condition on the raw action vector only (FiLM embedding lives inside the model)
                if cond_j is not None:
                    cond_full = cond_j
                else:
                    need = int(getattr(args, "cond_dim", 0))
                    cond_full = torch.zeros((B, need), device=args.device, dtype=data_pf.dtype) if need > 0 else None
            elif cond_j is None:
                need = int(getattr(args, "cond_dim", 0))
                if need > 0:
                    pad = torch.zeros((z.size(0), need), device=args.device, dtype=z.dtype)
                    cond_full = torch.cat([z, pad], dim=1)
                else:
                    cond_full = z
            else:
                cond_full = torch.cat([z, cond_j], dim=1)


            # NOTE: compute the per-epoch cond-dropout prob; shared by PF and LF
            drop_p_now = 0.0
            if args.cfg_drop_p > 0.0:
                # Linear warmup: 0 -> cfg_drop_p over cfg_drop_warmup_epochs epochs
                drop_p_now = float(args.cfg_drop_p) * min(
                    1.0, max(0.0, (ep / max(1, args.cfg_drop_warmup_epochs)))
                )

            # For PF, implement CFG training via a per-sample drop mask
            cond_drop_mask = None
            if (drop_p_now > 0.0) and (cond_full is not None):
                # mask shape: (B,1) values in {0,1}; 1 means drop cond for that sample
                drop = (torch.rand(B, device=args.device) < drop_p_now).to(data_pf.dtype)
                cond_drop_mask = drop[:, None]

            with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                pred_v = model_pf(x_t, t_pts, cond_full, cond_drop_mask=cond_drop_mask)
                if D == 6:
                    if (rgb is not None) and use_rgb_this_epoch:
                        # Full 6D supervision: geometry + color
                        loss_pos = F.mse_loss(pred_v[..., :3], target_v[..., :3])
                        loss_col = F.mse_loss(pred_v[..., 3:], target_v[..., 3:])
                        loss_point = loss_pos + args.lambda_color * loss_col
                    else:
                        # Warmup: geometry supervision only
                        loss_point = F.mse_loss(pred_v[..., :3], target_v[..., :3])
                else:
                    loss_point = F.mse_loss(pred_v, target_v)


            # ---- Latent-flow FM (optionally conditioned; shares drop_p_now with PF) ----
            loss_latent = torch.tensor(0.0, device=args.device, dtype=pts.dtype)
            if (not args.use_vae) and (lf is not None):
                with torch.no_grad(): z_det = z.detach()
                eps_z = torch.randn_like(z_det) * args.latent_prior_std
                beta_latent = torch.distributions.Beta(concentration1=args.t_beta_a, concentration0=1.0)
                t_z = beta_latent.sample((B,)).to(device=args.device, dtype=z_det.dtype)
                y_t = (1.0 - t_z)[:, None] * eps_z + t_z[:, None] * z_det
                target_v_z = (z_det - eps_z)
                with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
                    # ---- Latent-flow FM (joint conditioning controlled by a flag) ----
                    cond_lf = cond_j.to(z_det.dtype) if (args.lf_use_joint_cond and cond_j is not None) else None
                    pred_v_z = lf(
                        y_t, t_z,
                        cond=cond_lf,
                        # When unconditional, cond_lf=None and cond_drop_p=0
                        cond_drop_p=(drop_p_now if cond_lf is not None else 0.0)
                    )
                    loss_latent = F.mse_loss(pred_v_z, target_v_z)



            # ---- Total loss ----
            if args.use_vae:
                # Beta annealing: linearly ramp to beta_kl (or keep constant if kl_warmup_epochs=0)
                if args.kl_warmup_epochs and args.kl_warmup_epochs > 0:
                    beta_now = float(args.beta_kl) * min(1.0, max(0.0, (ep / float(args.kl_warmup_epochs))))
                else:
                    beta_now = float(args.beta_kl)
                # Defensive: if kl is None due to numerical issues, treat it as 0
                kl_term = (kl if (kl is not None) else torch.tensor(0.0, device=args.device, dtype=pts.dtype))
                loss = args.lambda_point * loss_point + beta_now * kl_term
            else:
                loss = args.lambda_point * loss_point + args.lambda_latent * loss_latent

            # ===== Adversarial: remove joint info from z =====
            adv_loss = torch.tensor(0.0, device=args.device, dtype=pts.dtype)
            if (adv is not None) and (cond_j is not None):
                # Linearly warm up GRL strength
                if args.adv_warmup_epochs and args.adv_warmup_epochs > 0:
                    grl_now = float(args.adv_grl_max) * min(1.0, max(0.0, ep / float(args.adv_warmup_epochs)))
                else:
                    grl_now = float(args.adv_grl_max)

                # Important: cast inputs to adv weight dtype (DDP-safe)
                adv_t = (adv.module if hasattr(adv, "module") else adv)
                adv_dtype = next(adv_t.parameters()).dtype

                # Adversarial head on encoder z (main term)
                z_grl = grl(z, lambd=grl_now).to(adv_dtype)
                pred_c = adv(z_grl)
                adv_loss_enc = F.mse_loss(pred_c, cond_j)  # cond_j is already float32

                # Optional: adversarial head on z sampled by latent-flow
                adv_loss_lf = torch.tensor(0.0, device=args.device, dtype=pts.dtype)
                if args.adv_on_lf and (not args.use_vae) and (lf is not None):
                    with torch.no_grad():
                        z_lf = z.detach()
                    z_lf_grl = grl(z_lf, lambd=grl_now).to(adv_dtype)
                    pred_c_lf = adv(z_lf_grl)
                    adv_loss_lf = F.mse_loss(pred_c_lf, cond_j)

                adv_loss = adv_loss_enc + 0.5 * adv_loss_lf
            loss = loss + float(args.adv_lambda) * adv_loss


            gnorm = None

            # ---- Backprop / step ----
            scaler.scale(loss).backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                scaler.unscale_(opt)
                params_for_clip = (list(enc.parameters()) if enc is not None else []) + list(pf.parameters())
                if lf is not None:
                    params_for_clip += list(lf.parameters())
                gnorm = torch.nn.utils.clip_grad_norm_(params_for_clip, args.grad_clip_norm)

            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            # ---- EMA ----
            ema_pf.update(pf.module if hasattr(pf, "module") else pf)
            if lf is not None:
                ema_lf.update(lf.module if hasattr(lf, "module") else lf)

            # ---- LR schedule ----
            # if args.use_cosine_lr:
            #     lr_enc_now = cosine_lr(args.global_step, args.total_steps, args.lr_enc, args.min_lr, args.warmup_steps)
            #     lr_pf_now  = cosine_lr(args.global_step, args.total_steps, args.lr_pf,  args.min_lr, args.warmup_steps)
            #     lr_lf_now  = cosine_lr(args.global_step, args.total_steps, args.lr_lf,  args.min_lr, args.warmup_steps)
            #     # NOTE: param_groups order: enc / pf / lf
            #     opt.param_groups[0]['lr'] = lr_enc_now
            #     opt.param_groups[1]['lr'] = lr_pf_now
            #     opt.param_groups[2]['lr'] = lr_lf_now
            if args.use_cosine_lr:
                # NOTE: use the named group index map (enc group is absent with --no_shape_latent)
                if "enc" in pg_idx:
                    lr_enc_now = cosine_lr(args.global_step, args.total_steps, args.lr_enc, args.min_lr, args.warmup_steps)
                    opt.param_groups[pg_idx["enc"]]['lr'] = lr_enc_now
                lr_pf_now  = cosine_lr(args.global_step, args.total_steps, args.lr_pf,  args.min_lr, args.warmup_steps)
                opt.param_groups[pg_idx["pf"]]['lr'] = lr_pf_now
                if "lf" in pg_idx:
                    lr_lf_now  = cosine_lr(args.global_step, args.total_steps, args.lr_lf,  args.min_lr, args.warmup_steps)
                    opt.param_groups[pg_idx["lf"]]['lr'] = lr_lf_now

            args.global_step += 1

            # ===== TensorBoard scalars =====
            if writer is not None and (args.global_step % max(1, args.tb_log_every) == 0):
                # Current learning rates (named group indices; enc group absent with --no_shape_latent)
                if "enc" in pg_idx:
                    writer.add_scalar("train/lr_enc", opt.param_groups[pg_idx["enc"]]['lr'], args.global_step)
                writer.add_scalar("train/lr_pf", opt.param_groups[pg_idx["pf"]]['lr'], args.global_step)
                if gnorm is not None:
                    writer.add_scalar(
                        "train/grad_norm",
                        float(gnorm.detach().cpu() if torch.is_tensor(gnorm) else gnorm),
                        args.global_step
                    )
                if "lf" in pg_idx:
                    writer.add_scalar("train/lr_lf", opt.param_groups[pg_idx["lf"]]['lr'], args.global_step)
                if adv is not None:
                    writer.add_scalar("train/adv_loss", float(adv_loss.detach().cpu()), args.global_step)
                    writer.add_scalar("train/adv_grl_now", float(grl_now if 'grl_now' in locals() else 0.0), args.global_step)
                # Losses
                if args.use_vae:
                    klv = (float(kl.detach().cpu()) if (kl is not None and torch.is_tensor(kl)) else 0.0)
                    writer.add_scalar("train/loss_point", float(loss_point.detach().cpu()), args.global_step)
                    writer.add_scalar("train/kl", klv, args.global_step)
                    writer.add_scalar("train/loss_total", float(loss.detach().cpu()), args.global_step)
                else:
                    writer.add_scalar("train/loss_point", float(loss_point.detach().cpu()), args.global_step)
                    writer.add_scalar("train/loss_latent", float(loss_latent.detach().cpu()), args.global_step)
                    writer.add_scalar("train/loss_total", float(loss.detach().cpu()), args.global_step)

                # CFG cond-dropout probability
                writer.add_scalar("train/cfg_drop_p_now", float(drop_p_now), args.global_step)


            if pbar is not None:
                if args.use_vae:
                    klv = float(kl.detach().cpu()) if (kl is not None and torch.is_tensor(kl)) else 0.0
                    pbar.set_postfix(lp=float(loss_point.detach().cpu()), kl=klv)
                else:
                    pbar.set_postfix(lp=float(loss_point.detach().cpu()), lz=float(loss_latent.detach().cpu()))
                pbar.update(1)

        
        if pbar is not None: pbar.close()

        # ---- Save & Eval ----
        if rank == 0:
            ckpt = {
                "epoch": ep,
                **({"encoder": (enc.module if hasattr(enc, "module") else enc).state_dict()} if enc is not None else {}),
                "pf":      (pf.module  if hasattr(pf,  "module") else pf).state_dict(),
                **({"lf": (lf.module if hasattr(lf, "module") else lf).state_dict()} if lf is not None else {}),
                "ema_pf": ema_pf.shadow,
                **({"ema_lf": ema_lf.shadow} if (ema_lf is not None) else {}),
                "args": {
                    **vars(args),
                    "enc_in_channels": enc_in_ch,
                    "pf_point_dim": pf_point_dim,
                },
                "cond_dim": args.cond_dim,
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if (args.amp and (not args.use_bf16)) else None,
                "global_step": args.global_step,
                # (NEW) RNG
                "rng_state": _capture_rng_states(),
                "adv": (adv.module if (adv is not None and hasattr(adv, "module")) else adv.state_dict() if adv is not None else None),

            }
            ckpt_dir = os.path.join(args.out_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)

            # 1) Write latest.pt via atomic replace (every epoch)
            atomic_torch_save(ckpt, os.path.join(ckpt_dir, "latest.pt"))

            # 2) Archive by save_every
            if (ep % args.save_every) == 0 or ep == args.epochs:
                torch.save(ckpt, os.path.join(ckpt_dir, f"hybrid_ep{ep:04d}.pt"))

        if is_dist and dist.is_initialized():
            dist.barrier()

        do_val = ((ep % max(1, args.val_every)) == 0) or (ep == args.epochs)
        if do_val:
            if is_dist and dist.is_initialized():
                dist.barrier()
            save_val_recon(ep)
            save_val_samples(ep)
            if writer is not None and args.rank == 0:
                writer.flush()


        # Flush TensorBoard logs promptly
        if writer is not None and args.rank == 0:
            writer.flush()


    if writer is not None:
        writer.close()

    cleanup_distributed()

if __name__ == "__main__":
    main()