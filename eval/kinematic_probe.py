"""Interventional kinematic probe: one-joint-at-a-time action sweeps (Fig 9 protocol).

For each shape latent (sampled from the prior and/or slerp-interpolated), sweep each
action dimension independently while holding the others at a base value, integrating
every cloud FROM THE SAME x0 so points are index-corresponded across the whole probe.
Saves one npz per shape: clouds [J, S, N, 3] (float16), base cloud, and meta.

Run from $RSS with the articflow env, e.g.:
  python -u analysis/kinematic_probe.py \
    --ckpt checkpoints/arm_all_hybrid_11_11_uncond_adv.pt \
    --out rebuttal/kinprobe/arm --n_latents 2 --n_interp 1
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo import (build_from_ckpt, heun_integrate_pf, heun_integrate_lf,
                  make_pf_prior_like, slerp_path, use_ema_weights)


def sample_latents(lf, lf_loaded, latent_dim, latent_std, n, device, gen):
    zs = []
    for _ in range(n):
        y0 = torch.randn(1, latent_dim, generator=gen, device=device) * latent_std
        z = heun_integrate_lf(lf, y0, steps=100, cond=None) if lf_loaded else y0
        zs.append(z)
    return zs


@torch.no_grad()
def sweep_shape(pf, ema_pf, x0, z, cond_dim, base, sweep_vals, steps, gscale, batch_sweep,
                action_noise=0.0, noise_seed=0, only_dims=None):
    """Return clouds [J, S, N, 3] and base cloud [N, 3]."""
    device = x0.device
    N = x0.shape[1]
    J, S = cond_dim, len(sweep_vals)
    out = np.zeros((J, S, N, 3), dtype=np.float16)

    def gen_batch(actions):  # actions [B, J] -> xyz [B, N, 3]
        B = actions.shape[0]
        cond = torch.cat([z.expand(B, -1), actions], dim=1)
        with use_ema_weights(pf, ema_pf, enabled=True):
            x = heun_integrate_pf(pf, x0.expand(B, -1, -1).clone(), cond, steps=steps,
                                  guidance_scale=gscale)
        return x[..., :3].float().cpu().numpy()

    base_act = torch.full((1, cond_dim), base, device=device)
    base_cloud = gen_batch(base_act)[0].astype(np.float16)

    rng = torch.Generator(device=device).manual_seed(noise_seed)
    for j in range(J):
        if only_dims is not None and j not in only_dims:
            continue
        acts = base_act.expand(S, -1).clone()
        acts[:, j] = torch.tensor(sweep_vals, device=device)
        if action_noise > 0:
            # AC #8 "action noise": the CONDITIONING the generator receives is imprecise.
            # Perturb every dim of every frame's action vector, then clamp to the valid
            # range; the nominal sweep_vals still label the frames downstream, exactly as
            # a real system believes its commanded positions.
            acts = (acts + torch.randn(acts.shape, generator=rng, device=device)
                    * action_noise).clamp(0.0, 1.0)
        if batch_sweep:
            try:
                out[j] = gen_batch(acts).astype(np.float16)
                continue
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        for s in range(S):
            out[j, s] = gen_batch(acts[s:s + 1])[0].astype(np.float16)
    return out, base_cloud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_latents", type=int, default=2, help="latents sampled from prior")
    ap.add_argument("--n_interp", type=int, default=1,
                    help="slerp midpoints between consecutive sampled latents")
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=50, help="PF Heun steps")
    ap.add_argument("--sweep_steps", type=int, default=7)
    ap.add_argument("--base", type=float, default=0.5, help="held value for other dims")
    ap.add_argument("--joint_min", type=float, default=0.0)
    ap.add_argument("--joint_max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_batch_sweep", action="store_true")
    ap.add_argument("--action_noise", type=float, default=0.0,
                    help="sigma of Gaussian noise on the action vector (fraction of range)")
    ap.add_argument("--only_dims", type=int, nargs="+", default=None,
                    help="generate only these dims (others stay zero in the npz)")
    a = ap.parse_args()

    device = "cuda"
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    pf, lf, lf_loaded, info, ema_pf, ema_lf = build_from_ckpt(a.ckpt, device)
    pf.eval(); lf.eval()
    J = int(info["cond_dim"]); latent_dim = int(info["latent_dim"])
    D = int(info["pf_point_dim"])
    print(f"[probe] ckpt={os.path.basename(a.ckpt)} cond_dim={J} latent_dim={latent_dim} D={D}")

    zs = sample_latents(lf, lf_loaded, latent_dim, float(info["latent_prior_std"]),
                        a.n_latents, device, gen)
    shapes = [(f"sample{i}", z) for i, z in enumerate(zs)]
    for i in range(min(a.n_interp, len(zs) - 1)):
        mid = slerp_path(zs[i], zs[i + 1], 3)[1]
        shapes.append((f"interp{i}-{i+1}", mid))

    sweep_vals = np.linspace(a.joint_min, a.joint_max, a.sweep_steps).tolist()
    for name, z in shapes:
        x0 = make_pf_prior_like(1, a.n_points, D, float(info["point_prior_std"]),
                                str(info["color_prior"]), float(info["color_prior_std"]),
                                device=device, dtype=torch.float32)
        clouds, base_cloud = sweep_shape(pf, ema_pf, x0, z, J, a.base, sweep_vals,
                                         a.steps, float(info["guidance_scale"]),
                                         batch_sweep=not a.no_batch_sweep,
                                         action_noise=a.action_noise, noise_seed=a.seed,
                                         only_dims=a.only_dims)
        path = os.path.join(a.out, f"{name}.npz")
        np.savez_compressed(path, clouds=clouds, base=base_cloud,
                            sweep_vals=np.array(sweep_vals, dtype=np.float32))
        print(f"[probe] {name}: clouds {clouds.shape} -> {path}")

    meta = dict(ckpt=a.ckpt, cond_dim=J, latent_dim=latent_dim, n_points=a.n_points,
                steps=a.steps, sweep_steps=a.sweep_steps, base=a.base,
                joint_min=a.joint_min, joint_max=a.joint_max, seed=a.seed,
                shapes=[n for n, _ in shapes])
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[probe] done")


if __name__ == "__main__":
    main()
