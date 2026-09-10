"""The two locomotion deliverables: overlapping learning curves + mesh-visual filmstrip.

Curves: eval/episode_reward for every IcosQuadrupedJoystickOmni run — generated dogs vs the
real-robot (solo12) and manually designed references, SAME env and reward, so the curves are directly
comparable. Go1 trains in a different playground env with different reward shaping; its
absolute rewards are not comparable and it is drawn only if --with_go1, dashed, with the
caveat in the label.

Filmstrip: the 8 saved rollout frames of the mesh-rendered generated dog (the robot LOOKS
like what ArticFlow produced, not a capsule caricature), with the GT solo12 rollout as the
second row.

  python analysis/loco_figures.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RSS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGROOT = os.path.expanduser("~/Documents/CML_2026/mujoco_playground/logs")

RUNS = [  # (rewards.json key, display label, colour, ours?)
    ("joy_gt_solo12", "solo12 (real robot model)", "#1a7a3a", False),
    ("joy_handmade", "manually designed quadruped", "#888888", False),
    ("joy_gen_dog", "generated dog A (interpolated)", "#1f4e79", True),
    ("joy_sample3", "generated dog B (sample3)", "#7a1fa2", True),
    ("joy_sample4", "generated dog C (sample4)", "#b34700", True),
]


def tb_series(logdir, tag="eval/episode_reward", std_tag="eval/episode_reward_std"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = os.path.join(logdir, "tb")          # brax PPO writes into <logdir>/tb/
    acc = EventAccumulator(tb if os.path.isdir(tb) else logdir, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    ev = acc.Scalars(tag)
    x = np.array([e.step for e in ev]); y = np.array([e.value for e in ev])
    sd = None
    if std_tag in acc.Tags().get("scalars", []):
        sv = acc.Scalars(std_tag)
        if len(sv) == len(ev):
            sd = np.array([e.value for e in sv])
    return x, y, sd


def main():
    rew = json.load(open(os.path.join(RSS, "rebuttal/loco/rewards.json")))

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for key, label, col, ours in RUNS:
        if key not in rew:
            print(f"  [curves] {key} not in rewards.json, skipped")
            continue
        s = tb_series(rew[key]["log_dir"])
        if s is None:
            print(f"  [curves] {key}: tag missing, skipped")
            continue
        x, y, sd = s
        # standard RL presentation: mean line with a +/-1 std band over the 128 eval
        # episodes at each point (the only variance the single-seed logs contain)
        if sd is not None:
            ax.fill_between(x / 1e6, y - sd, y + sd, color=col, alpha=0.15, lw=0)
        ax.plot(x / 1e6, y, color=col, lw=2.2 if ours else 1.8,
                ls="-" if ours else "--", label=label, alpha=0.95)
        print(f"  [curves] {label}: {len(x)} evals, best {y.max():.1f}, final {y[-1]:.1f}")
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("eval episode reward (speed in commanded direction)")
    ax.set_title("Joystick-commanded locomotion — generated quadrupeds vs references\n"
                 "(identical env, reward and PPO config for all curves)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out1 = os.path.join(RSS, "rebuttal/loco/learning_curves.png")
    fig.savefig(out1); plt.close(fig)
    print("  ->", out1)

    # --- filmstrip: mesh-rendered generated dog above GT solo12 -----------------------
    rows = [("rollout_mesh_mesh_frames.npy", "generated dog (ArticFlow mesh)"),
            ("rollout_joy_gt_solo12_frames.npy", "solo12 (real robot model)")]
    ims, labels = [], []
    for f, lab in rows:
        p = os.path.join(RSS, "rebuttal/loco", f)
        if not os.path.exists(p):
            print(f"  [strip] missing {f}"); continue
        fr = np.load(p)                                     # (8, H, W, 3)
        ims.append(np.concatenate(list(fr), axis=1))
        labels.append(lab)
    H = ims[0].shape[0]
    fig, axes = plt.subplots(len(ims), 1, figsize=(16, 2.6 * len(ims)), dpi=140)
    for ax, im, lab in zip(np.atleast_1d(axes), ims, labels):
        ax.imshow(im); ax.axis("off")
        ax.set_title(lab, fontsize=10, loc="left")
    fig.suptitle("policy rollout, 8 frames left to right (trained joystick-commanded locomotion)",
                 fontsize=10)
    fig.tight_layout()
    out2 = os.path.join(RSS, "rebuttal/loco/loco_filmstrip.png")
    fig.savefig(out2); plt.close(fig)
    print("  ->", out2)


if __name__ == "__main__":
    main()
