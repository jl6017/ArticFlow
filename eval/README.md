# Evaluation suite (CoRL 2026 rebuttal experiments)

These scripts produced the kinematic-consistency, novelty, and simulation-readiness
numbers reported in the paper's evaluation and rebuttal. They expect the experiment-tree
layout (checkpoints + generated PLY/H5 sweeps); paths inside each header docstring show
the intended invocation.

| script | what it measures |
|---|---|
| `kinematic_probe.py` | per-joint action sweeps of a checkpoint (one dim at a time, fixed latent) → point-cloud sequence npz; supports safe-direction (collision-free) sweeps and action noise |
| `kinematic_tree_eval.py` | axis spread/deviation, rigid-fit residuals (ICP corr-free + correspondence), realized rotation, hierarchical containment from probe npz |
| `gt_probe_exam.py` | validates the estimator on FK-sampled ground-truth robots (axis error, commanded-gain error, joint-position error) |
| `eval_k3.py` | distribution-level generation metrics (MMD-CD / COV / 1-NNA) for ArticFlow vs same-backbone controls (PVD-objective diffusion, one-stage FM) |
| `build_identity_split.py` / `eval_idsplit.py` / `enrich_idsplit.py` | identity-held-out retraining protocol and seen/held-out reference scoring |
| `nn_viz.py` | rank-selected nearest-neighbour panels (generated vs training / held-out references) with train–train NN-CD baseline |
| `heldout_frame_select.py` | GT-free model selection: fit sweep angles on early frames, score extrapolation on the held-out final frame |
| `batch_assembly_stats.py` | MJCF assembly / load / actuator checks over a generated batch |
| `loco_figures.py` | learning-curve and rollout-filmstrip figures from brax/MJX training logs |
