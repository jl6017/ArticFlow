from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional


def parse_args():
    ap = argparse.ArgumentParser(description="PartNet-Mobility dataset pipeline (index -> collect -> per-category h5).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 1) index
    pi = sub.add_parser("index",help="Scan dataset root and write a CSV index (no category filtering).")
    pi.add_argument("--dataset-dir", type=Path, required=True,help="Root of PartNet-Mobility (first-level subfolders = models).")
    pi.add_argument("--out", type=Path, default=Path("partnet_mobility_index.csv"),help="Output CSV path.")

    # 2) collect
    pc = sub.add_parser("collect",help="Generate GLB + colored PLY from index for (optionally) selected categories.")
    pc.add_argument("--index", type=Path, required=True, help="CSV path produced by 'index'.")
    pc.add_argument("--dataset-dir", type=Path, required=True, help="Dataset root for resolving URDF/assets.")
    pc.add_argument("--out-dir", type=Path, default=Path("MobilityColored"), help="Output root for generated data.")
    pc.add_argument("--categories", type=str, default="",help="Comma-separated category names; empty = use all.")
    pc.add_argument("--joint-types", type=str, default="revolute",help="Allowed joint types, e.g., 'revolute,prismatic'.")
    pc.add_argument("--steps", type=int, default=10, help="Global discretization steps for every joint.")
    pc.add_argument("--num-combos", type=int, default=200, help="Number of random joint combinations per model.")
    pc.add_argument("--points", type=int, default=4096, help="Points per pose (for PLY).")
    pc.add_argument("--seed", type=int, default=0, help="Random seed.")
    pc.add_argument("--group-by-cat", action="store_true",help="Write outputs under <out_dir>/<Category>/<AnnoID>/...")
    pc.add_argument("--allow-duplicate-combos", action="store_true", help="Allow duplicate joint combinations.")
    pc.add_argument("--ply-ascii", action="store_true", help="Export ASCII PLY with RGB columns (inspectable).")
    pc.add_argument("--glb-bake", action="store_true", help="Bake textures/material colors to GLB vertex colors.")
    pc.add_argument("--workers", type=int, default=1, help="Process-level parallelism.")
    pc.add_argument("--init-only", action="store_true", help="Export only the initial pose per model.")
    pc.add_argument("--point-sampling", choices=["random", "even", "fps"], default="random",help="Surface sampling mode.")
    pc.add_argument("--fps-oversample", type=int, default=8,help="Oversampling factor (>=2) for even/fps modes.")

    # 3) h5 (per-category; includes normalization and preview)
    ph = sub.add_parser("h5",help="Export per-category H5 with category-wise normalization + previews + H5 structure print.")
    ph.add_argument("--in-root", type=Path, required=True, help="Input root from 'collect'.")
    ph.add_argument("--categories", type=str, default="",help="Comma-separated category names. Empty = auto-discover subfolders under --in-root.")
    ph.add_argument("--out-root", type=Path, required=True, help="Output root for H5 files.")
    ph.add_argument("--npoints", type=int, default=4096, help="Number of points per sample in H5.")
    ph.add_argument("--voxel-size", type=float, default=0.003,help="Voxel size before enforcing npoints & in stats.")
    ph.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction.")
    ph.add_argument("--test-frac", type=float, default=0.1, help="Test fraction.")
    ph.add_argument("--seed", type=int, default=42, help="Random seed.")
    ph.add_argument("--save-rgb", action="store_true", help="Store per-point RGB (uint8) in 'rgb'.")
    ph.add_argument("--workers", type=int, default=1, help="CPU workers for preprocessing.")
    ph.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="Float dtype for H5 data arrays.")
    ph.add_argument("--save-pose-idx", action="store_true", help="Also store integer pose_idx per sample.")
    ph.add_argument("--preview-samples", type=int, default=5,help="Number of preview (raw+norm) PLY pairs to export per category.")
    ph.add_argument("--preview-dir", type=Path, default=None,help="Optional preview output directory.")
    ph.add_argument("--stats-sample", type=int, default=0,help="Max #poses used to compute normalization stats per category (0 = use all).")
    ph.add_argument("--stats-frac", type=float, default=0.0,help="Fraction of poses used for stats (0 = use all). If both stats-sample and stats-frac are set, the smaller suggestion is used.")
    ph.add_argument("--proc-sample", type=int, default=0,help="Randomly limit #poses to export to H5 per category (0 = export all).")

    return ap.parse_args()


def main():
    args = parse_args()

    # ------------------------------
    # Stage 1: Indexing
    # ------------------------------
    if args.cmd == "index":
        from datamaker import build_index
        build_index(args.dataset_dir, args.out)
        return

    # ------------------------------
    # Stage 2: Collect (GLB + PLY)
    # ------------------------------
    if args.cmd == "collect":
        from datamaker.collect import collect_from_index
        cats: Optional[List[str]] = (
            [c.strip() for c in args.categories.split(",") if c.strip()]
            if args.categories else None
        )
        collect_from_index(
            args.index, args.dataset_dir, args.out_dir, categories=cats,
            joint_types=args.joint_types, steps=args.steps, num_combos=args.num_combos, points=args.points,
            seed=args.seed, group_by_cat=args.group_by_cat, allow_duplicate_combos=args.allow_duplicate_combos,
            ply_ascii=args.ply_ascii, glb_bake=args.glb_bake, workers=args.workers,
            init_only=args.init_only, point_sampling=args.point_sampling, fps_oversample=args.fps_oversample
        )
        return

    # ------------------------------
    # Stage 3: H5 export (per-category)
    # ------------------------------
    if args.cmd == "h5":
        from datamaker.h5_maker import make_h5_for_category
        def _discover_categories(in_root: Path):
            """Auto discover categories as direct subfolders under in_root."""
            return sorted([d.name for d in Path(in_root).iterdir() if d.is_dir()])

        cats = [c.strip() for c in (args.categories or "").split(",") if c.strip()]
        if not cats:
            cats = _discover_categories(args.in_root)
            print(f"[h5] auto-discovered categories under {args.in_root}: {cats}")

        for cat in cats:
            make_h5_for_category(
                in_root=args.in_root, out_root=args.out_root, category=cat,
                npoints=args.npoints, voxel_size=args.voxel_size, val_frac=args.val_frac, test_frac=args.test_frac,
                seed=args.seed, save_rgb=args.save_rgb, workers=args.workers, dtype=args.dtype,
                save_pose_idx=args.save_pose_idx, preview_samples=args.preview_samples, preview_dir=args.preview_dir,
                stats_sample=args.stats_sample, stats_frac=args.stats_frac, proc_sample=args.proc_smaple if hasattr(args, "proc_smaple") else args.proc_sample
            )
        return


if __name__ == "__main__":
    main()

'''
python make_partnet_dataset.py index `
  --dataset-dir partnet `
  --out partnet_index.csv
  
python make_partnet_dataset.py collect `
  --index partnet_index.csv `
  --dataset-dir partnet `
  --out-dir MobilityColored `
  --categories Eyeglasses,Scissors,Pliers,Box,FoldingChair,Laptop `
  --joint-types revolute `
  --steps 100 --num-combos 50 --points 20000 --seed 0 `
  --ply-ascii --point-sampling random --workers 8
  
python make_partnet_dataset.py h5 `
  --in-root MobilityColored `
  --categories Eyeglasses,Scissors,Pliers,Box,FoldingChair,Laptop `
  --out-root H5 `
  --npoints 20000 --voxel-size 0.001 `
  --val-frac 0.1 --test-frac 0.1 `
  --seed 42 --save-rgb --workers 8 `
  --dtype float32 `
  --preview-samples 6 `
  --stats-sample 0 `
  --proc-sample 0
'''