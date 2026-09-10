from __future__ import annotations
import json, math, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import h5py
except Exception:
    h5py = None

try:
    import open3d as o3d
except Exception:
    o3d = None
try:
    from plyfile import PlyData, PlyElement
except Exception:
    PlyData, PlyElement = None, None
try:
    import trimesh
except Exception:
    trimesh = None

# We keep using the normalization/statistics helpers from norm.py
from .norm import compute_category_stats, normalize_with_stats, list_category_samples

__all__ = ["make_h5_for_category", "print_h5_structure"]

# ------------------- Small I/O helpers -------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def _load_ply_points_colors(path: Path):
    """
    Robust PLY reader (Open3D -> plyfile -> trimesh fallback).
    Returns:
      xyz: (N, 3) float32
      rgb: Optional[(N, 3) uint8]
    """
    # Try open3d
    if o3d is not None:
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            xyz = np.asarray(pcd.points, dtype=np.float32)
            rgb = None
            if len(pcd.colors) > 0:
                col = np.asarray(pcd.colors, dtype=np.float32)
                if col.ndim == 2 and col.shape[1] >= 3:
                    rgb = np.clip((col[:, :3] * 255.0).round(), 0, 255).astype(np.uint8)
            return xyz, rgb
        except Exception:
            pass
    # Try plyfile
    if PlyData is not None:
        try:
            ply = PlyData.read(str(path))
            v = ply["vertex"]
            x = np.asarray(v["x"], dtype=np.float32)
            y = np.asarray(v["y"], dtype=np.float32)
            z = np.asarray(v["z"], dtype=np.float32)
            xyz = np.column_stack([x, y, z]).astype(np.float32)
            rgb = None
            if "red" in v:
                r = np.asarray(v["red"], dtype=np.uint8)
                g = np.asarray(v["green"], dtype=np.uint8)
                b = np.asarray(v["blue"], dtype=np.uint8)
                rgb = np.column_stack([r, g, b]).astype(np.uint8)
            return xyz, rgb
        except Exception:
            pass
    # Fallback trimesh
    if trimesh is not None:
        try:
            obj = trimesh.load(str(path), process=False)
            if hasattr(obj, "vertices"):
                xyz = np.asarray(obj.vertices, dtype=np.float32)
                rgb = None
                if hasattr(obj, "colors") and obj.colors is not None and len(obj.colors) == len(xyz):
                    C = np.asarray(obj.colors)
                    if C.ndim == 2 and C.shape[1] >= 3: rgb = C[:, :3].astype(np.uint8)
                return xyz, rgb
        except Exception:
            pass
    raise RuntimeError(f"Failed to read PLY: {path}")

def _voxel_down(xyz: np.ndarray, rgb: Optional[np.ndarray], voxel_size: float):
    """
    Simple voxel grid downsampling using integer grid hashing. We keep the first point per voxel.
    """
    if xyz.shape[0] == 0 or voxel_size <= 0: return xyz, rgb
    K = np.floor(xyz / float(voxel_size)).astype(np.int64)
    _, idx = np.unique(K, axis=0, return_index=True)
    idx = np.sort(idx)
    xyz_ds = xyz[idx]
    rgb_ds = rgb[idx] if rgb is not None and len(rgb) == len(xyz) else None
    return xyz_ds, rgb_ds

def _rand_downsample(xyz: np.ndarray, rgb: Optional[np.ndarray], npoints: int, seed: int):
    """
    Uniform random downsampling to exactly npoints.
    """
    if xyz.shape[0] <= npoints: return xyz, rgb
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(xyz.shape[0], size=npoints, replace=False))
    return xyz[idx], (rgb[idx] if rgb is not None and len(rgb) == len(xyz) else None)

def _pad_or_dup(xyz: np.ndarray, rgb: Optional[np.ndarray], npoints: int, seed: int):
    """
    If fewer than npoints, pad by sampling-with-replacement; if more, call _rand_downsample.
    """
    N = xyz.shape[0]
    if N == npoints: return xyz, rgb
    if N > npoints: return _rand_downsample(xyz, rgb, npoints, seed)
    rng = np.random.RandomState(seed)
    pad_idx = rng.choice(N, size=npoints - N, replace=True)
    xyz_out = np.concatenate([xyz, xyz[pad_idx]], axis=0)
    if rgb is not None and len(rgb) == len(xyz):
        rgb_out = np.concatenate([rgb, rgb[pad_idx]], axis=0)
    else:
        rgb_out = None
    return xyz_out, rgb_out

def _normalize_motors(angles: List[float], joints_meta: List[dict], D_pad: int) -> np.ndarray:
    """
    Normalize joint angles to [0,1] given per-joint limits. Pad to D_pad with NaN for compatibility.
    """
    vals = []
    import numpy as _np
    for i, a in enumerate(angles):
        j = joints_meta[i] if i < len(joints_meta) else None
        if j is None:
            v = _np.nan
        else:
            lo = float(j.get("limit_lower", j.get("lower", -_np.pi)))
            hi = float(j.get("limit_upper", j.get("upper",  _np.pi)))
            if not _np.isfinite(lo) or not _np.isfinite(hi) or hi <= lo:
                if str(j.get("type","")).lower().startswith("rev"): lo, hi = -_np.pi, _np.pi
                else: lo, hi = -0.5, 0.5
            t = (float(a) - lo) / (hi - lo + 1e-12)
            v = float(_np.clip(t, 0.0, 1.0))
        vals.append(v)
    while len(vals) < int(D_pad): vals.append(_np.nan)
    return np.array(vals[:int(D_pad)], dtype=np.float32).reshape(-1)

# ------------------- H5 writer -------------------
def _write_h5_shards(
    out_dir: Path, split_name: str, samples: List[dict], npoints: int, dtype: str,
    save_rgb: bool, save_pose_idx: bool, center: np.ndarray, scale: float
) -> int:
    """
    Write a split into HDF5 shards. We record the category-global normalization stats
    as file-level attributes and also keep per-sample center/scale datasets for backward compatibility.
    """
    out_split = _ensure_dir(out_dir / split_name); tot = len(samples)
    if tot == 0: return 0
    shard_size = 2048; nshards = math.ceil(tot / shard_size)
    dtype_np = np.float16 if dtype == "float16" else np.float32
    for si in range(nshards):
        sl = samples[si * shard_size : (si + 1) * shard_size]
        if not sl: break
        fn = out_split / f"shard-{si:05d}.h5"
        with h5py.File(str(fn), "w") as f:
            B = len(sl)
            f.create_dataset("data", shape=(B, npoints, 3), dtype=dtype_np)
            f.create_dataset("data_norm", shape=(B, npoints, 3), dtype=dtype_np)
            f.create_dataset("center", shape=(B, 3), dtype=np.float32)
            f.create_dataset("scale",  shape=(B,),   dtype=np.float32)
            Dmot = int(sl[0]["motors"].shape[0])
            f.create_dataset("motors", shape=(B, Dmot), dtype=np.float32)
            str_dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("anno_id", shape=(B,), dtype=str_dt)
            if save_pose_idx: f.create_dataset("pose_idx", shape=(B,), dtype=np.int32)
            if save_rgb:      f.create_dataset("rgb", shape=(B, npoints, 3), dtype=np.uint8)

            # category-wise stats as file attributes
            f.attrs["global_center"] = np.asarray(center, dtype=np.float32)
            f.attrs["global_scale"]  = np.float32(scale)
            f.attrs["normalization"] = "global_center_radius"

            for i, item in enumerate(sl):
                f["data"][i]      = item["data"].astype(dtype_np)
                f["data_norm"][i] = item["data_norm"].astype(dtype_np)
                f["center"][i]    = item["center"].astype(np.float32)
                f["scale"][i]     = np.float32(item["scale"])
                f["motors"][i]    = item["motors"].astype(np.float32)
                f["anno_id"][i]   = str(item["anno_id"])
                if save_pose_idx: f["pose_idx"][i] = int(item["pose_idx"])
                if save_rgb:      f["rgb"][i]      = item["rgb"].astype(np.uint8)
    return nshards

# ------------------- Preview (PLY) -------------------
def _save_preview_ply(path: Path, pts: np.ndarray, rgb: Optional[np.ndarray]) -> None:
    """
    Save a PLY quick-view file. Use trimesh if available; otherwise fall back to plyfile ASCII writer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if trimesh is not None:
        pc = trimesh.points.PointCloud(pts.astype(np.float32), colors=(rgb if rgb is not None else None))
        pc.export(str(path)); return
    if PlyElement is not None:
        V = pts.shape[0]
        vertex = np.empty(V, dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
        vertex["x"]=pts[:,0].astype("f4"); vertex["y"]=pts[:,1].astype("f4"); vertex["z"]=pts[:,2].astype("f4")
        if rgb is None or len(rgb)!=V: rgb = np.tile(np.array([200,200,200], dtype=np.uint8), (V,1))
        vertex["red"]=rgb[:,0]; vertex["green"]=rgb[:,1]; vertex["blue"]=rgb[:,2]
        PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(str(path)); return
    print(f"[WARN] Cannot write preview PLY: {path}", file=sys.stderr)

# =========================================================
# Windows-safe, top-level helpers for multiprocessing
# =========================================================
def _compute_center_scale_over_recs(recs: List[dict], voxel_size: float):
    """
    Two-pass statistics over a chosen subset of records (used when sampling is enabled):
      - center: mean of all points after (optional) voxel downsampling
      - scale : max radius (max ||x - center||)
    Top-level function so it is picklable under Windows 'spawn' mode.
    """
    import numpy as _np
    total_sum = _np.zeros(3, _np.float64)
    total_cnt = 0
    for r in recs:
        xyz, _ = _load_ply_points_colors(r["pc_path"])
        if voxel_size > 0:
            xyz, _ = _voxel_down(xyz, None, voxel_size)
        total_sum += xyz.astype(_np.float64).sum(axis=0)
        total_cnt += xyz.shape[0]
    if total_cnt <= 0:
        raise SystemExit("[H5] Empty subset after voxel filtering.")
    center = (total_sum / total_cnt).astype(_np.float32)

    max_r = 0.0
    for r in recs:
        xyz, _ = _load_ply_points_colors(r["pc_path"])
        if voxel_size > 0:
            xyz, _ = _voxel_down(xyz, None, voxel_size)
        if xyz.shape[0] == 0:
            continue
        d = _np.linalg.norm(xyz.astype(_np.float32) - center.reshape(1, 3), axis=1)
        mv = float(d.max())
        if mv > max_r:
            max_r = mv
    scale = float(max(max_r, 1e-6))
    return center, scale

def _h5_process_one(payload: dict, npoints: int, voxel_size: float, save_rgb: bool,
                    center: np.ndarray, scale: float) -> dict:
    """
    Per-sample processing (top-level version):
      - load PLY
      - optional voxel downsampling
      - pad/dup or random-downsample to npoints
      - global (category-wise) normalization
      - build output record
    """
    xyz, rgb = _load_ply_points_colors(payload["pc_path"])
    if voxel_size > 0:
        xyz, rgb = _voxel_down(xyz, rgb, voxel_size)
    xyz, rgb = _pad_or_dup(xyz, rgb, npoints, seed=payload["seed"])
    Pn = normalize_with_stats(xyz, center, scale)
    out = {
        "anno_id": payload["anno_id"], "pose_idx": payload["pose_idx"],
        "data": xyz.astype(np.float32), "data_norm": Pn.astype(np.float32),
        "center": np.asarray(center, dtype=np.float32), "scale": float(scale),
        "motors": payload["motors"].astype(np.float32),
    }
    if save_rgb:
        if rgb is None:
            rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
        out["rgb"] = rgb.astype(np.uint8)
    return out

def _h5_wrap(args):
    """
    Glue for multiprocessing.Pool.imap_unordered:
    (split_name, payload, npoints, voxel_size, save_rgb, center, scale) -> (split_name, out)
    Kept as a top-level function to be picklable under Windows.
    """
    split_name, payload, npoints, voxel_size, save_rgb, center, scale = args
    return split_name, _h5_process_one(payload, npoints, voxel_size, save_rgb, center, scale)

# ------------------- H5 main per-category -------------------
def make_h5_for_category(
    in_root: Path,
    out_root: Path,
    category: str,
    *,
    npoints: int = 4096,
    voxel_size: float = 0.003,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    save_rgb: bool = False,
    workers: int = 1,
    dtype: str = "float32",
    save_pose_idx: bool = False,
    preview_samples: int = 5,
    preview_dir: Optional[Path] = None,
    # Optional sampling knobs for speed/robustness
    stats_sample: int = 0,     # (0 = use all) max number of poses used for stats
    stats_frac: float = 0.0,   # fraction (0 = all); if both are set, use the smaller
    proc_sample: int = 0,      # (0 = export all) randomly limit #poses exported to H5
) -> None:
    """
    Export train/val/test H5 for a single *category* with *category-wise* normalization.
    Additionally export a few preview PLY pairs (raw + normalized) and print the H5 structure.

    Args:
      in_root:        Root produced by `collect_from_index`.
      out_root:       Output root for H5 files; category subfolder will be created.
      category:       Category name to export (e.g., "Door").
      npoints:        Fixed number of points per sample (pad/dup or downsample).
      voxel_size:     Voxel size before enforcing npoints (also used in stats).
      val_frac/test_frac: Split fractions (train is the remainder).
      seed:           Random seed for splitting and resampling.
      save_rgb:       If True, store 'rgb' (uint8) per point.
      workers:        CPU workers for preprocessing (per-category).
      dtype:          Float dtype for data arrays.
      save_pose_idx:  If True, store pose_idx per sample.
      preview_samples:Number of (raw + norm) PLY preview pairs to export (0 to disable).
      preview_dir:    Optional path to store previews; default: <out_root>/<category>/preview.
      stats_sample:   Limit the number of poses used to compute normalization statistics.
      stats_frac:     Use a fraction of poses for statistics; use min(stats_sample, stats_frac * N).
      proc_sample:    Limit the number of poses that are actually exported to H5.
    """
    if h5py is None: raise SystemExit("h5py is required. pip install h5py")
    recs = list_category_samples(in_root, category)
    if not recs:
        print(f"[H5] No samples for category: {category}", file=sys.stderr); return

    rng = np.random.RandomState(seed)

    # --------------------------------------------
    # (1) Choose subset for statistics (optional)
    # --------------------------------------------
    recs_for_stats = recs
    if stats_sample > 0 or (stats_frac > 0 and stats_frac < 1.0):
        k_frac = int(len(recs) * float(stats_frac)) if stats_frac > 0 else len(recs)
        k_abs  = int(stats_sample) if stats_sample > 0 else len(recs)
        k = max(1, min(len(recs), k_frac, k_abs))
        idx = rng.choice(len(recs), size=k, replace=False)
        recs_for_stats = [recs[i] for i in idx]
        print(f"[H5] {category}: using {k}/{len(recs)} poses for stats (sampling).")

    # --------------------------------------------
    # (2) Compute category-wise stats
    # --------------------------------------------
    if recs_for_stats is recs:
        # Full pass (original behavior)
        center, scale, _ = compute_category_stats(in_root, category, voxel_size=voxel_size)
    else:
        # Sampled statistics (top-level function; Windows-safe)
        center, scale = _compute_center_scale_over_recs(recs_for_stats, voxel_size)
    print(f"[H5] {category} center={center.tolist()}, scale={scale:.6f}")

    # --------------------------------------------
    # (3) Optionally subsample the exported set
    # --------------------------------------------
    if proc_sample > 0 and proc_sample < len(recs):
        idx = rng.choice(len(recs), size=int(proc_sample), replace=False)
        recs = [recs[i] for i in idx]
        print(f"[H5] {category}: exporting only {len(recs)} poses to H5 (proc_sample).")

    # motors padding length (keep compatibility with older readers)
    D_pad = max(len(r["joint_meta"].get("joints", [])) for r in recs)
    for r in recs:
        obj = _load_json(r["angles_path"])
        ang = obj.get("angles")
        if ang is None:
            if "theta_rad" in obj: ang = [float(obj["theta_rad"])]
            else: ang = []
        r["motors"] = _normalize_motors(ang, r["joint_meta"].get("joints", []), D_pad)

    # --------------------------------------------
    # (4) Split into train/val/test
    # --------------------------------------------
    idx = np.arange(len(recs)); rng.shuffle(idx)
    nval = int(len(recs) * val_frac); ntest = int(len(recs) * test_frac)
    val_set = set(idx[:nval]); test_set = set(idx[nval : nval + ntest])
    splits = {"train": [], "val": [], "test": []}
    for i in range(len(recs)):
        k = "train"
        if i in val_set: k = "val"
        elif i in test_set: k = "test"
        splits[k].append(recs[i])

    # --------------------------------------------
    # (5) Build tasks for processing
    # --------------------------------------------
    tasks = []
    for split_name, lst in splits.items():
        for i, r in enumerate(lst):
            tasks.append((split_name, dict(
                pc_path=r["pc_path"], anno_id=r["anno_id"], pose_idx=r["pose_idx"],
                motors=r["motors"], seed=seed + i
            )))

    # --------------------------------------------
    # (6) Process tasks (serial or parallel)
    #     Also do reservoir sampling for preview pairs.
    # --------------------------------------------
    results = {"train": [], "val": [], "test": []}
    reservoir: List[Tuple[str, dict]] = []
    K = max(0, int(preview_samples))
    seen = 0

    def _process_one(payload):
        # Serial path uses a local function (no pickling overhead).
        return _h5_process_one(payload, npoints, voxel_size, save_rgb, center, scale)

    if workers <= 1:
        for split_name, payload in tqdm(tasks, total=len(tasks), ncols=120, desc=f"[{category}] proc"):
            out = _process_one(payload)
            results[split_name].append(out)
            if K > 0:
                if len(reservoir) < K: reservoir.append((split_name, out))
                else:
                    j = random.randint(0, seen)
                    if j < K: reservoir[j] = (split_name, out)
                seen += 1
    else:
        # Windows-safe multiprocessing: use a TOP-LEVEL wrapper (picklable)
        import multiprocessing as mp
        with mp.Pool(processes=int(workers)) as pool:
            args_iter = (
                (sname, payload, npoints, voxel_size, save_rgb, center, scale)
                for sname, payload in tasks
            )
            it = pool.imap_unordered(_h5_wrap, args_iter, chunksize=64)
            for split_name, out in tqdm(it, total=len(tasks), ncols=120, desc=f"[{category}] proc*{workers}"):
                results[split_name].append(out)
                if K > 0:
                    if len(reservoir) < K: reservoir.append((split_name, out))
                    else:
                        j = random.randint(0, seen)
                        if j < K: reservoir[j] = (split_name, out)
                    seen += 1

    # --------------------------------------------
    # (7) Write H5 shards
    # --------------------------------------------
    out_cat_dir = _ensure_dir(Path(out_root) / category)
    nsh_train = _write_h5_shards(out_cat_dir, "train", results["train"], npoints, dtype, save_rgb, save_pose_idx, center, scale)
    nsh_val   = _write_h5_shards(out_cat_dir, "val",   results["val"],   npoints, dtype, save_rgb, save_pose_idx, center, scale)
    nsh_test  = _write_h5_shards(out_cat_dir, "test",  results["test"],  npoints, dtype, save_rgb, save_pose_idx, center, scale)

    # --------------------------------------------
    # (8) Write category metadata
    # --------------------------------------------
    meta = dict(
        total=sum(len(v) for v in results.values()),
        npoints=int(npoints), voxel_size=float(voxel_size),
        val_frac=float(val_frac), test_frac=float(test_frac),
        save_rgb=bool(save_rgb), dtype=str(dtype),
        Dmot=int(len(results["train"][0]["motors"]) if results["train"] else 0),
        nshards_train=int(nsh_train), nshards_val=int(nsh_val), nshards_test=int(nsh_test),
        global_center=[float(x) for x in center.reshape(-1)],
        global_scale=float(scale),
        normalization="global_center_radius",
    )
    with open(out_cat_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --------------------------------------------
    # (9) Export preview PLYs (raw + normalized)
    # --------------------------------------------
    if K > 0 and reservoir:
        # If didn't pass --preview-dir, use "<out_root>/<category>/preview"
        if preview_dir is None:
            pdir = out_cat_dir / "preview"
        else:
            pdir = Path(preview_dir)
            # Only join when it's a relative path (e.g. "preview" or "viz/preview")
            if not pdir.is_absolute():
                pdir = out_cat_dir / pdir

        pdir.mkdir(parents=True, exist_ok=True)
        for i, (split_name, item) in enumerate(reservoir):
            anno = item["anno_id"];
            pose = int(item.get("pose_idx", -1));
            rgb = item.get("rgb")
            _save_preview_ply(pdir / f"raw_{split_name}_{i:03d}_{anno}_pose{pose:03d}.ply", item["data"], rgb)
            _save_preview_ply(pdir / f"norm_{split_name}_{i:03d}_{anno}_pose{pose:03d}.ply", item["data_norm"], rgb)
        with open(pdir / "normalization_stats.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "global_center": [float(x) for x in center],
                    "global_scale": float(scale),
                    "normalization": "global_center_radius",
                    "n_preview_pairs": len(reservoir),
                },
                f, indent=2, ensure_ascii=False
            )

    # --------------------------------------------
    # (10) Print H5 structures
    # --------------------------------------------
    print_h5_structure(out_cat_dir)
    print(f"[{category}] wrote H5 to {out_cat_dir}")


def print_h5_structure(out_cat_dir: Path) -> None:
    """
    Print dataset names, shapes, dtypes + file-level attributes for all shards under a category.
    This is useful for sanity-checking that the schema is stable and compatible.
    """
    for split in ("train", "val", "test"):
        sdir = Path(out_cat_dir) / split
        if not sdir.exists(): continue
        for h5_path in sorted(sdir.glob("*.h5")):
            try:
                with h5py.File(str(h5_path), "r") as f:
                    print(f"\n[H5] {h5_path.relative_to(out_cat_dir)}")
                    for key in f.keys():
                        ds = f[key]
                        print(f"  - {key:10s} dtype={ds.dtype} shape={ds.shape}")
                    if len(f.attrs) > 0:
                        print("  attributes:")
                        for k in f.attrs.keys():
                            v = f.attrs[k]
                            v_show = v.tolist() if isinstance(v, np.ndarray) else v
                            print(f"    * {k}: {v_show}")
            except Exception as e:
                print(f"[WARN] failed to read {h5_path}: {e}", file=sys.stderr)
