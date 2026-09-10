from __future__ import annotations
import json, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None
try:
    from plyfile import PlyData
except Exception:
    PlyData = None
try:
    import trimesh
except Exception:
    trimesh = None

from tqdm import tqdm

__all__ = ["compute_category_stats", "normalize_with_stats", "list_category_samples"]

def list_category_samples(in_root: Path, category: str) -> List[dict]:
    """
    Enumerate samples under <in_root>/<category>/<AnnoID>/pose_xxx/.
    Returns a list of dicts with keys: pc_path, angles_path, anno_id, joint_meta, pose_idx.
    """
    cat_dir = Path(in_root) / category
    items = []
    if not cat_dir.exists(): return items
    for anno_dir in sorted([d for d in cat_dir.iterdir() if d.is_dir()]):
        anno_id = anno_dir.name
        joint_json = anno_dir / "joint.json"
        if not joint_json.exists(): continue
        try:
            with joint_json.open("r", encoding="utf-8") as f: joint_meta = json.load(f)
            if not joint_meta.get("joints"): continue
        except Exception:
            continue
        for pose_dir in sorted([p for p in anno_dir.iterdir() if p.is_dir() and p.name.startswith("pose_")]):
            pc_path = pose_dir / "pointcloud.ply"
            ang_path = pose_dir / "angles.json"
            if not pc_path.exists() or not ang_path.exists(): continue
            try: pose_idx = int(pose_dir.name.split("_")[-1])
            except Exception: pose_idx = -1
            items.append(dict(pc_path=pc_path, angles_path=ang_path, anno_id=anno_id, joint_meta=joint_meta, pose_idx=pose_idx))
    return items

def _load_ply_points_colors(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read PLY into (xyz float32, rgb uint8 or None). Try open3d -> plyfile -> trimesh."""
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

def _voxel_down(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if xyz.shape[0] == 0 or voxel_size <= 0: return xyz
    K = np.floor(xyz / float(voxel_size)).astype(np.int64)
    _, idx = np.unique(K, axis=0, return_index=True)
    return xyz[np.sort(idx)]

def compute_category_stats(in_root: Path, category: str, voxel_size: float = 0.003) -> Tuple[np.ndarray, float, int]:
    """
    Compute category-wise normalization stats via two passes:
      - center: mean of all points (after voxel downsampling)
      - scale:  max distance to the center
    Args:
      in_root:    Root of the collected dataset (the output of `collect_from_index`).
      category:   Category folder name (e.g., "Door").
      voxel_size: Voxel size for downsampling before statistics (speeds up & reduces memory).

    Returns:
      (center: float32[3], scale: float, num_samples: int)
    """
    recs = list_category_samples(in_root, category)
    if not recs:
        raise SystemExit(f"[Norm] No samples under {in_root}/{category}")
    # pass 1: mean
    total_sum = np.zeros(3, np.float64); total_cnt = 0
    for r in tqdm(recs, ncols=120, desc=f"[{category}] stats pass1(mean)"):
        xyz, _ = _load_ply_points_colors(r["pc_path"])
        xyz = _voxel_down(xyz, voxel_size)
        total_sum += xyz.astype(np.float64).sum(axis=0)
        total_cnt += xyz.shape[0]
    if total_cnt <= 0: raise SystemExit(f"[Norm] Empty after voxel filter: {category}")
    center = (total_sum / total_cnt).astype(np.float32)
    # pass 2: radius
    max_r = 0.0
    for r in tqdm(recs, ncols=120, desc=f"[{category}] stats pass2(radius)"):
        xyz, _ = _load_ply_points_colors(r["pc_path"])
        xyz = _voxel_down(xyz, voxel_size)
        if xyz.shape[0] == 0: continue
        d = np.linalg.norm(xyz.astype(np.float32) - center.reshape(1, 3), axis=1)
        max_r = max(max_r, float(d.max()))
    scale = float(max(max_r, 1e-6))
    return center, scale, len(recs)

def normalize_with_stats(xyz: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Apply category-wise normalization: (xyz - center)/scale."""
    return (xyz.astype(np.float32) - center.reshape(1, 3)) / (float(scale) + 1e-12)
