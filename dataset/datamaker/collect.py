from __future__ import annotations
import csv, json, math, random, re, sys, xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import trimesh
from PIL import Image

import pybullet as p
import pybullet_data

__all__ = ["collect_from_index"]

# ----------------------- Config -----------------------
@dataclass
class Config:
    points_per_pose: int = 4096
    vel_epsilon: float = 1e-3
    pos_tolerance: float = 1e-4
    stable_hold_steps: int = 30
    max_settle_steps: int = 2400
    physics_timestep: float = 1.0 / 240.0
    motor_force: float = 50.0
    ply_ascii: bool = False
    glb_bake: bool = False  # bake textures to vertex colors for GLB export
    point_sampling: str = "random"  # "random" | "fps" | "even"
    fps_oversample: int = 8         # oversample factor for fps/even

# --------------------- CSV helpers ---------------------
def _read_index_csv(index_csv: Path) -> List[Dict[str, str]]:
    rows = []
    with Path(index_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f): rows.append(r)
    return rows

def row_get_category(row: Dict[str, str]) -> str:
    for key in ("model_cat","model.category","category","meta.model_cat"):
        v = row.get(key) or ""
        if v: return str(v).strip()
    return ""

def include_row_by_cats(row: Dict[str, str], cats: Optional[List[str]]) -> bool:
    if not cats: return True
    return row_get_category(row).lower() in {c.strip().lower() for c in cats}

def choose_anno_id(row: Dict[str, str]) -> str:
    cand = [row.get("anno_id") or row.get("meta.anno_id"), row.get("model_id")]
    md = row.get("model_dir")
    if md:
        try: cand.append(Path(md).name)
        except Exception: pass
    ur = row.get("urdf_relpath")
    if ur:
        try: cand.append(Path(ur).parts[0])
        except Exception: pass
    cand.append(row.get("id"))
    for c in cand:
        if c and str(c).strip(): return str(c).strip()
    return "unknown"

# ---------------- Transforms ----------------
def quaternion_to_matrix(q):
    R = np.array(p.getMatrixFromQuaternion(q), dtype=np.float64).reshape(3,3)
    T = np.eye(4); T[:3,:3] = R; return T

def pose_to_matrix(pos, orn):
    T = quaternion_to_matrix(orn); T[:3,3] = np.array(pos, dtype=np.float64); return T

def get_link_world_pose(body_id, link_index):
    if link_index == -1:
        pos, orn = p.getBasePositionAndOrientation(body_id); return np.array(pos), np.array(orn)
    st = p.getLinkState(body_id, link_index, computeForwardKinematics=1)
    if len(st) >= 6 and st[4] is not None and st[5] is not None: pos, orn = st[4], st[5]
    else: pos, orn = st[0], st[1]
    return np.array(pos), np.array(orn)

# --------------- URDF materials (kept for parity) ----------------
def parse_urdf_materials(urdf_path: Path) -> Dict[str, Dict[str, object]]:
    """
    Parse URDF <material> and <visual> blocks.
    Return: { mesh_filename(str): {"rgba":[r,g,b,a]|None, "texture": "rel/or/abs/path"|None} }
    (Kept for parity with original logic; not strictly required for sampling path.)
    """
    out: Dict[str, Dict[str, object]] = {}
    if not urdf_path.exists(): return out
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return out

    named: Dict[str, Dict[str, object]] = {}
    for m in root.findall(".//material"):
        name = m.get("name")
        if not name: continue
        col = m.find("color"); tex = m.find("texture")
        rgba = None
        if col is not None and col.get("rgba"):
            try: rgba = [float(x) for x in col.get("rgba").split()]
            except Exception: rgba = None
        tpath = tex.get("filename") if (tex is not None and tex.get("filename")) else None
        named[name] = {"rgba": rgba, "texture": tpath}

    for v in root.findall(".//visual"):
        mesh = v.find("geometry/mesh")
        if mesh is None or not mesh.get("filename"): continue
        meshfn = mesh.get("filename")
        rgba = None; tpath = None
        mat = v.find("material")
        if mat is not None:
            col = mat.find("color"); tex = mat.find("texture")
            if col is not None and col.get("rgba"):
                try: rgba = [float(x) for x in col.get("rgba").split()]
                except Exception: rgba = None
            if tex is not None and tex.get("filename"): tpath = tex.get("filename")
            ref = mat.get("name")
            if (rgba is None or tpath is None) and ref and ref in named:
                if rgba is None: rgba = named[ref].get("rgba")
                if tpath is None: tpath = named[ref].get("texture")
        out[meshfn] = {"rgba": rgba, "texture": tpath}
    return out

# ---------------- Visual mesh loader ----------------
def load_visual_mesh(shape, urdf_dir: Path, urdf_mats=None) -> Optional[trimesh.Trimesh]:
    """
    Only load & transform the mesh. **Do not** paint fallback colors here.
    Keep the 3rd param for backward compatibility (ignored).
    """
    body_uid, link_idx, geom_type, dimensions, filename, lv_pos, lv_orn = shape[:7]

    mesh = None
    if geom_type == p.GEOM_MESH and filename:
        raw = filename.decode('utf-8','ignore') if isinstance(filename,(bytes,bytearray)) else str(filename)
        pth = Path(raw)
        candidates = [pth if pth.is_absolute() else urdf_dir / pth, urdf_dir.parent / pth]
        try: candidates.append(Path(pybullet_data.getDataPath()) / pth.name)
        except Exception: pass
        found = None
        for c in candidates:
            if c.exists(): found = c; break
        if found is None: return None
        mesh = trimesh.load(found, force="mesh", skip_missing=True, process=False)
        dims = np.array(dimensions, dtype=float).reshape(-1)
        try:
            if dims.size == 3 and not np.allclose(dims, 1.0): mesh.apply_scale(dims)
            elif dims.size == 1 and not np.isclose(dims[0], 1.0): mesh.apply_scale([dims[0]] * 3)
        except Exception: pass

    elif geom_type == p.GEOM_BOX:
        dims = np.array(dimensions, dtype=float).reshape(-1)
        if dims.size >= 3: mesh = trimesh.creation.box(extents=2.0 * dims[:3])
    elif geom_type == p.GEOM_SPHERE:
        r = float(dimensions[0]) if len(dimensions) >= 1 else 0.05
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif geom_type == p.GEOM_CYLINDER:
        if len(dimensions) >= 2:
            r, h = float(dimensions[0]), float(dimensions[1])
            mesh = trimesh.creation.cylinder(radius=r, height=h, sections=48)
    elif geom_type == p.GEOM_CAPSULE:
        if len(dimensions) >= 2:
            r, h = float(dimensions[0]), float(dimensions[1])
            mesh = trimesh.creation.capsule(radius=r, height=h, count=[24, 24])

    if mesh is None or mesh.is_empty:
        return None

    # apply local visual transform
    T_local = quaternion_to_matrix(lv_orn)
    T_local[:3,3] = np.array(lv_pos, dtype=np.float64)
    mesh.apply_transform(T_local)
    return mesh

# ------------- Material helpers (Kd per face) ----------------
def _material_color_to_rgba255(mat) -> Optional[np.ndarray]:
    """Read a material's inherent color (MTL Kd/Ka etc.) and convert to uint8 RGBA."""
    if mat is None: return None
    col = None
    for key in ('main_color', 'diffuse', 'Kd', 'ambient', 'Ka'):
        if hasattr(mat, key):
            col = getattr(mat, key)
            if col is not None: break
    if col is None: return None
    col = np.array(col, dtype=np.float32).reshape(-1)
    if col.size >= 3:
        if col.max() <= 1.0 + 1e-6: col = col * 255.0
        col = np.clip(col, 0, 255)
        if col.size == 3: col = np.append(col, 255.0)
        return col.astype(np.uint8)
    return None

def _face_rgba_from_materials(mesh: "trimesh.Trimesh") -> Optional[np.ndarray]:
    vis = getattr(mesh, 'visual', None)
    if vis is None or getattr(vis, 'kind', '') != 'texture': return None
    mats = getattr(vis, 'material', None)
    if mats is None: return None
    if isinstance(mats, (list, tuple, np.ndarray)):
        mat_list = list(mats)
    else:
        mat_list = [mats]
    face_mats = getattr(vis, 'face_materials', None)
    if face_mats is None or len(face_mats) != len(mesh.faces):
        face_mats = np.zeros(len(mesh.faces), dtype=np.int64)
    else:
        face_mats = np.asarray(face_mats, dtype=np.int64)

    cache_rgba = [_material_color_to_rgba255(m) for m in mat_list]
    if all(c is None for c in cache_rgba):
        return None
    fc = np.tile(np.array([180,180,180,255], dtype=np.uint8), (len(mesh.faces), 1))
    for midx, rgba in enumerate(cache_rgba):
        if rgba is None: continue
        mask = (face_mats == midx)
        if np.any(mask): fc[mask] = rgba
    return fc

# ------------- Geometry utils ----------------
def barycentric_weights(tri: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    tri: (N,3,3) triangle vertices, pts: (N,3)
    """
    v0 = tri[:,1] - tri[:,0]; v1 = tri[:,2] - tri[:,0]; v2 = pts - tri[:,0]
    d00 = np.einsum("ij,ij->i", v0, v0); d01 = np.einsum("ij,ij->i", v0, v1); d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0); d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01 + 1e-18
    v = (d11 * d20 - d01 * d21) / denom; w = (d00 * d21 - d01 * d20) / denom; u = 1.0 - v - w
    return np.column_stack([u, v, w])

def _fps_idx(points: np.ndarray, k: int) -> np.ndarray:
    N = len(points)
    if k >= N: return np.arange(N, dtype=np.int64)
    idx = np.empty(k, dtype=np.int64); far = int(np.random.randint(0, N)); d2 = np.full(N, np.inf, dtype=np.float64)
    for i in range(k):
        idx[i] = far
        dist = np.sum((points - points[far]) ** 2, axis=1)
        d2 = np.minimum(d2, dist); far = int(np.argmax(d2))
    return idx

def _sample_surface_even_fps(mesh: "trimesh.Trimesh", n: int, oversample: int) -> Tuple[np.ndarray, np.ndarray]:
    import trimesh.sample as ts
    m = max(1, int(n * max(2, oversample)))
    pts_dense, fidx_dense = ts.sample_surface(mesh, m)
    sel = _fps_idx(pts_dense, n)
    return pts_dense[sel], fidx_dense[sel]

# ------------- Sampling with color (strictly original order) -------------
def sample_piece_points_with_color(mesh: "trimesh.Trimesh", n: int,
                                   method: str = "random", oversample: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample n points and colors on one mesh.

    Color priority:
      A) If TextureVisuals with UV: for each point, use its face's material:
         - if that material has an image: sample texture via barycentric uv
         - else use that material's Kd color (main_color/diffuse)
      B) Else use face_colors, then vertex_colors (barycentric)
      C) Else use per-face Kd via (materials + face_materials)
      D) Else fallback gray
    """
    import trimesh.sample as ts
    if n <= 0 or mesh.is_empty:
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.uint8)

    if method.lower() == "random":
        pts, fidx = ts.sample_surface(mesh, n)
    else:  # "fps" / "even"
        pts, fidx = _sample_surface_even_fps(mesh, n, oversample=max(2, int(oversample)))

    rgb = np.tile(np.array([180,180,180], dtype=np.uint8), (pts.shape[0], 1))  # fallback
    vis = getattr(mesh, 'visual', None)

    # --- A. texture path (robust): try baking first, then fall back to manual UV ---
    if vis is not None and getattr(vis, "kind", "") == "texture":
        baked_ok = False
        try:
            # 1) Try baking textures/materials into vertex colors
            baked_vis = vis.to_color()  # -> ColorVisuals
            # Use baked colors to fill rgb directly
            # Prefer face_colors; fall back to vertex_colors (barycentric interpolation)
            fc = getattr(baked_vis, "face_colors", None)
            if fc is not None and len(fc) == len(mesh.faces):
                rgb = fc[fidx, :3].astype(np.uint8)
                baked_ok = True
            else:
                vc = getattr(baked_vis, "vertex_colors", None)
                if vc is not None and len(vc) == len(mesh.vertices):
                    faces = mesh.faces[fidx]
                    tris = mesh.vertices[faces]
                    wts = barycentric_weights(tris, pts)
                    c0, c1, c2 = vc[faces[:, 0], :3], vc[faces[:, 1], :3], vc[faces[:, 2], :3]
                    rgb = (wts[:, [0]] * c0 + wts[:, [1]] * c1 + wts[:, [2]] * c2).astype(np.uint8)
                    baked_ok = True
        except Exception:
            baked_ok = False

        if not baked_ok:
            # print("Baked Failed back to UV")
            mats = getattr(vis, "material", None) or getattr(vis, "materials", None)
            if mats is not None:
                mat_list = list(mats) if isinstance(mats, (list, tuple, np.ndarray)) else [mats]
                face_mats = getattr(vis, "face_materials", None)
                if face_mats is None or len(face_mats) != len(mesh.faces):
                    face_mats = np.zeros(len(mesh.faces), dtype=np.int64)
                else:
                    face_mats = np.asarray(face_mats, dtype=np.int64)

                mat_imgs, mat_rgba = [], []
                for mtl in mat_list:
                    try:
                        img = getattr(mtl, "image", None)
                        mat_imgs.append(None if img is None else np.asarray(img.convert("RGBA")))
                    except Exception:
                        mat_imgs.append(None)
                    mat_rgba.append(_material_color_to_rgba255(mtl))

                faces = mesh.faces[fidx]
                tris = mesh.vertices[faces]
                wts = barycentric_weights(tris, pts)
                uv = getattr(vis, "uv", None)
                if uv is not None:
                    uv_face = uv[faces]
                    uv_points = (uv_face * wts[..., None]).sum(axis=1)
                    u = np.mod(uv_points[:, 0], 1.0)
                    v = np.mod(uv_points[:, 1], 1.0)
                    fm_for_points = face_mats[fidx]
                    for midx in np.unique(fm_for_points):
                        mask = (fm_for_points == midx)
                        img = mat_imgs[midx] if midx < len(mat_imgs) else None
                        if img is not None:
                            H, W = img.shape[0], img.shape[1]
                            x = (u[mask] * (W - 1)).astype(np.int32)
                            y = ((1.0 - v[mask]) * (H - 1)).astype(np.int32)
                            rgb[mask] = img[y, x, :3].astype(np.uint8)
                        else:
                            rgba = mat_rgba[midx] if midx < len(mat_rgba) else None
                            if rgba is not None:
                                rgb[mask] = rgba[:3]

    unresolved = (rgb[:,0]==180) & (rgb[:,1]==180) & (rgb[:,2]==180)

    # --- B. face_colors / vertex_colors ---
    if np.any(unresolved) and vis is not None:
        try:
            fc = np.asanyarray(getattr(vis, 'face_colors', None), dtype=np.uint8)
            if fc is not None and len(fc) == len(mesh.faces):
                rgb[unresolved] = fc[fidx[unresolved], :3]
                unresolved = (rgb[:,0]==180) & (rgb[:,1]==180) & (rgb[:,2]==180)
        except Exception:
            pass

        if np.any(unresolved):
            try:
                vc = np.asanyarray(getattr(vis, 'vertex_colors', None), dtype=np.uint8)
                if vc is not None and len(vc) == len(mesh.vertices):
                    faces = mesh.faces[fidx[unresolved]]
                    tris  = mesh.vertices[faces]
                    wts   = barycentric_weights(tris, pts[unresolved])
                    c0 = vc[faces[:,0], :3]; c1 = vc[faces[:,1], :3]; c2 = vc[faces[:,2], :3]
                    rgb[unresolved] = (wts[:,[0]]*c0 + wts[:,[1]]*c1 + wts[:,[2]]*c2).astype(np.uint8)
                    unresolved = (rgb[:,0]==180) & (rgb[:,1]==180) & (rgb[:,2]==180)
            except Exception:
                pass

    # --- C. per-face Kd via materials+face_materials ---
    if np.any(unresolved):
        fc_from_mtl = _face_rgba_from_materials(mesh)
        if fc_from_mtl is not None:
            rgb[unresolved] = fc_from_mtl[fidx[unresolved], :3]

    return pts, rgb

def sample_model_points_colored(pieces: List["trimesh.Trimesh"], total_points: int,
                                method: str = "random", oversample: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    if total_points <= 0 or len(pieces) == 0:
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.uint8)
    areas = np.array([max(p.area, 1e-9) for p in pieces], dtype=float)
    frac = areas / areas.sum()
    alloc = np.maximum((frac * total_points).astype(int), 0)
    # rounding fix
    while alloc.sum() < total_points:
        alloc[np.argmax(frac - alloc/total_points)] += 1
    while alloc.sum() > total_points:
        i = np.argmax(alloc)
        if alloc[i] > 0: alloc[i] -= 1
        else: break
    all_pts, all_rgb = [], []
    for m, n in zip(pieces, alloc):
        if n <= 0: continue
        pts, rgb = sample_piece_points_with_color(m, int(n), method=method, oversample=oversample)
        if pts.shape[0] > 0: all_pts.append(pts); all_rgb.append(rgb)
    if not all_pts: return np.zeros((0,3)), np.zeros((0,3), dtype=np.uint8)
    return np.vstack(all_pts), np.vstack(all_rgb)

# ------------------ Joints & combinations ------------------
def joint_type_name(jtype: int) -> str:
    return {p.JOINT_REVOLUTE: "revolute", p.JOINT_PRISMATIC: "prismatic",
            p.JOINT_PLANAR: "planar", p.JOINT_FIXED: "fixed"}.get(jtype, f"type_{jtype}")

def find_joints(body_id: int, allow_types: List[str], name_regex: Optional[str]) -> List[Dict[str, object]]:
    allow = set(t.strip().lower() for t in allow_types)
    pattern = re.compile(name_regex) if name_regex else None
    out = []
    for j in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, j)
        jtype = info[2]; name = info[1].decode('utf-8','ignore')
        tname = joint_type_name(jtype).lower()
        if tname not in allow: continue
        if pattern and not pattern.fullmatch(name): continue
        lower, upper = float(info[8]), float(info[9])
        if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
            if jtype == p.JOINT_REVOLUTE: lower, upper = -math.pi, math.pi
            elif jtype == p.JOINT_PRISMATIC: lower, upper = -0.5, 0.5
            else: continue
        out.append({"index": j, "name": name, "type": tname, "lower": lower, "upper": upper})
    if not out:  # fallback to type-only
        for j in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, j)
            jtype = info[2]; name = info[1].decode('utf-8','ignore'); tname = joint_type_name(jtype).lower()
            if tname not in allow: continue
            lower, upper = float(info[8]), float(info[9])
            if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
                if jtype == p.JOINT_REVOLUTE: lower, upper = -math.pi, math.pi
                elif jtype == p.JOINT_PRISMATIC: lower, upper = -0.5, 0.5
                else: continue
            out.append({"index": j, "name": name, "type": tname, "lower": lower, "upper": upper})
    return out

def build_per_joint_grids(joints, steps, steps_override: Dict[str,int]):
    grids = []
    for j in joints:
        n = int(steps_override.get(j["name"], steps))
        grid = np.linspace(float(j["lower"]), float(j["upper"]), n, dtype=float)
        grids.append(grid)
    return grids

def random_combinations(grids: List[np.ndarray], num: int, seed: int, unique=True, max_tries_factor=20):
    rng = random.Random(seed)
    if not grids: return []
    total = 1
    for g in grids: total *= len(g)
    picks = []
    if not unique:
        for _ in range(num): picks.append([rng.choice(list(g)) for g in grids])
        return picks
    target = min(num, total); seen=set(); tries=0; max_tries=max_tries_factor*target
    while len(picks)<target and tries<max_tries:
        key = tuple(rng.randrange(len(g)) for g in grids)
        if key not in seen:
            seen.add(key)
            picks.append([grids[i][idx] for i,idx in enumerate(key)])
        tries += 1
    return picks

# ---------------- Simulation & export ----------------
def settle_multi_joints(body_id: int, joint_indices: List[int], targets: List[float], cfg: Config):
    for j in range(p.getNumJoints(body_id)):
        p.setJointMotorControl2(body_id, j, controlMode=p.VELOCITY_CONTROL, force=0.0)
    for j,tgt in zip(joint_indices, targets):
        p.setJointMotorControl2(body_id, j, controlMode=p.POSITION_CONTROL, targetPosition=float(tgt), force=cfg.motor_force)
    stable=0
    for _ in range(cfg.max_settle_steps):
        p.stepSimulation()
        ok=True
        for j,tgt in zip(joint_indices, targets):
            pos, vel, *_ = p.getJointState(body_id, j)
            if abs(pos - tgt) > cfg.pos_tolerance or abs(vel) > cfg.vel_epsilon:
                ok=False; break
        if ok:
            stable += 1
            if stable >= cfg.stable_hold_steps: break
        else:
            stable = 0

def save_pointcloud_ply(path: Path, pts: np.ndarray, rgb: np.ndarray, ascii_flag: bool):
    if ascii_flag:
        try:
            from plyfile import PlyData, PlyElement
            vertex = np.empty(pts.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4'),
                                                   ('red','u1'),('green','u1'),('blue','u1')])
            vertex['x']=pts[:,0].astype('f4'); vertex['y']=pts[:,1].astype('f4'); vertex['z']=pts[:,2].astype('f4')
            vertex['red']=rgb[:,0].astype('u1'); vertex['green']=rgb[:,1].astype('u1'); vertex['blue']=rgb[:,2].astype('u1')
            el = PlyElement.describe(vertex, 'vertex')
            PlyData([el], text=True).write(str(path))
            return
        except Exception as e:
            print(f"[WARN] ASCII PLY write failed (missing plyfile?). Falling back to binary: {e}", file=sys.stderr)
    pc = trimesh.points.PointCloud(pts, colors=rgb)
    pc.export(path)

def world_mesh_pieces(body_id: int, urdf_dir: Path,
                      urdf_mats: Optional[Dict[str, Dict[str, object]]] = None) -> List["trimesh.Trimesh"]:
    pieces: List["trimesh.Trimesh"] = []
    vdata = p.getVisualShapeData(body_id) or []
    link_world_T = {}
    bpos, born = p.getBasePositionAndOrientation(body_id)
    link_world_T[-1] = pose_to_matrix(bpos, born)
    for li in range(p.getNumJoints(body_id)):
        pos, orn = get_link_world_pose(body_id, li)
        link_world_T[li] = pose_to_matrix(pos, orn)
    for shape in vdata:
        link_idx = shape[1]
        m = load_visual_mesh(shape, urdf_dir, urdf_mats)  # keep 3-arg call; the 3rd is ignored
        if m is None or m.is_empty: continue
        T_world = link_world_T.get(link_idx, np.eye(4))
        m.apply_transform(T_world)
        pieces.append(m)
    return pieces

def process_one_combo(body_id: int, urdf_dir: Path, urdf_mats: Dict[str, Dict[str, object]],
                      joint_indices: List[int], joint_targets: List[float],
                      pose_dir: Path, cfg: Config) -> bool:
    settle_multi_joints(body_id, joint_indices, joint_targets, cfg)
    if p.getNumJoints(body_id) > 0: p.getLinkState(body_id, 0, computeForwardKinematics=1)

    pieces = world_mesh_pieces(body_id, urdf_dir, urdf_mats)
    if not pieces: return False
    pose_dir.mkdir(parents=True, exist_ok=True)

    # GLB export: bake textures to vertex colors; also bake MTL per-face Kd if possible
    if cfg.glb_bake:
        baked = []
        for m in pieces:
            vis = getattr(m, 'visual', None)
            # bake texture if present
            if vis is not None and getattr(vis, 'kind', '') == 'texture' and getattr(vis, 'uv', None) is not None and getattr(vis.material, 'image', None) is not None:
                try: m.visual = vis.to_color()
                except Exception: pass
            # if no colors yet, bake per-face Kd from materials
            try:
                has_fc = hasattr(m.visual, "face_colors") and m.visual.face_colors is not None and len(m.visual.face_colors)==len(m.faces)
                if not has_fc:
                    fc = _face_rgba_from_materials(m)
                    if fc is not None: m.visual.face_colors = fc
            except Exception: pass
            baked.append(m)
        scene = trimesh.Scene(baked)
    else:
        scene = trimesh.Scene(pieces)
    scene.export(pose_dir / "mesh.glb")

    # colored point cloud
    pts, rgb = sample_model_points_colored(
        pieces, cfg.points_per_pose, method=cfg.point_sampling, oversample=cfg.fps_oversample
    )
    if pts.shape[0] == 0: return False
    save_pointcloud_ply(pose_dir / "pointcloud.ply", pts, rgb, cfg.ply_ascii)

    with (pose_dir / "angles.json").open("w", encoding="utf-8") as f:
        json.dump({"angles": [float(x) for x in joint_targets]}, f, ensure_ascii=False, indent=2)
    return True

def process_one_model(row: Dict[str,str], args_ns, cfg: Config, out_root: Path) -> Tuple[str,int,int]:
    urdf_rel = row.get("urdf_relpath") or ""
    if not urdf_rel: return row.get('model_id','?'), 0, 0
    urdf_path = (args_ns.dataset_dir / urdf_rel).resolve()
    if not urdf_path.exists():
        print(f"[WARN] URDF doesn't exist: {urdf_path}", file=sys.stderr); return row.get('model_id','?'), 0, 1

    anno_id = choose_anno_id(row)
    category = row_get_category(row)
    # out_dir = (out_root / category / anno_id) if (args_ns.group_by_cat and category) else (out_root / anno_id)
    cat_name = category if category else "Unknown"
    out_dir = (out_root / cat_name / anno_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if p.isConnected(): p.resetSimulation()
    else: p.connect(p.DIRECT)
    p.setTimeStep(cfg.physics_timestep); p.setGravity(0,0,0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(str(args_ns.dataset_dir))
    urdf_dir = urdf_path.parent
    p.setAdditionalSearchPath(str(urdf_dir))

    urdf_mats = parse_urdf_materials(urdf_path)  # parsed but not mandatory for color sampling

    flags = p.URDF_USE_INERTIA_FROM_FILE
    try:
        bid = p.loadURDF(str(urdf_path), useFixedBase=1, flags=flags)
    except Exception as e:
        print(f"[{anno_id}] Failed to load: {e}", file=sys.stderr); return anno_id, 0, 1

    joint_types = [t.strip().lower() for t in args_ns.joint_types.split(",") if t.strip()]
    joints = find_joints(bid, joint_types, args_ns.joint_regex)
    if not joints:
        print(f"[{anno_id}] No matching joints found(type={joint_types}, regex={args_ns.joint_regex}); skipping.", file=sys.stderr)
        p.removeBody(bid); return anno_id, 0, 0

    # init-only
    if getattr(args_ns, "init_only", False):
        joint_indices = [int(j["index"]) for j in joints]
        cur = [float(p.getJointState(bid, j)[0]) for j in joint_indices]
        meta = {
            "anno_id": anno_id, "category": category, "urdf": str(urdf_path),
            "joints": [{"index": int(j["index"]), "name": str(j["name"]), "type": str(j["type"]),
                        "limit_lower": float(j["lower"]), "limit_upper": float(j["upper"]), "steps": 1} for j in joints],
            "num_combos": 1, "points_per_pose": cfg.points_per_pose,
            "sampling": {"joint_types": joint_types, "joint_regex": args_ns.joint_regex,
                         "global_steps": 1, "steps_override": {},
                         "unique_combos": True, "seed": args_ns.seed,
                         "ply_ascii": cfg.ply_ascii, "glb_bake": cfg.glb_bake,
                         "point_sampling": cfg.point_sampling, "fps_oversample": cfg.fps_oversample, "init_only": True}
        }
        with (out_dir / "joint.json").open("w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)
        ok = 0
        if process_one_combo(bid, urdf_dir, urdf_mats, joint_indices, cur, out_dir / "pose_000", cfg): ok += 1
        else: print(f"[{anno_id}] The initial pose has not generated a mesh.", file=sys.stderr)
        p.removeBody(bid)
        return anno_id, ok, (1 - ok)

    # normal multi-combo export
    steps_override = {}  # Unused: global 'steps' applies to all joints
    grids = build_per_joint_grids(joints, args_ns.steps, steps_override)
    combos = random_combinations(grids, args_ns.num_combos, seed=args_ns.seed, unique=(not args_ns.allow_duplicate_combos))

    meta = {
        "anno_id": anno_id, "category": category, "urdf": str(urdf_path),
        "joints": [
            {"index": int(j["index"]), "name": str(j["name"]), "type": str(j["type"]),
             "limit_lower": float(j["lower"]), "limit_upper": float(j["upper"]),
             "steps": int(steps_override.get(str(j["name"]), args_ns.steps))}
            for j in joints
        ],
        "num_combos": len(combos),
        "points_per_pose": cfg.points_per_pose,
        "sampling": {
            "joint_types": joint_types, "joint_regex": args_ns.joint_regex,
            "global_steps": args_ns.steps, "steps_override": steps_override,
            "unique_combos": (not args_ns.allow_duplicate_combos), "seed": args_ns.seed,
            "ply_ascii": cfg.ply_ascii, "glb_bake": cfg.glb_bake,
            "point_sampling": cfg.point_sampling, "fps_oversample": cfg.fps_oversample
        }
    }
    with (out_dir / "joint.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    ok=0
    joint_indices = [int(j["index"]) for j in joints]
    for i, angles in enumerate(combos):
        pose_dir = out_dir / f"pose_{i:03d}"
        if process_one_combo(bid, urdf_dir, urdf_mats, joint_indices, angles, pose_dir, cfg): ok += 1
        else: print(f"[{anno_id}] Combo {i} did not produce a mesh", file=sys.stderr)

    p.removeBody(bid)
    return anno_id, ok, len(combos)-ok

# ---------------- Worker (TOP-LEVEL for Windows spawn) ----------------
def worker_entry(row: Dict[str,str], args_payload: Dict[str,object]) -> Tuple[str,int,int]:
    args_ns = SimpleNamespace(
        dataset_dir=Path(args_payload["dataset_dir"]),
        joint_types=args_payload["joint_types"],
        joint_regex=args_payload["joint_regex"],   # fixed r"joint_\\d+"
        steps=int(args_payload["steps"]),
        num_combos=int(args_payload["num_combos"]),
        group_by_cat=bool(args_payload["group_by_cat"]),
        allow_duplicate_combos=bool(args_payload["allow_duplicate_combos"]),
        seed=int(args_payload["seed"]),
        init_only=bool(args_payload["init_only"])
    )
    cfg = Config(points_per_pose=int(args_payload["points"]),
                 ply_ascii=bool(args_payload["ply_ascii"]),
                 glb_bake=bool(args_payload["glb_bake"]),
                 point_sampling=str(args_payload["point_sampling"]),
                 fps_oversample=int(args_payload["fps_oversample"]))
    out_root = Path(args_payload["out_root"])

    # subprocess-local pybullet context
    if not p.isConnected(): p.connect(p.DIRECT)
    p.setTimeStep(cfg.physics_timestep); p.setGravity(0,0,0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setAdditionalSearchPath(str(args_ns.dataset_dir))
    try:
        return process_one_model(row, args_ns, cfg, out_root)
    finally:
        if p.isConnected():
            try: p.disconnect()
            except Exception: pass

# ---------------- Public API ----------------
def collect_from_index(
    index_csv: Path,
    dataset_dir: Path,
    out_dir: Path,
    categories: Optional[List[str]] = None,
    *,
    joint_types: str = "revolute",
    steps: int = 10,
    num_combos: int = 200,
    points: int = 4096,
    seed: int = 0,
    group_by_cat: bool = False,
    allow_duplicate_combos: bool = False,
    ply_ascii: bool = False,
    glb_bake: bool = False,
    workers: int = 1,
    init_only: bool = False,
    point_sampling: str = "random",
    fps_oversample: int = 8,
) -> None:
    """
    Build GLB (texture-aware) and colored PLY for models listed in CSV.
    This function strictly follows the original sampling & coloring logic, while:
      - Joint name regex is fixed to r"joint_\\d+"
      - Per-joint steps (steps-per) is removed; global `steps` applies to all joints.
    """
    rows = [r for r in _read_index_csv(index_csv) if include_row_by_cats(r, categories)]
    if not rows:
        print("[Collect] CSV filter resulted in empty set.", file=sys.stderr); return

    out_root = Path(out_dir).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    joint_regex = r"joint_\d+"  # Default joint name pattern

    if int(workers) <= 1:
        # single process (debug-friendly)
        if not p.isConnected(): p.connect(p.DIRECT)
        cfg = Config(points_per_pose=points, ply_ascii=ply_ascii, glb_bake=glb_bake,
                     point_sampling=point_sampling, fps_oversample=fps_oversample)
        p.setTimeStep(cfg.physics_timestep); p.setGravity(0,0,0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setAdditionalSearchPath(str(Path(dataset_dir).resolve()))
        ok=0; fail=0
        ns = SimpleNamespace(dataset_dir=Path(dataset_dir).resolve(), joint_types=joint_types, joint_regex=joint_regex,
                             steps=int(steps), num_combos=int(num_combos), group_by_cat=bool(group_by_cat),
                             allow_duplicate_combos=bool(allow_duplicate_combos), seed=int(seed), init_only=bool(init_only))
        for row in rows:
            _, ok_i, fail_i = process_one_model(row, ns, cfg, out_root)
            ok += (1 if ok_i>0 else 0); fail += fail_i
        if p.isConnected(): p.disconnect()
        print(f"[Collect] Finish. Success {ok} (Generate at least one pose), Fail {fail}. Output {out_root}")
        return

    # multiprocess: Windows-safe via top-level worker_entry
    from concurrent.futures import ProcessPoolExecutor, as_completed
    args_payload = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "out_root": str(out_root),
        "joint_types": joint_types,
        "joint_regex": joint_regex,
        "steps": int(steps),
        "num_combos": int(num_combos),
        "group_by_cat": bool(group_by_cat),
        "allow_duplicate_combos": bool(allow_duplicate_combos),
        "seed": int(seed),
        "ply_ascii": bool(ply_ascii),
        "glb_bake": bool(glb_bake),
        "points": int(points),
        "init_only": bool(init_only),
        "point_sampling": str(point_sampling),
        "fps_oversample": int(fps_oversample),
    }
    ok=0; fail=0
    with ProcessPoolExecutor(max_workers=int(max(1,workers))) as ex:
        futs = [ex.submit(worker_entry, r, args_payload) for r in rows]
        for fut in as_completed(futs):
            try:
                anno_id, ok_i, fail_i = fut.result()
                ok += (1 if ok_i>0 else 0); fail += fail_i
            except Exception as e:
                print(f"[worker] Fail: {e}", file=sys.stderr); fail += 1

    print(f"[Collect] Finish. Success {ok} (Generate at least one pose), Fail {fail}. Output {out_root}")
