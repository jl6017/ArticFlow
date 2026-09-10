from __future__ import annotations
import csv, json, sys, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable, Tuple, Set

__all__ = ["build_index"]

def _flatten_json(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested JSON objects to one-level dict with dot-separated keys."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        nk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_json(v, nk, sep=sep).items())
        elif isinstance(v, list):
            try: items.append((nk, json.dumps(v, ensure_ascii=False)))
            except Exception: items.append((nk, str(v)))
        else:
            items.append((nk, v))
    return dict(items)

def _find_meta_json(model_dir: Path) -> Optional[Path]:
    """Find meta.json under a model directory (recursively), preferring nearest to root."""
    cand: List[Path] = []
    for p in model_dir.rglob("*"):
        if p.is_file() and p.name.lower() == "meta.json":
            cand.append(p)
    if not cand: return None
    cand.sort(key=lambda p: len(p.relative_to(model_dir).parts))
    return cand[0]

def _find_first_urdf(model_dir: Path) -> Optional[Path]:
    for p in model_dir.rglob("*.urdf"):
        if p.is_file(): return p
    return None

def _scan_one_model(model_dir: Path, dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """Scan one model folder and return a record dict (minimal fields even if meta.json is missing)."""
    try:
        meta_path = _find_meta_json(model_dir)
        meta_flat: Dict[str, Any] = {}
        if meta_path and meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f: meta = json.load(f)
            if not isinstance(meta, dict):
                raise ValueError(f"meta.json is not an object: {meta_path}")
            meta_flat = _flatten_json(meta)
        urdf_path = _find_first_urdf(model_dir)
        rel_urdf = str(urdf_path.relative_to(dataset_dir)) if urdf_path else ""
        record: Dict[str, Any] = {
            "model_dir": str(model_dir.relative_to(dataset_dir)),
            "model_id": model_dir.name,
            "meta_json": str(meta_path.relative_to(dataset_dir)) if meta_path else "",
            "urdf_relpath": rel_urdf,
        }
        for k, v in meta_flat.items():
            if isinstance(v, (str, int, float)) or v is None:
                record[k] = v
            else:
                try: record[k] = json.dumps(v, ensure_ascii=False)
                except Exception: record[k] = str(v)
        return record
    except Exception as e:
        sys.stderr.write(f"[WARN] scan failed: {model_dir} -> {e}\n")
        traceback.print_exc(file=sys.stderr)
        return None

def _walk_model_dirs(dataset_dir: Path) -> Iterable[Path]:
    """Treat first-level subfolders as models (adjust here if your layout differs)."""
    for p in dataset_dir.iterdir():
        if p.is_dir(): yield p

def _write_csv(records: List[Dict[str, Any]], out_csv: Path) -> None:
    """Write a CSV with union of keys (base columns first, others alphabetical)."""
    base_cols = ["model_id", "model_dir", "meta_json", "urdf_relpath"]
    all_keys: Set[str] = set().union(*[set(r.keys()) for r in records]) if records else set()
    other_cols = [k for k in sorted(all_keys) if k not in base_cols]
    fieldnames = base_cols + other_cols
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records: w.writerow(r)

def build_index(dataset_dir: Path, out_csv: Path) -> int:
    """
    Build a CSV index for *all* models under dataset_dir (no category filtering).

    Args:
      dataset_dir: Root folder of PartNet-Mobility (first-level subfolders = models).
      out_csv:     Output CSV path.

    Returns:
      Number of indexed records written.
    """
    dataset_dir = Path(dataset_dir)
    records: List[Dict[str, Any]] = []
    total = 0
    for model_dir in _walk_model_dirs(dataset_dir):
        total += 1
        rec = _scan_one_model(model_dir, dataset_dir)
        if rec: records.append(rec)
    if not records:
        sys.stderr.write("No records found (meta.json missing or errors?).\n")
        return 0
    _write_csv(records, Path(out_csv))
    print(f"[Index] Wrote CSV: {out_csv}  (records={len(records)}, scanned={total})")
    return len(records)
