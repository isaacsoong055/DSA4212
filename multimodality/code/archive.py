"""Compressed, committable archive of the raw BO runs, with a manifest.

results/raw/ stays gitignored (uncompressed working files). `archive_bo(name)` writes one zip per finished
batch, results/archive/raw_bo_<name>.zip (deterministic: sorted, fixed timestamps) plus raw_bo_<name>_manifest.csv
(file, bytes, rows, sha256), both tracked. An existing archive is never rewritten. CMA-ES and random runs
reproduce exactly and are not archived.
"""
from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

from logging_utils import RAW_DIR, ROOT

ARCHIVE_DIR = ROOT / "results" / "archive"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def archive_bo(name: str, files: list[Path] | None = None, raw_dir: Path = RAW_DIR,
               out_dir: Path = ARCHIVE_DIR) -> tuple[Path, Path]:
    """Zip raw BO run CSVs (all under raw_dir, or the given files) as raw_bo_<name>.zip with a manifest.
    Refuses to overwrite an existing archive. Returns (zip path, manifest path)."""
    files = sorted(files) if files is not None else sorted(Path(raw_dir).glob("*/bo__*.csv"))
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath, mpath = out_dir / f"raw_bo_{name}.zip", out_dir / f"raw_bo_{name}_manifest.csv"
    if zpath.exists() or mpath.exists():
        raise FileExistsError(f"{zpath.name} already exists; archives are never rewritten (use a new batch name)")
    rows = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            data = f.read_bytes()
            info = zipfile.ZipInfo(str(f.relative_to(raw_dir)), ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data, compresslevel=9)
            rows.append(dict(file=info.filename, bytes=len(data), rows=data.count(b"\n") - 1,
                             sha256=hashlib.sha256(data).hexdigest()))
    with open(mpath, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "bytes", "rows", "sha256"])
        wr.writeheader()
        wr.writerows(rows)
    return zpath, mpath
