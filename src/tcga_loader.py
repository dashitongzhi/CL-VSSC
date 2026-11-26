"""Verified downloader and preprocessor for the Rappoport & Shamir 10-TCGA multi-omics benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd


BASE_URL = "https://acgt.cs.tau.ac.il/multi_omic_benchmark/data"

COHORTS = ["aml", "bic", "coad", "gbm", "kirc", "lihc", "lusc", "ov", "sarc", "skcm"]

COHORT_ARCHIVES: Dict[str, str] = {
    "aml": "aml.zip",
    "bic": "breast.zip",
    "coad": "colon.zip",
    "gbm": "gbm.zip",
    "kirc": "kidney.zip",
    "lihc": "liver.zip",
    "lusc": "lung.zip",
    "ov": "ovarian.zip",
    "sarc": "sarcoma.zip",
    "skcm": "melanoma.zip",
}

CANONICAL_K: Dict[str, int] = {
    "aml": 4,
    "bic": 5,
    "coad": 4,
    "gbm": 4,
    "kirc": 4,
    "lihc": 4,
    "lusc": 4,
    "ov": 4,
    "sarc": 4,
    "skcm": 4,
}

_SAMPLE_TYPE_PRIORITY = {"01": 0, "03": 1, "06": 2, "02": 3, "05": 4, "11": 9}


def _norm_id(raw: object) -> str:
    s = str(raw).strip().strip('"').strip("'").upper().replace(".", "-")
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[0-9]{2}[A-Z]?)?)", s)
    return m.group(1) if m else s


def _sample_type_rank(barcode: str) -> Tuple[int, str]:
    parts = barcode.split("-")
    stype = parts[3][:2] if len(parts) >= 4 else "99"
    return _SAMPLE_TYPE_PRIORITY.get(stype, 5), barcode


def _find_extracted_file(c_dir: Path, name: str) -> Path | None:
    direct = c_dir / name
    if direct.is_file() and direct.stat().st_size > 100:
        return direct
    for p in c_dir.rglob("*"):
        if p.is_file() and p.stem.lower() == name.lower() and p.stat().st_size > 100:
            return p
    return None


def download_and_extract_cohort(cohort: str, data_dir: Path) -> Path:
    """Download cohort ZIP via resilient curl resume and extract exp/methy/mirna/survival."""
    c_dir = data_dir / cohort
    c_dir.mkdir(parents=True, exist_ok=True)
    mods = ["exp", "methy", "mirna", "survival"]
    if all(_find_extracted_file(c_dir, m) is not None for m in mods):
        return c_dir

    zip_name = COHORT_ARCHIVES[cohort]
    url = f"{BASE_URL}/{zip_name}"
    zip_path = data_dir / zip_name

    need_download = True
    if zip_path.exists() and zip_path.stat().st_size > 1024:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if zf.testzip() is None:
                    need_download = False
        except Exception:
            need_download = True

    if need_download:
        try:
            subprocess.run(
                [
                    "curl",
                    "-k",
                    "-fL",
                    "--retry",
                    "6",
                    "--retry-delay",
                    "2",
                    "--retry-all-errors",
                    "-C",
                    "-",
                    "-o",
                    str(zip_path),
                    url,
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            if zip_path.exists():
                zip_path.unlink()
            subprocess.run(
                [
                    "curl",
                    "-k",
                    "-fL",
                    "--retry",
                    "6",
                    "--retry-delay",
                    "2",
                    "--retry-all-errors",
                    "-o",
                    str(zip_path),
                    url,
                ],
                check=True,
            )

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(c_dir)

    for m in mods:
        found = _find_extracted_file(c_dir, m)
        target = c_dir / m
        if found is not None and found != target:
            shutil.copy2(found, target)
    return c_dir


def _read_r_omic_matrix(path: Path) -> pd.DataFrame:
    """Parse R write.table matrix (samples as columns or rows) and standardize rows=samples."""
    df = pd.read_csv(path, sep=r"\s+")
    idx_hits = sum("TCGA" in str(i).upper() for i in df.index[:25])
    col_hits = sum("TCGA" in str(c).upper() for c in df.columns[:25])
    if col_hits > idx_hits:
        df = df.T
    df.index = [_norm_id(i) for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _dedup_by_priority(df: pd.DataFrame, key_len: int) -> pd.DataFrame:
    order = sorted(range(len(df)), key=lambda idx: _sample_type_rank(str(df.index[idx])))
    df_sorted = df.iloc[order].copy()
    df_sorted.index = [str(i)[:key_len] for i in df_sorted.index]
    return df_sorted[~pd.Index(df_sorted.index).duplicated(keep="first")]


def _read_survival_table(path: Path) -> pd.DataFrame:
    surv = pd.read_csv(path, sep=r"\s+")
    cols_lower = {str(c).strip().strip('"').lower(): c for c in surv.columns}
    id_col = next(
        (cols_lower[k] for k in ["patientid", "sampleid", "sample", "submitter_id"] if k in cols_lower),
        surv.columns[0],
    )
    t_col = next(
        (cols_lower[k] for k in ["survival", "os.time", "time", "days"] if k in cols_lower),
        surv.columns[1],
    )
    e_col = next(
        (cols_lower[k] for k in ["death", "os", "event", "status"] if k in cols_lower),
        surv.columns[2],
    )
    out = pd.DataFrame(
        {
            "barcode": [_norm_id(x) for x in surv[id_col]],
            "time": pd.to_numeric(surv[t_col], errors="coerce"),
            "event": pd.to_numeric(surv[e_col], errors="coerce"),
        }
    ).dropna()
    out = out[(out["time"] > 0) & (out["event"].isin([0, 1]))].copy()
    out = out.set_index("barcode")
    return out


def load_tcga_cohort(
    cohort: str,
    data_dir: Path,
    top_features: int = 2000,
) -> Tuple[List[np.ndarray], pd.DataFrame, List[str], Dict[str, object]]:
    """Download, align, log2-transform sequencing counts, and variance-filter one TCGA cohort."""
    c_dir = download_and_extract_cohort(cohort, data_dir)
    omics_raw: Dict[str, pd.DataFrame] = {}
    raw_shapes: Dict[str, int] = {}

    for mod in ["exp", "methy", "mirna"]:
        df = _read_r_omic_matrix(c_dir / mod)
        raw_shapes[mod] = int(df.shape[1])
        omics_raw[mod] = df

    surv_raw = _read_survival_table(c_dir / "survival")

    # Try 15-char sample barcode intersection first, then fall back to 12-char patient barcode
    om15 = {m: _dedup_by_priority(df, 15) for m, df in omics_raw.items()}
    sv15 = _dedup_by_priority(surv_raw, 15)
    common15 = sorted(set(om15["exp"].index) & set(om15["methy"].index) & set(om15["mirna"].index) & set(sv15.index))

    match_mode = "15-char"
    if len(common15) >= 40:
        # Collapse any dual aliquots of the same 12-char patient to prevent patient pseudoreplication
        om_aligned = {m: _dedup_by_priority(om15[m].loc[common15], 12) for m in om15}
        sv_aligned = _dedup_by_priority(sv15.loc[common15], 12)
        common = sorted(
            set(om_aligned["exp"].index)
            & set(om_aligned["methy"].index)
            & set(om_aligned["mirna"].index)
            & set(sv_aligned.index)
        )
    else:
        match_mode = "12-char"
        om_aligned = {m: _dedup_by_priority(df, 12) for m, df in omics_raw.items()}
        sv_aligned = _dedup_by_priority(surv_raw, 12)
        common = sorted(
            set(om_aligned["exp"].index)
            & set(om_aligned["methy"].index)
            & set(om_aligned["mirna"].index)
            & set(sv_aligned.index)
        )

    views: List[np.ndarray] = []
    meta: Dict[str, object] = {
        "cohort": cohort.upper(),
        "canonical_k": CANONICAL_K[cohort],
        "n_samples": len(common),
        "match_mode": match_mode,
        "common15_raw": len(common15),
    }

    surv_final = sv_aligned.loc[common].reset_index(drop=True)
    meta["n_events"] = int(surv_final["event"].sum())
    meta["event_rate"] = float(surv_final["event"].mean()) if len(surv_final) > 0 else 0.0

    for mod in ["exp", "methy", "mirna"]:
        sub = om_aligned[mod].loc[common].copy()
        sub = sub.dropna(axis=1, how="all")
        med = sub.median(axis=0)
        sub = sub.fillna(med).fillna(0.0)

        vals = sub.to_numpy(dtype=np.float64)
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        log2_applied = False
        if mod in ("exp", "mirna") and vmin >= -1e-6 and vmax > 50.0:
            vals = np.log2(1.0 + np.clip(vals, 0.0, None))
            log2_applied = True

        var = vals.var(axis=0)
        keep_mask = var > 1e-10
        vals = vals[:, keep_mask]
        var = var[keep_mask]
        if vals.shape[1] > top_features:
            top_idx = np.argsort(var)[-top_features:]
            top_idx.sort()
            vals = vals[:, top_idx]

        views.append(vals.astype(np.float32))
        meta[f"raw_d_{mod}"] = raw_shapes[mod]
        meta[f"d_{mod}"] = int(vals.shape[1])
        meta[f"log2_{mod}"] = log2_applied

    return views, surv_final, common, meta
