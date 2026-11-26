"""Stage B (v2): Large-Scale Single-Cell CITE-seq Ground-Truth Benchmark & 90k Scalability Curve on GPU (NCBI GEO GSE194122)."""
from __future__ import annotations

import gc
import gzip
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _cand in [str(_THIS_DIR), "/content", str(Path.cwd() / "src")]:
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from baselines import (  # noqa: E402
    corrupt_view,
    evaluate_clustering,
    run_attention_moae,
    run_lmvsc,
    run_mcgc,
    run_snf,
    run_subtype_gan,
    run_wnn,
)
from cl_vssc import CLVSSC, CLVSSCAnchor, get_device  # noqa: E402


BMMC_GZ_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def download_and_extract_bmmc(sc_dir: Path) -> Path:
    h5_path = sc_dir / "bmmc_cite.h5ad"
    if h5_path.exists() and h5_path.stat().st_size > 10_000_000 and h5py.is_hdf5(h5_path):
        return h5_path

    # Remove any invalid remnants (such as previous 0-byte downloads)
    for p in sc_dir.glob("*.h5ad"):
        if p.stat().st_size < 1_000_000 or not h5py.is_hdf5(p):
            p.unlink(missing_ok=True)

    gz_path = sc_dir / "bmmc_cite.h5ad.gz"
    sc_dir.mkdir(parents=True, exist_ok=True)
    if not (gz_path.exists() and gz_path.stat().st_size > 100_000_000):
        subprocess.run(
            ["curl", "-k", "-fL", "--retry", "6", "--retry-delay", "3", "-C", "-", "-o", str(gz_path), BMMC_GZ_URL],
            check=True,
        )
    print(f"  Downloaded BMMC gzip: {gz_path.stat().st_size / 1e6:.1f} MB; decompressing...", flush=True)
    tmp_path = sc_dir / "bmmc_cite.h5ad.tmp"
    with gzip.open(gz_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=8 * 1024 * 1024)
    tmp_path.replace(h5_path)
    if not h5py.is_hdf5(h5_path):
        h5_path.unlink(missing_ok=True)
        raise RuntimeError(f"Decompressed file {h5_path} is not a valid HDF5/h5ad archive.")
    gz_path.unlink(missing_ok=True)
    print(f"  Verified HDF5 archive: {h5_path.stat().st_size / 1e6:.1f} MB", flush=True)
    return h5_path


def extract_citeseq_views(
    adata: ad.AnnData,
    label_candidates: List[str] = ["cell_type", "celltype.l1", "cell_types", "celltype"],
    max_cells: Optional[int] = None,
    n_hvg: int = 2000,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, object]]:
    """Extract normalized scRNA-seq HVG and surface-protein ADT matrices aligned with ground-truth labels."""
    label_col = next((c for c in label_candidates if c in adata.obs.columns), None)
    if label_col is None:
        raise KeyError(f"No label column found in {list(adata.obs.columns)}")
    raw_labels = adata.obs[label_col].astype(str)
    valid = (~raw_labels.isin(["nan", "None", "NA", ""])) & (~adata.obs[label_col].isna())
    vc = raw_labels[valid].value_counts()
    keep_classes = set(vc[vc >= 15].index)
    keep_idx = np.flatnonzero(valid.to_numpy() & raw_labels.isin(keep_classes).to_numpy())

    rng = np.random.default_rng(seed)
    if max_cells is not None and len(keep_idx) > max_cells:
        keep_idx = np.sort(rng.choice(keep_idx, size=max_cells, replace=False))

    y_str = raw_labels.iloc[keep_idx].to_numpy()
    classes, y_int = np.unique(y_str, return_inverse=True)

    # Identify RNA vs ADT columns from adata.var['feature_types'] or adata.obsm
    rna_mask = None
    adt_mask = None
    for vcol in ["feature_types", "modality", "feature_type"]:
        if vcol in adata.var.columns:
            vals = adata.var[vcol].astype(str)
            adt_m = vals.str.contains("Antibody|ADT|Protein", case=False, regex=True).to_numpy()
            rna_m = vals.str.contains("Gene|GEX|RNA", case=False, regex=True).to_numpy()
            if adt_m.sum() >= 5 and rna_m.sum() >= 50:
                rna_mask = rna_m
                adt_mask = adt_m
                break

    has_counts_layer = "counts" in adata.layers
    X_source = adata.layers["counts"] if has_counts_layer else adata.X

    if rna_mask is not None and adt_mask is not None:
        X_rna_sub = X_source[keep_idx][:, rna_mask]
        X_adt_sub = X_source[keep_idx][:, adt_mask]
    else:
        prot_key = next((k for k in ["protein_expression", "protein_counts", "ADT", "protein"] if k in adata.obsm), None)
        if prot_key is None:
            raise KeyError(f"Could not locate ADT modality in var {list(adata.var.columns)} or obsm {list(adata.obsm.keys())}")
        X_rna_sub = X_source[keep_idx]
        adt_obj = adata.obsm[prot_key]
        X_adt_sub = (adt_obj.to_numpy() if isinstance(adt_obj, pd.DataFrame) else adt_obj)[keep_idx]

    # Process RNA (sparse-safe variance calculation)
    if sp.issparse(X_rna_sub):
        X_rna = X_rna_sub.tocsr().astype(np.float32)
        is_raw_counts = has_counts_layer or (float(X_rna.data.max() if X_rna.nnz > 0 else 0.0) > 35.0 and float(X_rna.data.min() if X_rna.nnz > 0 else 0.0) >= 0.0)
        if is_raw_counts:
            lib = np.asarray(X_rna.sum(axis=1)).ravel()
            inv_lib = 1e4 / np.maximum(lib, 1.0)
            X_rna = X_rna.multiply(inv_lib[:, None]).tocsr()
            X_rna.data = np.log1p(np.maximum(X_rna.data, 0.0))
        mean_g = np.asarray(X_rna.mean(axis=0)).ravel()
        mean_g2 = np.asarray(X_rna.power(2).mean(axis=0)).ravel()
        var_g = mean_g2 - mean_g ** 2
        top_g = np.argsort(var_g)[-min(n_hvg, X_rna.shape[1]) :]
        top_g.sort()
        X_rna_dense = X_rna[:, top_g].toarray().astype(np.float32)
    else:
        arr = np.asarray(X_rna_sub, dtype=np.float32)
        is_raw_counts = has_counts_layer or (float(arr.max()) > 35.0 and float(arr.min()) >= 0.0)
        if is_raw_counts:
            lib = arr.sum(axis=1, keepdims=True)
            arr = np.log1p(np.maximum(arr * (1e4 / np.maximum(lib, 1.0)), 0.0))
        var_g = arr.var(axis=0)
        top_g = np.argsort(var_g)[-min(n_hvg, arr.shape[1]) :]
        top_g.sort()
        X_rna_dense = arr[:, top_g].astype(np.float32)

    # Process ADT surface proteins (CLR normalization if raw counts, else row-centered)
    adt_arr = X_adt_sub.toarray().astype(np.float32) if sp.issparse(X_adt_sub) else np.asarray(X_adt_sub, dtype=np.float32)
    if has_counts_layer or (float(adt_arr.min()) >= -1e-6 and float(adt_arr.max()) > 25.0):
        adt_log = np.log1p(np.maximum(adt_arr, 0.0))
        X_adt = (adt_log - adt_log.mean(axis=1, keepdims=True)).astype(np.float32)
    else:
        X_adt = adt_arr

    meta = {
        "n_cells": int(len(keep_idx)),
        "n_classes": int(len(classes)),
        "d_rna": int(X_rna_dense.shape[1]),
        "d_adt": int(X_adt.shape[1]),
        "label_col": label_col,
    }
    return [X_rna_dense, X_adt], y_int, meta


def main() -> None:
    t0_all = time.time()
    work_dir = Path("/content") if Path("/content").exists() else Path.cwd()
    sc_dir = work_dir / "sc_data"
    out_dir = work_dir / "results_v2"
    sc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    print(f"[Stage B] Device: {dev} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    b1_csv = out_dir / "singlecell_benchmark_v2.csv"
    b2_csv = out_dir / "sc_fault_v2.csv"
    b3_csv = out_dir / "scalability_v2.csv"
    meta_json = out_dir / "stage_b_meta.json"
    if b1_csv.exists() and b2_csv.exists() and b3_csv.exists() and meta_json.exists():
        print("\n=== Stage B: All checkpoints exist, skipping ===", flush=True)
        return

    bmmc_path = download_and_extract_bmmc(sc_dir)
    print("  Loading BMMC CITE-seq h5ad...", flush=True)
    ad_bmmc = ad.read_h5ad(bmmc_path)
    print(
        f"  BMMC shape={ad_bmmc.shape} | layers={list(ad_bmmc.layers.keys())} | "
        f"obs={list(ad_bmmc.obs.columns)} | var={list(ad_bmmc.var.columns)}",
        flush=True,
    )
    if "feature_types" in ad_bmmc.var.columns:
        print(f"  feature_types: {ad_bmmc.var['feature_types'].value_counts().to_dict()}", flush=True)

    site_series = ad_bmmc.obs["site"].astype(str) if "site" in ad_bmmc.obs.columns else ad_bmmc.obs["batch"].astype(str).str[:2]
    site_counts = site_series.value_counts()
    s1_name, s2_name = str(site_counts.index[0]), str(site_counts.index[1])
    print(f"  Top sequencing sites: {s1_name} ({site_counts[s1_name]}), {s2_name} ({site_counts[s2_name]})", flush=True)

    datasets = {}
    ad_s1 = ad_bmmc[site_series == s1_name]
    datasets["BMMC-Site1"] = extract_citeseq_views(ad_s1, max_cells=None, seed=42)
    del ad_s1
    ad_s2 = ad_bmmc[site_series == s2_name]
    datasets["BMMC-Site2"] = extract_citeseq_views(ad_s2, max_cells=None, seed=42)
    del ad_s2
    datasets["BMMC-10k"] = extract_citeseq_views(ad_bmmc, max_cells=10000, seed=42)
    datasets["BMMC-30k"] = extract_citeseq_views(ad_bmmc, max_cells=30000, seed=42)
    gc.collect()

    stage_b_meta = {k: v[2] for k, v in datasets.items()}
    stage_b_meta["BMMC-Full"] = {"n_cells": int(ad_bmmc.n_obs), "n_features": int(ad_bmmc.n_vars)}
    meta_json.write_text(json.dumps(stage_b_meta, indent=2))
    print("  Extracted datasets:", json.dumps(stage_b_meta), flush=True)

    # Experiment B.1: Real Single-Cell CITE-seq Ground-Truth Benchmark (4 datasets x 3 seeds)
    if b1_csv.exists():
        print("\n=== Stage B.1: Skipping (checkpoint exists) ===", flush=True)
    else:
        print("\n=== Stage B.1: Single-Cell Ground-Truth Benchmark ===", flush=True)
        sc_rows = []
        for ds_name, (views, y_true, meta) in datasets.items():
            k_c = meta["n_classes"]
            n_c = meta["n_cells"]
            print(f"  Dataset {ds_name}: n={n_c}, K={k_c}, dims={[meta['d_rna'], meta['d_adt']]}", flush=True)
            for seed in [42, 43, 44]:
                runners = {
                    "CL-VSSC-Anchor": lambda v, k, s: CLVSSCAnchor(k, n_anchors=384, anchor_k=20, dead_zone=0.0, max_iter=30, seed=s).fit(v),
                    "CL-VSSC-Anchor (DZ-PI)": lambda v, k, s: CLVSSCAnchor(k, n_anchors=384, anchor_k=20, dead_zone=0.14, max_iter=30, seed=s).fit(v),
                    "CL-VSSC-Anchor (w/o FB)": lambda v, k, s: CLVSSCAnchor(k, n_anchors=384, anchor_k=20, use_feedback=False, max_iter=30, seed=s).fit(v),
                    "LMVSC": lambda v, k, s: run_lmvsc(v, k, n_anchors=384, anchor_k=20, seed=s),
                    "Seurat-WNN": lambda v, k, s: run_wnn(v, k, n_anchors=384, seed=s),
                    "Subtype-GAN": lambda v, k, s: run_subtype_gan(v, k, epochs=50, seed=s),
                    "Attention-MOAE": lambda v, k, s: run_attention_moae(v, k, epochs=50, seed=s),
                }
                for m_name, fn in runners.items():
                    sync_cuda()
                    t0 = time.time()
                    out = fn(views, k_c, seed)
                    sync_cuda()
                    dt = time.time() - t0
                    met = evaluate_clustering(out["labels"], true_labels=y_true, embedding=out.get("F"))
                    sc_rows.append(
                        {
                            "dataset": ds_name,
                            "n_cells": n_c,
                            "n_classes": k_c,
                            "seed": seed,
                            "method": m_name,
                            "ari": met["ari"],
                            "nmi": met["nmi"],
                            "acc": met["acc"],
                            "silhouette": met.get("silhouette", np.nan),
                            "runtime_sec": round(dt, 3),
                            "w_rna": float(out["final_weights"][0]) if "final_weights" in out else np.nan,
                            "w_adt": float(out["final_weights"][1]) if "final_weights" in out else np.nan,
                        }
                    )
            sub_ds = pd.DataFrame([r for r in sc_rows if r["dataset"] == ds_name])
            print(sub_ds.groupby("method")[["ari", "nmi", "acc", "runtime_sec"]].mean().round(3).to_string(), flush=True)
        pd.DataFrame(sc_rows).to_csv(b1_csv, index=False)

    # Experiment B.2: Single-Cell Ground-Truth Modality-Corruption Test on BMMC-10k
    if b2_csv.exists():
        print("\n=== Stage B.2: Skipping (checkpoint exists) ===", flush=True)
    else:
        print("\n=== Stage B.2: BMMC-10k Ground-Truth Modality Corruption ===", flush=True)
        sc_fault_rows = []
        v_b10, y_b10, m_b10 = datasets["BMMC-10k"]
        k_b10 = m_b10["n_classes"]
        sc_fault_runners = {
            "CL-VSSC-Anchor": lambda v: CLVSSCAnchor(k_b10, n_anchors=384, anchor_k=20, dead_zone=0.0, seed=42).fit(v),
            "CL-VSSC-Anchor (DZ-PI)": lambda v: CLVSSCAnchor(k_b10, n_anchors=384, anchor_k=20, dead_zone=0.14, seed=42).fit(v),
            "CL-VSSC-Anchor (w/o FB)": lambda v: CLVSSCAnchor(k_b10, n_anchors=384, anchor_k=20, use_feedback=False, seed=42).fit(v),
            "LMVSC": lambda v: run_lmvsc(v, k_b10, n_anchors=384, anchor_k=20, seed=42),
            "Seurat-WNN": lambda v: run_wnn(v, k_b10, n_anchors=384, seed=42),
            "Attention-MOAE": lambda v: run_attention_moae(v, k_b10, epochs=50, seed=42),
            "Subtype-GAN": lambda v: run_subtype_gan(v, k_b10, epochs=50, seed=42),
        }
        for v_idx, mod_lbl in [(0, "scRNA"), (1, "Protein_ADT")]:
            for sigma, drop in [(0.0, 0.0), (1.5, 0.30), (3.0, 0.50)]:
                seeds = [42] if sigma == 0.0 else [0, 1, 2]
                for s_seed in seeds:
                    v_c = v_b10 if sigma == 0.0 else corrupt_view(v_b10, view_idx=v_idx, noise_std=sigma, dropout_rate=drop, seed=200 * v_idx + int(sigma * 10) + s_seed)
                    for m_name, fn in sc_fault_runners.items():
                        out = fn(v_c)
                        met = evaluate_clustering(out["labels"], true_labels=y_b10)
                        sc_fault_rows.append(
                            {
                                "corrupted_modality": mod_lbl,
                                "noise_sigma": sigma,
                                "dropout": drop,
                                "seed": s_seed,
                                "method": m_name,
                                "gt_ari": met["ari"],
                                "gt_nmi": met["nmi"],
                                "gt_acc": met["acc"],
                                "w_corrupted": float(out["final_weights"][v_idx]) if "final_weights" in out else np.nan,
                            }
                        )
        pd.DataFrame(sc_fault_rows).to_csv(b2_csv, index=False)
        print(pd.DataFrame(sc_fault_rows).groupby(["corrupted_modality", "noise_sigma", "method"])[["gt_ari", "gt_acc", "w_corrupted"]].mean().round(3).to_string(), flush=True)

    # Experiment B.3: Large-Scale GPU Runtime & Peak VRAM Scalability Curve (1k -> Full ~90k)
    if b3_csv.exists():
        print("\n=== Stage B.3: Skipping (checkpoint exists) ===", flush=True)
    else:
        print("\n=== Stage B.3: GPU Scalability Benchmark (n = 1k -> Full 90k Atlas) ===", flush=True)
        del datasets
        gc.collect()
        scale_rows = []
        for n_sub in [1000, 2500, 5000, 10000, 25000, 50000, None]:
            v_s, y_s, m_s = extract_citeseq_views(ad_bmmc, max_cells=n_sub, seed=101)
            n_actual = m_s["n_cells"]
            k_s = m_s["n_classes"]
            scale_candidates = {
                "CL-VSSC-Anchor": (lambda: CLVSSCAnchor(k_s, n_anchors=384, anchor_k=20, dead_zone=0.0, seed=42).fit(v_s), True),
                "LMVSC": (lambda: run_lmvsc(v_s, k_s, n_anchors=384, anchor_k=20, seed=42), True),
                "Subtype-GAN": (lambda: run_subtype_gan(v_s, k_s, epochs=50, seed=42), True),
                "CL-VSSC (Dense)": (lambda: CLVSSC(k_s, dead_zone=0.0, seed=42).fit(v_s), n_actual <= 10000),
                "SNF": (lambda: run_snf(v_s, k_s, seed=42), n_actual <= 10000),
                "MCGC": (lambda: run_mcgc(v_s, k_s, seed=42), n_actual <= 10000),
            }
            for m_name, (fn, allowed) in scale_candidates.items():
                if not allowed:
                    continue
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    sync_cuda()
                    t0 = time.time()
                    out = fn()
                    sync_cuda()
                    dt = time.time() - t0
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                    met = evaluate_clustering(out["labels"], true_labels=y_s)
                    scale_rows.append(
                        {
                            "n_cells": n_actual,
                            "method": m_name,
                            "runtime_sec": round(dt, 3),
                            "peak_vram_mb": round(peak_mb, 1),
                            "gt_ari": met["ari"],
                            "gt_acc": met["acc"],
                        }
                    )
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    scale_rows.append({"n_cells": n_actual, "method": m_name, "runtime_sec": np.nan, "peak_vram_mb": np.nan, "gt_ari": np.nan, "gt_acc": np.nan})
            print(f"  [scale n={n_actual:5d}] done", flush=True)
        pd.DataFrame(scale_rows).to_csv(b3_csv, index=False)
    print(f"STAGE_B_COMPLETE in {time.time() - t0_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
