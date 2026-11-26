"""Incremental v3 GPU evaluation: Null-SNR PI (TCGA), Anchor-Local PI (GSE194122 CITE-seq), and Control-Parameter Sensitivity Grids."""
from __future__ import annotations

import gc
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import anndata as ad
import numpy as np
import pandas as pd
import torch

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _cand in [str(_THIS_DIR), "/content", str(Path.cwd() / "src")]:
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from baselines import corrupt_view, evaluate_clustering  # noqa: E402
from cl_vssc import CLVSSC, CLVSSCAnchor, get_device  # noqa: E402
from run_v2_singlecell_suite import download_and_extract_bmmc, extract_citeseq_views  # noqa: E402
from tcga_loader import CANONICAL_K, COHORTS, load_tcga_cohort  # noqa: E402


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def run_v3_tcga_null_snr(loaded: Dict[str, tuple], out_dir: Path) -> None:
    # V3.1a: Clean 10-TCGA benchmark
    clean_csv = out_dir / "tcga_null_snr_v3.csv"
    traj_json = out_dir / "traj_null_snr_v3.json"
    if not (clean_csv.exists() and traj_json.exists()):
        print("\n=== V3.1a: 10-TCGA Null-SNR PI Clean Benchmark ===", flush=True)
        rows = []
        traj: Dict[str, object] = {"clean": {}, "corrupted": {}}
        for cohort, (views, surv_df, sample_ids, _) in loaded.items():
            k_c = CANONICAL_K[cohort]
            sync_cuda()
            t0 = time.time()
            out = CLVSSC(k_c, use_null_snr=True, seed=42).fit(views)
            sync_cuda()
            dt = time.time() - t0
            met = evaluate_clustering(
                out["labels"], survival_df=surv_df, embedding=out.get("F"), compute_perm=True, n_perm=5000, perm_seed=42
            )
            fw = out["final_weights"]
            rows.append(
                {
                    "cohort": cohort.upper(),
                    "n_samples": len(sample_ids),
                    "k": k_c,
                    "method": "CL-VSSC (Null-SNR PI)",
                    "neg_log10_p": met.get("neg_log10_p", 0.0),
                    "logrank_p": met.get("logrank_p", 1.0),
                    "neg_log10_perm_p": met.get("neg_log10_perm_p", 0.0),
                    "perm_p": met.get("perm_p", 1.0),
                    "silhouette": met.get("silhouette", np.nan),
                    "min_cluster_size": met["min_cluster_size"],
                    "runtime_sec": round(dt, 4),
                    "w_exp": float(fw[0]),
                    "w_methy": float(fw[1]),
                    "w_mirna": float(fw[2]),
                    "lyap_init": float(out["lyapunov_history"][0]),
                    "lyap_final": float(out["lyapunov_history"][-1]),
                    "gap_init": float(out["spectral_gap_history"][0]),
                    "gap_final": float(out["spectral_gap_history"][-1]),
                }
            )
            traj["clean"][cohort.upper()] = {
                "lyapunov_history": [float(x) for x in out["lyapunov_history"]],
                "weights_history": [[float(y) for y in r] for r in out["weights_history"]],
                "spectral_gap_history": [float(x) for x in out["spectral_gap_history"]],
            }
            print(f"  [{cohort.upper():4s}] Null-SNR -log10perm={met['neg_log10_perm_p']:.2f} w={np.round(fw, 3)}", flush=True)
        df = pd.DataFrame(rows)
        df["fdr_q_perm"] = benjamini_hochberg(df["perm_p"].to_numpy())
        df.to_csv(clean_csv, index=False)
        traj_json.write_text(json.dumps(traj))
        print(
            f"  Null-SNR Clean summary: sig={int((df['perm_p'] < 0.05).sum())}/10 | "
            f"mean={df['neg_log10_perm_p'].mean():.3f} | med={df['neg_log10_perm_p'].median():.3f} | "
            f"sil={df['silhouette'].mean():.3f}",
            flush=True,
        )

    # V3.1b: k-sweep
    ks_csv = out_dir / "k_sens_null_snr_v3.csv"
    if not ks_csv.exists():
        print("\n=== V3.1b: Null-SNR PI k-Sensitivity Sweep ===", flush=True)
        k_rows = []
        for cohort, (views, surv_df, _, _) in loaded.items():
            for k_val in [3, 4, 5, 6]:
                out = CLVSSC(k_val, use_null_snr=True, seed=42).fit(views)
                met = evaluate_clustering(
                    out["labels"], survival_df=surv_df, embedding=out.get("F"), compute_perm=True, n_perm=2000, perm_seed=42
                )
                k_rows.append(
                    {
                        "cohort": cohort.upper(),
                        "k": k_val,
                        "method": "CL-VSSC (Null-SNR PI)",
                        "neg_log10_p": met.get("neg_log10_p", 0.0),
                        "neg_log10_perm_p": met.get("neg_log10_perm_p", 0.0),
                        "perm_p": met.get("perm_p", 1.0),
                    }
                )
        kdf = pd.DataFrame(k_rows)
        kdf.to_csv(ks_csv, index=False)
        print(f"  Null-SNR k-sweep: sig={int((kdf['perm_p'] < 0.05).sum())}/40 | mean={kdf['neg_log10_perm_p'].mean():.3f}", flush=True)

    # V3.1c: 3-modality fault stress (sigma in [0.0, 2.0, 4.0])
    fault_csv = out_dir / "fault_null_snr_v3.csv"
    if not fault_csv.exists():
        print("\n=== V3.1c: Null-SNR PI Fault Stress Test ===", flush=True)
        traj = json.loads(traj_json.read_text())
        f_rows = []
        stress_cohorts = ["bic", "kirc", "lihc", "gbm"]
        mod_names = {0: "mRNA", 1: "Methylation", 2: "miRNA"}
        for cohort in stress_cohorts:
            clean_views, surv_df, _, _ = loaded[cohort]
            k_c = CANONICAL_K[cohort]
            ref_lbl = CLVSSC(k_c, use_null_snr=True, seed=42).fit(clean_views)["labels"]
            for v_idx, mod_lbl in mod_names.items():
                for sigma, drop in [(0.0, 0.0), (2.0, 0.45), (4.0, 0.60)]:
                    seeds = [42] if sigma == 0.0 else [0, 1, 2]
                    for s_seed in seeds:
                        v_c = (
                            clean_views
                            if sigma == 0.0
                            else corrupt_view(clean_views, view_idx=v_idx, noise_std=sigma, dropout_rate=drop, seed=100 * v_idx + int(sigma * 10) + s_seed)
                        )
                        out = CLVSSC(k_c, use_null_snr=True, seed=42).fit(v_c)
                        met = evaluate_clustering(out["labels"], true_labels=ref_lbl, survival_df=surv_df, compute_perm=False)
                        w_corr = float(out["final_weights"][v_idx])
                        f_rows.append(
                            {
                                "cohort": cohort.upper(),
                                "view_idx": v_idx,
                                "modality": mod_lbl,
                                "noise_sigma": sigma,
                                "dropout": drop,
                                "seed": s_seed,
                                "method": "CL-VSSC (Null-SNR PI)",
                                "clean_retention_ari": met.get("ari", 1.0),
                                "clean_retention_nmi": met.get("nmi", 1.0),
                                "neg_log10_p": met.get("neg_log10_p", 0.0),
                                "w_corrupted": w_corr,
                            }
                        )
                        if cohort in ("bic", "kirc") and sigma == 4.0 and s_seed == 0:
                            traj["corrupted"][f"{cohort.upper()}_{mod_lbl}"] = {
                                "weights_history": [[float(y) for y in r] for r in out["weights_history"]],
                                "w_corr": w_corr,
                            }
        fdf = pd.DataFrame(f_rows)
        fdf.to_csv(fault_csv, index=False)
        traj_json.write_text(json.dumps(traj))
        print(fdf[fdf["noise_sigma"] == 4.0].groupby("modality")[["clean_retention_ari", "w_corrupted", "neg_log10_p"]].mean().round(3), flush=True)

    # V3.1d: Partial multi-omics (0%, 10%, 30%, 50%)
    part_csv = out_dir / "partial_null_snr_v3.csv"
    if not part_csv.exists():
        print("\n=== V3.1d: Null-SNR PI Partial Multi-Omics ===", flush=True)
        p_rows = []
        for cohort in ["bic", "kirc", "lihc", "gbm"]:
            clean_views, surv_df, _, _ = loaded[cohort]
            k_c = CANONICAL_K[cohort]
            n = clean_views[0].shape[0]
            ref_lbl = CLVSSC(k_c, use_null_snr=True, seed=42).fit(clean_views)["labels"]
            for mr in [0.0, 0.10, 0.30, 0.50]:
                seeds = [42] if mr == 0.0 else [0, 1, 2]
                for s_seed in seeds:
                    if mr == 0.0:
                        masks = None
                        views_p = clean_views
                    else:
                        rng = np.random.default_rng(500 + int(mr * 100) + s_seed)
                        masks = [rng.random(n) >= mr for _ in range(3)]
                        for i in range(n):
                            if not any(m[i] for m in masks):
                                masks[int(rng.integers(0, 3))][i] = True
                        views_p = []
                        for v_i in range(3):
                            arr = np.array(clean_views[v_i], copy=True, dtype=np.float32)
                            arr[~masks[v_i]] = 0.0
                            views_p.append(arr)
                    out = CLVSSC(k_c, use_null_snr=True, seed=42).fit(views_p, masks=masks)
                    met = evaluate_clustering(out["labels"], true_labels=ref_lbl, survival_df=surv_df, compute_perm=False)
                    p_rows.append(
                        {
                            "cohort": cohort.upper(),
                            "missing_rate": mr,
                            "seed": s_seed,
                            "method": "CL-VSSC (Null-SNR PI)",
                            "clean_retention_ari": met.get("ari", 1.0),
                            "clean_retention_acc": met.get("acc", 1.0),
                            "neg_log10_p": met.get("neg_log10_p", 0.0),
                        }
                    )
        pdf = pd.DataFrame(p_rows)
        pdf.to_csv(part_csv, index=False)
        print(pdf.groupby("missing_rate")[["clean_retention_ari", "clean_retention_acc", "neg_log10_p"]].mean().round(3), flush=True)


def run_v3_singlecell_and_sensitivity(loaded: Dict[str, tuple], work_dir: Path, out_dir: Path) -> None:
    sc_dir = work_dir / "sc_data"
    sc1_csv = out_dir / "sc_local_pi_v3.csv"
    sc2_csv = out_dir / "sc_fault_local_pi_v3.csv"
    sc3_csv = out_dir / "sc_scale_local_pi_v3.csv"
    sens_anc_csv = out_dir / "sensitivity_anchors_v3.csv"

    ad_bmmc = None
    bmmc10 = None
    if not (sc1_csv.exists() and sc2_csv.exists() and sc3_csv.exists() and sens_anc_csv.exists()):
        bmmc_path = download_and_extract_bmmc(sc_dir)
        ad_bmmc = ad.read_h5ad(bmmc_path)
        site_series = ad_bmmc.obs["site"].astype(str) if "site" in ad_bmmc.obs.columns else ad_bmmc.obs["batch"].astype(str).str[:2]
        site_counts = site_series.value_counts()
        s1_name, s2_name = str(site_counts.index[0]), str(site_counts.index[1])

        datasets = {}
        datasets["BMMC-Site1"] = extract_citeseq_views(ad_bmmc[site_series == s1_name], max_cells=None, seed=42)
        datasets["BMMC-Site2"] = extract_citeseq_views(ad_bmmc[site_series == s2_name], max_cells=None, seed=42)
        datasets["BMMC-10k"] = extract_citeseq_views(ad_bmmc, max_cells=10000, seed=42)
        datasets["BMMC-30k"] = extract_citeseq_views(ad_bmmc, max_cells=30000, seed=42)
        bmmc10 = datasets["BMMC-10k"]

        # V3.2a: Anchor-Local PI & Local+Null across 4 BMMC CITE-seq benchmarks
        if not sc1_csv.exists():
            print("\n=== V3.2a: CITE-seq Anchor-Local PI Benchmark ===", flush=True)
            rows = []
            for ds_name, (views, y_true, meta) in datasets.items():
                k_c, n_c = meta["n_classes"], meta["n_cells"]
                for seed in [42, 43, 44]:
                    runners = {
                        "CL-VSSC-Anchor (Null-SNR)": lambda v, k, s: CLVSSCAnchor(k, n_anchors=384, anchor_k=20, use_local_pi=False, use_null_snr=True, seed=s).fit(v),
                        "CL-VSSC-Anchor (Local PI)": lambda v, k, s: CLVSSCAnchor(k, n_anchors=384, anchor_k=20, use_local_pi=True, use_null_snr=False, seed=s).fit(v),
                        "CL-VSSC-Anchor (Local+Null)": lambda v, k, s: CLVSSCAnchor(k, n_anchors=384, anchor_k=20, use_local_pi=True, use_null_snr=True, seed=s).fit(v),
                    }
                    for m_name, fn in runners.items():
                        sync_cuda()
                        t0 = time.time()
                        out = fn(views, k_c, seed)
                        sync_cuda()
                        dt = time.time() - t0
                        met = evaluate_clustering(out["labels"], true_labels=y_true, embedding=out.get("F"))
                        rows.append(
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
                                "w_rna": float(out["final_weights"][0]),
                                "w_adt": float(out["final_weights"][1]),
                            }
                        )
            df1 = pd.DataFrame(rows)
            df1.to_csv(sc1_csv, index=False)
            print(df1.groupby(["dataset", "method"])[["ari", "nmi", "acc", "w_adt", "runtime_sec"]].mean().round(3), flush=True)

        # V3.2b: BMMC-10k scRNA & Protein_ADT corruption
        if not sc2_csv.exists():
            print("\n=== V3.2b: BMMC-10k Local PI Corruption Test ===", flush=True)
            v_b10, y_b10, m_b10 = bmmc10
            k_b10 = m_b10["n_classes"]
            f_rows = []
            runners = {
                "CL-VSSC-Anchor (Null-SNR)": lambda v: CLVSSCAnchor(k_b10, n_anchors=384, anchor_k=20, use_local_pi=False, use_null_snr=True, seed=42).fit(v),
                "CL-VSSC-Anchor (Local PI)": lambda v: CLVSSCAnchor(k_b10, n_anchors=384, anchor_k=20, use_local_pi=True, use_null_snr=False, seed=42).fit(v),
                "CL-VSSC-Anchor (Local+Null)": lambda v: CLVSSCAnchor(k_b10, n_anchors=384, anchor_k=20, use_local_pi=True, use_null_snr=True, seed=42).fit(v),
            }
            for v_idx, mod_lbl in [(0, "scRNA"), (1, "Protein_ADT")]:
                for sigma, drop in [(0.0, 0.0), (1.5, 0.30), (3.0, 0.50)]:
                    seeds = [42] if sigma == 0.0 else [0, 1, 2]
                    for s_seed in seeds:
                        v_c = v_b10 if sigma == 0.0 else corrupt_view(v_b10, view_idx=v_idx, noise_std=sigma, dropout_rate=drop, seed=200 * v_idx + int(sigma * 10) + s_seed)
                        for m_name, fn in runners.items():
                            out = fn(v_c)
                            met = evaluate_clustering(out["labels"], true_labels=y_b10)
                            f_rows.append(
                                {
                                    "corrupted_modality": mod_lbl,
                                    "noise_sigma": sigma,
                                    "dropout": drop,
                                    "seed": s_seed,
                                    "method": m_name,
                                    "gt_ari": met["ari"],
                                    "gt_nmi": met["nmi"],
                                    "gt_acc": met["acc"],
                                    "w_corrupted": float(out["final_weights"][v_idx]),
                                }
                            )
            df2 = pd.DataFrame(f_rows)
            df2.to_csv(sc2_csv, index=False)
            print(df2.groupby(["corrupted_modality", "noise_sigma", "method"])[["gt_ari", "gt_acc", "w_corrupted"]].mean().round(3), flush=True)

        # V3.2c: Full 1k -> 90,261 scaling for Local PI
        if not sc3_csv.exists():
            print("\n=== V3.2c: Full 1k->90k Scalability for Anchor-Local PI ===", flush=True)
            del datasets
            gc.collect()
            s_rows = []
            for n_sub in [1000, 2500, 5000, 10000, 25000, 50000, None]:
                v_s, y_s, m_s = extract_citeseq_views(ad_bmmc, max_cells=n_sub, seed=101)
                n_actual, k_s = m_s["n_cells"], m_s["n_classes"]
                for m_name, null_flag in [("CL-VSSC-Anchor (Local PI)", False), ("CL-VSSC-Anchor (Local+Null)", True)]:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    sync_cuda()
                    t0 = time.time()
                    out = CLVSSCAnchor(k_s, n_anchors=384, anchor_k=20, use_local_pi=True, use_null_snr=null_flag, seed=42).fit(v_s)
                    sync_cuda()
                    dt = time.time() - t0
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                    met = evaluate_clustering(out["labels"], true_labels=y_s)
                    s_rows.append(
                        {
                            "n_cells": n_actual,
                            "method": m_name,
                            "runtime_sec": round(dt, 3),
                            "peak_vram_mb": round(peak_mb, 1),
                            "gt_ari": met["ari"],
                            "gt_acc": met["acc"],
                        }
                    )
                print(f"  [scale local n={n_actual:5d}] done", flush=True)
            pd.DataFrame(s_rows).to_csv(sc3_csv, index=False)

        # V3.3c: Anchor count m sensitivity on BMMC-10k
        if not sens_anc_csv.exists():
            print("\n=== V3.3c: Anchor Count m Sensitivity on BMMC-10k ===", flush=True)
            v_b10, y_b10, m_b10 = bmmc10
            k_b10 = m_b10["n_classes"]
            v_b10_adt3 = corrupt_view(v_b10, view_idx=1, noise_std=3.0, dropout_rate=0.50, seed=200 + 30 + 0)
            a_rows = []
            for m_anc in [128, 256, 384, 512, 768]:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                sync_cuda()
                t0 = time.time()
                out_c = CLVSSCAnchor(k_b10, n_anchors=m_anc, anchor_k=20, use_local_pi=True, use_null_snr=False, seed=42).fit(v_b10)
                sync_cuda()
                dt = time.time() - t0
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                met_c = evaluate_clustering(out_c["labels"], true_labels=y_b10)

                out_f = CLVSSCAnchor(k_b10, n_anchors=m_anc, anchor_k=20, use_local_pi=True, use_null_snr=False, seed=42).fit(v_b10_adt3)
                met_f = evaluate_clustering(out_f["labels"], true_labels=y_b10)
                a_rows.append(
                    {
                        "n_anchors": m_anc,
                        "clean_ari": met_c["ari"],
                        "clean_nmi": met_c["nmi"],
                        "clean_acc": met_c["acc"],
                        "adt_fault_ari": met_f["ari"],
                        "adt_fault_acc": met_f["acc"],
                        "w_adt_fault": float(out_f["final_weights"][1]),
                        "runtime_sec": round(dt, 3),
                        "peak_vram_mb": round(peak_mb, 1),
                    }
                )
            adf = pd.DataFrame(a_rows)
            adf.to_csv(sens_anc_csv, index=False)
            print(adf.round(3).to_string(index=False), flush=True)

    # V3.3a & V3.3b: Control gain sensitivity grids on BIC & KIRC
    pi_csv = out_dir / "sensitivity_pi_grid_v3.csv"
    if not pi_csv.exists():
        print("\n=== V3.3a: PI Gain (kappa_I x tau) Sensitivity Grid (BIC & KIRC) ===", flush=True)
        pi_rows = []
        kappas = [0.0, 0.20, 0.35, 0.60, 1.0]
        taus = [0.10, 0.15, 0.20, 0.30, 0.40]
        for cohort in ["bic", "kirc"]:
            views, surv_df, _, _ = loaded[cohort]
            k_c = CANONICAL_K[cohort]
            views_corr = corrupt_view(views, view_idx=0, noise_std=4.0, dropout_rate=0.60, seed=40)
            for kp in kappas:
                for tau in taus:
                    out_c = CLVSSC(k_c, kappa_i=kp, tau=tau, seed=42).fit(views)
                    met_c = evaluate_clustering(out_c["labels"], survival_df=surv_df, compute_perm=True, n_perm=2000, perm_seed=42)
                    out_f = CLVSSC(k_c, kappa_i=kp, tau=tau, seed=42).fit(views_corr)
                    met_f = evaluate_clustering(out_f["labels"], true_labels=out_c["labels"], survival_df=surv_df, compute_perm=False)
                    pi_rows.append(
                        {
                            "cohort": cohort.upper(),
                            "kappa_i": kp,
                            "tau": tau,
                            "clean_neg_log10_perm_p": met_c["neg_log10_perm_p"],
                            "fault_ari": met_f["ari"],
                            "fault_w_mrna": float(out_f["final_weights"][0]),
                            "fault_neg_log10_p": met_f.get("neg_log10_p", 0.0),
                        }
                    )
        pd.DataFrame(pi_rows).to_csv(pi_csv, index=False)
        print("  Saved sensitivity_pi_grid_v3.csv", flush=True)

    mu_csv = out_dir / "sensitivity_mu_v3.csv"
    if not mu_csv.exists():
        print("\n=== V3.3b: Spectral-Gap Gain (mu_max) Sensitivity (BIC & KIRC) ===", flush=True)
        mu_rows = []
        for cohort in ["bic", "kirc"]:
            views, surv_df, _, _ = loaded[cohort]
            k_c = CANONICAL_K[cohort]
            for mu_m in [0.0, 0.06, 0.12, 0.18, 0.24]:
                out = CLVSSC(k_c, mu_max=mu_m, use_spectral_control=(mu_m > 0.0), seed=42).fit(views)
                met = evaluate_clustering(out["labels"], survival_df=surv_df, embedding=out.get("F"), compute_perm=True, n_perm=2000, perm_seed=42)
                mu_rows.append(
                    {
                        "cohort": cohort.upper(),
                        "mu_max": mu_m,
                        "clean_neg_log10_perm_p": met["neg_log10_perm_p"],
                        "silhouette": met.get("silhouette", np.nan),
                        "gap_init": float(out["spectral_gap_history"][0]),
                        "gap_final": float(out["spectral_gap_history"][-1]),
                        "lyap_delta": float(out["lyapunov_history"][-1] - out["lyapunov_history"][0]),
                    }
                )
        mdf = pd.DataFrame(mu_rows)
        mdf.to_csv(mu_csv, index=False)
        print(mdf.round(4).to_string(index=False), flush=True)


def main() -> None:
    t0 = time.time()
    work_dir = Path("/content") if Path("/content").exists() else Path.cwd()
    data_dir = work_dir / "tcga_benchmark_data"
    out_dir = work_dir / "results_v3"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[V3 Upgrades] Python {platform.python_version()} | Torch {torch.__version__} | {dev} ({gpu_name})", flush=True)

    loaded = {cohort: load_tcga_cohort(cohort, data_dir, top_features=2000) for cohort in COHORTS}
    run_v3_tcga_null_snr(loaded, out_dir)
    run_v3_singlecell_and_sensitivity(loaded, work_dir, out_dir)

    env = {"gpu_name": gpu_name, "torch_version": torch.__version__, "v3_elapsed_sec": round(time.time() - t0, 2)}
    (out_dir / "v3_env.json").write_text(json.dumps(env, indent=2))
    zip_path = Path(shutil.make_archive(str(work_dir / "cl_vssc_results_v3"), "zip", out_dir))
    print(f"V3_SUITE_COMPLETE: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB) in {env['v3_elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
