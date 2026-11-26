"""Stage A (v2): 10-TCGA Clean Benchmark, k-Sweep, 3-Modality Fault Tolerance, and Partial-Omics Benchmark on GPU."""
from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import lifelines
import numpy as np
import pandas as pd
import torch

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _cand in [str(_THIS_DIR), "/content", str(Path.cwd() / "src")]:
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from baselines import (  # noqa: E402
    corrupt_view,
    evaluate_clustering,
    run_attention_moae,
    run_mcgc,
    run_nemo,
    run_openloop_mavs,
    run_snf,
    run_subtype_gan,
    run_wnn,
)
from cl_vssc import CLVSSC, get_device  # noqa: E402
from tcga_loader import CANONICAL_K, COHORTS, load_tcga_cohort  # noqa: E402


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    raw_q = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    out = np.empty_like(q_sorted)
    out[order] = np.clip(q_sorted, 0.0, 1.0)
    return out


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    t_start = time.time()
    work_dir = Path("/content") if Path("/content").exists() else Path.cwd()
    data_dir = work_dir / "tcga_benchmark_data"
    out_dir = work_dir / "results_v2"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Stage A] Python {platform.python_version()} | PyTorch {torch.__version__} | {dev} ({gpu_name})", flush=True)

    loaded: Dict[str, tuple] = {}
    meta_rows = []
    for cohort in COHORTS:
        views, surv_df, ids, meta = load_tcga_cohort(cohort, data_dir, top_features=2000)
        loaded[cohort] = (views, surv_df, ids, meta)
        meta_rows.append(meta)
        print(f"  Loaded {cohort.upper():4s}: n={meta['n_samples']:3d}, events={meta['n_events']:3d}", flush=True)
    pd.DataFrame(meta_rows).to_csv(out_dir / "cohort_summary_v2.csv", index=False)

    # 1. Clean 10-TCGA Benchmark (12 methods)
    traj_path = out_dir / "diagnostics_trajectories_v2.json"
    bench_csv = out_dir / "tcga_benchmark_v2.csv"
    assign_csv = out_dir / "survival_assignments_v2.csv"
    if bench_csv.exists() and assign_csv.exists() and traj_path.exists():
        print("\n=== Stage A.1: Skipping (checkpoint exists) ===", flush=True)
        traj = json.loads(traj_path.read_text())
    else:
        print("\n=== Stage A.1: 10-TCGA 12-Method Clean Benchmark ===", flush=True)
        bench_rows = []
        assign_rows = []
        traj: Dict[str, object] = {"clean_dz": {}, "clean_nodz": {}, "corrupted_dz": {}, "corrupted_nodz": {}, "corrupted_ponly": {}}

        for cohort, (views, surv_df, sample_ids, _) in loaded.items():
            k_c = CANONICAL_K[cohort]
            runners = {
                "CL-VSSC (DZ-PI)": lambda v, k: CLVSSC(k, dead_zone=0.14, seed=42).fit(v),
                "CL-VSSC (No-DZ PI)": lambda v, k: CLVSSC(k, dead_zone=0.0, seed=42).fit(v),
                "CL-VSSC (P-only)": lambda v, k: CLVSSC(k, dead_zone=0.14, kappa_i=0.0, seed=42).fit(v),
                "CL-VSSC (w/o Feedback)": lambda v, k: CLVSSC(k, use_feedback=False, seed=42).fit(v),
                "CL-VSSC (w/o SpecCtrl)": lambda v, k: CLVSSC(k, dead_zone=0.14, use_spectral_control=False, seed=42).fit(v),
                "OpenLoop_MAVS": lambda v, k: run_openloop_mavs(v, k, seed=42),
                "SNF": lambda v, k: run_snf(v, k, seed=42),
                "NEMO": lambda v, k: run_nemo(v, k, seed=42),
                "MCGC": lambda v, k: run_mcgc(v, k, seed=42),
                "Subtype-GAN": lambda v, k: run_subtype_gan(v, k, epochs=100, seed=42),
                "Attention-MOAE": lambda v, k: run_attention_moae(v, k, epochs=100, seed=42),
                "Seurat-WNN": lambda v, k: run_wnn(v, k, seed=42),
            }
            cohort_labels: Dict[str, np.ndarray] = {}
            for m_name, fn in runners.items():
                sync_cuda()
                t0 = time.time()
                out = fn(views, k_c)
                sync_cuda()
                elapsed = time.time() - t0
                cohort_labels[m_name] = out["labels"]
                metrics = evaluate_clustering(
                    out["labels"], survival_df=surv_df, embedding=out.get("F"), compute_perm=True, n_perm=5000, perm_seed=42
                )
                row = {
                    "cohort": cohort.upper(),
                    "n_samples": len(sample_ids),
                    "n_events": int(surv_df["event"].sum()),
                    "k": k_c,
                    "method": m_name,
                    "neg_log10_p": metrics.get("neg_log10_p", 0.0),
                    "logrank_p": metrics.get("logrank_p", 1.0),
                    "neg_log10_perm_p": metrics.get("neg_log10_perm_p", 0.0),
                    "perm_p": metrics.get("perm_p", 1.0),
                    "silhouette": metrics.get("silhouette", np.nan),
                    "min_cluster_size": metrics["min_cluster_size"],
                    "runtime_sec": round(elapsed, 4),
                }
                if "final_weights" in out:
                    fw = out["final_weights"]
                    row["w_exp"], row["w_methy"], row["w_mirna"] = float(fw[0]), float(fw[1]), float(fw[2])
                if "lyapunov_history" in out:
                    row["lyap_init"] = float(out["lyapunov_history"][0])
                    row["lyap_final"] = float(out["lyapunov_history"][-1])
                    row["gap_init"] = float(out["spectral_gap_history"][0])
                    row["gap_final"] = float(out["spectral_gap_history"][-1])
                if m_name == "CL-VSSC (DZ-PI)":
                    traj["clean_dz"][cohort.upper()] = {
                        "lyapunov_history": [float(x) for x in out["lyapunov_history"]],
                        "weights_history": [[float(y) for y in r] for r in out["weights_history"]],
                        "spectral_gap_history": [float(x) for x in out["spectral_gap_history"]],
                        "mu_history": [float(x) for x in out["mu_history"]],
                    }
                elif m_name == "CL-VSSC (No-DZ PI)" and cohort in ("bic", "kirc"):
                    traj["clean_nodz"][cohort.upper()] = {
                        "weights_history": [[float(y) for y in r] for r in out["weights_history"]],
                    }
                bench_rows.append(row)

            for idx_s, sid in enumerate(sample_ids):
                assign_rows.append(
                    {
                        "cohort": cohort.upper(),
                        "sample_id": sid,
                        "time": float(surv_df.iloc[idx_s]["time"]),
                        "event": int(surv_df.iloc[idx_s]["event"]),
                        "cl_vssc_cluster": int(cohort_labels["CL-VSSC (DZ-PI)"][idx_s]),
                    }
                )
            dz_r = [r for r in bench_rows if r["cohort"] == cohort.upper() and r["method"] == "CL-VSSC (DZ-PI)"][0]
            print(
                f"  [{cohort.upper():4s}] DZ-PI -log10perm={dz_r['neg_log10_perm_p']:.2f} "
                f"w=[{dz_r['w_exp']:.2f},{dz_r['w_methy']:.2f},{dz_r['w_mirna']:.2f}] "
                f"V:{dz_r['lyap_init']:.3f}->{dz_r['lyap_final']:.3f}",
                flush=True,
            )

        bench_df = pd.DataFrame(bench_rows)
        for m_name, grp in bench_df.groupby("method"):
            bench_df.loc[grp.index, "fdr_q_asymp"] = benjamini_hochberg(grp["logrank_p"].to_numpy())
            bench_df.loc[grp.index, "fdr_q_perm"] = benjamini_hochberg(grp["perm_p"].to_numpy())
        bench_df.to_csv(bench_csv, index=False)
        pd.DataFrame(assign_rows).to_csv(assign_csv, index=False)
        traj_path.write_text(json.dumps(traj))

        print("  Clean summary (10 cohorts):")
        print(bench_df.groupby("method")[["neg_log10_perm_p", "neg_log10_p", "silhouette"]].agg(["mean", "median"]).round(3).to_string(), flush=True)

    # 2. k-Sensitivity Sweep (k in {3, 4, 5, 6})
    k_csv = out_dir / "k_sensitivity_v2.csv"
    if k_csv.exists():
        print("\n=== Stage A.2: Skipping (checkpoint exists) ===", flush=True)
    else:
        print("\n=== Stage A.2: k-Sensitivity Sweep ===", flush=True)
        k_rows = []
        sweep_methods = {
            "CL-VSSC (DZ-PI)": lambda v, k: CLVSSC(k, dead_zone=0.14, seed=42).fit(v),
            "OpenLoop_MAVS": lambda v, k: run_openloop_mavs(v, k, seed=42),
            "SNF": lambda v, k: run_snf(v, k, seed=42),
            "NEMO": lambda v, k: run_nemo(v, k, seed=42),
            "MCGC": lambda v, k: run_mcgc(v, k, seed=42),
            "Subtype-GAN": lambda v, k: run_subtype_gan(v, k, epochs=80, seed=42),
            "Attention-MOAE": lambda v, k: run_attention_moae(v, k, epochs=80, seed=42),
            "Seurat-WNN": lambda v, k: run_wnn(v, k, seed=42),
        }
        for cohort, (views, surv_df, _, _) in loaded.items():
            for k_val in [3, 4, 5, 6]:
                for m_name, fn in sweep_methods.items():
                    out = fn(views, k_val)
                    metrics = evaluate_clustering(
                        out["labels"], survival_df=surv_df, embedding=out.get("F"), compute_perm=True, n_perm=2000, perm_seed=42
                    )
                    k_rows.append(
                        {
                            "cohort": cohort.upper(),
                            "k": k_val,
                            "method": m_name,
                            "neg_log10_p": metrics.get("neg_log10_p", 0.0),
                            "neg_log10_perm_p": metrics.get("neg_log10_perm_p", 0.0),
                            "perm_p": metrics.get("perm_p", 1.0),
                        }
                    )
            print(f"  [k-sweep] {cohort.upper()}", flush=True)
        pd.DataFrame(k_rows).to_csv(k_csv, index=False)

    # 3. 3-Modality Sensor-Fault Stress Test (10 methods)
    stress_cohorts = ["bic", "kirc", "lihc", "gbm"]
    fault_csv = out_dir / "fault_tolerance_v2.csv"
    if fault_csv.exists() and "BIC_mRNA" in traj.get("corrupted_ponly", {}):
        print("\n=== Stage A.3: Skipping (checkpoint exists) ===", flush=True)
    else:
        print("\n=== Stage A.3: 3-Modality Sensor-Fault Stress Test ===", flush=True)
        stress_levels = [(0.0, 0.0), (0.5, 0.15), (1.0, 0.30), (2.0, 0.45), (4.0, 0.60)]
        mod_names = {0: "mRNA", 1: "Methylation", 2: "miRNA"}
        stress_methods = {
            "CL-VSSC (DZ-PI)": lambda v, k: CLVSSC(k, dead_zone=0.14, seed=42).fit(v),
            "CL-VSSC (No-DZ PI)": lambda v, k: CLVSSC(k, dead_zone=0.0, seed=42).fit(v),
            "CL-VSSC (w/o Feedback)": lambda v, k: CLVSSC(k, use_feedback=False, seed=42).fit(v),
            "OpenLoop_MAVS": lambda v, k: run_openloop_mavs(v, k, seed=42),
            "SNF": lambda v, k: run_snf(v, k, seed=42),
            "NEMO": lambda v, k: run_nemo(v, k, seed=42),
            "MCGC": lambda v, k: run_mcgc(v, k, seed=42),
            "Subtype-GAN": lambda v, k: run_subtype_gan(v, k, epochs=80, seed=42),
            "Attention-MOAE": lambda v, k: run_attention_moae(v, k, epochs=80, seed=42),
            "Seurat-WNN": lambda v, k: run_wnn(v, k, seed=42),
        }

        stress_rows = []
        for cohort in stress_cohorts:
            clean_views, surv_df, _, _ = loaded[cohort]
            k_c = CANONICAL_K[cohort]
            clean_refs = {m_name: fn(clean_views, k_c)["labels"] for m_name, fn in stress_methods.items()}
            for view_idx, mod_label in mod_names.items():
                for sigma, drop in stress_levels:
                    seeds = [42] if sigma == 0.0 else [0, 1, 2]
                    for s_seed in seeds:
                        views_corr = (
                            clean_views
                            if sigma == 0.0
                            else corrupt_view(clean_views, view_idx=view_idx, noise_std=sigma, dropout_rate=drop, seed=100 * view_idx + int(sigma * 10) + s_seed)
                        )
                        for m_name, fn in stress_methods.items():
                            out = fn(views_corr, k_c)
                            metrics = evaluate_clustering(
                                out["labels"], true_labels=clean_refs[m_name], survival_df=surv_df, embedding=out.get("F"), compute_perm=False
                            )
                            w_corr = float(out["final_weights"][view_idx]) if "final_weights" in out else np.nan
                            stress_rows.append(
                                {
                                    "cohort": cohort.upper(),
                                    "view_idx": view_idx,
                                    "modality": mod_label,
                                    "noise_sigma": sigma,
                                    "dropout": drop,
                                    "seed": s_seed,
                                    "method": m_name,
                                    "clean_retention_ari": metrics.get("ari", 1.0),
                                    "clean_retention_nmi": metrics.get("nmi", 1.0),
                                    "neg_log10_p": metrics.get("neg_log10_p", 0.0),
                                    "w_corrupted": w_corr,
                                }
                            )
                            if cohort in ("bic", "kirc") and sigma == 4.0 and s_seed == 0:
                                key = f"{cohort.upper()}_{mod_label}"
                                if m_name == "CL-VSSC (DZ-PI)":
                                    traj["corrupted_dz"][key] = {
                                        "lyapunov_history": [float(x) for x in out["lyapunov_history"]],
                                        "weights_history": [[float(y) for y in r] for r in out["weights_history"]],
                                        "spectral_gap_history": [float(x) for x in out["spectral_gap_history"]],
                                        "w_corr": w_corr,
                                    }
                                elif m_name == "CL-VSSC (No-DZ PI)":
                                    traj["corrupted_nodz"][key] = {
                                        "weights_history": [[float(y) for y in r] for r in out["weights_history"]],
                                    }
            print(f"  [stress] {cohort.upper()}", flush=True)
        pd.DataFrame(stress_rows).to_csv(fault_csv, index=False)

        # Save P-only corrupted trajectories on BIC/KIRC mRNA
        for cohort in ["bic", "kirc"]:
            clean_views, _, _, _ = loaded[cohort]
            k_c = CANONICAL_K[cohort]
            views_corr = corrupt_view(clean_views, view_idx=0, noise_std=4.0, dropout_rate=0.60, seed=40)
            out_p = CLVSSC(k_c, dead_zone=0.14, kappa_i=0.0, seed=42).fit(views_corr)
            traj["corrupted_ponly"][f"{cohort.upper()}_mRNA"] = {
                "weights_history": [[float(y) for y in r] for r in out_p["weights_history"]],
            }
        traj_path.write_text(json.dumps(traj))

    # 4. Partial / Incomplete Multi-Omics Benchmark
    partial_csv = out_dir / "partial_omics_v2.csv"
    if partial_csv.exists():
        print("\n=== Stage A.4: Skipping (checkpoint exists) ===", flush=True)
    else:
        print("\n=== Stage A.4: Partial Multi-Omics Benchmark ===", flush=True)
        partial_rows = []
        miss_rates = [0.0, 0.10, 0.30, 0.50]
        partial_methods = {
            "CL-VSSC (DZ-PI)": lambda v, k, m: CLVSSC(k, dead_zone=0.14, seed=42).fit(v, masks=m),
            "NEMO": lambda v, k, m: run_nemo(v, k, masks=m, seed=42),
            "SNF (0-imp)": lambda v, k, m: run_snf(v, k, seed=42),
            "OpenLoop_MAVS (0-imp)": lambda v, k, m: run_openloop_mavs(v, k, seed=42),
            "Subtype-GAN (0-imp)": lambda v, k, m: run_subtype_gan(v, k, epochs=80, seed=42),
        }

        for cohort in stress_cohorts:
            clean_views, surv_df, _, _ = loaded[cohort]
            k_c = CANONICAL_K[cohort]
            n = clean_views[0].shape[0]
            clean_refs = {m_name: fn(clean_views, k_c, None)["labels"] for m_name, fn in partial_methods.items()}
            for mr in miss_rates:
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
                    for m_name, fn in partial_methods.items():
                        out = fn(views_p, k_c, masks)
                        metrics = evaluate_clustering(
                            out["labels"], true_labels=clean_refs[m_name], survival_df=surv_df, compute_perm=False
                        )
                        partial_rows.append(
                            {
                                "cohort": cohort.upper(),
                                "missing_rate": mr,
                                "seed": s_seed,
                                "method": m_name,
                                "clean_retention_ari": metrics.get("ari", 1.0),
                                "clean_retention_acc": metrics.get("acc", 1.0),
                                "neg_log10_p": metrics.get("neg_log10_p", 0.0),
                            }
                        )
            print(f"  [partial] {cohort.upper()}", flush=True)
        pd.DataFrame(partial_rows).to_csv(partial_csv, index=False)

    env = {"gpu_name": gpu_name, "torch_version": torch.__version__, "stage_a_sec": round(time.time() - t_start, 2)}
    (out_dir / "stage_a_env.json").write_text(json.dumps(env, indent=2))
    print(f"STAGE_A_COMPLETE in {env['stage_a_sec']}s", flush=True)


if __name__ == "__main__":
    main()
