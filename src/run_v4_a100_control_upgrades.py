"""V4 A100 Control Upgrades:
1. Dense PI Gain (K_p, K_i) & Dead-Zone (delta_dz) Stability Phase Diagram (BIC & KIRC, 4-shard parallel).
2. Ultra-Large-Scale Linear Complexity (N = 10k -> 300k on GSE194122 + GSE194122-Expanded).
3. N = 200,000 Closed-Loop Static & Mid-Stream Step-Fault Self-Healing Verification.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import contextlib
import gc
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _cand in [str(_THIS_DIR), "/content/paper2_v4", "/content", str(Path.cwd() / "src")]:
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from baselines import corrupt_view, evaluate_clustering  # noqa: E402
from cl_vssc import CLVSSC, CLVSSCAnchor, get_device  # noqa: E402
from run_v2_singlecell_suite import download_and_extract_bmmc, extract_citeseq_views  # noqa: E402
from tcga_loader import CANONICAL_K, load_tcga_cohort  # noqa: E402


def _single_thread_ctx():
    if threadpool_limits is not None:
        return threadpool_limits(limits=1)
    return contextlib.nullcontext()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _entropy_norm(w: np.ndarray) -> float:
    w_clip = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(w_clip * np.log(w_clip)) / np.log(max(2, len(w_clip))))


def _classify_regime(
    fault_w_final: float,
    clean_ent: float,
    tail_osc_l1: float,
    lyap_delta: float,
    lyap_nonmono_frac: float,
    total_variation_w: float,
    net_disp_w: float,
) -> str:
    """Objective regime classification for closed-loop PI mirror-descent stability."""
    if clean_ent < 0.35 or lyap_delta > 0.0:
        return "Collapse / Unstable"
    if fault_w_final > 0.05:
        return "Under-Actuated / Stalled"
    ringing = (
        tail_osc_l1 > 0.010
        or lyap_nonmono_frac > 0.20
        or (net_disp_w > 1e-4 and total_variation_w > 1.35 * net_disp_w)
    )
    if ringing:
        return "Oscillatory"
    return "Convergent"


def _eval_v41_point(
    cohort: str,
    views: List[np.ndarray],
    views_corr: List[np.ndarray],
    surv_df: pd.DataFrame,
    ref_clean: np.ndarray,
    k_c: int,
    dz: float,
    kp: float,
    kr: float,
    eta_w: float = 0.30,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    tau = eta_w / kp
    ki = kp * kr
    with _single_thread_ctx():
        out_c = CLVSSC(k_c, eta_w=eta_w, tau=tau, kappa_i=kr, dead_zone=dz, seed=42).fit(views)
        met_c = evaluate_clustering(
            out_c["labels"], survival_df=surv_df, embedding=out_c.get("F"), compute_perm=False
        )
        out_f = CLVSSC(k_c, eta_w=eta_w, tau=tau, kappa_i=kr, dead_zone=dz, seed=42).fit(views_corr)
        met_f = evaluate_clustering(
            out_f["labels"], true_labels=ref_clean, survival_df=surv_df, compute_perm=False
        )

    wh_f = np.asarray(out_f["weights_history"], dtype=np.float64)
    vh_f = np.asarray(out_f["lyapunov_history"], dtype=np.float64)
    w_mrna = wh_f[:, 0]
    w_corr_final = float(w_mrna[-1])

    settled_flags = w_mrna <= 0.05
    settling_iter = int(len(w_mrna) - 1)
    for idx_t in range(len(w_mrna)):
        if np.all(settled_flags[idx_t:]):
            settling_iter = int(idx_t)
            break

    dw = np.abs(np.diff(wh_f, axis=0)).sum(axis=1)
    tail_osc = float(dw[-10:].max()) if len(dw) >= 10 else float(dw.max())
    tv_w = float(dw.sum())
    net_disp_w = float(np.abs(wh_f[-1] - wh_f[0]).sum())

    dv = np.diff(vh_f)
    lyap_delta = float(vh_f[-1] - vh_f[0])
    lyap_nonmono = float(np.mean(dv[5:] > 1e-4)) if len(dv) > 5 else float(np.mean(dv > 1e-4))

    wh_c = np.asarray(out_c["weights_history"], dtype=np.float64)
    dw_c = np.abs(np.diff(wh_c, axis=0)).sum(axis=1)
    clean_ent = _entropy_norm(out_c["final_weights"])
    clean_osc = float(dw_c[-10:].max()) if len(dw_c) >= 10 else float(dw_c.max())

    regime = _classify_regime(
        fault_w_final=w_corr_final,
        clean_ent=clean_ent,
        tail_osc_l1=max(tail_osc, clean_osc),
        lyap_delta=lyap_delta,
        lyap_nonmono_frac=lyap_nonmono,
        total_variation_w=tv_w,
        net_disp_w=net_disp_w,
    )

    row: Dict[str, object] = {
        "cohort": cohort.upper(),
        "dead_zone": dz,
        "k_p": kp,
        "k_i": round(ki, 6),
        "kappa_i": kr,
        "tau": round(tau, 6),
        "fault_w_corr_final": w_corr_final,
        "settling_iter": settling_iter,
        "tail_osc_l1": tail_osc,
        "clean_tail_osc_l1": clean_osc,
        "total_variation_w": tv_w,
        "net_disp_w": net_disp_w,
        "lyap_delta": lyap_delta,
        "lyap_nonmono_frac": lyap_nonmono,
        "clean_entropy_norm": clean_ent,
        "clean_silhouette": float(met_c.get("silhouette", np.nan)),
        "clean_neg_log10_p": float(met_c.get("neg_log10_p", 0.0)),
        "fault_ari": float(met_f.get("ari", 0.0)),
        "regime": regime,
    }
    return row, wh_c, wh_f, vh_f


def _v41_shard_worker(
    rank: int,
    tasks: List[Tuple[str, float, float, float]],
    data_dir_str: str,
    out_dir_str: str,
) -> None:
    torch.set_num_threads(1)
    data_dir = Path(data_dir_str)
    out_dir = Path(out_dir_str)
    shard_csv = out_dir / f"_v41_shard_{rank}.csv"

    rows: List[Dict[str, object]] = []
    done_keys = set()
    if shard_csv.exists() and shard_csv.stat().st_size > 0:
        prev = pd.read_csv(shard_csv)
        rows = prev.to_dict("records")
        for r in rows:
            done_keys.add((str(r["cohort"]), round(float(r["dead_zone"]), 4), round(float(r["k_p"]), 4), round(float(r["kappa_i"]), 4)))

    cohort_cache: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], pd.DataFrame, np.ndarray, int]] = {}
    for cohort in sorted({t[0] for t in tasks}):
        views, surv_df, _, _ = load_tcga_cohort(cohort, data_dir, top_features=2000)
        k_c = CANONICAL_K[cohort]
        views_corr = corrupt_view(views, view_idx=0, noise_std=4.0, dropout_rate=0.60, seed=40)
        with _single_thread_ctx():
            ref_clean = CLVSSC(k_c, seed=42).fit(views)["labels"]
        cohort_cache[cohort] = (views, views_corr, surv_df, ref_clean, k_c)

    t0 = time.time()
    completed_now = 0
    for idx, (cohort, dz, kp, kr) in enumerate(tasks, start=1):
        key = (cohort.upper(), round(dz, 4), round(kp, 4), round(kr, 4))
        if key in done_keys:
            continue
        views, views_corr, surv_df, ref_clean, k_c = cohort_cache[cohort]
        row, _, _, _ = _eval_v41_point(cohort, views, views_corr, surv_df, ref_clean, k_c, dz, kp, kr)
        rows.append(row)
        completed_now += 1
        if completed_now % 15 == 0 or idx == len(tasks):
            pd.DataFrame(rows).to_csv(shard_csv, index=False)
            print(
                f"  [V4.1 shard {rank}] {len(rows)}/{len(tasks)} pts ({time.time() - t0:.1f}s)",
                flush=True,
            )

    pd.DataFrame(rows).to_csv(shard_csv, index=False)
    print(f"  [V4.1 shard {rank}] DONE {len(rows)}/{len(tasks)} pts in {time.time() - t0:.1f}s", flush=True)


def run_v4_pi_stability_phase(data_dir: Path, out_dir: Path) -> Dict[str, object]:
    """V4.1: Dense (K_p, K_i) & dead-zone delta_dz grid sweep on TCGA BIC and KIRC across 4 workers."""
    phase_csv = out_dir / "pi_stability_phase_v4.csv"
    traj_json = out_dir / "pi_phase_trajectories_v4.json"
    if phase_csv.exists() and traj_json.exists():
        print("\n=== V4.1: PI Stability Phase artifacts exist, skipping ===", flush=True)
        df = pd.read_csv(phase_csv)
        return {"n_points": len(df), "dz0_regime_counts": df[df["dead_zone"] == 0.0]["regime"].value_counts().to_dict()}

    print("\n=== V4.1: Dense (K_p, K_i) & Dead-Zone Stability Phase Sweep (4-shard) ===", flush=True)
    eta_w = 0.30
    kp_dense = [0.25, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 15.0, 25.0, 40.0]
    ki_ratio_dense = [0.0, 0.1, 0.2, 0.35, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0]

    kp_dz = [0.5, 1.0, 1.5, 2.5, 6.0, 15.0, 40.0]
    ki_ratio_dz = [0.0, 0.2, 0.35, 1.0, 2.5, 6.0, 10.0]

    grid_specs: List[Tuple[float, List[float], List[float]]] = [
        (0.0, kp_dense, ki_ratio_dense),
        (0.05, kp_dz, ki_ratio_dz),
        (0.14, kp_dz, ki_ratio_dz),
        (0.25, kp_dz, ki_ratio_dz),
    ]

    # Pre-download & verify BIC and KIRC sequentially before spawning workers
    for c in ["bic", "kirc"]:
        load_tcga_cohort(c, data_dir, top_features=2000)

    tasks: List[Tuple[str, float, float, float]] = []
    for cohort in ["bic", "kirc"]:
        for dz, kps, krs in grid_specs:
            for kp in kps:
                for kr in krs:
                    tasks.append((cohort, dz, kp, kr))

    n_workers = 4
    shards = [tasks[i::n_workers] for i in range(n_workers)]
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_v41_shard_worker, args=(rank, shards[rank], str(data_dir), str(out_dir)))
        for rank in range(n_workers)
    ]
    for p in procs:
        p.start()
    for rank, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"V4.1 shard {rank} failed with exitcode={p.exitcode}")

    shard_dfs = [pd.read_csv(out_dir / f"_v41_shard_{rank}.csv") for rank in range(n_workers)]
    df = pd.concat(shard_dfs, ignore_index=True)
    df = df.sort_values(["cohort", "dead_zone", "k_p", "kappa_i"]).reset_index(drop=True)
    df.to_csv(phase_csv, index=False)

    # Save 5 representative BIC trajectories
    views_bic, surv_bic, _, _ = load_tcga_cohort("bic", data_dir, top_features=2000)
    k_bic = CANONICAL_K["bic"]
    views_bic_corr = corrupt_view(views_bic, view_idx=0, noise_std=4.0, dropout_rate=0.60, seed=40)
    with _single_thread_ctx():
        ref_bic = CLVSSC(k_bic, seed=42).fit(views_bic)["labels"]

    rep_specs = [
        ("Convergent_Default", 0.0, 1.5, 0.35),
        ("Stalled_DeadZone", 0.25, 1.5, 0.35),
        ("UnderActuated_LowKp", 0.0, 0.25, 0.0),
        ("HighKi_Ringing", 0.0, 0.25, 10.0),
        ("Collapse_HighKp", 0.0, 40.0, 10.0),
    ]
    rep_trajs: Dict[str, object] = {}
    for save_key, dz, kp, kr in rep_specs:
        row, wh_c, wh_f, vh_f = _eval_v41_point(
            "bic", views_bic, views_bic_corr, surv_bic, ref_bic, k_bic, dz, kp, kr, eta_w=eta_w
        )
        rep_trajs[save_key] = {
            "cohort": "BIC",
            "dead_zone": dz,
            "k_p": kp,
            "k_i": round(kp * kr, 6),
            "kappa_i": kr,
            "tau": round(eta_w / kp, 6),
            "regime": row["regime"],
            "clean_weights_history": wh_c.tolist(),
            "fault_weights_history": wh_f.tolist(),
            "fault_lyapunov_history": vh_f.tolist(),
        }
    traj_json.write_text(json.dumps(rep_trajs, indent=2))

    for rank in range(n_workers):
        (out_dir / f"_v41_shard_{rank}.csv").unlink(missing_ok=True)

    counts = df[df["dead_zone"] == 0.0]["regime"].value_counts().to_dict()
    print(f"  Saved {phase_csv.name} ({len(df)} rows) | dz=0 regimes: {counts}", flush=True)
    return {"n_points": len(df), "dz0_regime_counts": counts}


def _make_expanded_cohort(
    views_base: List[np.ndarray],
    y_base: np.ndarray,
    target_n: int,
    seed: int = 123,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Stratified manifold-jittered expansion from GSE194122 (90,261 real cells) to N >= 150k."""
    rng = np.random.default_rng(seed + target_n)
    n_base = len(y_base)
    idx = rng.choice(n_base, size=target_n, replace=True)
    y_exp = y_base[idx]
    views_exp: List[np.ndarray] = []
    for arr in views_base:
        sub = arr[idx].astype(np.float32, copy=True)
        col_std = arr.std(axis=0, keepdims=True).astype(np.float32)
        col_std = np.where(col_std < 1e-6, 0.0, col_std)
        noise = rng.normal(0.0, 0.05, size=sub.shape).astype(np.float32) * col_std
        views_exp.append(sub + noise)
    return views_exp, y_exp


def run_v4_200k_scaling_and_healing(work_dir: Path, out_dir: Path) -> Dict[str, object]:
    """V4.2 (10k->300k scaling) and V4.3 (N=200k static + mid-stream step-fault self-healing)."""
    scale_csv = out_dir / "scale_200k_v4.csv"
    heal_csv = out_dir / "fault_self_healing_200k_v4.csv"
    heal_traj_json = out_dir / "fault_self_healing_200k_traj_v4.json"

    if scale_csv.exists() and heal_csv.exists() and heal_traj_json.exists():
        print("\n=== V4.2 & V4.3: 200k+ artifacts exist, skipping ===", flush=True)
        return {"skipped": True}

    torch.set_num_threads(4)
    sc_dir = work_dir / "sc_data"
    bmmc_path = download_and_extract_bmmc(sc_dir)
    ad_bmmc = ad.read_h5ad(bmmc_path)
    views_full, y_full, meta_full = extract_citeseq_views(ad_bmmc, max_cells=None, seed=101)
    n_real = meta_full["n_cells"]
    k_full = meta_full["n_classes"]
    print(f"\n=== Loaded full GSE194122 BMMC: n={n_real}, K={k_full}, dims={[v.shape[1] for v in views_full]} ===", flush=True)
    del ad_bmmc
    gc.collect()

    # V4.2: Ultra-Large-Scale Linear Complexity (10k -> 300k) with per-row flushing
    scale_summary: Dict[str, object] = {}
    if not scale_csv.exists():
        print("\n=== V4.2: Ultra-Large-Scale Linear Scaling (10k -> 300k) ===", flush=True)
        part_csv = out_dir / "_scale_200k_partial.csv"
        s_rows: List[Dict[str, object]] = []
        done_scale = set()
        if part_csv.exists() and part_csv.stat().st_size > 0:
            prev_s = pd.read_csv(part_csv)
            s_rows = prev_s.to_dict("records")
            for r in s_rows:
                done_scale.add((int(r["n_cells"]), str(r["method"])))

        rng_sub = np.random.default_rng(101)
        targets = [10000, 25000, 50000, n_real, 150000, 200000, 250000, 300000]
        for n_tgt in targets:
            if n_tgt < n_real:
                sel = np.sort(rng_sub.choice(n_real, size=n_tgt, replace=False))
                v_cur = [arr[sel] for arr in views_full]
                y_cur = y_full[sel]
                src_lbl = "GSE194122-Real"
            elif n_tgt == n_real:
                v_cur = views_full
                y_cur = y_full
                src_lbl = "GSE194122-Real"
            else:
                v_cur, y_cur = _make_expanded_cohort(views_full, y_full, target_n=n_tgt, seed=2026)
                src_lbl = "GSE194122-Expanded"

            for m_name, local_flag in [("CL-VSSC-Anchor (Smooth PI)", False), ("CL-VSSC-Anchor (Local PI)", True)]:
                if (int(len(y_cur)), m_name) in done_scale:
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                sync_cuda()
                t0 = time.perf_counter()
                with _single_thread_ctx():
                    out = CLVSSCAnchor(
                        k_full, n_anchors=384, anchor_k=20, max_iter=30, use_local_pi=local_flag, seed=42
                    ).fit(v_cur)
                sync_cuda()
                dt = time.perf_counter() - t0
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                with _single_thread_ctx():
                    met = evaluate_clustering(out["labels"], true_labels=y_cur)
                lyap = np.asarray(out["lyapunov_history"], dtype=np.float64)
                s_rows.append(
                    {
                        "n_cells": int(len(y_cur)),
                        "data_source": src_lbl,
                        "method": m_name,
                        "runtime_sec": round(float(dt), 4),
                        "peak_vram_mb": round(float(peak_mb), 1),
                        "gt_ari": round(float(met["ari"]), 5),
                        "gt_nmi": round(float(met["nmi"]), 5),
                        "gt_acc": round(float(met["acc"]), 5),
                        "lyap_init": round(float(lyap[0]), 5),
                        "lyap_final": round(float(lyap[-1]), 5),
                        "lyap_converged": bool(lyap[-1] < lyap[0]),
                    }
                )
                pd.DataFrame(s_rows).to_csv(part_csv, index=False)
                print(
                    f"  [n={len(y_cur):6d} | {m_name:28s}] time={dt:6.2f}s | VRAM={peak_mb:7.1f} MB | "
                    f"ARI={met['ari']:.3f} NMI={met['nmi']:.3f} | V:{lyap[0]:.3f}->{lyap[-1]:.3f}",
                    flush=True,
                )
            if n_tgt > n_real:
                del v_cur, y_cur
                gc.collect()

        sdf = pd.DataFrame(s_rows)
        sdf.to_csv(scale_csv, index=False)
        part_csv.unlink(missing_ok=True)

        for m_name in sdf["method"].unique():
            sub = sdf[sdf["method"] == m_name].sort_values("n_cells")
            lx = np.log(sub["n_cells"].to_numpy(dtype=np.float64))
            ly = np.log(sub["runtime_sec"].to_numpy(dtype=np.float64))
            slope, intercept = np.polyfit(lx, ly, 1)
            pred = slope * lx + intercept
            r2 = 1.0 - float(np.sum((ly - pred) ** 2) / max(1e-12, np.sum((ly - ly.mean()) ** 2)))
            lv = np.log(sub["peak_vram_mb"].to_numpy(dtype=np.float64))
            v_slope, _ = np.polyfit(lx, lv, 1)
            scale_summary[m_name] = {
                "runtime_power_alpha": round(float(slope), 4),
                "runtime_loglog_r2": round(float(r2), 4),
                "vram_power_alpha": round(float(v_slope), 4),
            }
        print(f"  Power-law fits: {scale_summary}", flush=True)

    # V4.3: N = 200,000 Closed-Loop Static & Mid-Stream Step-Fault Self-Healing
    if not (heal_csv.exists() and heal_traj_json.exists()):
        print("\n=== V4.3: N=200,000 Closed-Loop Fault Self-Healing Verification ===", flush=True)
        views_200k, y_200k = _make_expanded_cohort(views_full, y_full, target_n=200000, seed=2026)
        views_200k_static = corrupt_view(views_200k, view_idx=1, noise_std=3.0, dropout_rate=0.50, seed=230)

        configs = [
            ("CL-VSSC-Anchor (Smooth PI)", dict(use_feedback=True, use_null_snr=False, use_local_pi=False, dead_zone=0.0)),
            ("CL-VSSC-Anchor (Null-SNR PI)", dict(use_feedback=True, use_null_snr=True, use_local_pi=False, dead_zone=0.0)),
            ("CL-VSSC-Anchor (Local PI)", dict(use_feedback=True, use_null_snr=False, use_local_pi=True, dead_zone=0.0)),
            ("CL-VSSC-Anchor (DZ-PI)", dict(use_feedback=True, use_null_snr=False, use_local_pi=False, dead_zone=0.14)),
            ("CL-VSSC-Anchor (w/o FB)", dict(use_feedback=False, use_null_snr=False, use_local_pi=False, dead_zone=0.0)),
        ]

        part_heal = out_dir / "_heal_200k_partial.csv"
        h_rows: List[Dict[str, object]] = []
        h_trajs: Dict[str, object] = {}
        inject_iter = 10

        for scenario in ["StepFault_t10", "StaticFault_t0"]:
            for m_name, kw in configs:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                sync_cuda()
                t0 = time.perf_counter()
                with _single_thread_ctx():
                    model = CLVSSCAnchor(k_full, n_anchors=384, anchor_k=20, max_iter=30, seed=42, **kw)
                    if scenario == "StepFault_t10":
                        out = model.fit(
                            views_200k,
                            step_fault={"inject_iter": inject_iter, "view_idx": 1, "noise_std": 3.0, "dropout_rate": 0.50, "seed": 230},
                        )
                        onset = inject_iter
                    else:
                        out = model.fit(views_200k_static)
                        onset = 0
                sync_cuda()
                dt = time.perf_counter() - t0
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                with _single_thread_ctx():
                    met = evaluate_clustering(out["labels"], true_labels=y_200k)

                wh = np.asarray(out["weights_history"], dtype=np.float64)
                w_adt = wh[:, 1]
                wall = np.asarray(out["iter_wall_sec"], dtype=np.float64)
                lyap = np.asarray(out["lyapunov_history"], dtype=np.float64)

                w_pre = float(w_adt[onset])
                w_end = float(w_adt[-1])
                drop = w_pre - w_end

                if drop > 0.02:
                    target_90 = w_end + 0.10 * drop
                    heal_iters_90 = int(len(w_adt) - 1 - onset)
                    for idx_k in range(onset + 1, len(w_adt)):
                        if np.all(w_adt[idx_k:] <= target_90):
                            heal_iters_90 = int(idx_k - onset)
                            break
                    heal_wall_90 = float(wall[onset + heal_iters_90] - wall[onset])
                else:
                    heal_iters_90 = -1
                    heal_wall_90 = float("nan")

                below_25 = [idx_k for idx_k in range(onset + 1, len(w_adt)) if np.all(w_adt[idx_k:] <= 0.25)]
                heal_iters_25 = int(below_25[0] - onset) if below_25 else -1
                heal_wall_25 = float(wall[onset + heal_iters_25] - wall[onset]) if heal_iters_25 >= 0 else float("nan")

                h_rows.append(
                    {
                        "n_cells": 200000,
                        "scenario": scenario,
                        "method": m_name,
                        "w_adt_pre": round(w_pre, 5),
                        "w_adt_final": round(w_end, 5),
                        "heal_iters_90": heal_iters_90,
                        "heal_wall_sec_90": round(heal_wall_90, 4) if np.isfinite(heal_wall_90) else np.nan,
                        "heal_iters_lt025": heal_iters_25,
                        "heal_wall_sec_lt025": round(heal_wall_25, 4) if np.isfinite(heal_wall_25) else np.nan,
                        "gt_ari": round(float(met["ari"]), 5),
                        "gt_nmi": round(float(met["nmi"]), 5),
                        "gt_acc": round(float(met["acc"]), 5),
                        "runtime_sec": round(float(dt), 4),
                        "peak_vram_mb": round(float(peak_mb), 1),
                        "lyap_init": round(float(lyap[0]), 5),
                        "lyap_final": round(float(lyap[-1]), 5),
                    }
                )
                pd.DataFrame(h_rows).to_csv(part_heal, index=False)
                h_trajs[f"{scenario}__{m_name}"] = {
                    "w_adt_history": [round(float(x), 6) for x in w_adt],
                    "lyapunov_history": [round(float(x), 6) for x in lyap],
                    "iter_wall_sec": [round(float(x), 5) for x in wall],
                }
                print(
                    f"  [{scenario:14s} | {m_name:28s}] w_adt:{w_pre:.3f}->{w_end:.3f} | "
                    f"heal90={heal_iters_90}it ({heal_wall_90:.2f}s) | ARI={met['ari']:.3f} | {dt:.2f}s",
                    flush=True,
                )

        pd.DataFrame(h_rows).to_csv(heal_csv, index=False)
        part_heal.unlink(missing_ok=True)
        heal_traj_json.write_text(json.dumps(h_trajs, indent=2))
        del views_200k, views_200k_static
        gc.collect()

    return {"scale_summary": scale_summary}


def _query_nvidia_smi() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return "unavailable"


def main() -> None:
    t0 = time.time()
    work_dir = Path("/content/paper2_v4") if Path("/content").exists() else Path.cwd()
    data_dir = Path("/content/tcga_benchmark_data") if Path("/content").exists() else work_dir / "tcga_benchmark_data"
    out_dir = work_dir / "results_v4"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[V4 A100 Control Upgrades] Python {platform.python_version()} | Torch {torch.__version__} | {dev} ({gpu_name})", flush=True)

    v41_info = run_v4_pi_stability_phase(data_dir, out_dir)
    v42_info = run_v4_200k_scaling_and_healing(work_dir, out_dir)

    env = {
        "gpu_name": gpu_name,
        "nvidia_smi": _query_nvidia_smi(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "v4_elapsed_sec": round(time.time() - t0, 2),
        "v41_summary": v41_info,
        "v42_summary": v42_info,
    }
    (out_dir / "v4_env.json").write_text(json.dumps(env, indent=2))
    zip_path = Path(shutil.make_archive(str(work_dir / "cl_vssc_results_v4"), "zip", out_dir))
    print(f"V4_SUITE_COMPLETE: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB) in {env['v4_elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
