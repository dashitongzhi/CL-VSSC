"""Incremental completion of Paper 2 Null-SNR PI fault stress at intermediate severities sigma in {0.5, 1.0}."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = str(Path(__file__).resolve().parent) if "__file__" in globals() else str(Path.cwd())
for _cand in [str(Path.cwd() / "src"), _THIS_DIR]:
    if _cand in sys.path:
        sys.path.remove(_cand)
    sys.path.insert(0, _cand)

from baselines import corrupt_view, evaluate_clustering  # noqa: E402
from cl_vssc import CLVSSC  # noqa: E402
from tcga_loader import CANONICAL_K, load_tcga_cohort  # noqa: E402


def main() -> None:
    work_dir = Path("/content") if Path("/content").exists() else Path(__file__).resolve().parents[1]
    data_dir = work_dir / "tcga_benchmark_data"
    out_dir = work_dir / "results_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    stress_cohorts = ["bic", "kirc", "lihc", "gbm"]
    loaded = {c: load_tcga_cohort(c, data_dir) for c in stress_cohorts}
    mod_names = {0: "mRNA", 1: "Methylation", 2: "miRNA"}
    rows = []

    for cohort in stress_cohorts:
        clean_views, surv_df, _, _ = loaded[cohort]
        k_c = CANONICAL_K[cohort]
        ref_lbl = CLVSSC(k_c, use_null_snr=True, seed=42).fit(clean_views)["labels"]
        for v_idx, mod_lbl in mod_names.items():
            for sigma, drop in [(0.5, 0.15), (1.0, 0.30)]:
                for s_seed in [0, 1, 2]:
                    v_c = corrupt_view(clean_views, view_idx=v_idx, noise_std=sigma, dropout_rate=drop, seed=100 * v_idx + int(sigma * 10) + s_seed)
                    res = CLVSSC(k_c, use_null_snr=True, seed=42).fit(v_c)
                    met = evaluate_clustering(res["labels"], true_labels=ref_lbl, survival_df=surv_df, compute_perm=False)
                    rows.append({
                        "cohort": cohort.upper(),
                        "view_idx": v_idx,
                        "modality": mod_lbl,
                        "noise_sigma": sigma,
                        "dropout": drop,
                        "seed": s_seed,
                        "method": "CL-VSSC (Null-SNR PI)",
                        "clean_retention_ari": met["ari"],
                        "clean_retention_nmi": met["nmi"],
                        "neg_log10_p": met.get("neg_log10_p", 0.0),
                        "w_corrupted": float(res["final_weights"][v_idx]),
                    })
                    print(f"  [Null-SNR {cohort.upper()} {mod_lbl} sig={sigma} s={s_seed}] ARI={met['ari']:.3f}, w={res['final_weights'][v_idx]:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fault_null_snr_mid_v3.csv", index=False)
    print(f"[P2_MID_COMPLETE] Saved {len(df)} rows to fault_null_snr_mid_v3.csv", flush=True)


if __name__ == "__main__":
    main()
