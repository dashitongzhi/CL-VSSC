"""Local CPU smoke test for CL-VSSC v2: Dead-Zone PI, Partial Masks, Anchor Spectral, and Deep/Multimodal Baselines."""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_vssc import CLVSSC, CLVSSCAnchor  # noqa: E402
from baselines import (  # noqa: E402
    corrupt_view,
    evaluate_clustering,
    permutation_logrank_p,
    run_attention_moae,
    run_concat_spectral,
    run_lmvsc,
    run_mcgc,
    run_nemo,
    run_openloop_mavs,
    run_snf,
    run_subtype_gan,
    run_wnn,
)
from lifelines.statistics import multivariate_logrank_test  # noqa: E402


def make_synthetic_multiomics(
    n: int = 160,
    n_clusters: int = 4,
    dims: tuple[int, int, int] = (120, 100, 90),
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_clusters), n // n_clusters)
    rng.shuffle(labels)

    d0, d1, d2 = dims
    z0_axis = np.where(labels < 2, -1.6, 1.6)[:, None] + rng.normal(0.0, 0.55, size=(n, 1))
    z1_axis = np.where((labels % 2) == 0, -1.6, 1.6)[:, None] + rng.normal(0.0, 0.55, size=(n, 1))
    Z_shared = np.hstack([z0_axis, z1_axis, rng.normal(0.0, 0.4, size=(n, 4))])

    nuisance_0 = rng.normal(0.0, 1.8, size=(n, 4))
    nuisance_1 = rng.normal(0.0, 1.8, size=(n, 4))
    nuisance_2 = rng.normal(0.0, 1.8, size=(n, 4))

    X0 = z0_axis @ rng.normal(0.0, 1.2, size=(1, d0)) + nuisance_0 @ rng.normal(0.0, 0.9, size=(4, d0)) + rng.normal(0.0, 0.7, size=(n, d0))
    X1 = z1_axis @ rng.normal(0.0, 1.2, size=(1, d1)) + nuisance_1 @ rng.normal(0.0, 0.9, size=(4, d1)) + rng.normal(0.0, 0.7, size=(n, d1))
    X2 = Z_shared @ rng.normal(0.0, 0.8, size=(6, d2)) + nuisance_2 @ rng.normal(0.0, 0.9, size=(4, d2)) + rng.normal(0.0, 0.7, size=(n, d2))

    base_haz = np.array([1.0, 1.6, 2.5, 3.8])[labels]
    times = rng.exponential(1000.0 / base_haz)
    events = (rng.random(n) < 0.7).astype(int)
    surv = pd.DataFrame({"time": times, "event": events})
    return [X0, X1, X2], labels, surv


def main():
    # 1. Dead-zone PI test on clean vs corrupted data + all v2 baselines
    clean_views, y_true, surv = make_synthetic_multiomics(n=180, dims=(600, 500, 120), seed=99)
    res_clean_dz = CLVSSC(n_clusters=4, dead_zone=0.14, seed=42).fit(clean_views)
    w_clean = res_clean_dz["final_weights"]
    print(f"Dead-Zone PI clean weights (600,500,120): {np.round(w_clean, 4)} (spread={w_clean.max() - w_clean.min():.4f})")
    assert (w_clean.max() - w_clean.min()) < 0.08, f"Clean DZ weights drifted too far from uniform: {w_clean}"

    corrupted_views = corrupt_view(clean_views, view_idx=2, noise_std=4.0, dropout_rate=0.60, seed=7)
    res_corr_dz = CLVSSC(n_clusters=4, dead_zone=0.14, seed=42).fit(corrupted_views)
    w_corr = res_corr_dz["final_weights"]
    print(f"Dead-Zone PI corrupted view2 (sigma=4, 60% drop): {np.round(w_corr, 4)} | V: {res_corr_dz['lyapunov_history'][0]:.4f}->{res_corr_dz['lyapunov_history'][-1]:.4f}")
    assert w_corr[2] < 0.03, f"Corrupted view 2 not suppressed enough by Dead-Zone PI: {w_corr}"

    # Verify all new baselines on clean_views
    baselines = {
        "CL-VSSC (DZ-PI)": res_corr_dz,
        "CL-VSSC (No-DZ PI)": CLVSSC(n_clusters=4, dead_zone=0.0, seed=42).fit(corrupted_views),
        "OpenLoop_MAVS": run_openloop_mavs(corrupted_views, n_clusters=4, seed=42),
        "SNF": run_snf(corrupted_views, n_clusters=4, seed=42),
        "NEMO": run_nemo(corrupted_views, n_clusters=4, seed=42),
        "MCGC": run_mcgc(corrupted_views, n_clusters=4, seed=42),
        "Subtype-GAN": run_subtype_gan(corrupted_views, n_clusters=4, epochs=35, seed=42),
        "Attention-MOAE": run_attention_moae(corrupted_views, n_clusters=4, epochs=35, seed=42),
        "Seurat-WNN": run_wnn(corrupted_views, n_clusters=4, seed=42),
        "LMVSC": run_lmvsc(corrupted_views, n_clusters=4, n_anchors=48, seed=42),
        "ConcatSpectral": run_concat_spectral(corrupted_views, n_clusters=4, seed=42),
    }
    rows = []
    for name, out in baselines.items():
        m = evaluate_clustering(out["labels"], true_labels=y_true, survival_df=surv, embedding=out.get("F"))
        rows.append({"method": name, "ari": round(m["ari"], 3), "nmi": round(m["nmi"], 3), "acc": round(m["acc"], 3), "neg_log10_p": round(m["neg_log10_p"], 2)})
    print("\nBaselines smoke comparison (corrupted view 2):")
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. Partial-modality masking test (30% missing)
    rng = np.random.default_rng(12)
    n = len(y_true)
    masks = [rng.random(n) > 0.3 for _ in range(3)]
    for i in range(n):
        if not any(m[i] for m in masks):
            masks[rng.integers(0, 3)][i] = True
    views_part = [np.array(v, copy=True) for v in clean_views]
    for v_i in range(3):
        views_part[v_i][~masks[v_i]] = np.nan
    res_part_cl = CLVSSC(n_clusters=4, dead_zone=0.14, seed=42).fit(views_part, masks=masks)
    res_part_nemo = run_nemo(views_part, n_clusters=4, masks=masks, seed=42)
    m_cl = evaluate_clustering(res_part_cl["labels"], true_labels=y_true)
    m_nm = evaluate_clustering(res_part_nemo["labels"], true_labels=y_true)
    print(f"\nPartial-view (30% missing) -> CL-VSSC ARI={m_cl['ari']:.3f} ACC={m_cl['acc']:.3f} | NEMO ARI={m_nm['ari']:.3f} ACC={m_nm['acc']:.3f}")

    # 3. Large-scale CLVSSCAnchor test on n=4,000 synthetic single cells
    sc_views, sc_y, _ = make_synthetic_multiomics(n=4000, n_clusters=4, dims=(400, 150, 80), seed=77)
    sc_corr = corrupt_view(sc_views, view_idx=2, noise_std=4.0, dropout_rate=0.60, seed=5)
    t0 = time.time()
    res_anc = CLVSSCAnchor(n_clusters=4, n_anchors=192, dead_zone=0.0, seed=42).fit(sc_corr)
    dt = time.time() - t0
    m_anc = evaluate_clustering(res_anc["labels"], true_labels=sc_y, embedding=res_anc["F"])
    m_lmv = evaluate_clustering(run_lmvsc(sc_corr, n_clusters=4, n_anchors=192, seed=42)["labels"], true_labels=sc_y)
    print(
        f"\nCLVSSCAnchor (n=4000, CPU) in {dt:.2f}s | V:{res_anc['lyapunov_history'][0]:.4f}->{res_anc['lyapunov_history'][-1]:.4f} | "
        f"weights={np.round(res_anc['final_weights'], 4)} | ARI={m_anc['ari']:.3f} ACC={m_anc['acc']:.3f} (vs LMVSC ARI={m_lmv['ari']:.3f})"
    )
    assert res_anc["lyapunov_history"][-1] < res_anc["lyapunov_history"][0]
    assert res_anc["final_weights"][2] < 0.03, f"Anchor failed to suppress corrupted view: {res_anc['final_weights']}"
    assert m_anc["ari"] > 0.65, f"Anchor ARI unexpectedly low: {m_anc['ari']}"

    # 4. v3 Null-SNR & Anchor-Local PI checks
    res_null_clean = CLVSSC(n_clusters=4, use_null_snr=True, seed=42).fit(clean_views)
    res_null_corr = CLVSSC(n_clusters=4, use_null_snr=True, seed=42).fit(corrupted_views)
    m_nc = evaluate_clustering(res_null_corr["labels"], true_labels=y_true)
    print(f"\nNull-SNR clean weights: {np.round(res_null_clean['final_weights'], 4)} | corr weights: {np.round(res_null_corr['final_weights'], 4)} | corr ARI={m_nc['ari']:.3f}")
    assert res_null_corr["final_weights"][2] < 0.025, f"Null-SNR failed to suppress view 2: {res_null_corr['final_weights']}"

    res_loc = CLVSSCAnchor(n_clusters=4, n_anchors=192, use_local_pi=True, use_null_snr=True, seed=42).fit(sc_corr)
    m_loc = evaluate_clustering(res_loc["labels"], true_labels=sc_y)
    assert res_loc["local_weights"].shape == (192, 3)
    assert res_loc["final_weights"][2] < 0.03, f"Anchor-Local PI failed to suppress view 2: {res_loc['final_weights']}"
    assert m_loc["ari"] > 0.65, f"Anchor-Local PI ARI unexpectedly low: {m_loc['ari']}"
    print(f"Anchor-Local PI + Null-SNR (n=4000) | weights={np.round(res_loc['final_weights'], 4)} | ARI={m_loc['ari']:.3f} ACC={m_loc['acc']:.3f}")

    lr = multivariate_logrank_test(surv["time"], res_clean_dz["labels"], surv["event"])
    t_obs, _ = permutation_logrank_p(surv["time"].to_numpy(), surv["event"].to_numpy(), res_clean_dz["labels"], n_perm=500, seed=42)
    assert abs(t_obs - lr.test_statistic) < 1e-6
    print("ALL_V3_SMOKE_TESTS_PASSED")


if __name__ == "__main__":
    main()
