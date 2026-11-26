# Paper 2 — CL-VSSC: Closed-Loop Proportional–Integral Feedback & Linear-Time Bipartite Spectral Control

Standalone 9-page IEEE Transactions manuscript (100% self-contained control-theoretic & systems paper), complete A100 v2 + L4 v3 + A100-80GB v4 experimental bundles, Colab benchmark notebook, figure/table generator, and Chinese mastery / SJTU 816 defense guide.

- **Target Venues**: *IEEE Transactions on Cybernetics (TCYB)*, *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)*, *IEEE TSMC: Systems*, or *Information Fusion*
- **Public GitHub Repository**: [https://github.com/dashitongzhi/CL-VSSC](https://github.com/dashitongzhi/CL-VSSC)
- **Compiled Manuscript (9 pages) & Supplementary**: [CL-VSSC_ClosedLoop_Control_IEEE.pdf](CL-VSSC_ClosedLoop_Control_IEEE.pdf) (SHA-256 identical to [manuscript/main.pdf](manuscript/main.pdf)), [CL-VSSC_Supplementary_IEEE.pdf](CL-VSSC_Supplementary_IEEE.pdf) (Tables S-I–S-IV & expanded proofs)
- **TeX Source & References (31 verified entries)**: [manuscript/main.tex](manuscript/main.tex), [manuscript/supplementary.tex](manuscript/supplementary.tex), [manuscript/references.bib](manuscript/references.bib)
- **Turnkey Submission Kit**: [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md), Graphical Abstract ([PDF](submission/Graphical_Abstract.pdf) / [PNG](submission/Graphical_Abstract.png)), PDF Cover Letters ([IEEE TCYB](submission/Cover_Letter_IEEE_TCYB.pdf), [IEEE TNNLS](submission/Cover_Letter_IEEE_TNNLS.pdf), [Information Fusion](submission/Cover_Letter_Information_Fusion.pdf)), [Elsevier Highlights](submission/Highlights_Elsevier.pdf), [DECLARATIONS_AND_METADATA.md](submission/DECLARATIONS_AND_METADATA.md), and 1-command stamper [`src/apply_author_info.py`](src/apply_author_info.py)
- **Control & SJTU 816 Chinese Study Guide**: [STUDY_GUIDE_CN.md](STUDY_GUIDE_CN.md)

---

## Folder Contents

- `CL-VSSC_ClosedLoop_Control_IEEE.pdf`: Publication-ready 9-page `IEEEtran` manuscript (7 main display equations + 3 formal propositions with proofs in Appendix A, 6 figures, 3 tables).
- `manuscript/`: Self-contained LaTeX source (`main.tex`, `references.bib`, generated `figures/p2_fig1..6.pdf`, `tables/p2_tab1..3.tex`).
- `results_v2/`: `9` verified NVIDIA A100-SXM4-40GB CSV/JSON files (`tcga_benchmark_v2.csv`, `fault_tolerance_v2.csv`, `diagnostics_trajectories_v2.json`, `singlecell_benchmark_v2.csv`, `sc_fault_v2.csv`, `scalability_v2.csv`, etc.).
- `results_v3/`: `11` verified NVIDIA L4 CSV/JSON files (`tcga_null_snr_v3.csv`, `fault_null_snr_v3.csv`, `sc_local_pi_v3.csv`, `sc_fault_local_pi_v3.csv`, `sc_scale_local_pi_v3.csv`, `sensitivity_{pi_grid,mu,anchors}_v3.csv`, etc.).
- `results_v4/`: `7` verified NVIDIA A100-SXM4-80GB CSV/JSON files (`pi_stability_phase_v4.csv`, `pi_phase_trajectories_v4.json`, `scale_200k_v4.csv`, `fault_self_healing_200k_v4.csv`, `fault_self_healing_200k_traj_v4.json`, `v4_env.json`, `paper1_boundary_audit.json`).
- `colab_cl_vssc_benchmark.ipynb`: Self-contained 7-cell Colab A100/L4 reproduction notebook (automatically mounts Google Drive in Colab and syncs `results_v2/`, `results_v3/`, `.npz` weights/embeddings, zip bundles, and run manifests into `MyDrive/Paper/Paper2_CL-VSSC_ClosedLoop_Control/`).
- `src/`: Closed-loop solvers (`cl_vssc.py`), baselines (`baselines.py`), TCGA/CITE-seq benchmark runners (`run_v2_tcga_suite.py`, `run_v2_singlecell_suite.py`, `run_v3_upgrades.py`, `run_v4_a100_control_upgrades.py`), figure/table generator (`make_assets.py`), and consistency checker (`verify_paper.py`).
- `tests/smoke_test.py`: Offline regression verification suite.

---

## Core Experimental Highlights (`results_v2/` + `results_v3/`)

1. **Closed-Loop Controller Ablations & Lyapunov Monitor (`Table I`, `Fig. 2`)**:
   - `Smooth PI` ($\delta_{\mathrm{dz}}=0$) improves clean stratification (`5/10` cohorts, median `1.248`, Silhouette `0.509`) and fault rejection (`0.012`) over `DZ-PI` (`4/10`, median `0.845`, `0.024`) and eliminates `P-only` steady-state offset (`BIC mRNA 0.011` vs `0.031`) and `w/o Feedback` leakage (`0.333`).
   - $\mathcal{V}(t)$ decreases monotonically in `9/10` clean TCGA cohorts (`OV` flat at `+0.004`), and dual warmup release opens BIC eigengap `6.70x` (`0.0082 -> 0.0550`).
   - `Null-SNR PI` highlights the robustness-nominal Pareto trade-off: boosts clean Silhouette (`0.516`) and extreme fault rejection, at `3/10` clean survival significance (`0.882` median) due to attenuated low-dimensional miRNA weights.
2. **Three-Modality Sensor-Fault Tolerance (`1,560` Stress Runs, `Table II`, `Fig. 3`)**:
   - Under $\sigma=4.0$ ($60\%$ dropout), `Smooth PI` holds mean corrupted weight to **`0.012`** (vs `0.190` MCGC, `0.264` Seurat-WNN, `0.319` Attention-MOAE; `36/36` wins, one-sided paired Wilcoxon BH-FDR `q = 1.46e-11` vs. every weighted baseline). `Null-SNR PI` tightens suppression nearly `7x` to **`0.002`** (`0.001 / 0.003 / 0.001`) and lifts fault retention ARI `+0.103` (`0.484 -> 0.587`, leading all `10` configurations across $\sigma=2.0$ and $\sigma=4.0$; BH-FDR `q <= 0.0196` vs. `8/9` baselines at $\sigma=4.0$, and `0.796` ARI / `q <= 6.19e-4` vs. all `9/9` competitors on mRNA+miRNA faults).
3. **90,261-Cell NeurIPS `GSE194122` CITE-seq & Linear Scaling (`Table III`, `Fig. 4`)**:
   - **Gram Equivalence (Prop. 1)**: Cuts 10k-cell A100 runtime **`20.3x`** (`4.60 s` vs `93.58 s` Dense) and VRAM **`13.3x`** (`603 MB` vs `8,002 MB`), processing all **`90,261` cells** in `16.57 s` (`5,295 MB`).
   - **Double-Centered Anchor-Local PI**: Eliminates global integral drift and boosts clean BMMC ARI in `9/12` paired runs (`p = 0.032` Wilcoxon; `BMMC-10k` ARI `0.201 -> 0.218`, ACC `0.308 -> 0.316`; `BMMC-30k` ARI `0.212 -> 0.222`) at `21.71 s` (`5,434 MB` L4 VRAM, `+2.6%` overhead).
   - **Protein ADT Fault Leadership**: At $\sigma=3.0$, `50%` dropout on `BMMC-10k`, `CL-VSSC-Anchor` leads all methods (`ARI=0.169`, `ACC=0.282`, `w_ADT=0.317`; `Null-SNR 0.294`, `Local+Null 0.299`, winning `6/6` paired runs across $\sigma\in\{1.5,3.0\}$ at `p=0.0156`), while `Attention-MOAE` locks onto ADT (`0.866`) and collapses `-78.9%` (`0.051`).
4. **Control-Gain Pareto Sweeps (`Fig. 5`)**: Verifies Proposition 2–3 across $(\kappa_I \times \tau)$ (`fault ARI` up to `0.597` BIC / `0.990` KIRC), monotonic Silhouette (`BIC 0.4386->0.4765`) across $\mu_{\max}\in[0,0.24]$, and linear scaling across $m\in[128,768]$.
5. **A100-80GB PI Stability Phase Diagram & 200k–300k Fault Self-Healing (`results_v4/`, `Fig. 6`)**:
   - **536-Point $(K_p, K_i, \delta_{\mathrm{dz}})$ Phase Map**: Separates `Under-Actuated`, `Convergent` (`30/121` BIC, `15/121` KIRC; default $K_p=1.5,\kappa_I=0.35$ settles in `4–5` steps), `Oscillatory`, and `Collapse / Unstable` ($\Delta\mathcal{V}>0$ or single-view lock-in), showing $\delta_{\mathrm{dz}} \in [0, 0.25]$ shrinks Collapse (`56 -> 31 -> 1 -> 0` over the common 98-point grid) while widening Under-Actuated stalling (`13 -> 21`).
   - **10k–300k Linear Scaling & $N=200\text{k}$ Step-Fault Self-Healing**: Scales from real `GSE194122` (`90,261`) to `GSE194122-Expanded` (`300,000` cells in `53.56 s` Smooth PI / `52.98 s` Local PI, `17.55–18.04 GB` VRAM, $\mathcal{O}(N^{0.9955})$ VRAM); at $N=200,000$, recovers $90\%$ from a mid-stream $t=10$ ADT step fault within `7` iterations (`3.82 s` Smooth PI, `5.20 s` Null-SNR ARI `0.195`, `6.03 s` Local PI ARI `0.203`), while `DZ-PI` stalls at `0.500` (`0.136` ARI).

---

## Local Build & Verification

Run inside this folder using the shared `.venv` site-packages with Python 3.12 (or any activated env with the project dependencies):

```sh
PYTHONPATH=/Users/kral/Documents/Research/MAVS-Reliability-2026-11-05/.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3.12 -B src/make_assets.py
cd manuscript && tectonic --keep-logs main.tex && cp main.pdf ../CL-VSSC_ClosedLoop_Control_IEEE.pdf && cd ..
PYTHONPATH=/Users/kral/Documents/Research/MAVS-Reliability-2026-11-05/.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3.12 -B tests/smoke_test.py
PYTHONPATH=/Users/kral/Documents/Research/MAVS-Reliability-2026-11-05/.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3.12 -B src/verify_paper.py
```

---

## Publication & Cross-Citation Ethics

- Strictly separated from `Paper1_MIVS-CL_Incomplete_MultiOmics` in problem formulation, figures, tables, theorems, and prose (100% independent control formulation)—never duplicate plots across submissions or submit overlapping manuscripts simultaneously.
- If submitting sequentially or hosting public repositories alongside peer review, explicitly disclose and cite any accepted/preprint companion work accurately rather than omitting related outputs.
