# Cover Letter Template — IEEE Transactions Submission

> **Target Journal**: *IEEE Transactions on Cybernetics*  
> *(Note: If submitting to IEEE TNNLS or IEEE TSMC: Systems instead, replace the journal name in the salutation, first paragraph, and `manuscript/main.tex` `\markboth` line.)*  
> **Instructions**: Replace every bracketed `[TODO(PERSONAL_INFO): ...]` field below (or run `src/apply_author_info.py --profile submission/author_profile.json --build`), then upload `Cover_Letter_IEEE_TCYB.pdf` / `Cover_Letter_IEEE_TNNLS.pdf` alongside `CL-VSSC_Supplementary_IEEE.pdf`.

---

[TODO(PERSONAL_INFO): Date, e.g., November 26, 2026]

To the Editor-in-Chief and Editorial Board  
*IEEE Transactions on Cybernetics*

Dear Editor-in-Chief and Associate Editors,

We submit our manuscript entitled **"Closed-Loop Proportional--Integral Feedback and Linear-Time Bipartite Spectral Control for Fault-Tolerant Multi-Modal Clustering"** for consideration as a Regular Paper in *IEEE Transactions on Cybernetics*.

Existing multi-view consensus graph learners and deep multimodal encoders largely operate in open loop: modality weights lack closed-loop partition feedback, normalized Laplacian eigengaps remain unregulated, and dense $\mathcal{O}(n^3)$ eigendecompositions fail to scale to $10^5$-cell single-cell cohorts. Guided by online feedback optimization (OFO) and classical control stability theory, we formulate fault-tolerant multi-modal clustering as a discrete-time two-loop feedback control system (**CL-VSSC**) coupled with a linear-time bipartite landmark extension (**CL-VSSC-Anchor**). This alignment with autonomous feedback systems, large-scale cybernetic graph control, and guaranteed stability boundaries makes *IEEE Transactions on Cybernetics* an ideal home for this work.

### Summary of Main Contributions
1. **Closed-loop dual spectral--PI architecture**: Couples a Stiefel-manifold orthogonal variable-separation plant with an inner dual integral Ky Fan trace controller ($\mu$) that widens the clustering eigengap and an outer damped proportional--integral (PI) mirror-descent controller on the simplex (with optional column-shuffled Null-SNR calibration and double-centered Anchor-Local PI).
2. **Formal control and complexity guarantees**: Proves exact $\mathcal{O}(nm^2)$ Gram eigendecomposition equivalence without approximation error (Proposition 1), BIBO anti-windup bounds and $\exp(-(1+\kappa_I)\Delta e/\tau)$ exponential fault rejection with dead-zone stall analysis (Proposition 2), and local Input-to-State Stability (ISS) alongside a Davis--Kahan $\sin\Theta$ spectral perturbation bound (Proposition 3).
3. **Statistically verified multi-assay fault rejection and 90k--300k scaling**: Across 1,560 three-modality TCGA stress runs ($\sigma$ up to $4.0$, $60\%$ dropout), Smooth PI reduces faulty channel weights to $0.012$ and Null-SNR PI to $0.002$ ($36/36$ wins, paired Wilcoxon BH-FDR $q = 1.46 \times 10^{-11}$ vs.\ every weighted competitor), reaching $0.796$ ARI on mRNA/miRNA faults ($q \le 6.19 \times 10^{-4}$ vs.\ all 9 baselines). On the NeurIPS `GSE194122` CITE-seq atlas ($90{,}261$ cells), CL-VSSC-Anchor accelerates $10{,}000$-cell A100 runtime $20.3\times$, reduces VRAM $13.3\times$, and leads severe Protein-ADT fault robustness ($\sigma=3.0$).
4. **Classical gain-margin phase mapping and step-fault self-healing**: Maps 536 $(K_p, K_i, \delta_{\mathrm{dz}})$ stability points on A100-80GB, verifies $\mathcal{O}(N^{0.996})$ peak VRAM scaling up to $300{,}000$ cells ($<54\text{ s}$), and achieves $7$-iteration ($3.82\text{--}6.03\text{ s}$) mid-stream step-fault recovery at $N=200{,}000$ cells.

### Originality and IEEE PSPB Compliance
This manuscript is original, has not been published previously, and is not under concurrent review elsewhere in whole or in part. All authors have approved the manuscript and agree with this submission. A dedicated *Supplementary Material* PDF (`CL-VSSC_Supplementary_IEEE.pdf`, providing expanded step-by-step proofs, per-cohort 10-TCGA tables, and complete Wilcoxon/BH-FDR ledgers) and a Graphical Abstract accompany this submission.

### Reproducibility & Conflict of Interest
All source code, Colab A100/L4/80GB notebooks, figure/table generation scripts, and benchmark verification suites are publicly released under the MIT license at `https://github.com/dashitongzhi/CL-VSSC`. The authors declare no competing financial or non-financial interests.

Thank you for your time and consideration.

Sincerely,

**[TODO(PERSONAL_INFO): Corresponding Author Full Name]**  
[TODO(PERSONAL_INFO): Title, Department / School, University / Institution]  
[TODO(PERSONAL_INFO): City, Postal Code, Country]  
E-mail: `[TODO(PERSONAL_INFO): author@institution.edu]`  
ORCID: `[TODO(PERSONAL_INFO): 0000-0000-0000-0000]`
