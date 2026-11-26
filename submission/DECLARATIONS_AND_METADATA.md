# ScholarOne / Editorial Manager Metadata & Declarations (Ready to Copy-Paste)

---

## 1. Manuscript Metadata

- **Full Title**:
  `Closed-Loop Proportional–Integral Feedback and Linear-Time Bipartite Spectral Control for Fault-Tolerant Multi-Modal Clustering`
- **Short Running Title** (`<= 50` characters):
  `Closed-Loop PI & Bipartite Spectral Control` *(43 chars)*
- **Article Type**: Regular Paper (or Original Research Article)
- **Target Primary Venue**: *IEEE Transactions on Cybernetics*
- **Alternative Venues**: *IEEE Transactions on Neural Networks and Learning Systems*, *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, *Information Fusion*
- **Manuscript Length**: 9 double-column `IEEEtran` pages (within standard 10–12 page regular-paper budget without over-length charges)
- **GitHub Public Repository**: `https://github.com/dashitongzhi/CL-VSSC`

---

## 2. Plain-Text Abstract (ScholarOne `<= 250` Words Compliant)

> Copy-paste verbatim into the submission portal Abstract box (`<= 250` words):

Multi-modal representation learning separates shared biological structure from nuisance variance before spectral clustering. Existing consensus graph learners and deep encoders mostly run in open loop: view weights lack partition feedback, eigengaps stay unregulated, and O(n^3) eigendecompositions fail on 10^5-sample cohorts. We formulate multi-modal clustering as a discrete-time closed-loop feedback system (CL-VSSC) paired with a linear-time bipartite landmark controller (CL-VSSC-Anchor, O(nm^2+ndr)). An orthogonal Stiefel plant couples with a dual Ky Fan eigengap controller and a damped simplex proportional-integral (PI) controller (with Null-SNR and double-centered Anchor-Local modes) under Input-to-State Stability (ISS) and Davis-Kahan bounds. Across 1,560 TCGA three-modality stress runs (sigma <= 4.0, up to 60% dropout), Smooth PI lowers faulty weights to 0.012 (vs. 0.190 MCGC, 0.264 Seurat-WNN, 0.319 Attention-MOAE; BH-FDR q=1.46e-11). Null-SNR calibration cuts leakage to 0.002 and raises mean fault ARI by +0.103 (0.484 to 0.587; 0.796 on mRNA/miRNA faults, q <= 6.19e-4). On the NeurIPS GSE194122 CITE-seq atlas (90,261 cells, 45 types), CL-VSSC-Anchor accelerates 10,000-cell A100 runtime 20.3x (4.60 s vs. 93.58 s), cuts VRAM 13.3x (603 vs. 8,002 MB), processes all 90,261 cells in 16.57 s (5,295 MB; 21.71 s Local PI on L4), and leads Protein-ADT fault ARI (sigma=3.0: 0.169 vs. 0.051 Attention-MOAE). Extended A100-80GB validation maps 536 (Kp, Ki, delta_dz) stability points, follows O(N^0.996) VRAM scaling to 300,000 cells (52.98-53.56 s), and achieves 7-iteration (3.82-6.03 s) step-fault recovery at N=200,000.

---

## 3. Keywords (`IEEE Index Terms`)

`Closed-loop feedback control, PI mirror descent, input-to-state stability, bipartite spectral clustering, fault tolerance, CITE-seq.`

---

## 4. Standard Compliance Declarations (Copy-Paste Ready)

- **Data and Code Availability Statement**:
  > All experiments rely on publicly available de-identified benchmark datasets: the 10-cohort TCGA multi-omics cancer benchmark (Rappoport & Shamir, *Nucleic Acids Research*, 2018) and the NeurIPS Multimodal Single-Cell CITE-seq bone marrow atlas (NCBI GEO accession `GSE194122`; Luecken et al., *NeurIPS Datasets and Benchmarks*, 2021). Complete Python/PyTorch solver code, Google Colab A100/L4 reproduction notebooks, figure/table generators, and verified experimental result archives are publicly available under the MIT license at `https://github.com/dashitongzhi/CL-VSSC`.
- **Declaration of Competing Interest**:
  > The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
- **Ethics & Consent Statement**:
  > This study performs secondary methodological analysis exclusively on publicly available, de-identified cancer genomics (TCGA) and single-cell sequencing (`GSE194122`) datasets. No new human participants, clinical trials, or animal experiments were conducted; institutional ethical approval and informed consent were obtained by the original dataset consortia.
- **Funding Statement**:
  > `[TODO(PERSONAL_INFO): Insert grant agency and grant number, OR state "This research received no specific external grant from funding agencies in the public, commercial, or not-for-profit sectors."]`
- **CRediT Authorship Contribution Statement** *(for multi-author or Elsevier backup)*:
  > **[TODO(PERSONAL_INFO): Author 1 Name]**: Conceptualization, Methodology, Software, Validation, Formal analysis, Writing – original draft, Visualization. *(Add co-authors' Supervision / Funding acquisition / Writing – review & editing roles if applicable.)*

---

## 5. Elsevier *Information Fusion* Backup Kit (`Highlights <= 85` Characters Each)

1. Closed-loop PI and dual eigengap control regulates multi-modal spectral clustering.
2. Damped PI mirror descent gives ISS and Davis-Kahan bounds without dead-zone stall.
3. Bipartite Gram decomposition achieves exact O(nm^2) equivalence on 90k-300k cells.
4. Smooth and Null-SNR PI cut severe-fault leakage to 0.012 and 0.002 (q=1.46e-11).
5. 536-point gain phase map and 200k-cell 7-step recovery verify stability margins.
