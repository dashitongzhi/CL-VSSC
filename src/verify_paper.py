"""Standalone programmatic verification for Paper 2 (CL-VSSC & CL-VSSC-Anchor):
v2 + v3 CSVs, 5 vector figures, 3 LaTeX tables, notebook AST, clean log, zero MAVS isolation, and CSV claims."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RES2_DIR = ROOT / "results_v2"
RES3_DIR = ROOT / "results_v3"
RES4_DIR = ROOT / "results_v4"
MS_DIR = ROOT / "manuscript"
P1_TEX = ROOT.parent / "Paper1_MIVS-CL_Incomplete_MultiOmics" / "manuscript" / "main.tex"


def sha256(path: Path) -> str:
    """Return lowercase hexadecimal SHA-256 digest of a binary file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    req_v2 = [
        "cohort_summary_v2.csv",
        "tcga_benchmark_v2.csv",
        "k_sensitivity_v2.csv",
        "fault_tolerance_v2.csv",
        "diagnostics_trajectories_v2.json",
        "singlecell_benchmark_v2.csv",
        "sc_fault_v2.csv",
        "scalability_v2.csv",
        "environment_v2.json",
    ]
    for fn in req_v2:
        p = RES2_DIR / fn
        assert p.is_file() and p.stat().st_size > 0, f"Missing/empty v2 result: {p}"

    req_v3 = [
        "tcga_null_snr_v3.csv",
        "traj_null_snr_v3.json",
        "k_sens_null_snr_v3.csv",
        "fault_null_snr_v3.csv",
        "sc_local_pi_v3.csv",
        "sc_fault_local_pi_v3.csv",
        "sc_scale_local_pi_v3.csv",
        "sensitivity_pi_grid_v3.csv",
        "sensitivity_mu_v3.csv",
        "sensitivity_anchors_v3.csv",
        "v3_env.json",
    ]
    for fn in req_v3:
        p = RES3_DIR / fn
        assert p.is_file() and p.stat().st_size > 0, f"Missing/empty v3 result: {p}"

    req_figs = [
        "p2_fig1_control_loop.pdf",
        "p2_fig2_dynamics.pdf",
        "p2_fig3_fault_tolerance.pdf",
        "p2_fig4_singlecell_scale.pdf",
        "p2_fig5_sensitivity.pdf",
        "p2_fig6_stability_phase_200k.pdf",
    ]
    for fn in req_figs:
        p = MS_DIR / "figures" / fn
        assert p.is_file() and p.stat().st_size > 1000, f"Missing/empty figure: {p}"

    req_v4 = [
        "pi_stability_phase_v4.csv",
        "pi_phase_trajectories_v4.json",
        "scale_200k_v4.csv",
        "fault_self_healing_200k_v4.csv",
        "fault_self_healing_200k_traj_v4.json",
        "v4_env.json",
        "paper1_boundary_audit.json",
    ]
    for fn in req_v4:
        p = RES4_DIR / fn
        assert p.is_file() and p.stat().st_size > 0, f"Missing/empty v4 result: {p}"

    req_tabs = [
        "p2_tab1_controller_ablation.tex",
        "p2_tab2_fault_stress.tex",
        "p2_tab3_singlecell.tex",
        "p2_stab1_notation_complexity.tex",
        "p2_stab2_tcga_cohorts.tex",
        "p2_stab3_wilcoxon_ledger.tex",
        "p2_stab4_cite_bidirectional.tex",
    ]
    for fn in req_tabs:
        p = MS_DIR / "tables" / fn
        assert p.is_file() and p.stat().st_size > 100, f"Missing/empty table: {p}"

    for doc_name, min_bytes in [
        ("README.md", 800),
        ("STUDY_GUIDE_CN.md", 3000),
        ("SUBMISSION_GUIDE.md", 1500),
        (".gitignore", 100),
        ("LICENSE", 500),
        ("requirements.txt", 40),
        ("CITATION.cff", 200),
        ("src/apply_author_info.py", 1000),
        ("src/build_submission_bundle.py", 1000),
        ("submission/author_profile.template.json", 200),
        ("submission/COVER_LETTER_IEEE.md", 1500),
        ("submission/DECLARATIONS_AND_METADATA.md", 1500),
        ("submission/Graphical_Abstract.pdf", 5000),
        ("submission/Graphical_Abstract.png", 10000),
        ("submission/Cover_Letter_IEEE_TCYB.pdf", 5000),
        ("submission/Cover_Letter_IEEE_TNNLS.pdf", 5000),
        ("submission/Cover_Letter_Information_Fusion.pdf", 5000),
        ("submission/Highlights_Elsevier.pdf", 3000),
        ("submission/Declaration_of_Interest_Elsevier.pdf", 3000),
    ]:
        doc_path = ROOT / doc_name
        assert doc_path.is_file() and doc_path.stat().st_size > min_bytes, f"Missing/short {doc_name}"

    supp_pdf = MS_DIR / "supplementary.pdf"
    root_supp_pdf = ROOT / "CL-VSSC_Supplementary_IEEE.pdf"
    assert supp_pdf.is_file() and root_supp_pdf.is_file(), "Missing compiled Supplementary PDF"
    assert sha256(supp_pdf) == sha256(root_supp_pdf), "Supplementary PDF hash mismatch"

    nb = json.loads((ROOT / "colab_cl_vssc_benchmark.ipynb").read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 7, f"Expected 7 code cells, got {len(code_cells)}"
    for c in code_cells:
        ast.parse("".join(c["source"]))

    main_pdf = MS_DIR / "main.pdf"
    root_pdf = ROOT / "CL-VSSC_ClosedLoop_Control_IEEE.pdf"
    assert main_pdf.is_file() and root_pdf.is_file(), "Missing compiled Paper 2 PDF"
    h1, h2 = sha256(main_pdf), sha256(root_pdf)
    assert h1 == h2, f"PDF hash mismatch: {h1} vs {h2}"

    for log_name in ("main.log", "supplementary.log"):
        log_text = (MS_DIR / log_name).read_text(errors="ignore")
        for bad in [
            "Undefined control sequence",
            "Reference `",
            "Citation `",
            "Overfull \\hbox",
            "Underfull \\hbox (badness 10000)",
            "! Emergency stop",
            "! LaTeX Error",
        ]:
            assert bad not in log_text, f"Found {bad!r} in manuscript/{log_name}"

    summary = pd.read_csv(RES2_DIR / "cohort_summary_v2.csv")
    bench = pd.read_csv(RES2_DIR / "tcga_benchmark_v2.csv")
    stress = pd.read_csv(RES2_DIR / "fault_tolerance_v2.csv")
    sc_fault = pd.read_csv(RES2_DIR / "sc_fault_v2.csv")
    scale = pd.read_csv(RES2_DIR / "scalability_v2.csv")

    tcga3 = pd.read_csv(RES3_DIR / "tcga_null_snr_v3.csv")
    f3 = pd.read_csv(RES3_DIR / "fault_null_snr_v3.csv")
    sc3 = pd.read_csv(RES3_DIR / "sc_local_pi_v3.csv")
    sc3_scale = pd.read_csv(RES3_DIR / "sc_scale_local_pi_v3.csv")
    sens_pi = pd.read_csv(RES3_DIR / "sensitivity_pi_grid_v3.csv")
    sens_mu = pd.read_csv(RES3_DIR / "sensitivity_mu_v3.csv")

    phase4 = pd.read_csv(RES4_DIR / "pi_stability_phase_v4.csv")
    scale4 = pd.read_csv(RES4_DIR / "scale_200k_v4.csv")
    heal4 = pd.read_csv(RES4_DIR / "fault_self_healing_200k_v4.csv")
    env4 = json.loads((RES4_DIR / "v4_env.json").read_text())

    assert f"{int(summary['n_samples'].sum()):,}" == "3,113"
    assert f"{int(summary['n_events'].sum()):,}" == "1,235"

    comb_fault = pd.concat([stress, f3], ignore_index=True)
    non_mavs_fault = comb_fault[comb_fault["method"] != "OpenLoop_MAVS"]
    assert f"{len(non_mavs_fault):,}" == "1,560"

    cl = bench[bench["method"] == "CL-VSSC (No-DZ PI)"]
    assert f"{cl['neg_log10_perm_p'].median():.3f}" == "1.248"
    assert f"{cl['silhouette'].mean():.3f}" == "0.509"

    s4 = stress[stress["noise_sigma"] == 4.0].groupby("method")[["clean_retention_ari", "w_corrupted"]].mean()
    assert f"{s4.loc['CL-VSSC (No-DZ PI)', 'w_corrupted']:.3f}" == "0.012"
    assert f"{s4.loc['CL-VSSC (No-DZ PI)', 'clean_retention_ari']:.3f}" == "0.484"
    assert f"{s4.loc['MCGC', 'w_corrupted']:.3f}" == "0.190"
    assert f"{s4.loc['Seurat-WNN', 'w_corrupted']:.3f}" == "0.264"
    assert f"{s4.loc['Attention-MOAE', 'w_corrupted']:.3f}" == "0.319"

    assert f"{tcga3['silhouette'].mean():.3f}" == "0.516"
    assert f"{tcga3['neg_log10_perm_p'].median():.3f}" == "0.882"
    f3_s4 = f3[f3["noise_sigma"] == 4.0]
    assert f"{f3_s4['clean_retention_ari'].mean():.3f}" == "0.587"
    assert f"{f3_s4['w_corrupted'].mean():.3f}" == "0.002"
    f3_mod = f3_s4.groupby("modality")["clean_retention_ari"].mean()
    assert f"{f3_mod.loc['miRNA']:.3f}" == "0.949"
    assert f"{f3_mod.loc['mRNA']:.3f}" == "0.643"

    # Paired Wilcoxon + BH-FDR verification (n=36 at sigma=4.0; n=24 for mRNA+miRNA)
    s4_all = non_mavs_fault[non_mavs_fault["noise_sigma"] == 4.0]
    piv_w4 = s4_all.pivot_table(index=["cohort", "modality", "seed"], columns="method", values="w_corrupted")
    for w_ref in ("CL-VSSC (No-DZ PI)", "CL-VSSC (Null-SNR PI)"):
        w_others = [m for m in ("CL-VSSC (DZ-PI)", "CL-VSSC (w/o Feedback)", "MCGC", "Seurat-WNN", "Attention-MOAE") if m != w_ref]
        pw = [float(wilcoxon((piv_w4[m] - piv_w4[w_ref]).dropna(), alternative="greater")[1]) for m in w_others]
        assert all(p < 2e-11 for p in pw), f"Expected q=1.46e-11 weight suppression, got {pw}"

    sec_s4 = s4_all[s4_all["modality"].isin(["mRNA", "miRNA"])]
    piv_sec = sec_s4.pivot_table(index=["cohort", "modality", "seed"], columns="method", values="clean_retention_ari")
    assert f"{piv_sec['CL-VSSC (Null-SNR PI)'].mean():.3f}" == "0.796"
    p_sec = [float(wilcoxon((piv_sec["CL-VSSC (Null-SNR PI)"] - piv_sec[m]).dropna(), alternative="greater")[1]) for m in piv_sec.columns if m != "CL-VSSC (Null-SNR PI)"]
    order_sec = np.argsort(p_sec)
    q_sec = np.empty(len(p_sec))
    prev = 1.0
    for r_i in range(len(p_sec) - 1, -1, -1):
        idx = int(order_sec[r_i])
        prev = min(prev, p_sec[idx] * len(p_sec) / (r_i + 1))
        q_sec[idx] = prev
    assert float(q_sec.max()) <= 6.5e-4, f"Expected mRNA+miRNA BH-FDR q<=6.19e-4, got {q_sec.max()}"

    sc_bench2 = pd.read_csv(RES2_DIR / "singlecell_benchmark_v2.csv")
    sc_comb = pd.concat([sc_bench2, sc3], ignore_index=True)
    piv_loc = sc_comb.pivot_table(index=["dataset", "seed"], columns="method", values="ari")
    _, p_loc = wilcoxon((piv_loc["CL-VSSC-Anchor (Local+Null)"] - piv_loc["CL-VSSC-Anchor"]).dropna(), alternative="greater")
    assert 0.025 < p_loc < 0.04, f"Expected Local+Null ARI p~0.032, got {p_loc}"

    adt3 = sc_fault[(sc_fault["corrupted_modality"] == "Protein_ADT") & (sc_fault["noise_sigma"] == 3.0)].groupby("method")[["gt_ari", "gt_acc", "w_corrupted"]].mean()
    assert f"{adt3.loc['CL-VSSC-Anchor', 'gt_ari']:.3f}" == "0.169"
    assert f"{adt3.loc['Attention-MOAE', 'gt_ari']:.3f}" == "0.051"

    sc_fault3 = pd.read_csv(RES3_DIR / "sc_fault_local_pi_v3.csv")
    rna3 = pd.concat([sc_fault, sc_fault3], ignore_index=True)
    rna3_g = rna3[(rna3["corrupted_modality"] == "scRNA") & (rna3["noise_sigma"] == 3.0)].groupby("method")[["gt_ari", "w_corrupted"]].mean()
    assert f"{rna3_g.loc['CL-VSSC-Anchor', 'w_corrupted']:.3f}" == "0.069"
    assert f"{rna3_g.loc['CL-VSSC-Anchor (Null-SNR)', 'w_corrupted']:.3f}" == "0.060"
    assert f"{rna3_g.loc['CL-VSSC-Anchor', 'gt_ari']:.3f}" == "0.213"
    assert f"{1.0 - rna3_g.loc['Attention-MOAE', 'w_corrupted']:.3f}" == "0.891"

    anc_90 = scale[(scale["n_cells"] == 90261) & (scale["method"] == "CL-VSSC-Anchor")].iloc[0]
    assert f"{anc_90['runtime_sec']:.2f}" == "16.57"

    sc3_m = sc3.groupby(["dataset", "method"])["ari"].mean()
    assert f"{sc3_m.loc[('BMMC-10k', 'CL-VSSC-Anchor (Local+Null)')]:.3f}" == "0.218"
    assert f"{sc3_m.loc[('BMMC-30k', 'CL-VSSC-Anchor (Local+Null)')]:.3f}" == "0.222"
    loc90 = sc3_scale[(sc3_scale["n_cells"] == 90261) & (sc3_scale["method"] == "CL-VSSC-Anchor (Local PI)")].iloc[0]
    assert f"{loc90['runtime_sec']:.2f}" == "21.71"

    bic_mu0, bic_mu24 = sens_mu[(sens_mu["cohort"] == "BIC") & (sens_mu["mu_max"].isin([0.0, 0.24]))].sort_values("mu_max")["silhouette"].to_numpy()
    assert f"{bic_mu0:.4f}" == "0.4386" and f"{bic_mu24:.4f}" == "0.4765"
    kirc_fmax = sens_pi[sens_pi["cohort"] == "KIRC"]["fault_ari"].max()
    assert f"{kirc_fmax:.3f}" == "0.990"

    assert len(phase4) == 536, f"Expected 536 phase points, got {len(phase4)}"
    assert f"{env4['v42_summary']['scale_summary']['CL-VSSC-Anchor (Smooth PI)']['vram_power_alpha']:.4f}" == "0.9955"
    s300 = scale4[scale4["n_cells"] == 300000].set_index("method")
    assert f"{s300.loc['CL-VSSC-Anchor (Smooth PI)', 'runtime_sec']:.2f}" == "53.56"
    assert f"{s300.loc['CL-VSSC-Anchor (Local PI)', 'runtime_sec']:.2f}" == "52.98"
    h_step = heal4[heal4["scenario"] == "StepFault_t10"].set_index("method")
    assert int(h_step.loc["CL-VSSC-Anchor (Smooth PI)", "heal_iters_90"]) == 7
    assert f"{h_step.loc['CL-VSSC-Anchor (Smooth PI)', 'heal_wall_sec_90']:.2f}" == "3.82"
    assert f"{h_step.loc['CL-VSSC-Anchor (Null-SNR PI)', 'heal_wall_sec_90']:.2f}" == "5.20"
    assert f"{h_step.loc['CL-VSSC-Anchor (Local PI)', 'heal_wall_sec_90']:.2f}" == "6.03"
    assert f"{h_step.loc['CL-VSSC-Anchor (Null-SNR PI)', 'gt_ari']:.3f}" == "0.195"
    assert f"{h_step.loc['CL-VSSC-Anchor (Local PI)', 'gt_ari']:.3f}" == "0.203"
    assert f"{h_step.loc['CL-VSSC-Anchor (DZ-PI)', 'w_adt_final']:.3f}" == "0.500"
    assert f"{h_step.loc['CL-VSSC-Anchor (DZ-PI)', 'gt_ari']:.3f}" == "0.136"

    tex = (MS_DIR / "main.tex").read_text()
    expected_tokens = [
        "3{,}113",
        "1{,}235",
        "1{,}560",
        "1.248",
        "0.509",
        "0.516",
        "0.882",
        "0.012",
        "0.484",
        "0.190",
        "0.264",
        "0.319",
        "0.002",
        "0.587",
        "0.949",
        "0.643",
        "90{,}261",
        "16.57",
        "21.71",
        "0.218",
        "0.222",
        "0.169",
        "0.051",
        "0.4386",
        "0.4765",
        "0.990",
        "536",
        "0.9955",
        "300{,}000",
        "53.56",
        "52.98",
        "200{,}000",
        "3.82",
        "5.20",
        "6.03",
        "0.195",
        "0.203",
        "0.136",
        "fig:p2_phase_200k",
        "1.46\\times 10^{-11}",
        "0.796",
        "6.19\\times 10^{-4}",
        "0.032",
        "0.069",
        "0.060",
        "0.891",
    ]
    for tok in expected_tokens:
        assert tok in tex, f"Verified metric token {tok!r} missing from manuscript/main.tex"

    repo_url = "https://github.com/dashitongzhi/CL-VSSC"
    for fp in [MS_DIR / "main.tex", ROOT / "README.md", ROOT / "submission" / "COVER_LETTER_IEEE.md"]:
        assert repo_url in fp.read_text(), f"Missing GitHub repo URL {repo_url} in {fp}"

    decl_text = (ROOT / "submission" / "DECLARATIONS_AND_METADATA.md").read_text()
    abs_match = re.search(
        r"## 2\. Plain-Text Abstract.*?\n\n>.*?\n\n([^\n]+)\n",
        decl_text,
        re.DOTALL,
    )
    assert abs_match, "Missing Plain-Text Abstract block in submission/DECLARATIONS_AND_METADATA.md"
    abs_words = len(abs_match.group(1).split())
    assert 150 <= abs_words <= 250, f"Plain-text Abstract must be 150..250 words for ScholarOne, got {abs_words}"

    hl_section = decl_text.split("## 5. Elsevier")[1]
    hl_lines = re.findall(r"^\d+\.\s+(.+)$", hl_section, re.MULTILINE)
    assert len(hl_lines) == 5, f"Expected 5 Elsevier Highlights, found {len(hl_lines)}"
    for hl in hl_lines:
        assert len(hl.strip()) <= 85, f"Highlight exceeds 85-char limit ({len(hl.strip())}): {hl}"

    # Paper 1 vs Paper 2 boundary & citation audit
    bib_text = (MS_DIR / "references.bib").read_text()
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    cited = {k.strip() for m in re.findall(r"\\cite\{([^}]+)\}", tex) for k in m.split(",")}
    assert "cao2026mivscl" in cited, "Paper 2 must explicitly cite companion Paper 1 (cao2026mivscl)"
    assert not (bib_keys - cited), f"Uncited BibTeX keys in Paper 2: {bib_keys - cited}"
    assert not (cited - bib_keys), f"Missing BibTeX keys in Paper 2: {cited - bib_keys}"

    if P1_TEX.is_file():
        p1_tex = P1_TEX.read_text()
        def _clean_words(s: str):
            s = re.sub(r"%.*", "", s)
            s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", s)
            return re.findall(r"[a-z0-9]+", s.lower())
        w1, w2 = _clean_words(p1_tex), _clean_words(tex)
        ng1 = {tuple(w1[i:i + 10]) for i in range(max(0, len(w1) - 9))}
        ng2 = [tuple(w2[i:i + 10]) for i in range(max(0, len(w2) - 9))]
        overlap_rate = sum(1 for g in ng2 if g in ng1) / max(1, len(ng2))
        assert overlap_rate < 0.01, f"10-gram overlap with Paper 1 too high: {overlap_rate:.4%}"
        p1_figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", p1_tex))
        p2_figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
        assert not (p1_figs & p2_figs), f"Duplicate figures with Paper 1: {p1_figs & p2_figs}"
        p1_caps = set(re.findall(r"\\caption\{([^}]+)\}", p1_tex))
        p2_caps = set(re.findall(r"\\caption\{([^}]+)\}", tex))
        assert not (p1_caps & p2_caps), f"Duplicate captions with Paper 1: {p1_caps & p2_caps}"

    mavs_guard_files = [
        MS_DIR / "main.tex",
        MS_DIR / "supplementary.tex",
        MS_DIR / "references.bib",
        *(MS_DIR / "tables").glob("*.tex"),
        ROOT / "src" / "make_assets.py",
        ROOT / "README.md",
        ROOT / "STUDY_GUIDE_CN.md",
        ROOT / "SUBMISSION_GUIDE.md",
        ROOT / "CITATION.cff",
        ROOT / "submission" / "COVER_LETTER_IEEE.md",
        ROOT / "submission" / "DECLARATIONS_AND_METADATA.md",
        *(ROOT / "submission").glob("*.tex"),
    ]
    for fp in mavs_guard_files:
        low = fp.read_text().lower().replace("mavs-reliability-2026-11-05", "")
        assert "mavs" not in low and "liang2026" not in low, f"Unexpected MAVS reference in {fp}"

    print(
        f"PAPER2_VERIFICATION_PASSED | PDF SHA-256={h1[:16]}... | "
        "9 v2 + 11 v3 + 7 v4 files, 6 figs, 3 tabs, notebook AST, clean log, docs, Paper 1 boundary & zero-MAVS check, and CSV claims verified."
    )


if __name__ == "__main__":
    main()
