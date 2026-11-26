"""Generate publication-grade vector figures and programmatic LaTeX tables for Paper 2 (CL-VSSC & CL-VSSC-Anchor) from results_v2/ and results_v3/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES2_DIR = ROOT / "results_v2"
RES3_DIR = ROOT / "results_v3"
RES4_DIR = ROOT / "results_v4"

FIG_DIR = ROOT / "manuscript" / "figures"
TAB_DIR = ROOT / "manuscript" / "tables"
SUB_DIR = ROOT / "submission"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 8.6,
        "axes.titlesize": 9.6,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.1,
        "figure.dpi": 200,
    }
)

COHORTS = ["AML", "BIC", "COAD", "GBM", "KIRC", "LIHC", "LUSC", "OV", "SARC", "SKCM"]

PALETTE: Dict[str, str] = {
    "CL-VSSC (No-DZ PI)": "#0072B2",
    "CL-VSSC (Null-SNR PI)": "#9E0142",
    "CL-VSSC (DZ-PI)": "#009E73",
    "CL-VSSC (P-only)": "#56B4E9",
    "CL-VSSC (w/o Feedback)": "#4D4D4D",
    "SNF": "#009E73",
    "SNF (0-imp)": "#009E73",
    "NEMO": "#CC79A7",
    "MCGC": "#E69F00",
    "Seurat-WNN": "#8C564B",
    "Attention-MOAE": "#9467BD",
    "Subtype-GAN": "#7F7F7F",
    "Subtype-GAN (0-imp)": "#7F7F7F",
    "CL-VSSC-Anchor": "#0072B2",
    "CL-VSSC-Anchor (Local PI)": "#9E0142",
    "CL-VSSC-Anchor (Local+Null)": "#01665E",
    "CL-VSSC-Anchor (Null-SNR)": "#5E4FA2",
    "CL-VSSC-Anchor (DZ-PI)": "#009E73",
    "CL-VSSC-Anchor (w/o FB)": "#4D4D4D",
    "CL-VSSC (Dense)": "#003B5C",
    "LMVSC": "#D55E00",
}

LINESTYLES: Dict[str, object] = {
    "CL-VSSC (No-DZ PI)": "-",
    "CL-VSSC (Null-SNR PI)": "-",
    "CL-VSSC (DZ-PI)": "--",
    "CL-VSSC (w/o Feedback)": "--",
    "SNF": "-.",
    "SNF (0-imp)": "-.",
    "NEMO": ":",
    "MCGC": (0, (3, 1, 1, 1)),
    "Seurat-WNN": (0, (5, 1)),
    "Attention-MOAE": (0, (3, 1, 1, 1, 1, 1)),
    "Subtype-GAN": (0, (5, 2)),
    "Subtype-GAN (0-imp)": (0, (5, 2)),
    "CL-VSSC-Anchor": "-",
    "CL-VSSC-Anchor (Local PI)": "-",
    "CL-VSSC-Anchor (Local+Null)": "--",
    "CL-VSSC-Anchor (Null-SNR)": "-.",
    "CL-VSSC-Anchor (DZ-PI)": "--",
    "CL-VSSC-Anchor (w/o FB)": ":",
    "CL-VSSC (Dense)": "-.",
    "LMVSC": ":",
}

MARKERS: Dict[str, str] = {
    "CL-VSSC (No-DZ PI)": "o",
    "CL-VSSC (Null-SNR PI)": "*",
    "CL-VSSC (DZ-PI)": "P",
    "CL-VSSC (w/o Feedback)": "x",
    "SNF": "^",
    "SNF (0-imp)": "^",
    "NEMO": "D",
    "MCGC": "v",
    "Seurat-WNN": "*",
    "Attention-MOAE": "h",
    "Subtype-GAN": "d",
    "Subtype-GAN (0-imp)": "d",
    "CL-VSSC-Anchor": "o",
    "CL-VSSC-Anchor (Local PI)": "p",
    "CL-VSSC-Anchor (Local+Null)": "*",
    "CL-VSSC-Anchor (Null-SNR)": "h",
    "CL-VSSC-Anchor (DZ-PI)": "P",
    "CL-VSSC-Anchor (w/o FB)": "x",
    "CL-VSSC (Dense)": "s",
    "LMVSC": "<",
}

DISPLAY_P2: Dict[str, str] = {
    "CL-VSSC (No-DZ PI)": "CL-VSSC",
    "CL-VSSC (Null-SNR PI)": "CL-VSSC (Null-SNR)",
    "CL-VSSC (DZ-PI)": "CL-VSSC (DZ-PI)",
    "CL-VSSC (P-only)": "P-only",
    "CL-VSSC (w/o Feedback)": "w/o Feedback",
    "CL-VSSC (w/o SpecCtrl)": "w/o SpecCtrl",
    "SNF": "SNF",
    "NEMO": "NEMO",
    "MCGC": "MCGC",
    "Seurat-WNN": "Seurat-WNN",
    "Attention-MOAE": "Attention-MOAE",
    "Subtype-GAN": "Subtype-GAN",
    "CL-VSSC-Anchor": "CL-VSSC-Anchor",
    "CL-VSSC-Anchor (Local PI)": "Anchor (Local PI)",
    "CL-VSSC-Anchor (Local+Null)": "Anchor (Local+Null)",
    "CL-VSSC-Anchor (Null-SNR)": "Anchor (Null-SNR)",
    "CL-VSSC-Anchor (DZ-PI)": "Anchor (DZ-PI)",
    "CL-VSSC-Anchor (w/o FB)": "Anchor (w/o FB)",
    "CL-VSSC (Dense)": "CL-VSSC (Dense)",
    "LMVSC": "LMVSC",
}


def draw_box(ax, x, y, w, h, title, body, fc="#F5F9FC", ec="#0072B2", lw=1.4, title_color="#003B5C"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 1.7, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=title_color)
    ax.text(x + w / 2, y + h / 2 - 1.2, body, ha="center", va="center", fontsize=7.3, color="#222222", linespacing=1.36)


def draw_arrow(ax, x1, y1, x2, y2, label=None, lx=None, ly=None, color="#333333", rad=0.0, ls="-"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5,head_width=3.2",
        color=color, linewidth=1.4, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        tx = lx if lx is not None else 0.5 * (x1 + x2)
        ty = ly if ly is not None else 0.5 * (y1 + y2)
        ax.text(
            tx, ty, label, ha="center", va="center", fontsize=7.0, color=color,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.90),
        )


def sci_tex(val: float) -> str:
    s = f"{float(val):.1e}"
    base, exp = s.split("e")
    return f"${base}\\times 10^{{{int(exp)}}}$"


def _bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    qvals = np.empty(m, dtype=float)
    prev = 1.0
    for rank_rev in range(m - 1, -1, -1):
        idx = int(order[rank_rev])
        val = float(pvals[idx]) * m / float(rank_rev + 1)
        prev = min(prev, val)
        qvals[idx] = prev
    return qvals.tolist()


def _fmt_q(q: float) -> str:
    if q < 1e-3:
        s = f"{q:.1e}"
        base, exp = s.split("e")
        return f"${base}{{\\times}}10^{{{int(exp)}}}$"
    return f"{q:.3f}"


# =====================================================================
# PAPER 2 ASSETS (manuscript/): Closed-Loop Control & 90k Bipartite
# =====================================================================

def make_p2_fig1_control_loop() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    draw_box(
        ax, 1.2, 23.0, 18.0, 17.2,
        "Multi-Sensor Inputs",
        r"Multi-modal views $\{X^{(v)}\}_{v=1}^V$" + "\n"
        + r"Exogenous noise $\sigma$ / dropout" + "\n"
        + r"TCGA ($3{,}113$) & CITE-seq ($90{,}261$)",
        fc="#F4F6F8", ec="#555555", title_color="#222222",
    )
    draw_box(
        ax, 23.6, 22.8, 28.4, 17.6,
        "Stiefel Plant & Bipartite Graph",
        r"Orthogonal states: $Z^\top Z=I_r,\; Z^\top U^{(v)}=0$" + "\n"
        + r"Operator-norm scaling ($\sigma_{\max}^2$)" + "\n"
        + r"Dense $A^{(v)}$ ($n{\times}n$) or Landmark $\tilde{B}^{(v)}$ ($n{\times}m$)",
        fc="#EDF5FB", ec="#0072B2",
    )
    draw_box(
        ax, 56.4, 22.8, 27.6, 17.6,
        "Dual Eigengap Actuator",
        r"Simplex fusion $S_{\mathrm{w}} / B_{\mathrm{w}}$ & dual pruning" + "\n"
        + r"Anchor Gram $\mathcal{O}(nm^2)$: $\mathrm{eigh}(\tilde{B}^\top \tilde{B})$" + "\n"
        + r"Integral gain: $\mu(t{+}1)=\mathrm{clip}(\mu+\eta_\mu(J_{\mathrm{spec}}-\epsilon_{\mathrm{spec}}))$",
        fc="#EEF8F3", ec="#009E73", title_color="#005A40",
    )
    draw_box(
        ax, 87.8, 23.8, 11.0, 15.6,
        "Controlled\nPartition",
        r"Eigenvectors $F$" + "\n"
        + r"ISS bounded" + "\n"
        + r"Subtypes $\hat{y}$",
        fc="#F4F6F8", ec="#555555", title_color="#222222",
    )
    draw_box(
        ax, 55.2, 2.2, 31.0, 14.5,
        "Cut & Null-SNR Fault Sensor",
        r"Intrinsic + consensus cuts & DOF residual $e_v(t)$" + "\n"
        + r"Null-SNR baseline $c_v^{\mathrm{null}}$ (column shuffle)" + "\n"
        + r"Double-centered anchor deviation $\tilde{\delta}_{jv}^{\mathrm{loc}}$ + $\mathcal{V}(t)$",
        fc="#FFF6ED", ec="#D55E00", title_color="#8C3B00",
    )
    draw_box(
        ax, 12.2, 2.2, 35.5, 14.5,
        "PI Simplex Mirror-Descent Controller",
        r"Damped integrator: $I_v(t{+}1)=(1-\beta_I)I_v(t)+\beta_I d_v(t)$" + "\n"
        + r"Global $w(t)\in\Delta^V$ or Anchor-Local $W_{\mathrm{loc}}(t)\in\mathbb{R}^{m\times V}$" + "\n"
        + r"$\Rightarrow$ Exponential fault rejection $\propto e^{-(1+\kappa_I)\Delta e/\tau}$",
        fc="#FDF3F8", ec="#CC79A7", title_color="#782857",
    )

    def _ortho(pts, color="#333333", lw=1.4, ls="-"):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(pts) > 2:
            ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="round")
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.28", color=color, lw=lw, ls=ls))

    # Forward path arrows
    _ortho([(19.4, 31.6), (23.4, 31.6)], color="#333333")
    _ortho([(52.2, 31.6), (56.2, 31.6)], color="#0072B2")
    ax.text(54.2, 33.4, r"$A^{(v)}/\tilde{B}^{(v)}$", ha="center", va="bottom", fontsize=7.2, color="#0072B2", fontweight="bold")
    _ortho([(84.2, 31.6), (87.6, 31.6)], color="#009E73")

    # Orthogonal non-crossing feedback taps into Cut & Null-SNR Fault Sensor
    _ortho([(72.0, 22.6), (72.0, 16.9)], color="#D55E00")
    ax.text(73.1, 19.8, r"$F(t),\;\Delta\lambda_c$", ha="left", va="center", fontsize=7.0, color="#8C3B00")
    _ortho([(43.5, 22.6), (43.5, 19.6), (62.0, 19.6), (62.0, 16.9)], color="#D55E00", ls="--")
    ax.text(52.8, 20.1, r"$Z,\;U^{(v)},\;\tilde{r}_v$", ha="center", va="bottom", fontsize=6.9, color="#8C3B00")

    # Sensor -> PI Controller -> summing bus -> Plant & Actuator (zero crossings!)
    _ortho([(55.0, 9.45), (47.9, 9.45)], color="#8C3B00")
    ax.text(51.45, 10.6, r"$e_v(t)$", ha="center", va="bottom", fontsize=7.1, color="#8C3B00", fontweight="bold")

    # Summing bus node
    junc = Circle((31.2, 19.8), 1.05, facecolor="#FFFFFF", edgecolor="#782857", lw=1.3)
    ax.add_patch(junc)
    ax.text(31.2, 19.8, r"$\Sigma$", ha="center", va="center", fontsize=7.6, color="#782857", fontweight="bold")
    _ortho([(31.2, 16.9), (31.2, 18.75)], color="#CC79A7")
    _ortho([(31.2, 20.85), (31.2, 22.6)], color="#CC79A7")
    ax.text(29.8, 19.8, r"$w(t)$ to $Z$", ha="right", va="center", fontsize=7.0, color="#782857", fontweight="bold")
    _ortho([(32.25, 19.8), (39.0, 19.8), (39.0, 22.0), (57.2, 22.0), (57.2, 22.6)], color="#CC79A7")
    ax.text(53.0, 17.6, r"$w(t),\;W_{\mathrm{loc}}(t)$", ha="right", va="center", fontsize=7.0, color="#782857", fontweight="bold")

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "p2_fig1_control_loop.pdf", bbox_inches="tight")
    plt.close(fig)


def make_p2_fig2_dynamics(traj: dict, traj3: dict) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 6.7))

    ax = axs[0, 0]
    clean_d = traj["clean_nodz_full"]
    cohort_styles = [
        ("#1B4332", "-", "o", 1.8),
        ("#0072B2", "-", "s", 1.9),
        ("#56B4E9", "--", "^", 1.4),
        ("#009E73", "-.", "D", 1.7),
        ("#2A9D8F", "-", "P", 1.9),
        ("#9E0142", "--", "v", 1.8),
        ("#CC79A7", ":", "<", 1.5),
        ("#D55E00", "-", "X", 2.0),
        ("#E69F00", "-.", "p", 1.5),
        ("#5E4FA2", "--", "h", 1.5),
    ]
    for (c, d), (col, ls, mk, lw) in zip(clean_d.items(), cohort_styles):
        vh = np.array(d["lyapunov_history"])
        delta_v = vh - vh[0]
        ax.plot(
            delta_v,
            label="OV (+0.004 flat)" if c == "OV" else c,
            color=col, ls=ls, marker=mk, markevery=6, markersize=3.8, lw=lw, alpha=0.92,
        )
    ax.axhline(0.0, color="#666666", lw=0.8, ls=":")
    ax.set_title(r"(a) Relative Lyapunov Energy $\mathcal{V}(t)-\mathcal{V}(0)$ (10 Cohorts)")
    ax.set_xlabel(r"Closed-loop iteration $t$")
    ax.set_ylabel(r"$\mathcal{V}(t)-\mathcal{V}(0)$")
    ax.set_ylim(-0.125, 0.032)
    ax.legend(loc="lower left", ncol=3, fontsize=6.2, frameon=True, framealpha=0.93)
    ax.grid(alpha=0.20, ls=":")

    ax = axs[0, 1]
    bic = clean_d["BIC"]
    t_steps = np.arange(len(bic["spectral_gap_history"]))
    gap = np.array(bic["spectral_gap_history"])
    mu = np.array(bic["mu_history"])
    l1 = ax.plot(t_steps, gap, color="#0072B2", lw=2.1, label=r"Spectral gap $\lambda_{c+1}-\lambda_c$ (BIC)")
    ax.set_xlabel(r"Closed-loop iteration $t$")
    ax.set_ylabel(r"Normalized Laplacian gap $\Delta\lambda_c$", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.axvline(15, color="#444444", ls="--", lw=1.0)
    ax.text(15.6, gap.max() * 0.87, "Warmup ends\n(t = 15)", fontsize=7.0, color="#333333")
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    l2 = ax_r.plot(t_steps, mu, color="#009E73", lw=1.8, ls="-.", label=r"Dual integral gain $\mu(t)$")
    ax_r.set_ylabel(r"Dual gain $\mu(t)$", color="#009E73")
    ax_r.tick_params(axis="y", labelcolor="#009E73")
    ax.set_title(r"(b) Spectral-Gap Tracking Controller on BIC ($n=623$)")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.20, ls=":")

    ax = axs[1, 0]
    wh_pi = np.array(traj["corrupted_nodz"]["BIC_mRNA"]["weights_history"])
    wh_null = np.array(traj3["corrupted"]["BIC_mRNA"]["weights_history"])
    wh_dz = np.array(traj["corrupted_dz"]["BIC_mRNA"]["weights_history"])
    wh_p = np.array(traj["corrupted_ponly_nodz"]["BIC_mRNA"]["weights_history"])
    t_w = np.arange(len(wh_pi))
    ax.plot(t_w, wh_pi[:, 0], color="#D55E00", lw=2.1, label=f"Smooth PI: mRNA ({wh_pi[-1,0]:.3f})")
    ax.plot(t_w, wh_null[:, 0], color="#9E0142", lw=2.1, ls="-", label=f"Null-SNR PI: mRNA ({wh_null[-1,0]:.3f})")
    ax.plot(t_w, wh_pi[:, 1], color="#0072B2", lw=2.0, label=f"Smooth PI: Methy ({wh_pi[-1,1]:.3f})")
    ax.plot(t_w, wh_dz[:, 0], color="#CC79A7", lw=1.7, ls="-.", label=f"DZ-PI: mRNA ({wh_dz[-1,0]:.3f})")
    ax.plot(t_w, wh_p[:, 0], color="#D55E00", lw=1.5, ls="--", label=rf"P-only ($\kappa_I=0$): mRNA ({wh_p[-1,0]:.3f})")
    ax.axhline(1.0 / 3.0, color="#666666", lw=1.2, ls=":", label="w/o Feedback (0.333)")
    ax.set_title(r"(c) Modality Control under BIC mRNA Fault ($\sigma=4$)")
    ax.set_xlabel(r"Closed-loop iteration $t$")
    ax.set_ylabel(r"Modality weight $w_v(t)$")
    ax.set_ylim(-0.02, 0.92)
    ax.legend(loc="upper right", fontsize=5.8, frameon=True, framealpha=0.93)
    ax.grid(alpha=0.20, ls=":")
    ax_c = ax

    ax = axs[1, 1]
    specs = [
        ("KIRC_mRNA", 0, "mRNA", "#D55E00", "-"),
        ("KIRC_Methylation", 1, "Methylation", "#0072B2", "--"),
        ("KIRC_miRNA", 2, "miRNA", "#009E73", "-."),
    ]
    for key, idx_v, lbl, col, ls in specs:
        wh = np.array(traj["corrupted_nodz"][key]["weights_history"])
        wh3 = np.array(traj3["corrupted"][key]["weights_history"])
        ax.plot(
            np.arange(len(wh)), wh[:, idx_v], color=col, ls=ls, lw=2.0,
            label=f"{lbl}: Smooth {wh[-1, idx_v]:.3f} | Null-SNR {wh3[-1, idx_v]:.3f}",
        )
    ax.set_title(r"(d) Cross-Modal Fault Suppression on KIRC ($\sigma=4, 60\%$ drop)")
    ax.set_xlabel(r"Closed-loop iteration $t$")
    ax.set_ylabel(r"Corrupted weight $w_{\mathrm{corr}}(t)$")
    ax.set_ylim(-0.01, 0.36)
    ax.legend(loc="upper right", fontsize=6.2, frameon=True, framealpha=0.93)
    ax.grid(alpha=0.20, ls=":")

    fig.tight_layout(h_pad=2.1, w_pad=2.1)

    # Create zoomed inset AFTER tight_layout to avoid tight_layout warning & legend overlap
    ax_in = inset_axes(ax_c, width="100%", height="100%", bbox_to_anchor=(0.38, 0.16, 0.58, 0.30), bbox_transform=ax_c.transAxes, borderpad=0)
    ax_in.plot(t_w, wh_pi[:, 0], color="#D55E00", lw=1.8)
    ax_in.plot(t_w, wh_null[:, 0], color="#9E0142", lw=1.8)
    ax_in.plot(t_w, wh_dz[:, 0], color="#CC79A7", lw=1.4, ls="-.")
    ax_in.plot(t_w, wh_p[:, 0], color="#D55E00", lw=1.3, ls="--")
    ax_in.set_xlim(12, 44)
    ax_in.set_ylim(-0.001, 0.042)
    ax_in.tick_params(labelsize=6.0)
    ax_in.set_title("Steady-State Rejection ($t \\geq 12$)", fontsize=6.4, pad=2)
    ax_in.grid(alpha=0.25, ls=":")

    fig.savefig(FIG_DIR / "p2_fig2_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)


def make_p2_fig3_fault_tolerance(stress: pd.DataFrame, stress3: pd.DataFrame) -> None:
    modalities = ["mRNA", "Methylation", "miRNA"]
    comb = pd.concat([stress, stress3], ignore_index=True)
    all_sigs = [0.0, 0.5, 1.0, 2.0, 4.0]
    sig_to_idx = {s: i for i, s in enumerate(all_sigs)}
    ari_methods = [
        "CL-VSSC (No-DZ PI)",
        "CL-VSSC (Null-SNR PI)",
        "CL-VSSC (DZ-PI)",
        "CL-VSSC (w/o Feedback)",
        "SNF",
        "NEMO",
        "MCGC",
        "Seurat-WNN",
        "Attention-MOAE",
        "Subtype-GAN",
    ]
    weight_methods = [
        "CL-VSSC (No-DZ PI)",
        "CL-VSSC (Null-SNR PI)",
        "CL-VSSC (DZ-PI)",
        "CL-VSSC (w/o Feedback)",
        "MCGC",
        "Seurat-WNN",
        "Attention-MOAE",
    ]

    fig, axs = plt.subplots(2, 3, figsize=(11.6, 6.35), sharex=True)
    for col_idx, mod in enumerate(modalities):
        sub_m = comb[comb["modality"] == mod]
        ax_top = axs[0, col_idx]
        for m in ari_methods:
            s = sub_m[sub_m["method"] == m]
            grp = s.groupby("noise_sigma")["clean_retention_ari"]
            m_sigs = sorted(s["noise_sigma"].unique())
            means = grp.mean().reindex(m_sigs)
            se = (grp.std() / np.sqrt(grp.count())).reindex(m_sigs)
            x_ord = [sig_to_idx[float(sv)] for sv in m_sigs]
            ax_top.errorbar(
                x_ord, means.to_numpy(), yerr=se.to_numpy(),
                label=DISPLAY_P2[m], color=PALETTE[m], linestyle=LINESTYLES[m],
                marker=MARKERS[m], markersize=4.6 if "Null-SNR" in m else 4.0,
                lw=2.1 if ("No-DZ" in m or "Null-SNR" in m) else 1.25, capsize=2.0,
            )
        ax_top.set_title(f"{mod} Corruption", fontweight="bold")
        if col_idx == 0:
            ax_top.set_ylabel("Clean Subtype Retention (ARI)")
        ax_top.set_ylim(0.10, 1.04)
        ax_top.grid(alpha=0.20, ls=":")

        ax_bot = axs[1, col_idx]
        for m in weight_methods:
            s = sub_m[sub_m["method"] == m]
            grp = s.groupby("noise_sigma")["w_corrupted"]
            m_sigs = sorted(s["noise_sigma"].unique())
            means = grp.mean().reindex(m_sigs)
            se = (grp.std() / np.sqrt(grp.count())).reindex(m_sigs)
            x_ord = [sig_to_idx[float(sv)] for sv in m_sigs]
            ax_bot.errorbar(
                x_ord, means.to_numpy(), yerr=se.to_numpy(),
                label=DISPLAY_P2[m], color=PALETTE[m], linestyle=LINESTYLES[m],
                marker=MARKERS[m], markersize=4.6 if "Null-SNR" in m else 4.0,
                lw=2.1 if ("No-DZ" in m or "Null-SNR" in m) else 1.25, capsize=2.0,
            )
        ax_bot.set_xlabel(r"Noise severity $\sigma$ (dropout $0\% \rightarrow 60\%$)")
        if col_idx == 0:
            ax_bot.set_ylabel(r"Corrupted weight $w_{\mathrm{corr}}$")
        ax_bot.set_ylim(-0.01, 0.72)
        ax_bot.set_xticks(np.arange(len(all_sigs)))
        ax_bot.set_xticklabels([f"{sv:.1f}" for sv in all_sigs])
        ax_bot.grid(alpha=0.20, ls=":")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=5, frameon=True, fontsize=7.0)
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=1.7, w_pad=1.5)
    fig.savefig(FIG_DIR / "p2_fig3_fault_tolerance.pdf", bbox_inches="tight")
    plt.close(fig)


def make_p2_fig4_singlecell_scale(
    sc_bench: pd.DataFrame,
    sc_fault: pd.DataFrame,
    scale: pd.DataFrame,
    sc_bench3: pd.DataFrame,
    sc_fault3: pd.DataFrame,
    sc_scale3: pd.DataFrame,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11.4, 6.85))
    comb_b = pd.concat([sc_bench, sc_bench3], ignore_index=True)
    comb_f = pd.concat([sc_fault, sc_fault3], ignore_index=True)

    ax = axs[0, 0]
    ds_order = ["BMMC-Site1", "BMMC-Site2", "BMMC-10k", "BMMC-30k"]
    m_order = ["CL-VSSC-Anchor", "CL-VSSC-Anchor (Local+Null)", "LMVSC", "Seurat-WNN", "Attention-MOAE", "Subtype-GAN"]
    x = np.arange(len(ds_order))
    bw = 0.13
    for idx, m in enumerate(m_order):
        sub = comb_b[comb_b["method"] == m].groupby("dataset")["nmi"].agg(["mean", "std"]).reindex(ds_order)
        ax.bar(
            x + (idx - 2.5) * bw, sub["mean"], width=bw, yerr=sub["std"], capsize=1.8,
            color=PALETTE[m], label=DISPLAY_P2[m], edgecolor="white", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, rotation=10)
    ax.set_ylabel("Ground-truth Cell-Type NMI")
    ax.set_title("(a) CITE-seq BMMC Benchmarks (39–45 Cell Types)")
    ax.set_ylim(0.45, 0.72)
    ax.grid(axis="y", alpha=0.20, ls=":")
    ax.legend(loc="upper left", ncol=2, fontsize=6.1, frameon=True, framealpha=0.93)

    ax = axs[0, 1]
    sigmas = [0.0, 1.5, 3.0]
    f_methods = [
        "CL-VSSC-Anchor",
        "CL-VSSC-Anchor (Local+Null)",
        "CL-VSSC-Anchor (DZ-PI)",
        "CL-VSSC-Anchor (w/o FB)",
        "LMVSC",
        "Seurat-WNN",
        "Attention-MOAE",
        "Subtype-GAN",
    ]
    sub_adt = comb_f[comb_f["corrupted_modality"] == "Protein_ADT"]
    for m in f_methods:
        s = sub_adt[sub_adt["method"] == m].groupby("noise_sigma")["gt_ari"].mean().reindex(sigmas)
        ax.plot(
            sigmas, s.to_numpy(), label=DISPLAY_P2[m], color=PALETTE[m], linestyle=LINESTYLES[m],
            marker=MARKERS[m], lw=2.2 if m in ("CL-VSSC-Anchor", "CL-VSSC-Anchor (Local+Null)") else 1.35, markersize=4.8,
        )
    ax.set_xticks(sigmas)
    ax.set_xlabel(r"ADT surface-protein noise $\sigma$ (dropout $0\%\rightarrow 50\%$)")
    ax.set_ylabel("Ground-truth Cell-Type ARI")
    ax.set_title(r"(b) BMMC-10k Protein ADT Fault Test ($n=10{,}000$)")
    ax.set_ylim(0.03, 0.27)
    ax.grid(alpha=0.20, ls=":")
    ax.legend(loc="lower left", ncol=2, fontsize=5.9, frameon=True, framealpha=0.93)

    ax = axs[1, 0]
    s_methods = ["CL-VSSC-Anchor", "CL-VSSC (Dense)", "SNF", "MCGC", "LMVSC", "Subtype-GAN"]
    for m in s_methods:
        sub = scale[scale["method"] == m].dropna(subset=["runtime_sec"]).sort_values("n_cells")
        ax.loglog(
            sub["n_cells"], sub["runtime_sec"],
            label=DISPLAY_P2[m], color=PALETTE[m], linestyle=LINESTYLES[m],
            marker=MARKERS[m], lw=2.2 if "Anchor" in m else 1.4, markersize=5.0,
        )
    sub3 = sc_scale3[sc_scale3["method"] == "CL-VSSC-Anchor (Local+Null)"].sort_values("n_cells")
    ax.loglog(
        sub3["n_cells"], sub3["runtime_sec"],
        label="Anchor Local+Null (L4)", color=PALETTE["CL-VSSC-Anchor (Local+Null)"],
        linestyle="--", marker="*", lw=1.7, markersize=5.2,
    )
    ax.set_xlabel(r"Number of single cells $n$ (log scale)")
    ax.set_ylabel("GPU wall-clock runtime (s, log)")
    ax.set_title(r"(c) Runtime Scaling ($n = 1{,}000 \rightarrow 90{,}261$)")
    ax.grid(True, which="both", alpha=0.20, ls=":")
    ax.legend(loc="upper left", ncol=2, fontsize=6.1, frameon=True, framealpha=0.93)

    ax = axs[1, 1]
    for m in s_methods:
        sub = scale[scale["method"] == m].dropna(subset=["peak_vram_mb"]).sort_values("n_cells")
        ax.loglog(
            sub["n_cells"], sub["peak_vram_mb"],
            label=DISPLAY_P2[m], color=PALETTE[m], linestyle=LINESTYLES[m],
            marker=MARKERS[m], lw=2.2 if "Anchor" in m else 1.4, markersize=5.0,
        )
    ax.loglog(
        sub3["n_cells"], sub3["peak_vram_mb"],
        label="Anchor Local+Null (L4)", color=PALETTE["CL-VSSC-Anchor (Local+Null)"],
        linestyle="--", marker="*", lw=1.7, markersize=5.2,
    )
    ax.axhline(8192, color="#B22222", ls=":", lw=1.0, label="8 GB GPU ceiling")
    ax.set_xlabel(r"Number of single cells $n$ (log scale)")
    ax.set_ylabel("Peak GPU VRAM (MB, log)")
    ax.set_title(r"(d) Peak Memory Scaling ($13.3\times$ Lower at 10k)")
    ax.grid(True, which="both", alpha=0.20, ls=":")
    ax.legend(loc="lower right", ncol=2, fontsize=6.1, frameon=True, framealpha=0.93)

    fig.tight_layout(h_pad=2.1, w_pad=2.0)
    fig.savefig(FIG_DIR / "p2_fig4_singlecell_scale.pdf", bbox_inches="tight")
    plt.close(fig)


def make_p2_fig5_sensitivity(sens_pi: pd.DataFrame, sens_mu: pd.DataFrame, sens_anc: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.1))

    ax = axs[0]
    kappas = sorted(sens_pi["kappa_i"].unique())
    taus = sorted(sens_pi["tau"].unique())
    piv_ari = sens_pi.groupby(["kappa_i", "tau"])["fault_ari"].mean().unstack().loc[kappas, taus]
    piv_cln = sens_pi.groupby(["kappa_i", "tau"])["clean_neg_log10_perm_p"].mean().unstack().loc[kappas, taus]
    mat = piv_ari.to_numpy()
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.38, vmax=0.82)
    ax.set_xticks(np.arange(len(taus)))
    ax.set_xticklabels([f"{t:.2f}" for t in taus])
    ax.set_yticks(np.arange(len(kappas)))
    ax.set_yticklabels([f"{k:.2f}" for k in kappas])
    ax.set_xlabel(r"Mirror-descent entropy temperature $\tau$")
    ax.set_ylabel(r"Integral gain $\kappa_I$")
    ax.set_title(r"(a) PI Grid: Mean Fault ARI / Clean $-\log_{10}p$", pad=8)
    for i in range(len(kappas)):
        for j in range(len(taus)):
            v_ari = mat[i, j]
            v_cln = float(piv_cln.iloc[i, j])
            txt_c = "white" if v_ari >= 0.62 else "#111111"
            is_def = abs(kappas[i] - 0.35) < 1e-6 and abs(taus[j] - 0.20) < 1e-6
            ax.text(j, i, f"{v_ari:.2f}\n({v_cln:.2f})", ha="center", va="center",
                    color=txt_c, fontweight="bold" if is_def else "normal", fontsize=6.6)
            if is_def:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#D55E00", lw=1.8))
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Mean Fault ARI (BIC & KIRC)")

    ax = axs[1]
    mus = sorted(sens_mu["mu_max"].unique())
    bic_m = sens_mu[sens_mu["cohort"] == "BIC"].set_index("mu_max").reindex(mus)
    kirc_m = sens_mu[sens_mu["cohort"] == "KIRC"].set_index("mu_max").reindex(mus)
    l1 = ax.plot(mus, bic_m["silhouette"], "-o", color="#0072B2", lw=2.0, markersize=5.0, label="BIC Silhouette")
    l2 = ax.plot(mus, kirc_m["silhouette"], "--s", color="#01665E", lw=2.0, markersize=4.8, label="KIRC Silhouette")
    ax.set_xlabel(r"Dual spectral-gap ceiling $\mu_{\max}$")
    ax.set_ylabel("Mean Silhouette score", color="#003B5C")
    ax.set_ylim(0.425, 0.515)
    ax.set_xticks(mus)
    ax.axvline(0.12, color="#D55E00", ls=":", lw=1.3)
    ax.text(0.124, 0.429, "Default\n0.12", fontsize=6.7, color="#D55E00")
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    l3 = ax_r.plot(mus, bic_m["gap_final"], "-.^", color="#D55E00", lw=1.8, markersize=4.8, label=r"BIC Gap $\Delta\lambda_c$")
    l4 = ax_r.plot(mus, kirc_m["clean_neg_log10_perm_p"] / 25.0, ":d", color="#CC79A7", lw=1.7, markersize=4.6, label=r"KIRC $-\log_{10}p / 25$")
    ax_r.set_ylabel(r"Spectral gap $\Delta\lambda_c$ (orange)", color="#D55E00")
    ax_r.tick_params(axis="y", labelcolor="#D55E00")
    ax_r.set_ylim(0.02, 0.13)
    lines = l1 + l2 + l3 + l4
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=6.1, frameon=True, framealpha=0.93)
    ax.set_title(r"(b) Spectral-Gap Gain $\mu_{\max}$ Sweep", pad=8)
    ax.grid(alpha=0.20, ls=":")

    ax = axs[2]
    anc = sens_anc.sort_values("n_anchors")
    ms = anc["n_anchors"].to_numpy()
    p1 = ax.plot(ms, anc["clean_nmi"], "-o", color="#0072B2", lw=2.0, markersize=5.0, label="Clean NMI")
    p2 = ax.plot(ms, anc["adt_fault_acc"], "-s", color="#9E0142", lw=2.0, markersize=4.8, label="ADT-Fault ACC")
    p3 = ax.plot(ms, anc["adt_fault_ari"], "--^", color="#009E73", lw=1.8, markersize=4.8, label="ADT-Fault ARI")
    ax.axvline(384, color="#D55E00", ls=":", lw=1.3)
    ax.text(396, 0.08, "Default\nm=384", fontsize=6.7, color="#D55E00")
    ax.set_xlabel(r"Landmark anchor budget $m$")
    ax.set_ylabel("BMMC-10k Clustering Score")
    ax.set_xticks(ms)
    ax.set_ylim(0.03, 0.76)
    ax_r2 = ax.twinx()
    ax_r2.spines["top"].set_visible(False)
    p4 = ax_r2.plot(ms, anc["runtime_sec"], ":D", color="#555555", lw=1.7, markersize=4.5, label="L4 Runtime (s)")
    ax_r2.set_ylabel("L4 GPU Runtime (s)", color="#555555")
    ax_r2.tick_params(axis="y", labelcolor="#555555")
    ax_r2.set_ylim(3.0, 8.8)
    lines2 = p1 + p2 + p3 + p4
    ax.legend(lines2, [l.get_label() for l in lines2], loc="upper right", ncol=2, fontsize=6.0, frameon=True, framealpha=0.94)
    ax.set_title(r"(c) BMMC-10k Landmark Budget $m$ Sweep", pad=8)
    ax.grid(alpha=0.20, ls=":")

    fig.tight_layout(w_pad=2.1)
    fig.savefig(FIG_DIR / "p2_fig5_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def write_p2_tab1_controller_ablation(
    bench: pd.DataFrame,
    ksens: pd.DataFrame,
    stress: pd.DataFrame,
    traj: dict,
    tcga3: pd.DataFrame,
    ksens3: pd.DataFrame,
    stress3: pd.DataFrame,
    traj3: dict,
) -> None:
    ab_methods = [
        "CL-VSSC (No-DZ PI)",
        "CL-VSSC (Null-SNR PI)",
        "CL-VSSC (DZ-PI)",
        "CL-VSSC (P-only)",
        "CL-VSSC (w/o Feedback)",
        "CL-VSSC (w/o SpecCtrl)",
    ]
    comb_b = pd.concat([bench, tcga3], ignore_index=True)
    comb_k = pd.concat([ksens, ksens3], ignore_index=True)
    comb_s = pd.concat([stress, stress3], ignore_index=True)
    s4 = comb_s[comb_s["noise_sigma"] == 4.0]
    bic_map = {
        "CL-VSSC (No-DZ PI)": f"{traj['corrupted_nodz']['BIC_mRNA']['weights_history'][-1][0]:.3f}",
        "CL-VSSC (Null-SNR PI)": f"{traj3['corrupted']['BIC_mRNA']['weights_history'][-1][0]:.3f}",
        "CL-VSSC (DZ-PI)": f"{traj['corrupted_dz']['BIC_mRNA']['weights_history'][-1][0]:.3f}",
        "CL-VSSC (P-only)": f"{traj['corrupted_ponly_nodz']['BIC_mRNA']['weights_history'][-1][0]:.3f}",
        "CL-VSSC (w/o Feedback)": "0.333",
        "CL-VSSC (w/o SpecCtrl)": "---",
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Closed-Loop Controller and Sensor Ablation on 10 TCGA Cohorts and $\sigma=4.0$ Multi-Modal Sensor Faults}",
        r"\label{tab:p2_ablation}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Variant & \shortstack{Clean Sig.\\($p_{\mathrm{perm}}{<}0.05$)} & \shortstack{Clean Mean\\$-\log_{10}p_{\mathrm{perm}}$} & \shortstack{Clean Med.\\$-\log_{10}p_{\mathrm{perm}}$} & \shortstack{Clean\\Silhouette} & \shortstack{$k$-Sweep\\(/40)} & \shortstack{BIC Fault\\$w_{\mathrm{mRNA}}$} & \shortstack{Fault Mean\\$w_{\mathrm{corr}}$} & \shortstack{Fault ARI\\(36 runs)} \\",
        r"\midrule",
    ]
    for m in ab_methods:
        b = comb_b[comb_b["method"] == m]
        sig_cnt = int((b["perm_p"] < 0.05).sum())
        mn = float(b["neg_log10_perm_p"].mean())
        md = float(b["neg_log10_perm_p"].median())
        sil = float(b["silhouette"].mean())
        k_cnt = f"{int((comb_k[comb_k['method'] == m]['perm_p'] < 0.05).sum())}/40" if m in set(comb_k["method"]) else "---"
        st_m = s4[s4["method"] == m]
        w_m = f"{st_m['w_corrupted'].mean():.3f}" if len(st_m) > 0 and not np.isnan(st_m['w_corrupted'].mean()) else "---"
        ari_s = f"{st_m['clean_retention_ari'].mean():.3f}" if len(st_m) > 0 else "---"
        lbl = "CL-VSSC (Smooth PI)" if m == "CL-VSSC (No-DZ PI)" else DISPLAY_P2[m]
        sig_str = f"\\textbf{{{sig_cnt}/10}}" if sig_cnt == 5 else f"{sig_cnt}/10"
        sil_str = f"\\textbf{{{sil:.3f}}}" if abs(sil - 0.516) < 5e-4 else f"{sil:.3f}"
        k_str = f"\\textbf{{{k_cnt}}}" if k_cnt == "23/40" else k_cnt
        bic_str = f"\\textbf{{{bic_map[m]}}}" if bic_map[m] == "0.001" else bic_map[m]
        w_str = f"\\textbf{{{w_m}}}" if w_m == "0.002" else w_m
        ari_str = f"\\textbf{{{ari_s}}}" if ari_s == "0.587" else ari_s
        lines.append(f"{lbl} & {sig_str} & {mn:.3f} & {md:.3f} & {sil_str} & {k_str} & {bic_str} & {w_str} & {ari_str} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TAB_DIR / "p2_tab1_controller_ablation.tex").write_text("\n".join(lines) + "\n")


def write_p2_tab2_fault_stress(stress: pd.DataFrame, stress3: pd.DataFrame) -> None:
    comb_s = pd.concat([stress, stress3], ignore_index=True)
    methods = [
        "CL-VSSC (No-DZ PI)",
        "CL-VSSC (Null-SNR PI)",
        "CL-VSSC (DZ-PI)",
        "CL-VSSC (w/o Feedback)",
        "SNF",
        "NEMO",
        "MCGC",
        "Seurat-WNN",
        "Attention-MOAE",
        "Subtype-GAN",
    ]
    mods = ["mRNA", "Methylation", "miRNA"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Three-Modality Sensor-Fault Stress Summary across BIC, KIRC, LIHC, and GBM (4 Cohorts $\times$ 3 Seeds = 12 Runs per Setting; One-Sided Paired Wilcoxon BH-FDR $q$ Across $n=36$ $\sigma=4.0$ Runs, with $q = 1.46{\times}10^{-11}$ Fault-Weight Suppression vs.\ Every Weighted Baseline)}",
        r"\label{tab:p2_fault}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{llcccccccccc}",
        r"\toprule",
        r"Fault Scenario & Metric & \shortstack{CL-\\VSSC} & \shortstack{Null-\\SNR} & \shortstack{DZ-\\PI} & \shortstack{w/o\\FB} & SNF & NEMO & MCGC & \shortstack{Seurat\\WNN} & \shortstack{Att.\\MOAE} & \shortstack{Sub.\\GAN} \\",
        r"\midrule",
    ]
    for mod in mods:
        sub = comb_s[(comb_s["modality"] == mod) & (comb_s["noise_sigma"] == 4.0)]
        ari_vals = sub.groupby("method")["clean_retention_ari"].mean().to_dict()
        surv_vals = sub.groupby("method")["neg_log10_p"].mean().to_dict()
        w_vals = sub.groupby("method")["w_corrupted"].mean().to_dict()

        lines.append(f"{mod} ($\\sigma{{=}}4.0$) & ARI retention & " + " & ".join(f"{ari_vals[m]:.3f}" for m in methods) + r" \\")
        lines.append(r" & $-\log_{10}p_{\mathrm{asymp}}$ & " + " & ".join(f"{surv_vals[m]:.3f}" for m in methods) + r" \\")
        lines.append(r" & Fault weight & " + " & ".join("---" if np.isnan(w_vals[m]) else f"{w_vals[m]:.3f}" for m in methods) + r" \\")
        lines.append(r"\midrule")

    for sig, lbl in [(2.0, "45\\%"), (4.0, "60\\%")]:
        sub = comb_s[comb_s["noise_sigma"] == sig]
        ari_vals = sub.groupby("method")["clean_retention_ari"].mean().to_dict()
        mx_ari = max(ari_vals.values())
        surv_vals = sub.groupby("method")["neg_log10_p"].mean().to_dict()
        w_vals = sub.groupby("method")["w_corrupted"].mean().to_dict()

        lines.append(
            f"All assays ($\\sigma{{=}}{sig:.1f}$, {lbl}) & ARI retention & "
            + " & ".join(f"\\textbf{{{ari_vals[m]:.3f}}}" if ari_vals[m] >= mx_ari - 1e-6 else f"{ari_vals[m]:.3f}" for m in methods) + r" \\"
        )
        lines.append(r" & $-\log_{10}p_{\mathrm{asymp}}$ & " + " & ".join(f"{surv_vals[m]:.3f}" for m in methods) + r" \\")
        lines.append(
            r" & Fault weight & "
            + " & ".join("---" if np.isnan(w_vals[m]) else (f"\\textbf{{{w_vals[m]:.3f}}}" if m in ("CL-VSSC (No-DZ PI)", "CL-VSSC (Null-SNR PI)") else f"{w_vals[m]:.3f}") for m in methods) + r" \\"
        )
        if sig == 2.0:
            lines.append(r"\midrule")
        else:
            piv_ari = sub.pivot_table(index=["cohort", "modality", "seed"], columns="method", values="clean_retention_ari")
            ref_m = "CL-VSSC (Null-SNR PI)"
            others_m = [m for m in methods if m != ref_m]
            pvals = [float(wilcoxon((piv_ari[ref_m] - piv_ari[m]).dropna(), alternative="greater")[1]) for m in others_m]
            qmap = dict(zip(others_m, _bh_fdr(pvals)))
            lines.append(
                r" & BH-FDR $q$ (vs.\ Null) & "
                + " & ".join("---" if m == ref_m else _fmt_q(qmap[m]) for m in methods) + r" \\"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TAB_DIR / "p2_tab2_fault_stress.tex").write_text("\n".join(lines) + "\n")


def write_p2_tab3_singlecell(
    sc_bench: pd.DataFrame,
    sc_fault: pd.DataFrame,
    scale: pd.DataFrame,
    sc_bench3: pd.DataFrame,
    sc_fault3: pd.DataFrame,
    sc_scale3: pd.DataFrame,
) -> None:
    comb_b = pd.concat([sc_bench, sc_bench3], ignore_index=True)
    comb_f = pd.concat([sc_fault, sc_fault3], ignore_index=True)
    ds_order = ["BMMC-Site1", "BMMC-Site2", "BMMC-10k", "BMMC-30k"]
    methods = [
        "CL-VSSC-Anchor",
        "CL-VSSC-Anchor (Local PI)",
        "CL-VSSC-Anchor (Local+Null)",
        "CL-VSSC-Anchor (Null-SNR)",
        "CL-VSSC-Anchor (DZ-PI)",
        "CL-VSSC-Anchor (w/o FB)",
        "LMVSC",
        "Seurat-WNN",
        "Attention-MOAE",
        "Subtype-GAN",
    ]
    g = comb_b.groupby(["dataset", "method"])[["nmi", "acc"]].mean()
    sf3 = comb_f[(comb_f["corrupted_modality"] == "Protein_ADT") & (comb_f["noise_sigma"] == 3.0)].groupby("method")[["gt_ari", "gt_acc", "w_corrupted"]].mean()
    sc90 = scale[scale["n_cells"] == 90261].set_index("method")
    sc90_3 = sc_scale3[sc_scale3["n_cells"] == 90261].set_index("method")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Single-Cell CITE-seq Benchmark (GSE194122 BMMC, 39--45 Cell Types), Protein ADT Fault ($\sigma=3.0$), and 90{,}261-Cell Scaling ($^\dagger$L4 GPU)}",
        r"\label{tab:p2_singlecell}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & \shortstack{BMMC-Site1\\($32{,}004$, NMI/ACC)} & \shortstack{BMMC-Site2\\($25{,}161$, NMI/ACC)} & \shortstack{BMMC-10k\\(NMI/ACC)} & \shortstack{BMMC-30k\\(NMI/ACC)} & \shortstack{BMMC-10k ADT Fault\\($\sigma=3$, ARI/ACC)} & \shortstack{ADT Fault\\$w_{\mathrm{ADT}}$} & \shortstack{90{,}261 Cells\\Time (s) / MB} \\",
        r"\midrule",
    ]
    for m in methods:
        cells = [f"{g.loc[(ds, m), 'nmi']:.3f} / {g.loc[(ds, m), 'acc']:.3f}" for ds in ds_order]
        f_ari = sf3.loc[m, "gt_ari"]
        f_acc = sf3.loc[m, "gt_acc"]
        f_w = sf3.loc[m, "w_corrupted"]
        w_str = "---" if np.isnan(f_w) else f"{f_w:.3f}"
        f_str = f"\\textbf{{{f_ari:.3f}}} / \\textbf{{{f_acc:.3f}}}" if m == "CL-VSSC-Anchor" else f"{f_ari:.3f} / {f_acc:.3f}"
        if m in sc90.index:
            s_str = f"{sc90.loc[m, 'runtime_sec']:.2f} / {sc90.loc[m, 'peak_vram_mb']:,.0f}"
        elif m in sc90_3.index:
            s_str = f"{sc90_3.loc[m, 'runtime_sec']:.2f}$^\\dagger$ / {sc90_3.loc[m, 'peak_vram_mb']:,.0f}"
        else:
            s_str = "---"
        lines.append(f"{DISPLAY_P2[m]} & " + " & ".join(cells) + f" & {f_str} & {w_str} & {s_str} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TAB_DIR / "p2_tab3_singlecell.tex").write_text("\n".join(lines) + "\n")


def make_p2_fig6_stability_phase_200k(
    phase_df: pd.DataFrame,
    phase_trajs: dict,
    scale_v4: pd.DataFrame,
    heal_df: pd.DataFrame,
    heal_trajs: dict,
    v4_env: dict,
) -> None:
    """4-panel Fig. 6: (a) (K_p, kappa_I) stability phase diagram, (b) representative trajectories + dead-zone shift,
    (c) 10k->300k linear scaling, and (d) N=200k mid-stream step-fault self-healing."""
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 7.25))

    # Panel (a): (K_p, kappa_I) Closed-Loop Stability Phase Diagram (BIC, delta_dz=0)
    ax = axs[0, 0]
    sub0 = phase_df[(phase_df["cohort"] == "BIC") & (phase_df["dead_zone"] == 0.0)].copy()
    kps = sorted(sub0["k_p"].unique())
    krs = sorted(sub0["kappa_i"].unique())
    regime_order = [
        "Under-Actuated / Stalled",
        "Convergent",
        "Oscillatory",
        "Collapse / Unstable",
    ]
    regime_code = {r: idx for idx, r in enumerate(regime_order)}
    regime_abbr = {
        "Under-Actuated / Stalled": "U",
        "Convergent": "C",
        "Oscillatory": "O",
        "Collapse / Unstable": "X",
    }
    regime_colors = ["#56B4E9", "#009E73", "#E69F00", "#D55E00"]
    cmap = mpl.colors.ListedColormap(regime_colors)
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    piv_reg = sub0.pivot(index="kappa_i", columns="k_p", values="regime").loc[krs, kps]
    piv_set = sub0.pivot(index="kappa_i", columns="k_p", values="settling_iter").loc[krs, kps]
    mat = np.vectorize(regime_code.get)(piv_reg.to_numpy())
    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", origin="lower", alpha=0.86)

    ax.set_xticks(np.arange(len(kps)))
    ax.set_xticklabels([f"{k:g}" for k in kps], rotation=30, fontsize=7.0)
    ax.set_yticks(np.arange(len(krs)))
    ax.set_yticklabels([f"{k:g}" for k in krs], fontsize=7.0)
    ax.set_xlabel(r"Proportional mirror gain $K_p = \eta_w / \tau$")
    ax.set_ylabel(r"Integral-to-proportional ratio $\kappa_I = K_i / K_p$")
    ax.set_title(r"(a) Closed-Loop PI Stability Phase Diagram (BIC, $\delta_{\mathrm{dz}}=0$)")

    for i, kr in enumerate(krs):
        for j, kp in enumerate(kps):
            reg = str(piv_reg.iloc[i, j])
            s_it = int(piv_set.iloc[i, j])
            lbl = f"C({s_it})" if reg == "Convergent" else regime_abbr[reg]
            is_def = abs(kp - 1.5) < 1e-6 and abs(kr - 0.35) < 1e-6
            ax.text(
                j, i, lbl, ha="center", va="center",
                fontsize=5.8 if reg == "Convergent" else 6.4,
                fontweight="bold" if is_def else "normal",
                color="white" if reg in ("Convergent", "Collapse / Unstable") else "#111111",
            )
            if is_def:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#FFFFFF", lw=2.0))
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#111111", lw=1.0, ls="--"))

    legend_patches = [
        mpl.patches.Patch(facecolor="#009E73", label="Convergent (C, settle it.)"),
        mpl.patches.Patch(facecolor="#E69F00", label="Oscillatory (O)"),
        mpl.patches.Patch(facecolor="#56B4E9", label="Under-Actuated (U)"),
        mpl.patches.Patch(facecolor="#D55E00", label="Collapse / Unstable (X)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", ncol=2, fontsize=5.9, frameon=True, framealpha=0.94)

    # Panel (b): Representative Fault Weight Trajectories & Dead-Zone Margin Shift Inset
    ax = axs[0, 1]
    traj_specs = [
        ("Convergent_Default", r"Convergent ($K_p{=}1.5, \kappa_I{=}0.35, \delta_{\mathrm{dz}}{=}0$)", "#009E73", "-", 2.1),
        ("Stalled_DeadZone", r"Dead-Zone ($K_p{=}1.5, \kappa_I{=}0.35, \delta_{\mathrm{dz}}{=}0.25$)", "#5E4FA2", "--", 1.8),
        ("UnderActuated_LowKp", r"Under-Actuated ($K_p{=}0.25, \kappa_I{=}0$)", "#56B4E9", "-.", 1.9),
        ("HighKi_Ringing", r"High-$K_i$ Overshoot ($K_p{=}0.25, \kappa_I{=}10, \Delta\mathcal{V}{>}0$)", "#E69F00", ":", 1.9),
        ("Collapse_HighKp", r"Over-Gain Collapse ($K_p{=}40, \kappa_I{=}10$)", "#D55E00", "-", 1.7),
    ]
    for key, lbl, col, ls, lw in traj_specs:
        w_hist = np.array(phase_trajs[key]["fault_weights_history"])[:, 0]
        ax.plot(np.arange(len(w_hist)), w_hist, label=lbl, color=col, ls=ls, lw=lw)

    ax.axhline(0.05, color="#444444", ls=":", lw=1.0)
    ax.text(1.0, 0.056, r"$5\%$ fault threshold", fontsize=6.4, color="#333333")
    ax.set_xlim(0, 45)
    ax.set_ylim(-0.015, 0.435)
    ax.set_xlabel(r"Closed-loop iteration $t$")
    ax.set_ylabel(r"Corrupted mRNA weight $w_{\mathrm{mRNA}}(t)$")
    ax.set_title(r"(b) Gain Regimes & Dead-Zone $\delta_{\mathrm{dz}}$ Margin Shift")
    ax.grid(alpha=0.20, ls=":")
    ax.legend(loc="upper right", fontsize=5.6, frameon=True, framealpha=0.95)
    ax_b = ax

    # Panel (c): Ultra-Large-Scale Linear Complexity (10k -> 300k cells on A100-80GB)
    ax = axs[1, 0]
    sm = scale_v4[scale_v4["method"] == "CL-VSSC-Anchor (Smooth PI)"].sort_values("n_cells")
    lc = scale_v4[scale_v4["method"] == "CL-VSSC-Anchor (Local PI)"].sort_values("n_cells")
    ns_k = sm["n_cells"].to_numpy() / 1000.0
    l1 = ax.loglog(ns_k, sm["runtime_sec"], "-o", color="#0072B2", lw=2.0, markersize=4.8, label="Smooth PI Time (s)")
    l2 = ax.loglog(ns_k, lc["runtime_sec"], "-s", color="#9E0142", lw=1.9, markersize=4.6, label="Local PI Time (s)")
    ax.axvline(90.261, color="#555555", ls="--", lw=1.1)
    ax.text(85.0, 11.2, "Real\n90.3k", ha="right", va="bottom", fontsize=6.2, color="#444444")
    ax.text(97.0, 11.2, "Expanded\n150k–300k", ha="left", va="bottom", fontsize=6.2, color="#444444")
    ax.set_xlabel(r"Number of cells $N$ ($\times 10^3$, log scale)")
    ax.set_ylabel("A100 wall-clock runtime (s, log)", color="#003B5C")
    ax.set_ylim(10.0, 75.0)
    ax.set_xticks([10, 25, 50, 90.261, 150, 200, 300])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xticklabels(["10", "25", "50", "90.3", "150", "200", "300"], fontsize=7.0)

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    l3 = ax_r.loglog(ns_k, sm["peak_vram_mb"] / 1024.0, "--^", color="#009E73", lw=1.8, markersize=4.6, label="Smooth PI VRAM (GB)")
    l4 = ax_r.loglog(ns_k, lc["peak_vram_mb"] / 1024.0, ":D", color="#D55E00", lw=1.8, markersize=4.4, label="Local PI VRAM (GB)")
    ax_r.set_ylabel("Peak A100 VRAM (GB, log)", color="#005A40")
    ax_r.set_ylim(0.4, 30.0)

    vram_exp = float(v4_env["v42_summary"]["scale_summary"]["CL-VSSC-Anchor (Smooth PI)"]["vram_power_alpha"])
    ax.set_title(rf"(c) Scaling $10\mathrm{{k}} \rightarrow 300\mathrm{{k}}$ ($\mathcal{{O}}(N^{{{vram_exp:.3f}}})$ VRAM)")
    ax.grid(True, which="both", alpha=0.20, ls=":")
    lines = l1 + l2 + l3 + l4
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", ncol=2, fontsize=5.8, frameon=True, framealpha=0.94)

    # Panel (d): N = 200,000 Mid-Stream Step-Fault Self-Healing (inject_iter = 10)
    ax = axs[1, 1]
    heal_specs = [
        ("CL-VSSC-Anchor (Smooth PI)", "Smooth PI", "#0072B2", "-", 2.1),
        ("CL-VSSC-Anchor (Null-SNR PI)", "Null-SNR PI", "#9E0142", "-", 2.1),
        ("CL-VSSC-Anchor (Local PI)", "Local PI", "#01665E", "--", 1.9),
        ("CL-VSSC-Anchor (DZ-PI)", r"DZ-PI ($\delta_{\mathrm{dz}}{=}0.14$)", "#009E73", "-.", 1.8),
        ("CL-VSSC-Anchor (w/o FB)", "w/o Feedback", "#4D4D4D", ":", 1.8),
    ]
    step_df = heal_df[heal_df["scenario"] == "StepFault_t10"].set_index("method")
    for m_key, lbl, col, ls, lw in heal_specs:
        w_adt = np.array(heal_trajs[f"StepFault_t10__{m_key}"]["w_adt_history"])
        ari_v = float(step_df.loc[m_key, "gt_ari"])
        w_end = float(step_df.loc[m_key, "w_adt_final"])
        h_it = int(step_df.loc[m_key, "heal_iters_90"])
        h_s = step_df.loc[m_key, "heal_wall_sec_90"]
        if h_it > 0 and m_key != "CL-VSSC-Anchor (DZ-PI)":
            full_lbl = f"{lbl}: {w_end:.3f} ({h_it} it/{float(h_s):.1f}s, ARI {ari_v:.3f})"
        elif m_key == "CL-VSSC-Anchor (DZ-PI)":
            full_lbl = f"{lbl}: stalled {w_end:.3f} (ARI {ari_v:.3f})"
        else:
            full_lbl = f"{lbl}: {w_end:.3f} (ARI {ari_v:.3f})"
        ax.plot(np.arange(len(w_adt)), w_adt, label=full_lbl, color=col, ls=ls, lw=lw)

    ax.axvline(10, color="#B22222", ls="--", lw=1.3)
    ax.axvspan(10, 17, color="#FFF3E0", alpha=0.55)
    ax.text(10.3, 0.90, "ADT step fault\n($t{=}10,\\;\\sigma{=}3, 50\\%$ drop)", fontsize=6.4, color="#8C1D18", va="top")
    ax.text(13.8, 0.61, "7-iter\nself-heal\n(3.8–6.0s)", ha="center", va="center", fontsize=6.2, color="#8C3B00")
    ax.set_xlim(0, 30)
    ax.set_ylim(0.18, 0.94)
    ax.set_xlabel(r"Closed-loop iteration $t$ ($N=200{,}000$ cells)")
    ax.set_ylabel(r"Corrupted ADT weight $w_{\mathrm{ADT}}(t)$")
    ax.set_title(r"(d) $N=200{,}000$ Mid-Stream Step-Fault Self-Healing")
    ax.grid(alpha=0.20, ls=":")
    ax.legend(loc="lower left", fontsize=5.8, frameon=True, framealpha=0.94)

    fig.tight_layout(h_pad=2.1, w_pad=2.2)

    # Inset AFTER tight_layout: Dead-Zone Regime Distribution across common 7x7 grid (BIC + KIRC = 98 pts/bar)
    kp_dz = [0.5, 1.0, 1.5, 2.5, 6.0, 15.0, 40.0]
    kr_dz = [0.0, 0.2, 0.35, 1.0, 2.5, 6.0, 10.0]
    sub_dz = phase_df[phase_df["k_p"].isin(kp_dz) & phase_df["kappa_i"].isin(kr_dz)]
    dz_vals = sorted(sub_dz["dead_zone"].unique())
    dz_counts = sub_dz.groupby(["dead_zone", "regime"]).size().unstack(fill_value=0).reindex(index=dz_vals, columns=regime_order, fill_value=0)
    ax_in = inset_axes(ax_b, width="100%", height="100%", bbox_to_anchor=(0.45, 0.275, 0.51, 0.215), bbox_transform=ax_b.transAxes, borderpad=0)
    bottom = np.zeros(len(dz_vals))
    x_dz = np.arange(len(dz_vals))
    for reg_name, reg_col in zip(regime_order, regime_colors):
        vals = dz_counts[reg_name].to_numpy()
        ax_in.bar(x_dz, vals, bottom=bottom, width=0.58, color=reg_col, edgecolor="white", linewidth=0.4)
        bottom += vals
    ax_in.set_xticks(x_dz)
    ax_in.set_xticklabels([f"{d:.2f}" for d in dz_vals], fontsize=5.8)
    ax_in.tick_params(axis="y", labelsize=5.8)
    ax_in.set_xlabel(r"Dead-zone $\delta_{\mathrm{dz}}$", fontsize=6.0, labelpad=1)
    ax_in.set_ylabel("Grid pts", fontsize=6.0, labelpad=1)
    ax_in.set_title(r"$\delta_{\mathrm{dz}}$ Margin Shift (98 pts/bar)", fontsize=6.2, pad=2)
    ax_in.set_ylim(0, 104)

    fig.savefig(FIG_DIR / "p2_fig6_stability_phase_200k.pdf", bbox_inches="tight")
    plt.close(fig)


def write_p2_supplementary_tables(
    summary: pd.DataFrame,
    bench: pd.DataFrame,
    stress: pd.DataFrame,
    tcga3: pd.DataFrame,
    stress3: pd.DataFrame,
    sc_bench: pd.DataFrame,
    sc_fault: pd.DataFrame,
    sc_bench3: pd.DataFrame,
    sc_fault3: pd.DataFrame,
) -> None:
    """Generate Supplementary Tables S-I to S-IV programmatically from results_v2/v3."""
    # Table S-I: Notation, Default Parameters, and Asymptotic Complexity Comparison
    t1_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Methodological Mechanism, Asymptotic Complexity, and Fixed Control Parameters}",
        r"\label{tab:p2_stab1}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Method & Weighting / Fusion Mechanism & Time Complexity & Memory & Control Guarantees / Role \\",
        r"\midrule",
        r"SNF \cite{wang2014snf} & Open-loop cross-diffusion & $\mathcal{O}(T_{\mathrm{diff}} n^3)$ & $\mathcal{O}(V n^2)$ & Fixed diffusion; no partition feedback \\",
        r"NEMO \cite{rappoport2019nemo} & Open-loop partial averaging & $\mathcal{O}(n^3)$ & $\mathcal{O}(V n^2)$ & No adaptive modal rejection \\",
        r"MCGC \cite{zhan2019mcgc} & Inverse graph discrepancy & $\mathcal{O}(T n^3)$ & $\mathcal{O}(V n^2)$ & Heuristic weights ($w_{\mathrm{corr}}{=}0.190$ at $\sigma{=}4$) \\",
        r"Subtype-GAN \cite{yang2021subtypegan} & Deep WGAN + GMM latent & $\mathcal{O}(E n d_{\mathrm{tot}} h)$ & $\mathcal{O}(B d_{\mathrm{tot}})$ & Amortized encoder; noise-sensitive \\",
        r"Attention-MOAE \cite{moon2022moma} & Static attention autoencoder & $\mathcal{O}(E n d_{\mathrm{tot}} h)$ & $\mathcal{O}(B d_{\mathrm{tot}})$ & High-variance lock-in ($w_{\mathrm{ADT}}{=}0.866$) \\",
        r"Seurat-WNN \cite{hao2021seuratv4} & Local $k$-NN ratio weights & $\mathcal{O}(V n^2 \log n)$ & $\mathcal{O}(V n k)$ & No integral memory or dual pruning \\",
        r"LMVSC \cite{kang2020lmvsc} & Static anchor subspace fusion & $\mathcal{O}(n m^2 + m^3)$ & $\mathcal{O}(V n m)$ & Linear-time open loop; uniform fusion \\",
        r"\midrule",
        r"CL-VSSC (Dense) & Stiefel + PI + dual $\mu(t)$ & $\mathcal{O}(T(n^3 + \sum_v n d_v r))$ & $\mathcal{O}(V n^2)$ & Exact ISS / DK ($T{=}45$, $t_{\mathrm{warm}}{=}15$) \\",
        r"CL-VSSC-Anchor & Gram + Local/Null PI + $\mu(t)$ & $\mathcal{O}(T(nm^2 + \sum_v n d_v r))$ & $\mathcal{O}(Vnm + \sum_v nd_v)$ & Exact Prop.~1 ($\mathcal{O}(N^{0.996})$ VRAM) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (TAB_DIR / "p2_stab1_notation_complexity.tex").write_text("\n".join(t1_lines) + "\n")

    # Table S-II: 10-TCGA Cohort Metadata + Per-Cohort -log10(p_perm)
    comb_b = pd.concat([bench, tcga3], ignore_index=True)
    comb_b = comb_b[~comb_b["method"].str.startswith("OpenLoop_")].copy()
    piv_p = comb_b.pivot_table(index="cohort", columns="method", values="neg_log10_perm_p")
    piv_p_raw = comb_b.pivot_table(index="cohort", columns="method", values="perm_p")
    m_cols = [
        ("CL-VSSC (No-DZ PI)", r"\shortstack{Smooth\\PI}"),
        ("CL-VSSC (Null-SNR PI)", r"\shortstack{Null-\\SNR}"),
        ("CL-VSSC (DZ-PI)", r"\shortstack{DZ-\\PI}"),
        ("CL-VSSC (P-only)", r"\shortstack{P-\\only}"),
        ("CL-VSSC (w/o Feedback)", r"\shortstack{w/o\\FB}"),
        ("CL-VSSC (w/o SpecCtrl)", r"\shortstack{w/o\\Spec}"),
        ("SNF", "SNF"),
        ("NEMO", "NEMO"),
        ("MCGC", "MCGC"),
        ("Seurat-WNN", r"\shortstack{Seurat\\WNN}"),
        ("Attention-MOAE", r"\shortstack{Att.\\MOAE}"),
        ("Subtype-GAN", r"\shortstack{Sub.\\GAN}"),
    ]
    sum_idx = summary.set_index("cohort")
    t2_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-Cohort 10-TCGA Metadata ($N=3{,}113$, $1{,}235$ Events) and 5,000-Permutation Survival $-\log_{10}p_{\mathrm{perm}}$}",
        r"\label{tab:p2_stab2}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\begin{tabular}{lcc" + "c" * len(m_cols) + "}",
        r"\toprule",
        r"Cohort & $n$ / Events & $c$ & " + " & ".join(lbl for _, lbl in m_cols) + r" \\",
        r"\midrule",
    ]
    for coh in COHORTS:
        row_s = sum_idx.loc[coh]
        ne_str = f"{int(row_s['n_samples'])} / {int(row_s['n_events'])}"
        k_str = f"{int(row_s['canonical_k'])}"
        vals = [float(piv_p.loc[coh, mk]) for mk, _ in m_cols]
        mx = max(vals)
        cells = [
            f"\\textbf{{{v:.3f}}}" if abs(v - mx) < 1e-6 else f"{v:.3f}"
            for v in vals
        ]
        t2_lines.append(f"{coh} & {ne_str} & {k_str} & " + " & ".join(cells) + r" \\")
    t2_lines.append(r"\midrule")
    sig_row = [f"{int((piv_p_raw[mk] < 0.05).sum())}/10" for mk, _ in m_cols]
    med_row = [f"{float(piv_p[mk].median()):.3f}" for mk, _ in m_cols]
    mean_row = [f"{float(piv_p[mk].mean()):.3f}" for mk, _ in m_cols]
    t2_lines.append(r"Sig.\ ($p{<}0.05$) & --- & --- & " + " & ".join(sig_row) + r" \\")
    t2_lines.append(r"Median & --- & --- & " + " & ".join(med_row) + r" \\")
    t2_lines.append(r"Mean & 3,113 / 1,235 & --- & " + " & ".join(mean_row) + r" \\")
    t2_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TAB_DIR / "p2_stab2_tcga_cohorts.tex").write_text("\n".join(t2_lines) + "\n")

    # Table S-III: Paired Wilcoxon & BH-FDR Ledger across 1,560-run stress
    comb_s = pd.concat([stress, stress3], ignore_index=True)
    comb_s = comb_s[~comb_s["method"].str.startswith("OpenLoop_")].copy()
    ref_m = "CL-VSSC (Null-SNR PI)"
    comps = [
        ("CL-VSSC (No-DZ PI)", "CL-VSSC (Smooth PI)"),
        ("CL-VSSC (DZ-PI)", "CL-VSSC (DZ-PI)"),
        ("CL-VSSC (w/o Feedback)", "CL-VSSC (w/o FB)"),
        ("SNF", "SNF"),
        ("NEMO", "NEMO"),
        ("MCGC", "MCGC"),
        ("Seurat-WNN", "Seurat-WNN"),
        ("Attention-MOAE", "Attention-MOAE"),
        ("Subtype-GAN", "Subtype-GAN"),
    ]
    slices = [
        ("s4", None, [4.0]),
        ("s24", None, [2.0, 4.0]),
        ("sall", None, [0.5, 1.0, 2.0, 4.0]),
        ("sec4", ["mRNA", "miRNA"], [4.0]),
    ]
    stats_out: Dict[str, Dict[str, tuple]] = {}
    for skey, mods, sigs in slices:
        sub = comb_s[comb_s["noise_sigma"].isin(sigs)]
        if mods is not None:
            sub = sub[sub["modality"].isin(mods)]
        piv = sub.pivot_table(index=["cohort", "modality", "noise_sigma", "seed"], columns="method", values="clean_retention_ari")
        pvals = []
        tmp = []
        for mk, _ in comps:
            diff = (piv[ref_m] - piv[mk]).dropna()
            _, pv = wilcoxon(diff, alternative="greater")
            pvals.append(float(pv))
            tmp.append((mk, float(diff.mean()), int((diff > 1e-9).sum()), len(diff), float(pv)))
        qvals = _bh_fdr(pvals)
        stats_out[skey] = {mk: (dm, wins, tot, pv, qv) for (mk, dm, wins, tot, pv), qv in zip(tmp, qvals)}

    t3_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete Paired One-Sided Wilcoxon Signed-Rank $p$ and BH-FDR $q$ Comparison of Null-SNR PI Fault Retention ARI Across Stress Slices}",
        r"\label{tab:p2_stab3}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method (vs.\ Null-SNR) & \shortstack{$\sigma{=}4.0$ ($n{=}36$)\\$\Delta\mathrm{ARI}$ / W--T} & \shortstack{$\sigma{=}4.0$ ($n{=}36$)\\Wilcoxon $p$ / BH $q$} & \shortstack{$\sigma{\in}\{2,4\}$ ($n{=}72$)\\$\Delta\mathrm{ARI}$ / BH $q$} & \shortstack{All $\sigma{>}0$ ($n{=}144$)\\$\Delta\mathrm{ARI}$ / BH $q$} & \shortstack{mRNA+miRNA ($n{=}24$)\\$\Delta\mathrm{ARI}$ / BH $q$} \\",
        r"\midrule",
    ]
    for mk, lbl in comps:
        d4, w4, n4, p4, q4 = stats_out["s4"][mk]
        d24, _, _, _, q24 = stats_out["s24"][mk]
        dall, _, _, _, qall = stats_out["sall"][mk]
        dsec, _, _, _, qsec = stats_out["sec4"][mk]
        t3_lines.append(
            f"{lbl} & ${d4:+.3f}$ ({w4}/{n4}) & {_fmt_q(p4)} / {_fmt_q(q4)} & "
            f"${d24:+.3f}$ / {_fmt_q(q24)} & ${dall:+.3f}$ / {_fmt_q(qall)} & ${dsec:+.3f}$ / {_fmt_q(qsec)} \\\\"
        )
    t3_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TAB_DIR / "p2_stab3_wilcoxon_ledger.tex").write_text("\n".join(t3_lines) + "\n")

    # Table S-IV: Multi-site CITE-seq Clean ARI (mean+-SD) & Bidirectional scRNA/ADT Faults
    comb_scb = pd.concat([sc_bench, sc_bench3], ignore_index=True)
    comb_scf = pd.concat([sc_fault, sc_fault3], ignore_index=True)
    sc_methods = [
        "CL-VSSC-Anchor",
        "CL-VSSC-Anchor (Local PI)",
        "CL-VSSC-Anchor (Local+Null)",
        "CL-VSSC-Anchor (Null-SNR)",
        "CL-VSSC-Anchor (DZ-PI)",
        "CL-VSSC-Anchor (w/o FB)",
        "LMVSC",
        "Seurat-WNN",
        "Attention-MOAE",
        "Subtype-GAN",
    ]
    ds_ord = ["BMMC-Site1", "BMMC-Site2", "BMMC-10k", "BMMC-30k"]
    g_sc = comb_scb.groupby(["dataset", "method"])["ari"].agg(["mean", "std"])
    g_flt = comb_scf.groupby(["corrupted_modality", "noise_sigma", "method"])[["gt_ari", "w_corrupted"]].mean()
    t4_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CITE-seq Multi-Site Clean ARI ($\mathrm{Mean} \pm \mathrm{SD}$, 3 Seeds) and Bidirectional \texttt{BMMC-10k} scRNA vs.\ Protein-ADT Fault Isolation ($\mathrm{ARI} / w_{\mathrm{corr}}$)}",
        r"\label{tab:p2_stab4}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.13}",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Method & Site1 ARI & Site2 ARI & 10k ARI & 30k ARI & \shortstack{scRNA $\sigma{=}1.5$\\ARI / $w_{\mathrm{RNA}}$} & \shortstack{scRNA $\sigma{=}3.0$\\ARI / $w_{\mathrm{RNA}}$} & \shortstack{ADT $\sigma{=}1.5$\\ARI / $w_{\mathrm{ADT}}$} & \shortstack{ADT $\sigma{=}3.0$\\ARI / $w_{\mathrm{ADT}}$} \\",
        r"\midrule",
    ]
    for m in sc_methods:
        c_cells = [f"{g_sc.loc[(ds, m), 'mean']:.3f}$\\pm${g_sc.loc[(ds, m), 'std']:.3f}" for ds in ds_ord]
        f_cells = []
        for mod, sig in [("scRNA", 1.5), ("scRNA", 3.0), ("Protein_ADT", 1.5), ("Protein_ADT", 3.0)]:
            ari_v = float(g_flt.loc[(mod, sig, m), "gt_ari"])
            w_v = float(g_flt.loc[(mod, sig, m), "w_corrupted"])
            w_s = "---" if np.isnan(w_v) else f"{w_v:.3f}"
            f_cells.append(f"{ari_v:.3f} / {w_s}")
        t4_lines.append(f"{DISPLAY_P2[m]} & " + " & ".join(c_cells) + " & " + " & ".join(f_cells) + r" \\")
    t4_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TAB_DIR / "p2_stab4_cite_bidirectional.tex").write_text("\n".join(t4_lines) + "\n")


def make_graphical_abstract(
    stress: pd.DataFrame,
    stress3: pd.DataFrame,
    heal_df: pd.DataFrame,
    heal_trajs: dict,
    v4_env: dict,
) -> None:
    """Create 300-DPI IEEE / Elsevier Graphical Abstract (PDF + PNG) summarizing closed-loop architecture and empirical proof."""
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.6, 4.75), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.40)

    # Left column: full-height closed-loop control schematic
    ax_diag = fig.add_subplot(gs[:, 0])
    ax_diag.set_xlim(0, 100)
    ax_diag.set_ylim(0, 56)
    ax_diag.axis("off")
    ax_diag.set_title("Closed-Loop Multi-Modal Spectral Control (CL-VSSC / CL-VSSC-Anchor)", fontsize=9.8, fontweight="bold", pad=6)

    draw_box(ax_diag, 2, 33, 27, 17, "Stiefel Plant", "Orthogonal $Z,\\, U^{(v)}$\nOperator-norm $\\sigma_{\\max}^2$\nDense $A^{(v)}$ / Anchor $\\tilde{B}^{(v)}$", fc="#EDF5FB", ec="#0072B2")
    draw_box(ax_diag, 37, 33, 30, 17, "Dual Eigengap Actuator", "Anchor Gram $\\mathcal{O}(nm^2)$\nKy Fan pruning $\\mu(t)$\nOpens gap $\\Delta\\lambda_c$ (ISS)", fc="#EEF8F3", ec="#009E73", title_color="#005A40")
    draw_box(ax_diag, 74, 34, 24, 15, "Controlled Subtypes", "Eigenvectors $F(T)$\nDavis--Kahan bound\n90k--300k linear VRAM", fc="#F4F6F8", ec="#444444", title_color="#222222")
    draw_box(ax_diag, 52, 5, 44, 17, "Cut & Null-SNR Fault Sensor", "Intrinsic/consensus cuts + DOF residual $e_v(t)$\nNull-SNR column shuffle + double-centered $\\tilde{\\delta}^{\\mathrm{loc}}$", fc="#FFF6ED", ec="#D55E00", title_color="#8C3B00")
    draw_box(ax_diag, 3, 5, 42, 17, "Anti-Windup PI Simplex Controller", "Leaky integrator $I_v$ (BIBO) + $\\delta_{\\mathrm{dz}}{=}0$\nExponential fault rejection $e^{-(1+\\kappa_I)\\Delta e/\\tau}$", fc="#FDF3F8", ec="#CC79A7", title_color="#782857")

    draw_arrow(ax_diag, 29.3, 41.5, 36.7, 41.5, color="#0072B2")
    draw_arrow(ax_diag, 67.3, 41.5, 73.7, 41.5, color="#009E73")
    draw_arrow(ax_diag, 64.0, 32.7, 64.0, 22.4, label=r"$F(t), \tilde{r}_v$", lx=71.0, ly=27.6, color="#D55E00")
    draw_arrow(ax_diag, 51.7, 13.5, 45.3, 13.5, label=r"$e_v$", ly=16.0, color="#8C3B00")
    draw_arrow(ax_diag, 15.5, 22.3, 15.5, 32.7, label=r"$w(t)$", lx=10.5, ly=27.6, color="#CC79A7")
    draw_arrow(ax_diag, 34.0, 22.3, 44.0, 32.7, label=r"$w, W_{\mathrm{loc}}$", lx=45.0, ly=26.5, color="#CC79A7")

    # Top-right: 1,560-run TCGA sigma=4.0 fault weight suppression
    ax_bar = fig.add_subplot(gs[0, 1])
    comb_s = pd.concat([stress, stress3], ignore_index=True)
    s4 = comb_s[comb_s["noise_sigma"] == 4.0].groupby("method")["w_corrupted"].mean()
    bars = [
        ("Att.-MOAE", float(s4["Attention-MOAE"]), "#9467BD"),
        ("Seurat-WNN", float(s4["Seurat-WNN"]), "#8C564B"),
        ("MCGC", float(s4["MCGC"]), "#E69F00"),
        ("DZ-PI", float(s4["CL-VSSC (DZ-PI)"]), "#009E73"),
        ("Smooth PI", float(s4["CL-VSSC (No-DZ PI)"]), "#0072B2"),
        ("Null-SNR PI", float(s4["CL-VSSC (Null-SNR PI)"]), "#9E0142"),
    ]
    y_pos = np.arange(len(bars))
    ax_bar.barh(y_pos, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([b[0] for b in bars], fontsize=7.2)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 0.39)
    ax_bar.set_xlabel(r"Corrupted modality weight $w_{\mathrm{corr}}$ at $\sigma=4.0$ ($60\%$ drop)", fontsize=7.6)
    ax_bar.set_title(r"1,560-Run Sensor Rejection ($q=1.46\times 10^{-11}$)", fontsize=8.8, fontweight="bold")
    for idx, (_, val, _) in enumerate(bars):
        ax_bar.text(val + 0.006, idx, f"{val:.3f}", va="center", fontsize=7.0, fontweight="bold" if idx >= 4 else "normal")
    ax_bar.grid(axis="x", alpha=0.2, ls=":")

    # Bottom-right: N=200,000 mid-stream step-fault self-healing
    ax_heal = fig.add_subplot(gs[1, 1])
    for m_key, lbl, col, ls in [
        ("CL-VSSC-Anchor (Smooth PI)", "Smooth PI (7 it / 3.82s)", "#0072B2", "-"),
        ("CL-VSSC-Anchor (Null-SNR PI)", "Null-SNR (7 it / 5.20s)", "#9E0142", "-"),
        ("CL-VSSC-Anchor (Local PI)", "Local PI (7 it / 6.03s)", "#01665E", "--"),
        ("CL-VSSC-Anchor (DZ-PI)", "DZ-PI (stalled 0.500)", "#009E73", "-."),
    ]:
        w_hist = np.array(heal_trajs[f"StepFault_t10__{m_key}"]["w_adt_history"])
        ax_heal.plot(np.arange(len(w_hist)), w_hist, label=lbl, color=col, ls=ls, lw=1.8)
    ax_heal.axvline(10, color="#B22222", ls="--", lw=1.1)
    ax_heal.set_xlim(0, 30)
    ax_heal.set_ylim(0.20, 0.92)
    ax_heal.set_xlabel(r"Closed-loop iteration $t$ ($N=200{,}000$ cells)", fontsize=7.6)
    ax_heal.set_ylabel(r"Corrupted $w_{\mathrm{ADT}}(t)$", fontsize=7.6)
    vram_a = float(v4_env["v42_summary"]["scale_summary"]["CL-VSSC-Anchor (Smooth PI)"]["vram_power_alpha"])
    ax_heal.set_title(rf"200k Step-Fault Self-Healing ($\mathcal{{O}}(N^{{{vram_a:.3f}}})$ VRAM)", fontsize=8.8, fontweight="bold")
    ax_heal.legend(loc="lower left", fontsize=6.2, frameon=True, framealpha=0.94)
    ax_heal.grid(alpha=0.2, ls=":")

    fig.savefig(SUB_DIR / "Graphical_Abstract.pdf", bbox_inches="tight")
    fig.savefig(SUB_DIR / "Graphical_Abstract.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RES2_DIR / "cohort_summary_v2.csv")
    bench = pd.read_csv(RES2_DIR / "tcga_benchmark_v2.csv")
    ksens = pd.read_csv(RES2_DIR / "k_sensitivity_v2.csv")
    stress = pd.read_csv(RES2_DIR / "fault_tolerance_v2.csv")
    traj = json.loads((RES2_DIR / "diagnostics_trajectories_v2.json").read_text())
    sc_bench = pd.read_csv(RES2_DIR / "singlecell_benchmark_v2.csv")
    sc_fault = pd.read_csv(RES2_DIR / "sc_fault_v2.csv")
    scale = pd.read_csv(RES2_DIR / "scalability_v2.csv")

    tcga3 = pd.read_csv(RES3_DIR / "tcga_null_snr_v3.csv")
    ksens3 = pd.read_csv(RES3_DIR / "k_sens_null_snr_v3.csv")
    stress3 = pd.read_csv(RES3_DIR / "fault_null_snr_v3.csv")
    traj3 = json.loads((RES3_DIR / "traj_null_snr_v3.json").read_text())
    sc_bench3 = pd.read_csv(RES3_DIR / "sc_local_pi_v3.csv")
    sc_fault3 = pd.read_csv(RES3_DIR / "sc_fault_local_pi_v3.csv")
    sc_scale3 = pd.read_csv(RES3_DIR / "sc_scale_local_pi_v3.csv")
    sens_pi = pd.read_csv(RES3_DIR / "sensitivity_pi_grid_v3.csv")
    sens_mu = pd.read_csv(RES3_DIR / "sensitivity_mu_v3.csv")
    sens_anc = pd.read_csv(RES3_DIR / "sensitivity_anchors_v3.csv")

    phase_df = pd.read_csv(RES4_DIR / "pi_stability_phase_v4.csv")
    phase_trajs = json.loads((RES4_DIR / "pi_phase_trajectories_v4.json").read_text())
    scale_v4 = pd.read_csv(RES4_DIR / "scale_200k_v4.csv")
    heal_df = pd.read_csv(RES4_DIR / "fault_self_healing_200k_v4.csv")
    heal_trajs = json.loads((RES4_DIR / "fault_self_healing_200k_traj_v4.json").read_text())
    v4_env = json.loads((RES4_DIR / "v4_env.json").read_text())

    make_p2_fig1_control_loop()
    make_p2_fig2_dynamics(traj, traj3)
    make_p2_fig3_fault_tolerance(stress, stress3)
    make_p2_fig4_singlecell_scale(sc_bench, sc_fault, scale, sc_bench3, sc_fault3, sc_scale3)
    make_p2_fig5_sensitivity(sens_pi, sens_mu, sens_anc)
    make_p2_fig6_stability_phase_200k(phase_df, phase_trajs, scale_v4, heal_df, heal_trajs, v4_env)
    write_p2_tab1_controller_ablation(bench, ksens, stress, traj, tcga3, ksens3, stress3, traj3)
    write_p2_tab2_fault_stress(stress, stress3)
    write_p2_tab3_singlecell(sc_bench, sc_fault, scale, sc_bench3, sc_fault3, sc_scale3)
    write_p2_supplementary_tables(summary, bench, stress, tcga3, stress3, sc_bench, sc_fault, sc_bench3, sc_fault3)
    make_graphical_abstract(stress, stress3, heal_df, heal_trajs, v4_env)
    print("Generated Paper 2 assets:", len(list(FIG_DIR.glob("*.pdf"))), "figs,", len(list(TAB_DIR.glob("*.tex"))), "tabs + Graphical Abstract")

if __name__ == "__main__":
    main()
