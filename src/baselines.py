"""Baselines, deep multi-omics models, partial/scalable benchmarks, and evaluation metrics."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from lifelines.statistics import multivariate_logrank_test

from cl_vssc import (
    bipartite_spectral_eigendecomposition,
    build_bipartite_affinity,
    build_knn_affinity,
    doubly_stochastic_normalize,
    fast_left_svd,
    get_device,
    preprocess_views,
    spectral_eigendecomposition,
)


def _cluster_embeddings(F_np: np.ndarray, n_clusters: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(F_np, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    F_unit = F_np / norms
    n, d = F_unit.shape
    if n <= 5000:
        labels = KMeans(n_clusters=n_clusters, n_init=30, random_state=seed).fit_predict(F_unit)
    elif n <= 60000 and d <= 64:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(F_unit)
    else:
        labels = MiniBatchKMeans(
            n_clusters=n_clusters, n_init=10, batch_size=min(4096, n), random_state=seed
        ).fit_predict(F_unit)
    return labels, F_unit


def _njw_cluster(S: torch.Tensor, n_clusters: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    F, _, _, _ = spectral_eigendecomposition(S, n_clusters)
    return _cluster_embeddings(F.detach().cpu().numpy(), n_clusters, seed=seed)


def clustering_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unsupervised clustering accuracy (ACC) via optimal Hungarian assignment."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _, t_inv = np.unique(y_true, return_inverse=True)
    _, p_inv = np.unique(y_pred, return_inverse=True)
    dim = max(int(t_inv.max()) + 1, int(p_inv.max()) + 1)
    w = np.zeros((dim, dim), dtype=np.int64)
    np.add.at(w, (p_inv, t_inv), 1)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return float(w[row_ind, col_ind].sum() / max(1, len(y_true)))


@torch.no_grad()
def run_concat_spectral(
    views: List[np.ndarray], n_clusters: int, seed: int = 42, device: Optional[str] = None
) -> Dict[str, object]:
    dev = get_device(device)
    X_list, _ = preprocess_views(views, dev)
    X_cat = torch.cat(X_list, dim=1)
    n = X_cat.shape[0]
    k_nn = max(8, min(25, n // 8))
    S = build_knn_affinity(X_cat, k=k_nn)
    labels, emb = _njw_cluster(S, n_clusters, seed=seed)
    return {"labels": labels, "S": S.detach().cpu().numpy(), "F": emb}


@torch.no_grad()
def run_snf(
    views: List[np.ndarray],
    n_clusters: int,
    k: Optional[int] = None,
    t_iter: int = 20,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Similarity Network Fusion (Wang et al., Nature Methods 2014)."""
    dev = get_device(device)
    X_list, _ = preprocess_views(views, dev)
    V = len(X_list)
    n = X_list[0].shape[0]
    k_eff = k if k is not None else max(8, min(20, n // 10))

    P_list, S_list = [], []
    for x in X_list:
        dist = torch.cdist(x, x, p=2.0)
        d_sorted, idx = torch.sort(dist, dim=1)
        eps_i = d_sorted[:, 1 : k_eff + 1].mean(dim=1).clamp_min(1e-6)
        sigma = 0.5 * (eps_i[:, None] + eps_i[None, :] + dist) / 3.0 + 1e-6
        W = torch.exp(-(dist.pow(2)) / (0.5 * sigma.pow(2) + 1e-6))
        W = 0.5 * (W + W.T)

        W_off = W.clone()
        W_off.fill_diagonal_(0.0)
        row_s = W_off.sum(dim=1, keepdim=True).clamp_min(1e-8)
        P = 0.5 * (W_off / row_s)
        P.fill_diagonal_(0.5)
        P = 0.5 * (P + P.T)
        P_list.append(P)

        mask = torch.zeros((n, n), dtype=torch.bool, device=dev)
        row_ar = torch.arange(n, device=dev)[:, None]
        mask[row_ar, idx[:, 1 : k_eff + 1]] = True
        W_loc = W_off * mask.float()
        loc_s = W_loc.sum(dim=1, keepdim=True).clamp_min(1e-8)
        S_loc = W_loc / loc_s
        S_list.append(S_loc)

    if V > 1:
        for _ in range(t_iter):
            P_next = []
            for v in range(V):
                other_mean = sum(P_list[u] for u in range(V) if u != v) / (V - 1)
                P_new = S_list[v] @ other_mean @ S_list[v].T
                P_new = 0.5 * (P_new + P_new.T)
                row_s = P_new.sum(dim=1, keepdim=True).clamp_min(1e-8)
                P_new = P_new / row_s
                P_new.fill_diagonal_(0.5)
                P_new = 0.5 * (P_new + P_new.T)
                P_next.append(P_new)
            P_list = P_next

    W_fused = sum(P_list) / V
    W_fused.fill_diagonal_(0.0)
    W_fused = doubly_stochastic_normalize(W_fused)
    labels, emb = _njw_cluster(W_fused, n_clusters, seed=seed)
    return {"labels": labels, "S": W_fused.detach().cpu().numpy(), "F": emb}


@torch.no_grad()
def run_nemo(
    views: List[np.ndarray],
    n_clusters: int,
    k: Optional[int] = None,
    masks: Optional[List[np.ndarray]] = None,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Neighborhood-based Multi-Omics clustering with exact partial-view support (Rappoport & Shamir, 2019)."""
    dev = get_device(device)
    has_missing = masks is not None
    X_list, M_list = preprocess_views(views, dev, masks=masks)
    V = len(X_list)
    n = X_list[0].shape[0]
    k_eff = k if k is not None else max(8, min(25, n // 6))

    RS_list = []
    pair_masks = []
    for v_i, x in enumerate(X_list):
        if has_missing:
            obs_idx = torch.nonzero(M_list[v_i] > 0.5, as_tuple=False).squeeze(1)
            rs_full = torch.zeros((n, n), dtype=torch.float32, device=dev)
            if len(obs_idx) >= 3:
                x_sub = x[obs_idx]
                k_sub = min(k_eff, len(obs_idx) - 1)
                dist2 = torch.cdist(x_sub, x_sub, p=2.0).pow(2)
                dist2.fill_diagonal_(float("inf"))
                knn_vals, knn_idx = torch.topk(dist2, k=k_sub, dim=1, largest=False)
                sig_i = knn_vals.mean(dim=1).clamp_min(1e-6)
                dist2.fill_diagonal_(0.0)
                denom = sig_i[:, None] + sig_i[None, :] + 1e-6
                W = torch.exp(-dist2 / denom)
                mask = torch.zeros((len(obs_idx), len(obs_idx)), dtype=torch.bool, device=dev)
                row_ar = torch.arange(len(obs_idx), device=dev)[:, None]
                mask[row_ar, knn_idx] = True
                mask = mask | mask.T
                mask.fill_diagonal_(False)
                W_knn = W * mask.float()
                rs_sub = W_knn / W_knn.sum(dim=1, keepdim=True).clamp_min(1e-8)
                rs_sub = 0.5 * (rs_sub + rs_sub.T)
                rs_full[obs_idx[:, None], obs_idx[None, :]] = rs_sub
            RS_list.append(rs_full)
            pair_masks.append(M_list[v_i][:, None] * M_list[v_i][None, :])
        else:
            dist2 = torch.cdist(x, x, p=2.0).pow(2)
            dist2.fill_diagonal_(float("inf"))
            knn_vals, knn_idx = torch.topk(dist2, k=min(k_eff, n - 1), dim=1, largest=False)
            sig_i = knn_vals.mean(dim=1).clamp_min(1e-6)
            dist2.fill_diagonal_(0.0)
            denom = sig_i[:, None] + sig_i[None, :] + 1e-6
            W = torch.exp(-dist2 / denom)

            mask = torch.zeros((n, n), dtype=torch.bool, device=dev)
            row_ar = torch.arange(n, device=dev)[:, None]
            mask[row_ar, knn_idx] = True
            mask = mask | mask.T
            mask.fill_diagonal_(False)
            W_knn = W * mask.float()
            rs = W_knn / W_knn.sum(dim=1, keepdim=True).clamp_min(1e-8)
            RS_list.append(0.5 * (rs + rs.T))

    if has_missing:
        num = sum(RS_list)
        den = sum(pair_masks)
        ARS = torch.where(den > 0.5, num / den.clamp_min(1.0), torch.zeros_like(num))
        ARS.fill_diagonal_(0.0)
        empty_rows = ARS.sum(dim=1) < 1e-8
        if empty_rows.any():
            ARS[empty_rows] = 1.0 / max(1, n - 1)
            ARS.fill_diagonal_(0.0)
    else:
        ARS = sum(RS_list) / V

    ARS = doubly_stochastic_normalize(ARS)
    labels, emb = _njw_cluster(ARS, n_clusters, seed=seed)
    return {"labels": labels, "S": ARS.detach().cpu().numpy(), "F": emb}


@torch.no_grad()
def run_openloop_mavs(
    views: List[np.ndarray],
    n_clusters: int,
    shared_dim: int = 24,
    spec_dim: int = 8,
    n_iters: int = 20,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Open-loop variable separation + static/attention fusion + post-hoc spectral clustering."""
    dev = get_device(device)
    torch.manual_seed(seed)
    X_list, _ = preprocess_views(views, dev)
    V = len(X_list)
    n = X_list[0].shape[0]
    k_nn = max(8, min(25, n // 8))

    X_cat = torch.cat(X_list, dim=1)
    r = min(shared_dim, n - 1, X_cat.shape[1])
    Z, _ = fast_left_svd(X_cat, r)

    U_list, Q_list, P_list = [], [], []
    for x in X_list:
        r_v = max(1, min(spec_dim, n - r - 1 if n - r > 2 else max(1, n // 6), x.shape[1]))
        p_v = Z.T @ x
        res = x - Z @ p_v
        res = res - Z @ (Z.T @ res)
        u_v, _ = fast_left_svd(res, r_v)
        U_list.append(u_v)
        Q_list.append(u_v.T @ res)
        P_list.append(p_v)

    for _ in range(n_iters):
        grad_Z = torch.zeros_like(Z)
        for v_i, x in enumerate(X_list):
            x_sh = x - U_list[v_i] @ Q_list[v_i]
            P_list[v_i] = Z.T @ x_sh
            grad_Z = grad_Z + x_sh @ P_list[v_i].T
        u_z, _, vt_z = torch.linalg.svd(grad_Z, full_matrices=False)
        Z = u_z @ vt_z
        for v_i, x in enumerate(X_list):
            res = x - Z @ (Z.T @ x)
            r_v = U_list[v_i].shape[1]
            u_v, _ = fast_left_svd(res, r_v)
            u_v = u_v - Z @ (Z.T @ u_v)
            u_p, _, vt_p = torch.linalg.svd(u_v, full_matrices=False)
            U_list[v_i] = u_p @ vt_p
            Q_list[v_i] = U_list[v_i].T @ res
            P_list[v_i] = Z.T @ (x - U_list[v_i] @ Q_list[v_i])

    energies = []
    A_list = []
    for v_i, x in enumerate(X_list):
        x_sh = x - U_list[v_i] @ Q_list[v_i]
        A_list.append(build_knn_affinity(x_sh, k=k_nn))
        en = float((Z @ P_list[v_i]).pow(2).sum().item()) / (float(x.pow(2).sum().item()) + 1e-8)
        energies.append(en)
    w_open = torch.softmax(torch.tensor(energies, dtype=torch.float32, device=dev) / 0.25, dim=0)

    S_fused = sum(w_open[v] * A_list[v] for v in range(V))
    S_fused = doubly_stochastic_normalize(S_fused)
    labels, emb = _njw_cluster(S_fused, n_clusters, seed=seed)
    return {
        "labels": labels,
        "S": S_fused.detach().cpu().numpy(),
        "F": emb,
        "final_weights": w_open.detach().cpu().numpy(),
    }


@torch.no_grad()
def run_mcgc(
    views: List[np.ndarray],
    n_clusters: int,
    k: Optional[int] = None,
    n_iters: int = 25,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Multiview Consensus Graph Clustering with inverse-disagreement weights and CLR rank penalty."""
    dev = get_device(device)
    torch.manual_seed(seed)
    X_list, _ = preprocess_views(views, dev)
    V = len(X_list)
    n = X_list[0].shape[0]
    k_eff = k if k is not None else max(8, min(25, n // 6))

    A_list = [build_knn_affinity(x, k=k_eff) for x in X_list]
    w = torch.full((V,), 1.0 / V, dtype=torch.float32, device=dev)
    S = sum(w[v] * A_list[v] for v in range(V))
    S = 0.5 * (S + S.T)
    mu = 0.05

    for _ in range(n_iters):
        F, _, spec_trace, _ = spectral_eigendecomposition(S, n_clusters)
        f_norms = F.norm(dim=1, keepdim=True).clamp_min(1e-8)
        H = torch.cdist(F / f_norms, F / f_norms, p=2.0).pow(2)

        diffs = torch.stack([torch.norm(S - A_v, p="fro").clamp_min(1e-6) for A_v in A_list])
        inv_d = 1.0 / diffs
        w = inv_d / inv_d.sum().clamp_min(1e-8)

        S_weighted = sum(w[v] * A_list[v] for v in range(V))
        S_target = (S_weighted - (mu / k_eff) * H).clamp_min(0.0)
        S_target.fill_diagonal_(0.0)
        row_st = S_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
        S_target = S_target / row_st
        S_target = 0.5 * (S_target + S_target.T)
        S = 0.65 * S + 0.35 * S_target
        S = 0.5 * (S + S.T)

        if spec_trace > 0.02:
            mu = min(mu * 1.08, 0.20)

    S = doubly_stochastic_normalize(S)
    labels, emb = _njw_cluster(S, n_clusters, seed=seed)
    return {
        "labels": labels,
        "S": S.detach().cpu().numpy(),
        "F": emb,
        "final_weights": w.detach().cpu().numpy(),
    }


def run_subtype_gan(
    views: List[np.ndarray],
    n_clusters: int,
    latent_dim: int = 24,
    epochs: Optional[int] = None,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Adversarial multi-omics autoencoder baseline inspired by Subtype-GAN (Yang et al., 2021)."""
    dev = get_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_list, _ = preprocess_views(views, dev)
    n = X_list[0].shape[0]
    dims = [x.shape[1] for x in X_list]
    n_epochs = epochs if epochs is not None else (100 if n <= 2000 else 50)
    batch_size = min(1024, n)

    class _SubtypeGANNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encs = nn.ModuleList(
                [nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Linear(128, 64)) for d in dims]
            )
            self.to_z = nn.Linear(64, latent_dim)
            self.decs = nn.ModuleList(
                [nn.Sequential(nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, d)) for d in dims]
            )

        def encode(self, xs: List[torch.Tensor]) -> torch.Tensor:
            hs = torch.stack([enc(x) for enc, x in zip(self.encs, xs)], dim=0).mean(dim=0)
            return self.to_z(F_nn.leaky_relu(hs, 0.2))

        def decode(self, z: torch.Tensor) -> List[torch.Tensor]:
            return [dec(z) for dec in self.decs]

    class _Disc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1)
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)

    gen = _SubtypeGANNet().to(dev)
    disc = _Disc().to(dev)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=1e-5)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    with torch.enable_grad():
        gen.train()
        disc.train()
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) < 4:
                    continue
                xb = [x[idx] for x in X_list]
                z_fake = gen.encode(xb).detach()
                z_real = torch.randn_like(z_fake)
                opt_d.zero_grad()
                loss_d = 0.5 * (
                    bce(disc(z_real), torch.ones((len(idx), 1), device=dev))
                    + bce(disc(z_fake), torch.zeros((len(idx), 1), device=dev))
                )
                loss_d.backward()
                opt_d.step()

                opt_g.zero_grad()
                z = gen.encode(xb)
                recons = gen.decode(z)
                rec_loss = sum(F_nn.mse_loss(rh, x_v) for rh, x_v in zip(recons, xb))
                adv_loss = bce(disc(z), torch.ones((len(idx), 1), device=dev))
                loss_g = rec_loss + 0.05 * adv_loss
                loss_g.backward()
                opt_g.step()

    with torch.no_grad():
        gen.eval()
        zs = []
        for start in range(0, n, 2048):
            zs.append(gen.encode([x[start : start + 2048] for x in X_list]).cpu())
        Z_np = torch.cat(zs, dim=0).numpy()

    if n <= 5000:
        try:
            labels = GaussianMixture(n_components=n_clusters, covariance_type="diag", n_init=5, random_state=seed).fit_predict(Z_np)
        except Exception:
            labels, _ = _cluster_embeddings(Z_np, n_clusters, seed=seed)
        norms = np.linalg.norm(Z_np, axis=1, keepdims=True)
        F_unit = Z_np / np.where(norms < 1e-12, 1.0, norms)
    else:
        labels, F_unit = _cluster_embeddings(Z_np, n_clusters, seed=seed)
    return {"labels": labels, "F": F_unit}


def run_attention_moae(
    views: List[np.ndarray],
    n_clusters: int,
    latent_dim: int = 24,
    epochs: Optional[int] = None,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Attention-gated multi-omics autoencoder baseline inspired by MOMA (Moon & Lee, 2022)."""
    dev = get_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_list, _ = preprocess_views(views, dev)
    n = X_list[0].shape[0]
    dims = [x.shape[1] for x in X_list]
    n_epochs = epochs if epochs is not None else (100 if n <= 2000 else 50)
    batch_size = min(1024, n)

    class _AttMOAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encs = nn.ModuleList(
                [nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Linear(128, latent_dim)) for d in dims]
            )
            self.att_w = nn.Linear(latent_dim, 32)
            self.att_v = nn.Linear(32, 1, bias=False)
            self.decs = nn.ModuleList(
                [nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, d)) for d in dims]
            )

        def forward(self, xs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
            hs = torch.stack([enc(x) for enc, x in zip(self.encs, xs)], dim=1)  # (b, V, d)
            scores = self.att_v(torch.tanh(self.att_w(hs))).squeeze(-1)  # (b, V)
            alpha = torch.softmax(scores, dim=1)
            z = torch.sum(hs * alpha.unsqueeze(-1), dim=1)
            recons = [dec(z) for dec in self.decs]
            return z, recons, alpha

    model = _AttMOAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    with torch.enable_grad():
        model.train()
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) < 4:
                    continue
                xb = [x[idx] for x in X_list]
                opt.zero_grad()
                _, recons, _ = model(xb)
                loss = sum(F_nn.mse_loss(rh, x_v) for rh, x_v in zip(recons, xb))
                loss.backward()
                opt.step()

    with torch.no_grad():
        model.eval()
        zs, alphas = [], []
        for start in range(0, n, 2048):
            z_b, _, al_b = model([x[start : start + 2048] for x in X_list])
            zs.append(z_b.cpu())
            alphas.append(al_b.cpu())
        Z_np = torch.cat(zs, dim=0).numpy()
        mean_w = torch.cat(alphas, dim=0).mean(dim=0).numpy()

    labels, F_unit = _cluster_embeddings(Z_np, n_clusters, seed=seed)
    return {"labels": labels, "F": F_unit, "final_weights": mean_w}


@torch.no_grad()
def run_wnn(
    views: List[np.ndarray],
    n_clusters: int,
    k: Optional[int] = None,
    n_anchors: int = 384,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Weighted Nearest Neighbors multimodal clustering (Hao et al., Cell 2021)."""
    dev = get_device(device)
    X_list, _ = preprocess_views(views, dev)
    V = len(X_list)
    n = X_list[0].shape[0]

    if n > 6000:
        # Scalable anchor-WNN via cross-view local landmark predictability ratios
        X_cat = torch.cat(X_list, dim=1)
        Z0, _ = fast_left_svd(X_cat, min(24, n - 1, X_cat.shape[1]))
        m = min(n_anchors, n // 4)
        km = MiniBatchKMeans(n_clusters=m, n_init=3, batch_size=min(2048, n), max_iter=25, random_state=seed)
        km.fit(Z0.cpu().numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev)
        anc_idx = torch.argmin(torch.cdist(centers, Z0, p=2.0), dim=1)
        k_a = min(15, m)

        B_list = []
        ratio_scores = []
        for v_i, x in enumerate(X_list):
            B_v, _ = build_bipartite_affinity(x, x[anc_idx], k_anchor=k_a)
            B_list.append(B_v)
            pred_within = B_v @ x[anc_idx]
            err_within = (x - pred_within).pow(2).sum(dim=1).clamp_min(1e-6)
            cross_errs = []
            for u_i in range(V):
                if u_i == v_i:
                    continue
                pred_cross = B_list[u_i] @ x[anc_idx] if u_i < v_i else build_bipartite_affinity(X_list[u_i], X_list[u_i][anc_idx], k_anchor=k_a)[0] @ x[anc_idx]
                cross_errs.append((x - pred_cross).pow(2).sum(dim=1).clamp_min(1e-6))
            mean_cross = sum(cross_errs) / max(1, len(cross_errs))
            ratio_scores.append(torch.log(mean_cross / err_within).clamp(-5.0, 5.0))
        theta = torch.softmax(torch.stack(ratio_scores, dim=1), dim=1)  # (n, V)
        B_wnn = sum(theta[:, v : v + 1] * B_list[v] for v in range(V))
        F, _, _, _, _ = bipartite_spectral_eigendecomposition(B_wnn, n_clusters)
        labels, emb = _cluster_embeddings(F.detach().cpu().numpy(), n_clusters, seed=seed)
        return {"labels": labels, "F": emb, "final_weights": theta.mean(dim=0).detach().cpu().numpy()}

    k_eff = k if k is not None else max(10, min(25, n // 8))
    K_list = [build_knn_affinity(x, k=k_eff) for x in X_list]
    aff_scores = []
    for v_i, x in enumerate(X_list):
        pred_within = K_list[v_i] @ x
        err_within = (x - pred_within).pow(2).sum(dim=1).clamp_min(1e-6)
        cross_errs = []
        for u_i in range(V):
            if u_i != v_i:
                cross_errs.append((x - K_list[u_i] @ x).pow(2).sum(dim=1).clamp_min(1e-6))
        mean_cross = sum(cross_errs) / max(1, len(cross_errs))
        aff_scores.append(torch.log(mean_cross / err_within).clamp(-5.0, 5.0))
    theta = torch.softmax(torch.stack(aff_scores, dim=1), dim=1)  # (n, V)
    W_wnn = sum(theta[:, v : v + 1] * K_list[v] for v in range(V))
    W_wnn.fill_diagonal_(0.0)
    W_wnn = doubly_stochastic_normalize(0.5 * (W_wnn + W_wnn.T))
    labels, emb = _njw_cluster(W_wnn, n_clusters, seed=seed)
    return {
        "labels": labels,
        "S": W_wnn.detach().cpu().numpy(),
        "F": emb,
        "final_weights": theta.mean(dim=0).detach().cpu().numpy(),
    }


@torch.no_grad()
def run_lmvsc(
    views: List[np.ndarray],
    n_clusters: int,
    n_anchors: int = 384,
    anchor_k: int = 15,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Large-scale Multi-View Subspace Clustering in linear time (Kang et al., AAAI 2020)."""
    dev = get_device(device)
    torch.manual_seed(seed)
    X_list, _ = preprocess_views(views, dev)
    V = len(X_list)
    n = X_list[0].shape[0]
    m = min(n_anchors, max(n_clusters + 4, n // 4))
    k_a = min(anchor_k, m)

    Bt_views = []
    for v_i, x in enumerate(X_list):
        # View-specific or shared landmark selection via fast KMeans on reduced projection
        r_tmp = min(20, x.shape[1], n - 1)
        U_tmp, _ = fast_left_svd(x, r_tmp)
        km = MiniBatchKMeans(n_clusters=m, n_init=3, batch_size=min(2048, n), max_iter=25, random_state=seed + v_i)
        km.fit(U_tmp.cpu().numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev)
        anc_idx = torch.argmin(torch.cdist(centers, U_tmp, p=2.0), dim=1)
        _, Bt_v = build_bipartite_affinity(x, x[anc_idx], k_anchor=k_a)
        Bt_views.append(Bt_v)

    H_cat = torch.cat(Bt_views, dim=1) / np.sqrt(float(V))
    F, _ = fast_left_svd(H_cat, n_clusters)
    labels, emb = _cluster_embeddings(F.detach().cpu().numpy(), n_clusters, seed=seed)
    return {"labels": labels, "F": emb}


def corrupt_view(
    views: List[np.ndarray],
    view_idx: int,
    noise_std: float = 0.0,
    dropout_rate: float = 0.0,
    seed: int = 0,
) -> List[np.ndarray]:
    """Inject scale-invariant Gaussian noise and feature dropout into one modality."""
    rng = np.random.default_rng(seed)
    out = [np.array(v, copy=True, dtype=np.float32) for v in views]
    target = out[view_idx]
    feat_std = target.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    if noise_std > 0.0:
        target = target + rng.normal(0.0, noise_std, size=target.shape).astype(np.float32) * feat_std
    if dropout_rate > 0.0:
        mask = rng.random(size=target.shape) >= dropout_rate
        target = target * mask.astype(np.float32)
    out[view_idx] = target
    return out


def _logrank_stat_batch(
    obs_minus_exp_vec: np.ndarray,
    diag_w: np.ndarray,
    R_scaled: np.ndarray,
    G_batch: np.ndarray,
) -> np.ndarray:
    """Compute exact multivariate log-rank chi-square statistics for a batch of one-hot label matrices."""
    B, n, K = G_batch.shape
    Z = np.einsum("n,bnk->bk", obs_minus_exp_vec, G_batch)
    diag_val = np.einsum("n,bnk->bk", diag_w, G_batch)
    U = np.einsum("mn,bnk->bmk", R_scaled, G_batch)
    off = np.einsum("bmk,bml->bkl", U, U)
    V = -off
    idx = np.arange(K)
    V[:, idx, idx] += diag_val

    Z_red = Z[:, :-1]
    V_red = V[:, :-1, :-1]
    stats = np.zeros(B, dtype=np.float64)
    try:
        sol = np.linalg.solve(V_red, Z_red[..., None])[..., 0]
        stats = np.sum(Z_red * sol, axis=1)
    except np.linalg.LinAlgError:
        for b in range(B):
            try:
                stats[b] = float(Z_red[b] @ np.linalg.pinv(V_red[b]) @ Z_red[b])
            except Exception:
                stats[b] = 0.0
    return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def permutation_logrank_p(
    times: np.ndarray,
    events: np.ndarray,
    labels: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Empirical exact permutation log-rank test (Rappoport & Shamir, NAR 2018)."""
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    labels = np.asarray(labels)
    valid = np.isfinite(times) & np.isfinite(events) & (times > 0)
    times, events, labels = times[valid], events[valid], labels[valid]
    _, inv = np.unique(labels, return_inverse=True)
    K = int(inv.max()) + 1 if len(inv) > 0 else 0
    t_ev = np.unique(times[events > 0.5])
    if K < 2 or len(t_ev) == 0:
        return 0.0, 1.0

    R = (times[None, :] >= t_ev[:, None]).astype(np.float64)
    N_m = np.maximum(R.sum(axis=1), 1e-12)
    O_m = np.array([events[times == t].sum() for t in t_ev], dtype=np.float64)
    factor_m = np.where(N_m > 1.0, O_m * (N_m - O_m) / np.maximum(N_m - 1.0, 1.0), 0.0)

    obs_minus_exp_vec = events - ((O_m / N_m) @ R)
    diag_w = (factor_m / N_m) @ R
    R_scaled = (np.sqrt(np.maximum(factor_m, 0.0)) / N_m)[:, None] * R

    n = len(times)
    G_obs = np.zeros((1, n, K), dtype=np.float64)
    G_obs[0, np.arange(n), inv] = 1.0
    T_obs = float(_logrank_stat_batch(obs_minus_exp_vec, diag_w, R_scaled, G_obs)[0])

    rng = np.random.default_rng(seed)
    G_perm_base = G_obs[0]
    ge_count = 0
    batch_size = 500
    done = 0
    while done < n_perm:
        bs = min(batch_size, n_perm - done)
        perms = np.argsort(rng.random((bs, n)), axis=1)
        G_batch = G_perm_base[perms]
        T_perm = _logrank_stat_batch(obs_minus_exp_vec, diag_w, R_scaled, G_batch)
        ge_count += int(np.sum(T_perm >= T_obs - 1e-12))
        done += bs

    perm_p = float((1 + ge_count) / (1 + n_perm))
    return T_obs, perm_p


def evaluate_clustering(
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    survival_df: Optional[pd.DataFrame] = None,
    embedding: Optional[np.ndarray] = None,
    compute_perm: bool = False,
    n_perm: int = 5000,
    perm_seed: int = 42,
) -> Dict[str, float]:
    """Compute survival log-rank, permutation p-value, silhouette, ARI/NMI/ACC, and cluster balance."""
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    res: Dict[str, float] = {
        "n_clusters_found": float(len(counts)),
        "min_cluster_size": float(counts.min()),
        "max_cluster_size": float(counts.max()),
    }
    if embedding is not None and len(counts) > 1 and len(labels) > len(counts):
        s_size = min(2500, len(labels)) if len(labels) > 3000 else None
        res["silhouette"] = float(silhouette_score(embedding, labels, sample_size=s_size, random_state=42))
    if true_labels is not None:
        res["ari"] = float(adjusted_rand_score(true_labels, labels))
        res["nmi"] = float(normalized_mutual_info_score(true_labels, labels))
        res["acc"] = clustering_acc(true_labels, labels)
    if survival_df is not None and len(counts) > 1:
        df = survival_df.copy()
        df["cluster"] = labels
        df = df.dropna(subset=["time", "event"])
        df = df[df["time"] > 0]
        if df["cluster"].nunique() > 1:
            lr = multivariate_logrank_test(df["time"], df["cluster"], df["event"])
            p = max(float(lr.p_value), 1e-300)
            res["logrank_p"] = float(lr.p_value)
            res["neg_log10_p"] = float(-np.log10(p))
            if compute_perm:
                _, perm_p = permutation_logrank_p(
                    df["time"].to_numpy(),
                    df["event"].to_numpy(),
                    df["cluster"].to_numpy(),
                    n_perm=n_perm,
                    seed=perm_seed,
                )
                res["perm_p"] = perm_p
                res["neg_log10_perm_p"] = float(-np.log10(max(perm_p, 1e-12)))
        else:
            res["logrank_p"] = 1.0
            res["neg_log10_p"] = 0.0
            if compute_perm:
                res["perm_p"] = 1.0
                res["neg_log10_perm_p"] = 0.0
    elif survival_df is not None:
        res["logrank_p"] = 1.0
        res["neg_log10_p"] = 0.0
        if compute_perm:
            res["perm_p"] = 1.0
            res["neg_log10_perm_p"] = 0.0
    return res
