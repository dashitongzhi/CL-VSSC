"""Closed-Loop Feedback-Regulated Variable Separation and Controllable Spectral Clustering (CL-VSSC v2)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import time
import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_views(
    views: List[np.ndarray],
    device: torch.device,
    masks: Optional[List[np.ndarray]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Column-standardize each view (using observed rows if masks supplied) and normalize by sqrt(d_v)."""
    processed: List[torch.Tensor] = []
    mask_tensors: List[torch.Tensor] = []
    for v_idx, v in enumerate(views):
        arr = np.asarray(v, dtype=np.float32)
        n = arr.shape[0]
        if masks is not None:
            m = np.asarray(masks[v_idx], dtype=bool)
            obs = arr[m] if m.any() else arr
            mean = np.nanmean(obs, axis=0, keepdims=True)
            std = np.nanstd(obs, axis=0, keepdims=True)
            mean = np.nan_to_num(mean, nan=0.0)
            std = np.where((~np.isfinite(std)) | (std < 1e-6), 1.0, std)
            arr_clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            normed = (arr_clean - mean) / std
            normed[~m] = 0.0
            scale = np.sqrt(max(1, normed.shape[1])) * np.sqrt(max(1, int(m.sum())) / max(1, n))
            t = torch.tensor(normed / scale, dtype=torch.float32, device=device)
            mt = torch.tensor(m.astype(np.float32), dtype=torch.float32, device=device)
        else:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            normed = (arr - mean) / std
            scale = np.sqrt(max(1, normed.shape[1]))
            t = torch.tensor(normed / scale, dtype=torch.float32, device=device)
            mt = torch.ones((n,), dtype=torch.float32, device=device)
        processed.append(t)
        mask_tensors.append(mt)
    return processed, mask_tensors


def build_knn_affinity(X: torch.Tensor, k: int, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Self-tuning k-NN Gaussian affinity with symmetric row normalization (supports partial observation mask)."""
    n = X.shape[0]
    if obs_mask is not None and float(obs_mask.min().item()) < 0.5:
        obs_idx = torch.nonzero(obs_mask > 0.5, as_tuple=False).squeeze(1)
        A_full = torch.zeros((n, n), dtype=X.dtype, device=X.device)
        if len(obs_idx) < 3:
            return A_full
        A_sub = build_knn_affinity(X[obs_idx], k=min(k, len(obs_idx) - 1), obs_mask=None)
        A_full[obs_idx[:, None], obs_idx[None, :]] = A_sub
        return A_full

    k_eff = max(2, min(k, n - 1))
    dist2 = torch.cdist(X, X, p=2.0).pow(2)
    dist2.fill_diagonal_(float("inf"))
    knn_vals, knn_idx = torch.topk(dist2, k=k_eff, dim=1, largest=False)
    sig2 = knn_vals.mean(dim=1).clamp_min(1e-6)

    dist2.fill_diagonal_(0.0)
    denom = sig2[:, None] + sig2[None, :] + 1e-6
    kernel = torch.exp(-dist2 / denom)

    mask = torch.zeros((n, n), dtype=torch.bool, device=X.device)
    row_idx = torch.arange(n, device=X.device)[:, None]
    mask[row_idx, knn_idx] = True
    mask = mask | mask.T
    mask.fill_diagonal_(False)

    A = kernel * mask.float()
    row_sum = A.sum(dim=1, keepdim=True).clamp_min(1e-8)
    P = A / row_sum
    return 0.5 * (P + P.T)


def normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
    """Compute symmetric normalized graph Laplacian L(A) = I - D^{-1/2} A D^{-1/2}."""
    n = A.shape[0]
    deg = A.sum(dim=1).clamp_min(1e-8)
    d_inv_sqrt = deg.rsqrt()
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    L = torch.eye(n, dtype=A.dtype, device=A.device) - 0.5 * (A_norm + A_norm.T)
    return L


def doubly_stochastic_normalize(A: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Symmetric Sinkhorn-like scaling onto balanced affinity scale."""
    M = A.clamp_min(0.0)
    M = 0.5 * (M + M.T)
    eps = 1e-8
    for _ in range(n_iters):
        row_sum = M.sum(dim=1, keepdim=True).clamp_min(eps)
        M = M / row_sum
        M = 0.5 * (M + M.T)
    row_sum = M.sum(dim=1, keepdim=True).clamp_min(eps)
    M = M / row_sum
    return 0.5 * (M + M.T)


def polar_orthogonalize(M: torch.Tensor) -> torch.Tensor:
    """Nearest column-orthonormal matrix on the Stiefel manifold via thin SVD."""
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt


def fast_left_svd(M: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute top-rank left singular vectors and singular values efficiently."""
    min_dim = min(M.shape)
    q = min(rank, min_dim)
    if min_dim <= 640 or q >= min_dim // 2:
        U, S_vals, _ = torch.linalg.svd(M, full_matrices=False)
        return U[:, :q].clone(), S_vals[:q].clone()
    over = min(min_dim, q + 8)
    U, S_vals, _ = torch.svd_lowrank(M, q=over, niter=3)
    return U[:, :q].clone(), S_vals[:q].clone()


def spectral_eigendecomposition(
    S: torch.Tensor, n_clusters: int
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Compute bottom-(c+1) normalized Laplacian eigenvalues and bottom-c eigenvectors."""
    deg = S.sum(dim=1).clamp_min(1e-8)
    d_inv_sqrt = deg.rsqrt()
    S_norm = d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :]
    S_norm = 0.5 * (S_norm + S_norm.T)
    evals_s, evecs_s = torch.linalg.eigh(S_norm)
    lap_evals = (1.0 - evals_s.flip(0)).clamp_min(0.0)
    F = evecs_s[:, -n_clusters:].flip(1)
    spec_trace = float(lap_evals[:n_clusters].sum().item())
    gap = float((lap_evals[n_clusters] - lap_evals[n_clusters - 1]).item()) if len(lap_evals) > n_clusters else 0.0
    return F, d_inv_sqrt, spec_trace, gap


def _shuffle_observed_columns(
    X: torch.Tensor, obs_mask: Optional[torch.Tensor], seed: int
) -> torch.Tensor:
    """Independently permute observed entries within each feature column to form a structureless null baseline."""
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed)
    X_null = X.clone()
    if obs_mask is not None and float(obs_mask.min().item()) < 0.5:
        obs_idx = torch.nonzero(obs_mask > 0.5, as_tuple=False).squeeze(1)
        if len(obs_idx) < 4:
            return X_null
        x_obs = X[obs_idx]
        rand_keys = torch.rand((len(obs_idx), X.shape[1]), generator=gen, device=X.device)
        perm = torch.argsort(rand_keys, dim=0)
        X_null[obs_idx] = torch.gather(x_obs, 0, perm)
        return X_null
    rand_keys = torch.rand(X.shape, generator=gen, device=X.device)
    perm = torch.argsort(rand_keys, dim=0)
    return torch.gather(X, 0, perm)


def _compute_dense_null_cut(
    X: torch.Tensor, obs_mask: Optional[torch.Tensor], k_nn: int, n_clusters: int, seed: int
) -> float:
    """Null-hypothesis intrinsic normalized cut after independent column permutation."""
    x_null = _shuffle_observed_columns(X, obs_mask, seed=seed)
    if obs_mask is not None and float(obs_mask.min().item()) < 0.5:
        obs_idx = torch.nonzero(obs_mask > 0.5, as_tuple=False).squeeze(1)
        if len(obs_idx) < 4:
            return 1.0
        x_sub = x_null[obs_idx]
        A_null = build_knn_affinity(x_sub, k=min(k_nn, len(obs_idx) - 1), obs_mask=None)
        c_eff = min(n_clusters, max(2, len(obs_idx) - 1))
    else:
        A_null = build_knn_affinity(x_null, k=k_nn, obs_mask=None)
        c_eff = n_clusters
    _, _, null_tr, _ = spectral_eigendecomposition(A_null, c_eff)
    return max(0.15, float(null_tr) / max(1, n_clusters))


def _compute_anchor_null_cut(
    X: torch.Tensor, k_anchor: int, n_anchors: int, n_clusters: int, seed: int
) -> float:
    """Fast subsampled null bipartite normalized cut after independent column permutation."""
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed + 500)
    sub_n = min(X.shape[0], 4096)
    if sub_n < X.shape[0]:
        sub_idx = torch.randperm(X.shape[0], generator=gen, device=X.device)[:sub_n]
        x_sub = X[sub_idx]
    else:
        x_sub = X
    x_null = _shuffle_observed_columns(x_sub, None, seed=seed)
    m_sub = min(n_anchors, max(n_clusters + 4, sub_n // 4))
    anc_idx = torch.randperm(sub_n, generator=gen, device=X.device)[:m_sub]
    B_null, _ = build_bipartite_affinity(x_null, x_null[anc_idx], k_anchor=min(k_anchor, m_sub))
    _, _, _, null_tr, _ = bipartite_spectral_eigendecomposition(B_null, n_clusters)
    return max(0.15, float(null_tr) / max(1, n_clusters))


def _apply_deadzone_pi(
    e_vec: torch.Tensor,
    I_err: torch.Tensor,
    w: torch.Tensor,
    dead_zone: float,
    beta_i: float,
    kappa_i: float,
    eta_w: float,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dead-Zone PI mirror-descent simplex update supporting 1D (V,) and 2D (m, V) weight matrices."""
    if dead_zone > 0.0:
        d_vec = torch.clamp(e_vec - e_vec.min(dim=-1, keepdim=True).values - dead_zone, min=0.0)
    else:
        d_vec = e_vec
    I_next = (1.0 - beta_i) * I_err + beta_i * d_vec
    signal = d_vec + kappa_i * I_next
    signal_centered = signal - signal.mean(dim=-1, keepdim=True)
    log_w = (1.0 - eta_w) * torch.log(w.clamp_min(1e-12)) - (eta_w / tau) * signal_centered
    w_next = torch.softmax(log_w, dim=-1)
    return w_next, I_next


class CLVSSC:
    """Closed-Loop PI Feedback Variable Separation & Controllable Spectral Clustering."""

    def __init__(
        self,
        n_clusters: int,
        shared_dim: int = 24,
        spec_dim: int = 6,
        knn_k: Optional[int] = None,
        max_iter: int = 45,
        warmup_iters: int = 15,
        alpha_graph: float = 1.0,
        gamma_rec: float = 0.30,
        gamma_cons: float = 0.08,
        lambda_z: float = 2.0,
        beta_u: float = 0.35,
        eta_z: float = 0.35,
        eta_s: float = 0.35,
        rho_a: float = 0.25,
        eta_w: float = 0.30,
        beta_i: float = 0.4,
        kappa_i: float = 0.35,
        tau: float = 0.20,
        dead_zone: float = 0.0,
        mu_init: float = 0.0,
        eta_mu: float = 0.05,
        mu_max: float = 0.12,
        eps_spec: float = 0.02,
        use_feedback: bool = True,
        use_spectral_control: bool = True,
        use_variable_separation: bool = True,
        use_null_snr: bool = False,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.n_clusters = n_clusters
        self.shared_dim = shared_dim
        self.spec_dim = spec_dim
        self.knn_k = knn_k
        self.max_iter = max_iter
        self.warmup_iters = warmup_iters
        self.alpha_graph = alpha_graph
        self.gamma_rec = gamma_rec
        self.gamma_cons = gamma_cons
        self.lambda_z = lambda_z
        self.beta_u = beta_u
        self.eta_z = eta_z
        self.eta_s = eta_s
        self.rho_a = rho_a
        self.eta_w = eta_w
        self.beta_i = beta_i
        self.kappa_i = kappa_i
        self.tau = tau
        self.dead_zone = dead_zone
        self.mu_init = mu_init
        self.eta_mu = eta_mu
        self.mu_max = mu_max
        self.eps_spec = eps_spec
        self.use_feedback = use_feedback
        self.use_spectral_control = use_spectral_control
        self.use_variable_separation = use_variable_separation
        self.use_null_snr = use_null_snr
        self.seed = seed
        self.device = get_device(device)

    @torch.no_grad()
    def fit(self, views: List[np.ndarray], masks: Optional[List[np.ndarray]] = None) -> Dict[str, object]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        has_missing = masks is not None
        X_list, M_list = preprocess_views(views, self.device, masks=masks)
        pair_masks = [m[:, None] * m[None, :] for m in M_list]
        obs_counts = [int((m > 0.5).sum().item()) for m in M_list]

        V = len(X_list)
        n = X_list[0].shape[0]
        k_nn = self.knn_k if self.knn_k is not None else max(8, min(25, n // 6))
        x_norms = [float((x * m[:, None]).pow(2).sum().item()) + 1e-8 for x, m in zip(X_list, M_list)]
        op_norm2 = [float(fast_left_svd(x, 2)[1][0].pow(2).item()) + 1e-8 for x in X_list]

        # Initialize shared state Z via SVD of concatenated views
        X_cat = torch.cat(X_list, dim=1)
        r = min(self.shared_dim, n - 1, X_cat.shape[1])
        Z, _ = fast_left_svd(X_cat, r)

        # Iterative subspace imputation initialization on unobserved rows
        X_cur: List[torch.Tensor] = []
        P_list: List[torch.Tensor] = []
        for v_idx, x in enumerate(X_list):
            p_v = Z.T @ x
            if has_missing:
                m_col = M_list[v_idx][:, None]
                x_filled = m_col * x + (1.0 - m_col) * (Z @ p_v)
                p_v = Z.T @ x_filled
                X_cur.append(x_filled)
            else:
                X_cur.append(x)
            P_list.append(p_v)

        # Initialize orthogonal modality-specific factors U^(v), Q^(v)
        U_list: List[Optional[torch.Tensor]] = []
        Q_list: List[Optional[torch.Tensor]] = []
        for v_idx, x_f in enumerate(X_cur):
            if self.use_variable_separation:
                r_v = min(self.spec_dim, n - r - 1 if n - r > 2 else max(1, n // 6), x_f.shape[1])
                r_v = max(1, min(r_v, x_f.shape[1]))
                res = x_f - Z @ P_list[v_idx]
                res = res - Z @ (Z.T @ res)
                u_v, _ = fast_left_svd(res, r_v)
                q_v = u_v.T @ res
                U_list.append(u_v)
                Q_list.append(q_v)
            else:
                U_list.append(None)
                Q_list.append(None)

        dof_scales = [
            max(
                0.25,
                1.0
                - (r + (U_list[v_i].shape[1] if U_list[v_i] is not None else 0))
                / (min(obs_counts[v_i], X_list[v_i].shape[1]) + r + (U_list[v_i].shape[1] if U_list[v_i] is not None else 0)),
            )
            for v_i in range(V)
        ]

        # Initialize per-view k-NN affinities on softly denoised projections
        A_list: List[torch.Tensor] = []
        L_list: List[torch.Tensor] = []
        own_cuts: List[float] = []
        null_cuts: List[float] = []
        for v_idx, x_f in enumerate(X_cur):
            x_sh = x_f - self.beta_u * (U_list[v_idx] @ Q_list[v_idx]) if self.use_variable_separation else x_f
            m_v = M_list[v_idx] if has_missing else None
            A_v = build_knn_affinity(x_sh, k=k_nn, obs_mask=m_v)
            A_list.append(A_v)
            if has_missing:
                obs_idx = torch.nonzero(M_list[v_idx] > 0.5, as_tuple=False).squeeze(1)
                A_sub = A_v[obs_idx[:, None], obs_idx[None, :]]
                L_sub = normalized_laplacian(A_sub)
                L_full = torch.zeros_like(A_v)
                L_full[obs_idx[:, None], obs_idx[None, :]] = L_sub
                L_list.append(L_full)
                _, _, own_tr, _ = spectral_eigendecomposition(A_sub, min(self.n_clusters, max(2, len(obs_idx) - 1)))
                raw_cut = own_tr / max(1, self.n_clusters)
            else:
                L_list.append(normalized_laplacian(A_v))
                _, _, own_tr, _ = spectral_eigendecomposition(A_v, self.n_clusters)
                raw_cut = own_tr / max(1, self.n_clusters)
            if self.use_null_snr:
                c_n = _compute_dense_null_cut(x_sh, m_v, k_nn=k_nn, n_clusters=self.n_clusters, seed=self.seed + 1000 + v_idx)
                null_cuts.append(c_n)
                own_cuts.append(raw_cut / c_n)
            else:
                null_cuts.append(1.0)
                own_cuts.append(raw_cut)

        w = torch.full((V,), 1.0 / V, dtype=torch.float32, device=self.device)
        I_err = torch.zeros((V,), dtype=torch.float32, device=self.device)

        def fuse_weighted_affinities(cur_w: torch.Tensor) -> torch.Tensor:
            if not has_missing:
                S_w = sum(cur_w[v] * A_list[v] for v in range(V))
                return S_w
            num = sum(cur_w[v] * pair_masks[v] * A_list[v] for v in range(V))
            den = sum(cur_w[v] * pair_masks[v] for v in range(V))
            S_w = torch.where(den > 1e-8, num / den.clamp_min(1e-8), torch.zeros_like(num))
            S_w.fill_diagonal_(0.0)
            row_s = S_w.sum(dim=1, keepdim=True)
            if float(row_s.min().item()) < 1e-6:
                Z_aff = build_knn_affinity(Z, k=min(8, n - 1))
                S_w = torch.where(row_s > 1e-6, S_w, Z_aff)
            return 0.5 * (S_w + S_w.T)

        S = fuse_weighted_affinities(w)
        S = 0.5 * (S + S.T)

        mu = self.mu_init if self.use_spectral_control else 0.0
        F, d_inv_sqrt, spec_trace, gap = spectral_eigendecomposition(S, self.n_clusters)

        weights_history: List[List[float]] = []
        lyapunov_history: List[float] = []
        spectral_gap_history: List[float] = []
        spectral_trace_history: List[float] = []
        mu_history: List[float] = []

        def compute_errors_and_lyapunov(
            cur_Z: torch.Tensor,
            cur_P: List[torch.Tensor],
            cur_U: List[Optional[torch.Tensor]],
            cur_Q: List[Optional[torch.Tensor]],
            cur_L: List[torch.Tensor],
            cur_own_cuts: List[float],
            cur_F: torch.Tensor,
            cur_w: torch.Tensor,
            cur_trace: float,
        ) -> Tuple[torch.Tensor, float]:
            errs = []
            lyap = 0.0
            for v_i, x in enumerate(X_list):
                recon = cur_Z @ cur_P[v_i]
                if cur_U[v_i] is not None and cur_Q[v_i] is not None:
                    recon = recon + cur_U[v_i] @ cur_Q[v_i]
                diff = (x - recon) * M_list[v_i][:, None]
                rec_rel = (float(diff.pow(2).sum().item()) / x_norms[v_i]) / dof_scales[v_i]
                cons_raw = float(torch.trace(cur_F.T @ cur_L[v_i] @ cur_F).item())
                if has_missing:
                    obs_frac = max(0.1, obs_counts[v_i] / max(1, n))
                    cons_cut = (cons_raw / obs_frac) / max(1, self.n_clusters)
                else:
                    cons_cut = cons_raw / max(1, self.n_clusters)
                own_cut = cur_own_cuts[v_i]
                e_v = own_cut + self.gamma_cons * cons_cut + self.gamma_rec * rec_rel
                errs.append(e_v)
                lyap += float(cur_w[v_i].item()) * (rec_rel + self.alpha_graph * (own_cut + self.gamma_cons * cons_cut))
            entropy_term = float((cur_w * torch.log(cur_w.clamp_min(1e-12))).sum().item())
            spec_weight = self.mu_max if self.use_spectral_control else 0.0
            lyap += spec_weight * cur_trace + self.tau * entropy_term
            return torch.tensor(errs, dtype=torch.float32, device=self.device), lyap

        _, V_t = compute_errors_and_lyapunov(Z, P_list, U_list, Q_list, L_list, own_cuts, F, w, spec_trace)
        weights_history.append(w.detach().cpu().tolist())
        lyapunov_history.append(V_t)
        spectral_gap_history.append(gap)
        spectral_trace_history.append(spec_trace)
        mu_history.append(mu)

        for t in range(self.max_iter):
            # Step A: Orthogonal Variable Separation update (Z, P, U, Q)
            deg_s = S.sum(dim=1).clamp_min(1e-8)
            d_inv = deg_s.rsqrt()
            S_norm = d_inv[:, None] * S * d_inv[None, :]
            grad_Z = self.lambda_z * (S_norm @ Z)
            for v_i, x in enumerate(X_list):
                if has_missing:
                    m_col = M_list[v_i][:, None]
                    X_cur[v_i] = m_col * x + (1.0 - m_col) * (Z @ P_list[v_i])
                x_sh = X_cur[v_i] - (U_list[v_i] @ Q_list[v_i]) if self.use_variable_separation else X_cur[v_i]
                P_list[v_i] = Z.T @ x_sh
                grad_Z = grad_Z + (float(w[v_i].item()) / op_norm2[v_i]) * (x_sh @ P_list[v_i].T)

            Z_target = polar_orthogonalize(grad_Z)
            Z = polar_orthogonalize((1.0 - self.eta_z) * Z + self.eta_z * Z_target)

            for v_i in range(V):
                x_f = X_cur[v_i]
                if self.use_variable_separation:
                    res = x_f - Z @ (Z.T @ x_f)
                    r_v = U_list[v_i].shape[1]
                    U_cand, _ = fast_left_svd(res, r_v)
                    U_cand = U_cand - Z @ (Z.T @ U_cand)
                    U_cand = polar_orthogonalize(U_cand)
                    U_list[v_i] = U_cand
                    Q_list[v_i] = U_cand.T @ res
                    P_list[v_i] = Z.T @ (x_f - U_list[v_i] @ Q_list[v_i])
                else:
                    P_list[v_i] = Z.T @ x_f

            # Smoothly refresh per-view affinity graphs from softly denoised projection every 5 steps
            if self.use_variable_separation and (t + 1) % 5 == 0 and t < self.max_iter - 5:
                for v_i in range(V):
                    x_sh = X_cur[v_i] - self.beta_u * (U_list[v_i] @ Q_list[v_i])
                    m_v = M_list[v_i] if has_missing else None
                    A_new = build_knn_affinity(x_sh, k=k_nn, obs_mask=m_v)
                    A_v = (1.0 - self.rho_a) * A_list[v_i] + self.rho_a * A_new
                    A_list[v_i] = 0.5 * (A_v + A_v.T)
                    if has_missing:
                        obs_idx = torch.nonzero(M_list[v_i] > 0.5, as_tuple=False).squeeze(1)
                        A_sub = A_list[v_i][obs_idx[:, None], obs_idx[None, :]]
                        L_sub = normalized_laplacian(A_sub)
                        L_full = torch.zeros_like(A_list[v_i])
                        L_full[obs_idx[:, None], obs_idx[None, :]] = L_sub
                        L_list[v_i] = L_full
                        _, _, own_tr, _ = spectral_eigendecomposition(A_sub, min(self.n_clusters, max(2, len(obs_idx) - 1)))
                        own_cuts[v_i] = (own_tr / max(1, self.n_clusters)) / null_cuts[v_i]
                    else:
                        L_list[v_i] = normalized_laplacian(A_list[v_i])
                        _, _, own_tr, _ = spectral_eigendecomposition(A_list[v_i], self.n_clusters)
                        own_cuts[v_i] = (own_tr / max(1, self.n_clusters)) / null_cuts[v_i]

            # Step B: Consensus affinity S update + Dual spectral-gap feedback controller (after warmup)
            S_weighted = fuse_weighted_affinities(w)
            if self.use_spectral_control and t >= self.warmup_iters and mu > 0.0:
                f_norms = F.norm(dim=1, keepdim=True).clamp_min(1e-8)
                F_unit_t = F / f_norms
                H = torch.cdist(F_unit_t, F_unit_t, p=2.0).pow(2)
                S_target = (S_weighted - (0.14 * mu / k_nn) * H).clamp_min(0.0)
            else:
                S_target = S_weighted.clamp_min(0.0)
            S_target.fill_diagonal_(0.0)
            row_st = S_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
            S_target = S_target / row_st
            S_target = 0.5 * (S_target + S_target.T)
            S = (1.0 - self.eta_s) * S + self.eta_s * S_target
            S = 0.5 * (S + S.T)

            F, d_inv_sqrt, spec_trace, gap = spectral_eigendecomposition(S, self.n_clusters)
            if self.use_spectral_control and t >= self.warmup_iters:
                mu = float(np.clip(mu + self.eta_mu * (spec_trace - self.eps_spec), 0.0, self.mu_max))

            # Step C: Closed-loop Dead-Zone PI / mirror-descent modality weight controller
            e_vec, _ = compute_errors_and_lyapunov(Z, P_list, U_list, Q_list, L_list, own_cuts, F, w, spec_trace)
            if self.use_feedback:
                w, I_err = _apply_deadzone_pi(
                    e_vec, I_err, w,
                    dead_zone=self.dead_zone,
                    beta_i=self.beta_i,
                    kappa_i=self.kappa_i,
                    eta_w=self.eta_w,
                    tau=self.tau,
                )

            _, V_t = compute_errors_and_lyapunov(Z, P_list, U_list, Q_list, L_list, own_cuts, F, w, spec_trace)
            weights_history.append(w.detach().cpu().tolist())
            lyapunov_history.append(V_t)
            spectral_gap_history.append(gap)
            spectral_trace_history.append(spec_trace)
            mu_history.append(mu)

        F_np = F.detach().cpu().numpy()
        row_norms = np.linalg.norm(F_np, axis=1, keepdims=True)
        row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
        F_unit = F_np / row_norms
        labels = KMeans(n_clusters=self.n_clusters, n_init=30, random_state=self.seed).fit_predict(F_unit)

        return {
            "labels": labels,
            "S": S.detach().cpu().numpy(),
            "Z": Z.detach().cpu().numpy(),
            "F": F_unit,
            "final_weights": w.detach().cpu().numpy(),
            "weights_history": np.array(weights_history),
            "lyapunov_history": np.array(lyapunov_history),
            "spectral_gap_history": np.array(spectral_gap_history),
            "spectral_trace_history": np.array(spectral_trace_history),
            "mu_history": np.array(mu_history),
        }


def build_bipartite_affinity(
    X: torch.Tensor, anchors: torch.Tensor, k_anchor: int = 15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-normalized n x m bipartite affinity B and column-scaled B_tilde = B * diag(d_B^{-1/2})."""
    m = anchors.shape[0]
    k_eff = max(2, min(k_anchor, m))
    dist2 = torch.cdist(X, anchors, p=2.0).pow(2)
    knn_vals, knn_idx = torch.topk(dist2, k=k_eff, dim=1, largest=False)
    sig2 = knn_vals.mean(dim=1, keepdim=True).clamp_min(1e-6)
    weights = torch.exp(-knn_vals / (2.0 * sig2 + 1e-8))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    B = torch.zeros((X.shape[0], m), dtype=X.dtype, device=X.device)
    row_ar = torch.arange(X.shape[0], device=X.device)[:, None]
    B[row_ar, knn_idx] = weights

    col_deg = B.sum(dim=0).clamp_min(1e-8)
    B_tilde = B * col_deg.rsqrt()[None, :]
    return B, B_tilde


def bipartite_spectral_eigendecomposition(
    B: torch.Tensor, n_clusters: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Exact O(n m^2 + m^3) normalized spectral decomposition of P = B D_B^{-1} B^T."""
    col_deg = B.sum(dim=0).clamp_min(1e-8)
    B_tilde = B * col_deg.rsqrt()[None, :]
    G = B_tilde.T @ B_tilde
    G = 0.5 * (G + G.T)
    evals, V_anc = torch.linalg.eigh(G)
    # eigh returns ascending; largest eigenvalues correspond to smallest Laplacian eigenvalues 1 - sigma^2
    sigma2 = evals.flip(0).clamp(min=1e-10, max=1.0)
    V_desc = V_anc.flip(1)

    c_eff = min(n_clusters, len(sigma2))
    s_top = sigma2[:c_eff].sqrt().clamp_min(1e-6)
    U_anc = V_desc[:, :c_eff]
    F = B_tilde @ (U_anc / s_top[None, :])

    lap_evals = (1.0 - sigma2).clamp_min(0.0)
    spec_trace = float(lap_evals[:c_eff].sum().item())
    gap = float((lap_evals[c_eff] - lap_evals[c_eff - 1]).item()) if len(lap_evals) > c_eff else 0.0
    return F, U_anc, B_tilde, spec_trace, gap


class CLVSSCAnchor:
    """Linear-time O(n m^2 + n d r) bipartite landmark closed-loop spectral controller for large cohorts."""

    def __init__(
        self,
        n_clusters: int,
        n_anchors: int = 384,
        anchor_k: int = 20,
        shared_dim: int = 24,
        spec_dim: int = 6,
        max_iter: int = 30,
        warmup_iters: int = 10,
        alpha_graph: float = 1.0,
        gamma_rec: float = 0.30,
        gamma_cons: float = 0.08,
        lambda_z: float = 2.0,
        beta_u: float = 0.35,
        eta_z: float = 0.35,
        eta_s: float = 0.35,
        rho_a: float = 0.25,
        eta_w: float = 0.30,
        beta_i: float = 0.4,
        kappa_i: float = 0.35,
        tau: float = 0.20,
        dead_zone: float = 0.0,
        mu_init: float = 0.0,
        eta_mu: float = 0.05,
        mu_max: float = 0.12,
        eps_spec: float = 0.02,
        use_feedback: bool = True,
        use_spectral_control: bool = True,
        use_variable_separation: bool = True,
        use_null_snr: bool = False,
        use_local_pi: bool = False,
        alpha_local: float = 0.25,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.n_clusters = n_clusters
        self.n_anchors = n_anchors
        self.anchor_k = anchor_k
        self.shared_dim = shared_dim
        self.spec_dim = spec_dim
        self.max_iter = max_iter
        self.warmup_iters = warmup_iters
        self.alpha_graph = alpha_graph
        self.gamma_rec = gamma_rec
        self.gamma_cons = gamma_cons
        self.lambda_z = lambda_z
        self.beta_u = beta_u
        self.eta_z = eta_z
        self.eta_s = eta_s
        self.rho_a = rho_a
        self.eta_w = eta_w
        self.beta_i = beta_i
        self.kappa_i = kappa_i
        self.tau = tau
        self.dead_zone = dead_zone
        self.mu_init = mu_init
        self.eta_mu = eta_mu
        self.mu_max = mu_max
        self.eps_spec = eps_spec
        self.use_feedback = use_feedback
        self.use_spectral_control = use_spectral_control
        self.use_variable_separation = use_variable_separation
        self.use_null_snr = use_null_snr
        self.use_local_pi = use_local_pi
        self.alpha_local = alpha_local
        self.seed = seed
        self.device = get_device(device)

    @torch.no_grad()
    def fit(
        self,
        views: List[np.ndarray],
        step_fault: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fit_start = time.perf_counter()

        X_list, _ = preprocess_views(views, self.device)
        V = len(X_list)
        n = X_list[0].shape[0]
        m = min(self.n_anchors, max(self.n_clusters + 4, n // 4))
        k_a = min(self.anchor_k, m)
        x_norms = [float(x.pow(2).sum().item()) + 1e-8 for x in X_list]
        op_norm2 = [float(fast_left_svd(x, 2)[1][0].pow(2).item()) + 1e-8 for x in X_list]

        X_cat = torch.cat(X_list, dim=1)
        r = min(self.shared_dim, n - 1, X_cat.shape[1])
        Z, _ = fast_left_svd(X_cat, r)
        P_list = [Z.T @ x for x in X_list]

        U_list: List[Optional[torch.Tensor]] = []
        Q_list: List[Optional[torch.Tensor]] = []
        for v_idx, x in enumerate(X_list):
            if self.use_variable_separation:
                r_v = max(1, min(self.spec_dim, n - r - 1 if n - r > 2 else max(1, n // 6), x.shape[1]))
                res = x - Z @ P_list[v_idx]
                res = res - Z @ (Z.T @ res)
                u_v, _ = fast_left_svd(res, r_v)
                q_v = u_v.T @ res
                U_list.append(u_v)
                Q_list.append(q_v)
            else:
                U_list.append(None)
                Q_list.append(None)

        dof_scales = [
            max(
                0.25,
                1.0
                - (r + (U_list[v_i].shape[1] if U_list[v_i] is not None else 0))
                / (min(n, X_list[v_i].shape[1]) + r + (U_list[v_i].shape[1] if U_list[v_i] is not None else 0)),
            )
            for v_i in range(V)
        ]

        # Select representative landmark cells via fast MiniBatchKMeans centroids mapped to nearest real cells
        z_np = Z.detach().cpu().numpy()
        km = MiniBatchKMeans(n_clusters=m, n_init=3, batch_size=min(2048, n), max_iter=25, random_state=self.seed)
        km.fit(z_np)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=self.device)
        d2_anc = torch.cdist(centers, Z, p=2.0)
        anc_idx = torch.argmin(d2_anc, dim=1)

        B_list: List[torch.Tensor] = []
        Bt_list: List[torch.Tensor] = []
        own_cuts: List[float] = []
        null_cuts: List[float] = []
        for v_idx, x in enumerate(X_list):
            x_sh = x - self.beta_u * (U_list[v_idx] @ Q_list[v_idx]) if self.use_variable_separation else x
            B_v, Bt_v = build_bipartite_affinity(x_sh, x_sh[anc_idx], k_anchor=k_a)
            _, _, _, own_tr, _ = bipartite_spectral_eigendecomposition(B_v, self.n_clusters)
            B_list.append(B_v)
            Bt_list.append(Bt_v)
            raw_cut = own_tr / max(1, self.n_clusters)
            if self.use_null_snr:
                c_n = _compute_anchor_null_cut(x_sh, k_anchor=k_a, n_anchors=m, n_clusters=self.n_clusters, seed=self.seed + 1000 + v_idx)
                null_cuts.append(c_n)
                own_cuts.append(raw_cut / c_n)
            else:
                null_cuts.append(1.0)
                own_cuts.append(raw_cut)

        w = torch.full((V,), 1.0 / V, dtype=torch.float32, device=self.device)
        I_err = torch.zeros((V,), dtype=torch.float32, device=self.device)
        W_loc = torch.full((m, V), 1.0 / V, dtype=torch.float32, device=self.device)
        I_loc = torch.zeros((m, V), dtype=torch.float32, device=self.device)
        B_S = sum(w[v] * B_list[v] for v in range(V))
        F, U_anc, Bt_S, spec_trace, gap = bipartite_spectral_eigendecomposition(B_S, self.n_clusters)
        mu = self.mu_init if self.use_spectral_control else 0.0

        weights_history: List[List[float]] = []
        lyapunov_history: List[float] = []
        spectral_gap_history: List[float] = []
        iter_wall_sec: List[float] = []

        def compute_errors_and_lyap(
            cur_Z: torch.Tensor,
            cur_P: List[torch.Tensor],
            cur_U: List[Optional[torch.Tensor]],
            cur_Q: List[Optional[torch.Tensor]],
            cur_Bt: List[torch.Tensor],
            cur_own: List[float],
            cur_F: torch.Tensor,
            cur_w: torch.Tensor,
            cur_trace: float,
        ) -> Tuple[torch.Tensor, float]:
            errs = []
            lyap = 0.0
            c_float = float(self.n_clusters)
            for v_i, x in enumerate(X_list):
                recon = cur_Z @ cur_P[v_i]
                if cur_U[v_i] is not None and cur_Q[v_i] is not None:
                    recon = recon + cur_U[v_i] @ cur_Q[v_i]
                rec_rel = (float((x - recon).pow(2).sum().item()) / x_norms[v_i]) / dof_scales[v_i]
                # Exact Tr(F^T (I - Bt_v Bt_v^T) F) = c - ||Bt_v^T F||_F^2
                proj = cur_Bt[v_i].T @ cur_F
                cons_cut = max(0.0, (c_float - float(proj.pow(2).sum().item()))) / max(1, self.n_clusters)
                own_cut = cur_own[v_i]
                e_v = own_cut + self.gamma_cons * cons_cut + self.gamma_rec * rec_rel
                errs.append(e_v)
                lyap += float(cur_w[v_i].item()) * (rec_rel + self.alpha_graph * (own_cut + self.gamma_cons * cons_cut))
            entropy_term = float((cur_w * torch.log(cur_w.clamp_min(1e-12))).sum().item())
            spec_weight = self.mu_max if self.use_spectral_control else 0.0
            lyap += spec_weight * cur_trace + self.tau * entropy_term
            return torch.tensor(errs, dtype=torch.float32, device=self.device), lyap

        _, V_t = compute_errors_and_lyap(Z, P_list, U_list, Q_list, Bt_list, own_cuts, F, w, spec_trace)
        weights_history.append(w.detach().cpu().tolist())
        lyapunov_history.append(V_t)
        spectral_gap_history.append(gap)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_wall_sec.append(float(time.perf_counter() - t_fit_start))

        inject_iter = int(step_fault["inject_iter"]) if step_fault is not None else -1

        for t in range(self.max_iter):
            if step_fault is not None and t == inject_iter:
                v_f = int(step_fault["view_idx"])
                f_sigma = float(step_fault.get("noise_std", 3.0))
                f_drop = float(step_fault.get("dropout_rate", 0.50))
                f_seed = int(step_fault.get("seed", 230))
                f_gen = torch.Generator(device=self.device)
                f_gen.manual_seed(f_seed)
                x_raw = X_list[v_f]
                feat_std = x_raw.std(dim=0, keepdim=True)
                feat_std = torch.where(feat_std < 1e-6, torch.ones_like(feat_std), feat_std)
                if f_sigma > 0.0:
                    noise = torch.randn(x_raw.shape, generator=f_gen, device=self.device, dtype=x_raw.dtype)
                    x_raw = x_raw + f_sigma * noise * feat_std
                if f_drop > 0.0:
                    keep = (torch.rand(x_raw.shape, generator=f_gen, device=self.device, dtype=x_raw.dtype) >= f_drop).to(x_raw.dtype)
                    x_raw = x_raw * keep
                mean_f = x_raw.mean(dim=0, keepdim=True)
                std_f = x_raw.std(dim=0, keepdim=True)
                std_f = torch.where(std_f < 1e-6, torch.ones_like(std_f), std_f)
                x_normed = ((x_raw - mean_f) / std_f) / float(np.sqrt(max(1, x_raw.shape[1])))
                X_list[v_f] = x_normed
                x_norms[v_f] = float(x_normed.pow(2).sum().item()) + 1e-8
                op_norm2[v_f] = float(fast_left_svd(x_normed, 2)[1][0].pow(2).item()) + 1e-8
                if self.use_variable_separation and U_list[v_f] is not None:
                    res_f = x_normed - Z @ (Z.T @ x_normed)
                    r_vf = U_list[v_f].shape[1]
                    u_vf, _ = fast_left_svd(res_f, r_vf)
                    u_vf = polar_orthogonalize(u_vf - Z @ (Z.T @ u_vf))
                    U_list[v_f] = u_vf
                    Q_list[v_f] = u_vf.T @ res_f
                    P_list[v_f] = Z.T @ (x_normed - u_vf @ Q_list[v_f])
                    x_sh_f = x_normed - self.beta_u * (u_vf @ Q_list[v_f])
                else:
                    P_list[v_f] = Z.T @ x_normed
                    x_sh_f = x_normed
                B_new_f, Bt_new_f = build_bipartite_affinity(x_sh_f, x_sh_f[anc_idx], k_anchor=k_a)
                B_list[v_f] = B_new_f
                Bt_list[v_f] = Bt_new_f
                _, _, _, own_tr_f, _ = bipartite_spectral_eigendecomposition(B_new_f, self.n_clusters)
                raw_cut_f = own_tr_f / max(1, self.n_clusters)
                if self.use_null_snr:
                    c_n_f = _compute_anchor_null_cut(
                        x_sh_f, k_anchor=k_a, n_anchors=m, n_clusters=self.n_clusters, seed=self.seed + 1000 + v_f
                    )
                    null_cuts[v_f] = c_n_f
                    own_cuts[v_f] = raw_cut_f / c_n_f
                else:
                    null_cuts[v_f] = 1.0
                    own_cuts[v_f] = raw_cut_f

            # Step A: Orthogonal Variable Separation update using O(n m r) bipartite operator Bt_S @ (Bt_S.T @ Z)
            grad_Z = self.lambda_z * (Bt_S @ (Bt_S.T @ Z))
            for v_i, x in enumerate(X_list):
                x_sh = x - (U_list[v_i] @ Q_list[v_i]) if self.use_variable_separation else x
                P_list[v_i] = Z.T @ x_sh
                grad_Z = grad_Z + (float(w[v_i].item()) / op_norm2[v_i]) * (x_sh @ P_list[v_i].T)

            Z_target = polar_orthogonalize(grad_Z)
            Z = polar_orthogonalize((1.0 - self.eta_z) * Z + self.eta_z * Z_target)

            for v_i, x in enumerate(X_list):
                if self.use_variable_separation:
                    res = x - Z @ (Z.T @ x)
                    r_v = U_list[v_i].shape[1]
                    U_cand, _ = fast_left_svd(res, r_v)
                    U_cand = U_cand - Z @ (Z.T @ U_cand)
                    U_cand = polar_orthogonalize(U_cand)
                    U_list[v_i] = U_cand
                    Q_list[v_i] = U_cand.T @ res
                    P_list[v_i] = Z.T @ (x - U_list[v_i] @ Q_list[v_i])
                else:
                    P_list[v_i] = Z.T @ x

            # Refresh bipartite graphs periodically
            if self.use_variable_separation and (t + 1) % 10 == 0 and t < self.max_iter - 5:
                for v_i, x in enumerate(X_list):
                    x_sh = x - self.beta_u * (U_list[v_i] @ Q_list[v_i])
                    B_new, _ = build_bipartite_affinity(x_sh, x_sh[anc_idx], k_anchor=k_a)
                    B_v = (1.0 - self.rho_a) * B_list[v_i] + self.rho_a * B_new
                    B_v = B_v / B_v.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    col_deg = B_v.sum(dim=0).clamp_min(1e-8)
                    B_list[v_i] = B_v
                    Bt_list[v_i] = B_v * col_deg.rsqrt()[None, :]
                    _, _, _, own_tr, _ = bipartite_spectral_eigendecomposition(B_v, self.n_clusters)
                    own_cuts[v_i] = (own_tr / max(1, self.n_clusters)) / null_cuts[v_i]

            # Step B: Consensus bipartite affinity B_S + Dual spectral-gap feedback
            if self.use_local_pi:
                B_weighted = sum(B_list[v] * W_loc[:, v][None, :] for v in range(V))
            else:
                B_weighted = sum(w[v] * B_list[v] for v in range(V))
            if self.use_spectral_control and t >= self.warmup_iters and mu > 0.0:
                F_u = F / F.norm(dim=1, keepdim=True).clamp_min(1e-8)
                U_u = U_anc / U_anc.norm(dim=1, keepdim=True).clamp_min(1e-8)
                H_anc = torch.cdist(F_u, U_u, p=2.0).pow(2)
                B_tgt = (B_weighted - (0.14 * mu / k_a) * H_anc).clamp_min(0.0) + 1e-5 * B_weighted
            else:
                B_tgt = B_weighted.clamp_min(0.0)
            B_tgt = B_tgt / B_tgt.sum(dim=1, keepdim=True).clamp_min(1e-8)
            B_S = (1.0 - self.eta_s) * B_S + self.eta_s * B_tgt
            B_S = B_S / B_S.sum(dim=1, keepdim=True).clamp_min(1e-8)

            F, U_anc, Bt_S, spec_trace, gap = bipartite_spectral_eigendecomposition(B_S, self.n_clusters)
            if self.use_spectral_control and t >= self.warmup_iters:
                mu = float(np.clip(mu + self.eta_mu * (spec_trace - self.eps_spec), 0.0, self.mu_max))

            # Step C: Closed-loop global or Anchor-Local PI modality weight controller
            e_vec, _ = compute_errors_and_lyap(Z, P_list, U_list, Q_list, Bt_list, own_cuts, F, w, spec_trace)
            if self.use_feedback:
                if self.use_local_pi:
                    F_u_now = F / F.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    F_anc = (B_S.T @ F_u_now) / B_S.sum(dim=0).clamp_min(1e-8)[:, None]
                    F_anc_u = F_anc / F_anc.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    H_now = torch.cdist(F_u_now, F_anc_u, p=2.0).pow(2)
                    loc_cols = []
                    for v_i in range(V):
                        c_sum = B_list[v_i].sum(dim=0)
                        raw_l = 0.5 * (B_list[v_i] * H_now).sum(dim=0) / c_sum.clamp_min(1e-6)
                        fb_val = float(raw_l[c_sum > 1e-4].mean().item()) if (c_sum > 1e-4).any() else 0.0
                        loc_cols.append(torch.where(c_sum > 1e-4, raw_l, torch.full_like(raw_l, fb_val)))
                    loc_mat = torch.stack(loc_cols, dim=1)  # (m, V)
                    # Double-center across anchors (zero global bias) and views, then bound deviations
                    loc_c = loc_mat - loc_mat.mean(dim=0, keepdim=True)
                    loc_dev = torch.clamp(loc_c - loc_c.mean(dim=1, keepdim=True), -0.25, 0.25)
                    e_loc = e_vec[None, :] + self.alpha_local * loc_dev
                    W_loc, I_loc = _apply_deadzone_pi(
                        e_loc, I_loc, W_loc,
                        dead_zone=self.dead_zone,
                        beta_i=self.beta_i,
                        kappa_i=self.kappa_i,
                        eta_w=self.eta_w,
                        tau=self.tau,
                    )
                    w = W_loc.mean(dim=0)
                else:
                    w, I_err = _apply_deadzone_pi(
                        e_vec, I_err, w,
                        dead_zone=self.dead_zone,
                        beta_i=self.beta_i,
                        kappa_i=self.kappa_i,
                        eta_w=self.eta_w,
                        tau=self.tau,
                    )

            _, V_t = compute_errors_and_lyap(Z, P_list, U_list, Q_list, Bt_list, own_cuts, F, w, spec_trace)
            weights_history.append(w.detach().cpu().tolist())
            lyapunov_history.append(V_t)
            spectral_gap_history.append(gap)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iter_wall_sec.append(float(time.perf_counter() - t_fit_start))

        F_np = F.detach().cpu().numpy()
        row_norms = np.linalg.norm(F_np, axis=1, keepdims=True)
        row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
        F_unit = F_np / row_norms
        if n <= 100000:
            labels = KMeans(n_clusters=self.n_clusters, n_init=20 if n <= 15000 else 10, random_state=self.seed).fit_predict(F_unit)
        else:
            labels = MiniBatchKMeans(
                n_clusters=self.n_clusters, n_init=10, batch_size=8192, random_state=self.seed
            ).fit_predict(F_unit)

        res_dict = {
            "labels": labels,
            "Z": Z.detach().cpu().numpy(),
            "F": F_unit,
            "final_weights": w.detach().cpu().numpy(),
            "weights_history": np.array(weights_history),
            "lyapunov_history": np.array(lyapunov_history),
            "spectral_gap_history": np.array(spectral_gap_history),
            "iter_wall_sec": np.array(iter_wall_sec),
        }
        if self.use_local_pi:
            res_dict["local_weights"] = W_loc.detach().cpu().numpy()
        return res_dict
