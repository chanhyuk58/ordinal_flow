# ========================= ordinal_flow_core.py =========================

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline as uRQS,
)
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow
from statsmodels.miscmodels.ordinal_model import OrderedModel

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SQRT2 = math.sqrt(2.0)
LOG_HALF = math.log(0.5)


def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    y = torch.clamp(y, min=eps)
    return torch.log(torch.expm1(y))


def normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def category_effect_from_probs(p1: np.ndarray, p0: np.ndarray) -> np.ndarray:
    p1 = normalize_probs(p1)
    p0 = normalize_probs(p0)
    return np.mean(p1 - p0, axis=0)


def cumulative_ge_from_probs(p: np.ndarray) -> np.ndarray:
    p = normalize_probs(p)
    return np.cumsum(p[:, ::-1], axis=1)[:, ::-1]


def cumulative_ge_effect_from_probs(p1: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return np.mean(cumulative_ge_from_probs(p1) - cumulative_ge_from_probs(p0), axis=0)


def wasserstein_unit_from_probs(p1: np.ndarray, p0: np.ndarray) -> float:
    p1_bar = normalize_probs(p1).mean(axis=0)
    p0_bar = normalize_probs(p0).mean(axis=0)
    cdf1 = np.cumsum(p1_bar)
    cdf0 = np.cumsum(p0_bar)
    return float(np.sum(np.abs(cdf1[:-1] - cdf0[:-1])))

def effects_from_counterfactual_probs(p1: np.ndarray, p0: np.ndarray) -> Dict[str, object]:
    p1 = normalize_probs(p1)
    p0 = normalize_probs(p0)
    return {
        "category_effect": category_effect_from_probs(p1, p0),
        "cum_ge_effect": cumulative_ge_effect_from_probs(p1, p0),
        "wasserstein_unit": wasserstein_unit_from_probs(p1, p0),
    }


def empirical_probs_by_treatment(y: np.ndarray, d: np.ndarray, J: Optional[int] = None) -> Dict[str, np.ndarray]:
    y = np.asarray(y, dtype=int)
    d = np.asarray(d, dtype=int)
    if J is None:
        J = int(np.max(y))

    p = {}
    for val in [0, 1]:
        mask = d == val
        if not np.any(mask):
            raise ValueError(f"No observations with D={val}")
        p[val] = np.array([(y[mask] == j).mean() for j in range(1, J + 1)], dtype=float)

    p0 = np.tile(p[0], (len(y), 1))
    p1 = np.tile(p[1], (len(y), 1))
    return {
        "p0": p[0],
        "p1": p[1],
        "category_effect": p[1] - p[0],
        "cum_ge_effect": cumulative_ge_effect_from_probs(p1, p0),
        "wasserstein_unit": wasserstein_unit_from_probs(p1, p0),
    }

# ----------------------------------------------------------------------
# Global 1D RQS Transform
# ----------------------------------------------------------------------
class GlobalRQS1D(nn.Module):
    def __init__(self, K=16, bounds=8.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
        super().__init__()
        self.K = K
        self.bound = float(bounds)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.unnorm_widths = nn.Parameter(torch.zeros(K))
        self.unnorm_heights = nn.Parameter(torch.zeros(K))
        init_deriv = softplus_inv(torch.tensor(1.0))
        self.unnorm_derivs = nn.Parameter(torch.full((K + 1,), init_deriv))

    def _expand_params(self, batch_size):
        w = self.unnorm_widths.view(1, 1, -1).expand(batch_size, 1, self.K)
        h = self.unnorm_heights.view(1, 1, -1).expand(batch_size, 1, self.K)
        d = self.unnorm_derivs.view(1, 1, -1).expand(batch_size, 1, self.K + 1)
        return w, h, d

    def forward_with_logabsdet(self, z):
        z_in = z.unsqueeze(-1) if z.dim() == 1 else z
        B = z_in.shape[0]
        w, h, d = self._expand_params(B)
        y, ladj = uRQS(
            inputs=z_in,
            unnormalized_widths=w,
            unnormalized_heights=h,
            unnormalized_derivatives=d,
            inverse=False,
            tails='linear',
            tail_bound=self.bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        return y.squeeze(-1), ladj.squeeze(-1)

    def forward(self, z):
        y, _ = self.forward_with_logabsdet(z)
        return y

    def inverse_with_logabsdet(self, e):
        e_in = e.unsqueeze(-1) if e.dim() == 1 else e
        B = e_in.shape[0]
        w, h, d = self._expand_params(B)
        z, ladj = uRQS(
            inputs=e_in,
            unnormalized_widths=w,
            unnormalized_heights=h,
            unnormalized_derivatives=d,
            inverse=True,
            tails='linear',
            tail_bound=self.bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        return z.squeeze(-1), ladj.squeeze(-1)

    def inverse(self, e):
        z, _ = self.inverse_with_logabsdet(e)
        return z

    def sample_base(self, n):
        return torch.randn(n, device=self.unnorm_widths.device)

# ----------------------------------------------------------------------
# Baseline Semiparametric Ordered Flow Model
# ----------------------------------------------------------------------
class OrderedFlowModel(nn.Module):
    def __init__(self, p, J, q=0, flow_bins=16, bounds=8.0, min_gap=1e-4):
        super().__init__()
        assert J >= 2
        self.p = p
        self.q = q
        self.J = int(J)
        self.min_gap = min_gap

        self.beta = nn.Parameter(torch.zeros(p))
        self.alpha1_intercept = nn.Parameter(torch.tensor(0.0))
        if q > 0:
            self.alpha1_gamma = nn.Parameter(torch.zeros(q))
        else:
            self.register_parameter('alpha1_gamma', None)

        if self.J > 2:
            self.gap_intercepts_raw = nn.Parameter(torch.full((self.J - 2,), softplus_inv(torch.tensor(0.5))))
            if q > 0:
                self.gap_gammas = nn.Parameter(torch.zeros(self.J - 2, q))
            else:
                self.register_parameter('gap_gammas', None)
        else:
            self.register_parameter('gap_intercepts_raw', None)
            self.register_parameter('gap_gammas', None)

        # Optimization Anchors: a and b fixed to 1.0 and 0.0 respectively
        self.a_raw = nn.Parameter(softplus_inv(torch.tensor(1.0)), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.flow = GlobalRQS1D(K=flow_bins, bounds=bounds)

    def a(self):
        return torch.nn.functional.softplus(self.a_raw)

    def alphas_obs(self, Z=None):
        n = 1 if Z is None else Z.shape[0]
        if self.q == 0 or Z is None:
            alpha1 = self.alpha1_intercept.expand(n)
            if self.J == 2:
                return alpha1.unsqueeze(1)
            gaps = torch.nn.functional.softplus(self.gap_intercepts_raw) + self.min_gap
            alphas = torch.empty(n, self.J - 1, dtype=alpha1.dtype, device=alpha1.device)
            alphas[:, 0] = alpha1
            for k in range(1, self.J - 1):
                alphas[:, k] = alphas[:, k-1] + gaps[k-1]
            return alphas
        else:
            alpha1 = self.alpha1_intercept + Z.matmul(self.alpha1_gamma)
            alphas = torch.empty(Z.shape[0], self.J - 1, dtype=Z.dtype, device=Z.device)
            alphas[:, 0] = alpha1
            if self.J > 2:
                gaps = torch.nn.functional.softplus(
                    self.gap_intercepts_raw.unsqueeze(0) + Z.matmul(self.gap_gammas.T)
                ) + self.min_gap
                for k in range(1, self.J - 1):
                    alphas[:, k] = alphas[:, k-1] + gaps[:, k-1]
            return alphas

    def predict_proba(self, X, Z=None):
        n = X.shape[0]
        eta = X.matmul(self.beta)
        alphas_mat = self.alphas_obs(Z)

        # Completely vectorized boundaries construction
        B = torch.empty(n, self.J + 1, dtype=X.dtype, device=X.device)
        B[:, 0] = -torch.inf
        B[:, -1] = torch.inf
        B[:, 1:-1] = alphas_mat

        E = B - eta.unsqueeze(1)
        a = self.a()

        E_flat = E.reshape(-1)
        Z_flat = torch.empty_like(E_flat)
        finite_mask = torch.isfinite(E_flat)
        
        Z_flat[E_flat == -torch.inf] = -torch.inf
        Z_flat[E_flat == torch.inf] = torch.inf

        if finite_mask.any():
            E_finite = E_flat[finite_mask]
            Z_flat[finite_mask] = self.flow.inverse((E_finite - self.b) / a)

        z = Z_flat.reshape(n, self.J + 1)
        cdf_u = torch.full_like(z, torch.nan)
        neg_inf = torch.isneginf(z)
        pos_inf = torch.isposinf(z)
        
        cdf_u[neg_inf] = 0.0
        cdf_u[pos_inf] = 1.0

        finite_z = torch.isfinite(z)
        if finite_z.any():
            cdf_u[finite_z] = Normal(0.0, 1.0).cdf(z[finite_z])

        probs = cdf_u[:, 1:] - cdf_u[:, :-1]
        return torch.clamp(probs, min=1e-12, max=1.0)

    def neg_loglik(self, X, y, Z=None):
        probs = self.predict_proba(X, Z)
        y_idx = (y.long() - 1).view(-1, 1)
        p_y = probs.gather(1, y_idx).squeeze(1)
        return -torch.mean(torch.log(p_y))

    @torch.no_grad()
    def init_from_ordered_probit(self, X, y, Z=None, verbose=True):
        y0 = y.cpu().numpy().astype(int) - 1
        X_np = X.cpu().numpy()
        cols = [f"x{k}" for k in range(X_np.shape[1])]
        dfX = pd.DataFrame(X_np, columns=cols)
        mod = OrderedModel(y0, dfX, distr='probit')
        res = mod.fit(method='bfgs', disp=False)

        beta_hat = res.params[cols].values
        thr_vals = res.params.values[-(self.J - 1):]

        self.beta.data = torch.tensor(beta_hat, dtype=torch.float64, device=self.beta.device)
        self.alpha1_intercept.data = torch.tensor(thr_vals[0], dtype=torch.float64, device=self.beta.device)
        if self.q > 0:
            self.alpha1_gamma.data = torch.zeros_like(self.alpha1_gamma)
        if self.J > 2:
            gaps = np.diff(thr_vals)
            gaps = np.maximum(gaps, self.min_gap * 10)
            self.gap_intercepts_raw.data = softplus_inv(
                torch.tensor(gaps - self.min_gap, dtype=torch.float64, device=self.beta.device)
            )
            if self.q > 0 and self.gap_gammas is not None:
                self.gap_gammas.data = torch.zeros_like(self.gap_gammas)

        self._probit_nll = float(-res.llf)

    

    @torch.no_grad()
    def latent_coefficients(self, names: Optional[list[str]] = None) -> pd.Series:
        beta = self.beta.detach().cpu().numpy()
        if names is None:
            names = [f"x{k}" for k in range(len(beta))]
        return pd.Series(beta, index=names, name="structured_flow_beta")
    
    @torch.no_grad()
    def counterfactual_probs(self, X: torch.Tensor, treatment_idx: int = 0, Z: Optional[torch.Tensor] = None):
        X = X.to(device=device, dtype=torch.float64)
        X1 = X.clone()
        X0 = X.clone()
        X1[:, treatment_idx] = 1.0
        X0[:, treatment_idx] = 0.0
        p1 = self.predict_proba(X1, Z).cpu().numpy()
        p0 = self.predict_proba(X0, Z).cpu().numpy()
        return p1, p0

    @torch.no_grad()
    def compute_ame(self, X: torch.Tensor, treatment_idx: int = 0, Z: Optional[torch.Tensor] = None) -> np.ndarray:
        p1, p0 = self.counterfactual_probs(X, treatment_idx=treatment_idx, Z=Z)
        return category_effect_from_probs(p1, p0)

    @torch.no_grad()
    def compute_cumulative_ge_effect(self, X: torch.Tensor, treatment_idx: int = 0, Z: Optional[torch.Tensor] = None) -> np.ndarray:
        p1, p0 = self.counterfactual_probs(X, treatment_idx=treatment_idx, Z=Z)
        return cumulative_ge_effect_from_probs(p1, p0)

    @torch.no_grad()
    def compute_wasserstein_unit(self, X: torch.Tensor, treatment_idx: int = 0, Z: Optional[torch.Tensor] = None) -> float:
        p1, p0 = self.counterfactual_probs(X, treatment_idx=treatment_idx, Z=Z)
        return wasserstein_unit_from_probs(p1, p0)

# ----------------------------------------------------------------------
# Model-Free Conditional Normalizing Flow Model (CNF)
# ----------------------------------------------------------------------
class ModelFreeConditionalFlowModel(nn.Module):
    def __init__(self, p, J, flow_bins=16, bounds=8.0, min_gap=1e-4, hidden_features=32):
        super().__init__()
        self.p = p
        self.J = int(J)
        self.min_gap = min_gap
        context_dim = 1 + p  # Treatment D + Covariates X

        # standard 1D Conditional Spline Flow
        transform = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=1,
            hidden_features=hidden_features,
            context_features=context_dim,
            num_bins=flow_bins,
            tail_bound=bounds,
            tails='linear'
        )
        base_dist = StandardNormal(shape=[1])
        self.flow = Flow(transform=transform, distribution=base_dist)

        # Static Global Thresholds
        self.alpha1_intercept = nn.Parameter(torch.tensor(0.0))
        if self.J > 2:
            self.gap_intercepts_raw = nn.Parameter(torch.full((self.J - 2,), softplus_inv(torch.tensor(0.5))))
        else:
            self.register_parameter('gap_intercepts_raw', None)

    def get_alphas(self, n, device):
        alpha1 = self.alpha1_intercept.expand(n)
        if self.J == 2:
            return alpha1.unsqueeze(1)
        gaps = torch.nn.functional.softplus(self.gap_intercepts_raw) + self.min_gap
        alphas = torch.empty(n, self.J - 1, dtype=alpha1.dtype, device=device)
        alphas[:, 0] = alpha1
        for k in range(1, self.J - 1):
            alphas[:, k] = alphas[:, k-1] + gaps[k-1]
        return alphas

    def predict_proba(self, X, D):
        n = X.shape[0]
        alphas_mat = self.get_alphas(n, X.device)

        # Boundary matrix (n, J+1)
        B = torch.empty(n, self.J + 1, dtype=X.dtype, device=X.device)
        B[:, 0] = -torch.inf
        B[:, -1] = torch.inf
        B[:, 1:-1] = alphas_mat

        B_flat = B.reshape(-1, 1)
        
        # Build context C = [D, X] of shape (n, 1+p)
        D_col = D.reshape(-1, 1).to(X.dtype)
        C = torch.cat([D_col, X], dim=1)

        # Vectorized context replication
        C_expanded = C.unsqueeze(1).expand(-1, self.J + 1, -1).reshape(-1, 1 + self.p)

        z_flat = torch.full_like(B_flat, torch.nan)
        z_flat[B_flat == -torch.inf] = -torch.inf
        z_flat[B_flat == torch.inf] = torch.inf

        finite_mask = torch.isfinite(B_flat).squeeze(1)
        if finite_mask.any():
            B_finite = B_flat[finite_mask]
            C_finite = C_expanded[finite_mask]
            z_finite, _ = self.flow._transform.forward(B_finite, context=C_finite)
            z_flat[finite_mask, :] = z_finite

        z = z_flat.reshape(n, self.J + 1)
        cdf_u = torch.full_like(z, torch.nan)
        neg_inf = torch.isneginf(z)
        pos_inf = torch.isposinf(z)
        
        cdf_u[neg_inf] = 0.0
        cdf_u[pos_inf] = 1.0

        finite_z = torch.isfinite(z)
        if finite_z.any():
            cdf_u[finite_z] = Normal(0.0, 1.0).cdf(z[finite_z])

        probs = cdf_u[:, 1:] - cdf_u[:, :-1]
        return torch.clamp(probs, min=1e-12, max=1.0)

    def neg_loglik(self, X, y, D):
        probs = self.predict_proba(X, D)
        y_idx = (y.long() - 1).view(-1, 1)
        p_y = probs.gather(1, y_idx).squeeze(1)
        return -torch.mean(torch.log(p_y))

    @torch.no_grad()
    def counterfactual_probs(self, X: torch.Tensor):
        n = X.shape[0]
        D1 = torch.ones(n, device=X.device, dtype=X.dtype)
        D0 = torch.zeros(n, device=X.device, dtype=X.dtype)
        p1 = self.predict_proba(X, D1).cpu().numpy()
        p0 = self.predict_proba(X, D0).cpu().numpy()
        return p1, p0

    @torch.no_grad()
    def compute_ame(self, X: torch.Tensor) -> np.ndarray:
        p1, p0 = self.counterfactual_probs(X)
        return category_effect_from_probs(p1, p0)

    @torch.no_grad()
    def compute_cumulative_ge_effect(self, X: torch.Tensor) -> np.ndarray:
        p1, p0 = self.counterfactual_probs(X)
        return cumulative_ge_effect_from_probs(p1, p0)

    @torch.no_grad()
    def compute_wasserstein_unit(self, X: torch.Tensor) -> float:
        p1, p0 = self.counterfactual_probs(X)
        return wasserstein_unit_from_probs(p1, p0)

# ----------------------------------------------------------------------
# Model Training Routines
# ----------------------------------------------------------------------
def train_ordered_flow(X, y, Z=None, flow_bins=16, bounds=10.0, epochs=200, lr=1e-3, use_lbfgs=True, lbfgs_steps=50, init_probit=True, verbose=False):
    X = X.to(device)
    y = y.to(device)
    J = int(torch.max(y))
    q = Z.shape[1] if Z is not None else 0
    if Z is not None: Z = Z.to(device)

    model = OrderedFlowModel(p=X.shape[1], J=J, q=q, flow_bins=flow_bins, bounds=bounds).to(device)

    baseline_nll = None
    baseline_state = None
    if init_probit:
        try:
            model.init_from_ordered_probit(X.cpu(), y.cpu(), Z.cpu() if Z is not None else None, verbose)
            baseline_nll = float(model._probit_nll)
            baseline_state = {k: v.clone() for k, v in model.state_dict().items()}
        except Exception:
            pass

    # Adam Warm-up (filtering out requires_grad=False parameters)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    adam_opt = optim.Adam(trainable_params, lr=lr)
    for ep in range(1, epochs + 1):
        adam_opt.zero_grad()
        loss = model.neg_loglik(X, y, Z)
        if torch.isnan(loss): break
        loss.backward()
        # Safe gradient clipping inside Adam phase
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adam_opt.step()

    # L-BFGS Polish
    if use_lbfgs:
        trainable_params_list = list(filter(lambda p: p.requires_grad, model.parameters()))
        lbfgs_opt = optim.LBFGS(
            trainable_params_list,
            lr=1.0,
            max_iter=20,
            history_size=20,
            line_search_fn='strong_wolfe'
        )
        def closure():
            lbfgs_opt.zero_grad()
            loss = model.neg_loglik(X, y, Z)
            if torch.isnan(loss): return loss
            loss.backward()
            return loss

        for i in range(1, lbfgs_steps + 1):
            try:
                loss = lbfgs_opt.step(closure)
                cur_nll = float(loss.detach())
                if baseline_nll is None or cur_nll < baseline_nll:
                    baseline_nll = cur_nll
                    baseline_state = {k: v.clone() for k, v in model.state_dict().items()}
                if torch.isnan(loss): break
            except Exception:
                break

    if baseline_state is not None:
        model.load_state_dict(baseline_state)
    return model


def train_model_free_flow(X, y, D, flow_bins=12, bounds=10.0, epochs=200, lr=1e-3, use_lbfgs=True, lbfgs_steps=30, init_probit=True, verbose=False):
    X = X.to(device)
    y = y.to(device)
    D = D.to(device)
    J = int(torch.max(y))
    p = X.shape[1]

    model = ModelFreeConditionalFlowModel(p=p, J=J, flow_bins=flow_bins, bounds=bounds).to(device)

    # Static Thresholds Probit Initialization
    if init_probit:
        try:
            D_col = D.reshape(-1, 1).to(X.dtype)
            DX = torch.cat([D_col, X], dim=1).cpu().numpy()
            y0 = y.cpu().numpy().astype(int) - 1
            cols = [f"x{k}" for k in range(DX.shape[1])]
            dfX = pd.DataFrame(DX, columns=cols)
            mod = OrderedModel(y0, dfX, distr='probit')
            res = mod.fit(method='bfgs', disp=False)
            thr_vals = res.params.values[-J + 1:]
            model.alpha1_intercept.data = torch.tensor(thr_vals[0], dtype=torch.float64, device=device)
            if J > 2:
                gaps = np.diff(thr_vals)
                gaps = np.maximum(gaps, model.min_gap * 10)
                model.gap_intercepts_raw.data = softplus_inv(
                    torch.tensor(gaps - model.min_gap, dtype=torch.float64, device=device)
                )
        except Exception as e:
            if verbose:
                print("CNF Probit threshold initialization skipped:", e)

    opt_adam = optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        opt_adam.zero_grad()
        loss = model.neg_loglik(X, y, D)
        if torch.isnan(loss): break
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_adam.step()

    if use_lbfgs:
        opt_lbfgs = optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=20, history_size=20, line_search_fn='strong_wolfe'
        )
        def closure():
            opt_lbfgs.zero_grad()
            loss = model.neg_loglik(X, y, D)
            if torch.isnan(loss): return loss
            loss.backward()
            return loss

        for i in range(1, lbfgs_steps + 1):
            try:
                loss = opt_lbfgs.step(closure)
                if torch.isnan(loss): break
            except Exception:
                break
    return model


# ----------------------------------------------------------------------
# Standard Statsmodels Baseline Fits
# ----------------------------------------------------------------------
def fit_ordered_sm(y: torch.Tensor, X: torch.Tensor, link: str = "probit", normalize: bool = True, norm_criterion: str = "variance"):
    y0 = y.detach().cpu().numpy().astype(int) - 1
    X_np = X.detach().cpu().numpy()
    cols = [f"x{k}" for k in range(X_np.shape[1])]
    dfX = pd.DataFrame(X_np, columns=cols)
    distr = "probit" if link == "probit" else "logit"
    mod = OrderedModel(y0, dfX, distr=distr)
    res = mod.fit(method="bfgs", disp=False)

    params = res.params
    beta_hat = params[cols].values.astype(float)
    thr = params.values[-(len(np.unique(y0)) - 1):].astype(float)

    beta_report = beta_hat.copy()
    thr_report = thr.copy()
    if normalize and link == "logit":
        s_link = math.pi / math.sqrt(3.0) if norm_criterion == "variance" else 1.6
        beta_report = beta_report / s_link
        thr_report = thr_report / s_link

    return beta_report, thr_report, res


def predict_ordered_sm(res, X: torch.Tensor | np.ndarray) -> np.ndarray:
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=float)
    cols = [f"x{k}" for k in range(X_np.shape[1])]
    dfX = pd.DataFrame(X_np, columns=cols)
    probs = np.asarray(res.model.predict(res.params, exog=dfX), dtype=float)
    return normalize_probs(probs)


def counterfactual_probs_ordered_sm(res, X: torch.Tensor | np.ndarray, treatment_idx: int = 0):
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=float)
    X1 = X_np.copy()
    X0 = X_np.copy()
    X1[:, treatment_idx] = 1.0
    X0[:, treatment_idx] = 0.0
    return predict_ordered_sm(res, X1), predict_ordered_sm(res, X0)


def ordered_sm_effects(res, X: torch.Tensor | np.ndarray, treatment_idx: int = 0) -> Dict[str, object]:
    p1, p0 = counterfactual_probs_ordered_sm(res, X, treatment_idx=treatment_idx)
    out = effects_from_counterfactual_probs(p1, p0)
    out["p1"] = p1
    out["p0"] = p0
    return out
