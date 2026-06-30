# ========================= ordinal_flow_core.py =========================

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Uncomment the next line if you still experience CPU thrashing
# os.environ["OMP_NUM_THREADS"] = "4" 

import math
from typing import Dict, Optional, Tuple, Any, List

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

# ----------------------------------------------------------------------
# Numerical & Statistical Helpers
# ----------------------------------------------------------------------

def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    y = torch.clamp(y, min=eps)
    return torch.log(torch.expm1(y))

def log_ndtr(z: torch.Tensor) -> torch.Tensor:
    return LOG_HALF + torch.log(torch.special.erfc(-z / SQRT2) + 1e-12)

def log_surv_ndtr(z: torch.Tensor) -> torch.Tensor:
    return LOG_HALF + torch.log(torch.special.erfc(z / SQRT2) + 1e-12)

def log_diff_normal_cdfs(zl: torch.Tensor, zu: torch.Tensor) -> torch.Tensor:
    assert zl.shape == zu.shape
    left_mask = (zu <= 0) & (zl <= 0)
    right_mask = (zu >= 0) & (zl >= 0)
    mid_mask = ~(left_mask | right_mask)
    out = torch.empty_like(zu)

    if left_mask.any():
        a = log_ndtr(zu[left_mask])
        b = log_ndtr(zl[left_mask])
        out[left_mask] = a + torch.log1p(-torch.exp(b - a) + 1e-12)

    if right_mask.any():
        a = log_surv_ndtr(zl[right_mask])
        b = log_surv_ndtr(zu[right_mask])
        out[right_mask] = a + torch.log1p(-torch.exp(b - a) + 1e-12)

    if mid_mask.any():
        phi_u = Normal(0.0, 1.0).cdf(zu[mid_mask])
        phi_l = Normal(0.0, 1.0).cdf(zl[mid_mask])
        diff = torch.clamp(phi_u - phi_l, min=1e-12)
        out[mid_mask] = torch.log(diff)
    return out

def normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)

# ----------------------------------------------------------------------
# Probability Estimands
# ----------------------------------------------------------------------

def category_effect_from_probs(p1: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return np.mean(normalize_probs(p1) - normalize_probs(p0), axis=0)

def cumulative_ge_from_probs(p: np.ndarray) -> np.ndarray:
    p = normalize_probs(p)
    return np.cumsum(p[:, ::-1], axis=1)[:, ::-1]

def cumulative_ge_effect_from_probs(p1: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return np.mean(cumulative_ge_from_probs(p1) - cumulative_ge_from_probs(p0), axis=0)[1:]

def wasserstein_unit_from_probs(p1: np.ndarray, p0: np.ndarray) -> float:
    p1_bar = normalize_probs(p1).mean(axis=0)
    p0_bar = normalize_probs(p0).mean(axis=0)
    cdf1 = np.cumsum(p1_bar)
    cdf0 = np.cumsum(p0_bar)
    return float(np.sum(np.abs(cdf1[:-1] - cdf0[:-1])))

def effects_from_counterfactual_probs(p1: np.ndarray, p0: np.ndarray) -> Dict[str, object]:
    p1, p0 = normalize_probs(p1), normalize_probs(p0)
    return {
        "category_effect": category_effect_from_probs(p1, p0),
        "cum_ge_effect": cumulative_ge_effect_from_probs(p1, p0),
        "wasserstein_unit": wasserstein_unit_from_probs(p1, p0),
    }

# ----------------------------------------------------------------------
# Core Neural Network Modules
# ----------------------------------------------------------------------

class GlobalRQS1D(nn.Module):
    def __init__(self, K=16, bounds=8.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
        super().__init__()
        self.K = K
        self.bound = float(bounds)
        self.min_bin_width, self.min_bin_height, self.min_derivative = min_bin_width, min_bin_height, min_derivative
        self.unnorm_widths = nn.Parameter(torch.zeros(K))
        self.unnorm_heights = nn.Parameter(torch.zeros(K))
        self.unnorm_derivs = nn.Parameter(torch.full((K + 1,), softplus_inv(torch.tensor(1.0))))

    def _expand_params(self, batch_size):
        return (
            self.unnorm_widths.view(1, 1, -1).expand(batch_size, 1, self.K),
            self.unnorm_heights.view(1, 1, -1).expand(batch_size, 1, self.K),
            self.unnorm_derivs.view(1, 1, -1).expand(batch_size, 1, self.K + 1)
        )

    def inverse(self, e):
        e_in = e.unsqueeze(-1) if e.dim() == 1 else e
        w, h, d = self._expand_params(e_in.shape[0])
        z, _ = uRQS(
            inputs=e_in, unnormalized_widths=w, unnormalized_heights=h, unnormalized_derivatives=d,
            inverse=True, tails='linear', tail_bound=self.bound,
            min_bin_width=self.min_bin_width, min_bin_height=self.min_bin_height, min_derivative=self.min_derivative,
        )
        return z.squeeze(-1)


class OrderedFlowModel(nn.Module):
    def __init__(self, p, J, q=0, flow_bins=16, bounds=8.0, min_gap=1e-4):
        super().__init__()
        self.p, self.q, self.J, self.min_gap = p, q, J, min_gap
        self.beta = nn.Parameter(torch.zeros(p))
        self.alpha1_intercept = nn.Parameter(torch.tensor(0.0))
        self.alpha1_gamma = nn.Parameter(torch.zeros(q)) if q > 0 else None

        if self.J > 2:
            self.gap_intercepts_raw = nn.Parameter(torch.full((self.J - 2,), softplus_inv(torch.tensor(0.5))))
            self.gap_gammas = nn.Parameter(torch.zeros(self.J - 2, q)) if q > 0 else None
        else:
            self.register_parameter('gap_intercepts_raw', None)
            self.register_parameter('gap_gammas', None)

        self.a_raw = nn.Parameter(softplus_inv(torch.tensor(1.0)), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.flow = GlobalRQS1D(K=flow_bins, bounds=bounds)

    def alphas_obs(self, Z=None):
        n = 1 if Z is None else Z.shape[0]
        alpha1 = self.alpha1_intercept.expand(n) if self.q == 0 or Z is None else self.alpha1_intercept + Z.matmul(self.alpha1_gamma)
        if self.J == 2: return alpha1.unsqueeze(1)

        alphas = torch.empty(n, self.J - 1, dtype=alpha1.dtype, device=alpha1.device)
        alphas[:, 0] = alpha1
        
        gaps_raw = self.gap_intercepts_raw.unsqueeze(0) if self.q == 0 or Z is None else self.gap_intercepts_raw.unsqueeze(0) + Z.matmul(self.gap_gammas.T)
        gaps = torch.nn.functional.softplus(gaps_raw) + self.min_gap
        
        for k in range(1, self.J - 1):
            alphas[:, k] = alphas[:, k-1] + gaps[:, k-1] if gaps.dim() > 1 else alphas[:, k-1] + gaps[k-1]
        return alphas

    def _get_z_bounds(self, X, Z=None):
        eta = X.matmul(self.beta)
        alphas = self.alphas_obs(Z)
        B = torch.empty(X.shape[0], self.J + 1, dtype=X.dtype, device=X.device)
        B[:, 0], B[:, -1] = -torch.inf, torch.inf
        B[:, 1:-1] = alphas
        
        E = B - eta.unsqueeze(1)
        E_flat = E.reshape(-1)
        Z_flat = torch.empty_like(E_flat)
        Z_flat[E_flat == -torch.inf] = -torch.inf
        Z_flat[E_flat == torch.inf] = torch.inf
        
        finite_mask = torch.isfinite(E_flat)
        if finite_mask.any():
            Z_flat[finite_mask] = self.flow.inverse(E_flat[finite_mask])
            
        return Z_flat.reshape(X.shape[0], self.J + 1)

    def predict_proba(self, X, Z=None):
        z = self._get_z_bounds(X, Z)
        cdf_u = torch.full_like(z, torch.nan)
        cdf_u[torch.isneginf(z)], cdf_u[torch.isposinf(z)] = 0.0, 1.0
        finite_z = torch.isfinite(z)
        if finite_z.any():
            cdf_u[finite_z] = Normal(0.0, 1.0).cdf(z[finite_z])
        probs = cdf_u[:, 1:] - cdf_u[:, :-1]
        return torch.clamp(probs, min=1e-12, max=1.0)

    def neg_loglik(self, X, y, Z=None):
        z = self._get_z_bounds(X, Z)
        y_idx = (y.long() - 1).view(-1, 1)
        zl = z.gather(1, y_idx).squeeze(1)
        zu = z.gather(1, y_idx + 1).squeeze(1)
        return -torch.mean(log_diff_normal_cdfs(zl, zu))


class ModelFreeConditionalFlowModel(nn.Module):
    def __init__(self, p, J, flow_bins=16, bounds=8.0, min_gap=1e-4, hidden_features=32):
        super().__init__()
        self.p, self.J, self.min_gap = p, J, min_gap
        
        transform = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=1, hidden_features=hidden_features, context_features=p,
            num_bins=flow_bins, tail_bound=bounds, tails='linear'
        )
        self.flow = Flow(transform=transform, distribution=StandardNormal(shape=[1]))
        self.alpha1_intercept = nn.Parameter(torch.tensor(0.0))
        self.gap_intercepts_raw = nn.Parameter(torch.full((self.J - 2,), softplus_inv(torch.tensor(0.5)))) if J > 2 else None

    def _get_z_bounds(self, X):
        n = X.shape[0]
        alpha1 = self.alpha1_intercept.expand(n)
        B = torch.empty(n, self.J + 1, dtype=X.dtype, device=X.device)
        B[:, 0], B[:, -1] = -torch.inf, torch.inf
        B[:, 1] = alpha1
        
        if self.J > 2:
            gaps = torch.nn.functional.softplus(self.gap_intercepts_raw) + self.min_gap
            for k in range(2, self.J):
                B[:, k] = B[:, k-1] + gaps[k-2]

        B_flat = B.reshape(-1, 1)
        X_expanded = X.unsqueeze(1).expand(-1, self.J + 1, -1).reshape(-1, self.p)
        
        z_flat = torch.full_like(B_flat, torch.nan)
        z_flat[B_flat == -torch.inf] = -torch.inf
        z_flat[B_flat == torch.inf] = torch.inf
        
        finite_mask = torch.isfinite(B_flat).squeeze(1)
        if finite_mask.any():
            z_finite, _ = self.flow._transform.forward(B_flat[finite_mask], context=X_expanded[finite_mask])
            z_flat[finite_mask, :] = z_finite
            
        return z_flat.reshape(n, self.J + 1)

    def predict_proba(self, X):
        z = self._get_z_bounds(X)
        cdf_u = torch.full_like(z, torch.nan)
        cdf_u[torch.isneginf(z)], cdf_u[torch.isposinf(z)] = 0.0, 1.0
        finite_z = torch.isfinite(z)
        if finite_z.any():
            cdf_u[finite_z] = Normal(0.0, 1.0).cdf(z[finite_z])
        probs = cdf_u[:, 1:] - cdf_u[:, :-1]
        return torch.clamp(probs, min=1e-12, max=1.0)

    def neg_loglik(self, X, y):
        z = self._get_z_bounds(X)
        y_idx = (y.long() - 1).view(-1, 1)
        zl = z.gather(1, y_idx).squeeze(1)
        zu = z.gather(1, y_idx + 1).squeeze(1)
        return -torch.mean(log_diff_normal_cdfs(zl, zu))


# ----------------------------------------------------------------------
# Standardized Training Routine
# ----------------------------------------------------------------------

def train_flow_model(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, Z: Optional[torch.Tensor] = None,
    epochs: int = 300, lr: float = 5e-3, use_lbfgs: bool = True,
    patience: int = 15, delta_eps: float = 1e-4, verbose: bool = False
) -> Dict[str, Any]:
    
    # Warm start: Freeze flow parameters initially
    for name, param in model.named_parameters():
        if "flow" in name: param.requires_grad = False

    best_nll = float('inf')
    best_state = None
    neg_streak = 0
    history = {'train_nll': []}

    opt_adam = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    
    for ep in range(1, epochs + 1):
        if ep == 21:
            for name, param in model.named_parameters():
                if "flow" in name: param.requires_grad = True
            opt_adam = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        opt_adam.zero_grad()
        loss = model.neg_loglik(X, y, Z) if Z is not None else model.neg_loglik(X, y)
        if torch.isnan(loss): break
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_adam.step()
        
        cur_nll = float(loss.detach())
        history['train_nll'].append(cur_nll)

        if cur_nll < best_nll - delta_eps:
            best_nll = cur_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            neg_streak = 0
        else:
            neg_streak += 1

        if ep > 30 and neg_streak >= patience:
            if verbose: print(f"Adam early stopping at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if use_lbfgs:
        opt_lbfgs = optim.LBFGS(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.5, max_iter=20, history_size=20, line_search_fn="strong_wolfe"
        )
        def closure():
            opt_lbfgs.zero_grad()
            loss = model.neg_loglik(X, y, Z) if Z is not None else model.neg_loglik(X, y)
            if torch.isnan(loss): return loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            return loss

        for _ in range(15):
            try:
                loss = opt_lbfgs.step(closure)
                cur_nll = float(loss.detach())
                history['train_nll'].append(cur_nll)
                if cur_nll < best_nll - delta_eps:
                    best_nll = cur_nll
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if torch.isnan(loss): break
            except Exception:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        
    return history

# ----------------------------------------------------------------------
# Estimator API / Wrappers
# ----------------------------------------------------------------------

class BaseOrdinalFlowEstimator:
    def __init__(self, **kwargs):
        self.model = None
        self.history = None
        self.config = kwargs

    def _prepare_data(self, X, y=None, Z=None):
        X_t = torch.as_tensor(X, dtype=torch.float64, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float64, device=device) if y is not None else None
        Z_t = torch.as_tensor(Z, dtype=torch.float64, device=device) if Z is not None else None
        return X_t, y_t, Z_t

    def predict_proba(self, X, Z=None) -> np.ndarray:
        self.model.eval()
        X_t, _, Z_t = self._prepare_data(X, Z=Z)
        with torch.no_grad():
            probs = self.model.predict_proba(X_t, Z_t) if Z_t is not None else self.model.predict_proba(X_t)
        return probs.cpu().numpy()

    def counterfactual_probs(self, X, treatment_idx: int, Z=None):
        X_t, _, Z_t = self._prepare_data(X, Z=Z)
        X1, X0 = X_t.clone(), X_t.clone()
        X1[:, treatment_idx] = 1.0
        X0[:, treatment_idx] = 0.0
        
        self.model.eval()
        with torch.no_grad():
            p1 = self.model.predict_proba(X1, Z_t) if Z_t is not None else self.model.predict_proba(X1)
            p0 = self.model.predict_proba(X0, Z_t) if Z_t is not None else self.model.predict_proba(X0)
        return p1.cpu().numpy(), p0.cpu().numpy()

    def compute_effects(self, X, treatment_idx: int, Z=None) -> Dict[str, object]:
        p1, p0 = self.counterfactual_probs(X, treatment_idx, Z)
        return effects_from_counterfactual_probs(p1, p0)

    def bootstrap_effects(self, X, y, treatment_idx: int, Z=None, B: int = 100) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model must be fit on full data before bootstrapping.")
        
        X_t, y_t, Z_t = self._prepare_data(X, y, Z)
        n = len(X_t)
        results = []

        print(f"Starting {B} bootstrap iterations for effects...")
        for b in range(B):
            idx = torch.empty(n, dtype=torch.long, device=device)
            for c in torch.unique(y_t):
                c_mask = (y_t == c)
                c_indices = torch.where(c_mask)[0]
                idx[c_mask] = c_indices[torch.randint(0, len(c_indices), (len(c_indices),), device=device)]
                
            Xb, yb = X_t[idx], y_t[idx]
            Zb = Z_t[idx] if Z_t is not None else None
            
            estimator = self.__class__(**self.config)
            
            if isinstance(self, StructuredOrdinalFlow):
                estimator.model = OrderedFlowModel(p=X_t.shape[1], J=self.J, q=self.q, flow_bins=self.flow_bins, bounds=self.bounds, min_gap=self.min_gap).to(device)
            else:
                estimator.model = ModelFreeConditionalFlowModel(p=X_t.shape[1], J=self.J, flow_bins=self.flow_bins, bounds=self.bounds, hidden_features=self.hidden_features).to(device)
            
            estimator.model.load_state_dict(self.model.state_dict())
            
            # Train lightly (warm started)
            train_flow_model(estimator.model, Xb, yb, Z=Zb, epochs=30, lr=1e-3, use_lbfgs=True, verbose=False)
            
            eff = estimator.compute_effects(X_t, treatment_idx, Z=Z_t)
            
            flat_eff = {"wasserstein_unit": eff["wasserstein_unit"]}
            for i, val in enumerate(eff["category_effect"]): flat_eff[f"AME_cat_{i+1}"] = val
            for i, val in enumerate(eff["cum_ge_effect"]): flat_eff[f"CGE_cat_{i+2}"] = val # +2 because 1 is dropped
            results.append(flat_eff)
            
        return pd.DataFrame(results).std()


class StructuredOrdinalFlow(BaseOrdinalFlowEstimator):
    def __init__(self, J: int, q: int = 0, flow_bins: int = 16, bounds: float = 10.0, min_gap: float = 1e-4):
        super().__init__(J=J, q=q, flow_bins=flow_bins, bounds=bounds, min_gap=min_gap)
        self.J, self.q, self.flow_bins, self.bounds, self.min_gap = J, q, flow_bins, bounds, min_gap

    def fit(self, X, y, Z=None, epochs=300, lr=5e-3, use_lbfgs=True, verbose=False):
        X_t, y_t, Z_t = self._prepare_data(X, y, Z)
        self.model = OrderedFlowModel(p=X_t.shape[1], J=self.J, q=self.q, flow_bins=self.flow_bins, bounds=self.bounds, min_gap=self.min_gap).to(device)
        
        try:
            dfX = pd.DataFrame(X_t.cpu().numpy(), columns=[f"x{k}" for k in range(X_t.shape[1])])
            y0 = y_t.cpu().numpy().astype(int) - 1
            mod = OrderedModel(y0, dfX, distr='probit').fit(method='bfgs', disp=False)
            self.model.beta.data = torch.tensor(mod.params.values[:X_t.shape[1]], dtype=torch.float64, device=device)
            thr_vals = mod.params.values[-(self.J - 1):]
            self.model.alpha1_intercept.data = torch.tensor(thr_vals[0], dtype=torch.float64, device=device)
            if self.J > 2:
                gaps = np.maximum(np.diff(thr_vals), self.min_gap * 10)
                self.model.gap_intercepts_raw.data = softplus_inv(torch.tensor(gaps - self.min_gap, dtype=torch.float64, device=device))
        except Exception as e:
            if verbose: print(f"Probit init skipped: {e}")

        self.history = train_flow_model(self.model, X_t, y_t, Z=Z_t, epochs=epochs, lr=lr, use_lbfgs=use_lbfgs, verbose=verbose)
        return self

class ModelFreeOrdinalFlow(BaseOrdinalFlowEstimator):
    def __init__(self, J: int, flow_bins: int = 16, bounds: float = 10.0, hidden_features: int = 32):
        super().__init__(J=J, flow_bins=flow_bins, bounds=bounds, hidden_features=hidden_features)
        self.J, self.flow_bins, self.bounds, self.hidden_features = J, flow_bins, bounds, hidden_features

    def fit(self, X, y, epochs=500, lr=1e-2, use_lbfgs=True, verbose=False):
        X_t, y_t, _ = self._prepare_data(X, y)
        self.model = ModelFreeConditionalFlowModel(p=X_t.shape[1], J=self.J, flow_bins=self.flow_bins, bounds=self.bounds, hidden_features=self.hidden_features).to(device)
        self.history = train_flow_model(self.model, X_t, y_t, Z=None, epochs=epochs, lr=lr, use_lbfgs=use_lbfgs, verbose=verbose)
        return self

# ----------------------------------------------------------------------
# Standard Statsmodels Baseline Fits
# ----------------------------------------------------------------------

def fit_ordered_sm_baseline(X, y, treatment_idx: int = 0, link: str = "probit"):
    """
    Fits a standard statsmodels OrderedModel and returns probability estimands.
    """
    y0 = np.asarray(y, dtype=int) - 1
    X_np = np.asarray(X, dtype=float)
    cols = [f"x{k}" for k in range(X_np.shape[1])]
    dfX = pd.DataFrame(X_np, columns=cols)
    
    # Fit statsmodels OrderedModel
    mod = OrderedModel(y0, dfX, distr=link)
    res = mod.fit(method="bfgs", disp=False)

    def predict_probs(X_in):
        probs = np.asarray(res.model.predict(res.params, exog=pd.DataFrame(X_in, columns=cols)))
        return normalize_probs(probs)

    # Compute counterfactuals
    X1, X0 = X_np.copy(), X_np.copy()
    X1[:, treatment_idx] = 1.0
    X0[:, treatment_idx] = 0.0
    
    p1, p0 = predict_probs(X1), predict_probs(X0)
    effects = effects_from_counterfactual_probs(p1, p0)
    
    return effects, res
