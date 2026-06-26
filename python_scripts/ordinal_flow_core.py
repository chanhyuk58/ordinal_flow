# ========================= ordinal_flow_core.py =========================
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from scipy import stats
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline as uRQS,
)
from statsmodels.miscmodels.ordinal_model import OrderedModel

# ----------------------------------------------------------------------
# Numeric Helpers
# ----------------------------------------------------------------------
# {{{
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SQRT2 = math.sqrt(2.0)
LOG_HALF = math.log(0.5)

def softplus_inv(y):
    eps = 1e-12
    y = torch.clamp(y, min=eps)
    return torch.log(torch.expm1(y))

def log_ndtr(z):
    return LOG_HALF + torch.log(torch.special.erfc(-z / SQRT2))

def log_surv_ndtr(z):
    return LOG_HALF + torch.log(torch.special.erfc(z / SQRT2))

def log_diff_normal_cdfs(zl, zu):
    assert zl.shape == zu.shape
    left_mask = (zu <= 0) & (zl <= 0)
    right_mask = (zu >= 0) & (zl >= 0)
    mid_mask = ~(left_mask | right_mask)
    out = torch.empty_like(zu)

    if left_mask.any():
        a = log_ndtr(zu[left_mask])
        b = log_ndtr(zl[left_mask])
        out[left_mask] = a + torch.log1p(-torch.exp(b - a))

    if right_mask.any():
        a = log_surv_ndtr(zl[right_mask])
        b = log_surv_ndtr(zu[right_mask])
        out[right_mask] = a + torch.log1p(-torch.exp(b - a))

    if mid_mask.any():
        phi_u = Normal(0.0, 1.0).cdf(zu[mid_mask])
        phi_l = Normal(0.0, 1.0).cdf(zl[mid_mask])
        diff = torch.clamp(phi_u - phi_l, min=1e-300)
        out[mid_mask] = torch.log(diff)
    return out
# }}}

# ============ Global 1D RQS transform ============
class GlobalRQS1D(nn.Module): # {{{
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
        if z.dim() == 1:
            z_in = z.unsqueeze(-1)
        else:
            z_in = z
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
        if e.dim() == 1:
            e_in = e.unsqueeze(-1)
        else:
            e_in = e
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
# }}}

# ============ Ordered Flow Model with generalized thresholds ============
class OrderedFlowModel(nn.Module): # {{{
    """
    Generalized ordinal regression with flow-based latent error.
    y* = X beta + epsilon
    Thresholds: alpha_1(i) = c1 + Z_i gamma1
                gaps_k(i) = softplus(cgap_k + Z_i gammagap_k) + min_gap
                alpha_j(i) = alpha_{j-1}(i) + gaps_{j-1}(i)
    J categories => J-1 thresholds, k = 1..J-2 gaps
    """
    def __init__(self, p, J, q=0, flow_bins=16, bounds=8.0, min_gap=1e-4):
        super().__init__()
        assert J >= 2
        self.p = p
        self.q = q
        self.J = int(J)
        self.min_gap = min_gap

        # Linear predictor coefficients
        self.beta = nn.Parameter(torch.zeros(p))

        # Threshold coefficients: intercepts + slopes
        # alpha1: intercept + gamma1
        self.alpha1_intercept = nn.Parameter(torch.tensor(0.0))
        if q > 0:
            self.alpha1_gamma = nn.Parameter(torch.zeros(q))
        else:
            self.register_parameter('alpha1_gamma', None)

        # Gaps: intercepts + gammas, length J-2
        if self.J > 2:
            # initialize to softplus_inv(0.5) for intercepts of gaps
            self.gap_intercepts_raw = nn.Parameter(torch.full((self.J - 2,), softplus_inv(torch.tensor(0.5))))
            if q > 0:
                self.gap_gammas = nn.Parameter(torch.zeros(self.J - 2, q))
            else:
                self.register_parameter('gap_gammas', None)
        else:
            self.register_parameter('gap_intercepts_raw', None)
            self.register_parameter('gap_gammas', None)

        # Flow parameters
        self.flow = GlobalRQS1D(K=flow_bins, bounds=bounds)
        self.a_raw = nn.Parameter(softplus_inv(torch.tensor(1.0)))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.base = Normal(0.0, 1.0)

    def a(self):
        return torch.nn.functional.softplus(self.a_raw)

    def alphas_obs(self, Z=None):
        """
        Compute per-observation thresholds matrix (n, J-1).
        If q==0 or Z is None, thresholds depend only on intercepts.
        """
        n = 1
        if Z is not None:
            n = Z.shape[0]
        if self.q == 0 or Z is None:
            # broadcast intercept-only thresholds
            alpha1 = self.alpha1_intercept.expand(n)
            if self.J == 2:
                return alpha1.unsqueeze(1)
            gaps = torch.nn.functional.softplus(self.gap_intercepts_raw) + self.min_gap  # (J-2,)
            # cumulative thresholds per obs
            alphas = torch.empty(n, self.J - 1, dtype=alpha1.dtype, device=alpha1.device)
            alphas[:, 0] = alpha1
            for k in range(1, self.J - 1):
                alphas[:, k] = alphas[:, k-1] + gaps[k-1]
            return alphas
        else:
            # with covariate-dependent thresholds
            alpha1 = self.alpha1_intercept + Z.matmul(self.alpha1_gamma)  # (n,)
            alphas = torch.empty(Z.shape[0], self.J - 1, dtype=Z.dtype, device=Z.device)
            alphas[:, 0] = alpha1
            if self.J > 2:
                gaps = torch.nn.functional.softplus(
                    self.gap_intercepts_raw.unsqueeze(0) + Z.matmul(self.gap_gammas.T)
                ) + self.min_gap  # (n, J-2)
                for k in range(1, self.J - 1):
                    alphas[:, k] = alphas[:, k-1] + gaps[:, k-1]
            return alphas  # (n, J-1)

    def z_bounds(self, eta, alpha_lower, alpha_upper):
        a = self.a()
        e_lower = alpha_lower - eta
        e_upper = alpha_upper - eta
        zl = torch.empty_like(e_lower)
        zu = torch.empty_like(e_upper)
        lower_is_finite = torch.isfinite(e_lower)
        upper_is_finite = torch.isfinite(e_upper)
        zl[~lower_is_finite] = -torch.inf
        zu[~upper_is_finite] = torch.inf
        if lower_is_finite.any():
            zl[lower_is_finite] = self.flow.inverse((e_lower[lower_is_finite] - self.b) / a)
        if upper_is_finite.any():
            zu[upper_is_finite] = self.flow.inverse((e_upper[upper_is_finite] - self.b) / a)
        return zl, zu

    def neg_loglik(self, X, y, Z=None):
        """
        X: (n,p), y in {1,...,J}, Z: (n,q) or None
        """
        eta = X.matmul(self.beta)  # (n,)
        alphas_mat = self.alphas_obs(Z)  # (n, J-1)
        n = X.shape[0]
        # alpha_full: (n, J+1): [-inf, alphas..., +inf]
        alpha_full = torch.empty(n, self.J + 1, dtype=X.dtype, device=X.device)
        alpha_full[:, 0] = -torch.inf
        alpha_full[:, -1] = torch.inf
        alpha_full[:, 1:-1] = alphas_mat
        y_idx = y.long().view(-1, 1)  # (n,1)
        alpha_lower = alpha_full.gather(1, (y_idx - 1)).squeeze(1)
        alpha_upper = alpha_full.gather(1, y_idx).squeeze(1)
        zl, zu = self.z_bounds(eta, alpha_lower, alpha_upper)
        logp = log_diff_normal_cdfs(zl, zu)
        nll = -torch.sum(logp)
        return nll

    def predict_proba(self, X, Z=None):
        eta = X.matmul(self.beta)
        alphas_mat = self.alphas_obs(Z)  # (n, J-1)
        n = X.shape[0]
        a = self.a()
        probs = []
        for j in range(1, self.J + 1):
            if j == 1:
                lower = -torch.inf * torch.ones(n, dtype=X.dtype, device=X.device)
                upper = alphas_mat[:, 0]
            elif j == self.J:
                lower = alphas_mat[:, -1]
                upper = torch.inf * torch.ones(n, dtype=X.dtype, device=X.device)
            else:
                lower = alphas_mat[:, j - 2]
                upper = alphas_mat[:, j - 1]
            zu = torch.empty_like(upper)
            zl = torch.empty_like(lower)
            upper_is_finite = torch.isfinite(upper)
            lower_is_finite = torch.isfinite(lower)
            zu[~upper_is_finite] = torch.inf
            zl[~lower_is_finite] = -torch.inf
            if upper_is_finite.any():
                zu[upper_is_finite] = self.flow.inverse((upper[upper_is_finite] - eta[upper_is_finite] - self.b) / a)
            if lower_is_finite.any():
                zl[lower_is_finite] = self.flow.inverse((lower[lower_is_finite] - eta[lower_is_finite] - self.b) / a)
            pu = Normal(0.0, 1.0).cdf(zu)
            pl = Normal(0.0, 1.0).cdf(zl)
            p = torch.clamp(pu - pl, min=1e-12, max=1.0)
            probs.append(p)
        return torch.stack(probs, dim=1)  # (n, J)

    # Normalize var(epsilon) = 1
    @torch.no_grad()
    def project_identification(self, mc_samples=32768, Z_for_scale=None):
        """
        Re-center and re-scale error distribution:
        eps = a * flow(z) + b
        Compute m = E[eps], s = SD[eps], set:
          a_new = a / s, b_new = (b - m)/s
          beta <- beta / s
          Threshold params: subtract m from intercepts; divide both intercepts and gammas by s.
        """
        # 1) standardize eps
        z = self.flow.sample_base(mc_samples).to(self.a_raw.device)
        eps_base, _ = self.flow.forward_with_logabsdet(z)
        eps = self.a() * eps_base + self.b
        m = eps.mean()
        s = eps.std(unbiased=False)
        s = torch.clamp(s, min=1e-4)
        # update flow scalars
        a_new = self.a() / s
        a_new = torch.clamp(a_new - 1e-4, min=1e-9)
        b_new = (self.b - m) / s
        self.a_raw.data = softplus_inv(a_new).detach()
        self.b.data = b_new.detach()
        # 2) update beta
        self.beta.data = (self.beta / s).detach()
        # 3) update thresholds
        # subtract m from intercepts only
        self.alpha1_intercept.data = ((self.alpha1_intercept - m) / s).detach()
        if self.J > 2:
            # current gaps are on raw scale; we must update intercepts accordingly:
            # We need the implicit actual gap intercepts, but raw stores pre-softplus.
            # We can compute effective intercepts by inverting: softplus(c_raw) + min_gap.
            # For shift m: subtract m only from alpha1_intercept, not from gaps.
            # For scale s: both intercepts and gammas divided by s; we can update raw by recomputing desired gaps/s and mapping back.
            gaps = torch.nn.functional.softplus(self.gap_intercepts_raw) + self.min_gap
            gaps_new = gaps / s
            # ensure min gap
            gaps_new = torch.clamp(gaps_new - self.min_gap, min=1e-8) + self.min_gap
            self.gap_intercepts_raw.data = softplus_inv(gaps_new - self.min_gap).detach()
        # scale gammas
        if self.q > 0:
            if self.alpha1_gamma is not None:
                self.alpha1_gamma.data = (self.alpha1_gamma / s).detach()
            if self.gap_gammas is not None:
                self.gap_gammas.data = (self.gap_gammas / s).detach()

    @torch.no_grad()
    def init_from_ordered_probit(self, X, y, Z=None, verbose=True):
        """
        Initialize beta from ordered probit; thresholds:
        - if Z is None or q=0: constant thresholds set to ordered probit thresholds
        - else: set intercepts to probit thresholds; gammas to zero

        Also stores baseline probit NLL on the model as `self._probit_nll`.
        """
        y0 = y.cpu().numpy().astype(int) - 1
        X_np = X.cpu().numpy()
        cols = [f"x{k}" for k in range(X_np.shape[1])]
        dfX = pd.DataFrame(X_np, columns=cols)
        mod = OrderedModel(y0, dfX, distr='probit')
        res = mod.fit(method='bfgs', disp=False)

        beta_hat = res.params[cols].values
        thr_vals = res.params.values[-(self.J - 1):]

        # set parameters
        self.beta.data = torch.tensor(
            beta_hat,
            dtype=torch.get_default_dtype(),
            device=self.beta.device
        )
        self.alpha1_intercept.data = torch.tensor(
            thr_vals[0],
            dtype=torch.get_default_dtype(),
            device=self.beta.device
        )
        if self.q > 0:
            self.alpha1_gamma.data = torch.zeros_like(self.alpha1_gamma)
        if self.J > 2:
            gaps = np.diff(thr_vals)
            gaps = np.maximum(gaps, self.min_gap * 10)
            self.gap_intercepts_raw.data = softplus_inv(
                torch.tensor(
                    gaps - self.min_gap,
                    dtype=torch.get_default_dtype(),
                    device=self.beta.device
                )
            )
            if self.q > 0 and self.gap_gammas is not None:
                self.gap_gammas.data = torch.zeros_like(self.gap_gammas)

        # Initialize flow scalars
        self.a_raw.data = softplus_inv(torch.tensor(1.0))
        self.b.data = torch.tensor(0.0)

        # Store baseline probit NLL for comparison
        self._probit_nll = float(-res.llf)

        if verbose:
            print(
                f"init probit: beta {np.round(beta_hat,4)} | thr {np.round(thr_vals,4)} "
                f"| NLL_probit {self._probit_nll:.4f}"
            )


    @torch.no_grad()
    def learned_cdf_on_grid(self, egrid):
        a = self.a()
        z = self.flow.inverse((egrid - self.b) / a)
        return Normal(0.0, 1.0).cdf(z)

    @torch.no_grad()
    def learned_pdf_on_grid(self, egrid):
        a = self.a()
        z, ladj_inv = self.flow.inverse_with_logabsdet((egrid - self.b) / a)
        log_phi = -0.5 * z.pow(2) - 0.5 * math.log(2 * math.pi)
        log_dz_de = ladj_inv - torch.log(a)
        return torch.exp(log_phi + log_dz_de)
# }}}

# ======== True error distributions (from your original code) ========
def get_true_error(spec: str): # {{{
    """
    Returns:
    sample_eps(n): function sampling epsilon of size n on current device
    true_cdf_std(e): CDF of standardized error (mean 0, sd 1 when defined)
    true_pdf_std(e): PDF of standardized error
    label: human-readable label
    m: mean of epsilon (if undefined, 0.0 by convention)
    s: standard deviation of epsilon (if undefined, 1.0 by convention)
    """
    spec = spec.lower()
    if spec == "normal":
        StdN = torch.distributions.Normal(0.0, 1.0)
        def sample_eps(n): return StdN.sample((n,)).to(device)
        def true_cdf_std(e): return StdN.cdf(e)
        def true_pdf_std(e): return torch.exp(StdN.log_prob(e))
        label = "Standard Normal N(0,1)"
        m, s = 0.0, 1.0

    elif spec == "logistic":
        # Logistic(0,1): mean 0, var = pi^2/3, sd = pi/sqrt(3)
        m = 0.0
        s = math.pi / math.sqrt(3.0)
        def sample_eps(n):
            U = torch.rand(n, device=device)
            return torch.log(U) - torch.log(1.0 - U)
        def F_base(x): return 1.0 / (1.0 + torch.exp(-x))
        def f_base(x):
            sig = F_base(x)
            return sig * (1.0 - sig)
        def true_cdf_std(e): return F_base(m + s * e)
        def true_pdf_std(e): return s * f_base(m + s * e)
        label = "Logistic(0,1) standardized"
    
    elif spec == "t":
        nu = 5.0
        m = 0.0
        s = math.sqrt(nu / (nu - 2.0))
        StudentT = torch.distributions.StudentT(df=nu)
        def sample_eps(n): return StudentT.sample((n,)).to(device)
        def F_base(x): return StudentT.cdf(x)
        def f_base(x): return torch.exp(StudentT.log_prob(x))
        def true_cdf_std(e): return F_base(m + s * e)
        def true_pdf_std(e): return s * f_base(m + s * e)
        label = f"Student-t(ν={nu}) standardized"
    
    elif spec == "mixture_gaussian":
        weights = torch.tensor([0.6, 0.4], device=device)
        mus = torch.tensor([-1.0, 3.0], device=device)
        sigmas = torch.tensor([1.5, 0.5], device=device)
        m = (weights * mus).sum().item()
        v = (weights * (sigmas**2 + mus**2)).sum().item() - m**2
        s = math.sqrt(v)
        def sample_eps(n):
            cat = torch.distributions.Categorical(weights)
            comps = torch.distributions.Normal(mus, sigmas)
            idx = cat.sample((n,))
            eps = comps.sample((n,))
            return eps[torch.arange(n, device=device), idx]
        def F_base(x):
            X = x.unsqueeze(-1)
            cdfs = Normal(mus, sigmas).cdf(X)
            return (weights * cdfs).sum(dim=-1)
        def f_base(x):
            X = x.unsqueeze(-1)
            pdfs = torch.exp(Normal(mus, sigmas).log_prob(X))
            return (weights * pdfs).sum(dim=-1)
        def true_cdf_std(e): return F_base(m + s * e)
        def true_pdf_std(e): return s * f_base(m + s * e)
        label = "Mixture Gaussian standardized"
    
    elif spec == "lognormal":
        mu, sigma = 0.0, 1.0
        m = math.exp(mu + 0.5 * sigma**2)
        v = (math.exp(sigma**2) - 1.0) * math.exp(2 * mu + sigma**2)
        s = math.sqrt(v)
        def sample_eps(n):
            z = torch.randn(n, device=device)
            return torch.exp(mu + sigma * z)
        def F_base(x):
            out = torch.zeros_like(x)
            mask = x > 0
            if mask.any():
                out[mask] = Normal(0.0, 1.0).cdf((torch.log(x[mask]) - mu) / sigma)
            return out
        def f_base(x):
            out = torch.zeros_like(x)
            mask = x > 0
            if mask.any():
                xm = x[mask]
                out[mask] = torch.exp(-0.5 * ((torch.log(xm) - mu) / sigma) ** 2) / (xm * sigma * math.sqrt(2 * math.pi))
            return out
        def true_cdf_std(e): return F_base(m + s * e)
        def true_pdf_std(e): return s * f_base(m + s * e)
        label = "Log-normal standardized"
    
    elif spec == "chisq":
        k = 5.0
        m = k
        s = math.sqrt(2.0 * k)
        Chi2 = torch.distributions.Chi2(df=k)
        def sample_eps(n): return Chi2.sample((n,)).to(device)
        def F_base(x):
            out = torch.zeros_like(x)
            mask = x >= 0
            if mask.any():
                out[mask] = Chi2.cdf(x[mask])
            return out
        def f_base(x):
            out = torch.zeros_like(x)
            mask = x >= 0
            if mask.any():
                out[mask] = torch.exp(Chi2.log_prob(x[mask]))
            return out
        def true_cdf_std(e): return F_base(m + s * e)
        def true_pdf_std(e): return s * f_base(m + s * e)
        label = f"Chi-square(k={int(k)}) standardized"
    
    elif spec == "uniform":
        a_u, b_u = 0.0, 1.0
        m = 0.5 * (a_u + b_u)
        s = (b_u - a_u) / math.sqrt(12.0)
        def sample_eps(n): return torch.distributions.Uniform(low=a_u, high=b_u).sample((n,)).to(device)
        def F_base(x):
            out = torch.zeros_like(x)
            out = torch.where(x <= a_u, torch.zeros_like(x), out)
            out = torch.where(x >= b_u, torch.ones_like(x), out)
            mask = (x > a_u) & (x < b_u)
            out = torch.where(mask, (x - a_u) / (b_u - a_u), out)
            return out
        def f_base(x):
            out = torch.zeros_like(x)
            mask = (x >= a_u) & (x <= b_u)
            out = torch.where(mask, torch.full_like(x, 1.0 / (b_u - a_u)), out)
            return out
        def true_cdf_std(e): return F_base(m + s * e)
        def true_pdf_std(e): return s * f_base(m + s * e)
        label = f"Uniform[{a_u},{b_u}] standardized"
    
    elif spec == "cauchy":
        # No variance; return s=1 and note that normalization is not meaningful
        loc = 0.0
        scale = 1.0
        Cauchy = torch.distributions.Cauchy(loc=loc, scale=scale)
        def sample_eps(n): return Cauchy.sample((n,)).to(device)
        def F_base(x): return 0.5 + (1.0 / math.pi) * torch.atan((x - loc) / scale)
        def f_base(x): return 1.0 / (math.pi * scale * (1.0 + ((x - loc) / scale) ** 2))
        def true_cdf_std(e): return F_base(e)
        def true_pdf_std(e): return f_base(e)
        label = "Cauchy(0,1) (no unit-variance standardization)"
        m, s = 0.0, 1.0
    
    else:
        raise ValueError("Unknown error spec")
    
    return sample_eps, true_cdf_std, true_pdf_std, label, m, s
# }}}

# ======== statsmodels wrappers (parallel thresholds only) ========
def fit_ordered_sm(y, X, link='probit', normalize=True, norm_criterion='variance'): # {{{
    """
    Fit statsmodels OrderedModel with probit or logit link.
    Returns
    -------
    beta_hat : np.ndarray, shape (p,)
        Coefficients (normalized to unit-error-variance scale if normalize=True).
    thr : np.ndarray, shape (J-1,)
        Thresholds (same normalization as beta_hat).
    res : statsmodels result object
        Full fitted result from statsmodels.
    
    Notes
    -----
    - For 'probit': coefficients are already on unit-variance scale; no change.
    - For 'logit': coefficients are scaled to unit-variance by dividing by s_log.
      If norm_criterion='variance', s_log = pi/sqrt(3) (exact variance matching).
      If norm_criterion='slope', s_log ≈ 1.6 (matches slopes at 0; common rule of thumb).
    """
    y0 = y.detach().cpu().numpy().astype(int) - 1
    X_np = X.detach().cpu().numpy()
    J = np.unique(y0).size
    
    cols = [f"x{k}" for k in range(X_np.shape[1])]
    dfX = pd.DataFrame(X_np, columns=cols)
    
    distr = 'probit' if link == 'probit' else 'logit'
    mod = OrderedModel(y0, dfX, distr=distr)
    res = mod.fit(method='bfgs', disp=False)
    
    params = res.params  # pandas Series in recent statsmodels
    if hasattr(params, 'index'):
        beta_hat = params[cols].values
        thr = params.values[-(J - 1):]
    else:
        p = X_np.shape[1]
        beta_hat = params[:p]
        thr = params[-(J - 1):]
    
    if normalize:
        if link == 'logit':
            if norm_criterion == 'variance':
                s_link = math.pi / math.sqrt(3.0)  # ~1.8138
            elif norm_criterion == 'slope':
                s_link = 1.6  # matches slope at 0; rule-of-thumb
            else:
                raise ValueError("norm_criterion must be 'variance' or 'slope'")
            beta_hat = beta_hat / s_link
            thr = thr / s_link
        # probit: s_link = 1.0, no change
    
    return beta_hat, thr, res

# }}}

# ======== Training routine for flow model ========
# # Adam {{{
# def train_ordered_flow(
#     X, y, Z=None,
#     flow_bins=16, bounds=10.0,
#     epochs=1000,
#     lr=1e-3,              # base learning rate for all parameters
#     init_probit=True,
#     verbose=False,
#     monitor_nll=True,
#     check_every=10,
#     burn_in_checks=5,
#     neg_patience=10,
#     delta_eps=1e-5,       # treat delta < -delta_eps as effectively negative
#     use_lr_scheduler=False,
#     lr_step_size=10,      # decay LR every this many epochs (if scheduler is used)
#     lr_gamma=0.5          # multiply LR by this factor when stepping the scheduler
# ):
#     """
#     Train OrderedFlowModel with:
#     - Probit initialization: ordered probit is used to initialize parameters
#       and its NLL is treated as the first "best" model.
#     - Best model tracking: at each monitoring checkpoint, if the current flow
#       model achieves a lower NLL than the best-so-far, we update the best
#       (baseline) NLL and store the current state_dict as the new best model.
#     - Early stopping based on relative NLL changes Δ.
#     - Final returned model = best-so-far flow state (starting from probit),
#       followed by identification normalization via project_identification().
#     """
#     X = X.to(device)
#     y = y.to(device)
#     J = int(torch.max(y))
#     if Z is not None:
#         Z = Z.to(device)
#         q = Z.shape[1]
#     else:
#         q = 0
#
#     model = OrderedFlowModel(
#         p=X.shape[1], J=J, q=q,
#         flow_bins=flow_bins, bounds=bounds
#     ).to(device)
#
#     # 1. Initialize from ordered probit
#     if init_probit:
#         try:
#             model.init_from_ordered_probit(
#                 X.cpu(), y.cpu(), Z.cpu() if Z is not None else None, verbose
#             )
#         except Exception as e:
#             print("init_from_ordered_probit failed; using zeros", e)
#             model._probit_nll = None
#     else:
#         model._probit_nll = None
#
#     # 2. Set initial "best" model as probit-initialized state if available
#     if getattr(model, "_probit_nll", None) is not None:
#         baseline_nll = float(model._probit_nll)  # probit NLL as initial best
#         baseline_state = {k: v.clone() for k, v in model.state_dict().items()}
#         if verbose:
#             print(f"Initial best model: ordered probit (NLL_probit={baseline_nll:.4f})")
#     else:
#         # If probit failed, baseline will be set at first monitoring checkpoint
#         baseline_nll = None
#         baseline_state = {k: v.clone() for k, v in model.state_dict().items()}
#         if verbose:
#             print("No probit NLL available; best model will be set from first check.")
#
#     # 3. Optimizer and optional LR scheduler
#     opt = optim.Adam(model.parameters(), lr=lr)
#
#     scheduler = None
#     if use_lr_scheduler:
#         scheduler = torch.optim.lr_scheduler.StepLR(
#             opt, step_size=lr_step_size, gamma=lr_gamma
#         )
#
#     # Monitoring and early stopping state
#     nll_history = []
#     delta_history = []
#     neg_streak = 0
#     early_stopped = False
#     n_checks = 0
#
#     for ep in range(1, epochs + 1):
#         opt.zero_grad()
#         nll = model.neg_loglik(X, y, Z)
#         nll.backward()
#         opt.step()
#
#         # Step the LR scheduler at the end of the epoch
#         if scheduler is not None:
#             scheduler.step()
#
#         if verbose and (ep % 50 == 0 or ep == 1):
#             current_lr = opt.param_groups[0]["lr"]
#             print(f"Epoch {ep:4d} | NLL {nll.item():.4f} | lr={current_lr:.2e}")
#
#         # 4. Monitoring + early stopping based on Δ
#         if monitor_nll and (ep % check_every == 0 or ep == 1):
#             with torch.no_grad():
#                 cur_nll = model.neg_loglik(X, y, Z).item()
#             nll_history.append((ep, cur_nll))
#             n_checks += 1
#
#             # If baseline_nll not set (e.g., probit failed), set from first check
#             if baseline_nll is None:
#                 baseline_nll = cur_nll
#                 baseline_state = {k: v.clone() for k, v in model.state_dict().items()}
#                 delta = 0.0
#             else:
#                 denom = max(1.0, abs(baseline_nll))
#                 delta = (baseline_nll - cur_nll) / denom
#
#             delta_history.append((ep, delta))
#
#             if verbose:
#                 print(
#                     f"  [Monitor] Epoch {ep}: NLL={cur_nll:.4f}, "
#                     f"Delta_over_best={delta:.3e} "
#                     f"(best_nll={baseline_nll:.4f})"
#                 )
#
#             # If Δ > 0, current model is better than best-so-far
#             if delta > 0:
#                 baseline_nll = cur_nll
#                 baseline_state = {k: v.clone() for k, v in model.state_dict().items()}
#                 neg_streak = 0
#                 if verbose:
#                     print("    [Best] New best NLL found; updating baseline model.")
#
#             # Early stopping based on consecutive negative Δ after burn-in
#             if n_checks > burn_in_checks:
#                 if delta < -delta_eps:
#                     neg_streak += 1
#                 else:
#                     neg_streak = 0
#
#                 if verbose and neg_streak > 0:
#                     print(f"    [Δ-check] neg_streak={neg_streak}, "
#                           f"delta={delta:.3e}")
#
#                 if neg_streak >= neg_patience:
#                     if verbose:
#                         print(
#                             f"Early stopping at epoch {ep}: "
#                             f"{neg_patience} consecutive Δ < -{delta_eps}."
#                         )
#                     early_stopped = True
#                     break
#
#     # Attach histories for diagnostics
#     if monitor_nll:
#         model.nll_history = nll_history
#         model.delta_history = delta_history
#     model.nll_probit = getattr(model, "_probit_nll", None)
#
#     # 5. Load the best-so-far state (starting from probit, then updated)
#     if baseline_state is not None and baseline_nll is not None:
#         if verbose:
#             print(f"Loading best model state (best_nll={baseline_nll:.4f}).")
#         model.load_state_dict(baseline_state)
#     else:
#         if verbose:
#             print("No monitoring-based best state recorded; using final epoch state.")
#
#     # 6. Final identification normalization
#     model.project_identification(mc_samples=32768)
#     return model
# # }}}

# BFGS {{{
def train_ordered_flow(
    X, y, Z=None,
    flow_bins=16, bounds=10.0,
    epochs=200,           
    lr=1e-3,              
    use_lbfgs=True,       
    lbfgs_steps=50,      
    init_probit=True,
    verbose=False,
    monitor_nll=False,
    **kwargs # Captures old args like delta_eps, neg_patience so they don't break calls
):
    X = X.to(device)
    y = y.to(device)
    J = int(torch.max(y))
    q = Z.shape[1] if Z is not None else 0
    if Z is not None: Z = Z.to(device)

    model = OrderedFlowModel(
        p=X.shape[1], J=J, q=q,
        flow_bins=flow_bins, bounds=bounds
    ).to(device)

    # 1. Probit Init
    baseline_nll = None
    baseline_state = None
    if init_probit:
        try:
            model.init_from_ordered_probit(X.cpu(), y.cpu(), Z.cpu() if Z is not None else None, verbose)
            baseline_nll = float(model._probit_nll)
            baseline_state = {k: v.clone() for k, v in model.state_dict().items()}
        except Exception:
            pass
    
    # 2. Adam Warm-up
    adam_opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        adam_opt.zero_grad()
        loss = model.neg_loglik(X, y, Z)
        if torch.isnan(loss): break
        loss.backward()
        adam_opt.step()

    # 3. L-BFGS Polish
    if use_lbfgs:
        lbfgs_opt = torch.optim.LBFGS(
            model.parameters(), 
            lr=1.0, 
            max_iter=20, 
            history_size=20, 
            line_search_fn='strong_wolfe'
        )
        def closure():
            lbfgs_opt.zero_grad()
            loss = model.neg_loglik(X, y, Z)
            if torch.isnan(loss):
                return loss
            loss.backward()
            # CLIP GRADIENTS: Prevent parameters from jumping to 10^16
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                # If the line search or matrix inversion fails, stop L-BFGS and move on
                break

    # 4. Identification
    if baseline_state is not None:
        model.load_state_dict(baseline_state)
    
    model.project_identification(mc_samples=32768)
    return model
# }}}

# ======== Bootstrap standard errors for beta coefficients ========
# {{{
def bootstrap_beta_se_flow(
    X, y, Z=None,
    B=200,                 # number of bootstrap replications
    flow_bins=16, bounds=10.0,
    epochs=500,
    lr=1e-3,
    init_probit=True,
    verbose=False,
    train_kwargs=None,
    return_bootstrap_betas=False
):
    """
    Compute bootstrap standard errors for the beta coefficients of the
    OrderedFlowModel using nonparametric case bootstrap.

    Steps:
      1) Fit a flow model on the full sample using train_ordered_flow.
      2) For b = 1..B:
           - Resample n observations with replacement from (X, y, Z).
           - Fit a flow model on the bootstrap sample (same hyperparameters).
           - Store its beta coefficients.
      3) Compute the sample standard deviation across bootstrap betas.

    Parameters
    ----------
    X, y, Z : torch.Tensors
        Data (X: n x p, y: n, Z: n x q or None).
    B : int
        Number of bootstrap replications.
    flow_bins, bounds : int, float
        Flow architecture hyperparameters (passed to train_ordered_flow).
    epochs : int
        Number of training epochs in each bootstrap fit.
        (Can be smaller than for the main fit to save time.)
    lr : float
        Learning rate for Adam in each bootstrap fit.
    init_probit : bool
        If True, each bootstrap fit uses ordered-probit initialization.
    verbose : bool
        If True, print progress over bootstrap replications.
    train_kwargs : dict or None
        Additional keyword arguments passed to train_ordered_flow, e.g.,
        {
          "monitor_nll": False,
          "check_every": 20,
          "burn_in_checks": 5,
          "neg_patience": 5
        }
    return_bootstrap_betas : bool
        If True, also return the full matrix of bootstrap beta draws
        (B x p) as a torch.Tensor on CPU.

    Returns
    -------
    full_model : OrderedFlowModel
        The flow model fitted on the full sample.
    beta_hat : np.ndarray, shape (p,)
        Full-sample beta coefficients after identification projection.
    se_boot : np.ndarray, shape (p,)
        Bootstrap standard errors for beta_hat.
    beta_boot (optional) : torch.Tensor, shape (B, p)
        All bootstrap beta estimates (if return_bootstrap_betas=True).
    """
    if train_kwargs is None:
        train_kwargs = {}

    # Ensure tensors are on the global device
    X = X.to(device)
    y = y.to(device)
    if Z is not None:
        Z = Z.to(device)

    n, p = X.shape

    # 1) Fit full-sample model
    if verbose:
        print("Fitting full-sample flow model for beta_hat...")
    full_model = train_ordered_flow(
        X, y, Z,
        flow_bins=flow_bins,
        bounds=bounds,
        epochs=epochs,
        lr=lr,
        init_probit=init_probit,
        verbose=verbose,
        **train_kwargs
    )
    beta_hat = full_model.beta.detach().cpu().numpy()

    # 2) Bootstrap refits
    beta_boot = torch.empty(B, p, dtype=torch.get_default_dtype())

    for b in range(B):
        # Sample indices with replacement
        idx = torch.randint(low=0, high=n, size=(n,), device=device)
        Xb = X[idx]
        yb = y[idx]
        Zb = Z[idx] if Z is not None else None

        if verbose:
            print(f"Bootstrap replication {b+1}/{B}...")

        # Train model on bootstrap sample
        model_b = train_ordered_flow(
            Xb, yb, Zb,
            flow_bins=flow_bins,
            bounds=bounds,
            epochs=epochs,
            lr=lr,
            init_probit=init_probit,
            verbose=False,      # suppress inner training logs; outer verbosity is enough
            **train_kwargs
        )

        beta_boot[b, :] = model_b.beta.detach()

    # 3) Compute bootstrap standard errors (on CPU)
    beta_boot_cpu = beta_boot.cpu()
    se_boot = beta_boot_cpu.std(dim=0, unbiased=True).numpy()

    if return_bootstrap_betas:
        return full_model, beta_hat, se_boot, beta_boot_cpu
    else:
        return full_model, beta_hat, se_boot
# }}}
