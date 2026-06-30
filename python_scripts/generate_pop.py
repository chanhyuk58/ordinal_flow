# =========================== generate_pop.py ===========================

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_THRESHOLDS = np.array([-1.40, -0.45, 0.45, 1.40], dtype=float)
DEFAULT_SAMPLE_SIZES = [200, 500, 1000, 2000]

@dataclass(frozen=True)
class DGPConfig:
    setting: str
    description: str
    p: int
    J: int
    thresholds: Tuple[float, ...]
    tau: float
    error_type: str
    h_type: str
    favorable_to_nf: bool

SETTINGS: Dict[str, DGPConfig] = {
    "normal_linear": DGPConfig(
        setting="normal_linear",
        description="Correctly specified ordered-probit benchmark: linear index and Gaussian latent error.",
        p=5, J=5, thresholds=tuple(DEFAULT_THRESHOLDS), tau=0.55,
        error_type="normal", h_type="linear", favorable_to_nf=False,
    ),
    "logistic_linear": DGPConfig(
        setting="logistic_linear",
        description="Correctly specified ordered-logit-style benchmark: linear index and standardized logistic latent error.",
        p=5, J=5, thresholds=tuple(DEFAULT_THRESHOLDS), tau=0.55,
        error_type="logistic", h_type="linear", favorable_to_nf=False,
    ),
    "skewed_lognormal": DGPConfig(
        setting="skewed_lognormal",
        description="Linear index with strongly skewed centered log-normal latent error; favorable to flexible error modeling.",
        p=5, J=5, thresholds=(-1.20, -0.25, 0.55, 1.55), tau=0.55,
        error_type="lognormal", h_type="linear", favorable_to_nf=True,
    ),
    "polarized_mixture": DGPConfig(
        setting="polarized_mixture",
        description="Linear index with bimodal mixture error representing polarized latent attitudes; favorable to flexible error modeling.",
        p=5, J=5, thresholds=(-1.55, -0.55, 0.55, 1.55), tau=0.55,
        error_type="mixture", h_type="linear", favorable_to_nf=True,
    ),
    "heteroskedastic": DGPConfig(
        setting="heteroskedastic",
        description="Linear index with covariate- and treatment-dependent latent error scale; favorable to conditional distribution modeling.",
        p=5, J=5, thresholds=tuple(DEFAULT_THRESHOLDS), tau=0.50,
        error_type="heteroskedastic_normal", h_type="linear", favorable_to_nf=True,
    ),
    "nonlinear_moderates": DGPConfig(
        setting="nonlinear_moderates",
        description="Nonlinear treatment response: treatment mainly moves respondents near the middle of the latent scale.",
        p=5, J=5, thresholds=(-1.25, -0.35, 0.35, 1.25), tau=0.0,
        error_type="normal", h_type="nonlinear_moderates", favorable_to_nf=True,
    ),
    "high_dimensional": DGPConfig(
        setting="high_dimensional",
        description="Rich covariate setting with sparse nonlinear index and heterogeneous treatment response.",
        p=20, J=5, thresholds=(-1.35, -0.40, 0.40, 1.35), tau=0.0,
        error_type="mixture", h_type="high_dimensional", favorable_to_nf=True,
    ),
}

def make_covariates(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    if p < 2: raise ValueError("p must be at least 2 because x0 is the treatment.")
    X = rng.normal(size=(n, p))
    X[:, 0] = rng.binomial(1, 0.5, size=n)   # randomized treatment
    X[:, 1] = rng.binomial(1, 0.5, size=n)   # binary pretreatment covariate
    return X.astype(float)

def base_linear_index(X: np.ndarray) -> np.ndarray:
    p = X.shape[1]
    beta_cov = np.zeros(p)
    template = np.array([0.55, -0.55, 0.35, -0.30, 0.20, -0.18, 0.15, -0.12])
    m = min(p - 1, len(template))
    beta_cov[1 : 1 + m] = template[:m]
    return X @ beta_cov

def systematic_component(X: np.ndarray, d: int, cfg: DGPConfig) -> np.ndarray:
    Xd = X.copy()
    Xd[:, 0] = float(d)

    if cfg.h_type == "linear":
        return base_linear_index(Xd) + cfg.tau * float(d)

    if cfg.h_type == "nonlinear_moderates":
        eta0 = 1.2*np.sin(2*np.pi*Xd[:,1]) + 0.8*Xd[:,2]**2 - 0.8*Xd[:,3]*Xd[:,4]
        tau_i = 1.5*np.exp(-(eta0**2)/0.6)
        return eta0 + float(d) * tau_i

    if cfg.h_type == "high_dimensional":
        eta0 = base_linear_index(Xd) + 0.45 * np.sin(Xd[:, 5]) + 0.35 * Xd[:, 6] * Xd[:, 7] - 0.30 * (Xd[:, 8] > 0.75).astype(float)
        tau_i = 0.35 + 0.55 / (1.0 + np.exp(-Xd[:, 2])) - 0.25 * (np.abs(Xd[:, 3]) > 1.0)
        return eta0 + float(d) * tau_i

    raise ValueError(f"Unknown h_type: {cfg.h_type}")

def standardized_lognormal(z: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    raw = np.exp(sigma * z)
    mean = np.exp(0.5 * sigma**2)
    var = (np.exp(sigma**2) - 1.0) * np.exp(sigma**2)
    return (raw - mean) / np.sqrt(var)

def standardized_mixture(rng: np.random.Generator, n: int) -> np.ndarray:
    mu, sd = 2.6, 0.25
    u = rng.uniform(size=n)
    raw = np.empty(n)
    
    left, mid, right = (u < 0.45), ((u >= 0.45) & (u < 0.55)), (u >= 0.55)
    raw[left]  = rng.normal(-mu, sd, left.sum())
    raw[mid]   = rng.normal(0.0, 0.15, mid.sum())
    raw[right] = rng.normal(mu, sd, right.sum())
    
    return (raw - raw.mean()) / raw.std()

def draw_potential_errors(cfg: DGPConfig, X: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    z = rng.normal(size=n)

    if cfg.error_type == "normal":
        return z, z.copy(), z
    if cfg.error_type == "logistic":
        u = rng.uniform(size=n)
        eps = np.log(u / (1.0 - u)) / (np.pi / np.sqrt(3.0))
        return eps, eps.copy(), (np.log(u / (1.0 - u)))
    if cfg.error_type == "lognormal":
        eps = standardized_lognormal(z, sigma=1.75)
        return eps, eps.copy(), z
    if cfg.error_type == "mixture":
        eps = standardized_mixture(rng, n)
        return eps, eps.copy(), z
    if cfg.error_type == "heteroskedastic_normal":
        x2 = X[:, 2] if X.shape[1] > 2 else 0.0
        scale0 = np.clip(np.exp(0.8*x2), 0.30, 4.0)
        scale1 = np.clip(np.exp(0.8*x2 + 0.9), 0.40, 5.0)
        return scale0 * z, scale1 * z, z

    raise ValueError(f"Unknown error_type: {cfg.error_type}")

def categorize(y_star: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(y_star, thresholds, right=False).astype(np.int8) + 1

def compute_truth(y0: np.ndarray, y1: np.ndarray, J: int) -> Dict[str, object]:
    p0 = np.array([(y0 == j).mean() for j in range(1, J + 1)])
    p1 = np.array([(y1 == j).mean() for j in range(1, J + 1)])
    cat_effect = p1 - p0

    # FIX: Start range at 2 to drop the uninformative P(Y >= 1) which is always 1.0
    cum_ge0 = np.array([(y0 >= j).mean() for j in range(2, J + 1)])
    cum_ge1 = np.array([(y1 >= j).mean() for j in range(2, J + 1)])
    cum_ge_effect = cum_ge1 - cum_ge0

    wasserstein_unit = float(np.sum(np.abs(np.cumsum(p1)[:-1] - np.cumsum(p0)[:-1])))
    net_improvement = float(np.mean(y1 > y0) - np.mean(y0 > y1))

    return {
        "p0": p0.tolist(),
        "p1": p1.tolist(),
        "category_effect": cat_effect.tolist(),
        "cum_ge0": cum_ge0.tolist(),
        "cum_ge1": cum_ge1.tolist(),
        "cum_ge_effect": cum_ge_effect.tolist(),
        "wasserstein_unit": wasserstein_unit,
        "net_improvement_joint_only": net_improvement,
    }

def summarize_population(y_star0, y_star1, eps0, eps1, y0, y1) -> Dict[str, float]:
    def moments(x: np.ndarray, prefix: str) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        mu, sd = float(np.mean(x)), float(np.std(x))
        skew = float(np.mean(((x - mu) / sd) ** 3)) if sd > 0 else np.nan
        kurt = float(np.mean(((x - mu) / sd) ** 4)) if sd > 0 else np.nan
        return {f"{prefix}_mean": mu, f"{prefix}_sd": sd, f"{prefix}_skewness": skew, f"{prefix}_kurtosis": kurt}

    out = {}
    for var, name in [(y_star0, "y_star0"), (y_star1, "y_star1"), (eps0, "eps0"), (eps1, "eps1")]:
        out.update(moments(var, name))

    out.update({
        "share_improved_category": float(np.mean(y1 > y0)),
        "share_worsened_category": float(np.mean(y0 > y1)),
        "share_unchanged_category": float(np.mean(y1 == y0))
    })
    return out

def generate_population(cfg: DGPConfig, N: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    thresholds = np.array(cfg.thresholds, dtype=float)
    X = make_covariates(N, cfg.p, rng)
    D = X[:, 0].astype(np.int8)

    eta0, eta1 = systematic_component(X, 0, cfg), systematic_component(X, 1, cfg)
    eps0, eps1, eps_base = draw_potential_errors(cfg, X, rng)

    y_star0, y_star1 = eta0 + eps0, eta1 + eps1
    y0, y1 = categorize(y_star0, thresholds), categorize(y_star1, thresholds)

    y_star = np.where(D == 1, y_star1, y_star0)
    y = np.where(D == 1, y1, y0)
    eps = np.where(D == 1, eps1, eps0)
    eta = np.where(D == 1, eta1, eta0)

    # OPTIMIZATION: Downcasting to save massive amounts of RAM and Disk Space
    df = pd.DataFrame({
        "id": np.arange(N, dtype=np.int32),
        "y": y.astype(np.int8),
        "y_star": y_star.astype(np.float32),
        "eps": eps.astype(np.float32),
        "eta": eta.astype(np.float32),
        "y0": y0.astype(np.int8),
        "y1": y1.astype(np.int8),
        "y_star0": y_star0.astype(np.float32),
        "y_star1": y_star1.astype(np.float32),
        "eps0": eps0.astype(np.float32),
        "eps1": eps1.astype(np.float32),
        "eta0": eta0.astype(np.float32),
        "eta1": eta1.astype(np.float32),
        "eps_base": eps_base.astype(np.float32),
    })

    for j in range(cfg.p):
        if j in [0, 1]:  # x0 and x1 are binary
            df[f"x{j}"] = X[:, j].astype(np.int8)
        else:
            df[f"x{j}"] = X[:, j].astype(np.float32)

    for k, thr in enumerate(thresholds, start=1):
        df[f"thr{k}"] = float(thr)
    
    meta = {
        "setting": cfg.setting, "description": cfg.description, "N": int(N), "seed": int(seed),
        "p": int(cfg.p), "J": int(cfg.J), "thresholds": list(cfg.thresholds), "tau": float(cfg.tau),
        "error_type": cfg.error_type, "h_type": cfg.h_type, "favorable_to_nf": bool(cfg.favorable_to_nf),
        "truth": compute_truth(y0, y1, cfg.J),
        "latent": {"tau": float(cfg.tau), "thresholds": list(cfg.thresholds), "error_type": cfg.error_type, "h_type": cfg.h_type, "p": int(cfg.p), "J": int(cfg.J)},
        "population_summary": summarize_population(y_star0, y_star1, eps0, eps1, y0, y1),
    }
    return df, meta

def generate_indices(N: int, sample_sizes: list[int], R: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    records = []
    for n in sample_sizes:
        idx = rng.integers(low=0, high=N, size=(R, n))
        records.append(pd.DataFrame({
            "n": np.full(R * n, n, dtype=np.int32),
            "rep": np.repeat(np.arange(R, dtype=np.int32), n),
            "idx": idx.reshape(-1).astype(np.int32)
        }))
    return pd.concat(records, ignore_index=True)

def plot_population(df: pd.DataFrame, cfg: DGPConfig, fig_dir: str, N: int, max_plot: int = 200_000) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    rng = np.random.default_rng(999)
    plot_df = df.sample(n=min(len(df), max_plot), random_state=999)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(plot_df["y_star"], bins=200, density=True, alpha=0.40, color="steelblue")
    for t in cfg.thresholds:
        ax.axvline(t, color="red", linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"Latent outcome $Y^*$")
    ax.set_ylabel("Density")
    ax.set_title(cfg.setting.replace("_", " ").title())
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"latent_space_{cfg.setting}_N{N}.pdf"))
    plt.close(fig)

def parse_sample_sizes(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def run_one_setting(args: argparse.Namespace, cfg: DGPConfig) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    print(f"Generating setting={cfg.setting}, N={args.N}, seed={args.seed}")
    df, meta = generate_population(cfg, args.N, args.seed)

    pop_csv = os.path.join(args.out_dir, f"population_{cfg.setting}_N{args.N}.csv")
    df.to_csv(pop_csv, index=False)
    print(f"Saved population: {pop_csv}")

    meta_json = os.path.join(args.out_dir, f"metadata_{cfg.setting}_N{args.N}.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    df_idx = generate_indices(args.N, parse_sample_sizes(args.sample_sizes), args.R, args.seed)
    idx_csv = os.path.join(args.out_dir, f"indices_{cfg.setting}_N{args.N}.csv")
    df_idx.to_csv(idx_csv, index=False)
    print(f"Saved indices: {idx_csv}")

    if not args.no_plots:
        plot_population(df, cfg, args.fig_dir, args.N)

    print("True category effects:", np.round(meta["truth"]["category_effect"], 4))
    print("True cumulative >= effects:", np.round(meta["truth"]["cum_ge_effect"], 4))
    print("True unit Wasserstein:", round(meta["truth"]["wasserstein_unit"], 4))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="skewed_lognormal", choices=sorted(SETTINGS.keys()))
    parser.add_argument("--all-settings", action="store_true")
    parser.add_argument("--N", type=int, default=1_000_000)
    parser.add_argument("--R", type=int, default=1000)
    parser.add_argument("--sample-sizes", default=",".join(map(str, DEFAULT_SAMPLE_SIZES)))
    parser.add_argument("--seed", type=int, default=23048)
    parser.add_argument("--out-dir", default="../sim_data")
    parser.add_argument("--fig-dir", default="../figures")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    for cfg in (SETTINGS.values() if args.all_settings else [SETTINGS[args.setting]]):
        run_one_setting(args, cfg)

if __name__ == "__main__":
    main()
