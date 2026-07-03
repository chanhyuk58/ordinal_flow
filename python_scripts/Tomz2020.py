# ========================= Tomz2020.py =========================

import os
# Prevent OpenMP runtime crashes and thread contention
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Crucial for parallel bootstrapping

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor

from ordinal_flow_core import (
    device,
    StructuredOrdinalFlow,
    ModelFreeOrdinalFlow,
    fit_ordered_sm_baseline,
)

torch.set_default_dtype(torch.float64)

# ============================================================
# Load & Prepare Data
# ============================================================
tomz = pd.read_stata("../replication_data/2012-10-01-Main-prepped.dta")
tomz["f_strike5"] = pd.factorize(tomz["strike5"], sort=True)[0] + 1

y = tomz["f_strike5"].to_numpy(dtype=int)
X = tomz[["hrts", "democ", "h1", "i1", "p1", "e1", "r1", "male", "white", "age", "ed4"]].to_numpy(dtype=float)
Z = tomz[["h1", "i1", "p1", "e1", "r1", "male", "white", "age", "ed4"]].to_numpy(dtype=float)
treat = X[:, 0].astype(int)

J = len(np.unique(y))
treatment_idx = 0

# ============================================================
# Helper: Empirical Estimator
# ============================================================
def get_empirical_effects(y_arr, d_arr, num_classes):
    p = {}
    for val in [0, 1]:
        mask = (d_arr == val)
        p[val] = np.array([(y_arr[mask] == j).mean() for j in range(1, num_classes + 1)], dtype=float)

    cat_eff = p[1] - p[0]
    cge_eff = (np.cumsum(p[1][::-1])[::-1] - np.cumsum(p[0][::-1])[::-1])[1:]  # Drop P(Y >= 1)
    wass = np.sum(np.abs(np.cumsum(p[1])[:-1] - np.cumsum(p[0])[:-1]))

    return {
        "wasserstein_unit": wass,
        "category_effect": cat_eff,
        "cum_ge_effect": cge_eff,
    }

# ============================================================
# Single Bootstrap Replication Task
# ============================================================
def run_one_bootstrap(seed):
    """Runs a single stratified bootstrap replication across all models."""
    rng = np.random.default_rng(seed)
    n = len(y)
    
    # Stratified Resampling
    idx = np.empty(n, dtype=int)
    for c in np.unique(y):
        c_indices = np.where(y == c)[0]
        idx[y == c] = rng.choice(c_indices, size=len(c_indices), replace=True)
        
    Xb, yb, Zb = X[idx], y[idx], Z[idx]
    
    # 1. Empirical
    emp_b = get_empirical_effects(yb, Xb[:, 0].astype(int), J)
    
    # 2. Ordered Probit & Logit
    op_b, _ = fit_ordered_sm_baseline(Xb, yb, treatment_idx=treatment_idx, link="probit")
    ol_b, _ = fit_ordered_sm_baseline(Xb, yb, treatment_idx=treatment_idx, link="logit")
    
    # 3. Structured Flow (Warm started)
    sf_model = StructuredOrdinalFlow(J=J, q=Zb.shape[1], flow_bins=16, bounds=10.0)
    sf_model.fit(Xb, yb, Z=Zb, epochs=30, lr=5e-3, use_lbfgs=True, verbose=False)
    sf_b = sf_model.compute_effects(Xb, treatment_idx=treatment_idx, Z=Zb)
    
    # 4. Model-Free Flow (Warm started)
    mf_model = ModelFreeOrdinalFlow(J=J, flow_bins=16, bounds=10.0)
    mf_model.fit(Xb, yb, epochs=30, lr=1e-2, use_lbfgs=True, verbose=False)
    mf_b = mf_model.compute_effects(Xb, treatment_idx=treatment_idx)
    
    return emp_b, op_b, ol_b, sf_b, mf_b

# ============================================================
# Main Execution Block
# ============================================================
if __name__ == "__main__":
    # 1. Compute Main Estimates
    print("Computing main point estimates...")
    empirical_est = get_empirical_effects(y, treat, J)
    oprobit_est, _ = fit_ordered_sm_baseline(X, y, treatment_idx=treatment_idx, link="probit")
    ologit_est, _ = fit_ordered_sm_baseline(X, y, treatment_idx=treatment_idx, link="logit")
    
    sf_model = StructuredOrdinalFlow(J=J, q=Z.shape[1], flow_bins=16, bounds=10.0)
    sf_model.fit(X, y, Z=Z, epochs=200, lr=5e-3, use_lbfgs=True, verbose=False)
    structured_est = sf_model.compute_effects(X, treatment_idx=treatment_idx, Z=Z)
    
    mf_model = ModelFreeOrdinalFlow(J=J, flow_bins=16, bounds=10.0)
    mf_model.fit(X, y, epochs=200, lr=1e-2, use_lbfgs=True, verbose=False)
    model_free_est = mf_model.compute_effects(X, treatment_idx=treatment_idx)

    # 2. Parallel Bootstrap Loop
    B = 100  # Number of Bootstrap replications
    print(f"\nRunning B={B} Stratified Bootstraps in parallel...")
    
    # Auto-detects number of available CPU cores
    max_workers = os.cpu_count()
    print(f"Allocating work across {max_workers} CPU cores.")
    
    seeds = np.random.SeedSequence(12345).generate_state(B)
    
    boot_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # We submit jobs to the processor pool
        futures = [executor.submit(run_one_bootstrap, b, seed) for b in range(B)]
        for b, fut in enumerate(futures):
            boot_results.append(fut.result())
            if (b + 1) % 10 == 0:
                print(f"  Completed {b + 1}/{B} replications")

    # 3. Process Bootstrap standard errors
    model_names = ["empirical", "oprobit", "ologit", "structured flow", "model_free flow"]
    estimates = [empirical_est, oprobit_est, ologit_est, structured_est, model_free_est]
    
    # Reshape bootstrap draws to calculate standard deviations
    boot_draws = {name: {"category_effect": [], "cum_ge_effect": [], "wasserstein_unit": []} for name in model_names}
    for rep in boot_results:
        for m_idx, name in enumerate(model_names):
            boot_draws[name]["category_effect"].append(rep[m_idx]["category_effect"])
            boot_draws[name]["cum_ge_effect"].append(rep[m_idx]["cum_ge_effect"])
            boot_draws[name]["wasserstein_unit"].append(rep[m_idx]["wasserstein_unit"])

    # Calculate standard deviation of standard bootstrap distribution
    ses = {}
    for name in model_names:
        ses[name] = {
            "category_effect": np.std(boot_draws[name]["category_effect"], axis=0, ddof=1),
            "cum_ge_effect": np.std(boot_draws[name]["cum_ge_effect"], axis=0, ddof=1),
            "wasserstein_unit": np.std(boot_draws[name]["wasserstein_unit"], axis=0, ddof=1),
        }

    # 4. Save results to a clean CSV for R
    rows = []
    for m_idx, name in enumerate(model_names):
        est_obj = estimates[m_idx]
        se_obj = ses[name]
        
        # Wasserstein Scalar
        rows.append({
            "model": name, "metric": "wasserstein_unit", "index": 1,
            "estimate": est_obj["wasserstein_unit"], "se": se_obj["wasserstein_unit"]
        })
        
        # Category Effects Vector
        for j in range(J):
            rows.append({
                "model": name, "metric": "category_effect", "index": j + 1,
                "estimate": est_obj["category_effect"][j], "se": se_obj["category_effect"][j]
            })
            
        # Cumulative GE Effects Vector (index starts at 2)
        for j in range(J - 1):
            rows.append({
                "model": name, "metric": "cum_ge_effect", "index": j + 2,
                "estimate": est_obj["cum_ge_effect"][j], "se": se_obj["cum_ge_effect"][j]
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv("../replication_results/tomz_results.csv", index=False)
    print("\nSaved standard errors and estimates to: tomz_results.csv")
