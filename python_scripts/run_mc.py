# ========================= run_mc.py =========================
import os
import math
import numpy as np
import pandas as pd
import torch
from ordinal_flow_core import (
    device,
    get_true_error,
    train_ordered_flow,
    train_model_free_flow,
    fit_ordered_sm,
)

torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------
# Parametric AME Helpers for Probit and Logit Baselines
# ----------------------------------------------------------------------
def compute_parametric_ame(beta_hat, thresholds, X, treatment_idx, link='probit'):
    n, p = X.shape
    X1 = X.clone()
    X1[:, treatment_idx] = 1.0
    X0 = X.clone()
    X0[:, treatment_idx] = 0.0

    eta1 = X1.cpu().numpy().dot(beta_hat)
    eta0 = X0.cpu().numpy().dot(beta_hat)

    # Compute bounds and differences
    probs1 = []
    probs0 = []
    J = len(thresholds) + 1

    cdf_fn = stats.norm.cdf if link == 'probit' else lambda x: 1.0 / (1.0 + np.exp(-x))

    for j in range(1, J + 1):
        if j == 1:
            u1, l1 = thresholds[0] - eta1, -np.inf
            u0, l0 = thresholds[0] - eta0, -np.inf
        elif j == J:
            u1, l1 = np.inf, thresholds[-1] - eta1
            u0, l0 = np.inf, thresholds[-1] - eta0
        else:
            u1, l1 = thresholds[j-1] - eta1, thresholds[j-2] - eta1
            u0, l0 = thresholds[j-1] - eta0, thresholds[j-2] - eta0

        probs1.append(cdf_fn(u1) - cdf_fn(l1))
        probs0.append(cdf_fn(u0) - cdf_fn(l0))

    p1 = np.stack(probs1, axis=1)
    p0 = np.stack(probs0, axis=1)
    return np.mean(p1 - p0, axis=0)


def compute_parametric_concordance_prob(beta_hat, thresholds, X, treatment_idx, link='probit'):
    n, p = X.shape
    X1 = X.clone()
    X1[:, treatment_idx] = 1.0
    X0 = X.clone()
    X0[:, treatment_idx] = 0.0

    eta1 = X1.cpu().numpy().dot(beta_hat)
    eta0 = X0.cpu().numpy().dot(beta_hat)

    probs1 = []
    probs0 = []
    J = len(thresholds) + 1
    cdf_fn = stats.norm.cdf if link == 'probit' else lambda x: 1.0 / (1.0 + np.exp(-x))

    for j in range(1, J + 1):
        if j == 1:
            u1, l1 = thresholds[0] - eta1, -np.inf
            u0, l0 = thresholds[0] - eta0, -np.inf
        elif j == J:
            u1, l1 = np.inf, thresholds[-1] - eta1
            u0, l0 = np.inf, thresholds[-1] - eta0
        else:
            u1, l1 = thresholds[j-1] - eta1, thresholds[j-2] - eta1
            u0, l0 = thresholds[j-1] - eta0, thresholds[j-2] - eta0

        probs1.append(cdf_fn(u1) - cdf_fn(l1))
        probs0.append(cdf_fn(u0) - cdf_fn(l0))

    p1 = np.stack(probs1, axis=1)
    p0 = np.stack(probs0, axis=1)

    cum_p0 = np.cumsum(p0, axis=1)
    cum_p1 = np.cumsum(p1, axis=1)

    p_superior = np.sum(p1[:, 1:] * cum_p0[:, :-1], axis=1)
    p_inferior = np.sum(p0[:, 1:] * cum_p1[:, :-1], axis=1)
    return float(np.mean(p_superior - p_inferior))


# ----------------------------------------------------------------------
# Monte Carlo Loop over Sample Sizes
# ----------------------------------------------------------------------
from scipy import stats

def run_unified_mc(
    df_pop,
    df_idx,
    n,
    beta_true,
    true_error_spec,
    R,
    treatment_idx=0,
    flow_bins=12,
    flow_bounds=10.0,
):
    # Establish population scale of true error
    _, true_cdf_std, _, _, m_eps, s_eps = get_true_error(true_error_spec)
    beta_true_np = beta_true.numpy()
    beta_true_norm = beta_true_np / float(s_eps) if s_eps else beta_true_np
    
    p = len(beta_true_np)
    J = 5
    
    # Establish thresholds in standard normal space using standard pop thresholds
    # Typically: [-1.0, -0.3, 0.5, 1.2]
    true_thresholds = np.array([-1.0, -0.3, 0.5, 1.2])

    # True AMEs and True Concordance Probabilities
    # Calculated analytically on the standard scale using true population covariates
    X_cols = [f"x{j}" for j in range(p)]
    X_pop_all = torch.from_numpy(df_pop[X_cols].values)
    X1_pop = X_pop_all.clone()
    X1_pop[:, treatment_idx] = 1.0
    X0_pop = X_pop_all.clone()
    X0_pop[:, treatment_idx] = 0.0

    eta1_pop = X1_pop.numpy().dot(beta_true_norm)
    eta0_pop = X0_pop.numpy().dot(beta_true_norm)

    true_p1_list, true_p0_list = [], []
    for j in range(1, J + 1):
        if j == 1:
            u1, l1 = true_thresholds[0] - eta1_pop, -np.inf
            u0, l0 = true_thresholds[0] - eta0_pop, -np.inf
        elif j == J:
            u1, l1 = np.inf, true_thresholds[-1] - eta1_pop
            u0, l0 = np.inf, true_thresholds[-1] - eta0_pop
        else:
            u1, l1 = true_thresholds[j-1] - eta1_pop, true_thresholds[j-2] - eta1_pop
            u0, l0 = true_thresholds[j-1] - eta0_pop, true_thresholds[j-2] - eta0_pop

        # Map bounds using standardized true error CDF
        true_p1_list.append(true_cdf_std(torch.tensor(u1)).numpy() - true_cdf_std(torch.tensor(l1)).numpy())
        true_p0_list.append(true_cdf_std(torch.tensor(u0)).numpy() - true_cdf_std(torch.tensor(l0)).numpy())

    true_p1 = np.stack(true_p1_list, axis=1)
    true_p0 = np.stack(true_p0_list, axis=1)
    true_ames = np.mean(true_p1 - true_p0, axis=0)

    cum_p0_true = np.cumsum(true_p0, axis=1)
    cum_p1_true = np.cumsum(true_p1, axis=1)
    p_superior_true = np.sum(true_p1[:, 1:] * cum_p0_true[:, :-1], axis=1)
    p_inferior_true = np.sum(true_p0[:, 1:] * cum_p1_true[:, :-1], axis=1)
    true_delta = float(np.mean(p_superior_true - p_inferior_true))

    # Output storage
    est_flow_beta, est_probit_beta, est_logit_beta = [], [], []
    est_flow_ames, est_cnf_ames, est_probit_ames, est_logit_ames, est_lpm_ames, est_raw_ames = [], [], [], [], [], []
    est_flow_delta, est_cnf_delta, est_probit_delta, est_logit_delta = [], [], [], []

    df_n = df_idx[df_idx["n"] == n].copy()

    for r in range(R):
        df_r = df_n[df_n["rep"] == r]
        idx = df_r["idx"].values.astype(int)
        sample = df_pop.iloc[idx]
        y = torch.from_numpy(sample["y"].values.astype(int))
        X = torch.from_numpy(sample[X_cols].values)
        D = X[:, treatment_idx]

        # 1. Baseline Flow Model
        try:
            m_flow = train_ordered_flow(X, y, Z=None, flow_bins=flow_bins, bounds=flow_bounds, verbose=False)
            est_flow_beta.append(m_flow.beta.detach().cpu().numpy())
            est_flow_ames.append(m_flow.compute_ame(X, treatment_idx))
            est_flow_delta.append(m_flow.compute_concordance_prob(X, treatment_idx))
        except Exception:
            est_flow_beta.append(np.full(p, np.nan))
            est_flow_ames.append(np.full(J, np.nan))
            est_flow_delta.append(np.nan)

        # 2. Model-Free Conditional Flow (CNF)
        # Treatment variable passed as target; covariates are remaining columns
        try:
            X_covs = torch.cat([X[:, :treatment_idx], X[:, treatment_idx+1:]], dim=1) if p > 1 else torch.empty(n, 0)
            m_cnf = train_model_free_flow(X_covs, y, D, flow_bins=flow_bins, bounds=flow_bounds, verbose=False)
            est_cnf_ames.append(m_cnf.compute_ame(X_covs))
            est_cnf_delta.append(m_cnf.compute_concordance_prob(X_covs))
        except Exception:
            est_cnf_ames.append(np.full(J, np.nan))
            est_cnf_delta.append(np.nan)

        # 3. Ordered Probit
        try:
            b_p, thr_p, _ = fit_ordered_sm(y, X, link='probit', normalize=True)
            est_probit_beta.append(b_p)
            est_probit_ames.append(compute_parametric_ame(b_p, thr_p, X, treatment_idx, link='probit'))
            est_probit_delta.append(compute_parametric_concordance_prob(b_p, thr_p, X, treatment_idx, link='probit'))
        except Exception:
            est_probit_beta.append(np.full(p, np.nan))
            est_probit_ames.append(np.full(J, np.nan))
            est_probit_delta.append(np.nan)

        # 4. Ordered Logit
        try:
            b_l, thr_l, _ = fit_ordered_sm(y, X, link='logit', normalize=True)
            est_logit_beta.append(b_l)
            est_logit_ames.append(compute_parametric_ame(b_l, thr_l, X, treatment_idx, link='logit'))
            est_logit_delta.append(compute_parametric_concordance_prob(b_l, thr_l, X, treatment_idx, link='logit'))
        except Exception:
            est_logit_beta.append(np.full(p, np.nan))
            est_logit_ames.append(np.full(J, np.nan))
            est_logit_delta.append(np.nan)

        # 5. Linear Probability Models (LPM / OLS per threshold)
        lpm_r = []
        for j in range(2, J + 1):
            W = (y.numpy() >= j).astype(float)
            X_np = X.numpy()
            b_lpm = np.linalg.pinv(X_np.T @ X_np).dot(X_np.T @ W)
            lpm_r.append(b_lpm[treatment_idx])
        # Reconstruct category probabilities from threshold indicators
        # P(Y=j) = P(Y>=j) - P(Y>=j+1)
        lpm_prob_diffs = []
        lpm_r = [1.0] + lpm_r + [0.0]  # Boundaries
        for j in range(1, J + 1):
            lpm_prob_diffs.append(lpm_r[j-1] - lpm_r[j])
        est_lpm_ames.append(np.array(lpm_prob_diffs))

        # 6. Raw Differences in Proportions (Cross-Tabs)
        raw_r = []
        y_np = y.numpy()
        D_np = D.numpy()
        for j in range(1, J + 1):
            p1_raw = np.mean(y_np[D_np == 1.0] == j)
            p0_raw = np.mean(y_np[D_np == 0.0] == j)
            raw_r.append(p1_raw - p0_raw)
        est_raw_ames.append(np.array(raw_r))

        if (r + 1) % 10 == 0:
            print(f"  n={n}: replication {r+1}/{R} completed")

    # Metrics summarization helper
    def summarize(est_list, true_val):
        est_arr = np.array(est_list)
        err = est_arr - true_val[None, ...]
        bias = np.nanmean(err, axis=0)
        rmse = np.sqrt(np.nanmean(err**2, axis=0))
        return bias, rmse

    # Summarize Latent Coefficients
    bias_flow_b, rmse_flow_b = summarize(est_flow_beta, beta_true_norm)
    bias_probit_b, rmse_probit_b = summarize(est_probit_beta, beta_true_norm)
    bias_logit_b, rmse_logit_b = summarize(est_logit_beta, beta_true_norm)

    # Summarize Probability AMEs
    bias_flow_a, rmse_flow_a = summarize(est_flow_ames, true_ames)
    bias_cnf_a, rmse_cnf_a = summarize(est_cnf_ames, true_ames)
    bias_probit_a, rmse_probit_a = summarize(est_probit_ames, true_ames)
    bias_logit_a, rmse_logit_a = summarize(est_logit_ames, true_ames)
    bias_lpm_a, rmse_lpm_a = summarize(est_lpm_ames, true_ames)
    bias_raw_a, rmse_raw_a = summarize(est_raw_ames, true_ames)

    # Summarize Concordance Probabilities
    bias_flow_d, rmse_flow_d = summarize(est_flow_delta, true_delta)
    bias_cnf_d, rmse_cnf_d = summarize(est_cnf_delta, true_delta)
    bias_probit_d, rmse_probit_d = summarize(est_probit_delta, true_delta)
    bias_logit_d, rmse_logit_d = summarize(est_logit_delta, true_delta)

    # Compile unified JSON-like rows
    output_records = []
    # 1. Latent Coefficients
    for j in range(p):
        for name, b, r in [("flow", bias_flow_b, rmse_flow_b), 
                           ("probit", bias_probit_b, rmse_probit_b), 
                           ("logit", bias_logit_b, rmse_logit_b)]:
            output_records.append({
                "n": n, "metric": "latent_beta", "coef": j, "model": name,
                "bias": b[j], "rmse": r[j], "truth": beta_true_norm[j]
            })

    # 2. Probability AMEs
    for j in range(J):
        for name, b, r in [("flow", bias_flow_a, rmse_flow_a),
                           ("cnf", bias_cnf_a, rmse_cnf_a),
                           ("probit", bias_probit_a, rmse_probit_a),
                           ("logit", bias_logit_a, rmse_logit_a),
                           ("lpm", bias_lpm_a, rmse_lpm_a),
                           ("raw", bias_raw_a, rmse_raw_a)]:
            output_records.append({
                "n": n, "metric": "prob_ame", "coef": j, "model": name,
                "bias": b[j], "rmse": r[j], "truth": true_ames[j]
            })

    # 3. Concordance Probability (Delta)
    for name, b, r in [("flow", bias_flow_d, rmse_flow_d),
                       ("cnf", bias_cnf_d, rmse_cnf_d),
                       ("probit", bias_probit_d, rmse_probit_d),
                       ("logit", bias_logit_d, rmse_logit_d)]:
        output_records.append({
            "n": n, "metric": "concordance_delta", "coef": 0, "model": name,
            "bias": float(b), "rmse": float(r), "truth": true_delta
        })

    return output_records


def main():
    N = int(1e6)
    true_error_spec = "lognormal"
    sample_sizes = [500, 1000]
    R = 100
    beta_true = torch.tensor([1.0, 0.8, -0.8, 0.5, -0.5])

    # Directory Paths
    pop_csv = f"../sim_data/population_{true_error_spec}_N{N}.csv"
    idx_csv = f"../sim_data/indices_{true_error_spec}_N{N}.csv"
    out_dir = "../mc_results/"
    os.makedirs(out_dir, exist_ok=True)

    df_pop = pd.read_csv(pop_csv)
    df_idx = pd.read_csv(idx_csv)

    print(f"Unified MC Setup: Pop shape={df_pop.shape}, Error Spec={true_error_spec}")

    all_results = []
    for n in sample_sizes:
        print(f"Running Unified MC Loop for n={n}")
        res_n = run_unified_mc(
            df_pop=df_pop,
            df_idx=df_idx,
            n=n,
            beta_true=beta_true,
            true_error_spec=true_error_spec,
            R=R,
            flow_bins=12,
            flow_bounds=10.0
        )
        all_results.extend(res_n)

    df_res = pd.DataFrame(all_results)
    out_csv = os.path.join(out_dir, f"mc_unified_{R}_{true_error_spec}.csv")
    df_res.to_csv(out_csv, index=False)
    print("Simulation completed successfully. Sample Output (First Coefficient AMEs):")
    print(df_res.loc[(df_res.metric == "prob_ame") & (df_res.coef == 0), :].head(12))

if __name__ == "__main__":
    main()
