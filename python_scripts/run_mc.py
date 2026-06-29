# ========================= run_mc.py =========================
"""
Example
-------
python run_mc.py --setting skewed_lognormal --N 1000000 --R 100 --sample-sizes 500,1000
python run_mc.py --all-settings --N 500000 --R 50 --sample-sizes 500,1000
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ordinal_flow_core import (
    category_effect_from_probs,
    cumulative_ge_effect_from_probs,
    cumulative_ge_from_probs,
    device,
    empirical_probs_by_treatment,
    fit_ordered_sm,
    ordered_sm_effects,
    train_model_free_flow,
    train_ordered_flow,
    wasserstein_unit_from_probs,
)

torch.set_default_dtype(torch.float64)


DEFAULT_SETTINGS = [
    "normal_linear",
    "logistic_linear",
    "skewed_lognormal",
    "polarized_mixture",
    "heteroskedastic",
    "nonlinear_moderates",
    "high_dimensional",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_sample_sizes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def x_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def load_metadata(path: Optional[str]) -> Dict:
    if path is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def truth_from_metadata(meta: Dict) -> Dict[str, object]:
    if not meta or "truth" not in meta:
        raise ValueError(
            "Metadata JSON with a 'truth' field is required. "
            "Regenerate populations with the revised generate_pop.py."
        )

    truth = meta["truth"]
    return {
        "J": int(meta.get("J", len(truth["category_effect"]))),
        "p0": np.asarray(truth["p0"], dtype=float),
        "p1": np.asarray(truth["p1"], dtype=float),
        "category_effect": np.asarray(truth["category_effect"], dtype=float),
        "cum_ge0": np.asarray(truth["cum_ge0"], dtype=float),
        "cum_ge1": np.asarray(truth["cum_ge1"], dtype=float),
        "cum_ge_effect": np.asarray(truth["cum_ge_effect"], dtype=float),
        "wasserstein_unit": float(truth["wasserstein_unit"]),
    }

def safe_array(x, length: Optional[int] = None) -> np.ndarray:
    if x is None:
        if length is None:
            return np.array([np.nan])
        return np.full(length, np.nan)
    arr = np.asarray(x, dtype=float)
    if length is not None and arr.shape[0] != length:
        out = np.full(length, np.nan)
        m = min(length, arr.shape[0])
        out[:m] = arr[:m]
        return out
    return arr


def summarize_vector(estimates: List[np.ndarray], truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean estimate, bias, RMSE for vector estimates."""
    arr = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = arr - truth[None, :]
    mean_est = np.nanmean(arr, axis=0)
    bias = np.nanmean(err, axis=0)
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))
    return mean_est, bias, rmse


def summarize_scalar(estimates: List[float], truth: float) -> Tuple[float, float, float]:
    arr = np.asarray(estimates, dtype=float)
    err = arr - float(truth)
    mean_est = float(np.nanmean(arr))
    bias = float(np.nanmean(err))
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    return mean_est, bias, rmse


def append_vector_records(
    records: List[Dict],
    *,
    setting: str,
    n: int,
    model: str,
    metric: str,
    truth: np.ndarray,
    estimates: List[np.ndarray],
) -> None:
    mean_est, bias, rmse = summarize_vector(estimates, truth)
    for j in range(len(truth)):
        records.append({
            "setting": setting,
            "n": n,
            "model": model,
            "metric": metric,
            "index": j + 1,
            "truth": float(truth[j]),
            "estimate_mean": float(mean_est[j]),
            "bias": float(bias[j]),
            "rmse": float(rmse[j]),
        })


def append_scalar_record(
    records: List[Dict],
    *,
    setting: str,
    n: int,
    model: str,
    metric: str,
    truth: float,
    estimates: List[float],
) -> None:
    mean_est, bias, rmse = summarize_scalar(estimates, truth)
    records.append({
        "setting": setting,
        "n": n,
        "model": model,
        "metric": metric,
        "index": 1,
        "truth": float(truth),
        "estimate_mean": float(mean_est),
        "bias": float(bias),
        "rmse": float(rmse),
    })


def model_failure_row(setting: str, n: int, rep: int, model: str, err: Exception) -> Dict:
    return {
        "setting": setting,
        "n": n,
        "rep": rep,
        "model": model,
        "error": repr(err),
        "traceback": traceback.format_exc(limit=3),
    }


# ---------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------

def evaluate_empirical(y: np.ndarray, d: np.ndarray, J: int) -> Dict[str, object]:
    out = empirical_probs_by_treatment(y, d, J=J)
    return {
        "category_effect": safe_array(out["category_effect"], J),
        "cum_ge_effect": safe_array(out["cum_ge_effect"], J),
        "wasserstein_unit": float(out["wasserstein_unit"]),
        "beta": None,
    }


def evaluate_ordered_model(
    *,
    y: torch.Tensor,
    X: torch.Tensor,
    link: str,
    treatment_idx: int,
    J: int,
) -> Dict[str, object]:
    beta, thresholds, res = fit_ordered_sm(y, X, link=link, normalize=True)
    effects = ordered_sm_effects(res, X, treatment_idx=treatment_idx)
    return {
        "category_effect": safe_array(effects["category_effect"], J),
        "cum_ge_effect": safe_array(effects["cum_ge_effect"], J),
        "wasserstein_unit": float(effects["wasserstein_unit"]),
        "beta": safe_array(beta, X.shape[1]),
        "thresholds": safe_array(thresholds),
    }


def evaluate_structured_flow(
    *,
    y: torch.Tensor,
    X: torch.Tensor,
    treatment_idx: int,
    J: int,
    flow_bins: int,
    flow_bounds: float,
    epochs: int,
    lr: float,
    use_lbfgs: bool,
    lbfgs_steps: int,
    verbose: bool,
) -> Dict[str, object]:
    model = train_ordered_flow(
        X,
        y,
        Z=None,
        flow_bins=flow_bins,
        bounds=flow_bounds,
        epochs=epochs,
        lr=lr,
        use_lbfgs=use_lbfgs,
        lbfgs_steps=lbfgs_steps,
        init_probit=True,
        verbose=verbose,
    )
    p1, p0 = model.counterfactual_probs(X, treatment_idx=treatment_idx)
    beta = model.beta.detach().cpu().numpy()
    return {
        "category_effect": category_effect_from_probs(p1, p0),
        "cum_ge_effect": cumulative_ge_effect_from_probs(p1, p0),
        "wasserstein_unit": wasserstein_unit_from_probs(p1, p0),
        "beta": safe_array(beta, X.shape[1]),
    }


def evaluate_model_free_flow(
    *,
    y: torch.Tensor,
    X: torch.Tensor,
    treatment_idx: int,
    J: int,
    flow_bins: int,
    flow_bounds: float,
    epochs: int,
    lr: float,
    use_lbfgs: bool,
    lbfgs_steps: int,
    verbose: bool,
) -> Dict[str, object]:
    D = X[:, treatment_idx]
    if X.shape[1] > 1:
        X_cov = torch.cat([X[:, :treatment_idx], X[:, treatment_idx + 1:]], dim=1)
    else:
        X_cov = torch.empty((X.shape[0], 0), dtype=X.dtype, device=X.device)

    model = train_model_free_flow(
        X_cov,
        y,
        D,
        flow_bins=flow_bins,
        bounds=flow_bounds,
        epochs=epochs,
        lr=lr,
        use_lbfgs=use_lbfgs,
        lbfgs_steps=lbfgs_steps,
        init_probit=True,
        verbose=verbose,
    )
    p1, p0 = model.counterfactual_probs(X_cov)
    return {
        "category_effect": category_effect_from_probs(p1, p0),
        "cum_ge_effect": cumulative_ge_effect_from_probs(p1, p0),
        "wasserstein_unit": wasserstein_unit_from_probs(p1, p0),
        "beta": None,
    }


# ---------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------
def run_mc_for_n(
    *,
    df_pop: pd.DataFrame,
    df_idx: pd.DataFrame,
    truth: Dict[str, object],
    setting: str,
    n: int,
    R: int,
    J: int,
    treatment_idx: int,
    flow_bins: int,
    flow_bounds: float,
    epochs: int,
    lr: float,
    use_lbfgs: bool,
    lbfgs_steps_structured: int,
    lbfgs_steps_model_free: int,
    verbose: bool,
) -> Tuple[List[Dict], List[Dict]]:
    cols = x_columns(df_pop)
    if len(cols) == 0:
        raise ValueError("No x columns found in population data.")

    true_cat = np.asarray(truth["category_effect"], dtype=float)
    true_cum = np.asarray(truth["cum_ge_effect"], dtype=float)
    true_wass = float(truth["wasserstein_unit"])

    estimates: Dict[str, Dict[str, List]] = {
        "empirical": {"cat": [], "cum": [], "wass": [], "beta": []},
        "ordered_probit": {"cat": [], "cum": [], "wass": [], "beta": []},
        "ordered_logit": {"cat": [], "cum": [], "wass": [], "beta": []},
        "structured_flow": {"cat": [], "cum": [], "wass": [], "beta": []},
        "model_free_flow": {"cat": [], "cum": [], "wass": [], "beta": []},
    }

    failures: List[Dict] = []
    df_n = df_idx[df_idx["n"] == n].copy()

    for r in range(R):
        df_r = df_n[df_n["rep"] == r]
        if df_r.empty:
            failures.append({
                "setting": setting,
                "n": n,
                "rep": r,
                "model": "index",
                "error": "Missing replication indices.",
                "traceback": "",
            })
            continue

        idx = df_r["idx"].to_numpy(dtype=int)
        sample = df_pop.iloc[idx]

        y_np = sample["y"].to_numpy(dtype=int)
        X_np = sample[cols].to_numpy(dtype=float)

        y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
        X = torch.from_numpy(X_np).to(device=device, dtype=torch.float64)
        d_np = X_np[:, treatment_idx].astype(int)

        # 1. Empirical treatment-arm distribution
        try:
            out = evaluate_empirical(y_np, d_np, J)
            estimates["empirical"]["cat"].append(out["category_effect"])
            estimates["empirical"]["cum"].append(out["cum_ge_effect"])
            estimates["empirical"]["wass"].append(out["wasserstein_unit"])
        except Exception as err:
            failures.append(model_failure_row(setting, n, r, "empirical", err))
            estimates["empirical"]["cat"].append(np.full(J, np.nan))
            estimates["empirical"]["cum"].append(np.full(J, np.nan))
            estimates["empirical"]["wass"].append(np.nan)

        # 2. Ordered probit
        try:
            out = evaluate_ordered_model(
                y=y,
                X=X,
                link="probit",
                treatment_idx=treatment_idx,
                J=J,
            )
            estimates["ordered_probit"]["cat"].append(out["category_effect"])
            estimates["ordered_probit"]["cum"].append(out["cum_ge_effect"])
            estimates["ordered_probit"]["wass"].append(out["wasserstein_unit"])
            estimates["ordered_probit"]["beta"].append(out["beta"])
        except Exception as err:
            failures.append(model_failure_row(setting, n, r, "ordered_probit", err))
            estimates["ordered_probit"]["cat"].append(np.full(J, np.nan))
            estimates["ordered_probit"]["cum"].append(np.full(J, np.nan))
            estimates["ordered_probit"]["wass"].append(np.nan)
            estimates["ordered_probit"]["beta"].append(np.full(len(cols), np.nan))

        # 3. Ordered logit
        try:
            out = evaluate_ordered_model(
                y=y,
                X=X,
                link="logit",
                treatment_idx=treatment_idx,
                J=J,
            )
            estimates["ordered_logit"]["cat"].append(out["category_effect"])
            estimates["ordered_logit"]["cum"].append(out["cum_ge_effect"])
            estimates["ordered_logit"]["wass"].append(out["wasserstein_unit"])
            estimates["ordered_logit"]["beta"].append(out["beta"])
        except Exception as err:
            failures.append(model_failure_row(setting, n, r, "ordered_logit", err))
            estimates["ordered_logit"]["cat"].append(np.full(J, np.nan))
            estimates["ordered_logit"]["cum"].append(np.full(J, np.nan))
            estimates["ordered_logit"]["wass"].append(np.nan)
            estimates["ordered_logit"]["beta"].append(np.full(len(cols), np.nan))

        # 4. Structured flow
        try:
            out = evaluate_structured_flow(
                y=y,
                X=X,
                treatment_idx=treatment_idx,
                J=J,
                flow_bins=flow_bins,
                flow_bounds=flow_bounds,
                epochs=epochs,
                lr=lr,
                use_lbfgs=use_lbfgs,
                lbfgs_steps=lbfgs_steps_structured,
                verbose=verbose,
            )
            estimates["structured_flow"]["cat"].append(out["category_effect"])
            estimates["structured_flow"]["cum"].append(out["cum_ge_effect"])
            estimates["structured_flow"]["wass"].append(out["wasserstein_unit"])
            estimates["structured_flow"]["beta"].append(out["beta"])
        except Exception as err:
            failures.append(model_failure_row(setting, n, r, "structured_flow", err))
            estimates["structured_flow"]["cat"].append(np.full(J, np.nan))
            estimates["structured_flow"]["cum"].append(np.full(J, np.nan))
            estimates["structured_flow"]["wass"].append(np.nan)
            estimates["structured_flow"]["beta"].append(np.full(len(cols), np.nan))

        # 5. Model-free conditional flow
        try:
            out = evaluate_model_free_flow(
                y=y,
                X=X,
                treatment_idx=treatment_idx,
                J=J,
                flow_bins=flow_bins,
                flow_bounds=flow_bounds,
                epochs=epochs,
                lr=lr,
                use_lbfgs=use_lbfgs,
                lbfgs_steps=lbfgs_steps_model_free,
                verbose=verbose,
            )
            estimates["model_free_flow"]["cat"].append(out["category_effect"])
            estimates["model_free_flow"]["cum"].append(out["cum_ge_effect"])
            estimates["model_free_flow"]["wass"].append(out["wasserstein_unit"])
        except Exception as err:
            failures.append(model_failure_row(setting, n, r, "model_free_flow", err))
            estimates["model_free_flow"]["cat"].append(np.full(J, np.nan))
            estimates["model_free_flow"]["cum"].append(np.full(J, np.nan))
            estimates["model_free_flow"]["wass"].append(np.nan)

        if (r + 1) % max(1, min(10, R)) == 0:
            print(f"  setting={setting}, n={n}: replication {r + 1}/{R} completed")

    records: List[Dict] = []

    for model, vals in estimates.items():
        append_vector_records(
            records,
            setting=setting,
            n=n,
            model=model,
            metric="category_effect",
            truth=true_cat,
            estimates=vals["cat"],
        )
        append_vector_records(
            records,
            setting=setting,
            n=n,
            model=model,
            metric="cumulative_ge_effect",
            truth=true_cum,
            estimates=vals["cum"],
        )
        append_scalar_record(
            records,
            setting=setting,
            n=n,
            model=model,
            metric="wasserstein_unit",
            truth=true_wass,
            estimates=vals["wass"],
        )

        if vals["beta"]:
            beta_arr = np.asarray(vals["beta"], dtype=float)
            beta_mean = np.nanmean(beta_arr, axis=0)
            beta_sd = np.nanstd(beta_arr, axis=0)
            for k in range(beta_arr.shape[1]):
                records.append({
                    "setting": setting,
                    "n": n,
                    "model": model,
                    "metric": "latent_beta",
                    "index": k,
                    "truth": np.nan,
                    "estimate_mean": float(beta_mean[k]),
                    "bias": np.nan,
                    "rmse": float(beta_sd[k]),
                })

    return records, failures


# ---------------------------------------------------------------------
# File loading and main
# ---------------------------------------------------------------------

def paths_for_setting(args: argparse.Namespace, setting: str) -> Tuple[str, str, Optional[str]]:
    pop_csv = os.path.join(args.data_dir, f"population_{setting}_N{args.N}.csv")
    idx_csv = os.path.join(args.data_dir, f"indices_{setting}_N{args.N}.csv")
    meta_json = os.path.join(args.data_dir, f"metadata_{setting}_N{args.N}.json")
    return pop_csv, idx_csv, meta_json if os.path.exists(meta_json) else None


def run_one_setting(args: argparse.Namespace, setting: str) -> None:
    pop_csv, idx_csv, meta_json = paths_for_setting(args, setting)

    if not os.path.exists(pop_csv):
        raise FileNotFoundError(f"Population file not found: {pop_csv}")
    if not os.path.exists(idx_csv):
        raise FileNotFoundError(f"Index file not found: {idx_csv}")

    df_pop = pd.read_csv(pop_csv)
    df_idx = pd.read_csv(idx_csv)
    truth = truth_from_metadata(meta)

    J = int(args.J if args.J is not None else truth["J"])
    sample_sizes = parse_sample_sizes(args.sample_sizes)


    print(f"Monte Carlo setting={setting}")
    print(f"  population: {pop_csv}")
    print(f"  indices:    {idx_csv}")
    print(f"  metadata:   {meta_json}")
    print(f"  N={len(df_pop)}, J={J}, sample sizes={sample_sizes}, R={args.R}")

    all_records: List[Dict] = []
    all_failures: List[Dict] = []

    for n in sample_sizes:
        records, failures = run_mc_for_n(
            df_pop=df_pop,
            df_idx=df_idx,
            truth = truth,
            setting=setting,
            n=n,
            R=args.R,
            J=J,
            treatment_idx=args.treatment_idx,
            flow_bins=args.flow_bins,
            flow_bounds=args.flow_bounds,
            epochs=args.epochs,
            lr=args.lr,
            use_lbfgs=not args.no_lbfgs,
            lbfgs_steps_structured=args.lbfgs_steps_structured,
            lbfgs_steps_model_free=args.lbfgs_steps_model_free,
            verbose=args.verbose,
        )
        all_records.extend(records)
        all_failures.extend(failures)

    os.makedirs(args.out_dir, exist_ok=True)

    df_res = pd.DataFrame(all_records)
    out_csv = os.path.join(args.out_dir, f"mc_results_{setting}_R{args.R}_N{args.N}.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"Saved results: {out_csv}")

    if all_failures:
        df_fail = pd.DataFrame(all_failures)
        fail_csv = os.path.join(args.out_dir, f"mc_failures_{setting}_R{args.R}_N{args.N}.csv")
        df_fail.to_csv(fail_csv, index=False)
        print(f"Saved failures: {fail_csv}")

    print("Preview:")
    print(df_res.head(12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for ordinal response estimators.")
    parser.add_argument("--setting", default="skewed_lognormal")
    parser.add_argument("--all-settings", action="store_true")
    parser.add_argument("--N", type=int, default=1_000_000)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--sample-sizes", default="500,1000")
    parser.add_argument("--J", type=int, default=None)
    parser.add_argument("--treatment-idx", type=int, default=0)

    parser.add_argument("--data-dir", default="../sim_data")
    parser.add_argument("--out-dir", default="../mc_results")

    parser.add_argument("--flow-bins", type=int, default=12)
    parser.add_argument("--flow-bounds", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-lbfgs", action="store_true")
    parser.add_argument("--lbfgs-steps-structured", type=int, default=50)
    parser.add_argument("--lbfgs-steps-model-free", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    settings = DEFAULT_SETTINGS if args.all_settings else [args.setting]
    for setting in settings:
        run_one_setting(args, setting)


if __name__ == "__main__":
    main()
