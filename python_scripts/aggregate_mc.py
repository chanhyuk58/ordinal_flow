# ====================== aggregate_mc.py ======================

"""
Example:
python aggregate_mc.py --results-dir mc_results --out-dir mc_summary
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "setting",
    "n",
    "rep",
    "model",
    "metric",
    "index",
    "estimate",
    "truth",
}

def find_result_files(results_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    files = [f for f in files if "failures" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir} with pattern {pattern}")
    return files

def read_results(files: List[str]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{f} is missing columns: {sorted(missing)}")
        df["source_file"] = os.path.basename(f)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["n"] = out["n"].astype(int)
    out["rep"] = out["rep"].astype(int)
    out["index"] = out["index"].astype(int)
    out["estimate"] = pd.to_numeric(out["estimate"], errors="coerce")
    out["truth"] = pd.to_numeric(out["truth"], errors="coerce")
    return out

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["setting", "n", "model", "metric", "index"]

    for keys, g in df.groupby(group_cols, dropna=False):
        setting, n, model, metric, index = keys
        est = g["estimate"].to_numpy(dtype=float)
        truth_values = g["truth"].dropna().unique()

        if len(truth_values) == 0:
            truth = np.nan
            err = np.full_like(est, np.nan, dtype=float)
        else:
            truth = float(truth_values[0])
            err = est - truth

        valid = np.isfinite(est)
        valid_err = np.isfinite(err)

        mean_est = float(np.nanmean(est)) if valid.any() else np.nan
        sd_est = float(np.nanstd(est, ddof=1)) if valid.sum() > 1 else np.nan
        mcse_mean = float(sd_est / np.sqrt(valid.sum())) if valid.sum() > 1 else np.nan

        bias = float(np.nanmean(err)) if valid_err.any() else np.nan
        rmse = float(np.sqrt(np.nanmean(err ** 2))) if valid_err.any() else np.nan
        mae = float(np.nanmean(np.abs(err))) if valid_err.any() else np.nan

        rows.append({
            "setting": setting,
            "n": int(n),
            "model": model,
            "metric": metric,
            "index": int(index),
            "truth": truth,
            "estimate_mean": mean_est,
            "estimate_sd": sd_est,
            "bias": bias,
            "abs_bias": abs(bias) if np.isfinite(bias) else np.nan,
            "rmse": rmse,
            "mae": mae,
            "mcse_mean": mcse_mean,
            "n_rep": int(g["rep"].nunique()),
            "n_obs": int(len(g)),
            "n_missing": int((~valid).sum()),
            "source_files": int(g["source_file"].nunique()),
        })

    return pd.DataFrame(rows)

def aggregate_failures(results_dir: str, out_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(results_dir, "mc_failures_*.csv")))
    if not files: return

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            frames.append(df)
        except Exception:
            pass

    if frames:
        fail = pd.concat(frames, ignore_index=True)
        fail.to_csv(os.path.join(out_dir, "mc_failures_all.csv"), index=False)
        print(f"Aggregated {len(fail)} failure records into mc_failures_all.csv")

def print_quick_summary(summary: pd.DataFrame):
    """Prints a pivot table of RMSE and Bias for quick terminal inspection."""
    print("\n--- QUICK SUMMARY (RMSE) ---")
    try:
        # Filter out latent_beta since it doesn't have a ground truth for RMSE
        sub = summary[summary["metric"] != "latent_beta"]
        rmse_pivot = sub.pivot_table(index=["setting", "n", "metric"], columns="model", values="rmse")
        print(rmse_pivot.round(4).to_string())

        print("\n--- QUICK SUMMARY (BIAS) ---")
        bias_pivot = sub.pivot_table(index=["setting", "n", "metric"], columns="model", values="bias")
        print(bias_pivot.round(4).to_string())
    except Exception as e:
        print(f"Could not generate pivot tables: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate chunked Monte Carlo results.")
    parser.add_argument("--results-dir", default="mc_results")
    parser.add_argument("--out-dir", default="mc_summary")
    parser.add_argument("--pattern", default="mc_results_*.csv")
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = find_result_files(args.results_dir, args.pattern)
    print(f"Found {len(files)} result files.")

    raw = read_results(files)
    
    # Check for duplicates before dropping to warn the user
    original_len = len(raw)
    raw = raw.drop_duplicates(subset=["setting", "n", "rep", "model", "metric", "index"], keep="last")
    if len(raw) < original_len:
        print(f"Dropped {original_len - len(raw)} overlapping replication records (kept latest).")

    if args.save_raw:
        raw.to_csv(os.path.join(args.out_dir, "mc_raw_all.csv"), index=False)

    summary = aggregate(raw)
    summary.to_csv(os.path.join(args.out_dir, "mc_summary.csv"), index=False)

    aggregate_failures(args.results_dir, args.out_dir)

    print(f"\nSaved summary to: {os.path.join(args.out_dir, 'mc_summary.csv')}")
    
    # Print the table output
    print_quick_summary(summary)

if __name__ == "__main__":
    main()
