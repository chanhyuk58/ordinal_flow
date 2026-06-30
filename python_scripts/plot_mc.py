# ============================= plot_mc.py ============================

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_LABELS = {
    "empirical": "Empirical",
    "ordered_probit": "Ordered probit",
    "ordered_logit": "Ordered logit",
    "structured_flow": "Structured flow",
    "model_free_flow": "Model-free flow",
}

MODEL_ORDER = [
    "empirical",
    "ordered_probit",
    "ordered_logit",
    "structured_flow",
    "model_free_flow",
]

METRIC_LABELS = {
    "category_effect": "Category-specific effects",
    "cumulative_ge_effect": "Cumulative effects",
    "wasserstein_unit": "Wasserstein distance",
    "latent_beta": "Latent coefficients",
}


def safe_name(s: str) -> str:
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace(">", "ge")
        .replace("<", "le")
        .replace("=", "")
        .replace(",", "")
        .lower()
    )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["model_order"] = df["model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999
    )
    df = df.sort_values(["setting", "n", "metric", "index", "model_order"])
    return df


def plot_rmse_by_setting(df: pd.DataFrame, out_dir: str, fmt: str) -> None:
    metrics = ["category_effect", "cumulative_ge_effect", "wasserstein_unit"]

    for metric in metrics:
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            continue

        # Average across response categories for vector-valued metrics.
        agg = (
            sub.groupby(["setting", "n", "model", "model_label", "model_order"], as_index=False)
            .agg(rmse=("rmse", "mean"), bias=("bias", "mean"))
            .sort_values(["setting", "n", "model_order"])
        )

        for setting in agg["setting"].unique():
            sdat = agg[agg["setting"] == setting]
            for n in sorted(sdat["n"].unique()):
                ndat = sdat[sdat["n"] == n].sort_values("model_order")

                fig, ax = plt.subplots(figsize=(7.2, 4.2))
                ax.bar(ndat["model_label"], ndat["rmse"])
                ax.set_ylabel("RMSE")
                ax.set_xlabel("")
                ax.set_title(f"{METRIC_LABELS.get(metric, metric)}: {setting}, n={n}")
                ax.tick_params(axis="x", rotation=35)
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()

                fname = f"rmse_{safe_name(metric)}_{safe_name(setting)}_n{n}.{fmt}"
                fig.savefig(os.path.join(out_dir, fname), dpi=300)
                plt.close(fig)


def plot_bias_by_category(df: pd.DataFrame, out_dir: str, fmt: str) -> None:
    metrics = ["category_effect", "cumulative_ge_effect"]

    for metric in metrics:
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            continue

        for setting in sub["setting"].unique():
            sdat = sub[sub["setting"] == setting]
            for n in sorted(sdat["n"].unique()):
                ndat = sdat[sdat["n"] == n].copy()

                fig, ax = plt.subplots(figsize=(7.6, 4.6))

                for model in MODEL_ORDER:
                    mdat = ndat[ndat["model"] == model].sort_values("index")
                    if mdat.empty:
                        continue
                    ax.plot(
                        mdat["index"],
                        mdat["bias"],
                        marker="o",
                        label=MODEL_LABELS.get(model, model),
                    )

                ax.axhline(0, linewidth=1)
                ax.set_xlabel("Response category / threshold")
                ax.set_ylabel("Bias")
                ax.set_title(f"{METRIC_LABELS.get(metric, metric)} bias: {setting}, n={n}")
                ax.legend(frameon=False, fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()

                fname = f"bias_by_index_{safe_name(metric)}_{safe_name(setting)}_n{n}.{fmt}"
                fig.savefig(os.path.join(out_dir, fname), dpi=300)
                plt.close(fig)


def plot_rmse_by_n(df: pd.DataFrame, out_dir: str, fmt: str) -> None:
    metrics = ["category_effect", "cumulative_ge_effect", "wasserstein_unit"]

    for metric in metrics:
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            continue

        agg = (
            sub.groupby(["setting", "n", "model", "model_label", "model_order"], as_index=False)
            .agg(rmse=("rmse", "mean"))
            .sort_values(["setting", "model_order", "n"])
        )

        for setting in agg["setting"].unique():
            sdat = agg[agg["setting"] == setting]

            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for model in MODEL_ORDER:
                mdat = sdat[sdat["model"] == model].sort_values("n")
                if mdat.empty:
                    continue
                ax.plot(
                    mdat["n"],
                    mdat["rmse"],
                    marker="o",
                    label=MODEL_LABELS.get(model, model),
                )

            ax.set_xlabel("Sample size")
            ax.set_ylabel("RMSE")
            ax.set_title(f"{METRIC_LABELS.get(metric, metric)} RMSE: {setting}")
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()

            fname = f"rmse_by_n_{safe_name(metric)}_{safe_name(setting)}.{fmt}"
            fig.savefig(os.path.join(out_dir, fname), dpi=300)
            plt.close(fig)


def write_tables(df: pd.DataFrame, out_dir: str) -> None:
    main_metrics = ["category_effect", "cumulative_ge_effect", "wasserstein_unit"]
    sub = df[df["metric"].isin(main_metrics)].copy()

    table = (
        sub.groupby(["setting", "n", "model"], as_index=False)
        .agg(
            mean_abs_bias=("abs_bias", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            n_rep=("n_rep", "min"),
        )
        .sort_values(["setting", "n", "mean_rmse"])
    )
    table.to_csv(os.path.join(out_dir, "mc_table_model_performance.csv"), index=False)

    best = (
        table.sort_values(["setting", "n", "mean_rmse"])
        .groupby(["setting", "n"], as_index=False)
        .first()
    )
    best.to_csv(os.path.join(out_dir, "mc_table_best_by_setting.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Monte Carlo summary results.")
    parser.add_argument("--summary", default="mc_summary/mc_summary.csv")
    parser.add_argument("--out-dir", default="mc_figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.summary)
    df = prepare(df)

    plot_rmse_by_setting(df, args.out_dir, args.format)
    plot_bias_by_category(df, args.out_dir, args.format)
    plot_rmse_by_n(df, args.out_dir, args.format)
    write_tables(df, args.out_dir)

    print(f"Saved plots and tables to {args.out_dir}")


if __name__ == "__main__":
    main()
