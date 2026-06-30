import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import numpy as np
import pandas as pd
import torch

from ordinal_flow_core import (
    device,
    train_ordered_flow,
    train_model_free_flow,
    fit_ordered_sm,
    ordered_sm_effects,
    empirical_probs_by_treatment,
)

torch.set_default_dtype(torch.float64)

# ============================================================
# Load data
# ============================================================

tomz = pd.read_stata(
    "./2012-10-01-Main-prepped.dta"
)

tomz["f_strike5"] = pd.factorize(
    tomz["strike5"],
    sort=True
)[0] + 1

tomz["hrtsdemoc"] = tomz["hrts"] * tomz["democ"]

y_pd = tomz["f_strike5"]

X_pd = tomz[
    [
        "hrts",
        "democ",
        "h1",
        "i1",
        "p1",
        "e1",
        "r1",
        "male",
        "white",
        "age",
        "ed4",
    ]
]

Z_pd = tomz[
    [
        "h1",
        "i1",
        "p1",
        "e1",
        "r1",
        "male",
        "white",
        "age",
        "ed4",
    ]
]

treat = X_pd["hrts"].to_numpy()

y = torch.tensor(
    y_pd.to_numpy(),
    dtype=torch.long,
    device=device,
)

X = torch.tensor(
    X_pd.to_numpy(),
    dtype=torch.get_default_dtype(),
    device=device,
)

Z = torch.tensor(
    Z_pd.to_numpy(),
    dtype=torch.get_default_dtype(),
    device=device,
)

# ============================================================
# Ordered probit
# ============================================================

beta_op, thr_op, oprobit = fit_ordered_sm(
    y.cpu(),
    X.cpu(),
    link="probit"
)

oprobit_effects = ordered_sm_effects(
    oprobit,
    X.cpu(),
    treatment_idx=0
)

# ============================================================
# Ordered logit
# ============================================================

beta_ol, thr_ol, ologit = fit_ordered_sm(
    y.cpu(),
    X.cpu(),
    link="logit"
)

ologit_effects = ordered_sm_effects(
    ologit,
    X.cpu(),
    treatment_idx=0
)

# ============================================================
# Empirical estimator
# ============================================================

empirical = empirical_probs_by_treatment(
    y.cpu().numpy(),
    treat
)

# ============================================================
# Structured flow
# ============================================================

structured = train_ordered_flow(
    X,
    y,
    Z=Z,
    flow_bins=32,
    bounds=12,
    epochs=1000,
    lr=1e-3,
    use_lbfgs=True,
    init_probit=True,
    verbose=True,
)

structured_ame = structured.compute_ame(
    X,
    treatment_idx=0,
    Z=Z,
)

structured_cum = structured.compute_cumulative_ge_effect(
    X,
    treatment_idx=0,
    Z=Z,
)

structured_wasserstein = structured.compute_wasserstein_unit(
    X,
    treatment_idx=0,
    Z=Z,
)

# ============================================================
# Model-free flow
# ============================================================

model_free = train_model_free_flow(
    X,
    y,
    flow_bins=32,
    bounds=12,
    epochs=1000,
    lr=1e-3,
    use_lbfgs=True,
    verbose=True,
)

model_free_ame = model_free.compute_ame(
    X,
)

model_free_cum = model_free.compute_cumulative_ge_effect(
    X,
)

model_free_wasserstein = model_free.compute_wasserstein_unit(
    X,
)

# ============================================================
# Save outputs
# ============================================================

results = {
    "empirical": empirical,
    "oprobit": oprobit_effects,
    "ologit": ologit_effects,
    "structured": structured_effects,
    "model_free": model_free_effects,
}

for name, obj in results.items():
    print("\n==========================")
    print(name.upper())
    print("==========================")
    print(obj)

print("\nStructured Wasserstein:")
print(structured_wasserstein)

print("\nModel-free Wasserstein:")
print(model_free_wasserstein)
