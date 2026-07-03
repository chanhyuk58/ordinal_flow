import os
# Prevent OpenMP runtime crashes and thread contention
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch

from ordinal_flow_core import (
    device,
    StructuredOrdinalFlow,
    ModelFreeOrdinalFlow,
    fit_ordered_sm_baseline,
)

torch.set_default_dtype(torch.float64)

# ============================================================
# Load data
# ============================================================

tomz = pd.read_stata("../replication_data/2012-10-01-Main-prepped.dta")

tomz["f_strike5"] = pd.factorize(tomz["strike5"], sort=True)[0] + 1
tomz["hrtsdemoc"] = tomz["hrts"] * tomz["democ"]

y_pd = tomz["f_strike5"]
X_pd = tomz[["hrts", "democ", "h1", "i1", "p1", "e1", "r1", "male", "white", "age", "ed4"]]
Z_pd = tomz[["h1", "i1", "p1", "e1", "r1", "male", "white", "age", "ed4"]]

# Extract Data as Numpy Arrays (the Estimator wrappers handle Torch conversions internally)
y = y_pd.to_numpy(dtype=int)
X = X_pd.to_numpy(dtype=float)
Z = Z_pd.to_numpy(dtype=float)
treat = X_pd["hrts"].to_numpy(dtype=int)

J = len(np.unique(y))
treatment_idx = 0  # 'hrts' is the 0-th column in X_pd

# ============================================================
# Empirical estimator
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

empirical_effects = get_empirical_effects(y, treat, J)

# ============================================================
# Ordered Probit & Logit
# ============================================================
print("\nFitting Ordered Probit...")
oprobit_effects, _ = fit_ordered_sm_baseline(X, y, treatment_idx=treatment_idx, link="probit")

print("Fitting Ordered Logit...")
ologit_effects, _ = fit_ordered_sm_baseline(X, y, treatment_idx=treatment_idx, link="logit")

# ============================================================
# Structured flow
# ============================================================
print("\nFitting Structured Flow...")
structured_model = StructuredOrdinalFlow(J=J, q=Z.shape[1], flow_bins=32, bounds=12.0)
structured_model.fit(X, y, Z=Z, epochs=300, lr=5e-3, use_lbfgs=True, verbose=True)

structured_effects = structured_model.compute_effects(X, treatment_idx=treatment_idx, Z=Z)

# ============================================================
# Model-free flow
# ============================================================
print("\nFitting Model-Free Flow...")
model_free_model = ModelFreeOrdinalFlow(J=J, flow_bins=32, bounds=12.0)
model_free_model.fit(X, y, epochs=300, lr=1e-2, use_lbfgs=True, verbose=True)

model_free_effects = model_free_model.compute_effects(X, treatment_idx=treatment_idx)

# ============================================================
# Save outputs
# ============================================================

results = {
    "empirical": empirical_effects,
    "oprobit": oprobit_effects,
    "ologit": ologit_effects,
    "structured flow": structured_effects,
    "model_free flow": model_free_effects,
}

for name, obj in results.items():
    print(f"\n==========================")
    print(f"{name.upper()}")
    print(f"==========================")
    print(f"Wasserstein Unit : {obj['wasserstein_unit']:.5f}")
    print(f"Category Effects : {np.round(obj['category_effect'], 5)}")
    print(f"Cumulative GE    : {np.round(obj['cum_ge_effect'], 5)}")
