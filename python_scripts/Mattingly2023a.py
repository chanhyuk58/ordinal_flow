# ========================= Mattingly2023a.py =========================

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
    OrderedFlowModel,
    train_flow_model,
)

torch.set_default_dtype(torch.float64)

# ============================================================
# Load & Prepare Data
# ============================================================
# Check if path exists, fallback to local directory if run from different path
df = pd.read_csv("../replication_data/Mattingly2023a.csv")

# Recode variables
df["f_political_model"] = pd.factorize(df["political_model"], sort=True)[0] + 1
df["f_econ_model"] = pd.factorize(df["econ_model"], sort=True)[0] + 1
df["f_world_leader"] = pd.factorize(df["world_leader"], sort=True)[0] + 1

df = pd.concat([df, pd.get_dummies(df.treatment, prefix="treatment")], axis=1)
df = pd.concat([df, pd.get_dummies(df.country, prefix="country")], axis=1)
df = pd.concat([df, pd.get_dummies(df.gender, prefix="gender")], axis=1)

cov_list = (["age", "education", "national_pride", "leftright"] +
            [col for col in df if col.startswith("treatment")] + 
            [col for col in df if col.startswith("country")] + 
            [col for col in df if col.startswith("gender")]
            )
cov_list2 = (["age", "education", "national_pride", "leftright"] +
            [col for col in df if col.startswith("country")] + 
            [col for col in df if col.startswith("gender")]
            )

df_cleaned = df.loc[:, ["f_political_model"] + cov_list].dropna()

# Select columns for X and y
y_pd = df_cleaned["f_political_model"]
X_pd = df_cleaned[cov_list].drop(["treatment", "treatment_Control", "country", "country_3", "gender", "gender_0"], axis=1)
Z_pd = df_cleaned[cov_list2].drop(["country", "country_3", "gender", "gender_0"], axis=1)

y = y_pd.to_numpy(dtype=int)
X = X_pd.to_numpy(dtype=float)
Z = Z_pd.to_numpy(dtype=float)

J = len(np.unique(y))
p = X.shape[1]
q = Z.shape[1]
coef_names = X_pd.columns

# ============================================================
# Fit Main Structured Flow Model
# ============================================================
print("Fitting main Structured Flow model...")
main_estimator = StructuredOrdinalFlow(J=J, q=q, flow_bins=32, bounds=12.0)
main_estimator.fit(X, y, Z=Z, epochs=300, lr=5e-3, use_lbfgs=True, verbose=True)

coef_vec = main_estimator.model.beta.detach().cpu().numpy()

# Save main model state to CPU to safely share with child processes (prevents CUDA multiprocessing crash)
main_model_state_cpu = {k: v.cpu() for k, v in main_estimator.model.state_dict().items()}

# ============================================================
# Single Parallel Bootstrap Task (Warm Started)
# ============================================================
def run_one_bootstrap(seed, main_state_cpu, p_dim, q_dim, num_classes):
    """Refits the model on a stratified bootstrap sample using a warm start."""
    # Re-import PyTorch inside child process to handle isolated devices safely
    import torch
    from ordinal_flow_core import device, OrderedFlowModel, train_flow_model
    
    rng = np.random.default_rng(seed)
    n = len(y)
    
    # Stratified bootstrap sampling
    idx = np.empty(n, dtype=int)
    for c in np.unique(y):
        c_indices = np.where(y == c)[0]
        idx[y == c] = rng.choice(c_indices, size=len(c_indices), replace=True)
        
    Xb, yb, Zb = X[idx], y[idx], Z[idx]
    
    # Instantiate child model on local device
    model_b = OrderedFlowModel(p=p_dim, J=num_classes, q=q_dim, flow_bins=32, bounds=12.0).to(device)
    
    # Safely load warm weights onto the correct device
    model_b.load_state_dict({k: v.to(device) for k, v in main_state_cpu.items()})
    
    # Convert data for PyTorch
    Xb_t = torch.as_tensor(Xb, dtype=torch.float64, device=device)
    yb_t = torch.as_tensor(yb, dtype=torch.float64, device=device)
    Zb_t = torch.as_tensor(Zb, dtype=torch.float64, device=device)
    
    # Fine-tune warm-started model for 30 epochs
    train_flow_model(model_b, Xb_t, yb_t, Z=Zb_t, epochs=30, lr=1e-3, use_lbfgs=True, verbose=False)
    
    return model_b.beta.detach().cpu().numpy()

# ============================================================
# Execute Parallel Bootstrapping
# ============================================================
B = 100
print(f"\nRunning B={B} Stratified Bootstraps for beta standard errors...")
max_workers = os.cpu_count()
print(f"Parallelizing across {max_workers} CPU cores.")

seeds = np.random.SeedSequence(54321).generate_state(B)

boot_betas = []
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_one_bootstrap, seed, main_model_state_cpu, p, q, J) for seed in seeds]
    for b, fut in enumerate(futures):
        boot_betas.append(fut.result())
        if (b + 1) % 10 == 0:
            print(f"  Completed {b + 1}/{B} replications")

# Compute standard deviation of bootstrap estimates
se_boot = np.std(boot_betas, axis=0, ddof=1)

# ============================================================
# Save Summary
# ============================================================
summary = pd.DataFrame({
    "param": coef_names,
    "coef": coef_vec,
    "se_boot": se_boot
})

print("\n", summary.to_string(index=False))

out_dir = "../data" if os.path.exists("../data") else "."
out_csv = os.path.join(out_dir, "Mattingly2023a_econ_results_full.csv")
summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved final results to: {out_csv}")
