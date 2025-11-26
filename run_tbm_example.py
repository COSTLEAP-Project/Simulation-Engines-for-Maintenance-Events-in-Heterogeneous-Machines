"""
Example script for running the Time-Based Maintenance (TBM) simulation engine.

"""

import numpy as np
import pandas as pd

from time_based.simulation import simulate_all_machines


# -----------------------------
# 1. USER-SUPPLIED INPUTS
# -----------------------------

# (A) Number of machines and observation time
n_machines = 5
t_obs = 5.0
m = 1000
delta_t = t_obs / m

# (B) Preventive maintenance interval for each machine and PM effectiveness parameter ("push")
T_value = 1.0
T_machines = {i: T_value for i in range(1, n_machines + 1)}

# push = 0.0 → PM has no restorative effect (AGAO)
# push = 1.0 → PM restores fully (AGAN)
push = 0.5

# (C) Fixed covariates (user supplies — here we generate a simple example)
def generate_fixed_covs(n_machines, n_covs, probs):
    """
    Generate a fixed covariate matrix of shape (n_machines, n_covs).

    Parameters
    ----------
    n_machines : int
        Number of machines.
    n_covs : int
        Number of fixed covariates.
    probs : list of float
        Bernoulli probabilities for generating each covariate column.

    Returns
    -------
    numpy.ndarray
        A binary covariate matrix where each column j is drawn from
        Bernoulli(probs[j]).
    """
    if len(probs) != n_covs:
        raise ValueError(
            f"Length of probs ({len(probs)}) must equal n_covs ({n_covs})."
        )

    covs = np.zeros((n_machines, n_covs))

    for j in range(n_covs):
        covs[:, j] = np.random.binomial(1, probs[j], n_machines)

    return covs

# Generate fixed covariates (user-defined)
np.random.seed(4)  # for repetition
fixed_covs = generate_fixed_covs(n_machines=5, n_covs=4, probs=[0.5, 0.5, 0.5, 0.5])

# (D) Dynamic covariates (user supplies initial trajectories)
n_dynamic_features = 1
dynamic_covs = np.zeros((m+1, n_dynamic_features))
machines_dynamic_covs = {
    machine_id: dynamic_covs.copy()
    for machine_id in range(1, n_machines + 1)
}

# User-defined dynamic covariate update rule
def example_covariate_update_type3_col0(dynamic_covs, failure_type, valid_indices, machine_id):
    """
    Example covariate update function: Update first column when failure type 3 occurs
    This is the default behavior from the original implementation.
    
    Parameters:
    -----------
    valid_indices : int
        Time index where the failure occurred
    Returns:
    --------
    tuple : (updated_dynamic_covs, was_updated)
        - updated_dynamic_covs: Updated covariate array (copy of input if updated)
        - was_updated: Boolean indicating whether update occurred
    
    Example Usage:
    --------------
    This function updates the first column (index 0) of dynamic_covs from valid_indices onwards,
    setting it to 1, but only if:
    1. The failure type is 3
    2. The first column hasn't been updated yet (all zeros)
    """
    # Check if update should occur
    if failure_type == 3 and sum(dynamic_covs[:, 0]) == 0:
        dynamic_covs[valid_indices+1:, 0]=1
        updated_covs = dynamic_covs.copy()
        return updated_covs, True
    else:
        # No update needed
        return dynamic_covs, False

cov_update_fn = example_covariate_update_type3_col0


# -----------------------------
# 2. Model PARAMETERS
# -----------------------------

# --- Hazard covariate coefficients --
beta_fixed = np.array([-0.2, 0.3, 0.4, -0.1])
beta_dynamic = np.array([0.1])

# --- Minor hazard model ---
include_minor = True
model_type_minor = "weibull"  # CHANGED
shape_minor = 2.0             # CHANGED: >1 ⇒ hazard(0)=0
scale_minor = 2.5             # CHANGED: tune to get realistic times
intercept_minor = None        # not used for Weibull


# --- Catastrophic hazard model ---
include_catas = True
model_type_catas = "weibull"
shape_catas = 2.0  # k_c
scale_catas = 5   # alpha_c (corresponds to alpha_c = 0.2 in EJOR paper)
intercept_catas = None  # Not for Weibull
with_covariates_catas = False

# --- Minor type multinomial (n-1 columns → n=3 minor types) ---
n_minor_types = 3
beta_multinom_fixed = np.array([
    [0.9, 0.9],   
    [0.4, 0.5],   
    [0.1, 0.0],   
    [0.0, 0.2]    
])
beta_multinom_dynamic = np.array([
    [0.1, 0.2]  
])


# -----------------------------
# 3. COST PARAMETERS
# -----------------------------

use_covariates = False

# Catastrophic (target mean ~150)
loc_fixed_cat   = 50.0
shape_cat       = 8.0
scale_cat       = 8

gamma_coeffs_cat_fixed   = np.array([ 4.0, -2.0,  1.5,  1.0])
gamma_coeffs_cat_dynamic = np.array([ 3.0])

# SD_cat = sqrt(50)*2 ≈ 14.14

# Minor type base costs (types 1 & 2 explicitly; type 3 will be composite)
loc_fixed_minor1 = 30.0; shape_minor1 = 6.0; scale_minor1 = 5.0
loc_fixed_minor2 = 20.0; shape_minor2 = 5.0; scale_minor2 = 5.0

# Build lists of minor-type params (length = n_minor_types = 3)
shape_minor_list     = [shape_minor1,     shape_minor2,     4.0]  # type 3 values won't be used if composite
scale_minor_list     = [scale_minor1,     scale_minor2,     6.0]
loc_fixed_minor_list = [loc_fixed_minor1, loc_fixed_minor2, 25.0]

# Minor-type cost coefficients (fixed + dynamic) — lists of length 3
gamma_coeffs_minor1_fixed   = np.array([ 2.0,  0.5, -1.0,  0.4])
gamma_coeffs_minor1_dynamic = np.array([ 2.5])
gamma_coeffs_minor2_fixed   = np.array([ 1.5, -0.8,  0.8,  0.3])
gamma_coeffs_minor2_dynamic = np.array([ 1.5])

# For type 3 (composite), these entries won't be used, but lists must have length 3:
gamma_coeffs_minor3_fixed   = np.array([ 0.0,  0.0,  0.0,  0.0])
gamma_coeffs_minor3_dynamic = np.array([ 0.0])

gamma_coeffs_minor_fixed_list   = [gamma_coeffs_minor1_fixed,   gamma_coeffs_minor2_fixed,   gamma_coeffs_minor3_fixed]
gamma_coeffs_minor_dynamic_list = [gamma_coeffs_minor1_dynamic, gamma_coeffs_minor2_dynamic, gamma_coeffs_minor3_dynamic]

# --- Composite minor type: define 3 as simultaneous (1, 2) ---
minor_combo_map = {3: (1, 2)}


# Frank copula theta policy
theta_copula = {3: 2.0}

# --- PM cost parameters ---
gamma_coeffs_pm_fixed   = np.array([0.2, 0.1, -0.1, 0.3])
gamma_coeffs_pm_dynamic = np.array([0.05])
shape_pm = 2.0
scale_pm = 10.0
loc_fixed_pm = 30.0



# -----------------------------
# 4. RUN SIMULATION
# -----------------------------


results_df, all_machines_dynamic_covs = simulate_all_machines(n_machines, t_obs, m, n_dynamic_features, delta_t, T_machines, push,
                     include_minor, model_type_minor, shape_minor, scale_minor, intercept_minor, with_covariates_minor,
                     include_catas, model_type_catas, shape_catas, scale_catas, intercept_catas, with_covariates_catas,
                     fixed_covs, machines_dynamic_covs, beta_fixed, beta_dynamic, beta_multinom_fixed, beta_multinom_dynamic,
                     n_minor_types, cov_update_fn,
                     # cost-related (lists for minor types)
                     gamma_coeffs_cat_fixed, gamma_coeffs_cat_dynamic,
                     gamma_coeffs_minor_fixed_list, gamma_coeffs_minor_dynamic_list,
                     theta_copula, 
                     shape_cat, scale_cat, loc_fixed_cat,
                     shape_minor_list, scale_minor_list, loc_fixed_minor_list,
                     use_covariates, minor_combo_map,
                     # PM cost
                     gamma_coeffs_pm_fixed, gamma_coeffs_pm_dynamic, shape_pm, scale_pm, loc_fixed_pm)



print("\n--- Simulation completed ---")
#print(results_df.head())

