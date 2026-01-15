"""
Main simulation orchestration module.

This module contains the high-level simulation functions:
- simulate_single_cycle: Simulate from AGAN state to failure or observation end
- simulation_complete_observed_period: Handle catastrophic resets
- simulate_machine_full_observed_period: Complete machine simulation with costs
- simulate_all_machines: Multi-machine simulation
"""

import numpy as np
import pandas as pd
from .failure_time import getFailureTime
from .failure_type import get_failure_type, get_minor_failure_type
from .failure_cost import simulate_failure_costs, simulate_periodical_pm


# =========================================
# Simulation failures for Single Cycle: From AGAN State to Failure or Observation End 
# =========================================
def simulate_single_cycle(t_obs, machine_id, m, delta_t,
                          include_minor, T, push,
                          model_type_minor, shape_minor, scale_minor, intercept_minor,
                          with_covariates_minor, fixed_covs, dynamic_covs, beta_fixed, beta_dynamic,
                          include_catas, model_type_catas, shape_catas, scale_catas, intercept_catas,
                          with_covariates_catas, beta_multinom_fixed, beta_multinom_dynamic,
                          n_minor_types, cov_update_fn=None):
    """
    Simulate a single cycle from AGAN state until either catastrophic failure or observation end (which comes first)
    
    Parameters:
    -----------
    t_obs : float
        Observation time
    cov_update_fn : callable or None
        Custom function to update dynamic covariates.
        Function signature: cov_update_fn(dynamic_covs, failure_type, valid_indices, machine_id)
        Should return: (updated_dynamic_covs, was_updated)
        If None, no covariate update is performed.
    
    Returns:
    --------
    tuple : (failure_index, failure_time, risk_type, failure_type, censor_status, updated_dynamic_covs)
        - failure_index: list of failure indices
        - failure_time: list of failure times
        - risk_type: list of risk types (1=minor, 2=catastrophic)
        - failure_type: list of specific failure types (minor failure 1,2,...)
        - censor_status: list of censoring status (1=observed, 0=censored)
        - updated_dynamic_covs: updated dynamic covariates array
    """
    failure_index = []
    failure_time = []
    risk_type = []
    failure_type = []
    censor_status = []  # Censoring status: 1=observed failure, 0=censored

    s = 0
    cumulative_integrals_minor = []
    cumulative_integrals_catas = []
    cumulative_integrals = []
    cumulative_value = 0
    s_record = [0]

    valid_indices = -1  # Initial failure index
    ft = -1  # Initial failure time
    f_type = -1  # Initial minor failure type
    dynamic_covs_changed = False  # Flag for dynamic covariate change

    # Continue until failure time exceeds observation time
    while ft < t_obs:
        ###### Get failure time
        result = getFailureTime(s, cumulative_integrals_minor, cumulative_integrals_catas, cumulative_integrals,
                    dynamic_covs_changed, machine_id, valid_indices, m, delta_t,
                    # Minor failure parameters
                    include_minor=include_minor,
                    model_type_minor=model_type_minor, shape_minor=shape_minor, scale_minor=scale_minor, intercept_minor=intercept_minor,
                    fixed_covs=fixed_covs, dynamic_covs=dynamic_covs, beta_fixed=beta_fixed, beta_dynamic=beta_dynamic,
                    with_covariates_minor=with_covariates_minor, T=T, push=push,
                    # Catastrophic failure parameters
                    include_catas=include_catas,
                    model_type_catas=model_type_catas, shape_catas=shape_catas, scale_catas=scale_catas, intercept_catas=intercept_catas,
                    with_covariates_catas=with_covariates_catas)

        valid_indices, ft, s, cumulative_integrals_minor, cumulative_integrals_catas, cumulative_integrals, is_censored = result

        updated_dynamic_covs = dynamic_covs.copy() if dynamic_covs is not None else None  # Default: return initial values if not updated
        
        if is_censored:
            # Censored case: record censoring event and stop
            failure_index.append(m)  # the last time point
            failure_time.append(t_obs)  # Record observation end time
            risk_type.append(0)  # Censored event
            failure_type.append(0)  # Censored event
            censor_status.append(0)  # 0=censored
            break

        # Check if failure time exceeds observation time
        if ft >= t_obs:
            # No failure before observation end, record as censored
            failure_index.append(m)
            failure_time.append(t_obs)
            risk_type.append(0)
            failure_type.append(0)
            censor_status.append(0)
            break

        # Normal failure case
        failure_index.append(valid_indices)
        failure_time.append(ft)
        censor_status.append(1)  # 1=observed failure

       ##### Determine failure type
        dynamic_cov_t = dynamic_covs[valid_indices] if (dynamic_covs is not None) else None
        # If only one failure type is included, skip competing risk calculation
        if include_minor and not include_catas:
            # Only minor failures possible
            type_risk = 1
            p_minor = 1.0
        elif include_catas and not include_minor:
            # Only catastrophic failures possible
            type_risk = 2
            p_minor = 0.0
        else:
            # Both types included - use competing risks
            p_minor, type_risk = get_failure_type(
                machine_id, ft, T, push, scale_minor, intercept_minor, shape_minor,
                with_covariates_minor, model_type_minor,
                fixed_covs, dynamic_cov_t, beta_fixed, beta_dynamic,
                scale_catas, shape_catas, intercept_catas, model_type_catas, with_covariates_catas
            )
        
        risk_type.append(type_risk)
        
        # Continue even if catastrophic failure, as observation time is fixed
        if type_risk == 2:
            failure_type.append(-1)  # Ensure list lengths match
            break
        else:
            f_type = get_minor_failure_type(machine_id, beta_multinom_fixed, beta_multinom_dynamic, 
                                           fixed_covs, dynamic_cov_t, n_minor_types)
            failure_type.append(f_type)
            
            # Apply custom covariate update function if provided
            if cov_update_fn is not None and dynamic_covs is not None:
                updated_dynamic_covs, was_updated = cov_update_fn(
                    dynamic_covs, f_type, valid_indices, machine_id
                )
                if was_updated:
                    dynamic_covs = updated_dynamic_covs.copy()
                    dynamic_covs_changed = True

    return failure_index, failure_time, risk_type, failure_type, censor_status, updated_dynamic_covs

# =========================================
# Simulation failures for Complete Observation Period 
# =========================================
def simulation_complete_observed_period(
        t_obs, machine_id, m, n_dynamic_features, delta_t, T, push,
        include_minor, model_type_minor, shape_minor, scale_minor, intercept_minor,
        with_covariates_minor, include_catas, model_type_catas, shape_catas, scale_catas, intercept_catas,
        with_covariates_catas, fixed_covs, dynamic_covs, beta_fixed, beta_dynamic,
        beta_multinom_fixed, beta_multinom_dynamic, n_minor_types, cov_update_fn=None):
    """
    Complete observation period simulation, handling resets after catastrophic failures
    Supports both minor and catastrophic failures with optional covariates
    
    Returns:
    --------
    tuple : Complete simulation results including failure indices, times, types, and covariate trajectories
    """
    # Store results from all cycles
    all_failure_index = []
    all_real_failure_time = []
    all_abs_failure_time = []
    all_risk_type = []
    all_failure_type = []
    all_censor_status = []

    # allow missing dynamic covariates
    if dynamic_covs is not None:
        n_dynamic_features = dynamic_covs.shape[1]
        all_dynamic_covs = np.empty((0, n_dynamic_features), dtype=float)
        initial_dynamic_covs = dynamic_covs.copy()
    else:
        n_dynamic_features = 0
        all_dynamic_covs = None              # keep None to skip concatenations safely
        initial_dynamic_covs = None

    #updated_dynamic_covs = None

    t_obs_remain = t_obs
    current_time_offset = 0.0
    absolute_index_offset = 0

    while t_obs_remain > 0:
        # Current number of intervals
        current_m = (len(dynamic_covs) - 1) if (dynamic_covs is not None) else m

        # Run single cycle simulation
        failure_index, failure_time, risk_type, failure_type, censor_status, updated_dynamic_covs = simulate_single_cycle(
            t_obs_remain, machine_id, current_m, delta_t,
            include_minor, T, push,
            model_type_minor, shape_minor, scale_minor, intercept_minor,
            with_covariates_minor, fixed_covs, dynamic_covs, beta_fixed, beta_dynamic,
            include_catas, model_type_catas, shape_catas, scale_catas, intercept_catas,
            with_covariates_catas, beta_multinom_fixed, beta_multinom_dynamic,
            n_minor_types, cov_update_fn)

        # Adjust failure times and indices to absolute values
        abs_failure_time = [ft + current_time_offset if ft > 0 else ft for ft in failure_time]
        abs_failure_index = [fi + absolute_index_offset if fi > 0 else fi for fi in failure_index]

        # Save results
        all_failure_index.extend(abs_failure_index)
        all_real_failure_time.extend(failure_time)  #failure time in its cycle (restes if a catatrophic failure occurs)
        all_abs_failure_time.extend(abs_failure_time) #absolute failure time in the observation horizon
        all_risk_type.extend(risk_type)
        all_failure_type.extend(failure_type)
        all_censor_status.extend(censor_status)

        # Check for catastrophic failure
        if (len(failure_type) > 0 and failure_type[-1] == -1 and
            len(failure_time) > 0 and failure_time[-1] < t_obs_remain):
            catastrophic_time = all_abs_failure_time[-1]

            # Update remaining observation time and time offset
            t_obs_remain = t_obs - catastrophic_time
            current_time_offset = catastrophic_time

            # Update absolute index offset
            absolute_index_offset = abs_failure_index[-1]

            if initial_dynamic_covs is not None:
                # Reset dynamic_covs to AGAN state for remaining time
                dynamic_covs = initial_dynamic_covs[absolute_index_offset:].copy()
            if all_dynamic_covs is not None and updated_dynamic_covs is not None:
                # Truncate all_dynamic_covs: keep only pre-catastrophic portion
                k = int(failure_index[-1])
                all_dynamic_covs = np.concatenate((all_dynamic_covs, updated_dynamic_covs[:k]), axis=0)
        else:
            if all_dynamic_covs is not None and updated_dynamic_covs is not None:
                all_dynamic_covs = np.concatenate((all_dynamic_covs, updated_dynamic_covs), axis=0)
            break

    return (all_failure_index, all_real_failure_time, all_abs_failure_time,
            all_risk_type, all_failure_type, all_censor_status, all_dynamic_covs)


# =========================================
# Simulation of maintenances (failure_PM) and costs for Complete Observation Period 
# =========================================
def simulate_machine_full_observed_period(
    t_obs, machine_id, m, n_dynamic_features, delta_t, T, push,
    include_minor, model_type_minor, shape_minor, scale_minor, intercept_minor, with_covariates_minor,
    include_catas, model_type_catas, shape_catas, scale_catas, intercept_catas, with_covariates_catas,
    fixed_covs, dynamic_covs, beta_fixed, beta_dynamic, beta_multinom_fixed, beta_multinom_dynamic,
    n_minor_types, cov_update_fn=None,
    # cost-related parameters
    gamma_coeffs_cat_fixed=None, gamma_coeffs_cat_dynamic=None,
    gamma_coeffs_minor_fixed_list=None, gamma_coeffs_minor_dynamic_list=None,
    theta_copula=None, 
    shape_cat=2.0, scale_cat=1.0, loc_fixed_cat=0.0,
    shape_minor_list=None,  scale_minor_list=None, loc_fixed_minor_list=None,
    use_covariates=True, minor_combo_map=None,
    # pm cost
    gamma_coeffs_pm_fixed=None, gamma_coeffs_pm_dynamic=None, shape_pm=2.0, scale_pm=1.0, loc_fixed_pm=0.0
):

    """
    Simulate a single machine over the entire observation period and generate cost information
    for each failure event. Supports flexible n minor failure types.
    
    Parameters:
    -----------
    n_minor_types : int
        Number of minor failure types (e.g., 3 means types 1, 2, and 3)
    cov_update_fn : callable or None
        Custom function to update dynamic covariates based on failure type.
        Function signature: covariate_update_function(dynamic_covs, failure_type, valid_indices, machine_id)
        Should return: (updated_dynamic_covs, was_updated)
        If None, no covariate update is performed.
    gamma_coeffs_minor_fixed_list : list of array-like
        List of fixed covariate coefficients for each minor type
    gamma_coeffs_minor_dynamic_list : list of array-like
        List of dynamic covariate coefficients for each minor type
    shape_minor_list, scale_minor_list, loc_fixed_minor_list : list of float
        Lists of Gamma parameters for each minor type
    
    Returns:
    --------
    tuple : (df_failures, all_dynamic_covs, pm_index, pm_times, pm_costs)
        - df_failures: DataFrame containing all failure events with columns:
            * event_index: discrete time index (0-based)
            * abs_event_time: absolute failure time considering catastrophic resets
            * risk_type: risk type encoding (1=minor, 2=catastrophic, 0=censored)
            * event_type: specific failure type (-1=catastrophic, 0=censored, 1 to J=minor types)
            * censor_status: 1=observed failure, 0=censored
            * event_cost: maintenance cost for each event
        - all_dynamic_covs: dynamic covariate trajectory
        - pm_index, pm_times, pm_costs: PM-related information
    """

    # 1. Run failure simulation
    all_failure_index, all_real_failure_time, all_abs_failure_time, all_risk_type, all_failure_type, all_censor_status, all_dynamic_covs = \
        simulation_complete_observed_period(
            t_obs, machine_id, m, n_dynamic_features, delta_t, T, push,
            include_minor, model_type_minor, shape_minor, scale_minor, intercept_minor, with_covariates_minor,
            include_catas, model_type_catas, shape_catas, scale_catas, intercept_catas, with_covariates_catas,
            fixed_covs, dynamic_covs, beta_fixed, beta_dynamic, beta_multinom_fixed, beta_multinom_dynamic, n_minor_types, cov_update_fn
        )
    
    # 2. Generate costs for each failure  (use final dynamic covariate trajectory)
    failure_costs = simulate_failure_costs(
        machine_id, all_failure_index, all_abs_failure_time, all_failure_type,
        fixed_covs, all_dynamic_covs, n_minor_types,
        gamma_coeffs_cat_fixed, gamma_coeffs_cat_dynamic,
        gamma_coeffs_minor_fixed_list,gamma_coeffs_minor_dynamic_list,
        theta_copula,
        shape_cat, scale_cat, loc_fixed_cat,
        shape_minor_list, scale_minor_list, loc_fixed_minor_list,
        use_covariates=use_covariates, minor_combo_map=minor_combo_map
    )
    # Save to dictionary
    data_dict = {
        'event_index': all_failure_index,
        #'event_time': all_real_failure_time,
        'abs_event_time': all_abs_failure_time,
        'risk_type': all_risk_type,
        'event_type': all_failure_type,
        'censor_status': all_censor_status,
        'event_cost': failure_costs
    }

    # Construct DataFrame
    df_failures = pd.DataFrame(data_dict)

    # 3. Generate PM times and costs (only if T <= t_obs and use final dynamic covariate trajectory)
    if T > t_obs:
        # No PM events occur within observation period
        pm_index = None
        pm_times = None
        pm_costs = None
    else:
        # PM events occur - generate them
        pm_index, pm_times, pm_costs = simulate_periodical_pm(t_obs, T, delta_t, machine_id, fixed_covs, all_dynamic_covs,
                           gamma_coeffs_pm_fixed, gamma_coeffs_pm_dynamic,
                           shape_pm, scale_pm, loc_fixed_pm, use_covariates)

    return df_failures, all_dynamic_covs, pm_index, pm_times, pm_costs

# =========================================
# Simulation of multiple machines 
# =========================================
def simulate_all_machines(n_machines, t_obs, m, n_dynamic_features, delta_t, T_machines, push,
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
                         gamma_coeffs_pm_fixed, gamma_coeffs_pm_dynamic, shape_pm, scale_pm, loc_fixed_pm):

    """
    Simulate all machines and return a combined DataFrame plus the final dynamic-covariate
    trajectories per machine.

    Notes
    -----
    - `machines_dynamic_covs` must be a dict {machine_id: dynamic_covs_init}, where each
      value is a (m+1, n_dynamic_features) ndarray.
    - This function calls `simulate_machine_full_observed_period` for each machine and
      uses its returned `all_dynamic_covs` as the source of truth for
      event-time dynamic covariates.
    """
    all_results = []
    all_machines_dynamic_covs = {}

    for machine_id in range(1, n_machines + 1):
        T = T_machines[machine_id]
        # Get the per-machine initial dynamic covariates
        dynamic_covs = machines_dynamic_covs.get(machine_id) if machines_dynamic_covs else None  #initial dynamic covs

        # Run single-machine simulation
        df_failures, machine_dynamic_covs, pm_index, pm_times, pm_costs = simulate_machine_full_observed_period(
            t_obs, machine_id, m, n_dynamic_features, delta_t, T, push,
            include_minor, model_type_minor, shape_minor, scale_minor, intercept_minor, with_covariates_minor,
            include_catas, model_type_catas, shape_catas, scale_catas, intercept_catas, with_covariates_catas,
            fixed_covs, dynamic_covs, beta_fixed, beta_dynamic, beta_multinom_fixed, beta_multinom_dynamic,
            n_minor_types, cov_update_fn,
            gamma_coeffs_cat_fixed, gamma_coeffs_cat_dynamic,
            gamma_coeffs_minor_fixed_list, gamma_coeffs_minor_dynamic_list,
            theta_copula, 
            shape_cat, scale_cat, loc_fixed_cat,
            shape_minor_list,  scale_minor_list, loc_fixed_minor_list,
            use_covariates, minor_combo_map,
            # pm cost
            gamma_coeffs_pm_fixed, gamma_coeffs_pm_dynamic, shape_pm, scale_pm, loc_fixed_pm
        )

        # Keep the final dynamic covariates for this machine
        if machine_dynamic_covs is not None:
            all_machines_dynamic_covs[machine_id] = machine_dynamic_covs

        # Build PM DataFrame (only if PM events actually occurred)
        if pm_times is not None and len(pm_times) > 0:
            df_pm = pd.DataFrame({
                "event_index": pm_index,           # index at which PM happens
                "abs_event_time": pm_times,        # absolute time of PM
                "risk_type": -2,                   # tag for PM
                "event_type": -2,                  # tag for PM
                "censor_status": 1,                # PM is an observed event
                "event_cost": pm_costs
            })
            # If the last PM lands exactly at t_obs, mark as censored=0 to avoid duplicate end marker
            if df_pm["abs_event_time"].iloc[-1] == t_obs:
                df_pm.loc[df_pm.index[-1], "censor_status"] = 0
            # Combine failures + PM, sort by absolute time
            df_machine = pd.concat([df_failures, df_pm], ignore_index=True)
            df_machine = df_machine.sort_values(by="abs_event_time").reset_index(drop=True)
        else:
            # No PM events - just use failures DataFrame directly
            df_machine = df_failures.copy()

        # Drop duplicate terminal censored line: (t_obs, cost=0)
        mask = (
            (df_machine["abs_event_time"] == t_obs) &
            (df_machine["censor_status"] == 0) &
            (df_machine["event_cost"] == 0)
        )
        df_machine = df_machine[~mask]

        # Add machine id and fixed covariates
        df_machine['machine_id'] = machine_id

        if fixed_covs is not None:
            for i in range(fixed_covs.shape[1]):
                df_machine[f'fixed_cov_{i+1}'] = fixed_covs[machine_id - 1, i]

        # Add dynamic covariates at event time
        if machine_dynamic_covs is not None and n_dynamic_features > 0:
            for i in range(n_dynamic_features):
                dynamic_cov_at_failure = []
                for idx, failure_idx in enumerate(df_machine['event_index']):
                    # FIXED: Ensure failure_idx is valid integer and within bounds
                    if pd.notna(failure_idx) and failure_idx >= 0:
                        failure_idx_int = int(failure_idx)  # Convert to integer
                        # Check if index is within bounds
                        if failure_idx_int < len(machine_dynamic_covs):
                            dynamic_cov_value = machine_dynamic_covs[failure_idx_int, i]
                        else:
                            # Index out of bounds - use last available value or NaN
                            dynamic_cov_value = np.nan
                    else:  # Invalid index (NaN, negative, or PM/censored)
                        dynamic_cov_value = np.nan
                    dynamic_cov_at_failure.append(dynamic_cov_value)
    
                df_machine[f'dynamic_cov_{i+1}_at_failure'] = dynamic_cov_at_failure

        all_results.append(df_machine)

    # Aggregate all machines
    final_df = pd.concat(all_results, ignore_index=True)

    # Put machine_id first
    cols = ['machine_id'] + [c for c in final_df.columns if c != 'machine_id']
    final_df = final_df[cols]

    # Build event-type label map once (numeric -> string)
    event_type_label_map = {
        -2: "PM",
        -1: "Catastrophic",
         0: "Censored",
    }
    combo_map = minor_combo_map or {}
    for k in range(1, n_minor_types + 1):
        if k in combo_map:
            a, b = combo_map[k]
            event_type_label_map[k] = f"Minor_{k} (combo {a}&{b})"
        else:
            event_type_label_map[k] = f"Minor_{k}"
    
    # Map on final_df
    final_df["event_type_label"] = final_df["event_type"].map(event_type_label_map).fillna("Unknown")
    # delete unused cols
    final_df = final_df.drop(columns=["event_index", "risk_type"])


    return final_df, all_machines_dynamic_covs
