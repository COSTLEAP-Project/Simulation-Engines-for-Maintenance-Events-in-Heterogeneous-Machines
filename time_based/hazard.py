"""
Hazard function implementations for failure intensity modeling.

This module contains:
- lambda_f: Unified hazard function for both minor and catastrophic failures
- integrated_lambda_closed: Closed-form cumulative hazard
- integrated_lambda_numeric: Numerical integration of hazard
- compute_cumulative_integrals: Update cumulative integrals
"""

import numpy as np
from scipy.integrate import quad


def lambda_f(
    machine_id, t, pm_affects=True, T=None, push=0.0, scale=None, intercept=None, shape=None,
    fixed_covs=None, dynamic_cov_t=None, beta_fixed=None, beta_dynamic=None, 
    model_type="weibull", with_covariates=True
):
    """
    Unified hazard function for both minor and catastrophic failures
    
    Parameters:
    -----------
    machine_id : int
        Machine identifier
    t : float
        Absolute Time or time since last catastrophic failure
    pm_affects : bool
        True -> minor failure (affected by PM), False -> catastrophic failure (not affected by PM)
    T : float
        PM interval/time
    push : float
        PM effectiveness parameter (0=AGAO, 1=AGAN)
    scale, intercept, shape : float
        Parameters for hazard function based on model_type
    fixed_covs : array-like
        Fixed covariates for all machines
    dynamic_cov_t : array-like
        Dynamic covariates at time t
    beta_fixed, beta_dynamic : array-like
        Coefficients for fixed and dynamic covariates
    model_type : str
        Type of hazard model: 'linear', 'log-linear', or 'weibull'
    with_covariates : bool
        Whether to include covariate effects
    
    Returns:
    --------
    float : Hazard rate at time t
    """
    # Calculate virtual time V_t
    if pm_affects and T is not None:
        time_in_cycle = t % T
        completed_cycles = np.floor(t / T)
        V_t = time_in_cycle + (1 - push) * T * completed_cycles
    else:
        V_t = t

    # Baseline hazard
    if model_type == "linear":
        hazard = intercept + scale * V_t
    elif model_type == "log-linear":
        hazard = np.exp(intercept + scale * V_t)
    elif model_type == "weibull":
        if shape is None or scale is None:
            raise ValueError("Weibull requires shape and scale parameters")
        hazard = (shape / scale) * (V_t / scale) ** (shape - 1)
    else:
        raise ValueError("model_type must be 'linear', 'log-linear', or 'weibull'")

    if not with_covariates:
        return hazard

    # Add covariate effects
    cov_effect = 0.0
    
    if fixed_covs is not None and beta_fixed is not None and len(beta_fixed) > 0:
        cov_effect += np.dot(fixed_covs[machine_id-1], beta_fixed)
    
    if dynamic_cov_t is not None and beta_dynamic is not None and len(beta_dynamic) > 0:
        cov_effect += np.dot(dynamic_cov_t, beta_dynamic)

    return hazard * np.exp(cov_effect)


def integrated_lambda_closed(b, shape, scale, model_type="weibull", intercept=None):
    """
    Closed-form cumulative hazard Λ(b) for catastrophic failures
    Only applicable when pm_affects=False and with_covariates=False
    
    Parameters:
    -----------
    b : float
        Upper bound of integration interval [0, b]
    shape : float
        Shape parameter (for Weibull)
    scale : float
        Scale parameter
    model_type : str
        Type of model: 'linear', 'log-linear', or 'weibull'
    intercept : float
        Intercept parameter (for linear and log-linear)
    
    Returns:
    --------
    float : Cumulative hazard over [0, b]
    """
    if model_type == "linear":
        return intercept * b + 0.5 * scale * b**2
    elif model_type == "log-linear":
        return (np.exp(intercept) / scale) * (np.exp(scale * b) - 1)
    elif model_type == "weibull":
        if shape is None:
            raise ValueError("Weibull requires shape parameter")
        return (b / scale) ** shape
    else:
        raise ValueError("model_type must be 'linear', 'log-linear', or 'weibull'")


def integrated_lambda_numeric(
    machine_id, a, b, pm_affects=True, T=None, push=0.0, scale=None, intercept=None, shape=None,
    fixed_covs=None, dynamic_cov_t=None, beta_fixed=None, beta_dynamic=None,
    model_type="weibull", with_covariates=True
):
    """
    Numerical integration of hazard function over interval [a, b]
    
    Parameters:
    -----------
    machine_id : int
        Machine identifier
    a, b : float
        Integration interval bounds
    pm_affects : bool
        True -> minor failure (affected by PM), False -> catastrophic failure
    (other parameters same as lambda_f)
    
    Returns:
    --------
    float : Integrated hazard over [a, b]
    """
    integrand = lambda t: lambda_f(
        machine_id, t, pm_affects=pm_affects, T=T, push=push,
        scale=scale, intercept=intercept, shape=shape,
        fixed_covs=fixed_covs, dynamic_cov_t=dynamic_cov_t,
        beta_fixed=beta_fixed, beta_dynamic=beta_dynamic,
        model_type=model_type, with_covariates=with_covariates
    )

    result, _ = quad(integrand, a, b)
    return result


def compute_cumulative_integrals(
    cumulative_integrals, machine_id, valid_indices, m, delta_t, shape, scale, intercept=None,
    fixed_covs=None, dynamic_covs=None, beta_fixed=None, beta_dynamic=None, model_type="weibull",
    with_covariates=True, pm_affects=False, T=None, push=0.0
):
    """
    Update cumulative integrals (applicable for both minor and catastrophic failures)
    
    Parameters:
    -----------
    cumulative_integrals : list
        Existing cumulative integral values
    machine_id : int
        Machine identifier
    valid_indices : int
        Last failure index in the time series
    m : int
        Total number of discretized time intervals
    delta_t : float
        Time interval length
    shape, scale, intercept : float
        Hazard function parameters
    fixed_covs : array-like
        Fixed covariates
    dynamic_covs : array-like
        Dynamic covariates over time
    beta_fixed, beta_dynamic : array-like
        Covariate coefficients
    model_type : str
        Type of hazard model
    with_covariates : bool
        Whether to include covariates
    pm_affects : bool
        True -> minor failure (affected by PM), False -> catastrophic failure
    T : float
        PM interval
    push : float
        PM effectiveness
  
    Returns:
    --------
    list : Updated cumulative integral values
    """
    if len(cumulative_integrals) == 0:
        kept_integrals = [0]
        cumulative_value = 0
        start_index = 0
    else:
        kept_integrals = cumulative_integrals[:valid_indices+1]
        cumulative_value = cumulative_integrals[valid_indices]
        start_index = valid_indices

    for i in range(start_index, m):
        time_start = i * delta_t
        time_end = (i + 1) * delta_t

        if with_covariates:
            # With covariates -> numerical integration
            dynamic_cov_t = dynamic_covs[i] if dynamic_covs is not None else None
            integral_val = integrated_lambda_numeric(
                machine_id, time_start, time_end,
                pm_affects=pm_affects, T=T, push=push,
                scale=scale, intercept=intercept, shape=shape,
                fixed_covs=fixed_covs, dynamic_cov_t=dynamic_cov_t,
                beta_fixed=beta_fixed, beta_dynamic=beta_dynamic,
                model_type=model_type, with_covariates=True
            )
            cumulative_value += integral_val
        else:
            if pm_affects:
                # Minor failure: affected by PM, no closed-form solution
                integral_val = integrated_lambda_numeric(
                    machine_id, time_start, time_end,
                    pm_affects=True, T=T, push=push,
                    scale=scale, intercept=intercept, shape=shape,
                    fixed_covs=None, dynamic_cov_t=None,
                    beta_fixed=None, beta_dynamic=None,
                    model_type=model_type, with_covariates=False
                )
                cumulative_value += integral_val
            else:
                # Catastrophic failure: no covariates -> use closed-form
                Λ_b = integrated_lambda_closed(
                    time_end, shape, scale, model_type=model_type, intercept=intercept
                )
                cumulative_value = Λ_b

        kept_integrals.append(cumulative_value)

    return kept_integrals
