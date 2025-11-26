"""
Failure type classification module.

This module contains functions to:
- Determine if a failure is minor or catastrophic (competing risks)
- Classify minor failure subtypes using multinomial logistic regression
"""

import numpy as np
from .hazard import lambda_f


def get_failure_type(
    machine_id, ft, T, push, scale, intercept, shape, with_covariates_minor, model_type_minor,
    fixed_covs, dynamic_cov_t, beta_fixed, beta_dynamic,
    scale_c, shape_c, intercept_c, model_type_catas, with_covariates_catas
):
    """
    Determine whether failure at time ft is minor or catastrophic
    
    Parameters:
    -----------
    machine_id : int
        Machine identifier
    ft : float
        Failure time
    T : float
        PM interval
    push : float
        PM effectiveness
    scale, intercept, shape : float
        Minor failure hazard parameters
    with_covariates_minor : bool
        Whether minor failures use covariates
    model_type_minor : str
        Minor failure hazard model type
    fixed_covs : array-like
        Fixed covariates
    dynamic_cov_t : array-like
        Dynamic covariates at time ft
    beta_fixed, beta_dynamic : array-like
        Covariate coefficients
    scale_c, shape_c, intercept_c : float
        Catastrophic failure hazard parameters
    model_type_catas : str
        Catastrophic failure hazard model type
    with_covariates_catas : bool
        Whether catastrophic failures use covariates
    
    Returns:
    --------
    tuple : (p_minor, type_risk)
        - p_minor: probability of being a minor failure
        - type_risk: 1=minor, 2=catastrophic
    """
    # Minor failure hazard
    lambda_minor = lambda_f(
        machine_id, ft, pm_affects=True, T=T, push=push,
        scale=scale, intercept=intercept, shape=shape,
        fixed_covs=fixed_covs, dynamic_cov_t=dynamic_cov_t, 
        beta_fixed=beta_fixed, beta_dynamic=beta_dynamic,
        model_type=model_type_minor, with_covariates=with_covariates_minor
    )

    # Catastrophic failure hazard
    lambda_catastrophic = lambda_f(
        machine_id, ft, pm_affects=False, T=None, push=0.0,
        scale=scale_c, intercept=intercept_c, shape=shape_c,
        fixed_covs=fixed_covs, dynamic_cov_t=dynamic_cov_t, 
        beta_fixed=beta_fixed, beta_dynamic=beta_dynamic,
        model_type=model_type_catas, with_covariates=with_covariates_catas
    )

    # Total hazard
    lambda_total = lambda_minor + lambda_catastrophic
    if lambda_total == 0:
        raise ValueError("lambda_total=0, hazard is zero everywhere")

    # Probability of minor failure
    p_minor = lambda_minor / lambda_total

    # Sample failure type
    is_minor = np.random.binomial(1, p_minor)
    # failure_type: 'minor' -> type_risk = 1; 'catastrophic' -> type_risk = 2
    type_risk = 1 if is_minor else 2
    
    return p_minor, type_risk


def get_minor_failure_type(
    machine_id, beta_multinom_fixed, beta_multinom_dynamic,
    fixed_covs, dynamic_cov_t, n_minor_types
):
    """
    Sample minor failure subtype using multinomial logistic regression with covariates
    
    Parameters:
    -----------
    machine_id : int
        Machine identifier
    beta_multinom_fixed : array-like
        Fixed covariate coefficients, shape (n_fixed_features, n_minor_types-1)
    beta_multinom_dynamic : array-like
        Dynamic covariate coefficients, shape (n_dynamic_features, n_minor_types-1)
    fixed_covs : array-like
        Fixed covariates for all machines, shape (n_machines, n_fixed_features)
    dynamic_cov_t : array-like
        Dynamic covariates at current time, shape (n_dynamic_features,)
    n_minor_types : int
        Number of minor failure types
    
    Returns:
    --------
    int : Minor failure subtype (1 to n_minor_types)
    """
    # Robustness: if someone sets n < 2, fall back to a single class
    if n_minor_types is None or n_minor_types < 2:
        return 1

    Jm1 = n_minor_types - 1  # constant length we must maintain

    # ----- Fixed part (length J-1) -----
    if (fixed_covs is not None) and (beta_multinom_fixed is not None):
        fixed_part = np.dot(fixed_covs[machine_id - 1], beta_multinom_fixed)  # -> (J-1,)
    else:
        fixed_part = np.zeros(Jm1, dtype=float)

    # ----- Dynamic part (length J-1) -----
    if (dynamic_cov_t is not None) and (beta_multinom_dynamic is not None):
        dyn = np.ravel(dynamic_cov_t)  
        dynamic_part = np.dot(dyn, beta_multinom_dynamic)  # -> (J-1,)
    else:
        dynamic_part = np.zeros(Jm1, dtype=float)

    # Combine fixed and dynamic parts
    logits = fixed_part + dynamic_part  # shape (J_minor_types-1,)

    # Numerical stability: append reference category (logit = 0)
    logits = np.append(logits, 0.0)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)

    # Normalize to get probabilities
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample category according to probabilities
    f_type = np.random.choice(len(probs), p=probs) + 1

    return f_type  # Returns category 1 to n_minor_types
