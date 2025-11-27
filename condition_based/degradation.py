"""
Degradation Process Module

This module provides functions for generating degradation increments
with covariate effects using various stochastic processes.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_phm_intensity(
    base_intensity: float,
    covariates: np.ndarray,
    beta_coeffs: np.ndarray,
    t: float = None
) -> float:
    """
    Compute Proportional Hazards Model (PHM) intensity function.
    
    Formula: λ(t|Z(t)) = λ₀(t) × exp(β'Z(t))
    
    IMPORTANT: PHM covariate effects are instantaneous. At time t, we use
    the covariate values Z(t) at that time point to calculate the intensity,
    not historical cumulative values. This is a core assumption of PHM.
    
    Mathematical Expression:
    - λ(t|Z(t)) = λ₀(t) × exp(β₁Z₁(t) + β₂Z₂(t) + ... + βₚZₚ(t))
    - where Z₁(t), Z₂(t), ..., Zₚ(t) are covariate values at time t
    
    Args:
        base_intensity: Baseline intensity λ₀(t) - intensity when all covariates are 0
        covariates: Covariate vector at time t: Z(t) = [Z₁(t), Z₂(t), ..., Zₚ(t)]
        beta_coeffs: Regression coefficient vector β = [β₁, β₂, ..., βₚ]
        t: Current time point (for debugging/logging purposes)
    
    Returns:
        Adjusted intensity at time t: λ(t|Z(t))
    
    Example:
        At time t=10:
        - Temperature Z₁(10) = 25°C, β₁ = 0.02
        - Load Z₂(10) = 1.5, β₂ = 0.5
        - Baseline intensity λ₀(10) = 0.1
        Then λ(10|Z(10)) = 0.1 × exp(0.02×25 + 0.5×1.5) = 0.1 × exp(1.25) ≈ 0.349
    """
    if len(covariates) != len(beta_coeffs):
        raise ValueError("Covariates and coefficients must have same length")
    
    # Calculate linear predictor: β₁Z₁(t) + β₂Z₂(t) + ... + βₚZₚ(t)
    linear_predictor = np.dot(beta_coeffs, covariates)
    
    # Return instantaneous intensity: λ₀(t) × exp(β'Z(t))
    instantaneous_intensity = base_intensity * np.exp(linear_predictor)
    
    return instantaneous_intensity


def generate_degradation_increment_with_covariates(
    degradation_type: str,
    dt: float,
    params: Dict[str, Any],
    covariates: np.ndarray = None,
    covariate_effects: Dict[str, np.ndarray] = None
) -> float:
    """
    Generate degradation increment with covariate effects.
    
    Supports multiple stochastic processes:
    - gamma: Gamma process
    - inverse_gaussian: Inverse Gaussian process
    - wiener: Wiener process (Brownian motion with drift)
    - compound_poisson: Compound Poisson process with PHM for shock arrival
    - combined: Combination of base process and compound Poisson shocks
    
    Args:
        degradation_type: Type of degradation process
        dt: Time step size
        params: Base parameters for the degradation process
        covariates: Covariate vector at current time
        covariate_effects: Dictionary mapping parameter names to coefficient vectors
                          Only parameters affected by covariates should be included
                          Note: 'lambda_shock' is handled specially in compound_poisson
    
    Returns:
        Degradation increment for time step dt
    
    Notes:
        - Covariate effects are applied to regular parameters EXCEPT lambda_shock
        - lambda_shock is handled specially in compound_poisson branch using PHM
        - Parameters are adjusted using log-link: param_adjusted = param_base × exp(β'Z)
          to ensure positivity
    """
    
    # Copy base parameters to avoid modifying original
    adjusted_params = params.copy() if params is not None else {}
    
    # Apply covariate effects to regular parameters (except lambda_shock)
    if covariates is not None and covariate_effects is not None:
        for param_name, beta_coeffs in covariate_effects.items():
            # Skip lambda_shock - it's handled specially in compound_poisson
            if param_name == "lambda_shock":
                continue
            
            if param_name in adjusted_params:
                base_value = adjusted_params[param_name]
                linear_predictor = np.dot(beta_coeffs, covariates)
                # Use log-link to ensure parameter positivity
                adjusted_params[param_name] = base_value * np.exp(linear_predictor)
    
    # Generate increment based on degradation type
    if degradation_type == "gamma":
        # Gamma process with two methods for handling covariates
        # Method 1: Direct parameter adjustment
        alpha = adjusted_params.get("alpha", 0.5)
        beta = adjusted_params.get("beta", 1.0)
        return np.random.gamma(alpha * dt, beta)
    
    elif degradation_type == "inverse_gaussian":
        # Inverse Gaussian process
        mu = adjusted_params.get("mu", 1.0)
        lam = adjusted_params.get("lambda", 1.0)
        return np.random.wald(mu * dt, lam * dt**2)
    
    elif degradation_type == "wiener":
        # Wiener process (Brownian motion with drift)
        mu = adjusted_params.get("mu", 0.1)
        sigma = adjusted_params.get("sigma", 0.2)
        return mu * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
    
    elif degradation_type == "compound_poisson":
        # Compound Poisson process - use PHM to adjust shock arrival intensity
        base_lambda = adjusted_params.get("lambda_shock", 0.1)
        
        # If covariate effects specifically target arrival rate
        if covariates is not None and "lambda_shock" in covariate_effects:
            beta_coeffs = covariate_effects["lambda_shock"]
            # KEY POINT: Use covariate values Z(t) at current time to compute instantaneous intensity
            adjusted_lambda = compute_phm_intensity(base_lambda, covariates, beta_coeffs)
        else:
            adjusted_lambda = base_lambda
        
        shock_dist = adjusted_params.get("shock_dist", "exponential")
        shock_scale = adjusted_params.get("shock_scale", 1.0)
        
        # Generate number of shocks in time interval dt
        n_shocks = np.random.poisson(adjusted_lambda * dt)
        
        if n_shocks == 0:
            return 0.0
        
        # Generate shock magnitudes based on distribution
        incr = 0.0
        if shock_dist == "exponential":
            incr = np.sum(np.random.exponential(shock_scale, size=n_shocks))
        elif shock_dist == "gamma":
            shape = adjusted_params.get("shock_shape", 2.0)
            scale = adjusted_params.get("shock_scale", 1.0)
            incr = np.sum(np.random.gamma(shape, scale, size=n_shocks))
        elif shock_dist == "lognormal":
            mu = adjusted_params.get("shock_mu", 0.0)
            sigma = adjusted_params.get("shock_sigma", 1.0)
            incr = np.sum(np.random.lognormal(mu, sigma, size=n_shocks))
        
        return incr
    
    elif degradation_type == "combined":
        # Combined process: base process + compound Poisson shocks
        base_process = adjusted_params.get("base_process", "gamma")
        
        # Generate base process increment
        base_incr = generate_degradation_increment_with_covariates(
            base_process, dt, adjusted_params, covariates, covariate_effects
        )
        
        # Generate shock process increment
        shock_incr = generate_degradation_increment_with_covariates(
            "compound_poisson", dt, adjusted_params, covariates, covariate_effects
        )
        
        return base_incr + shock_incr
    
    else:
        raise ValueError(f"Unsupported degradation type: {degradation_type}")


def add_observation_noise(
    x: float,
    dt: float,
    noise: Optional[Dict[str, Any]]
) -> float:
    """
    Add observation noise to a scalar degradation level.
    
    Supports two types of noise:
    - additive_normal: x_obs = x + N(0, σ²)
    - brownian_increment: x_obs = x + N(0, σ²×dt)
    
    Args:
        x: True degradation level
        dt: Time step (used for brownian_increment)
        noise: Noise specification dictionary with keys:
               - "type": "none" | "additive_normal" | "brownian_increment"
               - "sigma": standard deviation or diffusion volatility
    
    Returns:
        Observed degradation level (always non-negative)
    
    Example:
        noise = {"type": "additive_normal", "sigma": 0.1}
        or
        noise = {"type": "brownian_increment", "sigma": 0.05}
    """
    if not noise or noise.get("type", "none") == "none":
        return x
    
    noise_type = noise["type"]
    sigma = float(noise.get("sigma", 0.0))
    
    if noise_type == "additive_normal":
        # Add Gaussian noise with fixed variance
        return max(0.0, x + np.random.normal(0.0, sigma))
    
    elif noise_type == "brownian_increment":
        # Add Brownian increment with variance σ²×dt
        return max(0.0, x + np.random.normal(0.0, sigma * np.sqrt(dt)))
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
