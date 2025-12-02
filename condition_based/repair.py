"""
Repair Module

This module provides functions for modeling repair effectiveness
in imperfect maintenance scenarios.
"""

import numpy as np
from typing import Dict, Any, Optional


def sample_post_repair_mixed(
    y_lower: float,
    y_upper: float,
    params: Optional[Dict[str, Any]] = None
) -> float:
    """
    Mixed/relaxed version of post-repair sampling for imperfect maintenance.
    
    This function models the degradation level after an imperfect repair.
    With probability p_major, performs major repair supporting [0, y_upper].
    With probability 1-p_major, performs minor repair supporting [y_lower, y_upper].
    
    Special Case Handling:
    - If y_upper <= y_lower: degradation level after last repair is still above PM threshold,
      execute perfect preventive maintenance (complete restoration to 0).
    - If y_upper <= 0: already at perfect condition, return 0.
    
    Args:
        y_lower: Lower bound (typically best (lowest) degradation level that we can reach after this repair)
        y_upper: Upper bound (typically current degradation level before repair)
        params: Dictionary containing repair parameters:
            - p_major: Probability of major repair (default: 0.2)
            - dist_minor: Distribution for minor repair ('uniform'/'beta'/'proportional')
            - dist_major: Distribution for major repair ('uniform'/'beta'/'proportional')
            - a_minor, b_minor: Beta distribution parameters for minor repair
            - a_major, b_major: Beta distribution parameters for major repair
            - rho_minor: Proportional parameter for minor repair (reduction ratio)
            - rho_major: Proportional parameter for major repair (reduction ratio)
    
    Returns:
        Post-repair degradation level
    
    Distribution Types:
        - uniform: Uniformly distributed in support interval
        - beta: Beta distribution scaled to support interval
        - proportional: y_new = y_lower + (1-rho)*(y_upper-y_lower) for minor
                       y_new = (1-rho)*y_upper for major
    
    Examples:
        # Uniform distribution with 20% major repair probability
        params = {"p_major": 0.2, "dist_minor": "uniform", "dist_major": "uniform"}
        
        # Beta distribution with custom parameters
        params = {
            "p_major": 0.3,
            "dist_minor": "beta", "a_minor": 2.0, "b_minor": 2.0,
            "dist_major": "beta", "a_major": 2.0, "b_major": 5.0
        }
        
        # Proportional reduction
        params = {
            "p_major": 0.2,
            "dist_minor": "proportional", "rho_minor": 0.5,
            "dist_major": "proportional", "rho_major": 0.7
        }
    """
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    p_major = params.get("p_major", 0.2)
    dist_minor = params.get("dist_minor", "uniform")
    dist_major = params.get("dist_major", "uniform")
    a_minor = params.get("a_minor", 2.0)
    b_minor = params.get("b_minor", 2.0)
    a_major = params.get("a_major", 2.0)
    b_major = params.get("b_major", 5.0)
    rho_minor = params.get("rho_minor", 0.5)
    rho_major = params.get("rho_major", 0.7)
    
    # Handle edge case: scheduled PM at y_upper=0
    if y_upper <= 0:
        return 0.0
    
    # Special case: y_upper <= y_lower means post-repair level still above PM threshold
    # Execute perfect preventive maintenance
    if y_upper <= y_lower:
        print(f"Warning: y_upper={y_upper:.6f} <= y_lower={y_lower:.6f}, "
              f"executing perfect preventive maintenance")
        return 0.0  # Perfect repair: complete restoration to initial state
    
    # Decide major or minor repair
    if np.random.rand() < p_major:
        # Major repair: support interval [0, y_upper]
        if dist_major == "uniform":
            return np.random.uniform(0, y_upper)
        elif dist_major == "beta":
            u = np.random.beta(a_major, b_major)
            return u * y_upper
        elif dist_major == "proportional":
            return (1 - rho_major) * y_upper
        else:
            raise ValueError(f"Unsupported dist_major type: {dist_major}")
    else:
        # Minor repair: support interval [y_lower, y_upper]
        if dist_minor == "uniform":
            return np.random.uniform(y_lower, y_upper)
        elif dist_minor == "beta":
            u = np.random.beta(a_minor, b_minor)
            return y_lower + u * (y_upper - y_lower)
        elif dist_minor == "proportional":
            return y_lower + (1 - rho_minor) * (y_upper - y_lower)
        else:
            raise ValueError(f"Unsupported dist_minor type: {dist_minor}")


def compute_repair_effectiveness(
    level_before: float,
    level_after: float
) -> float:
    """
    Compute repair effectiveness as the reduction ratio.
    
    Formula: u = (y_before - y_after) / y_before
    
    Args:
        level_before: Degradation level before repair
        level_after: Degradation level after repair
    
    Returns:
        Repair effectiveness in [0, 1]
        - 0: No improvement (y_after = y_before)
        - 1: Perfect repair (y_after = 0)
    
    Notes:
        - Used in cost calculation for imperfect PM
        - Returns 0 if level_before is 0 (edge case)
    """
    if level_before <= 0:
        return 0.0
    return (level_before - level_after) / level_before
