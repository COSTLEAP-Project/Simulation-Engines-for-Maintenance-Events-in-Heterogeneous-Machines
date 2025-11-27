"""
Simulation Module

This module provides the main simulation engine for degradation processes
with maintenance actions and covariate effects.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional

from covariates import CovariateManager, CovariateSpec
from degradation import generate_degradation_increment_with_covariates, add_observation_noise
from repair import sample_post_repair_mixed, compute_repair_effectiveness
from cost import CostParams, compute_maintenance_cost


def simulate_path_with_covariates(
    degradation_type: str = "compound_poisson",
    degradation_params: Dict[str, Any] = None,
    covariate_specs: List[CovariateSpec] = None,
    covariate_effects: Dict[str, np.ndarray] = None,
    dt: float = 0.01,
    PM_level: float = 2.0,
    PM_interval: float = None,
    L: float = 5.0,
    x0: float = 0.0,
    repair_func: Callable = None,
    repair_params: Dict = None,
    obs_time: float = 100.0,
    random_seed: int = None,
    noise: Optional[Dict[str, Any]] = None,
    cost_params: CostParams = None,
    cost_covariate_specs: List[CovariateSpec] = None,
    cost_covariate_effects: Dict[str, np.ndarray] = None
) -> Dict[str, Any]:
    """
    Simulate degradation path with covariates, distinguishing latent true wear (X_latent)
    and noisy observations (X_obs), and compute maintenance costs.
    
    Decision Logic:
    - CM (Corrective Maintenance): Triggered when x_latent >= L (true failure)
    - PM (Preventive Maintenance): Triggered when x_observed >= PM_level (decision based on observable info)
    
    Maintenance Strategies:
    - Level-only: PM triggered when observed degradation reaches PM_level
    - Time-and-level: PM triggered by either time (PM_interval) or level (PM_level)
    
    Args:
        degradation_type: Type of degradation process ('gamma', 'wiener', 'compound_poisson', etc.)
        degradation_params: Parameters for degradation process
        covariate_specs: List of covariate specifications for degradation
        covariate_effects: Dictionary mapping parameter names to coefficient vectors
        dt: Time step size
        PM_level: Preventive maintenance threshold (based on observed level)
        PM_interval: Time interval for scheduled PM (None for level-only strategy)
        L: Catastrophic failure threshold (based on latent level)
        x0: Initial degradation level
        repair_func: Function for imperfect repair (default: sample_post_repair_mixed)
        repair_params: Parameters for repair function
        obs_time: Total observation time
        random_seed: Random seed for reproducibility
        noise: Observation noise specification
        cost_params: Cost model parameters
        cost_covariate_specs: List of covariate specifications for cost (independent from degradation covariates)
        cost_covariate_effects: Covariate effects on cost parameters
    
    Returns:
        Dictionary containing:
            - times: Time points
            - degra_level_latent: Latent degradation levels
            - degra_level_observed: Observed degradation levels
            - events: List of maintenance events
            - covariate_history: History of degradation covariates
            - cost_covariate_history: History of cost covariates
            - total_cost: Total maintenance cost
            - cost_by_type: Cost breakdown by maintenance type
            - count_by_type: Count of each maintenance type
            - average_cost_by_type: Average cost by maintenance type
            - strategy: Maintenance strategy used
            - covariate_names: Names of degradation covariates
            - cost_covariate_names: Names of cost covariates
            - obs_time: Total observation time
            - final_level_latent: Final latent degradation level
            - final_level_observed: Final observed degradation level
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if degradation_params is None:
        degradation_params = {"lambda_shock": 0.1, "shock_scale": 1.0}
    
    if covariate_specs is None:
        covariate_specs = []
    
    # Initialize degradation covariate manager
    covariate_manager = CovariateManager(covariate_specs) if covariate_specs else None
    
    # Initialize cost covariate manager
    cost_covariate_manager = CovariateManager(cost_covariate_specs) if cost_covariate_specs else None
    
    # Initialize cost parameters (use defaults if not provided)
    if cost_params is None:
        cost_params = CostParams()
    
    # Set default repair function if not provided
    if repair_func is None:
        repair_func = sample_post_repair_mixed
    
    # Initialize tracking variables
    times = [0.0]
    degra_level_latent = [x0]
    degra_level_observed = [x0]
    events = []
    covariate_history = []
    cost_covariate_history = []
    total_cost = 0.0
    
    # Simulation state variables
    t = 0.0
    x_lat = x0  # Latent degradation level
    x_obs = x0  # Observed degradation level
    last_repair_post_lat = x0  # Degradation level after last repair
    last_pm_time = 0.0  # Time of last PM
    has_scheduled_pm = (PM_interval is not None)
    
    # Main simulation loop
    while t < obs_time:
        # Update degradation covariates
        if covariate_manager:
            covariate_manager.update_covariates(t, x_lat, dt)
            current_covariates_vec = covariate_manager.get_covariate_vector()
            current_covariates_dict = dict(
                zip(covariate_manager.get_covariate_names(), current_covariates_vec)
            )
            covariate_history.append({'time': t, 'covariates': current_covariates_dict})
        else:
            current_covariates_vec = None
            current_covariates_dict = {}
        
        # Update cost covariates
        if cost_covariate_manager:
            cost_covariate_manager.update_covariates(t, x_lat, dt)
            current_cost_covariates_vec = cost_covariate_manager.get_covariate_vector()
            current_cost_covariates_dict = dict(
                zip(cost_covariate_manager.get_covariate_names(), current_cost_covariates_vec)
            )
            cost_covariate_history.append({'time': t, 'cost_covariates': current_cost_covariates_dict})
        else:
            current_cost_covariates_vec = None
            current_cost_covariates_dict = {}
        
        # Generate degradation increment
        incr = generate_degradation_increment_with_covariates(
            degradation_type=degradation_type,
            dt=dt,
            params=degradation_params,
            covariates=current_covariates_vec,
            covariate_effects=covariate_effects
        )
        
        x_lat += incr
        t += dt
        x_obs = add_observation_noise(x_lat, dt, noise)
        
        # --- Catastrophic Failure (based on latent degradation level) ---
        if x_lat >= L:
            times.append(t)
            degra_level_latent.append(x_lat)
            degra_level_observed.append(x_obs)
            
            # Compute CM cost (using degradation covariates)
            cm_cost = compute_maintenance_cost(
                maintenance_type='cm',
                cost_params=cost_params,
                cost_covariates=current_covariates_vec,
                cost_covariate_effects=cost_covariate_effects
            )
            total_cost += cm_cost
            
            events.append({
                'time': t,
                'type': 'catastrophic_failure_replacement',
                'level_before_latent': x_lat,
                'level_before_observed': x_obs,
                'level_after_latent': 0.0,
                'level_after_observed': 0.0,
                'covariates': current_covariates_dict,
                'cost_covariates': current_cost_covariates_dict,
                'cost': cm_cost
            })
            
            # Reset to perfect condition
            last_repair_post_lat = 0.0
            x_lat = 0.0
            x_obs = 0.0
            last_pm_time = t
            
            if t < obs_time:
                times.append(t)
                degra_level_latent.append(x_lat)
                degra_level_observed.append(x_obs)
            continue
        
        times.append(t)
        degra_level_latent.append(x_lat)
        degra_level_observed.append(x_obs)
        
        # --- PM Logic (based on observed degradation level) ---
        if not has_scheduled_pm:
            # Level-only strategy (based on observed level)
            if x_obs >= PM_level:
                z_lat = x_lat
                y_lat = repair_func(last_repair_post_lat, z_lat, params=repair_params)
                
                # Compute imperfect PM cost (using cost covariates)
                repair_eff = compute_repair_effectiveness(z_lat, y_lat)
                ipm_cost = compute_maintenance_cost(
                    maintenance_type='imperfect_pm',
                    cost_params=cost_params,
                    cost_covariates=current_cost_covariates_vec,
                    cost_covariate_effects=cost_covariate_effects,
                    repair_effectiveness=repair_eff
                )
                total_cost += ipm_cost
                
                events.append({
                    'time': t,
                    'type': 'imperfect_repair',
                    'trigger_reason': 'level_threshold_observed',
                    'level_before_latent': z_lat,
                    'level_before_observed': x_obs,
                    'level_after_latent': y_lat,
                    'level_after_observed': y_lat,
                    'last_repair_post_before_latent': last_repair_post_lat,
                    'repair_effectiveness': repair_eff,
                    'covariates': current_covariates_dict,
                    'cost_covariates': current_cost_covariates_dict,
                    'cost': ipm_cost
                })
                
                last_repair_post_lat = y_lat
                x_lat = y_lat
                x_obs = y_lat
                times.append(t)
                degra_level_latent.append(x_lat)
                degra_level_observed.append(x_obs)
                
                # Check if post-repair level still above threshold
                if y_lat >= PM_level:
                    # Compute perfect PM cost (using degradation covariates)
                    ppm_cost = compute_maintenance_cost(
                        maintenance_type='perfect_pm',
                        cost_params=cost_params,
                        cost_covariates=current_covariates_vec,
                        cost_covariate_effects=cost_covariate_effects
                    )
                    total_cost += ppm_cost
                    
                    events.append({
                        'time': t,
                        'type': 'perfect_preventive_maintenance',
                        'trigger_reason': 'post_repair_still_above_threshold',
                        'level_before_latent': y_lat,
                        'level_before_observed': x_obs,
                        'level_after_latent': 0.0,
                        'level_after_observed': 0.0,
                        'last_repair_post_before_latent': last_repair_post_lat,
                        'covariates': current_covariates_dict,
                        'cost_covariates': current_cost_covariates_dict,
                        'cost': ppm_cost
                    })
                    
                    last_repair_post_lat = 0.0
                    x_lat = 0.0
                    x_obs = 0.0
                    times.append(t)
                    degra_level_latent.append(x_lat)
                    degra_level_observed.append(x_obs)
        
        else:
            # Time-and-level strategy (based on observed level)
            level_triggered = (x_obs >= PM_level)
            time_triggered = (t - last_pm_time >= PM_interval)
            
            if level_triggered:
                # Compute perfect PM cost (using degradation covariates)
                ppm_cost = compute_maintenance_cost(
                    maintenance_type='perfect_pm',
                    cost_params=cost_params,
                    cost_covariates=current_covariates_vec,
                    cost_covariate_effects=cost_covariate_effects
                )
                total_cost += ppm_cost
                
                events.append({
                    'time': t,
                    'type': 'perfect_preventive_maintenance',
                    'trigger_reason': 'level_threshold_observed',
                    'level_before_latent': x_lat,
                    'level_before_observed': x_obs,
                    'level_after_latent': 0.0,
                    'level_after_observed': 0.0,
                    'last_repair_post_before_latent': last_repair_post_lat,
                    'covariates': current_covariates_dict,
                    'cost_covariates': current_cost_covariates_dict,
                    'cost': ppm_cost
                })
                
                last_repair_post_lat = 0.0
                x_lat = 0.0
                x_obs = 0.0
                times.append(t)
                degra_level_latent.append(x_lat)
                degra_level_observed.append(x_obs)
            
            elif time_triggered:
                z_lat = x_lat
                z_obs = x_obs
                y_lat = repair_func(last_repair_post_lat, z_lat, params=repair_params)
                y_obs = y_lat
                
                # Compute imperfect PM cost (using cost covariates)
                repair_eff = compute_repair_effectiveness(z_lat, y_lat)
                ipm_cost = compute_maintenance_cost(
                    maintenance_type='imperfect_pm',
                    cost_params=cost_params,
                    cost_covariates=current_cost_covariates_vec,
                    cost_covariate_effects=cost_covariate_effects,
                    repair_effectiveness=repair_eff
                )
                total_cost += ipm_cost
                
                events.append({
                    'time': t,
                    'type': 'imperfect_repair',
                    'trigger_reason': 'scheduled_time',
                    'level_before_latent': z_lat,
                    'level_before_observed': z_obs,
                    'level_after_latent': y_lat,
                    'level_after_observed': y_obs,
                    'last_repair_post_before_latent': last_repair_post_lat,
                    'repair_effectiveness': repair_eff,
                    'covariates': current_covariates_dict,
                    'cost_covariates': current_cost_covariates_dict,
                    'cost': ipm_cost
                })
                
                last_repair_post_lat = y_lat
                x_lat = y_lat
                x_obs = y_obs
                last_pm_time = t
                times.append(t)
                degra_level_latent.append(x_lat)
                degra_level_observed.append(x_obs)
    
    # Compute cost statistics
    cost_by_type = {
        'perfect_pm': sum(e['cost'] for e in events if e['type'] == 'perfect_preventive_maintenance'),
        'imperfect_pm': sum(e['cost'] for e in events if e['type'] == 'imperfect_repair'),
        'cm': sum(e['cost'] for e in events if e['type'] == 'catastrophic_failure_replacement')
    }
    
    count_by_type = {
        'perfect_pm': sum(1 for e in events if e['type'] == 'perfect_preventive_maintenance'),
        'imperfect_pm': sum(1 for e in events if e['type'] == 'imperfect_repair'),
        'cm': sum(1 for e in events if e['type'] == 'catastrophic_failure_replacement')
    }
    
    return {
        'times': np.array(times),
        'degra_level_latent': np.array(degra_level_latent),
        'degra_level_observed': np.array(degra_level_observed),
        'events': events,
        'covariate_history': covariate_history,
        'cost_covariate_history': cost_covariate_history,
        'obs_time': obs_time,
        'final_level_latent': degra_level_latent[-1] if degra_level_latent else x0,
        'final_level_observed': degra_level_observed[-1] if degra_level_observed else x0,
        'strategy': 'level_only' if not has_scheduled_pm else 'time_and_level',
        'covariate_names': (covariate_manager.get_covariate_names()
                            if covariate_manager else []),
        'cost_covariate_names': (cost_covariate_manager.get_covariate_names()
                                 if cost_covariate_manager else []),
        'total_cost': total_cost,
        'cost_by_type': cost_by_type,
        'count_by_type': count_by_type,
        'average_cost_by_type': {k: (v / count_by_type[k] if count_by_type[k] > 0 else 0.0) 
                                 for k, v in cost_by_type.items()}
    }
