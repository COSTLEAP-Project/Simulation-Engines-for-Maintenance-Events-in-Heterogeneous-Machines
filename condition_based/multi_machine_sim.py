"""
Multi-Machine Simulation Module

This module provides functions for simulating multiple machines sequentially
and aggregating results across the fleet.

Note: Each machine independently samples its fixed covariates at initialization, so machines with the same covariate specifications may have different fixed values.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import copy

from .single_machine_sim import simulate_path_with_covariates
from .covariates import CovariateSpec
from .cost import CostParams

def simulate_multiple_machines(
    n_machines: int,
    degradation_type: str = "compound_poisson",
    degradation_params: Dict[str, Any] = None,
    covariate_specs: List[CovariateSpec] = None,
    covariate_effects: Dict[str, np.ndarray] = None,
    dt: float = 0.01,
    PM_level: Union[float, List[float], Dict[int, float]] = 2.0,
    PM_interval: Union[float, List[float], Dict[int, float], None] = None,
    L: float = 5.0,
    x0: float = 0.0,
    repair_func: Any = None,
    repair_params: Dict = None,
    obs_time: float = 100.0,
    random_seed_base: int = None,
    noise: Optional[Dict[str, Any]] = None,
    cost_params: CostParams = None,
    cost_covariate_specs: List[CovariateSpec] = None,
    cost_covariate_effects: Dict[str, np.ndarray] = None
):
    """
    Simulate multiple machines with the same or different maintenance policies.
    
    This function simulates a fleet of machines operating under the same
    degradation parameters but potentially different maintenance policies,
    with independent random realizations.
    
    IMPORTANT: 
    - Fixed covariates (type='fixed') are independently sampled for each machine
    - PM_level and PM_interval can be specified per machine or shared across fleet
    
    Args:
        n_machines: Number of machines to simulate
        degradation_type: Type of degradation process
        degradation_params: Parameters for degradation process
        covariate_specs: List of covariate specifications for degradation
                        NOTE: Fixed covariates will be resampled for each machine
        covariate_effects: Dictionary mapping parameter names to coefficient vectors
        dt: Time step size
        PM_level: Preventive maintenance threshold. Can be:
                 - float: Same PM_level for all machines (e.g., 2.0)
                 - list: PM_level for each machine (length must equal n_machines)
                 - dict: {machine_id: PM_level} for specific machines, others use default
        PM_interval: Time interval for scheduled PM (None for level-only strategy). Can be:
                    - None: All machines use level-only strategy
                    - float: Same PM_interval for all machines (e.g., 15.0)
                    - list: PM_interval for each machine (length must equal n_machines)
                    - dict: {machine_id: PM_interval} for specific machines, others use None
        L: Catastrophic failure threshold
        x0: Initial degradation level
        repair_func: Function for imperfect repair
        repair_params: Parameters for repair function
        obs_time: Total observation time
        random_seed_base: Base random seed (each machine gets seed + machine_id)
        noise: Observation noise specification
        cost_params: Cost model parameters
        cost_covariate_specs: List of covariate specifications for cost
                             NOTE: Fixed cost covariates will be resampled for each machine
        cost_covariate_effects: Covariate effects on cost parameters
    
    Returns:
        Dictionary containing:
            - machine_results: List of individual machine simulation results
            - fleet_costs: Fleet-level cost analysis
            - n_machines: Number of machines simulated
            - event_logs: DataFrame of all events (maintenance-related information) 
            - machine_logs: DataFrame of machine-level summary information 
    """
    
    # Convert PM_level and PM_interval to per-machine values (to a list of values for each machine)
    pm_levels = _parse_machine_parameter(PM_level, n_machines, default=2.0, param_name="PM_level")
    pm_intervals = _parse_machine_parameter(PM_interval, n_machines, default=None, param_name="PM_interval")
    
    # Simulate each machine sequentially
    machine_results = []
    
    for i in range(n_machines):
        # Set random seed for this machine
        seed = random_seed_base + i if random_seed_base is not None else None
        
        # Get machine-specific maintenance policy
        machine_pm_level = pm_levels[i]
        machine_pm_interval = pm_intervals[i]
        
        # Deep copy covariate specs so each machine can have independent fixed values
        machine_covariate_specs = copy.deepcopy(covariate_specs) if covariate_specs else None
        machine_cost_covariate_specs = copy.deepcopy(cost_covariate_specs) if cost_covariate_specs else None
        
        # Run simulation for this machine
        result = simulate_path_with_covariates(
            degradation_type=degradation_type,
            degradation_params=degradation_params,
            covariate_specs=machine_covariate_specs,
            covariate_effects=covariate_effects,
            dt=dt,
            PM_level=machine_pm_level,
            PM_interval=machine_pm_interval,
            L=L,
            x0=x0,
            repair_func=repair_func,
            repair_params=repair_params,
            obs_time=obs_time,
            random_seed=seed,
            noise=noise,
            cost_params=cost_params,
            cost_covariate_specs=machine_cost_covariate_specs,
            cost_covariate_effects=cost_covariate_effects
        )
        
        # Add machine ID and policy info to result
        result['machine_id'] = i
        result['PM_level'] = machine_pm_level
        result['PM_interval'] = machine_pm_interval
        machine_results.append(result)

    
    # Compute fleet-level cost analysis
    fleet_costs = compute_fleet_costs(machine_results)

    # Record events of multiple machines
    event_logs = export_event_level_df(machine_results)
    
    # Record events of multiple machines
    machine_logs = export_machine_level_df(machine_results)
    
    return {
        'machine_results': machine_results,
        'fleet_costs': fleet_costs,
        'n_machines': n_machines,
        'event_logs': event_logs,
        'machine_logs': machine_logs
    }

def _parse_machine_parameter(
    param: Union[float, int, List, Dict, None],
    n_machines: int,
    default: Any,
    param_name: str
) -> List[Any]:
    """
    Parse machine-specific parameter to a list of values for each machine.
    
    Args:
        param: Parameter value(s) - can be scalar, list, dict, or None
        n_machines: Number of machines
        default: Default value if not specified
        param_name: Parameter name (for error messages)
    
    Returns:
        List of parameter values, one for each machine
    """
    if param is None:
        # None for all machines
        return [None] * n_machines
    
    elif isinstance(param, (int, float)):
        # Same value for all machines
        return [param] * n_machines
    
    elif isinstance(param, list):
        # List of values, one per machine
        if len(param) != n_machines:
            raise ValueError(f"{param_name} list length ({len(param)}) must equal n_machines ({n_machines})")
        return param
    
    elif isinstance(param, dict):
        # Dictionary: {machine_id: value}
        result = [default] * n_machines
        for machine_id, value in param.items():
            if machine_id < 0 or machine_id >= n_machines:
                raise ValueError(f"Invalid machine_id {machine_id} in {param_name} dict (must be 0 to {n_machines-1})")
            result[machine_id] = value
        return result
    
    else:
        raise ValueError(f"{param_name} must be None, scalar, list, or dict")



def compute_fleet_costs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute fleet-level cost analysis.
    
    Args:
        results: List of simulation results from all machines
    
    Returns:
        Dictionary containing fleet-level cost analysis
    """
    n_machines = len(results)
    obs_time = results[0]['obs_time']
    
    # Aggregate costs by type
    total_perfect_pm_cost = sum(r['cost_by_type']['perfect_pm'] for r in results)
    total_imperfect_pm_cost = sum(r['cost_by_type']['imperfect_pm'] for r in results)
    total_cm_cost = sum(r['cost_by_type']['cm'] for r in results)
    total_fleet_cost = sum(r['total_cost'] for r in results)
    
    # Count events by type
    total_perfect_pm_count = sum(r['count_by_type']['perfect_pm'] for r in results)
    total_imperfect_pm_count = sum(r['count_by_type']['imperfect_pm'] for r in results)
    total_cm_count = sum(r['count_by_type']['cm'] for r in results)
    
    # Average costs per event (across fleet)
    avg_perfect_pm_cost = total_perfect_pm_cost / total_perfect_pm_count if total_perfect_pm_count > 0 else 0.0
    avg_imperfect_pm_cost = total_imperfect_pm_cost / total_imperfect_pm_count if total_imperfect_pm_count > 0 else 0.0
    avg_cm_cost = total_cm_cost / total_cm_count if total_cm_count > 0 else 0.0
    
    # Cost rate (cost per unit time per machine)
    cost_rate_per_machine = total_fleet_cost / (n_machines * obs_time)
    
    fleet_costs = {
        'total_fleet_cost': total_fleet_cost,
        'cost_by_type': {
            'perfect_pm': total_perfect_pm_cost,
            'imperfect_pm': total_imperfect_pm_cost,
            'cm': total_cm_cost
        },
        'cost_percentage_by_type': {
            'perfect_pm': 100 * total_perfect_pm_cost / total_fleet_cost if total_fleet_cost > 0 else 0.0,
            'imperfect_pm': 100 * total_imperfect_pm_cost / total_fleet_cost if total_fleet_cost > 0 else 0.0,
            'cm': 100 * total_cm_cost / total_fleet_cost if total_fleet_cost > 0 else 0.0
        },
        'average_cost_per_event': {
            'perfect_pm': avg_perfect_pm_cost,
            'imperfect_pm': avg_imperfect_pm_cost,
            'cm': avg_cm_cost
        },
        'event_counts': {
            'perfect_pm': total_perfect_pm_count,
            'imperfect_pm': total_imperfect_pm_count,
            'cm': total_cm_count
        },
        'cost_rate_per_machine': cost_rate_per_machine,
        'average_cost_per_machine': total_fleet_cost / n_machines
    }
    
    return fleet_costs


def export_machine_level_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Export fleet simulation results to a pandas DataFrame.
    
    Args:
        results: List of simulation results from machines
    
    Returns:
        DataFrame with one row per machine containing summary metrics
    """
    records = []
    
    for result in results:
        record = {
            'machine_id': result['machine_id'],
            'PM_level': result.get('PM_level', np.nan),
            'PM_interval': result.get('PM_interval', np.nan),
            'strategy': result['strategy'],
            'total_cost': result['total_cost'],
            'final_level_latent': result['final_level_latent'],
            'final_level_observed': result['final_level_observed'],
            'n_perfect_pm': result['count_by_type']['perfect_pm'],
            'n_imperfect_pm': result['count_by_type']['imperfect_pm'],
            'n_cm': result['count_by_type']['cm'],
            'total_events': len(result['events']),
            'cost_perfect_pm': result['cost_by_type']['perfect_pm'],
            'cost_imperfect_pm': result['cost_by_type']['imperfect_pm'],
            'cost_cm': result['cost_by_type']['cm'],
            'obs_time': result['obs_time']
        }
        
        # Add fixed covariate values for each machine (if available)
        if result.get('covariate_history') and len(result['covariate_history']) > 0:
            first_cov = result['covariate_history'][0]['covariates']
            for cov_name, cov_value in first_cov.items():
                # Only add if it looks like a fixed covariate (constant value)
                record[f'covariate_{cov_name}'] = cov_value
        
        records.append(record)
    
    return pd.DataFrame(records)


def export_event_level_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Export all maintenance events from fleet to a pandas DataFrame.
    
    Args:
        results: List of simulation results from machines
    
    Returns:
        DataFrame with one row per maintenance event
    """
    records = []
    
    for result in results:
        machine_id = result['machine_id']
        machine_pm_level = result.get('PM_level', np.nan)
        machine_pm_interval = result.get('PM_interval', np.nan)
        
        for event in result['events']:
            record = {
                'machine_id': machine_id,
                'PM_level': machine_pm_level,
                'PM_interval': machine_pm_interval,
                'time': event['time'],
                'type': event['type'],
                'trigger_reason': event.get('trigger_reason', 'N/A'),
                'level_before_latent': event.get('level_before_latent', np.nan),
                'level_before_observed': event.get('level_before_observed', np.nan),
                'level_after_latent': event.get('level_after_latent', np.nan),
                'level_after_observed': event.get('level_after_observed', np.nan),
                'repair_effectiveness': event.get('repair_effectiveness', np.nan),
                'cost': event['cost']
            }
            records.append(record)
    
    return pd.DataFrame(records)
