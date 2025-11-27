"""
Multi-Machine Simulation Module

This module provides functions for simulating multiple machines in parallel
and aggregating results across the fleet.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from simulation import simulate_path_with_covariates
from covariates import CovariateSpec
from cost import CostParams


def simulate_single_machine_wrapper(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function for simulating a single machine.
    Used for parallel processing with ProcessPoolExecutor.
    
    Args:
        args: Dictionary containing all simulation parameters
    
    Returns:
        Simulation results dictionary with added machine_id
    """
    machine_id = args.pop('machine_id')
    
    try:
        result = simulate_path_with_covariates(**args)
        result['machine_id'] = machine_id
        return result
    except Exception as e:
        warnings.warn(f"Machine {machine_id} simulation failed: {str(e)}")
        return {
            'machine_id': machine_id,
            'error': str(e),
            'success': False
        }


def simulate_multiple_machines(
    n_machines: int,
    degradation_type: str = "compound_poisson",
    degradation_params: Dict[str, Any] = None,
    covariate_specs: List[CovariateSpec] = None,
    covariate_effects: Dict[str, np.ndarray] = None,
    dt: float = 0.01,
    PM_level: float = 2.0,
    PM_interval: float = None,
    L: float = 5.0,
    x0: float = 0.0,
    repair_func: Any = None,
    repair_params: Dict = None,
    obs_time: float = 100.0,
    random_seed_base: int = None,
    noise: Optional[Dict[str, Any]] = None,
    cost_params: CostParams = None,
    cost_covariate_specs: List[CovariateSpec] = None,
    cost_covariate_effects: Dict[str, np.ndarray] = None,
    parallel: bool = True,
    max_workers: int = None
) -> Dict[str, Any]:
    """
    Simulate multiple machines with the same configuration.
    
    This function simulates a fleet of machines operating under the same
    degradation and maintenance parameters, with independent random realizations.
    
    Args:
        n_machines: Number of machines to simulate
        degradation_type: Type of degradation process
        degradation_params: Parameters for degradation process
        covariate_specs: List of covariate specifications for degradation
        covariate_effects: Dictionary mapping parameter names to coefficient vectors
        dt: Time step size
        PM_level: Preventive maintenance threshold
        PM_interval: Time interval for scheduled PM (None for level-only strategy)
        L: Catastrophic failure threshold
        x0: Initial degradation level
        repair_func: Function for imperfect repair
        repair_params: Parameters for repair function
        obs_time: Total observation time
        random_seed_base: Base random seed (each machine gets seed + machine_id)
        noise: Observation noise specification
        cost_params: Cost model parameters
        cost_covariate_specs: List of covariate specifications for cost
        cost_covariate_effects: Covariate effects on cost parameters
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None = use CPU count)
    
    Returns:
        Dictionary containing:
            - machine_results: List of individual machine simulation results
            - summary_statistics: Aggregated statistics across all machines
            - fleet_costs: Fleet-level cost analysis
            - n_machines: Number of machines simulated
            - n_successful: Number of successful simulations
    """
    
    # Prepare arguments for each machine
    machine_args = []
    for i in range(n_machines):
        args = {
            'machine_id': i,
            'degradation_type': degradation_type,
            'degradation_params': degradation_params,
            'covariate_specs': covariate_specs,
            'covariate_effects': covariate_effects,
            'dt': dt,
            'PM_level': PM_level,
            'PM_interval': PM_interval,
            'L': L,
            'x0': x0,
            'repair_func': repair_func,
            'repair_params': repair_params,
            'obs_time': obs_time,
            'random_seed': random_seed_base + i if random_seed_base is not None else None,
            'noise': noise,
            'cost_params': cost_params,
            'cost_covariate_specs': cost_covariate_specs,
            'cost_covariate_effects': cost_covariate_effects
        }
        machine_args.append(args)
    
    # Run simulations
    machine_results = []
    
    if parallel and n_machines > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(simulate_single_machine_wrapper, args) 
                      for args in machine_args]
            
            for future in as_completed(futures):
                result = future.result()
                machine_results.append(result)
    else:
        # Sequential execution
        for args in machine_args:
            result = simulate_single_machine_wrapper(args)
            machine_results.append(result)
    
    # Sort results by machine_id
    machine_results.sort(key=lambda x: x['machine_id'])
    
    # Filter successful simulations
    successful_results = [r for r in machine_results if r.get('success', True)]
    n_successful = len(successful_results)
    
    if n_successful == 0:
        warnings.warn("All machine simulations failed!")
        return {
            'machine_results': machine_results,
            'summary_statistics': {},
            'fleet_costs': {},
            'n_machines': n_machines,
            'n_successful': 0
        }
    
    # Compute summary statistics
    summary_stats = compute_fleet_summary(successful_results)
    
    # Compute fleet-level cost analysis
    fleet_costs = compute_fleet_costs(successful_results)
    
    return {
        'machine_results': machine_results,
        'summary_statistics': summary_stats,
        'fleet_costs': fleet_costs,
        'n_machines': n_machines,
        'n_successful': n_successful
    }


def compute_fleet_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics across all machines.
    
    Args:
        results: List of simulation results from successful machines
    
    Returns:
        Dictionary containing fleet-level summary statistics
    """
    n_machines = len(results)
    
    # Extract key metrics
    total_costs = [r['total_cost'] for r in results]
    final_levels_latent = [r['final_level_latent'] for r in results]
    final_levels_observed = [r['final_level_observed'] for r in results]
    
    # Count maintenance events by type
    perfect_pm_counts = [r['count_by_type']['perfect_pm'] for r in results]
    imperfect_pm_counts = [r['count_by_type']['imperfect_pm'] for r in results]
    cm_counts = [r['count_by_type']['cm'] for r in results]
    
    # Total events
    total_events = [len(r['events']) for r in results]
    
    summary = {
        # Cost statistics
        'total_cost': {
            'mean': np.mean(total_costs),
            'std': np.std(total_costs),
            'min': np.min(total_costs),
            'max': np.max(total_costs),
            'median': np.median(total_costs),
            'q25': np.percentile(total_costs, 25),
            'q75': np.percentile(total_costs, 75)
        },
        
        # Final degradation level statistics
        'final_level_latent': {
            'mean': np.mean(final_levels_latent),
            'std': np.std(final_levels_latent),
            'min': np.min(final_levels_latent),
            'max': np.max(final_levels_latent),
            'median': np.median(final_levels_latent)
        },
        
        'final_level_observed': {
            'mean': np.mean(final_levels_observed),
            'std': np.std(final_levels_observed),
            'min': np.min(final_levels_observed),
            'max': np.max(final_levels_observed),
            'median': np.median(final_levels_observed)
        },
        
        # Maintenance event counts
        'perfect_pm_count': {
            'mean': np.mean(perfect_pm_counts),
            'std': np.std(perfect_pm_counts),
            'total': np.sum(perfect_pm_counts)
        },
        
        'imperfect_pm_count': {
            'mean': np.mean(imperfect_pm_counts),
            'std': np.std(imperfect_pm_counts),
            'total': np.sum(imperfect_pm_counts)
        },
        
        'cm_count': {
            'mean': np.mean(cm_counts),
            'std': np.std(cm_counts),
            'total': np.sum(cm_counts)
        },
        
        'total_events': {
            'mean': np.mean(total_events),
            'std': np.std(total_events),
            'total': np.sum(total_events)
        },
        
        # Fleet-level metrics
        'n_machines': n_machines,
        'obs_time': results[0]['obs_time'],
        'strategy': results[0]['strategy']
    }
    
    return summary


def compute_fleet_costs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute fleet-level cost analysis.
    
    Args:
        results: List of simulation results from successful machines
    
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


def export_fleet_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Export fleet simulation results to a pandas DataFrame.
    
    Args:
        results: List of simulation results from machines
    
    Returns:
        DataFrame with one row per machine containing summary metrics
    """
    records = []
    
    for result in results:
        if not result.get('success', True):
            continue
        
        record = {
            'machine_id': result['machine_id'],
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
            'strategy': result['strategy'],
            'obs_time': result['obs_time']
        }
        records.append(record)
    
    return pd.DataFrame(records)


def export_all_events_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Export all maintenance events from fleet to a pandas DataFrame.
    
    Args:
        results: List of simulation results from machines
    
    Returns:
        DataFrame with one row per maintenance event
    """
    records = []
    
    for result in results:
        if not result.get('success', True):
            continue
        
        machine_id = result['machine_id']
        
        for event in result['events']:
            record = {
                'machine_id': machine_id,
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
