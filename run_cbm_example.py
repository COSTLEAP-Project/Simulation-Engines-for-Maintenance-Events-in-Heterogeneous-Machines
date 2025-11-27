import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from condition_based.covariates import CovariateSpec
from condition_based.cost import CostParams
from condition_based.repair import sample_post_repair_mixed
from condition_based.multi_machine_sim import simulate_multiple_machines

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("Feature-Based Multi-Machine Degradation Simulation with Cost Models")
    print("="*80)

    # =========================================================================
    # GLOBAL SIMULATION PARAMETERS
    # =========================================================================
    N_MACHINES = 10          # Number of machines to simulate
    OBS_TIME = 50.0          # Total observation time
    RANDOM_SEED_BASE = 42    # Base random seed for reproducibility
    DT = 0.1                 # Time step size
    L = 5.5                  # Catastrophic failure threshold
    X0 = 0.0                 # Initial degradation level
    
    # Dictionary to store results for different scenarios
    fleet_results = {}

    # =========================================================================
    # COST MODEL PARAMETERS (Shared across all scenarios)
    # =========================================================================
    
    # Cost Covariates (affect maintenance operation costs)
    # These are independent from degradation covariates
    cost_covariates_shared = [
        CovariateSpec(
            name="labor_cost_index",
            type="time_dependent",
            time_function=lambda t: 1.0 + 0.01 * t,  # Inflation: +1% per time unit
            noise_std=0.02
        ),
        CovariateSpec(
            name="material_availability",
            type="fixed",
            initial_value={"values": [0.9, 1.0, 1.1], "probs": [0.2, 0.6, 0.2]}
            # Each machine randomly gets one value: 0.9 (scarce), 1.0 (normal), 1.1 (abundant)
        )
    ]
    
    # Cost Model Parameters
    cost_params_shared = CostParams(
        # Perfect PM cost: Gamma(shape, scale) + location
        pm_shape=2.0,
        pm_scale=50.0,      # Base cost ~100 (mean = shape * scale)
        
        # Corrective Maintenance cost: Gamma(shape, scale) + location
        cm_shape=2.0,
        cm_scale=200.0,     # Base cost ~400 (4x higher than PM)
        
        # Imperfect PM cost: c_fix + c_0*u + ε
        c_0=80.0,           # Effectiveness coefficient
        gamma_coeffs=np.array([0.5, 0.3]),  # c_fix = exp(0.5*W1 + 0.3*W2)
        epsilon_std=8.0     # Noise standard deviation
    )
    
    # Repair Function Parameters
    repair_params_shared = {
        'p_major': 0.3,              # 30% chance of major repair
        'dist_minor': 'proportional', # Minor repair distribution type
        'rho_minor': 0.5             # Minor repair effectiveness
    }
    
    # Observation Noise Parameters
    noise_params_shared = {
        "type": "additive_normal",
        "sigma": 0.15
    }

    # =========================================================================
    # SCENARIO 1: Compound Poisson Process with Varied Maintenance Policies
    # =========================================================================
    print("\n" + "="*80)
    print("SCENARIO 1: Compound Poisson Process")
    print("="*80)
    print(f"Simulating {N_MACHINES} machines with DIFFERENT maintenance policies")
    
    # --- Degradation Model Parameters ---
    scenario1_degradation_type = "compound_poisson"
    
    scenario1_degradation_params = {
        "lambda_shock": 0.1,        # Base shock arrival rate
        "shock_dist": "exponential", # Shock magnitude distribution
        "shock_scale": 1.0          # Shock magnitude scale parameter
    }
    
    # --- Degradation Covariates ---
    scenario1_covariates = [
        CovariateSpec(
            name="temperature",
            type="time_dependent",
            initial_value=20.0,
            time_function=lambda t: 20 + 10*np.sin(t/10) + 5*np.sin(t/5),
            noise_std=1.0
        ),
        CovariateSpec(
            name="load_factor",
            type="fixed",
            initial_value={"values": [1.0, 1.2, 1.5], "probs": [0.3, 0.5, 0.2]}
            # Light (1.0), Medium (1.2), Heavy (1.5) load
        )
    ]
    
    # --- Covariate Effects on Degradation ---
    scenario1_covariate_effects = {
        "lambda_shock": np.array([0.02, 0.5])  # [temp_effect, load_effect]
        # Higher temperature and load → more frequent shocks
    }
    
    # --- Covariate Effects on Costs ---
    scenario1_cost_covariate_effects = {
        "pm_location": np.array([20.0, 15.0]),   # [temp_effect, load_effect] on Perfect PM
        "cm_location": np.array([120.0, 80.0]),  # [temp_effect, load_effect] on CM
        # Machines in harsh conditions have higher base maintenance costs
    }
    
    # --- MAINTENANCE POLICIES (Different for each machine) ---
    # Strategy: Test increasingly conservative policies
    scenario1_pm_levels = [4.0, 3.5, 4.0, 4.0, 4.0, 4.5, 4.5, 5.0, 5.0, 5.0]
    # Aggressive (3.5) → Conservative (5.0)
    
    scenario1_pm_intervals = [15, 12, 15, 15, 18, 20, 20, 25, 30, None]
    # Frequent (10) → Infrequent (30) → Level-only (None)
    
    print(f"\nMaintenance Policies:")
    print(f"  PM_level range: {min(scenario1_pm_levels)} to {max(scenario1_pm_levels)}")
    print(f"  PM_interval range: {min([x for x in scenario1_pm_intervals if x is not None])} to "
          f"{max([x for x in scenario1_pm_intervals if x is not None])}")
    print(f"  Level-only strategy: {sum(1 for x in scenario1_pm_intervals if x is None)} machines")

    # --- Run Simulation ---
    fleet_results['Scenario1_CompoundPoisson'] = simulate_multiple_machines(
        n_machines=N_MACHINES,
        degradation_type=scenario1_degradation_type,
        degradation_params=scenario1_degradation_params,
        covariate_specs=scenario1_covariates,
        covariate_effects=scenario1_covariate_effects,
        dt=DT,
        PM_level=scenario1_pm_levels,
        PM_interval=scenario1_pm_intervals,
        L=L,
        x0=X0,
        repair_func=sample_post_repair_mixed,
        repair_params=repair_params_shared,
        obs_time=OBS_TIME,
        random_seed_base=RANDOM_SEED_BASE,
        noise=noise_params_shared,
        cost_params=cost_params_shared,
        cost_covariate_specs=cost_covariates_shared,
        cost_covariate_effects=scenario1_cost_covariate_effects
    )

    # =========================================================================
    # SCENARIO 2: Gamma Process with Uniform Maintenance Policy
    # =========================================================================
    print("\n" + "="*80)
    print("SCENARIO 2: Gamma Process")
    print("="*80)
    print(f"Simulating {N_MACHINES} machines with SAME maintenance policy")
    
    # --- Degradation Model Parameters ---
    scenario2_degradation_type = "gamma"
    
    scenario2_degradation_params = {
        "alpha": 0.5,  # Shape parameter for degradation increment
        "beta": 1.0    # Scale parameter for degradation increment
    }
    
    # --- Degradation Covariates ---
    scenario2_covariates = [
        CovariateSpec(
            name="stress",
            type="time_dependent",
            time_function=lambda t: 1.0 + 0.5*np.exp(t/20)  # Exponentially increasing stress
        ),
        CovariateSpec(
            name="environment_class",
            type="fixed",
            initial_value={"values": [0, 1, 2], "probs": [0.3, 0.5, 0.2]}
            # Indoor (0), Outdoor-covered (1), Outdoor-exposed (2)
        )
    ]
    
    # --- Covariate Effects on Degradation ---
    scenario2_covariate_effects = {
        "beta": np.array([-0.1, 0.1])  # [stress_effect, environment_effect]
        # Higher stress decreases scale (faster degradation)
        # Harsher environment increases scale (faster degradation)
    }
    
    # --- Covariate Effects on Costs ---
    scenario2_cost_covariate_effects = {
        "pm_location": np.array([25.0, 15.0]),   # [stress_effect, environment_effect]
        "cm_location": np.array([100.0, 60.0]),  # [stress_effect, environment_effect]
    }
    
    # --- MAINTENANCE POLICIES (Same for all machines) ---
    scenario2_pm_level = 4.0      # Single value for all machines
    scenario2_pm_interval = 15.0  # Single value for all machines
    
    print(f"\nMaintenance Policy:")
    print(f"  PM_level: {scenario2_pm_level} (uniform across fleet)")
    print(f"  PM_interval: {scenario2_pm_interval} (uniform across fleet)")

    # --- Run Simulation ---
    fleet_results['Scenario2_Gamma'] = simulate_multiple_machines(
        n_machines=N_MACHINES,
        degradation_type=scenario2_degradation_type,
        degradation_params=scenario2_degradation_params,
        covariate_specs=scenario2_covariates,
        covariate_effects=scenario2_covariate_effects,
        dt=DT,
        PM_level=scenario2_pm_level,
        PM_interval=scenario2_pm_interval,
        L=L,
        x0=X0,
        repair_func=sample_post_repair_mixed,
        repair_params=repair_params_shared,
        obs_time=OBS_TIME,
        random_seed_base=RANDOM_SEED_BASE + 1000,
        noise = noise_params_shared,
        cost_params=cost_params_shared,
        cost_covariate_specs=cost_covariates_shared,
        cost_covariate_effects=scenario2_cost_covariate_effects
    )

    # =========================================================================
    # SCENARIO 3: Combined Process with Selective Maintenance Policies
    # =========================================================================
    print("\n" + "="*80)
    print("SCENARIO 3: Combined Process (Gamma + Compound Poisson)")
    print("="*80)
    print(f"Simulating {N_MACHINES} machines with SELECTIVE maintenance policies")
    
    # --- Degradation Model Parameters ---
    scenario3_degradation_type = "combined"
    
    scenario3_degradation_params = {
        # Base process (Gamma)
        "base_process": "gamma",
        "alpha": 0.3,
        "beta": 1.2,
        # Shock process (Compound Poisson)
        "lambda_shock": 0.05,
        "shock_dist": "exponential",
        "shock_scale": 0.8
    }
    
    # --- Degradation Covariates ---
    scenario3_covariates = [
        CovariateSpec(
            name="operational_hours",
            type="time_dependent",
            time_function=lambda t: t  # Cumulative hours
        ),
        CovariateSpec(
            name="maintenance_quality",
            type="fixed",
            initial_value=0.9  # Fixed for all machines
        ),
        CovariateSpec(
            name="cumulative_damage",
            type="path_dependent",
            path_function=lambda x: x**0.5  # Square root of degradation level
        )
    ]
    
    # --- Covariate Effects on Degradation ---
    scenario3_covariate_effects = {
        "lambda_shock": np.array([0.005, -0.3, 0.2])
        # [hours_effect, quality_effect, damage_effect]
    }
    
    # --- Covariate Effects on Costs ---
    scenario3_cost_covariate_effects = {
        "pm_location": np.array([15.0, 10.0, 20.0]),
        "cm_location": np.array([80.0, 40.0, 100.0]),
    }
    
    # --- MAINTENANCE POLICIES (Dictionary-based: specific machines get custom values) ---
    scenario3_pm_level = {
        0: 3.0,   # Machine 0: Very aggressive
        1: 3.0,   # Machine 1: Very aggressive
        5: 5.0,   # Machine 5: Very conservative
        9: 5.0    # Machine 9: Very conservative
        # Others use default: 4.0
    }
    
    scenario3_pm_interval = {
        0: 10,    # Machine 0: Frequent maintenance
        1: 10,    # Machine 1: Frequent maintenance
        5: None,  # Machine 5: Level-only strategy
        9: None   # Machine 9: Level-only strategy
        # Others use default: 15
    }

    # --- Run Simulation ---
    fleet_results['Scenario3_Combined'] = simulate_multiple_machines(
        n_machines=N_MACHINES,
        degradation_type=scenario3_degradation_type,
        degradation_params=scenario3_degradation_params,
        covariate_specs=scenario3_covariates,
        covariate_effects=scenario3_covariate_effects,
        dt=DT,
        PM_level=scenario3_pm_level,
        PM_interval=scenario3_pm_interval,
        L=L,
        x0=X0,
        repair_func=sample_post_repair_mixed,
        repair_params=repair_params_shared,
        obs_time=OBS_TIME,
        random_seed_base=RANDOM_SEED_BASE + 2000,
        noise = noise_params_shared,
        cost_params=cost_params_shared,
        cost_covariate_specs=cost_covariates_shared,
        cost_covariate_effects=scenario3_cost_covariate_effects
    )

    # =========================================================================
    # RESULTS ANALYSIS
    # =========================================================================
    
    print("\n" + "="*80)
    print("FLEET COST SUMMARY")
    print("="*80)
    
    for scenario_name, fleet_res in fleet_results.items():
        print(f"\n{scenario_name}:")
        print("-"*80)
        
        fleet_costs = fleet_res['fleet_costs']
        summary_stats = fleet_res['summary_statistics']
        policy_summary = fleet_res['policy_summary']
        
        print(f"Number of machines: {fleet_res['n_machines']}")
        print(f"Observation time: {summary_stats['obs_time']}")
        
        # Print policy summary
        print(f"\nMaintenance Policy Summary:")
        print(f"  Level-only strategy: {policy_summary['n_level_only']} machines")
        print(f"  Time-and-level strategy: {policy_summary['n_time_and_level']} machines")
        print(f"  PM_level - Min: {policy_summary['PM_level_stats']['min']:.1f}, "
              f"Max: {policy_summary['PM_level_stats']['max']:.1f}, "
              f"Mean: {policy_summary['PM_level_stats']['mean']:.2f}")
        if policy_summary['PM_interval_stats']['mean'] is not None:
            print(f"  PM_interval - Min: {policy_summary['PM_interval_stats']['min']:.1f}, "
                  f"Max: {policy_summary['PM_interval_stats']['max']:.1f}, "
                  f"Mean: {policy_summary['PM_interval_stats']['mean']:.2f}")
        
        # Print cost summary
        print(f"\nCost Summary:")
        print(f"  Total fleet cost: {fleet_costs['total_fleet_cost']:.2f}")
        print(f"  Average cost per machine: {fleet_costs['average_cost_per_machine']:.2f}")
        print(f"  Cost rate per machine: {fleet_costs['cost_rate_per_machine']:.2f} per unit time")
        
        print(f"\n{'Maintenance Type':<30} {'Count':>10} {'Total Cost':>15} {'%':>8} {'Avg/Event':>12}")
        print("-"*80)
        
        type_names = {
            'perfect_pm': 'Perfect PM',
            'imperfect_pm': 'Imperfect PM',
            'cm': 'Corrective Maintenance'
        }
        
        for mtype in ['perfect_pm', 'imperfect_pm', 'cm']:
            count = fleet_costs['event_counts'][mtype]
            total = fleet_costs['cost_by_type'][mtype]
            pct = fleet_costs['cost_percentage_by_type'][mtype]
            avg = fleet_costs['average_cost_per_event'][mtype]
            
            print(f"{type_names[mtype]:<30} {count:>10} {total:>15.2f} {pct:>7.1f}% {avg:>12.2f}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # --- Plot 1: Sample Degradation Paths ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    
    colors = {
        'imperfect_repair': 'green',
        'perfect_preventive_maintenance': 'cyan',
        'catastrophic_failure_replacement': 'red'
    }
    markers = {
        'imperfect_repair': 'o',
        'perfect_preventive_maintenance': 's',
        'catastrophic_failure_replacement': 'X'
    }
    labels = {
        'imperfect_repair': 'Imperfect repair',
        'perfect_preventive_maintenance': 'Perfect PM',
        'catastrophic_failure_replacement': 'Perfect CM'
    }
    
    for ax, (scenario_name, fleet_res) in zip(axes, fleet_results.items()):
        # Get first machine result
        machine_0 = fleet_res['machine_results'][0]
        
        t = machine_0['times']
        y_lat = machine_0.get('degra_level_latent', None)
        y_obs = machine_0.get('degra_level_observed', None)
    
        # Plot observed (noisy)
        if y_obs is not None and len(y_obs) == len(t):
            ax.plot(t, y_obs, '-', linewidth=1.8, color='tab:blue',
                    label='Observed degradation')
    
        # Plot latent (true)
        if y_lat is not None and len(y_lat) == len(t):
            ax.plot(t, y_lat, '--', linewidth=1.6, color='0.35', alpha=0.9,
                    label='Latent degradation')
    
        # Event markers
        used_labels = set()
        for ev in machine_0['events']:
            etype = ev['type']
            color = colors[etype]
            marker = markers[etype]
            y_ev = ev.get('level_after_observed',
                          ev.get('level_after_latent', np.nan))
            lbl = labels[etype] if etype not in used_labels else ""
            if etype not in used_labels:
                used_labels.add(etype)
            ax.plot(ev['time'], y_ev, marker=marker, color=color,
                    markersize=8, label=lbl)
    
        # Thresholds
        pm_level = machine_0.get('PM_level', 4.0)
        ax.axhline(pm_level, color="orange", linestyle="--", linewidth=2, 
                   label=f"PM threshold ({pm_level})")
        ax.axhline(L, color="red", linestyle=":", linewidth=2, 
                   label=f"Failure threshold ({L})")
    
        ax.set_ylabel("Degradation level", fontsize=12)
        
        total_cost = machine_0['total_cost']
        pm_int = machine_0.get('PM_interval', 'None')
        ax.set_title(f"{scenario_name} - Machine 0\n"
                     f"PM_level={pm_level}, PM_interval={pm_int}, Total Cost={total_cost:.2f}", 
                     fontsize=14)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time", fontsize=12)
    plt.tight_layout()
    plt.savefig('degradation_paths_sample.png', dpi=150, bbox_inches='tight')
    print("✓ Sample degradation paths saved to 'degradation_paths_sample.png'")
    
    # --- Plot 2: Cost vs Policy Analysis for Scenario 1 ---
    scenario1_df = export_machine_level_df(
        fleet_results['Scenario1_CompoundPoisson']['machine_results']
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost vs PM_level
    ax = axes[0]
    scatter = ax.scatter(scenario1_df['PM_level'], scenario1_df['total_cost'], 
                        c=scenario1_df['n_cm'], cmap='YlOrRd', s=100, alpha=0.7,
                        edgecolors='black')
    ax.set_xlabel('PM Level', fontsize=12)
    ax.set_ylabel('Total Cost', fontsize=12)
    ax.set_title('Total Cost vs PM Level\n(Color = Number of CM events)', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='CM Count')
    
    # Cost vs PM_interval
    ax = axes[1]
    # Filter out None values
    mask = scenario1_df['PM_interval'].notna()
    scatter = ax.scatter(scenario1_df.loc[mask, 'PM_interval'], 
                        scenario1_df.loc[mask, 'total_cost'],
                        c=scenario1_df.loc[mask, 'n_cm'], cmap='YlOrRd', s=100, alpha=0.7,
                        edgecolors='black')
    ax.set_xlabel('PM Interval', fontsize=12)
    ax.set_ylabel('Total Cost', fontsize=12)
    ax.set_title('Total Cost vs PM Interval\n(Color = Number of CM events)', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='CM Count')
    
    plt.tight_layout()
    #plt.savefig('cost_vs_policy_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Cost vs policy analysis saved to 'cost_vs_policy_analysis.png'")
    
    plt.show()
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*80)
