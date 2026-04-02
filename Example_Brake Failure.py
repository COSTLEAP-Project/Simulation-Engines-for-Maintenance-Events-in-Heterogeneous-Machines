import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from condition_based.covariates import CovariateSpec
from condition_based.cost import CostParams
from condition_based.repair import sample_post_repair_mixed
from condition_based.multi_machine_sim import simulate_multiple_machines, export_machine_level_df

if __name__ == "__main__":

    print("\n" + "="*80)
    print("BRAKE WEAR DEGRADATION DATA GENERATION")
    print("="*80)

    # =========================================================================
    # GLOBAL PARAMETERS
    # =========================================================================
    N_VEHICLES       = 30
    OBS_TIME         = 450.0   # ~3 expected lifetimes
    RANDOM_SEED_BASE = 42
    DT               = 1.0     # 1 step = 1 day
    X0               = 0.0
    L                = 10.0    # failure threshold

    # =========================================================================
    # DEGRADATION MODEL
    # =========================================================================
    # Gamma process — monotonically increasing wear, appropriate for brake pads
    # Baseline (no covariate boost):
    #   E[X(t)] = alpha * beta * t
  
    degradation_type = "gamma"
    degradation_params = {
        "alpha": 0.180,   # up from 0.128
        "beta":  0.231,
    }

    # =========================================================================
    # DAILY SCHEDULE BUILDER  (route + speed + load, per vehicle)
    # =========================================================================
    def build_vehicle_schedules(obs_time, dt, vehicle_seed,
                                p_highway=0.3, p_city=0.7):
        """
        Return (route_sched, speed_sched, load_sched) as flat lists,
        one entry per time step, ready for discrete_series.

        route: 0 = highway, 1 = city
        p_highway / p_city: daily route probabilities (must sum to 1)
        """
        rng           = np.random.default_rng(vehicle_seed)
        n_steps       = int(obs_time / dt)
        steps_per_day = max(1, int(1.0 / dt))
        n_days        = int(np.ceil(obs_time))

        # vehicle-specific load mean — fixed for this vehicle's lifetime
        load_mu = float(np.clip(rng.normal(0.6, 0.08), 0.0, 1.0))

        route_list, speed_list, load_list = [], [], []

        for _ in range(n_days):
            route = int(rng.choice([0, 1], p=[p_highway, p_city]))
            if route == 0:                          # highway
                speed = float(np.clip(rng.normal(100, 7),  10, 120))
            else:                                   # city
                speed = float(np.clip(rng.normal(30,  5),  10, 120))
            load  = float(np.clip(rng.normal(load_mu, 0.10), 0.0, 1.0))

            route_list.extend([float(route)] * steps_per_day)
            speed_list.extend([speed]        * steps_per_day)
            load_list.extend([load]          * steps_per_day)

        return (route_list[:n_steps],
                speed_list[:n_steps],
                load_list[:n_steps])

    # =========================================================================
    # BUILD PER-MACHINE COVARIATE SPECS UPFRONT
    # =========================================================================
    print(f"\nBuilding per-vehicle covariate schedules ...")

    all_vehicle_cov_specs  = []
    all_vehicle_cost_specs = []

    for i in range(N_VEHICLES):
        vehicle_seed = RANDOM_SEED_BASE + i

        route_sched, speed_sched, load_sched = build_vehicle_schedules(
            OBS_TIME, DT, vehicle_seed,
            p_highway=0.3, p_city=0.7
        )

        # degradation covariates
        # Covariate vector order (must match covariate_effects array below):
        #   [region, route, speed, load,
        #    brake_temperature, vibration_level, wear_acceleration]
        #
        # region            — fixed categorical: 0/1/2/3, probs [0.2,0.4,0.3,0.1]
        #                     systematic road quality / weather / landscape stress
        #
        # route             — discrete_series: 0=highway, 1=city
        #                     city stop-and-go braking >> highway cruising
        #
        # speed             — discrete_series, conditioned on route
        #                     higher speed → more kinetic energy per braking event
        #
        # load              — discrete_series, daily fluctuation around vehicle μ
        #                     heavier vehicle → greater braking force required
        #
        # brake_temperature — path_dependent, centred: 40*(x/L)^1.5
        #                     excess temperature above 80°C resting baseline
        #                     rises steadily, accelerates approaching failure
        #                     noise_std=3.0 reflects thermocouple sensor noise
        #
        # vibration_level   — path_dependent, centred: 2.5*(x/L)^2
        #                     excess vibration above 1.0 resting baseline
        #                     quadratic — stays near zero early, spikes sharply late
        #                     noise_std=0.15 reflects accelerometer noise floor
        #                     complementary shape to brake_temperature: the two
        #                     sensors carry different information across the lifetime
        #
        # wear_acceleration — path_dependent, internal rate-shaping variable
        #                     NOT an observable sensor — purely a modelling device
        #                     = 0 exactly when x < 0.3*L  (first 30% of life)
        #                     grows as (x/L - 0.3)^2 thereafter
        #                     with large coefficient β=2 this suppresses early
        #                     wear rate and creates the "slow start, fast finish" shape

        all_vehicle_cov_specs.append([
            CovariateSpec(
                name="region",
                type="fixed",
                initial_value={"values": [0.0, 1.0, 2.0, 3.0],
                               "probs":  [0.20, 0.40, 0.30, 0.10]}
            ),
            CovariateSpec(
                name="route",
                type="discrete_series",
                values=route_sched
            ),
            CovariateSpec(
                name="speed",
                type="discrete_series",
                values=speed_sched
            ),
            CovariateSpec(
                name="load",
                type="discrete_series",
                values=load_sched
            ),
            CovariateSpec(
                name="brake_temperature",
                type="path_dependent",
                path_function=lambda x: 40 * (x / L) ** 1.5,
                noise_std=3.0,
                lower_bound=0.0
            ),
            CovariateSpec(
                name="vibration_level",
                type="path_dependent",
                path_function=lambda x: 2.5 * (x / L) ** 2,
                noise_std=0.15,
                lower_bound=0.0
            ),
            CovariateSpec(
                name="wear_acceleration",
                type="path_dependent",
                path_function=lambda x: max(0.0, (x / L - 0.3)) ** 2.0,
                # = 0.000  when x < 0.3*L  (first 30% of life, genuinely flat)
                # = 0.040  when x = 0.5*L  (mid-life, just starting)
                # = 0.160  when x = 0.7*L  (noticeable)
                # = 0.360  when x = 0.9*L  (clearly faster)
                # = 0.490  when x = L      (at failure)
                noise_std=0.0
            ),
        ])

        # cost covariates
        all_vehicle_cost_specs.append([
            CovariateSpec(
                name="labor_cost_index",
                type="time_dependent",
                time_function=lambda t: 1.0 + 0.003 * t,
                noise_std=0.02
            ),
            CovariateSpec(
                name="parts_availability",
                type="fixed",
                initial_value={"values": [0.9, 1.0, 1.1],
                               "probs":  [0.2,  0.6,  0.2]}
            ),
        ])

    # =========================================================================
    # COVARIATE EFFECTS  β in exp(β · Z)
    # =========================================================================

    covariate_effects = {
        "alpha": np.array([
            0.08,    # region
            0.10,    # route
            0.003,   # speed
            0.15,    # load
            0.020,   # brake_temperature
            0.15,    # vibration_level
            2.0,    # wear_acceleration
        ])
    }

    # =========================================================================
    # REPAIR PARAMETERS
    # =========================================================================
    repair_params = {
        "p_major":    0.15,
        "dist_minor": "beta",
        "a_minor":    2.0,
        "b_minor":    4.0,
        "dist_major": "beta",
        "a_major":    1.5,
        "b_major":    5.0,
    }

    # =========================================================================
    # COST MODEL
    # =========================================================================
    cost_params = CostParams(
        pm_shape=2.0,  pm_scale=150.0,
        cm_shape=2.0,  cm_scale=600.0,
        c_0=200.0,
        gamma_coeffs=np.array([0.15, 0.10]),
        epsilon_std=20.0
    )

    cost_covariate_effects = {
        "pm_location": np.array([20.0, 10.0]),
        "cm_location": np.array([80.0, 40.0]),
    }

    # =========================================================================
    # OBSERVATION NOISE
    # =========================================================================
    noise_params = {
        "type":  "additive_normal",
        "sigma": 0.15,
    }

    # =========================================================================
    # PHASE 1: RUN-TO-FAILURE (No Maintenance)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: RUN-TO-FAILURE SIMULATION")
    print("="*80)
    print("Generating baseline failure data for RUL model training...")

    phase1_results = simulate_multiple_machines(
        n_machines              = N_VEHICLES,
        degradation_type        = degradation_type,
        degradation_params      = degradation_params,
        covariate_specs_list    = all_vehicle_cov_specs,
        covariate_effects       = covariate_effects,
        dt                      = DT,
        PM_level                = L + 100,   # above failure → no PM occurs
        PM_interval             = None,
        L                       = L,
        x0                      = X0,
        repair_func             = None,
        repair_params           = None,
        obs_time                = OBS_TIME,
        random_seed_base        = RANDOM_SEED_BASE,
        noise                   = noise_params,
        cost_params             = None,
        cost_covariate_specs_list  = None,
        cost_covariate_effects     = None,
    )

    phase1_summary = phase1_results['summary_statistics']
    phase1_failure_rate = phase1_summary['cm_count']['total'] / N_VEHICLES

    print(f"\nPhase 1 Results:")
    print(f"  Total failures:    {phase1_summary['cm_count']['total']}")
    print(f"  Failure rate:      {phase1_failure_rate:.1%}")
    print(f"  Avg failures/veh:  {phase1_summary['cm_count']['mean']:.2f}")
