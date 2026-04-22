"""Default options for the Amigo interior-point optimizer.

Holds default values for all user-facing settings: barrier strategy,
line search, filter, watchdog, convergence tolerances, regularization,
inertia correction, quality function, and NLP scaling.
"""


def get_default_options(options={}):
    """Return the merged options dict with all defaults.

    Parameters
    ----------
    options : dict
        User-provided overrides.

    Returns
    -------
    dict
        Complete options with defaults filled in.
    """
    default = {
        "max_iterations": 100,
        "barrier_strategy": "heuristic",  # "heuristic", "monotone", "quality_function"
        "monotone_barrier_fraction": 0.1,
        "convergence_tolerance": 1e-8,
        "dual_inf_tol": 1.0,
        "constr_viol_tol": 1e-4,
        "compl_inf_tol": 1e-4,
        "diverging_iterates_tol": 1e20,
        "fraction_to_boundary": 0.99,
        "initial_barrier_param": 1.0,
        "max_line_search_iterations": 40,
        "check_update_step": False,
        "backtracking_factor": 0.5,
        "record_components": [],
        "equal_primal_dual_step": False,
        "init_least_squares_multipliers": True,
        "init_affine_step_multipliers": False,
        # Heuristic barrier parameter options
        "heuristic_barrier_gamma": 0.1,
        "heuristic_barrier_r": 0.95,
        "verbose_barrier": False,
        "continuation_control": None,
        # Regularization options
        "regularization_eps_x": 1e-8,  # Primal regularization
        "regularization_eps_z": 1e-8,  # Dual regularization
        "adaptive_regularization": True,  # Increase regularization on failure
        "max_regularization": 1e-2,  # Maximum regularization value
        "regularization_increase_factor": 10.0,  # Factor to increase regularization
        # Line search options
        "armijo_constant": 1e-4,  # Armijo sufficient decrease constant
        "use_armijo_line_search": True,  # Use Armijo condition
        "second_order_correction": True,
        # Filter line search: bi-objective (theta, psi) acceptance.
        # Accepts steps improving feasibility OR merit, with feasibility
        # restoration when both fail.
        "filter_line_search": True,
        "filter_gamma_theta": 1e-5,
        "filter_gamma_phi": 1e-8,
        "filter_delta": 1.0,
        "filter_s_theta": 1.1,
        "filter_s_phi": 2.3,
        "filter_eta_phi": 1e-8,
        "filter_max_soc": 4,
        "filter_kappa_soc": 0.99,
        "filter_restoration_max_iter": 20,  # Max restoration iterations
        "filter_reset_trigger": 5,  # Consecutive filter rejections before reset
        "max_filter_resets": 5,  # Maximum number of filter resets per subproblem
        # Watchdog procedure
        "watchdog_shortened_iter_trigger": 10,  # 0 = disable watchdog
        "watchdog_trial_iter_max": 3,  # Max trial iterations in watchdog
        # Acceptable convergence
        "acceptable_tol": 1e-6,
        "acceptable_iter": 15,
        "acceptable_dual_inf_tol": 1e10,
        "acceptable_constr_viol_tol": 1e-2,
        "acceptable_compl_inf_tol": 1e-2,
        # Advanced features
        "adaptive_tau": True,  # Adaptive fraction-to-boundary
        "tau_min": 0.99,  # Minimum tau value
        "progress_based_barrier": True,  # Only reduce barrier when making progress
        "barrier_progress_tol": 10.0,  # kappa_epsilon factor for barrier subproblem tolerance
        # Variable-specific regularization for zero-Hessian (linearly-appearing) variables
        "zero_hessian_variables": [],  # Variable names (e.g., ["dyn.qdot"])
        "regularization_eps_x_zero_hessian": 1.0,  # Strong eps_x for those variables
        # Algorithm IC inertia correction (auto-detected from solver)
        "max_consecutive_rejections": 5,  # Before barrier increase
        "barrier_increase_factor": 5.0,  # Barrier *= this when stuck
        "max_inertia_corrections": 40,
        "inertia_tolerance": 0,  # Allow n_pos/n_neg to differ by this many from expected
        # Adaptive barrier (quality_function strategy)
        "mu_max_fact": 1e3,  # mu_max = mu_max_fact * initial_avg_comp
        "mu_min": 1e-11,  # absolute floor for mu
        "mu_linear_decrease_factor": 0.2,  # kappa_mu for monotone decrease
        "mu_superlinear_decrease_power": 1.5,  # theta_mu for superlinear
        "barrier_tol_factor": 10.0,  # kappa_eps: reduce when E_mu <= kappa_eps*mu
        "adaptive_mu_globalization": "obj-constr-filter",
        "adaptive_mu_kkterror_red_iters": 4,  # l_max reference values
        "adaptive_mu_kkterror_red_fact": 0.9999,  # kappa_red
        "adaptive_mu_monotone_init_factor": 0.8,  # mu_bar = factor * avg_comp
        "adaptive_mu_safeguard_factor": 0.0,  # 0 = disabled
        "quality_function_sigma_max": 100.0,
        "quality_function_sigma_min": 1e-6,
        "quality_function_section_sigma_tol": 1e-2,
        "quality_function_section_qf_tol": 0.0,
        "quality_function_golden_iters": 8,
        "quality_function_norm_scaling": True,
        "quality_function_centrality": "none",  # "none","log","reciprocal","cubed-reciprocal"
        "quality_function_balancing_term": "none",  # "none","cubic"
        "quality_function_predictor_corrector": False,
        "nlp_scaling_max_gradient": 100.0,
        "debug_cycling": False,
    }

    for name in options:
        if name in default:
            default[name] = options[name]
        else:
            raise ValueError(f"Unrecognized option {name}")

    return default
