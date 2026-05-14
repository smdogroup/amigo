import json
import amigo as am

# Set the filenames
filename = "cart_pole_model.json"

# Load the model
with open(filename, "r") as fp:
    data = json.load(fp)
model = am.Model.deserialize(data)

# Build the model module
model.build_module()

# Initialize everything
model.initialize()

# Extract the other vectors for the initial point and lower/upper bounds
x = model.get_values_from_meta("value")
lower = model.get_values_from_meta("lower")
upper = model.get_values_from_meta("upper")

# Set up the optimizer
opt = am.Optimizer(model, x, lower=lower, upper=upper, solver="amigo")

opt_options = {
    "initial_barrier_param": 1.0,
    "max_iterations": 100,
    "max_line_search_iterations": 30,
    "convergence_tolerance": 1e-8,
    "init_least_squares_multipliers": True,
    "barrier_strategy": "quality_function",
    "quality_function_predictor_corrector": False,
    "quality_function_balancing_term": "cubic",
    "adaptive_mu_safeguard_factor": 1e-1,
    "filter_line_search": True,
    "second_order_correction": False,
    "verbose_barrier": False,
}

# Optimize
opt_data = opt.optimize(opt_options)
