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
opt = am.Optimizer(model, x, lower=lower, upper=upper)

opt_options = {
    "initial_barrier_param": 0.1,
    "convergence_tolerance": 5e-7,
    "max_line_search_iterations": 4,
    "max_iterations": 500,
}

# Optimize
opt_data = opt.optimize(opt_options)
