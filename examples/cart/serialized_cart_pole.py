import json
import amigo as am
from pathlib import Path


source_dir = Path(__file__).resolve().parent

# Set the filenames
filename = "cart_pole_model.json"
vecfile = "cart_pole_vectors.json"

# Load the model
with open(filename, "r") as fp:
    data = json.load(fp)
model = am.Model.deserialize(data)

# Build the model module
model.build_module(source_dir=source_dir)

# Initialize everything
model.initialize()

# Extract the vectors
with open(vecfile, "r") as fp:
    data = json.load(fp)
vecs = model.deserialize_vectors(data)

# Set the data into the data vector itself
data = model.get_data_vector()
data[:] = vecs["data"][:]

# Extract the other vectors for the initial point and lower/upper bounds
x = vecs["x"]
lower = vecs["lower"]
upper = vecs["upper"]

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
