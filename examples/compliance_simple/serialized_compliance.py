import json
import amigo as am
from pathlib import Path


# Set the filenames
filename = "compliance_model.json"
vecfile = "compliance_vectors.json"

# Load the model
with open(filename, "r") as fp:
    data = json.load(fp)
model = am.Model.deserialize(data)

# Build the model module
source_dir = Path(__file__).resolve().parent
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

options = {
    "max_iterations": 50,
    "initial_barrier_param": 0.1,
    "max_line_search_iterations": 1,
}

# Optimize
opt_data = opt.optimize(options)
