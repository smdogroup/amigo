"""
Generate cart-pole computational graph for documentation
Run this after building the cart-pole model
"""
import sys
sys.path.insert(0, '../../..')  # Add amigo to path

import amigo as am
import numpy as np

# Import the cart-pole components
sys.path.insert(0, '../../../examples/cart')
from cart_pole import create_cart_model

# Create the model
model = create_cart_model()
model.build_module()
model.initialize()

# Generate graph for specific timesteps
graph = model.create_graph(timestep=[0, 5, 10])

# Create visualization
from pyvis.network import Network

net = Network(
    notebook=False,
    height="1000px",
    width="100%",
    bgcolor="#ffffff",
    font_color="black"
)

net.from_nx(graph)
net.set_options("""
var options = {
    "interaction": {
        "dragNodes": false
    }
}
""")

# Save to static/img directory for documentation
net.show("cart_pole_graph.html")

print("Graph generated: cart_pole_graph.html")
print("Move this file to website/static/img/ to embed in documentation")

