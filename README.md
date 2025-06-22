# Amigo: A friendly library for MDO on HPC

Amigo is a python library that is designed for solving multidisciplinary analysis and optimization problems with high-performance computing resources through automatically generated c++ wrappers. 

All application code is written in python and automatically compiled to c++. Automatic differentiation is used throughout to evaluate first and second derivatives using A2D. Different backends can be used: Serial, OpenMP and CUDA for Nvidia GPUs. The user written python code is independent of the backend used.

## A simple example

To illustrate some of the features of Amigo, below is some code from the cart pole optimization example.

The system dynamics are encoded within a `CartComponent` class that inherits from `amigo.Component`. In the constructor, you must specify what `constants`, `inputs`, `outputs` and `data` (not illustrated in this example) the component requires.

The only class member function that is used by Amigo is the `compute` function. The `inputs`, `outputs`, `constants` and `data` must be extracted from the dictionary-like member objects in the compute function. These are not numerical objects, but instead encode the mathematical operations that are reinterpreted to generate c++ code.

In some applications it can be simpler to create multiple versions of the compute function. In this case, you can set `amigo.Component.set_compute_args()`. which takes a list of dictionaries of keyword arguments, which will provide the compute function the provided keyword arguments.

```python
import amigo as am

class CartComponent(am.Component):
    def __init__(self):
        super().__init__()

        # Set constant values (these are compiled into static constexpr values in c++)
        self.add_constant("g", value=9.81)
        self.add_constant("L", value=0.5)
        self.add_constant("m1", value=1.0)
        self.add_constant("m2", value=0.3)

        # Input values specify variables that are under the control of the optimizer
        self.add_input("x", lower=-50, upper=50, units="N", label="control")
        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")

        # Output values are constraints within the optimization problem
        self.add_output("res", shape=(4), lower=0.0, upper=0.0, label="residual")

        return

    def compute(self):
        # The compute functions take 
        g = self.constants["g"]
        L = self.constants["L"]
        m1 = self.constants["m1"]
        m2 = self.constants["m2"]

        # Extract the input objects
        x = self.inputs["x"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Compute intermediate variables
        self.vars["sint"] = am.sin(q[1])
        self.vars["cost"] = am.cos(q[1])

        # Extract a reference to the variables (this is required to use these variables)
        sint = self.vars["sint"]
        cost = self.vars["cost"]

        # Compute the residual
        res = 4 * [None]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]
        res[2] = (m1 + m2 * (1.0 - cost * cost)) * qdot[2] - (
            L * m2 * sint * q[3] * q[3] * x + m2 * g * cost * sint
        )
        res[3] = L * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] + (
            L * m2 * cost * sint * q[3] * q[3] + x * cost + (m1 + m2) * g * sint
        )

        # Set the output
        self.outputs["res"] = res

        return
```

To create a model, you create the components, and specify how many instances of that component are used within the component group.

Variables are linked between components 

```python
# Create instances of the component classes
cart = CartComponent()
ic = InitialConditions()
fc = FinalConditions()

# Specify the module name
module_name = "cart_pole"
model = am.Model(module_name)

# Add the component classes to the model
model.add_component("cart", num_time_steps + 1, cart)
model.add_component("ic", 1, ic)
model.add_component("fc", 1, fc)
```

Indices within the model are linked with text-based linking arguments. The names provided to the linking command are scoped by `component_name.variable_name`. 

You can also add sub-models to the model by calling `model.sub_model("sub_mudel", sub_model)`. In this case the scope becomes `sub_model.component_name.variable_name`. In this case, any links specified in the sub-model are added to the model.

Linking establishes that two `inputs` from different components are the same. You cannot link `inputs` to `outputs`. Linking two `outputs` together means that the sum of the two output values are used as the constraint.

```python
# Link the initial and final conditions
model.link("cart.q[0, :]", "ic.q[0, :]")
model.link(f"cart.q[{num_time_steps}, :]", "fc.q[0, :]")

# After variables are all linked, initialize the model
model.initialize()
```

Source and target indices can be supplied for more general linking relationships.
