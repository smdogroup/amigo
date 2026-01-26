---
sidebar_position: 5
---

# Models

The `Model` class is responsible for assembling individual components into a complete optimization problem. It manages component instantiation, variable linking, code generation, and compilation.

## Creating a Model

A model is created with a unique name that will be used for generated C++ files:

```python
import amigo as am

model = am.Model("my_optimization")
```

The model name should be a valid Python identifier (no spaces or special characters).

## Adding Components

Components are added to the model using `add_component`, which takes three arguments:

```python
add_component(name, num_instances, component_object)
```

- `name`: Unique identifier for this component group
- `num_instances`: Number of instances to create
- `component_object`: Instance of your Component class

### Single Instance

For static analysis or single-point optimization:

```python
rosenbrock = Rosenbrock()
model.add_component("opt", 1, rosenbrock)
```

### Multiple Instances

For time-dependent problems, repeated structures, or spatial discretization:

```python
# Optimal control with 100 time steps
dynamics = DynamicsComponent()
model.add_component("dynamics", 100, dynamics)

# This creates 100 copies of the component
```

Each instance maintains its own variables, indexed from 0 to num_instances-1.

## Variable Linking

Linking establishes relationships between variables across components. The linking syntax uses scoped names: `component_name.variable_name[indices]`

### Basic Linking

Link two inputs together (they become the same variable):

```python
# Link inputs from two components
model.link("comp1.x", "comp2.x")
```

### Array Linking

Link array elements or slices (inputs to inputs):

```python
# Link entire array of inputs
model.link("comp1.forces[:3]", "comp2.forces[:3]")

# Link specific elements
model.link("comp1.load[0]", "comp2.load")

# Link with different indices
model.link("comp1.state[0:3]", "comp2.state[1:4]")
```

### Linking Across Instances

For time-marching or sequential processes:

```python
num_steps = 50

# Link state at end of step k to state at start of step k+1
for k in range(num_steps - 1):
    model.link(f"dynamics.q_end[{k}, :]", f"dynamics.q_start[{k+1}, :]")
```

### Linking Rules

Variables are linked between components through an explicit linking process. Indices within the model are linked with text-based linking arguments. The names provided to the linking command are scoped by `component_name.variable_name`.

**Inputs to Inputs**: Linking establishes that two inputs from different components are the same variable (shared)

```python
model.link("comp1.x", "comp2.x")
# comp1.x and comp2.x are now the same variable
```

**Outputs to Outputs**: Linking two outputs together means the sum of the two output values

```python
model.link("comp1.force", "comp2.force")
# The sum: comp1.force + comp2.force
```

**Constraints to Constraints**: Linking two constraints together means the sum of the constraint values

```python
model.link("comp1.residual", "comp2.residual")
# The sum: comp1.residual + comp2.residual
```

:::warning

You cannot link between different variable types. For instance, you cannot link inputs to constraints or outputs to objectives.

:::

## Build and Initialize

After adding all components and links, compile and initialize:

```python
# Generate and compile C++ code with automatic differentiation
model.build_module()

# Initialize the model (allocate memory, set initial values)
model.initialize()
```

### The Build Process

When `build_module()` is called:

1. Python code in `compute()` methods is analyzed
2. Optimized C++ code is generated
3. Automatic differentiation (A2D) is incorporated
4. Code is compiled to a Python extension module

Generated files:
- `{model_name}.cpp` - C++ source
- `{model_name}.h` - Header file
- `{model_name}.pyd/.so` - Compiled binary

:::warning

Whenever you modify `compute()` methods, you must call `build_module()` again to recompile.

:::

## Complete Example

```python
import amigo as am

# Define components
class Dynamics(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x", shape=2)  # Current state
        self.add_input("u")           # Control input
        self.add_input("x_next", shape=2)  # Next state
        self.add_constraint("residual", shape=2)
    
    def compute(self):
        x = self.inputs["x"]
        u = self.inputs["u"]
        x_next = self.inputs["x_next"]
        
        # Simple dynamics: x_next = f(x, u)
        dt = 0.1
        self.constraints["residual"] = [
            x_next[0] - (x[0] + dt * x[1]),
            x_next[1] - (x[1] + dt * u)
        ]

class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x_final", shape=2)
        self.add_objective("cost")
    
    def compute(self):
        x = self.inputs["x_final"]
        # Minimize distance from origin
        self.objective["cost"] = x[0]**2 + x[1]**2

# Build model
model = am.Model("trajectory")

# Add components (10 time steps)
num_steps = 10
model.add_component("dynamics", num_steps, Dynamics())
model.add_component("obj", 1, Objective())

# Link states across time steps (input to input)
for k in range(num_steps - 1):
    model.link(f"dynamics.x_next[{k}, :]", f"dynamics.x[{k+1}, :]")

# Link final state to objective (input to input)
model.link(f"dynamics.x[{num_steps-1}, :]", "obj.x_final")

# Build and initialize
model.build_module()
model.initialize()

# Now ready for optimization
opt = am.Optimizer(model)
opt.optimize()
```

## Sub-Models

You can add sub-models to a model by calling `model.add_sub_model("sub_model", sub_model)`. In this case, the scope becomes `sub_model.component_name.variable_name`. Any links specified in the sub-model are automatically added to the main model.

```python
# Create sub-model for wing analysis
wing = am.Model("wing")
wing.add_component("aero", 1, AeroComponent())
wing.add_component("structure", 1, StructureComponent())
# Link shared input between components (e.g., both need velocity)
wing.link("aero.velocity", "structure.velocity")

# Create main aircraft model
aircraft = am.Model("aircraft")
aircraft.add_sub_model("left_wing", wing)
aircraft.add_sub_model("right_wing", wing)

# Add a component that takes total weight as input
aircraft.add_component("performance", 1, PerformanceComponent())

# Link wing weight inputs to performance component (input to input)
aircraft.link("left_wing.structure.weight_input", "performance.weight_left")
aircraft.link("right_wing.structure.weight_input", "performance.weight_right")
```

Variable paths in sub-models use the format: `sub_model_name.component_name.variable_name`

