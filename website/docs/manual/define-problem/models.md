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

Link scalar variables:

```python
# Output of comp1 becomes input to comp2
model.link("comp1.output", "comp2.input")
```

### Array Linking

Link array elements or slices:

```python
# Link entire array
model.link("comp1.forces[:3]", "comp2.loads[:3]")

# Link specific elements
model.link("comp1.stress[0]", "comp2.load")

# Link with different indices
model.link("comp1.state[0:3]", "comp2.input[1:4]")
```

### Linking Across Instances

For time-marching or sequential processes:

```python
num_steps = 50

# Link state at time k+1 to previous state at time k
for k in range(num_steps):
    model.link(f"dynamics.q[{k+1}, :]", f"dynamics.q_prev[{k}, :]")
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
class Source(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x", value=1.0, lower=0.0, upper=10.0)
        self.add_output("y")
    
    def compute(self):
        self.outputs["y"] = self.inputs["x"]**2

class Processor(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("z")
        self.add_output("w")
        self.add_objective("cost")
    
    def compute(self):
        z = self.inputs["z"]
        self.outputs["w"] = am.sqrt(z)
        self.objective["cost"] = z

# Build model
model = am.Model("pipeline")

# Add components
model.add_component("source", 1, Source())
model.add_component("proc", 3, Processor())

# Link components
model.link("source.y", "proc.z[0]")
for i in range(2):
    model.link(f"proc.w[{i}]", f"proc.z[{i+1}]")

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
wing.link("aero.pressure", "structure.loads")

# Create main aircraft model
aircraft = am.Model("aircraft")
aircraft.add_sub_model("left_wing", wing)
aircraft.add_sub_model("right_wing", wing)

# Link between sub-models
aircraft.link("left_wing.structure.weight", "total_weight")
aircraft.link("right_wing.structure.weight", "total_weight")
```

Variable paths in sub-models use the format: `sub_model_name.component_name.variable_name`

