---
sidebar_position: 3
---

# Model API

The `Model` class assembles components into a complete optimization problem.

## Class: `amigo.Model`

### Constructor

```python
amigo.Model(name, backend="serial")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Model name (used for generated files) |
| `backend` | str | Computational backend: "serial", "openmp", "mpi", "cuda" |

**Example:**

```python
import amigo as am

# Serial execution (default)
model = am.Model("my_model")

# OpenMP parallel
model = am.Model("my_model", backend="openmp")

# MPI distributed
model = am.Model("my_model", backend="mpi")
```

## Adding Components

### `add_component()`

Add component instances to the model.

```python
model.add_component(name, num_instances, component)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Component name in model |
| `num_instances` | int | Number of component instances |
| `component` | Component | Component object |

**Example:**

```python
class MyComponent(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.0)
        self.add_output("y")
    
    def compute(self):
        self.outputs["y"] = 2 * self.inputs["x"]

model = am.Model("example")

# Single instance
model.add_component("comp", 1, MyComponent())

# Multiple instances (e.g., for time steps)
model.add_component("dynamics", 100, DynamicsComponent())
```

## Linking Variables

### `link()`

Connect variables between components.

```python
model.link(source, target, src_indices=None, tgt_indices=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | str | Source variable path |
| `target` | str | Target variable path |
| `src_indices` | list | Source indices (optional) |
| `tgt_indices` | list | Target indices (optional) |

**Variable Path Format:**

```
component_name.variable_name[indices]
```

**Examples:**

```python
# Link scalar variables
model.link("comp1.output", "comp2.input")

# Link array elements
model.link("comp1.forces[0]", "comp2.load")

# Link array slices
model.link("comp1.states[0:3]", "comp2.inputs[0:3]")

# Link with explicit indices
model.link("comp1.forces", "comp2.loads",
          src_indices=[0, 2, 4],
          tgt_indices=[1, 3, 5])

# Link across time steps
for k in range(num_steps):
    model.link(f"dynamics.state[{k+1}, :]", 
              f"dynamics.state_prev[{k}, :]")
```

### Linking Rules

**Inputs to Inputs:**
Links design variables (they become the same variable)

```python
# x1 and x2 will be the same variable
model.link("comp1.x1", "comp2.x2")
```

**Outputs to Outputs:**
Sum the outputs

```python
# total = y1 + y2
model.link("comp1.y1", "total")
model.link("comp2.y2", "total")
```

**Constraints to Constraints:**
Sum the constraints

```python
# g_total = g1 + g2
model.link("comp1.g1", "constraint_sum")
model.link("comp2.g2", "constraint_sum")
```

:::warning
Cannot link between different variable types (e.g., inputs to constraints).
:::

## Sub-Models

### `add_sub_model()`

Add hierarchical sub-models.

```python
model.add_sub_model(name, sub_model)
```

**Example:**

```python
# Create sub-model
wing_model = am.Model("wing")
wing_model.add_component("aero", 1, AeroComponent())
wing_model.add_component("structure", 1, StructureComponent())
wing_model.link("aero.pressure", "structure.loads")

# Add to main model
main_model = am.Model("aircraft")
main_model.add_sub_model("wing", wing_model)
main_model.add_sub_model("fuselage", fuselage_model)

# Link between sub-models
main_model.link("wing.aero.lift", "fuselage.loads.lift_force")
```

Variable paths in sub-models:
```
sub_model_name.component_name.variable_name
```

## Building and Initialization

### `build_module()`

Compile Python code to C++ with automatic differentiation.

```python
model.build_module(options=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `options` | dict | Compilation options (optional) |

**Options:**

```python
options = {
    'optimization_level': 3,      # -O3 optimization
    'verbose': True,              # Print compilation details
    'keep_source': True,          # Keep generated C++ files
    'compiler': 'g++',            # C++ compiler
}

model.build_module(options=options)
```

**Generated Files:**

- `{name}.cpp` - C++ source code
- `{name}.h` - Header file
- `{name}.pyd` or `{name}.so` - Compiled Python extension

:::tip
Call `build_module()` again after modifying `compute()` methods to recompile.
:::

### `initialize()`

Initialize the model before optimization.

```python
model.initialize()
```

Must be called after `build_module()` and before optimization.

## Variable Access

### `get_input()` / `set_input()`

Access input variables.

```python
# Get value
value = model.get_input("comp.x")
array_value = model.get_input("comp.forces[:]")

# Set value
model.set_input("comp.x", 5.0)
model.set_input("comp.forces[:]", [1.0, 2.0, 3.0])
```

### `get_output()`

Get output variable values.

```python
value = model.get_output("comp.y")
```

### `get_constraint()`

Get constraint values.

```python
value = model.get_constraint("comp.g")
```

### `get_objective()`

Get objective value.

```python
value = model.get_objective("comp.f")
```

## Model Inspection

### `print_model_structure()`

Print model hierarchy and connections.

```python
model.print_model_structure()
```

Output:
```
Model: my_model
Components:
  - comp1 (1 instance)
      Inputs: x1, x2
      Outputs: y
      Constraints: g
      Objective: f
  - comp2 (10 instances)
      Inputs: z
      Outputs: w
Links:
  comp1.y → comp2.z[0]
  ...
```

### `export_model_json()`

Export model structure to JSON.

```python
model.export_model_json("model_structure.json")
```

### `visualize_model()`

Generate model visualization.

```python
model.visualize_model("model_graph.html")
```

Creates an interactive graph showing components and connections.

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
        self.outputs["y"] = self.inputs["x"] ** 2

class Processor(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("input")
        self.add_output("output")
        self.add_objective("cost")
    
    def compute(self):
        x = self.inputs["input"]
        self.outputs["output"] = am.sqrt(x)
        self.objective["cost"] = x

class Sink(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("z")
        self.add_constraint("target", value=0.0, 
                          lower=0.0, upper=0.0)
    
    def compute(self):
        self.constraints["target"] = self.inputs["z"] - 2.0

# Build model
model = am.Model("pipeline")

# Add components
model.add_component("source", 1, Source())
model.add_component("proc", 3, Processor())
model.add_component("sink", 1, Sink())

# Link components
model.link("source.y", "proc.input[0]")
for i in range(2):
    model.link(f"proc.output[{i}]", f"proc.input[{i+1}]")
model.link("proc.output[2]", "sink.z")

# Build and initialize
model.build_module()
model.initialize()

# Inspect model
model.print_model_structure()

# Ready for optimization
opt = am.Optimizer(model)
opt.optimize()
```

## See Also

- [Component API](./component.md) - Creating components
- [Optimizer API](./optimizer.md) - Solving models
- [Tutorials](../tutorials/intro.md) - Complete examples

