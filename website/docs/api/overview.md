---
sidebar_position: 1
---

# API Overview

The Amigo API is organized into several key modules that work together to define, build, and solve optimization problems.

## Core Modules

### Component
The foundation of Amigo models - defines individual analysis components.

```python
import amigo as am

class MyComponent(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.0)
        self.add_output("y")
    
    def compute(self):
        self.outputs["y"] = 2 * self.inputs["x"]
```

**Key Methods:**
- `add_input()` - Define design variables
- `add_output()` - Define computed outputs
- `add_constraint()` - Define constraints
- `add_objective()` - Define objective functions
- `add_constant()` - Define constants
- `compute()` - Implement component logic

[→ Component API Reference](./component.md)

### Model
Assembles components into a complete optimization model.

```python
model = am.Model("my_model")
model.add_component("comp1", 10, MyComponent())
model.link("comp1.y[0:9]", "comp2.x[0:9]")
model.build_module()
model.initialize()
```

**Key Methods:**
- `add_component()` - Add component instances
- `link()` - Connect variables between components
- `add_sub_model()` - Add hierarchical sub-models
- `build_module()` - Compile to C++
- `initialize()` - Initialize the model
- `get_input()` / `set_input()` - Access variables

[→ Model API Reference](./model.md)

### Optimizer
Solves the optimization problem.

```python
opt = am.Optimizer(model)
opt.set_options({'max_iter': 100, 'tol': 1e-6})
opt.optimize()
```

**Key Methods:**
- `optimize()` - Solve the problem
- `set_options()` - Configure optimizer
- `get_history()` - Retrieve convergence history

[→ Optimizer API Reference](./optimizer.md)

### ExternalComponent
Interface to external models (e.g., OpenMDAO).

```python
from amigo import ExternalComponent

class OpenMDAOWrapper(ExternalComponent):
    def __init__(self, om_component):
        super().__init__()
        self.om_comp = om_component
        # Define interface
```

[→ ExternalComponent API Reference](./external-component.md)

## Mathematical Operations

Amigo provides mathematical functions that work with symbolic variables:

### Basic Operations
- `+`, `-`, `*`, `/` - Arithmetic
- `**` - Power
- `am.sin()`, `am.cos()`, `am.tan()` - Trigonometry
- `am.exp()`, `am.log()` - Exponentials
- `am.sqrt()`, `am.abs()` - Basic math

### Array Operations
- Indexing: `x[i]`
- Slicing: `x[i:j]`
- Dot products: `am.dot(a, b)`
- Matrix multiply: `am.matmul(A, b)`

## Variable Types

### Inputs
Design variables that can be optimized.

```python
self.add_input("x", 
              value=1.0,        # Initial value
              lower=-10.0,      # Lower bound
              upper=10.0,       # Upper bound
              shape=(3,),       # Shape (optional)
              units="m",        # Units (optional)
              label="state")    # Label (optional)
```

### Outputs
Computed quantities that can be linked to other components.

```python
self.add_output("y",
               shape=(3,),
               label="result")
```

### Constraints
Equality or inequality constraints.

```python
# Inequality: lower ≤ constraint ≤ upper
self.add_constraint("g",
                   lower=0.0,
                   upper=float("inf"),
                   shape=(2,))

# Equality: constraint = value
self.add_constraint("h",
                   value=0.0,
                   lower=0.0,
                   upper=0.0)
```

### Objectives
Functions to minimize.

```python
self.add_objective("f")  # Minimize f
```

### Constants
Compile-time constants (become `constexpr` in C++).

```python
self.add_constant("pi", value=3.14159)
self.add_constant("g", value=9.81)
```

## Naming and Linking

### Variable Naming

Variables are scoped by component name:

```
component_name.variable_name[indices]
```

Examples:
- `"wing.lift"` - Scalar variable
- `"wing.forces[0]"` - Single element
- `"wing.forces[0:3]"` - Slice
- `"wing.forces[:, 0]"` - Multi-dimensional slice

### Linking Variables

Link variables between components:

```python
# Link scalar variables
model.link("comp1.output", "comp2.input")

# Link arrays
model.link("comp1.forces[0:3]", "comp2.loads[0:3]")

# Link with source/target indices
model.link("comp1.forces", "comp2.loads",
          src_indices=[0, 2, 4],
          tgt_indices=[1, 3, 5])
```

## Backend Selection

Choose computational backend:

```python
# Serial (default)
model = am.Model("name", backend="serial")

# OpenMP parallelization
model = am.Model("name", backend="openmp")

# MPI distributed
model = am.Model("name", backend="mpi")

# CUDA (under development)
model = am.Model("name", backend="cuda")
```

## Code Generation

When you call `model.build_module()`, Amigo:

1. **Analyzes** the Python `compute()` methods
2. **Generates** optimized C++ code
3. **Compiles** with automatic differentiation (A2D)
4. **Links** to create a Python module

Generated files:
- `{model_name}.cpp` - C++ implementation
- `{model_name}.h` - Header file
- `{model_name}.pyd/.so` - Compiled Python extension

## Error Handling

```python
try:
    model.build_module()
except am.CompilationError as e:
    print(f"Compilation failed: {e}")

try:
    opt.optimize()
except am.OptimizationError as e:
    print(f"Optimization failed: {e}")
```

## Best Practices

1. **Component Design**
   - Keep components focused on single responsibilities
   - Use descriptive variable names
   - Document component interfaces

2. **Model Building**
   - Test components individually before combining
   - Use meaningful component names
   - Verify links are correct

3. **Optimization**
   - Provide good initial guesses
   - Scale variables appropriately
   - Monitor convergence

4. **Performance**
   - Use constants for fixed parameters
   - Leverage vectorization
   - Choose appropriate backend

## Quick Reference

| Task | Code |
|------|------|
| Import | `import amigo as am` |
| Create component | `class C(am.Component):` |
| Add input | `self.add_input("x", value=0.0)` |
| Add constraint | `self.add_constraint("g", lower=0.0)` |
| Create model | `model = am.Model("name")` |
| Add component | `model.add_component("c", 1, C())` |
| Link variables | `model.link("c1.y", "c2.x")` |
| Compile | `model.build_module()` |
| Initialize | `model.initialize()` |
| Optimize | `opt = am.Optimizer(model)` <br/> `opt.optimize()` |

## Next Steps

- [Component API](./component.md) - Detailed component documentation
- [Model API](./model.md) - Model building reference
- [Optimizer API](./optimizer.md) - Optimization configuration
- [Tutorials](../tutorials/intro.md) - Learn by example

