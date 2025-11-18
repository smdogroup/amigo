---
sidebar_position: 2
---

# Component API

The `Component` class is the foundation of Amigo models. Each component represents an analysis or computation with inputs, outputs, constraints, and objectives.

## Class: `amigo.Component`

Base class for defining analysis components.

### Constructor

```python
class MyComponent(am.Component):
    def __init__(self):
        super().__init__()
        # Define interface here
```

Always call `super().__init__()` in your component's constructor.

## Adding Variables

### `add_input()`

Define an input (design variable).

```python
self.add_input(name, value=0.0, lower=None, upper=None, 
               shape=None, units=None, label=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Variable name (required) |
| `value` | float/array | Initial value (default: 0.0) |
| `lower` | float/array | Lower bound (default: None) |
| `upper` | float/array | Upper bound (default: None) |
| `shape` | tuple | Array shape (default: scalar) |
| `units` | str | Physical units (optional) |
| `label` | str | Category label (optional) |

**Examples:**

```python
# Scalar input
self.add_input("x", value=1.0, lower=-10.0, upper=10.0)

# Vector input
self.add_input("forces", shape=(3,), value=0.0)

# Matrix input
self.add_input("stiffness", shape=(3, 3))

# With units
self.add_input("velocity", value=10.0, units="m/s")

# With label for grouping
self.add_input("q", shape=(4,), label="state")
```

### `add_output()`

Define an output variable.

```python
self.add_output(name, shape=None, units=None, label=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Variable name (required) |
| `shape` | tuple | Array shape (default: scalar) |
| `units` | str | Physical units (optional) |
| `label` | str | Category label (optional) |

**Examples:**

```python
# Scalar output
self.add_output("lift")

# Vector output
self.add_output("forces", shape=(3,))

# With units
self.add_output("power", units="W")
```

### `add_constraint()`

Define a constraint.

```python
self.add_constraint(name, value=None, lower=None, upper=None,
                   shape=None, units=None, label=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Variable name (required) |
| `value` | float/array | Equality constraint value |
| `lower` | float/array | Lower bound |
| `upper` | float/array | Upper bound |
| `shape` | tuple | Array shape (default: scalar) |
| `units` | str | Physical units (optional) |
| `label` | str | Category label (optional) |

**Constraint Types:**

```python
# Inequality: lower ≤ g(x) ≤ upper
self.add_constraint("stress", lower=0.0, upper=100.0)

# One-sided inequality: g(x) ≤ 0
self.add_constraint("buckling", upper=0.0)

# Equality: h(x) = value
self.add_constraint("balance", value=0.0, lower=0.0, upper=0.0)

# Vector constraint
self.add_constraint("residual", shape=(10,), value=0.0, 
                   lower=0.0, upper=0.0)
```

### `add_objective()`

Define an objective function to minimize.

```python
self.add_objective(name, units=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Variable name (required) |
| `units` | str | Physical units (optional) |

**Example:**

```python
self.add_objective("mass")
self.add_objective("cost", units="USD")
```

:::tip
To maximize an objective, negate it: `self.objective["f"] = -profit`
:::

### `add_constant()`

Define a compile-time constant.

```python
self.add_constant(name, value, units=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Constant name (required) |
| `value` | float/array | Constant value (required) |
| `units` | str | Physical units (optional) |

**Examples:**

```python
self.add_constant("g", value=9.81, units="m/s^2")
self.add_constant("E", value=200e9, units="Pa")
self.add_constant("rho", value=7850.0, units="kg/m^3")
```

:::info
Constants become `constexpr` in generated C++ code for maximum performance.
:::

## Accessing Variables

### In `compute()` Method

Access variables through dictionary interfaces:

```python
def compute(self):
    # Read inputs
    x = self.inputs["x"]
    forces = self.inputs["forces"]
    
    # Read constants
    g = self.constants["g"]
    
    # Store intermediate variables (optional)
    self.vars["temp"] = x * x
    
    # Write outputs
    self.outputs["y"] = 2 * x
    
    # Write constraints
    self.constraints["g"] = x * x - 1.0
    
    # Write objective
    self.objective["f"] = x * x + forces[0]
```

### Array Indexing

```python
def compute(self):
    forces = self.inputs["forces"]  # shape (3,)
    
    # Element access
    fx = forces[0]
    fy = forces[1]
    fz = forces[2]
    
    # Array slicing
    f_xy = forces[0:2]
    
    # Matrix operations
    K = self.inputs["K"]  # shape (3, 3)
    element = K[1, 2]
    row = K[0, :]
    col = K[:, 0]
```

## Mathematical Operations

### Arithmetic

```python
# Basic operations
y = x1 + x2
y = x1 - x2
y = x1 * x2
y = x1 / x2
y = x1 ** 2  # Power

# Combined operations
y = 2 * x1 + 3 * x2
y = (x1 + x2) / (x1 - x2)
```

### Functions

```python
# Trigonometry
y = am.sin(x)
y = am.cos(x)
y = am.tan(x)
y = am.asin(x)
y = am.acos(x)
y = am.atan(x)
y = am.atan2(y, x)

# Exponential and logarithmic
y = am.exp(x)
y = am.log(x)  # Natural log
y = am.log10(x)

# Other functions
y = am.sqrt(x)
y = am.abs(x)
y = am.pow(x, n)
```

### Array Operations

```python
# Dot product
result = am.dot(a, b)

# Matrix-vector multiply
result = am.matmul(A, x)

# Norm
magnitude = am.norm(v)

# Sum
total = am.sum(array)
```

## Advanced Features

### Multiple Compute Methods

Use different compute methods for different contexts:

```python
class MyComponent(am.Component):
    def __init__(self):
        super().__init__()
        # Define interface
        
        # Set compute arguments for different instances
        self.set_args([
            {"mode": "forward"},   # Instance 0
            {"mode": "backward"},  # Instance 1
        ])
    
    def compute(self, mode="forward"):
        if mode == "forward":
            # Forward computation
            pass
        else:
            # Backward computation
            pass
```

### Conditional Logic

```python
def compute(self):
    x = self.inputs["x"]
    
    # Use am.if_else for conditional operations
    y = am.if_else(x > 0, x, -x)  # abs(x)
```

:::warning
Standard Python `if`/`else` statements don't work with symbolic variables. Use `am.if_else()` instead.
:::

## Complete Example

```python
import amigo as am

class StructuralComponent(am.Component):
    """
    Computes stress in a structural member.
    
    Inputs
    ------
    force : float
        Applied force [N]
    area : float
        Cross-sectional area [m²]
    
    Outputs
    -------
    stress : float
        Stress in member [Pa]
    
    Constraints
    -----------
    stress_limit : float
        Stress must not exceed yield strength
    
    Constants
    ---------
    sigma_y : float
        Yield strength [Pa]
    """
    
    def __init__(self):
        super().__init__()
        
        # Constants
        self.add_constant("sigma_y", value=250e6, units="Pa")
        
        # Inputs
        self.add_input("force", value=1000.0, lower=0.0, 
                      upper=10000.0, units="N")
        self.add_input("area", value=0.01, lower=0.001, 
                      upper=0.1, units="m^2")
        
        # Outputs
        self.add_output("stress", units="Pa")
        
        # Constraints
        self.add_constraint("stress_limit", upper=0.0, units="Pa")
        
        # Objective (minimize weight, proportional to area)
        self.add_objective("weight", units="kg")
    
    def compute(self):
        # Get inputs
        force = self.inputs["force"]
        area = self.inputs["area"]
        sigma_y = self.constants["sigma_y"]
        
        # Compute stress
        stress = force / area
        self.outputs["stress"] = stress
        
        # Stress constraint: σ ≤ σ_y  →  σ - σ_y ≤ 0
        self.constraints["stress_limit"] = stress - sigma_y
        
        # Objective: minimize weight (assuming unit length and density)
        self.objective["weight"] = area * 7850.0  # steel density
```

## See Also

- [Model API](./model.md) - Building models with components
- [Optimizer API](./optimizer.md) - Solving optimization problems
- [Tutorials](../tutorials/intro.md) - Component examples

