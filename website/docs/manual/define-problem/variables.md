---
sidebar_position: 2
---

# Variables

Amigo components work with several types of variables, each serving a distinct purpose in defining optimization problems. Understanding these variable types is essential for building Amigo models.

## Inputs

Inputs are design variables that the optimizer can modify during the optimization process. They represent the degrees of freedom in your problem.

### Scalar Inputs

```python
self.add_input("x", value=1.0, lower=-10.0, upper=10.0)
```

### Vector Inputs

```python
# Define a 3-element vector
self.add_input("forces", shape=(3,), value=0.0, lower=-100.0, upper=100.0)

# Access individual elements
def compute(self):
    forces = self.inputs["forces"]
    fx = forces[0]
    fy = forces[1]
    fz = forces[2]
```

### Matrix Inputs

```python
# Define a 3x3 matrix
self.add_input("K", shape=(3, 3))
```

### With Units and Labels

```python
self.add_input("velocity", value=10.0, units="m/s", label="state")
self.add_input("force", value=100.0, units="N", label="control")
```

The `label` parameter is useful for grouping related variables, especially in optimal control problems where you might label certain inputs as `"state"` or `"control"`.

## Outputs

Outputs are computed quantities that can be linked to inputs of other components. They enable component coupling in multidisciplinary systems.

```python
# Scalar output
self.add_output("lift", units="N")

# Vector output
self.add_output("stress", shape=(3,))
```

Outputs are computed in the `compute()` method:

```python
def compute(self):
    velocity = self.inputs["velocity"]
    self.outputs["lift"] = 0.5 * 1.225 * velocity**2 * 10.0  # Simplified
```

:::tip

Outputs can be linked to inputs of other components using `model.link()`, enabling data flow between disciplines.

:::

## Constants

Constants are compile-time values that do not change during optimization. They are compiled as `constexpr` in the generated C++ code for maximum performance.

```python
self.add_constant("pi", value=3.14159)
self.add_constant("g", value=9.81, units="m/s^2")
self.add_constant("E", value=200e9, units="Pa")
```

Constants are accessed in `compute()` just like inputs:

```python
def compute(self):
    g = self.constants["g"]
    mass = self.inputs["mass"]
    self.outputs["weight"] = mass * g
```

## Data

Data variables allow you to pass external information into the problem that is not optimized but may change between solves.

```python
self.add_data("temperature_field", shape=(100, 100))
self.add_data("boundary_conditions", shape=(10,))
```

Data is useful for:
- External loads or boundary conditions
- Mesh data or geometry information
- Parameters for sensitivity studies

## Intermediate Variables

The `self.vars` dictionary stores intermediate computation results. These variables are symbolic and participate in automatic differentiation.

```python
def compute(self):
    q = self.inputs["q"]
    
    # Compute and store intermediate values
    self.vars["sint"] = am.sin(q[1])
    self.vars["cost"] = am.cos(q[1])
    
    # Use intermediate variables
    sint = self.vars["sint"]
    cost = self.vars["cost"]
    
    self.outputs["rotation"] = [cost, sint]
```

Intermediate variables help:
- Organize complex computations
- Avoid repeating calculations
- Improve code readability

:::note

Intermediate variables in `self.vars` are automatically included in the automatic differentiation process.

:::

