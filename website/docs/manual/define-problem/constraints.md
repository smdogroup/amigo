---
sidebar_position: 3
---

# Constraints

Constraints define the feasible region for your optimization problem. They specify conditions that must be satisfied by the solution.

## Constraint Declaration

Constraints are declared in the component constructor using `add_constraint`:

```python
self.add_constraint(name, value=None, lower=0.0, upper=0.0, 
                   shape=None, units=None, label=None)
```

## Equality Constraints

Equality constraints enforce that a function equals a specific value. In Amigo, this is specified by setting `lower` and `upper` to the same value:

```python
# h(x) = 0
self.add_constraint("balance", value=0.0, lower=0.0, upper=0.0)
```

The `value` parameter provides the target value, while `lower` and `upper` being equal enforces the equality.

In the `compute()` method, set the constraint value:

```python
def compute(self):
    x = self.inputs["x"]
    
    # Constraint: x^2 - 1 = 0
    self.constraints["balance"] = x**2 - 1.0
```

:::note

By default, `add_constraint` creates an equality constraint with `lower=0.0` and `upper=0.0`.

:::

## Inequality Constraints

### One-Sided Inequalities

Upper bound constraint (g(x) ≤ 0):

```python
# Declare: stress ≤ σ_yield  →  stress - σ_yield ≤ 0
self.add_constraint("stress", upper=0.0)

# Compute
def compute(self):
    stress = self.compute_stress()
    sigma_y = self.constants["sigma_y"]
    self.constraints["stress"] = stress - sigma_y
```

Lower bound constraint (g(x) ≥ 0):

```python
# Declare: clearance ≥ 0
self.add_constraint("clearance", lower=0.0, upper=float("inf"))

# Compute
def compute(self):
    gap = self.compute_gap()
    self.constraints["clearance"] = gap
```

### Bounded Inequalities

Double-sided constraints (lower ≤ g(x) ≤ upper):

```python
self.add_constraint("temperature", lower=20.0, upper=100.0)

def compute(self):
    T = self.compute_temperature()
    self.constraints["temperature"] = T
```

## Vector Constraints

Define multiple constraints simultaneously using the `shape` parameter:

```python
# System of 4 equality constraints
self.add_constraint("residual", shape=(4,), value=0.0, 
                   lower=0.0, upper=0.0)
```

Set vector constraint values:

```python
def compute(self):
    # Compute 4 residual equations
    res = [None] * 4
    res[0] = self.inputs["q"][0] - self.inputs["qdot"][0]
    res[1] = self.inputs["q"][1] - self.inputs["qdot"][1]
    res[2] = ... 
    res[3] = ...
    
    self.constraints["residual"] = res
```

## Example: Structural Component

A complete example with multiple constraint types:

```python
import amigo as am

class BeamComponent(am.Component):
    def __init__(self):
        super().__init__()
        
        # Constants
        self.add_constant("sigma_y", value=250e6)  # Yield strength [Pa]
        self.add_constant("E", value=200e9)        # Young's modulus [Pa]
        
        # Inputs
        self.add_input("force", value=1000.0, units="N")
        self.add_input("area", value=0.01, lower=0.001, upper=0.1, units="m^2")
        
        # Constraints
        self.add_constraint("stress_limit", upper=0.0)    # σ ≤ σ_y
        self.add_constraint("min_thickness", lower=0.0)   # t ≥ t_min
        
        # Objective
        self.add_objective("mass", units="kg")
    
    def compute(self):
        F = self.inputs["force"]
        A = self.inputs["area"]
        sigma_y = self.constants["sigma_y"]
        
        # Compute stress
        stress = F / A
        
        # Stress constraint: σ - σ_y ≤ 0
        self.constraints["stress_limit"] = stress - sigma_y
        
        # Minimum thickness constraint: A - A_min ≥ 0
        self.constraints["min_thickness"] = A - 0.001
        
        # Minimize mass (proportional to area)
        self.objective["mass"] = A * 7850.0 * 1.0  # ρ * A * L
```

## Constraint Linking

When constraints from different components are linked, their values are summed:

```python
# In component 1
self.constraints["total_force"] = force1

# In component 2  
self.constraints["total_force"] = force2

# After linking: total_force = force1 + force2
model.link("comp1.total_force", "comp2.total_force")
```

This is useful for enforcing global constraints across multiple components.

