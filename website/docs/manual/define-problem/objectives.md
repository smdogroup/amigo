---
sidebar_position: 4
---

# Objectives

The objective function defines the quantity to be minimized during optimization. Every optimization problem in Amigo must have at least one objective.

## Declaring an Objective

Objectives are declared in the component constructor:

```python
self.add_objective("cost")
```

Objectives are always scalar values. Only one objective per component is allowed.

## Setting the Objective Value

The objective value is computed in the `compute()` method:

```python
def compute(self):
    x = self.inputs["x"]
    
    # Minimize the quadratic function f(x) = x²
    self.objective["cost"] = x * x
```

## Minimization vs Maximization

Amigo optimizers minimize objectives by default. To maximize a function, negate it:

```python
def compute(self):
    revenue = self.inputs["sales"] * self.inputs["price"]
    cost = self.inputs["expenses"]
    profit = revenue - cost
    
    # Maximize profit by minimizing -profit
    self.objective["profit"] = -profit
```

## Multiple Components with Objectives

When multiple components define objectives, the total objective is the sum of all component objectives:

```python
# Component 1: minimize weight
class StructureComponent(am.Component):
    def compute(self):
        self.objective["f"] = self.inputs["mass"]

# Component 2: minimize drag  
class AeroComponent(am.Component):
    def compute(self):
        self.objective["f"] = self.inputs["drag"]

# Total objective = weight + drag (multi-objective as weighted sum)
```

This allows for natural multi-objective optimization through weighted sums.

## Weighting Objectives

Apply weights to balance multiple objectives:

```python
def compute(self):
    weight = self.inputs["mass"]
    drag = self.inputs["drag"]
    
    # Weighted combination
    w1 = 0.7  # Weight for mass
    w2 = 0.3  # Weight for drag
    
    self.objective["total"] = w1 * weight + w2 * drag
```

## Units

Specify units for documentation and clarity:

```python
self.add_objective("mass", units="kg")
self.add_objective("energy", units="J")
self.add_objective("time", units="s")
self.add_objective("cost", units="USD")
```

## Example: Time-Optimal Control

Minimize final time in an optimal control problem:

```python
class TimeObjective(am.Component):
    def __init__(self):
        super().__init__()
        
        # Final time as an input (design variable)
        self.add_input("T", value=10.0, lower=0.1, upper=100.0, units="s")
        
        # Objective: minimize final time
        self.add_objective("time_cost", units="s")
    
    def compute(self):
        T = self.inputs["T"]
        
        # Minimize T
        self.objective["time_cost"] = T
```

## Example: Integral Cost

For optimal control, you may want to minimize an integral cost over time:

```python
class ControlCostComponent(am.Component):
    def __init__(self):
        super().__init__()
        
        self.add_input("u", label="control")
        self.add_constant("dt", value=0.1)  # Time step
        
        self.add_objective("control_effort")
    
    def compute(self):
        u = self.inputs["u"]
        dt = self.constants["dt"]
        
        # Discrete approximation of ∫ u²  dt
        self.objective["control_effort"] = u * u * dt
```

When this component is instantiated multiple times (one per time step), the objectives are summed automatically, approximating the continuous integral.

:::tip

For trajectory optimization, create multiple instances of cost components to approximate integral costs.

:::

