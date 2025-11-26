---
sidebar_position: 1
---

# Components

The `Component` class is the foundation of Amigo. All analysis and computation in Amigo occurs within classes that inherit from `amigo.Component`. A component encapsulates a computational model with well-defined inputs, outputs, constraints, and objectives.

## Basic Structure

The minimal structure of an Amigo component consists of:

1. A class inheriting from `amigo.Component`
2. An `__init__` method defining the interface
3. A `compute` method implementing the analysis

```python
import amigo as am

class MyComponent(am.Component):
    def __init__(self):
        super().__init__()
        # Define inputs, outputs, constraints, objectives
        self.add_input("x", value=1.0, lower=0.0, upper=10.0)
        self.add_output("y")
    
    def compute(self):
        # Extract inputs
        x = self.inputs["x"]
        
        # Perform computation
        self.outputs["y"] = 2 * x
```

:::note

Always call `super().__init__()` at the beginning of your component's constructor.

:::

## The Constructor

The constructor defines the component's interface. This is where you declare all variables your component will use:

```python
def __init__(self):
    super().__init__()
    
    # Constants (compile-time values)
    self.add_constant("g", value=9.81)
    
    # Design variables (optimizer can change these)
    self.add_input("x", value=0.0, lower=-10.0, upper=10.0)
    
    # Outputs (computed values that can be linked)
    self.add_output("lift")
    
    # Constraints (must be satisfied)
    self.add_constraint("stress", upper=0.0)
    
    # Objective (function to minimize)
    self.add_objective("weight")
```

## The Compute Method

The `compute` method contains your analysis logic. Values are accessed through dictionary-like member objects:

```python
def compute(self):
    # Read inputs and constants
    x = self.inputs["x"]
    g = self.constants["g"]
    
    # Store intermediate variables (optional)
    self.vars["temp"] = x * x
    
    # Set outputs
    self.outputs["lift"] = 10 * x
    
    # Set constraints
    self.constraints["stress"] = x * x - 100.0
    
    # Set objective
    self.objective["weight"] = x
```

:::warning

Variables in `compute()` are symbolic, not numeric. They encode mathematical operations for automatic differentiation and C++ code generation.

:::

## Example: Rosenbrock Function

A complete example implementing the Rosenbrock optimization problem:

```python
import amigo as am

class Rosenbrock(am.Component):
    def __init__(self):
        super().__init__()
        
        # Design variables
        self.add_input("x1", value=-1.0, lower=-2.0, upper=2.0)
        self.add_input("x2", value=-1.0, lower=-2.0, upper=2.0)
        
        # Objective to minimize
        self.add_objective("obj")
        
        # Constraint: x1² + x2² ≤ 1
        self.add_constraint("con", upper=0.0)
    
    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        
        # Rosenbrock function
        self.objective["obj"] = (1 - x1)**2 + 100*(x2 - x1**2)**2
        
        # Circular constraint
        self.constraints["con"] = x1**2 + x2**2 - 1.0
```


