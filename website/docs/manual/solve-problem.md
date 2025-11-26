---
sidebar_position: 2
---

# Solve a problem

Once you have defined your optimization problem by creating components and building a model, you can solve it using the `Optimizer` class.

## Creating an Optimizer

```python
import amigo as am

# Assume model is already built and initialized
opt = am.Optimizer(model)
```

## Running Optimization

The simplest way to solve a problem is:

```python
opt.optimize()
```

This runs the optimization using default settings and updates the model with the optimal solution.

## Optimizer Options

Configure the optimizer with custom options:

```python
options = {
    'max_iter': 500,
    'tol': 1e-8,
    'print_level': 5,
}

opt.set_options(options)
opt.optimize()
```

Common options include:
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `print_level`: Verbosity level (0-12)
- `mu_strategy`: Barrier parameter strategy

## Accessing Results

After optimization, extract the solution from the model:

```python
# Get optimal values
x_opt = model.get_input("component.x")
f_opt = model.get_objective("component.f")
constraint = model.get_constraint("component.g")

print(f"Optimal x: {x_opt}")
print(f"Objective: {f_opt}")
print(f"Constraint: {constraint}")
```

## Complete Workflow

```python
import amigo as am

# 1. Create component
class MyComponent(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.0, lower=-10.0, upper=10.0)
        self.add_objective("f")
        self.add_constraint("g", upper=0.0)
    
    def compute(self):
        x = self.inputs["x"]
        self.objective["f"] = x * x
        self.constraints["g"] = x - 5.0

# 2. Build model
model = am.Model("optimization")
model.add_component("comp", 1, MyComponent())
model.build_module()
model.initialize()

# 3. Solve
opt = am.Optimizer(model)
opt.optimize()

# 4. Extract solution
x_opt = model.get_input("comp.x")
f_opt = model.get_objective("comp.f")

print(f"Solution: x = {x_opt:.6f}, f(x) = {f_opt:.6f}")
```

:::tip

Set `print_level: 5` to see detailed iteration output during optimization.

:::

See the [Optimizer API documentation](../api/optimizer.md) for complete details on configuration options.

