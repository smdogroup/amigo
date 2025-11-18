---
sidebar_position: 4
---

# Optimizer API

The `Optimizer` class solves optimization problems defined in Amigo models.

## Class: `amigo.Optimizer`

### Constructor

```python
amigo.Optimizer(model, algorithm="ipopt")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Model | Initialized Amigo model |
| `algorithm` | str | Optimization algorithm (default: "ipopt") |

**Supported Algorithms:**

- `"ipopt"` - Interior Point Optimizer (default, recommended)
- `"snopt"` - Sequential Quadratic Programming
- `"scipy"` - SciPy optimizers

**Example:**

```python
import amigo as am

model = am.Model("my_model")
# ... build model ...

opt = am.Optimizer(model)
# or
opt = am.Optimizer(model, algorithm="snopt")
```

## Optimization

### `optimize()`

Solve the optimization problem.

```python
opt.optimize()
```

**Returns:** None (solution is stored in the model)

**Example:**

```python
opt = am.Optimizer(model)
opt.optimize()

# Extract solution
x_opt = model.get_input("comp.x")
f_opt = model.get_objective("comp.f")

print(f"Optimal x: {x_opt}")
print(f"Optimal objective: {f_opt}")
```

## Configuration

### `set_options()`

Configure optimizer settings.

```python
opt.set_options(options)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `options` | dict | Dictionary of optimizer options |

**Common Options:**

```python
options = {
    # Convergence
    'tol': 1e-6,                    # Optimality tolerance
    'max_iter': 100,                # Maximum iterations
    'max_time': 3600,               # Maximum time (seconds)
    
    # Output
    'print_level': 5,               # Verbosity (0-12)
    'output_file': 'output.txt',    # Output file
    
    # Algorithm
    'mu_strategy': 'adaptive',      # Barrier parameter strategy
    'hessian_approximation': 'limited-memory',  # Hessian type
    
    # Line search
    'alpha_for_y': 'primal',        # Line search method
    'max_soc': 4,                   # Second-order corrections
}

opt.set_options(options)
```

**IPOPT-Specific Options:**

```python
ipopt_options = {
    'tol': 1e-7,
    'acceptable_tol': 1e-6,
    'max_iter': 3000,
    'print_level': 5,
    'sb': 'yes',                    # Suppress banner
    'linear_solver': 'mumps',       # Linear solver
    'mu_strategy': 'adaptive',
    'hessian_approximation': 'limited-memory',
}
```

**SNOPT-Specific Options:**

```python
snopt_options = {
    'Major optimality tolerance': 1e-6,
    'Major feasibility tolerance': 1e-6,
    'Major iterations limit': 1000,
    'Print file': 'snopt_output.txt',
}
```

## Monitoring

### `get_history()`

Retrieve optimization history.

```python
history = opt.get_history()
```

**Returns:** Dictionary with optimization history

```python
{
    'iter': [0, 1, 2, ...],
    'objective': [10.5, 8.3, 5.1, ...],
    'constraint_violation': [0.5, 0.2, 0.01, ...],
    'optimality': [1.0, 0.5, 0.1, ...],
    'step_size': [1.0, 1.0, 0.8, ...],
}
```

**Example - Plot Convergence:**

```python
import matplotlib.pyplot as plt

opt.optimize()
history = opt.get_history()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Objective
axes[0].plot(history['iter'], history['objective'])
axes[0].set_ylabel('Objective')
axes[0].set_xlabel('Iteration')
axes[0].grid(True)

# Constraint violation
axes[1].semilogy(history['iter'], history['constraint_violation'])
axes[1].set_ylabel('Constraint Violation')
axes[1].set_xlabel('Iteration')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### `set_callback()`

Register callback function called at each iteration.

```python
def callback(iteration, objective, constraint_violation):
    print(f"Iter {iteration}: f={objective:.6f}, "
          f"c_viol={constraint_violation:.6e}")

opt.set_callback(callback)
opt.optimize()
```

## Advanced Features

### Warm Start

Continue optimization from previous solution:

```python
# First optimization
opt.optimize()

# Modify problem slightly
model.set_input("comp.param", new_value)

# Warm start from previous solution
opt.set_options({'warm_start_init_point': 'yes'})
opt.optimize()
```

### Derivative Checking

Verify automatic differentiation:

```python
opt.set_options({
    'derivative_test': 'first-order',
    'derivative_test_tol': 1e-6,
})
opt.optimize()
```

### Scaling

Improve convergence with variable scaling:

```python
# Set scaling factors
model.set_input_scaling("comp.x", 1e-3)   # x ~ O(1000)
model.set_output_scaling("comp.y", 1e6)    # y ~ O(1e-6)
```

### Parallel Evaluation

For models with multiple components:

```python
# Enable parallel function evaluations
opt.set_options({
    'parallel': True,
    'num_threads': 4,
})
```

## Return Status

Check optimization status:

```python
opt.optimize()
status = opt.get_status()

if status['success']:
    print("Optimization successful!")
    print(f"Iterations: {status['num_iter']}")
    print(f"Objective: {status['objective']}")
else:
    print(f"Optimization failed: {status['message']}")
```

**Status Dictionary:**

```python
{
    'success': True/False,
    'num_iter': 42,
    'objective': 1.234,
    'constraint_violation': 1e-8,
    'optimality': 1e-7,
    'message': 'Optimal solution found',
    'exit_code': 0,
}
```

## Error Handling

```python
try:
    opt.optimize()
except am.OptimizationError as e:
    print(f"Optimization failed: {e}")
    print(f"Last feasible point:")
    print(f"  x = {model.get_input('comp.x')}")
```

## Complete Example

```python
import amigo as am
import matplotlib.pyplot as plt

# Create and build model
model = am.Model("optimization_example")
# ... add components and links ...
model.build_module()
model.initialize()

# Create optimizer
opt = am.Optimizer(model, algorithm="ipopt")

# Configure
opt.set_options({
    'tol': 1e-8,
    'max_iter': 500,
    'print_level': 5,
    'hessian_approximation': 'limited-memory',
})

# Set callback
def monitor(iter, obj, c_viol):
    if iter % 10 == 0:
        print(f"{iter:4d}  {obj:12.6e}  {c_viol:12.6e}")

print("Iter    Objective      Constraint Viol")
print("-" * 45)
opt.set_callback(monitor)

# Optimize
opt.optimize()

# Check status
status = opt.get_status()
if status['success']:
    print(f"\nOptimization converged in {status['num_iter']} iterations")
    print(f"Final objective: {status['objective']:.6e}")
else:
    print(f"\nOptimization failed: {status['message']}")

# Plot history
history = opt.get_history()
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.semilogy(history['iter'], history['objective'])
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.semilogy(history['iter'], history['constraint_violation'])
plt.xlabel('Iteration')
plt.ylabel('Constraint Violation')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.semilogy(history['iter'], history['optimality'])
plt.xlabel('Iteration')
plt.ylabel('Optimality')
plt.grid(True)

plt.tight_layout()
plt.savefig('convergence.png')
plt.show()
```

## See Also

- [Model API](./model.md) - Building models
- [Component API](./component.md) - Defining components
- [Tutorials](../tutorials/intro.md) - Optimization examples

