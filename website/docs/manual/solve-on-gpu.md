---
sidebar_position: 3
---

# Solve on GPU

GPU acceleration for Amigo optimization is currently under development. This page will be updated as CUDA backend capabilities become available.

## Current Status

:::warning

The CUDA backend for Nvidia GPUs is currently under development and not yet available in the release version.

:::

## Planned Features

When GPU support is released, Amigo will enable:

- **Parallel function evaluations** on GPU
- **Automatic differentiation** using GPU-accelerated A2D
- **Large-scale problems** leveraging GPU memory and compute
- **Seamless backend switching** - same Python code runs on CPU or GPU

## Backend Selection

In the future, GPU execution will be enabled by specifying the backend:

```python
# This will work once CUDA backend is released
model = am.Model("my_model", backend="cuda")
```

## Supported Hardware

The CUDA backend will target:
- Nvidia GPUs with CUDA compute capability 6.0 or higher
- Sufficient GPU memory for your problem size

## Development Timeline

GPU support is actively being developed. Check the [GitHub repository](https://github.com/your-org/amigo) for updates and progress.

## Alternatives

While GPU support is in development, consider these options for performance:

### OpenMP Backend

Multi-threaded CPU execution:

```python
model = am.Model("my_model", backend="openmp")
```

### MPI Backend

Distributed execution across multiple nodes:

```python
model = am.Model("my_model", backend="mpi")
```

See the [Installation guide](../getting-started/installation.md) for setting up MPI dependencies.

## Stay Updated

To be notified when GPU support is released:

1. Star the [Amigo repository](https://github.com/your-org/amigo)
2. Watch for release announcements
3. Join the discussion forum

We're excited to bring GPU acceleration to Amigo users soon!

