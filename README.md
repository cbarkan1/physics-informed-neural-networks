# Physics-Informed Neural Networks for High-Dimensional PDEs

## Overview

This repository demonstrates Physics-Informed Neural Networks (PINNs) for solving the heat equation in multiple dimensions (1D, 2D, 4D, 6D, and 10D). Traditional grid-based PDE solvers struggle with high dimensions (d > 4), but PINNs can handle much higher dimensions.

Each dimension folder (1D, 2D, 4D, 6D, 10D) contains:
- `Evaluating_PINN.ipynb`, A notebook comparing the trained PINN to the exact analytical solution
- Scripts for PINN architecture, training, and trained weights.

`Notes on PyTorch's grad_outputs.ipynb` contains notes on an aspect of computing partial derivatives in PyTorch that is not clearly specified in PyTorch's documentation.

## Mathematical Formulation

### Heat Equation

$$\frac{\partial}{\partial t} u(\vec{x},t) = \alpha \sum_{i=1}^N \frac{\partial^2}{\partial x_i^2} u(\vec{x},t)$$

**Domain**: x ∈ [0,1]^N (N-dimensional unit cube)

**Boundary conditions**: Zero Dirichlet conditions on all boundaries:
```math
u(0,x₂,...,x_N,t) = u(1,x₂,...,x_N,t) = ... = u(x₁,...,0,t) = u(x₁,...,1,t) = 0
```

**Initial conditions**: A few different initial conditions are used, all of which yield fairly simple exact analytical solutions.

### Neural Network Architecture

The solution is approximated using:

$$u(\vec{x},t) = b(\vec{x})  g_\theta(\vec{x},t)$$

where $g_\theta$ is a multilayer perceptron with parameters $\theta$ and

$$b(\vec{x}) = \prod_{i=1}^N 4x_i(1-x_i \) $$

enforces the boundary conditions.

The boundary function b(x) equals 0 on cube boundaries and 1 at the center, automatically satisfying boundary conditions without additional loss terms.

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## References

Inspiration for this repository came from:

Hu, Shukla, Karniadakis, and Kawaguchi (2024). "Physics-informed neural networks for solving high-dimensional partial differential equations." *Neural Networks*. [Link to paper](https://www.sciencedirect.com/science/article/pii/S0893608024002934)