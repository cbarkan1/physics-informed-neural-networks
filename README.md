# Physics-Informed Neural Networks for High-Dimensional PDEs

## Overview

This repository demonstrates Physics-Informed Neural Networks (PINNs) for solving the heat equation in multiple dimensions (1D, 2D, 4D, 6D, and 10D). Traditional grid-based PDE solvers struggle with high dimensions (d > 4), but PINNs can handle much higher dimensions.

Each dimension folder (1D, 2D, 4D, 6D, 10D) contains:
- `Evaluating_PINN.ipynb`, A notebook comparing the trained PINN to the exact analytical solution
- Scripts for PINN architecture, training, and trained weights.

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Mathematical Formulation

### Heat Equation

$$\frac{\partial}{\partial t} u(\vec{x},t) = \alpha \sum_{i=1}^N \frac{\partial^2}{\partial x_i^2} u(\vec{x},t)$$

**Domain**: $\vec{x} \in [0,1]^N$ (N-dimensional unit cube)

**Boundary conditions**: Zero Dirichlet conditions on all boundaries:
$$u(0,x_2,\cdots,x_N,t)=u(1,x_2,\cdots,x_N,t)=\cdots=u(x_1,\cdots,0,t)=u(x_1,\cdots,1,t)=0$$

**Initial conditions**: A few different initial conditions are used, all of which yield fairly simple exact analytical solutions.

### Neural Network Architecture

The solution is approximated using:
$$u(\vec{x},t) = b(\vec{x}) \cdot g_\theta(\vec{x},t)$$

where:
- $g_\theta$ is a multilayer perceptron with parameters $\theta$
- $b(\vec{x}) = \prod_{i=1}^N 4x_i(1-x_i)$ enforces boundary conditions

The boundary function $b(\vec{x})$ equals 0 on cube boundaries and 1 at the center, automatically satisfying boundary conditions without additional loss terms.

## References

Inspiration for this repository came from:

Hu, Shukla, Karniadakis, and Kawaguchi (2024). "Physics-informed neural networks for solving high-dimensional partial differential equations." *Neural Networks*. [Link to paper](https://www.sciencedirect.com/science/article/pii/S0893608024002934)