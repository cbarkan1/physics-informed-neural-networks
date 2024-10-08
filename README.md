## Using neural networks to solve high-dimensional partial differential equations (PDEs)

This repository contains code to solve the heat equation in high-dimensions using physics-informed neural networks (PINNs).

* Directories contain code for the heat equation in 1, 2, 4, 6, and 10 spatial dimensions.
* Each directory contains a jupyter notebook (Evaluating_PINN.ipynb) that summarizes the PINN's accuracy.
* Notes on computing model derivatives using PyTorch's autograd.grad function are given in "Notes on PyTorch's grad_outputs.ipynb"

The heat equation is:

$$\frac{\partial}{\partial t} u(\vec{x},t) = \alpha \sum_{i=1}^N \frac{\partial^2}{\partial x_i^2} u(\vec{x},t)$$

The domain considered here is

$$\vec{x} \in [0,1]^N$$

with boundary conditions

$$u(0,x_2,\cdots,x_N,t)=u(1,x_2,\cdots,x_N,t)=\cdots=u(x_1,\cdots,0,t)=u(x_1,\cdots,1,t)=0$$

Initial conditions are chosen such that the exact solution is known, so that the PINN's accuracy can be evaluated.

The solution is approximated by a neural network of the form

$$b(\vec x)g_\theta(\vec x,t)$$

where $g_\theta$ is a multilayer perceptron with parameters $\theta$, and $b$ is a function which enforces the boundary condition. I use

$$b(\vec x)=\Pi_{i=1}^N4x_i(1-x_i)$$

which equals 0 on the boundaries of the cube and 1 at the center of the cube.



Note: In principle, one could approximate $u$ directly with a multilayer perceptron, and enforce the boundary conditions with a loss term during training. However, my experience has been that enforcing the boundary condition with the function $b$ works much better.

Credit: Inspiration for this repo came from Hu, Shukla, Karniadakis, and Kawaguchi (2024) Neural Networks: [link to paper](https://www.sciencedirect.com/science/article/pii/S0893608024002934)