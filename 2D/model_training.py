import torch
import torch.optim as optim
from model_architecture import PINN
import numpy as np
import matplotlib.pyplot as plt
from time import time
pi = np.pi


def initial_condition(x, y):
    return torch.sin(pi*x) * torch.sin(2*pi*y)


def exact_solution(x, y, t):
    # Exact solution for reference (not used in this script)
    return torch.sin(pi*x) * torch.sin(2*pi*y) * torch.exp(-5*alpha*pi**2*t)


torch.manual_seed(1234)

alpha = 0.05  # Thermal diffusivity

model = PINN()

# Load saved weights into model:
#model.load_state_dict(torch.load('weights1.pth'))

n_epochs = 1000

optimizer = optim.Adam(model.parameters(), lr=0.001)
time0 = time()

for epoch in range(n_epochs):

    optimizer.zero_grad()
    
    # Evaluate model at random points in domain
    x = torch.rand(1000, 1, requires_grad=True)
    y = torch.rand(1000, 1, requires_grad=True)
    t = torch.rand(1000, 1, requires_grad=True)
    u = model(x, y, t)
    
    # PDE residual
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    pde_residual = u_t - alpha * (u_xx + u_yy)
    
    # initial condition residual
    x_ic = torch.rand(200, 1)
    y_ic = torch.rand(200, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = model(x_ic, y_ic, t_ic)
    ic_residual = u_ic - initial_condition(x_ic, y_ic)
    
    # Total loss
    loss = torch.mean(pde_residual**2) + torch.mean(ic_residual**2)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}, Time/epoc: {(time()-time0)/epoch:.4f}s')


torch.save(model.state_dict(), 'weights1.pth')
