import torch
import torch.optim as optim
from model_architecture import PINN
from time import time
pi = 3.1415926535898


def initial_condition(x_list):
    return torch.sin(pi*x_list[0])*torch.sin(2*pi*x_list[1])*torch.sin(pi*x_list[2])*torch.sin(pi*x_list[3]) \
          *torch.sin(pi*x_list[4])*torch.sin(pi*x_list[5])*torch.sin(pi*x_list[6])*torch.sin(pi*x_list[7]) \
          *torch.sin(pi*x_list[8])*torch.sin(pi*x_list[9])


def exact_solution(x_list, t):
    # Exact solution for reference (not used in this script)
    return torch.sin(pi*x_list[0])*torch.sin(2*pi*x_list[1])*torch.sin(pi*x_list[2])*torch.sin(pi*x_list[3]) \
          *torch.sin(pi*x_list[4])*torch.sin(pi*x_list[5])*torch.sin(pi*x_list[6])*torch.sin(pi*x_list[7]) \
          *torch.sin(pi*x_list[8])*torch.sin(pi*x_list[9])*torch.exp(-13*alpha*pi**2*t)


alpha = 0.025  # Thermal diffusivity

spatial_dim = 10

torch.manual_seed(123)
model = PINN(input_dim=spatial_dim+1, hidden_dim=128, num_layers=4)

# Load saved model weights:
#model.load_state_dict(torch.load('weights1.pth'))

n_epochs = 1000

optimizer = optim.Adam(model.parameters(), lr=0.0001)
time0 = time()

for epoch in range(n_epochs):

    optimizer.zero_grad()

    # PDE residual
    x_list = [torch.rand(1000, 1, requires_grad=True) for i in range(spatial_dim)]
    t = torch.rand(1000, 1, requires_grad=True)
    u = model(x_list, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    pde_residual = u_t
    for i in range(spatial_dim):
        u_i = torch.autograd.grad(u, x_list[i], grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_ii = torch.autograd.grad(u_i, x_list[i], grad_outputs=torch.ones_like(u), create_graph=True)[0]
        pde_residual -= alpha*u_ii

    # initial condition residual
    x_ic_list = [torch.rand(200, 1) for i in range(spatial_dim)]
    t_ic = torch.zeros(200, 1)
    u_ic = model(x_ic_list, t_ic)
    ic_residual = u_ic - initial_condition(x_ic_list)
    
    loss = torch.mean(pde_residual**2) + torch.mean(ic_residual**2)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}, Time/epoc: {(time()-time0)/epoch:.4f}s')

# Save the model parameters
torch.save(model.state_dict(), 'weights1.pth')
