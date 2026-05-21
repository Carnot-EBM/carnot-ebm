import torch
import torch.nn as nn
import numpy as np

class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_knots=8, degree=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.n_params = num_knots + degree - 1 # Fixed
        self.control_points = nn.Parameter(torch.empty(out_dim, in_dim, self.n_params).uniform_(-0.1, 0.1))
        
        grid = torch.linspace(-1, 1, steps=num_knots)
        step = grid[1] - grid[0]
        grid = torch.cat([grid[0] - step * torch.arange(degree, 0, -1), grid, grid[-1] + step * torch.arange(1, degree + 1)])
        self.register_buffer('grid', grid)

    def b_spline(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.degree + 1):
            left_denom = grid[k:-1] - grid[:-k-1]
            left_term = (x - grid[:-k-1]) / torch.where(left_denom == 0, torch.ones_like(left_denom), left_denom) * bases[..., :-1]
            right_denom = grid[k+1:] - grid[1:-k]
            right_term = (grid[k+1:] - x) / torch.where(right_denom == 0, torch.ones_like(right_denom), right_denom) * bases[..., 1:]
            bases = left_term + right_term
        return bases

    def forward(self, x):
        bases = self.b_spline(x)
        return torch.einsum('bin,oin->bo', bases, self.control_points)

class KAN(nn.Module):
    def __init__(self, in_dim, hidden_dim=4, out_dim=1, num_knots=8):
        super().__init__()
        self.layer1 = KANLayer(in_dim, hidden_dim, num_knots=num_knots)
        self.layer2 = KANLayer(hidden_dim, out_dim, num_knots=num_knots)
        
    def forward(self, x):
        x = torch.tanh(x)
        h = self.layer1(x)
        h = torch.tanh(h)
        return self.layer2(h)

model = KAN(10)
x = torch.randn(4, 10, requires_grad=True)
out = model(x)
grads = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
local_lip = torch.norm(grads, p=2, dim=1)
print("local_lip:", local_lip)
