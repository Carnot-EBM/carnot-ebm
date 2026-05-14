import numpy as np
from carnot.samplers.backend import CpuBackend
from thrml.models.ising import IsingEBM
import thrml

N = 128
J_val = 1.0
beta = 1.2

j_mat = np.ones((N, N)) / N
np.fill_diagonal(j_mat, 0)
bias = np.zeros(N)

# Carnot
cb = CpuBackend(seed=42)
carnot_samples = cb.sample(bias, j_mat, 1000, {'beta': beta, 'n_warmup': 1000, 'steps_per_sample': 10})
c_mean = np.mean(carnot_samples * 2 - 1)

print("Carnot direct J/n mapped to {-1,1} mean:", c_mean)

# thrml
edges = []
weights = []
for i in range(N):
    for j in range(i+1, N):
        edges.append((i, j))
        weights.append(1.0/N)

model = IsingEBM(nodes=list(range(N)), edges=edges, weights=weights, biases=bias.tolist(), beta=beta)
thrml_samples = thrml.sample_states(model, n_samples=1000, n_warmup=1000)
thrml_samples = np.concatenate(thrml_samples, axis=1) # shape (n_samples, n_spins)
t_mean = np.mean(thrml_samples * 2 - 1)
print("THRML direct J/n mapped to {-1,1} mean:", t_mean)
