import numpy as np
from experiment_3563_p01_route1_graph_coloring_multiseed_second_generator_v4 import _parallel_tempering_solve, GraphColoringInstance

def make_geometric_planted_instance(n: int, k: int, radius: float, rng, instance_id: int):
    groups = [[] for _ in range(k)]
    for v in range(n):
        groups[v % k].append(v)
    
    planted_colors = [0] * n
    for color, group in enumerate(groups):
        for v in group:
            planted_colors[v] = color

    points = rng.uniform(0, 1, size=(n, 2))
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if planted_colors[i] != planted_colors[j]:
                dist = np.linalg.norm(points[i] - points[j])
                if dist < radius:
                    edges.append((i, j))
            
    return GraphColoringInstance(instance_id, n, k, edges, planted_colors, "hard", "geometric", radius, 0.0)

rng = np.random.default_rng(1234)
for radius in [0.24, 0.25, 0.26]:
    pt_solved_count = 0
    for i in range(10):
        inst = make_geometric_planted_instance(60, 3, radius, rng, i)
        pt_solved, _ = _parallel_tempering_solve(inst, seed=i, n_steps=3000)
        if pt_solved:
            pt_solved_count += 1
    print(f"radius={radius}, PT={pt_solved_count/10}")
