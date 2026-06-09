import numpy as np

def make_geometric_planted_instance(n: int, k: int, radius: float, rng):
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
            
    return n, k, edges

def _build_neighbors(n: int, edges: list) -> dict:
    neighbors = {v: [] for v in range(n)}
    for u, w in edges:
        neighbors[u].append(w)
        neighbors[w].append(u)
    return neighbors

def _vanilla_descent_solve(n: int, k: int, edges: list, seed: int, max_iter: int = 1000) -> bool:
    import numpy as np
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, k, size=n).tolist()
    neighbors = _build_neighbors(n, edges)

    for _ in range(max_iter):
        improved = False
        vertex_order = rng.permutation(n).tolist()
        for v in vertex_order:
            neighbor_color_counts = {}
            for nb in neighbors[v]:
                c = colors[nb]
                neighbor_color_counts[c] = neighbor_color_counts.get(c, 0) + 1

            current_conflicts = neighbor_color_counts.get(colors[v], 0)
            best_color = colors[v]
            best_conflicts = current_conflicts

            for c in range(k):
                if c != colors[v]:
                    c_conflicts = neighbor_color_counts.get(c, 0)
                    if c_conflicts < best_conflicts:
                        best_conflicts = c_conflicts
                        best_color = c

            if best_color != colors[v]:
                colors[v] = best_color
                improved = True

        if not improved:
            break

    conflicts = sum(1 for u, v in edges if colors[u] == colors[v])
    return conflicts == 0

rng = np.random.default_rng(1234)
for radius in [0.21, 0.22, 0.23, 0.24, 0.25]:
    results_dsatur = []
    results_vd = []
    for i in range(30):
        n, k, edges = make_geometric_planted_instance(60, 3, radius, rng)
        # DSATUR
        from test_generators3 import _dsatur_solve
        dsatur_solved = _dsatur_solve(n, k, edges)
        results_dsatur.append(dsatur_solved)
        
        # VD
        vd_solved = _vanilla_descent_solve(n, k, edges, seed=i)
        results_vd.append(vd_solved)
        
    print(f"radius={radius}, dsatur={sum(results_dsatur)/len(results_dsatur)}, vd={sum(results_vd)/len(results_vd)}")
