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

def _dsatur_solve(n, k, edges) -> bool:
    neighbors = _build_neighbors(n, edges)
    colors = [-1] * n
    saturation = [0] * n
    neighbor_colors = [set() for _ in range(n)]

    uncolored = set(range(n))
    while uncolored:
        v = max(uncolored, key=lambda x: (saturation[x], len(neighbors[x])))
        used = neighbor_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c

        for nb in neighbors[v]:
            if nb in uncolored:
                if c not in neighbor_colors[nb]:
                    neighbor_colors[nb].add(c)
                    saturation[nb] += 1

        uncolored.remove(v)

    max_color = max(colors) if colors else -1
    no_conflicts = all(colors[u] != colors[v] for u, v in edges)
    return (no_conflicts and max_color < k)

rng = np.random.default_rng(1234)
for radius in [0.21, 0.22, 0.23, 0.24, 0.25, 0.26]:
    results = []
    for _ in range(50):
        n, k, edges = make_geometric_planted_instance(60, 3, radius, rng)
        results.append(_dsatur_solve(n, k, edges))
    print(f"radius={radius}, dsatur_rate={sum(results)/len(results)}")
