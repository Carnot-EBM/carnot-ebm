import numpy as np

def make_barabasi_albert_planted_instance(n: int, k: int, m: int, rng, instance_id: int):
    groups = [[] for _ in range(k)]
    for v in range(n):
        groups[v % k].append(v)
    
    planted_colors = [0] * n
    for color, group in enumerate(groups):
        for v in group:
            planted_colors[v] = color

    edges = []
    degrees = [0] * n
    
    for v in range(n):
        candidates = [u for u in range(v) if planted_colors[u] != planted_colors[v]]
        if not candidates:
            continue
        n_edges_to_add = min(m, len(candidates))
        weights = [degrees[u] + 1 for u in candidates]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        chosen = rng.choice(candidates, size=n_edges_to_add, replace=False, p=probs)
        for u in chosen:
            edges.append((int(u), int(v)))
            degrees[u] += 1
            degrees[v] += 1
            
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

rng = np.random.default_rng(42)
for m in [2, 3, 4, 5, 6, 7]:
    results = []
    for _ in range(50):
        n, k, edges = make_barabasi_albert_planted_instance(60, 3, m, rng, 0)
        results.append(_dsatur_solve(n, k, edges))
    print(f"m={m}, dsatur_rate={sum(results)/len(results)}")
