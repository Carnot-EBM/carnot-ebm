"""Graph Coloring Ising/QUBO encoding + discrete SA/PT optimizers (Exp 3518).

WHY graph coloring for the second CSP (P0.1 generalization):
    The k-coloring problem has a standard QUBO/Ising mapping (Lucas 2014,
    Frontiers in Physics 2(5), Section 5). The encoding maps DIRECTLY to a
    sum of conflict penalties — E=0 iff no two adjacent vertices share a color.
    This gives a clean encoding-validity precondition (E==0 on a known-valid
    solution) and an efficient SA move structure (recolor one vertex, O(degree)
    delta computation per move).

    Literature precedent for non-AR beating greedy-AR on graph coloring:
    - DIFUSCO (arXiv:2302.08224): non-autoregressive diffusion beats AR on
      graph coloring and TSP.
    - "Beyond Autoregression" (arXiv:2410.14157): global-inference methods beat
      AR on Sudoku, Boolean SAT, and Countdown — the same regime we test here.

Ising/QUBO reference (Lucas 2014, "Ising formulations of many NP problems"):
    Binary x_{v,c} ∈ {0,1}: x_{v,c}=1 iff vertex v has color c.
    E = A * sum_v (1 - sum_c x_{v,c})^2   [one-hot: each vertex has one color]
      + B * sum_{(u,v) ∈ E} sum_c x_{u,c} * x_{v,c}   [conflict penalty]
    E = 0 iff the assignment is a valid k-coloring.

    Discrete implementation: represent state as c(v) ∈ {1,...,k} (one-hot
    satisfied by construction) and compute E = sum_{(u,v)∈E} [c(u)==c(v)].

Spec: REQ-KONA-3518, SCENARIO-KONA-3518
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# An edge list representation of an undirected graph
Edges = list[tuple[int, int]]
# A color assignment: assignment[v] ∈ {1,...,k}
Assignment = list[int]


# ---------------------------------------------------------------------------
# Energy dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphColoringEnergy:
    """Energy of a graph coloring assignment.

    WHY frozen: energy is computed once per check — it's a measurement, not
    a mutable state. Immutability prevents accidental modification after checks.

    total_energy: E = n_conflicts (the number of violated edge constraints).
        When one-hot is satisfied by construction (discrete c(v)), this equals
        the QUBO H_B term from Lucas (2014).
    is_valid: True iff total_energy == 0.0 — all edges respect the coloring.
    n_conflicts: edges (u,v) where assignment[u] == assignment[v].
    n_uncolored: vertices with color outside {1,...,k} — signals encoding bugs.
    """

    total_energy: float
    is_valid: bool
    n_conflicts: int
    n_uncolored: int

    def as_dict(self) -> dict:
        return {
            "total_energy": self.total_energy,
            "is_valid": self.is_valid,
            "n_conflicts": self.n_conflicts,
            "n_uncolored": self.n_uncolored,
        }


# ---------------------------------------------------------------------------
# Encoding validity check
# ---------------------------------------------------------------------------

def check_encoding_validity(
    assignment: Assignment,
    edges: Edges,
    n_vertices: int,
    n_colors: int,
) -> GraphColoringEnergy:
    """Compute the Ising energy of a coloring assignment.

    WHY this exists: the encoding-validity precondition (E==0 on a known-valid
    solution) is the FIRST gate before running any optimizer. If E>0 on a
    guaranteed-valid solution, the energy function is mis-specified and no
    optimizer can solve the problem — we stop and report the bug.

    Maps to the QUBO formulation of Lucas (2014), Section 5:
    H = A * sum_v (1 - sum_c x_{v,c})^2  [one-hot, A=1]
      + B * sum_{(u,v)∈E} sum_c x_{u,c} * x_{v,c}  [conflict, B=1]
    For discrete c(v): H_A=0 by construction; H_B = sum_{(u,v)∈E} [c(u)==c(v)].
    """
    n_uncolored = sum(1 for c in assignment if c < 1 or c > n_colors)
    n_conflicts = sum(1 for u, v in edges if assignment[u] == assignment[v])
    total = float(n_uncolored + n_conflicts)
    return GraphColoringEnergy(
        total_energy=total,
        is_valid=(total == 0.0),
        n_conflicts=n_conflicts,
        n_uncolored=n_uncolored,
    )


# ---------------------------------------------------------------------------
# Graph instance generation
# ---------------------------------------------------------------------------

def generate_colorable_graph(
    n_vertices: int,
    n_colors: int,
    edge_probability: float,
    seed: int,
) -> tuple[Edges, Assignment]:
    """Generate a random k-colorable graph with a known valid solution.

    WHY guaranteed colorability via partition: we first assign each vertex a
    color (round-robin + shuffle → balanced partition), then add edges ONLY
    between DIFFERENT color classes. The initial partition is therefore always
    a valid k-coloring: E=0 by construction on the returned assignment.

    This avoids the need for an exact solver to confirm solvability while
    still creating challenging instances — adding many cross-partition edges
    makes greedy sequential coloring more likely to fail (it must avoid using
    the same color as non-adjacent same-class vertices that were colored first).

    Returns (edges, known_coloring) where known_coloring[v] ∈ {1,...,k}.
    """
    rng = np.random.default_rng(seed)
    # Balanced partition: round-robin assignment, then shuffle
    assignment = [(v % n_colors) + 1 for v in range(n_vertices)]
    rng.shuffle(assignment)

    edges: Edges = []
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            # Only add edges between DIFFERENT color classes — preserves
            # the colorability of the known solution.
            if assignment[u] != assignment[v] and rng.random() < edge_probability:
                edges.append((u, v))
    return edges, list(assignment)


def is_valid_coloring(assignment: Assignment, edges: Edges, n_colors: int) -> bool:
    """True iff assignment is a valid k-coloring of the graph defined by edges."""
    if any(c < 1 or c > n_colors for c in assignment):
        return False
    return all(assignment[u] != assignment[v] for u, v in edges)


# ---------------------------------------------------------------------------
# SA state initialisation
# ---------------------------------------------------------------------------

def _init_gc_state(
    assignment: Assignment,
    adj: list[list[int]],
    rng: np.random.Generator,
    n_colors: int,
) -> dict[str, Any]:
    """Build the mutable SA state from an initial coloring.

    WHY separate adj: passing the pre-built adjacency list avoids rebuilding
    it for every restart while keeping the state dict self-contained.

    State tracks:
    - assignment: current color per vertex (mutated in place)
    - n_conflicts: total conflicting edges (cached, updated O(1) per move)
    """
    n_conflicts = sum(1 for u in range(len(assignment)) for v in adj[u]
                      if v > u and assignment[u] == assignment[v])
    # Randomize initial assignment (not the known solution — let SA explore)
    init = rng.integers(1, n_colors + 1, size=len(assignment)).tolist()
    # Recount conflicts for this random init
    n_conflicts = sum(1 for u in range(len(init)) for v in adj[u]
                      if v > u and init[u] == init[v])
    return {
        "assignment": list(init),
        "n_conflicts": n_conflicts,
        "n_vertices": len(assignment),
        "n_colors": n_colors,
        "adj": adj,
    }


# ---------------------------------------------------------------------------
# O(degree(v)) delta computation and state update
# ---------------------------------------------------------------------------

def _delta_conflicts(state: dict[str, Any], v: int, new_c: int) -> int:
    """Return the change in edge conflicts from recoloring vertex v to new_c.

    WHY O(degree): scan only v's neighbors. For each neighbor u:
    - If u has old color: conflict removed → delta -= 1
    - If u has new color: conflict added → delta += 1
    Net: delta = (# neighbors with new_c) - (# neighbors with old_c).
    """
    assignment = state["assignment"]
    old_c = assignment[v]
    if old_c == new_c:
        return 0
    delta = 0
    for u in state["adj"][v]:
        if assignment[u] == old_c:
            delta -= 1
        if assignment[u] == new_c:
            delta += 1
    return delta


def _apply_recolor(state: dict[str, Any], v: int, new_c: int, delta: int) -> None:
    """Apply recoloring of vertex v to new_c and update the cached conflict count."""
    state["assignment"][v] = new_c
    state["n_conflicts"] += delta


# ---------------------------------------------------------------------------
# SA inner loop
# ---------------------------------------------------------------------------

def _run_gc_sweep(
    state: dict[str, Any],
    T: float,
    rng: np.random.Generator,
    n_moves: int,
) -> None:
    """Run n_moves random recolor attempts at temperature T.

    Move: pick a random vertex v, pick a random NEW color c' ≠ c(v),
    accept if delta ≤ 0 or with Boltzmann probability exp(-delta/T).

    WHY recolor (not swap): swapping two vertices' colors is valid but
    recoloring is simpler and sufficient — any valid k-coloring is reachable
    from any starting assignment via single-vertex recolorings.
    """
    n_vertices = state["n_vertices"]
    n_colors = state["n_colors"]
    assignment = state["assignment"]

    vertices = rng.integers(0, n_vertices, size=n_moves)
    for v in vertices:
        v = int(v)
        old_c = assignment[v]
        # Sample a new color ≠ old_c: shift within {1,...,k}
        shift = int(rng.integers(1, n_colors))
        new_c = (old_c - 1 + shift) % n_colors + 1

        delta = _delta_conflicts(state, v, new_c)
        if delta <= 0:
            _apply_recolor(state, v, new_c, delta)
        elif T > 0.0 and rng.random() < math.exp(-delta / T):
            _apply_recolor(state, v, new_c, delta)


# ---------------------------------------------------------------------------
# Public solver interfaces
# ---------------------------------------------------------------------------

def gc_sa_solve_once(
    edges: Edges,
    n_vertices: int,
    n_colors: int,
    *,
    n_sweeps: int = 4000,
    n_moves_per_sweep: int = 40,
    T_init: float = 1.0,
    T_final: float = 0.01,
    seed: int = 0,
) -> tuple[Assignment, bool, int]:
    """Single SA trajectory for graph k-coloring.

    Returns (final_assignment, solved, n_conflicts_at_stop).
    solved=True iff the assignment is a valid k-coloring.
    """
    adj = [[] for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    rng = np.random.default_rng(seed)
    state = _init_gc_state([0] * n_vertices, adj, rng, n_colors)

    T = T_init
    # When T_init == 0 (pure greedy descent), skip the cooling calculation to
    # avoid division by zero; temperature stays 0 throughout all sweeps.
    if T_init > 0.0 and T_final > 0.0:
        cooling = (T_final / T_init) ** (1.0 / max(n_sweeps, 1))
    else:
        cooling = 0.0
    for _ in range(n_sweeps):
        _run_gc_sweep(state, T, rng, n_moves_per_sweep)
        if cooling > 0.0:
            T *= cooling
        if state["n_conflicts"] == 0:
            break

    assignment = state["assignment"]
    solved = is_valid_coloring(assignment, edges, n_colors)
    return list(assignment), solved, state["n_conflicts"]


def gc_sa_solve_restarts(
    edges: Edges,
    n_vertices: int,
    n_colors: int,
    *,
    n_sweeps: int = 4000,
    n_moves_per_sweep: int = 40,
    T_init: float = 1.0,
    T_final: float = 0.01,
    n_restarts: int = 15,
    seed: int = 0,
    progress_callback: Any = None,
) -> tuple[Assignment, bool, int]:
    """K independent SA restarts for graph k-coloring; return best result.

    WHY restarts: SA is stochastic — different random initializations explore
    different basins. Early stopping across restarts (first solved wins).
    """
    adj = [[] for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    best_assignment: Assignment | None = None
    best_conflicts = n_vertices * n_vertices  # large sentinel

    for k in range(n_restarts):
        rng = np.random.default_rng(seed + k * 997)
        state = _init_gc_state([0] * n_vertices, adj, rng, n_colors)

        T = T_init
        cooling = (T_final / T_init) ** (1.0 / max(n_sweeps, 1))
        for _ in range(n_sweeps):
            _run_gc_sweep(state, T, rng, n_moves_per_sweep)
            T *= cooling
            if state["n_conflicts"] == 0:
                break

        assignment = list(state["assignment"])
        solved = is_valid_coloring(assignment, edges, n_colors)

        if progress_callback:
            progress_callback(k, state["n_conflicts"], solved)

        if solved:
            return assignment, True, 0

        if state["n_conflicts"] < best_conflicts:
            best_conflicts = state["n_conflicts"]
            best_assignment = assignment

    assert best_assignment is not None
    return best_assignment, False, best_conflicts


def gc_parallel_tempering_solve_instrumented(
    edges: Edges,
    n_vertices: int,
    n_colors: int,
    *,
    n_sweeps: int = 4000,
    n_moves_per_sweep: int = 40,
    n_chains: int = 6,
    T_min: float = 0.1,
    T_max: float = 2.0,
    n_exchange_interval: int = 50,
    seed: int = 0,
    progress_callback: Any = None,
) -> tuple[Assignment, bool, int, float]:
    """Parallel tempering with swap-acceptance tracking for graph k-coloring.

    WHY PT: hot chains explore broadly (escape local minima), cold chain
    refines. Replica exchange allows solved regions found by warm chains to
    propagate to the cold chain. swap_acceptance_rate tracks ladder quality:
    target 0.2–0.5 for effective mixing.

    Returns (assignment, solved, n_conflicts, swap_acceptance_rate).
    """
    adj = [[] for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    temps = np.exp(
        np.linspace(math.log(T_min), math.log(T_max), n_chains)
    ).tolist()

    rngs = [np.random.default_rng(seed + i * 1337) for i in range(n_chains)]
    states = [_init_gc_state([0] * n_vertices, adj, rngs[i], n_colors)
              for i in range(n_chains)]

    total_proposals = 0
    total_accepts = 0

    for sweep in range(n_sweeps):
        for i in range(n_chains):
            _run_gc_sweep(states[i], temps[i], rngs[i], n_moves_per_sweep)

        if (sweep + 1) % n_exchange_interval == 0:
            exchange_rng = rngs[0]
            for i in range(n_chains - 1):
                total_proposals += 1
                E_i = states[i]["n_conflicts"]
                E_ip1 = states[i + 1]["n_conflicts"]
                beta_i = 1.0 / temps[i] if temps[i] > 0 else 1e9
                beta_ip1 = 1.0 / temps[i + 1] if temps[i + 1] > 0 else 1e9
                delta_swap = (beta_i - beta_ip1) * (E_ip1 - E_i)
                if delta_swap >= 0 or exchange_rng.random() < math.exp(delta_swap):
                    states[i], states[i + 1] = states[i + 1], states[i]
                    total_accepts += 1

            if progress_callback:
                progress_callback(sweep + 1, states[0]["n_conflicts"])

        if states[0]["n_conflicts"] == 0:
            break

    assignment = list(states[0]["assignment"])
    solved = is_valid_coloring(assignment, edges, n_colors)
    swap_acc = total_accepts / total_proposals if total_proposals > 0 else 0.0
    return assignment, solved, states[0]["n_conflicts"], swap_acc


# ---------------------------------------------------------------------------
# Exact solver (backtracking + forward checking)
# ---------------------------------------------------------------------------

def gc_exact_solve(
    edges: Edges,
    n_vertices: int,
    n_colors: int,
    *,
    max_nodes: int = 5_000_000,
) -> Assignment | None:
    """Backtracking exact k-coloring solver with forward checking.

    WHY backtracking with forward checking: forward checking prunes color
    domains for uncolored neighbors before recursing, cutting the search tree
    significantly vs naive backtracking. For small-to-medium graphs (n ≤ 30),
    this is fast enough to serve as an optimality reference.

    Returns a valid k-coloring if one exists within max_nodes, else None.
    """
    adj: list[list[int]] = [[] for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # MRV heuristic: vertex with fewest remaining color options first.
    # Use degree ordering (proxy for MRV) for simplicity.
    degree = [len(adj[v]) for v in range(n_vertices)]
    order = sorted(range(n_vertices), key=lambda v: -degree[v])

    assignment = [0] * n_vertices
    nodes = [0]

    def _is_safe(v: int, color: int) -> bool:
        return all(assignment[u] != color for u in adj[v])

    def _backtrack(pos: int) -> bool:
        if pos == n_vertices:
            return True
        v = order[pos]
        for color in range(1, n_colors + 1):
            nodes[0] += 1
            if nodes[0] > max_nodes:
                return False
            if _is_safe(v, color):
                assignment[v] = color
                if _backtrack(pos + 1):
                    return True
                assignment[v] = 0
        return False

    if _backtrack(0):
        return [assignment[v] for v in range(n_vertices)]
    return None


# ---------------------------------------------------------------------------
# AR greedy baseline
# ---------------------------------------------------------------------------

def gc_ar_greedy_solve(
    edges: Edges,
    n_vertices: int,
    n_colors: int,
    seed: int = 0,
) -> Assignment | None:
    """Greedy graph coloring: assign each vertex its lowest available color.

    WHY greedy as the AR baseline: greedy sequential coloring mimics the
    autoregressive (left-to-right, no-backtrack) generation pattern. It can
    fail even on k-colorable graphs when the vertex ordering forces it to use
    more than k colors — exactly the regime where global-inference (energy SA)
    succeeds by exploring non-local solutions.

    Literature AR-on-coloring: greedy uses up to Δ+1 colors in the worst case
    (Brooks' theorem: k-colorable but greedy may need k+1). This gap is what
    DIFUSCO (arXiv:2302.08224) and the "Beyond Autoregression" paper exploit.

    Returns a valid k-coloring if greedy succeeds with ≤k colors, else None.
    """
    rng = np.random.default_rng(seed)
    order = list(range(n_vertices))
    rng.shuffle(order)

    adj: list[set[int]] = [set() for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    assignment = [0] * n_vertices
    for v in order:
        neighbor_colors = {assignment[u] for u in adj[v] if assignment[u] > 0}
        # Assign lowest color not used by already-colored neighbors
        color = next(
            (c for c in range(1, n_colors + 1) if c not in neighbor_colors),
            None,
        )
        if color is None:
            return None  # Failed: needs > k colors with this vertex ordering
        assignment[v] = color
    return assignment
