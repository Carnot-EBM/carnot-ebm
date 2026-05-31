"""Tests for experiment 3528: P0.1 graph coloring headroom with DSATUR strong baseline.

Covers:
  - compute_energy: valid coloring => E=0, invalid coloring => E > 0
  - _dsatur_solve: produces valid colorings on known graphs
  - _vanilla_descent_solve: solves easy instances
  - _ar_greedy_solve: solves bipartite (always 2-colorable)
  - _reproducibility_checksum: deterministic and seed-sensitive
  - GraphColoringInstance dataclass construction

Spec: REQ-KONA-3528, SCENARIO-KONA-3528
"""

from __future__ import annotations

import sys
import os

# WHY sys.path insert: the experiment script lives in scripts/, not in a package,
# so we add scripts/ to sys.path to allow direct import of the module by filename.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

from experiment_3528_p01_graph_coloring_headroom_strong_baseline_v1 import (
    GraphColoringInstance,
    _ar_greedy_solve,
    _dsatur_solve,
    _reproducibility_checksum,
    _vanilla_descent_solve,
    compute_energy,
)


# ---------------------------------------------------------------------------
# compute_energy tests
# ---------------------------------------------------------------------------


def test_compute_energy_valid_coloring_is_zero():
    """Known valid 3-coloring of K_3 triangle has E=0.

    WHY: the fundamental correctness test for the QUBO/Ising encoding.
    E=0 on the known-valid coloring is the gating condition (Step 0a)
    that all other experiment steps depend on.
    """
    edges = [(0, 1), (1, 2), (0, 2)]
    colors = [0, 1, 2]  # all different colors: valid 3-coloring of K_3
    E = compute_energy(colors, 3, 3, edges)
    assert E == 0.0, f"Expected 0 but got {E}"


def test_compute_energy_invalid_coloring_is_positive():
    """Invalid coloring (conflict) has E > 0.

    WHY: the encoding must distinguish valid from invalid colorings via energy.
    If E were 0 for an invalid coloring, the optimizer would have no signal
    to improve toward a valid solution.
    """
    edges = [(0, 1), (1, 2), (0, 2)]
    colors = [0, 0, 2]  # vertices 0 and 1 both assigned color 0 — conflict on edge (0,1)
    E = compute_energy(colors, 3, 3, edges)
    assert E > 0.0, f"Expected >0 but got {E}"


def test_compute_energy_path_graph_two_coloring():
    """Path 0-1-2-3 with alternating 2-coloring has E=0.

    WHY: tests a simple non-complete graph to ensure the energy function
    correctly handles non-triangle topologies.
    """
    edges = [(0, 1), (1, 2), (2, 3)]
    colors = [0, 1, 0, 1]  # alternating colors on a path: always valid 2-coloring
    E = compute_energy(colors, 4, 2, edges)
    assert E == 0.0


def test_compute_energy_counts_conflicts_correctly():
    """Energy counts the number of conflicting edges, not vertices.

    WHY: verifies the counting logic is edge-based, not vertex-based.
    Two edges can share a conflicting vertex but each contributes its own
    conflict count independently.
    """
    # Star graph: center vertex 0, leaves 1, 2, 3. All edges go through 0.
    edges = [(0, 1), (0, 2), (0, 3)]
    # If center and leaf 1 have same color: 1 conflict
    colors = [0, 0, 1, 2]
    E = compute_energy(colors, 4, 3, edges)
    assert E == 1.0, f"Expected 1 conflict but got {E}"


def test_compute_energy_all_same_color_maximum_conflicts():
    """All vertices same color on K_3 gives maximum conflicts = 3 edges.

    WHY: boundary test — maximum energy case should equal total number of edges.
    """
    edges = [(0, 1), (1, 2), (0, 2)]
    colors = [0, 0, 0]  # all same color: all 3 edges conflict
    E = compute_energy(colors, 3, 3, edges)
    assert E == 3.0, f"Expected 3 but got {E}"


def test_compute_energy_empty_graph():
    """Empty graph (no edges) always has E=0 regardless of coloring.

    WHY: ensures the function handles edge cases correctly, including the
    degenerate case of a graph with no constraints.
    """
    colors = [0, 0, 0]  # all same color, but no edges => no conflicts
    E = compute_energy(colors, 3, 3, edges=[])
    assert E == 0.0


# ---------------------------------------------------------------------------
# _dsatur_solve tests
# ---------------------------------------------------------------------------


def test_dsatur_on_path_graph():
    """DSATUR on path graph 0-1-2-3-4 should produce a valid coloring.

    WHY: path graphs are 2-colorable (bipartite). DSATUR should always
    find a valid coloring using at most 2 colors, well within the k=3 limit.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=5,
        k=3,  # allow up to 3 colors; path only needs 2
        edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
        planted_colors=[0, 1, 0, 1, 0],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=1.6,
    )
    colors, valid = _dsatur_solve(inst)
    assert valid, f"DSATUR should produce valid coloring, got valid={valid}"
    # Verify no conflicts directly.
    for u, v in inst.edges:
        assert colors[u] != colors[v], f"Conflict at edge ({u},{v}): color={colors[u]}"


def test_dsatur_on_k3_triangle():
    """DSATUR on K_3 (complete triangle) needs exactly 3 colors.

    WHY: K_3 is the canonical 3-chromatic graph. DSATUR must use all 3 colors
    and should succeed within the k=3 limit.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=3,
        k=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        planted_colors=[0, 1, 2],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=2.0,
    )
    colors, valid = _dsatur_solve(inst)
    assert valid, f"DSATUR should solve K_3 with 3 colors, got valid={valid}"


def test_dsatur_on_bipartite_graph():
    """DSATUR on complete bipartite K_{2,2} produces valid 2-coloring.

    WHY: K_{2,2} is 2-colorable (bipartite). DSATUR should find a valid
    coloring easily since bipartite graphs are trivially 2-colorable.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=4,
        k=3,
        edges=[(0, 2), (0, 3), (1, 2), (1, 3)],  # K_{2,2}: {0,1} and {2,3}
        planted_colors=[0, 0, 1, 1],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=2.0,
    )
    colors, valid = _dsatur_solve(inst)
    assert valid, f"DSATUR should solve K_{{2,2}}, got valid={valid}"
    for u, v in inst.edges:
        assert colors[u] != colors[v], f"Conflict at edge ({u},{v})"


def test_dsatur_isolated_vertices():
    """DSATUR on graph with isolated vertices (no edges) gives valid coloring.

    WHY: edge case — vertices with no neighbors have no constraints. DSATUR
    should still assign colors without errors.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=3,
        k=3,
        edges=[],  # no edges at all
        planted_colors=[0, 1, 2],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=0.0,
    )
    colors, valid = _dsatur_solve(inst)
    assert valid, f"DSATUR on isolated vertices should be valid, got valid={valid}"


# ---------------------------------------------------------------------------
# _vanilla_descent_solve tests
# ---------------------------------------------------------------------------


def test_vanilla_descent_trivial_instance():
    """Vanilla descent solves a very easy instance (K_{2,2} bipartite).

    WHY: K_{2,2} is 2-colorable and very easy to solve. Vanilla descent
    should always find the solution since there are no local minima for
    2-colorable graphs — any conflict can always be resolved.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=4,
        k=3,
        edges=[(0, 2), (0, 3), (1, 2), (1, 3)],  # K_{2,2}
        planted_colors=[0, 0, 1, 1],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=2.0,
    )
    solved = _vanilla_descent_solve(inst, seed=42, max_iter=500)
    assert solved, "Vanilla descent should solve trivial K_{2,2} instance"


def test_vanilla_descent_path_graph():
    """Vanilla descent solves a path graph (no local minima for 2-colorings).

    WHY: path graphs are 2-colorable and have no local minima — any conflict
    can be resolved by recoloring one endpoint of the conflicting edge.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=6,
        k=3,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        planted_colors=[0, 1, 0, 1, 0, 1],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=1.67,
    )
    solved = _vanilla_descent_solve(inst, seed=42, max_iter=500)
    assert solved, "Vanilla descent should solve path graph"


def test_vanilla_descent_returns_bool():
    """_vanilla_descent_solve always returns a bool.

    WHY: function contract — callers check the return value as a bool.
    Must not return None or other types.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=3,
        k=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        planted_colors=[0, 1, 2],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=2.0,
    )
    result = _vanilla_descent_solve(inst, seed=42, max_iter=10)
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"


# ---------------------------------------------------------------------------
# _ar_greedy_solve tests
# ---------------------------------------------------------------------------


def test_ar_greedy_on_bipartite():
    """AR greedy on bipartite path graph should always find valid 2-coloring.

    WHY: bipartite graphs (paths, trees, even cycles) are 2-colorable. Greedy
    coloring always succeeds on 2-colorable graphs because when processing each
    vertex, at most 1 color is excluded per connected neighbor color class.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=6,
        k=3,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        planted_colors=[0, 1, 0, 1, 0, 1],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=1.67,
    )
    solved = _ar_greedy_solve(inst, seed=42)
    assert solved, "AR greedy should solve path graph (2-colorable)"


def test_ar_greedy_returns_bool():
    """_ar_greedy_solve always returns a bool.

    WHY: function contract — callers use the return value as a solved indicator.
    Must not return None or other types.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=3,
        k=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        planted_colors=[0, 1, 2],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=2.0,
    )
    result = _ar_greedy_solve(inst, seed=42)
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"


def test_ar_greedy_k3_triangle_sometimes_fails():
    """AR greedy on K_3 triangle sometimes fails (needs 3 colors but may not find them).

    WHY: K_3 requires exactly 3 colors. Greedy coloring with a bad vertex ordering
    can still succeed if the first two vertices get distinct colors and the third
    vertex then gets the third color. But the test verifies the function runs
    without error — we don't assert solved=True since it's probabilistic.
    """
    inst = GraphColoringInstance(
        instance_id=0,
        n_vertices=3,
        k=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        planted_colors=[0, 1, 2],
        difficulty="easy",
        p_cross=0.0,
        avg_degree=2.0,
    )
    # Run multiple seeds to ensure the function works correctly.
    results = [_ar_greedy_solve(inst, seed=s) for s in range(10)]
    # At least some seeds should succeed (K_3 is 3-colorable and greedy often works).
    assert any(results), "AR greedy should succeed on K_3 with at least one seed out of 10"
    # All results should be bools.
    assert all(isinstance(r, bool) for r in results)


# ---------------------------------------------------------------------------
# _reproducibility_checksum tests
# ---------------------------------------------------------------------------


def test_reproducibility_checksum_stable():
    """Same inputs produce same checksum.

    WHY: reproducibility checksums exist to let third parties verify they are
    running the same experiment. If the checksum changes with same inputs,
    it cannot serve as a content-addressed fingerprint.
    """
    instances = [
        GraphColoringInstance(
            instance_id=i,
            n_vertices=10,
            k=3,
            edges=[(0, 1)],
            planted_colors=[0] * 10,
            difficulty="easy",
            p_cross=0.1,
            avg_degree=0.2,
        )
        for i in range(3)
    ]
    c1 = _reproducibility_checksum(instances, seed=42, optimizer_configs={"n_restarts": 5})
    c2 = _reproducibility_checksum(instances, seed=42, optimizer_configs={"n_restarts": 5})
    assert c1 == c2, "Checksum must be deterministic"


def test_reproducibility_checksum_different_seed():
    """Different seed produces different checksum.

    WHY: the seed is part of the experiment identity. Two runs with different
    seeds are different experiments and must have different checksums to allow
    disambiguation in the audit trail.
    """
    instances = [
        GraphColoringInstance(
            instance_id=0,
            n_vertices=10,
            k=3,
            edges=[(0, 1)],
            planted_colors=[0] * 10,
            difficulty="easy",
            p_cross=0.1,
            avg_degree=0.2,
        )
    ]
    c1 = _reproducibility_checksum(instances, seed=42, optimizer_configs={})
    c2 = _reproducibility_checksum(instances, seed=999, optimizer_configs={})
    assert c1 != c2, "Different seeds should produce different checksums"


def test_reproducibility_checksum_different_optimizer_configs():
    """Different optimizer configs produce different checksums.

    WHY: optimizer configuration is part of the experiment definition. If SA
    uses 5 restarts vs 20 restarts, those are different experiments with
    potentially different solve rates.
    """
    instances = [
        GraphColoringInstance(
            instance_id=0,
            n_vertices=10,
            k=3,
            edges=[(0, 1)],
            planted_colors=[0] * 10,
            difficulty="easy",
            p_cross=0.1,
            avg_degree=0.2,
        )
    ]
    c1 = _reproducibility_checksum(instances, seed=42, optimizer_configs={"n_restarts": 5})
    c2 = _reproducibility_checksum(instances, seed=42, optimizer_configs={"n_restarts": 20})
    assert c1 != c2, "Different optimizer configs should produce different checksums"


def test_reproducibility_checksum_returns_hex_string():
    """Checksum is a non-empty hex string (SHA256 output format).

    WHY: the checksum is stored in JSON artifacts and should be a valid hex
    string that can be verified with standard SHA256 tools.
    """
    instances = [
        GraphColoringInstance(
            instance_id=0,
            n_vertices=5,
            k=3,
            edges=[],
            planted_colors=[0, 1, 2, 0, 1],
            difficulty="easy",
            p_cross=0.0,
            avg_degree=0.0,
        )
    ]
    c = _reproducibility_checksum(instances, seed=0, optimizer_configs={})
    assert isinstance(c, str), f"Expected str, got {type(c)}"
    assert len(c) == 64, f"SHA256 hex should be 64 chars, got {len(c)}"
    assert all(ch in "0123456789abcdef" for ch in c), "Expected hex chars only"
