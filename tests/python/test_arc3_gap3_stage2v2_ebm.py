"""Unit tests for the GAP-3 Stage-2 v2 building blocks — the structured same-shape near-miss
curriculum and the spatial FiLM energy (the two panel-mandated fixes). CPU, deterministic, asserting.
"""

# Path-resolution traceability: the repo-root/sys.path resolution in this file traces to
# REQ-ARC-WMTE-6043 (centralised output-path resolution). That is the ONLY behaviour in this
# file covered by that requirement -- the GAP-3/GAP-4 assertions below predate spec
# traceability and are recorded as pre-existing debt in ops/known-issues.md, not claimed here.

import sys

import numpy as np
import pytest
import torch

from carnot.paths import repo_root

# The GAP-3/GAP-4 experiment modules live in scripts/experiments/, which is not a
# package, so it has to go on sys.path to be importable. Resolved from the repo root
# rather than hardcoded: a hardcoded absolute path here poisons the WHOLE pytest-xdist
# worker -- every later import in that worker resolves against the operator's checkout,
# so even correctly repo-relative scripts write their output into the wrong tree.
sys.path.insert(0, str(repo_root() / "scripts" / "experiments"))

from arc3_gap3_stage2v2_transition_ebm import (  # noqa: E402
    MAX_HW,
    N_COLORS,
    SpatialTransitionEBM,
    components,
    encode_pair,
    forward_batch_v2,
    ghash,
    near_miss_negatives,
)


def _grid(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, N_COLORS, size=(h, w))


def test_components_labels_4connected_same_color():
    # SCENARIO: the near-miss families operate on connected components; mislabeled components would
    # produce corrupted negatives that are not single-object perturbations.
    g = np.zeros((5, 5), dtype=int)
    g[0, 0:3] = 4  # one 3-cell component
    g[2, 2] = 4  # separate (diagonal does not connect)
    g[4, 0:2] = 7  # different color component
    comps = components(g)
    assert sorted((c, len(cells)) for c, cells in comps) == [(4, 1), (4, 3), (7, 2)]


def test_near_miss_negatives_same_shape_and_distinct():
    # SCENARIO: the 91.5% real-error class is SAME-SHAPE near-misses; every generated negative must
    # preserve gold's shape exactly and differ from gold (and siblings) by content hash.
    rng = np.random.default_rng(11)
    gold = _grid(9, 7, seed=3)
    negs = near_miss_negatives(rng, gold, k=10)
    assert len(negs) >= 6
    hashes = {ghash(gold)}
    for nm in negs:
        assert nm.shape == gold.shape  # same-shape by construction
        h = ghash(nm)
        assert h not in hashes
        hashes.add(h)


def test_near_miss_max_frac_bounds_changed_cells():
    # SCENARIO: the gate-1 definition is <=5% cells changed; max_frac must enforce it so the gate
    # measures the class it claims to measure.
    rng = np.random.default_rng(5)
    gold = _grid(20, 20, seed=8)
    negs = near_miss_negatives(rng, gold, k=8, max_frac=0.05)
    limit = max(1, int(np.ceil(gold.size * 0.05)))
    for nm in negs:
        assert int((nm != gold).sum()) <= limit


def test_spatial_energy_finite_and_demo_order_invariant():
    # SCENARIO: rule embedding is a mean over demo embeddings — demo ORDER must not change E; all
    # energies finite (the FiLM/local-map path must not NaN on masked cells).
    torch.manual_seed(0)
    model = SpatialTransitionEBM().eval()
    demos = [(_grid(6, 6, i), _grid(6, 6, i + 40)) for i in range(3)]
    dt = torch.zeros(1, 4, 22, MAX_HW, MAX_HW)
    dm = torch.zeros(1, 4)
    for j, (a, b) in enumerate(demos):
        dt[0, j] = encode_pair(a, b)
        dm[0, j] = 1.0
    ct = torch.stack([encode_pair(_grid(6, 6, 90), _grid(6, 6, 91))]).unsqueeze(0)
    with torch.no_grad():
        e1 = forward_batch_v2(model, dt, dm, ct, torch.device("cpu"))
        dt2 = dt.clone()
        dt2[0, 0], dt2[0, 2] = dt[0, 2].clone(), dt[0, 0].clone()
        e2 = forward_batch_v2(model, dt2, dm, ct, torch.device("cpu"))
    assert torch.isfinite(e1).all()
    assert torch.allclose(e1, e2, atol=1e-5)


def test_spatial_energy_is_cell_sensitive():
    # SCENARIO: the entire v2 architectural point — a SINGLE flipped cell must change the energy
    # (v1's global mean-pool was structurally insensitive at this resolution). We assert the energies
    # differ; direction is learned, not architectural.
    torch.manual_seed(1)
    model = SpatialTransitionEBM().eval()
    tin = _grid(8, 8, seed=2)
    out = _grid(8, 8, seed=3)
    out2 = out.copy()
    out2[4, 4] = (out2[4, 4] + 1) % N_COLORS
    dt = torch.zeros(1, 4, 22, MAX_HW, MAX_HW)
    dm = torch.zeros(1, 4)
    dt[0, 0] = encode_pair(_grid(8, 8, 5), _grid(8, 8, 6))
    dm[0, 0] = 1.0
    ct = torch.stack([encode_pair(tin, out), encode_pair(tin, out2)]).unsqueeze(0)
    with torch.no_grad():
        E = forward_batch_v2(model, dt, dm, ct, torch.device("cpu"))[0]
    assert abs(float(E[0] - E[1])) > 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
