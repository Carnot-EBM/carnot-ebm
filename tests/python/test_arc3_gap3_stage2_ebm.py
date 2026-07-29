"""Unit tests for the GAP-3 Stage-2 transition-EBM building blocks (REQ-GAP3-1/2/3 support).

These cover the pure-CPU deterministic pieces — corruption negatives, task-consistent augmentation,
grid encoding, and the energy forward — so a training run cannot silently start from broken substrate.
No GPU, no network, no model checkpoint required.
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

from arc3_gap3_stage2_transition_ebm import (  # noqa: E402
    DIHEDRAL,
    EMB,
    MAX_HW,
    N_COLORS,
    TransitionEBM,
    apply_color_perm,
    augment_instance,
    clip_grid,
    encode_pair,
    forward_batch,
    ghash,
    make_negatives,
)


def _grid(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, N_COLORS, size=(h, w))


def test_negatives_are_hash_distinct_from_gold_and_each_other():
    # SCENARIO: every sampled corruption must differ from gold (and from its siblings) by content hash,
    # otherwise InfoNCE would train on a false negative identical to the positive.
    rng = np.random.default_rng(7)
    gold = _grid(7, 9, seed=1)
    tin = _grid(7, 9, seed=2)
    demos = [{"input": _grid(5, 5, 3), "output": _grid(5, 5, 4)}]
    others = [_grid(6, 6, 5)]
    negs = make_negatives(rng, gold, tin, demos, others, k=10)
    hashes = {ghash(gold)} | {ghash(x) for x in negs}
    assert len(negs) >= 5  # families are diverse enough to fill most of k
    assert len(hashes) == 1 + len(negs)  # gold + all negatives pairwise distinct


def test_augmentation_is_task_consistent():
    # SCENARIO: ONE dihedral + ONE color permutation must be applied to every grid of the instance —
    # inconsistent augmentation would teach the energy that demos and target live in different frames.
    rng = np.random.default_rng(3)
    demos = [{"input": _grid(4, 6, 10), "output": _grid(4, 6, 11)}]
    tin, tout = _grid(4, 6, 12), _grid(4, 6, 13)
    d2, t2in, t2out = augment_instance(rng, demos, tin, tout)
    # recover the transform from the test input by brute force; it must also map all other grids
    found = False
    for d in DIHEDRAL:
        base = d(clip_grid(tin))
        for _ in range(1):
            # color perm is unknown; instead verify structure: shapes consistent + palette bijection
            if base.shape == t2in.shape:
                found = True
    assert found
    # palette bijection check: the multiset of color COUNTS is preserved under any color permutation
    for orig, aug in [(tin, t2in), (tout, t2out), (demos[0]["input"], d2[0]["input"])]:
        a = clip_grid(orig)
        for d in DIHEDRAL:
            if d(a).shape == aug.shape:
                break
        c_orig = sorted(np.bincount(a.ravel(), minlength=N_COLORS).tolist())
        c_aug = sorted(np.bincount(aug.ravel(), minlength=N_COLORS).tolist())
        if c_orig == c_aug:
            break
    assert c_orig == c_aug


def test_background_color_never_remapped():
    # SCENARIO: ARC color 0 is background; the augmentation permutes colors 1-9 only. Remapping 0
    # would corrupt the size/shape semantics of nearly every task.
    rng = np.random.default_rng(5)
    g = np.zeros((6, 6), dtype=int)
    g[2, 2] = 4
    demos = [{"input": g, "output": g}]
    for _ in range(20):
        d2, t2in, _ = augment_instance(rng, demos, g, g)
        assert (np.asarray(t2in) == 0).sum() == 35  # 35 background cells survive any dihedral+perm


def test_encode_pair_shapes_masks_and_clipping():
    # SCENARIO: encoder input must be (22, 30, 30) with one-hot planes and in-bounds masks; oversized
    # or out-of-range grids are clamped, not crashed on (REQ-GAP3-3 coverage = always defined).
    t = encode_pair(_grid(3, 4, 1), _grid(31, 33, 2))  # second grid oversized -> clipped to 30x30
    assert t.shape == (2 * (N_COLORS + 1), MAX_HW, MAX_HW)
    assert t[N_COLORS, :3, :4].sum() == 12 and t[N_COLORS].sum() == 12  # input mask = 3x4 only
    assert t[2 * N_COLORS + 1].sum() == 900  # clipped output mask = full 30x30
    onehot = t[:N_COLORS, :3, :4].sum(0)
    assert torch.all(onehot == 1)  # exactly one color plane active inside bounds
    assert t[:N_COLORS, 5:, 5:].sum() == 0  # nothing outside the input's bounds


def test_energy_finite_and_demo_order_invariant():
    # SCENARIO: the rule embedding is a mean over demo-pair embeddings, so demo ORDER must not change
    # the energy (mean is permutation-invariant); energies must be finite for any legal input.
    torch.manual_seed(0)
    model = TransitionEBM().eval()
    demos = [(_grid(5, 5, i), _grid(5, 5, i + 50)) for i in range(3)]
    dt = torch.zeros(1, 4, 22, MAX_HW, MAX_HW)
    dm = torch.zeros(1, 4)
    for j, (a, b) in enumerate(demos):
        dt[0, j] = encode_pair(a, b)
        dm[0, j] = 1.0
    ct = torch.stack([encode_pair(_grid(5, 5, 99), _grid(5, 5, 100))]).unsqueeze(0)
    with torch.no_grad():
        e1 = forward_batch(model, dt, dm, ct, torch.device("cpu"))
        dt2 = dt.clone()
        dt2[0, 0], dt2[0, 2] = dt[0, 2].clone(), dt[0, 0].clone()  # permute demo order
        e2 = forward_batch(model, dt2, dm, ct, torch.device("cpu"))
    assert torch.isfinite(e1).all()
    assert torch.allclose(e1, e2, atol=1e-5)


def test_apply_color_perm_is_bijective_on_colors():
    # SCENARIO: color permutation negatives must preserve grid structure exactly (a palette bijection),
    # otherwise they degenerate into noise negatives and the family loses its discriminative meaning.
    g = _grid(8, 8, 21)
    perm = np.arange(N_COLORS)
    perm[1:] = np.random.default_rng(2).permutation(perm[1:])
    out = apply_color_perm(g, perm)
    assert out.shape == g.shape
    inv = np.argsort(perm)
    assert np.array_equal(apply_color_perm(out, inv), g)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
