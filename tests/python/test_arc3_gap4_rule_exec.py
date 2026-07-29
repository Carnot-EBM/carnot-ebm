"""Unit tests for the GAP-4 rule-execution verifier building blocks — sandbox, demo-fit gate, and the
vote-primary gated rerank. No codex, no network: programs are injected as code strings.
"""

# Path-resolution traceability: the repo-root/sys.path resolution in this file traces to
# REQ-ARC-WMTE-6043 (centralised output-path resolution). That is the ONLY behaviour in this
# file covered by that requirement -- the GAP-3/GAP-4 assertions below predate spec
# traceability and are recorded as pre-existing debt in ops/known-issues.md, not claimed here.

import sys

import numpy as np
import pytest

from carnot.paths import repo_root

# The GAP-3/GAP-4 experiment modules live in scripts/experiments/, which is not a
# package, so it has to go on sys.path to be importable. Resolved from the repo root
# rather than hardcoded: a hardcoded absolute path here poisons the WHOLE pytest-xdist
# worker -- every later import in that worker resolves against the operator's checkout,
# so even correctly repo-relative scripts write their output into the wrong tree.
sys.path.insert(0, str(repo_root() / "scripts" / "experiments"))

from arc3_gap4_rule_exec_verifier import (  # noqa: E402
    _extract_code,
    build_rankers,
    demo_fit,
    ghash,
    norm_hamming,
    safe_transform_from_code,
)
from arc3_gap3_stage2_transition_ebm import _pass  # noqa: E402


def test_sandbox_rejects_forbidden_tokens():
    # SCENARIO: codex-written code runs in-process; file/os/network/introspection access must be
    # rejected at the token level before compilation.
    for bad in [
        "import os\ndef transform(g): return g",
        "def transform(g):\n    open('/etc/passwd')\n    return g",
        "def transform(g):\n    return eval('g')",
        "def transform(g):\n    return globals()['g']",
    ]:
        assert safe_transform_from_code(bad) is None


def test_transform_supports_shape_change_and_rejects_illegal():
    # SCENARIO: ARC rules crop/tile/grow — the wrapper must pass shape-changing outputs through,
    # while rejecting non-2D / oversized / out-of-palette outputs as abstention (None).
    fn = safe_transform_from_code("def transform(grid):\n    return np.tile(grid, (2, 2))")
    out = fn(np.ones((3, 3), dtype=int))
    assert out is not None and out.shape == (6, 6)
    fn_bad = safe_transform_from_code("def transform(grid):\n    return grid * 99")
    assert fn_bad(np.ones((3, 3), dtype=int)) is None  # palette violation -> abstain
    fn_crash = safe_transform_from_code("def transform(grid):\n    return grid[100]")
    assert fn_crash(np.ones((3, 3), dtype=int)) is None  # crash -> abstain, never trusted


def test_sandbox_allows_numpy_internal_lazy_imports():
    # SCENARIO (regression — the smoke-run bug): numpy ops like np.unique/np.where lazily call
    # __import__ internally; a bare builtins dict crashed EVERY program (demo_fit=0.0 across the
    # board). The numpy-only __import__ must let these run while still blocking everything else.
    fn = safe_transform_from_code(
        "def transform(grid):\n"
        "    vals, counts = np.unique(grid, return_counts=True)\n"
        "    bg = vals[np.argmax(counts)]\n"
        "    rows = np.where((grid != bg).any(axis=1))[0]\n"
        "    return grid[rows.min():rows.max() + 1] if len(rows) else grid\n"
    )
    g = np.zeros((5, 5), dtype=int)
    g[2, 2] = 3
    out = fn(g)
    assert out is not None and out.shape == (1, 5)


def test_sandbox_blocks_numpy_io_and_type_vectors():
    # SCENARIO (panel hardening, corrigendum_2026_06_10_gap4): np file-IO and type( escape vectors
    # must be rejected at the token level.
    for bad in [
        "def transform(g):\n    return np.load('/tmp/x.npy')",
        "def transform(g):\n    np.save('/tmp/x', g)\n    return g",
        "def transform(g):\n    return np.fromfile('/etc/passwd', dtype=np.int64)[:4].reshape(2,2)",
        "def transform(g):\n    return type(g).__mro__[1]",
    ]:
        assert safe_transform_from_code(bad) is None


def test_sandbox_word_boundary_does_not_false_reject():
    # SCENARIO (regression — the ARC-2 round bug): bare substring matching rejected legitimate numpy
    # code ('type(' matched 'astype(', 'os.' matched 'pos.'), costing a demo-perfect program. The
    # word-boundary check must pass these through while still blocking the bare tokens.
    fn = safe_transform_from_code(
        "def transform(grid):\n"
        "    pos = np.argwhere(grid > 0)\n"
        "    out = grid.astype(np.int64)\n"
        "    return out if len(pos) else grid\n"
    )
    assert fn is not None
    out = fn(np.ones((3, 3), dtype=int))
    assert out is not None and out.shape == (3, 3)


def test_sandbox_times_out_runaway_program():
    # SCENARIO (panel hardening): a runaway loop must abandon the call within EXEC_TIMEOUT_S and
    # abstain (None), not hang the run. Uses a long finite loop ('while True' would also be caught
    # by the timeout, but a bounded loop keeps the leaked thread short-lived).
    fn = safe_transform_from_code(
        "def transform(grid):\n"
        "    x = 0\n"
        "    for i in range(10**10):\n"
        "        x += i\n"
        "    return grid\n"
    )
    import time as _t

    t0 = _t.time()
    out = fn(np.ones((3, 3), dtype=int))
    assert out is None
    assert _t.time() - t0 < 30  # well-bounded: EXEC_TIMEOUT_S=5 plus overhead


def test_demo_fit_gate():
    # SCENARIO: demo_fit is the oracle-free verification — only exact reproduction of every demo
    # output counts; a near-miss program must NOT reach 1.0.
    demos = [
        {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
        {"input": [[0, 0], [2, 0]], "output": [[0, 0], [0, 2]]},
    ]
    fn_right = safe_transform_from_code("def transform(grid):\n    return grid[:, ::-1]")
    fn_wrong = safe_transform_from_code("def transform(grid):\n    return grid")
    assert demo_fit(fn_right, demos) == 1.0
    assert demo_fit(fn_wrong, demos) == 0.0


def test_gated_rerank_promotes_match_and_abstains_safely():
    # SCENARIO: the headline ranker. With a demo-perfect program whose prediction matches a
    # candidate, that candidate is promoted to rank 1; without a gate, the ranker IS vote.
    gold = [[1, 1], [2, 2]]
    junk = [[0, 0], [0, 0]]
    tasks = [
        {  # gated task: prediction == gold (which vote mis-ranks to 2nd)
            "cands": [
                {"votes": 10, "q_mean": 0.5, "correct": False, "grid": junk},
                {"votes": 9, "q_mean": 0.5, "correct": True, "grid": gold},
            ],
            "prog": {"demo_perfect": True, "pred_hash": ghash(gold), "pred_grid": gold},
        },
        {  # abstained task: no program -> pure vote order preserved
            "cands": [
                {"votes": 10, "q_mean": 0.5, "correct": True, "grid": gold},
                {"votes": 9, "q_mean": 0.5, "correct": False, "grid": junk},
            ],
            "prog": None,
        },
    ]
    rankers = build_rankers(tasks)
    got = {name: _pass(tasks, key, ks=(1,)) for name, key in rankers.items()}
    assert got["TRM_VOTE"]["pass@1"] == 0.5  # vote misses the gated task at rank 1
    assert got["GAP4_GATED"]["pass@1"] == 1.0  # gate promotes gold; abstained task untouched


def test_norm_hamming_orders_shape_mismatch_below_same_shape():
    # SCENARIO: the graded energy must rank ANY same-shape candidate above a shape-mismatched one
    # (shape mismatch with a demo-perfect prediction is maximal rule-inconsistency).
    pred = np.zeros((4, 4), dtype=int)
    same_far = np.ones((4, 4), dtype=int)
    wrong_shape = np.zeros((2, 2), dtype=int)
    assert norm_hamming(pred, pred) == 0.0
    assert norm_hamming(same_far, pred) <= 1.0 < norm_hamming(wrong_shape, pred)


def test_extract_code_takes_last_transform_block():
    text = "blah\n```python\ndef helper(): pass\n```\ntext\n```python\ndef transform(grid):\n    return grid\n```"
    code = _extract_code(text)
    assert code is not None and "def transform" in code


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
