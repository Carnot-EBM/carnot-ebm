"""Invariants this measurement's own harness must hold. Run before the analysis is believed.

These are not tests of the shipped code (that is
`tests/python/test_arc_inert_engine_rejection_2026_08_01.py`). They are tests of the MEASUREMENT:
each one pins a property that, if it silently broke, would make the reported numbers describe
something other than what the artifact says they describe.

SCENARIO-ARC-FCP-5699-43-PROBE-IS-THE-VALIDATED-MEASURE
SCENARIO-ARC-FCP-5699-43-DEFAULT-OFF-IS-BYTE-FOR-BYTE-THE-OLD-PATH
SCENARIO-ARC-FCP-5699-43-OUTCOME-IS-NOT-THE-TREATMENT
SCENARIO-ARC-FCP-5699-43-CLUSTER-AT-THE-GAME
SCENARIO-ARC-FCP-5699-43-MISSING-IS-NOT-ZERO
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot")
VALIDATED = REPO / "results/arc_metric_validity_20260801/score_worker.py"


def _executable_ast(path: pathlib.Path, name: str) -> str:
    """The function's AST with docstrings stripped, so prose may differ and code may not."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            src = textwrap.dedent(
                "\n".join(path.read_text().splitlines()[node.lineno - 1 : node.end_lineno])
            )
            fn = ast.parse(src).body[0]
            fn.body = [
                s
                for s in fn.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            ]
            return ast.dump(fn)
    raise AssertionError(f"{name} not found in {path}")


def test_probe_is_the_measure_that_was_validated():
    """`probe_depth_reached` is quoted from the metric-validity run as the measure that PREDICTS
    plannability (AUC 0.787) where change_fidelity does not. A number computed by a
    reimplementation of that probe is not that measure, and pooling it with those results would
    be a category error. The first draft of `score_worker.py` claimed a verbatim copy while
    having renamed one output field, dropped another, and reshaped the error field -- so this is
    checked mechanically rather than by reading the docstring."""
    assert _executable_ast(HERE / "score_worker.py", "_state_graph_probe") == _executable_ast(
        VALIDATED, "_state_graph_probe"
    )


def test_the_flag_under_test_is_shipped_default_off():
    """If the flag defaulted ON, the control arm would not be the shipped path and the whole
    comparison would be ON vs ON. `run_ab.py` also checks this as a precondition; this asserts it
    without needing a GPU or a collection run."""
    import sys

    sys.path.insert(0, str(REPO / "python"))
    import os

    from carnot.agentic import arc_executable_world_model as e3

    os.environ.pop("CARNOT_ARC_INDUCE_REJECT_INERT", None)
    assert e3._reject_inert_engines() is False


def test_the_outcome_definition_is_not_the_treatment():
    """THE CIRCULARITY CHECK. The primary is the SHIPPED `validate_engine_code`, which must call
    an inert engine CLEAN. If inertness were ever folded into it, the treatment and the outcome
    would be the same object and arm B would win by construction rather than by measurement."""
    import sys

    sys.path.insert(0, str(REPO / "python"))
    import numpy as np

    from carnot.agentic import arc_engine_static_validation as sv

    class _T:
        def __init__(self, g, a):
            self.grid, self.action, self.data = g, a, None

    g = np.zeros((6, 6), dtype=int)
    trans = [_T(g.copy(), a) for a in (1, 2, 3)]
    identity = "def engine(grid, action, data=None):\n    return grid.copy()\n"
    assert sv.validate_engine_code(identity, transitions=trans) == []
    assert sv.engine_inertness_defect(identity, trans) is not None


def test_analysis_clusters_at_the_game():
    """Pseudo-replication is the failure this project has already committed once: treating
    replicates as independent trials inflated a p from 0.125 to 0.049 on 2026-07-31. Assert that
    the analysis averages replicates into a per-game mean BEFORE the sign test, so the n of every
    test is a count of games."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("an", HERE / "analyse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # two games, three replicates each; the treatment wins every replicate of game A only.
    cells = {}
    for rep in range(3):
        cells[("A", rep)] = {"off": {"usable": False}, "on": {"usable": True}}
        cells[("B", rep)] = {"off": {"usable": True}, "on": {"usable": True}}

    def metric(r):
        return 1.0 if r["usable"] else 0.0

    deltas, detail = mod.per_game_means(cells, metric)
    assert set(deltas) == {"A", "B"}, "units must be games, not cells"
    assert detail["A"]["n_replicates_used"] == 3
    res = mod.sign_test(deltas)
    assert res["n_pairs"] == 2, "6 cells must collapse to 2 paired units"
    assert res["n_discordant"] == 1 and res["n_positive"] == 1
    # 1 discordant pair can never reach 0.05, and the report must say so rather than call it null
    assert res["min_reachable_two_sided_p_at_this_discordance"] == 1.0


def test_a_missing_observation_is_never_a_zero():
    """A cell the generator failed to answer must be EXCLUDED, not scored 0. Scoring it 0 would
    convert an infrastructure failure into evidence against whichever arm happened to hit it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("an2", HERE / "analyse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cells = {
        ("A", 0): {
            "off": {"missing": True, "usable": False},
            "on": {"missing": False, "usable": True},
        },
        ("A", 1): {
            "off": {"missing": False, "usable": True},
            "on": {"missing": False, "usable": True},
        },
    }

    def m_usable(r):
        return None if r.get("missing") else (1.0 if r.get("usable") else 0.0)

    deltas, detail = mod.per_game_means(cells, m_usable)
    assert detail["A"]["n_replicates_used"] == 1, "the missing replicate must be dropped"
    assert detail["A"]["replicates_used"] == [1]
    assert deltas["A"] == 0.0, "a dropped pair must not manufacture a delta"
