"""PROSE MUST NOT SATISFY A CHECK THAT IS ASKING FOR A NUMBER OR A BOOLEAN.

REQ-QA-TYPED-FIELDS-1 (2026-07-27 adversarial review, finding 6; CLAUDE.md "QA-Layer
Authenticity Discipline").

THE ORIGIN INCIDENT, exactly. `check_value_routing_cost_control_overclaim` asked whether an
artifact reported `per_node_feature_cost_ms` (a finite cost in milliseconds) and
`sim_timed_out` (a boolean). It answered that question with:

    cost_values = _real_field_values(d, "per_node_feature_cost_ms")
    if not cost_values: omitted.append("per_node_feature_cost_ms")

`_real_field_values` collects ANY value found under the wanted key, and the caller only
tested the list for non-emptiness -- so a STRING satisfied it. A first draft of the
2026-07-27 first-win artifact wrote those two exact key names with explanatory prose as
their values and the WARN CLEARED: adversarial_verify went from 1 flagged artifact to 0,
solely because a string is truthy. The artifact's authors noticed and renamed the keys to
avoid accidentally clearing their own flag, but the HOLE stayed open -- and it is a hole in
the layer that decides whether every other result in this project counts as clean.

WHY THIS IS ITS OWN TEST FILE RATHER THAN AN ASSERTION ADDED TO AN EXISTING ONE: this
project's repeated lesson is that a guard which does not fire on its OWN origin incident
reads as reassurance. These tests use the literal prose values from that draft, so a future
refactor that reintroduces bare non-emptiness fails here with the incident in the message.

SCENARIO-PROSE-DOES-NOT-CLEAR   prose under the wanted key leaves the WARN standing.
SCENARIO-REAL-VALUE-CLEARS      a finite number / real boolean still clears it (no over-fire).
SCENARIO-PRINCIPLE-WRAPPED      an honestly principle-annotated {"principle","value"} field
                                still clears it -- the filter rejects the wrong TYPE, not
                                annotation (CLAUDE.md Principle-Annotated Artifact Fields).
SCENARIO-CONTAINER-KEYS         `metric_harness_fixed` is legitimately a bool in one real
                                artifact and a dict in another, so it uses the weaker
                                "structured" filter; demanding a bool would over-fire on
                                results/experiment_4664_l2_goal_predicate_induction_live.json.
"""

from __future__ import annotations

import importlib.util
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AV = os.path.join(_REPO, "scripts", "adversarial_verify.py")


def _load():
    spec = importlib.util.spec_from_file_location("adversarial_verify_typed", _AV)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


av = _load()

# The LITERAL strings from the draft that cleared the WARN. Kept verbatim so the regression
# is against the real incident and not a stylised stand-in.
_PROSE_COST = (
    "NOT MEASURED by this harness, and deliberately NOT invented to clear the flag. This "
    "measurement times whole cells (elapsed_s per row), not per-node feature extraction."
)
_PROSE_TIMEOUT = (
    "No cell timed out or errored: n_cell_errors is 0 in every arm and every cell ran its "
    "full 200-action budget or terminated on its own first-win break."
)


def test_prose_under_a_numeric_key_is_not_a_number() -> None:
    assert (
        av._typed_field_values(
            {"per_node_feature_cost_ms": _PROSE_COST}, "per_node_feature_cost_ms", "number"
        )
        == []
    ), "a prose string is being accepted as a finite cost in milliseconds"


def test_prose_under_a_boolean_key_is_not_a_boolean() -> None:
    assert (
        av._typed_field_values({"sim_timed_out": _PROSE_TIMEOUT}, "sim_timed_out", "bool") == []
    ), "a prose string is being accepted as a boolean timeout control"


def test_real_values_still_clear_the_filter() -> None:
    """The no-over-fire control: if this failed, the fix would flag every honest artifact."""
    assert av._typed_field_values(
        {"per_node_feature_cost_ms": 3.25}, "per_node_feature_cost_ms", "number"
    ) == [3.25]
    assert av._typed_field_values({"sim_timed_out": False}, "sim_timed_out", "bool") == [False]
    # 0 is a legitimate cost and False is a legitimate answer; neither may be dropped for
    # being falsy -- that would be the mirror-image bug of the one under repair.
    assert av._typed_field_values(
        {"per_node_feature_cost_ms": 0}, "per_node_feature_cost_ms", "number"
    ) == [0]


def test_a_bool_is_not_accepted_as_a_number() -> None:
    """True is an int in Python. A gate asking for a per-node cost in ms must not read
    `per_node_feature_cost_ms: true` as "0.0 ms, measured"."""
    assert (
        av._typed_field_values(
            {"per_node_feature_cost_ms": True}, "per_node_feature_cost_ms", "number"
        )
        == []
    )


def test_principle_wrapped_values_are_unwrapped_not_rejected() -> None:
    """CLAUDE.md allows ANY field to be written as {"principle": ..., "value": ...}. Origin
    bug #2 of the QA-Layer Authenticity Discipline was a check that did not unwrap; this fix
    must not reintroduce it in the opposite direction."""
    wrapped_num = {"per_node_feature_cost_ms": {"principle": "cost control", "value": 1.5}}
    assert av._typed_field_values(wrapped_num, "per_node_feature_cost_ms", "number") == [1.5]
    wrapped_bool = {"sim_timed_out": {"principle": "no truncation", "value": False}}
    assert av._typed_field_values(wrapped_bool, "sim_timed_out", "bool") == [False]
    # ...but a wrapper whose VALUE is prose is still prose.
    wrapped_prose = {"sim_timed_out": {"principle": "x", "value": _PROSE_TIMEOUT}}
    assert av._typed_field_values(wrapped_prose, "sim_timed_out", "bool") == []


def test_structured_filter_accepts_both_real_shapes_of_metric_harness_fixed() -> None:
    """`metric_harness_fixed` is a bool in results/experiment_4669_integration_gate.json and
    a {break_at_first_win, port, qwen_port_props_verified, target_levels} dict in
    results/experiment_4664_l2_goal_predicate_induction_live.json. Both are honest; only a
    bare string is the defect."""
    as_bool = av._typed_field_values(
        {"metric_harness_fixed": True}, "metric_harness_fixed", "structured"
    )
    assert as_bool == [True], as_bool
    as_dict = {"metric_harness_fixed": {"break_at_first_win": False, "target_levels": 2}}
    assert av._typed_field_values(as_dict, "metric_harness_fixed", "structured"), as_dict
    as_prose = {"metric_harness_fixed": "the degenerate harness fixed (target_levels>=2 ...)"}
    assert av._typed_field_values(as_prose, "metric_harness_fixed", "structured") == []


def test_the_warn_stands_on_the_prose_artifact_and_clears_on_the_real_one() -> None:
    """END TO END through the real check, not just the helper -- the helper being right does
    not prove the caller uses it."""
    base = {
        "experiment": "arc_value_routing_live_probe",
        "schema": "carnot.arc.value_routing.v1",
        "game": "lp85",
        "solve_provenance": "live_agent_self_discovery",
        "value_weight": 0.5,
        "per_node_feature_cost_ms": _PROSE_COST,
        "sim_timed_out": _PROSE_TIMEOUT,
    }
    flags: list = []
    av.check_value_routing_cost_control_overclaim(dict(base), flags)
    omitted = [f for f in flags if f.kind == av.VALUE_ROUTING_COST_CONTROL_OMITTED_KIND]
    assert omitted, (
        "PROSE CLEARED THE COST-CONTROL WARN -- the origin incident is live again; "
        f"flags={[f.kind for f in flags]}"
    )
    assert "per_node_feature_cost_ms" in omitted[0].detail
    assert "sim_timed_out" in omitted[0].detail

    real = dict(base, per_node_feature_cost_ms=2.0, sim_timed_out=False)
    flags2: list = []
    av.check_value_routing_cost_control_overclaim(real, flags2)
    assert not [f for f in flags2 if f.kind == av.VALUE_ROUTING_COST_CONTROL_OMITTED_KIND], (
        f"real measured values must clear the WARN; got {[f.detail for f in flags2]}"
    )


def test_per_game_dict_shape_is_accepted_not_flagged() -> None:
    """THE FALSE POSITIVE THIS FIX ALMOST SHIPPED. A first draft demanded a BARE bool, which
    over-fired on the real shape these fields use: `goal_predicate_satisfiable: {"lp85":
    true}` is a per-game dict, and tests/python/test_adversarial_verify_hardening_4671.py
    exercises exactly that. Requiring the top value to be the scalar would have flagged an
    honest artifact -- the mirror image of the defect under repair, and strictly worse,
    because a false positive quarantines real work and trains the operator to ignore the
    gate. Leaf recovery accepts it while still rejecting prose."""
    per_game = {"goal_predicate_satisfiable": {"lp85": True, "sp80": False}}
    got = av._typed_field_values(per_game, "goal_predicate_satisfiable", "bool")
    assert sorted(got) == [False, True], got
    prose = {"goal_predicate_satisfiable": "we did not measure this"}
    assert av._typed_field_values(prose, "goal_predicate_satisfiable", "bool") == []


def test_wrapped_and_nested_numbers_come_back_as_numbers_not_containers() -> None:
    """Callers type-test the RETURNED items -- the CRITICAL branch of the cost-control check
    does `any(_is_finite_number(v) for v in cost_values)`. Returning the wrapper dict would
    make that False on an honest principle-annotated artifact and fire a CRITICAL."""
    got = av._typed_field_values(
        {"per_node_feature_cost_ms": {"principle": "x", "value": 1.5}},
        "per_node_feature_cost_ms",
        "number",
    )
    assert got == [1.5] and all(av._is_finite_number(v) for v in got), got
