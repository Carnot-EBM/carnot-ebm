"""REQ-ARC-WMTE-7580 tests for fail-closed E3 world-model support."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot import experiment_7580_v662_arc_verifier_support as exp
from carnot.agentic import arc_competition_agent as competition
from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier


def _row(value: int, *, action: int = 1, changed: bool = True) -> Transition:
    grid = np.asarray([[value]], dtype=np.int64)
    next_grid = np.asarray([[value + int(changed)]], dtype=np.int64)
    return Transition(grid, action, None, next_grid, 0, 0)


def _factory(source: str):
    def build():
        namespace: dict[str, Any] = {"np": np}
        exec(compile(source, "<generated-fixture>", "exec"), namespace)
        return namespace["engine"]

    return build


def test_req_7580_reproduces_exception_denominator_loss_through_current_verifier() -> None:
    """SCENARIO-ARC-WMTE-7580-DENOMINATORS keeps all eight attempts."""

    rows = [_row(i) for i in range(8)]

    def raises_seven(grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        if int(grid[0, 0]) < 7:
            raise RuntimeError("fixture")
        return grid + 1

    current = WorldModelVerifier(rows).score(raises_seven)
    assert current.n == 8
    assert current.n_engine_raised == 7
    assert current.n_changing == 1
    assert current.change_fidelity == 1.0

    source = """\
def engine(grid, action, data=None):
    if int(grid[0, 0]) < 7:
        raise RuntimeError("fixture")
    return grid + 1
"""
    decision = exp.verify_integrity(
        rows,
        _factory(source),
        prompt_transition_ids=(),
        refactor_transition_ids=(),
        heldout_rows=rows,
    )
    assert decision["accepted"] is False
    assert decision["reason"] == "engine_exception"
    assert decision["exact_denominator"] == 8
    assert decision["changing_denominator"] == 8
    assert decision["exception_count"] == 7
    assert decision["change_fidelity_numerator"] == pytest.approx(1.0)
    assert decision["change_fidelity"] == pytest.approx(1.0 / 8.0)


def test_req_7580_reproduces_stateful_order_and_rejects_impurity() -> None:
    """SCENARIO-ARC-WMTE-7580-PURITY rejects an order-sensitive counter."""

    rows = [
        Transition(np.asarray([[base]]), 1, None, np.asarray([[base + step]]), 0, 0)
        for base, step in ((0, 1), (10, 2), (20, 3))
    ]

    def counter_engine():
        state = {"n": 0}

        def engine(grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
            state["n"] += 1
            return grid + state["n"]

        return engine

    chronological = WorldModelVerifier(rows).score(counter_engine())
    reversed_score = WorldModelVerifier(list(reversed(rows))).score(counter_engine())
    assert chronological.accuracy == 1.0
    assert reversed_score.accuracy < 1.0

    source = """\
counter = 0
def engine(grid, action, data=None):
    global counter
    counter += 1
    return grid + counter
"""
    guarded = exp.verify_integrity(
        rows,
        _factory(source),
        heldout_rows=rows,
        prompt_transition_ids=(),
        refactor_transition_ids=(),
    )
    assert guarded["accepted"] is False
    assert guarded["reason"] == "impure_engine"
    assert guarded["purity_passed"] is False
    assert any(row["repeat_same_instance_equal"] is False for row in guarded["support_rows"])


def test_req_7580_one_row_overlap_is_insufficient_support() -> None:
    """SCENARIO-ARC-WMTE-7580-INDEPENDENT-SUPPORT refuses aliased splits."""

    only = _row(4)
    prefix, heldout = exp.current_prefix_heldout([only])
    assert prefix[0] is heldout[0]
    assert WorldModelVerifier(prefix).score(lambda grid, action, data=None: grid + 1).accuracy == 1
    assert WorldModelVerifier(heldout).score(lambda grid, action, data=None: grid + 1).accuracy == 1

    transition_id = exp.transition_id(only)
    guarded = exp.verify_integrity(
        [only],
        _factory("def engine(grid, action, data=None):\n    return grid + 1\n"),
        heldout_rows=[only],
        prompt_transition_ids=(transition_id,),
        refactor_transition_ids=(),
    )
    assert guarded["accepted"] is False
    assert guarded["reason"] == "insufficient_support"
    assert guarded["overlap_transition_ids"] == [transition_id]


def test_req_7580_independent_positive_and_noop_exception_accounting() -> None:
    """SCENARIO-ARC-WMTE-7580-INDEPENDENT-SUPPORT retains honest controls."""

    prompt = [_row(0), _row(1)]
    heldout = [_row(10), _row(20)]
    positive = exp.verify_integrity(
        [*prompt, *heldout],
        _factory("def engine(grid, action, data=None):\n    return grid + 1\n"),
        heldout_rows=heldout,
        prompt_transition_ids=tuple(exp.transition_id(row) for row in prompt),
        refactor_transition_ids=(),
    )
    assert positive["accepted"] is True
    assert positive["reason"] == "accepted"
    assert positive["exact_accuracy"] == 1.0
    assert positive["change_fidelity"] == 1.0
    assert positive["distinct_heldout_transition_count"] == 2
    assert positive["purity_passed"] is True

    noop_rows = [_row(30, changed=False), _row(31, changed=False)]
    raised = exp.verify_integrity(
        noop_rows,
        _factory("def engine(grid, action, data=None):\n    raise ValueError('no answer')\n"),
        heldout_rows=noop_rows,
        prompt_transition_ids=(),
        refactor_transition_ids=(),
    )
    assert raised["noop_denominator"] == 2
    assert raised["noop_hallucination_numerator"] == 2
    assert raised["noop_hallucination_rate"] == 1.0


@pytest.mark.parametrize(
    ("rows", "expected_detail"),
    [
        ([_row(7), _row(7)], "duplicate_heldout_transition_ids"),
        (
            [
                Transition(np.asarray([[9]]), 1, None, np.asarray([[10]]), 0, 0),
                Transition(np.asarray([[9]]), 1, None, np.asarray([[11]]), 0, 0),
            ],
            "contradictory_same_input_transitions",
        ),
    ],
)
def test_req_7580_duplicate_or_contradictory_support_never_passes(
    rows: list[Transition], expected_detail: str
) -> None:
    """SCENARIO-ARC-WMTE-7580-INDEPENDENT-SUPPORT is fail closed."""

    result = exp.verify_integrity(
        rows,
        _factory("def engine(grid, action, data=None):\n    return grid + 1\n"),
        heldout_rows=rows,
        prompt_transition_ids=(),
        refactor_transition_ids=(),
    )
    assert result["accepted"] is False
    assert result["reason"] == "insufficient_support"
    assert expected_detail in result["support_failures"]


def test_req_7580_guard_flag_is_strict_and_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-7580-E3-PARITY-AND-REJECTION preserves defaults."""

    monkeypatch.delenv(exp.GUARD_ENV, raising=False)
    assert exp.integrity_guard_enabled() is False
    for value, expected in (("", False), ("0", False), ("true", False), ("1", True)):
        monkeypatch.setenv(exp.GUARD_ENV, value)
        assert exp.integrity_guard_enabled() is expected


def test_req_7580_e3_call_site_rejects_generated_adversarial_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7580-E3-PARITY-AND-REJECTION reaches E3."""

    rows = [_row(i) for i in range(6)]
    source = """\
counter = 0
def engine(grid, action, data=None):
    global counter
    counter += 1
    return grid + counter
"""
    factory = _factory(source)
    outcome = SimpleNamespace(
        planned=True,
        plan=[{"action": 1}],
        engine=factory(),
        selected_candidate_name="loaded_world_model.py",
        accepted_by_heldout_verifier=True,
        refinement_rounds_used=1,
        skipped="",
    )
    policy = competition.E3AgentPolicy("zz99", proposer=SimpleNamespace(), explore_budget=100)
    attempt: dict[str, Any] = {}

    monkeypatch.setenv(exp.GUARD_ENV, "1")
    guarded = policy._apply_world_model_integrity_guard(
        attempt,
        outcome,
        game="zz99",
        transitions=rows,
        load_engine=lambda game: (factory(), None),
    )
    assert guarded is outcome
    assert outcome.planned is False
    assert outcome.plan == []
    assert outcome.accepted_by_heldout_verifier is False
    assert outcome.skipped == "verifier_integrity_impure_engine"
    assert attempt["verifier_integrity_guard"]["accepted"] is False


def test_req_7580_disabled_real_policy_parity_covers_required_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7580-E3-PARITY-AND-REJECTION compares real calls."""

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    parity = exp.run_disabled_e3_parity(n_actions=6)
    assert parity["passed"] is True
    assert parity["actions_equal"] is True
    assert parity["calls_equal"] is True
    assert parity["provenance_equal"] is True
    assert parity["environment_work_equal"] is True
    assert parity["rng_state_equal"] is True
    assert parity["current_model_calls"] == 0


def test_req_7580_live_protocol_is_frozen_and_registry_prechecked(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7580-PROTOCOL-AND-TERMINAL freezes 24 units."""

    registry = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "\n".join(f"- game: {game}\n  full_game_clear: true" for game in exp.PANEL_GAMES) + "\n",
        encoding="utf-8",
    )
    protocol = exp.build_live_panel_protocol(tmp_path)
    assert protocol["panels"] == {"A": ["su15", "sp80", "ft09"], "B": ["sb26", "g50t", "dc22"]}
    assert protocol["seeds"] == [7582001, 7582002]
    assert protocol["arms"] == ["current_verifier", "integrity_guard"]
    assert protocol["max_actions_per_episode"] == 600
    assert protocol["max_inductions_per_episode"] == 1
    assert protocol["plan_outcome_window_actions"] == 32
    assert len(protocol["rows"]) == 24
    assert all(row["registry_prechecked"] for row in protocol["rows"])
    assert all(row["solve_credit_allowed"] is False for row in protocol["rows"])
    assert protocol["source_derived_masks_allowed"] is False
    assert protocol["source_derived_models_allowed"] is False

    path = tmp_path / exp.LIVE_PROTOCOL_REL
    exp.atomic_json(path, protocol)
    assert exp.validate_live_panel_protocol(json.loads(path.read_text())) == []


def test_req_7580_fixture_measurement_and_independent_reduction() -> None:
    """SCENARIO-ARC-WMTE-7580-DEFECTS retains faults and positive control."""

    measured = exp.measure_fixture_support()
    names = {row["unit"] for row in measured["rows"]}
    assert names == {
        "exception_denominator_loss",
        "stateful_order_dependence",
        "one_row_overlap",
        "independent_positive",
        "contradictory_same_input",
    }
    assert all(row["reproduced_or_control_passed"] for row in measured["rows"])
    assert measured["verifier_support_ready_score"] == 1
    reduced = exp.independent_reduce(measured["rows"])
    assert reduced == measured["independent_reduction"]

    changed = json.loads(json.dumps(measured["rows"]))
    changed[0]["guard_accepted"] = True
    assert exp.independent_reduce(changed)["verifier_support_ready_score"] == 0


def test_req_7580_artifact_schema_and_validation(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7580-PROTOCOL-AND-TERMINAL validates terminal fields."""

    registry = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "\n".join(f"- game: {game}\n  full_game_clear: true" for game in exp.PANEL_GAMES) + "\n",
        encoding="utf-8",
    )
    fixtures = exp.measure_fixture_support()
    protocol = exp.build_live_panel_protocol(tmp_path)
    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_date="20260924",
        duration_s=1.25,
        fixtures=fixtures,
        protocol=protocol,
        source_hashes={"fixture": "sha256:" + "0" * 64},
        validation_receipts=[{"name": "unit", "exit_code": 0, "passed": True}],
        e2e_receipts=[{"name": "cold_replay", "exit_code": 0, "passed": True}],
        terminal_receipts=[],
    )
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["flagged_adversarial"] is False
    assert artifact["MODEL_SPECS"] == []
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["production_defaults_changed"] is False
    assert artifact["verifier_support_ready_score"] == 1
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is True
    assert exp.validate_artifact(artifact) == []

    artifact["production_defaults_changed"] = True
    assert "production_defaults_changed" in exp.validate_artifact(artifact)


def test_req_7580_blocked_gate_summary_is_complete() -> None:
    """REQ-ARC-WMTE-7580 missing external inputs produce a complete blocked row."""

    blocked = exp.blocked_artifact(
        run_date="20260924",
        duration_s=0.2,
        check="registry_exists",
        upstream="ops",
        path="ops/arc_solve_registry.yaml",
        field="exists",
        expected=True,
        observed=False,
    )
    assert blocked["honest_verdict"] == "complete_blocked_registry_exists"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["verifier_support_ready_score"] == 0
    assert blocked["gate_check_summary"] == {
        "check": "registry_exists",
        "upstream": "ops",
        "path": "ops/arc_solve_registry.yaml",
        "field": "exists",
        "op": "==",
        "expected": True,
        "observed": False,
    }


def test_req_7580_guard_failure_branches_retain_attempted_rows() -> None:
    """REQ-ARC-WMTE-7580 malformed engines cannot escape denominators."""

    rows = [_row(1), _row(2)]
    malformed = exp.verify_integrity(
        rows,
        _factory("def engine(grid, action, data=None):\n    return 'bad'\n"),
        heldout_rows=rows,
    )
    assert malformed["exact_denominator"] == 2
    assert malformed["exception_count"] == 2
    assert all("OutputTypeError" in row["exception"] for row in malformed["support_rows"])

    missing = exp.verify_integrity(rows, None, heldout_rows=rows)
    assert missing["reason"] == "insufficient_support"
    assert missing["exception_count"] == 2
    assert "missing_fresh_engine_factory" in missing["support_failures"]

    def broken_factory():
        raise LookupError("no isolated engine")

    broken = exp.verify_integrity(rows, broken_factory, heldout_rows=rows)
    assert broken["exception_count"] == 2
    assert all(row["exception"].startswith("FactoryLookupError") for row in broken["support_rows"])

    calls = {"n": 0}

    def second_factory_fails():
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("fresh failure")
        return lambda grid, action, data=None: grid + 1

    no_fresh = exp.verify_integrity(rows, second_factory_fails, heldout_rows=rows)
    assert no_fresh["reason"] == "impure_engine"
    assert all(
        row["fresh_exception"].startswith("FactoryRuntimeError") for row in no_fresh["support_rows"]
    )


def test_req_7580_default_split_threshold_and_helpers(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-ARC-WMTE-7580 covers the default split and exact-threshold refusal."""

    started = 1.0
    exp.progress(started, "unit", "boundary", detail="truthful")
    assert "phase=unit event=boundary" in capsys.readouterr().out
    assert exp.canonical_hash(np.int64(3)).startswith("sha256:")

    rows = [_row(index) for index in range(8)]
    wrong = exp.verify_integrity(
        rows,
        _factory("def engine(grid, action, data=None):\n    return grid + 2\n"),
    )
    assert wrong["reason"] == "exact_below_threshold"
    assert wrong["exact_denominator"] >= 2


def test_req_7580_apply_guard_default_off_and_explicit_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7580 keeps off parity and accepts explicit split metadata."""

    rows = [_row(index) for index in range(4)]
    outcome = SimpleNamespace(
        planned=True,
        plan=[{"action": 1}],
        selected_candidate_name="loaded_world_model.py",
        accepted_by_heldout_verifier=True,
        refinement_rounds_used=1,
        skipped="",
    )
    monkeypatch.delenv(exp.GUARD_ENV, raising=False)
    attempt: dict[str, Any] = {}
    assert (
        exp.apply_e3_integrity_guard(
            attempt,
            outcome,
            game="zz99",
            transitions=rows,
            proposal_transitions=rows[:2],
            load_engine=lambda game: (
                _factory("def engine(grid, action, data=None):\n    return grid + 1\n")(),
                None,
            ),
        )
        is outcome
    )
    assert attempt == {}

    monkeypatch.setenv(exp.GUARD_ENV, "1")
    outcome.refinement_rounds_used = 2
    exp.apply_e3_integrity_guard(
        attempt,
        outcome,
        game="zz99",
        transitions=rows,
        proposal_transitions=rows[:2],
        load_engine=lambda game: (
            _factory("def engine(grid, action, data=None):\n    return grid + 1\n")(),
            None,
        ),
    )
    assert attempt["verifier_integrity_guard"]["selected_candidate_name"] == "loaded_world_model.py"


def test_req_7580_protocol_validator_names_each_drift(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7580 reports every frozen live-protocol field independently."""

    registry = tmp_path / "ops/arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "\n".join(f"- game: {game}\n  full_game_clear: true" for game in exp.PANEL_GAMES),
        encoding="utf-8",
    )
    valid = exp.build_live_panel_protocol(tmp_path)
    mutations = {
        "panels": None,
        "seeds": None,
        "arms": None,
        "max_actions_per_episode": 599,
        "max_inductions_per_episode": 2,
        "plan_outcome_window_actions": 31,
        "rows": [],
        "source_derived_masks_allowed": True,
        "source_derived_models_allowed": True,
    }
    for field, value in mutations.items():
        changed = json.loads(json.dumps(valid))
        changed[field] = value
        assert field in exp.validate_live_panel_protocol(changed)

    changed = json.loads(json.dumps(valid))
    changed["rows"][0]["registry_prechecked"] = False
    assert "registry_or_solve_credit" in exp.validate_live_panel_protocol(changed)

    malformed = tmp_path / "invalid.yaml"
    malformed.write_text("games: not-a-list\n", encoding="utf-8")
    assert exp._registry_games(malformed) == {}


def _valid_artifact(tmp_path: Path) -> dict[str, Any]:
    registry = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "\n".join(f"- game: {game}\n  full_game_clear: true" for game in exp.PANEL_GAMES),
        encoding="utf-8",
    )
    fixtures = exp.measure_fixture_support()
    return exp.build_artifact(
        repo_root=tmp_path,
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        fixtures=fixtures,
        protocol=exp.build_live_panel_protocol(tmp_path),
        source_hashes={},
        validation_receipts=[],
        e2e_receipts=[],
        terminal_receipts=[],
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("honest_verdict", "bad", "honest_verdict"),
        ("verdict_class", "bad", "verdict_class"),
        ("MODEL_SPECS", ["bad"], "model_declaration"),
        ("invocation_counts", {"load_calls": 1}, "invocation_counts"),
        ("inference_substrate_class", "live", "inference_substrate_class"),
        ("solve_provenance", "solve", "solve_provenance"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("field_principles", {}, "field_principles"),
        ("live_panel_protocol", None, "live_panel_protocol"),
        ("fixture_comparisons", None, "fixture_comparisons"),
        ("rows", [], "rows"),
        ("support_rows", [], "support_rows"),
        ("effective_flags", {}, "effective_flags"),
        ("verifier_support_ready_score", 2, "verifier_support_ready_score"),
    ],
)
def test_req_7580_artifact_validator_fails_closed(
    tmp_path: Path, field: str, value: Any, error: str
) -> None:
    """REQ-ARC-WMTE-7580 cold validation names malformed terminal fields."""

    artifact = _valid_artifact(tmp_path)
    artifact[field] = value
    assert error in exp.validate_artifact(artifact)


def test_req_7580_artifact_validator_checks_reductions_and_ready_claims(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7580 recomputes comparisons and readiness claims."""

    artifact = _valid_artifact(tmp_path)
    artifact["independent_reduction"] = {}
    artifact["rows"][0]["arm"] = "bad"
    artifact["verdict_class"] = "null"
    artifact["flagged_adversarial"] = True
    artifact["e3_disabled_parity"] = {"passed": False}
    artifact["e3_call_site_rejection"] = {"passed": False}
    artifact["source_artifact_hashes"]["drift"] = "sha256:" + "1" * 64
    errors = exp.validate_artifact(artifact)
    assert {
        "independent_reduction",
        "rows_per_arm",
        "ready_verdict_class",
        "ready_flagged",
        "e3_disabled_parity",
        "e3_call_site_rejection",
        "reproducibility_checksum",
    } <= set(errors)


def test_req_7580_real_preconditions_and_source_hashes_are_bound() -> None:
    """REQ-ARC-WMTE-7580 authenticates the owned worktree before measurement."""

    root = Path(exp.__file__).resolve().parents[2]
    rows, hashes = exp.collect_preconditions(root)
    assert rows
    assert all(row["passed"] for row in rows)
    assert set(exp.PREREQUISITE_PATHS) <= set(hashes)
    expanded = exp._source_hashes(root, hashes)
    assert exp.AFFECTED_MANIFEST["changed_modules"][0] in expanded
    assert exp.AFFECTED_MANIFEST["tests"][0] in expanded


def test_req_7580_measure_real_e3_call_site_rejection() -> None:
    """SCENARIO-ARC-WMTE-7580-E3-PARITY-AND-REJECTION uses the live E3 seam."""

    measured = exp.measure_e3_call_site_rejection()
    assert measured["passed"] is True
    assert measured["receipt"]["reason"] == "impure_engine"
    assert measured["outcome"]["planned"] is False


def test_req_7580_cold_and_independent_replays(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7580-PROTOCOL-AND-TERMINAL re-reduces raw rows."""

    artifact = _valid_artifact(tmp_path)
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path) == []
    assert exp.independent_replay(path) == []

    artifact["fixture_comparisons"] = None
    exp.atomic_json(path, artifact)
    assert exp.independent_replay(path) == ["fixture_comparisons"]

    artifact = _valid_artifact(tmp_path)
    artifact["independent_reduction"] = {}
    exp.atomic_json(path, artifact)
    assert exp.independent_replay(path) == ["independent_reduction"]

    artifact["fixture_comparisons"][0]["guard_accepted"] = True
    exp.atomic_json(path, artifact)
    cold_errors = exp.cold_replay(path)
    assert "cold_fixture_rows_mismatch" in cold_errors
    assert "cold_reduction_mismatch" in cold_errors


def test_req_7580_blocked_validator_rejects_bad_summary() -> None:
    """REQ-ARC-WMTE-7580 requires exact blocked-gate provenance."""

    blocked = exp.blocked_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.0,
        check="missing",
        upstream="worktree",
        path="missing",
        field="is_file",
        expected=True,
        observed=False,
    )
    blocked["gate_check_summary"] = {}
    blocked["verifier_support_ready_score"] = 1
    assert exp.validate_artifact(blocked) == ["gate_check_summary", "blocked_ready_score"]
