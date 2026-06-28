"""Tests for Exp 4929 .454 ARC bank and efficiency audit.

Spec refs: REQ-CAPSTONE-4929, SCENARIO-CAPSTONE-4929,
SCENARIO-CAPSTONE-4929-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4929_bank_and_efficiency_audit as exp4929


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry_text(*, sp80: int = 2, su15: int = 2) -> str:
    return f"""schema_version: 1
updated: '2026-06-28'
reproducible_total_levels: 69
games:
- game: sp80
  reproducibility: reproduced
  levels_reproduced: {sp80}
  solver: GameAdapter _sp80 plus scripts/arc_loop_solve.py.
  latest_exp4925_levelup_attempt:
    artifact: results/experiment_4925_levelup_attempt.json
    loop_artifact: results/arc_loop_solve_sp80.json
    offline_reproduced: false
    reproduced_levels: {sp80}
    new_levels_banked: 0
    residual_cause: duplicate_depth
    solve_provenance: live_agent_self_discovery
- game: su15
  reproducibility: reproduced
  levels_reproduced: {su15}
  solver: GameAdapter _su15 plus scripts/arc_loop_solve.py.
  latest_exp4926_levelup_attempt:
    artifact: results/experiment_4926_levelup_attempt.json
    loop_artifact: results/arc_loop_solve_su15.json
    offline_reproduced: false
    reproduced_levels: {su15}
    new_levels_banked: 0
    residual_cause: duplicate_depth
    solve_provenance: live_agent_self_discovery
"""


def _bank_artifact(
    *,
    game: str,
    exp_id: int,
    reached: int = 2,
    prior: int = 2,
    provenance: str = "live_agent_self_discovery",
    live_path: bool = True,
    outer_loop: bool = False,
) -> dict[str, object]:
    return {
        "experiment": f"experiment_{exp_id}_levelup_attempt",
        "honest_verdict": f"complete_{game}_no_new_level_residual_duplicate_depth",
        "target_game": game,
        "solve_provenance": provenance,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "new_levels_banked": 0,
        "prior_reproduced_level": prior,
        "live_path_reachable": live_path,
        "standing_loop_result_path": f"results/arc_loop_solve_{game}.json",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached,
            "reached_level": reached,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "registry_update": {
            "target_game": game,
            "prior_game_levels": prior,
            "new_game_levels": reached,
            "banked_levels": max(0, reached - prior),
            "reason": "duplicate_depth" if reached <= prior else "banked_offline_reproduced_level",
        },
        "offline_ground_truth_bfs": outer_loop,
    }


def _loop_artifact(game: str, *, reached: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "mode": "standing_arc_loop_offline_no_quota",
        "reached_level": reached,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached if reproduced else 0,
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached,
            "reached_level": reached,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
    }


def _efficiency_artifact(**overrides: object) -> dict[str, object]:
    artifact: dict[str, object] = {
        "experiment": "experiment_4933_matm_similarity_retrieval_efficiency",
        "honest_verdict": "retired_action_efficiency_null_reported_honestly",
        "verifier_is_oracle": False,
        "baseline_kind": "submitted_exact_hash",
        "baseline_hash_matches_submitted": True,
        "submitted_exact_hash_baseline": True,
        "zero_reached_level_regression": True,
        "reached_level_regressions": [],
        "parity_test_green": True,
        "leak_check": {"passed": True, "leak_detected": False, "same_state_target_shortcut": False},
        "same_state_target_shortcut": False,
        "efficiency_disposition": "retired_null",
        "null_reported_honestly": True,
    }
    artifact.update(overrides)
    return artifact


def _write_inputs(root: Path, *, include_d: bool = True, registry_text: str | None = None) -> None:
    (root / "results").mkdir()
    (root / "ops").mkdir()
    (root / "scripts").mkdir()
    (root / "openspec" / "capabilities" / "capstone").mkdir(parents=True)
    (root / "scripts" / "adversarial_verify.py").write_text("", encoding="utf-8")
    (root / "scripts" / "summarize_artifact.py").write_text("", encoding="utf-8")
    (root / "scripts" / "arc_orphan_solver_lint.py").write_text("", encoding="utf-8")
    (root / "openspec" / "capabilities" / "capstone" / "spec.md").write_text(
        "REQ-CAPSTONE-4929\nSCENARIO-CAPSTONE-4929\n",
        encoding="utf-8",
    )
    (root / exp4929.REGISTRY_RELATIVE_PATH).write_text(
        registry_text or _registry_text(),
        encoding="utf-8",
    )
    banks = (
        ("results/experiment_4925_levelup_attempt.json", _bank_artifact(game="sp80", exp_id=4925)),
        ("results/experiment_4926_levelup_attempt.json", _bank_artifact(game="su15", exp_id=4926)),
        ("results/arc_loop_solve_sp80.json", _loop_artifact("sp80")),
        ("results/arc_loop_solve_su15.json", _loop_artifact("su15")),
    )
    for relative, payload in banks:
        (root / relative).write_text(json.dumps(payload), encoding="utf-8")
    if include_d:
        (root / exp4929.EFFICIENCY_RELATIVE_PATH).write_text(
            json.dumps(_efficiency_artifact()),
            encoding="utf-8",
        )


def test_req_capstone_4929_spec_declares_audit_contract() -> None:
    """REQ-CAPSTONE-4929: OpenSpec names every required check and field."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4929.SPEC_REFS:
        assert ref in spec
    for check in exp4929.CHECK_KEYS:
        assert check in spec
    for field, principle in exp4929.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec
    assert exp4929.RESULT_RELATIVE_PATH in spec


def test_scenario_capstone_4929_bank_duplicate_fails_only_not_duplicate() -> None:
    """SCENARIO-CAPSTONE-4929: duplicate L2 replays are genuine but not bankable."""

    registry = yaml.safe_load(_registry_text())
    evidence = exp4929.audit_bank(
        label="A1",
        artifact=_bank_artifact(game="sp80", exp_id=4925),
        loop_artifact=_loop_artifact("sp80"),
        registry=registry,
        lint_result={"passed": True},
    )

    assert evidence["checks"] == {
        "reproduction_genuine": True,
        "not_duplicate": False,
        "self_discovery_provenance": True,
        "live_path_reachable": True,
    }
    assert evidence["failure_reasons"] == ["A1_not_duplicate_failed_duplicate_depth_sp80_L2"]
    assert evidence["registry_prior_level"] == 2
    assert evidence["claimed_reached_level"] == 2
    assert evidence["loop_cross_check"]["offline_reproduced"] is True


def test_req_capstone_4929_bank_checks_fail_closed_for_fabrication_patterns() -> None:
    """REQ-CAPSTONE-4929: bank checks catch fabricated, off-path, and outer-loop evidence."""

    registry = yaml.safe_load(_registry_text())
    evidence = exp4929.audit_bank(
        label="A2",
        artifact=_bank_artifact(
            game="su15",
            exp_id=4926,
            provenance="outer_loop_re",
            live_path=False,
            outer_loop=True,
        ),
        loop_artifact=_loop_artifact("su15", reached=1, reproduced=False),
        registry=registry,
        lint_result={"passed": False, "returncode": 1},
    )

    assert evidence["checks"] == {
        "reproduction_genuine": False,
        "not_duplicate": False,
        "self_discovery_provenance": False,
        "live_path_reachable": False,
    }
    assert evidence["failure_reasons"] == [
        "A2_reproduction_genuine_failed_loop_or_gate_mismatch_su15",
        "A2_not_duplicate_failed_duplicate_depth_su15_L2",
        "A2_self_discovery_provenance_failed_outer_loop_re",
        "A2_live_path_reachable_failed",
    ]


def test_scenario_capstone_4929_efficiency_checks_oracle_distinct_and_honest_ab() -> None:
    """SCENARIO-CAPSTONE-4929: D is trusted only with oracle-distinct honest A/B evidence."""

    clean = exp4929.audit_efficiency(_efficiency_artifact())

    assert clean["checks"] == {"oracle_distinct": True, "honest_ab": True}
    assert clean["failure_reasons"] == []

    bad = exp4929.audit_efficiency(
        _efficiency_artifact(
            verifier_is_oracle=True,
            baseline_kind="local_probe",
            baseline_hash_matches_submitted=False,
            zero_reached_level_regression=False,
            reached_level_regressions=[{"game": "sp80"}],
            parity_test_green=False,
            leak_check={"passed": False, "leak_detected": True},
            same_state_target_shortcut=True,
            null_reported_honestly=False,
        )
    )

    assert bad["checks"] == {"oracle_distinct": False, "honest_ab": False}
    assert bad["failure_reasons"] == [
        "D_oracle_distinct_failed_verifier_is_oracle_not_false",
        "D_honest_ab_failed_baseline_not_submitted_exact_hash",
        "D_honest_ab_failed_reached_level_regression",
        "D_honest_ab_failed_parity_test_red",
        "D_honest_ab_failed_leak_or_same_state_shortcut",
        "D_honest_ab_failed_null_not_reported_honestly",
    ]


def test_scenario_capstone_4929_run_records_missing_d_but_audits_present_banks(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4929-BLOCKED-PRECONDITION: missing D is recorded, not fabricated."""

    _write_inputs(tmp_path, include_d=False)

    artifact = exp4929.run(
        root=tmp_path,
        write=True,
        lint_runner=lambda _root: {"passed": True, "command": "fixture lint"},
        now=lambda: 10.0,
    )
    written = json.loads((tmp_path / exp4929.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert artifact["honest_verdict"] == (
        "blocked_experiment_4933_matm_similarity_retrieval_efficiency_missing"
    )
    assert artifact["preconditions_checked"]["d_artifact_present"] is False
    assert artifact["checks"] == {
        "reproduction_genuine": True,
        "not_duplicate": False,
        "self_discovery_provenance": True,
        "live_path_reachable": True,
        "oracle_distinct": False,
        "honest_ab": False,
    }
    assert artifact["banks_trustworthy"] is False
    assert artifact["efficiency_trustworthy"] is False
    assert "D_missing_experiment_4933_matm_similarity_retrieval_efficiency" in artifact[
        "audit_failure_reasons"
    ]
    assert "A1_not_duplicate_failed_duplicate_depth_sp80_L2" in artifact["audit_failure_reasons"]
    assert artifact["bank_evidence"]["A1"]["checks"]["reproduction_genuine"] is True
    assert artifact["bank_evidence"]["A2"]["checks"]["reproduction_genuine"] is True
    assert artifact["efficiency_evidence"]["present"] is False
    assert exp4929.artifact_schema_errors(artifact) == []


def test_req_capstone_4929_run_can_emit_fully_trusted_fixture(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4929: trust flags are ANDs over all six checks."""

    _write_inputs(tmp_path, registry_text=_registry_text(sp80=2, su15=2))
    for relative, payload in (
        (
            "results/experiment_4925_levelup_attempt.json",
            _bank_artifact(game="sp80", exp_id=4925, reached=3),
        ),
        (
            "results/experiment_4926_levelup_attempt.json",
            _bank_artifact(game="su15", exp_id=4926, reached=3),
        ),
        ("results/arc_loop_solve_sp80.json", _loop_artifact("sp80", reached=3)),
        ("results/arc_loop_solve_su15.json", _loop_artifact("su15", reached=3)),
    ):
        (tmp_path / relative).write_text(json.dumps(payload), encoding="utf-8")

    artifact = exp4929.run(
        root=tmp_path,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 20.0,
    )

    assert artifact["honest_verdict"] == "complete_v454_banks_and_efficiency_audited_trusted"
    assert artifact["checks"] == {key: True for key in exp4929.CHECK_KEYS}
    assert artifact["banks_trustworthy"] is True
    assert artifact["efficiency_trustworthy"] is True
    assert artifact["audit_failure_reasons"] == []
    assert artifact["duration_s"] == 1.0
    assert exp4929.artifact_schema_errors(artifact) == []


def test_req_capstone_4929_schema_and_blocked_registry_paths(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4929: malformed artifacts and registry preconditions fail closed."""

    _write_inputs(tmp_path)
    (tmp_path / exp4929.REGISTRY_RELATIVE_PATH).write_text("not: [valid", encoding="utf-8")

    blocked = exp4929.run(
        root=tmp_path,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 30.0,
    )
    assert blocked["honest_verdict"] == "blocked_arc_solve_registry_unloadable"
    assert blocked["preconditions_checked"]["registry_loadable"] is False
    assert blocked["banks_trustworthy"] is False
    assert blocked["efficiency_trustworthy"] is False

    valid = exp4929.run(
        root=tmp_path,
        write=False,
        registry_loader=lambda _root: yaml.safe_load(_registry_text()),
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 40.0,
    )
    bad = dict(valid)
    bad["banks_trustworthy"] = "no"
    bad["efficiency_trustworthy"] = "no"
    bad["checks"] = {"reproduction_genuine": True}
    bad["audit_failure_reasons"] = "none"
    bad["inference_substrate"] = "bad"
    bad["field_principles"] = {}
    bad["preconditions_checked"] = []
    bad["reproducibility_checksum"] = "bad"
    bad.pop("honest_verdict")

    errors = exp4929.artifact_schema_errors(bad)

    assert "missing required field honest_verdict" in errors
    assert "banks_trustworthy must be bare bool" in errors
    assert "efficiency_trustworthy must be bare bool" in errors
    assert "checks must contain the six required bare booleans" in errors
    assert "audit_failure_reasons must be a list" in errors
    assert "inference_substrate mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    with_errors = dict(valid)
    with_errors["schema_errors"] = ["stale"]
    assert "schema_errors must be empty" in exp4929.artifact_schema_errors(with_errors)

    out = exp4929.write_artifact(valid, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == valid


def test_req_capstone_4929_defensive_branches_fail_closed(tmp_path: Path, monkeypatch) -> None:
    """REQ-CAPSTONE-4929: defensive helpers keep malformed evidence untrusted."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    try:
        exp4929._read_json(bad_json)
    except ValueError as exc:
        assert "did not contain a JSON object" in str(exc)
    else:
        raise AssertionError("non-object JSON should fail")

    (tmp_path / "ops").mkdir()
    (tmp_path / exp4929.REGISTRY_RELATIVE_PATH).write_text("[]", encoding="utf-8")
    try:
        exp4929._load_registry(tmp_path)
    except ValueError as exc:
        assert "registry did not contain a mapping" in str(exc)
    else:
        raise AssertionError("non-mapping registry should fail")

    assert exp4929._registry_game_row({"games": []}, "missing") == {}
    assert exp4929._int_value("not-int", default=7) == 7
    assert exp4929._has_outer_loop_inputs({"nested": [{"calibration_inputs": True}]}) is True
    assert exp4929._live_game_adapter_evidence("sp80", {}) is True
    assert exp4929._live_game_adapter_evidence("not_a_game", {}) is False

    lint_root = tmp_path / "lint"
    (lint_root / "scripts").mkdir(parents=True)
    (lint_root / "scripts" / "arc_orphan_solver_lint.py").write_text(
        "print('lint ok')\n",
        encoding="utf-8",
    )
    lint = exp4929.run_arc_orphan_solver_lint(lint_root)
    assert lint["passed"] is True
    assert "lint ok" in lint["stdout_tail"]

    registry = yaml.safe_load(_registry_text())
    declared_outer = exp4929.audit_bank(
        label="A1",
        artifact=_bank_artifact(
            game="sp80",
            exp_id=4925,
            provenance="live_agent_self_discovery",
            outer_loop=True,
        ),
        loop_artifact=_loop_artifact("sp80"),
        registry=registry,
        lint_result={"passed": True},
    )
    assert "A1_self_discovery_provenance_failed_declared_outer_loop_input" in declared_outer[
        "failure_reasons"
    ]

    no_explicit_null = exp4929.audit_efficiency(
        _efficiency_artifact(null_reported_honestly=None, honest_verdict="complete_retired_null")
    )
    assert no_explicit_null["checks"]["honest_ab"] is True

    monkeypatch.setattr(
        exp4929,
        "_critical_circular_moat_flags",
        lambda _artifact: [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}],
    )
    circular = exp4929.audit_efficiency(_efficiency_artifact())
    assert circular["checks"]["oracle_distinct"] is False
    assert "D_oracle_distinct_failed_circular_moat_overclaim" in circular["failure_reasons"]
    monkeypatch.undo()

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    checked, registry_none = exp4929._preconditions(empty_root, None)
    assert registry_none is None
    assert checked["registry_loadable"] is False
    assert "arc_solve_registry" in checked["absent_inputs"]

    all_ok = {
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "d_artifact_present": True,
        "registry_present": True,
        "registry_loadable": True,
        "adversarial_verify_present": True,
        "summarize_artifact_present": True,
        "arc_orphan_solver_lint_present": True,
        "spec_has_req_4929": True,
    }
    assert exp4929._blocked_verdict(all_ok) is None
    for key, expected in (
        ("adversarial_verify_present", "blocked_scripts_adversarial_verify_missing"),
        ("summarize_artifact_present", "blocked_scripts_summarize_artifact_missing"),
        ("arc_orphan_solver_lint_present", "blocked_scripts_arc_orphan_solver_lint_missing"),
        ("spec_has_req_4929", "blocked_capstone_spec_req_4929_missing"),
    ):
        assert exp4929._blocked_verdict({**all_ok, key: False}) == expected

    bad_schema = exp4929._with_checksum_and_schema({"experiment": "wrong"})
    assert bad_schema["schema_errors"]
    assert bad_schema["reproducibility_checksum"].startswith("sha256:")
    assert exp4929._load_loop_for_bank(tmp_path, {}) is None

    missing_root = tmp_path / "missing_inputs"
    missing_root.mkdir()
    _write_inputs(missing_root)
    (missing_root / exp4929.A1_RELATIVE_PATH).unlink()
    (missing_root / "results/arc_loop_solve_su15.json").unlink()
    missing = exp4929.run(
        root=missing_root,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 50.0,
    )
    assert missing["bank_evidence"]["A1"]["failure_reasons"] == [
        "A1_missing_experiment_4925_levelup_attempt"
    ]
    assert missing["bank_evidence"]["A2"]["failure_reasons"] == ["A2_missing_loop_artifact"]

    valid = exp4929.run(
        root=tmp_path,
        write=False,
        registry_loader=lambda _root: yaml.safe_load(_registry_text()),
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 60.0,
    )
    mismatch = dict(valid)
    mismatch["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    assert "reproducibility_checksum mismatch" in exp4929.artifact_schema_errors(mismatch)
    try:
        exp4929.write_artifact({**valid, "banks_trustworthy": "bad"}, root=tmp_path)
    except ValueError as exc:
        assert "banks_trustworthy must be bare bool" in str(exc)
    else:
        raise AssertionError("invalid artifact should not be written")
