"""Tests for Exp 4941 .455 bank and pivot-readiness audit.

Spec refs: REQ-CAPSTONE-4941, SCENARIO-CAPSTONE-4941,
SCENARIO-CAPSTONE-4941-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4941_bank_and_pivot_audit as exp4941


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry_text(*, lf52: int = 2, sb26: int = 2) -> str:
    return f"""schema_version: 1
updated: '2026-06-28'
reproducible_total_levels: 69
games:
- game: lf52
  reproducibility: reproduced
  levels_reproduced: {lf52}
  solver: GameAdapter _lf52 plus scripts/arc_loop_solve.py.
- game: sb26
  reproducibility: reproduced
  levels_reproduced: {sb26}
  solver: GameAdapter _sb26 plus scripts/arc_loop_solve.py.
"""


def _bank_artifact(
    *,
    game: str,
    exp_id: int,
    reached: int = 2,
    prior: int = 2,
    claimed: bool = False,
    provenance: str = "live_agent_self_discovery",
    live_path: bool = True,
    outer_loop: bool = False,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "experiment": f"experiment_{exp_id}_levelup_attempt",
        "honest_verdict": f"complete_{game}_no_new_level_residual_no_grounded_l3_delta",
        "target_game": game,
        "solve_provenance": provenance,
        "offline_reproduced": False,
        "reproduced_levels": prior,
        "new_levels_banked": 0,
        "live_path_reachable": live_path,
        "standing_loop_result_path": f"results/arc_loop_solve_{game}.json",
        "standing_loop_ran": False,
        "reproduction_gate": {},
        "registry_update": {
            "target_game": game,
            "prior_game_levels": prior,
            "new_game_levels": reached,
            "banked_levels": 0,
            "reason": "no_grounded_l3_delta",
        },
        "offline_ground_truth_bfs": outer_loop,
    }
    if claimed:
        artifact.update(
            {
                "honest_verdict": f"success_{game}_levelup_banked",
                "offline_reproduced": True,
                "reproduced_levels": reached,
                "new_levels_banked": max(1, reached - prior),
                "standing_loop_ran": True,
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
                    "banked_levels": max(1, reached - prior),
                    "reason": "banked_offline_reproduced_level",
                },
            }
        )
    return artifact


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


def _pivot_artifact(**overrides: object) -> dict[str, object]:
    artifact: dict[str, object] = {
        "experiment": "experiment_4940_distributional_energy_verifier_executable_spec",
        "honest_verdict": "success_distributional_energy_verifier_pivot_executable_spec_ready",
        "verifier_is_oracle": False,
        "moat_proven_claimed": False,
        "arxiv_ids_cited": ["2605.18871", "2504.16828", "2502.01989"],
        "citations": {
            "2605.18871": {"http_status": 200, "title": "Distributional EBM", "url": "https://arxiv.org/abs/2605.18871"},
            "2504.16828": {"http_status": 200, "title": "ThinkPRM", "url": "https://arxiv.org/abs/2504.16828"},
            "2502.01989": {"http_status": 200, "title": "VFScale", "url": "https://arxiv.org/abs/2502.01989"},
        },
        "design_spec": {
            "decomposed_energy_verifier_column": {
                "model_identity_features_allowed": False,
                "oracle_labels_allowed_in_verifier": False,
            },
            "oracle_column": "cached labels or executable domain oracle used only to score correctness",
        },
        "validation_gate": {
            "beats_self_consistency_ci95_excludes_zero_required": True,
            "oracle_distinct_required": True,
            "no_model_identity_shortcut_required": True,
            "verifier_is_oracle_required_value": False,
            "claimed_met": False,
        },
        "field_principles": {
            "verifier_is_oracle": {
                "principle": (
                    "false -- the DESIGN TARGET is oracle-distinct, not the "
                    "executable oracle that defines correctness."
                )
            }
        },
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
        "REQ-CAPSTONE-4941\nSCENARIO-CAPSTONE-4941\n",
        encoding="utf-8",
    )
    (root / exp4941.REGISTRY_RELATIVE_PATH).write_text(
        registry_text or _registry_text(),
        encoding="utf-8",
    )
    for relative, payload in (
        (exp4941.A1_RELATIVE_PATH, _bank_artifact(game="lf52", exp_id=4936)),
        (exp4941.A2_RELATIVE_PATH, _bank_artifact(game="sb26", exp_id=4937)),
        ("results/arc_loop_solve_lf52.json", _loop_artifact("lf52")),
        ("results/arc_loop_solve_sb26.json", _loop_artifact("sb26")),
    ):
        (root / relative).write_text(json.dumps(payload), encoding="utf-8")
    if include_d:
        (root / exp4941.PIVOT_RELATIVE_PATH).write_text(
            json.dumps(_pivot_artifact()),
            encoding="utf-8",
        )


def test_req_capstone_4941_spec_declares_audit_contract() -> None:
    """REQ-CAPSTONE-4941: OpenSpec names every check and required field."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4941.SPEC_REFS:
        assert ref in spec
    for check in exp4941.CHECK_KEYS:
        assert check in spec
    for field, principle in exp4941.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec
    assert exp4941.RESULT_RELATIVE_PATH in spec


def test_scenario_capstone_4941_no_bank_claims_are_vacuously_trusted(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4941: honest A1/A2 no-bank dead ends are not failures."""

    _write_inputs(tmp_path)

    artifact = exp4941.run(
        root=tmp_path,
        write=True,
        lint_runner=lambda _root: {"passed": True, "command": "fixture lint"},
        now=lambda: 10.0,
    )
    written = json.loads((tmp_path / exp4941.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert artifact["honest_verdict"] == "complete_v455_banks_and_pivot_audited_trusted"
    assert artifact["checks"] == {key: True for key in exp4941.CHECK_KEYS}
    assert artifact["banks_trustworthy"] is True
    assert artifact["pivot_readiness_trustworthy"] is True
    assert artifact["audit_failure_reasons"] == []
    assert artifact["bank_evidence"]["A1"]["bank_claimed"] is False
    assert artifact["bank_evidence"]["A2"]["bank_claimed"] is False
    assert artifact["duration_s"] == 1.0
    assert exp4941.artifact_schema_errors(artifact) == []


def test_req_capstone_4941_claimed_bank_checks_catch_adversarial_patterns() -> None:
    """REQ-CAPSTONE-4941: claimed banks must be genuine, new, self-discovered, and live-path."""

    registry = yaml.safe_load(_registry_text())
    clean = exp4941.audit_bank(
        label="A1",
        artifact=_bank_artifact(game="lf52", exp_id=4936, reached=3, prior=2, claimed=True),
        loop_artifact=_loop_artifact("lf52", reached=3),
        registry=registry,
        lint_result={"passed": True},
    )
    assert clean["bank_claimed"] is True
    assert clean["checks"] == {key: True for key in exp4941.BANK_CHECK_KEYS}
    assert clean["failure_reasons"] == []

    bad = exp4941.audit_bank(
        label="A2",
        artifact=_bank_artifact(
            game="sb26",
            exp_id=4937,
            reached=2,
            prior=2,
            claimed=True,
            provenance="outer_loop_re",
            live_path=False,
            outer_loop=True,
        ),
        loop_artifact=_loop_artifact("sb26", reached=1, reproduced=False),
        registry=registry,
        lint_result={"passed": False, "returncode": 1},
    )

    assert bad["checks"] == {
        "reproduction_genuine": False,
        "not_duplicate": False,
        "self_discovery_provenance": False,
        "live_path_reachable": False,
    }
    assert bad["failure_reasons"] == [
        "A2_reproduction_genuine_failed_loop_or_gate_mismatch_sb26",
        "A2_not_duplicate_failed_duplicate_depth_sb26_L2",
        "A2_self_discovery_provenance_failed_outer_loop_re",
        "A2_live_path_reachable_failed",
    ]


def test_scenario_capstone_4941_pivot_checks_oracle_distinct_and_honest_readiness(monkeypatch) -> None:
    """SCENARIO-CAPSTONE-4941: D readiness is gated on oracle-distinct design and honest scope."""

    clean = exp4941.audit_pivot_readiness(_pivot_artifact())

    assert clean["checks"] == {"oracle_distinct_design": True, "honest_readiness": True}
    assert clean["failure_reasons"] == []

    bad = exp4941.audit_pivot_readiness(
        _pivot_artifact(
            verifier_is_oracle=True,
            moat_proven_claimed=True,
            arxiv_ids_cited=["2605.18871"],
            citations={"2605.18871": {"http_status": 404, "title": "", "url": "bad"}},
            design_spec={
                "decomposed_energy_verifier_column": {
                    "model_identity_features_allowed": True,
                    "oracle_labels_allowed_in_verifier": True,
                },
                "matm_similarity_retrieval": {"proposed": True},
            },
            validation_gate={
                "beats_self_consistency_ci95_excludes_zero_required": False,
                "oracle_distinct_required": False,
                "no_model_identity_shortcut_required": False,
                "verifier_is_oracle_required_value": True,
                "claimed_met": True,
            },
            field_principles={"verifier_is_oracle": {"principle": "oracle"}},
        )
    )

    assert bad["checks"] == {"oracle_distinct_design": False, "honest_readiness": False}
    assert bad["failure_reasons"] == [
        "D_oracle_distinct_design_failed_verifier_is_oracle_not_false",
        "D_oracle_distinct_design_failed_design_target_not_declared",
        "D_oracle_distinct_design_failed_model_identity_shortcut_allowed",
        "D_oracle_distinct_design_failed_oracle_labels_allowed",
        "D_honest_readiness_failed_arxiv_ids_not_exact",
        "D_honest_readiness_failed_citation_metadata_not_real",
        "D_honest_readiness_failed_validation_gate_not_precise",
        "D_honest_readiness_failed_moat_proven_claimed",
        "D_honest_readiness_failed_matm_reproposed",
    ]

    monkeypatch.setattr(
        exp4941,
        "_critical_circular_moat_flags",
        lambda _artifact: [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}],
    )
    circular = exp4941.audit_pivot_readiness(_pivot_artifact())
    assert circular["checks"]["oracle_distinct_design"] is False
    assert "D_oracle_distinct_design_failed_circular_moat_overclaim" in circular[
        "failure_reasons"
    ]


def test_scenario_capstone_4941_missing_d_blocks_but_audits_present_banks(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4941-BLOCKED-PRECONDITION: missing D is recorded."""

    _write_inputs(tmp_path, include_d=False)

    artifact = exp4941.run(
        root=tmp_path,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 20.0,
    )

    assert artifact["honest_verdict"] == (
        "blocked_experiment_4940_distributional_energy_verifier_executable_spec_missing"
    )
    assert artifact["preconditions_checked"]["d_artifact_present"] is False
    assert artifact["checks"] == {
        "reproduction_genuine": True,
        "not_duplicate": True,
        "self_discovery_provenance": True,
        "live_path_reachable": True,
        "oracle_distinct_design": False,
        "honest_readiness": False,
    }
    assert artifact["banks_trustworthy"] is True
    assert artifact["pivot_readiness_trustworthy"] is False
    assert "D_missing_experiment_4940_distributional_energy_verifier_executable_spec" in artifact[
        "audit_failure_reasons"
    ]
    assert artifact["pivot_readiness_evidence"]["present"] is False
    assert exp4941.artifact_schema_errors(artifact) == []


def test_req_capstone_4941_schema_and_blocked_registry_paths(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4941: malformed registries and malformed artifacts fail closed."""

    _write_inputs(tmp_path)
    (tmp_path / exp4941.REGISTRY_RELATIVE_PATH).write_text("not: [valid", encoding="utf-8")

    blocked = exp4941.run(
        root=tmp_path,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 30.0,
    )
    assert blocked["honest_verdict"] == "blocked_arc_solve_registry_unloadable"
    assert blocked["preconditions_checked"]["registry_loadable"] is False
    assert blocked["banks_trustworthy"] is False
    assert blocked["pivot_readiness_trustworthy"] is True

    valid = exp4941.run(
        root=tmp_path,
        write=False,
        registry_loader=lambda _root: yaml.safe_load(_registry_text()),
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 40.0,
    )
    bad = dict(valid)
    bad["banks_trustworthy"] = "yes"
    bad["pivot_readiness_trustworthy"] = "yes"
    bad["checks"] = {"reproduction_genuine": True}
    bad["audit_failure_reasons"] = "none"
    bad["inference_substrate"] = "bad"
    bad["field_principles"] = {}
    bad["preconditions_checked"] = []
    bad["reproducibility_checksum"] = "bad"
    bad.pop("honest_verdict")

    errors = exp4941.artifact_schema_errors(bad)

    assert "missing required field honest_verdict" in errors
    assert "banks_trustworthy must be bare bool" in errors
    assert "pivot_readiness_trustworthy must be bare bool" in errors
    assert "checks must contain the six required bare booleans" in errors
    assert "audit_failure_reasons must be a list" in errors
    assert "inference_substrate mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    with_errors = dict(valid)
    with_errors["schema_errors"] = ["stale"]
    assert "schema_errors must be empty" in exp4941.artifact_schema_errors(with_errors)

    mismatch = dict(valid)
    mismatch["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    assert "reproducibility_checksum mismatch" in exp4941.artifact_schema_errors(mismatch)

    out = exp4941.write_artifact(valid, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == valid
    try:
        exp4941.write_artifact({**valid, "banks_trustworthy": "bad"}, root=tmp_path)
    except ValueError as exc:
        assert "banks_trustworthy must be bare bool" in str(exc)
    else:
        raise AssertionError("invalid artifact should not be written")


def test_req_capstone_4941_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4941: defensive helper paths keep malformed evidence untrusted."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    try:
        exp4941._read_json(bad_json)
    except ValueError as exc:
        assert "did not contain a JSON object" in str(exc)
    else:
        raise AssertionError("non-object JSON should fail")

    (tmp_path / "ops").mkdir()
    (tmp_path / exp4941.REGISTRY_RELATIVE_PATH).write_text("[]", encoding="utf-8")
    try:
        exp4941._load_registry(tmp_path)
    except ValueError as exc:
        assert "registry did not contain a mapping" in str(exc)
    else:
        raise AssertionError("non-mapping registry should fail")

    assert exp4941._registry_game_row({"games": []}, "missing") == {}
    assert exp4941._int_value("bad", default=7) == 7
    assert exp4941._claims_bank({"honest_verdict": "success_demo_levelup_banked"}) is True
    assert exp4941._claims_bank({"offline_reproduced": True, "new_levels_banked": 0}) is True
    assert exp4941._registry_prior_level({}, yaml.safe_load(_registry_text()), "lf52") == 2
    assert exp4941._has_outer_loop_inputs({"nested": [{"calibration_inputs": True}]}) is True
    assert exp4941._live_game_adapter_evidence("lf52", {"solver": "GameAdapter fixture"}) is True
    assert exp4941._live_game_adapter_evidence("not_a_game", {}) is False
    assert exp4941._contains_matm_reproposal({"note": "retired MATM null"}) is False
    assert exp4941._contains_matm_reproposal({"matm": {"proposed": True}}) is True
    assert exp4941._contains_matm_reproposal({"matm_similarity_retrieval": {"proposed": True}}) is True
    assert exp4941._contains_matm_reproposal("MATM similarity-keyed retrieval proposed again") is True
    assert exp4941._citation_metadata_real(
        _pivot_artifact(
            citations={
                "2605.18871": {"http_status": 200, "title": "x", "url": "bad"},
                "2504.16828": {
                    "http_status": 200,
                    "title": "x",
                    "url": "https://arxiv.org/abs/2504.16828",
                },
                "2502.01989": {
                    "http_status": 200,
                    "title": "x",
                    "url": "https://arxiv.org/abs/2502.01989",
                },
            }
        )
    ) is False
    assert exp4941._citation_metadata_real(
        _pivot_artifact(
            citations={
                "2605.18871": {
                    "http_status": 200,
                    "title": "",
                    "url": "https://arxiv.org/abs/2605.18871",
                },
                "2504.16828": {
                    "http_status": 200,
                    "title": "x",
                    "url": "https://arxiv.org/abs/2504.16828",
                },
                "2502.01989": {
                    "http_status": 200,
                    "title": "x",
                    "url": "https://arxiv.org/abs/2502.01989",
                },
            }
        )
    ) is False

    lint_root = tmp_path / "lint"
    (lint_root / "scripts").mkdir(parents=True)
    (lint_root / "scripts" / "arc_orphan_solver_lint.py").write_text(
        "print('lint ok')\n",
        encoding="utf-8",
    )
    lint = exp4941.run_arc_orphan_solver_lint(lint_root)
    assert lint["passed"] is True
    assert "lint ok" in lint["stdout_tail"]

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    checked, registry_none = exp4941._preconditions(empty_root, None)
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
        "spec_has_req_4941": True,
    }
    assert exp4941._blocked_verdict(all_ok) is None
    for key, expected in (
        ("a1_artifact_present", "blocked_experiment_4936_levelup_attempt_missing"),
        ("a2_artifact_present", "blocked_experiment_4937_levelup_attempt_missing"),
        ("adversarial_verify_present", "blocked_scripts_adversarial_verify_missing"),
        ("summarize_artifact_present", "blocked_scripts_summarize_artifact_missing"),
        ("arc_orphan_solver_lint_present", "blocked_scripts_arc_orphan_solver_lint_missing"),
        ("spec_has_req_4941", "blocked_capstone_spec_req_4941_missing"),
    ):
        assert exp4941._blocked_verdict({**all_ok, key: False}) == expected

    missing_bank = exp4941._missing_bank_evidence("A1", "A1_missing")
    assert missing_bank["failure_reasons"] == ["A1_missing"]

    missing_pivot = exp4941._missing_pivot_evidence("D_missing")
    assert missing_pivot["checks"] == {"oracle_distinct_design": False, "honest_readiness": False}

    declared_outer = exp4941.audit_bank(
        label="A1",
        artifact=_bank_artifact(
            game="lf52",
            exp_id=4936,
            reached=3,
            prior=2,
            claimed=True,
            outer_loop=True,
        ),
        loop_artifact=_loop_artifact("lf52", reached=3),
        registry=yaml.safe_load(_registry_text()),
        lint_result={"passed": True},
    )
    assert "A1_self_discovery_provenance_failed_declared_outer_loop_input" in declared_outer[
        "failure_reasons"
    ]

    no_loop = exp4941.audit_bank(
        label="A1",
        artifact=_bank_artifact(game="lf52", exp_id=4936, reached=3, prior=2, claimed=True),
        loop_artifact=None,
        registry=yaml.safe_load(_registry_text()),
        lint_result={"passed": True},
    )
    assert "A1_missing_loop_artifact" in no_loop["failure_reasons"]
    assert exp4941._aggregate_checks(
        {"A1": declared_outer},
        {"checks": {"oracle_distinct_design": True, "honest_readiness": True}},
    )["self_discovery_provenance"] is False
    assert exp4941._load_loop_for_bank(tmp_path, {}) is None

    bad_schema = exp4941._with_checksum_and_schema({"experiment": "wrong"})
    assert bad_schema["schema_errors"]
    assert bad_schema["reproducibility_checksum"].startswith("sha256:")

    missing_root = tmp_path / "missing_inputs"
    missing_root.mkdir()
    _write_inputs(missing_root)
    (missing_root / exp4941.A1_RELATIVE_PATH).unlink()
    missing = exp4941.run(
        root=missing_root,
        write=False,
        lint_runner=lambda _root: {"passed": True},
        now=lambda: 50.0,
    )
    assert missing["bank_evidence"]["A1"]["failure_reasons"] == [
        "A1_missing_experiment_4936_levelup_attempt"
    ]
