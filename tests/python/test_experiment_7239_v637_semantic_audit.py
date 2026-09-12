"""Focused tests for the V637 semantic audit.

Spec refs: REQ-VERIFY-7239 and SCENARIO-VERIFY-7239-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7239_v637_semantic_audit as exp


REPO = Path(__file__).resolve().parents[2]


def _paired_rows(*, pointer_wins: bool = True) -> list[dict[str, object]]:
    """Build 64 paired bases with invalid pointer outputs kept as errors."""

    rows: list[dict[str, object]] = []
    for index in range(64):
        unit_id = f"base-{index:02d}"
        pointer_abstains = index >= 56
        for arm in exp.ARMS:
            if arm == "mention_pointer":
                correct = not pointer_abstains if pointer_wins else index % 2 == 0
                abstention = pointer_abstains
                fidelity_limit = 56 if pointer_wins else 32
                source_fidelity: bool | None = index < fidelity_limit
                claim_fidelity: bool | None = index < fidelity_limit
                false_accept = False
            elif arm == "explicit_schema_offset_control":
                correct = False if pointer_wins else index % 2 == 0
                abstention = False
                source_fidelity = False
                claim_fidelity = False
                false_accept = index % 4 == 0
            else:
                correct = False if pointer_wins else index % 2 == 0
                abstention = False
                source_fidelity = None
                claim_fidelity = None
                false_accept = index % 8 == 0
            rows.append(
                {
                    "unit_id": unit_id,
                    "arm": arm,
                    "condition": "supported" if index % 4 == 0 else "reversed",
                    "expected_decision": "supported" if index % 4 == 0 else "contradicted",
                    "predicted_decision": "unknown" if abstention else "supported",
                    "representation_valid": not abstention,
                    "mention_resolution": not abstention if arm == "mention_pointer" else None,
                    "source_fidelity": source_fidelity,
                    "claim_fidelity": claim_fidelity,
                    "decision_correct": correct,
                    "fully_correct": correct and not abstention,
                    "abstention": abstention,
                    "false_accept": false_accept,
                    "missing_output_penalty": False,
                    "metric": int(correct and not abstention),
                    "error": None if correct and not abstention else "incorrect_or_unusable",
                    "final_response_provenance": {},
                }
            )
    return rows


def test_req_verify_7239_contract_and_first_failure() -> None:
    """REQ-VERIFY-7239 authenticates bytes before rejecting quarantine."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-VERIFY-7239" in spec
    assert exp.MODEL_SPECS == []
    assert exp.BOOTSTRAP_DRAWS == 10_000
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.unwrap_principle({"principle": "why", "value": 3}) == 3
    assert exp.unwrap_principle({"principle": "why"}) == {"principle": "why"}
    assert exp.unwrap_principle(True) is True

    bundle = exp.load_upstream_bundle(REPO)
    checks = exp.upstream_gate_rows(bundle)
    assert checks[0]["check"] == "exact_upstream_bytes"
    assert checks[0]["passed"] is True
    failed = [row for row in checks if row["passed"] is False]
    assert failed[0]["check"] == "structured_quarantine"
    assert failed[0]["artifact_field"] == "flagged_adversarial|quarantined|fabricated"
    assert failed[0]["observed_value"] is True
    assert exp.gate_summary(failed[0]) == {
        "passed": False,
        "failed_check": "structured_quarantine",
        "upstream": "experiment_7237",
        "artifact_field": "flagged_adversarial|quarantined|fabricated",
        "expected_value": False,
        "observed_value": True,
    }
    assert exp.gate_summary(None)["passed"] is True

    changed = deepcopy(bundle)
    changed["canary"] = {**changed["canary"], "flagged_adversarial": False}
    changed["canary_bytes"] = exp.canonical_json(changed["canary"]).encode()
    changed_checks = exp.upstream_gate_rows(changed)
    assert changed_checks[0]["passed"] is False
    assert (
        next(row for row in changed_checks if row["check"] == "structured_quarantine")["passed"]
        is True
    )


def test_scenario_verify_7239_block_preserves_calibration(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7239-BLOCK writes one valid terminal explanation."""

    artifact = exp.run_experiment(REPO, exp.RUN_DATE, output_root=tmp_path)
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_exp7239_structured_quarantine"
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["current_invocation_counts"] == {
        "model_loads": 0,
        "generations": 0,
        "model_invocations": 0,
    }
    assert artifact["semantic_audit_complete_score"] == 0
    assert artifact["semantic_value_score"] == 0
    assert artifact["rows"] == artifact["paired_comparison_rows"] == []
    calibration = artifact["calibration_observations"]
    assert calibration["eligible_for_promotion"] is False
    assert calibration["quarantine_observed"] is True
    assert calibration["arms"]["mention_pointer"] == {
        "units": 8,
        "source_fidelity_count": 8,
        "claim_fidelity_count": 8,
        "covered_count": 6,
        "decision_correct_count": 8,
        "false_accept_count": 0,
        "fully_correct_count": 8,
    }
    assert calibration["arms"]["original_offset"]["covered_count"] == 0
    assert artifact["positive_control_results"]["status"] == "not_run_external_block"
    assert all(row["evaluated"] is False for row in artifact["acceptance_gate_results"])
    assert exp.validate_artifact(artifact, REPO) == []
    result = tmp_path / exp.RESULT_PATH
    checkpoint = tmp_path / exp.CHECKPOINT_PATH
    assert result.is_file() and checkpoint.is_file()
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert exp.replay_terminal_artifact(REPO, output_root=tmp_path) == artifact


def test_scenario_verify_7239_artifact_attaches_validation_receipts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7239-ARTIFACT attaches only exact observed receipts."""

    artifact = exp.run_experiment(REPO, exp.RUN_DATE, output_root=tmp_path)
    receipts = [
        {
            "command": "pytest focused",
            "exit_code": 0,
            "classification": "passed",
            "summary": "10 passed",
        }
    ]
    attached = exp.attach_validation_receipts(artifact, receipts)
    assert attached["validation_command_rows"] == receipts
    for key, value in artifact.items():
        if key not in {"validation_command_rows", "reproducibility_checksum"}:
            assert attached[key] == value
    assert exp.validate_artifact(attached, REPO) == []

    with pytest.raises(ValueError, match="validation_receipt_source_artifact"):
        exp.attach_validation_receipts({**artifact, "status": "running"}, receipts)
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp.attach_validation_receipts(artifact, [{"command": "missing fields"}])


def test_scenario_verify_7239_replay_metrics_keep_invalid_rows() -> None:
    """SCENARIO-VERIFY-7239-REPLAY keeps all errors and selective risk separate."""

    rows = _paired_rows()
    assert exp.paired_row_errors(rows) == []
    metrics = exp.summarize_arms(rows)
    pointer = metrics["mention_pointer"]
    assert pointer["denominator"] == 64
    assert pointer["source_fidelity"] == 0.875
    assert pointer["coverage"] == 0.875
    assert pointer["decision_error"] == 0.125
    assert pointer["selective_risk"] == 0.0
    assert pointer["selective_denominator"] == 56
    assert metrics["explicit_schema_offset_control"]["decision_error"] == 1.0
    costs = exp.summarize_costs(
        [
            {
                "arm": arm,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "elapsed_s": 0.25,
                "timeout": arm == "direct_judge",
                "transport_complete": arm != "direct_judge",
            }
            for arm in exp.ARMS
        ]
    )
    assert costs["mention_pointer"]["prompt_tokens"] == 10
    assert costs["direct_judge"]["timeout_count"] == 1

    malformed = deepcopy(rows)
    malformed.pop()
    assert "row_count" in exp.paired_row_errors(malformed)
    duplicate = deepcopy(rows)
    duplicate[-1] = deepcopy(duplicate[0])
    assert "duplicate_pair" in exp.paired_row_errors(duplicate)
    unknown_arm = deepcopy(rows)
    unknown_arm[0]["arm"] = "unknown"
    assert "arm_roster" in exp.paired_row_errors(unknown_arm)


def test_scenario_verify_7239_bootstrap_and_value_gate() -> None:
    """SCENARIO-VERIFY-7239-BOOTSTRAP uses bases and all frozen criteria."""

    rows = _paired_rows()
    intervals = exp.paired_bootstrap(rows, exp.BOOTSTRAP_SEED, exp.BOOTSTRAP_DRAWS)
    assert intervals == exp.paired_bootstrap(rows, exp.BOOTSTRAP_SEED, exp.BOOTSTRAP_DRAWS)
    assert len(intervals) == 4
    assert {row["metric"] for row in intervals} == {
        "decision_error_difference",
        "false_accept_difference",
    }
    error_rows = [row for row in intervals if row["metric"] == "decision_error_difference"]
    assert all(row["ci95"][1] < 0 for row in error_rows)
    false_accept_rows = [row for row in intervals if row["metric"] == "false_accept_difference"]
    assert all(row["ci95"][1] <= 0 for row in false_accept_rows)

    controls = {
        "status": "passed",
        "non_degenerate": True,
        "authority_boundary_passed": True,
    }
    criteria = exp.acceptance_results(exp.summarize_arms(rows), intervals, controls)
    assert all(row["passed"] is True for row in criteria)
    assert exp.classify_value(criteria) == {
        "semantic_value_score": 1,
        "verdict_class": "circular_positive",
        "honest_verdict": "complete_circular_positive_semantic_value_fixed_criteria_passed",
        "failed_layer": None,
    }

    no_headroom = {**controls, "non_degenerate": False, "status": "no_headroom"}
    failed = exp.acceptance_results(exp.summarize_arms(rows), intervals, no_headroom)
    outcome = exp.classify_value(failed)
    assert outcome["semantic_value_score"] == 0
    assert outcome["verdict_class"] == "null"
    assert outcome["failed_layer"] == "headroom"
    assert "uninformative_no_headroom" in outcome["honest_verdict"]

    weak = exp.summarize_arms(_paired_rows(pointer_wins=False))
    weak_criteria = exp.acceptance_results(weak, intervals, controls)
    assert exp.classify_value(weak_criteria)["failed_layer"] == "extraction"
    with pytest.raises(ValueError, match="paired_rows"):
        exp.paired_bootstrap(rows[:-1], exp.BOOTSTRAP_SEED, 5)
    with pytest.raises(ValueError, match="interval:pointer_vs_"):
        exp.acceptance_results(exp.summarize_arms(rows), intervals[:-1], controls)


def test_scenario_verify_7239_controls_and_authority_boundary() -> None:
    """SCENARIO-VERIFY-7239-CONTROLS preserves every required intervention."""

    bundle = exp.load_upstream_bundle(REPO)
    controls = exp.recompute_controls(bundle)
    assert controls["status"] == "passed"
    assert controls["authority_boundary_passed"] is True
    assert controls["non_degenerate"] is True
    by_name = {row["control"]: row for row in controls["interventions"]}
    assert set(by_name) == {
        "mention_id_permutation",
        "surface_renaming",
        "relation_reversal",
        "joint_support_deletion",
        "authority_file_access_denial",
    }
    assert all(row["passed"] is True for row in by_name.values())
    assert controls["gold_relation_upper_bound"] == 1.0
    assert controls["control_decision_accuracy"] < 1.0
    boundary = exp.authority_boundary(bundle)
    assert boundary["gold_source_graph"]["model_visible"] is False
    assert boundary["extracted_graph"]["source"] == "capture_source_and_claim_outputs"
    assert boundary["independent_labels"]["joined_after_prediction"] is True


def test_req_verify_7239_parse_and_reducer_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7239 rejects malformed sources and reducer drift."""

    artifact_list = tmp_path / "artifact-list.json"
    artifact_list.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="capture_artifact_mapping"):
        exp.load_upstream_bundle(REPO, {"capture": artifact_list})
    manifest_list = tmp_path / "manifest-list.json"
    manifest_list.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="public_mapping"):
        exp.load_upstream_bundle(REPO, {"public": manifest_list})

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    checks, bundle = exp.collect_preconditions(
        REPO, exp.RUN_DATE, tmp_path, path_overrides={"capture": bad_json}
    )
    assert bundle == {}
    assert checks[-1]["check"] == "upstream_parse"

    original_access = exp.os.access
    blocked_root = tmp_path / "blocked-output"

    def selective_access(path: object, mode: int) -> bool:
        if str(path).startswith(str(blocked_root)):
            return False
        return original_access(path, mode)

    monkeypatch.setattr(exp.os, "access", selective_access)
    checks, bundle = exp.collect_preconditions(REPO, exp.RUN_DATE, blocked_root)
    assert bundle == {}
    assert checks[-1]["check"] == "output_destination"
    monkeypatch.setattr(exp.os, "access", original_access)

    paired = _paired_rows()
    costs = [{"arm": arm} for arm in exp.ARMS]
    reducer_bundle = {
        "capture": {
            "schedule": ["schedule"],
            "raw_rows": ["raw"],
            "paired_unit_rows": paired,
            "decoding_cost_rows": costs,
        },
        "paths": {"public": REPO / exp.PUBLIC_PATH, "authority": REPO / exp.AUTHORITY_PATH},
    }
    monkeypatch.setattr(exp.capture_module, "load_held_out_manifests", lambda *_args: ([], []))
    monkeypatch.setattr(exp.capture_module, "schedule_errors", lambda *_args: [])
    monkeypatch.setattr(exp.capture_module, "replay_completion_rows", lambda *_args: ["replayed"])
    monkeypatch.setattr(exp.capture_module, "score_semantics", lambda *_args: paired)
    monkeypatch.setattr(exp.capture_module, "decoding_cost_rows", lambda *_args: costs)
    assert exp.cold_replay_capture(reducer_bundle) == (paired, costs)
    monkeypatch.setattr(exp.capture_module, "schedule_errors", lambda *_args: ["bad"])
    with pytest.raises(ValueError, match="capture_schedule"):
        exp.cold_replay_capture(reducer_bundle)
    monkeypatch.setattr(exp.capture_module, "schedule_errors", lambda *_args: [])
    monkeypatch.setattr(exp.capture_module, "score_semantics", lambda *_args: [])
    with pytest.raises(ValueError, match="capture_semantic_rows"):
        exp.cold_replay_capture(reducer_bundle)
    monkeypatch.setattr(exp.capture_module, "score_semantics", lambda *_args: paired)
    monkeypatch.setattr(exp.capture_module, "decoding_cost_rows", lambda *_args: [])
    with pytest.raises(ValueError, match="capture_cost_rows"):
        exp.cold_replay_capture(reducer_bundle)


def test_req_verify_7239_complete_path_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7239 covers the authentic complete branch without model work."""

    bundle = exp.load_upstream_bundle(REPO)
    rows = _paired_rows()
    costs = [
        {
            "arm": arm,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "elapsed_s": 0.01,
            "timeout": False,
            "transport_complete": True,
        }
        for arm in exp.ARMS
    ]
    controls = {
        "status": "passed",
        "non_degenerate": True,
        "authority_boundary_passed": True,
        "headroom_diagnosis": "headroom_present",
    }
    monkeypatch.setattr(exp, "collect_preconditions", lambda *_args, **_kwargs: ([], bundle))
    monkeypatch.setattr(exp, "cold_replay_capture", lambda _bundle: (rows, costs))
    monkeypatch.setattr(exp, "recompute_controls", lambda _bundle: controls)
    artifact = exp.run_experiment(REPO, exp.RUN_DATE, output_root=tmp_path)
    assert artifact["status"] == "complete"
    assert artifact["semantic_audit_complete_score"] == 1
    assert artifact["semantic_value_score"] == 1
    assert artifact["sample_size_budget"]["completed_independent_units"] == 64
    assert exp.validate_artifact(artifact, REPO) == []

    invalid = deepcopy(artifact)
    invalid.update(
        {
            "arm_metrics": {},
            "paired_interval_rows": [],
            "acceptance_gate_results": [],
            "semantic_audit_complete_score": 0,
            "rows": [],
            "inference_substrate": "blocked_no_run",
            "verdict_class": "positive",
        }
    )
    invalid["reproducibility_checksum"] = exp.artifact_checksum(invalid)
    errors = exp.validate_artifact(invalid, REPO)
    assert {
        "arm_metrics",
        "paired_interval_rows",
        "acceptance_gate_results",
        "terminal_classification",
        "complete_terminal_state",
    } <= set(errors)
    no_controls = deepcopy(artifact)
    no_controls["positive_control_results"] = []
    no_controls["reproducibility_checksum"] = exp.artifact_checksum(no_controls)
    assert "positive_control_results" in exp.validate_artifact(no_controls, REPO)
    bad_rows = deepcopy(artifact)
    bad_rows["paired_comparison_rows"] = {}
    bad_rows["reproducibility_checksum"] = exp.artifact_checksum(bad_rows)
    assert "paired_comparison_rows" in exp.validate_artifact(bad_rows, REPO)

    original_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp7239 artifact"):
        exp.run_experiment(REPO, exp.RUN_DATE, output_root=tmp_path)
    monkeypatch.setattr(exp, "validate_artifact", original_validate)


def test_req_verify_7239_replay_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7239 makes independent replay failures terminal."""

    blocked = exp.run_experiment(REPO, exp.RUN_DATE, output_root=tmp_path)
    original_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="terminal_artifact_validation"):
        exp.replay_terminal_artifact(REPO, output_root=tmp_path)
    monkeypatch.setattr(exp, "validate_artifact", original_validate)
    monkeypatch.setattr(
        exp,
        "upstream_gate_rows",
        lambda _bundle: [
            exp.gate_row(
                "different",
                True,
                False,
                False,
                upstream="test",
                artifact_field="field",
            )
        ],
    )
    with pytest.raises(ValueError, match="blocked_upstream_replay"):
        exp.replay_terminal_artifact(REPO, output_root=tmp_path)

    complete = deepcopy(blocked)
    complete["status"] = "complete"
    complete["reproducibility_checksum"] = exp.artifact_checksum(complete)
    (tmp_path / exp.RESULT_PATH).write_text(json.dumps(complete), encoding="utf-8")
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(exp, "upstream_gate_rows", lambda _bundle: [])
    monkeypatch.setattr(exp, "cold_replay_capture", lambda _bundle: ([{"row": 1}], []))
    with pytest.raises(ValueError, match="paired_replay"):
        exp.replay_terminal_artifact(REPO, output_root=tmp_path)
    complete["paired_comparison_rows"] = [{"row": 1}]
    complete["cost_metrics"] = {"different": True}
    (tmp_path / exp.RESULT_PATH).write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ValueError, match="cost_replay"):
        exp.replay_terminal_artifact(REPO, output_root=tmp_path)


def test_scenario_verify_7239_artifact_rejects_drift_and_missing_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-VERIFY-7239-ARTIFACT rejects terminal and precondition drift."""

    artifact = exp.run_experiment(REPO, exp.RUN_DATE, output_root=tmp_path)

    def changed(key: str, value: object) -> list[str]:
        candidate = deepcopy(artifact)
        candidate[key] = value
        candidate["reproducibility_checksum"] = exp.artifact_checksum(candidate)
        return exp.validate_artifact(candidate, REPO)

    assert exp.validate_artifact([]) == ["artifact_mapping"]
    assert exp.validate_artifact({})[0].startswith("missing_required_field:")
    assert "field_principles" in changed("field_principles", {})
    assert "run_date" in changed("run_date", "20260911")
    assert "execution_identity" in changed("execution_venue", "gpu")
    assert "duration_s" in changed("duration_s", -1)
    assert "model_contract" in changed("model_invoked", True)
    assert "verifier_is_oracle" in changed("verifier_is_oracle", False)
    assert "random_seed" in changed("random_seed", 0)
    assert "blocked_terminal_state" in changed("semantic_value_score", 1)
    assert "gate_check_summary" in changed("gate_check_summary", {})
    assert "calibration_observations" in changed("calibration_observations", {})
    assert "positive_control_results" in changed("positive_control_results", {})
    evaluated = deepcopy(artifact["acceptance_gate_results"])
    evaluated[0]["evaluated"] = True
    assert "acceptance_gate_results" in changed("acceptance_gate_results", evaluated)
    assert "source_artifact_hashes" in changed("source_artifact_hashes", {})
    assert "reproducibility_checksum" in exp.validate_artifact(
        {**artifact, "reproducibility_checksum": "bad"}, REPO
    )
    absent_root = tmp_path / "absent-repository"
    absent = deepcopy(artifact)
    absent["source_artifact_hashes"] = exp._source_hashes(exp._resolved_paths(absent_root))
    absent["reproducibility_checksum"] = exp.artifact_checksum(absent)
    assert "calibration_observations" in exp.validate_artifact(absent, absent_root)
    running = changed("status", "running")
    assert "status" in running

    missing = tmp_path / "missing.json"
    checks, _bundle = exp.collect_preconditions(
        REPO,
        exp.RUN_DATE,
        tmp_path,
        path_overrides={"capture": missing},
    )
    failed = next(row for row in checks if row["passed"] is False)
    assert failed["check"] == "required_source"
    assert failed["upstream"] == "capture"
    wrong_date, _bundle = exp.collect_preconditions(REPO, "20260911", tmp_path)
    assert wrong_date[0]["passed"] is False

    assert exp._date_argument(exp.RUN_DATE) == exp.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        exp._date_argument("20260911")
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "honest_verdict": "blocked_test",
            "semantic_audit_complete_score": 0,
            "semantic_value_score": 0,
        },
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: [])
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert "terminal verdict=blocked_test" in capsys.readouterr().out
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1
    assert "invalid artifact" in capsys.readouterr().out

    monkeypatch.setattr(exp, "main", lambda: 7)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(REPO / exp.WRAPPER_PATH), run_name="__main__")
    assert raised.value.code == 7
