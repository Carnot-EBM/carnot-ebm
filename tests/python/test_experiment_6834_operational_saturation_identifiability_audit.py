"""REQ-CONSTRAINT-6834 cold saturation and identifiability audit tests."""

from __future__ import annotations

import ast
import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6834_operational_saturation_identifiability_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source_payloads() -> tuple[dict, dict, bytes, bytes]:
    """REQ-CONSTRAINT-6834 uses the frozen fixture and corpus bytes."""

    fixture_bytes = (REPO_ROOT / exp.FIXTURE_PATH).read_bytes()
    corpus_bytes = (REPO_ROOT / exp.CORPUS_PATH).read_bytes()
    return json.loads(fixture_bytes), json.loads(corpus_bytes), fixture_bytes, corpus_bytes


@pytest.fixture(scope="module")
def recomputed(source_payloads: tuple[dict, dict, bytes, bytes]) -> list[dict]:
    """SCENARIO-CONSTRAINT-6834-INDEPENDENT-TRUTH reparses each source row once."""

    fixture, corpus, _, _ = source_payloads
    return exp.recompute_source_rows(fixture, corpus["per_unit_rows"])


def test_req_6834_spec_precedes_implementation() -> None:
    """REQ-CONSTRAINT-6834 declares every scenario and required artifact field."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("## REQ-CONSTRAINT-6834:") :]
    for marker in (
        "SCENARIO-CONSTRAINT-6834-PRECONDITIONS",
        "SCENARIO-CONSTRAINT-6834-INDEPENDENT-TRUTH",
        "SCENARIO-CONSTRAINT-6834-MATCHED-INFERENCE",
        "SCENARIO-CONSTRAINT-6834-SATURATION",
        "SCENARIO-CONSTRAINT-6834-IDENTIFIABILITY",
        "SCENARIO-CONSTRAINT-6834-ATTACKS",
        "SCENARIO-CONSTRAINT-6834-COMPLETENESS",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_scenario_6834_parser_is_strict_and_independent() -> None:
    """SCENARIO-CONSTRAINT-6834-INDEPENDENT-TRUTH rejects every repair path."""

    good = exp.canonical_bytes({"selected_action_ids": ["action-b", "action-a"]})
    assert exp.parse_raw_output(good) == {
        "error": None,
        "parsed": True,
        "selected_action_ids": ["action-a", "action-b"],
    }
    cases = [
        (b"\xff", "invalid_utf8"),
        (b"not-json", "invalid_json"),
        (b'{"selected_action_ids": ["action-a"]}', "non_canonical_json"),
        (b"[]", "invalid_response_fields"),
        (b'{"extra":1}', "invalid_response_fields"),
        (b'{"selected_action_ids":1}', "invalid_action_list"),
        (b'{"selected_action_ids":[""]}', "invalid_action_id"),
        (b'{"selected_action_ids":["a","a"]}', "duplicate_action_id"),
    ]
    for raw, error in cases:
        assert exp.parse_raw_output(raw) == {
            "error": error,
            "parsed": False,
            "selected_action_ids": [],
        }

    tree = ast.parse((REPO_ROOT / exp.MODULE_PATH).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("experiment_6832" in name or "experiment_6833" in name for name in imports)


def test_scenario_6834_checker_reimplementation_matches_all_source_rows(
    source_payloads: tuple[dict, dict, bytes, bytes], recomputed: list[dict]
) -> None:
    """SCENARIO-CONSTRAINT-6834-INDEPENDENT-TRUTH rechecks fields and joints."""

    fixture, corpus, _, _ = source_payloads
    scenario = fixture["scenarios"][0]
    legal = exp.canonical_bytes({"selected_action_ids": scenario["legal_action_ids"]})
    checked = exp.check_joint(scenario, legal)
    assert checked["parsed"] is True
    assert checked["passed"] is True
    assert all(
        all(result["fields"].values()) and result["passed"]
        for result in checked["obligation_checks"].values()
    )
    assert exp.check_obligation(scenario, (), "missing")["passed"] is False
    assert exp.check_joint(scenario, b"not-json")["parse_error"] == "invalid_json"

    assert len(recomputed) == 900
    assert all(row["producer_score_match"] for row in recomputed)
    assert sum(row["parse_status"]["parsed"] for row in recomputed) == 65
    assert sum(row["joint_passed"] for row in recomputed) == 37
    assert (
        sum(
            result["passed"] is True
            for row in recomputed
            for result in row["obligation_results"].values()
        )
        == 43
    )
    assert corpus["exact_scores"]["joint_passed"] == 37


def test_scenario_6834_preconditions_and_blocked_stop(
    source_payloads: tuple[dict, dict, bytes, bytes],
) -> None:
    """SCENARIO-CONSTRAINT-6834-PRECONDITIONS blocks before row reduction."""

    _, corpus, fixture_bytes, corpus_bytes = source_payloads
    checks = exp.evaluate_preconditions(fixture_bytes, corpus_bytes)
    assert checks
    assert all(row["passed"] for row in checks)

    changed = deepcopy(corpus)
    changed["operational_saturation_corpus_ready"] = False
    changed_bytes = json.dumps(changed).encode()
    failed = exp.evaluate_preconditions(fixture_bytes, changed_bytes)
    assert any(not row["passed"] for row in failed)
    blocked = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        fixture_bytes=fixture_bytes,
        corpus_bytes=changed_bytes,
    )
    assert blocked["status"] == "complete_blocked_operational_saturation_identifiability_audit"
    assert blocked["honest_verdict"] == blocked["status"]
    assert blocked["verdict_class"] == "blocked"
    assert blocked["per_unit_rows"] == []
    assert blocked["gate_check_summary"]["failed_check"] is not None
    assert blocked["operational_saturation_audit_complete"] is False


def test_scenario_6834_coverage_detects_deletion_duplicates_and_model_masking(
    source_payloads: tuple[dict, dict, bytes, bytes], recomputed: list[dict]
) -> None:
    """SCENARIO-CONSTRAINT-6834-ATTACKS rejects missing, duplicate, and masked identities."""

    fixture, corpus, _, _ = source_payloads
    expected = exp.expected_row_ids(fixture, corpus["MODEL_SPECS"])
    clean = exp.coverage_report(recomputed, expected)
    assert clean["valid"] is True
    assert clean["observed_unique"] == 900

    deleted = exp.coverage_report(recomputed[1:], expected)
    assert deleted["valid"] is False
    assert deleted["missing"] == [recomputed[0]["row_id"]]

    duplicated = exp.coverage_report([*recomputed, recomputed[0]], expected)
    assert duplicated["valid"] is False
    assert duplicated["duplicates"] == [recomputed[0]["row_id"]]

    masked = exp.mask_model_labels(recomputed)
    masked_report = exp.coverage_report(masked, expected)
    assert masked_report["valid"] is False
    assert masked_report["model_identity_count"] == 1


def test_scenario_6834_matched_joins_bootstrap_and_count_order(
    recomputed: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6834-MATCHED-INFERENCE uses scenario pairs and frozen order."""

    pairs = exp.join_prompt_arms(recomputed)
    assert len(pairs) == 450
    assert all(pair["scenario_id"] in pair["pair_id"] for pair in pairs)
    assert all(pair["typed"]["arm"] == "typed" for pair in pairs)
    assert all(pair["compressed"]["arm"] == "compressed" for pair in pairs)

    estimate = exp.bootstrap_interval([0.0, 1.0, 1.0], seed=6834, replicates=50)
    assert estimate["estimate"] == pytest.approx(2 / 3)
    assert estimate["unit_count"] == 3
    assert estimate["bootstrap_replicates"] == 50
    assert estimate["ci95_low"] <= estimate["estimate"] <= estimate["ci95_high"]
    assert exp.bootstrap_interval([], seed=6834, replicates=10)["estimate"] is None

    effects = exp.build_paired_arm_effects(pairs)
    assert len(effects) == 15
    assert sorted({row["obligation_count"] for row in effects}) == [1, 2, 4, 6, 8]
    assert all(row["unit_count"] == 30 for row in effects)
    assert all(row["effect_definition"] == "typed_joint_minus_compressed_joint" for row in effects)


def test_scenario_6834_metrics_separate_transport_interaction_and_decay(
    recomputed: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6834-SATURATION keeps outcome classes distinct."""

    metrics = exp.reduce_metrics(recomputed)
    assert len(metrics["per_obligation_metrics"]) == 150
    assert len(metrics["joint_success_metrics"]) == 30
    assert len(metrics["parse_failure_metrics"]) == 30
    assert len(metrics["interaction_penalties"]) == 30
    assert len(metrics["saturation_curves"]) == 6
    assert len(metrics["paired_arm_effects"]) == 15
    assert all(
        [point["obligation_count"] for point in curve["points"]] == [1, 2, 4, 6, 8]
        for curve in metrics["saturation_curves"]
    )
    assert sum(row["joint_passed"] for row in metrics["joint_success_metrics"]) == 37
    assert sum(row["parse_failed"] for row in metrics["parse_failure_metrics"]) == 835
    assert {curve["model_id"] for curve in metrics["saturation_curves"]} == {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    }
    assert all("by_interaction_class" in row for row in metrics["joint_success_metrics"])


def test_scenario_6834_label_permutations_and_checker_mutation(
    source_payloads: tuple[dict, dict, bytes, bytes], recomputed: list[dict]
) -> None:
    """SCENARIO-CONSTRAINT-6834-ATTACKS preserves labels and detects checker drift."""

    fixture, _, _, _ = source_payloads
    scenario = fixture["scenarios"][2]
    raw = exp.canonical_bytes({"selected_action_ids": scenario["legal_action_ids"]})
    permuted_scenario, permuted_raw = exp.permute_action_labels(scenario, raw)
    assert exp.check_joint(scenario, raw)["passed"] is True
    assert exp.check_joint(permuted_scenario, permuted_raw)["passed"] is True

    reversed_signature = exp.metric_signature(list(reversed(recomputed)))
    assert reversed_signature == exp.metric_signature(recomputed)
    model_permuted, model_inverse = exp.permute_group_labels(recomputed, "model_id")
    prompt_permuted, prompt_inverse = exp.permute_group_labels(recomputed, "arm")
    assert exp.canonicalized_metric_signature(model_permuted, "model_id", model_inverse) == (
        exp.metric_signature(recomputed)
    )
    assert exp.canonicalized_metric_signature(prompt_permuted, "arm", prompt_inverse) == (
        exp.metric_signature(recomputed)
    )

    passed_row = next(row for row in recomputed if row["joint_passed"])
    source_scenario = next(
        row for row in fixture["scenarios"] if row["scenario_id"] == passed_row["scenario_id"]
    )
    changed = exp.mutate_checker_contract(source_scenario)
    raw_bytes = base64.b64decode(passed_row["raw_output_bytes_b64"], validate=True)
    assert exp.check_joint(changed, raw_bytes) != exp.check_joint(source_scenario, raw_bytes)


def test_scenario_6834_collision_construction_and_minimum_support(
    source_payloads: tuple[dict, dict, bytes, bytes], recomputed: list[dict]
) -> None:
    """SCENARIO-CONSTRAINT-6834-IDENTIFIABILITY emits finite policy collisions."""

    fixture, corpus, _, _ = source_payloads
    result, minimum, witnesses = exp.audit_identifiability(
        fixture, corpus["MODEL_SPECS"], recomputed
    )
    assert result["disposition"] == "not_separated"
    assert result["target_cell_count"] == 18_900
    assert result["observed_cell_count"] == 395
    assert result["missing_cell_count"] == 18_505
    assert result["compatible_policy_count"] == "2^18505"
    assert minimum["minimum_support_size"] == 18_900
    assert minimum["current_support_size"] == 395
    assert minimum["smallest_added_cell_count"] == 18_505
    assert len(minimum["smallest_added_cells"]) == 18_505
    assert witnesses
    witness = witnesses[0]
    assert witness["observed_signature_left"] == witness["observed_signature_right"]
    assert witness["target_value_left"] != witness["target_value_right"]
    assert witness["smallest_added_cells"] == [witness["differing_target_cell"]]
    assert exp.validate_collision_witness(witness, set(minimum["observed_cells"])) is True

    separated_rows = deepcopy(recomputed)
    for row in separated_rows:
        if not row["parse_status"]["parsed"]:
            row["parse_status"] = {"error": None, "parsed": True}
            row["obligation_results"] = {
                obligation_id: {
                    "fields": {field: False for field in exp.OBLIGATION_FIELDS},
                    "passed": False,
                }
                for obligation_id in row["obligation_results"]
            }
    separated, _, no_witnesses = exp.audit_identifiability(
        fixture, corpus["MODEL_SPECS"], separated_rows
    )
    assert separated["disposition"] == "separated"
    assert no_witnesses == []


def test_scenario_6834_attack_suite_and_complete_artifact(
    source_payloads: tuple[dict, dict, bytes, bytes], recomputed: list[dict]
) -> None:
    """SCENARIO-CONSTRAINT-6834-COMPLETENESS is procedural, not effect-signed."""

    fixture, corpus, fixture_bytes, corpus_bytes = source_payloads
    attacks = exp.run_attack_suite(fixture, corpus, recomputed)
    assert {row["attack_id"] for row in attacks} == set(exp.ATTACK_IDS)
    assert all(row["passed"] for row in attacks)

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        fixture_bytes=fixture_bytes,
        corpus_bytes=corpus_bytes,
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["operational_saturation_audit_complete"] is True
    assert artifact["identifiability_result"]["disposition"] == "not_separated"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["independent_parser_id"] == exp.INDEPENDENT_PARSER_ID
    assert artifact["independent_reducer_id"] == exp.INDEPENDENT_REDUCER_ID
    assert artifact["gate_check_summary"]["passed"] is True
    assert len([row for row in artifact["per_unit_rows"] if row["row_type"] == "source"]) == 900
    assert len([row for row in artifact["per_unit_rows"] if row["row_type"] == "attack"]) == len(
        exp.ATTACK_IDS
    )
    assert exp.validate_artifact(artifact) == []

    rebuilt = deepcopy(artifact)
    rebuilt["duration_s"] = 999.0
    assert exp.reproducibility_checksum(rebuilt) == artifact["reproducibility_checksum"]
    broken = deepcopy(artifact)
    broken["field_principles"].pop("schema")
    broken["operational_saturation_audit_complete"] = False
    assert {
        "field principles do not cover every top-level field",
        "complete artifact is not marked complete",
        "reproducibility checksum mismatch",
    } <= set(exp.validate_artifact(broken))


def test_req_6834_atomic_writer_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_payloads: tuple[dict, dict, bytes, bytes],
) -> None:
    """REQ-CONSTRAINT-6834 writes the dated terminal artifact through its command surface."""

    _, _, fixture_bytes, corpus_bytes = source_payloads
    direct_path = tmp_path / "direct.json"
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        fixture_bytes=fixture_bytes,
        corpus_bytes=corpus_bytes,
    )
    exp.write_artifact(direct_path, artifact)
    assert json.loads(direct_path.read_text(encoding="utf-8")) == artifact

    output_path = tmp_path / "cli.json"
    monkeypatch.setattr(exp, "FIXTURE_PATH", REPO_ROOT / exp.FIXTURE_PATH)
    monkeypatch.setattr(exp, "CORPUS_PATH", REPO_ROOT / exp.CORPUS_PATH)
    assert exp.main(["--date", "20260901", "--output", str(output_path)]) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["run_date"] == "20260901"
    assert written["operational_saturation_audit_complete"] is True


def test_req_6834_defensive_parser_join_and_reducer_paths(
    source_payloads: tuple[dict, dict, bytes, bytes], recomputed: list[dict]
) -> None:
    """REQ-CONSTRAINT-6834 fails closed on malformed rows, joins, and count cells."""

    fixture, _, _, _ = source_payloads
    assert exp._json_object(b"not-json") == {}
    assert exp._json_object(b"[]") == {}
    assert exp._rate(1, 0) is None

    source = deepcopy(recomputed[0])
    source.pop("raw_output_bytes_b64")
    assert exp._decode_receipt(source)[1] == "missing_raw_output_bytes"
    source["raw_output_bytes_b64"] = "not-base64"
    assert exp._decode_receipt(source)[1] == "invalid_raw_output_base64"
    source = deepcopy(recomputed[0])
    source["raw_output_byte_length"] += 1
    assert exp._decode_receipt(source)[1] == "raw_output_length_mismatch"
    source = deepcopy(recomputed[0])
    source["raw_output_sha256"] = "sha256:" + "0" * 64
    assert exp._decode_receipt(source)[1] == "raw_output_hash_mismatch"

    with pytest.raises(exp.AuditError, match="unknown_scenario"):
        exp.recompute_source_rows(fixture, [{"scenario_id": "missing"}])
    assert len(exp.join_prompt_arms([*recomputed, {"row_type": "attack"}])) == 450
    with pytest.raises(exp.AuditError, match="duplicate_matched_arm"):
        exp.join_prompt_arms([recomputed[0], recomputed[0]])
    with pytest.raises(exp.AuditError, match="incomplete_matched_pair"):
        exp.join_prompt_arms([recomputed[0]])
    with pytest.raises(exp.AuditError, match="bootstrap_replicates"):
        exp.bootstrap_interval([1.0], seed=1, replicates=0)
    with pytest.raises(exp.AuditError, match="not_enough_labels"):
        exp.permute_group_labels([{"arm": "typed"}], "arm")
    with pytest.raises(exp.AuditError, match="action_label_attack"):
        exp.permute_action_labels(fixture["scenarios"][0], b"not-json")

    bad = deepcopy(recomputed[0])
    bad["template_id"] = "bad"
    with pytest.raises(exp.AuditError, match="invalid_template_id"):
        exp._interaction_pair_key(bad)
    bad = deepcopy(recomputed[0])
    bad["scenario_id"] = "bad"
    with pytest.raises(exp.AuditError, match="invalid_scenario_id"):
        exp._interaction_pair_key(bad)
    with pytest.raises(exp.AuditError, match="duplicate_interaction_pair"):
        exp._build_interaction_penalties([recomputed[0], recomputed[0]])
    with pytest.raises(exp.AuditError, match="incomplete_interaction_pair"):
        exp._build_interaction_penalties([recomputed[0]])

    joint_cell = exp._build_joint_metrics(recomputed)[0]
    with pytest.raises(exp.AuditError, match="invalid_count_order"):
        exp._build_saturation_curves([joint_cell])
    assert exp.metric_signature([recomputed[0], {"row_type": "attack"}]) == exp.metric_signature(
        [recomputed[0]]
    )


def test_scenario_6834_malformed_source_receipts_are_detected(
    source_payloads: tuple[dict, dict, bytes, bytes],
) -> None:
    """SCENARIO-CONSTRAINT-6834-PRECONDITIONS reports byte and process damage."""

    _, corpus, _, _ = source_payloads
    row = deepcopy(corpus["per_unit_rows"][0])
    row["raw_output_sha256"] = "sha256:" + "0" * 64
    row["prompt_bytes_b64"] = "not-base64"
    errors = exp._raw_receipt_errors([row])
    assert any("raw_output_hash_mismatch" in error for error in errors)
    assert any("invalid_prompt_base64" in error for error in errors)

    row = deepcopy(corpus["per_unit_rows"][0])
    row["prompt_byte_length"] += 1
    row["prompt_sha256"] = "sha256:" + "0" * 64
    errors = exp._raw_receipt_errors([row])
    assert any("prompt_length_mismatch" in error for error in errors)
    assert any("prompt_hash_mismatch" in error for error in errors)

    missing = {
        "per_unit_rows": [{"model_id": "model", "process_identity": {"session_id": "missing"}}],
        "process_receipts": [],
    }
    missing_errors = exp._process_receipt_errors(missing)
    assert "missing_process_receipt:missing" in missing_errors
    assert "incomplete_process_or_teardown:missing" in missing_errors
    assert "incomplete_model_canaries" in missing_errors

    receipt = {
        "session_id": "session",
        "purpose": "corpus",
        "model_id": "other-model",
        "authentic": False,
    }
    damaged = {
        "per_unit_rows": [{"model_id": "model", "process_identity": {"session_id": "session"}}],
        "process_receipts": [receipt],
    }
    damaged_errors = exp._process_receipt_errors(damaged)
    assert "process_model_mismatch:session" in damaged_errors
    assert "incomplete_process_or_teardown:session" in damaged_errors


def test_req_6834_validator_and_unreadable_source_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_payloads: tuple[dict, dict, bytes, bytes],
) -> None:
    """REQ-CONSTRAINT-6834 validates each terminal invariant and read failure."""

    _, corpus, fixture_bytes, corpus_bytes = source_payloads
    complete = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        fixture_bytes=fixture_bytes,
        corpus_bytes=corpus_bytes,
    )

    def errors_after(**changes: object) -> set[str]:
        changed = deepcopy(complete)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        return set(exp.validate_artifact(changed))

    missing = deepcopy(complete)
    missing.pop("source_artifact_hashes")
    missing["field_principles"].pop("source_artifact_hashes")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    assert "required artifact fields are missing" in exp.validate_artifact(missing)
    assert "inference substrate mismatch" in errors_after(inference_substrate="wrong")
    assert "independent parser identity mismatch" in errors_after(independent_parser_id="wrong")
    assert "independent reducer identity mismatch" in errors_after(independent_reducer_id="wrong")
    assert "verifier_is_oracle must be false" in errors_after(verifier_is_oracle=True)
    assert "verdict class is outside the closed set" in errors_after(verdict_class="wrong")
    assert "honest verdict lacks a terminal prefix" in errors_after(honest_verdict="partial")

    blocked = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        fixture_bytes=fixture_bytes,
        corpus_bytes=json.dumps({**corpus, "operational_saturation_corpus_ready": False}).encode(),
    )

    def blocked_errors(**changes: object) -> set[str]:
        changed = deepcopy(blocked)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        return set(exp.validate_artifact(changed))

    assert "blocked status mismatch" in blocked_errors(status="wrong")
    assert "blocked artifact reduced source rows" in blocked_errors(per_unit_rows=[{}])
    assert "blocked artifact lacks failed check" in blocked_errors(gate_check_summary={})

    assert "complete artifact source-row count mismatch" in errors_after(per_unit_rows=[])
    assert "complete artifact row coverage is invalid" in errors_after(
        row_coverage={"valid": False}
    )
    assert "complete artifact attack rows are invalid" in errors_after(
        per_unit_rows=[row for row in complete["per_unit_rows"] if row["row_type"] == "source"]
    )
    assert "attack results do not match per-unit attack rows" in errors_after(attack_results=[])
    assert "identifiability disposition is not terminal" in errors_after(
        identifiability_result={"disposition": "unknown"}
    )

    unreadable = exp._read_source(tmp_path / "missing.json")
    assert exp._json_object(unreadable)["read_error"] == "FileNotFoundError"
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced failure"])
    monkeypatch.setattr(exp, "FIXTURE_PATH", REPO_ROOT / exp.FIXTURE_PATH)
    monkeypatch.setattr(exp, "CORPUS_PATH", REPO_ROOT / exp.CORPUS_PATH)
    assert exp.main(["--date", "20260901", "--output", str(tmp_path / "never.json")]) == 1
