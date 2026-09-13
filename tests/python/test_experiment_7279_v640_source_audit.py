"""Tests for REQ-VERIFY-7279 and SCENARIO-VERIFY-7279-*.

The tests use the frozen Exp7278 evidence. They never call a model or write to
the checked-in result paths.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7279_v640_source_audit as audit


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bundle() -> audit.JsonDict:
    """Load the frozen bytes once so each test audits the same evidence."""

    return audit.load_upstream_bundle(ROOT)


@pytest.fixture(scope="module")
def reduced(bundle: audit.JsonDict) -> audit.JsonDict:
    """Run the independent parser once for all semantic assertions."""

    return audit.independent_reduce(bundle)


@pytest.fixture(scope="module")
def terminal_artifact(bundle: audit.JsonDict, reduced: audit.JsonDict) -> audit.JsonDict:
    """Build one cold-valid terminal artifact for mutation checks."""

    comparisons = audit.paired_bootstrap(reduced["rows"], audit.BOOTSTRAP_SEED)
    controls = audit.causal_controls(bundle, reduced)
    receipts = [
        {
            "name": name,
            "command": f"fixture:{name}",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"results/raw/experiment_7279/validation/{name}.log",
            "log_sha256": audit.sha256_bytes(name.encode()),
        }
        for name in audit.REQUIRED_VALIDATION_NAMES
    ]
    return audit.assemble_measured_artifact(
        ROOT,
        bundle,
        reduced,
        comparisons,
        controls,
        validation_receipts=receipts,
        started_at_utc="2026-09-13T12:00:00+00:00",
        completed_at_utc="2026-09-13T12:00:01+00:00",
        duration_s=1.0,
    )


def test_raw_reducer_ignores_parser_broken_producer_summary(
    bundle: audit.JsonDict, reduced: audit.JsonDict
) -> None:
    """REQ-VERIFY-7279 / SCENARIO-VERIFY-7279-RAW."""

    changed = deepcopy(bundle)
    changed["upstream"]["rows"] = []
    changed["upstream"]["representation_metric_rows"] = [
        {"arm": "mention_pointer", "accuracy": 0.0}
    ]
    changed["upstream"]["source_value_observation"] = {"accuracy_delta": 1.0}

    replayed = audit.independent_reduce(changed)

    assert replayed == reduced
    assert reduced["replay_mismatches"] == []
    assert len(reduced["raw_replay_rows"]) == 256
    assert len(reduced["rows"]) == 256
    assert {row["base_group_id"] for row in reduced["rows"]} == {
        f"g-{index:02d}" for index in range(16)
    }


def test_arm_counts_preserve_null_science_and_all_unknowns(reduced: audit.JsonDict) -> None:
    """REQ-VERIFY-7279 keeps raw numerators before any pooled headline."""

    summaries = reduced["arm_summaries"]
    verifier = summaries["verifier"]
    direct = summaries["direct_self_consistency"]

    assert verifier["accuracy"] == {"numerator": 62, "denominator": 64, "value": 0.96875}
    assert direct["accuracy"] == {"numerator": 64, "denominator": 64, "value": 1.0}
    assert verifier["source_exactness"] == {
        "numerator": 60,
        "denominator": 64,
        "value": 0.9375,
    }
    assert verifier["claim_exactness"] == {"numerator": 64, "denominator": 64, "value": 1.0}
    assert verifier["abstentions"]["numerator"] == 18
    assert direct["abstentions"]["numerator"] == 16
    assert summaries["direct_one_shot"]["secondary"] is True
    assert summaries["source_shuffle_control"]["additional_model_calls"] == 0
    assert reduced["matched_coverage_risk"]["threshold_selected_from_labels"] is False


def test_bootstrap_resamples_sixteen_bases_not_sixty_four_conditions(
    reduced: audit.JsonDict,
) -> None:
    """REQ-VERIFY-7279 / SCENARIO-VERIFY-7279-PAIRED."""

    first = audit.paired_bootstrap(reduced["rows"], seed=audit.BOOTSTRAP_SEED, draws=10_000)
    second = audit.paired_bootstrap(reduced["rows"], seed=audit.BOOTSTRAP_SEED, draws=10_000)
    by_metric = {row["comparison"] + ":" + row["metric"]: row for row in first}

    assert first == second
    assert all(row["independent_base_groups"] == 16 for row in first)
    assert all(row["conditions_per_base_group"] == 4 for row in first)
    accuracy = by_metric["verifier_minus_direct_self_consistency:accuracy_difference"]
    assert accuracy["estimate"] == -0.03125
    assert accuracy["ci95"][0] <= accuracy["ci95"][1] <= 0.0
    false_accept = by_metric["verifier_minus_direct_self_consistency:false_accept_difference"]
    assert false_accept["estimate"] == 0.0
    assert false_accept["ci95"] == [0.0, 0.0]


def test_controls_have_fixed_dispositions_and_failed_value_is_complete_null(
    bundle: audit.JsonDict, reduced: audit.JsonDict
) -> None:
    """REQ-VERIFY-7279 / SCENARIO-VERIFY-7279-CONTROLS."""

    comparisons = audit.paired_bootstrap(reduced["rows"], audit.BOOTSTRAP_SEED)
    controls = audit.causal_controls(bundle, reduced)
    criteria = audit.acceptance_results(reduced, comparisons, controls)
    outcome = audit.classify_result(criteria, complete=True)

    assert {row["control"] for row in controls} == {
        "wrong_authority",
        "source_replacement",
        "shuffled_labels",
        "consistent_renaming",
        "deleted_evidence",
    }
    assert all("expected" in row and "observed" in row and "passed" in row for row in controls)
    assert (
        next(row for row in criteria if row["criterion"] == "accuracy_ci95_lower_above_zero")[
            "passed"
        ]
        is False
    )
    assert outcome == {
        "status": "complete",
        "source_audit_complete_score": 1,
        "source_promotion_score": 0,
        "verdict_class": "null",
        "honest_verdict": "complete_null_verifier_did_not_beat_matched_direct_self_consistency",
    }


def test_authentication_detects_source_and_authority_replacement(bundle: audit.JsonDict) -> None:
    """REQ-VERIFY-7279 fails closed on unauthenticated upstream bytes."""

    assert all(row["passed"] for row in audit.authenticate_bundle(bundle))
    changed_public = deepcopy(bundle)
    changed_public["public"]["rows"][0]["source"]["text"] = "replacement"
    changed_authority = deepcopy(bundle)
    changed_authority["authority"]["rows"][0]["expected_decision"] = "unknown"

    public_checks = audit.authenticate_bundle(changed_public)
    authority_checks = audit.authenticate_bundle(changed_authority)

    assert audit.gate_summary(public_checks)["failed_check"] == "public_manifest_identity"
    assert audit.gate_summary(authority_checks)["failed_check"] == "private_authority_identity"


def test_deleted_raw_evidence_is_a_replay_mismatch_and_abstention(
    bundle: audit.JsonDict,
) -> None:
    """REQ-VERIFY-7279 retains missing evidence instead of filtering its unit."""

    changed = deepcopy(bundle)
    del changed["call_files"][0]

    reduced = audit.independent_reduce(changed)

    assert "call_0:missing_file" in reduced["replay_mismatches"]
    affected = [
        row
        for row in reduced["rows"]
        if row["unit_id"] == bundle["schedule"][0]["unit_id"] and row["arm"] == "verifier"
    ]
    assert affected[0]["abstention"] is True
    assert affected[0]["censored"] is True


def test_terminal_artifact_is_complete_null_and_cold_valid(
    tmp_path: Path, terminal_artifact: audit.JsonDict
) -> None:
    """REQ-VERIFY-7279 / SCENARIO-VERIFY-7279-E2E."""

    artifact = deepcopy(terminal_artifact)
    output = tmp_path / "artifact.json"
    audit.write_terminal_artifact(artifact, output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert audit.validate_artifact(artifact, require_validations=True) == []
    assert artifact["status"] == "complete"
    assert artifact["source_audit_complete_score"] == 1
    assert artifact["source_promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == audit.ZERO_INVOCATION_COUNTS
    assert artifact["claim_boundary"]["gsm8k_claim"] is False


def test_missing_external_input_is_terminal_blocked(tmp_path: Path) -> None:
    """REQ-VERIFY-7279 / SCENARIO-VERIFY-7279-PREFLIGHT."""

    checks, loaded = audit.collect_preconditions(tmp_path)
    artifact = audit.finalize_blocked_artifact(
        audit.base_artifact(audit.RUN_DATE), checks, duration_s=0.0
    )

    assert loaded == {}
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["upstream"] == "exp7278-source-measurement"
    assert artifact["source_audit_complete_score"] == 0
    assert artifact["source_promotion_score"] == 0
    assert artifact["invocation_counts"] == audit.ZERO_INVOCATION_COUNTS
    assert audit.validate_artifact(artifact, require_validations=False) == []


def test_fixed_date_parser_rejects_other_dates() -> None:
    """REQ-VERIFY-7279 fixes the execution date before measurement."""

    assert audit._date_argument(audit.RUN_DATE) == audit.RUN_DATE
    with pytest.raises(Exception, match="run date must be 20260913"):
        audit._date_argument("20260912")


def test_successful_preflight_and_manifest_helpers(bundle: audit.JsonDict, tmp_path: Path) -> None:
    """REQ-VERIFY-7279 authenticates the complete source chain before reduction."""

    checks, loaded = audit.collect_preconditions(ROOT)
    assert loaded["upstream"] == bundle["upstream"]
    assert all(row["passed"] for row in checks)
    assert audit._manifest_lists_experiment({"nested": [{"experiment_id": "x"}]}, {"x"})
    assert audit._manifest_lists_experiment({"outer": {"id": "x"}}, {"x"})
    assert not audit._manifest_lists_experiment({"note": "x"}, {"x"})

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping_required"):
        audit._read_json(list_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in (
        "public_manifest.json",
        "private_authority_manifest.json",
        "schedule.json",
        "raw_call_manifest.json",
    ):
        (raw_dir / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schedule_list"):
        audit.load_upstream_bundle(ROOT, raw_dir=raw_dir)


def test_preflight_records_missing_raw_and_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7279 turns raw absence and parse failure into exact gates."""

    checks, loaded = audit.collect_preconditions(ROOT, raw_dir=tmp_path)
    assert loaded == {}
    assert audit.gate_summary(checks)["failed_check"] == "required_raw_input"

    def fail_load(*_args: object, **_kwargs: object) -> audit.JsonDict:
        raise ValueError("fixture failure")

    monkeypatch.setattr(audit, "load_upstream_bundle", fail_load)
    checks, loaded = audit.collect_preconditions(ROOT)
    assert loaded == {}
    assert audit.gate_summary(checks)["failed_check"] == "upstream_parse"


def test_authentication_and_leakage_failure_paths(bundle: audit.JsonDict) -> None:
    """REQ-VERIFY-7279 detects missing, changed, retired, and leaked evidence."""

    missing = deepcopy(bundle)
    missing["call_files"].pop(0)
    assert audit.gate_summary(audit.authenticate_bundle(missing))["failed_check"] == (
        "raw_call_file_set"
    )
    changed = deepcopy(bundle)
    changed["call_files"][0]["bytes"] = b"changed"
    assert audit.gate_summary(audit.authenticate_bundle(changed))["failed_check"] == (
        "raw_call_file_set"
    )
    retired = deepcopy(bundle)
    retired["exclusion_manifest"] = {"nested": [{"experiment_ids": [audit.EXPERIMENT_ID]}]}
    assert audit.gate_summary(audit.authenticate_bundle(retired))["failed_check"] == (
        "retirement_manifest"
    )
    quarantined = deepcopy(bundle)
    quarantined["upstream"]["quarantined"] = True
    quarantined["upstream"]["reproducibility_checksum"] = audit.artifact_checksum(
        quarantined["upstream"]
    )
    assert "structured_quarantine" in {
        row["check"] for row in audit.authenticate_bundle(quarantined) if not row["passed"]
    }
    leaked = deepcopy(bundle["schedule"][:1])
    leaked[0]["expected_decision"] = "supported"
    leakage = audit.source_leakage_errors(leaked, {"authority_path_opened_by_model_worker": True})
    assert leakage == [
        "call_0:private_fields:expected_decision",
        "authority_path_opened_by_model_worker",
    ]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("not json", "pointer_json_parse"),
        ('{"outcome":"known"}', "pointer_shape"),
        ('{"outcome":"known","relations":[]}', "pointer_outcome"),
        (
            '{"outcome":"known","relations":[{"predicate":"precedes"}]}',
            "pointer_relation_shape",
        ),
        (
            '{"outcome":"known","relations":[{"object_pointer":"bad",'
            '"polarity":"positive","predicate":"precedes","subject_pointer":"m000"}]}',
            "pointer_unknown_mention",
        ),
        (
            '{"outcome":"known","relations":[{"object_pointer":"m001",'
            '"polarity":"bad","predicate":"precedes","subject_pointer":"m000"}]}',
            "pointer_polarity",
        ),
        (
            '{"outcome":"known","relations":[{"object_pointer":"m001",'
            '"polarity":"positive","predicate":"","subject_pointer":"m000"}]}',
            "pointer_predicate",
        ),
    ],
)
def test_independent_pointer_parser_rejects_invalid_shapes(
    bundle: audit.JsonDict, content: str, expected: str
) -> None:
    """REQ-VERIFY-7279 keeps every pointer parser failure explicit."""

    parsed, errors = audit._parse_pointer(content, bundle["schedule"][0]["document"])
    assert parsed is None
    assert errors == [expected]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("not json", "direct_json_parse"),
        ('{"other":"supported"}', "direct_shape"),
        ('{"decision":"bad"}', "direct_decision"),
    ],
)
def test_independent_direct_parser_rejects_invalid_shapes(content: str, expected: str) -> None:
    """REQ-VERIFY-7279 keeps every direct parser failure explicit."""

    decision, errors = audit._parse_direct(content)
    assert decision is None
    assert errors == [expected]


def test_low_level_parser_and_executor_failure_paths(bundle: audit.JsonDict) -> None:
    """REQ-VERIFY-7279 preserves corrupt transport and unresolved relations."""

    with pytest.raises(ValueError, match="base64_type"):
        audit._decode_b64(None)
    with pytest.raises(ValueError, match="base64_invalid"):
        audit._decode_b64("!")

    public = bundle["public"]["rows"][0]
    unknown_relation = {
        "outcome": "known",
        "relations": [
            {
                "object_pointer": "m001",
                "polarity": "negative",
                "predicate": "precedes",
                "subject_pointer": "m000",
            }
        ],
    }
    assert audit._decide_pointer(
        public["source"], public["claim"], unknown_relation, unknown_relation
    ) == ("supported", [])
    changed_claim = deepcopy(unknown_relation)
    changed_claim["relations"][0]["predicate"] = "ends before"
    assert audit._decide_pointer(
        public["source"], public["claim"], unknown_relation, changed_claim
    ) == ("unknown", ["relation_not_resolved"])
    assert audit._direct_vote([]) == ("unknown", "unknown", False, [])


def test_raw_replay_reports_each_integrity_failure(bundle: audit.JsonDict) -> None:
    """REQ-VERIFY-7279 authenticates both transport bytes and schedule joins."""

    malformed = deepcopy(bundle)
    malformed["schedule"] = malformed["schedule"][:-1]
    malformed["call_files"][0]["value"]["schedule"] = {}
    malformed["call_files"][1]["value"]["completion"] = None
    malformed["call_files"][2]["bytes"] = b"different"
    malformed["call_files"][3]["value"]["completion"]["raw_request_bytes_b64"] = "!"
    completion = malformed["call_files"][4]["value"]["completion"]
    completion["request_bytes_sha256"] = "wrong"
    completion["response_bytes_sha256"] = "wrong"
    completion["actual_parameters"] = {}
    completion["raw_completion"] = "wrong"
    completion["raw_completion_sha256"] = "wrong"
    malformed["schedule"][4]["prompt"] = "wrong prompt"
    malformed["call_files"][5]["value"]["completion"]["raw_response_bytes_b64"] = base64.b64encode(
        b'{"choices":[{"message":{"content":1}}]}'
    ).decode("ascii")

    _rows, errors = audit.replay_raw_calls(malformed)

    assert "schedule_denominator" in errors
    assert "call_0:schedule" in errors
    assert "call_1:completion" in errors
    assert "call_2:file_sha256" in errors
    assert any(error.startswith("call_3:transport:") for error in errors)
    assert "call_4:request_sha256" in errors
    assert "call_4:response_sha256" in errors
    assert "call_4:request_payload" in errors
    assert "call_4:request_schedule_join" in errors
    assert "call_4:response_content" in errors
    assert "call_4:completion_sha256" in errors
    assert any("call_5:transport:ValueError:transport_shape" in error for error in errors)


def test_bootstrap_and_classification_reject_incomplete_inputs(reduced: audit.JsonDict) -> None:
    """REQ-VERIFY-7279 rejects invalid resampling and keeps unfinished work partial."""

    with pytest.raises(ValueError, match="bootstrap_draws"):
        audit.paired_bootstrap(reduced["rows"], audit.BOOTSTRAP_SEED, draws=0)
    with pytest.raises(ValueError, match="paired_base_groups"):
        audit.paired_bootstrap(reduced["rows"][:-1], audit.BOOTSTRAP_SEED, draws=1)
    with pytest.raises(ValueError, match="comparison"):
        audit._comparison([], "missing", "accuracy")
    assert audit.classify_result([], complete=False)["verdict_class"] == "partial"
    passing = [
        {"category": category, "passed": True}
        for category in ("completeness", "value", "independence")
    ]
    assert audit.classify_result(passing, complete=True)["verdict_class"] == ("circular_positive")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(schema="wrong"), "identity"),
        (lambda value: value.update(field_principles={}), "field_principles"),
        (lambda value: value.update(model_invoked=True), "model_contract"),
        (lambda value: value.update(execution_host=""), "execution_identity"),
        (lambda value: value.update(duration_s=-1), "duration_s"),
        (lambda value: value.update(verifier_is_oracle=False), "verifier_is_oracle"),
        (lambda value: value.update(arm_summaries={}), "arm_summaries"),
        (lambda value: value.update(paired_comparisons=[]), "paired_comparisons"),
        (lambda value: value.update(causal_control_rows=[]), "causal_control_rows"),
        (lambda value: value.update(acceptance_gate_results=[]), "acceptance_gate_results"),
        (lambda value: value.update(honest_verdict="complete_wrong"), "terminal_classification"),
        (lambda value: value.update(inference_substrate="aggregation"), "inference_substrate"),
        (lambda value: value.update(validation_receipts=[]), "validation_receipts"),
        (lambda value: value.update(claim_boundary={}), "claim_boundary"),
    ],
)
def test_cold_validator_rejects_terminal_mutations(
    terminal_artifact: audit.JsonDict,
    mutation: object,
    expected: str,
) -> None:
    """REQ-VERIFY-7279 cold validation recomputes every terminal decision."""

    changed = deepcopy(terminal_artifact)
    assert callable(mutation)
    mutation(changed)
    changed["reproducibility_checksum"] = audit.artifact_checksum(changed)
    assert expected in audit.validate_artifact(changed, require_validations=True)


def test_cold_validator_early_failures_and_invalid_write(
    terminal_artifact: audit.JsonDict, tmp_path: Path
) -> None:
    """REQ-VERIFY-7279 refuses malformed, incomplete, and checksum-broken artifacts."""

    assert audit.validate_artifact([], require_validations=False) == ["artifact_mapping"]
    assert audit.validate_artifact({}, require_validations=False)[0].startswith(
        "missing_required_field:"
    )
    broken_checksum = deepcopy(terminal_artifact)
    broken_checksum["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum" in audit.validate_artifact(
        broken_checksum, require_validations=True
    )
    running = deepcopy(terminal_artifact)
    running["status"] = "running"
    running["reproducibility_checksum"] = audit.artifact_checksum(running)
    assert "status" in audit.validate_artifact(running, require_validations=False)
    no_rows = deepcopy(terminal_artifact)
    no_rows["rows"] = []
    no_rows["reproducibility_checksum"] = audit.artifact_checksum(no_rows)
    assert "row_denominator" in audit.validate_artifact(no_rows, require_validations=False)
    invalid_groups = deepcopy(terminal_artifact)
    for row in invalid_groups["rows"]:
        row["base_group_id"] = "one-group"
    invalid_groups["arm_summaries"] = audit.summarize_arms(invalid_groups["rows"])
    invalid_groups["reproducibility_checksum"] = audit.artifact_checksum(invalid_groups)
    assert "paired_base_groups" in audit.validate_artifact(
        invalid_groups, require_validations=False
    )
    invalid_arms = deepcopy(terminal_artifact)
    invalid_arms["rows"][0]["arm"] = "direct_self_consistency"
    invalid_arms["reproducibility_checksum"] = audit.artifact_checksum(invalid_arms)
    assert "arm_denominators" in audit.validate_artifact(invalid_arms, require_validations=False)
    with pytest.raises(ValueError, match="invalid Exp7279 artifact"):
        audit.write_terminal_artifact(no_rows, tmp_path / "invalid.json")


def test_blocked_validator_rejects_success_shape(tmp_path: Path) -> None:
    """REQ-VERIFY-7279 keeps blocked evidence distinct from a success placeholder."""

    checks, _loaded = audit.collect_preconditions(tmp_path)
    artifact = audit.finalize_blocked_artifact(
        audit.base_artifact(audit.RUN_DATE), checks, duration_s=0
    )
    artifact["rows"] = [{"fake": True}]
    artifact["reproducibility_checksum"] = audit.artifact_checksum(artifact)
    assert "blocked_terminal_state" in audit.validate_artifact(artifact, require_validations=False)
