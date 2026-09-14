"""Tests for REQ-VERIFY-7294 and SCENARIO-VERIFY-7294-*.

The tests use the immutable live capture as read-only evidence. They never call
the experiment entrypoint, so the repository result cannot change during a
test run.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7294_v641_reuse_audit as mod


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reconstructed() -> dict[str, object]:
    """Authenticate and reconstruct the live rows once for the test module."""

    checks, bundle = mod.authenticate_inputs(ROOT)
    assert all(row["passed"] is True for row in checks)
    return mod.reconstruct_audit(bundle)


def test_scenario_verify_7294_auth_ignores_upstream_efficacy() -> None:
    """SCENARIO-VERIFY-7294-AUTH gates on capture, not upstream efficacy."""

    upstream = {
        "experiment_id": "exp7293-reuse-measurement",
        "milestone": "2026.09.641",
        "status": "complete",
        "reuse_capture_complete_score": 1,
        "reuse_value_score": 0,
        "verdict_class": "disqualified",
        "honest_verdict": "complete_disqualified_reuse_validation_failed",
    }

    checks = mod.measurement_contract_checks(upstream)

    assert all(row["passed"] is True for row in checks)
    assert "reuse_value_score" not in mod.canonical_json(checks)
    assert "verdict_class" not in mod.canonical_json(checks)
    failed = deepcopy(upstream)
    failed["reuse_capture_complete_score"] = 0
    summary = mod.gate_summary(mod.measurement_contract_checks(failed))
    assert summary == {
        "failed_check": "reuse_capture_complete",
        "upstream": "exp7293-reuse-measurement",
        "field": "reuse_capture_complete_score",
        "expected_value": 1,
        "observed_value": 0,
    }


def test_scenario_verify_7294_replay_reconstructs_live_rows(
    reconstructed: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7294-REPLAY rebuilds every decision from saved bytes."""

    rows = reconstructed["rows"]
    group_rows = reconstructed["group_rows"]
    accounting = reconstructed["audit_reconstruction"]
    assert isinstance(rows, list)
    assert isinstance(group_rows, list)
    assert isinstance(accounting, dict)

    assert len(rows) == 384
    assert len(group_rows) == 48
    assert accounting["authenticated_call_file_count"] == 672
    assert accounting["replayed_call_count"] == 672
    assert accounting["decision_row_count"] == 384
    assert accounting["upstream_headline_aggregates_used"] is False
    assert reconstructed["cached_fresh_prediction_mismatch_count"] == 0
    assert reconstructed["stale_served_count"] == 0
    assert reconstructed["unsupported_claim_denominator"] > 0
    assert reconstructed["span_checks"]["mismatch_count"] == 0

    by_arm = reconstructed["condition_results"]
    assert isinstance(by_arm, dict)
    for arm in mod.ARMS:
        result = by_arm[arm]
        assert result["query_count"] == 128
        assert result["accuracy_denominator"] == 128
        assert result["coverage_denominator"] == 128
        assert result["false_accept_denominator"] == 128
        assert result["abstention_count"] + result["coverage_count"] == 128

    source_failures = reconstructed["source_failure_clusters"]
    assert isinstance(source_failures, list)
    assert all(row["dependent_claim_count"] == 4 for row in source_failures)


def test_scenario_verify_7294_bootstrap_resamples_whole_groups(
    reconstructed: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7294-BOOTSTRAP keeps correlated claims together."""

    intervals = mod.paired_group_intervals(
        reconstructed["rows"],
        reconstructed["group_cost_rows"],
        seed=mod.BOOTSTRAP_SEED,
        draws=10_000,
    )

    assert intervals["resampling_unit"] == "source_group"
    assert intervals["independent_group_count"] == 16
    assert intervals["claims_per_group"] == 8
    assert intervals["draw_count"] == 10_000
    assert intervals["bootstrap_seed"] == mod.BOOTSTRAP_SEED
    for name in ("accuracy_difference", "coverage_difference", "full_cost_speedup"):
        assert intervals[name]["group_denominator"] == 16
        assert "estimate" in intervals[name]
        assert "one_sided_95_lower" in intervals[name]
    assert intervals["false_accept_counts"]["denominator_per_arm"] == 128
    assert intervals == mod.paired_group_intervals(
        reconstructed["rows"],
        reconstructed["group_cost_rows"],
        seed=mod.BOOTSTRAP_SEED,
        draws=10_000,
    )


def test_scenario_verify_7294_controls_reject_every_corruption(
    reconstructed: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7294-CONTROLS detects all five frozen mutations."""

    controls = mod.run_integrity_controls(reconstructed["control_basis"])

    assert [row["mutation"] for row in controls] == list(mod.MUTATIONS)
    assert all(row["expected_rejection"] is True for row in controls)
    assert all(row["audit_rejected"] is True for row in controls)
    assert all(row["passed"] is True for row in controls)
    assert len({row["mutated_evidence_sha256"] for row in controls}) == 5
    assert all(
        row["baseline_evidence_sha256"] != row["mutated_evidence_sha256"] for row in controls
    )


def test_scenario_verify_7294_gates_require_cold_and_steady_gain(
    reconstructed: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7294-GATES rejects warm-only value."""

    intervals = mod.paired_group_intervals(
        reconstructed["rows"],
        reconstructed["group_cost_rows"],
        seed=mod.BOOTSTRAP_SEED,
        draws=400,
    )
    controls = mod.run_integrity_controls(reconstructed["control_basis"])
    warm_only = deepcopy(reconstructed)
    warm_only["cost_gains"]["cold_gain_s"] = -0.01
    warm_only["cost_gains"]["steady_state_gain_s"] = 1.0

    gates = mod.acceptance_gates(warm_only, intervals, controls)
    by_name = {row["criterion"]: row for row in gates}

    assert by_name["cold_cost_positive_gain"]["passed"] is False
    assert by_name["steady_state_positive_gain"]["passed"] is True
    assert mod.audit_complete_score(gates) == 1
    assert mod.promotion_score(gates, verifier_is_oracle=True) == 0
    assert mod.classify_verdict(gates, verifier_is_oracle=True) == (
        "null",
        "complete_null_reuse_promotion_gates_failed",
    )


def test_req_verify_7294_blocked_and_schema_boundaries() -> None:
    """REQ-VERIFY-7294 keeps absent evidence blocked and validates terminal schema."""

    checks, bundle = mod.authenticate_inputs(ROOT / "not-present")
    assert bundle == {}
    assert any(row["passed"] is False for row in checks)
    assert mod.gate_summary(checks)["failed_check"] == "exp7293_artifact_hash"

    base = mod.base_artifact("20260914")
    blocked = mod.finalize_blocked_artifact(base, checks)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["reuse_audit_complete_score"] == 0
    assert blocked["reuse_promotion_score"] == 0
    assert mod.validate_artifact(blocked) == []

    broken = deepcopy(blocked)
    broken["model_invoked"] = True
    broken["reproducibility_checksum"] = mod.artifact_checksum(broken)
    assert "model_invoked" in mod.validate_artifact(broken)
    with pytest.raises(ValueError, match="run_date"):
        mod._date_argument("20260915")
    assert mod._date_argument("20260914") == "20260914"


def test_scenario_verify_7294_e2e_candidate_is_cold_valid(
    reconstructed: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-7294-E2E builds one cold-valid measured candidate."""

    intervals = mod.paired_group_intervals(
        reconstructed["rows"],
        reconstructed["group_cost_rows"],
        seed=mod.BOOTSTRAP_SEED,
        draws=400,
    )
    controls = mod.run_integrity_controls(reconstructed["control_basis"])
    artifact = mod.assemble_measured_artifact(
        mod.base_artifact("20260914"),
        reconstructed,
        intervals,
        controls,
        source_hashes={"fixture": "sha256:test"},
    )
    artifact["validation_receipts"] = mod.validation_receipt_fixture()
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)

    assert mod.validate_artifact(artifact) == []
    assert artifact["reuse_audit_complete_score"] == 1
    assert artifact["reuse_promotion_score"] in {0, 1}
    target = tmp_path / "terminal.json"
    mod.write_terminal_artifact(artifact, target)
    assert json.loads(target.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_7294_defensive_boundaries_are_fail_closed(
    reconstructed: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-VERIFY-7294-E2E exercises corrupt-input and schema boundaries."""

    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping_required"):
        mod._read_mapping(non_mapping)

    call_dir = tmp_path / mod.UPSTREAM_RAW_DIR
    call_dir.mkdir(parents=True)
    bad_hash = call_dir / "call_01.json"
    bad_hash.write_text('{"completion": {"call_id": "bad-hash"}}\n', encoding="utf-8")
    bad_completion = call_dir / "call_02.json"
    bad_completion.write_text('{"completion": []}\n', encoding="utf-8")
    wrong_identity = call_dir / "call_03.json"
    wrong_identity.write_text('{"completion": {"call_id": "wrong-identity"}}\n', encoding="utf-8")
    manifest = {
        "calls": [
            {},
            {"sha256": "sha256:wrong"},
            {"sha256": mod.sha256_file(bad_completion)},
            {"sha256": mod.sha256_file(wrong_identity), "call_id": "expected"},
        ]
    }
    call_checks, completions = mod._call_file_checks(tmp_path, manifest)
    assert call_checks[0]["observed_value"]["errors"] == [
        "call_0:missing",
        "call_1:hash",
        "call_2:completion",
        "call_3:identity",
    ]
    assert len(completions) == 1

    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod,
            "_path_hash_checks",
            lambda _root: [mod.gate_row("hashes", 1, 1, True, upstream="test", field="sha256")],
        )
        parse_checks, parse_bundle = mod.authenticate_inputs(tmp_path)
    assert parse_bundle == {}
    assert parse_checks[-1]["check"] == "input_parse"
    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod,
            "measurement_contract_checks",
            lambda _upstream: [
                mod.gate_row(
                    "forced_contract_failure", 1, 0, False, upstream="test", field="capture"
                )
            ],
        )
        failed_checks, failed_bundle = mod.authenticate_inputs(ROOT)
    assert failed_bundle == {}
    assert any(row["check"] == "forced_contract_failure" for row in failed_checks)

    assert mod._compiled(None)["errors"] == ["missing_compilation"]
    assert mod._execute_pair({}, None, None)["errors"] == ["source_compile_failed"]
    usable_source = {"usable": True, "compiled_completion": {}}
    assert mod._execute_pair({}, usable_source, None)["errors"] == ["claim_compile_failed"]
    span_rows = [
        {"call_id": "no-document"},
        {
            "call_id": "bad-span",
            "compiled_completion": {
                "relations": [
                    {
                        "subject_start": 0,
                        "subject_end": 1,
                        "subject_surface": "wrong",
                        "object_start": 1,
                        "object_end": 2,
                        "object_surface": "b",
                    }
                ]
            },
        },
    ]
    spans = mod._span_checks(
        span_rows,
        [{"call_id": "no-document"}, {"call_id": "bad-span", "document": {"text": "ab"}}],
    )
    assert spans == {
        "checked_span_count": 2,
        "mismatch_count": 1,
        "mismatches": ["bad-span:subject"],
    }

    missing_group = deepcopy(reconstructed["control_basis"])
    missing_group["source_receipts"][0]["group_id"] = "absent-group"
    assert "source_version_receipt_mismatch" in mod.integrity_findings(missing_group)

    intervals = mod.paired_group_intervals(
        reconstructed["rows"],
        reconstructed["group_cost_rows"],
        seed=mod.BOOTSTRAP_SEED,
        draws=400,
    )
    controls = mod.run_integrity_controls(reconstructed["control_basis"])
    gates = mod.acceptance_gates(reconstructed, intervals, controls)
    assert mod.classify_verdict([], verifier_is_oracle=True)[0] == "disqualified"
    all_passed = [dict(row, passed=True) for row in gates]
    assert mod.classify_verdict(all_passed, verifier_is_oracle=True)[0] == "circular_positive"
    assert mod.classify_verdict(all_passed, verifier_is_oracle=False)[0] == "positive"
    assert mod.promotion_score(all_passed, verifier_is_oracle=False) == 1

    source_hashes = mod._source_hashes(ROOT)
    assert source_hashes[mod.MODULE_PATH.as_posix()]["sha256"].startswith("sha256:")
    commands = mod.validation_commands(ROOT, ROOT / mod.RAW_DIR)
    assert {name for name, _command in commands} == set(mod.REQUIRED_VALIDATIONS)
    mod._progress(9, "coverage_boundary", unit=1)
    assert "phase=9 event=coverage_boundary unit=1" in capsys.readouterr().out

    artifact = mod.assemble_measured_artifact(
        mod.base_artifact("20260914"),
        reconstructed,
        intervals,
        controls,
        source_hashes=source_hashes,
    )
    artifact["validation_receipts"] = mod.validation_receipt_fixture()
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)

    invalid = deepcopy(artifact)
    invalid.update(
        {
            "schema": "wrong",
            "experiment_id": "wrong",
            "run_date": "wrong",
            "MODEL_SPECS": [{}],
            "model_invoked": True,
            "invocation_counts": {},
            "inference_substrate": "wrong",
            "inference_substrate_class": "wrong",
            "execution_venue": "wrong",
            "status": "partial",
            "verdict_class": "partial",
            "honest_verdict": "partial",
            "rows": [],
            "group_condition_rows": [],
            "source_integrity_control_rows": [],
            "paired_group_intervals": {"draw_count": 0},
            "reuse_audit_complete_score": 0,
            "reuse_promotion_score": 1,
            "validation_receipts": [],
        }
    )
    invalid.pop("field_principles")
    invalid_errors = set(mod.validate_artifact(invalid))
    assert {
        "schema",
        "identity",
        "run_date",
        "required_fields",
        "MODEL_SPECS",
        "invocation_counts",
        "status",
        "verdict_class",
        "rows",
        "validation_receipts",
        "reproducibility_checksum",
    }.issubset(invalid_errors)
    with pytest.raises(ValueError, match="invalid Exp7294"):
        mod.write_terminal_artifact(invalid, tmp_path / "invalid.json")

    receipt_invalid = deepcopy(artifact)
    receipt_invalid["validation_receipts"][0]["exit_code"] = None
    receipt_invalid["validation_receipts"][0]["passed"] = False
    receipt_invalid["verdict_class"] = "null"
    receipt_errors = mod.validate_artifact(receipt_invalid)
    assert "validation_receipt_exit_codes" in receipt_errors
    assert "validation_verdict" in receipt_errors

    blocked = mod.finalize_blocked_artifact(
        mod.base_artifact("20260914"),
        [mod.gate_row("external", 1, 0, False, upstream="test", field="capture")],
    )
    blocked["honest_verdict"] = "wrong"
    blocked["verdict_class"] = "null"
    blocked["reuse_audit_complete_score"] = 1
    blocked["gate_check_summary"] = {}
    blocked_errors = mod.validate_artifact(blocked)
    assert "blocked_verdict" in blocked_errors
    assert "blocked_scores" in blocked_errors
    assert "blocked_gate_check_summary" in blocked_errors

    empty_summary, empty_errors = mod.replay_terminal_candidate(ROOT / "not-present")
    assert empty_summary == {}
    assert empty_errors[0] == "exp7293_artifact_hash"
    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "audit_complete_score", lambda _gates: 0)
        replay_summary, replay_errors = mod.replay_terminal_candidate(ROOT)
    assert replay_summary["rows"] == 384
    assert "audit_complete_score" in replay_errors
