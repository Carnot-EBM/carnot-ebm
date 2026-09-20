"""Tests for REQ-VERIFY-7452 and SCENARIO-VERIFY-7452-*.

These tests cover the label-blind eligibility pass, authenticated final-token
vectors, identity-only joins, shard replay controls, and readiness reduction.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from carnot import experiment_7452_v653_source_embeddings as exp


def _predictor(row: str, group: str, role: str = "training") -> dict[str, object]:
    """Build one public predictor row with no evaluator-only fields."""

    return {
        "row_key": row,
        "group_id": group,
        "corpus": "ragtruth",
        "role": role,
        "source_text": f"Source for {group}.",
        "response_text": f"Response for {row}.",
        "source_features": {"numeric_novelty_with_context": 0.0},
    }


def _evaluator(row: str, group: str, response: str) -> dict[str, object]:
    """Build one private evaluator identity with an irrelevant label."""

    return {
        "row_key": row,
        "group_id": group,
        "corpus": "ragtruth",
        "role": "training",
        "source_id": f"source-{group}",
        "response_id": response,
        "official_split": "train",
        "label": 1,
        "annotation_labels": [],
        "ambiguous": False,
        "label_policy": "fixture",
    }


class _Tokenizer:
    """Return byte-sized token IDs so tests can exercise the exact ceiling."""

    def tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


def _completed_row(row_key: str, group_id: str, role: str = "training") -> dict[str, object]:
    """Return one completed cell produced through the real vector reducer."""

    identity = {
        "row_key": row_key,
        "group_id": group_id,
        "source_hash": f"sha256:{group_id:0<64}"[:71],
        "response_id": f"response-{row_key}",
        "role": role,
        "corpus": "ragtruth",
        "arm": "response",
    }
    return exp.vector_row(identity, [1.0, -2.0, 3.0], token_count=3)


def test_req_verify_7452_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7452 and all seven scenarios exist before behavior changes."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "### REQ-VERIFY-7452:" in text
    for number in range(1, 8):
        assert f"SCENARIO-VERIFY-7452-{number:02d}" in text


def test_scenario_verify_7452_01_authenticates_exact_upstream_values(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7452-01 records distinct missing, null, zero, and flag causes."""

    results = tmp_path / "results"
    results.mkdir()
    lifecycle = {
        "capture_lifecycle_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    protocol = {
        "source_protocol_ready_score": 1,
        "verdict_class": "positive",
        "flagged_adversarial": False,
    }
    (results / exp.LIFECYCLE_PATH.name).write_text(json.dumps(lifecycle), encoding="utf-8")
    (results / exp.PROTOCOL_PATH.name).write_text(json.dumps(protocol), encoding="utf-8")
    checks, hashes = exp.collect_upstream_preconditions(tmp_path)
    assert all(row["passed"] for row in checks)
    assert exp.gate_check_summary(checks) == {"passed": True, "failed_check": None}
    assert set(hashes) == {exp.LIFECYCLE_PATH.as_posix(), exp.PROTOCOL_PATH.as_posix()}

    protocol["source_protocol_ready_score"] = 0
    protocol["flagged_adversarial"] = True
    (results / exp.PROTOCOL_PATH.name).write_text(json.dumps(protocol), encoding="utf-8")
    failed, _ = exp.collect_upstream_preconditions(tmp_path)
    summary = exp.gate_check_summary(failed)
    assert summary["field"] == "source_protocol_ready_score"
    assert summary["observed"] == 0
    assert any(row["field"] == "flagged_adversarial" and not row["passed"] for row in failed)

    (results / exp.LIFECYCLE_PATH.name).unlink()
    missing, _ = exp.collect_upstream_preconditions(tmp_path)
    assert any(row["observed"] == "missing" for row in missing)


def test_scenario_verify_7452_03_token_eligibility_is_label_blind_and_ordered() -> None:
    """SCENARIO-VERIFY-7452-03 counts complete inputs before evaluator access."""

    predictors = [_predictor("row-b", "group-b"), _predictor("row-a", "group-a")]
    rows = exp.build_eligibility_rows(predictors, _Tokenizer(), token_ceiling=40)
    assert [row["seal_order"] for row in rows] == list(range(4))
    assert [row["seal_hash"] for row in rows] == sorted(row["seal_hash"] for row in rows)
    assert all("label" not in row and "response_id" not in row for row in rows)
    assert any(
        row["eligible"] is False and row["status"] == "excluded_token_ceiling" for row in rows
    )
    assert any(row["eligible"] is True and row["status"] == "unstarted" for row in rows)
    assert all(row["truncated"] is False for row in rows)


def test_scenario_verify_7452_04_join_uses_identities_not_position() -> None:
    """SCENARIO-VERIFY-7452-04 joins a permuted evaluator view by row identity."""

    predictors = [_predictor("row-a", "group-a"), _predictor("row-b", "group-b")]
    eligibility = exp.build_eligibility_rows(predictors, _Tokenizer(), token_ceiling=100)
    evaluators = [
        _evaluator("row-b", "group-b", "response-b"),
        _evaluator("row-a", "group-a", "response-a"),
    ]
    joined = exp.bind_response_identities(eligibility, evaluators)
    assert {row["row_key"]: row["response_id"] for row in joined} == {
        "row-a": "response-a",
        "row-b": "response-b",
    }
    corrupted = deepcopy(evaluators)
    corrupted[0]["group_id"] = "group-a"
    with pytest.raises(exp.CaptureInvalid, match="identity mismatch"):
        exp.bind_response_identities(eligibility, corrupted)
    with pytest.raises(exp.CaptureInvalid, match="identity set mismatch"):
        exp.bind_response_identities(eligibility, evaluators[:-1])


def test_scenario_verify_7452_02_vector_reducer_is_finite_stable_and_projected() -> None:
    """SCENARIO-VERIFY-7452-02 records an exact raw final-token vector."""

    identity = {
        "row_key": "row-a",
        "group_id": "group-a",
        "source_hash": "sha256:" + "a" * 64,
        "response_id": "response-a",
        "role": "training",
        "corpus": "ragtruth",
        "arm": "response",
    }
    row = exp.vector_row(identity, [1.25, -2.5, 4.0], token_count=7)
    assert row["status"] == "completed"
    assert row["dimension"] == 3
    assert row["norm"] == pytest.approx(math.sqrt(23.8125))
    assert len(row["projected_features"]) == exp.PROJECTION_DIMENSIONS
    assert exp.decode_vector(row) == pytest.approx([1.25, -2.5, 4.0])
    assert exp.vector_row(identity, [1.25, -2.5, 4.0], token_count=7) == row
    with pytest.raises(exp.CaptureInvalid, match="nonfinite"):
        exp.vector_row(identity, [float("nan")], token_count=1)
    with pytest.raises(exp.CaptureInvalid, match="empty"):
        exp.vector_row(identity, [], token_count=0)


def test_scenario_verify_7452_05_shard_reload_and_corruption_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7452-05 rejects identity permutation and false dimensions."""

    rows = [_completed_row("a", "a"), _completed_row("b", "b")]
    shards = exp.write_feature_shards(tmp_path, rows, groups_per_shard=16)
    assert len(shards) == 1
    loaded = exp.reload_feature_shards(tmp_path, shards)
    assert [row["vector_hash"] for row in loaded] == [row["vector_hash"] for row in rows]
    controls = exp.run_integrity_controls(loaded)
    assert controls == {
        "authentic_subset_reloaded": True,
        "permuted_join_rejected": True,
        "fabricated_dimension_rejected": True,
        "passed": True,
    }

    altered = deepcopy(shards)
    altered[0]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(exp.CaptureInvalid, match="shard hash"):
        exp.reload_feature_shards(tmp_path, altered)

    wrong_count = deepcopy(shards)
    wrong_count[0]["rows"] = 99
    with pytest.raises(exp.CaptureInvalid, match="row count"):
        exp.reload_feature_shards(tmp_path, wrong_count)


def test_scenario_verify_7452_05_vector_corruptions_fail_closed() -> None:
    """SCENARIO-VERIFY-7452-05 validates each raw numerical receipt."""

    row = _completed_row("a", "a")
    corruptions = []
    bad_encoding = deepcopy(row)
    bad_encoding["vector_b64"] = "not base64!"
    corruptions.append((bad_encoding, "encoding"))
    bad_hash = deepcopy(row)
    bad_hash["vector_hash"] = "sha256:" + "0" * 64
    corruptions.append((bad_hash, "hash"))
    bad_norm = deepcopy(row)
    bad_norm["norm"] = 0.0
    corruptions.append((bad_norm, "norm"))
    bad_projection = deepcopy(row)
    bad_projection["projected_features"] = [0.0] * exp.PROJECTION_DIMENSIONS
    corruptions.append((bad_projection, "projection"))
    nan_payload = struct.pack("<f", float("nan"))
    bad_finite = deepcopy(row)
    bad_finite.update(
        {
            "dimension": 1,
            "vector_b64": base64.b64encode(nan_payload).decode("ascii"),
            "vector_hash": "sha256:" + hashlib.sha256(nan_payload).hexdigest(),
        }
    )
    corruptions.append((bad_finite, "nonfinite"))
    for corrupted, message in corruptions:
        with pytest.raises(exp.CaptureInvalid, match=message):
            exp.decode_vector(corrupted)


def test_scenario_verify_7452_04_shard_size_limit_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7452-04 refuses a shard at the twenty MiB ceiling."""

    monkeypatch.setattr(exp, "_jsonl_bytes", lambda rows: b"x" * (20 * 1024 * 1024))
    monkeypatch.setattr(exp, "_atomic_bytes", lambda path, payload: None)
    with pytest.raises(exp.CaptureInvalid, match="20 MiB"):
        exp.write_feature_shards(tmp_path, [_completed_row("a", "a")])


def test_scenario_verify_7452_06_readiness_requires_complete_paired_role_minima() -> None:
    """SCENARIO-VERIFY-7452-06 retains all dispositions and enforces group minima."""

    rows: list[dict[str, object]] = []
    for role, count in exp.MINIMUM_GROUPS.items():
        for index in range(count):
            group = f"{role}-{index}"
            for arm in exp.ARMS:
                row = _completed_row(f"{group}-{arm}", group, role)
                row["arm"] = arm
                rows.append(row)
    reduction = exp.reduce_capture(rows, controls_passed=True, validation_passed=True)
    assert reduction["embedding_capture_ready_score"] == 1
    assert reduction["eligible_complete_groups"] == exp.MINIMUM_GROUPS
    assert reduction["honest_verdict"] == "complete_null_source_embeddings_captured"

    incomplete = deepcopy(rows)
    incomplete[-1] = {
        **incomplete[-1],
        "status": "failed",
        "completed": False,
        "failed": True,
    }
    failed = exp.reduce_capture(incomplete, controls_passed=True, validation_passed=True)
    assert failed["embedding_capture_ready_score"] == 0
    assert failed["verdict_class"] == "null"
    assert failed["coverage_diagnostics"]["failed_cells"] == 1

    partial_rows = deepcopy(rows)
    partial_rows[-1] = {
        **partial_rows[-1],
        "status": "censored_forward_budget",
        "completed": False,
        "censored": True,
    }
    partial = exp.reduce_capture(partial_rows, controls_passed=True, validation_passed=True)
    assert partial["verdict_class"] == "partial"

    disqualified = exp.reduce_capture(rows, controls_passed=False, validation_passed=True)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["flagged_adversarial"] is True

    deferred = exp.reduce_capture(
        [
            {
                "role": "external",
                "group_id": "deferred",
                "arm": "response",
                "eligible": None,
                "status": "unstarted",
                "attempted": False,
                "completed": False,
                "failed": False,
                "censored": False,
                "unstarted": True,
            }
        ],
        controls_passed=True,
        validation_passed=True,
    )
    assert deferred["coverage_diagnostics"]["planned_cells"] == 1
    assert deferred["coverage_diagnostics"]["eligible_cells"] == 0
    assert deferred["coverage_diagnostics"]["unstarted_cells"] == 1


def test_scenario_verify_7452_07_artifact_validation_recomputes_checksum(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7452-07 cold validation rejects changed terminal bytes."""

    rows = [_completed_row("a-response", "a"), _completed_row("a-source", "a")]
    rows[1]["arm"] = "source_response"
    reduction = exp.reduce_capture(rows, controls_passed=True, validation_passed=True)
    artifact = exp.build_artifact_for_test(rows=rows, reduction=reduction)
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    errors = exp.validate_artifact(changed)
    assert "promotion_score_invalid" in errors
    assert "reproducibility_checksum_invalid" in errors

    invalid = deepcopy(artifact)
    invalid.pop("schema")
    invalid.update(
        {
            "milestone": "wrong",
            "MODEL_SPECS": [],
            "inference_substrate_class": "wrong",
            "execution_venue": "wrong",
            "verdict_class": "wrong",
            "verifier_is_oracle": True,
            "field_principles": {},
            "final_layer_scope_statement": "wrong",
        }
    )
    invalid["invocation_counts"]["generation_calls"]["attempted"] = 1
    invalid_errors = exp.validate_artifact(invalid)
    assert {
        "identity_invalid",
        "run_identity_invalid",
        "model_specs_invalid",
        "generation_counts_invalid",
        "inference_substrate_class_invalid",
        "execution_venue_invalid",
        "verdict_class_invalid",
        "verifier_authority_invalid",
        "field_principles_invalid",
        "surface_scope_invalid",
    } <= set(invalid_errors)

    output = tmp_path / "artifact.json"
    exp.atomic_json(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == exp.SCHEMA


def test_cli_date_is_fixed() -> None:
    """REQ-VERIFY-7452 rejects an execution date outside the frozen milestone."""

    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE
    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "20260919"])
