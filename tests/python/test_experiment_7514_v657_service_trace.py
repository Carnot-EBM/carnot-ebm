"""Tests for the controlled durable service trace.

Spec refs: REQ-CL-7514 and SCENARIO-CL-7514-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7514_v657_service_trace as exp


def _feature_rows() -> list[dict[str, object]]:
    """Return exactly the qualified training denominator."""

    return [
        {
            "group_id": f"group-{index:03d}",
            "role": "training",
            "features": [float(index + offset) / 100.0 for offset in range(10)],
            "label": index % 2,
            "raw_whole_expectation": 0.2 + 0.6 * (index % 2),
            "source_hash": f"sha256:source-{index:03d}",
        }
        for index in range(176)
    ]


def _plan_rows() -> list[dict[str, object]]:
    """Give the first 24 groups whole and focused calls in both orders."""

    rows: list[dict[str, object]] = []
    orders = (
        ["supported", "contains_unsupported"],
        ["contains_unsupported", "supported"],
    )
    for index in range(24):
        for window_index in (None, 0, 1, 2):
            for order_index, order in enumerate(orders):
                rows.append(
                    {
                        "call_id": f"call-{index:03d}-{window_index}-{order_index}",
                        "group_id": f"group-{index:03d}",
                        "source_group_id": f"group-{index:03d}",
                        "role": "training",
                        "arm": "whole_response" if window_index is None else "focused_window",
                        "window_index": window_index,
                        "option_order": order,
                        "prompt": f"prompt-{index}-{window_index}-{order_index}",
                        "eligible": True,
                    }
                )
    return rows


def _native_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach finite semantic probabilities to a frozen schedule."""

    rows = []
    for index, source in enumerate(schedule):
        probability = 0.25 + (index % 5) * 0.1
        rows.append(
            {
                **{key: value for key, value in source.items() if key != "prompt"},
                "disposition": "complete",
                "gold_label": None,
                "source_sha256": f"sha256:source-{index // 8:03d}",
                "response_sha256": f"sha256:response-{index // 8:03d}",
                "probabilities_by_option_id": {
                    "supported": 1.0 - probability,
                    "contains_unsupported": probability,
                },
                "label_to_option_id": {
                    " A": source["option_order"][0],
                    " B": source["option_order"][1],
                },
                "generated_tokens": 0,
            }
        )
    return rows


def test_frozen_schedule_uses_24_of_176_training_groups() -> None:
    """REQ-CL-7514: group support and the native call ceiling are fixed."""

    schedule = exp.build_frozen_schedule(_feature_rows(), _plan_rows())
    assert len({row["group_id"] for row in schedule}) == 24
    assert len(schedule) == 192
    assert all(row["role"] == "training" for row in schedule)
    assert {tuple(row["option_order"]) for row in schedule} == set(exp.OPTION_ORDERS)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("denominator", "training_group_denominator_invalid"),
        ("window", "qualified_group_window_count_invalid"),
        ("order", "semantic_order_pair_invalid"),
    ],
)
def test_frozen_schedule_rejects_changed_qualifiers(mutation: str, message: str) -> None:
    """SCENARIO-CL-7514-BLOCKED: a changed schedule cannot be truncated."""

    features = _feature_rows()
    plan = _plan_rows()
    if mutation == "denominator":
        features.pop()
    elif mutation == "window":
        plan.append({**plan[0], "call_id": "extra", "window_index": 3})
    else:
        plan[1]["option_order"] = list(plan[0]["option_order"])
    with pytest.raises(exp.ServiceTraceError, match=message):
        exp.build_frozen_schedule(features, plan)


def test_numeric_pair_has_durable_parity_and_alternating_order(tmp_path: Path) -> None:
    """SCENARIO-CL-7514-TRACE: both arms cross the same durable boundary."""

    training = np.asarray([row["features"][:4] for row in _feature_rows()])
    head = exp.make_update_head(training, learning_rate=0.03, residual_bound=1.0)
    rows = []
    for index in range(20):
        rows.extend(
            exp.trace_numeric_pair(
                request_id=f"request-{index:02d}",
                group_id=f"group-{index:03d}",
                features=_feature_rows()[index]["features"][:4],
                base_probability=0.3 + 0.02 * (index % 10),
                label=index % 2,
                update_head=head,
                durable_root=tmp_path,
                update_first=index % 2 == 0,
            )
        )
    by_request: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_request.setdefault(str(row["request_id"]), []).append(row)
        assert row["acknowledged"] is True
        assert row["fsync_completed"] is True
        assert row["stage_order"] == list(exp.NUMERIC_STAGES)
        assert row["accounted_ns"] == row["total_ns"]
        assert row["unaccounted_ns"] == 0
        assert Path(str(row["durable_path"])).is_file()
    assert all(len(value) == 2 for value in by_request.values())
    assert [row["arm"] for row in by_request["request-00"]] == ["update", "no_update"]
    assert [row["arm"] for row in by_request["request-01"]] == ["no_update", "update"]


def test_reducer_separates_shared_readout_and_numeric_overhead(tmp_path: Path) -> None:
    """SCENARIO-CL-7514-BOUNDARY: cached arithmetic is not a model call."""

    training = np.asarray([row["features"][:4] for row in _feature_rows()])
    head = exp.make_update_head(training, learning_rate=0.03, residual_bound=1.0)
    request_rows = []
    for index in range(20):
        pair = exp.trace_numeric_pair(
            request_id=f"request-{index:02d}",
            group_id=f"group-{index:03d}",
            features=_feature_rows()[index]["features"][:4],
            base_probability=0.4,
            label=index % 2,
            update_head=head,
            durable_root=tmp_path,
            update_first=index % 2 == 0,
        )
        request_rows.extend(pair)
    shared = [
        {
            "request_id": f"request-{index:02d}",
            "group_id": f"group-{index:03d}",
            "complete": True,
            "native_forward_count": 8,
            "generated_token_count": 0,
            "shared_readout_ns": 1_000_000 + index,
        }
        for index in range(20)
    ]
    reduction = exp.reduce_service_trace(request_rows, shared)
    assert reduction["paired_request_count"] == 20
    assert reduction["completed_requests_per_arm"] == {"no_update": 20, "update": 20}
    assert reduction["durable_parity_passed"] is True
    assert reduction["interval_coverage_passed"] is True
    assert reduction["paired_support_passed"] is True
    assert reduction["shared_native_forward_calls"] == 160
    assert reduction["generated_token_count"] == 0
    assert reduction["paired_overhead_ns"]["median"] is not None
    assert reduction["paired_overhead_ns"]["p95"] is not None
    assert reduction["amdahl_bounds"]["ideal_infinite_speed_update_kernel_ceiling"] >= 1.0


def test_fixture_artifact_validates_and_mutations_fail(tmp_path: Path) -> None:
    """REQ-CL-7514: fresh readers recompute scores and byte identities."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=True) == []
    reduction = exp.independent_reduce(artifact, root=tmp_path, require_terminal=True)
    assert reduction["passed"] is True
    assert reduction["service_trace_complete_score"] == 1
    assert reduction["durable_service_claim_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["readout_kind"] == "option_logits"
    assert artifact["generated_token_count"] == 0
    assert artifact["service_boundaries"]["human_feedback_acquisition_time"] == "unknown_excluded"
    assert set(artifact["field_principles"]) == set(artifact)

    changed = deepcopy(artifact)
    changed["request_rows"][0]["unaccounted_ns"] = 1
    assert "service_reduction_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, require_validation=True
    )
    changed = deepcopy(artifact)
    changed["invocation_counts"]["forward_calls"]["completed"] -= 1
    assert "invocation_counts_invalid" in exp.validate_artifact(
        changed, root=tmp_path, require_validation=True
    )
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, require_validation=True
    )


def test_support_failure_keeps_complete_trace_but_closes_claim(tmp_path: Path) -> None:
    """SCENARIO-CL-7514-GATES: readiness support is independent of validity."""

    artifact = exp.build_artifact_for_test(tmp_path, request_count=19)
    assert artifact["service_trace_complete_score"] == 1
    assert artifact["durable_service_claim_ready_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_checks"][0]["check"] == "paired_support"


def test_blocked_artifact_names_exact_failed_operand(tmp_path: Path) -> None:
    """SCENARIO-CL-7514-BLOCKED: external absence produces no invented rows."""

    failed = {
        "check": "cached_gguf",
        "upstream": "/models/qwen.gguf",
        "field": "is_file",
        "path": "/models/qwen.gguf",
        "expected": True,
        "observed": False,
        "op": "eq",
        "passed": False,
        "principle": "Exact absence prevents a fabricated live trace.",
    }
    artifact = exp.build_blocked_artifact(failed, root=tmp_path)
    assert artifact["honest_verdict"] == "complete_blocked_cached_gguf"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["request_rows"] == []
    assert artifact["service_trace_complete_score"] == 0
    assert artifact["durable_service_claim_ready_score"] == 0
    assert artifact["gate_check_summary"]["failure"] == {
        key: failed[key] for key in ("check", "upstream", "field", "path", "expected", "observed")
    }
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []


def test_artifact_checksum_ignores_only_its_own_field(tmp_path: Path) -> None:
    """REQ-CL-7514: every evidence mutation changes the terminal checksum."""

    artifact = exp.build_artifact_for_test(tmp_path)
    original = exp.artifact_checksum(artifact)
    artifact["reproducibility_checksum"] = "sha256:anything"
    assert exp.artifact_checksum(artifact) == original
    artifact["random_seed"]["audit"] += 1
    assert exp.artifact_checksum(artifact) != original


def test_main_reader_modes_report_valid_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7514: the thin CLI readers are fresh-process compatible."""

    artifact = exp.build_artifact_for_test(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(candidate)]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    assert exp.main(["--date", exp.RUN_DATE, "--independent-reduce", str(candidate)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    with pytest.raises(exp.ServiceTraceError, match="run_date_mismatch"):
        exp.main(["--date", "20260921", "--cold-replay", str(candidate)])


def test_schedule_and_percentile_defensive_boundaries() -> None:
    """REQ-CL-7514: malformed or insufficient schedules fail closed."""

    assert exp._percentile([], 0.5) is None
    assert exp._percentile([7], 0.95) == 7.0
    plan = _plan_rows()
    plan = [row for row in plan if row["group_id"] != "group-023"]
    with pytest.raises(exp.ServiceTraceError, match="qualified_training_support_invalid"):
        exp.build_frozen_schedule(_feature_rows(), plan)

    plan = _plan_rows()
    for row in plan:
        if row["group_id"] == "group-000" and row["window_index"] == 2:
            row["window_index"] = 4
    with pytest.raises(exp.ServiceTraceError, match="qualified_group_window_count_invalid"):
        exp.build_frozen_schedule(_feature_rows(), plan)

    plan = _plan_rows()
    plan[0]["arm"] = "unknown"
    with pytest.raises(exp.ServiceTraceError, match="qualified_group_cell_invalid"):
        exp.build_frozen_schedule(_feature_rows(), plan)

    plan = _plan_rows()
    plan.append({**plan[0], "call_id": "duplicate-cell"})
    with pytest.raises(exp.ServiceTraceError, match="semantic_order_pair_invalid"):
        exp.build_frozen_schedule(_feature_rows(), plan)

    plan = _plan_rows()
    for order_index, order in enumerate(exp.OPTION_ORDERS):
        plan.append(
            {
                **plan[0],
                "call_id": f"extra-window-{order_index}",
                "arm": "focused_window",
                "window_index": 3,
                "option_order": list(order),
            }
        )
    with pytest.raises(exp.ServiceTraceError, match="qualified_training_support_invalid"):
        exp.build_frozen_schedule(_feature_rows(), plan)


def test_reducer_detects_duplicate_arm(tmp_path: Path) -> None:
    """SCENARIO-CL-7514-TRACE: duplicate arm rows break interval coverage."""

    artifact = exp.build_artifact_for_test(tmp_path)
    duplicated = [*artifact["request_rows"], deepcopy(artifact["request_rows"][0])]
    reduced = exp.reduce_service_trace(duplicated, artifact["shared_readout_rows"])
    assert reduced["interval_coverage_passed"] is False


@pytest.mark.parametrize("mutation", ["shape", "receipt", "path", "size"])
def test_raw_receipt_mutations_fail(tmp_path: Path, mutation: str) -> None:
    """REQ-CL-7514: raw sidecar declarations are byte-bound."""

    artifact = exp.build_artifact_for_test(tmp_path)
    if mutation == "shape":
        artifact["raw_receipts"] = {}
    elif mutation == "receipt":
        artifact["raw_receipts"]["request_rows"] = "bad"
    elif mutation == "path":
        artifact["raw_receipts"]["request_rows"]["path"] = str(tmp_path / "missing")
    else:
        artifact["raw_receipts"]["request_rows"]["bytes"] += 1
    assert "raw_receipts_invalid" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=True
    )


def test_validator_rejects_schema_and_blocked_mutations(tmp_path: Path) -> None:
    """SCENARIO-CL-7514-BLOCKED: terminal and blocked schemas fail closed."""

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["schema"] = "changed"
    assert "field_invalid:schema" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=True
    )
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["request_rows"] = "bad"
    assert "request_rows_invalid" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=True
    )
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["native_call_rows"].pop()
    assert "native_call_rows_invalid" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=True
    )
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["validation_receipts"] = []
    assert "required_validation_failed" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=True
    )
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["field_principles"] = {}
    assert "field_principles_incomplete" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=True
    )

    failed = {
        "check": "missing",
        "upstream": "external",
        "field": "available",
        "path": "/missing",
        "expected": True,
        "observed": False,
    }
    blocked = exp.build_blocked_artifact(failed, root=tmp_path)
    blocked["request_rows"] = [{}]
    assert "blocked_rows_or_score_invalid" in exp.validate_artifact(
        blocked, root=tmp_path, require_validation=False
    )
    blocked = exp.build_blocked_artifact(failed, root=tmp_path)
    blocked["gate_check_summary"] = {}
    assert "blocked_gate_summary_invalid" in exp.validate_artifact(
        blocked, root=tmp_path, require_validation=False
    )


def test_provenance_helpers_and_live_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7514: helper receipts preserve boundaries and CLI output."""

    assert exp.utc_now().endswith("Z")
    exp.progress(0.0, "fixture", "complete", units=1)
    assert "phase=fixture event=complete" in capsys.readouterr().out
    row = exp._precondition("check", "upstream", "field", "path", 1, 1)
    assert row["passed"] is True
    span = exp._phase_span("phase", 2.0, 1.0, 3, "sealed")
    assert span["start_s"] == 1.0
    bound = exp._exp7513_substitution_bound({"numeric_placement_ready_score": 1})
    assert bound["qualified"] is False

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"honest_verdict": "complete_fixture"}), encoding="utf-8")
    receipt = exp._source(source, tmp_path)
    assert receipt["path"] == "source.json"
    assert receipt["original_honest_verdict"] == "complete_fixture"
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"value": 1}\n\n', encoding="utf-8")
    assert exp._load_jsonl(jsonl) == [{"value": 1}]

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _root, _date: {
            "honest_verdict": "complete_fixture",
            "service_trace_complete_score": 1,
            "durable_service_claim_ready_score": 1,
            "verdict_class": "null",
        },
    )
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert json.loads(capsys.readouterr().out)["honest_verdict"] == "complete_fixture"
