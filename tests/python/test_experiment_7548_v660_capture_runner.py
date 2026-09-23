"""Tests for the V660 resumable native capture contract.

Spec refs: REQ-VERIFY-7548 and SCENARIO-VERIFY-7548-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7548_v660_capture_runner as exp


ROLE_COUNTS = {"fit": 2, "development": 1}


def _group(role: str, index: int) -> dict[str, object]:
    component = exp.sha256_text(f"component-{role}-{index}")
    donor = exp.sha256_text(f"donor-{role}-{index}")
    requests: list[dict[str, object]] = []
    for condition in exp.CONDITIONS:
        for order in exp.OPTION_ORDERS:
            prompt = f"{role} {index} {condition} {'/'.join(order)}"
            requests.append(
                {
                    "condition": condition,
                    "option_order": list(order),
                    "prompt": prompt,
                    "prompt_sha256": exp.sha256_text(prompt),
                    "prompt_token_count": len(prompt.split()) + 1,
                    "readout_kind": "option_logits",
                    "generated_token_budget": 0,
                }
            )
    return {
        "component_hash": component,
        "donor_component_hash": donor,
        "role": role,
        "tool_type": "grep",
        "label_scope": "original_source_only",
        "requests": requests,
    }


def _schedule() -> list[exp.JsonDict]:
    groups = [_group("fit", 0), _group("fit", 1), _group("development", 0)]
    return exp.compile_capture_schedule(groups, expected_role_counts=ROLE_COUNTS)


def _native_row(request: exp.JsonDict, index: int) -> exp.JsonDict:
    order = list(request["option_order"])
    logits = {order[0]: 1.5 + index / 100.0, order[1]: -0.5}
    probabilities = exp.native_pilot._softmax_pair(logits, order)
    count = int(request["prompt_token_count"])
    return {
        **deepcopy(request),
        "call_id": f"scripted-{index}",
        "disposition": "complete",
        "prompt_utf8_bytes": len(str(request["prompt"]).encode()),
        "prompt_token_ids": list(range(count)),
        "requested_score_position": count - 1,
        "actual_last_evaluated_position": count - 1,
        "display_labels": [" A", " B"],
        "label_token_ids": [101, 102],
        "label_to_option_id": dict(zip((" A", " B"), order, strict=True)),
        "full_logits_by_option_id": logits,
        "probabilities_by_option_id": probabilities,
        "generated_tokens": 0,
        "forward_seconds": 0.01,
        "state_reset": True,
        "server_receipt": {
            "transport": "private_scripted_transport",
            "pid": 123,
            "process_start_ticks": 456,
            "gpu_uuid": "GPU-fixture",
            "n_ctx": exp.N_CTX,
        },
        "error": None,
    }


class ScriptedTransport:
    """Return deterministic native-shaped rows and optionally fail one call."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.calls = 0

    def score(self, request: exp.JsonDict, owner: exp.JsonDict) -> exp.JsonDict:
        del owner
        index = self.calls
        self.calls += 1
        if index == self.fail_at:
            raise RuntimeError("scripted interruption")
        return _native_row(request, index)


def test_schedule_freezes_exact_roles_and_six_cells() -> None:
    """REQ-VERIFY-7548 / SCENARIO-VERIFY-7548-SCHEDULE."""

    schedule = _schedule()
    reduction = exp.validate_capture_schedule(schedule, expected_role_counts=ROLE_COUNTS)

    assert reduction["passed"] is True
    assert reduction["group_counts"] == ROLE_COUNTS
    assert reduction["forward_count"] == 18
    assert len(reduction["role_schedule_hashes"]) == 2
    for group_hash in {row["group_hash"] for row in schedule}:
        rows = [row for row in schedule if row["group_hash"] == group_hash]
        assert len(rows) == 6
        assert {row["condition"] for row in rows} == set(exp.CONDITIONS)


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("donor_component_hash", "sha256:changed", "donor_identity"),
        ("option_order", ["contains_unsupported", "supported"], "request_identity"),
        ("request_id", "changed-request", "request_identity"),
        ("prompt_token_count", 999, "prompt_token_custody"),
        ("group_hash", "sha256:changed", "group_identity"),
    ],
)
def test_independent_reducer_rejects_identity_mutations(
    field: str, value: object, expected_error: str
) -> None:
    """REQ-VERIFY-7548 / SCENARIO-VERIFY-7548-MUTATIONS."""

    schedule = _schedule()
    rows = [_native_row(row, index) for index, row in enumerate(schedule)]
    rows[0][field] = value

    reduction = exp.reduce_capture_rows(rows, schedule, expected_role_counts=ROLE_COUNTS)

    assert reduction["passed"] is False
    assert expected_error in reduction["failed_checks"]


def test_runner_discards_partial_group_and_resumes(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 / SCENARIO-VERIFY-7548-RESUME."""

    schedule = _schedule()
    checkpoint = tmp_path / "checkpoint.json"
    first = exp.ResumableCaptureRunner(
        schedule=schedule,
        transport=ScriptedTransport(fail_at=8),
        checkpoint_path=checkpoint,
        roles=("fit", "development"),
        validation_reserve_s=900.0,
        hard_limit_s=4200.0,
        owner={"pid": 123, "pid_start_ticks": 456},
    ).run()

    assert first["status"] == "partial"
    assert first["completed_group_count"] == 1
    assert first["partial_group_discarded"] == 1
    assert len(first["rows"]) == 6

    second = exp.ResumableCaptureRunner(
        schedule=schedule,
        transport=ScriptedTransport(),
        checkpoint_path=checkpoint,
        roles=("fit", "development"),
        validation_reserve_s=900.0,
        hard_limit_s=4200.0,
        owner={"pid": 123, "pid_start_ticks": 456},
    ).run()
    reduction = exp.reduce_capture_rows(second["rows"], schedule, expected_role_counts=ROLE_COUNTS)

    assert second["status"] == "complete"
    assert second["resumed_group_count"] == 1
    assert second["completed_group_count"] == 3
    assert len(second["rows"]) == 18
    assert reduction["passed"] is True


def test_runner_only_invokes_registered_roles(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 requires explicitly bounded role invocations."""

    transport = ScriptedTransport()
    result = exp.ResumableCaptureRunner(
        schedule=_schedule(),
        transport=transport,
        checkpoint_path=tmp_path / "checkpoint.json",
        roles=("development",),
        validation_reserve_s=900.0,
        hard_limit_s=4200.0,
        owner={},
    ).run()

    assert result["status"] == "complete"
    assert result["completed_group_count"] == 1
    assert transport.calls == 6
    assert {row["role"] for row in result["rows"]} == {"development"}


def test_runner_rejects_checkpoint_from_another_schedule(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 binds resume state to exact schedule bytes."""

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"schedule_sha256": "sha256:wrong", "rows": []}), encoding="utf-8"
    )

    with pytest.raises(exp.CaptureRunnerError, match="checkpoint_schedule_mismatch"):
        exp.ResumableCaptureRunner(
            schedule=_schedule(),
            transport=ScriptedTransport(),
            checkpoint_path=checkpoint,
            roles=("fit",),
            validation_reserve_s=900.0,
            hard_limit_s=4200.0,
            owner={},
        ).run()


def test_schedule_and_rows_sidecars_reduce_independently(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 requires hash-bound independent raw reduction."""

    schedule = _schedule()
    rows = [_native_row(row, index) for index, row in enumerate(schedule)]
    schedule_receipt = exp.write_jsonl_sidecar(tmp_path / "schedule.jsonl", schedule, root=tmp_path)
    rows_receipt = exp.write_jsonl_sidecar(tmp_path / "rows.jsonl", rows, root=tmp_path)
    value = {
        "capture_schedule": schedule_receipt,
        "qualification_rows": rows_receipt,
        "qualification_role_counts": ROLE_COUNTS,
    }

    reduced = exp.independent_reduce(value, root=tmp_path)
    (tmp_path / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    changed = exp.independent_reduce(value, root=tmp_path)

    assert reduced["passed"] is True
    assert reduced["complete_group_count"] == 3
    assert changed == {"passed": False, "error": "sidecar_receipt_invalid"}


def test_gpu_capacity_requires_supported_truly_idle_device() -> None:
    """REQ-VERIFY-7548 / SCENARIO-VERIFY-7548-RESOURCE."""

    idle = {
        "gpu_uuid": "GPU-idle",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "gpu_memory_used_mb": 4,
        "gpu_memory_free_mb": 24000,
        "pid": None,
        "ppid": None,
        "start_time_ticks": None,
    }
    busy = {
        **idle,
        "gpu_uuid": "GPU-busy",
        "gpu_memory_used_mb": 18000,
        "pid": 900,
        "ppid": 800,
        "start_time_ticks": 700,
        "command": ["llama-server"],
    }

    available = exp.reduce_gpu_capacity([busy, idle], [], observed_at="2026-09-23T00:00:00Z")
    unavailable = exp.reduce_gpu_capacity([busy], [], observed_at="2026-09-23T00:00:00Z")

    assert available["gpu_capacity_observed_score"] == 1
    assert available["idle_supported_gpu_uuids"] == ["GPU-idle"]
    assert unavailable["gpu_capacity_observed_score"] == 0
    assert unavailable["signals_sent"] == []
    assert unavailable["leases_acquired"] == []


def test_protocol_authentication_names_all_sealed_boundaries() -> None:
    """REQ-VERIFY-7548 authenticates Exp7533 without opening label rows."""

    checks, context = exp.authenticate_protocol(exp.REPO_ROOT)

    assert all(row["passed"] for row in checks)
    assert context["role_counts"] == exp.UPSTREAM_ROLE_COUNTS
    assert context["actual_label_rows_read"] == 0
    assert set(context["intervention_receipts"]) == {"interventions_000", "interventions_001"}


def test_qualification_controls_fail_every_registered_mutation(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 / SCENARIO-VERIFY-7548-MUTATIONS."""

    controls = exp.run_qualification_controls(tmp_path)

    assert controls["passed"] is True
    assert controls["resume_parity"] is True
    assert controls["mutation_failures"] == {
        "donor_component_hash": True,
        "group_hash": True,
        "option_order": True,
        "prompt_token_count": True,
        "request_id": True,
    }


def test_artifact_keeps_runner_readiness_separate_from_busy_gpu() -> None:
    """REQ-VERIFY-7548 keeps validity separate from resource availability."""

    artifact = exp.build_artifact_for_test(gpu_capacity_observed_score=0)

    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert artifact["capture_runner_ready_score"] == 1
    assert artifact["gpu_capacity_observed_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_gpu_capacity_unavailable"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["positive_claim"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_artifact_idle_observation_remains_null_not_positive() -> None:
    """REQ-VERIFY-7548 says resource capacity cannot become benefit."""

    artifact = exp.build_artifact_for_test(gpu_capacity_observed_score=1)

    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert artifact["capture_runner_ready_score"] == 1
    assert artifact["gpu_capacity_observed_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["positive_claim"] is False
    assert (
        artifact["honest_no_headroom_annotation"] == "runner_qualification_only_benefit_unmeasured"
    )


def test_blocked_artifact_retains_exact_failed_operand() -> None:
    """REQ-VERIFY-7548 routes external absence with exact evidence."""

    failed = exp.gate(
        "sealed_protocol_missing",
        "external_precondition",
        True,
        None,
        "==",
        False,
        "A missing input cannot become fabricated collection evidence.",
        upstream="exp7533-v659-tool-protocol",
        field="sealed_shards.predictor",
        path="results/raw/experiment_7533_v659_tool_protocol/predictor.jsonl",
    )
    artifact = exp.build_blocked_artifact(
        failed_gate=failed,
        preconditions=[failed],
        duration_s=0.1,
        phase_spans=[],
    )

    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["capture_runner_ready_score"] == 0
    assert artifact["sample_size_budget"]["unstarted"] == exp.TOTAL_GROUPS
    assert artifact["gate_check_summary"]["first_failure"]["observed"] is None


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ({"capture_runner_ready_score": True}, "capture_runner_ready_score_not_bare_int"),
        ({"gpu_capacity_observed_score": 2}, "gpu_capacity_observed_score_not_bare_int"),
        ({"model_invoked": True}, "model_invoked_mismatch"),
        ({"MODEL_SPECS": ["legacy-model"]}, "MODEL_SPECS_mismatch"),
        ({"verdict_class": "success"}, "verdict_class_invalid"),
        ({"flagged_adversarial": True}, "flagged_adversarial_mismatch"),
    ],
)
def test_artifact_validator_rejects_overclaim_mutations(
    mutation: dict[str, object], expected_error: str
) -> None:
    """REQ-VERIFY-7548 cold validation fails closed on claim drift."""

    artifact = exp.build_artifact_for_test(gpu_capacity_observed_score=1)
    artifact.update(mutation)
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)

    assert expected_error in exp.validate_artifact(artifact, require_validation=False)


def test_checksum_and_field_principles_cover_terminal_artifact() -> None:
    """REQ-VERIFY-7548 binds all fields and their failure-prevention reasons."""

    artifact = exp.build_artifact_for_test(gpu_capacity_observed_score=1)
    changed = deepcopy(artifact)
    changed["rows"][0]["complete_count"] -= 1

    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert exp.artifact_checksum(changed) != artifact["reproducibility_checksum"]
    assert set(artifact) <= set(artifact["field_principles"])
    assert all(artifact["field_principles"].values())


def test_cli_date_and_read_modes_are_fixed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7548-E2E fixes date and read-only replay modes."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")

    args = exp.parse_args(["--date", exp.RUN_DATE, "--cold-replay", str(candidate)])
    assert args.date == exp.RUN_DATE
    assert args.cold_replay == candidate
    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "20260924"])


def test_main_cold_replay_and_independent_reduce(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-VERIFY-7548-E2E executes fresh-process reader behavior."""

    artifact = exp.build_artifact_for_test(gpu_capacity_observed_score=1)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")

    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(candidate)]) == 0
    assert "cold_replay_passed" in capsys.readouterr().out


def _passing_receipts(include_terminal: bool = False) -> list[exp.JsonDict]:
    names = list(exp.AFFECTED_CHECK_NAMES)
    if include_terminal:
        names.extend(exp.TERMINAL_CHECK_NAMES)
    return [
        {"name": name, "passed": True, "exit_code": 0, "log_sha256": "sha256:fixture"}
        for name in names
    ]


def test_complete_artifact_builder_reduces_private_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7548 derives readiness from raw controls and validation."""

    schedule = _schedule()
    schedule_receipt = exp.write_jsonl_sidecar(
        tmp_path / "capture.jsonl", schedule, root=exp.REPO_ROOT
    )
    controls = exp.run_qualification_controls(tmp_path / "controls")
    monkeypatch.setattr(exp, "ROLE_COUNTS", ROLE_COUNTS)
    monkeypatch.setattr(exp, "TOTAL_GROUPS", 3)
    monkeypatch.setattr(exp, "TOTAL_FORWARDS", 18)
    manifest = {
        "role_counts": ROLE_COUNTS,
        "forward_count": 18,
        "online_groups_excluded": 160,
        "labels_read": 0,
        "role_schedule_hashes": exp.validate_capture_schedule(
            schedule, expected_role_counts=ROLE_COUNTS
        )["role_schedule_hashes"],
    }
    precondition = exp.gate(
        "fixture",
        "validity",
        True,
        True,
        "==",
        True,
        "Fixture is authenticated.",
        upstream="fixture",
        field="ready",
    )
    resource = {
        "gpu_capacity_observed_score": 1,
        "observed_at_utc": "2026-09-23T00:00:00Z",
        "signals_sent": [],
        "leases_acquired": [],
    }

    ready = exp.build_complete_artifact(
        preconditions=[precondition],
        source_hashes={},
        schedule_receipt=schedule_receipt,
        schedule_manifest=manifest,
        controls=controls,
        resource=resource,
        validation_receipts=_passing_receipts(),
        duration_s=1.0,
        phase_spans=[],
        include_terminal_validation=False,
    )
    blocked = exp.build_complete_artifact(
        preconditions=[precondition],
        source_hashes={},
        schedule_receipt=schedule_receipt,
        schedule_manifest=manifest,
        controls=controls,
        resource={**resource, "gpu_capacity_observed_score": 0},
        validation_receipts=_passing_receipts(),
        duration_s=1.0,
        phase_spans=[],
        include_terminal_validation=False,
    )
    disqualified = exp.build_complete_artifact(
        preconditions=[precondition],
        source_hashes={},
        schedule_receipt=schedule_receipt,
        schedule_manifest=manifest,
        controls=controls,
        resource=resource,
        validation_receipts=[],
        duration_s=1.0,
        phase_spans=[],
        include_terminal_validation=False,
    )

    assert ready["verdict_class"] == "null"
    assert ready["capture_runner_ready_score"] == 1
    assert blocked["verdict_class"] == "blocked"
    assert disqualified["verdict_class"] == "disqualified"


def test_named_preconditions_include_requirement_and_literal_prior_failure() -> None:
    """REQ-VERIFY-7548 checks named inputs before schedule measurement."""

    checks, context = exp.collect_preconditions(exp.REPO_ROOT)

    assert all(row["passed"] for row in checks)
    assert context["historical_pilot"]["honest_verdict"] == ("complete_blocked_owned_gpu_available")
    assert exp.SPEC_PATH.as_posix() in context["source_artifact_hashes"]


def test_independent_reducer_accepts_schema_complete_blocked_result() -> None:
    """REQ-VERIFY-7548 keeps absent external work blocked, not partial."""

    artifact = exp.build_blocked_artifact(
        failed_gate=exp.gate(
            "missing",
            "external_precondition",
            True,
            None,
            "==",
            False,
            "Missing stays blocked.",
            upstream="fixture",
            field="path",
        ),
        preconditions=[],
        duration_s=0.1,
        phase_spans=[],
    )

    assert exp.independent_reduce(artifact) == {
        "passed": True,
        "mode": "blocked_no_measurement",
        "row_count": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ({"invocation_counts": {}}, "invocation_counts_mismatch"),
        ({"honest_verdict": "null"}, "honest_verdict_terminal_prefix_missing"),
        ({"verdict_class": "null", "gpu_capacity_observed_score": 0}, "busy_gpu_verdict_mismatch"),
        ({"verdict_class": "blocked"}, "ready_verdict_mismatch"),
        ({"capture_runner_ready_score": 0, "verdict_class": "null"}, "unready_verdict_mismatch"),
        ({"resource_observations": None}, "resource_observations_invalid"),
        ({"sample_size_budget": {}}, "sample_size_budget_invalid"),
        ({"acceptance_gate_results": None}, "acceptance_gate_results_invalid"),
        ({"field_principles": {}}, "field_principles_incomplete"),
        ({"duration_s": 0}, "duration_invalid"),
        ({"preconditions_checked": None}, "preconditions_checked_invalid"),
        ({"role_schedule_hashes": {}}, "role_schedule_hashes_invalid"),
    ],
)
def test_artifact_validator_covers_fail_closed_schema_branches(
    mutation: dict[str, object], expected_error: str
) -> None:
    """REQ-VERIFY-7548 rejects malformed terminal evidence."""

    artifact = exp.build_artifact_for_test(gpu_capacity_observed_score=1)
    artifact.update(mutation)
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)

    assert expected_error in exp.validate_artifact(artifact, require_validation=False)


def test_artifact_validator_rejects_non_object_and_missing_validation() -> None:
    """REQ-VERIFY-7548 cold readers fail closed."""

    artifact = exp.build_artifact_for_test(gpu_capacity_observed_score=1)

    assert exp.validate_artifact([]) == ["artifact_not_object"]
    assert "required_validation_missing_or_failed" in exp.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (lambda rows: rows[0].update(component_hash="bad"), "donor_identity"),
        (
            lambda rows: rows[0].update(donor_component_hash=rows[0]["component_hash"]),
            "donor_identity",
        ),
        (lambda rows: rows[0].update(group_hash="sha256:bad"), "group_identity"),
        (lambda rows: rows[0].update(condition="unknown"), "request_identity"),
        (lambda rows: rows[0].update(request_id="bad"), "request_identity"),
        (lambda rows: rows[1].update(request_id=rows[0]["request_id"]), "request_identity"),
        (lambda rows: rows[0].update(schedule_index=True), "request_identity"),
        (lambda rows: rows[1].update(schedule_index=rows[0]["schedule_index"]), "request_identity"),
        (lambda rows: rows[0].update(prompt_token_count=0), "prompt_token_custody"),
        (lambda rows: rows[0].update(n_ctx=1), "request_identity"),
        (lambda rows: rows.pop(), "group_cell_product"),
        (lambda rows: rows[0].update(role="other"), "group_identity"),
        (lambda rows: [rows.pop(0) for _ in range(6)], "role_group_counts"),
        (
            lambda rows: [
                row.update(component_hash=rows[0]["component_hash"]) for row in rows[6:12]
            ],
            "component_role_overlap",
        ),
    ],
)
def test_schedule_validator_names_malformed_boundaries(
    mutator: object, expected_error: str
) -> None:
    """SCENARIO-VERIFY-7548-MUTATIONS fails schedule drift closed."""

    schedule = _schedule()
    mutator(schedule)  # type: ignore[operator]

    reduced = exp.validate_capture_schedule(schedule, expected_role_counts=ROLE_COUNTS)

    assert reduced["passed"] is False
    assert expected_error in reduced["failed_checks"]


def test_schedule_compiler_rejects_malformed_group_and_request() -> None:
    """REQ-VERIFY-7548 refuses incomplete source groups before capture."""

    with pytest.raises(exp.CaptureRunnerError, match="group_requests_missing"):
        exp.compile_capture_schedule([{"role": "fit"}], expected_role_counts={"fit": 1})
    group = _group("fit", 0)
    group["requests"] = [None]
    with pytest.raises(exp.CaptureRunnerError, match="request_not_object"):
        exp.compile_capture_schedule([group], expected_role_counts={"fit": 1})
    group = _group("fit", 0)
    group["requests"][0]["condition"] = "unknown"  # type: ignore[index]
    with pytest.raises(exp.CaptureRunnerError, match="request_semantics_invalid"):
        exp.compile_capture_schedule([group], expected_role_counts={"fit": 1})
    group = _group("fit", 0)
    group["requests"].pop()  # type: ignore[union-attr]
    with pytest.raises(exp.CaptureRunnerError, match="capture_schedule_invalid"):
        exp.compile_capture_schedule([group], expected_role_counts={"fit": 1})


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (lambda rows: rows[0].update(full_logits_by_option_id=None), "finite_logits"),
        (
            lambda rows: rows[0].update(
                full_logits_by_option_id={"supported": float("nan"), "contains_unsupported": 0.0}
            ),
            "finite_logits",
        ),
        (lambda rows: rows.append(deepcopy(rows[0])), "request_identity"),
        (lambda rows: rows[0].update(disposition="failed"), "complete_forward_custody"),
        (lambda rows: rows[0].update(probabilities_by_option_id={}), "finite_probabilities"),
        (
            lambda rows: rows[0].update(
                probabilities_by_option_id={
                    "supported": float("nan"),
                    "contains_unsupported": float("nan"),
                }
            ),
            "finite_probabilities",
        ),
        (lambda rows: rows.pop(), "partial_group"),
    ],
)
def test_raw_reducer_names_native_custody_failures(mutator: object, expected_error: str) -> None:
    """REQ-VERIFY-7548 rejects malformed or partial raw groups."""

    schedule = _schedule()
    rows = [_native_row(row, index) for index, row in enumerate(schedule)]
    mutator(rows)  # type: ignore[operator]

    reduced = exp.reduce_capture_rows(rows, schedule, expected_role_counts=ROLE_COUNTS)

    assert reduced["passed"] is False
    assert expected_error in reduced["failed_checks"]


def test_sidecar_reader_rejects_non_object_and_count_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 reads only exact JSONL receipts."""

    path = tmp_path / "bad.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    receipt = {
        "path": "bad.jsonl",
        "sha256": exp.sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": 1,
    }
    with pytest.raises(exp.CaptureRunnerError, match="sidecar_row_not_object"):
        exp._read_jsonl_receipt(receipt, tmp_path)
    path.write_text("{}\n", encoding="utf-8")
    receipt.update(sha256=exp.sha256_file(path), bytes=path.stat().st_size, rows=2)
    with pytest.raises(exp.CaptureRunnerError, match="sidecar_row_count_mismatch"):
        exp._read_jsonl_receipt(receipt, tmp_path)


def test_runner_rejects_invalid_bounds_and_roles(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 requires bounded roles and a real validation reserve."""

    common = {
        "schedule": _schedule(),
        "transport": ScriptedTransport(),
        "checkpoint_path": tmp_path / "checkpoint.json",
        "owner": {},
    }
    with pytest.raises(exp.CaptureRunnerError, match="invalid_capture_deadline"):
        exp.ResumableCaptureRunner(**common, roles=("fit",), validation_reserve_s=1, hard_limit_s=1)
    with pytest.raises(exp.CaptureRunnerError, match="bounded_roles_invalid"):
        exp.ResumableCaptureRunner(**common, roles=("online",))


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not-json", "checkpoint_invalid"),
        ("[]", "checkpoint_invalid"),
        (
            json.dumps({"schedule_sha256": exp.canonical_hash(_schedule()), "rows": {}}),
            "checkpoint_rows_invalid",
        ),
        (
            json.dumps({"schedule_sha256": exp.canonical_hash(_schedule()), "rows": []}),
            None,
        ),
    ],
)
def test_checkpoint_parser_is_fail_closed(
    tmp_path: Path, payload: str, expected: str | None
) -> None:
    """SCENARIO-VERIFY-7548-RESUME accepts only its exact checkpoint shape."""

    path = tmp_path / "checkpoint.json"
    path.write_text(payload, encoding="utf-8")
    runner = exp.ResumableCaptureRunner(
        schedule=_schedule(),
        transport=ScriptedTransport(),
        checkpoint_path=path,
        roles=("fit", "development"),
        owner={},
    )
    if expected is None:
        assert runner._load_checkpoint() == []
    else:
        with pytest.raises(exp.CaptureRunnerError, match=expected):
            runner._load_checkpoint()


def test_checkpoint_rejects_partial_native_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7548-RESUME never adopts a partial group."""

    schedule = _schedule()
    selected = schedule
    row = _native_row(schedule[0], 0)
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps({"schedule_sha256": exp.canonical_hash(selected), "rows": [row]}),
        encoding="utf-8",
    )
    runner = exp.ResumableCaptureRunner(
        schedule=schedule,
        transport=ScriptedTransport(),
        checkpoint_path=path,
        roles=("fit", "development"),
        owner={},
    )

    with pytest.raises(exp.CaptureRunnerError, match="checkpoint_partial_or_changed"):
        runner._load_checkpoint()


def test_independent_reducer_rejects_missing_counts_and_receipts(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 requires both raw receipts and their role contract."""

    schedule = _schedule()
    rows = [_native_row(row, index) for index, row in enumerate(schedule)]
    schedule_receipt = exp.write_jsonl_sidecar(tmp_path / "schedule.jsonl", schedule, root=tmp_path)
    rows_receipt = exp.write_jsonl_sidecar(tmp_path / "rows.jsonl", rows, root=tmp_path)

    assert exp.independent_reduce({}, root=tmp_path) == {
        "passed": False,
        "error": "sidecar_receipt_invalid",
    }
    assert exp.independent_reduce(
        {"capture_schedule": schedule_receipt, "qualification_rows": rows_receipt},
        root=tmp_path,
    ) == {"passed": False, "error": "sidecar_receipt_invalid"}


def test_main_independent_reader_reports_raw_reduction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7548-E2E exposes the independent fresh-process mode."""

    schedule = _schedule()
    rows = [_native_row(row, index) for index, row in enumerate(schedule)]
    schedule_receipt = exp.write_jsonl_sidecar(tmp_path / "schedule.jsonl", schedule, root=tmp_path)
    rows_receipt = exp.write_jsonl_sidecar(tmp_path / "rows.jsonl", rows, root=tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "capture_schedule": schedule_receipt,
                "qualification_rows": rows_receipt,
                "qualification_role_counts": ROLE_COUNTS,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)

    assert exp.main(["--date", exp.RUN_DATE, "--independent-reduce", candidate.name]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_validation_helpers_cover_terminal_and_phase_receipts() -> None:
    """REQ-VERIFY-7548 distinguishes affected checks from terminal replay."""

    started = exp.time.monotonic()
    span = exp._phase_span("fixture", started, started, 1, "done")

    assert exp._required_validation_passed(_passing_receipts(), include_terminal=False)
    assert exp._required_validation_passed(
        _passing_receipts(include_terminal=True), include_terminal=True
    )
    assert span["phase"] == "fixture"
    assert span["completed_units"] == 1


def test_raw_reducer_rejects_non_numeric_logit() -> None:
    """REQ-VERIFY-7548 requires finite numeric logits."""

    schedule = _schedule()
    rows = [_native_row(row, index) for index, row in enumerate(schedule)]
    rows[0]["full_logits_by_option_id"] = {
        "supported": object(),
        "contains_unsupported": 0.0,
    }

    assert (
        "finite_logits"
        in exp.reduce_capture_rows(rows, schedule, expected_role_counts=ROLE_COUNTS)[
            "failed_checks"
        ]
    )


def test_runner_stops_before_validation_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7548 reserves 900 seconds before starting a group."""

    ticks = iter((0.0, 2.0, 3.0))
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    result = exp.ResumableCaptureRunner(
        schedule=_schedule(),
        transport=ScriptedTransport(),
        checkpoint_path=tmp_path / "checkpoint.json",
        roles=("fit",),
        validation_reserve_s=900.0,
        hard_limit_s=901.0,
        owner={},
    ).run()

    assert result["status"] == "partial"
    assert result["error"] == "validation_reserve_reached"


def test_runner_discards_group_that_fails_native_custody(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 checkpoints no malformed group."""

    class BadTransport(ScriptedTransport):
        def score(self, request: exp.JsonDict, owner: exp.JsonDict) -> exp.JsonDict:
            row = super().score(request, owner)
            row["state_reset"] = False
            return row

    result = exp.ResumableCaptureRunner(
        schedule=_schedule(),
        transport=BadTransport(),
        checkpoint_path=tmp_path / "checkpoint.json",
        roles=("fit",),
        owner={},
    ).run()

    assert result["status"] == "partial"
    assert result["partial_group_discarded"] == 1
    assert result["error"].startswith("group_custody_failed:")


def test_object_loader_returns_empty_for_invalid_or_non_object(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 readers fail closed on malformed JSON."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")

    assert exp._load_object(missing) == {}
    assert exp._load_object(malformed) == {}


def test_protocol_authentication_reports_missing_shards_and_receipts(tmp_path: Path) -> None:
    """REQ-VERIFY-7548 names malformed sealed protocol operands."""

    source = json.loads((exp.REPO_ROOT / exp.UPSTREAM_PROTOCOL_PATH).read_text())
    source["sealed_shards"] = {
        "predictor": "invalid",
        "fit_labels": {
            "path": "missing.jsonl",
            "bytes": 1,
            "sha256": "sha256:missing",
        },
    }
    source["validation_receipts"] = []
    path = tmp_path / exp.UPSTREAM_PROTOCOL_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(source), encoding="utf-8")

    checks, _context = exp.authenticate_protocol(tmp_path)
    by_check = {row["check"]: row for row in checks}

    assert by_check["sealed_shard_hashes"]["passed"] is False
    assert "required_shards_missing" in by_check["sealed_shard_hashes"]["observed"]
    assert by_check["qualified_terminal_checks"]["passed"] is False


def test_artifact_validator_recomputes_summary_principles_checksum_and_raw() -> None:
    """REQ-VERIFY-7548 validates its own derived evidence."""

    summary = exp.build_artifact_for_test(gpu_capacity_observed_score=1)
    summary["gate_check_summary"] = {}
    summary["reproducibility_checksum"] = exp.artifact_checksum(summary)
    principles = exp.build_artifact_for_test(gpu_capacity_observed_score=1)
    principles["field_principles"]["rows"] = ""
    principles["reproducibility_checksum"] = exp.artifact_checksum(principles)
    checksum = exp.build_artifact_for_test(gpu_capacity_observed_score=1)
    checksum["rows"][0]["complete_count"] = 0
    raw = exp.build_artifact_for_test(gpu_capacity_observed_score=1)
    raw["capture_schedule"] = {"path": "missing"}
    raw["reproducibility_checksum"] = exp.artifact_checksum(raw)

    assert "gate_check_summary_mismatch" in exp.validate_artifact(summary, require_validation=False)
    assert "field_principles_empty" in exp.validate_artifact(principles, require_validation=False)
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        checksum, require_validation=False
    )
    assert "independent_raw_reduction_failed" in exp.validate_artifact(
        raw, require_validation=False
    )


def test_main_delegates_declared_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7548-E2E delegates the declared live mode once."""

    monkeypatch.setattr(exp, "run_experiment", lambda root, date: int(date != exp.RUN_DATE))

    assert exp.main(["--date", exp.RUN_DATE]) == 0
