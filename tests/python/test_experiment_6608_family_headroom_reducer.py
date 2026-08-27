"""Focused tests for the Exp6608 independent family reducer.

Spec: REQ-REPORT-6608 and SCENARIO-REPORT-6608-INDEPENDENT-REPLAY through
SCENARIO-REPORT-6608-ATTACKS-AND-ATOMIC.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6604_exact_two_level_plan_corpus as exp6604
from carnot import experiment_6608_family_headroom_reducer as mod


REPO = Path(__file__).resolve().parents[2]


def _fixture() -> dict[str, object]:
    tasks = exp6604.generate_plan_tasks()
    selected = [tasks[0], tasks[36]]
    rows = exp6604._plan_fixture_rows(selected)  # noqa: SLF001
    payload: dict[str, object] = {
        "schema": "carnot.experiment_6604.exact_two_level_plan_corpus.v1",
        "status": "complete",
        "honest_verdict": "complete: synthetic fixture",
        "verdict_class": "null",
        "headroom_fixture_ready_score": 1.0,
        "plan_fixture_rows": rows,
        "fixture_and_split_receipts": exp6604._fixture_receipts(selected),  # noqa: SLF001
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def _registry(*, seeds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict[str, dict[str, object]]:
    return {
        "qwen36": {
            "path": "results/qwen.json",
            "repository_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_name": "Qwen3.6-35B-A3B",
            "seed_schedule": seeds,
            "ready_field": "qwen_headroom_ready_score",
        }
    }


def _identity(config: dict[str, object], row_count: int) -> tuple[dict, dict]:
    repository_id = str(config["repository_id"])
    model_hash = "sha256:" + "a" * 64
    template_hash = "sha256:" + "b" * 64
    identity = {
        "MODEL_SPECS": [
            {
                "name": config["model_name"],
                "hf_id": repository_id,
                "repository_id": repository_id,
                "model_path": "/models/mandated.gguf",
                "quantization": "Q4_K_M",
                "headline_eligible": True,
            }
        ],
        "hub_id": repository_id,
        "model_path": "/models/mandated.gguf",
        "model_sha256": model_hash,
        "quantization": "Q4_K_M",
        "gguf_shards": [
            {
                "path": "/models/mandated.gguf",
                "sha256": model_hash,
                "byte_count": 1,
            }
        ],
        "embedded_tokenizer": {
            "source": "embedded_gguf",
            "loadable": True,
            "token_count": 10,
            "identity_sha256": "sha256:" + "c" * 64,
        },
        "embedded_chat_template": {
            "source": "tokenizer.chat_template",
            "present": True,
            "sha256": template_hash,
        },
        "llama_cpp": {"cuda_linked": True},
        "auto_tokenizer_used": False,
        "download_performed": False,
        "legacy_headline_row_count": 0,
    }
    session = {
        "session_id": "session-1",
        "pid": 12345,
        "repository_id": repository_id,
        "model_sha256": model_hash,
        "row_count": row_count,
        "owned_child": True,
        "cpu_fallback": False,
        "cuda_offload": True,
        "offloaded_layers": 1,
        "server_healthy": True,
        "shutdown_requested": True,
        "normal_shutdown": True,
        "worker_absent_after_exit": True,
        "port_closed": True,
        "memory_recovered": True,
        "signals_sent_to_unrelated_pids": [],
    }
    return identity, {"sessions": [session], "all_sessions_authentic": True}


def _raw_row(
    fixture_row: dict[str, object],
    seed: int,
    *,
    success: bool,
    config: dict[str, object],
) -> dict[str, object]:
    source = str(fixture_row["source_bytes"]).encode()
    task = json.loads(source)
    response = str(task["gold_witness"]) if success else "NOT_A_PLAN"
    raw = response.encode()
    exact = exp6604.IndependentExactExecutor().execute(task, response)
    failure = None if success else "syntax_failure"
    model_hash = "sha256:" + "a" * 64
    row: dict[str, object] = {
        "schema": "carnot.direct_plan_row.v1",
        "row_id": f"{fixture_row['task_id']}|seed-{seed}",
        "task_id": fixture_row["task_id"],
        "split": fixture_row["split"],
        "seed": seed,
        "task_sha256": mod.sha256_bytes(source),
        "task_source_bytes_b64": base64.b64encode(source).decode(),
        "prompt_bytes_b64": base64.b64encode(
            str(fixture_row["model_prompt_bytes"]).encode()
        ).decode(),
        "prompt_sha256": mod.sha256_bytes(str(fixture_row["model_prompt_bytes"]).encode()),
        "raw_response_bytes_b64": base64.b64encode(raw).decode(),
        "raw_response_byte_count": len(raw),
        "raw_response_sha256": mod.sha256_bytes(raw),
        "raw_recorded_before_parse": True,
        "parsed_plan": response,
        "parse_state": "parsed_canonical_candidate" if success else "syntax_invalid",
        "exact_executor_result": exact,
        "exact_executor_call_count": 1,
        "exact_success": success,
        "failure_class": failure,
        "charged_failure": not success,
        "failure_flags": {name: name == failure for name in mod.FAILURE_CLASSES},
        "finish_reason": "stop",
        "attempt_count": 1,
        "regeneration_count": 0,
        "response_regenerated": False,
        "model_process": {
            "session_id": "session-1",
            "pid": 12345,
            "repository_id": config["repository_id"],
            "model_sha256": model_hash,
            "owned_child": True,
            "cpu_fallback": False,
            "cuda_offload": True,
            "offloaded_layers": 1,
            "tokenizer_source": "embedded_gguf",
            "chat_template_sha256": "sha256:" + "b" * 64,
        },
    }
    row["row_hash"] = mod.row_hash(row)
    return row


def _source(
    fixture: dict[str, object],
    config: dict[str, object],
    *,
    held_successes: int,
    blocked: bool = False,
) -> dict[str, object]:
    if blocked:
        payload: dict[str, object] = {
            "status": "blocked_gpu_ownership",
            "honest_verdict": "blocked_gpu_ownership: synthetic",
            "verdict_class": "blocked",
            "gate_check_summary": {
                "all_passed": False,
                "failed_condition": "gpu_ownership",
                "expected": True,
                "observed": False,
            },
            "per_unit_rows": [],
            "reproducibility_checksum": "",
        }
        payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
        return payload

    rows = []
    seeds = tuple(config["seed_schedule"])
    for fixture_row in fixture["plan_fixture_rows"]:
        for index, seed in enumerate(seeds):
            success = fixture_row["split"] == "calibration" or index < held_successes
            rows.append(
                _raw_row(
                    fixture_row,
                    int(seed),
                    success=success,
                    config=config,
                )
            )
    identity, processes = _identity(config, len(rows))
    payload = {
        "status": "complete",
        "honest_verdict": "complete: synthetic baseline",
        "verdict_class": "null",
        "gate_check_summary": {"all_passed": True},
        "per_unit_rows": rows,
        "model_spec_and_identity": identity,
        "gpu_process_receipts": processes,
        "family_headroom_summary": mod.source_family_summary(rows),
        str(config["ready_field"]): 1.0,
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def _all_families_blocked() -> dict[str, dict[str, object]]:
    """Give every family a blocked upstream artifact.

    Reading the real `results/` files bound these tests to whatever state the
    upstream experiments happened to be in, so repairing an upstream turned them
    red. The blocked path is what they mean to exercise, so state it directly.
    """

    return {
        family: {"status": "blocked_upstream_fixture", "verdict_class": "blocked"}
        for family in mod.FAMILY_REGISTRY
    }


def _report(held_successes: int = 1) -> dict[str, object]:
    fixture = _fixture()
    registry = _registry()
    source = _source(fixture, registry["qwen36"], held_successes=held_successes)
    return mod.build_report(
        REPO,
        "20260825",
        fixture_artifact=fixture,
        sources={"qwen36": source},
        family_registry=registry,
        tests_run=[],
    )


# REQ-REPORT-6608 and SCENARIO-REPORT-6608-GATE-OWNERSHIP.
def test_spec_and_mandated_registry_are_explicit() -> None:
    spec = (REPO / mod.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    for anchor in (
        "REQ-REPORT-6608-REPLAY",
        "REQ-REPORT-6608-PARTIAL",
        "REQ-REPORT-6608-HEADROOM",
        "REQ-REPORT-6608-NO-CHERRY-PICK",
        "REQ-REPORT-6608-FREEZE",
        "REQ-REPORT-6608-GATE",
        "REQ-REPORT-6608-ATTACKS",
        "REQ-REPORT-6608-ATOMIC",
        "SCENARIO-REPORT-6608-GATE-OWNERSHIP",
    ):
        assert anchor in spec
    assert [row["repository_id"] for row in mod.FAMILY_REGISTRY.values()] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]


# REQ-REPORT-6608-PARTIAL and SCENARIO-REPORT-6608-PARTIAL-PRESERVATION.
def test_missing_and_blocked_families_keep_every_expected_key() -> None:
    fixture = _fixture()
    registry = _registry(seeds=(1, 2))
    registry["gemma31"] = {
        **registry["qwen36"],
        "path": "results/gemma.json",
        "repository_id": "unsloth/gemma-4-31B-it-GGUF",
        "model_name": "Gemma4-31B-it",
        "ready_field": "gemma31_headroom_ready_score",
    }
    blocked = _source(fixture, registry["qwen36"], held_successes=0, blocked=True)
    reduction = mod.reduce_sources(
        fixture,
        {"qwen36": blocked, "gemma31": None},
        family_registry=registry,
    )
    assert len(reduction["per_unit_rows"]) == 8
    assert {row["replay_state"] for row in reduction["per_unit_rows"]} == {
        "blocked_upstream",
        "missing_artifact",
    }
    assert {row["family"] for row in reduction["per_unit_rows"]} == {
        "qwen36",
        "gemma31",
    }
    assert reduction["eligible_model_specs"] == []
    assert reduction["headroom_benchmark_ready_score"] == 0.0


# REQ-REPORT-6608-HEADROOM and SCENARIO-REPORT-6608-HEADROOM-BOUNDARIES.
@pytest.mark.parametrize(
    ("held_successes", "eligible"),
    [(1, True), (4, True), (5, False)],
)
def test_exact_recomputation_uses_closed_headroom_interval(
    held_successes: int, eligible: bool
) -> None:
    report = _report(held_successes)
    headroom = report["family_headroom_rows"][0]
    assert headroom["held"]["exact_success_rate"] == held_successes / 5
    assert headroom["eligible"] is eligible
    assert report["headroom_benchmark_ready_score"] == float(eligible)
    assert report["verdict_class"] == "null"


# REQ-REPORT-6608-COMPLETENESS and SCENARIO-REPORT-6608-INDEPENDENT-REPLAY.
def test_reported_aggregate_disagreement_fails_family_closed() -> None:
    fixture = _fixture()
    registry = _registry()
    source = _source(fixture, registry["qwen36"], held_successes=1)
    source["family_headroom_summary"]["held"]["exact_success_count"] = 5
    source["reproducibility_checksum"] = mod.artifact_checksum(source)
    reduction = mod.reduce_sources(
        fixture,
        {"qwen36": source},
        family_registry=registry,
    )
    family = reduction["family_replay_rows"][0]
    assert family["reported_aggregate_matches_recomputed"] is False
    assert family["source_complete"] is False
    assert reduction["eligible_model_specs"] == []


# REQ-REPORT-6608-NO-CHERRY-PICK and SCENARIO-REPORT-6608-NO-UNIT-SELECTION.
def test_frozen_hashes_cover_full_held_split_without_outcomes() -> None:
    report = _report(1)
    frozen = report["frozen_held_unit_hashes"]
    assert frozen["selection_policy"] == "full_held_split_without_outcome_selection"
    assert len(frozen["task_hashes"]) == 1
    assert len(frozen["selected_family_row_hashes"]) == 5
    assert all("exact_success" not in row for row in frozen["selected_family_row_hashes"])
    assert frozen["immutable"] is True


# REQ-REPORT-6608-GATE and SCENARIO-REPORT-6608-GATE-OWNERSHIP.
def test_complete_no_headroom_is_null_not_blocked() -> None:
    report = _report(5)
    assert report["status"] == "complete_no_family_headroom"
    assert report["honest_verdict"].startswith("complete:")
    assert report["verdict_class"] == "null"
    assert report["headroom_benchmark_ready_score"] == 0.0


# REQ-REPORT-6608-ATTACKS and SCENARIO-REPORT-6608-ATTACKS-AND-ATOMIC.
@pytest.mark.parametrize("attack_id", mod.REQUIRED_ATTACK_IDS)
def test_required_attacks_fail_closed_after_self_hash_repair(attack_id: str) -> None:
    report = _report(1)
    attacked = mod.attack_candidate(report, attack_id)
    attacked["reproducibility_checksum"] = mod.artifact_checksum(attacked)
    reduction = mod.readiness_reducer(attacked)
    assert reduction["headroom_benchmark_ready_score"] == 0.0
    assert reduction["checks"][mod.ATTACK_CHECKS[attack_id]] is False


# REQ-REPORT-6608-ATOMIC and SCENARIO-REPORT-6608-ATTACKS-AND-ATOMIC.
def test_required_fields_validate_and_atomic_writer_replaces(tmp_path: Path) -> None:
    report = _report(1)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_report(report) == []
    target = tmp_path / "result.json"
    receipt = mod.atomic_write_report(target, report)
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert receipt["atomic_replace"] is True
    assert receipt["directory_fsync"] is True
    assert list(tmp_path.glob(".exp6608-*")) == []


# REQ-REPORT-6608-PARTIAL and SCENARIO-REPORT-6608-GATE-OWNERSHIP.
def test_checked_in_blocked_upstreams_emit_648_rows_and_named_gate() -> None:
    # Inject three absent families rather than reading `results/`. Those files are
    # evidence and change when an upstream is repaired, so binding this test to
    # their current contents made it assert that the repository stays broken.
    report = mod.build_report(
        REPO,
        "20260825",
        sources=_all_families_blocked(),
        tests_run=[],
    )
    assert len(report["per_unit_rows"]) == 648
    assert {row["replay_state"] for row in report["per_unit_rows"]} == {"blocked_upstream"}
    assert report["eligible_model_specs"] == []
    assert report["headroom_benchmark_ready_score"] == 0.0
    assert report["status"].startswith("blocked_")
    assert report["verdict_class"] == "blocked"
    assert report["gate_check_summary"]["failed_checks"]
    assert mod.validate_report(report) == []

    # Load the real upstream files too, but assert only what holds whatever state
    # they are in. The reducer must always emit the full matrix and legal states.
    # It cannot assert a clean report today: with a repaired upstream the writer
    # reports `reducer_checks_failed:row_replay`, an open defect recorded in
    # ops/known-issues.md 2026-08-27 (reduced rows drop `finish_reason`).
    on_disk = mod.build_report(REPO, "20260825", tests_run=[])
    assert len(on_disk["per_unit_rows"]) == 648
    assert {row["replay_state"] for row in on_disk["per_unit_rows"]} <= {
        "replayed",
        "blocked_upstream",
        "missing_artifact",
        "missing_row",
    }


# REQ-REPORT-6608-REPLAY and SCENARIO-REPORT-6608-INDEPENDENT-REPLAY.
def test_raw_exact_and_failure_mutation_is_not_trusted() -> None:
    report = _report(1)
    attacked = deepcopy(report)
    failed = next(row for row in attacked["per_unit_rows"] if row["failure_class"] is not None)
    failed["failure_class"] = None
    failed["exact_success"] = True
    failed["row_hash"] = mod.row_hash(failed)
    attacked["reproducibility_checksum"] = mod.artifact_checksum(attacked)
    reduction = mod.readiness_reducer(attacked)
    assert reduction["checks"]["row_replay"] is False
    assert reduction["headroom_benchmark_ready_score"] == 0.0


# REQ-REPORT-6608-PRECONDITIONS and REQ-REPORT-6608-REPLAY.
def test_hash_unwrap_and_malformed_input_helpers_fail_closed(tmp_path: Path) -> None:
    assert mod.sha256_file(tmp_path / "absent") is None
    assert mod.unwrap_value({"principle": "why", "value": {"value": 1}}) == 1
    fixture = _fixture()
    fixture["plan_fixture_rows"][0]["source_bytes"] = "{"  # type: ignore[index]
    fixture["reproducibility_checksum"] = mod.artifact_checksum(fixture)
    assert mod._fixture_contract(fixture)["valid"] is False
    identity = mod._identity_replay(
        _registry()["qwen36"],
        {"model_spec_and_identity": [], "gpu_process_receipts": []},
        10,
    )
    assert identity["model_identity_valid"] is False
    assert identity["process_identity_valid"] is False


# REQ-REPORT-6608-REPLAY and SCENARIO-REPORT-6608-INDEPENDENT-REPLAY.
@pytest.mark.parametrize(
    ("raw", "decoded", "row", "exact", "expected"),
    [
        (b"", "", {"failure_class": "timeout"}, {}, "timeout"),
        (b"x", "x", {"finish_reason": "length"}, {}, "invalid_generation"),
        (b"", "", {}, {}, "invalid_generation"),
        (b"sorry", "sorry", {}, {}, "refusal"),
        (b"x", "x", {}, {"reason": "syntax_error"}, "syntax_failure"),
        (b"x", "x", {}, {"reason": "ordering_violation"}, "semantic_failure"),
        (b"x", "x", {}, {"reason": "unmet_goal"}, "unmet_goal"),
        (b"x", "x", {}, {"reason": "unknown"}, "invalid_generation"),
    ],
)
def test_failure_replay_has_a_closed_classification(
    raw: bytes,
    decoded: str | None,
    row: dict[str, object],
    exact: dict[str, object],
    expected: str,
) -> None:
    assert mod._failure_from_replay(raw, decoded, row, exact) == expected


# REQ-REPORT-6608-COMPLETENESS and REQ-REPORT-6608-PARTIAL.
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bad_base64", "bad_utf8"])
def test_source_row_matrix_and_raw_byte_corruption_fail_closed(mutation: str) -> None:
    fixture = _fixture()
    registry = _registry()
    source = _source(fixture, registry["qwen36"], held_successes=1)
    if mutation == "missing":
        source["per_unit_rows"].pop()
    elif mutation == "duplicate":
        source["per_unit_rows"].append(deepcopy(source["per_unit_rows"][0]))
    elif mutation == "bad_base64":
        source["per_unit_rows"][0]["raw_response_bytes_b64"] = "***"
        source["per_unit_rows"][0]["row_hash"] = mod.row_hash(source["per_unit_rows"][0])
    else:
        source["per_unit_rows"][0]["raw_response_bytes_b64"] = base64.b64encode(b"\xff").decode()
        source["per_unit_rows"][0]["raw_response_sha256"] = mod.sha256_bytes(b"\xff")
        source["per_unit_rows"][0]["raw_response_byte_count"] = 1
        source["per_unit_rows"][0]["row_hash"] = mod.row_hash(source["per_unit_rows"][0])
    source["family_headroom_summary"] = mod.source_family_summary(source["per_unit_rows"])
    source["reproducibility_checksum"] = mod.artifact_checksum(source)
    reduction = mod.reduce_sources(
        fixture,
        {"qwen36": source},
        family_registry=registry,
    )
    family = reduction["family_replay_rows"][0]
    assert family["source_complete"] is False
    assert reduction["headroom_benchmark_ready_score"] == 0.0


# REQ-REPORT-6608-PRECONDITIONS and REQ-REPORT-6608-ATOMIC.
def test_loader_and_reduced_row_defenses_return_false(tmp_path: Path) -> None:
    assert mod._load_json(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._load_json(malformed) == {}
    report = _report(1)
    row = report["per_unit_rows"][0]
    task_map = {item["task_id"]: item for item in report["preconditions_checked"]["fixture_tasks"]}
    bad_hash = deepcopy(row)
    bad_hash["row_hash"] = "sha256:" + "0" * 64
    assert mod._reduced_row_valid(bad_hash, task_map) is False
    bad_state = deepcopy(row)
    bad_state["replay_state"] = "invalid_row"
    bad_state["row_hash"] = mod.row_hash(bad_state)
    assert mod._reduced_row_valid(bad_state, task_map) is False
    assert mod._reduced_row_valid(row, {}) is False
    bad_bytes = deepcopy(row)
    bad_bytes["task_source_bytes_b64"] = "***"
    bad_bytes["row_hash"] = mod.row_hash(bad_bytes)
    assert mod._reduced_row_valid(bad_bytes, task_map) is False
    with pytest.raises(ValueError, match="unknown attack"):
        mod.attack_candidate(report, "unknown")


# REQ-REPORT-6608-GATE and REQ-REPORT-6608-ATOMIC.
def test_validator_names_each_terminal_contract_failure(tmp_path: Path) -> None:
    eligible = _report(1)
    missing = deepcopy(eligible)
    del missing["status"]
    assert mod.validate_report(missing) == ["missing_required_fields:status"]

    invalid = deepcopy(eligible)
    invalid["inference_substrate"] = "wrong"
    invalid["verifier_is_oracle"] = False
    invalid["verdict_class"] = "wrong"
    invalid["field_provenance"] = {}
    invalid["per_unit_rows"][0]["row_hash"] = "wrong"
    invalid["status"] = "blocked_wrong"
    invalid["attack_rows"] = []
    invalid["duration_s"] = -1.0
    invalid["reproducibility_checksum"] = "wrong"
    errors = mod.validate_report(invalid)
    assert "inference_substrate_mismatch" in errors
    assert "verifier_is_oracle_mismatch" in errors
    assert "verdict_class_invalid" in errors
    assert "field_provenance_mismatch" in errors
    assert any(error.startswith("reducer_checks_failed:") for error in errors)
    assert "eligible_disposition_mismatch" in errors
    assert "attack_rows_failed" in errors
    assert "duration_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors

    no_headroom = _report(5)
    no_headroom["status"] = "complete_wrong"
    no_headroom["verdict_class"] = "blocked"
    no_headroom["reproducibility_checksum"] = mod.artifact_checksum(no_headroom)
    assert "no_headroom_disposition_mismatch" in mod.validate_report(no_headroom)

    blocked = mod.build_report(
        REPO,
        "20260825",
        sources=_all_families_blocked(),
        tests_run=[],
    )
    blocked["status"] = "complete_wrong"
    blocked["verdict_class"] = "null"
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert "blocked_disposition_mismatch" in mod.validate_report(blocked)

    invalid_writer = deepcopy(eligible)
    invalid_writer["reproducibility_checksum"] = "wrong"
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        mod.atomic_write_report(tmp_path / "invalid.json", invalid_writer)
