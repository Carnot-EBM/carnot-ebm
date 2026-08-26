"""Focused checks for the V577 execution contract.

Spec refs: REQ-REPORT-6616, REQ-REPORT-6616-TASKS,
REQ-REPORT-6616-GATES, REQ-REPORT-6616-PRIOR-FAILURES,
REQ-REPORT-6616-PHASE-RECEIPT, REQ-REPORT-6616-ACCELERATOR-RECEIPT,
REQ-REPORT-6616-MODELS, REQ-REPORT-6616-VERDICTS,
REQ-REPORT-6616-PROTECTION, REQ-REPORT-6616-ATTACKS,
REQ-REPORT-6616-READY, SCENARIO-REPORT-6616-EXACT-TASKS,
SCENARIO-REPORT-6616-GATE-OWNERS,
SCENARIO-REPORT-6616-PRIOR-FAILURES,
SCENARIO-REPORT-6616-RECEIPTS,
SCENARIO-REPORT-6616-CLOSED-VERDICTS, SCENARIO-REPORT-6616-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6616_v577_execution_contract as mod


REPO = Path(__file__).resolve().parents[2]


def _principled_prompt(fields: list[str], models: tuple[str, ...] = ()) -> str:
    lines = ["MODEL_SPECS:", *[f"  - {model}" for model in models]]
    lines.append("REQUIRED ARTIFACT FIELDS:")
    for field in fields:
        lines.extend((f"  {field}:", f'    principle: "Principle for {field}."'))
    lines.append("Set inference_substrate=test_no_llm")
    return "\n".join(lines)


def _complete_roadmap() -> dict[str, object]:
    owner_fields = {
        gate["upstream"]: gate["artifact_field"]
        for gates in mod.EXPECTED_GATES.values()
        for gate in gates
    }
    tasks = []
    for task_id, contract in mod.EXPECTED_TASKS.items():
        models = mod.REQUIRED_MODELS_BY_TASK.get(task_id, ())
        fields = ["status", "honest_verdict", "verdict_class", "gate_check_summary"]
        if task_id in owner_fields:
            fields.append(owner_fields[task_id])
        tasks.append(
            {
                "id": task_id,
                "milestone": mod.MILESTONE,
                "deliverable": contract["deliverable"],
                "title": task_id,
                "prompt": _principled_prompt(fields, models),
                "requires_gpu": task_id in mod.GPU_TASK_IDS,
                "model": "opus",
                "gated_on": deepcopy(mod.EXPECTED_GATES.get(task_id, [])),
            }
        )
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": "test",
        "milestone_doc": mod.ROADMAP_DOCUMENT.as_posix(),
        "tasks": tasks,
    }


def _valid_phase_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "task_id": "exp6617-gpu-lease-phase-receipts",
        "phase_name": "validating",
        "state": "terminal_complete",
        "monotonic_started_s": 10.0,
        "monotonic_ended_s": 12.0,
        "process_identity": {"pid": 42, "pid_start_time": 9.0},
        "resource_owner": "exp6617-gpu-lease-phase-receipts",
        "input_hashes": {"input": "sha256:abc"},
        "output_hashes": {"output": "sha256:def"},
        "heartbeat_monotonic_s": 11.0,
        "terminal_reason": "complete",
    }
    receipt["checksum"] = mod.receipt_checksum(receipt)
    return receipt


def test_req_report_6616_spec_owns_every_contract_anchor() -> None:
    """REQ-REPORT-6616: the reporting spec owns the full contract."""

    text = (REPO / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-REPORT-6616", 1)[1]
    for marker in (
        "SCENARIO-REPORT-6616-EXACT-TASKS",
        "SCENARIO-REPORT-6616-GATE-OWNERS",
        "SCENARIO-REPORT-6616-PRIOR-FAILURES",
        "SCENARIO-REPORT-6616-RECEIPTS",
        "SCENARIO-REPORT-6616-CLOSED-VERDICTS",
        "SCENARIO-REPORT-6616-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    assert set(mod.REQUIRED_FIELDS) <= set(mod.FIELD_PRINCIPLES)


def test_scenario_report_6616_exact_tasks_fail_closed_on_active_roadmap() -> None:
    """SCENARIO-REPORT-6616-EXACT-TASKS: missing YAML tasks remain visible."""

    roadmap = yaml.safe_load((REPO / mod.ACTIVE_ROADMAP).read_text(encoding="utf-8"))
    audit = mod.validate_roadmap_contract(roadmap, retired_ids=set())
    assert len(audit["task_contract_rows"]) == 13
    assert {row["task_id"] for row in audit["task_contract_rows"]} == set(mod.EXPECTED_TASKS)
    missing = [row["task_id"] for row in audit["task_contract_rows"] if not row["yaml_present"]]
    assert missing == list(mod.EXPECTED_TASKS)[3:]
    assert any(error["check"] == "exact_task_set" for error in audit["errors"])
    assert not audit["passed"]


def test_scenario_report_6616_complete_contract_and_gate_ownership() -> None:
    """SCENARIO-REPORT-6616-GATE-OWNERS: a complete exact graph validates."""

    roadmap = _complete_roadmap()
    audit = mod.validate_roadmap_contract(roadmap, retired_ids=set())
    assert audit["passed"]
    assert len(audit["gate_owner_rows"]) == 9
    assert all(row["passed"] for row in audit["gate_owner_rows"])

    bad = deepcopy(roadmap)
    bad["tasks"][1]["gated_on"][0]["artifact_field"] = "misspelled_ready_score"
    assert not mod.validate_roadmap_contract(bad, retired_ids=set())["passed"]

    forward = deepcopy(roadmap)
    forward["tasks"][0]["gated_on"] = [
        {
            "upstream": list(mod.EXPECTED_TASKS)[1],
            "artifact_field": "later",
            "op": "==",
            "value": 1.0,
        }
    ]
    assert not mod.validate_roadmap_contract(forward, retired_ids=set())["passed"]


def test_req_report_6616_roadmap_attacks_are_rejected() -> None:
    """REQ-REPORT-6616-ATTACKS: structural attacks fail closed."""

    base = _complete_roadmap()
    mutations = []

    duplicate_id = deepcopy(base)
    duplicate_id["tasks"][-1]["id"] = duplicate_id["tasks"][0]["id"]
    mutations.append(duplicate_id)

    duplicate_deliverable = deepcopy(base)
    duplicate_deliverable["tasks"][-1]["deliverable"] = duplicate_deliverable["tasks"][0][
        "deliverable"
    ]
    mutations.append(duplicate_deliverable)

    stale = deepcopy(base)
    stale["tasks"][0]["milestone"] = "2026.08.576"
    mutations.append(stale)

    missing_principle = deepcopy(base)
    missing_principle["tasks"][0]["prompt"] = missing_principle["tasks"][0]["prompt"].replace(
        '    principle: "Principle for status."', ""
    )
    mutations.append(missing_principle)

    unsupported_op = deepcopy(base)
    unsupported_op["tasks"][1]["gated_on"][0]["op"] = "contains"
    mutations.append(unsupported_op)

    retired = deepcopy(base)
    retired_owner = retired["tasks"][0]["id"]
    assert all(
        not mod.validate_roadmap_contract(
            candidate, retired_ids={retired_owner} if candidate is retired else set()
        )["passed"]
        for candidate in [*mutations, retired]
    )


def test_req_report_6616_model_policy_is_exact() -> None:
    """REQ-REPORT-6616-MODELS: mandated families cannot be substituted."""

    base = _complete_roadmap()
    audit = mod.validate_roadmap_contract(base, retired_ids=set())
    assert all(row["passed"] for row in audit["model_policy_receipts"])

    bad = deepcopy(base)
    qwen_task = next(task for task in bad["tasks"] if task["id"].startswith("exp6619-"))
    qwen_task["prompt"] = qwen_task["prompt"].replace(mod.QWEN_MODEL, mod.LEGACY_MODELS[0])
    failed = mod.validate_roadmap_contract(bad, retired_ids=set())
    assert not failed["passed"]
    assert any(not row["passed"] for row in failed["model_policy_receipts"])


def test_scenario_report_6616_prior_failures_match_v576_verdicts() -> None:
    """SCENARIO-REPORT-6616-PRIOR-FAILURES: prior verdicts stay exact."""

    roadmap = yaml.safe_load((REPO / mod.ACTIVE_ROADMAP).read_text(encoding="utf-8"))
    rows, errors = mod.reconcile_prior_failures(REPO, roadmap["tasks"])
    assert len([row for row in rows if row["declared_by_task"]]) == 5
    declared = [row for row in rows if row["declared_by_task"]]
    assert all(row["verdict_matches"] for row in declared)
    assert all(row["retire_if_same_verdict"] for row in declared)
    assert all(row["changed_condition_concrete"] for row in declared)
    by_number = {row["experiment_number"]: row for row in rows}
    assert by_number[6610]["source_state"] == "missing"
    assert by_number[6614]["source_verdict_class"] == "blocked_tests"
    assert by_number[6614]["contract_disposition"] == "invalid_verdict_class"
    assert errors == []

    bad = deepcopy(roadmap["tasks"])
    bad[1]["prior_failures"][0]["verdict"] = "null"
    _, bad_errors = mod.reconcile_prior_failures(REPO, bad)
    assert any(error["check"] == "prior_failure_verdict" for error in bad_errors)


def test_scenario_report_6616_receipts_reject_time_pid_and_tamper() -> None:
    """SCENARIO-REPORT-6616-RECEIPTS: time, PID, and checksums are stable."""

    phase = _valid_phase_receipt()
    assert mod.validate_phase_receipt(phase) == []

    reversed_time = deepcopy(phase)
    reversed_time["monotonic_ended_s"] = 8.0
    reversed_time["checksum"] = mod.receipt_checksum(reversed_time)
    assert "phase timestamp reversal" in mod.validate_phase_receipt(reversed_time)

    reused_pid = deepcopy(phase)
    reused_pid["process_identity"]["pid_start_time"] = 10.5
    reused_pid["checksum"] = mod.receipt_checksum(reused_pid)
    assert "PID start time is after phase start" in mod.validate_phase_receipt(reused_pid)

    tampered = deepcopy(phase)
    tampered["task_id"] = "changed"
    assert "checksum mismatch" in mod.validate_phase_receipt(tampered)

    accelerator = {
        **phase,
        "device_uuid": "GPU-test",
        "pid_start_time": 9.0,
        "model_hash": "sha256:model",
        "vram_before_mib": 100,
        "vram_after_mib": 100,
        "offload_layers": 1,
        "unload_evidence": {"requested": True, "completed": True},
        "lease_token": "opaque-token",
    }
    accelerator["checksum"] = mod.receipt_checksum(accelerator)
    assert mod.validate_accelerator_receipt(accelerator) == []
    no_unload = deepcopy(accelerator)
    no_unload["unload_evidence"] = {"requested": True, "completed": False}
    no_unload["checksum"] = mod.receipt_checksum(no_unload)
    assert "accelerator unload is incomplete" in mod.validate_accelerator_receipt(no_unload)


def test_scenario_report_6616_blocked_artifact_is_complete_and_atomic(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6616-ATOMIC: a blocked contract is valid and atomic."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        run_date="20260826",
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 1.0}],
        duration_s=2.0,
    )
    assert artifact["status"].startswith("blocked_")
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["execution_contract_ready_score"] == 0.0
    assert artifact["gate_check_summary"]["failed_checks"]
    assert len(artifact["task_contract_rows"]) == 13
    assert len(artifact["attack_rows"]) == len(mod.ATTACK_IDS)
    assert all(row["fail_closed"] for row in artifact["attack_rows"])
    assert set(mod.REQUIRED_FIELDS) <= set(artifact["field_provenance"])
    mod.validate_artifact(artifact)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "mutated"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad)

    bad_score = deepcopy(artifact)
    bad_score["execution_contract_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="readiness"):
        mod.validate_artifact(bad_score)

    output = tmp_path / "nested" / "contract.json"
    mod.write_artifact_atomic(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert not list(output.parent.glob("*.tmp"))


def test_req_report_6616_pre_staged_source_and_protection_are_explicit() -> None:
    """REQ-REPORT-6616-PROTECTION: hashes and activated source are retained."""

    payload, receipt = mod.load_pre_staged_roadmap(REPO)
    assert payload["milestone"] == mod.MILESTONE
    assert receipt["source_state"] in {"working_tree", "historical_git_blob"}
    assert receipt["sha256"].startswith("sha256:")
    protected = mod.protected_file_receipts(REPO)
    assert protected["all_unchanged"]
    assert {row["path"] for row in protected["rows"]} == {
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    }


def test_req_report_6616_complete_synthetic_artifact_is_null() -> None:
    """REQ-REPORT-6616-READY: only the full exact contract opens readiness."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        run_date="20260826",
        roadmap_payload=_complete_roadmap(),
        roadmap_source={
            "path": "synthetic/research-roadmap-next.yaml",
            "source_state": "test_fixture",
            "sha256": "sha256:test",
        },
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 1.0}],
        duration_s=2.0,
    )
    assert artifact["execution_contract_ready_score"] == 1.0
    assert artifact["verdict_class"] == "null"
    assert artifact["status"] == "complete_execution_contract_ready"
    mod.validate_artifact(artifact)


def test_req_report_6616_defensive_schema_and_receipt_paths(tmp_path: Path) -> None:
    """REQ-REPORT-6616-TASKS: malformed inputs produce named failures."""

    assert mod.extract_required_fields("no required field block") == {}
    malformed = mod.validate_roadmap_contract(
        {"milestone": "2026.08.576", "tasks": "not-a-list"}, retired_ids=set()
    )
    assert {error["check"] for error in malformed["errors"]} >= {
        "roadmap_schema",
        "roadmap_milestone",
        "exact_task_set",
    }

    wrong_gpu = _complete_roadmap()
    wrong_gpu["tasks"][2]["requires_gpu"] = False
    assert any(
        error["check"] == "task_gpu_policy"
        for error in mod.validate_roadmap_contract(wrong_gpu, retired_ids=set())["errors"]
    )

    bad_prior = _complete_roadmap()
    bad_prior["tasks"][0]["prior_failures"] = [
        {
            "experiment_id": "",
            "verdict": "",
            "addressed_by": "retry",
            "retire_if_same_verdict": False,
        }
    ]
    assert any(
        error["check"] == "prior_failure_completeness"
        for error in mod.validate_roadmap_contract(bad_prior, retired_ids=set())["errors"]
    )

    assert mod.validate_phase_receipt({})[0].startswith("missing field:")
    phase = _valid_phase_receipt()
    phase["process_identity"] = "bad"
    phase["state"] = "running"
    phase["checksum"] = mod.receipt_checksum(phase)
    phase_errors = mod.validate_phase_receipt(phase)
    assert "invalid process identity" in phase_errors
    assert "nonterminal phase state" in phase_errors

    missing_accelerator = _valid_phase_receipt()
    assert any(
        error.startswith("missing accelerator field:")
        for error in mod.validate_accelerator_receipt(missing_accelerator)
    )
    accelerator = {
        **_valid_phase_receipt(),
        "device_uuid": "GPU-test",
        "pid_start_time": 8.0,
        "model_hash": "sha256:model",
        "vram_before_mib": 10,
        "vram_after_mib": 10,
        "offload_layers": -1,
        "unload_evidence": {"completed": True},
        "lease_token": "token",
    }
    accelerator["checksum"] = mod.receipt_checksum(accelerator)
    accelerator_errors = mod.validate_accelerator_receipt(accelerator)
    assert "accelerator PID start time differs from process identity" in accelerator_errors
    assert "invalid accelerator offload layer count" in accelerator_errors

    assert mod._experiment_number("not-an-experiment") is None
    wrapped = {"value": "blocked", "principle": "test"}
    assert mod._unwrap(wrapped) == "blocked"
    ordinary = {"value": "blocked", "extra": "ordinary"}
    assert mod._unwrap(ordinary) is ordinary
    bad_json = tmp_path / "list.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact root"):
        mod._read_json(bad_json)


def test_scenario_report_6616_failure_receipts_and_source_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6616-PRIOR-FAILURES: all failure paths retain evidence."""

    roadmap = yaml.safe_load((REPO / mod.ACTIVE_ROADMAP).read_text(encoding="utf-8"))
    bad = deepcopy(roadmap["tasks"])
    bad[1]["prior_failures"][0]["addressed_by"] = "retry"
    bad[1]["prior_failures"][0]["retire_if_same_verdict"] = False
    _, errors = mod.reconcile_prior_failures(REPO, bad)
    assert {error["check"] for error in errors} >= {
        "prior_failure_changed_condition",
        "prior_failure_retirement",
    }

    staged = tmp_path / mod.PRE_STAGED_ROADMAP
    staged.write_text(yaml.safe_dump(_complete_roadmap()), encoding="utf-8")
    loaded, receipt = mod.load_pre_staged_roadmap(tmp_path)
    assert loaded["milestone"] == mod.MILESTONE
    assert receipt["source_state"] == "working_tree"

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        mod.load_pre_staged_roadmap(empty)

    complete = mod.build_artifact(
        repo_root=REPO,
        run_date="20260826",
        roadmap_payload=_complete_roadmap(),
        roadmap_source={"path": "fixture", "source_state": "test", "sha256": "sha256:x"},
        tests_run=[{"command": "failed", "exit_code": 1, "duration_s": 0.1}],
        duration_s=1.0,
        protected_before={path.as_posix(): "sha256:wrong" for path in mod.PROTECTED_PATHS},
    )
    assert any(row["check"] == "tests" for row in complete["gate_check_summary"]["failed_checks"])
    assert any(
        row["check"] == "protected_files" for row in complete["gate_check_summary"]["failed_checks"]
    )

    artifact = mod.build_artifact(
        repo_root=REPO,
        run_date="20260826",
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
        duration_s=1.0,
    )
    invalid_cases = []
    missing = deepcopy(artifact)
    del missing["status"]
    missing["reproducibility_checksum"] = mod.reproducibility_checksum(missing)
    invalid_cases.append((missing, "missing required fields"))
    invalid_class = deepcopy(artifact)
    invalid_class["verdict_class"] = "invalid"
    invalid_class["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_class)
    invalid_cases.append((invalid_class, "closed enum"))
    invalid_substrate = deepcopy(artifact)
    invalid_substrate["inference_substrate"] = "llm"
    invalid_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_substrate)
    invalid_cases.append((invalid_substrate, "substrate"))
    invalid_rows = deepcopy(artifact)
    invalid_rows["task_contract_rows"] = []
    invalid_rows["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_rows)
    invalid_cases.append((invalid_rows, "row count"))
    invalid_attacks = deepcopy(artifact)
    invalid_attacks["attack_rows"][0]["fail_closed"] = False
    invalid_attacks["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_attacks)
    invalid_cases.append((invalid_attacks, "attack contract"))
    invalid_block = deepcopy(artifact)
    invalid_block["gate_check_summary"]["failed_checks"] = []
    invalid_block["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_block)
    invalid_cases.append((invalid_block, "blocked readiness"))
    invalid_binary = deepcopy(artifact)
    invalid_binary["execution_contract_ready_score"] = 0.5
    invalid_binary["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_binary)
    invalid_cases.append((invalid_binary, "binary"))
    for candidate, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(candidate)

    output = tmp_path / "replace-failure.json"
    original_replace = os.replace

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        mod.write_artifact_atomic(output, artifact)
    monkeypatch.setattr(os, "replace", original_replace)
    assert not list(tmp_path.glob("*.tmp"))
