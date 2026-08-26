"""Focused tests for the V578 activation contract.

Spec refs: REQ-REPORT-6619, REQ-REPORT-6619-COMPLETE-ACTIVATION,
REQ-REPORT-6619-DOCUMENT-YAML, REQ-REPORT-6619-UNIQUE-IDENTITY,
REQ-REPORT-6619-GATE-OWNERSHIP, REQ-REPORT-6619-PRIOR-FAILURES,
REQ-REPORT-6619-MODEL-POLICY, REQ-REPORT-6619-ARTIFACT-CONTRACT,
REQ-REPORT-6619-PROMPT-ENDINGS, REQ-REPORT-6619-PROTECTION-ATOMIC,
REQ-REPORT-6619-FAIL-CLOSED, SCENARIO-REPORT-6619-COMPLETE-ACTIVATION,
SCENARIO-REPORT-6619-DOCUMENT-YAML, SCENARIO-REPORT-6619-GATE-OWNERS,
SCENARIO-REPORT-6619-PRIOR-FAILURES,
SCENARIO-REPORT-6619-MODELS-AND-FIELDS,
SCENARIO-REPORT-6619-ATTACKS, and SCENARIO-REPORT-6619-ATOMIC-BLOCK.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6619_v578_activation_contract as mod


REPO = Path(__file__).resolve().parents[2]


def _prompt(fields: list[str], models: tuple[str, ...] = ()) -> str:
    """Build one complete prompt fixture with all execution evidence policy."""

    lines = ["MODEL_SPECS", *models]
    if models:
        lines.extend(
            (
                "Record exact model identity, model hash, and quant hash receipts.",
                "Derive tokenizer and chat-template behavior from GGUF metadata.",
                "Refuse silent fallback.",
                "Legacy Qwen3.5-0.8B and Gemma-4-E4B-it are CPU smoke only and cannot satisfy readiness.",
            )
        )
    lines.append("REQUIRED ARTIFACT FIELDS:")
    for field in fields:
        lines.append(f'  {field}: {{principle: "Principle for {field}."}}')
    lines.extend(
        (
            "Set inference_substrate=fixture_no_llm and verifier_is_oracle=true.",
            "Do NOT push. Do NOT modify scripts/research_conductor.py.",
        )
    )
    return "\n".join(lines)


def _complete_roadmap() -> dict[str, object]:
    """Create the complete fourteen-task contract used by positive fixtures."""

    tasks = []
    owner_fields = {
        gate["upstream"]: gate["artifact_field"]
        for gates in mod.EXPECTED_GATES.values()
        for gate in gates
    }
    for order, task_id in enumerate(mod.EXPECTED_TASK_IDS):
        fields = list(mod.COMMON_TASK_FIELDS)
        if task_id in owner_fields:
            fields.append(owner_fields[task_id])
        if task_id in mod.COMPARATIVE_TASK_IDS:
            fields.append("rows")
        tasks.append(
            {
                "id": task_id,
                "milestone": mod.MILESTONE,
                "deliverable": mod.EXPECTED_DELIVERABLES[task_id],
                "title": f"Task {order}",
                "track": "fixture",
                "requires_gpu": task_id in mod.EXPECTED_GPU_TASK_IDS,
                "per_unit_rows": task_id in mod.COMPARATIVE_TASK_IDS,
                "agent_type": "codex" if order % 2 else None,
                "model": "gpt-5.6-sol" if order % 2 else "opus",
                "gated_on": deepcopy(mod.EXPECTED_GATES.get(task_id, [])),
                "prior_failures": [],
                "prompt": _prompt(fields, mod.REQUIRED_MODELS_BY_TASK.get(task_id, ())),
            }
        )
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": mod.ROADMAP_DOCUMENT.as_posix(),
        "tasks": tasks,
    }


def test_req_report_6619_spec_and_document_define_fourteen_tasks() -> None:
    """REQ-REPORT-6619-COMPLETE-ACTIVATION: spec and document are explicit."""

    spec = (REPO / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec.split("REQ-REPORT-6619", 1)[1]
    for marker in (
        "SCENARIO-REPORT-6619-COMPLETE-ACTIVATION",
        "SCENARIO-REPORT-6619-DOCUMENT-YAML",
        "SCENARIO-REPORT-6619-GATE-OWNERS",
        "SCENARIO-REPORT-6619-PRIOR-FAILURES",
        "SCENARIO-REPORT-6619-MODELS-AND-FIELDS",
        "SCENARIO-REPORT-6619-ATTACKS",
        "SCENARIO-REPORT-6619-ATOMIC-BLOCK",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    document = mod.parse_document_contract(
        (REPO / mod.ROADMAP_DOCUMENT).read_text(encoding="utf-8")
    )
    assert document["experiment_numbers"] == list(range(6619, 6633))
    assert len(document["deliverables"]) == 14
    assert document["gates"] == mod.EXPECTED_GATES_BY_NUMBER


def test_scenario_report_6619_active_and_staged_omissions_fail_closed() -> None:
    """SCENARIO-REPORT-6619-COMPLETE-ACTIVATION: omissions stay visible."""

    sources = mod.load_roadmap_sources(REPO)
    assert sources["active"]["receipt"]["source_state"] == "working_tree"
    assert sources["pre_staged"]["receipt"]["source_state"] == "historical_git_blob"
    document = mod.parse_document_contract(
        (REPO / mod.ROADMAP_DOCUMENT).read_text(encoding="utf-8")
    )
    audit = mod.validate_activation_contract(
        sources["pre_staged"]["payload"],
        sources["active"]["payload"],
        document,
        retired_ids=set(),
    )
    assert len(audit["task_contract_rows"]) == 14
    assert audit["document_yaml_diff"]["missing_active_task_numbers"] == [6629, 6630, 6631, 6632]
    assert audit["document_yaml_diff"]["missing_pre_staged_task_numbers"] == [
        6629,
        6630,
        6631,
        6632,
    ]
    assert not audit["passed"]


def test_scenario_report_6619_complete_document_yaml_and_gate_ownership() -> None:
    """SCENARIO-REPORT-6619-GATE-OWNERS: exact earlier fields validate."""

    roadmap = _complete_roadmap()
    document = mod.synthetic_document_contract()
    audit = mod.validate_activation_contract(
        roadmap, deepcopy(roadmap), document, retired_ids=set()
    )
    assert audit["passed"]
    assert len(audit["gate_owner_rows"]) == 13
    assert all(row["passed"] for row in audit["gate_owner_rows"])
    assert all(audit["document_yaml_diff"]["matches"].values())

    bad = deepcopy(roadmap)
    bad["tasks"][1]["gated_on"][0]["artifact_field"] = "misspelled_ready_score"
    failed = mod.validate_activation_contract(bad, deepcopy(bad), document, retired_ids=set())
    assert {error["check"] for error in failed["errors"]} >= {
        "document_gate_set",
        "gate_owner_contract",
    }

    forward = deepcopy(roadmap)
    forward["tasks"][0]["gated_on"] = [
        {
            "upstream": mod.EXPECTED_TASK_IDS[1],
            "artifact_field": "gpu_lease_scheduler_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ]
    assert not mod.validate_activation_contract(
        forward, deepcopy(forward), document, retired_ids=set()
    )["passed"]


def test_req_report_6619_identity_model_field_and_ending_attacks() -> None:
    """REQ-REPORT-6619-MODEL-POLICY: prompt and identity attacks are rejected."""

    document = mod.synthetic_document_contract()
    cases = []
    duplicate = _complete_roadmap()
    duplicate["tasks"][-1]["deliverable"] = duplicate["tasks"][0]["deliverable"]
    cases.append(duplicate)
    wrong_id = _complete_roadmap()
    wrong_id["tasks"][2]["deliverable"] = "results/experiment_9999_wrong.json"
    cases.append(wrong_id)
    wrong_model = _complete_roadmap()
    wrong_model["tasks"][2]["prompt"] = wrong_model["tasks"][2]["prompt"].replace(
        mod.QWEN_MODEL, "wrong/model-GGUF"
    )
    cases.append(wrong_model)
    no_principle = _complete_roadmap()
    no_principle["tasks"][0]["prompt"] = no_principle["tasks"][0]["prompt"].replace(
        '{principle: "Principle for status."}', "{}"
    )
    cases.append(no_principle)
    no_ending = _complete_roadmap()
    no_ending["tasks"][0]["prompt"] += "\nTrailing instruction."
    cases.append(no_ending)
    no_rows = _complete_roadmap()
    comparative = next(task for task in no_rows["tasks"] if task["id"] in mod.COMPARATIVE_TASK_IDS)
    comparative["per_unit_rows"] = False
    comparative["prompt"] = comparative["prompt"].replace("  rows:", "  removed_rows:")
    cases.append(no_rows)
    assert all(
        not mod.validate_activation_contract(case, deepcopy(case), document, retired_ids=set())[
            "passed"
        ]
        for case in cases
    )


def test_scenario_report_6619_prior_failures_are_exact_and_concrete() -> None:
    """SCENARIO-REPORT-6619-PRIOR-FAILURES: stored verdicts stay exact."""

    roadmap = yaml.safe_load((REPO / mod.ACTIVE_ROADMAP).read_text(encoding="utf-8"))
    rows, errors = mod.reconcile_prior_failures(REPO, roadmap["tasks"])
    row_6616 = next(row for row in rows if row["experiment_number"] == 6616)
    assert row_6616["verdict_matches"]
    assert row_6616["changed_condition_concrete"]
    assert row_6616["retire_if_same_verdict"]
    assert not [error for error in errors if error.get("experiment_number") == 6616]

    bad = deepcopy(roadmap["tasks"])
    bad[0]["prior_failures"][0].update(
        verdict="wrong", addressed_by="retry", retire_if_same_verdict=False
    )
    _, bad_errors = mod.reconcile_prior_failures(REPO, bad)
    assert {error["check"] for error in bad_errors} >= {
        "prior_failure_verdict",
        "prior_failure_changed_condition",
        "prior_failure_retirement",
    }


def test_scenario_report_6619_all_requested_attacks_fail_closed() -> None:
    """SCENARIO-REPORT-6619-ATTACKS: every requested mutation is rejected."""

    rows = mod.run_attacks()
    assert {row["attack_id"] for row in rows} == set(mod.ATTACK_IDS)
    assert all(row["mutation_applied"] and row["fail_closed"] for row in rows)


def test_scenario_report_6619_blocked_and_ready_artifacts_are_valid(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6619-ATOMIC-BLOCK: blocked and ready results are exact."""

    blocked = mod.build_artifact(
        repo_root=REPO,
        run_date="20260826",
        tests_run=[{"command": "focused", "scope": "focused", "exit_code": 0}],
        duration_s=2.0,
    )
    assert blocked["status"] == "blocked_v578_activation_contract_incomplete"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["activation_contract_ready_score"] == 0.0
    assert blocked["gate_check_summary"]["failed_checks"]
    assert len(blocked["task_contract_rows"]) == 14
    assert set(mod.REQUIRED_FIELDS) == set(blocked["field_provenance"])
    mod.validate_artifact(blocked)

    roadmap = _complete_roadmap()
    ready = mod.build_artifact(
        repo_root=REPO,
        run_date="20260826",
        pre_staged_payload=roadmap,
        active_payload=deepcopy(roadmap),
        document_contract=mod.synthetic_document_contract(),
        tests_run=[{"command": "focused", "scope": "focused", "exit_code": 0}],
        duration_s=1.0,
    )
    assert ready["activation_contract_ready_score"] == 1.0
    assert ready["verdict_class"] == "null"
    assert ready["status"] == "complete_v578_activation_contract_ready"
    mod.validate_artifact(ready)

    output = tmp_path / "nested" / "contract.json"
    mod.write_artifact_atomic(output, blocked)
    assert json.loads(output.read_text(encoding="utf-8")) == blocked
    assert not list(output.parent.glob("*.tmp"))


def test_req_report_6619_artifact_mutations_and_atomic_failure_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6619-PROTECTION-ATOMIC: mutation cannot survive validation."""

    artifact = mod.build_artifact(repo_root=REPO, run_date="20260826", duration_s=1.0)
    mutations = []
    missing = deepcopy(artifact)
    del missing["status"]
    mutations.append((missing, "missing required fields"))
    wrong_checksum = deepcopy(artifact)
    wrong_checksum["status"] = "mutated"
    mutations.append((wrong_checksum, "checksum"))
    wrong_class = deepcopy(artifact)
    wrong_class["verdict_class"] = "invalid"
    wrong_class["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_class)
    mutations.append((wrong_class, "closed enum"))
    wrong_substrate = deepcopy(artifact)
    wrong_substrate["inference_substrate"] = "llm"
    wrong_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_substrate)
    mutations.append((wrong_substrate, "substrate"))
    wrong_rows = deepcopy(artifact)
    wrong_rows["task_contract_rows"] = []
    wrong_rows["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_rows)
    mutations.append((wrong_rows, "row count"))
    wrong_attacks = deepcopy(artifact)
    wrong_attacks["attack_rows"][0]["fail_closed"] = False
    wrong_attacks["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_attacks)
    mutations.append((wrong_attacks, "attack contract"))
    wrong_score = deepcopy(artifact)
    wrong_score["activation_contract_ready_score"] = 1.0
    wrong_score["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_score)
    mutations.append((wrong_score, "readiness"))
    nonbinary = deepcopy(artifact)
    nonbinary["activation_contract_ready_score"] = 0.5
    nonbinary["reproducibility_checksum"] = mod.reproducibility_checksum(nonbinary)
    mutations.append((nonbinary, "binary"))
    for candidate, message in mutations:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(candidate)

    original_replace = os.replace

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        mod.write_artifact_atomic(tmp_path / "contract.json", artifact)
    monkeypatch.setattr(os, "replace", original_replace)
    assert not list(tmp_path.glob("*.tmp"))


def test_req_report_6619_defensive_parsers_and_protection(tmp_path: Path) -> None:
    """REQ-REPORT-6619-FAIL-CLOSED: malformed sources retain named failures."""

    assert mod.extract_required_fields("no field block") == {}
    inline = 'REQUIRED ARTIFACT FIELDS:\n  status: {principle: "why"}\nSet inference_substrate=x'
    assert mod.extract_required_fields(inline) == {"status": "why"}
    multiline = (
        'REQUIRED ARTIFACT FIELDS:\n  status:\n    principle: "why"\nSet inference_substrate=x'
    )
    assert mod.extract_required_fields(multiline) == {"status": "why"}
    assert mod._experiment_number("no experiment") is None

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        mod.load_roadmap_sources(empty)

    no_history = tmp_path / "no-history"
    no_history.mkdir()
    (no_history / mod.ACTIVE_ROADMAP).write_text("milestone: test\ntasks: []\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        mod.load_roadmap_sources(no_history)

    working = tmp_path / "working"
    working.mkdir()
    payload = _complete_roadmap()
    for relative in (mod.ACTIVE_ROADMAP, mod.PRE_STAGED_ROADMAP):
        (working / relative).write_text(yaml.safe_dump(payload), encoding="utf-8")
    working_sources = mod.load_roadmap_sources(working)
    assert working_sources["pre_staged"]["receipt"]["source_state"] == "working_tree"

    protected = mod.protected_file_receipts(REPO)
    assert protected["all_unchanged"]
    assert {row["path"] for row in protected["rows"]} == {
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    }
    changed = mod.protected_file_receipts(
        REPO, {path.as_posix(): "sha256:wrong" for path in mod.PROTECTED_PATHS}
    )
    assert not changed["all_unchanged"]

    malformed = {"milestone": "old", "tasks": "bad"}
    audit = mod.validate_activation_contract(
        malformed, malformed, mod.synthetic_document_contract(), retired_ids=set()
    )
    assert {error["check"] for error in audit["errors"]} >= {
        "roadmap_schema",
        "roadmap_milestone",
        "document_task_order",
    }


def test_req_report_6619_remaining_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-REPORT-6619-FAIL-CLOSED: every defensive branch names its defect."""

    document = mod.synthetic_document_contract()
    duplicate = _complete_roadmap()
    duplicate["tasks"][-1]["id"] = duplicate["tasks"][0]["id"]
    stale = _complete_roadmap()
    stale["tasks"][0]["milestone"] = "2026.08.577"
    missing_fields = _complete_roadmap()
    missing_fields["tasks"][0]["prompt"] = mod.PROMPT_TERMINATOR
    for roadmap, check in (
        (duplicate, "duplicate_task_id"),
        (stale, "task_milestone"),
        (missing_fields, "required_artifact_fields"),
    ):
        audit = mod.validate_activation_contract(
            roadmap, deepcopy(roadmap), document, retired_ids=set()
        )
        assert check in {error["check"] for error in audit["errors"]}

    bad_json = tmp_path / "list.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact root"):
        mod._read_json(bad_json)
    assert mod._unwrap({"value": "blocked", "principle": "why"}) == "blocked"

    failed_tests = mod.build_artifact(
        repo_root=REPO,
        run_date="20260826",
        pre_staged_payload=_complete_roadmap(),
        active_payload=_complete_roadmap(),
        document_contract=document,
        tests_run=[{"command": "failed", "scope": "focused", "exit_code": 1}],
        duration_s=1.0,
        protected_before={path.as_posix(): "sha256:wrong" for path in mod.PROTECTED_PATHS},
    )
    checks = {row["check"] for row in failed_tests["gate_check_summary"]["failed_checks"]}
    assert {"tests", "protected_files"} <= checks

    bad_provenance = deepcopy(failed_tests)
    del bad_provenance["field_provenance"]["status"]
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field provenance"):
        mod.validate_artifact(bad_provenance)

    bad_block = deepcopy(failed_tests)
    bad_block["gate_check_summary"]["failed_checks"] = []
    bad_block["reproducibility_checksum"] = mod.reproducibility_checksum(bad_block)
    with pytest.raises(ValueError, match="blocked readiness"):
        mod.validate_artifact(bad_block)
