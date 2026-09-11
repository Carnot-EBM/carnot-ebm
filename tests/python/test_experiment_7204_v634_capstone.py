"""Tests for REQ-REPORT-7204 and the V634 capstone scenarios."""

from __future__ import annotations

from copy import deepcopy
import ctypes
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/experiments/experiment_7204_v634_capstone.py"
MODULE_SPEC = importlib.util.spec_from_file_location("experiment_7204_v634_capstone", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
exp = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(exp)


def _trim_json_memory() -> None:
    """Return the large Exp7199 parse arena before the memory watchdog samples RSS."""

    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)


def _clean_adversarial(path: Path) -> dict[str, Any]:
    """Return a clean checker result while retaining the real checker shape."""

    return {
        "artifact": str(path),
        "loaded": True,
        "flag_count": 0,
        "max_severity": -1,
        "flags": [],
        "gate_version": "fixture-gate",
    }


def _clean_row(_path: Path) -> tuple[str, list[str]]:
    """Keep capstone unit tests independent of producer lint policy changes."""

    return "ok", []


def _checker_loader(_root: Path) -> tuple[Any, Any]:
    """Expose the two shipped checker interfaces without rereading large files."""

    return _clean_adversarial, _clean_row


def _publication_gate(_root: Path) -> dict[str, Any]:
    """Return the unchanged G1-G4 response contract for deterministic tests."""

    return {
        "paper_ready": False,
        "gates": {name: {"pass": name != "G2", "detail": name} for name in exp.GATE_IDS},
        "unmet_gates": ["G2"],
        "note": "fixture",
    }


def _contract_tasks() -> list[dict[str, Any]]:
    """Make the exact frozen roster with each real structured gate."""

    tasks = []
    for task_id, title, deliverable in exp.EXPECTED_CONTRACT:
        tasks.append(
            {
                "id": task_id,
                "title": title,
                "deliverable": deliverable,
                "milestone": exp.MILESTONE,
                "per_unit_rows": True,
                "gated_on": deepcopy(exp.EXPECTED_GATES.get(task_id, [])),
                "prior_failures": [
                    {
                        "experiment_id": f"prior-{task_id}",
                        "verdict": f"prior verdict for {task_id}",
                        "addressed_by": "A changed mechanism was planned.",
                        "retire_if_same_verdict": True,
                    }
                ],
            }
        )
    return tasks


def _write_contract(root: Path, *, frozen: bool = True) -> Path:
    """Write a small matching contract at the selected authority path."""

    relative = exp.FROZEN_ROADMAP_PATH if frozen else exp.ACTIVE_ROADMAP_PATH
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"milestone": exp.MILESTONE, "tasks": _contract_tasks()}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_req_report_7204_spec_precedes_implementation() -> None:
    """REQ-REPORT-7204: the spec names every required field and scenario."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7204") :]
    assert all(field in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    for name in (
        "CONTRACT",
        "INTAKE",
        "CLAIMS",
        "BOUNDARIES",
        "DECISIONS",
        "PUBLICATION",
        "BLOCKED",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7204-{name}" in section


def test_scenario_report_7204_contract_prefers_frozen_roster(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7204-CONTRACT: frozen matching bytes outrank mutable files."""

    frozen = _write_contract(tmp_path)
    active = tmp_path / exp.ACTIVE_ROADMAP_PATH
    active.write_text(yaml.safe_dump({"milestone": "wrong", "tasks": []}), encoding="utf-8")
    tasks, sources, selected, errors = exp.resolve_contract(tmp_path)
    assert errors == []
    assert selected == exp.FROZEN_ROADMAP_PATH
    assert [task["id"] for task in tasks] == list(exp.EXPECTED_TASK_IDS)
    assert sources[0]["selected"] is True
    assert sources[0]["sha256"] == exp.sha256_path(frozen)


def test_scenario_report_7204_contract_falls_back_and_rejects_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7204-CONTRACT: fallback needs exact milestone and fields."""

    _write_contract(tmp_path, frozen=False)
    tasks, _sources, selected, errors = exp.resolve_contract(tmp_path)
    assert len(tasks) == 13 and selected == exp.ACTIVE_ROADMAP_PATH and errors == []

    tasks[0]["title"] = "changed"
    (tmp_path / exp.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump({"milestone": exp.MILESTONE, "tasks": tasks}, sort_keys=False),
        encoding="utf-8",
    )
    _tasks, _sources, _selected, errors = exp.resolve_contract(tmp_path)
    assert "contract_fields" in errors

    (tmp_path / exp.ACTIVE_ROADMAP_PATH).write_text("tasks: [", encoding="utf-8")
    tasks, sources, selected, errors = exp.resolve_contract(tmp_path)
    assert tasks == [] and selected is None and "contract_authority" in errors
    assert sources[1]["read_error"].startswith("ParserError:")


def test_scenario_report_7204_intake_declared_then_canonical_and_quarantine(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7204-INTAKE: quarantine excludes a gate-passing producer."""

    task = _contract_tasks()[2]
    declared = tmp_path / str(task["deliverable"])
    declared.parent.mkdir(parents=True)
    declared.write_text(
        json.dumps(
            {
                "status": "complete",
                "verdict_class": "positive",
                "arc_gap_audit_complete_score": 1,
                "flagged_adversarial": True,
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    receipt = exp.load_task_evidence(tmp_path, task, {"retired": []})
    assert receipt["evidence_source"] == "declared_deliverable"
    assert receipt["quarantined"] is True
    assert receipt["accepted_for_promoted_evidence"] is False

    declared.unlink()
    canonical = tmp_path / exp.canonical_gate_block_path(str(task["id"]))
    canonical.write_text(json.dumps({"status": "blocked", "rows": []}), encoding="utf-8")
    receipt = exp.load_task_evidence(tmp_path, task, {"retired": []})
    assert receipt["selected_evidence_path"] == exp.canonical_gate_block_path(str(task["id"]))
    assert receipt["evidence_source"] == "conductor_gate_block"

    canonical.unlink()
    receipt = exp.load_task_evidence(tmp_path, task, {"retired": []})
    assert receipt["evidence_source"] == "missing"
    assert receipt["selected_evidence_path"] is None


@pytest.fixture(scope="module")
def real_claim_rows() -> list[dict[str, Any]]:
    """Read each V634 producer once and reconstruct its selected headlines."""

    rows: list[dict[str, Any]] = []
    for task_id, _title, deliverable in exp.EXPECTED_CONTRACT[:-1]:
        payload, error = exp.read_json_summary(ROOT / deliverable)
        assert error is None and payload is not None
        rows.extend(exp.recompute_claims(task_id, payload))
        del payload
    _trim_json_memory()
    return rows


def test_scenario_report_7204_claims_recompute_real_producers(
    real_claim_rows: list[dict[str, Any]],
) -> None:
    """SCENARIO-REPORT-7204-CLAIMS: all selected headlines match raw rows."""

    assert real_claim_rows
    assert all(row["matches"] for row in real_claim_rows)
    assert {row["task_id"] for row in real_claim_rows} == set(exp.EXPECTED_TASK_IDS[:-1])
    assert (
        next(row for row in real_claim_rows if row["claim"] == "acquisition_value_score")[
            "recomputed_value"
        ]
        == 0
    )
    assert (
        next(row for row in real_claim_rows if row["claim"] == "hardware_envelope_complete_score")[
            "recomputed_value"
        ]
        == 1
    )


def test_scenario_report_7204_claim_mutation_is_detected() -> None:
    """SCENARIO-REPORT-7204-CLAIMS: a forged learning value fails row replay."""

    payload, error = exp.read_json_summary(
        ROOT / "results/experiment_7199_v634_bounded_acquisition.json"
    )
    assert error is None and payload is not None
    payload["acquisition_value_score"] = 1
    rows = exp.recompute_claims("exp7199-bounded-acquisition", payload)
    assert any(row["claim"] == "acquisition_value_score" and not row["matches"] for row in rows)
    del payload
    _trim_json_memory()


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the real matrix once with deterministic checker and publication responses."""

    directory = tmp_path_factory.mktemp("exp7204")
    result = exp.build_artifact(
        ROOT,
        exp.RUN_DATE,
        directory / "experiment_7204.json",
        directory / "results/checkpoints/checkpoint.json",
        checker_loader=_checker_loader,
        publication_runner=_publication_gate,
    )
    _trim_json_memory()
    return result


def test_scenario_report_7204_boundaries_keep_complete_nulls_separate(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-7204-BOUNDARIES: completion is not PRD value."""

    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["capstone_complete_score"] == 1
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    matrix = {row["task_id"]: row for row in artifact["evidence_matrix"]}
    assert matrix["exp7193-arc-direct-tool"]["measurement_completed"] is True
    assert matrix["exp7193-arc-direct-tool"]["method_value_established"] is False
    assert matrix["exp7195-typed-grounding"]["execution_grounded_circular_gain"] is True
    assert matrix["exp7199-bounded-acquisition"]["useful_continual_learning"] is False
    assert matrix["exp7203-hardware-correction"]["measured_production_deployment"] is False


def test_scenario_report_7204_decisions_cover_exact_roster(artifact: dict[str, Any]) -> None:
    """SCENARIO-REPORT-7204-DECISIONS: every branch has one bounded action."""

    decisions = {row["task_id"]: row for row in artifact["branch_decisions"]}
    assert set(decisions) == set(exp.EXPECTED_TASK_IDS)
    assert {row["action"] for row in decisions.values()} <= exp.BRANCH_ACTIONS
    assert decisions["exp7194-arc-gap-audit"]["action"] == "retire"
    assert decisions["exp7199-bounded-acquisition"]["action"] == "retire"
    assert decisions["exp7203-hardware-correction"]["action"] == ("needs_changed_prerequisite")
    assert decisions["exp7204-capstone"]["prior_failures"][0]["experiment_id"] == (
        "exp7191-capstone"
    )


def test_scenario_report_7204_publication_preserves_gate_shape(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-7204-PUBLICATION: the G1-G4 response remains unchanged."""

    publication = artifact["publication_gate"]
    assert tuple(publication["gates"]) == exp.GATE_IDS
    assert publication["paper_ready"] is False
    assert publication["unmet_gates"] == ["G2"]
    assert publication["operator_only"] is True


def test_scenario_report_7204_artifact_validator_rejects_mutations(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-7204-ARTIFACT: roster, claims, gates, and checksum fail closed."""

    assert exp.validate_artifact(artifact, root=ROOT) == []
    mutations = (
        ("rows", []),
        ("recomputed_claim_rows", []),
        ("branch_decisions", []),
        ("publication_gate", {}),
        ("verdict_class", "positive"),
    )
    for field, value in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert exp.validate_artifact(changed, root=ROOT), field

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_scenario_report_7204_rows_are_numeric_comparison_evidence(
    artifact: dict[str, Any], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7204-CLAIMS: the protected row lint sees measured comparisons."""

    assert artifact["rows"] == artifact["recomputed_claim_rows"]
    required = {"unit_id", "arm", "seed", "metric", "error", "abstention", "claim_score"}
    assert artifact["rows"] and all(required <= row.keys() for row in artifact["rows"])
    path = tmp_path / "capstone.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    _adversarial, row_check = exp.default_checker_loader(ROOT)
    assert row_check(path) == ("ok", [])


def test_scenario_report_7204_blocked_preflight_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7204-BLOCKED: an external absence writes a complete block."""

    output = tmp_path / "results/result.json"
    checkpoint = tmp_path / "results/checkpoints/checkpoint.json"
    result = exp.build_artifact(
        tmp_path,
        exp.RUN_DATE,
        output,
        checkpoint,
        checker_loader=_checker_loader,
        publication_runner=_publication_gate,
    )
    assert output.is_file() and checkpoint.is_file()
    assert result["status"] == "blocked"
    assert result["verdict_class"] == "blocked"
    assert result["inference_substrate_class"] == "blocked_no_run"
    assert result["capstone_complete_score"] == 0
    assert result["gate_check_summary"]["failed_check"] == "driving_capability_spec"
    assert exp.validate_artifact(result) == []


def test_req_report_7204_cli_and_edge_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7204: CLI validation, dates, JSON, and gate parsing fail closed."""

    with pytest.raises(ValueError, match="20260911"):
        exp.initialize_artifact("20260910")
    with pytest.raises(ValueError, match="20260911"):
        exp.initialize_artifact("not-a-date")
    with pytest.raises(ValueError, match="invalid task id"):
        exp.task_number("wrong")
    payload, error = exp.read_json_object(tmp_path / "missing.json")
    assert payload is None and error.startswith("FileNotFoundError:")
    malformed = tmp_path / "bad.json"
    malformed.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(malformed) == (None, "json_root_not_object")

    monkeypatch.setattr(exp, "run_publication_gate", _publication_gate)
    output = tmp_path / "capstone.json"
    checkpoint = tmp_path / "results/checkpoints/checkpoint.json"
    args = [
        "--root",
        str(ROOT),
        "--date",
        exp.RUN_DATE,
        "--artifact-path",
        str(output),
        "--checkpoint-path",
        str(checkpoint),
    ]
    assert exp.main(args, checker_loader=_checker_loader) == 0
    assert exp.main(["--root", str(ROOT), "--validate", "--artifact-path", str(output)]) == 0
    output.write_text("{}", encoding="utf-8")
    assert exp.main(["--root", str(ROOT), "--validate", "--artifact-path", str(output)]) == 1


def test_req_report_7204_publication_runner_rejects_bad_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7204: the subprocess receipt needs valid G1-G4 JSON."""

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="G1-G4"):
        exp.run_publication_gate(tmp_path)


def test_req_report_7204_parser_and_loader_defensive_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7204: malformed helper, roadmap, and artifact inputs fail closed."""

    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(RuntimeError, match="cannot load shipped"):
        exp._load_v633_base()
    monkeypatch.undo()

    assert exp._contract_tasks("bad") == (None, [])
    assert exp._contract_tasks([]) == (None, [])
    assert exp._contract_tasks({"milestone": exp.MILESTONE, "tasks": "bad"}) == (
        exp.MILESTONE,
        [],
    )
    assert (
        exp._contract_tasks([{"milestone": exp.MILESTONE, "id": exp.EXPECTED_TASK_IDS[0]}])[0]
        == exp.MILESTONE
    )
    assert "contract_identity" in exp._contract_errors(exp.MILESTONE, [{"id": "unknown"}])
    changed_gate = _contract_tasks()
    changed_gate[2]["gated_on"] = []
    assert "contract_gates" in exp._contract_errors(exp.MILESTONE, changed_gate)
    assert exp._unwrap({"principle": "why", "value": True}) is True
    assert exp._grounding_value({}) == 0
    assert exp._source_hash_shape({}) == {
        "type": "NoneType",
        "entry_count": 0,
        "valid_sha256_count": 0,
    }

    task = _contract_tasks()[0]
    path = tmp_path / task["deliverable"]
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    receipt = exp.load_task_evidence(tmp_path, task, {"retired": []})
    assert receipt["evidence_source"] == "unreadable"


def test_req_report_7204_large_reader_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7204: the bounded large-file child reports every parse failure."""

    path = tmp_path / "experiment_7199_large.json"
    with path.open("wb") as handle:
        handle.truncate(exp.LARGE_ARTIFACT_THRESHOLD_BYTES + 1)

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "child failed"

    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: Completed())
    assert exp.read_json_summary(path) == (None, "isolated_json_read_failed:child failed")
    Completed.returncode = 0
    Completed.stdout = "{"
    assert exp.read_json_summary(path)[1].startswith("JSONDecodeError:")
    Completed.stdout = "[]"
    assert exp.read_json_summary(path) == (None, "json_root_not_object")


def test_req_report_7204_publication_runner_error_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7204: command failure, invalid JSON, and success remain distinct."""

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "gate failed"

    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="gate failed"):
        exp.run_publication_gate(tmp_path)
    Completed.returncode = 0
    Completed.stdout = "{"
    with pytest.raises(RuntimeError, match="invalid JSON"):
        exp.run_publication_gate(tmp_path)
    Completed.stdout = json.dumps(_publication_gate(tmp_path))
    assert tuple(exp.run_publication_gate(tmp_path)["gates"]) == exp.GATE_IDS


def test_req_report_7204_precondition_source_and_directory_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7204: missing source bytes and unwritable parents stop early."""

    spec = tmp_path / exp.SPEC_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("### REQ-REPORT-7204: fixture", encoding="utf-8")
    checks, _hashes, failed = exp._preconditions(
        tmp_path,
        tmp_path / "out/result.json",
        tmp_path / "results/checkpoints/checkpoint.json",
    )
    assert failed == checks[-1] and failed["check"] == "required_source_bytes"

    monkeypatch.setattr(exp, "SOURCE_PATHS", ())
    bad_parent = tmp_path / "parent-file"
    bad_parent.write_text("not a directory", encoding="utf-8")
    checks, _hashes, failed = exp._preconditions(
        tmp_path,
        bad_parent / "result.json",
        tmp_path / "results/checkpoints/checkpoint.json",
    )
    assert failed == checks[-1] and failed["check"] == "output_directory"


def test_scenario_report_7204_scientific_block_diagnostics() -> None:
    """SCENARIO-REPORT-7204-BLOCKED: each external block retains exact fields."""

    self_row = {"task_id": "self"}
    missing = exp._first_scientific_block(
        [{"task_id": "a", "evidence_source": "missing"}, self_row], []
    )
    assert missing["failed_check"] == "upstream_terminal_evidence"
    quarantine = exp._first_scientific_block(
        [
            {
                "task_id": "a",
                "evidence_source": "declared_deliverable",
                "quarantined": True,
                "quarantine_receipt": {"declared_flags": {"quarantined": True}},
            },
            self_row,
        ],
        [],
    )
    assert quarantine["failed_check"] == "upstream_quarantine"
    blocked = exp._first_scientific_block(
        [
            {
                "task_id": "a",
                "evidence_source": "declared_deliverable",
                "quarantined": False,
                "verdict_class": "blocked",
                "producer_gate_check_summary": {
                    "failed_check": "external",
                    "field": "device",
                    "expected_value": "present",
                    "observed_value": "missing",
                },
            },
            self_row,
        ],
        [],
    )
    assert blocked["failed_check"] == "external"
    failed_gate = exp._first_scientific_block(
        [self_row],
        [
            {
                "passed": False,
                "producer": "a",
                "field": "ready",
                "expected_value": 1,
                "observed_value": 0,
            }
        ],
    )
    assert failed_gate["failed_check"] == "same_milestone_gate"


def test_req_report_7204_validator_fail_closed_classes(artifact: dict[str, Any]) -> None:
    """REQ-REPORT-7204: each derived validation class rejects its mutation."""

    blocked = exp.initialize_artifact(exp.RUN_DATE)
    blocked["status"] = "complete"
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert "blocked_preflight_state" in exp.validate_artifact(blocked)

    mutations: list[tuple[str, Any, str]] = [
        (
            "branch_decisions",
            [{**row, "action": "bad"} for row in artifact["branch_decisions"]],
            "branch_decision_action",
        ),
        (
            "recomputed_claim_rows",
            [
                row
                for row in artifact["recomputed_claim_rows"]
                if row["task_id"] != "exp7192-source-contract"
            ],
            "recomputed_claim_task_roster",
        ),
        (
            "recomputed_claim_rows",
            [{**row, "matches": False} for row in artifact["recomputed_claim_rows"]],
            "recomputed_claim_mismatch",
        ),
    ]
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["publication_gate"]["paper_ready"] = not changed["publication_gate"]["paper_ready"]
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "publication_gate_derivation" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["evidence_matrix"][0]["selected_evidence_path"] = "results/missing.json"
    changed["rows"] = changed["evidence_matrix"]
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "producer_path_replay" in exp.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["evidence_matrix"][0]["artifact_sha256"] = "sha256:" + "0" * 64
    changed["rows"] = changed["evidence_matrix"]
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "producer_hash_replay" in exp.validate_artifact(changed, root=ROOT)


def test_req_report_7204_build_contract_and_validation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7204: build-time contract and final parser guards stop writes."""

    monkeypatch.setattr(exp, "_preconditions", lambda *_args: ([], {}, None))
    monkeypatch.setattr(
        exp,
        "resolve_contract",
        lambda _root: ([], [], None, ["contract_authority"]),
    )
    output = tmp_path / "blocked.json"
    checkpoint = tmp_path / "results/checkpoints/blocked.json"
    result = exp.build_artifact(
        tmp_path,
        exp.RUN_DATE,
        output,
        checkpoint,
        checker_loader=_checker_loader,
        publication_runner=_publication_gate,
    )
    assert result["status"] == "blocked" and output.is_file()

    monkeypatch.setattr(
        exp,
        "resolve_contract",
        lambda _root: (_contract_tasks(), [], exp.FROZEN_ROADMAP_PATH, []),
    )
    frozen = tmp_path / exp.FROZEN_ROADMAP_PATH
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text("fixture", encoding="utf-8")
    manifest = tmp_path / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("retired: []", encoding="utf-8")

    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(RuntimeError, match="capstone validation failed"):
        exp.build_artifact(
            tmp_path,
            exp.RUN_DATE,
            tmp_path / "invalid.json",
            tmp_path / "results/checkpoints/invalid.json",
            checker_loader=_checker_loader,
            publication_runner=_publication_gate,
        )

    calls = iter(([], ["bad final"]))
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: next(calls))
    with pytest.raises(RuntimeError, match="final file-to-parser gate failed"):
        exp.build_artifact(
            tmp_path,
            exp.RUN_DATE,
            tmp_path / "invalid-final.json",
            tmp_path / "results/checkpoints/invalid-final.json",
            checker_loader=_checker_loader,
            publication_runner=_publication_gate,
        )
