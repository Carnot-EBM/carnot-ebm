"""Spec refs: REQ-INFRA-6260, SCENARIO-INFRA-6260-1,
SCENARIO-INFRA-6260-2, SCENARIO-INFRA-6260-3,
SCENARIO-INFRA-6260-4, SCENARIO-INFRA-6260-5,
SCENARIO-INFRA-6260-6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6260_v540_terminal_transition as exp6260


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _write_artifact(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload
            or {
                "status": "complete",
                "honest_verdict": "complete: fixture",
                "duration_s": 1.0,
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "verifier_is_oracle": False,
                "reproducibility_checksum": "sha256:test",
            }
        ),
        encoding="utf-8",
    )


def test_openspec_names_req_6260_and_scenarios() -> None:
    """REQ-INFRA-6260: OpenSpec records the V540 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6260") :]
    for token in (
        "REQ-INFRA-6260",
        "SCENARIO-INFRA-6260-1",
        "SCENARIO-INFRA-6260-2",
        "SCENARIO-INFRA-6260-3",
        "SCENARIO-INFRA-6260-4",
        "SCENARIO-INFRA-6260-5",
        "SCENARIO-INFRA-6260-6",
        "experiment_6260_v540_terminal_transition.py",
    ):
        assert token in section


def test_exact_path_and_preconditions_only_classification(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6260-1: exact path outranks aliases.

    SCENARIO-INFRA-6260-2: Exp6228-style preconditions stay nonterminal.
    """

    exact = tmp_path / "results/experiment_6228_declared.json"
    alias = tmp_path / "results/experiment_6228_complete_alias.json"
    _write_artifact(
        exact,
        {
            "status": "preconditions_recorded",
            "honest_verdict": None,
            "two_family_runtime_ready_score": None,
            "three_family_runtime_ready_score": None,
        },
    )
    _write_artifact(alias)

    rows = exp6260.classify_v539_declared_tasks(
        tmp_path,
        {
            "exp6228-supervised-three-family-runtime-endurance": {
                "task_id": "exp6228-supervised-three-family-runtime-endurance",
                "title": "Runtime preconditions",
                "track": "infrastructure",
                "declared_deliverable": "results/experiment_6228_declared.json",
                "terminal_class": "complete",
            }
        },
    )

    row = rows["exp6228-supervised-three-family-runtime-endurance"]
    assert row["classification"] == "unknown"
    assert row["terminal"] is False
    assert row["status_raw"] == "preconditions_recorded"
    assert row["same_number_alias_used"] is False
    assert row["same_number_alias_candidates_ignored"] == [
        "results/experiment_6228_complete_alias.json"
    ]
    assert row["capstone_terminal_class"] == "complete"


def test_reserved_collision_scan_keeps_concurrent_ids_separate(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6260-3: reserved id collisions fail closed."""

    _write_artifact(tmp_path / "results/experiment_6240_change_magnitude_prior.json")
    _write_artifact(tmp_path / "results/experiment_6244_mode_a_error_class_diagnosis.json")
    _write_artifact(tmp_path / "results/experiment_6260_v540_terminal_transition.json")
    pycache = tmp_path / "python/carnot/__pycache__/experiment_6260_fixture.cpython-312.pyc"
    pycache.parent.mkdir(parents=True, exist_ok=True)
    pycache.write_bytes(b"bytecode")
    unexpected = tmp_path / "python/carnot/experiment_6265_old.py"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("# fixture\n", encoding="utf-8")

    receipt = exp6260.scan_reserved_id_collisions(
        tmp_path,
        allowed_reserved_paths={"results/experiment_6260_v540_terminal_transition.json"},
    )

    assert receipt["concurrent_exp_ids"] == [6240, 6244, 6245, 6246]
    assert receipt["reserved_exp_ids"] == list(range(6260, 6272))
    assert receipt["concurrent_paths_by_exp_id"]["6240"] == [
        "results/experiment_6240_change_magnitude_prior.json"
    ]
    assert receipt["unexpected_reserved_collision_count"] == 1
    assert receipt["unexpected_reserved_paths_by_exp_id"]["6265"] == [
        "python/carnot/experiment_6265_old.py"
    ]


def test_current_v540_roadmap_contracts_and_dirty_failures() -> None:
    """SCENARIO-INFRA-6260-4: V540 roadmap contracts are mechanical."""

    data, identity = exp6260.load_v540_roadmap(REPO)
    retired = exp6260.load_retired_exp_ids(REPO / "ops/exclusion_manifest.yaml")
    clean = exp6260.validate_v540_roadmap_data(data, retired)

    assert identity["milestone"] == exp6260.MILESTONE_V540
    assert clean["schema_validation"]["ok"] is True
    assert clean["task_id_validation"]["expected_order"] is True
    assert clean["task_count"] == 12
    assert clean["retired_dependency_count"] == 0
    assert clean["id_collision_count"] == 0
    assert clean["dependency_validation"]["ok"] is True
    assert clean["gated_on_validation"]["ok"] is True
    assert clean["prior_failure_validation"]["ok"] is True
    assert clean["model_policy_validation"]["ok"] is True
    assert clean["prompt_contract_validation"]["ok"] is True

    dirty = json.loads(json.dumps(data))
    dirty["tasks"][1]["id"] = dirty["tasks"][0]["id"]
    dirty["tasks"][1]["agent_type"] = "claude"
    dirty["tasks"][1]["model"] = "sonnet"
    dirty["tasks"][1]["requires_gpu"] = True
    dirty["tasks"][2]["requires"] = [dirty["tasks"][2]["id"], "exp2091-retired"]
    dirty["tasks"][3]["gated_on"] = [
        {"upstream": dirty["tasks"][0]["id"], "artifact_field": "missing", "op": "==", "value": 1}
    ]
    dirty["tasks"][4]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    dirty["tasks"][5]["prompt"] = "Run command: broken"
    dirty_result = exp6260.validate_v540_roadmap_data(dirty, {2091})

    assert dirty_result["task_id_validation"]["expected_order"] is False
    assert dirty_result["id_collision_count"] == 1
    assert dirty_result["retired_dependency_count"] == 1
    assert dirty_result["dependency_validation"]["ok"] is False
    assert dirty_result["gated_on_validation"]["ok"] is False
    assert dirty_result["prior_failure_validation"]["ok"] is False
    assert dirty_result["model_policy_validation"]["ok"] is False
    assert dirty_result["prompt_contract_validation"]["ok"] is False


def test_protected_hash_comparison_detects_mutation(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6260-5: protected file hashes prove byte identity."""

    protected = (
        Path("research-roadmap.yaml"),
        Path("scripts/research_conductor.py"),
        Path("research-roadmap-next.yaml"),
    )
    for rel in protected[:2]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rel.as_posix(), encoding="utf-8")
    before = exp6260.protected_hashes(tmp_path, protected)

    (tmp_path / "scripts/research_conductor.py").write_text("mutated", encoding="utf-8")
    receipt = exp6260.protected_files_unchanged(tmp_path, before, protected)

    assert receipt["unchanged"] is False
    assert receipt["paths"]["research-roadmap.yaml"]["unchanged"] is True
    assert receipt["paths"]["scripts/research_conductor.py"]["unchanged"] is False
    assert receipt["paths"]["research-roadmap-next.yaml"]["before_sha256"] is None


def test_report_builder_records_v539_to_v540_handoff() -> None:
    """SCENARIO-INFRA-6260-6: report validation is machine-checkable."""

    before = exp6260.protected_hashes(REPO)
    report = exp6260.build_report(
        REPO,
        date="20260809",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=before,
        git_status_before=["M fixture"],
        git_status_after_tests=["M after-test-fixture"],
        started_at=0.0,
    )

    assert exp6260.validate_report(report) == []
    assert report["task_count"] == 12
    assert report["retired_dependency_count"] == 0
    assert report["id_collision_count"] == 0
    assert report["exp6228_nonterminal_classification"]["terminal"] is False
    assert report["exp6228_nonterminal_classification"]["classification"] == "unknown"
    assert (
        report["v539_task_terminal_matrix"]["exp6228-supervised-three-family-runtime-endurance"][
            "status_raw"
        ]
        == "preconditions_recorded"
    )
    assert (
        report["missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts"][
            "nonterminal"
        ]
        >= 1
    )
    assert report["v540_roadmap_path_and_hash"]["milestone"] == exp6260.MILESTONE_V540
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert report["preconditions_checked"]["git_status_after_tests"] == ["M after-test-fixture"]

    clean_status_report = exp6260.build_report(
        REPO,
        date="20260809",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=before,
        git_status_before=[],
        git_status_after_tests=[],
        started_at=0.0,
    )
    assert clean_status_report["preconditions_checked"]["git_status_before"] == []
    assert clean_status_report["preconditions_checked"]["git_status_after_tests"] == []

    blocked = exp6260.build_report(
        REPO,
        date="20260809",
        command_receipts=[{"command": "timed-out", "exit_code": 124}],
        before_hashes=before,
        git_status_before=["M fixture"],
        started_at=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"] == (
        "blocked: one or more recorded validation commands failed or timed out"
    )


def test_report_validation_and_helper_edges(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6260-6: invalid reports and malformed inputs fail closed."""

    assert exp6260.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("a: 1\n", encoding="utf-8")
    assert exp6260.read_yaml_mapping(yaml_path) == {"a": 1}
    yaml_path.write_text("- 1\n", encoding="utf-8")
    assert exp6260.read_yaml_mapping(yaml_path) == {}
    yaml_path.write_text("a: [\n", encoding="utf-8")
    assert exp6260.read_yaml_mapping(yaml_path) == {}

    payload, meta = exp6260.read_json_mapping(tmp_path / "missing.json")
    assert payload == {}
    assert meta["error"] == "missing"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    payload, meta = exp6260.read_json_mapping(malformed)
    assert payload == {}
    assert str(meta["error"]).startswith("json_error:")
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]", encoding="utf-8")
    payload, meta = exp6260.read_json_mapping(non_mapping)
    assert payload == {}
    assert meta["error"] == "json_not_mapping"

    assert exp6260.exp_number("no experiment") is None
    assert exp6260.same_number_aliases(tmp_path, "not-an-exp", Path("results/x.json")) == []
    assert exp6260.classify_v539_declared_tasks(tmp_path, {"bad": "row"}) == {}
    assert exp6260.required_artifact_fields_from_prompt("no block") == set()
    multiline_fields = exp6260.required_artifact_fields_from_prompt(
        "REQUIRED ARTIFACT FIELDS: status,\n  ready_score, other_field.\nCONCRETE STEPS\n"
    )
    assert {"status", "ready_score", "other_field"} <= multiline_fields
    assert exp6260.gate_ok("bad", {"x": {}}, set()) == (False, "gate_not_mapping")
    assert exp6260.gate_ok({"upstream": "x"}, {"x": {}}, set()) == (
        False,
        "missing_artifact_field",
    )
    assert exp6260.gate_ok(
        {"upstream": "x", "artifact_field": "score", "op": "contains", "value": 1},
        {"x": {}},
        {"x": {"score"}},
    ) == (False, "bad_op")
    assert exp6260.gate_ok(
        {"upstream": "missing", "artifact_field": "score", "op": "==", "value": 1},
        {"x": {}},
        {"x": {"score"}},
    ) == (False, "unknown_upstream")
    assert exp6260.gate_ok(
        {"upstream": "x", "artifact_field": "score", "op": "==", "value": 1},
        {"x": {}},
        {"score"},
    ) == (True, None)
    assert exp6260.prior_ok("bad") == (False, "prior_not_mapping")
    assert exp6260.prior_ok(
        {
            "experiment_id": "exp1",
            "verdict": "blocked",
            "addressed_by": "changed",
            "retire_if_same_verdict": False,
        }
    ) == (False, "retire_if_same_verdict_not_true")
    assert (
        exp6260.module_name_for_task({"deliverable": "results/custom-name.json"}) == "custom_name"
    )

    manifest = tmp_path / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        yaml.safe_dump(
            {
                "retired": {"not": "a-list"},
                "retired_experiments": ["bad-row"],
                "retired_extras": [{"experiment_ids": ["exp103-old"]}],
            }
        ),
        encoding="utf-8",
    )
    assert exp6260.load_retired_exp_ids(manifest) == {103}

    next_root = tmp_path / "next"
    next_root.mkdir()
    (next_root / "research-roadmap-next.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": exp6260.MILESTONE_V540,
                "milestone_title": "Next",
                "milestone_doc": "doc.md",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    _data, identity = exp6260.load_v540_roadmap(next_root)
    assert identity["path"] == "research-roadmap-next.yaml"

    bad_root = tmp_path / "bad-root"
    bad_root.mkdir()
    (bad_root / "research-roadmap.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": "2026.08.000",
                "milestone_title": "Old",
                "milestone_doc": "doc.md",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    _data, identity = exp6260.load_v540_roadmap(bad_root)
    assert (
        identity["selection_note"]
        == "V540 roadmap milestone was not found in next or active roadmap"
    )

    dirty_data = {
        "milestone": exp6260.MILESTONE_V540,
        "milestone_title": "Dirty",
        "milestone_doc": "doc.md",
        "tasks": [
            {
                "id": "exp6260-v540-terminal-transition",
                "milestone": exp6260.MILESTONE_V540,
                "deliverable": "not-results.txt",
                "title": "Dirty task",
                "track": "arc",
                "requires_gpu": False,
                "prior_failures": [],
                "prompt": (
                    "REQUIRED ARTIFACT FIELDS: status\n"
                    "TASK\n"
                    "Load an LLM for this fixture.\n"
                    "Run command: broken"
                ),
                "agent_type": "codex",
                "model": "gpt-5.5",
            }
        ],
    }
    dirty_validation = exp6260.validate_v540_roadmap_data(dirty_data, set())
    assert dirty_validation["schema_validation"]["ok"] is False
    assert dirty_validation["prior_failure_validation"]["failures"][0]["reason"] == (
        "missing_prior_failures"
    )
    assert dirty_validation["model_policy_validation"]["forbidden_execution_failures"] == [
        {"task_id": "exp6260-v540-terminal-transition", "reason": "arc_track_task"},
        {
            "task_id": "exp6260-v540-terminal-transition",
            "reason": "prompt_schedules_forbidden_execution",
            "hits": ["load an llm"],
        },
    ]

    report = {field: None for field in exp6260.REQUIRED_ARTIFACT_FIELDS}
    del report["status"]
    errors = exp6260.validate_report(report)
    assert "missing required field: status" in errors
    assert "field_principles is not a mapping" in errors
    assert "field_provenance is not a mapping" in errors
    assert "task_count must be 12" in errors
    assert "retired_dependency_count must be bare 0" in errors
    assert "id_collision_count must be bare 0" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "reproducibility_checksum missing" in errors

    clean = {field: "x" for field in exp6260.REQUIRED_ARTIFACT_FIELDS}
    clean["task_count"] = 12
    clean["retired_dependency_count"] = 0
    clean["id_collision_count"] = 0
    clean["inference_substrate"] = exp6260.INFERENCE_SUBSTRATE
    clean["verifier_is_oracle"] = False
    clean["field_principles"] = dict(exp6260.FIELD_PRINCIPLES)
    clean["field_provenance"] = {
        field: {"sources": ["REQ-INFRA-6260"], "principle": exp6260.FIELD_PRINCIPLES[field]}
        for field in exp6260.REQUIRED_ARTIFACT_FIELDS
    }
    clean["honest_verdict"] = "complete: x"
    clean["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in exp6260.validate_report(clean)

    report_root = tmp_path / "report-root"
    (report_root / "results").mkdir(parents=True)
    (report_root / "results/experiment_6238_v539_adversarial_capstone.json").write_text(
        json.dumps({"roadmap_path_hash_and_task_ids": "bad", "exact_task_artifact_matrix": {}}),
        encoding="utf-8",
    )
    (report_root / "results/operational_retro_2026_08_539.json").write_text(
        json.dumps({"milestone": exp6260.MILESTONE_V539}),
        encoding="utf-8",
    )
    (report_root / "research-roadmap.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": "2026.08.000",
                "milestone_title": "Old",
                "milestone_doc": "doc.md",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    blocked = exp6260.build_report(
        report_root,
        date="20260809",
        command_receipts=[],
        before_hashes=exp6260.protected_hashes(report_root),
        git_status_before=["fixture"],
        collision_receipt_before=exp6260.scan_reserved_id_collisions(report_root),
        input_hashes_before=exp6260.input_hashes(report_root),
        started_at=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["v539_milestone_roadmap_and_hash"]["roadmap_path"] is None


def test_check_roadmap_only_reports_clean_current_contract() -> None:
    """REQ-INFRA-6260: check-roadmap-only validates the current V540 roadmap."""

    result = exp6260.check_roadmap_only(REPO)
    assert result["ok"] is True
