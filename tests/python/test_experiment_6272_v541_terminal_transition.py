"""Tests for Exp6272 V541 terminal transition.

Spec refs: REQ-INFRA-6272, SCENARIO-INFRA-6272-1,
SCENARIO-INFRA-6272-2, SCENARIO-INFRA-6272-3,
SCENARIO-INFRA-6272-4, SCENARIO-INFRA-6272-5,
SCENARIO-INFRA-6272-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from carnot import experiment_6272_v541_terminal_transition as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _terminal_payload(
    status: str = "complete",
    verdict: str = "complete: fixture",
    **extra: object,
) -> dict[str, object]:
    return {
        "status": status,
        "honest_verdict": verdict,
        "duration_s": 1.0,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "verifier_is_oracle": False,
        "reproducibility_checksum": "sha256:fixture",
        **extra,
    }


def test_spec_declares_req_6272_fields_and_scenarios() -> None:
    """REQ-INFRA-6272: OpenSpec records the V541 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6272") :]

    for token in (
        "REQ-INFRA-6272",
        "SCENARIO-INFRA-6272-1",
        "SCENARIO-INFRA-6272-2",
        "SCENARIO-INFRA-6272-3",
        "SCENARIO-INFRA-6272-4",
        "SCENARIO-INFRA-6272-5",
        "SCENARIO-INFRA-6272-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_v540_exact_path_classification_ignores_receipts_and_aliases(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6272-1: exact V540 paths outrank conductor receipts."""

    declared = tmp_path / "results/experiment_6266_declared.json"
    alias = tmp_path / "results/experiment_6266_complete_alias.json"
    _write_json(declared, _terminal_payload("in_progress", "in_progress"))
    _write_json(alias, _terminal_payload())
    matrix = {
        "exp6266-family-task-holdout-csl-audit": {
            "task_id": "exp6266-family-task-holdout-csl-audit",
            "title": "Holdout audit",
            "track": "continuous_learning",
            "declared_deliverable": "results/experiment_6266_declared.json",
            "terminal_class": "missing",
        },
        "exp6267-gate-skip": {
            "title": "Skip",
            "declared_deliverable": "results/experiment_6267_missing.json",
        },
    }
    receipts = {
        "exp6266-family-task-holdout-csl-audit": "bad receipt",
        "exp6267-gate-skip": {"status": "GATE_BLOCK"},
    }

    rows = mod.classify_v540_declared_tasks(tmp_path, matrix, receipts)

    row = rows["exp6266-family-task-holdout-csl-audit"]
    assert row["terminal_class"] == "running"
    assert row["terminal"] is False
    assert row["receipt_override_attempted"] is False
    assert row["receipt_overrode"] is False
    assert row["same_number_alias_used"] is False
    assert row["same_number_alias_candidates_ignored"] == [
        "results/experiment_6266_complete_alias.json"
    ]

    missing = rows["exp6267-gate-skip"]
    assert missing["terminal_class"] == "missing"
    assert missing["receipt_override_attempted"] is True
    assert missing["terminal"] is False


def test_validation_receipts_preserve_focused_passes_and_broad_failures(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6272-2: focused and broad receipts stay separate."""

    deliverable = tmp_path / "results/experiment_6264_fixture.json"
    _write_json(
        deliverable,
        _terminal_payload(
            "complete_null",
            "complete_null: broad suite failed",
            test_commands=[
                ".venv/bin/pytest tests/python/test_experiment_6264_fixture.py -q",
                ".venv/bin/coverage report --fail-under=100",
                ".venv/bin/pytest tests/python -q",
            ],
            test_exit_codes={
                ".venv/bin/pytest tests/python/test_experiment_6264_fixture.py -q": 0,
                ".venv/bin/coverage report --fail-under=100": 0,
                ".venv/bin/pytest tests/python -q": 2,
            },
        ),
    )
    matrix = {
        "exp6264-energy-familiarity-memory-gate": {
            "declared_deliverable": "results/experiment_6264_fixture.json"
        },
        "bad-row": "ignored",
        "malformed": {"declared_deliverable": "results/malformed.json"},
        "exp6266-family-task-holdout-csl-audit": {
            "declared_deliverable": "results/experiment_6266_missing.json"
        },
    }
    (tmp_path / "results/malformed.json").write_text("{", encoding="utf-8")

    receipts = mod.focused_and_broad_validation_receipts_by_task(tmp_path, matrix)

    row = receipts["exp6264-energy-familiarity-memory-gate"]
    assert row["focused"]["command_count"] == 2
    assert row["focused"]["failed_count"] == 0
    assert row["broad"]["command_count"] == 1
    assert row["broad"]["failed_count"] == 1
    assert row["broad"]["commands"][0]["exit_code"] == 2
    assert row["artifact_classification"] == "null"

    missing = receipts["exp6266-family-task-holdout-csl-audit"]
    assert missing["receipt_state"] == "missing_artifact"
    assert missing["focused"]["command_count"] == 0
    assert missing["broad"]["command_count"] == 0
    assert receipts["malformed"]["receipt_state"] == "unloadable_artifact"
    assert "bad-row" not in receipts


def test_current_v541_roadmap_contracts_and_dirty_failures() -> None:
    """SCENARIO-INFRA-6272-3, SCENARIO-INFRA-6272-4, and SCENARIO-INFRA-6272-5."""

    data, identity = mod.load_v541_roadmap(REPO)
    retired = mod.load_retired_exp_ids(REPO / "ops/exclusion_manifest.yaml")
    clean = mod.validate_v541_roadmap_data(data, retired)

    assert identity["milestone"] == mod.MILESTONE_V541
    assert identity["path"] == "research-roadmap.yaml"
    assert identity["research_roadmap_next_present"] is False
    assert clean["schema_validation"]["ok"] is True
    assert clean["task_id_validation"]["expected_order"] is True
    assert clean["task_count"] == 12
    assert clean["retired_dependency_count"] == 0
    assert clean["id_collision_count"] == 0
    assert clean["dependency_validation"]["ok"] is True
    assert clean["gated_on_validation"]["ok"] is True
    assert clean["prior_failure_validation"]["ok"] is True
    assert clean["agent_routing_validation"]["ok"] is True
    assert clean["model_policy_validation"]["ok"] is True
    assert clean["prompt_contract_validation"]["ok"] is True

    dirty = copy.deepcopy(data)
    dirty["tasks"][1]["id"] = dirty["tasks"][0]["id"]
    dirty["tasks"][1]["agent_type"] = "codex"
    dirty["tasks"][1]["model"] = "gpt-5.5"
    dirty["tasks"][2]["requires"] = [dirty["tasks"][2]["id"], "exp2091.retired"]
    dirty["tasks"][2]["deliverable"] = "not-results.txt"
    dirty["tasks"][3]["gated_on"] = [
        {"upstream": dirty["tasks"][0]["id"], "artifact_field": "missing", "op": "==", "value": 1}
    ]
    dirty["tasks"][3]["model"] = "sonnet"
    dirty["tasks"][4]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    dirty["tasks"][8]["agent_type"] = None
    dirty["tasks"][8]["model"] = None
    dirty["tasks"][9]["model"] = "opus"
    dirty["tasks"][10]["agent_type"] = "gemini"
    dirty["tasks"][10]["model"] = "gemini-3.1-pro-preview"
    dirty["tasks"][5]["prior_failures"] = []
    dirty["tasks"][5]["agent_type"] = "codex"
    dirty["tasks"][5]["model"] = "gpt-5.5"
    dirty["tasks"][5]["prompt"] = "Run command: broken"

    dirty_result = mod.validate_v541_roadmap_data(dirty, {2091})

    assert dirty_result["task_id_validation"]["expected_order"] is False
    assert dirty_result["id_collision_count"] == 1
    assert dirty_result["retired_dependency_count"] == 1
    assert dirty_result["dependency_validation"]["ok"] is False
    assert dirty_result["gated_on_validation"]["ok"] is False
    assert dirty_result["prior_failure_validation"]["ok"] is False
    assert dirty_result["agent_routing_validation"]["ok"] is False
    assert dirty_result["model_policy_validation"]["ok"] is False
    assert dirty_result["prompt_contract_validation"]["ok"] is False
    assert dirty_result["schema_validation"]["ok"] is False
    assert any(
        row["reason"] == "empty_prior_failures"
        for row in dirty_result["prior_failure_validation"]["failures"]
    )

    no_prior = {
        "milestone": mod.MILESTONE_V541,
        "milestone_title": "Fixture",
        "milestone_doc": "doc.md",
        "tasks": [
            {
                "id": "exp6273-v541-post-marker-source-scope-freeze",
                "milestone": mod.MILESTONE_V541,
                "deliverable": "results/experiment_6273_fixture.json",
                "title": "Default route fixture",
                "prompt": (
                    "REQUIRED ARTIFACT FIELDS: status\n"
                    "Run command: .venv/bin/python -m carnot.experiment_6273_fixture --date "
                    "{date}\n"
                    "Do NOT push. Do NOT modify scripts/research_conductor.py."
                ),
            }
        ],
    }
    no_prior_result = mod.validate_v541_roadmap_data(no_prior, set())
    assert no_prior_result["prior_failure_validation"]["ok"] is True


def test_reserved_collision_scan_uses_tracked_and_untracked_files(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6272-3: staged ids cannot collide with existing files."""

    owned = tmp_path / "python/carnot/experiment_6272_v541_terminal_transition.py"
    owned.parent.mkdir(parents=True, exist_ok=True)
    owned.write_text("# owned\n", encoding="utf-8")
    unexpected = tmp_path / "results/experiment_6278_old.json"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("{}", encoding="utf-8")
    pycache = tmp_path / "python/carnot/__pycache__/experiment_6273.cpython-312.pyc"
    pycache.parent.mkdir(parents=True, exist_ok=True)
    pycache.write_bytes(b"ignored")

    receipt = mod.scan_reserved_id_collisions(
        tmp_path,
        allowed_reserved_paths={"python/carnot/experiment_6272_v541_terminal_transition.py"},
    )

    assert receipt["reserved_exp_ids"] == list(range(6272, 6284))
    assert receipt["tracked_and_untracked_basis"].startswith("filesystem scan")
    assert receipt["unexpected_reserved_collision_count"] == 1
    assert receipt["unexpected_reserved_paths_by_exp_id"] == {
        "6278": ["results/experiment_6278_old.json"]
    }


def test_report_builder_records_v540_to_v541_handoff() -> None:
    """SCENARIO-INFRA-6272-6: report validation is machine-checkable."""

    before = mod.protected_hashes(REPO)
    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_6272_v541_terminal_transition.py -q --no-cov -n 0",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 2},
        ],
        before_hashes=before,
        git_status_before=["M openspec/capabilities/research-harnesses/spec.md"],
        git_status_after_tests=["M openspec/capabilities/research-harnesses/spec.md"],
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    assert report["status"] == "complete"
    assert report["task_count"] == 12
    assert report["retired_dependency_count"] == 0
    assert report["id_collision_count"] == 0
    assert report["v540_milestone_roadmap_and_hash"]["milestone"] == mod.MILESTONE_V540
    assert report["v541_roadmap_path_and_hash"]["milestone"] == mod.MILESTONE_V541
    assert report["v541_roadmap_path_and_hash"]["research_roadmap_next_present"] is False
    assert (
        report["v540_task_terminal_matrix"]["exp6263-clean-sota-event-replay-bridge"][
            "terminal_class"
        ]
        == "ready"
    )
    assert (
        report["v540_task_terminal_matrix"]["exp6264-energy-familiarity-memory-gate"][
            "terminal_class"
        ]
        == "null"
    )
    assert (
        report["v540_task_terminal_matrix"]["exp6266-family-task-holdout-csl-audit"][
            "terminal_class"
        ]
        == "missing"
    )
    assert (
        report["focused_and_broad_validation_receipts_by_task"][
            "exp6264-energy-familiarity-memory-gate"
        ]["broad"]["failed_count"]
        >= 1
    )
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert report["preconditions_checked"]["git_status_after_tests"] == [
        "M openspec/capabilities/research-harnesses/spec.md"
    ]

    blocked = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 1}],
        before_hashes=before,
        started_at=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")


def test_report_validation_requires_principles_bare_zero_and_checksum() -> None:
    """SCENARIO-INFRA-6272-6: artifact fields are checked before write."""

    report = {field: f"fixture-{field}" for field in mod.REQUIRED_ARTIFACT_FIELDS}
    report["status"] = "complete"
    report["task_count"] = 12
    report["retired_dependency_count"] = 0
    report["id_collision_count"] = 0
    report["inference_substrate"] = mod.INFERENCE_SUBSTRATE
    report["verifier_is_oracle"] = False
    report["field_principles"] = dict(mod.FIELD_PRINCIPLES)
    report["field_provenance"] = {
        field: {"sources": ["REQ-INFRA-6272"], "principle": mod.FIELD_PRINCIPLES[field]}
        for field in mod.REQUIRED_ARTIFACT_FIELDS
    }
    report["test_exit_codes"] = {}
    report["honest_verdict"] = "complete: fixture"
    report["duration_s"] = 1.0
    report["reproducibility_checksum"] = ""
    report["reproducibility_checksum"] = mod.payload_checksum(report)

    assert mod.validate_report(report) == []

    broken = dict(report)
    broken["retired_dependency_count"] = 0.0
    assert "retired_dependency_count must be bare integer 0" in mod.validate_report(broken)

    broken = dict(report)
    broken["id_collision_count"] = 1
    assert "id_collision_count must be bare integer 0" in mod.validate_report(broken)

    broken = dict(report)
    broken["field_principles"] = {}
    assert "missing field_principles entry: status" in mod.validate_report(broken)


def test_write_report_uses_artifact_root_override(tmp_path: Path) -> None:
    """REQ-INFRA-6272: artifact writes are atomic and test-isolated."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})

    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_helper_edges_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6272-4 and SCENARIO-INFRA-6272-6: helpers fail closed."""

    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("a: 1\n", encoding="utf-8")
    assert mod.read_yaml_mapping(yaml_path) == {"a": 1}
    yaml_path.write_text("- 1\n", encoding="utf-8")
    assert mod.read_yaml_mapping(yaml_path) == {}
    yaml_path.write_text("a: [\n", encoding="utf-8")
    assert mod.read_yaml_mapping(yaml_path) == {}

    payload, meta = mod.read_json_mapping(tmp_path / "missing.json")
    assert payload == {}
    assert meta["error"] == "missing"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    payload, meta = mod.read_json_mapping(malformed)
    assert payload == {}
    assert str(meta["error"]).startswith("json_error:")
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]", encoding="utf-8")
    payload, meta = mod.read_json_mapping(non_mapping)
    assert payload == {}
    assert meta["error"] == "json_not_mapping"

    assert mod.exp_number("not an experiment") is None
    assert mod.same_number_aliases(tmp_path, "not-an-exp", Path("results/x.json")) == []
    assert mod.classify_v540_declared_tasks(tmp_path, {"bad": "row"}, {}) == {}
    assert mod.required_artifact_fields_from_prompt("no block") == set()
    fields = mod.required_artifact_fields_from_prompt(
        "REQUIRED ARTIFACT FIELDS: status,\n  ready_score, other_field.\nCONCRETE STEPS\n"
    )
    assert {"status", "ready_score", "other_field"} <= fields

    assert mod.gate_ok("bad", {"x": {}}, {}) == (False, "gate_not_mapping")
    assert mod.gate_ok({"upstream": "x"}, {"x": {}}, {}) == (
        False,
        "missing_artifact_field",
    )
    assert mod.gate_ok(
        {"upstream": "x", "artifact_field": "score", "op": "contains", "value": 1},
        {"x": {}},
        {"x": {"score"}},
    ) == (False, "bad_op")
    assert mod.gate_ok(
        {"upstream": "missing", "artifact_field": "score", "op": "==", "value": 1},
        {"x": {}},
        {"x": {"score"}},
    ) == (False, "unknown_upstream")
    assert mod.gate_ok(
        {"upstream": "x", "artifact_field": "score", "op": "==", "value": 1},
        {"x": {}},
        {"score"},
    ) == (True, None)

    assert mod.prior_ok("bad") == (False, "prior_not_mapping")
    assert mod.prior_ok(
        {
            "experiment_id": "exp1",
            "verdict": "blocked",
            "addressed_by": "changed",
            "retire_if_same_verdict": False,
        }
    ) == (False, "retire_if_same_verdict_not_true")
    assert mod.module_name_for_task({"deliverable": "results/custom-name.json"}) == "custom_name"
    assert mod._artifact_command_rows(
        {"test_commands": ["cmd"], "test_exit_codes": {"cmd": "not-an-int"}}
    ) == [{"command": "cmd", "exit_code": 1}]

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
    assert mod.load_retired_exp_ids(manifest) == {103}

    next_root = tmp_path / "next"
    next_root.mkdir()
    (next_root / "research-roadmap-next.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": mod.MILESTONE_V541,
                "milestone_title": "Next",
                "milestone_doc": "doc.md",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    _data, identity = mod.load_v541_roadmap(next_root)
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
    _data, identity = mod.load_v541_roadmap(bad_root)
    assert identity["selection_note"] == "V541 roadmap milestone was not found"

    invalid = {"status": "complete"}
    errors = mod.validate_report(invalid)
    assert "missing required field: v540_milestone_roadmap_and_hash" in errors
    assert "field_principles is not a mapping" in errors
    assert "field_provenance is not a mapping" in errors
    assert "task_count must be 12" in errors
    assert "wrong inference_substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "reproducibility_checksum missing" in errors

    try:
        mod.write_report({"status": "complete"}, tmp_path)
    except ValueError as exc:
        assert "invalid Exp6272 report" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid report unexpectedly wrote")

    report_root = tmp_path / "report-root"
    (report_root / "results").mkdir(parents=True)
    (report_root / "results/experiment_6271_v540_adversarial_capstone.json").write_text(
        json.dumps(
            {
                "milestone_roadmap_path_and_hash": "not-a-mapping",
                "exact_declared_deliverable_matrix": {},
                "conductor_receipt_matrix": {},
            }
        ),
        encoding="utf-8",
    )
    (report_root / "results/operational_retro_2026_08_540.json").write_text(
        json.dumps({"milestone": mod.MILESTONE_V540}),
        encoding="utf-8",
    )
    (report_root / "research-roadmap.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": mod.MILESTONE_V541,
                "milestone_title": "Fixture",
                "milestone_doc": "doc.md",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    root_report = mod.build_report(
        report_root,
        date="20260810",
        command_receipts=[],
        before_hashes=mod.protected_hashes(report_root),
        collision_receipt_before=mod.scan_reserved_id_collisions(report_root),
        input_hashes_before=mod.input_hashes(report_root),
        git_status_before=[],
        started_at=0.0,
    )
    assert root_report["v540_milestone_roadmap_and_hash"]["recorded_roadmap_path"] is None


def test_check_roadmap_only_reports_clean_current_contract() -> None:
    """REQ-INFRA-6272: check-roadmap-only validates the current V541 roadmap."""

    result = mod.check_roadmap_only(REPO)
    assert result["ok"] is True
