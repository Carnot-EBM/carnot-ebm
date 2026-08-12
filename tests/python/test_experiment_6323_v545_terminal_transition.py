"""Tests for Exp6323 V545 terminal transition.

Spec refs: REQ-INFRA-6323, SCENARIO-INFRA-6323-1,
SCENARIO-INFRA-6323-2, SCENARIO-INFRA-6323-3,
SCENARIO-INFRA-6323-4, SCENARIO-INFRA-6323-5,
SCENARIO-INFRA-6323-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6323_v545_terminal_transition as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def test_req_infra_6323_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6323: the spec names the transition contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6323") :]

    for marker in (
        "SCENARIO-INFRA-6323-1",
        "SCENARIO-INFRA-6323-2",
        "SCENARIO-INFRA-6323-3",
        "SCENARIO-INFRA-6323-4",
        "SCENARIO-INFRA-6323-5",
        "SCENARIO-INFRA-6323-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "python/carnot/terminal_artifacts.py",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenarios_6323_preserve_v544_classes_and_failed_commands() -> None:
    """SCENARIO-INFRA-6323-1 and SCENARIO-INFRA-6323-2: old states stay exact."""

    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    matrix = report["v544_task_terminal_matrix"]
    assert len(matrix) == 13
    assert (
        matrix["exp6312-model-local-representation-surface-preflight"]["terminal_class"] == "null"
    )
    assert matrix["exp6314-three-family-model-local-state-corpus"]["terminal_class"] == "skipped"
    assert (
        matrix["exp6315-model-local-paired-difference-energy-probes"]["terminal_class"] == "missing"
    )
    assert matrix["exp6316-model-local-probe-integrity-audit"]["terminal_class"] == "flagged"
    assert matrix["exp6320-online-self-evolution-safety-audit"]["safety_only"] is True
    assert matrix["exp6321-arc-target-licensed-route-live-shadow-ab"]["shadow_only"] is True
    assert matrix["exp6322-v544-adversarial-capstone"]["terminal_class"] == "blocked"

    counts = report[
        "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts"
    ]
    assert counts["task_count"] == 13
    assert counts["terminal_class_task_count_sum"] == 13
    assert counts["missing"] == 2
    assert counts["nonterminal"] == 2
    assert counts["blocked"] == 2
    assert counts["skipped"] == 1
    assert counts["null"] == 2
    assert counts["flagged"] == 1
    assert counts["ready"] == 2
    assert counts["positive"] == 3
    assert counts["safety_only"] == 1
    assert counts["shadow_only"] == 1

    failures = report["v544_validation_failure_receipts"]
    assert failures["broad_validation"]["failed_count"] == 1
    assert failures["determination_validation"]["failed_count"] == 1
    assert failures["nonzero_exit_codes_by_command"][mod.FULL_PYTEST_COMMAND] == 3
    assert failures["nonzero_exit_codes_by_command"][mod.DETERMINATION_COMMAND] == 1


def test_scenarios_6323_v545_roadmap_contract_is_exact() -> None:
    """SCENARIO-INFRA-6323-3 to SCENARIO-INFRA-6323-5: V545 metadata is checked."""

    data, identity = mod.load_v545_roadmap(REPO)
    retired = mod.load_retired_exp_ids(REPO / "ops/exclusion_manifest.yaml")
    result = mod.validate_v545_roadmap_data(data, retired)

    assert identity["milestone"] == mod.MILESTONE_V545
    assert identity["path"] == mod.MILESTONE_DOC_RELATIVE_PATH.as_posix()
    assert identity["research_roadmap_next_present"] is False
    assert identity["active_roadmap_task_count"] == 7
    assert result["schema_validation"]["ok"] is True
    assert result["task_count"] == 14
    assert result["task_id_validation"]["expected_order"] is True
    assert result["deliverable_validation"]["ok"] is True
    assert result["dependency_validation"]["ok"] is True
    assert result["gated_on_validation"]["ok"] is True
    assert result["prior_failure_validation"]["ok"] is True
    assert result["retired_dependency_count"] == 0
    assert result["id_collision_count"] == 0
    assert result["agent_routing_validation"]["ok"] is False
    assert result["agent_routing_validation"]["missing_structured_route_count"] == 7
    assert result["model_policy_validation"]["ok"] is False
    assert result["prompt_contract_validation"]["ok"] is False
    assert result["prompt_contract_validation"]["available_prompt_count"] == 7
    assert result["prompt_contract_validation"]["forbidden_scope_validation"]["ok"] is True

    deliverables = mod.v545_task_ids_and_deliverables(data)
    assert [row["task_id"] for row in deliverables] == list(mod.EXPECTED_V545_TASK_IDS)
    assert [row["deliverable"] for row in deliverables][0] == (mod.RESULT_RELATIVE_PATH.as_posix())
    assert any(
        set(row["model_specs_named_in_prompt"]) == mod.MANDATED_HEADLINE_GGUF_IDS
        for row in deliverables
        if row["requires_gpu"]
    )


def test_dirty_v545_roadmap_fails_closed() -> None:
    """REQ-INFRA-6323: corrupt V545 metadata cannot pass validation."""

    data, _identity = mod.load_v545_roadmap(REPO)
    dirty = copy.deepcopy(data)
    tasks = dirty["tasks"]

    tasks[1]["id"] = tasks[0]["id"]
    tasks[2]["requires"] = [tasks[2]["id"], "exp6316-model-local-probe-integrity-audit"]
    tasks[3]["deliverable"] = "not-results.txt"
    tasks[4]["gated_on"] = [
        {
            "upstream": tasks[0]["id"],
            "artifact_field": "not_declared",
            "op": "==",
            "value": 1,
        }
    ]
    tasks[5]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    tasks[6]["agent_type"] = "gemini"
    tasks[6]["model"] = "gemini-3.1-pro-preview"
    tasks[7]["agent_type"] = "codex"
    tasks[7]["model"] = "opus"
    tasks[8]["prompt"] = tasks[8]["prompt"].replace(
        "Do NOT push. Do NOT modify scripts/research_conductor.py.",
        "Do NOT push.",
    )
    tasks[9]["prompt"] = tasks[9]["prompt"].replace(
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "Qwen/Qwen3.5-0.8B",
    )
    tasks[10]["prompt"] += "\nSchedule hidden state, KAN, and public ARC re-solve work."

    result = mod.validate_v545_roadmap_data(dirty, {6316})

    assert result["schema_validation"]["ok"] is False
    assert result["task_id_validation"]["expected_order"] is False
    assert result["deliverable_validation"]["ok"] is False
    assert result["dependency_validation"]["ok"] is False
    assert result["gated_on_validation"]["ok"] is False
    assert result["prior_failure_validation"]["ok"] is False
    assert result["agent_routing_validation"]["ok"] is False
    assert result["model_policy_validation"]["ok"] is False
    assert result["prompt_contract_validation"]["ok"] is False
    assert result["prompt_contract_validation"]["forbidden_scope_validation"]["ok"] is False
    assert result["retired_dependency_count"] == 1
    assert result["id_collision_count"] == 1


def test_edge_cases_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFRA-6323: malformed inputs and stray reserved IDs stay auditable."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(bad_json)[1]["error"].startswith("json_error:")

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(scalar_json)[1]["error"] == "json_not_mapping"
    assert mod._roadmap_tasks({"tasks": "not-a-list"}) == []

    data, _identity = mod.load_v545_roadmap(REPO)
    dirty = copy.deepcopy(data)
    dirty["tasks"][0]["prior_failures"] = None
    dirty["tasks"][1]["prior_failures"] = "bad-prior-shape"
    dirty["tasks"][2]["requires_gpu"] = True
    dirty["tasks"][2]["prompt"] += "\nUse Bad/Unexpected-GGUF local GGUF model."
    result = mod.validate_v545_roadmap_data(dirty, set())
    assert any(
        row.get("reason") == "prior_failures_not_list"
        for row in result["prior_failure_validation"]["failures"]
    )
    assert any(
        row.get("reason") == "non_mandated_gguf_id"
        for row in result["model_policy_validation"]["failures"]
    )

    assert mod.classify_v544_tasks(tmp_path, {"declared_task_ids_and_deliverables": "bad"}) == {}
    assert mod.classify_v544_tasks(tmp_path, {"declared_task_ids_and_deliverables": [None]}) == {}
    assert mod._experiment_paths(tmp_path) == []
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "experiment_6324_stray.json").write_text("{}", encoding="utf-8")
    collision = mod.scan_reserved_id_collisions(tmp_path, set())
    assert collision["unexpected_reserved_paths_by_exp_id"]["6324"] == [
        "results/experiment_6324_stray.json"
    ]

    def fake_read_json_mapping(path: Path) -> tuple[dict[str, object], dict[str, object]]:
        if path.name == mod.V544_CAPSTONE_RELATIVE_PATH.name:
            return (
                {"declared_task_ids_and_deliverables": [], "roadmap_path_and_hash": "bad"},
                {
                    "path": path.as_posix(),
                    "present": True,
                    "loadable": True,
                    "sha256": "fixture",
                    "error": None,
                },
            )
        return {}, {"path": path.as_posix(), "present": False, "sha256": None}

    monkeypatch.setattr(mod, "read_json_mapping", fake_read_json_mapping)
    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )
    assert report["v544_roadmap_path_and_hash"]["recorded_roadmap"] == {}


def test_report_schema_write_and_validation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6323-6: output is validated, checksummed, and atomic."""

    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert report["status"] == "blocked"
    assert report["task_count"] == 14
    assert report["retired_dependency_count"] == 0
    assert report["id_collision_count"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert report["preconditions_checked"]["research_roadmap_next_was_not_activated"] is True
    assert report["honest_verdict"].startswith("blocked:")

    bad = copy.deepcopy(report)
    bad["task_count"] = 13
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "task_count must be 14" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["retired_dependency_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "retired_dependency_count must be 0" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert any("missing field_principles entry" in err for err in mod.validate_report(bad))

    bad = copy.deepcopy(report)
    bad["missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts"][
        "task_count"
    ] = 12
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "V544 counts task_count must be 13" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts"][
        "terminal_class_task_count_sum"
    ] = 12
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "terminal class counts must conserve 13 V544 tasks" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts"][
        "count_principles"
    ] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert any("missing count principle" in err for err in mod.validate_report(bad))

    bad = copy.deepcopy(report)
    bad["honest_verdict"] = "transition ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["reproducibility_checksum"] = "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=REPO, write=True, command_receipts=None: {
            "status": f"complete-{date}"
        },
    )
    assert mod.main(["--date", "20260812"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_run_paths_and_external_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFRA-6323: CLI helpers preserve receipts and write only the result."""

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    receipt_path.write_text(json.dumps({"focused": 0, "broad": 3}), encoding="utf-8")
    assert mod.read_external_test_receipts() == [
        {"command": "focused", "exit_code": 0},
        {"command": "broad", "exit_code": 3},
    ]

    receipt_path.write_text(
        json.dumps([{"command": "coverage", "exit_code": 0}, {"bad": 1}]),
        encoding="utf-8",
    )
    assert mod.read_external_test_receipts() == [{"command": "coverage", "exit_code": 0}]

    receipt_path.write_text("{bad", encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    writes: list[dict[str, object]] = []
    original_write_report = mod.write_report
    monkeypatch.setattr(mod, "git_status_lines", lambda root: [" M fixture"])

    def fake_write_report(
        report: dict[str, object], root: Path = REPO, *, env: object = None
    ) -> Path:
        writes.append(report)
        return tmp_path / mod.RESULT_RELATIVE_PATH.name

    monkeypatch.setattr(mod, "write_report", fake_write_report)
    run_report = mod.run(
        date="20260812",
        root=REPO,
        write=True,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert writes and run_report["status"] == "blocked"

    no_write_report = mod.run(
        date="20260812",
        root=REPO,
        write=False,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert no_write_report["status"] == "blocked"

    with pytest.raises(ValueError, match="invalid Exp6323 report"):
        original_write_report({"status": "complete"}, REPO)
