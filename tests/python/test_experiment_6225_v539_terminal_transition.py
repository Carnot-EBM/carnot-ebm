"""Spec refs: REQ-INFRA-6225, SCENARIO-INFRA-6225-1,
SCENARIO-INFRA-6225-2, SCENARIO-INFRA-6225-3,
SCENARIO-INFRA-6225-4, SCENARIO-INFRA-6225-5,
SCENARIO-INFRA-6225-6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6225_v539_terminal_transition as exp6225


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _write_artifact(path: Path, status: str = "complete") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "honest_verdict": f"{status}: exact fixture",
                "duration_s": 1.0,
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "verifier_is_oracle": False,
                "reproducibility_checksum": "sha256:test",
            }
        ),
        encoding="utf-8",
    )


def test_openspec_names_req_6225_and_scenarios() -> None:
    """REQ-INFRA-6225: OpenSpec records the transition contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6225") :]
    for token in (
        "REQ-INFRA-6225",
        "SCENARIO-INFRA-6225-1",
        "SCENARIO-INFRA-6225-2",
        "SCENARIO-INFRA-6225-3",
        "SCENARIO-INFRA-6225-4",
        "SCENARIO-INFRA-6225-5",
        "SCENARIO-INFRA-6225-6",
        "experiment_6225_v539_terminal_transition.py",
    ):
        assert token in section


def test_exact_declared_path_outranks_same_number_alias(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6225-1: exact paths are classified, aliases are ignored."""

    exact = tmp_path / "results/experiment_9000_exact.json"
    alias = tmp_path / "results/experiment_9000_sidecar.json"
    _write_artifact(exact, status="complete")
    _write_artifact(alias, status="blocked")

    rows = exp6225.classify_declared_deliverables(
        tmp_path,
        [
            {
                "task_id": "exp9000-exact",
                "title": "Exact fixture",
                "deliverable": "results/experiment_9000_exact.json",
            }
        ],
    )

    row = rows["exp9000-exact"]
    assert row["classification"] == "complete"
    assert row["present"] is True
    assert row["same_number_alias_used"] is False
    assert row["same_number_alias_candidates_ignored"] == ["results/experiment_9000_sidecar.json"]


def test_missing_declared_artifact_fails_closed_despite_alias(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6225-2: missing exact deliverables stay nonterminal."""

    _write_artifact(tmp_path / "results/experiment_9001_sidecar.json")
    rows = exp6225.classify_declared_deliverables(
        tmp_path,
        [
            {
                "task_id": "exp9001-missing",
                "title": "Missing exact fixture",
                "deliverable": "results/experiment_9001_declared.json",
            }
        ],
    )

    row = rows["exp9001-missing"]
    assert row["classification"] == "missing"
    assert row["terminal"] is False
    assert row["present"] is False
    assert row["same_number_alias_used"] is False
    assert row["same_number_alias_candidates_ignored"] == ["results/experiment_9001_sidecar.json"]


def test_v539_roadmap_identity_and_collision_detection() -> None:
    """SCENARIO-INFRA-6225-3: duplicate ids and retired dependencies fail closed."""

    data = yaml.safe_load((REPO / "research-roadmap.yaml").read_text(encoding="utf-8"))
    clean = exp6225.validate_v539_roadmap_data(data, retired_exp_ids=set())
    assert clean["task_id_validation"]["expected_order"] is True
    assert clean["task_count"] == 14
    assert clean["id_collision_count"] == 0
    assert clean["retired_dependency_count"] == 0

    duplicate = json.loads(json.dumps(data))
    duplicate["tasks"][1]["id"] = duplicate["tasks"][0]["id"]
    duplicate["tasks"][2]["requires"] = ["exp2091-retired"]
    dirty = exp6225.validate_v539_roadmap_data(duplicate, retired_exp_ids={2091})
    assert dirty["id_collision_count"] == 1
    assert dirty["retired_dependency_count"] == 1
    assert dirty["task_id_validation"]["expected_order"] is False


def test_prompt_contracts_for_current_v539_roadmap() -> None:
    """SCENARIO-INFRA-6225-4: LLM, ARC, and ending contracts are mechanical."""

    data = yaml.safe_load((REPO / "research-roadmap.yaml").read_text(encoding="utf-8"))
    validation = exp6225.validate_v539_roadmap_data(data, retired_exp_ids=set())

    assert validation["model_policy_validation"]["ok"] is True
    assert validation["model_policy_validation"]["llm_task_failures"] == []
    assert validation["model_policy_validation"]["arc_live_path_failures"] == []
    assert validation["prompt_contract_validation"]["ok"] is True
    assert validation["prompt_contract_validation"]["missing_required_ending"] == []


def test_protected_hash_comparison_detects_mutation(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6225-5: protected file hashes prove byte identity."""

    protected = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))
    for rel in protected:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rel.as_posix(), encoding="utf-8")
    before = exp6225.protected_hashes(tmp_path, protected)

    (tmp_path / "scripts/research_conductor.py").write_text("mutated", encoding="utf-8")
    receipt = exp6225.protected_files_unchanged(tmp_path, before, protected)

    assert receipt["unchanged"] is False
    assert receipt["paths"]["research-roadmap.yaml"]["unchanged"] is True
    assert receipt["paths"]["scripts/research_conductor.py"]["unchanged"] is False


def test_artifact_schema_requires_principles_and_checksum() -> None:
    """SCENARIO-INFRA-6225-6: report validation is machine-checkable."""

    report = {field: f"fixture-{field}" for field in exp6225.REQUIRED_ARTIFACT_FIELDS}
    report["status"] = "complete"
    report["retired_dependency_count"] = 0
    report["id_collision_count"] = 0
    report["inference_substrate"] = exp6225.INFERENCE_SUBSTRATE
    report["verifier_is_oracle"] = False
    report["field_principles"] = dict(exp6225.FIELD_PRINCIPLES)
    report["field_provenance"] = {
        field: {"sources": ["REQ-INFRA-6225"], "principle": exp6225.FIELD_PRINCIPLES[field]}
        for field in exp6225.REQUIRED_ARTIFACT_FIELDS
    }
    report["duration_s"] = 1.0
    report["reproducibility_checksum"] = ""
    report["honest_verdict"] = "complete: fixture"
    report["reproducibility_checksum"] = exp6225.payload_checksum(report)

    assert exp6225.validate_report(report) == []

    broken = dict(report)
    broken["field_principles"] = {
        key: value for key, value in report["field_principles"].items() if key != "status"
    }
    assert "missing field_principles entry: status" in exp6225.validate_report(broken)


def test_report_builder_records_current_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFRA-6225: report builder emits the required transition artifact fields."""

    def fake_summary(root: Path, rel: Path) -> dict[str, object]:
        return {
            "command": f"summarize {rel.as_posix()}",
            "exit_code": 0,
            "classification": "passed",
            "stdout_tail": "LIVE re-check: clean",
            "stderr_tail": "",
        }

    monkeypatch.setattr(exp6225, "_summary_receipt", fake_summary)
    before = exp6225.protected_hashes(REPO)
    report = exp6225.build_report(
        REPO,
        date="20260809",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=before,
        started_at=0.0,
    )

    assert exp6225.validate_report(report) == []
    assert report["task_count"] == 14
    assert report["retired_dependency_count"] == 0
    assert report["id_collision_count"] == 0
    assert (
        report["v538_task_terminal_matrix"]["exp6224-v538-adversarial-capstone"]["classification"]
        == "complete"
    )
    assert report["blocked_skipped_partial_flagged_and_ready_counts"]["flagged"] == 2
    assert report["architecture_staleness_receipt"]["stale_by_30_day_rule"] is True
    assert (
        report["research_complete_duplicate_record_note"]["action"]
        == "recorded_only_not_deduplicated"
    )


def test_helper_edge_cases_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6225-2: malformed inputs return structured failures."""

    assert exp6225.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("a: 1\n", encoding="utf-8")
    assert exp6225.read_yaml_mapping(yaml_path) == {"a": 1}

    payload, meta = exp6225.read_json_mapping(tmp_path / "missing.json")
    assert payload == {}
    assert meta["error"] == "missing"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    payload, meta = exp6225.read_json_mapping(malformed)
    assert payload == {}
    assert str(meta["error"]).startswith("json_error:")
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]", encoding="utf-8")
    payload, meta = exp6225.read_json_mapping(non_mapping)
    assert payload == {}
    assert meta["error"] == "json_not_mapping"

    assert exp6225.exp_number("no experiment here") is None
    assert exp6225.same_number_aliases(tmp_path, "not-an-exp", Path("results/x.json")) == []
    assert (
        exp6225._module_name_for_task({"deliverable": "results/custom-name.json"}) == "custom_name"
    )


def test_temp_research_complete_and_retired_id_helpers(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6225-1: archived V538 tasks and retired ids are parsed."""

    (tmp_path / "research-complete.yaml").write_text(
        yaml.safe_dump(
            {
                "milestones": [
                    {"id": "other", "tasks": []},
                    {
                        "id": exp6225.MILESTONE_V538,
                        "tasks": [
                            {
                                "id": "exp9100-fixture",
                                "title": "Fixture",
                                "deliverable": "results/experiment_9100_fixture.json",
                            }
                        ],
                    },
                    {
                        "id": exp6225.MILESTONE_V538,
                        "tasks": [
                            {
                                "id": "exp9101-ignored",
                                "title": "Ignored duplicate",
                                "deliverable": "results/experiment_9101_fixture.json",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_artifact(tmp_path / "results/experiment_9100_fixture.json")
    matrix = exp6225.build_v538_task_terminal_matrix(
        tmp_path,
        {
            "exact_artifact_paths_hashes_and_terminal_classifications": {
                "exp9100-fixture": {
                    "flag_count": 3,
                    "critical_adversarial_flag_count": 1,
                }
            }
        },
    )
    assert list(matrix) == ["exp9100-fixture"]
    assert matrix["exp9100-fixture"]["capstone_flag_count"] == 3
    assert matrix["exp9100-fixture"]["capstone_critical_flag_count"] == 1

    manifest = tmp_path / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        yaml.safe_dump(
            {
                "retired": [{"experiment_id": 100}],
                "retired_experiments": [{"experiment_id": "exp101-old"}],
                "retired_extras": [{"experiment_ids": ["102", "exp103-old"]}],
            }
        ),
        encoding="utf-8",
    )
    assert exp6225.load_retired_exp_ids(manifest) == {100, 101, 102, 103}

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert exp6225.build_v538_task_terminal_matrix(empty_root, {}) == {}

    manifest.write_text(
        yaml.safe_dump({"retired": {"not": "a-list"}, "retired_experiments": ["bad-row"]}),
        encoding="utf-8",
    )
    assert exp6225.load_retired_exp_ids(manifest) == set()


def test_dirty_roadmap_validation_reports_each_contract_failure() -> None:
    """SCENARIO-INFRA-6225-3: malformed gates, priors, model rules, and prompts fail."""

    task = {
        "id": "exp6225-v539-terminal-transition",
        "milestone": exp6225.MILESTONE_V539,
        "deliverable": "results/experiment_6225_v539_terminal_transition.json",
        "title": "Fixture",
        "track": "arc",
        "requires_gpu": True,
        "requires": ["exp6225-v539-terminal-transition", "exp2091-retired"],
        "gated_on": [
            {"upstream": "missing", "artifact_field": "score", "op": "contains", "value": 1}
        ],
        "prior_failures": [{"experiment_id": "", "verdict": "", "addressed_by": ""}],
        "prompt": "MODEL_SPECS:\n- hf_id: legacy/model-GGUF\nRun command: broken",
    }
    data = {
        "milestone": exp6225.MILESTONE_V539,
        "milestone_title": "Fixture",
        "milestone_doc": "doc.md",
        "tasks": [task],
    }

    result = exp6225.validate_v539_roadmap_data(data, retired_exp_ids={2091})

    assert result["schema_validation"]["ok"] is False
    assert result["dependency_validation"]["ok"] is False
    assert result["gated_on_validation"]["failures"][0]["reason"] == "bad_op"
    assert result["prior_failure_validation"]["ok"] is False
    assert result["model_policy_validation"]["llm_task_failures"][0]["task_id"] == task["id"]
    assert result["model_policy_validation"]["arc_live_path_failures"]
    assert result["prompt_contract_validation"]["missing_required_ending"] == [task["id"]]
    assert result["retired_dependency_count"] == 1

    assert exp6225._gate_ok("bad", set()) == (False, "gate_not_mapping")
    assert exp6225._gate_ok({"upstream": "x"}, {"x"}) == (False, "missing_artifact_field")
    assert exp6225._gate_ok(
        {"upstream": "missing", "artifact_field": "score", "op": "==", "value": 1},
        {"x"},
    ) == (False, "unknown_upstream")
    assert exp6225._prior_ok("bad") == (False, "prior_not_mapping")
    assert exp6225._prior_ok(
        {
            "experiment_id": "exp1",
            "verdict": "blocked",
            "addressed_by": "changed",
            "retire_if_same_verdict": False,
        }
    ) == (False, "retire_if_same_verdict_not_true")

    no_prior = json.loads(json.dumps(data))
    no_prior["tasks"][0]["prior_failures"] = []
    assert (
        exp6225.validate_v539_roadmap_data(no_prior, retired_exp_ids=set())[
            "prior_failure_validation"
        ]["failures"][0]["reason"]
        == "missing_prior_failures"
    )


def test_architecture_and_validation_error_branches(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6225-6: stale checks and schema errors stay explicit."""

    arch = tmp_path / "_bmad/architecture.md"
    arch.parent.mkdir(parents=True, exist_ok=True)
    arch.write_text("# Architecture\n", encoding="utf-8")
    receipt = exp6225.architecture_staleness(tmp_path, "20260809")
    assert receipt["last_reconciled"] is None
    assert receipt["stale_by_30_day_rule"] is True

    report = {field: None for field in exp6225.REQUIRED_ARTIFACT_FIELDS}
    del report["status"]
    errors = exp6225.validate_report(report)
    assert "missing required field: status" in errors
    assert "field_principles is not a mapping" in errors
    assert "field_provenance is not a mapping" in errors
    assert "wrong inference_substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "retired_dependency_count must be bare 0" in errors
    assert "id_collision_count must be bare 0" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "reproducibility_checksum missing" in errors

    clean = {field: "x" for field in exp6225.REQUIRED_ARTIFACT_FIELDS}
    clean["retired_dependency_count"] = 0
    clean["id_collision_count"] = 0
    clean["inference_substrate"] = exp6225.INFERENCE_SUBSTRATE
    clean["verifier_is_oracle"] = False
    clean["field_principles"] = dict(exp6225.FIELD_PRINCIPLES)
    clean["field_provenance"] = {
        field: {"sources": ["REQ-INFRA-6225"]} for field in exp6225.REQUIRED_ARTIFACT_FIELDS
    }
    clean["duration_s"] = 1.0
    clean["honest_verdict"] = "complete: x"
    clean["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in exp6225.validate_report(clean)


def test_report_builder_blocked_and_bad_old_graph_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6225-6: report builder keeps blocked branches explicit."""

    def fake_summary(root: Path, rel: Path) -> dict[str, object]:
        return {"command": rel.as_posix(), "exit_code": 0, "classification": "passed"}

    monkeypatch.setattr(exp6225, "_summary_receipt", fake_summary)
    blocked = exp6225.build_report(
        REPO,
        date="20260809",
        command_receipts=[],
        before_hashes={"research-roadmap.yaml": "sha256:not-current"},
        started_at=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["protected_files_unchanged"]["unchanged"] is False

    (tmp_path / "results").mkdir()
    (tmp_path / "results/experiment_6224_v538_adversarial_capstone.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "honest_verdict": "complete: bad graph fixture",
                "declared_task_ids_and_deliverables": "not-a-mapping",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "results/operational_retro_2026_08_538.json").write_text(
        json.dumps({"milestone": exp6225.MILESTONE_V538, "summary": "fixture"}),
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": exp6225.MILESTONE_V539,
                "milestone_title": "Fixture",
                "milestone_doc": "doc.md",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    report = exp6225.build_report(tmp_path, date="20260809", command_receipts=[], started_at=0.0)
    assert report["v538_milestone_and_roadmap_hash"]["capstone_declared_roadmap_path"] is None


def test_check_roadmap_only_reports_clean_current_contract() -> None:
    """SCENARIO-INFRA-6225-4: check-roadmap-only validates current prompts."""

    result = exp6225.check_roadmap_only(REPO)
    assert result["ok"] is True
