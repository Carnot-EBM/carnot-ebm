"""Tests for Exp6284 V542 terminal transition.

Spec refs: REQ-INFRA-6284, SCENARIO-INFRA-6284-1,
SCENARIO-INFRA-6284-2, SCENARIO-INFRA-6284-3,
SCENARIO-INFRA-6284-4, SCENARIO-INFRA-6284-5.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6284_v542_terminal_transition as mod
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


def test_spec_declares_req_6284_fields_and_scenarios() -> None:
    """REQ-INFRA-6284: OpenSpec records the V542 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6284") :]

    for token in (
        "REQ-INFRA-6284",
        "SCENARIO-INFRA-6284-1",
        "SCENARIO-INFRA-6284-2",
        "SCENARIO-INFRA-6284-3",
        "SCENARIO-INFRA-6284-4",
        "SCENARIO-INFRA-6284-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_v541_exact_path_classification_ignores_receipts_and_raw_aliases(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6284-1: exact V541 paths outrank conductor receipts."""

    declared = tmp_path / "results/experiment_6275_declared.json"
    alias = tmp_path / "results/experiment_6275_raw_receipt.json"
    _write_json(declared, _terminal_payload("in_progress", "in_progress"))
    _write_json(alias, _terminal_payload("complete", "complete: alias"))
    capstone_matrix = {
        "exp6275-flagship-asp-constraint-verification-benchmark": {
            "task_id": "exp6275-flagship-asp-constraint-verification-benchmark",
            "title": "Flagship fixture",
            "track": "live_verification",
            "declared_deliverable": "results/experiment_6275_declared.json",
        },
        "bad-row": "ignored",
    }
    receipts = {
        "exp6275-flagship-asp-constraint-verification-benchmark": {"status": "OK"},
    }

    matrix = mod.classify_v541_declared_tasks(tmp_path, capstone_matrix, receipts)

    row = matrix["exp6275-flagship-asp-constraint-verification-benchmark"]
    assert row["terminal_class"] == "running"
    assert row["terminal"] is False
    assert row["receipt_override_attempted"] is True
    assert row["receipt_overrode"] is False
    assert row["same_number_alias_used"] is False
    assert row["same_number_alias_candidates_ignored"] == [
        "results/experiment_6275_raw_receipt.json"
    ]
    assert "bad-row" not in matrix


def test_raw_evidence_policy_keeps_receipts_below_artifact_eligibility() -> None:
    """SCENARIO-INFRA-6284-2: raw receipts cannot promote flagged artifacts."""

    matrix = {
        "exp6274-asp-energy-semantic-compiler": {
            "terminal": True,
            "terminal_class": "complete",
            "flagged_adversarial_stamped": False,
            "corrigendum_pending_stamped": False,
            "declared_deliverable": "results/experiment_6274_fixture.json",
        },
        "exp6275-flagship-asp-constraint-verification-benchmark": {
            "terminal": True,
            "terminal_class": "flagged",
            "flagged_adversarial_stamped": True,
            "corrigendum_pending_stamped": False,
            "declared_deliverable": "results/experiment_6275_fixture.json",
        },
        "exp6282-arc-mechanic-class-live-router": {
            "terminal": True,
            "terminal_class": "flagged",
            "flagged_adversarial_stamped": True,
            "corrigendum_pending_stamped": True,
            "declared_deliverable": "results/experiment_6282_fixture.json",
        },
    }

    policy = mod.raw_evidence_eligibility_policy(matrix)

    assert policy["exact_path_over_raw_receipt_precedence"] is True
    assert policy["rows"]["exp6274-asp-energy-semantic-compiler"]["artifact_gate_eligible"] is True
    flagship = policy["rows"]["exp6275-flagship-asp-constraint-verification-benchmark"]
    assert flagship["raw_receipts_may_be_reused"] is True
    assert flagship["artifact_gate_eligible"] is False
    assert flagship["artifact_level_quarantine_reason"] == "flagged_artifact"
    arc = policy["rows"]["exp6282-arc-mechanic-class-live-router"]
    assert arc["source_may_be_reused"] is True
    assert arc["artifact_gate_eligible"] is False


def test_current_v542_roadmap_contracts_and_dirty_failures() -> None:
    """SCENARIO-INFRA-6284-3 and SCENARIO-INFRA-6284-4: V542 contracts are checked."""

    data, identity = mod.load_v542_roadmap(REPO)
    retired = mod.load_retired_exp_ids(REPO / "ops/exclusion_manifest.yaml")
    clean = mod.validate_v542_roadmap_data(data, retired)

    assert identity["milestone"] == mod.MILESTONE_V542
    assert identity["path"] == "research-roadmap.yaml"
    assert identity["research_roadmap_next_present"] is False
    assert clean["schema_validation"]["ok"] is True
    assert clean["task_id_validation"]["expected_order"] is True
    assert clean["task_count"] == 13
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
    dirty["tasks"][2]["requires"] = [dirty["tasks"][2]["id"], "exp2091-retired"]
    dirty["tasks"][3]["deliverable"] = "not-results.txt"
    dirty["tasks"][4]["gated_on"] = [
        {"upstream": dirty["tasks"][0]["id"], "artifact_field": "missing", "op": "==", "value": 1}
    ]
    dirty["tasks"][5]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    dirty["tasks"][6]["agent_type"] = "codex"
    dirty["tasks"][6]["model"] = "opus"
    dirty["tasks"][8]["agent_type"] = "gemini"
    dirty["tasks"][8]["model"] = "gemini-3.1-pro-preview"
    dirty["tasks"][9]["model"] = "opus"
    dirty["tasks"][9]["prompt"] = dirty["tasks"][9]["prompt"].replace(
        "Do NOT push. Do NOT modify scripts/research_conductor.py.",
        "Do NOT push.",
    )
    dirty["tasks"][10]["prompt"] = dirty["tasks"][10]["prompt"].replace(
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "Qwen/Qwen3.5-0.8B",
    )

    dirty_result = mod.validate_v542_roadmap_data(dirty, {2091})

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


def test_reserved_collision_scan_checks_future_deliverables(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6284-3: reserved ids and staged deliverables cannot collide."""

    owned = tmp_path / "python/carnot/experiment_6284_v542_terminal_transition.py"
    owned.parent.mkdir(parents=True, exist_ok=True)
    owned.write_text("# owned\n", encoding="utf-8")
    unexpected = tmp_path / "results/experiment_6288_old.json"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("{}", encoding="utf-8")
    deliverable = (
        tmp_path / "results/experiment_6289_flagship_exact_state_refinement_benchmark.json"
    )
    deliverable.write_text("{}", encoding="utf-8")

    receipt = mod.scan_reserved_id_collisions(
        tmp_path,
        allowed_reserved_paths={"python/carnot/experiment_6284_v542_terminal_transition.py"},
        staged_deliverables={
            "results/experiment_6289_flagship_exact_state_refinement_benchmark.json"
        },
    )

    assert receipt["reserved_exp_ids"] == list(range(6284, 6297))
    assert receipt["unexpected_reserved_collision_count"] == 2
    assert receipt["unexpected_reserved_paths_by_exp_id"] == {
        "6288": ["results/experiment_6288_old.json"],
        "6289": ["results/experiment_6289_flagship_exact_state_refinement_benchmark.json"],
    }
    assert receipt["existing_unallowed_deliverables"] == [
        "results/experiment_6289_flagship_exact_state_refinement_benchmark.json"
    ]


def test_report_builder_records_v541_to_v542_handoff() -> None:
    """SCENARIO-INFRA-6284-5: report validation is machine-checkable."""

    before = mod.protected_hashes(REPO)
    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_6284_v542_terminal_transition.py -q --no-cov -n 0",
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
    assert report["task_count"] == 13
    assert report["retired_dependency_count"] == 0
    assert report["id_collision_count"] == 0
    assert report["v541_milestone_roadmap_and_hash"]["milestone"] == mod.MILESTONE_V541
    assert report["v542_roadmap_path_and_hash"]["milestone"] == mod.MILESTONE_V542
    assert report["v542_roadmap_path_and_hash"]["research_roadmap_next_present"] is False
    assert (
        report["v541_task_terminal_matrix"][
            "exp6275-flagship-asp-constraint-verification-benchmark"
        ]["terminal_class"]
        == "flagged"
    )
    assert (
        report["v541_task_terminal_matrix"]["exp6277-chronological-certified-csl-ab"][
            "terminal_class"
        ]
        == "skipped"
    )
    assert (
        report["focused_and_broad_validation_receipts_by_task"][
            "exp6276-certified-dual-cache-admission"
        ]["broad"]["failed_count"]
        >= 1
    )
    assert (
        report["raw_evidence_eligibility_policy"]["rows"][
            "exp6275-flagship-asp-constraint-verification-benchmark"
        ]["artifact_gate_eligible"]
        is False
    )
    assert report["protected_files_unchanged"]["unchanged"] is True

    blocked = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 1}],
        before_hashes=before,
        started_at=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")


def test_report_validation_requires_principles_bare_zero_prefix_and_checksum() -> None:
    """SCENARIO-INFRA-6284-5: required fields are validated before write."""

    report = {field: f"fixture-{field}" for field in mod.REQUIRED_ARTIFACT_FIELDS}
    report["status"] = "complete"
    report["task_count"] = 13
    report["retired_dependency_count"] = 0
    report["id_collision_count"] = 0
    report["inference_substrate"] = mod.INFERENCE_SUBSTRATE
    report["verifier_is_oracle"] = False
    report["field_principles"] = dict(mod.FIELD_PRINCIPLES)
    report["field_provenance"] = {
        field: {"sources": ["REQ-INFRA-6284"], "principle": mod.FIELD_PRINCIPLES[field]}
        for field in mod.REQUIRED_ARTIFACT_FIELDS
    }
    report["test_exit_codes"] = {}
    report["honest_verdict"] = "complete: fixture"
    report["duration_s"] = 1.0
    report["reproducibility_checksum"] = ""
    report["reproducibility_checksum"] = mod.payload_checksum(report)

    assert mod.validate_report(report) == []

    broken = dict(report)
    broken["task_count"] = 12
    assert "task_count must be 13" in mod.validate_report(broken)

    broken = dict(report)
    broken["retired_dependency_count"] = 0.0
    assert "retired_dependency_count must be bare integer 0" in mod.validate_report(broken)

    broken = dict(report)
    broken["id_collision_count"] = 1
    assert "id_collision_count must be bare integer 0" in mod.validate_report(broken)

    broken = dict(report)
    broken["honest_verdict"] = "success: fixture"
    assert "honest_verdict lacks accepted Exp6284 prefix" in mod.validate_report(broken)


def test_write_report_uses_artifact_root_override(tmp_path: Path) -> None:
    """REQ-INFRA-6284: artifact writes are atomic and test-isolated."""

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
    """SCENARIO-INFRA-6284-3 and SCENARIO-INFRA-6284-5: helper edges fail closed."""

    assert mod.validate_v542_roadmap_data({"tasks": "bad"}, set())["task_count"] == 0
    assert mod.required_artifact_fields_from_prompt("no block") == set()
    assert mod.module_name_for_task({"deliverable": "results/custom-name.json"}) == "custom_name"
    assert mod.prior_ok("bad") == (False, "prior_not_mapping")
    assert mod.gate_ok("bad", {"x": {}}, {}) == (False, "gate_not_mapping")
    assert mod.model_specs_named_in_prompt("no model specs") == []
    assert mod.raw_evidence_eligibility_policy({"bad": "row"})["rows"] == {}
    assert mod._validate_model_policy(
        {"id": "gpu", "requires_gpu": True, "prompt": "no model specs"}
    ) == [{"task_id": "gpu", "reason": "missing_model_specs_gguf_ids"}]
    non_mandated = mod._validate_model_policy(
        {
            "id": "bad-model",
            "prompt": "MODEL_SPECS must include fake/legacy-model-GGUF",
        }
    )
    assert non_mandated == [
        {
            "task_id": "bad-model",
            "reason": "non_mandated_gguf_id",
            "ids": ["fake/legacy-model-GGUF"],
        }
    ]
    all_three = mod._validate_model_policy(
        {
            "id": "three",
            "prompt": "MODEL_SPECS must include all three mandated unsloth/Qwen3.6-35B-A3B-GGUF",
        }
    )
    assert all_three[0]["reason"] == "missing_all_three_mandated_gguf_ids"
    assert mod._validate_model_policy({"id": "odd", "model": "weird", "prompt": ""}) == [
        {"task_id": "odd", "reason": "unknown_agent_model", "model": "weird"}
    ]

    manifest = tmp_path / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(yaml.safe_dump({"retired_extras": [{"experiment_ids": ["exp103-old"]}]}))
    assert mod.load_retired_exp_ids(manifest) == {103}

    prior_data = {
        "milestone": mod.MILESTONE_V542,
        "milestone_title": "Fixture",
        "milestone_doc": "doc.md",
        "tasks": [
            {
                "id": "exp6284-v542-terminal-transition",
                "milestone": mod.MILESTONE_V542,
                "deliverable": "results/experiment_6284_fixture.json",
                "title": "Fixture",
                "prompt": "REQUIRED ARTIFACT FIELDS: status\nRun command: missing",
            },
            {
                "id": "exp6285-v542-post-marker-source-scope-freeze",
                "milestone": mod.MILESTONE_V542,
                "deliverable": "results/experiment_6285_fixture.json",
                "title": "Fixture",
                "prompt": "REQUIRED ARTIFACT FIELDS: status\nRun command: missing",
                "prior_failures": [],
            },
        ],
    }
    prior_result = mod.validate_v542_roadmap_data(prior_data, set())
    assert prior_result["prior_failure_validation"]["failures"] == [
        {
            "task_id": "exp6285-v542-post-marker-source-scope-freeze",
            "reason": "empty_prior_failures",
        }
    ]

    dir_deliverable = tmp_path / "results/experiment_6290_dir.json"
    dir_deliverable.mkdir(parents=True)
    dir_collision = mod.scan_reserved_id_collisions(
        tmp_path,
        staged_deliverables={"results/experiment_6290_dir.json"},
    )
    assert dir_collision["unexpected_reserved_paths_by_exp_id"] == {
        "6290": ["results/experiment_6290_dir.json"]
    }

    next_root = tmp_path / "next"
    next_root.mkdir()
    (next_root / "research-roadmap-next.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": mod.MILESTONE_V542,
                "milestone_title": "Next",
                "milestone_doc": "doc.md",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    _data, identity = mod.load_v542_roadmap(next_root)
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
    _data, identity = mod.load_v542_roadmap(bad_root)
    assert identity["selection_note"] == "V542 roadmap milestone was not found"

    branch_root = tmp_path / "branch"
    branch_root.mkdir()
    (branch_root / "research-roadmap.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": mod.MILESTONE_V542,
                "milestone_title": "Branch",
                "milestone_doc": "doc.md",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    _write_json(
        branch_root / mod.V541_CAPSTONE_RELATIVE_PATH,
        {
            "status": "complete",
            "honest_verdict": "complete: fixture",
            "milestone_roadmap_path_and_hash": [],
            "exact_declared_deliverable_matrix": {},
        },
    )
    _write_json(
        branch_root / mod.OPERATIONAL_RETRO_RELATIVE_PATH, {"milestone": mod.MILESTONE_V541}
    )
    branch_report = mod.build_report(
        branch_root,
        date="20260810",
        before_hashes=mod.protected_hashes(branch_root),
        started_at=0.0,
    )
    assert branch_report["v541_milestone_roadmap_and_hash"]["recorded_roadmap_path"] is None
    assert mod.check_roadmap_only(REPO)["ok"] is True

    invalid = {"status": "complete"}
    errors = mod.validate_report(invalid)
    assert "missing required field: v541_milestone_roadmap_and_hash" in errors
    assert "field_principles is not a mapping" in errors
    assert "field_provenance is not a mapping" in errors
    assert "task_count must be 13" in errors
    assert "wrong inference_substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "honest_verdict lacks accepted Exp6284 prefix" in errors
    with pytest.raises(ValueError, match="invalid Exp6284 report"):
        mod.write_report(invalid, REPO, env={ARTIFACT_ROOT_ENV: str(tmp_path)})
