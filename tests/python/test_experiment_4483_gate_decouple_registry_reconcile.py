"""Tests for Exp 4483 gate decoupling and ARC registry reconciliation.

Spec refs: REQ-REPORT-4483, SCENARIO-REPORT-4483.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from carnot import experiment_4483_gate_decouple_registry_reconcile as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _roadmap_payload() -> dict[str, Any]:
    return {
        "milestone": "2026.06.414",
        "tasks": [
            {
                "id": "exp4480-a6",
                "track": "arc-north-star",
                "title": "independent solve",
                "deliverable": "results/experiment_4480_solve_bp35_goal_directed.json",
                "gated_on": [{"task_id": "exp4479-a5", "field": "honest_verdict"}],
                "prompt": "Solve bp35 independently.",
            },
            {
                "id": "exp4481-a7",
                "track": "arc-north-star",
                "title": "variant benchmark mentions gated_on in prose",
                "deliverable": "results/experiment_4481_variant_transfer_benchmark.json",
                "prompt": "No structured gate, just a gated_on advisory mention.",
            },
            {
                "id": "exp4484-c",
                "track": "hardware",
                "title": "hardware continuity",
                "deliverable": "results/experiment_4484_hardware_continuity_audit.json",
                "gated_on": [{"task_id": "exp4483-b2", "field": "honest_verdict"}],
                "prompt": "Hardware chain is not an independent ARC solve.",
            },
            {
                "id": "exp4486-e",
                "track": "capstone",
                "title": "capstone gated_on wording",
                "deliverable": "results/experiment_4486_capstone_v414.json",
                "prompt": "Aggregate whatever exists; no gated_on structured chain.",
            },
        ],
    }


def _registry_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-20",
        "games": [
            {"game": "lp85", "reproducibility": "reproduced", "levels_reproduced": 5},
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "re86", "reproducibility": "reproduced", "levels_reproduced": 1},
        ],
        "reproducible_total_levels": 7,
        "reproducible_total_games": 3,
        "provisional_total_levels": 0,
        "latest_hygiene_4461": {
            "artifact": "results/experiment_4461_registry_gaps_hygiene.json",
            "reproducible_total_levels": 39,
            "reproducible_total_games": 20,
        },
        "latest_hygiene_4474": {
            "artifact": "results/experiment_4474_registry_gaps_hygiene.json",
            "reproducible_total_levels": 45,
            "reproducible_total_games": 22,
        },
    }


def _fixture_repo(root: Path) -> None:
    _write_yaml(root / mod.ROADMAP_RELATIVE_PATH, _roadmap_payload())
    _write_yaml(root / mod.ARC_REGISTRY_RELATIVE_PATH, _registry_payload())


def test_req_report_4483_spec_declares_gate_decouple_contract() -> None:
    """REQ-REPORT-4483: OpenSpec names the gate decouple and registry contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4483" in spec
    assert "SCENARIO-REPORT-4483" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    assert "aggregation_from_upstream_artifacts" in spec


def test_scenario_report_4483_removes_arc_solve_gates_and_reconciles_registry(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4483: independent ARC solve gates are removed."""

    _fixture_repo(tmp_path)

    artifact = mod.run(tmp_path, now=lambda: 20.0)

    assert artifact["honest_verdict"] == "complete: gate_decoupled_registry_reconciled_4483"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 7
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["gate_decoupling"]["removed_gate_task_ids"] == ["exp4480-a6"]
    assert artifact["gate_decoupling"]["structured_gate_count_before"] == 2
    assert artifact["gate_decoupling"]["structured_gate_count_after"] == 1
    assert artifact["gate_decoupling"]["advisory_text_task_ids"] == [
        "exp4481-a7",
        "exp4486-e",
    ]
    assert artifact["registry_reconciliation"]["authoritative_header"] == {
        "reproducible_total_levels": 7,
        "reproducible_total_games": 3,
    }
    assert artifact["registry_reconciliation"]["stale_hygiene_keys"] == [
        "latest_hygiene_4461",
        "latest_hygiene_4474",
    ]
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert mod.artifact_schema_errors(artifact) == []

    roadmap = yaml.safe_load((tmp_path / mod.ROADMAP_RELATIVE_PATH).read_text())
    tasks = {task["id"]: task for task in roadmap["tasks"]}
    assert "gated_on" not in tasks["exp4480-a6"]
    assert "gated_on" in tasks["exp4484-c"]

    registry = yaml.safe_load((tmp_path / mod.ARC_REGISTRY_RELATIVE_PATH).read_text())
    block = registry["latest_gate_decouple_registry_reconcile_4483"]
    assert block["artifact"] == mod.RESULT_RELATIVE_PATH
    assert block["reproducible_total_levels"] == 7
    assert block["reproducible_total_games"] == 3
    assert block["supersedes_stale_hygiene_keys"] == [
        "latest_hygiene_4461",
        "latest_hygiene_4474",
    ]
    assert registry["latest_hygiene_4461"]["reproducible_total_levels"] == 39

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_report_4483_schema_checks_and_noop_decouple(tmp_path: Path) -> None:
    """REQ-REPORT-4483: no structured ARC gates is terminal and schema is strict."""

    roadmap = _roadmap_payload()
    roadmap["tasks"][0].pop("gated_on")
    _write_yaml(tmp_path / mod.ROADMAP_RELATIVE_PATH, roadmap)
    _write_yaml(tmp_path / mod.ARC_REGISTRY_RELATIVE_PATH, _registry_payload())

    artifact = mod.run(tmp_path)

    assert artifact["gate_decoupling"]["removed_gate_task_ids"] == []
    assert artifact["gate_decoupling"]["structured_gate_count_before"] == 1
    assert artifact["gate_decoupling"]["structured_gate_count_after"] == 1
    assert artifact["registry_reconciliation"]["reproduced_counts_match_header"] is True

    bad = {
        **artifact,
        "honest_verdict": "partial: no",
        "inference_substrate": "",
        "offline_reproduced": "true",
        "reproduced_levels": True,
        "preconditions_checked": [],
        "gate_decoupling": [],
        "registry_reconciliation": [],
        "field_principles": {"honest_verdict": {"principle": "wrong"}},
        "spec_refs": [],
        "random_seed": "4483",
        "reproducibility_checksum": "bad",
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must equal aggregation_from_upstream_artifacts" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "preconditions_checked must be dict" in errors
    assert "gate_decoupling must be dict" in errors
    assert "registry_reconciliation must be dict" in errors
    assert "field_principles must match REQ-REPORT-4483" in errors
    assert "spec_refs must include REQ-REPORT-4483 and SCENARIO-REPORT-4483" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
