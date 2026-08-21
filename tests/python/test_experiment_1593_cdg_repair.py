"""Tests for Exp 1593 constraint dependency graph root-cause repair acceptance rates.

Spec: REQ-VERIFY-1593, SCENARIO-VERIFY-1593.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verifiers import cdg_repair_1593 as exp


def test_scenario_verify_1593_runner_writes_ready_manifest_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1593: runner outputs JSON artifact with required schema fields."""

    e2e_manifest = tmp_path / "runtime_contract_e2e.jsonl"
    repair_artifact = tmp_path / "experiment_1521.json"
    repair_manifest = tmp_path / "repair_1521.jsonl"
    output = tmp_path / "experiment_1593_cdg_repair.json"
    cdg_manifest = tmp_path / "cdg_1593.jsonl"

    exp._write_jsonl(
        e2e_manifest,
        [
            {
                "row_type": "contract_case",
                "contract_case_id": "parse-bad",
                "expected_label": False,
                "final_deterministic_accept": False,
                "certificate_parse_result": {"linked": True, "parsed": False},
                "source_family": "grammar_certificate",
            },
            {
                "row_type": "contract_case",
                "contract_case_id": "monitor-bad",
                "expected_label": False,
                "final_deterministic_accept": False,
                "monitor_event_result": {"linked": True, "validation_status": "fail"},
                "source_family": "monitor_event",
            },
        ],
    )

    exp._write_json(
        repair_artifact,
        {
            "status": "complete",
            "repair_manifest_path": str(repair_manifest),
        },
    )

    exp._write_jsonl(
        repair_manifest,
        [
            {
                "row_type": "repair_result",
                "contract_case_id": "parse-bad",
                "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "repair_outcome": "accepted",
                "false_accept": False,
            },
            {
                "row_type": "repair_result",
                "contract_case_id": "monitor-bad",
                "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "repair_outcome": "rejected",
                "false_accept": False,
            },
        ],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        e2e_manifest_path=e2e_manifest,
        repair_artifact_path=repair_artifact,
        output_path=output,
        cdg_manifest_path=cdg_manifest,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert "cdg_nodes" in artifact
    assert "cdg_edges" in artifact
    assert "flat_acceptance_rate" in artifact
    assert "cdg_acceptance_rate" in artifact
    assert artifact["flat_acceptance_rate"] is not None
    assert artifact["cdg_acceptance_rate"] is not None
    assert "honest_verdict" in artifact
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["model_specs"][0]["hf_repo_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"


def test_req_verify_1593_runner_blocks_missing_manifest(tmp_path: Path) -> None:
    """REQ-VERIFY-1593: missing e2e manifest is a blocker."""
    output = tmp_path / "experiment_1593_cdg_repair.json"
    cdg_manifest = tmp_path / "cdg_1593.jsonl"

    missing = exp.run_experiment(
        project_root=tmp_path,
        e2e_manifest_path=tmp_path / "missing.jsonl",
        repair_artifact_path=tmp_path / "missing_repair.json",
        output_path=output,
        cdg_manifest_path=cdg_manifest,
    )

    assert missing["status"] == "blocked"
    assert any("missing_runtime_contract_e2e_manifest" in b for b in missing["blockers"])


def test_contract_failure_categories() -> None:
    case = {
        "row_type": "contract_case",
        "contract_case_id": "parse-bad",
        "expected_label": False,
        "final_deterministic_accept": False,
        "certificate_parse_result": {"linked": True, "parsed": False},
    }
    cats = exp.contract_failure_categories(case)
    assert "parse" in cats
    assert "certificate" in cats
    assert "final_accept" in cats


def test_contract_failure_categories_extended() -> None:
    case_safe = {
        "row_type": "contract_case",
        "contract_case_id": "safe-bad",
        "expected_label": False,
        "final_deterministic_accept": False,
        "safe_dsl_verifier_result": {"linked": True, "false_accept_row_id": "safe-bad"},
    }
    cats = exp.contract_failure_categories(case_safe)
    assert "safe_dsl_verifier" in cats

    case_struct = {
        "row_type": "contract_case",
        "contract_case_id": "struct-bad",
        "expected_label": False,
        "final_deterministic_accept": False,
        "structural_contract_result": {"linked": True, "detected_violation": True},
    }
    cats = exp.contract_failure_categories(case_struct)
    assert "structural_dependency" in cats

    case_solver = {
        "row_type": "contract_case",
        "contract_case_id": "solver-bad",
        "expected_label": False,
        "final_deterministic_accept": False,
        "solver_oracle_result": {"linked": True, "oracle_agreement": False},
    }
    cats = exp.contract_failure_categories(case_solver)
    assert "solver_oracle" in cats

    assert exp._root_cause_category(["final_accept"]) == "final_accept"


def test_req_verify_1593_runner_blocks_no_failing_cases(tmp_path: Path) -> None:
    e2e_manifest = tmp_path / "runtime_contract_e2e.jsonl"
    repair_artifact = tmp_path / "missing_repair.json"
    output = tmp_path / "experiment_1593_cdg_repair.json"
    cdg_manifest = tmp_path / "cdg_1593.jsonl"
    exp._write_jsonl(
        e2e_manifest,
        [{"row_type": "contract_case", "expected_label": True, "final_deterministic_accept": True}],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        e2e_manifest_path=e2e_manifest,
        repair_artifact_path=repair_artifact,
        output_path=output,
        cdg_manifest_path=cdg_manifest,
    )
    assert artifact["status"] == "blocked"
    assert "no_failing_cases" in artifact["blockers"]


def test_load_repair_evidence_continue() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repair_artifact_path = root / "repair_artifact.json"
        manifest_path = root / "manifest.jsonl"
        exp._write_json(repair_artifact_path, {"repair_manifest_path": str(manifest_path)})
        exp._write_jsonl(manifest_path, [{"row_type": "not_repair_result"}])
        ev = exp._load_repair_evidence(root, repair_artifact_path)
        assert ev == {"repair_rows_by_case": {}}


def test_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    argv = [
        "--e2e-manifest",
        "missing.jsonl",
        "--output",
        str(tmp_path / "experiment_1593_cdg_repair.json"),
        "--cdg-manifest",
        str(tmp_path / "cdg_1593.jsonl"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert exp.main(argv) == 0
