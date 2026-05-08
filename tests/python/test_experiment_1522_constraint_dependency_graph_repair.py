"""Tests for Exp 1522 constraint dependency graph root-cause repair.

Spec: REQ-VERIFY-1522, SCENARIO-VERIFY-1522.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import constraint_dependency_graph_repair as exp


def test_req_verify_1522_builds_deterministic_cdg_from_lifecycle_and_cofailures() -> None:
    """REQ-VERIFY-1522: graph edges combine lifecycle order and observed co-failures."""

    rows = [
        _case(
            "parse-bad",
            expected_label=False,
            final_accept=False,
            certificate_parse_result={
                "linked": True,
                "parsed": False,
                "deterministic_validation_passed": False,
                "verifier_accepted": False,
            },
        ),
        _case(
            "structural-bad",
            expected_label=False,
            final_accept=False,
            source_family="structural_contract",
            structural_contract_result={
                "linked": True,
                "detected_violation": True,
                "contract_family": "acquisition_path",
            },
        ),
    ]

    graph = exp.build_constraint_dependency_graph(rows)
    edge_by_pair = {(edge["source"], edge["target"]): edge for edge in graph["edges"]}

    assert [node["id"] for node in graph["nodes"]] == [
        "parse",
        "certificate",
        "safe_dsl_verifier",
        "monitor_event",
        "structural_dependency",
        "solver_oracle",
        "final_accept",
    ]
    assert edge_by_pair[("parse", "certificate")]["observed_cofailure_count"] == 1
    assert edge_by_pair[("parse", "certificate")]["reason"] == "lifecycle_order"
    assert edge_by_pair[("structural_dependency", "final_accept")][
        "observed_cofailure_count"
    ] == 1
    assert graph == exp.build_constraint_dependency_graph(reversed(rows))


def test_req_verify_1522_cdg_prioritizes_upstream_root_cause_before_final_accept() -> None:
    """REQ-VERIFY-1522: CDG ordering localizes upstream causes before final failure."""

    case = _case(
        "parse-bad",
        expected_label=False,
        final_accept=False,
        certificate_parse_result={
            "linked": True,
            "parsed": False,
            "deterministic_validation_passed": False,
            "verifier_accepted": False,
        },
    )

    result = exp.analyze_root_cause_case(
        case,
        graph=exp.build_constraint_dependency_graph([case]),
        repair_rows_by_case={},
    )

    assert result["failure_categories"] == ["parse", "certificate", "final_accept"]
    assert result["root_cause_category"] == "parse"
    assert result["flat_order"][0] == "final_accept"
    assert result["cdg_order"][0] == "parse"
    assert result["flat_root_cause_rank"] > result["cdg_root_cause_rank"]
    assert result["cdg_efficiency"] > result["flat_efficiency"]
    assert result["deterministic_validator_accept"] is True
    assert result["false_accept"] is False


def test_scenario_verify_1522_runner_writes_ready_manifest_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1522: runner writes case rows, graph summary, and ready artifact."""

    e2e_manifest = tmp_path / "runtime_contract_e2e.jsonl"
    repair_artifact = tmp_path / "experiment_1521.json"
    repair_manifest = tmp_path / "repair_1521.jsonl"
    output = tmp_path / "experiment_1522.json"
    cdg_manifest = tmp_path / "cdg_1522.jsonl"
    _write_jsonl(
        e2e_manifest,
        [
            _case("accept-ok", expected_label=True, final_accept=True),
            _case(
                "parse-bad",
                expected_label=False,
                final_accept=False,
                certificate_parse_result={
                    "linked": True,
                    "parsed": False,
                    "deterministic_validation_passed": False,
                    "verifier_accepted": False,
                },
            ),
            _case(
                "monitor-bad",
                expected_label=None,
                final_accept=False,
                source_family="monitor_event",
                monitor_event_result={
                    "linked": True,
                    "validation_status": "fail",
                    "verifier_false_accept": False,
                },
            ),
            {"row_type": "summary", "contract_cases_total": 3},
        ],
    )
    _write_json(
        repair_artifact,
        {
            "status": "complete",
            "live_sota_model_inference_used": True,
            "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
            "repair_manifest_path": str(repair_manifest),
        },
    )
    _write_jsonl(
        repair_manifest,
        [
            {"row_type": "summary", "repair_cases_attempted": 1},
            {
                "row_type": "repair_result",
                "contract_case_id": "parse-bad",
                "model_hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "mode": "draft_conditioned",
                "repair_outcome": "accepted",
                "false_accept": False,
            }
        ],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        e2e_manifest_path=e2e_manifest,
        repair_artifact_path=repair_artifact,
        cdg_manifest_path=cdg_manifest,
        output_path=output,
    )
    rows = _read_jsonl(cdg_manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["cdg_root_cause_repair_ready"] is True
    assert artifact["e2e_cases_loaded"] == 3
    assert artifact["root_cause_cases_attempted"] == 2
    assert artifact["false_accept_count"] == 0
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["row_type"] == "cdg_graph_summary"
    assert rows[-1]["cdg_efficiency_delta"] == artifact["cdg_efficiency_delta"]
    assert rows[0]["exp1521_repair_rows_linked"] == 1
    assert rows[0]["root_cause_category"] == "parse"


def test_req_verify_1522_runner_blocks_missing_or_nonfailing_e2e_manifest(tmp_path: Path) -> None:
    """REQ-VERIFY-1522: missing or non-failing Exp 1520 rows are terminal blockers."""

    output = tmp_path / "experiment_1522.json"
    cdg_manifest = tmp_path / "cdg_1522.jsonl"

    missing = exp.run_experiment(
        project_root=tmp_path,
        e2e_manifest_path=tmp_path / "missing.jsonl",
        cdg_manifest_path=cdg_manifest,
        output_path=output,
    )
    e2e_manifest = tmp_path / "runtime_contract_e2e.jsonl"
    _write_jsonl(e2e_manifest, [_case("accept-ok", expected_label=True, final_accept=True)])
    nonfailing = exp.run_experiment(
        project_root=tmp_path,
        e2e_manifest_path=e2e_manifest,
        cdg_manifest_path=cdg_manifest,
        output_path=output,
    )

    assert missing["status"] == "blocked"
    assert any(
        str(blocker).startswith("missing_runtime_contract_e2e_manifest:")
        for blocker in missing["blockers"]
    )
    assert missing["cdg_root_cause_repair_ready"] is False
    assert nonfailing["status"] == "blocked"
    assert "no_failing_runtime_contract_cases" in nonfailing["blockers"]
    assert cdg_manifest.read_text(encoding="utf-8") == ""
    assert nonfailing["honest_verdict"].startswith("complete:")


def test_req_verify_1522_branch_coverage_for_safe_solver_unlabeled_and_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1522: less common categories and blocker branches stay deterministic."""

    safe_case = _case(
        "safe-bad",
        expected_label=False,
        final_accept=True,
        source_family="safe_dsl",
        safe_dsl_verifier_result={"linked": True, "false_accept_row_id": "safe-bad"},
    )
    solver_case = _case(
        "solver-oracle-bad",
        expected_label=False,
        final_accept=False,
        source_family="product_line_solver_oracle",
    )
    final_only = _case(
        "final-only",
        expected_label=False,
        final_accept=True,
        source_family="unknown",
    )
    unlabeled_monitor = _case(
        "monitor-unlabeled",
        expected_label=None,
        final_accept=False,
        source_family="monitor_event",
        monitor_event_result={"linked": True, "validation_status": "fail"},
    )

    assert exp.contract_failure_categories(safe_case) == [
        "safe_dsl_verifier",
        "final_accept",
    ]
    assert exp.contract_failure_categories(solver_case) == ["solver_oracle", "final_accept"]
    assert exp.analyze_root_cause_case(
        final_only,
        graph=exp.build_constraint_dependency_graph([final_only]),
        repair_rows_by_case={},
    )["root_cause_category"] == "final_accept"
    unlabeled_row = exp.analyze_root_cause_case(
        unlabeled_monitor,
        graph=exp.build_constraint_dependency_graph([unlabeled_monitor]),
        repair_rows_by_case={},
    )
    assert exp._artifact_metrics([unlabeled_row])["false_accept_rate"] == pytest.approx(0.0)

    e2e_manifest = tmp_path / "runtime_contract_e2e.jsonl"
    repair_artifact = tmp_path / "experiment_1521.json"
    repair_manifest = tmp_path / "repair_1521.jsonl"
    _write_jsonl(e2e_manifest, [safe_case])
    _write_json(
        repair_artifact,
        {
            "status": "complete",
            "live_sota_model_inference_used": True,
            "models_used": ["legacy/tiny-model"],
            "repair_manifest_path": str(repair_manifest),
        },
    )
    _write_jsonl(repair_manifest, [])

    def force_false_accept(case: dict[str, Any]) -> dict[str, Any]:
        validation = dict(case)
        validation["final_deterministic_accept"] = True
        validation["final_deterministic_decision"] = "accept"
        return validation

    monkeypatch.setattr(exp, "_candidate_repair_validation_row", force_false_accept)
    artifact = exp.run_experiment(
        project_root=tmp_path,
        e2e_manifest_path=e2e_manifest,
        repair_artifact_path=repair_artifact,
        cdg_manifest_path=tmp_path / "cdg_1522.jsonl",
        output_path=tmp_path / "experiment_1522.json",
    )

    assert artifact["status"] == "blocked"
    assert artifact["false_accept_rate"] == pytest.approx(1.0)
    assert "false_accept_rate_nonzero_or_unmeasured" in artifact["blockers"]
    assert "no_mandated_sota_model_in_repair_evidence" in artifact["blockers"]


def _case(
    case_id: str,
    *,
    expected_label: bool | None,
    final_accept: bool,
    source_family: str = "grammar_certificate",
    certificate_parse_result: dict[str, Any] | None = None,
    safe_dsl_verifier_result: dict[str, Any] | None = None,
    monitor_event_result: dict[str, Any] | None = None,
    structural_contract_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "row_type": "contract_case",
        "contract_schema_version": "runtime-contract-e2e/v1",
        "contract_case_id": case_id,
        "prompt_or_case_id": case_id,
        "proposed_output": case_id,
        "certificate_parse_result": certificate_parse_result or {"linked": False},
        "safe_dsl_verifier_result": safe_dsl_verifier_result or {"linked": False},
        "monitor_event_result": monitor_event_result or {"linked": False},
        "structural_contract_result": structural_contract_result or {"linked": False},
        "expected_label": expected_label,
        "final_deterministic_accept": final_accept,
        "final_deterministic_decision": "accept" if final_accept else "reject",
        "source_family": source_family,
        "source_path": "tests",
        "source_line": 1,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
