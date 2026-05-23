"""Tests for Exp 2935 cross-corpus matrix v10.

Spec refs: REQ-REPORT-2935, SCENARIO-REPORT-2935.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v10_2935 as exp2935


REQUIRED_FIELDS = {
    "honest_verdict",
    "matrix_v10_ready",
    "matrix_v10_paper_boundary_ready",
    "no_new_llm_call",
    "no_new_hardware_run",
    "row_classification_counts",
    "headline_eligible_rows",
    "flagged_rows",
    "blocked_rows",
    "projection_only_rows",
    "pilot_only_rows",
    "paper_claim_boundary",
    "source_artifact_checksums",
    "adversarial_audit_rerun",
    "inference_substrate",
    "duration_s",
    "run_date",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pending(kind: str) -> dict[str, str]:
    return {"kind": kind, "severity": "critical", "detail": f"{kind} fixture flag"}


def _model_spec() -> list[dict[str, str]]:
    return [{"name": "Gemma4-26B-A4B-it", "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}]


def _v9_row(
    row_id: str,
    status: str,
    *,
    headline: bool = False,
    pilot: bool = False,
    live_model: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "headline_eligible": headline,
        "pilot_only": pilot,
        "inference_substrate": "live_llm_inference" if live_model else "deterministic_fixture",
    }
    if live_model:
        summary["model_specs"] = _model_spec()
    return {
        "row_id": row_id,
        "row_label": row_id.replace("_", " "),
        "row_kind": "v9_fixture",
        "row_status": status,
        "headline_eligible": headline,
        "claim_boundary": f"bounded claim for {row_id}" if headline else "",
        "non_headline_reason": "" if headline else "not headline",
        "flag_reasons": [_pending("DURATION_TOO_SHORT")["kind"]] if status == "flagged" else [],
        "summary": summary,
        "source_artifact": "results/upstream_fixture.json",
        "source_experiment_id": "exp2902",
        "source_sha256": "fixture",
    }


def _write_ready_sources(root: Path) -> None:
    v9_rows = [
        _v9_row("corpus:FoVer", "clean", headline=True),
        _v9_row("corpus:MBPP", "flagged", pilot=True),
        _v9_row("exp2910_sota_codegen", "clean", headline=True, live_model=True),
        _v9_row("exp2911_code_hallucination_verifier", "flagged"),
        _v9_row("exp2915_gatemate_bitstream", "missing"),
        _v9_row("exp2916_thrml_parity", "diagnostic_only"),
        _v9_row("exp2919_constraintbench_mini", "flagged"),
    ]
    _write_json(
        root,
        exp2935.MATRIX_V9_SOURCE,
        {
            "artifact": "experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1",
            "honest_verdict": "complete: matrix v9 fixture",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "matrix_rows": v9_rows,
            "headline_eligible_rows": ["corpus:FoVer", "exp2910_sota_codegen"],
            "flagged_rows": [
                "corpus:MBPP",
                "exp2911_code_hallucination_verifier",
                "exp2919_constraintbench_mini",
            ],
            "pilot_only_rows": ["corpus:MBPP"],
            "diagnostic_only_rows": ["exp2916_thrml_parity"],
            "missing_rows": ["exp2915_gatemate_bitstream"],
        },
    )

    _write_json(
        root,
        exp2935.EXP2924_SOURCE,
        {
            "honest_verdict": "complete: aggregation metadata clean",
            "aggregation_metadata_clean": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [_pending("DURATION_TOO_SHORT")],
            "adversarial_audit_rerun": {"audit_available": True, "flagged": False},
            "upstream_flagged_rows_preserved": [
                {"identifier": "exp2911_code_hallucination_verifier"},
                {"identifier": "exp2919_constraintbench_mini"},
            ],
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
    )
    _write_json(
        root,
        exp2935.EXP2925_SOURCE,
        {
            "honest_verdict": "complete: taxonomy corrigendum clean",
            "taxonomy_corrigendum_clean": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [_pending("DURATION_TOO_SHORT")],
            "adversarial_audit_rerun": {"audit_available": True, "flagged": False},
            "model_specs": [{"name": "deterministic-taxonomy", "llm_invoked": False}],
            "inference_substrate": "deterministic_verifier",
            "no_new_llm_call": True,
            "no_new_hardware_run": True,
        },
    )
    _write_json(
        root,
        exp2935.EXP2926_SOURCE,
        {
            "honest_verdict": "complete: constraintbench corrigendum",
            "constraintbench_corrigendum_ready": True,
            "flagged_adversarial": False,
            "model_specs": _model_spec(),
            "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "syntax_valid_rate": 0.33,
            "feasibility_rate_overall": 0.3,
            "optimality_rate_given_feasible": 0.44,
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root,
        exp2935.EXP2927_SOURCE,
        {
            "honest_verdict": "blocked_constraints_missing",
            "gatemate_himbaechel_ready": False,
            "constraints_ready": False,
            "inference_substrate": "hardware_toolchain_preflight",
        },
    )
    _write_json(
        root,
        exp2935.EXP2929_SOURCE,
        {
            "honest_verdict": "blocked_gatemate_bitstream_missing",
            "gatemate_flash_smoke_ready": False,
            "inference_substrate": "physical_board_smoke",
        },
    )
    _write_json(
        root,
        exp2935.EXP2930_SOURCE,
        {
            "honest_verdict": "complete: projection only",
            "kv260_scaling_projection_ready": True,
            "projection_only": True,
            "not_a_speedup_claim": True,
            "no_new_hardware_run": True,
            "inference_substrate": "aggregation_plus_simulation",
        },
    )
    _write_json(
        root,
        exp2935.EXP2931_SOURCE,
        {
            "honest_verdict": "blocked_z3_execution_incomplete",
            "logic_verifier_mini_ready": False,
            "model_specs": _model_spec(),
            "inference_substrate": "live_llm_inference_plus_z3",
        },
    )
    _write_json(
        root,
        exp2935.EXP2932_FIXTURE_SOURCE,
        {"schema": "carnot.citation_fixture.v1", "cases": [{"case_id": "fixture"}]},
    )
    _write_json(
        root,
        exp2935.EXP2932_SOURCE,
        {
            "artifact": "experiment_2932_citation_hallucination_field_verifier_v1",
            "honest_verdict": "complete:citation_field_verifier_ready",
            "citation_verifier_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [_pending("DURATION_TOO_SHORT")],
            "model_specs": _model_spec(),
            "inference_substrate": "live_llm_inference_plus_deterministic_verifier",
        },
    )
    _write_json(
        root,
        exp2935.EXP2933_SOURCE,
        {
            "honest_verdict": "complete: kan_rbf_importance_self_learning_passed",
            "kan_cl_self_learning_ready": True,
            "continuous_self_learning_targeted": True,
            "forgetting_rate": 0.0,
            "utility_delta_vs_replay_only": 0.5,
            "inference_substrate": "local_training_simulation",
        },
    )
    _write_json(
        root,
        exp2935.EXP2934_SOURCE,
        {
            "artifact": "experiment_2934_aquaforte_beaver_reformulation_pipeline_v1",
            "honest_verdict": "complete: reformulated and exact-verified",
            "reformulation_pipeline_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [_pending("DURATION_TOO_SHORT")],
            "model_specs": _model_spec(),
            "inference_substrate": "live_llm_inference_plus_exact_verifier",
        },
    )


def _row_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["row_id"]: row for row in artifact["matrix_rows"]}


def test_req_report_2935_spec_is_declared() -> None:
    """REQ-REPORT-2935: OpenSpec declares the matrix-v10 contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2935" in spec
    assert "SCENARIO-REPORT-2935" in spec
    assert str(exp2935.OUTPUT_REL_PATH) in spec


def test_req_report_2935_blocks_when_required_corrigendum_gate_is_missing(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2935: missing or false structured gates write the blocked artifact."""

    _write_json(
        tmp_path,
        exp2935.EXP2924_SOURCE,
        {"aggregation_metadata_clean": False, "inference_substrate": "aggregation"},
    )

    artifact = exp2935.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == "blocked_required_corrigendum_missing"
    assert artifact["matrix_v10_ready"] is False
    assert artifact["matrix_v10_paper_boundary_ready"] is False
    assert artifact["no_new_llm_call"] is True
    assert artifact["no_new_hardware_run"] is True
    assert artifact["headline_eligible_rows"] == []
    assert artifact["adversarial_audit_rerun"]["not_run_reason"] == "required_gate_failed"
    assert artifact["source_artifact_checksums"][str(exp2935.EXP2924_SOURCE)] == _sha256(
        tmp_path / exp2935.EXP2924_SOURCE
    )
    gate_fields = {error["required_field"] for error in artifact["required_gate_errors"]}
    assert {"aggregation_metadata_clean", "taxonomy_corrigendum_clean"} <= gate_fields


def test_scenario_report_2935_builds_v10_without_promoting_flagged_boundaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2935: corrigenda add clean rows without laundering old flags."""

    _write_ready_sources(tmp_path)
    audit = {
        "audit_available": True,
        "audit_tool": "fake-audit",
        "returncode": 0,
        "flagged": False,
        "findings": [],
    }

    artifact = exp2935.build_artifact(
        tmp_path,
        audit_result=audit,
        started_s=10.0,
        now_s=12.25,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["matrix_v10_ready"] is True
    assert artifact["matrix_v10_paper_boundary_ready"] is True
    assert artifact["no_new_llm_call"] is True
    assert artifact["no_new_hardware_run"] is True
    assert artifact["inference_substrate"] == exp2935.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["run_date"] == "20260523"

    counts = artifact["row_classification_counts"]
    assert counts["clean"] >= 5
    assert counts["flagged"] >= 3
    assert counts["blocked"] == 3
    assert counts["missing"] == 2
    assert counts["projection_only"] == 1
    assert counts["diagnostic_only"] == 1
    assert counts["pilot_only"] == 1

    rows = _row_by_id(artifact)
    assert rows["exp2925_taxonomy_corrigendum"]["row_class"] == "clean"
    assert rows["exp2925_taxonomy_corrigendum"]["headline_eligible"] is False
    assert rows["exp2925_taxonomy_corrigendum"]["paper_claim_eligible"] is True
    assert rows["exp2926_constraintbench_corrigendum"]["row_class"] == "clean"
    assert rows["exp2926_constraintbench_corrigendum"]["headline_eligible"] is True
    assert rows["exp2926_constraintbench_corrigendum"]["model_specs_if_live_llm"] == _model_spec()
    assert rows["exp2930_kv260_scaling_projection"]["row_class"] == "projection_only"
    assert rows["exp2930_kv260_scaling_projection"]["headline_eligible"] is False
    assert rows["exp2930_kv260_scaling_projection"]["paper_claim_eligible"] is False
    assert rows["exp2932_citation_field_verifier"]["row_class"] == "flagged"
    assert rows["exp2934_reformulation_pipeline"]["row_class"] == "flagged"
    assert rows["corpus:MBPP"]["row_class"] == "pilot_only"
    assert rows["corpus:MBPP"]["headline_eligible"] is False

    assert "corpus:MBPP" in artifact["pilot_only_rows"]
    assert "corpus:MBPP" not in artifact["flagged_rows"]
    assert "exp2911_code_hallucination_verifier" in artifact["flagged_rows"]
    assert "exp2928_gatemate_bitstream" in artifact["missing_rows"]
    assert "exp2930_kv260_scaling_projection" in artifact["projection_only_rows"]
    assert "exp2930_kv260_scaling_projection" not in artifact["headline_eligible_rows"]
    assert "exp2934_reformulation_pipeline" not in artifact["headline_eligible_rows"]

    boundary = artifact["paper_claim_boundary"]
    assert boundary["ready"] is True
    assert "exp2926_constraintbench_corrigendum" in boundary["headline_eligible_rows"]
    assert "exp2930_kv260_scaling_projection" in boundary["non_paper_claim_rows"]
    assert "exp2925_taxonomy_corrigendum" in boundary["supporting_paper_claim_rows"]
    assert any("Projection-only" in rule for rule in boundary["boundary_rules"])

    checksums = artifact["source_artifact_checksums"]
    assert checksums[str(exp2935.MATRIX_V9_SOURCE)] == _sha256(tmp_path / exp2935.MATRIX_V9_SOURCE)
    assert checksums[str(exp2935.EXP2928_SOURCE)] is None
    assert checksums[str(exp2935.EXP2932_FIXTURE_SOURCE)] == _sha256(
        tmp_path / exp2935.EXP2932_FIXTURE_SOURCE
    )
    assert any(
        item["identifier"] == "exp2911_code_hallucination_verifier"
        for item in artifact["upstream_flags_preserved"]
    )
    assert any(item["identifier"] == "corpus:MBPP" for item in artifact["upstream_flags_preserved"])
    no_audit_artifact = exp2935.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    assert no_audit_artifact["adversarial_audit_rerun"]["not_run_reason"] == ("audit_not_supplied")


def test_req_report_2935_write_artifact_persists_and_records_audit(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2935: write_artifact records the final local audit result."""

    _write_ready_sources(tmp_path)

    def audit_runner(root: Path, artifact_path: Path) -> dict[str, Any]:
        assert root == tmp_path
        assert artifact_path == tmp_path / exp2935.OUTPUT_REL_PATH
        return {
            "audit_available": True,
            "audit_tool": "fake-audit",
            "returncode": 0,
            "flagged": False,
            "findings": [],
        }

    out = exp2935.write_artifact(
        tmp_path,
        audit_runner=audit_runner,
        clock=lambda: 100.0,
    )

    assert out == tmp_path / exp2935.OUTPUT_REL_PATH
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["matrix_v10_ready"] is True
    assert artifact["adversarial_audit_rerun"]["audit_tool"] == "fake-audit"
    assert artifact["duration_s"] >= exp2935.AUDITABLE_MIN_DURATION_S


def test_req_report_2935_write_artifact_stops_before_audit_when_blocked(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2935: blocked gate writes are terminal and skip audit."""

    def audit_runner(root: Path, artifact_path: Path) -> dict[str, Any]:
        raise AssertionError(f"audit should not run for {root=} {artifact_path=}")

    out = exp2935.write_artifact(
        tmp_path,
        audit_runner=audit_runner,
        clock=lambda: 200.0,
    )

    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == "blocked_required_corrigendum_missing"
    assert artifact["matrix_v10_ready"] is False
    assert artifact["adversarial_audit_rerun"]["not_run_reason"] == "required_gate_failed"


def test_req_report_2935_audit_runner_fallbacks_and_parses_json(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2935: local audit runner records availability and exact findings."""

    unavailable = exp2935.run_adversarial_audit(tmp_path, tmp_path / "artifact.json")
    assert unavailable == {
        "audit_available": False,
        "not_run_reason": "audit_tool_unavailable",
        "flagged": False,
        "findings": [],
    }

    audit_tool = tmp_path / "scripts" / "adversarial_verify.py"
    audit_tool.parent.mkdir(parents=True)
    audit_tool.write_text("# fake audit tool\n", encoding="utf-8")

    completed = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(
            {
                "flagged_count": 1,
                "reports": [
                    {
                        "artifact": "artifact.json",
                        "flags": [
                            {
                                "kind": "DURATION_TOO_SHORT",
                                "severity": "critical",
                                "detail": "too fast",
                            }
                        ],
                    }
                ],
            }
        ),
        stderr="audit stderr",
    )

    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs["cwd"] == str(tmp_path)
        calls.append(command)
        return completed

    audit = exp2935.run_adversarial_audit(
        tmp_path,
        tmp_path / "artifact.json",
        runner=runner,
        python_executable="python",
    )

    assert audit["audit_available"] is True
    assert audit["audit_tool"] == "scripts/adversarial_verify.py"
    assert audit["returncode"] == 1
    assert audit["flagged"] is True
    assert audit["findings"] == [
        {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "too fast"}
    ]
    assert audit["stderr"] == "audit stderr"
    assert calls == [["python", str(audit_tool), str(tmp_path / "artifact.json"), "--json"]]


def test_req_report_2935_helper_edges_keep_boundaries_explicit() -> None:
    """REQ-REPORT-2935: defensive helper paths keep row boundaries explicit."""

    assert exp2935._blocked_verdict("gate_blocked_missing")
    assert exp2935._blocked_verdict("blocked_missing")
    assert not exp2935._blocked_verdict(None)
    assert exp2935._as_string_list(["x", 1]) == ["x", "1"]
    assert exp2935._as_findings([{"kind": "X"}, "bad"]) == [
        {"kind": "X", "severity": "unknown", "detail": ""}
    ]
    assert exp2935._row_flag_reasons(
        {"flagged_adversarial": True, "corrigendum_pending": [_pending("X")]}
    ) == ["flagged_adversarial=true", "X:critical"]
    assert exp2935._row_flag_reasons(
        {
            "adversarial_verify_passed": False,
            "adversarial_verify_flags": [_pending("Y")],
            "adversarial_audit_rerun": {"flagged": True},
        }
    ) == [
        "adversarial_verify_passed=false",
        "Y:critical",
        "adversarial_audit_rerun_flagged=true",
    ]
    assert exp2935._v9_flag_reasons({"flagged_adversarial": True}) == ["flagged_adversarial=true"]
    assert exp2935._classify_v9_row(_v9_row("p", "clean", pilot=True)) == "pilot_only"
    assert exp2935._classify_v9_row(_v9_row("p", "projection_only")) == "projection_only"
    assert exp2935._classify_v9_row(_v9_row("p", "blocked")) == "blocked"
    assert (
        exp2935._classify_dot276_source(
            exp2935.DOT276_SOURCE_BY_EXP["exp2930"],
            {"projection_only": True, "honest_verdict": "complete"},
        )
        == "projection_only"
    )
    assert (
        exp2935._classify_dot276_source(
            exp2935.DOT276_SOURCE_BY_EXP["exp2932"],
            {"citation_verifier_ready": True},
        )
        == "clean"
    )
    assert (
        exp2935._classify_dot276_source(
            exp2935.DOT276_SOURCE_BY_EXP["exp2932"],
            {"honest_verdict": "complete: context only"},
        )
        == "clean"
    )
    assert (
        exp2935._classify_dot276_source(
            exp2935.DOT276_SOURCE_BY_EXP["exp2932"],
            {"honest_verdict": "incomplete"},
        )
        == "blocked"
    )
    assert exp2935._row_inference_substrate({}, {}) == exp2935.INFERENCE_SUBSTRATE
    assert exp2935._hardware_substrate("row", {}, "hardware_smoke") == (
        "hardware substrate declared by source"
    )
    assert exp2935._dot276_claim_boundary("unknown_clean_row", {}, True) == (
        "Bounded clean supporting claim; no broader generalization is implied."
    )
