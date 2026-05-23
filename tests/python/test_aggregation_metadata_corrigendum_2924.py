"""Tests for the Exp 2924 aggregation-metadata corrigendum.

Spec refs: REQ-REPORT-2924, SCENARIO-REPORT-2924.

The corrigendum is a provenance repair, not a compute experiment. These tests
use tiny JSON fixtures to prove the workflow keeps inherited upstream flags
separate from metadata false positives on the aggregation artifacts themselves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot.reporting import aggregation_metadata_corrigendum_2924 as exp2924


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pending(kind: str, detail: str = "inherited compute-bound marker") -> dict[str, str]:
    return {"kind": kind, "severity": "critical", "detail": detail}


def _source_payload(substrate: str, *, pending: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: fixture source",
        "inference_substrate": substrate,
        "corrigendum_pending": pending or [],
        "flagged_adversarial": bool(pending),
    }


def _write_scenario_sources(root: Path) -> None:
    sources = {
        "results/experiment_2910_sota_code_generation_corrigendum_v2.json": _source_payload(
            "live_llm_inference"
        ),
        "results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json": _source_payload(
            "deterministic_taxonomy_verifier",
            pending=[_pending("DURATION_TOO_SHORT"), _pending("METHODOLOGY_MISSING")],
        ),
        "results/experiment_2919_constraintbench_mini_direct_optimization_v1.json": _source_payload(
            "live_llm_inference_plus_exact_verifier",
            pending=[_pending("TAUTOLOGY"), _pending("DURATION_TOO_SHORT")],
        ),
    }
    for rel_path, payload in sources.items():
        _write_json(root, rel_path, payload)

    matrix = {
        "artifact": "experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1",
        "schema": "carnot.cross_corpus_matrix.v9_paper_boundary.v1",
        "honest_verdict": "complete: cross-corpus matrix v9 and paper-v6 claim boundary built",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 0.008404,
        "flagged_adversarial": True,
        "flagged_rows": [
            "corpus:MBPP",
            "corpus:HumanEval",
            "exp2911_code_hallucination_verifier",
            "exp2919_constraintbench_mini",
        ],
        "corrigendum_pending": [
            _pending(
                "DURATION_TOO_SHORT",
                "duration_s=0.008404 but artifact references compute-bound markers",
            ),
            _pending("METHODOLOGY_MISSING", "Compute-bound artifact missing methodology"),
            _pending("UNRELATED_UPSTREAM_NOTE", "not a metadata false-positive kind"),
        ],
        "cited_upstream_artifacts": [
            {"experiment_id": "exp_bad_without_path"},
            {
                "experiment_id": "exp2910",
                "artifact_path": "results/experiment_2910_sota_code_generation_corrigendum_v2.json",
                "present": True,
                "row_id": "exp2910_sota_codegen",
            },
            {
                "experiment_id": "exp2911",
                "artifact_path": "results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json",
                "present": True,
                "row_id": "exp2911_code_hallucination_verifier",
            },
            {
                "experiment_id": "exp2915",
                "artifact_path": "results/experiment_2915_gatemate_n16_ising_tile_bitstream_build_v2.json",
                "present": False,
                "row_id": "exp2915_gatemate_bitstream",
            },
            {
                "experiment_id": "exp2919",
                "artifact_path": "results/experiment_2919_constraintbench_mini_direct_optimization_v1.json",
                "present": True,
                "row_id": "exp2919_constraintbench_mini",
            },
        ],
    }
    _write_json(root, exp2924.MATRIX_V9_SOURCE, matrix)

    capstone = {
        "artifact": "experiment_2922_capstone_v275",
        "schema": "carnot.milestone_capstone.v275",
        "honest_verdict": "complete: .275 capstone synthesized",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 0.007456,
        "flagged_adversarial": True,
        "flagged_artifacts": ["exp2911", "exp2919", "exp2921"],
        "corrigendum_pending": [
            _pending(
                "DURATION_TOO_SHORT",
                "duration_s=0.007456 but artifact references compute-bound markers",
            ),
            _pending("METHODOLOGY_MISSING", "Compute-bound artifact missing methodology"),
        ],
        "source_artifact_status": {
            "exp2911": {
                "path": "results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json",
                "present": True,
                "status": "flagged",
            },
            "exp2919": {
                "path": "results/experiment_2919_constraintbench_mini_direct_optimization_v1.json",
                "present": True,
                "status": "flagged",
            },
            "exp2921": {
                "path": str(exp2924.MATRIX_V9_SOURCE),
                "present": True,
                "status": "flagged",
            },
        },
        "cited_upstream_artifacts": [
            {
                "experiment_id": "exp2911",
                "artifact_path": "results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json",
                "present": True,
            },
            {
                "experiment_id": "exp2919",
                "artifact_path": "results/experiment_2919_constraintbench_mini_direct_optimization_v1.json",
                "present": True,
            },
            {
                "experiment_id": "exp2921",
                "artifact_path": str(exp2924.MATRIX_V9_SOURCE),
                "present": True,
            },
        ],
    }
    _write_json(root, exp2924.CAPSTONE_SOURCE, capstone)


def test_req_report_2924_spec_is_declared() -> None:
    """REQ-REPORT-2924: OpenSpec declares the corrigendum contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-REPORT-2924" in spec
    assert "SCENARIO-REPORT-2924" in spec
    assert "aggregation_provenance" in spec


def test_scenario_report_2924_separates_upstream_flags_from_false_positives(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2924: preserve upstream flags while clearing aggregation metadata."""

    _write_scenario_sources(tmp_path)
    audit = {
        "audit_available": True,
        "audit_tool": "fake-adversarial-verify",
        "returncode": 0,
        "flagged": False,
        "findings": [],
    }

    artifact = exp2924.build_artifact(
        tmp_path,
        audit_result=audit,
        started_s=10.0,
        now_s=12.5,
    )

    assert exp2924.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["aggregation_metadata_clean"] is True
    assert artifact["no_new_llm_call"] is True
    assert artifact["no_new_hardware_run"] is True
    assert artifact["aggregation_from_upstream_artifacts"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["run_date"] == "20260523"

    checksums = artifact["source_artifact_checksums"]
    assert checksums[str(exp2924.MATRIX_V9_SOURCE)] == _sha256(
        tmp_path / exp2924.MATRIX_V9_SOURCE
    )
    assert checksums[str(exp2924.CAPSTONE_SOURCE)] == _sha256(
        tmp_path / exp2924.CAPSTONE_SOURCE
    )
    assert (
        checksums["results/experiment_2915_gatemate_n16_ising_tile_bitstream_build_v2.json"]
        is None
    )

    provenance = artifact["aggregation_provenance"]
    assert any(
        row["artifact_path"] == str(exp2924.MATRIX_V9_SOURCE)
        and row["row_role"] == "corrigendum_subject"
        for row in provenance
    )
    assert any(
        row["row_role"] == "matrix_row_source"
        and row["source_inference_substrate"] == "deterministic_taxonomy_verifier"
        for row in provenance
    )
    assert any(row["present"] is False and row["checksum"] is None for row in provenance)
    assert all(row["current_task_reran_compute"] is False for row in provenance)

    preserved_ids = {
        row["experiment_id"]
        for row in artifact["upstream_flagged_rows_preserved"]
        if row.get("experiment_id")
    }
    preserved_identifiers = {
        row["identifier"] for row in artifact["upstream_flagged_rows_preserved"]
    }
    assert {"exp2911", "exp2919", "exp2921"} <= preserved_ids
    assert {"corpus:MBPP", "corpus:HumanEval"} <= preserved_identifiers

    false_positive_pairs = {
        (row["experiment_id"], row["kind"])
        for row in artifact["metadata_false_positive_findings"]
    }
    assert ("exp2921", "DURATION_TOO_SHORT") in false_positive_pairs
    assert ("exp2921", "METHODOLOGY_MISSING") in false_positive_pairs
    assert ("exp2922", "DURATION_TOO_SHORT") in false_positive_pairs
    assert ("exp2922", "METHODOLOGY_MISSING") in false_positive_pairs
    assert all(
        row["artifact_inference_substrate"] == "aggregation_from_upstream_artifacts"
        for row in artifact["metadata_false_positive_findings"]
    )
    assert all(
        row["experiment_id"] not in {"exp2911", "exp2919"}
        for row in artifact["metadata_false_positive_findings"]
    )
    assert artifact["adversarial_audit_rerun"] == audit


def test_req_report_2924_missing_upstream_blocks_without_audit(tmp_path: Path) -> None:
    """REQ-REPORT-2924: absent Exp 2921 or Exp 2922 writes a blocked artifact."""

    _write_json(
        tmp_path,
        exp2924.MATRIX_V9_SOURCE,
        {
            "honest_verdict": "complete: matrix exists",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
    )

    artifact = exp2924.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert artifact["aggregation_metadata_clean"] is False
    assert artifact["missing_upstream_artifacts"] == [str(exp2924.CAPSTONE_SOURCE)]
    assert artifact["no_new_llm_call"] is True
    assert artifact["no_new_hardware_run"] is True
    assert artifact["source_artifact_checksums"][str(exp2924.MATRIX_V9_SOURCE)] == _sha256(
        tmp_path / exp2924.MATRIX_V9_SOURCE
    )
    assert artifact["source_artifact_checksums"][str(exp2924.CAPSTONE_SOURCE)] is None
    assert artifact["adversarial_audit_rerun"]["not_run_reason"] == "upstream_missing"

    def fail_audit(root: Path, artifact_path: Path) -> dict[str, Any]:
        raise AssertionError("blocked precondition must exit before audit")

    out = exp2924.write_artifact(
        tmp_path,
        audit_runner=fail_audit,
        clock=_clock(2.0, 2.1),
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "blocked_upstream_artifact_missing"


def test_req_report_2924_audit_findings_make_current_artifact_unclean(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2924: current-artifact audit flags remain exact and separate."""

    _write_scenario_sources(tmp_path)
    audit = {
        "audit_available": True,
        "audit_tool": "fake-adversarial-verify",
        "returncode": 1,
        "flagged": True,
        "findings": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "exact"}],
    }

    artifact = exp2924.build_artifact(
        tmp_path,
        audit_result=audit,
        started_s=0.0,
        now_s=0.2,
    )

    assert artifact["aggregation_metadata_clean"] is False
    assert artifact["adversarial_audit_rerun"]["findings"] == audit["findings"]
    assert all(
        finding["kind"] != "TAUTOLOGY"
        for finding in artifact["metadata_false_positive_findings"]
    )


def test_req_report_2924_write_artifact_persists_json_and_runs_audit(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2924: write_artifact persists the final audited corrigendum."""

    _write_scenario_sources(tmp_path)
    calls: list[tuple[Path, Path]] = []

    def fake_audit(root: Path, artifact_path: Path) -> dict[str, Any]:
        calls.append((root, artifact_path))
        assert artifact_path.exists()
        return {
            "audit_available": True,
            "audit_tool": "fake-adversarial-verify",
            "returncode": 0,
            "flagged": False,
            "findings": [],
        }

    out = exp2924.write_artifact(
        tmp_path,
        audit_runner=fake_audit,
        clock=_clock(4.0, 4.1, 4.75),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / exp2924.DEFAULT_OUTPUT_PATH
    assert calls == [(tmp_path, out), (tmp_path, out)]
    assert payload["aggregation_metadata_clean"] is True
    assert payload["duration_s"] == pytest.approx(0.75)
    assert payload["adversarial_audit_rerun"]["audit_tool"] == "fake-adversarial-verify"


def test_req_report_2924_write_artifact_falls_back_after_unstable_audit(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2924: unstable audit output still leaves the latest findings recorded."""

    _write_scenario_sources(tmp_path)
    calls = 0

    def unstable_audit(root: Path, artifact_path: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "audit_available": True,
            "audit_tool": "unstable-adversarial-verify",
            "returncode": calls,
            "flagged": True,
            "findings": [
                {
                    "kind": f"KIND_{calls}",
                    "severity": "critical",
                    "detail": "unstable",
                }
            ],
        }

    out = exp2924.write_artifact(
        tmp_path,
        audit_runner=unstable_audit,
        clock=_clock(20.0, 20.1, 20.2, 20.3, 20.4),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert calls == 3
    assert payload["aggregation_metadata_clean"] is False
    assert payload["duration_s"] == pytest.approx(0.4)
    assert payload["adversarial_audit_rerun"]["returncode"] == 3
    assert payload["adversarial_audit_rerun"]["findings"][0]["kind"] == "KIND_3"


def test_req_report_2924_local_audit_runner_parses_json_and_absence(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2924: local audit runner records availability and exact findings."""

    unavailable = exp2924.run_adversarial_audit(
        tmp_path,
        tmp_path / exp2924.DEFAULT_OUTPUT_PATH,
    )
    assert unavailable["audit_available"] is False
    assert unavailable["not_run_reason"] == "audit_tool_unavailable"

    script = tmp_path / "scripts" / "adversarial_verify.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder\n", encoding="utf-8")

    completed = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(
            {
                "reports": [
                    {
                        "artifact": str(tmp_path / exp2924.DEFAULT_OUTPUT_PATH),
                        "flag_count": 1,
                        "flags": [
                            {
                                "kind": "DURATION_TOO_SHORT",
                                "severity": "critical",
                                "detail": "exact audit output",
                            }
                        ],
                    }
                ],
                "flagged_count": 1,
            }
        ),
        stderr="",
    )
    seen: dict[str, Any] = {}

    def fake_runner(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return completed

    parsed = exp2924.run_adversarial_audit(
        tmp_path,
        tmp_path / exp2924.DEFAULT_OUTPUT_PATH,
        runner=fake_runner,
        python_executable="pythonX",
    )

    assert parsed["audit_available"] is True
    assert parsed["audit_tool"] == "scripts/adversarial_verify.py"
    assert parsed["returncode"] == 1
    assert parsed["flagged"] is True
    assert parsed["findings"] == [
        {
            "kind": "DURATION_TOO_SHORT",
            "severity": "critical",
            "detail": "exact audit output",
        }
    ]
    assert seen["cmd"] == [
        "pythonX",
        str(script),
        str(tmp_path / exp2924.DEFAULT_OUTPUT_PATH),
        "--json",
    ]
    assert seen["kwargs"]["cwd"] == str(tmp_path)


def test_req_report_2924_defensive_helper_branches(tmp_path: Path) -> None:
    """REQ-REPORT-2924: defensive helper branches stay explicit and covered."""

    _write_scenario_sources(tmp_path)

    no_audit = exp2924.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    assert no_audit["aggregation_metadata_clean"] is False
    assert no_audit["adversarial_audit_rerun"]["not_run_reason"] == "audit_not_supplied"

    skipped = exp2924._metadata_false_positive_findings(
        {
            "exp2921": {
                "inference_substrate": "live_llm_inference",
                "corrigendum_pending": [_pending("DURATION_TOO_SHORT")],
            },
            "exp2922": {
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "corrigendum_pending": [_pending("UNRELATED_UPSTREAM_NOTE")],
            },
        }
    )
    assert skipped == []

    exp2922_row = exp2924._flag_row(tmp_path, "exp2922", "self", "field", {})
    assert exp2922_row["artifact_path"] == str(exp2924.CAPSTONE_SOURCE)

    generic_flag = exp2924._pending_findings({"flagged_adversarial": True})
    assert generic_flag == [
        {
            "kind": "flagged_adversarial",
            "severity": "warn",
            "detail": "upstream artifact marked flagged_adversarial=true",
        }
    ]
