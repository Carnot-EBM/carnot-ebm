"""Tests for Exp 3309 live-runtime provenance contract.

Spec refs: REQ-REPORT-3309, SCENARIO-REPORT-3309.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import live_runtime_provenance_contract_3309 as mod


MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checker_versions() -> dict[str, str]:
    return {
        "live_runtime_provenance_contract": mod.CONTRACT_VERSION,
        "executable_checker_path": mod.EXECUTABLE_CHECKER_PATH,
        "checker_file_sha256": "sha256:contract",
        "adversarial_verify": "scripts/adversarial_verify.py@sha256:adversarial",
        "spec_coverage": "scripts/check_spec_coverage.py@sha256:spec",
        "llama_cpp_python": "0.3.16",
        "selected_python_cuda_probe": "selected_python_cuda@sha256:cuda",
    }


def _metric_lineage() -> dict[str, dict[str, Any]]:
    return {
        "refusal_rate": {
            "numerator": 5,
            "denominator": 150,
            "source_filter": "all_probe_families",
            "source_row_count": 150,
            "calculation_function": "sum(refusal_count)/sum(probe_count)",
            "source_artifact_sha256": "sha256:exp3300",
        },
        "aligned_instruction_false_positive_rate": {
            "numerator": 1,
            "denominator": 30,
            "source_filter": "family=aligned_benign",
            "source_row_count": 30,
            "calculation_function": "false_positive_count/probe_count",
            "source_artifact_sha256": "sha256:exp3300",
        },
    }


def _headline_payload() -> dict[str, Any]:
    return {
        "artifact": "headline_live_fixture",
        "evidence_tier": "headline_live",
        "headline_result": True,
        "headline_claim_allowed": True,
        "runtime_contract_claimed": True,
        "duration_s": 75.25,
        "tokens_generated": 128,
        "inference_substrate": "llama_cpp_gpu_openai_adapter",
        "models_used": [
            {
                "model_id": MODEL_ID,
                "hf_id": MODEL_ID,
                "name": "Gemma4-26B-A4B-it",
                "role": "moe",
                "model_path": "/cache/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                "cache_root": "/cache/models",
                "snapshot_revision": "3365c68d",
                "size_bytes": 16_947_539_744,
                "quantization": "UD-Q4_K_M",
                "load_started_at": "2026-05-29T00:00:00Z",
                "load_finished_at": "2026-05-29T00:00:12Z",
                "generated_tokens": 128,
            }
        ],
        "runtime_provenance": {
            "command": [".venv/bin/python", "scripts/experiment_3312.py"],
            "argv": [".venv/bin/python", "scripts/experiment_3312.py"],
            "cwd": "/repo",
            "pid": 3312,
            "cuda_visible_devices": "0,1",
            "wall_clock_duration_s": 75.25,
            "model_load_started_at": "2026-05-29T00:00:00Z",
            "model_load_finished_at": "2026-05-29T00:00:12Z",
            "model_load_duration_s": 12.0,
            "generation_started_at": "2026-05-29T00:00:12Z",
            "generation_finished_at": "2026-05-29T00:01:15Z",
            "gpu_memory_samples": [
                {"phase": "before_load", "gpus": [{"index": 0, "memory_used_mib": 4}]},
                {"phase": "after_load", "gpus": [{"index": 0, "memory_used_mib": 18_152}]},
                {"phase": "after_generation", "gpus": [{"index": 0, "memory_used_mib": 18_220}]},
            ],
            "per_case_generation": [
                {
                    "case_id": "case-1",
                    "started_at": "2026-05-29T00:00:12Z",
                    "finished_at": "2026-05-29T00:00:18Z",
                    "generated_tokens": 128,
                }
            ],
        },
        "checker_versions": _checker_versions(),
        "metric_lineage": _metric_lineage(),
        "refusal_rate": 0.033333,
        "aligned_instruction_false_positive_rate": 0.033333,
    }


def _aggregation_audit_payload() -> dict[str, Any]:
    return {
        "artifact": "repair_audit_fixture",
        "evidence_tier": "aggregation_audit",
        "duration_s": 0.002,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "no_new_model_execution": True,
        "substrate_consistency_passed": True,
        "source_panel_runtime_contract": {
            "runtime_contract_passed": True,
            "duration_contract_passed": True,
            "critical_adversarial_flags": [],
        },
        "panel_case_count": 30,
        "source_panel_case_count": 30,
        "manifest_case_hashes": ["case-a", "case-b"],
        "source_manifest_case_hashes": ["case-a", "case-b"],
        "exact_checker_types": ["exact_integer_string", "exact_alias_string"],
        "source_exact_checker_types": ["exact_alias_string", "exact_integer_string"],
        "model_invocation_summary": {
            "used_model_ids": [MODEL_ID],
            "missing_model_ids": [QWEN_ID],
            "legacy_small_model_used": False,
        },
        "source_model_invocation_summary": {
            "used_model_ids": [MODEL_ID],
            "missing_model_ids": [QWEN_ID],
        },
        "checker_versions": _checker_versions(),
    }


def test_req_report_3309_spec_anchor_declares_runtime_contract_schema() -> None:
    """REQ-REPORT-3309: OpenSpec names the contract before implementation."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-REPORT-3309" in spec
    assert "SCENARIO-REPORT-3309" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.EXECUTABLE_CHECKER_PATH in spec
    assert mod.CONTRACT_VERSION in spec
    assert "minimum_live_duration_s >= 60.0" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3309_writes_contract_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3309: contract artifact exposes all downstream gates."""

    _write_json(
        tmp_path,
        mod.EXP3308_REL_PATH,
        {
            "artifact": "experiment_3308_quality_flag_root_cause_autopsy_v1",
            "experiment_id": "exp3308",
            "quality_flag_autopsy_ready": True,
            "minimum_live_duration_s": 60.0,
            "honest_verdict": "complete: fixture",
        },
    )

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=10.0,
        now_s=12.5,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["runtime_contract_ready"] is True
    assert artifact["contract_version"] == mod.CONTRACT_VERSION
    assert artifact["minimum_live_duration_s"] == pytest.approx(60.0)
    assert artifact["executable_checker_path"] == mod.EXECUTABLE_CHECKER_PATH
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["reproducibility_checksum"]) == 64

    categories = {row["category"] for row in artifact["required_provenance_fields"]}
    assert {
        "model_identity",
        "cache_path",
        "model_size",
        "load_timing",
        "wall_clock_duration",
        "generated_token_count",
        "command",
        "cuda_visibility",
        "gpu_memory",
        "checker_versions",
    } <= categories
    assert artifact["duration_guard_rules"]["headline_live_evidence"]["minimum_duration_s"] == 60.0
    assert artifact["duration_guard_rules"]["cpu_smoke_exception"]["headline_promotion_allowed"] is False
    assert "metric_lineage" in artifact["tautology_guard_rules"]["required_lineage_object"]
    assert (
        artifact["repair_substrate_rules"]["audit_substrate_exception"]["allowed_audit_substrate"]
        == "aggregation_from_upstream_artifacts"
    )

    source = artifact["source_artifacts"][0]
    assert source["experiment_id"] == "exp3308"
    assert source["present"] is True
    assert source["ready"] is True
    mod.validate_contract_artifact(artifact)


def test_req_report_3309_checker_accepts_headline_live_with_independent_lineage() -> None:
    """REQ-REPORT-3309: valid headline GGUF evidence can pass the checker."""

    check = mod.check_runtime_evidence_artifact(_headline_payload())

    assert check["evidence_tier"] == "headline_live"
    assert check["runtime_contract_passed"] is True
    assert check["duration_contract_passed"] is True
    assert check["tautology_guard_passed"] is True
    assert check["repair_substrate_passed"] is True
    assert check["headline_promotion_allowed"] is True
    assert check["violations"] == []


def test_req_report_3309_checker_rejects_too_short_headline_and_missing_receipts() -> None:
    """REQ-REPORT-3309: live markers below the floor cannot clear promotion."""

    payload = _headline_payload()
    payload["duration_s"] = 39.0
    del payload["runtime_provenance"]["model_load_started_at"]
    del payload["runtime_provenance"]["gpu_memory_samples"]

    check = mod.check_runtime_evidence_artifact(payload)
    kinds = {violation["kind"] for violation in check["violations"]}

    assert check["runtime_contract_passed"] is False
    assert check["duration_contract_passed"] is False
    assert check["headline_promotion_allowed"] is False
    assert {"DURATION_TOO_SHORT", "MISSING_PROVENANCE"} <= kinds


def test_req_report_3309_checker_allows_non_headline_cpu_smoke_exception() -> None:
    """REQ-REPORT-3309: CPU smoke can be short only when promotion is disabled."""

    payload = {
        "artifact": "cpu_smoke_fixture",
        "evidence_tier": "cpu_smoke",
        "cpu_smoke_only": True,
        "headline_result": False,
        "headline_claim_allowed": False,
        "duration_s": 0.25,
        "models_used": [{"model_id": MODEL_ID, "hf_id": MODEL_ID, "name": "Gemma smoke"}],
        "runtime_provenance": {"command": [".venv/bin/python", "smoke.py"], "cwd": "/repo"},
        "checker_versions": _checker_versions(),
    }

    check = mod.check_runtime_evidence_artifact(payload)

    assert check["evidence_tier"] == "cpu_smoke"
    assert check["runtime_contract_passed"] is True
    assert check["duration_contract_passed"] is True
    assert check["headline_promotion_allowed"] is False
    assert check["violations"] == []
    assert "non_headline_cpu_smoke_exception" in check["warnings"]


def test_req_report_3309_checker_rejects_tautology_lineage_reuse() -> None:
    """REQ-REPORT-3309: equal distinct metrics need independent lineage."""

    payload = _headline_payload()
    payload["metric_lineage"]["aligned_instruction_false_positive_rate"] = deepcopy(
        payload["metric_lineage"]["refusal_rate"]
    )

    check = mod.check_runtime_evidence_artifact(payload)

    assert check["runtime_contract_passed"] is False
    assert check["tautology_guard_passed"] is False
    assert any(violation["kind"] == "TAUTOLOGY" for violation in check["violations"])


def test_req_report_3309_checker_applies_repair_audit_substrate_rules() -> None:
    """REQ-REPORT-3309: repair audit and source panel must agree on substrate facts."""

    passing = mod.check_runtime_evidence_artifact(_aggregation_audit_payload())
    failing_payload = _aggregation_audit_payload()
    failing_payload["source_panel_runtime_contract"]["runtime_contract_passed"] = False
    failing_payload["source_manifest_case_hashes"] = ["different-case"]
    failing = mod.check_runtime_evidence_artifact(failing_payload)

    assert passing["runtime_contract_passed"] is True
    assert passing["repair_substrate_passed"] is True
    assert passing["headline_promotion_allowed"] is False
    assert failing["runtime_contract_passed"] is False
    assert failing["repair_substrate_passed"] is False
    assert any(violation["kind"] == "REPAIR_SUBSTRATE_INCONSISTENCY" for violation in failing["violations"])


def test_req_report_3309_validate_rejects_incomplete_contracts(tmp_path: Path) -> None:
    """REQ-REPORT-3309: malformed contract artifacts cannot masquerade as ready."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_contract_artifact({})
    with pytest.raises(ValueError, match="runtime_contract_ready"):
        mod.validate_contract_artifact(artifact | {"runtime_contract_ready": "true"})
    with pytest.raises(ValueError, match="contract_version"):
        mod.validate_contract_artifact(artifact | {"contract_version": "v0"})
    with pytest.raises(ValueError, match="minimum_live_duration_s"):
        mod.validate_contract_artifact(artifact | {"minimum_live_duration_s": 30.0})
    with pytest.raises(ValueError, match="required_provenance_fields"):
        mod.validate_contract_artifact(artifact | {"required_provenance_fields": []})
    with pytest.raises(ValueError, match="tautology_guard_rules"):
        mod.validate_contract_artifact(artifact | {"tautology_guard_rules": {}})
    with pytest.raises(ValueError, match="duration_guard_rules"):
        mod.validate_contract_artifact(artifact | {"duration_guard_rules": {}})
    with pytest.raises(ValueError, match="repair_substrate_rules"):
        mod.validate_contract_artifact(artifact | {"repair_substrate_rules": {}})
    with pytest.raises(ValueError, match="executable_checker_path"):
        mod.validate_contract_artifact(artifact | {"executable_checker_path": ""})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_contract_artifact(artifact | {"honest_verdict": "blocked"})


def test_req_report_3309_defensive_helper_branches_are_explicit() -> None:
    """REQ-REPORT-3309: defensive helpers classify malformed downstream rows."""

    missing = mod.headline_missing_provenance(
        {
            "models_used": [{"model_id": MODEL_ID}],
            "runtime_provenance": {},
            "checker_versions": {},
            "tokens_generated": 0,
        }
    )

    assert missing[0]["kind"] == "MISSING_PROVENANCE"
    assert "models_used[].hf_id" in missing[0]["detail"]
    assert "checker_versions.live_runtime_provenance_contract" in missing[0]["detail"]
    assert "tokens_generated" in missing[0]["detail"]
    assert "models_used[]" in mod.headline_missing_provenance({})[0]["detail"]

    assert mod.evidence_tier({"cpu_smoke_only": True}) == "cpu_smoke"
    assert mod.evidence_tier({"inference_substrate": "aggregation_from_upstream_artifacts"}) == "aggregation_audit"
    assert mod.evidence_tier({"headline_result": True}) == "headline_live"
    assert mod.evidence_tier({}) == "non_headline"
    assert mod.string_list("not-an-iterable-list") == []
    assert mod.string_list(42) == []
    assert mod.numeric("not-a-number") == 0.0
