"""Tests for Exp 3303 repair headline evidence audit v1.

Spec refs: REQ-VERIFY-3303, SCENARIO-VERIFY-3303.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.verify import repair_headline_evidence_audit_v1 as mod


REQUIRED_FIELDS = {
    "repair_headline_evidence_audit_ready",
    "headline_claim_allowed_after_audit",
    "audited_artifact",
    "panel_case_count",
    "exact_successes_audited",
    "false_accept_count",
    "llm_judge_dependency_count",
    "adversarial_verify_flags",
    "substrate_consistency_passed",
    "confidence_interval_present",
    "claim_boundaries",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _case_hashes() -> list[str]:
    return [f"{idx:064x}" for idx in range(1, 31)]


def _manifest_payload(case_hashes: list[str] | None = None) -> dict[str, Any]:
    hashes = list(case_hashes or _case_hashes())
    return {
        "artifact": "experiment_3301_exact_repair_panel_manifest_v11",
        "experiment_id": "exp3301",
        "repair_panel_manifest_ready": True,
        "panel_case_count": 30,
        "case_hashes": hashes,
        "llm_judge_required_count": 0,
        "exact_checker_types": [
            "exact_alias_string",
            "exact_bool_string",
            "exact_context_string",
            "exact_integer_string",
            "exact_stdout_string",
        ],
        "panel_cases": [
            {
                "case_id": f"case-{idx:02d}",
                "case_hash": case_hash,
                "llm_judge_required": False,
                "exact_checker_type": "exact_context_string",
            }
            for idx, case_hash in enumerate(hashes, start=1)
        ],
        "inference_substrate": "deterministic_exact_manifest_no_live_inference",
        "random_seed": 3301,
        "reproducibility_checksum": "a" * 64,
        "duration_s": 0.1,
        "honest_verdict": "complete: manifest ready",
    }


def _candidate_results(case_hashes: list[str] | None = None) -> list[dict[str, Any]]:
    hashes = list(case_hashes or _case_hashes())
    rows: list[dict[str, Any]] = []
    for idx, case_hash in enumerate(hashes, start=1):
        success = idx <= 27
        rows.append(
            {
                "case_id": f"case-{idx:02d}",
                "case_hash": case_hash,
                "verified_success": success,
                "exact_check_passed": True,
                "false_accept": False,
                "exact_checker_type": "exact_context_string",
                "calibrated_clean_verifier_decision": "accept" if success else "reject",
                "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            }
        )
    return rows


def _panel_payload(
    *,
    case_hashes: list[str] | None = None,
    headline_claim_allowed: bool = False,
    provenance_clean: bool = False,
) -> dict[str, Any]:
    hashes = list(case_hashes or _case_hashes())
    return {
        "artifact": "experiment_3302_headline_sota_repair_panel_v11",
        "experiment_id": "exp3302",
        "headline_repair_panel_ready": True,
        "repair_panel_ran": True,
        "headline_claim_allowed": headline_claim_allowed,
        "panel_case_count": 30,
        "verified_success_count": 27,
        "repair_success_rate": 0.9,
        "repair_success_ci95": [0.743789, 0.9654],
        "false_accept_count": 0,
        "false_accept_rate_ci95": [0.0, 0.113513],
        "abstention_count": 0,
        "candidate_results": _candidate_results(hashes),
        "manifest_case_hashes": hashes,
        "manifest_case_hashes_match": True,
        "provenance_clean": provenance_clean,
        "flagged_adversarial": not provenance_clean,
        "model_specs": {
            "mandated_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
            ],
            "mandated_models": {
                "unsloth/gemma-4-26B-A4B-it-GGUF": {
                    "cached": True,
                    "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "model_path": "/models/gemma.gguf",
                    "name": "Gemma4-26B-A4B-it",
                    "size_bytes": 16_947_539_744,
                },
                "unsloth/Qwen3.6-35B-A3B-GGUF": {
                    "cached": False,
                    "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "model_path": None,
                    "name": "Qwen3.6-35B-A3B",
                    "size_bytes": 0,
                },
                "unsloth/gemma-4-31B-it-GGUF": {
                    "cached": False,
                    "model_id": "unsloth/gemma-4-31B-it-GGUF",
                    "model_path": None,
                    "name": "Gemma4-31B-it",
                    "size_bytes": 0,
                },
            },
        },
        "models_used": [
            {
                "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "name": "Gemma4-26B-A4B-it",
                "model_path": "/models/gemma.gguf",
                "legacy_small_model": False,
            }
        ],
        "missing_model_specs": [
            {"model_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "reason": "not_cached"},
            {"model_id": "unsloth/gemma-4-31B-it-GGUF", "reason": "not_cached"},
        ],
        "inference_substrate": "live_local_sota_gguf_repair_plus_calibrated_clean_verifier",
        "random_seed": 3302,
        "reproducibility_checksum": "b" * 64,
        "duration_s": 15.496424,
        "honest_verdict": "complete: repair panel ran",
    }


def _write_sources(
    root: Path,
    panel: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    _write_json(root, mod.EXP3302_REL_PATH, panel or _panel_payload())
    _write_json(root, mod.EXP3301_REL_PATH, manifest or _manifest_payload())


def _critical_duration_report(path: Path) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "loaded": True,
        "flag_count": 1,
        "max_severity": 2,
        "flags": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=15.496424 but live model markers require >=60s",
            }
        ],
    }


def _clean_report(path: Path) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "loaded": True,
        "flag_count": 0,
        "max_severity": -1,
        "flags": [],
    }


def test_req_verify_3303_spec_anchor_declares_audit_schema() -> None:
    """REQ-VERIFY-3303: OpenSpec names the audit contract before code."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3303" in spec
    assert "SCENARIO-VERIFY-3303" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.EXP3302_REL_PATH.as_posix() in spec
    assert "scripts/research_conductor.py" in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3303_writes_bounded_audit_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3303: critical source flags block headline promotion."""

    _write_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=10.0,
        now_s=12.5,
        adversarial_reporter=_critical_duration_report,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_headline_evidence_audit_ready"] is True
    assert artifact["headline_claim_allowed_after_audit"] is False
    assert artifact["audited_artifact"] == mod.EXP3302_REL_PATH.as_posix()
    assert artifact["panel_case_count"] == 30
    assert artifact["exact_successes_audited"] == 27
    assert artifact["false_accept_count"] == 0
    assert artifact["llm_judge_dependency_count"] == 0
    assert artifact["adversarial_verify_flags"][0]["kind"] == "DURATION_TOO_SHORT"
    assert artifact["substrate_consistency_passed"] is False
    assert artifact["confidence_interval_present"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["source_artifacts"]["exp3302"]["present"] is True
    assert artifact["model_invocation_summary"]["used_model_ids"] == [
        "unsloth/gemma-4-26B-A4B-it-GGUF"
    ]
    assert artifact["exact_check_provenance"]["all_claimed_successes_exact_checked"] is True
    assert any("duration/substrate" in boundary for boundary in artifact["claim_boundaries"])
    assert any("Qwen" in boundary for boundary in artifact["claim_boundaries"])
    mod.validate_artifact(artifact)


def test_req_verify_3303_allows_clean_already_allowed_source(tmp_path: Path) -> None:
    """REQ-VERIFY-3303: headline promotion requires every audited gate clean."""

    clean_panel = _panel_payload(headline_claim_allowed=True, provenance_clean=True)
    clean_panel["duration_s"] = 90.0
    clean_panel["flagged_adversarial"] = False
    clean_panel["missing_model_specs"] = []
    clean_panel["model_specs"]["mandated_models"][
        "unsloth/Qwen3.6-35B-A3B-GGUF"
    ]["cached"] = True
    clean_panel["model_specs"]["mandated_models"][
        "unsloth/gemma-4-31B-it-GGUF"
    ]["cached"] = True
    clean_panel["models_used"].extend(
        [
            {
                "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "name": "Qwen3.6-35B-A3B",
                "model_path": "/models/qwen.gguf",
            },
            {
                "model_id": "unsloth/gemma-4-31B-it-GGUF",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "name": "Gemma4-31B-it",
                "model_path": "/models/gemma31.gguf",
            },
        ]
    )
    _write_sources(tmp_path, panel=clean_panel)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.25,
        adversarial_reporter=_clean_report,
    )

    assert artifact["headline_claim_allowed_after_audit"] is True
    assert artifact["substrate_consistency_passed"] is True
    assert artifact["claim_boundaries"] == [
        "Headline repair claim allowed only for the audited Exp 3302 fixed 30-case exact panel and its recorded SOTA GGUF model set."
    ]
    assert mod.rate(2, 4) == 0.5
    assert mod.rate(1, 0) == 0.0
    assert mod.duration(5.0, 4.0) == 0.0
    assert mod.mapping_list("bad") == []


def test_req_verify_3303_fail_closed_boundaries(tmp_path: Path) -> None:
    """REQ-VERIFY-3303: unsafe source changes become explicit boundaries."""

    panel = _panel_payload()
    panel["candidate_results"][0]["exact_check_passed"] = False
    panel["candidate_results"][1]["llm_judge_required"] = True
    panel["false_accept_count"] = 1
    panel["verified_success_count"] = 26
    panel.pop("repair_success_ci95")
    panel["manifest_case_hashes"][0] = "f" * 64
    panel["model_specs"] = {"mandated_model_ids": [], "mandated_models": {}}
    panel["models_used"][0]["legacy_small_model"] = True
    _write_sources(tmp_path, panel=panel)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=2.0,
        adversarial_reporter=_clean_report,
    )

    assert artifact["headline_claim_allowed_after_audit"] is False
    assert artifact["exact_check_provenance"]["all_claimed_successes_exact_checked"] is False
    assert artifact["llm_judge_dependency_count"] == 1
    assert artifact["false_accept_count"] == 1
    assert artifact["confidence_interval_present"] is False
    assert artifact["manifest_consistency"]["hashes_match_exp3301"] is False
    assert artifact["model_invocation_summary"]["actual_model_declarations_present"] is False
    boundaries = "\n".join(artifact["claim_boundaries"])
    assert "exact checker" in boundaries
    assert "verified_success_count" in boundaries
    assert "LLM judge" in boundaries
    assert "false accept" in boundaries
    assert "confidence interval" in boundaries
    assert "manifest hashes" in boundaries
    assert "actual invoked model" in boundaries
    assert "Legacy small-model" in boundaries

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.file_status(tmp_path / "not-present.json")["present"] is False
    unreadable = tmp_path / "directory-as-file"
    unreadable.mkdir()
    assert mod.file_status(unreadable)["readable"] is False
    assert mod.run_adversarial_report(Path("x"), lambda _path: []) == {
        "loaded": False,
        "flags": [],
    }
    assert mod.count_value(True) == 0
    assert mod.count_value(4.0) == 4
    assert mod.count_value("5") == 5
    assert mod.count_value("bad") == 0
    assert mod.string_list("bad") == []

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="repair_headline_evidence_audit_ready"):
        mod.validate_artifact(artifact | {"repair_headline_evidence_audit_ready": "true"})
    with pytest.raises(ValueError, match="headline_claim_allowed_after_audit"):
        mod.validate_artifact(artifact | {"headline_claim_allowed_after_audit": "false"})
    with pytest.raises(ValueError, match="audited_artifact"):
        mod.validate_artifact(artifact | {"audited_artifact": ""})
    with pytest.raises(ValueError, match="panel_case_count"):
        mod.validate_artifact(artifact | {"panel_case_count": 29})
    with pytest.raises(ValueError, match="exact_successes_audited"):
        mod.validate_artifact(artifact | {"exact_successes_audited": -1})
    with pytest.raises(ValueError, match="false_accept_count"):
        mod.validate_artifact(artifact | {"false_accept_count": -1})
    with pytest.raises(ValueError, match="llm_judge_dependency_count"):
        mod.validate_artifact(artifact | {"llm_judge_dependency_count": -1})
    with pytest.raises(ValueError, match="adversarial_verify_flags"):
        mod.validate_artifact(artifact | {"adversarial_verify_flags": "bad"})
    with pytest.raises(ValueError, match="substrate_consistency_passed"):
        mod.validate_artifact(artifact | {"substrate_consistency_passed": "bad"})
    with pytest.raises(ValueError, match="confidence_interval_present"):
        mod.validate_artifact(artifact | {"confidence_interval_present": "bad"})
    with pytest.raises(ValueError, match="claim_boundaries"):
        mod.validate_artifact(artifact | {"claim_boundaries": []})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "live_model"})
    with pytest.raises(ValueError, match="random_seed"):
        mod.validate_artifact(artifact | {"random_seed": True})
    with pytest.raises(ValueError, match="duration_s"):
        mod.validate_artifact(artifact | {"duration_s": -1})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
