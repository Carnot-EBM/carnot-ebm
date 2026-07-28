"""Tests for Exp 4856 ARC-AGI-3 package re-verification.

Spec refs: REQ-CAPSTONE-4856, SCENARIO-CAPSTONE-4856,
SCENARIO-CAPSTONE-4856-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4856-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4856_submission_package_harden as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
MODEL_BYTES = 5_868_826_976


def _preconditions(ok: bool = True, *, scripts_present: bool = True) -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4856": True,
        "packaging_requirements_doc_present": True,
        "prior_4846_ready_package_present": True,
        "submission_packaging_scripts_present": scripts_present,
        "arc_competition_agent_present": scripts_present,
        "ok": ok and scripts_present,
        **({} if ok and scripts_present else {"blocked_resource": "submission_packaging_scripts"}),
    }


def _package_builds(ok: bool = True) -> JsonDict:
    return {
        "dry_build_ran": True,
        "package_builds": ok,
        "entrypoint_compiles": ok,
        "manifest_present": ok,
        "kernel_main_present": ok,
        "submitted_to_leaderboard": False,
        "blocked_resource": "" if ok else "dry_build",
        "package_sha256": "sha256:package",
        "files": ["kernel-metadata.json", "main.py"] if ok else [],
    }


def _config_resolution(ok: bool = True) -> JsonDict:
    return {
        "resolved": ok,
        "blocked_resource": "" if ok else "agent_config",
        "model_id": "unsloth/gemma-4-31B-it-GGUF",
        "repo_substr": "gemma-4-31B-it",
        "model_filename": "gemma-4-31B-it-Q4_K_M.gguf",
        "model_path_env": "CARNOT_ARC_GGUF_PATH",
        "server_path_env": "CARNOT_LLAMA_SERVER",
        "llama_server_kind": "cuda-12.8-binary",
        "mtp": True,
        "spec_type": "draft-mtp",
        "kv_quant": "q8_0",
        "max_tokens": 2560,
        "n_predict_min": 2048,
        "checks": {
            "submitted_policy_e3": True,
            "submitted_cascade": True,
            "model_is_pinned_generator": True,
            "model_filename": True,
            "mtp_enabled": ok,
            "q8_kv": True,
            "no_think": True,
            "n_predict_floor": True,
            "cuda_128_server": True,
            "binary_not_wheel": True,
        },
    }


def _model_resolution(ok: bool = True) -> JsonDict:
    return {
        "resolved": ok,
        "blocked_resource": "" if ok else "model_paths",
        "gguf": {
            "path": "/cache/Qwen3.5-9B-Q4_K_M.gguf" if ok else "",
            "filename": "gemma-4-31B-it-Q4_K_M.gguf" if ok else "",
            "present": ok,
            "size_bytes": MODEL_BYTES if ok else 0,
            "size_gb": 5.868827 if ok else 0.0,
        },
        "llama_server": {
            "path": "/cache/llama-server" if ok else "",
            "filename": "llama-server" if ok else "",
            "present": ok,
            "cuda_12_8_capable": ok,
            "kind": "cuda-12.8-binary",
        },
    }


def _vram(fits: bool = True) -> JsonDict:
    total = 15.146 if fits else 17.0
    return {
        "vram_estimate_gb": total,
        "fits_16gb": fits,
        "limit_gb": 16.0,
        "remaining_headroom_gb": round(16.0 - total, 3),
        "model_copies": 2,
        "model_weights_gb": 5.869,
        "draft_model_weights_gb": 5.869,
        "kv_cache_gb": 1.208,
        "kv_quant": "q8_0",
        "context_tokens": 16_384,
        "runtime_overhead_gb": 0.7,
        "required_headroom_gb": 1.5,
        "total_with_headroom_gb": total,
    }


def _requirements(ok: bool = True) -> JsonDict:
    return {
        "requirements_doc_path": mod.REQUIREMENTS_RELATIVE_PATH,
        "doc_present": True,
        "ok": ok,
        "blocked_resource": "" if ok else "packaging_requirements",
        "checks": {key: ok for key in mod.REQUIREMENTS_CHECK_KEYS},
        "notes": ["operator-only requirements package cross-check passed"],
    }


def _prior_4846_ready() -> JsonDict:
    return {
        "experiment": "experiment_4846_submission_package_harden",
        "result_path": mod.PRIOR_READY_RELATIVE_PATH,
        "honest_verdict": "success_package_builds_vram_gate_green",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "vram_estimate_gb": 15.146,
        "package_builds": _package_builds(True),
        "agent_config_resolution": _config_resolution(True),
        "model_path_resolution": _model_resolution(True),
        "vram_breakdown": _vram(True),
        "packaging_requirements_crosscheck": _requirements(True),
    }


def test_req_capstone_4856_spec_declares_reverification_contract() -> None:
    """REQ-CAPSTONE-4856: OpenSpec declares the re-verification contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4856",
        "SCENARIO-CAPSTONE-4856",
        "SCENARIO-CAPSTONE-4856-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4856-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIOR_READY_RELATIVE_PATH,
        "ready_package_regression_check",
        "success_submission_package_ready",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4856_regression_check_accepts_446_ready_state() -> None:
    """SCENARIO-CAPSTONE-4856: .446 ready-state diff stays green when gates still pass."""

    check = mod.diff_against_ready_package(
        _prior_4846_ready(),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        packaging_requirements_crosscheck=_requirements(True),
    )

    assert check["ok"] is True
    assert check["prior_submission_package_ready"] is True
    assert check["prior_vram_estimate_gb"] == 15.146
    assert check["current_vram_estimate_gb"] == 15.146
    assert check["regressions"] == []
    assert check["diff"]["package_sha256"] == "unchanged"

    regressed = mod.diff_against_ready_package(
        _prior_4846_ready(),
        package_builds=_package_builds(False),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(False),
        packaging_requirements_crosscheck=_requirements(True),
    )
    assert regressed["ok"] is False
    assert "package_still_builds" in regressed["regressions"]
    assert "vram_still_fits_16gb" in regressed["regressions"]


def test_scenario_capstone_4856_ready_artifact_is_operator_only() -> None:
    """SCENARIO-CAPSTONE-4856: green gate writes the final operator-only checklist."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        packaging_requirements_crosscheck=_requirements(True),
        ready_package_regression_check=mod.diff_against_ready_package(
            _prior_4846_ready(),
            package_builds=_package_builds(True),
            agent_config_resolution=_config_resolution(True),
            model_path_resolution=_model_resolution(True),
            vram_breakdown=_vram(True),
            packaging_requirements_crosscheck=_requirements(True),
        ),
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "success_submission_package_ready"
    assert artifact["submission_package_ready"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["vram_estimate_gb"] == 15.146
    assert artifact["package_builds"]["package_builds"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert any("this task never submits" in step.lower() for step in artifact["operator_checklist"])
    assert any("experiment_4846" in step for step in artifact["operator_checklist"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4856_blocks_missing_packaging_without_submit_claim() -> None:
    """SCENARIO-CAPSTONE-4856-BLOCKED-PRECONDITION: missing package assets block honestly."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(False, scripts_present=False),
        package_builds=mod.blocked_package_builds_payload(),
        agent_config_resolution={},
        model_path_resolution={},
        vram_breakdown={},
        packaging_requirements_crosscheck=_requirements(False),
        ready_package_regression_check={"ok": False, "regressions": ["precondition"]},
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_packaging_scripts_missing"
    assert artifact["submission_package_ready"] is False
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["package_builds"]["dry_build_ran"] is False
    assert (
        "blocked until this JSON reports success_submission_package_ready"
        in (artifact["operator_checklist"][0])
    )
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4856_schema_rejects_false_ready_submission_and_checksum() -> None:
    """REQ-CAPSTONE-4856: schema guards readiness, required fields, and no-submit state."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        packaging_requirements_crosscheck=_requirements(True),
        ready_package_regression_check=mod.diff_against_ready_package(
            _prior_4846_ready(),
            package_builds=_package_builds(True),
            agent_config_resolution=_config_resolution(True),
            model_path_resolution=_model_resolution(True),
            vram_breakdown=_vram(True),
            packaging_requirements_crosscheck=_requirements(True),
        ),
        duration_s=0.0,
    )

    for field, bad_value, expected in (
        ("honest_verdict", "ready", "honest_verdict"),
        ("submitted_to_leaderboard", True, "submitted_to_leaderboard"),
        (
            "package_builds",
            {**artifact["package_builds"], "submitted_to_leaderboard": True},
            "package_builds_submitted_to_leaderboard",
        ),
        ("operator_checklist", ["submit now"], "operator_checklist"),
        ("field_principles", {}, "field_principles"),
        ("result_path", "wrong.json", "result_path"),
    ):
        malformed = dict(artifact, **{field: bad_value})
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert expected in mod.artifact_schema_errors(malformed)

    false_ready = dict(
        artifact,
        ready_package_regression_check={"ok": False, "regressions": ["package_still_builds"]},
    )
    false_ready["reproducibility_checksum"] = mod.payload_checksum(false_ready)
    assert "submission_package_ready_gate" in mod.artifact_schema_errors(false_ready)

    bad_checksum = dict(artifact, reproducibility_checksum="sha256:bad")
    assert "reproducibility_checksum" in mod.artifact_schema_errors(bad_checksum)

    missing = dict(artifact)
    del missing["package_builds"]
    assert "package_builds" in mod.artifact_schema_errors(missing)


def test_scenario_capstone_4856_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4856: runner writes the re-verification JSON without submitting."""

    ticks = iter((100.0, 100.25))
    prior = _prior_4846_ready()
    prior_path = tmp_path / mod.PRIOR_READY_RELATIVE_PATH
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(json.dumps(prior), encoding="utf-8")

    artifact = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        package_builder=lambda _root: _package_builds(True),
        agent_config_resolver=lambda: _config_resolution(True),
        model_path_resolver=lambda: _model_resolution(True),
        vram_estimator=lambda _paths, _config: _vram(True),
        requirements_crosschecker=lambda _root, **_kwargs: _requirements(True),
        write=True,
        now=lambda: next(ticks),
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["duration_s"] == 0.25
    assert artifact["honest_verdict"] == "success_submission_package_ready"
    assert artifact["submitted_to_leaderboard"] is False
