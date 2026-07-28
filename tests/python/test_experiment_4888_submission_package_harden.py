"""Tests for Exp 4888 ARC-AGI-3 final pre-deadline package hardening.

Spec refs: REQ-CAPSTONE-4888, SCENARIO-CAPSTONE-4888,
SCENARIO-CAPSTONE-4888-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4888-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4888_submission_package_harden as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
MODEL_BYTES = 5_868_826_976


def _preconditions(ok: bool = True, *, scripts_present: bool = True) -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4888": True,
        "packaging_requirements_doc_present": True,
        "prior_4877_ready_package_present": True,
        "submission_packaging_scripts_present": scripts_present,
        "arc_competition_agent_present": scripts_present,
        "ok": ok and scripts_present,
        **({} if ok and scripts_present else {"blocked_resource": "packaging_scripts_missing"}),
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
        "file_hashes": {"main.py": "sha256:main"} if ok else {},
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


def _prior_4877_ready() -> JsonDict:
    return {
        "experiment": "experiment_4877_submission_package_harden",
        "result_path": mod.PRIOR_READY_RELATIVE_PATH,
        "honest_verdict": "success_submission_package_ready",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "vram_estimate_gb": 15.146,
        "package_builds": _package_builds(True),
        "agent_config_resolution": _config_resolution(True),
        "model_path_resolution": _model_resolution(True),
        "vram_breakdown": _vram(True),
        "packaging_requirements_crosscheck": _requirements(True),
    }


def test_req_capstone_4888_spec_declares_final_pre_deadline_contract() -> None:
    """REQ-CAPSTONE-4888: OpenSpec declares the final checklist contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4888",
        "SCENARIO-CAPSTONE-4888",
        "SCENARIO-CAPSTONE-4888-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4888-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIOR_READY_RELATIVE_PATH,
        "success_submission_package_ready_final_pre_deadline",
        "operator_submission_checklist",
        "no regression vs .449",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4888_regression_check_accepts_449_ready_state() -> None:
    """SCENARIO-CAPSTONE-4888: no-regression gate compares against the .449 package."""

    check = mod.diff_against_ready_package(
        _prior_4877_ready(),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        packaging_requirements_crosscheck=_requirements(True),
    )

    assert check["ok"] is True
    assert check["prior_ready_artifact_path"] == mod.PRIOR_READY_RELATIVE_PATH
    assert check["prior_experiment"] == "experiment_4877_submission_package_harden"
    assert check["prior_submission_package_ready"] is True
    assert check["regressions"] == []
    assert check["diff"]["package_sha256"] == "unchanged"

    regressed = mod.diff_against_ready_package(
        _prior_4877_ready(),
        package_builds=_package_builds(False),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(False),
        vram_breakdown=_vram(False),
        packaging_requirements_crosscheck=_requirements(True),
    )
    assert regressed["ok"] is False
    assert "package_still_builds" in regressed["regressions"]
    assert "model_paths_still_resolve" in regressed["regressions"]
    assert "vram_still_fits_16gb" in regressed["regressions"]


def test_scenario_capstone_4888_ready_artifact_is_operator_only_final_checklist() -> None:
    """SCENARIO-CAPSTONE-4888: green gate writes the human operator checklist."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        packaging_requirements_crosscheck=_requirements(True),
        ready_package_regression_check=mod.diff_against_ready_package(
            _prior_4877_ready(),
            package_builds=_package_builds(True),
            agent_config_resolution=_config_resolution(True),
            model_path_resolution=_model_resolution(True),
            vram_breakdown=_vram(True),
            packaging_requirements_crosscheck=_requirements(True),
        ),
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "success_submission_package_ready_final_pre_deadline"
    assert artifact["submission_package_ready"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["vram_estimate_gb"] == 15.146
    assert artifact["package_builds"]["package_builds"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert any("6/30" in step for step in artifact["operator_submission_checklist"])
    assert any("experiment_4877" in step for step in artifact["operator_submission_checklist"])
    assert any(
        "this task never submits" in step.lower()
        for step in artifact["operator_submission_checklist"]
    )
    assert artifact["blocked_resource"] == ""
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4888_missing_model_path_blocks_without_submit_claim() -> None:
    """SCENARIO-CAPSTONE-4888-BLOCKED-PRECONDITION: missing frozen GGUF/server blocks."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(False),
        vram_breakdown=mod.blocked_vram_payload("model_paths"),
        packaging_requirements_crosscheck=_requirements(False),
        ready_package_regression_check={"ok": False, "regressions": ["model_paths_still_resolve"]},
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_model_paths"
    assert artifact["submission_package_ready"] is False
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["blocked_resource"] == "model_paths"
    assert artifact["vram_breakdown"]["blocked_resource"] == "model_paths"
    assert (
        "blocked until this JSON reports success_submission_package_ready_final_pre_deadline"
        in (artifact["operator_submission_checklist"][0])
    )
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4888_not_ready_verdicts_name_failed_gate() -> None:
    """SCENARIO-CAPSTONE-4888-BLOCKED-PRECONDITION: failed non-resource gates are explicit."""

    base = {
        "preconditions_checked": _preconditions(True),
        "package_builds": _package_builds(True),
        "agent_config_resolution": _config_resolution(True),
        "model_path_resolution": _model_resolution(True),
        "vram_breakdown": _vram(True),
        "packaging_requirements_crosscheck": _requirements(True),
        "ready_package_regression_check": {"ok": True, "regressions": []},
        "duration_s": 0.0,
    }

    blocked_precondition = mod.build_artifact(
        **{
            **base,
            "preconditions_checked": {
                **_preconditions(True),
                "ok": False,
                "blocked_resource": "prior_4877_ready_package_present",
            },
        }
    )
    assert blocked_precondition["honest_verdict"] == "blocked_prior_4877_ready_package_present"

    cases = (
        ({"package_builds": _package_builds(False)}, "not_ready_dry_build"),
        ({"agent_config_resolution": _config_resolution(False)}, "not_ready_agent_config"),
        ({"vram_breakdown": _vram(False)}, "not_ready_vram"),
        (
            {"packaging_requirements_crosscheck": _requirements(False)},
            "not_ready_packaging_requirements",
        ),
        (
            {
                "ready_package_regression_check": {
                    "ok": False,
                    "regressions": ["package_still_builds"],
                }
            },
            "not_ready_ready_package_regression_package_still_builds",
        ),
        (
            {"package_builds": {**_package_builds(True), "submitted_to_leaderboard": True}},
            "not_ready_unknown",
        ),
    )

    for overrides, verdict in cases:
        artifact = mod.build_artifact(**{**base, **overrides})
        assert artifact["honest_verdict"] == verdict
        assert artifact["submission_package_ready"] is False


def test_scenario_capstone_4888_preconditions_check_required_files(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4888-BLOCKED-PRECONDITION: filesystem preconditions are explicit."""

    assert mod.read_prior_ready_package(tmp_path) == {}

    missing = mod.check_preconditions(tmp_path)
    assert missing["ok"] is False
    assert missing["blocked_resource"] == "agents_md_read"

    for relative, content in (
        ("AGENTS.md", "instructions"),
        ("CODEX.md", "workflow"),
        (mod.SPEC_RELATIVE_PATH, "REQ-CAPSTONE-4888"),
        (mod.REQUIREMENTS_RELATIVE_PATH, "requirements"),
        (mod.PRIOR_READY_RELATIVE_PATH, json.dumps(_prior_4877_ready())),
        (str(Path(mod.PACKAGE_CORE.KERNEL_RELATIVE_DIR) / mod.PACKAGE_CORE.KERNEL_MAIN), "main"),
        (
            str(Path(mod.PACKAGE_CORE.KERNEL_RELATIVE_DIR) / mod.PACKAGE_CORE.KERNEL_METADATA),
            "{}",
        ),
        (str(mod.PACKAGE_CORE.AGENT_RELATIVE_PATH), "agent"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    ready = mod.check_preconditions(tmp_path)
    assert ready["ok"] is True
    assert ready["spec_has_req_4888"] is True
    assert ready["prior_4877_ready_package_present"] is True


def test_scenario_capstone_4888_run_exits_on_missing_packaging_scripts(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4888-BLOCKED-PRECONDITION: missing package assets skip probes."""

    ticks = iter((200.0, 200.5))

    artifact = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(False, scripts_present=False),
        write=False,
        now=lambda: next(ticks),
    )

    assert artifact["honest_verdict"] == "blocked_packaging_scripts_missing"
    assert artifact["package_builds"] == mod.blocked_package_builds_payload()
    assert artifact["packaging_requirements_crosscheck"] == mod.blocked_requirements_payload()
    assert artifact["ready_package_regression_check"]["prior_artifact_present"] is False
    assert artifact["duration_s"] == 0.5


def test_scenario_capstone_4888_run_blocks_missing_model_paths(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4888-BLOCKED-PRECONDITION: unresolved GGUF/server exits blocked."""

    ticks = iter((300.0, 300.25))
    prior_path = tmp_path / mod.PRIOR_READY_RELATIVE_PATH
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(json.dumps(_prior_4877_ready()), encoding="utf-8")

    artifact = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        package_builder=lambda _root: _package_builds(True),
        agent_config_resolver=lambda: _config_resolution(True),
        model_path_resolver=lambda: _model_resolution(False),
        vram_estimator=lambda _paths, _config: _vram(True),
        requirements_crosschecker=lambda _root, **_kwargs: _requirements(False),
        write=False,
        now=lambda: next(ticks),
    )

    assert artifact["honest_verdict"] == "blocked_model_paths"
    assert artifact["vram_breakdown"] == mod.blocked_vram_payload("model_paths")
    assert (
        artifact["ready_package_regression_check"]["checks"]["model_paths_still_resolve"] is False
    )
    assert artifact["duration_s"] == 0.25


def test_req_capstone_4888_schema_rejects_false_ready_submission_and_checksum() -> None:
    """REQ-CAPSTONE-4888: schema guards final readiness and no-submit state."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        packaging_requirements_crosscheck=_requirements(True),
        ready_package_regression_check=mod.diff_against_ready_package(
            _prior_4877_ready(),
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
        ("operator_submission_checklist", ["submit now"], "operator_submission_checklist"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("operator_only", False, "operator_only"),
        ("vram_estimate_gb", 99.0, "vram_estimate_gb"),
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
    del missing["operator_submission_checklist"]
    assert "operator_submission_checklist" in mod.artifact_schema_errors(missing)


def test_scenario_capstone_4888_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4888: runner writes the final JSON without submitting."""

    ticks = iter((100.0, 100.25))
    prior = _prior_4877_ready()
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
    assert artifact["honest_verdict"] == "success_submission_package_ready_final_pre_deadline"
    assert artifact["submitted_to_leaderboard"] is False
