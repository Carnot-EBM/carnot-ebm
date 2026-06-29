"""Tests for Exp 4997 FINAL pre-deadline submission package hardening.

Spec refs: REQ-CAPSTONE-4997, SCENARIO-CAPSTONE-4997,
SCENARIO-CAPSTONE-4997-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4997-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4997_submission_package_harden as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
MODEL_BYTES = 5_868_826_976


def _preconditions(ok: bool = True, *, scripts_present: bool = True) -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4997": True,
        "prior_4986_ready_package_present": ok,
        "prior_4986_frozen_stack_evidence_present": ok,
        "packaging_requirements_doc_present": True,
        "package_build_path_present": scripts_present,
        "submission_packaging_scripts_present": scripts_present,
        "arc_competition_agent_present": scripts_present,
        "ok": ok and scripts_present,
        **({} if ok and scripts_present else {"blocked_resource": "prior_4986_ready_package"}),
    }


def _package_build_check(ok: bool = True) -> JsonDict:
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
        "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "repo_substr": "Qwen3.5-9B-MTP",
        "model_filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "model_path_env": "CARNOT_ARC_GGUF_PATH",
        "server_path_env": "CARNOT_LLAMA_SERVER",
        "llama_server_kind": "cuda-12.8-binary",
        "mtp": ok,
        "spec_type": "draft-mtp",
        "kv_quant": "q8_0",
        "max_tokens": 2560,
        "n_predict_min": 2048,
        "checks": {"submitted_policy_e3": True, "model_is_qwen35_mtp": ok},
    }


def _model_resolution(ok: bool = True) -> JsonDict:
    return {
        "resolved": ok,
        "blocked_resource": "" if ok else "model_paths",
        "gguf": {
            "path": "/cache/Qwen3.5-9B-Q4_K_M.gguf" if ok else "",
            "filename": "Qwen3.5-9B-Q4_K_M.gguf" if ok else "",
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


def _stack_load(ok: bool = True, *, peak: float = 15.146, resource: str = "") -> JsonDict:
    blocked = "" if ok else (resource or "frozen_stack_load")
    return {
        "ok": ok,
        "frozen_stack_loads": ok,
        "blocked_resource": blocked,
        "peak_vram_gb": peak,
        "fits_16gb": ok and peak < mod.VRAM_LIMIT_GB,
        "limit_gb": mod.VRAM_LIMIT_GB,
        "server_reachable": ok,
        "igpu_hip_server_present": ok,
        "generator_backend": "igpu_hip" if ok else None,
        "uses_3090": False,
        "mtp": True,
        "kv_quant": "q8_0",
        "measurement_source": "experiment_4986_ready_artifact",
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


def _prior_4986_ready() -> JsonDict:
    return {
        "experiment": "experiment_4986_submission_package_harden",
        "result_path": mod.PRIOR_READY_RELATIVE_PATH,
        "honest_verdict": "success_submission_package_ready_final_pre_deadline",
        "submission_package_ready": True,
        "submits": False,
        "operator_only": True,
        "submitted_to_leaderboard": False,
        "peak_vram_gb": 15.146,
        "frozen_stack_loads": True,
        "package_build_check": _package_build_check(True),
        "agent_config_resolution": _config_resolution(True),
        "model_path_resolution": _model_resolution(True),
        "vram_breakdown": _vram(True),
        "frozen_stack_load_check": _stack_load(True),
        "packaging_requirements_crosscheck": _requirements(True),
        "ready_package_regression_check": {"ok": True, "regressions": []},
    }


def _ready_regression() -> JsonDict:
    return mod.confirm_ready_package_regression(
        _prior_4986_ready(),
        package_build_check=_package_build_check(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        frozen_stack_load_check=_stack_load(True),
        packaging_requirements_crosscheck=_requirements(True),
    )


def _ready_artifact() -> JsonDict:
    return mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_build_check=_package_build_check(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        frozen_stack_load_check=_stack_load(True),
        packaging_requirements_crosscheck=_requirements(True),
        ready_package_regression_check=_ready_regression(),
        duration_s=0.0,
    )


def test_req_capstone_4997_spec_declares_final_operator_only_package_contract() -> None:
    """REQ-CAPSTONE-4997: OpenSpec declares the final hardening contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4997",
        "SCENARIO-CAPSTONE-4997",
        "SCENARIO-CAPSTONE-4997-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4997-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIOR_READY_RELATIVE_PATH,
        "success_submission_package_ready_final_pre_deadline",
        "operator_submission_checklist",
        "submits=false",
        "operator_only=true",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4997_regression_check_accepts_459_ready_state() -> None:
    """SCENARIO-CAPSTONE-4997: regression accepts the green .459 package."""

    check = _ready_regression()

    assert check["ok"] is True
    assert check["prior_ready_artifact_path"] == mod.PRIOR_READY_RELATIVE_PATH
    assert check["prior_experiment"] == "experiment_4986_submission_package_harden"
    assert check["prior_submission_package_ready"] is True
    assert check["checks"]["submits_still_false"] is True
    assert check["regressions"] == []
    assert check["diff"]["package_sha256"] == "unchanged"
    assert check["diff"]["peak_vram_gb"] == "unchanged"

    regressed = mod.confirm_ready_package_regression(
        {**_prior_4986_ready(), "submits": True},
        package_build_check=_package_build_check(False),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(False),
        frozen_stack_load_check=_stack_load(False, peak=17.0, resource="frozen_stack_load"),
        packaging_requirements_crosscheck=_requirements(False),
    )
    assert regressed["ok"] is False
    assert "package_still_builds" in regressed["regressions"]
    assert "requirements_still_pass" in regressed["regressions"]
    assert "vram_still_fits_16gb" in regressed["regressions"]
    assert "frozen_stack_still_loads" in regressed["regressions"]
    assert "peak_vram_still_fits_16gb" in regressed["regressions"]
    assert "submits_still_false" in regressed["regressions"]


def test_scenario_capstone_4997_ready_artifact_is_operator_only_no_submit() -> None:
    """SCENARIO-CAPSTONE-4997: green gate writes a ready operator-only checklist."""

    artifact = _ready_artifact()

    assert artifact["honest_verdict"] == "success_submission_package_ready_final_pre_deadline"
    assert artifact["submission_package_ready"] is True
    assert artifact["submits"] is False
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["peak_vram_gb"] == 15.146
    assert artifact["frozen_stack_loads"] is True
    assert artifact["package_builds"] is True
    assert artifact["package_build_check"]["package_builds"] is True
    assert artifact["ready_package_regression_ok"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == mod.AGGREGATION_DURATION_FLOOR_S
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["blocked_resource"] == ""
    assert any("OPERATOR" in step for step in artifact["operator_submission_checklist"])
    assert any("experiment_4986" in step for step in artifact["operator_submission_checklist"])
    assert any("FINAL" in step for step in artifact["operator_submission_checklist"])
    assert any(
        "never submits" in step.lower() for step in artifact["operator_submission_checklist"]
    )
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4997_missing_frozen_stack_evidence_blocks() -> None:
    """SCENARIO-CAPSTONE-4997-BLOCKED-PRECONDITION: missing stack evidence blocks."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(False),
        package_build_check=_package_build_check(True),
        agent_config_resolution=_config_resolution(True),
        model_path_resolution=_model_resolution(True),
        vram_breakdown=_vram(True),
        frozen_stack_load_check=_stack_load(False, resource="frozen_stack_evidence"),
        packaging_requirements_crosscheck=_requirements(True),
        ready_package_regression_check={"ok": False, "regressions": ["frozen_stack_still_loads"]},
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_prior_4986_ready_package"
    assert artifact["submission_package_ready"] is False
    assert artifact["frozen_stack_loads"] is False
    assert artifact["package_builds"] is True
    assert artifact["submits"] is False
    assert artifact["operator_only"] is True
    assert (
        "blocked until this JSON reports success_submission_package_ready_final_pre_deadline"
        in (artifact["operator_submission_checklist"][0])
    )
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4997_not_ready_verdicts_name_failed_gate() -> None:
    """SCENARIO-CAPSTONE-4997-BLOCKED-PRECONDITION: failed gates are explicit."""

    base = {
        "preconditions_checked": _preconditions(True),
        "package_build_check": _package_build_check(True),
        "agent_config_resolution": _config_resolution(True),
        "model_path_resolution": _model_resolution(True),
        "vram_breakdown": _vram(True),
        "frozen_stack_load_check": _stack_load(True),
        "packaging_requirements_crosscheck": _requirements(True),
        "ready_package_regression_check": {"ok": True, "regressions": []},
        "duration_s": 0.0,
    }

    cases = (
        ({"package_build_check": _package_build_check(False)}, "not_ready_dry_build"),
        ({"agent_config_resolution": _config_resolution(False)}, "not_ready_agent_config"),
        ({"model_path_resolution": _model_resolution(False)}, "blocked_model_paths"),
        ({"vram_breakdown": _vram(False)}, "not_ready_vram"),
        (
            {"frozen_stack_load_check": _stack_load(False, resource="frozen_stack_load")},
            "not_ready_frozen_stack_load",
        ),
        ({"frozen_stack_load_check": _stack_load(True, peak=16.25)}, "not_ready_peak_vram"),
        (
            {"packaging_requirements_crosscheck": _requirements(False)},
            "not_ready_packaging_requirements",
        ),
        (
            {
                "ready_package_regression_check": {
                    "ok": False,
                    "regressions": ["submits_still_false"],
                }
            },
            "not_ready_ready_package_regression_submits_still_false",
        ),
        (
            {
                "package_build_check": {
                    **_package_build_check(True),
                    "submitted_to_leaderboard": True,
                }
            },
            "not_ready_submission_boundary",
        ),
        (
            {"frozen_stack_load_check": {**_stack_load(True), "uses_3090": True}},
            "not_ready_unknown",
        ),
    )

    for overrides, verdict in cases:
        artifact = mod.build_artifact(**{**base, **overrides})
        assert artifact["honest_verdict"] == verdict
        assert artifact["submission_package_ready"] is False


def test_scenario_capstone_4997_preconditions_check_required_files(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4997-BLOCKED-PRECONDITION: filesystem checks are explicit."""

    assert mod.read_prior_ready_package(tmp_path) == {}
    non_object_prior = tmp_path / mod.PRIOR_READY_RELATIVE_PATH
    non_object_prior.parent.mkdir(parents=True, exist_ok=True)
    non_object_prior.write_text("[]", encoding="utf-8")
    assert mod.read_prior_ready_package(tmp_path) == {}
    non_object_prior.unlink()

    missing = mod.check_preconditions(tmp_path)
    assert missing["ok"] is False
    assert missing["blocked_resource"] == "agents_md_read"

    for relative, content in (
        ("AGENTS.md", "instructions"),
        ("CODEX.md", "workflow"),
        (mod.SPEC_RELATIVE_PATH, "REQ-CAPSTONE-4997"),
        (mod.REQUIREMENTS_RELATIVE_PATH, "requirements"),
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

    no_prior = mod.check_preconditions(tmp_path)
    assert no_prior["ok"] is False
    assert no_prior["blocked_resource"] == "prior_4986_ready_package"

    prior_path = tmp_path / mod.PRIOR_READY_RELATIVE_PATH
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(
        json.dumps({**_prior_4986_ready(), "frozen_stack_loads": False}), encoding="utf-8"
    )
    no_stack_evidence = mod.check_preconditions(tmp_path)
    assert no_stack_evidence["ok"] is False
    assert no_stack_evidence["blocked_resource"] == "frozen_stack_evidence"

    prior_path.write_text(json.dumps(_prior_4986_ready()), encoding="utf-8")
    ready = mod.check_preconditions(tmp_path)
    assert (
        mod.read_prior_ready_package(tmp_path)["experiment"]
        == "experiment_4986_submission_package_harden"
    )
    assert ready["ok"] is True
    assert ready["spec_has_req_4997"] is True
    assert ready["prior_4986_ready_package_present"] is True
    assert ready["prior_4986_frozen_stack_evidence_present"] is True


def test_scenario_capstone_4997_run_paths_write_or_block(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4997: runner writes the final JSON without submitting."""

    blocked_ticks = iter((200.0, 200.5))
    blocked = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(False, scripts_present=False),
        write=False,
        now=lambda: next(blocked_ticks),
    )
    assert blocked["honest_verdict"] == "blocked_packaging_scripts_missing"
    assert blocked["package_build_check"] == mod.blocked_package_build_check(
        "packaging_scripts_missing"
    )
    assert blocked["ready_package_regression_check"]["prior_artifact_present"] is False
    assert blocked["duration_s"] == 0.5

    missing_prior_ticks = iter((250.0, 250.25))
    missing_prior_evidence = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        package_builder=lambda _root: _package_build_check(True),
        agent_config_resolver=lambda: _config_resolution(True),
        model_path_resolver=lambda: _model_resolution(True),
        requirements_crosschecker=lambda _root, **_kwargs: _requirements(True),
        write=False,
        now=lambda: next(missing_prior_ticks),
    )
    assert missing_prior_evidence["vram_breakdown"]["blocked_resource"] == (
        "prior_4986_ready_package"
    )
    assert missing_prior_evidence["frozen_stack_load_check"]["blocked_resource"] == (
        "frozen_stack_evidence"
    )

    prior_path = tmp_path / mod.PRIOR_READY_RELATIVE_PATH
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(json.dumps(_prior_4986_ready()), encoding="utf-8")

    model_ticks = iter((300.0, 300.25))
    missing_model = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        package_builder=lambda _root: _package_build_check(True),
        agent_config_resolver=lambda: _config_resolution(True),
        model_path_resolver=lambda: _model_resolution(False),
        requirements_crosschecker=lambda _root, **_kwargs: _requirements(False),
        write=False,
        now=lambda: next(model_ticks),
    )
    assert missing_model["honest_verdict"] == "blocked_model_paths"
    assert (
        missing_model["ready_package_regression_check"]["checks"]["model_paths_still_resolve"]
        is False
    )
    assert missing_model["duration_s"] == 0.25

    ready_ticks = iter((100.0, 100.25))
    artifact = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        package_builder=lambda _root: _package_build_check(True),
        agent_config_resolver=lambda: _config_resolution(True),
        model_path_resolver=lambda: _model_resolution(True),
        requirements_crosschecker=lambda _root, **_kwargs: _requirements(True),
        write=True,
        now=lambda: next(ready_ticks),
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["duration_s"] == 0.25
    assert artifact["honest_verdict"] == "success_submission_package_ready_final_pre_deadline"
    assert artifact["submits"] is False


def test_req_capstone_4997_schema_rejects_false_ready_submission_and_checksum() -> None:
    """REQ-CAPSTONE-4997: schema guards readiness and operator-only state."""

    artifact = _ready_artifact()

    for field, bad_value, expected in (
        ("honest_verdict", "ready", "honest_verdict"),
        ("submits", True, "submits"),
        ("operator_only", False, "operator_only"),
        ("submitted_to_leaderboard", True, "submitted_to_leaderboard"),
        ("package_builds", False, "package_builds"),
        (
            "package_build_check",
            {**artifact["package_build_check"], "submitted_to_leaderboard": True},
            "package_build_check_submitted_to_leaderboard",
        ),
        ("operator_submission_checklist", ["submit now"], "operator_submission_checklist"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("field_principles", {}, "field_principles"),
        ("result_path", "wrong.json", "result_path"),
        ("peak_vram_gb", 99.0, "peak_vram_gb"),
        ("frozen_stack_loads", False, "frozen_stack_loads"),
        ("ready_package_regression_ok", False, "ready_package_regression_ok"),
        ("duration_s", 0.0, "duration_s_aggregation_floor"),
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
