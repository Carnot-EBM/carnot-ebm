"""Tests for Exp 4836 ARC-AGI-3 submission-package hardening.

Spec refs: REQ-CAPSTONE-4836, SCENARIO-CAPSTONE-4836,
SCENARIO-CAPSTONE-4836-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4836-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4836_submission_package_harden as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
MODEL_BYTES = 5_868_826_976


def _submitted_agent_config_fixture(*, a1_enabled: bool = False) -> JsonDict:
    return {
        "policy": "E3AgentPolicy",
        "cascade": True,
        "amortized_first_contact_prior_enabled": a1_enabled,
        "go_explore_archive_enabled": a1_enabled,
        "frozen_generator": {
            "model_id": "unsloth/gemma-4-31B-it-GGUF",
            "repo_substr": "gemma-4-31B-it",
            "model_filename": "gemma-4-31B-it-Q4_K_M.gguf",
            "model_path_env": "CARNOT_ARC_GGUF_PATH",
            "server_path_env": "CARNOT_LLAMA_SERVER",
            "llama_server_kind": "cuda-12.8-binary",
            "binary_not_wheel": True,
            "mtp": True,
            "spec_type": "draft-mtp",
            "kv_quant": "q8_0",
            "no_think_prefix": "",
            "max_tokens": 2560,
            "n_predict_min": 2048,
            "wheel_fallback_allowed": False,
        },
    }


def _a1_prior_payload(*, passed: bool) -> JsonDict:
    return {
        "honest_verdict": (
            "success_amortized_prior_raises_first_win_above_baseline"
            if passed
            else "complete_amortized_prior_no_first_win_lift_l1_wall_survives"
        ),
        "baseline_first_win_rate": 0.04,
        "prior_changed_proposals": True,
        "first_win_rate_with_prior": 0.08 if passed else 0.0,
        "first_win_rate_no_prior_ablation": 0.04 if passed else 0.0,
        "first_win_delta_ci95": {"low": 0.01 if passed else 0.0, "high": 0.09},
        "go_explore_archive_alive": {
            "alive": True,
            "observations": 2,
            "stored_cells": 2,
            "prefixes_injected": 1,
        },
        "imitation_control_heldout_games": {
            "heldout_not_in_distillation_set": True,
            "lift_holds": passed,
        },
    }


def _write_a1_prior(root: Path, *, passed: bool) -> None:
    path = root / mod.A1_PRIOR_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_a1_prior_payload(passed=passed)), encoding="utf-8")


def _write_submission_kernel(root: Path) -> None:
    kernel = root / "scripts" / "kaggle" / "submission_kernel"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel-metadata.json").write_text(
        json.dumps({"code_file": "main.py", "kernel_type": "script"}),
        encoding="utf-8",
    )
    (kernel / "main.py").write_text(
        "print('Qwen3.5-9B-MTP draft-mtp q8_0 llama-server')\n",
        encoding="utf-8",
    )


def _write_repo_preconditions(root: Path, *, with_spec: bool = True) -> None:
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = root / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-CAPSTONE-4836\n" if with_spec else "REQ-CAPSTONE-OTHER\n",
        encoding="utf-8",
    )
    agent = root / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
    agent.parent.mkdir(parents=True, exist_ok=True)
    agent.write_text("SUBMITTED_AGENT_CONFIG = {}\n", encoding="utf-8")
    _write_submission_kernel(root)


def _preconditions(ok: bool = True) -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4836": True,
        "submission_kernel_present": True,
        "arc_competition_agent_present": True,
        "ok": ok,
        **({} if ok else {"blocked_resource": "arc_competition_agent"}),
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
        "package_sha256": "sha256:abc" if ok else "",
        "files": ["kernel-metadata.json", "main.py"] if ok else [],
    }


def _config_resolution(*, ok: bool = True, a1_enabled: bool = False) -> JsonDict:
    config = mod.resolve_agent_config(_submitted_agent_config_fixture(a1_enabled=a1_enabled))
    if ok:
        return config
    return {
        **config,
        "resolved": False,
        "checks": {**config["checks"], "mtp_enabled": False},
        "blocked_resource": "agent_config",
    }


def _model_resolution(tmp_path: Path, ok: bool = True) -> JsonDict:
    gguf = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    server = tmp_path / "llama-server"
    gguf.write_text("fixture\n", encoding="utf-8")
    server.write_text("server\n", encoding="utf-8")
    return mod.resolve_model_paths(
        gguf_paths=[str(gguf)] if ok else [],
        llama_server_paths=[str(server)],
        cuda_inspector=lambda _path: True,
        model_size_bytes=MODEL_BYTES,
    )


def _ready_vram() -> JsonDict:
    return mod.estimate_vram(
        model_size_bytes=MODEL_BYTES,
        mtp_enabled=True,
        kv_quant="q8_0",
        context_tokens=16_384,
    )


def test_req_capstone_4836_spec_declares_operator_package_contract() -> None:
    """REQ-CAPSTONE-4836: OpenSpec declares the package-hardening contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4836",
        "SCENARIO-CAPSTONE-4836",
        "SCENARIO-CAPSTONE-4836-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4836-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        "a1_prior_inclusion",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4836_preconditions_and_dry_build(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4836: package scripts and agent file dry-build cleanly."""

    _write_repo_preconditions(tmp_path)

    preconditions = mod.check_preconditions(tmp_path)
    package_builds = mod.dry_build_package(tmp_path)

    assert preconditions["ok"] is True
    assert preconditions["spec_has_req_4836"] is True
    assert package_builds["package_builds"] is True
    assert package_builds["entrypoint_compiles"] is True
    assert package_builds["submitted_to_leaderboard"] is False
    assert package_builds["package_sha256"].startswith("sha256:")

    _write_repo_preconditions(tmp_path, with_spec=False)
    blocked = mod.check_preconditions(tmp_path)
    assert blocked["ok"] is False
    assert blocked["blocked_resource"] == "spec_has_req_4836"


def test_scenario_capstone_4836_frozen_stack_a1_decision_and_vram_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4836: frozen Qwen/MTP/q8/CUDA stack fits and A1 is gated."""

    _write_a1_prior(tmp_path, passed=False)

    config = _config_resolution(a1_enabled=False)
    paths = _model_resolution(tmp_path, ok=True)
    vram = _ready_vram()
    a1 = mod.resolve_a1_prior_inclusion(tmp_path, config)

    assert config["resolved"] is True
    assert config["checks"]["model_is_pinned_generator"] is True
    assert config["checks"]["mtp_enabled"] is True
    assert config["checks"]["q8_kv"] is True
    assert config["checks"]["cuda_128_server"] is True
    assert config["amortized_first_contact_prior_enabled"] is False
    assert paths["resolved"] is True
    assert paths["gguf"]["filename"] == "gemma-4-31B-it-Q4_K_M.gguf"
    assert paths["llama_server"]["cuda_12_8_capable"] is True
    assert vram["fits_16gb"] is True
    assert vram["model_copies"] == 2
    assert 14.0 < vram["vram_estimate_gb"] < 16.0
    assert vram["remaining_headroom_gb"] > 0.0
    assert mod._runtime_vram_estimate(paths, config) == vram
    assert a1["passed"] is False
    assert a1["included"] is False
    assert a1["reason"] == "not_included_a1_prior_did_not_pass"


def test_scenario_capstone_4836_a1_pass_requires_frozen_config_inclusion(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4836: passed A1 prior must be present in the frozen config."""

    _write_a1_prior(tmp_path, passed=True)
    disabled_config = _config_resolution(a1_enabled=False)
    enabled_config = _config_resolution(a1_enabled=True)

    disabled_a1 = mod.resolve_a1_prior_inclusion(tmp_path, disabled_config)
    enabled_a1 = mod.resolve_a1_prior_inclusion(tmp_path, enabled_config)

    assert disabled_a1["passed"] is True
    assert disabled_a1["included"] is False
    assert disabled_a1["reason"] == "passed_but_not_in_frozen_config"
    assert enabled_a1["passed"] is True
    assert enabled_a1["included"] is True
    assert enabled_a1["reason"] == "passed_and_included_in_frozen_config"

    blocked = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=disabled_config,
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=_ready_vram(),
        a1_prior_inclusion=disabled_a1,
        duration_s=0.0,
    )
    ready = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=enabled_config,
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=_ready_vram(),
        a1_prior_inclusion=enabled_a1,
        duration_s=0.0,
    )

    assert blocked["honest_verdict"] == "complete_package_not_ready_a1_prior_inclusion"
    assert blocked["submission_package_ready"] is False
    assert ready["honest_verdict"] == "success_package_builds_vram_gate_green"
    assert ready["submission_package_ready"] is True


def test_scenario_capstone_4836_ready_artifact_is_operator_only(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4836: green package gate marks operator-ready only."""

    _write_a1_prior(tmp_path, passed=False)
    config = _config_resolution(a1_enabled=False)
    vram = _ready_vram()
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=config,
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=vram,
        a1_prior_inclusion=mod.resolve_a1_prior_inclusion(tmp_path, config),
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "success_package_builds_vram_gate_green"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["submission_package_ready"] is True
    assert artifact["vram_estimate_gb"] == vram["vram_estimate_gb"]
    assert artifact["vram_estimate_gb"] < 16.0
    assert artifact["a1_prior_inclusion"]["included"] is False
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert all(step.startswith("OPERATOR-CHECK:") for step in artifact["operator_checklist"])
    assert any("this task never submits" in step.lower() for step in artifact["operator_checklist"])
    assert artifact["result_path"] == mod.RESULT_RELATIVE_PATH
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4836_blocks_failed_gates_without_false_readiness(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4836-BLOCKED-PRECONDITION: failed gates do not claim readiness."""

    config = _config_resolution(a1_enabled=False)
    a1 = mod.resolve_a1_prior_inclusion(tmp_path, config)
    too_large_vram = mod.estimate_vram(
        model_size_bytes=MODEL_BYTES * 2,
        mtp_enabled=True,
        kv_quant="q8_0",
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=config,
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=too_large_vram,
        a1_prior_inclusion=a1,
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "complete_package_not_ready_vram"
    assert artifact["submission_package_ready"] is False
    assert artifact["vram_breakdown"]["fits_16gb"] is False
    assert "blocked until this JSON reports success_" in artifact["operator_checklist"][0]
    assert mod.artifact_schema_errors(artifact) == []

    missing_paths = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=config,
        model_path_resolution=_model_resolution(tmp_path, ok=False),
        vram_breakdown=_ready_vram(),
        a1_prior_inclusion=a1,
        duration_s=0.0,
    )
    assert missing_paths["honest_verdict"] == "complete_package_not_ready_model_paths"

    bad_config = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(ok=False),
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=_ready_vram(),
        a1_prior_inclusion=a1,
        duration_s=0.0,
    )
    assert bad_config["honest_verdict"] == "complete_package_not_ready_agent_config"

    bad_package = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(False),
        agent_config_resolution=config,
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=_ready_vram(),
        a1_prior_inclusion=a1,
        duration_s=0.0,
    )
    assert bad_package["honest_verdict"] == "complete_package_not_ready_dry_build"

    bad_precondition = mod.build_artifact(
        preconditions_checked=_preconditions(False),
        package_builds=_package_builds(True),
        agent_config_resolution=config,
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=_ready_vram(),
        a1_prior_inclusion=a1,
        duration_s=0.0,
    )
    assert bad_precondition["honest_verdict"] == (
        "complete_package_not_ready_arc_competition_agent"
    )


def test_req_capstone_4836_schema_rejects_false_ready_submission_and_checksum(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4836: schema guards terminal verdict, readiness, and operator-only fields."""

    _write_a1_prior(tmp_path, passed=False)
    config = _config_resolution(a1_enabled=False)
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=config,
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=_ready_vram(),
        a1_prior_inclusion=mod.resolve_a1_prior_inclusion(tmp_path, config),
        duration_s=0.0,
    )

    for field, bad_value, expected in (
        ("honest_verdict", "ready", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("field_principles", {}, "field_principles"),
        ("submitted_to_leaderboard", True, "submitted_to_leaderboard"),
        ("operator_only", False, "operator_only"),
        ("operator_checklist", ["submit now"], "operator_checklist"),
        ("result_path", "wrong.json", "result_path"),
        ("a1_prior_inclusion", [], "a1_prior_inclusion"),
    ):
        malformed = dict(artifact, **{field: bad_value})
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert expected in mod.artifact_schema_errors(malformed)

    false_ready = dict(artifact, vram_breakdown={**artifact["vram_breakdown"], "fits_16gb": False})
    false_ready["reproducibility_checksum"] = mod.payload_checksum(false_ready)
    assert "submission_package_ready_gate" in mod.artifact_schema_errors(false_ready)

    nested_submit = dict(
        artifact,
        package_builds={**artifact["package_builds"], "submitted_to_leaderboard": True},
    )
    nested_submit["reproducibility_checksum"] = mod.payload_checksum(nested_submit)
    assert "package_builds_submitted_to_leaderboard" in mod.artifact_schema_errors(nested_submit)

    mismatched_vram = dict(artifact, vram_estimate_gb=0.0)
    mismatched_vram["reproducibility_checksum"] = mod.payload_checksum(mismatched_vram)
    assert "vram_estimate_gb" in mod.artifact_schema_errors(mismatched_vram)

    bad_checksum = dict(artifact, reproducibility_checksum="sha256:bad")
    assert "reproducibility_checksum" in mod.artifact_schema_errors(bad_checksum)

    missing = dict(artifact)
    del missing["vram_estimate_gb"]
    assert "vram_estimate_gb" in mod.artifact_schema_errors(missing)

    bad_a1 = dict(
        artifact,
        a1_prior_inclusion={**artifact["a1_prior_inclusion"], "passed": True, "included": False},
    )
    bad_a1["reproducibility_checksum"] = mod.payload_checksum(bad_a1)
    assert "submission_package_ready_gate" in mod.artifact_schema_errors(bad_a1)


def test_scenario_capstone_4836_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4836: runner writes the operator-ready JSON without submitting."""

    ticks = iter((100.0, 100.25))
    config = _config_resolution(a1_enabled=False)
    artifact = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        package_builder=lambda _root: _package_builds(True),
        agent_config_resolver=lambda: config,
        model_path_resolver=lambda: _model_resolution(tmp_path, ok=True),
        vram_estimator=lambda _paths, _config: _ready_vram(),
        a1_prior_resolver=lambda root, agent_config: mod.resolve_a1_prior_inclusion(
            root,
            agent_config,
        ),
        write=True,
        now=lambda: next(ticks),
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["duration_s"] == 0.25
    assert artifact["submission_package_ready"] is True
    assert artifact["submitted_to_leaderboard"] is False
