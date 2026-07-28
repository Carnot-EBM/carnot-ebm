"""Tests for Exp 4846 ARC-AGI-3 submission-package hardening.

Spec refs: REQ-CAPSTONE-4846, SCENARIO-CAPSTONE-4846,
SCENARIO-CAPSTONE-4846-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4846-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4846_submission_package_harden as mod

# The canonical generator pin, imported rather than re-typed. These names were introduced into
# this file's assertions by the 2026-07-28 gemma migration but never imported, so every test
# referencing them died with NameError -- a failure that looks like a broken gate rather than a
# broken test.
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MODEL_FILENAME,
    ARC_LIVE_GENERATOR_MODEL_ID,
    ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
    ARC_LIVE_GENERATOR_REPO_SUBSTR,
)


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


def _requirements_doc_text() -> str:
    return """
You submit an AGENT, not a scorecard or a file of answers. No internet access during evaluation.
The GGUF is gemma-4-31B-it-Q4_K_M.gguf and the env path is CARNOT_ARC_GGUF_PATH.
Bundle a CUDA llama.cpp BINARY (llama-server), NOT a Python wheel, and set CARNOT_LLAMA_SERVER.
The validated deploy config uses draft-mtp plus q8_0 KV; CARNOT_ARC_MTP=0 may disable MTP on tight VRAM.
The public entry is an operator-gated Kaggle submission, and this task never submits.
The final Kaggle probe is covered by scripts/kaggle/build_verify_llamacpp_mtp.py.
"""


def _write_submission_kernel(root: Path) -> None:
    kernel = root / "scripts" / "kaggle" / "submission_kernel"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "code_file": "main.py",
                "kernel_type": "script",
                "enable_gpu": True,
                "enable_internet": False,
                "dataset_sources": [
                    "iancblenke/carnot-agent-code",
                    "iancblenke/carnot-llamacpp-mtp-binary",
                    # The gemma main-weights dataset. The Qwen one this replaced is retired
                    # (2026-07-28 operator directive); `REQUIRED_DATASETS` was migrated with
                    # the module and this fixture was not, so the gate under test was being
                    # handed a manifest the real kernel-metadata.json no longer writes.
                    "iancblenke/carnot-gemma4-31b-it-gguf",
                    # The MTP draft head. Not in REQUIRED_DATASETS -- a missing head is a
                    # degraded-but-valid scored run (no speculative decoding) rather than a
                    # blocker -- but the real manifest attaches it, so the fixture should
                    # look like the real manifest.
                    "iancblenke/carnot-gemma4-31b-mtp-head",
                ],
                "competition_sources": ["arc-prize-2026-arc-agi-3"],
            }
        ),
        encoding="utf-8",
    )
    (kernel / "main.py").write_text(
        "\n".join(
            [
                "import os",
                "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):",
                "    print('CARNOT_ARC_GGUF_PATH CARNOT_LLAMA_SERVER llama-server')",
                "    print('Qwen3.5-9B-Q4_K_M.gguf draft-mtp q8_0')",
                "    os.environ['CARNOT_ARC_MTP'] = '0'",
                "else:",
                "    print('/kaggle/working/submission.parquet')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    build_probe = root / "scripts" / "kaggle" / "build_verify_llamacpp_mtp.py"
    build_probe.write_text("print('probe')\n", encoding="utf-8")


def _write_repo_preconditions(
    root: Path,
    *,
    with_spec: bool = True,
    with_doc: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = root / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-CAPSTONE-4846\n" if with_spec else "REQ-CAPSTONE-OTHER\n",
        encoding="utf-8",
    )
    if with_doc:
        doc = root / mod.REQUIREMENTS_RELATIVE_PATH
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(_requirements_doc_text(), encoding="utf-8")
    agent = root / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
    agent.parent.mkdir(parents=True, exist_ok=True)
    agent.write_text("SUBMITTED_AGENT_CONFIG = {}\n", encoding="utf-8")
    _write_submission_kernel(root)


def _preconditions(ok: bool = True) -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4846": True,
        "packaging_requirements_doc_present": True,
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


def _a1_prior_inclusion(*, passed: bool = False, included: bool = False) -> JsonDict:
    return {
        "source_artifact_path": mod.A1_PRIOR_RELATIVE_PATH,
        "artifact_present": True,
        "passed": passed,
        "included": included,
        "reason": (
            "passed_and_included_in_frozen_config"
            if included
            else "not_included_a1_prior_did_not_pass"
        ),
    }


def _requirements_ok() -> JsonDict:
    return {
        "requirements_doc_path": mod.REQUIREMENTS_RELATIVE_PATH,
        "doc_present": True,
        "ok": True,
        "blocked_resource": "",
        "checks": {key: True for key in mod.REQUIREMENTS_CHECK_KEYS},
        "notes": ["requirements package cross-check passed"],
    }


def test_req_capstone_4846_spec_declares_operator_package_contract() -> None:
    """REQ-CAPSTONE-4846: OpenSpec declares the package-hardening contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4846",
        "SCENARIO-CAPSTONE-4846",
        "SCENARIO-CAPSTONE-4846-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4846-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        "packaging_requirements_crosscheck",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4846_preconditions_dry_build_and_doc_crosscheck(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4846: package scripts, agent, and packaging spec cross-check."""

    _write_repo_preconditions(tmp_path)
    config = _config_resolution()
    paths = _model_resolution(tmp_path, ok=True)

    preconditions = mod.check_preconditions(tmp_path)
    package_builds = mod.dry_build_package(tmp_path)
    crosscheck = mod.cross_check_packaging_requirements(
        tmp_path,
        package_builds=package_builds,
        agent_config_resolution=config,
        model_path_resolution=paths,
    )

    assert preconditions["ok"] is True
    assert preconditions["spec_has_req_4846"] is True
    assert package_builds["package_builds"] is True
    assert package_builds["submitted_to_leaderboard"] is False
    assert crosscheck["ok"] is True
    assert all(crosscheck["checks"].values())
    assert "operator-only" in " ".join(crosscheck["notes"])

    _write_repo_preconditions(tmp_path, with_spec=False)
    blocked = mod.check_preconditions(tmp_path)
    assert blocked["ok"] is False
    assert blocked["blocked_resource"] == "spec_has_req_4846"

    missing_doc = tmp_path / mod.REQUIREMENTS_RELATIVE_PATH
    missing_doc.unlink()
    crosscheck = mod.cross_check_packaging_requirements(
        tmp_path,
        package_builds=package_builds,
        agent_config_resolution=config,
        model_path_resolution=paths,
    )
    assert crosscheck["ok"] is False
    assert crosscheck["blocked_resource"] == "packaging_requirements"

    _write_repo_preconditions(tmp_path)
    (tmp_path / "scripts" / "kaggle" / "submission_kernel" / "kernel-metadata.json").unlink()
    crosscheck = mod.cross_check_packaging_requirements(
        tmp_path,
        package_builds=package_builds,
        agent_config_resolution=config,
        model_path_resolution=paths,
    )
    assert crosscheck["doc_present"] is True
    assert crosscheck["checks"]["internet_disabled"] is False


def test_scenario_capstone_4846_vram_estimate_uses_selected_qwen_size() -> None:
    """SCENARIO-CAPSTONE-4846: Qwen3.5 Q4 + q8 KV is estimated under 16GB."""

    missing_paths = {
        "resolved": False,
        "gguf": {"size_bytes": 0},
        "llama_server": {"present": True, "cuda_12_8_capable": True},
    }
    vram = mod.runtime_vram_estimate(missing_paths, _config_resolution())

    assert vram["model_size_source"] == "packaging_spec_default"
    assert vram["selected_model_size_bytes"] == MODEL_BYTES
    assert vram["fits_16gb"] is True
    assert vram["model_copies"] == 2
    assert 14.0 < vram["vram_estimate_gb"] < 16.0
    assert vram["remaining_headroom_gb"] > 0.0


def test_scenario_capstone_4846_ready_artifact_is_operator_only(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4846: green package gate marks operator-ready only."""

    config = _config_resolution(a1_enabled=False)
    vram = _ready_vram()
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=config,
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=vram,
        a1_prior_inclusion=_a1_prior_inclusion(),
        packaging_requirements_crosscheck=_requirements_ok(),
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "success_package_builds_vram_gate_green"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["submission_package_ready"] is True
    assert artifact["vram_estimate_gb"] == vram["vram_estimate_gb"]
    assert artifact["vram_estimate_gb"] < 16.0
    assert artifact["packaging_requirements_crosscheck"]["ok"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert all(step.startswith("OPERATOR-CHECK:") for step in artifact["operator_checklist"])
    assert any("this task never submits" in step.lower() for step in artifact["operator_checklist"])
    assert artifact["result_path"] == mod.RESULT_RELATIVE_PATH
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4846_blocks_failed_gates_without_false_readiness(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4846-BLOCKED-PRECONDITION: failed gates do not claim readiness."""

    config = _config_resolution(a1_enabled=False)
    a1 = _a1_prior_inclusion()
    common = {
        "preconditions_checked": _preconditions(True),
        "package_builds": _package_builds(True),
        "agent_config_resolution": config,
        "model_path_resolution": _model_resolution(tmp_path, ok=True),
        "vram_breakdown": _ready_vram(),
        "a1_prior_inclusion": a1,
        "packaging_requirements_crosscheck": _requirements_ok(),
        "duration_s": 0.0,
    }

    too_large_vram = mod.estimate_vram(
        model_size_bytes=MODEL_BYTES * 2,
        mtp_enabled=True,
        kv_quant="q8_0",
    )
    cases = [
        (
            {"preconditions_checked": _preconditions(False)},
            "complete_package_not_ready_arc_competition_agent",
        ),
        (
            {"package_builds": _package_builds(False)},
            "complete_package_not_ready_dry_build",
        ),
        (
            {"agent_config_resolution": _config_resolution(ok=False)},
            "complete_package_not_ready_agent_config",
        ),
        (
            {"model_path_resolution": _model_resolution(tmp_path, ok=False)},
            "complete_package_not_ready_model_paths",
        ),
        (
            {"vram_breakdown": too_large_vram},
            "complete_package_not_ready_vram",
        ),
        (
            {"packaging_requirements_crosscheck": {**_requirements_ok(), "ok": False}},
            "complete_package_not_ready_packaging_requirements",
        ),
        (
            {"a1_prior_inclusion": _a1_prior_inclusion(passed=True, included=False)},
            "complete_package_not_ready_a1_prior_inclusion",
        ),
    ]

    for overrides, verdict in cases:
        artifact = mod.build_artifact(**{**common, **overrides})
        assert artifact["honest_verdict"] == verdict
        assert artifact["submission_package_ready"] is False
        assert "blocked until this JSON reports success_" in artifact["operator_checklist"][0]
        assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4846_schema_rejects_false_ready_submission_and_checksum(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4846: schema guards verdict, readiness, and operator-only fields."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_builds=_package_builds(True),
        agent_config_resolution=_config_resolution(),
        model_path_resolution=_model_resolution(tmp_path, ok=True),
        vram_breakdown=_ready_vram(),
        a1_prior_inclusion=_a1_prior_inclusion(),
        packaging_requirements_crosscheck=_requirements_ok(),
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
        ("packaging_requirements_crosscheck", [], "packaging_requirements_crosscheck"),
    ):
        malformed = dict(artifact, **{field: bad_value})
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert expected in mod.artifact_schema_errors(malformed)

    false_ready = dict(
        artifact,
        packaging_requirements_crosscheck={
            **artifact["packaging_requirements_crosscheck"],
            "ok": False,
        },
    )
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


def test_scenario_capstone_4846_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4846: runner writes the operator-ready JSON without submitting."""

    ticks = iter((100.0, 100.25))
    config = _config_resolution(a1_enabled=False)
    artifact = mod.run(
        tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        package_builder=lambda _root: _package_builds(True),
        agent_config_resolver=lambda: config,
        model_path_resolver=lambda: _model_resolution(tmp_path, ok=True),
        vram_estimator=lambda _paths, _config: _ready_vram(),
        a1_prior_resolver=lambda _root, _agent_config: _a1_prior_inclusion(),
        requirements_crosschecker=lambda _root, **_kwargs: _requirements_ok(),
        write=True,
        now=lambda: next(ticks),
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["duration_s"] == 0.25
    assert artifact["submission_package_ready"] is True
    assert artifact["submitted_to_leaderboard"] is False
