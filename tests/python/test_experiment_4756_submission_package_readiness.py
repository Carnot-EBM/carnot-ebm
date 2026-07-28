"""Tests for Exp 4756 Kaggle submission-package readiness.

Spec refs: REQ-CAPSTONE-4756, SCENARIO-CAPSTONE-4756,
SCENARIO-CAPSTONE-4756-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4756-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4756_submission_package_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_kernel(root: Path) -> None:
    kernel = root / "scripts" / "kaggle" / "submission_kernel"
    kernel.mkdir(parents=True)
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "iancblenke/carnot-arc-agi3-submission",
                "title": "carnot-arc-agi3-submission",
                "code_file": "main.py",
                "language": "python",
                "kernel_type": "script",
                "enable_gpu": True,
                "machine_shape": "NvidiaL4",
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
        )
        + "\n",
        encoding="utf-8",
    )
    (kernel / "main.py").write_text(
        "\n".join(
            [
                "import os, subprocess, sys",
                "COMP = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3'",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',",
                "    '--find-links', f'{COMP}/arc_agi_3_wheels', 'arc-agi', 'python-dotenv', '--quiet'])",
                "AGENT_SRC = 'Qwen3.5-9B CARNOT_LLAMA_SERVER CARNOT_ARC_GGUF_PATH llama-server Qwen3.5-9B-Q4_K_M.gguf'",
                "print('/kaggle/input ARC-AGI-3-Agents my_agent.py KAGGLE_IS_COMPETITION_RERUN gateway:8001 submission.parquet pandas to_parquet')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _preconditions(ok: bool = True) -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4756": True,
        "submission_kernel_present": True,
        "qwen35_mtp_gguf_cached": ok,
        "qwen35_mtp_gguf_paths": ["/models/Qwen3.5-9B-Q4_K_M.gguf"] if ok else [],
        "llama_server_binary_present": ok,
        "llama_server_paths": ["/bin/llama-server"] if ok else [],
        "ok": ok,
        **({} if ok else {"blocked_resource": "qwen35_mtp_gguf_cached"}),
    }


def _package_builds(ok: bool = True) -> JsonDict:
    return {
        "assembled": ok,
        "entrypoint_compiles": ok,
        "manifest_complete": ok,
        "requirements_complete": ok,
        "clean_env_smoke_ran": ok,
        "clean_env": True,
        "submitted_to_leaderboard": False,
        "blocked_resource": "" if ok else "clean_env_smoke",
        "assembly": {"files": ["kernel-metadata.json", "main.py"], "sha256": "sha256:abc"},
        "entrypoint_smoke": {"passed": ok, "policy_class": "E3AgentPolicy"},
    }


def _smoke_failed_builds() -> JsonDict:
    build = _package_builds(True)
    build["clean_env_smoke_ran"] = False
    build["blocked_resource"] = "clean_env_smoke"
    build["entrypoint_smoke"] = {"passed": False, "policy_class": ""}
    return build


def test_req_capstone_4756_spec_declares_package_readiness_contract() -> None:
    """REQ-CAPSTONE-4756: OpenSpec declares the deadline package-readiness contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4756",
        "SCENARIO-CAPSTONE-4756",
        "SCENARIO-CAPSTONE-4756-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4756-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4756_manifest_and_requirements_are_checked(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4756: manifest and offline requirements are complete."""

    _write_kernel(tmp_path)

    manifest = mod.inspect_package_manifest(tmp_path)
    requirements = mod.inspect_requirements(tmp_path)

    assert manifest["complete"] is True
    assert manifest["checks"]["internet_off"] is True
    assert manifest["checks"]["gpu_enabled"] is True
    assert manifest["checks"]["datasets_attached"] is True
    assert manifest["checks"]["competition_attached"] is True
    assert requirements["complete"] is True
    assert requirements["checks"]["offline_pip_no_index"] is True
    assert requirements["checks"]["llama_server_env"] is True
    assert requirements["checks"]["gguf_env"] is True
    assert requirements["checks"]["placeholder_parquet"] is True

    bad_metadata = json.loads(
        (tmp_path / "scripts/kaggle/submission_kernel/kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    bad_metadata["enable_internet"] = True
    (tmp_path / "scripts/kaggle/submission_kernel/kernel-metadata.json").write_text(
        json.dumps(bad_metadata), encoding="utf-8"
    )
    assert mod.inspect_package_manifest(tmp_path)["complete"] is False


def test_scenario_capstone_4756_builds_ready_artifact_with_operator_checklist() -> None:
    """SCENARIO-CAPSTONE-4756: green package gate marks operator-ready without submission."""

    manifest = {"complete": True, "checks": {"internet_off": True}}
    requirements = {"complete": True, "checks": {"offline_pip_no_index": True}}
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_manifest=manifest,
        requirements_check=requirements,
        package_builds=_package_builds(True),
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "success_package_ready_offline_smoke_green"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts; 100us floor."
    assert artifact["submission_package_ready"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["package_builds"]["clean_env_smoke_ran"] is True
    assert artifact["package_manifest"] == manifest
    assert artifact["requirements_check"] == requirements
    assert all(step.startswith("OPERATOR-ACTION:") for step in artifact["operator_checklist"])
    assert any("submit" in step.lower() for step in artifact["operator_checklist"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4756_blocks_missing_resources_without_false_ready() -> None:
    """SCENARIO-CAPSTONE-4756-BLOCKED-PRECONDITION: missing GGUF blocks readiness."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(False),
        package_manifest={"complete": True},
        requirements_check={"complete": True},
        package_builds=_package_builds(True),
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_qwen35_mtp_gguf_cached"
    assert artifact["submission_package_ready"] is False
    assert artifact["package_builds"]["assembled"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4756_schema_rejects_false_ready_submission_and_checksum() -> None:
    """REQ-CAPSTONE-4756: schema guards readiness, operator-only, and checksums."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_manifest={"complete": True},
        requirements_check={"complete": True},
        package_builds=_smoke_failed_builds(),
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "complete_package_not_ready_clean_env_smoke"
    assert artifact["submission_package_ready"] is False

    false_ready = dict(artifact, submission_package_ready=True)
    false_ready["reproducibility_checksum"] = mod.payload_checksum(false_ready)
    assert "submission_package_ready_gate" in mod.artifact_schema_errors(false_ready)

    submitted = dict(artifact, submitted_to_leaderboard=True)
    submitted["reproducibility_checksum"] = mod.payload_checksum(submitted)
    assert "submitted_to_leaderboard" in mod.artifact_schema_errors(submitted)

    bad_checklist = dict(artifact, operator_checklist=["submit now"])
    bad_checklist["reproducibility_checksum"] = mod.payload_checksum(bad_checklist)
    assert "operator_checklist" in mod.artifact_schema_errors(bad_checklist)

    drifted = dict(artifact, duration_s=99.0)
    assert "reproducibility_checksum" in mod.artifact_schema_errors(drifted)


def test_scenario_capstone_4756_runner_writes_stable_result(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4756: runner writes the requested result JSON."""

    _write_kernel(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        manifest_inspector=lambda _root: mod.inspect_package_manifest(_root),
        requirements_inspector=lambda _root: mod.inspect_requirements(_root),
        package_validator=lambda _root, _pre, manifest, requirements: {
            **_package_builds(True),
            "manifest_complete": manifest["complete"],
            "requirements_complete": requirements["complete"],
        },
        write=True,
        now=lambda: 10.0,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert written["submission_package_ready"] is True
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)
    assert mod.artifact_schema_errors(written) == []


def test_req_capstone_4756_precondition_selection_and_build_gate() -> None:
    """REQ-CAPSTONE-4756: blocked resource selection covers preconditions and build gates."""

    assert mod._blocked_resource(_preconditions(False), _package_builds(True)) == (
        "qwen35_mtp_gguf_cached"
    )
    for field, expected in (
        ("assembled", "package_assembly"),
        ("entrypoint_compiles", "entrypoint_compile"),
        ("manifest_complete", "manifest"),
        ("requirements_complete", "requirements"),
        ("clean_env_smoke_ran", "clean_env_smoke"),
    ):
        build = _package_builds(True)
        build[field] = False
        assert mod._blocked_resource(_preconditions(True), build) == expected
    assert mod._blocked_resource(_preconditions(True), _package_builds(True)) == "unknown"


def test_req_capstone_4756_preconditions_hash_and_malformed_manifest(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4756: precondition and manifest helpers fail closed."""

    _write_kernel(tmp_path)
    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-CAPSTONE-4756\n", encoding="utf-8")
    hashed = tmp_path / "Qwen3.5-9B-Q4_K_M.gguf"
    hashed.write_text("weights\n", encoding="utf-8")

    checks = mod.check_preconditions(
        tmp_path,
        gguf_finder=lambda: [str(hashed)],
        llama_server_finder=lambda: ["/bin/llama-server"],
    )

    assert checks["ok"] is True
    assert checks["qwen35_mtp_gguf_paths"] == [str(hashed)]
    assert mod._file_sha256(hashed).startswith("sha256:")

    blocked = mod.check_preconditions(
        tmp_path,
        gguf_finder=lambda: [],
        llama_server_finder=lambda: ["/bin/llama-server"],
    )
    assert blocked["ok"] is False
    assert blocked["blocked_resource"] == "qwen35_mtp_gguf_cached"

    metadata = tmp_path / "scripts" / "kaggle" / "submission_kernel" / "kernel-metadata.json"
    metadata.write_text("{not json", encoding="utf-8")
    malformed = mod.inspect_package_manifest(tmp_path)
    assert malformed["complete"] is False
    assert "code_file_main" in malformed["blocked_resources"]


def test_req_capstone_4756_schema_reports_remaining_guard_branches() -> None:
    """REQ-CAPSTONE-4756: schema reports malformed terminal and operator-only fields."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        package_manifest={"complete": True},
        requirements_check={"complete": True},
        package_builds=_package_builds(True),
        duration_s=0.1,
    )

    missing = dict(artifact)
    del missing["package_manifest"]
    assert "package_manifest" in mod.artifact_schema_errors(missing)

    for field, bad_value, expected in (
        ("honest_verdict", "ready", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("field_principles", {}, "field_principles"),
        ("operator_only", False, "operator_only"),
    ):
        malformed = dict(artifact, **{field: bad_value})
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert expected in mod.artifact_schema_errors(malformed)

    build_submitted = dict(
        artifact,
        package_builds={**artifact["package_builds"], "submitted_to_leaderboard": True},
    )
    build_submitted["reproducibility_checksum"] = mod.payload_checksum(build_submitted)
    assert "package_builds_submitted_to_leaderboard" in mod.artifact_schema_errors(build_submitted)
