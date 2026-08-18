"""Tests for Exp 4744 submission-package readiness validation.

Spec refs: REQ-CAPSTONE-4744, SCENARIO-CAPSTONE-4744,
SCENARIO-CAPSTONE-4744-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4744-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4744_submission_package_readiness as mod
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MODEL_ID as _LIVE_MODEL_ID,
)
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT as _SCORED_MTP,
)
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_REPO_SUBSTR as _LIVE_REPO_SUBSTR,
)

# DERIVED FROM THE LIVE CONSTANTS, NOT LITERALS (2026-08-18). This fixture stands in for the
# frozen generator so the readiness gate's LOGIC can be exercised. The gate compares the config
# against these same constants, so literals naming a previous model make a CORRECT gate report
# `frozen_generator_confirmed: False` -- which is what happened when the generator moved to
# Qwen3.8-27B and these five tests went red without anyone noticing. The scored-MTP CONTRACT is
# pinned as a literal in `tests/python/test_arc_submitted_agent_parity.py`, deliberately and in
# exactly one place; duplicating it here as a second literal is what let two tests disagree about
# one constant.
_SCORED_MTP_ON = _SCORED_MTP != "0"


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
SUBMITTED_AGENT_CONFIG_FIXTURE: JsonDict = {
    "policy": "E3AgentPolicy",
    "cascade": True,
    "frozen_generator": {
        "model_id": _LIVE_MODEL_ID,
        "repo_substr": _LIVE_REPO_SUBSTR,
        "model_filename": f"{_LIVE_REPO_SUBSTR}-Q4_K_M.gguf",
        "model_path_env": "CARNOT_ARC_GGUF_PATH",
        "server_path_env": "CARNOT_LLAMA_SERVER",
        "llama_server_kind": "cuda-12.8-binary",
        "binary_not_wheel": True,
        "required_shared_libraries": [
            "libllama-common",
            "libllama",
            "libggml",
            "libggml-cuda",
        ],
        "mtp": _SCORED_MTP_ON,
        "spec_type": "draft-mtp" if _SCORED_MTP_ON else None,
        "kv_quant": "q8_0",
        "no_think_prefix": "",
        "max_tokens": 2560,
        "n_predict_min": 2048,
        "port_strategy": "free_non_8919",
        "props_verify_endpoint": "/props",
        "wheel_fallback_allowed": False,
        "forbidden_models": ["gemma-8919"],
        "forbidden_gpu_targets": ["3090"],
        "gpu_target": "kaggle_cuda_gpu_not_3090",
    },
}


def _preconditions(ok: bool = True) -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "spec_has_req_4744": True,
        "arc_competition_agent_importable": ok,
        "offline_arcade_importable": ok,
        "ok": ok,
        **({} if ok else {"blocked_resource": "offline_arcade"}),
    }


def _entrypoint(ok: bool = True) -> JsonDict:
    return {
        "imported": ok,
        "constructed": ok,
        "entrypoint": "carnot.agentic.arc_competition_agent.make_carnot_agent",
        "policy_class": "E3AgentPolicy" if ok else "",
        "max_actions": 400 if ok else 0,
        "blocked_resource": "" if ok else "entrypoint",
    }


def _smoke(ok: bool = True) -> JsonDict:
    return {
        "ran": ok,
        "game": "ar25" if ok else "",
        "actions_taken": 1 if ok else 0,
        "action_budget": 5,
        "rpm_cap": 600,
        "within_action_budget": ok,
        "within_rpm_budget": ok,
        "solve_claim_made": False,
        "blocked_resource": "" if ok else "offline_smoke",
    }


def _parity(ok: bool = True) -> JsonDict:
    return {
        "passed": ok,
        "command": ".venv/bin/pytest tests/python/test_arc_submitted_agent_parity.py -q",
        "returncode": 0 if ok else 1,
    }


def _manifest(tmp_path: Path, *, complete: bool = True) -> JsonDict:
    model = tmp_path / f"{_LIVE_REPO_SUBSTR}-Q4_K_M.gguf"
    server = tmp_path / "llama-server"
    if complete:
        shared = [
            tmp_path / "libllama-common.so",
            tmp_path / "libllama.so",
            tmp_path / "libggml.so",
            tmp_path / "libggml-cuda.so",
        ]
        for path in (model, server, *shared):
            path.write_text("fixture\n", encoding="utf-8")
    return mod.build_package_manifest(
        env={
            "CARNOT_ARC_GGUF_PATH": str(model),
            "CARNOT_LLAMA_SERVER": str(server),
        },
        shared_library_search_dirs=(tmp_path,),
    )


def test_req_capstone_4744_spec_declares_submission_package_contract() -> None:
    """REQ-CAPSTONE-4744: OpenSpec declares the package-readiness contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4744",
        "SCENARIO-CAPSTONE-4744",
        "SCENARIO-CAPSTONE-4744-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4744-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4744_confirms_frozen_generator_from_submitted_config() -> None:
    """SCENARIO-CAPSTONE-4744: frozen stack is wired into SUBMITTED_AGENT_CONFIG."""

    config = mod.frozen_generator_config_from_submitted(SUBMITTED_AGENT_CONFIG_FIXTURE)

    assert config["confirmed"] is True
    assert config["model_id"] == _LIVE_MODEL_ID
    assert config["repo_substr"] == _LIVE_REPO_SUBSTR
    assert config["mtp"] is _SCORED_MTP_ON
    assert config["kv_quant"] == "q8_0"
    assert config["max_tokens"] >= 2048
    assert config["no_think_prefix"] == ""
    assert config["llama_server_kind"] == "cuda-12.8-binary"
    assert config["wheel_fallback_allowed"] is False
    assert config["port_strategy"] == "free_non_8919"
    assert config["props_verify_endpoint"] == "/props"
    assert config["forbidden_models"] == ["gemma-8919"]
    assert config["forbidden_gpu_targets"] == ["3090"]


def test_scenario_capstone_4744_ready_artifact_has_pass_checklist(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4744: all gates passing marks operator-ready, not submitted."""

    manifest = _manifest(tmp_path, complete=True)
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        entrypoint_validation=_entrypoint(True),
        frozen_generator_config=mod.frozen_generator_config_from_submitted(
            SUBMITTED_AGENT_CONFIG_FIXTURE
        ),
        package_manifest=manifest,
        smoke_episode=_smoke(True),
        parity_test=_parity(True),
        duration_s=1.0,
        submitted_agent_config=SUBMITTED_AGENT_CONFIG_FIXTURE,
    )

    assert artifact["honest_verdict"] == "success: submission_package_ready"
    assert artifact["submission_package_ready"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["frozen_generator_confirmed"] is True
    assert artifact["parity_test_green"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["smoke_episode"]["solve_claim_made"] is False
    assert artifact["package_manifest"]["complete"] is True
    assert [item["status"] for item in artifact["readiness_checklist"]] == ["pass"] * 5
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4744_blocks_missing_manifest_resources(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4744-BLOCKED-PRECONDITION: missing resources do not fabricate readiness."""

    manifest = _manifest(tmp_path, complete=False)
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        entrypoint_validation=_entrypoint(True),
        frozen_generator_config=mod.frozen_generator_config_from_submitted(
            SUBMITTED_AGENT_CONFIG_FIXTURE
        ),
        package_manifest=manifest,
        smoke_episode=_smoke(True),
        parity_test=_parity(True),
        duration_s=1.0,
        submitted_agent_config=SUBMITTED_AGENT_CONFIG_FIXTURE,
    )
    checklist = {item["id"]: item for item in artifact["readiness_checklist"]}

    assert artifact["honest_verdict"] == "complete: submission_package_blocked_manifest_resources"
    assert artifact["submission_package_ready"] is False
    assert artifact["frozen_generator_confirmed"] is True
    assert artifact["parity_test_green"] is True
    assert checklist["manifest_complete"]["status"] == "blocked"
    assert "model_file" in manifest["blocked_resources"]
    assert "llama_server_binary" in manifest["blocked_resources"]
    assert "libllama-common" in manifest["blocked_resources"]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4744_runner_writes_checksum_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4744: runner writes the readiness artifact."""

    manifest = _manifest(tmp_path, complete=True)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(True),
        entrypoint_validator=lambda: _entrypoint(True),
        frozen_generator_loader=lambda: (
            mod.frozen_generator_config_from_submitted(SUBMITTED_AGENT_CONFIG_FIXTURE),
            SUBMITTED_AGENT_CONFIG_FIXTURE,
        ),
        manifest_builder=lambda: manifest,
        smoke_runner=lambda: _smoke(True),
        parity_runner=lambda _root: _parity(True),
        write=True,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)
    assert written["submission_package_ready"] is True
    assert mod.artifact_schema_errors(written) == []


def test_req_capstone_4744_schema_rejects_false_ready_and_checksum_drift(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4744: schema rejects false readiness and checksum drift."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        entrypoint_validation=_entrypoint(True),
        frozen_generator_config=mod.frozen_generator_config_from_submitted(
            SUBMITTED_AGENT_CONFIG_FIXTURE
        ),
        package_manifest=_manifest(tmp_path, complete=False),
        smoke_episode=_smoke(True),
        parity_test=_parity(True),
        duration_s=1.0,
        submitted_agent_config=SUBMITTED_AGENT_CONFIG_FIXTURE,
    )

    false_ready = dict(artifact, submission_package_ready=True)
    false_ready["reproducibility_checksum"] = mod.payload_checksum(false_ready)
    assert "submission_package_ready_gate" in mod.artifact_schema_errors(false_ready)

    drifted = dict(artifact, duration_s=99.0)
    assert "reproducibility_checksum" in mod.artifact_schema_errors(drifted)

    missing_principles = dict(artifact, field_principles={})
    missing_principles["reproducibility_checksum"] = mod.payload_checksum(missing_principles)
    assert "field_principles" in mod.artifact_schema_errors(missing_principles)


def test_req_capstone_4744_blocked_resource_selection_covers_all_gates() -> None:
    """REQ-CAPSTONE-4744: blocked verdicts name the first failed gate."""

    base = [
        {"id": "entrypoint_imports", "status": "pass"},
        {"id": "frozen_generator_wired", "status": "pass"},
        {"id": "manifest_complete", "status": "pass"},
        {"id": "smoke_episode_ran", "status": "pass"},
        {"id": "parity_green", "status": "pass"},
    ]

    assert mod._blocked_resource({"ok": False, "blocked_resource": "offline_arcade"}, []) == (
        "offline_arcade"
    )
    for index, expected in (
        (0, "entrypoint"),
        (1, "frozen_generator"),
        (2, "manifest_resources"),
        (3, "offline_smoke"),
        (4, "parity_test"),
    ):
        rows = [dict(item) for item in base]
        rows[index]["status"] = "blocked"
        assert mod._blocked_resource({"ok": True}, rows) == expected
    assert mod._blocked_resource({"ok": True}, base) == "unknown"


def test_req_capstone_4744_schema_rejects_required_field_and_boolean_drift(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4744: schema catches malformed package-readiness evidence."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        entrypoint_validation=_entrypoint(True),
        frozen_generator_config=mod.frozen_generator_config_from_submitted(
            SUBMITTED_AGENT_CONFIG_FIXTURE
        ),
        package_manifest=_manifest(tmp_path, complete=True),
        smoke_episode=_smoke(True),
        parity_test=_parity(True),
        duration_s=1.0,
        submitted_agent_config=SUBMITTED_AGENT_CONFIG_FIXTURE,
    )

    missing = dict(artifact)
    del missing["package_manifest"]
    assert "package_manifest" in mod.artifact_schema_errors(missing)

    for field, bad_value, expected in (
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("submitted_to_leaderboard", True, "submitted_to_leaderboard"),
        ("operator_only", False, "operator_only"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ):
        malformed = dict(artifact, **{field: bad_value})
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert expected in mod.artifact_schema_errors(malformed)

    frozen_bad = dict(artifact, frozen_generator_confirmed=False)
    frozen_bad["reproducibility_checksum"] = mod.payload_checksum(frozen_bad)
    assert "frozen_generator_confirmed" in mod.artifact_schema_errors(frozen_bad)

    parity_bad = dict(artifact, parity_test_green=False)
    parity_bad["reproducibility_checksum"] = mod.payload_checksum(parity_bad)
    assert "parity_test_green" in mod.artifact_schema_errors(parity_bad)

    smoke_bad = dict(artifact, smoke_episode=dict(artifact["smoke_episode"], solve_claim_made=True))
    smoke_bad["reproducibility_checksum"] = mod.payload_checksum(smoke_bad)
    assert "smoke_episode_solve_claim" in mod.artifact_schema_errors(smoke_bad)

    blocked = mod.build_artifact(
        preconditions_checked=_preconditions(True),
        entrypoint_validation=_entrypoint(True),
        frozen_generator_config=mod.frozen_generator_config_from_submitted(
            SUBMITTED_AGENT_CONFIG_FIXTURE
        ),
        package_manifest=_manifest(tmp_path / "missing", complete=False),
        smoke_episode=_smoke(True),
        parity_test=_parity(True),
        duration_s=1.0,
        submitted_agent_config=SUBMITTED_AGENT_CONFIG_FIXTURE,
    )
    bad_verdict = dict(blocked, honest_verdict="blocked_manifest_resources")
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    assert "honest_verdict" in mod.artifact_schema_errors(bad_verdict)


def test_scenario_capstone_4744_runner_writes_blocked_precondition_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4744-BLOCKED-PRECONDITION: runner exits honestly on missing arcade."""

    manifest = _manifest(tmp_path, complete=True)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(False),
        frozen_generator_loader=lambda: (
            mod.frozen_generator_config_from_submitted(SUBMITTED_AGENT_CONFIG_FIXTURE),
            SUBMITTED_AGENT_CONFIG_FIXTURE,
        ),
        manifest_builder=lambda: manifest,
        write=True,
    )

    assert artifact["honest_verdict"] == "complete: submission_package_blocked_offline_arcade"
    assert artifact["submission_package_ready"] is False
    assert artifact["entrypoint_validation"]["blocked_resource"] == "offline_arcade"
    assert artifact["smoke_episode"]["blocked_resource"] == "offline_arcade"
    assert artifact["parity_test"]["blocked_resource"] == "offline_arcade"
    assert mod.artifact_schema_errors(artifact) == []
