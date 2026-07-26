"""Tests for Exp5928 ARC live-runner execution binding.

Spec refs: REQ-ARC-LREB-5928,
SCENARIO-ARC-LREB-5928-PARENT-CHILD-CONSUME,
SCENARIO-ARC-LREB-5928-DENIAL-MATRIX,
SCENARIO-ARC-LREB-5928-TEARDOWN-IMMUTABILITY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from carnot.agentic import arc_live_runner_execution_binding as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/agentic-harness/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _context(tmp_path: Path) -> mod.ProcessBindingContext:
    run_id = "exp5928-unit-run"
    output = tmp_path / "child-receipt.json"
    argv = mod.child_argv(run_id=run_id, output_path=output, nonce_ledger=tmp_path / "nonce-ledger")
    env = mod.child_environment(run_id=run_id, root=REPO)
    return mod.ProcessBindingContext.from_values(
        pid=12345,
        ppid=54321,
        executable_path=Path(os.sys.executable),
        argv=argv,
        environment=env,
        output_path=output,
    )


def test_req_arc_lreb_5928_spec_declares_execution_binding_contract() -> None:
    """REQ-ARC-LREB-5928: OpenSpec freezes execution binding and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-LREB-5928") :]

    for marker in (
        "SCENARIO-ARC-LREB-5928-PARENT-CHILD-CONSUME",
        "SCENARIO-ARC-LREB-5928-DENIAL-MATRIX",
        "SCENARIO-ARC-LREB-5928-TEARDOWN-IMMUTABILITY",
        mod.RESULT_RELATIVE_PATH,
        "parent/controller-issued",
        "process-bound capability",
        "wrong-process",
        "wrong-command",
        "wrong-environment",
        "wrong-executable",
        "output-mismatched",
        "actual_live_runner_capability_preflight_no_llm",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for field, principle in mod.REQUIRED_FIELD_PROVENANCE.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_req_arc_lreb_5928_capability_schema_binds_process_command_env_and_output(
    tmp_path: Path,
) -> None:
    """REQ-ARC-LREB-5928: capability binds child PID, executable, argv, env, scope, and output."""

    issuer = mod.ParentIssuer.fixture()
    context = _context(tmp_path)
    ledger = mod.NonceLedger(tmp_path / "nonce-ledger")
    capability = mod.issue_capability(
        issuer,
        context,
        issued_monotonic_s=100.0,
        expires_monotonic_s=105.0,
        nonce="unit-nonce-0001",
        run_id="exp5928-unit-run",
    )

    result = mod.verify_and_consume_capability(
        capability,
        context,
        public_key=issuer.public_key_hex,
        now_monotonic_s=101.0,
        nonce_ledger=ledger,
        adapter_enabled=False,
    )

    assert result.allowed is True
    assert result.reason == "allowed"
    assert ledger.contains("unit-nonce-0001") is True
    assert capability["child_process_identity"]["pid"] == 12345
    assert capability["binding"]["argv_hash"] == context.argv_hash
    assert capability["binding"]["environment_allowlist_hash"] == context.environment_allowlist_hash
    assert capability["binding"]["output_path"] == str(context.output_path)
    assert mod.capability_schema_receipt()["required_fields"] == list(mod.REQUIRED_CAPABILITY_FIELDS)


def test_req_arc_lreb_5928_defensive_capability_denials(tmp_path: Path) -> None:
    """REQ-ARC-LREB-5928: malformed capabilities fail closed before action."""

    issuer = mod.ParentIssuer.fixture()
    context = _context(tmp_path)
    capability = mod.issue_capability(
        issuer,
        context,
        issued_monotonic_s=100.0,
        expires_monotonic_s=105.0,
        nonce="defensive-nonce",
        run_id="exp5928-unit-run",
    )

    def reason(mutated: dict[str, object], *, public_key: str | None = None) -> str:
        return mod.verify_and_consume_capability(
            mutated,
            context,
            public_key=public_key or issuer.public_key_hex,
            now_monotonic_s=101.0,
            nonce_ledger=mod.NonceLedger(tmp_path / f"ledger-{len(str(mutated))}"),
            adapter_enabled=False,
        ).reason

    missing = dict(capability)
    del missing["signature"]
    assert reason(missing) == "missing_field:signature"
    assert reason({**capability, "schema": "wrong"}) == "wrong_schema"
    assert reason({**capability, "binding": ["not-a-binding"]}) == "wrong_command"
    assert reason(capability, public_key="00" * 32) == "signature_mismatch"
    assert reason({**capability, "signature": "sha256:not-ed25519"}) == "signature_mismatch"
    assert mod._verify_signature({}, "ed25519:00", "not-hex") is False
    assert reason({**capability, "capability_hash": "sha256:bad"}) == "capability_hash_mismatch"
    assert mod._pid_exists(os.getpid()) is True
    assert mod._read_yaml(tmp_path / "missing.yaml") == {}


def test_scenario_arc_lreb_5928_denial_matrix_blocks_before_environment_action(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-LREB-5928-DENIAL-MATRIX: all variants deny before environment action."""

    matrix = mod.run_denial_matrix(REPO, tmp_path)
    reasons = {row["case"]: row["reason"] for row in matrix["rows"]}

    assert matrix["all_denied_before_environment_action"] is True
    assert matrix["environment_action_count"] == 0
    assert reasons == {
        "absent": "missing_capability",
        "self_issued": "self_issued",
        "expired": "expired",
        "replayed": "nonce_replay",
        "wrong_process": "wrong_process",
        "wrong_command": "wrong_command",
        "wrong_environment": "wrong_environment",
        "wrong_executable": "wrong_executable",
        "adapter_enabled": "adapter_enabled",
        "scope_broadened": "scope_broadened",
        "output_mismatch": "output_mismatch",
    }


def test_scenario_arc_lreb_5928_actual_parent_child_dry_run_consumes_in_live_entrypoint(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-LREB-5928-PARENT-CHILD-CONSUME: real child consumes before action."""

    receipt = mod.run_parent_child_dry_run(REPO, tmp_path / "dry-run")
    consume = receipt["actual_live_entrypoint_consumption_receipt"]
    teardown = receipt["teardown_nonce_invalidation_and_orphan_check"]

    assert receipt["ok"] is True
    assert receipt["parent_pid"] != consume["child_pid"]
    assert consume["actual_live_entrypoint"] == mod.ACTUAL_LIVE_ENTRYPOINT
    assert consume["capability_consumed_before_environment_action"] is True
    assert consume["fixture_only_validation"] is False
    assert consume["environment_action_count"] == 1
    assert consume["model_load_count"] == 0
    assert consume["level_attempt_count"] == 0
    assert consume["adapter_disabled"] is True
    assert receipt["non_scoring_dry_run_receipt"]["returncode"] == 0
    assert receipt["non_scoring_dry_run_receipt"]["no_model_load"] is True
    assert receipt["non_scoring_dry_run_receipt"]["no_level_attempt"] is True
    assert teardown["nonce_replay_denied_before_teardown"] is True
    assert teardown["nonce_ledger_removed_after_teardown"] is True
    assert teardown["child_process_orphaned"] is False
    assert teardown["issuer_secret_persisted"] is False


def test_req_arc_lreb_5928_artifact_schema_and_ready_gates(tmp_path: Path) -> None:
    """REQ-ARC-LREB-5928: artifact validation rejects overclaims and checksum drift."""

    artifact = mod.build_artifact(
        root=REPO,
        work_dir=tmp_path / "artifact-work",
        result_output_path=tmp_path / "experiment_5928.json",
        test_exit_codes={"focused_unit": 0},
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["live_runner_execution_binding_ready_score"] == 1.0
    assert artifact["no_model_inference_or_level_attempt"]["model_load_count"] == 0
    assert artifact["no_model_inference_or_level_attempt"]["level_attempt_count"] == 0
    assert artifact["registry_unchanged"]["unchanged"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="registry_unchanged"):
        bad = json.loads(json.dumps(artifact))
        bad["registry_unchanged"]["unchanged"] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="missing required fields"):
        bad = json.loads(json.dumps(artifact))
        del bad["status"]
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="model_load_count"):
        bad = json.loads(json.dumps(artifact))
        bad["no_model_inference_or_level_attempt"]["model_load_count"] = 1
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="level_attempt_count"):
        bad = json.loads(json.dumps(artifact))
        bad["no_model_inference_or_level_attempt"]["level_attempt_count"] = 1
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="fixture-only"):
        bad = json.loads(json.dumps(artifact))
        bad["actual_live_entrypoint_consumption_receipt"]["fixture_only_validation"] = True
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="consumed before environment"):
        bad = json.loads(json.dumps(artifact))
        bad["actual_live_entrypoint_consumption_receipt"][
            "capability_consumed_before_environment_action"
        ] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="denial"):
        bad = json.loads(json.dumps(artifact))
        bad[
            "absent_self_issued_expired_replayed_wrong_process_command_environment_scope_and_output_denials"
        ]["all_denied_before_environment_action"] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="orphan"):
        bad = json.loads(json.dumps(artifact))
        bad["teardown_nonce_invalidation_and_orphan_check"]["child_process_orphaned"] = True
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="nonce replay"):
        bad = json.loads(json.dumps(artifact))
        bad["teardown_nonce_invalidation_and_orphan_check"][
            "nonce_replay_denied_before_teardown"
        ] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="protected files"):
        bad = json.loads(json.dumps(artifact))
        bad["protected_files_unchanged"]["all_unchanged"] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="ready score"):
        bad = json.loads(json.dumps(artifact))
        bad["actual_live_entrypoint_consumption_receipt"]["actual_live_entrypoint"] = "fixture"
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact({**artifact, "inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact({**artifact, "honest_verdict": "ready: bad_prefix"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact({**artifact, "reproducibility_checksum": "sha256:bad"})


def test_req_arc_lreb_5928_atomic_writer_roundtrips(tmp_path: Path) -> None:
    """REQ-ARC-LREB-5928: writer emits the validated artifact atomically."""

    output = tmp_path / "nested" / "experiment_5928.json"
    artifact = mod.write_artifact(
        root=REPO,
        work_dir=tmp_path / "writer-work",
        output_path=output,
        test_exit_codes={"focused_unit": 0},
    )
    reread = json.loads(output.read_text(encoding="utf-8"))

    assert reread == artifact
    assert not output.with_suffix(output.suffix + ".tmp").exists()
    mod.validate_artifact(reread)


def test_scenario_arc_lreb_5928_repository_artifact_is_stable() -> None:
    """SCENARIO-ARC-LREB-5928-TEARDOWN-IMMUTABILITY: checked-in result is valid."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["status"] == "complete_ready"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["live_runner_execution_binding_ready_score"] == 1.0
    assert artifact["registry_unchanged"]["unchanged"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    mod.validate_artifact(artifact)
