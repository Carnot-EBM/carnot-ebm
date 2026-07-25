"""Tests for Exp5915 ARC live-runner capability lease.

Spec refs: REQ-ARC-LRCL-5915,
SCENARIO-ARC-LRCL-5915-BOUND-LEASE-DRY-RUN,
SCENARIO-ARC-LRCL-5915-DENIAL-MATRIX,
SCENARIO-ARC-LRCL-5915-STABLE-ARTIFACT.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.agentic import arc_live_runner_capability_lease as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/agentic-harness/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _binding() -> mod.ConductorBinding:
    return mod.ConductorBinding(
        authority_id="conductor-exp5916-fixture",
        authority_key=b"exp5916-bound-authority",
        task_id=mod.GRANTEE_TASK_ID,
        runner_id=mod.RUNNER_ID,
        environment_id=mod.ENVIRONMENT_ID,
    )


def _lease() -> dict[str, object]:
    return mod.issue_lease(
        _binding(),
        issued_at=mod.FIXED_NOW,
        expires_at=mod.FIXED_EXPIRY,
        nonce="test-nonce-0001",
    )


def test_req_arc_lrcl_5915_spec_declares_capability_contract() -> None:
    """REQ-ARC-LRCL-5915: OpenSpec freezes lease fields and denial paths."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-LRCL-5915") :]

    for marker in (
        "SCENARIO-ARC-LRCL-5915-BOUND-LEASE-DRY-RUN",
        "SCENARIO-ARC-LRCL-5915-DENIAL-MATRIX",
        "SCENARIO-ARC-LRCL-5915-STABLE-ARTIFACT",
        mod.RESULT_RELATIVE_PATH,
        "authority source",
        "grantee task ID",
        "allowed command pattern",
        "adapter-disabled requirement",
        "replayed-nonce",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for field, principle in mod.REQUIRED_FIELD_PROVENANCE.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_req_arc_lrcl_5915_lease_schema_and_binding_are_exact() -> None:
    """REQ-ARC-LRCL-5915: lease validation binds task, runner, env, scope, and signature."""

    binding = _binding()
    lease = _lease()
    replay = mod.NonceReplayLedger()

    result = mod.validate_lease(
        lease,
        binding,
        command=mod.ALLOWED_COMMAND,
        episode_class=mod.ALLOWED_EPISODE_CLASS,
        adapter_enabled=False,
        now=mod.FIXED_NOW,
        replay_ledger=replay,
    )

    assert result.allowed is True
    assert result.reason == "allowed"
    assert replay.seen == {"test-nonce-0001"}
    assert mod.lease_schema_receipt()["required_fields"] == list(mod.REQUIRED_LEASE_FIELDS)

    tampered = copy.deepcopy(lease)
    tampered["runner_identity"]["runner_id"] = "different-runner"  # type: ignore[index]
    tampered_result = mod.validate_lease(
        tampered,
        binding,
        command=mod.ALLOWED_COMMAND,
        episode_class=mod.ALLOWED_EPISODE_CLASS,
        adapter_enabled=False,
        now=mod.FIXED_NOW,
        replay_ledger=mod.NonceReplayLedger(),
    )
    assert tampered_result.allowed is False
    assert tampered_result.reason == "wrong_runner"


def test_scenario_arc_lrcl_5915_denial_matrix_blocks_before_execution() -> None:
    """SCENARIO-ARC-LRCL-5915-DENIAL-MATRIX: every denial happens before execution."""

    matrix = mod.run_denial_matrix(_binding())
    reasons = {row["case"]: row["reason"] for row in matrix["rows"]}

    assert matrix["all_denied_before_execution"] is True
    assert matrix["live_execution_count"] == 0
    assert reasons == {
        "missing": "missing_lease",
        "expired": "expired",
        "wrong_task": "wrong_task",
        "wrong_environment": "wrong_environment",
        "widened_command": "command_not_allowed",
        "revoked": "revoked",
        "adapter_enabled": "adapter_enabled",
        "replayed_nonce": "nonce_replay",
    }


def test_scenario_arc_lrcl_5915_guarded_dry_run_isolates_and_tears_down() -> None:
    """SCENARIO-ARC-LRCL-5915-BOUND-LEASE-DRY-RUN: allowed path is bounded and isolated."""

    receipt = mod.run_bounded_non_scored_dry_run(_binding())

    assert receipt["ok"] is True
    assert receipt["e3_policy_importable"] is True
    assert receipt["scored_public_execution_count"] == 0
    assert receipt["model_load_count"] == 0
    assert receipt["source_bfs_adapter_prior_game_and_hidden_state_access_count"] == 0
    assert receipt["executions"] == 2
    assert receipt["cell_initial_state_lengths"] == [0, 0]
    assert receipt["cell_final_state_lengths_before_teardown"] == [1, 1]
    assert receipt["cell_state_lengths_after_teardown"] == [0, 0]
    assert receipt["persistent_cross_cell_state_detected"] is False
    assert receipt["teardown_called_count"] == 2


def test_req_arc_lrcl_5915_artifact_schema_rejects_overclaims() -> None:
    """REQ-ARC-LRCL-5915: artifact validation rejects scope, verdict, and checksum drift."""

    artifact = mod.build_artifact(root=REPO, test_exit_codes={"focused_unit": 0})

    assert artifact["live_runner_capability_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="public_level_target_selected"):
        mod.validate_artifact({**artifact, "public_level_target_selected": True})
    with pytest.raises(ValueError, match="scored_public_execution_count"):
        mod.validate_artifact({**artifact, "scored_public_execution_count": 1})
    with pytest.raises(ValueError, match="model_load_count"):
        mod.validate_artifact({**artifact, "model_load_count": 1})
    with pytest.raises(ValueError, match="source_bfs_adapter"):
        mod.validate_artifact(
            {
                **artifact,
                "source_bfs_adapter_prior_game_and_hidden_state_access_count": 1,
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact({**artifact, "inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact({**artifact, "verifier_is_oracle": True})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact({**artifact, "honest_verdict": "ready: bad_prefix"})
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(
            {
                **artifact,
                "live_runner_capability_ready_score": 1.0,
                "denial_path_matrix": {"all_denied_before_execution": False},
            }
        )
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact({**artifact, "reproducibility_checksum": "sha256:bad"})


def test_req_arc_lrcl_5915_defensive_denials_and_writer(tmp_path: Path) -> None:
    """REQ-ARC-LRCL-5915: malformed leases fail closed and writer round-trips."""

    binding = _binding()
    base = _lease()

    def validate(mutated: dict[str, object]) -> str:
        return mod.validate_lease(
            mutated,
            binding,
            command=mod.ALLOWED_COMMAND,
            episode_class=mod.ALLOWED_EPISODE_CLASS,
            adapter_enabled=False,
            now=mod.FIXED_NOW,
            replay_ledger=mod.NonceReplayLedger(),
        ).reason

    missing = dict(base)
    del missing["signature"]
    assert validate(missing) == "missing_field:signature"
    assert validate({**base, "schema": "wrong"}) == "wrong_schema"
    assert (
        mod.validate_lease(
            base,
            binding,
            command=mod.ALLOWED_COMMAND,
            episode_class="scored_public_game",
            adapter_enabled=False,
            now=mod.FIXED_NOW,
            replay_ledger=mod.NonceReplayLedger(),
        ).reason
        == "episode_class_not_allowed"
    )

    widened_bounds = copy.deepcopy(base)
    widened_bounds["resource_bounds"]["max_actions"] = 3  # type: ignore[index]
    assert validate(widened_bounds) == "resource_bounds_exceeded"

    invalid_bounds = copy.deepcopy(base)
    invalid_bounds["resource_bounds"]["max_actions"] = "many"  # type: ignore[index]
    assert validate(invalid_bounds) == "resource_bounds_exceeded"

    bad_signature = {**base, "signature": "sha256:bad"}
    assert validate(bad_signature) == "signature_mismatch"

    bad_hash = {**base, "lease_hash": "sha256:bad"}
    assert validate(bad_hash) == "lease_hash_mismatch"

    artifact = mod.build_artifact(root=REPO, test_exit_codes={"focused_unit": 0})
    missing_field = dict(artifact)
    del missing_field["status"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_field)

    output = tmp_path / "experiment_5915.json"
    written = mod.write_artifact(
        root=REPO,
        output_path=output,
        test_exit_codes={"focused_unit": 0},
    )
    assert json.loads(output.read_text(encoding="utf-8")) == written
    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    mod.validate_artifact(written)


def test_scenario_arc_lrcl_5915_repository_artifact_is_stable() -> None:
    """SCENARIO-ARC-LRCL-5915-STABLE-ARTIFACT: checked-in result is valid and scoped."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["status"] == "complete_ready"
    assert artifact["public_level_target_selected"] is False
    assert artifact["scored_public_execution_count"] == 0
    assert artifact["model_load_count"] == 0
    assert artifact["source_bfs_adapter_prior_game_and_hidden_state_access_count"] == 0
    assert artifact["registry_unchanged"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["live_runner_capability_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["honest_verdict"].startswith("complete_ready:")
    mod.validate_artifact(artifact)
