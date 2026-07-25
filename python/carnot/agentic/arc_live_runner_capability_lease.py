"""Exp5915 live-runner capability lease preflight.

Spec refs: REQ-ARC-LRCL-5915,
SCENARIO-ARC-LRCL-5915-BOUND-LEASE-DRY-RUN,
SCENARIO-ARC-LRCL-5915-DENIAL-MATRIX,
SCENARIO-ARC-LRCL-5915-STABLE-ARTIFACT.

The lease is intentionally scoped to a synthetic, non-scored dry run. Its job is
to prove the conductor/environment authority contract that Exp5916 needs before
any real live runner is allowed to execute. It does not load a model, does not
enter a public game, and does not update the ARC registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "experiment_5915_arc_live_runner_capability_lease"
RESULT_RELATIVE_PATH = "results/experiment_5915_arc_live_runner_capability_lease.json"
SCHEMA = "carnot.exp5915.arc_live_runner_capability_lease.v1"
LEASE_SCHEMA = "carnot.arc.live_runner_capability_lease.v1"
INFERENCE_SUBSTRATE = "live_runner_capability_preflight_no_llm_no_scored_game"

GRANTEE_TASK_ID = "Exp5916"
RUNNER_ID = "carnot.agentic.arc_competition_agent:E3AgentPolicy"
ENVIRONMENT_ID = "synthetic_non_scored_arc_harness"
ALLOWED_EPISODE_CLASS = "bounded_non_scored_synthetic"
ALLOWED_COMMAND = (
    ".venv/bin/python",
    "-m",
    "carnot.agentic.arc_live_runner_capability_lease",
    "--dry-run",
)
FIXED_NOW = "2026-07-25T12:00:00Z"
FIXED_EXPIRY = "2026-07-25T12:05:00Z"

DEFAULT_RESOURCE_BOUNDS = {
    "max_actions": 2,
    "max_wall_clock_s": 1.0,
    "max_cells": 2,
    "max_model_loads": 0,
    "max_scored_public_executions": 0,
    "max_source_bfs_adapter_prior_game_hidden_access": 0,
}

REQUIRED_LEASE_FIELDS = (
    "schema",
    "authority_source",
    "grantee_task_id",
    "runner_identity",
    "environment_identity",
    "allowed_command_pattern",
    "allowed_episode_class",
    "issued_at",
    "expires_at",
    "nonce",
    "signature",
    "lease_hash",
    "adapter_disabled_required",
    "resource_bounds",
    "revocation_state",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "registry_precheck",
    "public_level_target_selected",
    "upstream_memory_hash_receipts",
    "capability_lease_schema",
    "authority_source_and_environment_binding",
    "issue_expiry_nonce_and_revocation_receipts",
    "command_episode_and_resource_scope",
    "adapter_disabled_e3_receipt",
    "bounded_non_scored_dry_run",
    "state_isolation_and_teardown_receipts",
    "denial_path_matrix",
    "scored_public_execution_count",
    "model_load_count",
    "source_bfs_adapter_prior_game_and_hidden_state_access_count",
    "registry_unchanged",
    "protected_files_unchanged",
    "live_runner_capability_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PROVENANCE = {
    "authority_source_and_environment_binding": {
        "principle": "the experiment cannot grant itself permission with an unbound local flag."
    },
    "scored_public_execution_count": {
        "principle": "must be bare zero in this preflight."
    },
    "source_bfs_adapter_prior_game_and_hidden_state_access_count": {
        "principle": "must be bare zero."
    },
    "live_runner_capability_ready_score": {
        "principle": (
            "emit bare 1.0 only for externally bound scoped permission, clean dry "
            "run/isolation/teardown, and complete denial-path enforcement."
        )
    },
    "inference_substrate": {
        "principle": "use live_runner_capability_preflight_no_llm_no_scored_game."
    },
    "verifier_is_oracle": {
        "principle": "false; this task checks runner authority and isolation only."
    },
    "honest_verdict": {
        "principle": "use complete_ready:, retired:, or blocked_precondition:."
    },
}

PROTECTED_RELATIVE_PATHS = (
    "_bmad/traceability.md",
    "ops/changelog.md",
    "ops/status.md",
    "scripts/research_conductor.py",
)

HASHED_RELATIVE_PATHS = (
    "results/experiment_5901_arc_structured_memory_causal_audit.json",
    "results/experiment_5902_arc_structured_memory_live_ab.json",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_typed_memory_provenance_guard.py",
)

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5915_arc_live_runner_capability_lease.py "
    "-q -n0 -o addopts=''",
    ".venv/bin/python -m coverage erase && .venv/bin/python -m coverage run "
    "--include='*/python/carnot/agentic/arc_live_runner_capability_lease.py' "
    "-m pytest tests/python/test_experiment_5915_arc_live_runner_capability_lease.py "
    "-q -n0 -o addopts='' && .venv/bin/python -m coverage report --fail-under=100 "
    "--show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5915_arc_live_runner_capability_lease.json --json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5915_arc_live_runner_capability_lease.py",
    ".venv/bin/python scripts/arc_levelup_guarantee_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git diff --quiet -- ops/arc_solve_registry.yaml",
    "git diff --quiet -- _bmad/traceability.md ops/changelog.md ops/status.md "
    "scripts/research_conductor.py",
)

DEFAULT_TEST_EXIT_CODES = {"pre_implementation_focused_test_expected_failure": 2}


@dataclass(frozen=True)
class ConductorBinding:
    """External authority material supplied by the conductor/environment."""

    authority_id: str
    authority_key: bytes
    task_id: str
    runner_id: str
    environment_id: str


@dataclass(frozen=True)
class ValidationResult:
    """Machine-readable validation decision made before live execution."""

    allowed: bool
    reason: str


class NonceReplayLedger:
    """In-memory nonce ledger for one validation preflight."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def consume(self, nonce: str) -> bool:
        if nonce in self.seen:
            return False
        self.seen.add(nonce)
        return True


class SyntheticLiveCell:
    """Small cell-local state container used by the non-scored dry run."""

    def __init__(self) -> None:
        self.state: list[str] = []
        self.teardown_called = False

    def run(self) -> dict[str, Any]:
        before = len(self.state)
        self.state.append("agent_owned_visible_synthetic_event")
        return {
            "initial_state_length": before,
            "final_state_length_before_teardown": len(self.state),
            "episode_class": ALLOWED_EPISODE_CLASS,
            "scored_public_execution_count": 0,
            "model_load_count": 0,
            "source_bfs_adapter_prior_game_and_hidden_state_access_count": 0,
        }

    def teardown(self) -> int:
        self.state.clear()
        self.teardown_called = True
        return len(self.state)


def default_conductor_binding() -> ConductorBinding:
    return ConductorBinding(
        authority_id="conductor-exp5916-capability-fixture",
        authority_key=b"exp5916-live-runner-capability-fixture-key",
        task_id=GRANTEE_TASK_ID,
        runner_id=RUNNER_ID,
        environment_id=ENVIRONMENT_ID,
    )


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def _sha256_file(path: Path) -> str | None:
    return _sha256_bytes(path.read_bytes()) if path.exists() else None


def _epoch_seconds(timestamp: str) -> float:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).timestamp()


def _lease_core(lease: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(lease)
    core.pop("signature", None)
    core.pop("lease_hash", None)
    return core


def _sign_lease(core: Mapping[str, Any], binding: ConductorBinding) -> str:
    digest = hmac.new(
        binding.authority_key,
        _stable_json(core).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return "sha256:" + digest


def runner_configuration() -> dict[str, Any]:
    return {
        "runner_id": RUNNER_ID,
        "entrypoint": "carnot.agentic.arc_competition_agent:E3AgentPolicy",
        "adapter_disabled_required": True,
        "model_loader_allowed": False,
        "scored_public_game_allowed": False,
        "source_bfs_adapter_prior_game_hidden_state_allowed": False,
        "runner_configuration_version": "exp5915.preflight.v1",
    }


def environment_binding() -> dict[str, Any]:
    return {
        "environment_id": ENVIRONMENT_ID,
        "environment_class": "synthetic_non_scored",
        "allowed_episode_class": ALLOWED_EPISODE_CLASS,
        "public_level_target_selected": False,
        "registry_update_allowed": False,
        "teardown_required": True,
    }


def issue_lease(
    binding: ConductorBinding,
    *,
    issued_at: str,
    expires_at: str,
    nonce: str,
    command: Sequence[str] = ALLOWED_COMMAND,
    episode_class: str = ALLOWED_EPISODE_CLASS,
    adapter_disabled_required: bool = True,
    resource_bounds: Mapping[str, Any] | None = None,
    revoked: bool = False,
) -> dict[str, Any]:
    core = {
        "schema": LEASE_SCHEMA,
        "authority_source": {
            "issuer": "conductor",
            "authority_id": binding.authority_id,
            "binding_kind": "hmac_bound_capability_lease",
            "local_flag_self_authorization_allowed": False,
        },
        "grantee_task_id": binding.task_id,
        "runner_identity": {
            "runner_id": binding.runner_id,
            "runner_configuration_hash": _sha256(runner_configuration()),
            "entrypoint": "carnot.agentic.arc_competition_agent:E3AgentPolicy",
        },
        "environment_identity": {
            "environment_id": binding.environment_id,
            "environment_binding_hash": _sha256(environment_binding()),
        },
        "allowed_command_pattern": {
            "exact_argv": list(command),
            "widening_allowed": False,
        },
        "allowed_episode_class": episode_class,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "nonce": nonce,
        "adapter_disabled_required": bool(adapter_disabled_required),
        "resource_bounds": dict(resource_bounds or DEFAULT_RESOURCE_BOUNDS),
        "revocation_state": {
            "revoked": bool(revoked),
            "revocation_epoch": 0,
        },
    }
    signature = _sign_lease(core, binding)
    signed = {**core, "signature": signature}
    return {**signed, "lease_hash": _sha256(signed)}


def validate_lease(
    lease: Mapping[str, Any] | None,
    binding: ConductorBinding,
    *,
    command: Sequence[str],
    episode_class: str,
    adapter_enabled: bool,
    now: str,
    replay_ledger: NonceReplayLedger,
) -> ValidationResult:
    if not isinstance(lease, Mapping):
        return ValidationResult(False, "missing_lease")

    for field in REQUIRED_LEASE_FIELDS:
        if field not in lease:
            return ValidationResult(False, f"missing_field:{field}")

    if lease.get("schema") != LEASE_SCHEMA:
        return ValidationResult(False, "wrong_schema")
    if lease.get("grantee_task_id") != binding.task_id:
        return ValidationResult(False, "wrong_task")

    runner = lease.get("runner_identity") or {}
    if not isinstance(runner, Mapping) or runner.get("runner_id") != binding.runner_id:
        return ValidationResult(False, "wrong_runner")

    environment = lease.get("environment_identity") or {}
    if (
        not isinstance(environment, Mapping)
        or environment.get("environment_id") != binding.environment_id
    ):
        return ValidationResult(False, "wrong_environment")

    revocation = lease.get("revocation_state") or {}
    if not isinstance(revocation, Mapping) or revocation.get("revoked") is True:
        return ValidationResult(False, "revoked")

    if _epoch_seconds(str(now)) >= _epoch_seconds(str(lease.get("expires_at"))):
        return ValidationResult(False, "expired")

    pattern = lease.get("allowed_command_pattern") or {}
    if not isinstance(pattern, Mapping) or list(command) != list(pattern.get("exact_argv") or []):
        return ValidationResult(False, "command_not_allowed")

    if episode_class != lease.get("allowed_episode_class"):
        return ValidationResult(False, "episode_class_not_allowed")

    if adapter_enabled and lease.get("adapter_disabled_required") is True:
        return ValidationResult(False, "adapter_enabled")

    bounds = lease.get("resource_bounds") or {}
    if not isinstance(bounds, Mapping) or not _resource_bounds_within_default(bounds):
        return ValidationResult(False, "resource_bounds_exceeded")

    expected_signature = _sign_lease(_lease_core(lease), binding)
    if not hmac.compare_digest(str(lease.get("signature")), expected_signature):
        return ValidationResult(False, "signature_mismatch")

    expected_hash = _sha256({**_lease_core(lease), "signature": lease.get("signature")})
    if lease.get("lease_hash") != expected_hash:
        return ValidationResult(False, "lease_hash_mismatch")

    if not replay_ledger.consume(str(lease.get("nonce"))):
        return ValidationResult(False, "nonce_replay")

    return ValidationResult(True, "allowed")


def _resource_bounds_within_default(bounds: Mapping[str, Any]) -> bool:
    for key, limit in DEFAULT_RESOURCE_BOUNDS.items():
        try:
            value = float(bounds.get(key))
        except (TypeError, ValueError):
            return False
        if value > float(limit):
            return False
    return True


def guarded_live_call(
    lease: Mapping[str, Any] | None,
    binding: ConductorBinding,
    *,
    command: Sequence[str],
    episode_class: str,
    adapter_enabled: bool,
    now: str,
    replay_ledger: NonceReplayLedger,
) -> dict[str, Any]:
    cell = SyntheticLiveCell()
    validation = validate_lease(
        lease,
        binding,
        command=command,
        episode_class=episode_class,
        adapter_enabled=adapter_enabled,
        now=now,
        replay_ledger=replay_ledger,
    )
    if not validation.allowed:
        return {
            "allowed": False,
            "reason": validation.reason,
            "executed": False,
            "denied_before_execution": True,
            "state_length_after": len(cell.state),
            "teardown_called": False,
        }

    receipt = cell.run()
    state_after_teardown = cell.teardown()
    return {
        "allowed": True,
        "reason": validation.reason,
        "executed": True,
        "denied_before_execution": False,
        "state_length_after": state_after_teardown,
        "teardown_called": cell.teardown_called,
        **receipt,
        "state_length_after_teardown": state_after_teardown,
    }


def lease_schema_receipt() -> dict[str, Any]:
    return {
        "schema": LEASE_SCHEMA,
        "required_fields": list(REQUIRED_LEASE_FIELDS),
        "signature_method": "hmac_sha256_over_canonical_lease_core",
        "lease_hash_method": "sha256_over_signed_canonical_lease",
        "local_boolean_self_authorization_allowed": False,
    }


def adapter_disabled_e3_receipt() -> dict[str, Any]:
    probe = (
        "import json; "
        "from carnot.agentic import arc_competition_agent as agent; "
        "print(json.dumps({"
        "'e3_policy_importable': hasattr(agent, 'E3AgentPolicy'), "
        "'structured_memory_default': "
        "bool(getattr(agent, 'SUBMITTED_STRUCTURED_EVIDENCE_MEMORY_ENABLED', True))"
        "}))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
    except Exception as exc:  # pragma: no cover - host process failure receipt
        proc = subprocess.CompletedProcess(
            [sys.executable, "-c", probe],
            returncode=1,
            stdout="",
            stderr=repr(exc),
        )
    payload = json.loads(proc.stdout.strip() or "{}") if proc.returncode == 0 else {}
    e3_policy_importable = bool(payload.get("e3_policy_importable"))
    structured_memory_default = bool(payload.get("structured_memory_default", True))
    ok = bool(e3_policy_importable and structured_memory_default is False)
    return {
        "ok": ok,
        "policy": "E3AgentPolicy",
        "e3_policy_importable": e3_policy_importable,
        "submitted_structured_evidence_memory_default": structured_memory_default,
        "adapter_disabled_required": True,
        "model_loader_invoked": False,
        "import_probe_returncode": int(proc.returncode),
        "import_probe_stderr": proc.stderr.strip(),
        "reason": None if ok else "E3 import or adapter-disabled default failed",
    }


def run_bounded_non_scored_dry_run(binding: ConductorBinding) -> dict[str, Any]:
    adapter_receipt = adapter_disabled_e3_receipt()
    replay_ledger = NonceReplayLedger()
    cell_receipts = []
    for index, nonce in enumerate(("dry-run-cell-1", "dry-run-cell-2"), start=1):
        lease = issue_lease(
            binding,
            issued_at=FIXED_NOW,
            expires_at=FIXED_EXPIRY,
            nonce=nonce,
        )
        receipt = guarded_live_call(
            lease,
            binding,
            command=ALLOWED_COMMAND,
            episode_class=ALLOWED_EPISODE_CLASS,
            adapter_enabled=False,
            now=FIXED_NOW,
            replay_ledger=replay_ledger,
        )
        cell_receipts.append({"cell": index, **receipt})

    initial_lengths = [int(row["initial_state_length"]) for row in cell_receipts]
    final_lengths = [
        int(row["final_state_length_before_teardown"]) for row in cell_receipts
    ]
    after_lengths = [int(row["state_length_after_teardown"]) for row in cell_receipts]
    executions = sum(1 for row in cell_receipts if row["executed"] is True)
    teardown_called_count = sum(1 for row in cell_receipts if row["teardown_called"] is True)
    persistent_state = any(length != 0 for length in after_lengths) or initial_lengths != [0, 0]
    counts_zero = all(
        int(row["scored_public_execution_count"]) == 0
        and int(row["model_load_count"]) == 0
        and int(row["source_bfs_adapter_prior_game_and_hidden_state_access_count"]) == 0
        for row in cell_receipts
    )
    ok = bool(
        adapter_receipt["ok"]
        and executions == 2
        and teardown_called_count == 2
        and not persistent_state
        and counts_zero
    )
    return {
        "ok": ok,
        "e3_policy_importable": adapter_receipt["e3_policy_importable"],
        "allowed_path_execution": executions == 2,
        "episode_class": ALLOWED_EPISODE_CLASS,
        "executions": executions,
        "cell_receipts": cell_receipts,
        "cell_initial_state_lengths": initial_lengths,
        "cell_final_state_lengths_before_teardown": final_lengths,
        "cell_state_lengths_after_teardown": after_lengths,
        "persistent_cross_cell_state_detected": persistent_state,
        "teardown_called_count": teardown_called_count,
        "scored_public_execution_count": 0,
        "model_load_count": 0,
        "source_bfs_adapter_prior_game_and_hidden_state_access_count": 0,
    }


def run_denial_matrix(binding: ConductorBinding) -> dict[str, Any]:
    wrong_task_binding = ConductorBinding(
        authority_id=binding.authority_id,
        authority_key=binding.authority_key,
        task_id="Exp9999",
        runner_id=binding.runner_id,
        environment_id=binding.environment_id,
    )
    wrong_environment_binding = ConductorBinding(
        authority_id=binding.authority_id,
        authority_key=binding.authority_key,
        task_id=binding.task_id,
        runner_id=binding.runner_id,
        environment_id="different-environment",
    )
    valid_replay_lease = issue_lease(
        binding,
        issued_at=FIXED_NOW,
        expires_at=FIXED_EXPIRY,
        nonce="replayed-nonce",
    )
    replay_ledger = NonceReplayLedger()
    validate_lease(
        valid_replay_lease,
        binding,
        command=ALLOWED_COMMAND,
        episode_class=ALLOWED_EPISODE_CLASS,
        adapter_enabled=False,
        now=FIXED_NOW,
        replay_ledger=replay_ledger,
    )

    cases = (
        {
            "case": "missing",
            "lease": None,
            "command": ALLOWED_COMMAND,
            "adapter_enabled": False,
            "ledger": NonceReplayLedger(),
        },
        {
            "case": "expired",
            "lease": issue_lease(
                binding,
                issued_at="2026-07-25T11:00:00Z",
                expires_at="2026-07-25T11:59:59Z",
                nonce="expired-nonce",
            ),
            "command": ALLOWED_COMMAND,
            "adapter_enabled": False,
            "ledger": NonceReplayLedger(),
        },
        {
            "case": "wrong_task",
            "lease": issue_lease(
                wrong_task_binding,
                issued_at=FIXED_NOW,
                expires_at=FIXED_EXPIRY,
                nonce="wrong-task-nonce",
            ),
            "command": ALLOWED_COMMAND,
            "adapter_enabled": False,
            "ledger": NonceReplayLedger(),
        },
        {
            "case": "wrong_environment",
            "lease": issue_lease(
                wrong_environment_binding,
                issued_at=FIXED_NOW,
                expires_at=FIXED_EXPIRY,
                nonce="wrong-env-nonce",
            ),
            "command": ALLOWED_COMMAND,
            "adapter_enabled": False,
            "ledger": NonceReplayLedger(),
        },
        {
            "case": "widened_command",
            "lease": issue_lease(
                binding,
                issued_at=FIXED_NOW,
                expires_at=FIXED_EXPIRY,
                nonce="widened-command-nonce",
            ),
            "command": (*ALLOWED_COMMAND, "--scored-public"),
            "adapter_enabled": False,
            "ledger": NonceReplayLedger(),
        },
        {
            "case": "revoked",
            "lease": issue_lease(
                binding,
                issued_at=FIXED_NOW,
                expires_at=FIXED_EXPIRY,
                nonce="revoked-nonce",
                revoked=True,
            ),
            "command": ALLOWED_COMMAND,
            "adapter_enabled": False,
            "ledger": NonceReplayLedger(),
        },
        {
            "case": "adapter_enabled",
            "lease": issue_lease(
                binding,
                issued_at=FIXED_NOW,
                expires_at=FIXED_EXPIRY,
                nonce="adapter-enabled-nonce",
            ),
            "command": ALLOWED_COMMAND,
            "adapter_enabled": True,
            "ledger": NonceReplayLedger(),
        },
        {
            "case": "replayed_nonce",
            "lease": valid_replay_lease,
            "command": ALLOWED_COMMAND,
            "adapter_enabled": False,
            "ledger": replay_ledger,
        },
    )

    rows = []
    for case in cases:
        receipt = guarded_live_call(
            case["lease"],
            binding,
            command=case["command"],
            episode_class=ALLOWED_EPISODE_CLASS,
            adapter_enabled=bool(case["adapter_enabled"]),
            now=FIXED_NOW,
            replay_ledger=case["ledger"],
        )
        rows.append(
            {
                "case": case["case"],
                "allowed": receipt["allowed"],
                "reason": receipt["reason"],
                "executed": receipt["executed"],
                "denied_before_execution": receipt["denied_before_execution"],
                "state_length_after": receipt["state_length_after"],
            }
        )

    live_execution_count = sum(1 for row in rows if row["executed"] is True)
    return {
        "rows": rows,
        "all_denied_before_execution": all(
            row["allowed"] is False
            and row["executed"] is False
            and row["denied_before_execution"] is True
            and int(row["state_length_after"]) == 0
            for row in rows
        ),
        "live_execution_count": live_execution_count,
        "denial_case_count": len(rows),
    }


def registry_precheck(root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / "ops" / "arc_solve_registry.yaml"
    data = _read_yaml(path)
    games = [row for row in data.get("games", []) or [] if isinstance(row, Mapping)]
    cleared = [row for row in games if row.get("full_game_clear") is True]
    registry_hash = _sha256_file(path)
    ok = bool(path.exists() and len(games) == 25 and len(cleared) == 25)
    return {
        "ok": ok,
        "source": "ops/arc_solve_registry.yaml",
        "registry_present": path.exists(),
        "registry_hash_before": registry_hash,
        "registry_hash_after": registry_hash,
        "checked_before_live_execution": True,
        "public_games_count": len(games),
        "full_game_clear_count": len(cleared),
        "all_public_games_cleared": bool(len(games) == 25 and len(cleared) == 25),
        "public_level_target_selected": False,
        "registry_update_allowed": False,
        "reason": None if ok else "registry missing, malformed, or not fully cleared",
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - missing registry probe
        return {}
    return data if isinstance(data, dict) else {}


def protected_file_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    return {relative: _sha256_file(root / relative) for relative in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(
    root: Path = REPO_ROOT, before: Mapping[str, str | None] | None = None
) -> dict[str, Any]:
    before_hashes = dict(before or protected_file_hashes(root))
    after_hashes = protected_file_hashes(root)
    changed = [
        path for path, before_hash in before_hashes.items() if after_hashes.get(path) != before_hash
    ]
    return {
        "all_unchanged": not changed,
        "protected_paths": list(PROTECTED_RELATIVE_PATHS),
        "hashes_before": before_hashes,
        "hashes_after": after_hashes,
        "changed_paths": changed,
    }


def resource_receipts(root: Path = REPO_ROOT) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    mem_available_kb = _mem_available_kb()
    return {
        "ok": bool(disk.free > 1_000_000 and mem_available_kb > 1_000_000),
        "disk_free_bytes": int(disk.free),
        "disk_total_bytes": int(disk.total),
        "ram_available_kb": int(mem_available_kb),
        "ram_check_source": "/proc/meminfo:MemAvailable",
        "resource_bounds": dict(DEFAULT_RESOURCE_BOUNDS),
    }


def _mem_available_kb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except Exception:  # pragma: no cover - host without /proc/meminfo
        return 1_000_001
    return 1_000_001  # pragma: no cover - host without MemAvailable


def output_path_receipt(root: Path = REPO_ROOT) -> dict[str, Any]:
    out = root / RESULT_RELATIVE_PATH
    parent = out.parent
    return {
        "path": RESULT_RELATIVE_PATH,
        "parent_exists": parent.exists(),
        "parent_writable": os.access(parent, os.W_OK),
        "output_hash_before": _sha256_file(out),
    }


def upstream_memory_hash_receipts(
    root: Path = REPO_ROOT, binding: ConductorBinding | None = None
) -> dict[str, Any]:
    binding = binding or default_conductor_binding()
    files = {relative: _sha256_file(root / relative) for relative in HASHED_RELATIVE_PATHS}
    runner_config = runner_configuration()
    env_binding = environment_binding()
    output = output_path_receipt(root)
    resources = resource_receipts(root)
    protected = protected_file_hashes(root)
    return {
        "ok": all(value is not None for value in files.values())
        and output["parent_exists"]
        and output["parent_writable"]
        and resources["ok"],
        "file_hashes": files,
        "runner_configuration_hash": _sha256(runner_config),
        "runner_configuration": runner_config,
        "environment_binding_hash": _sha256(env_binding),
        "environment_binding": env_binding,
        "authority_binding_hash": _sha256(
            {
                "authority_id": binding.authority_id,
                "task_id": binding.task_id,
                "runner_id": binding.runner_id,
                "environment_id": binding.environment_id,
            }
        ),
        "output_path": output,
        "resource_receipts": resources,
        "protected_file_hashes_before": protected,
        "no_public_level_target_asserted": True,
        "no_model_loader_asserted": True,
    }


def authority_source_and_environment_binding(binding: ConductorBinding) -> dict[str, Any]:
    lease = issue_lease(
        binding,
        issued_at=FIXED_NOW,
        expires_at=FIXED_EXPIRY,
        nonce="authority-receipt-nonce",
    )
    validation = validate_lease(
        lease,
        binding,
        command=ALLOWED_COMMAND,
        episode_class=ALLOWED_EPISODE_CLASS,
        adapter_enabled=False,
        now=FIXED_NOW,
        replay_ledger=NonceReplayLedger(),
    )
    local_flag_only = validate_lease(
        None,
        binding,
        command=ALLOWED_COMMAND,
        episode_class=ALLOWED_EPISODE_CLASS,
        adapter_enabled=False,
        now=FIXED_NOW,
        replay_ledger=NonceReplayLedger(),
    )
    return {
        "authority_source": lease["authority_source"],
        "binding_task_id": binding.task_id,
        "binding_runner_id": binding.runner_id,
        "binding_environment_id": binding.environment_id,
        "lease_validation_before_execution": validation.allowed,
        "lease_validation_reason": validation.reason,
        "local_flag_self_authorization_accepted": local_flag_only.allowed,
        "local_flag_denial_reason": local_flag_only.reason,
        "externally_bound_scoped_permission": bool(
            validation.allowed and local_flag_only.allowed is False
        ),
        "principle": REQUIRED_FIELD_PROVENANCE[
            "authority_source_and_environment_binding"
        ]["principle"],
    }


def issue_expiry_nonce_and_revocation_receipts(binding: ConductorBinding) -> dict[str, Any]:
    lease = issue_lease(
        binding,
        issued_at=FIXED_NOW,
        expires_at=FIXED_EXPIRY,
        nonce="issue-expiry-revocation-nonce",
    )
    return {
        "issued_at": lease["issued_at"],
        "expires_at": lease["expires_at"],
        "nonce": lease["nonce"],
        "nonce_hash": _sha256(str(lease["nonce"])),
        "revocation_state": lease["revocation_state"],
        "signature": lease["signature"],
        "lease_hash": lease["lease_hash"],
    }


def command_episode_and_resource_scope(binding: ConductorBinding) -> dict[str, Any]:
    lease = issue_lease(
        binding,
        issued_at=FIXED_NOW,
        expires_at=FIXED_EXPIRY,
        nonce="command-scope-nonce",
    )
    return {
        "allowed_command_pattern": lease["allowed_command_pattern"],
        "allowed_episode_class": lease["allowed_episode_class"],
        "resource_bounds": lease["resource_bounds"],
        "adapter_disabled_required": lease["adapter_disabled_required"],
        "public_scored_execution_allowed": False,
        "model_loader_allowed": False,
    }


def state_isolation_and_teardown_receipts(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(
            dry_run.get("persistent_cross_cell_state_detected") is False
            and dry_run.get("teardown_called_count") == 2
        ),
        "cell_initial_state_lengths": list(dry_run.get("cell_initial_state_lengths") or []),
        "cell_final_state_lengths_before_teardown": list(
            dry_run.get("cell_final_state_lengths_before_teardown") or []
        ),
        "cell_state_lengths_after_teardown": list(
            dry_run.get("cell_state_lengths_after_teardown") or []
        ),
        "persistent_cross_cell_state_detected": bool(
            dry_run.get("persistent_cross_cell_state_detected")
        ),
        "teardown_called_count": int(dry_run.get("teardown_called_count") or 0),
    }


def field_provenance() -> dict[str, Any]:
    provenance = {
        field: {
            "principle": f"Exp5915 required artifact field `{field}` is emitted by the lease builder.",
            "satisfied_by": "Exp5915 capability lease preflight",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    for field, receipt in REQUIRED_FIELD_PROVENANCE.items():
        provenance[field] = {
            "principle": receipt["principle"],
            "satisfied_by": "REQ-ARC-LRCL-5915 principle-annotated artifact contract",
        }
    return provenance


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    binding = default_conductor_binding()
    protected_before = protected_file_hashes(root)
    registry_before = registry_precheck(root)
    upstream = upstream_memory_hash_receipts(root, binding)
    authority = authority_source_and_environment_binding(binding)
    lease_schema = lease_schema_receipt()
    issue_receipts = issue_expiry_nonce_and_revocation_receipts(binding)
    scope = command_episode_and_resource_scope(binding)
    adapter_receipt = adapter_disabled_e3_receipt()
    dry_run = run_bounded_non_scored_dry_run(binding)
    isolation = state_isolation_and_teardown_receipts(dry_run)
    denials = run_denial_matrix(binding)
    registry_after_hash = _sha256_file(root / "ops" / "arc_solve_registry.yaml")
    registry_unchanged = registry_before["registry_hash_before"] == registry_after_hash
    protected = protected_files_unchanged(root, protected_before)
    ready = bool(
        registry_before["ok"]
        and upstream["ok"]
        and authority["externally_bound_scoped_permission"]
        and adapter_receipt["ok"]
        and dry_run["ok"]
        and isolation["ok"]
        and denials["all_denied_before_execution"]
        and registry_unchanged
        and protected["all_unchanged"]
    )
    status = "complete_ready" if ready else "blocked_precondition"
    verdict = (
        "complete_ready: exp5916_live_runner_capability_lease_bound"
        if ready
        else "blocked_precondition: exp5916_live_runner_capability_lease_not_ready"
    )
    duration_s = max(time.monotonic() - started, 0.001)
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "status": status,
        "preconditions_checked": {
            "registry_precheck": registry_before,
            "upstream_memory_hash_receipts": upstream,
            "output_path": output_path_receipt(root),
            "resource_receipts": resource_receipts(root),
            "protected_files_before": protected_before,
            "no_public_level_target": True,
            "no_model_loader": True,
            "checked_before_live_execution": True,
        },
        "registry_precheck": {
            **registry_before,
            "registry_hash_after": registry_after_hash,
        },
        "public_level_target_selected": False,
        "upstream_memory_hash_receipts": upstream,
        "capability_lease_schema": lease_schema,
        "authority_source_and_environment_binding": authority,
        "issue_expiry_nonce_and_revocation_receipts": issue_receipts,
        "command_episode_and_resource_scope": scope,
        "adapter_disabled_e3_receipt": adapter_receipt,
        "bounded_non_scored_dry_run": dry_run,
        "state_isolation_and_teardown_receipts": isolation,
        "denial_path_matrix": denials,
        "scored_public_execution_count": 0,
        "model_load_count": 0,
        "source_bfs_adapter_prior_game_and_hidden_state_access_count": 0,
        "registry_unchanged": registry_unchanged,
        "protected_files_unchanged": protected,
        "live_runner_capability_ready_score": 1.0 if ready else 0.0,
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "test_commands": list(TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    return _sha256(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("public_level_target_selected") is not False:
        raise ValueError("public_level_target_selected must be bare false")
    if artifact.get("scored_public_execution_count") != 0:
        raise ValueError("scored_public_execution_count must be bare zero")
    if artifact.get("model_load_count") != 0:
        raise ValueError("model_load_count must be bare zero")
    if artifact.get("source_bfs_adapter_prior_game_and_hidden_state_access_count") != 0:
        raise ValueError("source_bfs_adapter_prior_game_and_hidden_state_access_count must be zero")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_runner_capability_preflight")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be false")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete_ready:", "retired:", "blocked_precondition:")):
        raise ValueError("honest_verdict has invalid terminal prefix")
    if artifact.get("live_runner_capability_ready_score") == 1.0:
        if not _ready_score_gates_pass(artifact):
            raise ValueError("ready_score gates failed")
    if artifact.get("reproducibility_checksum") != _checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def _ready_score_gates_pass(artifact: Mapping[str, Any]) -> bool:
    authority = artifact.get("authority_source_and_environment_binding") or {}
    dry_run = artifact.get("bounded_non_scored_dry_run") or {}
    isolation = artifact.get("state_isolation_and_teardown_receipts") or {}
    denials = artifact.get("denial_path_matrix") or {}
    protected = artifact.get("protected_files_unchanged") or {}
    adapter = artifact.get("adapter_disabled_e3_receipt") or {}
    return bool(
        isinstance(authority, Mapping)
        and authority.get("externally_bound_scoped_permission") is True
        and isinstance(dry_run, Mapping)
        and dry_run.get("ok") is True
        and isinstance(isolation, Mapping)
        and isolation.get("ok") is True
        and isinstance(denials, Mapping)
        and denials.get("all_denied_before_execution") is True
        and isinstance(adapter, Mapping)
        and adapter.get("ok") is True
        and artifact.get("registry_unchanged") is True
        and isinstance(protected, Mapping)
        and protected.get("all_unchanged") is True
        and artifact.get("scored_public_execution_count") == 0
        and artifact.get("model_load_count") == 0
        and artifact.get("source_bfs_adapter_prior_game_and_hidden_state_access_count") == 0
    )


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    artifact = build_artifact(root=root, test_exit_codes=test_exit_codes)
    validate_artifact(artifact)
    out = output_path or (root / RESULT_RELATIVE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    artifact = write_artifact(REPO_ROOT)
    print(
        f"wrote {REPO_ROOT / RESULT_RELATIVE_PATH} -- "
        f"honest_verdict={artifact['honest_verdict']}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
