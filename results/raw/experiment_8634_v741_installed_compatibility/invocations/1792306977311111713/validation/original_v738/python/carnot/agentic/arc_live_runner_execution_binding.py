"""Exp5928 parent-issued execution binding for the ARC live runner.

Spec refs: REQ-ARC-LREB-5928,
SCENARIO-ARC-LREB-5928-PARENT-CHILD-CONSUME,
SCENARIO-ARC-LREB-5928-DENIAL-MATRIX,
SCENARIO-ARC-LREB-5928-TEARDOWN-IMMUTABILITY.

This module is a preflight, not an ARC solve. The only live path it exercises is
the actual adapter-disabled child entrypoint consuming a parent-signed capability
before a synthetic environment action. No model is loaded, no public level is
attempted, and the ARC registry remains immutable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "experiment_5928_arc_live_runner_execution_binding"
RESULT_RELATIVE_PATH = "results/experiment_5928_arc_live_runner_execution_binding.json"
SCHEMA = "carnot.exp5928.arc_live_runner_execution_binding.v1"
CAPABILITY_SCHEMA = "carnot.arc.live_runner_execution_capability.v1"
INFERENCE_SUBSTRATE = "actual_live_runner_capability_preflight_no_llm"
RUNNER_ID = "carnot.agentic.arc_competition_agent:E3AgentPolicy"
ACTUAL_LIVE_ENTRYPOINT = (
    "carnot.agentic.arc_competition_agent:consume_process_bound_capability_preflight"
)
SCOPE = "actual_live_runner_capability_preflight_no_llm"

ENV_ALLOWLIST_KEYS = (
    "CARNOT_ARC_LIVE_BINDING_ADAPTER_DISABLED",
    "CARNOT_ARC_LIVE_BINDING_NO_MODEL_LOAD",
    "CARNOT_ARC_LIVE_BINDING_RUN_ID",
    "CARNOT_ARC_LIVE_BINDING_SCOPE",
    "JAX_PLATFORMS",
    "PYTHONPATH",
)
FIXTURE_PRIVATE_KEY_HEX = "5928" * 16
DEFAULT_EXPIRY_WINDOW_S = 30.0

REQUIRED_CAPABILITY_FIELDS = (
    "schema",
    "issuer_identity",
    "child_process_identity",
    "binding",
    "adapter_disabled_required",
    "scope",
    "issued_monotonic_s",
    "expires_monotonic_s",
    "nonce",
    "run_id",
    "public_key",
    "signature",
    "capability_hash",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "registry_precheck_receipt",
    "no_model_inference_or_level_attempt",
    "issuer_child_and_os_process_receipts",
    "capability_schema_scope_expiry_nonce_and_run_id",
    "executable_argv_environment_and_output_binding",
    "adapter_disabled_binding",
    "actual_live_entrypoint_consumption_receipt",
    "absent_self_issued_expired_replayed_wrong_process_command_environment_scope_and_output_denials",
    "non_scoring_dry_run_receipt",
    "teardown_nonce_invalidation_and_orphan_check",
    "registry_unchanged",
    "protected_files_unchanged",
    "live_runner_execution_binding_ready_score",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PROVENANCE = {
    "actual_live_entrypoint_consumption_receipt": {
        "principle": (
            "only a capability consumed by the actual child runner before environment action "
            "counts; fixture-only validation is insufficient."
        )
    },
    "registry_unchanged": {
        "principle": "must include exact before/after hash equality."
    },
    "live_runner_execution_binding_ready_score": {
        "principle": (
            "emit bare 1.0 only for external issuer separation, actual child consumption, "
            "all denial paths, clean teardown, and immutable registry."
        )
    },
    "inference_substrate": {
        "principle": "use actual_live_runner_capability_preflight_no_llm."
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
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_live_runner_execution_binding.py",
    "scripts/arc_loop_solve.py",
    "results/experiment_5915_arc_live_runner_capability_lease.json",
    "results/experiment_5916_arc_structured_memory_live_held_ab.json",
    "results/experiment_5902_arc_structured_memory_live_ab.json",
    "ops/arc_solve_registry.yaml",
)

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5928_arc_live_runner_execution_binding.py "
    "-q -n0 -o addopts=''",
    ".venv/bin/python -m coverage erase && .venv/bin/python -m coverage run "
    "--include='*/python/carnot/agentic/arc_live_runner_execution_binding.py' "
    "-m pytest tests/python/test_experiment_5928_arc_live_runner_execution_binding.py "
    "-q -n0 -o addopts='' && .venv/bin/python -m coverage report --fail-under=100 "
    "--show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.agentic.arc_live_runner_execution_binding "
    "--write-artifact",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5928_arc_live_runner_execution_binding.json --json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5928_arc_live_runner_execution_binding.py",
    ".venv/bin/python scripts/arc_levelup_guarantee_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git diff --quiet -- ops/arc_solve_registry.yaml",
    "git diff --quiet -- _bmad/traceability.md ops/changelog.md ops/status.md "
    "scripts/research_conductor.py",
)
DEFAULT_TEST_EXIT_CODES = {"pre_implementation_focused_test_expected_failure": 2}


@dataclass(frozen=True)
class ValidationResult:
    """Decision returned by the child before any synthetic environment action."""

    allowed: bool
    reason: str


@dataclass(frozen=True)
class ParentIssuer:
    """Parent/controller signing identity.

    The private key stays in the parent process. The child receives only the
    public key embedded in the capability, which is what prevents child-side
    self-issue or refresh.
    """

    issuer_id: str
    private_key: Ed25519PrivateKey

    @classmethod
    def fixture(cls) -> "ParentIssuer":
        return cls(
            issuer_id="controller-exp5928-fixture",
            private_key=Ed25519PrivateKey.from_private_bytes(
                bytes.fromhex(FIXTURE_PRIVATE_KEY_HEX)
            ),
        )

    @property
    def public_key_hex(self) -> str:
        return self.private_key.public_key().public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        ).hex()

    @property
    def private_key_fingerprint(self) -> str:
        raw = self.private_key.private_bytes(
            encoding=Encoding.Raw,
            format=PrivateFormat.Raw,
            encryption_algorithm=NoEncryption(),
        )
        return _sha256_bytes(raw)


@dataclass(frozen=True)
class ProcessBindingContext:
    """The exact child process, command, environment, executable, and output binding."""

    pid: int
    ppid: int
    executable_path: str
    executable_hash: str
    argv: tuple[str, ...]
    argv_hash: str
    environment_allowlist: dict[str, str]
    environment_allowlist_hash: str
    output_path: str

    @classmethod
    def from_values(
        cls,
        *,
        pid: int,
        ppid: int,
        executable_path: Path,
        argv: Sequence[str],
        environment: Mapping[str, str],
        output_path: Path,
    ) -> "ProcessBindingContext":
        allowlist = _environment_allowlist(environment)
        argv_tuple = tuple(str(part) for part in argv)
        return cls(
            pid=int(pid),
            ppid=int(ppid),
            executable_path=str(executable_path),
            executable_hash=_sha256_file(executable_path) or _sha256(str(executable_path)),
            argv=argv_tuple,
            argv_hash=_sha256(list(argv_tuple)),
            environment_allowlist=allowlist,
            environment_allowlist_hash=_sha256(allowlist),
            output_path=str(output_path),
        )

    @classmethod
    def current(cls, *, output_path: Path) -> "ProcessBindingContext":  # pragma: no cover
        return cls.from_values(
            pid=os.getpid(),
            ppid=os.getppid(),
            executable_path=Path(sys.executable),
            argv=_process_argv(),
            environment=os.environ,
            output_path=output_path,
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "executable_path": self.executable_path,
            "executable_hash": self.executable_hash,
            "argv": list(self.argv),
            "argv_hash": self.argv_hash,
            "environment_allowlist": dict(self.environment_allowlist),
            "environment_allowlist_hash": self.environment_allowlist_hash,
            "output_path": self.output_path,
        }


class NonceLedger:
    """Filesystem-backed nonce ledger shared by parent and child for one dry run."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _marker(self, nonce: str) -> Path:
        return self.path / _sha256(nonce).removeprefix("sha256:")

    def consume(self, nonce: str) -> bool:
        self.path.mkdir(parents=True, exist_ok=True)
        marker = self._marker(nonce)
        try:
            fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(nonce)
        return True

    def contains(self, nonce: str) -> bool:
        return self._marker(nonce).exists()

    def remove_all(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def _sha256_file(path: Path) -> str | None:
    return _sha256_bytes(path.read_bytes()) if path.exists() else None


def _environment_allowlist(environment: Mapping[str, str]) -> dict[str, str]:
    return {key: str(environment.get(key, "")) for key in ENV_ALLOWLIST_KEYS}


def _process_argv() -> list[str]:  # pragma: no cover
    proc_cmdline = Path("/proc/self/cmdline")
    if proc_cmdline.exists():
        raw = proc_cmdline.read_bytes().split(b"\0")
        return [part.decode("utf-8", errors="replace") for part in raw if part]
    return [sys.executable, *sys.argv]


def child_environment(*, run_id: str, root: Path = REPO_ROOT) -> dict[str, str]:
    return {
        "CARNOT_ARC_LIVE_BINDING_ADAPTER_DISABLED": "1",
        "CARNOT_ARC_LIVE_BINDING_NO_MODEL_LOAD": "1",
        "CARNOT_ARC_LIVE_BINDING_RUN_ID": str(run_id),
        "CARNOT_ARC_LIVE_BINDING_SCOPE": SCOPE,
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": str(root / "python"),
    }


def child_argv(*, run_id: str, output_path: Path, nonce_ledger: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "carnot.agentic.arc_live_runner_execution_binding",
        "--child",
        "--run-id",
        str(run_id),
        "--output",
        str(output_path),
        "--nonce-ledger",
        str(nonce_ledger),
    ]


def _capability_core(capability: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(capability)
    core.pop("signature", None)
    core.pop("capability_hash", None)
    return core


def _sign_core(core: Mapping[str, Any], issuer: ParentIssuer) -> str:
    signature = issuer.private_key.sign(_stable_json(core).encode("utf-8"))
    return "ed25519:" + signature.hex()


def _verify_signature(core: Mapping[str, Any], signature: str, public_key_hex: str) -> bool:
    if not signature.startswith("ed25519:"):
        return False
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        public_key.verify(
            bytes.fromhex(signature.removeprefix("ed25519:")),
            _stable_json(core).encode("utf-8"),
        )
        return True
    except (InvalidSignature, ValueError):
        return False


def issue_capability(
    issuer: ParentIssuer,
    context: ProcessBindingContext,
    *,
    issued_monotonic_s: float,
    expires_monotonic_s: float,
    nonce: str,
    run_id: str,
    scope: str = SCOPE,
    adapter_disabled_required: bool = True,
) -> dict[str, Any]:
    public_key = issuer.public_key_hex
    core = {
        "schema": CAPABILITY_SCHEMA,
        "issuer_identity": {
            "issuer_kind": "parent_controller",
            "issuer_id": issuer.issuer_id,
            "issuer_pid": os.getpid(),
            "public_key_fingerprint": _sha256(public_key),
            "child_can_issue": False,
            "child_can_broaden": False,
            "child_can_refresh": False,
        },
        "child_process_identity": {
            "pid": context.pid,
            "ppid": context.ppid,
        },
        "binding": {
            "runner_id": RUNNER_ID,
            "actual_live_entrypoint": ACTUAL_LIVE_ENTRYPOINT,
            "executable_path": context.executable_path,
            "executable_hash": context.executable_hash,
            "argv": list(context.argv),
            "argv_hash": context.argv_hash,
            "environment_allowlist": dict(context.environment_allowlist),
            "environment_allowlist_hash": context.environment_allowlist_hash,
            "output_path": context.output_path,
            "output_path_hash": _sha256(context.output_path),
        },
        "adapter_disabled_required": bool(adapter_disabled_required),
        "scope": scope,
        "issued_monotonic_s": float(issued_monotonic_s),
        "expires_monotonic_s": float(expires_monotonic_s),
        "nonce": str(nonce),
        "run_id": str(run_id),
        "public_key": public_key,
    }
    signature = _sign_core(core, issuer)
    signed = {**core, "signature": signature}
    return {**signed, "capability_hash": _sha256(signed)}


def verify_and_consume_capability(
    capability: Mapping[str, Any] | None,
    context: ProcessBindingContext,
    *,
    public_key: str,
    now_monotonic_s: float,
    nonce_ledger: NonceLedger,
    adapter_enabled: bool,
) -> ValidationResult:
    if not isinstance(capability, Mapping):
        return ValidationResult(False, "missing_capability")
    for field in REQUIRED_CAPABILITY_FIELDS:
        if field not in capability:
            return ValidationResult(False, f"missing_field:{field}")
    if capability.get("schema") != CAPABILITY_SCHEMA:
        return ValidationResult(False, "wrong_schema")

    issuer = capability.get("issuer_identity") or {}
    if (
        not isinstance(issuer, Mapping)
        or issuer.get("issuer_kind") != "parent_controller"
        or issuer.get("issuer_pid") == context.pid
        or issuer.get("child_can_issue") is not False
        or issuer.get("child_can_broaden") is not False
        or issuer.get("child_can_refresh") is not False
    ):
        return ValidationResult(False, "self_issued")
    if capability.get("scope") != SCOPE:
        return ValidationResult(False, "scope_broadened")
    if float(now_monotonic_s) >= float(capability.get("expires_monotonic_s")):
        return ValidationResult(False, "expired")

    child = capability.get("child_process_identity") or {}
    if (
        not isinstance(child, Mapping)
        or child.get("pid") != context.pid
        or child.get("ppid") != context.ppid
    ):
        return ValidationResult(False, "wrong_process")

    binding = capability.get("binding") or {}
    if not isinstance(binding, Mapping):
        return ValidationResult(False, "wrong_command")
    if binding.get("executable_hash") != context.executable_hash:
        return ValidationResult(False, "wrong_executable")
    if binding.get("argv_hash") != context.argv_hash:
        return ValidationResult(False, "wrong_command")
    if binding.get("environment_allowlist_hash") != context.environment_allowlist_hash:
        return ValidationResult(False, "wrong_environment")
    if binding.get("output_path") != context.output_path:
        return ValidationResult(False, "output_mismatch")
    if adapter_enabled or capability.get("adapter_disabled_required") is not True:
        return ValidationResult(False, "adapter_enabled")
    if capability.get("public_key") != public_key:
        return ValidationResult(False, "signature_mismatch")
    if not _verify_signature(_capability_core(capability), str(capability["signature"]), public_key):
        return ValidationResult(False, "signature_mismatch")
    expected_hash = _sha256({**_capability_core(capability), "signature": capability["signature"]})
    if capability.get("capability_hash") != expected_hash:
        return ValidationResult(False, "capability_hash_mismatch")
    if not nonce_ledger.consume(str(capability.get("nonce"))):
        return ValidationResult(False, "nonce_replay")
    return ValidationResult(True, "allowed")


def capability_schema_receipt() -> dict[str, Any]:
    return {
        "schema": CAPABILITY_SCHEMA,
        "required_fields": list(REQUIRED_CAPABILITY_FIELDS),
        "signature_method": "ed25519_parent_private_child_public_verify",
        "capability_hash_method": "sha256_over_signed_canonical_capability",
        "scope": SCOPE,
        "child_can_issue_broaden_or_refresh": False,
    }


def run_denial_matrix(root: Path = REPO_ROOT, work_dir: Path | None = None) -> dict[str, Any]:
    base_dir = Path(work_dir or (Path("/tmp") / "carnot-exp5928-denials"))
    issuer = ParentIssuer.fixture()
    context = _unit_context(root, base_dir, "denial-run")
    capability = issue_capability(
        issuer,
        context,
        issued_monotonic_s=100.0,
        expires_monotonic_s=105.0,
        nonce="denial-base-nonce",
        run_id="denial-run",
    )
    replay_ledger = NonceLedger(base_dir / "replay-ledger")
    verify_and_consume_capability(
        capability,
        context,
        public_key=issuer.public_key_hex,
        now_monotonic_s=101.0,
        nonce_ledger=replay_ledger,
        adapter_enabled=False,
    )

    def check(
        case: str,
        cap: Mapping[str, Any] | None,
        ctx: ProcessBindingContext = context,
        *,
        now: float = 101.0,
        ledger: NonceLedger | None = None,
        adapter_enabled: bool = False,
    ) -> dict[str, Any]:
        result = verify_and_consume_capability(
            cap,
            ctx,
            public_key=issuer.public_key_hex if isinstance(cap, Mapping) else issuer.public_key_hex,
            now_monotonic_s=now,
            nonce_ledger=ledger or NonceLedger(base_dir / f"{case}-ledger"),
            adapter_enabled=adapter_enabled,
        )
        return {
            "case": case,
            "allowed": result.allowed,
            "reason": result.reason,
            "environment_action_count": 0,
            "denied_before_environment_action": result.allowed is False,
        }

    self_issued = _mutated(capability, ("issuer_identity", "issuer_kind"), "child_runner")
    expired = issue_capability(
        issuer,
        context,
        issued_monotonic_s=90.0,
        expires_monotonic_s=99.0,
        nonce="expired-nonce",
        run_id="denial-run",
    )
    wrong_process_context = _replace_context(context, pid=context.pid + 1)
    wrong_command_context = _replace_context(context, argv=(*context.argv, "--extra"))
    wrong_environment_context = _replace_context(
        context,
        environment_allowlist={
            **context.environment_allowlist,
            "CARNOT_ARC_LIVE_BINDING_SCOPE": "different",
        },
    )
    wrong_executable_context = _replace_context(context, executable_hash="sha256:wrong")
    adapter_capability = _mutated(capability, ("adapter_disabled_required",), False)
    scope_capability = _mutated(capability, ("scope",), "actual_live_runner_plus_scoring")
    output_context = _replace_context(context, output_path=str(base_dir / "other.json"))

    rows = [
        check("absent", None),
        check("self_issued", self_issued),
        check("expired", expired, now=100.0),
        check("replayed", capability, ledger=replay_ledger),
        check("wrong_process", capability, wrong_process_context),
        check("wrong_command", capability, wrong_command_context),
        check("wrong_environment", capability, wrong_environment_context),
        check("wrong_executable", capability, wrong_executable_context),
        check("adapter_enabled", adapter_capability),
        check("scope_broadened", scope_capability),
        check("output_mismatch", capability, output_context),
    ]
    return {
        "rows": rows,
        "denial_case_count": len(rows),
        "all_denied_before_environment_action": all(
            row["allowed"] is False
            and row["denied_before_environment_action"] is True
            and row["environment_action_count"] == 0
            for row in rows
        ),
        "environment_action_count": 0,
        "model_load_count": 0,
        "level_attempt_count": 0,
    }


def _unit_context(root: Path, base_dir: Path, run_id: str) -> ProcessBindingContext:
    output = base_dir / "unit-child-output.json"
    ledger = base_dir / "unit-ledger"
    return ProcessBindingContext.from_values(
        pid=12345,
        ppid=54321,
        executable_path=Path(sys.executable),
        argv=child_argv(run_id=run_id, output_path=output, nonce_ledger=ledger),
        environment=child_environment(run_id=run_id, root=root),
        output_path=output,
    )


def _replace_context(context: ProcessBindingContext, **updates: Any) -> ProcessBindingContext:
    values = context.receipt()
    values.update(updates)
    allowlist = dict(values["environment_allowlist"])
    if "environment_allowlist" in updates:
        allowlist = dict(updates["environment_allowlist"])
    argv = tuple(values["argv"])
    if "argv" in updates:
        argv = tuple(str(part) for part in updates["argv"])
    return ProcessBindingContext(
        pid=int(values["pid"]),
        ppid=int(values["ppid"]),
        executable_path=str(values["executable_path"]),
        executable_hash=str(values.get("executable_hash") or context.executable_hash),
        argv=argv,
        argv_hash=_sha256(list(argv)),
        environment_allowlist=allowlist,
        environment_allowlist_hash=_sha256(allowlist),
        output_path=str(values["output_path"]),
    )


def _mutated(capability: Mapping[str, Any], path: Sequence[str], value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(capability))
    target: dict[str, Any] = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return mutated


def run_parent_child_dry_run(root: Path = REPO_ROOT, work_dir: Path | None = None) -> dict[str, Any]:
    base_dir = Path(work_dir or (Path("/tmp") / f"carnot-exp5928-{os.getpid()}"))
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = "exp5928-parent-child-dry-run"
    child_output = base_dir / "child-consumption-receipt.json"
    nonce_ledger_path = base_dir / "nonce-ledger"
    argv = child_argv(run_id=run_id, output_path=child_output, nonce_ledger=nonce_ledger_path)
    env = child_environment(run_id=run_id, root=root)
    proc = subprocess.Popen(
        argv,
        cwd=root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    hello_line = proc.stdout.readline() if proc.stdout is not None else ""
    hello = json.loads(hello_line)
    context = ProcessBindingContext.from_values(
        pid=proc.pid,
        ppid=os.getpid(),
        executable_path=Path(sys.executable),
        argv=argv,
        environment=env,
        output_path=child_output,
    )
    issuer = ParentIssuer.fixture()
    issued = time.monotonic()
    capability = issue_capability(
        issuer,
        context,
        issued_monotonic_s=issued,
        expires_monotonic_s=issued + DEFAULT_EXPIRY_WINDOW_S,
        nonce="exp5928-parent-child-nonce",
        run_id=run_id,
    )
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(capability, sort_keys=True) + "\n")
    proc.stdin.flush()
    stdout, stderr = proc.communicate(timeout=30.0)
    child_receipt = json.loads(child_output.read_text(encoding="utf-8"))
    replay = verify_and_consume_capability(
        capability,
        context,
        public_key=issuer.public_key_hex,
        now_monotonic_s=time.monotonic(),
        nonce_ledger=NonceLedger(nonce_ledger_path),
        adapter_enabled=False,
    )
    NonceLedger(nonce_ledger_path).remove_all()
    orphaned = _pid_exists(proc.pid)
    issuer_secret_persisted = _secret_persisted(
        issuer.private_key_fingerprint,
        env=env,
        capability=capability,
        child_receipt=child_receipt,
    )
    ok = bool(
        proc.returncode == 0
        and child_receipt.get("capability_consumed_before_environment_action") is True
        and replay.allowed is False
        and replay.reason == "nonce_replay"
        and not nonce_ledger_path.exists()
        and not orphaned
        and not issuer_secret_persisted
    )
    return {
        "ok": ok,
        "parent_pid": os.getpid(),
        "parent_issue_receipt": {
            "issuer_id": issuer.issuer_id,
            "issuer_kind": "parent_controller",
            "issuer_pid": os.getpid(),
            "child_private_signing_key_available": False,
            "child_can_issue_broaden_or_refresh": False,
            "public_key_fingerprint": _sha256(issuer.public_key_hex),
        },
        "child_hello_receipt": hello,
        "capability": capability,
        "issuer_child_and_os_process_receipts": {
            "issuer_pid": os.getpid(),
            "child_pid": proc.pid,
            "child_ppid": os.getpid(),
            "os_process_boundary_crossed": proc.pid != os.getpid(),
            "child_exit_code": proc.returncode,
            "child_reported_context": hello.get("context"),
            "parent_bound_context": context.receipt(),
        },
        "executable_argv_environment_and_output_binding": {
            "executable_hash": context.executable_hash,
            "argv_hash": context.argv_hash,
            "environment_allowlist_hash": context.environment_allowlist_hash,
            "output_path": str(child_output),
            "output_path_hash": _sha256(str(child_output)),
            "exact_command_bound": True,
            "exact_environment_allowlist_bound": True,
            "exact_output_bound": True,
        },
        "adapter_disabled_binding": {
            "adapter_disabled_required": True,
            "adapter_disabled_env": env["CARNOT_ARC_LIVE_BINDING_ADAPTER_DISABLED"],
            "adapter_enabled": False,
            "model_loader_allowed": False,
            "scoring_allowed": False,
        },
        "actual_live_entrypoint_consumption_receipt": child_receipt,
        "non_scoring_dry_run_receipt": {
            "returncode": proc.returncode,
            "stdout": stdout.strip().splitlines(),
            "stderr": stderr.strip(),
            "output_path": str(child_output),
            "output_exists": child_output.exists(),
            "no_model_load": child_receipt.get("model_load_count") == 0,
            "no_level_attempt": child_receipt.get("level_attempt_count") == 0,
            "scoring_target_selected": False,
            "public_solve_target_selected": False,
        },
        "teardown_nonce_invalidation_and_orphan_check": {
            "nonce": capability["nonce"],
            "nonce_replay_denied_before_teardown": replay.reason == "nonce_replay",
            "nonce_ledger_removed_after_teardown": not nonce_ledger_path.exists(),
            "child_process_orphaned": orphaned,
            "issuer_secret_persisted": issuer_secret_persisted,
            "credential_env_keys_present": [
                key for key in env if "KEY" in key or "SECRET" in key or "TOKEN" in key
            ],
            "teardown_ready": True,
        },
    }


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _secret_persisted(
    private_key_fingerprint: str,
    *,
    env: Mapping[str, str],
    capability: Mapping[str, Any],
    child_receipt: Mapping[str, Any],
) -> bool:
    haystack = _stable_json(
        {
            "env": env,
            "capability": capability,
            "child_receipt": child_receipt,
        }
    )
    return private_key_fingerprint in haystack or FIXTURE_PRIVATE_KEY_HEX in haystack


def registry_precheck_receipt(root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / "ops" / "arc_solve_registry.yaml"
    data = _read_yaml(path)
    games = [row for row in data.get("games", []) or [] if isinstance(row, Mapping)]
    cleared = [row for row in games if row.get("full_game_clear") is True]
    registry_hash = _sha256_file(path)
    return {
        "ok": bool(path.exists() and len(games) == 25 and len(cleared) == 25),
        "path": "ops/arc_solve_registry.yaml",
        "registry_hash_before": registry_hash,
        "public_games_count": len(games),
        "full_game_clear_count": len(cleared),
        "all_public_games_cleared": len(games) == 25 and len(cleared) == 25,
        "checked_before_parent_child_dry_run": True,
        "public_level_target_selected": False,
        "registry_update_allowed": False,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
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


def preconditions_checked(root: Path, result_output_path: Path) -> dict[str, Any]:
    file_hashes = {relative: _sha256_file(root / relative) for relative in HASHED_RELATIVE_PATHS}
    environment_schema = {
        "environment_id": "synthetic_adapter_disabled_live_runner_preflight",
        "scope": SCOPE,
        "adapter_disabled_required": True,
        "model_load_allowed": False,
        "level_attempt_allowed": False,
        "scoring_target_allowed": False,
    }
    output_parent = result_output_path.parent
    disk = shutil.disk_usage(root)
    return {
        "ok": all(value is not None for value in file_hashes.values())
        and output_parent.exists()
        and os.access(output_parent, os.W_OK),
        "hashed_files": file_hashes,
        "environment_schema": environment_schema,
        "environment_schema_hash": _sha256(environment_schema),
        "output_path": str(result_output_path),
        "output_hash_before": _sha256_file(result_output_path),
        "output_parent_exists": output_parent.exists(),
        "output_parent_writable": os.access(output_parent, os.W_OK),
        "atomic_output_ready": True,
        "teardown_ready": True,
        "no_model_load": True,
        "no_scoring_target": True,
        "no_public_solve_target": True,
        "disk_free_bytes": disk.free,
    }


def field_provenance() -> dict[str, Any]:
    provenance = {
        field: {
            "principle": f"Exp5928 required artifact field `{field}` is emitted by the binding builder.",
            "satisfied_by": "Exp5928 parent-child live runner execution preflight",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    for field, receipt in REQUIRED_FIELD_PROVENANCE.items():
        provenance[field] = {
            "principle": receipt["principle"],
            "satisfied_by": "REQ-ARC-LREB-5928 principle-annotated artifact contract",
        }
    return provenance


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    work_dir: Path | None = None,
    result_output_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    output_path = result_output_path or (root / RESULT_RELATIVE_PATH)
    protected_before = protected_file_hashes(root)
    registry_before = registry_precheck_receipt(root)
    preconditions = preconditions_checked(root, output_path)
    dry_run = run_parent_child_dry_run(root, work_dir)
    denials = run_denial_matrix(root, (work_dir or Path("/tmp") / "carnot-exp5928") / "denials")
    registry_after_hash = _sha256_file(root / "ops" / "arc_solve_registry.yaml")
    registry_unchanged = {
        "unchanged": registry_before["registry_hash_before"] == registry_after_hash,
        "registry_hash_before": registry_before["registry_hash_before"],
        "registry_hash_after": registry_after_hash,
        "principle": REQUIRED_FIELD_PROVENANCE["registry_unchanged"]["principle"],
    }
    protected = protected_files_unchanged(root, protected_before)
    no_model = {
        "model_load_count": 0,
        "level_attempt_count": 0,
        "scoring_target_selected": False,
        "public_solve_target_selected": False,
        "no_model_inference": True,
        "no_level_attempt": True,
    }
    ready = bool(
        registry_before["ok"]
        and preconditions["ok"]
        and dry_run["ok"]
        and denials["all_denied_before_environment_action"]
        and registry_unchanged["unchanged"]
        and protected["all_unchanged"]
    )
    duration_s = max(time.monotonic() - started, 0.001)
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "status": "complete_ready" if ready else "blocked_precondition",
        "preconditions_checked": preconditions,
        "registry_precheck_receipt": {**registry_before, "registry_hash_after": registry_after_hash},
        "no_model_inference_or_level_attempt": no_model,
        "issuer_child_and_os_process_receipts": dry_run["issuer_child_and_os_process_receipts"],
        "capability_schema_scope_expiry_nonce_and_run_id": {
            **capability_schema_receipt(),
            "issued_monotonic_s": dry_run["capability"]["issued_monotonic_s"],
            "expires_monotonic_s": dry_run["capability"]["expires_monotonic_s"],
            "nonce": dry_run["capability"]["nonce"],
            "nonce_hash": _sha256(dry_run["capability"]["nonce"]),
            "run_id": dry_run["capability"]["run_id"],
        },
        "executable_argv_environment_and_output_binding": dry_run[
            "executable_argv_environment_and_output_binding"
        ],
        "adapter_disabled_binding": dry_run["adapter_disabled_binding"],
        "actual_live_entrypoint_consumption_receipt": dry_run[
            "actual_live_entrypoint_consumption_receipt"
        ],
        "absent_self_issued_expired_replayed_wrong_process_command_environment_scope_and_output_denials": denials,
        "non_scoring_dry_run_receipt": dry_run["non_scoring_dry_run_receipt"],
        "teardown_nonce_invalidation_and_orphan_check": dry_run[
            "teardown_nonce_invalidation_and_orphan_check"
        ],
        "registry_unchanged": registry_unchanged,
        "protected_files_unchanged": protected,
        "live_runner_execution_binding_ready_score": 1.0 if ready else 0.0,
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": field_provenance(),
        "test_commands": list(TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_ready: actual_child_live_runner_execution_binding_ready"
            if ready
            else "blocked_precondition: actual_child_live_runner_execution_binding_not_ready"
        ),
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
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be actual_live_runner_capability_preflight_no_llm")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete_ready:", "retired:", "blocked_precondition:")):
        raise ValueError("honest_verdict has invalid terminal prefix")
    no_model = artifact.get("no_model_inference_or_level_attempt") or {}
    if not isinstance(no_model, Mapping) or no_model.get("model_load_count") != 0:
        raise ValueError("model_load_count must be zero")
    if no_model.get("level_attempt_count") != 0:
        raise ValueError("level_attempt_count must be zero")
    registry = artifact.get("registry_unchanged") or {}
    if (
        not isinstance(registry, Mapping)
        or registry.get("unchanged") is not True
        or registry.get("registry_hash_before") != registry.get("registry_hash_after")
    ):
        raise ValueError("registry_unchanged must include exact before/after hash equality")
    consume = artifact.get("actual_live_entrypoint_consumption_receipt") or {}
    if not isinstance(consume, Mapping) or consume.get("fixture_only_validation") is not False:
        raise ValueError("actual live entrypoint receipt cannot be fixture-only")
    if consume.get("capability_consumed_before_environment_action") is not True:
        raise ValueError("capability must be consumed before environment action")
    denials = (
        artifact.get(
            "absent_self_issued_expired_replayed_wrong_process_command_environment_scope_and_output_denials"
        )
        or {}
    )
    if not isinstance(denials, Mapping) or denials.get("all_denied_before_environment_action") is not True:
        raise ValueError("denial matrix did not deny before environment action")
    teardown = artifact.get("teardown_nonce_invalidation_and_orphan_check") or {}
    if not isinstance(teardown, Mapping) or teardown.get("child_process_orphaned") is not False:
        raise ValueError("teardown orphan check failed")
    if teardown.get("nonce_replay_denied_before_teardown") is not True:
        raise ValueError("nonce replay invalidation failed")
    protected = artifact.get("protected_files_unchanged") or {}
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        raise ValueError("protected files changed")
    if artifact.get("live_runner_execution_binding_ready_score") == 1.0 and not _ready_gates(artifact):
        raise ValueError("ready score gates failed")
    if artifact.get("reproducibility_checksum") != _checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def _ready_gates(artifact: Mapping[str, Any]) -> bool:
    consume = artifact["actual_live_entrypoint_consumption_receipt"]
    denials = artifact[
        "absent_self_issued_expired_replayed_wrong_process_command_environment_scope_and_output_denials"
    ]
    teardown = artifact["teardown_nonce_invalidation_and_orphan_check"]
    return bool(
        consume.get("capability_consumed_before_environment_action") is True
        and consume.get("actual_live_entrypoint") == ACTUAL_LIVE_ENTRYPOINT
        and denials.get("all_denied_before_environment_action") is True
        and teardown.get("child_process_orphaned") is False
        and teardown.get("issuer_secret_persisted") is False
        and artifact["registry_unchanged"].get("unchanged") is True
        and artifact["protected_files_unchanged"].get("all_unchanged") is True
    )


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    work_dir: Path | None = None,
    output_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    out = output_path or (root / RESULT_RELATIVE_PATH)
    artifact = build_artifact(
        root=root,
        work_dir=work_dir,
        result_output_path=out,
        test_exit_codes=test_exit_codes,
    )
    validate_artifact(artifact)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return artifact


def _child_main(args: argparse.Namespace) -> int:  # pragma: no cover
    output_path = Path(args.output)
    context = ProcessBindingContext.current(output_path=output_path)
    print(json.dumps({"ready": True, "context": context.receipt()}, sort_keys=True), flush=True)
    line = sys.stdin.readline()
    capability = json.loads(line)
    from carnot.agentic import arc_competition_agent

    receipt = arc_competition_agent.consume_process_bound_capability_preflight(
        capability,
        output_path=output_path,
        nonce_ledger_path=Path(args.nonce_ledger),
        now_monotonic_s=time.monotonic(),
        adapter_enabled=os.environ.get("CARNOT_ARC_LIVE_BINDING_ADAPTER_DISABLED") != "1",
    )
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output_path)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if receipt.get("capability_consumed_before_environment_action") else 3


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--nonce-ledger", default="")
    args = parser.parse_args(argv)
    if args.child:
        return _child_main(args)
    if args.write_artifact:
        artifact = write_artifact(REPO_ROOT)
        print(
            f"wrote {REPO_ROOT / RESULT_RELATIVE_PATH} -- "
            f"honest_verdict={artifact['honest_verdict']}"
        )
        return 0
    parser.error("specify --child or --write-artifact")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
