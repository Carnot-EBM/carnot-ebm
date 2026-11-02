"""Run the independent REQ-ARC-7031 model identity mutation audit.

The parent process checks the sealed upstream evidence. It then starts an
isolated Python process that creates new files and calls the production bridge.
No model or GPU starts during this audit.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7031
SCHEMA = "carnot.exp7031.arc_model_identity_cold_audit.v1"
RANDOM_SEED = 7_031_202_609_05
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
HF_ID = "cold-auditor/independent-model-GGUF"
REVISION = "7031coldrevision"
FILENAME = "independent-cold-audit.gguf"
UPSTREAM_RELATIVE_PATH = Path("results/experiment_7030_arc_gguf_model_identity_bridge.json")
RESULT_RELATIVE_PATH = Path("results/experiment_7031_arc_model_identity_cold_audit.json")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "upstream_gate_rows",
    "rows",
    "positive_identity_rows",
    "alias_mutation_rows",
    "hash_mutation_rows",
    "hub_revision_mutation_rows",
    "path_type_rows",
    "stale_server_rows",
    "legacy_regression_rows",
    "fresh_process_rows",
    "consumer_reachability_rows",
    "command_receipt_rows",
    "arc_model_identity_audit_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why the audit claim needs that evidence.",
    "preconditions_checked": "Exact gates stop a stale upstream result before mutation work starts.",
    "inference_substrate": "The substrate states that this audit aggregates deterministic evidence.",
    "duration_s": "Measured wall time makes the cost and termination of the audit visible.",
    "source_artifact_hashes": "File hashes bind the audit to the exact upstream evidence and code.",
    "cited_upstream_artifacts": "Upstream citations identify the readiness claim under independent test.",
    "upstream_gate_rows": "Gate rows preserve the exact upstream score and validity decisions.",
    "rows": "Summary rows make every readiness factor independently countable.",
    "positive_identity_rows": "Positive rows prove valid requested and observed identities still join.",
    "alias_mutation_rows": "Alias rows test names and paths that can confuse identity without changing bytes.",
    "hash_mutation_rows": "Hash rows prove content bytes, rather than size, control acceptance.",
    "hub_revision_mutation_rows": "Hub rows bind the requested file to one repository revision.",
    "path_type_rows": "Path rows reject broken, non-file, and multiply linked filesystem identities.",
    "stale_server_rows": "Stale rows prevent an earlier server model from receiving current credit.",
    "legacy_regression_rows": "Regression rows preserve valid direct files and complete legacy records.",
    "fresh_process_rows": "Process rows prove the audit used an isolated interpreter and private files.",
    "consumer_reachability_rows": "Reachability rows prove the live consumer uses the audited shared code.",
    "command_receipt_rows": "Command receipts expose the isolated imports and worker termination status.",
    "arc_model_identity_audit_ready_score": "One requires every positive, negative, source, and reachability gate.",
    "random_seed": "A fixed seed labels the deterministic audit configuration.",
    "reproducibility_checksum": "A stable digest detects any later change to the complete artifact.",
    "gate_check_summary": "The first exact failure makes a blocked or disqualified result actionable.",
    "verifier_is_oracle": "False states that identity validation does not decide ARC correctness.",
    "verdict_class": "A closed class gives the terminal result one machine-readable meaning.",
    "honest_verdict": "A class-consistent prefix prevents failed evidence from reading as success.",
}

CURRENT_SOURCE_PATHS = (
    Path("openspec/capabilities/arc-agi/spec.md"),
    Path("python/carnot/agentic/arc_belief_shadow_live_trace.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/experiment_7031_arc_model_identity_cold_audit.py"),
    Path("scripts/experiments/experiment_7031_arc_model_identity_cold_audit.py"),
    Path("tests/python/test_experiment_7031_arc_model_identity_cold_audit.py"),
)

MUTATION_CASES = {
    "alias_mutation_rows": frozenset({"requested_filename", "observed_path"}),
    "hash_mutation_rows": frozenset({"content_hash", "same_size_different_bytes"}),
    "hub_revision_mutation_rows": frozenset({"repository", "revision"}),
    "path_type_rows": frozenset({"broken_link", "path_type", "ambiguous_hard_link"}),
    "stale_server_rows": frozenset({"stale_server_identity"}),
}


def sha256_file(path: str | Path) -> str:
    """Hash one file in bounded chunks so large cited sources remain safe."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash stable JSON bytes so dictionary order cannot change the result."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every artifact field except its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def gate_row(check: str, expected: Any, observed: Any) -> JsonDict:
    """Keep both sides of one gate so a failure has an exact diagnosis."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "terminal": True,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failure while retaining the complete ordered gate list."""

    copied = [dict(row) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": True if failed is None else failed.get("expected_value"),
        "observed_value": True if failed is None else failed.get("observed_value"),
        "checks": copied,
    }


def _load_json(path: Path) -> JsonDict:
    """Read one JSON object and return explicit absence for malformed input."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _private_environment() -> dict[str, str]:
    """Give the worker only interpreter, locale, and deterministic hash settings."""

    return {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3"),
        "TPU_SKIP_MDS_QUERY": os.environ.get("TPU_SKIP_MDS_QUERY", "1"),
    }


def _restrict_process() -> None:  # pragma: no cover - runs only inside the child
    """Disable core dumps and make every new worker file private by default."""

    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _isolated_command(
    argv: Sequence[str], *, work_dir: Path, timeout_s: float = 60.0
) -> tuple[subprocess.CompletedProcess[str], JsonDict]:
    """Run one isolated interpreter command and preserve its exact receipt."""

    completed = subprocess.run(
        list(argv),
        cwd=work_dir,
        env=_private_environment(),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        start_new_session=True,
        check=False,
    )
    receipt = {
        "argv": list(argv),
        "cwd": str(work_dir.resolve()),
        "environment_keys": sorted(_private_environment()),
        "returncode": completed.returncode,
        "stdout_sha256": sha256_json(completed.stdout),
        "stderr_sha256": sha256_json(completed.stderr),
        "terminal": True,
    }
    return completed, receipt


def collect_preconditions(
    root: Path,
    output_path: Path,
    work_dir: Path,
    upstream_path: Path | None = None,
) -> tuple[list[JsonDict], JsonDict, JsonDict, list[JsonDict]]:
    """Check the sealed upstream artifact, its sources, imports, and writable paths."""

    upstream_path = upstream_path or root / UPSTREAM_RELATIVE_PATH
    upstream = _load_json(upstream_path)
    checks = [gate_row(UPSTREAM_RELATIVE_PATH.as_posix(), True, bool(upstream))]
    score_row = gate_row(
        "exp7030_ready_score", 1, upstream.get("arc_model_identity_bridge_ready_score")
    )
    checks.append(score_row)

    try:
        upstream_module = importlib.import_module(
            "carnot.experiment_7030_arc_gguf_model_identity_bridge"
        )
        upstream_errors = upstream_module.validate_artifact(upstream)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        upstream_errors = [f"{type(exc).__name__}: {exc}"]
    validity_row = gate_row("exp7030_artifact_valid", [], upstream_errors)
    checks.append(validity_row)

    claimed_hashes = upstream.get("source_artifact_hashes")
    claimed_hashes = dict(claimed_hashes) if isinstance(claimed_hashes, Mapping) else {}
    source_hashes: JsonDict = {}
    if upstream_path.is_file():
        source_hashes[UPSTREAM_RELATIVE_PATH.as_posix()] = sha256_file(upstream_path)
    for relative, expected in claimed_hashes.items():
        path = root / str(relative)
        observed = sha256_file(path) if path.is_file() else None
        if observed is not None:
            source_hashes[str(relative)] = observed
        checks.append(gate_row(f"exp7030_source_hash:{relative}", expected, observed))
    checks.append(
        gate_row(
            "exp7030_artifact_hash_recomputed",
            True,
            str(source_hashes.get(UPSTREAM_RELATIVE_PATH.as_posix(), "")).startswith("sha256:"),
        )
    )

    for relative in CURRENT_SOURCE_PATHS:
        path = root / relative
        exists = path.is_file()
        checks.append(gate_row(f"current_source:{relative.as_posix()}", True, exists))
        if exists:
            source_hashes[relative.as_posix()] = sha256_file(path)

    import_argv = (
        sys.executable,
        "-I",
        "-c",
        (
            "from carnot.agentic.arc_eval_provenance import "
            "build_arc_model_identity_receipt,validate_arc_eval_provenance;print('shared-import-ok')"
        ),
    )
    completed, import_receipt = _isolated_command(import_argv, work_dir=work_dir)
    import_observed = completed.returncode == 0 and completed.stdout.strip() == "shared-import-ok"
    checks.append(gate_row("isolated_shared_code_importable", True, import_observed))

    for label, path in (
        ("temporary_path_writable", work_dir),
        ("artifact_path_writable", output_path.parent),
    ):
        checks.append(gate_row(label, True, path.is_dir() and os.access(path, os.W_OK)))

    hash_rows = [row for row in checks if str(row["check"]).startswith("exp7030_source_hash:")]
    upstream_gate_rows = [
        score_row,
        validity_row,
        gate_row(
            "exp7030_source_hashes_match",
            True,
            bool(hash_rows) and all(row["passed"] for row in hash_rows),
        ),
    ]
    evidence = {
        "upstream": upstream,
        "upstream_gate_rows": upstream_gate_rows,
        "artifact_hash": source_hashes.get(UPSTREAM_RELATIVE_PATH.as_posix()),
    }
    return checks, evidence, source_hashes, [import_receipt]


def _snapshot_fixture(directory: Path, payload: bytes | None = None) -> tuple[JsonDict, Path, Path]:
    """Create a new snapshot symlink and extensionless content-addressed blob."""

    value = payload if payload is not None else b"independent Exp7031 GGUF identity bytes\n"
    digest = hashlib.sha256(value).hexdigest()
    model_root = directory / "hub" / "models--cold-auditor--independent-model-GGUF"
    blob = model_root / "blobs" / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(value)
    requested = model_root / "snapshots" / REVISION / FILENAME
    requested.parent.mkdir(parents=True, exist_ok=True)
    requested.symlink_to(Path("../../blobs") / digest)
    spec = {
        "model_path": str(requested.absolute()),
        "model_filename": FILENAME,
        "hf_id": HF_ID,
        "revision": REVISION,
        "model_file_hash": "sha256:" + digest,
    }
    return spec, blob.absolute(), requested.absolute()


def _direct_fixture(directory: Path) -> tuple[JsonDict, Path]:
    """Create a single-link regular GGUF inside the selected snapshot."""

    requested = (
        directory
        / "hub"
        / "models--cold-auditor--independent-model-GGUF"
        / "snapshots"
        / REVISION
        / FILENAME
    )
    requested.parent.mkdir(parents=True, exist_ok=True)
    requested.write_bytes(b"independent direct GGUF regression bytes\n")
    spec = {
        "model_path": str(requested.absolute()),
        "model_filename": FILENAME,
        "hf_id": HF_ID,
        "revision": REVISION,
        "model_file_hash": sha256_file(requested),
    }
    return spec, requested.absolute()


def _legacy_input(provenance: Any, **changes: Any) -> Any:
    """Build one complete prior-schema input without current identity fields."""

    source = provenance.ArcEvalProvenanceInput(
        inference_substrate=provenance.LIVE_LLM_INFERENCE_SUBSTRATE,
        gpu_uuid="GPU-70310000-0000-0000-0000-000000000001",
        gpu_model="NVIDIA RTX 3090",
        cuda_device=0,
        model_repository=HF_ID,
        model_filename=FILENAME,
        model_hash="sha256:" + "1" * 64,
        n_ctx=4096,
        server_binary="/opt/llama.cpp/llama-server",
        server_binary_hash="sha256:" + "2" * 64,
        server_command_hash="sha256:" + "3" * 64,
        endpoint="http://127.0.0.1:17031",
        port=17031,
        lease_id="lease-7031",
        lease_hash="sha256:" + "4" * 64,
        lease_issued_at="2026-09-05T00:00:00+00:00",
        lease_expires_at="2026-09-05T02:00:00+00:00",
        lease_checked_at="2026-09-05T01:00:00+00:00",
        request_count=1,
        completion_count=1,
        error_count=0,
        policy_hash="sha256:" + "5" * 64,
        factory_hash="sha256:" + "6" * 64,
        git_commit="7" * 40,
        solve_provenance="live_agent_self_discovery",
    )
    return replace(source, **changes)


def _current_record(provenance: Any, identity: Mapping[str, str]) -> JsonDict:
    """Embed the full identity receipt in the current shared provenance schema."""

    return provenance.build_arc_eval_provenance(
        replace(
            _legacy_input(
                provenance,
                model_filename=identity["requested_model_filename"],
                model_hash=identity["model_file_hash"],
            ),
            schema_version=provenance.ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
            **identity,
        )
    )


def _rejection_row(case: str, spec: Mapping[str, Any], observed: Path) -> JsonDict:
    """Call the shared bridge and preserve its complete rejection text."""

    provenance = importlib.import_module("carnot.agentic.arc_eval_provenance")
    accepted = True
    reason: str | None = None
    try:
        provenance.build_arc_model_identity_receipt(
            selected_model_spec=spec,
            observed_server_model_path=str(observed),
        )
    except (TypeError, ValueError) as exc:
        accepted = False
        reason = str(exc)
    return {
        "case": case,
        "changed_factor": case,
        "accepted": accepted,
        "reject_reason": reason,
        "terminal": True,
    }


def _consumer_reachability(provenance: Any) -> list[JsonDict]:
    """Prove the live consumer imports shared function objects and has no copies."""

    shadow = importlib.import_module("carnot.agentic.arc_belief_shadow_live_trace")
    shared_path = Path(
        inspect.getsourcefile(provenance.build_arc_model_identity_receipt) or ""
    ).resolve()
    shadow_source = Path(shadow.__file__).read_text(encoding="utf-8")
    bindings = (
        (
            "build_arc_model_identity_receipt",
            shadow.build_arc_model_identity_receipt is provenance.build_arc_model_identity_receipt,
            shadow.build_arc_model_identity_receipt,
        ),
        (
            "validate_arc_evaluation_row",
            shadow.validate_arc_evaluation_row is provenance.validate_arc_evaluation_row,
            shadow.validate_arc_evaluation_row,
        ),
        (
            "validate_arc_eval_provenance",
            provenance.validate_arc_evaluation_row.__globals__.get("validate_arc_eval_provenance")
            is provenance.validate_arc_eval_provenance,
            provenance.validate_arc_eval_provenance,
        ),
    )
    return [
        {
            "consumer": "carnot.agentic.arc_belief_shadow_live_trace",
            "symbol": symbol,
            "shared_function_identity": identity,
            "shared_source_file": Path(inspect.getsourcefile(function) or "").resolve()
            == shared_path,
            "local_copy_present": f"def {symbol}(" in shadow_source,
            "shared_source_path": str(shared_path),
            "terminal": True,
        }
        for symbol, identity, function in bindings
    ]


def execute_worker(
    work_dir: Path, *, parent_pid: int | None = None, isolated_python: bool | None = None
) -> JsonDict:
    """Create independent fixtures and audit the shared bridge in this process."""

    provenance = importlib.import_module("carnot.agentic.arc_eval_provenance")
    work_dir.mkdir(parents=True, exist_ok=True)

    spec, blob, _requested = _snapshot_fixture(work_dir / "positive-symlink")
    identity = provenance.build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(blob),
    )
    record = _current_record(provenance, identity)
    round_trip = json.loads(json.dumps(record, sort_keys=True))
    decision = provenance.validate_arc_eval_provenance(round_trip)
    positive_rows = [
        {
            "case": "snapshot_symlink_to_extensionless_blob",
            "accepted": decision.valid,
            "headline_eligible": decision.headline_eligible,
            "receipt_round_trip_equal": round_trip == record,
            "receipt": identity,
            "validation_errors": list(decision.errors),
            "terminal": True,
        }
    ]

    mutation_rows: dict[str, list[JsonDict]] = {field: [] for field in MUTATION_CASES}

    spec, blob, _ = _snapshot_fixture(work_dir / "mut-content-hash")
    spec["model_file_hash"] = "sha256:" + "0" * 64
    mutation_rows["hash_mutation_rows"].append(_rejection_row("content_hash", spec, blob))

    spec, blob, _ = _snapshot_fixture(work_dir / "mut-repository")
    spec["hf_id"] = "cold-auditor/wrong-repository"
    mutation_rows["hub_revision_mutation_rows"].append(_rejection_row("repository", spec, blob))

    spec, blob, _ = _snapshot_fixture(work_dir / "mut-revision")
    spec["revision"] = "stale-revision"
    mutation_rows["hub_revision_mutation_rows"].append(_rejection_row("revision", spec, blob))

    spec, blob, _ = _snapshot_fixture(work_dir / "mut-requested-filename")
    spec["model_filename"] = "confused-name.gguf"
    mutation_rows["alias_mutation_rows"].append(_rejection_row("requested_filename", spec, blob))

    spec, blob, _ = _snapshot_fixture(work_dir / "mut-observed-path")
    observed_alias = work_dir / "mut-observed-path" / "server-model-alias"
    observed_alias.symlink_to(blob)
    mutation_rows["alias_mutation_rows"].append(
        _rejection_row("observed_path", spec, observed_alias)
    )

    spec, blob, requested = _snapshot_fixture(work_dir / "mut-broken-link")
    requested.unlink()
    requested.symlink_to("../../blobs/absent-content")
    mutation_rows["path_type_rows"].append(_rejection_row("broken_link", spec, blob))

    spec, blob, requested = _snapshot_fixture(work_dir / "mut-path-type")
    requested.unlink()
    requested.mkdir()
    mutation_rows["path_type_rows"].append(_rejection_row("path_type", spec, blob))

    spec, _blob, _ = _snapshot_fixture(work_dir / "mut-stale-server")
    stale_payload = b"stale server identity from an earlier process\n"
    stale_digest = hashlib.sha256(stale_payload).hexdigest()
    stale = (
        work_dir
        / "mut-stale-server"
        / "hub"
        / "models--cold-auditor--independent-model-GGUF"
        / "blobs"
        / stale_digest
    )
    stale.write_bytes(stale_payload)
    mutation_rows["stale_server_rows"].append(_rejection_row("stale_server_identity", spec, stale))

    spec, _blob, _ = _snapshot_fixture(work_dir / "mut-same-size")
    baseline = b"independent Exp7031 GGUF identity bytes\n"
    different = b"X" * len(baseline)
    different_digest = hashlib.sha256(different).hexdigest()
    same_size = (
        work_dir
        / "mut-same-size"
        / "hub"
        / "models--cold-auditor--independent-model-GGUF"
        / "blobs"
        / different_digest
    )
    same_size.write_bytes(different)
    mutation_rows["hash_mutation_rows"].append(
        _rejection_row("same_size_different_bytes", spec, same_size)
    )

    direct_spec, direct = _direct_fixture(work_dir / "mut-hard-link")
    os.link(direct, direct.with_name("second-name.gguf"))
    mutation_rows["path_type_rows"].append(
        _rejection_row("ambiguous_hard_link", direct_spec, direct)
    )

    direct_spec, direct = _direct_fixture(work_dir / "regression-direct")
    direct_identity = provenance.build_arc_model_identity_receipt(
        selected_model_spec=direct_spec,
        observed_server_model_path=str(direct),
    )
    legacy = provenance.build_arc_eval_provenance(_legacy_input(provenance))
    legacy_copy = json.loads(json.dumps(legacy, sort_keys=True))
    legacy_decision = provenance.validate_arc_eval_provenance(legacy_copy)
    legacy_rows = [
        {
            "case": "direct_non_symlink_snapshot_gguf",
            "accepted": direct_identity["resolved_model_path"] == str(direct),
            "current_receipt": direct_identity,
            "terminal": True,
        },
        {
            "case": "complete_legacy_v1_provenance",
            "accepted": legacy_decision.valid,
            "current_fields_inferred": any(
                key in legacy_copy for key in provenance.ARC_MODEL_IDENTITY_KEYS
            ),
            "validation_errors": list(legacy_decision.errors),
            "terminal": True,
        },
    ]

    isolated = bool(sys.flags.isolated) if isolated_python is None else isolated_python
    parent = os.getppid() if parent_pid is None else parent_pid
    allowed_environment = (
        set(_private_environment())
        | {"PYTHONPATH"}
        | {key for key in os.environ if key.startswith("CARNOT_CHILD_GUARD_")}
    )
    process_rows = [
        {
            "pid": os.getpid(),
            "parent_pid": parent,
            "isolated_python": isolated,
            "private_work_dir": Path.cwd().resolve() == work_dir.resolve(),
            "minimal_environment": set(os.environ) <= allowed_environment,
            "environment_keys": sorted(os.environ),
            "core_dump_limit": list(resource.getrlimit(resource.RLIMIT_CORE)),
            "fixture_origin": "generated_by_exp7031_worker",
            "round_trip_valid": decision.valid and round_trip == record,
            "returncode": 0,
            "terminal": True,
        }
    ]
    return {
        "positive_identity_rows": positive_rows,
        **mutation_rows,
        "legacy_regression_rows": legacy_rows,
        "fresh_process_rows": process_rows,
        "consumer_reachability_rows": _consumer_reachability(provenance),
    }


def _empty_evidence() -> JsonDict:
    """Create every row field before a precondition can stop the audit."""

    return {
        "rows": [],
        "positive_identity_rows": [],
        "alias_mutation_rows": [],
        "hash_mutation_rows": [],
        "hub_revision_mutation_rows": [],
        "path_type_rows": [],
        "stale_server_rows": [],
        "legacy_regression_rows": [],
        "fresh_process_rows": [],
        "consumer_reachability_rows": [],
    }


def _base_artifact(
    *,
    execution_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    evidence: Mapping[str, Any],
    command_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the common complete schema for every terminal result class."""

    upstream = evidence.get("upstream")
    upstream = upstream if isinstance(upstream, Mapping) else {}
    citation = {
        "path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "artifact_hash": evidence.get("artifact_hash"),
        "gate_field": "arc_model_identity_bridge_ready_score",
        "gate_value": upstream.get("arc_model_identity_bridge_ready_score"),
        "terminal": True,
    }
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": str(execution_date),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": dict(source_hashes),
        "cited_upstream_artifacts": [citation],
        "upstream_gate_rows": [dict(row) for row in evidence.get("upstream_gate_rows", [])],
        **_empty_evidence(),
        "command_receipt_rows": [dict(row) for row in command_rows],
        "arc_model_identity_audit_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_check_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_arc_model_identity_cold_audit:unknown_precondition",
    }


def _audit_checks(
    worker: Mapping[str, Any], command_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reduce independent rows to the exact gates that control readiness."""

    positive = worker.get("positive_identity_rows")
    legacy = worker.get("legacy_regression_rows")
    fresh = worker.get("fresh_process_rows")
    reachability = worker.get("consumer_reachability_rows")
    mutation_rows = [
        row
        for field in MUTATION_CASES
        for row in (worker.get(field) if isinstance(worker.get(field), list) else [])
    ]
    expected_mutations = set().union(*MUTATION_CASES.values())
    observed_mutations = {row.get("case") for row in mutation_rows if isinstance(row, Mapping)}
    return [
        gate_row(
            "positive_identity_rows",
            True,
            isinstance(positive, list)
            and bool(positive)
            and all(row.get("accepted") is True for row in positive),
        ),
        gate_row(
            "all_identity_mutations_rejected",
            True,
            observed_mutations == expected_mutations
            and all(
                row.get("accepted") is False
                and isinstance(row.get("reject_reason"), str)
                and bool(row.get("reject_reason"))
                for row in mutation_rows
            ),
        ),
        gate_row(
            "legacy_regression_rows",
            True,
            isinstance(legacy, list)
            and {row.get("case") for row in legacy}
            == {"direct_non_symlink_snapshot_gguf", "complete_legacy_v1_provenance"}
            and all(row.get("accepted") is True for row in legacy),
        ),
        gate_row(
            "fresh_restricted_process",
            True,
            isinstance(fresh, list)
            and len(fresh) == 1
            and fresh[0].get("pid") != fresh[0].get("parent_pid")
            and all(
                fresh[0].get(key) is True
                for key in (
                    "isolated_python",
                    "private_work_dir",
                    "minimal_environment",
                    "round_trip_valid",
                )
            ),
        ),
        gate_row(
            "consumer_reachability",
            True,
            isinstance(reachability, list)
            and len(reachability) == 3
            and all(
                row.get("shared_function_identity") is True
                and row.get("shared_source_file") is True
                and row.get("local_copy_present") is False
                for row in reachability
            ),
        ),
        gate_row(
            "command_receipts",
            True,
            len(command_rows) >= 2
            and all(
                row.get("returncode") == 0 and row.get("terminal") is True for row in command_rows
            ),
        ),
    ]


def _run_worker(work_dir: Path) -> tuple[JsonDict, JsonDict]:
    """Start the audit worker with isolated imports and parse its sole JSON result."""

    worker_dir = work_dir / "fresh-worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    argv = (
        sys.executable,
        "-I",
        "-m",
        "carnot.experiment_7031_arc_model_identity_cold_audit",
        "--worker",
        "--work-dir",
        str(worker_dir),
        "--parent-pid",
        str(os.getpid()),
    )
    completed, receipt = _isolated_command(argv, work_dir=worker_dir)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {}
    return (dict(payload) if isinstance(payload, dict) else {}), receipt


def build_artifact(
    root: Path,
    *,
    execution_date: str,
    output_path: Path,
    work_dir: Path,
    upstream_path: Path | None = None,
) -> JsonDict:
    """Check the upstream gate, run the cold worker, and build one terminal result."""

    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    checks, evidence, source_hashes, command_rows = collect_preconditions(
        root, output_path, work_dir, upstream_path
    )
    artifact = _base_artifact(
        execution_date=execution_date,
        duration_s=0.0,
        checks=checks,
        source_hashes=source_hashes,
        evidence=evidence,
        command_rows=command_rows,
    )
    if any(row.get("passed") is not True for row in checks):
        summary = gate_check_summary(checks)
        artifact["duration_s"] = time.perf_counter() - started
        artifact["gate_check_summary"] = summary
        artifact["honest_verdict"] = "blocked_arc_model_identity_cold_audit:" + str(
            summary["failed_check"]
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    worker, worker_receipt = _run_worker(work_dir)
    command_rows.append(worker_receipt)
    audit_checks = _audit_checks(worker, command_rows)
    all_checks = [*checks, *audit_checks]
    ready = all(row["passed"] is True for row in all_checks)
    for field in _empty_evidence():
        artifact[field] = worker.get(field, [])
    artifact["rows"] = [
        {
            "check": row["check"],
            "passed": row["passed"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
            "terminal": True,
        }
        for row in audit_checks
    ]
    artifact["command_receipt_rows"] = command_rows
    artifact["arc_model_identity_audit_ready_score"] = 1 if ready else 0
    artifact["gate_check_summary"] = gate_check_summary(all_checks)
    artifact["verdict_class"] = "positive" if ready else "disqualified"
    artifact["honest_verdict"] = (
        "complete_positive_arc_model_identity_cold_audit_ready"
        if ready
        else "disqualified_arc_model_identity_cold_audit:"
        + str(artifact["gate_check_summary"]["failed_check"])
    )
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _ready_evidence(artifact: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Recompute every row condition that can support a positive score."""

    errors: list[str] = []
    positive = artifact.get("positive_identity_rows")
    if (
        not isinstance(positive, list)
        or not positive
        or any(not isinstance(row, Mapping) or row.get("accepted") is not True for row in positive)
    ):
        errors.append("positive_identity_rows_invalid")

    for field, cases in MUTATION_CASES.items():
        rows = artifact.get(field)
        if (
            not isinstance(rows, list)
            or {row.get("case") for row in rows if isinstance(row, Mapping)} != cases
            or any(
                not isinstance(row, Mapping)
                or row.get("accepted") is not False
                or row.get("changed_factor") != row.get("case")
                or not isinstance(row.get("reject_reason"), str)
                or not str(row.get("reject_reason")).startswith("invalid ARC model identity:")
                for row in rows
            )
        ):
            errors.append(f"mutation_rows_invalid:{field}")

    legacy = artifact.get("legacy_regression_rows")
    if (
        not isinstance(legacy, list)
        or {row.get("case") for row in legacy if isinstance(row, Mapping)}
        != {"direct_non_symlink_snapshot_gguf", "complete_legacy_v1_provenance"}
        or any(not isinstance(row, Mapping) or row.get("accepted") is not True for row in legacy)
    ):
        errors.append("legacy_regression_rows_invalid")

    fresh = artifact.get("fresh_process_rows")
    if (
        not isinstance(fresh, list)
        or len(fresh) != 1
        or not isinstance(fresh[0], Mapping)
        or fresh[0].get("pid") == fresh[0].get("parent_pid")
        or any(
            fresh[0].get(key) is not True
            for key in (
                "isolated_python",
                "private_work_dir",
                "minimal_environment",
                "round_trip_valid",
            )
        )
    ):
        errors.append("fresh_process_rows_invalid")

    reachability = artifact.get("consumer_reachability_rows")
    if (
        not isinstance(reachability, list)
        or {row.get("symbol") for row in reachability if isinstance(row, Mapping)}
        != {
            "build_arc_model_identity_receipt",
            "validate_arc_evaluation_row",
            "validate_arc_eval_provenance",
        }
        or any(
            not isinstance(row, Mapping)
            or row.get("shared_function_identity") is not True
            or row.get("shared_source_file") is not True
            or row.get("local_copy_present") is not False
            for row in reachability
        )
    ):
        errors.append("consumer_reachability_rows_invalid")

    commands = artifact.get("command_receipt_rows")
    if (
        not isinstance(commands, list)
        or len(commands) < 2
        or any(
            not isinstance(row, Mapping)
            or row.get("returncode") != 0
            or row.get("terminal") is not True
            for row in commands
        )
    ):
        errors.append("command_receipt_rows_invalid")

    upstream_rows = artifact.get("upstream_gate_rows")
    if (
        not isinstance(upstream_rows, list)
        or len(upstream_rows) != 3
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True for row in upstream_rows
        )
    ):
        errors.append("upstream_gate_rows_invalid")
    preconditions = artifact.get("preconditions_checked")
    if (
        not isinstance(preconditions, list)
        or not preconditions
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True for row in preconditions
        )
    ):
        errors.append("preconditions_invalid")
    return not errors, errors


def validate_artifact(artifact: Any) -> list[str]:
    """Recompute schema, evidence, score, verdict, and checksum without trust."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not isinstance(principles.get(field), str) or not principles.get(field)
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        errors.append("duration_s_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    score = artifact.get("arc_model_identity_audit_ready_score")
    if not isinstance(score, int) or isinstance(score, bool) or score not in (0, 1):
        errors.append("ready_score_invalid")

    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    summary = artifact.get("gate_check_summary")
    if verdict_class == "blocked":
        if score != 0:
            errors.append("blocked_ready_score_invalid")
        if (
            not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or not summary.get("failed_check")
            or "expected_value" not in summary
            or "observed_value" not in summary
        ):
            errors.append("blocked_gate_check_summary_invalid")
        if not verdict.startswith("blocked_arc_model_identity_cold_audit:"):
            errors.append("blocked_verdict_invalid")
    else:
        ready, evidence_errors = _ready_evidence(artifact)
        errors.extend(evidence_errors)
        rows = artifact.get("rows")
        if not isinstance(rows, list) or len(rows) != 6:
            errors.append("rows_invalid")
        if score != int(ready):
            errors.append("ready_score_not_supported")
        if ready:
            if not isinstance(summary, Mapping) or summary.get("passed") is not True:
                errors.append("gate_check_summary_invalid")
            if verdict_class != "positive" or not verdict.startswith("complete_positive_"):
                errors.append("positive_verdict_invalid")
        elif verdict_class != "disqualified" or not verdict.startswith("disqualified_"):
            errors.append("disqualified_verdict_invalid")
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish one terminal audit artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp7031 artifact: " + ";".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(artifact), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(root: Path, *, execution_date: str, output_path: Path) -> JsonDict:
    """Run with a private temporary directory and write one validated result."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp7031-") as name:
        artifact = build_artifact(
            root,
            execution_date=execution_date,
            output_path=output_path,
            work_dir=Path(name),
        )
        write_artifact(output_path, artifact)
    return artifact


def _worker_main(args: argparse.Namespace) -> int:  # pragma: no cover - subprocess protocol
    """Emit one worker payload and fail if any audited identity gate is open."""

    _restrict_process()
    payload = execute_worker(
        args.work_dir,
        parent_pid=args.parent_pid,
        isolated_python=bool(sys.flags.isolated),
    )
    print(json.dumps(payload, sort_keys=True))
    commands = [{"returncode": 0, "terminal": True}, {"returncode": 0, "terminal": True}]
    return 0 if all(row["passed"] for row in _audit_checks(payload, commands)[:-1]) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Support only the private worker protocol; the public runner has its own script."""

    parser = argparse.ArgumentParser(description="Run the Exp7031 isolated audit worker")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("--worker is required")
    return _worker_main(args)


if __name__ == "__main__":  # pragma: no cover - exercised by the fresh subprocess
    raise SystemExit(main())
