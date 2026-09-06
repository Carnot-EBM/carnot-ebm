"""Audit the REQ-ARC-7052 typed model identity bridge in a fresh process."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from carnot import experiment_7051_v618_model_report_requalification as exp7051
from carnot.agentic import arc_eval_provenance as provenance


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7052
SCHEMA = "carnot.exp7052.v618_typed_identity_attack_audit.v1"
RANDOM_SEED = 7_052_202_609_06
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
UPSTREAM_RELATIVE_PATH = Path("results/experiment_7051_v618_model_report_requalification.json")
RESULT_RELATIVE_PATH = Path("results/experiment_7052_v618_typed_identity_attack_audit.json")
HF_ID = "frozen-owner/frozen-model-GGUF"
REVISION = "7052frozenrevision"
FILENAME = "frozen-model.gguf"
PAYLOAD = b"frozen Exp7051 GGUF report fixture bytes\n"
ATTACK_CASES = (
    "artifact_checksum_change",
    "broken_link",
    "changed_hub",
    "changed_revision",
    "conflicting_props_fields",
    "hard_link_ambiguity",
    "missing_evidence",
    "relative_alias",
    "same_size_different_content",
    "symlink_swap",
    "wrong_snapshot",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "upstream_gate_rows",
    "rows",
    "identity_schema_version",
    "raw_report_reproduction_rows",
    "identity_obligation_rows",
    "positive_fixture_rows",
    "attack_rows",
    "unknown_evidence_rows",
    "legacy_compatibility_rows",
    "producer_wiring_rows",
    "consumer_wiring_rows",
    "subprocess_rows",
    "requested_model_path",
    "requested_model_filename",
    "requested_hf_id",
    "requested_revision",
    "launch_model_argument",
    "observed_server_model_path",
    "observed_server_resolved_path",
    "resolved_model_path",
    "model_file_hash",
    "typed_identity_attack_audit_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Each required field states the scientific reason for retaining that evidence.",
    "preconditions_checked": "Exact preconditions stop invalid upstream evidence before the audit starts.",
    "inference_substrate": "The substrate separates deterministic evidence aggregation from model inference.",
    "duration_s": "Measured wall time makes audit completion and cost visible.",
    "source_artifact_hashes": "Independent file hashes bind the result to exact code and upstream evidence.",
    "cited_upstream_artifacts": "The citation names the exact upstream report and gate under audit.",
    "upstream_gate_rows": "Separate rows show score, schema, checksum, and file-hash decisions.",
    "rows": "Summary rows make every readiness condition independently countable.",
    "identity_schema_version": "The version prevents old and current receipt fields from being mixed.",
    "raw_report_reproduction_rows": "Raw rows prove that missing and relative values remain unchanged.",
    "identity_obligation_rows": "Typed obligations expose support, contradiction, and unknown evidence.",
    "positive_fixture_rows": "Three positive rows preserve the explicit path-form branches.",
    "attack_rows": "One-factor attacks show which invalid identities fail closed.",
    "unknown_evidence_rows": "Unknown rows prove that absence does not become inferred support.",
    "legacy_compatibility_rows": "Legacy rows prove explicit read-only handling without current-field inference.",
    "producer_wiring_rows": "Producer rows show that current receipts come from shared code.",
    "consumer_wiring_rows": "Consumer rows preserve evaluator, reuse, and belief-shadow reachability.",
    "subprocess_rows": "Process rows prove that a new interpreter repeated the audit.",
    "requested_model_path": "The requested path preserves the selected snapshot intent.",
    "requested_model_filename": "The requested file name prevents basename substitution.",
    "requested_hf_id": "The requested hub ID prevents cross-repository substitution.",
    "requested_revision": "The requested revision binds the selected snapshot.",
    "launch_model_argument": "The launch argument remains distinct from the later server report.",
    "observed_server_model_path": "The raw server path remains unchanged before resolution.",
    "observed_server_resolved_path": "The server resolution stays separate from its raw report.",
    "resolved_model_path": "The selected path resolution identifies the canonical file.",
    "model_file_hash": "The content hash rejects same-size files with different bytes.",
    "typed_identity_attack_audit_ready_score": "One requires all positive, attack, process, legacy, and wiring gates.",
    "random_seed": "A fixed label identifies this deterministic audit configuration.",
    "reproducibility_checksum": "A canonical digest detects any later included-field change.",
    "gate_check_summary": "The first exact failure gives a blocked result an actionable cause.",
    "verifier_is_oracle": "False states that model identity does not decide ARC correctness.",
    "verdict_class": "A closed class gives the terminal result one machine-readable meaning.",
    "honest_verdict": "A class-consistent prefix prevents a block from reading as success.",
}
SOURCE_PATHS = (
    Path("openspec/capabilities/arc-agi/spec.md"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/agentic/arc_belief_shadow_live_trace.py"),
    Path("python/carnot/experiment_7052_v618_typed_identity_attack_audit.py"),
    Path("scripts/experiments/experiment_7052_v618_typed_identity_attack_audit.py"),
    Path("tests/python/test_arc_typed_identity_attack_audit_20260906.py"),
    UPSTREAM_RELATIVE_PATH,
)


def sha256_file(path: str | Path) -> str | None:
    """Hash one file with an independent bounded reader."""

    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the digest itself."""

    body = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return (
        "sha256:" + hashlib.sha256(provenance.canonical_arc_eval_provenance_bytes(body)).hexdigest()
    )


def gate_row(check: str, expected: Any, observed: Any) -> JsonDict:
    """Retain both sides of one exact terminal gate."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "terminal": True,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failure and keep all ordered gate evidence."""

    copied = [dict(row) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": True if failed is None else failed.get("expected_value"),
        "observed_value": True if failed is None else failed.get("observed_value"),
        "checks": copied,
    }


def _source_artifact(path: Path, raw_props: Mapping[str, Any]) -> None:
    """Write the frozen report with the same terminal checksum projection as Exp7051."""

    body: JsonDict = {
        "raw_server_props": dict(raw_props),
        "checksum_recomputation_rows": [],
        "reproducibility_checksum": "",
    }
    body["reproducibility_checksum"] = provenance.identity_source_payload_sha256(body)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snapshot_fixture(
    directory: Path, *, report_blob: bool = False
) -> tuple[JsonDict, Path, Path, JsonDict, Path]:
    """Create or reopen one deterministic snapshot-to-blob fixture."""

    model_root = directory / "hub" / "models--frozen-owner--frozen-model-GGUF"
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    blob = model_root / "blobs" / digest
    requested = model_root / "snapshots" / REVISION / FILENAME
    source_path = directory / "frozen-exp7051-source.json"
    if not requested.exists():
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            blob.write_bytes(PAYLOAD)
        requested.parent.mkdir(parents=True, exist_ok=True)
        requested.symlink_to(Path("../../blobs") / digest)
    raw_props: JsonDict = {
        "model_path": str(blob if report_blob else requested),
        "model_alias": FILENAME,
    }
    if not source_path.exists():
        _source_artifact(source_path, raw_props)
    spec = {
        "model_path": str(requested),
        "model_filename": FILENAME,
        "hf_id": HF_ID,
        "revision": REVISION,
        "model_file_hash": "sha256:" + digest,
    }
    return spec, requested, blob, raw_props, source_path


def _direct_fixture(directory: Path) -> tuple[JsonDict, Path, JsonDict, Path]:
    """Create or reopen a regular GGUF inside an explicit selected snapshot."""

    requested = (
        directory
        / "hub"
        / "models--frozen-owner--frozen-model-GGUF"
        / "snapshots"
        / REVISION
        / FILENAME
    )
    source_path = directory / "frozen-exp7051-source.json"
    if not requested.exists():
        requested.parent.mkdir(parents=True, exist_ok=True)
        requested.write_bytes(PAYLOAD)
    raw_props: JsonDict = {"model_path": str(requested), "model_alias": FILENAME}
    if not source_path.exists():
        _source_artifact(source_path, raw_props)
    return (
        {
            "model_path": str(requested),
            "model_filename": FILENAME,
            "hf_id": HF_ID,
            "revision": REVISION,
            "model_file_hash": "sha256:" + hashlib.sha256(PAYLOAD).hexdigest(),
        },
        requested,
        raw_props,
        source_path,
    )


def _typed_receipt(
    spec: Mapping[str, Any], raw_props: Mapping[str, Any], source_path: Path
) -> JsonDict:
    """Capture then rebuild one file-backed typed receipt."""

    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=raw_props,
        requested_model_path=spec.get("model_path"),
        source_kind="frozen_exp7051_report_fixture",
        source_artifact_path=source_path,
    )
    return provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=spec.get("model_path"),
        raw_server_props=raw_props,
        source_provenance=source,
    )


def complete_legacy_v1_fixture() -> JsonDict:
    """Return one complete old row for the named read-only compatibility path."""

    return provenance.build_arc_eval_provenance(
        provenance.ArcEvalProvenanceInput(
            inference_substrate=provenance.LIVE_LLM_INFERENCE_SUBSTRATE,
            gpu_uuid="GPU-7052legacy",
            gpu_model="NVIDIA GeForce RTX 3090",
            cuda_device=0,
            model_repository=HF_ID,
            model_filename=FILENAME,
            model_hash="sha256:" + "1" * 64,
            n_ctx=4096,
            server_binary="/opt/llama-server",
            server_binary_hash="sha256:" + "2" * 64,
            server_command_hash="sha256:" + "3" * 64,
            endpoint="http://127.0.0.1:17052",
            port=17052,
            lease_id="legacy-lease",
            lease_hash="sha256:" + "4" * 64,
            lease_issued_at="2026-09-06T00:00:00+00:00",
            lease_expires_at="2026-09-06T00:10:00+00:00",
            lease_checked_at="2026-09-06T00:01:00+00:00",
            request_count=1,
            completion_count=1,
            error_count=0,
            policy_hash="sha256:" + "5" * 64,
            factory_hash="sha256:" + "6" * 64,
            git_commit="7" * 40,
            solve_provenance="live_agent_self_discovery",
        )
    )


def _legacy_rows(spec: Mapping[str, Any], blob: Path) -> list[JsonDict]:
    """Exercise complete v1 and v2 rows without adding or rewriting fields."""

    legacy_v1 = complete_legacy_v1_fixture()
    legacy_v2_identity = provenance.build_arc_model_identity_receipt(
        selected_model_spec=spec, observed_server_model_path=str(blob)
    )
    legacy_fields = {
        key: value
        for key, value in legacy_v1.items()
        if key not in {"schema_version", "provenance_hash"}
    }
    legacy_fields.update(
        {
            "model_repository": legacy_v2_identity["requested_hf_id"],
            "model_filename": legacy_v2_identity["requested_model_filename"],
            "model_hash": legacy_v2_identity["model_file_hash"],
        }
    )
    v2_input = provenance.ArcEvalProvenanceInput(
        **legacy_fields,
        schema_version=provenance.ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
        **legacy_v2_identity,
    )
    legacy_v2 = provenance.build_arc_eval_provenance(v2_input)
    rows = []
    for record in (legacy_v1, legacy_v2):
        reread = provenance.read_complete_legacy_arc_eval_provenance(record)
        rows.append(
            {
                "schema_version": record["schema_version"],
                "accepted": reread == record and reread is not record,
                "current_fields_inferred": any(
                    key in reread
                    for key in (
                        "launch_model_argument",
                        "observed_server_resolved_path",
                        "identity_obligation_rows",
                    )
                ),
                "artifact_rewritten": False,
                "terminal": True,
            }
        )
    return rows


def _attack_receipt(directory: Path, case: str) -> JsonDict:
    """Apply one named mutation after sealing an otherwise valid source."""

    spec, requested, blob, raw_props, source_path = _snapshot_fixture(directory)
    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=raw_props,
        requested_model_path=spec["model_path"],
        source_kind="frozen_exp7051_report_fixture",
        source_artifact_path=source_path,
    )
    launch = spec["model_path"]
    if case == "relative_alias":
        raw_props = {"model_path": FILENAME, "model_alias": FILENAME}
    elif case == "broken_link":
        blob.unlink()
    elif case == "wrong_snapshot":
        spec["model_path"] = str(spec["model_path"]).replace(REVISION, "wrong-snapshot")
    elif case == "same_size_different_content":
        changed = b"X" * len(PAYLOAD)
        changed_path = blob.with_name(hashlib.sha256(changed).hexdigest())
        changed_path.write_bytes(changed)
        raw_props = {"model_path": str(changed_path), "model_alias": FILENAME}
    elif case == "changed_hub":
        spec["hf_id"] = "other-owner/frozen-model-GGUF"
    elif case == "changed_revision":
        spec["revision"] = "changed-revision"
    elif case == "conflicting_props_fields":
        other = directory / "other.gguf"
        other.write_bytes(PAYLOAD)
        raw_props["model"] = str(other)
    elif case == "hard_link_ambiguity":
        os.link(blob, blob.with_name("second-link"))
    elif case == "symlink_swap":
        other_root = directory / "other-hub" / "blobs"
        other_root.mkdir(parents=True)
        other_blob = other_root / blob.name
        other_blob.write_bytes(PAYLOAD)
        requested.unlink()
        requested.symlink_to(other_blob)
    elif case == "missing_evidence":
        raw_props = {"model_alias": FILENAME}
    elif case == "artifact_checksum_change":
        changed_source = json.loads(source_path.read_text(encoding="utf-8"))
        changed_source["post_capture_change"] = True
        source_path.write_text(json.dumps(changed_source, sort_keys=True), encoding="utf-8")
    else:
        raise ValueError(f"unknown attack case: {case}")
    return provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=launch,
        raw_server_props=raw_props,
        source_provenance=source,
    )


def _wiring_row(root: Path, consumer: str, relative_path: Path, symbol: str) -> JsonDict:
    """Inspect one named consumer without importing a second validator copy."""

    source = (root / relative_path).read_text(encoding="utf-8")
    try:
        function_source = inspect.getsource(
            getattr(
                importlib.import_module(
                    "carnot.agentic." + relative_path.stem
                    if relative_path.parent.name == "agentic"
                    else "carnot." + relative_path.stem
                ),
                symbol,
            )
        )
    except (AttributeError, ImportError, OSError, TypeError):
        function_source = source
    return {
        "consumer": consumer,
        "path": relative_path.as_posix(),
        "symbol": symbol,
        "shared_builder": "build_typed_arc_model_identity_receipt" in function_source,
        "shared_validator": "validate_typed_arc_model_identity_receipt" in function_source,
        "local_copy_present": (
            relative_path != Path("python/carnot/agentic/arc_eval_provenance.py")
            and "def build_typed_arc_model_identity_receipt" in source
        ),
        "terminal": True,
    }


def consumer_wiring_rows(root: Path) -> list[JsonDict]:
    """Audit the evaluator, reuse check, and belief-shadow identity consumers."""

    return [
        _wiring_row(
            root,
            "submitted_arc_evaluator",
            Path("python/carnot/agentic/arc_eval_provenance.py"),
            "build_arc_eval_provenance_for_policy",
        ),
        _wiring_row(
            root,
            "reusable_server_check",
            Path("python/carnot/agentic/arc_executable_world_model.py"),
            "LocalGGUFProposer",
        ),
        _wiring_row(
            root,
            "belief_shadow_consumer",
            Path("python/carnot/agentic/arc_belief_shadow_live_trace.py"),
            "run_live_trace",
        ),
    ]


def producer_wiring_rows(root: Path) -> list[JsonDict]:
    """Record both current locations that produce a typed receipt."""

    consumers = consumer_wiring_rows(root)
    return [
        {**row, "producer": row.pop("consumer")}
        for row in deepcopy(consumers)
        if row["consumer"] in {"submitted_arc_evaluator", "belief_shadow_consumer"}
    ]


def execute_audit_fixture(work_dir: Path, *, parent_pid: int | None = None) -> JsonDict:
    """Build all positive forms and attacks from private deterministic files."""

    work_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot_fixture(work_dir / "positive-snapshot")
    spec, _requested, blob, raw_props, source_path = snapshot
    primary = _typed_receipt(spec, raw_props, source_path)
    positive_rows: list[JsonDict] = []
    for path_form in ("snapshot_alias", "canonical_blob", "direct_file"):
        if path_form == "snapshot_alias":
            candidate = primary
        elif path_form == "canonical_blob":
            c_spec, _c_requested, _c_blob, c_props, c_source = _snapshot_fixture(
                work_dir / "positive-canonical", report_blob=True
            )
            candidate = _typed_receipt(c_spec, c_props, c_source)
        else:
            d_spec, _direct, d_props, d_source = _direct_fixture(work_dir / "positive-direct")
            candidate = _typed_receipt(d_spec, d_props, d_source)
        decision = provenance.validate_typed_arc_model_identity_receipt(candidate)
        positive_rows.append(
            {
                "case": path_form,
                "path_form": candidate["path_form"],
                "accepted": decision.valid,
                "validation_errors": list(decision.errors),
                "all_obligations_supported": all(
                    row["status"] == "supported" for row in candidate["identity_obligation_rows"]
                ),
                "terminal": True,
            }
        )

    attack_root = work_dir / "attacks" / str(os.getpid())
    attack_rows: list[JsonDict] = []
    for case in ATTACK_CASES:
        receipt = _attack_receipt(attack_root / case, case)
        decision = provenance.validate_typed_arc_model_identity_receipt(receipt)
        attack_rows.append(
            {
                "case": case,
                "changed_factor": case,
                "accepted": decision.valid,
                "validation_errors": list(decision.errors),
                "non_supported_obligations": [
                    row["obligation"]
                    for row in receipt["identity_obligation_rows"]
                    if row["status"] != "supported"
                ],
                "terminal": True,
            }
        )
    legacy_rows = _legacy_rows(spec, blob)
    return {
        "primary_receipt": primary,
        "positive_obligation_bytes": provenance.canonical_arc_eval_provenance_bytes(
            {"identity_obligation_rows": primary["identity_obligation_rows"]}
        ).decode("ascii"),
        "positive_fixture_rows": positive_rows,
        "attack_rows": attack_rows,
        "legacy_compatibility_rows": legacy_rows,
        "subprocess_rows": [
            {
                "pid": os.getpid(),
                "parent_pid": parent_pid if parent_pid is not None else os.getppid(),
                "isolated_python": bool(sys.flags.isolated),
                "fresh_process": parent_pid is not None and os.getpid() != parent_pid,
                "private_work_dir": work_dir.is_absolute(),
                "terminal": True,
            }
        ],
    }


def _private_environment() -> dict[str, str]:
    """Give the worker only interpreter, locale, and deterministic hash settings."""

    return {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TPU_SKIP_MDS_QUERY": "1",
    }


def run_fresh_subprocess(work_dir: Path) -> tuple[JsonDict, JsonDict]:
    """Start an isolated interpreter and parse its only JSON response."""

    work_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-I",
        "-m",
        "carnot.experiment_7052_v618_typed_identity_attack_audit",
        "--worker",
        "--work-dir",
        str(work_dir.resolve()),
        "--parent-pid",
        str(os.getpid()),
    ]
    completed = subprocess.run(
        argv,
        cwd=work_dir,
        env=_private_environment(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        start_new_session=True,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        value = {}
    receipt = {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout_sha256": "sha256:" + hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": "sha256:" + hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "terminal": True,
    }
    return (dict(value) if isinstance(value, dict) else {}), receipt


def _empty_rows() -> JsonDict:
    """Keep every evidence family present on an upstream block."""

    return {
        "rows": [],
        "raw_report_reproduction_rows": [],
        "identity_obligation_rows": [],
        "positive_fixture_rows": [],
        "attack_rows": [],
        "unknown_evidence_rows": [],
        "legacy_compatibility_rows": [],
        "producer_wiring_rows": [],
        "consumer_wiring_rows": [],
        "subprocess_rows": [],
    }


def blocked_artifact(
    *,
    execution_date: str,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    duration_s: float,
    citation: Mapping[str, Any] | None = None,
    upstream_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one schema-complete terminal block without claiming partial work."""

    summary = gate_check_summary(checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": execution_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_hashes),
        "cited_upstream_artifacts": [dict(citation)] if citation else [],
        "upstream_gate_rows": [dict(row) for row in upstream_rows],
        **_empty_rows(),
        "identity_schema_version": provenance.ARC_MODEL_IDENTITY_SCHEMA_VERSION,
        "requested_model_path": None,
        "requested_model_filename": None,
        "requested_hf_id": None,
        "requested_revision": None,
        "launch_model_argument": None,
        "observed_server_model_path": None,
        "observed_server_resolved_path": None,
        "resolved_model_path": None,
        "model_file_hash": None,
        "typed_identity_attack_audit_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_typed_identity_attack_audit:" + str(summary["failed_check"]),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _source_hashes(root: Path) -> JsonDict:
    return {
        path.as_posix(): sha256_file(root / path)
        for path in SOURCE_PATHS
        if (root / path).is_file()
    }


def collect_preconditions(
    root: Path, output_path: Path, work_dir: Path
) -> tuple[list[JsonDict], JsonDict, list[JsonDict], JsonDict]:
    """Validate Exp7051, imports, independent hashes, and writable paths."""

    upstream_path = root / UPSTREAM_RELATIVE_PATH
    try:
        raw_bytes = upstream_path.read_bytes()
        upstream = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError):
        upstream = {}
        raw_bytes = b""
    upstream = upstream if isinstance(upstream, dict) else {}
    python_hash = "sha256:" + hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else None
    sha_command = subprocess.run(
        ["sha256sum", str(upstream_path)], capture_output=True, text=True, check=False
    )
    command_digest = (
        "sha256:" + sha_command.stdout.split()[0]
        if sha_command.returncode == 0 and sha_command.stdout.split()
        else None
    )
    validation_errors = exp7051.validate_artifact(upstream) if upstream else ["unreadable"]
    recomputed_checksum = exp7051.canonical_artifact_checksum(upstream) if upstream else None
    score = upstream.get("model_report_evidence_ready_score")
    upstream_rows = [
        gate_row(
            "model_report_evidence_ready_score",
            1,
            score if isinstance(score, int) and not isinstance(score, bool) else None,
        ),
        gate_row("exp7051_artifact_validation", [], validation_errors),
        gate_row(
            "exp7051_terminal_checksum",
            upstream.get("reproducibility_checksum"),
            recomputed_checksum,
        ),
        gate_row("exp7051_source_hash_recomputed", python_hash, command_digest),
    ]
    import_rows = []
    for module_name in (
        "carnot.agentic.arc_eval_provenance",
        "carnot.agentic.arc_executable_world_model",
        "carnot.agentic.arc_belief_shadow_live_trace",
    ):
        try:
            importlib.import_module(module_name)
            observed = True
        except Exception:
            observed = False
        import_rows.append(gate_row(f"importable:{module_name}", True, observed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    writable_rows = [
        gate_row(
            "writable_code_path",
            True,
            os.access(root / "python/carnot/agentic/arc_eval_provenance.py", os.W_OK),
        ),
        gate_row(
            "writable_test_path",
            True,
            os.access(
                root / "tests/python/test_arc_typed_identity_attack_audit_20260906.py", os.W_OK
            ),
        ),
        gate_row("writable_artifact_path", True, os.access(output_path.parent, os.W_OK)),
        gate_row("writable_temporary_path", True, os.access(work_dir, os.W_OK)),
    ]
    checks = [*upstream_rows, *import_rows, *writable_rows]
    citation = {
        "path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "artifact_hash": python_hash,
        "gate_field": "model_report_evidence_ready_score",
        "gate_value": score,
        "reproducibility_checksum": upstream.get("reproducibility_checksum"),
        "terminal": True,
    }
    return checks, citation, upstream_rows, _source_hashes(root)


def _audit_gate_rows(
    parent: Mapping[str, Any], child: Mapping[str, Any], command: Mapping[str, Any], root: Path
) -> list[JsonDict]:
    """Reduce every evidence family to a separate readiness decision."""

    positives = parent.get("positive_fixture_rows")
    attacks = parent.get("attack_rows")
    legacy = parent.get("legacy_compatibility_rows")
    child_attacks = child.get("attack_rows")
    consumers = consumer_wiring_rows(root)
    producers = producer_wiring_rows(root)
    return [
        gate_row(
            "positive_paths",
            True,
            isinstance(positives, list)
            and {row.get("case") for row in positives}
            == {"snapshot_alias", "canonical_blob", "direct_file"}
            and all(
                row.get("accepted") is True and row.get("all_obligations_supported") is True
                for row in positives
            ),
        ),
        gate_row(
            "parent_attacks_fail_closed",
            True,
            isinstance(attacks, list)
            and {row.get("case") for row in attacks} == set(ATTACK_CASES)
            and all(
                row.get("accepted") is False and row.get("non_supported_obligations")
                for row in attacks
            ),
        ),
        gate_row(
            "fresh_process_agreement",
            True,
            command.get("returncode") == 0
            and child.get("positive_obligation_bytes") == parent.get("positive_obligation_bytes")
            and isinstance(child_attacks, list)
            and {row.get("case") for row in child_attacks} == set(ATTACK_CASES)
            and all(row.get("accepted") is False for row in child_attacks),
        ),
        gate_row(
            "legacy_handling_explicit",
            True,
            isinstance(legacy, list)
            and len(legacy) == 2
            and all(
                row.get("accepted") is True
                and row.get("current_fields_inferred") is False
                and row.get("artifact_rewritten") is False
                for row in legacy
            ),
        ),
        gate_row(
            "all_consumers_use_shared_code",
            True,
            len(consumers) == 3
            and all(
                row["shared_builder"] is True
                and row["shared_validator"] is True
                and row["local_copy_present"] is False
                for row in consumers
            ),
        ),
        gate_row(
            "all_producers_use_shared_code",
            True,
            len(producers) == 2
            and all(
                row["shared_builder"] is True and row["shared_validator"] is True
                for row in producers
            ),
        ),
    ]


def _positive_artifact(
    *,
    execution_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    citation: Mapping[str, Any],
    upstream_rows: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    command: Mapping[str, Any],
    root: Path,
) -> JsonDict:
    """Assemble a terminal artifact from independently checked rows."""

    receipt = dict(parent["primary_receipt"])
    audit_rows = _audit_gate_rows(parent, child, command, root)
    checks = [*[dict(row) for row in preconditions], *audit_rows]
    ready = all(row["passed"] is True for row in checks)
    raw_rows = list(receipt["raw_report_observations"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": execution_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_hashes),
        "cited_upstream_artifacts": [dict(citation)],
        "upstream_gate_rows": [dict(row) for row in upstream_rows],
        "rows": audit_rows,
        "identity_schema_version": receipt["identity_schema_version"],
        "raw_report_reproduction_rows": raw_rows,
        "identity_obligation_rows": list(receipt["identity_obligation_rows"]),
        "positive_fixture_rows": list(parent["positive_fixture_rows"]),
        "attack_rows": list(parent["attack_rows"]),
        "unknown_evidence_rows": [row for row in raw_rows if row["status"] == "unknown"],
        "legacy_compatibility_rows": list(parent["legacy_compatibility_rows"]),
        "producer_wiring_rows": producer_wiring_rows(root),
        "consumer_wiring_rows": consumer_wiring_rows(root),
        "subprocess_rows": [*list(child.get("subprocess_rows", [])), dict(command)],
        **{
            key: receipt[key]
            for key in (
                "requested_model_path",
                "requested_model_filename",
                "requested_hf_id",
                "requested_revision",
                "launch_model_argument",
                "observed_server_model_path",
                "observed_server_resolved_path",
                "resolved_model_path",
                "model_file_hash",
            )
        },
        "typed_identity_attack_audit_ready_score": 1 if ready else 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_check_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "disqualified",
        "honest_verdict": (
            "complete_positive_typed_identity_attack_audit_ready"
            if ready
            else "disqualified_typed_identity_attack_audit:"
            + str(gate_check_summary(checks)["failed_check"])
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_audit_artifact_from_frozen_fixture(*, execution_date: str, work_dir: Path) -> JsonDict:
    """Build an in-test artifact without reading the mutable Exp7051 result."""

    started = time.perf_counter()
    parent = execute_audit_fixture(work_dir)
    child, command = run_fresh_subprocess(work_dir)
    preconditions = [gate_row("frozen_test_precondition", True, True)]
    return _positive_artifact(
        execution_date=execution_date,
        duration_s=time.perf_counter() - started,
        preconditions=preconditions,
        citation={"path": "temporary-frozen-exp7051-fixture", "terminal": True},
        upstream_rows=preconditions,
        source_hashes={},
        parent=parent,
        child=child,
        command=command,
        root=Path(__file__).resolve().parents[2],
    )


def validate_artifact(artifact: Any) -> list[str]:
    """Recompute schema, evidence, score, verdict, and checksum without trust."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    if missing := sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)):
        errors.append("required_fields_missing:" + ",".join(missing))
    principles = artifact.get("field_principles")
    if principles != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("identity_schema_version") != provenance.ARC_MODEL_IDENTITY_SCHEMA_VERSION:
        errors.append("identity_schema_version_invalid")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        errors.append("duration_s_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    score = artifact.get("typed_identity_attack_audit_ready_score")
    if not isinstance(score, int) or isinstance(score, bool) or score not in {0, 1}:
        errors.append("ready_score_invalid")
    verdict_class = artifact.get("verdict_class")
    verdict = artifact.get("honest_verdict")
    summary = artifact.get("gate_check_summary")
    if verdict_class == "blocked":
        if score != 0 or not isinstance(verdict, str) or not verdict.startswith("blocked"):
            errors.append("blocked_verdict_invalid")
        if (
            not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or not summary.get("failed_check")
            or "expected_value" not in summary
            or "observed_value" not in summary
        ):
            errors.append("blocked_gate_check_summary_invalid")
    else:
        positives = artifact.get("positive_fixture_rows")
        if (
            not isinstance(positives, list)
            or {row.get("case") for row in positives if isinstance(row, Mapping)}
            != {"snapshot_alias", "canonical_blob", "direct_file"}
            or any(
                not isinstance(row, Mapping) or row.get("accepted") is not True for row in positives
            )
        ):
            errors.append("positive_fixture_rows_invalid")
        attacks = artifact.get("attack_rows")
        if (
            not isinstance(attacks, list)
            or {row.get("case") for row in attacks if isinstance(row, Mapping)} != set(ATTACK_CASES)
            or any(
                not isinstance(row, Mapping) or row.get("accepted") is not False for row in attacks
            )
        ):
            errors.append("attack_rows_invalid")
        obligations = artifact.get("identity_obligation_rows")
        if (
            not isinstance(obligations, list)
            or [row.get("obligation") for row in obligations if isinstance(row, Mapping)]
            != list(provenance.IDENTITY_OBLIGATIONS)
            or any(row.get("status") != "supported" for row in obligations)
        ):
            errors.append("identity_obligation_rows_invalid")
        legacy = artifact.get("legacy_compatibility_rows")
        if (
            not isinstance(legacy, list)
            or len(legacy) != 2
            or any(
                row.get("accepted") is not True
                or row.get("current_fields_inferred") is not False
                or row.get("artifact_rewritten") is not False
                for row in legacy
            )
        ):
            errors.append("legacy_compatibility_rows_invalid")
        consumers = artifact.get("consumer_wiring_rows")
        if (
            not isinstance(consumers, list)
            or len(consumers) != 3
            or any(
                row.get("shared_builder") is not True
                or row.get("shared_validator") is not True
                or row.get("local_copy_present") is not False
                for row in consumers
            )
        ):
            errors.append("consumer_wiring_rows_invalid")
        processes = artifact.get("subprocess_rows")
        if (
            not isinstance(processes, list)
            or not any(
                row.get("fresh_process") is True for row in processes if isinstance(row, Mapping)
            )
            or not any(row.get("returncode") == 0 for row in processes if isinstance(row, Mapping))
        ):
            errors.append("subprocess_rows_invalid")
        ready = not errors
        if score != int(ready):
            errors.append("ready_score_contradiction")
        if verdict_class not in {"positive", "disqualified"}:
            errors.append("verdict_class_invalid")
        elif verdict_class == "positive" and (
            not ready or not isinstance(verdict, str) or not verdict.startswith("complete_positive")
        ):
            errors.append("positive_verdict_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def build_artifact(
    root: Path, *, execution_date: str, output_path: Path, work_dir: Path
) -> JsonDict:
    """Validate Exp7051, run both audits, and return one terminal result."""

    started = time.perf_counter()
    checks, citation, upstream_rows, source_hashes = collect_preconditions(
        root, output_path, work_dir
    )
    if any(row["passed"] is not True for row in checks):
        return blocked_artifact(
            execution_date=execution_date,
            checks=checks,
            source_hashes=source_hashes,
            duration_s=time.perf_counter() - started,
            citation=citation,
            upstream_rows=upstream_rows,
        )
    parent = execute_audit_fixture(work_dir)
    child, command = run_fresh_subprocess(work_dir)
    return _positive_artifact(
        execution_date=execution_date,
        duration_s=time.perf_counter() - started,
        preconditions=checks,
        citation=citation,
        upstream_rows=upstream_rows,
        source_hashes=source_hashes,
        parent=parent,
        child=child,
        command=command,
        root=root,
    )


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate then atomically publish one terminal artifact."""

    if errors := validate_artifact(artifact):
        raise ValueError("invalid Exp7052 artifact: " + ";".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(artifact), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(root: Path, *, execution_date: str, output_path: Path) -> JsonDict:
    """Use a temporary audit directory and write one stable terminal result."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp7052-") as directory:
        artifact = build_artifact(
            root,
            execution_date=execution_date,
            output_path=output_path,
            work_dir=Path(directory),
        )
    write_artifact(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the isolated worker protocol only when explicitly requested."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    if not args.worker or args.work_dir is None or args.parent_pid is None:
        return 2
    result = execute_audit_fixture(args.work_dir, parent_pid=args.parent_pid)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
