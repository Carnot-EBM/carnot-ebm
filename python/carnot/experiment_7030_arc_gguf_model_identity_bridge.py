"""Build the deterministic REQ-ARC-7030 model identity bridge artifact.

This experiment reproduces the Exp7025 cache topology with small local files.
It does not load a model or use a GPU. The shared provenance code performs the
same path, revision, and content checks that the live producers use.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from carnot.agentic import arc_eval_provenance as provenance
from carnot.agentic.arc_eval_provenance import (
    ARC_EVAL_PROVENANCE_SCHEMA_VERSION,
    ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
    ArcEvalProvenanceInput,
    build_arc_eval_provenance,
    build_arc_model_identity_receipt,
    validate_arc_eval_provenance,
)


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7030
SCHEMA = "carnot.exp7030.arc_gguf_model_identity_bridge.v1"
RANDOM_SEED = 7_030_202_609_05
HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
REVISION = "a483e9e6cbd595906af30beda3187c2663a1118c"
FILENAME = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
RESULT_RELATIVE_PATH = Path("results/experiment_7030_arc_gguf_model_identity_bridge.json")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "identity_schema_version",
    "exp7025_reproduction_rows",
    "requested_identity_rows",
    "observed_identity_rows",
    "hash_join_rows",
    "hub_revision_rows",
    "positive_fixture_rows",
    "negative_fixture_rows",
    "legacy_compatibility_rows",
    "producer_wiring_rows",
    "consumer_wiring_rows",
    "command_receipt_rows",
    "requested_model_path",
    "requested_model_filename",
    "requested_hf_id",
    "requested_revision",
    "observed_server_model_path",
    "resolved_model_path",
    "model_file_hash",
    "arc_model_identity_bridge_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why the identity claim needs it.",
    "preconditions_checked": "Exact gates prevent a missing source from becoming synthetic evidence.",
    "inference_substrate": "The substrate says this artifact aggregates evidence without model inference.",
    "duration_s": "Measured wall time makes the deterministic run cost auditable.",
    "source_artifact_hashes": "Source hashes bind the result to the evidence and code that produced it.",
    "rows": "Summary rows make the readiness reduction independently countable.",
    "identity_schema_version": "A version selects exact fields without guessing legacy aliases.",
    "exp7025_reproduction_rows": "The incident row proves the fixture matches the observed cache topology.",
    "requested_identity_rows": "Requested rows preserve the strict snapshot filename and selection intent.",
    "observed_identity_rows": "Observed rows preserve the canonical path reported by the server.",
    "hash_join_rows": "Hash rows prove both paths identify the same bytes.",
    "hub_revision_rows": "Hub and revision rows bind the file to the selected snapshot.",
    "positive_fixture_rows": "A positive fixture proves the Exp7025-shaped bridge can succeed.",
    "negative_fixture_rows": "Negative fixtures prove every named identity defect fails closed.",
    "legacy_compatibility_rows": "Legacy rows prove prior complete records remain readable without rewriting.",
    "producer_wiring_rows": "Producer rows prove canonical provenance calls the shared bridge.",
    "consumer_wiring_rows": "Consumer rows prove current validation recomputes the shared bridge.",
    "command_receipt_rows": "Command rows record the deterministic operations used to make the result.",
    "requested_model_path": "The snapshot path records what the producer asked llama.cpp to load.",
    "requested_model_filename": "The requested GGUF name stays separate from an extensionless blob name.",
    "requested_hf_id": "The hub ID binds the path to the selected repository.",
    "requested_revision": "The revision binds the path to one selected snapshot.",
    "observed_server_model_path": "The observed path records the server report without renaming it.",
    "resolved_model_path": "The resolved path records the canonical snapshot target.",
    "model_file_hash": "The content hash joins requested and observed files by bytes.",
    "arc_model_identity_bridge_ready_score": "One requires every positive, negative, legacy, and wiring gate.",
    "random_seed": "A fixed seed makes the deterministic fixture identity reproducible.",
    "reproducibility_checksum": "A canonical digest detects later artifact changes.",
    "gate_check_summary": "The first exact failed check makes a blocked verdict actionable.",
    "verifier_is_oracle": "False states that this receipt validates identity, not ARC correctness.",
    "verdict_class": "A closed class gives the terminal result one machine-readable meaning.",
    "honest_verdict": "A class-consistent prefix prevents a blocked result from reading as success.",
}

SOURCE_PATHS = (
    Path("results/experiment_7025_belief_shadow_live_trace.json"),
    Path("results/experiment_7027_v615_capstone.json"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/agentic/arc_belief_shadow_live_trace.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("tests/python/test_arc_eval_provenance_contract_20260905.py"),
    Path("tests/python/test_experiment_7025_belief_shadow_live_trace.py"),
    Path("tests/python/test_arc_model_identity_bridge_20260905.py"),
    Path("openspec/capabilities/arc-agi/spec.md"),
    Path("openspec/capabilities/research-reporting/spec.md"),
)


def sha256_file(path: str | Path) -> str:
    """Hash one source without loading a large file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash stable JSON bytes for cross-process artifact validation."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every artifact field except the self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def gate_row(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record both sides of one exact precondition or readiness check."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "terminal": True,
    }


def gate_check_summary(checks: list[JsonDict]) -> JsonDict:
    """Keep every check and name the first failure without inference."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed["check"],
        "expected_value": True if failed is None else failed["expected_value"],
        "observed_value": True if failed is None else failed["observed_value"],
        "checks": checks,
    }


def _load_json(path: Path) -> JsonDict:
    """Read one source artifact as an object or return an explicit absence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _exp7025_snapshot_path(artifact: Mapping[str, Any]) -> Path | None:
    """Find the named snapshot link that reaches Exp7025's recorded blob."""

    specs = artifact.get("MODEL_SPECS")
    spec = specs[0] if isinstance(specs, list) and specs else {}
    if not isinstance(spec, Mapping) or spec.get("hf_id") != HF_ID:
        return None
    observed = Path(str(spec.get("model_path") or ""))
    model_root = observed.parent.parent if observed.parent.name == "blobs" else None
    if model_root is None or model_root.name != "models--" + HF_ID.replace("/", "--"):
        return None
    for candidate in sorted((model_root / "snapshots").glob("*/*.gguf")):
        try:
            if candidate.is_symlink() and candidate.resolve(strict=True) == observed.resolve(strict=True):
                return candidate
        except OSError:
            continue
    return None


def collect_preconditions(root: Path, output_path: Path, work_dir: Path) -> tuple[list[JsonDict], JsonDict]:
    """Check all required sources and writable targets before fixture work."""

    checks = [gate_row(path.as_posix(), True, (root / path).is_file()) for path in SOURCE_PATHS]
    exp7025 = _load_json(root / SOURCE_PATHS[0])
    snapshot = _exp7025_snapshot_path(exp7025)
    checks.append(gate_row("exp7025_named_cached_model_path", True, snapshot is not None))
    for label, path in (
        ("code_path_writable", root / "python/carnot/agentic/arc_eval_provenance.py"),
        ("test_path_writable", root / "tests/python/test_arc_model_identity_bridge_20260905.py"),
        ("artifact_path_writable", output_path.parent),
        ("fixture_path_writable", work_dir),
    ):
        target = path if path.is_dir() else path.parent
        checks.append(gate_row(label, True, target.is_dir() and os.access(target, os.W_OK)))
    evidence = {
        "exp7025": exp7025,
        "snapshot_path": None if snapshot is None else str(snapshot),
    }
    return checks, evidence


def _make_snapshot_fixture(directory: Path) -> tuple[JsonDict, Path]:
    """Create the snapshot-symlink-to-hash-blob topology seen in Exp7025."""

    model_root = directory / "hub" / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    payload = b"GGUF-shaped Exp7025 identity fixture\n"
    digest = hashlib.sha256(payload).hexdigest()
    blob = model_root / "blobs" / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(payload)
    requested = model_root / "snapshots" / REVISION / FILENAME
    requested.parent.mkdir(parents=True, exist_ok=True)
    requested.symlink_to(Path("../../blobs") / digest)
    spec = {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": HF_ID,
        "gpu": 0,
        "model_path": str(requested),
        "model_filename": FILENAME,
        "revision": REVISION,
        "model_file_hash": "sha256:" + digest,
        "resolved_via": "cached_sota_pair",
    }
    return spec, blob


def _legacy_input(**changes: Any) -> ArcEvalProvenanceInput:
    """Build a complete v1 row so compatibility uses the real validator."""

    source = ArcEvalProvenanceInput(
        inference_substrate="local_gguf_cuda",
        gpu_uuid="GPU-70300000-0000-0000-0000-000000000001",
        gpu_model="NVIDIA RTX 3090",
        cuda_device=0,
        model_repository=HF_ID,
        model_filename=FILENAME,
        model_hash="sha256:" + "1" * 64,
        n_ctx=4096,
        server_binary="/opt/llama.cpp/llama-server",
        server_binary_hash="sha256:" + "2" * 64,
        server_command_hash="sha256:" + "3" * 64,
        endpoint="http://127.0.0.1:17030",
        port=17030,
        lease_id="lease-7030",
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


def _negative_rows(work_dir: Path) -> list[JsonDict]:
    """Run every REQ-ARC-7030 rejection against a fresh filesystem fixture."""

    rows: list[JsonDict] = []
    cases = (
        "wrong_hash",
        "wrong_hub",
        "wrong_revision",
        "missing_requested_file",
        "broken_symlink",
        "directory_path",
        "misleading_gguf_basename",
        "observed_blob_not_reachable",
    )
    for index, case in enumerate(cases):
        spec, blob = _make_snapshot_fixture(work_dir / f"negative-{index}")
        observed = blob
        requested = Path(spec["model_path"])
        if case == "wrong_hash":
            spec["model_file_hash"] = "sha256:" + "0" * 64
        elif case == "wrong_hub":
            spec["hf_id"] = "unsloth/Not-Qwen-GGUF"
        elif case == "wrong_revision":
            spec["revision"] = "wrong-revision"
        elif case == "missing_requested_file":
            requested.unlink()
        elif case == "broken_symlink":
            requested.unlink()
            requested.symlink_to("../../blobs/missing")
        elif case == "directory_path":
            requested.unlink()
            requested.mkdir()
        elif case == "misleading_gguf_basename":
            spec["model_filename"] = "alias.gguf"
        else:
            observed = work_dir / f"negative-{index}" / "foreign" / "blobs" / blob.name
            observed.parent.mkdir(parents=True)
            observed.write_bytes(blob.read_bytes())
        accepted = True
        error = None
        try:
            build_arc_model_identity_receipt(
                selected_model_spec=spec,
                observed_server_model_path=str(observed),
            )
        except (TypeError, ValueError) as exc:
            accepted = False
            error = str(exc)
        rows.append({"case": case, "accepted": accepted, "error": error, "terminal": True})
    return rows


def _blocked_artifact(
    *, execution_date: str, duration_s: float, checks: list[JsonDict], source_hashes: JsonDict
) -> JsonDict:
    """Return every required field when source evidence is unavailable."""

    summary = gate_check_summary(checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": execution_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": duration_s,
        "source_artifact_hashes": source_hashes,
        "rows": [],
        "identity_schema_version": ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
        "exp7025_reproduction_rows": [],
        "requested_identity_rows": [],
        "observed_identity_rows": [],
        "hash_join_rows": [],
        "hub_revision_rows": [],
        "positive_fixture_rows": [],
        "negative_fixture_rows": [],
        "legacy_compatibility_rows": [],
        "producer_wiring_rows": [],
        "consumer_wiring_rows": [],
        "command_receipt_rows": [],
        "requested_model_path": None,
        "requested_model_filename": None,
        "requested_hf_id": None,
        "requested_revision": None,
        "observed_server_model_path": None,
        "resolved_model_path": None,
        "model_file_hash": None,
        "arc_model_identity_bridge_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_arc_model_identity_bridge:" + str(summary["failed_check"]),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(root: Path, *, execution_date: str, work_dir: Path) -> JsonDict:
    """Run deterministic positive, negative, legacy, and production wiring checks."""

    started = time.perf_counter()
    output_path = root / RESULT_RELATIVE_PATH
    work_dir.mkdir(parents=True, exist_ok=True)
    checks, evidence = collect_preconditions(root, output_path, work_dir)
    source_hashes = {
        path.as_posix(): sha256_file(root / path) for path in SOURCE_PATHS if (root / path).is_file()
    }
    if any(row["passed"] is not True for row in checks):
        return _blocked_artifact(
            execution_date=execution_date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            source_hashes=source_hashes,
        )

    spec, blob = _make_snapshot_fixture(work_dir / "positive")
    identity = build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(blob),
    )
    current = build_arc_eval_provenance(
        replace(
            _legacy_input(
                model_filename=identity["requested_model_filename"],
                model_hash=identity["model_file_hash"],
            ),
            schema_version=ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
            **identity,
        )
    )
    current_valid = validate_arc_eval_provenance(current).valid
    negative_rows = _negative_rows(work_dir)
    legacy = build_arc_eval_provenance(_legacy_input())
    legacy_valid = validate_arc_eval_provenance(json.loads(json.dumps(legacy))).valid

    builder_source = inspect.getsource(provenance.build_arc_eval_provenance_for_policy)
    validator_source = inspect.getsource(provenance.validate_arc_eval_provenance)
    shadow_source = (root / "python/carnot/agentic/arc_belief_shadow_live_trace.py").read_text()
    submitted_source = (root / "scripts/arc_leaderboard_eval.py").read_text()
    producer_rows = [
        {
            "producer": "build_arc_eval_provenance_for_policy",
            "shared_bridge_called": "build_arc_model_identity_receipt" in builder_source,
            "terminal": True,
        },
        {
            "producer": "arc_belief_shadow_live_trace",
            "shared_bridge_called": "build_arc_model_identity_receipt" in shadow_source,
            "terminal": True,
        },
        {
            "producer": "arc_leaderboard_eval",
            "canonical_builder_called": "build_arc_eval_provenance_for_policy" in submitted_source,
            "terminal": True,
        },
    ]
    consumer_rows = [
        {
            "consumer": "validate_arc_eval_provenance",
            "shared_bridge_recomputed": "build_arc_model_identity_receipt" in validator_source,
            "legacy_version_explicit": "ARC_EVAL_PROVENANCE_SCHEMA_VERSION" in validator_source,
            "current_version_explicit": "ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2" in validator_source,
            "terminal": True,
        }
    ]
    exp7025 = evidence["exp7025"]
    exp_spec = exp7025["MODEL_SPECS"][0]
    incident = exp7025.get("gate_check_summary", {}).get("observed_value")
    exp7025_rows = [
        {
            "artifact": SOURCE_PATHS[0].as_posix(),
            "requested_model_path": evidence["snapshot_path"],
            "observed_server_model_path": exp_spec["model_path"],
            "model_file_hash": exp_spec["model_file_hash"],
            "observed_blob_extensionless": Path(exp_spec["model_path"]).suffix == "",
            "incident_observed": incident,
            "topology_reproduced": True,
            "terminal": True,
        }
    ]
    positive_rows = [
        {
            "case": "exp7025_snapshot_symlink_to_extensionless_blob",
            "accepted": current_valid,
            "schema_version": current["schema_version"],
            "terminal": True,
        }
    ]
    hash_rows = [
        {
            "requested_hash": sha256_file(identity["requested_model_path"]),
            "observed_hash": sha256_file(identity["observed_server_model_path"]),
            "recorded_hash": identity["model_file_hash"],
            "passed": (
                sha256_file(identity["requested_model_path"])
                == sha256_file(identity["observed_server_model_path"])
                == identity["model_file_hash"]
            ),
            "terminal": True,
        }
    ]
    hub_rows = [
        {
            "requested_hf_id": identity["requested_hf_id"],
            "selected_hf_id": spec["hf_id"],
            "requested_revision": identity["requested_revision"],
            "selected_revision": spec["revision"],
            "passed": identity["requested_hf_id"] == spec["hf_id"]
            and identity["requested_revision"] == spec["revision"],
            "terminal": True,
        }
    ]
    readiness_checks = [
        *checks,
        gate_row("positive_fixture", True, all(row["accepted"] for row in positive_rows)),
        gate_row("negative_fixtures_fail_closed", True, all(not row["accepted"] for row in negative_rows)),
        gate_row("legacy_valid_rows_readable", True, legacy_valid),
        gate_row(
            "producer_wiring",
            True,
            all(row.get("shared_bridge_called", row.get("canonical_builder_called")) is True for row in producer_rows),
        ),
        gate_row(
            "consumer_wiring",
            True,
            all(
                row["shared_bridge_recomputed"]
                and row["legacy_version_explicit"]
                and row["current_version_explicit"]
                for row in consumer_rows
            ),
        ),
        gate_row("hash_join", True, all(row["passed"] for row in hash_rows)),
        gate_row("hub_revision_join", True, all(row["passed"] for row in hub_rows)),
    ]
    ready = all(row["passed"] is True for row in readiness_checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": execution_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": readiness_checks,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": time.perf_counter() - started,
        "source_artifact_hashes": source_hashes,
        "rows": [
            {"check": "positive", "row_count": len(positive_rows), "passed": current_valid, "terminal": True},
            {"check": "negative", "row_count": len(negative_rows), "passed": all(not row["accepted"] for row in negative_rows), "terminal": True},
            {"check": "legacy", "row_count": 1, "passed": legacy_valid, "terminal": True},
            {"check": "wiring", "row_count": len(producer_rows) + len(consumer_rows), "passed": ready, "terminal": True},
        ],
        "identity_schema_version": ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
        "exp7025_reproduction_rows": exp7025_rows,
        "requested_identity_rows": [
            {key: identity[key] for key in ("requested_model_path", "requested_model_filename", "requested_hf_id", "requested_revision")}
        ],
        "observed_identity_rows": [
            {key: identity[key] for key in ("observed_server_model_path", "resolved_model_path", "model_file_hash")}
        ],
        "hash_join_rows": hash_rows,
        "hub_revision_rows": hub_rows,
        "positive_fixture_rows": positive_rows,
        "negative_fixture_rows": negative_rows,
        "legacy_compatibility_rows": [
            {
                "schema_version": legacy["schema_version"],
                "accepted": legacy_valid,
                "current_fields_inferred": any(key in legacy for key in provenance.ARC_MODEL_IDENTITY_KEYS),
                "terminal": True,
            }
        ],
        "producer_wiring_rows": producer_rows,
        "consumer_wiring_rows": consumer_rows,
        "command_receipt_rows": [
            {
                "command": "build_arc_model_identity_receipt(Exp7025-shaped fixture)",
                "returncode": 0 if current_valid else 1,
                "terminal": True,
            },
            {
                "command": "validate_arc_eval_provenance(v1 and v2)",
                "returncode": 0 if current_valid and legacy_valid else 1,
                "terminal": True,
            },
        ],
        **identity,
        "arc_model_identity_bridge_ready_score": 1 if ready else 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_check_summary(readiness_checks),
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "partial",
        "honest_verdict": (
            "complete_positive_arc_model_identity_bridge_ready"
            if ready
            else "partial_arc_model_identity_bridge_not_ready"
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Any) -> list[str]:
    """Recompute the terminal artifact gates without trusting its ready score."""

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
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate_invalid")
    if not isinstance(artifact.get("duration_s"), (int, float)) or artifact.get("duration_s", -1) < 0:
        errors.append("duration_s_invalid")
    if artifact.get("identity_schema_version") != ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2:
        errors.append("identity_schema_version_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")

    score = artifact.get("arc_model_identity_bridge_ready_score")
    if not isinstance(score, int) or isinstance(score, bool) or score not in (0, 1):
        errors.append("ready_score_invalid")
    summary = artifact.get("gate_check_summary")
    verdict_class = artifact.get("verdict_class")
    verdict = artifact.get("honest_verdict")
    if verdict_class == "blocked":
        if score != 0:
            errors.append("blocked_ready_score_invalid")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            errors.append("blocked_gate_check_summary_invalid")
        if not str(verdict).startswith("blocked_"):
            errors.append("blocked_verdict_invalid")
        if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
            errors.append("reproducibility_checksum_mismatch")
        return list(dict.fromkeys(errors))

    positive = artifact.get("positive_fixture_rows")
    negative = artifact.get("negative_fixture_rows")
    legacy = artifact.get("legacy_compatibility_rows")
    producers = artifact.get("producer_wiring_rows")
    consumers = artifact.get("consumer_wiring_rows")
    hash_rows = artifact.get("hash_join_rows")
    hub_rows = artifact.get("hub_revision_rows")
    if not isinstance(positive, list) or not positive or any(row.get("accepted") is not True for row in positive):
        errors.append("positive_fixture_missing")
    if not isinstance(negative, list) or len(negative) != 8 or any(row.get("accepted") is not False for row in negative):
        errors.append("negative_fixture_accepted")
    if not isinstance(legacy, list) or not legacy or any(
        row.get("accepted") is not True or row.get("current_fields_inferred") is not False for row in legacy
    ):
        errors.append("legacy_compatibility_invalid")
    if not isinstance(producers, list) or len(producers) != 3 or any(
        row.get("shared_bridge_called", row.get("canonical_builder_called")) is not True for row in producers
    ):
        errors.append("producer_wiring_invalid")
    if not isinstance(consumers, list) or not consumers or any(
        row.get("shared_bridge_recomputed") is not True
        or row.get("legacy_version_explicit") is not True
        or row.get("current_version_explicit") is not True
        for row in consumers
    ):
        errors.append("consumer_wiring_invalid")
    if not isinstance(hash_rows, list) or not hash_rows or any(row.get("passed") is not True for row in hash_rows):
        errors.append("hash_join_invalid")
    if not isinstance(hub_rows, list) or not hub_rows or any(row.get("passed") is not True for row in hub_rows):
        errors.append("hub_revision_invalid")

    ready_evidence = not any(
        name in errors
        for name in (
            "positive_fixture_missing",
            "negative_fixture_accepted",
            "legacy_compatibility_invalid",
            "producer_wiring_invalid",
            "consumer_wiring_invalid",
            "hash_join_invalid",
            "hub_revision_invalid",
        )
    )
    if score == 1 and not ready_evidence:
        errors.append("ready_score_not_supported")
    if score == 1:
        if not isinstance(summary, Mapping) or summary.get("passed") is not True:
            errors.append("gate_check_summary_invalid")
        if verdict_class != "positive" or not str(verdict).startswith("complete_positive_"):
            errors.append("positive_verdict_invalid")
    elif verdict_class != "partial" or not str(verdict).startswith("partial_"):
        errors.append("partial_verdict_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish one terminal artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp7030 artifact: " + ";".join(errors))
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
    """Build and write the artifact with temporary GGUF-shaped fixtures."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp7030-") as name:
        artifact = build_artifact(root, execution_date=execution_date, work_dir=Path(name))
        write_artifact(output_path, artifact)
    return artifact
