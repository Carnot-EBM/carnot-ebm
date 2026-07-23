"""Exp5825 certified adaptive memory event/state contract.

Spec refs: REQ-LEARN-5825, SCENARIO-LEARN-5825-ADAPTERS,
SCENARIO-LEARN-5825-FAIL-CLOSED, SCENARIO-LEARN-5825-ARTIFACT.

This module builds one deterministic event and state vocabulary over the
credited Exp5761, Exp5762, Exp5763, and Exp5785 artifacts. It does not rerun
benchmark generation, learner updates, model inference, or scientific scoring.
The contract hashes sealed upstream rows and receipts, adapts them into a
canonical boundary, and rejects chronology, visibility, oracle, collision, and
state-replay defects before downstream memory code can consume the history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5825_certified_adaptive_memory_contract.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5825_certified_adaptive_memory_contract.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5825_certified_adaptive_memory_contract.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")

EXP5761_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5761_exact_constraint_acquisition_benchmark.json"
)
EXP5761_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5761_exact_constraint_acquisition_benchmark.instances.jsonl"
)
EXP5762_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5762_query_driven_constraint_lifecycle.json"
)
EXP5763_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5763_dependent_task_constraint_acquisition.json"
)
EXP5785_ARTIFACT_RELATIVE_PATH = Path("results/experiment_5785_hardness_surface_fixture.json")
EXP5785_ROWS_RELATIVE_PATH = Path("results/experiment_5785_hardness_surface_fixture.rows.jsonl")

SCHEMA = "carnot.experiment_5825.certified_adaptive_memory_contract.v1"
EVENT_SCHEMA_VERSION = SCHEMA + ".event.v1"
STATE_SCHEMA_VERSION = SCHEMA + ".state.v1"
EXPERIMENT = 5825
EXPERIMENT_ID = "experiment_5825_certified_adaptive_memory_contract"
MILESTONE = "2026.07.519"
RUN_DATE = "20260723"
RANDOM_SEED = 5825
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STATE_IDENTITY_RULE = (
    "state_id=state::<schema_version>::<source_adapter>::<sequence>::<state_hash_prefix>"
)
EVENT_IDENTITY_RULE = (
    "event_id=event::<schema_version>::<source_adapter>::<sequence>::<event_type>::<payload_hash_prefix>"
)

SPEC_REFS = (
    "REQ-LEARN-5825",
    "SCENARIO-LEARN-5825-ADAPTERS",
    "SCENARIO-LEARN-5825-FAIL-CLOSED",
    "SCENARIO-LEARN-5825-ARTIFACT",
)

REQUIRED_EVENT_TYPES = (
    "observation",
    "exact_membership_outcome",
    "minimal_core_evidence",
    "constraint_birth",
    "quarantine",
    "promotion",
    "supersession",
    "recurrence",
    "collision_split",
    "rollback",
    "protected_prefix_replay",
    "sealed_future_evaluation",
)

VISIBILITY_VALUES = ("train", "dev", "science", "calibration", "future_test")

REQUIRED_ADVERSARIAL_CASES = (
    "leakage",
    "forged_oracle_labels",
    "collision_without_split",
    "stale_supersession",
    "rollback_mismatch",
    "missing_protected_prefix_evidence",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_artifact_hashes",
    "canonical_event_schema",
    "canonical_state_schema",
    "adapter_round_trip_receipts",
    "chronology_and_visibility_checks",
    "adversarial_contract_results",
    "adaptive_memory_contract_ready_score",
    "schema_errors",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal contract state distinguishes reusable infrastructure from an incomplete schema.",
    "preconditions_checked": "Upstream hashes, verdicts, validators, resources, and writable paths prevent fabricated readiness.",
    "upstream_artifact_hashes": "Exact hashes bind the contract to the credited acquisition evidence.",
    "canonical_event_schema": "One typed event vocabulary prevents each experiment from redefining learning history.",
    "canonical_state_schema": "Versioned immutable state identities make promotion, supersession, and rollback auditable.",
    "adapter_round_trip_receipts": "Round-trip hashes prove existing evidence was adapted rather than regenerated.",
    "chronology_and_visibility_checks": "Monotone sequence and split visibility prevent future-label leakage.",
    "adversarial_contract_results": "Forged, colliding, stale, and rollback-mismatched events must fail closed.",
    "adaptive_memory_contract_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5826 to build on this boundary.",
    "schema_errors": "Explicit errors prevent a partial contract from being narrated as ready.",
    "duration_s": "Measured wall time exposes bootstrap-only execution.",
    "inference_substrate": "`aggregation_from_upstream_artifacts` declares no LLM or new scientific inference.",
    "verifier_is_oracle": "True records that exact solvers and validators remain circular correctness authority.",
    "field_provenance": "Every field maps to an upstream artifact, schema rule, or adversarial test.",
    "test_commands": "Commands document round-trip, leakage, schema, and failure-closed tests.",
    "test_exit_codes": "Exit codes prevent failed contract checks from becoming readiness.",
    "reproducibility_checksum": "A checksum detects later adapter, schema, or upstream drift.",
    "honest_verdict": "A `complete:` or `blocked:` prefix provides a terminal infrastructure outcome.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned artifact schema id for Exp5825.",
    "experiment": "Numeric experiment id binds the artifact to the requested run.",
    "experiment_id": "Stable slug ties the artifact to the module and result path.",
    "milestone": "Binds the contract to milestone .519.",
    "run_date": "Operator-requested run date for deterministic artifact metadata.",
    "random_seed": "Deterministic metadata for a no-learning aggregation artifact.",
    "spec_refs": "OpenSpec anchors for the contract and adversarial scenarios.",
    "result_path": "Declares the intended terminal JSON path.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5825_certified_adaptive_memory_contract.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5825_certified_adaptive_memory_contract.py "
    "-m pytest tests/python/test_experiment_5825_certified_adaptive_memory_contract.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5825_certified_adaptive_memory_contract.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python/test_experiment_5824_v519_source_delta_ingestion.py "
    "-q --no-cov -n 0",
)

UPSTREAM_PATHS: dict[str, Path] = {
    "exp5761_artifact": EXP5761_ARTIFACT_RELATIVE_PATH,
    "exp5761_instances": EXP5761_ROWS_RELATIVE_PATH,
    "exp5762_artifact": EXP5762_ARTIFACT_RELATIVE_PATH,
    "exp5763_artifact": EXP5763_ARTIFACT_RELATIVE_PATH,
    "exp5785_artifact": EXP5785_ARTIFACT_RELATIVE_PATH,
    "exp5785_rows": EXP5785_ROWS_RELATIVE_PATH,
}


class ContractValidationError(ValueError):
    """Raised when canonical event/state evidence fails the contract."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for stable text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without trusting filesystem metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.is_file() else "missing"


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required: {path}")
        rows.append(dict(payload))
    return rows


def _memory_probe() -> JsonDict:
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover - host fallback.
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _disk_probe(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _output_path_receipt(result_path: Path) -> JsonDict:
    parent = result_path.parent
    return {
        "result_path": str(result_path),
        "parent_exists": parent.exists(),
        "parent_writable": parent.exists() and os.access(parent, os.W_OK),
        "result_writable": (parent.exists() and os.access(parent, os.W_OK))
        and (not result_path.exists() or os.access(result_path, os.W_OK)),
    }


def _terminal_verdict_ok(value: Any) -> bool:
    verdict = str(value)
    return verdict.startswith(("complete:", "blocked:"))


def _load_upstream_bundle(root: Path) -> tuple[dict[str, JsonDict], dict[str, list[JsonDict]]]:
    artifacts = {
        "exp5761": _read_json(root / EXP5761_ARTIFACT_RELATIVE_PATH),
        "exp5762": _read_json(root / EXP5762_ARTIFACT_RELATIVE_PATH),
        "exp5763": _read_json(root / EXP5763_ARTIFACT_RELATIVE_PATH),
        "exp5785": _read_json(root / EXP5785_ARTIFACT_RELATIVE_PATH),
    }
    rows = {
        "exp5761_instances": _read_jsonl(root / EXP5761_ROWS_RELATIVE_PATH),
        "exp5785_rows": _read_jsonl(root / EXP5785_ROWS_RELATIVE_PATH),
    }
    return artifacts, rows


def collect_preconditions(
    root: Path = REPO_ROOT,
    *,
    result_path: Path | None = None,
) -> JsonDict:
    """Verify sealed upstream artifacts, resources, and output path gates."""

    root = Path(root)
    result_path = Path(result_path or (root / RESULT_RELATIVE_PATH))
    upstream_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    blocked: list[str] = []
    if any(value == "missing" for value in upstream_hashes.values()):
        blocked.append("missing_upstream_artifact")

    artifacts: dict[str, JsonDict] = {}
    rows: dict[str, list[JsonDict]] = {}
    corrupt: list[str] = []
    if not blocked:
        try:
            artifacts, rows = _load_upstream_bundle(root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupt.append(type(exc).__name__)
            blocked.append("corrupt_upstream_artifact")

    verdicts: JsonDict = {}
    validators: JsonDict = {}
    row_counts = {"exp5761_instances": 0, "exp5785_rows": 0}
    split_hashes: JsonDict = {}
    if artifacts and rows:
        verdicts = {
            name: {
                "status": artifact.get("status"),
                "honest_verdict": artifact.get("honest_verdict"),
                "terminal_prefix_ok": _terminal_verdict_ok(artifact.get("honest_verdict")),
                "verifier_is_oracle": artifact.get("verifier_is_oracle"),
            }
            for name, artifact in artifacts.items()
        }
        if not all(receipt["status"] == "complete" for receipt in verdicts.values()):
            blocked.append("upstream_status_not_complete")
        if not all(receipt["terminal_prefix_ok"] is True for receipt in verdicts.values()):
            blocked.append("upstream_honest_verdict_not_terminal")
        if not all(receipt["verifier_is_oracle"] is True for receipt in verdicts.values()):
            blocked.append("upstream_verifier_not_oracle")
        row_counts = {
            "exp5761_instances": len(rows["exp5761_instances"]),
            "exp5785_rows": len(rows["exp5785_rows"]),
        }
        if row_counts["exp5761_instances"] != int(artifacts["exp5761"].get("instance_count", -1)):
            blocked.append("exp5761_row_count_mismatch")
        if artifacts["exp5785"].get("row_file_sha256") != upstream_hashes["exp5785_rows"]:
            blocked.append("exp5785_row_hash_mismatch")
        validators = {
            "exp5761_solver_versions": artifacts["exp5761"].get("solver_versions") or {},
            "exp5762_oracle_boundary": artifacts["exp5762"].get("inference_substrate"),
            "exp5763_query_label_verifiers": sorted(
                {
                    str(receipt.get("verifier"))
                    for receipt in artifacts["exp5763"].get("query_label_receipts", [])
                }
            ),
            "exp5785_validator_receipt_count": len(
                artifacts["exp5785"].get("exact_validator_receipts", [])
            ),
        }
        if not validators["exp5761_solver_versions"]:
            blocked.append("missing_exact_validator_versions")
        split_hashes = {
            "exp5761": (artifacts["exp5761"].get("split_manifest") or {}).get("split_hashes") or {},
            "exp5762_science": artifacts["exp5762"].get("science_split_hash"),
            "exp5763_stream": artifacts["exp5763"].get("stream_root_hash"),
            "exp5785": (
                artifacts["exp5785"].get("chronological_split_receipts") or {}
            ).get("split_hashes")
            or {},
        }
        if not split_hashes["exp5761"] or not split_hashes["exp5785"]:
            blocked.append("missing_split_hashes")

    memory = _memory_probe()
    disk = _disk_probe(root)
    output_paths = _output_path_receipt(result_path)
    if memory["ok"] is not True:
        blocked.append("insufficient_free_ram")
    if disk["ok"] is not True:
        blocked.append("insufficient_free_disk")
    if output_paths["result_writable"] is not True:
        blocked.append("output_path_not_writable")

    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "upstream_artifact_hashes": upstream_hashes,
        "upstream_terminal_verdicts": verdicts,
        "exact_validator_versions": validators,
        "upstream_row_counts": row_counts,
        "split_hashes": split_hashes,
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output_paths,
        "llm_calls_made": 0,
        "new_learning_performed": False,
        "corrupt_upstream_errors": corrupt,
        "preconditions_ready": not blocked,
        "blocked_reasons": blocked,
    }


def _short_hash(value: str) -> str:
    return value.replace("sha256:", "")[:16] if value else "missing"


def _axes(
    *,
    family: Any = "",
    hardness: Any = "",
    surface: Any = "",
    change: str = "",
) -> JsonDict:
    return {
        "family": str(family or "unknown"),
        "hardness": str(hardness or "unspecified"),
        "surface": str(surface or "unspecified"),
        "change": change,
    }


def _source_hash(row: Mapping[str, Any]) -> str:
    for key in ("row_hash", "receipt_hash", "ledger_row_hash", "transition_hash", "query_hash"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith("sha256:"):
            return value
    return sha256_json(row)


def _state_payload_hash(state: Mapping[str, Any]) -> str:
    stable = _copy_json(state)
    stable["state_id"] = ""
    stable["state_hash"] = ""
    return sha256_json(stable)


def expected_state_id(state: Mapping[str, Any]) -> str:
    """Recompute the canonical state id from schema, adapter, sequence, and hash."""

    return (
        f"state::{STATE_SCHEMA_VERSION}::{state['source_adapter']}::"
        f"{int(state['state_sequence_index']):06d}::{_short_hash(str(state['state_hash']))}"
    )


def canonical_state_hash(state: Mapping[str, Any]) -> str:
    """Replay the immutable state hash from all state fields except identity hash."""

    return _state_payload_hash(state)


def make_state(
    *,
    source_adapter: str,
    sequence: int,
    state_label: str,
    source_artifact: str,
    source_artifact_hash: str,
    source_hash: str,
    visibility: str,
    axes: Mapping[str, Any],
    parent_state_hash: str = "",
    mutation_receipt_hash: str = "",
    lifecycle_operation: str = "observe",
) -> JsonDict:
    """Build a canonical immutable state shell for one source receipt."""

    state: JsonDict = {
        "schema": STATE_SCHEMA_VERSION,
        "state_id": "",
        "state_version": "adaptive_memory_state_v1",
        "state_sequence_index": int(sequence),
        "source_adapter": source_adapter,
        "state_label": state_label,
        "parent_state_hash": parent_state_hash,
        "mutation_receipt_hash": mutation_receipt_hash,
        "lifecycle_operation": lifecycle_operation,
        "source_artifact": source_artifact,
        "source_artifact_hash": source_artifact_hash,
        "source_row_hash": source_hash,
        "visibility": visibility,
        "axes": _copy_json(axes),
        "immutable": True,
        "state_hash": "",
    }
    state["state_hash"] = canonical_state_hash(state)
    state["state_id"] = expected_state_id(state)
    return state


def canonical_event_hash(event: Mapping[str, Any]) -> str:
    """Replay the canonical event hash with id/hash fields blanked."""

    stable = _copy_json(event)
    stable["event_id"] = ""
    stable["event_hash"] = ""
    return sha256_json(stable)


def expected_event_id(event: Mapping[str, Any]) -> str:
    """Recompute the canonical event id from schema, adapter, sequence, type, and payload."""

    return (
        f"event::{EVENT_SCHEMA_VERSION}::{event['source_adapter']}::"
        f"{int(event['causal_sequence_index']):06d}::{event['event_type']}::"
        f"{_short_hash(str(event['payload_hash']))}"
    )


def make_event(
    *,
    event_type: str,
    source_adapter: str,
    sequence: int,
    source_artifact: str,
    source_artifact_hash: str,
    source_hash: str,
    visibility: str,
    axes: Mapping[str, Any],
    payload: Mapping[str, Any],
    parent_state: Mapping[str, Any],
    resulting_state: Mapping[str, Any],
    oracle_provenance: Mapping[str, Any],
) -> JsonDict:
    """Build one canonical event with stable ids and payload/source hashes."""

    if event_type not in REQUIRED_EVENT_TYPES:
        raise ValueError(f"unsupported_event_type:{event_type}")
    payload_copy = _copy_json(payload)
    payload_hash = sha256_json(payload_copy)
    event: JsonDict = {
        "schema": EVENT_SCHEMA_VERSION,
        "event_id": "",
        "event_type": event_type,
        "source_adapter": source_adapter,
        "causal_sequence_index": int(sequence),
        "parent_state_id": str(parent_state["state_id"]),
        "resulting_state_id": str(resulting_state["state_id"]),
        "family_hardness_surface_change_axes": _copy_json(axes),
        "visibility": visibility,
        "payload": payload_copy,
        "payload_hash": payload_hash,
        "source_artifact": source_artifact,
        "source_artifact_hash": source_artifact_hash,
        "source_row_hash": source_hash,
        "source_receipt_hash": str(payload_copy.get("receipt_hash") or source_hash),
        "oracle_provenance": _copy_json(oracle_provenance),
        "state_mutation_receipt_hash": str(resulting_state.get("mutation_receipt_hash") or ""),
        "event_hash": "",
    }
    event["event_hash"] = canonical_event_hash(event)
    event["event_id"] = expected_event_id(event)
    return event


def _base_oracle(source: str, *, minted_before_learner: bool = True) -> JsonDict:
    return {
        "authority": "exact_solver_or_validator",
        "source": source,
        "label_minted_before_learner": minted_before_learner,
        "hidden_label_access": False,
        "forged_label": False,
    }


def _add_event(
    events: list[JsonDict],
    states: dict[str, JsonDict],
    *,
    event_type: str,
    source_adapter: str,
    sequence: int,
    source_artifact: str,
    source_artifact_hash: str,
    source_hash: str,
    visibility: str,
    axes: Mapping[str, Any],
    payload: Mapping[str, Any],
    operation: str,
    oracle: Mapping[str, Any],
) -> int:
    parent = make_state(
        source_adapter=source_adapter,
        sequence=sequence * 2,
        state_label="parent",
        source_artifact=source_artifact,
        source_artifact_hash=source_artifact_hash,
        source_hash=source_hash,
        visibility=visibility,
        axes=axes,
        lifecycle_operation="before_" + operation,
    )
    receipt_hash = str(payload.get("receipt_hash") or payload.get("ledger_row_hash") or source_hash)
    result = make_state(
        source_adapter=source_adapter,
        sequence=sequence * 2 + 1,
        state_label="result",
        source_artifact=source_artifact,
        source_artifact_hash=source_artifact_hash,
        source_hash=source_hash,
        visibility=visibility,
        axes=axes,
        parent_state_hash=str(parent["state_hash"]),
        mutation_receipt_hash=receipt_hash if operation != "observe" else "",
        lifecycle_operation=operation,
    )
    states[parent["state_id"]] = parent
    states[result["state_id"]] = result
    event = make_event(
        event_type=event_type,
        source_adapter=source_adapter,
        sequence=sequence,
        source_artifact=source_artifact,
        source_artifact_hash=source_artifact_hash,
        source_hash=source_hash,
        visibility=visibility,
        axes=axes,
        payload=payload,
        parent_state=parent,
        resulting_state=result,
        oracle_provenance=oracle,
    )
    events.append(event)
    return sequence + 1


def _adapter_receipt(
    *,
    adapter_id: str,
    source_artifact_hash: str,
    source_row_count: int = 0,
    source_event_count: int = 0,
    events: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
) -> JsonDict:
    event_hashes = [event["event_hash"] for event in events]
    state_hashes = [state["state_hash"] for state in states]
    return {
        "adapter_id": adapter_id,
        "source_artifact_hash": source_artifact_hash,
        "source_row_count": source_row_count,
        "source_event_count": source_event_count,
        "canonical_event_count": len(events),
        "canonical_state_count": len(states),
        "canonical_event_root_hash": sha256_json(event_hashes),
        "canonical_state_root_hash": sha256_json(state_hashes),
        "round_trip_ok": all(event["event_hash"] == canonical_event_hash(event) for event in events)
        and all(state["state_hash"] == canonical_state_hash(state) for state in states),
        "sample_event_hashes": event_hashes[:3],
    }


def _adapt_exp5761(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    upstream_hashes: Mapping[str, str],
    sequence: int,
) -> tuple[list[JsonDict], dict[str, JsonDict], int, JsonDict]:
    events: list[JsonDict] = []
    states: dict[str, JsonDict] = {}
    start = sequence
    source_hash = str(upstream_hashes["exp5761_artifact"])
    for row in rows:
        visibility = str(row["split"])
        row_hash = str(row["row_hash"])
        axes = _axes(family=row.get("family"), change="row_observed")
        sequence = _add_event(
            events,
            states,
            event_type="observation",
            source_adapter="exp5761",
            sequence=sequence,
            source_artifact=EXP5761_ROWS_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5761_instances"]),
            source_hash=row_hash,
            visibility=visibility,
            axes=axes,
            payload={
                "case_id": row["case_id"],
                "case_sequence_index": row["case_sequence_index"],
                "domain_artifact_hash": row["domain_artifact_hash"],
                "row_hash": row_hash,
            },
            operation="observe",
            oracle=_base_oracle("exp5761_exact_constraint_acquisition"),
        )
        for variant in row.get("variants", []):
            variant_hash = str(variant.get("expected_repair_hash") or _source_hash(variant))
            variant_axes = _axes(
                family=row.get("family"),
                change="minimal_core_" + str(variant.get("variant_kind")),
            )
            sequence = _add_event(
                events,
                states,
                event_type="minimal_core_evidence",
                source_adapter="exp5761",
                sequence=sequence,
                source_artifact=EXP5761_ROWS_RELATIVE_PATH.as_posix(),
                source_artifact_hash=str(upstream_hashes["exp5761_instances"]),
                source_hash=variant_hash,
                visibility=visibility,
                axes=variant_axes,
                payload={
                    "case_id": row["case_id"],
                    "variant_id": variant.get("variant_id"),
                    "variant_kind": variant.get("variant_kind"),
                    "expected_repair_hash": variant.get("expected_repair_hash"),
                    "distinguishing_query_hash": variant.get("distinguishing_query_hash"),
                    "minimal": (variant.get("distinguishing_query_receipt") or {}).get(
                        "minimal"
                    ),
                    "receipt_hash": variant_hash,
                },
                operation="minimal_core",
                oracle=_base_oracle("exp5761_exact_minimal_core_validator"),
            )
    adapter_events = events[start - start :]
    receipt = _adapter_receipt(
        adapter_id="exp5761",
        source_artifact_hash=source_hash,
        source_row_count=len(rows),
        source_event_count=len(rows) + sum(len(row.get("variants", [])) for row in rows),
        events=adapter_events,
        states=list(states.values()),
    )
    return events, states, sequence, receipt


def _adapt_exp5762(
    artifact: Mapping[str, Any],
    upstream_hashes: Mapping[str, str],
    sequence: int,
) -> tuple[list[JsonDict], dict[str, JsonDict], int, JsonDict]:
    events: list[JsonDict] = []
    states: dict[str, JsonDict] = {}
    for receipt in artifact.get("membership_query_receipts", []):
        axes = _axes(change="membership_query")
        sequence = _add_event(
            events,
            states,
            event_type="exact_membership_outcome",
            source_adapter="exp5762",
            sequence=sequence,
            source_artifact=EXP5762_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5762_artifact"]),
            source_hash=str(receipt["query_hash"]),
            visibility="science",
            axes=axes,
            payload={
                "query_id": receipt["query_id"],
                "episode_id": receipt["episode_id"],
                "assignment_hash": receipt["assignment_hash"],
                "oracle_accepts": receipt["oracle_accepts"],
                "receipt_hash": receipt["query_hash"],
            },
            operation="membership_query",
            oracle=_base_oracle("exp5762_exact_membership_oracle"),
        )
    for receipt in artifact.get("constraint_birth_receipts", []):
        source_hash = str(receipt["receipt_hash"])
        axes = _axes(change="constraint_birth")
        sequence = _add_event(
            events,
            states,
            event_type="constraint_birth",
            source_adapter="exp5762",
            sequence=sequence,
            source_artifact=EXP5762_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5762_artifact"]),
            source_hash=source_hash,
            visibility="science",
            axes=axes,
            payload={
                "episode_id": receipt["episode_id"],
                "constraint_hash": receipt["constraint_hash"],
                "pre_state_hash": receipt["pre_state_hash"],
                "post_state_hash": receipt["post_state_hash"],
                "receipt_hash": source_hash,
            },
            operation="birth",
            oracle=_base_oracle("exp5762_promotion_gate_validator"),
        )
        promotion_payload = {
            "episode_id": receipt["episode_id"],
            "constraint_hash": receipt["constraint_hash"],
            "promotion_gates": receipt.get("promotion_gates") or {},
            "receipt_hash": source_hash,
        }
        sequence = _add_event(
            events,
            states,
            event_type="promotion",
            source_adapter="exp5762",
            sequence=sequence,
            source_artifact=EXP5762_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5762_artifact"]),
            source_hash=source_hash,
            visibility="science",
            axes=_axes(change="constraint_promotion"),
            payload=promotion_payload,
            operation="promotion",
            oracle=_base_oracle("exp5762_protected_prefix_promotion_gate"),
        )
    for receipt in artifact.get("constraint_quarantine_receipts", []):
        sequence = _add_event(
            events,
            states,
            event_type="quarantine",
            source_adapter="exp5762",
            sequence=sequence,
            source_artifact=EXP5762_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5762_artifact"]),
            source_hash=str(receipt["receipt_hash"]),
            visibility="science",
            axes=_axes(change="quarantine"),
            payload={
                "episode_id": receipt["episode_id"],
                "quarantined_constraint_hash": receipt["quarantined_constraint_hash"],
                "pre_state_hash": receipt["pre_state_hash"],
                "post_state_hash": receipt["post_state_hash"],
                "receipt_hash": receipt["receipt_hash"],
            },
            operation="quarantine",
            oracle=_base_oracle("exp5762_contradiction_validator"),
        )
    for receipt in artifact.get("constraint_supersession_receipts", []):
        sequence = _add_event(
            events,
            states,
            event_type="supersession",
            source_adapter="exp5762",
            sequence=sequence,
            source_artifact=EXP5762_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5762_artifact"]),
            source_hash=str(receipt["receipt_hash"]),
            visibility="science",
            axes=_axes(change="supersession"),
            payload={
                "episode_id": receipt["episode_id"],
                "active_state_hash": receipt["active_state_hash"],
                "superseded_constraint_ids": receipt["superseded_constraint_ids"],
                "receipt_hash": receipt["receipt_hash"],
            },
            operation="supersession",
            oracle=_base_oracle("exp5762_supersession_validator"),
        )
    for receipt in artifact.get("constraint_lifecycle_ledger", []):
        sequence = _add_event(
            events,
            states,
            event_type="protected_prefix_replay",
            source_adapter="exp5762",
            sequence=sequence,
            source_artifact=EXP5762_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5762_artifact"]),
            source_hash=str(receipt["ledger_row_hash"]),
            visibility="science",
            axes=_axes(change="protected_prefix_replay"),
            payload={
                "episode_id": receipt["episode_id"],
                "restart_hash_matches": receipt["restart_hash_matches"],
                "rollback_hash_matches": receipt["rollback_hash_matches"],
                "protected_prefix_hash": receipt["final_state_hash"],
                "replay_passed": bool(
                    receipt["restart_hash_matches"] is True and receipt["rollback_hash_matches"] is True
                ),
                "receipt_hash": receipt["ledger_row_hash"],
            },
            operation="protected_prefix_replay",
            oracle=_base_oracle("exp5762_restart_rollback_replay"),
        )
    source_event_count = (
        len(artifact.get("membership_query_receipts", []))
        + len(artifact.get("constraint_birth_receipts", []))
        + len(artifact.get("constraint_quarantine_receipts", []))
        + len(artifact.get("constraint_supersession_receipts", []))
        + len(artifact.get("constraint_lifecycle_ledger", []))
    )
    receipt = _adapter_receipt(
        adapter_id="exp5762",
        source_artifact_hash=str(upstream_hashes["exp5762_artifact"]),
        source_event_count=source_event_count,
        events=events,
        states=list(states.values()),
    )
    return events, states, sequence, receipt


def _transition_event_type(operation: str) -> str:
    if operation == "add":
        return "constraint_birth"
    if operation == "quarantine":
        return "quarantine"
    if operation == "supersede":
        return "supersession"
    if operation == "rollback":
        return "rollback"
    return "recurrence"


def _adapt_exp5763(
    artifact: Mapping[str, Any],
    upstream_hashes: Mapping[str, str],
    sequence: int,
) -> tuple[list[JsonDict], dict[str, JsonDict], int, JsonDict]:
    events: list[JsonDict] = []
    states: dict[str, JsonDict] = {}
    for receipt in artifact.get("query_label_receipts", []):
        sequence = _add_event(
            events,
            states,
            event_type="exact_membership_outcome",
            source_adapter="exp5763",
            sequence=sequence,
            source_artifact=EXP5763_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5763_artifact"]),
            source_hash=str(receipt["receipt_hash"]),
            visibility="science",
            axes=_axes(change="dependent_membership_query"),
            payload={
                "session_id": receipt["session_id"],
                "query_id": receipt["query_id"],
                "query_assignment_hash": receipt["query_assignment_hash"],
                "oracle_accepts": receipt["oracle_accepts"],
                "label_minted_before_learner": receipt["label_minted_before_learner"],
                "receipt_hash": receipt["receipt_hash"],
            },
            operation="dependent_membership_query",
            oracle=_base_oracle(
                "exp5763_dependent_exact_membership_validator",
                minted_before_learner=receipt.get("label_minted_before_learner") is True,
            ),
        )
    for receipt in artifact.get("transition_receipts", []):
        operation = str(receipt["operation"])
        sequence = _add_event(
            events,
            states,
            event_type=_transition_event_type(operation),
            source_adapter="exp5763",
            sequence=sequence,
            source_artifact=EXP5763_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5763_artifact"]),
            source_hash=str(receipt["transition_hash"]),
            visibility="science",
            axes=_axes(change="dependent_" + operation),
            payload={
                "transition_id": receipt["transition_id"],
                "operation": operation,
                "accepted": receipt["accepted"],
                "pre_state_hash": receipt["pre_state_hash"],
                "post_state_hash": receipt["post_state_hash"],
                "active_state_hash": receipt["post_state_hash"],
                "restart_state_hash": receipt["restart_state_hash"],
                "rollback_state_hash": receipt["rollback_state_hash"],
                "expected_state_hash": (
                    receipt["pre_state_hash"] if operation == "rollback" else receipt["post_state_hash"]
                ),
                "restored_state_hash": receipt["rollback_state_hash"],
                "receipt_hash": receipt["transition_hash"],
            },
            operation=operation,
            oracle=_base_oracle("exp5763_transition_validator"),
        )
    for receipt in artifact.get("dependent_session_ledger", []):
        sequence = _add_event(
            events,
            states,
            event_type="protected_prefix_replay",
            source_adapter="exp5763",
            sequence=sequence,
            source_artifact=EXP5763_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5763_artifact"]),
            source_hash=str(receipt["row_hash"]),
            visibility="science",
            axes=_axes(family=receipt.get("family_shift"), change="session_prefix"),
            payload={
                "session_id": receipt["session_id"],
                "session_index": receipt["session_index"],
                "protected_prefix_hash": receipt["protected_prefix_hash"],
                "learner_target_boundary": receipt["learner_target_boundary"],
                "replay_passed": receipt["learner_target_boundary"] == "membership_answer_only",
                "receipt_hash": receipt["row_hash"],
            },
            operation="protected_prefix_replay",
            oracle=_base_oracle("exp5763_protected_prefix_validator"),
        )
    for receipt in artifact.get("recovery_receipts", []):
        sequence = _add_event(
            events,
            states,
            event_type="rollback",
            source_adapter="exp5763",
            sequence=sequence,
            source_artifact=EXP5763_ARTIFACT_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5763_artifact"]),
            source_hash=str(receipt["recovery_hash"]),
            visibility="science",
            axes=_axes(change="recovery_rollback"),
            payload={
                "session_id": receipt["session_id"],
                "boundary": receipt["boundary"],
                "expected_state_hash": receipt["expected_state_hash"],
                "restored_state_hash": receipt["restored_state_hash"],
                "rejected_update_propagation_count": receipt[
                    "rejected_update_propagation_count"
                ],
                "receipt_hash": receipt["recovery_hash"],
            },
            operation="rollback",
            oracle=_base_oracle("exp5763_recovery_validator"),
        )
    source_event_count = (
        len(artifact.get("query_label_receipts", []))
        + len(artifact.get("transition_receipts", []))
        + len(artifact.get("dependent_session_ledger", []))
        + len(artifact.get("recovery_receipts", []))
    )
    receipt = _adapter_receipt(
        adapter_id="exp5763",
        source_artifact_hash=str(upstream_hashes["exp5763_artifact"]),
        source_event_count=source_event_count,
        events=events,
        states=list(states.values()),
    )
    return events, states, sequence, receipt


def _adapt_exp5785(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    upstream_hashes: Mapping[str, str],
    sequence: int,
) -> tuple[list[JsonDict], dict[str, JsonDict], int, JsonDict]:
    events: list[JsonDict] = []
    states: dict[str, JsonDict] = {}
    for row in rows:
        split = str(row["split"])
        event_type = "sealed_future_evaluation" if split == "future_test" else "observation"
        if row.get("surface_kind") != "canonical" and split != "future_test":
            event_type = "collision_split"
        payload = {
            "row_id": row["row_id"],
            "unit_id": row["unit_id"],
            "split": split,
            "surface_kind": row["surface_kind"],
            "row_hash": row["row_hash"],
            "candidate_domain_hash": sha256_json(row.get("candidate_domain") or []),
            "protected_fact_hash": row["protected_fact_hash"],
            "exact_label_hidden_from_prompt": True,
            "receipt_hash": row["row_hash"],
        }
        if event_type == "collision_split":
            payload["split_receipt_hash"] = sha256_json(
                {
                    "unit_id": row["unit_id"],
                    "surface_kind": row["surface_kind"],
                    "proof_preserving": row["proof_preserving"],
                }
            )
        sequence = _add_event(
            events,
            states,
            event_type=event_type,
            source_adapter="exp5785",
            sequence=sequence,
            source_artifact=EXP5785_ROWS_RELATIVE_PATH.as_posix(),
            source_artifact_hash=str(upstream_hashes["exp5785_rows"]),
            source_hash=str(row["row_hash"]),
            visibility=split,
            axes=_axes(
                family=row.get("family"),
                hardness=row.get("solver_effort_bin"),
                surface=row.get("surface_kind"),
                change="sealed_future" if split == "future_test" else str(row.get("surface_kind")),
            ),
            payload=payload,
            operation="sealed_fixture",
            oracle=_base_oracle("exp5785_exact_fixture_validator"),
        )
    receipt = _adapter_receipt(
        adapter_id="exp5785",
        source_artifact_hash=str(upstream_hashes["exp5785_artifact"]),
        source_row_count=len(rows),
        source_event_count=len(rows),
        events=events,
        states=list(states.values()),
    )
    return events, states, sequence, receipt


def adapt_all_upstreams(root: Path = REPO_ROOT) -> JsonDict:
    """Adapt every sealed upstream row or lifecycle receipt into canonical events."""

    root = Path(root)
    artifacts, rows = _load_upstream_bundle(root)
    upstream_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    all_events: list[JsonDict] = []
    all_states: dict[str, JsonDict] = {}
    receipts: JsonDict = {"schema": SCHEMA + ".adapter_receipts", "adapters": {}}
    sequence = 0

    for adapter in (
        lambda seq: _adapt_exp5761(
            artifacts["exp5761"],
            rows["exp5761_instances"],
            upstream_hashes,
            seq,
        ),
        lambda seq: _adapt_exp5762(artifacts["exp5762"], upstream_hashes, seq),
        lambda seq: _adapt_exp5763(artifacts["exp5763"], upstream_hashes, seq),
        lambda seq: _adapt_exp5785(
            artifacts["exp5785"],
            rows["exp5785_rows"],
            upstream_hashes,
            seq,
        ),
    ):
        events, states, sequence, receipt = adapter(sequence)
        all_events.extend(events)
        all_states.update(states)
        receipts["adapters"][receipt["adapter_id"]] = receipt

    state_values = list(all_states.values())
    receipts["total_canonical_event_count"] = len(all_events)
    receipts["total_canonical_state_count"] = len(state_values)
    receipts["canonical_event_root_hash"] = sha256_json(
        [event["event_hash"] for event in all_events]
    )
    receipts["canonical_state_root_hash"] = sha256_json(
        [state["state_hash"] for state in state_values]
    )
    receipts["all_round_trips_ok"] = all(
        receipt["round_trip_ok"] is True for receipt in receipts["adapters"].values()
    )
    return {"events": all_events, "states": state_values, "receipts": receipts}


def _event_error(error_code: str, event: Mapping[str, Any] | None = None) -> JsonDict:
    return {
        "error_code": error_code,
        "event_id": "" if event is None else str(event.get("event_id") or ""),
        "event_type": "" if event is None else str(event.get("event_type") or ""),
    }


def validate_event_stream(
    events: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return typed schema errors for canonical chronology, visibility, and state defects."""

    errors: list[JsonDict] = []
    seen_events: set[str] = set()
    seen_states: set[str] = set()
    previous_sequence = -1
    state_by_id: dict[str, Mapping[str, Any]] = {}

    for state in states:
        state_id = str(state.get("state_id") or "")
        if state_id in seen_states:
            errors.append(_event_error("duplicate_state_identity"))
        seen_states.add(state_id)
        state_by_id[state_id] = state
        if not str(state.get("source_artifact_hash") or "").startswith("sha256:"):
            errors.append(_event_error("missing_hash"))
        if not str(state.get("source_row_hash") or "").startswith("sha256:"):
            errors.append(_event_error("missing_hash"))
        if state.get("state_hash") != canonical_state_hash(state):
            errors.append(_event_error("state_hash_mismatch"))
        elif state_id != expected_state_id(state):
            errors.append(_event_error("state_id_mismatch"))

    for event in events:
        event_id = str(event.get("event_id") or "")
        if event_id in seen_events:
            errors.append(_event_error("duplicate_event_identity", event))
        seen_events.add(event_id)
        sequence = int(event.get("causal_sequence_index", -1))
        if sequence <= previous_sequence:
            errors.append(_event_error("non_monotone_chronology", event))
        previous_sequence = sequence
        if event.get("event_hash") != canonical_event_hash(event):
            errors.append(_event_error("event_hash_mismatch", event))
        elif event_id != expected_event_id(event):
            errors.append(_event_error("event_id_mismatch", event))
        if event.get("payload_hash") != sha256_json(event.get("payload") or {}):
            errors.append(_event_error("payload_hash_mismatch", event))
        if event.get("event_type") not in REQUIRED_EVENT_TYPES:
            errors.append(_event_error("unsupported_event_type", event))
        if event.get("visibility") not in VISIBILITY_VALUES:
            errors.append(_event_error("invalid_visibility", event))
        required_hashes = (
            "payload_hash",
            "source_artifact_hash",
            "source_row_hash",
            "source_receipt_hash",
        )
        if any(not str(event.get(field) or "").startswith("sha256:") for field in required_hashes):
            errors.append(_event_error("missing_hash", event))
        provenance = dict(event.get("oracle_provenance") or {})
        if provenance.get("hidden_label_access") is True and event.get("visibility") in {
            "science",
            "future_test",
        }:
            errors.append(_event_error("hidden_science_label_access", event))
        if provenance.get("forged_label") is True or provenance.get("authority") != "exact_solver_or_validator":
            errors.append(_event_error("forged_oracle_label", event))
        payload = dict(event.get("payload") or {})
        if (
            payload.get("identity_collision") is True
            and event.get("event_type") != "collision_split"
        ) or (
            event.get("event_type") == "collision_split"
            and not str(payload.get("split_receipt_hash") or "").startswith("sha256:")
        ):
            errors.append(_event_error("collision_without_split", event))
        if event.get("event_type") == "supersession" and (
            payload.get("stale_supersession") is True
            or not str(payload.get("active_state_hash") or "").startswith("sha256:")
        ):
            errors.append(_event_error("stale_supersession", event))
        if event.get("event_type") == "rollback" and (
            not str(payload.get("expected_state_hash") or "").startswith("sha256:")
            or not str(payload.get("restored_state_hash") or "").startswith("sha256:")
            or payload.get("expected_state_hash") != payload.get("restored_state_hash")
        ):
            errors.append(_event_error("rollback_mismatch", event))
        if event.get("event_type") == "protected_prefix_replay" and (
            not str(payload.get("protected_prefix_hash") or "").startswith("sha256:")
            or payload.get("replay_passed") is not True
        ):
            errors.append(_event_error("missing_protected_prefix_evidence", event))
        parent_state = state_by_id.get(str(event.get("parent_state_id") or ""))
        result_state = state_by_id.get(str(event.get("resulting_state_id") or ""))
        if parent_state is None:
            errors.append(_event_error("missing_parent_state", event))
        if result_state is None:
            errors.append(_event_error("missing_resulting_state", event))
        elif parent_state is not None and result_state.get("parent_state_hash") != parent_state.get(
            "state_hash"
        ):
            errors.append(_event_error("parent_state_hash_mismatch", event))
        elif (
            event.get("parent_state_id") != event.get("resulting_state_id")
            and event.get("event_type") not in {"observation", "sealed_future_evaluation"}
            and not str(result_state.get("mutation_receipt_hash") or "").startswith("sha256:")
        ):
            errors.append(_event_error("state_mutation_without_receipt", event))
    return errors


def assert_valid_event_stream(
    events: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
) -> bool:
    """Raise on the first canonical event/state validation error."""

    errors = validate_event_stream(events, states)
    if errors:
        raise ContractValidationError(errors[0]["error_code"])
    return True


def _positive_control_stream() -> tuple[list[JsonDict], list[JsonDict]]:
    state0 = make_state(
        source_adapter="control",
        sequence=0,
        state_label="parent",
        source_artifact="adversarial_fixture",
        source_artifact_hash=sha256_text("adversarial_fixture"),
        source_hash=sha256_text("row"),
        visibility="science",
        axes=_axes(family="control", change="control"),
    )
    state1 = make_state(
        source_adapter="control",
        sequence=1,
        state_label="result",
        source_artifact="adversarial_fixture",
        source_artifact_hash=sha256_text("adversarial_fixture"),
        source_hash=sha256_text("row"),
        visibility="science",
        axes=_axes(family="control", change="control"),
        parent_state_hash=state0["state_hash"],
        mutation_receipt_hash=sha256_text("receipt"),
        lifecycle_operation="control",
    )
    event = make_event(
        event_type="promotion",
        source_adapter="control",
        sequence=0,
        source_artifact="adversarial_fixture",
        source_artifact_hash=sha256_text("adversarial_fixture"),
        source_hash=sha256_text("row"),
        visibility="science",
        axes=_axes(family="control", change="control"),
        payload={"receipt_hash": sha256_text("receipt")},
        parent_state=state0,
        resulting_state=state1,
        oracle_provenance=_base_oracle("adversarial_fixture"),
    )
    return [event], [state0, state1]


def _mutated_error(mutate: Any) -> str:
    events, states = _positive_control_stream()
    mutate(events, states)
    event = events[0]
    event["payload_hash"] = sha256_json(event["payload"])
    event["event_hash"] = canonical_event_hash(event)
    event["event_id"] = expected_event_id(event)
    errors = validate_event_stream(events, states)
    return errors[0]["error_code"] if errors else "not_detected"


def adversarial_contract_results() -> JsonDict:
    """Run fixed negative fixtures for leakage, forgery, collision, and state replay."""

    mutations = {
        "leakage": lambda events, states: events[0]["oracle_provenance"].update(
            {"hidden_label_access": True}
        ),
        "forged_oracle_labels": lambda events, states: events[0]["oracle_provenance"].update(
            {"forged_label": True}
        ),
        "collision_without_split": lambda events, states: events[0]["payload"].update(
            {"identity_collision": True}
        ),
        "stale_supersession": lambda events, states: (
            events[0].update({"event_type": "supersession"}),
            events[0]["payload"].update({"stale_supersession": True}),
        ),
        "rollback_mismatch": lambda events, states: (
            events[0].update({"event_type": "rollback"}),
            events[0]["payload"].update(
                {
                    "expected_state_hash": sha256_text("expected"),
                    "restored_state_hash": sha256_text("wrong"),
                }
            ),
        ),
        "missing_protected_prefix_evidence": lambda events, states: (
            events[0].update({"event_type": "protected_prefix_replay"}),
            events[0]["payload"].update({"protected_prefix_hash": "", "replay_passed": False}),
        ),
    }
    expected = {
        "leakage": "hidden_science_label_access",
        "forged_oracle_labels": "forged_oracle_label",
        "collision_without_split": "collision_without_split",
        "stale_supersession": "stale_supersession",
        "rollback_mismatch": "rollback_mismatch",
        "missing_protected_prefix_evidence": "missing_protected_prefix_evidence",
    }
    return {
        name: {
            "passed": (error_code := _mutated_error(mutate)) == expected[name],
            "error_code": error_code,
            "expected_error_code": expected[name],
            "fixture_hash": sha256_json({"name": name, "expected": expected[name]}),
        }
        for name, mutate in mutations.items()
    }


def canonical_event_schema() -> JsonDict:
    """Describe the event vocabulary and required identity/hash fields."""

    return {
        "schema": EVENT_SCHEMA_VERSION,
        "event_types": list(REQUIRED_EVENT_TYPES),
        "identity_rule": EVENT_IDENTITY_RULE,
        "required_fields": [
            "event_id",
            "event_type",
            "causal_sequence_index",
            "parent_state_id",
            "resulting_state_id",
            "family_hardness_surface_change_axes",
            "visibility",
            "payload_hash",
            "source_artifact_hash",
            "source_row_hash",
            "source_receipt_hash",
            "oracle_provenance",
            "event_hash",
        ],
        "visibility_values": list(VISIBILITY_VALUES),
        "immutable_payload_hash_rule": "sha256(canonical_json(payload))",
        "oracle_provenance_rule": "authority must be exact_solver_or_validator and no hidden science/future label access",
    }


def canonical_state_schema() -> JsonDict:
    """Describe the immutable state identity and mutation receipt boundary."""

    return {
        "schema": STATE_SCHEMA_VERSION,
        "identity_rule": STATE_IDENTITY_RULE,
        "required_fields": [
            "state_id",
            "state_version",
            "state_sequence_index",
            "state_hash",
            "parent_state_hash",
            "mutation_receipt_hash",
            "lifecycle_operation",
            "source_artifact_hash",
            "source_row_hash",
            "visibility",
            "axes",
        ],
        "mutation_rule": "state changes require a sha256 mutation_receipt_hash unless the event is observation or sealed_future_evaluation",
        "rollback_rule": "rollback restored_state_hash must equal expected_state_hash",
    }


def chronology_and_visibility_checks(
    events: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize canonical chronology and hidden-label visibility validation."""

    errors = validate_event_stream(events, states)
    counts: dict[str, int] = {}
    for error in errors:
        code = str(error["error_code"])
        counts[code] = counts.get(code, 0) + 1
    return {
        "schema": SCHEMA + ".chronology_visibility",
        "all_passed": not errors,
        "event_count": len(events),
        "state_count": len(states),
        "first_sequence_index": events[0]["causal_sequence_index"] if events else None,
        "last_sequence_index": events[-1]["causal_sequence_index"] if events else None,
        "non_monotone_count": counts.get("non_monotone_chronology", 0),
        "hidden_label_access_count": counts.get("hidden_science_label_access", 0),
        "duplicate_identity_count": counts.get("duplicate_event_identity", 0)
        + counts.get("duplicate_state_identity", 0),
        "missing_hash_count": counts.get("missing_hash", 0),
        "state_mutation_without_receipt_count": counts.get(
            "state_mutation_without_receipt",
            0,
        ),
        "visibility_values": sorted({str(event.get("visibility")) for event in events}),
        "sealed_future_visibility_count": sum(
            1 for event in events if event.get("visibility") == "future_test"
        ),
        "schema_error_counts": counts,
        "schema_errors": errors,
    }


def _field_provenance() -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "sources": [
                "task_prompt",
                SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    provenance.update(
        {
            field: {"principle": principle, "sources": ["local_metadata"]}
            for field, principle in FIELD_PRINCIPLE_EXTRAS.items()
        }
    )
    provenance["upstream_sources"] = {
        name: relative.as_posix() for name, relative in UPSTREAM_PATHS.items()
    }
    provenance["schema_rule_sources"] = {
        "canonical_event_schema": "make_event and validate_event_stream",
        "canonical_state_schema": "make_state and validate_event_stream",
    }
    provenance["adversarial_test_sources"] = {
        name: "adversarial_contract_results" for name in REQUIRED_ADVERSARIAL_CASES
    }
    return provenance


def adaptive_memory_contract_ready_score_from_artifact(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when every contract gate is clean."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    receipts = dict(artifact.get("adapter_round_trip_receipts") or {})
    adapters = dict(receipts.get("adapters") or {})
    chronology = dict(artifact.get("chronology_and_visibility_checks") or {})
    adversarial = dict(artifact.get("adversarial_contract_results") or {})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = bool(
        artifact.get("status") == "complete"
        and preconditions.get("preconditions_ready") is True
        and receipts.get("all_round_trips_ok") is True
        and set(adapters) == {"exp5761", "exp5762", "exp5763", "exp5785"}
        and all(dict(row).get("round_trip_ok") is True for row in adapters.values())
        and chronology.get("all_passed") is True
        and artifact.get("schema_errors") == []
        and set(adversarial) == set(REQUIRED_ADVERSARIAL_CASES)
        and all(dict(row).get("passed") is True for row in adversarial.values())
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with the self-referential checksum blanked."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _blocked_schema_errors(preconditions: Mapping[str, Any]) -> list[str]:
    reasons = [str(item) for item in preconditions.get("blocked_reasons", [])]
    return reasons or ["preconditions_blocked"]


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the terminal Exp5825 contract artifact."""

    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path or (root / RESULT_RELATIVE_PATH))
    preconditions = collect_preconditions(root, result_path=result_path)
    measured_duration = (
        float(duration_s)
        if duration_s is not None
        else round(time.perf_counter() - started, 6)
    )
    bundle: JsonDict = {"events": [], "states": [], "receipts": {"adapters": {}}}
    schema_errors = _blocked_schema_errors(preconditions)
    chronology = chronology_and_visibility_checks([], [])
    if preconditions["preconditions_ready"] is True:
        bundle = adapt_all_upstreams(root)
        chronology = chronology_and_visibility_checks(bundle["events"], bundle["states"])
        schema_errors = [str(error["error_code"]) for error in chronology["schema_errors"]]
    status = "complete" if preconditions["preconditions_ready"] is True and not schema_errors else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "preconditions_checked": preconditions,
        "upstream_artifact_hashes": preconditions["upstream_artifact_hashes"],
        "canonical_event_schema": canonical_event_schema(),
        "canonical_state_schema": canonical_state_schema(),
        "adapter_round_trip_receipts": bundle["receipts"],
        "chronology_and_visibility_checks": chronology,
        "adversarial_contract_results": adversarial_contract_results(),
        "adaptive_memory_contract_ready_score": 0.0,
        "schema_errors": schema_errors,
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands or []),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["adaptive_memory_contract_ready_score"] = (
        adaptive_memory_contract_ready_score_from_artifact(artifact)
    )
    if artifact["adaptive_memory_contract_ready_score"] != 1.0:
        artifact["status"] = "blocked"
        artifact["adaptive_memory_contract_ready_score"] = (
            adaptive_memory_contract_ready_score_from_artifact(artifact)
        )
    if artifact["adaptive_memory_contract_ready_score"] == 1.0:
        artifact["honest_verdict"] = "complete: certified_adaptive_memory_contract_ready"
    else:
        failed_commands = [
            command for command, code in artifact["test_exit_codes"].items() if code != 0
        ]
        missing_exit_codes = set(artifact["test_commands"]) != set(artifact["test_exit_codes"])
        reasons = (
            artifact["schema_errors"]
            or preconditions["blocked_reasons"]
            or (["failed_test_exit_codes"] if failed_commands or missing_exit_codes else [])
            or ["contract_gates"]
        )
        artifact["honest_verdict"] = "blocked: " + ",".join(str(reason) for reason in reasons[:8])
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, field provenance, readiness, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("status")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        receipt = dict(provenance.get(field) or {})
        if receipt.get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_score = adaptive_memory_contract_ready_score_from_artifact(artifact)
    if artifact["adaptive_memory_contract_ready_score"] != expected_score:
        raise ValueError("adaptive_memory_contract_ready_score")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict")
    if artifact["status"] == "complete" and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if artifact["status"] == "blocked" and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build, validate, and write the terminal Exp5825 artifact."""

    output_path = Path(result_path or (Path(root) / RESULT_RELATIVE_PATH))
    artifact = build_artifact(
        root=root,
        result_path=output_path,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run() -> JsonDict:
    """Write the default artifact using declared validation commands."""

    return build_and_write_artifact(test_commands=DEFAULT_TEST_COMMANDS)


def main() -> None:
    """CLI entry point for local artifact regeneration."""

    run()


if __name__ == "__main__":  # pragma: no cover - CLI guard.
    main()
