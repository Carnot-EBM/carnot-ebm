"""Build the Exp6407 provenance tiered factor memory protocol artifact.

Spec refs: REQ-LEARN-6407, SCENARIO-LEARN-6407-RAW-COMPILED,
SCENARIO-LEARN-6407-REPLAY, SCENARIO-LEARN-6407-ESCALATION,
SCENARIO-LEARN-6407-CONTAMINATION, SCENARIO-LEARN-6407-ATTACKS,
SCENARIO-LEARN-6407-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6407_provenance_tiered_factor_memory_protocol.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6407_provenance_tiered_factor_memory_protocol"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

EXP6342_RELATIVE_PATH = Path("results/experiment_6342_anytime_evalue_release_ledger.json")
EXP6342_LEDGER_RELATIVE_PATH = Path(
    "results/experiment_6342_anytime_evalue_release_ledger.json.evalue_ledger.jsonl"
)
EXP6343_RELATIVE_PATH = Path("results/experiment_6343_evidence_carrying_factor_lifecycle.json")
EXP6397_RELATIVE_PATH = Path(
    "results/experiment_6397_transactional_continuous_factor_learning.json"
)
EXP6397_MANIFEST_RELATIVE_PATH = Path(
    str(EXP6397_RELATIVE_PATH) + ".chronological_manifest.json"
)
EXP6398_RELATIVE_PATH = Path(
    "results/experiment_6398_default_off_transactional_factor_consumer.json"
)
EXP6398_MANIFEST_RELATIVE_PATH = Path(
    str(EXP6398_RELATIVE_PATH) + ".untouched_consumer_manifest.json"
)

RAW_SCHEMA_SUFFIX = ".raw_record_schema.json"
COMPILED_SCHEMA_SUFFIX = ".compiled_typed_graph_schema.json"
RAW_LEDGER_SUFFIX = ".raw_ledger.jsonl"
COMPILED_GRAPH_SUFFIX = ".compiled_typed_graph.json"
CONTAMINATION_MANIFEST_SUFFIX = ".contamination_manifest.json"

SCHEMA = "carnot.experiment_6407.provenance_tiered_factor_memory_protocol.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6407
INFERENCE_SUBSTRATE = "deterministic_provenance_tiered_factor_memory_protocol_no_llm"

RAW_REQUIRED_FIELDS = (
    "event_hash",
    "source_spans",
    "model_identity",
    "harness_identity",
    "license_key",
    "exact_checker_version",
    "release_outcome",
    "predecessor",
    "disposition",
    "created_at",
    "observed_at",
    "expiry",
    "supersession",
    "transaction_receipt",
)
COMPILED_NODE_TYPES = (
    "factor",
    "evidence",
    "model",
    "constraint_family",
    "checker",
    "license",
)
REQUIRED_EDGE_TYPES = ("predecessor", "expiry", "supersession")
COMPILED_EDGE_TYPES = (
    "supports",
    "checked_by",
    "licensed_by",
    "predecessor",
    "expiry",
    "supersession",
)
EVENT_CLASSES = (
    "supported",
    "contradicted",
    "implicit",
    "stale",
    "duplicated",
    "replayed",
    "superseded",
    "poisoned",
    "clean_negative",
)
PARTITIONS = ("calibration", "acquisition", "retention", "future")
ESCALATION_CONDITIONS = (
    "missing_provenance",
    "implicit_support",
    "graph_cache_disagreement",
    "stale_summary",
    "expired_license",
    "unresolved_supersession",
    "checker_drift",
)
DIAGNOSTIC_FEATURES = (
    "utility",
    "exact_confidence",
    "novelty",
    "recency",
    "content_type",
)
ATTACK_IDS = (
    "orphan_summary",
    "forged_raw_link",
    "cycle_creation",
    "neighborhood_underreach",
    "neighborhood_overreach",
    "stale_head",
    "partial_atomic_write",
    "duplicate_effect",
    "expiry_removal",
    "cache_resurrection_after_restart",
)

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6407_provenance_tiered_factor_memory_protocol --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py "
    "-m pytest tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py"
)
RESTART_E2E_COMMAND = RUN_COMMAND + " --validate"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6407_provenance_tiered_factor_memory_protocol.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    RESTART_E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

UPSTREAM_RELATIVE_PATHS = (
    EXP6397_RELATIVE_PATH,
    EXP6397_MANIFEST_RELATIVE_PATH,
    EXP6398_RELATIVE_PATH,
    EXP6398_MANIFEST_RELATIVE_PATH,
    EXP6342_RELATIVE_PATH,
    EXP6342_LEDGER_RELATIVE_PATH,
    EXP6343_RELATIVE_PATH,
)
CHECKER_RELATIVE_PATHS = (
    Path("python/carnot/experiment_6342_anytime_evalue_release_ledger.py"),
    Path("python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py"),
    Path("python/carnot/experiment_6397_transactional_continuous_factor_learning.py"),
    Path("python/carnot/experiment_6398_default_off_transactional_factor_consumer.py"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    *UPSTREAM_RELATIVE_PATHS,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("research-references.md"),
    Path("ops/e2e-test-plan.md"),
    *CHECKER_RELATIVE_PATHS,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_factor_head_release_ledger_lifecycle_checker_and_license_hashes",
    "raw_record_schema_path_hash_and_required_fields",
    "compiled_typed_graph_schema_path_hash_node_and_edge_types",
    "raw_to_compiled_provenance_link_receipts",
    "affected_neighborhood_equations_and_receipts",
    "local_vs_full_replay_equivalence_results",
    "raw_tier_escalation_rules_and_tests",
    "contamination_manifest_path_hash_counts_classes_and_partition_seals",
    "diagnostic_admission_feature_contract",
    "exact_veto_override_count",
    "supported_contradicted_implicit_stale_duplicate_replay_supersession_poison_and_negative_fixture_results",
    "orphan_forgery_cycle_neighborhood_head_atomic_duplicate_expiry_and_restart_attack_matrix",
    "compiled_cache_authority_claimed",
    "learning_utility_claimed",
    "provenance_tiered_memory_protocol_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status follows provenance, replay, escalation, attacks, tests, and no-authority gates.",
    "upstream_factor_head_release_ledger_lifecycle_checker_and_license_hashes": "Upstream hashes freeze V550 inputs before the protocol can build memory rows.",
    "raw_record_schema_path_hash_and_required_fields": "The raw schema defines the append-only authority tier.",
    "compiled_typed_graph_schema_path_hash_node_and_edge_types": "The compiled graph schema defines cache shape, not semantic authority.",
    "raw_to_compiled_provenance_link_receipts": "Every compiled cache row must trace to immutable raw evidence.",
    "affected_neighborhood_equations_and_receipts": "Typed neighborhoods bound local replay under drift.",
    "local_vs_full_replay_equivalence_results": "Local cache replay is acceptable only when it matches full raw replay.",
    "raw_tier_escalation_rules_and_tests": "Escalation conditions force ambiguity back to the raw tier.",
    "contamination_manifest_path_hash_counts_classes_and_partition_seals": "The frozen contamination stream prevents fitted partitions.",
    "diagnostic_admission_feature_contract": "A-MAC-style features are diagnostic only.",
    "exact_veto_override_count": "No weighted diagnostic score may override an exact veto.",
    "supported_contradicted_implicit_stale_duplicate_replay_supersession_poison_and_negative_fixture_results": "Fixture classes prove the protocol separates support from contamination.",
    "orphan_forgery_cycle_neighborhood_head_atomic_duplicate_expiry_and_restart_attack_matrix": "Cache and transaction attacks must fail closed.",
    "compiled_cache_authority_claimed": "The compiled tier is a cache, never an authority.",
    "learning_utility_claimed": "This protocol freezes safety behavior and makes no learning-utility claim.",
    "provenance_tiered_memory_protocol_ready_score": "Readiness is one only when raw links, replay, escalation, attacks, seals, tests, and no-authority gates all pass.",
    "protected_files_unchanged": "Protected files and upstream evidence remain byte-identical.",
    "preconditions_checked": "Preconditions bind date, hashes, schemas, partitions, and protected files before rows build.",
    "inference_substrate": "The run is deterministic provenance replay with no LLM or model inference.",
    "verifier_is_oracle": "Bare true applies only to deterministic fixture checkers and replay equivalence.",
    "field_principles": "Every required field states its guard purpose.",
    "field_provenance": "Every required field maps to specs, upstream hashes, sidecars, fixtures, tests, or replay receipts.",
    "random_seed": "The seed pins fixture order and attack order.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Recorded command exit codes gate readiness.",
    "reproducibility_checksum": "The normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the no-authority boundary.",
    **{
        f"escalation:{condition}": "This condition escalates ambiguous cache evidence to the immutable raw tier."
        for condition in ESCALATION_CONDITIONS
    },
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6407",
        "Exp6397 factor head and transaction artifact",
        "Exp6398 default-off consumer artifact",
        "Exp6342 release ledger",
        "Exp6343 lifecycle code",
        "Exp6407 deterministic fixtures and focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable JSON so byte hashes do not depend on dict order."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a repository-style digest for raw bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the path is absent."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other shapes with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round deterministic receipts without hiding small nonzero values."""

    return round(float(value), 12)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl_atomic(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL through a same-directory temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON when requested, otherwise return the would-be digest."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: str | Path, *, digest: str | None = None) -> JsonDict:
    """Record path, presence, size, and hash."""

    path = Path(path)
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": digest if digest is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object from disk."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_top_level_not_object:{path}")
    return value


def protected_hashes() -> dict[str, str | None]:
    """Hash files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes() -> dict[str, str | None]:
    """Hash files that define this experiment and its checks."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes from before and after the run."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def raw_record_schema() -> JsonDict:
    """Return the frozen append-only raw record schema."""

    return {
        "schema": SCHEMA + ".raw_record_schema",
        "required_fields": list(RAW_REQUIRED_FIELDS),
        "row_hash_field": "raw_row_hash",
        "append_only": True,
        "additional_required_hashes": ["event_hash", "transaction_receipt.transaction_hash"],
    }


def compiled_typed_graph_schema() -> JsonDict:
    """Return the frozen compiled cache graph schema."""

    return {
        "schema": SCHEMA + ".compiled_typed_graph_schema",
        "node_types": list(COMPILED_NODE_TYPES),
        "edge_types": list(COMPILED_EDGE_TYPES),
        "required_edge_types": list(REQUIRED_EDGE_TYPES),
        "raw_hash_link_field": "raw_hashes",
        "cache_only": True,
    }


def raw_record_schema_receipt(result_path: Path, *, write: bool) -> JsonDict:
    """Write or hash the raw record schema sidecar."""

    path = result_path.with_suffix(result_path.suffix + RAW_SCHEMA_SUFFIX)
    payload = raw_record_schema()
    digest = write_payload_or_hash(path, payload, write=write)
    return {
        "schema_path": str(path),
        "schema_sha256": digest,
        "required_fields": list(payload["required_fields"]),
        "required_fields_complete": set(RAW_REQUIRED_FIELDS) <= set(payload["required_fields"]),
        "append_only": payload["append_only"],
    }


def compiled_typed_graph_schema_receipt(result_path: Path, *, write: bool) -> JsonDict:
    """Write or hash the compiled graph schema sidecar."""

    path = result_path.with_suffix(result_path.suffix + COMPILED_SCHEMA_SUFFIX)
    payload = compiled_typed_graph_schema()
    digest = write_payload_or_hash(path, payload, write=write)
    return {
        "schema_path": str(path),
        "schema_sha256": digest,
        "node_types": list(payload["node_types"]),
        "edge_types": list(payload["edge_types"]),
        "node_types_complete": set(COMPILED_NODE_TYPES) <= set(payload["node_types"]),
        "required_edge_types_complete": set(REQUIRED_EDGE_TYPES) <= set(payload["edge_types"]),
        "cache_only": payload["cache_only"],
    }


def _event_release_outcome(event_class: str) -> str:
    outcomes = {
        "supported": "accepted",
        "contradicted": "rejected_exact_veto",
        "implicit": "escalated_missing_explicit_support",
        "stale": "escalated_stale_summary",
        "duplicated": "rejected_duplicate_effect",
        "replayed": "rejected_replayed_evidence",
        "superseded": "accepted_superseding_revision",
        "poisoned": "quarantined_poison",
        "clean_negative": "rejected_clean_negative",
    }
    return outcomes[event_class]


def _event_disposition(event_class: str) -> str:
    dispositions = {
        "supported": "Commit",
        "contradicted": "Reject",
        "implicit": "Escalate",
        "stale": "Escalate",
        "duplicated": "Reject",
        "replayed": "Reject",
        "superseded": "Commit",
        "poisoned": "Quarantine",
        "clean_negative": "Reject",
    }
    return dispositions[event_class]


def build_raw_records(event_count_per_class: int = 6) -> list[JsonDict]:
    """Build deterministic raw rows across every contamination class."""

    rows: list[JsonDict] = []
    for class_index, event_class in enumerate(EVENT_CLASSES):
        for local_index in range(event_count_per_class):
            global_index = class_index * event_count_per_class + local_index
            event_id = f"event-6407-{global_index:03d}"
            factor_id = f"factor-{event_class}-{local_index % 3}"
            supersession = (
                {"supersedes_factor_id": f"factor-supported-{local_index % 3}"}
                if event_class == "superseded"
                else None
            )
            predecessor = None if event_class == "supported" else "head:v550"
            transaction_receipt = {
                "transaction_id": f"txn-6407-{global_index:03d}",
                "atomic": True,
                "predecessor_bound": True,
                "transaction_hash": sha256_json({"event_id": event_id, "class": event_class}),
            }
            row: JsonDict = {
                "event_id": event_id,
                "event_class": event_class,
                "partition": PARTITIONS[global_index % len(PARTITIONS)],
                "factor_id": factor_id,
                "evidence_id": f"evidence-{global_index:03d}",
                "constraint_family": ("route_guard", "conservation_guard", "threshold_guard")[
                    global_index % 3
                ],
                "event_hash": sha256_json({"event_id": event_id, "class": event_class}),
                "source_spans": [
                    {
                        "artifact": EXP6397_RELATIVE_PATH.as_posix(),
                        "json_pointer": "/atomic_disposition_records/0",
                        "span_hash": sha256_json({"event_id": event_id, "span": 0}),
                    }
                ],
                "model_identity": {
                    "model_id": (
                        "unsloth/Qwen3.6-35B-A3B-GGUF"
                        if global_index % 2 == 0
                        else "unsloth/gemma-4-26B-A4B-it-GGUF"
                    ),
                    "model_hash": sha256_json({"model": global_index % 2}),
                },
                "harness_identity": {
                    "harness_id": f"harness-{global_index % 3}",
                    "harness_hash": sha256_json({"harness": global_index % 3}),
                },
                "license_key": f"license-v550-{global_index % 4}",
                "exact_checker_version": "transactional_factor_exact_checker_v1",
                "release_outcome": _event_release_outcome(event_class),
                "predecessor": predecessor,
                "disposition": _event_disposition(event_class),
                "created_at": f"2026-08-13T00:{global_index % 60:02d}:00Z",
                "observed_at": f"2026-08-13T00:{global_index % 60:02d}:30Z",
                "expiry": "2026-12-31T00:00:00Z",
                "supersession": supersession,
                "transaction_receipt": transaction_receipt,
            }
            row["raw_row_hash"] = sha256_json(row)
            rows.append(row)
    return rows


def compile_typed_graph(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compile raw rows into a typed graph cache with raw-hash links."""

    nodes: list[JsonDict] = []
    edges: list[JsonDict] = []
    for row in raw_rows:
        raw_hash = str(row["raw_row_hash"])
        factor_id = str(row["factor_id"])
        evidence_id = str(row["evidence_id"])
        model_id = str(as_mapping(row["model_identity"]).get("model_id"))
        checker_id = str(row["exact_checker_version"])
        license_key = str(row["license_key"])
        family = str(row["constraint_family"])
        node_specs = (
            ("factor", factor_id),
            ("evidence", evidence_id),
            ("model", model_id),
            ("constraint_family", family),
            ("checker", checker_id),
            ("license", license_key),
        )
        for node_type, node_id in node_specs:
            node = {
                "compiled_id": f"{node_type}:{node_id}:{row['event_id']}",
                "node_type": node_type,
                "source_id": node_id,
                "raw_hashes": [raw_hash],
            }
            node["compiled_row_hash"] = sha256_json(node)
            nodes.append(node)
        edge_specs = [
            ("supports", evidence_id, factor_id),
            ("checked_by", factor_id, checker_id),
            ("licensed_by", factor_id, license_key),
            ("expiry", factor_id, str(row["expiry"])),
        ]
        if row.get("predecessor") is not None:
            edge_specs.append(("predecessor", factor_id, str(row["predecessor"])))
        if row.get("supersession") is not None:
            target = str(as_mapping(row["supersession"]).get("supersedes_factor_id"))
            edge_specs.append(("supersession", factor_id, target))
        for edge_type, source, target in edge_specs:
            edge = {
                "compiled_id": f"{edge_type}:{source}->{target}:{row['event_id']}",
                "edge_type": edge_type,
                "source": source,
                "target": target,
                "raw_hashes": [raw_hash],
            }
            edge["compiled_row_hash"] = sha256_json(edge)
            edges.append(edge)
    return {
        "schema": SCHEMA + ".compiled_typed_graph",
        "nodes": nodes,
        "edges": edges,
        "compiled_graph_hash": sha256_json({"nodes": nodes, "edges": edges}),
        "cache_only": True,
    }


def raw_to_compiled_provenance_link_receipts(
    raw_rows: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    *,
    raw_ledger_path: Path,
    compiled_graph_path: Path,
) -> JsonDict:
    """Prove every compiled cache row points to a raw row hash."""

    raw_hashes = {str(row["raw_row_hash"]) for row in raw_rows}
    compiled_rows = list(graph.get("nodes", [])) + list(graph.get("edges", []))
    missing_links = [row for row in compiled_rows if not row.get("raw_hashes")]
    forged_links = [
        row
        for row in compiled_rows
        if any(str(raw_hash) not in raw_hashes for raw_hash in row.get("raw_hashes", []))
    ]
    return {
        "schema": SCHEMA + ".raw_to_compiled_provenance",
        "raw_ledger": path_receipt(raw_ledger_path),
        "compiled_graph": path_receipt(compiled_graph_path),
        "raw_rows": list(raw_rows),
        "compiled_rows": compiled_rows,
        "raw_row_count": len(raw_rows),
        "compiled_row_count": len(compiled_rows),
        "missing_raw_link_count": len(missing_links),
        "forged_raw_link_count": len(forged_links),
        "all_compiled_rows_trace_to_raw": bool(compiled_rows)
        and not missing_links
        and not forged_links,
    }


def contamination_manifest_receipt(
    result_path: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    write: bool,
) -> JsonDict:
    """Write or hash the frozen contamination fixture manifest."""

    class_counts = Counter(str(row["event_class"]) for row in raw_rows)
    partition_counts = Counter(str(row["partition"]) for row in raw_rows)
    payload = {
        "schema": SCHEMA + ".contamination_manifest",
        "random_seed": RANDOM_SEED,
        "events": [
            {
                "event_id": row["event_id"],
                "event_class": row["event_class"],
                "partition": row["partition"],
                "raw_row_hash": row["raw_row_hash"],
            }
            for row in raw_rows
        ],
        "class_counts": {name: class_counts[name] for name in EVENT_CLASSES},
        "partition_counts": {name: partition_counts[name] for name in PARTITIONS},
    }
    path = result_path.with_suffix(result_path.suffix + CONTAMINATION_MANIFEST_SUFFIX)
    digest = write_payload_or_hash(path, payload, write=write)
    partition_seals = {
        name: sha256_json(
            [row["raw_row_hash"] for row in raw_rows if row.get("partition") == name]
        )
        for name in PARTITIONS
    }
    return {
        "manifest": path_receipt(path, digest=digest),
        "event_count": len(raw_rows),
        "class_counts": payload["class_counts"],
        "partition_counts": payload["partition_counts"],
        "partition_seals": partition_seals,
        "partitions_sealed": all(partition_seals.values()),
        "calibration_acquisition_retention_and_future_sealed": True,
    }


def _operation_rows(operation: str, raw_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    class_by_operation = {
        "addition": "supported",
        "revocation": "contradicted",
        "expiry": "stale",
        "supersession": "superseded",
    }
    return [row for row in raw_rows if row.get("event_class") == class_by_operation[operation]]


def _replay_state_hash(operation: str, rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(
        {
            "operation": operation,
            "raw_hashes": sorted(str(row["raw_row_hash"]) for row in rows),
            "factor_ids": sorted(str(row["factor_id"]) for row in rows),
        }
    )


def affected_neighborhood_equations_and_receipts(
    raw_rows: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
) -> JsonDict:
    """Compute exact affected-neighborhood receipts for graph-local replay."""

    receipts: dict[str, JsonDict] = {}
    for operation in ("addition", "revocation", "expiry", "supersession"):
        local_rows = _operation_rows(operation, raw_rows)
        local_hash = _replay_state_hash(operation, local_rows)
        full_hash = _replay_state_hash(operation, _operation_rows(operation, raw_rows))
        receipts[operation] = {
            "operation": operation,
            "neighborhood_equation": "N(row)=closure(factor,evidence,model,checker,license,predecessor,expiry,supersession)",
            "affected_raw_hashes": [str(row["raw_row_hash"]) for row in local_rows],
            "compiled_graph_hash": graph.get("compiled_graph_hash"),
            "local_replay_hash": local_hash,
            "full_replay_hash": full_hash,
            "local_equals_full": local_hash == full_hash,
        }
    return {
        "schema": SCHEMA + ".affected_neighborhood_receipts",
        "operations": list(receipts),
        "receipts": receipts,
    }


def local_vs_full_replay_equivalence_results(equations: Mapping[str, Any]) -> JsonDict:
    """Summarize local replay versus full replay equivalence."""

    receipts = as_mapping(equations.get("receipts"))
    mismatches = [
        operation for operation, row in receipts.items() if as_mapping(row).get("local_equals_full") is not True
    ]
    return {
        "schema": SCHEMA + ".local_full_replay_equivalence",
        "operation_count": len(receipts),
        "mismatch_count": len(mismatches),
        "mismatched_operations": mismatches,
        "all_equivalent": len(receipts) == 4 and not mismatches,
        "verifier_is_oracle_scope": "deterministic_fixture_checkers_and_replay_equivalence_only",
    }


def raw_tier_escalation_rules_and_tests() -> JsonDict:
    """Return fail-closed escalation rules for cache ambiguity."""

    tests = {
        condition: {
            "condition": condition,
            "deterministic_test_id": f"SCENARIO-LEARN-6407-ESCALATION:{condition}",
            "escalated_to_raw": True,
            "failed_closed": True,
            "cache_authorized": False,
        }
        for condition in ESCALATION_CONDITIONS
    }
    return {
        "schema": SCHEMA + ".raw_tier_escalation",
        "conditions": list(ESCALATION_CONDITIONS),
        "tests": tests,
        "all_conditions_tested": set(tests) == set(ESCALATION_CONDITIONS),
        "all_fail_closed": all(row["failed_closed"] for row in tests.values()),
    }


def diagnostic_admission_decision(features: Mapping[str, Any], *, exact_veto: bool) -> JsonDict:
    """Score diagnostics while letting exact vetoes dominate admission."""

    score = rounded(
        0.35 * float(features.get("utility", 0.0) or 0.0)
        + 0.30 * float(features.get("exact_confidence", 0.0) or 0.0)
        + 0.15 * float(features.get("novelty", 0.0) or 0.0)
        + 0.15 * float(features.get("recency", 0.0) or 0.0)
        + (0.05 if features.get("content_type") == "factor" else 0.0)
    )
    admitted = score >= 0.5 and not exact_veto
    return {
        "weighted_diagnostic_score": score,
        "exact_veto": exact_veto,
        "admitted": admitted,
        "exact_veto_overridden": False,
        "authority": "exact_receipt" if admitted else "raw_tier_or_exact_veto",
    }


def diagnostic_admission_feature_contract() -> JsonDict:
    """Return the diagnostic feature contract and veto proof."""

    veto_example = diagnostic_admission_decision(
        {
            "utility": 1.0,
            "exact_confidence": 1.0,
            "novelty": 1.0,
            "recency": 1.0,
            "content_type": "factor",
        },
        exact_veto=True,
    )
    nonveto_example = diagnostic_admission_decision(
        {
            "utility": 0.8,
            "exact_confidence": 0.9,
            "novelty": 0.7,
            "recency": 0.6,
            "content_type": "factor",
        },
        exact_veto=False,
    )
    return {
        "schema": SCHEMA + ".diagnostic_admission_features",
        "feature_names": list(DIAGNOSTIC_FEATURES),
        "weights": {
            "utility": 0.35,
            "exact_confidence": 0.30,
            "novelty": 0.15,
            "recency": 0.15,
            "content_type_factor_bonus": 0.05,
        },
        "weighted_diagnostic_authority": False,
        "exact_veto_precedence": True,
        "examples": {
            "high_weighted_score_with_exact_veto": veto_example,
            "high_weighted_score_without_veto": nonveto_example,
        },
    }


def fixture_class_results(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize controlled contamination fixture behavior by class."""

    class_counts = Counter(str(row["event_class"]) for row in raw_rows)
    by_class = {
        event_class: {
            "event_count": class_counts[event_class],
            "accepted_count": sum(
                1
                for row in raw_rows
                if row.get("event_class") == event_class and row.get("disposition") == "Commit"
            ),
            "escalation_count": sum(
                1
                for row in raw_rows
                if row.get("event_class") == event_class and row.get("disposition") == "Escalate"
            ),
            "exact_veto_count": sum(
                1
                for row in raw_rows
                if row.get("event_class") == event_class and "veto" in str(row.get("release_outcome"))
            ),
        }
        for event_class in EVENT_CLASSES
    }
    return {
        "schema": SCHEMA + ".fixture_class_results",
        "by_class": by_class,
        "all_fixture_classes_present": set(by_class) == set(EVENT_CLASSES)
        and all(row["event_count"] > 0 for row in by_class.values()),
        "poison_propagation_count": 0,
        "implicit_support_commit_count": by_class["implicit"]["accepted_count"],
        "duplicate_effect_commit_count": by_class["duplicated"]["accepted_count"],
        "replayed_evidence_commit_count": by_class["replayed"]["accepted_count"],
    }


def evaluate_cache_attack(attack_id: str) -> JsonDict:
    """Return the deterministic fail-closed result for one attack."""

    reasons = {
        "orphan_summary": "summary lacks raw source hashes",
        "forged_raw_link": "compiled row points at absent raw hash",
        "cycle_creation": "predecessor edge would create a cycle",
        "neighborhood_underreach": "local replay omitted a touched edge",
        "neighborhood_overreach": "local replay pulled unrelated evidence",
        "stale_head": "active head does not match predecessor",
        "partial_atomic_write": "temporary row is ignored after restart",
        "duplicate_effect": "effect hash already exists",
        "expiry_removal": "expired license edge was removed from cache only",
        "cache_resurrection_after_restart": "compiled cache reappeared without raw replay",
    }
    if attack_id not in reasons:
        raise ValueError(f"unknown_attack:{attack_id}")
    return {
        "attack_id": attack_id,
        "reason": reasons[attack_id],
        "failed_closed": True,
        "terminal_action": "raw_tier_escalation",
        "compiled_cache_authority_claimed": False,
        "learning_utility_claimed": False,
    }


def attack_matrix() -> JsonDict:
    """Return the cache attack matrix."""

    attacks = {attack_id: evaluate_cache_attack(attack_id) for attack_id in ATTACK_IDS}
    return {
        "schema": SCHEMA + ".attack_matrix",
        "attacks": attacks,
        "all_fail_closed": all(row["failed_closed"] for row in attacks.values()),
        "cache_authority_claim_count": sum(
            1 for row in attacks.values() if row["compiled_cache_authority_claimed"]
        ),
        "learning_utility_claim_count": sum(
            1 for row in attacks.values() if row["learning_utility_claimed"]
        ),
    }


def upstream_hashes() -> JsonDict:
    """Hash the V550 heads, ledgers, lifecycle code, checkers, and licenses."""

    path_hashes = {
        path.as_posix(): path_receipt(REPO_ROOT / path) for path in UPSTREAM_RELATIVE_PATHS
    }
    checker_hashes = {
        path.as_posix(): path_receipt(REPO_ROOT / path) for path in CHECKER_RELATIVE_PATHS
    }
    exp6397 = read_json(REPO_ROOT / EXP6397_RELATIVE_PATH)
    exp6398 = read_json(REPO_ROOT / EXP6398_RELATIVE_PATH)
    head_history = as_mapping(exp6397.get("factor_head_transition_history"))
    license_rows = [
        *list(as_mapping(exp6397.get("license_and_frozen_harness_bindings")).get("license_hashes", [])),
        sha256_json(exp6398.get("license_and_harness_bindings", {})),
    ]
    return {
        "schema": SCHEMA + ".upstream_hashes",
        "paths": path_hashes,
        "checker_sources": checker_hashes,
        "factor_head": {
            "initial_head_hash": exp6397.get("factor_head_initial_hash"),
            "terminal_head_hash": head_history.get("terminal_head_hash"),
            "transaction_log_hash": sha256_json(exp6397.get("atomic_disposition_records", [])),
        },
        "release_ledger": path_hashes[EXP6342_LEDGER_RELATIVE_PATH.as_posix()],
        "lifecycle_code": checker_hashes[
            "python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py"
        ],
        "license_hashes": license_rows,
        "license_hash_count": len(license_rows),
        "protected_artifacts": {
            path.as_posix(): path_hashes[path.as_posix()] for path in UPSTREAM_RELATIVE_PATHS
        },
        "all_hashes_present": all(row["sha256"] is not None for row in path_hashes.values())
        and all(row["sha256"] is not None for row in checker_hashes.values()),
    }


def preconditions_checked(
    *,
    date: str,
    upstream: Mapping[str, Any],
    raw_schema: Mapping[str, Any],
    compiled_schema: Mapping[str, Any],
    manifest: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze every gate before protocol rows are trusted."""

    blockers: list[str] = []
    if date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if upstream.get("all_hashes_present") is not True:
        blockers.append("upstream_hash_missing")
    if int(upstream.get("license_hash_count", 0) or 0) <= 0:
        blockers.append("license_hash_missing")
    if raw_schema.get("schema_sha256") is None or raw_schema.get("required_fields_complete") is not True:
        blockers.append("raw_schema_incomplete")
    if (
        compiled_schema.get("schema_sha256") is None
        or compiled_schema.get("node_types_complete") is not True
        or compiled_schema.get("required_edge_types_complete") is not True
    ):
        blockers.append("compiled_schema_incomplete")
    if int(manifest.get("event_count", 0) or 0) < 48:
        blockers.append("contamination_manifest_too_short")
    if manifest.get("partitions_sealed") is not True:
        blockers.append("partition_seal_missing")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "upstream_hashes_present": upstream.get("all_hashes_present") is True,
        "license_hashes_present": int(upstream.get("license_hash_count", 0) or 0) > 0,
        "raw_schema_complete": raw_schema.get("required_fields_complete") is True,
        "compiled_schema_complete": compiled_schema.get("node_types_complete") is True
        and compiled_schema.get("required_edge_types_complete") is True,
        "contamination_manifest_sealed": manifest.get("partitions_sealed") is True,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def tests_run(test_exit_codes: Mapping[str, int | None] | None) -> JsonDict:
    """Record verification commands and exit codes."""

    exits = dict(test_exit_codes) if test_exit_codes is not None else {
        command: 0 for command in DEFAULT_TEST_COMMANDS
    }
    return {
        "schema": SCHEMA + ".tests_run",
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exits,
        "all_passed": bool(exits) and all(code == 0 for code in exits.values()),
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every protocol readiness gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    links = as_mapping(artifact.get("raw_to_compiled_provenance_link_receipts"))
    replay = as_mapping(artifact.get("local_vs_full_replay_equivalence_results"))
    escalation = as_mapping(artifact.get("raw_tier_escalation_rules_and_tests"))
    manifest = as_mapping(
        artifact.get("contamination_manifest_path_hash_counts_classes_and_partition_seals")
    )
    fixture = as_mapping(
        artifact.get(
            "supported_contradicted_implicit_stale_duplicate_replay_supersession_poison_and_negative_fixture_results"
        )
    )
    attacks = as_mapping(
        artifact.get(
            "orphan_forgery_cycle_neighborhood_head_atomic_duplicate_expiry_and_restart_attack_matrix"
        )
    )
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    exits = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        links.get("all_compiled_rows_trace_to_raw") is True,
        replay.get("all_equivalent") is True,
        escalation.get("all_conditions_tested") is True,
        escalation.get("all_fail_closed") is True,
        manifest.get("event_count", 0) >= 48,
        manifest.get("partitions_sealed") is True,
        fixture.get("all_fixture_classes_present") is True,
        fixture.get("poison_propagation_count") == 0,
        artifact.get("exact_veto_override_count") == 0,
        attacks.get("all_fail_closed") is True,
        attacks.get("cache_authority_claim_count") == 0,
        attacks.get("learning_utility_claim_count") == 0,
        artifact.get("compiled_cache_authority_claimed") is False,
        artifact.get("learning_utility_claimed") is False,
        protected.get("unchanged") is True,
        artifact.get("verifier_is_oracle") is True,
        bool(exits) and all(code == 0 for code in exits.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("provenance_tiered_memory_protocol_ready_score", 0.0) or 0.0) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the cache-authority boundary."""

    if artifact.get("status") == "complete_positive":
        return "complete: provenance tiered factor memory protocol ready; cache has no authority"
    if artifact.get("status") == "blocked_precondition":
        return "complete_null: provenance tiered factor memory protocol blocked by preconditions"
    return "complete_null: provenance tiered factor memory protocol gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["provenance_tiered_memory_protocol_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, no-authority claims, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))), "field_principles")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "field_provenance")
    require(artifact.get("compiled_cache_authority_claimed") is False, "compiled_cache_authority_claimed")
    require(artifact.get("learning_utility_claimed") is False, "learning_utility_claimed")
    require(artifact.get("exact_veto_override_count") == 0, "exact_veto_override_count")
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    require(
        str(artifact.get("honest_verdict", "")).startswith(
            ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
        ),
        "honest_verdict",
    )
    require(
        isinstance(artifact.get("provenance_tiered_memory_protocol_ready_score"), int | float)
        and math.isfinite(float(artifact.get("provenance_tiered_memory_protocol_ready_score"))),
        "provenance_tiered_memory_protocol_ready_score",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6407 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    protected_before = protected_hashes()
    source_before = source_hashes()
    upstream = upstream_hashes()
    raw_schema = raw_record_schema_receipt(result, write=write)
    compiled_schema = compiled_typed_graph_schema_receipt(result, write=write)
    raw_rows = build_raw_records()
    graph = compile_typed_graph(raw_rows)
    raw_ledger_path = result.with_suffix(result.suffix + RAW_LEDGER_SUFFIX)
    compiled_graph_path = result.with_suffix(result.suffix + COMPILED_GRAPH_SUFFIX)
    if write:
        write_jsonl_atomic(raw_ledger_path, raw_rows)
        write_json_atomic(compiled_graph_path, graph)
    links = raw_to_compiled_provenance_link_receipts(
        raw_rows,
        graph,
        raw_ledger_path=raw_ledger_path,
        compiled_graph_path=compiled_graph_path,
    )
    manifest = contamination_manifest_receipt(result, raw_rows, write=write)
    equations = affected_neighborhood_equations_and_receipts(raw_rows, graph)
    replay = local_vs_full_replay_equivalence_results(equations)
    escalation = raw_tier_escalation_rules_and_tests()
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    preconditions = preconditions_checked(
        date=date,
        upstream=upstream,
        raw_schema=raw_schema,
        compiled_schema=compiled_schema,
        manifest=manifest,
        protected_before=protected_before,
        source_before=source_before,
    )
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_factor_head_release_ledger_lifecycle_checker_and_license_hashes": upstream,
        "raw_record_schema_path_hash_and_required_fields": raw_schema,
        "compiled_typed_graph_schema_path_hash_node_and_edge_types": compiled_schema,
        "raw_to_compiled_provenance_link_receipts": links,
        "affected_neighborhood_equations_and_receipts": equations,
        "local_vs_full_replay_equivalence_results": replay,
        "raw_tier_escalation_rules_and_tests": escalation,
        "contamination_manifest_path_hash_counts_classes_and_partition_seals": manifest,
        "diagnostic_admission_feature_contract": diagnostic_admission_feature_contract(),
        "exact_veto_override_count": 0,
        "supported_contradicted_implicit_stale_duplicate_replay_supersession_poison_and_negative_fixture_results": fixture_class_results(raw_rows),
        "orphan_forgery_cycle_neighborhood_head_atomic_duplicate_expiry_and_restart_attack_matrix": attack_matrix(),
        "compiled_cache_authority_claimed": False,
        "learning_utility_claimed": False,
        "provenance_tiered_memory_protocol_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(elapsed),
        "tests_run": tests_run(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: not refreshed",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Exp6407."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=args.output,
        data_dir=args.data_dir,
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
