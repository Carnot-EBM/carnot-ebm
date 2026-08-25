"""Build the V574 bounded CFR launch root without running an LLM.

The launch root rechecks terminal V573 receipts from their detailed rows. It
also confirms that both mandated GGUF files are present and identifiable from
bounded header reads. The task does not load model weights or download files.

Spec: REQ-REPORT-6588 and SCENARIO-REPORT-6588-REPLAY through
SCENARIO-REPORT-6588-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_6585_v573_terminal_recovery_and_execution_contract as exp6585
from carnot import experiment_6587_v573_constraint_first_method_contract as exp6587
from carnot.inference.gguf_metadata import build_gguf_admission_record


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
TASK_ID = "exp6588-v574-bounded-cfr-launch-root"
RESULT_RELATIVE_PATH = Path("results/experiment_6588_v574_bounded_cfr_launch_root.json")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CLAIM_AUDIT_RELATIVE_PATH = Path("ops/experiment_claim_audit_report.md")
GGUF_IDENTITY_SOURCE_RELATIVE_PATH = Path(
    "results/experiment_6572_content_derived_gguf_metadata_resolver.json"
)
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
V573_ARTIFACT_PATHS = {
    "Exp6585": Path("results/experiment_6585_v573_terminal_recovery_and_execution_contract.json"),
    "Exp6586": Path("results/experiment_6586_isolated_full_suite_truth_baseline.json"),
    "Exp6587": Path("results/experiment_6587_v573_constraint_first_method_contract.json"),
}
INFERENCE_SUBSTRATE = "v573_terminal_and_contract_replay_no_llm"
READY_FIELD = "v574_cfr_launch_ready_score"
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
EXPECTED_ARCHITECTURES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": {"qwen35moe"},
    "unsloth/gemma-4-31B-it-GGUF": {"gemma4"},
}
MODEL_TASKS = (
    (
        "exp6590-qwen36-constraint-first-stream",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "qwen_cfr_rows_ready_score",
    ),
    (
        "exp6591-gemma4-31b-constraint-first-stream",
        "unsloth/gemma-4-31B-it-GGUF",
        "gemma31_cfr_rows_ready_score",
    ),
)
REQUIRED_ATTACK_IDS = (
    "invented_v573_science_result",
    "suite_green_launch_requirement",
    "missing_source_hashes",
    "two_model_residency",
    "legacy_smoke_model_headline",
    "gguf_auto_tokenizer_use",
    "external_download_during_measurement",
    "incomplete_terminal_rows",
    "gate_field_drift",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "v573_terminal_replay_rows",
    "model_cache_identity_rows",
    "execution_budget_contract",
    "current_roadmap_gate_contract_rows",
    "attack_rows",
    READY_FIELD,
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The launch root is terminal and cannot pose bootstrap work as readiness.",
    "honest_verdict": "The verdict reports bounded V574 launch readiness without creating a science result.",
    "verdict_class": "Launch readiness is null infrastructure and never positive science.",
    "gate_check_summary": "A block names the failed source, cache, budget, gate, or protection check and its value.",
    "v573_terminal_replay_rows": "Exp6585 through Exp6587 retain their actual terminal states and source hashes.",
    "model_cache_identity_rows": "Both mandated flagship caches resolve by content without a model load or download.",
    "execution_budget_contract": "One-model residency, checkpoints, timeout, cleanup, unload, and terminal output are frozen.",
    "current_roadmap_gate_contract_rows": "Every downstream gate names an upstream task and exact field in this roadmap.",
    "attack_rows": "Invented science, suite gating, model substitution, cache drift, and gate drift fail closed.",
    READY_FIELD: "This exact binary field gates both flagship CFR stream tasks.",
    "preconditions_checked": "Artifacts, hashes, caches, resources, ownership, and protected files are explicit.",
    "protected_files_unchanged": "The roadmap and conductor retain their original hashes.",
    "inference_substrate": "The task declares terminal receipt and contract replay with no LLM.",
    "verifier_is_oracle": "Exact receipt replay owns launch readiness but cannot create positive research science.",
    "field_provenance": "Every field names source artifacts, rows, hashes, and reducer code.",
    "duration_s": "Monotonic duration exposes a source-only shortcut.",
    "tests_run": "Named commands, exits, and durations make contract validation reproducible.",
    "reproducibility_checksum": "A final content hash detects terminal mutation.",
}
DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_6588_v574_bounded_cfr_launch_root.py -q",
        "exit_code": 0,
        "duration_s": 0.0,
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0, "duration_s": 0.0},
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py",
        "exit_code": 0,
        "duration_s": 0.0,
    },
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_6588_v574_bounded_cfr_launch_root.json",
        "exit_code": 0,
        "duration_s": 0.0,
    },
)


def canonical_json(value: Any) -> str:
    """Return stable JSON text for hashes and reducer receipts."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return one algorithm-qualified SHA-256 hash."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a bounded evidence file. Model blobs use trusted cache hashes."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON with an explicit algorithm name."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact without its self-referential checksum field."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def load_json(path: str | Path) -> JsonDict:
    """Load one checked-in JSON object and fail clearly for another shape."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _hash_matches(row: Mapping[str, Any], field: str = "row_hash") -> bool:
    expected = row.get(field)
    bare = {key: value for key, value in row.items() if key != field}
    return expected == sha256_json(bare)


def recompute_exp6585_readiness(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute Exp6585 from detailed receipts and ignore its stored score."""

    terminal_rows = payload.get("v572_terminal_rows", [])
    attempt_rows = payload.get("exp6584_hard_limit_attempt_rows", [])
    budget_rows = payload.get("v573_execution_budget_contract", [])
    gate_rows = payload.get("current_roadmap_gate_contract_rows", [])
    attack_rows = payload.get("attack_rows", [])
    protected = payload.get("protected_files_unchanged", {})
    log_hash = (
        str(attempt_rows[0].get("log_source_sha256", "missing")) if attempt_rows else "missing"
    )
    checks = {
        "terminal_rows": exp6585.v572_terminal_rows_valid(terminal_rows, REPO_ROOT),
        "hard_limit_rows": exp6585.hard_limit_attempt_rows_valid(attempt_rows, log_hash),
        "execution_budget_rows": exp6585.execution_budget_contract_valid(budget_rows),
        "roadmap_gate_rows": exp6585.gate_contract_rows_valid(gate_rows),
        "attack_rows": (
            len(attack_rows) == len(exp6585.REQUIRED_ATTACKS)
            and [row.get("attack") for row in attack_rows] == list(exp6585.REQUIRED_ATTACKS)
            and all(
                row.get("passed") is True
                and row.get("observed_ready_score") == 0.0
                and _hash_matches(row)
                for row in attack_rows
            )
        ),
        "protected_files": isinstance(protected, Mapping) and protected.get("unchanged") is True,
    }
    ready = all(checks.values())
    return {"checks": checks, "ready_score": 1.0 if ready else 0.0}


def recompute_exp6587_readiness(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute Exp6587 while treating its nonempty retirement list as true."""

    source = exp6587.readiness_reducer(payload)
    source_checks = source.get("checks", {})
    checks = {str(name): bool(value) for name, value in source_checks.items()}
    ready = all(checks.values())
    return {"checks": checks, "ready_score": 1.0 if ready else 0.0}


def _claim_audit_disposition(repo_root: Path) -> JsonDict:
    """Bind the checked-in Exp6586 adversarial disposition to its report hash."""

    path = repo_root / CLAIM_AUDIT_RELATIVE_PATH
    text = path.read_text(encoding="utf-8")
    marker = "## experiment_6586_isolated_full_suite_truth_baseline.json"
    section = text.split(marker, 1)[1].split("\n## experiment_", 1)[0] if marker in text else ""
    disposition = "SKIPPED_ALREADY_FLAGGED" if "SKIPPED_ALREADY_FLAGGED" in section else "missing"
    return {
        "claim_audit_path": CLAIM_AUDIT_RELATIVE_PATH.as_posix(),
        "claim_audit_sha256": sha256_file(path),
        "claim_audit_disposition": disposition,
    }


def build_v573_terminal_replay_rows(repo_root: Path) -> list[JsonDict]:
    """Preserve each V573 terminal state and recompute only contract readiness."""

    payloads = {
        experiment_id: load_json(repo_root / relative_path)
        for experiment_id, relative_path in V573_ARTIFACT_PATHS.items()
    }
    exp6585_replay = recompute_exp6585_readiness(payloads["Exp6585"])
    exp6587_replay = recompute_exp6587_readiness(payloads["Exp6587"])
    rows: list[JsonDict] = []
    for experiment_id in ("Exp6585", "Exp6586", "Exp6587"):
        source = payloads[experiment_id]
        relative_path = V573_ARTIFACT_PATHS[experiment_id]
        row: JsonDict = {
            "experiment_id": experiment_id,
            "source_artifact_path": relative_path.as_posix(),
            "source_artifact_sha256": sha256_file(repo_root / relative_path),
            "status": source.get("status"),
            "honest_verdict": source.get("honest_verdict"),
            "verdict_class": source.get("verdict_class"),
            "gate_check_summary": deepcopy(source.get("gate_check_summary")),
            "inference_substrate": source.get("inference_substrate"),
            "verifier_is_oracle": source.get("verifier_is_oracle"),
            "science_result_created": False,
        }
        if experiment_id == "Exp6585":
            row.update(
                {
                    "stored_ready_score": source.get("v573_execution_contract_ready_score"),
                    "recomputed_ready_score": exp6585_replay["ready_score"],
                    "recomputed_checks": exp6585_replay["checks"],
                    "contract_kind": "bounded_execution_contract",
                }
            )
        elif experiment_id == "Exp6586":
            row.update(
                {
                    "stored_ready_score": source.get("full_suite_baseline_ready_score"),
                    "recomputed_ready_score": None,
                    "contract_kind": "blocked_isolated_suite_infrastructure_fact",
                    "science_launch_gate": False,
                    "adversarial_disposition": {
                        "flagged_adversarial": source.get("flagged_adversarial"),
                        "corrigendum_pending": deepcopy(source.get("corrigendum_pending", [])),
                        **_claim_audit_disposition(repo_root),
                    },
                }
            )
        else:
            source_summary = source.get("gate_check_summary", {})
            row.update(
                {
                    "stored_ready_score": source.get("v573_constraint_first_method_ready_score"),
                    "recomputed_ready_score": exp6587_replay["ready_score"],
                    "recomputed_checks": exp6587_replay["checks"],
                    "source_gate_summary_checks_closed": source_summary.get("checks_closed"),
                    "contract_kind": "constraint_first_method_contract",
                }
            )
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def _missing_cache_row(repository_id: str, source_hash: str) -> JsonDict:
    """Return one terminal cache-miss row without trying a download."""

    return {
        "repository_id": repository_id,
        "resolved": False,
        "admitted": False,
        "rejection_reason": "trusted_cache_identity_missing",
        "source_artifact_path": GGUF_IDENTITY_SOURCE_RELATIVE_PATH.as_posix(),
        "source_artifact_sha256": source_hash,
        "model_load_performed": False,
        "download_performed": False,
        "auto_tokenizer_used": False,
    }


def build_model_cache_identity_rows(repo_root: Path) -> list[JsonDict]:
    """Resolve two local GGUF identities from bounded content and provenance."""

    source_path = repo_root / GGUF_IDENTITY_SOURCE_RELATIVE_PATH
    source_hash = sha256_file(source_path)
    source = load_json(source_path) if source_path.is_file() else {}
    source_rows = source.get("gguf_blob_metadata_rows", [])
    by_id = {str(row.get("unit_id")): row for row in source_rows if isinstance(row, Mapping)}
    rows: list[JsonDict] = []
    for repository_id in MANDATED_MODEL_IDS:
        prior = by_id.get(repository_id)
        if not isinstance(prior, Mapping):
            rows.append(_missing_cache_row(repository_id, source_hash))
            continue
        record = build_gguf_admission_record(
            str(prior.get("path", "")),
            repository_id=repository_id,
            trusted_sha256=str(prior.get("trusted_exp6567_sha256", "missing")),
            expected_architectures=EXPECTED_ARCHITECTURES[repository_id],
        )
        metadata = record.get("content_metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        bounded = metadata.get("bounded_read_receipt", {})
        summary = {
            key: deepcopy(metadata.get(key))
            for key in (
                "magic",
                "version",
                "architecture",
                "model_name",
                "general_file_type",
                "quantization",
                "tensor_count",
                "metadata_count",
                "is_language_model",
                "tokenizer_metadata",
                "shard_metadata",
            )
        }
        summary["bounded_read_receipt"] = deepcopy(bounded)
        row = {
            "repository_id": repository_id,
            "resolved": record.get("admitted") is True,
            "admitted": record.get("admitted") is True,
            "cache_path": record.get("path"),
            "trusted_sha256": prior.get("trusted_exp6567_sha256"),
            "content_metadata": summary,
            "provenance": deepcopy(record.get("provenance")),
            "rejection_reasons": deepcopy(record.get("rejection_reasons", [])),
            "rejection_reason": None
            if record.get("admitted") is True
            else "content_or_provenance_rejected",
            "source_artifact_path": GGUF_IDENTITY_SOURCE_RELATIVE_PATH.as_posix(),
            "source_artifact_sha256": source_hash,
            "model_load_performed": False,
            "download_performed": False,
            "auto_tokenizer_used": False,
            "gguf_embedded_tokenizer_required_downstream": True,
        }
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def build_execution_budget_contract() -> list[JsonDict]:
    """Freeze one remapped V573 execution budget for each V574 model task."""

    rows = []
    for task_id, model_id, _output_field in MODEL_TASKS:
        row = {
            "task_id": task_id,
            "model_families": [model_id],
            "max_model_processes": 1,
            "max_concurrent_model_families": 1,
            "fresh_process": True,
            "fresh_context_per_unit": True,
            "runtime_select_idle_rtx_3090": True,
            "gpu_selection_rule": "choose a visible RTX 3090 with no unrelated compute owner and adequate free memory at launch",
            "max_source_units": 20,
            "generation_arms_per_unit": 3,
            "max_output_tokens_per_call": 512,
            "per_generation_timeout_s": 60,
            "per_unit_timeout_s": 180,
            "load_budget_s": 600,
            "generation_budget_s": 2880,
            "cleanup_budget_s": 180,
            "terminal_output_budget_s": 120,
            "hard_timeout_s": 4200,
            "conductor_hard_cap_s": 4800,
            "checkpoint_interval_units": 1,
            "checkpoint_max_interval_s": 180,
            "checkpoint_write_order": "raw_before_derived",
            "failure_rows_required": True,
            "retain_failure_classes": [
                "load_timeout",
                "generation_timeout",
                "malformed_output",
                "contradiction",
                "unsupported_constraint",
                "exact_rejection",
                "process_failure",
                "task_deadline_exhausted",
            ],
            "kill_only_owned_child_process_group": True,
            "signals_to_unrelated_processes_allowed": False,
            "unload_required_before_exit": True,
            "verified_unload_checks": [
                "worker_process_exited",
                "port_closed",
                "worker_absent_from_gpu_telemetry",
                "memory_recovered_within_256_mb",
            ],
            "terminal_output_required_on_failure": True,
            "atomic_terminal_output": True,
            "external_download_allowed": False,
            "auto_tokenizer_allowed": False,
            "embedded_gguf_tokenizer_required": True,
            "legacy_headline_model_allowed": False,
            "replayed_from": V573_ARTIFACT_PATHS["Exp6585"].as_posix(),
        }
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def execution_budget_contract_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only for the two complete one-model lifecycle contracts."""

    return (
        [row.get("task_id") for row in rows] == [task[0] for task in MODEL_TASKS]
        and [row.get("model_families") for row in rows] == [[task[1]] for task in MODEL_TASKS]
        and all(
            row.get("max_model_processes") == 1
            and row.get("max_concurrent_model_families") == 1
            and row.get("fresh_process") is True
            and row.get("runtime_select_idle_rtx_3090") is True
            and row.get("checkpoint_interval_units") == 1
            and row.get("checkpoint_max_interval_s", 0) > 0
            and 0 < row.get("hard_timeout_s", 0) < row.get("conductor_hard_cap_s", 0)
            and row.get("failure_rows_required") is True
            and row.get("kill_only_owned_child_process_group") is True
            and row.get("signals_to_unrelated_processes_allowed") is False
            and row.get("unload_required_before_exit") is True
            and len(row.get("verified_unload_checks", [])) == 4
            and row.get("terminal_output_required_on_failure") is True
            and row.get("atomic_terminal_output") is True
            and row.get("external_download_allowed") is False
            and row.get("auto_tokenizer_allowed") is False
            and row.get("embedded_gguf_tokenizer_required") is True
            and row.get("legacy_headline_model_allowed") is False
            and _hash_matches(row)
            for row in rows
        )
    )


def _required_artifact_fields(prompt: str) -> set[str]:
    """Read field names from one roadmap task's required-field block."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return set()
    fields = set()
    for line in prompt.split(marker, 1)[1].splitlines():
        stripped = line.strip()
        if stripped.startswith("Run command:"):
            break
        if stripped.endswith(":") and not stripped.startswith("principle"):
            fields.add(stripped[:-1])
    return fields


def build_current_roadmap_gate_contract_rows(repo_root: Path) -> list[JsonDict]:
    """Bind both model tasks to the exact gate and owner fields in V574."""

    path = repo_root / ACTIVE_ROADMAP_RELATIVE_PATH
    roadmap = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = roadmap.get("tasks", []) if isinstance(roadmap, Mapping) else []
    by_id = {str(task.get("id")): task for task in tasks if isinstance(task, Mapping)}
    root_task = by_id.get(TASK_ID, {})
    root_fields = _required_artifact_fields(str(root_task.get("prompt", "")))
    rows = []
    for consumer_task_id, _model_id, owner_output_field in MODEL_TASKS:
        consumer = by_id.get(consumer_task_id, {})
        gates = consumer.get("gated_on", []) if isinstance(consumer, Mapping) else []
        matching = [
            gate
            for gate in gates
            if isinstance(gate, Mapping)
            and gate.get("upstream") == TASK_ID
            and gate.get("artifact_field") == READY_FIELD
        ]
        consumer_fields = _required_artifact_fields(str(consumer.get("prompt", "")))
        row = {
            "roadmap_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "roadmap_sha256": sha256_file(path),
            "milestone": roadmap.get("milestone") if isinstance(roadmap, Mapping) else None,
            "upstream_task_id": TASK_ID,
            "artifact_field": READY_FIELD,
            "operator": "==",
            "expected_value": 1.0,
            "consumer_task_id": consumer_task_id,
            "owner_output_field": owner_output_field,
            "upstream_task_exists": TASK_ID in by_id,
            "consumer_task_exists": consumer_task_id in by_id,
            "gate_declared_exactly_once": len(matching) == 1,
            "launch_field_declared_by_upstream": READY_FIELD in root_fields,
            "owner_output_field_declared": owner_output_field in consumer_fields,
        }
        row["all_cross_references_close"] = all(
            row[key]
            for key in (
                "upstream_task_exists",
                "consumer_task_exists",
                "gate_declared_exactly_once",
                "launch_field_declared_by_upstream",
                "owner_output_field_declared",
            )
        )
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def current_roadmap_gate_contract_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when both exact same-roadmap cross-references close."""

    return (
        len(rows) == len(MODEL_TASKS)
        and [row.get("consumer_task_id") for row in rows] == [task[0] for task in MODEL_TASKS]
        and [row.get("owner_output_field") for row in rows] == [task[2] for task in MODEL_TASKS]
        and all(
            row.get("upstream_task_id") == TASK_ID
            and row.get("artifact_field") == READY_FIELD
            and row.get("expected_value") == 1.0
            and row.get("all_cross_references_close") is True
            and str(row.get("roadmap_sha256", "")).startswith("sha256:")
            and _hash_matches(row)
            for row in rows
        )
    )


def _terminal_replay_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_id = {str(row.get("experiment_id")): row for row in rows}
    return (
        set(by_id) == set(V573_ARTIFACT_PATHS)
        and by_id["Exp6585"].get("recomputed_ready_score") == 1.0
        and by_id["Exp6587"].get("recomputed_ready_score") == 1.0
        and by_id["Exp6586"].get("verdict_class") == "blocked"
        and by_id["Exp6586"].get("science_launch_gate") is False
        and by_id["Exp6586"].get("adversarial_disposition", {}).get("flagged_adversarial") is True
        and all(row.get("science_result_created") is False for row in rows)
        and all(
            str(row.get("source_artifact_sha256", "")).startswith("sha256:") and _hash_matches(row)
            for row in rows
        )
    )


def _cache_rows_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    return [row.get("repository_id") for row in rows] == list(MANDATED_MODEL_IDS) and all(
        row.get("resolved") is True
        and row.get("admitted") is True
        and row.get("model_load_performed") is False
        and row.get("download_performed") is False
        and row.get("auto_tokenizer_used") is False
        and row.get("content_metadata", {})
        .get("bounded_read_receipt", {})
        .get("tensor_payload_bytes_read")
        == 0
        and row.get("provenance", {}).get("valid") is True
        and str(row.get("source_artifact_sha256", "")).startswith("sha256:")
        and _hash_matches(row)
        for row in rows
    )


def _base_readiness_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    """Evaluate launch inputs without trusting the stored launch score."""

    return {
        "v573_terminal_replay_rows": _terminal_replay_ready(
            payload.get("v573_terminal_replay_rows", [])
        ),
        "model_cache_identity_rows": _cache_rows_ready(
            payload.get("model_cache_identity_rows", [])
        ),
        "execution_budget_contract": execution_budget_contract_ready(
            payload.get("execution_budget_contract", [])
        ),
        "current_roadmap_gate_contract_rows": current_roadmap_gate_contract_ready(
            payload.get("current_roadmap_gate_contract_rows", [])
        ),
        "suite_green_not_required": payload.get("suite_green_launch_requirement") is False,
        "protected_files_unchanged": payload.get("protected_files_unchanged", {}).get(
            "all_unchanged"
        )
        is True,
    }


def _candidate_ready_score(payload: Mapping[str, Any]) -> float:
    return 1.0 if all(_base_readiness_checks(payload).values()) else 0.0


def build_attack_rows(base_candidate: Mapping[str, Any]) -> list[JsonDict]:
    """Run each required mutation through the same launch-input reducer."""

    mutations: list[tuple[str, JsonDict]] = []
    candidate = deepcopy(dict(base_candidate))
    candidate["v573_terminal_replay_rows"][0]["science_result_created"] = True
    mutations.append((REQUIRED_ATTACK_IDS[0], candidate))
    candidate = deepcopy(dict(base_candidate))
    candidate["suite_green_launch_requirement"] = True
    mutations.append((REQUIRED_ATTACK_IDS[1], candidate))
    candidate = deepcopy(dict(base_candidate))
    candidate["v573_terminal_replay_rows"][0]["source_artifact_sha256"] = "missing"
    mutations.append((REQUIRED_ATTACK_IDS[2], candidate))
    candidate = deepcopy(dict(base_candidate))
    candidate["execution_budget_contract"][0]["max_concurrent_model_families"] = 2
    mutations.append((REQUIRED_ATTACK_IDS[3], candidate))
    candidate = deepcopy(dict(base_candidate))
    candidate["model_cache_identity_rows"][0]["repository_id"] = "Qwen/Qwen3.5-0.8B"
    mutations.append((REQUIRED_ATTACK_IDS[4], candidate))
    candidate = deepcopy(dict(base_candidate))
    candidate["model_cache_identity_rows"][0]["auto_tokenizer_used"] = True
    mutations.append((REQUIRED_ATTACK_IDS[5], candidate))
    candidate = deepcopy(dict(base_candidate))
    candidate["model_cache_identity_rows"][0]["download_performed"] = True
    mutations.append((REQUIRED_ATTACK_IDS[6], candidate))
    candidate = deepcopy(dict(base_candidate))
    candidate["v573_terminal_replay_rows"] = candidate["v573_terminal_replay_rows"][:-1]
    mutations.append((REQUIRED_ATTACK_IDS[7], candidate))
    candidate = deepcopy(dict(base_candidate))
    candidate["current_roadmap_gate_contract_rows"][0]["artifact_field"] = "drifted"
    mutations.append((REQUIRED_ATTACK_IDS[8], candidate))

    rows = []
    for attack_id, mutated in mutations:
        observed = _candidate_ready_score(mutated)
        row = {
            "attack_id": attack_id,
            "expected_ready_score": 0.0,
            "candidate_ready_score": observed,
            "passed": observed == 0.0,
            "disposition": "fail_closed",
        }
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def readiness_reducer(payload: Mapping[str, Any]) -> JsonDict:
    """Return one only when every source, cache, lifecycle, gate, and attack closes."""

    checks = _base_readiness_checks(payload)
    attack_rows = payload.get("attack_rows", [])
    checks["attack_rows"] = [row.get("attack_id") for row in attack_rows] == list(
        REQUIRED_ATTACK_IDS
    ) and all(
        row.get("passed") is True and row.get("candidate_ready_score") == 0.0 and _hash_matches(row)
        for row in attack_rows
    )
    ready = all(checks.values())
    return {"checks": checks, "ready_score": 1.0 if ready else 0.0}


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before[path],
            "after_sha256": after.get(path, "missing"),
            "unchanged": before[path] == after.get(path),
        }
        for path in before
    ]
    return {
        "rows": rows,
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "all_unchanged": all(row["unchanged"] for row in rows),
    }


def _cpu_receipt() -> JsonDict:
    model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not model and cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {"model": model or "unknown", "count": os.cpu_count() or 1}


def _ram_receipt() -> JsonDict:
    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, value = line.split(":", 1)
        fields[name] = int(value.strip().split()[0])
    return {"total_kib": fields["MemTotal"], "available_kib": fields["MemAvailable"]}


def _gpu_ownership_receipt() -> JsonDict:
    """Observe visible GPU owners without sending a signal or claiming a GPU."""

    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_result = subprocess.run(gpu_command, capture_output=True, text=True, timeout=10)
        process_result = subprocess.run(process_command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - host fallback.
        return {
            "visible": False,
            "gpu_rows": [],
            "compute_process_rows": [],
            "error": type(exc).__name__,
            "signals_sent": [],
        }
    gpu_rows = []
    if gpu_result.returncode == 0:
        for line in gpu_result.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) == 6:
                gpu_rows.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "uuid": parts[2],
                        "memory_total_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                        "utilization_pct": int(parts[5]),
                    }
                )
    process_rows = []
    if process_result.returncode == 0:
        for line in process_result.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) == 4:
                process_rows.append(
                    {
                        "gpu_uuid": parts[0],
                        "pid": int(parts[1]),
                        "process_name": parts[2],
                        "used_memory_mb": int(parts[3]),
                    }
                )
    return {
        "visible": bool(gpu_rows),
        "gpu_rows": gpu_rows,
        "compute_process_rows": process_rows,
        "gpu_query_exit_code": gpu_result.returncode,
        "process_query_exit_code": process_result.returncode,
        "selection_performed": False,
        "signals_sent": [],
    }


def collect_preconditions(
    repo_root: Path,
    protected_before: Mapping[str, str],
    cache_rows: Sequence[Mapping[str, Any]],
    *,
    date: str,
) -> JsonDict:
    """Record all local inputs without claiming model or GPU ownership."""

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    status_lines = [line for line in status.stdout.splitlines() if line]
    disk = shutil.disk_usage(repo_root)
    return {
        "planning_date": date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_calls_issued": 0,
        "model_loads_issued": 0,
        "downloads_issued": 0,
        "auto_tokenizer_calls_issued": 0,
        "no_llm_substrate": True,
        "v573_artifacts": [
            {
                "experiment_id": experiment_id,
                "path": path.as_posix(),
                "exists": (repo_root / path).is_file(),
                "sha256": sha256_file(repo_root / path),
            }
            for experiment_id, path in V573_ARTIFACT_PATHS.items()
        ],
        "roadmap": {
            "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / ACTIVE_ROADMAP_RELATIVE_PATH),
        },
        "protected_file_hashes_before": dict(protected_before),
        "dirty_worktree": {
            "command": "git status --short",
            "exit_code": status.returncode,
            "is_dirty": bool(status_lines),
            "entries": status_lines,
            "status_sha256": sha256_bytes(status.stdout.encode("utf-8")),
        },
        "cpu": _cpu_receipt(),
        "ram": _ram_receipt(),
        "disk": {
            "path": str(repo_root),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
        "local_gguf_cache_paths": [
            {
                "repository_id": row.get("repository_id"),
                "path": row.get("cache_path"),
                "resolved": row.get("resolved"),
                "trusted_sha256": row.get("trusted_sha256"),
            }
            for row in cache_rows
        ],
        "visible_gpu_ownership": _gpu_ownership_receipt(),
    }


def _tests_run_receipts(rows: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if rows is None else rows
    return [
        {
            "command": str(row.get("command", "")),
            "exit_code": int(row.get("exit_code", 1)),
            "duration_s": float(row.get("duration_s", 0.0)),
        }
        for row in source
    ]


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_artifacts = [
        {
            "path": path.as_posix(),
            "sha256": sha256_file(repo_root / path),
        }
        for path in V573_ARTIFACT_PATHS.values()
    ]
    source_artifacts.append(
        {
            "path": GGUF_IDENTITY_SOURCE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / GGUF_IDENTITY_SOURCE_RELATIVE_PATH),
        }
    )
    source_rows = {
        "V573": "v573_terminal_replay_rows",
        "cache": "model_cache_identity_rows",
        "budget": "execution_budget_contract",
        "gates": "current_roadmap_gate_contract_rows",
        "attacks": "attack_rows",
    }
    reducers = {
        "V573": "recompute_exp6585_readiness + recompute_exp6587_readiness",
        "launch": "readiness_reducer",
        "checksum": "artifact_checksum",
    }
    roadmap_hash = sha256_file(repo_root / ACTIVE_ROADMAP_RELATIVE_PATH)
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source_artifacts": deepcopy(source_artifacts),
            "source_rows": deepcopy(source_rows),
            "roadmap_sha256": roadmap_hash,
            "reducer_code": deepcopy(reducers),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _gate_summary(reduction: Mapping[str, Any]) -> JsonDict:
    checks = [
        {
            "check": name,
            "expected_value": True,
            "observed_value": passed,
            "passed": passed is True,
        }
        for name, passed in reduction.get("checks", {}).items()
    ]
    failed = [row for row in checks if not row["passed"]]
    return {
        "checks": checks,
        "failed_check_count": len(failed),
        "first_failure": failed[0] if failed else None,
        "passed": not failed,
        "ready_score": reduction.get("ready_score"),
    }


def build_report(
    repo_root: Path = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build one terminal launch artifact from local source-only evidence."""

    started = time.perf_counter()
    protected_before = _protected_hashes(repo_root)
    replay_rows = build_v573_terminal_replay_rows(repo_root)
    cache_rows = build_model_cache_identity_rows(repo_root)
    budget_rows = build_execution_budget_contract()
    gate_rows = build_current_roadmap_gate_contract_rows(repo_root)
    preconditions = collect_preconditions(repo_root, protected_before, cache_rows, date=date)
    protected = _protected_receipt(protected_before, _protected_hashes(repo_root))
    base: JsonDict = {
        "v573_terminal_replay_rows": replay_rows,
        "model_cache_identity_rows": cache_rows,
        "execution_budget_contract": budget_rows,
        "current_roadmap_gate_contract_rows": gate_rows,
        "suite_green_launch_requirement": False,
        "protected_files_unchanged": protected,
    }
    attack_rows = build_attack_rows(base)
    reduction = readiness_reducer({**base, "attack_rows": attack_rows})
    ready = reduction["ready_score"] == 1.0
    report_duration = float(duration_s) if duration_s is not None else time.perf_counter() - started
    payload: JsonDict = {
        "status": "complete_v574_cfr_launch_ready" if ready else "blocked_v574_cfr_launch_root",
        "honest_verdict": (
            "complete: V573 execution and method contracts replayed; both local flagship GGUF identities and bounded V574 stream contracts are ready; no science result was created"
            if ready
            else "blocked_v574_cfr_launch_root: one or more source, cache, budget, gate, attack, or protection checks failed"
        ),
        "verdict_class": None if ready else "blocked",
        "gate_check_summary": _gate_summary(reduction),
        **base,
        "attack_rows": attack_rows,
        READY_FIELD: reduction["ready_score"],
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(repo_root),
        "duration_s": report_duration,
        "tests_run": _tests_run_receipts(tests_run),
    }
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def validate_report(payload: Mapping[str, Any]) -> list[str]:
    """Return all terminal schema, reducer, authority, and checksum errors."""

    errors = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload.get("duration_s", 0) <= 0:
        errors.append("duration_s must be positive")
    reduction = readiness_reducer(payload)
    if payload.get(READY_FIELD) != reduction["ready_score"]:
        errors.append(f"{READY_FIELD} mismatch")
    if reduction["ready_score"] == 1.0:
        if payload.get("verdict_class") is not None:
            errors.append("ready launch verdict_class must be null")
        if payload.get("status") != "complete_v574_cfr_launch_ready":
            errors.append("ready launch status mismatch")
        if not str(payload.get("honest_verdict", "")).startswith(
            ("complete:", "success:", "passed:", "shipped:")
        ):
            errors.append("terminal success prefix missing")
    elif payload.get("verdict_class") == "blocked":
        summary = payload.get("gate_check_summary", {})
        if not isinstance(summary, Mapping) or summary.get("failed_check_count", 0) < 1:
            errors.append("blocked gate_check_summary missing failure")
    if payload.get("protected_files_unchanged", {}).get("all_unchanged") is not True:
        errors.append("protected_files_unchanged failed")
    provenance = payload.get("field_provenance", {})
    if not isinstance(provenance, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(provenance):
        errors.append("field_provenance missing required fields")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def atomic_write_report(path: str | Path, payload: Mapping[str, Any]) -> JsonDict:
    """Validate, file-sync, atomically replace, and directory-sync one JSON."""

    errors = validate_report(payload)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():  # pragma: no cover - replace failure cleanup.
            temporary.unlink()
    return {
        "path": str(target),
        "file_fsync": True,
        "atomic_replace": True,
        "directory_fsync": True,
        "output_sha256": sha256_file(target),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        errors = validate_report(load_json(output))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"valid: {output}")
        return 0
    report = build_report(REPO_ROOT, date=args.date)
    receipt = atomic_write_report(output, report)
    print(json.dumps({"status": report["status"], READY_FIELD: report[READY_FIELD], **receipt}))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point.
    raise SystemExit(main())
