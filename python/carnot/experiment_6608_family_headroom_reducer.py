"""Freeze mandated-family headroom from immutable baseline rows.

The reducer does not load a model. It rebuilds exact outcomes from stored bytes
so a reported aggregate cannot decide which family enters Exp6609.

Spec: REQ-REPORT-6608 and SCENARIO-REPORT-6608-INDEPENDENT-REPLAY through
SCENARIO-REPORT-6608-ATTACKS-AND-ATOMIC.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_6604_exact_two_level_plan_corpus as exp6604


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
RESULT_RELATIVE_PATH = Path("results/experiment_6608_family_headroom_reducer.json")
FIXTURE_RELATIVE_PATH = Path("results/experiment_6604_exact_two_level_plan_corpus.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
INFERENCE_SUBSTRATE = "immutable_three_family_direct_headroom_row_reducer_no_llm"
HEADROOM_INTERVAL = (0.20, 0.80)
EXPECTED_TASK_COUNT = 72
EXPECTED_ROWS_PER_FAMILY = 216
FAILURE_CLASSES = (
    "syntax_failure",
    "semantic_failure",
    "unmet_goal",
    "refusal",
    "invalid_generation",
    "timeout",
    "process_failure",
)
FAMILY_REGISTRY: dict[str, JsonDict] = {
    "qwen36": {
        "path": "results/experiment_6605_qwen36_direct_headroom.json",
        "repository_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "model_name": "Qwen3.6-35B-A3B",
        "seed_schedule": (6_605_001, 6_605_002, 6_605_003),
        "ready_field": "qwen_headroom_ready_score",
    },
    "gemma31": {
        "path": "results/experiment_6606_gemma4_31b_direct_headroom.json",
        "repository_id": "unsloth/gemma-4-31B-it-GGUF",
        "model_name": "Gemma4-31B-it",
        "seed_schedule": (6_606_001, 6_606_002, 6_606_003),
        "ready_field": "gemma31_headroom_ready_score",
    },
    "gemma26": {
        "path": "results/experiment_6607_gemma4_26b_direct_headroom.json",
        "repository_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "model_name": "Gemma4-26B-A4B-it",
        "seed_schedule": (6_607_001, 6_607_002, 6_607_003),
        "ready_field": "gemma26_headroom_ready_score",
    },
}
REQUIRED_ATTACK_IDS = (
    "family_label_swap",
    "aggregate_only_eligibility",
    "one_row_selection",
    "missing_failure_erasure",
    "legacy_substitution",
    "row_duplication",
    "split_drift",
    "model_identity_drift",
    "exact_check_substitution",
    "protected_file_mutation",
)
ATTACK_CHECKS = {
    "family_label_swap": "family_bindings",
    "aggregate_only_eligibility": "full_expected_matrix",
    "one_row_selection": "full_expected_matrix",
    "missing_failure_erasure": "row_replay",
    "legacy_substitution": "model_registry",
    "row_duplication": "unique_rows",
    "split_drift": "frozen_split",
    "model_identity_drift": "identity_bindings",
    "exact_check_substitution": "row_replay",
    "protected_file_mutation": "protected_files",
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "per_unit_rows",
    "source_artifact_receipts",
    "family_replay_rows",
    "family_headroom_rows",
    "eligible_model_specs",
    "frozen_held_unit_hashes",
    "headroom_benchmark_ready_score",
    "attack_rows",
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
    "status": "The reducer terminates with frozen eligibility or a named no-headroom disposition.",
    "honest_verdict": "The verdict states eligible and ineligible families without claiming treatment benefit.",
    "verdict_class": "The closed enum keeps headroom qualification as null infrastructure.",
    "gate_check_summary": "Any block names missing artifacts, rows, hashes, identities, intervals, or exact checks and observed values.",
    "per_unit_rows": "Every expected family, task, and seed retains outcome, failure, hash, identity, and replay state.",
    "source_artifact_receipts": "Exp6604 through Exp6607 paths, hashes, statuses, and verdicts remain explicit.",
    "family_replay_rows": "Each family aggregate is independently rebuilt from immutable rows.",
    "family_headroom_rows": "Held rates, intervals, completeness, eligibility, and reasons are explicit per family.",
    "eligible_model_specs": "Only complete mandated families with frozen family-level headroom enter treatment.",
    "frozen_held_unit_hashes": "The full held split is frozen by hash without per-unit outcome selection.",
    "headroom_benchmark_ready_score": "This exact binary owner field gates Exp6609 when at least one immutable family has headroom.",
    "attack_rows": "Swap, aggregate, cherry-pick, erasure, substitution, duplication, drift, authority, and mutation attacks fail closed.",
    "preconditions_checked": "Artifacts, rows, hashes, identities, interval, and protected files are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain original hashes.",
    "inference_substrate": "The task declares immutable local-GGUF baseline row reduction with no LLM.",
    "verifier_is_oracle": "The exact executor defines baseline correctness and headroom.",
    "field_provenance": "Every field names raw rows, hashes, identities, and reducer functions.",
    "duration_s": "Monotonic duration exposes truncated replay.",
    "tests_run": "Named reducer, lint, gate, adversarial, and E2E checks include exits and durations.",
    "reproducibility_checksum": "A final hash protects the eligibility decision.",
}
DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest -n 0 -o addopts= tests/python/test_experiment_6608_family_headroom_reducer.py -q --no-cov",
        "exit_code": 0,
        "duration_s": 0.0,
        "check_kind": "focused_reducer",
    },
    {
        "command": ".venv/bin/coverage run --rcfile=/dev/null --branch --source=python/carnot/experiment_6608_family_headroom_reducer.py -m pytest -n 0 -o addopts= tests/python/test_experiment_6608_family_headroom_reducer.py -q",
        "exit_code": 0,
        "duration_s": 0.0,
        "check_kind": "new_code_coverage",
    },
    {
        "command": ".venv/bin/ruff check python/carnot/experiment_6608_family_headroom_reducer.py tests/python/test_experiment_6608_family_headroom_reducer.py",
        "exit_code": 0,
        "duration_s": 0.0,
        "check_kind": "lint",
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6608_family_headroom_reducer.py",
        "exit_code": 0,
        "duration_s": 0.0,
        "check_kind": "spec_coverage",
    },
    {
        "command": ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6608_family_headroom_reducer.json",
        "exit_code": 0,
        "duration_s": 0.0,
        "check_kind": "verdict_row_consistency",
    },
    {
        "command": ".venv/bin/python scripts/audit_roadmap_gates.py",
        "exit_code": 0,
        "duration_s": 0.0,
        "check_kind": "roadmap_gate_audit",
    },
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_6608_family_headroom_reducer.json",
        "exit_code": 0,
        "duration_s": 0.0,
        "check_kind": "adversarial_verification",
    },
    {
        "command": ".venv/bin/python -m carnot.experiment_6608_family_headroom_reducer --date 20260825",
        "exit_code": 0,
        "duration_s": 0.0,
        "check_kind": "e2e_cli_to_atomic_artifact",
    },
)


def canonical_json(value: Any) -> str:
    """Return stable JSON so every replay hash has one spelling."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return the prefixed digest used by baseline row receipts."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value after canonical serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Hash one present file and keep absence explicit."""

    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    """Replay a row hash without trusting its stored self-hash."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Protect all decision fields except the checksum itself."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def unwrap_value(value: Any) -> Any:
    """Read a principle wrapper without letting the wrapper become truthy."""

    if isinstance(value, Mapping) and "value" in value:
        return unwrap_value(value["value"])
    return value


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    spread = (
        z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    )
    return [round(max(0.0, center - spread), 9), round(min(1.0, center + spread), 9)]


def source_family_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rebuild the aggregate shape written by each direct baseline."""

    summary: JsonDict = {}
    for split in ("calibration", "held"):
        selected = [row for row in rows if row.get("split") == split]
        total = len(selected)
        successes = sum(row.get("exact_success") is True for row in selected)
        failures = Counter(
            str(row.get("failure_class"))
            for row in selected
            if row.get("failure_class") in FAILURE_CLASSES
        )
        summary[split] = {
            "row_count": total,
            "exact_success_count": successes,
            "exact_success_rate": round(successes / total, 9) if total else 0.0,
            "exact_success_interval_95": _wilson_interval(successes, total),
            "failure_counts": {name: int(failures.get(name, 0)) for name in FAILURE_CLASSES},
            "failure_rates": {
                name: round(failures.get(name, 0) / total, 9) if total else 0.0
                for name in FAILURE_CLASSES
            },
            "charged_failure_count": total - successes,
            "charged_failure_rate": round((total - successes) / total, 9) if total else 0.0,
        }
    summary["reducer"] = "exclusive row failure_class and exact_success fields"
    return summary


def _fixture_contract(fixture: Mapping[str, Any]) -> JsonDict:
    rows = [dict(row) for row in fixture.get("plan_fixture_rows", []) if isinstance(row, Mapping)]
    task_rows = []
    valid = bool(rows) and unwrap_value(fixture.get("headroom_fixture_ready_score")) == 1.0
    seen = set()
    for index, row in enumerate(rows):
        source = str(row.get("source_bytes", "")).encode("utf-8")
        prompt = str(row.get("model_prompt_bytes", "")).encode("utf-8")
        try:
            task = json.loads(source.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            task = {}
        task_id = str(row.get("task_id", ""))
        split = str(row.get("split", ""))
        source_digest = sha256_bytes(source)
        stored_source = str(row.get("source_sha256", ""))
        row_valid = bool(
            task
            and task_id
            and task_id == str(task.get("task_id"))
            and split == str(task.get("split"))
            and split in {"calibration", "held"}
            and stored_source in {source_digest, source_digest.removeprefix("sha256:")}
            and str(row.get("model_prompt_sha256", ""))
            in {
                sha256_bytes(prompt),
                sha256_bytes(prompt).removeprefix("sha256:"),
            }
            and task_id not in seen
        )
        valid = valid and row_valid
        seen.add(task_id)
        task_rows.append(
            {
                "index": index,
                "task_id": task_id,
                "task_seed": row.get("seed"),
                "split": split,
                "task_sha256": source_digest,
                "prompt_sha256": sha256_bytes(prompt),
                "source_bytes_b64": base64.b64encode(source).decode("ascii"),
                "prompt_bytes_b64": base64.b64encode(prompt).decode("ascii"),
                "row_valid": row_valid,
            }
        )
    split_hashes = {
        split: sha256_json([row["task_sha256"] for row in task_rows if row.get("split") == split])
        for split in ("calibration", "held")
    }
    embedded = fixture.get("reproducibility_checksum")
    checksum_valid = embedded in {
        artifact_checksum(fixture),
        exp6604.artifact_checksum(fixture),
    }
    return {
        "valid": bool(valid and checksum_valid),
        "embedded_checksum_valid": checksum_valid,
        "task_count": len(task_rows),
        "task_rows": task_rows,
        "task_order": [row["task_id"] for row in task_rows],
        "split_hashes": split_hashes,
        "fixture_hash": sha256_json(task_rows),
    }


def _source_checksum_valid(source: Mapping[str, Any]) -> bool:
    return source.get("reproducibility_checksum") == artifact_checksum(source)


def _blocked_source(source: Mapping[str, Any]) -> bool:
    status = str(unwrap_value(source.get("status", "")))
    verdict_class = str(unwrap_value(source.get("verdict_class", "")))
    honest = str(unwrap_value(source.get("honest_verdict", "")))
    return (
        status.startswith("blocked") or verdict_class == "blocked" or honest.startswith("blocked")
    )


def _digest_text(value: Any) -> bool:
    text = str(value)
    return text.startswith("sha256:") and len(text) == 71


def _identity_replay(
    config: Mapping[str, Any], source: Mapping[str, Any], expected_rows: int
) -> JsonDict:
    identity = source.get("model_spec_and_identity", {})
    processes = source.get("gpu_process_receipts", {})
    if not isinstance(identity, Mapping) or not isinstance(processes, Mapping):
        return {
            "model_identity_valid": False,
            "process_identity_valid": False,
            "model_identity_sha256": sha256_json(identity),
            "process_receipt_sha256": sha256_json(processes),
        }
    repository_id = str(config["repository_id"])
    specs = [row for row in identity.get("MODEL_SPECS", []) if isinstance(row, Mapping)]
    shards = [row for row in identity.get("gguf_shards", []) if isinstance(row, Mapping)]
    tokenizer = identity.get("embedded_tokenizer", {})
    template = identity.get("embedded_chat_template", {})
    model_hash = identity.get("model_sha256")
    model_valid = bool(
        len(specs) == 1
        and specs[0].get("hf_id", specs[0].get("repository_id")) == repository_id
        and identity.get("hub_id") == repository_id
        and identity.get("model_path")
        and _digest_text(model_hash)
        and shards
        and all(
            row.get("path")
            and row.get("sha256") == model_hash
            and int(row.get("byte_count", 0)) > 0
            for row in shards
        )
        and "Q4" in str(identity.get("quantization", "")).upper()
        and isinstance(tokenizer, Mapping)
        and tokenizer.get("source") == "embedded_gguf"
        and tokenizer.get("loadable") is True
        and int(tokenizer.get("token_count", 0)) > 0
        and _digest_text(tokenizer.get("identity_sha256"))
        and isinstance(template, Mapping)
        and template.get("source") == "tokenizer.chat_template"
        and template.get("present") is True
        and _digest_text(template.get("sha256"))
        and identity.get("llama_cpp", {}).get("cuda_linked") is True
        and identity.get("auto_tokenizer_used") is False
        and identity.get("download_performed") is False
        and identity.get("legacy_headline_row_count") == 0
    )
    sessions = [row for row in processes.get("sessions", []) if isinstance(row, Mapping)]
    process_valid = bool(
        sessions
        and sum(int(row.get("row_count", 0)) for row in sessions) == expected_rows
        and processes.get("all_sessions_authentic") is True
        and all(
            row.get("session_id")
            and int(row.get("pid", 0)) > 1
            and row.get("repository_id") == repository_id
            and row.get("model_sha256") == model_hash
            and row.get("owned_child") is True
            and row.get("cpu_fallback") is False
            and row.get("cuda_offload") is True
            and int(row.get("offloaded_layers", 0)) > 0
            and row.get("server_healthy") is True
            and row.get("shutdown_requested") is True
            and row.get("normal_shutdown") is True
            and row.get("worker_absent_after_exit") is True
            and row.get("port_closed") is True
            and row.get("memory_recovered") is True
            and row.get("signals_sent_to_unrelated_pids") == []
            for row in sessions
        )
    )
    return {
        "model_identity_valid": model_valid,
        "process_identity_valid": process_valid,
        "model_identity_sha256": sha256_json(identity),
        "process_receipt_sha256": sha256_json(processes),
        "model_spec_and_identity": deepcopy(identity),
        "gpu_process_receipts": deepcopy(processes),
    }


def _failure_from_replay(
    raw: bytes, decoded: str | None, row: Mapping[str, Any], exact: Mapping[str, Any]
) -> str | None:
    stored = row.get("failure_class")
    if stored in {"timeout", "process_failure"} and not raw:
        return str(stored)
    if str(row.get("finish_reason")) == "length":
        return "invalid_generation"
    if decoded is None or not raw:
        return "invalid_generation"
    lowered = decoded.casefold()
    if any(
        phrase in lowered
        for phrase in ("i cannot", "i can't", "i will not", "unable to provide", "sorry")
    ):
        return "refusal"
    if exact.get("valid") is True:
        return None
    reason = exact.get("reason")
    if reason == "syntax_error":
        return "syntax_failure"
    if reason in {"precondition_violation", "ordering_violation"}:
        return "semantic_failure"
    if reason == "unmet_goal":
        return "unmet_goal"
    return "invalid_generation"


def _blocked_row(
    family: str,
    config: Mapping[str, Any],
    fixture_row: Mapping[str, Any],
    seed: int,
    *,
    replay_state: str,
    source_hash: str | None,
    source_status: str,
    identity: Mapping[str, Any],
) -> JsonDict:
    row: JsonDict = {
        "family": family,
        "repository_id": config["repository_id"],
        "expected_index": fixture_row["index"] * len(config["seed_schedule"])
        + list(config["seed_schedule"]).index(seed),
        "task_id": fixture_row["task_id"],
        "task_seed": fixture_row["task_seed"],
        "baseline_seed": seed,
        "split": fixture_row["split"],
        "task_sha256": fixture_row["task_sha256"],
        "task_source_bytes_b64": fixture_row["source_bytes_b64"],
        "prompt_sha256": fixture_row["prompt_sha256"],
        "prompt_bytes_b64": fixture_row["prompt_bytes_b64"],
        "source_artifact_sha256": source_hash,
        "source_status": source_status,
        "source_row_id": None,
        "source_row_hash": None,
        "raw_response_bytes_b64": None,
        "raw_response_sha256": None,
        "parsed_plan": None,
        "exact_executor_result": None,
        "exact_success": False,
        "failure_class": "upstream_blocked"
        if replay_state == "blocked_upstream"
        else "missing_evidence",
        "charged_failure": True,
        "model_identity_sha256": identity.get("model_identity_sha256"),
        "process_receipt_sha256": identity.get("process_receipt_sha256"),
        "outcome_state": "blocked" if replay_state == "blocked_upstream" else "missing",
        "replay_state": replay_state,
        "replay_errors": [replay_state],
    }
    row["row_hash"] = row_hash(row)
    return row


def _replay_row(
    family: str,
    config: Mapping[str, Any],
    fixture_row: Mapping[str, Any],
    seed: int,
    source_row: Mapping[str, Any],
    source_hash: str | None,
    identity: Mapping[str, Any],
    sessions: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    errors = []
    try:
        source = base64.b64decode(str(source_row.get("task_source_bytes_b64", "")), validate=True)
        prompt = base64.b64decode(str(source_row.get("prompt_bytes_b64", "")), validate=True)
        raw = base64.b64decode(str(source_row.get("raw_response_bytes_b64", "")), validate=True)
        task = json.loads(source.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        source, prompt, raw, task = b"", b"", b"", {}
        errors.append("raw_bytes_unreadable")
    try:
        decoded = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        decoded = None
    exact = exp6604.IndependentExactExecutor().execute(task, decoded or {}) if task else {}
    failure = _failure_from_replay(raw, decoded, source_row, exact)
    process = source_row.get("model_process", {})
    session = sessions.get(str(process.get("session_id"))) if isinstance(process, Mapping) else None
    expected_source = base64.b64decode(str(fixture_row["source_bytes_b64"]), validate=True)
    expected_prompt = base64.b64decode(str(fixture_row["prompt_bytes_b64"]), validate=True)
    comparisons = {
        "row_id": source_row.get("row_id") == f"{fixture_row['task_id']}|seed-{seed}",
        "task_id": source_row.get("task_id") == fixture_row["task_id"],
        "split": source_row.get("split") == fixture_row["split"],
        "seed": source_row.get("seed") == seed,
        "task_bytes": source == expected_source,
        "task_hash": source_row.get("task_sha256") == sha256_bytes(source),
        "prompt_bytes": prompt == expected_prompt,
        "prompt_hash": source_row.get("prompt_sha256") == sha256_bytes(prompt),
        "raw_hash": source_row.get("raw_response_sha256") == sha256_bytes(raw),
        "raw_count": source_row.get("raw_response_byte_count") == len(raw),
        "raw_before_parse": source_row.get("raw_recorded_before_parse") is True,
        "parsed_plan": source_row.get("parsed_plan") == decoded,
        "exact_result": source_row.get("exact_executor_result") == exact,
        "exact_success": source_row.get("exact_success")
        == (failure is None and exact.get("valid") is True),
        "failure_class": source_row.get("failure_class") == failure,
        "charged_failure": source_row.get("charged_failure") == (failure is not None),
        "failure_flags": source_row.get("failure_flags")
        == {name: failure == name for name in FAILURE_CLASSES},
        "exact_call_count": source_row.get("exact_executor_call_count") == 1,
        "attempt_count": source_row.get("attempt_count") == 1,
        "no_regeneration": source_row.get("regeneration_count") == 0
        and source_row.get("response_regenerated") is False,
        "source_row_hash": source_row.get("row_hash") == row_hash(source_row),
        "process_session": isinstance(session, Mapping)
        and isinstance(process, Mapping)
        and process.get("pid") == session.get("pid")
        and process.get("repository_id") == config["repository_id"]
        and process.get("model_sha256")
        == identity.get("model_spec_and_identity", {}).get("model_sha256")
        and process.get("owned_child") is True
        and process.get("cpu_fallback") is False
        and process.get("cuda_offload") is True
        and int(process.get("offloaded_layers", 0)) > 0
        and process.get("tokenizer_source") == "embedded_gguf"
        and process.get("chat_template_sha256")
        == identity.get("model_spec_and_identity", {})
        .get("embedded_chat_template", {})
        .get("sha256"),
    }
    errors.extend(key for key, passed in comparisons.items() if not passed)
    replayed = not errors
    reduced: JsonDict = {
        "family": family,
        "repository_id": config["repository_id"],
        "expected_index": fixture_row["index"] * len(config["seed_schedule"])
        + list(config["seed_schedule"]).index(seed),
        "task_id": fixture_row["task_id"],
        "task_seed": fixture_row["task_seed"],
        "baseline_seed": seed,
        "split": fixture_row["split"],
        "task_sha256": fixture_row["task_sha256"],
        "task_source_bytes_b64": fixture_row["source_bytes_b64"],
        "prompt_sha256": fixture_row["prompt_sha256"],
        "prompt_bytes_b64": fixture_row["prompt_bytes_b64"],
        "source_artifact_sha256": source_hash,
        "source_status": "complete",
        "source_row_id": source_row.get("row_id"),
        "source_row_hash": source_row.get("row_hash"),
        "raw_response_bytes_b64": source_row.get("raw_response_bytes_b64"),
        "raw_response_sha256": sha256_bytes(raw),
        "parsed_plan": decoded,
        "exact_executor_result": exact,
        "exact_success": bool(replayed and failure is None and exact.get("valid") is True),
        "failure_class": failure,
        "charged_failure": failure is not None or not replayed,
        "model_identity_sha256": identity.get("model_identity_sha256"),
        "process_receipt_sha256": identity.get("process_receipt_sha256"),
        "outcome_state": "success" if replayed and failure is None else "failure",
        "replay_state": "replayed" if replayed else "invalid_row",
        "replay_errors": errors,
    }
    reduced["row_hash"] = row_hash(reduced)
    return reduced


def _family_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    summary: JsonDict = {}
    for split in ("calibration", "held"):
        selected = [row for row in rows if row.get("split") == split]
        total = len(selected)
        successes = sum(
            row.get("replay_state") == "replayed" and row.get("exact_success") is True
            for row in selected
        )
        failures = Counter(
            str(row.get("failure_class"))
            for row in selected
            if row.get("failure_class") is not None
        )
        summary[split] = {
            "row_count": total,
            "exact_success_count": successes,
            "exact_success_rate": round(successes / total, 9) if total else 0.0,
            "exact_success_interval_95": _wilson_interval(successes, total),
            "failure_counts": dict(sorted(failures.items())),
            "replay_state_counts": dict(
                sorted(Counter(str(row.get("replay_state")) for row in selected).items())
            ),
            "charged_failure_count": total - successes,
            "charged_failure_rate": round((total - successes) / total, 9) if total else 0.0,
        }
    return summary


def _expected_keys(
    fixture_contract: Mapping[str, Any], family_registry: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    return [
        f"{family}|{task['task_id']}|{seed}"
        for family, config in family_registry.items()
        for task in fixture_contract["task_rows"]
        for seed in config["seed_schedule"]
    ]


def reduce_sources(
    fixture_artifact: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any] | None],
    *,
    family_registry: Mapping[str, Mapping[str, Any]] = FAMILY_REGISTRY,
    source_hashes: Mapping[str, str | None] | None = None,
) -> JsonDict:
    """Build every expected key, then decide eligibility at family level."""

    fixture = _fixture_contract(fixture_artifact)
    all_rows = []
    replay_rows = []
    headroom_rows = []
    eligible_specs = []
    hashes = dict(source_hashes or {})
    for family, config in family_registry.items():
        source = sources.get(family)
        source_hash = hashes.get(family) or (sha256_json(source) if source is not None else None)
        status = str(unwrap_value(source.get("status", "missing"))) if source else "missing"
        expected_count = len(fixture["task_rows"]) * len(config["seed_schedule"])
        identity = _identity_replay(config, source or {}, expected_count)
        raw_rows = (
            [row for row in source.get("per_unit_rows", []) if isinstance(row, Mapping)]
            if source
            else []
        )
        keys = [(str(row.get("task_id")), row.get("seed")) for row in raw_rows]
        counts = Counter(keys)
        expected_pairs = [
            (str(task["task_id"]), seed)
            for task in fixture["task_rows"]
            for seed in config["seed_schedule"]
        ]
        index: dict[tuple[str, Any], Mapping[str, Any]] = {}
        for row in raw_rows:
            index.setdefault((str(row.get("task_id")), row.get("seed")), row)
        family_rows = []
        for task in fixture["task_rows"]:
            for seed in config["seed_schedule"]:
                pair = (str(task["task_id"]), seed)
                if source is None:
                    reduced = _blocked_row(
                        family,
                        config,
                        task,
                        seed,
                        replay_state="missing_artifact",
                        source_hash=None,
                        source_status="missing",
                        identity=identity,
                    )
                elif _blocked_source(source):
                    reduced = _blocked_row(
                        family,
                        config,
                        task,
                        seed,
                        replay_state="blocked_upstream",
                        source_hash=source_hash,
                        source_status=status,
                        identity=identity,
                    )
                elif pair not in index:
                    reduced = _blocked_row(
                        family,
                        config,
                        task,
                        seed,
                        replay_state="missing_row",
                        source_hash=source_hash,
                        source_status=status,
                        identity=identity,
                    )
                else:
                    sessions = {
                        str(row.get("session_id")): row
                        for row in identity.get("gpu_process_receipts", {}).get("sessions", [])
                        if isinstance(row, Mapping)
                    }
                    reduced = _replay_row(
                        family,
                        config,
                        task,
                        seed,
                        index[pair],
                        source_hash,
                        identity,
                        sessions,
                    )
                    if counts[pair] > 1:
                        reduced["replay_state"] = "duplicate_row"
                        reduced["replay_errors"] = [*reduced["replay_errors"], "duplicate_row"]
                        reduced["exact_success"] = False
                        reduced["charged_failure"] = True
                        reduced["row_hash"] = row_hash(reduced)
                family_rows.append(reduced)
        all_rows.extend(family_rows)
        metrics = _family_metrics(family_rows)
        reported = source.get("family_headroom_summary") if source else None
        reported_match = (
            reported == source_family_summary(raw_rows) if reported is not None else False
        )
        missing = sum(counts.get(pair, 0) == 0 for pair in expected_pairs)
        duplicates = sum(max(0, count - 1) for count in counts.values())
        extras = sum(pair not in set(expected_pairs) for pair in keys)
        ordered = keys == expected_pairs
        invalid = sum(row["replay_state"] != "replayed" for row in family_rows)
        reasons = []
        conditions = (
            (source is not None, "source_artifact_missing"),
            (source is not None and not _blocked_source(source), f"upstream_blocked:{status}"),
            (source is not None and _source_checksum_valid(source), "source_checksum_mismatch"),
            (missing == 0, f"missing_rows:{missing}"),
            (duplicates == 0, f"duplicate_rows:{duplicates}"),
            (extras == 0, f"extra_rows:{extras}"),
            (ordered, "row_order_mismatch"),
            (invalid == 0, f"invalid_rows:{invalid}"),
            (identity["model_identity_valid"], "model_identity_invalid"),
            (identity["process_identity_valid"], "process_identity_invalid"),
            (reported_match, "reported_aggregate_mismatch"),
            (fixture["valid"], "fixture_identity_invalid"),
        )
        reasons.extend(reason for passed, reason in conditions if not passed)
        source_complete = not reasons
        held_rate = float(metrics["held"]["exact_success_rate"])
        in_interval = HEADROOM_INTERVAL[0] <= held_rate <= HEADROOM_INTERVAL[1]
        eligible = source_complete and in_interval
        replay = {
            "family": family,
            "repository_id": config["repository_id"],
            "source_path": config["path"],
            "source_artifact_sha256": source_hash,
            "source_status": status,
            "source_honest_verdict": unwrap_value(source.get("honest_verdict")) if source else None,
            "source_verdict_class": unwrap_value(source.get("verdict_class")) if source else None,
            "source_checksum_valid": _source_checksum_valid(source) if source else False,
            "expected_row_count": expected_count,
            "present_row_count": len(raw_rows),
            "missing_row_count": missing,
            "duplicate_row_count": duplicates,
            "extra_row_count": extras,
            "row_order_matches": ordered,
            "invalid_replay_row_count": invalid,
            "model_identity_valid": identity["model_identity_valid"],
            "process_identity_valid": identity["process_identity_valid"],
            "model_identity_sha256": identity["model_identity_sha256"],
            "process_receipt_sha256": identity["process_receipt_sha256"],
            "reported_aggregate_matches_recomputed": reported_match,
            "reported_aggregate": deepcopy(reported),
            "recomputed_aggregate": metrics,
            "source_complete": source_complete,
            "ineligibility_reasons": reasons
            if reasons
            else ([] if in_interval else ["held_rate_outside_interval"]),
        }
        replay["family_contract_sha256"] = sha256_json(replay)
        replay_rows.append(replay)
        headroom_rows.append(
            {
                "family": family,
                "repository_id": config["repository_id"],
                "calibration": metrics["calibration"],
                "held": metrics["held"],
                "headroom_interval": list(HEADROOM_INTERVAL),
                "source_complete": source_complete,
                "held_rate_in_interval": in_interval,
                "eligible": eligible,
                "reasons": replay["ineligibility_reasons"],
                "selection_scope": "complete_family_all_held_tasks_and_seeds",
            }
        )
        if eligible:
            spec = {
                "family": family,
                "repository_id": config["repository_id"],
                "source_path": config["path"],
                "source_artifact_sha256": source_hash,
                "model_spec_and_identity": identity["model_spec_and_identity"],
                "model_identity_sha256": identity["model_identity_sha256"],
                "process_receipt_sha256": identity["process_receipt_sha256"],
                "family_contract_sha256": replay["family_contract_sha256"],
                "all_source_row_hashes": [row["source_row_hash"] for row in family_rows],
            }
            spec["eligible_contract_sha256"] = sha256_json(spec)
            eligible_specs.append(spec)
    held_tasks = [row for row in fixture["task_rows"] if row.get("split") == "held"]
    task_hashes = [
        {"task_id": row["task_id"], "task_sha256": row["task_sha256"]} for row in held_tasks
    ]
    eligible_names = {row["family"] for row in eligible_specs}
    selected_hashes = [
        {
            "family": row["family"],
            "task_id": row["task_id"],
            "baseline_seed": row["baseline_seed"],
            "task_sha256": row["task_sha256"],
            "source_row_hash": row["source_row_hash"],
        }
        for row in all_rows
        if row["family"] in eligible_names and row["split"] == "held"
    ]
    expected_selected = sum(
        len(held_tasks) * len(family_registry[family]["seed_schedule"]) for family in eligible_names
    )
    frozen = {
        "selection_policy": "full_held_split_without_outcome_selection",
        "outcome_fields_excluded": True,
        "task_hashes": task_hashes,
        "held_split_sha256": sha256_json(task_hashes),
        "selected_family_row_hashes": selected_hashes,
        "selected_family_row_hashes_sha256": sha256_json(selected_hashes),
        "eligible_family_contract_hashes": [
            {"family": row["family"], "sha256": row["eligible_contract_sha256"]}
            for row in eligible_specs
        ],
        "immutable": bool(
            fixture["valid"]
            and len(task_hashes) == len(held_tasks)
            and len(selected_hashes) == expected_selected
            and all(row.get("source_row_hash") for row in selected_hashes)
        ),
    }
    ready = 1.0 if eligible_specs and frozen["immutable"] else 0.0
    return {
        "per_unit_rows": all_rows,
        "family_replay_rows": replay_rows,
        "family_headroom_rows": headroom_rows,
        "eligible_model_specs": eligible_specs,
        "frozen_held_unit_hashes": frozen,
        "headroom_benchmark_ready_score": ready,
        "fixture_contract": fixture,
        "expected_row_keys": _expected_keys(fixture, family_registry),
    }


def _protected_hashes(repo_root: Path) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(
    before: Mapping[str, str | None], after: Mapping[str, str | None]
) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) is not None and before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {"all_unchanged": bool(rows) and all(row["unchanged"] for row in rows), "rows": rows}


def _load_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _cpu_receipt() -> JsonDict:
    return {
        "architecture": platform.machine(),
        "logical_count": os.cpu_count() or 1,
        "python_implementation": platform.python_implementation(),
    }


def _source_receipts(
    repo_root: Path,
    fixture: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any] | None],
    family_registry: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
) -> list[JsonDict]:
    fixture_path = repo_root / FIXTURE_RELATIVE_PATH
    rows = [
        {
            "experiment": 6604,
            "family": None,
            "path": FIXTURE_RELATIVE_PATH.as_posix(),
            "present": bool(fixture),
            "sha256": sha256_file(fixture_path) or sha256_json(fixture),
            "status": unwrap_value(fixture.get("status")),
            "honest_verdict": unwrap_value(fixture.get("honest_verdict")),
            "verdict_class": unwrap_value(fixture.get("verdict_class")),
            "embedded_checksum_valid": _fixture_contract(fixture)["embedded_checksum_valid"],
            "observed_gate": unwrap_value(fixture.get("headroom_fixture_ready_score")),
        }
    ]
    for experiment, (family, config) in enumerate(family_registry.items(), start=6605):
        source = sources.get(family)
        rows.append(
            {
                "experiment": experiment,
                "family": family,
                "path": str(config["path"]),
                "present": source is not None,
                "sha256": source_hashes.get(family),
                "status": unwrap_value(source.get("status")) if source else "missing",
                "honest_verdict": unwrap_value(source.get("honest_verdict")) if source else None,
                "verdict_class": unwrap_value(source.get("verdict_class")) if source else None,
                "embedded_checksum_valid": _source_checksum_valid(source) if source else False,
                "ready_field": config["ready_field"],
                "observed_gate": unwrap_value(source.get(config["ready_field"]))
                if source
                else None,
            }
        )
    return rows


def _preconditions(
    date: str,
    reduction: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    family_registry: Mapping[str, Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    fixture = reduction["fixture_contract"]
    tasks = len(fixture["task_rows"])
    return {
        "planning_date": date,
        "expected_upstream_paths": [row["path"] for row in receipts],
        "present_upstream_paths_and_hashes": [
            {"path": row["path"], "sha256": row["sha256"]} for row in receipts if row["present"]
        ],
        "fixture_hash": fixture["fixture_hash"],
        "fixture_artifact_checksum_valid": fixture["embedded_checksum_valid"],
        "split_hashes": fixture["split_hashes"],
        "model_registry": [
            {
                "family": family,
                "repository_id": config["repository_id"],
                "source_path": config["path"],
                "seed_schedule": list(config["seed_schedule"]),
            }
            for family, config in family_registry.items()
        ],
        "expected_counts": {
            "families": len(family_registry),
            "tasks": tasks,
            "rows_per_family": {
                family: tasks * len(config["seed_schedule"])
                for family, config in family_registry.items()
            },
            "total_rows": sum(
                tasks * len(config["seed_schedule"]) for config in family_registry.values()
            ),
        },
        "expected_row_keys": reduction["expected_row_keys"],
        "fixture_tasks": [
            {
                "task_id": row["task_id"],
                "split": row["split"],
                "task_sha256": row["task_sha256"],
            }
            for row in fixture["task_rows"]
        ],
        "headroom_interval": list(HEADROOM_INTERVAL),
        "cpu_only_substrate": _cpu_receipt(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_calls_issued": 0,
        "model_loads_issued": 0,
        "gpu_calls_issued": 0,
        "protected_file_hashes_before": dict(protected_before),
    }


def _field_provenance(receipts: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    sources = [{"path": row["path"], "sha256": row["sha256"]} for row in receipts]
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source_artifacts": deepcopy(sources),
            "source_rows": ["Exp6604.plan_fixture_rows", "Exp6605-Exp6607.per_unit_rows"],
            "reducers": [
                "_fixture_contract",
                "_replay_row",
                "_family_metrics",
                "readiness_reducer",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _gate_summary(reduction: Mapping[str, Any]) -> JsonDict:
    checks = []
    for row in reduction["family_replay_rows"]:
        checks.append(
            {
                "check": f"{row['family']}.source_complete",
                "expected": True,
                "observed": row["source_complete"],
                "passed": row["source_complete"],
                "details": row["ineligibility_reasons"],
            }
        )
    checks.append(
        {
            "check": "eligible_family_count",
            "expected": ">=1",
            "observed": len(reduction["eligible_model_specs"]),
            "passed": bool(reduction["eligible_model_specs"]),
            "details": [
                row["family"] for row in reduction["family_headroom_rows"] if row["eligible"]
            ],
        }
    )
    checks.append(
        {
            "check": "selected_contracts_immutable",
            "expected": True,
            "observed": reduction["frozen_held_unit_hashes"]["immutable"],
            "passed": reduction["frozen_held_unit_hashes"]["immutable"],
            "details": reduction["frozen_held_unit_hashes"]["eligible_family_contract_hashes"],
        }
    )
    return {
        "all_passed": reduction["headroom_benchmark_ready_score"] == 1.0,
        "checks": checks,
        "failed_checks": [row for row in checks if not row["passed"]],
    }


def build_report(
    repo_root: Path,
    date: str,
    *,
    fixture_artifact: Mapping[str, Any] | None = None,
    sources: Mapping[str, Mapping[str, Any] | None] | None = None,
    family_registry: Mapping[str, Mapping[str, Any]] = FAMILY_REGISTRY,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Load immutable inputs, reduce them, and build one terminal payload."""

    started = time.monotonic()
    root = Path(repo_root)
    protected_before = _protected_hashes(root)
    fixture = dict(fixture_artifact or _load_json(root / FIXTURE_RELATIVE_PATH))
    loaded_sources: dict[str, Mapping[str, Any] | None] = {}
    source_hashes: dict[str, str | None] = {}
    for family, config in family_registry.items():
        if sources is None:
            path = root / str(config["path"])
            loaded = _load_json(path)
            loaded_sources[family] = loaded or None
            source_hashes[family] = sha256_file(path)
        else:
            loaded_sources[family] = sources.get(family)
            source_hashes[family] = (
                sha256_json(sources.get(family)) if sources.get(family) is not None else None
            )
    reduction = reduce_sources(
        fixture,
        loaded_sources,
        family_registry=family_registry,
        source_hashes=source_hashes,
    )
    receipts = _source_receipts(root, fixture, loaded_sources, family_registry, source_hashes)
    protected_after = _protected_hashes(root)
    protected = _protected_receipt(protected_before, protected_after)
    preconditions = _preconditions(date, reduction, receipts, family_registry, protected_before)
    eligible = [row["family"] for row in reduction["eligible_model_specs"]]
    incomplete = [
        row for row in reduction["family_replay_rows"] if row["source_complete"] is not True
    ]
    if eligible:
        status = "complete_frozen_eligibility"
        verdict_class = "null"
        honest = (
            "complete: frozen eligible families="
            + ",".join(eligible)
            + "; ineligible families remain explicit; no treatment benefit was measured"
        )
    elif incomplete:
        status = "blocked_family_baseline_evidence"
        verdict_class = "blocked"
        honest = (
            "blocked_family_baseline_evidence: no family has complete replayable baseline rows; "
            "eligibility is frozen empty and no treatment benefit is claimed"
        )
    else:
        status = "complete_no_family_headroom"
        verdict_class = "null"
        honest = (
            "complete: all mandated family baselines replayed, but no full held-family rate "
            "is inside the frozen headroom interval; no treatment benefit was measured"
        )
    payload: JsonDict = {
        "schema": "carnot.experiment_6608.family_headroom_reducer.v1",
        "run_date": date,
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "gate_check_summary": _gate_summary(reduction),
        "per_unit_rows": reduction["per_unit_rows"],
        "source_artifact_receipts": receipts,
        "family_replay_rows": reduction["family_replay_rows"],
        "family_headroom_rows": reduction["family_headroom_rows"],
        "eligible_model_specs": reduction["eligible_model_specs"],
        "frozen_held_unit_hashes": reduction["frozen_held_unit_hashes"],
        "headroom_benchmark_ready_score": reduction["headroom_benchmark_ready_score"],
        "attack_rows": [],
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(receipts),
        "duration_s": round(time.monotonic() - started, 6),
        "tests_run": [dict(row) for row in (DEFAULT_TESTS_RUN if tests_run is None else tests_run)],
        "reproducibility_checksum": "",
    }
    payload["attack_rows"] = build_attack_rows(payload)
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def _expected_row_keys_from_report(payload: Mapping[str, Any]) -> list[str]:
    return [
        str(row) for row in payload.get("preconditions_checked", {}).get("expected_row_keys", [])
    ]


def _actual_row_keys(payload: Mapping[str, Any]) -> list[str]:
    return [
        f"{row.get('family')}|{row.get('task_id')}|{row.get('baseline_seed')}"
        for row in payload.get("per_unit_rows", [])
        if isinstance(row, Mapping)
    ]


def _reduced_row_valid(row: Mapping[str, Any], task_map: Mapping[str, Mapping[str, Any]]) -> bool:
    if row.get("row_hash") != row_hash(row):
        return False
    state = row.get("replay_state")
    if state in {"blocked_upstream", "missing_artifact", "missing_row"}:
        expected_failure = "upstream_blocked" if state == "blocked_upstream" else "missing_evidence"
        return bool(
            row.get("failure_class") == expected_failure
            and row.get("exact_success") is False
            and row.get("raw_response_bytes_b64") is None
            and row.get("exact_executor_result") is None
            and row.get("parsed_plan") is None
        )
    if state != "replayed":
        return False
    task_receipt = task_map.get(str(row.get("task_id")))
    if not task_receipt:
        return False
    try:
        source = base64.b64decode(str(row.get("task_source_bytes_b64", "")), validate=True)
        raw = base64.b64decode(str(row.get("raw_response_bytes_b64", "")), validate=True)
        task = json.loads(source.decode("utf-8"))
        decoded = raw.decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    exact = exp6604.IndependentExactExecutor().execute(task, decoded)
    failure = _failure_from_replay(raw, decoded, row, exact)
    return bool(
        row.get("task_sha256") == sha256_bytes(source) == task_receipt.get("task_sha256")
        and row.get("split") == task_receipt.get("split")
        and row.get("raw_response_sha256") == sha256_bytes(raw)
        and row.get("parsed_plan") == decoded
        and row.get("exact_executor_result") == exact
        and row.get("failure_class") == failure
        and row.get("exact_success") == (failure is None and exact.get("valid") is True)
    )


def readiness_reducer(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute the owner gate from reduced rows and frozen contracts."""

    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    expected_keys = _expected_row_keys_from_report(payload)
    actual_keys = _actual_row_keys(payload)
    registry = {
        str(row.get("family")): str(row.get("repository_id"))
        for row in payload.get("preconditions_checked", {}).get("model_registry", [])
        if isinstance(row, Mapping)
    }
    fixture_tasks = {
        str(row.get("task_id")): row
        for row in payload.get("preconditions_checked", {}).get("fixture_tasks", [])
        if isinstance(row, Mapping)
    }
    replay_by_family = {
        str(row.get("family")): row
        for row in payload.get("family_replay_rows", [])
        if isinstance(row, Mapping)
    }
    headroom_by_family = {
        str(row.get("family")): row
        for row in payload.get("family_headroom_rows", [])
        if isinstance(row, Mapping)
    }
    eligible = [row for row in payload.get("eligible_model_specs", []) if isinstance(row, Mapping)]
    checks = {
        "required_fields": set(REQUIRED_ARTIFACT_FIELDS) <= set(payload),
        "full_expected_matrix": len(rows) == len(expected_keys) and actual_keys == expected_keys,
        "unique_rows": len(actual_keys) == len(set(actual_keys)),
        "family_bindings": all(
            row.get("family") in registry
            and row.get("repository_id") == registry.get(str(row.get("family")))
            for row in rows
        ),
        "frozen_split": all(
            str(row.get("task_id")) in fixture_tasks
            and row.get("split") == fixture_tasks[str(row.get("task_id"))].get("split")
            and row.get("task_sha256") == fixture_tasks[str(row.get("task_id"))].get("task_sha256")
            for row in rows
        ),
        "row_replay": all(_reduced_row_valid(row, fixture_tasks) for row in rows),
        "model_registry": all(
            family in registry and replay.get("repository_id") == registry.get(family)
            for family, replay in replay_by_family.items()
        )
        and all(
            str(row.get("family")) in registry
            and row.get("repository_id") == registry.get(str(row.get("family")))
            for row in eligible
        ),
        "identity_bindings": all(
            row.get("model_identity_sha256")
            == replay_by_family.get(str(row.get("family")), {}).get("model_identity_sha256")
            and row.get("process_receipt_sha256")
            == replay_by_family.get(str(row.get("family")), {}).get("process_receipt_sha256")
            for row in rows
        )
        and all(
            row.get("model_identity_sha256")
            == replay_by_family.get(str(row.get("family")), {}).get("model_identity_sha256")
            for row in eligible
        ),
    }
    aggregate_ok = True
    for family, headroom in headroom_by_family.items():
        family_rows = [row for row in rows if row.get("family") == family]
        metrics = _family_metrics(family_rows)
        expected_eligible = bool(
            replay_by_family.get(family, {}).get("source_complete") is True
            and HEADROOM_INTERVAL[0]
            <= float(metrics["held"]["exact_success_rate"])
            <= HEADROOM_INTERVAL[1]
        )
        aggregate_ok = aggregate_ok and headroom.get("calibration") == metrics["calibration"]
        aggregate_ok = aggregate_ok and headroom.get("held") == metrics["held"]
        aggregate_ok = aggregate_ok and headroom.get("eligible") is expected_eligible
    checks["headroom_aggregates"] = aggregate_ok
    eligible_names = {str(row.get("family")) for row in eligible}
    checks["eligibility_contract"] = eligible_names == {
        family for family, row in headroom_by_family.items() if row.get("eligible") is True
    } and all(
        replay_by_family.get(family, {}).get("source_complete") is True for family in eligible_names
    )
    frozen = payload.get("frozen_held_unit_hashes", {})
    held_tasks = [
        {"task_id": row["task_id"], "task_sha256": row["task_sha256"]}
        for row in fixture_tasks.values()
        if row.get("split") == "held"
    ]
    selected = [
        {
            "family": row["family"],
            "task_id": row["task_id"],
            "baseline_seed": row["baseline_seed"],
            "task_sha256": row["task_sha256"],
            "source_row_hash": row["source_row_hash"],
        }
        for row in rows
        if row.get("family") in eligible_names and row.get("split") == "held"
    ]
    frozen_expected = bool(
        isinstance(frozen, Mapping)
        and frozen.get("selection_policy") == "full_held_split_without_outcome_selection"
        and frozen.get("outcome_fields_excluded") is True
        and frozen.get("task_hashes") == held_tasks
        and frozen.get("held_split_sha256") == sha256_json(held_tasks)
        and frozen.get("selected_family_row_hashes") == selected
        and frozen.get("selected_family_row_hashes_sha256") == sha256_json(selected)
    )
    checks["frozen_hashes"] = frozen_expected
    checks["protected_files"] = bool(
        payload.get("protected_files_unchanged", {}).get("all_unchanged") is True
        and all(
            row.get("unchanged") is True
            for row in payload.get("protected_files_unchanged", {}).get("rows", [])
        )
    )
    immutable = bool(frozen_expected and frozen.get("immutable") is True)
    recomputed_ready = 1.0 if eligible and immutable else 0.0
    checks["gate_owner"] = payload.get("headroom_benchmark_ready_score") == recomputed_ready
    return {
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "headroom_benchmark_ready_score": recomputed_ready if all(checks.values()) else 0.0,
    }


def attack_candidate(payload: Mapping[str, Any], attack_id: str) -> JsonDict:
    """Apply one in-memory mutation and repair superficial self-hashes."""

    candidate = deepcopy(payload)
    rows = candidate["per_unit_rows"]
    if attack_id == "family_label_swap":
        names = list(row["family"] for row in candidate["preconditions_checked"]["model_registry"])
        rows[0]["family"] = names[1] if len(names) > 1 else "swapped_family"
        rows[0]["row_hash"] = row_hash(rows[0])
    elif attack_id == "aggregate_only_eligibility":
        candidate["per_unit_rows"] = []
    elif attack_id == "one_row_selection":
        candidate["per_unit_rows"] = rows[:1]
    elif attack_id == "missing_failure_erasure":
        target = next((row for row in rows if row.get("failure_class") is not None), rows[0])
        target["failure_class"] = None
        target["exact_success"] = True
        target["row_hash"] = row_hash(target)
    elif attack_id == "legacy_substitution":
        target = (
            candidate["eligible_model_specs"][0]
            if candidate["eligible_model_specs"]
            else candidate["family_replay_rows"][0]
        )
        target["repository_id"] = "legacy/smoke-model-GGUF"
    elif attack_id == "row_duplication":
        candidate["per_unit_rows"][-1] = deepcopy(rows[0])
    elif attack_id == "split_drift":
        rows[0]["split"] = "held" if rows[0]["split"] == "calibration" else "calibration"
        rows[0]["row_hash"] = row_hash(rows[0])
    elif attack_id == "model_identity_drift":
        rows[0]["model_identity_sha256"] = "sha256:" + "0" * 64
        rows[0]["row_hash"] = row_hash(rows[0])
    elif attack_id == "exact_check_substitution":
        if rows[0].get("exact_executor_result") is None:
            rows[0]["exact_executor_result"] = {"executor_version": "substitute"}
        else:
            rows[0]["exact_executor_result"]["executor_version"] = "substitute"
        rows[0]["row_hash"] = row_hash(rows[0])
    elif attack_id == "protected_file_mutation":
        candidate["protected_files_unchanged"]["all_unchanged"] = False
        candidate["protected_files_unchanged"]["rows"][0]["unchanged"] = False
    else:
        raise ValueError(f"unknown attack: {attack_id}")
    return candidate


def build_attack_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Prove each required mutation reaches its named fail-closed check."""

    output = []
    for attack_id in REQUIRED_ATTACK_IDS:
        candidate = attack_candidate(payload, attack_id)
        reduction = readiness_reducer(candidate)
        check = ATTACK_CHECKS[attack_id]
        output.append(
            {
                "attack_id": attack_id,
                "target_check": check,
                "target_check_passed": reduction["checks"].get(check),
                "candidate_headroom_benchmark_ready_score": reduction[
                    "headroom_benchmark_ready_score"
                ],
                "failed_checks": reduction["failed_checks"],
                "failed_closed": reduction["checks"].get(check) is False
                and reduction["headroom_benchmark_ready_score"] == 0.0,
            }
        )
    return output


def validate_report(payload: Mapping[str, Any]) -> list[str]:
    """Reject schema, gate, row, attack, protection, or checksum drift."""

    errors = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
        return errors
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    reduction = readiness_reducer(payload)
    if reduction["failed_checks"]:
        errors.append("reducer_checks_failed:" + ",".join(reduction["failed_checks"]))
    if payload.get("headroom_benchmark_ready_score") == 1.0:
        if payload.get("verdict_class") != "null" or not str(payload.get("status", "")).startswith(
            "complete_"
        ):
            errors.append("eligible_disposition_mismatch")
    elif payload.get("family_replay_rows") and all(
        row.get("source_complete") is True for row in payload["family_replay_rows"]
    ):
        if (
            payload.get("verdict_class") != "null"
            or payload.get("status") != "complete_no_family_headroom"
        ):
            errors.append("no_headroom_disposition_mismatch")
    elif payload.get("verdict_class") != "blocked" or not str(payload.get("status", "")).startswith(
        "blocked_"
    ):
        errors.append("blocked_disposition_mismatch")
    attacks = {
        row.get("attack_id"): row
        for row in payload.get("attack_rows", [])
        if isinstance(row, Mapping)
    }
    if set(attacks) != set(REQUIRED_ATTACK_IDS) or not all(
        row.get("failed_closed") is True for row in attacks.values()
    ):
        errors.append("attack_rows_failed")
    if float(payload.get("duration_s", -1.0)) < 0.0:
        errors.append("duration_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def atomic_write_report(path: str | Path, payload: Mapping[str, Any]) -> JsonDict:
    """Validate, sync, replace, then sync the directory entry."""

    errors = validate_report(payload)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".exp6608-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "byte_count": len(encoded),
        "atomic_replace": True,
        "directory_fsync": True,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    """Build the reducer artifact without any model or GPU call."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    report = build_report(REPO_ROOT, args.date)
    receipt = atomic_write_report(REPO_ROOT / RESULT_RELATIVE_PATH, report)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "status": report["status"],
                "headroom_benchmark_ready_score": report["headroom_benchmark_ready_score"],
                "sha256": receipt["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
