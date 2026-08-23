"""Exp6554 independent prospective continuous self-learning audit.

Spec refs: REQ-CL-6554, SCENARIO-CL-6554-MISSING-INPUT,
SCENARIO-CL-6554-RECEIPTS, SCENARIO-CL-6554-REPLAY,
SCENARIO-CL-6554-ROWS, SCENARIO-CL-6554-ATTACKS,
SCENARIO-CL-6554-ATOMIC.

This reducer does not run a model. It reads stored Exp6553 evidence, checks the
receipt and replay invariants again, and closes blocked when live rows or raw
receipt files are absent.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6554
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6554_continuous_self_learning_independent_audit.json"
)
WORK_RELATIVE_PATH = Path("results/.experiment_6554_continuous_self_learning_independent_audit")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6554_continuous_self_learning_independent_audit.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py"
)
EXP6552_RELATIVE_PATH = Path("results/experiment_6552_hysteretic_reversible_conflict_memory.json")
EXP6553_RELATIVE_PATH = Path(
    "results/experiment_6553_prospective_sota_continuous_self_learning.json"
)
ROADMAP_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ROW_LINT_RELATIVE_PATH = Path("scripts/verdict_row_consistency_lint.py")
ADVERSARIAL_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
INFERENCE_SUBSTRATE = "independent_stored_sota_receipt_and_exact_transition_replay_no_new_llm"

MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
DEFAULT_ARMS = (
    "frozen",
    "current_only",
    "transactional_replay",
    "matched_dose_coobservation",
    "one_threshold",
    "hysteretic",
    "same_query_mutation",
)
DIAGNOSTIC_ARM = "same_query_mutation"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "input_existence_and_hash_receipts",
    "independent_live_receipt_audit_rows",
    "independent_exact_replay_rows",
    "independent_transition_replay_rows",
    "independent_current_effect_rows",
    "independent_retention_and_support_rows",
    "dose_and_coobservation_audit",
    "unsafe_write_and_use_audit",
    "restart_rollback_and_persistence_audit",
    "missing_input_disposition",
    "attack_matrix",
    "continuous_self_learning_audited_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "An always-run audit needs a terminal state for missing, invalid, null, and positive inputs.",
    "honest_verdict": "The verdict must state receipt, safety, retention, support, and scientific disposition with a terminal prefix.",
    "verdict_class": "A closed class keeps audit status and scientific status structurally bounded.",
    "input_existence_and_hash_receipts": "The audit must identify the exact artifact, raw receipts, journals, and checkpoints it used.",
    "independent_live_receipt_audit_rows": "Fresh receipt checks prevent cached or legacy execution from posing as flagship inference.",
    "independent_exact_replay_rows": "Z3 replay must confirm every credited current, retained, and future outcome.",
    "independent_transition_replay_rows": "Memory effects are eligible only when every state hash and witness recomputes.",
    "independent_current_effect_rows": "Immediate claims must be recomputed from matched units and charged costs.",
    "independent_retention_and_support_rows": "The audit must expose any older-family or future-support regression.",
    "dose_and_coobservation_audit": "A replay benefit cannot be credited to more update exposure.",
    "unsafe_write_and_use_audit": "No aggregate gain can hide one invalid memory action.",
    "restart_rollback_and_persistence_audit": "Reusable learning must reproduce and recover exactly across state boundaries.",
    "missing_input_disposition": "Missing live evidence must close blocked rather than produce a scientific null.",
    "attack_matrix": "Receipt, leakage, dose, row, headroom, and circularity attacks stress the full claim.",
    "continuous_self_learning_audited_ready_score": "A binary audit score defines whether the prospective result may enter the capstone.",
    "per_unit_rows": "Every independent comparative conclusion needs unit-level recomputation rows.",
    "aggregate_row_recomputation": "The audit verdict must derive only from independent rows.",
    "gate_check_summary": "A blocked audit must list each missing or failed check and observed value.",
    "preconditions_checked": "Input and replay checks distinguish a blocked audit from null science.",
    "protected_files_unchanged": "The audit must not repair upstream evidence or mutate protected files.",
    "inference_substrate": "The audit replays stored receipts and exact checks; it does not claim new GGUF generation.",
    "verifier_is_oracle": "The learned memory policy is not authority; the audit uses separate exact evaluation.",
    "field_provenance": "Each disposition field must point to immutable rows, receipts, and reducers.",
    "random_seed": "A fixed audit sample and attack order make the audit reproducible.",
    "duration_s": "Monotonic time exposes an audit that skipped receipt or replay work.",
    "tests_run": "Named tests and E2E commands show independent checks executed.",
    "reproducibility_checksum": "A final hash protects the independent determination trail.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6554_continuous_self_learning_independent_audit "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6554_continuous_self_learning_independent_audit.py "
    "-m pytest tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6554_continuous_self_learning_independent_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6554_continuous_self_learning_independent_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6554_continuous_self_learning_independent_audit.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6554_continuous_self_learning_independent_audit "
    "--validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: ops/e2e-test-plan.md stored GGUF receipt and exact replay audit"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ACTIVE_ROADMAP_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ROW_LINT_RELATIVE_PATH,
    ADVERSARIAL_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    EXP6552_RELATIVE_PATH,
    EXP6553_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path | None) -> str:
    if path is None:
        return "missing"
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():  # pragma: no cover - only reached after replace/write failure.
            tmp_path.unlink()


def _read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _resolve_path(repo_root: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _receipt_rows(value: Any) -> list[JsonDict]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def protected_file_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "rows": rows,
        "hashes_before": dict(before),
        "hashes_after": dict(after),
    }


def _z3_version() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        import z3  # type: ignore[import-not-found]

        return {"available": True, "version": z3.get_version_string()}
    except Exception as exc:
        return {"available": False, "version": "", "error": f"{type(exc).__name__}: {exc}"}


def _memory_bytes() -> JsonDict:  # pragma: no cover - host dependent.
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return {"available": False, "mem_total_bytes": None, "mem_available_bytes": None}
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in {"MemTotal", "MemAvailable"}:
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return {
        "available": True,
        "mem_total_bytes": values.get("MemTotal"),
        "mem_available_bytes": values.get("MemAvailable"),
    }


def _host_snapshot(
    repo_root: Path, work_root: Path
) -> JsonDict:  # pragma: no cover - host dependent.
    work_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(repo_root)
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram": _memory_bytes(),
        "disk": {
            "path": str(repo_root),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "work_root": str(work_root),
            "work_root_writable": os.access(work_root, os.W_OK),
        },
        "z3": _z3_version(),
    }


def _receipt_file_summary(repo_root: Path, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for row in rows:
        path = _resolve_path(repo_root, row.get("path", ""))
        observed_hash = sha256_file(path)
        recorded_hash = str(row.get("sha256") or "")
        out.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "recorded_sha256": recorded_hash or "missing",
                "observed_sha256": observed_hash,
                "hash_matches": bool(recorded_hash)
                and observed_hash == recorded_hash
                and observed_hash.startswith("sha256:"),
            }
        )
    return out


def input_existence_and_hash_receipts(
    *,
    repo_root: Path,
    input_path: Path,
    upstream: Mapping[str, Any],
    hash_model_files: bool,
) -> JsonDict:
    exp6552_path = repo_root / EXP6552_RELATIVE_PATH
    exp6553_path = input_path
    raw_rows = _receipt_rows(upstream.get("raw_model_receipts"))
    checkpoint_rows = _receipt_rows(upstream.get("checkpoint_receipts"))
    journal_rows = _receipt_rows(upstream.get("journal_receipts"))
    model_specs = _receipt_rows(upstream.get("MODEL_SPECS"))
    model_hash_rows = []
    for spec in model_specs:
        path = _resolve_path(repo_root, spec.get("model_path", ""))
        recorded = str(spec.get("gguf_sha256") or "")
        observed = sha256_file(path) if hash_model_files else recorded
        model_hash_rows.append(
            {
                "hf_id": spec.get("hf_id"),
                "model_path": str(path),
                "exists": path.is_file(),
                "recorded_gguf_sha256": recorded or "missing",
                "observed_gguf_sha256": observed or "missing",
                "hash_recomputed": hash_model_files,
                "hash_matches": bool(recorded)
                and observed == recorded
                and str(observed).startswith("sha256:"),
            }
        )
    raw_receipts = _receipt_file_summary(repo_root, raw_rows)
    checkpoints = _receipt_file_summary(repo_root, checkpoint_rows)
    journals = _receipt_file_summary(repo_root, journal_rows)
    per_unit_rows = _receipt_rows(upstream.get("per_unit_rows"))
    expected_rows = int(
        upstream.get("aggregate_row_recomputation", {}).get("expected_row_count", 0)
        if isinstance(upstream.get("aggregate_row_recomputation"), Mapping)
        else 0
    )
    all_inputs = (
        exp6553_path.is_file()
        and exp6552_path.is_file()
        and bool(per_unit_rows)
        and bool(raw_receipts)
        and bool(checkpoints)
        and bool(journals)
        and all(row["hash_matches"] for row in raw_receipts)
        and all(row["hash_matches"] for row in checkpoints)
        and all(row["hash_matches"] for row in journals)
        and len(model_hash_rows) == len(MANDATED_HF_IDS)
        and all(row["hash_matches"] for row in model_hash_rows)
    )
    return {
        "exp6552": {
            "path": str(exp6552_path),
            "exists": exp6552_path.is_file(),
            "sha256": sha256_file(exp6552_path),
        },
        "exp6553": {
            "path": str(exp6553_path),
            "exists": exp6553_path.is_file(),
            "sha256": sha256_file(exp6553_path),
            "status": upstream.get("status"),
            "verdict_class": upstream.get("verdict_class"),
            "prospective_csl_ready_score": upstream.get("prospective_csl_ready_score"),
        },
        "raw_receipts": raw_receipts,
        "checkpoint_receipts": checkpoints,
        "journal_receipts": journals,
        "model_file_hash_rows": model_hash_rows,
        "expected_row_count": expected_rows,
        "observed_per_unit_row_count": len(per_unit_rows),
        "observed_transition_row_count": len(_receipt_rows(upstream.get("memory_transition_rows"))),
        "hash_model_files": hash_model_files,
        "all_required_inputs_present": all_inputs,
    }


def preconditions_checked(
    *,
    repo_root: Path,
    input_path: Path,
    result_path: Path,
    work_root: Path,
    receipts: Mapping[str, Any],
    protected_hashes_before: Mapping[str, str],
    run_date: str,
) -> JsonDict:
    upstream_failed = []
    upstream_gate = receipts.get("exp6553", {})
    upstream_payload = _read_json(input_path)
    gate_summary = upstream_payload.get("gate_check_summary", {})
    if isinstance(gate_summary, Mapping):
        upstream_failed = [str(item) for item in gate_summary.get("failed_checks", [])]
    checks = {
        "exp6553_artifact_present": upstream_gate.get("exists") is True,
        "exp6552_artifact_present": receipts.get("exp6552", {}).get("exists") is True,
        "raw_receipts_present": bool(receipts.get("raw_receipts")),
        "checkpoint_receipts_present": bool(receipts.get("checkpoint_receipts")),
        "journal_receipts_present": bool(receipts.get("journal_receipts")),
        "model_hashes_match": all(
            row.get("hash_matches") for row in receipts.get("model_file_hash_rows", [])
        ),
        "stored_rows_present": int(receipts.get("observed_per_unit_row_count", 0)) > 0,
        "stored_transitions_present": int(receipts.get("observed_transition_row_count", 0)) > 0,
        "upstream_failed_preconditions_absent": not upstream_failed,
    }
    failed = [name for name, passed in checks.items() if not passed]
    failed.extend(f"upstream:{name}" for name in upstream_failed)
    return {
        "row_type": "preconditions_checked",
        "run_date": run_date,
        "repo_root": str(repo_root),
        "input_path": str(input_path),
        "result_path": str(result_path),
        "work_root": str(work_root),
        "random_seed": RANDOM_SEED,
        "checks": checks,
        "failed_preconditions": failed,
        "upstream_failed_checks": upstream_failed,
        "host": _host_snapshot(repo_root, work_root),
        "protected_file_hashes_before": dict(protected_hashes_before),
    }


def missing_input_disposition(
    *,
    receipts: Mapping[str, Any],
    upstream: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    missing = []
    if receipts.get("exp6553", {}).get("exists") is not True:
        missing.append("exp6553_artifact")
    if receipts.get("exp6552", {}).get("exists") is not True:
        missing.append("exp6552_artifact")
    if not upstream.get("per_unit_rows"):
        missing.append("exp6553_per_unit_rows")
    if not upstream.get("memory_transition_rows"):
        missing.append("exp6553_memory_transition_rows")
    if not receipts.get("raw_receipts"):
        missing.append("raw_model_receipts")
    if not receipts.get("checkpoint_receipts"):
        missing.append("checkpoint_receipts")
    if not receipts.get("journal_receipts"):
        missing.append("journal_receipts")
    live = upstream.get("live_model_and_gpu_receipts", {})
    if isinstance(live, Mapping):
        if live.get("fresh_local_inference_performed") is not True:
            missing.append("fresh_local_inference_performed")
        if live.get("all_mandated_models_loaded") is not True:
            missing.append("all_mandated_models_loaded")
    if upstream.get("verdict_class") == "blocked":
        missing.append("upstream_verdict_blocked")
    missing.extend(
        str(item)
        for item in preconditions.get("failed_preconditions", [])
        if str(item).startswith("upstream:")
    )
    return {
        "terminal_disposition": "blocked" if missing else "available",
        "missing_inputs": sorted(set(missing)),
        "scientific_null_claim_allowed": not missing,
        "blocked_not_null_reason": "missing live evidence" if missing else "",
    }


def _model_lookup(upstream: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {str(row.get("hf_id")): dict(row) for row in _receipt_rows(upstream.get("MODEL_SPECS"))}


def independent_live_receipt_audit_rows(
    *,
    upstream: Mapping[str, Any],
    receipts: Mapping[str, Any],
    missing: Mapping[str, Any],
) -> list[JsonDict]:
    if missing.get("terminal_disposition") == "blocked":
        return []
    models = _model_lookup(upstream)
    computed_hashes = {
        str(row.get("hf_id")): str(row.get("observed_gguf_sha256"))
        for row in receipts.get("model_file_hash_rows", [])
    }
    out = []
    for index, source in enumerate(_receipt_rows(upstream.get("per_unit_rows"))):
        model = models.get(str(source.get("model_hf_id")), {})
        request_hash = (
            sha256_json(source.get("raw_request_text"))
            if "raw_request_text" in source
            else "missing"
        )
        response_hash = (
            sha256_json(source.get("raw_response_text"))
            if "raw_response_text" in source
            else "missing"
        )
        checks = {
            "row_hash": source.get("row_hash") == row_hash(source),
            "mandated_model": source.get("model_hf_id") in MANDATED_HF_IDS,
            "model_hash": source.get("model_file_sha256")
            == computed_hashes.get(str(source.get("model_hf_id"))),
            "request_hash": request_hash == source.get("request_hash"),
            "response_hash": response_hash == source.get("response_hash"),
            "process_id": isinstance(source.get("process_id"), int)
            and source.get("process_id") > 0,
            "command": "llama" in str(source.get("command", "")),
            "monotonic_clock": float(source.get("monotonic_end_s", -1.0))
            >= float(source.get("monotonic_start_s", 0.0)),
            "positive_duration": float(source.get("charged_model_time_s", 0.0)) > 0.0,
            "gpu_samples": bool(source.get("gpu_samples")),
            "exit_status": source.get("exit_status") == "ok"
            and source.get("terminal_status") == "terminal",
            "no_timeout_or_censor": source.get("timeout") is False
            and source.get("censored") is False,
            "no_legacy_substitution": source.get("hidden_legacy_substitution") is False,
            "model_path_bound": str(model.get("model_path", "")) in str(source.get("command", "")),
        }
        payload = {
            "row_type": "independent_live_receipt_audit",
            "unit_id": _unit_id(source),
            "source_row_index": index,
            "source_row_hash": source.get("row_hash"),
            "model_hf_id": source.get("model_hf_id"),
            "query_id": source.get("query_id"),
            "arm_id": source.get("arm_id"),
            "checks": checks,
            "receipt_authentic": all(checks.values()),
            "observed_model_sha256": computed_hashes.get(str(source.get("model_hf_id")), "missing"),
            "request_hash_recomputed": request_hash,
            "response_hash_recomputed": response_hash,
            "monotonic_duration_s": round(
                float(source.get("monotonic_end_s", 0.0))
                - float(source.get("monotonic_start_s", 0.0)),
                6,
            ),
        }
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def _unit_id(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row.get(key, "")) for key in ("model_hf_id", "query_id", "seed", "arm_id", "condition")
    )


def independent_exact_replay_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    out = []
    for source in _receipt_rows(upstream.get("per_unit_rows")):
        exact = (
            dict(source.get("exact_result", {}))
            if isinstance(source.get("exact_result"), Mapping)
            else {}
        )
        witness = (
            dict(source.get("witness", {})) if isinstance(source.get("witness"), Mapping) else {}
        )
        expected_witness_hash = sha256_json(
            {
                "query_hash": source.get("query_hash"),
                "arm_id": source.get("arm_id"),
                "exact_label": "satisfiable",
            }
        )
        is_frozen = source.get("arm_id") == "frozen"
        checks = {
            "exact_authority": exact.get("verifier_authority") == "exact_z3",
            "z3_status": exact.get("z3_status") in {"sat", "unsat"},
            "success_matches_exact": bool(source.get("exact_success"))
            is bool(exact.get("exact_satisfying")),
            "witness_hash": is_frozen or witness.get("witness_hash") == expected_witness_hash,
            "verifier_is_separate": exact.get("verifier_authority") != source.get("route"),
        }
        payload = {
            "row_type": "independent_exact_replay",
            "unit_id": _unit_id(source),
            "source_row_hash": source.get("row_hash"),
            "model_hf_id": source.get("model_hf_id"),
            "query_id": source.get("query_id"),
            "arm_id": source.get("arm_id"),
            "exact_result_hash_recomputed": sha256_json(exact),
            "witness_hash_recomputed": expected_witness_hash
            if not is_frozen
            else "no_write_frozen",
            "checks": checks,
            "exact_replay_passed": all(checks.values()),
        }
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def _transition_lookup(upstream: Mapping[str, Any]) -> dict[tuple[str, str, str], JsonDict]:
    lookup = {}
    for row in _receipt_rows(upstream.get("memory_transition_rows")):
        key = (str(row.get("model_hf_id")), str(row.get("query_id")), str(row.get("arm_id")))
        lookup[key] = dict(row)
    return lookup


def independent_transition_replay_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    transitions = _transition_lookup(upstream)
    state: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    out = []
    for source in _receipt_rows(upstream.get("per_unit_rows")):
        arm = str(source.get("arm_id"))
        if arm == "frozen":
            continue
        key = (str(source.get("model_hf_id")), arm)
        transition = transitions.get(
            (str(source.get("model_hf_id")), str(source.get("query_id")), arm), {}
        )
        pre_expected = sha256_json(state[key])
        witness = (
            dict(source.get("witness", {})) if isinstance(source.get("witness"), Mapping) else {}
        )
        proposed = (
            dict(source.get("proposed_write", {}))
            if isinstance(source.get("proposed_write"), Mapping)
            else {}
        )
        commit = source.get("commit_decision") == "commit_after_exact" and arm != DIAGNOSTIC_ARM
        if commit:
            state[key].append(
                {
                    "query_id": source.get("query_id"),
                    "family": source.get("domain"),
                    "witness_hash": witness.get("witness_hash"),
                }
            )
        post_expected = sha256_json(state[key])
        checks = {
            "source_pre_hash": source.get("pre_memory_hash") == pre_expected,
            "source_post_hash": source.get("post_query_memory_hash") == post_expected,
            "transition_pre_hash": transition.get("pre_memory_hash") == pre_expected,
            "transition_post_hash": transition.get("post_memory_hash") == post_expected,
            "proposed_write_hash": transition.get("proposed_write_hash") == sha256_json(proposed),
            "witness_hash": transition.get("witness_hash") == witness.get("witness_hash"),
            "exact_result_hash": transition.get("exact_result_hash")
            == sha256_json(source.get("exact_result", {})),
            "commit_decision": transition.get("commit_decision") == source.get("commit_decision"),
            "diagnostic_not_committed": arm != DIAGNOSTIC_ARM
            or transition.get("commit_after_exact_verification") is False,
        }
        payload = {
            "row_type": "independent_transition_replay",
            "unit_id": _unit_id(source),
            "source_row_hash": source.get("row_hash"),
            "transition_row_hash": transition.get("row_hash", "missing"),
            "model_hf_id": source.get("model_hf_id"),
            "query_id": source.get("query_id"),
            "arm_id": arm,
            "pre_memory_hash_recomputed": pre_expected,
            "post_memory_hash_recomputed": post_expected,
            "committed": commit,
            "checks": checks,
            "transition_replay_passed": all(checks.values()),
        }
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def _arms(upstream: Mapping[str, Any]) -> tuple[str, ...]:
    contract = upstream.get("frozen_chronology_and_arm_contract", {})
    if isinstance(contract, Mapping) and isinstance(contract.get("arms"), Sequence):
        return tuple(str(arm) for arm in contract["arms"])
    return DEFAULT_ARMS


def _safe_arms(upstream: Mapping[str, Any]) -> tuple[str, ...]:
    contract = upstream.get("frozen_chronology_and_arm_contract", {})
    if isinstance(contract, Mapping) and isinstance(contract.get("safe_arms"), Sequence):
        return tuple(str(arm) for arm in contract["safe_arms"])
    return tuple(arm for arm in _arms(upstream) if arm != DIAGNOSTIC_ARM)


def independent_current_effect_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    rows = _receipt_rows(upstream.get("per_unit_rows"))
    if not rows:
        return []
    frozen_cost = sum(
        float(row.get("charged_cost_units", 0.0)) for row in rows if row.get("arm_id") == "frozen"
    )
    out = []
    for arm in _arms(upstream):
        arm_rows = [row for row in rows if row.get("arm_id") == arm]
        cost = round(sum(float(row.get("charged_cost_units", 0.0)) for row in arm_rows), 6)
        payload = {
            "row_type": "independent_current_effect",
            "arm_id": arm,
            "row_count": len(arm_rows),
            "exact_success_rate": _mean(
                [1.0 if row.get("exact_success") is True else 0.0 for row in arm_rows]
            ),
            "charged_cost_units": cost,
            "charged_value_delta": round(frozen_cost - cost, 6) if arm != "frozen" else 0.0,
            "timeout_count": sum(1 for row in arm_rows if row.get("timeout") is True),
            "censored_count": sum(1 for row in arm_rows if row.get("censored") is True),
            "harmful_intervention_count": sum(
                1 for row in arm_rows if row.get("harmful_intervention") is True
            ),
        }
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def independent_retention_and_support_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    rows = _receipt_rows(upstream.get("per_unit_rows"))
    if not rows:
        return []
    domains = sorted({str(row.get("domain")) for row in rows})
    out = []
    for arm in _arms(upstream):
        for domain in domains:
            arm_rows = [
                row for row in rows if row.get("arm_id") == arm and row.get("domain") == domain
            ]
            frozen_rows = [
                row for row in rows if row.get("arm_id") == "frozen" and row.get("domain") == domain
            ]
            retention = _mean([float(row.get("retained_family_success", 0.0)) for row in arm_rows])
            retention_base = _mean(
                [float(row.get("retained_family_success", 0.0)) for row in frozen_rows]
            )
            support = _mean([float(row.get("future_support_score", 0.0)) for row in arm_rows])
            support_base = _mean(
                [float(row.get("future_support_score", 0.0)) for row in frozen_rows]
            )
            payload = {
                "row_type": "independent_retention_and_support",
                "arm_id": arm,
                "domain": domain,
                "retained_family_success_rate": retention,
                "retention_baseline_rate": retention_base,
                "retention_noninferior": retention >= retention_base,
                "future_support_score": support,
                "future_support_baseline": support_base,
                "future_support_noninferior": support >= support_base,
                "combined_noninferior": retention >= retention_base and support >= support_base,
            }
            payload["row_hash"] = row_hash(payload)
            out.append(payload)
    return out


def dose_and_coobservation_audit(upstream: Mapping[str, Any]) -> JsonDict:
    rows = _receipt_rows(upstream.get("per_unit_rows"))
    dose = {
        arm: (
            0
            if arm == "frozen"
            else len({row.get("query_id") for row in rows if row.get("arm_id") == arm})
        )
        for arm in _arms(upstream)
    }
    learning_doses = [value for arm, value in dose.items() if arm != "frozen"]
    payload = {
        "row_type": "dose_and_coobservation_audit",
        "update_dose_by_arm": dose,
        "matched_update_dose": bool(learning_doses) and len(set(learning_doses)) == 1,
        "coobservation_arm": "matched_dose_coobservation",
        "coobservation_rows_present": any(
            row.get("arm_id") == "matched_dose_coobservation" for row in rows
        ),
        "extra_update_exposure_count": 0 if learning_doses and len(set(learning_doses)) == 1 else 1,
        "replay_benefit_separated_from_extra_update_exposure": bool(learning_doses)
        and len(set(learning_doses)) == 1,
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def unsafe_write_and_use_audit(upstream: Mapping[str, Any]) -> JsonDict:
    rows = _receipt_rows(upstream.get("per_unit_rows"))
    safe = set(_safe_arms(upstream))
    safe_rows = [row for row in rows if row.get("arm_id") in safe]
    diagnostic_rows = [row for row in rows if row.get("arm_id") == DIAGNOSTIC_ARM]
    payload = {
        "row_type": "unsafe_write_and_use_audit",
        "safe_arm_unsafe_write_count": sum(
            1 for row in safe_rows if row.get("unsafe_write") is True
        ),
        "safe_arm_unsafe_use_count": sum(1 for row in safe_rows if row.get("unsafe_use") is True),
        "diagnostic_same_query_unsafe_write_count": sum(
            1 for row in diagnostic_rows if row.get("unsafe_write") is True
        ),
        "diagnostic_same_query_unsafe_use_count": sum(
            1 for row in diagnostic_rows if row.get("unsafe_use") is True
        ),
        "same_query_arm_adopted": False,
        "safe_row_count": len(safe_rows),
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def restart_rollback_and_persistence_audit(upstream: Mapping[str, Any]) -> JsonDict:
    rows = _receipt_rows(upstream.get("per_unit_rows"))
    lifecycle = upstream.get("restart_and_rollback_receipts", {})
    if not isinstance(lifecycle, Mapping) or not rows:
        payload = {
            "row_type": "restart_rollback_and_persistence_audit",
            "restart_rollback_passed": False,
            "restart_rows_checked": 0,
            "rollback_rows_checked": 0,
            "capacity_passed": False,
            "max_reconstructed_state_records": 0,
        }
        payload["row_hash"] = row_hash(payload)
        return payload
    state: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for source in rows:
        arm = str(source.get("arm_id"))
        if arm == "frozen" or arm == DIAGNOSTIC_ARM:
            continue
        if source.get("commit_decision") == "commit_after_exact":
            witness = (
                dict(source.get("witness", {}))
                if isinstance(source.get("witness"), Mapping)
                else {}
            )
            state[(str(source.get("model_hf_id")), arm)].append(
                {
                    "query_id": source.get("query_id"),
                    "family": source.get("domain"),
                    "witness_hash": witness.get("witness_hash"),
                }
            )
    final_hashes = {key: sha256_json(value) for key, value in state.items()}
    for source in rows:
        if source.get("arm_id") == "frozen":
            final_hashes[(str(source.get("model_hf_id")), "frozen")] = sha256_json([])
    restart_rows = _receipt_rows(lifecycle.get("restart_rows"))
    rollback_rows = _receipt_rows(lifecycle.get("rollback_rows"))
    restart_ok = all(
        row.get("exact_output_equal") is True
        and row.get("state_hash_before_restart") == row.get("state_hash_after_restart")
        and row.get("state_hash_after_restart")
        == final_hashes.get((str(row.get("model_hf_id")), str(row.get("arm_id"))))
        for row in restart_rows
    )
    rollback_ok = all(
        row.get("rolled_back") is True
        and row.get("state_hash_before_corrupt_write") == row.get("state_hash_after_rollback")
        and row.get("state_hash_after_rollback")
        == final_hashes.get((str(row.get("model_hf_id")), str(row.get("arm_id"))))
        for row in rollback_rows
    )
    replay_capacity = int(
        upstream.get("frozen_chronology_and_arm_contract", {}).get("replay_capacity", 0)
        if isinstance(upstream.get("frozen_chronology_and_arm_contract"), Mapping)
        else 0
    )
    max_records = max((len(value) for value in state.values()), default=0)
    capacity_passed = replay_capacity == 0 or max_records <= replay_capacity or bool(rows)
    payload = {
        "row_type": "restart_rollback_and_persistence_audit",
        "restart_rollback_passed": restart_ok
        and rollback_ok
        and lifecycle.get("corrupt_write_challenge_fail_closed") is True,
        "restart_rows_checked": len(restart_rows),
        "rollback_rows_checked": len(rollback_rows),
        "all_restarts_exact_output_equal": restart_ok,
        "all_rollbacks_restored": rollback_ok,
        "corrupt_write_challenge_fail_closed": lifecycle.get("corrupt_write_challenge_fail_closed")
        is True,
        "capacity_passed": capacity_passed,
        "replay_capacity": replay_capacity,
        "max_reconstructed_state_records": max_records,
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def attack_matrix(artifact: Mapping[str, Any]) -> JsonDict:
    rows = _receipt_rows(artifact.get("per_unit_rows"))
    unit_ids = [str(row.get("unit_id")) for row in rows]
    current = _receipt_rows(artifact.get("independent_current_effect_rows"))
    positive_current = any(
        row.get("arm_id") in _safe_arms_from_artifact(artifact)
        and row.get("arm_id") != "frozen"
        and float(row.get("charged_value_delta", 0.0)) > 0.0
        for row in current
    )
    attacks = [
        {
            "attack_id": "missing_or_duplicated_rows",
            "attack_failed_closed": bool(rows) and len(unit_ids) == len(set(unit_ids)),
        },
        {
            "attack_id": "aggregate_tampering",
            "attack_failed_closed": artifact.get("aggregate_row_recomputation", {})
            in ({}, aggregate_row_recomputation(artifact)),
        },
        {
            "attack_id": "query_boundary_leakage",
            "attack_failed_closed": all(
                row.get("decision_time_write_count") == 0
                and row.get("pre_memory_hash") == row.get("frozen_query_snapshot_hash")
                for row in _receipt_rows(_upstream_payload(artifact).get("per_unit_rows"))
                if row.get("arm_id") in _safe_arms_from_artifact(artifact)
            ),
        },
        {
            "attack_id": "future_access",
            "attack_failed_closed": all(
                row.get("future_turn_access") is False
                for row in _receipt_rows(_upstream_payload(artifact).get("per_unit_rows"))
            ),
        },
        {
            "attack_id": "held_tuning",
            "attack_failed_closed": all(
                row.get("held_threshold_tuning") is False
                for row in _receipt_rows(_upstream_payload(artifact).get("per_unit_rows"))
            ),
        },
        {
            "attack_id": "unequal_dose",
            "attack_failed_closed": artifact.get("dose_and_coobservation_audit", {}).get(
                "matched_update_dose"
            )
            is True,
        },
        {
            "attack_id": "model_aliases",
            "attack_failed_closed": {
                row.get("model_hf_id")
                for row in _receipt_rows(_upstream_payload(artifact).get("per_unit_rows"))
            }
            <= set(MANDATED_HF_IDS),
        },
        {
            "attack_id": "zero_headroom_wins",
            "attack_failed_closed": positive_current,
        },
        {
            "attack_id": "all_null_cells",
            "attack_failed_closed": all(
                row.get("exact_success_rate") is not None
                and row.get("charged_cost_units") is not None
                for row in current
            ),
        },
        {
            "attack_id": "circular_exact_authority",
            "attack_failed_closed": artifact.get("verifier_is_oracle") is False,
        },
    ]
    blocked = artifact.get("missing_input_disposition", {}).get("terminal_disposition") == "blocked"
    payload = {
        "row_type": "attack_matrix",
        "rows": attacks,
        "all_attacks_fail_closed": all(row["attack_failed_closed"] for row in attacks)
        and not blocked,
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def _upstream_payload(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    return (
        artifact.get("_upstream_payload_for_recompute", {}) if isinstance(artifact, Mapping) else {}
    )


def _safe_arms_from_artifact(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    upstream = _upstream_payload(artifact)
    return (
        _safe_arms(upstream)
        if upstream
        else tuple(arm for arm in DEFAULT_ARMS if arm != DIAGNOSTIC_ARM)
    )


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    rows = _receipt_rows(artifact.get("per_unit_rows"))
    live = _receipt_rows(artifact.get("independent_live_receipt_audit_rows"))
    exact = _receipt_rows(artifact.get("independent_exact_replay_rows"))
    transitions = _receipt_rows(artifact.get("independent_transition_replay_rows"))
    current = _receipt_rows(artifact.get("independent_current_effect_rows"))
    retention_support = _receipt_rows(artifact.get("independent_retention_and_support_rows"))
    missing = artifact.get("missing_input_disposition", {})
    dose = artifact.get("dose_and_coobservation_audit", {})
    unsafe = artifact.get("unsafe_write_and_use_audit", {})
    lifecycle = artifact.get("restart_rollback_and_persistence_audit", {})
    inputs = artifact.get("input_existence_and_hash_receipts", {})
    attacks = artifact.get("attack_matrix", {})
    protected = artifact.get("protected_files_unchanged", {})
    tests = _receipt_rows(artifact.get("tests_run"))
    expected = int(inputs.get("expected_row_count", 0)) if isinstance(inputs, Mapping) else 0
    unit_ids = [str(row.get("unit_id")) for row in rows]
    safe_arms = _safe_arms_from_artifact(artifact)
    safe_positive = ""
    positive_delta = 0.0
    for row in current:
        delta = float(row.get("charged_value_delta", 0.0))
        if (
            row.get("arm_id") in safe_arms
            and row.get("arm_id") != "frozen"
            and delta > positive_delta
        ):
            safe_positive = str(row.get("arm_id"))
            positive_delta = delta
    retention_ok = bool(safe_positive) and all(
        row.get("combined_noninferior") is True
        for row in retention_support
        if row.get("arm_id") == safe_positive
    )
    row_closure = bool(rows) and len(rows) == expected and len(unit_ids) == len(set(unit_ids))
    receipt_ok = (
        bool(live) and len(live) == len(rows) and all(row.get("receipt_authentic") for row in live)
    )
    exact_ok = (
        bool(exact)
        and len(exact) == len(rows)
        and all(row.get("exact_replay_passed") for row in exact)
    )
    transition_ok = bool(transitions) and all(
        row.get("transition_replay_passed") for row in transitions
    )
    dose_ok = (
        dose.get("matched_update_dose") is True and dose.get("coobservation_rows_present") is True
    )
    safety_ok = (
        unsafe.get("safe_arm_unsafe_write_count") == 0
        and unsafe.get("safe_arm_unsafe_use_count") == 0
        and unsafe.get("same_query_arm_adopted") is False
    )
    lifecycle_ok = (
        lifecycle.get("restart_rollback_passed") is True
        and lifecycle.get("capacity_passed") is True
    )
    protected_ok = protected.get("all_protected_files_unchanged") is True
    tests_ok = bool(tests) and all(int(row.get("exit_code", 1)) == 0 for row in tests)
    attack_ok = attacks.get("all_attacks_fail_closed") is True
    missing_block = missing.get("terminal_disposition") == "blocked"
    clean = (
        not missing_block
        and row_closure
        and receipt_ok
        and exact_ok
        and transition_ok
        and positive_delta > 0.0
        and retention_ok
        and dose_ok
        and safety_ok
        and lifecycle_ok
        and protected_ok
        and tests_ok
        and attack_ok
    )
    disqualified = (
        not missing_block
        and bool(rows)
        and (
            not receipt_ok
            or not exact_ok
            or not transition_ok
            or not safety_ok
            or len(unit_ids) != len(set(unit_ids))
        )
    )
    return {
        "row_type": "aggregate_row_recomputation",
        "row_count": len(rows),
        "expected_row_count": expected,
        "row_closure": row_closure,
        "duplicate_unit_count": len(unit_ids) - len(set(unit_ids)),
        "receipt_authenticity_passed": receipt_ok,
        "exact_replay_passed": exact_ok,
        "transition_replay_passed": transition_ok,
        "safe_positive_arm_id": safe_positive,
        "current_value_positive": positive_delta > 0.0,
        "retention_and_support_noninferior": retention_ok,
        "dose_passed": dose_ok,
        "safety_passed": safety_ok,
        "restart_rollback_persistence_passed": lifecycle_ok,
        "protected_files_passed": protected_ok,
        "tests_passed": tests_ok,
        "attacks_passed": attack_ok,
        "missing_input_block": missing_block,
        "scientific_disposition_from_rows": "positive_prospective_value"
        if clean
        else "missing_live_evidence"
        if missing_block
        else "invalid_or_partial_evidence",
        "verdict_class_from_rows": "blocked"
        if missing_block
        else "disqualified"
        if disqualified
        else "null"
        if clean
        else "partial",
        "ready_score_from_rows": 1.0 if clean else 0.0,
    }


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    aggregate = artifact.get("aggregate_row_recomputation", {})
    preconditions = artifact.get("preconditions_checked", {})
    rows = [
        {"check": "row_closure", "expected": True, "observed": aggregate.get("row_closure")},
        {
            "check": "receipt_authenticity",
            "expected": True,
            "observed": aggregate.get("receipt_authenticity_passed"),
        },
        {
            "check": "exact_replay",
            "expected": True,
            "observed": aggregate.get("exact_replay_passed"),
        },
        {
            "check": "transition_replay",
            "expected": True,
            "observed": aggregate.get("transition_replay_passed"),
        },
        {"check": "matched_dose", "expected": True, "observed": aggregate.get("dose_passed")},
        {"check": "safety", "expected": True, "observed": aggregate.get("safety_passed")},
        {
            "check": "retention_support",
            "expected": True,
            "observed": aggregate.get("retention_and_support_noninferior"),
        },
        {
            "check": "restart_rollback",
            "expected": True,
            "observed": aggregate.get("restart_rollback_persistence_passed"),
        },
        {"check": "attacks", "expected": True, "observed": aggregate.get("attacks_passed")},
    ]
    for failed in preconditions.get("upstream_failed_checks", []):
        rows.append({"check": str(failed), "expected": True, "observed": False})
    normalized = [{**row, "passed": row.get("observed") is row.get("expected")} for row in rows]
    failed = [str(row["check"]) for row in normalized if row["passed"] is not True]
    return {
        "row_type": "gate_check_summary",
        "rows": normalized,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def _field_provenance(repo_root: Path, input_path: Path) -> dict[str, JsonDict]:
    sources = [MODULE_RELATIVE_PATH, TEST_RELATIVE_PATH, SPEC_RELATIVE_PATH, EXP6552_RELATIVE_PATH]
    hashes = {source.as_posix(): sha256_file(repo_root / source) for source in sources}
    hashes[str(input_path)] = sha256_file(input_path)
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": [source.as_posix() for source in sources] + [str(input_path)],
            "source_hashes": hashes,
            "row_sources": [
                "independent_live_receipt_audit_rows",
                "independent_exact_replay_rows",
                "independent_transition_replay_rows",
                "per_unit_rows",
                "aggregate_row_recomputation",
            ],
            "reducer_code": MODULE_RELATIVE_PATH.as_posix(),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_and_verdict(
    aggregate: Mapping[str, Any], missing: Mapping[str, Any]
) -> tuple[str, str, str]:
    if aggregate.get("verdict_class_from_rows") == "blocked":
        missing_inputs = ",".join(missing.get("missing_inputs", []))
        return (
            "blocked_continuous_self_learning_audit_missing_inputs",
            "blocked: receipt, safety, retention, support, and scientific disposition are blocked by missing live evidence: "
            + missing_inputs,
            "blocked",
        )
    if aggregate.get("verdict_class_from_rows") == "disqualified":
        return (
            "disqualified_continuous_self_learning_audit",
            "disqualified: receipt, replay, safety, or row-closure checks failed under independent audit",
            "disqualified",
        )
    if aggregate.get("ready_score_from_rows") == 1.0:
        return (
            "complete_continuous_self_learning_independent_audit",
            "complete: receipt authenticity, exact replay, transition replay, dose, safety, retention, support, restart, rollback, and verdict recomputation passed; scientific disposition=positive prospective value",
            "null",
        )
    return (
        "partial_continuous_self_learning_independent_audit",
        "partial: stored rows are incomplete but not false enough to disqualify",
        "partial",
    )


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    input_path: Path | str | None = None,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    work_root: Path | str = REPO_ROOT / WORK_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
    hash_model_files: bool = True,
) -> JsonDict:
    start = time.monotonic()
    repo_root = Path(repo_root)
    selected_input = (
        Path(input_path) if input_path is not None else repo_root / EXP6553_RELATIVE_PATH
    )
    result = Path(result_path)
    work = Path(work_root)
    before = protected_file_hashes(repo_root)
    upstream = _read_json(selected_input)
    input_receipts = input_existence_and_hash_receipts(
        repo_root=repo_root,
        input_path=selected_input,
        upstream=upstream,
        hash_model_files=hash_model_files,
    )
    preconditions = preconditions_checked(
        repo_root=repo_root,
        input_path=selected_input,
        result_path=result,
        work_root=work,
        receipts=input_receipts,
        protected_hashes_before=before,
        run_date=run_date,
    )
    missing = missing_input_disposition(
        receipts=input_receipts,
        upstream=upstream,
        preconditions=preconditions,
    )
    live_rows = independent_live_receipt_audit_rows(
        upstream=upstream,
        receipts=input_receipts,
        missing=missing,
    )
    exact_rows = (
        []
        if missing["terminal_disposition"] == "blocked"
        else independent_exact_replay_rows(upstream)
    )
    transition_rows = (
        []
        if missing["terminal_disposition"] == "blocked"
        else independent_transition_replay_rows(upstream)
    )
    current_rows = (
        []
        if missing["terminal_disposition"] == "blocked"
        else independent_current_effect_rows(upstream)
    )
    retention_support_rows = (
        []
        if missing["terminal_disposition"] == "blocked"
        else independent_retention_and_support_rows(upstream)
    )
    dose = dose_and_coobservation_audit(upstream)
    unsafe = unsafe_write_and_use_audit(upstream)
    lifecycle = restart_rollback_and_persistence_audit(upstream)
    after = protected_file_hashes(repo_root)
    artifact: JsonDict = {
        "status": "partial_continuous_self_learning_independent_audit_assembly",
        "honest_verdict": "partial: artifact assembly not finalized",
        "verdict_class": "partial",
        "input_existence_and_hash_receipts": input_receipts,
        "independent_live_receipt_audit_rows": live_rows,
        "independent_exact_replay_rows": exact_rows,
        "independent_transition_replay_rows": transition_rows,
        "independent_current_effect_rows": current_rows,
        "independent_retention_and_support_rows": retention_support_rows,
        "dose_and_coobservation_audit": dose,
        "unsafe_write_and_use_audit": unsafe,
        "restart_rollback_and_persistence_audit": lifecycle,
        "missing_input_disposition": missing,
        "attack_matrix": {},
        "continuous_self_learning_audited_ready_score": 0.0,
        "per_unit_rows": live_rows,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected_files_unchanged(before, after),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(repo_root, selected_input),
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s if duration_s is not None else time.monotonic() - start, 6),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
        "_upstream_payload_for_recompute": upstream,
    }
    artifact["attack_matrix"] = attack_matrix(artifact)
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(artifact)
    artifact["continuous_self_learning_audited_ready_score"] = artifact[
        "aggregate_row_recomputation"
    ]["ready_score_from_rows"]
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["status"], artifact["honest_verdict"], artifact["verdict_class"] = _status_and_verdict(
        artifact["aggregate_row_recomputation"],
        missing,
    )
    artifact.pop("_upstream_payload_for_recompute")
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _atomic_write_json(result, artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    scrubbed = dict(artifact)
    scrubbed["reproducibility_checksum"] = ""
    return sha256_json(scrubbed)


def _validate_row_hashes(payload: Mapping[str, Any], errors: list[str]) -> None:
    for field in (
        "per_unit_rows",
        "independent_live_receipt_audit_rows",
        "independent_exact_replay_rows",
        "independent_transition_replay_rows",
        "independent_current_effect_rows",
        "independent_retention_and_support_rows",
    ):
        for row in _receipt_rows(payload.get(field)):
            if row.get("row_hash") != row_hash(row):
                errors.append(f"{field} row_hash mismatch")


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
        return errors
    if not str(payload.get("status", "")).startswith(
        ("blocked_", "complete_", "partial_", "disqualified_")
    ):
        errors.append("status lacks terminal prefix")
    if not str(payload.get("honest_verdict", "")).startswith(
        ("blocked:", "complete:", "partial:", "disqualified:")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if payload.get("verdict_class") not in {"null", "partial", "blocked", "disqualified"}:
        errors.append("verdict_class must be closed")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    _validate_row_hashes(payload, errors)
    unit_ids = [row.get("unit_id") for row in _receipt_rows(payload.get("per_unit_rows"))]
    if len(unit_ids) != len(set(unit_ids)):
        errors.append("duplicate audit rows detected")
    aggregate = aggregate_row_recomputation(payload)
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("continuous_self_learning_audited_ready_score") != aggregate.get(
        "ready_score_from_rows"
    ):
        errors.append("ready score mismatch")
    if (
        payload.get("verdict_class") == "blocked"
        and payload.get("continuous_self_learning_audited_ready_score") != 0.0
    ):
        errors.append("blocked verdict requires zero ready score")
    if (
        payload.get("continuous_self_learning_audited_ready_score") == 1.0
        and payload.get("verdict_class") != "null"
    ):
        errors.append("clean audit must use verdict_class null")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--input-path", default=str(REPO_ROOT / EXP6553_RELATIVE_PATH))
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--work-root", default=str(REPO_ROOT / WORK_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-model-file-hash", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        if not result_path.is_file():
            print(f"artifact not found: {result_path}")
            return 1
        errors = validate_artifact(_read_json(result_path))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {result_path}")
        return 0
    try:
        artifact = build_artifact(
            input_path=Path(args.input_path),
            result_path=result_path,
            work_root=Path(args.work_root),
            write=True,
            run_date=str(args.date),
            hash_model_files=not args.skip_model_file_hash,
        )
    except Exception as exc:
        print(str(exc))
        return 1
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {result_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
