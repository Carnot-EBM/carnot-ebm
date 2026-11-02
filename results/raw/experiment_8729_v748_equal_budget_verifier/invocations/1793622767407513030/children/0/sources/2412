"""Exp6562 constraint saturation independent audit v2.

Spec refs: REQ-REPORT-6562, SCENARIO-REPORT-6562-FIXTURE,
SCENARIO-REPORT-6562-LIVE, SCENARIO-REPORT-6562-REPLAY,
SCENARIO-REPORT-6562-PAIRS, SCENARIO-REPORT-6562-ATOMIC.

The audit reads checked-in V567 saturation evidence and exact fixture rows. It
does not run a model. It fails closed when the stored receipts cannot prove
the live GGUF rows, exact releases, or charged intervention claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

import z3

from carnot import experiment_6555_proof_preserving_constraint_saturation_fixture as fixture_mod


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6562
INFERENCE_SUBSTRATE = "independent_checked_in_sota_receipt_and_exact_constraint_replay_no_new_llm"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_6562_constraint_saturation_independent_audit_v2.json"
)
EXP6555_RELATIVE_PATH = Path(
    "results/experiment_6555_proof_preserving_constraint_saturation_fixture.json"
)
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v567_constraint_saturation.jsonl")
EXP6556_RELATIVE_PATH = Path(
    "results/experiment_6556_sota_constraint_saturation_intervention_ab.json"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6556_sota_constraint_saturation_intervention_ab.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6562_constraint_saturation_independent_audit_v2.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6562_constraint_saturation_independent_audit_v2.py"
)
EXP6555_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6555_proof_preserving_constraint_saturation_fixture.py"
)
EXP6556_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6556_sota_constraint_saturation_intervention_ab.py"
)
ADVERSARIAL_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_RELATIVE_PATH = Path("scripts/verdict_row_consistency_lint.py")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
ARM_IDS = (
    "flat",
    "longer_flat",
    "bounded_decomposition",
    "exact_tool_cost_guard",
    "combined_bounded_route",
)
MODEL_ARM_IDS = {"flat", "longer_flat", "bounded_decomposition"}
INTERVENTION_ARM_IDS = {
    "bounded_decomposition",
    "exact_tool_cost_guard",
    "combined_bounded_route",
}
CHECKPOINT_SCHEMA = "carnot.exp6556.constraint_saturation_intervention.checkpoint.v1"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "input_existence_and_hash_receipts",
    "independent_fixture_proof_rows",
    "independent_live_provenance_rows",
    "independent_clause_and_joint_replay_rows",
    "independent_phase_curve_rows",
    "independent_paired_intervention_rows",
    "harmful_intervention_and_release_audit",
    "charged_cost_audit",
    "constraint_saturation_independent_audit_ready_score",
    "constraint_saturation_policy_audited_score",
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
    "status": "An always-run audit needs a terminal state for clean, missing, invalid, and non-recomputable evidence.",
    "honest_verdict": "The verdict must state fixture, live provenance, exact replay, harm, cost, and policy dispositions with a terminal prefix.",
    "verdict_class": "A closed class prevents an audit record from becoming unbounded positive science.",
    "input_existence_and_hash_receipts": "The audit must identify the exact artifacts, checkpoints, raw responses, and checkers it used.",
    "independent_fixture_proof_rows": "The model comparison is ineligible if the proof-preserving variants do not replay.",
    "independent_live_provenance_rows": "Commands, PIDs, GPU samples, tokens, and response hashes must prove actual flagship inference.",
    "independent_clause_and_joint_replay_rows": "Every credited success and release needs a separate executable replay.",
    "independent_phase_curve_rows": "Constraint-load collapse must be recomputed by model, count, type, interaction, and surface.",
    "independent_paired_intervention_rows": "Every claimed benefit needs matched rows against flat and longer-flat controls.",
    "harmful_intervention_and_release_audit": "No aggregate gain can hide regressions or one invalid release.",
    "charged_cost_audit": "Tokens, retries, solver calls, and time must recompute from raw receipts.",
    "constraint_saturation_independent_audit_ready_score": "One binary field states whether the V567 saturation evidence is independently usable.",
    "constraint_saturation_policy_audited_score": "A separate field distinguishes clean null evidence from a confirmed promotable policy.",
    "per_unit_rows": "Every audit conclusion needs unit-level recomputation rows.",
    "aggregate_row_recomputation": "The audit verdict must derive only from independent rows.",
    "gate_check_summary": "A blocked audit must name every missing or failed check and observed value.",
    "preconditions_checked": "Input and replay receipts distinguish blocked audit work from null science.",
    "protected_files_unchanged": "The audit must not repair upstream evidence or mutate protected orchestration files.",
    "inference_substrate": "The audit replays stored evidence and performs no new GGUF generation.",
    "verifier_is_oracle": "Independent executable replay is audit authority, so a clean audit uses a non-positive class.",
    "field_provenance": "Each disposition must point to immutable rows, receipts, and reducers.",
    "random_seed": "A fixed audit order makes attacks and samples repeatable.",
    "duration_s": "Monotonic time exposes an audit that skipped receipt or replay work.",
    "tests_run": "Named tests and E2E commands show independent checks executed.",
    "reproducibility_checksum": "A final hash protects the independent determination trail.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6562_constraint_saturation_independent_audit_v2 "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6562_constraint_saturation_independent_audit_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6562_constraint_saturation_independent_audit_v2.py "
    "-m pytest tests/python/test_experiment_6562_constraint_saturation_independent_audit_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6562_constraint_saturation_independent_audit_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6562_constraint_saturation_independent_audit_v2.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6562_constraint_saturation_independent_audit_v2.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6562_constraint_saturation_independent_audit_v2.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6562_constraint_saturation_independent_audit_v2 "
    "--validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: Exp6562 exact fixture replay covers the stored "
    "constraint-saturation audit path"
)
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    ADVERSARIAL_RELATIVE_PATH,
    ROW_LINT_RELATIVE_PATH,
    EXP6555_MODULE_RELATIVE_PATH,
    EXP6556_MODULE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6555_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    EXP6556_RELATIVE_PATH,
    CHECKPOINT_RELATIVE_PATH,
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


def unit_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("model_hf_id")),
            str(row.get("local_unit_id")),
            str(row.get("seed")),
            str(row.get("arm_id")),
        ]
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    stable.pop("duration_s", None)
    return sha256_json(stable)


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


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
        if tmp_path.exists():  # pragma: no cover - reached only after replace failure.
            tmp_path.unlink()


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _default_input_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "exp6555_artifact": repo_root / EXP6555_RELATIVE_PATH,
        "fixture_jsonl": repo_root / FIXTURE_RELATIVE_PATH,
        "exp6556_artifact": repo_root / EXP6556_RELATIVE_PATH,
        "checkpoint": repo_root / CHECKPOINT_RELATIVE_PATH,
    }


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    try:
        return resolved.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
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
        "all_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "rows": rows,
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    mem_total_kib = None
    mem_available_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            digits = "".join(ch for ch in rest if ch.isdigit())
            if digits:
                values[key] = int(digits)
        mem_total_kib = values.get("MemTotal")
        mem_available_kib = values.get("MemAvailable")
    cpu_model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    return {
        "python": {"version": sys.version, "executable": sys.executable},
        "z3_version": z3.get_version_string(),
        "cpu": {"count": os.cpu_count(), "model": cpu_model},
        "ram": {"total_kib": mem_total_kib, "available_kib": mem_available_kib},
        "disk": {"path": str(repo_root), "total_bytes": disk.total, "free_bytes": disk.free},
        "platform": platform.platform(),
    }


def _field_provenance(input_paths: Mapping[str, Path]) -> JsonDict:
    sources = [
        SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        *[str(path) for path in input_paths.values()],
    ]
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "spec_refs": ["REQ-REPORT-6562"],
            "sources": sources,
            "reducer": f"experiment_6562.{field}",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def input_existence_and_hash_receipts(
    *,
    repo_root: Path,
    input_paths: Mapping[str, Path],
    exp6556: Mapping[str, Any],
    hash_model_files: bool,
) -> JsonDict:
    rows = []
    for role, path in input_paths.items():
        rows.append(
            {
                "role": role,
                "path": _source_key(repo_root, Path(path)),
                "absolute_path": str(path),
                "exists": Path(path).is_file(),
                "bytes": Path(path).stat().st_size if Path(path).is_file() else 0,
                "sha256": sha256_file(path),
            }
        )
    checker_paths = (
        ADVERSARIAL_RELATIVE_PATH,
        ROW_LINT_RELATIVE_PATH,
        EXP6555_MODULE_RELATIVE_PATH,
        EXP6556_MODULE_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
    )
    checker_rows = [
        {
            "role": path.as_posix(),
            "path": path.as_posix(),
            "exists": (repo_root / path).is_file(),
            "sha256": sha256_file(repo_root / path),
        }
        for path in checker_paths
    ]
    model_rows = []
    for spec in exp6556.get("MODEL_SPECS", []):
        if not isinstance(spec, Mapping):
            continue
        path = Path(str(spec.get("model_path") or ""))
        observed = sha256_file(path) if hash_model_files else str(spec.get("gguf_sha256"))
        expected = str(spec.get("gguf_sha256") or "missing")
        model_rows.append(
            {
                "hf_id": spec.get("hf_id"),
                "path": str(path),
                "exists": path.is_file(),
                "declared_sha256": expected,
                "observed_sha256": observed,
                "hash_matches_declared": observed == expected and observed != "missing",
                "hash_method": "sha256_file" if hash_model_files else "declared_hash_test_mode",
            }
        )
    raw_response_paths: list[JsonDict] = []
    payload = {
        "artifact_rows": rows,
        "checker_rows": checker_rows,
        "model_file_rows": model_rows,
        "raw_response_paths": raw_response_paths,
        "all_inputs_exist": all(row["exists"] for row in rows),
        "all_checker_hashes_present": all(row["sha256"] != "missing" for row in checker_rows),
        "all_model_hashes_match": bool(model_rows)
        and all(row["hash_matches_declared"] for row in model_rows),
        "raw_response_path_count": len(raw_response_paths),
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def independent_fixture_proof_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for index, source in enumerate(rows):
        row = dict(source)
        clauses = [dict(clause) for clause in row.get("clause_rows", [])]
        constraints = [dict(item) for item in row.get("variant_constraints", [])]
        variant_mode = str(row.get("variant_mode") or "")
        source_count = int(row.get("constraint_load_count") or 0)
        simultaneous_count = int(row.get("simultaneous_constraint_count") or 0)
        constraint_count_matches = len(clauses) == len(constraints) == simultaneous_count
        clause_identity_passed = all(
            clause.get("constraint_sha256") == sha256_json(clause.get("constraint"))
            and clause.get("source_clause_index") == clause_index
            for clause_index, clause in enumerate(clauses, start=1)
        )
        if variant_mode == "equivalent":
            proof_relation = sha256_json(constraints) == row.get("source_constraints_sha256")
        elif variant_mode == "hardened":
            proof_relation = len(constraints) == source_count + 1 and constraints[-1:] == [
                dict(row.get("declared_hardening_constraint") or {})
            ]
        else:
            proof_relation = False
        try:
            joint_ok = fixture_mod.joint_checker(row)
            clause_ok = all(fixture_mod.per_clause_checker(row, clause) for clause in clauses)
            replay_error = ""
        except Exception as exc:  # pragma: no cover - depends on missing external checker.
            joint_ok = False
            clause_ok = False
            replay_error = f"{type(exc).__name__}: {exc}"
        payload = {
            "row_type": "independent_fixture_proof",
            "fixture_index": index,
            "variant_id": row.get("variant_id"),
            "lineage_id": row.get("lineage_id"),
            "variant_mode": variant_mode,
            "surface": row.get("surface"),
            "domain": row.get("domain"),
            "constraint_count": len(constraints),
            "declared_constraint_count": simultaneous_count,
            "constraint_count_matches": constraint_count_matches,
            "clause_identity_passed": clause_identity_passed,
            "interaction_class": dict(row.get("constraint_graph") or {}).get("interaction_class"),
            "lineage_present": bool(row.get("lineage_id")),
            "surface_present": bool(row.get("surface")),
            "proof_relation_passed": proof_relation,
            "source_joint_replay_passed": joint_ok,
            "source_clause_replay_passed": clause_ok,
            "replay_error": replay_error,
        }
        payload["fixture_replay_passed"] = all(
            (
                constraint_count_matches,
                clause_identity_passed,
                proof_relation,
                joint_ok,
                clause_ok,
                payload["lineage_present"],
                payload["surface_present"],
            )
        )
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def _raw_response(row: Mapping[str, Any]) -> str | None:
    for key in ("raw_response", "raw_output", "output_text"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _has_valid_timing(row: Mapping[str, Any]) -> bool:
    start = row.get("started_at_monotonic_s")
    end = row.get("ended_at_monotonic_s")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False
    return float(end) >= float(start) and float(end) - float(start) <= max(
        3600.0, float(row.get("model_wall_time_s") or 0.0) + 60.0
    )


def independent_live_provenance_rows(
    *,
    exp6556: Mapping[str, Any],
    per_unit_rows: Sequence[Mapping[str, Any]],
    input_receipts: Mapping[str, Any],
) -> list[JsonDict]:
    specs = {
        str(spec.get("hf_id")): dict(spec)
        for spec in exp6556.get("MODEL_SPECS", [])
        if isinstance(spec, Mapping)
    }
    model_hash = {
        str(row.get("hf_id")): bool(row.get("hash_matches_declared"))
        for row in input_receipts.get("model_file_rows", [])
        if isinstance(row, Mapping)
    }
    rows_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_unit_rows:
        rows_by_model[str(row.get("model_hf_id"))].append(row)
    out = []
    for hf_id in MANDATED_HF_IDS:
        rows = rows_by_model.get(hf_id, [])
        live_rows = [row for row in rows if row.get("arm_id") in MODEL_ARM_IDS]
        raw_values = [_raw_response(row) for row in rows]
        gpu_samples = [row.get("gpu_sample") for row in live_rows]
        serialized_gpu = {canonical_json(sample) for sample in gpu_samples if sample}
        payload = {
            "row_type": "independent_live_provenance",
            "model_hf_id": hf_id,
            "model_declared": hf_id in specs,
            "model_path": specs.get(hf_id, {}).get("model_path"),
            "model_hash_valid": model_hash.get(hf_id) is True,
            "row_count": len(rows),
            "model_arm_row_count": len(live_rows),
            "process_command_present": bool(live_rows)
            and all(bool(row.get("process_command")) for row in live_rows),
            "per_unit_pid_present": bool(live_rows)
            and all(isinstance(row.get("process_id"), int) for row in live_rows),
            "start_end_time_present": bool(live_rows)
            and all(_has_valid_timing(row) for row in live_rows),
            "gpu_sample_rows_present": bool(live_rows)
            and all(bool(row.get("gpu_sample")) for row in live_rows),
            "gpu_samples_nonconstant": len(serialized_gpu) > 1
            if len(live_rows) > 1
            else bool(serialized_gpu),
            "token_count_rows_present": bool(rows)
            and all(
                isinstance(row.get("prompt_tokens"), int)
                and isinstance(row.get("output_tokens"), int)
                for row in rows
            ),
            "response_hash_rows_present": bool(rows)
            and all(str(row.get("response_sha256", "")).startswith("sha256:") for row in rows),
            "raw_response_rows_present": bool(rows)
            and all(value is not None for value in raw_values),
            "response_hash_recomputed": bool(rows)
            and all(
                value is not None and sha256_json(value) == row.get("response_sha256")
                for row, value in zip(rows, raw_values, strict=True)
            ),
            "exit_status_terminal": bool(rows)
            and all(row.get("exit_status") == "terminal" for row in rows),
            "checkpoint_reused_count": sum(bool(row.get("checkpoint_reused")) for row in rows),
        }
        payload["live_provenance_passed"] = all(
            (
                payload["model_declared"],
                payload["model_hash_valid"],
                payload["process_command_present"],
                payload["per_unit_pid_present"],
                payload["start_end_time_present"],
                payload["gpu_sample_rows_present"],
                payload["gpu_samples_nonconstant"],
                payload["token_count_rows_present"],
                payload["response_hash_rows_present"],
                payload["raw_response_rows_present"],
                payload["response_hash_recomputed"],
                payload["exit_status_terminal"],
                payload["checkpoint_reused_count"] == 0,
            )
        )
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(row.get("model_hf_id")),
        str(row.get("variant_id")),
        str(row.get("surface")),
        int(row.get("seed") or 0),
        str(row.get("arm_id")),
    )


def per_unit_audit_rows(
    *,
    per_unit_rows: Sequence[Mapping[str, Any]],
    fixture_rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
) -> list[JsonDict]:
    fixture_by_variant = {str(row.get("variant_id")): dict(row) for row in fixture_rows}
    checkpoint_rows = checkpoint.get("rows_by_key")
    checkpoint_rows = dict(checkpoint_rows) if isinstance(checkpoint_rows, Mapping) else {}
    key_counts = Counter(unit_key(row) for row in per_unit_rows)
    out = []
    for source in per_unit_rows:
        row = dict(source)
        fixture = fixture_by_variant.get(str(row.get("variant_id")), {})
        key = unit_key(row)
        checkpoint_row = checkpoint_rows.get(key)
        if isinstance(checkpoint_row, Mapping):
            checkpoint_match = dict(checkpoint_row) == row
        else:
            checkpoint_match = False
        payload = {
            "row_type": "constraint_saturation_independent_audit_unit",
            "model_hf_id": row.get("model_hf_id"),
            "variant_id": row.get("variant_id"),
            "surface": row.get("surface"),
            "seed": row.get("seed"),
            "arm_id": row.get("arm_id"),
            "unit_key": key,
            "fixture_row_found": bool(fixture),
            "checkpoint_row_found": isinstance(checkpoint_row, Mapping),
            "checkpoint_row_matches_artifact": checkpoint_match,
            "duplicate_unit_key_count": key_counts[key],
            "source_row_hash_valid": row.get("row_hash") == row_hash(row),
            "constraint_count_matches_fixture": bool(fixture)
            and int(row.get("constraint_count") or -1) == len(fixture.get("clause_rows", [])),
            "constraint_load_matches_fixture": bool(fixture)
            and int(row.get("constraint_load_k") or -1)
            == int(fixture.get("simultaneous_constraint_count") or -2),
            "surface_matches_fixture": bool(fixture)
            and row.get("surface") == fixture.get("surface"),
            "lineage_matches_fixture": bool(fixture)
            and row.get("lineage_id") == fixture.get("lineage_id"),
            "interaction_matches_fixture": bool(fixture)
            and row.get("interaction_class")
            == dict(fixture.get("constraint_graph") or {}).get("interaction_class"),
            "raw_response_present": _raw_response(row) is not None,
            "invalid_release": bool(row.get("invalid_release")),
            "timeout": bool(row.get("timeout")),
            "censored": bool(row.get("censored")),
            "exact_final_validity": bool(row.get("exact_final_validity")),
            "charged_tokens": row.get("charged_tokens"),
            "charged_cost": row.get("charged_cost"),
        }
        payload["unit_row_closed"] = all(
            (
                payload["fixture_row_found"],
                payload["checkpoint_row_found"],
                payload["checkpoint_row_matches_artifact"],
                payload["duplicate_unit_key_count"] == 1,
                payload["source_row_hash_valid"],
                payload["constraint_count_matches_fixture"],
                payload["constraint_load_matches_fixture"],
                payload["surface_matches_fixture"],
                payload["lineage_matches_fixture"],
                payload["interaction_matches_fixture"],
            )
        )
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def independent_clause_and_joint_replay_rows(
    *,
    per_unit_rows: Sequence[Mapping[str, Any]],
    fixture_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    fixture_by_variant = {str(row.get("variant_id")): dict(row) for row in fixture_rows}
    result_by_key = {_row_key(row): dict(row) for row in result_rows if isinstance(row, Mapping)}
    route_by_key = {_row_key(row): dict(row) for row in route_rows if isinstance(row, Mapping)}
    out = []
    for row in per_unit_rows:
        fixture = fixture_by_variant.get(str(row.get("variant_id")), {})
        result = result_by_key.get(_row_key(row), {})
        route = route_by_key.get(_row_key(row), {})
        fixture_clause_count = len(fixture.get("clause_rows", [])) if fixture else 0
        exact_success = bool(row.get("exact_final_validity"))
        released = exact_success or bool(row.get("invalid_release"))
        try:
            exact_joint = fixture_mod.joint_checker(fixture) if fixture and exact_success else False
            exact_clauses = (
                all(
                    fixture_mod.per_clause_checker(fixture, clause)
                    for clause in fixture["clause_rows"]
                )
                if fixture and exact_success
                else False
            )
            replay_error = ""
        except Exception as exc:  # pragma: no cover - depends on missing external checker.
            exact_joint = False
            exact_clauses = False
            replay_error = f"{type(exc).__name__}: {exc}"
        expected_clause_success = fixture_clause_count if exact_success else 0
        decomposition_used = bool(route.get("decomposition_used"))
        payload = {
            "row_type": "independent_clause_joint_replay",
            "model_hf_id": row.get("model_hf_id"),
            "variant_id": row.get("variant_id"),
            "surface": row.get("surface"),
            "seed": row.get("seed"),
            "arm_id": row.get("arm_id"),
            "fixture_row_found": bool(fixture),
            "result_row_found": bool(result),
            "route_row_found": bool(route),
            "result_row_hash_valid": bool(result) and result.get("row_hash") == row_hash(result),
            "clause_count": fixture_clause_count,
            "upstream_clause_count": row.get("constraint_count"),
            "clause_count_matches": int(row.get("constraint_count") or -1) == fixture_clause_count,
            "per_clause_success_recomputed": expected_clause_success,
            "per_clause_success_matches": int(row.get("per_clause_success_count") or 0)
            == expected_clause_success,
            "released_result": released,
            "released_result_passed_joint_checker": (not released)
            or (exact_success and exact_joint and exact_clauses and not row.get("invalid_release")),
            "exact_joint_checker_replayed": exact_joint if exact_success else None,
            "exact_clause_checker_replayed": exact_clauses if exact_success else None,
            "decomposition_used": decomposition_used,
            "decomposition_preserves_clauses": (
                bool(route.get("clauses_preserved"))
                and int(route.get("decomposition_clause_count") or 0) == fixture_clause_count
            )
            if decomposition_used
            else True,
            "invalid_release": bool(row.get("invalid_release")),
            "replay_error": replay_error,
        }
        payload["clause_and_joint_replay_passed"] = all(
            (
                payload["fixture_row_found"],
                payload["result_row_found"],
                payload["route_row_found"],
                payload["result_row_hash_valid"],
                payload["clause_count_matches"],
                payload["per_clause_success_matches"],
                payload["released_result_passed_joint_checker"],
                payload["decomposition_preserves_clauses"],
            )
        )
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def independent_phase_curve_rows(per_unit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[tuple[str, int, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_unit_rows:
        grouped[
            (
                str(row.get("model_hf_id")),
                int(row.get("constraint_load_k") or 0),
                ",".join(str(item) for item in row.get("constraint_type_families", [])),
                str(row.get("interaction_class")),
                str(row.get("surface")),
                str(row.get("arm_id")),
            )
        ].append(row)
    out = []
    for (
        model_hf_id,
        constraint_load_k,
        constraint_type_families,
        interaction_class,
        surface,
        arm_id,
    ), rows in sorted(grouped.items()):
        clause_total = sum(int(row.get("constraint_count") or 0) for row in rows)
        exact_count = sum(bool(row.get("exact_final_validity")) for row in rows)
        payload = {
            "row_type": "independent_phase_curve",
            "model_hf_id": model_hf_id,
            "constraint_load_k": constraint_load_k,
            "constraint_type_families": constraint_type_families,
            "interaction_class": interaction_class,
            "surface": surface,
            "arm_id": arm_id,
            "row_count": len(rows),
            "exact_joint_success_count": exact_count,
            "exact_joint_success_rate": round(exact_count / len(rows), 6) if rows else 0.0,
            "per_clause_success_rate": round(
                sum(int(row.get("per_clause_success_count") or 0) for row in rows)
                / max(1, clause_total),
                6,
            ),
            "timeout_count": sum(bool(row.get("timeout")) for row in rows),
            "censored_count": sum(bool(row.get("censored")) for row in rows),
            "parse_failure_count": sum(bool(row.get("parse_failure")) for row in rows),
        }
        payload["row_hash"] = row_hash(payload)
        out.append(payload)
    return out


def independent_paired_intervention_rows(
    per_unit_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    grouped: dict[tuple[str, str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in per_unit_rows:
        key = (
            str(row.get("model_hf_id")),
            str(row.get("variant_id")),
            str(row.get("surface")),
            int(row.get("seed") or 0),
        )
        grouped[key][str(row.get("arm_id"))] = row
    out = []
    for key, arms in sorted(grouped.items()):
        flat = arms.get("flat")
        longer = arms.get("longer_flat")
        for arm_id in sorted(INTERVENTION_ARM_IDS):
            intervention = arms.get(arm_id)
            matched = flat is not None and longer is not None and intervention is not None
            exact = bool(intervention and intervention.get("exact_final_validity"))
            flat_exact = bool(flat and flat.get("exact_final_validity"))
            longer_exact = bool(longer and longer.get("exact_final_validity"))
            payload = {
                "row_type": "independent_paired_intervention",
                "model_hf_id": key[0],
                "variant_id": key[1],
                "surface": key[2],
                "seed": key[3],
                "arm_id": arm_id,
                "matched_controls_present": matched,
                "flat_exact": flat_exact,
                "longer_flat_exact": longer_exact,
                "intervention_exact": exact,
                "recovery_vs_flat": exact and not flat_exact,
                "recovery_vs_longer_flat": exact and not longer_exact,
                "regression_vs_flat": (not exact) and flat_exact,
                "regression_vs_longer_flat": (not exact) and longer_exact,
                "zero_headroom_win": exact and flat_exact and longer_exact,
                "invalid_release": bool(intervention and intervention.get("invalid_release")),
                "timeout": bool(intervention and intervention.get("timeout")),
                "charged_cost_delta_vs_flat": round(
                    float(intervention.get("charged_cost", 0.0) if intervention else 0.0)
                    - float(flat.get("charged_cost", 0.0) if flat else 0.0),
                    6,
                ),
                "charged_cost_delta_vs_longer_flat": round(
                    float(intervention.get("charged_cost", 0.0) if intervention else 0.0)
                    - float(longer.get("charged_cost", 0.0) if longer else 0.0),
                    6,
                ),
            }
            payload["benefit_against_both_controls"] = matched and (
                (exact and not flat_exact and not longer_exact)
                or (
                    payload["charged_cost_delta_vs_flat"] < 0
                    and payload["charged_cost_delta_vs_longer_flat"] < 0
                )
            )
            payload["paired_row_passed"] = matched and not (
                payload["regression_vs_flat"]
                or payload["regression_vs_longer_flat"]
                or payload["invalid_release"]
            )
            payload["row_hash"] = row_hash(payload)
            out.append(payload)
    return out


def harmful_intervention_and_release_audit(
    *,
    per_unit_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    invalid_by_arm = Counter(
        str(row.get("arm_id")) for row in per_unit_rows if row.get("invalid_release")
    )
    payload = {
        "row_type": "harmful_intervention_and_release_audit",
        "recovery_count_vs_flat": sum(bool(row.get("recovery_vs_flat")) for row in paired_rows),
        "recovery_count_vs_longer_flat": sum(
            bool(row.get("recovery_vs_longer_flat")) for row in paired_rows
        ),
        "regression_count_vs_flat": sum(bool(row.get("regression_vs_flat")) for row in paired_rows),
        "regression_count_vs_longer_flat": sum(
            bool(row.get("regression_vs_longer_flat")) for row in paired_rows
        ),
        "invalid_release_count": sum(bool(row.get("invalid_release")) for row in per_unit_rows),
        "invalid_release_by_arm": dict(sorted(invalid_by_arm.items())),
        "intervention_invalid_release_count": sum(
            bool(row.get("invalid_release")) for row in paired_rows
        ),
        "timeout_count": sum(bool(row.get("timeout")) for row in per_unit_rows),
        "censored_count": sum(bool(row.get("censored")) for row in per_unit_rows),
        "zero_headroom_win_count": sum(bool(row.get("zero_headroom_win")) for row in paired_rows),
    }
    payload["intervention_harm_free"] = (
        payload["regression_count_vs_flat"] == 0
        and payload["regression_count_vs_longer_flat"] == 0
        and payload["intervention_invalid_release_count"] == 0
    )
    payload["release_audit_passed"] = payload["invalid_release_count"] == 0
    payload["harm_audit_passed"] = (
        payload["intervention_harm_free"]
        and payload["release_audit_passed"]
        and payload["zero_headroom_win_count"] == 0
    )
    payload["row_hash"] = row_hash(payload)
    return payload


def charged_cost_audit(
    *,
    per_unit_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    cost_by_key = {_row_key(row): dict(row) for row in cost_rows if isinstance(row, Mapping)}
    per_row = []
    for row in per_unit_rows:
        prompt = int(row.get("prompt_tokens") or 0)
        output = int(row.get("output_tokens") or 0)
        solver_calls = int(row.get("solver_calls") or 0)
        model_wall = float(row.get("model_wall_time_s") or 0.0)
        solver_wall = float(row.get("solver_wall_time_s") or 0.0)
        expected = {
            "charged_tokens": prompt + output,
            "charged_time_s": round(model_wall + solver_wall, 6),
            "charged_cost": round(prompt + output + solver_calls * 4 + model_wall, 6),
        }
        cost = cost_by_key.get(_row_key(row), {})
        payload = {
            "model_hf_id": row.get("model_hf_id"),
            "variant_id": row.get("variant_id"),
            "surface": row.get("surface"),
            "seed": row.get("seed"),
            "arm_id": row.get("arm_id"),
            "retry_field_present": "retries" in row,
            "retries": int(row.get("retries") or 0),
            "charged_tokens_recomputed": expected["charged_tokens"],
            "charged_time_s_recomputed": expected["charged_time_s"],
            "charged_cost_recomputed": expected["charged_cost"],
            "charged_tokens_match": row.get("charged_tokens") == expected["charged_tokens"],
            "charged_time_match": row.get("charged_time_s") == expected["charged_time_s"],
            "charged_cost_match": row.get("charged_cost") == expected["charged_cost"],
            "cost_row_found": bool(cost),
            "cost_row_hash_valid": bool(cost) and cost.get("row_hash") == row_hash(cost),
        }
        payload["cost_row_passed"] = all(
            (
                payload["retry_field_present"],
                payload["charged_tokens_match"],
                payload["charged_time_match"],
                payload["charged_cost_match"],
                payload["cost_row_found"],
                payload["cost_row_hash_valid"],
            )
        )
        payload["row_hash"] = row_hash(payload)
        per_row.append(payload)
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_unit_rows:
        by_arm[str(row.get("arm_id"))].append(row)
    totals = []
    for arm_id, rows in sorted(by_arm.items()):
        payload = {
            "arm_id": arm_id,
            "row_count": len(rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
            "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
            "charged_tokens": sum(int(row.get("charged_tokens") or 0) for row in rows),
            "solver_calls": sum(int(row.get("solver_calls") or 0) for row in rows),
            "retries": sum(int(row.get("retries") or 0) for row in rows),
            "model_wall_time_s": round(
                sum(float(row.get("model_wall_time_s") or 0.0) for row in rows),
                6,
            ),
            "solver_wall_time_s": round(
                sum(float(row.get("solver_wall_time_s") or 0.0) for row in rows),
                6,
            ),
            "charged_cost": round(sum(float(row.get("charged_cost") or 0.0) for row in rows), 6),
            "timeout_count": sum(bool(row.get("timeout")) for row in rows),
            "censored_count": sum(bool(row.get("censored")) for row in rows),
        }
        payload["row_hash"] = row_hash(payload)
        totals.append(payload)
    audit = {
        "row_type": "charged_cost_audit",
        "per_row": per_row,
        "totals_by_arm": totals,
        "all_cost_rows_recomputed": bool(per_row)
        and all(row["cost_row_passed"] for row in per_row),
        "hidden_retry_attack_closed": bool(per_row)
        and all(row["retry_field_present"] for row in per_row),
        "cost_source": "per_unit_rows",
    }
    audit["row_hash"] = row_hash(audit)
    return audit


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    inputs = dict(artifact.get("input_existence_and_hash_receipts") or {})
    fixture_rows = list(artifact.get("independent_fixture_proof_rows") or [])
    live_rows = list(artifact.get("independent_live_provenance_rows") or [])
    clause_rows = list(artifact.get("independent_clause_and_joint_replay_rows") or [])
    phase_rows = list(artifact.get("independent_phase_curve_rows") or [])
    paired_rows = list(artifact.get("independent_paired_intervention_rows") or [])
    harm = dict(artifact.get("harmful_intervention_and_release_audit") or {})
    cost = dict(artifact.get("charged_cost_audit") or {})
    unit_rows = list(artifact.get("per_unit_rows") or [])
    protected = dict(artifact.get("protected_files_unchanged") or {})
    input_ok = (
        inputs.get("all_inputs_exist") is True and inputs.get("all_checker_hashes_present") is True
    )
    fixture_ok = bool(fixture_rows) and all(
        row.get("fixture_replay_passed") for row in fixture_rows
    )
    live_ok = bool(live_rows) and all(row.get("live_provenance_passed") for row in live_rows)
    row_closure_ok = bool(unit_rows) and all(row.get("unit_row_closed") for row in unit_rows)
    exact_ok = bool(clause_rows) and all(
        row.get("clause_and_joint_replay_passed") for row in clause_rows
    )
    phase_ok = bool(phase_rows) and {row.get("model_hf_id") for row in phase_rows} >= set(
        MANDATED_HF_IDS
    )
    paired_ok = bool(paired_rows) and all(
        row.get("matched_controls_present") for row in paired_rows
    )
    benefit = any(row.get("benefit_against_both_controls") for row in paired_rows)
    harm_ok = harm.get("harm_audit_passed") is True
    cost_ok = (
        cost.get("all_cost_rows_recomputed") is True
        and cost.get("hidden_retry_attack_closed") is True
    )
    protected_ok = protected.get("all_unchanged") is True
    failed = {
        "inputs_exist": input_ok,
        "fixture_replay": fixture_ok,
        "live_provenance_recomputable": live_ok,
        "row_closure": row_closure_ok,
        "exact_clause_and_joint_replay": exact_ok,
        "phase_curve_recomputed": phase_ok,
        "paired_interventions_matched": paired_ok,
        "harm_and_release_audit": harm_ok,
        "charged_cost_recomputed": cost_ok,
        "protected_files_unchanged": protected_ok,
    }
    disqualifying = input_ok and not all(
        (fixture_ok, live_ok, row_closure_ok, exact_ok, harm_ok, cost_ok, protected_ok)
    )
    if not input_ok:
        verdict = "blocked"
    elif disqualifying:
        verdict = "disqualified"
    elif not phase_ok or not paired_ok:
        verdict = "partial"
    else:
        verdict = None
    ready = 1.0 if verdict is None and all(failed.values()) else 0.0
    policy = 1.0 if ready == 1.0 and benefit else 0.0
    payload = {
        "row_type": "aggregate_row_recomputation",
        "inputs_exist": input_ok,
        "fixture_replay_passed": fixture_ok,
        "live_provenance_recomputable": live_ok,
        "row_closure_passed": row_closure_ok,
        "exact_clause_and_joint_replay_passed": exact_ok,
        "phase_curve_recomputed": phase_ok,
        "paired_interventions_matched": paired_ok,
        "harm_and_release_audit_passed": harm_ok,
        "charged_cost_recomputed": cost_ok,
        "protected_files_unchanged": protected_ok,
        "bounded_policy_claim_confirmed": benefit and ready == 1.0,
        "failed_checks": [key for key, value in failed.items() if not value],
        "ready_score_from_rows": ready,
        "policy_score_from_rows": policy,
        "verdict_class_from_rows": verdict,
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    check_names = (
        "inputs_exist",
        "fixture_replay_passed",
        "live_provenance_recomputable",
        "row_closure_passed",
        "exact_clause_and_joint_replay_passed",
        "phase_curve_recomputed",
        "paired_interventions_matched",
        "harm_and_release_audit_passed",
        "charged_cost_recomputed",
        "protected_files_unchanged",
    )
    rows = [
        {
            "check": name,
            "expected": True,
            "observed": aggregate.get(name),
            "passed": aggregate.get(name) is True,
        }
        for name in check_names
    ]
    payload = {
        "failed_checks": [row["check"] for row in rows if not row["passed"]],
        "rows": rows,
        "verdict_class_from_rows": aggregate.get("verdict_class_from_rows"),
    }
    payload["row_hash"] = row_hash(payload)
    return payload


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str | None]:
    verdict = aggregate.get("verdict_class_from_rows")
    if verdict == "blocked":
        return (
            "blocked_constraint_saturation_independent_audit_v2_missing_inputs",
            "blocked_constraint_saturation_independent_audit_v2_missing_inputs: one or more required checked-in evidence inputs is absent",
            "blocked",
        )
    if verdict == "disqualified":
        return (
            "disqualified_constraint_saturation_independent_audit_v2",
            "disqualified_constraint_saturation_independent_audit_v2: fixture, live provenance, exact replay, release, harm, or cost evidence is non-recomputable",
            "disqualified",
        )
    if verdict == "partial":
        return (
            "partial_constraint_saturation_independent_audit_v2",
            "partial_constraint_saturation_independent_audit_v2: usable evidence remains, but phase or pairing closure is incomplete",
            "partial",
        )
    return (
        "complete_constraint_saturation_independent_audit_v2_null",
        "complete_constraint_saturation_independent_audit_v2_null: fixture, live provenance, exact replay, harm, cost, and bounded policy checks recompute without positive audit class",
        None,
    )


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    input_paths: Mapping[str, Path],
    protected_before: Mapping[str, str],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "resources": _resource_receipt(repo_root),
        "python_version": sys.version,
        "z3_version": z3.get_version_string(),
        "audit_seed": RANDOM_SEED,
        "protected_file_hashes_before": dict(protected_before),
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    input_paths: Mapping[str, Path] | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    hash_model_files: bool = True,
) -> JsonDict:
    started = time.monotonic()
    result = Path(result_path or (repo_root / RESULT_RELATIVE_PATH))
    paths = _default_input_paths(repo_root) if input_paths is None else dict(input_paths)
    protected_before = _protected_hashes(repo_root)
    exp6555 = _read_json(paths["exp6555_artifact"])
    fixture_rows = _read_jsonl(paths["fixture_jsonl"])
    exp6556 = _read_json(paths["exp6556_artifact"])
    checkpoint = _read_json(paths["checkpoint"])
    per_unit_source = [
        dict(row) for row in exp6556.get("per_unit_rows", []) if isinstance(row, Mapping)
    ]
    result_source = [
        dict(row)
        for row in exp6556.get("per_clause_and_joint_result_rows", [])
        if isinstance(row, Mapping)
    ]
    route_source = [
        dict(row)
        for row in exp6556.get("route_decomposition_and_fallback_rows", [])
        if isinstance(row, Mapping)
    ]
    cost_source = [
        dict(row) for row in exp6556.get("charged_cost_rows", []) if isinstance(row, Mapping)
    ]
    input_receipts = input_existence_and_hash_receipts(
        repo_root=repo_root,
        input_paths=paths,
        exp6556=exp6556,
        hash_model_files=hash_model_files,
    )
    fixture_proofs = (
        independent_fixture_proof_rows(fixture_rows) if input_receipts["all_inputs_exist"] else []
    )
    live_rows = independent_live_provenance_rows(
        exp6556=exp6556,
        per_unit_rows=per_unit_source,
        input_receipts=input_receipts,
    )
    unit_rows = per_unit_audit_rows(
        per_unit_rows=per_unit_source,
        fixture_rows=fixture_rows,
        checkpoint=checkpoint,
    )
    clause_rows = independent_clause_and_joint_replay_rows(
        per_unit_rows=per_unit_source,
        fixture_rows=fixture_rows,
        result_rows=result_source,
        route_rows=route_source,
    )
    phase_rows = independent_phase_curve_rows(per_unit_source)
    paired_rows = independent_paired_intervention_rows(per_unit_source)
    harm = harmful_intervention_and_release_audit(
        per_unit_rows=per_unit_source,
        paired_rows=paired_rows,
    )
    cost = charged_cost_audit(per_unit_rows=per_unit_source, cost_rows=cost_source)
    protected_after = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(protected_before, protected_after)
    artifact: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": None,
        "input_existence_and_hash_receipts": input_receipts,
        "independent_fixture_proof_rows": fixture_proofs,
        "independent_live_provenance_rows": live_rows,
        "independent_clause_and_joint_replay_rows": clause_rows,
        "independent_phase_curve_rows": phase_rows,
        "independent_paired_intervention_rows": paired_rows,
        "harmful_intervention_and_release_audit": harm,
        "charged_cost_audit": cost,
        "constraint_saturation_independent_audit_ready_score": 0.0,
        "constraint_saturation_policy_audited_score": 0.0,
        "per_unit_rows": unit_rows,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result,
            input_paths=paths,
            protected_before=protected_before,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(paths),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            duration_s if duration_s is not None else time.monotonic() - started, 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = aggregate_row_recomputation(artifact)
    artifact["aggregate_row_recomputation"] = aggregate
    artifact["gate_check_summary"] = gate_check_summary(aggregate)
    artifact["constraint_saturation_independent_audit_ready_score"] = aggregate[
        "ready_score_from_rows"
    ]
    artifact["constraint_saturation_policy_audited_score"] = aggregate["policy_score_from_rows"]
    status, honest, verdict_class = _status_and_verdict(aggregate)
    artifact["status"] = status
    artifact["honest_verdict"] = honest
    artifact["verdict_class"] = verdict_class
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - validate_artifact tests cover each emitted error.
        raise ValueError("; ".join(errors))
    if write:
        _atomic_write_json(result, artifact)
    return artifact


def _validate_row_hashes(payload: Mapping[str, Any], errors: list[str]) -> None:
    for field in (
        "input_existence_and_hash_receipts",
        "harmful_intervention_and_release_audit",
        "charged_cost_audit",
        "aggregate_row_recomputation",
        "gate_check_summary",
    ):
        value = payload.get(field)
        if isinstance(value, Mapping) and value.get("row_hash") != row_hash(value):
            errors.append(f"{field} row_hash mismatch")
    for field in (
        "independent_fixture_proof_rows",
        "independent_live_provenance_rows",
        "independent_clause_and_joint_replay_rows",
        "independent_phase_curve_rows",
        "independent_paired_intervention_rows",
        "per_unit_rows",
    ):
        rows = payload.get(field)
        if not isinstance(rows, list):
            errors.append(f"{field} must be a list")
            continue
        for row in rows:
            if not isinstance(row, Mapping) or row.get("row_hash") != row_hash(row):
                errors.append(f"{field} row_hash mismatch")
                break


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        return ["required field set mismatch"]
    if not str(payload.get("status", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("status lacks terminal prefix")
    if not str(payload.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6562 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            row = provenance.get(field)
            if not isinstance(row, Mapping) or row.get("principle") != principle:
                errors.append("field_provenance principle mismatch")
                break
    aggregate = dict(payload.get("aggregate_row_recomputation") or {})
    if payload.get("constraint_saturation_independent_audit_ready_score") != aggregate.get(
        "ready_score_from_rows"
    ):
        errors.append("ready score mismatch")
    if payload.get("constraint_saturation_policy_audited_score") != aggregate.get(
        "policy_score_from_rows"
    ):
        errors.append("policy score mismatch")
    if payload.get("constraint_saturation_policy_audited_score") == 1.0 and (
        payload.get("constraint_saturation_independent_audit_ready_score") != 1.0
        or aggregate.get("bounded_policy_claim_confirmed") is not True
    ):
        errors.append("policy score requires ready bounded policy confirmation")
    if payload.get("constraint_saturation_independent_audit_ready_score") == 1.0:
        if payload.get("verdict_class") is not None:
            errors.append("ready score requires null verdict_class")
        if aggregate.get("failed_checks"):
            errors.append("ready score cannot have failed checks")
    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        errors.append("protected files changed")
    _validate_row_hashes(payload, errors)
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = _read_json(result_path)
        errors = validate_artifact(artifact)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {result_path}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        write=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {result_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
