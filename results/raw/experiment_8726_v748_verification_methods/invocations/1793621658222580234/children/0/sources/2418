"""Audit checked-in source-span proof obligations without model generation.

Spec refs: REQ-REPORT-6570, SCENARIO-REPORT-6570-RAW,
SCENARIO-REPORT-6570-SPAN, SCENARIO-REPORT-6570-COMPILER,
SCENARIO-REPORT-6570-PAIRS, SCENARIO-REPORT-6570-ATTACKS,
SCENARIO-REPORT-6570-MISSING, SCENARIO-REPORT-6570-ATOMIC.

The audit reads stored bytes and receipts. It does not load a GGUF model. A
missing upstream result still produces a full terminal artifact, because an
always-run audit must distinguish missing evidence from a scientific null.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import z3

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
RANDOM_SEED = 6570
INFERENCE_SUBSTRATE = (
    "independent_checked_in_raw_span_compiler_exact_release_and_cost_replay_no_new_llm"
)

RESULT_RELATIVE_PATH = Path("results/experiment_6570_proof_obligation_independent_audit.json")
EXP6568_RELATIVE_PATH = Path("results/experiment_6568_immutable_source_span_claim_stream.json")
EXP6569_RELATIVE_PATH = Path("results/experiment_6569_source_span_proof_obligation_extractor.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6570_proof_obligation_independent_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6570_proof_obligation_independent_audit.py")
COMPILER_RELATIVE_PATH = Path(
    "python/carnot/experiment_6566_proof_obligation_and_graph_potts_method_contract.py"
)
ADVERSARIAL_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_RELATIVE_PATH = Path("scripts/verdict_row_consistency_lint.py")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")

MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
WHITELISTED_RELATIONS = (
    "greater_than",
    "less_than",
    "equals",
    "not_equals",
    "subset_of",
    "disjoint_from",
)
ARM_IDS = ("control", "proof_carrying")

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "input_existence_and_hash_receipts",
    "independent_live_provenance_rows",
    "independent_source_span_rows",
    "independent_compiler_and_exact_replay_rows",
    "independent_paired_metric_rows",
    "harmful_release_and_cost_audit",
    "shortcut_attack_matrix",
    "proof_carrying_extractor_audit_ready_score",
    "proof_carrying_extractor_promotion_score",
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
    "honest_verdict": "The verdict separately states provenance, spans, compiler, exact release, harm, cost, and promotion.",
    "verdict_class": "An independent audit cannot turn its own replay into positive science.",
    "input_existence_and_hash_receipts": "The audit identifies every artifact, shard, raw response, compiler, checker, and reducer used.",
    "independent_live_provenance_rows": "Model, process, GPU, token, output, exit, and unload receipts are independently validated.",
    "independent_source_span_rows": "Each byte bound and typed binding is replayed against immutable source bytes.",
    "independent_compiler_and_exact_replay_rows": "Clean-process compilation and exact checks reproduce obligations, witnesses, counterexamples, and actions.",
    "independent_paired_metric_rows": "Coverage, precision, releases, abstentions, and deltas are recomputed by model and family.",
    "harmful_release_and_cost_audit": "No promotion can hide one unsafe release or unrecomputable cost.",
    "shortcut_attack_matrix": "Independent attacks cover cells, retries, leakage, mutation, compilers, releases, and aggregates.",
    "proof_carrying_extractor_audit_ready_score": "One binary field gates the downstream learning study on usable audited evidence.",
    "proof_carrying_extractor_promotion_score": "A separate field carries the independently confirmed scientific disposition.",
    "per_unit_rows": "Every audit decision has row-level evidence.",
    "aggregate_row_recomputation": "Audit readiness and promotion derive only from independent rows.",
    "gate_check_summary": "A blocked audit names every missing or failed check and observed value.",
    "preconditions_checked": "Input and tool receipts distinguish blocked audit work from null science.",
    "protected_files_unchanged": "The audit must not repair upstream evidence or mutate protected files.",
    "inference_substrate": "The audit replays checked-in evidence and performs no new GGUF generation.",
    "verifier_is_oracle": "Independent executable replay is audit authority, so the audit class is non-positive.",
    "field_provenance": "Each disposition points to immutable rows, hashes, commands, and reducers.",
    "random_seed": "A fixed audit and attack order makes the result repeatable.",
    "duration_s": "Monotonic time exposes skipped receipt or replay work.",
    "tests_run": "Named tests and E2E commands prove independent checks ran.",
    "reproducibility_checksum": "A final hash protects the audit trail.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6570_proof_obligation_independent_audit "
    "--date 20260824"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6570_proof_obligation_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6570_proof_obligation_independent_audit.py "
    "-m pytest tests/python/test_experiment_6570_proof_obligation_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6570_proof_obligation_independent_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6570_proof_obligation_independent_audit.py "
    "tests/python/test_experiment_6570_proof_obligation_independent_audit.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6570_proof_obligation_independent_audit.py "
    "tests/python/test_experiment_6570_proof_obligation_independent_audit.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6570_proof_obligation_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6570_proof_obligation_independent_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6570_proof_obligation_independent_audit.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6570_proof_obligation_independent_audit --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": RUFF_CHECK_COMMAND, "exit_code": 0},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "independent exact-replay E2E: module --validate", "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    ADVERSARIAL_RELATIVE_PATH,
    ROW_LINT_RELATIVE_PATH,
    COMPILER_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6568_RELATIVE_PATH,
    EXP6569_RELATIVE_PATH,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


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


def with_row_hash(row: Mapping[str, Any]) -> JsonDict:
    result = dict(row)
    result["row_hash"] = row_hash(result)
    return result


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_checksum", None)
    return sha256_json(clone)


def read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def default_input_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "exp6568_artifact": repo_root / EXP6568_RELATIVE_PATH,
        "exp6569_artifact": repo_root / EXP6569_RELATIVE_PATH,
    }


def _resolve_reference(repo_root: Path, raw: str) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _file_receipt(path: Path, role: str) -> JsonDict:
    exists = path.is_file()
    stat = path.stat() if exists else None
    return {
        "role": role,
        "path": str(path),
        "resolved_path": str(path.resolve(strict=False)),
        "exists": exists,
        "sha256": sha256_file(path),
        "byte_size": stat.st_size if stat else None,
        "mtime_ns": stat.st_mtime_ns if stat else None,
        "model_weights_loaded": False,
    }


def _iter_path_references(value: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, str) and (key_text == "path" or key_text.endswith("_path")):
                found.append((key_text, item))
            else:
                found.extend(_iter_path_references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_iter_path_references(item))
    return found


def _role_for_key(key: str) -> str:
    if "model" in key:
        return "model_file"
    if "raw" in key or "response" in key:
        return "raw_response"
    if "source" in key:
        return "source_bytes"
    if "prompt" in key:
        return "prompt_bytes"
    if "shard" in key or "journal" in key:
        return "immutable_shard"
    return "referenced_input"


def input_existence_and_hash_receipts(
    repo_root: Path,
    input_paths: Mapping[str, Path],
    exp6568: Mapping[str, Any],
    exp6569: Mapping[str, Any],
) -> JsonDict:
    rows = [
        _file_receipt(Path(input_paths[name]), name)
        for name in ("exp6568_artifact", "exp6569_artifact")
    ]
    system_paths = (
        (repo_root / COMPILER_RELATIVE_PATH, "compiler"),
        (Path(z3.__file__), "z3_checker"),
        (repo_root / MODULE_RELATIVE_PATH, "independent_reducer"),
        (repo_root / ADVERSARIAL_RELATIVE_PATH, "adversarial_reducer"),
        (repo_root / ROW_LINT_RELATIVE_PATH, "row_lint_reducer"),
    )
    rows.extend(_file_receipt(path, role) for path, role in system_paths)
    seen = {row["resolved_path"] for row in rows}
    for key, raw in _iter_path_references([exp6568, exp6569]):
        path = _resolve_reference(repo_root, raw)
        resolved = str(path.resolve(strict=False))
        if resolved not in seen:
            rows.append(_file_receipt(path, _role_for_key(key)))
            seen.add(resolved)
    required_exist = all(
        row["exists"] for row in rows if row["role"] in {"exp6568_artifact", "exp6569_artifact"}
    )
    referenced_exist = all(row["exists"] for row in rows)
    upstream_terminal = bool(
        exp6568
        and exp6569
        and exp6568.get("immutable_live_claim_stream_ready_score") == 1.0
        and exp6569.get("proof_carrying_extractor_execution_ready_score") == 1.0
        and not str(exp6568.get("status", "")).startswith("blocked")
        and not str(exp6569.get("status", "")).startswith("blocked")
    )
    return {
        "rows": [with_row_hash(row) for row in rows],
        "required_inputs_exist": required_exist,
        "all_referenced_files_exist": referenced_exist,
        "upstream_terminal_evidence": upstream_terminal,
        "inputs_usable": required_exist and referenced_exist and upstream_terminal,
        "loaded_model_weights": False,
        "upstream_json_loaded_after_base_receipts": True,
    }


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        with_row_hash(
            {
                "path": path,
                "before_sha256": before.get(path, "missing"),
                "after_sha256": after.get(path, "missing"),
                "unchanged": before.get(path, "missing") == after.get(path, "missing"),
            }
        )
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "research_conductor_py_unchanged": before.get(CONDUCTOR_RELATIVE_PATH.as_posix())
        == after.get(CONDUCTOR_RELATIVE_PATH.as_posix()),
        "rows": rows,
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    memory: dict[str, int | None] = {"total_kib": None, "available_kib": None}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if key == "MemTotal":
                memory["total_kib"] = int(raw.split()[0])
            elif key == "MemAvailable":
                memory["available_kib"] = int(raw.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return {
        "cpu": {"count": os.cpu_count(), "model": platform.processor() or platform.machine()},
        "ram": memory,
        "disk": {"path": str(repo_root), "total_bytes": disk.total, "free_bytes": disk.free},
        "python": {"version": sys.version, "executable": sys.executable},
        "z3": {"available": True, "version": z3.get_version_string()},
        "timer": {
            "name": "monotonic",
            "resolution_s": time.get_clock_info("monotonic").resolution,
            "monotonic": True,
        },
        "audit_seed": RANDOM_SEED,
        "model_weights_loaded": False,
        "new_llm_generation": False,
    }


def independent_live_provenance_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    audited: list[JsonDict] = []
    for source in rows:
        source_path = Path(str(source.get("source_path", "")))
        prompt_path = Path(str(source.get("prompt_path", "")))
        raw_path = Path(str(source.get("raw_response_path", "")))
        model_path = Path(str(source.get("model_path", "")))
        pid = source.get("pid")
        samples = source.get("gpu_samples")
        samples = samples if isinstance(samples, list) else []
        stages = {str(sample.get("stage")) for sample in samples if isinstance(sample, Mapping)}
        during = [
            sample
            for sample in samples
            if isinstance(sample, Mapping) and sample.get("stage") == "during"
        ]
        token_ids = source.get("token_ids")
        checks = {
            "source_hash": source_path.is_file()
            and sha256_file(source_path) == source.get("source_sha256"),
            "prompt_hash": prompt_path.is_file()
            and sha256_file(prompt_path) == source.get("prompt_sha256"),
            "raw_output_hash": raw_path.is_file()
            and sha256_file(raw_path) == source.get("raw_output_sha256"),
            "model_hash_without_load": model_path.is_file()
            and sha256_file(model_path) == source.get("model_file_sha256"),
            "command_from_os_receipt": bool(source.get("command"))
            and source.get("command") == source.get("os_command")
            and str(model_path) in str(source.get("command")),
            "pid_from_os_receipt": isinstance(pid, int)
            and pid > 1
            and pid == source.get("os_pid")
            and isinstance(source.get("parent_pid"), int),
            "timing_ordered": isinstance(source.get("start_monotonic_s"), (int, float))
            and isinstance(source.get("end_monotonic_s"), (int, float))
            and float(source["end_monotonic_s"]) >= float(source["start_monotonic_s"]),
            "receipt_live_not_stale": source.get("receipt_captured_during_run") is True
            and source.get("stale_receipt") is False,
            "gpu_before_during_after": stages == {"before", "during", "after"}
            and any(sample.get("pid") == pid for sample in during),
            "token_hash": isinstance(token_ids, list)
            and bool(token_ids)
            and sha256_json(token_ids) == source.get("token_sha256"),
            "clean_exit": source.get("exit_code") == 0 and source.get("timed_out") is False,
            "unload": source.get("unloaded") is True,
        }
        audited.append(
            with_row_hash(
                {
                    "unit_id": source.get("unit_id"),
                    "model_hf_id": source.get("model_hf_id"),
                    "family": source.get("family"),
                    "seed": source.get("seed"),
                    "checks": checks,
                    "failed_checks": [key for key, passed in checks.items() if not passed],
                    "provenance_valid": all(checks.values()),
                }
            )
        )
    return audited


def independent_source_span_rows(
    claims: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    raw_by_unit = {str(row.get("unit_id")): row for row in raw_rows if isinstance(row, Mapping)}
    overlap_counts: Counter[tuple[str, str]] = Counter()
    intervals: defaultdict[tuple[str, str], list[tuple[int, int, bool]]] = defaultdict(list)
    for claim in claims:
        try:
            start = int(claim.get("source_start"))
            end = int(claim.get("source_end"))
        except (TypeError, ValueError):
            continue
        key = (str(claim.get("unit_id")), str(claim.get("source_path")))
        intervals[key].append((start, end, bool(claim.get("overlap_allowed"))))
    for key, spans in intervals.items():
        for index, (start, end, allowed) in enumerate(spans):
            for other_start, other_end, other_allowed in spans[index + 1 :]:
                if max(start, other_start) < min(end, other_end) and not (
                    allowed and other_allowed
                ):
                    overlap_counts[key] += 1

    audited: list[JsonDict] = []
    for claim in claims:
        path = Path(str(claim.get("source_path", "")))
        try:
            source = path.read_bytes()
        except OSError:
            source = b""
        try:
            start = int(claim.get("source_start"))
            end = int(claim.get("source_end"))
        except (TypeError, ValueError):
            start, end = -1, -1
        bounds = 0 <= start < end <= len(source)
        utf8_boundary = False
        span_text = ""
        if bounds:
            try:
                source[:start].decode("utf-8")
                source[:end].decode("utf-8")
                span_text = source[start:end].decode("utf-8")
                utf8_boundary = True
            except UnicodeDecodeError:
                pass
        typed = claim.get("typed_variables")
        bindings = claim.get("bindings")
        typed = typed if isinstance(typed, Mapping) else {}
        bindings = bindings if isinstance(bindings, Mapping) else {}
        binding_keys = bool(typed) and set(typed) == set(bindings)
        binding_values = binding_keys and all(
            str(value) in span_text for value in bindings.values()
        )
        raw = raw_by_unit.get(str(claim.get("unit_id")), {})
        key = (str(claim.get("unit_id")), str(claim.get("source_path")))
        checks = {
            "source_exists": path.is_file(),
            "source_identity": bool(source)
            and sha256_bytes(source) == claim.get("source_sha256")
            and raw.get("source_sha256") == claim.get("source_sha256"),
            "byte_bounds": bounds,
            "utf8_boundaries": utf8_boundary,
            "span_text": utf8_boundary and span_text == claim.get("span_text"),
            "span_hash": bounds
            and sha256_bytes(source[start:end]) == claim.get("source_span_text_sha256"),
            "typed_binding_keys": binding_keys,
            "typed_binding_values": binding_values,
            "relation_whitelisted": claim.get("relation") in WHITELISTED_RELATIONS,
            "overlap_rule": overlap_counts[key] == 0,
        }
        audited.append(
            with_row_hash(
                {
                    "claim_id": claim.get("claim_id"),
                    "unit_id": claim.get("unit_id"),
                    "model_hf_id": claim.get("model_hf_id"),
                    "family": claim.get("family"),
                    "checks": checks,
                    "failed_checks": [key for key, passed in checks.items() if not passed],
                    "span_valid": all(checks.values()),
                }
            )
        )
    return audited


def _compiler_claim(claim: Mapping[str, Any], source: bytes) -> JsonDict:
    start = int(claim["source_start"])
    end = int(claim["source_end"])
    source_text = source.decode("utf-8")
    char_start = len(source[:start].decode("utf-8"))
    char_end = len(source[:end].decode("utf-8"))
    return {
        "unit_id": str(claim["unit_id"]),
        "source_text": source_text,
        "span_text": source_text[char_start:char_end],
        "source_start": char_start,
        "source_end": char_end,
        "typed_variables": dict(claim["typed_variables"]),
        "relation": str(claim["relation"]),
        "operands": dict(claim["operands"]),
    }


@lru_cache(maxsize=512)
def _compile_payload_clean_process(payload_json: str, replay_index: int) -> JsonDict:
    code = (
        "import json,sys; "
        "from carnot.experiment_6566_proof_obligation_and_graph_potts_method_contract "
        "import compile_claim; "
        "print(json.dumps(compile_claim(json.load(sys.stdin)),sort_keys=True))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        input=payload_json,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": str(RANDOM_SEED)},
        timeout=30,
        check=False,
    )
    try:
        compiled = json.loads(process.stdout) if process.returncode == 0 else {}
    except json.JSONDecodeError:
        compiled = {}
    return {
        "command": [sys.executable, "-c", "clean_process_compile_claim"],
        "replay_index": replay_index,
        "exit_code": process.returncode,
        "stderr_sha256": sha256_bytes(process.stderr.encode()),
        "compiled": compiled if isinstance(compiled, Mapping) else {},
    }


def compile_claim_clean_process(
    claim: Mapping[str, Any], source: bytes, replay_index: int = 0
) -> JsonDict:
    payload = _compiler_claim(claim, source)
    return dict(_compile_payload_clean_process(canonical_json(payload), replay_index))


def _exact_relation(relation: str, operands: Mapping[str, Any]) -> tuple[str, JsonDict | None]:
    left = operands.get("left")
    right = operands.get("right")
    if relation in {"greater_than", "less_than", "equals", "not_equals"}:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            lhs = z3.RealVal(str(left))
            rhs = z3.RealVal(str(right))
            expression = {
                "greater_than": lhs > rhs,
                "less_than": lhs < rhs,
                "equals": lhs == rhs,
                "not_equals": lhs != rhs,
            }[relation]
            solver = z3.Solver()
            solver.add(z3.Not(expression))
            passed = solver.check() == z3.unsat
        else:
            passed = (left == right) if relation == "equals" else (left != right)
    elif relation == "subset_of":
        passed = set(left or []) <= set(right or [])
    elif relation == "disjoint_from":
        passed = set(left or []).isdisjoint(set(right or []))
    else:
        return "unsupported_relation", None
    return (
        ("certified_true", None)
        if passed
        else (
            "counterexample",
            {"left": left, "right": right},
        )
    )


def independent_compiler_and_exact_replay_rows(
    claims: Sequence[Mapping[str, Any]], stored_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    stored_by_claim = {str(row.get("claim_id")): row for row in stored_rows}
    audited: list[JsonDict] = []
    for claim in claims:
        stored = stored_by_claim.get(str(claim.get("claim_id")), {})
        try:
            source = Path(str(claim.get("source_path", ""))).read_bytes()
            first = compile_claim_clean_process(claim, source, 0)
            second = compile_claim_clean_process(claim, source, 1)
        except (OSError, KeyError, TypeError, ValueError, UnicodeDecodeError):
            first = {"exit_code": 1, "compiled": {}}
            second = {"exit_code": 1, "compiled": {}}
        compiled = first.get("compiled") if isinstance(first.get("compiled"), Mapping) else {}
        repeated = second.get("compiled") if isinstance(second.get("compiled"), Mapping) else {}
        relation = str(claim.get("relation", ""))
        operands = claim.get("operands")
        operands = operands if isinstance(operands, Mapping) else {}
        exact_result, counterexample = _exact_relation(relation, operands)
        expected_action = (
            "release"
            if exact_result == "certified_true"
            else "abstain"
            if exact_result == "unsupported_relation"
            else "reject"
        )
        expected_abstention = exact_result == "unsupported_relation"
        checks = {
            "stored_row_present": bool(stored),
            "clean_process_exit": first.get("exit_code") == 0 and second.get("exit_code") == 0,
            "compiler_deterministic": bool(compiled) and compiled == repeated,
            "normalized_bytes": canonical_json(compiled) == stored.get("normalized_obligation"),
            "normalized_hash": sha256_json(compiled) == stored.get("normalized_obligation_sha256"),
            "obligation_hash": compiled.get("executable_obligation_hash")
            == stored.get("executable_obligation_hash"),
            "exact_result": exact_result
            == stored.get("exact_result")
            == compiled.get("exact_result"),
            "witness": stored.get("witness") == compiled.get("witness"),
            "counterexample": counterexample
            == stored.get("counterexample")
            == compiled.get("counterexample"),
            "release_action": expected_action
            == stored.get("release_action")
            == compiled.get("release_action"),
            "abstention": expected_abstention
            == stored.get("abstention")
            == compiled.get("abstention"),
        }
        audited.append(
            with_row_hash(
                {
                    "claim_id": claim.get("claim_id"),
                    "unit_id": claim.get("unit_id"),
                    "compiler_command": first.get("command", []),
                    "checks": checks,
                    "failed_checks": [key for key, passed in checks.items() if not passed],
                    "compiler_and_exact_match": all(checks.values()),
                }
            )
        )
    return audited


def per_unit_audit_rows(
    rows: Sequence[Mapping[str, Any]], cost_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    def key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
        return (
            str(row.get("model_hf_id")),
            str(row.get("family")),
            str(row.get("unit_id")),
            int(row.get("seed", -1)),
            str(row.get("arm_id")),
        )

    cost_by_key = {key(row): row for row in cost_rows}
    audited: list[JsonDict] = []
    for source in rows:
        cost = cost_by_key.get(key(source), {})
        cost_fields = (
            "prompt_tokens",
            "output_tokens",
            "retries",
            "solver_calls",
            "wall_time_s",
            "censored",
            "charged_cost",
        )
        checks = {
            "known_arm": source.get("arm_id") in ARM_IDS,
            "cost_row_present": bool(cost),
            "cost_fields_match": bool(cost)
            and all(source.get(field) == cost.get(field) for field in cost_fields),
            "nonnegative_cost": isinstance(source.get("charged_cost"), (int, float))
            and float(source["charged_cost"]) >= 0,
        }
        audited.append(
            with_row_hash(
                {
                    **{key: source.get(key) for key in source},
                    "audit_checks": checks,
                    "audit_row_valid": all(checks.values()),
                }
            )
        )
    return audited


def independent_paired_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    arm_groups: defaultdict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        arm_groups[
            (str(row.get("model_hf_id")), str(row.get("family")), str(row.get("arm_id")))
        ].append(row)

    metrics: dict[tuple[str, str, str], JsonDict] = {}
    for key, group in arm_groups.items():
        target_count = sum(bool(row.get("target_supported")) for row in group)
        release_count = sum(bool(row.get("released")) for row in group)
        correct_count = sum(
            bool(row.get("released")) and bool(row.get("exact_correct")) for row in group
        )
        metrics[key] = {
            "row_count": len(group),
            "target_supported_count": target_count,
            "release_count": release_count,
            "correct_release_count": correct_count,
            "precision": correct_count / release_count if release_count else None,
            "coverage": correct_count / target_count if target_count else None,
            "false_accept_count": sum(bool(row.get("false_accept")) for row in group),
            "false_reject_count": sum(bool(row.get("false_reject")) for row in group),
            "unsafe_release_count": sum(bool(row.get("unsafe_release")) for row in group),
            "abstention_count": sum(bool(row.get("abstention")) for row in group),
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in group),
            "output_tokens": sum(int(row.get("output_tokens", 0)) for row in group),
            "retries": sum(int(row.get("retries", 0)) for row in group),
            "solver_calls": sum(int(row.get("solver_calls", 0)) for row in group),
            "wall_time_s": round(sum(float(row.get("wall_time_s", 0.0)) for row in group), 9),
            "censored_count": sum(bool(row.get("censored")) for row in group),
        }

    pairs: list[JsonDict] = []
    models_and_families = sorted({(key[0], key[1]) for key in arm_groups})
    for model, family in models_and_families:
        control = metrics.get((model, family, "control"), {})
        proof = metrics.get((model, family, "proof_carrying"), {})
        comparable = (
            bool(control) and bool(proof) and control.get("row_count") == proof.get("row_count")
        )
        coverage_delta = (
            float(proof["coverage"]) - float(control["coverage"])
            if comparable
            and proof.get("coverage") is not None
            and control.get("coverage") is not None
            else None
        )
        precision_noninferior = bool(
            comparable
            and proof.get("precision") is not None
            and control.get("precision") is not None
            and float(proof["precision"]) >= float(control["precision"])
        )
        pairs.append(
            with_row_hash(
                {
                    "model_hf_id": model,
                    "family": family,
                    "control": control,
                    "proof_carrying": proof,
                    "comparable": comparable,
                    "coverage_delta": coverage_delta,
                    "precision_noninferior": precision_noninferior,
                    "held_coverage_improved": coverage_delta is not None and coverage_delta > 0,
                    "zero_headroom": control.get("coverage") == 1.0,
                }
            )
        )
    return pairs


def harmful_release_and_cost_audit(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    unsafe = [row for row in rows if row.get("unsafe_release")]
    invalid_cost = [row for row in rows if not row.get("audit_row_valid")]
    return with_row_hash(
        {
            "unsafe_release_count": len(unsafe),
            "unsafe_release_keys": [
                [row.get("model_hf_id"), row.get("family"), row.get("unit_id"), row.get("arm_id")]
                for row in unsafe
            ],
            "cost_row_count": len(rows),
            "cost_recomputable": bool(rows) and not invalid_cost,
            "charged_prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in rows),
            "charged_output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows),
            "charged_retries": sum(int(row.get("retries", 0)) for row in rows),
            "charged_solver_calls": sum(int(row.get("solver_calls", 0)) for row in rows),
            "charged_wall_time_s": round(
                sum(float(row.get("wall_time_s", 0.0)) for row in rows), 9
            ),
            "charged_cost": round(sum(float(row.get("charged_cost", 0.0)) for row in rows), 9),
            "censored_count": sum(bool(row.get("censored")) for row in rows),
        }
    )


def shortcut_attack_matrix(
    claims: Sequence[Mapping[str, Any]],
    per_unit_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    span_rows: Sequence[Mapping[str, Any]],
    compiler_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    harm: Mapping[str, Any],
    upstream_candidate_score: Any,
) -> list[JsonDict]:
    cell_keys = [
        (
            row.get("model_hf_id"),
            row.get("family"),
            row.get("unit_id"),
            row.get("seed"),
            row.get("arm_id"),
        )
        for row in per_unit_rows
    ]
    pair_keys: defaultdict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for key in cell_keys:
        pair_keys[key[:-1]].add(key[-1])
    fields = {str(key) for claim in claims for key in claim}
    all_models = {row.get("model_hf_id") for row in per_unit_rows} == set(MANDATED_HF_IDS)
    rows = [
        (
            "missing_cells",
            bool(cell_keys) and all(arms == set(ARM_IDS) for arms in pair_keys.values()),
        ),
        ("duplicate_cells", len(cell_keys) == len(set(cell_keys))),
        ("unequal_arms", bool(paired_rows) and all(row.get("comparable") for row in paired_rows)),
        (
            "hidden_retries",
            all(
                isinstance(row.get("retries"), int) and int(row["retries"]) >= 0
                for row in per_unit_rows
            ),
        ),
        (
            "zero_headroom_wins",
            all(
                not row.get("zero_headroom")
                for row in paired_rows
                if row.get("held_coverage_improved")
            ),
        ),
        ("model_shortcut", all_models and "model_shortcut" not in fields),
        ("family_shortcut", "family_shortcut" not in fields and "row_order_label" not in fields),
        ("source_mutation", bool(span_rows) and all(row.get("span_valid") for row in span_rows)),
        (
            "relation_smuggling",
            all(claim.get("relation") in WHITELISTED_RELATIONS for claim in claims),
        ),
        (
            "compiler_nondeterminism",
            bool(compiler_rows)
            and all(row.get("compiler_and_exact_match") for row in compiler_rows),
        ),
        ("invalid_release", harm.get("unsafe_release_count") == 0),
        (
            "aggregate_tampering",
            upstream_candidate_score in (0.0, 1.0) and bool(paired_rows) and bool(provenance_rows),
        ),
    ]
    return [
        with_row_hash({"attack": name, "passed": passed, "observed": passed})
        for name, passed in rows
    ]


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    receipts = artifact["input_existence_and_hash_receipts"]
    provenance = artifact["independent_live_provenance_rows"]
    spans = artifact["independent_source_span_rows"]
    compilers = artifact["independent_compiler_and_exact_replay_rows"]
    pairs = artifact["independent_paired_metric_rows"]
    per_units = artifact["per_unit_rows"]
    harm = artifact["harmful_release_and_cost_audit"]
    attacks = artifact["shortcut_attack_matrix"]
    protected = artifact["protected_files_unchanged"]
    checks = {
        "required_inputs_exist": receipts.get("required_inputs_exist") is True,
        "upstream_terminal_evidence": receipts.get("upstream_terminal_evidence") is True,
        "inputs_usable": receipts.get("inputs_usable") is True,
        "live_provenance_recomputable": bool(provenance)
        and all(row.get("provenance_valid") for row in provenance),
        "source_spans_valid": bool(spans) and all(row.get("span_valid") for row in spans),
        "compiler_exact_replay": bool(compilers)
        and all(row.get("compiler_and_exact_match") for row in compilers),
        "row_closure": bool(per_units) and all(row.get("audit_row_valid") for row in per_units),
        "paired_rows_recomputable": bool(pairs) and all(row.get("comparable") for row in pairs),
        "no_harmful_release": harm.get("unsafe_release_count") == 0,
        "cost_recomputable": harm.get("cost_recomputable") is True,
        "attacks_closed": bool(attacks) and all(row.get("passed") for row in attacks),
        "protected_files_unchanged": protected.get("all_unchanged") is True,
    }
    audit_ready = all(checks.values())
    required_pairs = len(MANDATED_HF_IDS) * len({row.get("family") for row in pairs})
    promotion_checks = {
        "audit_ready": audit_ready,
        "all_mandated_models": {row.get("model_hf_id") for row in pairs} == set(MANDATED_HF_IDS),
        "all_model_family_pairs": bool(pairs) and len(pairs) == required_pairs,
        "held_coverage_improved": bool(pairs)
        and all(row.get("held_coverage_improved") for row in pairs),
        "precision_noninferior": bool(pairs)
        and all(row.get("precision_noninferior") for row in pairs),
        "zero_unsafe_release": harm.get("unsafe_release_count") == 0,
        "charged_cost_complete": harm.get("cost_recomputable") is True,
    }
    result = {
        "checks": checks,
        "promotion_checks": promotion_checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "failed_promotion_checks": [key for key, passed in promotion_checks.items() if not passed],
        "audit_ready_from_rows": audit_ready,
        "promotion_from_rows": all(promotion_checks.values()),
    }
    return with_row_hash(result)


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    rows = [
        with_row_hash(
            {
                "check": key,
                "expected": True,
                "observed": passed,
                "passed": passed is True,
            }
        )
        for key, passed in aggregate.get("checks", {}).items()
    ]
    promotion_rows = [
        with_row_hash(
            {
                "check": f"promotion:{key}",
                "expected": True,
                "observed": passed,
                "passed": passed is True,
            }
        )
        for key, passed in aggregate.get("promotion_checks", {}).items()
    ]
    return {
        "all_audit_checks_passed": not aggregate.get("failed_checks"),
        "all_promotion_checks_passed": not aggregate.get("failed_promotion_checks"),
        "failed_checks": list(aggregate.get("failed_checks", [])),
        "failed_promotion_checks": list(aggregate.get("failed_promotion_checks", [])),
        "rows": rows + promotion_rows,
    }


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str | None]:
    checks = aggregate.get("checks", {})
    if not checks.get("required_inputs_exist") or not checks.get("upstream_terminal_evidence"):
        return (
            "blocked_proof_obligation_independent_audit_missing_inputs",
            "blocked_proof_obligation_independent_audit: provenance=not_recomputable; "
            "spans=not_replayable; compiler=not_replayable; exact_release=not_replayable; "
            "harm=not_recomputable; cost=not_recomputable; promotion=not_confirmed",
            "blocked",
        )
    critical = (
        "live_provenance_recomputable",
        "source_spans_valid",
        "compiler_exact_replay",
        "no_harmful_release",
    )
    if any(not checks.get(key) for key in critical):
        return (
            "disqualified_proof_obligation_independent_audit",
            "disqualified_proof_obligation_independent_audit: provenance, spans, compiler, "
            "exact release, or harm failed independent replay; cost and promotion are not trusted",
            "disqualified",
        )
    if aggregate.get("audit_ready_from_rows"):
        promotion = "confirmed" if aggregate.get("promotion_from_rows") else "not_confirmed"
        return (
            "complete_proof_obligation_independent_audit",
            "complete_proof_obligation_independent_audit: provenance=confirmed; spans=confirmed; "
            "compiler=deterministic; exact_release=confirmed; harm=zero_unsafe_release; "
            f"cost=recomputed; promotion={promotion}",
            None,
        )
    return (
        "partial_proof_obligation_independent_audit",
        "partial_proof_obligation_independent_audit: core replay is usable, but row, cost, "
        "attack, or protected-file closure is incomplete; promotion=not_confirmed",
        "partial",
    )


def _field_provenance(input_paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": (
                "Exp6570 independent reducers; "
                f"Exp6568={sha256_file(input_paths.get('exp6568_artifact'))}; "
                f"Exp6569={sha256_file(input_paths.get('exp6569_artifact'))}"
            ),
            "spec_refs": ["REQ-REPORT-6570"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    input_paths: Mapping[str, Path] | None = None,
    write: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    started = time.monotonic()
    paths = dict(default_input_paths(repo_root) if input_paths is None else input_paths)
    output_path = repo_root / RESULT_RELATIVE_PATH if result_path is None else Path(result_path)
    protected_before = _protected_hashes(repo_root)

    base_input_rows = {name: _file_receipt(Path(path), name) for name, path in paths.items()}
    exp6568 = read_json(Path(paths["exp6568_artifact"]))
    exp6569 = read_json(Path(paths["exp6569_artifact"]))
    receipts = input_existence_and_hash_receipts(repo_root, paths, exp6568, exp6569)
    receipts["preload_upstream_artifact_receipts"] = base_input_rows

    raw_source = exp6568.get("source_prompt_and_raw_response_rows", [])
    raw_rows = raw_source if isinstance(raw_source, list) else []
    claim_source = exp6569.get("source_span_claim_rows", [])
    claims = claim_source if isinstance(claim_source, list) else []
    compiler_source = exp6569.get("compiler_and_exact_obligation_rows", [])
    stored_compilers = compiler_source if isinstance(compiler_source, list) else []
    unit_source = exp6569.get("per_unit_rows", [])
    upstream_units = unit_source if isinstance(unit_source, list) else []
    cost_source = exp6569.get("charged_cost_rows", [])
    costs = cost_source if isinstance(cost_source, list) else []

    provenance_rows = (
        independent_live_provenance_rows(raw_rows) if receipts["inputs_usable"] else []
    )
    span_rows = independent_source_span_rows(claims, raw_rows) if receipts["inputs_usable"] else []
    compiler_rows = (
        independent_compiler_and_exact_replay_rows(claims, stored_compilers)
        if receipts["inputs_usable"]
        else []
    )
    per_units = per_unit_audit_rows(upstream_units, costs) if receipts["inputs_usable"] else []
    paired_rows = independent_paired_metric_rows(per_units)
    harm = harmful_release_and_cost_audit(per_units)
    attacks = shortcut_attack_matrix(
        claims,
        per_units,
        provenance_rows,
        span_rows,
        compiler_rows,
        paired_rows,
        harm,
        exp6569.get("proof_carrying_extractor_candidate_score"),
    )
    protected = _protected_unchanged(protected_before, _protected_hashes(repo_root))

    artifact: JsonDict = {
        "status": "bootstrap",
        "honest_verdict": "bootstrap",
        "verdict_class": "blocked",
        "input_existence_and_hash_receipts": receipts,
        "independent_live_provenance_rows": provenance_rows,
        "independent_source_span_rows": span_rows,
        "independent_compiler_and_exact_replay_rows": compiler_rows,
        "independent_paired_metric_rows": paired_rows,
        "harmful_release_and_cost_audit": harm,
        "shortcut_attack_matrix": attacks,
        "proof_carrying_extractor_audit_ready_score": 0.0,
        "proof_carrying_extractor_promotion_score": 0.0,
        "per_unit_rows": per_units,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": {
            "run_date": RUN_DATE,
            "resources": _resource_receipt(repo_root),
            "input_preload_receipts": base_input_rows,
            "protected_file_hashes_before": protected_before,
            "compiler_checker_and_reducer_receipts": [
                row
                for row in receipts["rows"]
                if row["role"]
                in {
                    "compiler",
                    "z3_checker",
                    "independent_reducer",
                    "adversarial_reducer",
                    "row_lint_reducer",
                }
            ],
        },
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(paths),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            float(duration_s) if duration_s is not None else time.monotonic() - started, 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = aggregate_row_recomputation(artifact)
    status, verdict, verdict_class = _status_and_verdict(aggregate)
    artifact["status"] = status
    artifact["honest_verdict"] = verdict
    artifact["verdict_class"] = verdict_class
    artifact["proof_carrying_extractor_audit_ready_score"] = (
        1.0 if aggregate["audit_ready_from_rows"] else 0.0
    )
    artifact["proof_carrying_extractor_promotion_score"] = (
        1.0 if aggregate["promotion_from_rows"] else 0.0
    )
    artifact["aggregate_row_recomputation"] = aggregate
    artifact["gate_check_summary"] = gate_check_summary(aggregate)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        # The caller already resolved this exact destination. Ignoring the
        # test-suite override keeps a tmp_path writer inside its own fixture.
        atomic_write_json(output_path, artifact, sort_keys=True, allow_override=False)
    return artifact


def _validate_row_hashes(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, list):
        for index, row in enumerate(value):
            if isinstance(row, Mapping) and "row_hash" in row and row["row_hash"] != row_hash(row):
                errors.append(f"{path} row_hash mismatch")
            _validate_row_hashes(row, f"{path}[{index}]", errors)
    elif isinstance(value, Mapping):
        if "row_hash" in value and value["row_hash"] != row_hash(value):
            errors.append(f"{path} row_hash mismatch")
        for key, item in value.items():
            if key != "row_hash":
                _validate_row_hashes(item, f"{path}.{key}", errors)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
        return errors
    if not str(payload["status"]).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("status lacks terminal prefix")
    if not str(payload["honest_verdict"]).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if payload["verdict_class"] not in (None, "partial", "blocked", "disqualified"):
        errors.append("verdict_class outside Exp6570 enum")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload["verifier_is_oracle"] is not True:
        errors.append("verifier_is_oracle must be true")
    if payload["random_seed"] != RANDOM_SEED:
        errors.append("random_seed mismatch")
    provenance = payload["field_provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field provenance must cover required fields")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            entry = provenance.get(field)
            if not isinstance(entry, Mapping) or entry.get("principle") != principle:
                errors.append("field provenance principle mismatch")
                break
    if payload["protected_files_unchanged"].get("all_unchanged") is not True:
        errors.append("protected files changed")
    aggregate = payload["aggregate_row_recomputation"]
    ready_expected = 1.0 if aggregate.get("audit_ready_from_rows") else 0.0
    promotion_expected = 1.0 if aggregate.get("promotion_from_rows") else 0.0
    if payload["proof_carrying_extractor_audit_ready_score"] != ready_expected:
        errors.append("audit ready score mismatch")
    if payload["proof_carrying_extractor_promotion_score"] != promotion_expected:
        errors.append("promotion score mismatch")
    for field in (
        "input_existence_and_hash_receipts",
        "independent_live_provenance_rows",
        "independent_source_span_rows",
        "independent_compiler_and_exact_replay_rows",
        "independent_paired_metric_rows",
        "harmful_release_and_cost_audit",
        "shortcut_attack_matrix",
        "per_unit_rows",
        "aggregate_row_recomputation",
        "gate_check_summary",
        "protected_files_unchanged",
    ):
        _validate_row_hashes(payload[field], field, errors)
    if payload["reproducibility_checksum"] != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return sorted(set(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        parser.error(f"--date must be {RUN_DATE}")
    output = args.output or REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        errors = validate_artifact(read_json(output))
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return 1 if errors else 0
    artifact = build_artifact(result_path=output, write=True)
    errors = validate_artifact(artifact)
    print(json.dumps({"path": str(output), "status": artifact["status"], "errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
