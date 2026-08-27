"""Qualify the frozen Exp6661 fixture with task-owned evidence.

The repository suite stays visible as a diagnostic. Only the focused Exp6661
nodes and their recorded checks can release this infrastructure fixture. The
module replays existing builders and does not create or change corpus rows.

Spec: REQ-REPORT-6675 and SCENARIO-REPORT-6675-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6661_triggered_tail_fixture as fixture


JsonDict = dict[str, Any]
CommandRunner = Callable[[list[str], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260827"
RANDOM_SEED = 6_675_027
INFERENCE_SUBSTRATE = "cpu_fixture_receipt_and_exact_checks_no_llm"

RESULT_PATH = Path("results/experiment_6675_triggered_tail_scope_receipt.json")
MODULE_PATH = Path("python/carnot/experiment_6675_triggered_tail_scope_receipt.py")
TEST_PATH = Path("tests/python/test_experiment_6675_triggered_tail_scope_receipt.py")
REPORT_SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"
REPORT_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXP6661_ARTIFACT_PATH = Path("results/experiment_6661_triggered_tail_fixture.json")
EXP6661_SOURCE_PATH = Path("python/carnot/experiment_6661_triggered_tail_fixture.py")
EXP6661_TEST_PATH = Path("tests/python/test_experiment_6661_triggered_tail_fixture.py")
EXP6661_SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
PARSER_SUPPORT_PATH = Path("python/carnot/inference/grammar.py")
V582_DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
GLOBAL_CACHE_PATH = Path(".pytest_cache/v/cache/lastfailed")
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))
OWNED_NODE_PREFIX = EXP6661_TEST_PATH.as_posix()

FOCUSED_TEST_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6661_triggered_tail_fixture.py -m pytest "
    "tests/python/test_experiment_6661_triggered_tail_fixture.py -q --no-cov -n 0 "
    "-o addopts="
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6661_triggered_tail_fixture.py "
    "--fail-under=100 --show-missing"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6661_triggered_tail_fixture.py "
    "tests/python/test_experiment_6661_triggered_tail_fixture.py"
)
FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6661_triggered_tail_fixture.py "
    "tests/python/test_experiment_6661_triggered_tail_fixture.py"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6661_triggered_tail_fixture.py"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
COLLECT_COMMAND = (
    ".venv/bin/pytest --collect-only -q "
    "tests/python/test_experiment_6661_triggered_tail_fixture.py "
    "--no-cov -n 0 -o addopts="
)
EXP6661_OWNED_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    RUFF_COMMAND,
    FORMAT_COMMAND,
    SPEC_COMMAND,
)


def _definition(ordinal: int, check_id: str, command: str) -> JsonDict:
    return {
        "ordinal": ordinal,
        "check_id": check_id,
        "command": command,
        "expected_exit_code": 0,
        "expected_node_count": 33,
        "expected_coverage_percent": 100.0 if check_id == "scoped_coverage" else None,
    }


OWNED_CHECK_DEFINITIONS = (
    _definition(1, "focused_tests", FOCUSED_TEST_COMMAND),
    _definition(2, "scoped_coverage", COVERAGE_COMMAND),
    _definition(3, "ruff_check", RUFF_COMMAND),
    _definition(4, "format_check", FORMAT_COMMAND),
    _definition(5, "spec_coverage", SPEC_COMMAND),
)

REQUIRED_ATTACK_TYPES = {
    "answer_permutation",
    "label_renaming",
    "grammar_only_generation",
    "trigger_collision",
    "premature_trigger",
    "missing_trigger",
    "malformed_tail",
    "unknown_fields",
    "semantically_wrong_syntactically_valid_tail",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "frozen_input_receipts",
    "owned_test_rows",
    "global_suite_diagnostic",
    "exact_checker_rows",
    "leakage_attack_rows",
    "triggered_tail_fixture_ready",
    "per_unit_rows",
    "aggregate_row_recomputation",
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

FIELD_PRINCIPLES = {
    "status": "The terminal state comes from deterministic process evidence.",
    "honest_verdict": "The verdict uses measured owned-scope evidence only.",
    "verdict_class": "A closed class keeps ready infrastructure null.",
    "gate_check_summary": "Exact failed values localize an owned defect.",
    "frozen_input_receipts": "Stable hashes preserve immutable fixture provenance.",
    "owned_test_rows": "Commands and nodes define the task-scoped verification boundary.",
    "global_suite_diagnostic": "Repository failures stay visible without becoming a gate.",
    "exact_checker_rows": "Positive and negative controls keep executable authority explicit.",
    "leakage_attack_rows": "Adversarial rows test syntax and semantic separation.",
    "triggered_tail_fixture_ready": "One Boolean reduces only complete owned checks.",
    "per_unit_rows": "Raw test, task, checker, and attack rows remain recheckable.",
    "aggregate_row_recomputation": "Readiness is rebuilt from retained rows.",
    "preconditions_checked": "Measured inputs and resources establish provenance.",
    "protected_files_unchanged": "Before and after hashes protect active operations.",
    "inference_substrate": "The declared CPU-only path prevents a model claim.",
    "verifier_is_oracle": "Exact fixtures define readiness and expose circularity.",
    "field_provenance": "Each field names its source, function, and hash.",
    "random_seed": "A fixed seed preserves deterministic attack order.",
    "duration_s": "Monotonic time records the measured run.",
    "tests_run": "Exact commands, exits, and summaries reproduce verification.",
    "reproducibility_checksum": "A canonical content hash detects artifact changes.",
}

PARSER_FUNCTIONS = (
    "_parse_natural",
    "_parse_json_tail",
    "_parse_immediate_json",
    "_parse_triggered_tail",
    "parse_arm_output",
)

FROZEN_FILE_PATHS = (
    EXP6661_ARTIFACT_PATH,
    EXP6661_SOURCE_PATH,
    EXP6661_TEST_PATH,
    EXP6661_SPEC_PATH,
    PARSER_SUPPORT_PATH,
    V582_DESIGN_PATH,
    *PROTECTED_PATHS,
    GLOBAL_CACHE_PATH,
)


def canonical_json(value: Any) -> str:
    """Serialize receipts without optional whitespace or unstable key order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed digest so hash identity cannot be mistaken for text."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON instead of interpreter-specific object text."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash one file and keep a missing input distinct from empty bytes."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def receipt_hash(value: Any, *, excluded: Sequence[str] = ()) -> str:
    """Hash one receipt after removing only named self-referential fields."""

    if isinstance(value, Mapping):
        ignored = set(excluded)
        value = {key: item for key, item in value.items() if key not in ignored}
    return sha256_json(value)


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every final artifact field except the checksum that stores the digest."""

    return receipt_hash(payload, excluded=("reproducibility_checksum",))


def load_json(path: Path) -> JsonDict:
    """Load one required JSON object without repairing malformed evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(value)


def _function_hash(function: Callable[..., Any]) -> str:
    return sha256_bytes(inspect.getsource(function).encode("utf-8"))


def parser_function_hashes() -> dict[str, str]:
    """Hash every parser function that turns transport bytes into a certificate."""

    return {name: _function_hash(getattr(fixture, name)) for name in PARSER_FUNCTIONS}


def checker_function_hashes() -> dict[str, str]:
    """Hash each family checker that supplies exact semantic authority."""

    return {family: fixture._checker_identity(family)["sha256"] for family in fixture.FAMILY_ORDER}


def replay_exp6661_fixture(root: Path = REPO_ROOT) -> JsonDict:
    """Replay frozen builders without changing any source corpus or result file."""

    del root
    manifest = fixture.build_frozen_task_manifest()
    arms = fixture.build_arm_contracts()
    grammar = fixture.build_syntax_only_grammar_receipt(manifest)
    fixture_rows = fixture.build_fixture_rows(manifest, arms)
    controls = fixture.build_exact_checker_rows(manifest)
    attacks = fixture.build_leakage_attack_rows(manifest, arms)
    aggregate = fixture.recompute_aggregate_rows(
        manifest=manifest,
        arm_contracts=arms,
        fixture_rows=fixture_rows,
        exact_checker_rows=controls,
        leakage_attack_rows=attacks,
    )
    return {
        "manifest": manifest,
        "arm_contracts": arms,
        "grammar": grammar,
        "fixture_rows": fixture_rows,
        "exact_checker_rows": controls,
        "leakage_attack_rows": attacks,
        "aggregate": aggregate,
    }


def capture_frozen_snapshot(
    root: Path = REPO_ROOT, *, replay: Mapping[str, Any] | None = None
) -> JsonDict:
    """Hash files and derived contracts before any task-owned command runs."""

    rows = dict(replay or replay_exp6661_fixture(root))
    return {
        "file_hashes": {path.as_posix(): sha256_file(root / path) for path in FROZEN_FILE_PATHS},
        "derived_hashes": {
            "manifest": sha256_json(rows["manifest"]),
            "arm_contracts": sha256_json(rows["arm_contracts"]),
            "fixture_rows": sha256_json(rows["fixture_rows"]),
            "grammar": rows["grammar"]["grammar_sha256"],
            "parser_hashes": sha256_json(parser_function_hashes()),
            "checker_hashes": sha256_json(checker_function_hashes()),
        },
    }


def _artifact_checker_hashes(source: Mapping[str, Any]) -> dict[str, str]:
    return {
        family: next(
            task["checker"]["sha256"]
            for task in source["frozen_task_manifest"]
            if task["family"] == family
        )
        for family in fixture.FAMILY_ORDER
    }


def build_frozen_input_receipts(
    root: Path,
    before: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> JsonDict:
    """Compare before/after bytes and replayed contracts with the frozen artifact."""

    source = load_json(root / EXP6661_ARTIFACT_PATH)
    after = capture_frozen_snapshot(root, replay=replay)
    recorded_inputs = source.get("preconditions_checked", {}).get("input_hashes", {})
    file_receipts = {
        path: {
            "before_sha256": before.get("file_hashes", {}).get(path),
            "after_sha256": after["file_hashes"].get(path),
            "unchanged": before.get("file_hashes", {}).get(path) == after["file_hashes"].get(path),
            "exp6661_recorded_sha256": recorded_inputs.get(path),
            "exp6661_recorded_matches_current": (
                recorded_inputs.get(path) == after["file_hashes"].get(path)
                if path in recorded_inputs
                else None
            ),
        }
        for path in before.get("file_hashes", {})
    }
    contract_matches = {
        "manifest": replay["manifest"] == source.get("frozen_task_manifest"),
        "arm_contracts": replay["arm_contracts"] == source.get("arm_contracts"),
        "fixture_rows": replay["fixture_rows"] == source.get("fixture_rows"),
        "grammar": (
            replay["grammar"].get("grammar_sha256")
            == source.get("syntax_only_grammar_receipt", {}).get("grammar_sha256")
            and replay["grammar"].get("proof_sha256")
            == source.get("syntax_only_grammar_receipt", {}).get("proof_sha256")
        ),
        "parser_hashes": (
            before.get("derived_hashes", {}).get("parser_hashes")
            == after["derived_hashes"]["parser_hashes"]
        ),
        "checker_hashes": checker_function_hashes() == _artifact_checker_hashes(source),
    }
    all_match = all(row["unchanged"] for row in file_receipts.values()) and all(
        contract_matches.values()
    )
    return {
        "schema": "carnot.experiment_6675.frozen_inputs.v1",
        "match_basis": (
            "Exp6675 before-and-after file identity plus equality of the frozen manifest, "
            "arms, fixture rows, grammar, parser functions, and checker functions"
        ),
        "exp6661_artifact_checksum_valid": (
            source.get("reproducibility_checksum") == fixture.artifact_checksum(source)
        ),
        "file_receipts": file_receipts,
        "derived_hashes_before": deepcopy(before.get("derived_hashes", {})),
        "derived_hashes_after": deepcopy(after["derived_hashes"]),
        "parser_hashes": parser_function_hashes(),
        "checker_hashes": checker_function_hashes(),
        "contract_matches": contract_matches,
        "all_hashes_match": all_match,
    }


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash the active roadmap and conductor before task work."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def protected_files_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    """Prove protected operational files stayed byte-identical."""

    after = protected_hashes(root)
    return {
        "before": dict(before),
        "after": after,
        "files": {
            path: {
                "before_sha256": before.get(path),
                "after_sha256": after.get(path),
                "unchanged": before.get(path) == after.get(path),
            }
            for path in sorted(set(before) | set(after))
        },
        "unchanged": bool(before) and dict(before) == after,
    }


def _ram_total_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def collect_preconditions(root: Path, frozen_before: Mapping[str, Any]) -> JsonDict:
    """Record input hashes, tools, host resources, and the exact no-LLM path."""

    disk = shutil.disk_usage(root)
    return {
        "schema": "carnot.experiment_6675.preconditions.v1",
        "planning_date": RUN_DATE,
        "root": str(root.resolve()),
        "frozen_hashes_before": deepcopy(frozen_before),
        "task_module_sha256": sha256_file(root / MODULE_PATH),
        "task_test_sha256": sha256_file(root / TEST_PATH),
        "reporting_spec_sha256": sha256_file(root / REPORT_SPEC_RELATIVE_PATH),
        "resources": {
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count() or 1,
            "ram_bytes": _ram_total_bytes(),
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "tools": {
            "pytest": (root / ".venv/bin/pytest").is_file(),
            "coverage": (root / ".venv/bin/coverage").is_file(),
            "ruff": (root / ".venv/bin/ruff").is_file(),
            "spec_coverage": (root / "scripts/check_spec_coverage.py").is_file(),
            "adversarial_verify": (root / "scripts/adversarial_verify.py").is_file(),
        },
        "e2e_plan": {
            "path": "ops/e2e-test-plan.md",
            "applicable_ids": [],
            "reason": (
                "The listed E2E cases exercise model, binding, and learning pipelines. "
                "This task replays a CPU fixture and exact checks only."
            ),
        },
        "no_llm": {
            "declared": INFERENCE_SUBSTRATE,
            "model_load_attempt_count": 0,
            "generation_attempt_count": 0,
            "exact_fixture_replay_only": True,
        },
    }


def _full_suite_source_receipt(root: Path) -> JsonDict:
    source = load_json(root / EXP6661_ARTIFACT_PATH)
    return dict(
        next(
            row
            for row in source.get("tests_run", [])
            if isinstance(row, Mapping) and row.get("command") == FULL_SUITE_COMMAND
        )
    )


def load_global_suite_diagnostic(
    root: Path = REPO_ROOT, *, cache_path: Path | None = None
) -> JsonDict:
    """Attribute every cached failure while keeping the global result non-gating."""

    path = cache_path or root / GLOBAL_CACHE_PATH
    try:
        cache = load_json(path)
        cache_error = None
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        cache = {}
        cache_error = f"{type(exc).__name__}: {exc}"
    nodes = sorted(str(node) for node in cache)
    owned = [node for node in nodes if node.startswith(OWNED_NODE_PREFIX + "::")]
    unrelated = [node for node in nodes if node not in set(owned)]
    source_receipt = _full_suite_source_receipt(root)
    diagnostic: JsonDict = {
        "schema": "carnot.experiment_6675.global_suite_diagnostic.v1",
        "command": FULL_SUITE_COMMAND,
        "exit_code": source_receipt.get("exit_code"),
        "exit_code_receipt_scope": (
            "Exp6661 recorded run; the current cache count and hash are measured separately"
        ),
        "summary": source_receipt.get("summary"),
        "failure_count": len(nodes),
        "exp6661_owned_failure_count": len(owned),
        "owned_failure_nodes": owned,
        "unrelated_failure_nodes": unrelated,
        "node_attribution_complete": len(nodes) == len(owned) + len(unrelated),
        "cache_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "cache_sha256": sha256_file(path),
        "cache_read_error": cache_error,
        "source_artifact": EXP6661_ARTIFACT_PATH.as_posix(),
        "source_receipt": source_receipt,
        "known_issue": "ops/known-issues.md:91",
        "gating": False,
        "non_gating_rationale": (
            "The repository suite contains unrelated nodes. Only exact Exp6661 test-file "
            "nodes can change this task's owned readiness."
        ),
    }
    diagnostic["receipt_sha256"] = receipt_hash(diagnostic)
    return diagnostic


def make_owned_test_row(
    definition: Mapping[str, Any],
    *,
    node_set: Sequence[str],
    exit_code: int,
    coverage_percent: float | None,
    duration_s: float,
    summary: str,
    output_sha256: str,
) -> JsonDict:
    """Bind one measured command to the exact 33-node Exp6661 test scope."""

    nodes = list(node_set)
    passed = (
        exit_code == definition.get("expected_exit_code")
        and len(nodes) == definition.get("expected_node_count")
        and all(node.startswith(OWNED_NODE_PREFIX + "::") for node in nodes)
        and (
            definition.get("expected_coverage_percent") is None
            or coverage_percent == definition.get("expected_coverage_percent")
        )
    )
    row: JsonDict = {
        "row_kind": "owned_test_command",
        "ordinal": definition.get("ordinal"),
        "check_id": definition.get("check_id"),
        "command": definition.get("command"),
        "expected_exit_code": definition.get("expected_exit_code"),
        "exit_code": exit_code,
        "node_set": nodes,
        "node_count": len(nodes),
        "expected_node_count": definition.get("expected_node_count"),
        "coverage_percent": coverage_percent,
        "expected_coverage_percent": definition.get("expected_coverage_percent"),
        "duration_s": round(float(duration_s), 6),
        "summary": summary,
        "output_sha256": output_sha256,
        "passed": passed,
    }
    row["receipt_sha256"] = receipt_hash(row)
    return row


def reduce_owned_test_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], JsonDict]:
    """Reduce exact ordered command receipts without accepting missing values."""

    row_list = [dict(row) for row in rows]
    counts = Counter(row.get("check_id") for row in row_list)
    failures: list[JsonDict] = []
    for index, definition in enumerate(OWNED_CHECK_DEFINITIONS):
        check_id = definition["check_id"]
        matches = [row for row in row_list if row.get("check_id") == check_id]
        observed = None
        if not matches:
            reason = "missing_receipt"
        else:
            row = matches[0]
            observed = row.get("passed")
            if counts[check_id] != 1:
                reason = "duplicate_receipt"
            elif index >= len(row_list) or row_list[index].get("check_id") != check_id:
                reason = "receipt_order_mismatch"
            elif any(
                row.get(key) != definition.get(key)
                for key in (
                    "ordinal",
                    "check_id",
                    "command",
                    "expected_exit_code",
                    "expected_node_count",
                    "expected_coverage_percent",
                )
            ):
                reason = "definition_mismatch"
            elif row.get("receipt_sha256") != receipt_hash(row, excluded=("receipt_sha256",)):
                reason = "receipt_hash_mismatch"
            elif row.get("passed") is not True:
                reason = "observed_value_mismatch"
            else:
                reason = None
        if reason is not None:
            failures.append(
                {
                    "check": check_id,
                    "expected_value": True,
                    "observed_value": observed,
                    "reason": reason,
                }
            )
    focused = next((row for row in row_list if row.get("check_id") == "focused_tests"), {})
    coverage = next((row for row in row_list if row.get("check_id") == "scoped_coverage"), {})
    return failures, {
        "ready": not failures and len(row_list) == len(OWNED_CHECK_DEFINITIONS),
        "row_count": len(row_list),
        "node_count": focused.get("node_count"),
        "node_set_sha256": sha256_json(focused.get("node_set", [])),
        "coverage_percent": coverage.get("coverage_percent"),
        "failed_checks": failures,
    }


def default_command_runner(command: list[str], cwd: Path) -> JsonDict:
    """Run one command and retain output identity plus monotonic duration."""

    started = time.monotonic()
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    lines = output.splitlines()
    return {
        "command": " ".join(command),
        "exit_code": proc.returncode,
        "output": output,
        "summary": "\n".join(lines[-12:]) if lines else "no output",
        "output_sha256": sha256_bytes(output.encode("utf-8")),
        "duration_s": round(time.monotonic() - started, 6),
    }


def _command_argv(command: str) -> list[str]:
    return command.split()


def _collected_nodes(receipt: Mapping[str, Any]) -> list[str]:
    return [
        line
        for line in str(receipt.get("output", "")).splitlines()
        if line.startswith(OWNED_NODE_PREFIX + "::")
    ]


def _coverage_percent(output: str) -> float | None:
    match = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%", output, re.MULTILINE)
    return float(match.group(1)) if match else None


def run_owned_verification(
    root: Path = REPO_ROOT, *, command_runner: CommandRunner | None = None
) -> list[JsonDict]:
    """Run the exact five recorded checks and attach the collected 33-node set."""

    runner = command_runner or default_command_runner
    collection = runner(_command_argv(COLLECT_COMMAND), root)
    nodes = _collected_nodes(collection) if collection.get("exit_code") == 0 else []
    rows = []
    for definition in OWNED_CHECK_DEFINITIONS:
        receipt = runner(_command_argv(str(definition["command"])), root)
        coverage = (
            _coverage_percent(str(receipt.get("output", "")))
            if definition["check_id"] == "scoped_coverage"
            else None
        )
        row = make_owned_test_row(
            definition,
            node_set=nodes,
            exit_code=int(receipt.get("exit_code", -1)),
            coverage_percent=coverage,
            duration_s=float(receipt.get("duration_s", 0.0)),
            summary=str(receipt.get("summary", "no output")),
            output_sha256=str(receipt.get("output_sha256", "missing")),
        )
        row["collection_command"] = COLLECT_COMMAND
        row["collection_exit_code"] = collection.get("exit_code")
        row["collection_output_sha256"] = collection.get("output_sha256")
        row["receipt_sha256"] = receipt_hash(row, excluded=("receipt_sha256",))
        rows.append(row)
    return rows


def _test_unit_rows(owned_test_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    focused = next((row for row in owned_test_rows if row.get("check_id") == "focused_tests"), {})
    return [
        {
            "row_kind": "test",
            "node_id": node,
            "expected": "pass",
            "observed": "pass" if focused.get("passed") is True else "failed_command",
            "passed": focused.get("passed") is True,
            "command_receipt_sha256": focused.get("receipt_sha256"),
            "row_sha256": sha256_json(
                {
                    "node_id": node,
                    "passed": focused.get("passed") is True,
                    "command_receipt_sha256": focused.get("receipt_sha256"),
                }
            ),
        }
        for node in focused.get("node_set", [])
    ]


def build_per_unit_rows(
    replay: Mapping[str, Any], owned_test_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Retain every test, task, checker control, and attack as a raw unit."""

    return [
        *_test_unit_rows(owned_test_rows),
        *({"row_kind": "task", **deepcopy(dict(task))} for task in replay["manifest"]),
        *(deepcopy(dict(row)) for row in replay["exact_checker_rows"]),
        *(deepcopy(dict(row)) for row in replay["leakage_attack_rows"]),
    ]


def _aggregate_rows(
    *,
    owned_test_rows: Sequence[Mapping[str, Any]],
    global_diagnostic: Mapping[str, Any],
    frozen_receipts: Mapping[str, Any],
    replay: Mapping[str, Any],
    per_unit_rows: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
) -> JsonDict:
    owned_failures, owned_summary = reduce_owned_test_rows(owned_test_rows)
    kinds = Counter(row.get("row_kind") for row in per_unit_rows)
    fixture_aggregate = replay["aggregate"]
    checks = {
        "owned_tests": owned_summary["ready"] is True,
        "frozen_inputs": frozen_receipts.get("all_hashes_match") is True,
        "global_diagnostic_available": (
            global_diagnostic.get("cache_read_error") is None
            and global_diagnostic.get("node_attribution_complete") is True
        ),
        "global_owned_failures": global_diagnostic.get("exp6661_owned_failure_count") == 0,
        "manifest_rows": fixture_aggregate.get("checks", {}).get("manifest_task_count") is True,
        "arm_contracts": fixture_aggregate.get("checks", {}).get("arm_contract_keys") is True,
        "fixture_rows": (
            fixture_aggregate.get("checks", {}).get("fixture_row_keys") is True
            and fixture_aggregate.get("checks", {}).get("fixture_outcomes") is True
        ),
        "exact_checker_rows": (
            fixture_aggregate.get("checks", {}).get("checker_control_keys") is True
            and fixture_aggregate.get("checks", {}).get("checker_control_outcomes") is True
        ),
        "leakage_attack_rows": (
            fixture_aggregate.get("checks", {}).get("attack_row_keys") is True
            and fixture_aggregate.get("checks", {}).get("attack_outcomes") is True
            and REQUIRED_ATTACK_TYPES
            <= {row.get("attack_type") for row in replay["leakage_attack_rows"]}
        ),
        "semantic_free_grammar": replay["grammar"].get("answer_semantics_absent") is True,
        "per_unit_rows": kinds
        == Counter({"test": 33, "task": 18, "checker_control": 36, "leakage_attack": 540}),
        "protected_files": protected.get("unchanged") is True,
    }
    observed = {
        "owned_tests": owned_failures,
        "frozen_inputs": frozen_receipts.get("all_hashes_match"),
        "global_diagnostic_available": global_diagnostic.get("cache_read_error"),
        "global_owned_failures": global_diagnostic.get("exp6661_owned_failure_count"),
        "manifest_rows": len(replay["manifest"]),
        "arm_contracts": len(replay["arm_contracts"]),
        "fixture_rows": len(replay["fixture_rows"]),
        "exact_checker_rows": len(replay["exact_checker_rows"]),
        "leakage_attack_rows": len(replay["leakage_attack_rows"]),
        "semantic_free_grammar": replay["grammar"].get("answer_semantics_absent"),
        "per_unit_rows": dict(kinds),
        "protected_files": protected.get("unchanged"),
    }
    failed = [
        {"check": check, "expected_value": True, "observed_value": observed[check]}
        for check, passed in checks.items()
        if not passed
    ]
    return {
        "ready": not failed,
        "checks": checks,
        "failed_checks": failed,
        "owned_test_recomputation": owned_summary,
        "exp6661_fixture_recomputation": deepcopy(fixture_aggregate),
        "counts": {
            "tasks": len(replay["manifest"]),
            "arm_contracts": len(replay["arm_contracts"]),
            "fixture_rows": len(replay["fixture_rows"]),
            "checker_controls": len(replay["exact_checker_rows"]),
            "attack_rows": len(replay["leakage_attack_rows"]),
            "owned_test_rows": len(owned_test_rows),
            "owned_test_nodes": kinds.get("test", 0),
            "global_failures": global_diagnostic.get("failure_count"),
            "global_owned_failures": global_diagnostic.get("exp6661_owned_failure_count"),
            "per_unit_rows": len(per_unit_rows),
        },
    }


def _field_provenance(
    root: Path, global_diagnostic: Mapping[str, Any]
) -> dict[str, JsonDict]:
    module_hash = sha256_file(root / MODULE_PATH)
    functions = {
        "frozen_input_receipts": "build_frozen_input_receipts",
        "owned_test_rows": "run_owned_verification",
        "global_suite_diagnostic": "load_global_suite_diagnostic",
        "exact_checker_rows": "fixture.build_exact_checker_rows",
        "leakage_attack_rows": "fixture.build_leakage_attack_rows",
        "per_unit_rows": "build_per_unit_rows",
        "aggregate_row_recomputation": "_aggregate_rows",
        "preconditions_checked": "collect_preconditions",
        "protected_files_unchanged": "protected_files_receipt",
        "tests_run": "run_owned_verification",
    }
    source_paths = {
        "frozen_input_receipts": EXP6661_ARTIFACT_PATH.as_posix(),
        "global_suite_diagnostic": GLOBAL_CACHE_PATH.as_posix(),
        "exact_checker_rows": EXP6661_SOURCE_PATH.as_posix(),
        "leakage_attack_rows": EXP6661_SOURCE_PATH.as_posix(),
    }
    rows = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        source_path = source_paths.get(field, MODULE_PATH.as_posix())
        source_hash = sha256_file(root / source_path)
        if field == "global_suite_diagnostic":
            source_hash = str(global_diagnostic.get("cache_sha256", "missing"))
        rows[field] = {
            "source_path": source_path,
            "function": functions.get(field, "build_artifact"),
            "sha256": source_hash if source_hash != "missing" else module_hash,
            "principle": FIELD_PRINCIPLES[field],
        }
    return rows


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    date: str = RUN_DATE,
    duration_s: float,
    owned_test_rows: Sequence[Mapping[str, Any]],
    global_suite_diagnostic: Mapping[str, Any],
    frozen_before: Mapping[str, Any],
    protected_before: Mapping[str, str],
    replay: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal receipt from fresh rows and a non-gating diagnostic."""

    replayed = dict(replay or replay_exp6661_fixture(root))
    owned_rows = [deepcopy(dict(row)) for row in owned_test_rows]
    global_diagnostic = deepcopy(dict(global_suite_diagnostic))
    frozen = build_frozen_input_receipts(root, frozen_before, replayed)
    protected = protected_files_receipt(root, protected_before)
    per_unit = build_per_unit_rows(replayed, owned_rows)
    aggregate = _aggregate_rows(
        owned_test_rows=owned_rows,
        global_diagnostic=global_diagnostic,
        frozen_receipts=frozen,
        replay=replayed,
        per_unit_rows=per_unit,
        protected=protected,
    )
    ready = aggregate["ready"] is True
    first_failure = aggregate["failed_checks"][0]["check"] if not ready else None
    artifact: JsonDict = {
        "schema": "carnot.experiment_6675.triggered_tail_scope_receipt.v1",
        "run_date": date,
        "spec_traces": [
            "REQ-REPORT-6675",
            "SCENARIO-REPORT-6675-OWNED-READY",
            "SCENARIO-REPORT-6675-GLOBAL-DIAGNOSTIC",
            "SCENARIO-REPORT-6675-FAIL-CLOSED",
            "SCENARIO-REPORT-6675-ATOMIC-PROVENANCE",
        ],
        "status": "complete_ready" if ready else "blocked_owned_fixture_check",
        "honest_verdict": (
            "complete: triggered-tail fixture is ready under task-owned checks; "
            "the repository-suite failures remain a non-gating diagnostic"
            if ready
            else f"blocked_triggered_tail_scope_receipt: {first_failure} failed"
        ),
        "verdict_class": "null" if ready else "blocked",
        "gate_check_summary": deepcopy(aggregate["failed_checks"]),
        "frozen_input_receipts": frozen,
        "owned_test_rows": owned_rows,
        "global_suite_diagnostic": global_diagnostic,
        "frozen_task_manifest": deepcopy(replayed["manifest"]),
        "arm_contracts": deepcopy(replayed["arm_contracts"]),
        "syntax_only_grammar_receipt": deepcopy(replayed["grammar"]),
        "fixture_rows": deepcopy(replayed["fixture_rows"]),
        "exact_checker_rows": deepcopy(replayed["exact_checker_rows"]),
        "leakage_attack_rows": deepcopy(replayed["leakage_attack_rows"]),
        "triggered_tail_fixture_ready": ready,
        "per_unit_rows": per_unit,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": collect_preconditions(root, frozen_before),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(root, global_diagnostic),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 6),
        "tests_run": deepcopy(owned_rows),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _replay_from_payload(payload: Mapping[str, Any]) -> JsonDict:
    aggregate = fixture.recompute_aggregate_rows(
        manifest=payload["frozen_task_manifest"],
        arm_contracts=payload["arm_contracts"],
        fixture_rows=payload["fixture_rows"],
        exact_checker_rows=payload["exact_checker_rows"],
        leakage_attack_rows=payload["leakage_attack_rows"],
    )
    return {
        "manifest": payload["frozen_task_manifest"],
        "arm_contracts": payload["arm_contracts"],
        "grammar": payload["syntax_only_grammar_receipt"],
        "fixture_rows": payload["fixture_rows"],
        "exact_checker_rows": payload["exact_checker_rows"],
        "leakage_attack_rows": payload["leakage_attack_rows"],
        "aggregate": aggregate,
    }


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Reject boundary drift, row tampering, and inconsistent readiness."""

    if any(field not in payload for field in REQUIRED_ARTIFACT_FIELDS):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    owned_failures, _owned_summary = reduce_owned_test_rows(payload["owned_test_rows"])
    if owned_failures and payload.get("triggered_tail_fixture_ready") is True:
        errors.append("owned_test_receipts_invalid")
    global_diagnostic = payload["global_suite_diagnostic"]
    if global_diagnostic.get("gating") is not False:
        errors.append("global_diagnostic_gating")
    if global_diagnostic.get("receipt_sha256") != receipt_hash(
        global_diagnostic, excluded=("receipt_sha256",)
    ):
        errors.append("global_diagnostic_hash_mismatch")
    if global_diagnostic.get("failure_count") != len(
        global_diagnostic.get("owned_failure_nodes", [])
    ) + len(global_diagnostic.get("unrelated_failure_nodes", [])):
        errors.append("global_diagnostic_count_mismatch")
    if global_diagnostic.get("exp6661_owned_failure_count") != len(
        global_diagnostic.get("owned_failure_nodes", [])
    ):
        errors.append("global_owned_count_mismatch")
    if payload["frozen_input_receipts"].get("all_hashes_match") is not True:
        errors.append("frozen_hash_mismatch")
    if payload["protected_files_unchanged"].get("unchanged") is not True:
        errors.append("protected_files_changed")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    provenance = payload["field_provenance"]
    if set(REQUIRED_ARTIFACT_FIELDS) - set(provenance) or any(
        not {"source_path", "function", "sha256", "principle"} <= set(row)
        for row in provenance.values()
    ):
        errors.append("field_provenance_invalid")
    try:
        replay = _replay_from_payload(payload)
        recomputed = _aggregate_rows(
            owned_test_rows=payload["owned_test_rows"],
            global_diagnostic=global_diagnostic,
            frozen_receipts=payload["frozen_input_receipts"],
            replay=replay,
            per_unit_rows=payload["per_unit_rows"],
            protected=payload["protected_files_unchanged"],
        )
    except (KeyError, TypeError, ValueError):
        recomputed = {"ready": False, "failed_checks": []}
        errors.append("aggregate_row_recomputation_failed")
    if recomputed != payload["aggregate_row_recomputation"]:
        errors.append("aggregate_row_recomputation_mismatch")
    ready = recomputed.get("ready") is True
    if payload.get("triggered_tail_fixture_ready") is not ready:
        errors.append("readiness_mismatch")
    expected_status = "complete_ready" if ready else "blocked_owned_fixture_check"
    if payload.get("status") != expected_status:
        errors.append("status_mismatch")
    expected_class = "null" if ready else "blocked"
    if payload.get("verdict_class") != expected_class:
        errors.append("verdict_class_mismatch")
    expected_summary = recomputed.get("failed_checks", [])
    if payload.get("gate_check_summary") != expected_summary:
        errors.append("gate_check_summary_mismatch")
    if ready and not str(payload.get("honest_verdict", "")).startswith("complete:"):
        errors.append("honest_verdict_mismatch")
    if not ready and not str(payload.get("honest_verdict", "")).startswith("blocked_"):
        errors.append("honest_verdict_mismatch")
    if payload.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload["duration_s"] < 0:
        errors.append("duration_invalid")
    return list(dict.fromkeys(errors))


def write_artifact_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Sync complete JSON before one same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    owned_test_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Measure owned checks, validate the receipt, and write it atomically."""

    started = time.monotonic()
    replay = replay_exp6661_fixture(root)
    frozen_before = capture_frozen_snapshot(root, replay=replay)
    protected_before = protected_hashes(root)
    measured_rows = (
        [deepcopy(dict(row)) for row in owned_test_rows]
        if owned_test_rows is not None
        else run_owned_verification(root)
    )
    diagnostic = load_global_suite_diagnostic(root)
    artifact = build_artifact(
        root=root,
        date=date,
        duration_s=time.monotonic() - started,
        owned_test_rows=measured_rows,
        global_suite_diagnostic=diagnostic,
        frozen_before=frozen_before,
        protected_before=protected_before,
        replay=replay,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp6675 artifact: " + ",".join(errors))
    write_artifact_atomic(output_path or root / RESULT_PATH, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the receipt or validate an existing artifact without changing it."""

    args = _parse_args(argv)
    target = args.output or REPO_ROOT / RESULT_PATH
    if args.validate:
        if not target.is_file():
            print(json.dumps({"valid": False, "errors": ["artifact_missing"]}, sort_keys=True))
            return 1
        try:
            artifact = load_json(target)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(
                json.dumps(
                    {"valid": False, "errors": [f"artifact_unreadable:{type(exc).__name__}"]},
                    sort_keys=True,
                )
            )
            return 1
        errors = validate_artifact(artifact)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(date=args.date, output_path=target)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "triggered_tail_fixture_ready": artifact["triggered_tail_fixture_ready"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module command.
    raise SystemExit(main())
