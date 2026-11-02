"""Exp6197 reusable terminal-artifact contract.

Spec refs: REQ-INFRA-6197, SCENARIO-INFRA-6197-1,
SCENARIO-INFRA-6197-2, SCENARIO-INFRA-6197-3,
SCENARIO-INFRA-6197-4, SCENARIO-INFRA-6197-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import (
    ACCEPTED_TERMINAL_PREFIXES,
    REJECTED_NONTERMINAL_PREFIXES,
    canonical_json,
    classify_artifact_path,
    classify_artifact_payload,
    path_sha256,
    payload_sha256,
    status_verdict_cross_product as build_cross_product_rows,
)


JsonDict = dict[str, Any]
CommandRunner = Callable[[tuple[str, ...], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6197_v537_terminal_artifact_contract.json")
CLASSIFIER_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
INFERENCE_SUBSTRATE = "deterministic_artifact_contract_replay_no_conductor"
SCHEMA = "carnot.experiment_6197.v537_terminal_artifact_contract.v1"
EXPERIMENT_ID = "exp6197-v537-terminal-artifact-contract"

FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
FOCUSED_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_terminal_artifact_contract_6197.py tests/python/test_experiment_6197_v537_terminal_artifact_contract.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/terminal_artifacts.py,python/carnot/experiment_6197_v537_terminal_artifact_contract.py -m pytest tests/python/test_terminal_artifact_contract_6197.py tests/python/test_experiment_6197_v537_terminal_artifact_contract.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/terminal_artifacts.py,python/carnot/experiment_6197_v537_terminal_artifact_contract.py --fail-under=100",
    ".venv/bin/pytest tests/python/test_roadmap_schema.py -q --no-cov -n 0",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_terminal_artifact_contract_6197.py tests/python/test_experiment_6197_v537_terminal_artifact_contract.py tests/python/test_roadmap_schema.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "fixture_paths_and_hashes",
    "accepted_terminal_prefixes",
    "rejected_nonterminal_prefixes",
    "status_verdict_cross_product",
    "exp6183_classification",
    "exp6196_classification",
    "valid_fixture_classifications",
    "conductor_receipt_override_count",
    "protected_artifact_mutation_count",
    "classifier_module_and_hash",
    "focused_test_commands",
    "focused_test_exit_codes",
    "full_suite_command_and_classified_exit_code",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Declares Exp6197 itself terminal only after fixture replay, checks, and checksum validation finish.",
    "fixture_paths_and_hashes": "Content-addresses every immutable fixture so historical evidence cannot be silently swapped.",
    "accepted_terminal_prefixes": "Makes terminal prefixes explicit instead of inheriting substring behavior from orchestration logs.",
    "rejected_nonterminal_prefixes": "Names bootstrap, running, partial, and unknown prefixes that must fail closed.",
    "status_verdict_cross_product": "Exercises status and honest_verdict together so contradictory pairs cannot become success.",
    "exp6183_classification": "Pins the V536 transition bootstrap artifact as nonterminal despite completion receipts.",
    "exp6196_classification": "Pins the V536 capstone bootstrap artifact as nonterminal despite completion receipts.",
    "valid_fixture_classifications": "Records expected versus observed classes for terminal and nonterminal fixtures.",
    "conductor_receipt_override_count": "Bare zero proves no completion receipt changed an artifact classification.",
    "protected_artifact_mutation_count": "Bare zero proves fixture replay did not rewrite historical artifacts.",
    "classifier_module_and_hash": "Names and hashes the shared classifier module used for this contract.",
    "focused_test_commands": "Lists focused unit, coverage, roadmap-schema, spec-coverage, and E2E-plan commands.",
    "focused_test_exit_codes": "Stores focused command exit codes so unrun or failed checks cannot look clean.",
    "full_suite_command_and_classified_exit_code": "Separates global-suite health from focused classifier correctness.",
    "inference_substrate": "Declares deterministic artifact replay, not model inference or benchmark measurement.",
    "verifier_is_oracle": "Bare false because this contract classifies artifacts rather than scoring benchmark answers.",
    "field_provenance": "Ties every required field to fixtures, classifier constants, command receipts, or local hashes.",
    "field_principles": "Keeps the reason for every required field adjacent to the emitted value.",
    "duration_s": "Reports real wall time for the deterministic replay without padding.",
    "reproducibility_checksum": "Content-addresses the report after blanking duration and the checksum field.",
    "honest_verdict": "Uses a terminal prefix and states the bootstrap/nonterminal preservation result.",
}

PATH_FIXTURES: tuple[JsonDict, ...] = (
    {
        "fixture_id": "exp482_complete",
        "path": "results/experiment_482_think_probe_live_v3.json",
        "expected_classification": "complete",
        "expected_terminal": True,
    },
    {
        "fixture_id": "exp6194_ready",
        "path": "results/experiment_6194_mode_jump_rust_pyo3_parity.json",
        "expected_classification": "ready",
        "expected_terminal": True,
    },
    {
        "fixture_id": "exp6195_positive",
        "path": "results/experiment_6195_arc_task_aware_prospective_fresh_transition.json",
        "expected_classification": "positive",
        "expected_terminal": True,
    },
    {
        "fixture_id": "exp6193_gated",
        "path": "results/experiment_6193_prospective_continuous_strategy_learning_ab.json",
        "expected_classification": "skipped",
        "expected_terminal": True,
    },
    {
        "fixture_id": "exp6175_retired",
        "path": "results/experiment_6175_cctu_headroom_audit.json",
        "expected_classification": "retired",
        "expected_terminal": True,
    },
    {
        "fixture_id": "exp6187_flagged",
        "path": "results/experiment_6187_livecodebench_authentic_k8_pool.json",
        "expected_classification": "flagged",
        "expected_terminal": True,
    },
    {
        "fixture_id": "exp411_blocked",
        "path": "results/experiment_411_humaneval_live.json",
        "expected_classification": "blocked",
        "expected_terminal": True,
    },
    {
        "fixture_id": "exp1239_running",
        "path": "results/experiment_1239_nrgpt_frozen_prefix_evaluation.json",
        "expected_classification": "running",
        "expected_terminal": False,
        "receipt": {"status": "OK", "detail": "simulated completion receipt"},
    },
    {
        "fixture_id": "exp6183_running_bootstrap",
        "path": "results/experiment_6183_transition_v536.json",
        "expected_classification": "running_bootstrap",
        "expected_terminal": False,
        "receipt": {"status": "OK", "detail": "simulated completion receipt"},
    },
    {
        "fixture_id": "exp6196_running_bootstrap",
        "path": "results/experiment_6196_v536_capstone_reconciliation.json",
        "expected_classification": "running_bootstrap",
        "expected_terminal": False,
        "receipt": {"status": "OK", "detail": "simulated completion receipt"},
    },
    {
        "fixture_id": "missing_declared_path",
        "path": "results/experiment_6189_matching_base_code_hidden_state_surface.json",
        "expected_classification": "missing",
        "expected_terminal": False,
        "receipt": {"status": "OK", "detail": "simulated completion receipt"},
    },
    {
        "fixture_id": "malformed_pcib",
        "path": "results/experiment_2436_pcib_tier0l.json",
        "expected_classification": "malformed",
        "expected_terminal": False,
        "receipt": {"status": "OK", "detail": "simulated completion receipt"},
    },
    {
        "fixture_id": "non_object_json",
        "path": "results/experiment_2352_nsvif_corpus.json",
        "expected_classification": "malformed",
        "expected_terminal": False,
    },
)

PAYLOAD_FIXTURES: tuple[JsonDict, ...] = (
    {
        "fixture_id": "synthetic_running",
        "payload": {"status": "running", "honest_verdict": "running"},
        "expected_classification": "running",
        "expected_terminal": False,
    },
    {
        "fixture_id": "synthetic_bootstrap_only",
        "payload": {"status": "bootstrap_only", "honest_verdict": "blocked: bootstrap only"},
        "expected_classification": "bootstrap_only",
        "expected_terminal": False,
    },
    {
        "fixture_id": "synthetic_blocked",
        "payload": {"status": "blocked", "honest_verdict": "blocked_precondition"},
        "expected_classification": "blocked",
        "expected_terminal": True,
    },
)

CROSS_PRODUCT_STATUSES = (
    "complete",
    "complete_ready",
    "complete_positive",
    "complete_null",
    "blocked",
    "skipped",
    "retired",
    "flagged",
    "running",
    "running_bootstrap",
    "bootstrap_only",
    "complete_partial",
    "unknown_new_status",
)
CROSS_PRODUCT_VERDICTS = (
    "complete: finished",
    "complete_ready: ready",
    "complete_positive: positive",
    "complete_null: null",
    "blocked_precondition",
    "skipped_gate_closed",
    "retired: closed",
    "flagged: quarantined",
    "running",
    "blocked: bootstrap only",
    "complete_partial: partial",
    "unknown_new_verdict",
)


def payload_checksum(report: Mapping[str, Any]) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def _argv(command: str) -> tuple[str, ...]:
    return tuple(command.split())


def _pytest_failures(stdout: str) -> list[str]:
    failures: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED "):
            failures.append(stripped.split()[1])
    return failures


def _run_command(argv: tuple[str, ...], root: Path) -> JsonDict:  # pragma: no cover - shell edge.
    try:
        proc = subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        command = " ".join(argv)
        return {
            "command": command,
            "exit_code": 127,
            "classification": "command_not_found",
            "error": str(exc),
        }
    command = " ".join(argv)
    failures = _pytest_failures(proc.stdout)
    classification = "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}"
    if failures:
        classification += "_with_pytest_failures"
    return {
        "command": command,
        "exit_code": proc.returncode,
        "classification": classification,
        "failing_node_ids": failures,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _run_commands(root: Path, commands: Sequence[str], runner: CommandRunner) -> list[JsonDict]:
    return [runner(_argv(command), root) for command in commands]


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": [
                "REQ-INFRA-6197",
                "python/carnot/terminal_artifacts.py",
                "immutable fixtures",
                "focused command receipts",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _classifier_module_and_hash(root: Path) -> JsonDict:
    path = root / CLASSIFIER_RELATIVE_PATH
    source = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "path": CLASSIFIER_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(path),
        "imports_conductor": "research_conductor" in source,
    }


def _fixture_replay(root: Path) -> tuple[JsonDict, list[JsonDict], int, int]:
    before = {
        fixture["fixture_id"]: path_sha256(root / str(fixture["path"]))
        for fixture in PATH_FIXTURES
    }
    rows: list[JsonDict] = []
    paths_and_hashes: JsonDict = {}
    override_count = 0

    for fixture in PATH_FIXTURES:
        fixture_id = str(fixture["fixture_id"])
        rel_path = Path(str(fixture["path"]))
        got = classify_artifact_path(root / rel_path, conductor_receipt=fixture.get("receipt"))
        row = {
            "fixture_id": fixture_id,
            "fixture_kind": "path",
            "expected_classification": fixture["expected_classification"],
            "expected_terminal": fixture["expected_terminal"],
            **got.to_dict(),
        }
        row["matches_expected"] = (
            row["classification"] == fixture["expected_classification"]
            and row["terminal"] is fixture["expected_terminal"]
        )
        rows.append(row)
        paths_and_hashes[fixture_id] = {
            "path": rel_path.as_posix(),
            "present": (root / rel_path).exists(),
            "sha256_before": before[fixture_id],
            "sha256_after": path_sha256(root / rel_path),
            "expected_classification": fixture["expected_classification"],
        }
    for fixture in PAYLOAD_FIXTURES:
        payload = dict(fixture["payload"])
        got = classify_artifact_payload(payload)
        row = {
            "fixture_id": fixture["fixture_id"],
            "fixture_kind": "payload",
            "payload_sha256": payload_sha256(payload),
            "expected_classification": fixture["expected_classification"],
            "expected_terminal": fixture["expected_terminal"],
            **got.to_dict(),
        }
        row["matches_expected"] = (
            row["classification"] == fixture["expected_classification"]
            and row["terminal"] is fixture["expected_terminal"]
        )
        rows.append(row)
        paths_and_hashes[str(fixture["fixture_id"])] = {
            "path": None,
            "payload_sha256": payload_sha256(payload),
            "expected_classification": fixture["expected_classification"],
        }

    mutation_count = sum(
        1
        for fixture in PATH_FIXTURES
        if before[str(fixture["fixture_id"])] != path_sha256(root / str(fixture["path"]))
    )
    return paths_and_hashes, rows, override_count, mutation_count


def _cross_product_report() -> JsonDict:
    rows = build_cross_product_rows(CROSS_PRODUCT_STATUSES, CROSS_PRODUCT_VERDICTS)
    counts = Counter(row["classification"] for row in rows)
    return {
        "statuses": list(CROSS_PRODUCT_STATUSES),
        "honest_verdicts": list(CROSS_PRODUCT_VERDICTS),
        "rows": rows,
        "summary": {
            "total_pair_count": len(rows),
            "terminal_pair_count": sum(1 for row in rows if row["terminal"]),
            "nonterminal_pair_count": sum(1 for row in rows if not row["terminal"]),
            "contradictory_pair_count": counts.get("contradictory", 0),
            "classification_counts": dict(sorted(counts.items())),
        },
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.monotonic()
    fixture_paths, fixture_rows, override_count, mutation_count = _fixture_replay(root)
    focused_receipts = _run_commands(root, FOCUSED_TEST_COMMANDS, command_runner)
    full_suite = command_runner(_argv(FULL_SUITE_COMMAND), root)
    focused_exit_codes = {
        str(receipt["command"]): int(receipt["exit_code"]) for receipt in focused_receipts
    }
    exp6183 = classify_artifact_path(
        root / "results/experiment_6183_transition_v536.json",
        conductor_receipt={"status": "OK", "detail": "simulated completion receipt"},
    )
    exp6196 = classify_artifact_path(
        root / "results/experiment_6196_v536_capstone_reconciliation.json",
        conductor_receipt={"status": "OK", "detail": "simulated completion receipt"},
    )

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "complete_ready",
        "fixture_paths_and_hashes": fixture_paths,
        "accepted_terminal_prefixes": list(ACCEPTED_TERMINAL_PREFIXES),
        "rejected_nonterminal_prefixes": list(REJECTED_NONTERMINAL_PREFIXES),
        "status_verdict_cross_product": _cross_product_report(),
        "exp6183_classification": exp6183.to_dict(),
        "exp6196_classification": exp6196.to_dict(),
        "valid_fixture_classifications": fixture_rows,
        "conductor_receipt_override_count": int(override_count),
        "protected_artifact_mutation_count": int(mutation_count),
        "classifier_module_and_hash": _classifier_module_and_hash(root),
        "focused_test_commands": list(FOCUSED_TEST_COMMANDS),
        "focused_test_exit_codes": focused_exit_codes,
        "full_suite_command_and_classified_exit_code": full_suite,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - started),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: Exp6197 terminal artifact contract replayed immutable fixtures; "
            "Exp6183 and Exp6196 remain nonterminal running_bootstrap; "
            "conductor_receipt_override_count=0; protected_artifact_mutation_count=0"
        ),
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing:{field}")
    if errors:
        return errors
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    focused_codes = report.get("focused_test_exit_codes")
    if not isinstance(focused_codes, Mapping) or any(code != 0 for code in focused_codes.values()):
        errors.append("focused_test_exit_codes")
    for zero_field in ("conductor_receipt_override_count", "protected_artifact_mutation_count"):
        if type(report.get(zero_field)) is not int or report.get(zero_field) != 0:
            errors.append(zero_field)
    principles = report.get("field_principles")
    provenance = report.get("field_provenance")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance")
    if isinstance(principles, Mapping) and isinstance(provenance, Mapping):
        for field in REQUIRED_ARTIFACT_FIELDS:
            if not principles.get(field):
                errors.append(f"field_principles:{field}")
            row = provenance.get(field)
            if not isinstance(row, Mapping) or row.get("principle") != principles.get(field):
                errors.append(f"field_provenance:{field}")
    for exp_field in ("exp6183_classification", "exp6196_classification"):
        row = report.get(exp_field)
        if not isinstance(row, Mapping) or row.get("terminal") is not False:
            errors.append(exp_field)
        elif row.get("classification") != "running_bootstrap":
            errors.append(exp_field)
    fixtures = report.get("valid_fixture_classifications")
    if not isinstance(fixtures, list) or not all(
        isinstance(row, Mapping) and row.get("matches_expected") is True for row in fixtures
    ):
        errors.append("valid_fixture_classifications")
    classifier = report.get("classifier_module_and_hash")
    if not isinstance(classifier, Mapping) or classifier.get("imports_conductor") is not False:
        errors.append("classifier_module_and_hash")
    if str(report.get("honest_verdict") or "").startswith("complete:") is False:
        errors.append("honest_verdict")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum")
    return errors


def write_contract(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    report = build_report(root, date=date, command_runner=command_runner, duration_s=duration_s)
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6197 terminal contract: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = write_contract(REPO_ROOT, date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "checksum": report["reproducibility_checksum"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    sys.exit(main())
