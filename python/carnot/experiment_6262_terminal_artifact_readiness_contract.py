"""Exp6262 terminal-artifact readiness contract.

Spec refs: REQ-INFRA-6262, SCENARIO-INFRA-6262-1,
SCENARIO-INFRA-6262-2, SCENARIO-INFRA-6262-3,
SCENARIO-INFRA-6262-4, SCENARIO-INFRA-6262-5,
SCENARIO-INFRA-6262-6.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any

import scripts.adversarial_verify as adversarial_verify
from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import (
    NONTERMINAL_CLASSES,
    TERMINAL_CLASSES,
    canonical_json,
    classify_artifact_path,
    classify_artifact_payload,
    gate_field_eligibility,
    gate_field_eligibility_for_path,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
CommandRunner = Callable[[tuple[str, ...], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6262_terminal_artifact_readiness_contract.json")
EXP6228_RELATIVE_PATH = Path("results/experiment_6228_supervised_three_family_runtime_endurance.json")
CLASSIFIER_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
ADVERSARIAL_VERIFIER_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
INFERENCE_SUBSTRATE = "deterministic_terminal_artifact_readiness_replay_no_model"
SCHEMA = "carnot.experiment_6262.terminal_artifact_readiness_contract.v1"
EXPERIMENT_ID = "exp6262-terminal-artifact-readiness-contract"
NONTERMINAL_FLAG_KIND = adversarial_verify.NONTERMINAL_FLAG_KIND

FOCUSED_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_terminal_artifact_readiness_contract_6262.py tests/python/test_experiment_6262_terminal_artifact_readiness_contract.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6262_terminal_artifact_readiness_contract.py -m pytest tests/python/test_terminal_artifact_readiness_contract_6262.py tests/python/test_experiment_6262_terminal_artifact_readiness_contract.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6262_terminal_artifact_readiness_contract.py --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_terminal_artifact_readiness_contract_6262.py tests/python/test_experiment_6262_terminal_artifact_readiness_contract.py",
)
QA_COMMANDS = (
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/verifier_authenticity_lint.py",
    ".venv/bin/python scripts/determination_preservation_lint.py --all",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
)
COMMAND_TIMEOUTS_S = {
    ".venv/bin/pytest tests/python -q": 3600,
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6228_path_hash_and_exact_classification",
    "classifier_source_hash_before_after",
    "adversarial_verifier_source_hash_before_after",
    "supported_terminal_classes",
    "rejected_nonterminal_classes",
    "exact_path_over_receipt_precedence",
    "gate_field_eligibility_contract",
    "exp6228_regression_flag_code_and_severity",
    "honest_blocked_control_result",
    "gate_skip_control_result",
    "receipt_override_negative_control",
    "readiness_missing_negative_control",
    "false_positive_fixture_results",
    "focused_test_results",
    "qa_layer_audit_results",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "terminal_artifact_contract_ready_score",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Declares this report terminal only when all readiness controls and commands pass.",
    "exp6228_path_hash_and_exact_classification": "Pins Exp6228 to its exact path and current nonterminal class.",
    "classifier_source_hash_before_after": "Proves the shared classifier source did not change during replay.",
    "adversarial_verifier_source_hash_before_after": "Proves the adversarial boundary source did not change during replay.",
    "supported_terminal_classes": "Names terminal classes that may satisfy artifact readiness.",
    "rejected_nonterminal_classes": "Names nonterminal classes that must receive a critical finding.",
    "exact_path_over_receipt_precedence": "Shows a conductor receipt cannot override exact artifact state.",
    "gate_field_eligibility_contract": "Shows gates may read only terminal artifacts with exact bare fields.",
    "exp6228_regression_flag_code_and_severity": "Records the critical Exp6228 regression finding.",
    "honest_blocked_control_result": "Keeps honest blocked terminal artifacts from false positives.",
    "gate_skip_control_result": "Keeps gate-skipped terminal artifacts from false positives.",
    "receipt_override_negative_control": "Rejects receipt-only readiness data from downstream gates.",
    "readiness_missing_negative_control": "Keeps absent declared artifacts distinct from malformed artifacts.",
    "false_positive_fixture_results": "Records clean terminal controls for complete, null, blocked, and gate-skip states.",
    "focused_test_results": "Stores focused unit, coverage, and spec checks for this contract.",
    "qa_layer_audit_results": "Stores whole-suite, verifier, preservation, clutter, and E2E-plan checks.",
    "protected_files_unchanged": "Compares protected evidence and source hashes before and after checks.",
    "preconditions_checked": "Records git status, fixture hashes, source hashes, and Exp6228 classification.",
    "inference_substrate": "Declares deterministic artifact replay with no model inference.",
    "verifier_is_oracle": "False because this verifies artifact state, not benchmark answer truth.",
    "field_provenance": "Ties each required field to the spec, classifier, verifier, or command receipts.",
    "field_principles": "Keeps the reason for each required field next to the emitted value.",
    "test_commands": "Lists every command whose exit code contributes to this report.",
    "test_exit_codes": "Stores command exit codes as bare integers.",
    "terminal_artifact_contract_ready_score": "Bare one means every readiness control and command passed.",
    "duration_s": "Reports real wall time for deterministic replay without padding.",
    "reproducibility_checksum": "Content-addresses the normalized report payload.",
    "honest_verdict": "Uses a terminal prefix and states that Exp6228 remains nonterminal.",
}

PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    CLASSIFIER_RELATIVE_PATH,
    ADVERSARIAL_VERIFIER_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    Path("scripts/conductor_gates.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("ops/e2e-test-plan.md"),
    Path("tests/python/test_terminal_artifact_readiness_contract_6262.py"),
    Path("tests/python/test_experiment_6262_terminal_artifact_readiness_contract.py"),
    EXP6228_RELATIVE_PATH,
)


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def _argv(command: str) -> tuple[str, ...]:
    return tuple(shlex.split(command))


def _command_text(argv: Sequence[str]) -> str:  # pragma: no cover - shell edge.
    return " ".join(shlex.quote(part) for part in argv)


def _run_command(argv: tuple[str, ...], root: Path) -> JsonDict:  # pragma: no cover - shell edge.
    command = _command_text(argv)
    try:
        proc = subprocess.run(
            argv,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=COMMAND_TIMEOUTS_S.get(command),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "classification": "timeout",
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "exit_code": 127,
            "classification": "command_not_found",
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    return {
        "command": command,
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _run_commands(root: Path, commands: Sequence[str], runner: CommandRunner) -> list[JsonDict]:
    return [runner(_argv(command), root) for command in commands]


def _git_status(root: Path) -> list[str]:
    proc = subprocess.run(
        ("git", "status", "--short"),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return [f"git_status_failed:{proc.returncode}:{proc.stderr.strip()}"]  # pragma: no cover
    return proc.stdout.splitlines()


def _hashes(root: Path, paths: Sequence[Path]) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in paths}


def _source_hash_before_after(root: Path, path: Path, before: JsonMap, after: JsonMap) -> JsonDict:
    key = path.as_posix()
    return {
        "path": key,
        "before": before.get(key),
        "after": after.get(key),
        "unchanged": before.get(key) == after.get(key),
    }


def _protected_files_unchanged(before: JsonMap, after: JsonMap) -> JsonDict:
    changed = [path for path in before if before.get(path) != after.get(path)]
    conductor_key = RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix()
    return {
        "before": dict(before),
        "after": dict(after),
        "changed_paths": changed,
        "unchanged": not changed,
        "scripts_research_conductor_py_untouched": before.get(conductor_key) == after.get(conductor_key),
    }


def _readiness_flags_for_classification(classification: Any) -> list[JsonDict]:
    flags: list[adversarial_verify.Flag] = []
    adversarial_verify.check_terminal_artifact_readiness(classification, flags)
    return [flag.to_dict() for flag in flags]


def _control_result(payload: JsonMap) -> JsonDict:
    classification = classify_artifact_payload(payload)
    flags = _readiness_flags_for_classification(classification)
    return {
        "classification": classification.classification,
        "terminal": classification.terminal,
        "reason": classification.reason,
        "flag_count": len(flags),
        "flags": flags,
        "payload_sha256": payload_sha256(payload),
    }


def _missing_negative_control(root: Path) -> JsonDict:
    path = root / "results/experiment_6262_missing_readiness_negative_control.json"
    classification = classify_artifact_path(path)
    flags = _readiness_flags_for_classification(classification)
    return {
        "path": path.as_posix(),
        "classification": classification.classification,
        "reason": classification.reason,
        "severity": flags[0]["severity"] if flags else None,
        "kind": flags[0]["kind"] if flags else None,
        "flag_count": len(flags),
        "flags": flags,
    }


def _rejected_nonterminal_classes(root: Path) -> JsonDict:
    malformed_dir = Path(tempfile.gettempdir()) / "carnot_exp6262_terminal_artifact_readiness"
    malformed_dir.mkdir(parents=True, exist_ok=True)
    malformed_path = malformed_dir / "experiment_6262_malformed_readiness_negative_control.json"
    malformed_path.write_text("{not json", encoding="utf-8")
    payload_cases: tuple[tuple[str, JsonMap], ...] = (
        ("running", {"status": "running", "honest_verdict": "running"}),
        ("running_bootstrap", {"status": "running_bootstrap", "honest_verdict": "running"}),
        ("bootstrap_only", {"status": "bootstrap_only", "honest_verdict": "blocked: bootstrap only"}),
        ("partial", {"status": "complete_partial", "honest_verdict": "complete_partial: partial"}),
        ("contradictory", {"status": "complete_ready", "honest_verdict": "blocked_precondition"}),
        ("unknown", {"status": "preconditions_recorded", "honest_verdict": None}),
    )
    rows: JsonDict = {}
    for name, payload in payload_cases:
        classification = classify_artifact_payload(payload)
        flags = _readiness_flags_for_classification(classification)
        rows[name] = {
            "classification": classification.classification,
            "terminal": classification.terminal,
            "reason": classification.reason,
            "flag_count": len(flags),
            "severity": flags[0]["severity"] if flags else None,
            "kind": flags[0]["kind"] if flags else None,
        }
    for name, path in (
        ("missing", root / "results/experiment_6262_missing_readiness_negative_control.json"),
        ("malformed", malformed_path),
    ):
        classification = classify_artifact_path(path)
        flags = _readiness_flags_for_classification(classification)
        rows[name] = {
            "classification": classification.classification,
            "terminal": classification.terminal,
            "reason": classification.reason,
            "flag_count": len(flags),
            "severity": flags[0]["severity"] if flags else None,
            "kind": flags[0]["kind"] if flags else None,
        }
    return {
        "declared_classes": sorted(NONTERMINAL_CLASSES),
        "rows": rows,
        "all_rejected": all(
            row.get("terminal") is False
            and row.get("kind") == NONTERMINAL_FLAG_KIND
            and row.get("severity") == "critical"
            for row in rows.values()
        ),
    }


def _supported_terminal_classes() -> JsonDict:
    payloads: dict[str, JsonMap] = {
        "complete": {"status": "complete", "honest_verdict": "complete: clean"},
        "ready": {"status": "complete_ready", "honest_verdict": "complete_ready: clean"},
        "positive": {"status": "complete_positive", "honest_verdict": "complete_positive: clean"},
        "null": {"status": "complete_null", "honest_verdict": "complete_null: clean"},
        "blocked": {"status": "blocked", "honest_verdict": "blocked_precondition"},
        "skipped": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [{"passed": False}],
        },
        "retired": {"status": "retired", "honest_verdict": "retired: closed"},
        "flagged": {"status": "complete", "honest_verdict": "complete: quarantined", "flagged_adversarial": True},
    }
    rows = {name: _control_result(payload) for name, payload in payloads.items()}
    return {
        "declared_classes": sorted(TERMINAL_CLASSES),
        "rows": rows,
        "all_terminal": all(row["terminal"] is True for row in rows.values()),
    }


def _gate_field_contract() -> JsonDict:
    terminal = {"status": "complete_ready", "honest_verdict": "complete_ready: ok"}
    return {
        "terminal_exact_bare": gate_field_eligibility(
            {**terminal, "ready_score": 1}, "ready_score"
        ).to_dict(),
        "terminal_wrapped": gate_field_eligibility(
            {**terminal, "ready_score": {"value": 1, "principle": "fixture"}}, "ready_score"
        ).to_dict(),
        "terminal_nested": gate_field_eligibility(
            {**terminal, "metrics": {"ready_score": 1}}, "ready_score"
        ).to_dict(),
        "nonterminal_exact_bare": gate_field_eligibility(
            {"status": "running", "honest_verdict": "running", "ready_score": 1},
            "ready_score",
        ).to_dict(),
    }


def _false_positive_controls() -> JsonDict:
    payloads: dict[str, JsonMap] = {
        "complete": {"status": "complete", "honest_verdict": "complete: clean"},
        "null": {"status": "complete_null", "honest_verdict": "complete_null: clean"},
        "blocked": {"status": "blocked", "honest_verdict": "blocked_precondition"},
        "gate_skip": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [{"passed": False}],
        },
    }
    return {name: _control_result(payload) for name, payload in payloads.items()}


def _exp6228_flag(exp6228_path: Path, exp6228_classification: JsonMap) -> JsonDict:
    report = adversarial_verify.verify_artifact(exp6228_path)
    flags = [
        flag
        for flag in report.get("flags", [])
        if isinstance(flag, Mapping) and flag.get("kind") == NONTERMINAL_FLAG_KIND
    ]
    flag = flags[0] if flags else {}
    return {
        "kind": flag.get("kind"),
        "severity": flag.get("severity"),
        "detail": flag.get("detail"),
        "classification": exp6228_classification.get("classification"),
        "flag_count": len(flags),
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": [
                "REQ-INFRA-6262",
                CLASSIFIER_RELATIVE_PATH.as_posix(),
                ADVERSARIAL_VERIFIER_RELATIVE_PATH.as_posix(),
                EXP6228_RELATIVE_PATH.as_posix(),
                "focused command receipts",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exit_codes(rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in rows
        if row.get("command")
    }


def _commands_pass(rows: Sequence[JsonMap]) -> bool:
    return bool(rows) and all(int(row.get("exit_code", 1)) == 0 for row in rows)


def _required_controls_pass(report: JsonMap) -> bool:
    gate = report.get("gate_field_eligibility_contract")
    false_positive = report.get("false_positive_fixture_results")
    return (
        report.get("exp6228_regression_flag_code_and_severity", {}).get("kind")
        == NONTERMINAL_FLAG_KIND
        and report.get("exp6228_regression_flag_code_and_severity", {}).get("severity") == "critical"
        and report.get("supported_terminal_classes", {}).get("all_terminal") is True
        and report.get("rejected_nonterminal_classes", {}).get("all_rejected") is True
        and report.get("exact_path_over_receipt_precedence", {}).get("receipt_override_attempted")
        is True
        and report.get("exact_path_over_receipt_precedence", {}).get("receipt_overrode") is False
        and isinstance(gate, Mapping)
        and gate.get("terminal_exact_bare", {}).get("eligible") is True
        and gate.get("terminal_wrapped", {}).get("eligible") is False
        and gate.get("terminal_nested", {}).get("eligible") is False
        and gate.get("nonterminal_exact_bare", {}).get("eligible") is False
        and report.get("honest_blocked_control_result", {}).get("flag_count") == 0
        and report.get("gate_skip_control_result", {}).get("flag_count") == 0
        and report.get("receipt_override_negative_control", {}).get("eligible") is False
        and report.get("readiness_missing_negative_control", {}).get("severity") == "critical"
        and isinstance(false_positive, Mapping)
        and all(
            isinstance(row, Mapping)
            and row.get("terminal") is True
            and row.get("flag_count") == 0
            for row in false_positive.values()
        )
        and _commands_pass(report.get("focused_test_results", []))
        and _commands_pass(report.get("qa_layer_audit_results", []))
        and report.get("classifier_source_hash_before_after", {}).get("unchanged") is True
        and report.get("adversarial_verifier_source_hash_before_after", {}).get("unchanged") is True
        and report.get("protected_files_unchanged", {}).get("unchanged") is True
    )


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.perf_counter()
    protected_before = _hashes(root, PROTECTED_RELATIVE_PATHS)
    git_status_before = _git_status(root)
    exp6228_path = root / EXP6228_RELATIVE_PATH
    exp6228_classification = classify_artifact_path(exp6228_path)
    receipt_classification = classify_artifact_path(
        exp6228_path,
        conductor_receipt={"status": "OK", "detail": "negative control"},
    )
    focused_results = _run_commands(root, FOCUSED_TEST_COMMANDS, command_runner)
    qa_results = _run_commands(root, QA_COMMANDS, command_runner)
    protected_after = _hashes(root, PROTECTED_RELATIVE_PATHS)
    exp6228_classification_after = classify_artifact_path(exp6228_path)

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "blocked",
        "exp6228_path_hash_and_exact_classification": {
            "path": EXP6228_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(exp6228_path),
            "classification": exp6228_classification.to_dict(),
        },
        "classifier_source_hash_before_after": _source_hash_before_after(
            root, CLASSIFIER_RELATIVE_PATH, protected_before, protected_after
        ),
        "adversarial_verifier_source_hash_before_after": _source_hash_before_after(
            root, ADVERSARIAL_VERIFIER_RELATIVE_PATH, protected_before, protected_after
        ),
        "supported_terminal_classes": _supported_terminal_classes(),
        "rejected_nonterminal_classes": _rejected_nonterminal_classes(root),
        "exact_path_over_receipt_precedence": receipt_classification.to_dict(),
        "gate_field_eligibility_contract": _gate_field_contract(),
        "exp6228_regression_flag_code_and_severity": _exp6228_flag(
            exp6228_path, exp6228_classification.to_dict()
        ),
        "honest_blocked_control_result": _control_result(
            {"status": "blocked", "honest_verdict": "blocked_precondition"}
        ),
        "gate_skip_control_result": _control_result(
            {
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "gates_evaluated": [{"passed": False}],
            }
        ),
        "receipt_override_negative_control": gate_field_eligibility_for_path(
            exp6228_path,
            "three_family_runtime_ready_score",
            conductor_receipt={"status": "OK", "three_family_runtime_ready_score": 1},
        ).to_dict(),
        "readiness_missing_negative_control": _missing_negative_control(root),
        "false_positive_fixture_results": _false_positive_controls(),
        "focused_test_results": focused_results,
        "qa_layer_audit_results": qa_results,
        "protected_files_unchanged": _protected_files_unchanged(protected_before, protected_after),
        "preconditions_checked": {
            "run_date": date,
            "git_status_before": git_status_before,
            "git_status_after_tests": _git_status(root),
            "protected_hashes_before": dict(protected_before),
            "protected_hashes_after": dict(protected_after),
            "exp6228_classification_before": exp6228_classification.to_dict(),
            "exp6228_classification_after": exp6228_classification_after.to_dict(),
            "exact_source_hashes_before": {
                CLASSIFIER_RELATIVE_PATH.as_posix(): protected_before.get(
                    CLASSIFIER_RELATIVE_PATH.as_posix()
                ),
                ADVERSARIAL_VERIFIER_RELATIVE_PATH.as_posix(): protected_before.get(
                    ADVERSARIAL_VERIFIER_RELATIVE_PATH.as_posix()
                ),
            },
            "fixture_hashes": {
                EXP6228_RELATIVE_PATH.as_posix(): path_sha256(exp6228_path),
            },
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(FOCUSED_TEST_COMMANDS) + list(QA_COMMANDS),
        "test_exit_codes": _test_exit_codes(focused_results + qa_results),
        "terminal_artifact_contract_ready_score": 0,
        "duration_s": float(duration_s if duration_s is not None else time.perf_counter() - started),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: Exp6262 readiness controls did not all pass",
    }
    if _required_controls_pass(report):
        report["status"] = "complete_ready"
        report["terminal_artifact_contract_ready_score"] = 1
        report["honest_verdict"] = (
            "complete: terminal artifact readiness contract is enforced outside the conductor; "
            "Exp6228 remains a critical nonterminal artifact"
        )
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing:{field}")
    score = report.get("terminal_artifact_contract_ready_score")
    controls_pass = _required_controls_pass(report)
    if type(score) is not int or score not in (0, 1) or (score == 1) != controls_pass:
        errors.append("terminal_artifact_contract_ready_score")
    if report.get("exp6228_regression_flag_code_and_severity", {}).get("kind") != NONTERMINAL_FLAG_KIND or report.get("exp6228_regression_flag_code_and_severity", {}).get("severity") != "critical":
        errors.append("exp6228_regression_flag_code_and_severity")
    if score == 1 and (
        not _commands_pass(report.get("focused_test_results", []))
        or not _commands_pass(report.get("qa_layer_audit_results", []))
    ):
        errors.append("focused_test_results")
    for hash_field in (
        "classifier_source_hash_before_after",
        "adversarial_verifier_source_hash_before_after",
        "protected_files_unchanged",
    ):
        if report.get(hash_field, {}).get("unchanged") is not True:
            errors.append(hash_field)
    false_positive = report.get("false_positive_fixture_results")
    if not isinstance(false_positive, Mapping) or not all(
        isinstance(row, Mapping) and row.get("flag_count") == 0 and row.get("terminal") is True
        for row in false_positive.values()
    ):
        errors.append("false_positive_fixture_results")
    principles = report.get("field_principles")
    provenance = report.get("field_provenance")
    if isinstance(principles, Mapping) and isinstance(provenance, Mapping):
        for field in REQUIRED_ARTIFACT_FIELDS:
            if not principles.get(field):
                errors.append(f"field_principles:{field}")
    verdict = str(report.get("honest_verdict") or "")
    if score == 1 and verdict.startswith("complete:") is False:
        errors.append("honest_verdict")
    if score == 0 and verdict.startswith("blocked:") is False:
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
        raise ValueError(f"invalid Exp6262 terminal readiness contract: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = write_contract(REPO_ROOT, date=args.date)
    print(
        json.dumps(
            {
                "path": RESULT_RELATIVE_PATH.as_posix(),
                "checksum": report["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    sys.exit(main())
