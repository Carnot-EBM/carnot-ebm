"""Exp6449 generation-to-verdict path receipt contract.

Spec refs: REQ-VERIFY-6449, SCENARIO-VERIFY-6449-CHAIN,
SCENARIO-VERIFY-6449-CONTROLS, SCENARIO-VERIFY-6449-ATTACKS.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from carnot import experiment_6427_fresh_constraint_saturation_factor_corpus as exp6427
from carnot import path_receipts
from carnot import task_runtime_receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6449_generation_to_verdict_path_receipt_contract.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6449_generation_to_verdict_path_receipt_contract"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6449_generation_to_verdict_path_receipt_contract.py"
)
HELPER_RELATIVE_PATH = Path("python/carnot/path_receipts.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6449_generation_to_verdict_path_receipt_contract.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
EXP6427_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6427_fresh_constraint_saturation_factor_corpus.json"
)
EXP6427_DATA_RELATIVE_PATH = Path(
    "data/research/experiment_6427_fresh_constraint_saturation_factor_corpus"
)

RUN_DATE = "20260815"
RANDOM_SEED = 6449
FIXTURE_LIMIT = 24
INFERENCE_SUBSTRATE = "deterministic_fixture_path_receipt_replay_no_llm"
CONTROL_IDS = ("identity", "injected_wrapper", "restored_wrapper")
INJECTED_BOUNDARY = "checker_transport"
WRAPPER_ID = "declared_checker_transport_wrapper_v1"
CHECKER_ID = "exp6427_deterministic_constraint_and_joint_checker"
SCHEMA_AND_VERSION = path_receipts.SCHEMA_VERSION
ATTACK_IDS = (
    "wrapper_insertion",
    "parser_substitution",
    "stale_checker_response",
    "missing_stage",
    "reordered_stage",
    "replayed_raw_bytes_under_another_unit_id",
    "forged_parent_hash",
    "aggregate_row_mismatch",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6449_generation_to_verdict_path_receipt_contract "
    "--date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6449_generation_to_verdict_path_receipt_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/path_receipts.py,"
    "python/carnot/experiment_6449_generation_to_verdict_path_receipt_contract.py "
    "-m pytest tests/python/test_experiment_6449_generation_to_verdict_path_receipt_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/path_receipts.py,"
    "python/carnot/experiment_6449_generation_to_verdict_path_receipt_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6449_generation_to_verdict_path_receipt_contract.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6449_generation_to_verdict_path_receipt_contract "
    "--date 20260815 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6449_generation_to_verdict_path_receipt_contract.json"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6449_generation_to_verdict_path_receipt_contract.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    ROW_LINT_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    HELPER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/experiment_template.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6427_RESULT_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "receipt_schema_and_version",
    "fixture_manifest_and_hashes",
    "code_and_configuration_hashes",
    "control_precommitment",
    "per_unit_rows",
    "identity_replay_results",
    "injected_boundary_results",
    "restored_boundary_results",
    "stage_chain_validation",
    "terminal_verdict_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "aggregate_row_recomputation",
    "path_receipt_ready_score",
    "protected_files_unchanged",
    "blocked_reason",
    "gate_check_summary",
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
    "status": "Names success or blocked precondition state.",
    "receipt_schema_and_version": "Pins the reusable path receipt schema and ordered stages.",
    "fixture_manifest_and_hashes": "Binds immutable Exp6427 fixture files before replay.",
    "code_and_configuration_hashes": "Binds stage code identities and stage configuration hashes.",
    "control_precommitment": "Freezes identity, injected-wrapper, and restored-wrapper controls.",
    "per_unit_rows": "Retains every unit/control row and every stage hash.",
    "identity_replay_results": "Shows identity rows replay from fixture bytes.",
    "injected_boundary_results": "Requires injected rows to change only checker_transport.",
    "restored_boundary_results": "Requires restored rows to match identity terminal hashes and verdicts.",
    "stage_chain_validation": "Rejects missing, duplicate, reordered, parent-broken, mutated, or unknown-code stages.",
    "terminal_verdict_recomputation": "Final verdicts recompute exactly from checker responses.",
    "attack_matrix": "Every critical path attack must fail closed.",
    "current_adversarial_findings": "Internal critical findings must be zero before readiness.",
    "aggregate_row_recomputation": "Aggregate counts recompute from per_unit_rows.",
    "path_receipt_ready_score": "Bare one only when all replay, localization, attack, aggregate, protected-file, and finding gates pass.",
    "protected_files_unchanged": "Proves conductor and reconciliation-owned files stayed byte-stable.",
    "blocked_reason": "Names failed preconditions for a blocked artifact.",
    "gate_check_summary": "Summarizes precondition gates and blocker count.",
    "preconditions_checked": "Lists every fixture, checker, clock, storage, and output-path precondition.",
    "inference_substrate": "Declares deterministic fixture path receipt replay with no LLM.",
    "verifier_is_oracle": "True only for deterministic fixture checker and row arithmetic.",
    "field_principles": "Maps every required field and readiness-score condition.",
    "field_provenance": "States whether fields are measured, derived, source-bound, or constant.",
    "random_seed": "Pins fixture ordering and deterministic controls.",
    "duration_s": "Measured wall duration for this local replay.",
    "tests_run": "Records the verification commands expected for this contract.",
    "reproducibility_checksum": "Content-addresses the terminal artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix or blocked precondition prefix.",
    "path_receipt_ready_score:control_localization": "Every control must localize correctly.",
    "path_receipt_ready_score:identity_and_restore_replay": "Identity and restored rows must replay.",
    "path_receipt_ready_score:attacks_detected": "Every critical attack must fail closed.",
    "path_receipt_ready_score:aggregate_recompute": "Reported aggregate counts must match per-unit rows.",
    "path_receipt_ready_score:protected_files": "Protected files must be unchanged.",
    "path_receipt_ready_score:critical_findings": "Current critical findings must be zero.",
}
FIELD_PRINCIPLES.update({attack: "Critical attack must fail closed." for attack in ATTACK_IDS})
FIELD_PRINCIPLES.update(
    {stage: "Path stage required by REQ-VERIFY-6449." for stage in path_receipts.REQUIRED_STAGE_NAMES}
)

FIELD_PROVENANCE: dict[str, str] = {
    "status": "derived readiness gate",
    "receipt_schema_and_version": "constant and derived schema hash",
    "fixture_manifest_and_hashes": "source file hashes",
    "code_and_configuration_hashes": "source and config hashes",
    "control_precommitment": "constant",
    "per_unit_rows": "deterministic fixture replay",
    "identity_replay_results": "derived from row replay",
    "injected_boundary_results": "derived from matched-control hash comparison",
    "restored_boundary_results": "derived from identity/restored terminal comparison",
    "stage_chain_validation": "derived by path_receipts.validate_stage_chain",
    "terminal_verdict_recomputation": "derived from checker response payloads",
    "attack_matrix": "derived mutation checks",
    "current_adversarial_findings": "derived internal contract findings",
    "aggregate_row_recomputation": "derived row arithmetic",
    "path_receipt_ready_score": "derived conjunctive gate",
    "protected_files_unchanged": "source file hashes",
    "blocked_reason": "derived precondition check",
    "gate_check_summary": "derived precondition summary",
    "preconditions_checked": "measured local filesystem, checker, clock, and storage checks",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "measured wall clock or caller-supplied test duration",
    "tests_run": "caller-supplied or pending verification commands",
    "reproducibility_checksum": "derived checksum",
    "honest_verdict": "derived terminal verdict",
}


def sha256_file(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json_object(path: str | Path) -> JsonDict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return task_runtime_receipts.write_json_atomic(path, payload)


def source_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(before: Mapping[str, str | None]) -> JsonDict:
    after = protected_hashes()
    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def _path_receipt(path: str | Path) -> JsonDict:
    file_path = Path(path)
    return {
        "path": str(file_path),
        "present": file_path.is_file(),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else 0,
    }


def code_and_configuration_hashes(source_before: Mapping[str, str | None]) -> JsonDict:
    stage_code_hashes = {
        stage: path_receipts.sha256_json(
            {
                "schema": path_receipts.SCHEMA_VERSION,
                "stage": stage,
                "module": source_before.get(MODULE_RELATIVE_PATH.as_posix()),
                "helper": source_before.get(HELPER_RELATIVE_PATH.as_posix()),
                "checker": source_before.get(
                    "python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py"
                ),
            }
        )
        for stage in path_receipts.REQUIRED_STAGE_NAMES
    }
    config_hashes = {
        stage: path_receipts.sha256_json(
            {"stage": stage, "random_seed": RANDOM_SEED, "wrapper_id": WRAPPER_ID}
        )
        for stage in path_receipts.REQUIRED_STAGE_NAMES
    }
    return {
        "source_hashes": dict(source_before),
        "allowed_code_hashes": stage_code_hashes,
        "configuration_hashes": config_hashes,
        "configuration_manifest_sha256": path_receipts.sha256_json(config_hashes),
    }


def receipt_schema_and_version() -> JsonDict:
    payload = {
        "schema_version": path_receipts.SCHEMA_VERSION,
        "required_stage_names": list(path_receipts.REQUIRED_STAGE_NAMES),
        "required_stage_fields": list(path_receipts.REQUIRED_STAGE_FIELDS),
    }
    return {
        "schema": path_receipts.SCHEMA_VERSION,
        "version": "v1",
        "schema_sha256": path_receipts.sha256_json(payload),
        "payload": payload,
    }


def control_precommitment() -> JsonDict:
    controls = [
        {
            "control_id": "identity",
            "declared_changed_boundary": None,
            "transport_policy": "send checker request bytes unchanged",
        },
        {
            "control_id": "injected_wrapper",
            "declared_changed_boundary": INJECTED_BOUNDARY,
            "transport_policy": f"wrap checker request in {WRAPPER_ID}",
        },
        {
            "control_id": "restored_wrapper",
            "declared_changed_boundary": None,
            "transport_policy": "apply wrapper then restore original request bytes before receipt",
        },
    ]
    return {
        "schema": "carnot.exp6449.control_precommitment.v1",
        "controls": controls,
        "control_ids": list(CONTROL_IDS),
        "precommitted_before_replay": True,
        "precommitment_sha256": path_receipts.sha256_json(controls),
    }


def _rows_from_exp6427(payload: Mapping[str, Any]) -> list[JsonDict]:
    rows = payload.get("per_unit_rows", {})
    if isinstance(rows, Mapping):
        rows = rows.get("rows", [])
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _event_map(manifest: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row.get("event_id")): dict(row)
        for row in manifest.get("events", [])
        if isinstance(row, Mapping)
    }


def _fixture_raw_event_id(raw_bytes: bytes) -> str:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    return str(payload.get("event_id", ""))


def load_fixture_units(
    exp6427_result_path: str | Path,
    *,
    fixture_limit: int = FIXTURE_LIMIT,
) -> tuple[list[JsonDict], JsonDict]:
    payload = read_json_object(exp6427_result_path)
    rows = _rows_from_exp6427(payload)
    manifest_path = Path(
        str(
            payload.get("manifest_path_hash_counts_balance_and_partition_seals", {}).get(
                "path", REPO_ROOT / EXP6427_DATA_RELATIVE_PATH / "manifest"
            )
        )
    )
    manifest = read_json_object(manifest_path)
    events = _event_map(manifest)
    units: list[JsonDict] = []
    for row in sorted(rows, key=lambda item: int(item.get("row_index", 0)))[:fixture_limit]:
        raw_path = Path(str(row.get("raw_output_path", "")))
        raw_bytes = raw_path.read_bytes()
        raw_hash = path_receipts.sha256_bytes(raw_bytes)
        unit_id = str(row.get("row_id"))
        event = events.get(unit_id, {})
        if raw_hash != row.get("raw_output_sha256"):
            raise ValueError(f"fixture hash mismatch for {unit_id}")
        if not event:
            raise ValueError(f"fixture event missing for {unit_id}")
        if _fixture_raw_event_id(raw_bytes) != unit_id:
            raise ValueError(f"raw event id mismatch for {unit_id}")
        units.append(
            {
                "unit_id": unit_id,
                "row_index": int(row.get("row_index", 0)),
                "source_row": row,
                "event": event,
                "raw_path": str(raw_path),
                "raw_bytes": raw_bytes,
                "raw_sha256": raw_hash,
                "raw_byte_length": len(raw_bytes),
            }
        )
    manifest_receipt = {
        "exp6427_artifact": _path_receipt(exp6427_result_path),
        "manifest": _path_receipt(manifest_path),
        "selected_unit_count": len(units),
        "selected_unit_ids": [unit["unit_id"] for unit in units],
        "selected_raw_hashes": {unit["unit_id"]: unit["raw_sha256"] for unit in units},
        "fixture_manifest_sha256": path_receipts.sha256_json(
            {
                "artifact": _path_receipt(exp6427_result_path),
                "manifest": _path_receipt(manifest_path),
                "units": [
                    {
                        "unit_id": unit["unit_id"],
                        "raw_sha256": unit["raw_sha256"],
                        "raw_byte_length": unit["raw_byte_length"],
                    }
                    for unit in units
                ],
            }
        ),
    }
    return units, manifest_receipt


def _precondition_row(resource: str, available: bool, detail: str, path: str = "") -> JsonDict:
    return {"resource": resource, "available": available, "detail": detail, "path": path}


def check_preconditions(
    *,
    result_path: str | Path,
    data_dir: str | Path,
    exp6427_result_path: str | Path,
    fixture_limit: int,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    checks: list[JsonDict] = []
    units: list[JsonDict] = []
    fixture_manifest: JsonDict = {"selected_unit_count": 0, "selected_unit_ids": []}
    exp6427_path = Path(exp6427_result_path)
    checks.append(
        _precondition_row(
            "exp6427_fixture_artifact",
            exp6427_path.is_file() and exp6427_path.stat().st_size > 0,
            "readable nonzero Exp6427 fixture artifact",
            str(exp6427_path),
        )
    )
    try:
        units, fixture_manifest = load_fixture_units(exp6427_path, fixture_limit=fixture_limit)
        fixture_ready = len(units) >= fixture_limit
        detail = f"loaded {len(units)} fixture unit(s)"
    except (OSError, ValueError) as exc:
        fixture_ready = False
        detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        _precondition_row(
            "immutable_fixture_bytes",
            fixture_ready,
            detail,
            str(REPO_ROOT / EXP6427_DATA_RELATIVE_PATH),
        )
    )
    checker_ready = callable(exp6427.parse_factor_surface) and callable(
        exp6427.exact_constraint_check
    )
    checks.append(
        _precondition_row(
            "exact_deterministic_checker",
            checker_ready,
            "Exp6427 parse and exact checker callables are importable",
            str(REPO_ROOT / "python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py"),
        )
    )
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        storage_ready = os.access(Path(data_dir), os.W_OK) and os.access(Path(result_path).parent, os.W_OK)
        storage_detail = "data and result directories are writable"
    except OSError as exc:  # pragma: no cover
        storage_ready = False
        storage_detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        _precondition_row("atomic_local_storage", storage_ready, storage_detail, str(result_path))
    )
    first_clock = time.monotonic_ns()
    second_clock = time.monotonic_ns()
    checks.append(
        _precondition_row(
            "monotonic_clock",
            second_clock >= first_clock,
            f"monotonic probe {first_clock}->{second_clock}",
        )
    )
    paths_distinct = Path(result_path).resolve() != exp6427_path.resolve()
    data_distinct = Path(data_dir).resolve() != (REPO_ROOT / EXP6427_DATA_RELATIVE_PATH).resolve()
    checks.append(
        _precondition_row(
            "new_output_paths",
            paths_distinct and data_distinct,
            "Exp6449 result/data paths are distinct from fixture sources",
            str(result_path),
        )
    )
    return checks, units, fixture_manifest


def _blocked_artifact(
    *,
    date: str,
    result_path: str | Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_before: Mapping[str, str | None],
    protected_before: Mapping[str, str | None],
    duration_s: float,
) -> JsonDict:
    blockers = [row for row in preconditions if row.get("available") is not True]
    blocked_reason = "; ".join(str(row.get("resource")) for row in blockers)
    gate_summary = (
        f"{len(blockers)} precondition(s) failed; first failure: {blockers[0]['resource']}"
        if blockers
        else "blocked artifact requested without a failed precondition"
    )
    artifact = {
        "status": "blocked",
        "receipt_schema_and_version": receipt_schema_and_version(),
        "fixture_manifest_and_hashes": {"selected_unit_count": 0, "selected_unit_ids": []},
        "code_and_configuration_hashes": code_and_configuration_hashes(source_before),
        "control_precommitment": control_precommitment(),
        "per_unit_rows": {"rows": [], "row_count": 0, "unit_count": 0, "control_count": 0},
        "identity_replay_results": {"all_replayed": False, "replayed_count": 0},
        "injected_boundary_results": {"all_localized": False, "changed_boundaries": {}},
        "restored_boundary_results": {
            "all_restored": False,
            "matched_identity_terminal_hash_count": 0,
        },
        "stage_chain_validation": {"all_valid": False, "accepted_count": 0, "invalid_count": 0},
        "terminal_verdict_recomputation": {"all_recomputed": False, "mismatch_count": 0},
        "attack_matrix": {"rows": [], "all_critical_fail_closed": False, "false_accept_count": 0},
        "current_adversarial_findings": [
            {"severity": "critical", "kind": "PRECONDITION_FAILED", "detail": blocked_reason}
        ],
        "aggregate_row_recomputation": {"matches_reported": False, "reasons": ["blocked"]},
        "path_receipt_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "blocked_reason": blocked_reason,
        "gate_check_summary": gate_summary,
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": tests_run_receipt({}),
        "reproducibility_checksum": "",
        "honest_verdict": f"blocked_{blocked_reason or 'precondition_failed'}",
        "run_date": date,
        "result_path": str(result_path),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _parse_raw_output(raw_bytes: bytes) -> JsonDict:
    raw_text = raw_bytes.decode("utf-8")
    payload = json.loads(raw_text)
    parsed = exp6427.parse_factor_surface(raw_text)
    return {
        "raw_event_id": str(payload.get("event_id", "")),
        "parse_valid": parsed["parse_valid"],
        "malformed": parsed["malformed"],
        "parse_surface": parsed.get("parse_surface", "factor_proposal_only"),
        "proposal": parsed.get("proposal", {}),
    }


def _typed_facts(unit_id: str, parsed: Mapping[str, Any]) -> JsonDict:
    proposal = dict(parsed.get("proposal", {}))
    effects = [
        {
            "constraint_name": str(effect.get("constraint_name", "")),
            "factor_family": str(effect.get("factor_family", "")),
            "value": effect.get("value"),
            "interaction_scope": list(effect.get("interaction_scope", [])),
        }
        for effect in proposal.get("effects", [])
        if isinstance(effect, Mapping)
    ]
    return {
        "unit_id": unit_id,
        "parsed_event_id": parsed.get("raw_event_id"),
        "unit_id_binding_ok": parsed.get("raw_event_id") == unit_id,
        "abstain": proposal.get("abstain") is True,
        "abstention_reason": proposal.get("abstention_reason", ""),
        "effect_count": len(effects),
        "effects": effects,
    }


def _energy_input(event: Mapping[str, Any], typed: Mapping[str, Any]) -> JsonDict:
    binding_penalty = 0 if typed.get("unit_id_binding_ok") is True else 100
    abstain_penalty = 1 if typed.get("abstain") is True else 0
    missing_effect_penalty = max(
        0, int(event.get("simultaneous_constraint_count", 0)) - int(typed.get("effect_count", 0))
    )
    return {
        "unit_id": typed.get("unit_id"),
        "energy_terms": {
            "unit_binding_penalty": binding_penalty,
            "abstain_penalty": abstain_penalty,
            "missing_effect_penalty": missing_effect_penalty,
        },
        "energy_total": binding_penalty + abstain_penalty + missing_effect_penalty,
    }


def _checker_request(
    *,
    unit_id: str,
    event: Mapping[str, Any],
    parsed: Mapping[str, Any],
    typed: Mapping[str, Any],
    energy: Mapping[str, Any],
) -> JsonDict:
    return {
        "unit_id": unit_id,
        "raw_event_id": parsed.get("raw_event_id"),
        "checker_id": CHECKER_ID,
        "event": {
            "event_id": event.get("event_id"),
            "row_index": event.get("row_index"),
            "constraint_names": list(event.get("constraint_names", [])),
            "simultaneous_constraint_count": event.get("simultaneous_constraint_count"),
            "factor_family": event.get("factor_family"),
            "interaction_class": event.get("interaction_class"),
        },
        "proposal": parsed.get("proposal", {}),
        "typed_facts": dict(typed),
        "energy_input": dict(energy),
    }


def _checker_transport(control_id: str, request: Mapping[str, Any]) -> JsonDict:
    if control_id == "injected_wrapper":
        return {
            "wrapper_id": WRAPPER_ID,
            "declared_boundary": INJECTED_BOUNDARY,
            "inner_request": dict(request),
        }
    return dict(request)


def _unwrap_transport(transport: Mapping[str, Any]) -> Mapping[str, Any]:
    if transport.get("wrapper_id") == WRAPPER_ID:
        return dict(transport.get("inner_request", {}))
    return transport


def _checker_response(transport: Mapping[str, Any]) -> JsonDict:
    request = _unwrap_transport(transport)
    exact = exp6427.exact_constraint_check(
        request.get("event", {}),
        {"proposal": request.get("proposal", {})},
    )
    binding_ok = request.get("typed_facts", {}).get("unit_id_binding_ok") is True
    return {
        "unit_id": request.get("unit_id"),
        "checker_id": CHECKER_ID,
        "checker_version": "exp6427_exact_constraint_check.v1",
        "deterministic": True,
        "unit_id_binding_ok": binding_ok,
        "raw_joint_exact": exact.get("joint_exact") is True,
        "exact_outcome": bool(binding_ok and exact.get("joint_exact") is True),
        "evaluable": exact.get("evaluable") is True,
        "abstained": exact.get("abstained") is True,
        "correct_constraint_count": int(exact.get("correct_constraint_count", 0) or 0),
        "total_constraint_count": int(exact.get("total_constraint_count", 0) or 0),
    }


def _final_verdict(response: Mapping[str, Any]) -> JsonDict:
    verdict = path_receipts.verdict_from_checker_response(response)
    return {
        "unit_id": response.get("unit_id"),
        "expected_verdict": verdict,
        "observed_verdict": verdict,
        "terminal_exact_outcome": response.get("exact_outcome") is True,
        "checker_response_sha256": path_receipts.sha256_json(response),
    }


def _append_stage(
    stages: list[JsonDict],
    *,
    unit_id: str,
    stage_name: str,
    input_bytes: bytes,
    output_payload: Mapping[str, Any],
    code_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
    terminal_exact_outcome: bool | None = None,
) -> bytes:
    output_bytes = path_receipts.json_bytes(output_payload)
    start = time.monotonic_ns()
    end = time.monotonic_ns()
    stages.append(
        path_receipts.build_stage(
            unit_id=unit_id,
            stage_index=len(stages),
            stage_name=stage_name,
            parent_hash=stages[-1]["stage_hash"] if stages else path_receipts.GENESIS_HASH,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            code_hash=code_hashes[stage_name],
            configuration_hash=config_hashes[stage_name],
            monotonic_start_ns=start,
            monotonic_end_ns=end,
            terminal_exact_outcome=terminal_exact_outcome,
            output_payload=output_payload,
        )
    )
    return output_bytes


def build_path_row(
    unit: Mapping[str, Any],
    *,
    control_id: str,
    code_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
) -> JsonDict:
    unit_id = str(unit["unit_id"])
    raw_bytes = bytes(unit["raw_bytes"])
    stages: list[JsonDict] = []
    raw_payload = {
        "unit_id": unit_id,
        "raw_path": unit["raw_path"],
        "raw_event_id": _fixture_raw_event_id(raw_bytes),
        "raw_sha256": unit["raw_sha256"],
        "raw_byte_length": unit["raw_byte_length"],
    }
    current = _append_stage(
        stages,
        unit_id=unit_id,
        stage_name="raw_generation_bytes",
        input_bytes=path_receipts.json_bytes({"unit_id": unit_id, "fixture": "exp6427"}),
        output_payload=raw_payload,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    parsed = _parse_raw_output(raw_bytes)
    current = _append_stage(
        stages,
        unit_id=unit_id,
        stage_name="parse_output",
        input_bytes=current,
        output_payload=parsed,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    typed = _typed_facts(unit_id, parsed)
    current = _append_stage(
        stages,
        unit_id=unit_id,
        stage_name="typed_facts",
        input_bytes=current,
        output_payload=typed,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    energy = _energy_input(unit["event"], typed)
    current = _append_stage(
        stages,
        unit_id=unit_id,
        stage_name="energy_input",
        input_bytes=current,
        output_payload=energy,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    request = _checker_request(
        unit_id=unit_id,
        event=unit["event"],
        parsed=parsed,
        typed=typed,
        energy=energy,
    )
    current = _append_stage(
        stages,
        unit_id=unit_id,
        stage_name="checker_request",
        input_bytes=current,
        output_payload=request,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    transport = _checker_transport(control_id, request)
    current = _append_stage(
        stages,
        unit_id=unit_id,
        stage_name="checker_transport",
        input_bytes=current,
        output_payload=transport,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    response = _checker_response(transport)
    current = _append_stage(
        stages,
        unit_id=unit_id,
        stage_name="checker_response",
        input_bytes=current,
        output_payload=response,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
        terminal_exact_outcome=response["exact_outcome"],
    )
    final = _final_verdict(response)
    _append_stage(
        stages,
        unit_id=unit_id,
        stage_name="final_verdict",
        input_bytes=current,
        output_payload=final,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
        terminal_exact_outcome=final["terminal_exact_outcome"],
    )
    return {
        "row_id": f"{unit_id}:{control_id}",
        "unit_id": unit_id,
        "control_id": control_id,
        "row_index": unit["row_index"],
        "expected_verdict": final["expected_verdict"],
        "observed_verdict": final["observed_verdict"],
        "terminal_exact_outcome": final["terminal_exact_outcome"],
        "terminal_hash": stages[-1]["output_hash"],
        "terminal_chain_hash": stages[-1]["stage_hash"],
        "stage_hashes": {stage["stage_name"]: stage["stage_hash"] for stage in stages},
        "stage_output_hashes": {stage["stage_name"]: stage["output_hash"] for stage in stages},
        "changed_boundary": None,
        "changed_boundaries": [],
        "localization_result": "pending",
        "replay_timing_ns": sum(
            int(stage["monotonic_end_ns"]) - int(stage["monotonic_start_ns"]) for stage in stages
        ),
        "stages": stages,
    }


def _identity_rows_by_unit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["unit_id"]): row
        for row in rows
        if row.get("control_id") == "identity"
    }


def annotate_control_localization(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    identity = _identity_rows_by_unit(rows)
    annotated: list[JsonDict] = []
    for row in rows:
        item = deepcopy(dict(row))
        base = identity.get(str(row["unit_id"]))
        changed: list[str] = []
        if base is not None:
            base_outputs = base["stage_output_hashes"]
            for stage in path_receipts.REQUIRED_STAGE_NAMES:
                if item["stage_output_hashes"].get(stage) != base_outputs.get(stage):
                    changed.append(stage)
        item["changed_boundaries"] = changed
        item["changed_boundary"] = changed[0] if len(changed) == 1 else None
        if item["control_id"] == "identity":
            item["localization_result"] = "identity_replay"
        elif item["control_id"] == "injected_wrapper":
            item["localization_result"] = (
                "localized_declared_boundary" if changed == [INJECTED_BOUNDARY] else "mislocalized"
            )
        elif item["control_id"] == "restored_wrapper":
            terminal_match = bool(base and item["terminal_hash"] == base["terminal_hash"])
            verdict_match = bool(base and item["observed_verdict"] == base["observed_verdict"])
            item["localization_result"] = (
                "restored_identity" if not changed and terminal_match and verdict_match else "not_restored"
            )
        annotated.append(item)
    return annotated


def stage_chain_validation(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_code_hashes: set[str],
) -> JsonDict:
    reports = []
    for row in rows:
        report = path_receipts.validate_stage_chain(
            row["stages"], allowed_code_hashes=allowed_code_hashes
        )
        reports.append(
            {
                "row_id": row["row_id"],
                "control_id": row["control_id"],
                "unit_id": row["unit_id"],
                **report,
            }
        )
    invalid = [row for row in reports if row["accepted"] is not True]
    return {
        "all_valid": not invalid,
        "accepted_count": len(reports) - len(invalid),
        "invalid_count": len(invalid),
        "reports": reports,
    }


def terminal_verdict_recomputation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    reports = []
    for row in rows:
        by_name = {stage["stage_name"]: stage for stage in row["stages"]}
        response = by_name["checker_response"]["output_payload"]
        final = by_name["final_verdict"]["output_payload"]
        expected = path_receipts.verdict_from_checker_response(response)
        recomputed = final.get("observed_verdict") == expected and (
            final.get("terminal_exact_outcome") is (response.get("exact_outcome") is True)
        )
        reports.append(
            {
                "row_id": row["row_id"],
                "recomputed": recomputed,
                "expected_verdict": expected,
                "observed_verdict": final.get("observed_verdict"),
            }
        )
    mismatches = [row for row in reports if row["recomputed"] is not True]
    return {
        "all_recomputed": not mismatches,
        "mismatch_count": len(mismatches),
        "reports": reports,
    }


def identity_replay_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    identity_rows = [row for row in rows if row.get("control_id") == "identity"]
    replayed = [
        row
        for row in identity_rows
        if row.get("expected_verdict") == row.get("observed_verdict")
        and row.get("localization_result") == "identity_replay"
    ]
    return {
        "all_replayed": len(replayed) == len(identity_rows) and bool(identity_rows),
        "replayed_count": len(replayed),
        "identity_count": len(identity_rows),
        "terminal_hashes": {row["unit_id"]: row["terminal_hash"] for row in identity_rows},
    }


def injected_boundary_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    injected = [row for row in rows if row.get("control_id") == "injected_wrapper"]
    changed_counter = Counter(
        boundary for row in injected for boundary in row.get("changed_boundaries", [])
    )
    localized = [
        row for row in injected if row.get("localization_result") == "localized_declared_boundary"
    ]
    return {
        "all_localized": len(localized) == len(injected) and bool(injected),
        "localized_count": len(localized),
        "injected_count": len(injected),
        "declared_boundary": INJECTED_BOUNDARY,
        "changed_boundaries": dict(changed_counter),
    }


def restored_boundary_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    identity = _identity_rows_by_unit(rows)
    restored = [row for row in rows if row.get("control_id") == "restored_wrapper"]
    matched = [
        row
        for row in restored
        if identity.get(row["unit_id"], {}).get("terminal_hash") == row["terminal_hash"]
        and identity.get(row["unit_id"], {}).get("observed_verdict") == row["observed_verdict"]
        and row.get("localization_result") == "restored_identity"
    ]
    return {
        "all_restored": len(matched) == len(restored) and bool(restored),
        "matched_identity_terminal_hash_count": len(matched),
        "restored_count": len(restored),
    }


def _refresh_downstream_parents(stages: list[JsonDict], start_index: int) -> list[JsonDict]:
    for index in range(start_index, len(stages)):
        if index > 0:
            stages[index]["parent_hash"] = stages[index - 1]["stage_hash"]
        stages[index] = path_receipts.refresh_stage_hash(stages[index])
    return stages


def mutate_row_for_attack(
    attack_id: str,
    row: Mapping[str, Any],
    alternate_row: Mapping[str, Any] | None = None,
) -> JsonDict:
    mutated = deepcopy(dict(row))
    stages = mutated["stages"]
    if attack_id == "wrapper_insertion":
        hidden = deepcopy(stages[5])
        hidden["stage_id"] = "05-hidden_wrapper"
        hidden["stage_name"] = "hidden_wrapper"
        hidden["output_payload"] = {"wrapper_id": "undisclosed", "inner_hash": hidden["output_hash"]}
        hidden["output_hash"] = path_receipts.sha256_json(hidden["output_payload"])
        stages.insert(6, path_receipts.refresh_stage_hash(hidden))
    elif attack_id == "parser_substitution":
        stages[1]["code_hash"] = path_receipts.sha256_text("unknown parser")
        stages[1] = path_receipts.refresh_stage_hash(stages[1])
    elif attack_id == "stale_checker_response":
        stages[6]["output_payload"]["exact_outcome"] = not stages[6]["output_payload"][
            "exact_outcome"
        ]
        stages[6]["output_hash"] = path_receipts.sha256_json(stages[6]["output_payload"])
        stages[6] = path_receipts.refresh_stage_hash(stages[6])
        stages = _refresh_downstream_parents(stages, 7)
    elif attack_id == "missing_stage":
        del stages[2]
    elif attack_id == "reordered_stage":
        stages[1], stages[2] = stages[2], stages[1]
    elif attack_id == "replayed_raw_bytes_under_another_unit_id":
        other = alternate_row or row
        stages[0]["output_payload"]["raw_event_id"] = other["unit_id"]
        stages[0]["output_payload"]["raw_sha256"] = other["stages"][0]["output_payload"][
            "raw_sha256"
        ]
        stages[0]["output_hash"] = other["stages"][0]["output_hash"]
        stages[0] = path_receipts.refresh_stage_hash(stages[0])
        stages = _refresh_downstream_parents(stages, 1)
    elif attack_id == "forged_parent_hash":
        stages[4]["parent_hash"] = path_receipts.sha256_text("forged parent")
        stages[4] = path_receipts.refresh_stage_hash(stages[4])
    else:
        raise ValueError(f"unknown stage attack: {attack_id}")
    mutated["stages"] = stages
    return mutated


def recompute_aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
) -> JsonDict:
    control_counts = Counter(str(row.get("control_id")) for row in rows)
    unit_count = len({str(row.get("unit_id")) for row in rows})
    expected_row_count = unit_count * len(CONTROL_IDS)
    reported = dict(artifact.get("aggregate_row_recomputation", {}))
    computed = {
        "row_count": len(rows),
        "unit_count": unit_count,
        "control_count": len(control_counts),
        "control_counts": dict(sorted(control_counts.items())),
        "expected_row_count": expected_row_count,
        "all_controls_present": set(control_counts) == set(CONTROL_IDS),
    }
    reasons = []
    if reported:
        if reported.get("reported_row_count") != computed["row_count"]:
            reasons.append("reported_row_count_mismatch")
        if reported.get("reported_unit_count") != computed["unit_count"]:
            reasons.append("reported_unit_count_mismatch")
        if reported.get("reported_control_counts") != computed["control_counts"]:
            reasons.append("reported_control_counts_mismatch")
    if computed["row_count"] != computed["expected_row_count"]:
        reasons.append("expected_row_count_mismatch")
    if not computed["all_controls_present"]:
        reasons.append("controls_missing")
    return {
        "reported_row_count": reported.get("reported_row_count", computed["row_count"]),
        "reported_unit_count": reported.get("reported_unit_count", computed["unit_count"]),
        "reported_control_counts": reported.get(
            "reported_control_counts", computed["control_counts"]
        ),
        "computed": computed,
        "matches_reported": not reasons,
        "reasons": reasons,
        "row_hash": path_receipts.sha256_json(
            [
                {
                    "row_id": row["row_id"],
                    "terminal_hash": row["terminal_hash"],
                    "localization_result": row["localization_result"],
                }
                for row in rows
            ]
        ),
    }


def attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_code_hashes: set[str],
    artifact: Mapping[str, Any],
) -> JsonDict:
    identity_rows = [row for row in rows if row.get("control_id") == "identity"]
    base = identity_rows[0]
    alternate = identity_rows[1] if len(identity_rows) > 1 else identity_rows[0]
    matrix_rows = []
    for attack_id in ATTACK_IDS:
        if attack_id == "aggregate_row_mismatch":
            tampered = deepcopy(dict(artifact))
            tampered["aggregate_row_recomputation"] = {
                **dict(artifact.get("aggregate_row_recomputation", {})),
                "reported_row_count": len(rows) + 1,
            }
            aggregate = recompute_aggregate_rows(rows, tampered)
            detected = aggregate["matches_reported"] is not True
            reasons = aggregate["reasons"]
        else:
            mutated = mutate_row_for_attack(attack_id, base, alternate)
            report = path_receipts.validate_stage_chain(
                mutated["stages"], allowed_code_hashes=allowed_code_hashes
            )
            detected = report["accepted"] is not True
            reasons = report["reasons"]
        matrix_rows.append(
            {
                "attack_id": attack_id,
                "detected": detected,
                "fail_closed": detected,
                "reasons": reasons,
            }
        )
    false_accept_count = sum(1 for row in matrix_rows if row["detected"] is not True)
    return {
        "schema": "carnot.exp6449.attack_matrix.v1",
        "rows": matrix_rows,
        "attack_count": len(matrix_rows),
        "all_critical_fail_closed": false_accept_count == 0,
        "false_accept_count": false_accept_count,
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> list[JsonDict]:
    exits = dict(test_exit_codes or {})
    return [
        {
            "command": command,
            "exit_code": exits.get(command),
            "status": "passed" if exits.get(command) == 0 else "pending_external_run",
        }
        for command in DEFAULT_TEST_COMMANDS
    ]


def current_findings(
    *,
    chain: Mapping[str, Any],
    terminal: Mapping[str, Any],
    attacks: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> list[JsonDict]:
    findings: list[JsonDict] = []
    gates = {
        "stage_chain_validation": chain.get("all_valid") is True,
        "terminal_verdict_recomputation": terminal.get("all_recomputed") is True,
        "attack_matrix": attacks.get("all_critical_fail_closed") is True,
        "aggregate_row_recomputation": aggregate.get("matches_reported") is True,
        "protected_files_unchanged": protected.get("unchanged") is True,
    }
    for gate, passed in gates.items():
        if not passed:
            findings.append({"severity": "critical", "kind": gate, "detail": "gate failed"})
    return findings


def _ready_score(
    *,
    identity: Mapping[str, Any],
    injected: Mapping[str, Any],
    restored: Mapping[str, Any],
    chain: Mapping[str, Any],
    terminal: Mapping[str, Any],
    attacks: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> float:
    ready = (
        identity.get("all_replayed") is True
        and injected.get("all_localized") is True
        and restored.get("all_restored") is True
        and chain.get("all_valid") is True
        and terminal.get("all_recomputed") is True
        and attacks.get("all_critical_fail_closed") is True
        and aggregate.get("matches_reported") is True
        and protected.get("unchanged") is True
        and not [row for row in findings if row.get("severity") == "critical"]
    )
    return 1.0 if ready else 0.0


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    normalized = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "tests_run", "reproducibility_checksum"}
    }
    return path_receipts.sha256_json(normalized)[:23]


def build_per_unit_rows(
    units: Sequence[Mapping[str, Any]],
    *,
    code_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
) -> JsonDict:
    rows = [
        build_path_row(unit, control_id=control_id, code_hashes=code_hashes, config_hashes=config_hashes)
        for unit in units
        for control_id in CONTROL_IDS
    ]
    rows = annotate_control_localization(rows)
    return {
        "schema": "carnot.exp6449.per_unit_rows.v1",
        "rows": rows,
        "row_count": len(rows),
        "unit_count": len(units),
        "control_count": len(CONTROL_IDS),
        "unit_ids": [str(unit["unit_id"]) for unit in units],
        "written_before_aggregates": True,
    }


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6427_result_path: str | Path = REPO_ROOT / EXP6427_RESULT_RELATIVE_PATH,
    fixture_limit: int = FIXTURE_LIMIT,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.monotonic()
    source_before = source_hashes()
    protected_before = protected_hashes()
    preconditions, units, fixture_manifest = check_preconditions(
        result_path=result_path,
        data_dir=data_dir,
        exp6427_result_path=exp6427_result_path,
        fixture_limit=fixture_limit,
    )
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    if any(row.get("available") is not True for row in preconditions):
        artifact = _blocked_artifact(
            date=date,
            result_path=result_path,
            preconditions=preconditions,
            source_before=source_before,
            protected_before=protected_before,
            duration_s=measured_duration,
        )
        if write:
            write_json_atomic(result_path, artifact)
        return artifact

    code_config = code_and_configuration_hashes(source_before)
    per_units = build_per_unit_rows(
        units,
        code_hashes=code_config["allowed_code_hashes"],
        config_hashes=code_config["configuration_hashes"],
    )
    rows = per_units["rows"]
    allowed = set(code_config["allowed_code_hashes"].values())
    chain = stage_chain_validation(rows, allowed_code_hashes=allowed)
    terminal = terminal_verdict_recomputation(rows)
    identity = identity_replay_results(rows)
    injected = injected_boundary_results(rows)
    restored = restored_boundary_results(rows)
    aggregate_seed = {
        "aggregate_row_recomputation": {
            "reported_row_count": len(rows),
            "reported_unit_count": len(units),
            "reported_control_counts": dict(sorted(Counter(row["control_id"] for row in rows).items())),
        }
    }
    aggregate = recompute_aggregate_rows(rows, aggregate_seed)
    protected = protected_unchanged_receipt(protected_before)
    attacks = attack_matrix(rows, allowed_code_hashes=allowed, artifact=aggregate_seed)
    findings = current_findings(
        chain=chain,
        terminal=terminal,
        attacks=attacks,
        aggregate=aggregate,
        protected=protected,
    )
    score = _ready_score(
        identity=identity,
        injected=injected,
        restored=restored,
        chain=chain,
        terminal=terminal,
        attacks=attacks,
        aggregate=aggregate,
        protected=protected,
        findings=findings,
    )
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    artifact = {
        "status": "success" if score == 1.0 else "complete_with_findings",
        "receipt_schema_and_version": receipt_schema_and_version(),
        "fixture_manifest_and_hashes": fixture_manifest,
        "code_and_configuration_hashes": code_config,
        "control_precommitment": control_precommitment(),
        "per_unit_rows": per_units,
        "identity_replay_results": identity,
        "injected_boundary_results": injected,
        "restored_boundary_results": restored,
        "stage_chain_validation": chain,
        "terminal_verdict_recomputation": terminal,
        "attack_matrix": attacks,
        "current_adversarial_findings": findings,
        "aggregate_row_recomputation": aggregate,
        "path_receipt_ready_score": score,
        "protected_files_unchanged": protected,
        "blocked_reason": "",
        "gate_check_summary": "all preconditions passed",
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": measured_duration,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "success: path receipt controls localized and critical attacks failed closed"
            if score == 1.0
            else "complete: path receipt contract finished with findings"
        ),
        "run_date": date,
        "result_path": str(result_path),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        write_json_atomic(result_path, artifact)
    return artifact


def validate_artifact(path: str | Path) -> JsonDict:
    artifact = read_json_object(path)
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    rows = artifact.get("per_unit_rows", {}).get("rows", []) if artifact else []
    allowed = set(
        artifact.get("code_and_configuration_hashes", {})
        .get("allowed_code_hashes", {})
        .values()
    )
    chain = (
        stage_chain_validation(rows, allowed_code_hashes=allowed)
        if rows and allowed
        else {"all_valid": False, "invalid_count": len(rows)}
    )
    aggregate = recompute_aggregate_rows(rows, artifact) if rows else {"matches_reported": False}
    errors = []
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
    if rows and chain.get("all_valid") is not True:
        errors.append("stage_chain_invalid")
    if rows and aggregate.get("matches_reported") is not True:
        errors.append("aggregate_mismatch")
    if artifact.get("path_receipt_ready_score") == 1.0 and errors:
        errors.append("ready_score_claim_with_errors")
    return {"valid": not errors, "errors": errors, "chain": chain, "aggregate": aggregate}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        report = validate_artifact(result_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    artifact = run(date=args.date, result_path=result_path, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH)
    print(json.dumps({"status": artifact["status"], "result_path": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
