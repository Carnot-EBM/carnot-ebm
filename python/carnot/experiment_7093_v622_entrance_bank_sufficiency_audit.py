"""Promote the tested entrance-bank audit to the V622 recovery task.

Spec refs: REQ-VERIFY-7093 and SCENARIO-VERIFY-7093-*.

Exp7087 owns the row-level science. This module keeps that implementation and
adds the new identity, fixed inputs, missing attacks, focused test receipt, and
terminal result rules. The audit reads saved bytes and never loads a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7086_v621_three_family_entrance_bank as producer
from carnot import experiment_7087_v621_entrance_bank_sufficiency_audit as legacy


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "carnot.experiment_7093_v622_entrance_bank_sufficiency_audit"
RUN_DATE = "20260907"
RANDOM_SEED = 7_093_202_609_07
EXPERIMENT_ID = "experiment_7093_v622_entrance_bank_sufficiency_audit"
SCHEMA = "carnot.experiment_7093.v622_entrance_bank_sufficiency_audit.v1"
INFERENCE_SUBSTRATE = legacy.INFERENCE_SUBSTRATE
RESULT_PATH = REPO_ROOT / "results/experiment_7093_v622_entrance_bank_sufficiency_audit.json"
BANK_PATH = legacy.BANK_PATH
FIXTURE_PATH = legacy.FIXTURE_PATH
AUDIT_ROOT = REPO_ROOT / "results/.experiment_7093_v622_entrance_bank_sufficiency_audit"
PINNED_BANK_SHA256 = legacy.PINNED_BANK_SHA256
PINNED_FIXTURE_SHA256 = legacy.PINNED_FIXTURE_SHA256
MINIMUM_HEADROOM_UNITS = legacy.MINIMUM_HEADROOM_UNITS
REQUIRED_SOURCE_MODEL_IDS = tuple(producer.REQUIRED_MODEL_IDS)
LEGACY_SMALL_MODEL_MARKERS = ("qwen3.5-0.8b", "gemma-4-e4b-it")

FOCUSED_TEST_FILES = (
    "tests/python/test_experiment_7087_v621_entrance_bank_sufficiency_audit.py",
    "tests/python/test_experiment_7093_v622_entrance_bank_sufficiency_audit.py",
)
FOCUSED_COVERAGE_SCOPES = (
    "carnot.experiment_7087_v621_entrance_bank_sufficiency_audit",
    "carnot.experiment_7093_v622_entrance_bank_sufficiency_audit",
)
FOCUSED_COVERAGE_FILES = tuple(
    f"python/{scope.replace('.', '/')}.py" for scope in FOCUSED_COVERAGE_SCOPES
)
REQUIRED_ATTACKS = (
    "family_deletion",
    "source_swap",
    "raw_byte_mutation",
    "duplicate",
    "conflict",
    "future_label_leakage",
)
REQUIRED_ARTIFACT_FIELDS = tuple(
    dict.fromkeys(
        (
            *legacy.REQUIRED_ARTIFACT_FIELDS,
            "execution_venue",
            "source_model_specs",
            "focused_test_rows",
        )
    )
)
FIELD_PRINCIPLES = {
    **legacy.FIELD_PRINCIPLES,
    "execution_venue": "The host receipt prevents a saved replay from implying new model or board execution.",
    "source_model_specs": "The exact three-family roster prevents legacy-small model substitution or regeneration.",
    "focused_test_rows": "Scoped coverage proves the promoted code and reused audit logic ran without a repository-wide denominator.",
}

canonical_json = legacy.canonical_json
sha256_text = legacy.sha256_text
sha256_file = legacy.sha256_file
gate_row = legacy.gate_row
gate_summary = legacy.gate_summary
artifact_checksum = legacy.artifact_checksum
build_required_support_schema = legacy.build_required_support_schema
recompute_audit = legacy.recompute_audit
measure_headroom = legacy.measure_headroom
_models_and_seeds = legacy._models_and_seeds
_artifact_gate_rows = legacy._artifact_gate_rows
_public_preconditions = legacy._public_preconditions


def focused_test_command() -> list[str]:
    """Build the fixed command that measures only the two audit modules."""

    return [
        sys.executable,
        "-m",
        "coverage",
        "run",
        f"--include={','.join(FOCUSED_COVERAGE_FILES)}",
        "-m",
        "pytest",
        *FOCUSED_TEST_FILES,
        "-q",
        "-o",
        "addopts=",
    ]


def focused_coverage_report_command(*, new_code_only: bool) -> list[str]:
    """Build a two-module report or the strict report for only new code."""

    included = FOCUSED_COVERAGE_FILES[-1:] if new_code_only else FOCUSED_COVERAGE_FILES
    command = [
        sys.executable,
        "-m",
        "coverage",
        "report",
        f"--include={','.join(included)}",
        "--show-missing",
    ]
    if new_code_only:
        command.append("--fail-under=100")
    return command


def _attack_row(name: str, result: Mapping[str, Any], baseline_count: int) -> JsonDict:
    """Reduce one mutation from detailed replay gates without using pooled rates."""

    return {
        "attack": name,
        "baseline_raw_row_count": baseline_count,
        "attacked_raw_row_count": baseline_count,
        "pooled_row_count_preserved": True,
        "attack_detected": bool(result.get("errors"))
        and (
            result.get("authenticity_passed") is False
            or result.get("family_sufficiency_passed") is False
            or result.get("conflict_resolution_passed") is False
            or result.get("leakage_passed") is False
        ),
        "failed_checks": deepcopy(list(result.get("errors", []))),
    }


def run_counterfactual_attacks(
    bank: Mapping[str, Any], fixture: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[JsonDict]:
    """Keep the four tested attacks and add count-preserving duplicate and leakage attacks."""

    rows = legacy.run_counterfactual_attacks(bank, fixture, schema)
    for row in rows:
        if row.get("attack") == "label_conflict":
            row["attack"] = "conflict"
    raw_rows = [deepcopy(dict(row)) for row in bank.get("raw_proposal_rows", [])]
    baseline_count = len(raw_rows)

    duplicate_rows = deepcopy(raw_rows)
    if len(duplicate_rows) >= 2:
        duplicate_rows[-1] = deepcopy(duplicate_rows[0])
    duplicate_bank = dict(bank)
    duplicate_bank["raw_proposal_rows"] = duplicate_rows
    duplicate_result = legacy.recompute_audit(duplicate_bank, fixture, schema)

    leakage_rows = deepcopy(raw_rows)
    if leakage_rows:
        generation = deepcopy(dict(leakage_rows[0].get("generation_config") or {}))
        generation["future_label"] = {"reachable": True}
        leakage_rows[0]["generation_config"] = generation
    leakage_bank = dict(bank)
    leakage_bank["raw_proposal_rows"] = leakage_rows
    leakage_result = legacy.recompute_audit(leakage_bank, fixture, schema)

    return [
        *rows,
        _attack_row("duplicate", duplicate_result, baseline_count),
        _attack_row("future_label_leakage", leakage_result, baseline_count),
    ]


def _upstream_citations(source_artifact_hashes: Mapping[str, Any]) -> list[JsonDict]:
    """Name the two fixed inputs and the row fields that the audit reads."""

    return [
        {
            "path": str(BANK_PATH.relative_to(REPO_ROOT)),
            "sha256": source_artifact_hashes.get(BANK_PATH.name),
            "fields_imported": ["raw_proposal_rows", "primary execution receipts"],
        },
        {
            "path": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
            "sha256": source_artifact_hashes.get(FIXTURE_PATH.name),
            "fields_imported": ["unit_rows", "entrance_rows", "split_manifest"],
        },
    ]


def _promote_artifact(
    artifact: Mapping[str, Any],
    *,
    source_model_specs: Sequence[Mapping[str, Any]],
    focused_test_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Apply only the V622 identity and terminal rules to Exp7087 evidence."""

    promoted = deepcopy(dict(artifact))
    promoted.update(
        {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "field_principles": deepcopy(FIELD_PRINCIPLES),
            "execution_venue": "host",
            "source_model_specs": [deepcopy(dict(row)) for row in source_model_specs],
            "focused_test_rows": [deepcopy(dict(row)) for row in focused_test_rows],
            "random_seed": RANDOM_SEED,
        }
    )
    blocked = promoted.get("verdict_class") == "blocked"
    support_ready = promoted.get("entrance_support_audit_ready_score") == 1
    headroom_ready = promoted.get("entrance_selector_headroom_ready_score") == 1
    verdict_class = (
        "blocked"
        if blocked
        else "circular_positive"
        if support_ready and headroom_ready
        else "null"
    )
    promoted["verdict_class"] = verdict_class
    promoted["honest_verdict"] = (
        "blocked: entrance support audit precondition failed"
        if blocked
        else "circular_positive: support and selector headroom audit ready"
        if verdict_class == "circular_positive"
        else "null: completed entrance support audit is insufficient"
    )
    promoted["reproducibility_checksum"] = artifact_checksum(promoted)
    return promoted


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Mapping[str, Any],
    source_artifact_hashes: Mapping[str, Any],
    source_model_specs: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Promote a complete Exp7087 preflight block to the Exp7093 schema."""

    artifact = legacy.build_blocked_artifact(
        run_date=run_date,
        duration_s=duration_s,
        preconditions=preconditions,
        source_artifact_hashes=source_artifact_hashes,
    )
    return _promote_artifact(
        artifact,
        source_model_specs=source_model_specs,
        focused_test_rows=[],
    )


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    bank: Mapping[str, Any],
    fixture: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    source_artifact_hashes: Mapping[str, Any],
    replay: Mapping[str, Any],
    fresh_process_rows: Sequence[Mapping[str, Any]],
    focused_test_rows: Sequence[Mapping[str, Any]],
    minimum_headroom_units: int = MINIMUM_HEADROOM_UNITS,
    counterfactual_swap_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the full Exp7087 evidence first, then apply the recovery contract."""

    artifact = legacy.build_artifact(
        run_date=run_date,
        duration_s=duration_s,
        bank=bank,
        fixture=fixture,
        preconditions=preconditions,
        source_artifact_hashes=source_artifact_hashes,
        replay=replay,
        fresh_process_rows=fresh_process_rows,
        minimum_headroom_units=minimum_headroom_units,
        counterfactual_swap_rows=counterfactual_swap_rows,
    )
    artifact["gpu_telemetry_rows"] = [
        {**deepcopy(dict(row)), "scope": "upstream"}
        for row in artifact.get("gpu_telemetry_rows", [])
    ]
    artifact["cited_upstream_artifacts"] = _upstream_citations(source_artifact_hashes)
    return _promote_artifact(
        artifact,
        source_model_specs=bank.get("model_specs", []),
        focused_test_rows=focused_test_rows,
    )


def _legacy_projection(artifact: Mapping[str, Any]) -> JsonDict:
    """Restore the old terminal rule so the tested Exp7087 validator can audit its rows."""

    projected = deepcopy(dict(artifact))
    projected["field_principles"] = deepcopy(legacy.FIELD_PRINCIPLES)
    support_ready = projected.get("entrance_support_audit_ready_score") == 1
    headroom_ready = projected.get("entrance_selector_headroom_ready_score") == 1
    blocked = projected.get("verdict_class") == "blocked"
    projected_class = (
        "blocked"
        if blocked
        else "circular_positive"
        if support_ready and headroom_ready
        else "partial"
        if support_ready
        else "null"
    )
    projected["verdict_class"] = projected_class
    projected["honest_verdict"] = f"{projected_class}: legacy validation projection"
    projected["reproducibility_checksum"] = artifact_checksum(projected)
    return projected


def _focused_receipt_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check that one successful receipt names both tests and both coverage scopes."""

    expected_command = focused_test_command()
    return len(rows) == 1 and all(
        (
            row.get("command") == expected_command
            and row.get("coverage_report_command")
            == focused_coverage_report_command(new_code_only=False)
            and row.get("new_code_coverage_report_command")
            == focused_coverage_report_command(new_code_only=True)
            and row.get("test_files") == list(FOCUSED_TEST_FILES)
            and row.get("coverage_scopes") == list(FOCUSED_COVERAGE_SCOPES)
            and row.get("coverage_fail_under") == 100
            and row.get("returncode") == 0
            and row.get("test_returncode") == 0
            and row.get("coverage_returncode") == 0
            and row.get("new_code_coverage_returncode") == 0
            and row.get("passed") is True
        )
        for row in rows
    )


def validate_artifact(artifact: Any) -> list[str]:
    """Run the reused validator, then enforce the stricter Exp7093 contract."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors = [f"missing_field:{field}" for field in missing]
    blocked = artifact.get("verdict_class") == "blocked"
    source_ids = [str(row.get("hf_id")) for row in artifact.get("source_model_specs", [])]
    attack_rows = list(artifact.get("counterfactual_swap_rows", []))
    attack_names = [str(row.get("attack")) for row in attack_rows]
    source_hashes = dict(artifact.get("source_artifact_hashes") or {})
    support_ready = artifact.get("entrance_support_audit_ready_score") == 1
    headroom_ready = artifact.get("entrance_selector_headroom_ready_score") == 1
    expected_class = (
        "blocked"
        if blocked
        else "circular_positive"
        if support_ready and headroom_ready
        else "null"
    )
    checks = [
        (
            "field_principles_mismatch",
            set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS)
            or any(
                not str(value).strip() for value in artifact.get("field_principles", {}).values()
            ),
        ),
        (
            "identity_mismatch",
            artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID,
        ),
        ("run_date_mismatch", artifact.get("run_date") != RUN_DATE),
        ("execution_venue_mismatch", artifact.get("execution_venue") != "host"),
        (
            "upstream_hash_mismatch",
            not blocked
            and (
                source_hashes.get(BANK_PATH.name) != PINNED_BANK_SHA256
                or source_hashes.get(FIXTURE_PATH.name) != PINNED_FIXTURE_SHA256
            ),
        ),
        (
            "source_model_specs_mismatch",
            not blocked
            and (
                source_ids != list(REQUIRED_SOURCE_MODEL_IDS)
                or list(dict(artifact.get("required_support_schema") or {}).get("model_ids", []))
                != list(REQUIRED_SOURCE_MODEL_IDS)
            ),
        ),
        (
            "focused_test_rows_mismatch",
            (not blocked and not _focused_receipt_valid(artifact.get("focused_test_rows", [])))
            or (blocked and bool(artifact.get("focused_test_rows"))),
        ),
        (
            "counterfactual_attack_rows_mismatch",
            (
                not blocked
                and (
                    len(attack_names) != len(REQUIRED_ATTACKS)
                    or set(attack_names) != set(REQUIRED_ATTACKS)
                    or not all(row.get("attack_detected") is True for row in attack_rows)
                )
            )
            or (blocked and bool(attack_rows)),
        ),
        ("verdict_class_mismatch", artifact.get("verdict_class") != expected_class),
        (
            "honest_verdict_prefix_mismatch",
            not str(artifact.get("honest_verdict", "")).startswith(f"{expected_class}:"),
        ),
        (
            "reproducibility_checksum_mismatch",
            artifact.get("reproducibility_checksum") != artifact_checksum(artifact),
        ),
    ]
    errors.extend(name for name, failed in checks if failed)
    legacy_errors = legacy.validate_artifact(_legacy_projection(artifact))
    errors.extend(f"legacy:{error}" for error in legacy_errors)
    return list(dict.fromkeys(errors))


def _fresh_worker(
    bank_path: Path, fixture_path: Path, output_path: Path
) -> int:  # pragma: no cover - private child process boundary.
    """Use the tested Exp7087 replay inside the new module process."""

    return legacy._fresh_worker(bank_path, fixture_path, output_path)


def run_fresh_replay(
    bank_path: Path, fixture_path: Path, audit_dir: Path
) -> tuple[JsonDict, list[JsonDict]]:
    """Start the replay through the Exp7093 module and record process identity."""

    audit_dir.mkdir(parents=True, exist_ok=False)
    output_path = audit_dir / "fresh-replay.json"
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--fresh-worker",
        "--bank-path",
        str(bank_path),
        "--fixture-path",
        str(fixture_path),
        "--worker-output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=audit_dir,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"fresh_replay_failed:returncode={completed.returncode}:stderr={completed.stderr[-500:]}"
        )
    replay = json.loads(output_path.read_text(encoding="utf-8"))
    process = dict(replay.pop("_fresh_process"))
    process.update(
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout_hash": sha256_text(completed.stdout),
            "stderr_hash": sha256_text(completed.stderr),
            "child_distinct_from_parent": process.get("pid") != os.getpid(),
        }
    )
    process["passed"] = bool(process.get("passed")) and process["child_distinct_from_parent"]
    return replay, [process]


def collect_preconditions(
    *,
    bank_path: Path,
    fixture_path: Path,
    result_path: Path,
    audit_root: Path,
) -> JsonDict:
    """Extend the tested preflight with exact roster and exclusive-path checks."""

    result = legacy.collect_preconditions(
        bank_path=bank_path,
        fixture_path=fixture_path,
        result_path=result_path,
        audit_root=audit_root,
    )
    bank = dict(result.get("bank") or {})
    spec_ids = [str(row.get("hf_id")) for row in bank.get("model_specs", [])]
    identity_rows = list(bank.get("model_identity_rows", []))
    identity_ids = [str(row.get("model_id")) for row in identity_rows]
    identity_ready = (
        spec_ids == list(REQUIRED_SOURCE_MODEL_IDS)
        and identity_ids == list(REQUIRED_SOURCE_MODEL_IDS)
        and all(
            row.get("passed") is True and row.get("identity_matches") is True
            for row in identity_rows
        )
    )
    checkpoint_receipts = list(result.get("checkpoint_receipts", []))
    checkpoint_ready = len(checkpoint_receipts) == 2 * len(REQUIRED_SOURCE_MODEL_IDS) and all(
        row.get("passed") is True for row in checkpoint_receipts
    )
    lowered_ids = " ".join(spec_ids).lower()
    legacy_absent = not any(marker in lowered_ids for marker in LEGACY_SMALL_MODEL_MARKERS)
    extra_checks = [
        gate_row("checkpoint_readability", True, checkpoint_ready),
        gate_row(
            "source_model_identities", list(REQUIRED_SOURCE_MODEL_IDS), spec_ids, identity_ready
        ),
        gate_row("legacy_small_models_absent", True, legacy_absent),
        gate_row("result_path_absent", True, not result_path.exists()),
    ]
    result["checks"] = [*result.get("checks", []), *extra_checks]
    result["all_passed"] = all(row.get("passed") is True for row in result["checks"])
    result["source_model_identity_receipts"] = deepcopy(identity_rows)
    return result


def _hash_if_file(path: Path) -> str | None:  # pragma: no cover - dated source receipt.
    """Return one content hash without treating a missing precondition as an exception."""

    return sha256_file(path) if path.is_file() else None


def _source_hashes(
    *, bank_path: Path = BANK_PATH, fixture_path: Path = FIXTURE_PATH
) -> JsonDict:  # pragma: no cover - hashes the live checkout.
    """Bind the result to both upstream artifacts and every reviewed source file."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-references.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("openspec/capabilities/research-reporting/spec.md"),
        Path("python/carnot/experiment_7087_v621_entrance_bank_sufficiency_audit.py"),
        Path("python/carnot/experiment_7093_v622_entrance_bank_sufficiency_audit.py"),
        Path("tests/python/test_experiment_7087_v621_entrance_bank_sufficiency_audit.py"),
        Path("tests/python/test_experiment_7093_v622_entrance_bank_sufficiency_audit.py"),
        Path("scripts/experiments/experiment_7087_v621_entrance_bank_sufficiency_audit.py"),
        Path("scripts/experiments/experiment_7093_v622_entrance_bank_sufficiency_audit.py"),
    )
    hashes = {str(path): _hash_if_file(REPO_ROOT / path) for path in paths}
    hashes[BANK_PATH.name] = _hash_if_file(bank_path)
    hashes[FIXTURE_PATH.name] = _hash_if_file(fixture_path)
    return hashes


def run_focused_tests() -> list[JsonDict]:  # pragma: no cover - run by the dated command.
    """Run the fixed coverage command and return hashes instead of large test logs."""

    command = focused_test_command()
    coverage_command = focused_coverage_report_command(new_code_only=False)
    new_coverage_command = focused_coverage_report_command(new_code_only=True)
    with tempfile.TemporaryDirectory(prefix="exp7093-coverage-") as temporary:
        environment = {**os.environ, "COVERAGE_FILE": str(Path(temporary) / ".coverage")}
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        coverage = subprocess.run(
            coverage_command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        new_coverage = subprocess.run(
            new_coverage_command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    returncode = next(
        (
            code
            for code in (completed.returncode, coverage.returncode, new_coverage.returncode)
            if code
        ),
        0,
    )
    return [
        {
            "command": command,
            "coverage_report_command": coverage_command,
            "new_code_coverage_report_command": new_coverage_command,
            "test_files": list(FOCUSED_TEST_FILES),
            "coverage_scopes": list(FOCUSED_COVERAGE_SCOPES),
            "coverage_fail_under": 100,
            "returncode": returncode,
            "test_returncode": completed.returncode,
            "coverage_returncode": coverage.returncode,
            "new_code_coverage_returncode": new_coverage.returncode,
            "stdout_hash": sha256_text(completed.stdout + coverage.stdout + new_coverage.stdout),
            "stderr_hash": sha256_text(completed.stderr + coverage.stderr + new_coverage.stderr),
            "passed": returncode == 0,
        }
    ]


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:  # pragma: no cover
    """Create the terminal artifact once so a later run cannot replace evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    bank_path: Path = BANK_PATH,
    fixture_path: Path = FIXTURE_PATH,
    audit_root: Path = AUDIT_ROOT,
) -> JsonDict:  # pragma: no cover - required dated fresh-process execution.
    """Check inputs, prove focused coverage, replay evidence, attack it, and write once."""

    started = time.perf_counter()
    preconditions = collect_preconditions(
        bank_path=bank_path,
        fixture_path=fixture_path,
        result_path=result_path,
        audit_root=audit_root,
    )
    source_hashes = _source_hashes(bank_path=bank_path, fixture_path=fixture_path)
    public_preconditions = _public_preconditions(preconditions)
    if preconditions.get("all_passed") is not True:
        artifact = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=public_preconditions,
            source_artifact_hashes=source_hashes,
            source_model_specs=dict(preconditions.get("bank") or {}).get("model_specs", []),
        )
        _write_new_json(result_path, artifact)
        return artifact
    focused_rows = run_focused_tests()
    if not _focused_receipt_valid(focused_rows):
        raise RuntimeError("focused_test_coverage_failed")
    bank = dict(preconditions["bank"])
    fixture = dict(preconditions["fixture"])
    with tempfile.TemporaryDirectory(prefix="fresh-replay-", dir=audit_root) as temporary:
        replay, process_rows = run_fresh_replay(bank_path, fixture_path, Path(temporary) / "worker")
    attacks = run_counterfactual_attacks(bank, fixture, replay["required_support_schema"])
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        bank=bank,
        fixture=fixture,
        preconditions=public_preconditions,
        source_artifact_hashes=source_hashes,
        replay=replay,
        fresh_process_rows=process_rows,
        focused_test_rows=focused_rows,
        counterfactual_swap_rows=attacks,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    _write_new_json(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command surface.
    """Run, validate, or serve the private Exp7093 worker command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--bank-path", type=Path, default=BANK_PATH)
    parser.add_argument("--fixture-path", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--fresh-worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.fresh_worker:
        if args.worker_output is None:
            parser.error("--fresh-worker requires --worker-output")
        return _fresh_worker(args.bank_path, args.fixture_path, args.worker_output)
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        bank_path=args.bank_path,
        fixture_path=args.fixture_path,
        audit_root=args.audit_root,
    )
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "entrance_support_audit_ready_score": artifact[
                    "entrance_support_audit_ready_score"
                ],
                "entrance_selector_headroom_ready_score": artifact[
                    "entrance_selector_headroom_ready_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
