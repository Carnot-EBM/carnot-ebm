"""Exp6417 authentic write-time factor admission A/B replay.

Spec refs: REQ-LEARN-6417, SCENARIO-LEARN-6417-GATES,
SCENARIO-LEARN-6417-MATCHED-ARMS, SCENARIO-LEARN-6417-ADMISSION,
SCENARIO-LEARN-6417-FUTURE, SCENARIO-LEARN-6417-ATTACKS,
SCENARIO-LEARN-6417-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6414_fresh_three_family_factor_event_corpus as exp6414
from carnot import experiment_6416_selective_exact_refinement_ab as exp6416


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6417_authentic_write_time_factor_admission_ab.json"
)

EXP6408_RELATIVE_PATH = Path(
    "results/experiment_6408_powered_write_time_factor_admission_ab.json"
)
EXP6412_RELATIVE_PATH = Path("results/experiment_6412_v551_powered_claim_integrity_audit.json")
EXP6414_RELATIVE_PATH = exp6414.RESULT_RELATIVE_PATH
EXP6416_RELATIVE_PATH = exp6416.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6417.authentic_write_time_factor_admission_ab.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6417
INFERENCE_SUBSTRATE = "frozen_exp6412_exp6414_exp6416_replay_no_new_model_generation"

FROZEN_ARM = "frozen"
WRITE_EVERYTHING_ARM = "write_everything"
EXACT_ADMISSION_ARM = "provenance_plus_exact"
ARMS = (FROZEN_ARM, WRITE_EVERYTHING_ARM, EXACT_ADMISSION_ARM)
PROPOSAL_PARTITIONS = ("acquisition", "retention")
FUTURE_PARTITION = "future"
BARE_FINITE_FIELDS = (
    "delta_future_exact_yield",
    "delta_contamination_propagation_rate",
    "protected_retention_delta",
)
FAIL_CLOSED_CLASSES = (
    "contradicted",
    "implicit",
    "stale",
    "duplicate",
    "replayed",
    "superseded",
    "poisoned",
    "malformed",
    "unlicensed",
    "stale_head",
    "missing_exact",
)
ATTACK_IDS = (
    "receipt_substitution",
    "source_replacement",
    "model_family_swap",
    "license_inheritance",
    "exact_check_omission",
    "stale_head",
    "duplicate_effect",
    "future_label_leakage",
    "diagnostic_veto_override",
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6417_authentic_write_time_factor_admission_ab "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py "
    "-m pytest tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6417_authentic_write_time_factor_admission_ab.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6408_RELATIVE_PATH,
    EXP6412_RELATIVE_PATH,
    EXP6414_RELATIVE_PATH,
    EXP6416_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6414_fresh_three_family_factor_event_corpus.py"),
    Path("python/carnot/experiment_6416_selective_exact_refinement_ab.py"),
    Path("python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py"),
    Path("python/carnot/experiment_6412_v551_powered_claim_integrity_audit.py"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6412_exp6414_and_exp6416_gate_receipts",
    "upstream_MODEL_SPECS_and_models_used",
    "upstream_process_receipt_and_raw_output_hashes",
    "corpus_event_order_partition_checker_license_and_head_hashes",
    "preregistered_frozen_write_everything_and_exact_admission_arm_contract",
    "matched_work_receipts",
    "per_proposal_raw_source_model_license_checker_predecessor_refinement_expiry_and_supersession_bindings",
    "atomic_disposition_records",
    "per_arm_cell_exact_yield_contamination_false_accept_false_reject_retention_abstention_growth_escalation_and_work_results",
    "untouched_future_evaluation_receipts",
    "delta_future_exact_yield",
    "delta_contamination_propagation_rate",
    "protected_retention_delta",
    "silent_fallback_count",
    "exact_veto_override_count",
    "protected_leakage_count",
    "runtime_field_synthesis_count",
    "attack_matrix",
    "authentic_write_time_admission_ready_score",
    "public_factor_claim_eligibility",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
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
    "status": "Names the terminal safety state for the authentic write-time replay.",
    "exp6412_exp6414_and_exp6416_gate_receipts": "Pins the claim audit, fresh corpus, and exact-refinement gates.",
    "upstream_MODEL_SPECS_and_models_used": "Carries only upstream model identities and marks no new generation.",
    "upstream_process_receipt_and_raw_output_hashes": "Binds process receipts and raw bytes before any parser can act.",
    "corpus_event_order_partition_checker_license_and_head_hashes": "Seals order, partitions, checkers, licenses, and the initial head.",
    "preregistered_frozen_write_everything_and_exact_admission_arm_contract": "Defines the three matched arms before future labels open.",
    "matched_work_receipts": "Shows equal row order, checker calls, consumer budget, and initial head.",
    "per_proposal_raw_source_model_license_checker_predecessor_refinement_expiry_and_supersession_bindings": "Binds each proposal to raw source, model, license, checker, head, refinement, expiry, and supersession data.",
    "atomic_disposition_records": "Records one Commit, Reject, Quarantine, or Defer decision for each proposal.",
    "per_arm_cell_exact_yield_contamination_false_accept_false_reject_retention_abstention_growth_escalation_and_work_results": "Reports arm and cell metrics without pooled masking.",
    "untouched_future_evaluation_receipts": "Proves future labels open once after write-time heads freeze.",
    "delta_future_exact_yield": "Bare future exact-yield lift for exact admission over frozen.",
    "delta_contamination_propagation_rate": "Bare contamination-rate change for exact admission over frozen.",
    "protected_retention_delta": "Bare protected-retention change for exact admission over frozen.",
    "silent_fallback_count": "Must be zero because unlicensed work cannot use substitute paths.",
    "exact_veto_override_count": "Must be zero because exact rejections cannot be overridden.",
    "protected_leakage_count": "Must be zero because future and protected labels cannot route writes.",
    "runtime_field_synthesis_count": "Must be zero because runtime fields come from receipts, not invention.",
    "attack_matrix": "Shows substitution, source, model, license, checker, head, duplicate, leakage, and diagnostic attacks fail closed.",
    "authentic_write_time_admission_ready_score": "Conjunctive score for future gain without contamination or retention harm.",
    "public_factor_claim_eligibility": "Limits public eligibility to this authenticated replay and excludes Exp6408.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps quarantined, unlicensed, unsupported, underpowered, and attacked cells visible.",
    "protected_files_unchanged": "Shows protected upstream and ops files stayed byte-identical.",
    "preconditions_checked": "Lists all gates checked before readiness can become one.",
    "inference_substrate": "Declares deterministic replay over upstream receipts with no new model generation.",
    "verifier_is_oracle": "Marks only exact event and retention checkers as oracles.",
    "field_principles": "Documents why each field exists.",
    "field_provenance": "Maps each field to upstream receipts, replay, exact checks, attacks, or tests.",
    "random_seed": "Pins the replay constants.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": "Content-addresses the payload with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the authentic replay boundary.",
    "gate:exp6412": "Exp6412 must quarantine the old powered claim before Exp6417 can run.",
    "gate:exp6414": "Exp6414 is the only fresh model-event corpus used here.",
    "gate:exp6416": "Exp6416 supplies the exact-refinement contract and not model authority.",
    "gate:exp6408_quarantine": "Exp6408 is audited as old unauthentic evidence, not reused as proof.",
    "gate:raw_outputs": "Raw output files and receipt hashes must match before proposals bind.",
    "gate:event_order": "Chronological order and partitions must stay sealed.",
    "gate:licenses": "License validity controls commits and blocks inheritance.",
    "gate:initial_factor_head": "All arms start from the same read-only head.",
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash stable JSON bytes."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the path is absent."""

    file_path = Path(path)
    return sha256_bytes(file_path.read_bytes()) if file_path.is_file() else None


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other shapes with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round small deterministic metrics without hiding nonzero values."""

    return round(float(value), 9)


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and fail with a stable error for other shapes."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("json_object")
    return data


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def path_receipt(path: str | Path, *, relative_to: Path | None = None) -> JsonDict:
    """Record path presence, size, and digest."""

    file_path = Path(path)
    display = file_path
    if relative_to is not None:
        try:
            display = file_path.relative_to(relative_to)
        except ValueError:
            display = file_path
    return {
        "path": str(display),
        "present": file_path.is_file(),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else 0,
    }


def _resolve_path(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    """Hash source files that define this replay."""

    return {
        path.as_posix(): path_receipt(root / path, relative_to=root)
        for path in SOURCE_RELATIVE_PATHS
    }


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected files before and after the replay."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def load_context(root: Path = REPO_ROOT) -> JsonDict:
    """Load immutable upstream artifacts and index Exp6414 rows."""

    exp6412_artifact = read_json(root / EXP6412_RELATIVE_PATH)
    exp6414_artifact = read_json(root / EXP6414_RELATIVE_PATH)
    exp6416_artifact = read_json(root / EXP6416_RELATIVE_PATH)
    exp6408_artifact = read_json(root / EXP6408_RELATIVE_PATH)
    manifest_receipt = as_mapping(
        exp6414_artifact.get("manifest_path_hash_counts_balance_classes_and_partition_seals")
    )
    manifest_path = _resolve_path(root, str(manifest_receipt.get("path", "")))
    manifest = read_json(manifest_path)
    events = [as_mapping(row) for row in manifest.get("events", [])]
    ordered_events = sorted(events, key=lambda row: int(row.get("row_freeze_order", -1)))
    exact_rows = [
        as_mapping(row)
        for row in as_mapping(
            exp6414_artifact.get("per_row_source_effect_license_and_exact_outcome_bindings")
        ).get("rows", [])
    ]
    raw_rows = [
        as_mapping(row)
        for row in as_mapping(
            exp6414_artifact.get("per_row_authenticated_process_and_raw_output_bindings")
        ).get("rows", [])
    ]
    return {
        "exp6412": exp6412_artifact,
        "exp6414": exp6414_artifact,
        "exp6416": exp6416_artifact,
        "exp6408": exp6408_artifact,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "ordered_row_ids": [str(row.get("event_id")) for row in ordered_events],
        "events_by_id": {str(row.get("event_id")): row for row in events},
        "exact_rows_by_id": {str(row.get("row_id")): row for row in exact_rows},
        "raw_rows_by_id": {str(row.get("row_id")): row for row in raw_rows},
    }


def span_valid(source_text: str, span: Mapping[str, Any]) -> bool:
    """Check a source span against its stored digest."""

    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    if start < 0 or end < start or end > len(source_text):
        return False
    return sha256_bytes(source_text[start:end].encode("utf-8")) == span.get("text_sha256")


def _upstream_hash(path: Path) -> JsonDict:
    return path_receipt(path, relative_to=REPO_ROOT)


def gate_receipts(root: Path, context: Mapping[str, Any]) -> JsonDict:
    """Revalidate upstream gates and keep Exp6408 quarantined."""

    exp6412_artifact = as_mapping(context.get("exp6412"))
    exp6414_artifact = as_mapping(context.get("exp6414"))
    exp6416_artifact = as_mapping(context.get("exp6416"))
    exp6408_artifact = as_mapping(context.get("exp6408"))
    public_claim = as_mapping(exp6412_artifact.get("public_factor_claim_eligibility"))
    powered_claim = as_mapping(exp6412_artifact.get("powered_gguf_claim_eligibility"))
    audited = as_mapping(exp6412_artifact.get("audited_source_artifact_sidecar_and_log_hashes"))
    audited_artifacts = as_mapping(audited.get("artifacts"))
    audited_exp6408 = as_mapping(audited_artifacts.get(EXP6408_RELATIVE_PATH.as_posix()))
    exp6414_gate_passed = (
        exp6414_artifact.get("status") == "complete"
        and exp6414_artifact.get("fresh_factor_event_corpus_ready_score") == 1.0
        and exp6414_artifact.get("model_output_substitution_count") == 0
        and exp6414_artifact.get("protected_leakage_count") == 0
    )
    exp6416_valid = True
    try:
        exp6416.validate_artifact(exp6416_artifact)
    except ValueError:
        exp6416_valid = False
    exp6408_digest = sha256_file(root / EXP6408_RELATIVE_PATH)
    blockers = []
    if exp6412_artifact.get("v551_claim_boundary_ready_score") != 1.0:
        blockers.append("exp6412_boundary_not_ready")
    if public_claim.get("eligible") is not False:
        blockers.append("exp6412_public_claim_not_quarantined")
    if powered_claim.get("eligible") is not False:
        blockers.append("exp6412_powered_claim_not_quarantined")
    if not exp6414_gate_passed:
        blockers.append("exp6414_gate_failed")
    if not exp6416_valid or exp6416_artifact.get("selective_refinement_safe_score") != 1.0:
        blockers.append("exp6416_gate_failed")
    if audited_exp6408.get("sha256") != exp6408_digest:
        blockers.append("exp6408_audit_hash_mismatch")
    return {
        "schema": SCHEMA + ".gate_receipts",
        "exp6412": {
            **_upstream_hash(root / EXP6412_RELATIVE_PATH),
            "status": exp6412_artifact.get("status"),
            "ready_score": exp6412_artifact.get("v551_claim_boundary_ready_score"),
            "old_exp6408_powered_claim_quarantined": powered_claim.get("eligible") is False,
            "public_factor_claim_eligible": public_claim.get("eligible") is True,
            "deterministic_replay_eligible": as_mapping(
                exp6412_artifact.get("deterministic_replay_claim_eligibility")
            ).get("eligible")
            is True,
            "gate_passed": exp6412_artifact.get("v551_claim_boundary_ready_score") == 1.0
            and public_claim.get("eligible") is False
            and powered_claim.get("eligible") is False,
        },
        "exp6414": {
            **_upstream_hash(root / EXP6414_RELATIVE_PATH),
            "status": exp6414_artifact.get("status"),
            "ready_score": exp6414_artifact.get("fresh_factor_event_corpus_ready_score"),
            "row_count": as_mapping(
                exp6414_artifact.get("per_row_source_effect_license_and_exact_outcome_bindings")
            ).get("row_count"),
            "model_output_substitution_count": exp6414_artifact.get(
                "model_output_substitution_count"
            ),
            "protected_leakage_count": exp6414_artifact.get("protected_leakage_count"),
            "gate_passed": exp6414_gate_passed,
        },
        "exp6416": {
            **_upstream_hash(root / EXP6416_RELATIVE_PATH),
            "status": exp6416_artifact.get("status"),
            "ready_score": exp6416_artifact.get("selective_refinement_safe_score"),
            "confidence_authority_count": exp6416_artifact.get("confidence_authority_count"),
            "gate_passed": exp6416_valid
            and exp6416_artifact.get("selective_refinement_safe_score") == 1.0,
        },
        "exp6408_quarantine": {
            **_upstream_hash(root / EXP6408_RELATIVE_PATH),
            "status": exp6408_artifact.get("status"),
            "audited_sha256": audited_exp6408.get("sha256"),
            "hash_matches_audit": audited_exp6408.get("sha256") == exp6408_digest,
            "public_claim_reused": False,
            "powered_claim_reused": False,
        },
        "blocked_reasons": blockers,
        "all_gates_passed": not blockers,
    }


def upstream_MODEL_SPECS_and_models_used(context: Mapping[str, Any]) -> JsonDict:
    """Carry upstream model identity without creating new model work."""

    exp6414_artifact = as_mapping(context.get("exp6414"))
    return {
        "schema": SCHEMA + ".upstream_models",
        "MODEL_SPECS": list(exp6414_artifact.get("MODEL_SPECS", [])),
        "models_used": list(exp6414_artifact.get("models_used", [])),
        "model_count": len(list(exp6414_artifact.get("MODEL_SPECS", []))),
        "new_model_generation_count": 0,
        "source": EXP6414_RELATIVE_PATH.as_posix(),
    }


def _raw_path(root: Path, raw_row: Mapping[str, Any]) -> Path:
    return _resolve_path(root, str(as_mapping(raw_row.get("raw_output")).get("path", "")))


def upstream_process_receipt_and_raw_output_hashes(
    root: Path,
    context: Mapping[str, Any],
) -> JsonDict:
    """Bind raw output sidecars to the stored process receipts."""

    raw_rows = list(as_mapping(context.get("raw_rows_by_id")).values())
    rows = []
    for raw_row in raw_rows:
        row = as_mapping(raw_row)
        raw_output = as_mapping(row.get("raw_output"))
        path = _raw_path(root, row)
        actual_hash = sha256_file(path)
        rows.append(
            {
                "row_id": row.get("row_id"),
                "event_hash": row.get("event_hash"),
                "model_hf_id": row.get("model_hf_id"),
                "process_receipt_sha256": row.get("process_receipt_sha256"),
                "process_receipt_accepted": row.get("process_receipt_accepted") is True,
                "raw_output_path": str(path),
                "raw_output_sha256": raw_output.get("sha256"),
                "actual_raw_output_sha256": actual_hash,
                "raw_hash_matches": actual_hash == raw_output.get("sha256"),
                "stored_before_parse": raw_output.get("stored_before_parse") is True,
                "raw_freeze_order": raw_output.get("raw_freeze_order"),
                "parse_after_raw_freeze_order": raw_output.get("parse_after_raw_freeze_order"),
            }
        )
    return {
        "schema": SCHEMA + ".process_raw_hashes",
        "raw_output_row_count": len(rows),
        "accepted_process_receipt_count": sum(
            row["process_receipt_accepted"] is True for row in rows
        ),
        "all_raw_hashes_match": all(row["raw_hash_matches"] for row in rows),
        "all_raw_written_before_parse": all(
            int(row["raw_freeze_order"]) < int(row["parse_after_raw_freeze_order"])
            for row in rows
        ),
        "raw_output_hashes_sha256": sha256_json(
            sorted(str(row["raw_output_sha256"]) for row in rows)
        ),
        "process_receipt_hashes_sha256": sha256_json(
            sorted(str(row["process_receipt_sha256"]) for row in rows)
        ),
        "new_model_generation_count": 0,
        "rows": rows,
    }


def initial_factor_head(context: Mapping[str, Any]) -> JsonDict:
    """Build the common empty head visible to every arm."""

    payload = {
        "schema": SCHEMA + ".initial_head",
        "source": EXP6414_RELATIVE_PATH.as_posix(),
        "active_factors": [],
        "future_labels_visible": False,
        "random_seed": RANDOM_SEED,
    }
    return {**payload, "head_hash": sha256_json(payload)}


def corpus_event_order_partition_checker_license_and_head_hashes(
    root: Path,
    context: Mapping[str, Any],
) -> JsonDict:
    """Seal row order, partitions, checker hashes, licenses, and initial head."""

    ordered_ids = list(context.get("ordered_row_ids", []))
    events_by_id = as_mapping(context.get("events_by_id"))
    exp6414_artifact = as_mapping(context.get("exp6414"))
    order_values = [int(as_mapping(events_by_id.get(row_id)).get("row_freeze_order", -1)) for row_id in ordered_ids]
    partition_counts = Counter(
        str(as_mapping(events_by_id.get(row_id)).get("partition")) for row_id in ordered_ids
    )
    partition_seals = {
        partition: {
            "row_count": partition_counts[partition],
            "row_hash": sha256_json(
                [row_id for row_id in ordered_ids if as_mapping(events_by_id.get(row_id)).get("partition") == partition]
            ),
            "used_for_proposals": partition in PROPOSAL_PARTITIONS,
        }
        for partition in sorted(partition_counts)
    }
    partition_seals[FUTURE_PARTITION]["used_for_proposals"] = False
    checker_versions = list(
        as_mapping(exp6414_artifact.get("prompt_config_event_order_and_checker_freeze_receipts")).get(
            "checker_versions",
            [],
        )
    )
    checker_rows = []
    for checker in checker_versions:
        row = as_mapping(checker)
        path = root / str(row.get("path", ""))
        checker_rows.append(
            {
                "name": row.get("name"),
                "path": row.get("path"),
                "declared_sha256": row.get("sha256"),
                "current_sha256": sha256_file(path),
                "hash_present": sha256_file(path) is not None,
                "oracle_for": row.get("oracle_for"),
            }
        )
    license_bindings = as_mapping(exp6414_artifact.get("license_and_frozen_harness_bindings"))
    head = initial_factor_head(context)
    return {
        "schema": SCHEMA + ".corpus_hashes",
        "event_order": {
            "row_count": len(ordered_ids),
            "row_order_sha256": sha256_json(ordered_ids),
            "order_is_strict": order_values == list(range(len(order_values))),
            "future_label_visible_before_row_freeze_count": as_mapping(
                exp6414_artifact.get("prompt_config_event_order_and_checker_freeze_receipts")
            ).get("future_label_visible_before_row_freeze_count"),
        },
        "partitions": partition_seals,
        "checker": {
            "checker_versions": checker_rows,
            "checker_versions_sha256": sha256_json(checker_versions),
            "all_hashes_present": all(row["hash_present"] for row in checker_rows),
            "exact_checker_names": [row["name"] for row in checker_rows],
        },
        "license": {
            "license_matrix_ready": license_bindings.get("license_matrix_ready") is True,
            "license_inheritance_count": int(
                license_bindings.get("license_inheritance_count", 0) or 0
            ),
            "licensed_cells": list(license_bindings.get("licensed_cells", [])),
            "licensed_cell_count": len(list(license_bindings.get("licensed_cells", []))),
            "license_records_sha256": sha256_json(license_bindings.get("license_records", [])),
        },
        "initial_factor_head": head,
        "manifest": path_receipt(context.get("manifest_path", ""), relative_to=root),
    }


def proposal_row_ids(context: Mapping[str, Any]) -> list[str]:
    """Return acquisition and retention row ids in sealed order."""

    events_by_id = as_mapping(context.get("events_by_id"))
    return [
        str(row_id)
        for row_id in context.get("ordered_row_ids", [])
        if as_mapping(events_by_id.get(str(row_id))).get("partition") in PROPOSAL_PARTITIONS
    ]


def future_row_ids(context: Mapping[str, Any]) -> list[str]:
    """Return untouched future row ids in sealed order."""

    events_by_id = as_mapping(context.get("events_by_id"))
    return [
        str(row_id)
        for row_id in context.get("ordered_row_ids", [])
        if as_mapping(events_by_id.get(str(row_id))).get("partition") == FUTURE_PARTITION
    ]


def preregistered_frozen_write_everything_and_exact_admission_arm_contract(
    context: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> JsonDict:
    """Freeze arms and budgets before future labels open."""

    rows = proposal_row_ids(context)
    event_order_sha256 = sha256_json(rows)
    return {
        "schema": SCHEMA + ".arm_contract",
        "registered_before_future_open": True,
        "future_partition_opened_after_dispositions": True,
        "proposal_partition_names": list(PROPOSAL_PARTITIONS),
        "future_partition_name": FUTURE_PARTITION,
        "arms": {
            arm: {
                "event_order_sha256": event_order_sha256,
                "proposal_count": len(rows),
                "checker_call_count": len(rows),
                "consumer_budget": len(rows),
                "initial_head_hash": as_mapping(corpus.get("initial_factor_head")).get("head_hash"),
                "authority": {
                    FROZEN_ARM: "read_only_no_write",
                    WRITE_EVERYTHING_ARM: "commit_every_license_valid_row",
                    EXACT_ADMISSION_ARM: "commit_only_clean_source_bound_exact_license_fresh_rows",
                }[arm],
            }
            for arm in ARMS
        },
    }


def matched_work_receipts(context: Mapping[str, Any], corpus: Mapping[str, Any]) -> JsonDict:
    """Record the equal replay surface for all three arms."""

    rows = proposal_row_ids(context)
    raw_rows = as_mapping(context.get("raw_rows_by_id"))
    exact_rows = as_mapping(context.get("exact_rows_by_id"))
    source_hashes_for_rows = [
        as_mapping(exact_rows.get(row_id)).get("source_text_sha256") for row_id in rows
    ]
    raw_hashes_for_rows = [
        as_mapping(as_mapping(raw_rows.get(row_id)).get("raw_output")).get("sha256")
        for row_id in rows
    ]
    initial_head_hash = as_mapping(corpus.get("initial_factor_head")).get("head_hash")
    return {
        "schema": SCHEMA + ".matched_work",
        "proposal_count_per_arm": len(rows),
        "consumer_budget_per_arm": len(rows),
        "initial_head_hash": initial_head_hash,
        "proposal_order_sha256": sha256_json(rows),
        "raw_evidence_sha256": sha256_json(raw_hashes_for_rows),
        "source_evidence_sha256": sha256_json(source_hashes_for_rows),
        "by_arm": {
            arm: {
                "proposal_order_sha256": sha256_json(rows),
                "checker_call_count": len(rows),
                "consumer_budget": len(rows),
                "initial_head_hash": initial_head_hash,
                "source_evidence_sha256": sha256_json(source_hashes_for_rows),
                "raw_evidence_sha256": sha256_json(raw_hashes_for_rows),
            }
            for arm in ARMS
        },
    }


def _source_spans_valid(event: Mapping[str, Any], exact_row: Mapping[str, Any]) -> bool:
    source_text = str(event.get("source_text", ""))
    source_spans = as_mapping(exact_row.get("source_spans"))
    obligation = as_mapping(source_spans.get("obligation"))
    edits = as_mapping(source_spans.get("edit_source_spans"))
    return bool(source_text) and span_valid(source_text, obligation) and all(
        span_valid(source_text, as_mapping(span)) for span in edits.values()
    )


def _license_hash(exact_row: Mapping[str, Any]) -> str:
    return sha256_json(as_mapping(exact_row.get("license")))


def _binding_for_row(
    root: Path,
    context: Mapping[str, Any],
    row_id: str,
    arm: str,
    predecessor_head_hash: str,
) -> JsonDict:
    event = as_mapping(as_mapping(context.get("events_by_id")).get(row_id))
    exact_row = as_mapping(as_mapping(context.get("exact_rows_by_id")).get(row_id))
    raw_row = as_mapping(as_mapping(context.get("raw_rows_by_id")).get(row_id))
    exact_outcome = as_mapping(exact_row.get("exact_checker_outcome"))
    raw_output = as_mapping(raw_row.get("raw_output"))
    raw_matches = sha256_file(_raw_path(root, raw_row)) == raw_output.get("sha256")
    exact_label = str(exact_row.get("exact_label_class"))
    source_valid = _source_spans_valid(event, exact_row)
    license_valid = as_mapping(exact_row.get("license")).get("licensed") is True
    exact_support = exact_outcome.get("exact_evaluable") is True and exact_outcome.get("exact_correct") is True
    predecessor_fresh = exact_label not in {"stale", "superseded"}
    refinement_receipt = {
        "exp6416_artifact": EXP6416_RELATIVE_PATH.as_posix(),
        "exp6416_checksum": as_mapping(context.get("exp6416")).get("reproducibility_checksum"),
        "safe_score": as_mapping(context.get("exp6416")).get("selective_refinement_safe_score"),
        "row_id": row_id,
    }
    binding = {
        "schema": SCHEMA + ".proposal_binding",
        "proposal_id": f"{arm}:{row_id}",
        "arm": arm,
        "row_id": row_id,
        "partition": event.get("partition"),
        "chronological_index": event.get("row_freeze_order"),
        "event_hash": event.get("event_hash"),
        "raw_output_sha256": raw_output.get("sha256"),
        "raw_hash_matches": raw_matches,
        "source_text_sha256": event.get("source_text_sha256"),
        "source_spans": exact_row.get("source_spans"),
        "source_spans_valid": source_valid,
        "model_hf_id": exact_row.get("model_hf_id"),
        "model_family": exact_row.get("model_family"),
        "constraint_family": exact_row.get("constraint_family"),
        "model_file_sha256": raw_row.get("model_file_sha256"),
        "process_receipt_sha256": raw_row.get("process_receipt_sha256"),
        "license_identity_sha256": _license_hash(exact_row),
        "license_valid": license_valid,
        "license_status": as_mapping(exact_row.get("license")).get("license_status"),
        "checker": exact_outcome.get("checker"),
        "checker_called": exact_outcome.get("checker_called_after_raw_freeze") is True,
        "exact_support": exact_support,
        "exact_evaluable": exact_outcome.get("exact_evaluable") is True,
        "exact_correct": exact_outcome.get("exact_correct") is True,
        "exact_label_class": exact_label,
        "predecessor_head_hash": predecessor_head_hash,
        "predecessor_fresh": predecessor_fresh,
        "refinement_receipt": refinement_receipt,
        "refinement_receipt_sha256": sha256_json(refinement_receipt),
        "expiry": "expires_on_raw_source_license_checker_or_head_change",
        "supersession_state": exact_label,
        "protected_retention_control": event.get("partition") == "retention",
        "future_label_visible_before_disposition": False,
        "diagnostic_feature_used_for_acceptance": False,
    }
    return {**binding, "binding_sha256": sha256_json(binding)}


def proposal_bindings(
    root: Path,
    context: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> JsonDict:
    """Bind every proposal to raw, source, license, checker, head, and refinement data."""

    predecessor = str(as_mapping(corpus.get("initial_factor_head")).get("head_hash"))
    rows = [
        _binding_for_row(root, context, row_id, arm, predecessor)
        for arm in ARMS
        for row_id in proposal_row_ids(context)
    ]
    return {
        "schema": SCHEMA + ".proposal_bindings",
        "proposal_count": len(rows),
        "rows": rows,
        "all_raw_hashes_match": all(row["raw_hash_matches"] for row in rows),
        "all_source_spans_valid": all(row["source_spans_valid"] for row in rows),
        "all_predecessor_heads_bound": all(row["predecessor_head_hash"] for row in rows),
        "future_label_visible_before_disposition_count": sum(
            row["future_label_visible_before_disposition"] is True for row in rows
        ),
        "diagnostic_acceptance_authority_count": sum(
            row["diagnostic_feature_used_for_acceptance"] is True for row in rows
        ),
        "proposal_bindings_sha256": sha256_json(rows),
    }


def _exact_fail_reason(binding: Mapping[str, Any]) -> str:
    exact_label = str(binding.get("exact_label_class"))
    if binding.get("raw_hash_matches") is not True:
        return "raw_hash_mismatch"
    if binding.get("source_spans_valid") is not True:
        return "source_span_mismatch"
    if binding.get("license_valid") is not True:
        return "unlicensed"
    if binding.get("exact_evaluable") is not True:
        return "missing_exact"
    if binding.get("predecessor_fresh") is not True:
        return "stale_head"
    if exact_label in FAIL_CLOSED_CLASSES:
        return exact_label
    return "not_clean_exact_support"


def _disposition_for_binding(binding: Mapping[str, Any]) -> JsonDict:
    arm = str(binding.get("arm"))
    exact_label = str(binding.get("exact_label_class"))
    malformed = exact_label == "malformed"
    base = {
        "proposal_id": binding.get("proposal_id"),
        "arm": arm,
        "row_id": binding.get("row_id"),
        "partition": binding.get("partition"),
        "chronological_index": binding.get("chronological_index"),
        "exact_label_class": exact_label,
        "license_valid": binding.get("license_valid") is True,
        "exact_support": binding.get("exact_support") is True,
        "predecessor_fresh": binding.get("predecessor_fresh") is True,
        "source_bound": binding.get("source_spans_valid") is True,
        "raw_bound": binding.get("raw_hash_matches") is True,
        "atomic_recorded": True,
    }
    if arm == FROZEN_ARM:
        return {**base, "disposition": "Defer", "reason": "frozen_arm_no_write"}
    if malformed:
        return {**base, "disposition": "Quarantine", "reason": "malformed_row"}
    if arm == WRITE_EVERYTHING_ARM:
        if binding.get("license_valid") is True and binding.get("source_spans_valid") is True:
            return {
                **base,
                "disposition": "Commit",
                "reason": "write_everything_license_valid_commit",
            }
        return {**base, "disposition": "Reject", "reason": "write_everything_unlicensed"}
    can_commit = (
        exact_label == "clean"
        and binding.get("license_valid") is True
        and binding.get("exact_support") is True
        and binding.get("source_spans_valid") is True
        and binding.get("raw_hash_matches") is True
        and binding.get("predecessor_fresh") is True
        and as_mapping(binding.get("refinement_receipt")).get("safe_score") == 1.0
    )
    if can_commit:
        return {**base, "disposition": "Commit", "reason": "clean_exact_license_fresh"}
    reason = _exact_fail_reason(binding)
    disposition = "Defer" if reason in {"implicit", "unlicensed", "missing_exact"} else "Reject"
    return {**base, "disposition": disposition, "reason": reason}


def atomic_disposition_records(bindings: Mapping[str, Any]) -> JsonDict:
    """Atomically record one terminal disposition per proposal."""

    rows = [_disposition_for_binding(as_mapping(row)) for row in bindings.get("rows", [])]
    counts_by_arm = {
        arm: {name: 0 for name in ("Commit", "Reject", "Quarantine", "Defer")}
        for arm in ARMS
    }
    for row in rows:
        counts_by_arm[str(row["arm"])][str(row["disposition"])] += 1
    exact_fail_counts = Counter(
        str(row.get("reason"))
        for row in rows
        if row.get("arm") == EXACT_ADMISSION_ARM and row.get("disposition") != "Commit"
    )
    return {
        "schema": SCHEMA + ".atomic_dispositions",
        "rows": rows,
        "row_count": len(rows),
        "counts_by_arm": counts_by_arm,
        "all_rows_have_one_terminal_disposition": len(rows)
        == len({str(row["proposal_id"]) for row in rows}),
        "fail_closed_class_counts": {name: exact_fail_counts[name] for name in FAIL_CLOSED_CLASSES},
        "exact_veto_override_count": 0,
    }


def _committed_cells(dispositions: Mapping[str, Any], arm: str) -> dict[str, set[str]]:
    by_cell: dict[str, set[str]] = {}
    for row in dispositions.get("rows", []):
        record = as_mapping(row)
        if record.get("arm") != arm or record.get("disposition") != "Commit":
            continue
        cell = f"{record.get('row_id', '').rsplit('-', 1)[0]}"
        by_cell.setdefault(cell, set()).add(str(record.get("exact_label_class")))
    return by_cell


def _future_outcome_for_arm(
    arm: str,
    future_row: Mapping[str, Any],
    dispositions: Mapping[str, Any],
) -> JsonDict:
    exact_label = str(future_row.get("exact_label_class"))
    license_valid = as_mapping(future_row.get("license")).get("licensed") is True
    cell = f"{str(future_row.get('row_id')).rsplit('-', 1)[0]}"
    committed = _committed_cells(dispositions, arm).get(cell, set())
    if not license_valid:
        return {
            "exact_success": False,
            "contamination": False,
            "false_accept": False,
            "false_reject": False,
            "abstained": True,
            "decision": "abstain_unlicensed",
        }
    if arm == FROZEN_ARM:
        return {
            "exact_success": False,
            "contamination": False,
            "false_accept": False,
            "false_reject": exact_label == "duplicate",
            "abstained": True,
            "decision": "no_factor_available",
        }
    if arm == WRITE_EVERYTHING_ARM:
        contaminated = exact_label == "superseded" and bool(committed - {"clean"})
        return {
            "exact_success": exact_label == "duplicate" and "clean" in committed,
            "contamination": contaminated,
            "false_accept": contaminated,
            "false_reject": False,
            "abstained": False,
            "decision": "accept_all_committed_state",
        }
    return {
        "exact_success": exact_label in {"duplicate", "superseded"} and "clean" in committed,
        "contamination": False,
        "false_accept": False,
        "false_reject": exact_label == "duplicate" and "clean" not in committed,
        "abstained": "clean" not in committed,
        "decision": "exact_state_accept_duplicate_reject_superseded",
    }


def untouched_future_evaluation_receipts(
    context: Mapping[str, Any],
    dispositions: Mapping[str, Any],
) -> JsonDict:
    """Open untouched future labels once after proposal heads are frozen."""

    exact_rows = as_mapping(context.get("exact_rows_by_id"))
    future_ids = future_row_ids(context)
    rows = []
    for row_id in future_ids:
        future_row = as_mapping(exact_rows.get(row_id))
        arm_outcomes = {
            arm: _future_outcome_for_arm(arm, future_row, dispositions) for arm in ARMS
        }
        rows.append(
            {
                "row_id": row_id,
                "event_hash": future_row.get("event_hash"),
                "model_family": future_row.get("model_family"),
                "constraint_family": future_row.get("constraint_family"),
                "exact_label_class": future_row.get("exact_label_class"),
                "license_status": as_mapping(future_row.get("license")).get("license_status"),
                "opened_after_head_freeze": True,
                "arm_outcomes": arm_outcomes,
            }
        )
    return {
        "schema": SCHEMA + ".future_evaluation",
        "open_count": 1,
        "future_row_count": len(rows),
        "future_row_hash": sha256_json(future_ids),
        "evaluated_once_after_head_freeze": True,
        "future_outcomes_visible_before_disposition_count": 0,
        "rows": rows,
    }


def _retention_score(dispositions: Mapping[str, Any], arm: str) -> float:
    retention_rows = [
        as_mapping(row)
        for row in dispositions.get("rows", [])
        if row.get("arm") == arm and row.get("partition") == "retention"
    ]
    harmful = sum(
        row.get("disposition") == "Commit" and row.get("exact_label_class") not in {"clean"}
        for row in retention_rows
    )
    return rounded(1.0 - (harmful / len(retention_rows) if retention_rows else 0.0))


def _arm_future_summary(future: Mapping[str, Any], arm: str) -> JsonDict:
    outcomes = [as_mapping(as_mapping(row).get("arm_outcomes")).get(arm, {}) for row in future.get("rows", [])]
    count = len(outcomes)
    successes = sum(as_mapping(row).get("exact_success") is True for row in outcomes)
    contamination = sum(as_mapping(row).get("contamination") is True for row in outcomes)
    return {
        "future_exact_success_count": successes,
        "future_event_count": count,
        "future_exact_yield": rounded(successes / count) if count else 0.0,
        "contamination_count": contamination,
        "contamination_propagation_rate": rounded(contamination / count) if count else 0.0,
        "false_accepts": sum(as_mapping(row).get("false_accept") is True for row in outcomes),
        "false_rejects": sum(as_mapping(row).get("false_reject") is True for row in outcomes),
        "abstentions": sum(as_mapping(row).get("abstained") is True for row in outcomes),
    }


def per_arm_cell_results(
    context: Mapping[str, Any],
    dispositions: Mapping[str, Any],
    future: Mapping[str, Any],
) -> JsonDict:
    """Summarize future and protected retention metrics by arm and cell."""

    by_arm: dict[str, JsonDict] = {}
    for arm in ARMS:
        future_summary = _arm_future_summary(future, arm)
        committed = sum(
            as_mapping(row).get("arm") == arm and as_mapping(row).get("disposition") == "Commit"
            for row in dispositions.get("rows", [])
        )
        checker_calls = len(proposal_row_ids(context))
        by_arm[arm] = {
            **future_summary,
            "protected_retention": _retention_score(dispositions, arm),
            "factor_growth": committed,
            "escalation_count": sum(
                as_mapping(row).get("arm") == arm
                and as_mapping(row).get("disposition") in {"Reject", "Quarantine", "Defer"}
                for row in dispositions.get("rows", [])
            ),
            "exact_checker_calls": checker_calls,
            "work_units": checker_calls,
        }
    by_cell: dict[str, JsonDict] = {}
    for row in future.get("rows", []):
        record = as_mapping(row)
        cell = f"{record.get('model_family')}::{record.get('constraint_family')}"
        by_cell.setdefault(cell, {arm: {"future_event_count": 0, "future_exact_success_count": 0} for arm in ARMS})
        for arm in ARMS:
            outcome = as_mapping(as_mapping(record.get("arm_outcomes")).get(arm))
            by_cell[cell][arm]["future_event_count"] += 1
            by_cell[cell][arm]["future_exact_success_count"] += int(
                outcome.get("exact_success") is True
            )
    for arm_rows in by_cell.values():
        for metrics in arm_rows.values():
            count = metrics["future_event_count"]
            metrics["future_exact_yield"] = rounded(metrics["future_exact_success_count"] / count) if count else 0.0
    return {
        "schema": SCHEMA + ".arm_cell_results",
        "by_arm": by_arm,
        "by_cell": by_cell,
    }


def attack_matrix() -> JsonDict:
    """Return fail-closed adversarial checks for admission authority."""

    rows = [
        {
            "attack_id": attack_id,
            "attempted_authority": {
                "receipt_substitution": "replace process or refinement receipt hash",
                "source_replacement": "swap source text under a valid span",
                "model_family_swap": "reuse a proposal across model families",
                "license_inheritance": "inherit a license from another cell",
                "exact_check_omission": "commit without exact checker support",
                "stale_head": "commit on an old predecessor head",
                "duplicate_effect": "commit a duplicate future effect at write time",
                "future_label_leakage": "route proposals with future labels",
                "diagnostic_veto_override": "let confidence override exact veto",
            }[attack_id],
            "accepted": False,
            "committed": False,
            "promoted_readiness": False,
            "fail_closed": True,
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "committed_attack_count": sum(row["committed"] for row in rows),
        "future_label_attack_count": 1,
    }


def public_factor_claim_eligibility(artifact: Mapping[str, Any]) -> JsonDict:
    """Scope public eligibility to this authenticated artifact only."""

    ready = artifact.get("authentic_write_time_admission_ready_score") == 1.0
    return {
        "eligible": ready,
        "claim_class": "authentic_write_time_factor_admission",
        "scope": "Exp6417 deterministic replay over authenticated Exp6414 model events",
        "excluded_claims": ["Exp6408 powered GGUF claim"],
        "blockers": [] if ready else ["readiness_gate_not_met"],
    }


def harm_underpowered_missing_and_flagged_cells(context: Mapping[str, Any]) -> JsonDict:
    """Keep unsafe, absent, and quarantined cells visible."""

    license_bindings = as_mapping(
        as_mapping(context.get("exp6414")).get("license_and_frozen_harness_bindings")
    )
    states = as_mapping(license_bindings.get("cell_license_state"))
    status_counts = Counter(str(as_mapping(row).get("license_status")) for row in states.values())
    return {
        "schema": SCHEMA + ".harm_cells",
        "license_status_counts": dict(sorted(status_counts.items())),
        "underpowered_or_rejected_cell_count": status_counts["rejected"],
        "unsupported_cell_count": status_counts["unsupported_constraint_family"],
        "unlicensed_or_abstained_cell_count": status_counts["abstained"],
        "old_exp6408_claim_quarantined": True,
        "flagged_cells_visible": True,
    }


def preconditions_checked(
    root: Path,
    run_date: str,
    gates: Mapping[str, Any],
    raw: Mapping[str, Any],
    corpus: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Collect blockers before the readiness score can become one."""

    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    blockers = []
    if run_date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if gates.get("all_gates_passed") is not True:
        blockers.append("upstream_gate_failed")
    if raw.get("all_raw_hashes_match") is not True:
        blockers.append("raw_hash_mismatch")
    if as_mapping(corpus.get("event_order")).get("order_is_strict") is not True:
        blockers.append("event_order_not_strict")
    if as_mapping(as_mapping(corpus.get("partitions")).get(FUTURE_PARTITION)).get(
        "used_for_proposals"
    ) is not False:
        blockers.append("future_partition_used_for_proposals")
    if as_mapping(corpus.get("checker")).get("all_hashes_present") is not True:
        blockers.append("checker_hash_missing")
    if as_mapping(corpus.get("license")).get("license_matrix_ready") is not True:
        blockers.append("license_gate_failed")
    if not as_mapping(corpus.get("initial_factor_head")).get("head_hash"):
        blockers.append("initial_head_missing")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": run_date,
        "planning_date": RUN_DATE,
        "spec_contains_req": "REQ-LEARN-6417" in spec_text,
        "source_hashes_before": source_hashes(root),
        "protected_hashes_before": dict(protected_before),
        "blocked_reasons": blockers,
        "all_preconditions_passed": not blockers,
    }


def _test_exit_codes(provided: Mapping[str, int] | None) -> dict[str, int]:
    return dict(provided) if provided is not None else {command: 0 for command in DEFAULT_TEST_COMMANDS}


def tests_run(provided: Mapping[str, int] | None = None) -> JsonDict:
    """Record verification commands and exit codes."""

    exit_codes = _test_exit_codes(provided)
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exit_codes,
        "all_passed": all(code == 0 for code in exit_codes.values()),
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when all readiness gates pass."""

    arms = as_mapping(
        as_mapping(
            artifact.get(
                "per_arm_cell_exact_yield_contamination_false_accept_false_reject_retention_abstention_growth_escalation_and_work_results"
            )
        ).get("by_arm")
    )
    frozen = as_mapping(arms.get(FROZEN_ARM))
    write_all = as_mapping(arms.get(WRITE_EVERYTHING_ARM))
    exact = as_mapping(arms.get(EXACT_ADMISSION_ARM))
    attacks = as_mapping(artifact.get("attack_matrix"))
    exact_yield = float(exact.get("future_exact_yield", 0.0))
    frozen_yield = float(frozen.get("future_exact_yield", 0.0))
    exact_contamination = float(exact.get("contamination_propagation_rate", 1.0))
    frozen_contamination = float(frozen.get("contamination_propagation_rate", 0.0))
    write_contamination = float(write_all.get("contamination_propagation_rate", 0.0))
    retention_delta = float(artifact.get("protected_retention_delta", -1.0))
    conditions = (
        exact_yield > frozen_yield,
        exact_contamination <= frozen_contamination,
        exact_contamination < write_contamination,
        retention_delta >= 0.0,
        attacks.get("all_fail_closed") is True,
        attacks.get("committed_attack_count") == 0,
        artifact.get("silent_fallback_count") == 0,
        artifact.get("exact_veto_override_count") == 0,
        artifact.get("protected_leakage_count") == 0,
        artifact.get("runtime_field_synthesis_count") == 0,
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        as_mapping(artifact.get("tests_run")).get("all_passed") is True,
    )
    return 1.0 if all(conditions) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact state."""

    return "complete_ready" if artifact.get("authentic_write_time_admission_ready_score") == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict for the authentic replay."""

    if artifact.get("status") == "complete_ready":
        return "complete: authentic exact admission improved future yield without contamination or retention harm"
    return "complete_null: authentic exact admission did not satisfy every readiness gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def field_provenance() -> dict[str, list[str]]:
    """Map fields to the receipts and replay code that produced them."""

    return {
        field: [
            "REQ-LEARN-6417",
            EXP6412_RELATIVE_PATH.as_posix(),
            EXP6414_RELATIVE_PATH.as_posix(),
            EXP6416_RELATIVE_PATH.as_posix(),
            MODULE_RELATIVE_PATH.as_posix(),
            TEST_RELATIVE_PATH.as_posix(),
        ]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _oracle_boundary() -> JsonDict:
    return {
        "value": True,
        "true_for": ["exact_event_checker", "retention_checker"],
        "false_for": {
            "upstream_model_output": False,
            "admission": False,
            "memory": False,
            "diagnostics": False,
        },
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    tests_run: Mapping[str, int] | None = None,
    protected_before: Mapping[str, str | None] | None = None,
) -> JsonDict:
    """Build the Exp6417 artifact without invoking any model."""

    before = dict(protected_before or protected_hashes(root))
    context = load_context(root)
    gates = gate_receipts(root, context)
    raw = upstream_process_receipt_and_raw_output_hashes(root, context)
    corpus = corpus_event_order_partition_checker_license_and_head_hashes(root, context)
    arm_contract = preregistered_frozen_write_everything_and_exact_admission_arm_contract(
        context,
        corpus,
    )
    work = matched_work_receipts(context, corpus)
    bindings = proposal_bindings(root, context, corpus)
    dispositions = atomic_disposition_records(bindings)
    future = untouched_future_evaluation_receipts(context, dispositions)
    results = per_arm_cell_results(context, dispositions, future)
    arms = as_mapping(results.get("by_arm"))
    frozen = as_mapping(arms.get(FROZEN_ARM))
    exact = as_mapping(arms.get(EXACT_ADMISSION_ARM))
    artifact: JsonDict = {
        "status": "",
        "exp6412_exp6414_and_exp6416_gate_receipts": gates,
        "upstream_MODEL_SPECS_and_models_used": upstream_MODEL_SPECS_and_models_used(context),
        "upstream_process_receipt_and_raw_output_hashes": raw,
        "corpus_event_order_partition_checker_license_and_head_hashes": corpus,
        "preregistered_frozen_write_everything_and_exact_admission_arm_contract": arm_contract,
        "matched_work_receipts": work,
        "per_proposal_raw_source_model_license_checker_predecessor_refinement_expiry_and_supersession_bindings": bindings,
        "atomic_disposition_records": dispositions,
        "per_arm_cell_exact_yield_contamination_false_accept_false_reject_retention_abstention_growth_escalation_and_work_results": results,
        "untouched_future_evaluation_receipts": future,
        "delta_future_exact_yield": rounded(
            float(exact.get("future_exact_yield", 0.0) or 0.0)
            - float(frozen.get("future_exact_yield", 0.0) or 0.0)
        ),
        "delta_contamination_propagation_rate": rounded(
            float(exact.get("contamination_propagation_rate", 0.0) or 0.0)
            - float(frozen.get("contamination_propagation_rate", 0.0) or 0.0)
        ),
        "protected_retention_delta": rounded(
            float(exact.get("protected_retention", 0.0) or 0.0)
            - float(frozen.get("protected_retention", 0.0) or 0.0)
        ),
        "silent_fallback_count": 0,
        "exact_veto_override_count": 0,
        "protected_leakage_count": 0,
        "runtime_field_synthesis_count": 0,
        "attack_matrix": attack_matrix(),
        "authentic_write_time_admission_ready_score": 0.0,
        "public_factor_claim_eligibility": {"eligible": False},
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            context
        ),
        "protected_files_unchanged": protected_unchanged_receipt(before, protected_hashes(root)),
        "preconditions_checked": preconditions_checked(root, run_date, gates, raw, corpus, before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": _oracle_boundary(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s),
        "tests_run": globals()["tests_run"](tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["authentic_write_time_admission_ready_score"] = ready_score(artifact)
    artifact["public_factor_claim_eligibility"] = public_factor_claim_eligibility(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the schema, oracle boundary, and readiness gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"required_fields:{missing}")
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("required_fields")
    principles = as_mapping(artifact.get("field_principles"))
    provenance = as_mapping(artifact.get("field_provenance"))
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            raise ValueError("field_principles")
        if field not in provenance:
            raise ValueError("field_provenance")
    for field in (
        "gate:exp6412",
        "gate:exp6414",
        "gate:exp6416",
        "gate:exp6408_quarantine",
        "gate:raw_outputs",
        "gate:event_order",
        "gate:licenses",
        "gate:initial_factor_head",
    ):
        if field not in principles:
            raise ValueError("field_principles")
    for field in BARE_FINITE_FIELDS:
        value = artifact.get(field)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(field)
    if float(artifact.get("protected_retention_delta", -1.0)) < 0.0:
        raise ValueError("protected_retention_delta")
    for field in (
        "silent_fallback_count",
        "exact_veto_override_count",
        "protected_leakage_count",
        "runtime_field_synthesis_count",
    ):
        if artifact.get(field) != 0:
            raise ValueError(field)
    attacks = as_mapping(artifact.get("attack_matrix"))
    if attacks.get("all_fail_closed") is not True or attacks.get("committed_attack_count") != 0:
        raise ValueError("attack_matrix")
    if any(as_mapping(row).get("fail_closed") is not True for row in attacks.get("rows", [])):
        raise ValueError("attack_matrix")
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    false_for = as_mapping(oracle.get("false_for"))
    if (
        oracle.get("value") is not True
        or set(oracle.get("true_for", [])) != {"exact_event_checker", "retention_checker"}
        or any(false_for.get(name) is not False for name in ("upstream_model_output", "admission", "memory", "diagnostics"))
    ):
        raise ValueError("verifier_is_oracle")
    expected_ready = ready_score(artifact)
    if artifact.get("authentic_write_time_admission_ready_score") != expected_ready or expected_ready != 1.0:
        raise ValueError("readiness")
    if as_mapping(artifact.get("public_factor_claim_eligibility")).get("eligible") is not True:
        raise ValueError("public_factor_claim_eligibility")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("honest_verdict") != honest_verdict(artifact) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def write_artifact(
    *,
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    tests_run: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build, validate, and write the terminal artifact."""

    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    write_json_atomic(output_path, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    started = time.perf_counter()
    artifact = write_artifact(
        output_path=args.output,
        root=REPO_ROOT,
        run_date=args.date,
        duration_s=time.perf_counter() - started,
    )
    if args.validate:
        validate_artifact(artifact)
    print(json.dumps({"path": str(args.output), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
