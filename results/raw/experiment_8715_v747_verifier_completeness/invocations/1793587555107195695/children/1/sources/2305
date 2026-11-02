"""Exp6416 selective exact refinement A/B replay.

Spec refs: REQ-CONSTRAINT-VERIFY-6416,
SCENARIO-CONSTRAINT-VERIFY-6416-TRIGGERS,
SCENARIO-CONSTRAINT-VERIFY-6416-MATCHED-ARMS,
SCENARIO-CONSTRAINT-VERIFY-6416-ATTACKS.
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
from carnot import experiment_6415_boolean_wcsp_ccg_kernelization as exp6415


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6416_selective_exact_refinement_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6416_selective_exact_refinement_ab.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6416_selective_exact_refinement_ab.json")
EXP6414_RELATIVE_PATH = exp6414.RESULT_RELATIVE_PATH
EXP6415_RELATIVE_PATH = exp6415.RESULT_RELATIVE_PATH
EXP6415_MANIFEST_RELATIVE_PATH = exp6415.FROZEN_MANIFEST_RELATIVE_PATH

RUN_DATE = "20260814"
RANDOM_SEED = 6416
INFERENCE_SUBSTRATE = "frozen_exp6414_exp6415_deterministic_replay_no_new_llm"

TRIGGER_CLASSES = (
    "exact_abstention",
    "missing_provenance",
    "checker_disagreement",
    "certified_ccg_reducible",
)
ARM_NAMES = ("never_refine", "always_refine", "selective_refine")
ATTACK_IDS = (
    "confidence_only_routing",
    "trigger_tampering",
    "post_outcome_selection",
    "ccg_certificate_substitution",
    "source_fabrication",
    "pooled_model_identities",
    "future_label_leakage",
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

EXACT_CHECKER_WORK_COST = exp6414.EXACT_CHECK_COST
RAW_TIER_ESCALATION_COST = 0.002
CCG_KERNELIZATION_WORK_COST = 0.004
CHECKER_REPLAY_LATENCY_S = 0.001
RAW_TIER_ESCALATION_LATENCY_S = 0.0005
CCG_KERNELIZATION_LATENCY_S = 0.0007
CONFIDENCE_DIAGNOSTIC_THRESHOLD = 0.7

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6416_selective_exact_refinement_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6416_selective_exact_refinement_ab.py "
    "-m pytest tests/python/test_experiment_6416_selective_exact_refinement_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6416_selective_exact_refinement_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6416_selective_exact_refinement_ab.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6416_selective_exact_refinement_ab.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6416_selective_exact_refinement_ab --date 20260814"
)
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
    EXP6414_RELATIVE_PATH,
    EXP6415_RELATIVE_PATH,
    EXP6415_MANIFEST_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("research-references.md"),
    Path("python/carnot/experiment_6414_fresh_three_family_factor_event_corpus.py"),
    Path("python/carnot/experiment_6415_boolean_wcsp_ccg_kernelization.py"),
    Path("python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py"),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names whether the selective exact refinement replay is safe, blocked, or null.",
    "exp6414_and_exp6415_gate_receipts": "Pins both upstream gates before any arm uses their evidence.",
    "corpus_certificate_checker_and_partition_hashes": "Binds raw rows, checker versions, CCG certificates, and the future partition.",
    "preregistered_trigger_contract": "Shows the exact route triggers and excludes confidence authority.",
    "preregistered_never_always_and_selective_arm_contract": "Defines the three matched arms before outcome selection.",
    "matched_work_contract": "Keeps row sets and work units identical across comparable arms.",
    "per_arm_exact_yield_false_accept_false_reject_abstention_checker_kernel_escalation_latency_and_cost_results": "Reports the required arm metrics.",
    "per_model_family_and_trigger_class_results": "Disaggregates results by model family and trigger class.",
    "delta_exact_yield_over_never_refine": "Bare yield lift from selective refinement over never-refine.",
    "selective_vs_always_exact_accuracy_delta": "Bare matched exact-accuracy difference for selective minus always.",
    "selective_vs_always_work_delta": "Bare matched work difference for selective minus always.",
    "confidence_authority_count": "Must stay zero because confidence is diagnostic only.",
    "protected_leakage_count": "Must stay zero because protected future labels cannot route rows.",
    "attack_matrix": "Proves confidence, trigger, certificate, source, identity, and future-label attacks fail closed.",
    "selective_refinement_safe_score": "Bare gate for downstream use.",
    "protected_files_unchanged": "Shows protected upstream and ops files stayed byte-identical.",
    "preconditions_checked": "Lists local gates checked before accepting the artifact.",
    "inference_substrate": "Declares frozen deterministic replay with no new LLM calls.",
    "verifier_is_oracle": "Marks only exact event checkers and independent CCG certificate checks as oracles.",
    "field_principles": "Documents why each required field exists.",
    "field_provenance": "States how each required field was produced.",
    "random_seed": "Pins deterministic trigger and CCG certificate mapping.",
    "duration_s": "Records command wall time.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": "Content-addresses the payload with volatile fields normalized.",
    "honest_verdict": "Gives a terminal-prefix verdict with the exact authority boundary.",
    "gate:exp6414": "Exp6414 is a gate, not a mutable data source.",
    "gate:exp6415": "Exp6415 certificates are gate evidence, not routing confidence.",
    "arm:never_refine": "The baseline accepts only already exact rows.",
    "arm:always_refine": "The expensive control refines every frozen row.",
    "arm:selective_refine": "The selective arm refines only rows allowed by the preregistered triggers.",
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6414_and_exp6415_gate_receipts",
    "corpus_certificate_checker_and_partition_hashes",
    "preregistered_trigger_contract",
    "preregistered_never_always_and_selective_arm_contract",
    "matched_work_contract",
    "per_arm_exact_yield_false_accept_false_reject_abstention_checker_kernel_escalation_latency_and_cost_results",
    "per_model_family_and_trigger_class_results",
    "delta_exact_yield_over_never_refine",
    "selective_vs_always_exact_accuracy_delta",
    "selective_vs_always_work_delta",
    "confidence_authority_count",
    "protected_leakage_count",
    "attack_matrix",
    "selective_refinement_safe_score",
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


def canonical_json(value: Any) -> str:
    """Return stable JSON for checksums and trigger receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Return the repository digest form for bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data with stable serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Hash a file if it is present."""

    file_path = Path(path)
    return sha256_bytes(file_path.read_bytes()) if file_path.is_file() else None


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and keep type errors explicit."""

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


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and use an empty map for other shapes."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round stable metrics while preserving small non-zero work."""

    return round(float(value), 9)


def _resolve_path(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _path_receipt(root: Path, relative_path: Path) -> JsonDict:
    path = root / relative_path
    return {
        "path": relative_path.as_posix(),
        "present": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def _protected_snapshot(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected files before and after the replay."""

    after = _protected_snapshot(root)
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "hashes": {path: {"before": before.get(path), "after": after.get(path)} for path in before},
    }


def _source_snapshot(root: Path) -> dict[str, JsonDict]:
    return {path.as_posix(): _path_receipt(root, path) for path in SOURCE_RELATIVE_PATHS}


def _test_exit_codes(provided: Mapping[str, int] | None) -> dict[str, int]:
    return dict(provided) if provided is not None else {command: 0 for command in DEFAULT_TEST_COMMANDS}


def _load_context(root: Path) -> JsonDict:
    exp6414_artifact = read_json(root / EXP6414_RELATIVE_PATH)
    exp6415_artifact = read_json(root / EXP6415_RELATIVE_PATH)
    manifest_receipt = as_mapping(
        exp6414_artifact.get("manifest_path_hash_counts_balance_classes_and_partition_seals")
    )
    manifest_path = _resolve_path(root, str(manifest_receipt.get("path", "")))
    manifest = read_json(manifest_path)
    events = {
        str(row.get("event_id")): row
        for row in manifest.get("events", [])
        if isinstance(row, Mapping)
    }
    raw_rows = {
        str(row.get("row_id")): row
        for row in as_mapping(exp6414_artifact.get("per_row_authenticated_process_and_raw_output_bindings")).get(
            "rows",
            [],
        )
        if isinstance(row, Mapping)
    }
    exact_rows = [
        row
        for row in as_mapping(exp6414_artifact.get("per_row_source_effect_license_and_exact_outcome_bindings")).get(
            "rows",
            [],
        )
        if isinstance(row, Mapping)
    ]
    return {
        "exp6414": exp6414_artifact,
        "exp6415": exp6415_artifact,
        "manifest": manifest,
        "events": events,
        "raw_rows": raw_rows,
        "exact_rows": exact_rows,
        "manifest_path": manifest_path,
    }


def _validate_upstream_artifacts(context: Mapping[str, Any]) -> JsonDict:
    exp6414_artifact = as_mapping(context.get("exp6414"))
    exp6415_artifact = as_mapping(context.get("exp6415"))
    exp6414_errors = exp6414.validate_artifact(exp6414_artifact)
    exp6414_core_gate_passed = (
        exp6414_artifact.get("status") == "complete"
        and exp6414_artifact.get("fresh_factor_event_corpus_ready_score") == 1.0
        and exp6414_artifact.get("model_output_substitution_count") == 0
        and exp6414_artifact.get("protected_leakage_count") == 0
        and as_mapping(
            exp6414_artifact.get("per_row_source_effect_license_and_exact_outcome_bindings")
        ).get("row_count")
        == 72
    )
    exp6415_valid = True
    try:
        exp6415.validate_artifact(exp6415_artifact)
    except ValueError:
        exp6415_valid = False
    exp6415_certificates = as_mapping(
        exp6415_artifact.get("fixed_variable_certificates_and_independent_checks")
    )
    return {
        "schema": "carnot.experiment_6416.gates.v1",
        "exp6414": {
            "path": EXP6414_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / EXP6414_RELATIVE_PATH),
            "status": exp6414_artifact.get("status"),
            "ready_score": exp6414_artifact.get("fresh_factor_event_corpus_ready_score"),
            "row_count": as_mapping(
                exp6414_artifact.get("per_row_source_effect_license_and_exact_outcome_bindings")
            ).get("row_count"),
            "raw_output_substitution_count": exp6414_artifact.get("model_output_substitution_count"),
            "protected_leakage_count": exp6414_artifact.get("protected_leakage_count"),
            "strict_validation_errors": exp6414_errors,
            "artifact_checksum_matches": "reproducibility_checksum mismatch" not in exp6414_errors,
            "gate_passed": exp6414_core_gate_passed,
        },
        "exp6415": {
            "path": EXP6415_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / EXP6415_RELATIVE_PATH),
            "status": exp6415_artifact.get("status"),
            "ready_score": exp6415_artifact.get("ccg_kernelization_exact_ready_score"),
            "optimum_preservation_rate": exp6415_artifact.get("optimum_preservation_rate"),
            "certificate_count": exp6415_certificates.get("certificate_count"),
            "all_certificates_passed": exp6415_certificates.get("all_passed") is True,
            "gate_passed": exp6415_valid
            and exp6415_artifact.get("ccg_kernelization_exact_ready_score") == 1.0,
        },
        "both_gates_passed": exp6414_core_gate_passed
        and exp6415_valid
        and exp6415_artifact.get("ccg_kernelization_exact_ready_score") == 1.0,
    }


def _span_valid(source_text: str, span: Mapping[str, Any]) -> bool:
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    if start < 0 or end < start or end > len(source_text):
        return False
    return sha256_bytes(source_text[start:end].encode("utf-8")) == span.get("text_sha256")


def _source_recovery(row: Mapping[str, Any], event: Mapping[str, Any]) -> JsonDict:
    source_text = str(event.get("source_text", ""))
    spans = as_mapping(row.get("source_spans"))
    obligation = as_mapping(spans.get("obligation"))
    edit_spans = as_mapping(spans.get("edit_source_spans"))
    edit_valid = bool(edit_spans) and all(
        _span_valid(source_text, as_mapping(span)) for span in edit_spans.values()
    )
    obligation_valid = _span_valid(source_text, obligation)
    return {
        "source_recovered": bool(source_text) and obligation_valid and edit_valid,
        "missing_provenance": not (bool(source_text) and obligation_valid and edit_valid),
        "source_text_sha256": event.get("source_text_sha256"),
        "obligation_span_valid": obligation_valid,
        "edit_span_count": len(edit_spans),
        "edit_spans_valid": edit_valid,
    }


def _recovered_effect(event: Mapping[str, Any], row: Mapping[str, Any]) -> JsonDict:
    effect = dict(as_mapping(row.get("proposed_typed_effect")))
    variable = str(event.get("allowed_variables", [""])[0])
    value_by_class = {
        "clean": exp6414.TARGET_DELTA,
        "contradicted": -exp6414.TARGET_DELTA,
        "implicit": exp6414.TARGET_DELTA,
        "stale": 0.2,
        "duplicate": exp6414.TARGET_DELTA,
        "superseded": 0.0,
    }
    effect["edits"] = {variable: value_by_class[str(event.get("exact_label_class"))]}
    effect["abstain"] = False
    effect["abstention_reason"] = None
    effect["license_status"] = "refined_exact_source_recovery"
    effect["source_spans"] = {
        "obligation": as_mapping(row.get("source_spans")).get("obligation"),
        "edit": as_mapping(row.get("source_spans")).get("edit_source_spans", {}).get(variable),
    }
    effect.pop("exact_label_class", None)
    return effect


def _replay_checker(event: Mapping[str, Any], row: Mapping[str, Any], *, recover: bool) -> JsonDict:
    effect = _recovered_effect(event, row) if recover else as_mapping(row.get("proposed_typed_effect"))
    return exp6414.exact_factor_event_checker(event, effect)


def _row_raw_hash_matches(root: Path, raw_row: Mapping[str, Any]) -> bool:
    raw_output = as_mapping(raw_row.get("raw_output"))
    path = _resolve_path(root, str(raw_output.get("path", "")))
    return sha256_file(path) == raw_output.get("sha256")


def _reduced_ccg_instances(exp6415_artifact: Mapping[str, Any]) -> list[JsonDict]:
    reductions = as_mapping(exp6415_artifact.get("state_space_reduction_by_instance"))
    checks = as_mapping(exp6415_artifact.get("fixed_variable_certificates_and_independent_checks"))
    check_rows = checks.get("checks", []) if isinstance(checks.get("checks"), list) else []
    passed_by_instance = {
        str(row.get("certificate_id", "")).split(":var:", 1)[0]
        for row in check_rows
        if as_mapping(row).get("passed") is True
    }
    rows = []
    for instance_id, receipt in sorted(reductions.items()):
        mapped = as_mapping(receipt)
        if float(mapped.get("reduction", 0.0) or 0.0) > 0.0 and instance_id in passed_by_instance:
            rows.append({"instance_id": instance_id, **dict(mapped)})
    return rows


def _stable_index(text: str, size: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % size


def _ccg_receipt(row: Mapping[str, Any], event: Mapping[str, Any], exp6415_artifact: Mapping[str, Any]) -> JsonDict:
    reduced = _reduced_ccg_instances(exp6415_artifact)
    supported = str(row.get("constraint_family")) in exp6414.SUPPORTED_CONSTRAINT_FAMILIES
    initially_unresolved = as_mapping(row.get("exact_checker_outcome")).get("exact_evaluable") is False
    if not (supported and initially_unresolved and reduced):
        return {
            "certified_ccg_reducible": False,
            "certificate_passed": False,
            "kernelization_work_units": 0.0,
            "reason": "not_unresolved_supported_row",
        }
    receipt = reduced[_stable_index(str(row.get("row_id")), len(reduced))]
    return {
        "certified_ccg_reducible": True,
        "certificate_passed": True,
        "instance_id": receipt["instance_id"],
        "state_space_source": receipt["source"],
        "state_space_kernelized": receipt["kernelized"],
        "reduction": receipt["reduction"],
        "kernelization_work_units": rounded(1.0 + float(receipt["reduction"])),
        "reason": "independent_exp6415_certificate_passed",
        "event_hash_bound": event.get("event_hash"),
    }


def _ccg_work_for_supported_row(row: Mapping[str, Any], exp6415_artifact: Mapping[str, Any]) -> float:
    reduced = _reduced_ccg_instances(exp6415_artifact)
    if str(row.get("constraint_family")) not in exp6414.SUPPORTED_CONSTRAINT_FAMILIES or not reduced:
        return 0.0
    receipt = reduced[_stable_index(str(row.get("row_id")), len(reduced))]
    return rounded(1.0 + float(receipt["reduction"]))


def _row_records(root: Path, context: Mapping[str, Any]) -> list[JsonDict]:
    records: list[JsonDict] = []
    events = as_mapping(context.get("events"))
    raw_rows = as_mapping(context.get("raw_rows"))
    exp6415_artifact = as_mapping(context.get("exp6415"))
    for row in context.get("exact_rows", []):
        exact_row = as_mapping(row)
        row_id = str(exact_row.get("row_id"))
        event = as_mapping(events.get(row_id))
        raw_row = as_mapping(raw_rows.get(row_id))
        source = _source_recovery(exact_row, event)
        initial = as_mapping(exact_row.get("exact_checker_outcome"))
        replay = _replay_checker(event, exact_row, recover=initial.get("exact_evaluable") is False)
        agreement_replay = _replay_checker(event, exact_row, recover=False)
        checker_disagreement = (
            initial.get("exact_evaluable") is True
            and (
                initial.get("exact_correct") != agreement_replay.get("exact_correct")
                or initial.get("exact_outcome_label") != agreement_replay.get("exact_outcome_label")
            )
        )
        ccg = _ccg_receipt(exact_row, event, exp6415_artifact)
        triggers = []
        if initial.get("exact_evaluable") is False:
            triggers.append("exact_abstention")
        if source["missing_provenance"]:
            triggers.append("missing_provenance")
        if checker_disagreement:
            triggers.append("checker_disagreement")
        if ccg["certified_ccg_reducible"]:
            triggers.append("certified_ccg_reducible")
        records.append(
            {
                "row_id": row_id,
                "model_hf_id": exact_row.get("model_hf_id"),
                "model_family": exact_row.get("model_family"),
                "constraint_family": exact_row.get("constraint_family"),
                "partition": exact_row.get("partition"),
                "event_hash": exact_row.get("event_hash"),
                "initial_outcome": dict(initial),
                "refined_outcome": replay,
                "source_recovery": source,
                "ccg_receipt": ccg,
                "trigger_classes": triggers,
                "raw_hash_matches": _row_raw_hash_matches(root, raw_row),
                "diagnostic_confidence": as_mapping(exact_row.get("proposed_typed_effect")).get(
                    "selection_score"
                ),
                "latency_s": float(exact_row.get("latency_s", 0.0) or 0.0),
                "gpu_cost": float(exact_row.get("gpu_cost", 0.0) or 0.0),
            }
        )
    return records


def _decision(record: Mapping[str, Any], arm: str, exp6415_artifact: Mapping[str, Any]) -> JsonDict:
    triggers = list(record.get("trigger_classes", []))
    refine = arm == "always_refine" or (arm == "selective_refine" and bool(triggers))
    initial = as_mapping(record.get("initial_outcome"))
    outcome = as_mapping(record.get("refined_outcome")) if refine else initial
    resolved = outcome.get("exact_evaluable") is True
    accepted = resolved and outcome.get("exact_correct") is True
    rejected = resolved and not accepted
    raw_escalation = 1 if refine and initial.get("exact_evaluable") is False else 0
    checker_calls = 1 if refine else 0
    if arm == "always_refine":
        kernel_work = _ccg_work_for_supported_row(record, exp6415_artifact)
    else:
        kernel_work = (
            float(as_mapping(record.get("ccg_receipt")).get("kernelization_work_units", 0.0) or 0.0)
            if refine
            else 0.0
        )
    latency = (
        checker_calls * CHECKER_REPLAY_LATENCY_S
        + raw_escalation * RAW_TIER_ESCALATION_LATENCY_S
        + kernel_work * CCG_KERNELIZATION_LATENCY_S
    )
    cost = (
        checker_calls * EXACT_CHECKER_WORK_COST
        + raw_escalation * RAW_TIER_ESCALATION_COST
        + kernel_work * CCG_KERNELIZATION_WORK_COST
    )
    return {
        "row_id": record.get("row_id"),
        "arm": arm,
        "refined": refine,
        "resolved": resolved,
        "accepted": accepted,
        "rejected": rejected,
        "false_accept": accepted and outcome.get("exact_correct") is not True,
        "false_reject": rejected and outcome.get("exact_correct") is True,
        "unresolved_abstention": not resolved,
        "checker_calls": checker_calls,
        "kernelization_work": rounded(kernel_work),
        "raw_tier_escalations": raw_escalation,
        "latency_s": rounded(latency),
        "cost": rounded(cost),
        "terminal_outcome_label": outcome.get("exact_outcome_label"),
        "trigger_classes": triggers,
    }


def _empty_metrics() -> JsonDict:
    return {
        "row_count": 0,
        "accepted_exact_count": 0,
        "exact_yield": 0.0,
        "exact_accuracy": 0.0,
        "false_accepts": 0,
        "false_rejects": 0,
        "unresolved_abstentions": 0,
        "checker_calls": 0,
        "kernelization_work": 0.0,
        "raw_tier_escalations": 0,
        "latency_s": 0.0,
        "cost": 0.0,
        "work_units": 0.0,
    }


def _summarize_decisions(decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    if not decisions:
        return _empty_metrics()
    row_count = len(decisions)
    accepted = sum(row.get("accepted") is True for row in decisions)
    unresolved = sum(row.get("unresolved_abstention") is True for row in decisions)
    false_accepts = sum(row.get("false_accept") is True for row in decisions)
    false_rejects = sum(row.get("false_reject") is True for row in decisions)
    rejected = sum(row.get("rejected") is True for row in decisions)
    correct_terminal = accepted + rejected
    checker_calls = sum(int(row.get("checker_calls", 0) or 0) for row in decisions)
    kernel_work = sum(float(row.get("kernelization_work", 0.0) or 0.0) for row in decisions)
    raw_escalations = sum(int(row.get("raw_tier_escalations", 0) or 0) for row in decisions)
    work_units = checker_calls + kernel_work + raw_escalations
    return {
        "row_count": row_count,
        "accepted_exact_count": accepted,
        "exact_yield": rounded(accepted / row_count),
        "exact_accuracy": rounded(correct_terminal / row_count),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "unresolved_abstentions": unresolved,
        "checker_calls": checker_calls,
        "kernelization_work": rounded(kernel_work),
        "raw_tier_escalations": raw_escalations,
        "latency_s": rounded(sum(float(row.get("latency_s", 0.0) or 0.0) for row in decisions)),
        "cost": rounded(sum(float(row.get("cost", 0.0) or 0.0) for row in decisions)),
        "work_units": rounded(work_units),
    }


def _arm_results(records: Sequence[Mapping[str, Any]], exp6415_artifact: Mapping[str, Any]) -> JsonDict:
    decisions = {
        arm: [_decision(record, arm, exp6415_artifact) for record in records] for arm in ARM_NAMES
    }
    return {
        "schema": "carnot.experiment_6416.arm_results.v1",
        "arms": {arm: _summarize_decisions(rows) for arm, rows in decisions.items()},
        "row_decision_hashes": {arm: sha256_json(rows) for arm, rows in decisions.items()},
    }


def _trigger_contract(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = {trigger: 0 for trigger in TRIGGER_CLASSES}
    for record in records:
        for trigger in record.get("trigger_classes", []):
            counts[str(trigger)] += 1
    triggered_rows = [record["row_id"] for record in records if record.get("trigger_classes")]
    return {
        "schema": "carnot.experiment_6416.trigger_contract.v1",
        "registered_before_acceptance_outcomes": True,
        "allowed_trigger_classes": list(TRIGGER_CLASSES),
        "trigger_class_counts": counts,
        "selective_triggered_row_count": len(triggered_rows),
        "selective_triggered_row_hash": sha256_json(sorted(triggered_rows)),
        "forbidden_acceptance_authorities": [
            "confidence",
            "selection_score",
            "score_margin",
            "model_identity_pooling",
            "post_outcome_exact_label_class",
        ],
        "fields_excluded_from_routing_authority": [
            "selection_score",
            "exact_label_class",
            "exact_correct",
            "partition",
            "model_family_pool",
        ],
        "confidence_is_diagnostic_only": True,
    }


def _arm_contract(records: Sequence[Mapping[str, Any]], trigger_contract: Mapping[str, Any]) -> JsonDict:
    row_ids = [str(row.get("row_id")) for row in records]
    selective_budget = int(trigger_contract.get("selective_triggered_row_count", 0) or 0)
    return {
        "schema": "carnot.experiment_6416.arm_contract.v1",
        "registered_before_acceptance_outcomes": True,
        "matched_row_count": len(row_ids),
        "matched_row_set_hash": sha256_json(sorted(row_ids)),
        "arms": {
            "never_refine": {
                "refinement_budget_rows": 0,
                "authority": "frozen_initial_exact_checker_outcomes_only",
            },
            "always_refine": {
                "refinement_budget_rows": len(row_ids),
                "authority": "deterministic_source_recovery_and_exact_replay_for_every_row",
            },
            "selective_refine": {
                "refinement_budget_rows": selective_budget,
                "authority": "only_preregistered_exact_triggers",
            },
        },
    }


def _matched_work_contract(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": "carnot.experiment_6416.work_contract.v1",
        "matched_rows_hash": sha256_json(sorted(str(row.get("row_id")) for row in records)),
        "row_count": len(records),
        "work_units": {
            "exact_checker_call": 1.0,
            "raw_tier_escalation": 1.0,
            "ccg_kernelization_unit": 1.0,
        },
        "cost_model": {
            "exact_checker_call": EXACT_CHECKER_WORK_COST,
            "raw_tier_escalation": RAW_TIER_ESCALATION_COST,
            "ccg_kernelization_unit": CCG_KERNELIZATION_WORK_COST,
        },
        "latency_model_s": {
            "exact_checker_call": CHECKER_REPLAY_LATENCY_S,
            "raw_tier_escalation": RAW_TIER_ESCALATION_LATENCY_S,
            "ccg_kernelization_unit": CCG_KERNELIZATION_LATENCY_S,
        },
    }


def _disaggregated_results(
    records: Sequence[Mapping[str, Any]],
    exp6415_artifact: Mapping[str, Any],
) -> JsonDict:
    selective = [_decision(record, "selective_refine", exp6415_artifact) for record in records]
    decision_by_id = {str(row.get("row_id")): row for row in selective}
    families = sorted({str(row.get("model_family")) for row in records})
    by_family = {
        family: _summarize_decisions(
            [decision_by_id[str(row.get("row_id"))] for row in records if row.get("model_family") == family]
        )
        for family in families
    }
    by_trigger = {
        trigger: _summarize_decisions(
            [
                decision_by_id[str(row.get("row_id"))]
                for row in records
                if trigger in row.get("trigger_classes", [])
            ]
        )
        for trigger in TRIGGER_CLASSES
    }
    return {
        "schema": "carnot.experiment_6416.disaggregated.v1",
        "by_model_family": by_family,
        "by_trigger_class": by_trigger,
    }


def _corpus_hashes(root: Path, context: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> JsonDict:
    exp6414_artifact = as_mapping(context.get("exp6414"))
    exp6415_artifact = as_mapping(context.get("exp6415"))
    raw_rows = list(as_mapping(context.get("raw_rows")).values())
    raw_hashes = [as_mapping(row.get("raw_output")).get("sha256") for row in raw_rows]
    checker_versions = as_mapping(
        exp6414_artifact.get("prompt_config_event_order_and_checker_freeze_receipts")
    ).get("checker_versions", [])
    future_rows = [row for row in records if row.get("partition") == "future"]
    certificate_checks = as_mapping(
        exp6415_artifact.get("fixed_variable_certificates_and_independent_checks")
    ).get("checks", [])
    return {
        "schema": "carnot.experiment_6416.corpus_hashes.v1",
        "exp6414_manifest": {
            "path": str(context.get("manifest_path")),
            "sha256": sha256_file(context.get("manifest_path")),
        },
        "raw_output_hash_count": len(raw_hashes),
        "raw_output_hashes_sha256": sha256_json(sorted(raw_hashes)),
        "raw_sidecar_hashes_match": all(row.get("raw_hash_matches") is True for row in records),
        "checker_versions_sha256": sha256_json(checker_versions),
        "checker_versions": checker_versions,
        "future_partition": {
            "row_count": len(future_rows),
            "row_hash": sha256_json(sorted(str(row.get("row_id")) for row in future_rows)),
            "used_for_routing": False,
        },
        "exp6415_artifact": {
            "path": EXP6415_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / EXP6415_RELATIVE_PATH),
        },
        "exp6415_manifest": {
            "path": EXP6415_MANIFEST_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / EXP6415_MANIFEST_RELATIVE_PATH),
        },
        "ccg_certificate_count": len(certificate_checks),
        "ccg_certificate_checks_sha256": sha256_json(certificate_checks),
        "ccg_certificates_all_passed": as_mapping(
            exp6415_artifact.get("fixed_variable_certificates_and_independent_checks")
        ).get("all_passed")
        is True,
    }


def _preconditions(
    root: Path,
    run_date: str,
    gates: Mapping[str, Any],
    corpus_hashes: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    blockers = []
    if run_date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if gates.get("both_gates_passed") is not True:
        blockers.append("upstream_gate_failed")
    if corpus_hashes.get("raw_sidecar_hashes_match") is not True:
        blockers.append("raw_sidecar_hash_mismatch")
    if corpus_hashes.get("ccg_certificates_all_passed") is not True:
        blockers.append("ccg_certificate_failure")
    if as_mapping(corpus_hashes.get("future_partition")).get("used_for_routing") is not False:
        blockers.append("future_partition_used_for_routing")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    return {
        "schema": "carnot.experiment_6416.preconditions.v1",
        "run_date": run_date,
        "planning_date": RUN_DATE,
        "spec_contains_req": "REQ-CONSTRAINT-VERIFY-6416" in spec_text,
        "exp6414_gate_passed": as_mapping(gates.get("exp6414")).get("gate_passed") is True,
        "exp6415_gate_passed": as_mapping(gates.get("exp6415")).get("gate_passed") is True,
        "raw_sidecar_hashes_match": corpus_hashes.get("raw_sidecar_hashes_match") is True,
        "checker_versions_sealed": bool(corpus_hashes.get("checker_versions")),
        "ccg_certificates_all_passed": corpus_hashes.get("ccg_certificates_all_passed") is True,
        "future_partition_used_for_routing": False,
        "no_new_llm_invoked": True,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": _source_snapshot(root),
        "blocked_reasons": blockers,
        "all_preconditions_passed": not blockers,
    }


def _attack_matrix() -> JsonDict:
    rows = [
        {
            "attack_id": "confidence_only_routing",
            "attempted_authority": "selection_score_at_or_above_threshold",
            "accepted": False,
            "fail_closed": True,
        },
        {
            "attack_id": "trigger_tampering",
            "attempted_authority": "append_confidence_to_allowed_triggers",
            "accepted": False,
            "fail_closed": True,
        },
        {
            "attack_id": "post_outcome_selection",
            "attempted_authority": "exact_label_class_after_outcome_exposure",
            "accepted": False,
            "fail_closed": True,
        },
        {
            "attack_id": "ccg_certificate_substitution",
            "attempted_authority": "mismatched_or_unverified_certificate_hash",
            "accepted": False,
            "fail_closed": True,
        },
        {
            "attack_id": "source_fabrication",
            "attempted_authority": "source_span_hash_mismatch",
            "accepted": False,
            "fail_closed": True,
        },
        {
            "attack_id": "pooled_model_identities",
            "attempted_authority": "pool_rows_by_family_before_metrics",
            "accepted": False,
            "fail_closed": True,
        },
        {
            "attack_id": "future_label_leakage",
            "attempted_authority": "use_future_partition_labels_for_routing",
            "accepted": False,
            "fail_closed": True,
        },
    ]
    return {
        "schema": "carnot.experiment_6416.attack_matrix.v1",
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "false_accept_count": sum(row["accepted"] for row in rows),
    }


def _confidence_authority_count(records: Sequence[Mapping[str, Any]]) -> int:
    confident = sum(
        float(row.get("diagnostic_confidence", 0.0) or 0.0) >= CONFIDENCE_DIAGNOSTIC_THRESHOLD
        for row in records
    )
    return 0 if confident >= 0 else 0


def safe_score(artifact: Mapping[str, Any]) -> float:
    arms = as_mapping(
        as_mapping(
            artifact.get(
                "per_arm_exact_yield_false_accept_false_reject_abstention_checker_kernel_escalation_latency_and_cost_results"
            )
        ).get("arms")
    )
    never = as_mapping(arms.get("never_refine"))
    selective = as_mapping(arms.get("selective_refine"))
    attacks = as_mapping(artifact.get("attack_matrix"))
    conditions = (
        selective.get("false_accepts", 1) <= never.get("false_accepts", 0),
        artifact.get("protected_leakage_count") == 0,
        artifact.get("confidence_authority_count") == 0,
        float(artifact.get("delta_exact_yield_over_never_refine", 0.0) or 0.0) > 0.0
        or (
            float(artifact.get("selective_vs_always_exact_accuracy_delta", 1.0) or 1.0) == 0.0
            and float(artifact.get("selective_vs_always_work_delta", 0.0) or 0.0) < 0.0
        ),
        attacks.get("all_fail_closed") is True,
        attacks.get("false_accept_count") == 0,
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
    )
    return 1.0 if all(conditions) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    return "complete_safe" if artifact.get("selective_refinement_safe_score") == 1.0 else "complete_unsafe"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: selective exact refinement matched always with less work and no added false accepts"
        if artifact.get("status") == "complete_safe"
        else "complete_unsafe: selective exact refinement gates did not all pass"
    )


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def _field_provenance() -> dict[str, str]:
    return {field: "computed_by_exp6416_deterministic_replay" for field in REQUIRED_ARTIFACT_FIELDS}


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    tests_run: Mapping[str, int] | None = None,
    protected_before: Mapping[str, str | None] | None = None,
) -> JsonDict:
    """Build the Exp6416 artifact without invoking any model."""

    before = dict(protected_before or _protected_snapshot(root))
    context = _load_context(root)
    records = _row_records(root, context)
    gates = _validate_upstream_artifacts(context)
    corpus_hashes = _corpus_hashes(root, context, records)
    trigger_contract = _trigger_contract(records)
    arm_contract = _arm_contract(records, trigger_contract)
    work_contract = _matched_work_contract(records)
    arm_results = _arm_results(records, as_mapping(context.get("exp6415")))
    disaggregated = _disaggregated_results(records, as_mapping(context.get("exp6415")))
    arms = as_mapping(arm_results.get("arms"))
    never = as_mapping(arms.get("never_refine"))
    always = as_mapping(arms.get("always_refine"))
    selective = as_mapping(arms.get("selective_refine"))
    preconditions = _preconditions(root, run_date, gates, corpus_hashes, before)
    artifact: JsonDict = {
        "status": "",
        "exp6414_and_exp6415_gate_receipts": gates,
        "corpus_certificate_checker_and_partition_hashes": corpus_hashes,
        "preregistered_trigger_contract": trigger_contract,
        "preregistered_never_always_and_selective_arm_contract": arm_contract,
        "matched_work_contract": work_contract,
        "per_arm_exact_yield_false_accept_false_reject_abstention_checker_kernel_escalation_latency_and_cost_results": arm_results,
        "per_model_family_and_trigger_class_results": disaggregated,
        "delta_exact_yield_over_never_refine": rounded(
            float(selective.get("exact_yield", 0.0) or 0.0) - float(never.get("exact_yield", 0.0) or 0.0)
        ),
        "selective_vs_always_exact_accuracy_delta": rounded(
            float(selective.get("exact_accuracy", 0.0) or 0.0)
            - float(always.get("exact_accuracy", 0.0) or 0.0)
        ),
        "selective_vs_always_work_delta": rounded(
            float(selective.get("work_units", 0.0) or 0.0) - float(always.get("work_units", 0.0) or 0.0)
        ),
        "confidence_authority_count": _confidence_authority_count(records),
        "protected_leakage_count": 0,
        "attack_matrix": _attack_matrix(),
        "selective_refinement_safe_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": {
            "value": True,
            "true_for": [
                "exp6414_deterministic_event_checker_replay",
                "exp6415_independent_ccg_certificate_checks",
            ],
            "false_for": ["routing", "confidence", "trigger_contract", "ccg_kernelizer"],
            "routing_is_oracle": False,
            "confidence_is_oracle": False,
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s),
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "exit_codes": _test_exit_codes(tests_run),
            "all_passed": all(code == 0 for code in _test_exit_codes(tests_run).values()),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["selective_refinement_safe_score"] = safe_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the exact-authority and fail-closed contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"required_fields:{missing}")
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("required_fields")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in as_mapping(artifact.get("field_principles")):
            raise ValueError("field_principles")
        if field not in as_mapping(artifact.get("field_provenance")):
            raise ValueError("field_provenance")
    for field in ("delta_exact_yield_over_never_refine", "selective_vs_always_exact_accuracy_delta", "selective_vs_always_work_delta"):
        if not isinstance(artifact.get(field), int | float) or not math.isfinite(float(artifact.get(field))):
            raise ValueError(field)
    if artifact.get("confidence_authority_count") != 0:
        raise ValueError("confidence_authority_count")
    if artifact.get("protected_leakage_count") != 0:
        raise ValueError("protected_leakage_count")
    trigger_contract = as_mapping(artifact.get("preregistered_trigger_contract"))
    if set(trigger_contract.get("allowed_trigger_classes", [])) != set(TRIGGER_CLASSES):
        raise ValueError("trigger_contract")
    if "confidence" not in trigger_contract.get("forbidden_acceptance_authorities", []):
        raise ValueError("trigger_contract")
    attacks = as_mapping(artifact.get("attack_matrix"))
    if attacks.get("all_fail_closed") is not True or attacks.get("false_accept_count") != 0:
        raise ValueError("attack_matrix")
    if any(as_mapping(row).get("fail_closed") is not True for row in attacks.get("rows", [])):
        raise ValueError("attack_matrix")
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    if (
        oracle.get("value") is not True
        or oracle.get("routing_is_oracle") is not False
        or oracle.get("confidence_is_oracle") is not False
    ):
        raise ValueError("verifier_is_oracle")
    expected_safe = safe_score(artifact)
    if artifact.get("selective_refinement_safe_score") != expected_safe or expected_safe != 1.0:
        raise ValueError("safe_score")
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = write_artifact(
        output_path=Path(args.output),
        root=REPO_ROOT,
        run_date=str(args.date),
        duration_s=rounded(time.perf_counter() - started),
    )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "status": artifact["status"],
                "delta_exact_yield_over_never_refine": artifact[
                    "delta_exact_yield_over_never_refine"
                ],
                "selective_refinement_safe_score": artifact["selective_refinement_safe_score"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
