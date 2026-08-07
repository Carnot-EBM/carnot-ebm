"""Exp6175 CCTU headroom audit over the Exp6174 K8 pool.

Spec refs: REQ-CONSTRAINT-VERIFY-6175,
SCENARIO-CONSTRAINT-VERIFY-6175-RAW-REVALIDATION,
SCENARIO-CONSTRAINT-VERIFY-6175-NO-HELD-ROWS,
SCENARIO-CONSTRAINT-VERIFY-6175-RETIRE-PARSE-FAILURE.

The audit is deliberately selector-free. It uses exact validators only to
evaluate frozen rows after generation, and it emits held results as aggregate
qualification fields so later selector work cannot see held row labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from carnot.verify import cctu_item_bank_6173 as exp6173


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260807"
SCHEMA = "carnot.experiment_6175.cctu_headroom_audit.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6175_cctu_headroom_audit.json")
EXP6174_RELATIVE_PATH = Path("results/experiment_6174_cctu_authentic_k8_pool.json")
RAW_TRACE_RELATIVE_PATH = Path("results/experiment_6174_cctu_authentic_k8_pool.raw_traces.jsonl")
CALIBRATION_LABEL_RELATIVE_PATH = Path(
    "results/experiment_6174_cctu_authentic_k8_pool.calibration_labels.jsonl"
)
HELD_LABEL_RELATIVE_PATH = Path("results/experiment_6174_cctu_authentic_k8_pool.held_labels.jsonl")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/verify/cctu_headroom_audit_6175.py")
TEST_RELATIVE_PATH = Path("tests/python/test_cctu_headroom_audit_6175.py")
INFERENCE_SUBSTRATE = "deterministic_exact_tool_trace_headroom_audit"
K_SAMPLES = 8

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "upstream_corpus_bank_split_validator_and_preregistration_hashes",
    "label_revalidation_receipt",
    "all_sample_and_parseable_denominators",
    "family_constraint_count_and_violation_strata",
    "exact_floor_definition_value_and_provenance",
    "per_candidate_competence_and_clustered_interval",
    "saturation_and_error_diversity_metrics",
    "tuned_oracle_blind_consensus_definition_freeze_and_accuracy",
    "oracle_at_8_accuracy",
    "oracle_minus_consensus_delta_and_clustered_interval",
    "consensus_wrong_oracle_right_group_count",
    "duplicate_and_shortcut_audits",
    "held_aggregate_qualification_and_row_label_seal_hash",
    "parseability_competence_unsaturation_headroom_and_minority_gate_matrix",
    "phase_d_headroom_ready_score",
    "future_rows_allowed_by_this_artifact",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FOCUSED_COMMAND = ".venv/bin/pytest tests/python/test_cctu_headroom_audit_6175.py -q --no-cov -n 0"
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/verify/cctu_headroom_audit_6175.py "
    "-m pytest tests/python/test_cctu_headroom_audit_6175.py -q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/verify/cctu_headroom_audit_6175.py --fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_cctu_headroom_audit_6175.py"
)
SCHEMA_COMMAND = ".venv/bin/python -m carnot.verify.cctu_headroom_audit_6175 --validate"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6175_cctu_headroom_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    SCHEMA_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)


FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state is retired unless every preregistered headroom conjunct passes.",
    "preconditions_checked": "Exp6174 gate, hashes, raw-before-label, seals, K coverage, no retry, paths, exclusions, and protected files are checked first.",
    "structured_gate_receipt": "The Exp6174 structured generation gate is copied and explicitly checked.",
    "upstream_corpus_bank_split_validator_and_preregistration_hashes": "Current bytes are compared against Exp6174's declared upstream corpus anchors.",
    "label_revalidation_receipt": "Every sidecar label is replayed from immutable raw completion text with the exact Exp6173 validator.",
    "all_sample_and_parseable_denominators": "Headline denominators keep parse failures and exact-validator failures in the sample count.",
    "family_constraint_count_and_violation_strata": "Family and constraint-count aggregates show where failures concentrate without dropping rows.",
    "exact_floor_definition_value_and_provenance": "The exact random executable-plan floor comes from the Exp6173 preregistration.",
    "per_candidate_competence_and_clustered_interval": "Candidate competence is all-sample exact-validator accuracy with case-clustered uncertainty.",
    "saturation_and_error_diversity_metrics": "Unsaturation and error diversity are measured from exact labels and normalized outcome clusters.",
    "tuned_oracle_blind_consensus_definition_freeze_and_accuracy": "Consensus is frozen from calibration-only normalized clusters and does not see oracle labels at selection time.",
    "oracle_at_8_accuracy": "Oracle@8 is the exact upper bound that any row-level selector could reach on the frozen K candidates.",
    "oracle_minus_consensus_delta_and_clustered_interval": "The primary estimand is case-clustered oracle@8 minus tuned consensus accuracy.",
    "consensus_wrong_oracle_right_group_count": "Counts selectable minority groups where exact oracle selection could beat consensus.",
    "duplicate_and_shortcut_audits": "Duplicate, position, identifier, hidden-state, and shortcut checks protect against artificial headroom.",
    "held_aggregate_qualification_and_row_label_seal_hash": "Held labels are used only internally and exported as aggregates plus a seal hash.",
    "parseability_competence_unsaturation_headroom_and_minority_gate_matrix": "Readiness is a strict conjunction, never a weighted average.",
    "phase_d_headroom_ready_score": "Bare one only when every preregistered conjunct passes.",
    "future_rows_allowed_by_this_artifact": "Future rows are allowed only when readiness is bare one.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "duration_s": "Measured wall time for the deterministic audit.",
    "inference_substrate": "Set deterministic_exact_tool_trace_headroom_audit.",
    "verifier_is_oracle": "Exact validators define evaluation labels; tuned consensus remains oracle-blind.",
    "field_provenance": "Every field traces to REQ-CONSTRAINT-VERIFY-6175, Exp6173, Exp6174, exact validators, or tests.",
    "test_commands": "Commands cover focused tests, new-code coverage, spec coverage, schema, adversarial verify, root clutter, protected files, and full pytest.",
    "test_exit_codes": "Exit codes make failed checks visible in the artifact.",
    "reproducibility_checksum": "Checksum detects source, corpus, label, aggregate, and command drift while excluding duration.",
    "honest_verdict": "Terminal prefix names passed or failed conjuncts.",
}

FIELD_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "status": ("REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT",),
    "preconditions_checked": ("REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",),
    "structured_gate_receipt": ("REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",),
    "upstream_corpus_bank_split_validator_and_preregistration_hashes": (
        "REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",
    ),
    "label_revalidation_receipt": ("REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",),
    "all_sample_and_parseable_denominators": (
        "REQ-CONSTRAINT-VERIFY-6175-PARSEABILITY",
    ),
    "family_constraint_count_and_violation_strata": (
        "REQ-CONSTRAINT-VERIFY-6175-PARSEABILITY",
        "REQ-CONSTRAINT-VERIFY-6175-ERROR-DIVERSITY",
    ),
    "exact_floor_definition_value_and_provenance": (
        "REQ-CONSTRAINT-VERIFY-6175-EXACT-FLOOR",
    ),
    "per_candidate_competence_and_clustered_interval": (
        "REQ-CONSTRAINT-VERIFY-6175-COMPETENCE",
        "REQ-CONSTRAINT-VERIFY-6175-CLUSTERED-INFERENCE",
    ),
    "saturation_and_error_diversity_metrics": (
        "REQ-CONSTRAINT-VERIFY-6175-UNSATURATION",
        "REQ-CONSTRAINT-VERIFY-6175-ERROR-DIVERSITY",
    ),
    "tuned_oracle_blind_consensus_definition_freeze_and_accuracy": (
        "REQ-CONSTRAINT-VERIFY-6175-CONSENSUS",
        "REQ-CONSTRAINT-VERIFY-6175-NO-SELECTOR",
    ),
    "oracle_at_8_accuracy": (
        "REQ-CONSTRAINT-VERIFY-6175-ORACLE-K",
        "REQ-CONSTRAINT-VERIFY-6175-CLUSTERED-INFERENCE",
    ),
    "oracle_minus_consensus_delta_and_clustered_interval": (
        "REQ-CONSTRAINT-VERIFY-6175-ORACLE-K",
        "REQ-CONSTRAINT-VERIFY-6175-CLUSTERED-INFERENCE",
    ),
    "consensus_wrong_oracle_right_group_count": (
        "REQ-CONSTRAINT-VERIFY-6175-ORACLE-K",
        "REQ-CONSTRAINT-VERIFY-6175-ERROR-DIVERSITY",
    ),
    "duplicate_and_shortcut_audits": (
        "REQ-CONSTRAINT-VERIFY-6175-ERROR-DIVERSITY",
        "REQ-CONSTRAINT-VERIFY-6175-NO-SELECTOR",
    ),
    "held_aggregate_qualification_and_row_label_seal_hash": (
        "REQ-CONSTRAINT-VERIFY-6175-HELD-AGGREGATE",
        "REQ-CONSTRAINT-VERIFY-6175-NO-SELECTOR",
    ),
    "parseability_competence_unsaturation_headroom_and_minority_gate_matrix": (
        "REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT",
    ),
    "phase_d_headroom_ready_score": (
        "REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT",
    ),
    "future_rows_allowed_by_this_artifact": (
        "REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT",
        "REQ-CONSTRAINT-VERIFY-6175-NO-SELECTOR",
    ),
    "protected_files_unchanged": ("REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",),
    "duration_s": ("REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",),
    "inference_substrate": ("REQ-CONSTRAINT-VERIFY-6175",),
    "verifier_is_oracle": ("REQ-CONSTRAINT-VERIFY-6175-EXACT-FLOOR",),
    "field_provenance": ("REQ-CONSTRAINT-VERIFY-6175",),
    "test_commands": ("REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",),
    "test_exit_codes": ("REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",),
    "reproducibility_checksum": ("REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY",),
    "honest_verdict": ("REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT",),
}


def run(
    *,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp6175 audit artifact."""

    started = time.perf_counter()
    paths = default_paths(result_path=result_path)
    protected_before = protected_file_hash_map()
    exp6174_artifact = read_json(paths["exp6174"])
    preregistration = read_json(paths["preregistration"])
    raw_rows = read_jsonl(paths["raw_trace"])
    calibration_labels = read_jsonl(paths["calibration_label"])
    held_labels = read_jsonl(paths["held_label"])
    label_rows = calibration_labels + held_labels

    revalidation = revalidate_labels_from_raw(raw_rows, label_rows)
    records = build_candidate_records(raw_rows)
    consensus = freeze_and_score_consensus(records)
    oracle_metrics = oracle_at_8(records)
    delta_metrics = oracle_minus_consensus(records, consensus["selected_rule"])
    denominators = all_sample_and_parseable_denominators(records)
    competence = per_candidate_competence(records, preregistration)
    saturation = saturation_and_error_diversity(records, consensus)
    family_strata = family_constraint_count_and_violation_strata(records)
    minority = consensus_wrong_oracle_right_group_count(records, consensus["selected_rule"])
    held_aggregate = held_aggregate_qualification(records, consensus["selected_rule"], paths)
    duplicate_audits = duplicate_and_shortcut_audits(raw_rows, records)
    upstream = upstream_hash_receipt(exp6174_artifact)
    preconditions = preconditions_checked(
        paths=paths,
        exp6174_artifact=exp6174_artifact,
        raw_rows=raw_rows,
        label_rows=label_rows,
        upstream=upstream,
        revalidation=revalidation,
        protected_before=protected_before,
    )
    exact_floor = exact_floor_definition_value_and_provenance(preregistration)
    gate_matrix = gate_matrix_for_readiness(
        preconditions=preconditions,
        denominators=denominators,
        competence=competence,
        saturation=saturation,
        delta_metrics=delta_metrics,
        minority=minority,
        family_strata=family_strata,
        revalidation=revalidation,
    )
    ready = 1.0 if gate_matrix["all_conjuncts_passed"] else 0.0
    status = (
        "complete_ready"
        if ready == 1.0
        else ("blocked" if not preconditions["passed"] else "retired")
    )
    future_rows = ready == 1.0
    protected_after = protected_file_hash_map()
    measured_duration = round(
        duration_s if duration_s is not None else time.perf_counter() - started, 6
    )

    artifact: JsonDict = {
        "status": status,
        "preconditions_checked": preconditions,
        "structured_gate_receipt": exp6174_artifact.get("structured_gate_receipt", {}),
        "upstream_corpus_bank_split_validator_and_preregistration_hashes": upstream,
        "label_revalidation_receipt": revalidation,
        "all_sample_and_parseable_denominators": denominators,
        "family_constraint_count_and_violation_strata": family_strata,
        "exact_floor_definition_value_and_provenance": exact_floor,
        "per_candidate_competence_and_clustered_interval": competence,
        "saturation_and_error_diversity_metrics": saturation,
        "tuned_oracle_blind_consensus_definition_freeze_and_accuracy": consensus,
        "oracle_at_8_accuracy": oracle_metrics,
        "oracle_minus_consensus_delta_and_clustered_interval": delta_metrics,
        "consensus_wrong_oracle_right_group_count": minority,
        "duplicate_and_shortcut_audits": duplicate_audits,
        "held_aggregate_qualification_and_row_label_seal_hash": held_aggregate,
        "parseability_competence_unsaturation_headroom_and_minority_gate_matrix": gate_matrix,
        "phase_d_headroom_ready_score": ready,
        "future_rows_allowed_by_this_artifact": future_rows,
        "protected_files_unchanged": protected_files_unchanged(protected_before, protected_after),
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(status, gate_matrix),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        write_json(paths["result"], artifact)
    return artifact


def default_paths(*, result_path: Path | None = None) -> dict[str, Path]:
    """Resolve all local Exp6175 input and output paths."""

    return {
        "result": result_path or REPO_ROOT / RESULT_RELATIVE_PATH,
        "exp6174": REPO_ROOT / EXP6174_RELATIVE_PATH,
        "raw_trace": REPO_ROOT / RAW_TRACE_RELATIVE_PATH,
        "calibration_label": REPO_ROOT / CALIBRATION_LABEL_RELATIVE_PATH,
        "held_label": REPO_ROOT / HELD_LABEL_RELATIVE_PATH,
        "preregistration": REPO_ROOT
        / "results/experiment_6173_cctu_item_bank_preregistration.json",
        "item_bank": REPO_ROOT / "data/research/cctu_item_bank_6173.jsonl",
        "split": REPO_ROOT / "data/research/cctu_item_bank_6173_splits.json",
        "validator": REPO_ROOT / "python/carnot/verify/cctu_item_bank_6173.py",
        "held_access_log": REPO_ROOT / "data/research/cctu_item_bank_6173_held_access_log.json",
        "exclusion_manifest": REPO_ROOT / "ops/exclusion_manifest.yaml",
    }


def read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_pretty(value), encoding="utf-8")


def revalidate_labels_from_raw(
    raw_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Replay every sidecar label from immutable raw completion text."""

    raw_by_key = index_raw_rows(raw_rows)
    label_seen: set[str] = set()
    bank_by_id = {case.case_id: case for case in exp6173.build_item_bank()}
    mismatch_count = 0
    raw_hash_mismatch_count = 0
    version_mismatch_count = 0
    missing_raw_count = 0
    terminal_pass_count = 0
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    step_pass_counts: dict[str, Counter[str]] = defaultdict(Counter)
    violation_taxonomy: Counter[str] = Counter()

    for label in label_rows:
        sample_key = str(label.get("sample_key"))
        label_seen.add(sample_key)
        raw = raw_by_key.get(sample_key)
        if raw is None:
            missing_raw_count += 1
            continue
        if label.get("raw_row_hash") != raw.get("row_hash") or raw.get("row_hash") != raw_row_hash(
            raw
        ):
            raw_hash_mismatch_count += 1
        case = bank_by_id[str(label["case_id"])]
        validation = exp6173.validate_candidate_trace(case, str(raw.get("raw_completion_text", "")))
        if validation != label.get("validator_result"):
            mismatch_count += 1
        if label.get("validator_version") != exp6173.VALIDATOR_VERSION:
            version_mismatch_count += 1
        terminal_passed = bool(validation["terminal_passed"])
        terminal_pass_count += int(terminal_passed)
        split = str(label.get("split", raw.get("split", "unknown")))
        split_counts[split]["labels"] += 1
        split_counts[split]["terminal_passed"] += int(terminal_passed)
        for step in validation["step_results"]:
            category = str(step["category"])
            step_pass_counts[category]["total"] += 1
            step_pass_counts[category]["passed"] += int(bool(step["passed"]))
        for violation in validation["violations"]:
            violation_taxonomy[str(violation["category"])] += 1

    return {
        "schema": SCHEMA + ".label_revalidation",
        "raw_rows_revalidated": len(raw_rows),
        "label_rows_checked": len(label_rows),
        "raw_rows_without_label_count": len(set(raw_by_key) - label_seen),
        "label_rows_without_raw_count": missing_raw_count,
        "raw_row_hash_mismatch_count": raw_hash_mismatch_count,
        "validator_result_mismatch_count": mismatch_count,
        "validator_version": exp6173.VALIDATOR_VERSION,
        "validator_version_mismatch_count": version_mismatch_count,
        "terminal_pass_count": terminal_pass_count,
        "all_labels_match_revalidation": (
            mismatch_count == 0
            and raw_hash_mismatch_count == 0
            and version_mismatch_count == 0
            and missing_raw_count == 0
        ),
        "split_counts": _counter_mapping(split_counts),
        "partial_step_satisfaction": _step_satisfaction(step_pass_counts),
        "violation_taxonomy": dict(sorted(violation_taxonomy.items())),
        "held_row_labels_exposed": False,
    }


def index_raw_rows(raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Index raw rows by immutable sample key and reject duplicates."""

    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        key = str(raw.get("sample_key"))
        if key in indexed:
            raise ValueError(f"duplicate raw sample_key: {key}")
        indexed[key] = raw
    return indexed


def build_candidate_records(raw_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create internal row records from exact validator replay."""

    bank_by_id = {case.case_id: case for case in exp6173.build_item_bank()}
    records: list[JsonDict] = []
    for raw in raw_rows:
        case = bank_by_id[str(raw["case_id"])]
        validation = exp6173.validate_candidate_trace(case, str(raw.get("raw_completion_text", "")))
        parseable = _parse_json_ok(validation)
        records.append(
            {
                "case_id": raw["case_id"],
                "split": raw["split"],
                "family": case.family,
                "primary_constraint": case.primary_constraint,
                "constraint_count": len(case.taxonomy),
                "sample_index": raw["sample_index"],
                "terminal_passed": bool(validation["terminal_passed"]),
                "parseable": parseable,
                "cluster": normalize_action_terminal_cluster(
                    str(raw.get("raw_completion_text", ""))
                ),
                "raw_completion_sha256": raw.get("raw_completion_sha256"),
                "truncated": bool(raw.get("truncated")),
                "timeout": bool(raw.get("timeout")),
                "refusal": bool(raw.get("refusal")),
                "violation_categories": sorted(
                    {str(violation["category"]) for violation in validation["violations"]}
                ),
                "step_results": validation["step_results"],
            }
        )
    return records


def normalize_action_terminal_cluster(raw_text: str) -> str:
    """Normalize a candidate by visible actions and terminal outcome only."""

    stripped = raw_text.strip()
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        return f"unparseable:{exc.msg}:{sha256_text(stripped)[:18]}"
    if stripped[end:].strip() or not isinstance(parsed, Mapping):
        return f"unparseable:trailing-or-nonobject:{sha256_text(stripped)[:18]}"
    steps = parsed.get("steps") if isinstance(parsed.get("steps"), list) else []
    tools = [
        f"tool:{step.get('tool')}"
        for step in steps
        if isinstance(step, Mapping) and isinstance(step.get("tool"), str)
    ]
    final = parsed.get("final") if isinstance(parsed.get("final"), Mapping) else {}
    answer = str(final.get("answer", "<missing>")) if isinstance(final, Mapping) else "<missing>"
    abstain = (
        str(bool(final.get("abstain", False))).lower() if isinstance(final, Mapping) else "false"
    )
    return "|".join([*tools, f"final:{answer}|abstain:{abstain}"])


def freeze_and_score_consensus(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Tune the oracle-blind consensus rule on calibration clusters only."""

    rules = candidate_consensus_rules()
    calibration = [record for record in records if record["split"] == "calibration"]
    candidates = []
    for rule in rules:
        accuracy = _consensus_accuracy(calibration, rule)
        candidates.append({"rule": rule, "calibration_accuracy": accuracy})
    selected = sorted(
        candidates, key=lambda row: (-row["calibration_accuracy"], row["rule"]["rank"])
    )[0]
    selected_rule = selected["rule"]
    return {
        "schema": SCHEMA + ".oracle_blind_consensus",
        "tuning_split": "calibration",
        "selected_rule": selected_rule,
        "candidate_rule_scores": candidates,
        "calibration_accuracy": selected["calibration_accuracy"],
        "held_aggregate_accuracy": _consensus_accuracy(
            [record for record in records if record["split"] == "held"],
            selected_rule,
        ),
        "all_sample_case_accuracy": _consensus_accuracy(records, selected_rule),
        "normalized_cluster_fields": ["tool sequence", "final answer", "final abstain"],
        "forbidden_selection_inputs": {
            "validator_labels_at_selection_time": False,
            "held_labels_for_tuning": False,
            "hidden_states": False,
            "arbitrary_ids": False,
            "answer_positions": False,
            "sample_indexes": False,
        },
        "definition_freeze": (
            "calibration-only plurality over normalized action/terminal-outcome clusters; "
            "abstention/no-consensus is scored false"
        ),
    }


def candidate_consensus_rules() -> list[JsonDict]:
    """Return the preregistered oracle-blind consensus rule family."""

    return [
        {"name": "plurality_all_clusters", "min_agreement": 1, "parseable_only": False, "rank": 0},
        {
            "name": "plurality_parseable_preferred",
            "min_agreement": 1,
            "parseable_only": True,
            "rank": 1,
        },
        {
            "name": "plurality_all_clusters_min2",
            "min_agreement": 2,
            "parseable_only": False,
            "rank": 2,
        },
        {
            "name": "plurality_all_clusters_min3",
            "min_agreement": 3,
            "parseable_only": False,
            "rank": 3,
        },
    ]


def all_sample_and_parseable_denominators(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    parseable = [record for record in records if record["parseable"]]
    passed = [record for record in records if record["terminal_passed"]]
    by_split: dict[str, JsonDict] = {}
    for split in sorted({str(record["split"]) for record in records}):
        split_rows = [record for record in records if record["split"] == split]
        by_split[split] = {
            "all_count": len(split_rows),
            "parseable_count": sum(bool(row["parseable"]) for row in split_rows),
            "terminal_pass_count": sum(bool(row["terminal_passed"]) for row in split_rows),
            "parseability_rate": rate(
                sum(bool(row["parseable"]) for row in split_rows), len(split_rows)
            ),
        }
    return {
        "schema": SCHEMA + ".denominators",
        "all_samples": {
            "count": len(records),
            "terminal_pass_count": len(passed),
            "accuracy": rate(len(passed), len(records)),
        },
        "parseable_samples": {
            "count": len(parseable),
            "terminal_pass_count": sum(bool(record["terminal_passed"]) for record in parseable),
            "parseability_rate": rate(len(parseable), len(records)),
            "accuracy_if_parseable": rate(
                sum(bool(record["terminal_passed"]) for record in parseable),
                len(parseable),
            ),
        },
        "by_split": by_split,
        "headline_denominator_policy": {
            "never_drop_failures": True,
            "included_failure_surfaces": [
                "parse_failure",
                "validator_failure",
                "duplicate",
                "refusal",
                "timeout",
                "truncation",
            ],
        },
    }


def per_candidate_competence(
    records: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
) -> JsonDict:
    grouped = group_by_case(records)
    case_means = [
        sum(bool(row["terminal_passed"]) for row in rows) / len(rows) for rows in grouped.values()
    ]
    interval = clustered_interval(case_means)
    floor = float(
        preregistration.get("exact_floor_definition_and_provenance", {}).get(
            "floor_upper_bound", 0.05
        )
    )
    accuracy = rate(sum(bool(record["terminal_passed"]) for record in records), len(records))
    return {
        "schema": SCHEMA + ".competence",
        "accuracy_all_sample": accuracy,
        "candidate_count": len(records),
        "case_cluster_count": len(grouped),
        "clustered_interval": interval,
        "exact_floor": floor,
        "floor_source": "results/experiment_6173_cctu_item_bank_preregistration.json",
        "above_exact_floor_gate_passed": bool(interval["lower"] > floor),
    }


def saturation_and_error_diversity(
    records: Sequence[Mapping[str, Any]],
    consensus: Mapping[str, Any],
) -> JsonDict:
    grouped = group_by_case(records)
    unique_clusters = [len({str(row["cluster"]) for row in rows}) for rows in grouped.values()]
    cluster_shares = []
    for rows in grouped.values():
        counts = Counter(str(row["cluster"]) for row in rows)
        cluster_shares.append(max(counts.values()) / len(rows))
    candidate_interval = clustered_interval(
        [sum(bool(row["terminal_passed"]) for row in rows) / len(rows) for rows in grouped.values()]
    )
    consensus_accuracy = float(consensus["all_sample_case_accuracy"])
    return {
        "schema": SCHEMA + ".saturation_error_diversity",
        "candidate_accuracy_upper_ci": candidate_interval["upper"],
        "tuned_consensus_accuracy": consensus_accuracy,
        "unsaturation_gate_passed": candidate_interval["upper"] < 0.85
        and consensus_accuracy < 0.90,
        "mean_unique_normalized_clusters_per_case": round(
            sum(unique_clusters) / len(unique_clusters), 6
        ),
        "mean_max_cluster_share_per_case": round(sum(cluster_shares) / len(cluster_shares), 6),
        "cases_with_more_than_one_cluster": sum(value > 1 for value in unique_clusters),
        "cases_with_parseable_error_diversity": sum(
            len({str(row["cluster"]) for row in rows if row["parseable"]}) > 1
            for rows in grouped.values()
        ),
        "dominant_failure_surface": _dominant_failure_surface(records),
    }


def oracle_at_8(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped = group_by_case(records)
    values = [
        1.0 if any(bool(row["terminal_passed"]) for row in rows) else 0.0
        for rows in grouped.values()
    ]
    return {
        "schema": SCHEMA + ".oracle_at_8",
        "k": K_SAMPLES,
        "accuracy": round(sum(values) / len(values), 6),
        "clustered_interval": clustered_interval(values),
        "by_split": _case_metric_by_split(
            records, lambda rows: any(bool(row["terminal_passed"]) for row in rows)
        ),
    }


def oracle_minus_consensus(
    records: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> JsonDict:
    grouped = group_by_case(records)
    deltas = []
    for rows in grouped.values():
        oracle = 1.0 if any(bool(row["terminal_passed"]) for row in rows) else 0.0
        consensus = 1.0 if selected_cluster_correct(rows, rule) else 0.0
        deltas.append(oracle - consensus)
    interval = clustered_interval(deltas)
    return {
        "schema": SCHEMA + ".oracle_minus_consensus",
        "delta": round(sum(deltas) / len(deltas), 6),
        "minimum_required_delta": 0.10,
        "clustered_interval": interval,
        "lower_ci_excludes_zero": interval["lower"] > 0.0,
        "by_split": _delta_by_split(records, rule),
    }


def consensus_wrong_oracle_right_group_count(
    records: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> JsonDict:
    grouped = group_by_case(records)
    by_split: Counter[str] = Counter()
    count = 0
    for case_id, rows in grouped.items():
        oracle = any(bool(row["terminal_passed"]) for row in rows)
        consensus = selected_cluster_correct(rows, rule)
        if oracle and not consensus:
            count += 1
            by_split[str(rows[0]["split"])] += 1
        _ = case_id
    return {
        "schema": SCHEMA + ".minority_groups",
        "count": count,
        "minimum_required": 30,
        "passed": count >= 30,
        "by_split": dict(sorted(by_split.items())),
    }


def family_constraint_count_and_violation_strata(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["family"]),
                str(record["primary_constraint"]),
                int(record["constraint_count"]),
            )
        ].append(record)
    strata = []
    for (family, primary, constraint_count), rows in sorted(grouped.items()):
        violations = Counter(
            category for row in rows for category in row.get("violation_categories", [])
        )
        strata.append(
            {
                "family": family,
                "primary_constraint": primary,
                "constraint_count": constraint_count,
                "sample_count": len(rows),
                "case_count": len({row["case_id"] for row in rows}),
                "parseable_count": sum(bool(row["parseable"]) for row in rows),
                "terminal_pass_count": sum(bool(row["terminal_passed"]) for row in rows),
                "violation_taxonomy": dict(sorted(violations.items())),
            }
        )
    return {
        "schema": SCHEMA + ".family_constraint_violation_strata",
        "strata": strata,
        "family_count": len({str(row["family"]) for row in records}),
        "primary_constraint_count": len({str(row["primary_constraint"]) for row in records}),
        "all_samples_retained": True,
    }


def exact_floor_definition_value_and_provenance(preregistration: Mapping[str, Any]) -> JsonDict:
    floor = dict(preregistration.get("exact_floor_definition_and_provenance", {}))
    return {
        "schema": SCHEMA + ".exact_floor",
        "floor_name": floor.get("floor_name", "exact_random_executable_plan_floor"),
        "value": float(floor.get("floor_upper_bound", 0.05)),
        "provenance": floor.get("provenance"),
        "source_artifact": "results/experiment_6173_cctu_item_bank_preregistration.json",
        "finite_choice_floor_used": bool(floor.get("finite_choice_floor_used", False)),
        "answer_position_floor_used": bool(floor.get("answer_position_floor_used", False)),
    }


def duplicate_and_shortcut_audits(
    raw_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        by_case[str(raw["case_id"])].append(raw)
    duplicate_count = sum(
        len(rows) - len({str(row.get("raw_completion_sha256")) for row in rows})
        for rows in by_case.values()
    )
    raw_keys = {key for row in raw_rows for key in row}
    return {
        "schema": SCHEMA + ".duplicate_shortcut_audits",
        "duplicate_raw_completion_count": duplicate_count,
        "duplicate_case_cluster_count": sum(
            len(rows) != len({str(row.get("raw_completion_sha256")) for row in rows})
            for rows in by_case.values()
        ),
        "answer_position_channel_detected": any("answer_position" in key for key in raw_keys),
        "hidden_state_fields_detected": bool(
            {"hidden_state", "hidden_states", "activations"} & raw_keys
        ),
        "arbitrary_id_used_by_consensus": False,
        "sample_index_used_by_consensus": False,
        "max_duplicate_share_per_case": _max_duplicate_share(by_case),
        "top_normalized_cluster_hashes": _top_cluster_hashes(records),
    }


def held_aggregate_qualification(
    records: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> JsonDict:
    held_records = [record for record in records if record["split"] == "held"]
    grouped = group_by_case(held_records)
    oracle_values = [any(bool(row["terminal_passed"]) for row in rows) for rows in grouped.values()]
    consensus_values = [selected_cluster_correct(rows, rule) for rows in grouped.values()]
    candidate_pass = sum(bool(row["terminal_passed"]) for row in held_records)
    aggregate = {
        "schema": SCHEMA + ".held_aggregate",
        "held_rows_exposed": False,
        "held_case_count": len(grouped),
        "held_sample_count": len(held_records),
        "parseable_count": sum(bool(row["parseable"]) for row in held_records),
        "candidate_accuracy": rate(candidate_pass, len(held_records)),
        "oracle_at_8_accuracy": rate(sum(oracle_values), len(oracle_values)),
        "tuned_consensus_accuracy": rate(sum(consensus_values), len(consensus_values)),
        "consensus_wrong_oracle_right_count": sum(
            bool(oracle) and not bool(consensus)
            for oracle, consensus in zip(oracle_values, consensus_values, strict=True)
        ),
        "sealed_row_label_hash": sha256_file(paths["held_label"]),
        "sealed_row_label_hash_scope": "held label sidecar bytes only; no row labels emitted",
    }
    aggregate["aggregate_signature_sha256"] = sha256_json(aggregate)
    return aggregate


def preconditions_checked(
    *,
    paths: Mapping[str, Path],
    exp6174_artifact: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    revalidation: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    raw_receipt = exp6174_artifact.get("raw_before_label_commit_receipts", {})
    no_retry = exp6174_artifact.get("no_correctness_conditioned_retry_or_replacement_receipt", {})
    raw_declared = exp6174_artifact.get("raw_trace_corpus_path_hash_count_and_schema", {})
    labels_declared = exp6174_artifact.get("exact_label_sidecar_paths_hashes_and_counts", {})
    access_logs = exp6174_artifact.get("calibration_and_held_access_logs", {})
    output_paths = exp6174_artifact.get("preconditions_checked", {}).get("output_paths", {})
    per_case = Counter(str(row["case_id"]) for row in raw_rows)
    label_counts_by_split = Counter(str(row.get("split")) for row in label_rows)
    current_raw_sha256 = sha256_file(paths["raw_trace"])
    current_calibration_label_sha256 = sha256_file(paths["calibration_label"])
    current_held_label_sha256 = sha256_file(paths["held_label"])
    checks = {
        "exp6174_status_complete_ready": exp6174_artifact.get("status") == "complete_ready",
        "structured_gate_passed": bool(
            exp6174_artifact.get("structured_gate_receipt", {}).get("passed")
        ),
        "upstream_hashes_match": bool(upstream.get("all_current_hashes_match_exp6174")),
        "raw_trace_hash_matches_exp6174": raw_declared.get("sha256") == current_raw_sha256
        and raw_receipt.get("raw_corpus_sha256") == current_raw_sha256
        and raw_declared.get("count") == len(raw_rows),
        "label_sidecar_hashes_match": (
            labels_declared.get("calibration", {}).get("sha256")
            == current_calibration_label_sha256
            and labels_declared.get("held", {}).get("sha256") == current_held_label_sha256
            and labels_declared.get("calibration", {}).get("count")
            == label_counts_by_split["calibration"]
            and labels_declared.get("held", {}).get("count") == label_counts_by_split["held"]
        ),
        "raw_before_label_receipt": raw_receipt.get("validation_started_after_raw_commit") is True
        and raw_receipt.get("raw_rows_complete_before_validation") is True,
        "exact_validator_version": revalidation.get("validator_version_mismatch_count") == 0,
        "calibration_and_held_seals": (
            labels_declared.get("labels_inaccessible_to_generation") is True
            and access_logs.get("held_aggregate_outcomes_inspected") is False
            and access_logs.get("calibration", {}).get("exists") is True
            and access_logs.get("held", {}).get("exists") is True
        ),
        "calibration_and_held_label_counts": len(label_rows) == len(raw_rows) == 120 * K_SAMPLES,
        "k_completeness": len(per_case) == 120 and min(per_case.values(), default=0) >= K_SAMPLES,
        "no_retry_receipt": no_retry
        == {
            "correctness_conditioned_retry_count": 0,
            "parser_repair_count": 0,
            "model_judge_count": 0,
            "candidate_replacement_count": 0,
            "preserved_all_raw_rows": True,
        },
        "preregistered_gates_and_power_present": bool(
            read_json(paths["preregistration"]).get(
                "parseability_competence_unsaturation_headroom_and_minority_gates"
            )
        )
        and bool(read_json(paths["preregistration"]).get("clustered_inference_and_power_plan")),
        "output_paths_declared": all(
            output_paths.get(name)
            for name in ("raw_trace", "calibration_label", "held_label", "result")
        ),
        "output_path_parent_writable": _parent_writable(paths["result"]),
        "exclusion_manifest_present": paths["exclusion_manifest"].is_file(),
        "protected_files_present": all((REPO_ROOT / path).is_file() for path in PROTECTED_FILES),
    }
    blocked = [name for name, passed in checks.items() if not passed]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "passed": not blocked,
        "blocked_reasons": blocked,
        "checks": checks,
        "raw_trace_path": str(paths["raw_trace"]),
        "result_path": str(paths["result"]),
        "audit_source_hashes": [
            file_receipt(SPEC_RELATIVE_PATH),
            file_receipt(MODULE_RELATIVE_PATH),
            file_receipt(TEST_RELATIVE_PATH),
        ],
        "protected_file_hashes_before": dict(protected_before),
        "held_labels_sealed_during_tuning": True,
        "selector_or_hidden_state_extraction_attempted": False,
    }


def upstream_hash_receipt(exp6174_artifact: Mapping[str, Any]) -> JsonDict:
    declared = exp6174_artifact.get("upstream_bank_split_validator_and_preregistration_hashes", {})
    current = {
        "preregistration": file_receipt(
            Path("results/experiment_6173_cctu_item_bank_preregistration.json")
        ),
        "item_bank": file_receipt(Path("data/research/cctu_item_bank_6173.jsonl")),
        "split": file_receipt(Path("data/research/cctu_item_bank_6173_splits.json")),
        "held_access_log": file_receipt(
            Path("data/research/cctu_item_bank_6173_held_access_log.json")
        ),
        "validator": file_receipt(Path("python/carnot/verify/cctu_item_bank_6173.py")),
    }
    comparisons = {}
    for key, receipt in current.items():
        comparisons[key] = {
            "declared_sha256": declared.get(key, {}).get("sha256"),
            "current_sha256": receipt.get("sha256"),
            "matches": declared.get(key, {}).get("sha256") == receipt.get("sha256"),
        }
    return {
        "schema": SCHEMA + ".upstream_hashes",
        "declared_by_exp6174": declared,
        "current": current,
        "comparisons": comparisons,
        "all_current_hashes_match_exp6174": all(row["matches"] for row in comparisons.values()),
    }


def gate_matrix_for_readiness(
    *,
    preconditions: Mapping[str, Any],
    denominators: Mapping[str, Any],
    competence: Mapping[str, Any],
    saturation: Mapping[str, Any],
    delta_metrics: Mapping[str, Any],
    minority: Mapping[str, Any],
    family_strata: Mapping[str, Any],
    revalidation: Mapping[str, Any],
) -> JsonDict:
    parseability = float(denominators["parseable_samples"]["parseability_rate"] or 0.0)
    family_support = all(stratum["parseable_count"] > 0 for stratum in family_strata["strata"])
    conjuncts = {
        "preconditions": {
            "passed": bool(preconditions["passed"]),
            "value": preconditions["passed"],
        },
        "label_revalidation": {
            "passed": bool(revalidation["all_labels_match_revalidation"]),
            "value": revalidation["all_labels_match_revalidation"],
        },
        "parseability": {"passed": parseability >= 0.95, "value": parseability, "threshold": 0.95},
        "competence": {
            "passed": bool(competence["above_exact_floor_gate_passed"]),
            "value": competence["clustered_interval"]["lower"],
            "threshold": f"> {competence['exact_floor']}",
        },
        "unsaturation": {
            "passed": bool(saturation["unsaturation_gate_passed"]),
            "value": {
                "candidate_upper_ci": saturation["candidate_accuracy_upper_ci"],
                "consensus_accuracy": saturation["tuned_consensus_accuracy"],
            },
            "threshold": "candidate upper CI < 0.85 and consensus < 0.90",
        },
        "headroom": {
            "passed": bool(
                delta_metrics["delta"] >= 0.10 and delta_metrics["lower_ci_excludes_zero"]
            ),
            "value": {
                "delta": delta_metrics["delta"],
                "lower_ci": delta_metrics["clustered_interval"]["lower"],
            },
            "threshold": "delta >= 0.10 and lower CI > 0",
        },
        "minority": {
            "passed": bool(minority["passed"]),
            "value": minority["count"],
            "threshold": minority["minimum_required"],
        },
        "family_support": {
            "passed": family_support,
            "value": {
                "families_with_parseable_candidates": sum(
                    stratum["parseable_count"] > 0 for stratum in family_strata["strata"]
                ),
                "family_strata": len(family_strata["strata"]),
            },
            "threshold": "each family/constraint stratum has parseable support",
        },
    }
    failed = [name for name, row in conjuncts.items() if not row["passed"]]
    return {
        "schema": SCHEMA + ".gate_matrix",
        "conjuncts": conjuncts,
        "failed_conjuncts": failed,
        "all_conjuncts_passed": not failed,
        "ready_score_principle": "one only when every preregistered conjunct passes; no weighted averaging",
    }


def validate_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors = []
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("bad_inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_not_true")
    ready = artifact.get("phase_d_headroom_ready_score")
    future = artifact.get("future_rows_allowed_by_this_artifact")
    if ready == 1.0 and future is not True:
        errors.append("ready_without_future_rows")
    if ready != 1.0 and future is not False:
        errors.append("nonready_allows_future_rows")
    if ready != 1.0 and artifact.get("status") == "complete_ready":
        errors.append("complete_ready_without_ready_score_one")
    held = json.dumps(artifact.get("held_aggregate_qualification_and_row_label_seal_hash", {}))
    if any(token in held for token in ("sample_key", "raw_row_hash", "validator_result")):
        errors.append("held_rows_exposed")
    return {"schema": SCHEMA + ".schema_validation", "ok": not errors, "errors": errors}


def selected_cluster_correct(rows: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> bool:
    cluster = select_consensus_cluster(rows, rule)
    if cluster is None:
        return False
    return any(bool(row["terminal_passed"]) for row in rows if row["cluster"] == cluster)


def select_consensus_cluster(
    rows: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> str | None:
    candidates = list(rows)
    if rule.get("parseable_only"):
        parseable = [row for row in rows if row["parseable"]]
        if parseable:
            candidates = parseable
    counts = Counter(str(row["cluster"]) for row in candidates)
    if not counts:
        return None
    cluster, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    if count < int(rule.get("min_agreement", 1)):
        return None
    return cluster


def clustered_interval(
    cluster_values: Sequence[float],
    *,
    seed: int = 6175,
    resamples: int = 10000,
) -> JsonDict:
    values = [float(value) for value in cluster_values]
    if not values:
        return {"estimate": None, "lower": None, "upper": None, "method": "cluster_bootstrap"}
    estimate = sum(values) / len(values)
    if len(set(values)) == 1:
        rounded = round(estimate, 6)
        return {
            "estimate": rounded,
            "lower": rounded,
            "upper": rounded,
            "method": "constant_cluster",
        }
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lower = boot[int(0.025 * (len(boot) - 1))]
    upper = boot[int(0.975 * (len(boot) - 1))]
    return {
        "estimate": round(estimate, 6),
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "method": "deterministic_case_cluster_bootstrap",
        "resamples": resamples,
        "seed": seed,
    }


def group_by_case(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(record)
    return dict(grouped)


def rate(numerator: int | float | bool, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / denominator, 6)


def raw_row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_pretty(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_receipt(relative: Path) -> JsonDict:
    path = REPO_ROOT / relative
    return {
        "path": relative.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def protected_file_hash_map() -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(REPO_ROOT / relative)
        for relative in PROTECTED_FILES
        if (REPO_ROOT / relative).is_file()
    }


def protected_files_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {
        "schema": SCHEMA + ".protected_files",
        "unchanged": not changed,
        "changed_paths": changed,
        "before": dict(before),
        "after": dict(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
        "ops_status_changelog_traceability_untouched": not (
            {"ops/changelog.md", "ops/status.md", "_bmad/traceability.md"} & set(changed)
        ),
    }


def field_provenance() -> JsonDict:
    return {
        field: ["REQ-CONSTRAINT-VERIFY-6175", *FIELD_REQUIREMENTS[field], FIELD_PRINCIPLES[field]]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def honest_verdict(status: str, gate_matrix: Mapping[str, Any]) -> str:
    failed = ", ".join(gate_matrix.get("failed_conjuncts", []))
    if status == "complete_ready":
        return "complete_ready: Exp6175 all preregistered headroom conjuncts passed"
    if status == "complete_null":
        return "complete_null: Exp6175 measured no selector headroom"
    if status == "blocked":
        return f"blocked: Exp6175 preconditions failed; failed_conjuncts={failed}"
    return f"retired: Exp6174 CCTU pool failed preregistered headroom conjuncts: {failed}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.validate:
        path = args.output or REPO_ROOT / RESULT_RELATIVE_PATH
        artifact = read_json(path) if path.exists() else run(result_path=path)
        validation = validate_artifact(artifact)
        print(canonical_json({"ok": validation["ok"], "errors": validation["errors"]}))
        return 0 if validation["ok"] else 1
    artifact = run(result_path=args.output)
    print(
        canonical_json(
            {
                "artifact": str(default_paths(result_path=args.output)["result"]),
                "status": artifact["status"],
            }
        )
    )
    return 0


def _consensus_accuracy(records: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> float:
    grouped = group_by_case(records)
    if not grouped:
        return 0.0
    correct = sum(selected_cluster_correct(rows, rule) for rows in grouped.values())
    return round(correct / len(grouped), 6)


def _parse_json_ok(validation: Mapping[str, Any]) -> bool:
    return any(
        step.get("step_id") == "parse_json" and bool(step.get("passed"))
        for step in validation.get("step_results", [])
    )


def _counter_mapping(value: Mapping[str, Counter[str]]) -> JsonDict:
    return {key: dict(sorted(counter.items())) for key, counter in sorted(value.items())}


def _step_satisfaction(step_pass_counts: Mapping[str, Counter[str]]) -> JsonDict:
    rows = {}
    for category, counts in sorted(step_pass_counts.items()):
        rows[category] = {
            "passed": counts["passed"],
            "total": counts["total"],
            "rate": rate(counts["passed"], counts["total"]),
        }
    return rows


def _case_metric_by_split(
    records: Sequence[Mapping[str, Any]],
    fn: Callable[[Sequence[Mapping[str, Any]]], bool],
) -> JsonDict:
    grouped = group_by_case(records)
    by_split: dict[str, list[float]] = defaultdict(list)
    for rows in grouped.values():
        by_split[str(rows[0]["split"])].append(1.0 if fn(rows) else 0.0)
    return {
        split: {
            "case_count": len(values),
            "accuracy": round(sum(values) / len(values), 6) if values else None,
            "clustered_interval": clustered_interval(values),
        }
        for split, values in sorted(by_split.items())
    }


def _delta_by_split(records: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> JsonDict:
    grouped = group_by_case(records)
    by_split: dict[str, list[float]] = defaultdict(list)
    for rows in grouped.values():
        oracle = 1.0 if any(bool(row["terminal_passed"]) for row in rows) else 0.0
        consensus = 1.0 if selected_cluster_correct(rows, rule) else 0.0
        by_split[str(rows[0]["split"])].append(oracle - consensus)
    return {
        split: {
            "case_count": len(values),
            "delta": round(sum(values) / len(values), 6) if values else None,
            "clustered_interval": clustered_interval(values),
        }
        for split, values in sorted(by_split.items())
    }


def _dominant_failure_surface(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = Counter()
    for record in records:
        if record["truncated"]:
            counts["truncation"] += 1
        if not record["parseable"]:
            counts["parse_failure"] += 1
        if record["timeout"]:
            counts["timeout"] += 1
        if record["refusal"]:
            counts["refusal"] += 1
    name, count = counts.most_common(1)[0] if counts else ("none", 0)
    return {"name": name, "count": count, "share": rate(count, len(records))}


def _max_duplicate_share(by_case: Mapping[str, Sequence[Mapping[str, Any]]]) -> float:
    shares = []
    for rows in by_case.values():
        counts = Counter(str(row.get("raw_completion_sha256")) for row in rows)
        shares.append(max(counts.values()) / len(rows))
    return round(max(shares), 6) if shares else 0.0


def _top_cluster_hashes(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    counts = Counter(str(record["cluster"]) for record in records)
    return [
        {"cluster_hash": sha256_text(cluster), "count": count}
        for cluster, count in counts.most_common(5)
    ]


def _parent_writable(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / ".exp6175-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


if __name__ == "__main__":  # pragma: no cover - exercised through main() tests.
    raise SystemExit(main())
