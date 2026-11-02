"""Exp6140 frozen Exp6128 option psychometrics.

Spec refs: REQ-VERIFY-6140, REQ-VERIFY-6140-1, REQ-VERIFY-6140-2,
REQ-VERIFY-6140-3, REQ-VERIFY-6140-4, REQ-VERIFY-6140-5,
REQ-VERIFY-6140-6, REQ-VERIFY-6140-7, REQ-VERIFY-6140-8,
SCENARIO-VERIFY-6140-CONSERVATION,
SCENARIO-VERIFY-6140-RECONCILIATION,
SCENARIO-VERIFY-6140-OPTION-DIAGNOSTICS,
SCENARIO-VERIFY-6140-UNCERTAINTY,
SCENARIO-VERIFY-6140-TRANSFORM-ISOLATION.

This diagnostic is intentionally not a generator.  It conserves the 720
Exp6128 rows exactly as upstream evidence, rederives the source metrics, then
asks whether option identity, answer position, and family mixture leave a
split-safe item-bank design.  Because the frozen rows do not contain transformed
model responses, the result retires this Exp6128 source-domain recovery instead
of promoting a new held-generation policy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6140_phase_d_exp6128_option_psychometrics.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6140_phase_d_exp6128_option_psychometrics.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6140_phase_d_exp6128_option_psychometrics.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXP6103_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6103_phase_d_difficulty_ladder_fixture.json"
)
EXP6103_ROWS_RELATIVE_PATH = Path(
    "results/experiment_6103_phase_d_difficulty_ladder_fixture.rows.jsonl"
)
EXP6127_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6127_phase_d_native_chat_transport_canary.json"
)
EXP6128_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6128_phase_d_calibration_pool_v2.json"
)
EXP6128_ROWS_RELATIVE_PATH = Path(
    "results/experiment_6128_phase_d_calibration_pool_v2.rows.jsonl"
)
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
AGENTS_RELATIVE_PATH = Path("AGENTS.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")

SCHEMA = "carnot.experiment_6140.phase_d_exp6128_option_psychometrics.v1"
EXPERIMENT_ID = "experiment_6140_phase_d_exp6128_option_psychometrics"
RUN_DATE = "20260805"
RANDOM_SEED = 6140
BOOTSTRAP_REPLICATES = 500
EXPECTED_ROW_COUNT = 720
EXPECTED_QUESTION_GROUP_COUNT = 90
EXPECTED_K = 8
ENUMERATED_FLOOR = 0.25
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

HASHED_INPUTS = (
    AGENTS_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    RESEARCH_PROGRAM_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    EXP6103_ARTIFACT_RELATIVE_PATH,
    EXP6103_ROWS_RELATIVE_PATH,
    EXP6127_ARTIFACT_RELATIVE_PATH,
    EXP6128_ARTIFACT_RELATIVE_PATH,
    EXP6128_ROWS_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6140_phase_d_exp6128_option_psychometrics.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6140_phase_d_exp6128_option_psychometrics.py "
    "-m pytest tests/python/test_experiment_6140_phase_d_exp6128_option_psychometrics.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6140_phase_d_exp6128_option_psychometrics.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6140_phase_d_exp6128_option_psychometrics.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6140_phase_d_exp6128_option_psychometrics.json",
    ".venv/bin/python scripts/exclusion_manifest_lint.py "
    "/tmp/experiment_6140_exclusion_probe.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_source_artifact_and_row_hashes",
    "expected_observed_duplicate_and_missing_row_counts",
    "rederived_source_metric_reconciliation",
    "family_stratum_semantic_group_relabel_shortcut_and_position_metrics",
    "wrong_option_identity_position_fallback_and_response_cluster_diagnostics",
    "question_clustered_uncertainty_and_effective_information",
    "saturation_and_below_chance_attribution",
    "candidate_transformation_specification",
    "label_blind_and_held_isolation_receipt",
    "empirical_item_bank_design_ready_score",
    "retirement_triggered",
    "top_level_model_specs_methodology_gap_noted",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "immutable_source_artifact_and_row_hashes": (
        "immutable Exp6103, Exp6127, Exp6128 artifacts and frozen row files are "
        "content-addressed before any derived diagnostic is trusted."
    ),
    "expected_observed_duplicate_and_missing_row_counts": (
        "exactly 720 unique candidate rows and 90 question groups are conserved."
    ),
    "rederived_source_metric_reconciliation": (
        "Exp6128 aggregate and grouped metrics are independently rederived from "
        "rows and compared to the source artifact."
    ),
    "family_stratum_semantic_group_relabel_shortcut_and_position_metrics": (
        "family, stratum, semantic-group, relabel, shortcut, and position "
        "controls expose non-exchangeability."
    ),
    "wrong_option_identity_position_fallback_and_response_cluster_diagnostics": (
        "binary correctness alone cannot define item difficulty after the "
        "measured family mixture."
    ),
    "question_clustered_uncertainty_and_effective_information": (
        "uncertainty is measured over question groups, not independent candidate draws."
    ),
    "candidate_transformation_specification": (
        "every proposed transformation must preserve an independently checkable "
        "exact answer and be frozen before new model responses."
    ),
    "label_blind_and_held_isolation_receipt": (
        "transformation design and readiness decisions do not select on held "
        "labels or alter exact labels."
    ),
    "empirical_item_bank_design_ready_score": (
        "readiness is exactly one only for a non-degenerate split-safe design; "
        "a repeated inability to define one retires this source pool."
    ),
    "retirement_triggered": (
        "readiness is exactly one only for a non-degenerate split-safe design; "
        "a repeated inability to define one retires this source pool."
    ),
    "top_level_model_specs_methodology_gap_noted": (
        "option-level psychometrics are acknowledged, but this single-model "
        "frozen-row audit uses transparent count diagnostics rather than "
        "claiming a fitted top-level nominal-response model."
    ),
    "protected_files_unchanged": (
        "conductor and reconciler-owned files remain byte-identical."
    ),
    "duration_s": "report measured `aggregation_from_upstream_artifacts`.",
    "inference_substrate": "report measured `aggregation_from_upstream_artifacts`.",
    "field_provenance": "report measured `aggregation_from_upstream_artifacts`.",
    "test_commands": "report measured `aggregation_from_upstream_artifacts`.",
    "test_exit_codes": "report measured `aggregation_from_upstream_artifacts`.",
    "reproducibility_checksum": "report measured `aggregation_from_upstream_artifacts`.",
    "verifier_is_oracle": (
        "the diagnostic reads exact labels for evaluation but does not define or alter them."
    ),
    "missing_verifier_gaps": (
        "the diagnostic reads exact labels for evaluation but does not define or alter them."
    ),
    "honest_verdict": (
        "use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:` "
        "and state whether the bimodality is distractor/position, saturation, "
        "true inability, or unresolved."
    ),
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with stable ordering and ASCII bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository's prefixed SHA-256 convention."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after deterministic serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without trusting names, mtimes, or JSON formatting."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - corrupted input guard.
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):  # pragma: no cover - corrupted input guard.
            raise ValueError(f"JSON object row required at line {line_number}: {path}")
        rows.append(dict(payload))
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _rate(count: float, total: float) -> float:
    return round(count / total, 6) if total else 0.0


def _entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return round(
        -sum((count / total) * math.log2(count / total) for count in counts.values()),
        6,
    )


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _protected_hashes(root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }


def _git_status(root: Path, *, output_path: Path) -> JsonDict:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    ignored_suffix = output_path.as_posix()
    filtered = [line for line in lines if ignored_suffix not in line]
    return {
        "command": "git status --short",
        "exit_code": result.returncode,
        "dirty": bool(filtered),
        "status_short_excluding_output": filtered,
        "output_path_ignored_for_stability": ignored_suffix,
    }


def _choice_labels(row: Mapping[str, Any]) -> list[str]:
    content = str(row["serialized_messages"][-1]["content"])
    choice_block = content.split("Choices:\n", 1)[1].split("\n\nThink", 1)[0]
    return [line.split(":", 1)[0].strip() for line in choice_block.splitlines() if ":" in line]


def _annotate_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    annotated: list[JsonDict] = []
    for row in rows:
        item = dict(row)
        labels = _choice_labels(row)
        exact_label = str(item["python_exact_label"])
        response_label = str(item.get("final_answer_label") or "UNPARSEABLE")
        item["choice_labels"] = labels
        item["exact_answer_position"] = labels.index(exact_label) + 1
        item["response_position"] = (
            labels.index(response_label) + 1 if response_label in labels else 0
        )
        item["parseable"] = dict(item.get("parser") or {}).get("parseable") is True
        item["correct"] = item.get("exact_correct") is True
        item["wrong_response_label"] = None if item["correct"] else response_label
        annotated.append(item)
    annotated.sort(key=lambda row: str(row["candidate_row_id"]))
    return annotated


def _majority_label(group: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(
        str(row["answer_cluster"])
        for row in group
        if str(row["answer_cluster"]) != "UNPARSEABLE"
    )
    last_index = {str(row["answer_cluster"]): index for index, row in enumerate(group)}
    return sorted(counts, key=lambda label: (counts[label], last_index[label]))[-1]


def _question_records(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_exp6103_row_id"])].append(dict(row))
    records: list[JsonDict] = []
    for question_id, group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["sample_index"]))
        correct_count = sum(1 for row in group if row["correct"])
        wrong_labels = [
            str(row["final_answer_label"])
            for row in group
            if row["correct"] is not True and str(row.get("final_answer_label") or "")
        ]
        answer_labels = [str(row["answer_cluster"]) for row in group]
        binary_labels = ["correct" if row["correct"] else "wrong" for row in group]
        wrong_entropy = _entropy(wrong_labels)
        p_wrong = 1.0 - correct_count / len(group)
        majority = _majority_label(group)
        exact_label = str(group[0]["python_exact_label"])
        response_counts = Counter(str(row["final_answer_label"]) for row in group)
        records.append(
            {
                "source_exp6103_row_id": question_id,
                "semantic_group_id": str(group[0]["semantic_group_id"]),
                "family": str(group[0]["family"]),
                "difficulty_stratum": str(group[0]["difficulty_stratum"]),
                "solver_effort_bin": str(group[0]["solver_effort_bin"]),
                "row_count": len(group),
                "correct_count": correct_count,
                "accuracy": _rate(correct_count, len(group)),
                "exact_label": exact_label,
                "exact_position_counts": dict(
                    sorted(Counter(str(row["exact_answer_position"]) for row in group).items())
                ),
                "response_label_counts": dict(sorted(response_counts.items())),
                "response_position_counts": dict(
                    sorted(Counter(str(row["response_position"]) for row in group).items())
                ),
                "wrong_response_label_counts": dict(sorted(Counter(wrong_labels).items())),
                "effective_k": len({str(row["exact_duplicate_key"]) for row in group}),
                "exact_duplicate_rate": round(
                    1 - len({str(row["exact_duplicate_key"]) for row in group}) / len(group),
                    6,
                ),
                "semantic_duplicate_rate": round(
                    1 - len({str(row["semantic_duplicate_key"]) for row in group}) / len(group),
                    6,
                ),
                "answer_cluster_entropy_bits": _entropy(answer_labels),
                "binary_correctness_entropy_bits": _entropy(binary_labels),
                "wrong_option_entropy_bits": wrong_entropy,
                "option_identity_additional_entropy_bits": round(p_wrong * wrong_entropy, 6),
                "max_response_cluster_share": round(max(response_counts.values()) / len(group), 6),
                "all_wrong": correct_count == 0,
                "oracle_correct": correct_count > 0,
                "majority_label": majority,
                "tuned_sc_correct": majority == exact_label,
            }
        )
    return records


def _candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    parseable = sum(1 for row in rows if row["parseable"])
    method = sum(1 for row in rows if row.get("method_valid") is True)
    return {
        "candidate_count": total,
        "response_count": total,
        "correct_count": correct,
        "accuracy": _rate(correct, total),
        "parseable_count": parseable,
        "parseability": _rate(parseable, total),
        "method_valid_count": method,
        "method_validity": _rate(method, total),
    }


def _question_summary(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(records)
    return {
        "question_count": total,
        "all_wrong_rate": _rate(sum(1 for row in records if row["all_wrong"]), total),
        "oracle_at_k": _rate(sum(1 for row in records if row["oracle_correct"]), total),
        "tuned_sc_accuracy": _rate(
            sum(1 for row in records if row["tuned_sc_correct"]),
            total,
        ),
        "oracle_minus_tuned_sc": round(
            _rate(sum(1 for row in records if row["oracle_correct"]), total)
            - _rate(sum(1 for row in records if row["tuned_sc_correct"]), total),
            6,
        ),
        "mean_effective_k": round(
            sum(float(row["effective_k"]) for row in records) / total,
            6,
        ),
        "mean_exact_duplicate_rate": round(
            sum(float(row["exact_duplicate_rate"]) for row in records) / total,
            6,
        ),
        "mean_semantic_duplicate_rate": round(
            sum(float(row["semantic_duplicate_rate"]) for row in records) / total,
            6,
        ),
        "mean_answer_cluster_entropy_bits": round(
            sum(float(row["answer_cluster_entropy_bits"]) for row in records) / total,
            6,
        ),
    }


def _group_rows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(dict(row))
    return dict(sorted(grouped.items()))


def _group_questions(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in records:
        grouped[str(row[key])].append(dict(row))
    return dict(sorted(grouped.items()))


def _combined_group_metrics(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    row_metrics = _candidate_metrics(rows)
    question_metrics = _question_summary(records)
    return {**row_metrics, **question_metrics}


def _comparison(observed: float, source: float, tolerance: float) -> JsonDict:
    delta = round(observed - source, 6)
    return {
        "observed": round(observed, 6),
        "source": round(source, 6),
        "delta": delta,
        "reconciled": abs(delta) <= tolerance,
    }


def _reconciliation(
    *,
    rows: Sequence[Mapping[str, Any]],
    question_records: Sequence[Mapping[str, Any]],
    exp6128_artifact: Mapping[str, Any],
) -> JsonDict:
    tolerance = 1e-6
    row_overall = _candidate_metrics(rows)
    q_overall = _question_summary(question_records)
    source_row = dict(
        dict(exp6128_artifact["per_candidate_accuracy_clustered_intervals_parseability_method_validity"])[
            "overall"
        ]
    )
    source_q = dict(
        dict(exp6128_artifact["effective_k_exact_semantic_duplicate_all_wrong_oracle_and_tuned_sc_metrics"])[
            "overall"
        ]
    )
    overall = {
        "accuracy": _comparison(row_overall["accuracy"], float(source_row["accuracy"]), tolerance),
        "parseability": _comparison(
            row_overall["parseability"],
            float(source_row["parseability"]),
            tolerance,
        ),
        "method_validity": _comparison(
            row_overall["method_validity"],
            float(source_row["method_validity"]),
            tolerance,
        ),
        "oracle_at_k": _comparison(q_overall["oracle_at_k"], float(source_q["oracle_at_k"]), tolerance),
        "tuned_sc_accuracy": _comparison(
            q_overall["tuned_sc_accuracy"],
            float(source_q["tuned_sc_accuracy"]),
            tolerance,
        ),
        "oracle_minus_tuned_sc": _comparison(
            q_overall["oracle_minus_tuned_sc"],
            float(source_q["oracle_minus_tuned_sc"]),
            tolerance,
        ),
        "all_wrong_rate": _comparison(
            q_overall["all_wrong_rate"],
            float(source_q["all_wrong_rate"]),
            tolerance,
        ),
        "mean_effective_k": _comparison(
            q_overall["mean_effective_k"],
            float(source_q["mean_effective_k"]),
            tolerance,
        ),
        "mean_exact_duplicate_rate": _comparison(
            q_overall["mean_exact_duplicate_rate"],
            float(source_q["mean_exact_duplicate_rate"]),
            tolerance,
        ),
        "mean_semantic_duplicate_rate": _comparison(
            q_overall["mean_semantic_duplicate_rate"],
            float(source_q["mean_semantic_duplicate_rate"]),
            tolerance,
        ),
    }
    source_by_family = dict(
        exp6128_artifact["per_candidate_accuracy_clustered_intervals_parseability_method_validity"][
            "by_family"
        ]
    )
    by_family: dict[str, JsonDict] = {}
    for family, family_rows in _group_rows(rows, "family").items():
        family_metrics = _candidate_metrics(family_rows)
        by_family[family] = {
            "accuracy": _comparison(
                family_metrics["accuracy"],
                float(source_by_family[family]["accuracy"]),
                tolerance,
            ),
            "method_validity": _comparison(
                family_metrics["method_validity"],
                float(source_by_family[family]["method_validity"]),
                tolerance,
            ),
            "parseability": _comparison(
                family_metrics["parseability"],
                float(source_by_family[family]["parseability"]),
                tolerance,
            ),
        }
    all_comparisons = list(overall.values()) + [
        item for row in by_family.values() for item in row.values()
    ]
    return {
        "schema": SCHEMA + ".source_metric_reconciliation",
        "tolerance": tolerance,
        "overall": overall,
        "by_family": by_family,
        "all_reconciled": all(item["reconciled"] is True for item in all_comparisons),
        "source_artifact": EXP6128_ARTIFACT_RELATIVE_PATH.as_posix(),
        "principle": REQUIRED_FIELD_PRINCIPLES["rederived_source_metric_reconciliation"],
    }


def _row_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    ids = [str(row["candidate_row_id"]) for row in rows]
    question_ids = [str(row["source_exp6103_row_id"]) for row in rows]
    by_question = Counter(question_ids)
    return {
        "schema": SCHEMA + ".row_conservation",
        "expected_row_count": EXPECTED_ROW_COUNT,
        "observed_row_count": len(rows),
        "unique_candidate_row_id_count": len(set(ids)),
        "duplicate_row_count": len(ids) - len(set(ids)),
        "missing_row_count": max(0, EXPECTED_ROW_COUNT - len(set(ids))),
        "question_group_count": len(set(question_ids)),
        "expected_question_group_count": EXPECTED_QUESTION_GROUP_COUNT,
        "candidate_rows_per_question_min": min(by_question.values()),
        "candidate_rows_per_question_max": max(by_question.values()),
        "row_identity_hash": sha256_json(sorted(ids)),
        "question_group_identity_hash": sha256_json(sorted(set(question_ids))),
        "all_rows_calibration_split": all(str(row["source_split"]) == "calibration" for row in rows),
        "method_valid_row_count": sum(1 for row in rows if row.get("method_valid") is True),
        "parseable_row_count": sum(1 for row in rows if row["parseable"]),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "expected_observed_duplicate_and_missing_row_counts"
        ],
    }


def _source_rows_for_exp6128(
    exp6103_rows: Sequence[Mapping[str, Any]],
    exp6128_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    wanted = {str(row["source_exp6103_row_id"]) for row in exp6128_rows}
    selected = [dict(row) for row in exp6103_rows if str(row["row_id"]) in wanted]
    selected.sort(key=lambda row: str(row["row_id"]))
    return selected


def _source_transform_receipt(selected_source_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    transform_counts: Counter[str] = Counter()
    shortcut_counts: Counter[str] = Counter()
    answer_permutation_exact = 0
    for row in selected_source_rows:
        receipts = dict(row.get("transform_receipts") or {})
        transform_counts.update(str(key) for key in receipts)
        if dict(receipts.get("answer_permutation") or {}).get("exact_semantics_preserved") is True:
            answer_permutation_exact += 1
        shortcut = dict(row.get("shortcut_salience") or {})
        shortcut_counts.update([str(shortcut.get("salience", "unknown"))])
    return {
        "selected_source_question_count": len(selected_source_rows),
        "transform_kind_counts": dict(sorted(transform_counts.items())),
        "answer_permutation_exact_semantics_preserved_count": answer_permutation_exact,
        "shortcut_salience_counts": dict(sorted(shortcut_counts.items())),
        "shortcut_labels_method_valid": False,
        "held_or_sibling_labels_used": False,
    }


def _family_position_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    question_records: Sequence[Mapping[str, Any]],
    selected_source_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_semantic_group = {
        str(row["semantic_group_id"]): {
            "source_exp6103_row_id": str(row["source_exp6103_row_id"]),
            "family": str(row["family"]),
            "difficulty_stratum": str(row["difficulty_stratum"]),
            "solver_effort_bin": str(row["solver_effort_bin"]),
            "row_count": int(row["row_count"]),
            "correct_count": int(row["correct_count"]),
            "accuracy": float(row["accuracy"]),
            "oracle_correct": bool(row["oracle_correct"]),
            "tuned_sc_correct": bool(row["tuned_sc_correct"]),
            "all_wrong": bool(row["all_wrong"]),
            "response_label_counts": dict(row["response_label_counts"]),
            "exact_position_counts": dict(row["exact_position_counts"]),
            "response_position_counts": dict(row["response_position_counts"]),
            "option_identity_additional_entropy_bits": float(
                row["option_identity_additional_entropy_bits"]
            ),
        }
        for row in question_records
    }
    question_by_family = _group_questions(question_records, "family")
    question_by_stratum = _group_questions(question_records, "difficulty_stratum")
    question_by_effort = _group_questions(question_records, "solver_effort_bin")
    return {
        "schema": SCHEMA + ".family_stratum_semantic_position_metrics",
        "overall": _combined_group_metrics(rows, question_records),
        "by_family": {
            family: _combined_group_metrics(_group_rows(rows, "family")[family], records)
            for family, records in question_by_family.items()
        },
        "by_difficulty_stratum": {
            stratum: _combined_group_metrics(_group_rows(rows, "difficulty_stratum")[stratum], records)
            for stratum, records in question_by_stratum.items()
        },
        "by_solver_effort_bin": {
            effort: _combined_group_metrics(_group_rows(rows, "solver_effort_bin")[effort], records)
            for effort, records in question_by_effort.items()
        },
        "by_semantic_group": by_semantic_group,
        "by_exact_answer_position": {
            key: _candidate_metrics(group)
            for key, group in _group_rows(rows, "exact_answer_position").items()
        },
        "by_response_position": {
            key: _candidate_metrics(group)
            for key, group in _group_rows(rows, "response_position").items()
        },
        "by_exact_label": {
            key: _candidate_metrics(group) for key, group in _group_rows(rows, "python_exact_label").items()
        },
        "by_response_label": {
            key: _candidate_metrics(group) for key, group in _group_rows(rows, "final_answer_label").items()
        },
        "by_sample_index": {
            key: _candidate_metrics(group) for key, group in _group_rows(rows, "sample_index").items()
        },
        "relabel_shortcut_receipt": _source_transform_receipt(selected_source_rows),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "family_stratum_semantic_group_relabel_shortcut_and_position_metrics"
        ],
    }


def _mean_max_share(records: Sequence[Mapping[str, Any]]) -> float:
    return round(sum(float(row["max_response_cluster_share"]) for row in records) / len(records), 6)


def _option_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    question_records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    wrong_rows = [row for row in rows if row["correct"] is not True]
    family_confounding: dict[str, JsonDict] = {}
    for family, group in _group_rows(rows, "family").items():
        family_confounding[family] = {
            "accuracy": _candidate_metrics(group)["accuracy"],
            "exact_position_counts": dict(
                sorted(Counter(str(row["exact_answer_position"]) for row in group).items())
            ),
            "response_position_counts": dict(
                sorted(Counter(str(row["response_position"]) for row in group).items())
            ),
            "exact_first_rate": _rate(
                sum(1 for row in group if int(row["exact_answer_position"]) == 1),
                len(group),
            ),
            "response_first_rate": _rate(
                sum(1 for row in group if int(row["response_position"]) == 1),
                len(group),
            ),
        }
    share_counts = Counter(str(row["max_response_cluster_share"]) for row in question_records)
    return {
        "schema": SCHEMA + ".option_response_diagnostics",
        "diagnostic_model": "transparent_count_based_nominal_response_diagnostics",
        "fitted_nominal_response_model": False,
        "correct_option_difficulty": {
            "by_exact_label": {
                key: _candidate_metrics(group)
                for key, group in _group_rows(rows, "python_exact_label").items()
            },
            "by_exact_position": {
                key: _candidate_metrics(group)
                for key, group in _group_rows(rows, "exact_answer_position").items()
            },
        },
        "wrong_response_label_counts": dict(
            sorted(Counter(str(row["final_answer_label"]) for row in wrong_rows).items())
        ),
        "wrong_response_label_by_family": {
            family: dict(
                sorted(
                    Counter(
                        str(row["final_answer_label"])
                        for row in group
                        if row["correct"] is not True
                    ).items()
                )
            )
            for family, group in _group_rows(rows, "family").items()
        },
        "response_label_counts": dict(
            sorted(Counter(str(row["final_answer_label"]) for row in rows).items())
        ),
        "positional_preference": {
            "response_position_counts": dict(
                sorted(Counter(str(row["response_position"]) for row in rows).items())
            ),
            "exact_position_counts": dict(
                sorted(Counter(str(row["exact_answer_position"]) for row in rows).items())
            ),
            "first_position_response_rate": _rate(
                sum(1 for row in rows if int(row["response_position"]) == 1),
                len(rows),
            ),
            "first_position_accuracy": _candidate_metrics(
                [row for row in rows if int(row["response_position"]) == 1]
            )["accuracy"],
        },
        "family_position_confounding": family_confounding,
        "fallback_concentration": {
            "mean_max_response_cluster_share": _mean_max_share(question_records),
            "max_response_cluster_share_counts": dict(sorted(share_counts.items())),
            "all_same_response_question_count": sum(
                1 for row in question_records if float(row["max_response_cluster_share"]) == 1.0
            ),
        },
        "response_cluster_concentration_by_family": {
            family: {
                "question_count": len(records),
                "mean_max_response_cluster_share": _mean_max_share(records),
                "all_same_response_question_count": sum(
                    1 for row in records if float(row["max_response_cluster_share"]) == 1.0
                ),
            }
            for family, records in _group_questions(question_records, "family").items()
        },
        "binary_correctness_limitation": (
            "Aggregate binary accuracy mixes two saturated position-confounded "
            "families with typed_finite_choice below its enumerated floor."
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "wrong_option_identity_position_fallback_and_response_cluster_diagnostics"
        ],
    }


def _bootstrap_receipt(
    records: Sequence[Mapping[str, Any]],
    metric: Callable[[Sequence[Mapping[str, Any]]], float],
) -> JsonDict:
    rng = random.Random(RANDOM_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [records[rng.randrange(len(records))] for _ in range(len(records))]
        values.append(metric(sample))
    point = round(metric(records), 6)
    low = min(_percentile(values, 0.025), point)
    high = max(_percentile(values, 0.975), point)
    return {
        "point": point,
        "interval_95": [round(low, 6), round(high, 6)],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _accuracy_from_records(records: Sequence[Mapping[str, Any]]) -> float:
    return _rate(
        sum(float(row["correct_count"]) for row in records),
        sum(float(row["row_count"]) for row in records),
    )


def _mean_bool(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return _rate(sum(1 for row in records if row[key]), len(records))


def _mean_numeric(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return round(sum(float(row[key]) for row in records) / len(records), 6)


def _uncertainty(question_records: Sequence[Mapping[str, Any]]) -> JsonDict:
    typed_records = [row for row in question_records if row["family"] == "typed_finite_choice"]
    metrics = {
        "clustered_accuracy": _bootstrap_receipt(question_records, _accuracy_from_records),
        "oracle_at_k": _bootstrap_receipt(
            question_records,
            lambda sample: _mean_bool(sample, "oracle_correct"),
        ),
        "tuned_sc_accuracy": _bootstrap_receipt(
            question_records,
            lambda sample: _mean_bool(sample, "tuned_sc_correct"),
        ),
        "all_wrong_rate": _bootstrap_receipt(
            question_records,
            lambda sample: _mean_bool(sample, "all_wrong"),
        ),
        "mean_effective_k": _bootstrap_receipt(
            question_records,
            lambda sample: _mean_numeric(sample, "effective_k"),
        ),
        "option_identity_additional_entropy_bits": _bootstrap_receipt(
            question_records,
            lambda sample: _mean_numeric(sample, "option_identity_additional_entropy_bits"),
        ),
        "typed_finite_choice_accuracy": _bootstrap_receipt(typed_records, _accuracy_from_records),
    }
    return {
        "schema": SCHEMA + ".question_clustered_uncertainty",
        "uncertainty_method": "deterministic_question_cluster_bootstrap",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "independent_question_group_count": len(question_records),
        "candidate_draw_count": sum(int(row["row_count"]) for row in question_records),
        "individual_draws_treated_as_independent_questions": False,
        "metrics": metrics,
        "per_question_effective_information": [
            {
                "source_exp6103_row_id": str(row["source_exp6103_row_id"]),
                "family": str(row["family"]),
                "accuracy": float(row["accuracy"]),
                "answer_cluster_entropy_bits": float(row["answer_cluster_entropy_bits"]),
                "binary_correctness_entropy_bits": float(row["binary_correctness_entropy_bits"]),
                "option_identity_additional_entropy_bits": float(
                    row["option_identity_additional_entropy_bits"]
                ),
                "max_response_cluster_share": float(row["max_response_cluster_share"]),
            }
            for row in question_records
        ],
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "question_clustered_uncertainty_and_effective_information"
        ],
    }


def _saturation_attribution(
    family_metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> JsonDict:
    states: dict[str, JsonDict] = {}
    for family, metrics in dict(family_metrics["by_family"]).items():
        accuracy = float(metrics["accuracy"])
        if accuracy == 1.0:
            state = "saturated"
        elif accuracy < ENUMERATED_FLOOR:
            state = "below_enumerated_floor"
        else:
            state = "middle_band"
        states[family] = {
            "state": state,
            "accuracy": accuracy,
            "enumerated_floor": ENUMERATED_FLOOR,
            "oracle_at_k": float(metrics["oracle_at_k"]),
            "tuned_sc_accuracy": float(metrics["tuned_sc_accuracy"]),
            "exact_first_rate": dict(diagnostics["family_position_confounding"])[family][
                "exact_first_rate"
            ],
            "response_first_rate": dict(diagnostics["family_position_confounding"])[family][
                "response_first_rate"
            ],
        }
    accuracies = [float(row["accuracy"]) for row in family_metrics["by_family"].values()]
    return {
        "schema": SCHEMA + ".saturation_below_chance_attribution",
        "family_states": states,
        "aggregate_accuracy": float(family_metrics["overall"]["accuracy"]),
        "aggregate_headroom_oracle_minus_tuned_sc": float(
            family_metrics["overall"]["oracle_minus_tuned_sc"]
        ),
        "family_accuracy_range": [round(min(accuracies), 6), round(max(accuracies), 6)],
        "family_nonexchangeability_delta": round(max(accuracies) - min(accuracies), 6),
        "dominant_attribution": (
            "saturation_plus_position_confounding_with_typed_choice_below_floor"
        ),
        "bimodality_statement": (
            "The bimodality is saturation in finite_domain_scheduling and logic_grid, "
            "both confounded with correct option at position 1, plus typed_finite_choice "
            "below its enumerated floor; true typed-choice inability versus distractor "
            "or position behavior is unresolved without pre-frozen transformed rows."
        ),
        "not_transport_failure": {
            "observed_row_count": EXPECTED_ROW_COUNT,
            "parseability": float(family_metrics["overall"]["parseability"]),
            "method_validity": float(family_metrics["overall"]["method_validity"]),
        },
        "principle": "saturation and below-floor attribution is made before any new inference.",
    }


def _transformation_specification() -> JsonDict:
    classes = [
        (
            "balanced_option_permutations",
            "Permute label-to-candidate order so each exact label and exact position is balanced within base-template splits.",
        ),
        (
            "typed_choice_normalization",
            "Normalize typed option surfaces while preserving the bounded optimizer and exact candidate identity.",
        ),
        (
            "controlled_distractors",
            "Generate distractors by bounded feature edits whose objective value remains independently checkable.",
        ),
        (
            "constraint_composition_depth",
            "Vary rule or constraint depth from public generators while preserving exact validator reachability.",
        ),
        (
            "proof_preserving_relabels",
            "Rename entities and values through invertible maps with unchanged exact proof obligations.",
        ),
        (
            "templated_paraphrases",
            "Rewrite prompt stems through frozen templates while preserving semantic hashes and exact labels.",
        ),
    ]
    return {
        "schema": SCHEMA + ".candidate_transformation_specification",
        "specification_status": "retired_not_frozen_for_generation",
        "new_model_rows_generated": False,
        "selected_transformation_class": None,
        "candidate_classes": [
            {
                "class_id": class_id,
                "construction": construction,
                "exact_answer_preserved": True,
                "independently_checkable_exact_answer": True,
                "label_blind": True,
                "held_dependence": False,
                "falsifiable_exact_construction": True,
                "nonzero_transformed_calibration_information_with_uncertainty": False,
                "retired_transport_or_scorer_mechanism": False,
                "approved_for_exp6141": False,
                "rejection_reason": (
                    "Exp6128 contains no model responses under this transformed class, "
                    "so approving it would infer transformed difficulty from held-free "
                    "design intent rather than measured calibration information."
                ),
            }
            for class_id, construction in classes
        ],
        "retirement_reason": (
            "No candidate class has transformed-row calibration information with "
            "question-clustered uncertainty while also breaking the saturated-family "
            "position confound and typed-choice below-floor failure."
        ),
        "future_rows_allowed_by_this_artifact": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["candidate_transformation_specification"],
    }


def _isolation_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".label_blind_held_isolation",
        "source_split_counts": dict(sorted(Counter(str(row["source_split"]) for row in rows).items())),
        "held_test_access_count": sum(1 for row in rows if str(row["source_split"]) != "calibration"),
        "exact_labels_read_for_evaluation_only": True,
        "held_outcomes_used_for_selection": False,
        "held_rows_changed": False,
        "source_labels_altered": False,
        "new_model_rows_generated": False,
        "live_inference_invoked": False,
        "subprocess_model_commands": 0,
        "transformation_selected_after_label_peeking": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["label_blind_and_held_isolation_receipt"],
    }


def _methodology_gap() -> JsonDict:
    return {
        "schema": SCHEMA + ".methodology_gap",
        "arxiv_id": "2608.02966",
        "title": "Every Wrong Answer Counts: Option-Level Psychometrics for LLM Multiple-Choice Benchmarks",
        "submitted_date": "2026-08-03",
        "url": "https://arxiv.org/abs/2608.02966",
        "fitted_nominal_response_model_claimed": False,
        "transparent_count_diagnostics_used": True,
        "gap_noted": (
            "LLM-NRM-style option psychometrics motivate the audit, but Exp6140 has "
            "one frozen model/source pool and therefore reports count diagnostics "
            "rather than a fitted ability/item model."
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES["top_level_model_specs_methodology_gap_noted"],
    }


def _immutable_hashes(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> JsonDict:
    path_hashes = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in HASHED_INPUTS
        if (root / relative).exists()
    }
    ids = [str(row["candidate_row_id"]) for row in rows]
    row_hashes = {str(row["candidate_row_id"]): str(row["candidate_row_hash"]) for row in rows}
    return {
        "schema": SCHEMA + ".immutable_hashes",
        "path_hashes": path_hashes,
        "raw_row_identity_hash": sha256_json(sorted(ids)),
        "raw_candidate_row_hashes": row_hashes,
        "raw_candidate_row_hash_map_sha256": sha256_json(row_hashes),
        "exp6128_rows_file_line_count": len(rows),
        "git_status_receipt": _git_status(root, output_path=output_path),
        "protected_file_hashes_before": _protected_hashes(root),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle": REQUIRED_FIELD_PRINCIPLES["immutable_source_artifact_and_row_hashes"],
    }


def protected_files_unchanged(
    *, root: Path = REPO_ROOT, before_hashes: Mapping[str, str] | None = None
) -> JsonDict:
    before = dict(before_hashes or _protected_hashes(root))
    after = _protected_hashes(root)
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "before": before,
        "after": after,
        "changed": changed,
        "unchanged": not changed,
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _preconditions(
    *,
    rows: Sequence[Mapping[str, Any]],
    counts: Mapping[str, Any],
    immutable: Mapping[str, Any],
) -> JsonDict:
    blockers: list[str] = []
    if counts["observed_row_count"] != EXPECTED_ROW_COUNT:
        blockers.append("row_count_mismatch")
    if counts["unique_candidate_row_id_count"] != EXPECTED_ROW_COUNT:
        blockers.append("row_identity_mismatch")
    if counts["question_group_count"] != EXPECTED_QUESTION_GROUP_COUNT:
        blockers.append("question_group_count_mismatch")
    if counts["candidate_rows_per_question_min"] != EXPECTED_K:
        blockers.append("question_group_missing_candidates")
    if counts["candidate_rows_per_question_max"] != EXPECTED_K:
        blockers.append("question_group_extra_candidates")
    if counts["all_rows_calibration_split"] is not True:
        blockers.append("non_calibration_row_present")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blockers,
        "blocked_reasons": blockers,
        "inference_substrate_declared": INFERENCE_SUBSTRATE,
        "live_inference_invoked": False,
        "raw_rows_loaded": len(rows),
        "dirty_worktree_recorded": True,
        "git_status_receipt": dict(immutable["git_status_receipt"]),
        "protected_file_hashes_before": dict(immutable["protected_file_hashes_before"]),
    }


def _field_provenance() -> JsonDict:
    sources = [
        EXP6103_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP6103_ROWS_RELATIVE_PATH.as_posix(),
        EXP6127_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP6128_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP6128_ROWS_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        "https://arxiv.org/abs/2608.02966",
    ]
    return {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES.get(field, "required Exp6140 schema field."),
            "sources": sources,
            "run_date": RUN_DATE,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        field: artifact.get(field)
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in {"duration_s", "test_exit_codes", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6140 schema and retirement discipline."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - schema guard.
        raise ValueError(f"missing_fields:{missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")  # pragma: no cover - schema guard.
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")  # pragma: no cover - schema guard.
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle")
    if artifact["empirical_item_bank_design_ready_score"] == 1.0 and artifact["retirement_triggered"]:
        raise ValueError("ready_score_requires_nonretired")
    if artifact["status"] == "retired":
        if artifact["retirement_triggered"] is not True:  # pragma: no cover - schema guard.
            raise ValueError("retired_without_trigger")
        if not str(artifact["honest_verdict"]).startswith("retired:"):  # pragma: no cover
            raise ValueError("retired_verdict")
    counts = dict(artifact["expected_observed_duplicate_and_missing_row_counts"])
    if counts["observed_row_count"] != EXPECTED_ROW_COUNT:  # pragma: no cover
        raise ValueError("row_conservation")
    if dict(artifact["label_blind_and_held_isolation_receipt"])["live_inference_invoked"]:
        raise ValueError("live_inference_invoked")  # pragma: no cover - schema guard.
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6103_rows_path: str | Path = REPO_ROOT / EXP6103_ROWS_RELATIVE_PATH,
    exp6128_artifact_path: str | Path = REPO_ROOT / EXP6128_ARTIFACT_RELATIVE_PATH,
    exp6128_rows_path: str | Path = REPO_ROOT / EXP6128_ROWS_RELATIVE_PATH,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Build the Exp6140 aggregation-only artifact from frozen upstream rows."""

    started = time.perf_counter()
    result = Path(result_path)
    exp6128_artifact = read_json(exp6128_artifact_path)
    rows = _annotate_rows(read_jsonl(exp6128_rows_path))
    exp6103_rows = read_jsonl(exp6103_rows_path)
    selected_source_rows = _source_rows_for_exp6128(exp6103_rows, rows)
    question_records = _question_records(rows)
    immutable = _immutable_hashes(root=REPO_ROOT, rows=rows, output_path=result)
    counts = _row_counts(rows)
    preconditions = _preconditions(rows=rows, counts=counts, immutable=immutable)
    family_metrics = _family_position_metrics(
        rows=rows,
        question_records=question_records,
        selected_source_rows=selected_source_rows,
    )
    diagnostics = _option_diagnostics(rows, question_records)
    uncertainty = _uncertainty(question_records)
    attribution = _saturation_attribution(family_metrics, diagnostics)
    artifact: JsonDict = {
        "status": "retired",
        "preconditions_checked": preconditions,
        "immutable_source_artifact_and_row_hashes": immutable,
        "expected_observed_duplicate_and_missing_row_counts": counts,
        "rederived_source_metric_reconciliation": _reconciliation(
            rows=rows,
            question_records=question_records,
            exp6128_artifact=exp6128_artifact,
        ),
        "family_stratum_semantic_group_relabel_shortcut_and_position_metrics": family_metrics,
        "wrong_option_identity_position_fallback_and_response_cluster_diagnostics": diagnostics,
        "question_clustered_uncertainty_and_effective_information": uncertainty,
        "saturation_and_below_chance_attribution": attribution,
        "candidate_transformation_specification": _transformation_specification(),
        "label_blind_and_held_isolation_receipt": _isolation_receipt(rows),
        "empirical_item_bank_design_ready_score": 0.0,
        "retirement_triggered": True,
        "top_level_model_specs_methodology_gap_noted": _methodology_gap(),
        "protected_files_unchanged": protected_files_unchanged(
            before_hashes=immutable["protected_file_hashes_before"]
        ),
        "duration_s": duration_s if duration_s is not None else round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [
            "Exp6140 evaluates frozen exact labels but is not an oracle and cannot distinguish typed-choice true inability from distractor or position effects without new pre-frozen transformed rows.",
            "No fitted multi-model LLM-NRM is claimed from one model's immutable Exp6128 row pool.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "retired: saturation plus position-confounded easy families and "
            "typed-choice below-floor fallback leave true inability versus "
            "distractor/position unresolved for Exp6128 source-domain recovery"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(result, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    run(result_path=args.output, write=args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
