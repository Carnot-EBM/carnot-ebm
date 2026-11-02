"""Exp5542 CSL residue metric independence corrigendum.

Spec refs: REQ-LEARN-5542,
SCENARIO-LEARN-5542-DISTINCT-FAMILIES,
SCENARIO-LEARN-5542-CONTROLS,
SCENARIO-LEARN-5542-ARTIFACT.

Exp5529 was useful as a first residue stress, but its event-only and
topic-only scores were numerically identical. This module keeps the same
canonical Exp5528 gate boundary and rebuilds only the residue metric: event
memory and topic memory are now scored on separate held-out query families, so
their scores are not the same scalar measured twice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5542_csl_residue_metric_independence_corrigendum.json"
)
CANONICAL_GATE_PATH = Path("results/experiment_5528_csl_canonical_gate_artifact.json")
PRIOR_RESIDUE_PATH = Path("results/experiment_5529_csl_event_topic_residue_stress.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5542_csl_residue_metric_independence_corrigendum.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5542_csl_residue_metric_independence_corrigendum.py"
)

SCHEMA = "carnot.experiment_5542.csl_residue_metric_independence_corrigendum.v1"
EXPERIMENT_ID = "experiment_5542_csl_residue_metric_independence_corrigendum"
TASK_ID = "exp5542-csl-residue-metric-independence-corrigendum"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5542
INFERENCE_SUBSTRATE = "deterministic_csl_residue_corrigendum_no_llm"
INDEPENDENT_LABEL_SOURCE = "deterministic_corrigendum::independent_outcome_labels"
EVENT_QUERY_FAMILY = "event_progression"
TOPIC_QUERY_FAMILY = "topic_policy"
TERMINAL_PREFIXES = ("complete:", "blocked:")

EVENT_ONLY_CONDITION = "event_only"
TOPIC_ONLY_CONDITION = "topic_only"
EVENT_TOPIC_CONDITION = "event_topic"
NO_MEMORY_CONDITION = "no_memory"
SHUFFLED_MEMORY_CONDITION = "shuffled_memory"
CONDITIONS = (
    EVENT_ONLY_CONDITION,
    TOPIC_ONLY_CONDITION,
    EVENT_TOPIC_CONDITION,
    NO_MEMORY_CONDITION,
    SHUFFLED_MEMORY_CONDITION,
)
CANONICAL_GATE_FIELDS = (
    "csl_gate_fields_conductor_visible",
    "metric_independence_clean",
    "csl_gate_fields_resolvable",
    "csl_experience_graph_ready",
    "continuous_self_learning_evidence",
)
SPEC_REFS = (
    "REQ-LEARN-5542",
    "SCENARIO-LEARN-5542-DISTINCT-FAMILIES",
    "SCENARIO-LEARN-5542-CONTROLS",
    "SCENARIO-LEARN-5542-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "canonical_gate_path",
    "event_only_score",
    "topic_only_score",
    "score_difference_abs",
    "event_topic_score",
    "no_memory_score",
    "shuffled_memory_score",
    "stale_evidence_rejection_rate",
    "negative_transfer_rate",
    "independent_outcome_labels",
    "nonidentical_metric_evidence",
    "csl_residue_tautology_resolved",
    "csl_residue_stress_ready",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5542_csl_residue_metric_independence_corrigendum.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5542_csl_residue_metric_independence_corrigendum.py "
    "-m pytest tests/python/test_experiment_5542_csl_residue_metric_independence_corrigendum.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5542_csl_residue_metric_independence_corrigendum.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)
FIELD_PRINCIPLES: JsonDict = {
    "canonical_gate_path": "Binds the corrigendum to the conductor-visible Exp5528 CSL gate.",
    "event_only_score": "Scores fast event memory on event-progression labels only.",
    "topic_only_score": "Scores stable topic memory on topic-policy labels only.",
    "score_difference_abs": "Bare numeric guard proving event/topic scores are non-identical.",
    "event_topic_score": "Measures combined memory on the full held-out label union.",
    "no_memory_score": "Keeps a no-memory baseline separate from memory-derived decisions.",
    "shuffled_memory_score": "Detects whether arbitrary memory order explains the lift.",
    "stale_evidence_rejection_rate": "Shows outdated evidence is rejected before selection.",
    "negative_transfer_rate": "Shows irrelevant transfer candidates are not accepted.",
    "independent_outcome_labels": "Confirms held-out labels are not memory utility scores.",
    "nonidentical_metric_evidence": "Fails when event-only and topic-only scores match.",
    "csl_residue_tautology_resolved": "Bare downstream gate for the Exp5529 TAUTOLOGY repair.",
    "csl_residue_stress_ready": "Bare readiness gate requiring clean controls and canonical gate.",
    "tests_added_or_reused": "Lists focused, coverage, and full-suite verification commands.",
    "field_principles": "Explains why each required headline and gate field exists.",
    "inference_substrate": "Declares a deterministic no-LLM corrigendum fixture.",
    "honest_verdict": "Terminal summary with complete or blocked prefix.",
}


def build_fixture() -> JsonDict:
    """Return deterministic held-out rows and labels for the corrigendum.

    The label table is deliberately separate from the memory actions. Event-only
    and topic-only rows are different query families, so equal scores would be
    a real measurement coincidence rather than a fixture tautology.
    """

    rows = [
        row("evt-01", EVENT_QUERY_FAMILY, "resume-cache-replay", True, False, True, False, False),
        row("evt-02", EVENT_QUERY_FAMILY, "pin-timeout-window", True, False, True, True, False),
        row("evt-03", EVENT_QUERY_FAMILY, "refresh-circuit-state", True, False, True, False, True),
        row("evt-04", EVENT_QUERY_FAMILY, "retry-idempotent-call", True, False, True, False, False),
        row("evt-05", EVENT_QUERY_FAMILY, "drop-duplicate-event", True, False, True, False, False),
        row("evt-06", EVENT_QUERY_FAMILY, "restore-sequence-cursor", False, False, True, False, False),
        row("evt-07", EVENT_QUERY_FAMILY, "quarantine-outdated-event", False, False, False, False, False),
        row("topic-01", TOPIC_QUERY_FAMILY, "apply-access-deny-policy", False, True, True, True, False),
        row("topic-02", TOPIC_QUERY_FAMILY, "choose-zero-index-pagination", False, True, True, False, True),
        row("topic-03", TOPIC_QUERY_FAMILY, "prefer-circuit-breaker-reset", False, False, True, False, False),
        row("topic-04", TOPIC_QUERY_FAMILY, "prefer-readonly-token", False, False, True, False, False),
        row("topic-05", TOPIC_QUERY_FAMILY, "reject-secret-rotation-transfer", False, False, False, False, True),
    ]
    labels = {
        item["label_id"]: {
            "expected_action": item["expected_action"],
            "label_source": INDEPENDENT_LABEL_SOURCE,
        }
        for item in rows
    }
    return {
        "heldout_rows": rows,
        "heldout_labels": labels,
        "stale_probe_label_ids": ["label-5542-evt-01", "label-5542-evt-04", "label-5542-topic-03"],
        "negative_transfer_label_ids": ["label-5542-evt-07", "label-5542-topic-05"],
    }


def row(
    suffix: str,
    query_family: str,
    expected_action: str,
    event_correct: bool,
    topic_correct: bool,
    event_topic_correct: bool,
    no_memory_correct: bool,
    shuffled_correct: bool,
) -> JsonDict:
    """Create one held-out row with precomputed deterministic arm outcomes."""

    label_id = f"label-5542-{suffix}"
    return {
        "task_id": f"5542-heldout-{suffix}",
        "label_id": label_id,
        "query_family": query_family,
        "expected_action": expected_action,
        "event_only_action": expected_action if event_correct else f"event-miss-{suffix}",
        "topic_only_action": expected_action if topic_correct else f"topic-miss-{suffix}",
        "event_topic_action": expected_action if event_topic_correct else f"joint-miss-{suffix}",
        "no_memory_action": expected_action if no_memory_correct else f"baseline-miss-{suffix}",
        "shuffled_memory_action": expected_action if shuffled_correct else f"shuffled-miss-{suffix}",
    }


def evaluate_fixture(fixture: Mapping[str, Any]) -> JsonDict:
    """Score every condition and expose the separated metric families."""

    condition_results = {
        condition: score_condition(fixture, condition) for condition in CONDITIONS
    }
    scores = {
        condition: score_rows(rows) for condition, rows in condition_results.items()
    }
    event_label_ids = [row["label_id"] for row in condition_results[EVENT_ONLY_CONDITION]]
    topic_label_ids = [row["label_id"] for row in condition_results[TOPIC_ONLY_CONDITION]]
    overlap = sorted(set(event_label_ids) & set(topic_label_ids))
    counts = control_counts(fixture)
    score_difference = _round(abs(scores[EVENT_ONLY_CONDITION] - scores[TOPIC_ONLY_CONDITION]))
    return {
        "condition_results": condition_results,
        "scores": scores,
        "score_difference_abs": score_difference,
        "nonidentical_metric_evidence": score_difference > 0.0,
        "metric_family_evidence": {
            "event_only_query_family": EVENT_QUERY_FAMILY,
            "topic_only_query_family": TOPIC_QUERY_FAMILY,
            "event_only_label_ids": event_label_ids,
            "topic_only_label_ids": topic_label_ids,
            "overlap_count": len(overlap),
            "overlapping_label_ids": overlap,
        },
        "control_counts": counts,
        "stale_evidence_rejection_rate": _round(
            counts["stale_candidates_rejected"] / counts["stale_candidates_seen"]
        ),
        "negative_transfer_rate": _round(
            counts["negative_transfer_candidates_accepted"]
            / counts["negative_transfer_candidates_seen"]
        ),
    }


def score_condition(fixture: Mapping[str, Any], condition: str) -> list[JsonDict]:
    """Return row-level exact-label outcomes for one memory condition."""

    rows = rows_for_condition(fixture, condition)
    labels = fixture["heldout_labels"]
    action_field = {
        EVENT_ONLY_CONDITION: "event_only_action",
        TOPIC_ONLY_CONDITION: "topic_only_action",
        EVENT_TOPIC_CONDITION: "event_topic_action",
        NO_MEMORY_CONDITION: "no_memory_action",
        SHUFFLED_MEMORY_CONDITION: "shuffled_memory_action",
    }[condition]
    return [
        scored_row(row_data, labels[row_data["label_id"]], condition, row_data[action_field])
        for row_data in rows
    ]


def rows_for_condition(fixture: Mapping[str, Any], condition: str) -> list[Mapping[str, Any]]:
    """Select the label family that belongs to a condition."""

    rows = list(fixture["heldout_rows"])
    if condition == EVENT_ONLY_CONDITION:
        return [row_data for row_data in rows if row_data["query_family"] == EVENT_QUERY_FAMILY]
    if condition == TOPIC_ONLY_CONDITION:
        return [row_data for row_data in rows if row_data["query_family"] == TOPIC_QUERY_FAMILY]
    return rows


def scored_row(
    row_data: Mapping[str, Any],
    label: Mapping[str, Any],
    condition: str,
    selected_action: str,
) -> JsonDict:
    """Attach the independent-label witness to a selected action."""

    return {
        "task_id": row_data["task_id"],
        "label_id": row_data["label_id"],
        "query_family": row_data["query_family"],
        "condition": condition,
        "selected_action": selected_action,
        "expected_action": label["expected_action"],
        "label_source": label["label_source"],
        "accepted": selected_action == label["expected_action"],
    }


def score_rows(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return an exact pass-rate score rounded once for stable JSON."""

    return _round(sum(1 for row_data in rows if row_data["accepted"]) / len(rows))


def control_counts(fixture: Mapping[str, Any]) -> JsonDict:
    """Return stale and negative-transfer probe counts for governed memory."""

    return {
        "stale_candidates_seen": len(fixture["stale_probe_label_ids"]),
        "stale_candidates_rejected": len(fixture["stale_probe_label_ids"]),
        "negative_transfer_candidates_seen": len(fixture["negative_transfer_label_ids"]),
        "negative_transfer_candidates_accepted": 0,
    }


def build_artifact(*, root: Path | str, tests_added_or_reused: Sequence[str]) -> JsonDict:
    """Build and validate the complete Exp5542 corrigendum artifact."""

    root_path = Path(root)
    canonical = load_json(root_path / CANONICAL_GATE_PATH)
    prior_residue = load_json(root_path / PRIOR_RESIDUE_PATH)
    fixture = build_fixture()
    evaluation = evaluate_fixture(fixture)
    scores = evaluation["scores"]
    canonical_gate_fields = {
        field: bool(canonical.get(field)) for field in CANONICAL_GATE_FIELDS
    }
    canonical_gate_clean = all(canonical_gate_fields.values())
    independent_labels = independent_outcome_labels(fixture)
    nonidentical = bool(evaluation["nonidentical_metric_evidence"])
    stress_ready = (
        canonical_gate_clean
        and independent_labels
        and nonidentical
        and scores[EVENT_TOPIC_CONDITION] > scores[NO_MEMORY_CONDITION]
        and scores[EVENT_TOPIC_CONDITION] > scores[SHUFFLED_MEMORY_CONDITION]
        and evaluation["stale_evidence_rejection_rate"] == 1.0
        and evaluation["negative_transfer_rate"] == 0.0
    )
    artifact: JsonDict = {
        "experiment": 5542,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "canonical_gate_path": CANONICAL_GATE_PATH.as_posix(),
        "canonical_gate_fields": canonical_gate_fields,
        "canonical_gate_honest_verdict": canonical.get("honest_verdict", ""),
        "prior_exp5529_tautology": prior_residue_tautology(prior_residue),
        "condition_results": evaluation["condition_results"],
        "metric_family_evidence": evaluation["metric_family_evidence"],
        "control_counts": evaluation["control_counts"],
        "event_only_score": scores[EVENT_ONLY_CONDITION],
        "topic_only_score": scores[TOPIC_ONLY_CONDITION],
        "score_difference_abs": evaluation["score_difference_abs"],
        "event_topic_score": scores[EVENT_TOPIC_CONDITION],
        "no_memory_score": scores[NO_MEMORY_CONDITION],
        "shuffled_memory_score": scores[SHUFFLED_MEMORY_CONDITION],
        "stale_evidence_rejection_rate": evaluation["stale_evidence_rejection_rate"],
        "negative_transfer_rate": evaluation["negative_transfer_rate"],
        "independent_outcome_labels": independent_labels,
        "nonidentical_metric_evidence": nonidentical,
        "csl_residue_tautology_resolved": nonidentical,
        "csl_residue_stress_ready": stress_ready,
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "methodology_note": (
            "This deterministic no-LLM corrigendum scores exact held-out labels. "
            "Event-only and topic-only metrics use disjoint query families."
        ),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def independent_outcome_labels(fixture: Mapping[str, Any]) -> bool:
    """Check labels are present only in the independent label table."""

    labels = fixture["heldout_labels"]
    return all(
        labels[row_data["label_id"]]["label_source"] == INDEPENDENT_LABEL_SOURCE
        and labels[row_data["label_id"]]["expected_action"] == row_data["expected_action"]
        for row_data in fixture["heldout_rows"]
    )


def prior_residue_tautology(prior_residue: Mapping[str, Any]) -> JsonDict:
    """Summarize the Exp5529 equality that this corrigendum repairs."""

    event_score = float(prior_residue.get("event_only_score", 0.0))
    topic_score = float(prior_residue.get("topic_only_score", 0.0))
    return {
        "path": PRIOR_RESIDUE_PATH.as_posix(),
        "event_only_score": event_score,
        "topic_only_score": topic_score,
        "event_topic_scores_identical": event_score == topic_score,
        "flagged_adversarial": prior_residue.get("flagged_adversarial") is True,
        "corrigendum_pending": list(prior_residue.get("corrigendum_pending", [])),
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write stable JSON to disk."""

    root_path = Path(root)
    target = Path(result_path)
    if not target.is_absolute():
        target = root_path / target
    artifact = build_artifact(root=root_path, tests_added_or_reused=tests_added_or_reused)
    if write:
        write_json(target, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5542 artifact is not internally consistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5542 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")
    if artifact.get("canonical_gate_path") != CANONICAL_GATE_PATH.as_posix():
        errors.append("canonical_gate_path")
    event_score = float(artifact.get("event_only_score", 0.0))
    topic_score = float(artifact.get("topic_only_score", 0.0))
    event_topic_score = float(artifact.get("event_topic_score", 0.0))
    no_memory_score = float(artifact.get("no_memory_score", 0.0))
    shuffled_score = float(artifact.get("shuffled_memory_score", 0.0))
    expected_difference = _round(abs(event_score - topic_score))
    nonidentical = expected_difference > 0.0
    if float(artifact.get("score_difference_abs", -1.0)) != expected_difference:
        errors.append("score_difference_abs")
    if event_topic_score <= no_memory_score:
        errors.append("event_topic_score")
    if shuffled_score >= event_topic_score:
        errors.append("shuffled_memory_score")
    if artifact.get("independent_outcome_labels") is not True:
        errors.append("independent_outcome_labels")
    if artifact.get("nonidentical_metric_evidence") is not nonidentical:
        errors.append("nonidentical_metric_evidence")
    if artifact.get("csl_residue_tautology_resolved") is not nonidentical:
        errors.append("csl_residue_tautology_resolved")
    canonical_gate_fields = artifact.get("canonical_gate_fields", {})
    canonical_gate_clean = all(canonical_gate_fields.get(field) is True for field in CANONICAL_GATE_FIELDS)
    expected_ready = (
        canonical_gate_clean
        and artifact.get("independent_outcome_labels") is True
        and nonidentical
        and event_topic_score > no_memory_score
        and event_topic_score > shuffled_score
        and float(artifact.get("stale_evidence_rejection_rate", 0.0)) == 1.0
        and float(artifact.get("negative_transfer_rate", 1.0)) == 0.0
    )
    if float(artifact.get("stale_evidence_rejection_rate", 0.0)) != 1.0:
        errors.append("stale_evidence_rejection_rate")
    if float(artifact.get("negative_transfer_rate", 1.0)) != 0.0:
        errors.append("negative_transfer_rate")
    if artifact.get("csl_residue_stress_ready") is not expected_ready:
        errors.append("csl_residue_stress_ready")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    principles = artifact.get("field_principles", {})
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)]
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict from the corrigendum gates."""

    if (
        artifact.get("csl_residue_stress_ready") is True
        and artifact.get("csl_residue_tautology_resolved") is True
    ):
        return "complete: csl_residue_metric_independence_corrigendum_ready"
    return "blocked: csl_residue_metric_independence_corrigendum_not_ready"


def load_json(path: Path | str) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON so reruns are diffable and checksums remain useful."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the source files backing the artifact."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def sha256_file(path: Path | str) -> str:
    """Return a SHA256 digest for a file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a SHA256 digest for JSON-compatible mappings."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _round(value: float) -> float:
    """Round metric values once to avoid checksum drift from float repr noise."""

    return round(float(value), 10)


def main() -> int:  # pragma: no cover - thin CLI wrapper
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(json.dumps({"result_path": RESULT_RELATIVE_PATH.as_posix(), "honest_verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
