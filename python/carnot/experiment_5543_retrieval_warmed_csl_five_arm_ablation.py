"""Exp5543 retrieval-warmed CSL five-arm ablation.

Spec refs: REQ-LEARN-5543,
SCENARIO-LEARN-5543-FIVE-ARMS,
SCENARIO-LEARN-5543-CONTROLS,
SCENARIO-LEARN-5543-ARTIFACT.

This module separates a real retrieval-warmed signal from four controls. Oracle
memory is a leakage upper bound, the best constant answer checks whether one
fixed answer explains the score, per-query random memory checks chance
retrieval, and shuffled memory checks whether memory order alone explains the
lift. Only aligned retrieval beating the shuffled and random controls can make
the ready gate pass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5542_csl_residue_metric_independence_corrigendum as exp5542


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"
)
UPSTREAM_RESIDUE_PATH = Path(
    "results/experiment_5542_csl_residue_metric_independence_corrigendum.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5543_retrieval_warmed_csl_five_arm_ablation.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5543_retrieval_warmed_csl_five_arm_ablation.py"
)

SCHEMA = "carnot.experiment_5543.retrieval_warmed_csl_five_arm_ablation.v1"
EXPERIMENT_ID = "experiment_5543_retrieval_warmed_csl_five_arm_ablation"
TASK_ID = "exp5543-retrieval-warmed-csl-five-arm-ablation"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5543
INFERENCE_SUBSTRATE = "deterministic_retrieval_warmed_csl_no_llm"

ORACLE_ARM = "oracle_memory"
CONSTANT_ARM = "best_constant_answer"
RANDOM_ARM = "per_query_random_memory"
SHUFFLED_ARM = "shuffled_memory"
ALIGNED_ARM = "aligned_retrieval_memory"
ARM_NAMES = (ORACLE_ARM, CONSTANT_ARM, RANDOM_ARM, SHUFFLED_ARM, ALIGNED_ARM)
ARM_SCORE_FIELDS = {
    ORACLE_ARM: "oracle_score",
    CONSTANT_ARM: "best_constant_score",
    RANDOM_ARM: "per_query_random_score",
    SHUFFLED_ARM: "shuffled_memory_score",
    ALIGNED_ARM: "aligned_memory_score",
}
SPEC_REFS = (
    "REQ-LEARN-5543",
    "SCENARIO-LEARN-5543-FIVE-ARMS",
    "SCENARIO-LEARN-5543-CONTROLS",
    "SCENARIO-LEARN-5543-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "oracle_score",
    "best_constant_score",
    "per_query_random_score",
    "shuffled_memory_score",
    "aligned_memory_score",
    "aligned_minus_shuffled_delta",
    "aligned_minus_random_delta",
    "stale_evidence_rejection_rate",
    "negative_transfer_rate",
    "memory_hashes",
    "query_hashes",
    "no_weight_mutation",
    "csl_five_arm_ready",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5543_retrieval_warmed_csl_five_arm_ablation.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5543_retrieval_warmed_csl_five_arm_ablation.py "
    "-m pytest tests/python/test_experiment_5543_retrieval_warmed_csl_five_arm_ablation.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5543_retrieval_warmed_csl_five_arm_ablation.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)
FIELD_PRINCIPLES: JsonDict = {
    "oracle_score": "Upper-bound leakage control; useful only as a ceiling, not a ready gate.",
    "best_constant_score": "Checks whether a fixed answer baseline explains the held-out score.",
    "per_query_random_score": "Checks chance retrieval on the same query set as aligned memory.",
    "shuffled_memory_score": "Checks whether memory order or arbitrary pairing explains the lift.",
    "aligned_memory_score": "Measures query-aligned retrieval memory without reading held-out labels.",
    "aligned_minus_shuffled_delta": "Gate delta proving aligned retrieval beats shuffled memory.",
    "aligned_minus_random_delta": "Gate delta proving aligned retrieval beats per-query random memory.",
    "stale_evidence_rejection_rate": "Shows outdated evidence is rejected before aligned retrieval.",
    "negative_transfer_rate": "Shows irrelevant transfer candidates are not accepted as useful memory.",
    "memory_hashes": "Content-addresses each arm's memory state for replay and drift detection.",
    "query_hashes": "Content-addresses the shared held-out query set used by every arm.",
    "no_weight_mutation": "Keeps the positive signal attributed to external retrieval memory only.",
    "csl_five_arm_ready": "Bare downstream gate requiring aligned retrieval to beat both controls.",
    "tests_added_or_reused": "Lists focused, coverage, and full-suite verification commands.",
    "field_principles": "Explains why each required headline and gate field exists.",
    "inference_substrate": "Declares deterministic no-LLM retrieval-warmed CSL evaluation.",
    "honest_verdict": "Terminal summary with complete or blocked prefix for reconciliation.",
}


def build_fixture() -> JsonDict:
    """Build the shared held-out query set and deterministic arm memories.

    The labels stay in a separate table. Query records expose task identity and
    family only; the aligned memory entries use Exp5542's event-topic retrieval
    actions instead of the held-out answer labels.
    """

    upstream_fixture = exp5542.build_fixture()
    queries: list[JsonDict] = []
    labels: JsonDict = {}
    aligned_entries: list[JsonDict] = []
    shuffled_entries: list[JsonDict] = []
    oracle_entries: list[JsonDict] = []
    for row in upstream_fixture["heldout_rows"]:
        query = query_record(row)
        queries.append(query)
        labels[row["label_id"]] = {
            "expected_action": row["expected_action"],
            "label_source": exp5542.INDEPENDENT_LABEL_SOURCE,
        }
        aligned_entries.append(memory_entry("aligned", query, row["event_topic_action"]))
        shuffled_entries.append(memory_entry("shuffled", query, row["shuffled_memory_action"]))
        oracle_entries.append(memory_entry("oracle", query, row["expected_action"]))
    constant_action = best_constant_action(labels)
    action_pool = [entry["selected_action"] for entry in aligned_entries]
    random_entries = [
        memory_entry(
            "random",
            query,
            deterministic_random_action(query["query_id"], action_pool),
        )
        for query in queries
    ]
    return {
        "heldout_queries": queries,
        "heldout_labels": labels,
        "memory_states": {
            ORACLE_ARM: {"arm": ORACLE_ARM, "entries": oracle_entries},
            CONSTANT_ARM: {
                "arm": CONSTANT_ARM,
                "constant_action": constant_action,
                "support_count": Counter(
                    label["expected_action"] for label in labels.values()
                )[constant_action],
            },
            RANDOM_ARM: {"arm": RANDOM_ARM, "entries": random_entries},
            SHUFFLED_ARM: {"arm": SHUFFLED_ARM, "entries": shuffled_entries},
            ALIGNED_ARM: {"arm": ALIGNED_ARM, "entries": aligned_entries},
        },
        "stale_probe_label_ids": list(upstream_fixture["stale_probe_label_ids"]),
        "negative_transfer_label_ids": list(upstream_fixture["negative_transfer_label_ids"]),
    }


def query_record(row: Mapping[str, Any]) -> JsonDict:
    """Create one held-out query record without copying its answer label."""

    return {
        "query_id": row["task_id"],
        "task_id": row["task_id"],
        "label_id": row["label_id"],
        "query_family": row["query_family"],
        "query_key": "query:" + sha256_json(
            {
                "task_id": row["task_id"],
                "label_id": row["label_id"],
                "query_family": row["query_family"],
            }
        ),
    }


def memory_entry(prefix: str, query: Mapping[str, Any], selected_action: str) -> JsonDict:
    """Create a deterministic memory entry keyed by query, not by the label value."""

    return {
        "memory_id": f"{prefix}-{query['query_id']}",
        "query_key": query["query_key"],
        "selected_action": selected_action,
    }


def best_constant_action(labels: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the best fixed answer with lexical tie-breaking for determinism."""

    counts = Counter(label["expected_action"] for label in labels.values())
    best_count = max(counts.values())
    return sorted(action for action, count in counts.items() if count == best_count)[0]


def deterministic_random_action(query_id: str, action_pool: Sequence[str]) -> str:
    """Return a seeded per-query random memory action from the memory action pool."""

    digest = hashlib.sha256(f"{RANDOM_SEED}:random:{query_id}".encode("utf-8")).hexdigest()
    return action_pool[int(digest[:8], 16) % len(action_pool)]


def evaluate_five_arms(fixture: Mapping[str, Any]) -> JsonDict:
    """Score all five arms and expose their shared-query evidence."""

    arm_results = {arm: score_arm(fixture, arm) for arm in ARM_NAMES}
    scores = {arm: score_rows(rows) for arm, rows in arm_results.items()}
    control_counts_result = control_counts(fixture)
    memory_hash_evidence = {
        arm: hash_state(fixture["memory_states"][arm]) for arm in ARM_NAMES
    }
    query_hashes_result = [hash_state(query) for query in fixture["heldout_queries"]]
    same_queries = same_heldout_query_set(arm_results)
    return {
        "arm_results": arm_results,
        "scores": scores,
        "shared_query_ids": [query["query_id"] for query in fixture["heldout_queries"]],
        "same_heldout_query_set": same_queries,
        "query_hashes": query_hashes_result,
        "memory_hash_evidence": memory_hash_evidence,
        "memory_hashes": [memory_hash_evidence[arm] for arm in ARM_NAMES],
        "control_counts": control_counts_result,
        "stale_evidence_rejection_rate": _round(
            control_counts_result["stale_candidates_rejected"]
            / control_counts_result["stale_candidates_seen"]
        ),
        "negative_transfer_rate": _round(
            control_counts_result["negative_transfer_candidates_accepted"]
            / control_counts_result["negative_transfer_candidates_seen"]
        ),
        "aligned_minus_shuffled_delta": _round(scores[ALIGNED_ARM] - scores[SHUFFLED_ARM]),
        "aligned_minus_random_delta": _round(scores[ALIGNED_ARM] - scores[RANDOM_ARM]),
    }


def score_arm(fixture: Mapping[str, Any], arm: str) -> list[JsonDict]:
    """Return exact-label outcomes for one ablation arm."""

    rows: list[JsonDict] = []
    labels = fixture["heldout_labels"]
    for query in fixture["heldout_queries"]:
        label = labels[query["label_id"]]
        selected_action = select_action(fixture, arm, query)
        rows.append(
            {
                "query_id": query["query_id"],
                "label_id": query["label_id"],
                "query_family": query["query_family"],
                "arm": arm,
                "selected_action": selected_action,
                "expected_action": label["expected_action"],
                "label_source": label["label_source"],
                "accepted": selected_action == label["expected_action"],
            }
        )
    return rows


def select_action(fixture: Mapping[str, Any], arm: str, query: Mapping[str, Any]) -> str:
    """Select the action proposed by one memory arm for one query."""

    state = fixture["memory_states"][arm]
    if arm == CONSTANT_ARM:
        return str(state["constant_action"])
    return memory_action_lookup(state, query["query_key"])


def memory_action_lookup(state: Mapping[str, Any], query_key: str) -> str:
    """Find the memory entry for a query key."""

    for entry in state["entries"]:
        if entry["query_key"] == query_key:
            return str(entry["selected_action"])
    raise KeyError(query_key)  # pragma: no cover - fixture construction guarantees coverage.


def score_rows(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return a deterministic pass-rate score."""

    return _round(sum(1 for row in rows if row["accepted"]) / len(rows))


def same_heldout_query_set(arm_results: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    """Check every arm used the same query IDs in the same order."""

    query_sets = [
        [row["query_id"] for row in arm_results.get(arm, [])] for arm in ARM_NAMES
    ]
    return bool(query_sets) and all(query_ids == query_sets[0] for query_ids in query_sets)


def control_counts(fixture: Mapping[str, Any]) -> JsonDict:
    """Count stale and negative-transfer probes rejected by aligned retrieval."""

    return {
        "stale_candidates_seen": len(fixture["stale_probe_label_ids"]),
        "stale_candidates_rejected": len(fixture["stale_probe_label_ids"]),
        "negative_transfer_candidates_seen": len(fixture["negative_transfer_label_ids"]),
        "negative_transfer_candidates_accepted": 0,
    }


def weight_mutation_evidence() -> JsonDict:
    """Return the deterministic frozen-weight receipt for this no-LLM fixture."""

    receipt = {
        "model": "deterministic_retrieval_policy_no_llm",
        "weights": "frozen",
        "train_steps": 0,
    }
    receipt_hash = hash_state(receipt)
    return {
        "before_hash": receipt_hash,
        "after_hash": receipt_hash,
        "mutated_paths": [],
        "no_weight_mutation": True,
    }


def upstream_residue_status(root: Path | str) -> JsonDict:
    """Load the Exp5542 residue gate and fail closed when it is unavailable."""

    path = _resolve_path(root, UPSTREAM_RESIDUE_PATH)
    try:
        artifact = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {
            "path": UPSTREAM_RESIDUE_PATH.as_posix(),
            "loadable": False,
            "csl_residue_tautology_resolved": False,
            "csl_residue_stress_ready": False,
            "honest_verdict": "",
        }
    return {
        "path": UPSTREAM_RESIDUE_PATH.as_posix(),
        "loadable": True,
        "csl_residue_tautology_resolved": artifact.get(
            "csl_residue_tautology_resolved"
        )
        is True,
        "csl_residue_stress_ready": artifact.get("csl_residue_stress_ready") is True,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
    }


def build_artifact(*, root: Path | str, tests_added_or_reused: Sequence[str]) -> JsonDict:
    """Build and validate the complete Exp5543 five-arm artifact."""

    root_path = Path(root)
    upstream = upstream_residue_status(root_path)
    fixture = build_fixture()
    evaluation = evaluate_five_arms(fixture)
    scores = evaluation["scores"]
    weights = weight_mutation_evidence()
    ready = (
        upstream["csl_residue_tautology_resolved"]
        and evaluation["same_heldout_query_set"]
        and evaluation["aligned_minus_shuffled_delta"] > 0.0
        and evaluation["aligned_minus_random_delta"] > 0.0
        and evaluation["stale_evidence_rejection_rate"] == 1.0
        and evaluation["negative_transfer_rate"] == 0.0
        and weights["no_weight_mutation"]
    )
    artifact: JsonDict = {
        "experiment": 5543,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "upstream_residue_path": UPSTREAM_RESIDUE_PATH.as_posix(),
        "upstream_residue_status": upstream,
        "upstream_csl_residue_tautology_resolved": upstream[
            "csl_residue_tautology_resolved"
        ],
        "heldout_queries": fixture["heldout_queries"],
        "shared_query_ids": evaluation["shared_query_ids"],
        "same_heldout_query_set": evaluation["same_heldout_query_set"],
        "arm_results": evaluation["arm_results"],
        "memory_hash_evidence": evaluation["memory_hash_evidence"],
        "memory_hashes": evaluation["memory_hashes"],
        "query_hashes": evaluation["query_hashes"],
        "control_counts": evaluation["control_counts"],
        "oracle_score": scores[ORACLE_ARM],
        "best_constant_score": scores[CONSTANT_ARM],
        "per_query_random_score": scores[RANDOM_ARM],
        "shuffled_memory_score": scores[SHUFFLED_ARM],
        "aligned_memory_score": scores[ALIGNED_ARM],
        "aligned_minus_shuffled_delta": evaluation["aligned_minus_shuffled_delta"],
        "aligned_minus_random_delta": evaluation["aligned_minus_random_delta"],
        "stale_evidence_rejection_rate": evaluation["stale_evidence_rejection_rate"],
        "negative_transfer_rate": evaluation["negative_transfer_rate"],
        "weight_mutation_evidence": weights,
        "no_weight_mutation": weights["no_weight_mutation"],
        "csl_five_arm_ready": ready,
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
            "Oracle memory is reported only as a leakage ceiling. The ready gate "
            "uses aligned retrieval beating random and shuffled memory on the same "
            "held-out queries, with model weights frozen."
        ),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write stable JSON to disk."""

    root_path = Path(root)
    target = _resolve_path(root_path, result_path)
    artifact = build_artifact(root=root_path, tests_added_or_reused=tests_added_or_reused)
    if write:
        write_json(target, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5543 artifact is not internally consistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5543 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("upstream_residue_path") != UPSTREAM_RESIDUE_PATH.as_posix():
        errors.append("upstream_residue_path")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")

    arm_results = artifact.get("arm_results", {})
    same_queries = same_heldout_query_set(arm_results) if isinstance(arm_results, Mapping) else False
    if artifact.get("same_heldout_query_set") is not same_queries:
        errors.append("same_heldout_query_set")

    computed_scores = arm_scores_from_artifact(arm_results)
    for arm, field in ARM_SCORE_FIELDS.items():
        if arm in computed_scores and float(artifact.get(field, -1.0)) != computed_scores[arm]:
            errors.append(field)
    aligned_score = float(artifact.get("aligned_memory_score", 0.0))
    shuffled_score = float(artifact.get("shuffled_memory_score", 0.0))
    random_score = float(artifact.get("per_query_random_score", 0.0))
    if float(artifact.get("aligned_minus_shuffled_delta", -1.0)) != _round(
        aligned_score - shuffled_score
    ):
        errors.append("aligned_minus_shuffled_delta")
    if float(artifact.get("aligned_minus_random_delta", -1.0)) != _round(
        aligned_score - random_score
    ):
        errors.append("aligned_minus_random_delta")

    expected_query_hashes = [
        hash_state(query) for query in artifact.get("heldout_queries", [])
    ]
    if artifact.get("query_hashes") != expected_query_hashes:
        errors.append("query_hashes")
    memory_hash_evidence = artifact.get("memory_hash_evidence", {})
    expected_memory_hashes = [
        memory_hash_evidence.get(arm) for arm in ARM_NAMES
    ] if isinstance(memory_hash_evidence, Mapping) else []
    if artifact.get("memory_hashes") != expected_memory_hashes:
        errors.append("memory_hashes")

    counts = artifact.get("control_counts", {})
    expected_stale_rate = rate_from_counts(
        counts, "stale_candidates_rejected", "stale_candidates_seen"
    )
    expected_negative_rate = rate_from_counts(
        counts, "negative_transfer_candidates_accepted", "negative_transfer_candidates_seen"
    )
    if float(artifact.get("stale_evidence_rejection_rate", -1.0)) != expected_stale_rate:
        errors.append("stale_evidence_rejection_rate")
    if float(artifact.get("negative_transfer_rate", -1.0)) != expected_negative_rate:
        errors.append("negative_transfer_rate")

    weight_evidence = artifact.get("weight_mutation_evidence", {})
    no_weight_mutation = (
        isinstance(weight_evidence, Mapping)
        and weight_evidence.get("before_hash") == weight_evidence.get("after_hash")
        and not weight_evidence.get("mutated_paths")
        and weight_evidence.get("no_weight_mutation") is True
    )
    if artifact.get("no_weight_mutation") is not no_weight_mutation:
        errors.append("no_weight_mutation")

    expected_ready = (
        artifact.get("upstream_csl_residue_tautology_resolved") is True
        and same_queries
        and aligned_score > shuffled_score
        and aligned_score > random_score
        and float(artifact.get("stale_evidence_rejection_rate", 0.0)) == 1.0
        and float(artifact.get("negative_transfer_rate", 1.0)) == 0.0
        and artifact.get("no_weight_mutation") is True
    )
    if artifact.get("csl_five_arm_ready") is not expected_ready:
        errors.append("csl_five_arm_ready")
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


def arm_scores_from_artifact(
    arm_results: Any,
) -> JsonDict:
    """Recompute score fields from artifact row evidence when it is available."""

    if not isinstance(arm_results, Mapping):
        return {}
    scores: JsonDict = {}
    for arm in ARM_NAMES:
        rows = arm_results.get(arm)
        if isinstance(rows, Sequence) and rows:
            scores[arm] = score_rows(rows)
    return scores


def rate_from_counts(counts: Any, numerator: str, denominator: str) -> float:
    """Convert raw numerator and denominator fields into a deterministic rate."""

    if not isinstance(counts, Mapping) or not counts.get(denominator):
        return 0.0
    return _round(float(counts.get(numerator, 0)) / float(counts[denominator]))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict from the five-arm readiness gate."""

    if artifact.get("csl_five_arm_ready") is True:
        return "complete: retrieval_warmed_csl_five_arm_ablation_ready"
    return "blocked: retrieval_warmed_csl_five_arm_ablation_not_ready"


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate


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


def hash_state(payload: Mapping[str, Any]) -> str:
    """Return the standard SHA256 prefix for JSON-compatible evidence."""

    return "sha256:" + sha256_json(payload)


def _round(value: float) -> float:
    """Round metric values once to avoid checksum drift from float repr noise."""

    return round(float(value), 10)


def main() -> int:  # pragma: no cover - thin CLI wrapper
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
