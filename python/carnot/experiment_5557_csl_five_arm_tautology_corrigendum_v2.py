"""Exp5557 CSL five-arm tautology corrigendum v2.

Spec refs: REQ-LEARN-5557,
SCENARIO-LEARN-5557-BASELINES,
SCENARIO-LEARN-5557-CONTROLS,
SCENARIO-LEARN-5557-ARTIFACT.

Exp5543 used semantically different best-constant and per-query-random
controls, but the deterministic random draw self-matched one query and landed
on the same 1/12 score as the constant baseline. This module keeps the
repaired Exp5542 independent labels, adds the missing no-memory arm, and makes
the random control sample non-self memory rows so a baseline tie becomes an
explicit failure instead of a clean gate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5542_csl_residue_metric_independence_corrigendum as exp5542


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5557_csl_five_arm_tautology_corrigendum_v2.json"
)
UPSTREAM_CSL_RESIDUE_CORRIGENDUM = Path(
    "results/experiment_5542_csl_residue_metric_independence_corrigendum.json"
)
UPSTREAM_FLAGGED_ABLATION = Path(
    "results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5557_csl_five_arm_tautology_corrigendum_v2.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5557_csl_five_arm_tautology_corrigendum_v2.py"
)

SCHEMA = "carnot.experiment_5557.csl_five_arm_tautology_corrigendum_v2"
EXPERIMENT_ID = "experiment_5557_csl_five_arm_tautology_corrigendum_v2"
TASK_ID = "exp5557-csl-five-arm-tautology-corrigendum-v2"
MILESTONE = "2026.07.503"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5557
EQUALITY_TOLERANCE = 1e-10
INFERENCE_SUBSTRATE = "deterministic_csl_ablation_no_llm"

BEST_CONSTANT_ARM = "best_constant"
PER_QUERY_RANDOM_ARM = "per_query_random"
NO_MEMORY_ARM = "no_memory"
SHUFFLED_MEMORY_ARM = "shuffled_memory"
ALIGNED_MEMORY_ARM = "aligned_memory"
ARM_NAMES = (
    BEST_CONSTANT_ARM,
    PER_QUERY_RANDOM_ARM,
    NO_MEMORY_ARM,
    SHUFFLED_MEMORY_ARM,
    ALIGNED_MEMORY_ARM,
)
ARM_SCORE_FIELDS = {
    BEST_CONSTANT_ARM: "best_constant_score",
    PER_QUERY_RANDOM_ARM: "per_query_random_score",
    NO_MEMORY_ARM: "no_memory_score",
    SHUFFLED_MEMORY_ARM: "shuffled_memory_score",
    ALIGNED_MEMORY_ARM: "aligned_memory_score",
}
HEADLINE_SCORE_FIELDS = tuple(ARM_SCORE_FIELDS[arm] for arm in ARM_NAMES)
SPEC_REFS = (
    "REQ-LEARN-5557",
    "SCENARIO-LEARN-5557-BASELINES",
    "SCENARIO-LEARN-5557-CONTROLS",
    "SCENARIO-LEARN-5557-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "upstream_csl_residue_corrigendum",
    "upstream_flagged_ablation",
    "llm_invoked",
    "no_model_specs_required",
    "best_constant_score",
    "per_query_random_score",
    "no_memory_score",
    "shuffled_memory_score",
    "aligned_memory_score",
    "aligned_delta_over_shuffled",
    "equality_tolerance",
    "duplicated_metric_pairs",
    "tautology_resolved",
    "csl_five_arm_clean",
    "adversarial_clean",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5557_csl_five_arm_tautology_corrigendum_v2.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5557_csl_five_arm_tautology_corrigendum_v2.py "
    "-m pytest tests/python/test_experiment_5557_csl_five_arm_tautology_corrigendum_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5557_csl_five_arm_tautology_corrigendum_v2.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)
FIELD_PRINCIPLES: JsonDict = {
    "upstream_csl_residue_corrigendum": "Binds the receipt to the repaired Exp5542 independent-label metric.",
    "upstream_flagged_ablation": "Preserves the Exp5543 flagged artifact as the exact upstream failure.",
    "llm_invoked": "Documents that no live model call can explain or fabricate the receipt.",
    "no_model_specs_required": "Explains why model specs are absent for a deterministic no-LLM ablation.",
    "best_constant_score": "Measures whether one fixed answer explains held-out performance.",
    "per_query_random_score": "Measures chance retrieval with query-keyed non-self random memory.",
    "no_memory_score": "Keeps retrieval benefit separate from the no-memory baseline.",
    "shuffled_memory_score": "Detects whether arbitrary memory pairing explains the lift.",
    "aligned_memory_score": "Measures query-aligned external memory against independent labels.",
    "aligned_delta_over_shuffled": "Gate delta requiring aligned memory to beat shuffled memory.",
    "equality_tolerance": "Makes duplicate-score detection deterministic and auditable.",
    "duplicated_metric_pairs": "Lists score pairs whose equality would trigger a tautology concern.",
    "tautology_resolved": "Fails when constant and random controls still collapse.",
    "csl_five_arm_clean": "Bare gate for downstream CSL tasks after controls separate.",
    "adversarial_clean": "Conductor-facing clean flag requiring no duplicate metric pairs.",
    "tests_added_or_reused": "Lists focused, coverage, and full-suite verification commands.",
    "field_principles": "Explains why each required headline and gate field exists.",
    "inference_substrate": "Declares deterministic CSL ablation with no LLM invocation.",
    "honest_verdict": "Terminal summary with complete or blocked prefix.",
}


def build_fixture() -> JsonDict:
    """Return the Exp5542 held-out labels with five separated control states."""

    upstream = exp5542.build_fixture()
    queries = [query_record(row) for row in upstream["heldout_rows"]]
    labels = {
        row["label_id"]: {
            "expected_action": row["expected_action"],
            "label_source": exp5542.INDEPENDENT_LABEL_SOURCE,
        }
        for row in upstream["heldout_rows"]
    }
    aligned_entries = [
        memory_entry("aligned", query, row["event_topic_action"])
        for query, row in zip(queries, upstream["heldout_rows"], strict=True)
    ]
    random_entries = [
        memory_entry(
            "random",
            query,
            deterministic_nonself_random_action(query, aligned_entries),
        )
        for query in queries
    ]
    return {
        "heldout_queries": queries,
        "heldout_labels": labels,
        "memory_states": {
            BEST_CONSTANT_ARM: {
                "arm": BEST_CONSTANT_ARM,
                "constant_action": best_constant_action(labels),
            },
            PER_QUERY_RANDOM_ARM: {"arm": PER_QUERY_RANDOM_ARM, "entries": random_entries},
            NO_MEMORY_ARM: {
                "arm": NO_MEMORY_ARM,
                "entries": [
                    memory_entry("no-memory", query, row["no_memory_action"])
                    for query, row in zip(queries, upstream["heldout_rows"], strict=True)
                ],
            },
            SHUFFLED_MEMORY_ARM: {
                "arm": SHUFFLED_MEMORY_ARM,
                "entries": [
                    memory_entry("shuffled", query, row["shuffled_memory_action"])
                    for query, row in zip(queries, upstream["heldout_rows"], strict=True)
                ],
            },
            ALIGNED_MEMORY_ARM: {"arm": ALIGNED_MEMORY_ARM, "entries": aligned_entries},
        },
    }


def query_record(row: Mapping[str, Any]) -> JsonDict:
    """Create one held-out query record without copying the answer label."""

    payload = {
        "task_id": row["task_id"],
        "label_id": row["label_id"],
        "query_family": row["query_family"],
    }
    return {
        "query_id": row["task_id"],
        "task_id": row["task_id"],
        "label_id": row["label_id"],
        "query_family": row["query_family"],
        "query_key": "query:" + sha256_json(payload),
    }


def memory_entry(prefix: str, query: Mapping[str, Any], selected_action: str) -> JsonDict:
    """Create one memory row keyed by query identity rather than label value."""

    return {
        "memory_id": f"{prefix}-{query['query_id']}",
        "query_key": query["query_key"],
        "selected_action": selected_action,
    }


def best_constant_action(labels: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the fixed answer with maximum support and deterministic ties."""

    counts = Counter(label["expected_action"] for label in labels.values())
    best_count = max(counts.values())
    return sorted(action for action, count in counts.items() if count == best_count)[0]


def deterministic_nonself_random_action(
    query: Mapping[str, Any],
    aligned_entries: Sequence[Mapping[str, Any]],
) -> str:
    """Pick a deterministic random memory action from rows not keyed to this query."""

    candidates = [
        entry for entry in aligned_entries if entry["query_key"] != query["query_key"]
    ]
    ranked = sorted(
        candidates,
        key=lambda entry: sha256_json(
            {
                "seed": RANDOM_SEED,
                "query_id": query["query_id"],
                "candidate_memory_id": entry["memory_id"],
            }
        ),
    )
    return str(ranked[0]["selected_action"])


def evaluate_controls(fixture: Mapping[str, Any]) -> JsonDict:
    """Score all controls and compute the duplicate-score evidence."""

    arm_results = {arm: score_arm(fixture, arm) for arm in ARM_NAMES}
    scores = {
        ARM_SCORE_FIELDS[arm]: score_rows(rows) for arm, rows in arm_results.items()
    }
    duplicates = duplicated_metric_pairs(scores, EQUALITY_TOLERANCE)
    aligned_delta = _round(
        scores["aligned_memory_score"] - scores["shuffled_memory_score"]
    )
    return {
        "arm_results": arm_results,
        "scores": scores,
        "shared_query_ids": [query["query_id"] for query in fixture["heldout_queries"]],
        "same_heldout_query_set": same_heldout_query_set(arm_results),
        "query_hashes": [hash_state(query) for query in fixture["heldout_queries"]],
        "memory_hashes": {
            arm: hash_state(fixture["memory_states"][arm]) for arm in ARM_NAMES
        },
        "aligned_delta_over_shuffled": aligned_delta,
        "duplicated_metric_pairs": duplicates,
        "tautology_resolved": tautology_resolved(duplicates),
    }


def score_arm(fixture: Mapping[str, Any], arm: str) -> list[JsonDict]:
    """Return exact-label outcomes for one control arm."""

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
    """Select the action proposed by one control arm for one query."""

    state = fixture["memory_states"][arm]
    if arm == BEST_CONSTANT_ARM:
        return str(state["constant_action"])
    return memory_action_lookup(state, query["query_key"])


def memory_action_lookup(state: Mapping[str, Any], query_key: str) -> str:
    """Find a selected action by query key."""

    for entry in state["entries"]:
        if entry["query_key"] == query_key:
            return str(entry["selected_action"])
    raise KeyError(query_key)  # pragma: no cover - fixture construction covers all keys.


def score_rows(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return a rounded exact-label pass rate for row evidence."""

    return _round(sum(1 for row in rows if row["accepted"]) / len(rows))


def fixture_is_non_degenerate(fixture: Mapping[str, Any]) -> bool:
    """Confirm the fixture has more than one label and more than one answer."""

    labels = fixture["heldout_labels"].values()
    expected_actions = {label["expected_action"] for label in labels}
    return len(fixture["heldout_queries"]) > 1 and len(expected_actions) > 1


def same_heldout_query_set(arm_results: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    """Check that every control scored the same query IDs in the same order."""

    query_sets = [[row["query_id"] for row in arm_results.get(arm, [])] for arm in ARM_NAMES]
    return bool(query_sets) and all(query_ids == query_sets[0] for query_ids in query_sets)


def duplicated_metric_pairs(
    scores: Mapping[str, Any],
    tolerance: float,
) -> list[JsonDict]:
    """Return all headline score pairs equal within the configured tolerance."""

    pairs: list[JsonDict] = []
    for left, right in itertools.combinations(HEADLINE_SCORE_FIELDS, 2):
        left_score = float(scores[left])
        right_score = float(scores[right])
        delta = _round(abs(left_score - right_score))
        if delta <= tolerance:
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "left_score": _round(left_score),
                    "right_score": _round(right_score),
                    "delta_abs": delta,
                    "equality_tolerance": tolerance,
                }
            )
    return pairs


def tautology_resolved(duplicates: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when no headline score pair is duplicated."""

    return len(duplicates) == 0


def upstream_residue_status(root: Path | str) -> JsonDict:
    """Load the Exp5542 repaired residue gate."""

    try:
        artifact = load_json(Path(root) / UPSTREAM_CSL_RESIDUE_CORRIGENDUM)
    except (OSError, json.JSONDecodeError):
        return {
            "path": UPSTREAM_CSL_RESIDUE_CORRIGENDUM.as_posix(),
            "loadable": False,
            "csl_residue_tautology_resolved": False,
            "csl_residue_stress_ready": False,
        }
    return {
        "path": UPSTREAM_CSL_RESIDUE_CORRIGENDUM.as_posix(),
        "loadable": True,
        "csl_residue_tautology_resolved": artifact.get(
            "csl_residue_tautology_resolved"
        )
        is True,
        "csl_residue_stress_ready": artifact.get("csl_residue_stress_ready") is True,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
    }


def upstream_flagged_ablation_status(root: Path | str) -> JsonDict:
    """Load Exp5543 and record whether its specific tautology was observed."""

    try:
        artifact = load_json(Path(root) / UPSTREAM_FLAGGED_ABLATION)
    except (OSError, json.JSONDecodeError):
        return {
            "path": UPSTREAM_FLAGGED_ABLATION.as_posix(),
            "loadable": False,
            "flagged_adversarial": False,
            "tautology_pair_observed": False,
        }
    best = float(artifact.get("best_constant_score", -1.0))
    random = float(artifact.get("per_query_random_score", -2.0))
    details = " ".join(
        str(item.get("detail", "")) for item in artifact.get("corrigendum_pending", [])
    )
    return {
        "path": UPSTREAM_FLAGGED_ABLATION.as_posix(),
        "loadable": True,
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "tautology_pair_observed": (
            abs(best - random) <= EQUALITY_TOLERANCE
            and "best_constant_score" in details
            and "per_query_random_score" in details
        ),
        "best_constant_score": _round(best),
        "per_query_random_score": _round(random),
        "csl_five_arm_ready": artifact.get("csl_five_arm_ready") is True,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
    }


def build_artifact(*, root: Path | str, tests_added_or_reused: Sequence[str]) -> JsonDict:
    """Build and validate the complete Exp5557 receipt."""

    root_path = Path(root)
    residue = upstream_residue_status(root_path)
    flagged = upstream_flagged_ablation_status(root_path)
    fixture = build_fixture()
    evaluation = evaluate_controls(fixture)
    scores = evaluation["scores"]
    llm_invoked = False
    no_model_specs_required = True
    clean = (
        residue["csl_residue_tautology_resolved"]
        and flagged["tautology_pair_observed"]
        and evaluation["same_heldout_query_set"]
        and evaluation["tautology_resolved"]
        and evaluation["aligned_delta_over_shuffled"] > 0.0
        and not llm_invoked
        and no_model_specs_required
    )
    artifact: JsonDict = {
        "experiment": 5557,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "upstream_csl_residue_corrigendum": UPSTREAM_CSL_RESIDUE_CORRIGENDUM.as_posix(),
        "upstream_flagged_ablation": UPSTREAM_FLAGGED_ABLATION.as_posix(),
        "upstream_residue_status": residue,
        "upstream_flagged_ablation_status": flagged,
        "llm_invoked": llm_invoked,
        "no_model_specs_required": no_model_specs_required,
        "heldout_queries": fixture["heldout_queries"],
        "shared_query_ids": evaluation["shared_query_ids"],
        "same_heldout_query_set": evaluation["same_heldout_query_set"],
        "arm_results": evaluation["arm_results"],
        "query_hashes": evaluation["query_hashes"],
        "memory_hashes": evaluation["memory_hashes"],
        "best_constant_score": scores["best_constant_score"],
        "per_query_random_score": scores["per_query_random_score"],
        "no_memory_score": scores["no_memory_score"],
        "shuffled_memory_score": scores["shuffled_memory_score"],
        "aligned_memory_score": scores["aligned_memory_score"],
        "aligned_delta_over_shuffled": evaluation["aligned_delta_over_shuffled"],
        "equality_tolerance": EQUALITY_TOLERANCE,
        "duplicated_metric_pairs": evaluation["duplicated_metric_pairs"],
        "tautology_resolved": evaluation["tautology_resolved"],
        "csl_five_arm_clean": clean,
        "adversarial_clean": clean,
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_note": (
            "All scores are deterministic fixture evaluations against Exp5542 "
            "independent labels; no model is loaded or invoked."
        ),
        "exp5543_scoring_path_diagnosis": (
            "Exp5543 used a fixed constant answer and query-keyed random memory, "
            "but both controls scored 1/12 on the fixture; Exp5557 excludes the "
            "query's own aligned memory row from the random control and rejects "
            "any remaining duplicate score pairs."
        ),
        "honest_verdict": "",
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
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
    """Build the artifact and optionally write stable JSON."""

    root_path = Path(root)
    artifact = build_artifact(
        root=root_path,
        tests_added_or_reused=tests_added_or_reused,
    )
    if write:
        write_json(_resolve_path(root_path, result_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5557 artifact is internally inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5557 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("upstream_csl_residue_corrigendum") != UPSTREAM_CSL_RESIDUE_CORRIGENDUM.as_posix():
        errors.append("upstream_csl_residue_corrigendum")
    if artifact.get("upstream_flagged_ablation") != UPSTREAM_FLAGGED_ABLATION.as_posix():
        errors.append("upstream_flagged_ablation")
    if artifact.get("llm_invoked") is not False:
        errors.append("llm_invoked")
    if artifact.get("no_model_specs_required") is not True:
        errors.append("no_model_specs_required")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")

    arm_results = artifact.get("arm_results", {})
    same_queries = same_heldout_query_set(arm_results) if isinstance(arm_results, Mapping) else False
    if artifact.get("same_heldout_query_set") is not same_queries:
        errors.append("same_heldout_query_set")

    computed_scores = arm_scores_from_artifact(arm_results)
    for field, score in computed_scores.items():
        if float(artifact.get(field, -1.0)) != score:
            errors.append(field)

    scores = headline_scores_from_artifact(artifact)
    expected_delta = _round(scores["aligned_memory_score"] - scores["shuffled_memory_score"])
    if float(artifact.get("aligned_delta_over_shuffled", -1.0)) != expected_delta:
        errors.append("aligned_delta_over_shuffled")
    if float(artifact.get("equality_tolerance", -1.0)) != EQUALITY_TOLERANCE:
        errors.append("equality_tolerance")

    expected_duplicates = duplicated_metric_pairs(scores, EQUALITY_TOLERANCE)
    if artifact.get("duplicated_metric_pairs") != expected_duplicates:
        errors.append("duplicated_metric_pairs")
    expected_tautology_resolved = tautology_resolved(expected_duplicates)
    if artifact.get("tautology_resolved") is not expected_tautology_resolved:
        errors.append("tautology_resolved")

    expected_query_hashes = [hash_state(query) for query in artifact.get("heldout_queries", [])]
    if artifact.get("query_hashes") != expected_query_hashes:
        errors.append("query_hashes")

    residue = artifact.get("upstream_residue_status", {})
    flagged = artifact.get("upstream_flagged_ablation_status", {})
    expected_clean = (
        isinstance(residue, Mapping)
        and isinstance(flagged, Mapping)
        and residue.get("csl_residue_tautology_resolved") is True
        and flagged.get("tautology_pair_observed") is True
        and same_queries
        and expected_tautology_resolved
        and expected_delta > 0.0
        and artifact.get("llm_invoked") is False
        and artifact.get("no_model_specs_required") is True
    )
    if artifact.get("csl_five_arm_clean") is not expected_clean:
        errors.append("csl_five_arm_clean")
    if artifact.get("adversarial_clean") is not expected_clean:
        errors.append("adversarial_clean")

    principles = artifact.get("field_principles", {})
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)]
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def arm_scores_from_artifact(arm_results: Any) -> JsonDict:
    """Recompute score fields from row evidence."""

    if not isinstance(arm_results, Mapping):
        return {}
    return {
        ARM_SCORE_FIELDS[arm]: score_rows(rows)
        for arm in ARM_NAMES
        if isinstance((rows := arm_results.get(arm)), Sequence) and rows
    }


def headline_scores_from_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Read headline score fields from an artifact-like mapping."""

    return {field: float(artifact[field]) for field in HEADLINE_SCORE_FIELDS}


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict from the clean gate."""

    if artifact.get("csl_five_arm_clean") is True and artifact.get("adversarial_clean") is True:
        return "complete: csl_five_arm_tautology_corrigendum_v2_clean"
    return "blocked: csl_five_arm_tautology_corrigendum_v2_not_clean"


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
    """Write stable JSON for diffable receipts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {
        key: value for key, value in artifact.items() if key != "reproducibility_checksum"
    }
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the source files backing the receipt."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def sha256_file(path: Path | str) -> str:
    """Return a SHA256 digest for one file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a SHA256 digest for a JSON-compatible mapping."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_state(payload: Mapping[str, Any]) -> str:
    """Return a prefixed content hash for fixture evidence."""

    return "sha256:" + sha256_json(payload)


def _round(value: float) -> float:
    """Round metric values once so JSON stays stable across reruns."""

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
