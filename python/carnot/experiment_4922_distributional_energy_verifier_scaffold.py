"""Exp 4922 distributional-energy verifier pivot scaffold.

Spec refs: REQ-KONA-4922, SCENARIO-KONA-4922-DRY-RUN,
SCENARIO-KONA-4922-BLOCKED, SCENARIO-KONA-4922-NO-WIN-CLAIM.

This module is an offline de-risking scaffold. It ports the FoVer harness shape
to a tiny TravelPlanner-style structured-reasoning slice, emits the three
comparison columns required by the post-sprint pivot, and stops short of any
verifier win claim. All scoring is against cached candidates; no live model is
loaded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]
JsonDict = dict[str, Any]

EXPERIMENT_ID = 4922
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4922_distributional_energy_verifier_scaffold.json"
HARNESS_SKELETON_PATH = "python/carnot/experiment_4922_distributional_energy_verifier_scaffold.py"
DEFAULT_DOMAIN_SLICE_RELATIVE_PATH = "data/experiment_4922_travelplanner_structured_slice.jsonl"
FOVER_RUNBOOK_RELATIVE_PATH = "ops/reproduction-runbook-fover-headline.md"
FOVER_HARNESS_RELATIVE_PATH = "python/carnot/eval/fover_memory_leakage_v3.py"
EXP4911_PIVOT_RELATIVE_PATH = "results/experiment_4911_sota_ingestion_v453_frontier.json"
RESULT_PATH = REPO_ROOT / RESULT_RELATIVE_PATH
DEFAULT_DOMAIN_SLICE_PATH = REPO_ROOT / DEFAULT_DOMAIN_SLICE_RELATIVE_PATH
ARXIV_ID = "2605.18871"
ARXIV_URL = "https://arxiv.org/abs/2605.18871"
HONEST_VERDICT = "success_distributional_energy_verifier_pivot_scaffolded"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 20260628
DURATION_S = 1.0
SELF_CONSISTENCY_SATURATION_THRESHOLD = 0.90
THREE_COMPARISON_COLUMNS = (
    "distributional_energy_verifier",
    "self_consistency",
    "llm_judge",
)
TERMINAL_PREFIXES = (
    "blocked_",
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_USER_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; success_distributional_energy_verifier_pivot_scaffolded."
    },
    "pivot_executable_on_6_30": {
        "principle": (
            "true -- the harness skeleton + dry-run make the post-sprint verifier-moat pivot "
            "executable the instant the sprint retires."
        )
    },
    "harness_skeleton_path": {
        "principle": (
            "the offline FoVer->non-saturated-domain harness skeleton (the de-risking deliverable)."
        )
    },
    "dry_run_three_columns": {
        "principle": (
            "the tiny-slice dry-run output: {distributional_energy_verifier, self_consistency, "
            "llm_judge} columns -- proves the harness runs, NOT a headline."
        )
    },
    "validation_gate": {
        "principle": (
            "the gate the real post-6/30 experiment must pass: beats SC CI95-excl-0, "
            "no model-identity shortcut, oracle-distinct."
        )
    },
    "arxiv_id_cited": {
        "principle": "2605.18871 (HTTP-200 verified by exp4911) -- no fabrication."
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the distributional energy verifier is oracle-distinct (the moat domain has "
            "no cheap executable oracle)."
        )
    },
    "self_consistency_saturated": {
        "principle": (
            "false -- the moat only exists where self-consistency is NOT near-ceiling "
            "(the domain choice)."
        )
    },
    "no_verifier_win_claimed": {
        "principle": (
            "true -- this is a SCAFFOLD + dry-run; the win is claimed only by the real "
            "post-6/30 experiment that passes the validation gate."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates (scores cached rows in the dry-run; "
            "1s floor)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records FoVer-harness + domain-slice presence; a missing resource emits blocked_."
        )
    },
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "comparison_stubbed": {
        "principle": (
            "true -- exp4922 proves harness wiring only; the real comparison is post-6/30."
        )
    },
    "domain_slice_path": {
        "principle": "points to the tiny TravelPlanner-style cached structured-reasoning slice."
    },
    "fover_harness_sources": {
        "principle": "records the FoVer runbook and scorer source this skeleton ports from."
    },
    "no_headline_claim": {
        "principle": "true -- the dry-run is not promoted into a result headline."
    },
    "random_seed": {
        "principle": "determinism for dry-run row ordering and checksum construction."
    },
    "reproducibility_checksum": {
        "principle": "content hash of the cached slice, comparison columns, and guardrails."
    },
    "duration_s": {
        "principle": "1.0s cached-scoring floor for verifier_ensemble_against_cached_candidates."
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "pivot_executable_on_6_30",
    "harness_skeleton_path",
    "dry_run_three_columns",
    "validation_gate",
    "arxiv_id_cited",
    "verifier_is_oracle",
    "self_consistency_saturated",
    "no_verifier_win_claimed",
    "inference_substrate",
    "preconditions_checked",
    "comparison_stubbed",
    "domain_slice_path",
    "fover_harness_sources",
    "no_headline_claim",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

VALIDATION_GATE: JsonDict = {
    "real_post_6_30_experiment_must_pass": (
        "distributional_energy_verifier beats self-consistency with CI95 excluding zero"
    ),
    "ci95_excludes_zero_required": True,
    "adversarial_verify_no_model_identity_shortcut_required": True,
    "oracle_distinct_required": True,
    "oracle_distinct_note": "no cheap executable oracle for the evaluated domain",
    "promotion_note": (
        "the dry-run is not a headline; promotion requires the real post-6/30 experiment"
    ),
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _path_present(repo_root: Path, relative_path: str) -> bool:
    return (repo_root / relative_path).exists()


def _candidate_id(candidate: JsonMap) -> str:
    return str(candidate.get("candidate_id") or "")


def _require_candidates(candidates: Sequence[JsonMap]) -> Sequence[JsonMap]:
    if not candidates:
        raise ValueError("candidate list is empty")
    return candidates


def distributional_energy(candidate: JsonMap) -> float:
    """Return the scaffold's distributional energy for one cached candidate.

    Lower is better. The stub uses exactly the three pieces required by
    arXiv:2605.18871-style structured verification: learned quality,
    deterministic constraint penalty, and uncertainty. It intentionally ignores
    `model_id`, answer-key labels, and any executable oracle.
    """

    learned_quality = float(candidate.get("learned_quality_mean", 0.0))
    constraint_penalty = float(candidate.get("deterministic_constraint_penalty", 0.0))
    uncertainty = float(candidate.get("uncertainty", 0.0))
    return -learned_quality + constraint_penalty + uncertainty


def select_distributional_energy(candidates: Sequence[JsonMap]) -> JsonMap:
    """Select the lowest-energy cached candidate."""

    checked = _require_candidates(candidates)
    return min(checked, key=lambda candidate: (distributional_energy(candidate), _candidate_id(candidate)))


def select_self_consistency(candidates: Sequence[JsonMap]) -> JsonMap:
    """Select the self-consistency majority answer from cached sample counts."""

    checked = _require_candidates(candidates)
    counts = Counter(str(candidate.get("answer")) for candidate in checked)
    best_answer, _count = counts.most_common(1)[0]
    for candidate in checked:
        if str(candidate.get("answer")) == best_answer:
            return candidate
    raise ValueError("self-consistency majority answer not found")  # pragma: no cover


def select_llm_judge(candidates: Sequence[JsonMap]) -> JsonMap:
    """Select the cached LLM-judge stub's highest-scored candidate."""

    checked = _require_candidates(candidates)
    return max(checked, key=lambda candidate: (float(candidate.get("llm_judge_score", 0.0)), _candidate_id(candidate)))


def _summary(candidate: JsonMap, *, score_name: str, score: float) -> JsonDict:
    return {
        "selected_candidate_id": _candidate_id(candidate),
        "answer": str(candidate.get("answer") or ""),
        score_name: round(float(score), 6),
        "stubbed": True,
    }


def score_cached_row(row: JsonMap) -> JsonDict:
    """Score one structured-reasoning row with the three pivot columns."""

    candidates = list(row.get("candidates") or [])
    energy_candidate = select_distributional_energy(candidates)
    sc_candidate = select_self_consistency(candidates)
    judge_candidate = select_llm_judge(candidates)
    return {
        "problem_id": str(row.get("problem_id") or ""),
        "distributional_energy_verifier": _summary(
            energy_candidate,
            score_name="energy",
            score=distributional_energy(energy_candidate),
        ),
        "self_consistency": _summary(
            sc_candidate,
            score_name="vote_count",
            score=float(sc_candidate.get("sample_count", 0.0)),
        ),
        "llm_judge": _summary(
            judge_candidate,
            score_name="judge_score",
            score=float(judge_candidate.get("llm_judge_score", 0.0)),
        ),
    }


def run_dry_run(rows: Sequence[JsonMap], *, limit: int = 3) -> JsonDict:
    """Run the tiny cached dry-run and return the required three-column table."""

    selected_rows = list(rows)[:limit]
    return {
        "columns": list(THREE_COMPARISON_COLUMNS),
        "n_rows": len(selected_rows),
        "rows": [score_cached_row(row) for row in selected_rows],
        "dry_run_note": (
            "SCAFFOLD + dry-run only; emits the three comparison columns and is NOT a headline."
        ),
    }


def validate_rows(rows: Sequence[JsonMap]) -> None:
    """Validate the minimal cached-slice schema used by the scaffold."""

    if not rows:
        raise ValueError("domain slice has no rows")
    for row in rows:
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"row {row.get('problem_id')} has no candidates")
        if row.get("cheap_executable_oracle_available") is not False:
            raise ValueError(f"row {row.get('problem_id')} is not oracle-distinct")
        if not any("label_correct" in candidate for candidate in candidates):
            raise ValueError(f"row {row.get('problem_id')} lacks cached labels")


def load_domain_slice(path: Path) -> list[JsonDict]:
    """Load the tiny cached structured-reasoning slice from JSONL."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    validate_rows(rows)
    return rows


def self_consistency_accuracy(rows: Sequence[JsonMap]) -> float:
    """Compute the dry-run self-consistency accuracy for saturation gating."""

    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        selected = select_self_consistency(list(row.get("candidates") or []))
        correct += int(bool(selected.get("label_correct")))
    return correct / len(rows)


def check_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    domain_slice_path: Path = DEFAULT_DOMAIN_SLICE_PATH,
) -> JsonDict:
    """Record FoVer-harness and domain-slice availability before scoring."""

    fover_sources = [
        {"path": FOVER_RUNBOOK_RELATIVE_PATH, "present": _path_present(repo_root, FOVER_RUNBOOK_RELATIVE_PATH)},
        {"path": FOVER_HARNESS_RELATIVE_PATH, "present": _path_present(repo_root, FOVER_HARNESS_RELATIVE_PATH)},
    ]
    fover_present = all(source["present"] for source in fover_sources)
    domain_present = domain_slice_path.exists()
    rows: list[JsonDict] = []
    domain_valid = False
    domain_error = None
    sc_accuracy = None
    if domain_present:
        try:
            rows = load_domain_slice(domain_slice_path)
            domain_valid = True
            sc_accuracy = self_consistency_accuracy(rows)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            domain_error = str(exc)
    domain_saturated = (
        bool(sc_accuracy is not None and sc_accuracy >= SELF_CONSISTENCY_SATURATION_THRESHOLD)
    )
    return {
        "fover_harness_present": fover_present,
        "fover_harness_sources": fover_sources,
        "exp4911_pivot_map_present": _path_present(repo_root, EXP4911_PIVOT_RELATIVE_PATH),
        "domain_slice_present": domain_present,
        "domain_slice_path": domain_slice_path.as_posix(),
        "domain_slice_valid": domain_valid,
        "domain_slice_rows": len(rows),
        "domain_slice_non_saturated": bool(domain_valid and not domain_saturated),
        "self_consistency_dry_run_accuracy": sc_accuracy,
        "self_consistency_saturation_threshold": SELF_CONSISTENCY_SATURATION_THRESHOLD,
        "domain_error": domain_error,
        "blocked_resource": blocked_resource_from_preconditions(
            fover_present=fover_present,
            domain_present=domain_present,
            domain_valid=domain_valid,
            domain_saturated=domain_saturated,
        ),
    }


def blocked_resource_from_preconditions(
    *,
    fover_present: bool,
    domain_present: bool,
    domain_valid: bool,
    domain_saturated: bool,
) -> str | None:
    if not fover_present:
        return "fover_harness"
    if not domain_present:
        return "domain_slice"
    if not domain_valid:
        return "domain_slice_invalid"
    if domain_saturated:
        return "self_consistency_saturated"
    return None


def _checksum(
    *,
    preconditions: JsonMap,
    rows: Sequence[JsonMap],
    blocked_resource: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _json_dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "arxiv_id": ARXIV_ID,
                "columns": THREE_COMPARISON_COLUMNS,
                "blocked_resource": blocked_resource,
                "preconditions": preconditions,
                "rows": list(rows),
                "no_verifier_win_claimed": True,
            }
        ).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def _empty_dry_run() -> JsonDict:
    return {
        "columns": list(THREE_COMPARISON_COLUMNS),
        "n_rows": 0,
        "rows": [],
        "dry_run_note": "blocked before dry-run; no verifier win claimed.",
    }


def build_blocked_artifact(preconditions: JsonMap, *, blocked_resource: str) -> JsonDict:
    """Build a terminal blocked artifact that still carries all required fields."""

    return {
        "honest_verdict": f"blocked_{blocked_resource}_missing"
        if blocked_resource in {"fover_harness", "domain_slice"}
        else f"blocked_{blocked_resource}",
        "pivot_executable_on_6_30": False,
        "harness_skeleton_path": HARNESS_SKELETON_PATH,
        "dry_run_three_columns": _empty_dry_run(),
        "validation_gate": dict(VALIDATION_GATE),
        "arxiv_id_cited": ARXIV_ID,
        "verifier_is_oracle": False,
        "self_consistency_saturated": False,
        "no_verifier_win_claimed": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions),
        "comparison_stubbed": True,
        "domain_slice_path": str(preconditions.get("domain_slice_path") or ""),
        "fover_harness_sources": list(preconditions.get("fover_harness_sources") or []),
        "no_headline_claim": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _checksum(
            preconditions=preconditions,
            rows=[],
            blocked_resource=blocked_resource,
        ),
        "duration_s": DURATION_S,
    }


def build_success_artifact(
    *,
    rows: Sequence[JsonMap],
    preconditions: JsonMap,
) -> JsonDict:
    """Build the successful scaffold artifact from validated cached rows."""

    return {
        "honest_verdict": HONEST_VERDICT,
        "pivot_executable_on_6_30": True,
        "harness_skeleton_path": HARNESS_SKELETON_PATH,
        "dry_run_three_columns": run_dry_run(rows),
        "validation_gate": dict(VALIDATION_GATE),
        "arxiv_id_cited": ARXIV_ID,
        "verifier_is_oracle": False,
        "self_consistency_saturated": False,
        "no_verifier_win_claimed": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions),
        "comparison_stubbed": True,
        "domain_slice_path": str(preconditions.get("domain_slice_path") or DEFAULT_DOMAIN_SLICE_RELATIVE_PATH),
        "fover_harness_sources": list(preconditions.get("fover_harness_sources") or []),
        "no_headline_claim": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _checksum(
            preconditions=preconditions,
            rows=rows,
            blocked_resource=None,
        ),
        "duration_s": DURATION_S,
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    domain_slice_path: Path = DEFAULT_DOMAIN_SLICE_PATH,
) -> JsonDict:
    """Build either the successful scaffold artifact or a blocked artifact."""

    preconditions = check_preconditions(
        repo_root=repo_root,
        domain_slice_path=domain_slice_path,
    )
    blocked_resource = preconditions["blocked_resource"]
    if blocked_resource is not None:
        return build_blocked_artifact(preconditions, blocked_resource=str(blocked_resource))
    rows = load_domain_slice(domain_slice_path)
    return build_success_artifact(rows=rows, preconditions=preconditions)


def validate_artifact(artifact: JsonMap) -> None:
    """Fail closed when an artifact violates the scaffold guardrails."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    extra = set(artifact) - set(REQUIRED_ARTIFACT_FIELDS)
    if missing or extra:
        raise ValueError(f"artifact fields mismatch missing={sorted(missing)} extra={sorted(extra)}")
    verdict = str(artifact["honest_verdict"])
    if not any(verdict.startswith(prefix) for prefix in TERMINAL_PREFIXES):
        raise ValueError("honest_verdict lacks terminal prefix")
    if artifact["arxiv_id_cited"] != ARXIV_ID:
        raise ValueError("arxiv_id_cited must be 2605.18871")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact["self_consistency_saturated"] is not False:
        raise ValueError("self_consistency_saturated must be false")
    if artifact["no_verifier_win_claimed"] is not True:
        raise ValueError("no_verifier_win_claimed must be true")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be verifier_ensemble_against_cached_candidates")
    dry_run = artifact["dry_run_three_columns"]
    if not isinstance(dry_run, Mapping) or dry_run.get("columns") != list(THREE_COMPARISON_COLUMNS):
        raise ValueError("dry_run_three_columns columns must match the three comparison columns")
    gate = artifact["validation_gate"]
    if not isinstance(gate, Mapping) or gate.get("ci95_excludes_zero_required") is not True:
        raise ValueError("validation_gate must require CI95 excluding zero")
    if gate.get("adversarial_verify_no_model_identity_shortcut_required") is not True:
        raise ValueError("validation_gate must require adversarial_verify no model-identity shortcut")
    if gate.get("oracle_distinct_required") is not True:
        raise ValueError("validation_gate must require oracle-distinct evaluation")
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or set(REQUIRED_USER_FIELD_PRINCIPLES) - set(principles):
        raise ValueError("field_principles missing required user fields")


def write_artifact(artifact: JsonMap, path: Path = RESULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(
    *,
    repo_root: Path = REPO_ROOT,
    domain_slice_path: Path = DEFAULT_DOMAIN_SLICE_PATH,
    result_path: Path = RESULT_PATH,
) -> JsonDict:
    artifact = build_artifact(repo_root=repo_root, domain_slice_path=domain_slice_path)
    validate_artifact(artifact)
    write_artifact(artifact, result_path)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    main()
