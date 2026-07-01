"""Exp 5111 blocked artifact for the retracted FoVer in-domain pool task.

Spec refs: REQ-REPORT-5111, SCENARIO-REPORT-5111,
SCENARIO-REPORT-5111-BLOCKED-MISSING-CORRECTION.

The original task asked for an in-domain FoVer candidate-selection pool. The
active correction says that premise was false: FoVer's real corpus is a flat
step-correctness classification dataset, not a natural K-candidate selection
dataset. This module therefore preserves the corrected FoVer verifier-value
answer and writes a blocked artifact instead of fabricating headroom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5111_fover_in_domain_pool_v469.json"
CORRECTED_RESULT_RELATIVE_PATH = "results/experiment_fover_stepverifier_vs_cheap_baseline.json"
KNOWN_ISSUES_RELATIVE_PATH = "ops/known-issues.md"
EXPERIMENT_ID = "exp5111-fover-in-domain-pool-v469"
MILESTONE = "2026.07.469"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
BLOCKED_VERDICT = (
    "blocked_fover_indomain_pool_retracted_see_experiment_fover_stepverifier_vs_cheap_baseline"
)
MISSING_CORRECTED_RESULT_VERDICT = (
    "blocked_fover_indomain_pool_retracted_see_missing_corrected_result"
)
TERMINAL_PREFIXES = ("blocked_", "complete_", "success_", "partial_", "passed_")
RETRACTION_SOURCES = [
    "ops/known-issues.md#NUDGE-2026-07-01-RETRACTED",
    "ops/known-issues.md#MOAT-REDIRECT-2026-06-30-RETRACTED",
]
RETRACTION_MARKERS = (
    "NUDGE 2026-07-01",
    "MOAT REDIRECT 2026-06-30",
    "RETRACTED",
    "construction artifact",
    "no natural multi-candidate structure",
)
REQUIRED_CORRECTED_RESULT_FIELDS = frozenset(
    {
        "n_rows",
        "verifier_auroc",
        "cheap_baseline_auroc",
        "delta_auroc",
        "delta_auroc_ci95",
        "beats_cheap_baseline",
        "framing_change_from_retracted_claim",
        "length_confound",
        "model_specs",
        "random_seed",
        "reproducibility_checksum",
    }
)
REQUIRED_USER_ARTIFACT_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "preconditions_checked",
        "pool_path",
        "pool_sha256",
        "pool_n",
        "candidates_per_item",
        "vote_at_1",
        "tuned_self_consistency",
        "oracle_at_k",
        "headroom_present",
        "verifier_is_oracle",
        "model_specs",
        "seeds_or_checksums",
        "flagged_adversarial",
        "tests_run",
    }
)
EXTRA_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "corrected_result_path",
        "corrected_result_sha256",
        "corrected_result_summary",
        "retraction_sources",
        "retracted_claims",
        "run_date",
    }
)
REQUIRED_ARTIFACT_FIELDS = REQUIRED_USER_ARTIFACT_FIELDS | EXTRA_ARTIFACT_FIELDS
FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "preconditions_checked": "compute/data preflight accountability",
    "pool_path": "deliverable reproducibility",
    "pool_sha256": "data integrity",
    "pool_n": "downstream gate signal",
    "candidates_per_item": "selection-headroom clarity",
    "vote_at_1": "baseline transparency",
    "tuned_self_consistency": "fair baseline",
    "oracle_at_k": "headroom measurement",
    "headroom_present": "no no-headroom moat claims",
    "verifier_is_oracle": "oracle-distinctness",
    "model_specs": "SOTA model accountability when LLMs are used",
    "seeds_or_checksums": "reproducibility",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
    "field_principles": "principle annotations for every top-level artifact field",
    "corrected_result_path": "terminal FoVer verifier-value evidence",
    "corrected_result_sha256": "corrected-result data integrity",
    "corrected_result_summary": "no stale headroom premise",
    "retraction_sources": "retraction provenance",
    "retracted_claims": "no fabricated candidate-selection claim",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5111_fover_in_domain_pool_v469.py --date 20260701",
    '.venv/bin/pytest tests/python/test_experiment_5111_fover_in_domain_pool_v469.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run "
    "--include='/home/ianblenke/github.com/ianblenke/carnot/python/carnot/"
    "experiment_5111_fover_in_domain_pool_v469.py' -m pytest "
    'tests/python/test_experiment_5111_fover_in_domain_pool_v469.py -q -o addopts="" && '
    ".venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/python/carnot/"
    "experiment_5111_fover_in_domain_pool_v469.py' --fail-under=100 -m",
    ".venv/bin/ruff check python/carnot/experiment_5111_fover_in_domain_pool_v469.py "
    "scripts/experiment_5111_fover_in_domain_pool_v469.py "
    "tests/python/test_experiment_5111_fover_in_domain_pool_v469.py",
    ".venv/bin/ruff format --check python/carnot/experiment_5111_fover_in_domain_pool_v469.py "
    "scripts/experiment_5111_fover_in_domain_pool_v469.py "
    "tests/python/test_experiment_5111_fover_in_domain_pool_v469.py",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5111_fover_in_domain_pool_v469.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5111_fover_in_domain_pool_v469.json",
    ".venv/bin/pytest tests/python -q",
]


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_corrected_result(
    corrected_result_text: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if corrected_result_text is None:
        return None, "missing corrected result file"
    try:
        parsed = json.loads(corrected_result_text)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, "corrected result is not a JSON object"
    missing = REQUIRED_CORRECTED_RESULT_FIELDS - set(parsed)
    if missing:
        return None, f"corrected result missing fields: {sorted(missing)}"
    return parsed, None


def _known_issues_has_retraction(known_issues_text: str) -> bool:
    return all(marker in known_issues_text for marker in RETRACTION_MARKERS)


def _corrected_result_summary(corrected_result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if corrected_result is None:
        return None
    return {
        "n_rows": corrected_result["n_rows"],
        "verifier_auroc": corrected_result["verifier_auroc"],
        "cheap_baseline_auroc": corrected_result["cheap_baseline_auroc"],
        "delta_auroc": corrected_result["delta_auroc"],
        "delta_auroc_ci95": corrected_result["delta_auroc_ci95"],
        "beats_cheap_baseline": corrected_result["beats_cheap_baseline"],
        "framing_change_from_retracted_claim": corrected_result[
            "framing_change_from_retracted_claim"
        ],
        "cheap_baseline_root_cause": corrected_result["length_confound"]["interpretation"],
    }


def _model_specs(corrected_result: Mapping[str, Any] | None) -> dict[str, Any]:
    corrected_specs = corrected_result["model_specs"] if corrected_result is not None else {}
    return {
        "generative_llms_used": [],
        "corrected_result_embedding_model": corrected_specs.get("embedding_model"),
        "corrected_result_cheap_baseline_model": corrected_specs.get("cheap_baseline_model"),
    }


def _seeds_or_checksums(
    *,
    corrected_result: Mapping[str, Any] | None,
    corrected_result_sha256: str | None,
) -> dict[str, Any]:
    return {
        "corrected_result_random_seed": None
        if corrected_result is None
        else corrected_result["random_seed"],
        "corrected_result_reproducibility_checksum": (
            None if corrected_result is None else corrected_result["reproducibility_checksum"]
        ),
        "corrected_result_sha256": corrected_result_sha256,
    }


def build_artifact(
    *,
    corrected_result_text: str | None,
    known_issues_text: str,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """Build the Exp 5111 artifact without constructing a FoVer candidate pool."""

    corrected_result, parse_error = _parse_corrected_result(corrected_result_text)
    corrected_result_sha256 = (
        _sha256_text(corrected_result_text) if corrected_result_text is not None else None
    )
    known_issues_retraction_found = _known_issues_has_retraction(known_issues_text)
    honest_verdict = (
        BLOCKED_VERDICT if corrected_result is not None else MISSING_CORRECTED_RESULT_VERDICT
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "preconditions_checked": {
            "corrected_result_path": CORRECTED_RESULT_RELATIVE_PATH,
            "corrected_result_read": corrected_result is not None,
            "corrected_result_parse_error": parse_error,
            "known_issues_path": KNOWN_ISSUES_RELATIVE_PATH,
            "known_issues_retraction_found": known_issues_retraction_found,
            "candidate_pool_generation_attempted": False,
            "local_llm_generation_attempted": False,
            "pool_fabrication_blocked": True,
        },
        "pool_path": None,
        "pool_sha256": None,
        "pool_n": 0,
        "candidates_per_item": 0,
        "vote_at_1": None,
        "tuned_self_consistency": None,
        "oracle_at_k": None,
        "headroom_present": False,
        "verifier_is_oracle": False,
        "model_specs": _model_specs(corrected_result),
        "seeds_or_checksums": _seeds_or_checksums(
            corrected_result=corrected_result,
            corrected_result_sha256=corrected_result_sha256,
        ),
        "flagged_adversarial": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "corrected_result_path": CORRECTED_RESULT_RELATIVE_PATH,
        "corrected_result_sha256": corrected_result_sha256,
        "corrected_result_summary": _corrected_result_summary(corrected_result),
        "retraction_sources": RETRACTION_SOURCES,
        "retracted_claims": {
            "candidate_selection_headroom_claim_retracted": True,
            "load_fover_domain_pool_mode_formula_artifact": True,
            "real_corpus_has_no_natural_multi_candidate_structure": True,
            "synthetic_pool_must_not_be_built": True,
            "real_task_shape": "flat per-step correctness classification",
            "real_corpus_question_id_multiplicity": "6544/6546 question_ids have exactly one row",
        },
        "run_date": run_date,
    }
    validate_artifact(artifact)
    return artifact


def _verdict_has_terminal_prefix(verdict: str) -> bool:
    return verdict.startswith(TERMINAL_PREFIXES)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate that Exp 5111 remains blocked-safe after the FoVer retraction."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"required field missing: {sorted(missing)}")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    verdict = str(artifact["honest_verdict"])
    if not _verdict_has_terminal_prefix(verdict) or not verdict.startswith(
        "blocked_fover_indomain_pool_retracted_see_"
    ):
        raise ValueError("honest_verdict must preserve the FoVer pool retraction prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact["pool_path"] is not None:
        raise ValueError("pool_path must remain null for the retracted pool task")
    if artifact["pool_sha256"] is not None:
        raise ValueError("pool_sha256 must remain null for the retracted pool task")
    if artifact["pool_n"] != 0:
        raise ValueError("pool_n must remain 0 when no pool is built")
    if artifact["candidates_per_item"] != 0:
        raise ValueError("candidates_per_item must remain 0 when no pool is built")
    if any(
        artifact[key] is not None for key in ("vote_at_1", "tuned_self_consistency", "oracle_at_k")
    ):
        raise ValueError("candidate-selection metrics must remain null after the FoVer retraction")
    if artifact["headroom_present"] is not False:
        raise ValueError("headroom_present must remain false")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must remain false")
    if artifact["flagged_adversarial"] is not False:
        raise ValueError("flagged_adversarial must remain false unless an audit flags the artifact")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must cover the required artifact fields")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must record verification evidence")
    if artifact["corrected_result_path"] != CORRECTED_RESULT_RELATIVE_PATH:
        raise ValueError("corrected_result_path mismatch")

    preconditions = artifact["preconditions_checked"]
    if preconditions["candidate_pool_generation_attempted"] is not False:
        raise ValueError("candidate_pool_generation_attempted must remain false")
    if preconditions["local_llm_generation_attempted"] is not False:
        raise ValueError("local_llm_generation_attempted must remain false")
    if preconditions["pool_fabrication_blocked"] is not True:
        raise ValueError("pool_fabrication_blocked must remain true")

    retracted_claims = artifact["retracted_claims"]
    required_retraction_flags = {
        "candidate_selection_headroom_claim_retracted",
        "load_fover_domain_pool_mode_formula_artifact",
        "real_corpus_has_no_natural_multi_candidate_structure",
        "synthetic_pool_must_not_be_built",
    }
    if any(retracted_claims[key] is not True for key in required_retraction_flags):
        raise ValueError("retracted_claims must preserve the no-pool correction")

    summary = artifact["corrected_result_summary"]
    if summary is None:
        if artifact["honest_verdict"] != MISSING_CORRECTED_RESULT_VERDICT:
            raise ValueError(
                "missing corrected_result_summary requires the missing-correction verdict"
            )
    elif summary["beats_cheap_baseline"] is not False:
        raise ValueError("beats_cheap_baseline must remain false for the corrected FoVer result")


def write_artifact(
    *,
    root: Path,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """Read local inputs and write the blocked Exp 5111 JSON artifact."""

    corrected_result_path = root / CORRECTED_RESULT_RELATIVE_PATH
    known_issues_path = root / KNOWN_ISSUES_RELATIVE_PATH
    corrected_result_text = (
        corrected_result_path.read_text(encoding="utf-8")
        if corrected_result_path.exists()
        else None
    )
    known_issues_text = (
        known_issues_path.read_text(encoding="utf-8") if known_issues_path.exists() else ""
    )
    artifact = build_artifact(
        corrected_result_text=corrected_result_text,
        known_issues_text=known_issues_text,
        duration_s=duration_s,
        run_date=run_date,
        tests_run=tests_run,
    )
    artifact_path = root / RESULT_RELATIVE_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def main(
    *,
    root: Path = REPO_ROOT,
    date: str = "20260701",
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Run Exp 5111 and return the blocked artifact path."""

    start = time.perf_counter()
    run_tests = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    elapsed = time.perf_counter() - start if duration_s is None else duration_s
    write_artifact(root=root, duration_s=elapsed, run_date=date, tests_run=run_tests)
    return root / RESULT_RELATIVE_PATH


if __name__ == "__main__":  # pragma: no cover
    main()
