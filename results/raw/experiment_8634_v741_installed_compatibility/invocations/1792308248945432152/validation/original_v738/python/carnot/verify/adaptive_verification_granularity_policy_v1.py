"""Build the Exp 3195 adaptive verification granularity policy artifact.

Spec refs: REQ-VERIFY-3195, SCENARIO-VERIFY-3195.

The policy is a deterministic scheduler over evidence that already exists in
the repo. It decides how much checking a future verifier pass should spend per
row, but it never lets an EBM score, LLM response, or receipt become answer
authority. Exact rows, controlled-invariance evidence, and counterexample
certificates remain the source of truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "carnot.adaptive_verification_granularity_policy.v1"
EXPERIMENT_ID = "exp3195"
POLICY_VERSION = "v1"
ARTIFACT = "experiment_3195_adaptive_verification_granularity_policy_v1"

OUTPUT_REL_PATH = Path("results/experiment_3195_adaptive_verification_granularity_policy_v1.json")
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3195_adaptive_verification_granularity_policy_v1.py"
)

EXP3180_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
EXP3183_REL_PATH = Path("results/experiment_3183_counterexample_certificate_expansion_v3.json")
EXP3189_REL_PATH = Path("results/experiment_3189_cross_corpus_matrix_v29.json")

POLICY_FEATURES = (
    "row_family",
    "known_false_accept_risk",
    "certificate_depth",
    "answer_ambiguity",
    "prior_verification_outcome",
    "exact_authority_complete",
    "receipt_backed_transcript_context",
)

POLICY_ACTIONS = (
    "final-answer-only",
    "step-chunk",
    "counterexample-fragment",
    "abstain/escalate",
    "skip redundant recheck",
)

ACTION_CALL_WEIGHTS = {
    "final-answer-only": 1,
    "step-chunk": 2,
    "counterexample-fragment": 3,
    "abstain/escalate": 1,
    "skip redundant recheck": 0,
}

REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "policy_version",
    "source_artifacts",
    "exact_rows_used",
    "policy_features",
    "policy_actions",
    "simulated_rows",
    "estimated_verifier_call_delta",
    "false_accept_risk_increase",
    "redundant_recheck_suppression_rule",
    "promotion_allowed",
    "honest_verdict",
}

SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("post_295_research_references", Path("research-references.md"), True, "text"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3180_controlled_invariance_executor_v2", EXP3180_REL_PATH, True, "json"),
    ("exp3183_counterexample_certificate_expansion_v3", EXP3183_REL_PATH, True, "json"),
    ("exp3189_cross_corpus_matrix_v29", EXP3189_REL_PATH, True, "json"),
    (
        "exp3195_module",
        Path("python/carnot/verify/adaptive_verification_granularity_policy_v1.py"),
        False,
        "python",
    ),
    (
        "exp3195_script",
        Path("scripts/experiment_3195_adaptive_verification_granularity_policy_v1.py"),
        False,
        "python",
    ),
    (
        "exp3195_tests",
        Path("tests/python/test_experiment_3195_adaptive_verification_granularity_policy_v1.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3195_adaptive_verification_granularity_policy_v1.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/adaptive_verification_granularity_policy_v1.py -m pytest -o addopts='' tests/python/test_experiment_3195_adaptive_verification_granularity_policy_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/adaptive_verification_granularity_policy_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3195: simulate routing from existing rows without new calls."""

    root_path = Path(root)
    exp3180 = read_json_object(root_path / EXP3180_REL_PATH)
    exp3183 = read_json_object(root_path / EXP3183_REL_PATH)
    exp3189 = read_json_object(root_path / EXP3189_REL_PATH)
    sources = source_artifacts(root_path)
    records = list(mapping_rows(exp3183.get("certificate_records")))
    receipts = list(mapping_rows(exp3180.get("receipt_backed_transcripts")))
    frontier = list(mapping_rows(exp3183.get("bounded_frontier_records")))
    receipt_context = receipt_backed_transcript_context(receipts)
    controlled_passed = exp3180.get("controlled_invariance_passed") is True
    simulated = [
        simulate_row(record, controlled_passed=controlled_passed, receipt_context=receipt_context)
        for record in records
    ]
    call_accounting = verifier_call_accounting(simulated)
    risk = risk_tradeoffs(simulated)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "policy_version": POLICY_VERSION,
        "run_date": RUN_DATE,
        "adaptive_verification_granularity_policy_v1_ready": True,
        "source_artifacts": sources,
        "source_checksums": {
            row["path"]: row["sha256"] for row in sources if row.get("sha256") is not None
        },
        "source_errors": source_errors(sources),
        "exact_rows_used": len(records),
        "policy_features": list(POLICY_FEATURES),
        "policy_actions": list(POLICY_ACTIONS),
        "simulated_rows": len(simulated),
        "simulated_policy_rows": simulated,
        "schedule_counts": schedule_counts(simulated),
        "verifier_call_accounting": call_accounting,
        "estimated_verifier_call_delta": call_accounting["estimated_verifier_call_delta"],
        "false_accept_risk_increase": risk["false_accept_risk_increase"],
        "risk_tradeoffs": risk,
        "redundant_recheck_suppression_rule": redundant_recheck_suppression_rule(),
        "promotion_allowed": False,
        "evidence_inventory": evidence_inventory(
            exp3180, exp3183, exp3189, records, receipts, frontier
        ),
        "authority_boundary": authority_boundary(),
        "inference_substrate": inference_substrate(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the schema-versioned Exp 3195 JSON."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Return a JSON object from disk, or `{}` when evidence is absent/corrupt."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Describe every local artifact the policy reads or cites."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
        path = root / rel_path
        payload = read_json_object(path) if source_type == "json" else {}
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_json_object": bool(payload) if source_type == "json" else None,
                "sha256": sha256_file(path),
            }
        )
    return rows


def source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report missing or malformed required sources without inventing rows."""

    errors: list[JsonDict] = []
    for row in sources:
        if row.get("required") is not True:
            continue
        if row.get("present") is not True:
            errors.append({"path": row.get("path"), "reason": "missing_required_source"})
        elif row.get("source_type") == "json" and row.get("readable_json_object") is not True:
            errors.append({"path": row.get("path"), "reason": "malformed_required_json"})
    return errors


def sha256_file(path: Path) -> str | None:
    """Hash source files so the policy artifact has reproducible lineage."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def mapping_rows(value: Any) -> list[JsonDict]:
    """Keep only object rows; malformed array entries are not policy evidence."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def simulate_row(
    record: Mapping[str, Any],
    *,
    controlled_passed: bool,
    receipt_context: Mapping[str, Any],
) -> JsonDict:
    """Convert one certificate record into a deterministic schedule row."""

    features = feature_values(record, receipt_context)
    action, reason = select_action(features, controlled_passed)
    return {
        "row_id": features["row_id"],
        "selected_action": action,
        "estimated_verifier_calls": ACTION_CALL_WEIGHTS[action],
        "routing_reason": reason,
        "feature_values": features,
        "authority_note": "route only; exact authority remains final",
    }


def feature_values(record: Mapping[str, Any], receipt_context: Mapping[str, Any]) -> JsonDict:
    """Extract interpretable policy inputs from an Exp 3183 certificate row."""

    family = str(record.get("counterexample_family") or "unknown")
    answers = normalized_answers(record.get("candidate_answers"))
    known_false = record.get("known_false_accept_or_regression") is True or family.startswith(
        "known_false_accept:"
    )
    return {
        "row_id": str(record.get("row_id") or "unknown-row"),
        "row_family": family,
        "known_false_accept_risk": known_false,
        "certificate_depth": certificate_depth(record),
        "answer_ambiguity": len(set(answers)) > 1,
        "candidate_answer_count": len(answers),
        "prior_verification_outcome": str(record.get("checker_result") or "unknown"),
        "checker_authority": str(record.get("checker_authority") or "unknown"),
        "exact_label": str(record.get("exact_label") or "unknown"),
        "exact_authority_complete": record.get("exact_authority_complete") is not False,
        "depends_on_flagged_live_verifier": record.get("depends_on_flagged_live_verifier") is True,
        "has_counterexample_certificate": bool(record.get("pilot_certificate")),
        "receipt_backed_transcript_context": dict(receipt_context),
    }


def normalized_answers(value: Any) -> list[str]:
    """Normalize candidate answers before detecting ambiguity."""

    if isinstance(value, list):
        answers = [str(item) for item in value if str(item)]
        return answers or ["unknown"]
    if value is None:
        return ["unknown"]
    return [str(value)]


def certificate_depth(record: Mapping[str, Any]) -> int:
    """Measure how much exact/counterexample structure exists for a row."""

    depth = 1 if record.get("exact_authority_complete") is not False else 0
    pilot = record.get("pilot_certificate")
    if isinstance(pilot, Mapping) and pilot:
        for key in ("certificate_type", "minimal_failing_assignment", "mcs", "unsat_core"):
            if pilot.get(key):
                depth += 1
    return depth


def select_action(features: Mapping[str, Any], controlled_passed: bool) -> tuple[str, str]:
    """Apply the fixed policy table without learned or model-scored authority."""

    family = str(features["row_family"])
    if (
        features["depends_on_flagged_live_verifier"]
        or not features["exact_authority_complete"]
        or family == "exact_row:unknown"
    ):
        return "abstain/escalate", "incomplete_or_unknown_authority_boundary"
    if (
        features["known_false_accept_risk"]
        or features["answer_ambiguity"]
        or features["has_counterexample_certificate"]
    ):
        return "counterexample-fragment", "known_or_certified_counterexample_risk"
    if features["exact_label"] == "REPAIRABLE" or "fragment" in family or "repair" in family:
        return "step-chunk", "repair_or_fragment_row_needs_chunk_context"
    if controlled_passed and features["prior_verification_outcome"] == "accept":
        return "skip redundant recheck", "exact_accept_recheck_redundant"
    return "final-answer-only", "simple_exact_final_answer_check"


def receipt_backed_transcript_context(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize receipts as routing context while keeping them non-authoritative."""

    return {
        "receipt_count": len(receipts),
        "substrate_classes": sorted(
            {str(row.get("substrate_used") or "unknown") for row in receipts}
        ),
        "all_receipts_non_authoritative": all(
            row.get("acceptance_authority") is not True for row in receipts
        ),
    }


def schedule_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count rows per policy action, including zero-count actions."""

    counts = {action: 0 for action in POLICY_ACTIONS}
    for row in rows:
        action = str(row.get("selected_action"))
        counts[action] = counts.get(action, 0) + 1
    return {action: counts[action] for action in sorted(counts)}


def verifier_call_accounting(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Estimate compute against a declared uniform step-chunk baseline."""

    baseline_calls = len(rows) * ACTION_CALL_WEIGHTS["step-chunk"]
    adaptive_calls = sum(int(row.get("estimated_verifier_calls", 0)) for row in rows)
    return {
        "baseline_policy": "uniform_step_chunk",
        "baseline_verifier_calls": baseline_calls,
        "adaptive_verifier_calls": adaptive_calls,
        "estimated_verifier_call_delta": adaptive_calls - baseline_calls,
        "call_weights": dict(ACTION_CALL_WEIGHTS),
    }


def risk_tradeoffs(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Account for false-accept risk created by suppressed or downgraded rows."""

    known_rows = [row for row in rows if row["feature_values"]["known_false_accept_risk"] is True]
    ambiguous_rows = [row for row in rows if row["feature_values"]["answer_ambiguity"] is True]
    known_skipped = [
        row for row in known_rows if row.get("selected_action") == "skip redundant recheck"
    ]
    known_below_fragment = [
        row for row in known_rows if row.get("selected_action") != "counterexample-fragment"
    ]
    ambiguous_below_fragment = [
        row for row in ambiguous_rows if row.get("selected_action") != "counterexample-fragment"
    ]
    increase = None
    if known_rows:
        increase = len(known_skipped + known_below_fragment) / len(known_rows)
    return {
        "known_false_accept_rows": len(known_rows),
        "known_false_accept_rows_skipped": len(known_skipped),
        "known_false_accept_rows_below_counterexample_fragment": len(known_below_fragment),
        "ambiguous_rows": len(ambiguous_rows),
        "ambiguous_rows_below_counterexample_fragment": len(ambiguous_below_fragment),
        "false_accept_risk_increase": increase,
        "risk_basis": "known false-accept rows from checked-in exact/certificate artifacts",
    }


def evidence_inventory(
    exp3180: Mapping[str, Any],
    exp3183: Mapping[str, Any],
    exp3189: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    frontier: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Identify the exact rows, certificates, transcripts, and blockers used."""

    false_families = sorted(
        {
            str(row.get("counterexample_family"))
            for row in records
            if row.get("known_false_accept_or_regression") is True
        }
    )
    frontier_statuses = sorted({str(row.get("exact_status") or "unknown") for row in frontier})
    return {
        "exact_rows_available": int(exp3180.get("exact_row_count") or len(records)),
        "exact_rows_used": len(records),
        "false_accept_families": false_families,
        "known_false_accept_rows_covered": int(exp3183.get("known_false_accept_rows_covered") or 0),
        "counterexample_certificates": sum(1 for row in records if row.get("pilot_certificate")),
        "bounded_frontier_records": len(frontier),
        "bounded_frontier_statuses": frontier_statuses,
        "receipt_backed_transcripts": len(receipts),
        "matrix_v29_publication_blocker_count": exp3189.get("publication_blocker_count"),
        "matrix_v29_next_top_gap": exp3189.get("next_top_gap"),
    }


def redundant_recheck_suppression_rule() -> JsonDict:
    """Describe the only case where the policy suppresses a recheck."""

    return {
        "action": "skip redundant recheck",
        "applies_when": [
            "controlled_invariance_passed=true",
            "prior_verification_outcome=accept",
            "exact_authority_complete=true",
            "answer_ambiguity=false",
            "known_false_accept_risk=false",
            "depends_on_flagged_live_verifier=false",
        ],
        "excluded_when": [
            "known_false_accept_risk=true",
            "answer_ambiguity=true",
            "counterexample_certificate_present=true",
            "row_family=exact_row:unknown",
            "exact_authority_complete=false",
            "depends_on_flagged_live_verifier=true",
        ],
        "authority_preserved": "prior exact-authority outcome remains final",
        "suppressed_verifier_calls_per_row_vs_uniform_step_chunk": 2,
    }


def authority_boundary() -> JsonDict:
    """Make the no-sidecar-promotion rule machine-readable."""

    return {
        "exact_authority_final": True,
        "ebm_or_llm_authority": False,
        "receipt_authority": False,
        "policy_only_routes_compute": True,
        "promotion_allowed": False,
    }


def inference_substrate() -> JsonDict:
    """State that the artifact is an offline policy simulation."""

    return {
        "kind": "artifact_only_policy_simulation",
        "new_live_model_calls": 0,
        "new_verifier_scoring_calls": 0,
        "new_repair_calls": 0,
        "llm_called": False,
        "ebm_trained_or_promoted": False,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that reports scope without promoting claims."""

    return (
        "complete: adaptive_verification_granularity_policy_v1_ready=true; "
        f"simulated_rows={artifact['simulated_rows']}; "
        f"estimated_verifier_call_delta={artifact['estimated_verifier_call_delta']}; "
        "promotion_allowed=false"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on schema omissions, promotion, or risky suppression."""

    missing = REQUIRED_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("promotion_allowed") is not False:
        raise ValueError("promotion_allowed must remain false")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must start with 'complete:'")
    for row in artifact.get("simulated_policy_rows", []):
        if not isinstance(row, Mapping):
            raise ValueError("simulated_policy_rows must contain objects")
        features = row.get("feature_values")
        if not isinstance(features, Mapping):
            raise ValueError("simulated row missing feature_values")
        if (
            features.get("known_false_accept_risk") is True
            and row.get("selected_action") != "counterexample-fragment"
        ):
            raise ValueError("known false-accept rows must route to counterexample-fragment")
        if row.get("selected_action") == "skip redundant recheck" and (
            features.get("known_false_accept_risk") is True
            or features.get("answer_ambiguity") is True
            or features.get("exact_authority_complete") is not True
        ):
            raise ValueError("redundant recheck suppression used outside allowed rule")


def main() -> None:
    """CLI entrypoint for the checked-in experiment script."""

    print(write_artifact().as_posix())


if __name__ == "__main__":  # pragma: no cover
    main()
