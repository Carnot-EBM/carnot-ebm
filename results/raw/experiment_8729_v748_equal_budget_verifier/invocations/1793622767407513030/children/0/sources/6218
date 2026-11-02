"""Build the Exp 3184 repair-gate decision v4 artifact.

Spec refs: REQ-VERIFY-3184, SCENARIO-VERIFY-3184.

This module is an aggregation gate, not a repair runner. It reads the local
receipt, invariance, clean-verifier, certificate, and prior-gate artifacts and
turns those upstream facts into one machine state. The conservative rule is
simple: if a load-bearing source is absent, false, flagged, or not backed by
computed verifier metrics, live repair remains blocked.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3184_repair_gate_decision_v4"
SCHEMA = "carnot.repair_gate_decision.v4"
OUTPUT_REL_PATH = Path("results/experiment_3184_repair_gate_decision_v4.json")

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
EXP3168_REL_PATH = Path("results/experiment_3168_repair_gate_decision_v3.json")
EXP3179_REL_PATH = Path("results/experiment_3179_local_sota_receipt_smoke_v3.json")
EXP3180_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
EXP3181_REL_PATH = Path("results/experiment_3181_clean_live_sota_verifier_rerun_v10.json")
EXP3183_REL_PATH = Path("results/experiment_3183_counterexample_certificate_expansion_v3.json")

FALSE_ACCEPT_GATE = 0.10
ALLOWED_STATES = {
    "blocked_missing_artifact",
    "blocked_receipt_precondition",
    "blocked_controlled_invariance",
    "blocked_clean_verifier_flagged",
    "blocked_headline_claim_blocked",
    "blocked_false_accept_gate",
    "blocked_certificate_not_ready",
    "blocked_other",
    "unblocked_for_bounded_repair_ladder",
}
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = {
    "repair_gate_decision_v4_ready",
    "repair_gate_state",
    "unblocking_predicates",
    "blocker_reasons",
    "missing_artifacts",
    "allowed_repair_attempt_budget",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
PREDICATE_ORDER = (
    "receipt_smoke_clean_rerun_allowed",
    "controlled_invariance_passed",
    "clean_verifier_not_flagged_adversarial",
    "headline_claim_allowed_for_verifier_metrics",
    "false_accept_gate_acceptable",
    "certificate_repair_call_ready",
)
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), False, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False, "text"),
    ("verification_openspec_gate_policy", SPEC_REL_PATH, True, "text"),
    ("exp3168_prior_repair_gate_v3", EXP3168_REL_PATH, True, "json"),
    ("exp3179_receipt_smoke_v3", EXP3179_REL_PATH, True, "json"),
    ("exp3180_controlled_invariance_v2", EXP3180_REL_PATH, True, "json"),
    ("exp3181_clean_verifier_v10", EXP3181_REL_PATH, True, "json"),
    ("exp3183_certificate_expansion_v3", EXP3183_REL_PATH, True, "json"),
    ("exp3184_module", Path("python/carnot/verify/repair_gate_decision_v4.py"), False, "python"),
    (
        "exp3184_tests",
        Path("tests/python/test_experiment_3184_repair_gate_decision_v4.py"),
        False,
        "python",
    ),
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3184_repair_gate_decision_v4.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3184_repair_gate_decision_v4.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/repair_gate_decision_v4.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3184_repair_gate_decision_v4.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3184: aggregate repair unblocking predicates without inference."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    payloads = load_payloads(root_path)
    sources = source_artifacts(root_path)
    missing = missing_required_artifacts(sources)
    predicates = unblocking_predicates(payloads)
    blockers = blocker_reasons(missing, predicates)
    state = repair_gate_state(missing, predicates)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3184", "SCENARIO-VERIFY-3184"],
        "repair_gate_decision_v4_ready": True,
        "repair_gate_state": state,
        "unblocking_predicates": predicates,
        "blocker_reasons": blockers,
        "missing_artifacts": missing,
        "allowed_repair_attempt_budget": allowed_repair_attempt_budget(state),
        "source_artifacts": sources,
        "source_checksums": {
            row["path"]: row["sha256"] for row in sources if row.get("sha256") is not None
        },
        "source_gate_summary": source_gate_summary(payloads),
        "field_principles": field_principles(),
        "inference_substrate": inference_substrate(payloads),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3184 decision JSON."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_payloads(root: Path) -> dict[str, JsonDict]:
    """Load the upstream artifacts whose fields are load-bearing for the gate."""

    return {
        "exp3168": read_json_object(root / EXP3168_REL_PATH),
        "exp3179": read_json_object(root / EXP3179_REL_PATH),
        "exp3180": read_json_object(root / EXP3180_REL_PATH),
        "exp3181": read_json_object(root / EXP3181_REL_PATH),
        "exp3183": read_json_object(root / EXP3183_REL_PATH),
    }


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, returning empty evidence when the file is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance rows for every policy or artifact file consulted."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_SPECS:
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


def missing_required_artifacts(sources: Sequence[Mapping[str, Any]]) -> list[str]:
    """List required sources that are absent or malformed instead of inferring them."""

    missing: list[str] = []
    for row in sources:
        if row.get("required") is not True:
            continue
        malformed_json = row.get("source_type") == "json" and row.get("readable_json_object") is not True
        if row.get("present") is not True or malformed_json:
            missing.append(str(row.get("path") or ""))
    return sorted(missing)


def unblocking_predicates(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Build auditable predicate rows in the exact gate order."""

    receipt = payloads["exp3179"]
    invariance = payloads["exp3180"]
    clean = payloads["exp3181"]
    certificate = payloads["exp3183"]
    false_accept_rate = finite_rate(clean.get("false_accept_rate"))
    false_accept_passed = bool(
        clean.get("clean_live_sota_verifier_rerun_v10_ready") is True
        and clean.get("gated_skip") is not True
        and clean.get("metrics_computed") is True
        and false_accept_rate is not None
        and false_accept_rate <= FALSE_ACCEPT_GATE
        and not clean.get("known_false_accepts_accepted")
    )
    return {
        "receipt_smoke_clean_rerun_allowed": predicate_row(
            source=EXP3179_REL_PATH,
            field="clean_rerun_allowed",
            expected=True,
            actual=receipt.get("clean_rerun_allowed"),
            passed=receipt.get("local_sota_receipt_smoke_v3_ready") is True
            and receipt.get("clean_rerun_allowed") is True,
            blocker_reason="exp3179.clean_rerun_allowed is not true",
        ),
        "controlled_invariance_passed": predicate_row(
            source=EXP3180_REL_PATH,
            field="controlled_invariance_passed",
            expected=True,
            actual=invariance.get("controlled_invariance_passed"),
            passed=invariance.get("controlled_invariance_executor_v2_ready") is True
            and invariance.get("controlled_invariance_passed") is True,
            blocker_reason="exp3180.controlled_invariance_passed is not true",
        ),
        "clean_verifier_not_flagged_adversarial": predicate_row(
            source=EXP3181_REL_PATH,
            field="flagged_adversarial",
            expected=False,
            actual=clean.get("flagged_adversarial"),
            passed=clean.get("clean_live_sota_verifier_rerun_v10_ready") is True
            and clean.get("flagged_adversarial") is False,
            blocker_reason="exp3181.flagged_adversarial is not false",
        ),
        "headline_claim_allowed_for_verifier_metrics": predicate_row(
            source=EXP3181_REL_PATH,
            field="headline_claim_allowed",
            expected=True,
            actual=clean.get("headline_claim_allowed"),
            passed=clean.get("clean_live_sota_verifier_rerun_v10_ready") is True
            and clean.get("headline_claim_allowed") is True,
            blocker_reason="exp3181.headline_claim_allowed is not true",
        ),
        "false_accept_gate_acceptable": predicate_row(
            source=EXP3181_REL_PATH,
            field="false_accept_rate",
            expected=f"finite <= {FALSE_ACCEPT_GATE} from computed clean verifier metrics",
            actual={
                "false_accept_rate": false_accept_rate,
                "metrics_computed": clean.get("metrics_computed"),
                "gated_skip": clean.get("gated_skip"),
                "known_false_accepts_accepted": clean.get("known_false_accepts_accepted") or [],
            },
            passed=false_accept_passed,
            blocker_reason="clean verifier false-accept gate is not acceptable",
        ),
        "certificate_repair_call_ready": predicate_row(
            source=EXP3183_REL_PATH,
            field="repair_call_ready",
            expected=True,
            actual=certificate.get("repair_call_ready"),
            passed=certificate.get("counterexample_certificate_expansion_v3_ready") is True
            and certificate.get("repair_call_ready") is True,
            blocker_reason="exp3183.repair_call_ready is not true",
        ),
    }


def predicate_row(
    *,
    source: Path,
    field: str,
    expected: Any,
    actual: Any,
    passed: bool,
    blocker_reason: str,
) -> JsonDict:
    """Return one auditable gate predicate with source and failed-action text."""

    return {
        "source_artifact": source.as_posix(),
        "field": field,
        "expected": expected,
        "actual": actual,
        "passed": bool(passed),
        "blocker_reason": "" if passed else blocker_reason,
    }


def finite_rate(value: Any) -> float | None:
    """Return a finite rate in [0, 1], or None for malformed metric evidence."""

    if not isinstance(value, (int, float)):
        return None
    rate = float(value)
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        return None
    return rate


def blocker_reasons(
    missing_artifacts: Sequence[str],
    predicates: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Record every missing source and failed predicate in deterministic order."""

    blockers = [f"required artifact missing or malformed: {path}" for path in missing_artifacts]
    for name in PREDICATE_ORDER:
        row = predicates[name]
        if row.get("passed") is not True:
            blockers.append(str(row.get("blocker_reason") or f"{name} failed"))
    return blockers


def repair_gate_state(
    missing_artifacts: Sequence[str],
    predicates: Mapping[str, Mapping[str, Any]],
) -> str:
    """Collapse predicate rows to the single downstream repair machine state."""

    if missing_artifacts:
        return "blocked_missing_artifact"
    if predicates["receipt_smoke_clean_rerun_allowed"]["passed"] is not True:
        return "blocked_receipt_precondition"
    if predicates["controlled_invariance_passed"]["passed"] is not True:
        return "blocked_controlled_invariance"
    if predicates["clean_verifier_not_flagged_adversarial"]["passed"] is not True:
        return "blocked_clean_verifier_flagged"
    if predicates["headline_claim_allowed_for_verifier_metrics"]["passed"] is not True:
        return "blocked_headline_claim_blocked"
    if predicates["false_accept_gate_acceptable"]["passed"] is not True:
        return "blocked_false_accept_gate"
    if predicates["certificate_repair_call_ready"]["passed"] is not True:
        return "blocked_certificate_not_ready"
    if not all_predicates_passed(predicates):
        return "blocked_other"
    return "unblocked_for_bounded_repair_ladder"


def all_predicates_passed(predicates: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return true only when every load-bearing predicate passed."""

    return all(row.get("passed") is True for row in predicates.values())


def allowed_repair_attempt_budget(state: str) -> JsonDict:
    """Expose a strict bounded budget only for the fully unblocked gate."""

    unblocked = state == "unblocked_for_bounded_repair_ladder"
    return {
        "enabled": unblocked,
        "max_total_repair_attempts": 4 if unblocked else 0,
        "max_attempts_per_row": 2 if unblocked else 0,
        "max_distinct_rows": 2 if unblocked else 0,
        "requires_mandated_local_sota": True,
        "requires_exact_authority_acceptance": True,
        "requires_certificate_repair_call_ready": True,
        "stop_on_first_exact_accept_per_row": True,
        "no_headline_claim_from_gate_alone": True,
        "disabled_reason": "" if unblocked else state,
    }


def source_gate_summary(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Keep compact upstream values beside the predicate rows for auditability."""

    return {
        "exp3168_repair_gate_state": payloads["exp3168"].get("repair_gate_state"),
        "exp3179_clean_rerun_allowed": payloads["exp3179"].get("clean_rerun_allowed"),
        "exp3179_substrate_classification": payloads["exp3179"].get("substrate_classification"),
        "exp3180_controlled_invariance_passed": payloads["exp3180"].get(
            "controlled_invariance_passed"
        ),
        "exp3181_gated_skip": payloads["exp3181"].get("gated_skip"),
        "exp3181_flagged_adversarial": payloads["exp3181"].get("flagged_adversarial"),
        "exp3181_headline_claim_allowed": payloads["exp3181"].get("headline_claim_allowed"),
        "exp3181_false_accept_rate": finite_rate(payloads["exp3181"].get("false_accept_rate")),
        "exp3181_metrics_computed": payloads["exp3181"].get("metrics_computed"),
        "exp3183_repair_call_ready": payloads["exp3183"].get("repair_call_ready"),
    }


def inference_substrate(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Declare that Exp 3184 only aggregated prior evidence and made no calls."""

    receipt = payloads["exp3179"]
    clean = payloads["exp3181"]
    certificate = payloads["exp3183"]
    return {
        "kind": "deterministic_repair_gate_decision_v4",
        "aggregation_only": True,
        "no_live_inference": True,
        "no_llm_calls": True,
        "executes_models": False,
        "executes_repairs": False,
        "executes_verifiers": False,
        "executes_solvers": False,
        "downloads_models": False,
        "live_model_calls": 0,
        "new_live_model_calls": 0,
        "repair_calls": 0,
        "source_receipt_live_model_calls": int_value(receipt.get("live_call_count")),
        "source_clean_verifier_live_model_calls": int_value(clean.get("live_call_count")),
        "source_certificate_live_model_calls": int_value(
            (certificate.get("inference_substrate") or {}).get("live_model_calls")
            if isinstance(certificate.get("inference_substrate"), Mapping)
            else 0
        ),
    }


def int_value(value: Any) -> int:
    """Convert nonnegative numeric artifact counters while failing malformed values to zero."""

    return int(value) if isinstance(value, int) and value >= 0 else 0


def field_principles() -> JsonDict:
    """Name why each required field exists for downstream gate consumers."""

    return {
        "repair_gate_decision_v4_ready": "repair gate must be explicit",
        "repair_gate_state": "downstream repair must read one machine state",
        "unblocking_predicates": "gate logic must be auditable",
        "blocker_reasons": "blocked repair must be actionable",
        "missing_artifacts": "absence must not be hidden",
        "allowed_repair_attempt_budget": "unblocked repair must remain bounded",
        "source_artifacts": "gate summaries must trace to files",
        "inference_substrate": "aggregation work must declare no live model inference",
        "honest_verdict": "terminal verdict must honestly reflect blocked preconditions",
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal v4 gate shape before writing it to disk."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3184 artifact missing required fields: {missing}")
    state = str(artifact.get("repair_gate_state") or "")
    if state not in ALLOWED_STATES:
        raise ValueError(f"repair_gate_state must be an allowed state, got {state!r}")
    predicates = artifact.get("unblocking_predicates")
    predicate_rows = predicates if isinstance(predicates, Mapping) else {}
    unblocked = state == "unblocked_for_bounded_repair_ladder"
    if unblocked and not all_predicates_passed(predicate_rows):
        raise ValueError("unblocked_for_bounded_repair_ladder cannot have failed predicates")
    substrate = artifact.get("inference_substrate")
    inference = substrate if isinstance(substrate, Mapping) else {}
    if inference.get("live_model_calls") or inference.get("repair_calls"):
        raise ValueError("Exp 3184 must not perform live model or repair calls")
    budget = artifact.get("allowed_repair_attempt_budget")
    attempt_budget = budget if isinstance(budget, Mapping) else {}
    if unblocked and (
        attempt_budget.get("enabled") is not True
        or int_value(attempt_budget.get("max_total_repair_attempts")) <= 0
        or int_value(attempt_budget.get("max_attempts_per_row")) <= 0
    ):
        raise ValueError("unblocked gate requires a positive bounded repair budget")
    verdict = str(artifact.get("honest_verdict") or "")
    if unblocked and not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("unblocked verdict must start with a terminal success prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict text for conductor and downstream gates."""

    state = str(artifact.get("repair_gate_state") or "blocked_other")
    if state == "unblocked_for_bounded_repair_ladder":
        budget = artifact.get("allowed_repair_attempt_budget")
        attempt_budget = budget if isinstance(budget, Mapping) else {}
        return (
            "complete: repair_gate_state=unblocked_for_bounded_repair_ladder; "
            f"max_total_repair_attempts={attempt_budget.get('max_total_repair_attempts')}; "
            f"max_distinct_rows={attempt_budget.get('max_distinct_rows')}"
        )
    blockers = artifact.get("blocker_reasons")
    first = str(blockers[0]) if isinstance(blockers, list) and blockers else "precondition blocked"
    return f"{state}: {first}"


def sha256_file(path: Path) -> str | None:
    """Checksum local source bytes so the decision traces to exact evidence."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a stable nonnegative elapsed duration for the artifact."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
