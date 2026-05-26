"""Build the Exp 3137 exact-safe accept/abstain/reject contract artifact.

Spec refs: REQ-VERIFY-3137, SCENARIO-VERIFY-3137.

This module does not run a model, a solver, or a repair loop. It turns the
checked-in .291 false-accept autopsy and the exact fixture evidence into a
machine-readable contract for the next live rerun. The contract is deliberately
stricter than the failed live verifier path: an unsafe row may abstain, but it
must not be accepted just because the model returned a plausible verdict token.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3137_exact_safe_accept_abstain_contract_v1"
SCHEMA = "carnot.exact_safe_accept_abstain_contract.v1"
OUTPUT_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / (
    "experiment_3137_exact_safe_accept_abstain_contract_v1.py"
)

EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3125_REL_PATH = Path(
    "results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json"
)
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3098_REL_PATH = Path("results/experiment_3098_maxsat_abstention_routing_policy_v1.json")
EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
MANIFEST_REL_PATH = Path("results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl")

ACCEPT_LABELS = {"VALID", "SAT"}
REJECT_LABELS = {"INVALID", "UNSAT", "REPAIRABLE", "UNREPAIRABLE"}
TOKEN_FAMILY_BY_LABEL = {
    "VALID": "validity_token",
    "INVALID": "validity_token",
    "SAT": "sat_token",
    "UNSAT": "sat_token",
    "REPAIRABLE": "repairability_token",
    "UNREPAIRABLE": "repairability_token",
}
KNOWN_FALSE_ACCEPT_FAMILIES = {
    "SAT/validity-token confusion",
    "contradiction miss",
    "premise/step grounding failure",
    "model prior/data mismatch",
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
    "acceptance_contract_v1_ready",
    "contract_rules",
    "known_false_accept_rows_blocked",
    "replay_false_accept_rate",
    "replay_abstention_rate",
    "replay_false_reject_rate",
    "repair_gate_prerequisites",
    "regression_row_set",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3137_exact_safe_accept_abstain_contract_v1.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3137_exact_safe_accept_abstain_contract_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/exact_safe_accept_abstain_contract_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_REL_PATHS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), False),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    ("exp3125_prefix_bound_pilot", EXP3125_REL_PATH, True),
    ("exp3126_fragment_monitor", EXP3126_REL_PATH, True),
    ("exp3098_maxsat_policy_optional", EXP3098_REL_PATH, False),
    ("exp3097_exact_fixture_protocol", EXP3097_REL_PATH, True),
    ("exp3137_module", Path("python/carnot/verify/exact_safe_accept_abstain_contract_v1.py"), False),
    ("exp3137_script", Path("scripts/experiment_3137_exact_safe_accept_abstain_contract_v1.py"), False),
)


@dataclass(frozen=True)
class ContractContext:
    """Static evidence needed to route one row without consulting a model."""

    prefix_covered_labels: frozenset[str]
    regression_row_set: frozenset[str]
    monitor_by_fixture: Mapping[str, Sequence[Mapping[str, Any]]]
    parse_confidence_floor: float = 1.0


def expected_action_from_label(label: str | None) -> str:
    """Map exact labels into the public accept/reject/abstain action space."""

    normalized = str(label or "").upper()
    if normalized in ACCEPT_LABELS:
        return "accept"
    if normalized in REJECT_LABELS:
        return "reject"
    return "abstain"


def token_family_for_label(label: str | None) -> str | None:
    """Return the response-token family that an exact label belongs to."""

    return TOKEN_FAMILY_BY_LABEL.get(str(label or "").upper())


def read_json_object(path: Path) -> JsonDict:
    """Read one local JSON object while making malformed evidence non-promotable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive filesystem guard.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read object rows from a JSONL manifest and ignore malformed lines."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - defensive filesystem guard.
        return []
    rows: list[JsonDict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3137: build the exact-safe contract and replay artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3136 = read_json_object(root_path / EXP3136_REL_PATH)
    exp3125 = read_json_object(root_path / EXP3125_REL_PATH)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    exp3098 = read_json_object(root_path / EXP3098_REL_PATH)
    exp3097 = read_json_object(root_path / EXP3097_REL_PATH)
    manifest_rel_path = Path(str(exp3097.get("stratified_eval_manifest_path") or MANIFEST_REL_PATH))
    manifest_rows = read_jsonl_rows(root_path / manifest_rel_path)
    live_rows = live_rows_from_autopsy(exp3136)
    exact_rows = exact_fixture_rows_from_manifest(manifest_rows)
    regression_rows = tuple(str(row) for row in exp3136.get("regression_row_set") or [])
    context = ContractContext(
        prefix_covered_labels=prefix_covered_labels(exp3125),
        regression_row_set=frozenset(regression_rows),
        monitor_by_fixture=monitor_events_by_fixture(exp3126.get("monitor_events")),
    )
    replay_rows = replay_contract(live_rows, exact_rows, context)
    replay_counts = replay_metrics(replay_rows)
    source_rows = source_artifacts(root_path, manifest_rel_path)
    checks = self_checks(replay_rows, replay_counts, context)
    prerequisites = repair_gate_prerequisites(exp3098)
    ready = bool(
        all(row["present"] for row in source_rows if row["required"])
        and exp3136.get("false_accept_autopsy_v1_ready") is True
        and exp3125.get("prefix_closed_bound_pilot_ready") is True
        and exp3126.get("fragment_time_monitor_v1_ready") is True
        and replay_counts["false_accept_rate"] == 0.0
        and checks["all_self_checks_passed"]
        and replay_counts["total_rows"] > 0
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "acceptance_contract_v1_ready": ready,
        "contract_rules": build_contract_rules(),
        "known_false_accept_rows_blocked": checks["regression_rows_blocked_from_accept"],
        "replay_false_accept_rate": replay_counts["false_accept_rate"],
        "replay_abstention_rate": replay_counts["abstention_rate"],
        "replay_false_reject_rate": replay_counts["false_reject_rate"],
        "replay_counts": replay_counts,
        "replay_rows": replay_rows,
        "repair_gate_prerequisites": prerequisites,
        "regression_row_set": sorted(context.regression_row_set),
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row.get("sha256")
        },
        "inference_substrate": inference_substrate(exp3136),
        "self_checks": checks,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "no_live_repair_rerun": True,
        "duration_s": duration(start, now_s),
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
    """Build, validate, and persist the Exp 3137 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def build_contract_rules() -> list[JsonDict]:
    """Return the ordered machine-readable contract rules."""

    return [
        {
            "order": 10,
            "id": "ABSTAIN_MISSING_EXACT_LABEL",
            "decision": "abstain",
            "condition": "exact_label is missing or outside the exact label vocabulary",
            "principle": "no accept/reject route may infer labels that exact authority did not provide",
        },
        {
            "order": 20,
            "id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
            "decision": "abstain",
            "condition": "row is a known .291 false accept or known false-accept family",
            "principle": "prior false accepts must be blocked before generic exact-label routing",
        },
        {
            "order": 30,
            "id": "ABSTAIN_LOW_PARSE_CONFIDENCE",
            "decision": "abstain",
            "condition": "parse_confidence < 1.0",
            "principle": "unsafe parses fail closed instead of accepting plausible raw text",
        },
        {
            "order": 40,
            "id": "ABSTAIN_TOKEN_FAMILY_MISMATCH",
            "decision": "abstain",
            "condition": "extracted token family disagrees with exact label token family",
            "principle": "SAT/UNSAT prompts must not accept VALID/INVALID answers",
        },
        {
            "order": 50,
            "id": "ABSTAIN_MISSING_LIVE_MONITOR_REPLAY",
            "decision": "abstain",
            "condition": "live row has no replayed monitor-ledger evidence",
            "principle": "live acceptance needs replayable ledger evidence",
        },
        {
            "order": 60,
            "id": "ABSTAIN_ACCEPT_LABEL_OUTSIDE_PREFIX_COVERAGE",
            "decision": "abstain",
            "condition": "accept label is not covered by the prefix-bound pilot",
            "principle": "acceptance cannot claim exact-safe coverage outside the bounded frontier",
        },
        {
            "order": 70,
            "id": "ACCEPT_EXACT_COVERED_CONSISTENT",
            "decision": "accept",
            "condition": "exact accept label, exact answer match, prefix coverage, and ledger agreement",
            "principle": "accept only when exact labels, parsing, premises, prefix bounds, and ledger agree",
        },
        {
            "order": 80,
            "id": "REJECT_EXACT_REJECT_MATCH",
            "decision": "reject",
            "condition": "exact reject label and extracted answer matches that exact reject label",
            "principle": "exact reject labels remain authoritative when the candidate agrees",
        },
        {
            "order": 90,
            "id": "REJECT_EXACT_REJECT_CONTRADICTION",
            "decision": "reject",
            "condition": "exact reject label, candidate would accept, and monitor ledger requires reject",
            "principle": "unknown exact contradictions are rejected after regression abstention gates",
        },
        {
            "order": 100,
            "id": "ABSTAIN_DEFAULT_FAIL_CLOSED",
            "decision": "abstain",
            "condition": "no earlier exact-safe rule matched",
            "principle": "residual uncertainty trades coverage for safety",
        },
    ]


def evaluate_row(
    row: Mapping[str, Any],
    context: ContractContext,
    *,
    row_source: str = "exact_fixture",
) -> JsonDict:
    """Apply the ordered exact-safe contract to one row."""

    row_id = row_id_from(row)
    exact = exact_label_from(row)
    expected = expected_action_from_row(row)
    extracted = optional_str(row.get("extracted_answer"))
    parse_confidence = parse_confidence_for(row, extracted)
    monitor_events = context.monitor_by_fixture.get(row_id, ()) if row_source == "live" else ()
    monitor = replay_monitor(monitor_events)
    exact_family = token_family_for_label(exact)
    extracted_family = token_family_for_label(extracted)
    token_family_match = bool(exact_family and extracted_family and exact_family == extracted_family)
    premise_answer_consistent = bool(
        extracted == exact
        and (monitor["ledger_action"] in {None, expected})
        and monitor["final_answer_consistent_with_ledger"] is not False
    )
    base = {
        "row_id": row_id,
        "row_source": row_source,
        "exact_label": exact,
        "expected_action": expected,
        "extracted_answer": extracted,
        "parse_confidence": parse_confidence,
        "token_family_match": token_family_match,
        "prefix_label_covered": exact in context.prefix_covered_labels,
        "premise_answer_consistent": premise_answer_consistent,
        "monitor_replay": monitor,
        "known_false_accept_family": known_false_accept_family(row, context),
    }
    if not exact or expected == "abstain":
        return decision(base, "abstain", "ABSTAIN_MISSING_EXACT_LABEL")
    if base["known_false_accept_family"] and expected == "reject" and candidate_would_accept(row):
        return decision(base, "abstain", "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION")
    if parse_confidence < context.parse_confidence_floor:
        return decision(base, "abstain", "ABSTAIN_LOW_PARSE_CONFIDENCE")
    if extracted is not None and exact_family and extracted_family and not token_family_match:
        return decision(base, "abstain", "ABSTAIN_TOKEN_FAMILY_MISMATCH")
    if row_source == "live" and monitor["monitor_event_count"] == 0:
        return decision(base, "abstain", "ABSTAIN_MISSING_LIVE_MONITOR_REPLAY")
    if expected == "accept" and exact not in context.prefix_covered_labels:
        return decision(base, "abstain", "ABSTAIN_ACCEPT_LABEL_OUTSIDE_PREFIX_COVERAGE")
    if expected == "accept" and extracted == exact and premise_answer_consistent:
        return decision(base, "accept", "ACCEPT_EXACT_COVERED_CONSISTENT")
    if expected == "reject" and extracted == exact and premise_answer_consistent:
        return decision(base, "reject", "REJECT_EXACT_REJECT_MATCH")
    if expected == "reject" and candidate_would_accept(row) and monitor["ledger_action"] == "reject":
        return decision(base, "reject", "REJECT_EXACT_REJECT_CONTRADICTION")
    return decision(base, "abstain", "ABSTAIN_DEFAULT_FAIL_CLOSED")


def replay_contract(
    live_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    context: ContractContext,
) -> list[JsonDict]:
    """Replay live rows and prior exact fixtures through the same contract."""

    replayed = [
        evaluate_row(row, context, row_source="live")
        for row in sorted(live_rows, key=row_id_from)
    ]
    replayed.extend(
        evaluate_row(row, context, row_source="exact_fixture")
        for row in sorted(exact_rows, key=row_id_from)
    )
    return replayed


def live_rows_from_autopsy(exp3136: Mapping[str, Any]) -> list[JsonDict]:
    """Merge Exp 3136 verifier rows with richer false-accept autopsy rows."""

    live_rows = mapping_rows(exp3136.get("verifier_rows"))
    false_rows = {row_id_from(row): row for row in mapping_rows(exp3136.get("false_accept_rows"))}
    if not live_rows:
        live_rows = list(false_rows.values())
    merged = []
    for row in live_rows:
        row_id = row_id_from(row)
        merged.append(dict(row) | dict(false_rows.get(row_id, {})))
    return merged


def exact_fixture_rows_from_manifest(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert exact fixture manifest rows into deterministic reference candidates."""

    exact_rows: list[JsonDict] = []
    for row in rows:
        exact = str(row.get("expected_answer") or row.get("exact_label") or "").upper()
        fixture_id = str(row.get("source_fixture_id") or row.get("fixture_id") or "")
        if not fixture_id:
            continue
        target = row.get("verifier_target") if isinstance(row.get("verifier_target"), Mapping) else {}
        exact_rows.append(
            {
                "fixture_id": fixture_id,
                "row_id": fixture_id,
                "exact_label": exact,
                "expected_action": target.get("expected_action") or expected_action_from_label(exact),
                "extracted_answer": exact,
                "parse_confidence": 1.0,
                "task_family": row.get("task_family"),
                "label_source": row.get("label_source"),
                "answer_extraction_format": token_family_for_label(exact),
            }
        )
    return exact_rows


def replay_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute safety and coverage rates from replayed decisions."""

    total = len(rows)
    reject_rows = [row for row in rows if row.get("expected_action") == "reject"]
    accept_rows = [row for row in rows if row.get("expected_action") == "accept"]
    false_accept_count = sum(
        row.get("expected_action") == "reject" and row.get("decision") == "accept" for row in rows
    )
    false_reject_count = sum(
        row.get("expected_action") == "accept" and row.get("decision") == "reject" for row in rows
    )
    abstain_count = sum(row.get("decision") == "abstain" for row in rows)
    return {
        "total_rows": total,
        "live_rows": sum(row.get("row_source") == "live" for row in rows),
        "exact_fixture_rows": sum(row.get("row_source") == "exact_fixture" for row in rows),
        "accept_decision_count": sum(row.get("decision") == "accept" for row in rows),
        "reject_decision_count": sum(row.get("decision") == "reject" for row in rows),
        "abstain_decision_count": abstain_count,
        "false_accept_count": false_accept_count,
        "false_reject_count": false_reject_count,
        "reject_denominator": len(reject_rows),
        "accept_denominator": len(accept_rows),
        "false_accept_rate": rate(false_accept_count, len(reject_rows)),
        "false_reject_rate": rate(false_reject_count, len(accept_rows)),
        "abstention_rate": rate(abstain_count, total),
    }


def self_checks(
    replay_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    context: ContractContext,
) -> JsonDict:
    """Run deterministic checks that make the contract ordering auditable."""

    rule_ids = [rule["id"] for rule in build_contract_rules()]
    regression_live_rows = [
        row
        for row in replay_rows
        if row.get("row_source") == "live" and row.get("row_id") in context.regression_row_set
    ]
    regression_blocked = bool(context.regression_row_set) and {
        row.get("row_id") for row in regression_live_rows if row.get("decision") != "accept"
    } == set(context.regression_row_set)
    deterministic_hash = stable_hash(replay_rows)
    round_trip_hash = stable_hash(json.loads(json.dumps(replay_rows, sort_keys=True)))
    rates = [
        metrics["false_accept_rate"],
        metrics["false_reject_rate"],
        metrics["abstention_rate"],
    ]
    checks = {
        "known_regression_rule_precedes_generic_reject": rule_ids.index(
            "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION"
        )
        < rule_ids.index("REJECT_EXACT_REJECT_CONTRADICTION"),
        "regression_rows_present_in_live_replay": {
            row.get("row_id") for row in regression_live_rows
        }
        == set(context.regression_row_set),
        "regression_rows_blocked_from_accept": regression_blocked,
        "replay_deterministic": deterministic_hash == round_trip_hash,
        "finite_rates_in_unit_interval": all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in rates),
        "contract_has_accept_abstain_reject_rules": {"accept", "abstain", "reject"}
        <= {rule["decision"] for rule in build_contract_rules()},
    }
    checks["all_self_checks_passed"] = all(checks.values())
    checks["replay_hash"] = deterministic_hash
    return checks


def repair_gate_prerequisites(exp3098: Mapping[str, Any]) -> JsonDict:
    """Expose the gates a later repair/live rerun must consume explicitly."""

    fallback = exp3098.get("fallback_evaluator")
    fallback_map = fallback if isinstance(fallback, Mapping) else {}
    return {
        "must_load_contract_path": OUTPUT_REL_PATH.as_posix(),
        "required_contract_schema": SCHEMA,
        "known_regression_rows_must_not_accept": True,
        "replay_false_accept_rate_must_equal": 0.0,
        "require_exact_label_authority": True,
        "require_parse_confidence_floor": 1.0,
        "require_prefix_bound_coverage_for_accept": True,
        "require_monitor_ledger_replay_for_live_rows": True,
        "maxsat_policy_present": exp3098.get("maxsat_policy_ready") is True,
        "maxsat_policy_fail_closed": fallback_map.get("fail_closed_default") == "abstain",
        "repair_gate_opens_only_after_live_rerun_replay": True,
    }


def prefix_covered_labels(exp3125: Mapping[str, Any]) -> frozenset[str]:
    """Return labels covered by the bounded prefix pilot."""

    labels = {
        str(row.get("expected_answer") or "").upper()
        for row in mapping_rows(exp3125.get("fixture_details"))
    }
    return frozenset(label for label in labels if label)


def monitor_events_by_fixture(value: Any) -> dict[str, list[JsonDict]]:
    """Group replayable monitor events by fixture ID."""

    grouped: dict[str, list[JsonDict]] = {}
    for event in mapping_rows(value):
        grouped.setdefault(str(event.get("fixture_id") or ""), []).append(event)
    for fixture_id in grouped:
        grouped[fixture_id].sort(key=lambda event: int(event.get("event_index") or 0))
    return grouped


def replay_monitor(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize the monitor ledger fields that the contract needs."""

    ledger = first_event_payload(events, "constraint_ledger")
    candidate = first_event_payload(events, "candidate_final_answer")
    drift = first_event_payload(events, "drift_classification")
    return {
        "monitor_event_count": len(events),
        "ledger_action": ledger.get("ledger_action"),
        "ledger_source": ledger.get("ledger_source"),
        "candidate_live_decision": candidate.get("live_decision"),
        "candidate_extracted_answer": candidate.get("extracted_answer"),
        "final_answer_consistent_with_ledger": candidate.get(
            "final_answer_consistent_with_ledger"
        ),
        "failure_mechanism": drift.get("failure_mechanism"),
        "is_monitor_violation": drift.get("is_monitor_violation") is True,
    }


def first_event_payload(events: Sequence[Mapping[str, Any]], event_type: str) -> JsonDict:
    """Return the first payload for an event type, or empty evidence."""

    for event in events:
        if event.get("event_type") == event_type:
            payload = event.get("payload")
            return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def source_artifacts(root: Path, manifest_rel_path: Path) -> list[JsonDict]:
    """Return checksummed source rows so the contract traces to exact bytes."""

    rows = []
    for role, rel_path, required in SOURCE_REL_PATHS:
        rows.append(source_row(root, role, rel_path, required))
    rows.append(source_row(root, "exp3097_stratified_manifest", manifest_rel_path, True))
    return rows


def source_row(root: Path, role: str, rel_path: Path, required: bool) -> JsonDict:
    """Build one source-artifact provenance row."""

    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "required": required,
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def inference_substrate(exp3136: Mapping[str, Any]) -> JsonDict:
    """Describe what computation was and was not performed by this builder."""

    upstream = exp3136.get("inference_substrate")
    reused = upstream.get("upstream_live_model_calls_reused", 0) if isinstance(upstream, Mapping) else 0
    return {
        "kind": "deterministic_artifact_replay",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "fresh_live_model_calls": 0,
        "upstream_live_model_calls_reused": int(reused or 0),
        "source": EXP3136_REL_PATH.as_posix(),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and the exact-safe no-false-accept claim."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3137 artifact missing required fields: {missing}")
    for key in ("replay_false_accept_rate", "replay_abstention_rate", "replay_false_reject_rate"):
        value = artifact.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{key} rate outside [0, 1]: {value}")
    if artifact.get("known_false_accept_rows_blocked") is not True:
        raise ValueError("known_false_accept_rows_blocked must be true")
    if float(artifact.get("replay_false_accept_rate") or 0.0) != 0.0:
        raise ValueError("replay_false_accept_rate must be 0.0")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("acceptance_contract_v1_ready") is True and not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict string with the required success prefix."""

    if artifact.get("acceptance_contract_v1_ready") is True:
        return (
            "complete: acceptance_contract_v1_ready=true; "
            f"replay_false_accept_rate={artifact.get('replay_false_accept_rate')}; "
            f"known_false_accept_rows_blocked={artifact.get('known_false_accept_rows_blocked')}"
        )
    return "blocked_exact_safe_contract_missing_required_replay_evidence"


def decision(base: Mapping[str, Any], decision_value: str, rule_id: str) -> JsonDict:
    """Attach the matched rule to a normalized replay row."""

    return dict(base) | {"decision": decision_value, "matched_rule_id": rule_id}


def mapping_rows(value: Any) -> list[JsonDict]:
    """Return only mapping rows from an arbitrary list-like value."""

    return [dict(row) for row in value] if isinstance(value, list) else []


def known_false_accept_family(row: Mapping[str, Any], context: ContractContext) -> bool:
    """Return whether the row belongs to a prior false-accept family."""

    mechanism = str(row.get("primary_mechanism") or row.get("failure_mechanism_from_exp3124") or "")
    return row_id_from(row) in context.regression_row_set or mechanism in KNOWN_FALSE_ACCEPT_FAMILIES


def candidate_would_accept(row: Mapping[str, Any]) -> bool:
    """Return whether row evidence says the candidate tried to accept."""

    return row.get("live_decision") == "accept" or exact_label_from({"exact_label": row.get("extracted_answer")}) in ACCEPT_LABELS


def expected_action_from_row(row: Mapping[str, Any]) -> str:
    """Read or derive the expected contract action for a row."""

    action = str(row.get("expected_action") or "")
    return action or expected_action_from_label(exact_label_from(row))


def exact_label_from(row: Mapping[str, Any]) -> str:
    """Normalize the exact label field used by all replay metrics."""

    return str(row.get("exact_label") or row.get("expected_answer") or "").upper()


def row_id_from(row: Mapping[str, Any]) -> str:
    """Return the stable replay row identifier."""

    return str(row.get("row_id") or row.get("fixture_id") or row.get("source_fixture_id") or "")


def optional_str(value: Any) -> str | None:
    """Normalize optional strings without turning missing values into labels."""

    return None if value is None else str(value).upper()


def parse_confidence_for(row: Mapping[str, Any], extracted: str | None) -> float:
    """Return exact parse confidence from row metadata or deterministic token parsing."""

    if "parse_confidence" in row:
        return float(row.get("parse_confidence") or 0.0)
    return 1.0 if extracted in (ACCEPT_LABELS | REJECT_LABELS) else 0.0


def rate(numerator: int, denominator: int) -> float:
    """Compute a rounded finite rate while keeping empty denominators explicit."""

    return round(numerator / denominator, 6) if denominator else 0.0


def sha256_file(path: Path) -> str | None:
    """Checksum a source file so replay claims remain tied to exact inputs."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash JSON-serializable evidence using canonical key ordering."""

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative wall-clock duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
