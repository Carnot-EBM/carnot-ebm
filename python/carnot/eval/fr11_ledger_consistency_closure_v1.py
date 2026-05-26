"""Exp 3156 FR-11 ledger-consistency closure replay.

Spec refs: REQ-LEARN-3156, SCENARIO-LEARN-3156,
SCENARIO-LEARN-3156-BLOCKED.

The closure question is narrower than "did a model learn?"  This module only
replays controller/environment memory and experience-routing memory against
checked-in exact evidence. It deliberately recomputes ledger consistency from
actions and exact replay outcomes instead of trusting stored booleans whose
names already say "consistent"; that keeps the artifact useful when the prior
ledger still contains counterexamples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.eval import fr11_constraint_memory_retention_drift_audit_v1 as memory_audit
from carnot.eval import fr11_vera_evoenv_hardening_v2 as vera


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3156_fr11_ledger_consistency_closure_v1"
SCHEMA = "carnot.fr11.ledger_consistency_closure.v1"
OUTPUT_REL_PATH = Path("results/experiment_3156_fr11_ledger_consistency_closure_v1.json")
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3128_REL_PATH = Path(
    "results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"
)
EXP3129_REL_PATH = Path(
    "results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json"
)
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3142_REL_PATH = Path("results/experiment_3142_fr11_vera_evoenv_hardening_v2.json")
EXP3143_REL_PATH = Path(
    "results/experiment_3143_fr11_experience_driven_verifier_memory_v1.json"
)
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")

REQUIRED_ARTIFACT_FIELDS = {
    "fr11_ledger_consistency_closure_v1_ready",
    "continuous_self_learning_targeted",
    "replay_panel_count",
    "ledger_consistency_rate",
    "soundness_errors",
    "completeness_errors",
    "residual_mismatch_rows",
    "promotion_recommendation",
    "no_weight_update_claim",
    "methodology_complete",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
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
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3156_fr11_ledger_consistency_closure_v1.py -q",
    ".venv/bin/coverage run --source=python/carnot/eval/fr11_ledger_consistency_closure_v1.py -m pytest -o addopts='' tests/python/test_experiment_3156_fr11_ledger_consistency_closure_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_ledger_consistency_closure_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3126_fragment_time_monitor", EXP3126_REL_PATH, True),
    ("exp3128_evoenv_admission", EXP3128_REL_PATH, True),
    ("exp3129_constraint_memory_audit", EXP3129_REL_PATH, True),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    ("exp3142_vera_evoenv_replay", EXP3142_REL_PATH, True),
    ("exp3143_experience_memory", EXP3143_REL_PATH, True),
    (
        "exp3156_module",
        Path("python/carnot/eval/fr11_ledger_consistency_closure_v1.py"),
        False,
    ),
    (
        "exp3156_tests",
        Path("tests/python/test_experiment_3156_fr11_ledger_consistency_closure_v1.py"),
        False,
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load every checked-in artifact needed by the closure replay."""

    root_path = Path(root)
    return {
        "exp3126": read_json_object(root_path / EXP3126_REL_PATH),
        "exp3128": read_json_object(root_path / EXP3128_REL_PATH),
        "exp3129": read_json_object(root_path / EXP3129_REL_PATH),
        "exp3136": read_json_object(root_path / EXP3136_REL_PATH),
        "exp3142": read_json_object(root_path / EXP3142_REL_PATH),
        "exp3143": read_json_object(root_path / EXP3143_REL_PATH),
    }


def build_replay_panel(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """REQ-LEARN-3156-1/2/3: build the exact replay denominator."""

    exp3126 = sources.get("exp3126", {})
    exp3128 = sources.get("exp3128", {})
    exp3136 = sources.get("exp3136", {})
    exp3142 = sources.get("exp3142", {})
    exp3143 = sources.get("exp3143", {})
    rows: list[JsonDict] = []
    rows.extend(admitted_environment_rows(exp3128))
    rows.extend(variant_rows(exp3142))
    historical_rows, represented_ids = historical_false_accept_rows(exp3136, exp3143)
    rows.extend(historical_rows)
    rows.extend(residual_monitor_rows(exp3126, represented_ids))
    consistent_count = sum(1 for row in rows if row["consistent"])
    soundness_errors = sum(int(row.get("soundness_errors") or 0) for row in rows)
    completeness_errors = sum(int(row.get("completeness_errors") or 0) for row in rows)
    mismatch_rows = [counterexample_row(row) for row in rows if not row["consistent"]]
    return {
        "rows": rows,
        "replay_panel_count": len(rows),
        "ledger_consistent_count": consistent_count,
        "ledger_consistency_rate": rate(consistent_count, len(rows)),
        "soundness_errors": soundness_errors,
        "completeness_errors": completeness_errors,
        "residual_mismatch_rows": mismatch_rows,
        "category_counts": dict(Counter(str(row["panel_category"]) for row in rows)),
    }


def admitted_environment_rows(exp3128: Mapping[str, Any]) -> list[JsonDict]:
    """Replay admitted environment rows from executable constraints."""

    rows: list[JsonDict] = []
    for raw in exp3128.get("admitted_environments", []):
        if not isinstance(raw, Mapping):
            continue
        replay = memory_audit.exact_environment_replay(raw)
        consistent = bool(replay["exact_replay_passed"])
        rows.append(
            replay_row(
                row_id=str(replay["environment_id"]),
                panel_category="admitted_environment",
                source_artifact=EXP3128_REL_PATH.as_posix(),
                fixture_family=str(replay["family_id"]),
                expected_action="accept",
                ledger_action="accept",
                observed_action="accept" if consistent else "reject",
                routing_decision="normal",
                soundness_errors=int(replay["soundness_errors"]),
                completeness_errors=int(replay["completeness_errors"]),
                extra={
                    "assignment_count": int(replay["assignment_count"]),
                    "valid_assignment_count": int(replay["valid_assignment_count"]),
                    "exact_replay_passed": consistent,
                    "reference_consistent": bool(replay["reference_consistent"]),
                    "tautological_consistency_fields_ignored": False,
                },
            )
        )
    return rows


def variant_rows(exp3142: Mapping[str, Any]) -> list[JsonDict]:
    """Replay VeRA variants from their executable environments."""

    rows: list[JsonDict] = []
    for raw in exp3142.get("variant_records", []):
        if not isinstance(raw, Mapping):
            continue
        environment_row = raw.get("environment")
        if not isinstance(environment_row, Mapping):
            continue
        environment = memory_audit.environment_from_row(environment_row)
        replay = vera.exact_variant_replay(environment)
        reference = environment.compute_reference()
        reference_accepted = environment.score_response(reference.canonical_assignment).accepted
        consistent = bool(replay["exact_replay_passed"]) and reference_accepted
        kind = str(raw.get("variant_kind") or "variant")
        category = "hardened_variant" if kind == "hardened" else "equivalent_variant"
        rows.append(
            replay_row(
                row_id=str(raw.get("variant_id") or replay["variant_id"]),
                panel_category=category,
                source_artifact=EXP3142_REL_PATH.as_posix(),
                fixture_family=str(environment.family_id),
                expected_action="accept",
                ledger_action="accept",
                observed_action="accept" if consistent else "reject",
                routing_decision="normal",
                soundness_errors=int(replay["soundness_errors"]),
                completeness_errors=int(replay["completeness_errors"]),
                extra={
                    "source_environment_id": str(raw.get("source_environment_id") or ""),
                    "variant_kind": kind,
                    "assignment_count": int(replay["assignment_count"]),
                    "valid_assignment_count": int(replay["valid_assignment_count"]),
                    "exact_replay_passed": bool(replay["exact_replay_passed"]),
                    "reference_accepted": reference_accepted,
                    "tautological_consistency_fields_ignored": False,
                },
            )
        )
    return rows


def historical_false_accept_rows(
    exp3136: Mapping[str, Any],
    exp3143: Mapping[str, Any],
) -> tuple[list[JsonDict], set[str]]:
    """Replay verifier rows from families with historical false accepts."""

    raw_rows = [row for row in exp3136.get("verifier_rows", []) if isinstance(row, Mapping)]
    row_ids = {str(row.get("row_id") or "") for row in raw_rows}
    false_accept_ids = {str(row_id) for row_id in exp3136.get("false_accept_row_ids", [])}
    parsed = [(row, actions_from_row(row)) for row in raw_rows]
    false_families = {
        str(row.get("fixture_family") or "unknown")
        for row, actions in parsed
        if str(row.get("row_id") or "") in false_accept_ids
        or (
            actions["expected_action"] == "reject"
            and actions["observed_action"] == "accept"
        )
    }
    routing = routing_decisions(exp3143)
    rows: list[JsonDict] = []
    for row, actions in parsed:
        if str(row.get("fixture_family") or "unknown") not in false_families:
            continue
        row_id = str(row.get("row_id") or "")
        rows.append(
            monitor_backed_row(
                row_id=row_id,
                panel_category="historical_false_accept_family",
                source_artifact=EXP3136_REL_PATH.as_posix(),
                fixture_family=str(row.get("fixture_family") or "unknown"),
                routing_decision=routing.get(row_id, "normal"),
                actions=actions,
            )
        )
    return rows, row_ids


def residual_monitor_rows(
    exp3126: Mapping[str, Any],
    represented_ids: set[str],
) -> list[JsonDict]:
    """Add observed inconsistent monitor rows not already represented by Exp 3136."""

    rows: list[JsonDict] = []
    for fixture_id, events in grouped_monitor_events(exp3126.get("monitor_events", [])).items():
        if fixture_id in represented_ids:
            continue
        actions = actions_from_events(events)
        draft = monitor_backed_row(
            row_id=fixture_id,
            panel_category="residual_monitor_inconsistent",
            source_artifact=EXP3126_REL_PATH.as_posix(),
            fixture_family="residual_monitor",
            routing_decision="normal",
            actions=actions,
        )
        if not draft["consistent"] and actions["observed_action"] != "missing":
            rows.append(draft)
    return rows


def replay_row(
    *,
    row_id: str,
    panel_category: str,
    source_artifact: str,
    fixture_family: str,
    expected_action: str,
    ledger_action: str,
    observed_action: str,
    routing_decision: str,
    soundness_errors: int,
    completeness_errors: int,
    extra: Mapping[str, Any],
) -> JsonDict:
    """Construct a normalized panel row and classify it if inconsistent."""

    row: JsonDict = {
        "row_id": row_id,
        "panel_category": panel_category,
        "source_artifact": source_artifact,
        "fixture_family": fixture_family,
        "expected_action": normalize_action(expected_action),
        "ledger_action": normalize_action(ledger_action),
        "observed_action": normalize_action(observed_action),
        "routing_decision": normalize_action(routing_decision),
        "soundness_errors": int(soundness_errors),
        "completeness_errors": int(completeness_errors),
    }
    row.update(dict(extra))
    row["consistent"] = row_is_consistent(row)
    row["mismatch_class"] = "" if row["consistent"] else classify_mismatch(row)
    return row


def monitor_backed_row(
    *,
    row_id: str,
    panel_category: str,
    source_artifact: str,
    fixture_family: str,
    routing_decision: str,
    actions: Mapping[str, Any],
) -> JsonDict:
    """Build a panel row from monitor ledger, exact action, and live action."""

    expected = normalize_action(actions.get("expected_action"))
    observed = normalize_action(actions.get("observed_action"))
    soundness = int(expected == "reject" and observed == "accept" and routing_decision == "suppress")
    completeness = int(expected == "accept" and observed == "reject" and routing_decision == "suppress")
    return replay_row(
        row_id=row_id,
        panel_category=panel_category,
        source_artifact=source_artifact,
        fixture_family=fixture_family,
        expected_action=expected,
        ledger_action=normalize_action(actions.get("ledger_action")),
        observed_action=observed,
        routing_decision=routing_decision,
        soundness_errors=soundness,
        completeness_errors=completeness,
        extra={
            "exact_label": str(actions.get("exact_label") or "unknown"),
            "has_returned_answer": bool(actions.get("has_returned_answer")),
            "tautological_consistency_fields_ignored": bool(
                actions.get("tautological_consistency_fields_present")
            ),
        },
    )


def actions_from_row(row: Mapping[str, Any]) -> JsonDict:
    """Extract non-tautological replay actions from one verifier row."""

    actions = actions_from_events(row.get("monitor_events", []))
    if actions["expected_action"] == "unknown":
        actions["expected_action"] = normalize_action(row.get("expected_action"))
    if actions["observed_action"] == "missing":
        actions["observed_action"] = normalize_action(row.get("live_decision"))
        actions["has_returned_answer"] = actions["observed_action"] != "missing"
    if actions["exact_label"] == "unknown":
        actions["exact_label"] = str(row.get("exact_label") or "unknown")
    return actions


def actions_from_events(events: Any) -> JsonDict:
    """Extract ledger, exact, and observed actions while ignoring stored booleans."""

    expected_action = "unknown"
    ledger_action = "unknown"
    observed_action = "missing"
    exact_label = "unknown"
    returned = False
    tautological_present = False
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        events = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        event_type = event.get("event_type")
        if event_type == "constraint_ledger":
            ledger_action = normalize_action(payload.get("ledger_action"))
        if event_type == "exact_test_z3_result":
            expected_action = normalize_action(payload.get("expected_action"))
            exact_label = str(payload.get("exact_label") or exact_label)
        if event_type == "candidate_final_answer":
            returned = payload.get("has_returned_answer") is True
            observed_action = normalize_action(payload.get("live_decision")) if returned else "missing"
            tautological_present = tautological_present or (
                "final_answer_consistent_with_ledger" in payload
            )
    if ledger_action == "unknown":
        ledger_action = expected_action
    return {
        "expected_action": expected_action,
        "ledger_action": ledger_action,
        "observed_action": observed_action,
        "exact_label": exact_label,
        "has_returned_answer": returned,
        "tautological_consistency_fields_present": tautological_present,
    }


def grouped_monitor_events(events: Any) -> dict[str, list[Mapping[str, Any]]]:
    """Group monitor events by fixture id while preserving row-local order."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return {}
    for event in events:
        if isinstance(event, Mapping):
            fixture_id = str(event.get("fixture_id") or "")
            if fixture_id:
                grouped[fixture_id].append(event)
    return dict(grouped)


def routing_decisions(exp3143: Mapping[str, Any]) -> dict[str, str]:
    """Return Exp 3143 route decisions by replay row id."""

    decisions: dict[str, str] = {}
    for row in exp3143.get("routing_rows", []):
        if isinstance(row, Mapping):
            decisions[str(row.get("row_id") or "")] = normalize_action(row.get("routing_decision"))
    return decisions


def row_is_consistent(row: Mapping[str, Any]) -> bool:
    """Recompute consistency from actions and exact replay status."""

    expected = normalize_action(row.get("expected_action"))
    ledger = normalize_action(row.get("ledger_action"))
    observed = normalize_action(row.get("observed_action"))
    executable_ok = (
        int(row.get("soundness_errors") or 0) == 0
        and int(row.get("completeness_errors") or 0) == 0
        and row.get("exact_replay_passed", True) is not False
    )
    return (
        executable_ok
        and observed != "missing"
        and expected != "unknown"
        and ledger != "unknown"
        and expected == ledger
        and observed == ledger
    )


def classify_mismatch(row: Mapping[str, Any]) -> str:
    """REQ-LEARN-3156-4: classify one inconsistent row into the fixed taxonomy."""

    observed = normalize_action(row.get("observed_action"))
    expected = normalize_action(row.get("expected_action"))
    ledger = normalize_action(row.get("ledger_action"))
    category = str(row.get("panel_category") or "")
    if observed in {"", "missing", "unknown"}:
        return "missing_label"
    if category in {"admitted_environment", "equivalent_variant", "hardened_variant"}:
        return "variant_generation_error"
    if expected != "unknown" and ledger != "unknown" and expected != ledger:
        return "monitor_replay_error"
    if normalize_action(row.get("routing_decision")) == "suppress":
        return "stale_memory"
    if observed != expected or observed != ledger:
        return "contradictory_memory"
    return "monitor_replay_error"


def counterexample_row(row: Mapping[str, Any]) -> JsonDict:
    """Return the replayable counterexample fields for a residual mismatch."""

    return {
        "row_id": str(row.get("row_id") or ""),
        "source_artifact": str(row.get("source_artifact") or ""),
        "panel_category": str(row.get("panel_category") or ""),
        "fixture_family": str(row.get("fixture_family") or ""),
        "expected_action": normalize_action(row.get("expected_action")),
        "ledger_action": normalize_action(row.get("ledger_action")),
        "observed_action": normalize_action(row.get("observed_action")),
        "routing_decision": normalize_action(row.get("routing_decision")),
        "mismatch_class": str(row.get("mismatch_class") or classify_mismatch(row)),
        "counterexample": (
            f"{row.get('row_id')}: expected={normalize_action(row.get('expected_action'))}, "
            f"ledger={normalize_action(row.get('ledger_action'))}, "
            f"observed={normalize_action(row.get('observed_action'))}"
        ),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3156 closure artifact from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = precondition_blocker(
        sources["exp3128"],
        sources["exp3129"],
        sources["exp3136"],
        sources["exp3142"],
        sources["exp3143"],
        sources["exp3126"],
    )
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact
    panel = build_replay_panel(sources)
    methodology_complete = panel["replay_panel_count"] > 0 and required_categories_present(panel)
    recommendation = promotion_recommendation(
        methodology_complete,
        float(panel["ledger_consistency_rate"]),
        int(panel["soundness_errors"]),
        int(panel["completeness_errors"]),
    )
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_ledger_consistency_closure_v1_ready": methodology_complete,
        "continuous_self_learning_targeted": True,
        "replay_panel_count": int(panel["replay_panel_count"]),
        "ledger_consistency_rate": float(panel["ledger_consistency_rate"]),
        "soundness_errors": int(panel["soundness_errors"]),
        "completeness_errors": int(panel["completeness_errors"]),
        "residual_mismatch_rows": panel["residual_mismatch_rows"],
        "promotion_recommendation": recommendation,
        "no_weight_update_claim": True,
        "methodology_complete": methodology_complete,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": inference_substrate(),
        "precondition_checks": precondition_checks(sources),
        "category_counts": panel["category_counts"],
        "ledger_consistent_count": int(panel["ledger_consistent_count"]),
        "replay_panel_rows": panel["rows"],
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(methodology_complete, float(panel["ledger_consistency_rate"]), recommendation),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    start: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """Return a schema-complete artifact when source evidence is missing."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_ledger_consistency_closure_v1_ready": False,
        "continuous_self_learning_targeted": True,
        "replay_panel_count": 0,
        "ledger_consistency_rate": 0.0,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "residual_mismatch_rows": [],
        "promotion_recommendation": "block_fr11_ledger_consistency_closure_missing_source_evidence",
        "no_weight_update_claim": True,
        "methodology_complete": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root),
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "precondition_checks": {},
        "category_counts": {},
        "ledger_consistent_count": 0,
        "replay_panel_rows": [],
        "blocked_reason": blocker,
        "duration_s": duration(start, now_s),
        "honest_verdict": f"blocked_precondition_failed: {blocker}",
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3156 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def precondition_blocker(
    exp3128: Mapping[str, Any],
    exp3129: Mapping[str, Any],
    exp3136: Mapping[str, Any],
    exp3142: Mapping[str, Any],
    exp3143: Mapping[str, Any],
    exp3126: Mapping[str, Any] | None = None,
) -> str:
    """Return the first missing closure source artifact."""

    if exp3128.get("fr11_evoenv_pilot_v1_ready") is not True:
        return "exp3128_evoenv_missing_or_not_ready"
    if exp3129.get("fr11_constraint_memory_audit_v1_ready") is not True:
        return "exp3129_constraint_memory_missing_or_not_ready"
    if exp3136.get("false_accept_autopsy_v1_ready") is not True:
        return "exp3136_false_accept_autopsy_missing_or_not_ready"
    if exp3142.get("fr11_vera_evoenv_v2_ready") is not True:
        return "exp3142_vera_evoenv_missing_or_not_ready"
    if exp3143.get("fr11_experience_verifier_memory_v1_ready") is not True:
        return "exp3143_experience_memory_missing_or_not_ready"
    if exp3126 is not None and exp3126.get("fragment_time_monitor_v1_ready") is not True:
        return "exp3126_fragment_time_monitor_missing_or_not_ready"
    return ""


def precondition_checks(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Expose source readiness in the terminal artifact."""

    return {
        "exp3126_fragment_time_monitor_ready": sources["exp3126"].get(
            "fragment_time_monitor_v1_ready"
        )
        is True,
        "exp3128_evoenv_ready": sources["exp3128"].get("fr11_evoenv_pilot_v1_ready") is True,
        "exp3129_constraint_memory_ready": sources["exp3129"].get(
            "fr11_constraint_memory_audit_v1_ready"
        )
        is True,
        "exp3136_false_accept_autopsy_ready": sources["exp3136"].get(
            "false_accept_autopsy_v1_ready"
        )
        is True,
        "exp3142_vera_evoenv_ready": sources["exp3142"].get("fr11_vera_evoenv_v2_ready")
        is True,
        "exp3143_experience_memory_ready": sources["exp3143"].get(
            "fr11_experience_verifier_memory_v1_ready"
        )
        is True,
    }


def required_categories_present(panel: Mapping[str, Any]) -> bool:
    """Check the minimum closure panel coverage required by REQ-LEARN-3156."""

    counts = panel.get("category_counts", {})
    return all(
        int(counts.get(category, 0)) > 0
        for category in (
            "admitted_environment",
            "equivalent_variant",
            "hardened_variant",
            "historical_false_accept_family",
        )
    )


def promotion_recommendation(
    ready: bool,
    ledger_consistency_rate: float,
    soundness_errors: int,
    completeness_errors: int,
) -> str:
    """REQ-LEARN-3156-5: apply the exact FR-11 promotion gate."""

    if not ready:
        return "block_fr11_ledger_consistency_closure_incomplete"
    if soundness_errors or completeness_errors:
        return "block_fr11_promotion_soundness_or_completeness_regression"
    if ledger_consistency_rate < 1.0:
        return "block_fr11_promotion_until_ledger_consistency_reaches_1.0"
    return "promote_controller_environment_memory_only"


def inference_substrate(mode: str = "solver_only_memory_ledger_replay") -> JsonDict:
    """Declare that this replay uses no live LLM inference or weight updates."""

    return {
        "mode": mode,
        "controller_environment_memory_only": True,
        "experience_memory_replay": True,
        "uses_checked_in_artifacts_only": True,
        "executes_exact_solvers": True,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and artifacts with checksums for replay traceability."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3156 artifact violates the closure contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("no_weight_update_claim") is not True:
        raise ValueError("no_weight_update_claim must be true")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or any(
        substrate.get(flag) is True
        for flag in ("model_weight_mutation", "model_weight_training", "base_model_weights_updated")
    ):
        raise ValueError("model_weight_mutation must remain false")
    if int(substrate.get("fresh_live_inference_calls") or 0) != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    ledger_rate = float(artifact.get("ledger_consistency_rate", math.nan))
    if not math.isfinite(ledger_rate) or not 0.0 <= ledger_rate <= 1.0:
        raise ValueError("ledger_consistency_rate must be finite and within [0, 1]")
    if artifact.get("fr11_ledger_consistency_closure_v1_ready") is not True:
        return
    if int(artifact.get("replay_panel_count") or 0) <= 0:
        raise ValueError("replay_panel_count must be positive for readiness")
    if ledger_rate < 1.0 and not str(artifact.get("promotion_recommendation") or "").startswith(
        "block_fr11"
    ):
        raise ValueError("promotion_recommendation must block imperfect ledgers")
    if any(
        row.get("required") and not row.get("exists")
        for row in artifact.get("source_artifacts", [])
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def honest_verdict(ready: bool, ledger_consistency_rate: float, recommendation: str) -> str:
    """Return a conductor-compatible terminal verdict."""

    if ready:
        return (
            "complete: fr11 ledger consistency closure replay finished; "
            f"ledger_consistency_rate={round_float(ledger_consistency_rate)}; "
            f"promotion_recommendation={recommendation}; no model-weight update claimed"
        )
    return "blocked_precondition_failed: fr11 ledger consistency closure sources missing"


def normalize_action(value: Any) -> str:
    """Normalize small action tokens used by monitor and memory artifacts."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate, using zero for empty denominators."""

    if denominator <= 0:
        return 0.0
    return round_float(numerator / denominator)


def round_float(value: float) -> float:
    """Round artifact floats to stable six-decimal precision."""

    return round(float(value), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return stable elapsed seconds for artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round_float(max(0.0, end - started_s))


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when the path exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output for deterministic artifact diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
