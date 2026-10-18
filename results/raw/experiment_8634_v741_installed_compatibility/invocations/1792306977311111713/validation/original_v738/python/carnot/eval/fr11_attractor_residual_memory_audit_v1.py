"""Exp 3157 FR-11 attractor residual memory audit.

Spec refs: REQ-LEARN-3157, SCENARIO-LEARN-3157,
SCENARIO-LEARN-3157-BLOCKED.

This audit treats attractor and residual language as controller diagnostics,
not as model-weight learning.  It replays exact, checked-in FR-11 evidence and
asks whether simple residual signals can route future verifier work: suppress
only clearly redundant checks, escalate historically risky families, and report
any unsafe skip instead of hiding verifier work.
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


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3157_fr11_attractor_residual_memory_audit_v1"
SCHEMA = "carnot.fr11.attractor_residual_memory_audit.v1"
OUTPUT_REL_PATH = Path("results/experiment_3157_fr11_attractor_residual_memory_audit_v1.json")
EXP3156_REL_PATH = Path("results/experiment_3156_fr11_ledger_consistency_closure_v1.json")
EXP3143_REL_PATH = Path("results/experiment_3143_fr11_experience_driven_verifier_memory_v1.json")
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")

REQUIRED_ARTIFACT_FIELDS = {
    "fr11_attractor_residual_memory_audit_v1_ready",
    "continuous_self_learning_targeted",
    "residual_signal_definitions",
    "replay_panel_count",
    "risky_family_escalation_rate",
    "redundant_check_suppression_rate",
    "unsafe_skip_count",
    "promotion_recommendation",
    "no_weight_update_claim",
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
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3157_fr11_attractor_residual_memory_audit_v1.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3157_fr11_attractor_residual_memory_audit_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_attractor_residual_memory_audit_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3157_fr11_attractor_residual_memory_audit_v1.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3156_ledger_closure", EXP3156_REL_PATH, True),
    ("exp3143_experience_memory", EXP3143_REL_PATH, True),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    (
        "exp3157_module",
        Path("python/carnot/eval/fr11_attractor_residual_memory_audit_v1.py"),
        False,
    ),
    (
        "exp3157_tests",
        Path("tests/python/test_experiment_3157_fr11_attractor_residual_memory_audit_v1.py"),
        False,
    ),
)


def residual_signal_definitions() -> list[JsonDict]:
    """REQ-LEARN-3157-1: define exact replay signals used by this audit."""

    return [
        {
            "signal_id": "repeated_mismatch_count",
            "measurement": (
                "Count of non-consistent Exp 3156 closure rows and Exp 3136 "
                "false-accept rows sharing the same fixture family."
            ),
            "exact_replay_fields": [
                "replay_panel_rows[].fixture_family",
                "replay_panel_rows[].consistent",
                "false_accept_row_ids",
            ],
            "policy_use": "Escalate rows from families with repeated mismatch evidence.",
        },
        {
            "signal_id": "stable_verdict_convergence",
            "measurement": (
                "Whether exact expected action, ledger action, and observed action "
                "converge to the same non-missing verdict."
            ),
            "exact_replay_fields": [
                "expected_action",
                "ledger_action",
                "observed_action",
                "consistent",
            ],
            "policy_use": "Allow suppression only when the verdict is stable.",
        },
        {
            "signal_id": "contradiction_core_stability",
            "measurement": (
                "Whether the row avoids contradictory-memory and monitor-replay "
                "mismatch classes in exact replay."
            ),
            "exact_replay_fields": ["mismatch_class", "expected_action", "observed_action"],
            "policy_use": "Escalate contradiction-core instability.",
        },
        {
            "signal_id": "memory_routing_entropy",
            "measurement": (
                "Normalized Shannon entropy of controller routing actions in the "
                "row's fixture family."
            ),
            "exact_replay_fields": ["routing_rows[].routing_decision", "routing_decision"],
            "policy_use": "Expose unstable routing memory as a diagnostic signal.",
        },
    ]


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load checked-in source artifacts for Exp 3157."""

    root_path = Path(root)
    return {
        "exp3156": read_json_object(root_path / EXP3156_REL_PATH),
        "exp3143": read_json_object(root_path / EXP3143_REL_PATH),
        "exp3136": read_json_object(root_path / EXP3136_REL_PATH),
    }


def audit_residual_memory(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """REQ-LEARN-3157-2/3/4/5: replay residual memory over the closure panel."""

    exp3156 = sources.get("exp3156", {})
    exp3143 = sources.get("exp3143", {})
    exp3136 = sources.get("exp3136", {})
    closure_rows = [
        dict(row) for row in exp3156.get("replay_panel_rows", []) if isinstance(row, Mapping)
    ]
    routing_by_id = row_lookup(exp3143)
    risky_families = risky_family_set(closure_rows, exp3136)
    mismatch_counts = mismatch_count_by_family(closure_rows, exp3136)
    source_routes = [
        source_routing_decision(row, routing_by_id)
        for row in closure_rows
        if str(row.get("row_id") or "")
    ]
    family_routes = routes_by_family(closure_rows, routing_by_id)
    enriched_rows: list[JsonDict] = []
    for row in closure_rows:
        row_id = str(row.get("row_id") or "")
        family = normalize_token(row.get("fixture_family") or "unknown")
        repeated_mismatches = int(mismatch_counts.get(family, 0))
        source_route = source_routing_decision(row, routing_by_id)
        stable_verdict = stable_verdict_convergence(row)
        core_stable = contradiction_core_stability(row)
        risky_family = family in risky_families
        residual_route = residual_policy_route(
            source_route=source_route,
            risky_family=risky_family,
            repeated_mismatch_count=repeated_mismatches,
            stable_verdict=stable_verdict,
            contradiction_core_stable=core_stable,
        )
        unsafe_skip = unsafe_skip_detected(
            residual_route=residual_route,
            expected_action=normalize_action(row.get("expected_action")),
            stable_verdict=stable_verdict,
            risky_family=risky_family,
        )
        enriched_rows.append(
            dict(row)
            | {
                "row_id": row_id,
                "fixture_family": family,
                "source_routing_decision": source_route,
                "residual_policy_route": residual_route,
                "repeated_mismatch_count": repeated_mismatches,
                "stable_verdict_convergence": stable_verdict,
                "contradiction_core_stability": core_stable,
                "risky_family": risky_family,
                "memory_routing_entropy": routing_entropy(family_routes.get(family, [])),
                "unsafe_skip": unsafe_skip,
            }
        )

    replay_panel_count = len(enriched_rows)
    risky_rows = [row for row in enriched_rows if row["risky_family"]]
    risky_escalated = [row for row in risky_rows if row["residual_policy_route"] == "escalate"]
    safe_suppressed = [
        row
        for row in enriched_rows
        if row["residual_policy_route"] == "suppress" and not row["unsafe_skip"]
    ]
    unsafe_skip_count = sum(1 for row in enriched_rows if row["unsafe_skip"])
    return {
        "residual_memory_rows": enriched_rows,
        "replay_panel_count": replay_panel_count,
        "risky_family_count": len(risky_rows),
        "risky_family_escalated_count": len(risky_escalated),
        "risky_family_escalation_rate": rate(len(risky_escalated), len(risky_rows)),
        "redundant_check_suppression_rate": rate(len(safe_suppressed), replay_panel_count),
        "suppressed_redundant_check_count": len(safe_suppressed),
        "unsafe_skip_count": unsafe_skip_count,
        "memory_routing_entropy": routing_entropy(source_routes),
        "ledger_consistency_rate": float(exp3156.get("ledger_consistency_rate") or 0.0),
        "risky_families": sorted(risky_families),
    }


def row_lookup(payload: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Return row dictionaries keyed by row id."""

    rows: dict[str, JsonDict] = {}
    for row in payload.get("routing_rows", []):
        if isinstance(row, Mapping):
            row_id = str(row.get("row_id") or "")
            if row_id:
                rows[row_id] = dict(row)
    return rows


def source_routing_decision(
    row: Mapping[str, Any],
    routing_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return the prior memory route when present, else the closure route."""

    row_id = str(row.get("row_id") or "")
    routing_row = routing_by_id.get(row_id, {})
    return normalize_action(routing_row.get("routing_decision") or row.get("routing_decision"))


def routes_by_family(
    rows: Sequence[Mapping[str, Any]],
    routing_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Group source routing decisions by fixture family."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        family = normalize_token(row.get("fixture_family") or "unknown")
        grouped[family].append(source_routing_decision(row, routing_by_id))
    return dict(grouped)


def risky_family_set(
    closure_rows: Sequence[Mapping[str, Any]],
    exp3136: Mapping[str, Any],
) -> set[str]:
    """Find families with exact false-accept or ledger-mismatch evidence."""

    false_ids = {str(row_id) for row_id in exp3136.get("false_accept_row_ids", [])}
    risky: set[str] = set()
    for row in closure_rows:
        family = normalize_token(row.get("fixture_family") or "unknown")
        row_id = str(row.get("row_id") or "")
        if row_id in false_ids or row.get("consistent") is False:
            risky.add(family)
    for row in exp3136.get("verifier_rows", []):
        if not isinstance(row, Mapping):
            continue
        row_id = str(row.get("row_id") or "")
        expected = normalize_action(row.get("expected_action"))
        observed = normalize_action(row.get("live_decision"))
        if row_id in false_ids or (expected == "reject" and observed == "accept"):
            risky.add(normalize_token(row.get("fixture_family") or "unknown"))
    return risky


def mismatch_count_by_family(
    closure_rows: Sequence[Mapping[str, Any]],
    exp3136: Mapping[str, Any],
) -> Counter[str]:
    """Count exact residual mismatches by family without double-counting row ids."""

    false_ids = {str(row_id) for row_id in exp3136.get("false_accept_row_ids", [])}
    counts: Counter[str] = Counter()
    represented_ids: set[str] = set()
    for row in closure_rows:
        family = normalize_token(row.get("fixture_family") or "unknown")
        row_id = str(row.get("row_id") or "")
        if row.get("consistent") is False or row_id in false_ids:
            counts[family] += 1
            represented_ids.add(row_id)
    for row in exp3136.get("verifier_rows", []):
        if not isinstance(row, Mapping):
            continue
        row_id = str(row.get("row_id") or "")
        if row_id not in represented_ids and row_id in false_ids:
            counts[normalize_token(row.get("fixture_family") or "unknown")] += 1
    return counts


def stable_verdict_convergence(row: Mapping[str, Any]) -> bool:
    """Return whether expected, ledger, and observed actions share one verdict."""

    expected = normalize_action(row.get("expected_action"))
    ledger = normalize_action(row.get("ledger_action"))
    observed = normalize_action(row.get("observed_action"))
    return (
        row.get("consistent") is True
        and observed not in {"missing", "unknown"}
        and expected == ledger == observed
    )


def contradiction_core_stability(row: Mapping[str, Any]) -> bool:
    """Return whether exact replay avoided contradiction-core instability."""

    mismatch_class = normalize_token(row.get("mismatch_class") or "")
    return mismatch_class not in {"contradictory_memory", "monitor_replay_error"}


def residual_policy_route(
    *,
    source_route: str,
    risky_family: bool,
    repeated_mismatch_count: int,
    stable_verdict: bool,
    contradiction_core_stable: bool,
) -> str:
    """Simulate the bounded residual memory policy without hiding unsafe skips."""

    if source_route == "suppress":
        return "suppress"
    if (
        risky_family
        or repeated_mismatch_count > 0
        or not stable_verdict
        or not contradiction_core_stable
    ):
        return "escalate"
    if source_route == "escalate":
        return "escalate"
    return "normal"


def unsafe_skip_detected(
    *,
    residual_route: str,
    expected_action: str,
    stable_verdict: bool,
    risky_family: bool,
) -> bool:
    """REQ-LEARN-3157-5: count suppressed rows that would hide verifier work."""

    return residual_route == "suppress" and (
        expected_action == "reject" or not stable_verdict or risky_family
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3157 terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = precondition_blocker(
        sources["exp3156"],
        sources["exp3143"],
        sources["exp3136"],
    )
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    audit = audit_residual_memory(sources)
    ready = int(audit["replay_panel_count"]) > 0
    recommendation = promotion_recommendation(
        ready=ready,
        ledger_consistency_rate=float(audit["ledger_consistency_rate"]),
        unsafe_skip_count=int(audit["unsafe_skip_count"]),
    )
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_attractor_residual_memory_audit_v1_ready": ready,
        "continuous_self_learning_targeted": True,
        "residual_signal_definitions": residual_signal_definitions(),
        "replay_panel_count": int(audit["replay_panel_count"]),
        "risky_family_escalation_rate": float(audit["risky_family_escalation_rate"]),
        "redundant_check_suppression_rate": float(audit["redundant_check_suppression_rate"]),
        "unsafe_skip_count": int(audit["unsafe_skip_count"]),
        "promotion_recommendation": recommendation,
        "no_weight_update_claim": True,
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": inference_substrate(),
        "honest_verdict": honest_verdict(ready, recommendation),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "precondition_checks": precondition_checks(sources),
        "ledger_consistency_rate": float(audit["ledger_consistency_rate"]),
        "risky_family_count": int(audit["risky_family_count"]),
        "risky_family_escalated_count": int(audit["risky_family_escalated_count"]),
        "suppressed_redundant_check_count": int(audit["suppressed_redundant_check_count"]),
        "memory_routing_entropy": float(audit["memory_routing_entropy"]),
        "risky_families": audit["risky_families"],
        "residual_memory_rows": audit["residual_memory_rows"],
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
    """Return a schema-complete blocked artifact when sources are absent."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_attractor_residual_memory_audit_v1_ready": False,
        "continuous_self_learning_targeted": True,
        "residual_signal_definitions": residual_signal_definitions(),
        "replay_panel_count": 0,
        "risky_family_escalation_rate": 0.0,
        "redundant_check_suppression_rate": 0.0,
        "unsafe_skip_count": 0,
        "promotion_recommendation": "block_fr11_residual_memory_missing_source_evidence",
        "no_weight_update_claim": True,
        "source_artifacts": source_artifacts(root),
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "honest_verdict": f"blocked_precondition_failed: {blocker}",
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "precondition_checks": {},
        "ledger_consistency_rate": 0.0,
        "risky_family_count": 0,
        "risky_family_escalated_count": 0,
        "suppressed_redundant_check_count": 0,
        "memory_routing_entropy": 0.0,
        "risky_families": [],
        "residual_memory_rows": [],
        "blocked_reason": blocker,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3157 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def precondition_blocker(
    exp3156: Mapping[str, Any],
    exp3143: Mapping[str, Any],
    exp3136: Mapping[str, Any],
) -> str:
    """Return the first missing residual audit source."""

    if exp3156.get("fr11_ledger_consistency_closure_v1_ready") is not True:
        return "exp3156_ledger_closure_missing_or_not_ready"
    if exp3143.get("fr11_experience_verifier_memory_v1_ready") is not True:
        return "exp3143_experience_memory_missing_or_not_ready"
    if exp3136.get("false_accept_autopsy_v1_ready") is not True:
        return "exp3136_false_accept_autopsy_missing_or_not_ready"
    return ""


def precondition_checks(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Expose source readiness in the terminal artifact."""

    return {
        "exp3156_ledger_closure_ready": sources["exp3156"].get(
            "fr11_ledger_consistency_closure_v1_ready"
        )
        is True,
        "exp3143_experience_memory_ready": sources["exp3143"].get(
            "fr11_experience_verifier_memory_v1_ready"
        )
        is True,
        "exp3136_false_accept_autopsy_ready": sources["exp3136"].get(
            "false_accept_autopsy_v1_ready"
        )
        is True,
    }


def promotion_recommendation(
    ready: bool,
    ledger_consistency_rate: float,
    unsafe_skip_count: int,
) -> str:
    """REQ-LEARN-3157-5: apply residual-memory promotion gates."""

    if not ready:
        return "block_fr11_residual_memory_audit_incomplete"
    if unsafe_skip_count > 0:
        return "block_fr11_residual_memory_unsafe_skip_detected"
    if ledger_consistency_rate < 1.0:
        return "block_fr11_promotion_until_ledger_consistency_reaches_1.0"
    return "promote_controller_residual_memory_diagnostics_only"


def inference_substrate(mode: str = "exact_replay_residual_controller_memory") -> JsonDict:
    """Declare that this audit uses no live LLM inference or weight updates."""

    return {
        "mode": mode,
        "controller_residual_memory_only": True,
        "controller_routing_memory_only": True,
        "uses_checked_in_artifacts_only": True,
        "executes_exact_replay": True,
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
    """Raise when the Exp 3157 artifact violates the residual-memory contract."""

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
    for field in ("risky_family_escalation_rate", "redundant_check_suppression_rate"):
        value = float(artifact.get(field, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be finite and within [0, 1]")
    if artifact.get("fr11_attractor_residual_memory_audit_v1_ready") is not True:
        return
    if int(artifact.get("replay_panel_count") or 0) <= 0:
        raise ValueError("replay_panel_count must be positive for readiness")
    if int(artifact.get("unsafe_skip_count") or 0) > 0 and not str(
        artifact.get("promotion_recommendation") or ""
    ).startswith("block_fr11_residual_memory_unsafe_skip"):
        raise ValueError("promotion_recommendation must block unsafe skips")
    if float(artifact.get("ledger_consistency_rate") or 0.0) < 1.0 and not str(
        artifact.get("promotion_recommendation") or ""
    ).startswith("block_fr11"):
        raise ValueError("promotion_recommendation must block imperfect ledgers")
    if any(
        row.get("required") and not row.get("exists")
        for row in artifact.get("source_artifacts", [])
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    signal_ids = {str(item.get("signal_id") or "") for item in residual_signal_definitions()}
    artifact_signal_ids = {
        str(item.get("signal_id") or "")
        for item in artifact.get("residual_signal_definitions", [])
        if isinstance(item, Mapping)
    }
    if signal_ids - artifact_signal_ids:
        raise ValueError("residual_signal_definitions missing required signals")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def honest_verdict(ready: bool, recommendation: str) -> str:
    """Return a conductor-compatible terminal verdict."""

    if ready:
        return (
            "complete: fr11 attractor residual memory audit finished; "
            f"promotion_recommendation={recommendation}; no model-weight update claimed"
        )
    return "blocked_precondition_failed: fr11 attractor residual memory sources missing"


def routing_entropy(routes: Sequence[str]) -> float:
    """Return normalized Shannon entropy for routing tokens."""

    if not routes:
        return 0.0
    counts = Counter(normalize_action(route) for route in routes)
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round_float(entropy / math.log2(3))


def normalize_action(value: Any) -> str:
    """Normalize routing and verdict tokens used by prior artifacts."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def normalize_token(value: Any) -> str:
    """Normalize family and mismatch labels for stable grouping."""

    return normalize_action(value)


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


if __name__ == "__main__":  # pragma: no cover
    write_artifact()
