"""Build the Exp 3140 repair-gate unlock decision artifact.

Spec refs: REQ-VERIFY-3140, SCENARIO-VERIFY-3140.

This module is an aggregation gate, not a repair runner. It carries the
false-accept, exact-label, monitor-ledger, live-rerun, and repair-row evidence
forward into one conductor-readable decision field. That keeps the expensive
repair path fail-closed unless the checked-in evidence says it is safe to spend
new live repair calls.
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
RUN_DATE = "20260526"
ARTIFACT = "experiment_3140_repair_gate_unlock_decision_v1"
SCHEMA = "carnot.repair_gate_unlock_decision.v1"
OUTPUT_REL_PATH = Path("results/experiment_3140_repair_gate_unlock_decision_v1.json")

EXP3139_REL_PATH = Path("results/experiment_3139_live_sota_verifier_rerun_v7.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3125_REL_PATH = Path(
    "results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json"
)
EXP3115_REL_PATH = Path("results/experiment_3115_explicit_repair_gate_micro_panel_v4.json")
REPAIR_TARGET_MANIFEST_REL_PATH = Path(
    "results/fragment_verification_pilot_3114/repair_target_manifest.jsonl"
)

FALSE_ACCEPT_GATE = 0.10
ALLOWED_STATES = {
    "unblocked",
    "blocked_false_accept",
    "blocked_missing_live_model",
    "blocked_missing_exact_labels",
    "blocked_other",
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
    "repair_gate_decision_v1_ready",
    "repair_gate_state",
    "false_accept_rate",
    "false_accept_gate_passed",
    "regression_rows_included",
    "exact_authority_ready",
    "monitor_ledger_ready",
    "selected_repair_rows",
    "repair_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3140_repair_gate_unlock_decision_v1.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3140_repair_gate_unlock_decision_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/repair_gate_unlock_decision_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_REL_PATHS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True),
    ("exp3139_live_verifier_rerun", EXP3139_REL_PATH, True),
    ("exp3137_exact_safe_contract", EXP3137_REL_PATH, True),
    ("exp3126_monitor_ledger", EXP3126_REL_PATH, True),
    ("exp3125_prefix_bound", EXP3125_REL_PATH, True),
    ("exp3115_prior_repair_panel", EXP3115_REL_PATH, True),
    ("exp3140_module", Path("python/carnot/verify/repair_gate_unlock_decision_v1.py"), False),
    (
        "exp3140_tests",
        Path("tests/python/test_experiment_3140_repair_gate_unlock_decision_v1.py"),
        False,
    ),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3140: build the terminal repair-gate decision artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3139 = read_json_object(root_path / EXP3139_REL_PATH)
    exp3137 = read_json_object(root_path / EXP3137_REL_PATH)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    exp3125 = read_json_object(root_path / EXP3125_REL_PATH)
    exp3115 = read_json_object(root_path / EXP3115_REL_PATH)
    manifest_rel_path = repair_manifest_path(exp3115)
    source_rows = source_artifacts(root_path, manifest_rel_path)
    source_missing = [
        row["path"] for row in source_rows if row["required"] is True and row["present"] is False
    ]

    false_accept_rate = finite_metric(exp3139.get("false_accept_rate"))
    false_accept_gate_passed = bool(
        math.isfinite(false_accept_rate) and false_accept_rate <= FALSE_ACCEPT_GATE
    )
    regression_rows_included = exp3139.get("regression_rows_included") is True
    known_blocked = known_false_accepts_blocked(exp3137, exp3139)
    exact_ready = exact_authority_ready(exp3139, exp3137, exp3125)
    monitor_ready = monitor_ledger_ready(exp3126, exp3137)
    live_ready = live_model_ready(exp3139)
    headline_blocks = headline_disqualifiers(exp3139, exp3137, exp3126, exp3125, exp3115)
    repair_rows = selected_repair_rows(read_jsonl_rows(root_path / manifest_rel_path))
    repair_rows_ready = repair_rows_have_constraints(repair_rows)
    blockers = repair_blockers(
        false_accept_rate=false_accept_rate,
        false_accept_gate_passed=false_accept_gate_passed,
        regression_rows_included=regression_rows_included,
        known_false_accepts_blocked=known_blocked,
        exact_authority_ready=exact_ready,
        monitor_ledger_ready=monitor_ready,
        live_model_ready=live_ready,
        repair_rows_ready=repair_rows_ready,
        headline_disqualifiers=headline_blocks,
        source_missing=source_missing,
    )
    state = repair_gate_state(
        false_accept_gate_passed=false_accept_gate_passed,
        live_model_ready=live_ready,
        exact_authority_ready=exact_ready,
        blockers=blockers,
    )
    selected_rows = repair_rows if state == "unblocked" else []
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "repair_gate_decision_v1_ready": state in ALLOWED_STATES
        and (state == "unblocked" or bool(blockers)),
        "repair_gate_state": state,
        "false_accept_rate": false_accept_rate,
        "false_accept_gate_passed": false_accept_gate_passed,
        "regression_rows_included": regression_rows_included,
        "exact_authority_ready": exact_ready,
        "monitor_ledger_ready": monitor_ready,
        "known_false_accepts_blocked": known_blocked,
        "live_model_ready": live_ready,
        "headline_disqualifiers": headline_blocks,
        "repair_rows_available": repair_rows_ready,
        "selected_repair_rows": selected_rows,
        "repair_blockers": blockers,
        "gate_criteria": {
            "max_false_accept_rate": FALSE_ACCEPT_GATE,
            "regression_rows_required": True,
            "known_false_accepts_must_be_blocked": True,
            "exact_labels_required": True,
            "monitor_ledger_replay_required": True,
            "bounded_live_model_rerun_required": True,
            "headline_disqualifiers_allowed": False,
        },
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row.get("sha256")
        },
        "inference_substrate": inference_substrate(exp3139, exp3115),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
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
    """Build, validate, and persist the Exp 3140 decision JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def repair_manifest_path(exp3115: Mapping[str, Any]) -> Path:
    """Return the repair-target manifest path named by the prior repair panel."""

    return Path(str(exp3115.get("repair_target_manifest_path") or REPAIR_TARGET_MANIFEST_REL_PATH))


def selected_repair_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize repair rows into the explicit denominator a repair rerun may use."""

    selected: list[JsonDict] = []
    for index, row in enumerate(rows):
        evidence = row.get("solver_evidence")
        selected.append(
            {
                "row_index": index,
                "fixture_id": str(row.get("fixture_id") or ""),
                "fragment_id": str(row.get("fragment_id") or ""),
                "constraints": {
                    "failing_constraint": str(row.get("failing_constraint") or ""),
                    "expected_direction": str(row.get("expected_direction") or ""),
                    "solver_evidence": dict(evidence) if isinstance(evidence, Mapping) else {},
                },
            }
        )
    return selected


def repair_rows_have_constraints(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether every selected repair row has the constraints needed by repair."""

    return bool(rows) and all(
        row.get("fixture_id")
        and row.get("fragment_id")
        and isinstance(row.get("constraints"), Mapping)
        and row["constraints"].get("failing_constraint")
        and row["constraints"].get("expected_direction")
        for row in rows
    )


def false_accept_regression_ids(exp3137: Mapping[str, Any], exp3139: Mapping[str, Any]) -> set[str]:
    """Return known false-accept row IDs that must not receive accept decisions."""

    ids = {str(row_id) for row_id in exp3137.get("regression_row_set") or [] if str(row_id)}
    for row in mapping_rows(exp3139.get("rerun_rows")):
        if row.get("is_regression_row") is True:
            ids.add(row_id_from(row))
    return ids


def known_false_accepts_blocked(exp3137: Mapping[str, Any], exp3139: Mapping[str, Any]) -> bool:
    """Return whether Exp 3137 and the rerun both block known false accepts."""

    if exp3137.get("known_false_accept_rows_blocked") is not True:
        return False
    ids = false_accept_regression_ids(exp3137, exp3139)
    if not ids:
        return False
    for row in mapping_rows(exp3139.get("rerun_rows")):
        if row_id_from(row) in ids and row.get("contract_decision") == "accept":
            return False
    return True


def exact_labels_present(exp3139: Mapping[str, Any]) -> bool:
    """Return whether the live rerun carries explicit exact labels for scored rows."""

    rows = mapping_rows(exp3139.get("rerun_rows"))
    return int(exp3139.get("exact_ground_truth_count") or 0) > 0 and bool(rows) and all(
        str(row.get("exact_label") or "") for row in rows
    )


def exact_authority_ready(
    exp3139: Mapping[str, Any],
    exp3137: Mapping[str, Any],
    exp3125: Mapping[str, Any],
) -> bool:
    """Return whether exact labels, exact-safe contract, and prefix semantics are ready."""

    coverage = exp3125.get("semantic_coverage")
    label_coverage = coverage.get("answer_label_semantics") if isinstance(coverage, Mapping) else {}
    prerequisites = exp3137.get("repair_gate_prerequisites")
    prereq = prerequisites if isinstance(prerequisites, Mapping) else {}
    return bool(
        exp3137.get("acceptance_contract_v1_ready") is True
        and finite_metric(exp3137.get("replay_false_accept_rate")) == 0.0
        and prereq.get("require_exact_label_authority") is True
        and exp3125.get("prefix_closed_bound_pilot_ready") is True
        and isinstance(label_coverage, Mapping)
        and label_coverage.get("covered") is True
        and exact_labels_present(exp3139)
    )


def monitor_ledger_ready(exp3126: Mapping[str, Any], exp3137: Mapping[str, Any]) -> bool:
    """Return whether repair can replay the fragment-time monitor ledger."""

    constraints = exp3126.get("downstream_repair_constraints")
    repair_constraints = constraints if isinstance(constraints, Mapping) else {}
    summary = exp3126.get("ledger_replay_summary")
    ledger_summary = summary if isinstance(summary, Mapping) else {}
    prerequisites = exp3137.get("repair_gate_prerequisites")
    prereq = prerequisites if isinstance(prerequisites, Mapping) else {}
    return bool(
        exp3126.get("fragment_time_monitor_v1_ready") is True
        and repair_constraints.get("must_replay_before_repair") is True
        and repair_constraints.get("repair_requires_monitor_evidence") is True
        and prereq.get("require_monitor_ledger_replay_for_live_rows") is True
        and int(ledger_summary.get("monitor_event_count") or 0) > 0
    )


def live_model_ready(exp3139: Mapping[str, Any]) -> bool:
    """Return whether Exp 3139 spent bounded live model calls on mandated models."""

    substrate = exp3139.get("inference_substrate")
    inference = substrate if isinstance(substrate, Mapping) else {}
    selected_ids = [str(model_id) for model_id in exp3139.get("selected_model_ids") or []]
    selected_model = inference.get("selected_model_id")
    return bool(
        int(exp3139.get("live_call_count") or 0) > 0
        and int(inference.get("live_model_calls") or 0) > 0
        and inference.get("executes_models") is True
        and (selected_ids or selected_model)
    )


def headline_disqualifiers(
    exp3139: Mapping[str, Any],
    exp3137: Mapping[str, Any],
    exp3126: Mapping[str, Any],
    exp3125: Mapping[str, Any],
    exp3115: Mapping[str, Any],
) -> list[str]:
    """Return source-level reasons that prevent headline-backed repair spending."""

    disqualifiers: list[str] = []
    if exp3139.get("headline_claim_allowed") is not True:
        disqualifiers.append("headline_claim_allowed is not true")
    substrate = exp3139.get("inference_substrate")
    if isinstance(substrate, Mapping) and substrate.get("uses_legacy_small_model_for_headline"):
        disqualifiers.append("legacy small model evidence is not headline eligible")
    for name, payload in (
        ("exp3139", exp3139),
        ("exp3137", exp3137),
        ("exp3126", exp3126),
        ("exp3125", exp3125),
        ("exp3115", exp3115),
    ):
        if payload.get("flagged_adversarial") is True:
            disqualifiers.append(f"{name} flagged_adversarial=true")
        if bool(payload.get("corrigendum_pending")):
            disqualifiers.append(f"{name} corrigendum_pending=true")
    return disqualifiers


def repair_blockers(
    *,
    false_accept_rate: float,
    false_accept_gate_passed: bool,
    regression_rows_included: bool,
    known_false_accepts_blocked: bool,
    exact_authority_ready: bool,
    monitor_ledger_ready: bool,
    live_model_ready: bool,
    repair_rows_ready: bool,
    headline_disqualifiers: Sequence[str],
    source_missing: Sequence[str],
) -> list[str]:
    """Build actionable blockers in the same order as the gate criteria."""

    blockers: list[str] = []
    if not false_accept_gate_passed:
        blockers.append(f"false_accept_rate={false_accept_rate} exceeds gate <= {FALSE_ACCEPT_GATE}")
    if not regression_rows_included:
        blockers.append("regression_rows_included is not true")
    if not known_false_accepts_blocked:
        blockers.append("known false accepts are not blocked from accept")
    if not live_model_ready:
        blockers.append("bounded live model rerun is missing")
    if not exact_authority_ready:
        blockers.append("exact authority is not ready")
    if not monitor_ledger_ready:
        blockers.append("monitor ledger replay is not ready")
    if not repair_rows_ready:
        blockers.append("selected repair row constraints are missing")
    blockers.extend(headline_disqualifiers)
    blockers.extend(f"required source missing: {path}" for path in source_missing)
    return blockers


def repair_gate_state(
    *,
    false_accept_gate_passed: bool,
    live_model_ready: bool,
    exact_authority_ready: bool,
    blockers: Sequence[str],
) -> str:
    """Collapse gate checks to the one conductor-facing repair state."""

    if not false_accept_gate_passed:
        return "blocked_false_accept"
    if not live_model_ready:
        return "blocked_missing_live_model"
    if not exact_authority_ready:
        return "blocked_missing_exact_labels"
    if blockers:
        return "blocked_other"
    return "unblocked"


def inference_substrate(exp3139: Mapping[str, Any], exp3115: Mapping[str, Any]) -> JsonDict:
    """Declare that this decision aggregates evidence without new live inference."""

    rerun_substrate = exp3139.get("inference_substrate")
    repair_substrate = exp3115.get("inference_substrate")
    rerun = rerun_substrate if isinstance(rerun_substrate, Mapping) else {}
    repair = repair_substrate if isinstance(repair_substrate, Mapping) else {}
    return {
        "kind": "deterministic_repair_gate_decision_v1",
        "executes_models": False,
        "executes_repairs": False,
        "no_live_inference": True,
        "live_model_calls": 0,
        "repair_calls": 0,
        "source_live_model_calls_reused": int(rerun.get("live_model_calls") or 0),
        "source_repair_run_executed": repair.get("repair_run_executed") is True,
        "aggregation_only": True,
    }


def source_artifacts(root: Path, manifest_rel_path: Path) -> list[JsonDict]:
    """Return checksummed source rows for every artifact the decision consumes."""

    rows = [source_row(root, role, rel_path, required) for role, rel_path, required in SOURCE_REL_PATHS]
    rows.append(source_row(root, "exp3115_repair_target_manifest", manifest_rel_path, True))
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


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and the fail-closed repair gate contract."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3140 artifact missing required fields: {missing}")  # pragma: no cover
    state = str(artifact.get("repair_gate_state") or "")
    if state not in ALLOWED_STATES:
        raise ValueError(f"repair_gate_state must be an allowed state, got {state!r}")
    false_accept_rate = finite_metric(artifact.get("false_accept_rate"))
    if not math.isfinite(false_accept_rate) or not 0.0 <= false_accept_rate <= 1.0:
        raise ValueError("false_accept_rate must be a finite rate in [0, 1]")  # pragma: no cover
    if artifact.get("false_accept_gate_passed") is True and false_accept_rate > FALSE_ACCEPT_GATE:
        raise ValueError("false_accept_gate_passed conflicts with false_accept_rate")  # pragma: no cover
    blockers = artifact.get("repair_blockers")
    selected_rows = artifact.get("selected_repair_rows")
    if state == "unblocked" and (blockers or not selected_rows):
        raise ValueError("unblocked gate requires selected rows and no blockers")  # pragma: no cover
    if state != "unblocked" and not blockers:
        raise ValueError("blocked gate requires repair_blockers")  # pragma: no cover
    substrate = artifact.get("inference_substrate")
    inference = substrate if isinstance(substrate, Mapping) else {}
    if inference.get("executes_models") or inference.get("live_model_calls") or inference.get("repair_calls"):
        raise ValueError("Exp 3140 must not execute live inference or repair calls")  # pragma: no cover
    verdict = str(artifact.get("honest_verdict") or "")
    if state == "unblocked" and not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("unblocked verdict must start with a success prefix")  # pragma: no cover
    if state != "unblocked" and not verdict.startswith(f"{state}:"):
        raise ValueError("blocked verdict must start with the blocked state")  # pragma: no cover


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return conductor-compatible terminal verdict wording."""

    state = str(artifact.get("repair_gate_state") or "blocked_other")
    if state == "unblocked":
        return (
            "complete: repair_gate_state=unblocked; "
            f"false_accept_rate={artifact.get('false_accept_rate')}; "
            f"selected_repair_rows={len(artifact.get('selected_repair_rows') or [])}"
        )
    blockers = artifact.get("repair_blockers")
    first_blocker = str(blockers[0]) if isinstance(blockers, list) and blockers else "missing blocker"
    return f"{state}: {first_blocker}"


def read_json_object(path: Path) -> JsonDict:
    """Read one local JSON object, making missing or malformed evidence non-promotable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive filesystem guard.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read object rows from a JSONL repair target manifest."""

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
        except json.JSONDecodeError:  # pragma: no cover - malformed rows are ignored.
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def mapping_rows(value: Any) -> list[JsonDict]:
    """Return only mapping rows from an arbitrary list-like value."""

    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def row_id_from(row: Mapping[str, Any]) -> str:
    """Return the stable row identifier shared by fixture and regression rows."""

    return str(row.get("fixture_id") or row.get("row_id") or row.get("source_fixture_id") or "")


def finite_metric(value: Any) -> float:
    """Return a float metric or NaN when the input is not numeric."""

    if isinstance(value, (float, int)):
        return float(value)
    return math.nan


def sha256_file(path: Path) -> str | None:
    """Checksum source bytes so the decision traces to exact local files."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative elapsed duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
