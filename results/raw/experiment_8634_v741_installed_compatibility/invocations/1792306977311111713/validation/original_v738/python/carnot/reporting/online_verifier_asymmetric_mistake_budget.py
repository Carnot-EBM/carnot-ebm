"""Exp 1472 online verifier asymmetric mistake-budget audit.

This module re-reads Exp 1471 under the online verifier learnability framing
from arXiv:2603.03538. The important distinction is directional: accepting a
bad case into memory is a soundness failure and can poison later feedback
loops, while rejecting a good case is a completeness failure and mainly makes
the verifier conservative. The audit therefore prices soundness mistakes more
heavily and preserves the self-learning claim only when the dangerous side is
empty.

Spec: REQ-LEARN-1472, SCENARIO-LEARN-1473, SCENARIO-LEARN-1474.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_NOTES_DIR = REPO_ROOT / "docs" / "research-notes"

OUTPUT_FILE = "experiment_1472_online_verifier_asymmetric_mistake_budget.json"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_EXP1471_PATH = DEFAULT_RESULTS_DIR / "experiment_1471_fr11_v8_verified_memory_growth_pivot.json"
DEFAULT_AUDIT_NOTE_PATH = DEFAULT_NOTES_DIR / "fr11_v8_asymmetric_mistake_budget.md"

EXPERIMENT = "1472_online_verifier_asymmetric_mistake_budget"
SCHEMA = "online_verifier_asymmetric_mistake_budget_v1"
RUN_DATE = "20260507"
SOURCE_EXPERIMENT = "experiment_1471_fr11_v8_verified_memory_growth_pivot"

SOUNDNESS_COST_WEIGHT = 10.0
COMPLETENESS_COST_WEIGHT = 1.0
ACCEPTABLE_SOUNDNESS_MISTAKE_BUDGET = 0

PARETO_PRESERVE = "preserve_narrow_claim_on_soundness_frontier_with_completeness_caveat"
PARETO_RETIRE_SOUNDNESS = "retire_claim_soundness_risk_exceeds_budget"
PARETO_RETIRE_SOURCE = "retire_claim_source_experiment_gate_failed"

PRESERVED_VERDICT = "self_learning_claim_preserved_zero_soundness_mistakes"
SOURCE_FAILED_VERDICT = "self_learning_claim_retired_source_gate_failed"
SOUNDNESS_FAILED_VERDICT = "self_learning_claim_retired_soundness_risk"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "source_experiment",
    "soundness_mistakes",
    "completeness_mistakes",
    "asymmetric_cost_weights",
    "asymmetric_cost_score",
    "pareto_decision",
    "self_learning_claim_preserved",
    "self_learning_claim_retired",
    "audit_note_path",
    "honest_verdict",
)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _to_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")  # pragma: no cover
    return payload


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1472-1: write the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec": ["REQ-LEARN-1472", "SCENARIO-LEARN-1473", "SCENARIO-LEARN-1474"],
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "source_experiment": SOURCE_EXPERIMENT,
            "soundness_mistakes": None,
            "completeness_mistakes": None,
            "asymmetric_cost_weights": {
                "soundness": SOUNDNESS_COST_WEIGHT,
                "completeness": COMPLETENESS_COST_WEIGHT,
            },
            "asymmetric_cost_score": None,
            "pareto_decision": None,
            "self_learning_claim_preserved": False,
            "self_learning_claim_retired": False,
            "audit_note_path": None,
            "honest_verdict": "in_progress",
        },
    )


def _source_experiment_name(exp1471_artifact: Mapping[str, Any]) -> str:
    raw = str(exp1471_artifact.get("experiment") or SOURCE_EXPERIMENT)
    return raw if raw.startswith("experiment_") else f"experiment_{raw}"


def _ledger_summary(exp1471_artifact: Mapping[str, Any]) -> dict[str, Any]:
    updates = exp1471_artifact.get("memory_updates")
    memory_updates = updates if isinstance(updates, Mapping) else {}
    promoted = _as_sequence(memory_updates.get("promoted"))
    demoted = _as_sequence(memory_updates.get("demoted"))
    rejection_counts = memory_updates.get("rejection_reason_counts")
    rejections = rejection_counts if isinstance(rejection_counts, Mapping) else {}
    return {
        "promoted_memory_count": _to_int(
            memory_updates.get("promoted_memory_count"),
            len(promoted),
        ),
        "demoted_memory_count": _to_int(
            memory_updates.get("demoted_memory_count"),
            len(demoted),
        ),
        "promoted_ledger_count": len(promoted),
        "demoted_ledger_count": len(demoted),
        "verifier_rejection_count": _to_int(rejections.get("verifier_rejection")),
    }


def _asymmetric_cost_weights() -> dict[str, Any]:
    return {
        "soundness": SOUNDNESS_COST_WEIGHT,
        "completeness": COMPLETENESS_COST_WEIGHT,
        "rationale": (
            "Soundness mistakes are false accepts into the feedback loop; "
            "completeness mistakes are conservative false flags."
        ),
    }


def compute_asymmetric_cost(soundness_mistakes: int, completeness_mistakes: int) -> float:
    """REQ-LEARN-1472-3: compute weighted asymmetric mistake cost."""

    return float(
        soundness_mistakes * SOUNDNESS_COST_WEIGHT
        + completeness_mistakes * COMPLETENESS_COST_WEIGHT
    )


def _source_gate_passed(exp1471_artifact: Mapping[str, Any]) -> bool:
    return (
        exp1471_artifact.get("status") == "complete"
        and exp1471_artifact.get("headline_result_allowed") is True
        and exp1471_artifact.get("pivot_preserved") is True
    )


def _soundness_risk_acceptable(soundness_mistakes: int) -> bool:
    return soundness_mistakes <= ACCEPTABLE_SOUNDNESS_MISTAKE_BUDGET


def _pareto_decision(*, source_gate_passed: bool, soundness_risk_acceptable: bool) -> str:
    if not source_gate_passed:
        return PARETO_RETIRE_SOURCE
    if not soundness_risk_acceptable:
        return PARETO_RETIRE_SOUNDNESS
    return PARETO_PRESERVE


def _honest_verdict(pareto_decision: str) -> str:
    if pareto_decision == PARETO_RETIRE_SOURCE:
        return SOURCE_FAILED_VERDICT
    if pareto_decision == PARETO_RETIRE_SOUNDNESS:
        return SOUNDNESS_FAILED_VERDICT
    return PRESERVED_VERDICT


def build_artifact(
    *,
    exp1471_artifact: Mapping[str, Any],
    audit_note_path: Path | str,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1472-2/3/4/5: build the terminal mistake-budget artifact."""

    soundness_mistakes = _to_int(exp1471_artifact.get("soundness_mistakes"))
    completeness_mistakes = _to_int(exp1471_artifact.get("completeness_mistakes"))
    ledger_summary = _ledger_summary(exp1471_artifact)
    ledger_summary["mismatch_count_from_fields"] = soundness_mistakes + completeness_mistakes
    ledger_summary["demotion_count_matches_mistake_count"] = (
        ledger_summary["demoted_memory_count"] == ledger_summary["mismatch_count_from_fields"]
    )
    source_gate_passed = _source_gate_passed(exp1471_artifact)
    soundness_risk_acceptable = _soundness_risk_acceptable(soundness_mistakes)
    pareto_decision = _pareto_decision(
        source_gate_passed=source_gate_passed,
        soundness_risk_acceptable=soundness_risk_acceptable,
    )
    preserved = pareto_decision == PARETO_PRESERVE

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1472", "SCENARIO-LEARN-1473", "SCENARIO-LEARN-1474"],
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "source_experiment": _source_experiment_name(exp1471_artifact),
        "source_experiment_path": str(DEFAULT_EXP1471_PATH.relative_to(REPO_ROOT)),
        "source_status": str(exp1471_artifact.get("status") or ""),
        "source_headline_result_allowed": bool(exp1471_artifact.get("headline_result_allowed")),
        "source_pivot_preserved": bool(exp1471_artifact.get("pivot_preserved")),
        "source_nonforgetting_rate": _to_float(exp1471_artifact.get("nonforgetting_rate")),
        "source_self_learning_delta_overall": _to_int(
            exp1471_artifact.get("self_learning_delta_overall")
        ),
        "soundness_mistakes": soundness_mistakes,
        "completeness_mistakes": completeness_mistakes,
        "acceptable_soundness_mistake_budget": ACCEPTABLE_SOUNDNESS_MISTAKE_BUDGET,
        "soundness_risk_acceptable": soundness_risk_acceptable,
        "asymmetric_cost_weights": _asymmetric_cost_weights(),
        "asymmetric_cost_score": compute_asymmetric_cost(
            soundness_mistakes,
            completeness_mistakes,
        ),
        "score_formula": "10.0 * soundness_mistakes + 1.0 * completeness_mistakes",
        "ledger_summary": ledger_summary,
        "pareto_decision": pareto_decision,
        "self_learning_claim_preserved": preserved,
        "self_learning_claim_retired": not preserved,
        "audit_note_path": str(audit_note_path),
        "caveats": [
            (
                "The available per-case ledger records promoted/demoted IDs and "
                "aggregate rejection reasons, not full per-row semantic states."
            ),
            (
                "Completeness mistakes are conservative false flags; they limit "
                "usable growth but are not evidence of poisoned memory promotion."
            ),
            "The preserved claim is the narrow Exp 1471 memory-growth claim only.",
        ],
        "commands_run": list(commands_run or []),
        "honest_verdict": _honest_verdict(pareto_decision),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1472-5: enforce the final artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] == "in_progress":
        return

    weights = artifact["asymmetric_cost_weights"]
    soundness_weight = float(weights["soundness"])
    completeness_weight = float(weights["completeness"])
    if soundness_weight <= completeness_weight:
        raise AssertionError("soundness cost weight must exceed completeness cost weight")

    soundness_mistakes = int(artifact["soundness_mistakes"])
    completeness_mistakes = int(artifact["completeness_mistakes"])
    expected_score = soundness_mistakes * soundness_weight + completeness_mistakes * completeness_weight
    if float(artifact["asymmetric_cost_score"]) != expected_score:
        raise AssertionError("asymmetric_cost_score must match weighted mistake counts")

    preserved = bool(artifact["self_learning_claim_preserved"])
    retired = bool(artifact["self_learning_claim_retired"])
    if preserved == retired:
        raise AssertionError("exactly one of preservation or retirement must be true")


def write_audit_note(artifact: Mapping[str, Any], path: Path | str) -> str:
    """REQ-LEARN-1472-6: write the Markdown decision note and caveats."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    decision = (
        "Preserve narrow self-learning claim"
        if artifact["self_learning_claim_preserved"]
        else "Retire self-learning claim"
    )
    content = f"""# FR-11 V8 Asymmetric Mistake-Budget Audit

Run date: {artifact['run_date']}
Decision: {decision}
Honest verdict: {artifact['honest_verdict']}

## Audit Framing

This is an asymmetric mistake-budget audit for Exp 1471 under the online
verifier learnability framing in arXiv:2603.03538. Soundness mistakes are
dangerous missed errors that can enter the feedback loop. Completeness mistakes
are conservative false flags that withhold usable cases.

## Evidence

- Source experiment: `{artifact['source_experiment']}`
- Source status: `{artifact['source_status']}`
- Source headline gate: `{artifact['source_headline_result_allowed']}`
- Source pivot preserved: `{artifact['source_pivot_preserved']}`
- Soundness mistakes: `{artifact['soundness_mistakes']}`
- Completeness mistakes: `{artifact['completeness_mistakes']}`
- Cost weights: `soundness={artifact['asymmetric_cost_weights']['soundness']}`,
  `completeness={artifact['asymmetric_cost_weights']['completeness']}`
- Asymmetric cost score: `{artifact['asymmetric_cost_score']}`
- Pareto decision: `{artifact['pareto_decision']}`

## Caveats

- Exp 1471 has aggregate soundness/completeness fields and a promoted/demoted
  memory ledger, but not full per-row semantic-state detail in the result JSON.
- The 140 conservative false flags in the live artifact are a completeness and
  candidate-supply limitation, not evidence of memory poisoning.
- The preserved claim, if preserved, is only the narrow Exp 1471 verified
  memory-growth claim. It is not a broad claim that online verifier learning is
  generally complete.
"""
    destination.write_text(content, encoding="utf-8")
    return content


def run(
    *,
    exp1471_path: Path | str = DEFAULT_EXP1471_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    audit_note_path: Path | str = DEFAULT_AUDIT_NOTE_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1472 and write the terminal JSON plus audit note."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    artifact = build_artifact(
        exp1471_artifact=load_json(exp1471_path),
        audit_note_path=audit_note_path,
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=commands_run,
    )
    write_audit_note(artifact, audit_note_path)
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
