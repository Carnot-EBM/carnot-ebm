"""Build the Exp 1358 verifier-selected continuous self-learning artifact.

Spec: REQ-LEARN-1358, SCENARIO-LEARN-1358.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILE = "experiment_1358_continuous_self_learning_verifier_selected_memory.json"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1358_continuous_self_learning_verifier_selected_memory"
SCHEMA = "continuous_self_learning_verifier_selected_memory_v1"
RUN_DATE = "20260505"

EXP1344_REQUESTED_FILE = "experiment_1344_failure_type_memory_policy_self_learning.json"
EXP1344_FALLBACK_FILE = "experiment_1344_continuous_self_learning_failure_type_memory_policy.json"
EXP1353_FILE = "experiment_1353_triggered_certificate_v7_truncproof_sota.json"
EXP1355_FILE = "experiment_1355_logitext_nsvif_partial_smt_validator.json"

POLICY_PROMOTE = "promote"
POLICY_DEMOTE = "demote"
POLICY_QUARANTINE = "quarantine"
POLICY_REQUEST_FRESH_VERIFIER = "request_fresh_verifier"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "replay_cases_used",
    "fresh_verified_sample_count",
    "variant_question_count",
    "self_learning_delta_overall",
    "nonforgetting_certificate_rate",
    "memory_regression_count",
    "accepted_violation_delta",
    "promoted_memory_count",
    "demoted_memory_count",
    "dvi_ready",
    "headline_result_allowed",
    "honest_verdict",
)

_SOURCE_FILES = {
    "exp1344": (EXP1344_REQUESTED_FILE, EXP1344_FALLBACK_FILE),
    "exp1353": (EXP1353_FILE,),
    "exp1355": (EXP1355_FILE,),
}


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1358-1: persist the bootstrap marker before input loading.

    The conductor can be interrupted between artifact creation and source
    loading. Writing the marker first makes the run state inspectable instead
    of leaving a missing deliverable that looks like the task never started.
    """

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifacts": [],
            "input_resolution": {},
            "inputs_unavailable": [],
            "status": "in_progress",
            "replay_cases_used": 0,
            "fresh_verified_sample_count": 0,
            "variant_question_count": 0,
            "self_learning_delta_overall": 0.0,
            "nonforgetting_certificate_rate": 0.0,
            "memory_regression_count": 0,
            "accepted_violation_delta": 0.0,
            "promoted_memory_count": 0,
            "demoted_memory_count": 0,
            "dvi_ready": False,
            "headline_result_allowed": False,
            "honest_verdict": "in_progress",
        },
    )


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1358-1/2: write bootstrap, load evidence, and finalize."""

    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    payloads, unavailable_inputs, input_resolution, source_artifacts = load_inputs(results_dir)
    artifact = build_artifact(
        exp1344_artifact=payloads.get("exp1344", {}),
        exp1353_artifact=payloads.get("exp1353", {}),
        exp1355_artifact=payloads.get("exp1355", {}),
        unavailable_inputs=unavailable_inputs,
        input_resolution=input_resolution,
        source_artifacts=source_artifacts,
        project_root=project_root,
        run_date=run_date,
    )
    validate_artifact(artifact)
    return _write_json(out_path, artifact)


def load_inputs(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, dict[str, str | None]], list[str]]:
    """Load source artifacts and keep missing optional verifier evidence auditable."""

    results_path = Path(results_dir)
    payloads: dict[str, dict[str, Any]] = {}
    unavailable: list[str] = []
    resolution: dict[str, dict[str, str | None]] = {}
    sources: list[str] = []

    for key, candidates in _SOURCE_FILES.items():
        requested = candidates[0]
        used: str | None = None
        for index, filename in enumerate(candidates):
            path = results_path / filename
            if path.exists():
                payloads[key] = json.loads(path.read_text(encoding="utf-8"))
                used = f"results/{filename}"
                sources.append(used)
                break
            if index == 0:
                unavailable.append(f"results/{filename}")
        if used is None:
            for fallback in candidates[1:]:
                unavailable.append(f"results/{fallback}")
        resolution[key] = {"requested": f"results/{requested}", "used": used}

    return payloads, unavailable, resolution, sources


def build_artifact(
    *,
    exp1344_artifact: Mapping[str, Any],
    exp1353_artifact: Mapping[str, Any],
    exp1355_artifact: Mapping[str, Any],
    unavailable_inputs: Sequence[str] = (),
    input_resolution: Mapping[str, Mapping[str, str | None]] | None = None,
    source_artifacts: Sequence[str] | None = None,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1358-3/6: compute memory updates and headline gating."""

    fresh_samples = extract_fresh_verified_samples(exp1353_artifact, exp1355_artifact)
    variants = build_variant_questions(fresh_samples, exp1344_artifact)
    memory_updates = apply_memory_updates(variants)
    fresh_verified_count = len([sample for sample in fresh_samples if sample["verifier_accepted"]])
    update_is_replay_only = fresh_verified_count == 0

    self_learning_delta = _float(exp1344_artifact.get("self_learning_delta_overall"), 0.0)
    nonforgetting_rate = _float(exp1344_artifact.get("nonforgetting_certificate_rate"), 0.0)
    regression_count = _int(exp1344_artifact.get("memory_regression_count"))
    accepted_violation_delta = _float(exp1344_artifact.get("accepted_violation_delta"), 0.0)
    dvi_ready = derive_dvi_ready(
        self_learning_delta_overall=self_learning_delta,
        accepted_violation_delta=accepted_violation_delta,
        nonforgetting_certificate_rate=nonforgetting_rate,
        memory_regression_count=regression_count,
    )
    headline_allowed = bool(dvi_ready and fresh_verified_count > 0 and not update_is_replay_only)

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "source_artifacts": list(source_artifacts or _default_source_artifacts()),
        "source_honest_verdicts": {
            "exp1344": exp1344_artifact.get("honest_verdict"),
            "exp1353": exp1353_artifact.get("honest_verdict"),
            "exp1355": exp1355_artifact.get("honest_verdict"),
        },
        "input_resolution": dict(input_resolution or {}),
        "inputs_unavailable": list(unavailable_inputs),
        "status": "complete",
        "replay_cases_used": _int(exp1344_artifact.get("replay_cases_used")),
        "fresh_verified_sample_count": fresh_verified_count,
        "variant_question_count": len(variants),
        "self_learning_delta_overall": self_learning_delta,
        "nonforgetting_certificate_rate": nonforgetting_rate,
        "memory_regression_count": regression_count,
        "accepted_violation_delta": accepted_violation_delta,
        "promoted_memory_count": memory_updates["promoted_memory_count"],
        "demoted_memory_count": memory_updates["demoted_memory_count"],
        "quarantined_memory_count": memory_updates["quarantined_memory_count"],
        "memory_updates": memory_updates,
        "variant_questions": variants,
        "fresh_verified_samples": fresh_samples,
        "update_is_replay_only": update_is_replay_only,
        "dvi_ready": dvi_ready,
        "headline_result_allowed": headline_allowed,
        "honest_verdict": derive_honest_verdict(
            dvi_ready=dvi_ready,
            update_is_replay_only=update_is_replay_only,
            headline_result_allowed=headline_allowed,
        ),
        "measurement_note": (
            "Fresh verifier-selected samples are required for headline claims. "
            "When none pass verifier acceptance, Exp 1344 replay evidence is "
            "used only as a non-headline fallback baseline."
        ),
    }


def extract_fresh_verified_samples(
    exp1353_artifact: Mapping[str, Any],
    exp1355_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """REQ-LEARN-1358-3: accept only terminal verifier-approved fresh rows."""

    samples: list[dict[str, Any]] = []
    if exp1353_artifact.get("status") == "complete":
        for row in _rows(exp1353_artifact, ("certificate_rows",)):
            if _exp1353_row_accepted(row):
                samples.append(
                    _fresh_sample(
                        source="exp1353",
                        case_id=str(row.get("case_id") or f"exp1353-{len(samples)}"),
                        verifier_accepted=True,
                        semantic_rejected=False,
                        support=1,
                        evidence=row,
                    )
                )

    if exp1355_artifact.get("status") == "complete":
        for row in _rows(
            exp1355_artifact,
            ("semantic_validator_rows", "validator_rows", "case_rows", "cases", "results"),
        ):
            accepted = _truthy(
                row.get("verifier_accepted"),
                row.get("accepted"),
                row.get("semantic_valid"),
                row.get("passed"),
            )
            rejected = _semantic_rejected(row)
            if accepted or rejected:
                samples.append(
                    _fresh_sample(
                        source="exp1355",
                        case_id=str(
                            row.get("case_id") or row.get("id") or f"exp1355-{len(samples)}"
                        ),
                        verifier_accepted=bool(accepted and not rejected),
                        semantic_rejected=rejected,
                        support=1,
                        evidence=row,
                    )
                )

    return samples


def build_variant_questions(
    fresh_samples: Sequence[Mapping[str, Any]],
    exp1344_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """REQ-LEARN-1358-4: build fresh variants first, otherwise replay variants."""

    if fresh_samples:
        return [_fresh_variant(sample) for sample in fresh_samples]

    policy = exp1344_artifact.get("failure_type_policy", {})
    if not isinstance(policy, Mapping):
        return []

    variants: list[dict[str, Any]] = []
    for failure_type in sorted(policy):
        entry = policy[failure_type]
        if not isinstance(entry, Mapping):
            continue
        action = str(entry.get("action") or entry.get("policy") or "")
        support = max(_int(entry.get("failure_count")), 1)
        variants.append(
            {
                "variant_id": f"replay:{failure_type}",
                "source": "exp1344_replay",
                "case_id": failure_type,
                "question": _replay_question(failure_type, entry),
                "verifier_accepted": action == POLICY_PROMOTE,
                "semantic_rejected": action in {POLICY_DEMOTE, POLICY_QUARANTINE},
                "memory_action": _memory_action_for_replay_policy(action),
                "support": support,
            }
        )
    return variants


def apply_memory_updates(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """REQ-LEARN-1358-5: promote accepted variants and reject semantic failures.

    Counts are support-weighted because a replay variant can summarize many
    source cases from the fallback policy, while a fresh verifier row has
    support one. Semantic rejects are counted as demoted control pressure even
    when their exact update action is quarantine.
    """

    promoted: list[str] = []
    demoted: list[str] = []
    quarantined: list[str] = []
    held: list[str] = []
    promoted_count = 0
    demoted_count = 0
    quarantined_count = 0

    for variant in variants:
        variant_id = str(variant.get("variant_id") or variant.get("case_id") or "unknown")
        support = max(_int(variant.get("support")), 1)
        accepted = bool(variant.get("verifier_accepted"))
        semantic_rejected = bool(variant.get("semantic_rejected"))
        action = str(variant.get("memory_action") or "")

        if accepted and not semantic_rejected:
            promoted.append(variant_id)
            promoted_count += support
        elif semantic_rejected:
            demoted.append(variant_id)
            demoted_count += support
            if action == POLICY_QUARANTINE:
                quarantined.append(variant_id)
                quarantined_count += support
        else:
            held.append(variant_id)

    return {
        "promoted": promoted,
        "demoted": demoted,
        "quarantined": quarantined,
        "held_for_fresh_verifier": held,
        "promoted_memory_count": promoted_count,
        "demoted_memory_count": demoted_count,
        "quarantined_memory_count": quarantined_count,
    }


def derive_dvi_ready(
    *,
    self_learning_delta_overall: float,
    accepted_violation_delta: float,
    nonforgetting_certificate_rate: float,
    memory_regression_count: int,
) -> bool:
    """REQ-LEARN-1358-7: gate DVI on positive learning and clean controls."""

    return (
        self_learning_delta_overall > 0.0
        and accepted_violation_delta <= 0.0
        and nonforgetting_certificate_rate == 1.0
        and memory_regression_count == 0
    )


def derive_honest_verdict(
    *,
    dvi_ready: bool,
    update_is_replay_only: bool,
    headline_result_allowed: bool,
) -> str:
    """REQ-LEARN-1358-7: keep replay fallback out of headline claims."""

    if not dvi_ready:
        return "verifier_selected_memory_controls_blocked_non_headline"
    if headline_result_allowed:
        return "verifier_selected_memory_fresh_verified_dvi_ready_headline_eligible"
    if update_is_replay_only:
        return "verifier_selected_memory_replay_only_dvi_ready_non_headline"
    return "verifier_selected_memory_fresh_verified_dvi_ready_non_headline"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1358-6: assert the reconciliation-facing artifact schema."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise AssertionError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    rate = artifact["nonforgetting_certificate_rate"]
    if not isinstance(rate, (int, float)) or not 0.0 <= float(rate) <= 1.0:
        raise AssertionError("nonforgetting_certificate_rate must be between 0 and 1")
    for field in (
        "replay_cases_used",
        "fresh_verified_sample_count",
        "variant_question_count",
        "memory_regression_count",
        "promoted_memory_count",
        "demoted_memory_count",
    ):
        if not isinstance(artifact[field], int) or artifact[field] < 0:
            raise AssertionError(f"{field} must be a non-negative integer")
    if artifact["headline_result_allowed"] and artifact["fresh_verified_sample_count"] <= 0:
        raise AssertionError("headline_result_allowed requires fresh verified samples")


def _fresh_sample(
    *,
    source: str,
    case_id: str,
    verifier_accepted: bool,
    semantic_rejected: bool,
    support: int,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": f"fresh:{source}:{case_id}",
        "source": source,
        "case_id": case_id,
        "verifier_accepted": verifier_accepted,
        "semantic_rejected": semantic_rejected,
        "support": support,
        "evidence_summary": _evidence_summary(evidence),
    }


def _fresh_variant(sample: Mapping[str, Any]) -> dict[str, Any]:
    accepted = bool(sample.get("verifier_accepted"))
    semantic_rejected = bool(sample.get("semantic_rejected"))
    return {
        "variant_id": str(sample.get("sample_id") or sample.get("case_id") or "fresh:unknown"),
        "source": str(sample.get("source") or "fresh"),
        "case_id": str(sample.get("case_id") or "unknown"),
        "question": f"Fresh verifier-selected variant for {sample.get('case_id') or 'unknown'}",
        "verifier_accepted": accepted,
        "semantic_rejected": semantic_rejected,
        "memory_action": POLICY_PROMOTE
        if accepted and not semantic_rejected
        else POLICY_DEMOTE
        if semantic_rejected
        else POLICY_REQUEST_FRESH_VERIFIER,
        "support": max(_int(sample.get("support")), 1),
    }


def _exp1353_row_accepted(row: Mapping[str, Any]) -> bool:
    errors = row.get("errors", [])
    has_errors = (
        bool(errors) if isinstance(errors, Sequence) and not isinstance(errors, str) else False
    )
    return bool(row.get("parseable") is True and row.get("truthful") is True and not has_errors)


def _semantic_rejected(row: Mapping[str, Any]) -> bool:
    if _truthy(row.get("semantic_rejected"), row.get("semantic_reject"), row.get("rejected")):
        return True
    status_text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("status", "verdict", "validator_status", "honest_verdict")
    )
    return "reject" in status_text or "invalid" in status_text


def _rows(artifact: Mapping[str, Any], keys: Sequence[str]) -> list[Mapping[str, Any]]:
    for key in keys:
        rows = artifact.get(key, [])
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            return [row for row in rows if isinstance(row, Mapping)]
    return []


def _replay_question(failure_type: str, entry: Mapping[str, Any]) -> str:
    actions = entry.get("source_next_actions", [])
    if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)) and actions:
        return f"Replay {failure_type}: {actions[0]}"
    return f"Replay {failure_type}: apply failure-type memory policy"


def _memory_action_for_replay_policy(action: str) -> str:
    if action in {POLICY_PROMOTE, POLICY_DEMOTE, POLICY_QUARANTINE}:
        return action
    return POLICY_REQUEST_FRESH_VERIFIER


def _evidence_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "expected_state",
        "parseable",
        "truthful",
        "verifier_accepted",
        "semantic_valid",
        "semantic_rejected",
        "status",
        "verdict",
    )
    return {key: row[key] for key in keys if key in row}


def _default_source_artifacts() -> list[str]:
    return [
        f"results/{EXP1344_REQUESTED_FILE}",
        f"results/{EXP1344_FALLBACK_FILE}",
        f"results/{EXP1353_FILE}",
        f"results/{EXP1355_FILE}",
    ]


def _truthy(*values: Any) -> bool:
    return any(
        value is True or str(value).lower() in {"true", "yes", "passed", "accepted"}
        for value in values
    )


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return round(float(value), 6)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":  # pragma: no cover
    run()
