"""Build the Exp 1344 failure-type governed memory-policy audit artifact.

Spec: REQ-LEARN-1344, SCENARIO-LEARN-1344.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILE = "experiment_1344_continuous_self_learning_failure_type_memory_policy.json"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1344_continuous_self_learning_failure_type_memory_policy"
SCHEMA = "continuous_self_learning_failure_type_memory_policy_v1"
RUN_DATE = "20260505"

EXP1303_REQUESTED_FILE = "experiment_1303_online_memory_policy_v2.json"
EXP1303_FALLBACK_FILE = "experiment_1303_querybandits_ngc_online_memory_policy.json"
EXP1315_REQUESTED_FILE = "experiment_1315_online_memory_policy_nonforgetting_audit.json"
EXP1315_FALLBACK_FILE = "experiment_1315_continuous_self_learning_cerce_nonforgetting_audit.json"
EXP1324_FILE = "experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json"
EXP1341_FILE = "experiment_1341_halluguard_certificate_failure_split.json"

POLICY_PROMOTE = "promote"
POLICY_DEMOTE = "demote"
POLICY_QUARANTINE = "quarantine"
POLICY_REQUEST_FRESH_VERIFIER = "request_fresh_verifier"
POLICY_ACTIONS = (
    POLICY_PROMOTE,
    POLICY_DEMOTE,
    POLICY_QUARANTINE,
    POLICY_REQUEST_FRESH_VERIFIER,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "self_learning_delta_overall",
    "nonforgetting_certificate_rate",
    "memory_regression_count",
    "accepted_violation_delta",
    "failure_type_policy",
    "promoted_memory_count",
    "demoted_memory_count",
    "replay_cases_used",
    "headline_certificate_cases",
    "dvi_ready",
    "headline_result_allowed",
    "honest_verdict",
)

_REQUESTED_INPUTS = {
    "exp1303": (EXP1303_REQUESTED_FILE, EXP1303_FALLBACK_FILE),
    "exp1315": (EXP1315_REQUESTED_FILE, EXP1315_FALLBACK_FILE),
    "exp1324": (EXP1324_FILE,),
    "exp1341": (EXP1341_FILE,),
}

_POLICY_BY_FAILURE_TYPE = {
    "semantic_invalidity": POLICY_PROMOTE,
    "possible_hardcoded_solution_leakage": POLICY_DEMOTE,
    "unknown_state_mishandling": POLICY_QUARANTINE,
    "parser_schema_mismatch": POLICY_REQUEST_FRESH_VERIFIER,
    "undergeneration": POLICY_REQUEST_FRESH_VERIFIER,
    "solver_disagreement": POLICY_REQUEST_FRESH_VERIFIER,
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
    """REQ-LEARN-1344-1: persist a bootstrap marker before reading inputs."""

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
            "spec_resolution": _spec_resolution(project_root),
            "status": "in_progress",
            "self_learning_delta_overall": 0.0,
            "nonforgetting_certificate_rate": 0.0,
            "memory_regression_count": 0,
            "accepted_violation_delta": 0.0,
            "failure_type_policy": {},
            "promoted_memory_count": 0,
            "demoted_memory_count": 0,
            "replay_cases_used": 0,
            "headline_certificate_cases": [],
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
    """REQ-LEARN-1344-1/2: write in-progress, load inputs, and write audit."""

    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    payloads, unavailable_inputs, input_resolution, source_artifacts = load_inputs(results_dir)
    artifact = build_artifact(
        exp1303_artifact=payloads.get("exp1303", {}),
        exp1315_artifact=payloads.get("exp1315", {}),
        exp1324_artifact=payloads.get("exp1324", {}),
        exp1341_artifact=payloads.get("exp1341", {}),
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
    """Load requested artifacts, using documented fallbacks when aliases are absent."""

    results_path = Path(results_dir)
    payloads: dict[str, dict[str, Any]] = {}
    unavailable: list[str] = []
    resolution: dict[str, dict[str, str | None]] = {}
    sources: list[str] = []

    for key, candidates in _REQUESTED_INPUTS.items():
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
        resolution[key] = {
            "requested": f"results/{requested}",
            "used": used,
        }

    return payloads, unavailable, resolution, sources


def build_artifact(
    *,
    exp1303_artifact: Mapping[str, Any],
    exp1315_artifact: Mapping[str, Any],
    exp1324_artifact: Mapping[str, Any],
    exp1341_artifact: Mapping[str, Any],
    unavailable_inputs: Sequence[str] = (),
    input_resolution: Mapping[str, Mapping[str, str | None]] | None = None,
    source_artifacts: Sequence[str] | None = None,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1344-3/4: build replay metrics and failure-type routing."""

    failure_counts = failure_type_counts(exp1324_artifact, exp1341_artifact)
    failure_type_policy = build_failure_type_policy(failure_counts, exp1341_artifact)
    self_learning_delta = _float(
        exp1315_artifact.get("self_learning_delta_overall"),
        _float(exp1303_artifact.get("self_learning_delta_overall"), 0.0),
    )
    nonforgetting_rate = _float(exp1315_artifact.get("nonforgetting_certificate_rate"), 0.0)
    regression_count = _int(exp1315_artifact.get("memory_regression_count"))
    accepted_violation_delta = _float(
        exp1315_artifact.get("accepted_violation_delta"),
        _float(exp1303_artifact.get("accepted_violation_delta"), 0.0),
    )
    promoted_count = _int(exp1315_artifact.get("promoted_memory_count")) + _policy_count(
        failure_type_policy,
        POLICY_PROMOTE,
    )
    demoted_count = _int(exp1315_artifact.get("demoted_memory_count")) + _policy_count(
        failure_type_policy,
        POLICY_DEMOTE,
    )
    replay_cases_used = _int(exp1315_artifact.get("replay_case_count")) + _certificate_case_count(
        exp1324_artifact,
        failure_counts,
    )
    headline_cases = current_104_certificate_cases(exp1324_artifact, exp1341_artifact)
    dvi_ready = derive_dvi_ready(
        self_learning_delta_overall=self_learning_delta,
        accepted_violation_delta=accepted_violation_delta,
        nonforgetting_certificate_rate=nonforgetting_rate,
        memory_regression_count=regression_count,
    )
    headline_allowed = bool(dvi_ready and headline_cases)

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "source_artifacts": list(source_artifacts) if source_artifacts is not None else _source_artifacts(),
        "source_honest_verdicts": {
            "exp1303": exp1303_artifact.get("honest_verdict"),
            "exp1315": exp1315_artifact.get("honest_verdict"),
            "exp1324": exp1324_artifact.get("honest_verdict"),
            "exp1341": exp1341_artifact.get("honest_verdict"),
        },
        "input_resolution": dict(input_resolution or {}),
        "inputs_unavailable": list(unavailable_inputs),
        "spec_resolution": _spec_resolution(project_root),
        "status": "complete",
        "self_learning_delta_overall": self_learning_delta,
        "nonforgetting_certificate_rate": nonforgetting_rate,
        "memory_regression_count": regression_count,
        "accepted_violation_delta": accepted_violation_delta,
        "failure_type_policy": failure_type_policy,
        "promoted_memory_count": promoted_count,
        "demoted_memory_count": demoted_count,
        "replay_cases_used": replay_cases_used,
        "headline_certificate_cases": headline_cases,
        "dvi_ready": dvi_ready,
        "headline_result_allowed": headline_allowed,
        "honest_verdict": derive_honest_verdict(
            dvi_ready=dvi_ready,
            headline_result_allowed=headline_allowed,
        ),
        "measurement_note": (
            "Replay-only policy audit from already-written artifacts; no fresh "
            "certificate generation or headline Phase 1 gate rerun was performed."
        ),
    }


def failure_type_counts(
    exp1324_artifact: Mapping[str, Any],
    exp1341_artifact: Mapping[str, Any],
) -> dict[str, int]:
    """Collect failure-class counts from taxonomy records and split proxies."""

    counts: dict[str, int] = {}
    modes = exp1324_artifact.get("formalizer_failure_modes", [])
    if isinstance(modes, Sequence) and not isinstance(modes, (str, bytes)):
        for mode in modes:
            if not isinstance(mode, Mapping):
                continue
            failure_type = str(mode.get("class") or "")
            if failure_type:
                counts[failure_type] = _int(mode.get("count"))

    direct_fields = {
        "parser_schema_mismatch": exp1324_artifact.get("parser_failure_count")
        if exp1324_artifact.get("parser_failure_count") is not None
        else exp1341_artifact.get("parser_schema_risk_count"),
        "undergeneration": exp1324_artifact.get("undergeneration_failure_count")
        if exp1324_artifact.get("undergeneration_failure_count") is not None
        else exp1341_artifact.get("undergeneration_risk_count"),
        "semantic_invalidity": exp1324_artifact.get("semantic_failure_count")
        if exp1324_artifact.get("semantic_failure_count") is not None
        else exp1341_artifact.get("semantic_invalidity_count"),
        "unknown_state_mishandling": exp1324_artifact.get("unknown_state_mishandling_count")
        if exp1324_artifact.get("unknown_state_mishandling_count") is not None
        else exp1341_artifact.get("unknown_mishandling_count"),
        "possible_hardcoded_solution_leakage": exp1324_artifact.get(
            "possible_hardcoded_solution_leakage_count"
        ),
    }
    for failure_type, value in direct_fields.items():
        if value is not None:
            counts[failure_type] = max(counts.get(failure_type, 0), _int(value))

    source_cases = exp1341_artifact.get("source_cases_available", {})
    source_classes = source_cases.get("source_failure_classes") if isinstance(source_cases, Mapping) else []
    if isinstance(source_classes, Sequence) and not isinstance(source_classes, (str, bytes)):
        for failure_type in source_classes:
            counts.setdefault(str(failure_type), 0)

    return {key: value for key, value in counts.items() if key}


def build_failure_type_policy(
    failure_counts: Mapping[str, int],
    exp1341_artifact: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Map every observed certificate failure class to a verifier-governed policy."""

    repair_policy = exp1341_artifact.get("repair_policy_by_failure_type", {})
    policy: dict[str, dict[str, Any]] = {}
    for failure_type in sorted(failure_counts):
        action = _POLICY_BY_FAILURE_TYPE.get(failure_type, POLICY_QUARANTINE)
        entry = repair_policy.get(failure_type, {}) if isinstance(repair_policy, Mapping) else {}
        next_actions = entry.get("next_actions", []) if isinstance(entry, Mapping) else []
        policy[failure_type] = {
            "action": action,
            "policy": action,
            "failure_count": _int(failure_counts[failure_type]),
            "nonforgetting_check_required": action != POLICY_REQUEST_FRESH_VERIFIER,
            "certificate_tail_update_allowed": action == POLICY_PROMOTE,
            "source_next_actions": list(next_actions)
            if isinstance(next_actions, Sequence) and not isinstance(next_actions, (str, bytes))
            else [],
        }
    return policy


def derive_dvi_ready(
    *,
    self_learning_delta_overall: float,
    accepted_violation_delta: float,
    nonforgetting_certificate_rate: float,
    memory_regression_count: int,
) -> bool:
    """REQ-LEARN-1344-5: gate DVI on non-negative learning and no regressions."""

    return (
        self_learning_delta_overall >= 0.0
        and accepted_violation_delta <= 0.0
        and nonforgetting_certificate_rate == 1.0
        and memory_regression_count == 0
    )


def derive_honest_verdict(*, dvi_ready: bool, headline_result_allowed: bool) -> str:
    """REQ-LEARN-1344-6: keep replay-only results out of headline claims."""

    if not dvi_ready:
        return "failure_type_memory_policy_blocked_non_headline"
    if headline_result_allowed:
        return "failure_type_memory_policy_dvi_ready_headline_eligible"
    return "failure_type_memory_policy_dvi_ready_replay_non_headline"


def current_104_certificate_cases(
    exp1324_artifact: Mapping[str, Any],
    exp1341_artifact: Mapping[str, Any],
) -> list[Any]:
    """Return headline cases whose metadata identifies current `.104` evidence."""

    candidates: list[Any] = []
    for artifact in (exp1324_artifact, exp1341_artifact):
        cases = artifact.get("headline_certificate_cases", [])
        if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
            candidates.extend(cases)
    return [case for case in candidates if _is_current_104_case(case)]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1344-7: assert the fields that downstream reconciliation needs."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise AssertionError(f"missing required fields: {sorted(missing)}")
    rate = artifact["nonforgetting_certificate_rate"]
    if not isinstance(rate, (int, float)) or not 0.0 <= float(rate) <= 1.0:
        raise AssertionError("nonforgetting_certificate_rate must be between 0 and 1")
    if not isinstance(artifact["failure_type_policy"], Mapping):
        raise AssertionError("failure_type_policy must be a mapping")
    for failure_type, entry in artifact["failure_type_policy"].items():
        if not isinstance(entry, Mapping):
            raise AssertionError(f"policy for {failure_type} must be a mapping")
        action = entry.get("action", entry.get("policy"))
        if action not in POLICY_ACTIONS:
            raise AssertionError(f"unsupported policy for {failure_type}: {action}")
        if not isinstance(entry.get("nonforgetting_check_required"), bool):
            raise AssertionError(f"nonforgetting_check_required missing for {failure_type}")
        if not isinstance(entry.get("certificate_tail_update_allowed"), bool):
            raise AssertionError(f"certificate_tail_update_allowed missing for {failure_type}")


def _source_artifacts() -> list[str]:
    return [
        f"results/{EXP1303_REQUESTED_FILE}",
        f"results/{EXP1303_FALLBACK_FILE}",
        f"results/{EXP1315_REQUESTED_FILE}",
        f"results/{EXP1315_FALLBACK_FILE}",
        f"results/{EXP1324_FILE}",
        f"results/{EXP1341_FILE}",
    ]


def _certificate_case_count(
    exp1324_artifact: Mapping[str, Any],
    failure_counts: Mapping[str, int],
) -> int:
    metrics = exp1324_artifact.get("source_metrics", {})
    if isinstance(metrics, Mapping):
        attempt_count = _int(metrics.get("exp1312_certificate_attempt_count"))
        if attempt_count:
            return attempt_count
    attempt_count = _int(exp1324_artifact.get("certificate_attempt_count"))
    return attempt_count if attempt_count else sum(_int(count) for count in failure_counts.values())


def _policy_count(failure_type_policy: Mapping[str, Mapping[str, Any]], action: str) -> int:
    return sum(
        _int(entry.get("failure_count"))
        for entry in failure_type_policy.values()
        if entry.get("action", entry.get("policy")) == action
    )


def _is_current_104_case(case: Any) -> bool:
    if isinstance(case, Mapping):
        milestone = str(case.get("source_milestone") or case.get("milestone") or "")
        case_id = str(case.get("case_id") or case.get("id") or "")
        return milestone == ".104" or "104" in case_id
    return "104" in str(case)


def _spec_resolution(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    requested = root / "openspec" / "capabilities" / "online-learning" / "spec.md"
    fallback = root / "openspec" / "capabilities" / "self-learning" / "spec.md"
    return {
        "requested_online_learning_spec": str(requested),
        "requested_online_learning_spec_available": requested.exists(),
        "used_self_learning_spec": str(fallback),
        "used_self_learning_spec_available": fallback.exists(),
    }


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
