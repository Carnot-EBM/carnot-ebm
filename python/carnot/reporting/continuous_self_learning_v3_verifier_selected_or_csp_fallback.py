"""Build the Exp 1374 continuous self-learning v3 artifact.

Spec: REQ-LEARN-1374, SCENARIO-LEARN-1374.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILE = "experiment_1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback.json"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback"
SCHEMA = "continuous_self_learning_v3_verifier_selected_or_csp_fallback_v1"
RUN_DATE = "20260505"

EXP1358_FILE = "experiment_1358_continuous_self_learning_verifier_selected_memory.json"
EXP1365_FILE = "experiment_1365_eidoku_csp_neuro_symbolic_verification_probe.json"
EXP1369_FILE = "experiment_1369_semantic_validator_v2_nsvif_z3_constraints.json"

PATH_PRIMARY_SEMANTIC = "primary_semantic_verified"
PATH_FALLBACK_CSP = "fallback_csp_selected"
PATH_FALLBACK_REPLAY = "fallback_replay"

POLICY_PROMOTE = "promote"
POLICY_DEMOTE = "demote"
POLICY_QUARANTINE = "quarantine"
POLICY_HOLD = "hold"

CSP_FEASIBILITY_THRESHOLD = 0.70

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "path_used",
    "replay_cases_used",
    "fresh_verified_sample_count",
    "csp_selected_sample_count",
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
    "exp1358": EXP1358_FILE,
    "exp1365": EXP1365_FILE,
    "exp1369": EXP1369_FILE,
}


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1374-1: make the run auditable before source loading starts.

    Autonomous conductor runs can be interrupted after a task is dequeued but
    before all source artifacts are read. Persisting a schema-shaped
    ``in_progress`` artifact first prevents that interruption from looking like
    a missing experiment.
    """

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifacts": [],
            "inputs_unavailable": [],
            "status": "in_progress",
            "path_used": None,
            "replay_cases_used": 0,
            "fresh_verified_sample_count": 0,
            "csp_selected_sample_count": 0,
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
    """REQ-LEARN-1374-1/5: write bootstrap, select a verifier path, and finalize."""

    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    payloads, unavailable_inputs, source_artifacts = load_inputs(results_dir)
    artifact = build_artifact(
        exp1358_artifact=payloads.get("exp1358", {}),
        exp1365_artifact=payloads.get("exp1365", {}),
        exp1369_artifact=payloads.get("exp1369", {}),
        unavailable_inputs=unavailable_inputs,
        source_artifacts=source_artifacts,
        project_root=project_root,
        run_date=run_date,
    )
    validate_artifact(artifact)
    return _write_json(out_path, artifact)


def load_inputs(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Load source artifacts while keeping missing gates visible in the output."""

    results_path = Path(results_dir)
    payloads: dict[str, dict[str, Any]] = {}
    unavailable: list[str] = []
    sources: list[str] = []
    for key, filename in _SOURCE_FILES.items():
        path = results_path / filename
        relative = f"results/{filename}"
        if path.exists():
            payloads[key] = json.loads(path.read_text(encoding="utf-8"))
            sources.append(relative)
        else:
            unavailable.append(relative)
    return payloads, unavailable, sources


def build_artifact(
    *,
    exp1358_artifact: Mapping[str, Any],
    exp1365_artifact: Mapping[str, Any],
    exp1369_artifact: Mapping[str, Any],
    unavailable_inputs: Sequence[str] = (),
    source_artifacts: Sequence[str] | None = None,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1374-2/3/4/5: select the highest-qualified FR-11 path.

    Exp 1374 is intentionally conservative about the headline claim boundary:
    Exp 1369 semantic rows are treated as verifier-accepted fresh cases, Exp
    1365 CSP rows are only fallback promotion signal, and Exp 1358 remains the
    replay control source for self-learning and non-forgetting metrics.
    """

    path_used = select_path(exp1369_artifact=exp1369_artifact, exp1365_artifact=exp1365_artifact)
    if path_used == PATH_PRIMARY_SEMANTIC:
        variants = build_primary_semantic_variants(exp1369_artifact)
    elif path_used == PATH_FALLBACK_CSP:
        variants = build_csp_selected_variants(exp1365_artifact)
    else:
        variants = build_replay_variants(exp1358_artifact)

    memory_updates = apply_memory_updates(variants)
    fresh_verified_count = (
        sum(1 for variant in variants if _variant_is_primary_accept(variant))
        if path_used == PATH_PRIMARY_SEMANTIC
        else 0
    )
    csp_selected_count = (
        sum(
            max(_int(variant.get("support")), 1)
            for variant in variants
            if variant.get("csp_selected")
        )
        if path_used == PATH_FALLBACK_CSP
        else 0
    )

    self_learning_delta = _float(exp1358_artifact.get("self_learning_delta_overall"), 0.0)
    nonforgetting_rate = _float(exp1358_artifact.get("nonforgetting_certificate_rate"), 0.0)
    regression_count = _int(exp1358_artifact.get("memory_regression_count"))
    accepted_violation_delta = _float(exp1358_artifact.get("accepted_violation_delta"), 0.0)
    dvi_ready = derive_dvi_ready(
        self_learning_delta_overall=self_learning_delta,
        nonforgetting_certificate_rate=nonforgetting_rate,
        memory_regression_count=regression_count,
        accepted_violation_delta=accepted_violation_delta,
    )
    headline_allowed = derive_headline_result_allowed(
        path_used=path_used,
        fresh_verified_sample_count=fresh_verified_count,
        dvi_ready=dvi_ready,
    )

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "source_artifacts": list(source_artifacts or _default_source_artifacts()),
        "source_honest_verdicts": {
            "exp1358": exp1358_artifact.get("honest_verdict"),
            "exp1365": exp1365_artifact.get("honest_verdict"),
            "exp1369": exp1369_artifact.get("honest_verdict"),
        },
        "inputs_unavailable": list(unavailable_inputs),
        "status": "complete",
        "path_used": path_used,
        "replay_cases_used": _int(exp1358_artifact.get("replay_cases_used")),
        "fresh_verified_sample_count": fresh_verified_count,
        "csp_selected_sample_count": csp_selected_count,
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
        "dvi_ready": dvi_ready,
        "headline_result_allowed": headline_allowed,
        "honest_verdict": derive_honest_verdict(
            path_used=path_used,
            dvi_ready=dvi_ready,
            headline_result_allowed=headline_allowed,
        ),
        "csp_feasibility_threshold": CSP_FEASIBILITY_THRESHOLD,
        "headline_claim_rule": (
            "Headline self-learning is allowed only for primary semantic verified "
            "fresh samples with passing non-forgetting controls."
        ),
        "measurement_note": (
            "Exp 1358 supplies the replay control metrics. Exp 1369 supplies "
            "fresh semantic-validator memory promotion when its claim gate is open. "
            "Exp 1365 CSP-selected updates remain non-headline until independently "
            "validated."
        ),
    }
    return artifact


def select_path(
    *,
    exp1369_artifact: Mapping[str, Any],
    exp1365_artifact: Mapping[str, Any],
) -> str:
    """REQ-LEARN-1374-2/3/4: choose primary, then CSP fallback, then replay."""

    if exp1369_artifact.get("semantic_validator_claim_allowed") is True:
        return PATH_PRIMARY_SEMANTIC
    if exp1365_artifact.get("eidoku_csp_viable") is True:
        return PATH_FALLBACK_CSP
    return PATH_FALLBACK_REPLAY


def build_primary_semantic_variants(
    exp1369_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """REQ-LEARN-1374-2: convert accepted semantic rows into memory variants."""

    variants: list[dict[str, Any]] = []
    for row in _rows(exp1369_artifact, ("semantic_validator_rows", "validator_rows", "cases")):
        case_id = str(row.get("case_id") or f"exp1369-{len(variants)}")
        accepted = _semantic_row_accepted(row)
        rejected = _semantic_row_rejected(row)
        variants.append(
            {
                "variant_id": f"semantic:exp1369:{case_id}",
                "source": "exp1369_semantic_validator",
                "case_id": case_id,
                "question": f"Semantic validator memory update for {case_id}",
                "verifier_accepted": accepted,
                "semantic_rejected": rejected,
                "memory_action": POLICY_PROMOTE
                if accepted and not rejected
                else POLICY_DEMOTE
                if rejected
                else POLICY_HOLD,
                "support": 1,
                "evidence_summary": _semantic_evidence_summary(row),
            }
        )
    return variants


def build_csp_selected_variants(
    exp1365_artifact: Mapping[str, Any],
    *,
    threshold: float = CSP_FEASIBILITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """REQ-LEARN-1374-3: convert CSP feasibility evidence into fallback variants."""

    row_variants = _csp_row_variants(exp1365_artifact, threshold=threshold)
    if row_variants:
        return row_variants

    feasibility_rate = _float(exp1365_artifact.get("csp_feasibility_rate"), 0.0)
    corpus_count = _int(exp1365_artifact.get("corpus_cases_used"))
    selected_count = (
        int(round(corpus_count * feasibility_rate)) if feasibility_rate >= threshold else 0
    )
    if selected_count <= 0:
        return []
    return [
        {
            "variant_id": "csp:exp1365:aggregate_feasible_cases",
            "source": "exp1365_eidoku_csp",
            "case_id": "aggregate_feasible_cases",
            "question": "CSP-selected aggregate FoVer feasibility memory update",
            "verifier_accepted": True,
            "semantic_rejected": False,
            "memory_action": POLICY_PROMOTE,
            "support": selected_count,
            "csp_selected": True,
            "csp_feasibility_score": feasibility_rate,
            "selection_threshold": threshold,
        }
    ]


def build_replay_variants(exp1358_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-LEARN-1374-4: preserve Exp 1358 replay variants when no fresh gate opens."""

    variants = _rows(exp1358_artifact, ("variant_questions",))
    if variants:
        return [_normalize_replay_variant(variant) for variant in variants]

    replay_variants: list[dict[str, Any]] = []
    promoted_count = _int(exp1358_artifact.get("promoted_memory_count"))
    demoted_count = _int(exp1358_artifact.get("demoted_memory_count"))
    if promoted_count:
        replay_variants.append(_aggregate_replay_variant("promoted", True, False, promoted_count))
    if demoted_count:
        replay_variants.append(_aggregate_replay_variant("demoted", False, True, demoted_count))
    return replay_variants


def apply_memory_updates(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """REQ-LEARN-1374-2/3/4: support-weighted promotion and demotion accounting."""

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
        "held": held,
        "promoted_memory_count": promoted_count,
        "demoted_memory_count": demoted_count,
        "quarantined_memory_count": quarantined_count,
    }


def derive_dvi_ready(
    *,
    self_learning_delta_overall: float,
    nonforgetting_certificate_rate: float,
    memory_regression_count: int,
    accepted_violation_delta: float,
) -> bool:
    """REQ-LEARN-1374-6: require positive learning and clean control metrics."""

    return (
        self_learning_delta_overall > 0.0
        and nonforgetting_certificate_rate == 1.0
        and memory_regression_count == 0
        and accepted_violation_delta <= 0.0
    )


def derive_headline_result_allowed(
    *,
    path_used: str,
    fresh_verified_sample_count: int,
    dvi_ready: bool,
) -> bool:
    """REQ-LEARN-1374-7: CSP and replay paths are never headline evidence."""

    return dvi_ready and path_used == PATH_PRIMARY_SEMANTIC and fresh_verified_sample_count > 0


def derive_honest_verdict(
    *,
    path_used: str,
    dvi_ready: bool,
    headline_result_allowed: bool,
) -> str:
    """REQ-LEARN-1374-5/7: name both the selected path and claim boundary."""

    if not dvi_ready:
        return "continuous_self_learning_v3_controls_blocked_non_headline"
    if headline_result_allowed:
        return "continuous_self_learning_v3_primary_semantic_verified_dvi_ready_headline_allowed"
    if path_used == PATH_PRIMARY_SEMANTIC:
        return "continuous_self_learning_v3_primary_semantic_verified_dvi_ready_non_headline"
    if path_used == PATH_FALLBACK_CSP:
        return "continuous_self_learning_v3_csp_selected_dvi_ready_non_headline"
    return "continuous_self_learning_v3_replay_only_dvi_ready_non_headline"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1374-5/7: assert required fields and headline invariants."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise AssertionError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["path_used"] not in {
        None,
        PATH_PRIMARY_SEMANTIC,
        PATH_FALLBACK_CSP,
        PATH_FALLBACK_REPLAY,
    }:
        raise AssertionError(f"unsupported path_used: {artifact['path_used']}")
    rate = artifact["nonforgetting_certificate_rate"]
    if not isinstance(rate, (int, float)) or not 0.0 <= float(rate) <= 1.0:
        raise AssertionError("nonforgetting_certificate_rate must be between 0 and 1")
    for field in (
        "replay_cases_used",
        "fresh_verified_sample_count",
        "csp_selected_sample_count",
        "variant_question_count",
        "memory_regression_count",
        "promoted_memory_count",
        "demoted_memory_count",
    ):
        if not isinstance(artifact[field], int) or artifact[field] < 0:
            raise AssertionError(f"{field} must be a non-negative integer")
    if artifact["headline_result_allowed"]:
        if artifact["path_used"] != PATH_PRIMARY_SEMANTIC:
            raise AssertionError("headline_result_allowed requires primary semantic path")
        if artifact["fresh_verified_sample_count"] <= 0:
            raise AssertionError("headline_result_allowed requires fresh verified samples")
        if not artifact["dvi_ready"]:
            raise AssertionError("headline_result_allowed requires dvi_ready")


def _semantic_row_accepted(row: Mapping[str, Any]) -> bool:
    if _semantic_row_rejected(row):
        return False
    if row.get("constraint_passed") is not True:
        return False
    expected = row.get("expected_state")
    result = row.get("semantic_result")
    if expected is not None and result is not None:
        return str(expected) == str(result)
    return True


def _semantic_row_rejected(row: Mapping[str, Any]) -> bool:
    if _truthy(row.get("semantic_rejected"), row.get("semantic_reject"), row.get("rejected")):
        return True
    if row.get("constraint_passed") is False:
        return True
    expected = row.get("expected_state")
    result = row.get("semantic_result")
    if expected is not None and result is not None and str(expected) != str(result):
        return True
    status_text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("status", "verdict", "validator_status", "honest_verdict")
    )
    return "reject" in status_text or "invalid" in status_text


def _variant_is_primary_accept(variant: Mapping[str, Any]) -> bool:
    return (
        variant.get("source") == "exp1369_semantic_validator"
        and variant.get("verifier_accepted") is True
        and not variant.get("semantic_rejected")
    )


def _csp_row_variants(
    exp1365_artifact: Mapping[str, Any],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for row in _rows(exp1365_artifact, ("scored_cases", "case_scores", "scores", "cases")):
        score = _csp_feasibility_score(row)
        if score < threshold:
            continue
        case_id = str(row.get("case_id") or row.get("id") or f"exp1365-{len(variants)}")
        variants.append(
            {
                "variant_id": f"csp:exp1365:{case_id}",
                "source": "exp1365_eidoku_csp",
                "case_id": case_id,
                "question": f"CSP-selected memory update for {case_id}",
                "verifier_accepted": True,
                "semantic_rejected": False,
                "memory_action": POLICY_PROMOTE,
                "support": 1,
                "csp_selected": True,
                "csp_feasibility_score": score,
                "selection_threshold": threshold,
            }
        )
    return variants


def _csp_feasibility_score(row: Mapping[str, Any]) -> float:
    if row.get("csp_feasible") is True:
        return 1.0
    if row.get("csp_feasible") is False:
        return 0.0
    for key in ("csp_feasibility_score", "feasibility_score", "csp_feasibility"):
        if key in row:
            return _float(row.get(key), 0.0)
    if "csp_violation_score" in row:
        return max(0.0, min(1.0, 1.0 - _float(row.get("csp_violation_score"), 1.0)))
    return 0.0


def _normalize_replay_variant(variant: Mapping[str, Any]) -> dict[str, Any]:
    accepted = bool(variant.get("verifier_accepted"))
    rejected = bool(variant.get("semantic_rejected"))
    action = str(variant.get("memory_action") or "")
    if not action:
        action = (
            POLICY_PROMOTE
            if accepted and not rejected
            else POLICY_DEMOTE
            if rejected
            else POLICY_HOLD
        )
    return {
        "variant_id": str(variant.get("variant_id") or variant.get("case_id") or "replay:unknown"),
        "source": str(variant.get("source") or "exp1358_replay"),
        "case_id": str(variant.get("case_id") or "unknown"),
        "question": str(variant.get("question") or "Exp 1358 replay memory update"),
        "verifier_accepted": accepted,
        "semantic_rejected": rejected,
        "memory_action": action,
        "support": max(_int(variant.get("support")), 1),
    }


def _aggregate_replay_variant(
    label: str,
    verifier_accepted: bool,
    semantic_rejected: bool,
    support: int,
) -> dict[str, Any]:
    return {
        "variant_id": f"replay:exp1358:{label}",
        "source": "exp1358_replay",
        "case_id": label,
        "question": f"Exp 1358 aggregate {label} memory update",
        "verifier_accepted": verifier_accepted,
        "semantic_rejected": semantic_rejected,
        "memory_action": POLICY_PROMOTE if verifier_accepted else POLICY_DEMOTE,
        "support": support,
    }


def _semantic_evidence_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "certificate_state",
        "expected_state",
        "semantic_result",
        "claim_route",
        "constraint_evaluated",
        "constraint_passed",
    )
    return {key: row[key] for key in keys if key in row}


def _rows(artifact: Mapping[str, Any], keys: Sequence[str]) -> list[Mapping[str, Any]]:
    for key in keys:
        rows = artifact.get(key, [])
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            return [row for row in rows if isinstance(row, Mapping)]
    return []


def _default_source_artifacts() -> list[str]:
    return [f"results/{filename}" for filename in _SOURCE_FILES.values()]


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
