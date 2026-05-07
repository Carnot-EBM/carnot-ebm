"""Exp 1485 FR-11 completeness-reduction audit.

This audit reads Exp 1484's bounded query-time replay ledger and asks a narrow
question: can query-time verified memory reduce conservative false rejects
without introducing any false accepts? Soundness remains the hard gate. A
candidate that accepts even one negative-control replay row is rejected even if
it fixes every completeness mistake.

Spec: REQ-LEARN-1485, SCENARIO-LEARN-1486, SCENARIO-LEARN-1487.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

OUTPUT_FILE = "experiment_1485_fr11_completeness_reduction_audit.json"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_EXP1484_PATH = DEFAULT_RESULTS_DIR / "experiment_1484_fr11_v9_query_time_memory_policy.json"

EXPERIMENT = "1485_fr11_completeness_reduction_audit"
SCHEMA = "fr11_completeness_reduction_audit_v1"
RUN_DATE = "20260507"
SOURCE_EXPERIMENT = "experiment_1484_fr11_v9_query_time_memory_policy"

ALLOWED_VERDICT = "completeness_reduction_candidate_allowed_zero_soundness"
NO_ALLOWED_REDUCTION_VERDICT = "no_completeness_reduction_without_soundness_risk"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "source_experiment",
    "completeness_reduction_audit_complete",
    "baseline_completeness_mistakes",
    "candidate_completeness_mistakes",
    "completeness_mistake_delta",
    "baseline_soundness_mistakes",
    "candidate_soundness_mistakes",
    "candidate_policy",
    "policy_change_allowed",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class CandidateVariant:
    """One replay-routing candidate with its gated soundness/completeness score."""

    name: str
    source_variant: str
    routing: str
    threshold: str
    candidate_completeness_mistakes: int
    candidate_soundness_mistakes: int
    replay_case_count: int
    allowed: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_variant": self.source_variant,
            "routing": self.routing,
            "threshold": self.threshold,
            "candidate_completeness_mistakes": self.candidate_completeness_mistakes,
            "candidate_soundness_mistakes": self.candidate_soundness_mistakes,
            "replay_case_count": self.replay_case_count,
            "allowed": self.allowed,
            "rejection_reason": self.rejection_reason,
        }


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
    """REQ-LEARN-1485-1/4: write the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec": ["REQ-LEARN-1485", "SCENARIO-LEARN-1486", "SCENARIO-LEARN-1487"],
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "source_experiment": SOURCE_EXPERIMENT,
            "completeness_reduction_audit_complete": False,
            "baseline_completeness_mistakes": None,
            "candidate_completeness_mistakes": None,
            "completeness_mistake_delta": None,
            "baseline_soundness_mistakes": None,
            "candidate_soundness_mistakes": None,
            "candidate_policy": None,
            "policy_change_allowed": None,
            "tests_run": [],
            "honest_verdict": "in_progress",
        },
    )


def _source_experiment_name(exp1484_artifact: Mapping[str, Any]) -> str:
    raw = str(exp1484_artifact.get("experiment") or SOURCE_EXPERIMENT)
    return raw if raw.startswith("experiment_") else f"experiment_{raw}"


def _replay_mapping(exp1484_artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    replay = exp1484_artifact.get("memory_policy_replay")
    if not isinstance(replay, Mapping):
        raise AssertionError("memory_policy_replay is required")
    return replay


def _replay_eval(replay: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    replay_eval = replay.get(name)
    if not isinstance(replay_eval, Mapping):
        raise AssertionError(f"{name} replay ledger is required")
    return replay_eval


def _decisions(replay_eval: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    raw = replay_eval.get("decisions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise AssertionError(f"{name}.decisions must be a list")  # pragma: no cover
    if not all(isinstance(item, Mapping) for item in raw):
        raise AssertionError(f"{name}.decisions entries must be objects")  # pragma: no cover
    return list(raw)


def _allowed_gate(
    *,
    candidate_soundness_mistakes: int,
    baseline_soundness_mistakes: int,
) -> tuple[bool, str | None]:
    if candidate_soundness_mistakes > 0:
        return False, "candidate_soundness_mistakes_exceeds_zero"
    if candidate_soundness_mistakes > baseline_soundness_mistakes:
        return False, "candidate_soundness_mistakes_exceeds_baseline"
    return True, None


def _variant_from_replay_eval(
    *,
    name: str,
    source_variant: str,
    replay_eval: Mapping[str, Any],
    baseline_soundness_mistakes: int,
    routing: str,
    threshold: str,
) -> CandidateVariant:
    decisions = _decisions(replay_eval, source_variant)
    soundness = _to_int(replay_eval.get("soundness_mistakes"))
    allowed, reason = _allowed_gate(
        candidate_soundness_mistakes=soundness,
        baseline_soundness_mistakes=baseline_soundness_mistakes,
    )
    return CandidateVariant(
        name=name,
        source_variant=source_variant,
        routing=routing,
        threshold=threshold,
        candidate_completeness_mistakes=_to_int(replay_eval.get("completeness_mistakes")),
        candidate_soundness_mistakes=soundness,
        replay_case_count=len(decisions),
        allowed=allowed,
        rejection_reason=reason,
    )


def _unsafe_accept_all_variant(
    *,
    baseline_eval: Mapping[str, Any],
    baseline_soundness_mistakes: int,
) -> CandidateVariant:
    decisions = _decisions(baseline_eval, "baseline_memory_disabled")
    positive_ids = {
        str(item.get("case_id"))
        for item in decisions
        if bool(item.get("completeness_mistake"))
    }
    negative_control_count = sum(1 for item in decisions if str(item.get("case_id")) not in positive_ids)
    allowed, reason = _allowed_gate(
        candidate_soundness_mistakes=negative_control_count,
        baseline_soundness_mistakes=baseline_soundness_mistakes,
    )
    return CandidateVariant(
        name="unsafe_accept_all_replay_ids",
        source_variant="synthetic_negative_control_probe",
        routing="accept every replay ID as a query-time memory hit",
        threshold="none",
        candidate_completeness_mistakes=0,
        candidate_soundness_mistakes=negative_control_count,
        replay_case_count=len(decisions),
        allowed=allowed,
        rejection_reason=reason,
    )


def evaluate_candidate_variants(exp1484_artifact: Mapping[str, Any]) -> tuple[CandidateVariant, ...]:
    """REQ-LEARN-1485-2/3: score replay-routing candidates on the same cases."""

    replay = _replay_mapping(exp1484_artifact)
    baseline_eval = _replay_eval(replay, "baseline_memory_disabled")
    memory_eval = _replay_eval(replay, "memory_enabled")
    baseline_soundness = _to_int(baseline_eval.get("soundness_mistakes"))
    return (
        _variant_from_replay_eval(
            name="baseline_memory_disabled",
            source_variant="baseline_memory_disabled",
            replay_eval=baseline_eval,
            baseline_soundness_mistakes=baseline_soundness,
            routing="suppress all query-time memory hits",
            threshold="memory_enabled=false",
        ),
        _variant_from_replay_eval(
            name="exp1484_opt_in_verified_memory_enabled",
            source_variant="memory_enabled",
            replay_eval=memory_eval,
            baseline_soundness_mistakes=baseline_soundness,
            routing="allow hits only for Exp 1484 verified promoted memory IDs",
            threshold="verified_memory_index_membership",
        ),
        _unsafe_accept_all_variant(
            baseline_eval=baseline_eval,
            baseline_soundness_mistakes=baseline_soundness,
        ),
    )


def _best_allowed_variant(variants: Sequence[CandidateVariant]) -> CandidateVariant:
    allowed = [variant for variant in variants if variant.allowed]
    if not allowed:
        raise AssertionError("at least one zero-soundness candidate is required")  # pragma: no cover
    return min(allowed, key=lambda variant: variant.candidate_completeness_mistakes)


def build_artifact(
    *,
    exp1484_artifact: Mapping[str, Any],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1485: build the terminal completeness-reduction audit artifact."""

    variants = evaluate_candidate_variants(exp1484_artifact)
    baseline = variants[0]
    candidate = _best_allowed_variant(variants)
    delta = (
        candidate.candidate_completeness_mistakes
        - baseline.candidate_completeness_mistakes
    )
    policy_change_allowed = (
        candidate.candidate_soundness_mistakes == 0
        and candidate.candidate_soundness_mistakes <= baseline.candidate_soundness_mistakes
        and delta < 0
    )

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1485", "SCENARIO-LEARN-1486", "SCENARIO-LEARN-1487"],
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "source_experiment": _source_experiment_name(exp1484_artifact),
        "source_experiment_path": str(DEFAULT_EXP1484_PATH.relative_to(REPO_ROOT)),
        "completeness_reduction_audit_complete": True,
        "baseline_completeness_mistakes": baseline.candidate_completeness_mistakes,
        "candidate_completeness_mistakes": candidate.candidate_completeness_mistakes,
        "completeness_mistake_delta": delta,
        "baseline_soundness_mistakes": baseline.candidate_soundness_mistakes,
        "candidate_soundness_mistakes": candidate.candidate_soundness_mistakes,
        "candidate_policy": candidate.to_dict(),
        "policy_change_allowed": policy_change_allowed,
        "tests_run": list(commands_run or []),
        "honest_verdict": (
            ALLOWED_VERDICT if policy_change_allowed else NO_ALLOWED_REDUCTION_VERDICT
        ),
        "candidate_variants": [variant.to_dict() for variant in variants],
        "candidate_variants_audited": len(variants),
        "source_replay_case_count": baseline.replay_case_count,
        "audit_note": (
            "Policy change allowed means the selected routing candidate clears "
            "the zero-soundness gate and reduces bounded replay false rejects. "
            "This audit does not broaden memory beyond Exp 1484's opt-in "
            "verified-memory routing."
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1485-4/5: enforce required fields and safety gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] == "in_progress":
        return
    if artifact["status"] != "complete":
        raise AssertionError(f"unsupported status: {artifact['status']}")  # pragma: no cover
    if artifact["completeness_reduction_audit_complete"] is not True:
        raise AssertionError("complete artifact must mark audit complete")  # pragma: no cover

    baseline_completeness = int(artifact["baseline_completeness_mistakes"])
    candidate_completeness = int(artifact["candidate_completeness_mistakes"])
    expected_delta = candidate_completeness - baseline_completeness
    if int(artifact["completeness_mistake_delta"]) != expected_delta:
        raise AssertionError("completeness_mistake_delta must equal candidate minus baseline")

    baseline_soundness = int(artifact["baseline_soundness_mistakes"])
    candidate_soundness = int(artifact["candidate_soundness_mistakes"])
    if candidate_soundness > baseline_soundness or candidate_soundness > 0:
        raise AssertionError("candidate soundness must not exceed baseline or zero")

    expected_allowed = (
        candidate_soundness == 0
        and candidate_soundness <= baseline_soundness
        and expected_delta < 0
    )
    if bool(artifact["policy_change_allowed"]) != expected_allowed:
        raise AssertionError("policy_change_allowed must match soundness and delta gates")


def run(
    *,
    exp1484_path: Path | str = DEFAULT_EXP1484_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1485 and write the terminal JSON audit artifact."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    artifact = build_artifact(
        exp1484_artifact=load_json(exp1484_path),
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=commands_run,
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
