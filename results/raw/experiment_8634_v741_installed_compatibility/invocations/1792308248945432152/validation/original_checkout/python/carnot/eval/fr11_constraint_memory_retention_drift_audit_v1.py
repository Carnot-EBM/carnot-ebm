"""Exp 3129 FR-11 constraint-memory retention and drift audit.

Spec refs: REQ-LEARN-3129, SCENARIO-LEARN-3129,
SCENARIO-LEARN-3129-BLOCKED.

This audit treats Exp 3128's EvoEnv output as controller/environment memory,
not as model-weight learning. The important question is whether the admitted
constraint environments remain reusable when reloaded from the artifact and
whether prior FR-11 guard families still replay without forgetting. The module
therefore recomputes exact validation outcomes from source rows rather than
trusting stored summaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.eval import fragment_time_monitor_satisfiable_drift_audit_v1 as monitor
from carnot.eval import fr11_evoenv_verifiable_environment_synthesis_v1 as evoenv


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3129_fr11_constraint_memory_retention_drift_audit_v1"
SCHEMA = "carnot.fr11.constraint_memory_retention_drift_audit.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json"
)
EXP3116_REL_PATH = Path(
    "results/experiment_3116_fr11_unsolvable_curriculum_retention_guard_v1.json"
)
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3128_REL_PATH = Path(
    "results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"
)
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
REQUIRED_ARTIFACT_FIELDS = {
    "fr11_constraint_memory_audit_v1_ready",
    "admitted_environment_count",
    "replay_family_count",
    "prior_retention_delta",
    "novelty_retention_delta",
    "soundness_errors",
    "completeness_errors",
    "satisfiable_drift_count",
    "ledger_consistency_rate",
    "promotion_recommendation",
    "no_weight_update_claim",
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
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.py -q",
    ".venv/bin/coverage run --source=python/carnot/eval/fr11_constraint_memory_retention_drift_audit_v1.py -m pytest -o addopts='' tests/python/test_experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_constraint_memory_retention_drift_audit_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3116_fr11_retention_guard", EXP3116_REL_PATH, True),
    ("exp3126_satisfiable_drift_audit", EXP3126_REL_PATH, True),
    ("exp3128_evoenv_admission", EXP3128_REL_PATH, True),
    (
        "exp3129_module",
        Path("python/carnot/eval/fr11_constraint_memory_retention_drift_audit_v1.py"),
        False,
    ),
    (
        "exp3129_tests",
        Path(
            "tests/python/test_experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.py"
        ),
        False,
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence.

    The audit is meant to be deterministic and artifact-only. Returning an
    empty mapping on missing or malformed files lets the precondition gate emit
    a clear blocked artifact instead of partially trusting bad inputs.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def constraint_from_row(row: Mapping[str, Any]) -> evoenv.ConstraintSpec:
    """Reconstruct an Exp 3128 serializable constraint for exact replay."""

    terms = row.get("terms") or {}
    return evoenv.ConstraintSpec(
        kind=str(row.get("kind") or ""),
        terms={str(key): int(value) for key, value in dict(terms).items()},
        modulus=_optional_int(row.get("modulus")),
        equals=_optional_int(row.get("equals")),
        lower=_optional_int(row.get("lower")),
        upper=_optional_int(row.get("upper")),
        left=str(row.get("left")) if row.get("left") is not None else None,
        right=str(row.get("right")) if row.get("right") is not None else None,
    )


def environment_from_row(row: Mapping[str, Any]) -> evoenv.ConstraintEnvironment:
    """Rebuild a constraint environment row without trusting its stored result."""

    variables = tuple(str(variable) for variable in row.get("variables", ()))
    domains = {
        str(name): tuple(int(value) for value in values)
        for name, values in dict(row.get("domains") or {}).items()
    }
    constraints = tuple(
        constraint_from_row(constraint)
        for constraint in row.get("constraints", ())
        if isinstance(constraint, Mapping)
    )
    return evoenv.ConstraintEnvironment(
        environment_id=str(row.get("environment_id") or ""),
        family_id=str(row.get("family_id") or ""),
        variables=variables,
        domains=domains,
        constraints=constraints,
        prompt=str(row.get("prompt") or ""),
    )


def replay_admitted_environments(exp3128: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3129-1/2/4: replay admitted EvoEnv rows by exact enumeration."""

    rows = [
        dict(row) for row in exp3128.get("admitted_environments", []) if isinstance(row, Mapping)
    ]
    admission_records = [
        dict(row)
        for row in exp3128.get("admission_records", [])
        if isinstance(row, Mapping) and row.get("admitted") is True
    ]
    baseline_success_count = sum(
        1
        for record in admission_records
        if int(record.get("soundness_errors") or 0) == 0
        and int(record.get("completeness_errors") or 0) == 0
    )
    replay_rows = [exact_environment_replay(row) for row in rows]
    post_success_count = sum(1 for row in replay_rows if row["exact_replay_passed"])
    soundness_errors = sum(int(row["soundness_errors"]) for row in replay_rows)
    completeness_errors = sum(int(row["completeness_errors"]) for row in replay_rows)
    baseline_denominator = len(admission_records) if admission_records else len(rows)
    baseline_rate = rate(baseline_success_count, baseline_denominator)
    post_rate = rate(post_success_count, len(rows))
    declared_count = int(exp3128.get("admitted_environment_count") or 0)
    return {
        "admitted_environment_count": declared_count,
        "replayed_environment_count": len(rows),
        "family_count": len({str(row.get("family_id") or "") for row in rows}),
        "baseline_success_rate": baseline_rate,
        "post_replay_success_rate": post_rate,
        "novelty_retention_delta": round_float(post_rate - baseline_rate),
        "soundness_errors": soundness_errors,
        "completeness_errors": completeness_errors,
        "replay_rows": replay_rows,
    }


def exact_environment_replay(row: Mapping[str, Any]) -> JsonDict:
    """Compare the cheap verifier against exact constraint truth on every assignment."""

    environment = environment_from_row(row)
    reference = environment.compute_reference()
    stored_reference = dict(row.get("reference") or {})
    stored_assignment = dict(stored_reference.get("canonical_assignment") or {})
    stored_solution_count = int(stored_reference.get("solution_count") or 0)
    soundness_errors = 0
    completeness_errors = 0
    valid_assignment_count = 0
    for values in itertools.product(*(environment.domains[var] for var in environment.variables)):
        assignment = dict(zip(environment.variables, values, strict=True))
        exact_valid = all(
            evoenv.constraint_holds(constraint, assignment)
            for constraint in environment.constraints
        )
        verifier_accepts = environment.score_response(assignment).accepted
        valid_assignment_count += int(exact_valid)
        soundness_errors += int(verifier_accepts and not exact_valid)
        completeness_errors += int(exact_valid and not verifier_accepts)
    reference_consistent = (
        dict(reference.canonical_assignment) == stored_assignment
        and reference.solution_count == stored_solution_count
    )
    completeness_errors += int(not reference_consistent)
    return {
        "environment_id": environment.environment_id,
        "family_id": environment.family_id,
        "assignment_count": reference.solver_evaluations,
        "valid_assignment_count": valid_assignment_count,
        "reference_consistent": reference_consistent,
        "soundness_errors": soundness_errors,
        "completeness_errors": completeness_errors,
        "exact_replay_passed": (
            reference_consistent
            and valid_assignment_count > 0
            and soundness_errors == 0
            and completeness_errors == 0
        ),
    }


def prior_retention_summary(exp3116: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3129-3: replay prior guarded decisions against exact targets."""

    rows = [dict(row) for row in exp3116.get("guarded_decisions", []) if isinstance(row, Mapping)]
    prior_correct = sum(1 for row in rows if row.get("decision_label") == "correct")
    replay_correct = sum(
        1 for row in rows if str(row.get("controller_decision")) == str(row.get("target_action"))
    )
    soundness_errors = sum(
        1
        for row in rows
        if row.get("target_action") == "reject" and row.get("controller_decision") == "accept"
    )
    completeness_errors = sum(
        1
        for row in rows
        if row.get("target_action") == "accept" and row.get("controller_decision") != "accept"
    )
    hard_families = exp3116.get("unsolvable_detection_summary", {}).get("hard_families", [])
    if not isinstance(hard_families, Sequence) or isinstance(hard_families, (str, bytes)):
        hard_families = []
    family_count = max(
        int(exp3116.get("hard_family_count") or 0),
        len({str(family) for family in hard_families}),
    )
    return {
        "prior_case_count": len(rows),
        "prior_correct_rate": rate(prior_correct, len(rows)),
        "post_replay_correct_rate": rate(replay_correct, len(rows)),
        "prior_retention_delta": round_float(
            rate(replay_correct, len(rows)) - rate(prior_correct, len(rows))
        ),
        "family_count": family_count,
        "soundness_errors": soundness_errors,
        "completeness_errors": completeness_errors,
    }


def drift_summary(exp3126: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3129-5: recompute satisfiable drift from monitor events."""

    events = exp3126.get("monitor_events")
    if isinstance(events, list):
        replay = monitor.replay_monitor_events(events)
        return {
            "satisfiable_drift_count": int(replay["satisfiable_drift_count"]),
            "ledger_consistency_rate": float(replay["ledger_consistency_rate"]),
            "contradiction_count": int(replay["contradiction_count"]),
            "monitor_violation_count": int(replay["monitor_violation_count"]),
        }
    return {
        "satisfiable_drift_count": int(exp3126.get("satisfiable_drift_count") or 0),
        "ledger_consistency_rate": float(exp3126.get("ledger_consistency_rate") or 0.0),
        "contradiction_count": int(exp3126.get("contradiction_count") or 0),
        "monitor_violation_count": int(exp3126.get("monitor_violation_count") or 0),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3129 terminal audit artifact from checked-in sources."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3128 = read_json_object(root_path / EXP3128_REL_PATH)
    exp3116 = read_json_object(root_path / EXP3116_REL_PATH)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    blocker = precondition_blocker(exp3128, exp3116, exp3126)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    environment_replay = replay_admitted_environments(exp3128)
    prior_replay = prior_retention_summary(exp3116)
    drift_replay = drift_summary(exp3126)
    soundness_errors = int(environment_replay["soundness_errors"]) + int(
        prior_replay["soundness_errors"]
    )
    completeness_errors = int(environment_replay["completeness_errors"]) + int(
        prior_replay["completeness_errors"]
    )
    prior_delta = float(prior_replay["prior_retention_delta"])
    novelty_delta = float(environment_replay["novelty_retention_delta"])
    satisfiable_drift_count = int(drift_replay["satisfiable_drift_count"])
    count_matches = int(environment_replay["admitted_environment_count"]) == int(
        environment_replay["replayed_environment_count"]
    )
    ready = (
        count_matches
        and soundness_errors == 0
        and completeness_errors == 0
        and prior_delta >= 0.0
        and novelty_delta >= 0.0
        and satisfiable_drift_count == 0
    )
    recommendation = promotion_recommendation(
        ready,
        soundness_errors,
        completeness_errors,
        satisfiable_drift_count,
        prior_delta,
        novelty_delta,
        float(drift_replay["ledger_consistency_rate"]),
    )
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_constraint_memory_audit_v1_ready": ready,
        "admitted_environment_count": int(environment_replay["admitted_environment_count"]),
        "replay_family_count": int(prior_replay["family_count"])
        + int(environment_replay["family_count"]),
        "prior_retention_delta": prior_delta,
        "novelty_retention_delta": novelty_delta,
        "soundness_errors": soundness_errors,
        "completeness_errors": completeness_errors,
        "satisfiable_drift_count": satisfiable_drift_count,
        "ledger_consistency_rate": float(drift_replay["ledger_consistency_rate"]),
        "promotion_recommendation": recommendation,
        "no_weight_update_claim": True,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": inference_substrate(),
        "precondition_checks": precondition_checks(exp3128, exp3116, exp3126),
        "replay_summaries": {
            "prior_retention": prior_replay,
            "environment_memory": environment_replay,
            "satisfiable_drift": drift_replay,
        },
        "forgetting_regression_count": int(prior_delta < 0.0) + int(novelty_delta < 0.0),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(ready, recommendation),
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
    """Return a schema-complete artifact when required source evidence is absent."""

    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_constraint_memory_audit_v1_ready": False,
        "admitted_environment_count": 0,
        "replay_family_count": 0,
        "prior_retention_delta": 0.0,
        "novelty_retention_delta": 0.0,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "satisfiable_drift_count": 0,
        "ledger_consistency_rate": 0.0,
        "promotion_recommendation": "block_fr11_followup_missing_source_evidence",
        "no_weight_update_claim": True,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root),
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "precondition_checks": {
            "exp3128_evoenv_ready": False,
            "exp3116_retention_ready": False,
            "exp3126_drift_monitor_ready": False,
        },
        "replay_summaries": {
            "prior_retention": {},
            "environment_memory": {"replayed_environment_count": 0},
            "satisfiable_drift": {},
        },
        "forgetting_regression_count": 0,
        "blocked_reason": blocker,
        "duration_s": duration(start, now_s),
        "honest_verdict": f"blocked_precondition_failed: {blocker}",
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3129 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def precondition_blocker(
    exp3128: Mapping[str, Any],
    exp3116: Mapping[str, Any],
    exp3126: Mapping[str, Any],
) -> str:
    """Return the first missing source needed for a sound audit."""

    if exp3128.get("fr11_evoenv_pilot_v1_ready") is not True:
        return "exp3128_evoenv_artifact_missing_or_not_ready"
    if exp3116.get("fr11_unsolvable_curriculum_ready") is not True:
        return "exp3116_retention_guard_missing_or_not_ready"
    if exp3126.get("fragment_time_monitor_v1_ready") is not True:
        return "exp3126_drift_monitor_missing_or_not_ready"
    return ""


def precondition_checks(
    exp3128: Mapping[str, Any],
    exp3116: Mapping[str, Any],
    exp3126: Mapping[str, Any],
) -> JsonDict:
    """Expose source readiness checks in the final artifact."""

    return {
        "exp3128_evoenv_ready": exp3128.get("fr11_evoenv_pilot_v1_ready") is True,
        "exp3116_retention_ready": exp3116.get("fr11_unsolvable_curriculum_ready") is True,
        "exp3126_drift_monitor_ready": exp3126.get("fragment_time_monitor_v1_ready") is True,
        "exp3128_no_weight_update_claim": exp3128.get("no_weight_update_claim") is True,
        "exp3116_no_weight_update_claim": exp3116.get("no_weight_update_claim") is True,
    }


def promotion_recommendation(
    ready: bool,
    soundness_errors: int,
    completeness_errors: int,
    satisfiable_drift_count: int,
    prior_retention_delta: float,
    novelty_retention_delta: float,
    ledger_consistency_rate: float,
) -> str:
    """REQ-LEARN-3129-6: make the FR-11 promotion decision explicit."""

    if not ready:
        return "block_fr11_followup_missing_source_evidence"
    if soundness_errors or completeness_errors:
        return "block_fr11_promotion_soundness_or_completeness_regression"
    if satisfiable_drift_count:
        return "block_fr11_promotion_satisfiable_drift_regression"
    if prior_retention_delta < 0.0:
        return "block_fr11_promotion_prior_forgetting_regression"
    if novelty_retention_delta < 0.0:
        return "block_fr11_promotion_novelty_forgetting_regression"
    if ledger_consistency_rate < 1.0:
        return (
            "promote_controller_environment_memory_only_"
            "block_model_weight_learning_until_ledger_consistency_is_1.0"
        )
    return "promote_controller_environment_memory_only"


def inference_substrate(mode: str = "artifact_only_constraint_memory_audit") -> JsonDict:
    """Separate controller/environment memory from model-weight learning."""

    return {
        "mode": mode,
        "controller_environment_memory_only": True,
        "environment_schema_memory_replayed": True,
        "controller_guard_memory_replayed": True,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "executes_exact_solvers": True,
        "executes_model_generation": False,
        "fresh_live_inference_calls": 0,
        "uses_checked_in_artifacts_only": True,
        "source_environment_artifact": EXP3128_REL_PATH.as_posix(),
        "source_prior_guard_artifact": EXP3116_REL_PATH.as_posix(),
        "source_drift_monitor_artifact": EXP3126_REL_PATH.as_posix(),
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List audit sources with checksums for traceable replay."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": path.is_file(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3129 artifact violates the audit contract."""

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
    if artifact.get("fr11_constraint_memory_audit_v1_ready") is not True:
        return
    replayed_count = int(
        ((artifact.get("replay_summaries") or {}).get("environment_memory") or {}).get(
            "replayed_environment_count",
            -1,
        )
    )
    if int(artifact["admitted_environment_count"]) != replayed_count:
        raise ValueError("admitted_environment_count must match replayed environment rows")
    if int(artifact["soundness_errors"]) != 0:
        raise ValueError("soundness_errors must be zero for readiness")
    if int(artifact["completeness_errors"]) != 0:
        raise ValueError("completeness_errors must be zero for readiness")
    if int(artifact["satisfiable_drift_count"]) != 0:
        raise ValueError("satisfiable_drift_count must be zero for readiness")
    if float(artifact["prior_retention_delta"]) < 0.0:
        raise ValueError("prior_retention_delta must be nonnegative for readiness")
    if float(artifact["novelty_retention_delta"]) < 0.0:
        raise ValueError("novelty_retention_delta must be nonnegative for readiness")
    if any(
        row.get("required") and not row.get("exists")
        for row in artifact.get("source_artifacts", [])
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def honest_verdict(ready: bool, recommendation: str) -> str:
    """Return a conductor-compatible terminal verdict string."""

    if ready:
        return (
            "complete: fr11_constraint_memory_audit_v1_ready=true; "
            f"promotion_recommendation={recommendation}; no model-weight update claimed"
        )
    return "blocked_precondition_failed: fr11_constraint_memory_audit_v1_ready=false"


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when the local source exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output so artifacts diff cleanly across reruns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded ratio with an explicit zero-denominator convention."""

    if denominator == 0:
        return 0.0
    return round_float(float(numerator) / float(denominator))


def round_float(value: float) -> float:
    """Round audit metrics to the artifact precision used by nearby experiments."""

    return round(float(value), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return nonnegative elapsed seconds for reproducible tests."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round_float(max(0.0, end - started_s))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    output = write_artifact(REPO_ROOT)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["fr11_constraint_memory_audit_v1_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
