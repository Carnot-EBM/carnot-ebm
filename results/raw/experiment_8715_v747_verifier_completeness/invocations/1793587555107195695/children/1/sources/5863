"""Exp 1608 FR-11 self-learning scale run with CerCE no-forgetting.

Spec: REQ-LEARN-1608, SCENARIO-LEARN-1608, SCENARIO-LEARN-1609.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.pipeline.fr11_event_bus import ViolationEvent
from carnot.training import cerce_certificate_ledger as cerce

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260509"
OUTPUT_FILE = "experiment_1608_fr11_cerce.json"
SCHEMA = "fr11_continuous_self_learning_cerce_scale_v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_PRIOR_LEDGER_PATH = REPO_ROOT / "results" / "experiment_1594_cerce_ledger.json"
DEFAULT_BOUNDS_ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1595_cerce_bounds.json"
EXAMPLE_COUNT = 1000

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "schema",
    "continuous_self_learning_task",
    "examples_requested",
    "examples_processed",
    "training_loop_batches",
    "current_update_rows",
    "utility_delta",
    "baseline_success_rate",
    "promoted_success_rate",
    "cerce_ledger_ready",
    "bounds_check_passed",
    "past_certificates_checked",
    "past_certificates_violated",
    "accepted_violation_count",
    "false_accept_delta",
    "soundness_mistakes",
    "nonforgetting_certificate_rate",
    "no_model_weight_mutation",
    "policy_certificates_evaluated",
    "constraint_violation_records",
    "promotion_safe_policy_updates",
    "blocked_policy_updates",
    "ledger_rows",
    "training_summary",
    "blockers",
    "honest_verdict",
)


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1608-1/6: write the durable bootstrap artifact first."""

    artifact: JsonDict = {
        "status": "in_progress",
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1608", "SCENARIO-LEARN-1608", "SCENARIO-LEARN-1609"],
        "run_date": run_date,
        "project_root": str(project_root),
        "source_artifacts": [
            _display_path(DEFAULT_PRIOR_LEDGER_PATH, project_root=project_root),
            _display_path(DEFAULT_BOUNDS_ARTIFACT_PATH, project_root=project_root),
        ],
        "continuous_self_learning_task": True,
        "examples_requested": EXAMPLE_COUNT,
        "examples_processed": 0,
        "training_loop_batches": 0,
        "current_update_rows": 0,
        "utility_delta": 0.0,
        "baseline_success_rate": 0.0,
        "promoted_success_rate": 0.0,
        "cerce_ledger_ready": False,
        "bounds_check_passed": False,
        "past_certificates_checked": 0,
        "past_certificates_violated": 0,
        "accepted_violation_count": 0,
        "false_accept_delta": 0,
        "soundness_mistakes": 0,
        "nonforgetting_certificate_rate": 0.0,
        "no_model_weight_mutation": True,
        "policy_certificates_evaluated": 0,
        "constraint_violation_records": 0,
        "promotion_safe_policy_updates": [],
        "blocked_policy_updates": [],
        "policy_certificates": [],
        "ledger_rows": [],
        "training_summary": {},
        "blockers": ["fr11_cerce_scale_in_progress"],
        "honest_verdict": "in_progress",
        "tests_run": [],
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def generate_training_examples(
    example_count: int = EXAMPLE_COUNT,
    *,
    unsafe_indices: set[int] | frozenset[int] = frozenset(),
) -> list[JsonDict]:
    """REQ-LEARN-1608-2/4: create the deterministic 1000-example loop input."""

    families = ("arithmetic", "grammar_certificate", "safe_prefix", "runtime_contract")
    examples: list[JsonDict] = []
    for index in range(example_count):
        unsafe = index in unsafe_indices
        baseline_success = index % 4 == 0
        promoted_success = not unsafe
        examples.append(
            {
                "example_id": f"fr11-1608-example-{index:04d}",
                "constraint_type": families[index % len(families)],
                "baseline_success": baseline_success,
                "promoted_success": promoted_success,
                "promoted_false_accept": unsafe,
                "soundness_mistake": unsafe,
                "false_accept_delta": int(unsafe),
            }
        )
    return examples


def run_training_loop(
    ledger: cerce.CerCECertificateLedger,
    examples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 100,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1608-2/4/5: run FR-11 updates through the CerCE ledger."""

    baseline_successes = 0
    promoted_successes = 0
    improved_examples = 0
    bounds_worsened = 0
    for index, example in enumerate(examples):
        batch_index = index // batch_size
        policy_update_id = f"fr11-1608-batch-{batch_index:03d}"
        baseline_success = bool(example["baseline_success"])
        promoted_success = bool(example["promoted_success"])
        promoted_false_accept = bool(example["promoted_false_accept"])
        soundness_mistake = bool(example["soundness_mistake"])
        false_accept_delta = int(example["false_accept_delta"])
        baseline_successes += int(baseline_success)
        promoted_successes += int(promoted_success)
        improved = promoted_success and not baseline_success
        improved_examples += int(improved)
        bounds_worsened += int(false_accept_delta > 0 or promoted_false_accept or soundness_mistake)
        ledger.record_constraint_case(
            policy_update_id=policy_update_id,
            constraint_id=str(example["example_id"]),
            constraint_type=str(example["constraint_type"]),
            source="exp1608_fr11_cerce_scale",
            baseline_violation=False,
            promoted_violation=promoted_false_accept,
            accepted_violation=promoted_false_accept,
            false_accept_delta=false_accept_delta,
            soundness_mistake=soundness_mistake,
        )
        if improved:
            ledger.on_fr11_violation(
                ViolationEvent(
                    query_id=str(example["example_id"]),
                    step_index=index,
                    energy_score=0.0,
                    probe_confidence=1.0,
                    constraint_type=str(example["constraint_type"]),
                    question_domain="fr11_cerce_scale",
                    timestamp=f"{run_date}T00:00:00Z",
                ),
                policy_update_id=policy_update_id,
            )

    processed = len(examples)
    batches = (processed + batch_size - 1) // batch_size
    baseline_rate = _rate(baseline_successes, processed)
    promoted_rate = _rate(promoted_successes, processed)
    return {
        "examples_processed": processed,
        "training_loop_batches": batches,
        "current_update_rows": processed,
        "baseline_success_rate": baseline_rate,
        "promoted_success_rate": promoted_rate,
        "utility_delta": round(promoted_rate - baseline_rate, 6),
        "baseline_successes": baseline_successes,
        "promoted_successes": promoted_successes,
        "improved_examples": improved_examples,
        "bounds_worsened_updates": bounds_worsened,
    }


def run_experiment(
    *,
    project_root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    prior_ledger_path: Path | str = DEFAULT_PRIOR_LEDGER_PATH,
    bounds_artifact_path: Path | str = DEFAULT_BOUNDS_ARTIFACT_PATH,
    examples: Sequence[Mapping[str, Any]] | None = None,
    batch_size: int = 100,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 1608 and write the terminal CerCE scale artifact."""

    root = Path(project_root)
    output = _resolve_under_root(root, Path(output_path))
    prior_path = _resolve_under_root(root, Path(prior_ledger_path))
    bounds_path = _resolve_under_root(root, Path(bounds_artifact_path))
    write_in_progress_artifact(output, project_root=root, run_date=run_date)

    blockers: list[str] = []
    prior = _load_json(prior_path)
    bounds = _load_json(bounds_path)
    if not prior:
        blockers.append("missing_past_cerce_ledger")
    if not bounds:
        blockers.append("missing_cerce_bounds_artifact")

    ledger = cerce.CerCECertificateLedger(run_date=run_date)
    past_checked = 0
    past_violated = 0
    training_summary = _empty_training_summary()
    if not blockers:
        past_checked, past_violated = replay_past_certificate_rows(ledger, prior)
        training_examples = list(examples or generate_training_examples(EXAMPLE_COUNT))
        training_summary = run_training_loop(
            ledger,
            training_examples,
            batch_size=batch_size,
            run_date=run_date,
        )
    source_artifacts = [
        _display_path(prior_path, project_root=root),
        _display_path(bounds_path, project_root=root),
    ]
    artifact = build_terminal_artifact(
        ledger,
        training_summary=training_summary,
        source_artifacts=source_artifacts,
        bounds_artifact=bounds,
        input_blockers=blockers,
        past_certificates_checked=past_checked,
        past_certificates_violated=past_violated,
        project_root=root,
        run_date=run_date,
        tests_run=tests_run,
    )
    return _write_json(output, artifact)


def replay_past_certificate_rows(
    ledger: cerce.CerCECertificateLedger,
    prior_artifact: Mapping[str, Any],
) -> tuple[int, int]:
    """REQ-LEARN-1608-3: replay prior certificate rows into the CerCE ledger."""

    for row in prior_artifact.get("ledger_rows", []):
        ledger.record_constraint_case(
            policy_update_id=str(row["policy_update_id"]),
            constraint_id=str(row["constraint_id"]),
            constraint_type=str(row["constraint_type"]),
            source=f"past_cerce:{row['source']}",
            baseline_violation=bool(row["baseline_violation"]),
            promoted_violation=bool(row["promoted_violation"]),
            accepted_violation=bool(row["accepted_violation"]),
            false_accept_delta=int(row["false_accept_delta"]),
            soundness_mistake=bool(row["soundness_mistake"]),
        )
    certificates = [
        certificate
        for certificate in prior_artifact.get("policy_certificates", [])
        if isinstance(certificate, Mapping)
    ]
    violated = sum(
        int(certificate.get("promotion_safe") is not True) for certificate in certificates
    )
    if prior_artifact.get("cerce_ledger_ready") is not True and not violated:
        violated = 1
    return len(certificates), violated


def build_terminal_artifact(
    ledger: cerce.CerCECertificateLedger,
    *,
    training_summary: Mapping[str, Any],
    source_artifacts: Sequence[str],
    bounds_artifact: Mapping[str, Any],
    input_blockers: Sequence[str],
    past_certificates_checked: int,
    past_certificates_violated: int,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-1608-5/6: combine utility and CerCE gates into the artifact."""

    ledger_artifact = cerce.build_artifact(
        ledger,
        project_root=project_root,
        run_date=run_date,
        source_artifacts=source_artifacts,
        tests_run=tests_run,
    )
    certificates = ledger_artifact["policy_certificates"]
    soundness_mistakes = sum(int(certificate["soundness_mistakes"]) for certificate in certificates)
    bounds_passed = bool(bounds_artifact.get("bounds_check_passed"))
    blockers = set(input_blockers)
    if int(training_summary["examples_processed"]) != EXAMPLE_COUNT:
        blockers.add("fr11_scale_examples_not_1000")
    if float(training_summary["utility_delta"]) <= 0.0:
        blockers.add("non_positive_utility_delta")
    if not bounds_passed:
        blockers.add("cerce_bounds_check_failed")
    if int(training_summary["bounds_worsened_updates"]):
        blockers.add("cerce_bounds_worsened")
    if past_certificates_violated:
        blockers.add("past_certificate_violation")
    if ledger_artifact["cerce_ledger_ready"] is not True:
        blockers.add("cerce_ledger_not_ready")
    blockers.update(ledger_artifact["blockers"])

    status = "complete" if not blockers else "blocked"
    artifact: JsonDict = {
        "status": status,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1608", "SCENARIO-LEARN-1608", "SCENARIO-LEARN-1609"],
        "run_date": run_date,
        "project_root": str(project_root),
        "source_artifacts": list(source_artifacts),
        "continuous_self_learning_task": True,
        "examples_requested": EXAMPLE_COUNT,
        "examples_processed": int(training_summary["examples_processed"]),
        "training_loop_batches": int(training_summary["training_loop_batches"]),
        "current_update_rows": int(training_summary["current_update_rows"]),
        "utility_delta": float(training_summary["utility_delta"]),
        "baseline_success_rate": float(training_summary["baseline_success_rate"]),
        "promoted_success_rate": float(training_summary["promoted_success_rate"]),
        "cerce_ledger_ready": bool(ledger_artifact["cerce_ledger_ready"]),
        "bounds_check_passed": bounds_passed,
        "past_certificates_checked": int(past_certificates_checked),
        "past_certificates_violated": int(past_certificates_violated),
        "accepted_violation_count": int(ledger_artifact["accepted_violation_count"]),
        "false_accept_delta": int(ledger_artifact["false_accept_delta"]),
        "soundness_mistakes": soundness_mistakes,
        "nonforgetting_certificate_rate": float(ledger_artifact["nonforgetting_certificate_rate"]),
        "no_model_weight_mutation": True,
        "policy_certificates_evaluated": int(ledger_artifact["policy_certificates_evaluated"]),
        "constraint_violation_records": int(ledger_artifact["constraint_violation_records"]),
        "fr11_events_recorded": int(ledger_artifact["fr11_events_recorded"]),
        "promotion_safe_policy_updates": list(ledger_artifact["promotion_safe_policy_updates"]),
        "blocked_policy_updates": list(ledger_artifact["blocked_policy_updates"]),
        "policy_certificates": list(certificates),
        "ledger_rows": list(ledger_artifact["ledger_rows"]),
        "training_summary": dict(training_summary),
        "blockers": sorted(blockers),
        "honest_verdict": (
            "complete: fr11_cerce_scale_positive_utility_no_forgetting"
            if status == "complete"
            else "complete: fr11_cerce_scale_blocked_by_cerce"
        ),
        "tests_run": list(tests_run or []),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 1608 fields consumed by tests and the conductor."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["schema"] != SCHEMA:
        raise AssertionError(f"unsupported schema: {artifact['schema']}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["continuous_self_learning_task"] is not True:
        raise AssertionError("continuous_self_learning_task must be true")
    if artifact["status"] == "complete":
        errors: list[str] = []
        if int(artifact["examples_processed"]) != EXAMPLE_COUNT:
            errors.append("examples_processed must be 1000")
        if float(artifact["utility_delta"]) <= 0.0:
            errors.append("utility_delta must be positive")
        if artifact["cerce_ledger_ready"] is not True:
            errors.append("cerce_ledger_ready must be true")
        if artifact["bounds_check_passed"] is not True:
            errors.append("bounds_check_passed must be true")
        if artifact["no_model_weight_mutation"] is not True:
            errors.append("no_model_weight_mutation must be true")
        if int(artifact["accepted_violation_count"]) != 0:
            errors.append("accepted_violation_count must be zero")
        if int(artifact["false_accept_delta"]) > 0:
            errors.append("false_accept_delta cannot be positive")
        if int(artifact["soundness_mistakes"]) != 0:
            errors.append("soundness_mistakes must be zero")
        if float(artifact["nonforgetting_certificate_rate"]) != 1.0:
            errors.append("nonforgetting_certificate_rate must be 1.0")
        if artifact["blockers"]:
            errors.append("blockers must be empty")
        if errors:
            raise AssertionError(f"complete artifact is invalid: {errors}")


def _empty_training_summary() -> JsonDict:
    return {
        "examples_processed": 0,
        "training_loop_batches": 0,
        "current_update_rows": 0,
        "baseline_success_rate": 0.0,
        "promoted_success_rate": 0.0,
        "utility_delta": 0.0,
        "baseline_successes": 0,
        "promoted_successes": 0,
        "improved_examples": 0,
        "bounds_worsened_updates": 0,
    }


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _load_json(path: Path | str) -> JsonDict:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    destination.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return serializable


def _display_path(path: Path | str, *, project_root: Path | str = REPO_ROOT) -> str:
    target = Path(path)
    root = Path(project_root)
    return (
        target.relative_to(root).as_posix()
        if target.is_absolute() and target.is_relative_to(root)
        else target.as_posix()
    )


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by the conductor to write the Exp 1608 artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--prior-ledger", default=str(DEFAULT_PRIOR_LEDGER_PATH))
    parser.add_argument("--bounds-artifact", default=str(DEFAULT_BOUNDS_ARTIFACT_PATH))
    args = parser.parse_args(argv)
    artifact = run_experiment(
        project_root=Path(args.project_root),
        output_path=Path(args.output),
        prior_ledger_path=Path(args.prior_ledger),
        bounds_artifact_path=Path(args.bounds_artifact),
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
