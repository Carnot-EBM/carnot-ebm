"""Authenticate the V638 held-out capture before any semantic replay.

This audit does not invoke a model. It writes a complete blocked record when
the external Exp7251 capture is absent, so later work cannot mistake silence
for a scientific result.

Spec refs: REQ-VERIFY-7252, SCENARIO-VERIFY-7252-BLOCK, and
SCENARIO-VERIFY-7252-REPLAY.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

from carnot import experiment_7239_v637_semantic_audit as prior_audit
from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260912"
MILESTONE = "2026.09.638"
EXPERIMENT_ID = "exp7252-semantic-audit"
RANDOM_SEED = 7_252_001
BOOTSTRAP_DRAWS = 10_000
MODEL_SPECS: list[JsonDict] = []

RESULT_PATH = Path("results/experiment_7252_v638_semantic_audit.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7252/running.json")
CAPTURE_PATH = Path("results/experiment_7251_v638_mention_heldout.json")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7252_v638_semantic_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7252_v638_semantic_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7252_v638_semantic_audit.py")

SOURCE_PATHS = {
    "agents": Path("AGENTS.md"),
    "claude": Path("CLAUDE.md"),
    "codex": Path("CODEX.md"),
    "research_program": Path("research-program.md"),
    "research_roadmap": Path("research-roadmap.yaml"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "verification_spec": SPEC_PATH,
    "prior_audit_module": Path("python/carnot/experiment_7239_v637_semantic_audit.py"),
    "mention_fixture_module": Path("python/carnot/experiment_7236_v637_mention_fixture.py"),
    "prior_audit_artifact": Path("results/experiment_7239_v637_semantic_audit.json"),
    "row_consistency_lint": Path("scripts/verdict_row_consistency_lint.py"),
    "adversarial_verifier": Path("scripts/adversarial_verify.py"),
    "capture": CAPTURE_PATH,
    "module": MODULE_PATH,
    "entrypoint": WRAPPER_PATH,
    "focused_tests": TEST_PATH,
}

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version this artifact; also emit experiment_id and milestone as ordinary top-level values.",
    "status": "A terminal artifact records complete or blocked; unfinished work belongs in a separate checkpoint.",
    "run_date": "Use 20260912 and retain actual UTC start and end timestamps.",
    "field_principles": "Store explanations here while leaving ordinary values at top level for consumers.",
    "preconditions_checked": "Record exact observed inputs, resource ownership, hashes and failed checks before expensive work.",
    "MODEL_SPECS": "Name only models this invocation may execute; source-model history lives in hashed sidecars.",
    "model_invoked": "Derive from actual current calls, not usable-answer count or a nested control arm.",
    "inference_substrate": "Describe the compute actually performed with a recognized literal.",
    "inference_substrate_class": "Use the closed compute class and its duration floor; never sleep or relabel to pass.",
    "execution_venue": "Use host for orchestration; actual board receipts separately name kv260, gatemate or polarfire.",
    "execution_host": "Record the real hostname separately from the closed venue vocabulary.",
    "duration_s": "Measure monotonic elapsed time for this invocation, with disjoint phase spans and no invented time.",
    "random_seed": "Freeze seeds before seeing outcomes so replay cannot select favorable runs.",
    "reproducibility_checksum": "Bind source code, input manifests, configuration and raw rows to the result.",
    "source_artifact_hashes": "Authenticate input bytes and preserve quarantine; readiness alone is insufficient.",
    "rows": "Retain every unit, arm, seed, metric, error, abstention and censoring state; aggregates must be recomputable.",
    "sample_size_budget": "State planned, attempted, completed and censored independent units and the fixed stopping rule.",
    "acceptance_gate_results": "For each frozen criterion record expected, observed and passed, plus its principle.",
    "gate_check_summary": "For every blocked_* verdict name the upstream, exact field or check, observed and expected values.",
    "verifier_is_oracle": "Expose exact-oracle use; oracle conformance cannot become learned verification evidence.",
    "honest_verdict": "Use complete_* for terminal measured findings and blocked_* for absent external prerequisites.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; a failed scientific acceptance gate forbids positive. External incompleteness is blocked, not retryable partial.",
    "validation_receipts": "Record actual command, exit code and log hash; no skipped, weakened, deleted or reverted tests.",
    "semantic_audit_complete_score": "One means the independent audit ran completely, even when value is null.",
    "semantic_value_score": "One requires all preregistered source-fidelity, coverage, safety and matched-control value gates.",
    "paired_interval_rows": "Retain sampling unit, group count, statistic and paired interval for every comparison.",
    "mutation_rows": "Name each adversarial change, expected failure and observed checker response.",
    "experiment_id": "Bind the ordinary experiment identifier used by roadmap consumers.",
    "milestone": "Bind the ordinary milestone identifier used by roadmap consumers.",
    "timestamps": "Record real UTC start and completion observations.",
    "phase_spans": "Keep measured phase work separate from total duration.",
    "current_invocation_counts": "Keep current model load, generation, and invocation counts separate from source history.",
}
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "validation_receipts",
    "semantic_audit_complete_score",
    "semantic_value_score",
    "paired_interval_rows",
    "mutation_rows",
)

canonical_json = prior_audit.canonical_json
sha256_bytes = prior_audit.sha256_bytes
sha256_file = prior_audit.sha256_file
artifact_checksum = prior_audit.artifact_checksum
gate_row = prior_audit.gate_row
gate_summary = prior_audit.gate_summary
unwrap_principle = prior_audit.unwrap_principle


def _utc_now() -> str:
    """Return an actual UTC observation for this process."""

    return datetime.now(UTC).isoformat()


def _progress(phase: int, event: str, detail: str) -> None:
    """Flush each observed boundary so the blocked audit stays inspectable."""

    print(f"[exp7252] phase {phase} {event}: {detail}", flush=True)


def _resolved_paths(root: Path, overrides: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Resolve fixed repository inputs and allow isolated missing-input tests."""

    paths = {name: root / path for name, path in SOURCE_PATHS.items()}
    for name, path in (overrides or {}).items():
        paths[name] = path if path.is_absolute() else root / path
    return paths


def _source_hashes(paths: Mapping[str, Path]) -> JsonDict:
    """Hash exact readable bytes and state which declared path is absent."""

    return {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else "missing",
        }
        for name, path in sorted(paths.items())
    }


def _unevaluated_acceptance() -> list[JsonDict]:
    """Keep each frozen value gate visible when upstream capture is absent."""

    criteria: list[tuple[str, str, Any]] = [
        ("supported_pointer_semantic_fidelity", ">=0.80", 0.80),
        ("pointer_coverage", ">=0.50", 0.50),
        ("fidelity_delta_over_offset_ci95_lower", ">0", 0.0),
        ("selective_error_delta_over_direct_ci95_upper", "<0", 0.0),
        ("contradiction_control_false_accepts", "==0", 0),
        ("unknown_unsupported_distinction_preserved", "==true", True),
    ]
    return [
        {
            "criterion": name,
            "expected": operator,
            "expected_value": expected,
            "observed": None,
            "evaluated": False,
            "passed": False,
            "principle": "Do not infer semantic value from an unavailable upstream capture.",
        }
        for name, operator, expected in criteria
    ]


def base_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete checkpoint before any fallible source read."""

    return {
        "schema": {
            "name": "carnot.experiment_7252_v638_semantic_audit",
            "version": 1,
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
        },
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_independent_units": 64,
            "planned_arms": 3,
            "planned_comparison_rows": 192,
            "planned_bootstrap_resamples": BOOTSTRAP_DRAWS,
            "attempted_independent_units": 0,
            "completed_independent_units": 0,
            "censored_independent_units": 64,
            "stopping_rule": "audit all 64 paired groups once or stop at the first external Exp7251 block",
        },
        "acceptance_gate_results": _unevaluated_acceptance(),
        "gate_check_summary": gate_summary(None),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_exp7252_running_checkpoint_only",
        "verdict_class": "partial",
        "validation_receipts": [],
        "semantic_audit_complete_score": 0,
        "semantic_value_score": 0,
        "paired_interval_rows": [],
        "mutation_rows": [],
        "timestamps": {"started_at_utc": _utc_now(), "completed_at_utc": None},
        "phase_spans": [],
        "current_invocation_counts": {
            "model_loads": 0,
            "generations": 0,
            "model_invocations": 0,
        },
    }


def collect_preconditions(
    root: Path,
    run_date: str,
    output_root: Path,
    *,
    path_overrides: Mapping[str, Path] | None = None,
) -> tuple[list[JsonDict], JsonDict | None]:
    """Observe sources, imports, outputs, hashes, quarantine, and readiness."""

    paths = _resolved_paths(root, path_overrides)
    checks = [
        gate_row(
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            upstream=None,
            artifact_field="run_date",
        )
    ]
    capture_path = paths["capture"]
    capture: JsonDict | None = None
    capture_observed: Any = "missing_artifact"
    if capture_path.is_file() and os.access(capture_path, os.R_OK):
        try:
            loaded = json.loads(capture_path.read_bytes())
        except (OSError, json.JSONDecodeError):
            capture_observed = "invalid_json"
        else:
            if isinstance(loaded, dict):
                capture = loaded
                capture_observed = unwrap_principle(loaded.get("mention_capture_complete_score"))
            else:
                capture_observed = "artifact_not_mapping"
    checks.append(
        gate_row(
            "upstream_capture_ready",
            1,
            capture_observed,
            capture_observed == 1,
            upstream="experiment_7251_v638_mention_heldout",
            artifact_field="mention_capture_complete_score",
        )
    )

    for name, path in paths.items():
        if name == "capture":
            continue
        readable = path.is_file() and os.access(path, os.R_OK)
        checks.append(
            gate_row(
                "required_source",
                "readable_file",
                str(path) if readable else "missing_or_unreadable",
                readable,
                upstream=name,
                artifact_field="path",
            )
        )
    spec_ready = paths["verification_spec"].is_file() and (
        "REQ-VERIFY-7252" in paths["verification_spec"].read_text(encoding="utf-8")
    )
    checks.append(
        gate_row(
            "driving_spec",
            True,
            spec_ready,
            spec_ready,
            upstream="openspec/capabilities/verification/spec.md",
            artifact_field="REQ-VERIFY-7252",
        )
    )
    imports_ready = callable(prior_audit.sha256_file) and callable(prior_audit.validate_artifact)
    checks.append(
        gate_row(
            "required_imports",
            True,
            imports_ready,
            imports_ready,
            upstream="python/carnot",
            artifact_field="experiment_7239_v637_semantic_audit",
        )
    )
    for name, path in (
        ("checkpoint", output_root / CHECKPOINT_PATH),
        ("result", output_root / RESULT_PATH),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = path.parent.is_dir() and os.access(path.parent, os.W_OK)
        checks.append(
            gate_row(
                "output_destination",
                "writable_directory",
                str(path.parent) if writable else "missing_or_unwritable",
                writable,
                upstream=name,
                artifact_field="parent",
            )
        )

    if capture is not None:
        quarantine = prior_audit._quarantined(capture)
        checks.append(
            gate_row(
                "structured_quarantine",
                False,
                quarantine,
                not quarantine,
                upstream="experiment_7251_v638_mention_heldout",
                artifact_field="flagged_adversarial|quarantined|fabricated",
            )
        )
    else:
        checks.append(
            gate_row(
                "structured_quarantine",
                False,
                "not_observed_missing_artifact",
                False,
                upstream="experiment_7251_v638_mention_heldout",
                artifact_field="flagged_adversarial|quarantined|fabricated",
            )
        )
    return checks, capture


def _seal(artifact: JsonDict, path: Path, started: float, *, terminal: bool = False) -> None:
    """Refresh measured duration and checksum immediately before atomic write."""

    artifact["duration_s"] = time.monotonic() - started
    if terminal:
        artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    atomic_write_json(path, artifact, allow_override=False, sort_keys=True)


def validate_artifact(
    value: object,
    root: Path | None = None,
    *,
    path_overrides: Mapping[str, Path] | None = None,
) -> list[str]:
    """Cold-check the terminal block, exact source state, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    repository = root or find_repo_root(start=__file__)
    paths = _resolved_paths(repository, path_overrides)
    if value.get("schema") != {
        "name": "carnot.experiment_7252_v638_semantic_audit",
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
    }:
        errors.append("schema")
    if value.get("experiment_id") != EXPERIMENT_ID or value.get("milestone") != MILESTONE:
        errors.append("ordinary_identity")
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("execution_venue") != "host" or not value.get("execution_host"):
        errors.append("execution_identity")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_invoked") is not False
        or value.get("current_invocation_counts")
        != {"model_loads": 0, "generations": 0, "model_invocations": 0}
    ):
        errors.append("model_contract")
    if value.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if value.get("source_artifact_hashes") != _source_hashes(paths):
        errors.append("source_artifact_hashes")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")

    summary = value.get("gate_check_summary")
    blocked_state = (
        value.get("status") == "blocked"
        and value.get("verdict_class") == "blocked"
        and value.get("honest_verdict") == "blocked_exp7252_missing_upstream_artifact"
        and value.get("semantic_audit_complete_score") == 0
        and value.get("semantic_value_score") == 0
        and value.get("rows") == []
        and value.get("paired_interval_rows") == []
        and value.get("mutation_rows") == []
        and value.get("inference_substrate") == "blocked_no_run"
        and value.get("inference_substrate_class") == "blocked_no_run"
    )
    if not blocked_state:
        errors.append("blocked_terminal_state")
    expected_summary = {
        "passed": False,
        "failed_check": "upstream_capture_ready",
        "upstream": "experiment_7251_v638_mention_heldout",
        "artifact_field": "mention_capture_complete_score",
        "expected_value": 1,
        "observed_value": "missing_artifact",
    }
    if summary != expected_summary:
        errors.append("gate_check_summary")
    capture = paths["capture"]
    if capture.is_file():
        errors.append("upstream_state_changed")
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list) or gates != _unevaluated_acceptance():
        errors.append("acceptance_gate_results")
    return list(dict.fromkeys(errors))


def attach_validation_receipts(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    root: Path | None = None,
    *,
    path_overrides: Mapping[str, Path] | None = None,
) -> JsonDict:
    """Attach observed command receipts only after cold validation passes."""

    if validate_artifact(artifact, root, path_overrides=path_overrides):
        raise ValueError("validation_receipt_source_artifact")
    required = {"command", "exit_code", "classification", "log_sha256"}
    if any(set(row) != required for row in rows):
        raise ValueError("validation_receipt_schema")
    value = deepcopy(dict(artifact))
    value["validation_receipts"] = deepcopy(list(rows))
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    output_root: Path | None = None,
    path_overrides: Mapping[str, Path] | None = None,
) -> JsonDict:
    """Write the required external-prerequisite block without model work."""

    started = time.monotonic()
    repository = root or find_repo_root(start=__file__)
    destination = output_root or repository
    checkpoint = destination / CHECKPOINT_PATH
    result = destination / RESULT_PATH

    _progress(0, "start", "write schema-complete provisional checkpoint")
    artifact = base_artifact(run_date)
    _seal(artifact, checkpoint, started)
    _progress(0, "end", str(checkpoint))

    phase_started = time.monotonic()
    _progress(1, "start", "authenticate upstream bytes, quarantine, imports, and outputs")
    checks, capture = collect_preconditions(
        repository,
        run_date,
        destination,
        path_overrides=path_overrides,
    )
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(_resolved_paths(repository, path_overrides))
    artifact["phase_spans"].append(
        {
            "phase": 1,
            "name": "preconditions_and_upstream_authentication",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    failure = next((row for row in checks if row.get("passed") is not True), None)
    _progress(1, "end", f"first_failure={failure.get('check') if failure else None}")

    _progress(2, "start", "verify capture readiness without producer aggregate trust")
    if failure is None or capture is not None:
        raise ValueError("Exp7252 complete replay requires the authentic Exp7251 raw capture")
    _progress(2, "end", "capture_bytes=absent completed_units=0")

    _progress(3, "start", "record blocked reducer and bootstrap disposition")
    artifact.update(
        {
            "status": "blocked",
            "gate_check_summary": gate_summary(failure),
            "verdict_class": "blocked",
            "honest_verdict": "blocked_exp7252_missing_upstream_artifact",
        }
    )
    _progress(3, "end", "paired_groups=0 bootstrap_draws=0 model_calls=0")

    _progress(4, "start", "record offline mutation disposition")
    _progress(4, "end", "mutations=0 reason=external_block")
    _progress(5, "start", "apply frozen promotion gate disposition")
    _progress(5, "end", "complete_score=0 value_score=0 verdict=blocked")
    _progress(6, "start", "fresh-process replay deferred to validation command")
    _progress(6, "end", "current_model_loads=0 current_generations=0")

    _progress(7, "validation_start", "cold-check terminal artifact before atomic write")
    artifact["duration_s"] = time.monotonic() - started
    artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, repository, path_overrides=path_overrides)
    _progress(7, "validation_end", f"errors={errors}")
    if errors:
        raise ValueError(f"invalid Exp7252 artifact: {errors}")

    _progress(8, "write_start", f"atomic terminal artifact={result}")
    _seal(artifact, checkpoint, started)
    _seal(artifact, result, started, terminal=True)
    _progress(8, "write_end", str(result))
    return artifact


def replay_terminal_artifact(
    root: Path | None = None,
    *,
    output_root: Path | None = None,
    path_overrides: Mapping[str, Path] | None = None,
) -> JsonDict:
    """Cold-reconstruct the blocked disposition without invoking a model."""

    repository = root or find_repo_root(start=__file__)
    destination = output_root or repository
    _progress(9, "subprocess_start", "independent terminal artifact replay")
    artifact = json.loads((destination / RESULT_PATH).read_text(encoding="utf-8"))
    errors = validate_artifact(artifact, repository, path_overrides=path_overrides)
    if errors:
        raise ValueError(f"terminal_artifact_validation:{errors}")
    _progress(9, "subprocess_end", "status=blocked regenerated_model_calls=0")
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the task contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Build and cold-check the fixed-date terminal audit disposition."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    root = find_repo_root(start=__file__)
    artifact = run_experiment(root, args.date)
    errors = validate_artifact(artifact, root)
    if errors:
        print(f"[exp7252] invalid artifact: {errors}", flush=True)
        return 1
    print(
        f"[exp7252] terminal verdict={artifact['honest_verdict']} "
        f"complete={artifact['semantic_audit_complete_score']} "
        f"value={artifact['semantic_value_score']}",
        flush=True,
    )
    return 0
