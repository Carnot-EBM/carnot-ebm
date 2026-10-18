"""Independently audit the attempted V649 delayed-adapter online trial.

The producer result can be absent even when large checkpoint files remain. In
that case this module records the missing identity as an external block. It
does not infer a producer verdict from checkpoints or from historical prose.

Spec refs: REQ-REPORT-7401 and SCENARIO-REPORT-7401-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
RUN_DATE = "20260918"
MILESTONE = "2026.09.649"
EXPERIMENT_ID = "exp7401-online-audit"
SCHEMA = "carnot.exp7401.v649.online_audit.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7401_v649_online_audit.json")
RAW_DIR = Path("results/raw/experiment_7401_v649_online_audit")
MODULE_PATH = Path("python/carnot/experiment_7401_v649_online_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7401_v649_online_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7401_v649_online_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ADAPTER_PATH = Path("results/experiment_7397_v649_delayed_adapter.json")
TRIAL_PATH = Path("results/experiment_7399_v649_online_trial.json")
PROTOCOL_PATH = Path("results/experiment_7382_v648_decision_protocol.json")
TRAINING_PATH = Path("results/experiment_7385_v648_decision_training.json")
CORPUS_PATH = Path("data/fover_corpus_v4.json")
PRODUCER_INPUTS = (ADAPTER_PATH, TRIAL_PATH)
SUPPORTING_INPUTS = (
    PROTOCOL_PATH,
    TRAINING_PATH,
    CORPUS_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7382_v648_decision_protocol.py"),
    Path("python/carnot/experiment_7385_v648_decision_training.py"),
    Path("python/carnot/experiment_7397_v649_delayed_adapter.py"),
    Path("python/carnot/experiment_7399_v649_online_trial.py"),
)
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
TRAINING_SEEDS = (7397001, 7397002, 7397003, 7397004, 7397005)
MUTATIONS = (
    "future_label_leak",
    "duplicate_update",
    "swapped_partition",
    "missing_cost",
    "changed_probability",
    "favorable_seed_deletion",
)
V649_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so evidence changes have one exact identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a source as bytes because parsed equality is not source equality."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON value so readers never see a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _load_object(path: Path) -> JsonDict:
    """Return an object only when external bytes contain a JSON dictionary."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Keep each failed input gate attributable to one exact field."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], dict[str, JsonDict]]:
    """Authenticate named current inputs without promoting historical descriptions."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    sources: dict[str, JsonDict] = {}
    for relative in (*PRODUCER_INPUTS, SPEC_PATH, *SUPPORTING_INPUTS):
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        if present:
            hashes[relative.as_posix()] = sha256_file(path)
        if relative in PRODUCER_INPUTS:
            checks.append(
                _precondition(
                    f"source_bytes:{relative.as_posix()}",
                    relative.as_posix(),
                    "bytes",
                    "readable_nonempty_bytes",
                    "readable_nonempty_bytes" if present else None,
                )
            )
            artifact = _load_object(path) if present else {}
            sources[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": hashes.get(relative.as_posix()),
                "status": artifact.get("status"),
                "verdict_class": artifact.get("verdict_class"),
                "flagged_adversarial": artifact.get("flagged_adversarial"),
            }
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7401",
            "REQ-REPORT-7401" if "REQ-REPORT-7401" in spec_text else None,
        )
    )
    adapter = _load_object(root / ADAPTER_PATH)
    trial = _load_object(root / TRIAL_PATH)
    if adapter:
        checks.extend(
            (
                _precondition(
                    "adapter_identity",
                    ADAPTER_PATH.as_posix(),
                    "experiment_id",
                    "exp7397-delayed-adapter",
                    adapter.get("experiment_id"),
                ),
                _precondition(
                    "adapter_ready",
                    ADAPTER_PATH.as_posix(),
                    "delayed_adapter_ready_score",
                    1,
                    adapter.get("delayed_adapter_ready_score"),
                ),
                _precondition(
                    "adapter_flag",
                    ADAPTER_PATH.as_posix(),
                    "flagged_adversarial",
                    False,
                    adapter.get("flagged_adversarial"),
                ),
            )
        )
    if trial:
        checks.extend(
            (
                _precondition(
                    "trial_identity",
                    TRIAL_PATH.as_posix(),
                    "experiment_id",
                    "exp7399-online-trial",
                    trial.get("experiment_id"),
                ),
                _precondition(
                    "trial_capture",
                    TRIAL_PATH.as_posix(),
                    "online_capture_complete_score",
                    1,
                    trial.get("online_capture_complete_score"),
                ),
                _precondition(
                    "trial_flag",
                    TRIAL_PATH.as_posix(),
                    "flagged_adversarial",
                    False,
                    trial.get("flagged_adversarial"),
                ),
            )
        )
    return checks, hashes, sources


def _sigmoid(value: float) -> float:
    """Convert a finite logit without overflow at large magnitudes."""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _typed_action(probability: float, policy: Mapping[str, Any]) -> str:
    """Select a typed action while leaving the original answer unchanged."""

    if policy.get("accept_enabled") is True and probability <= float(policy["accept_threshold"]):
        return "accept"
    if policy.get("reject_enabled") is True and probability >= float(policy["reject_threshold"]):
        return "reject"
    return "escalate"


def recompute_stream(
    events: Sequence[Mapping[str, Any]], unit: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[JsonDict]:
    """Recompute affine predictions and updates without a producer reducer."""

    affine = dict(unit["affine"])
    seen: set[str] = set()
    rows: list[JsonDict] = []
    for index, source in enumerate(events):
        event = dict(source)
        event_id = str(event["event_id"])
        if event.get("label_visible_before_prediction") is True:
            raise ValueError(f"future_label_leak:{event_id}")
        if event_id in seen:
            raise ValueError(f"duplicate_update:{event_id}")
        seen.add(event_id)
        energy = float(event["raw_energy"])
        before = canonical_hash(affine)
        probability = _sigmoid(float(affine["a"]) * energy + float(affine["b"]))
        action = _typed_action(probability, policy)
        label = int(event["label"])
        brier = (probability - label) ** 2
        log_loss = -(
            label * math.log(max(probability, 1e-15))
            + (1 - label) * math.log(max(1.0 - probability, 1e-15))
        )
        disposition = str(event.get("feedback_disposition", "pending"))
        if disposition == "committed":
            error = probability - label
            gradient = [error * energy, error]
            norm = math.sqrt(sum(value * value for value in gradient))
            cap = float(unit.get("gradient_norm_cap", 1.0))
            scale = min(1.0, cap / norm) if norm else 1.0
            rate = float(unit.get("learning_rate", 0.01))
            affine["a"] = min(4.0, max(0.25, float(affine["a"]) - rate * gradient[0] * scale))
            affine["b"] = min(8.0, max(-8.0, float(affine["b"]) - rate * gradient[1] * scale))
        producer_action = event.get("producer_typed_action", action)
        rows.append(
            {
                "event_id": event_id,
                "group_id": str(event.get("group_id", event_id)),
                "stream_index": index,
                "arm": str(unit["arm"]),
                "seed": int(unit["seed"]),
                "partition": str(event.get("partition")),
                "raw_energy": energy,
                "label": label,
                "probability": probability,
                "typed_action": action,
                "typed_action_changed": action != producer_action,
                "original_answer_corrected": False,
                "action_harm": bool(
                    (action == "accept" and label == 1) or (action == "reject" and label == 0)
                ),
                "brier_loss": brier,
                "log_loss": log_loss,
                "full_cost_s": float(event["full_cost_s"]),
                "feedback_disposition": disposition,
                "prediction_before_feedback": True,
                "state_hash_before_update": before,
                "state_hash_after_update": canonical_hash(affine),
            }
        )
    return rows


def build_fixture_evidence() -> JsonDict:
    """Build small valid evidence for mutation and cold-reader tests."""

    rows: list[JsonDict] = []
    for seed in TRAINING_SEEDS:
        rows.append(
            {
                "event_id": f"event-{seed}",
                "group_id": "fixture-group",
                "ordering": "fixed_hash_order",
                "feedback_condition": "primary_delay_1",
                "arm": "adaptive_affine_gibbs",
                "seed": seed,
                "partition": "training",
                "raw_energy": 0.0,
                "affine_a": 1.0,
                "affine_b": 0.0,
                "probability": 0.5,
                "full_cost_s": 0.01,
                "label_visible_before_prediction": False,
                "update_count": 1,
                "benefit_delta": -0.01 if seed == TRAINING_SEEDS[0] else 0.0,
            }
        )
    return {
        "registered_seeds": list(TRAINING_SEEDS),
        "scorer_class": "learned_affine",
        "discrimination_claim": False,
        "rows": rows,
    }


def evidence_errors(evidence: Mapping[str, Any]) -> list[str]:
    """Reject private evidence changes at the boundary each change attacks."""

    rows = [dict(row) for row in evidence.get("rows", []) if isinstance(row, Mapping)]
    errors: list[str] = []
    if any(row.get("label_visible_before_prediction") is True for row in rows):
        errors.append("future_label_leak")
    identities = [
        (
            row.get("ordering"),
            row.get("feedback_condition"),
            row.get("seed"),
            row.get("arm"),
            row.get("event_id"),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        errors.append("duplicate_update")
    if any(row.get("partition") != "training" for row in rows):
        errors.append("swapped_partition")
    if any(not isinstance(row.get("full_cost_s"), (int, float)) for row in rows):
        errors.append("missing_cost")
    for row in rows:
        required = ("raw_energy", "affine_a", "affine_b", "probability")
        if all(isinstance(row.get(name), (int, float)) for name in required):
            expected = _sigmoid(
                float(row["affine_a"]) * float(row["raw_energy"]) + float(row["affine_b"])
            )
            if not math.isclose(float(row["probability"]), expected, rel_tol=0.0, abs_tol=1e-12):
                errors.append("probability_mismatch")
                break
    observed_seeds = sorted({int(row["seed"]) for row in rows if isinstance(row.get("seed"), int)})
    if observed_seeds != sorted(int(seed) for seed in evidence.get("registered_seeds", [])):
        errors.append("registered_seed_missing")
    probabilities = [
        float(row["probability"])
        for row in rows
        if isinstance(row.get("probability"), (int, float))
    ]
    if (
        evidence.get("discrimination_claim") is True
        and evidence.get("scorer_class") != "prevalence_baseline"
        and probabilities
        and max(probabilities) == min(probabilities)
    ):
        errors.append("learned_constant_discrimination_claim")
    return list(dict.fromkeys(errors))


def run_mutation_checks(evidence: Mapping[str, Any]) -> list[JsonDict]:
    """Change private copies and require each named boundary to reject it."""

    mutations: dict[str, tuple[str, Any]] = {
        "future_label_leak": (
            "future_label_leak",
            lambda value: value["rows"][0].__setitem__("label_visible_before_prediction", True),
        ),
        "duplicate_update": (
            "duplicate_update",
            lambda value: value["rows"].append(deepcopy(value["rows"][0])),
        ),
        "swapped_partition": (
            "swapped_partition",
            lambda value: value["rows"][0].__setitem__("partition", "final_test"),
        ),
        "missing_cost": ("missing_cost", lambda value: value["rows"][0].pop("full_cost_s")),
        "changed_probability": (
            "probability_mismatch",
            lambda value: value["rows"][0].__setitem__("probability", 0.9),
        ),
        "favorable_seed_deletion": (
            "registered_seed_missing",
            lambda value: value.__setitem__(
                "rows", [row for row in value["rows"] if row["seed"] != TRAINING_SEEDS[0]]
            ),
        ),
    }
    output: list[JsonDict] = []
    for name in MUTATIONS:
        expected, mutate = mutations[name]
        private = deepcopy(dict(evidence))
        mutate(private)
        errors = evidence_errors(private)
        output.append(
            {
                "mutation": name,
                "private_copy": True,
                "rejecting_check": expected,
                "observed_errors": errors,
                "rejected": expected in errors,
            }
        )
    return output


def classify_terminal(
    inputs_present: bool,
    inputs_valid: bool,
    audit_complete: bool,
    efficacy_passed: bool,
    required_validation_passed: bool,
) -> JsonDict:
    """Keep external absence, invalid evidence, completion, and value separate."""

    if not inputs_present:
        verdict, honest = "blocked", "blocked_missing_unchanged_online_producer_inputs"
    elif not inputs_valid or not required_validation_passed:
        verdict, honest = (
            "disqualified",
            "complete_disqualified_invalid_online_evidence_or_validation",
        )
    elif efficacy_passed and audit_complete:
        verdict, honest = "positive", "complete_positive_online_value_independently_confirmed"
    else:
        verdict, honest = "null", "complete_null_online_value_not_confirmed"
    return {
        "verdict_class": verdict,
        "honest_verdict": honest,
        "online_audit_complete_score": int(audit_complete and verdict in {"positive", "null"}),
        "online_value_confirmed_score": int(verdict == "positive"),
    }


def _gate_summary(preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact upstream failure while retaining every failure."""

    failures = [dict(row) for row in preconditions if row.get("passed") is not True]
    return {
        "all_required_passed": not failures,
        "required_failure_count": len(failures),
        "first_required_failure": failures[0] if failures else None,
        "required_failures": failures,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary fields without changing their machine-readable shape."""

    return {key: f"{key} is recorded directly under the Exp7401 audit contract." for key in keys}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind source identities, gates, rows, mutations, and terminal class."""

    excluded = {
        "reproducibility_checksum",
        "field_principles",
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans",
        "validation_receipts",
    }
    return canonical_hash({key: value for key, value in artifact.items() if key not in excluded})


def build_blocked_artifact(
    *,
    run_date: str,
    started_at: str,
    completed_at: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    source_sidecar: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal record required when unchanged producers are absent."""

    failures = [dict(row) for row in preconditions if row.get("passed") is not True]
    gates = [
        {
            "category": "completion",
            "check": str(row["check"]),
            "operator": "==",
            "expected": row.get("expected"),
            "observed": row.get("observed"),
            "passed": row.get("passed") is True,
        }
        for row in preconditions
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 3,
        "status": "terminal_blocked_missing_online_producer_inputs",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "duration_s": duration_s,
        "phase_spans": [dict(row) for row in phase_spans],
        "preconditions_checked": [dict(row) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "Host CPU aggregation with independent Python scalar, NumPy-compatible checkpoint, JAX-source, and exact-reducer inspection; no LLM load or generation.",
        "inference_substrate_class": "aggregation",
        "inference_substrate_details": {
            "device": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax_execution_performed": False,
            "exact_solver_execution_performed": False,
        },
        "execution_venue": "host",
        "random_seed": {
            "experiment": 7401001,
            "registered_training_seeds": list(TRAINING_SEEDS),
            "resampling": 7399307,
        },
        "source_artifact_hashes": dict(source_hashes),
        "source_authentication_sidecar": dict(source_sidecar),
        "rows": [],
        "mutation_rows": [],
        "sample_size_budget": {
            "planned_comparative_rows": 0,
            "attempted_comparative_rows": 0,
            "completed_comparative_rows": 0,
            "censored_comparative_rows": 0,
            "unstarted_comparative_rows": 0,
            "effective_independent_group_count": 0,
            "limits": "The absent producer artifact prevents a trustworthy row roster.",
            "stop_rule": "Stop dependent science at the first missing unchanged named input.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(preconditions),
        "verifier_is_oracle": False,
        "honest_verdict": "blocked_missing_unchanged_online_producer_inputs",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [dict(row) for row in validation_receipts],
        "repository_health": {
            "status": "not_used_for_scientific_readiness",
            "affects_required_checks": False,
            "unrelated_findings": [],
        },
        "promotion_score": 0,
        "online_audit_complete_score": 0,
        "online_value_confirmed_score": 0,
        "small_ebm_training": {"performed": False, "current_llm_calls": 0},
        "diagnostic_recomputation": {
            "performed": False,
            "reason": "Exact producer rows and producer eligibility fields are absent.",
            "missing_input_count": len(failures),
        },
        "generator_claim": False,
        "population_calibration_claim": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, declarations, blocked gates, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
        or not isinstance(artifact.get("inference_substrate"), str)
    ):
        errors.append("substrate_declaration_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("oracle_declaration_mismatch")
    if artifact.get("verdict_class") == "blocked":
        failure = (artifact.get("gate_check_summary") or {}).get("first_required_failure")
        if not isinstance(failure, Mapping) or not {
            "upstream",
            "check",
            "artifact_field",
            "expected",
            "observed",
        } <= set(failure):
            errors.append("blocked_gate_summary_missing")
        if artifact.get("rows") or artifact.get("mutation_rows"):
            errors.append("blocked_artifact_has_dependent_rows")
        if any(
            artifact.get(field) != 0
            for field in (
                "promotion_score",
                "online_audit_complete_score",
                "online_value_confirmed_score",
            )
        ):
            errors.append("blocked_scores_nonzero")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_commands(
    repo_root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:
    """Freeze the exact Exp7358 affected command set for this module."""

    return build_command_plan(repo_root, V649_MANIFEST, private_root)


def _utc_now() -> str:  # pragma: no cover - real clock boundary.
    return datetime.now(UTC).isoformat()


def _progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - live progress boundary.
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7401] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float
) -> JsonDict:  # pragma: no cover - real clock boundary.
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "checkpoint_utc": _utc_now(),
    }


def _write_source_sidecar(
    root: Path, sources: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - filesystem boundary.
    path = root / RAW_DIR / "source_authentication.json"
    value = {
        "schema": "carnot.exp7401.source_authentication.v1",
        "sources": dict(sources),
        "historical_invocation_counters_copied": False,
        "historical_substrate_declarations_copied": False,
    }
    atomic_json(path, value)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _terminal_commands(
    candidate: Path,
) -> list[PlannedCommand]:  # pragma: no cover - subprocess plan.
    python = ".venv/bin/python"
    cold = "import json,pathlib,sys;from carnot.experiment_7401_v649_online_audit import validate_artifact;v=json.loads(pathlib.Path(sys.argv[1]).read_text());e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_cold_replay",
                (python, "-u", "-c", cold, str(candidate)),
                "capability_end_to_end",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
    ]


def _entrypoint_receipt(
    root: Path, started_at: str, duration_s: float
) -> JsonDict:  # pragma: no cover - live invocation receipt.
    path = root / RAW_DIR / "validation/entrypoint/declared_entrypoint_e2e.log"
    argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    atomic_json(
        path,
        {
            "argv": argv,
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "duration_s": duration_s,
        },
    )
    return {
        "name": "declared_entrypoint_e2e",
        "command_argv": argv,
        "command": " ".join(argv),
        "command_environment": {
            key: os.environ[key]
            for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
        "scope": "capability_end_to_end",
        "exit_code": 0,
        "duration_s": duration_s,
        "log_path": path.relative_to(root).as_posix(),
        "log_sha256": sha256_file(path),
        "passed": True,
        "timed_out": False,
        "required": True,
        "command_category": "completion",
    }


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the declared E2E.
    """Authenticate sources, run scoped checks, and publish one terminal artifact."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    _progress(started, "startup", "start")

    phase_started = time.monotonic()
    _progress(started, "preconditions", "start")
    preconditions, source_hashes, sources = collect_preconditions(root)
    sidecar = _write_source_sidecar(root, sources)
    spans.append(_span("preconditions", phase_started, started))
    _progress(
        started,
        "preconditions",
        "end",
        failures=sum(row["passed"] is not True for row in preconditions),
    )

    private_root = Path(tempfile.mkdtemp(prefix="exp7401-validation-", dir="/tmp"))
    phase_started = time.monotonic()
    commands = build_validation_commands(root, private_root)
    plan_errors = validate_command_plan(root, V649_MANIFEST, commands)
    _progress(started, "affected_validation", "before_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    spans.append(_span("affected_validation", phase_started, started))
    _progress(started, "affected_validation", "after_subprocesses", receipts=len(affected))

    candidate = build_blocked_artifact(
        run_date=run_date,
        started_at=started_at,
        completed_at=_utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        source_hashes=source_hashes,
        source_sidecar=sidecar,
        validation_receipts=affected,
        phase_spans=spans,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    _progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    _progress(started, "candidate_write", "after_atomic", bytes=candidate_path.stat().st_size)

    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_subprocesses")
    terminal = run_categorized_commands(
        root, _terminal_commands(candidate_path), log_dir=root / RAW_DIR / "validation/terminal"
    )
    spans.append(_span("terminal_validation", phase_started, started))
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in terminal),
    )

    receipts = [
        *affected,
        *terminal,
        _entrypoint_receipt(root, started_at, time.monotonic() - started),
    ]
    final = build_blocked_artifact(
        run_date=run_date,
        started_at=started_at,
        completed_at=_utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        source_hashes=source_hashes,
        source_sidecar=sidecar,
        validation_receipts=receipts,
        phase_spans=spans,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "terminal_write", "before_atomic", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "terminal_write", "after_atomic", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public audit entrypoint arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit or its fresh-process terminal reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = validate_artifact(_load_object(args.cold_replay))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
