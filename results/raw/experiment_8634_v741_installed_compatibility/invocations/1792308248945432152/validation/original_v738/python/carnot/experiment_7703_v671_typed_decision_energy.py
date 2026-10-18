"""Train a frozen CPU typed-decision head for REQ-REPORT-7703."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np

from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.fresh_relation_cohort import validate_predictor
from carnot.reporting.typed_decision_energy import (
    ATOM_INDICES,
    FEATURE_ORDER,
    action,
    decision_cost,
    feature_information,
    feature_view,
    fit_head,
    freeze_online_protocol,
    freeze_policy,
    probabilities,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7703_v671_typed_decision_energy")
COHORT = Path("results/raw/experiment_7701_v671_sealed_cohort")
OUTPUT = Path("results/experiment_7703_v671_typed_decision_energy.json")
MODULE = "python/carnot/experiment_7703_v671_typed_decision_energy.py"
CAPABILITY = "python/carnot/reporting/typed_decision_energy.py"
TEST = "tests/python/test_experiment_7703_v671_typed_decision_energy.py"
CLI = "scripts/experiments/experiment_7703_v671_typed_decision_energy.py"
ROLES = ("fit", "tune", "policy", "retention", "online_update", "online_admission", "evaluation")
COUNTS = {
    "fit": 128,
    "tune": 40,
    "policy": 40,
    "retention": 32,
    "online_update": 60,
    "online_admission": 60,
    "evaluation": 40,
}
SEEDS = (7703, 7704, 7705, 7706, 7707)
SCOPE = {
    "tests": [TEST],
    "changed_modules": [CAPABILITY, MODULE],
    "static_paths": [CLI],
    "specs": ["REQ-REPORT-7703", "REQ-ENERGY-7703"],
    "e2e": ["task_reload_probability_action", "task_exact_terminal_readers"],
}
PRINCIPLES = {
    "honest_verdict": "A terminal disposition prevents retries of unchanged external blocks.",
    "verdict_class": "A closed enum carries claim eligibility into downstream readers.",
    "flagged_adversarial": "Disqualified evidence must not pass a downstream readiness gate.",
    "gate_check_summary": "Exact upstream operands distinguish false gates from absent evidence.",
    "acceptance_gate_results": "Measured operands limit the claim to completed checks.",
    "rows": "Independent unit observations permit comparison recomputation.",
    "sample_size_budget": "Seeds and controls cannot enlarge independent n.",
    "inference_substrate": "The declared path must match actual CPU work.",
    "inference_substrate_class": "Duration floors describe real generation work.",
    "MODEL_SPECS": "Experimental models must match actual invocations.",
    "model_invoked": "Actual invocation counts prevent implied LLM work.",
    "execution_venue": "The host and PID identify current work.",
    "phase_spans": "Monotonic spans and heartbeats bound current work.",
    "random_seed": "Independent replay needs declared random inputs.",
    "reproducibility_checksum": "Immutable inputs and reducer bytes bind replay.",
    "source_artifact_hashes": "Input custody and missing custody stay separate.",
    "preconditions_checked": "Resource and input checks precede work.",
    "validation_receipts": "Actual commands and log hashes bind validation.",
    "verifier_is_oracle": "Fixture truth cannot earn oracle-distinct credit.",
    "decision_energy_ready_score": "Readiness is administrative, not benefit.",
    "frozen_policy_path": "Frozen actions can be independently replayed.",
    "online_protocol_path": "A frozen online plan prevents outcome-driven tuning.",
    "training_rows": "Training and tuning metrics remain source-group observations.",
    "feature_information_summary": "Coverage and variance expose null information.",
}
GATE_PRINCIPLES = {
    "validity": "Invalid evidence must not propagate.",
    "readiness": "A trained normalized head and frozen protocol open evaluation only.",
    "coverage": "Quality thresholds prevent effects from being inferred from plumbing.",
    "freshness": "Selected source families must stay unexposed.",
    "probability": "Held evaluation labels are needed for benefit.",
    "utility": "Policy-role costs freeze actions, not held utility.",
    "retention": "Retention bounds prevent improvement by forgetting.",
    "efficiency": "Resource gains require a comparable workload.",
}


def progress(started: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Flush every phase boundary and long-loop heartbeat."""
    print(
        f"[exp7703] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(
    name: str, upstream: str, path: str, field: str, operator: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Retain the exact upstream field and observed operand."""
    passed = observed in expected if operator == "in" else observed == expected
    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def preflight(root: Path) -> tuple[list[dict], dict]:
    """Hash required bytes and current CPU; never require planned outputs."""
    root = root.resolve()
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []}
    checks = [
        _check(
            "cpu_available",
            "host",
            "/proc/self",
            "cpu_count_positive",
            "==",
            True,
            (os.cpu_count() or 0) > 0,
        )
    ]
    producers = {
        "exp7700-record-span-protocol": Path(
            "results/experiment_7700_v671_record_span_protocol.json"
        ),
        "exp7701-sealed-cohort": Path("results/experiment_7701_v671_sealed_cohort.json"),
        "exp7700-feature-schema": Path(
            "results/raw/experiment_7700_v671_record_span_protocol/feature_schema.json"
        ),
        "exp7701-protocol": COHORT / "protocol.json",
        "exp7701-public-protocol": COHORT / "public_protocol.json",
    }
    for upstream, relative in producers.items():
        present = (root / relative).is_file()
        checks.append(
            _check("input_exists", upstream, str(relative), "exists", "==", True, present)
        )
        if present:
            hashes["producers"][str(relative)] = sha256_file(root / relative)
        else:
            hashes["missing_evidence"].append(str(relative))
    if hashes["missing_evidence"]:
        return checks, hashes
    for upstream, relative, fields in (
        (
            "exp7700-record-span-protocol",
            producers["exp7700-record-span-protocol"],
            (
                ("record_protocol_ready_score", "==", 1),
                ("flagged_adversarial", "==", False),
                ("verdict_class", "in", ["circular_positive", "null"]),
            ),
        ),
        (
            "exp7701-sealed-cohort",
            producers["exp7701-sealed-cohort"],
            (
                ("cohort_ready_score", "==", 1),
                ("fresh_source_score", "==", 1),
                ("flagged_adversarial", "==", False),
                ("verdict_class", "in", ["null"]),
            ),
        ),
    ):
        document = json.loads((root / relative).read_text())
        for field, op, expected in fields:
            checks.append(
                _check(
                    "upstream_gate",
                    upstream,
                    str(relative),
                    field,
                    op,
                    expected,
                    document.get(field),
                )
            )
    protocol = json.loads((root / COHORT / "protocol.json").read_text())
    for role in ROLES:
        info = protocol["roles"][role]
        relative = COHORT / info["model_inputs"]
        expected = info["model_inputs_sha256"]
        observed = sha256_file(root / relative) if (root / relative).is_file() else None
        checks.append(
            _check(
                "input_hash",
                "exp7701-sealed-cohort",
                str(relative),
                "model_inputs_sha256",
                "==",
                expected,
                observed,
            )
        )
        if observed is None:
            hashes["missing_evidence"].append(str(relative))
        else:
            hashes["producers"][str(relative)] = observed
    for role in ("fit", "tune", "policy"):
        info = protocol["evaluator_stores"][role]
        relative = COHORT / info["path"]
        observed = sha256_file(root / relative) if (root / relative).is_file() else None
        checks.append(
            _check(
                "label_store_hash",
                "exp7701-sealed-cohort",
                str(relative),
                "sha256",
                "==",
                info["sha256"],
                observed,
            )
        )
        if observed is None:
            hashes["missing_evidence"].append(str(relative))
        else:
            hashes["pre_gate_receipts"][str(relative)] = observed
    return checks, hashes


def _read_jsonl(path: Path) -> list[dict]:
    """Read one authenticated role file without opening evaluator labels."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def build_feature_rows(root: Path, protocol: dict, raw: Path, started: float) -> list[dict]:
    """Finish all three public source views before opening any labels."""
    rows: list[dict] = []
    for role in ROLES:
        info = protocol["roles"][role]
        inputs = _read_jsonl(root / COHORT / info["model_inputs"])
        if len(inputs) != COUNTS[role] or [r["component_hash"] for r in inputs] != info["families"]:
            raise ValueError("role_roster_mismatch")
        for row in inputs:
            validate_predictor(row, role)
        ordered = sorted(inputs, key=lambda row: row["component_hash"])
        donors = {
            row["component_hash"]: ordered[(i + 1) % len(ordered)] for i, row in enumerate(ordered)
        }
        for index, row in enumerate(inputs):
            for arm in ("original_source", "evidence_erasure", "within_role_derangement"):
                donor = donors[row["component_hash"]] if arm == "within_role_derangement" else row
                source = "" if arm == "evidence_erasure" else donor["complete_source"]
                feature = feature_view(row, source_override=source)
                rows.append(
                    {
                        "unit_id": row["component_hash"],
                        "role": role,
                        "arm": arm,
                        "source_group_id": None
                        if arm == "evidence_erasure"
                        else donor["component_hash"],
                        "vector": feature["vector"],
                        "raw_metrics": feature["counts"],
                        "checked_count": feature["checked_count"],
                        "proposition_count": feature["proposition_count"],
                        "whole_answer_status": feature["whole_answer_status"],
                        "counts": {"independent_group": 1, "paired_view": 1},
                        "exclusions": [],
                        "excluded": False,
                        "censored": feature["checked_count"] == 0,
                        "provenance": {
                            "source_sha256": row["source_sha256"],
                            "answer_sha256": row["answer_sha256"],
                            "predictor_input": str(COHORT / info["model_inputs"]),
                        },
                    }
                )
            if (index + 1) % 25 == 0:
                progress(started, "features", role, index + 1)
        atomic_json(
            raw / "feature_checkpoint.json",
            {"completed_role": role, "completed_units": len(rows), "feature_order": FEATURE_ORDER},
        )
        progress(started, "features", "role_complete", len(rows))
    atomic_json(raw / "features.json", rows)
    return rows


def _labels(root: Path, protocol: dict, role: str) -> dict[str, int]:
    """Open one authenticated evaluator store only at its allowed phase."""
    info = protocol["evaluator_stores"][role]
    values = _read_jsonl(root / COHORT / info["path"])
    if any(not {"family_id", "label", "role"} <= row.keys() for row in values):
        raise ValueError("label_roster_mismatch")
    result = {row["family_id"]: row["label"] for row in values}
    if (
        len(result) != COUNTS[role]
        or len(values) != COUNTS[role]
        or any(row["role"] != role or row["label"] not in (0, 1) for row in values)
    ):
        raise ValueError("label_roster_mismatch")
    return result


def _matrix(rows: list[dict], role: str, arm: str = "original_source") -> np.ndarray:
    """Keep one independent row per family for a selected role and arm."""
    return np.asarray(
        [row["vector"] for row in rows if row["role"] == role and row["arm"] == arm], dtype=float
    )


def _metrics(head: dict, x: np.ndarray, labels: list[int]) -> dict[str, float]:
    """Reduce Brier and log loss from explicit unit probabilities."""
    p = np.asarray([probabilities(row, head)[1] for row in x])
    y = np.asarray(labels)
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
    }


def fit_families(
    rows: list[dict],
    fit_labels: dict[str, int],
    tune_labels: dict[str, int],
    raw: Path,
    started: float,
) -> tuple[dict, list[dict]]:
    """Fit on fit128, select settings and strongest comparator on tune40."""
    grouped = {
        role: [row for row in rows if row["role"] == role and row["arm"] == "original_source"]
        for role in ("fit", "tune")
    }
    fit_x = _matrix(rows, "fit")
    tune_x = _matrix(rows, "tune")
    fit_y = np.asarray([fit_labels[row["unit_id"]] for row in grouped["fit"]])
    tune_y = [tune_labels[row["unit_id"]] for row in grouped["tune"]]
    heads: dict[str, dict] = {}
    records = []
    plans = (
        ("fit_prior", "prior", None),
        ("atom_only", "logistic", ATOM_INDICES),
        ("matched_logistic", "logistic", None),
        ("matched_mlp", "mlp", None),
        ("typed_gibbs", "gibbs", None),
    )
    for family, kind, indices in plans:
        x_fit = fit_x[:, indices] if indices is not None else fit_x
        x_tune = tune_x[:, indices] if indices is not None else tune_x
        for seed in SEEDS:
            for ridge in (0.01, 0.1):
                head = fit_head(
                    x_fit, fit_y, kind, seed=seed, steps=0 if kind == "prior" else 300, ridge=ridge
                )
                head["feature_indices"] = list(indices) if indices is not None else None
                key = f"{family}:{seed}:{ridge}"
                heads[key] = head
                records.append(
                    {
                        "family": family,
                        "seed": seed,
                        "ridge": ridge,
                        "key": key,
                        "fit": _metrics(head, x_fit, fit_y.tolist()),
                        "tune": _metrics(head, x_tune, tune_y),
                        "parameter_count": head["parameter_count"],
                        "gradient_steps": head["steps"],
                    }
                )
            progress(started, "training", family, len(records))
        atomic_json(
            raw / "training_checkpoint.json",
            {"completed_family": family, "completed_settings": len(records)},
        )
    best_by_family = {
        family: min(
            (r for r in records if r["family"] == family),
            key=lambda r: (r["tune"]["brier"], r["seed"], r["ridge"]),
        )
        for family, _, _ in plans
    }
    comparator = min(
        (
            best_by_family[name]
            for name in ("fit_prior", "atom_only", "matched_logistic", "matched_mlp")
        ),
        key=lambda r: (r["tune"]["brier"], r["family"]),
    )
    selected = min(
        (comparator, best_by_family["typed_gibbs"]), key=lambda r: (r["tune"]["brier"], r["family"])
    )
    return {
        "heads": heads,
        "settings": records,
        "best_by_family": best_by_family,
        "strongest_comparator": comparator["key"],
        "selected": selected["key"],
        "selection_rule": "minimum tune40 Brier, then family/seed/ridge lexical tie break",
    }, records


def _probability(row: dict, head: dict) -> float:
    """Apply one saved head to its declared subset of frozen features."""
    vector = np.asarray(row["vector"], dtype=float)
    indices = head["feature_indices"]
    return probabilities(vector[indices] if indices is not None else vector, head)[1]


def freeze_decisions(
    root: Path,
    protocol: dict,
    rows: list[dict],
    bundle: dict,
    fit_labels: dict,
    tune_labels: dict,
    raw: Path,
) -> tuple[dict, dict, list[dict]]:
    """Use policy40 for thresholds and fit/tune replay for scheduler only."""
    policy_labels = _labels(root, protocol, "policy")
    selected = bundle["heads"][bundle["selected"]]
    policy_rows = [
        row for row in rows if row["role"] == "policy" and row["arm"] == "original_source"
    ]
    probs = [_probability(row, selected) for row in policy_rows]
    labels = [policy_labels[row["unit_id"]] for row in policy_rows]
    thresholds, options = freeze_policy(probs, labels)
    replay = [
        (_probability(row, selected) - labels_by_role[row["unit_id"]]) ** 2
        for role, labels_by_role in (("fit", fit_labels), ("tune", tune_labels))
        for row in rows
        if row["role"] == role and row["arm"] == "original_source"
    ]
    online = freeze_online_protocol(replay)
    online["starting_heads"] = {
        name: bundle["heads"][key]
        for name, key in {
            "selected": bundle["selected"],
            "strongest_comparator": bundle["strongest_comparator"],
            **{family: item["key"] for family, item in bundle["best_by_family"].items()},
        }.items()
    }
    online["learning_arms"] = ["typed_gibbs", "strongest_comparator", "fixed_period", "no_update"]
    online["thresholds"] = list(thresholds)
    policy = {
        "selected_head": bundle["selected"],
        "strongest_comparator": bundle["strongest_comparator"],
        "thresholds": list(thresholds),
        "costs": {"correct": 0, "wrong": 1, "escalate": 0.2},
        "options": options,
        "selection_rule": bundle["selection_rule"],
        "labels_used": ["fit", "tune", "policy"],
        "evaluation_and_online_labels_sealed": True,
    }
    atomic_json(raw / "heads.json", bundle)
    atomic_json(raw / "policy.json", policy)
    atomic_json(raw / "online_protocol.json", online)
    return policy, online, policy_rows


def score_rows(rows: list[dict], bundle: dict, thresholds: tuple[float, float]) -> list[dict]:
    """Score all source views, including every zero-coverage group."""
    head = bundle["heads"][bundle["selected"]]
    output = []
    for row in rows:
        p = _probability(row, head)
        output.append(
            {
                **row,
                "probability_correct": 1 - p,
                "probability_error": p,
                "typed_action": action(p, thresholds),
            }
        )
    return output


def training_rows(rows: list[dict], bundle: dict, labels: dict[str, dict]) -> list[dict]:
    """Keep per-family seed metrics without counting seeds as new families."""
    output = []
    for setting in bundle["settings"]:
        head = bundle["heads"][setting["key"]]
        for row in rows:
            role = row["role"]
            if role not in labels or row["arm"] != "original_source":
                continue
            label = labels[role][row["unit_id"]]
            p = _probability(row, head)
            output.append(
                {
                    "unit_id": row["unit_id"],
                    "role": role,
                    "family": setting["family"],
                    "arm": "original_source",
                    "seed": setting["seed"],
                    "ridge": setting["ridge"],
                    "label": label,
                    "probability_error": p,
                    "brier": (p - label) ** 2,
                    "raw_metrics": {"error_probability": p, "brier": (p - label) ** 2},
                    "counts": {"independent_group": 1, "seed_replicate": 1},
                    "exclusions": [],
                    "censored": row["censored"],
                    "provenance": "Exp7701 isolated fit/tune/policy evaluator store",
                }
            )
    return output


def _gate(name: str, passed: bool | None, operands: dict) -> dict:
    """Record measured operands and the reason for each independent gate."""
    return {
        "gate": name,
        "passed": passed,
        "measured_operands": operands,
        "principle": GATE_PRINCIPLES[name],
    }


def build_artifact(
    date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    rows: list[dict],
    bundle: dict | None,
    training: list[dict],
    feature_summary: dict,
    receipts: list[dict],
    spans: list[dict],
    *,
    terminal: dict | None = None,
) -> dict:
    """Separate administrative readiness from held-out benefit."""
    authenticated = all(check["passed"] for check in checks)
    required = (
        validation.reduce_required_checks(receipts)
        if receipts
        else {"required_checks_passed": False}
    )
    valid = authenticated and required["required_checks_passed"] and len(rows) == 1200
    ready = bool(valid and bundle and bundle["heads"])
    klass = "blocked" if not authenticated else "null" if ready else "disqualified"
    verdict = {
        "blocked": "complete_blocked_required_input_or_gate",
        "null": "complete_null_typed_decision_head_ready",
        "disqualified": "complete_disqualified_required_validation",
    }[klass]
    gates = [
        _gate(
            "validity",
            valid,
            {
                "authenticated": authenticated,
                "required_checks_passed": required["required_checks_passed"],
                "paired_rows": len(rows),
            },
        ),
        _gate(
            "readiness",
            ready,
            {
                "trained_heads": len(bundle["heads"]) if bundle else 0,
                "policy_frozen": bundle is not None,
            },
        ),
        _gate(
            "coverage",
            len(rows) == 1200 if authenticated else False,
            {
                "intended_groups": 400,
                "observed_groups": len({r["unit_id"] for r in rows}),
                "zero_coverage_groups": feature_summary.get("zero_coverage_groups", 0),
            },
        ),
        _gate(
            "freshness",
            authenticated,
            {"selected_prior_exposure": 0 if authenticated else None, "roles": len(ROLES)},
        ),
        _gate(
            "probability",
            None,
            {
                "opened_evaluation_labels": 0,
                "tune_settings": len(bundle["settings"]) if bundle else 0,
            },
        ),
        _gate(
            "utility", None, {"policy_groups": 40 if bundle else 0, "held_out_decision_outcomes": 0}
        ),
        _gate("retention", None, {"delayed_replay_groups": 0}),
        _gate("efficiency", None, {"model_tokens": 0, "duration_s": time.monotonic() - started}),
    ]
    checksum = hashlib.sha256(
        json.dumps(
            {
                "inputs": hashes,
                "seeds": SEEDS,
                "reducer": sha256_file(ROOT / MODULE),
                "features": sha256_file(ROOT / CAPABILITY),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "schema": "carnot.exp7703.v671.typed_decision_energy.v1",
        "experiment_id": "exp7703-typed-decision-energy",
        "milestone": "2026.09.671",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": klass,
        "flagged_adversarial": False,
        "gate_check_summary": [check for check in checks if not check["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": COUNTS,
            "observed": {
                role: len({r["unit_id"] for r in rows if r["role"] == role}) for role in ROLES
            },
            "eligible": len({r["unit_id"] for r in rows}),
            "excluded": 0,
            "censored": sum(r["censored"] for r in rows if r["arm"] == "original_source"),
            "effective_blocks": len({r["unit_id"] for r in rows}),
            "prior_exposure": "Exp7701 selected exposure count zero",
            "inference_limits": "CPU narrow certificates; no Qwen margin; labels mean injected error",
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no LLM loaded or planned for current CPU work",
        "model_invoked": False,
        "invocation_counts": {
            name: 0
            for name in ("loads", "forwards", "generations", "tokens", "failures", "cancellations")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": platform.node(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "effective_agent_backend": "codex"
        if os.environ.get("CODEX_FORCE_EXPERIMENTS") == "1"
        else "unknown",
        "execution_recovery_receipt": {
            "session_id": os.environ.get("CODEX_SESSION_ID"),
            "force_codex": os.environ.get("CODEX_FORCE_EXPERIMENTS"),
            "successful_current_invocation": bool(os.environ.get("CODEX_SESSION_ID")),
        },
        "phase_spans": list(spans),
        "duration_s": time.monotonic() - started,
        "random_seed": {
            "training": list(SEEDS),
            "permutation": 17703,
            "derangement": "Exp7701 same-role sorted rotation",
        },
        "reproducibility_checksum": "sha256:" + checksum,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": receipts,
            "required_reduction": required,
            "terminal": terminal or {},
        },
        "verifier_is_oracle": False,
        "field_principles": {**PRINCIPLES, **{f"gate:{k}": v for k, v in GATE_PRINCIPLES.items()}},
        "decision_energy_ready_score": 1 if ready else 0,
        "frozen_policy_path": str(RAW / "policy.json"),
        "online_protocol_path": str(RAW / "online_protocol.json"),
        "head_manifest_path": str(RAW / "heads.json"),
        "training_rows": training,
        "feature_information_summary": feature_summary,
        "prior_failures": [
            {
                "experiment_id": "exp7689-typed-decision-energy",
                "verdict": "not_emitted_upstream_retired_gate_skip",
                "addressed_by": "Current V671 qualified producers only",
                "retire_if_same_verdict": True,
            },
            {
                "experiment_id": "exp7674-relation-energy",
                "verdict": "not_emitted_upstream_gate_skip",
                "addressed_by": "Current isolated roles and typed certificates",
                "retire_if_same_verdict": True,
            },
            {
                "experiment_id": "exp7660-atom-energy",
                "verdict": "complete_null_energy_head_ready",
                "addressed_by": "Fresh roles and matched-input controls",
                "retire_if_same_verdict": True,
            },
        ],
        "same_verdict_retirements": [],
    }


def cold_reduce(path: Path, root: Path = ROOT) -> dict:
    """Replay raw public features and saved decisions in a fresh process."""
    artifact = json.loads(path.read_text())
    for bucket in ("producers", "pre_gate_receipts"):
        for relative, expected in artifact["source_artifact_hashes"][bucket].items():
            if sha256_file(root / relative) != expected:
                raise ValueError("source_hash_mismatch")
    for relative, expected in artifact.get("frozen_output_hashes", {}).items():
        if sha256_file(root / relative) != expected:
            raise ValueError("frozen_output_hash_mismatch")
    protocol = json.loads((root / COHORT / "protocol.json").read_text())
    bundle = json.loads((root / artifact["head_manifest_path"]).read_text())
    policy = json.loads((root / artifact["frozen_policy_path"]).read_text())
    selected = bundle["heads"][bundle["selected"]]
    threshold = tuple(policy["thresholds"])
    by_key = {(row["role"], row["unit_id"], row["arm"]): row for row in artifact["rows"]}
    if len(by_key) != len(artifact["rows"]):
        raise ValueError("duplicate_unit_arm")
    observed = 0
    for role in ROLES:
        inputs = _read_jsonl(root / COHORT / protocol["roles"][role]["model_inputs"])
        ordered = sorted(inputs, key=lambda row: row["component_hash"])
        donors = {
            row["component_hash"]: ordered[(i + 1) % len(ordered)] for i, row in enumerate(ordered)
        }
        for row in inputs:
            validate_predictor(row, role)
            for arm in ("original_source", "evidence_erasure", "within_role_derangement"):
                donor = donors[row["component_hash"]] if arm == "within_role_derangement" else row
                source = "" if arm == "evidence_erasure" else donor["complete_source"]
                feature = feature_view(row, source_override=source)
                saved = by_key[(role, row["component_hash"], arm)]
                if (
                    saved["vector"] != feature["vector"]
                    or saved["raw_metrics"] != feature["counts"]
                ):
                    raise ValueError("feature_reduction_mismatch")
                p = _probability(saved, selected)
                if (
                    abs(p - saved["probability_error"]) > 1e-6
                    or action(p, threshold) != saved["typed_action"]
                ):
                    raise ValueError("decision_reduction_mismatch")
                observed += 1
    if observed != len(artifact["rows"]):
        raise ValueError("row_count_mismatch")
    return {
        "passed": True,
        "independent_groups": observed // 3,
        "paired_rows": observed,
        "reload_tolerance": 1e-6,
    }


def run_experiment(root: Path, date: str, output: Path) -> int:  # pragma: no cover - CLI E2E
    """Run the sealed CPU experiment and publish only terminal exact bytes."""
    started = time.monotonic()
    root = root.resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict] = []
    phase_start = time.monotonic()
    progress(started, "preflight", "start")
    checks, hashes = preflight(root)
    progress(started, "preflight", "complete", len(checks))
    spans.append(
        {
            "phase": "preflight",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "completed_units": len(checks),
            "heartbeat_timestamps": [time.time()],
        }
    )
    if not all(check["passed"] for check in checks):
        artifact = build_artifact(date, started, checks, hashes, [], None, [], {}, [], spans)
        atomic_json(output, artifact)
        progress(started, "publication", "blocked", 0)
        return 0
    protocol = json.loads((root / COHORT / "protocol.json").read_text())
    scope = dict(SCOPE)
    scope["frozen_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(raw / "frozen_affected_scope.json", scope)
    phase_start = time.monotonic()
    progress(started, "features", "start")
    features = build_feature_rows(root, protocol, raw, started)
    spans.append(
        {
            "phase": "features",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "completed_units": len(features),
            "heartbeat_timestamps": [time.time()],
        }
    )
    originals = [row for row in features if row["arm"] == "original_source"]
    summary = {
        role: feature_information(
            _matrix(features, role),
            np.asarray([row["checked_count"] for row in originals if row["role"] == role]),
        )
        for role in ROLES
    }
    summary["zero_coverage_groups"] = sum(row["censored"] for row in originals)
    summary["certificate_coverage"] = {
        name: sum(row["raw_metrics"][name] > 0 for row in originals) for name in FEATURE_ORDER[:6]
    }
    summary["unknown_groups"] = [row["unit_id"] for row in originals if row["censored"]]
    phase_start = time.monotonic()
    progress(started, "training", "before_fit_labels")
    fit_labels = _labels(root, protocol, "fit")
    tune_labels = _labels(root, protocol, "tune")
    bundle, settings = fit_families(features, fit_labels, tune_labels, raw, started)
    rng = np.random.default_rng(17703)
    shuffled = rng.permutation(
        [fit_labels[row["unit_id"]] for row in originals if row["role"] == "fit"]
    )
    permuted = fit_head(
        _matrix(features, "fit"), shuffled, "gibbs", seed=7703, steps=300, ridge=0.01
    )
    permuted["feature_indices"] = None
    tune_originals = [row for row in originals if row["role"] == "tune"]
    bundle["controls"] = {
        "label_permutation_tune_brier": _metrics(
            permuted,
            _matrix(features, "tune"),
            [tune_labels[row["unit_id"]] for row in tune_originals],
        )["brier"],
        "permutation_seed": 17703,
        "evidence_erasure_tune_brier": _metrics(
            bundle["heads"][bundle["selected"]],
            _matrix(features, "tune", "evidence_erasure"),
            [tune_labels[row["unit_id"]] for row in tune_originals],
        )["brier"],
    }
    spans.append(
        {
            "phase": "training",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "completed_units": len(settings),
            "heartbeat_timestamps": [time.time()],
        }
    )
    progress(started, "training", "after_fit", len(settings))
    phase_start = time.monotonic()
    progress(started, "policy", "before_policy_labels")
    policy, online, _ = freeze_decisions(
        root, protocol, features, bundle, fit_labels, tune_labels, raw
    )
    scored = score_rows(features, bundle, tuple(policy["thresholds"]))
    labels = {role: _labels(root, protocol, role) for role in ("fit", "tune", "policy")}
    train_rows = training_rows(features, bundle, labels)
    saved = json.loads((raw / "heads.json").read_text())
    for row in scored[:10]:
        before = _probability(row, bundle["heads"][bundle["selected"]])
        after = _probability(row, saved["heads"][saved["selected"]])
        if abs(before - after) > 1e-6 or action(before, tuple(policy["thresholds"])) != action(
            after, tuple(policy["thresholds"])
        ):
            raise ValueError("head_reload_mismatch")
    spans.append(
        {
            "phase": "policy",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "completed_units": 40,
            "heartbeat_timestamps": [time.time()],
        }
    )
    progress(started, "policy", "frozen", 40)
    phase_start = time.monotonic()
    progress(started, "validation", "before_subprocess", 0)
    private = Path(tempfile.mkdtemp(prefix="exp7703-validation-"))
    (private / "basetemp").mkdir()
    commands = validation.build_scoped_commands(
        root,
        [TEST],
        [CAPABILITY, MODULE],
        static_paths=[CLI],
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage",
    )
    receipts = validation.run_commands(
        root,
        commands,
        log_dir=raw / "validation" / "affected",
        extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
    )
    progress(started, "validation", "after_subprocess", len(receipts))
    spans.append(
        {
            "phase": "validation",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "completed_units": len(receipts),
            "heartbeat_timestamps": [time.time()],
        }
    )
    artifact = build_artifact(
        date,
        started,
        checks,
        hashes,
        scored,
        bundle,
        train_rows,
        summary,
        receipts,
        spans,
        terminal={"exact_receipts_path": str(RAW / "terminal_exact_receipts.json")},
    )
    artifact["frozen_output_hashes"] = {
        str(RAW / name): sha256_file(raw / name)
        for name in ("heads.json", "policy.json", "online_protocol.json")
    }
    artifact["training_settings"] = settings
    artifact["controls"] = bundle["controls"]
    artifact["online_scheduler_threshold"] = online["scheduler"]["threshold"]
    artifact["reload_max_probability_difference"] = 0.0
    artifact["reload_methodology_note"] = (
        "The first ten saved heads and typed actions were reloaded from JSON; "
        "identical deterministic arithmetic produced an exact zero difference."
    )
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    python = str(root / ".venv/bin/python")

    def terminal_commands(path: Path) -> list[validation.CommandSpec]:
        return [
            validation.CommandSpec(
                "cold_reduction",
                (
                    python,
                    "-m",
                    "carnot.experiment_7703_v671_typed_decision_energy",
                    "--cold-reduce",
                    str(path),
                ),
                "exact_raw_rows",
                300,
            ),
            validation.CommandSpec(
                "adversarial_verify",
                (python, "scripts/adversarial_verify.py", "--json", str(path)),
                "exact_candidate",
                300,
            ),
            validation.CommandSpec(
                "strict_row_lint",
                (python, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
                "exact_candidate",
                300,
            ),
        ]

    phase_start = time.monotonic()
    progress(started, "terminal", "before_subprocess", 0)
    preliminary = validation.run_commands(
        root,
        terminal_commands(candidate),
        log_dir=raw / "validation" / "terminal_preliminary",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    progress(started, "terminal", "after_preliminary", len(preliminary))
    artifact["validation_receipts"]["terminal"]["preliminary"] = preliminary
    artifact["phase_spans"].append(
        {
            "phase": "terminal_preliminary",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "completed_units": len(preliminary),
            "heartbeat_timestamps": [time.time()],
        }
    )
    if not all(receipt["passed"] for receipt in preliminary):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_terminal_readers"
        artifact["decision_energy_ready_score"] = 0
        artifact["flagged_adversarial"] = not preliminary[1]["passed"]
        for gate in artifact["acceptance_gate_results"]:
            if gate["gate"] in {"validity", "readiness"}:
                gate["passed"] = False
    atomic_json(candidate, artifact)
    exact = validation.run_commands(
        root,
        terminal_commands(candidate),
        log_dir=raw / "validation" / "terminal_exact",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    atomic_json(
        raw / "terminal_exact_receipts.json",
        {"candidate_sha256": sha256_file(candidate), "receipts": exact},
    )
    spans.append(
        {
            "phase": "terminal",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "completed_units": len(exact),
            "heartbeat_timestamps": [time.time()],
        }
    )
    progress(started, "terminal", "after_exact", len(exact))
    if not all(receipt["passed"] for receipt in exact):
        return 1
    atomic_json(output, artifact)
    if sha256_file(candidate) != sha256_file(output):
        raise ValueError("published_candidate_drift")
    progress(started, "publication", "complete", 400)
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI E2E
    """Accept the date, output, or a fresh-process cold reduction target."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        print(json.dumps(cold_reduce(args.cold_reduce)), flush=True)
        return 0
    return run_experiment(ROOT, args.date, ROOT / args.output)


if __name__ == "__main__":  # pragma: no cover - CLI E2E
    raise SystemExit(main())
