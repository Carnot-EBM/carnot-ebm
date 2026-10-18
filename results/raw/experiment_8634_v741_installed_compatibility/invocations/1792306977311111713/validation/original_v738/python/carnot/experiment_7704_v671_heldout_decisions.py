"""Orchestrate the sealed forty-family Exp7704 decision measurement."""

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

from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.heldout_decisions import paired_reduce, score_evaluation

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7704_v671_heldout_decisions")
OUTPUT = Path("results/experiment_7704_v671_heldout_decisions.json")
COHORT = Path("results/raw/experiment_7701_v671_sealed_cohort")
HEAD = Path("results/raw/experiment_7703_v671_typed_decision_energy")
PRIOR = Path("results/experiment_7703_v671_typed_decision_energy.json")
COHORT_RESULT = Path("results/experiment_7701_v671_sealed_cohort.json")
MODULE = "python/carnot/experiment_7704_v671_heldout_decisions.py"
CAPABILITY = "python/carnot/reporting/heldout_decisions.py"
TEST = "tests/python/test_experiment_7704_v671_heldout_decisions.py"
CLI = "scripts/experiments/experiment_7704_v671_heldout_decisions.py"
SEED = 7704
MODEL_SPECS: list[str] = []
SCOPE = {
    "tests": [TEST],
    "changed_modules": [CAPABILITY, MODULE],
    "static_paths": [CLI],
    "specs": ["REQ-REPORT-7704", "REQ-ENERGY-7704"],
    "e2e": ["task_fresh_process_raw_reduction", "task_exact_terminal_readers"],
}
PRINCIPLES = {
    "honest_verdict": "A terminal disposition prevents retries of unchanged external blocks.",
    "verdict_class": "A closed enum carries claim eligibility into downstream readers.",
    "flagged_adversarial": "Disqualified evidence must not pass a downstream readiness gate.",
    "gate_check_summary": "Exact upstream fields and values distinguish a false gate from missing evidence.",
    "acceptance_gate_results": "Measured operands limit the claim to checked work.",
    "rows": "Original unit and arm rows let another reader recompute every comparison.",
    "sample_size_budget": "Transformed views and seeds never enlarge independent n.",
    "inference_substrate": "The declared path must match actual current work.",
    "inference_substrate_class": "Duration floors describe actual generation or no-generation work.",
    "MODEL_SPECS": "Experimental models match actual invocations, not the coding agent.",
    "model_invoked": "Actual invocation counts prevent implied LLM work.",
    "execution_venue": "The host and PID identify current work.",
    "phase_spans": "Monotonic spans and heartbeats bound current work.",
    "random_seed": "Independent replay needs the same declared random inputs.",
    "reproducibility_checksum": "Immutable bytes and reducer code bind replay.",
    "source_artifact_hashes": "Producer, pre-gate and missing custody stay separate.",
    "preconditions_checked": "Inputs and resources are checked before opening labels.",
    "validation_receipts": "Commands, exits and log hashes bind exact validation.",
    "verifier_is_oracle": "Fixture correctness cannot earn oracle-distinct credit.",
    "decision_measurement_complete_score": "All forty dispositions and independent reduction establish completion.",
    "registered_decision_benefit_score": "Both interval bounds and useful coverage are required.",
    "paired_brier_reduction_ci": "Paired family intervals preserve the registered Brier question.",
    "paired_cost_reduction_ci": "Paired family intervals charge every escalation.",
}
GATE_PRINCIPLES = {
    "validity": "Invalid evidence must not propagate.",
    "readiness": "Readiness is administrative, not scientific benefit.",
    "coverage": "Quality thresholds prevent effects from being inferred from plumbing.",
    "freshness": "Selected families must remain unexposed until evaluation.",
    "probability": "A calibrated gain needs a paired held-out interval.",
    "utility": "Useful decisions need cost reduction and non-escalation coverage.",
    "retention": "Retention bounds prevent improvement by forgetting.",
    "efficiency": "Resource benefit needs a comparable measured workload.",
}


def progress(started: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Emit a flushed boundary with elapsed time and completed units."""
    print(
        f"[exp7704] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(name: str, upstream: str, path: str, field: str, expected: Any, observed: Any) -> dict:
    """Keep exact failed operands for a missing or false upstream gate."""
    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def authenticate_inputs(root: Path) -> tuple[list[dict], dict]:
    """Authenticate public roster, head, controls and sealed store bytes first."""
    root = root.resolve()
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []}
    checks = [
        _check(
            "cpu_available",
            "host",
            "/proc/self",
            "cpu_count_positive",
            True,
            (os.cpu_count() or 0) > 0,
        )
    ]
    paths = {
        "exp7703": PRIOR,
        "exp7701": COHORT_RESULT,
        "exp7703-heads": HEAD / "heads.json",
        "exp7703-policy": HEAD / "policy.json",
        "exp7703-online": HEAD / "online_protocol.json",
        "exp7701-protocol": COHORT / "protocol.json",
        "exp7701-evaluation-inputs": COHORT / "evaluation_model_inputs.jsonl",
        "exp7701-evaluation-labels": COHORT / "evaluation_evaluator_store.jsonl",
    }
    for upstream, relative in paths.items():
        present = (root / relative).is_file()
        checks.append(_check("input_exists", upstream, str(relative), "exists", True, present))
        bucket = "pre_gate_receipts" if upstream.endswith("labels") else "producers"
        if present:
            hashes[bucket][str(relative)] = sha256_file(root / relative)
        else:
            hashes["missing_evidence"].append(str(relative))
    if hashes["missing_evidence"]:
        return checks, hashes
    prior = json.loads((root / PRIOR).read_text())
    cohort = json.loads((root / COHORT_RESULT).read_text())
    protocol = json.loads((root / COHORT / "protocol.json").read_text())
    heads = json.loads((root / HEAD / "heads.json").read_text())
    policy = json.loads((root / HEAD / "policy.json").read_text())
    online = json.loads((root / HEAD / "online_protocol.json").read_text())
    for upstream, relative, value, fields in (
        (
            "exp7703",
            PRIOR,
            prior,
            {
                "decision_energy_ready_score": 1,
                "flagged_adversarial": False,
                "verdict_class": "null",
            },
        ),
        (
            "exp7701",
            COHORT_RESULT,
            cohort,
            {"cohort_ready_score": 1, "fresh_source_score": 1, "flagged_adversarial": False},
        ),
    ):
        for field, expected in fields.items():
            checks.append(
                _check("upstream_gate", upstream, str(relative), field, expected, value.get(field))
            )
    for name in ("heads.json", "policy.json", "online_protocol.json"):
        relative = HEAD / name
        checks.append(
            _check(
                "frozen_head_hash",
                "exp7703",
                str(relative),
                "sha256",
                prior.get("frozen_output_hashes", {}).get(str(relative)),
                sha256_file(root / relative),
            )
        )
    evaluation = protocol.get("roles", {}).get("evaluation", {})
    store = protocol.get("evaluator_stores", {}).get("evaluation", {})
    for relative, expected, upstream, field in (
        (
            COHORT / "evaluation_model_inputs.jsonl",
            evaluation.get("model_inputs_sha256"),
            "exp7701",
            "model_inputs_sha256",
        ),
        (COHORT / "evaluation_evaluator_store.jsonl", store.get("sha256"), "exp7701", "sha256"),
    ):
        checks.append(
            _check(
                "protocol_hash",
                upstream,
                str(relative),
                field,
                expected,
                sha256_file(root / relative),
            )
        )
    checks.extend(
        (
            _check(
                "evaluation_roster",
                "exp7701",
                str(COHORT / "protocol.json"),
                "roles.evaluation.group_count",
                40,
                evaluation.get("group_count"),
            ),
            _check(
                "evaluation_store_count",
                "exp7701",
                str(COHORT / "protocol.json"),
                "evaluator_stores.evaluation.count",
                40,
                store.get("count"),
            ),
            _check(
                "head_selection",
                "exp7703",
                str(HEAD / "policy.json"),
                "selected_head",
                heads.get("selected"),
                policy.get("selected_head"),
            ),
            _check(
                "strongest_control",
                "exp7703",
                str(HEAD / "policy.json"),
                "strongest_comparator",
                heads.get("strongest_comparator"),
                policy.get("strongest_comparator"),
            ),
            _check(
                "sealed_evaluation",
                "exp7703",
                str(HEAD / "policy.json"),
                "evaluation_and_online_labels_sealed",
                True,
                policy.get("evaluation_and_online_labels_sealed"),
            ),
            _check(
                "online_labels_sealed",
                "exp7703",
                str(HEAD / "online_protocol.json"),
                "evaluation",
                True,
                "evaluation" in online.get("labels_sealed", []),
            ),
            _check(
                "thresholds_frozen",
                "exp7703",
                str(HEAD / "online_protocol.json"),
                "thresholds",
                policy.get("thresholds"),
                online.get("thresholds"),
            ),
        )
    )
    return checks, hashes


def _inputs(root: Path) -> tuple[list[dict], list[dict], dict, dict, list[str], str]:
    """Open labels only after authenticate_inputs returns no failed check."""
    protocol = json.loads((root / COHORT / "protocol.json").read_text())
    prior = json.loads((root / PRIOR).read_text())
    heads = json.loads((root / HEAD / "heads.json").read_text())
    policy = json.loads((root / HEAD / "policy.json").read_text())
    labels = [
        json.loads(line)
        for line in (root / COHORT / "evaluation_evaluator_store.jsonl").read_text().splitlines()
        if line
    ]
    features = [row for row in prior["rows"] if row["role"] == "evaluation"]
    roster = protocol["roles"]["evaluation"]["families"]
    comparator = next(
        setting["family"]
        for setting in heads["settings"]
        if setting["key"] == heads["strongest_comparator"]
    )
    return features, labels, heads, policy, roster, comparator


def cold_reduce(path: Path, root: Path = ROOT) -> dict:
    """Reopen candidate and immutable bytes in a fresh process and replay all gates."""
    artifact = json.loads(path.read_text())
    for bucket in ("producers", "pre_gate_receipts"):
        for label, expected in artifact["source_artifact_hashes"][bucket].items():
            if sha256_file(root / label) != expected:
                raise ValueError("source_hash_mismatch")
    checks, _ = authenticate_inputs(root)
    if not all(check["passed"] for check in checks):
        raise ValueError("preflight_mismatch")
    features, labels, heads, policy, roster, control = _inputs(root)
    rows = score_evaluation(features, labels, heads, policy, roster)
    reduction = paired_reduce(rows, roster, control, seed=SEED)
    if rows != artifact["rows"] or reduction != artifact["independent_reduction"]:
        raise ValueError("raw_reduction_mismatch")
    if artifact["paired_brier_reduction_ci"] != reduction["paired_brier_reduction_ci"]:
        raise ValueError("brier_interval_mismatch")
    if artifact["paired_cost_reduction_ci"] != reduction["paired_cost_reduction_ci"]:
        raise ValueError("cost_interval_mismatch")
    if (
        artifact["registered_decision_benefit_score"]
        != reduction["registered_decision_benefit_score"]
    ):
        raise ValueError("benefit_gate_mismatch")
    required = validation.reduce_required_checks(
        artifact["validation_receipts"]["required_commands"]
    )
    valid = required["required_checks_passed"] and len(rows) == 600
    expected_gates = {
        "validity": valid,
        "readiness": valid,
        "coverage": True,
        "freshness": True,
        "probability": valid and reduction["paired_brier_reduction_ci"]["lower"] > 0.01,
        "utility": valid
        and reduction["paired_cost_reduction_ci"]["lower"] > 0.01
        and reduction["non_escalation_coverage"] >= 0.20,
        "retention": None,
        "efficiency": None,
    }
    observed_gates = {gate["gate"]: gate["passed"] for gate in artifact["acceptance_gate_results"]}
    if observed_gates != expected_gates or artifact["decision_measurement_complete_score"] != int(
        valid
    ):
        raise ValueError("acceptance_gate_mismatch")
    return {"passed": True, "families": len(roster), "rows": len(rows), "all_gates_replayed": True}


def _artifact(
    date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    rows: list[dict],
    reduced: dict,
    receipts: list[dict],
    spans: list[dict],
    control: str | None,
) -> dict:
    """Keep administrative validity and registered scientific benefit separate."""
    authenticated = all(check["passed"] for check in checks)
    required = (
        validation.reduce_required_checks(receipts)
        if receipts
        else {"required_checks_passed": False}
    )
    complete = len(rows) == 40 * 5 * 3 and reduced.get("decision_measurement_complete_score") == 1
    valid = authenticated and complete and required["required_checks_passed"]
    klass = "blocked" if not authenticated else "null" if valid else "disqualified"
    if valid and reduced["registered_decision_benefit_score"]:
        klass = "positive"
    verdict = {
        "blocked": "complete_blocked_required_upstream_input_or_gate",
        "disqualified": "complete_disqualified_required_validation",
        "null": "complete_null_no_registered_decision_benefit",
        "positive": "complete_positive_registered_decision_benefit",
    }[klass]
    brier = reduced.get("paired_brier_reduction_ci")
    cost = reduced.get("paired_cost_reduction_ci")
    coverage = reduced.get("non_escalation_coverage")

    def gate(name: str, passed: bool | None, operands: dict) -> dict:
        return {
            "gate": name,
            "passed": passed,
            "measured_operands": operands,
            "principle": GATE_PRINCIPLES[name],
        }

    gates = [
        gate(
            "validity",
            valid,
            {
                "authenticated": authenticated,
                "required_checks_passed": required["required_checks_passed"],
                "row_count": len(rows),
            },
        ),
        gate(
            "readiness",
            valid,
            {
                "complete_families": reduced.get("effective_blocks", 0),
                "independent_reduction": bool(reduced),
            },
        ),
        gate(
            "coverage",
            complete,
            {
                "observed_families": reduced.get("effective_blocks", 0),
                "non_escalation_coverage": coverage,
                "zero_checked_families": sum(
                    r["checked_count"] == 0
                    for r in rows
                    if r["arm"] == "typed_gibbs:original_source"
                ),
            },
        ),
        gate(
            "freshness",
            authenticated,
            {
                "official_test_families": 40 if authenticated else 0,
                "prior_selected_exposure": 0 if authenticated else None,
            },
        ),
        gate(
            "probability",
            bool(valid and brier and brier["lower"] > 0.01),
            {"paired_brier_reduction_ci": brier, "required_lower": 0.01},
        ),
        gate(
            "utility",
            bool(
                valid
                and cost
                and cost["lower"] > 0.01
                and coverage is not None
                and coverage >= 0.20
            ),
            {
                "paired_cost_reduction_ci": cost,
                "non_escalation_coverage": coverage,
                "required_lower": 0.01,
                "required_coverage": 0.20,
            },
        ),
        gate("retention", None, {"delayed_replay_groups": 0}),
        gate(
            "efficiency",
            None,
            {"current_model_tokens": 0, "duration_s": time.monotonic() - started},
        ),
    ]
    checksum = hashlib.sha256(
        json.dumps(
            {
                "inputs": hashes,
                "seed": SEED,
                "reducer": sha256_file(ROOT / CAPABILITY),
                "orchestrator": sha256_file(ROOT / MODULE),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "schema": "carnot.exp7704.v671.heldout_decisions.v1",
        "experiment_id": "exp7704-v671-heldout-decisions",
        "milestone": "2026.09.671",
        "run_date": date,
        "label_definition": "annotated injected-error presence, not general natural hallucination truth",
        "honest_verdict": verdict,
        "verdict_class": klass,
        "flagged_adversarial": False,
        "gate_check_summary": [check for check in checks if not check["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": 40,
            "observed": reduced.get("effective_blocks", 0),
            "eligible": reduced.get("effective_blocks", 0),
            "excluded": 0,
            "censored": sum(
                r["censored"] for r in rows if r["arm"] == "typed_gibbs:original_source"
            ),
            "effective_blocks": reduced.get("effective_blocks", 0),
            "prior_exposure": 0 if authenticated else None,
            "inference_limits": "Source-specific LettuceDetect injected-error pilot; no general natural hallucination or FoVer headline inference",
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no model loaded or planned for this CPU evaluation",
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
        "random_seed": {"paired_family_bootstrap": SEED, "draws": 10_000},
        "reproducibility_checksum": "sha256:" + checksum,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": receipts,
            "required_reduction": required,
            "terminal": {"exact_receipts_path": str(RAW / "terminal_exact_receipts.json")},
        },
        "repository_health": {
            "full_python_suite": next(
                (
                    {
                        "passed": receipt["passed"],
                        "exit_code": receipt["exit_code"],
                        "log_path": receipt["log_path"],
                        "log_sha256": receipt["log_sha256"],
                    }
                    for receipt in receipts
                    if receipt["name"] == "full_python_suite"
                ),
                None,
            ),
            "affects_required_scope": False,
        },
        "verifier_is_oracle": False,
        "field_principles": {
            **PRINCIPLES,
            **{f"gate:{key}": value for key, value in GATE_PRINCIPLES.items()},
        },
        "decision_measurement_complete_score": int(valid),
        "registered_decision_benefit_score": int(
            valid and bool(reduced.get("registered_decision_benefit_score"))
        ),
        "paired_brier_reduction_ci": brier,
        "paired_cost_reduction_ci": cost,
        "independent_reduction": reduced,
        "strongest_tune_non_energy_control": control,
        "energy_form_contrast": reduced.get("energy_form_contrast"),
        "feature_information_contrast": reduced.get("feature_information_contrast"),
        "evidence_interventions": reduced.get("evidence_interventions"),
        "headline_scope": "FoVer headline unchanged; this is a source-specific injected-error pilot",
        "prior_failures": [
            {
                "experiment_id": "exp7690-heldout-decisions",
                "custody": "not_emitted_upstream_retired_gate_skip",
                "scientific_verdict": None,
            },
            {
                "experiment_id": "exp7675-fresh-decision-evaluation",
                "custody": "not_emitted_upstream_gate_skip",
                "scientific_verdict": None,
            },
            {
                "experiment_id": "exp7661-decision-evaluation",
                "custody": "emitted",
                "scientific_verdict": "complete_null_no_registered_decision_benefit",
            },
        ],
        "same_verdict_retirements": [
            {
                "prior_experiment_id": "exp7661-decision-evaluation",
                "mechanism": "frozen source-specific atom/typed-certificate energy decision head on LettuceDetect injected-error families",
                "reason": "Second registered held-out utility null; retire this narrow mechanism, not the broader energy research direction",
            }
        ]
        if klass == "null"
        else [],
    }


def run_experiment(root: Path, date: str, output: Path) -> int:  # pragma: no cover - CLI E2E
    """Freeze inputs, measure once, validate, cold replay and publish exact bytes."""
    root = root.resolve()
    started = time.monotonic()
    progress(started, "preflight", "start")
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "frozen_affected_scope.json", SCOPE)
    checks, hashes = authenticate_inputs(root)
    spans: list[dict] = []
    progress(started, "preflight", "complete", len(checks))
    if not all(check["passed"] for check in checks):
        artifact = _artifact(date, started, checks, hashes, [], {}, [], spans, None)
        atomic_json(output, artifact)
        progress(started, "publication", "blocked", 0)
        return 0
    phase_start = time.monotonic()
    progress(started, "measurement", "before_label_open")
    features, labels, heads, policy, roster, control = _inputs(root)
    rows = score_evaluation(features, labels, heads, policy, roster)
    reduced = paired_reduce(rows, roster, control, seed=SEED)
    spans.append(
        {
            "phase": "measurement",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "completed_units": len(roster),
            "heartbeat_timestamps": [time.time()],
            "checkpoint": str(raw / "measurement_checkpoint.json"),
        }
    )
    atomic_json(
        raw / "measurement_checkpoint.json", {"rows": rows, "independent_reduction": reduced}
    )
    progress(started, "measurement", "complete", len(roster))
    private = Path(tempfile.mkdtemp(prefix="exp7704-validation-", dir="/tmp"))
    (private / "basetemp").mkdir()
    commands = validation.build_scoped_commands(
        root,
        SCOPE["tests"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage",
    )
    progress(started, "validation", "before_subprocess")
    phase_start = time.monotonic()
    receipts = validation.run_commands(
        root,
        commands,
        log_dir=raw / "validation" / "affected",
        extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
    )
    full = validation.CommandSpec(
        "full_python_suite",
        (
            str(root / ".venv/bin/pytest"),
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={private / 'basetemp' / 'full'}",
        ),
        "repository_health",
        1200,
    )
    receipts.extend(
        validation.run_commands(
            root, [full], log_dir=raw / "validation" / "full", extra_env={"JAX_PLATFORMS": "cpu"}
        )
    )
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
    progress(started, "validation", "after_subprocess", len(receipts))
    artifact = _artifact(date, started, checks, hashes, rows, reduced, receipts, spans, control)
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    python = str(root / ".venv/bin/python")
    terminal = [
        validation.CommandSpec(
            "cold_reduction",
            (
                python,
                "-m",
                "carnot.experiment_7704_v671_heldout_decisions",
                "--cold-reduce",
                str(candidate),
            ),
            "exact_raw_rows",
            300,
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_candidate",
            300,
        ),
        validation.CommandSpec(
            "strict_row_lint",
            (python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            300,
        ),
    ]
    progress(started, "terminal", "before_subprocess")
    phase_start = time.monotonic()
    preliminary = validation.run_commands(
        root,
        terminal,
        log_dir=raw / "validation" / "terminal_preliminary",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    artifact["validation_receipts"]["terminal"]["preliminary"] = preliminary
    spans.append(
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
        artifact["flagged_adversarial"] = not preliminary[1]["passed"]
        artifact["decision_measurement_complete_score"] = 0
        artifact["registered_decision_benefit_score"] = 0
        for gate in artifact["acceptance_gate_results"]:
            if gate["gate"] in {"validity", "readiness"}:
                gate["passed"] = False
    atomic_json(candidate, artifact)
    exact = validation.run_commands(
        root,
        terminal,
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
    progress(started, "terminal", "after_subprocess", len(exact))
    if not all(receipt["passed"] for receipt in exact):
        return 1
    atomic_json(output, artifact)
    if sha256_file(output) != sha256_file(candidate):
        raise ValueError("published_candidate_drift")
    progress(started, "publication", "complete", len(roster))
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI E2E
    """Accept the declared run path or a fresh-process cold reducer target."""
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
