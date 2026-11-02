"""Publish the V668 continuous source-atom measurement (REQ-REPORT-7663)."""

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

from carnot.experiment_7662_v668_delayed_update_protocol import (
    _online_inputs,
    authenticate as authenticate_7662,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)
from carnot.reporting.experiment_7646_source_features import read_jsonl
from carnot.reporting.experiment_7660_atom_energy import score
from carnot.reporting.experiment_7662_delayed_protocol import _probability, source_stratum
from carnot.reporting.experiment_7663_continuous_learning import (
    decision,
    paired_block_interval,
    replay_arm,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7663_v668_continuous_atom_learning")
OUTPUT = Path("results/experiment_7663_v668_continuous_atom_learning.json")
MODULE = "python/carnot/reporting/experiment_7663_continuous_learning.py"
ORCHESTRATION = "python/carnot/experiment_7663_v668_continuous_atom_learning.py"
WRAPPER = "scripts/experiments/experiment_7663_v668_continuous_atom_learning.py"
TEST = "tests/python/test_experiment_7663_v668_continuous_atom_learning.py"
PRIOR = Path("results/experiment_7662_v668_delayed_update_protocol.json")
ARMS = ("source", "scalar", "frozen", "permuted", "omission")
MODEL_SPECS: list[str] = []


def _check(check: str, upstream: str, path: str, field: str, expected: Any, observed: Any) -> dict:
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def authenticate(root: Path) -> tuple[dict, dict, dict, list[dict], dict]:
    """Bind all exposed input bytes and resource checks before opening labels."""
    protocol, manifest, heads, _prior, checks, hashes = authenticate_7662(root)
    checks.append(
        _check(
            "disk_available",
            "host",
            str(root),
            "free_bytes_at_least_10m",
            True,
            os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 10_000_000,
        )
    )
    for relative, upstream, bucket in ((PRIOR, "Exp7662", "pre_gate_receipts"),):
        path = root / relative
        checks.append(
            _check("input_exists", upstream, str(relative), "exists", True, path.is_file())
        )
        if path.is_file():
            hashes[bucket][str(relative)] = sha256_file(path)
            prior = json.loads(path.read_text())
            checks.append(
                _check(
                    "upstream_gate",
                    upstream,
                    str(relative),
                    "delayed_protocol_ready_score",
                    1,
                    prior.get("delayed_protocol_ready_score"),
                )
            )
            checks.append(
                _check(
                    "upstream_flag",
                    upstream,
                    str(relative),
                    "flagged_adversarial",
                    False,
                    prior.get("flagged_adversarial"),
                )
            )
        else:
            hashes["missing_inputs"].append(str(relative))
    if not protocol or not manifest:
        return {}, {}, {}, checks, hashes
    for role in ("fit", "evaluation"):
        info = manifest["roles"][role]
        label = protocol["reader_sidecars"]["evaluator_stores"][role]
        for relative, expected, upstream, bucket in (
            (info["feature_path"], info["feature_sha256"], "Exp7659", "producers"),
            (label["path"], label["sha256"], "Exp7602", "pre_gate_receipts"),
        ):
            path = root / relative
            observed = sha256_file(path) if path.is_file() else None
            checks.append(_check("input_hash", upstream, relative, "sha256", expected, observed))
            if observed is None:
                hashes["missing_inputs"].append(relative)
            else:
                hashes[bucket][relative] = observed
    return protocol, manifest, heads, checks, hashes


def anchor_rows(
    root: Path, protocol: dict, manifest: dict, heads: dict, numerical: dict
) -> list[dict]:
    """Score exposed anchors once from immutable rows after the online state freezes."""
    result = []
    head = heads["heads"][heads["selected"]]
    fit_ids = set(protocol["learning_schedule"]["fit_anchor_ids"])
    for role in ("fit", "evaluation"):
        features = read_jsonl(root / manifest["roles"][role]["feature_path"])
        labels = read_jsonl(root / protocol["reader_sidecars"]["evaluator_stores"][role]["path"])
        by_id = {row["component_hash"]: row for row in labels}
        originals = [row for row in features if row["arm"] == "original_source"]
        selected = [row for row in originals if role == "evaluation" or row["unit_id"] in fit_ids]
        if len(selected) != (16 if role == "fit" else 40) or len(by_id) != len(labels):
            raise ValueError("anchor_roster_invalid")
        for feature in selected:
            unit = feature["unit_id"]
            label = by_id[unit]["label"]
            frozen = score(feature, by_id[unit]["raw_probability"], head)
            final = _probability(frozen, source_stratum(feature), numerical)
            result.append(
                {
                    "unit_id": unit,
                    "role": role,
                    "label": label,
                    "frozen_probability": frozen,
                    "final_probability": final,
                    "frozen_brier": (frozen - label) ** 2,
                    "final_brier": (final - label) ** 2,
                    "source_sha256": feature["source_sha256"],
                    "censored": feature["censored"],
                    "excluded": feature["excluded"],
                }
            )
    return result


def reduce_evidence(rows: list[dict], anchors: list[dict]) -> dict:
    """Cold arithmetic checks raw operands and resamples independent groups."""
    grouped = {arm: [] for arm in ARMS}
    for row in rows:
        if row["arm"] not in grouped or row["label"] not in (0, 1):
            raise ValueError("row_custody_invalid")
        action, cost = decision(row["probability"], row["label"])
        if (
            abs(row["brier"] - (row["probability"] - row["label"]) ** 2) > 1e-12
            or row["typed_action"] != action
            or abs(row["decision_cost"] - cost) > 1e-12
        ):
            raise ValueError("raw_metric_mismatch")
        if row["label_release_ordinal"] < row["origin_ordinal"] + 8:
            raise ValueError("future_feedback")
        grouped[row["arm"]].append(row)
    ids = [row["unit_id"] for row in grouped["source"]]
    if (
        len(ids) != 80
        or len(set(ids)) != 80
        or any([row["unit_id"] for row in grouped[arm]] != ids for arm in ARMS)
    ):
        raise ValueError("paired_roster_invalid")
    metrics = {
        arm: {
            "brier": sum(r["brier"] for r in grouped[arm]) / 80,
            "decision_cost": sum(r["decision_cost"] for r in grouped[arm]) / 80,
            "coverage": sum(r["typed_action"] != "escalate" for r in grouped[arm]) / 80,
            "independent_groups": 80,
        }
        for arm in ARMS
    }
    contrasts = {}
    for control in ("scalar", "frozen"):
        contrasts[control] = {
            metric: paired_block_interval(
                [
                    (a[metric], b[metric])
                    for a, b in zip(grouped["source"], grouped[control], strict=True)
                ],
                7663 + (0 if metric == "brier" else 10) + (0 if control == "scalar" else 1),
            )
            for metric in ("brier", "decision_cost")
        }
    retention = {}
    for role, count in (("fit", 16), ("evaluation", 40)):
        group = [r for r in anchors if r["role"] == role]
        if len(group) != count or len({r["unit_id"] for r in group}) != count:
            raise ValueError("retention_roster_invalid")
        for row in group:
            if (
                abs(row["frozen_brier"] - (row["frozen_probability"] - row["label"]) ** 2) > 1e-12
                or abs(row["final_brier"] - (row["final_probability"] - row["label"]) ** 2) > 1e-12
            ):
                raise ValueError("retention_metric_mismatch")
        retention[role] = paired_block_interval(
            [(r["frozen_brier"], r["final_brier"]) for r in group], 7664 + count
        )
    return {
        "prequential": metrics,
        "paired_contrasts": contrasts,
        "retention_degradation": retention,
    }


def _gate(name: str, passed: bool, operands: dict, principle: str) -> dict:
    return {"gate": name, "passed": passed, "measured_operands": operands, "principle": principle}


def build_artifact(
    date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    replays: dict,
    anchors: list[dict],
    receipts: list[dict],
    scope: dict,
    spans: list[dict],
) -> dict:
    """Separate validation, measurement, and scientific benefit judgments."""
    rows = [row for arm in ARMS for row in replays.get(arm, {}).get("rows", [])]
    causal = [row for arm in ARMS for row in replays.get(arm, {}).get("causal_feedback_rows", [])]
    decisions = [row for arm in ARMS for row in replays.get(arm, {}).get("admission_decisions", [])]
    authenticated = all(check["passed"] for check in checks)
    required = reduce_required_checks(receipts)
    reduction = reduce_evidence(rows, anchors) if len(rows) == 400 else {}
    parity = bool(
        replays
        and [r["probability"] for r in replays["source"]["rows"]]
        == [r["probability"] for r in replays["source_restart"]["rows"]]
        and replays["source"]["numerical_hash"] == replays["source_restart"]["numerical_hash"]
    )
    measured = len(rows) == 400 and len(anchors) == 56 and len(causal) == 400 and parity
    valid = authenticated and measured and required["required_checks_passed"]
    contrasts = reduction.get("paired_contrasts", {})
    retained = reduction.get("retention_degradation", {})
    probability = valid and all(
        contrasts[arm]["brier"]["lower_ci95"] > 0.01 for arm in ("frozen", "scalar")
    )
    utility = (
        valid
        and all(
            contrasts[arm]["decision_cost"]["lower_ci95"] > 0.01 for arm in ("frozen", "scalar")
        )
        and reduction["prequential"]["source"]["coverage"] >= 0.20
    )
    retention = valid and all(
        retained[role]["upper_ci95"] <= 0.01 for role in ("fit", "evaluation")
    )
    klass = (
        "blocked"
        if not authenticated
        else "disqualified"
        if not valid
        else "positive"
        if probability and retention
        else "null"
    )
    verdict = {
        "blocked": "complete_blocked_missing_external_evidence",
        "disqualified": "complete_disqualified_required_validation",
        "positive": "complete_positive_exposed_continuous_learning",
        "null": "complete_null_continuous_learning_no_registered_benefit",
    }[klass]
    gates = [
        _gate(
            "validity",
            valid,
            {
                "authenticated": authenticated,
                "required_checks_passed": required["required_checks_passed"],
            },
            "Immutable bytes and required validation must pass.",
        ),
        _gate(
            "readiness",
            valid and measured,
            {"event_rows": len(causal), "restart_parity": parity},
            "Causal replay, durable restart, and anchors must complete.",
        ),
        _gate(
            "coverage",
            valid and len(rows) == 400,
            {
                "independent_groups": len(rows) // 5,
                "paired_rows": len(rows),
                "checked_source_groups": sum(
                    r["raw_metrics"]["checked_structural_propositions"] > 0
                    for r in replays.get("source", {}).get("rows", [])
                ),
            },
            "Each original group owns one denominator; unknowns remain.",
        ),
        _gate(
            "probability_benefit",
            probability,
            {
                "paired_block_ci95": {
                    arm: contrasts.get(arm, {}).get("brier") for arm in ("frozen", "scalar")
                },
                "lower_bound_threshold": 0.01,
            },
            "Both block-eight lower Brier improvement bounds must exceed 0.01.",
        ),
        _gate(
            "utility",
            utility,
            {
                "paired_block_ci95": {
                    arm: contrasts.get(arm, {}).get("decision_cost") for arm in ("frozen", "scalar")
                },
                "coverage": reduction.get("prequential", {}).get("source", {}).get("coverage"),
                "threshold": 0.01,
                "coverage_floor": 0.20,
            },
            "Cost improvement has a separate paired gate and coverage floor.",
        ),
        _gate(
            "retention",
            retention,
            {
                "upper_degradation_ci95": {
                    role: retained.get(role) for role in ("fit", "evaluation")
                },
                "ceiling": 0.01,
            },
            "Both exposed anchor sets must bound forgetting.",
        ),
        _gate(
            "freshness",
            False,
            {"fresh_future_groups": 0, "previously_exposed": 80},
            "Exposed online and anchor groups do not confirm future benefit.",
        ),
    ]
    checksum = hashlib.sha256(
        json.dumps(
            {"inputs": hashes, "seed": 7663, "reducer": sha256_file(ROOT / MODULE)}, sort_keys=True
        ).encode()
    ).hexdigest()
    return {
        "schema": "carnot.exp7663.v668.continuous_atom_learning.v1",
        "experiment_id": "exp7663-v668-continuous-atom-learning",
        "milestone": "2026.09.668",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": klass,
        "flagged_adversarial": False,
        "gate_check_summary": [check for check in checks if not check["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "causal_feedback_rows": causal,
        "admission_decisions": decisions,
        "retention_rows": anchors,
        "independent_reduction": reduction,
        "sample_size_budget": {
            "intended": 80,
            "observed": len(rows) // 5,
            "eligible": len(rows) // 5,
            "excluded": sum(r["excluded"] for r in replays.get("source", {}).get("rows", [])),
            "censored": sum(r["censored"] for r in replays.get("source", {}).get("rows", [])),
            "effective_bootstrap_blocks": 10 if rows else 0,
            "prior_exposure": "online80, evaluation40 and fit16 previously exposed",
            "claim_limit": "exploratory; no broad significance or fresh future claim",
        },
        "inference_substrate": "cpu_continuous_source_atom_update_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "No model loaded or planned in current CPU source work.",
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in ("loads", "forwards", "generations", "tokens", "attempted", "cancelled")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": platform.node(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - started,
        "random_seed": {
            "bootstrap": 7663,
            "permutation": 7663,
            "purpose": "fixed block resampling and eligible-past control",
        },
        "reproducibility_checksum": "sha256:" + checksum,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": scope,
            "required_commands": receipts,
            **required,
            "unrelated_repository_suite_debt": [],
        },
        "verifier_is_oracle": False,
        "field_principles": {
            "honest_verdict": "Completion differs from scientific benefit.",
            "verdict_class": "Invalid receipts disqualify; missing external inputs block; causal null is terminal.",
            "flagged_adversarial": "A flagged reader cannot open a gate.",
            "rows": "One independent group per arm; repeated views do not enlarge N.",
            "causal_feedback_rows": "Prediction precedes release, and only past eligible feedback trains.",
            "retention_rows": "Exposed anchors are read only after the final state freezes.",
            "sample_size_budget": "Ten blocks of eight are a small effective sample.",
            "duration_s": "Actual CPU wall time has no padding.",
            "inference_substrate": "Current cached CPU work has no model invocation.",
            "inference_substrate_class": "No model load or generation occurred.",
            "MODEL_SPECS": "No current LLM task exists; historical provenance is separate.",
            "source_artifact_hashes": "Immutable upstream bytes bind replay, not planned output bytes.",
            "validation_receipts": "Frozen affected scope and exact-candidate readers govern readiness.",
            "verifier_is_oracle": "Dataset evaluator labels are isolated, not exact fixture truth.",
            "acceptance_gate_results": "Each scientific claim has an independent measured gate.",
        },
        "historical_model_id": "unsloth/Qwen3.8-27B-GGUF",
        "continuous_measurement_complete_score": int(valid),
        "continuous_benefit_score": int(probability and retention),
        "utility_benefit_score": int(utility),
        "restart_exact_parity": parity,
        "state_summaries": {
            arm: {
                key: value
                for key, value in replay.items()
                if key in {"numerical_hash", "state_hash", "state_bytes", "wall_time_per_event_s"}
            }
            for arm, replay in replays.items()
        },
        "updates_accepted": sum(d["accepted"] for d in decisions if d["arm"] == "source"),
        "updates_rejected": sum(not d["accepted"] for d in decisions if d["arm"] == "source"),
        "retirement": "Retire the unchanged V667 zero-check source grammar after its same null; absence of external resources is not scientific disproof.",
    }


def cold_reduce(path: Path) -> dict:
    """Reload exact candidate bytes and recompute exposed-group arithmetic."""
    artifact = json.loads(path.read_text())
    for bucket in ("producers", "pre_gate_receipts"):
        for relative, digest in artifact["source_artifact_hashes"][bucket].items():
            if sha256_file(ROOT / relative) != digest:
                raise ValueError("input_hash_mismatch")
    reduction = reduce_evidence(artifact["rows"], artifact["retention_rows"])
    if reduction != artifact["independent_reduction"]:
        raise ValueError("independent_reduction_mismatch")
    return {"passed": True, "independent_groups": 80, "paired_rows": 400, "anchor_groups": 56}


def run_experiment(root: Path, date: str, output: Path) -> dict:
    """Authenticate, measure, validate, and atomically publish one terminal row."""
    root = root.resolve()
    started = time.monotonic()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict] = []

    def progress(phase: str, event: str, detail: str = "") -> None:
        print(
            f"[exp7663] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {detail}",
            flush=True,
        )

    def close(phase: str, begin: float, units: int, checkpoint: Path) -> None:
        end = time.monotonic() - started
        spans.append(
            {
                "phase": phase,
                "start_offset_s": begin,
                "end_offset_s": end,
                "duration_s": end - begin,
                "completed_units": units,
                "heartbeat_times_s": [begin, end],
                "checkpoint": str(checkpoint.relative_to(root)),
            }
        )
        progress(phase, "complete", f"units={units}")

    progress("preconditions", "start", f"root={root}")
    begin = time.monotonic() - started
    protocol, manifest, heads, checks, hashes = authenticate(root)
    atomic_json(raw / "preconditions.json", {"checks": checks, "hashes": hashes})
    close("preconditions", begin, len(checks), raw / "preconditions.json")
    with tempfile.TemporaryDirectory(prefix="exp7663-", dir="/tmp") as private_dir:
        private = Path(private_dir)
        basetemp = private / "basetemp"
        basetemp.mkdir()
        commands = build_scoped_commands(
            root,
            [TEST],
            [MODULE],
            static_paths=[ORCHESTRATION, WRAPPER],
            basetemp=basetemp,
            coverage_file=private / ".coverage",
        )
        commands.append(
            CommandSpec(
                "orchestration_mypy",
                (str(root / ".venv/bin/mypy"), ORCHESTRATION),
                "changed_orchestration",
            )
        )
        scope = {
            "tests": [TEST],
            "changed_modules": [MODULE],
            "static_paths": [ORCHESTRATION, WRAPPER],
            "commands": [{"name": c.name, "argv": list(c.argv)} for c in commands],
        }
        atomic_json(raw / "frozen_validation_manifest.json", scope)
        progress("measurement", "start")
        begin = time.monotonic() - started
        replays: dict = {}
        anchors: list[dict] = []
        if all(check["passed"] for check in checks):
            inputs = _online_inputs(root, protocol, manifest)
            for arm in (*ARMS, "source_restart"):
                progress("measurement", "arm_start", arm)
                replays[arm] = replay_arm(raw, inputs, heads, arm, restart=arm == "source_restart")
                progress("measurement", "arm_complete", f"{arm} rows={len(replays[arm]['rows'])}")
            anchors = anchor_rows(
                root, protocol, manifest, heads, replays["source"]["numerical_state"]
            )
            atomic_json(
                raw / "measurement_checkpoint.json",
                {
                    "completed_units": 80,
                    "restart_exact_parity": replays["source"]["numerical_hash"]
                    == replays["source_restart"]["numerical_hash"],
                },
            )
        else:
            atomic_json(
                raw / "measurement_checkpoint.json",
                {
                    "completed_units": 0,
                    "blocked_checks": [check for check in checks if not check["passed"]],
                },
            )
        close("measurement", begin, 80 if replays else 0, raw / "measurement_checkpoint.json")
        progress("affected_validation", "before_subprocess")
        begin = time.monotonic() - started
        receipts = run_commands(
            root,
            commands,
            log_dir=raw / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        close("affected_validation", begin, len(receipts), raw / "validation/affected")
        artifact = build_artifact(
            date, started, checks, hashes, replays, anchors, receipts, scope, spans
        )
        candidate = raw / "exact_terminal_candidate.json"
        atomic_json(candidate, artifact)
        if artifact["verdict_class"] == "blocked":
            artifact["validation_receipts"]["terminal_readers"] = []
            artifact["validation_receipts"]["terminal_reader_disposition"] = (
                "blocked_before_evaluation_inputs_opened"
            )
        else:
            python = str(root / ".venv/bin/python")
            readers = [
                CommandSpec(
                    "cold_reduce",
                    (
                        python,
                        "-u",
                        "-m",
                        "carnot.experiment_7663_v668_continuous_atom_learning",
                        "--cold-reduce",
                        str(candidate),
                    ),
                    "exact_candidate",
                ),
                CommandSpec(
                    "adversarial_verify",
                    (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                    "exact_candidate",
                ),
                CommandSpec(
                    "verdict_row_consistency",
                    (
                        python,
                        "-u",
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ),
                    "exact_candidate",
                ),
            ]
            progress("terminal_readers", "before_subprocess")
            begin = time.monotonic() - started
            terminal = run_commands(
                root,
                readers,
                log_dir=raw / "validation/terminal",
                extra_env={"JAX_PLATFORMS": "cpu"},
                heartbeat_s=30,
            )
            close("terminal_readers", begin, len(terminal), raw / "validation/terminal")
            artifact["validation_receipts"]["terminal_readers"] = terminal
            artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = sha256_file(
                candidate
            )
            artifact["flagged_adversarial"] = not terminal[1]["passed"]
            if not all(receipt["passed"] for receipt in terminal):
                artifact["verdict_class"] = "disqualified"
                artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
                artifact["gate_check_summary"].extend(
                    _check(
                        "terminal_reader",
                        "current",
                        receipt["log_path"],
                        "exit_code",
                        0,
                        receipt["exit_code"],
                    )
                    for receipt in terminal
                    if not receipt["passed"]
                )
                for field in (
                    "continuous_measurement_complete_score",
                    "continuous_benefit_score",
                    "utility_benefit_score",
                ):
                    artifact[field] = 0
                for gate in artifact["acceptance_gate_results"]:
                    gate["passed"] = False
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        progress("publish", "before_atomic_write", f"verdict={artifact['honest_verdict']}")
        atomic_json(output, artifact)
        progress("publish", "complete", f"bytes={output.stat().st_size}")
        return artifact


def main(argv: list[str] | None = None) -> int:
    """Accept the declared entrypoint and the read-only cold reduction mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        print(json.dumps(cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    run_experiment(ROOT.resolve(), args.date, ROOT / args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - task E2E
    raise SystemExit(main())
