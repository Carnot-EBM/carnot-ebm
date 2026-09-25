"""Run the V668 delayed source update protocol (REQ-REPORT-7662)."""

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

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)
from carnot.reporting.experiment_7646_source_features import read_jsonl
from carnot.reporting.experiment_7660_atom_energy import score
from carnot.reporting.experiment_7662_delayed_protocol import (
    DelayedUpdateService,
    source_stratum,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7662_v668_delayed_update_protocol")
OUTPUT = Path("results/experiment_7662_v668_delayed_update_protocol.json")
PROTOCOL = Path("results/raw/experiment_7602_v664_evidence_requalification/protocol.json")
MANIFEST = Path("results/raw/experiment_7659_v668_atom_corpus/manifest.json")
HEADS = Path("results/raw/experiment_7660_v668_atom_energy/heads.json")
PRIOR = Path("results/experiment_7660_v668_atom_energy.json")
MODULE = "python/carnot/reporting/experiment_7662_delayed_protocol.py"
ORCHESTRATION = "python/carnot/experiment_7662_v668_delayed_update_protocol.py"
WRAPPER = "scripts/experiments/experiment_7662_v668_delayed_update_protocol.py"
TEST = "tests/python/test_experiment_7662_v668_delayed_update_protocol.py"
MODEL_SPECS: list[str] = []
ARMS = ("source", "scalar", "frozen")


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


def authenticate(root: Path) -> tuple[dict, dict, dict, dict, list[dict], dict]:
    """Bind upstream receipts and online bytes before opening evaluator labels."""

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
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []}
    inputs = (
        (PROTOCOL, "Exp7602", "producers"),
        (MANIFEST, "Exp7659", "producers"),
        (HEADS, "Exp7660", "producers"),
        (PRIOR, "Exp7660", "pre_gate_receipts"),
    )
    for label, upstream, bucket in inputs:
        path = root / label
        checks.append(
            _check("input_exists", upstream, label.as_posix(), "exists", True, path.is_file())
        )
        if path.is_file():
            hashes[bucket][label.as_posix()] = sha256_file(path)
        else:
            hashes["missing_inputs"].append(label.as_posix())
    if hashes["missing_inputs"]:
        return {}, {}, {}, {}, checks, hashes
    protocol = json.loads((root / PROTOCOL).read_text())
    manifest = json.loads((root / MANIFEST).read_text())
    heads = json.loads((root / HEADS).read_text())
    prior = json.loads((root / PRIOR).read_text())
    for field, expected in (("energy_ready_score", 1), ("flagged_adversarial", False)):
        checks.append(
            _check("upstream_gate", "Exp7660", PRIOR.as_posix(), field, expected, prior.get(field))
        )
    checks.append(
        _check(
            "online80",
            "Exp7602",
            PROTOCOL.as_posix(),
            "role_counts.online",
            80,
            protocol.get("role_counts", {}).get("online"),
        )
    )
    checks.append(
        _check(
            "delay",
            "Exp7602",
            PROTOCOL.as_posix(),
            "delayed_feedback_lag",
            8,
            protocol.get("delayed_feedback_lag"),
        )
    )
    feature_info = manifest["roles"]["online"]
    label_info = protocol["reader_sidecars"]["evaluator_stores"]["online"]
    for label, expected, upstream, bucket in (
        (feature_info["feature_path"], feature_info["feature_sha256"], "Exp7659", "producers"),
        (label_info["path"], label_info["sha256"], "Exp7602", "pre_gate_receipts"),
    ):
        observed = sha256_file(root / label) if (root / label).is_file() else None
        checks.append(_check("input_hash", upstream, label, "sha256", expected, observed))
        if observed is None:
            hashes["missing_inputs"].append(label)
        else:
            hashes[bucket][label] = observed
    return protocol, manifest, heads, prior, checks, hashes


def _online_inputs(root: Path, protocol: dict, manifest: dict) -> list[dict]:
    """Join exactly one original-source feature to each isolated online label."""

    info = manifest["roles"]["online"]
    label_info = protocol["reader_sidecars"]["evaluator_stores"]["online"]
    features = read_jsonl(root / info["feature_path"])
    labels = read_jsonl(root / label_info["path"])
    feature_by_id = {row["unit_id"]: row for row in features if row["arm"] == "original_source"}
    if (
        len(labels) != 80
        or len(feature_by_id) != 80
        or len({row["component_hash"] for row in labels}) != 80
    ):
        raise ValueError("online_roster_invalid")
    if set(feature_by_id) != set(info["group_ids"]) or set(feature_by_id) != {
        r["component_hash"] for r in labels
    }:
        raise ValueError("online_roster_mismatch")
    result = []
    for label in labels:
        feature = feature_by_id[label["component_hash"]]
        partition = "update" if label["learning_partition"] == "online_update" else "admission"
        expected = partition == "update"
        if (
            label["role"] != "online"
            or label["training_allowed"] is not expected
            or label["label"] not in (0, 1)
            or feature["partition"] != label["learning_partition"]
            or feature["source_sha256"] != feature["original_source_sha256"]
        ):
            raise ValueError("online_custody_invalid")
        result.append(
            {
                "unit_id": label["component_hash"],
                "label": label["label"],
                "raw_probability": label["raw_probability"],
                "partition": partition,
                "feature": feature,
            }
        )
    if sum(row["partition"] == "update" for row in result) != 40:
        raise ValueError("online_partition_invalid")
    return result


def _cost(probability: float, label: int) -> tuple[str, float]:
    costs = {"accept": 5 * probability, "reject": 1 - probability, "escalate": 0.2}
    action = min(costs, key=costs.__getitem__)
    realized = {"accept": 5.0 if label else 0.0, "reject": 0.0 if label else 1.0, "escalate": 0.2}[
        action
    ]
    return action, realized


def _measure_arm(
    root: Path, arm: str, inputs: list[dict], heads: dict, started: float
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Replay one arm with delayed release and one-use held-out admission."""

    state_path = root / RAW / f"{arm}_state.json"
    if state_path.exists():
        state_path.unlink()  # task-owned deterministic state; prior artifacts are untouched
    service = DelayedUpdateService(state_path, arm=arm)
    rows: list[dict] = []
    events: list[dict] = []
    decisions: list[dict] = []
    update_queue: list[str] = []
    admission_queue: list[str] = []
    by_id = {row["unit_id"]: row for row in inputs}
    head = heads["heads"][heads["selected"]]

    def advance() -> None:
        if service.state["proposal"] is None and len(update_queue) >= 5:
            selected = update_queue[:5]
            del update_queue[:5]
            service.propose(selected)
        if service.state["proposal"] is not None and len(admission_queue) >= 5:
            selected = admission_queue[:5]
            del admission_queue[:5]
            outcome = service.admit(selected)
            decisions.append(
                {
                    "arm": arm,
                    "update_ids": service.state["used_updates"][-5:],
                    "admission_ids": selected,
                    **outcome,
                }
            )
            advance()

    def release(origin: int, ordinal: int) -> None:
        item = inputs[origin]
        event_id = item["unit_id"]
        feature = item["feature"]
        prior_hash = service.state_hash
        acknowledgment = service.release(event_id, item["label"], ordinal)
        prediction = service.state["predictions"][event_id]
        probability = prediction["probability"]
        action, realized = _cost(probability, item["label"])
        rows.append(
            {
                "unit_id": event_id,
                "arm": arm,
                "origin_ordinal": origin,
                "partition": item["partition"],
                "label": item["label"],
                "probability": probability,
                "baseline_probability": prediction["base"],
                "typed_action": action,
                "decision_cost": realized,
                "brier": (probability - item["label"]) ** 2,
                "raw_metrics": {
                    "source_propositions_checked": feature["checked_structural_propositions"],
                    "unknown_claims": feature["unknown_claims"],
                },
                "counts": {"independent_group": 1, "paired_view": 1},
                "excluded": feature["excluded"],
                "exclusions": [],
                "censored": feature["censored"],
                "provenance": {
                    "source_sha256": feature["source_sha256"],
                    "label_sidecar": "Exp7602 online isolated evaluator",
                },
            }
        )
        events.append(
            {
                "arm": arm,
                "event_ordinal": len(events),
                "origin_ordinal": origin,
                "release_ordinal": ordinal,
                "partition": item["partition"],
                "event_id": event_id,
                "prior_state_hash": prior_hash,
                "next_state_hash": acknowledgment["state_hash"],
                "acknowledgment": acknowledgment,
            }
        )
        (update_queue if item["partition"] == "update" else admission_queue).append(event_id)
        advance()

    for origin, item in enumerate(inputs):
        feature = item["feature"]
        event_id = item["unit_id"]
        base = score(feature, item["raw_probability"], head)
        service.predict(event_id, origin, base, source_stratum(feature), item["partition"])
        atomic_json(
            root / RAW / "checkpoint.json",
            {"arm": arm, "completed_units": origin + 1, "state_hash": service.state_hash},
        )
        if origin >= 8:
            release(origin - 8, origin)
        if origin % 10 == 9:
            print(
                f"[exp7662] measurement arm={arm} completed={origin + 1}/80 "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    for origin in range(72, 80):
        release(origin, origin + 8)
    return (
        rows,
        events,
        decisions,
        {
            "state_hash": service.state_hash,
            "state_bytes": service.state_bytes,
            "numerical_state_bytes": len(json.dumps(service.state["numerical"]).encode()),
            "accepted_updates": sum(d["accepted"] for d in decisions),
            "candidate_count": len(decisions),
            "mean_update_ns": (
                sum(d["update_ns"] for d in decisions) / len(decisions) if decisions else 0
            ),
        },
    )


def reduce_rows(rows: list[dict]) -> dict:
    """Cold arithmetic uses retained label and forecast, not producer claims."""

    if len(rows) != 240:
        raise ValueError("row_count")
    grouped: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for row in rows:
        if row["arm"] not in grouped or row["label"] not in (0, 1):
            raise ValueError("row_custody")
        probability = row["probability"]
        action, cost = _cost(probability, row["label"])
        if (
            abs(row["brier"] - (probability - row["label"]) ** 2) > 1e-12
            or action != row["typed_action"]
            or abs(cost - row["decision_cost"]) > 1e-12
        ):
            raise ValueError("row_metric_mismatch")
        grouped[row["arm"]].append(row)
    if any(len(grouped[arm]) != 80 for arm in ARMS):
        raise ValueError("arm_count")
    ids = [r["unit_id"] for r in grouped["source"]]
    if len(set(ids)) != 80 or any([r["unit_id"] for r in grouped[arm]] != ids for arm in ARMS):
        raise ValueError("paired_roster")
    return {
        arm: {
            "brier": sum(r["brier"] for r in grouped[arm]) / 80,
            "decision_cost": sum(r["decision_cost"] for r in grouped[arm]) / 80,
            "independent_groups": 80,
        }
        for arm in ARMS
    }


def cold_reduce(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    for bucket in ("producers", "pre_gate_receipts"):
        for label, digest in artifact["source_artifact_hashes"][bucket].items():
            if sha256_file(ROOT / label) != digest:
                raise ValueError("input_hash_mismatch")
    reduction = reduce_rows(artifact["rows"])
    if reduction != artifact["independent_reduction"]:
        raise ValueError("reduction_mismatch")
    return {"passed": True, "independent_groups": 80, "paired_rows": 240}


def _artifact(
    date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    rows: list[dict],
    events: list[dict],
    decisions: list[dict],
    states: dict,
    receipts: list[dict],
    scope: dict,
    spans: list[dict],
    reduction: dict,
) -> dict:
    """Keep protocol readiness separate from scientific benefit and freshness."""

    required = reduce_required_checks(receipts) if receipts else {"required_checks_passed": False}
    authenticated = all(check["passed"] for check in checks)
    causal = len(events) == 240 and all(
        event["release_ordinal"] >= event["origin_ordinal"] + 8
        and event["acknowledgment"]["durable"]
        for event in events
    )
    one_use = all(
        len(
            {
                item
                for decision in decisions
                if decision["arm"] == arm
                for item in decision["admission_ids"]
            }
        )
        == 40
        for arm in ARMS
    )
    durable = all(states.get(arm, {}).get("state_bytes", 100_001) < 100_000 for arm in ARMS)
    measured = authenticated and len(rows) == 240 and causal and one_use and durable
    valid = measured and required["required_checks_passed"]
    klass = "blocked" if not authenticated else "disqualified" if not valid else "null"
    verdict = {
        "blocked": "complete_blocked_missing_external_evidence",
        "disqualified": "complete_disqualified_required_validation",
        "null": "complete_null_delayed_update_no_fresh_benefit",
    }[klass]

    def gate(name: str, passed: bool, operands: dict, principle: str) -> dict:
        return {
            "gate": name,
            "passed": passed,
            "measured_operands": operands,
            "principle": principle,
        }

    source = reduction.get("source", {})
    scalar = reduction.get("scalar", {})
    frozen = reduction.get("frozen", {})
    probability_benefit = bool(
        valid
        and source.get("brier", 1) + 0.01 < scalar.get("brier", 0)
        and source.get("brier", 1) + 0.01 < frozen.get("brier", 0)
    )
    utility_benefit = bool(
        valid
        and source.get("decision_cost", 1) + 0.01
        < min(scalar.get("decision_cost", 0), frozen.get("decision_cost", 0))
    )
    gates = [
        gate(
            "validity",
            valid,
            {
                "authenticated": authenticated,
                "required_checks_passed": required["required_checks_passed"],
                "paired_rows": len(rows),
            },
            "Immutable inputs and required receipts authenticate.",
        ),
        gate(
            "readiness",
            valid and causal and one_use and durable,
            {"causal": causal, "one_use": one_use, "durable": durable},
            "Readiness requires legal release, one-use admission, and durable state.",
        ),
        gate(
            "coverage",
            valid and len(rows) == 240,
            {"independent_groups": len(rows) // 3, "paired_rows": len(rows)},
            "Paired arms do not enlarge the independent sample.",
        ),
        gate(
            "probability_benefit",
            probability_benefit,
            {
                "brier": {arm: reduction.get(arm, {}).get("brier") for arm in ARMS},
                "threshold": 0.01,
            },
            "Source arm requires a Brier margin over both controls.",
        ),
        gate(
            "utility",
            utility_benefit,
            {
                "decision_cost": {arm: reduction.get(arm, {}).get("decision_cost") for arm in ARMS},
                "threshold": 0.01,
            },
            "Realized action cost must improve separately.",
        ),
        gate(
            "retention",
            valid and causal and one_use,
            {"delayed_feedback_events": len(events), "candidate_versions": len(decisions)},
            "Only delayed released updates and one-use held-out admissions count.",
        ),
        gate(
            "freshness",
            False,
            {"fresh_confirmatory_groups": 0},
            "The cached online roster was previously exposed.",
        ),
    ]
    checksum = hashlib.sha256(
        json.dumps(
            {"inputs": hashes, "seed": 7662, "reducer": sha256_file(ROOT / MODULE)}, sort_keys=True
        ).encode()
    ).hexdigest()
    return {
        "schema": "carnot.exp7662.v668.delayed_update_protocol.v1",
        "experiment_id": "exp7662-v668-delayed-update-protocol",
        "milestone": "2026.09.668",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": klass,
        "flagged_adversarial": False,
        "gate_check_summary": [c for c in checks if not c["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": 80,
            "observed": len(rows) // 3,
            "eligible": len(rows) // 3,
            "excluded": sum(r["excluded"] for r in rows if r["arm"] == "source"),
            "censored": sum(r["censored"] for r in rows if r["arm"] == "source"),
            "prior_exposure": "all online groups exposed; exploratory only",
            "claim_limit": "no fresh confirmatory benefit",
        },
        "inference_substrate": "cpu_delayed_source_residual_update_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no model loaded or planned in current CPU work",
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
            "protocol": 7662,
            "purpose": "deterministic roster replay; no random training",
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
            "rows": "One original source group per arm; repeat views do not increase N.",
            "sample_size_budget": "Unknown and censored evidence stays in the denominator.",
            "honest_verdict": "Completion, validity and scientific benefit are distinct.",
            "flagged_adversarial": "A flagged reader cannot open a gate.",
            "duration_s": "Actual current CPU time without padding.",
            "inference_substrate_class": "No current model load or generation.",
            "MODEL_SPECS": "Empty because this is CPU-only cached work.",
            "source_artifact_hashes": "Only immutable upstream bytes bind reproducibility.",
            "verifier_is_oracle": "Isolated dataset labels are not exact fixture truth.",
            "event_rows": "Each durable release names origin, release and state hashes.",
            "hardware_path": "CPU counters now; bounded native batch updates are next.",
            "acceptance_gate_results": "Validity, readiness, coverage, benefit, utility, retention and freshness differ.",
        },
        "historical_model_id": "unsloth/Qwen3.8-27B-GGUF",
        "independent_reduction": reduction,
        "event_rows": events,
        "admission_decisions": decisions,
        "state_summaries": states,
        "delayed_protocol_ready_score": int(valid and causal and one_use and durable),
        "probability_benefit_score": int(probability_benefit),
        "utility_benefit_score": int(utility_benefit),
        "feedback_protocol_path": (RAW / "protocol.json").as_posix(),
        "hardware_path": {
            "current": "CPU counters",
            "planned": "bounded native batch updates",
            "fpga_measured": False,
            "speedup_100x_measured": False,
        },
        "fixture_control": {
            "verdict_class": "circular_positive",
            "claim": "exact fixture truth only",
        },
        "retirement": "Retire the unchanged V667 zero-check grammar after the same null; external absence is not scientific disproof.",
    }


def run_experiment(root: Path, date: str, output: Path) -> dict:
    """Freeze affected scope, replay online events, then gate exact bytes."""

    root = root.resolve()
    started = time.monotonic()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict] = []

    def progress(phase: str, event: str, detail: str = "") -> None:
        print(
            f"[exp7662] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {detail}",
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
    protocol, manifest, heads, _prior, checks, hashes = authenticate(root)
    atomic_json(raw / "preconditions.json", {"checks": checks, "hashes": hashes})
    close("preconditions", begin, len(checks), raw / "preconditions.json")
    with tempfile.TemporaryDirectory(prefix="exp7662-", dir="/tmp") as private_dir:
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
        rows: list[dict] = []
        events: list[dict] = []
        decisions: list[dict] = []
        states: dict = {}
        reduction: dict = {}
        if all(c["passed"] for c in checks):
            inputs = _online_inputs(root, protocol, manifest)
            for arm in ARMS:
                progress("measurement", "arm_start", arm)
                arm_rows, arm_events, arm_decisions, state = _measure_arm(
                    root, arm, inputs, heads, started
                )
                rows.extend(arm_rows)
                events.extend(arm_events)
                decisions.extend(arm_decisions)
                states[arm] = state
                progress("measurement", "arm_complete", f"{arm} rows={len(arm_rows)}")
            reduction = reduce_rows(rows)
            atomic_json(
                raw / "protocol.json",
                {"event_rows": events, "admission_decisions": decisions, "state_summaries": states},
            )
        else:
            atomic_json(
                raw / "checkpoint.json",
                {"completed_units": 0, "blocked_checks": [c for c in checks if not c["passed"]]},
            )
        close("measurement", begin, len(rows) // 3, raw / "checkpoint.json")
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
        artifact = _artifact(
            date,
            started,
            checks,
            hashes,
            rows,
            events,
            decisions,
            states,
            receipts,
            scope,
            spans,
            reduction,
        )
        candidate = raw / "exact_terminal_candidate.json"
        atomic_json(candidate, artifact)
        if artifact["verdict_class"] == "blocked":
            artifact["validation_receipts"]["terminal_readers"] = []
            artifact["validation_receipts"]["terminal_reader_disposition"] = (
                "blocked_before_evaluation_inputs_opened"
            )
            artifact["duration_s"] = time.monotonic() - started
            progress("publish", "before_atomic_write", "blocked_missing_external_input")
            atomic_json(output, artifact)
            progress("publish", "complete", f"verdict={artifact['honest_verdict']}")
            return artifact
        python = str(root / ".venv/bin/python")
        readers = [
            CommandSpec(
                "cold_reduce",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7662_v668_delayed_update_protocol",
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
        artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = sha256_file(candidate)
        artifact["flagged_adversarial"] = not terminal[1]["passed"]
        if not all(receipt["passed"] for receipt in terminal):
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
            for field in (
                "delayed_protocol_ready_score",
                "probability_benefit_score",
                "utility_benefit_score",
            ):
                artifact[field] = 0
            for gate in artifact["acceptance_gate_results"]:
                gate["passed"] = False
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        progress("publish", "before_atomic_write")
        atomic_json(output, artifact)
        progress("publish", "complete", f"verdict={artifact['honest_verdict']}")
        return artifact


def main(argv: list[str] | None = None) -> int:
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
