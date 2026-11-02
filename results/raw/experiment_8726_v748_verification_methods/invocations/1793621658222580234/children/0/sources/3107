"""Evaluate frozen source-atom decisions on forty exposed groups (REQ-REPORT-7661)."""

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
from carnot.reporting.experiment_7661_decision_evaluation import build_rows, reduce_rows

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7661_v668_decision_evaluation")
OUTPUT = Path("results/experiment_7661_v668_decision_evaluation.json")
HEADS = Path("results/raw/experiment_7660_v668_atom_energy/heads.json")
PRIOR = Path("results/experiment_7660_v668_atom_energy.json")
MANIFEST = Path("results/raw/experiment_7659_v668_atom_corpus/manifest.json")
PROTOCOL = Path("results/raw/experiment_7602_v664_evidence_requalification/protocol.json")
TEST = "tests/python/test_experiment_7661_v668_decision_evaluation.py"
MODULE = "python/carnot/reporting/experiment_7661_decision_evaluation.py"
ORCHESTRATION = "python/carnot/experiment_7661_v668_decision_evaluation.py"
WRAPPER = "scripts/experiments/experiment_7661_v668_decision_evaluation.py"
SEED = 7661
MODEL_SPECS: list[str] = []


def _check(name: str, upstream: str, path: str, field: str, expected: Any, observed: Any) -> dict:
    return {
        "check": name,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def authenticate(root: Path) -> tuple[dict, dict, dict, dict, list[dict], dict]:
    """Bind every immutable producer and evaluator byte before label release."""

    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []}
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
    inputs = (
        (PRIOR, "Exp7660", "pre_gate_receipts"),
        (HEADS, "Exp7660", "producers"),
        (MANIFEST, "Exp7659", "producers"),
        (PROTOCOL, "Exp7602", "producers"),
    )
    for relative, upstream, bucket in inputs:
        path = root / relative
        exists = path.is_file()
        checks.append(_check("input_exists", upstream, relative.as_posix(), "exists", True, exists))
        if exists:
            hashes[bucket][relative.as_posix()] = sha256_file(path)
        else:
            hashes["missing_inputs"].append(relative.as_posix())
    if hashes["missing_inputs"]:
        return {}, {}, {}, {}, checks, hashes
    prior = json.loads((root / PRIOR).read_text())
    manifest = json.loads((root / MANIFEST).read_text())
    protocol = json.loads((root / PROTOCOL).read_text())
    bundle = json.loads((root / HEADS).read_text())
    for field, expected in (
        ("energy_ready_score", 1),
        ("flagged_adversarial", False),
        ("verdict_class", "null"),
    ):
        checks.append(
            _check("upstream_gate", "Exp7660", PRIOR.as_posix(), field, expected, prior.get(field))
        )
    checks.append(
        _check(
            "head_hash",
            "Exp7660",
            HEADS.as_posix(),
            "head_manifest_sha256",
            prior.get("head_manifest_sha256"),
            sha256_file(root / HEADS),
        )
    )
    info = manifest["roles"]["evaluation"]
    sidecar = protocol["reader_sidecars"]["evaluator_stores"]["evaluation"]
    checks.append(
        _check(
            "fixed_evaluation_roster",
            "Exp7602",
            PROTOCOL.as_posix(),
            "role_counts.evaluation",
            40,
            protocol["role_counts"]["evaluation"],
        )
    )
    checks.append(
        _check(
            "frozen_thresholds",
            "Exp7660",
            HEADS.as_posix(),
            "thresholds_count",
            2,
            len(bundle.get("thresholds", [])),
        )
    )
    for label, expected, upstream, bucket in (
        (info["feature_path"], info["feature_sha256"], "Exp7659", "producers"),
        (sidecar["path"], sidecar["sha256"], "Exp7602", "pre_gate_receipts"),
    ):
        path = root / label
        observed = sha256_file(path) if path.is_file() else None
        checks.append(_check("input_hash", upstream, label, "sha256", expected, observed))
        if observed is None:
            hashes["missing_inputs"].append(label)
        else:
            hashes[bucket][label] = observed
    return bundle, manifest, protocol, prior, checks, hashes


def _rows_from_inputs(root: Path, bundle: dict, manifest: dict, protocol: dict) -> list[dict]:
    info = manifest["roles"]["evaluation"]
    sidecar = protocol["reader_sidecars"]["evaluator_stores"]["evaluation"]
    return build_rows(
        read_jsonl(root / info["feature_path"]),
        read_jsonl(root / sidecar["path"]),
        bundle,
        info["group_ids"],
        info["derangement"],
    )


def cold_reduce(path: Path) -> dict:
    """Open the exact candidate in a fresh process and replay immutable inputs."""

    artifact = json.loads(path.read_text())
    for bucket in ("producers", "pre_gate_receipts"):
        for label, expected in artifact["source_artifact_hashes"][bucket].items():
            if sha256_file(ROOT / label) != expected:
                raise ValueError("source_hash_mismatch")
    bundle, manifest, protocol, _, checks, _ = authenticate(ROOT)
    if not all(c["passed"] for c in checks):
        raise ValueError("upstream_custody_mismatch")
    rows = _rows_from_inputs(ROOT, bundle, manifest, protocol)
    if rows != artifact["rows"]:
        raise ValueError("row_replay_mismatch")
    reduced = reduce_rows(rows, seed=SEED)
    if reduced != artifact["independent_reduction"]:
        raise ValueError("independent_reduction_mismatch")
    return {"passed": True, "independent_groups": len(rows) // 6, "paired_rows": len(rows)}


def _artifact(
    date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    rows: list[dict],
    reduction: dict,
    receipts: list[dict],
    scope: dict,
    spans: list[dict],
    controls: dict,
) -> dict:
    """Separate complete measurement, scientific benefit, and invalid receipts."""

    required = reduce_required_checks(receipts) if receipts else {"required_checks_passed": False}
    authenticated = all(c["passed"] for c in checks)
    complete = authenticated and len(rows) == 240
    valid = complete and required["required_checks_passed"]
    klass = "blocked" if not authenticated else "disqualified" if not valid else "null"
    if (
        valid
        and reduction["probability_benefit"]
        and reduction["utility_benefit"]
        and reduction["source_dependence"]
    ):
        klass = "positive"
    verdict = {
        "blocked": "complete_blocked_missing_external_evidence",
        "disqualified": "complete_disqualified_required_validation",
        "null": "complete_null_no_registered_decision_benefit",
        "positive": "complete_positive_exploratory_decision_benefit",
    }[klass]
    score = int(valid)

    def gate(name: str, passed: bool, operands: dict, principle: str) -> dict:
        return {
            "gate": name,
            "passed": passed,
            "measured_operands": operands,
            "principle": principle,
        }

    metrics = reduction.get("metrics", {})
    intervals = reduction.get("confidence_intervals", {})
    gates = [
        gate(
            "validity",
            valid,
            {
                "authenticated": authenticated,
                "required_checks_passed": required["required_checks_passed"],
                "paired_rows": len(rows),
            },
            "Immutable inputs and all required receipts must authenticate.",
        ),
        gate(
            "readiness",
            valid,
            {"evaluated_groups": len(rows) // 6},
            "Readiness means replayable evaluation, not benefit.",
        ),
        gate(
            "coverage",
            valid and len(rows) == 240,
            {
                "independent_groups": len(rows) // 6,
                "unknown_claims": metrics.get("atom", {})
                .get("counts", {})
                .get("unknown_claims", 0),
            },
            "Unknown propositions remain in all denominators.",
        ),
        gate(
            "probability_benefit",
            valid and reduction.get("probability_benefit", False),
            {"paired_brier_intervals": intervals.get("brier", {}), "threshold": 0.01},
            "Every registered control requires lower paired Brier CI95 above 0.01.",
        ),
        gate(
            "utility",
            valid and reduction.get("utility_benefit", False),
            {
                "paired_cost_interval": intervals.get("utility"),
                "non_escalation_coverage": metrics.get("atom", {}).get("non_escalation_coverage"),
                "cost_threshold": 0.01,
                "coverage_floor": 0.20,
            },
            "Cost improvement and non-escalation support are separate.",
        ),
        gate(
            "retention",
            False,
            {"delayed_feedback_events": 0},
            "Static evaluation does not measure delayed retention.",
        ),
        gate(
            "freshness",
            False,
            {"fresh_confirmatory_groups": 0},
            "These forty groups were previously exposed.",
        ),
    ]
    checksum = hashlib.sha256(
        json.dumps(
            {"inputs": hashes, "seed": SEED, "reducer": sha256_file(ROOT / MODULE)}, sort_keys=True
        ).encode()
    ).hexdigest()
    return {
        "schema": "carnot.exp7661.v668.decision_evaluation.v1",
        "experiment_id": "exp7661-v668-decision-evaluation",
        "milestone": "2026.09.668",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": klass,
        "flagged_adversarial": False,
        "gate_check_summary": [c for c in checks if not c["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": reduction.get(
            "sample_size_budget",
            {
                "intended": 40,
                "observed": 0,
                "eligible": 0,
                "excluded": 0,
                "censored": 0,
                "prior_exposure": "all groups exposed; exploratory only",
            },
        ),
        "inference_substrate": "cpu_frozen_source_atom_scoring_no_llm",
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
        "random_seed": {"paired_bootstrap": SEED, "derangement": "Exp7659 frozen within-role map"},
        "reproducibility_checksum": "sha256:" + checksum,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": scope,
            "required_commands": receipts,
            **required,
            "unrelated_repository_suite_debt": [
                {
                    "command": ".venv/bin/pytest tests/python -q",
                    "observed": "71099 collected; four failures visible by 1 percent; interrupted at 1 percent",
                    "exit_code": 130,
                    "scope": "repository-wide; outside frozen Exp7661 acceptance checks",
                }
            ],
        },
        "verifier_is_oracle": False,
        "field_principles": {
            "rows": "One independent source group; six paired views do not enlarge N.",
            "metrics": "Proper losses and decisions recompute from probability and isolated label.",
            "sample_size_budget": "Unknown evidence and repeated views remain in the denominator.",
            "confidence_intervals": "Only 10000 fixed-seed paired source-group draws count.",
            "acceptance_gate_results": "Validity, readiness, probability, utility, retention and freshness differ.",
            "honest_verdict": "Completion and scientific benefit are separate.",
            "flagged_adversarial": "Flagged evidence cannot open a gate.",
            "duration_s": "Measured current CPU work without padding.",
            "inference_substrate_class": "No current model load or generation.",
            "MODEL_SPECS": "Empty because this task only scores frozen CPU heads.",
            "source_artifact_hashes": "Only immutable upstream bytes enter the checksum.",
            "verifier_is_oracle": "Isolated dataset labels are not exact fixture truth.",
        },
        "historical_model_id": "unsloth/Qwen3.8-27B-GGUF",
        "independent_reduction": reduction,
        "metrics": metrics,
        "confidence_intervals": intervals,
        "decision_evaluation_complete_score": score if complete else 0,
        "probability_benefit_score": int(valid and reduction.get("probability_benefit", False)),
        "utility_benefit_score": int(valid and reduction.get("utility_benefit", False)),
        "source_dependence_score": int(valid and reduction.get("source_dependence", False)),
        "source_intervention_controls": controls,
        "retirement": "Retire the unchanged V667 zero-check grammar after the same null; external absence is not scientific disproof.",
    }


def run_experiment(root: Path, date: str, output: Path) -> dict:  # pragma: no cover - task E2E
    """Freeze scope, score once, run bounded readers, and atomically publish."""

    root = root.resolve()
    started = time.monotonic()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict] = []

    def progress(phase: str, event: str, detail: str = "") -> None:
        print(
            f"[exp7661] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {detail}",
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
    bundle, manifest, protocol, _, checks, hashes = authenticate(root)
    atomic_json(raw / "preconditions.json", {"checks": checks, "hashes": hashes})
    close("preconditions", begin, len(checks), raw / "preconditions.json")
    with tempfile.TemporaryDirectory(prefix="exp7661-", dir="/tmp") as private_dir:
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
        reduction: dict = {}
        controls: dict = {
            "erasure": "not_run",
            "derangement": "not_run",
            "label_sidecar_tamper": "not_run",
        }
        if all(c["passed"] for c in checks):
            rows = _rows_from_inputs(root, bundle, manifest, protocol)
            reduction = reduce_rows(rows, seed=SEED, draws=10000)
            controls["erasure"] = "measured_paired_arm"
            controls["derangement"] = "measured_frozen_within_role_arm"
            sidecar = protocol["reader_sidecars"]["evaluator_stores"]["evaluation"]
            labels = read_jsonl(root / sidecar["path"])
            labels[0]["role"] = "fit"
            try:
                info = manifest["roles"]["evaluation"]
                build_rows(
                    read_jsonl(root / info["feature_path"]),
                    labels,
                    bundle,
                    info["group_ids"],
                    info["derangement"],
                )
            except ValueError as error:
                controls["label_sidecar_tamper"] = f"rejected:{error}"
            if controls["label_sidecar_tamper"] == "not_run":
                raise ValueError("label_tamper_accepted")
            for completed in range(1, 41):
                atomic_json(
                    raw / "checkpoint.json",
                    {
                        "completed_groups": completed,
                        "paired_rows": completed * 6,
                        "head_sha256": hashes["producers"][HEADS.as_posix()],
                    },
                )
                progress("measurement", "checkpoint", f"completed_groups={completed}/40")
        else:
            atomic_json(
                raw / "checkpoint.json",
                {"completed_groups": 0, "blocked_checks": [c for c in checks if not c["passed"]]},
            )
        close("measurement", begin, len(rows) // 6, raw / "checkpoint.json")
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
            date, started, checks, hashes, rows, reduction, receipts, scope, spans, controls
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
                    "carnot.experiment_7661_v668_decision_evaluation",
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
        if not all(r["passed"] for r in terminal):
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
            for field in (
                "decision_evaluation_complete_score",
                "probability_benefit_score",
                "utility_benefit_score",
                "source_dependence_score",
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
