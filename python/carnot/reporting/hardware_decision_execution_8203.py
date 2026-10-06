"""REQ-REPORT-8203: checked software evidence precedes terminal publication."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import hardware_decision_8203 as h
from carnot.reporting import hardware_decision_inputs_8203 as inputs
from carnot.reporting import hardware_service_execution_8190 as qualified
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8203_v708_hardware_decision_boundary"
SCRIPT = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_hardware_decision_8203.py"
OWNED = [
    "python/carnot/reporting/" + name + ".py"
    for name in (
        "hardware_decision_8203",
        "hardware_decision_inputs_8203",
        "hardware_decision_execution_8203",
    )
]
MODEL_SPECS: list[Json] = []
execute = qualified.execute


def commands(private: Path) -> list[CommandSpec]:
    """Explicit files keep selectors away from static tools and bound new-code coverage."""
    cfg = private / "coverage.ini"
    cfg.write_text(
        "[run]\nparallel=True\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    */" + p + "\n" for p in [*OWNED, SCRIPT])
    )
    specs = build_scoped_commands(
        ROOT,
        [TEST],
        OWNED,
        static_paths=[SCRIPT],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    result = []
    for spec in specs:
        argv = spec.argv
        if spec.name == "changed_module_coverage":
            argv = (argv[0], "run", "--rcfile=" + str(cfg), *argv[2:])
        if spec.name == "changed_module_coverage_report":
            result.append(
                CommandSpec(
                    "coverage_combine",
                    (
                        str(ROOT / ".venv/bin/coverage"),
                        "combine",
                        "--rcfile=" + str(cfg),
                        str(private),
                    ),
                    "owned_coverage",
                )
            )
        argv = tuple(a + ",*/" + SCRIPT if a.startswith("--include=") else a for a in argv)
        if spec.name == "changed_module_mypy":
            argv += ("--strict", "--follow-imports=silent")
        result.append(replace(spec, argv=argv))
    result.append(
        CommandSpec(
            "private_E2E_success_missing_tamper_cold",
            (
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                TEST + "::test_private_cli_and_cold_tamper",
                TEST + "::test_missing_external_inputs",
                TEST + "::test_health_custody_and_independent_cost_replay",
            ),
            "private_e2e",
        )
    )
    return result


def terminal(path: Path) -> list[CommandSpec]:
    """Unmodified adversarial and strict row auditors inspect candidate bytes."""
    return [
        replace(
            s,
            argv=tuple(
                str(ROOT / SCRIPT) if a == str(ROOT / qualified.SCRIPT) else a for a in s.argv
            ),
        )
        for s in qualified.terminal(path)
    ]


def replay(path: Path) -> Json:
    """Reconstruct exact decisions from frozen inputs; timers remain recorded observations."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        checked(ref)
    fresh = h.reduce(json.loads(checked(value["replay_input_reference"]).read_bytes()))
    for key in (
        "honest_verdict",
        "verdict_class",
        "branch_readiness",
        "completed_count",
        "amdahl_bounds",
        "interval_bound_domain",
        "board_rows",
    ):
        if value["verdict_class"] == "disqualified" and key in {"honest_verdict", "verdict_class"}:
            continue
        if value[key] != fresh[key]:
            raise ValueError("reduction_drift:" + key)
    ignored = {"elapsed_ns", "fallback_ns"}
    stable = lambda rows: [{k: v for k, v in r.items() if k not in ignored} for r in rows]
    if stable(value["precision_rows"]) != stable(fresh["precision_rows"]):
        raise ValueError("precision_drift")
    summaries = []
    for kind in h.CONFIG["precisions"]:
        measured = [r for r in value["precision_rows"] if r["precision"] == kind]
        summaries.append(
            dict(
                precision=kind,
                maximum_error=max(
                    (
                        abs(r["probability_reference"] - r["probability_approximate"])
                        for r in measured
                    ),
                    default=0,
                ),
                final_mismatches=sum(r["final_typed_mismatch"] for r in measured),
                fallback_count=sum(bool(r["fallback"]) for r in measured),
            )
        )
    if value["probability_error_summary"] != summaries:
        raise ValueError("headline_drift")
    for bound in value["amdahl_bounds"]:
        rows = [r for r in value["workload_rows"] if r["condition"] == bound["condition"]]
        if bound["optimistic_ceiling"] != sum(r["numerator"] for r in rows) / sum(
            r["denominator"] for r in rows
        ):
            raise ValueError("independent_ceiling_drift")
    if value["required_checks_passed"] != all(
        r["passed"] and r["normal_exit"] for r in value["validation_receipts"]
    ):
        raise ValueError("validation_receipt_drift")
    return dict(passed=True, rows_checksum=canonical_hash(stable(fresh["precision_rows"])))


def main(argv: list[str] | None = None) -> int:
    """Freeze commands, measure cached CPU operations, validate and atomically publish."""
    began = time.monotonic()
    print("[exp8203] phase=start completed=0 pending=1 model_loads=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--repository-health-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        if args.input and output.is_relative_to(ROOT / "results"):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="carnot8203-"))
        plan = [] if args.input else commands(private)
        candidate = private / (NAME + ".json")
        health = CommandSpec(
            "repository_health_once",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health_not_science_gate",
            1800,
        )
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[asdict(s) for s in plan],
                terminal=[asdict(s) for s in terminal(candidate)],
                repository_health=asdict(health),
                config=h.CONFIG,
                frozen_before_measurement=True,
            ),
        )
        print("[exp8203] phase=preconditions_before completed=0 pending=1", flush=True)
        data = json.loads(args.input.read_bytes()) if args.input else inputs.load(args.root, raw)
        data["fixture"] = bool(args.input)
        atomic_json(raw / "replay_inputs.json", data)
        print("[exp8203] phase=preconditions_after completed=1 pending=0", flush=True)
        start = time.monotonic()
        value = h.reduce(data)
        reduction_s = time.monotonic() - start
        receipts = execute(
            plan,
            raw / "owned",
            dict(CARNOT_8203_COVERAGE_CONFIG=str(private / "coverage.ini"), JAX_PLATFORMS="cpu"),
        )
        health_receipts = (
            json.loads(args.repository_health_receipt.read_bytes())
            if args.repository_health_receipt
            else []
            if args.input
            else execute(
                [health],
                raw / "health",
                dict(
                    PYTEST_ADDOPTS="-n 0 --no-cov", COVERAGE_FILE=str(private / ".coverage.health")
                ),
            )
        )
        for receipt in health_receipts:
            log = checked(dict(path=receipt["log_path"], sha256=receipt["log_sha256"]))
            saved = raw / "health_custody" / (receipt["log_sha256"].split(":")[1] + ".log")
            saved.parent.mkdir(parents=True, exist_ok=True)
            if not saved.exists():
                saved.write_bytes(log.read_bytes())
            receipt["log_path"] = str(saved)
        passed = all(r["passed"] and r["normal_exit"] for r in receipts)
        if not passed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                hardware_boundary_ready_score=0,
            )
        atomic_json(
            raw / "primitive_rows.json", dict(rows=value["rows"], board_rows=value["board_rows"])
        )
        value.update(
            experiment_id=8203,
            task_id="exp8203-hardware-decision-boundary",
            run_date=args.date,
            schema="carnot.hardware_decision_boundary.v708.v1",
            config=h.CONFIG,
            random_seed=h.CONFIG["seed"],
            duration_s=time.monotonic() - began,
            phase_spans=[dict(phase="CPU_precision_reduction", duration_s=reduction_s)],
            required_checks_passed=passed,
            flagged_adversarial=False,
            validation_receipts=receipts,
            repository_health=health_receipts,
            preconditions_checked=data["checks"],
            claim_scope="Frozen exposed selective CPU precision; historical research service bounds and board custody",
            exposure_scope="private oracle fixture"
            if args.input
            else "exposed cached development sources",
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=MODEL_SPECS,
            model_invocation_counts=ZERO_INVOCATION_COUNTS,
            call_ledger=[],
            historical_model_provenance=data.get("historical_model_provenance", {}),
            source_artifact_hashes=data["references"],
            code_config_hashes=[
                reference(ROOT / p)
                for p in [
                    *OWNED,
                    SCRIPT,
                    TEST,
                    "python/carnot/verify/selective_rule_8194.py",
                    "python/carnot/verify/evidence_energy_8154.py",
                    "python/carnot/reporting/primary_publication.py",
                    "scripts/adversarial_verify.py",
                    "scripts/verdict_row_consistency_lint.py",
                ]
            ],
            raw_shard_hashes=[
                reference(raw / p)
                for p in ("replay_inputs.json", "primitive_rows.json", "validation_commands.json")
            ],
            replay_input_reference=reference(raw / "replay_inputs.json"),
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            reproducibility_checksum=canonical_hash(value["rows"]),
            field_principles=dict(
                readiness="Normal owned checks; unavailable external branches remain blocked",
                precision="Frozen scales and exact quantiles; uncertain decisions use float64",
                substrate="Imported Qwen calls are historical; current model and device calls are zero",
                costs="Only removable arithmetic yields a pure Amdahl bound; mixed timers are optimistic",
            ),
        )
        for r in receipts + health_receipts:
            value["raw_shard_hashes"].append(dict(path=r["log_path"], sha256=r["log_sha256"]))
        atomic_json(candidate, value)
        atomic_json(raw / "independent_reduction.json", replay(candidate))

        def validate(path: Path) -> Json:
            """Every final auditor receives exact candidate bytes and a fresh log location."""
            checks = execute(terminal(path), raw / ("terminal-" + str(time.time_ns())))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks
            )

        report = validate(candidate)
        if not report["passed"]:
            value.update(
                honest_verdict="complete_disqualified_terminal_validation",
                verdict_class="disqualified",
                required_checks_passed=False,
                hardware_boundary_ready_score=0,
            )
            atomic_json(raw / "failed_terminal_candidate.json", value)
            atomic_json(raw / "terminal_validation.json", report)
            return 1
        value["validation_receipts"] += report["receipts"]
        value["raw_shard_hashes"].append(reference(raw / "independent_reduction.json"))
        for r in report["receipts"]:
            value["raw_shard_hashes"].append(dict(path=r["log_path"], sha256=r["log_sha256"]))
        if output.exists():
            (raw / "preserved_primary.json").write_bytes(output.read_bytes())
        print("[exp8203] phase=publication_before completed=0 pending=1", flush=True)
        publication = publish_primary(output, value, validate)
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=publication, normal_process_exit=True),
        )
        print(
            f"[exp8203] phase=complete completed={value['completed_count']} pending=0 verdict={value['honest_verdict']}",
            flush=True,
        )
        return 0 if passed else 1
    except (ValueError, OSError, KeyError) as error:
        print(f"[exp8203] failed={type(error).__name__}:{error}", flush=True)
        return 1
