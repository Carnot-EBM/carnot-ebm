"""REQ-REPORT-8122: publish terminal accounting only after owned checks pass.

The capstone runs reductions of existing evidence. Its current process never
loads a model, trains a head or performs a service benchmark. A separate child
repeats the reduction so a normal exit is observed before primary publication.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import v701_capstone as previous
from carnot.reporting import v702_capstone_inputs as inputs
from carnot.reporting import v702_capstone_reduction as reduction
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = inputs.ROOT
NAME = "experiment_8122_v702_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v702_capstone_8122.py"
OWNED = [
    "python/carnot/reporting/v702_capstone.py",
    "python/carnot/reporting/v702_capstone_inputs.py",
    "python/carnot/reporting/v702_capstone_reduction.py",
    CLI,
]
reference = inputs.reference
progress = inputs.progress
qualify = previous.qualify


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze scoped current checks and a single bounded global health diagnostic."""
    specs = previous.commands(scratch)
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=True\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    mapping = dict(zip(previous.OWNED, OWNED[:2] + [CLI], strict=True))
    mapping[previous.TEST] = TEST
    result = []
    for spec in specs:
        argv = tuple(mapping.get(a, a) for a in spec.argv)
        if spec.name in ("ruff_check", "ruff_format", "strict_mypy"):
            argv += (OWNED[2],)
        result.append(CommandSpec(spec.name, argv, spec.scope, spec.timeout_s))
    return result


def replay(path: Path) -> Json:
    """Rehash saved operands and recompute decisions instead of trusting headlines."""
    value = json.loads(path.read_bytes())
    for ref in [
        value["replay_input_reference"],
        *value["source_artifact_hashes"],
        *value["code_config_hashes"],
        *value["raw_shard_hashes"],
    ]:
        if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("input_hash_drift:" + ref["path"])
    for receipt in value["validation_receipts"]:
        if reference(Path(receipt["log_path"]))["sha256"] != receipt["log_sha256"]:
            raise ValueError("validation_log_hash_drift")
    data = json.loads(Path(value["replay_input_reference"]["path"]).read_bytes())
    for index, task in enumerate(data["tasks"][:-1]):
        audit = data["independent_reductions"].get(task["id"], {})
        if audit.get("available"):
            operand = dict(
                raw_shard_hashes=[audit["primitive_reference"]],
                reductions=audit["result"],
                reduction=audit["result"],
            )
            recomputed = inputs.primitive_audit(operand, 8110 + index)
            if recomputed["result"] != audit["result"]:
                raise ValueError("primitive_reduction_drift:" + task["id"])
    fresh = reduction.reduce(data)
    if "required_checks_passed" in value:
        qualify(fresh, value["required_checks_passed"])
    for key, expected in fresh.items():
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal(path: Path) -> Json:
    """Bind cold replay and both auditors to exactly the bytes readers will select."""
    py = str(ROOT / ".venv/bin/python")
    plan = [
        CommandSpec(name, argv, "terminal", 90)
        for name, argv in (
            ("cold_replay", (py, "-u", str(ROOT / CLI), "--cold-replay", str(path))),
            ("adversarial", (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path))),
            (
                "strict_rows",
                (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            ),
        )
    ]
    receipts = run_commands(
        ROOT, plan, log_dir=path.parent / "terminal_logs" / str(time.time_ns()), heartbeat_s=10
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """A flushed bounded CLI turns external absence into a finished blocked result."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot8122-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            specs = [] if args.worker_input else commands(scratch)
            atomic_json(
                raw / "validation_manifest.json",
                dict(commands=[asdict(s) for s in specs], owned=OWNED),
            )
            progress("before_evidence_load")
            data = (
                json.loads(args.worker_input.read_bytes())
                if args.worker_input
                else inputs.load(args.root, raw)
            )
            atomic_json(raw / "replay_inputs.json", data)
            progress("after_evidence_load_before_reduction", 12, 1)
            value = reduction.reduce(data)
            measurement = []
            if not args.worker_input:
                child = CommandSpec(
                    "normal_reduction_exit",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(ROOT / CLI),
                        "--worker-input",
                        str(raw / "replay_inputs.json"),
                        "--output",
                        str(raw / "worker.json"),
                    ),
                    "measurement",
                    90,
                )
                measurement = run_commands(
                    ROOT, [child], log_dir=raw / "measurement_logs", heartbeat_s=10
                )
                child_value = json.loads((raw / "worker.json").read_bytes())
                if not measurement[0]["passed"] or any(
                    child_value[k] != v for k, v in value.items()
                ):
                    raise ValueError("independent_reduction_child_failed")
            atomic_json(raw / "independent_reduction.json", value)
            progress("after_reduction_before_validation", 13, len(specs))
            receipts = run_commands(
                ROOT,
                specs,
                log_dir=raw / "validation_logs",
                heartbeat_s=10,
                extra_env=dict(CARNOT_8122_COVERAGE_CONFIG=str(scratch / "coverage.ini")),
            )
            receipts = [
                dict(
                    r,
                    log_path=str((ROOT / r["log_path"]).resolve()),
                    argv=r["command_argv"],
                    normal_exit=r["exit_code"] == 0 and not r["timed_out"],
                )
                for r in receipts
            ]
            owned = [r for r in receipts if r["scope"] == "owned"]
            health = [r for r in receipts if r["scope"] == "repository_health"]
            passed = bool(owned) and all(r["passed"] for r in owned)
            if not args.worker_input:
                qualify(value, passed)
            publication = next((r for r in receipts if r["name"] == "publication_gate"), None)
            gates = (
                json.loads(Path(publication["log_path"]).read_bytes())
                if publication
                else dict(paper_ready=False, unmet_gates=["private_worker"])
            )
            coverage_path = scratch / "coverage.json"
            coverage = json.loads(coverage_path.read_bytes()) if coverage_path.is_file() else {}
            atomic_json(raw / "coverage.json", coverage)
            atomic_json(
                raw / "validation_receipts.json", dict(owned=owned, repository_health=health)
            )
            atomic_json(raw / "primitive_rows.json", dict(rows=value["rows"]))
            duration = time.monotonic() - began
            value.update(
                experiment_id=8122,
                task_id="exp8122-capstone",
                run_date=args.date,
                milestone="2026.10.702",
                schema="carnot.v702.capstone.v1",
                random_seed=702,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_invocation_counts=ZERO_INVOCATION_COUNTS,
                call_ledger=[],
                trained_head_specs=[],
                duration_s=duration,
                phase_spans=[
                    dict(
                        phase="evidence_reduction_validation",
                        start_s=0,
                        end_s=duration,
                        completed_units=13,
                        pending_units=0,
                    )
                ],
                methodology_note="Hash-bound primitive replay and independent exposed-development branch decisions; no current model work.",
                flagged_adversarial=False,
                validation_receipts=owned,
                repository_health=health,
                measurement_exit_receipts=measurement,
                source_artifact_hashes=data["references"],
                code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
                replay_input_reference=reference(raw / "replay_inputs.json"),
                raw_shard_hashes=[reference(p) for p in raw.rglob("*") if p.is_file()],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                coverage_statement_counts=coverage.get("files", {}),
                publication_gate_results=gates,
                paper_ready=gates["paper_ready"],
                unmet_gates=gates["unmet_gates"],
                external_publication_authorized=False,
                publication_evidence_scope="Stable historical FoVer G1-G4; existing corrigenda preserved; separate from V702 science.",
                acceptance_gates=dict(
                    current_owned_validation=passed,
                    all_science_inputs=value["science_ready_score"],
                    independent_generalization=False,
                    generalized_learning=False,
                ),
            )
            if not args.worker_input:
                value["required_checks_passed"] = passed
            value["reproducibility_checksum"] = canonical_hash(
                [value["replay_input_reference"], value["code_config_hashes"]]
            )
            value["field_principles"] = {
                k: f"{k} binds scoped observed evidence without independent generalization credit."
                for k in value
            }
            value["field_principles"].update(
                capstone_ready_score="Administrative completion cannot hide missing science or invalid measurements.",
                retirement_candidates="Exact verdicts and byte hashes retire only an unchanged qualified null mechanism.",
                H1="Paired typed costs aggregate by source; duplicate seeds supply no independent evidence.",
                H2="Finite causal benefit requires separate retention safety; it cannot close lifelong-learning claims.",
                complete_service_evidence_score="Host arithmetic cannot replace acquisition and natural learning-update joins.",
                publication_gate_results="FoVer paper readiness does not authorize publication or establish V702 science.",
            )
            progress("before_primary_publication", 13, 1)
            if args.worker_input:
                atomic_json(output, value)
            else:
                publication_receipt = publish_primary(output, value, terminal)
                atomic_json(
                    raw / "terminal_validation.json",
                    dict(
                        publication=publication_receipt,
                        normal_process_exit=measurement,
                        required_checks_passed=passed,
                    ),
                )
        progress("complete", 13, 0)
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
