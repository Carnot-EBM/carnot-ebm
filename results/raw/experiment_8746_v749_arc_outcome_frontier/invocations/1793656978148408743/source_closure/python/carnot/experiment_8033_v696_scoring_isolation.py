"""REQ-REPORT-8033: publish current scoring qualification with durable operands.

An execution condition can qualify numerical readiness without supporting a
correctness claim. Historical failures retain their original identities.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.inference import scoring_isolation_8033 as s
from carnot.inference import likelihood_isolation_runtime_8033 as runtime
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import reference, checked
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.experiment_8023_v695_likelihood_calibration import terminal_readers
from carnot.experiment_8011_v694_qwen_source_sensitivity import verify_references

Json = dict[str, Any]
ROOT = runtime.runtime.ROOT
NAME = "experiment_8033_v696_scoring_isolation"
TASK = "exp8033-scoring-isolation"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/inference/scoring_isolation_8033.py",
    "python/carnot/inference/likelihood_isolation_runtime_8033.py",
    CLI,
]
TEST = "tests/python/test_scoring_isolation_8033.py"


def operand(path: Path, field: str, expected: Any, observed: Any, upstream: str) -> Json:
    """A missing field is a contract failure, never an observed numerical zero."""
    return dict(
        check_name=field,
        upstream_id=upstream,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def authenticate(root: Path) -> Json:
    """Read only historical public token manifests; outcomes cannot select sources."""
    path = root / "results/experiment_8022_v695_likelihood_protocol.json"
    prior = root / "results/experiment_8023_v695_likelihood_calibration.json"
    value = json.loads(path.read_text()) if path.is_file() else {}
    historical = json.loads(prior.read_text()) if prior.is_file() else {}
    checks = [
        operand(path, k, v, value.get(k, "missing_field_contract_error"), "exp8022")
        for k, v in dict(
            experiment_id=8022,
            likelihood_protocol_ready_score=1,
            token_scoring_ready_score=1,
            flagged_adversarial=False,
        ).items()
    ]
    checks += [
        operand(prior, k, v, historical.get(k, "missing_field_contract_error"), "exp8023")
        for k, v in dict(experiment_id=8023, forward_pass_counts=384).items()
    ]
    panel = (
        s.select(value["public_panel_manifest"]["rows"]) if all(c["passed"] for c in checks) else []
    )
    return dict(
        checks=checks,
        panel=panel,
        methods=s.METHODS,
        gguf_sha256=value.get("gguf_sha256"),
        references=[dict(reference(p), scope="historical") for p in (path, prior) if p.is_file()],
        historical_failure=dict(
            verdict=historical.get("honest_verdict"),
            max_duplicate_drift=max(
                (r["drift"] for r in historical.get("duplicate_drift_rows", [])), default=None
            ),
            forward_pass_counts=historical.get("forward_pass_counts"),
            imported_current_calls=0,
        ),
    )


def commands(scratch: Path) -> list[CommandSpec]:
    """Keep coverage private and include actual script branches in the same total."""
    (scratch / "pytest").mkdir(parents=True, exist_ok=True)
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    specs = build_scoped_commands(
        ROOT,
        [TEST],
        OWNED[:-1],
        static_paths=[CLI],
        basetemp=scratch / "pytest",
        coverage_file=scratch / ".coverage",
    )
    specs = [
        replace(x, scope="owned", timeout_s=180)
        for x in specs
        if not x.name.startswith("changed_module_coverage")
    ]
    py = str(ROOT / ".venv/bin/python")
    specs[2:2] = [
        CommandSpec(
            "changed_code_coverage",
            (
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                TEST,
            ),
            "owned",
            180,
        ),
        CommandSpec(
            "combine", (py, "-m", "coverage", "combine", "--rcfile=" + str(config)), "owned", 30
        ),
        CommandSpec(
            "coverage_json",
            (
                py,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(scratch / "coverage.json"),
                "--fail-under=100",
            ),
            "owned",
            30,
        ),
    ]
    specs = [
        replace(
            x,
            argv=(
                x.argv[0],
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *OWNED[:-1],
            ),
        )
        if x.name == "changed_module_mypy"
        else x
        for x in specs
    ]
    specs.append(
        CommandSpec(
            "consumer_contracts",
            (
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_likelihood_protocol_8022.py",
                "tests/python/test_primary_publication_7928.py",
            ),
            "owned",
            180,
        )
    )
    specs.append(
        CommandSpec(
            "repository_health",
            (
                "timeout",
                "--kill-after=5s",
                "60s",
                str(ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
            ),
            "repository_health",
            60,
        )
    )
    return specs


def relocate(value: Any, source: Path, destination: Path) -> Any:
    """Relocated receipts name preserved bytes rather than expired child scratch."""
    if isinstance(value, dict):
        return {k: relocate(v, source, destination) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, source, destination) for v in value]
    return value.replace(str(source), str(destination)) if isinstance(value, str) else value


def build(
    plan: Json, measured: Json, raw: Path, receipts: list[Json], coverage: Json, elapsed: float
) -> Json:
    """Qualification requires complete changed execution and every owned check."""
    rows = measured["qualification"]["rows"]
    reduced = s.reduce(plan["panel"], rows)
    owned = [r for r in receipts if r["scope"] == "owned"]
    failures = [
        dict(
            r,
            check_name=r.get("check_name", r["artifact_field"]),
            sha256=r.get("sha256", r.get("hash")),
        )
        for r in plan["checks"] + measured.get("checks", [])
        if not r["passed"]
    ]
    expected_checks = [
        r["name"]
        for r in json.loads((raw / "validation_commands.json").read_text())["commands"]
        if r["scope"] == "owned"
    ]
    counts = measured.get("model_invocation_counts", ZERO_INVOCATION_COUNTS)
    gates = dict(
        complete=reduced["complete"] and len(plan["panel"]) == 8,
        changed_condition=reduced["selected_condition"] is not None,
        owned_checks=bool(expected_checks)
        and [r["name"] for r in owned] == expected_checks
        and all(r["passed"] for r in owned),
        coverage=coverage.get("percent_covered") == 100,
        current_runtime=counts["model_loads_completed"] >= 2
        and measured.get("cleanup", {}).get("model_closed", False)
        and measured.get("cleanup", {}).get("lease_released", False)
        and measured.get("offload_evidence", {}).get("supported", False),
        measured_floor=measured.get("duration_s", 0) >= 2,
    )
    ready = int(not failures and all(gates.values()))
    verdict = (
        "blocked"
        if failures or not gates["complete"]
        else "disqualified"
        if not all(
            gates[k] for k in ("owned_checks", "coverage", "current_runtime", "measured_floor")
        )
        else "null"
    )
    if not gates["complete"]:
        failures.append(
            operand(
                raw / "capture.json",
                "complete_target_capture",
                dict(targets=128, calls=132, complete=True),
                dict(
                    targets=sum(
                        r["status"] == "completed" and r["purpose"] == "target" for r in rows
                    ),
                    calls=sum(r["status"] == "completed" for r in rows),
                    complete=reduced["complete"],
                ),
                "exp8033_current",
            )
        )
    targets = [r for r in rows if r["purpose"] == "target"]
    sizes = dict(
        intended_count=128,
        eligible_count=128 if plan["panel"] else 0,
        completed_count=sum(r["status"] == "completed" for r in targets),
        excluded_count=0,
        failed_count=sum(r["status"] == "failed" for r in targets),
        censored_count=128 - sum(r["status"] in {"completed", "failed"} for r in targets),
        independent_count=8 if reduced["complete"] and plan["panel"] else 0,
    )
    value = dict(
        experiment_id=8033,
        task_id=TASK,
        milestone="2026.10.696",
        run_date="20261002",
        schema="carnot.v696.scoring_isolation.v1",
        honest_verdict="complete_scoring_isolation_blocked"
        if verdict == "blocked" and counts["model_loads_attempted"]
        else f"complete_{verdict}_scoring_isolation",
        verdict_class=verdict,
        claim_scope="Current eight exposed fit sources qualify numerical execution only; cache causality, correctness and deployment benefit remain unproven.",
        gate_check_summary=failures,
        preconditions_checked=plan["checks"] + measured.get("checks", []),
        rows=rows,
        sample_size_budget=dict(
            **sizes,
            target_pass_budget=128,
            scored_token_budget=80000,
            conditioning_passes=4,
            unit="target_forward; independent unit is original source",
            seeds_are_independent=False,
        ),
        **sizes,
        random_seed=69633,
        reproducibility_checksum=canonical_hash(dict(plan=plan, rows=rows)),
        cited_upstream_artifacts=plan["references"],
        code_config_hashes=plan["code_config_hashes"],
        raw_shard_hashes=[
            reference(raw / p)
            for p in ("plan.json", "capture.json", "validation_receipts.json", "coverage.json")
        ],
        checkpoint_references=[
            reference(p) for p in sorted((raw / "runtime/forwards").glob("*.json"))
        ],
        acceptance_gate_results=gates,
        verifier_is_oracle=False,
        genuine_headroom=None,
        positive_control_results=s.oracle(),
        generalized_learning_benefit_score=0,
        validation_receipts=owned,
        repository_health=[r for r in receipts if r["scope"] == "repository_health"],
        coverage_statement_counts=coverage,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        flagged_adversarial=False,
        inference_substrate="live_llm_inference",
        inference_mode="live_gpu",
        inference_substrate_class="model_load_no_generation"
        if counts["model_loads_attempted"]
        else "blocked_no_run",
        MODEL_SPECS=[runtime.runtime.MODEL],
        model_specs=[runtime.runtime.MODEL],
        trained_head_specs=[],
        model_invocation_counts={
            **{k: v for k, v in counts.items() if not k.startswith("generation_calls_")},
            "generation": {
                k.removeprefix("generation_calls_"): v
                for k, v in counts.items()
                if k.startswith("generation_calls_")
            },
        },
        current_invocation_ledger=measured.get("current_invocation_ledger", []),
        duration_s=elapsed,
        measured_duration_s=measured.get("duration_s", 0),
        phase_spans=measured.get("phase_spans", []),
        scoring_isolation_ready_score=ready,
        scoring_config_hash=plan["scoring_config_hash"],
        qualified_capture_implementation_hashes=[plan["scoring_config_hash"]] * 2 if ready else [],
        substrate_declaration=dict(
            substrate="live_llm_inference",
            inference_mode="live_gpu",
            mode="teacher_forced_only",
            sampled_tokens=0,
            diagnostic_floor_s=2,
        ),
        gpu_lease_receipt=measured.get("gpu_lease_receipt", {}),
        offload_evidence=measured.get("offload_evidence", {}),
        model_identity_receipt=measured.get("model_identity_receipt", {}),
        generated_tokens=0,
        historical_failure=plan["historical_failure"],
        methods=s.METHODS,
        **reduced,
    )
    value["field_principles"] = {
        k: "Bind this invocation's claims to original bytes; numerical readiness gives no independent science credit."
        for k in value
    }
    return value


def replay(value: Json) -> None:
    """Cold reduction authenticates original checkpoints and every claim field."""
    for ref in (
        value["raw_shard_hashes"]
        + value["checkpoint_references"]
        + value["code_config_hashes"]
        + value["cited_upstream_artifacts"]
    ):
        checked(ref)
    verify_references(value["validation_receipts"] + value["repository_health"])
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    plan, measured, receipts, coverage = [
        json.loads((raw / p).read_text())
        for p in ("plan.json", "capture.json", "validation_receipts.json", "coverage.json")
    ]
    checkpoints = [
        json.loads(p.read_text()) for p in sorted((raw / "runtime/forwards").glob("*.json"))
    ]
    if checkpoints != measured["qualification"]["rows"]:
        raise ValueError("checkpoint_drift")
    expected = build(plan, measured, raw, receipts["receipts"], coverage, value["duration_s"])
    if value != expected:
        raise ValueError("cold_reduction_drift")


def terminal(path: Path) -> Json:
    """Terminal consumers inspect the same bytes that cold reduction accepts."""
    replay(json.loads(path.read_text()))
    return terminal_readers(path)


def main(argv: list[str] | None = None) -> int:
    """Freeze producer bytes before an owned child; finish with atomic publication."""
    started = time.monotonic()
    s.progress("8033_start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument("--cold-replay", type=Path)
    routes.add_argument("--runtime-child", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            return 0
        if args.runtime_child:
            runtime.worker(args.runtime_child, args.output)
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        raw.mkdir(parents=True, exist_ok=True)
        if (raw / "capture.json").exists():
            raise ValueError("existing_capture_preserved_use_cold_replay")
        plan = authenticate(args.root)
        plan["code_config_hashes"] = [
            reference(ROOT / p)
            for p in OWNED
            + [
                TEST,
                "python/carnot/inference/fixed_answer_likelihood_8022.py",
                "python/carnot/inference/likelihood_runtime_8022.py",
            ]
        ]
        plan["scoring_config_hash"] = canonical_hash(
            dict(methods=s.METHODS, code=plan["code_config_hashes"], panel=plan["panel"])
        )
        atomic_json(raw / "plan.json", plan)
        with TemporaryDirectory(prefix="carnot-8033-") as directory:
            scratch = Path(directory)
            specs = commands(scratch)
            atomic_json(raw / "validation_commands.json", dict(commands=[asdict(x) for x in specs]))
            measured: Json = {}
            phase_started = time.monotonic()
            if all(r["passed"] for r in plan["checks"]):
                child_output = scratch / "runtime/capture.json"
                child = run_commands(
                    ROOT,
                    [
                        CommandSpec(
                            "owned_scoring_child",
                            (
                                str(ROOT / ".venv/bin/python"),
                                "-u",
                                CLI,
                                "--runtime-child",
                                str(raw / "plan.json"),
                                "--output",
                                str(child_output),
                            ),
                            "runtime",
                            960,
                        )
                    ],
                    log_dir=raw / "runtime_logs",
                    heartbeat_s=30,
                    extra_env=dict(
                        PYTHONUNBUFFERED="1", CARNOT_FORCE_LIVE="1", JAX_PLATFORMS="cpu"
                    ),
                )[0]
                if (scratch / "runtime").is_dir():
                    shutil.copytree(scratch / "runtime", raw / "runtime", dirs_exist_ok=True)
                measured = (
                    relocate(
                        json.loads(child_output.read_text()), scratch / "runtime", raw / "runtime"
                    )
                    if child_output.is_file()
                    else {}
                )
                measured["child_receipt"] = child
                if not child["passed"]:
                    measured.setdefault("checks", []).append(
                        operand(
                            ROOT / child["log_path"],
                            "child_exit",
                            0,
                            child["exit_code"],
                            "owned_runtime",
                        )
                    )
            if not measured.get("qualification", {}).get("rows"):
                rows = [
                    json.loads(p.read_text())
                    for p in sorted((raw / "runtime/forwards").glob("*.json"))
                ]
                if not rows:
                    rows = s.capture(plan["panel"], None, raw / "runtime/forwards", deadline=0)
                measured["qualification"] = dict(rows=rows)
            measured["phase_spans"] = [
                dict(
                    phase="owned_model_work",
                    start_s=phase_started - started,
                    end_s=time.monotonic() - started,
                )
            ]
            atomic_json(raw / "capture.json", measured)
            for p in (raw / "runtime").rglob("*.json"):
                atomic_json(
                    p, relocate(json.loads(p.read_text()), scratch / "runtime", raw / "runtime")
                )
            receipts = (
                run_commands(
                    ROOT,
                    specs,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=30,
                    extra_env=dict(
                        CARNOT_8033_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        JAX_PLATFORMS="cpu",
                    ),
                )
                if args.root == ROOT
                else []
            )
            coverage = (
                json.loads((scratch / "coverage.json").read_text())["totals"]
                if (scratch / "coverage.json").is_file()
                else {}
            )
            atomic_json(raw / "coverage.json", coverage)
            atomic_json(raw / "validation_receipts.json", dict(receipts=receipts))
            value = build(plan, measured, raw, receipts, coverage, time.monotonic() - started)
            candidate = raw / "candidate.json"
            atomic_json(candidate, value)
            cold = run_commands(
                ROOT,
                [
                    CommandSpec(
                        "cold_reduction",
                        (
                            str(ROOT / ".venv/bin/python"),
                            "-u",
                            CLI,
                            "--cold-replay",
                            str(candidate),
                        ),
                        "terminal",
                        120,
                    )
                ],
                log_dir=raw / "cold_logs",
            )
            if not all(r["passed"] for r in cold):
                raise ValueError("cold_reduction_failed")
            publication = publish_primary(output, value, terminal)
            report = terminal(output)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="scoring_isolation_ready_score",
                expected=value["scoring_isolation_ready_score"],
            )
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, cold=cold, published=report, readers=readers),
            )
            if not report["passed"] or not readers["passed"]:
                raise ValueError("published_validation_failed")
        s.progress("8033_complete", started, value["completed_count"], 0)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        print(f"[exp8033] rejected={type(error).__name__}:{error}", flush=True)
        return 1
