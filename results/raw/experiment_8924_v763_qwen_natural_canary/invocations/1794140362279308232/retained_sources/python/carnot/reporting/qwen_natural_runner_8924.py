"""REQ-REPORT-8924: publish checked canary evidence with a cold replay path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.inference.qwen_natural_runtime_8924 import run
from carnot.reporting import qwen_natural_canary_8924 as c
from carnot.reporting.bfs_error_pilot_8918 import terminal_passed
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import ExperimentTemplate
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]
ROOT = c.ROOT
CLI = "scripts/experiments/" + c.NAME + ".py"
TEST = "tests/python/test_qwen_natural_canary_8924.py"
OWNED = [
    "python/carnot/reporting/qwen_natural_canary_8924.py",
    "python/carnot/reporting/qwen_natural_runner_8924.py",
    "python/carnot/inference/qwen_natural_runtime_8924.py",
    CLI,
]


def checksum(value: Json) -> str:
    """Bind the complete primary without including its own digest."""
    return str(
        c.canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})
    )


def validation_plan(private: Path) -> list[CommandSpec]:
    """Measure new unit and CLI child statements without excluding any lines."""
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "".join("    */" + path + "\n" for path in OWNED)
        + "[report]\nexclude_lines =\n"
    )
    mypy = private / "mypy.ini"
    mypy.write_text("[mypy]\nstrict = true\nfollow_imports = skip\nignore_missing_imports = true\n")
    py, cov = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/coverage")
    data = "--data-file=" + str(private / ".coverage")
    rc = "--rcfile=" + str(config)
    commands = [
        CommandSpec(
            "owned_unit_child",
            (cov, "run", rc, data, "-m", "pytest", "-n0", "-o", "addopts=", "--no-cov", "-q", TEST),
            "owned",
            240,
        ),
        CommandSpec("coverage_combine", (cov, "combine", data, str(private)), "owned", 30),
        CommandSpec(
            "coverage_json",
            (cov, "json", rc, data, "-o", str(private / "coverage.json"), "--fail-under=100"),
            "owned",
            30,
        ),
        CommandSpec(
            "ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST), "owned", 30
        ),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST),
            "owned",
            30,
        ),
        CommandSpec(
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--config-file=" + str(mypy), *OWNED),
            "owned",
            90,
        ),
        CommandSpec("spec_coverage", (py, "scripts/check_spec_coverage.py", TEST), "owned", 30),
    ]
    for name, test in [
        ("e2e018", "tests/python/test_experiment_7891_v685_authority_lifecycle.py"),
        ("e2e021", "tests/python/test_restricted_decision_audit_8210.py"),
    ]:
        commands.append(
            CommandSpec(
                name,
                (
                    py,
                    "-u",
                    "-m",
                    "pytest",
                    "-n0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "--basetemp=" + str(private / name),
                    test,
                ),
                "private_e2e",
                240,
            )
        )
    return commands


def validate_owned(raw: Path) -> tuple[list[Json], Json]:
    """Keep raw exits and executed code hashes beside measured coverage."""
    private = Path(tempfile.mkdtemp(prefix="carnot8924-checks-"))
    receipts = run_commands(
        ROOT, validation_plan(private), log_dir=raw / "validation", heartbeat_s=15
    )
    hashes = [c.reference(ROOT / path) for path in OWNED + [TEST]]
    for receipt in receipts:
        receipt["executed_source_hashes"] = hashes
    coverage: Json = {}
    path = private / "coverage.json"
    if path.is_file():
        coverage = json.loads(path.read_text())["totals"]
        (raw / "coverage.json").write_bytes(path.read_bytes())
    for name in ("coverage.ini", "mypy.ini"):
        (raw / name).write_bytes((private / name).read_bytes())
    return receipts, coverage


def build(
    measurement: Json,
    refs: list[Json],
    sources: list[Json],
    receipts: list[Json],
    coverage: Json,
    duration: float,
) -> Json:
    """Only completed owned checks can qualify operational paired evidence."""
    runtime = measurement["runtime"]
    reduced = c.reduce(measurement["cases"], runtime["outputs"])
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and coverage.get("percent_covered") == 100
        and coverage.get("excluded_lines") == 0
        and not runtime.get("owned_failure")
    )
    model = runtime["model_receipt"]
    offload = model.get("offload_layers", [0, 0])
    live = bool(
        runtime["live"]
        and model.get("authenticated")
        and model.get("gguf_sha256") == model.get("preflight_gguf_sha256")
        and offload[0] > 0
        and offload[0] == offload[1]
        and runtime["counts"]["model_loads_completed"] == 1
        and runtime["counts"]["generation_calls_completed"] > 0
        and 10 <= runtime.get("generation_duration_s", 0) < 1800
    )
    ready, cls, verdict = c.verdict(reduced, owned=owned, live=live, blocker=runtime["blocker"])
    counts = runtime["counts"]
    substrate = (
        "live_llm_inference"
        if counts["generation_calls_attempted"]
        else "model_loading"
        if counts["model_loads_attempted"]
        else "blocked_no_run"
    )
    value: Json = ExperimentTemplate(
        8924, "Source-relative Qwen paired-criterion canary", "results/" + c.NAME + ".json"
    ).build_result(
        dict(
            experiment_id=8924,
            task_id=c.TASK,
            milestone="2026.11.763",
            run_date="20261108",
            honest_verdict=verdict,
            verdict_class=cls,
            verifier_is_oracle=False,
            exposure_scope="frozen_24_case_paired_dependent_annotations_only",
            inference_substrate=substrate,
            inference_substrate_class="model_bounded_generation"
            if counts["generation_calls_attempted"]
            else substrate,
            execution_venue="host",
            MODEL_SPECS=[dict(hf_id=c.SETTINGS["model"], format="GGUF", quantization="Q4_K_M")],
            model_invocation_counts=counts,
            numerical_training_calls=0,
            model_invoked=bool(counts["model_loads_attempted"]),
            duration_s=duration,
            phase_spans=runtime.get("phase_spans", []),
            preconditions_checked=runtime.get("preconditions_checked", {}),
            random_seed=c.SEED,
            source_artifact_hashes=sources,
            code_config_hashes=measurement.get("code_config_hashes", []),
            raw_shard_hashes=refs,
            cited_upstream_artifacts=measurement.get("upstream_references", []),
            canary_manifest=measurement.get("manifest", []),
            model_receipt=runtime["model_receipt"],
            resource_lease=runtime.get("resource_lease", {}),
            backend_receipt=runtime.get("backend", {}),
            generator_settings=c.SETTINGS,
            generation_duration_s=runtime.get("generation_duration_s", 0),
            generation_floor_s=10,
            inference_cap_s=1800,
            gate_check_summary=runtime["checks"],
            acceptance_gates=dict(
                owned_checks=owned,
                actual_generation=runtime["live"],
                all_cases_completed=reduced["completed_count"] == 24,
            ),
            required_checks_passed=owned,
            validation_receipts=receipts,
            owned_statement_coverage=coverage,
            natural_canary_ready_score=ready,
            flagged_adversarial=False,
            **reduced,
        ),
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
    )
    value["field_principles"] = {
        key: "This invocation measures operational extraction and dependent criterion sensitivity only; no independent natural-language generalization."
        for key in value
    }
    value["field_principles"].update(
        source_span_rows="Exact offsets locate retained source bytes; semantic correctness is unmeasured.",
        rows="One paired criterion and metric per frozen case; repeated criteria do not increase independent n.",
        natural_canary_ready_score="Complete operational paired evidence, independent of output quality or scientific benefit.",
        validation_receipts="Owned checks retain exits and executed-source hashes; final CLI and validator receipts are bound in the primary publication sidecar.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute every comparison from retained bytes and bind actual call evidence."""
    value = json.loads(path.read_text())
    if value["reproducibility_checksum"] != checksum(value):
        raise ValueError("checksum_drift")
    for ref in (
        value["raw_shard_hashes"] + value["source_artifact_hashes"] + value["code_config_hashes"]
    ):
        c.verify_reference(ref)
    operand = next(
        ref for ref in value["raw_shard_hashes"] if Path(ref["path"]).name == "measurement.json"
    )
    measurement = json.loads(Path(operand["path"]).read_text())
    runtime = measurement["runtime"]
    if runtime["counts"]["generation_calls_attempted"]:
        original = c.authenticate(ROOT / "results")
        if (
            measurement["cases"] != original["cases"]
            or measurement.get("manifest") != original["manifest"]
            or value["canary_manifest"] != original["manifest"]
        ):
            raise ValueError("frozen_source_drift")
        ledger_ref = next(
            ref
            for ref in value["raw_shard_hashes"]
            if Path(ref["path"]).name == "invocation_ledger.json"
        )
        ledger = Ledger(Path(ledger_ref["path"]))
        if ledger.rows != runtime["ledger"] or ledger.counts() != runtime["counts"]:
            raise ValueError("ledger_drift")
    reduced = c.reduce(measurement["cases"], runtime["outputs"])
    if any(value[key] != actual for key, actual in reduced.items()):
        raise ValueError("reduction_drift")
    if (
        value["model_invocation_counts"] != runtime["counts"]
        or value["model_receipt"] != runtime["model_receipt"]
        or value["generator_settings"] != c.SETTINGS
    ):
        raise ValueError("runtime_binding_drift")
    calls = {row["call_id"]: row for row in runtime["ledger"]}
    for output in runtime["outputs"]:
        if output.get("status", "completed") == "completed" and runtime["live"]:
            call = calls[output["unit_id"]]
            case = next(row for row in measurement["cases"] if row["unit_id"] == output["unit_id"])
            if "request_sha256" in call and call["request_sha256"] != c.canonical_hash(
                dict(c.SETTINGS, messages=[dict(role="user", content=c.prompt(case))])
            ):
                raise ValueError("prompt_binding_drift")
            if (
                call["response_sha256"] != c.canonical_hash(output["raw_response"])
                or output["answer"]
                != (output["raw_response"]["choices"][0]["message"].get("content") or "")
                or output["usage"] != output["raw_response"]["usage"]
            ):
                raise ValueError("output_binding_drift")
    expected = build(
        measurement,
        value["raw_shard_hashes"],
        value["source_artifact_hashes"],
        value["validation_receipts"],
        value["owned_statement_coverage"],
        value["duration_s"],
    )
    for key in [
        "natural_canary_ready_score",
        "verdict_class",
        "required_checks_passed",
        "acceptance_gates",
        "honest_verdict",
    ]:
        if value[key] != expected[key]:
            raise ValueError("qualification_drift")
    return dict(passed=True, replay_passed=True, completed_count=reduced["completed_count"])


def terminal_commands(candidate: Path) -> list[CommandSpec]:
    """Invoke the real CLI outside the checkout, then both unchanged validators."""
    py = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay_cli",
            (
                py,
                "-u",
                "-c",
                "import os,subprocess,sys,tempfile; env=dict(os.environ); env.pop('PYTHONPATH',None); raise SystemExit(subprocess.run(sys.argv[1:],cwd=tempfile.gettempdir(),env=env,timeout=45).returncode)",
                py,
                "-u",
                str(ROOT / CLI),
                "--cold-replay",
                str(candidate),
            ),
            "terminal",
            60,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
            "terminal",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal",
            60,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    """Validate before expensive work and publish only a checked terminal primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        try:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        except (ValueError, KeyError, OSError, StopIteration, c.ExternalGate) as error:
            print(json.dumps(dict(passed=False, error=str(error))), flush=True)
            return 1
    if args.date != "20261108":
        parser.error("this invocation requires --date 20261108")
    started = time.monotonic()
    phases = []
    c.progress("authenticate_upstreams", 0, 2)
    output = ROOT / "results" / (c.NAME + ".json")
    raw = output.parent / "raw" / c.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    private = Path(tempfile.mkdtemp(prefix="carnot8924-", dir=Path.home() / ".cache"))
    private.chmod(0o700)
    runtime: Json = dict(
        outputs=[],
        counts=dict(ZERO_INVOCATION_COUNTS),
        ledger=[],
        model_receipt={},
        checks=[],
        live=False,
        blocker=None,
        owned_failure=False,
    )
    evidence: Json = dict(
        cases=[
            dict(
                unit_id=f"pending:{i}",
                cluster_id=f"unknown:{i}",
                source_task="unknown",
                source="",
                response="",
                original=[],
                revised=[],
            )
            for i in range(24)
        ],
        manifest=[],
        upstream_references=[],
    )
    try:
        evidence = c.authenticate(ROOT / "results")
    except (c.ExternalGate, OSError, ValueError, KeyError) as error:
        runtime["blocker"] = str(error)
        runtime["checks"] = [
            error.row
            if isinstance(error, c.ExternalGate)
            else dict(
                check="source_dependency",
                upstream="original_producers",
                artifact_field="retained_corpus",
                expected="authenticated source custody",
                observed=str(error),
                path=str(ROOT / "results"),
                hash=None,
            )
        ]
    phases.append(dict(phase="upstream_authentication", duration_s=time.monotonic() - started))
    c.progress("authenticate_upstreams_complete", 2, 0)
    sources = []
    for name in OWNED + [
        TEST,
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/inference/qwen_sufficiency_7920.py",
        "python/carnot/inference/bfs_pilot_runtime_8918.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]:
        retained = raw / "retained_sources" / name
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes((c.ROOT / name).read_bytes())
        sources.append(c.reference(retained))
    phase_started = time.monotonic()
    receipts, coverage = validate_owned(raw)
    phases.append(dict(phase="owned_validation", duration_s=time.monotonic() - phase_started))
    c.progress("owned_checks_complete", len(receipts), 0)
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and coverage.get("percent_covered") == 100
        and coverage.get("excluded_lines") == 0
    )
    if owned and not runtime["blocker"]:
        runtime = run(evidence, private)
    phase_started = time.monotonic()
    health = run_commands(
        c.ROOT,
        [
            CommandSpec(
                "full_python_suite",
                (
                    "timeout",
                    "--kill-after=10s",
                    "300s",
                    str(c.ROOT / ".venv/bin/pytest"),
                    "tests/python",
                    "-q",
                ),
                "repository_health",
                330,
            )
        ],
        log_dir=raw / "repository_health",
        heartbeat_s=15,
    )
    phases.append(dict(phase="repository_health", duration_s=time.monotonic() - phase_started))
    runtime.setdefault("phase_spans", []).extend(phases)
    measurement = dict(
        cases=evidence["cases"],
        manifest=evidence["manifest"],
        upstream_references=evidence["upstream_references"],
        runtime=runtime,
        code_config_hashes=sources,
    )
    atomic_json(private / "measurement.json", measurement)
    refs = [c.reference(p) for p in sorted(private.rglob("*")) if p.is_file()]
    value = build(
        measurement,
        refs,
        evidence["upstream_references"],
        receipts,
        coverage,
        time.monotonic() - started,
    )
    value["repository_health"] = dict(receipts=health, affects_owned_readiness=False)
    value["field_principles"]["repository_health"] = (
        "Unrelated repository health does not change owned experiment readiness."
    )
    value["reproducibility_checksum"] = checksum(value)
    c.progress("before_terminal_checks")

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        terminal = run_commands(
            c.ROOT,
            terminal_commands(candidate),
            log_dir=raw / "terminal_validation",
            heartbeat_s=15,
        )
        report.update(
            passed=terminal_passed(terminal), receipts=terminal, executed_source_hashes=sources
        )
        return report

    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        receipts.append(
            dict(name="terminal_validation", passed=False, observed="candidate_rejected")
        )
        value = build(
            measurement,
            refs,
            evidence["upstream_references"],
            receipts,
            coverage,
            time.monotonic() - started,
        )
        value["repository_health"] = dict(receipts=health, affects_owned_readiness=False)
        value["reproducibility_checksum"] = checksum(value)
        published = publish_primary(output, value, replay)
    c.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(json.dumps(dict(**published, honest_verdict=value["honest_verdict"])), flush=True)
    return 0
