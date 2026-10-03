"""REQ-REPORT-8076: publish auditable constraint learning without future-safety credit.

The historical numerical head is independent of the new interaction fit. A
normally exited measurement child and current checks precede primary publication.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_8064_v698_fresh_feedback_learning as prior
from carnot import experiment_8075_v699_constraint_projection_kernel as qualification
from carnot import experiment_8072_v699_sealed_methods as methods
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import projected_online_8076 as m

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8076_v699_projected_online_learning"
TASK = "exp8076-projected-online-learning"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_projected_online_8076.py"
OWNED = [MODULE, "python/carnot/verify/projected_online_8076.py", CLI]
INPUTS = [
    *qualification.INPUTS,
    "python/carnot/experiment_8064_v698_fresh_feedback_learning.py",
    "python/carnot/verify/causal_online_8025.py",
    "results/experiment_8058_v698_sealed_evidence_methods.json",
    "results/experiment_8064_v698_fresh_feedback_learning.json",
    "results/experiment_8070_v699_contract_custody.json",
    "results/experiment_8075_v699_constraint_projection_kernel.json",
]


def prerequisites(root: Path, raw: Path) -> tuple[list[Json], list[Json]]:
    """Bind every named operand and terminal sidecar, retaining exact failures."""
    refs: list[Json] = []
    failures: list[Json] = []
    for label in INPUTS:
        path = root / label
        operand: Json = dict(
            check="input_exists",
            upstream=path.stem,
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            field="exists",
            op="==",
            expected=True,
            observed=path.is_file(),
        )
        if not path.is_file():
            failures.append(operand)
            continue
        target = raw / "inputs" / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        refs.append(dict(path=str(path), snapshot_path=str(target), sha256=sha256_file(target)))
        if label.startswith("results/experiment_"):
            try:
                value = json.loads(path.read_text())
                terminal = Path(value["terminal_validation_sidecar_path"])
                binding = json.loads(terminal.read_text())["publication"]
                sidecar = Path(binding["sidecar_path"])
                report = read_bound_sidecar(path, sidecar)
                checks: list[tuple[str, Any, Any]] = [
                    ("terminal.passed", True, report["report"]["passed"]),
                    ("flagged_adversarial", False, value.get("flagged_adversarial")),
                    ("terminal.primary_hash", sha256_file(path), binding["primary_sha256"]),
                    ("terminal.primary_path", str(path.absolute()), report["primary_path"]),
                ]
                for identity, field in [
                    (8070, "historical_learning_inputs_ready_score"),
                    (8072, "learning_protocol_ready_score"),
                    (8075, "projection_kernel_ready_score"),
                ]:
                    if value["experiment_id"] == identity:
                        checks.append((field, 1, value.get(field)))
                if value["experiment_id"] == 8072:
                    checks.append(
                        ("frozen_methods", methods.methods(), value["method_freeze"]["methods"])
                    )
                for field, expected, observed in checks:
                    if expected != observed:
                        failures.append(
                            dict(
                                operand,
                                check="input_authentication",
                                field=field,
                                expected=expected,
                                observed=observed,
                            )
                        )
                for side in (terminal, sidecar):
                    frozen = raw / "inputs/sidecars" / path.stem / side.name
                    frozen.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(side, frozen)
                    refs.append(
                        dict(path=str(side), snapshot_path=str(frozen), sha256=sha256_file(frozen))
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                failures.append(
                    dict(
                        operand,
                        check="terminal_binding",
                        field="terminal_validation_sidecar_path",
                        expected="authenticated terminal bytes",
                        observed=str(error),
                    )
                )
    for name in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / name
        if not path.is_file():
            failures.append(
                dict(
                    check="required_tool",
                    upstream="python_environment",
                    path=str(path),
                    hash=None,
                    field="exists",
                    op="==",
                    expected=True,
                    observed=False,
                )
            )
    return refs, failures


def worker(root: Path, raw: Path, fixture_input: Path | None = None) -> Json:
    """Numerical execution finishes in a child before any reader-visible primary."""
    started = time.monotonic()
    m.progress("preconditions_before")
    refs, failures = prerequisites(root, raw)
    with tempfile.TemporaryDirectory(prefix="carnot-8076-environment-") as temp:
        private = Path(temp)
        spec = manifest(private)[0]
        m.progress("subprocess_before_environment", 0, 1)
        environment_receipt = run_check(private, spec, private, raw / "precondition_logs")
        m.progress("subprocess_after_environment", 1, 0)
    data: Json = {}
    if fixture_input:
        data = json.loads(fixture_input.read_text())
        refs.append(reference(fixture_input))
        failures = []
    elif not failures:
        m.progress("historical_small_head_load_before")
        data, failures = prior.load_inputs(root, raw / "historical")
        refs.extend(data.get("references", []))
        m.progress("historical_small_head_load_after", int(bool(data.get("head"))), 0)
    if not environment_receipt["passed"]:
        failures.append(
            dict(
                check="python_environment",
                upstream="python_environment",
                path=environment_receipt.get("log_path"),
                hash=environment_receipt.get("log_sha256"),
                field="exit_code",
                op="==",
                expected=0,
                observed=environment_receipt.get("exit_code"),
            )
        )
    frozen = time.monotonic()
    m.progress("preconditions_after", len(refs), len(failures))
    evidence: Json = {}
    owned_failure = False
    if not failures:
        try:
            evidence = m.measure(data, raw / "trajectory")
            m.progress(
                "retention_before", len(evidence["final_head_seals"]), len(data["retention"])
            )
            evidence["retention_rows"] = prior.retention(data, evidence, raw)
            m.progress("retention_after", len(evidence["retention_rows"]), 0)
        except (OSError, ValueError, TimeoutError) as error:
            owned_failure = True
            failures.append(
                dict(
                    check="numerical_work",
                    upstream=TASK,
                    path=str(raw / "trajectory"),
                    hash=None,
                    field="complete_trajectory",
                    op="==",
                    expected=True,
                    observed=str(error),
                    classification="owned",
                )
            )
    atomic_json(raw / "evidence.json", evidence)
    work = dict(
        data=data,
        evidence=evidence,
        failures=failures,
        owned_failure=owned_failure,
        source_artifact_hashes=refs,
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                qualification.KERNEL,
                prior.MODULE,
                prior.OWNED[2],
                "python/carnot/verify/causal_online_8025.py",
            ]
        },
        phase_spans=[
            dict(phase="preconditions", duration_s=frozen - started),
            dict(phase="numerical_and_retention", duration_s=time.monotonic() - frozen),
        ],
        duration_s=time.monotonic() - started,
        environment=dict(
            python=sys.version, executable=sys.executable, receipt=environment_receipt
        ),
        raw_shard_hashes=[
            reference(p)
            for p in sorted(raw.rglob("*"))
            if p.is_file()
            and p.name not in {"work.json", "validation_commands.json"}
            and "inputs" not in p.parts
        ],
    )
    atomic_json(raw / "work.json", work)
    return work


def manifest(private: Path) -> list[Json]:
    """Reuse qualified checks, covering only this experiment's added statements."""
    specs = qualification.manifest(private)
    replacements = [*zip(qualification.OWNED, OWNED, strict=True), (qualification.TEST, TEST)]
    for spec in specs:
        argv = []
        for argument in spec["argv"]:
            for before, after in replacements:
                argument = argument.replace(before, after)
            argv.append(argument)
        spec["argv"] = argv
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    specs.insert(
        0,
        dict(
            name="python_environment",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-c",
                "import sys,numpy,scipy,pytest,coverage,ruff,mypy; assert sys.version_info >= (3,11); print(sys.version,numpy.__version__,scipy.__version__,flush=True)",
            ],
            deadline_s=60,
            expected_exit=0,
            classification="required",
        ),
    )
    return specs


def build(work: Json, raw: Path, receipts: list[Json], coverage: Json, *, fixture: bool) -> Json:
    """An audited null trajectory can qualify causality without claiming benefit."""
    required = [r for r in receipts if r.get("classification") != "diagnostic"]
    frozen = raw / "validation_commands.json"
    expected = ["measurement_normal_exit"]
    if frozen.is_file():
        expected = [
            r["name"]
            for r in json.loads(frozen.read_text())["commands"]
            if r.get("classification") != "diagnostic"
        ]
    else:
        with tempfile.TemporaryDirectory(prefix="carnot-8076-plan-") as temp:
            expected += [
                r["name"] for r in manifest(Path(temp)) if r.get("classification") != "diagnostic"
            ]
    covered = all(p in coverage and coverage[p]["missing_lines"] == 0 for p in OWNED)
    passed = (
        [r["name"] for r in required] == expected and all(r["passed"] for r in required) and covered
    )
    evidence = work["evidence"]
    failures = list(work["failures"])
    kind = (
        "disqualified"
        if work["owned_failure"]
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
        if passed and evidence
        else "disqualified"
    )
    if not failures and not passed and (not fixture or work["owned_failure"]):
        failures += [
            dict(
                check=r["name"],
                upstream=TASK,
                path=r.get("log_path"),
                hash=r.get("log_sha256"),
                field="exit_code",
                op="==",
                expected=r.get("expected_exit", 0),
                observed=r.get("exit_code"),
            )
            for r in required
            if not r["passed"]
        ]
        if not covered:
            failures.append(
                dict(
                    check="owned_statement_coverage",
                    upstream=TASK,
                    path=str(raw / "coverage.json"),
                    hash=None,
                    field="missing_lines",
                    op="==",
                    expected=0,
                    observed=coverage,
                )
            )
    value: Json = dict(
        experiment_id=8076,
        task_id=TASK,
        milestone="2026.10.699",
        schema="carnot.v699.projected_online_learning.v1",
        run_date="20261003",
        honest_verdict="complete_" + kind + "_projected_online_learning",
        verdict_class=kind,
        verifier_is_oracle=fixture,
        claim_scope="Causal finite exposed development trajectory only. No future safety theorem, H2 significance, generalized improvement, live generation or deployment credit.",
        flagged_adversarial=False,
        required_checks_passed=passed,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if evidence
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        rows=[],
        intended_count=256,
        eligible_count=0,
        independent_count=0,
        completed_count=0,
        censored_count=0,
        excluded_count=256,
        failed_count=0,
        sample_size_budget=dict(intended=256, completed=0, seeds_are_independent=False),
        gate_check_summary=failures,
        random_seed=6998076,
        source_artifact_hashes=work["source_artifact_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        duration_s=work["duration_s"],
        generalized_learning_benefit_score=0,
        learning_trajectory_ready_score=int(kind == "null" and bool(evidence)),
        trained_head_specs=[],
        config=m.CONFIG,
        frozen_methods=methods.methods(),
        methodology_note="Four own-incumbent gradients on released update-only rows; two independent newest64 memories; bounded projected endpoint versus raw ray; one-use fresh admission and shared clocks; rejected/no-op/reset work charged. No new generator training.",
        acceptance_certificate="empirical_only",
        trajectory_directory=str(raw / "trajectory"),
        retention_rows=[],
        behavior_counts={},
        cpu_update_costs=[],
        memory_bytes={},
        coverage_statement_counts=coverage,
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
        environment=work.get("environment", {}),
        validation_manifest_sha256=sha256_file(frozen) if frozen.is_file() else None,
    )
    value.update({field: [] for field in m.FIELDS.values()})
    value.update(evidence)
    if kind == "blocked":
        value["honest_verdict"] = "complete_blocked_" + str(failures[0]["upstream"]).replace(
            ".", "_"
        )
    value["trained_head_specs"] = [
        dict(
            arm=r["arm"],
            seed=r["seed"],
            head_hash=r["head_hash"],
            parameters=len(r["parameters"]),
            device="cpu",
            generator_training=False,
        )
        for r in value["final_head_seals"]
    ]
    value["substrate_declaration"] = dict(
        inference_substrate=value["inference_substrate"],
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        trained_head_specs=value["trained_head_specs"],
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            config=value["config"],
            code=value["code_config_hashes"],
            inputs=value["source_artifact_hashes"],
            raw=value["raw_shard_hashes"],
        )
    )
    value["field_principles"] = {
        k: f"Bind {k} to exact owned observations so cached development repetitions cannot become independent benefit claims."
        for k in value
    }
    value["field_principles"].update(
        learning_trajectory_ready_score="Causal completeness permits a null result and does not imply future improvement.",
        behavior_counts="No-op, rejected and reset events cannot count as beneficial learning.",
        cpu_update_costs="Charge projection, memory, validation, fallback and durable storage separately from matched gradients.",
        memory_bytes="Bound active memory while accounting for the larger retained audit journal separately.",
        per_arm_gradient_label_operation_budgets="Matching gradients and labels does not claim matching total compute.",
        admission_consumption_rows="One-use release-order evidence prevents repeated or favorable-label selection.",
        candidate_commit_rows="Hash endpoints, incumbents and memory before fresh labels to prevent hindsight selection.",
        final_head_seals="Retention labels cannot influence the sealed learned head.",
        repository_health="Global health failures remain visible separately from current owned qualification.",
    )
    return value


def replay(path: Path) -> bool:
    """Cold equations and byte checks reject altered primitives or readiness."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        if any(sha256_file(ROOT / p) != h for p, h in value["code_config_hashes"].items()):
            return False
        validation = json.loads((raw / "validation.json").read_text())
        for receipt in validation["receipts"]:
            if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
                return False
        work = json.loads((raw / "work.json").read_text())
        if work["evidence"]:
            reduced = m.reduce(raw / "trajectory")
            if any(work["evidence"][k] != v for k, v in reduced.items()):
                return False
            with tempfile.TemporaryDirectory(prefix="carnot-8076-retention-") as temp:
                retention = prior.retention(work["data"], reduced, Path(temp))
            if work["evidence"]["retention_rows"] != retention:
                return False
        return bool(
            build(
                work,
                raw,
                validation["receipts"],
                validation["coverage"],
                fixture=validation["fixture"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path) -> Json:
    """Exact candidate bytes pass cold replay and both existing auditors."""
    value = json.loads(path.read_text())
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    py = str(ROOT / ".venv/bin/python")
    commands = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    receipts: list[Json] = []
    with tempfile.TemporaryDirectory(prefix="carnot-8076-terminal-") as temp:
        for name, argv in commands:
            m.progress("subprocess_before_" + name, len(receipts), len(commands) - len(receipts))
            receipts.append(
                run_check(
                    Path(temp),
                    dict(name=name, argv=argv, deadline_s=120, expected_exit=0),
                    Path(temp),
                    raw / "terminal_logs",
                )
            )
            m.progress("subprocess_after_" + name, len(receipts), len(commands) - len(receipts))
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze validation before measurement and preserve prior primary bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    m.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            passed = replay(args.cold_replay)
            m.progress("cold_replay_passed" if passed else "cold_replay_rejected")
            return 0 if passed else 1
        if args.worker_output:
            worker(args.root, args.worker_output.parent, args.fixture_input)
            return 0
        output = (args.fixture_output or args.output).absolute()
        raw = output.parent / "raw" / output.stem
        if output.exists() or (raw / "work.json").exists():
            m.progress("existing_terminal_evidence_preserved")
            return 1
        with tempfile.TemporaryDirectory(prefix="carnot-8076-") as temp:
            private = Path(temp)
            specs = manifest(private)
            py = str(ROOT / ".venv/bin/python")
            command = [
                py,
                "-u",
                str(ROOT / CLI),
                "--root",
                str(args.root),
                "--worker-output",
                str(raw / "work.json"),
            ]
            if args.fixture_input:
                command += ["--fixture-input", str(args.fixture_input)]
            config = os.environ.get("CARNOT_8076_COVERAGE_CONFIG")
            if config:
                command = [
                    py,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + config,
                    "--parallel-mode",
                    *command[2:],
                ]
            child = dict(
                name="measurement_normal_exit",
                argv=command,
                deadline_s=1200,
                expected_exit=0,
                classification="required",
            )
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=[child, *specs],
                    code_hashes={p: sha256_file(ROOT / p) for p in OWNED},
                    config=m.CONFIG,
                    terminal_checks=["cold_replay", "adversarial", "strict_rows"],
                ),
            )
            m.progress("subprocess_before_measurement", 0, 1)
            receipt = run_check(private, child, private, raw / "validation_logs")
            m.progress("subprocess_after_measurement", 1, 0)
            if not receipt["passed"]:
                work = dict(
                    data={},
                    evidence={},
                    failures=[],
                    owned_failure=True,
                    source_artifact_hashes=[],
                    raw_shard_hashes=[],
                    code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
                    phase_spans=[],
                    duration_s=0.0,
                )
                atomic_json(raw / "work.json", work)
            work = json.loads((raw / "work.json").read_text())
            receipts, coverage = [receipt], {}
            if not args.fixture_output:
                os.environ["CARNOT_8076_COVERAGE_CONFIG"] = str(private / "coverage.ini")
                for index, spec in enumerate(specs):
                    m.progress("subprocess_before_" + spec["name"], index, len(specs) - index)
                    receipts.append(run_check(ROOT, spec, private, raw / "validation_logs"))
                    m.progress(
                        "subprocess_after_" + spec["name"], index + 1, len(specs) - index - 1
                    )
                if (private / "coverage.json").is_file():
                    report = json.loads((private / "coverage.json").read_text())["files"]
                    coverage = {
                        str(Path(p).resolve().relative_to(ROOT)): v["summary"]
                        for p, v in report.items()
                    }
                    atomic_json(raw / "coverage.json", dict(files=report))
            atomic_json(
                raw / "validation.json",
                dict(receipts=receipts, coverage=coverage, fixture=bool(args.fixture_input)),
            )
            value = build(work, raw, receipts, coverage, fixture=bool(args.fixture_input))
            publication = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, measurement_exit_receipt=receipt),
            )
            m.progress("terminal_published", len(value["rows"]), 0)
        return 0
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        m.progress("rejected_" + str(error))
        return 1
