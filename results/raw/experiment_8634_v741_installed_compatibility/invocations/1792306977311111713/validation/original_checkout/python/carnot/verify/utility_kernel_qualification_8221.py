"""REQ-REPORT-8221: authenticate frozen operands and qualify private mechanics.

Current source snapshots are separate from historical planning prose. Each
readiness field describes execution checks, never natural learning benefit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import hard_exit_learning_qualification_8206 as crash
from carnot.verify import utility_kernel_8221 as kernel
from carnot.verify import utility_patch_methods_8219 as frozen
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = frozen.ROOT
NAME = "experiment_8221_v711_utility_kernel"
TASK = "exp8221-utility-kernel"
MODULE = "python/carnot/verify/utility_kernel_qualification_8221.py"
KERNEL = "python/carnot/verify/utility_kernel_8221.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_utility_kernel_8221.py"
OWNED = [KERNEL, MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
PROTOCOL = frozen.PROTOCOL
PIN = frozen.PIN
PROTOCOL_VALUE = frozen.PROTOCOL_VALUE
BINDINGS = "openspec/change-proposals/v711-execution-bindings.json"
UPSTREAM = "results/experiment_8219_v710_utility_patch_methods.json"
run_check = frozen.run_check
reference = frozen.reference
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts make bounded CPU work distinguishable from stalled children."""
    print(f"[exp8221] phase={phase} completed={completed} pending={pending}", flush=True)


def static_checks() -> Json:
    """Independent scalar arithmetic verifies the saved ordered patch evaluator."""
    rows, labels = kernel.fixture("learnable")
    training = [dict(r, y=y) for r, y in zip(rows[:64], labels[:64], strict=True)]
    model = kernel.fit_patches(training)
    errors = []
    scalar_errors = []
    for row in rows:
        p = row["p"]
        for operation in model["patches"]:
            p = min(
                1 - 1e-6, max(1e-6, p + operation["delta"] * frozen.member(row, operation["group"]))
            )
        q = kernel.predict(model, row)
        scalar_errors.append(abs(p - q))
        good, bad = kernel.energies(q)
        errors.append(abs(kernel.rule.probability(good, bad, 1) - q))
    return dict(
        passed=max(errors) <= 1e-10 and max(scalar_errors) == 0 and bool(model["patches"]),
        energy_probability_error=max(errors),
        scalar_probability_error=max(scalar_errors),
        model=model,
    )


def qualify(raw: Path) -> Json:
    """Reuse the qualified crash callback and five direct Coverage.py children."""
    inputs = raw / "restart-input.json"
    rows, labels = kernel.fixture("learnable")
    atomic_json(inputs, dict(rows=rows, labels=labels, seed=101))
    with patch.object(crash.legacy, "CLI", CLI):
        specs = crash.restart_specs(raw, raw)
    config = raw / "child_coverage/coverage.ini"
    config.write_text(
        config.read_text().replace(
            "[report]", "".join("    " + str(ROOT / p) + "\n" for p in OWNED) + "[report]"
        )
    )
    atomic_json(raw / "restart_commands.json", dict(commands=specs))
    receipts = []
    for spec in specs:
        receipts.append(run_check(ROOT, spec, raw, raw / "restart_logs", heartbeat_s=20))
    measured = crash.child_coverage(raw)
    parent = os.environ.get("COVERAGE_RCFILE")
    if parent:
        for shard in (raw / "child_coverage").glob(".coverage.*"):
            shutil.copyfile(
                shard, Path(parent).parent / (shard.name + "-8221-" + str(time.time_ns()))
            )
    states = []
    for slot, name in [(90, "restart90"), (170, "restart")]:
        saved = json.loads((raw / name / "crash.json").read_bytes())
        states.append(
            dict(
                measured["restart_state_hashes"][len(states)],
                pending_ids=saved["pending"],
                rng_state=saved["rng_state"],
                consumed_feedback=saved["consumed"],
                predicates=kernel.GROUPS,
                mixture=saved["model"],
            )
        )
    final = json.loads((raw / "uninterrupted/final.json").read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts)
        and measured["passed"]
        and kernel.summary(final)["passed"],
        receipts=receipts,
        restart_state_hashes=states,
        coverage=measured,
        summary=kernel.summary(final),
        final=final,
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Authenticate external bytes before private numerical and recovery measurements."""
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[], refs=[], static={}, causal={}, owned_failure="", protocol=PROTOCOL_VALUE
    )
    progress("before_preconditions")
    try:
        with TemporaryDirectory(prefix="carnot-8221-probe-") as directory:
            probe = Path(directory) / "probe"
            probe.write_bytes(b"private writable scratch")
            frozen.gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private writable scratch",
            )
        frozen.gate(
            work, Path(sys.executable), "python_supported", True, sys.version_info >= (3, 11)
        )
        frozen.bind(work, root / PROTOCOL, PIN, raw)
        path = root / UPSTREAM
        frozen.gate(work, path, "exists", True, True if path.is_file() else None)
        primary = json.loads(path.read_bytes())
        for field, expected in [
            ("utility_protocol_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
            ("protocol_sha256", PIN),
        ]:
            frozen.gate(work, path, field, expected, primary.get(field))
        terminal_path = root / Path(primary["terminal_validation_sidecar_path"]).relative_to(ROOT)
        frozen.gate(work, terminal_path, "exists", True, True if terminal_path.is_file() else None)
        terminal = json.loads(terminal_path.read_bytes())
        sidecar = root / Path(terminal["publication"]["sidecar_path"]).relative_to(ROOT)
        frozen.gate(work, sidecar, "exists", True, True if sidecar.is_file() else None)
        frozen.gate(
            work,
            path,
            "terminal_publication_passed",
            True,
            read_bound_sidecar(path, sidecar)["report"]["passed"],
        )
        for operand in [
            dict(path=str(path), sha256=sha256_file(path)),
            dict(path=str(terminal_path), sha256=sha256_file(terminal_path)),
            dict(path=str(sidecar), sha256=sha256_file(sidecar)),
            PROTOCOL_VALUE["fit_measurement"],
            PROTOCOL_VALUE["sealed_predictions"],
            PROTOCOL_VALUE["delayed_memory"]["global_optimizer_protocol"],
        ]:
            source = root / Path(operand["path"]).relative_to(ROOT)
            frozen.bind(work, source, operand["sha256"], raw)
        measurement = json.loads(
            (root / Path(PROTOCOL_VALUE["fit_measurement"]["path"]).relative_to(ROOT)).read_bytes()
        )
        sealed_path = root / Path(PROTOCOL_VALUE["sealed_predictions"]["path"]).relative_to(ROOT)
        sealed = json.loads(sealed_path.read_bytes())
        frozen.gate(work, sealed_path, "labels_opened", False, sealed.get("labels_opened"))
        frozen.gate(
            work,
            path,
            "role_manifest",
            PROTOCOL_VALUE["role_manifest"],
            measurement["evidence"]["roles"],
        )
        frozen.gate(
            work,
            path,
            "frozen_witness_dictionary",
            PROTOCOL_VALUE["witness_dictionary"],
            frozen.freeze_dictionary(PROTOCOL_VALUE["public_fit_membership"]),
        )
        if mutation:
            frozen.gate(work, path, "source_custody", "authenticated", None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="authenticated_input_schema",
                    path=str(root),
                    upstream=str(root),
                    hash=None,
                    artifact_field="authenticated_input_schema",
                    op="==",
                    expected="valid",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_preconditions", len(work["checks"]))
    if all(c["passed"] for c in work["checks"]):
        try:
            progress("before_benchmark_static")
            work["static"] = static_checks()
            progress("after_benchmark_static", 1)
            progress("before_benchmark_causal")
            work["causal"] = qualify(raw)
            progress("after_benchmark_causal", 1)
        except (OSError, ValueError, KeyError, TypeError) as error:
            work["owned_failure"] = str(error)
    work["code_config_hashes"] = []
    for name in [*OWNED, TEST, BINDINGS]:
        path = ROOT / name
        snapshot = frozen.bind(dict(checks=[], refs=[]), path, sha256_file(path), raw)
        work["code_config_hashes"].append(dict(reference(path), snapshot_path=str(snapshot)))
    atomic_json(raw / "fixture_primitives.json", dict(static=work["static"], causal=work["causal"]))
    work["raw_shard_hashes"] = [
        reference(p)
        for p in raw.rglob("*")
        if p.is_file()
        and p.name
        not in [
            "measurement.json",
            "terminal_validation.json",
            "measurement.stdout",
            "measurement.stderr",
        ]
    ]
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    work["clock"] = dict(
        started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", 2)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Static and causal readiness have separate evidence and no benefit interpretation."""
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    failures = [c for c in work["checks"] if not c["passed"]]
    static = int(checked and not failures and work["static"].get("passed", False))
    causal = int(checked and not failures and work["causal"].get("passed", False))
    verdict = "disqualified" if not checked else "blocked" if failures else "circular_positive"
    operand = Path(failures[0]["path"]).stem.lower() if failures else "utility_kernel"
    rows = [
        dict(
            unit_id=name,
            source_cluster_id="private-" + name,
            arm=name,
            condition="private_mechanics",
            metric="kernel_ready",
            numerator=score,
            denominator=1,
            status="completed" if score else "excluded",
            exclusion_reason=None if score else "mechanics_unavailable",
            semantic_metric=None,
        )
        for name, score in [("static", static), ("causal", causal)]
    ]
    value: Json = dict(
        experiment_id=8221,
        task_id=TASK,
        milestone="2026.10.711",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_" + operand,
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads=0, generate_calls=0, forward_calls=0, model_count=0
        ),
        call_ledger=[],
        trained_head_specs=[
            dict(
                kind="finite_probability_patch",
                current_fit=True,
                maximum_patches=4,
                scope="private_deterministic_fixture",
            )
        ],
        rows=rows,
        intended_count=2,
        completed_count=static + causal,
        failed_count=0,
        censored_count=0,
        excluded_count=2 - static - causal,
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="private_deterministic_mechanics",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        flagged_adversarial=False,
        required_checks_passed=checked,
        acceptance_gates=dict(
            owned_validation=checked,
            input_authentication=not failures,
            static_mechanics=bool(static),
            causal_mechanics=bool(causal),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=[
            dict(resource=c["check"], available=c["passed"], path=c["path"]) for c in work["checks"]
        ],
        duration_s=work["duration_s"],
        random_seed=101,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        kernel_code_snapshots=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        phase_spans=[
            dict(phase="authenticate_and_qualify", **work["clock"], duration_s=work["duration_s"])
        ],
        measurement_reference=reference(raw / "measurement.json"),
        cited_upstream_artifacts=[
            dict(
                path=r["upstream_path"],
                sha256=r["sha256"],
                fields_imported=["protocol readiness and terminal receipt"]
                if r["upstream_path"].endswith(Path(UPSTREAM).name)
                else ["frozen named input bytes"],
            )
            for r in work["refs"]
        ],
        static_kernel_ready_score=static,
        causal_kernel_ready_score=causal,
        protocol_sha256=PIN,
        execution_bindings_path=str(ROOT / BINDINGS),
        state_schema=kernel.SCHEMA,
        crash_controls={k: v for k, v in work["causal"].items() if k != "final"},
        energy_probability_error=work["static"].get("energy_probability_error"),
        static_checks=work["static"],
        H1=PROTOCOL_VALUE.get("H1"),
        H2=PROTOCOL_VALUE.get("H2"),
        primary_comparator_selection=PROTOCOL_VALUE.get("comparator"),
        repository_health=work.get("global_health", {}),
        fixture_protocol_only=fixture,
        claim_scope="Private oracle mechanics only; no natural utility or learning benefit measured.",
        methodology_note="No generator load or calls. Authenticated cached operands bind the original protocol; private known targets qualify clipping, future-only admission and real hard-exit recovery. Source reuse and fixtures establish no independent generalization. H1/H2 remain unmeasured.",
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        key: "Bind private mechanics to actual invocation evidence; no natural benefit credit."
        for key in value
    }
    value["field_principles"].update(
        static_kernel_ready_score="Only static evaluator checks grant static readiness.",
        causal_kernel_ready_score="Only future-label admission and genuine exact crash recovery grant causal readiness.",
        state_schema="Saved probability trees retain clipping order and final-probability mixtures.",
        crash_controls="Actual child exits, pending IDs, RNG and consumed feedback prove recovery mechanics.",
        energy_probability_error="Measured probability equivalence supplies no added truth.",
        repository_health="Separate bounded repository diagnostic retains pre-existing failures.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute private primitives in a new interpreter before trusting headline fields."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if canonical_hash(value) != checksum:
            return False
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"] or (
                "snapshot_path" in ref and sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
        if work["protocol"] != PROTOCOL_VALUE:
            return False
        if work["static"]:
            if static_checks() != work["static"]:
                return False
            inputs = json.loads(
                (
                    Path(value["measurement_reference"]["path"]).parent / "restart-input.json"
                ).read_bytes()
            )
            rows, labels = kernel.fixture("learnable")
            if inputs != dict(rows=rows, labels=labels, seed=101):
                return False
            final = kernel.run(inputs["rows"], inputs["labels"], inputs["seed"])
            if final != work["causal"]["final"]:
                return False
            for name in ["uninterrupted", "restart90", "restart"]:
                actual = json.loads(
                    (
                        Path(value["measurement_reference"]["path"]).parent / name / "final.json"
                    ).read_bytes()
                )
                if actual != final:
                    return False
        for receipt in [*value["validation_receipts"], *work["causal"].get("receipts", [])]:
            for label in ["stdout", "stderr"]:
                if (
                    label + "_path" in receipt
                    and sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]
                ):
                    return False
        rebuilt = build(
            work,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
            fixture=value["fixture_protocol_only"],
        )
        return rebuilt == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze literal commands and measured statement scope before fixture outcomes."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("[run]", "[run]\npatch = _exit")
        + "[report]\nexclude_lines =\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].append("--basetemp=" + str(private / "owned_pytest"))
            spec["deadline_s"] = 300
        if spec["name"] == "consumer_and_E2E015_019":
            begin = spec["argv"].index("tests/python/test_development_methods_8098.py")
            spec["argv"][begin:] = [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "tests/python/test_hard_exit_learning_qualification_8206.py",
                "--basetemp=" + str(private / "consumer_pytest"),
            ]
            spec["deadline_s"] = 900
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                a.replace("--follow-imports=silent", "--follow-imports=skip") for a in spec["argv"]
            ]
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    specs["repository_health"]["deadline_s"] = 180
    specs["repository_health"]["argv"][:0] = [
        "/usr/bin/env",
        "COVERAGE_FILE=" + str(private / "repository_health.coverage"),
    ]
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use qualified publication and unchanged hard-exit callback, with no recovery framework."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--seed-input" in args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed-input", type=Path, required=True)
        parser.add_argument("--seed-output", type=Path, required=True)
        parser.add_argument("--resume-state", type=Path)
        parser.add_argument("--crash-slot", type=int, default=0)
        parsed = parser.parse_args(args)
        with patch.object(crash.legacy, "run", kernel.run):
            crash.legacy.seed_child(
                parsed.seed_input, parsed.seed_output, parsed.resume_state, parsed.crash_slot
            )
        return 0
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(args))
