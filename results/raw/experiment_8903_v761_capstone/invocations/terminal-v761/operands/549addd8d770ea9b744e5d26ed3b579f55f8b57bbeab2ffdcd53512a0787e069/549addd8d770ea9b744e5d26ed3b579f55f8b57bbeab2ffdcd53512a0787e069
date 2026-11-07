"""REQ-REPORT-8804: publish complete future-loss rows with separate cost readiness."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

IMPORT_STARTED = time.monotonic()

from coverage import Coverage

from carnot.learning import causal_template_streams_8804 as s
from carnot.reporting import v754_repaired_memory as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v753_qualification_runner import coverage_summary

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8804_v754_causal_template_streams"
CLI = f"scripts/experiments/{NAME}.py"
UPSTREAM = "results/experiment_8802_v754_causal_memory_prototype.json"
UPSTREAM_PIN = "sha256:ddbdbb66aa7118e6a03bb702d9f51e91a346337d1d89fa7befc9469ef1b8d497"
OWNED = [
    "python/carnot/learning/causal_template_streams_8804.py",
    "python/carnot/reporting/causal_template_streams_8804.py",
    CLI,
]
TESTS = [
    "tests/python/test_causal_template_streams_8804.py",
    "tests/python/test_causal_template_runner_8804.py",
]
reference = s.reference
IMPORT_DURATION = time.monotonic() - IMPORT_STARTED


def upstream_gate(path: Path) -> Json:
    """Distinguish absent operands, rejected producers and an observed readiness zero."""
    try:
        value = json.loads(path.read_bytes()) if path.is_file() else {}
    except ValueError:
        value = dict(verdict_class="disqualified")
    observed = (
        "missing_file"
        if not path.is_file()
        else value.get("memory_execution_ready_score", "missing_field")
    )
    if value.get("verdict_class") in {"blocked", "disqualified"}:
        observed = "rejected_producer"
    return dict(
        check="qualified_causal_execution",
        upstream="exp8802-causal-memory-prototype",
        artifact_field="memory_execution_ready_score",
        expected=1,
        observed=observed,
        path=str(path.absolute()),
        sha256=sha256_file(path) if path.is_file() else None,
        passed=observed == 1 and value.get("experiment_id") == 8802,
    )


def preflight(raw: Path, intended_count: int = 30600, workers: int = 30) -> Json:
    """Charge import, custody and compilation once; project a real complete cheap unit."""
    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    s.durable(raw / "scratch.json", dict(private=True))
    gate = upstream_gate(ROOT / UPSTREAM)
    sources: list[Json] = []
    upstream: Json = {}
    if gate["passed"]:
        upstream = json.loads((ROOT / UPSTREAM).read_bytes())
        sources = (
            [reference(ROOT / UPSTREAM)]
            + upstream["code_config_hashes"]
            + upstream["raw_shard_hashes"]
            + upstream["source_artifact_hashes"]
        )
    source_ok = all(
        Path(r["path"]).is_file() and sha256_file(Path(r["path"])) == r["sha256"] for r in sources
    )
    current = authority.authority(ROOT)
    historical_paths = [ROOT / path for path in authority.ARCHIVES]
    historical_available = all(path.is_file() for path in historical_paths)
    retained_refs = []
    for ref in current["references"] + [
        reference(path) for path in historical_paths if path.is_file()
    ]:
        retained = raw / "authority_inputs" / Path(ref["path"]).name
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref["path"], retained)
        retained_refs.append(reference(retained))
        source_ok &= retained_refs[-1]["sha256"] == ref["sha256"]
    sources.extend(retained_refs)
    atomic_json(raw / "current_authority.json", current)
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    checks = dict(
        upstream=gate["passed"],
        pinned_upstream=gate["sha256"] == UPSTREAM_PIN,
        source_hashes=source_ok,
        dependency_imports=True,
        current_authority=current["ready"],
        historical_authority=historical_available,
        cpu_backend=s.m.jax.default_backend() == "cpu",
        private_scratch=json.loads((raw / "scratch.json").read_bytes())["private"],
        available_memory=memory > 1024**3,
        disk=shutil.disk_usage(raw).free > 2 * 1024**3,
    )
    probe, wave, calls = 0.0, 0.0, 0
    probe_refs: list[Json] = []
    if all(checks.values()):
        s.progress("cheap_unit_before")
        params = s.m.initial_parameters(11)
        sample = s.payload(s.manifest()["streams"][-1]["worlds"][-1], "insufficient")
        counts = s.m.observe(sample, [])["counts"]
        s.m.update(params, counts, "insufficient")
        calls += 1
        s.probabilities(params, counts)
        custody = time.monotonic() - started + IMPORT_DURATION
        unit_start = time.monotonic()
        store = s.StreamStore(raw / "probe_state.json", params)
        store.predict("e0", sample, 0.5)
        store.close()
        store.release("e0", sample, 8, "combined")
        calls += 1
        assert s.m.Store(store.path, {}).state == store.state
        probe = time.monotonic() - unit_start
        s.progress("cheap_unit_after", 1, 0)
        probe_refs.append(reference(store.path))
        wave = probe
        if workers > 1:

            def unit(index: int) -> Json:
                tested = s.StreamStore(raw / "parallel_probe" / str(index) / "state.json", params)
                tested.predict("e0", sample, 0.5)
                tested.close()
                tested.release("e0", sample, 8, "combined")
                assert s.m.Store(tested.path, {}).state == tested.state
                return reference(tested.path)

            s.progress("parallel_probe_before", 0, workers)
            wave_start = time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                probe_refs.extend(pool.map(unit, range(workers)))
            wave = time.monotonic() - wave_start
            calls += workers
            s.progress("parallel_probe_after", workers, 0)
    else:
        custody = time.monotonic() - started + IMPORT_DURATION
    custody = time.monotonic() - started + IMPORT_DURATION
    projection = wave * intended_count / workers
    checks["projected_time"] = custody + projection + 900 < 4800
    return dict(
        passed=all(checks.values()),
        checked=checks,
        gate=gate,
        sources=sources,
        upstream=upstream,
        authority_reference=reference(raw / "current_authority.json"),
        import_and_custody_s=custody,
        measured_cheap_unit_s=probe,
        measured_parallel_wave_s=wave,
        parallel_workers=workers,
        probe_refs=probe_refs,
        projected_work_s=projection,
        validation_reserve_s=900,
        numerical_training_calls=calls,
        available_memory_bytes=memory,
        free_disk_bytes=shutil.disk_usage(raw).free,
        seed_authority=dict(
            current_task=[11, 23, 47],
            historical_protocol=[11, 23, 37],
            resolution="explicit V754 task seed 47; original upstream remains byte-identical",
        ),
    )


def run(raw: Path, date: str, *, frozen: Json | None = None) -> Json:
    """Retain raw work before qualification; terminal readiness is assigned only after tests."""
    started = time.monotonic()
    selected_manifest = frozen if frozen is not None else s.manifest()
    online, retained = s.census(selected_manifest, "preflight")
    s.progress("preconditions_before")
    pre = preflight(raw, len(online) + len(retained), min(30, len(selected_manifest["streams"])))
    atomic_json(raw / "preconditions.json", {k: v for k, v in pre.items() if k != "upstream"})
    s.progress("preconditions_after")
    atomic_json(raw / "frozen_manifest.json", selected_manifest)
    if pre["passed"]:
        selected, fit_calls = s.heads(pre["upstream"], selected_manifest["seeds"])
        s.progress("streams_before", 0, 30600)
        work = s.execute(selected_manifest, selected, raw / "streams")
        work["numerical_training_calls"] += fit_calls
        work["development_training_calls"] = fit_calls
        s.progress("streams_after", len(work["per_query_rows"]) + len(work["retention_rows"]), 0)
        verified = s.check_work(work, selected_manifest)
    else:
        work = s.empty_work(selected_manifest, "external_precondition_failed")
        verified = dict(passed=False, checks={})
    atomic_json(raw / "measurement.json", work)
    rows = work["per_query_rows"] + work["retention_rows"]
    gates = [pre["gate"]] + [
        dict(
            check=k,
            upstream="exp8804-preconditions",
            artifact_field="checked." + k,
            expected=True,
            observed=v,
            passed=v,
            **reference(raw / "preconditions.json"),
        )
        for k, v in pre["checked"].items()
    ]
    value = dict(
        experiment_id=8804,
        task_id="exp8804-causal-template-streams",
        milestone="2026.11.754",
        run_date=date,
        honest_verdict="complete_disqualified_pending_owned_validation"
        if pre["passed"]
        else "complete_blocked_preconditions",
        verdict_class="disqualified" if pre["passed"] else "blocked",
        inference_substrate="cpu_energy_training",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        verifier_is_oracle=True,
        duration_s=time.monotonic() - started,
        random_seed=8804,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=pre["sources"],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        raw_shard_hashes=[
            reference(raw / name)
            for name in [
                "measurement.json",
                "frozen_manifest.json",
                "current_authority.json",
                "preconditions.json",
            ]
        ]
        + pre["probe_refs"],
        preconditions_checked={k: v for k, v in pre.items() if k != "upstream"},
        phase_spans=[dict(phase="custody_and_streams", duration_s=time.monotonic() - started)],
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        excluded_count=0,
        independent_count=len(selected_manifest["streams"]) if pre["passed"] else 0,
        n_samples=len(selected_manifest["streams"]) if pre["passed"] else 0,
        protocol_scope="private_fixture" if frozen is not None else "production",
        gate_check_summary=gates,
        acceptance_gates=dict(
            preconditions=pre["passed"],
            rows=verified["passed"],
            cost=bool(verified.get("checks", {}).get("cost"))
            and bool(verified.get("checks", {}).get("tail")),
            owned_validation=False,
        ),
        required_checks_passed=False,
        owned_statement_coverage=dict(complete=False),
        validation_receipts=[],
        exposure_scope="constructed_encoded_graphs_with_exact_oracle_labels",
        independent_generalization_score=0,
        cited_upstream_artifacts=[
            dict(
                experiment_id=8802,
                sha256=pre["gate"]["sha256"],
                fields_imported=[
                    "memory_execution_ready_score",
                    "frozen_memory_protocol",
                    "calibrated_head_parameters",
                    "code_config_hashes",
                    "raw_shard_hashes",
                    "source_artifact_hashes",
                ],
            )
        ],
        memory_rows_ready_score=0,
        learning_cost_ready_score=0,
        work_checks=verified,
        numerical_training_calls_scope="actual probe, development seed47, every release and prepublication gradient reconstruction; control gradients are computed and discarded; scoped tests and cold replay belong to separate validation receipts",
        complete_operation_boundary="prediction start through eight later queries to verified gradient, atomic commit, file/directory fsync and exact restart; overlapping latency windows must not be summed",
        methodology_note="Constructed-label execution only. All arms pay five primitive serving checks, one grading check and one delayed feedback check. Exp8810 owns benefit decisions. No independent generalization claim.",
        global_reconciliation_debt="One whole-suite health receipt is separate; conductor owns ops and BMAD reconciliation.",
        field_principles={},
        **work,
    )
    value["numerical_training_calls"] += pre["numerical_training_calls"] + verified.get(
        "numerical_training_calls", 0
    )
    value["field_principles"] = {
        key: "Measured or frozen stream operand; constructed readiness does not certify independent learning benefit."
        for key in value
    }
    return value


def replay(path: Path) -> Json:
    """Reconstruct the terminal census and predictions from exact retained operands."""
    try:
        value = json.loads(path.read_bytes())
        refs = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        hashes = all(sha256_file(Path(ref["path"])) == ref["sha256"] for ref in refs)
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        frozen = json.loads(Path(value["raw_shard_hashes"][1]["path"]).read_bytes())
        expected_manifest = s.manifest()
        if value["protocol_scope"] == "private_fixture":
            expected_manifest["streams"] = expected_manifest["streams"][: len(frozen["streams"])]
            expected_manifest["seeds"] = [
                seed for seed in expected_manifest["seeds"] if seed in frozen["seeds"]
            ]
        rows = work["per_query_rows"] + work["retention_rows"]
        ready = int(
            all(value["acceptance_gates"].values())
            and value["owned_statement_coverage"]["complete"]
        )
        passed = hashes and all(
            value[k] == work[k] for k in work if k != "numerical_training_calls"
        )
        passed &= frozen == expected_manifest and value["protocol_scope"] in {
            "production",
            "private_fixture",
        }
        if ready:
            coverage_ref = value["owned_statement_coverage"]["combined_reference"]
            measured_coverage = coverage_summary(Path(coverage_ref["path"]), OWNED)
            passed &= (
                bool(value["validation_receipts"])
                and all(r["passed"] for r in value["validation_receipts"])
                and measured_coverage == value["owned_statement_coverage"]
            )
        passed &= (
            value["rows"] == rows
            and value["intended_count"] == len(rows)
            and value["completed_count"] == sum(r["status"] == "completed" for r in rows)
            and value["censored_count"] == sum(r["status"] == "censored" for r in rows)
            and value["failed_count"] == sum(r["status"] == "failed" for r in rows)
            and value["excluded_count"] == 0
            and value["reproducibility_checksum"] == canonical_hash(work)
            and value["memory_rows_ready_score"] == value["learning_cost_ready_score"] == ready
            and value["numerical_training_calls"]
            == work["numerical_training_calls"]
            + value["preconditions_checked"]["numerical_training_calls"]
            + value["work_checks"].get("numerical_training_calls", 0)
        )
        if value["acceptance_gates"]["preconditions"]:
            reconstructed = s.check_work(work, frozen)
            passed &= reconstructed == value["work_checks"]
            passed &= value["independent_count"] == len(frozen["streams"])
        else:
            expected = s.empty_work(frozen, "external_precondition_failed")
            passed &= work == expected and value["verdict_class"] == "blocked" and not ready
        return dict(passed=bool(passed), reconstructed_rows=len(rows))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="replay_operand_failure")


def validate(raw: Path) -> Json:
    """Combine real unit and standalone CLI SQLite data with no statement exclusions."""
    covdir = (raw / "coverage").absolute()
    covdir.mkdir(parents=True, exist_ok=True)
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=True\ndata_file="
        + str(covdir / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
        + "[report]\nexclude_lines=\n"
    )
    py = str(ROOT / ".venv/bin/python")
    paths = OWNED + TESTS
    private_coverage = Path(tempfile.mkdtemp(prefix="carnot8804-child-coverage-"))
    plan = [
        (
            "owned_pytest",
            [
                "env",
                "CARNOT_8804_COVERAGE_DIRECTORY=" + str(private_coverage),
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
                *TESTS,
            ],
            600,
        ),
        (
            "private_E2E_018_021",
            [
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
                "tests/python/test_primary_publication_7928.py",
            ],
            300,
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *paths], 60),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *paths], 60),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
            120,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS], 60),
    ]
    receipts = [
        child(name, argv, raw / "validation", deadline=deadline, heartbeat=60)
        for name, argv, deadline in plan
    ]
    for shard in private_coverage.glob(".coverage.child.*"):
        shutil.copyfile(shard, covdir / shard.name)
    cov = Coverage(data_file=str(covdir / ".coverage"), config_file=False)
    cov.combine(data_paths=[str(covdir)], keep=True)
    cov.save()
    summary = coverage_summary(covdir / ".coverage", OWNED)
    return dict(
        receipts=receipts,
        coverage=summary,
        passed=all(r["passed"] for r in receipts) and summary["complete"],
    )


def terminal_checks(candidate: Path, raw: Path) -> Json:
    """Use unchanged shared validators, then cold replay outside the checkout."""
    py = str(ROOT / ".venv/bin/python")
    plan = [
        ("adversarial", [py, "scripts/adversarial_verify.py", str(candidate), "--json"], 60),
        (
            "row_consistency",
            [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            60,
        ),
        (
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                "/tmp",
                py,
                "-u",
                str(ROOT / CLI),
                "--replay",
                str(candidate),
            ],
            600,
        ),
    ]
    receipts = [
        child(name, argv, raw / "terminal", deadline=deadline, heartbeat=60)
        for name, argv, deadline in plan
    ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def qualify_and_publish(value: Json, raw: Path, output: Path) -> None:
    """Failed owned checks disqualify; external blocks remain terminal without measurements."""
    started = time.monotonic()
    checked = validate(raw)
    value["validation_receipts"] = checked["receipts"]
    value["owned_statement_coverage"] = checked["coverage"]
    value["acceptance_gates"]["owned_validation"] = checked["passed"]
    ready = all(value["acceptance_gates"].values())
    value["memory_rows_ready_score"] = int(ready)
    value["learning_cost_ready_score"] = int(ready)
    value["required_checks_passed"] = ready
    if value["verdict_class"] != "blocked":
        value["verdict_class"] = "null" if ready else "disqualified"
        value["honest_verdict"] = (
            "complete_null_measurements_ready_benefit_reserved_for_exp8810"
            if ready
            else "complete_disqualified_owned_validation"
        )
    value["phase_spans"].append(
        dict(phase="owned_validation", duration_s=time.monotonic() - started)
    )
    value["duration_s"] += time.monotonic() - started
    health = ROOT / "results/raw" / NAME / "global-health/full_python_once.receipt.json"
    if health.is_file():
        value["repository_health_reference"] = reference(health)
    value["field_principles"] = {
        key: "Exact measured stream or validation operand; no independent benefit claim."
        for key in value
    }
    value["field_principles"].update(
        n_samples="The effective sample size is reset streams; paired arms and nested seeds add no independent units.",
        protocol_scope="Only the production CLI executes the entire frozen task; reduced private fixtures cannot change its roster.",
        numerical_training_calls="Count probe, development, released-feedback and prepublication reconstruction gradients; historical fits are not current work.",
        memory_rows_ready_score="Complete causal online and retention rows require passing owned tests and coverage, without a benefit claim.",
        learning_cost_ready_score="Complete parent-clock windows and restart parity qualify cost even when future loss does not improve.",
        tail_drain_cost_s="Measured total tail service time; zero with no tail receipts denotes no measurement and readiness zero.",
        complete_learning_cost_rows="Predict-to-ack latency windows overlap; sum service components rather than overlapping windows.",
        preconditions_checked="Bind exact current and retained historical authorities; record the explicit seed47 task override without rewriting seed37 history.",
        independent_generalization_score="Constructed encoded graph labels support no oracle-distinct generalization claim.",
    )
    atomic_json(raw / "prepublication.json", value)
    s.progress("terminal_validation_before")
    try:
        published = publish_primary(output, value, lambda path: terminal_checks(path, raw))
    except ValueError:
        atomic_json(raw / "rejected_candidate.json", value)
        raise
    atomic_json(raw / "publication.json", published)
    s.progress("terminal_publication_complete", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """The direct CLI owns no experiment logic, and replay needs no working-directory imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        result = replay(args.replay)
        print(json.dumps(dict(replay=result)), flush=True)
        return int(not result["passed"])
    raw = args.raw or ROOT / "results/raw" / NAME / "invocations" / str(time.time_ns())
    value = run(raw, args.date)
    qualify_and_publish(value, raw, args.output)
    print(
        json.dumps(
            {
                key: value.get(key)
                for key in [
                    "verdict_class",
                    "completed_count",
                    "independent_count",
                    "memory_rows_ready_score",
                    "learning_cost_ready_score",
                ]
            }
        ),
        flush=True,
    )
    return 0
