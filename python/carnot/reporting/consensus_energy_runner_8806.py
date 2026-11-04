"""REQ-VERIFY-8806: authenticate, measure, validate, then publish exact terminal bytes."""

import argparse
from datetime import UTC, datetime
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import consensus_energy_8806 as m
from carnot.reporting import v754_repaired_memory as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v746_retained_evidence import reference, load

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8806_v754_consensus_challenge_energy"
TASK = "exp8806-consensus-challenge-energy"
CLI = f"scripts/experiments/{NAME}.py"
SOURCE = ROOT / "results/experiment_8803_v754_repaired_evidence_methods.json"
SOURCE_PIN = "sha256:65708e874ea1095258c9a4fd63844a37b85b25626d7c1b0c5748ede1f5f5b5cf"
OWNED = [
    "python/carnot/reporting/consensus_energy_8806.py",
    "python/carnot/reporting/consensus_energy_runner_8806.py",
    CLI,
]
TEST = "tests/python/test_consensus_energy_8806.py"
MODEL_SPECS: list[Json] = []


def admit(source: Path) -> Json:
    """Distinguish absent, unready and replaced producers before consuming their cohort."""
    diagnostic = dict(
        check="missing_file",
        upstream="exp8803-repaired-evidence-methods",
        artifact_field="consensus_protocol_ready_score",
        expected=1,
        observed=None,
        path=str(source.absolute()),
        sha256=None,
    )
    try:
        ref = reference(source)
        diagnostic.update(sha256=ref["sha256"], check="rejected_producer")
        value = json.loads(source.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("producer_object")
        diagnostic.update(
            check="missing_field", observed=value.get("consensus_protocol_ready_score")
        )
        if "consensus_protocol_ready_score" not in value:
            raise ValueError("missing_field")
        diagnostic["check"] = "observed_zero"
        if value["consensus_protocol_ready_score"] != 1:
            raise ValueError("readiness")
        diagnostic.update(
            check="rejected_producer",
            artifact_field="exact_producer_sha256",
            expected=SOURCE_PIN,
            observed=ref["sha256"],
        )
        if ref["sha256"] != SOURCE_PIN or value["required_checks_passed"] is not True:
            raise ValueError("producer_identity_or_validation")
        refs = [ref, *value["cohort_references"].values(), *value["code_config_hashes"]]
        for operand in refs:
            if reference(Path(operand["path"])) != operand:
                raise ValueError("bound_operand_drift:" + operand["path"])
        return dict(passed=True, references=refs, producer=value, diagnostics=[])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, references=[], diagnostics=[dict(diagnostic, error=str(error))])


def preflight(source: Path, private: Path) -> Json:
    """Charge imports and custody once, then project cheap work plus a 900-second reserve."""
    start = time.monotonic()
    admitted = admit(source)
    imports = [
        importlib.import_module(n).__name__
        for n in ("jax", "coverage", "yaml", "numpy", "carnot.training.nce")
    ]
    (private / "writable").write_text("8806")
    current = authority.authority(ROOT)
    archive = ROOT / m.upstream.ARCHIVE
    historical = reference(archive)
    memory = next(
        int(x.split()[1]) * 1024
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemAvailable:")
    )
    excluded = yaml.safe_load((ROOT / "ops/exclusion_manifest.yaml").read_bytes())
    retired = any(
        str(row.get("experiment_id")) == "8806"
        for k in ("retired", "retired_experiments")
        for row in excluded.get(k, [])
        if isinstance(row, dict)
    )
    checks = dict(
        imports=bool(imports),
        private_scratch=(private / "writable").read_text() == "8806",
        current_authority=current["ready"],
        historical_authority=historical["sha256"] == m.upstream.ARCHIVE_PIN,
        source=admitted["passed"],
        not_retired=not retired,
        memory=memory > 512 * 1024**2,
        disk=shutil.disk_usage(private).free > 256 * 1024**2,
        tools=all((ROOT / ".venv/bin" / n).is_file() for n in ("python", "pytest", "ruff", "mypy")),
    )
    worlds = [
        w
        for w in admitted.get("producer", {}).get("split_manifest", {}).get("worlds", [])
        if w["split"] in ("consensus_dev", "consensus_test")
    ]
    checks["sealed_world_counts"] = (
        sum(w["split"] == "consensus_dev" for w in worlds),
        sum(w["split"] == "consensus_test" for w in worlds),
    ) == (60, 120)
    setup = time.monotonic() - start
    cheap = time.monotonic()
    if checks["source"] and worlds:
        m.signals(m.upstream.variants(worlds[0]["edges"])[0]["input"])
    unit = time.monotonic() - cheap
    checks["projected_time"] = setup + unit * 1620 + 900 < 4800
    diagnostics = admitted["diagnostics"] + [
        dict(
            check="rejected_producer",
            upstream=TASK,
            artifact_field=k,
            expected=True,
            observed=v,
            path=str(ROOT),
            sha256=None,
        )
        for k, v in checks.items()
        if not v and k != "source"
    ]
    return dict(
        passed=all(checks.values()),
        checks=checks,
        diagnostics=diagnostics,
        imports=imports,
        setup_s=setup,
        cheap_unit_s=unit,
        validation_reserve_s=900,
        projected_s=setup + unit * 1620 + 900,
        available_memory_bytes=memory,
        available_disk_bytes=shutil.disk_usage(private).free,
        references=admitted["references"]
        + [historical]
        + current.get("references", [])
        + [reference(ROOT / "ops/exclusion_manifest.yaml")],
        worlds=worlds,
        current_authority=current,
    )


def coverage_summary(directory: Path) -> Json:
    """Combine actual unit and CLI databases without excluding any owned statement."""
    from coverage import Coverage

    cov = Coverage(data_file=str(directory / ".coverage"), config_file=False)
    cov.combine(data_paths=[str(directory)], strict=True, keep=True)
    cov.save()
    files: list[Json] = []
    for name in OWNED:
        _, statements, excluded, missing, _ = cov.analysis2(str(ROOT / name))
        files.append(
            dict(path=name, statements=len(statements), missing=missing, excluded=excluded)
        )
    return dict(
        passed=all(not f["missing"] and not f["excluded"] for f in files),
        percent=100
        * sum(f["statements"] - len(f["missing"]) for f in files)
        / sum(f["statements"] for f in files),
        files=files,
        database=reference(directory / ".coverage"),
    )


def validate(raw: Path) -> Json:
    """Retain scoped checks and private historical E2E receipts as actual child work."""
    private = TemporaryDirectory(prefix="exp8806-validation-", dir="/var/tmp")
    database = Path(private.name) / "coverage"
    database.mkdir()
    config = database / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(database / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
        + "[report]\nexclude_lines =\n"
    )
    env = ["env", "COVERAGE_RCFILE=" + str(config), "COVERAGE_FILE=" + str(database / ".coverage")]
    pytest = ["-m", "pytest", "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    plans = [
        (
            "unit",
            env
            + [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config)]
            + pytest
            + [TEST, "--basetemp=" + str(Path(private.name) / "unit_tmp")],
            600,
        ),
        (
            "e2e018",
            [sys.executable]
            + pytest
            + [
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "--basetemp=" + str(Path(private.name) / "e2e018_tmp"),
            ],
            300,
        ),
        (
            "e2e021",
            [sys.executable]
            + pytest
            + [
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(Path(private.name) / "e2e021_tmp"),
            ],
            300,
        ),
        ("ruff", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST], 120),
        ("format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST], 120),
        (
            "mypy",
            [
                str(ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                "--ignore-missing-imports",
                *OWNED,
            ],
            120,
        ),
        ("spec", [sys.executable, "scripts/check_spec_coverage.py", TEST], 120),
    ]
    receipts = [
        child(name, argv, raw / "checks", deadline=deadline, heartbeat=60)
        for name, argv, deadline in plans
    ]
    coverage = coverage_summary(database)
    retained = raw / "coverage"
    retained.mkdir()
    for path in database.iterdir():
        shutil.copy2(path, retained / path.name)
    coverage["database"] = reference(retained / ".coverage")
    retained_refs = [reference(path) for path in retained.iterdir()]
    private.cleanup()
    return dict(
        passed=all(r["passed"] for r in receipts) and coverage["passed"],
        receipts=receipts,
        coverage=coverage,
        raw_references=retained_refs,
    )


def build(work: Json, validation: Json) -> Json:
    """Derive terminal claims from measurements; oracle success never qualifies learned benefit."""
    ready = work["preconditions_checked"]["passed"]
    summary = m.summarize(work["per_candidate_rows"]) if ready else {}
    controls = work["fit"].get("shortcut_controls", [])
    controls_ok = bool(controls) and all(c["passed"] for c in controls)
    checked = validation["passed"]
    verdict = (
        "blocked"
        if not ready
        else "disqualified"
        if not checked or not controls_ok
        else "positive"
        if summary["benefit_passed"]
        else "null"
    )
    groups: dict[str, list[Json]] = {}
    for row in work["per_candidate_rows"]:
        if row["arm"] == "learned_energy":
            groups.setdefault(row["unit_id"], []).append(row)
    rows = [
        dict(
            unit_id=unit,
            cluster_id=local[0]["cluster_id"],
            arm="learned_energy",
            seed=None,
            metric=sum(r["metric"] for r in local) / len(local),
            denominator=len(local),
            status="completed",
            missing_reason=None,
        )
        for unit, local in groups.items()
    ]
    if not ready:
        rows = [
            dict(
                unit_id=f"heldout/{i // 9}/{m.CONDITIONS[(i // 3) % 3]}/{m.AGREEMENTS[i % 3]}",
                cluster_id=f"heldout/{i // 9}",
                arm="scheduled",
                seed=None,
                metric=None,
                denominator=3,
                status="censored",
                missing_reason="upstream_or_precondition_gate",
            )
            for i in range(1080)
        ]
    result = dict(
        experiment_id=8806,
        task_id=TASK,
        milestone="2026.11.754",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_constructed_consensus_energy",
        verdict_class=verdict,
        inference_substrate="cpu_energy_training",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        target_model="GibbsModel(input_dim=5, hidden_dims=[4]); three numerical NCE heads per seed, initialized from scratch",
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        numerical_training_calls=work["fit"].get("numerical_training_calls", 0),
        verifier_is_oracle=False,
        duration_s=work["duration_s"],
        random_seed=m.SEEDS[0],
        reproducibility_checksum=canonical_hash(
            [
                m.SEEDS,
                work["code_config_hashes"],
                work["source_artifact_hashes"],
                work["raw_shard_hashes"],
            ]
        ).split(":")[-1][:16],
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        preconditions_checked=work["preconditions_checked"],
        phase_spans=work["phase_spans"],
        rows=rows,
        intended_count=1080,
        completed_count=1080 if ready else 0,
        censored_count=0 if ready else 1080,
        failed_count=0,
        excluded_count=0,
        independent_count=120 if ready else 0,
        gate_check_summary=work["preconditions_checked"]["diagnostics"],
        acceptance_gates=summary.get("acceptance_gates", {}),
        required_checks_passed=bool(checked and ready),
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        exposure_scope="constructed candidates on sealed relational worlds; no live-model outputs",
        independent_generalization_score=0,
        consensus_rows_ready_score=int(ready and checked and controls_ok),
        learned_energy_parameters=work["fit"].get("parameters", []),
        training_and_calibration_worlds=dict(
            training=work["fit"].get("train_worlds", []),
            calibration=work["fit"].get("calibration_worlds", []),
        ),
        per_candidate_rows=work["per_candidate_rows"],
        unanimous_wrong_false_accepts=summary.get("unanimous_wrong_false_accepts", []),
        brier_rows=summary.get("brier_rows", []),
        coverage_rows=summary.get("coverage_rows", []),
        cost_rows=work["fit"].get("costs", []) + work["cost_rows"],
        shortcut_controls=controls,
        oracle_reference_rows=summary.get("oracle_reference_rows", []),
        intervals_97_5=summary.get("intervals_97_5", {}),
        contingency_table=summary.get("contingency_table", []),
        methodology_note="NCE one-versus-rest heads predict support/refute/insufficient from five counts; candidate actions are accept/reject/escalate. Brier is the sum of three squared probability errors. Insufficient evidence always targets escalation. Coverage is non-escalation. Five seeds are nested within each world. Coverage is compared with both non-oracle baselines. Exact labels construct examples and score outcomes; they never enter learned features. No independent generalization is measured.",
        primitive_reference=work["primitive_reference"],
        validation_reference=work["validation_reference"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8803,
                fields_imported=[
                    "consensus_protocol_ready_score",
                    "split_manifest",
                    "cohort_references",
                ],
                sha256=work["source_artifact_hashes"][0]["sha256"],
            )
        ]
        if ready
        else [],
        repository_health_reference=work.get("repository_health_reference"),
        field_principles={},
    )
    result["field_principles"] = {
        key: (
            "Exact bound operands and measured work; seeds and sibling conditions are not independent worlds. Constructed oracle correctness is separate from learned evidence use."
            if key not in {"verdict_class", "consensus_rows_ready_score"}
            else "External prerequisites block; owned failure disqualifies with zero readiness; scientific gates can yield null."
        )
        for key in result
    }
    return result


def replay(path: Path) -> bool:
    """Rebuild claims from sealed inputs and weights, rejecting even repaired local hashes."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if any(value[k] != seal[k] for k in ("primitive_reference", "validation_reference")):
            return False
        work, validation = load(value["primitive_reference"]), load(value["validation_reference"])
        for ref in (
            work["source_artifact_hashes"]
            + work["code_config_hashes"]
            + work["raw_shard_hashes"]
            + validation["raw_references"]
        ):
            if reference(Path(ref["path"])) != ref:
                return False
        if work["preconditions_checked"]["passed"]:
            cohort = load(work["cohort_reference"])["rows"]
            if cohort != m.construct(work["preconditions_checked"]["worlds"]):
                return False
            rows = m.evaluate(cohort, work["fit"]["parameters"])
            if rows != work["per_candidate_rows"]:
                return False
        work.update(
            primitive_reference=value["primitive_reference"],
            validation_reference=value["validation_reference"],
        )
        return bool(build(work, validation) == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(candidate: Path, raw: Path) -> Json:
    """Run unchanged scanners and a fresh outside-checkout interpreter on exact candidate bytes."""
    plans = [
        (
            "adversarial",
            [sys.executable, "scripts/adversarial_verify.py", str(candidate), "--json"],
        ),
        (
            "row_consistency",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
        ),
        (
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                sys.executable,
                "-c",
                "import os,runpy,sys; os.chdir('/tmp'); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
                str(ROOT / CLI),
                "--cold-replay",
                str(candidate),
            ],
        ),
    ]
    receipts = [
        child(name, argv, raw / "terminal_checks", deadline=180, heartbeat=60)
        for name, argv in plans
    ]
    result = dict(passed=all(r["passed"] for r in receipts), receipts=receipts)
    atomic_json(raw / "terminal_validation.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Keep all receipts below the task directory and publish only validated terminal bytes."""
    m.progress("start_no_llm_load_or_generation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--upstream", type=Path, default=SOURCE)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    started = time.monotonic()
    spans: list[Json] = []

    def phase(name: str, begin: float) -> None:
        spans.append(
            dict(
                phase=name,
                started_monotonic_s=begin,
                ended_monotonic_s=time.monotonic(),
                ended_at_utc=datetime.now(UTC).isoformat(),
            )
        )
        m.progress(name + "_after")

    with TemporaryDirectory(prefix="exp8806-private-", dir="/var/tmp") as directory:
        begin = time.monotonic()
        m.progress("preconditions_before")
        try:
            pre = preflight(args.upstream, Path(directory))
        except (OSError, ImportError, ValueError, KeyError) as error:
            pre = dict(
                passed=False,
                diagnostics=[
                    dict(
                        check="missing_file" if isinstance(error, OSError) else "rejected_producer",
                        upstream=TASK,
                        artifact_field="preconditions",
                        expected=True,
                        observed=str(error),
                        path=str(args.upstream),
                        sha256=None,
                    )
                ],
                references=[],
                worlds=[],
            )
        phase("preconditions", begin)
        cohort: list[Json] = []
        fit: Json = {}
        rows: list[Json] = []
        costs: list[Json] = []
        if pre["passed"]:
            begin = time.monotonic()
            train_worlds, calibration_worlds = m.partition(pre["worlds"])
            atomic_json(
                raw / "split_before_labels.json",
                dict(training=train_worlds, calibration=calibration_worlds),
            )
            m.progress("split_sealed_before_labels", 60, 120)
            cohort = m.construct(pre["worlds"])
            atomic_json(raw / "cohort.json", dict(rows=cohort))
            phase("construction", begin)
            costs.append(
                dict(
                    arm="exact_oracle_construction",
                    duration_s=time.monotonic() - begin,
                    groups=len(cohort),
                )
            )
            begin = time.monotonic()
            fit = m.fit(cohort, list(m.SEEDS))
            phase("training_and_calibration", begin)
            atomic_json(raw / "parameters.json", fit)
            begin = time.monotonic()
            m.progress("heldout_before", 0, 1080)
            rows = m.evaluate(cohort, fit["parameters"], costs)
            phase("heldout", begin)
            costs.append(
                dict(
                    arm="all_serving_arms",
                    duration_s=time.monotonic() - begin,
                    groups=1080,
                    includes="constraint checks, baseline/oracle lookup, Gibbs NumPy forward, scoring",
                )
            )
        atomic_json(raw / "per_candidate_rows.json", dict(rows=rows))
        begin = time.monotonic()
        m.progress("validation_before")
        validation = validate(raw)
        phase("validation", begin)
        atomic_json(raw / "validation.json", validation)
        source_refs = pre["references"]
        work = dict(
            run_date=args.date,
            preconditions_checked=pre,
            fit=fit,
            per_candidate_rows=rows,
            source_artifact_hashes=source_refs,
            code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
            raw_shard_hashes=[reference(p) for p in sorted(raw.glob("*.json"))],
            phase_spans=spans,
            duration_s=time.monotonic() - started,
            cost_rows=costs,
            cohort_reference=reference(raw / "cohort.json") if pre["passed"] else None,
            repository_health_reference=reference(
                ROOT / "results/raw" / NAME / "repository_health.json"
            )
            if (ROOT / "results/raw" / NAME / "repository_health.json").is_file()
            else None,
        )
        atomic_json(raw / "work.json", work)
        work_ref, validation_ref = reference(raw / "work.json"), reference(raw / "validation.json")
        atomic_json(
            raw / "independent_seal.json",
            dict(primitive_reference=work_ref, validation_reference=validation_ref),
        )
        work.update(primitive_reference=work_ref, validation_reference=validation_ref)
        value = build(work, validation)
        m.progress("terminal_validation_before")
        try:
            published = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
        except ValueError as error:
            if str(error) != "candidate_rejected":
                raise
            rejected = output.parent / "raw" / NAME / "terminal_candidate.json"
            shutil.copy2(rejected, raw / "rejected_candidate.json")
            for stem in ("work", "validation", "independent_seal"):
                shutil.copy2(raw / (stem + ".json"), raw / ("rejected_" + stem + ".json"))
            terminal_failure = load(reference(raw / "terminal_validation.json"))
            validation["passed"] = False
            validation["receipts"].extend(terminal_failure["receipts"])
            atomic_json(raw / "validation.json", validation)
            work["raw_shard_hashes"] = [
                reference(Path(ref["path"])) for ref in work["raw_shard_hashes"]
            ]
            work.pop("primitive_reference")
            work.pop("validation_reference")
            atomic_json(raw / "work.json", work)
            work.update(
                primitive_reference=reference(raw / "work.json"),
                validation_reference=reference(raw / "validation.json"),
            )
            atomic_json(
                raw / "independent_seal.json",
                {k: work[k] for k in ("primitive_reference", "validation_reference")},
            )
            value = build(work, validation)
            published = publish_primary(
                output, value, lambda candidate: terminal(candidate, raw / "recovery")
            )
        atomic_json(raw / "publication_receipt.json", published)
        m.progress(
            "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
        )
    return 0
