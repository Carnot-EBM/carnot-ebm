"""REQ-REPORT-8810: publish qualified causal audits with exact input custody."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

IMPORT_STARTED = time.monotonic()

from coverage import Coverage

from carnot.reporting import causal_learning_audit_8810 as a
from carnot.reporting import v754_repaired_memory as authority
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v753_qualification_runner import coverage_summary

Json = dict[str, Any]
ROOT, NAME = a.ROOT, a.NAME
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/causal_learning_audit_8810.py",
    "python/carnot/reporting/causal_learning_runner_8810.py",
    CLI,
]
TESTS = ["tests/python/test_causal_learning_audit_8810.py"]
IMPORT_DURATION = time.monotonic() - IMPORT_STARTED
HISTORY_PINS = {
    "research-roadmap-v752-preserved-20261103.md": "sha256:9643522778637bc54e622a7a5dc62faf1279944464c6df2a5028d590478d9872",
    "research-roadmap-v753-preserved-20261104.md": "sha256:15b63fe517179897398afe775d4c136e0b453fc3b1b9bb57137dd2ada3f78d8d",
}


def upstream_gate(path: Path) -> Json:
    """Keep missing files, missing fields, rejected producers and measured zero distinct."""
    try:
        value = json.loads(path.read_bytes()) if path.is_file() else {}
    except ValueError:
        value = dict(verdict_class="disqualified")
    observed = (
        "missing_file"
        if not path.is_file()
        else value.get("memory_rows_ready_score", "missing_field")
    )
    if value.get("verdict_class") in {"blocked", "disqualified"}:
        observed = "rejected_producer"
    return dict(
        check="sealed_stream_rows",
        upstream="exp8804-causal-template-streams",
        artifact_field="memory_rows_ready_score",
        expected=1,
        observed=observed,
        path=str(path.absolute()),
        sha256=a.sha256_file(path) if path.is_file() else None,
        passed=observed == 1 and value.get("experiment_id") == 8804,
    )


def frozen_manifest() -> Json:
    """Reuse the frozen primitive roster and preserve the explicit current seed override."""
    frozen = a.primitive.freeze_manifest()
    frozen.update(seeds=[11, 23, 47], arms=list(a.primitive.ARMS), online_executed=True)
    return dict(frozen)


def preflight(raw: Path, evidence: tuple[Json, Json] | None = None) -> tuple[Json, Json, Json]:
    """Charge custody once and project one actual sealed read before the full audit."""
    start = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "scratch.json", dict(private=True))
    gates = []
    sources = []
    source: Json = {}
    frozen = frozen_manifest()
    if evidence is not None:
        source, frozen = deepcopy(evidence)
        checks = dict(private_fixture=True)
    else:
        gate = upstream_gate(ROOT / a.UPSTREAM)
        gates.append(gate)
        checks = dict(upstream=gate["passed"], pinned_primary=gate["sha256"] == a.PIN)
        if all(checks.values()):
            source = json.loads((ROOT / a.UPSTREAM).read_bytes())
            sources = [
                a.reference(ROOT / a.UPSTREAM),
                dict(path=str(ROOT / a.PROTOTYPE), sha256=a.PROTOTYPE_PIN),
            ]
            sources += (
                source["source_artifact_hashes"]
                + source["code_config_hashes"]
                + source["raw_shard_hashes"]
            )
            checks["source_hashes"] = all(
                Path(r["path"]).is_file() and a.sha256_file(Path(r["path"])) == r["sha256"]
                for r in sources
            )
            required = [
                "per_query_rows",
                "retention_rows",
                "release_order_rows",
                "complete_learning_cost_rows",
                "heads",
                "input_refs",
                "final_state_refs",
            ]
            checks["required_fields"] = all(k in source for k in required)
            checks["producer_qualification"] = (
                source.get("required_checks_passed") is True
                and source.get("flagged_adversarial", False) is False
            )
            checks["frozen_manifest"] = a.load(source["raw_shard_hashes"][1]) == frozen
            checks["sealed_files_available"] = all(
                Path(r.get("seal_ref", {}).get("path", "/missing")).is_file()
                for r in source["per_query_rows"] + source["retention_rows"]
            )
        current = authority.authority(ROOT)
        checks["current_authority"] = current["ready"]
        sources.extend(current["references"])
        history = [ROOT / p for p in authority.ARCHIVES]
        checks["historical_authority"] = all(
            p.is_file() and a.sha256_file(p) == HISTORY_PINS.get(p.name) for p in history
        )
        sources.extend(a.reference(p) for p in history if p.is_file())
        for index, ref in enumerate(
            current["references"] + [a.reference(p) for p in history if p.is_file()]
        ):
            target = raw / "authority_inputs" / (str(index) + "-" + Path(ref["path"]).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ref["path"], target)
            sources.append(a.reference(target))
        atomic_json(raw / "authority.json", current)
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    free = shutil.disk_usage(raw).free
    checks.update(
        private_scratch=json.loads((raw / "scratch.json").read_bytes())["private"],
        dependency_imports=True,
        available_memory=available > 1024**3,
        free_disk=free > 1024**3,
    )
    custody = time.monotonic() - start + IMPORT_DURATION
    probe = 0.0
    if all(checks.values()):
        a.progress("cheap_unit_before", 0, 1)
        began = time.monotonic()
        row = source["per_query_rows"][0]
        seal = a.load(row["seal_ref"])
        a.score(seal["sealed"]["prediction"], row["truth"], source["heads"][0]["threshold"])
        a.require(a.canonical_hash(seal["pre_state"]) == row["state_hash"], "probe_state")
        probe = time.monotonic() - began
        a.progress("cheap_unit_after", 1, 0)
    projection = probe * len(a.census(frozen, "projection")) * 5 + 20
    checks["projected_time"] = custody + projection + 900 < 4800
    pre = dict(
        passed=all(checks.values()),
        checked=checks,
        gates=gates,
        sources=sources,
        import_and_custody_s=custody,
        measured_cheap_unit_s=probe,
        projected_work_s=projection,
        validation_reserve_s=900,
        available_memory_bytes=available,
        free_disk_bytes=free,
        historical_resolution="Exact preserved V752/V753 files; no mutable vNEXT historical lookup.",
        seed_authority=dict(current=[11, 23, 47], historical=[11, 23, 37]),
    )
    return pre, source, frozen


def run(raw: Path, date: str, *, evidence: tuple[Json, Json] | None = None) -> Json:
    """Preserve input bytes before scoring and leave readiness zero until qualification."""
    start = time.monotonic()
    a.progress("preconditions_before")
    pre, source, frozen = preflight(raw, evidence)
    a.progress("preconditions_after")
    atomic_json(raw / "evidence.json", source)
    atomic_json(raw / "frozen_manifest.json", frozen)
    atomic_json(raw / "preconditions.json", pre)
    reduced = dict(
        rows=a.census(frozen, "external_precondition_failed"), leakage_checks=dict(checks={})
    )
    summary: Json = dict(
        familywise_intervals=[], coverage_rows=[], retention_checks={}, headroom_checks={}
    )
    failure = ""
    if pre["passed"]:
        a.progress("independent_reduction_before", 0, len(reduced["rows"]))
        try:
            reduced = a.reduce(source, frozen)
            if all(r["status"] == "completed" for r in reduced["rows"]):
                summary = a.summarize(reduced["rows"])
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            failure = str(error)
        a.progress(
            "independent_reduction_after",
            sum(r["status"] == "completed" for r in reduced["rows"]),
            0,
        )
    complete = (
        pre["passed"] and not failure and all(r["status"] == "completed" for r in reduced["rows"])
    )
    gates = a.benefit_gates(reduced, summary) if complete else {}
    verdict = "disqualified" if failure or complete else "blocked"
    value = dict(
        experiment_id=8810,
        task_id="exp8810-causal-learning-audit",
        milestone="2026.11.754",
        run_date=date,
        honest_verdict="complete_" + verdict + "_pending_qualification",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        verifier_is_oracle=True,
        duration_s=time.monotonic() - start,
        random_seed=8810,
        source_artifact_hashes=pre["sources"],
        code_config_hashes=[a.reference(ROOT / p) for p in OWNED + TESTS],
        raw_shard_hashes=[
            a.reference(raw / name)
            for name in ["evidence.json", "frozen_manifest.json", "preconditions.json"]
        ],
        preconditions_checked=pre,
        phase_spans=[
            dict(phase="custody_and_independent_audit", duration_s=time.monotonic() - start)
        ],
        **reduced,
        **summary,
        intended_count=len(reduced["rows"]),
        completed_count=sum(r["status"] == "completed" for r in reduced["rows"]),
        censored_count=sum(r["status"] == "censored" for r in reduced["rows"]),
        failed_count=0,
        excluded_count=0,
        independent_count=len(frozen["streams"]) if complete else 0,
        n_samples=len(frozen["streams"]) if complete else 0,
        gate_check_summary=pre["gates"]
        + [
            dict(
                check=k,
                upstream="exp8810-preconditions",
                artifact_field=k,
                expected=True,
                observed=v,
                passed=v,
                **a.reference(raw / "preconditions.json"),
            )
            for k, v in pre["checked"].items()
        ],
        acceptance_gates=dict(execution=complete, owned_validation=False, **gates),
        required_checks_passed=False,
        owned_statement_coverage=dict(complete=False),
        validation_receipts=[],
        exposure_scope="constructed_encoded_graphs_with_exact_oracle_labels",
        independent_generalization_score=0,
        learning_benefit_ready_score=0,
        independently_recomputed_rows=a.row_receipt(reduced["rows"]),
        claim_limits=[
            "Independent of producer aggregation; not independent external reproduction.",
            "Constructed feedback gains are circular_positive; no live hallucination reduction or independent generalization.",
            "A natural-data test requires new source clusters, prospective seals, delayed independently obtained labels, frozen budgets and thresholds, and the same retention and familywise gates.",
            "Complete snapshot identities and tail final state are authenticated; numerical gradients are not rerun.",
        ],
        protocol_scope="private_fixture" if evidence is not None else "production",
        owned_failure=failure,
        methodology_note="Three-way Brier from sealed probabilities. Future ordinals9-31. Average nested seeds within streams;10000 whole-stream draws; ten Bonferroni comparisons with familywise99.5 percent intervals. Error intervals use incidence per scheduled future slot; coverage and class denominators are separate.",
        global_reconciliation_debt="One bounded whole-suite receipt is external to owned qualification; conductor owns ops and traceability reconciliation.",
        cited_upstream_artifacts=[
            dict(experiment_id=i, fields_imported=fields, sha256=pin)
            for i, fields, pin in [
                (
                    8804,
                    [
                        "per_query_rows",
                        "retention_rows",
                        "release_order_rows",
                        "complete_learning_cost_rows",
                        "heads",
                        "input_refs",
                        "final_state_refs",
                        "raw_shard_hashes",
                    ],
                    a.PIN,
                ),
                (8802, ["frozen_memory_protocol", "source_artifact_hashes"], a.PROTOTYPE_PIN),
            ]
        ],
        field_principles={},
    )
    value["reproducibility_checksum"] = a.canonical_hash(dict(reduced=reduced, summary=summary))
    return value


def qualify(value: Json, checked: Json) -> None:
    """Scientific nulls can pass execution checks; failed owned validation disqualifies."""
    value["validation_receipts"] = checked["receipts"]
    value["owned_statement_coverage"] = checked["coverage"]
    value["acceptance_gates"]["owned_validation"] = checked["passed"]
    execution = value["acceptance_gates"]["execution"]
    value["required_checks_passed"] = bool(execution and checked["passed"])
    benefit = int(value["required_checks_passed"] and all(value["acceptance_gates"].values()))
    value["learning_benefit_ready_score"] = benefit
    if not checked["passed"] or value["owned_failure"]:
        verdict = "disqualified"
    elif not execution:
        verdict = "blocked"
    else:
        verdict = "circular_positive" if benefit else "null"
    value["verdict_class"] = verdict
    value["honest_verdict"] = "complete_" + verdict + "_causal_learning_audit"
    value["field_principles"] = {
        key: "Measured or frozen audit operand; exact-oracle constructed evidence cannot establish independent generalization."
        for key in value
    }
    value["field_principles"].update(
        learning_benefit_ready_score="Requires both causal Brier margins, six retention bounds, two error bounds, real changed checks, shuffled control, complete denominators and owned validation.",
        required_checks_passed="Execution and owned validation are separate from causal benefit; a measured null can pass.",
        familywise_intervals="Ten comparisons share whole-stream draws, with nested seeds averaged and Bonferroni familywise99.5 percent coverage.",
        independently_recomputed_rows="Structured census counts and a canonical row hash reconcile the same authenticated slots; they are not independent outcome metrics.",
        independent_count="Reset streams are independent clusters; seeds, arms and retention siblings do not add independent units.",
        numerical_training_calls="Zero gradients, fits, model loads or generations in this aggregation audit.",
    )


def replay(path: Path) -> Json:
    """Rebuild from pinned operands; repaired outer hashes cannot invent a causal gain."""
    try:
        value = json.loads(path.read_bytes())
        refs = (
            value["raw_shard_hashes"]
            + value["code_config_hashes"]
            + value["source_artifact_hashes"]
        )
        a.require(all(a.sha256_file(Path(ref["path"])) == ref["sha256"] for ref in refs), "custody")
        source, frozen, pre = [a.load(ref) for ref in value["raw_shard_hashes"][:3]]
        if value["protocol_scope"] == "production":
            a.require(frozen == frozen_manifest(), "production_roster")
            if pre["passed"]:
                a.require(
                    a.sha256_file(ROOT / a.UPSTREAM) == a.PIN
                    and a.sha256_file(ROOT / a.PROTOTYPE) == a.PROTOTYPE_PIN,
                    "primary_pins",
                )
                a.require(
                    source == json.loads((ROOT / a.UPSTREAM).read_bytes()), "source_projection"
                )
        else:
            a.require(value["protocol_scope"] == "private_fixture", "scope")
        reduced = dict(
            rows=a.census(frozen, "external_precondition_failed"), leakage_checks=dict(checks={})
        )
        summary: Json = dict(
            familywise_intervals=[], coverage_rows=[], retention_checks={}, headroom_checks={}
        )
        if pre["passed"] and not value["owned_failure"]:
            reduced = a.reduce(source, frozen)
            if all(r["status"] == "completed" for r in reduced["rows"]):
                summary = a.summarize(reduced["rows"])
        expected = deepcopy(value)
        expected.update(reduced, **summary)
        complete = (
            pre["passed"]
            and not value["owned_failure"]
            and all(r["status"] == "completed" for r in reduced["rows"])
        )
        scientific = a.benefit_gates(reduced, summary) if complete else {}
        expected["acceptance_gates"] = dict(
            execution=complete, owned_validation=False, **scientific
        )
        checked = dict(
            receipts=value["validation_receipts"],
            coverage=value["owned_statement_coverage"],
            passed=bool(value["acceptance_gates"]["owned_validation"]),
        )
        if checked["passed"]:
            a.require(
                bool(checked["receipts"]) and all(r["passed"] for r in checked["receipts"]),
                "validation_receipts",
            )
            ref = checked["coverage"]["combined_reference"]
            a.require(
                coverage_summary(Path(ref["path"]), OWNED) == checked["coverage"]
                and checked["coverage"]["complete"],
                "actual_coverage",
            )
        qualify(expected, checked)
        # Before qualification the explicit pending verdict still has zero readiness.
        if value["honest_verdict"].endswith("_pending_qualification"):
            expected.update(
                verdict_class="disqualified" if complete or value["owned_failure"] else "blocked",
                honest_verdict=value["honest_verdict"],
                field_principles=value["field_principles"],
            )
        expected.update(
            intended_count=len(reduced["rows"]),
            completed_count=sum(r["status"] == "completed" for r in reduced["rows"]),
            censored_count=sum(r["status"] == "censored" for r in reduced["rows"]),
            failed_count=0,
            excluded_count=0,
            independently_recomputed_rows=a.row_receipt(reduced["rows"]),
            independent_count=len(frozen["streams"]) if complete else 0,
            reproducibility_checksum=a.canonical_hash(dict(reduced=reduced, summary=summary)),
        )
        keys = (
            set(reduced)
            | set(summary)
            | {
                "acceptance_gates",
                "required_checks_passed",
                "learning_benefit_ready_score",
                "verdict_class",
                "intended_count",
                "completed_count",
                "censored_count",
                "failed_count",
                "excluded_count",
                "independently_recomputed_rows",
                "independent_count",
                "reproducibility_checksum",
            }
        )
        a.require(all(expected[k] == value[k] for k in keys), "reduction_drift")
        a.require(
            value["numerical_training_calls"] == 0
            and value["MODEL_SPECS"] == []
            and value["model_invocation_counts"] == ZERO_INVOCATION_COUNTS
            and value["independent_generalization_score"] == 0,
            "claim_scope",
        )
        return dict(passed=True, reconstructed_rows=len(reduced["rows"]))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return dict(passed=False, reason=str(error))


def validate(raw: Path) -> Json:
    """Combine real unit and child SQLite coverage; all new statements need evidence."""
    covdir = raw / "coverage"
    covdir.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot8810-child-coverage-"))
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=True\ndata_file="
        + str(covdir / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
        + "[report]\nexclude_lines=\n"
    )
    py = str(ROOT / ".venv/bin/python")
    plan = [
        (
            "owned_pytest",
            [
                "env",
                "CARNOT_8810_COVERAGE_DIRECTORY=" + str(private),
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
            300,
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
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS], 60),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS], 60),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
            120,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS], 60),
    ]
    atomic_json(raw / "validation_manifest.json", dict(commands=plan, owned=OWNED, tests=TESTS))
    receipts = [
        child(name, argv, raw / "validation", deadline=deadline, heartbeat=60)
        for name, argv, deadline in plan
    ]
    for shard in private.glob(".coverage.child.*"):
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
    """Use unchanged shared scanners and a fresh process outside the checkout."""
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
    """Requalify exact retained measurements after a schema repair, preserving old receipts."""
    began = time.monotonic()
    checked = validate(raw)
    qualify(value, checked)
    value["phase_spans"].append(dict(phase="owned_validation", duration_s=time.monotonic() - began))
    value["duration_s"] += time.monotonic() - began
    health = ROOT / "results/raw" / NAME / "global-health/full_python_once.receipt.json"
    if health.is_file():
        value["repository_health_reference"] = a.reference(health)
        value["field_principles"]["repository_health_reference"] = (
            "One unchanged full-suite health diagnostic; owned coverage is assessed separately."
        )
    atomic_json(raw / "prepublication.json", value)
    a.progress("terminal_checks_before")
    try:
        published = publish_primary(output, value, lambda path: terminal_checks(path, raw))
    except ValueError:
        atomic_json(raw / "rejected_candidate.json", value)
        raise
    atomic_json(raw / "publication.json", published)
    a.progress("terminal_publication_complete", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Run the audit through a direct CLI whose imports do not depend on cwd."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        report = replay(args.replay)
        print(json.dumps(report), flush=True)
        return int(not report["passed"])
    raw = (args.raw or ROOT / "results/raw" / NAME / "invocations" / str(time.time_ns())).absolute()
    value = run(raw, args.date)
    qualify_and_publish(value, raw, args.output)
    print(
        json.dumps(
            {
                k: value[k]
                for k in [
                    "verdict_class",
                    "learning_benefit_ready_score",
                    "completed_count",
                    "independent_count",
                ]
            }
        ),
        flush=True,
    )
    return 0
