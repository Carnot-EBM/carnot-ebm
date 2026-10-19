"""REQ-REPORT-8082: task completion and exposed development science are separate."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone, UTC
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v698_capstone as previous
from carnot.reporting import v699_capstone_reduction as reduction
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
)
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v686_contract_validation import run_check

Json = dict[str, Any]
ROOT = previous.ROOT
MILESTONE = "2026.10.699"
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
INPUT = "results/experiment_8070_v699_contract_custody.json"
CLI = "scripts/experiments/experiment_8082_v699_capstone.py"
TEST = "tests/python/test_v699_capstone_8082.py"
OWNED = [
    "python/carnot/reporting/v699_capstone.py",
    "python/carnot/reporting/v699_capstone_reduction.py",
    CLI,
]
NAMED = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    DESIGN,
    "python/carnot/reporting/v698_capstone.py",
    "python/carnot/reporting/v698_capstone_reduction.py",
    "python/carnot/reporting/roadmap_contract.py",
    "scripts/publication_gate.py",
    "results/experiment_8069_v698_capstone.json",
    "ops/north-star.md",
]
START = time.monotonic()
read = previous.read
reference = previous.reference
failure = previous.failure


def progress(phase: str, units: int = 0, pending: str = "") -> None:
    """Actual completed units and elapsed time make stalls visible to the conductor."""
    print(
        f"[exp8082] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed_units={units} pending={pending}",
        flush=True,
    )


def authorities(root: Path) -> tuple[list[Json], Json, list[Json]]:
    """Compare full preserved invocation bytes; consumed staging is never recreated."""
    data = read(root / INPUT)
    failures = []
    try:
        snapshots = data["authority_snapshots"]
        active, design = [
            checked(dict(path=snapshots[k]["snapshot_path"], sha256=snapshots[k]["sha256"]))
            for k in ("active", "design")
        ]
        text = design.read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        invocation = yaml.safe_load(active.read_bytes())
        digest = re.search(r"Canonical (?:complete-|full-)?task SHA-?256: `([0-9a-f]{64})`", text)
        shown = [
            dict(order=i + 1, **{k: t[k] for k in ("id", "title", "phase", "deliverable")})
            for i, t in enumerate(tasks)
        ]
        if (
            invocation["tasks"] != tasks
            or invocation["milestone"] != MILESTONE
            or table != shown
            or digest is None
            or digest[1] != tasks_digest(tasks)
            or tasks_digest(tasks) != data["canonical_tasks_sha256"]
            or [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(8070, 8083)]
        ):
            raise ValueError("immutable_authority_drift")
        staged = snapshots.get("staged", {})
        if staged.get("exists"):
            path = checked(dict(path=staged["snapshot_path"], sha256=staged["sha256"]))
            if yaml.safe_load(path.read_bytes())["tasks"] != tasks:
                raise ValueError("staged_invocation_drift")
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as error:
        failures.append(
            failure(
                root / INPUT,
                "exp8070",
                "immutable_authority",
                "authenticated thirteen-task invocation",
                str(error),
            )
        )
        design = root / DESIGN if (root / DESIGN).is_file() else ROOT / DESIGN
        _, tasks = parse_design(design.read_text(), milestone=MILESTONE)
    return tasks, data, failures


def claim(value: Json) -> Json:
    """Seal conclusions separately from a file's custody so altered claims fail replay."""
    return {
        k: value[k]
        for k in (
            "task_contract",
            "canonical_tasks_sha256",
            "H1",
            "H2",
            "primary_hypothesis_results",
            "gap_decisions",
            "science_ready",
            "service_partition_join",
            "retirement_rows",
            "historical_dispositions",
            "arc_monitoring",
            "board_custody",
        )
    }


def collect(root: Path, tasks: list[Json]) -> tuple[list[Json], list[Json], list[Json]]:
    """An invalid external identity closes only its branch, preserving other outcomes."""
    rows, refs, failures = [], [], []
    for task in tasks[:-1]:
        try:
            selected, inputs, issues = previous.collect(root, [task, tasks[-1]])
            rows.extend(selected)
            refs.extend(inputs)
            failures.extend(issues)
        except (ValueError, KeyError, TypeError, OSError) as error:
            path = root / task["deliverable"]
            issue = failure(
                path,
                task["id"],
                "upstream_identity",
                "authenticated task, path and skip hash",
                str(error),
            )
            rows.append(
                dict(
                    task_id=task["id"],
                    id=task["id"],
                    unit_id=task["id"],
                    unit=task["id"],
                    source=str(path),
                    path=str(path),
                    sha256=reference(path)["sha256"],
                    arm="task_disposition",
                    seed=None,
                    primary_present=path.is_file(),
                    producer_status="invalid_identity",
                    honest_verdict="complete_blocked_upstream_identity",
                    verdict_class="blocked",
                    eligible=False,
                    excluded=True,
                    started=path.is_file(),
                    completed=True,
                    failed=False,
                    censored=False,
                    status="completed",
                    numerator=0,
                    denominator=1,
                    raw_numerator=0,
                    raw_denominator=1,
                    exclusion_reason="unauthenticated_external_identity",
                    gate_check_summary=[issue],
                )
            )
            refs.append(reference(path, "invalid_identity"))
            failures.append(issue)
    return rows, refs, failures


def build(root: Path, date: str, durable: Path) -> Json:
    """Preserve each branch and independently reduce primitive evidence before claiming benefit."""
    progress("preconditions_before", pending="named inputs and Python tools")
    tasks, authority, failures = authorities(root)
    preconditions = [
        dict(reference(root / p), check="resource_exists", passed=(root / p).is_file())
        for p in NAMED
    ]
    preconditions += [
        dict(
            reference(ROOT / ".venv/bin" / p),
            check="tool_exists",
            passed=(ROOT / ".venv/bin" / p).is_file(),
        )
        for p in ("python", "pytest", "coverage", "ruff", "mypy")
    ]
    preconditions.append(
        dict(
            path=sys.executable,
            sha256=sha256_file(Path(sys.executable)),
            check="python_version>=3.11",
            passed=sys.version_info >= (3, 11),
        )
    )
    failures.extend(
        failure(Path(p["path"]), "preconditions", p["check"], True, False)
        for p in preconditions
        if not p["passed"]
    )
    frozen = dict(
        task_contract=tasks,
        canonical_tasks_sha256=tasks_digest(tasks),
        margins=[0.02, 0.02],
        draws=10000,
        alpha=0.05,
        blocks=[32, 16, 64],
        absent_invalid_unsafe_p=1,
        source_support=[72, 8],
        learning_support=[80, 10],
        retention_support=[48, 8],
        exposure="historically exposed development; seeds averaged within source",
    )
    atomic_json(durable / "method_freeze.json", frozen)
    progress("preconditions_after", len(preconditions), "task authentication")
    rows, refs, issues = collect(root, tasks)
    failures.extend(issues)
    audits = []
    for row in rows:
        row["condition"] = "actual_terminal_disposition"
        progress(
            "benchmark_before_independent_" + row["task_id"],
            len(audits),
            "primitive reconstruction",
        )
        try:
            audit = reduction.independent(read(Path(row["path"])), int(row["task_id"][3:7]))
        except (ValueError, KeyError, TypeError, OSError, IndexError, TimeoutError) as error:
            audit = dict(measurement_available=False, reduction_error=str(error))
            gate = failure(
                Path(row["path"]),
                row["task_id"],
                "independent_reduction",
                "valid primitive equations",
                str(error),
            )
            failures.append(gate)
            row["gate_check_summary"].append(gate)
            row.update(eligible=False, excluded=True, numerator=0, raw_numerator=0)
        path = durable / "reductions" / (row["task_id"] + ".json")
        atomic_json(path, audit)
        audits.append(
            dict(
                task_id=row["task_id"],
                **reference(path, "independent_reduction"),
                measurement_available=audit["measurement_available"],
            )
        )
        progress(
            "benchmark_after_independent_" + row["task_id"], len(audits), str(12 - len(audits))
        )
    results = [read(Path(a["path"])) for a in audits]
    family = reduction.holm(
        results[4].get("H1"), results[7].get("H2"), (rows[4]["eligible"], rows[7]["eligible"])
    )
    for hypothesis, audit in zip(family, [audits[4], audits[7]], strict=True):
        for field, expected, observed, op, passed in (
            (
                "beneficial_changed_sources",
                5,
                hypothesis["beneficial_changed_sources"],
                ">=",
                hypothesis["beneficial_changed_sources"] >= 5,
            ),
            (
                "observed_gain",
                0.02,
                hypothesis["observed_gain"],
                ">=",
                hypothesis["observed_gain"] is not None and hypothesis["observed_gain"] >= 0.02,
            ),
            (
                "holm_adjusted_p",
                0.05,
                hypothesis["holm_adjusted_p"],
                "<=",
                hypothesis["holm_adjusted_p"] <= 0.05,
            ),
        ):
            if not passed:
                gate = failure(
                    Path(audit["path"]),
                    audit["task_id"],
                    hypothesis["hypothesis"] + "." + field,
                    expected,
                    observed,
                )
                gate.update(op=op, classification="scientific")
                failures.append(gate)
    cache = [read(root / tasks[i]["deliverable"]) for i in (8, 9)]
    progress("benchmark_before_service_join", pending="exact six-mode identities")
    join = reduction.service_join(*cache, (rows[8]["eligible"], rows[9]["eligible"]))
    progress(
        "benchmark_after_service_join", len(join["mode_condition_rows"]), "three independent gaps"
    )
    prior = read(root / "results/experiment_8069_v698_capstone.json")
    gaps = dict(
        useful_source_verification=dict(
            requirements=["FR-06", "FR-12"],
            closed=False,
            decision="exposed_development_H1_margin_or_changed_source_floor_failed"
            if results[4].get("H1")
            else "blocked_H1_primitives",
            primary_evidence=family[0],
            reopen_condition="Collect unexposed complete source groups after freezing interaction/additive heads; pass .02 Holm margin, five beneficial changes and safety.",
        ),
        retained_causal_learning=dict(
            requirements=["FR-11"],
            closed=False,
            decision="projected_versus_ray_later_benefit_null"
            if results[7].get("H2")
            else "blocked_H2_primitives",
            primary_evidence=family[1],
            reopen_condition="Change candidate or feedback information; pass later .02 margin with five beneficial source changes and both guarded retention limits on unexposed sources.",
        ),
        reproducible_deployment=dict(
            requirements=["FR-05", "FR-08", "FR-09", "FR-10", "NFR-01"],
            closed=False,
            decision="bounded_host_cache_transactions_only_complete_acquisition_unpriced",
            complete_six_mode_credit=join["complete_six_mode_credit"],
            reopen_condition="Qualify useful decisions; price acquisition, original inference, external feedback, transport and recovery; reproduce complete native throughput >=10x.",
        ),
    )
    state = (
        "blocked"
        if any(g.get("classification") != "scientific" for g in failures)
        or any(r["verdict_class"] in ("blocked", "disqualified") for r in rows)
        else "null"
    )
    verdict = "complete_" + state + "_v699_capstone"
    own = dict(
        task_id=tasks[-1]["id"],
        id=tasks[-1]["id"],
        unit=tasks[-1]["id"],
        unit_id=tasks[-1]["id"],
        source="current_owned_reader",
        arm="task_disposition",
        seed=None,
        condition="owned_validation",
        path=None,
        sha256=None,
        primary_present=False,
        producer_status="pending_normal_exit",
        honest_verdict=verdict,
        verdict_class=state,
        completed=False,
        started=True,
        eligible=False,
        excluded=False,
        failed=False,
        censored=False,
        status="pending",
        numerator=0,
        denominator=1,
        raw_numerator=0,
        raw_denominator=1,
        exclusion_reason=None,
        gate_check_summary=[],
    )
    rows.append(own)
    retirement = reduction.retirements(tasks, rows, prior.get("task_dispositions", []))
    arc = read(root / tasks[10]["deliverable"])
    arc_monitoring = dict(
        producer_verdict=arc.get("honest_verdict"),
        producer_class=arc.get("verdict_class"),
        new_firing_count=arc.get("new_firing_count"),
        new_outcome_count=arc.get("new_outcome_count"),
        empty_delta_observed=arc.get("no_new_outcomes"),
        monitoring_qualified=bool(rows[10]["eligible"] and arc.get("arc_delta_ready_score") == 1),
        candidate_refinement=arc.get("candidate_refinement"),
        frontier=arc.get("frontier"),
        obligation="An authenticated valid empty delta remains mandatory monitoring; an external missing producer blocks qualification.",
    )
    refs += [
        reference(Path(s["snapshot_path"]), "authority_" + k)
        for k, s in authority.get("authority_snapshots", {}).items()
        if s.get("exists")
    ]
    refs += [
        reference(root / INPUT),
        reference(root / "results/experiment_8069_v698_capstone.json"),
    ]
    refs += [reference(ROOT / p, "current_code") for p in OWNED]
    refs.append(reference(reduction.HISTORICAL_SPEC, "archived_producer_spec"))
    unique = {r["path"]: r for r in refs}
    # Existing immutable upstream shards remain hash-bound; small owned snapshots retain their original identities.
    saved = [
        previous.previous.save(r, durable)
        for r in unique.values()
        if r["role"] != "primitive_shard" and Path(r["path"]).is_file()
    ]
    value: Json = dict(
        experiment_id=8082,
        experiment=8082,
        task_id="exp8082-capstone",
        milestone=MILESTONE,
        schema="carnot.v699.capstone.v1",
        title="Independent V699 outcome accounting",
        run_date=date,
        started_at=datetime.now(UTC).isoformat(),
        finished_at=None,
        status=state,
        input_root=str(root),
        honest_verdict=verdict,
        verdict_class=state,
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        canonical_tasks_sha256=tasks_digest(tasks),
        gate_check_summary=failures,
        H1=family[0],
        H2=family[1],
        primary_hypothesis_results=family,
        independent_reduction_rows=audits,
        gap_decisions=gaps,
        science_ready=False,
        generalized_learning_benefit_score=0,
        service_partition_join=join,
        retirement_rows=retirement,
        reopen_conditions=[g["reopen_condition"] for g in gaps.values()],
        next_research_recommendations=[g["reopen_condition"] for g in gaps.values()],
        historical_dispositions=prior.get("task_dispositions", []),
        arc_monitoring=arc_monitoring,
        board_custody=results[11].get("board_rows", []),
        authority_snapshots=authority.get("authority_snapshots", {}),
        contract_ready_score=int(not authorities(root)[2]),
        capstone_execution_ready_score=0,
        required_checks_passed=False,
        flagged_adversarial=False,
        verifier_is_oracle=False,
        validation_receipts=[],
        coverage_statement_counts={},
        preconditions_checked=preconditions,
        named_staged_input=dict(
            reference(root / "research-roadmap-next.yaml"),
            disposition="consumed_at_activation"
            if not (root / "research-roadmap-next.yaml").exists()
            else "present",
        ),
        source_artifact_hashes=list(unique.values()),
        raw_shard_hashes=saved,
        code_config_hashes=[reference(ROOT / p, "current_code") for p in OWNED],
        checkpoint_references=[reference(durable / "method_freeze.json", "method_freeze")] + audits,
        random_seed=6998072,
        reproducibility_checksum=canonical_hash(dict(freeze=frozen, inputs=unique)),
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[
            s
            for i in (4, 7)
            for s in read(root / tasks[i]["deliverable"]).get("trained_head_specs", [])
        ],
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"),
        ),
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        substrate_declaration=dict(
            custody="aggregation_from_upstream_artifacts",
            pretrained="no_model_load",
            MODEL_SPECS=[],
        ),
        claim_scope="Exact task custody and historically exposed development reductions; no generalized learning, new live solves or device speed claims.",
        methodology_note="Reopen primitive predictions, labels, causal SQLite journals, cache costs and historical board operands; independent two-test Holm family.",
        terminal_validation_sidecar_path=str(durable / "terminal_validation.json"),
        acceptance_gate_results=dict(validity=False, readiness=0),
        repository_health=[],
        publication_gate_results={},
        paper_ready=False,
        unmet_gates=["publication_gate_pending"],
        phase_spans=[],
        sample_size_budget=dict(
            intended=13,
            completed=12,
            eligible=sum(r["eligible"] for r in rows),
            excluded=sum(r["excluded"] for r in rows),
            failed=sum(r["failed"] for r in rows),
            censored=0,
            independent=0,
            unit="administrative_task_disposition; independent H1/H2 source counts are separate",
        ),
    )
    atomic_json(durable / "reduction_identity.json", dict(payload=claim(value)))
    value["checkpoint_references"].append(
        reference(durable / "reduction_identity.json", "reduction_identity")
    )
    return value


def cold_replay(value: Json) -> list[str]:
    """Hash-check inputs and repeat independent equations rather than trust producer totals."""
    progress("cold_replay_before", pending="immutable evidence")
    for ref in (
        value["source_artifact_hashes"]
        + value["raw_shard_hashes"]
        + value["code_config_hashes"]
        + value["checkpoint_references"]
    ):
        if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            return ["source_bytes_changed"]
        if ref["path"].endswith(".shards.json"):
            shards = read(Path(ref["path"]))
            digest = hashlib.sha256()
            for part in shards["shards"]:
                if reference(Path(part["path"]))["sha256"] != part["sha256"]:
                    return ["snapshot_shard_changed"]
                digest.update(Path(part["path"]).read_bytes())
            if "sha256:" + digest.hexdigest() != shards["original_sha256"]:
                return ["snapshot_reconstruction_drift"]
    identity = next(r for r in value["checkpoint_references"] if r["role"] == "reduction_identity")
    if read(Path(identity["path"]))["payload"] != claim(value):
        return ["reduction_claim_drift"]
    for row, ref in zip(value["rows"][:-1], value["independent_reduction_rows"], strict=True):
        try:
            observed = reduction.independent(read(Path(row["path"])), int(row["task_id"][3:7]))
        except (ValueError, KeyError, TypeError, OSError, IndexError, TimeoutError) as error:
            observed = dict(measurement_available=False, reduction_error=str(error))
        if observed != read(Path(ref["path"])):
            return ["independent_reduction_drift"]
    seals = [r for r in value["checkpoint_references"] if r["role"] == "claim_seal"]
    if seals and read(Path(seals[0]["path"])) != {
        k: v for k, v in value.items() if k not in ("checkpoint_references", "field_principles")
    }:
        return ["terminal_claim_drift"]
    progress("cold_replay_after", 12, "none")
    return []


def manifest(private: Path) -> Json:
    """Freeze the existing checks with coverage scoped only to this new reader."""
    value = previous.manifest(private)
    mapping = dict(zip(previous.OWNED + [previous.TEST], OWNED + [TEST], strict=True))
    for spec in value["commands"]:
        spec["argv"] = [mapping.get(a, a) for a in spec["argv"]]
        spec["argv"] = [
            "--include=" + ",".join(OWNED) if a.startswith("--include=") else a
            for a in spec["argv"]
        ]
    py = str(ROOT / ".venv/bin/python")
    fixture = private / "historical-intervention.json"
    old_cli = str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py")
    value["commands"] += [
        dict(
            name="E2E-016_fixture",
            argv=[py, old_cli, "--date", "20260929", "--fixture-e2e", str(fixture)],
            expected_exit=0,
            deadline_s=180,
        ),
        dict(
            name="E2E-016_cold",
            argv=[py, old_cli, "--date", "20260929", "--cold-replay", str(fixture)],
            expected_exit=0,
            deadline_s=180,
        ),
    ]
    value.update(
        coverage_includes=OWNED,
        dependency_hashes={p: sha256_file(ROOT / p) for p in OWNED + [TEST]},
    )
    value["repository_health_command"] = dict(
        name="full_python_suite",
        argv=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        expected_exit=0,
        deadline_s=600,
        classification="diagnostic",
    )
    return value


def complete(value: Json, receipts: list[Json], counts: Json) -> None:
    """Owned checks qualify administrative execution while external blocks stay terminal."""
    valid = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and all(
            p in counts
            and counts[p]["num_statements"] > 0
            and counts[p]["num_statements"] == counts[p]["covered_lines"]
            for p in OWNED
        )
    )
    value.update(
        validation_receipts=receipts,
        coverage_statement_counts=counts,
        required_checks_passed=valid,
        capstone_execution_ready_score=int(valid),
    )
    if not valid:
        value.update(
            honest_verdict="complete_disqualified_v699_owned_checks",
            verdict_class="disqualified",
            contract_ready_score=0,
        )
    value["rows"][-1].update(
        completed=True,
        eligible=valid,
        failed=not valid,
        numerator=int(valid),
        raw_numerator=int(valid),
        status="completed",
        producer_status="normal_reduction_child_exit",
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
    )
    value["acceptance_gate_results"].update(validity=valid, readiness=int(valid))
    budget = value["sample_size_budget"]
    budget.update(
        completed=13,
        eligible=sum(r["eligible"] for r in value["rows"]),
        failed=sum(r["failed"] for r in value["rows"]),
    )
    value.update({k + "_count": v for k, v in budget.items() if k != "unit"})


def publish(value: Json, output: Path, private: Path, durable: Path, *, fixture: bool) -> None:
    """Only checked bytes become the primary; failed owned terminal checks disqualify it."""
    private.mkdir(parents=True, exist_ok=True)
    candidate = private / output.name
    seal = durable / "claim_seal.json"
    atomic_json(
        seal,
        {k: v for k, v in value.items() if k not in ("checkpoint_references", "field_principles")},
    )
    value["checkpoint_references"].append(reference(seal, "claim_seal"))
    value["field_principles"] = {
        k: "Bind exact operands to their scope; task or fixture completion cannot supply independent scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        honest_verdict="A completed negative finding is terminal and must not trigger an unchanged retry.",
        verdict_class="Missing external evidence is blocked; only unfinished owned work is partial.",
        rows="Raw numerators, denominators and dispositions prevent an absent measurement being reported as measured zero.",
        task_dispositions="Thirteen accounted outcomes can include blocks and nulls without thirteen scientific successes.",
        sample_size_budget="Administrative counts, repeated timings and seeds do not add independent source groups.",
        H1="Source interactions must beat the preregistered additive margin with support and safety.",
        H2="Feasible constraints and accepted updates cannot replace measured later learning benefit.",
        primary_hypothesis_results="Exactly two Holm tests remain registered even if one is missing or unsafe.",
        gap_decisions="Source utility, retained learning and deployability require separate evidence.",
        service_partition_join="Six-mode credit requires identical workloads, required shared code and native builds in both partitions.",
        publication_gate_results="Historical G1-G4 publication readiness does not establish current source or deployment benefit.",
        paper_ready="The historical publication conjunction is separate from this milestone's science readiness.",
        generalized_learning_benefit_score="Exposed development cohorts cannot establish general lifelong improvement.",
        trained_head_specs="Small numerical heads are distinct from current pretrained model loads.",
        model_invocation_counts="Only current model operations count; upstream cited operations cannot create current calls.",
        source_artifact_hashes="Exact input bytes must stay attributable even if a path later changes.",
        raw_shard_hashes="Durable evidence copies permit reconstruction without another scientific run.",
        phase_spans="Elapsed spans document actual work and never pad model duration floors.",
        validation_receipts="Normal child exits, durations and log hashes distinguish executed checks from planned checks.",
        capstone_execution_ready_score="Owned administrative completion does not override blocked external science.",
        retirement_rows="Exact failed mechanisms retire narrowly; valid ARC monitoring remains mandatory.",
    )
    atomic_json(candidate, value)
    py = str(ROOT / ".venv/bin/python")
    commands = [("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(candidate)])]
    if not fixture:
        commands += [
            ("adversarial", [py, "scripts/adversarial_verify.py", "--json", str(candidate)]),
            (
                "strict_rows",
                [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            ),
        ]
    receipts = []
    for name, argv in commands:
        progress("subprocess_before_" + name, len(receipts), "terminal checks")
        receipts.append(
            run_check(
                ROOT,
                dict(name=name, argv=argv, expected_exit=0, deadline_s=600),
                private,
                durable / "terminal_logs",
            )
        )
        progress("subprocess_after_" + name, len(receipts), "terminal checks")
    passed = all(r["passed"] for r in receipts) and (
        fixture or read(Path(receipts[-2]["log_path"])).get("flagged_count", 1) == 0
    )
    if not passed:
        value.update(
            required_checks_passed=False,
            capstone_execution_ready_score=0,
            contract_ready_score=0,
            honest_verdict="complete_disqualified_v699_terminal_checks",
            verdict_class="disqualified",
        )
        value["rows"][-1].update(
            honest_verdict=value["honest_verdict"],
            verdict_class="disqualified",
            eligible=False,
            failed=True,
            numerator=0,
            raw_numerator=0,
        )
        atomic_json(durable / "failed_terminal.json", dict(receipts=receipts, candidate=value))
        raise ValueError("terminal_validation_failed")
    digest = sha256_file(candidate)
    publication = publish_primary(
        output, value, lambda p: dict(passed=sha256_file(p) == digest, receipts=receipts)
    )
    readers = reader_receipt(
        value["task_id"],
        output.parent,
        field="capstone_execution_ready_score",
        expected=value["capstone_execution_ready_score"],
    )
    if not readers["passed"]:
        raise ValueError("published_reader_drift")
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, readers=readers, receipts=receipts, normal_process_exit=True),
    )


def outcome_note(value: Json, root: Path) -> None:
    """State terminal findings and their limits without rewriting historical outcomes."""
    path = root / "docs/research-notes/milestone_2026_10_699_outcomes.md"
    text = [
        "# V699 outcomes — 2026-10-03",
        "",
        "Conductor completion establishes outcome accounting. The three PRD gaps remain open.",
        "",
        "| Task | Terminal outcome |",
        "|---|---|",
    ]
    text.extend(f"| {r['task_id']} | {r['honest_verdict']} |" for r in value["task_dispositions"])
    text += [
        "",
        f"H1 source cost gain: {value['H1']['observed_gain']}. H2 projected-versus-ray gain: {value['H2']['observed_gain']}.",
        "Both registered .02 margin tests remain in Holm .05. Each adjusted p is "
        + str([h["holm_adjusted_p"] for h in value["primary_hypothesis_results"]])
        + ".",
        "These are historically exposed development cohorts. They establish no general lifelong learning benefit.",
        "",
        "Cache six-mode credit: "
        + str(value["service_partition_join"]["complete_six_mode_credit"])
        + ". Acquisition, original Qwen inference and external feedback remain unpriced.",
        "Warm reuse does not close the 10x native throughput requirement. Projection feasibility does not demonstrate future learning.",
        "ARC outcome: "
        + str(value["arc_monitoring"]["producer_verdict"])
        + ". A valid empty delta remains mandatory monitoring; missing qualification stays blocked.",
        "Board custody preserves KV260 fabric k_max<=5, PolarFire Linux CPU dispatch and GateMate JTAG 0xffffffff separately. No device ran in this capstone.",
        "",
        "Historical publication scope: paper_ready="
        + str(value["paper_ready"])
        + "; unmet_gates="
        + str(value["unmet_gates"])
        + ". It is separate from current science readiness.",
        "",
        "Reopen only with changed evidence or mechanisms:",
        "",
    ]
    text.extend("- " + recommendation for recommendation in value["next_research_recommendations"])
    text += [
        "",
        "V698 outcomes, absent primaries and prior terminal bytes are preserved. Retirement rows bind each declared prior failure and its narrow mechanism scope.",
        "Owned checks and coverage are recorded in the primary and terminal sidecar. Repository-wide test failures remain separate diagnostics.",
        "The conductor owns status, changelog and traceability reconciliation for this task.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text))


def main(argv: list[str] | None = None) -> int:
    """Supervise a bounded reduction child and freeze checks before its measurement."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start", pending="preconditions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--durable", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-e2e", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            errors = cold_replay(read(args.cold_replay))
        except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
            errors = [str(error)]
        print(json.dumps(dict(errors=errors)), flush=True)
        return int(bool(errors))
    if args.date != "20261003":
        raise ValueError("date_must_be_20261003")
    root = args.root.resolve()
    output = args.output or root / "results/experiment_8082_v699_capstone.json"
    durable = args.durable or output.parent / "raw" / output.stem / "invocations" / str(
        time.time_ns()
    )
    if args.worker:
        atomic_json(output, build(root, args.date, durable))
        return 0
    started = time.monotonic_ns()
    with tempfile.TemporaryDirectory(prefix="capstone8082-") as directory:
        private = Path(directory)
        frozen = manifest(private)
        worker = dict(
            name="reduction_normal_exit",
            expected_exit=0,
            deadline_s=600,
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--worker",
                "--root",
                str(root),
                "--date",
                args.date,
                "--durable",
                str(durable),
                "--output",
                str(private / "candidate.json"),
            ],
        )
        frozen["reduction_command"] = worker
        atomic_json(durable / "validation_manifest.json", frozen)
        progress("subprocess_before_reduction", pending="1 child")
        exited = run_check(ROOT, worker, private, durable / "validation_logs")
        progress("subprocess_after_reduction", 1, "owned checks")
        if not exited["passed"]:
            raise ValueError("reduction_child_failed")
        value = read(private / "candidate.json")
        receipts = [exited]
        if not args.fixture_e2e:
            for spec in frozen["commands"]:
                progress("subprocess_before_" + spec["name"], len(receipts), "owned checks")
                receipts.append(run_check(ROOT, spec, private, durable / "validation_logs"))
                progress("subprocess_after_" + spec["name"], len(receipts), "owned checks")
            health = os.environ.get("CARNOT_8082_HEALTH_RECEIPT")
            value["repository_health"] = (
                [read(Path(health))]
                if health
                else [
                    run_check(
                        ROOT,
                        frozen["repository_health_command"],
                        private,
                        durable / "validation_logs",
                    )
                ]
            )
            for health_row in value["repository_health"]:
                log = Path(health_row["log_path"])
                if sha256_file(log) != health_row["log_sha256"]:
                    raise ValueError("repository_health_log_drift")
                target = durable / "repository_health" / log.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(log, target)
                health_row["log_path"] = str(target)
            publication = read(Path(receipts[1]["log_path"]))
            value.update(
                publication_gate_results=publication,
                paper_ready=publication["paper_ready"],
                unmet_gates=publication["unmet_gates"],
                **{f"g{i}": publication["gates"][f"G{i}"]["pass"] for i in range(1, 5)},
            )
        report = read(private / "coverage.json")
        complete(value, receipts, {p: f["summary"] for p, f in report.get("files", {}).items()})
        if args.fixture_e2e:
            value.update(
                verifier_is_oracle=True,
                required_checks_passed=False,
                honest_verdict="complete_blocked_fixture_science",
                verdict_class="blocked",
                capstone_execution_ready_score=0,
            )
        ended = time.monotonic_ns()
        value["finished_at"] = datetime.now(UTC).isoformat()
        value["status"] = value["verdict_class"]
        value["duration_s"] = (ended - started) / 1e9
        value["phase_spans"] = [
            dict(
                phase="frozen_reduction_and_validation",
                start_s=0,
                end_s=value["duration_s"],
                completed_units=13,
            )
        ]
        value["current_work_receipt"] = build_current_work_receipt(
            run_id=str(started),
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=value["inference_substrate"],
            inference_substrate_details=dict(model_work=False),
            inference_substrate_class="no_model_load",
            execution_venue="host",
            started_monotonic_ns=started,
            ended_monotonic_ns=ended,
        )
        value["checkpoint_references"].append(
            reference(durable / "validation_manifest.json", "command_freeze")
        )
        outcome_note(value, root)
        publish(value, output, private / "terminal", durable, fixture=args.fixture_e2e)
    progress("published", 13, "none")
    return 0
