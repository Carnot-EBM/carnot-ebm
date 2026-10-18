"""Reduce actual evidence without upgrading audit completion to scientific benefit.

REQ-REPORT-7927-V687. Historical failures remain evidence after current repairs.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting import v686_capstone as shared
from carnot.reporting.v687_contract_methods import assess

ROOT = Path(__file__).resolve().parents[3]
GATES = ("G1", "G2", "G3", "G4")
REPLAY_FIELDS = (
    "rows",
    "outcome_rows",
    "independent_reduction_rows",
    "sample_size_budget",
    "gap_decisions",
    "canonical_tasks_sha256",
    "activation_confirmed",
    "reproducibility_checksum",
)


def progress(started: float, phase: str, units: int) -> None:
    """Flushed boundaries expose progress without extending the execution deadline."""
    print(
        f"[exp7927] phase={phase} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def reduce_primitives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce family observations while keeping scoring and stress channels separate."""
    reduced = shared.reduce_primitives(rows)
    families: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    groups: dict[str, list[float]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    stress: dict[str, list[float]] = defaultdict(list)
    scores: dict[str, list[float]] = defaultdict(list)
    family_groups: dict[str, str] = {}
    for row in rows:
        if "future_label" in row.get("feature_fields", []) or (
            "label_release_step" in row and row["label_release_step"] > row["decision_step"]
        ):
            raise ValueError("future_label_exposure")
        if "score_draw_seed" in row and row["score_draw_seed"] == row.get("stress_draw_seed"):
            raise ValueError("score_stress_channels_overlap")
        family = str(row.get("family_id", row.get("family", "unknown")))
        group = str(row.get("source_group", family))
        if family in family_groups and family_groups[family] != group:
            raise ValueError("family_source_group_conflict")
        family_groups[family] = group
        arm = str(row.get("arm", "default"))
        if row.get("status") == "completed" or row.get("completed") is True:
            if type(row.get("probability")) in (int, float):
                families[family][arm].append(row["probability"])
            if type(row.get("latency_ms")) in (int, float):
                timings[arm].append(row["latency_ms"])
            if "stress_failure" in row:
                stress[family].append(float(bool(row["stress_failure"])))
            if type(row.get("fragility_score")) in (int, float):
                scores[family].append(row["fragility_score"])
    grouped_fragility = []
    for channel in (stress, scores):
        grouped: dict[str, list[float]] = defaultdict(list)
        for family, values in channel.items():
            grouped[family_groups[family]].append(sum(values) / len(values))
        group_means = [sum(values) / len(values) for values in grouped.values()]
        grouped_fragility.append(sum(group_means) / len(group_means) if group_means else None)
    for family, arms in families.items():
        if "witness_neighbors" in arms and "matched_control" in arms:
            groups[family_groups[family]].append(
                sum(arms["witness_neighbors"]) / len(arms["witness_neighbors"])
                - sum(arms["matched_control"]) / len(arms["matched_control"])
            )
    deltas = [sum(values) / len(values) for values in groups.values()]
    reduced.update(
        seed_count=len({r["seed"] for r in rows if r.get("seed") is not None}),
        source_groups=len(set(family_groups.values())),
        source_sensitivity=dict(
            independent_source_groups=len(deltas),
            mean_neighbors_minus_filler=sum(deltas) / len(deltas) if deltas else None,
            natural_correctness=None,
        ),
        fragility=dict(
            stress_rows=sum(len(values) for values in stress.values()),
            independent_source_groups=len({family_groups[family] for family in stress}),
            stress_failure_rate=grouped_fragility[0],
            mean_fragility_score=grouped_fragility[1],
            independent_benefit=None,
        ),
        service_timing_ms={arm: sum(values) / len(values) for arm, values in timings.items()},
    )
    return reduced


def publication_operands(publication: dict[str, Any]) -> dict[str, Any]:
    """Keep stable publication booleans independent of current research readiness."""
    gates = {
        key: bool(publication.get("gates", {}).get(key, {}).get("pass", False)) for key in GATES
    }
    return dict(
        **gates,
        paper_ready=all(gates.values()),
        unmet_gates=[key for key in GATES if not gates[key]],
    )


def build_candidate(
    root: Path,
    design: Path,
    active: Path,
    date: str,
    publication: dict[str, Any] | None = None,
    *,
    snapshots: Path | None = None,
) -> dict[str, Any]:
    """Read each producer once; reserve a self row rather than borrowing an old result."""
    started, start_ns = time.monotonic(), time.monotonic_ns()
    progress(started, "start", 0)
    if date != "20260930":
        raise ValueError("v687_date_changed")
    directory = snapshots or Path(tempfile.mkdtemp(prefix="carnot-7927-authority-"))
    authority = assess(design, root / "research-roadmap-next.yaml", active, directory)
    try:
        tasks = yaml.safe_load(
            Path(authority["authority_snapshots"]["active"]["snapshot_path"]).read_bytes()
        )["tasks"]
        if [t["id"] for t in tasks] != [
            f"exp{n}-" + slug
            for n, slug in zip(
                range(7915, 7928),
                (
                    "contract-methods",
                    "training-qualification",
                    "intervention-qualification",
                    "energy-fit",
                    "decision-abstention",
                    "qwen-sufficiency",
                    "evidence-fragility",
                    "causal-acquisition",
                    "delayed-calibration",
                    "arc-supervisor-delta",
                    "service-cost",
                    "hardware-evidence",
                    "capstone",
                ),
                strict=True,
            )
        ]:
            raise ValueError("task_roster_changed")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        tasks = yaml.safe_load(
            gzip.decompress((ROOT / "tests/fixtures/v687/active.yaml.gz").read_bytes())
        )["tasks"]
    failures = list(authority["gate_check_summary"])
    sources = [
        dict(path=s["source_path"], sha256=s["sha256"], role=role, exposure="administrative")
        for role, s in authority["authority_snapshots"].items()
    ]
    # Annotated so mypy 2.2.0 does not infer a star-unpacked tuple type; it crashes
    # with an internal AssertionError on the tuple unpacking further down.
    observed: dict[str, tuple[Any, ...]] = {
        t["id"]: (*shared.read(root / t["deliverable"]), root / t["deliverable"])
        for t in tasks[:-1]
    }
    sources.extend(
        dict(path=str(p), sha256=digest, role="declared_producer", exposure="exposed_development")
        for _, digest, p in observed.values()
    )
    receipts: dict[str, tuple[Any, ...]] = {}
    for task in tasks[:-1]:
        if not observed[task["id"]][0]:
            number, slug = task["id"].split("-", 1)
            path = root / f"results/experiment_{number[3:]}_{slug.replace('-', '_')}.json"
            receipts[task["id"]] = (*shared.read(path), path)
    progress(started, "resolved_paths_hashes_roles_operands", 12)
    outcomes, reductions, history = [], [], []
    for task in tasks:
        data, digest, path = observed.get(task["id"], ({}, None, root / task["deliverable"]))
        receipt, receipt_hash, receipt_path = receipts.get(task["id"], ({}, None, path))
        state = (
            "self_administrative" if task == tasks[-1] else str(data.get("verdict_class", "absent"))
        )
        own = []
        if task != tasks[-1] and not data:
            state = (
                "skipped" if receipt.get("blocked_at_layer") == "conductor_pre_gate" else "absent"
            )
            own.append(shared.operand(path, digest, task["id"], "producer_exists", True, False))
        if receipt:
            sources.append(
                dict(
                    path=str(receipt_path),
                    sha256=receipt_hash,
                    role="conductor_gate_receipt",
                    exposure="administrative",
                )
            )
            for gate in receipt.get("gates_evaluated", []):
                if not gate["passed"]:
                    failures.append(
                        shared.operand(
                            Path(gate["artifact_path"]),
                            gate.get("artifact_sha256"),
                            gate["upstream"],
                            gate["artifact_field"],
                            gate["expected"],
                            gate["actual"],
                            gate["op"],
                        )
                    )
        if data:
            expected = dict(
                experiment_id=int(task["id"][3:7]),
                task_id=task["id"],
                milestone="2026.09.687",
                run_date=date,
                MODEL_SPECS=task["MODEL_SPECS"],
                flagged_adversarial=False,
            )
            own.extend(
                shared.operand(path, digest, task["id"], key, val, data.get(key, "missing_field"))
                for key, val in expected.items()
                if data.get(key, "missing_field") != val
            )
            history.extend(data.get("historical_required_failures", []))
            history.append(
                dict(
                    upstream_id=task["id"],
                    path=str(path),
                    sha256=digest,
                    honest_verdict=data.get("honest_verdict"),
                    required_failures=data.get("gate_check_summary", []),
                    resolved=False,
                )
            )
        for gate in task.get("gated_on", []):
            prior, prior_hash, prior_path = observed[gate["upstream"]]
            actual = prior.get(
                gate["artifact_field"], "missing_field" if prior else "missing_source"
            )
            passed = (
                actual == gate["value"]
                if gate["op"] == "=="
                else actual in gate["value"]
                if gate["op"] == "in"
                else False
            )
            if not passed:
                failures.append(
                    shared.operand(
                        prior_path,
                        prior_hash,
                        gate["upstream"],
                        gate["artifact_field"],
                        gate["value"],
                        actual,
                        gate["op"],
                    )
                )
        raw = data.get("rows", [])
        try:
            reduced = reduce_primitives(raw)
        except (ValueError, TypeError, KeyError) as error:
            reduced = reduce_primitives([])
            own.append(
                shared.operand(
                    path, digest, task["id"], "primitive_audit", "valid rows", str(error)
                )
            )
        if own and state in shared.QUALIFIED:
            state = "disqualified"
        failures.extend(own)
        unit = (
            "board_obligation"
            if task["id"] == "exp7926-hardware-evidence"
            else "family_observation"
        )
        reductions.append(
            dict(
                upstream_id=task["id"],
                path=str(path),
                sha256=digest,
                unit=unit,
                primitive_rows=raw,
                **reduced,
            )
        )
        outcomes.append(
            dict(
                upstream_id=task["id"],
                task_id=task["id"],
                path=str(path),
                hash=digest,
                status=state,
                eligible=state in shared.QUALIFIED and not own,
                intended=1,
                started=1,
                completed=1,
                failed=0,
                censored=0,
                excluded=0,
                independent=0,
                arm="administrative_disposition",
                seed=7927,
                conductor_gate_receipt=dict(
                    path=str(receipt_path),
                    sha256=receipt_hash,
                    gates_evaluated=receipt.get("gates_evaluated", []),
                ),
                producer_budget=data.get("sample_size_budget"),
                exposure="exposed_development",
            )
        )
        progress(started, "independent_reduction", len(outcomes))
    for number, coverage in ((7902, "243/377"), (7914, "270/271")):
        path = root / f"results/experiment_{number}_v{685 if number == 7902 else 686}_capstone.json"
        previous, digest = shared.read(path)
        history.append(
            dict(
                upstream_id=f"exp{number}-capstone",
                path=str(path),
                sha256=digest,
                honest_verdict=previous.get("honest_verdict"),
                coverage=coverage,
                resolved=False,
                required_failures=previous.get("gate_check_summary", []),
            )
        )
        sources.append(
            dict(path=str(path), sha256=digest, role="historical_failure", exposure="historical")
        )
    science_ready = (
        authority["activated"] and all(row["eligible"] for row in outcomes[:-1]) and not failures
    )
    memberships = {
        "FR-12": range(7916, 7922),
        "FR-11": (7922, 7923),
        "FR-05/FR-08/FR-09/FR-10/NFR-01": (7915, 7924, 7925, 7926),
    }
    decisions = {}
    for gap, members in memberships.items():
        selected = [outcomes[n - 7915] for n in members]
        states = {row["status"] for row in selected}
        decisions[gap] = dict(
            decision="blocked"
            if states & {"absent", "skipped", "blocked"} or not authority["activated"]
            else "disqualified"
            if any(not row["eligible"] for row in selected)
            else "measured-null",
            required_producers=list(members),
            independent_benefit=False,
            continue_if="Changed inputs and qualified primitive comparisons meet the registered cost, retention and whole-service bounds.",
            retire_if="Retire unchanged scope when the same verdict recurs without changed inputs or qualification.",
        )
    publication = publication or {}
    value = build_current_work_receipt(
        run_id="exp7927-20260930",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"work": "independent primitive reduction"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    value.update(
        experiment_id=7927,
        task_id="exp7927-capstone",
        milestone="2026.09.687",
        run_date=date,
        honest_verdict="complete_null_independent_benefit_unshown"
        if science_ready
        else "complete_blocked_missing_science",
        verdict_class="null" if science_ready else "blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=outcomes,
        outcome_rows=outcomes,
        independent_reduction_rows=reductions,
        sample_size_budget=dict(
            zip(shared.COUNTS, (13, 13, 13, 13, 0, 0, 0, 0), strict=True),
            unit="task_disposition",
            science_unit_budgets={
                r["upstream_id"]: dict(
                    unit=r["unit"],
                    primitive_observations=r["intended"],
                    families=r["independent_families"],
                    source_groups=r["source_groups"],
                    seeds=r["seed_count"],
                    scientifically_independent=0,
                )
                for r in reductions
            },
        ),
        acceptance_gate_results=dict(
            validity=False,
            readiness=0,
            probability_quality=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        random_seed=7927,
        source_artifact_hashes=sources,
        preconditions_checked=failures,
        resolved_imports={
            "carnot.reporting.v687_capstone": str(Path(__file__).resolve()),
            "carnot.reporting.v686_capstone": str(Path(shared.__file__).resolve()),
            "carnot.reporting.v687_contract_methods": str(
                Path(assess.__code__.co_filename).resolve()
            ),
        },
        validation_receipts=[],
        validation_command_manifest_path=None,
        observed_child_commands=[],
        coverage_statement_counts={},
        historical_required_failures=history,
        repository_health=dict(status="historical_backlog_open", affects_required_checks=False),
        verifier_is_oracle=False,
        claim_scope=dict(
            natural_data="exposed_development",
            fixture_agreement="circular_positive",
            gap_oracle_distinct="open_after_20260928_corrigendum",
            independent_benefit=False,
            DiffusionGemma="pending_actual_distinct_oracle_accuracy_and_efficiency_evidence",
        ),
        model_specs=[],
        target_model="none",
        model_invocation_counts=dict(loads=0, calls=0, tokens=0),
        trained_head_specs=[],
        capstone_execution_ready_score=0,
        gap_decisions=decisions,
        retirement_decisions=[
            dict(
                prior=item,
                decision="retire_unchanged_scope"
                if item.get("retire_if_same_verdict")
                and item.get("verdict")
                == (
                    "complete_null_independent_benefit_unshown"
                    if science_ready
                    else "complete_blocked_missing_science"
                )
                else "retain_changed_contract_scope",
                unchanged_prior_verdict=item.get("verdict")
                == (
                    "complete_null_independent_benefit_unshown"
                    if science_ready
                    else "complete_blocked_missing_science"
                ),
            )
            for item in tasks[-1].get("prior_failures", [])
        ],
        authority_snapshots=authority["authority_snapshots"],
        canonical_tasks_sha256=authority["canonical_tasks_sha256"],
        activation_confirmed=authority["activated"],
        publication_gate_results=publication,
        report_path="docs/research-notes/experiment_7927_v687_capstone.md",
        historical_fixture_date="20260929",
        execution_date=date,
        **publication_operands(publication),
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["reproducibility_checksum"] = canonical_hash(
        dict(sources=sources, rows=reductions, seed=7927)
    )
    value["phase_spans"] = [
        dict(
            phase="authority_and_reduction",
            start_s=0,
            end_s=value["duration_s"],
            completed_units=13,
        )
    ]
    value["field_principles"] = {
        key: "Bind actual producer bytes and unit counts; audit completion alone proves no scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        {
            f"acceptance_gate_results.{key}": "Owned validation controls readiness; absent scientific measurements remain null."
            for key in value["acceptance_gate_results"]
        }
    )
    return value


def cold_replay(value: dict[str, Any], root: Path, design: Path, active: Path) -> list[str]:
    """Recompute claims from the same primitive bytes without invoking science."""
    expected = build_candidate(
        root, design, active, "20260930", value.get("publication_gate_results")
    )
    errors = [f"{key}_changed" for key in REPLAY_FIELDS if value.get(key) != expected[key]]
    for source in value["source_artifact_hashes"]:
        path = Path(source["path"])
        if (sha256_file(path) if path.is_file() else None) != source["sha256"]:
            errors.append("source_bytes_changed")
    if any(
        value.get(key) != val
        for key, val in publication_operands(value.get("publication_gate_results", {})).items()
    ):
        errors.append("publication_operands_changed")
    return sorted(set(errors))
