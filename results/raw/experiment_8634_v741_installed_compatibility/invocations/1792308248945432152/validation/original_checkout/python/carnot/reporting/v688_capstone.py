"""REQ-REPORT-7939-V688: reduce current evidence without inventing science.

Exact producer bytes determine custody. Administrative completeness and
scientific benefit remain separate even when external producers cannot run.
"""

from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v687_capstone as prior
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.v688_contract_methods import assess
from scripts.conductor_gates import _eval_op

ROOT = prior.ROOT
GATES = prior.GATES
publication_operands = prior.publication_operands
REPLAY_FIELDS = (*prior.REPLAY_FIELDS, "retirement_decisions")


def progress(started: float, phase: str, units: int) -> None:
    """Flushed boundaries expose progress without padding execution time."""
    print(
        f"[exp7939] phase={phase} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def reduce_primitives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve source clusters and role denominators while reusing audited reducers."""
    roles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    false_accepts: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed_primitive_row")
        item = {
            **row,
            "source_group": row.get(
                "source_group",
                row.get("source_cluster_id", row.get("family_id", row.get("family", "unknown"))),
            ),
        }
        if row.get("status") == "completed" or row.get("completed") is True:
            if row.get("action") in {"accept", "reject", "escalate"} and row.get("label") in (0, 1):
                item["cost"] = {
                    "accept": 5 * row["label"],
                    "reject": 1 - row["label"],
                    "escalate": 0.25,
                }[row["action"]]
                false_accepts[str(row.get("arm", "default"))][
                    str(row.get("family_id", row.get("family", "unknown")))
                ].append(float(row["action"] == "accept" and row["label"] == 1))
            if "started_monotonic_ns" in row and "ended_monotonic_ns" in row:
                item["latency_ms"] = (row["ended_monotonic_ns"] - row["started_monotonic_ns"]) / 1e6
        normalized.append(item)
        roles[str(row.get("role", "declared_producer_unit"))].append(item)
    value = prior.reduce_primitives(normalized)
    value["by_role"] = {role: prior.reduce_primitives(items) for role, items in roles.items()}
    value["unsupported_false_accepts_by_arm"] = {
        arm: sum(sum(v) / len(v) for v in families.values()) / len(families)
        for arm, families in false_accepts.items()
    }
    return value


def build_candidate(
    root: Path,
    design: Path,
    active: Path,
    date: str,
    publication: dict[str, Any] | None = None,
    *,
    snapshots: Path | None = None,
    seal_rows: bool = True,
) -> dict[str, Any]:
    """Read each producer once and keep absent-producer receipts in their actual role."""
    started, start_ns = time.monotonic(), time.monotonic_ns()
    progress(started, "start", 0)
    if date != "20260930":
        raise ValueError("v688_date_changed")
    directory = snapshots or Path(tempfile.mkdtemp(prefix="carnot-7939-authority-"))
    authority = assess(design, root / "research-roadmap-next.yaml", active, directory)
    try:
        tasks = yaml.safe_load(
            Path(authority["authority_snapshots"]["active"]["snapshot_path"]).read_bytes()
        )["tasks"]
        if len(tasks) != 12 or [t["id"].split("-", 1)[0] for t in tasks] != [
            f"exp{n}" for n in range(7928, 7940)
        ]:
            raise ValueError("task_roster_changed")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        tasks = yaml.safe_load(
            gzip.decompress((ROOT / "tests/fixtures/v688/active.yaml.gz").read_bytes())
        )["tasks"]
        authority["activated"] = False
    failures = list(authority["gate_check_summary"])
    sources = [
        dict(path=s["source_path"], sha256=s["sha256"], role=role, exposure="administrative")
        for role, s in authority["authority_snapshots"].items()
    ]
    observed = {
        t["id"]: (*prior.shared.read(root / t["deliverable"]), root / t["deliverable"])
        for t in tasks[:-1]
    }
    sources.extend(
        dict(path=str(p), sha256=d, role="declared_producer", exposure="exposed_development")
        for _, d, p in observed.values()
    )
    progress(started, "resolved_paths_hashes_roles_operands", 11)
    outcomes, reductions, history = [], [], []
    for index, task in enumerate(tasks):
        data, digest, path = observed.get(task["id"], ({}, None, root / task["deliverable"]))
        own = []
        receipt, receipt_hash, receipt_path = {}, None, path
        state = "self_administrative" if index == 11 else str(data.get("verdict_class", "absent"))
        if index != 11 and not data:
            receipt_path = (
                root
                / f"results/experiment_{task['id'][3:7]}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
            )
            receipt, receipt_hash = prior.shared.read(receipt_path)
            state = (
                "skipped" if receipt.get("blocked_at_layer") == "conductor_pre_gate" else "absent"
            )
            own.append(
                prior.shared.operand(path, digest, task["id"], "producer_exists", True, False)
            )
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
                    prior.shared.operand(
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
                milestone="2026.09.688",
                run_date=date,
                MODEL_SPECS=task["MODEL_SPECS"],
                flagged_adversarial=False,
            )
            own.extend(
                prior.shared.operand(
                    path, digest, task["id"], key, val, data.get(key, "missing_field")
                )
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
            if state not in prior.shared.QUALIFIED:
                own.append(
                    prior.shared.operand(
                        path,
                        digest,
                        task["id"],
                        "verdict_class",
                        sorted(prior.shared.QUALIFIED),
                        state,
                        "in",
                    )
                )
        for gate in task.get("gated_on", []):
            previous, previous_hash, previous_path = observed[gate["upstream"]]
            actual = previous.get(
                gate["artifact_field"], "missing_field" if previous else "missing_source"
            )
            if not _eval_op(actual, gate["op"], gate["value"])[0]:
                failures.append(
                    prior.shared.operand(
                        previous_path,
                        previous_hash,
                        gate["upstream"],
                        gate["artifact_field"],
                        gate["value"],
                        actual,
                        gate["op"],
                    )
                )
        raw = data.get("rows", [])
        raw_path = (
            root
            / "results/raw/experiment_7939_v688_capstone/rows"
            / f"{task['id']}-{canonical_hash(raw)[7:]}.json"
        )
        if seal_rows:
            atomic_json(raw_path, dict(producer_sha256=digest, rows=raw))
        sources.append(
            dict(
                path=str(raw_path),
                sha256=sha256_file(raw_path) if raw_path.is_file() else None,
                role="sealed_primitive_rows",
                exposure="exposed_development",
            )
        )
        try:
            reduced = reduce_primitives(raw)
        except (ValueError, TypeError, KeyError) as error:
            reduced = reduce_primitives([])
            own.append(
                prior.shared.operand(
                    path, digest, task["id"], "primitive_audit", "valid rows", str(error)
                )
            )
        if own and state in prior.shared.QUALIFIED:
            state = "disqualified"
        failures.extend(own)
        reductions.append(
            dict(
                upstream_id=task["id"],
                path=str(path),
                sha256=digest,
                unit="board_obligation" if index == 10 else "producer_primitive",
                primitive_rows_sha256=canonical_hash(raw),
                primitive_rows_path=str(raw_path),
                **reduced,
            )
        )
        outcomes.append(
            dict(
                task_id=task["id"],
                upstream_id=task["id"],
                path=str(path),
                hash=digest,
                status=state,
                eligible=int(state in prior.shared.QUALIFIED and not own),
                intended=1,
                started=1,
                completed=1,
                failed=0,
                censored=0,
                excluded=0,
                independent=0,
                arm="administrative_disposition",
                seed=7939,
                producer_budget=data.get("sample_size_budget"),
                exposure="exposed_development",
                conductor_gate_receipt=dict(
                    path=str(receipt_path),
                    sha256=receipt_hash,
                    gates_evaluated=receipt.get("gates_evaluated", []),
                ),
            )
        )
        progress(started, "independent_reduction", index + 1)
    for number, version in ((7902, 685), (7914, 686), (7927, 687)):
        path = root / f"results/experiment_{number}_v{version}_capstone.json"
        previous, digest = prior.shared.read(path)
        history.append(
            dict(
                upstream_id=f"exp{number}-capstone",
                path=str(path),
                sha256=digest,
                honest_verdict=previous.get("honest_verdict"),
                required_failures=previous.get("gate_check_summary", []),
                resolved=False,
            )
        )
        sources.append(
            dict(path=str(path), sha256=digest, role="historical_failure", exposure="historical")
        )
    ready = authority["activated"] and all(r["eligible"] for r in outcomes[:-1]) and not failures
    verdict = (
        "complete_null_independent_benefit_unshown" if ready else "complete_blocked_missing_science"
    )
    decisions = {}
    for gap, indices in {
        "FR-12": range(2, 6),
        "FR-11": (6, 7),
        "FR-05/FR-08/FR-09/FR-10/NFR-01": (0, 1, 8, 9, 10),
    }.items():
        selected = [outcomes[i] for i in indices]
        states = {r["status"] for r in selected}
        decisions[gap] = dict(
            decision="blocked"
            if states & {"absent", "skipped", "blocked", "partial"} or not authority["activated"]
            else "disqualified"
            if any(not r["eligible"] for r in selected)
            else "measured-null",
            required_producers=[7928 + i for i in indices],
            independent_benefit=False,
            continue_if="Changed qualified inputs meet registered cost, retention and whole-service bounds.",
            retire_if="Retire unchanged scope when the same verdict recurs without changed inputs or qualification.",
        )
    value = build_current_work_receipt(
        run_id="exp7939-20260930",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"work": "independent primitive reduction"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    publication = publication or {}
    value.update(
        experiment_id=7939,
        task_id="exp7939-capstone",
        milestone="2026.09.688",
        run_date=date,
        honest_verdict=verdict,
        verdict_class="null" if ready else "blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=outcomes,
        outcome_rows=outcomes,
        independent_reduction_rows=reductions,
        sample_size_budget=dict(
            zip(
                prior.shared.COUNTS,
                (12, sum(r["eligible"] for r in outcomes), 12, 12, 0, 0, 0, 0),
                strict=True,
            ),
            unit="task_disposition",
            science_unit_budgets={
                r["upstream_id"]: dict(
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
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        random_seed=7939,
        source_artifact_hashes=sources,
        preconditions_checked=failures,
        resolved_imports={
            name: str(Path(module.__file__).resolve())
            for name, module in {
                "carnot.reporting.v688_capstone": __import__(__name__, fromlist=["__file__"]),
                "carnot.reporting.v687_capstone": prior,
            }.items()
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
        science_ready=ready,
        gap_decisions=decisions,
        retirement_decisions=[
            dict(
                prior=item,
                unchanged_prior_verdict=item.get("verdict") == verdict,
                decision="retire_unchanged_scope"
                if item.get("retire_if_same_verdict") and item.get("verdict") == verdict
                else "retain_changed_contract_scope",
            )
            for item in tasks[-1].get("prior_failures", [])
        ],
        authority_snapshots=authority["authority_snapshots"],
        canonical_tasks_sha256=authority["canonical_tasks_sha256"],
        activation_confirmed=authority["activated"],
        publication_gate_results=publication,
        primary_resolution_receipt=dict(
            path=str(
                root / "results/raw/experiment_7939_v688_capstone/primary_resolution_receipt.json"
            )
        ),
        terminal_validation_sidecar_path=None,
        report_path="docs/research-notes/experiment_7939_v688_capstone.md",
        historical_fixture_date="20260929",
        execution_date=date,
        **publication_operands(publication),
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["reproducibility_checksum"] = canonical_hash(
        dict(sources=sources, rows=reductions, seed=7939, code_sha256=sha256_file(Path(__file__)))
    )
    value["phase_spans"] = [
        dict(
            phase="authority_and_reduction",
            start_s=0,
            end_s=value["duration_s"],
            completed_units=12,
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
    """Recompute each claim from exact producer bytes and reject changed custody."""
    expected = build_candidate(
        root, design, active, "20260930", value.get("publication_gate_results"), seal_rows=False
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
