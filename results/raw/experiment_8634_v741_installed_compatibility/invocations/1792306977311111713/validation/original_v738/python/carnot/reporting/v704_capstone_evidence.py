"""Bind V704 authority and reduce saved evidence without model execution.

REQ-REPORT-8149: absence and administrative completion cannot become science.
Each snapshot preserves original bytes; the current result owns only reductions.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from carnot.reporting import v704_contract_custody as custody
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v699_capstone_reduction import holm
from carnot.reporting.v703_capstone_inputs import deduplicate, failure, reference

Json = dict[str, Any]
ROOT = custody.ROOT
INPUT = "results/experiment_8136_v704_contract_custody.json"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Emit real completed and pending counts at every owned phase boundary."""
    print(f"[exp8149] phase={phase} completed={completed} pending={pending}", flush=True)


def primitive(value: Json, number: int) -> Json:
    """Reopen hash-bound primitives and use the same qualified independent equations."""
    if number == 8144 and "audit_statistics" in value:
        from carnot.verify.learning_audit_8144 import statistics

        ref = next(
            r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "primitive_rows.json"
        )
        result = statistics(json.loads(checked(ref).read_bytes())["rows"])
        if result != value["audit_statistics"]:
            raise ValueError("H2_primitive_reduction_drift")
    elif number in (8145, 8146) and value.get("raw_shard_hashes"):
        from carnot import experiment_8145_v704_natural_service_cost as natural
        from carnot import experiment_8146_v704_live_service_cost as live

        ref = next(
            r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "primitive_rows.json"
        )
        work = json.loads(checked(ref).read_bytes())
        result = natural.reduce_rows(work) if number == 8145 else live.reduce_rows(work)
        expected = (
            value["reduction"]
            if number == 8145
            else json.loads((checked(ref).parent / "independent_reduction.json").read_bytes())
        )
        if result != expected:
            raise ValueError("service_primitive_reduction_drift")
    elif number == 8148 and "replay_input_reference" in value:
        from carnot.reporting.hardware_workload_8148 import reduce

        ref = value["replay_input_reference"]
        fresh = reduce(json.loads(checked(ref).read_bytes()))
        result = {k: fresh[k] for k in ("board_rows", "amdahl_bounds", "quantization_rows")}
        if any(value[k] != v for k, v in result.items()):
            raise ValueError("hardware_primitive_reduction_drift")
    else:
        return dict(available=False)
    return dict(available=True, reference=ref, result=result)


def load(root: Path, raw: Path) -> Json:
    """Reconstruct dispositions from authentic primaries, skips and immutable authority."""
    fallback = json.loads((ROOT / INPUT).read_bytes())
    tasks = fallback["task_contract"]
    refs, issues, primaries, rows, audits = [], [], {}, [], {}
    receipt = json.loads((root / INPUT).read_bytes()) if (root / INPUT).is_file() else {}
    authority: Json = dict(activated=False)
    try:
        snaps = receipt["authority_snapshots"]
        active, design = [
            checked(dict(path=snaps[k]["snapshot_path"], sha256=snaps[k]["sha256"]))
            for k in ("active", "design")
        ]
        authority = custody.assess(
            design, root / "research-roadmap-next.yaml", active, raw / "authority"
        )
        table, tasks = parse_design(design.read_text(), milestone="2026.10.704")
        authority.update(visible_table=table, embedded_tasks=tasks)
        if tasks_digest(tasks) != receipt["canonical_tasks_sha256"]:
            raise ValueError("receipt_tasks_digest_drift")
        live = root / "research-roadmap.yaml"
        if live.is_file() and yaml.safe_load(live.read_bytes())["tasks"] != tasks:
            raise ValueError("live_tasks_digest_drift")
        refs.extend([reference(active), reference(design), reference(live)])
        issues.extend(authority["gate_check_summary"])
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        authority["activated"] = False
        issues.append(failure(root / INPUT, "V704_authority", "authority", True, str(error)))
    manifest = root / "ops/exclusion_manifest.yaml"
    retired = yaml.safe_load(manifest.read_bytes()) if manifest.is_file() else {}
    retired_ids = {int(r.get("experiment_id", -1)) for r in retired.get("retired", [])}
    log_path = root / "ops/conductor-log.md"
    log = log_path.read_text() if log_path.is_file() else ""
    refs.extend([reference(root / INPUT), reference(manifest), reference(log_path)])
    for index, task in enumerate(tasks[:-1]):
        progress("before_task", index, 13 - index)
        path = root / task["deliverable"]
        alternatives = sorted((root / "results").glob(f"experiment_{8136 + index}_*.json"))
        selected = path if path.is_file() or not alternatives else alternatives[0]
        value = json.loads(selected.read_bytes()) if selected.is_file() else {}
        conductor = value.get("schema") == "blocked_gate_check_v1"
        gates = []
        if not path.is_file():
            gates.append(failure(path, task["id"], "primary_exists", True, False))
        qualified = bool(
            value.get("required_checks_passed") is True
            and value.get("flagged_adversarial") is False
            and value.get("task_id") == task["id"]
            and 8136 + index not in retired_ids
        )
        if value and not conductor:
            for field, expected in [
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
                ("task_id", task["id"]),
                ("not_retired", True),
            ]:
                observed = (
                    8136 + index not in retired_ids
                    if field == "not_retired"
                    else value.get(field, "<missing>")
                )
                if observed != expected:
                    gates.append(failure(selected, task["id"], field, expected, observed))
            try:
                terminal = Path(value["terminal_validation_sidecar_path"])
                pub = json.loads(terminal.read_bytes())
                pub = pub.get("publication", pub)
                report = read_bound_sidecar(selected, Path(pub["sidecar_path"]))
                if report["report"]["passed"] is not True:
                    raise ValueError("terminal_report_failed")
                refs.extend([reference(terminal), reference(Path(pub["sidecar_path"]))])
            except (OSError, ValueError, KeyError, TypeError) as error:
                qualified = False
                gates.append(failure(selected, task["id"], "terminal_validation", True, str(error)))
        for gate in task["gated_on"]:
            parent = next(t for t in tasks if t["id"] == gate["upstream"])
            observed = primaries.get(parent["id"], {}).get(gate["artifact_field"], "<missing>")
            if observed != gate["value"]:
                gates.append(
                    failure(
                        root / parent["deliverable"],
                        parent["id"],
                        gate["artifact_field"],
                        gate["value"],
                        observed,
                    )
                )
        summary = value.get("gate_check_summary", [])
        if isinstance(summary, list):
            gates.extend(g for g in summary if not g.get("passed"))
        if conductor:
            gates.append(
                dict(
                    check=value["failed_field"],
                    upstream=value["failed_upstream"],
                    path=value["failed_evidence_path"],
                    hash=value["failed_evidence_sha256"],
                    artifact_field=value["failed_field"],
                    op=value["failed_operator"],
                    expected=value["failed_expected"],
                    observed=value["failed_observed"],
                    passed=False,
                )
            )
        audit = dict(available=False)
        if qualified:
            try:
                audit = primitive(value, 8136 + index)
                if audit["available"]:
                    refs.append(audit["reference"])
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                qualified = False
                gates.append(failure(selected, task["id"], "primitive_reduction", True, str(error)))
        audits[task["id"]] = audit
        state = value.get("verdict_class", "blocked")
        if not qualified and state not in {"blocked", "disqualified"}:
            state = "blocked"
        eligible = bool(
            qualified and state not in {"blocked", "disqualified", "partial"} and not gates
        )
        verdict = value.get("honest_verdict", "complete_blocked_primary_exists")
        if conductor:
            verdict = "complete_blocked_" + value["failed_field"]
        if not str(verdict).startswith("complete_"):
            verdict = "complete_blocked_upstream_qualification"
        lines = [s for s in log.splitlines() if task["title"][:48] in s]
        rows.append(
            dict(
                task_id=task["id"],
                unit_id=task["id"],
                source_cluster_id=task["id"],
                arm="task_disposition",
                condition="terminal_accounting",
                metric="eligible_branch",
                numerator=int(eligible),
                denominator=1,
                raw_numerator=int(eligible),
                raw_denominator=1,
                status="completed",
                completed=True,
                failed=state == "disqualified",
                censored=False,
                eligible=eligible,
                excluded=not eligible,
                qualified=qualified,
                primary_present=path.is_file(),
                primary_honest_verdict=value.get("honest_verdict", "<missing>")
                if not conductor
                else "<missing>",
                honest_verdict=verdict,
                verdict_class=state,
                disposition="conductor_skip"
                if conductor
                else "missing_primary"
                if not value
                else "producer",
                conductor_log_rows=lines,
                conductor_statuses=[s.split("|")[3].strip() for s in lines],
                exclusion_reason=None if eligible else state,
                gate_check_summary=deduplicate(gates),
                **reference(selected),
            )
        )
        primaries[task["id"]] = {
            k: v
            for k, v in value.items()
            if k
            not in {
                "rows",
                "validation_receipts",
                "code_config_hashes",
                "source_artifact_hashes",
                "field_principles",
                "gate_check_summary",
            }
        }
        refs.append(reference(selected))
        issues.extend(gates)
        progress("after_task", index + 1, 12 - index)
    historical = {r["task_id"]: r for r in receipt.get("historical_dispositions", [])}
    archive_path = root / "research-complete.yaml"
    archive = (
        yaml.load(archive_path.read_bytes(), Loader=yaml.CSafeLoader)
        if archive_path.is_file()
        else {}
    )
    archived = {}
    for milestone in archive.get("milestones", []):
        for entry in milestone.get("tasks", []):
            archived.setdefault(entry["id"], []).append(entry)
    refs.append(reference(archive_path))
    priors = {}
    for task in tasks:
        for prior in task["prior_failures"]:
            key = prior["experiment_id"]
            number = key.split("-")[0][3:]
            paths = sorted((root / "results").glob(f"experiment_{number}_*.json"))
            path = paths[0] if paths else root / "results" / ("missing_" + number + ".json")
            old = json.loads(path.read_bytes()) if path.is_file() else {}
            history = historical.get(key, {})
            archived_tasks = archived.get(key, [])
            prior_lines = [
                line
                for old_task in archived_tasks
                for line in log.splitlines()
                if old_task["title"][:48] in line
            ]
            priors[key] = dict(
                reference(path),
                honest_verdict=old.get("honest_verdict", "<missing>")
                if old.get("schema") != "blocked_gate_check_v1"
                else "<missing>",
                conductor_statuses=history.get(
                    "conductor_statuses", [line.split("|")[3].strip() for line in prior_lines]
                ),
                conductor_log_rows=history.get("conductor_log_rows", prior_lines),
                archive_results=[entry["result"] for entry in archived_tasks],
                retirement_history=old.get("retirement_decisions", []),
            )
            refs.append(reference(path))
    preserved = []
    for ref in {r["path"]: r for r in refs}.values():
        path = Path(ref["path"])
        saved = dict(ref)
        if path.is_file():
            target = raw / "custody" / (ref["sha256"].split(":")[-1] + path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            saved.update(source_path=str(path), path=str(target))
        preserved.append(saved)
    return dict(
        tasks=tasks,
        authority=authority,
        dispositions=rows,
        primaries=primaries,
        audits=audits,
        failures=deduplicate(issues),
        references=preserved,
        prior_evidence=priors,
    )


def branch(data: Json, index: int) -> Json:
    """One branch may use qualified subclaims even when another branch is blocked."""
    return (
        data["primaries"][data["tasks"][index]["id"]]
        if data["dispositions"][index]["qualified"]
        else {}
    )


def reduce(data: Json) -> Json:
    """REQ-VERIFY-8149: complete accounting and science readiness use separate operands."""
    tasks = data["tasks"]
    h1 = dict(
        status="blocked",
        raw_p_value=1.0,
        support_passed=False,
        safety_passed=False,
        observed_gain=None,
        beneficial_changed_sources=0,
        intended_count=128,
        completed_count=0,
        original_missing_mask=[True] * 128,
    )
    audit = data["audits"].get(tasks[8]["id"], {})
    stats = audit.get("result", {}) if data["dispositions"][8]["eligible"] else {}
    h2 = dict(
        stats,
        status="completed_null"
        if stats and not stats["h2_passed"]
        else "completed_signal"
        if stats
        else "blocked",
        raw_p_value=1.0,
        support_passed=stats.get("support_sufficient", False),
        safety_passed=stats.get("retention_passed", False),
        observed_gain=stats.get("paired_gain_interval", {}).get("mean_gain"),
        beneficial_changed_sources=stats.get("improved_sources", 0),
        intended_count=192,
        completed_count=stats.get("completed_count", 0),
        original_slot_mask=branch(data, 8).get("original_slot_mask", {}),
    )
    if stats:
        gains = np.array(
            [np.nan if r["gain"] is None else r["gain"] for r in stats["per_source_results"]]
        )
        starts = np.random.default_rng(7048144).integers(
            0, len(gains) - 15, size=(10000, int(np.ceil(len(gains) / 16)))
        )
        sample = gains[(starts[:, :, None] + np.arange(16)).reshape(10000, -1)[:, : len(gains)]]
        counts = np.isfinite(sample).sum(1)
        means = np.nansum(sample, axis=1) / np.maximum(counts, 1)
        h2["raw_p_value"] = float(
            (1 + np.sum(means[counts > 0] <= 0.02)) / (1 + np.sum(counts > 0))
        )
    family = holm(h1, h2, (False, bool(stats)))
    failed = [g for g in data["failures"] if not g.get("passed")]
    natural, service, arc, hardware = [branch(data, i) for i in (9, 10, 11, 12)]
    complete = service.get("complete_service_ready_score", 0) == 1
    if not complete:
        failed.append(
            failure(
                Path(data["dispositions"][10]["path"]),
                tasks[10]["id"],
                "complete_service_ready_score",
                1,
                service.get("complete_service_ready_score", "<missing>"),
            )
        )
    blocked = bool(failed) or h1["status"] == "blocked" or h2["status"] == "blocked"
    state = "blocked" if blocked else "null"
    verdict = (
        "complete_"
        + state
        + "_"
        + (str(failed[0]["check"]) if failed else "source_science_missing")
    )
    rows = deepcopy(data["dispositions"])
    rows.append(
        dict(
            task_id=tasks[-1]["id"],
            unit_id=tasks[-1]["id"],
            source_cluster_id="owned_accounting",
            arm="task_disposition",
            condition="terminal_accounting",
            metric="eligible_branch",
            numerator=0,
            denominator=1,
            raw_numerator=0,
            raw_denominator=1,
            eligible=False,
            excluded=True,
            completed=True,
            failed=False,
            censored=False,
            status="completed",
            honest_verdict=verdict,
            verdict_class=state,
            disposition="owned_capstone",
            exclusion_reason="external_science_block",
        )
    )
    retirements = [
        dict(
            prior,
            task_id=task["id"],
            prior_artifact=data["prior_evidence"].get(prior["experiment_id"], {}),
            prior_verdict_matches_artifact=data["prior_evidence"]
            .get(prior["experiment_id"], {})
            .get("honest_verdict")
            == prior["verdict"],
            current_honest_verdict=row["honest_verdict"],
            same_verdict=row["honest_verdict"] == prior["verdict"],
            task_config_sha256=canonical_hash(task),
            retire_exact_configuration=False,
            environmental_block_retires_method_family=False,
            decision="preserve_changed_configuration_or_external_block",
        )
        for task, row in zip(tasks, rows, strict=True)
        for prior in task["prior_failures"]
    ]
    actions = dict(
        source_decisions="Repair Exp8137 private-fixture validation without weakening immutability; qualify fresh fit/reserved capture and audit >=96 sources with >=12 per class.",
        later_learning_retention="Freeze a new matched method with delayed20/admission12 feedback; require >.02 later source cost gain, >=128 sources, >=5 improved sources and safe >=48-source retention.",
        service_deployment="Complete 24 matched full miss pairs with identical outputs and retained transport/startup/recovery costs; require a one-sided95 speed lower bound >=10.",
        arc_reader_frontier="Keep the qualified Exp8147 reader; scan only unseen event IDs after the saved frontier and require30 outcomes across5 games before priority claims.",
        independent_generalization="Freeze untouched roles on a separately collected human-labeled corpus and require independent reproduction before generalization claims.",
    )
    gaps = {
        k: dict(closed=False, requirements=reqs, next_action=actions[k])
        for k, reqs in [
            ("source_decisions", ["FR-06", "FR-12"]),
            ("later_learning_retention", ["FR-11"]),
            ("service_deployment", ["FR-05", "FR-08", "NFR-01"]),
            ("arc_reader_frontier", ["FR-06", "FR-12"]),
            ("independent_generalization", ["FR-06", "FR-11"]),
        ]
    }
    boards = deepcopy(hardware.get("board_rows", []))
    for board in boards:
        key = "hardware_" + board["board"]
        actions[key] = board["next_operator_or_device_change"]
        gaps[key] = dict(
            closed=False,
            requirements=["hardware_" + board["board"]],
            next_action=actions[key],
            falsifiable_evidence=board["next_missing_prerequisite"],
        )
    return dict(
        honest_verdict=verdict,
        verdict_class=state,
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        canonical_tasks_sha256=tasks_digest(tasks),
        authority=data["authority"],
        H1=h1,
        H2=h2,
        primary_hypothesis_results=family,
        multiplicity=dict(family=["H1", "H2"], method="Holm", alpha=0.05, unavailable_family_p=1),
        h1_development_signal_score=0,
        h2_development_signal_score=int(family[1]["positive_claim"]),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=data.get("verifier_is_oracle", False),
        claim_scope="terminal task accounting and exposed development reductions",
        exposure_scope="Previously exposed RAGTruth; oracle protocol controls give no natural credit",
        science_ready_score=0,
        capstone_execution_ready_score=0,
        intended_count=14,
        completed_count=14,
        eligible_count=sum(r["eligible"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        independent_count=0,
        sample_size_budget=dict(
            tasks=14,
            H1=128,
            H2=192,
            retention=64,
            seeds_are_sources=False,
            timing_repetitions_are_sources=False,
        ),
        gate_check_summary=deduplicate(failed),
        independent_reductions=data["audits"],
        retirement_decisions=retirements,
        gap_decisions=gaps,
        next_actions=actions,
        service_evidence_scope=dict(
            natural_host_qualified=natural.get("natural_service_ready_score") == 1,
            natural_update_ready=natural.get("natural_update_cost_ready_score", 0),
            whole_service_qualified=complete,
            nfr01_met=False,
            scope="Natural host measurements and bounded current full miss pairs; repeats are timing units, never semantic sources",
            natural_reduction=data["audits"].get(tasks[9]["id"], {}),
            live_reduction=data["audits"].get(tasks[10]["id"], {}),
        ),
        arc_evidence=dict(
            reader_ready=arc.get("supervisor_reader_ready_score", 0),
            new_outcome_count=arc.get("new_outcome_count", "<missing>"),
            frontier=arc.get("current_frontier"),
            event_frontier=arc.get("event_frontier"),
            solve_credit=0,
            policy_changed=False,
        ),
        board_obligations=boards,
    )
