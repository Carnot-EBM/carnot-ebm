"""REQ-REPORT-8163: preserve input bytes before independent terminal accounting.

A scheduled skip cannot supply a scientific result. Each branch retains its own
qualification, so failed learning does not erase qualified source decisions.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from carnot.reporting import v705_contract_custody as custody
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v699_capstone_reduction import holm
from carnot.reporting.v703_capstone_inputs import deduplicate, failure, reference

Json = dict[str, Any]
ROOT = custody.ROOT
INPUT = "results/experiment_8150_v705_contract_custody.json"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts because accounting performs no model or benchmark work."""
    print(f"[exp8163] phase={phase} completed={completed} pending={pending}", flush=True)


def primitive(value: Json, number: int) -> Json:
    """Reopen saved scientific operands instead of trusting reported headlines."""
    if number == 8156 and value.get("measurement_reference"):
        from carnot.verify.decision_audit_8156 import reduce

        ref = value["measurement_reference"]
        work = json.loads(checked(ref).read_bytes())
        result = reduce(work["evidence"])
        if any(value[k] != v for k, v in result.items()):
            raise ValueError("H1_primitive_reduction_drift")
    elif number == 8159 and value.get("primitive_rows"):
        from carnot.verify.durable_batch_8159 import reduce_rows

        ref = value["primitive_rows"]
        result = reduce_rows(json.loads(checked(ref).read_bytes()))
        if value["paired_speed_intervals"] != result["intervals"]:
            raise ValueError("service_primitive_reduction_drift")
    elif number == 8162 and value.get("replay_input_reference"):
        from carnot.reporting.hardware_workload_8162 import reduce

        ref = value["replay_input_reference"]
        result = reduce(json.loads(checked(ref).read_bytes()))
        if any(value[k] != result[k] for k in ("board_rows", "amdahl_bounds", "quantization_rows")):
            raise ValueError("hardware_primitive_reduction_drift")
    else:
        return dict(available=False)
    return dict(available=True, reference=ref, result=result)


def load(root: Path, raw: Path) -> Json:
    """Bind immutable activation and each actual producer or conductor disposition."""
    tasks = json.loads((ROOT / INPUT).read_bytes())["task_contract"]
    refs: list[Json] = []
    issues: list[Json] = []
    authority: Json = dict(activated=False)
    historical: Json = {}
    try:
        receipt = json.loads((root / INPUT).read_bytes())
        historical = {r["task_id"]: r for r in receipt.get("historical_dispositions", [])}
        snaps = receipt["authority_snapshots"]
        active, design = [
            checked(dict(path=snaps[k]["snapshot_path"], sha256=snaps[k]["sha256"]))
            for k in ("active", "design")
        ]
        authority = custody.assess(
            design, root / "research-roadmap-next.yaml", active, raw / "authority"
        )
        table, tasks = parse_design(design.read_text(), milestone="2026.10.705")
        authority.update(visible_table=table, embedded_tasks=tasks)
        if tasks_digest(tasks) != receipt["canonical_tasks_sha256"]:
            raise ValueError("receipt_tasks_digest_drift")
        refs.extend([reference(active), reference(design)])
        issues.extend(authority["gate_check_summary"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        authority["activated"] = False
        issues.append(failure(root / INPUT, "V705_authority", "authority", True, str(error)))
    manifest = root / "ops/exclusion_manifest.yaml"
    retired = yaml.safe_load(manifest.read_bytes()) if manifest.is_file() else {}
    retired_ids = {str(r.get("experiment_id")) for r in retired.get("retired", [])}
    log_path = root / "ops/conductor-log.md"
    log = log_path.read_text() if log_path.is_file() else ""
    refs.extend([reference(root / INPUT), reference(manifest), reference(log_path)])
    primaries: Json = {}
    audits: Json = {}
    rows = []
    for index, task in enumerate(tasks[:-1]):
        progress("before_task", index, 13 - index)
        number = 8150 + index
        path = root / task["deliverable"]
        alternatives = sorted((root / "results").glob(f"experiment_{number}_*.json"))
        selected = path if path.is_file() or not alternatives else alternatives[0]
        value = json.loads(selected.read_bytes()) if selected.is_file() else {}
        conductor = value.get("schema") == "blocked_gate_check_v1"
        gates = []
        qualified = bool(value and not conductor)
        operands = dict(
            primary_exists=path.is_file(),
            required_checks_passed=value.get("required_checks_passed"),
            flagged_adversarial=value.get("flagged_adversarial"),
            task_id=value.get("task_id"),
            not_retired=str(number) not in retired_ids,
        )
        for field, expected in dict(
            primary_exists=True,
            required_checks_passed=True,
            flagged_adversarial=False,
            task_id=task["id"],
            not_retired=True,
        ).items():
            if operands[field] != expected:
                qualified = False
                gates.append(failure(selected, task["id"], field, expected, operands[field]))
        if value and not conductor:
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
        if qualified and value.get("verdict_class") not in ("disqualified", "partial"):
            try:
                audit = primitive(value, number)
                if audit["available"]:
                    refs.append(audit["reference"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                qualified = False
                gates.append(failure(selected, task["id"], "primitive_reduction", True, str(error)))
        state = value.get("verdict_class", "blocked")
        if not qualified and state not in ("blocked", "disqualified"):
            state = "blocked"
        eligible = qualified and not gates and state not in ("blocked", "disqualified", "partial")
        lines = [s for s in log.splitlines() if task["title"][:48] in s]
        statuses = [s.split("|")[3].strip() for s in lines]
        skipped = conductor or "GATE_BLOCK" in statuses and not value
        verdict = value.get(
            "honest_verdict",
            "complete_blocked_" + str(gates[0]["check"] if gates else "primary_exists"),
        )
        if conductor:
            verdict = "complete_blocked_" + value["failed_field"]
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
                primary_present=bool(value) and not conductor,
                primary_honest_verdict=value.get("honest_verdict", "<missing>")
                if not conductor
                else "<missing>",
                honest_verdict=verdict,
                verdict_class=state,
                disposition="conductor_skip"
                if skipped
                else "producer"
                if value
                else "missing_primary",
                conductor_log_rows=lines,
                conductor_statuses=statuses,
                exclusion_reason=None if eligible else state,
                gate_check_summary=deduplicate(gates),
                **reference(selected),
            )
        )
        primaries[task["id"]] = {
            k: v
            for k, v in value.items()
            if k
            not in (
                "field_principles",
                "validation_receipts",
                "code_config_hashes",
                "source_artifact_hashes",
                "raw_shard_hashes",
                "gate_check_summary",
            )
            and (k != "rows" or number == 8156)
        }
        audits[task["id"]] = audit
        refs.append(reference(selected))
        issues.extend(gates)
        progress("after_task", index + 1, 12 - index)
    priors: Json = {}
    for task in tasks:
        for prior in task["prior_failures"]:
            key = prior["experiment_id"]
            number = key.split("-")[0][3:]
            paths = sorted((root / "results").glob(f"experiment_{number}_*.json"))
            path = paths[0] if paths else root / "results" / ("missing_" + number + ".json")
            old = json.loads(path.read_bytes()) if path.is_file() else {}
            priors[key] = dict(
                reference(path),
                honest_verdict=old.get("honest_verdict", "<missing>")
                if old.get("schema") != "blocked_gate_check_v1"
                else "<missing>",
                retired=str(number) in retired_ids,
                conductor_statuses=historical.get(key, {}).get("conductor_statuses", []),
                conductor_log_rows=historical.get(key, {}).get("conductor_log_rows", []),
                earlier_logged_verdicts=historical.get(key, {}).get("earlier_logged_verdicts", []),
            )
            refs.append(reference(path))
    archived: Json = dict(scope="archived_V704_only", available=False)
    old_path = root / "results/experiment_8144_v704_learning_audit.json"
    if old_path.is_file():
        from carnot.reporting.v704_capstone_evidence import primitive as archived_primitive

        old = json.loads(old_path.read_bytes())
        if old.get("required_checks_passed") is True and old.get("flagged_adversarial") is False:
            archived.update(archived_primitive(old, 8144))
            refs.extend([reference(old_path), archived["reference"]])
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
        archived_learning=archived,
    )


def reduce(data: Json) -> Json:
    """REQ-VERIFY-8163: each scientific conclusion keeps its own support and scope."""
    tasks = data["tasks"]
    audit = data["audits"].get(tasks[6]["id"], {})
    stats = audit.get("result", {}) if data["dispositions"][6]["eligible"] else {}
    h1 = dict(
        status="completed_signal"
        if stats.get("h1_development_signal_score")
        else "completed_null"
        if stats
        else "blocked",
        raw_p_value=1.0,
        support_passed=stats.get("support_passed", False),
        safety_passed=stats.get("safety_passed", False),
        observed_gain=stats.get("paired_intervals", {}).get("mean_gain"),
        beneficial_changed_sources=stats.get("improved_sources", 0),
        intended_count=128,
        completed_count=stats.get("eligible_count", 0),
        statistics={k: v for k, v in stats.items() if k != "rows"},
        original_missing_mask=[
            r["status"] != "completed" for r in stats.get("per_source_results", [])
        ],
    )
    if stats:
        import numpy as np
        from carnot.verify.decision_audit_8156 import CONFIG

        gains = np.asarray(
            [r["h1_gain"] for r in stats["per_source_results"] if r["h1_gain"] is not None]
        )
        if len(gains):
            progress("before_family_bootstrap", 0, 10000)
            draws = np.random.default_rng(CONFIG["seed"]).integers(
                0, len(gains), size=(10000, len(gains))
            )
            h1["raw_p_value"] = float((1 + np.sum(gains[draws].mean(axis=1) <= 0.02)) / 10001)
            progress("after_family_bootstrap", 10000, 0)
    h2 = dict(
        status="blocked",
        raw_p_value=1.0,
        support_passed=False,
        safety_passed=False,
        observed_gain=None,
        beneficial_changed_sources=0,
        intended_count=192,
        completed_count=0,
        learning_execution=False,
        useful_future_exposure=None,
        later_benefit=None,
        retention=None,
        reason="current learning producer and audit absent or gate-skipped",
    )
    family = holm(h1, h2, (bool(stats), False))
    failed = [g for g in data["failures"] if not g.get("passed")]
    state = (
        "blocked" if failed or h1["status"] == "blocked" or h2["status"] == "blocked" else "null"
    )
    verdict = (
        "complete_" + state + "_" + str(failed[0]["check"] if failed else "learning_execution")
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
            status="completed",
            eligible=False,
            excluded=True,
            completed=True,
            failed=False,
            censored=False,
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
            retire_exact_configuration=False,
            retire_method_family=False,
            documented_scope="only the named prior configuration; missing input gives no new method evidence",
            decision="preserve_changed_configuration_or_external_block",
        )
        for task, row in zip(tasks, rows, strict=True)
        for prior in task["prior_failures"]
    ]
    hardware = data["audits"].get(tasks[12]["id"], {}).get("result", {})
    boards = deepcopy(hardware.get("board_rows", []))
    actions = dict(
        source_decisions="Freeze an independently collected panel with >=96 paired sources and >=12 per class; test >.02 source cost gain with safe false accepts.",
        later_learning_retention="Repair the exact failed Exp8152 validation receipt; require useful post-admission exposure and >.02 later gain on >=128 sources with >=48 retention sources.",
        service_deployment="Qualify Exp8160 owned validation and measure independent end-to-end requests including startup, acquisition, transport, durability and recovery against the deployment target.",
        arc_reader_frontier="Reuse Exp8161 reader; inspect only event IDs beyond its frontier and require30 new outcomes across5 games before policy priority claims.",
        independent_generalization="Freeze untouched sources on a separately human-labeled corpus and obtain independent reproduction before granting generalization credit.",
    )
    gaps = {
        k: dict(closed=False, requirements=req, next_action=actions[k])
        for k, req in [
            ("source_decisions", ["FR-06", "FR-12"]),
            ("later_learning_retention", ["FR-11"]),
            ("service_deployment", ["FR-05", "FR-08", "NFR-01"]),
            ("arc_reader_frontier", ["FR-06", "FR-12"]),
            ("independent_generalization", ["FR-06", "FR-11"]),
        ]
    }
    for board in boards:
        key = "hardware_" + board["board"]
        actions[key] = board["next_operator_or_device_change"]
        gaps[key] = dict(
            closed=False,
            requirements=[key],
            next_action=actions[key],
            falsifiable_evidence=board["next_missing_prerequisite"],
        )
    host = data["primaries"][tasks[9]["id"]]
    acquisition = data["primaries"][tasks[10]["id"]]
    arc = data["primaries"][tasks[11]["id"]]
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
        archived_learning=data["archived_learning"],
        primary_hypothesis_results=family,
        multiplicity=dict(family=["H1", "H2"], method="Holm", alpha=0.05, unavailable_family_p=1),
        h1_development_signal_score=int(family[0]["positive_claim"]),
        h2_development_signal_score=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=data.get("verifier_is_oracle", False),
        claim_scope="terminal task accounting and exposed development reductions",
        exposure_scope="exposed development; fixture truth and archived evidence remain separate",
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
        preconditions_checked=True,
        gate_check_summary=deduplicate(failed),
        independent_reductions=data["audits"],
        retirement_decisions=retirements,
        gap_decisions=gaps,
        next_actions=actions,
        board_obligations=boards,
        service_evidence_scope=dict(
            host_batch_qualified=data["dispositions"][9]["eligible"]
            and host.get("host_batch_ready_score") == 1,
            acquisition_composition_qualified=data["dispositions"][10]["eligible"]
            and acquisition.get("acquisition_composition_ready_score") == 1,
            whole_service_qualified=False,
            nfr01_met=False,
            scope="host batching and composed acquisition are bounded component evidence",
            host_reduction=data["audits"].get(tasks[9]["id"], {}),
        ),
        arc_evidence=dict(
            reader_ready=arc.get("supervisor_reader_ready_score", 0),
            new_outcome_count=arc.get("new_outcome_count", "<missing>"),
            frontier=arc.get("current_frontier"),
            event_frontier=arc.get("event_frontier"),
            solve_credit=0,
            policy_changed=False,
        ),
    )
