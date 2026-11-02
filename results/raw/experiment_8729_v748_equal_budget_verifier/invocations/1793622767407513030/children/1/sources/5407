"""REQ-REPORT-8177: preserve input bytes before independent terminal accounting.

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

from carnot.reporting import v706_contract_custody as custody
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v699_capstone_reduction import holm
from carnot.reporting.v703_capstone_inputs import deduplicate, failure, reference

Json = dict[str, Any]
ROOT = custody.ROOT
INPUT = "results/experiment_8164_v706_contract_custody.json"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts because accounting performs no model or benchmark work."""
    print(f"[exp8177] phase={phase} completed={completed} pending={pending}", flush=True)


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
        table, tasks = parse_design(design.read_text(), milestone="2026.10.706")
        authority.update(visible_table=table, embedded_tasks=tasks)
        if tasks_digest(tasks) != receipt["canonical_tasks_sha256"]:
            raise ValueError("receipt_tasks_digest_drift")
        refs.extend([reference(active), reference(design)])
        issues.extend(authority["gate_check_summary"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        authority["activated"] = False
        issues.append(failure(root / INPUT, "V706_authority", "authority", True, str(error)))
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
        number = 8164 + index
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
        if selected != path and not conductor:
            qualified = False
            gates.append(failure(path, task["id"], "primary_exists", True, False))
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
            gates.extend(g for g in summary if isinstance(g, dict) and not g.get("passed"))
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
                    refs.extend(audit["references"])
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
            if (
                number != 8164
                or k
                in (
                    "experiment_id",
                    "task_id",
                    "honest_verdict",
                    "verdict_class",
                    "MODEL_SPECS",
                    "model_invocation_counts",
                    "trained_head_specs",
                )
            )
            and k
            not in (
                "field_principles",
                "validation_receipts",
                "code_config_hashes",
                "source_artifact_hashes",
                "gate_check_summary",
            )
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
    archived: Json = dict(available=False, scope="historical_only")
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
        literature_mapping=receipt.get("literature_mapping", {})
        if authority.get("activated")
        else {},
        prior_scope_ledger=receipt.get("prior_scope_ledger", [])
        if authority.get("activated")
        else [],
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


def primitive(value: Json, number: int) -> Json:
    """Reopen hash-bound scientific rows, keeping absent comparisons unavailable."""
    refs = []
    if number == 8167 and value.get("measurement_reference"):
        from carnot.verify.fit_sentence_capture_8167 import reduce as reducer

        ref = value["measurement_reference"]
        work = json.loads(checked(ref).read_bytes())
        result = reducer(work["result"]["rows"])
        expected = {k: value[k] for k in result}
    elif number == 8172 and value.get("raw_shard_hashes"):
        from carnot.verify.learning_benefit_audit_8172 import exposure, statistics

        ref = next(
            r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "primitive_rows.json"
        )
        result = statistics(json.loads(checked(ref).read_bytes())["rows"])
        expected = value["audit_statistics"]
        if value["paired_gain_interval"] != result["paired_gain_interval"]:
            raise ValueError("H2_primitive_reduction_drift")
        exposed = exposure(value["installation_exposure_summary"]["installation_rows"])
        if exposed != value["installation_exposure_summary"]:
            raise ValueError("exposure_primitive_reduction_drift")
        result = dict(result, installation_exposure_summary=exposed)
        expected = dict(
            expected, installation_exposure_summary=value["installation_exposure_summary"]
        )
    elif number in (8173, 8174) and value.get("raw_shard_hashes"):
        if number == 8173:
            from carnot.reporting.service_validation_8173 import reduce as reducer

            ref = value["raw_shard_hashes"][1]
        else:
            from carnot.verify.complete_request_8174 import reduce as reducer

            ref = value["raw_shard_hashes"][0]
        result = reducer(json.loads(checked(ref).read_bytes()))
        expected = {k: value[k] for k in result}
    elif number == 8176 and value.get("replay_input_reference"):
        from carnot.reporting.hardware_workload_8176 import reduce as reducer

        ref = value["replay_input_reference"]
        rebuilt = reducer(json.loads(checked(ref).read_bytes()))
        result = {k: rebuilt[k] for k in ("board_rows", "amdahl_bounds", "quantization_rows")}
        expected = {k: value[k] for k in result}
    else:
        return dict(available=False)
    refs.append(ref)
    if result != expected:
        raise ValueError("primitive_reduction_drift")
    return dict(available=True, reference=ref, references=refs, result=result)


def reduce(data: Json) -> Json:
    """REQ-VERIFY-8177: accounting cannot close an unmeasured science gap."""
    tasks = data["tasks"]
    rows = deepcopy(data["dispositions"])
    audit = data["audits"].get(tasks[8]["id"], {})
    stats = audit.get("result", {}) if rows[8]["eligible"] else {}
    h1 = dict(
        status="blocked",
        raw_p_value=1.0,
        support_passed=False,
        safety_passed=False,
        observed_gain=None,
        beneficial_changed_sources=0,
        intended_count=128,
        completed_count=0,
        reason="No qualified natural sentence decision audit; transport null is not H1 science",
        logistic_equivalence="Energy and equivalent logistic representations remain required; no EBM advantage measured",
    )
    h2 = dict(
        status="completed_signal"
        if stats.get("h2_passed")
        else "completed_null"
        if stats
        else "blocked",
        raw_p_value=1.0,
        support_passed=stats.get("support_sufficient", False),
        safety_passed=stats.get("retention_passed", False),
        observed_gain=stats.get("paired_gain_interval", {}).get("mean_gain"),
        beneficial_changed_sources=sum(
            r.get("gain", 0) > 0
            for r in stats.get("per_source_results", [])
            if r.get("gain") is not None
        ),
        intended_count=192,
        completed_count=sum(r.get("gain") is not None for r in stats.get("per_source_results", [])),
        learning_execution=rows[7]["eligible"],
        useful_future_exposure=stats.get("installation_exposure_summary"),
        later_benefit=stats.get("h2_passed"),
        retention=stats.get("retention_passed"),
        original_missing_mask=[r.get("gain") is None for r in stats.get("per_source_results", [])],
        statistics=stats,
        scope="exposed development; seeds averaged within original source",
    )
    family = holm(h1, h2, (False, bool(stats)))
    failed = [g for g in data["failures"] if not g.get("passed")]
    check = str(failed[0]["check"]) if failed else "natural_H1_unavailable"
    verdict = "complete_blocked_" + check
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
            completed=True,
            failed=False,
            censored=False,
            eligible=False,
            excluded=True,
            honest_verdict=verdict,
            verdict_class="blocked",
            disposition="owned_capstone",
            exclusion_reason="external_science_block",
        )
    )
    retirements = []
    for task, row in zip(tasks, rows, strict=True):
        for prior in task["prior_failures"]:
            old = data["prior_evidence"].get(prior["experiment_id"], {})
            authentic = old.get("honest_verdict") == prior["verdict"]
            same = authentic and row["honest_verdict"] == prior["verdict"]
            retirements.append(
                dict(
                    prior,
                    task_id=task["id"],
                    prior_artifact=old,
                    prior_verdict_matches_artifact=authentic,
                    same_verdict=same,
                    current_honest_verdict=row["honest_verdict"],
                    retire_exact_configuration=bool(same and prior.get("retire_if_same_verdict")),
                    retire_method_family=False,
                    documented_scope=prior["addressed_by"],
                    decision="retire_only_documented_scope"
                    if same
                    else "preserve_changed_or_untested_scope",
                )
            )
    service = data["audits"].get(tasks[10]["id"], {}).get("result", {})
    hardware = data["audits"].get(tasks[12]["id"], {}).get("result", {})
    actions = dict(
        source_decisions="Freeze fresh independently labeled sources; require >=96 paired sources, >=12 per class, and >.02 safe typed-cost gain against the equivalent logistic control.",
        later_learning_retention="Change the typed-decision exposure mechanism; require >.02 later cost gain on >=128 original sources with >=48 independently labeled retention sources under the frozen masks.",
        service_deployment="Reduce measured acquisition and queue cost, then test independent complete requests with transfer, startup, persistence and recovery; require a lower95 speed ratio >1 and the frozen NFR-01 target.",
        independent_generalization="Freeze untouched independently labeled sources and obtain independent reproduction before granting generalization credit.",
    )
    gaps = {
        k: dict(closed=False, requirements=req, next_action=actions[k])
        for k, req in [
            ("source_decisions", ["FR-06", "FR-12"]),
            ("later_learning_retention", ["FR-11"]),
            ("service_deployment", ["FR-05", "FR-08", "NFR-01"]),
        ]
    }
    return dict(
        honest_verdict=verdict,
        verdict_class="blocked",
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
        claim_scope="terminal accounting and cached exposed-development reductions",
        exposure_scope="exposed development; no independent generalization",
        science_ready_score=0,
        capstone_execution_ready_score=0,
        intended_count=14,
        completed_count=14,
        eligible_count=sum(r["eligible"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        independent_count=0,
        sample_size_budget=dict(tasks=14, H1=128, H2=192, retention=64, seeds_are_sources=False),
        preconditions_checked=True,
        gate_check_summary=deduplicate(failed),
        independent_reductions=data["audits"],
        retirement_decisions=retirements,
        gap_decisions=gaps,
        next_actions=actions,
        board_obligations=hardware.get("board_rows", []),
        literature_mapping=data["literature_mapping"],
        service_evidence_scope=dict(
            historical_host="qualified host-only service; acquisition excluded",
            historical_composition=data["audits"].get(tasks[9]["id"], {}),
            current_independent_requests=service,
            whole_service_qualified=False,
            nfr01_met=False,
            scope="historical host, unmeasured composition and current requests remain separate",
        ),
        arc_evidence=dict(
            reader=data["primaries"].get(tasks[11]["id"], {}), solve_credit=0, policy_changed=False
        ),
    )
