"""Reconcile V679 evidence without converting queue receipts into science.

REQ-REPORT-7822. Each claim is reduced from named bytes and keeps its scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import yaml

from carnot.experiment_7809_v679_contract_methods import compare_contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7820_v679_hardware_evidence import run_child

ROOT = Path(__file__).resolve().parents[2]
DESIGN = Path("docs/research-notes/v679-authority-snapshots/design.md")
ROADMAP = Path("docs/research-notes/v679-authority-snapshots/roadmap.yaml")
OUTPUT = Path("results/experiment_7822_v679_capstone.json")
RAW = Path("results/raw/experiment_7822_v679_capstone")
MANIFEST = RAW / "validation_command_manifest.json"
MODULE = Path("python/carnot/experiment_7822_v679_capstone.py")
CLI = Path("scripts/experiments/experiment_7822_v679_capstone.py")
MANIFEST_SHA = "9ebf8124a882dc92a2648dc887e873743fbf1d6f80940ee6f49249e196b99783"
PRE_GATE = {
    7812: "results/experiment_7812_view_energy_fit.json",
    7815: "results/experiment_7815_qwen_counter_evidence.json",
    7819: "results/experiment_7819_service_cost.json",
}
ELIGIBLE = {"positive", "circular_positive", "null"}
GATES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)
PRINCIPLES = {
    "experiment_id": "Each result has one owner.",
    "milestone": "The milestone identifies the authority.",
    "run_date": "The requested date binds this run.",
    "honest_verdict": "External missing inputs are blocked.",
    "verdict_class": "Claim strength travels with the record.",
    "flagged_adversarial": "Invalid evidence cannot open a gate.",
    "gate_check_summary": "A failed operand differs from a scientific null.",
    "rows": "Comparisons retain their units.",
    "acceptance_gate_results": "Fixture readiness is not scientific benefit.",
    "duration_s": "Duration is measured.",
    "phase_spans": "Phase boundaries use the same clock.",
    "random_seed": "This reduction uses no random sampling.",
    "reproducibility_checksum": "Replay needs identical inputs and code.",
    "sample_size_budget": "Seeds and views do not add source families.",
    "source_artifact_hashes": "Historical bytes cannot fill current gaps.",
    "preconditions_checked": "Cheap failures precede compute.",
    "validation_receipts": "Every required check must pass.",
    "verifier_is_oracle": "Fixture truth is circular.",
    "claim_scope": "Exposed data cannot prove hidden generalization.",
    "inference_substrate": "Floors follow invoked work.",
    "inference_substrate_class": "This task aggregates artifacts.",
    "MODEL_SPECS": "A cited model is not an invoked model.",
    "model_specs": "A cited model is not an invoked model.",
    "model_invocation_counts": "Counts describe actual calls and tokens.",
    "task_dispositions": "Queue completion differs from science completion.",
    "publication_gates": "The stable reader controls publication readiness.",
    "paper_ready": "All fixed publication gates must pass.",
    "unmet_gates": "Do not recount blockers as gates.",
    "mechanism_decisions": "Continuation needs a falsifiable trigger.",
    "next_prerequisites": "Every blocked branch names its missing input.",
    "outcome_note_path": "Interpretation stays with the result.",
    "validation_command_manifest_path": "Validation obligations are frozen first.",
    "validation_command_manifest_sha256": "Frozen argv cannot change unnoticed.",
    "observed_child_commands": "Actual dispatch must match the manifest.",
    "repository_health": "A diagnostic failure remains visible.",
}


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Show measured elapsed time and completed units at each boundary."""
    print(
        f"[exp7822] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def failure(
    number: int,
    path: str,
    digest: str | None,
    field: str,
    expected: Any,
    observed: Any,
    operator: str = "==",
) -> dict[str, Any]:
    """Retain both operands so a later reader can check the blocker."""
    return dict(
        upstream_id=f"Exp{number}",
        artifact_path=path,
        artifact_hash=digest,
        field=field,
        operator=operator,
        expected=expected,
        observed=observed,
    )


def authority(design: bytes, roadmap: bytes) -> dict[str, Any]:
    """Read the visible table, embedded JSON, and YAML independently."""
    try:
        plan = yaml.safe_load(roadmap)
        if not isinstance(plan, dict):
            raise ValueError("roadmap must be an object")
        return compare_contract(design.decode(), plan)
    except (UnicodeError, ValueError, KeyError, TypeError, IndexError) as exc:
        return {"passed": False, "errors": [str(exc)], "rows": []}


def account(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every declared task even when the science file never appeared."""
    if len(tasks) != 14 or [t["id"].split("-", 1)[0] for t in tasks] != [
        f"exp{n}" for n in range(7809, 7823)
    ]:
        raise ValueError("V679 fourteen-task order required")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(tasks):
        number = 7809 + index
        label = task["deliverable"]
        path = root / label
        exists = number != 7822 and path.is_file()
        digest = sha256_file(path) if exists else None
        value: dict[str, Any] = {}
        if exists:
            try:
                loaded = json.loads(path.read_bytes())
                if not isinstance(loaded, dict):
                    raise ValueError("producer must be an object")
                value = loaded
            except (ValueError, UnicodeError):
                failures.append(failure(number, label, digest, "schema", "JSON object", "invalid"))
        receipt_label = PRE_GATE.get(number)
        receipt = root / receipt_label if receipt_label else None
        receipt_hash = sha256_file(receipt) if receipt and receipt.is_file() else None
        state = (
            "planned_output"
            if number == 7822
            else "producer"
            if exists
            else "pre_gate_receipt"
            if receipt_hash
            else "absent"
        )
        eligible = bool(
            exists
            and value.get("experiment_id") in (number, task["id"])
            and value.get("milestone") == "2026.09.679"
            and value.get("run_date") == "20260928"
            and value.get("flagged_adversarial") is False
            and value.get("verdict_class") in ELIGIBLE
            and str(value.get("honest_verdict", "")).startswith("complete_")
        )
        raw_path = value.get("raw_rows_path")
        raw_hash = value.get("raw_rows_sha256")
        row = dict(
            unit_id=task["id"],
            task_id=task["id"],
            experiment_id=number,
            order=index + 1,
            phase=task["phase"],
            arm="task_accounting",
            producer_path=label,
            producer_hash=digest,
            availability=state,
            pre_gate_receipt_path=receipt_label if receipt_hash else None,
            pre_gate_receipt_hash=receipt_hash,
            producer_eligible=eligible,
            verdict_class=value.get("verdict_class"),
            honest_verdict=value.get("honest_verdict"),
            flagged_adversarial=value.get("flagged_adversarial"),
            raw_rows_path=raw_path,
            raw_rows_sha256=raw_hash,
            raw_metrics={k: v for k, v in value.items() if k.endswith("_score")},
            benefit=None,
            censored=state in {"absent", "pre_gate_receipt"},
            exclusions=[]
            if eligible
            else [state if state != "producer" else "ineligible_producer"],
            effective_independent_N=value.get("sample_size_budget", {}).get("independent_n"),
        )
        rows.append(row)
        values[task["id"]] = value
        if number == 7822:
            continue
        sources.append(
            dict(
                upstream_id=f"Exp{number}",
                path=label,
                sha256=digest,
                date=value.get("run_date"),
                imported_fields=[
                    "verdict_class",
                    "honest_verdict",
                    "flagged_adversarial",
                    "gate_check_summary",
                    "raw_rows_path",
                ]
                if exists
                else [],
                eligible=eligible,
                pre_gate_path=receipt_label if receipt_hash else None,
                pre_gate_sha256=receipt_hash,
                role="science_producer",
            )
        )
        if not exists:
            failures.append(
                failure(
                    number,
                    label,
                    None,
                    "producer_path",
                    "existing declared science producer",
                    "missing",
                )
            )
        elif not eligible:
            failures.append(
                failure(
                    number,
                    label,
                    digest,
                    "verdict_class",
                    sorted(ELIGIBLE),
                    value.get("verdict_class"),
                    "in",
                )
            )
        for check in value.get("gate_check_summary", []):
            if isinstance(check, dict):
                item = failure(
                    number,
                    str(check.get("artifact_path", label)),
                    check.get("artifact_hash", digest),
                    str(check.get("field", "upstream_gate")),
                    check.get("expected"),
                    check.get("observed"),
                    str(check.get("operator", "==")),
                )
                item["upstream_id"] = check.get("upstream_id", item["upstream_id"])
                failures.append(item)
    for task in tasks:
        for gate in task.get("gated_on", []):
            upstream = gate["upstream"]
            number = int(upstream.split("-", 1)[0][3:])
            source = rows[number - 7809]
            observed = values[upstream].get(gate["artifact_field"])
            expected = gate["value"]
            passed = (
                observed == expected
                if gate["op"] == "=="
                else observed in expected
                if gate["op"] == "in"
                else False
            )
            if not passed:
                failures.append(
                    failure(
                        number,
                        source["producer_path"],
                        source["producer_hash"],
                        gate["artifact_field"],
                        expected,
                        observed,
                        gate["op"],
                    )
                )
    return rows, sources, failures


def decisions(rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retire only a repeated listed scope and name a testable reopening input."""
    outcome = []
    for row, task in zip(rows, tasks, strict=True):
        same = [
            prior["experiment_id"]
            for prior in task.get("prior_failures", [])
            if prior.get("retire_if_same_verdict") is True
            and prior.get("verdict") == row.get("honest_verdict")
        ]
        if same:
            action = "retire"
            trigger = "A changed mechanism or independent evidence must change this exact verdict."
        elif row["producer_eligible"]:
            action = "continue"
            trigger = "An independent label or live scored row must change the measured outcome."
        else:
            action = "await_named_prerequisite"
            trigger = f"Qualify {row['producer_path']} with required validation and primitive rows."
        outcome.append(
            dict(
                task_id=task["id"],
                decision=action,
                trigger=trigger,
                matched_prior_ids=same,
                evidence_path=row["producer_path"],
                evidence_hash=row["producer_hash"],
            )
        )
    return outcome


def arc_scores(root: Path, value: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute public SDK episode scores as diagnostics when rows exist.

    A flagged producer remains excluded even if its saved score is accurate.
    """
    raw = value.get("raw_rows_path")
    if not isinstance(raw, str) or not raw or not (root / raw).is_file():
        return None
    from carnot.experiment_7818_v679_arc_organic_measurement import score_episode

    rows = json.loads((root / raw).read_bytes())
    mismatches = []
    for row in rows:
        scored = score_episode(row["actions"], row["human_baseline_actions"])
        if any(
            abs(scored[f"score_{mode}"] - row[f"score_{mode}"]) > 1e-9
            for mode in ("charged", "uncharged")
        ):
            mismatches.append(row["episode_id"])
    return dict(
        episode_count=len(rows),
        independent_games=len({row["game"] for row in rows}),
        score_mismatches=mismatches,
        eligible_for_benefit=value.get("flagged_adversarial") is False
        and value.get("verdict_class") in ELIGIBLE,
    )


def cold_replay(value: dict[str, Any], root: Path) -> list[str]:
    """Reopen sealed bytes and independently compare task rows with sources."""
    errors = []
    for source in value.get("source_artifact_hashes", []):
        label = source["path"]
        if label == str(OUTPUT):
            errors.append("self_input")
            continue
        path = root / label
        if (sha256_file(path) if path.is_file() else None) != source.get("sha256"):
            errors.append(label)
    for receipt in value.get("validation_receipts", {}).get("checks", []):
        path = Path(receipt["log_path"])
        if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
            errors.append("log_hash_mismatch")
    if [r.get("experiment_id") for r in value.get("task_dispositions", [])] != list(
        range(7809, 7823)
    ):
        errors.append("task_order")
    if value.get("schema") == "carnot.exp7822.v679.capstone.v1":
        design, roadmap = (root / DESIGN).read_bytes(), (root / ROADMAP).read_bytes()
        if not authority(design, roadmap)["passed"]:
            errors.append("authority_mismatch")
        tasks = yaml.safe_load(roadmap)["tasks"]
        rows, _, _ = account(root, tasks)
        if value["task_dispositions"][:-1] != rows[:-1]:
            errors.append("raw_to_summary_mismatch")
    return sorted(set(errors))


def build(
    root: Path,
    publication: dict[str, Any],
    validation: dict[str, Any],
    spans: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """Reduce qualified branches and preserve every current failed operand."""
    design, roadmap = (root / DESIGN).read_bytes(), (root / ROADMAP).read_bytes()
    contract = authority(design, roadmap)
    tasks = yaml.safe_load(roadmap)["tasks"]
    rows, sources, failures = account(root, tasks)
    for label in (DESIGN, ROADMAP):
        sources.append(
            dict(
                upstream_id="V679 authority",
                path=str(label),
                sha256=sha256_file(root / label),
                date="20260928",
                imported_fields=["contract_bytes"],
                eligible=True,
                role="authority",
            )
        )
    for label, original in (
        ("openspec/change-proposals/research-roadmap-vNEXT.md", design),
        ("research-roadmap.yaml", roadmap),
    ):
        path = root / label
        if path.is_file() and path.read_bytes() != original:
            failures.append(
                failure(
                    7822,
                    label,
                    sha256_file(path),
                    "authority_bytes",
                    canonical_hash(original.hex()),
                    sha256_file(path),
                )
            )
    if not contract["passed"]:
        failures.append(
            failure(
                7822,
                str(DESIGN),
                sha256_file(root / DESIGN),
                "table_json_yaml_match",
                True,
                contract["errors"],
            )
        )
    audit_path = root / tasks[12]["deliverable"]
    audit = json.loads(audit_path.read_bytes()) if audit_path.is_file() else {}
    from carnot.experiment_7821_v679_independent_evidence_audit import cold_replay as audit_replay

    audit_errors = audit_replay(audit_path) if audit else ["audit_missing"]
    if audit_errors or audit.get("independent_evidence_ready_score") != 1:
        failures.append(
            failure(
                7821,
                tasks[12]["deliverable"],
                rows[12]["producer_hash"],
                "independent_evidence_ready_score",
                1,
                audit.get("independent_evidence_ready_score"),
            )
        )
    arc_path = root / tasks[9]["deliverable"]
    arc = json.loads(arc_path.read_bytes()) if arc_path.is_file() else {}
    arc_recomputed = arc_scores(root, arc)
    if arc_recomputed and arc_recomputed["score_mismatches"]:
        failures.append(
            failure(
                7818,
                str(arc.get("raw_rows_path")),
                arc.get("raw_rows_sha256"),
                "sdk_score_recheck",
                [],
                arc_recomputed["score_mismatches"],
            )
        )
    hardware_path = root / tasks[11]["deliverable"]
    hardware = json.loads(hardware_path.read_bytes()) if hardware_path.is_file() else {}
    service_path = root / tasks[10]["deliverable"]
    service = json.loads(service_path.read_bytes()) if service_path.is_file() else {}
    stage = service.get("stage_times_ms") or {}
    host, whole = stage.get("host_stage"), stage.get("whole_service")
    service_eligible = (
        rows[10]["producer_eligible"]
        and isinstance(host, (int, float))
        and isinstance(whole, (int, float))
        and 0 <= host < whole
    )
    bound = 1 / (1 - host / whole) if service_eligible else None
    checks_passed = validation.get("required_checks_passed") is True
    if not checks_passed:
        failures.append(
            failure(
                7822,
                str(RAW / "validation"),
                None,
                "required_checks_passed",
                True,
                validation.get("required_checks_passed"),
            )
        )
    own_valid = contract["passed"] and checks_passed
    verdict = "disqualified" if not own_valid else "blocked" if failures else "null"
    honest = (
        "complete_disqualified_required_validation"
        if not own_valid
        else "complete_blocked_required_v679_evidence"
        if failures
        else "complete_null_v679_scientific_accounting"
    )
    rows[-1].update(
        verdict_class=verdict,
        honest_verdict=honest,
        raw_metrics={"capstone_complete_score": int(own_valid and not failures)},
    )
    outcome = decisions(rows, tasks)
    board_next = [
        dict(board=row.get("board"), prerequisite=row.get("next_missing_prerequisite"))
        for row in hardware.get("board_rows", [])
        if row.get("next_missing_prerequisite")
    ]
    gates = {
        name: publication.get("gates", {}).get(name, {}).get("pass")
        for name in ("G1", "G2", "G3", "G4")
    }
    budget = dict(
        intended=14,
        eligible=sum(row["producer_eligible"] for row in rows),
        started=sum(row["availability"] == "producer" for row in rows),
        completed=sum(row["verdict_class"] is not None for row in rows),
        excluded=sum(
            row["availability"] == "producer" and not row["producer_eligible"] for row in rows
        ),
        censored=sum(row["censored"] for row in rows),
        independent_N=None,
    )
    source_digest = canonical_hash([(item["path"], item["sha256"]) for item in sources])
    artifact: dict[str, Any] = dict(
        schema="carnot.exp7822.v679.capstone.v1",
        experiment_id=7822,
        milestone="2026.09.679",
        run_date="20260928",
        honest_verdict=honest,
        verdict_class=verdict,
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=rows,
        task_dispositions=[dict(row) for row in rows],
        acceptance_gate_results=dict(
            validity=own_valid,
            readiness=int(own_valid and not failures),
            probability_quality=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        duration_s=duration_s,
        phase_spans=spans,
        random_seed=dict(seeds=[], purpose="deterministic aggregation"),
        sample_size_budget=budget,
        source_artifact_hashes=sources,
        preconditions_checked=dict(
            authority_match=contract["passed"],
            authority_errors=contract["errors"],
            authority_paths=[str(DESIGN), str(ROADMAP)],
            declared_paths=[task["deliverable"] for task in tasks],
            input_presence=[
                dict(
                    task_id=row["task_id"],
                    availability=row["availability"],
                    sha256=row["producer_hash"],
                )
                for row in rows
            ],
            failed_operands=failures,
            backend="host_cpu_aggregation",
            resources="CPU and local files; no model or board load",
            required_resources=dict(cpu=True, model=False, board=False),
            external_inputs_present=all(row["availability"] == "producer" for row in rows[:-1]),
        ),
        validation_receipts=validation,
        verifier_is_oracle=False,
        claim_scope=dict(
            source="640 exposed development families; no hidden generalization",
            fixture="circular_positive_only",
            qwen="current science producer absent",
            ARC="public SDK score recheck is diagnostic while producer is flagged",
            hardware="no complete measured service or device bound",
            hidden_game="unmeasured",
            publication="stable_G1_to_G4_only",
            oracle_distinct_gap="reopened; historical corrigendum 2026-09-28",
        ),
        field_principles=PRINCIPLES,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="aggregation",
        planned_inference_substrate_class="aggregation",
        actual_inference_substrate_class="aggregation",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(
            loads=0,
            generations=0,
            forwards=0,
            input_tokens=0,
            output_tokens=0,
            failures=0,
            cancellations=0,
            loaded_files=[],
        ),
        mechanism_decisions=outcome,
        next_prerequisites=[
            item["trigger"] for item in outcome if item["decision"] == "await_named_prerequisite"
        ]
        + board_next,
        branch_outcomes=dict(
            source_learning=dict(
                state="blocked",
                audit_dispositions=audit.get("branch_dispositions", []),
                aligned_label_count=audit.get("aligned_label_count", 0),
            ),
            qwen=dict(state="blocked", producer=tasks[6]["deliverable"]),
            ARC=dict(state="disqualified", sdk_recheck=arc_recomputed),
            service=dict(state="blocked", producer=tasks[10]["deliverable"]),
            hardware=dict(state="blocked", board_rows=hardware.get("board_rows", [])),
        ),
        audit_raw_recheck_errors=audit_errors,
        arc_sdk_score_recomputed=arc_recomputed,
        hardware_complete_service_bound=bound,
        outcome_note_path="docs/research-notes/v679-outcomes.md",
        publication_gates=gates,
        paper_ready=publication.get("paper_ready"),
        unmet_gates=publication.get("unmet_gates"),
        publication_gate_results=publication,
        validation_command_manifest_path=str(MANIFEST),
        validation_command_manifest_sha256=sha256_file(root / MANIFEST),
        observed_child_commands=validation.get("checks", []),
        repository_health=validation.get("repository_health"),
        capstone_complete_score=int(own_valid and not failures),
    )
    artifact["reproducibility_checksum"] = canonical_hash(
        dict(
            code=sha256_file(root / MODULE),
            cli=sha256_file(root / CLI),
            inputs=source_digest,
            manifest=artifact["validation_command_manifest_sha256"],
            roles="V679 capstone",
            seeds=[],
        )
    )
    if not own_valid:
        artifact["acceptance_gate_results"] = {gate: 0 for gate in GATES}
    return artifact


def load_manifest(path: Path) -> dict[str, Any]:
    """Reject an added child or changed argv before launching any subprocess."""
    if sha256_file(path) != f"sha256:{MANIFEST_SHA}":
        raise ValueError("frozen validation manifest mismatch")
    value = json.loads(path.read_bytes())
    if (
        value.get("schema") != "carnot.exp7822.validation.v1"
        or len(value.get("commands", [])) != 31
    ):
        raise ValueError("validation manifest schema or command count")
    return value


def dispatch(
    root: Path,
    manifest: dict[str, Any],
    executor: Callable[[Path, dict[str, Any], Path], dict[str, Any]],
    before_child: Callable[[int, list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the exact frozen sequence and compare every returned child identity."""
    receipts: list[dict[str, Any]] = []
    for index, command in enumerate(manifest["commands"]):
        if before_child:
            before_child(index, receipts)
        receipt = executor(root, command, root / RAW / "validation")
        if (receipt.get("name"), receipt.get("command_argv"), receipt.get("classification")) != (
            command["name"],
            command["argv"],
            command["classification"],
        ):
            raise ValueError("undeclared child receipt")
        receipts.append(receipt)
    return receipts


def _span(name: str, started: float, ended: float, units: int) -> dict[str, Any]:
    """Preserve actual phase time without padding a fast run."""
    return dict(
        phase=name,
        start_s=started,
        end_s=ended,
        duration_s=ended - started,
        completed_units=units,
        run_date="20260928",
    )


def main(
    argv: list[str] | None = None,
    executor: Callable[[Path, dict[str, Any], Path], dict[str, Any]] = run_child,
) -> int:
    """Expose preparation, frozen dispatch, cold replay, and the real run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260928":
        raise ValueError("V679 run date must be 20260928")
    root = args.root.resolve()
    manifest_path = args.manifest or root / MANIFEST
    started = time.monotonic()
    progress(started, "entrypoint", "start", 0)
    manifest = load_manifest(manifest_path)
    if args.cold_replay:
        errors = cold_replay(json.loads(args.cold_replay.read_bytes()), root)
        print(json.dumps({"errors": errors}), flush=True)
        return int(bool(errors))
    if args.dispatch_only:
        receipts = dispatch(root, manifest, executor)
        print(json.dumps({"commands": len(receipts)}), flush=True)
        return 0
    progress(started, "preconditions", "before_authority", 0)
    phase = time.monotonic()
    design, roadmap = (root / DESIGN).read_bytes(), (root / ROADMAP).read_bytes()
    contract = authority(design, roadmap)
    if not contract["passed"]:
        raise ValueError(f"V679 authority mismatch: {contract['errors']}")
    if (root / "research-roadmap.yaml").is_file() and (
        root / "research-roadmap.yaml"
    ).read_bytes() != roadmap:
        raise ValueError("active V679 roadmap differs from immutable authority")
    if (root / "openspec/change-proposals/research-roadmap-vNEXT.md").is_file() and (
        root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    ).read_bytes() != design:
        raise ValueError("active V679 design differs from immutable authority")
    tasks = yaml.safe_load(roadmap)["tasks"]
    rows, _, failures = account(root, tasks)
    preflight = dict(
        authority_match=True,
        design_sha256=sha256_file(root / DESIGN),
        roadmap_sha256=sha256_file(root / ROADMAP),
        manifest_sha256=sha256_file(manifest_path),
        input_presence=[
            dict(
                task_id=row["task_id"],
                availability=row["availability"],
                sha256=row["producer_hash"],
            )
            for row in rows
        ],
        failed_operands=failures,
        backend="host_cpu_aggregation",
        required_resources=dict(cpu=True, model=False, board=False),
    )
    print(json.dumps({"preconditions_checked": preflight}, sort_keys=True), flush=True)
    spans = [_span("preconditions", phase, time.monotonic(), len(rows))]
    progress(started, "preconditions", "complete", len(rows))
    from scripts.publication_gate import evaluate

    if args.prepare:
        value = build(
            root, evaluate(), {"required_checks_passed": True}, spans, time.monotonic() - started
        )
        atomic_json(args.prepare, value)
        progress(started, "prepare", "complete", len(value["rows"]))
        return 0
    coverage_parent = Path(manifest["commands"][18]["argv"][2].split("=", 1)[1]).parent
    coverage_parent.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(manifest["candidate_path"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    progress(started, "validation", "before_subprocesses", 0)
    phase = time.monotonic()
    candidate: dict[str, Any] | None = None
    publication: dict[str, Any] | None = None

    def before_child(index: int, receipts: list[dict[str, Any]]) -> None:
        nonlocal candidate, publication
        progress(started, "validation", "command_boundary", index)
        if index == 17:
            publication = json.loads(Path(receipts[-1]["log_path"]).read_bytes())
            candidate = build(
                root,
                publication,
                {"required_checks_passed": True, "checks": receipts},
                spans,
                time.monotonic() - started,
            )
        if candidate is not None:
            candidate["validation_receipts"]["checks"] = list(receipts)
            candidate["observed_child_commands"] = list(receipts)
            atomic_json(candidate_path, candidate)

    receipts = dispatch(root, manifest, executor, before_child)
    spans.append(_span("validation", phase, time.monotonic(), len(receipts)))
    progress(started, "validation", "after_subprocesses", len(receipts))
    assert publication is not None
    required = [r for r in receipts if r["classification"] == "required"]
    passed = all(r.get("passed") is True and r.get("exit_code") == 0 for r in required)
    health = next(r for r in receipts if r["name"] == "repository_health")
    repository_health = dict(
        status="passed" if health["passed"] else "failed_diagnostic",
        exit_code=health["exit_code"],
        timed_out=health.get("timed_out", False),
        log_path=health["log_path"],
        log_sha256=health["log_sha256"],
    )
    validation = dict(
        checks=receipts,
        required_checks_passed=passed,
        frozen_affected_scope=manifest["affected_tests"],
        coverage="100% required by coverage_report",
        task_e2e=next(r for r in receipts if r["name"] == "task_e2e"),
        cold_replay=next(r for r in receipts if r["name"] == "cold_replay"),
        terminal_readers=[
            r for r in receipts if r["name"] in {"adversarial_verify", "strict_row_lint"}
        ],
        repository_health=repository_health,
    )
    progress(started, "terminal", "before_reduction", 0)
    phase = time.monotonic()
    result = build(root, publication, validation, spans, time.monotonic() - started)
    result["flagged_adversarial"] = not next(
        r for r in receipts if r["name"] == "adversarial_verify"
    )["passed"]
    if not passed:
        result["acceptance_gate_results"] = {gate: 0 for gate in GATES}
    spans.append(_span("terminal", phase, time.monotonic(), 1))
    result["phase_spans"] = spans
    result["duration_s"] = time.monotonic() - started
    atomic_json(root / OUTPUT, result)
    progress(started, "terminal", "complete", 1)
    print(
        json.dumps(
            {"honest_verdict": result["honest_verdict"], "verdict_class": result["verdict_class"]}
        ),
        flush=True,
    )
    return int(result["verdict_class"] == "disqualified")
