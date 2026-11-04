"""REQ-REPORT-8763: admit only real new outcomes beyond the qualified frontier."""

from datetime import datetime
from pathlib import Path
from typing import Any

from carnot.reporting import arc_authoritative_frontier_8215 as locator
from carnot.reporting import arc_supervisor_frontier_8720 as original
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v746_retained_evidence import load, reference

Json = dict[str, Any]
ROOT = original.ROOT
FRONTIER = "results/experiment_8760_v750_arc_horizon_sensitivity.json"
FRONTIER_PIN = "sha256:bae0676342b3f95be81e168a7dd56448d436bbb62e96aaf29390b6ba3e3c5903"


def admit(checked: Json, frontier: Json) -> list[Json]:
    """Selection and application must have an observed outcome after the frozen cutoff."""
    return (
        []
        if checked["failures"]
        else [
            row
            for row in checked["rows"]
            if row["receipt_id"] not in frontier["ledger_receipt_identities"]
            and row["status"] == "completed"
            and row.get("selected") is True
            and row.get("applied") is True
            and row.get("outcome") in {"no_progress_within_horizon", "level_progress_observed"}
            and datetime.fromisoformat(row["event_timestamp"])
            > datetime.fromisoformat(frontier["closed_at_utc"])
        ]
    )


def freeze(raw: Path) -> Json:
    """One bounded locator read authenticates the delta without reopening any game."""
    raw.mkdir(parents=True, exist_ok=True)
    path = ROOT / FRONTIER
    frontier_ref = dict(path=str(path), sha256=FRONTIER_PIN)
    refs, failures, rows, frontier = [], [], [], {}
    try:
        prior, refs = original.qualified(path, FRONTIER_PIN)
        if prior["arc_sensitivity_ready_score"] != 1:
            raise ValueError("rejected_producer:arc_sensitivity_ready_score")
        frontier = prior["frontier_after"]
        baseline = load(dict(path=str(ROOT / original.BASELINE), sha256=original.LEDGER_PIN))
        if sorted(baseline["entries"]) != frontier["ledger_receipt_identities"]:
            raise ValueError("frontier_baseline_identity")
        authority = locator.discover()
        atomic_json(raw / "arc_locator.json", authority)
        refs += [reference(raw / "arc_locator.json"), reference(ROOT / original.BASELINE)]
        refs += [dict(path=p, sha256=h) for p, h in authority["producer_code_hashes"].items()]
        ledger = load(authority["ledger"])
        refs.append(authority["ledger"])
        for key in ["prior", "inventory", "terminal"]:
            load(authority[key])
            refs.append(authority[key])
        registry = prior["registry_precheck"]
        load(prior["primitive_reference"])
        refs += [prior["primitive_reference"], reference(Path(registry["path"]))]
        if refs[-1]["sha256"] != registry["sha256"]:
            raise ValueError("registry_identity")
        named = {r["path"] for r in authority["sources"]}
        if not {e["source"] for e in ledger["entries"].values()} <= named:
            raise ValueError("locator_source_completeness")
        upstream = dict(prior, ended_at_utc=frontier["closed_at_utc"], censoring_rows=prior["rows"])
        delta, _ = original.delta(baseline, ledger, upstream)
        for row in delta:
            source = row["source_reference"]
            document = load(source)
            native = next(
                r
                for r in original.extract_rows(document)
                if r["game"] == row["game"] and r["seed"] == row["seed"]
            )
            row["receipt_id"] = original.receipt_id_for_row(native)
            row["event_timestamp"] = native["event_timestamp"]
            rows.append(row)
            refs += [source, row["producer_reference"]]
        rows = admit(dict(rows=rows, failures=[]), frontier)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        failures.append(
            dict(
                upstream="exp8760-arc-horizon-sensitivity",
                path=str(path),
                sha256=reference(path)["sha256"] if path.is_file() else None,
                check="qualified_frontier_and_locator",
                field="authenticated_new_outcomes",
                operator="==",
                expected=True,
                observed=str(error) if path.is_file() else "missing_file",
                passed=False,
            )
        )
    if not rows:
        failures.append(
            dict(
                upstream="exp8773-arc-outcome-frontier",
                path=str(ROOT.joinpath(*locator.DEFAULT_LEDGER_PARTS)),
                sha256=locator.reference(ROOT.joinpath(*locator.DEFAULT_LEDGER_PARTS))["sha256"],
                check="new_evidence_pre_gate",
                field="new_selected_applied_outcomes",
                operator=">",
                expected=0,
                observed=0,
                passed=False,
                reason="No authenticated selected/applied arm outcomes beyond Exp8760 frozen frontier.",
            )
        )
    return dict(
        ready=bool(rows) and not failures,
        frontier_reference=frontier_ref,
        frontier=frontier,
        new_outcome_refs=rows,
        references=refs,
        gate_check_summary=failures,
        locator_read_count=1,
        live_games_run=0,
    )
