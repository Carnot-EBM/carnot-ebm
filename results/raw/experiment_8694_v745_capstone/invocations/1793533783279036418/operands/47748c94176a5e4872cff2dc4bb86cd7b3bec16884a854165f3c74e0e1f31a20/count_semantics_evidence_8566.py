"""REQ-REPORT-8566: authenticate original bytes and reduce their primitive rows.

The original failed producer stays failed. This adapter only checks saved data;
it never calls the historical training runner or its validation controller.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting import count_semantics_8566 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
UPSTREAM = "experiment_8555_v735_canonical_update_accounting"
PIN = "sha256:e690ae89a0c2219d9ebea8558fca68168c07157190bef61ebe31313d4afee2f2"
HISTORY_RULE = "ascending numeric experiment ID then filename among top-level JSON with stored and current critical TAUTOLOGY; first two IDs"
HISTORY = ["experiment_581_coace_recall_diagnostic_v2.json", "experiment_696_icalm_abstention.json"]


def reference(path: Path) -> Json:
    """Hash exact bytes so copies and replay retain the original producer identity."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def read(ref: Json) -> Any:
    """Authenticate before parsing; missing bytes remain missing rather than zero."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("source_hash")
    return json.loads(path.read_bytes())


def authenticate(root: Path) -> Json:
    """Pin disqualified authority independently of mutable self-reported checksums."""
    value: Json = read(dict(path=str(root / "results" / (UPSTREAM + ".json")), sha256=PIN))
    if (
        value["experiment_id"] != 8555
        or value["verdict_class"] != "disqualified"
        or value["task_id"] != "exp8555-canonical-update-accounting"
    ):
        raise ValueError("historical_identity")
    return value


def measure(root: Path, raw: Path) -> Json:
    """Reduce saved arrays and authenticate each contributing original sealed step."""
    raw.mkdir(parents=True, exist_ok=True)
    result: Json = dict(
        failures=[],
        reproduced_findings=[],
        cardinality_reductions=[],
        owned_controller_failures=[],
        historical_controls=[],
        sources=[],
    )
    try:
        original = root / "results" / (UPSTREAM + ".json")
        value = authenticate(root)
        result["sources"].append(reference(original))
        private = raw / original.name
        shutil.copyfile(original, private)
        events = read(value["event_rows_reference"])
        copied = raw / "original_event_rows.json"
        shutil.copyfile(value["event_rows_reference"]["path"], copied)
        result["sources"].append(value["event_rows_reference"])
        result["private_original"] = reference(private)
        result["events"] = reference(copied)
        steps: Json = {}
        for index, row in enumerate(events):
            ref = row["source_reference"]
            if ref["path"] not in steps:
                steps[ref["path"]] = read(ref)
                result["sources"].append(ref)
            step = steps[ref["path"]]
            trace = row["label_release_record"]
            if (
                trace not in step["updates"]
                or row["admission_record"] not in step["sealed_state"]["pending"]
                or e.digest(row["before"]) != trace["before_head_hash"]
                or e.digest(row["after"]) != trace["after_head_hash"]
            ):
                raise ValueError("sealed_transition_identity")
            if index % 400 == 0:
                print(
                    f"[exp8566] sealed_rows completed={index} pending={len(events) - index}",
                    flush=True,
                )
        totals = e.reduce(events)
        saved = dict(
            value["metrics"],
            intended_count=value["intended_count"],
            completed_count=value["completed_count"],
        )
        if totals != saved or [r["unit_id"] for r in value["rows"]] != [
            r["namespace"] + ":" + r["event_id"] for r in events
        ]:
            raise ValueError("original_counter_roster")
        findings = e.critical(json.loads(private.read_bytes()))
        if findings != value["adversarial_findings"] or len(findings) != 7:
            raise ValueError("seven_original_findings")
        reductions = []
        for flag in findings:
            left, right = [
                part.split("=")[0] for part in flag["detail"].split(" agree")[0].split(" and ")
            ]
            reductions.append(
                dict(
                    left=left,
                    right=right,
                    left_value=totals[left],
                    right_value=totals[right],
                    proven=type(saved[left]) is int
                    and type(saved[right]) is int
                    and totals[left] == totals[right],
                    expectation="valid_discrete_identity",
                    old_expectation="critical",
                    table=reference(copied),
                    proof="unique sealed roster; boolean sums; unequal before/after arrays; assignment-list lengths",
                )
            )
        defects = [r for r in value["gate_check_summary"] if r.get("owned_required_check")]
        defects += [r for r in value["validation_receipts"] if not r["passed"]]
        for defect in defects:
            path = root / defect.get("path", defect.get("stdout_path", ""))
            expected = defect.get("sha256", defect.get("stdout_sha256"))
            if sha256_file(path) != expected:
                raise ValueError("controller_log_hash")
            result["sources"].append(reference(path))
        history = []
        for name in HISTORY:
            path = root / "results" / name
            copy = raw / name
            shutil.copyfile(path, copy)
            flags = e.critical(json.loads(copy.read_bytes()))
            if not flags:
                raise ValueError("historical_true_positive")
            history.append(
                dict(
                    source=reference(path),
                    copy=reference(copy),
                    current_critical_findings=flags,
                    expected="critical",
                    selection_rule=HISTORY_RULE,
                )
            )
        result.update(
            reproduced_findings=findings,
            cardinality_reductions=reductions,
            owned_controller_failures=defects,
            historical_controls=history,
            historical_verdict=value["verdict_class"],
            historical_model_provenance=value["historical_model_provenance"],
            metrics=saved,
            original_node_ids=value["historical_union_receipt"]["terminal_node_outcomes"],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failures"].append(
            dict(
                check="original_evidence",
                upstream="exp8555-canonical-update-accounting",
                path=str(root / "results" / (UPSTREAM + ".json")),
                sha256=PIN,
                field="authenticated_primitives",
                operator="==",
                expected=True,
                observed=None if isinstance(error, FileNotFoundError) else str(error),
                passed=False,
            )
        )
    atomic_json(raw / "reductions.json", result)
    return result
