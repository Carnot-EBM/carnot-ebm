"""Renew reader custody without erasing failures. Spec: REQ-REPORT-8147.

The stored event frontier supplies identities, not validation authority. Fresh
primitive controls and normal private children qualify the current reader.
This task never loads a model, plays a game or changes the live policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import arc_supervisor_v703_frontier as baseline
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

runner = baseline.runner
MODULE = "python/carnot/reporting/arc_supervisor_v704_renewal.py"
CLI = "scripts/experiments/experiment_8147_v704_arc_reader_renewal.py"
scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    EXPERIMENT_ID=8147,
    PRIOR_ID=8107,
    MILESTONE="2026.10.704",
    RUN_DATE="20261005",
    MODULE=MODULE,
    CLI=CLI,
    OUTPUT=runner.ROOT / "results/experiment_8147_v704_arc_reader_renewal.json",
    TEST="tests/python/test_arc_reader_renewal_8147.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    COVERAGE_TESTS=[],
    CONSUMERS=[
        "tests/python/test_arc_supervisor_qualification_8001.py",
        "tests/python/test_arc_supervisor_delta_7874.py",
        "tests/python/test_primary_publication_7928.py",
    ],
    QUALIFY_BEFORE_SCAN=True,
    FREEZE_SOURCES=True,
    PINNED=dict(baseline.baseline.scope.PINNED),
)
HISTORY = {
    "results/experiment_8120_v702_arc_supervisor_delta.json": "f7e27b2eabab005ec7813a6e13d8c1e369b9584b916af16d61a73ede603dc3b2",
    "results/experiment_8133_v703_arc_reader_frontier.json": "c641b002010dfdd578d9e77e87f203b201a503ebc93646805879e6303302030f",
    "results/raw/experiment_8120_v702_arc_supervisor_delta/validation_command_manifest.json": "2c852456ea418e9998d0a54c660cc12d0ee72d1e1376d40aacbbe3db9921e50a",
    "results/raw/experiment_8133_v703_arc_reader_frontier/validation_command_manifest.json": "eec1a054fcf6afcab0056ceddc29d00934e4f3e7ef0e01574096c21611dfbce9",
}


def inputs() -> dict[str, Any]:
    """Authenticate history bytes while leaving their failed assertions intact."""
    checked = runner.inputs(scope)
    history = []
    for label, digest in HISTORY.items():
        path = scope.ROOT / label
        actual = sha256_file(path) if path.is_file() else "missing"
        checked["checks"].append(
            runner.previous.operand(path, "sha256", "sha256:" + digest, actual, actual)
        )
        if actual == "sha256:" + digest and "raw/" not in label:
            document = json.loads(path.read_text())
            history.append(
                dict(
                    path=str(path),
                    sha256=actual,
                    verdict_class=document["verdict_class"],
                    gate_check_summary=document["gate_check_summary"],
                )
            )
    checked["prior"] = dict(checked["prior"], historical_required_failures=history)
    for row in checked["checks"]:
        row.update(
            check="authenticate_" + row["artifact_field"],
            upstream=row["upstream_id"],
            hash=row["sha256"],
            passed=row["expected"] == row["observed"],
        )
    checked["failures"] = [r for r in checked["checks"] if not r["passed"]]
    checked["additional_source_hashes"] = {
        r["path"]: r["sha256"] for r in checked["checks"] if r["passed"]
    }
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Freeze current checks; the old invalid manifest is never reused."""
    specs = [baseline.baseline.baseline.commands(private)[0], *runner.commands(private, scope)]
    specs = [s for s in specs if not s["name"].startswith("e2e_016")]
    launcher = "import os,runpy,sys; os.chdir(sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    for spec in specs:
        if spec["name"].startswith("cli_"):
            argv = [str(scope.ROOT / a) if a == CLI else a for a in spec["argv"]]
            spec["argv"] = [
                "env",
                "-u",
                "PYTHONPATH",
                str(scope.ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                launcher,
                str(private),
                *argv,
            ]
        if spec["name"] == "affected_pytest":
            spec["deadline_s"] = 180
    return specs


def control_errors(rows: list[dict[str, Any]], private: Path) -> list[str]:
    """Re-read sealed primitive events so copied hashes cannot count as revalidation."""
    errors = []
    if [r["condition"] for r in rows] != ["supported", "missing", "mutation", "zero"]:
        return ["control_conditions"]
    for row, expected in zip(rows, [1, 0, 0, 0], strict=True):
        root = private / row["condition"]
        raw = root / "results/raw/events.json"
        producer = root / "producer.json"
        atomic_json(raw, dict(rows=row["source_events"]))
        atomic_json(
            producer,
            dict(
                run_date=scope.RUN_DATE,
                verdict_class="null",
                source_artifact_hashes={str(raw): sha256_file(raw)},
            ),
        )
        value = baseline.baseline.scan(
            root,
            [producer],
            dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={}),
            root / "scan",
            current_date=scope.RUN_DATE,
        )
        if not (
            row["expected"] == row["observed"] == expected == len(value["new_event_rows"])
            and row["passed"]
        ):
            errors.append("control_" + row["condition"])
    return errors


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path, *, current_date: str
) -> dict[str, Any]:
    """Qualify visible controls and admit only unseen post-frontier identities."""
    controls = baseline.conformance(private / "controls")
    value = baseline.baseline.scan(root, producers, checked, private, current_date=current_date)
    seen = {r["event_id"] for r in checked["inventory"].get("rows", []) if "event_id" in r}
    cutoff = checked["prior"].get("finished_at")
    for row in value["rows"]:
        if row["status"] in {"completed", "censored", "unknown"}:
            if row["event_id"] in seen or cutoff and row["chronology"] != "after_cutoff":
                row.update(status="excluded", reason="previous_frontier_identity_or_clock")
            else:
                seen.add(row["event_id"])
        row.update(
            unit_id=row.get("event_id", row.get("receipt_id")),
            source_cluster_id=row.get("game"),
            exclusion_reason=row.get("reason"),
        )
        source = Path(row.get("source_path", ""))
        row["source_events"] = (
            runner.previous.reader.extract_rows(json.loads(source.read_text()))
            if source.is_file()
            else []
        )
    value.update(runner.previous.reduce(value["rows"]))
    value["reader_conformance_rows"] = controls
    errors = (
        control_errors(controls, private / "independent_controls")
        + runner.previous.replay(checked["inventory"])
        if "rows" in checked["inventory"]
        else control_errors(controls, private / "independent_controls")
    )
    if errors:
        value["scan_failures"].append(
            runner.previous.operand(
                scope.INVENTORY,
                "reader_conformance",
                [],
                errors,
                scope.PINNED[str(scope.INVENTORY)],
            )
        )
    durable = scope.OUTPUT.parent if root == scope.ROOT else root
    transcript = durable / "raw" / scope.OUTPUT.stem / "reader_conformance.json"
    atomic_json(
        transcript,
        dict(
            controls=controls,
            original_assertions=[[r["condition"], r["expected"], r["observed"]] for r in controls],
            preserved_frontier=checked["inventory"],
        ),
    )
    value["scan_source_hashes"][str(transcript)] = sha256_file(transcript)
    atomic_json(private / "primitive_rows.json", value)
    print(
        f"[exp8147] phase=frontier_terminal completed={len(producers)} pending=0 new={len(value['new_event_rows'])}",
        flush=True,
    )
    return value


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Require current normal checks and private replay before reader readiness."""
    fields = baseline.baseline.baseline.artifact_fields(value, checked)
    for row in value.get("validation_receipts", []):
        row.setdefault(
            "normal_exit", row.get("exit_code") is not None and not row.get("timed_out", False)
        )
    receipts = [
        r for r in value.get("validation_receipts", []) if r["classification"] == "required"
    ]
    ready = (
        bool(receipts)
        and all(r["passed"] and r.get("normal_exit", True) for r in receipts)
        and any(r["name"] == "cli_replay" and r["passed"] for r in receipts)
        and value.get("verdict_class") == "null"
        and bool(value.get("reader_conformance_rows"))
        and all(r["passed"] for r in value["reader_conformance_rows"])
    )
    hashes = runner.validation.dependency_hashes(
        scope.ROOT, paths=[*scope.ADDED, scope.TEST, *scope.CONSUMERS]
    )
    shards = {
        p: h
        for p, h in value.get("source_artifact_hashes", {}).items()
        if scope.OUTPUT.stem in p and "/raw/" in p
    }
    count = len(value["new_event_rows"])
    fields.update(
        task_id="exp8147-arc-reader-renewal",
        title="V704 independently renewed supervisor reader",
        supervisor_reader_ready_score=int(ready),
        new_outcome_ready_score=int(ready and count > 0),
        new_solve_claim=False,
        arm_recommendations=[],
        recommended_generalization_change=None,
        candidate_refinement=None,
        refinement_proposal=None,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        call_ledger=[],
        code_config_hashes=hashes,
        raw_shard_hashes=shards,
        required_checks_passed=bool(receipts)
        and all(r["passed"] for r in receipts)
        and value.get("verdict_class") != "disqualified",
        claim_scope="exposed_development_reader_conformance_and_observational_frontier",
        exposure_scope="historical_raw_events_and_circular_positive_private_controls",
        reader_receipt=dict(
            current_code_hashes=hashes,
            original_frontier_sha256=scope.PINNED[str(scope.INVENTORY)],
            conformance_sha256=canonical_hash(value.get("reader_conformance_rows", [])),
            source_event_shards={
                p: h for p, h in shards.items() if p.endswith("reader_conformance.json")
            },
            validation_manifest_sha256=value.get("source_artifact_hashes", {}).get(
                value.get("validation_command_manifest_path", "")
            ),
        ),
        event_frontier=dict(
            path=str(scope.INVENTORY),
            sha256=scope.PINNED[str(scope.INVENTORY)],
            after_finished_at=checked.get("prior", {}).get("finished_at")
            or value.get("event_frontier", {}).get("after_finished_at"),
            preserved_event_ids=sorted(
                r["event_id"]
                for r in checked.get("inventory", {}).get("rows", [])
                if "event_id" in r
            )
            or value.get("event_frontier", {}).get("preserved_event_ids", []),
        ),
        checkpoint_references=[
            dict(path=p, sha256=h)
            for p, h in shards.items()
            if p.endswith("receipt_inventory.json")
        ],
        acceptance_gates=dict(
            reader_qualified=int(ready),
            new_authenticated_outcomes=count,
            priority_change_minimum_resolved=30,
            priority_change_minimum_games=5,
            matched_within_game_and_held_game_required=True,
            causal_benefit_claimed=False,
        ),
        cited_upstream_artifacts=[
            dict(
                path=str(scope.ROOT / p),
                sha256="sha256:" + h,
                role="preserved_failed_history_only",
                fields_imported=["gate_check_summary", "historical_model_provenance"],
            )
            for p, h in HISTORY.items()
        ],
    )
    for key in (
        "intended",
        "eligible",
        "independent",
        "completed",
        "excluded",
        "censored",
        "failed",
    ):
        fields[key + "_count"] = value["sample_size_budget"][key]
    if value.get("verdict_class") == "null":
        fields["honest_verdict"] = (
            "complete_null_no_new_outcomes"
            if not count
            else "complete_null_descriptive_new_outcomes"
        )
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Recompute reductions and reject changed evidence instead of updating hashes."""
    import tempfile

    errors = runner.replay(value)
    if value.get("task_id") != "exp8147-arc-reader-renewal":
        return errors
    expected = artifact_fields(value, {})
    errors.extend(k for k in expected if value.get(k) != expected[k])
    for label, digest in value["source_artifact_hashes"].items():
        path = Path(label)
        path = path if path.is_absolute() else scope.ROOT / path
        if not path.is_file() or sha256_file(path) != digest:
            errors.append("source_sha256:" + label)
    for row in value["checkpoint_references"]:
        path = Path(row["path"])
        if path.is_file() and json.loads(path.read_text()).get("rows") != value["rows"]:
            errors.append("checkpoint_rows")
    for label in value["reader_receipt"]["source_event_shards"]:
        path = Path(label)
        if (
            path.is_file()
            and json.loads(path.read_text()).get("controls") != value["reader_conformance_rows"]
        ):
            errors.append("control_evidence_rows")
    with tempfile.TemporaryDirectory(prefix="carnot-8147-cold-", dir="/tmp") as private:
        errors.extend(control_errors(value["reader_conformance_rows"], Path(private)))
    return errors


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Run unchanged validators and a normal current CLI outside the checkout."""
    report = runner.previous.terminal(candidate, private, durable)
    launcher = "import os,runpy,sys; os.chdir(sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    row = runner.previous.run(
        dict(
            name="cold_primary_replay",
            argv=[
                "env",
                "-u",
                "PYTHONPATH",
                str(scope.ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                launcher,
                str(private),
                str(scope.ROOT / CLI),
                "--cold-replay",
                str(candidate),
            ],
            deadline_s=60,
            expected_exit=0,
            expected_text=None,
            classification="required",
        ),
        private,
        durable,
    )
    report["reports"].append(row)
    report["passed"] = (
        report["passed"] and row["passed"] and not replay(json.loads(candidate.read_text()))
    )
    return report


def execute(output: Path, private: Path) -> int:
    """Let the qualified runner freeze, validate and atomically publish evidence."""
    return int(runner.execute(output, private, scope))


scope.__dict__.update(
    inputs=inputs,
    commands=commands,
    scan=scan,
    artifact_fields=artifact_fields,
    replay=replay,
    terminal=terminal,
    execute=execute,
)
