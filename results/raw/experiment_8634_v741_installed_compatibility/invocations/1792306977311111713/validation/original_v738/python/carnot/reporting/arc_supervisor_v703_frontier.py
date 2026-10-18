"""Qualify current behavior without rewriting custody. Spec: REQ-REPORT-8133.

Historical hash failures remain in their original receipts. Private controls
check access to evidence, while only new authenticated events enter science.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import arc_supervisor_v702_delta as baseline
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

runner = baseline.runner
MODULE = "python/carnot/reporting/arc_supervisor_v703_frontier.py"
CLI = "scripts/experiments/experiment_8133_v703_arc_reader_frontier.py"
PRIOR, INVENTORY, SIDECAR = baseline.PRIOR, baseline.INVENTORY, baseline.SIDECAR
HISTORY = baseline.scope.OUTPUT
scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    EXPERIMENT_ID=8133,
    PRIOR_ID=8107,
    MILESTONE="2026.10.703",
    MODULE=MODULE,
    CLI=CLI,
    OUTPUT=runner.ROOT / "results/experiment_8133_v703_arc_reader_frontier.json",
    TEST="tests/python/test_arc_reader_frontier_8133.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    CONSUMERS=[
        baseline.scope.TEST,
        "tests/python/test_arc_supervisor_evidence_8107.py",
        "tests/python/test_arc_supervisor_qualification_8001.py",
        "tests/python/test_arc_supervisor_delta_7874.py",
        "tests/python/test_primary_publication_7928.py",
    ],
    PINNED=dict(
        baseline.scope.PINNED,
        **{str(HISTORY): "sha256:f7e27b2eabab005ec7813a6e13d8c1e369b9584b916af16d61a73ede603dc3b2"},
    ),
)


def inputs() -> dict[str, Any]:
    """Authenticate original events; changed reader code needs new behavior tests."""
    checked = runner.inputs(scope)
    for path in (SIDECAR, HISTORY):
        actual = sha256_file(path) if path.is_file() else "missing"
        checked["checks"].append(
            runner.previous.operand(path, "sha256", scope.PINNED[str(path)], actual, actual)
        )
        document = json.loads(path.read_text()) if actual == scope.PINNED[str(path)] else {}
        if path == HISTORY:
            checked["prior"].setdefault("historical_required_failures", []).append(
                dict(
                    path=str(path),
                    sha256=actual,
                    verdict_class=document.get("verdict_class"),
                    gate_check_summary=document.get("gate_check_summary", []),
                )
            )
        else:
            for key, expected, observed in (
                ("primary_sha256", scope.PINNED[str(PRIOR)], document.get("primary_sha256")),
                ("report.passed", True, document.get("report", {}).get("passed")),
            ):
                checked["checks"].append(
                    runner.previous.operand(path, key, expected, observed, actual)
                )
    for label in (
        *scope.ADDED,
        scope.TEST,
        *scope.CONSUMERS,
        *[".venv/bin/" + n for n in ("python", "pytest", "coverage", "ruff", "mypy")],
    ):
        path = scope.ROOT / label
        checked["checks"].append(
            runner.previous.operand(
                path, "is_file", True, path.is_file(), sha256_file(path) if path.is_file() else None
            )
        )
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
    """Freeze existing private controls with only the new statement denominator."""
    specs = [baseline.commands(private)[0], *runner.commands(private, scope)]
    specs = [s for s in specs if not s["name"].startswith("e2e_016")]
    launcher = "import os, runpy, sys; os.chdir(sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
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
    return specs


def conformance(private: Path) -> list[dict[str, Any]]:
    """Visible frame truth tests the reader; it supplies no natural observations."""
    frames = [dict(action_index=i, levels_completed=int(i == 2), frame=[[i]]) for i in (1, 2)]
    event = dict(
        game="protocol",
        seed=0,
        invocation_id="control",
        receipt_id="control",
        event_timestamp="2026-10-04T00:00:00Z",
        solve_provenance="live_agent_self_discovery",
        live_agent_provenance=dict(
            policy_class="E3AgentPolicy", agent_factory="make_carnot_agent", execution_mode="live"
        ),
        termination=dict(reason="finished"),
        action_frames=frames,
        trajectory_supervisor=dict(
            enabled=True,
            mode="applied",
            redirects=[
                dict(
                    id="r",
                    arm="drop_goal_bias",
                    fired=True,
                    action_index=1,
                    resolved_by_levelup=True,
                    actions_to_levelup=1,
                    pending_at_resolution=1,
                )
            ],
        ),
        run_receipt=dict(
            game="protocol",
            seed=0,
            invocation_id="control",
            execution_mode="live",
            entrypoint="CarnotAgent.choose_action",
            frames_sha256=canonical_hash(frames),
            applications=[dict(redirect_id="r", arm="drop_goal_bias", action_index=1)],
        ),
    )
    rows = []
    for condition, expected in (("supported", 1), ("missing", 0), ("mutation", 0), ("zero", 0)):
        root = private / condition
        episode = json.loads(json.dumps(event))
        if condition == "missing":
            episode.pop("action_frames")
        if condition == "mutation":
            episode["trajectory_supervisor"]["redirects"][0]["actions_to_levelup"] = 99
        if condition == "zero":
            episode["trajectory_supervisor"]["redirects"] = []
        raw = root / "results/raw/control.json"
        producer = root / "results/experiment_control.json"
        atomic_json(raw, dict(rows=[episode]))
        atomic_json(
            producer,
            dict(
                run_date=scope.RUN_DATE,
                verdict_class="null",
                source_artifact_hashes={str(raw): sha256_file(raw)},
            ),
        )
        value = baseline.scan(
            root,
            [producer],
            dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={}),
            root / "scratch",
            current_date=scope.RUN_DATE,
        )
        actual = len(value["new_event_rows"])
        rows.append(
            dict(
                condition=condition,
                expected=expected,
                observed=actual,
                passed=actual == expected,
                claim_scope="circular_positive",
                source_sha256=sha256_file(raw),
                source_events=[episode],
                reduction=value,
            )
        )
        print(
            f"[exp8133] phase=reader_control completed={len(rows)} pending={4 - len(rows)}",
            flush=True,
        )
    atomic_json(private / "conformance.json", rows)
    return rows


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path, *, current_date: str
) -> dict[str, Any]:
    """Admit only fresh identities after current behavior has passed its controls."""
    controls = conformance(private / "controls")
    value = baseline.scan(
        root,
        producers if all(r["passed"] for r in controls) else [],
        checked,
        private,
        current_date=current_date,
    )
    admitted: set[str] = set()
    cutoff = checked["prior"].get("finished_at")
    for row in value["rows"]:
        identity = row.get("event_id")
        if row["status"] in {"completed", "censored", "unknown"}:
            if identity in admitted:
                row.update(status="excluded", reason="duplicate_redirect_identity")
            elif cutoff and row["chronology"] != "after_cutoff":
                row.update(status="excluded", reason="not_after_qualified_frontier")
            else:
                admitted.add(identity)
        row.update(
            unit_id=identity or row.get("receipt_id") or row.get("source_sha256"),
            source_cluster_id=row.get("game") or row.get("producer_sha256"),
            arm=row.get("arm"),
            exclusion_reason=row.get("reason"),
        )
    value.update(runner.previous.reduce(value["rows"]))
    value["reader_conformance_rows"] = controls
    original_errors = (
        runner.previous.replay(checked["inventory"]) if "rows" in checked["inventory"] else []
    )
    if not all(r["passed"] for r in controls) or original_errors:
        value["scan_failures"].append(
            runner.previous.operand(
                INVENTORY,
                "reader_conformance",
                [True, []],
                [all(r["passed"] for r in controls), original_errors],
                scope.PINNED[str(INVENTORY)],
            )
        )
    durable = (scope.OUTPUT.parent if root == scope.ROOT else root) / "raw" / scope.OUTPUT.stem
    transcript = durable / "reader_conformance.json"
    atomic_json(
        transcript,
        dict(
            controls=controls,
            preserved_frontier=checked["inventory"],
            cold_reduction_errors=original_errors,
        ),
    )
    value["scan_source_hashes"][str(transcript)] = sha256_file(transcript)
    value["inventory_counts"] = dict(
        scanned=len(producers),
        raw_shards=len(value["scan_source_hashes"]) - 1,
        new=len(value["new_event_rows"]),
        malformed=sum(r["status"] == "malformed" for r in value["rows"]),
        incomplete=sum(r["status"] in {"unknown", "censored"} for r in value["rows"]),
        failed_operands=len(value["scan_failures"]),
    )
    atomic_json(private / "primitive_rows.json", value)
    print(
        f"[exp8133] phase=frontier_terminal completed={len(producers)} pending=0 new={len(value['new_event_rows'])}",
        flush=True,
    )
    return value


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """A qualified empty reader is ready; only new outcomes support behavior."""
    fields = baseline.artifact_fields(value, checked)
    ready = (
        fields["supervisor_reader_ready_score"] == 1
        and bool(value.get("reader_conformance_rows"))
        and all(r["passed"] for r in value["reader_conformance_rows"])
    )
    current = {
        p: sha256_file(scope.ROOT / p)
        for p in (
            *baseline.baseline.baseline.AUTH_CODE,
            baseline.baseline.MODULE,
            baseline.MODULE,
            *scope.ADDED,
            scope.TEST,
            *scope.CONSUMERS,
        )
    }
    receipt = dict(
        current_code_hashes=current,
        original_frontier_sha256=scope.PINNED[str(INVENTORY)],
        conformance_sha256=canonical_hash(value.get("reader_conformance_rows", [])),
        original_reduction_errors=[],
        source_event_shards={
            p: h
            for p, h in value.get("source_artifact_hashes", {}).items()
            if p.endswith("reader_conformance.json")
        },
    )
    count = len(value["new_event_rows"])
    budget = value["sample_size_budget"]
    fields.update(
        task_id="exp8133-arc-reader-frontier",
        title="V703 current supervisor reader and new event frontier",
        supervisor_reader_ready_score=int(ready),
        new_outcome_ready_score=int(ready and count > 0),
        reader_receipt=receipt,
        reader_conformance_rows=value.get("reader_conformance_rows", []),
        new_solve_claim=False,
        arm_recommendations=[],
        recommended_generalization_change=None,
        candidate_refinement=None,
        refinement_proposal=None,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        claim_scope="exposed_development_reader_conformance_and_observational_frontier",
        exposure_scope="preserved_raw_events_and_private_protocol_controls",
        intended_count=budget["intended"],
        eligible_count=budget["eligible"],
        independent_count=budget["independent"],
        completed_count=budget["completed"],
        excluded_count=budget["excluded"],
        censored_count=budget["censored"],
        failed_count=budget["failed"],
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
                path=str(p),
                sha256=scope.PINNED[str(p)],
                fields_imported=[
                    "receipt_inventory",
                    "outcome_hashes",
                    "gate_check_summary",
                    "historical_model_provenance",
                ],
                role="preserved_history_only",
            )
            for p in (PRIOR, INVENTORY, SIDECAR, HISTORY)
        ],
    )
    fields["event_frontier"].update(
        after_finished_at=checked.get("prior", {}).get("finished_at")
        or value.get("event_frontier", {}).get("after_finished_at"),
        preserved_event_ids=sorted(
            r["event_id"] for r in checked.get("inventory", {}).get("rows", []) if "event_id" in r
        ),
    )
    fields["checkpoint_references"] = [
        dict(path=p, sha256=h)
        for p, h in value.get("source_artifact_hashes", {}).items()
        if p.endswith("/receipt_inventory.json") and Path(p).parent.name == scope.OUTPUT.stem
    ]
    fields["code_config_hashes"] = current
    fields["raw_shard_hashes"] = {
        p: h
        for p, h in value.get("source_artifact_hashes", {}).items()
        if str(scope.OUTPUT.stem) in p and "/raw/" in p
    }
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Cold reductions reject altered current code, invented units and lost rows."""
    errors = runner.replay(value)
    if value.get("task_id") == "exp8133-arc-reader-frontier":
        expected = artifact_fields(value, {})
        errors.extend(
            k
            for k in expected
            if k not in value
            or value[k] != expected[k]
            and k not in {"honest_verdict", "cited_upstream_artifacts"}
        )
    identities = [
        r["event_id"] for r in value["rows"] if r["status"] in {"completed", "censored", "unknown"}
    ]
    if len(identities) != len(set(identities)):
        errors.append("duplicate_event_identity")
    for row in value.get("checkpoint_references", []):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            errors.append("checkpoint_sha256")
        elif json.loads(path.read_text()).get("rows") != value["rows"]:
            errors.append("checkpoint_rows")
    for label, digest in value.get("reader_receipt", {}).get("source_event_shards", {}).items():
        path = Path(label)
        if not path.is_file() or sha256_file(path) != digest:
            errors.append("control_evidence_sha256")
        elif json.loads(path.read_text()).get("controls") != value["reader_conformance_rows"]:
            errors.append("control_evidence_rows")
    return errors


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Run unmodified validators and the real CLI on the same frozen candidate."""
    report = runner.previous.terminal(candidate, private, durable)
    launcher = "import os, runpy, sys; os.chdir(sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
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
    """The shared runner freezes checks and publishes only validated bytes."""
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
