"""Bind new supervisor evidence to its immutable frontier. REQ-REPORT-8041.

The qualified reader supplies authentication and the shared runner publishes
checked bytes. This adapter freezes the support rule without running a game.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts.experiments import experiment_8028_v695_arc_supervisor_delta as baseline

runner = baseline.runner
MODULE = "python/carnot/reporting/arc_supervisor_v696_delta.py"
CLI = "scripts/experiments/experiment_8041_v696_arc_supervisor_delta.py"
PRIOR = baseline.scope.OUTPUT
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
scope = SimpleNamespace(**vars(baseline.scope))
scope.__dict__.update(
    PRIOR=PRIOR,
    INVENTORY=INVENTORY,
    PINNED={
        str(PRIOR): "sha256:00f1a56348a6a9df4c72e210ce0811b65c0ffed5817d82d3236b2dabcf513d7c",
        str(INVENTORY): "sha256:e0b1865e4e9590e9211c95d7edcf494e5f9d68c9c9e65a1100992511f7e933f6",
        str(runner.REGISTRY): runner.PINNED[str(runner.REGISTRY)],
    },
    EXPERIMENT_ID=8041,
    PRIOR_ID=8028,
    MILESTONE="2026.10.696",
    OUTPUT=runner.ROOT / "results/experiment_8041_v696_arc_supervisor_delta.json",
    MODULE=MODULE,
    CLI=CLI,
    TEST="tests/python/test_arc_supervisor_delta_8041.py",
    ADDED=[MODULE, CLI],
    INCLUDE=",".join("*/" + p for p in (MODULE, CLI)),
    CONSUMERS=[baseline.scope.TEST, *baseline.scope.CONSUMERS],
)


def inputs() -> dict[str, Any]:
    """A matching primary and inventory prevent an old retry becoming evidence."""
    checked = runner.inputs(scope)
    expected = dict(
        experiment_id=8028,
        task_id="exp8028-arc-supervisor-delta",
        run_date="20261002",
        honest_verdict="complete_null_no_new_outcomes",
        arc_delta_ready_score=1,
        no_new_outcomes=True,
        outcome_hashes=checked["inventory"].get("outcome_hashes", "missing"),
    )
    checked["checks"].extend(
        runner.previous.operand(
            scope.PRIOR, k, v, checked["prior"].get(k, "missing"), scope.PINNED[str(scope.PRIOR)]
        )
        for k, v in expected.items()
    )
    for row in checked["checks"]:
        row.update(
            passed=row["expected"] == row["observed"], check="frontier_" + row["artifact_field"]
        )
    checked["failures"] = [r for r in checked["checks"] if not r["passed"]]
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Keep all owned checks and one bounded health diagnostic in private scratch."""
    specs = [s for s in runner.commands(private, scope) if not s["name"].startswith("e2e_016")]
    return [baseline.commands(private)[0], *specs]


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path, *, current_date: str
) -> dict[str, Any]:
    """Missing authentication blocks; immutable clocks can classify new identities."""
    value = baseline.baseline.qualified.scan(
        root, producers, checked, private, current_date=current_date
    )
    auth_fields = {
        "missing_live_wrapper_receipt": "run_receipt",
        "non_live_wrapper": "run_receipt.entrypoint/execution_mode",
        "wrapper_identity_mismatch": "run_receipt.game/seed/invocation_id",
        "missing_or_changed_frames": "run_receipt.frames_sha256/action_frames",
        "noncontiguous_frames": "action_frames[*].action_index",
        "invalid_frame_evidence": "action_frames[*].frame/levels_completed",
        "missing_redirect_application": "run_receipt.applications",
        "frame_outcome_mismatch": "trajectory_supervisor.redirects[*].resolved_by_levelup/actions_to_levelup",
        "unqualified_live_agent_provenance": "live_agent_provenance/trajectory_supervisor.redirects",
    }
    for row in value["rows"]:
        reason = row.get("reason")
        if row["status"] in {"completed", "censored", "unknown"} and row["chronology"] == "unknown":
            episodes = runner.previous.reader.extract_rows(
                json.loads(Path(row["source_path"]).read_text())
            )
            episode = next(e for e in episodes if canonical_hash(e) == row["content_sha256"])
            row["chronology"] = runner.previous.reader.event_order(
                episode, {"event_timestamp": checked["prior"].get("finished_at", "")}, current_date
            )
            row["prospective"] = row["chronology"] == "after_cutoff"
            if row["chronology"] == "unknown":
                reason = "missing_authenticated_event_clock"
                row.update(status="excluded", reason=reason)
        if reason in auth_fields or reason == "missing_authenticated_event_clock":
            value["scan_failures"].append(
                runner.previous.operand(
                    Path(row["source_path"]),
                    auth_fields.get(reason, "event_timestamp/event_sequence"),
                    "authenticated_live_operand",
                    reason,
                    row["source_sha256"],
                )
            )
    for row in value["scan_failures"]:
        row.update(passed=False, check="authenticate_" + row["artifact_field"])
    value.update(runner.previous.reduce(value["rows"]))
    atomic_json(private / "primitive_rows.json", value)
    return value


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Support qualifies a future curated selection test, never causal solve credit."""
    fields = baseline.artifact_fields(value, checked)
    fields["frontier"].update(
        upstream_id="exp8028-arc-supervisor-delta",
        path=str(scope.INVENTORY),
        sha256=scope.PINNED[str(scope.INVENTORY)],
    )
    supported = sorted(
        arm
        for arm, cell in value["arm_outcomes"].items()
        if cell["uncensored"] >= 10 and cell["games"] >= 3
    )
    proposal = (
        dict(
            action="future_controlled_generalization_test",
            existing_arms=supported,
            causal_gain_claimed=False,
            defaults_changed=False,
            selection_callsite="E3AgentPolicy._maybe_supervise_trajectory",
            application_callsite="E3AgentPolicy._apply_trajectory_redirect",
        )
        if supported
        else None
    )
    raw = scope.OUTPUT.parent / "raw" / scope.OUTPUT.stem
    fields.update(
        event_frontier=fields["frontier"],
        candidate_refinement=proposal,
        proposed_generalization_refinement=proposal,
        sample_interpretation="directional_observational" if supported else "descriptive",
        current_game_execution_count=0,
        current_model_invocation_count=0,
        generalized_learning_benefit_score=0,
        checkpoint_references=[
            dict(path=p, sha256=h, role="current_durable_primitive_rows")
            for p, h in value.get("source_artifact_hashes", {}).items()
            if Path(p).is_relative_to(raw) and p.endswith("/receipt_inventory.json")
        ],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8028,
                path=str(p),
                sha256=scope.PINNED[str(p)],
                fields_imported=[
                    "receipt_inventory",
                    "seen_receipt_hashes",
                    "outcome_hashes",
                    "finished_at",
                ],
            )
            for p in (scope.PRIOR, scope.INVENTORY)
        ],
        substrate_declaration=dict(
            inference_substrate="aggregation_from_upstream_artifacts",
            cpu_work="verifier_scoring",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            pretrained_model_calls=0,
        ),
        transfer_to_live_agent=dict(
            entrypoints=["E3AgentPolicy", "make_carnot_agent"],
            pinned_local27B="unsloth/Qwen3.8-27B-GGUF",
            current_model_loaded=False,
            survives=[
                "persistent evidence memory",
                "mechanical stagnation supervision",
                "verifier-driven selection among curated arms",
                "bounded long-horizon search",
            ],
            does_not_transfer=["open-ended model-generated supervisor arms"],
            solve_credit=0,
        ),
    )
    for game, arms in fields["per_game_arm_statistics"].items():
        for arm, cell in arms.items():
            cell["censored"] = sum(
                r["status"] == "censored"
                for r in value["new_event_rows"]
                if str(r["game"]) == game and r["arm"] == arm
            )
    fields.update(
        {
            k + "_count": value["sample_size_budget"][k]
            for k in (
                "intended",
                "eligible",
                "completed",
                "excluded",
                "failed",
                "censored",
                "independent",
            )
        }
    )
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Cold reduction detects invented claims and modified durable checkpoints."""
    expected = artifact_fields(value, {})
    errors = runner.replay(value) + [
        k
        for k in expected
        if k in value
        and value[k] != expected[k]
        and k
        not in {
            "honest_verdict",
            "code_config_hashes",
            "raw_shard_hashes",
            "cited_upstream_artifacts",
        }
    ]
    for row in value.get("checkpoint_references", []):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            errors.append("checkpoint_sha256")
    return errors


def execute(output: Path, private: Path) -> int:
    """The shared runner owns validation and exact-byte atomic publication."""
    status = int(runner.execute(output, private, scope))
    durable = output.parent / "raw" / output.stem
    report = terminal(output, private, durable)
    atomic_json(
        durable / "published_terminal_reports.json",
        dict(primary_path=str(output), primary_sha256=sha256_file(output), report=report),
    )
    assert report["passed"], "published_byte_validation_failed"
    return status


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """A fresh child reduces the same candidate and durable checkpoints."""
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
    report["passed"] = report["passed"] and row["passed"]
    return report


scope.__dict__.update(
    inputs=inputs,
    commands=commands,
    scan=scan,
    artifact_fields=artifact_fields,
    replay=replay,
    execute=execute,
    terminal=terminal,
)
