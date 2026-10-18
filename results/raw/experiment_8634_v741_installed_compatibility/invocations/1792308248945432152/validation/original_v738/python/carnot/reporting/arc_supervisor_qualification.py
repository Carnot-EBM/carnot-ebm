"""Qualify the evidence route before reading outcomes. REQ-REPORT-8001.

Historical failures remain failures. Protocol controls test reader headroom;
only producer-bound live wrapper and frame receipts can supply observations.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import arc_supervisor_v690_delta as runner
from carnot.reporting import arc_supervisor_v692_delta as history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

ROOT = runner.ROOT
PRIOR = history.PRIOR
INVENTORY = history.INVENTORY
HISTORY = history.HISTORY
HISTORY_INVENTORY = history.HISTORY_INVENTORY
RECENT_HISTORY = history.OUTPUT
REGISTRY = runner.REGISTRY
HEALTH = ROOT / "results/raw/experiment_7975_v691_arc_supervisor_delta/prior_owned_attempt.json"
ORIGINAL_SOURCES = (
    ROOT
    / "results/raw/experiment_8001_v693_arc_supervisor_qualification/original_source_snapshots/manifest.json"
)
OUTPUT = ROOT / "results/experiment_8001_v693_arc_supervisor_qualification.json"
PINNED = dict(history.PINNED)
PINNED[str(RECENT_HISTORY)] = (
    "sha256:de4200e7be3dd874d52c633d6a5489380aa2993862158fed419dfaa691a9e8b3"
)
PINNED[str(HEALTH)] = "sha256:49982c258aa403ad2ab843ebc9ddc492f93b4bed3bfeff6ad36fa0cc437ce865"
PINNED[str(ORIGINAL_SOURCES)] = (
    "sha256:e381561afb6aaf3d4068194f20b74fc949ee911e28cc0997a2cdb9b145338d4b"
)
EXPERIMENT_ID = 8001
PRIOR_ID = 7962
MILESTONE = "2026.10.693"
RUN_DATE = "20261002"
MODULE = "python/carnot/reporting/arc_supervisor_qualification.py"
CLI = "scripts/experiments/experiment_8001_v693_arc_supervisor_qualification.py"
TEST = "tests/python/test_arc_supervisor_qualification_8001.py"
ADDED = [MODULE, CLI, *history.ADDED]
INCLUDE = ",".join("*/" + path for path in ADDED)
CONSUMERS = [history.TEST, *history.CONSUMERS]
COVERAGE_TESTS = [history.TEST, *history.COVERAGE_TESTS]
APPLICABLE_E2E = ["E2E-017"]
QUALIFY_BEFORE_SCAN = True
FREEZE_SOURCES = True
previous = runner.previous


def inputs() -> dict[str, Any]:
    """Use eligible frontier authority and preserve both failed primaries."""
    checked = runner.inputs(sys.modules[__name__])
    for path in (HISTORY, HISTORY_INVENTORY, RECENT_HISTORY, HEALTH, ORIGINAL_SOURCES):
        actual = sha256_file(path) if path.is_file() else "missing"
        checked["checks"].append(
            previous.operand(path, "sha256", PINNED[str(path)], actual, actual)
        )
        if path == HISTORY_INVENTORY or actual != PINNED[str(path)]:
            continue
        document = json.loads(path.read_text())
        if path == ORIGINAL_SOURCES:
            checked["original_source_snapshots"] = document["sources"]
            for label, receipt in document["sources"].items():
                original = Path(label)
                actual_snapshot = sha256_file(original) if original.is_file() else "missing"
                checked["checks"].append(
                    previous.operand(
                        original, "sha256", receipt["sha256"], actual_snapshot, actual_snapshot
                    )
                )
            continue
        if path == HEALTH:
            checked["owned_repository_health"] = document["repository_health"]
            checked["health_repair_snapshots"] = document["frozen_snapshots"]
            continue
        if path == RECENT_HISTORY:
            coverage = document["coverage_statement_counts"].values()
            checked["historical_coverage_failure"] = dict(
                statements=sum(r["num_statements"] for r in coverage),
                covered_statements=sum(r["covered_lines"] for r in coverage),
                failure_operand=document["gate_check_summary"][0],
                producer_execution_date=document["execution_date"],
            )
        for key, expected in (("verdict_class", "disqualified"), ("arc_evidence_ready_score", 0)):
            checked["checks"].append(
                previous.operand(path, key, expected, document.get(key, "missing"), actual)
            )
        checked["prior"].setdefault("historical_required_failures", []).append(
            dict(
                path=str(path),
                sha256=actual,
                producer_execution_date=document["execution_date"],
                verdict_class=document["verdict_class"],
                gate_check_summary=document["gate_check_summary"],
            )
        )
    checked["additional_source_hashes"] = {
        str(p): PINNED[str(p)]
        for p in (HISTORY, HISTORY_INVENTORY, RECENT_HISTORY, HEALTH, ORIGINAL_SOURCES)
    }
    checked["failures"] = [r for r in checked["checks"] if r["expected"] != r["observed"]]
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Retain the failed denominator and measure current health only once."""
    specs = [
        s
        for s in runner.commands(private, sys.modules[__name__])
        if not s["name"].startswith("e2e_016")
    ]
    specs.insert(
        0,
        dict(
            name="e2e_017",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_arc_supervisor_delta_7874.py",
                f"--basetemp={private / 'e2e017'}",
            ],
            expected_exit=0,
            expected_text=None,
            deadline_s=60,
            classification="required",
        ),
    )
    return specs


def authenticate_episode(episode: dict[str, Any]) -> str | None:
    """Reject unsupported outcomes rather than trusting a self-described level-up."""
    if episode.get("synthetic") or episode.get("fixture_claim_scope") or episode.get("is_fixture"):
        return "synthetic_fixture"
    receipt = episode.get("run_receipt")
    if not isinstance(receipt, dict):
        return "missing_live_wrapper_receipt"
    if (
        receipt.get("entrypoint") != "CarnotAgent.choose_action"
        or receipt.get("execution_mode") != "live"
    ):
        return "non_live_wrapper"
    if any(receipt.get(k) != episode.get(k) for k in ("game", "seed", "invocation_id")):
        return "wrapper_identity_mismatch"
    frames = episode.get("action_frames")
    if (
        not isinstance(frames, list)
        or len(frames) < 2
        or receipt.get("frames_sha256") != canonical_hash(frames)
    ):
        return "missing_or_changed_frames"
    indices = [f.get("action_index") for f in frames if isinstance(f, dict)]
    if (
        len(indices) != len(frames)
        or any(type(i) is not int for i in indices)
        or indices != list(range(indices[0], indices[0] + len(indices)))
    ):
        return "noncontiguous_frames"
    if any(
        not isinstance(f.get("frame"), list)
        or not f["frame"]
        or any(
            not isinstance(row, list) or not row or any(type(pixel) is not int for pixel in row)
            for row in f["frame"]
        )
        or type(f.get("levels_completed")) is not int
        or f["levels_completed"] < 0
        for f in frames
    ):
        return "invalid_frame_evidence"
    for redirect in episode["trajectory_supervisor"]["redirects"]:
        index = redirect.get("action_index")
        application = dict(redirect_id=redirect.get("id"), arm=redirect["arm"], action_index=index)
        if application not in receipt.get("applications", []) or index not in indices:
            return "missing_redirect_application"
        start = frames[indices.index(index)]
        levelups = [
            f
            for f in frames
            if f["action_index"] > index and f["levels_completed"] > start["levels_completed"]
        ]
        distance = levelups[0]["action_index"] - index if levelups else None
        if (
            redirect["resolved_by_levelup"] is not bool(levelups)
            or redirect["actions_to_levelup"] != distance
        ):
            return "frame_outcome_mismatch"
    return None


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path, *, current_date: str
) -> dict[str, Any]:
    """Keep the bounded content frontier and add wrapper/frame authentication."""
    started = time.monotonic()
    value = previous.scan(root, producers, checked, private, current_date=current_date)
    seen = set(checked["inventory"].get("outcome_hashes", []))
    episodes: dict[str, dict[str, Any]] = {}
    admitted: set[tuple[str, str]] = set()
    for row in value["rows"]:
        if row["status"] not in {"completed", "censored", "unknown"}:
            continue
        source = Path(row["source_path"])
        elapsed = time.monotonic() - started
        if elapsed >= 120:
            row.update(status="excluded", reason="scan_deadline")
            value["scan_failures"].append(
                previous.operand(source, "scan_deadline_s", 120, elapsed, row["source_sha256"])
            )
            continue
        previous.progress(started, "authenticate_outcome", len(admitted))
        if str(source) not in episodes:
            episodes[str(source)] = {
                canonical_hash(e): e
                for e in previous.reader.extract_rows(json.loads(source.read_text()))
            }
        episode = episodes[str(source)][row["content_sha256"]]
        reason = authenticate_episode(episode)
        producer = json.loads(Path(row["producer_path"]).read_text())
        if producer.get("verdict_class") == "circular_positive" or producer.get(
            "fixture_claim_scope"
        ):
            reason = "synthetic_fixture"
        outcome_hash = canonical_hash(
            [episode["trajectory_supervisor"], episode.get("action_frames")]
        )
        binding = (row["content_sha256"], row["source_sha256"])
        if reason is None and outcome_hash in seen and binding not in admitted:
            reason = "duplicate_outcome_bytes"
        if reason:
            row.update(status="excluded", reason=reason)
        else:
            row.update(frame_evidence_verified=True, outcome_sha256=outcome_hash)
            seen.add(outcome_hash)
            admitted.add(binding)
    value.update(previous.reduce(value["rows"]))
    value["outcome_hashes"] = sorted(seen)
    value["scan_duration_s"] = time.monotonic() - started
    atomic_json(private / "primitive_rows.json", value)
    return value


def positive_control() -> dict[str, Any]:
    """Run a small protocol control to show the reader can accept supported progress."""
    frames = [dict(action_index=i, levels_completed=int(i == 2), frame=[[i]]) for i in (1, 2)]
    value = dict(
        game="protocol",
        seed=0,
        invocation_id="control",
        trajectory_supervisor=dict(
            redirects=[
                dict(
                    id="r",
                    arm="drop_goal_bias",
                    action_index=1,
                    resolved_by_levelup=True,
                    actions_to_levelup=1,
                )
            ]
        ),
        action_frames=frames,
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
    accepted = authenticate_episode(value) is None
    value["synthetic"] = True
    fixture_rejected = authenticate_episode(value) == "synthetic_fixture"
    return dict(
        passed=accepted and fixture_rejected,
        supported_progress_accepted=accepted,
        fixture_rejected=fixture_rejected,
        claim_scope="protocol_fixture_only",
        natural_evidence_count=0,
        genuine_headroom=accepted,
    )


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Report qualification separately from underpowered observational science."""
    fields = history.artifact_fields(value, checked)
    qualified = [
        r for r in value.get("validation_receipts", []) if r["classification"] == "required"
    ]
    state = value.get("verdict_class", "null")
    no_new = not value["new_event_rows"]
    fields.update(
        honest_verdict="complete_null_no_new_outcomes"
        if state == "null" and no_new
        else "complete_null_underpowered_observational_outcomes"
        if state == "null"
        else "complete_" + state + "_supervisor_qualification",
        qualification_receipts=qualified,
        no_new_outcomes=no_new,
        positive_control_results=positive_control(),
        historical_coverage_failure=checked.get("historical_coverage_failure", {}),
        health_repair_snapshots=checked.get("health_repair_snapshots", {}),
        original_source_snapshots=checked.get("original_source_snapshots", {}),
        duration_scope="Monotonic invocation time through candidate freeze; terminal validator times are in the bound sidecar.",
        cited_upstream_artifacts=[
            dict(
                path=str(p),
                sha256=PINNED[str(p)],
                producer_execution_date="20261001",
                fields_imported=["receipt_inventory", "seen_receipt_hashes"]
                if p == INVENTORY
                else ["verdict_class", "gate_check_summary", "coverage_statement_counts"]
                if p == RECENT_HISTORY
                else ["verdict_class", "gate_check_summary"]
                if p == HISTORY
                else ["receipt_inventory", "source_artifact_hashes", "repository_health"],
                role="preserved_disqualified_history"
                if p in {HISTORY, RECENT_HISTORY}
                else "eligible_frontier_authority",
            )
            for p in (PRIOR, INVENTORY, HISTORY, RECENT_HISTORY)
        ],
        observational_limits=[
            "Arm allocation is observational; no causal benefit is estimated.",
            "Protocol controls supply no independent natural observations.",
            "No new game, solve or model invocation is planned.",
        ],
        proposed_generalization_refinement=None,
        retire_if_same_qualification_failure=dict(
            retired=state == "disqualified",
            required_qualification="E2E-017 and nonempty 100 percent owned statement coverage",
        ),
        preserved_disqualified_primaries=[
            dict(
                path=str(p),
                sha256=PINNED[str(p)],
                producer_execution_date="20261001",
                verdict_class="disqualified",
                arc_evidence_ready_score=0,
            )
            for p in (HISTORY, RECENT_HISTORY)
        ],
    )
    fields["frontier"].update(
        outcome_hashes=value.get("outcome_hashes", []), qualification_before_scan=True
    )
    return fields


def replay(value: dict[str, Any]) -> list[str]:
    """Reject supplemental summary drift as well as invented primitive firings."""
    errors = history.replay(value)
    expected = artifact_fields(value, {})
    return errors + [
        key
        for key in (
            "no_new_outcomes",
            "positive_control_results",
            "proposed_generalization_refinement",
        )
        if key in value and value[key] != expected[key]
    ]


def execute(output: Path, private: Path) -> int:
    """Delegate publication after freezing the qualification-first task scope."""
    return runner.execute(output, private, sys.modules[__name__])


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Check exact private candidate bytes, then replay without checkout overrides."""
    copy = private / "terminal-candidate.json"
    shutil.copyfile(candidate, copy)
    report = previous.terminal(copy, private, durable)
    launcher = "import os, runpy, sys; os.chdir(sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    row = previous.run(
        dict(
            name="cold_primary_replay",
            argv=[
                "env",
                "-u",
                "PYTHONPATH",
                str(ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                launcher,
                str(private),
                str(ROOT / CLI),
                "--cold-replay",
                str(copy),
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
    report["replay_errors"] = replay(json.loads(copy.read_text()))
    report["passed"] = (
        report["passed"]
        and row["passed"]
        and not report["replay_errors"]
        and sha256_file(copy) == sha256_file(candidate)
    )
    return report
