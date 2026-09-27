"""Run the V677 scored ARC transport qualification and publish its terminal receipt."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.experiment_7790_v677_arc_runner_qualification import (
    classify_readiness,
    current_metadata,
    verify_selector_assertions,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from scripts.experiments import experiment_7776_v676_arc_runner_qualification as previous

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7790_v677_arc_runner_qualification"
RESULT = ROOT / "results/experiment_7790_v677_arc_runner_qualification.json"
SCOPE = RAW / "frozen_affected_scope.json"
PRIVATE = Path("/tmp/exp7790-v677-validation")
OLD_RESULT = ROOT / "results/experiment_7776_v676_arc_runner_qualification.json"
OLD_PANEL = (
    ROOT / "results/raw/experiment_7776_v676_arc_runner_qualification/arc_panel_manifest.json"
)


def progress(start: float, phase: str, event: str, units: int = 0, **details: Any) -> None:
    """Emit a flushed boundary containing actual elapsed time and completed units."""
    print(
        f"[exp7790] phase={phase} event={event} elapsed_s={time.monotonic() - start:.2f} completed_units={units} {details}",
        flush=True,
    )


def configure_previous() -> None:
    """Keep all reused helper output inside V677's owned paths."""
    previous.RAW, previous.RESULT, previous.SCOPE = RAW, RESULT, SCOPE
    previous.PRIVATE, previous.OLD_RESULT, previous.OLD_PANEL = PRIVATE, OLD_RESULT, OLD_PANEL
    previous._configure_reused_runner()


def main(argv: list[str] | None = None) -> int:
    """Qualify actual SDK transitions and retain every failed validation operand."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "startup", "before", pid=os.getpid())
    configure_previous()
    if args.cold:
        progress(started, "cold_replay", "before")
        raw = json.loads(args.cold.read_text())
        observed = previous.reduce_probe_evidence(raw["schedule"], raw["probes"])
        if observed != raw["summary"]:
            raise ValueError("raw_reduction_mismatch")
        progress(started, "cold_replay", "after", observed["started"])
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    for phase in ("model_load", "generation"):
        progress(started, phase, "before")
        progress(started, phase, "after", 0)
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, arcade, context = previous.preflight(started)
    selector = ROOT / "tests/python/test_arc_ige_cell_selector.py"
    verify_selector_assertions(selector.read_text())
    checks.append(
        previous.prior.check(
            selector, "test_and_assertion_inventory", True, True, "selector_inventory"
        )
    )
    for relative, fields in (
        (
            "results/experiment_7776_v676_arc_runner_qualification.json",
            ["honest_verdict", "organic_runner_ready_score"],
        ),
        ("python/carnot/experiment_7790_v677_arc_runner_qualification.py", []),
        ("scripts/experiments/experiment_7790_v677_arc_runner_qualification.py", []),
    ):
        path = ROOT / relative
        hashes[relative] = {
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": args.date if path.is_file() else None,
            "imported_fields": fields,
            "eligible": path.is_file(),
        }
    spans.append(
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    )
    progress(
        started, "preconditions", "after", len(checks), failed=sum(not c["passed"] for c in checks)
    )
    progress(started, "panel_freeze", "before")
    phase = time.monotonic()
    panel = previous.prior.panel(context)
    old_panel = json.loads(OLD_PANEL.read_text())
    for key in (
        "games",
        "seeds",
        "arms",
        "controls",
        "round_robin_order",
        "max_actions_charged",
        "max_seconds_per_episode",
        "reset_charging_conventions",
        "rows",
    ):
        if panel[key] != old_panel[key]:
            raise ValueError(f"frozen_panel_changed:{key}")
    panel["schema"] = "carnot.exp7790.frozen_arc_panel.v1"
    panel["source_panel_sha256"] = sha256_file(OLD_PANEL)
    panel_path = RAW / "arc_panel_manifest.json"
    atomic_json(panel_path, panel)
    schedule = panel["rows"]
    spans.append(
        {
            "phase": "panel_freeze",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(schedule),
        }
    )
    progress(started, "panel_freeze", "after", len(schedule), sha256=sha256_file(panel_path))
    probes: list[dict[str, Any]] = []
    phase = time.monotonic()
    if all(check["passed"] for check in checks):
        for game, seed, source in (
            ("fixture", 67500, previous.prior._FixtureArcade()),
            ("r11l", 67501, arcade),
        ):
            for arm in previous.prior.ARMS:
                progress(started, "scored_probe", "before", len(probes), game=game, arm=arm)
                row = previous.prior.run_probe(
                    game, seed, arm, source, 3 if game == "fixture" else 12
                )
                probe_path = RAW / f"probe_{len(probes):02d}.json"
                atomic_json(probe_path, row)
                row["raw_path"] = str(probe_path.relative_to(ROOT))
                row["raw_sha256"] = sha256_file(probe_path)
                probes.append(row)
                progress(
                    started,
                    "scored_probe",
                    "after",
                    len(probes),
                    game=game,
                    arm=arm,
                    error=row["error"],
                )
    summary = previous.reduce_probe_evidence(schedule, probes)
    raw_path = RAW / "raw_probes.json"
    atomic_json(raw_path, {"schedule": schedule, "probes": probes, "summary": summary})
    spans.append(
        {
            "phase": "scored_probes",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(probes),
        }
    )
    progress(started, "cold_replay", "before", len(probes))
    phase = time.monotonic()
    python = str(ROOT / ".venv/bin/python")
    cold = run_commands(
        ROOT,
        [
            CommandSpec(
                "cold_reduce",
                (python, "-u", __file__, "--cold", str(raw_path)),
                "fresh_process_raw_reduction",
                120,
            )
        ],
        log_dir=RAW / "cold_logs",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": "cpu", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
    )
    spans.append(
        {
            "phase": "cold_replay",
            "duration_s": time.monotonic() - phase,
            "completed_units": int(cold[0]["passed"]),
        }
    )
    progress(started, "cold_replay", "after", int(cold[0]["passed"]))
    receipts = list(cold)
    diagnostic = None
    if all(check["passed"] for check in checks):
        phase = time.monotonic()
        progress(started, "validation", "before", len(receipts))
        current, diagnostic = previous.validation(started)
        receipts.extend(current)
        if diagnostic is not None:
            receipts.append(dict(diagnostic, name="full_python_suite"))
        spans.append(
            {
                "phase": "validation",
                "duration_s": time.monotonic() - phase,
                "completed_units": sum(row["passed"] for row in receipts),
            }
        )
        progress(
            started,
            "validation",
            "after",
            sum(row["passed"] for row in receipts),
            checks=len(receipts),
        )
    value = previous.prior.artifact(
        schedule,
        probes,
        checks,
        hashes,
        context["resource"],
        receipts,
        spans,
        started,
        args.date,
        panel_path,
    )
    value = current_metadata(value, args.date)
    value.update(
        frozen_validation_scope=json.loads(SCOPE.read_text()),
        organic_panel_manifest_path=str(panel_path.relative_to(ROOT)),
        organic_panel_manifest_sha256=sha256_file(panel_path),
        raw_reduction=summary,
        raw_probes_sha256=sha256_file(raw_path),
        repository_health={
            "current_collection_diagnostic": diagnostic,
            "historical_exp7776_verdict": json.loads(OLD_RESULT.read_text())["honest_verdict"],
        },
        supervisor_outcome_receipts=[
            {
                "episode_id": row["episode_id"],
                "observed_goal_firings": sum(
                    action.get("goal_firing") is not None for action in row["actions"]
                ),
                "arm_changed_by_supervisor": False,
                "source": "raw_action_telemetry",
            }
            for row in probes
        ],
        reset_charging_interpretations={
            "charged": {row["episode_id"]: row["actions_charged"] for row in probes},
            "uncharged": {
                row["episode_id"]: row["actions_charged"]
                - sum(action.get("action") == "RESET" for action in row["actions"])
                for row in probes
            },
            "budget_interpretation": "charged",
        },
    )
    value["source_artifact_hashes"] = hashes
    value["preconditions_checked"]["resources"]["sdk_version"] = importlib.metadata.version(
        "arc-agi"
    )
    value["reproducibility_checksum"] = canonical_hash(
        {
            "source_artifact_hashes": hashes,
            "scope_sha256": sha256_file(SCOPE),
            "panel_sha256": sha256_file(panel_path),
            "raw_sha256": sha256_file(raw_path),
            "seeds": list(previous.prior.SEEDS),
            "roles": list(previous.prior.ARMS),
            "parameters": panel["controls"],
        }
    )
    value["field_principles"].update(
        {
            "experiment_id": "Each result has one current owner.",
            "organic_panel_manifest_path": "The paired panel needs fixed games, roles, seeds, and bounds.",
            "repository_health": "Broad collection is reported separately from affected checks.",
            "reset_charging_interpretations": "Both RESET readings remain inspectable.",
            "source_artifact_hashes": "Exact source bytes bind imported fields and eligibility.",
            "reproducibility_checksum": "Code, data, roles, controls, and seeds bind replay.",
        }
    )
    value["honest_verdict"] = (
        "complete_blocked_preconditions"
        if any(not c["passed"] for c in checks)
        else "complete_null_runner_qualification_pending_terminal"
    )
    value["verdict_class"] = "blocked" if any(not c["passed"] for c in checks) else "null"
    candidate = RAW / "terminal_candidate.json"
    progress(started, "candidate", "before")
    atomic_json(candidate, value)
    progress(started, "candidate", "after", 1, sha256=sha256_file(candidate))
    phase = time.monotonic()
    first = previous.terminal_readers(started, candidate, "preliminary")
    sdk = [row for row in probes if row["claim_scope"] == "adapter_withheld_public"]
    required = value["frozen_validation_scope"]["required_checks"]
    ready, failed = classify_readiness([*receipts, *first], required, sdk)
    final_readers = first
    if ready:
        value["organic_runner_ready_score"] = 1
        value["acceptance_gate_results"]["readiness"] = True
        value["acceptance_gate_results"]["validity"] = True
        value["honest_verdict"] = "complete_null_runner_qualified_no_benefit_measurement"
        value["verdict_class"] = "null"
        ready_path = RAW / "terminal_candidate_ready.json"
        atomic_json(ready_path, value)
        final_readers = previous.terminal_readers(started, ready_path, "ready")
        ready, failed = classify_readiness([*receipts, *final_readers], required, sdk)
    if not ready:
        blocked = any(not c["passed"] for c in checks)
        value["organic_runner_ready_score"] = 0
        value["acceptance_gate_results"].update(validity=False, readiness=False)
        value["verdict_class"] = "blocked" if blocked else "disqualified"
        value["honest_verdict"] = (
            "complete_blocked_preconditions"
            if blocked
            else "complete_disqualified_required_runner_validation"
        )
        for name in failed:
            receipt = next(
                (row for row in [*receipts, *final_readers] if row["name"] == name), None
            )
            value["gate_check_summary"].append(
                previous.prior.check(
                    ROOT / receipt["log_path"] if receipt else RAW / "validation",
                    name,
                    0,
                    receipt["exit_code"] if receipt else "missing",
                    "current_validation",
                )
            )
    value["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in final_readers
    )
    value["validation_receipts"] = [*receipts, *final_readers]
    spans.append(
        {
            "phase": "terminal_readers",
            "duration_s": time.monotonic() - phase,
            "completed_units": sum(row["passed"] for row in final_readers),
        }
    )
    atomic_json(
        RAW / "terminal_receipts.json",
        {
            "preliminary_candidate_sha256": sha256_file(candidate),
            "preliminary_commands": first,
            "final_candidate_sha256": sha256_file(RAW / "terminal_candidate_ready.json")
            if (RAW / "terminal_candidate_ready.json").is_file()
            else None,
            "final_commands": final_readers,
        },
    )
    value["phase_spans"] = spans
    value["duration_s"] = time.monotonic() - started
    progress(started, "publish", "before", 0, verdict=value["honest_verdict"])
    atomic_json(RESULT, value)
    progress(started, "publish", "after", 1, sha256=sha256_file(RESULT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
