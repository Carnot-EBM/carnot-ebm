"""Read new live supervisor outcomes and preserve an honest no-change verdict.

The reader works from exact producer bytes. It makes no gameplay or model call.
REQ-ARC-7831 and SCENARIO-ARC-7831-* describe its evidence boundary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.agentic.arc_supervisor_refinement import (
    classify_receipt,
    extract_rows,
    receipt_id_for_row,
    wilson_bounds,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_REL = Path(
    "results/raw/experiment_7831_v680_arc_supervisor_refinement/validation_command_manifest.json"
)
RESULT_REL = Path("results/experiment_7831_v680_arc_supervisor_refinement.json")
DEFAULT_CANDIDATES = (
    {"producer": "results/experiment_7722_v672_arc_evidence_recovery.json"},
    {"producer": "results/experiment_7817_v679_arc_runner_qualification.json"},
    {
        "producer": "results/experiment_7818_v679_arc_organic_measurement.json",
        "raw": "results/raw/experiment_7818_v679_arc_organic_measurement/raw_rows.json",
    },
)
MODEL_SPECS: list[dict[str, Any]] = []


def progress(started: float, phase: str, completed: int) -> None:
    """Report completed units so a silent long scan is distinguishable from a stall."""

    print(
        f"[exp7831] phase={phase} elapsed_s={time.monotonic() - started:.3f} "
        f"completed_units={completed}",
        flush=True,
    )


def _expected_hash(producer: Mapping[str, Any], raw: str) -> str | None:
    hashes = producer.get("source_artifact_hashes") or {}
    entry = hashes.get(raw) if isinstance(hashes, Mapping) else None
    if isinstance(entry, str):
        return entry if entry.startswith("sha256:") else "sha256:" + entry
    if isinstance(entry, Mapping):
        value = entry.get("sha256")
        return (
            value
            if not isinstance(value, str) or value.startswith("sha256:")
            else "sha256:" + value
        )
    if raw == producer.get("raw_rows_path"):
        value = producer.get("raw_rows_sha256")
        return (
            value
            if not isinstance(value, str) or value.startswith("sha256:")
            else "sha256:" + value
        )
    return None


def _complete_receipt(receipt: Mapping[str, Any], mode: str) -> str | None:
    required = (
        ("redirects", "arm_outcomes")
        if mode == "applied"
        else (
            "would_have_redirects",
            "would_have_arm_outcomes",
        )
    )
    for field in (*required, "stagnations_unredirected"):
        if field not in receipt:
            return f"trajectory_supervisor.{field}"
    redirects = receipt[required[0]]
    outcomes = receipt[required[1]]
    if not isinstance(redirects, list) or not isinstance(outcomes, Mapping):
        return f"trajectory_supervisor.{required[0]}_or_{required[1]}"
    for redirect in redirects:
        if not isinstance(redirect, Mapping):
            return f"trajectory_supervisor.{required[0]}[]"
        fields = (
            ("arm", "resolved_by_levelup", "actions_to_levelup") if mode == "applied" else ("arm",)
        )
        for field in fields:
            if field not in redirect:
                return f"trajectory_supervisor.{required[0]}[].{field}"
    if mode == "applied":
        counted: dict[str, dict[str, int]] = defaultdict(lambda: {"fired": 0, "helped": 0})
        for redirect in redirects:
            arm = str(redirect["arm"])
            counted[arm]["fired"] += 1
            counted[arm]["helped"] += int(redirect["resolved_by_levelup"] is True)
        if dict(counted) != outcomes:
            return "trajectory_supervisor.arm_outcomes"
    return None


def inspect_candidates(
    root: Path, candidates: Sequence[Mapping[str, str]], baseline_hashes: set[str]
) -> dict[str, Any]:
    """Authenticate explicit source paths, then classify only real redirect receipts."""

    inventory: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    schema_missing: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        producer_label = candidate["producer"]
        producer_path = root / producer_label
        item: dict[str, Any] = {"producer": producer_label, "raw": candidate.get("raw")}
        inventory.append(item)
        if not producer_path.is_file():
            item["disposition"] = "missing"
            continue
        item["producer_sha256"] = sha256_file(producer_path)
        producer = json.loads(producer_path.read_text(encoding="utf-8"))
        item["run_date"] = producer.get("run_date")
        item["role"] = (
            "runner_qualification"
            if "runner_qualification" in producer_label
            else "science_producer"
        )
        raw_label = candidate.get("raw")
        raw_path = root / raw_label if raw_label else producer_path
        if not raw_path.is_file():
            item["disposition"] = "missing"
            continue
        item["raw_sha256"] = sha256_file(raw_path)
        if raw_label and _expected_hash(producer, raw_label) != item["raw_sha256"]:
            item["disposition"] = "hash_mismatch"
            continue
        if item["raw_sha256"] in baseline_hashes:
            item["disposition"] = "historical_duplicate"
            continue
        if producer.get("flagged_adversarial") is True or producer.get("verdict_class") in {
            "disqualified",
            "blocked",
        }:
            item["disposition"] = "disqualified"
            item["raw_organic_selector_comparison"] = producer.get("raw_reduction")
            continue
        doc = json.loads(raw_path.read_text(encoding="utf-8"))
        rows = extract_rows(doc)
        duplicate_count = 0
        for row in rows:
            kind = classify_receipt(row)
            if kind not in {"applied", "shadow"}:
                continue
            receipt = row["trajectory_supervisor"]
            missing = _complete_receipt(receipt, kind)
            if missing:
                schema_missing.append(
                    {
                        "path": raw_label or producer_label,
                        "field": missing,
                        "producer": producer_label,
                    }
                )
                continue
            identity = receipt_id_for_row(row)
            if identity in seen:
                duplicate_count += 1
                continue
            seen.add(identity)
            eligible.append(
                {"row": row, "path": raw_label or producer_label, "sha256": item["raw_sha256"]}
            )
        item["duplicate_rows"] = duplicate_count
        item["eligible_rows"] = len(
            [r for r in eligible if r["path"] == (raw_label or producer_label)]
        )
        item["disposition"] = (
            "eligible" if item["eligible_rows"] else "unqualified_no_redirect_schema"
        )
    return {"inventory": inventory, "eligible": eligible, "schema_missing_rows": schema_missing}


def reduce_eligible(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join applied redirects to their outcomes without crediting shadow proposals."""

    rows: list[dict[str, Any]] = []
    controls = 0
    per_game: dict[str, dict[str, int]] = defaultdict(
        lambda: {"fired": 0, "helped": 0, "censored": 0}
    )
    stagnations = 0
    for record in records:
        row = record["row"]
        receipt = row["trajectory_supervisor"]
        mode = classify_receipt(row)
        stagnations += int(receipt["stagnations_unredirected"])
        if mode == "shadow":
            controls += 1
            continue
        termination = row.get("termination") or {}
        reason = termination.get("reason") if isinstance(termination, Mapping) else None
        for redirect in receipt["redirects"]:
            resolved = redirect["resolved_by_levelup"] is True
            censored = not resolved and reason in {
                "action_limit",
                "time_limit",
                "timeout",
                "collection_cap",
            }
            game = str(row.get("game"))
            outcome = {
                "game": game,
                "seed": row.get("seed"),
                "arm": str(redirect["arm"]),
                "resolved_by_levelup": resolved,
                "actions_to_levelup": redirect["actions_to_levelup"],
                "stagnations_unredirected": receipt["stagnations_unredirected"],
                "censoring": {"censored": censored, "reason": reason if censored else None},
                "raw_provenance": {"path": record["path"], "sha256": record["sha256"]},
                "numerator": int(resolved),
                "denominator": int(not censored),
                "rate": None if censored else float(resolved),
                "direction": "helped" if resolved else ("censored" if censored else "not_helped"),
            }
            rows.append(outcome)
            per_game[game]["fired"] += 1
            per_game[game]["helped"] += int(resolved)
            per_game[game]["censored"] += int(censored)
    arm_statistics: dict[str, dict[str, Any]] = {}
    recommendations: list[dict[str, Any]] = []
    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        uncensored = [r for r in arm_rows if not r["censoring"]["censored"]]
        helped = sum(r["numerator"] for r in uncensored)
        games = sorted({r["game"] for r in uncensored})
        lower, upper = wilson_bounds(helped, len(uncensored))
        arm_statistics[arm] = {
            "fired": len(arm_rows),
            "uncensored": len(uncensored),
            "censored": len(arm_rows) - len(uncensored),
            "helped": helped,
            "games": games,
            "wilson_lower95": lower,
            "wilson_upper95": upper,
        }
        if len(uncensored) >= 8 and len(games) >= 3:
            if helped == 0:
                recommendations.append({"arm": arm, "kind": "retire_candidate"})
            elif lower > 0.25:
                recommendations.append({"arm": arm, "kind": "priority_consideration"})
    if not records:
        verdict = "complete_null_no_new_eligible_receipts"
    elif not rows:
        verdict = "complete_null_no_firings_nothing_to_refine"
    else:
        verdict = (
            "complete_positive_observational_arm_screen"
            if recommendations
            else ("complete_null_insufficient_arm_support")
        )
    return {
        "honest_verdict": verdict,
        "verdict_class": "null" if "null" in verdict else "positive",
        "rows": rows,
        "arm_statistics": arm_statistics,
        "per_game_results": dict(sorted(per_game.items())),
        "recommendation_rows": recommendations,
        "new_source_count": len(records),
        "shadow_control_count": controls,
        "stagnations_unredirected": stagnations,
        "general_mechanism_requirement": (
            "A game-blind search constraint is required after all enabled arms are spent."
            if any(
                isinstance(r["row"]["trajectory_supervisor"].get("arms_enabled"), list)
                and set(r["row"]["trajectory_supervisor"].get("arms_enabled") or [])
                <= set(r["row"]["trajectory_supervisor"].get("arms_used") or [])
                and r["row"]["trajectory_supervisor"]["stagnations_unredirected"] > 0
                for r in records
            )
            else None
        ),
    }


def validate_child_receipts(
    receipts: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], *, require_all: bool
) -> list[str]:
    """Check exact argv, unique names, exits, and sealed bytes in a cold process."""

    declared = {row["name"]: row for row in manifest["commands"]}
    errors: list[str] = []
    seen: set[str] = set()
    for receipt in receipts:
        name = str(receipt.get("name"))
        if name not in declared:
            errors.append(f"undeclared_child:{name}")
            continue
        if name in seen:
            errors.append(f"duplicate_child:{name}")
        seen.add(name)
        if receipt.get("command_argv") != declared[name]["argv"]:
            errors.append(f"argv_mismatch:{name}")
        if receipt.get("exit_code") != 0 and declared[name]["class"] == "required":
            errors.append(f"failed_required:{name}")
        log_path = Path(str(receipt.get("log_path")))
        if not log_path.is_file() or sha256_file(log_path) != receipt.get("log_sha256"):
            errors.append(f"log_hash_mismatch:{name}")
    if require_all:
        errors.extend(
            f"missing_required:{name}"
            for name, item in declared.items()
            if item["class"] == "required" and name not in seen
        )
    return errors


def _run_declared(
    manifest: Mapping[str, Any],
    started: float,
    reuse: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run frozen commands and seal each closed child log at an immutable byte path."""

    raw_root = ROOT / "results/raw/experiment_7831_v680_arc_supervisor_refinement"
    attempt = raw_root / "attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    # The frozen argv names this stable alias. Point it at a fresh private
    # parent before pytest starts, so retries cannot erase an earlier attempt.
    if any(
        arg.startswith("--basetemp=/tmp/carnot-exp7831-attempt/fixed/")
        for spec in manifest["commands"]
        for arg in spec["argv"]
    ):
        alias = Path("/tmp/carnot-exp7831-attempt/fixed")
        alias.parent.mkdir(parents=True, exist_ok=True)
        target = attempt / "pytest_parent"
        target.mkdir()
        if alias.is_symlink():
            alias.unlink()
        elif alias.exists():
            raise ValueError(f"pytest parent alias is not owned: {alias}")
        alias.symlink_to(target, target_is_directory=True)
    sealed = raw_root / "sealed_logs"
    sealed.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    for index, spec in enumerate(manifest["commands"]):
        name = spec["name"]
        argv = spec["argv"]
        if reuse and name in reuse:
            receipts.append({**reuse[name], "reused_from_prior_attempt": True})
            progress(started, f"reused_completed_child:{name}", index + 1)
            continue
        progress(started, f"before_subprocess:{name}", index)
        temp = attempt / f"{index:02d}_{name}.log"
        start = time.monotonic()
        with temp.open("wb") as stream:
            child = subprocess.Popen(  # noqa: S603 - frozen argv, no shell.
                argv,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": "python:.", "PYTHONUNBUFFERED": "1"},
            )
            deadline = start + (180 if name == "repository_health_180s" else 900)
            last_heartbeat = start
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                if time.monotonic() - last_heartbeat >= 30:
                    progress(started, f"child_outstanding:{name}", index)
                    last_heartbeat = time.monotonic()
                time.sleep(0.5)
            exit_code = child.wait()
        digest = sha256_file(temp)
        destination = sealed / f"{digest.removeprefix('sha256:')}.log"
        if destination.exists() and sha256_file(destination) != digest:
            raise ValueError(f"sealed log collision: {destination}")
        if not destination.exists():
            destination.write_bytes(temp.read_bytes())
        receipts.append(
            {
                "name": name,
                "command_argv": argv,
                "class": spec["class"],
                "exit_code": exit_code,
                "duration_s": time.monotonic() - start,
                "log_path": str(destination.relative_to(ROOT)),
                "log_sha256": digest,
            }
        )
        progress(started, f"after_subprocess:{name}", index + 1)
    return receipts


def _artifact(
    inventory: Mapping[str, Any], reduced: Mapping[str, Any], started: float
) -> dict[str, Any]:
    """Bind one current result to exact inputs and keep unmeasured gates null."""

    missing = [
        {
            "upstream_id": item["producer"],
            "path": item.get("raw") or item["producer"],
            "sha256": item.get("raw_sha256"),
            "field": "path_or_authenticated_hash",
            "op": "exists_and_equals",
            "expected": "authenticated_bytes",
            "observed": item["disposition"],
        }
        for item in inventory["inventory"]
        if item["disposition"] in {"missing", "hash_mismatch"}
    ]
    missing.extend(
        {
            "upstream_id": row["producer"],
            "path": row["path"],
            "sha256": next(
                (
                    item.get("raw_sha256")
                    for item in inventory["inventory"]
                    if item["producer"] == row["producer"]
                ),
                None,
            ),
            "field": row["field"],
            "op": "present_and_complete",
            "expected": True,
            "observed": False,
        }
        for row in inventory["schema_missing_rows"]
    )
    verdict = (
        "complete_blocked_missing_or_invalid_new_receipt" if missing else reduced["honest_verdict"]
    )
    manifest_path = ROOT / MANIFEST_REL
    hashes = [
        {
            "path": item["producer"],
            "sha256": item.get("producer_sha256"),
            "role": item.get("role", "science_producer"),
            "eligibility": item["disposition"],
            "date": item.get("run_date"),
        }
        for item in inventory["inventory"]
    ]
    hashes.extend(
        {
            "path": item["raw"],
            "sha256": item["raw_sha256"],
            "role": "raw_receipt",
            "eligibility": item["disposition"],
            "date": item.get("run_date"),
        }
        for item in inventory["inventory"]
        if item.get("raw_sha256") and item.get("raw")
    )
    checksum_input = json.dumps(
        [hashes, sha256_file(manifest_path), sha256_file(Path(__file__)), 7831], sort_keys=True
    )
    result: dict[str, Any] = {
        "schema": "carnot.experiment_7831_v680_arc_supervisor_refinement.v1",
        "experiment_id": 7831,
        "milestone": "2026.09.680",
        "run_date": "20260928",
        "honest_verdict": verdict,
        "verdict_class": "blocked" if missing else reduced["verdict_class"],
        "flagged_adversarial": False,
        "gate_check_summary": missing,
        "rows": [
            *reduced["rows"],
            *(
                {
                    "unit": item["producer"],
                    "check": "new_redirect_receipt_inventory",
                    "source_path": item.get("raw") or item["producer"],
                    "source_sha256": item.get("raw_sha256"),
                    "disposition": item["disposition"],
                    "eligible_receipt_count": item.get("eligible_rows", 0),
                    "duplicate_receipt_count": item.get("duplicate_rows", 0),
                }
                for item in inventory["inventory"]
            ),
        ],
        "acceptance_gate_results": {
            key: (not missing if key == "validity" else None)
            for key in (
                "validity",
                "readiness",
                "probability_quality",
                "decision_benefit",
                "retention",
                "efficiency",
            )
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": [],
        "random_seed": 7831,
        "reproducibility_checksum": "sha256:" + hashlib.sha256(checksum_input.encode()).hexdigest(),
        "sample_size_budget": {
            "intended": len(DEFAULT_CANDIDATES),
            "eligible": len(inventory["eligible"]),
            "started": len(inventory["inventory"]),
            "completed": len(inventory["inventory"]),
            "excluded": sum(item["disposition"] != "eligible" for item in inventory["inventory"]),
            "censored": sum(r["censoring"]["censored"] for r in reduced["rows"]),
            "independent_n": len({r["game"] for r in reduced["rows"]}),
        },
        "source_artifact_hashes": hashes,
        "preconditions_checked": {
            "explicit_paths": [item["producer"] for item in inventory["inventory"]],
            "manifest_sha256": sha256_file(manifest_path),
            "models_required": False,
        },
        "validation_receipts": [],
        "validation_command_manifest_path": MANIFEST_REL.as_posix(),
        "observed_child_commands": [],
        "repository_health": {
            "status": "degraded_open",
            "historical_full_suite_obligations_open": True,
        },
        "verifier_is_oracle": False,
        "claim_scope": "observational_redirect_outcome_ledger_no_solve_or_causal_claim",
        "inference_substrate": None if missing else "aggregation_from_upstream_artifacts",
        "inference_substrate_class": None if missing else "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"calls": 0, "tokens": 0, "file_hashes": []},
        "supervisor_inventory_ready_score": int(not missing),
        "arc_generalization_activity": "supervisor_refinement_from_outcome_bearing_redirect_ledger",
        "arm_statistics": reduced["arm_statistics"],
        "per_game_results": reduced["per_game_results"],
        "recommendation_rows": reduced["recommendation_rows"],
        "new_source_count": reduced["new_source_count"],
        "schema_missing_rows": inventory["schema_missing_rows"],
        "source_inventory": inventory["inventory"],
        "shadow_control_count": reduced["shadow_control_count"],
        "stagnations_unredirected": reduced["stagnations_unredirected"],
        "general_mechanism_requirement": reduced["general_mechanism_requirement"],
        "solve_claim": False,
        "production_defaults_changed": False,
    }
    result["field_principles"] = {key: "Preserve the stated evidence boundary." for key in result}
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch only frozen work or cold verification of exact sealed bytes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / MANIFEST_REL).read_text(encoding="utf-8"))
    if args.check_manifest:
        print(json.dumps({"command_names": [row["name"] for row in manifest["commands"]]}))
        return 0
    if args.cold_replay:
        candidate = json.loads(args.cold_replay.read_text(encoding="utf-8"))
        errors = validate_child_receipts(
            candidate["validation_receipts"], manifest, require_all=False
        )
        print(json.dumps({"errors": errors}))
        return int(bool(errors))
    if args.date != "20260928":
        parser.error("--date must be 20260928")
    started = time.monotonic()
    progress(started, "preconditions", 0)
    baseline_path = ROOT / "results/experiment_7625_v665_arc_supervisor_transfer.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old = {row["sha256"] for row in baseline["source_artifact_hashes"]["actual_producers"]}
    progress(started, "inventory", 0)
    inventory = inspect_candidates(ROOT, DEFAULT_CANDIDATES, old)
    progress(started, "inventory", len(inventory["inventory"]))
    reduced = reduce_eligible(inventory["eligible"])
    result = _artifact(inventory, reduced, started)
    result["source_artifact_hashes"].append(
        {
            "path": str(baseline_path.relative_to(ROOT)),
            "sha256": sha256_file(baseline_path),
            "role": "historical_dedup_baseline",
            "eligibility": "not_current_benefit",
            "date": baseline.get("run_date"),
        }
    )
    result["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                [
                    result["source_artifact_hashes"],
                    sha256_file(ROOT / MANIFEST_REL),
                    sha256_file(Path(__file__)),
                    7831,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )
    progress(started, "validation", 0)
    prior_path = ROOT / RESULT_REL
    reused: dict[str, dict[str, Any]] = {}
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        for receipt in prior.get("validation_receipts", []):
            if receipt.get("name") in {"all_python_tests", "repository_health_180s"}:
                errors_for_receipt = validate_child_receipts([receipt], manifest, require_all=False)
                if set(errors_for_receipt) <= {"failed_required:all_python_tests"}:
                    reused[receipt["name"]] = receipt
    receipts = _run_declared(manifest, started, reused)
    result["validation_receipts"] = receipts
    result["observed_child_commands"] = [
        {"name": row["name"], "argv": row["command_argv"], "class": row["class"]}
        for row in receipts
        if not row.get("reused_from_prior_attempt")
    ]
    errors = validate_child_receipts(receipts, manifest, require_all=True)
    if errors:
        result["honest_verdict"] = "complete_disqualified_required_validation"
        result["verdict_class"] = "disqualified"
        result["acceptance_gate_results"]["validity"] = False
        result["acceptance_gate_results"]["readiness"] = 0
        result["validation_errors"] = errors
    result["duration_s"] = time.monotonic() - started
    owned_validation_s = sum(
        row["duration_s"] for row in receipts if not row.get("reused_from_prior_attempt")
    )
    result["phase_spans"] = [
        {"phase": "validation", "duration_s": owned_validation_s},
        {
            "phase": "other_owned_work",
            "duration_s": max(0.0, result["duration_s"] - owned_validation_s),
        },
    ]
    result["field_principles"] = {key: "Preserve the stated evidence boundary." for key in result}
    progress(started, "atomic_result", len(receipts))
    atomic_json(ROOT / RESULT_REL, result)
    progress(started, "complete", len(receipts))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
