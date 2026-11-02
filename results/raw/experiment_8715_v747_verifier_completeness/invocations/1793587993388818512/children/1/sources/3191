"""Repair source-view custody under REQ-REPORT-7810.

A single family manifest controls both preparation order and the cold reader.
The current attempt keeps its candidate away from historical test scratch paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from carnot import experiment_7727_v673_development_corpus as corpus
from carnot import experiment_7768_v676_source_view_qualification as views
from carnot import experiment_7796_v678_source_view_qualification as prior
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7810_v679_source_view_qualification"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
COMMAND_MANIFEST = RAW / "validation_command_manifest.json"
COMMAND_MANIFEST_SHA256 = "sha256:72e15eda9cdc5f87dfcdb2e7f7598dca8258f63f5aeecc9233e37ca1a05cd4ab"
MODEL_SPECS: list[dict[str, Any]] = []


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Expose measured progress while large corpus and child checks run."""
    print(
        f"[exp7810] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def load_command_manifest() -> dict[str, Any]:
    """Read the prospective command bytes sealed before implementation."""
    if sha256_file(COMMAND_MANIFEST) != COMMAND_MANIFEST_SHA256:
        raise ValueError("validation_manifest_drift")
    value: dict[str, Any] = json.loads(COMMAND_MANIFEST.read_text())
    return value


def validate_command_manifest(value: dict[str, Any]) -> None:
    """Reject any post-freeze command, argument, or classification change."""
    if value != load_command_manifest():
        raise ValueError("validation_manifest_drift")
    commands = value["commands"]
    if len(commands) != 12 or len({row["name"] for row in commands}) != 12:
        raise ValueError("validation_manifest_drift")
    if any(
        row["classification"] != "required"
        for row in commands
        if row["name"] != "repository_health"
    ):
        raise ValueError("validation_manifest_drift")
    if commands[8]["name"] != "repository_health" or commands[8]["classification"] != "diagnostic":
        raise ValueError("validation_manifest_drift")


def _rows(path: Path) -> list[dict[str, Any]]:
    """Read sealed JSON lines without changing their family order."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check_candidate_roster(manifest_path: Path, raw: Path, candidate_path: Path) -> dict[str, Any]:
    """Compare every candidate unit with raw bytes and the manifest roster."""
    inventory = json.loads(manifest_path.read_text())
    public = _rows(raw / "rows.jsonl")
    targets = _rows(raw / "targets.jsonl")
    candidate = json.loads(candidate_path.read_text())
    expected_ids = [
        family for role in corpus.COUNTS for family in inventory["roles"][role]["families"]
    ]
    raw_ids = [row["family_id"] for row in public]
    if raw_ids != expected_ids or len(set(raw_ids)) != 640 or len(targets) != 640:
        raise ValueError("candidate_roster_mismatch")
    if [row["family_id"] for row in targets] != raw_ids:
        raise ValueError("candidate_roster_mismatch")
    expected = prior.build_candidate(
        ROOT, [], {"rows": public, "targets": targets}, [], [], raw=raw
    )["rows"]
    if candidate.get("rows") != expected:
        raise ValueError("candidate_roster_mismatch")
    return {"families": len(expected), "role_counts": corpus.COUNTS}


def cold_reduce(manifest_path: Path, raw: Path, candidate_path: Path) -> dict[str, Any]:
    """Recompute complete views from authenticated shards in a fresh process."""
    roster = check_candidate_roster(manifest_path, raw, candidate_path)
    replay = views.replay_corpus(manifest_path, raw, corpus.COUNTS)
    if replay["families"] != roster["families"]:
        raise ValueError("candidate_roster_mismatch")
    return roster


def validate_log_receipt(receipt: dict[str, Any]) -> None:
    """A changed durable log invalidates the exit it was meant to prove."""
    path = Path(receipt["log_path"])
    if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
        raise ValueError("validation_log_drift")


def execute_child(command: dict[str, Any], index: int, scope: dict[str, Any]) -> dict[str, Any]:
    """Wait for the owned child and seal its closed log at a unique digest path."""
    private = Path(scope["private_root"])
    raw = Path(scope["raw_root"])
    spec = CommandSpec(
        command["name"],
        tuple(command["argv"]),
        command["classification"],
        timeout_s=float(command["timeout_s"]),
    )
    receipt = run_commands(
        ROOT,
        [spec],
        log_dir=private / "logs" / f"{index:02d}_{command['name']}",
        extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )[0]
    source = ROOT / receipt["log_path"]
    digest = sha256_file(source)
    destination = raw / "validation_logs" / f"{index:02d}_{command['name']}_{digest[7:]}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("validation_log_path_reused")
    shutil.copyfile(source, destination)
    receipt.update(
        classification=command["classification"],
        log_path=str(destination),
        log_sha256=sha256_file(destination),
    )
    validate_log_receipt(receipt)
    return receipt


def dispatch(
    scope: dict[str, Any],
    executor: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the complete frozen list, including readers after helper commands."""
    validate_command_manifest(scope)
    (Path(scope["private_root"]) / "basetemp").mkdir(parents=True, exist_ok=True)
    chosen = executor or execute_child
    receipts = []
    for index, command in enumerate(scope["commands"]):
        receipt = chosen(command, index, scope)
        if (
            receipt["name"] != command["name"]
            or receipt["command_argv"] != command["argv"]
            or receipt["classification"] != command["classification"]
        ):
            raise ValueError("observed_child_command_drift")
        validate_log_receipt(receipt)
        receipts.append(receipt)
    return receipts


def _build_result(
    checks: list[dict[str, Any]],
    prepared: dict[str, Any] | None,
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    raw: Path,
    flagged: bool,
    started: float,
) -> dict[str, Any]:
    """Keep historical reducer semantics while binding the current owner."""
    required = [row for row in receipts if row["classification"] == "required"]
    value = prior.build_candidate(ROOT, checks, prepared, required, spans, flagged=flagged, raw=raw)
    manifest = load_command_manifest()
    identity = {
        "code": sha256_file(Path(__file__)),
        "wrapper": sha256_file(ROOT / "scripts/experiments" / f"{NAME}.py"),
        "inputs": value["source_artifact_hashes"],
        "roles": corpus.COUNTS,
        "validation_manifest": sha256_file(COMMAND_MANIFEST),
        "seed": 7810,
    }
    value.update(
        schema="carnot.exp7810.source_view_qualification.v1",
        experiment_id="exp7810-source-view-qualification",
        milestone="2026.09.679",
        run_date="20260928",
        duration_s=time.monotonic() - started,
        random_seed=7810,
        reproducibility_checksum=hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        validation_receipts=receipts,
        validation_command_manifest_path=str(COMMAND_MANIFEST),
        validation_command_manifest_sha256=sha256_file(COMMAND_MANIFEST),
        observed_child_commands=[
            {
                "name": row["name"],
                "argv": row["command_argv"],
                "classification": row["classification"],
            }
            for row in receipts
        ],
        repository_health={
            "historical_exp7782_verdict": "complete_disqualified_required_validation",
            "historical_exp7796_verdict": "complete_disqualified_required_validation",
            "broad_suite_exit": next(
                (row["exit_code"] for row in receipts if row["name"] == "repository_health"), None
            ),
            "broad_suite_classification": "diagnostic",
        },
        claim_scope="exposed_development_fixture_only; no hidden generalization or oracle-distinct benefit",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        planned_inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_specs=[],
        model_invocation_counts={"loads": 0, "calls": 0, "tokens": 0, "loaded_file_hashes": []},
    )
    value["field_principles"].update(
        validation_command_manifest_path="The command boundary is fixed before compute.",
        validation_command_manifest_sha256="Exact argv must survive replay.",
        observed_child_commands="A dispatcher cannot omit appended children.",
        repository_health="Broad health cannot become an affected-suite pass.",
    )
    return value


def run_experiment(date: str) -> dict[str, Any]:
    """Prepare exact current inputs, dispatch validation, and publish once."""
    started = time.monotonic()
    progress(started, "start", "begin")
    if date != "20260928":
        raise ValueError("run_date_mismatch")
    scope = load_command_manifest()
    validate_command_manifest(scope)
    raw = Path(scope["raw_root"])
    candidate = Path(scope["candidate_path"])
    if raw.exists():
        raise ValueError("attempt_root_reused")
    raw.mkdir(parents=True)
    progress(started, "preconditions", "begin")
    phase = time.monotonic()
    checks = prior.preconditions(ROOT)
    spans = [{"phase": "preconditions", "duration_s": time.monotonic() - phase}]
    progress(started, "preconditions", "complete", len(checks))
    if any(not row["passed"] for row in checks):
        result = _build_result(checks, None, [], spans, raw, False, started)
        atomic_json(OUTPUT, result)
        progress(started, "publish", "blocked")
        return result
    manifest_path = Path(scope["development_manifest_path"])
    progress(started, "prepare", "begin")
    phase = time.monotonic()
    prepared = views.prepare_corpus(manifest_path, raw, corpus.COUNTS)
    spans.append({"phase": "prepare", "duration_s": time.monotonic() - phase})
    progress(started, "prepare", "complete", len(prepared["rows"]))
    atomic_json(candidate, _build_result(checks, prepared, [], spans, raw, False, started))
    check_candidate_roster(manifest_path, raw, candidate)
    progress(started, "validation", "begin")
    phase = time.monotonic()
    receipts = dispatch(scope)
    spans.append({"phase": "validation", "duration_s": time.monotonic() - phase})
    progress(started, "validation", "complete", len(receipts))
    adverse = next(row for row in receipts if row["name"] == "adversarial_verify")
    try:
        flagged = json.loads(Path(adverse["log_path"]).read_text())["flagged_count"] > 0
    except (OSError, ValueError, KeyError):
        flagged = True
    result = _build_result(checks, prepared, receipts, spans, raw, flagged, started)
    if any(row["classification"] == "required" and not row["passed"] for row in receipts):
        result["sentence_protocol_ready_score"] = 0
        result["evidence_view_ready_score"] = 0
        result["source_view_manifest_path"] = None
    atomic_json(OUTPUT, result)
    progress(started, "publish", "complete", len(result["rows"]))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run a dated producer or cold-read explicit sealed input paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", nargs=3, metavar=("MANIFEST", "RAW", "CANDIDATE"))
    args = parser.parse_args(argv)
    if args.cold_replay:
        if args.date != "20260928":
            raise ValueError("run_date_mismatch")
        result = cold_reduce(*(Path(item) for item in args.cold_replay))
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    return int(run_experiment(args.date)["verdict_class"] == "disqualified")
