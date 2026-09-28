"""Requalify the original counter-evidence protocol with explicit coverage files.

REQ-REPORT-7828. A source citation is a proposed witness for one unchanged
answer sentence. This module records protocol custody without model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

from carnot import experiment_7814_v679_counter_evidence_protocol as prior
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7828_v680_counter_evidence_protocol"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
COMMAND_MANIFEST = RAW / "plans/attempt-7895ca1a92354884b2fddc5163c86764.json"
COMMAND_MANIFEST_SHA256 = "sha256:b08bd1836c1fcf402875e2aded10c41ef3bc17cf6fb2d22735977e8fa7f52781"
SEED = 68001
HISTORICAL_SHARD_SHA256 = "sha256:e528b4040e5ac4e427e6f121af5def00931ddf596d0495c68408a0f1790003ef"
digest = prior.digest
validate_log_receipt = prior.validate_log_receipt


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Show elapsed time at each boundary so stalled child work stays visible."""
    print(
        f"[exp7828] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def load_plan() -> dict[str, Any]:
    """Authenticate the command bytes before any child can be dispatched."""
    if sha256_file(COMMAND_MANIFEST) != COMMAND_MANIFEST_SHA256:
        raise ValueError("manifest_drift")
    return json.loads(COMMAND_MANIFEST.read_text())


def shard_checks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind each old completed shard to its own successful child exit."""
    old_path = ROOT / "results/experiment_7814_v679_counter_evidence_protocol.json"
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    receipts = {row["name"]: row for row in old.get("validation_receipts", [])}
    checks = []
    for path, name in zip(
        plan["historical_coverage_files"], ("coverage_protocol", "coverage_cli"), strict=True
    ):
        file = Path(path)
        checks.append(prior.prior.check("exp7814_coverage", file, "exists", True, file.is_file()))
        checks.append(
            prior.prior.check(
                "exp7814_coverage",
                file,
                "sha256",
                HISTORICAL_SHARD_SHA256,
                sha256_file(file) if file.is_file() else None,
            )
        )
        checks.append(
            prior.prior.check(
                "exp7814_coverage",
                old_path,
                name + ".exit_code",
                0,
                receipts.get(name, {}).get("exit_code"),
            )
        )
    return checks


def select_control(
    source: bytes,
    offsets: list[dict[str, Any]],
    witness: int,
    family_id: str,
    token_count: Callable[[str], int],
) -> int | None:
    """Use only original source bytes and the new seed to choose a control."""
    sizes = {
        item["source_sentence_id"]: token_count(
            source[item["start_byte"] : item["end_byte"]].decode()
        )
        for item in offsets
    }
    target = sizes.get(witness, 0)
    options = [
        item
        for item, size in sizes.items()
        if item != witness and target > 0 and 4 * abs(size - target) <= target
    ]
    return (
        min(options, key=lambda item: digest(f"{SEED}:{family_id}:{item}".encode()))
        if options
        else None
    )


def freeze_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sample source hashes before evaluator labels enter the process."""
    if len(rows) != 64 or len({row["source_sha256"] for row in rows}) != 64:
        raise ValueError("evaluation64_invalid")
    return sorted(rows, key=lambda row: digest(f"{SEED}:{row['source_sha256']}".encode()))[:48]


def dispatch(
    plan: dict[str, Any],
    executor: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run exactly the frozen argv and classes, then verify sealed log bytes."""
    if plan != load_plan():
        raise ValueError("manifest_drift")
    started = time.monotonic()
    (Path(plan["private_root"]) / "coverage").mkdir(parents=True, exist_ok=True)
    chosen = executor or prior.execute_child
    receipts = []
    for index, command in enumerate(plan["commands"]):
        progress(started, command["name"], "before_child", index)
        receipt = chosen(command, index, plan)
        if (receipt["name"], receipt["command_argv"], receipt["classification"]) != (
            command["name"],
            command["argv"],
            command["classification"],
        ):
            raise ValueError("observed_child_command_drift")
        validate_log_receipt(receipt)
        receipts.append(receipt)
        progress(started, command["name"], "after_child", len(receipts))
    return receipts


def build_result(
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    selected: list[dict[str, Any]],
    protocol_path: Path | None,
    family_path: Path | None,
    fixture_rows: list[dict[str, Any]],
    aligned: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    started: float,
    flagged: bool = False,
) -> dict[str, Any]:
    """Keep the historical result while giving this result its own identity."""
    result = prior.artifact(
        checks,
        hashes,
        selected,
        protocol_path,
        family_path,
        fixture_rows,
        aligned,
        receipts,
        spans,
        started,
        flagged,
    )
    result["gate_check_summary"] = [
        check for check in result["gate_check_summary"] if not check["passed"]
    ]
    result["gate_check_summary"].extend(
        prior.prior.check(
            "exp7828_validation",
            Path(receipt["log_path"]),
            receipt["name"] + ".passed",
            True,
            receipt["passed"],
        )
        for receipt in receipts
        if receipt["classification"] == "required" and not receipt["passed"]
    )
    result.update(
        schema="carnot.exp7828.counter_evidence_protocol_result.v1",
        experiment_id="exp7828-counter-evidence-protocol",
        milestone="2026.09.680",
        run_date="20260928",
        random_seed=SEED,
        validation_command_manifest_path=str(COMMAND_MANIFEST),
        validation_command_manifest_sha256=sha256_file(COMMAND_MANIFEST),
    )
    for row in result["rows"]:
        row["seed"] = SEED
    result["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(
            {
                "code": sha256_file(Path(__file__)),
                "wrapper": sha256_file(ROOT / "scripts/experiments" / f"{NAME}.py"),
                "inputs": hashes,
                "protocol": sha256_file(protocol_path) if protocol_path else None,
                "family_manifest": sha256_file(family_path) if family_path else None,
                "command_manifest": sha256_file(COMMAND_MANIFEST),
                "seed": SEED,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    result["field_principles"]["validation_command_manifest_path"] = (
        "Freeze the current exact child roster."
    )
    result["field_principles"]["repository_health"] = "Keep old and current broad failures visible."
    result["repository_health"]["historical_exp7814_verdict"] = (
        "complete_disqualified_required_validation"
    )
    result["repository_health"]["historical_exp7814_coverage_combine_exit"] = 1
    return result


def cold_reduce(path: Path) -> dict[str, Any]:
    """Reopen current bytes and check the unchanged source and answer spans."""
    result = prior.cold_reduce(path)
    candidate = json.loads(path.read_text())
    if candidate["experiment_id"] != "exp7828-counter-evidence-protocol":
        raise ValueError("wrong_result_owner")
    for receipt in candidate["validation_receipts"]:
        validate_log_receipt(receipt)
    return result


def run_experiment(
    date: str,
    child_executor: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare a 48-family plan, then qualify it through the frozen children."""
    started = time.monotonic()
    progress(started, "start", "begin")
    if date != "20260928":
        raise ValueError("run_date_mismatch")
    plan = load_plan()
    attempt = Path(plan["raw_root"])
    if attempt.exists():
        raise ValueError("attempt_root_reused")
    attempt.mkdir(parents=True)
    phase = time.monotonic()
    progress(started, "preconditions", "begin")
    rows, checks, hashes = prior.preflight(ROOT)
    checks.extend(shard_checks(plan))
    for index, file in enumerate(plan["historical_coverage_files"]):
        path = Path(file)
        hashes[f"historical_coverage_{index}"] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": "20260928",
            "imported_fields": ["completed_coverage_data"],
            "eligible": path.is_file(),
        }
    old_path = ROOT / "results/experiment_7814_v679_counter_evidence_protocol.json"
    hashes["historical_exp7814"] = {
        "path": str(old_path),
        "sha256": sha256_file(old_path) if old_path.is_file() else None,
        "date": "20260928",
        "imported_fields": ["honest_verdict", "validation_receipts"],
        "eligible": False,
    }
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    checks.append(
        prior.prior.check(
            "exp7814_historical",
            old_path,
            "honest_verdict",
            "complete_disqualified_required_validation",
            old.get("honest_verdict"),
        )
    )
    progress(started, "preconditions", "before_tokenizer_hash", len(checks))
    checks.extend(prior.resource_checks())
    progress(started, "preconditions", "after_tokenizer_hash", len(checks))
    spans = [
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    ]
    if any(not check["passed"] for check in checks):
        result = build_result(checks, hashes, [], None, None, [], [], [], spans, started)
        atomic_json(OUTPUT, result)
        progress(started, "publish", "blocked")
        return result
    progress(started, "prepare", "begin")
    phase = time.monotonic()
    selected = freeze_families(rows)
    progress(started, "prepare", "before_vocab_load", len(selected))
    counter = prior.GGUFTokenCounter(prior.GGUF_PATH, prior.GGUF_SHA256)
    progress(started, "prepare", "after_vocab_load", len(selected))
    frozen = prior.make_protocol(selected)
    frozen["schema"] = "carnot.exp7828.counter_evidence_protocol.v1"
    frozen["seed"] = SEED
    frozen["request"]["seed"] = SEED
    frozen["matching_rule"] = (
        "disjoint original source sentence; GGUF token difference <=25 percent; SHA-256 order with seed 68001"
    )
    frozen["bootstrap"]["seed"] = SEED
    frozen["tokenizer"] = {
        "gguf_path": str(prior.GGUF_PATH),
        "gguf_sha256": prior.GGUF_SHA256,
        "chat_template_sha256": counter.template_sha256,
        "vocabulary_only": True,
    }
    protocol_path = attempt / "counter_evidence_protocol.json"
    atomic_json(protocol_path, frozen)
    family = prior.build_family_manifest(selected, counter)
    family["schema"] = "carnot.exp7828.family_manifest.v1"
    family["seed"] = SEED
    family_path = attempt / "family_manifest.json"
    atomic_json(family_path, family)
    evaluator_path = ROOT / prior.prior.MANIFEST.parent / "evaluation_evaluator.jsonl"
    evaluator = {
        item["family_id"]: item
        for item in (json.loads(line) for line in evaluator_path.read_text().splitlines())
    }
    aligned = [
        {
            "family_id": row["family_id"],
            "label": prior.aligned_label(row, evaluator[row["family_id"]]),
            "target_sentence_span": prior.target_span(row["complete_response"].encode()),
            "annotation_source_sha256": sha256_file(evaluator_path),
        }
        for row in selected
    ]
    fixture = prior.fixture_e2e(attempt / "fixture_e2e_preflight.json")
    spans.append(
        {
            "phase": "prepare",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(selected),
        }
    )
    progress(started, "prepare", "complete", len(selected))
    candidate = build_result(
        checks,
        hashes,
        selected,
        protocol_path,
        family_path,
        fixture["rows"],
        aligned,
        [],
        spans,
        started,
    )
    for item in candidate["rows"]:
        if item["arm"] == "intact":
            row = next(row for row in selected if row["family_id"] == item["family_id"])
            try:
                prior.make_request(
                    row,
                    row["complete_source"].encode(),
                    prior.sentence_offsets(row["complete_source"].encode()),
                    "intact",
                    frozen,
                    counter,
                )
            except ValueError as error:
                item["disposition"] = "unstarted_" + str(error)
    atomic_json(Path(plan["candidate_path"]), candidate)
    phase = time.monotonic()
    progress(started, "validation", "begin")
    receipts = dispatch(plan, child_executor)
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    progress(started, "validation", "complete", len(receipts))
    adverse = next(row for row in receipts if row["name"] == "adversarial_verify")
    try:
        flagged = json.loads(Path(adverse["log_path"]).read_text())["flagged_count"] > 0
    except (OSError, ValueError, KeyError):
        flagged = True
    result = build_result(
        checks,
        hashes,
        selected,
        protocol_path,
        family_path,
        fixture["rows"],
        aligned,
        receipts,
        spans,
        started,
        flagged,
    )
    for before, after in zip(candidate["rows"], result["rows"], strict=True):
        after["disposition"] = before["disposition"]
    atomic_json(OUTPUT, result)
    progress(started, "publish", "complete", len(result["rows"]))
    return result


def main(
    argv: list[str] | None = None,
    *,
    child_executor: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]] | None = None,
) -> int:
    """Expose the dated run and current CPU and cold-reader checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--dispatch-check", action="store_true")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260928":
        raise ValueError("run_date_mismatch")
    if args.dispatch_check:
        dispatch(load_plan(), child_executor)
        return 0
    if args.fixture_e2e:
        print(
            json.dumps(prior.fixture_e2e(args.fixture_e2e)["reduced"], sort_keys=True), flush=True
        )
        return 0
    if args.cold_replay:
        print(json.dumps(cold_reduce(args.cold_replay), sort_keys=True), flush=True)
        return 0
    result = run_experiment(args.date, child_executor)
    return int(result["verdict_class"] == "disqualified")
