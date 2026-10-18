"""CPU-only qualification for the future source-sufficiency capture.

REQ-REPORT-7854-V682. Scripted transport checks bytes and parser outcomes;
its agreement cannot measure source sufficiency or scientific benefit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import context_sufficiency_7854 as context
from carnot.verify import source_interventions as source

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7854_v682_intervention_protocol"
OUTPUT = ROOT / "results" / f"{NAME}.json"
RAW = ROOT / "results/raw" / NAME
PRIVATE = Path("/tmp/carnot-7854-v682-20260929")
SEED = 68201
PLAN_SHA256 = "sha256:cad89d908e9d18c8b721fe53f44d870f87309c9d9f5a47aac0e6ee18fd0190ae"
MODEL_SPECS: list[dict[str, Any]] = []
PRIOR_ATTEMPT = RAW / "attempts/attempt_1_disqualified_adversarial.json"
Json = dict[str, Any]


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Show actual completed units before another bounded operation starts."""
    print(
        f"[exp7854] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def fixture_family(index: int) -> Json:
    """Give each conformance family distinct bytes and no semantic label."""
    if index % 12 == 4:
        text = ""
    elif index % 12 == 6:
        text = f"Only {index} one. Only {index} two. Only {index} three."
    elif index % 12 == 8:
        text = "Repeated. Repeated. Middle. Repeated. Repeated."
    else:
        text = (
            f"Café {index} first. Second {index} fact. Third {index} fact. "
            f"Fourth {index} fact. Fifth {index} fact."
        )
    answer = f"Café {index} first. Extra sentence not sent."
    return {
        "family_id": f"fixture-{index:02d}",
        "complete_source": text,
        "complete_response": answer,
        "source_sha256": source.digest(text.encode()),
        "response_sha256": source.digest(answer.encode()),
        "annotations": "unknown" if index % 12 == 9 else None,
    }


def fixture_reply(payload: Json, index: int) -> Json:
    """Inject one indexed transport failure without loading any weights."""
    case = index % 12
    if case == 3:
        raise TimeoutError
    if case == 10:
        raise InterruptedError
    body = json.loads(payload["messages"][1]["content"])
    witness = 99 if case == 5 else (1 if case == 6 else 2)
    selected = witness if body["arm"] == "full_source" else body["visible_source_sentence_ids"][0]
    content = json.dumps({"unsupported_probability": 0.2, "source_sentence_id": selected})
    return {
        "model": payload["model"],
        "choices": [
            {
                "message": {"content": "{" if case == 1 else content},
                "finish_reason": "length" if case == 2 else "stop",
            }
        ],
        "usage": {"completion_tokens": 7},
    }


def fixture_e2e(path: Path, checkpoints: Path) -> Json:
    """Save each finished family under its input and configuration identity."""
    started = time.monotonic()
    progress(started, "fixture", "begin")
    frozen = context.freeze_protocol(seed=SEED)
    rows: list[Json] = []
    families: list[Json] = []
    hits = 0
    for index in range(24):
        row = fixture_family(index)
        families.append(row)
        identity = canonical_hash({"row": row, "protocol": frozen})[7:]
        checkpoint = checkpoints / f"{identity}.json"
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text())
            if saved["identity"] != identity:
                raise ValueError("checkpoint_drift")
            family_rows = saved["rows"]
            hits += 1
        else:
            family_rows = context.capture_family(
                row,
                frozen,
                lambda payload, i=index: fixture_reply(payload, i),
                lambda text: len(text.split()),
            )
            atomic_json(checkpoint, {"identity": identity, "rows": family_rows})
        rows.extend(family_rows)
        if index % 6 == 5:
            progress(started, "fixture", "batch", index + 1)
    result = {
        "schema": "carnot.exp7854.fixture.v1",
        "independent_families": 24,
        "families": families,
        "rows": rows,
        "checkpoint_hits": hits,
        "model_calls": 0,
        "model_loads": 0,
    }
    atomic_json(path, result)
    progress(started, "fixture", "complete", 24)
    return result


def cold_replay(path: Path) -> Json:
    """Rebuild every started request and parse each stored response again."""
    data = json.loads(path.read_text())
    frozen = context.freeze_protocol(seed=SEED)
    grouped = {row["family_id"]: row for row in data["families"]}
    if len(grouped) != 24 or len(data["rows"]) != 96:
        raise ValueError("fixture_count_drift")
    for family_id, row in grouped.items():
        if source.digest(row["complete_source"].encode()) != row["source_sha256"]:
            raise ValueError("source_drift")
        if source.digest(row["complete_response"].encode()) != row["response_sha256"]:
            raise ValueError("answer_drift")
    for item in data["rows"]:
        request = item["request_bytes"]
        if request is None:
            continue
        if source.digest(request.encode()) != item["request_sha256"]:
            raise ValueError("request_drift")
        payload = json.loads(request)
        body = json.loads(payload["messages"][1]["content"])
        family_rows = [r for r in data["rows"] if r["family_id"] == item["family_id"]]
        witness = family_rows[0]["source_sentence_id"]
        rebuilt = context.build_request(
            grouped[item["family_id"]],
            item["arm"],
            witness,
            item["control_sentence_id"],
            frozen,
            lambda text: len(text.split()),
        )
        exact = json.dumps(rebuilt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if exact != request:
            raise ValueError("request_drift")
        response_bytes = item["response_bytes"]
        if response_bytes is not None:
            if source.digest(response_bytes.encode()) != item["response_sha256"]:
                raise ValueError("response_drift")
            parsed = context.parse_response(
                json.loads(response_bytes), payload["model"], body["visible_source_sentence_ids"]
            )
            if parsed["status"] != item["status"]:
                raise ValueError("parser_drift")
    return {"families": len(grouped), "rows": len(data["rows"])}


EXTERNAL = {
    "producer": ROOT / "results/experiment_7727_v673_development_corpus.json",
    "manifest": ROOT
    / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json",
    "public": ROOT / "results/raw/experiment_7727_v673_development_corpus/evaluation_public.jsonl",
    "historical": ROOT / "results/experiment_7839_v681_intervention_protocol.json",
}


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Record both sides of a gate so absence differs from a wrong value."""
    return {
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight() -> tuple[list[Json], list[Json], Json]:
    """Authenticate public source custody before making any fixture requests."""
    checks = [
        operand(name, path, "exists", True, path.is_file()) for name, path in EXTERNAL.items()
    ]
    hashes = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": "20260929",
            "role": name,
            "exposure_status": "historical" if name == "historical" else "exposed_development",
        }
        for name, path in EXTERNAL.items()
    }
    if any(not item["passed"] for item in checks):
        return [], checks, hashes
    producer = json.loads(EXTERNAL["producer"].read_text())
    manifest = json.loads(EXTERNAL["manifest"].read_text())
    old = json.loads(EXTERNAL["historical"].read_text())
    role = manifest.get("roles", {}).get("evaluation", {})
    observed = (
        (
            "producer",
            "development_cohort_ready_score",
            1,
            producer.get("development_cohort_ready_score"),
        ),
        ("producer", "flagged_adversarial", False, producer.get("flagged_adversarial")),
        ("manifest", "roles.evaluation.count", 64, role.get("count")),
        (
            "manifest",
            "roles.evaluation.public_sha256",
            sha256_file(EXTERNAL["public"]),
            role.get("public_sha256"),
        ),
        ("historical", "verdict_class", "disqualified", old.get("verdict_class")),
        (
            "historical",
            "intervention_protocol_ready_score",
            0,
            old.get("intervention_protocol_ready_score"),
        ),
        (
            "historical",
            "changed_coverage.passed",
            False,
            next(
                (
                    r.get("passed")
                    for r in old.get("validation_receipts", [])
                    if r.get("name") == "changed_coverage"
                ),
                None,
            ),
        ),
    )
    checks.extend(
        operand(name, EXTERNAL[name], field, expected, value)
        for name, field, expected, value in observed
    )
    if any(not item["passed"] for item in checks):
        return [], checks, hashes
    rows = [json.loads(line) for line in EXTERNAL["public"].read_text().splitlines()]
    checks.append(operand("public", EXTERNAL["public"], "row_count", 64, len(rows)))
    checks.append(
        operand(
            "public",
            EXTERNAL["public"],
            "family_ids",
            sorted(role["families"]),
            sorted(row["family_id"] for row in rows),
        )
    )
    for row in rows:
        checks.append(
            operand(
                "public",
                EXTERNAL["public"],
                row["family_id"] + ".source_sha256",
                source.digest(row["complete_source"].encode()),
                row["source_sha256"],
            )
        )
        checks.append(
            operand(
                "public",
                EXTERNAL["public"],
                row["family_id"] + ".response_sha256",
                source.digest(row["complete_response"].encode()),
                row["response_sha256"],
            )
        )
    return rows if all(item["passed"] for item in checks) else [], checks, hashes


def prepare_sources(public: list[Json], path: Path) -> tuple[list[Json], Json]:
    """Freeze 48 exposed families without inspecting evaluator annotations."""
    selected = sorted(
        public, key=lambda row: source.digest(f"{SEED}:{row['source_sha256']}".encode())
    )[:48]
    if len(selected) != 48 or len({row["source_sha256"] for row in selected}) != 48:
        raise ValueError("family_count_drift")
    frozen = context.freeze_protocol(seed=SEED)
    frozen["families"] = [
        {
            "family_id": row["family_id"],
            "source_sha256": row["source_sha256"],
            "answer_sha256": row["response_sha256"],
        }
        for row in selected
    ]
    atomic_json(path, frozen)
    rows = [
        {
            "family_id": row["family_id"],
            "source_family": row["family_id"],
            "arm": arm,
            "seed": SEED,
            "source_sha256": row["source_sha256"],
            "answer_sha256": row["response_sha256"],
            "status": "unstarted_no_model_execution",
            "started": False,
            "completed": False,
            "censored": False,
            "excluded": False,
            "independent": arm == "full_source",
            "probability": None,
            "original_label": None,
        }
        for row in selected
        for arm in context.ARMS
    ]
    return rows, frozen


def reduce_candidate(path: Path) -> Json:
    """Recompute the planned natural rows and the independent fixture replay."""
    artifact = json.loads(path.read_text())
    if artifact["experiment_id"] != 7854 or artifact["task_id"] != "exp7854-intervention-protocol":
        raise ValueError("wrong_result_owner")
    public, checks, _ = preflight()
    if any(not check["passed"] for check in checks):
        raise ValueError("source_drift")
    selected = sorted(
        public, key=lambda row: source.digest(f"{SEED}:{row['source_sha256']}".encode())
    )[:48]
    expected = [(r["family_id"], arm, r["source_sha256"]) for r in selected for arm in context.ARMS]
    actual = [(r["family_id"], r["arm"], r["source_sha256"]) for r in artifact["rows"]]
    if expected != actual:
        raise ValueError("row_drift")
    fixture_path = Path(artifact["fixture_rows_path"])
    if sha256_file(fixture_path) != artifact["fixture_rows_sha256"]:
        raise ValueError("fixture_drift")
    cold_replay(fixture_path)
    return {"families": 48, "rows": len(actual), "fixture_rows": 96}


def seal_receipt(receipt: Json, index: int) -> Json:
    """Copy a closed child log to an immutable content-addressed private path."""
    original = ROOT / receipt["log_path"]
    digest = sha256_file(original)
    sealed = PRIVATE / "sealed_logs" / f"{index:02d}_{receipt['name']}_{digest[7:]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.exists():
        if sha256_file(sealed) != digest:
            raise ValueError("sealed_log_drift")
    else:
        shutil.copyfile(original, sealed)
    return {**receipt, "log_path": str(sealed), "log_sha256": digest}


def validate(plan: Json, started: float) -> list[Json]:
    """Run only frozen bounded commands and retain each real child exit."""
    receipts = []
    for index, command in enumerate(plan["commands"]):
        progress(started, command["name"], "before_subprocess", index)
        for argument in command["argv"]:
            if argument.startswith(("--basetemp=", "--data-file=")):
                Path(argument.partition("=")[2]).parent.mkdir(parents=True, exist_ok=True)
        spec = CommandSpec(
            command["name"],
            tuple(command["argv"]),
            command["classification"],
            command["timeout_s"],
        )
        receipt = run_commands(
            ROOT,
            [spec],
            log_dir=PRIVATE / "child_logs" / command["name"],
            extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )[0]
        if receipt["command_argv"] != command["argv"]:
            raise ValueError("child_command_drift")
        receipts.append(
            {**seal_receipt(receipt, index), "classification": command["classification"]}
        )
        progress(started, command["name"], "after_subprocess", index + 1)
    return receipts


def build_artifact(
    checks: list[Json],
    hashes: Json,
    rows: list[Json],
    fixture_path: Path | None,
    protocol_path: Path | None,
    receipts: list[Json],
    started: float,
    spans: list[Json],
    flagged: bool,
) -> Json:
    """Keep protocol qualification separate from unmeasured model benefit."""
    failed_external = [check for check in checks if not check["passed"]]
    required = [r for r in receipts if r["classification"] == "required"]
    required_names = set(
        json.loads((RAW / "validation_command_manifest.json").read_text())["required_checks"]
    )
    failed_required = [r for r in required if not r["passed"]]
    missing_required = required_names - {r["name"] for r in required}
    blocked = bool(failed_external)
    disqualified = not blocked and (bool(failed_required) or bool(missing_required) or flagged)
    ready = not blocked and not disqualified and bool(receipts)
    verdict_class = (
        "blocked"
        if blocked
        else "disqualified"
        if disqualified
        else "circular_positive"
        if ready
        else "partial"
    )
    verdict = {
        "blocked": "complete_blocked_required_source",
        "disqualified": "complete_disqualified_required_checks",
        "circular_positive": "complete_circular_positive_protocol_qualification",
        "partial": "partial_pending_owned_validation",
    }[verdict_class]
    failures = failed_external + [
        operand("current_validation", Path(r["log_path"]), r["name"] + ".passed", True, False)
        for r in failed_required
    ]
    budget = {
        "intended": 192,
        "eligible": len(rows),
        "started": sum(bool(r["started"]) for r in rows),
        "completed": sum(bool(r["completed"]) for r in rows),
        "censored": sum(bool(r["censored"]) for r in rows),
        "excluded": sum(bool(r["excluded"]) for r in rows),
        "independent": len({r["family_id"] for r in rows}),
    }
    code = {
        str(ROOT / path): sha256_file(ROOT / path)
        for path in (
            "python/carnot/verify/context_sufficiency_7854.py",
            "python/carnot/experiment_7854_v682_intervention_protocol.py",
            "scripts/experiments/experiment_7854_v682_intervention_protocol.py",
        )
    }
    result = {
        "schema": "carnot.exp7854.intervention_result.v1",
        "experiment_id": 7854,
        "task_id": "exp7854-intervention-protocol",
        "milestone": "2026.09.682",
        "run_date": "20260929",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "gate_check_summary": failures,
        "rows": rows,
        "sample_size_budget": budget,
        "random_seed": SEED,
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "reproducibility_checksum": canonical_hash(
            {
                "code": code,
                "inputs": hashes,
                "seed": SEED,
                "manifest": sha256_file(RAW / "validation_command_manifest.json"),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "validation_command_manifest_path": str(RAW / "validation_command_manifest.json"),
        "observed_child_commands": [
            {"name": r["name"], "argv": r["command_argv"], "classification": r["classification"]}
            for r in receipts
        ],
        "repository_health": {
            "historical_exp7839_coverage_percent": 60,
            "historical_exp7839_orchestration_coverage_percent": 23,
            "historical_required_coverage_passed": False,
            "prior_required_failure": {
                "path": str(PRIOR_ATTEMPT),
                "sha256": sha256_file(PRIOR_ATTEMPT),
                "failed_checks": [
                    r["name"]
                    for r in json.loads(PRIOR_ATTEMPT.read_text())["validation_receipts"]
                    if r["classification"] == "required" and not r["passed"]
                ],
            }
            if PRIOR_ATTEMPT.is_file()
            else None,
            "current_diagnostic": next(
                (r for r in receipts if r["name"] == "repository_health_180s"), None
            ),
        },
        "acceptance_gate_results": {
            "validity": ready,
            "readiness": ready,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "verifier_is_oracle": True,
        "claim_scope": "scripted fixture conformance; exposed development sources; no measured sufficiency",
        "inference_substrate": "cpu_no_pretrained_model",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no model loaded or invoked)",
        "model_invocation_counts": {
            "loads": 0,
            "calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "model_file_hashes": [],
        },
        "intervention_protocol_ready_score": int(ready),
        "intervention_protocol_path": str(protocol_path) if protocol_path else None,
        "fixture_rows_path": str(fixture_path) if fixture_path else None,
        "fixture_rows_sha256": sha256_file(fixture_path) if fixture_path else None,
        "fixture_request_rows": [],
        "coverage_shard_rows": [],
    }
    result["field_principles"] = {
        key: "Retain exact scope, custody and failure for replay." for key in result
    }
    result["field_principles"].update(
        {
            key: "An unmeasured model outcome stays null."
            for key in result["acceptance_gate_results"]
        }
    )
    return result


def run_experiment(date: str) -> Json:
    """Own preflight, fixture, bounded validation and one terminal publication."""
    started = time.monotonic()
    progress(started, "start", "begin")
    if date != "20260929":
        raise ValueError("run_date_mismatch")
    plan_path = RAW / "validation_command_manifest.json"
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("validation_manifest_drift")
    plan = json.loads(plan_path.read_text())
    if plan["schema"] != "carnot.exp7854.validation_manifest.v1":
        raise ValueError("validation_manifest_drift")
    phase = time.monotonic()
    progress(started, "preconditions", "begin")
    public, checks, hashes = preflight()
    spans = [
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    ]
    progress(started, "preconditions", "complete", len(checks))
    if any(not item["passed"] for item in checks):
        blocked = build_artifact(checks, hashes, [], None, None, [], started, spans, False)
        atomic_json(OUTPUT, blocked)
        progress(started, "publish", "blocked")
        return blocked
    phase = time.monotonic()
    progress(started, "prepare", "begin")
    protocol_path = RAW / "context_sufficiency_protocol.json"
    rows, _ = prepare_sources(public, protocol_path)
    fixture_path = PRIVATE / "fixture.json"
    fixture = fixture_e2e(fixture_path, PRIVATE / "checkpoints")
    cold_replay(fixture_path)
    spans.append(
        {
            "phase": "prepare",
            "duration_s": time.monotonic() - phase,
            "completed_units": 48 + fixture["independent_families"],
        }
    )
    progress(started, "prepare", "complete", 72)
    pending = build_artifact(
        checks, hashes, rows, fixture_path, protocol_path, [], started, spans, False
    )
    pending.update(
        honest_verdict="partial_pending_owned_validation",
        verdict_class="partial",
        intervention_protocol_ready_score=0,
    )
    pending["acceptance_gate_results"].update(validity=False, readiness=False)
    pending["fixture_request_rows"] = fixture["rows"]
    candidate_path = PRIVATE / "pending_candidate.json"
    atomic_json(candidate_path, pending)
    phase = time.monotonic()
    progress(started, "validation", "begin")
    receipts = validate(plan, started)
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    adverse = next((r for r in receipts if r["name"] == "adversarial_verify"), None)
    try:
        flagged = bool(json.loads(Path(adverse["log_path"]).read_text())["flagged_count"])
    except (KeyError, TypeError, ValueError, OSError):
        flagged = True
    terminal = build_artifact(
        checks, hashes, rows, fixture_path, protocol_path, receipts, started, spans, flagged
    )
    terminal["fixture_request_rows"] = fixture["rows"]
    terminal["coverage_shard_rows"] = [
        {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "completed": path.is_file(),
        }
        for path in (Path(item) for item in plan["coverage_files"])
    ]
    atomic_json(OUTPUT, terminal)
    progress(started, "publish", "complete", len(rows))
    return terminal


def main(argv: list[str] | None = None) -> int:
    """Expose a nonrecursive real CLI for fixtures and cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260929":
        raise ValueError("run_date_mismatch")
    if args.fixture_e2e:
        result = fixture_e2e(args.fixture_e2e, args.fixture_e2e.parent / "checkpoints")
        print(json.dumps({"independent_families": result["independent_families"]}), flush=True)
        return 0
    if args.cold_replay:
        target = json.loads(args.cold_replay.read_text())
        result = (
            cold_replay(args.cold_replay)
            if target["schema"].endswith("fixture.v1")
            else reduce_candidate(args.cold_replay)
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    return int(run_experiment(args.date)["verdict_class"] == "disqualified")
