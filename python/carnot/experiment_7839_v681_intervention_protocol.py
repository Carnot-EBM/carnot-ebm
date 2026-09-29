"""CPU qualification for reusable sentence interventions and exact receipts.

REQ-REPORT-7839. Scripted replies test request plumbing; they cannot measure
whether an independent model finds unsupported answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import source_interventions as source

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7839_v681_intervention_protocol"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
PLAN_PATH = RAW / "validation_command_manifest.json"
PLAN_SHA256 = "sha256:de8dde37f4e66f5d87be511a329f5f48f1f4fbc1d7c64147465914141571fe98"
SEED = 68101
MODEL_SPECS: list[dict[str, Any]] = []
EXTERNAL = {
    "producer": ROOT / "results/experiment_7727_v673_development_corpus.json",
    "manifest": ROOT
    / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json",
    "public": ROOT / "results/raw/experiment_7727_v673_development_corpus/evaluation_public.jsonl",
    "evaluator": ROOT
    / "results/raw/experiment_7727_v673_development_corpus/evaluation_evaluator.jsonl",
    "source_view": ROOT / "results/experiment_7810_v679_source_view_qualification.json",
    "historical": ROOT / "results/experiment_7828_v680_counter_evidence_protocol.json",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Expose completed work before and after each bounded phase."""
    print(
        f"[exp7839] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def load_plan() -> dict[str, Any]:
    """Reject changed validation argv before a child can run."""
    if sha256_file(PLAN_PATH) != PLAN_SHA256:
        raise ValueError("validation_manifest_drift")
    return json.loads(PLAN_PATH.read_text())


def check(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep both gate operands and the exact supplying file."""
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


def preflight() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate public source and hidden labels before any fixture work."""
    checks = [check(key, path, "exists", True, path.is_file()) for key, path in EXTERNAL.items()]
    hashes = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": "20260928",
            "role": key,
            "eligible": key != "historical" and path.is_file(),
        }
        for key, path in EXTERNAL.items()
    }
    if any(not item["passed"] for item in checks):
        return [], checks, hashes
    producer = json.loads(EXTERNAL["producer"].read_text())
    manifest = json.loads(EXTERNAL["manifest"].read_text())
    source_view = json.loads(EXTERNAL["source_view"].read_text())
    historical = json.loads(EXTERNAL["historical"].read_text())
    role = manifest.get("roles", {}).get("evaluation", {})
    for upstream, path, field, expected, observed in (
        (
            "producer",
            EXTERNAL["producer"],
            "development_cohort_ready_score",
            1,
            producer.get("development_cohort_ready_score"),
        ),
        (
            "producer",
            EXTERNAL["producer"],
            "flagged_adversarial",
            False,
            producer.get("flagged_adversarial"),
        ),
        (
            "source_view",
            EXTERNAL["source_view"],
            "evidence_view_ready_score",
            1,
            source_view.get("evidence_view_ready_score"),
        ),
        (
            "source_view",
            EXTERNAL["source_view"],
            "flagged_adversarial",
            False,
            source_view.get("flagged_adversarial"),
        ),
        ("manifest", EXTERNAL["manifest"], "roles.evaluation.count", 64, role.get("count")),
        (
            "manifest",
            EXTERNAL["manifest"],
            "roles.evaluation.public_sha256",
            sha256_file(EXTERNAL["public"]),
            role.get("public_sha256"),
        ),
        (
            "manifest",
            EXTERNAL["manifest"],
            "roles.evaluation.evaluator_sha256",
            sha256_file(EXTERNAL["evaluator"]),
            role.get("evaluator_sha256"),
        ),
        (
            "historical",
            EXTERNAL["historical"],
            "verdict_class",
            "disqualified",
            historical.get("verdict_class"),
        ),
        (
            "historical",
            EXTERNAL["historical"],
            "worktree_imports.passed",
            False,
            next(
                (
                    r.get("passed")
                    for r in historical.get("validation_receipts", [])
                    if r.get("name") == "worktree_imports"
                ),
                None,
            ),
        ),
    ):
        checks.append(check(upstream, path, field, expected, observed))
    if any(not item["passed"] for item in checks):
        return [], checks, hashes
    rows = [json.loads(line) for line in EXTERNAL["public"].read_text().splitlines()]
    checks.append(check("public", EXTERNAL["public"], "row_count", 64, len(rows)))
    checks.append(
        check(
            "public",
            EXTERNAL["public"],
            "family_ids",
            sorted(role["families"]),
            sorted(r["family_id"] for r in rows),
        )
    )
    for row in rows:
        checks.append(
            check(
                "public",
                EXTERNAL["public"],
                row["family_id"] + ".source_sha256",
                source.digest(row["complete_source"].encode()),
                row["source_sha256"],
            )
        )
        checks.append(
            check(
                "public",
                EXTERNAL["public"],
                row["family_id"] + ".response_sha256",
                source.digest(row["complete_response"].encode()),
                row["response_sha256"],
            )
        )
    return rows if all(item["passed"] for item in checks) else [], checks, hashes


def freeze_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select evaluation source hashes without opening evaluator labels."""
    if len(rows) != 64 or len({row["source_sha256"] for row in rows}) != 64:
        raise ValueError("evaluation64_invalid")
    return sorted(rows, key=lambda row: source.digest(f"{SEED}:{row['source_sha256']}".encode()))[
        :48
    ]


def fixture_e2e(path: Path) -> dict[str, Any]:
    """Drive 24 distinct source families through all three scripted variants."""
    frozen = source.protocol([], seed=SEED)
    rows = []
    for index in range(24):
        text = f"Family {index} alpha. Family {index} beta. Family {index} gamma."
        answer = f"Family {index} alpha."
        row = {
            "family_id": f"fixture-{index:02d}",
            "complete_source": text,
            "complete_response": answer,
            "source_sha256": source.digest(text.encode()),
            "response_sha256": source.digest(answer.encode()),
        }

        def reply(payload: dict[str, Any]) -> dict[str, Any]:
            body = json.loads(payload["messages"][1]["content"])
            witness = body["source_sentence_offsets"][0]["source_sentence_id"]
            content = json.dumps({"unsupported_probability": 0.2, "source_sentence_id": witness})
            return {
                "model": source.MODEL_ID,
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 12},
            }

        rows.extend(source.capture_fixture(row, frozen, reply, lambda text: len(text.split())))
        if index % 8 == 7:
            progress(time.monotonic(), "fixture", "batch", index + 1)
    result = {"independent_families": 24, "rows": rows, "model_calls": 0}
    atomic_json(path, result)
    return result


def validate_log(receipt: dict[str, Any]) -> None:
    """Reject a changed or missing log after the owned child has exited."""
    path = Path(receipt["log_path"])
    if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
        raise ValueError("validation_log_drift")


def dispatch(plan: dict[str, Any], started: float) -> list[dict[str, Any]]:
    """Run only the frozen roster and seal each closed log under its byte hash."""
    if plan != load_plan():
        raise ValueError("validation_manifest_drift")
    receipts = []
    for index, command in enumerate(plan["commands"]):
        progress(started, command["name"], "before_subprocess", index)
        spec = CommandSpec(
            command["name"], tuple(command["argv"]), command["classification"], command["timeout_s"]
        )
        receipt = run_commands(
            ROOT,
            [spec],
            log_dir=Path(command["private_root"]) / "logs",
            extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )[0]
        if (receipt["name"], receipt["command_argv"], command["classification"]) != (
            command["name"],
            command["argv"],
            command["classification"],
        ):
            raise ValueError("observed_child_command_drift")
        original = ROOT / receipt["log_path"]
        digest = sha256_file(original)
        sealed = RAW / "validation_logs" / f"{index:02d}_{command['name']}_{digest[7:]}.log"
        sealed.parent.mkdir(parents=True, exist_ok=True)
        if sealed.exists():
            raise ValueError("validation_log_path_reused")
        shutil.copyfile(original, sealed)
        receipt.update(
            classification=command["classification"],
            log_path=str(sealed),
            log_sha256=sha256_file(sealed),
        )
        validate_log(receipt)
        receipts.append(receipt)
        progress(started, command["name"], "after_subprocess", len(receipts))
    return receipts


def cold_reduce(path: Path) -> dict[str, Any]:
    """Reopen primitive candidate bytes and recompute family and log custody."""
    artifact = json.loads(path.read_text())
    if artifact["experiment_id"] != 7839 or artifact["task_id"] != "exp7839-intervention-protocol":
        raise ValueError("wrong_result_owner")
    manifest_path = Path(artifact["family_manifest_path"])
    protocol_path = Path(artifact["protocol_path"])
    if artifact["family_manifest_sha256"] != sha256_file(manifest_path) or artifact[
        "protocol_sha256"
    ] != sha256_file(protocol_path):
        raise ValueError("source_manifest_drift")
    manifest = json.loads(manifest_path.read_text())
    families = manifest["families"]
    if len(families) != 48 or len({row["family_id"] for row in families}) != 48:
        raise ValueError("family_count_drift")
    for row in families:
        if (
            source.digest(row["complete_source"].encode()) != row["source_sha256"]
            or source.digest(row["complete_response"].encode()) != row["response_sha256"]
        ):
            raise ValueError("source_byte_drift")
        if source.target_span(row["complete_response"].encode()) != row["target_sentence_span"]:
            raise ValueError("target_byte_drift")
    for receipt in artifact.get("validation_receipts", []):
        validate_log(receipt)
    return {
        "families": len(families),
        "rows": len(artifact["rows"]),
        "aligned_label_count": artifact["aligned_label_count"],
    }


def coverage_check(folder: Path) -> int:
    """Run frozen unit and real-CLI shards, then combine named completed files."""
    plan = load_plan()
    folder.mkdir(parents=True, exist_ok=True)
    completed = []
    started = time.monotonic()
    for index, argv in enumerate(plan["coverage_commands"]):
        progress(started, "coverage", "before_subprocess", index)
        for arg in argv:
            if arg.startswith("--basetemp=") or arg.startswith("--data-file="):
                Path(arg.partition("=")[2]).parent.mkdir(parents=True, exist_ok=True)
        run = subprocess.run(argv, cwd=ROOT, check=False, timeout=120)
        completed.append({"argv": argv, "exit_code": run.returncode})
        progress(started, "coverage", "after_subprocess", index + 1)
        if run.returncode != 0:
            break
    for file in plan["coverage_files"]:
        if not Path(file).is_file():
            return 1
    return int(
        len(completed) != len(plan["coverage_commands"])
        or any(row["exit_code"] for row in completed)
    )


def build_artifact(
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    protocol_path: Path | None,
    family_path: Path | None,
    labels: list[dict[str, Any]],
    started: float,
    spans: list[dict[str, Any]],
    flagged: bool = False,
) -> dict[str, Any]:
    """Keep scientific gates separate from fixture and validation outcomes."""
    failed_external = [row for row in checks if not row["passed"]]
    failed_required = [
        row for row in receipts if row["classification"] == "required" and not row["passed"]
    ]
    required_names = set(load_plan()["required_checks"])
    missing_required = required_names - {row["name"] for row in receipts}
    blocked = bool(failed_external)
    disqualified = not blocked and (bool(failed_required) or bool(missing_required) or flagged)
    ready = not blocked and not disqualified
    verdict = (
        "complete_blocked_required_intervention_evidence"
        if blocked
        else (
            "complete_disqualified_required_checks"
            if disqualified
            else "complete_circular_positive_intervention_qualification"
        )
    )
    failures = failed_external + [
        check("current_validation", Path(row["log_path"]), row["name"] + ".passed", True, False)
        for row in failed_required
    ]
    budget = {
        "intended": 144,
        "eligible": len(rows),
        "started": sum(bool(row.get("started")) for row in rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "censored": sum(bool(row.get("censored")) for row in rows),
        "excluded": sum(bool(row.get("excluded")) for row in rows),
        "independent": 48 if protocol_path else 0,
    }
    code_hashes = {
        str(path): sha256_file(path)
        for path in (ROOT / p for p in load_plan()["affected_sources"])
        if path.is_file()
    }
    checksum = source.digest(
        json.dumps(
            {
                "code": code_hashes,
                "inputs": hashes,
                "seed": SEED,
                "protocol": sha256_file(protocol_path) if protocol_path else None,
                "manifest": sha256_file(PLAN_PATH),
            },
            sort_keys=True,
        ).encode()
    )
    return {
        "schema": "carnot.exp7839.intervention_result.v1",
        "experiment_id": 7839,
        "task_id": "exp7839-intervention-protocol",
        "milestone": "2026.09.681",
        "run_date": "20260928",
        "honest_verdict": verdict,
        "verdict_class": "blocked"
        if blocked
        else "disqualified"
        if disqualified
        else "circular_positive",
        "flagged_adversarial": flagged,
        "gate_check_summary": failures,
        "rows": rows,
        "sample_size_budget": budget,
        "random_seed": SEED,
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "reproducibility_checksum": checksum,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "validation_command_manifest_path": str(PLAN_PATH),
        "validation_command_manifest_sha256": sha256_file(PLAN_PATH),
        "observed_child_commands": [
            {
                "name": row["name"],
                "argv": row["command_argv"],
                "classification": row["classification"],
            }
            for row in receipts
        ],
        "repository_health": {
            "historical_exp7828_verdict": "complete_disqualified_required_validation",
            "historical_exp7828_worktree_imports_passed": False,
            "current_diagnostic": next(
                (row for row in receipts if row["name"] == "repository_health_180s"), None
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
        "claim_scope": "scripted fixture conformance; exposed development labels only; no fresh generalization",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "model_file_hashes": [],
        },
        "intervention_protocol_ready_score": int(ready),
        "protocol_path": str(protocol_path) if protocol_path else None,
        "protocol_sha256": sha256_file(protocol_path) if protocol_path else None,
        "family_manifest_path": str(family_path) if family_path else None,
        "family_manifest_sha256": sha256_file(family_path) if family_path else None,
        "aligned_label_count": sum(row["label"] is not None for row in labels),
        "field_principles": {
            key: "Preserve exact source, scope, status and limits for independent replay."
            for key in (
                "experiment_id",
                "task_id",
                "honest_verdict",
                "verdict_class",
                "rows",
                "sample_size_budget",
                "gate_check_summary",
                "validation_receipts",
                "source_artifact_hashes",
                "acceptance_gate_results",
                "intervention_protocol_ready_score",
            )
        },
    }


def run_experiment(date: str) -> dict[str, Any]:
    """Prepare public families, qualify children, then publish one terminal record."""
    started = time.monotonic()
    progress(started, "start", "begin")
    if date != "20260928":
        raise ValueError("run_date_mismatch")
    plan = load_plan()
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
        result = build_artifact(checks, hashes, [], [], None, None, [], started, spans)
        atomic_json(OUTPUT, result)
        progress(started, "publish", "blocked")
        return result
    phase = time.monotonic()
    progress(started, "prepare", "begin")
    selected = freeze_families(public)
    frozen = source.protocol(selected, seed=SEED)
    frozen["tokenizer_requirement"] = (
        "GGUF embedded tokenizer at real runtime; injected counter in CPU fixture"
    )
    frozen["matching_rule"] = (
        "disjoint original source sentence, <=25% witness GGUF tokens, seeded hash order"
    )
    protocol_path = RAW / "intervention_protocol.json"
    family_path = RAW / "family_manifest.json"
    label_path = RAW / "aligned_labels.json"
    family_rows = [
        {
            "family_id": row["family_id"],
            "complete_source": row["complete_source"],
            "complete_response": row["complete_response"],
            "source_sha256": row["source_sha256"],
            "response_sha256": row["response_sha256"],
            "target_sentence_span": source.target_span(row["complete_response"].encode()),
            "source_sentence_offsets": source.sentence_offsets(row["complete_source"].encode()),
        }
        for row in selected
    ]
    evaluator = {
        row["family_id"]: row
        for row in (json.loads(line) for line in EXTERNAL["evaluator"].read_text().splitlines())
    }
    labels = []
    for row in selected:
        end = source.target_span(row["complete_response"].encode())["end_byte"]
        annotations = evaluator[row["family_id"]].get("annotations", [])
        label = (
            1
            if any(a.get("start", end) < end and a.get("end", 0) > 0 for a in annotations)
            else None
        )
        labels.append(
            {
                "family_id": row["family_id"],
                "label": label,
                "target_sentence_span": source.target_span(row["complete_response"].encode()),
            }
        )
    atomic_json(protocol_path, frozen)
    atomic_json(
        family_path,
        {"schema": "carnot.exp7839.family_manifest.v1", "seed": SEED, "families": family_rows},
    )
    atomic_json(
        label_path,
        {
            "schema": "carnot.exp7839.aligned_labels.v1",
            "labels": labels,
            "evaluator_sha256": sha256_file(EXTERNAL["evaluator"]),
        },
    )
    fixture = fixture_e2e(RAW / "fixture_rows.json")
    rows = [
        {
            "family_id": item["family_id"],
            "arm": arm,
            "seed": SEED,
            "source_sha256": item["source_sha256"],
            "answer_sha256": item["response_sha256"],
            "family_manifest_path": str(family_path),
            "status": "unstarted_no_model_execution",
            "disposition": "unstarted_no_model_execution",
            "started": False,
            "censored": False,
            "excluded": False,
            "probability": None,
            "deleted_sentence_id": None,
            "original_label": None,
        }
        for item in family_rows
        for arm in source.ARMS
    ]
    spans.append(
        {
            "phase": "prepare",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(selected) + fixture["independent_families"],
        }
    )
    progress(started, "prepare", "complete", len(selected) + fixture["independent_families"])
    candidate = build_artifact(
        checks, hashes, rows, [], protocol_path, family_path, labels, started, spans
    )
    candidate["aligned_label_sidecar_path"] = str(label_path)
    candidate["aligned_label_sidecar_sha256"] = sha256_file(label_path)
    candidate["fixture_rows_path"] = str(RAW / "fixture_rows.json")
    candidate["fixture_independent_families"] = fixture["independent_families"]
    atomic_json(Path(plan["candidate_path"]), candidate)
    phase = time.monotonic()
    progress(started, "validation", "begin")
    receipts = dispatch(plan, started)
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    adverse = next((row for row in receipts if row["name"] == "adversarial_verify"), None)
    try:
        flagged = (
            bool(json.loads(Path(adverse["log_path"]).read_text())["flagged_count"])
            if adverse
            else True
        )
    except (OSError, ValueError, KeyError, TypeError):
        flagged = True
    result = build_artifact(
        checks, hashes, rows, receipts, protocol_path, family_path, labels, started, spans, flagged
    )
    for key in (
        "aligned_label_sidecar_path",
        "aligned_label_sidecar_sha256",
        "fixture_rows_path",
        "fixture_independent_families",
    ):
        result[key] = candidate[key]
    atomic_json(OUTPUT, result)
    progress(started, "publish", "complete", len(rows))
    return result


def main(argv: list[str] | None = None) -> int:
    """Expose dated capture and bounded validation children through one CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--coverage-check", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260928":
        raise ValueError("run_date_mismatch")
    if args.fixture_e2e:
        result = fixture_e2e(args.fixture_e2e)
        print(json.dumps({"independent_families": result["independent_families"]}), flush=True)
        return 0
    if args.cold_replay:
        print(json.dumps(cold_reduce(args.cold_replay), sort_keys=True), flush=True)
        return 0
    if args.coverage_check:
        return coverage_check(args.coverage_check)
    return int(run_experiment(args.date)["verdict_class"] == "disqualified")
