"""Qualify exposed source views for REQ-REPORT-7796.

This module owns a new validation record. The earlier experiment's preparation
functions remain the source of byte and label behavior; its failed verdict does
not become a current input or a current success.
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
from carnot import experiment_7768_v676_source_view_qualification as source_views
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.verify import evidence_views

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7796_v678_source_view_qualification"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
SCOPE = RAW / "frozen_affected_scope.json"
CANDIDATE = Path("/tmp/carnot-7796/candidate.json")
SCOPE_SHA256 = "sha256:c647c6ff7d5d2cd2c481bd50103ad9cb4b6dc02c54c7a9744db3e5b45bb91c62"
MANIFEST = ROOT / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"
MODEL_SPECS: list[dict[str, Any]] = []
FIELD_PRINCIPLES = {
    **source_views.PRINCIPLES,
    "role_hashes": "Roles must not change when outputs are read.",
    "exposure_manifest": "Exposed development families cannot establish fresh generalization.",
    "source_view_manifest_path": "Every head needs the same authenticated bytes.",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Show actual elapsed work at a phase boundary or long-loop heartbeat."""
    print(
        f"[exp7796] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def _check(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep a failed operand and the bytes of its named authority together."""
    return source_views._check(upstream, path, field, expected, observed)


def preconditions(root: Path) -> list[dict[str, Any]]:
    """Check science custody, the separate pre-gate, and CPU resources first."""
    checks = source_views.preflight(root)
    manifest = root / MANIFEST.relative_to(ROOT)
    if manifest.is_file() and not any(not row["passed"] for row in checks):
        inventory = json.loads(manifest.read_text())
        public_manifest = manifest.parent / "public_manifest.json"
        checks.append(
            _check(
                "exp7727",
                public_manifest,
                "public_manifest_sha256",
                inventory.get("public_manifest_sha256"),
                sha256_file(public_manifest) if public_manifest.is_file() else None,
            )
        )
        if checks[-1]["passed"]:
            try:
                observed = corpus.cold_reduce(manifest, corpus.COUNTS)
            except (KeyError, ValueError, OSError) as error:
                observed = f"{type(error).__name__}:{error}"
            checks.append(
                _check(
                    "exp7727",
                    manifest,
                    "cold_reduce",
                    640,
                    observed.get("families") if isinstance(observed, dict) else observed,
                )
            )
    checks.append(
        _check(
            "task_resource",
            root,
            "disk_free_ge_100MiB",
            True,
            shutil.disk_usage(root).free >= 100 * 1024 * 1024,
        )
    )
    checks.append(
        _check("task_backend", root, "inference_substrate_class", "no_model_load", "no_model_load")
    )
    return checks


def validate_scope(scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject any child argument that differs from the sealed file scope."""
    if sha256_file(SCOPE) != SCOPE_SHA256:
        raise ValueError("validation_scope_drift")
    frozen = json.loads(SCOPE.read_text())
    if scope != frozen:
        raise ValueError("validation_scope_drift")
    private = Path("/tmp/carnot-7796")
    built = build_scoped_commands(
        ROOT,
        scope["tests"],
        scope["changed_modules"],
        static_paths=scope["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / "coverage.7796",
    )
    if scope["commands"][: len(built)] != [
        {"name": item.name, "argv": list(item.argv)} for item in built
    ]:
        raise ValueError("validation_scope_drift")
    if len(scope["commands"]) != len(built) + 4:
        raise ValueError("validation_scope_drift")
    return scope["commands"]


def prepare_public_only(manifest_path: Path, counts: dict[str, int]) -> list[dict[str, Any]]:
    """Read public shards alone so a private label edit cannot select a view."""
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for role, count in counts.items():
        meta = manifest["roles"][role]
        public_path = manifest_path.parent / meta["public_path"]
        public = source_views._read_jsonl(public_path)
        if len(public) != count or sha256_file(public_path) != meta["public_sha256"]:
            raise ValueError("public_role_hash_or_count")
        rows.extend(source_views.prepare_public(row) for row in public)
    return rows


prepare = source_views.prepare_corpus
replay = source_views.replay_corpus
map_targets = source_views.checked_targets


def over_budget_decisions(pair: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep every arm in the denominator when the shared window cap fires."""
    if pair["a"]["abstention"] is None:
        raise ValueError("view_is_within_budget")
    return {
        name: {"risk": 0.5, "brier": 0.25, "cost": 0.25, "action": "escalate"}
        for name in evidence_views.ARMS
    }


def run_child(name: str, argv: list[str], log_path: Path) -> dict[str, Any]:
    """Run one bounded owned child and store the complete log under task raw."""
    timeout = 300.0 if name == "full_python_suite" else 900.0
    receipt = run_commands(
        ROOT,
        [CommandSpec(name, tuple(argv), "frozen_exp7796", timeout_s=timeout)],
        log_dir=log_path.parent,
        extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )[0]
    produced = ROOT / receipt["log_path"]
    if produced != log_path:
        shutil.copyfile(produced, log_path)
    receipt["log_path"] = str(log_path)
    receipt["log_sha256"] = sha256_file(log_path)
    return receipt


def run_validation(
    scope: dict[str, Any],
    log_dir: Path,
    *,
    before_readers: Callable[[], None] | None = None,
    prior_broad: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run sealed children; reuse only a hash-bound failed broad diagnostic."""
    commands = validate_scope(scope)
    receipts = []
    started = time.monotonic()
    for index, command in enumerate(commands, 1):
        if command["name"] == "cold_replay" and before_readers is not None:
            before_readers()
        progress(started, command["name"], "before_subprocess", index - 1)
        if (
            command["name"] == "full_python_suite"
            and prior_broad is not None
            and prior_broad.get("command_argv") == command["argv"]
            and prior_broad.get("passed") is False
            and Path(prior_broad["log_path"]).is_file()
            and sha256_file(Path(prior_broad["log_path"])) == prior_broad.get("log_sha256")
        ):
            receipt = {**prior_broad, "reused_prior_diagnostic": True}
            progress(started, command["name"], "reused_prior_diagnostic", index)
        else:
            receipt = run_child(
                command["name"], command["argv"], log_dir / f"{index:02d}_{command['name']}.log"
            )
        receipts.append(receipt)
        progress(started, command["name"], "after_subprocess", index)
    return receipts


def _sources(root: Path, manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Name every consumed authority and its exact current bytes."""
    paths = [
        (
            "exp7727",
            root / "results/experiment_7727_v673_development_corpus.json",
            ["development_cohort_ready_score", "development_manifest_sha256"],
        ),
        (
            "exp7727",
            root / MANIFEST.relative_to(ROOT),
            ["counts", "roles", "public_manifest_sha256"],
        ),
        (
            "exp7727",
            root / MANIFEST.relative_to(ROOT).parent / "public_manifest.json",
            ["counts", "roles"],
        ),
        (
            "exp7753_conductor_pre_gate",
            root / "results/experiment_7753_v675_contract_methods.json",
            ["contract_ready_score"],
        ),
    ]
    if manifest is not None:
        for role, meta in manifest["roles"].items():
            for kind in ("public", "evaluator"):
                paths.append(
                    (
                        "exp7727",
                        root / MANIFEST.relative_to(ROOT).parent / meta[f"{kind}_path"],
                        [role, kind],
                    )
                )
    return [
        {
            "upstream_id": owner,
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "exists": path.is_file(),
            "date": "20260926" if owner == "exp7727" else "20260927",
            "imported_fields": fields,
            "eligibility": "development" if owner == "exp7727" else "pre_gate_only",
        }
        for owner, path, fields in paths
    ]


def build_candidate(
    root: Path,
    checks: list[dict[str, Any]],
    prepared: dict[str, Any] | None,
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    *,
    flagged: bool = False,
    raw: Path = RAW,
) -> dict[str, Any]:
    """Reduce independent families and required exits into one honest record."""
    manifest_path = root / MANIFEST.relative_to(ROOT)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    failed = [
        {key: value for key, value in row.items() if key != "passed"}
        for row in checks
        if not row["passed"]
    ]
    required = [row for row in receipts if row["name"] != "full_python_suite"]
    for row in required:
        if not row["passed"]:
            failed.append(
                _check(
                    "exp7796_validation",
                    Path(row["log_path"]),
                    f"{row['name']}.exit_code",
                    0,
                    row["exit_code"],
                )
            )
    if flagged:
        failed.append(
            _check("exp7796_terminal_reader", root / "results", "flagged_adversarial", False, True)
        )
    ready = prepared is not None and not failed and len(required) == 11
    blocked = prepared is None and any(not row["passed"] for row in checks)
    rows = []
    if prepared is not None:
        rows_hash = sha256_file(raw / "rows.jsonl")
        for public, target in zip(prepared["rows"], prepared["targets"], strict=True):
            abstention = public["abstention"]
            rows.append(
                {
                    "family_id": public["family_id"],
                    "role": public["role"],
                    "source_sha256": public["source_sha256"],
                    "response_sha256": public["response_sha256"],
                    "source_view_rows_sha256": rows_hash,
                    "response_label": target["response_label"],
                    "sentence_targets": target["sentence_targets"],
                    "annotation_status": target["reason"],
                    "abstention": abstention,
                    "view_a_windows": len(public["view_a"]["windows"]),
                    "view_b_windows": len(public["view_b"]["windows"]),
                    "answer_units": len(public["view_a"]["answer_units"]),
                    "over_budget_arms": over_budget_decisions(
                        evidence_views.deserialize_pair(
                            {"a": public["view_a"], "b": public["view_b"]}
                        )
                    )
                    if abstention
                    else None,
                    "status": "rejected_over_budget" if abstention else "completed",
                }
            )
    source_hashes = _sources(root, manifest)
    identity = {
        "code": sha256_file(Path(__file__)),
        "wrapper": sha256_file(root / "scripts/experiments" / f"{NAME}.py")
        if (root / "scripts/experiments" / f"{NAME}.py").is_file()
        else None,
        "inputs": source_hashes,
        "roles": corpus.COUNTS,
        "scope": sha256_file(SCOPE),
        "seed": 7796,
    }
    checksum = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    duration = sum(span["duration_s"] for span in spans)
    return {
        "schema": "carnot.exp7796.source_view_qualification.v1",
        "experiment_id": "exp7796-source-view-qualification",
        "milestone": "2026.09.678",
        "run_date": "20260928",
        "honest_verdict": "complete_blocked_external_inputs"
        if blocked
        else "complete_circular_positive_source_view_readiness"
        if ready
        else "complete_disqualified_required_validation",
        "verdict_class": "blocked" if blocked else "circular_positive" if ready else "disqualified",
        "flagged_adversarial": flagged,
        "gate_check_summary": failed,
        "rows": rows,
        "acceptance_gate_results": {
            "validity": bool(prepared) and not failed,
            "readiness": int(ready),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": duration,
        "phase_spans": spans,
        "random_seed": 7796,
        "reproducibility_checksum": checksum,
        "sample_size_budget": {
            "intended": 640,
            "eligible": sum(row["status"] == "completed" for row in rows),
            "started": len(rows),
            "completed": len(rows),
            "excluded": 0,
            "censored": 0,
            "rejected": sum(row["status"] == "rejected_over_budget" for row in rows),
            "independent_n": len(rows),
            "role_counts": corpus.COUNTS,
        },
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "verifier_is_oracle": True,
        "claim_scope": "exposed_development_fixture_only; no hidden generalization or oracle-distinct benefit",
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "loaded_file_hashes": []},
        "sentence_protocol_ready_score": int(ready),
        "evidence_view_ready_score": int(ready),
        "source_view_manifest_path": prepared["source_view_manifest_path"]
        if ready and prepared
        else None,
        "role_hashes": {
            role: {kind: meta[f"{kind}_sha256"] for kind in ("public", "evaluator")}
            for role, meta in manifest["roles"].items()
        }
        if manifest
        else {},
        "exposure_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
            "development_only": True,
            "fresh_generalization_eligible": False,
        },
        "repository_health": {
            "historical_exp7782_verdict": "complete_disqualified_required_validation",
            "broad_suite_exit": next(
                (row["exit_code"] for row in receipts if row["name"] == "full_python_suite"), None
            ),
        },
    }


def cold_reduce(candidate_path: Path, raw: Path = RAW) -> dict[str, Any]:
    """Reopen sealed public and private shards in a fresh process."""
    candidate = json.loads(candidate_path.read_text())
    result = replay(MANIFEST, raw, corpus.COUNTS)
    rows = source_views._read_jsonl(raw / "rows.jsonl")
    if (
        result["families"] != sum(corpus.COUNTS.values())
        or len(candidate["rows"]) != len(rows)
        or [row["family_id"] for row in candidate["rows"]] != [row["family_id"] for row in rows]
    ):
        raise ValueError("candidate_roster_mismatch")
    return result


def run_experiment(date: str) -> dict[str, Any]:
    """Run current source preparation and its exact prospective checks."""
    start = time.monotonic()
    progress(start, "start", "begin")
    if date != "20260928":
        raise ValueError("run_date_mismatch")
    RAW.mkdir(parents=True, exist_ok=True)
    private = Path("/tmp/carnot-7796")
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    scope = json.loads(SCOPE.read_text())
    validate_scope(scope)
    progress(start, "preconditions", "begin")
    checks = preconditions(ROOT)
    spans = [{"phase": "preconditions", "duration_s": time.monotonic() - start}]
    if any(not row["passed"] for row in checks):
        candidate = build_candidate(ROOT, checks, None, [], spans)
        candidate["duration_s"] = time.monotonic() - start
        atomic_json(OUTPUT, candidate)
        progress(start, "publish", "blocked", 0)
        return candidate
    progress(start, "preconditions", "complete", len(checks))
    begun = time.monotonic()
    prepared = prepare(MANIFEST, RAW, corpus.COUNTS)
    spans.append({"phase": "prepare", "duration_s": time.monotonic() - begun})
    progress(start, "prepare", "complete", len(prepared["rows"]))
    candidate_path = CANDIDATE
    atomic_json(candidate_path, build_candidate(ROOT, checks, prepared, [], spans))
    prior_broad = None
    if OUTPUT.is_file():
        previous = json.loads(OUTPUT.read_text())
        prior_broad = next(
            (
                row
                for row in previous.get("validation_receipts", [])
                if row["name"] == "full_python_suite"
            ),
            None,
        )
    receipts = run_validation(
        scope,
        RAW / "validation_logs",
        before_readers=lambda: atomic_json(
            candidate_path, build_candidate(ROOT, checks, prepared, [], spans)
        ),
        prior_broad=prior_broad,
    )
    spans.extend(
        {"phase": row["name"], "duration_s": row.get("duration_s", 0.0)} for row in receipts
    )
    adversarial = next(row for row in receipts if row["name"] == "adversarial_verify")
    try:
        flagged = json.loads(Path(adversarial["log_path"]).read_text())["flagged_count"] > 0
    except (ValueError, KeyError, OSError):
        flagged = True
    final = build_candidate(ROOT, checks, prepared, receipts, spans, flagged=flagged)
    final["duration_s"] = time.monotonic() - start
    atomic_json(OUTPUT, final)
    progress(start, "publish", "complete", len(final["rows"]))
    return final


def main(argv: list[str] | None = None) -> int:
    """Expose one dated producer and one cold replay without model loading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay")
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = cold_reduce(Path(args.cold_replay))
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    result = run_experiment(args.date)
    return int(result["verdict_class"] == "disqualified")
