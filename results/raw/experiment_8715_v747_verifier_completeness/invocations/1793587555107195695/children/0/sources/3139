"""Run the exposed RAGTruth semantic evidence pilot. REQ-REPORT-7716."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import current_work_receipt as custody
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.verify import semantic_evidence as evidence


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7716_v672_qwen_semantic_pilot")
RESULT = Path("results/experiment_7716_v672_qwen_semantic_pilot.json")
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
ARMS = ("whole_source", "indexed_windows")
SEED = 7716
SCOPE = {
    "tests": ["tests/python/test_experiment_7716_v672_qwen_semantic_pilot.py"],
    "changed_modules": [
        "python/carnot/experiment_7716_v672_qwen_semantic_pilot.py",
        "python/carnot/verify/semantic_evidence.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7716_v672_qwen_semantic_pilot.py"],
    "specs": ["REQ-REPORT-7716", "REQ-VERIFY-7716"],
    "e2e": ["task_cold_raw_replay"],
}
PRINCIPLE = "Measured evidence bounds the claim and prevents invalid downstream use."
GATES = (
    "validity",
    "readiness",
    "probability",
    "utility",
    "coverage",
    "source_dependence",
    "retention",
    "efficiency",
)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush real stage and call boundaries with measured elapsed time."""
    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[exp7716] {phase} {event} elapsed_s={time.monotonic() - started:.2f} {suffix}", flush=True
    )


def gate(
    check: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Retain exact operands for a failed external or resource prerequisite."""
    return {
        "check": check,
        "upstream_id": upstream,
        "artifact_path": str(path.resolve()),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def make_request(row: dict[str, Any], arm: str) -> dict[str, Any]:
    """Show complete source and original answer, with no annotation channel."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    visible = {"answer": row["answer"]}
    if arm == "whole_source":
        visible["complete_source"] = row["source"]
    else:
        visible["source_windows"] = evidence.sentence_windows(row["source"])
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "/no_think\nJudge whether the original answer is supported, contradicted, or unknown from the complete source. Return exactly one JSON object with decision (support, contradiction, unknown) and quote (an exact source substring). A quote alone does not prove support. Use unknown when unsure.",
            },
            {"role": "user", "content": json.dumps(visible, ensure_ascii=False)},
        ],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


def freeze_panel(root: Path, started: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select exposed families by public bytes before opening evaluator labels."""
    from carnot.experiment_7423_v651_annotated_protocol import (
        DEFAULT_CACHE_ROOT,
        authenticate_assets,
        load_release,
    )
    from carnot.reporting import natural_source_cohort as cohort
    from carnot.reporting import fresh_relation_cohort as base

    receipt = authenticate_assets(DEFAULT_CACHE_ROOT)
    progress(started, "release_decode", "before")
    sources, responses = load_release(receipt, started=started)
    progress(started, "release_decode", "after", completed_units=len(responses))
    families, _ = cohort.build_public_families(sources, responses, "v672-natural-20260926")
    eligible = [
        f
        for f in families
        if len(f["view"]["complete_source"]) <= 5000 and len(f["view"]["complete_response"]) <= 1000
    ]
    selected = sorted(eligible, key=lambda f: base.digest("v672-exp7716:" + f["family_id"]))[:24]
    labels = {r["id"]: r.get("labels") for r in responses}
    panel = []
    for family in selected:
        view = family["view"]
        annotations = labels.get(view["response_id"])
        panel.append(
            {
                "family_id": family["family_id"],
                "source_id": view["source_id"],
                "response_id": view["response_id"],
                "source": view["complete_source"],
                "answer": view["complete_response"],
                "annotation_types": [item["label_type"] for item in annotations]
                if isinstance(annotations, list)
                else None,
                "official_split": family["official_split"],
                "prior_exposure": True,
            }
        )
    roster = json.loads(
        (
            root / "results/raw/experiment_7715_v672_natural_source_cohort/candidate_manifest.json"
        ).read_text()
    )
    return panel, {
        "fresh_roster_count": sum(roster["observed_candidate_counts"].values()),
        "authenticated_assets": receipt["files"],
    }


def preflight(
    root: Path, started: float
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate the qualified protocol, release, GGUF, and owned CUDA."""
    from carnot import experiment_7630_v666_cuda_ownership as ownership
    from carnot.inference.sota_models import cached_sota_pair
    from llama_cpp import llama_cpp

    checks = [
        gate(
            "absolute_root",
            "current_work",
            root,
            "resolved_repo_root",
            str(ROOT),
            str(root.resolve()),
        )
    ]
    hashes: dict[str, Any] = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    named = [
        (Path("results/experiment_7714_v672_alignment_protocol.json"), "exp7714"),
        (
            Path("results/raw/experiment_7715_v672_natural_source_cohort/candidate_manifest.json"),
            "exp7715_roster",
        ),
        (
            Path("results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json"),
            "exp7423",
        ),
        (Path("ops/exclusion_manifest.yaml"), "operator_manifest"),
    ]
    for relative, upstream in named:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate("required_input", upstream, path, "readable_nonempty_bytes", True, present)
        )
        if present:
            hashes["valid_producers"][str(relative)] = custody.sha256_file(path)
        else:
            hashes["missing_custody"].append(str(relative))
    if any(not row["passed"] for row in checks):
        return checks, hashes, {}
    upstream = json.loads((root / named[0][0]).read_text())
    for field, expected in (("alignment_protocol_ready_score", 1), ("flagged_adversarial", False)):
        checks.append(
            gate(
                "alignment_qualified",
                "exp7714",
                root / named[0][0],
                field,
                expected,
                upstream.get(field),
            )
        )
    checks.append(
        gate(
            "alignment_qualified",
            "exp7714",
            root / named[0][0],
            "verdict_class_in_qualified",
            True,
            upstream.get("verdict_class") in {"circular_positive", "null"},
        )
    )
    manifest = json.loads((root / named[2][0]).read_text())
    checks.append(
        gate(
            "ragtruth_schema",
            "exp7423",
            root / named[2][0],
            "license",
            "MIT",
            manifest.get("license"),
        )
    )
    for asset in manifest.get("asset_receipt", {}).get("files", []):
        path = Path(asset["cache_path"])
        actual = custody.sha256_file(path) if path.is_file() else None
        checks.append(
            gate("ragtruth_asset_hash", "RAGTruth", path, "sha256", asset["sha256"], actual)
        )
        if actual:
            hashes["valid_producers"][str(path)] = actual
    if any(not row["passed"] for row in checks):
        return checks, hashes, {}
    pair = cached_sota_pair()
    model = next((item for item in pair or [] if item.get("hf_id") == MODEL_ID), None)
    model_path = Path(str((model or {}).get("model_path") or "/missing-qwen.gguf"))
    checks.append(
        gate(
            "cached_sota_pair_qwen",
            "local_model_cache",
            model_path,
            "qwen_q4_gguf_over_15gb",
            True,
            bool(model and model_path.is_file() and model_path.stat().st_size > 15_000_000_000),
        )
    )
    checks.append(
        gate(
            "llama_cuda_offload",
            "llama_cpp_build",
            model_path,
            "supports_gpu_offload",
            True,
            bool(llama_cpp.llama_supports_gpu_offload()),
        )
    )
    if any(not row["passed"] for row in checks):
        return checks, hashes, {}
    progress(started, "model_hash", "before", path=model_path)
    model_hash = custody.sha256_file(model_path)
    progress(started, "model_hash", "after", sha256=model_hash)
    hashes["valid_producers"][str(model_path)] = model_hash
    registry = ownership.ProcessRegistry.current()
    selected, inventory_rows = ownership.select_owned_capacity(
        ownership._current_inventory(), registry
    )
    checks.append(
        gate(
            "owned_cuda_capacity",
            "exp7630_cuda_ownership",
            root / RAW,
            "exclusive_idle_device_with_20000_mb",
            True,
            selected is not None,
        )
    )
    return (
        checks,
        hashes,
        {
            "model": model,
            "model_path": model_path,
            "model_sha256": model_hash,
            "registry": registry,
            "selected": selected,
            "inventory_rows": inventory_rows,
        },
    )


def build_artifact(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    hashes: dict[str, Any],
    runtime: dict[str, Any],
    spans: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    """Bound the scientific claim by observed calls and current checks."""
    paired = evidence.reduce_pairs(rows)
    complete = len(rows) == 48 and paired["paired_families"] == 24
    blocked = bool(failures and not runtime.get("model_load_attempted"))
    verdict = (
        "complete_blocked_" + failures[0]["check"]
        if blocked
        else "complete_null_exposed_semantic_pilot"
        if complete
        else "complete_partial_owned_generation"
    )
    verdict_class = "blocked" if blocked else "null" if complete else "partial"
    gates = {
        name: {"passed": None, "measured_operands": {}, "principle": PRINCIPLE} for name in GATES
    }
    gates["validity"].update(
        passed=not failures, measured_operands={"failed_checks": len(failures)}
    )
    gates["readiness"].update(
        passed=complete and not failures,
        measured_operands={
            "terminal_calls": len(rows),
            "paired_families": paired["paired_families"],
        },
    )
    gates["coverage"].update(
        passed=complete,
        measured_operands={
            "intended_families": 24,
            "observed_families": len({r["family_id"] for r in rows}),
        },
    )
    gates["efficiency"]["measured_operands"] = {
        "input_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
        "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
        "duration_s": duration,
    }
    fields = (
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "model_invoked",
        "execution_venue",
        "phase_spans",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "preconditions_checked",
        "validation_receipts",
        "verifier_is_oracle",
        "qwen_pilot_complete_score",
        "semantic_decision_rows",
        "current_model_receipts",
    )
    invoked = bool(runtime.get("model_load_attempted"))
    return {
        "schema": "carnot.exp7716.v672.qwen_semantic_pilot.v1",
        "experiment_id": "exp7716-qwen-semantic-pilot",
        "milestone": "2026.09.672",
        "run_date": "20260926",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": rows,
        "semantic_decision_rows": [
            {
                "family_id": r["family_id"],
                "arm": r["arm"],
                "prediction": r["metrics"]["decision"],
                "human_annotation": r["metrics"]["human_annotation"],
                "unknown": r["metrics"]["unknown"],
                "evidence_quote_valid": r["metrics"]["address_valid"],
                "latency_s": r["latency_s"],
            }
            for r in rows
        ],
        "paired_family_results": paired,
        "sample_size_budget": {
            "intended_families": 24,
            "observed_families": len({r["family_id"] for r in rows}),
            "eligible_families": len({r["family_id"] for r in rows}),
            "excluded_families": 0,
            "censored_families": len({r["family_id"] for r in rows if r["censored"]}),
            "effective_blocks": paired["paired_families"],
            "roles": ["previously_exposed_RAGTruth"],
            "prior_exposure": True,
            "fresh_accuracy_claim": False,
            "calls_max": 48,
            "seeds_windows_arms_are_independent_families": False,
        },
        "inference_substrate": "owned_local_llama_cpp_bounded_generation"
        if invoked
        else "no_model_load",
        "inference_substrate_class": "model_bounded_generation" if invoked else "no_model_load",
        "planned_inference_substrate_class": "model_bounded_generation",
        "MODEL_SPECS": [MODEL_ID] if invoked else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_specs": [{"hf_id": MODEL_ID, "model_path": runtime.get("model_path")}]
        if invoked
        else [{"model": "none", "reason": "blocked_before_model_load"}],
        "model_invoked": invoked,
        "invocation_counts": {
            "loads": runtime.get("model_load_attempted", 0),
            "forwards": runtime.get("generation_attempted", 0),
            "generations": len(rows),
            "input_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
            "failures": runtime.get("generation_failures", 0),
            "cancellations": runtime.get("cancellations", 0),
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host_pid": os.getpid(),
            "server_pid": runtime.get("server_pid"),
            "gpu_uuid": runtime.get("device_uuid"),
        },
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {"generation": SEED, "panel_order": "sha256:v672-exp7716:family_id"},
        "reproducibility_checksum": custody.canonical_hash(
            {
                "sources": hashes.get("valid_producers", {}),
                "seed": SEED,
                "arms": ARMS,
                "reducer": custody.sha256_file(ROOT / "python/carnot/verify/semantic_evidence.py"),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": [],
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": [],
            "terminal_readers": [],
            "unrelated_full_suite_debt": [],
        },
        "verifier_is_oracle": False,
        "field_principles": {
            **{field: PRINCIPLE for field in fields},
            **{f"acceptance_gate_{name}": PRINCIPLE for name in GATES},
        },
        "qwen_pilot_complete_score": 0,
        "current_model_receipts": runtime,
        "inference_mode": "live_gpu" if invoked else "no_model_load",
        "force_live": os.environ.get("CARNOT_FORCE_LIVE"),
        "activation": False,
        "production_promotion": False,
    }


def reduce_model_response(
    row: dict[str, Any], arm: str, content: str, finish: str
) -> dict[str, Any]:
    """Give the common server runner an arm-compatible semantic reducer."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    return evidence.reduce_response(row["source"], content, finish, row["annotation_types"])


def measure(
    root: Path, panel: list[dict[str, Any]], context: dict[str, Any], started: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the authenticated owned native launch path for one call per arm."""
    from carnot import experiment_7665_v668_qwen_grounded_claims as native
    from carnot.reporting import fresh_relation_cohort as base

    prepared = [
        {
            **row,
            "unit_id": row["family_id"],
            "population": "exposed_RAGTruth",
            "source_sha256": base.digest(row["source"]),
            "answer_sha256": base.digest(row["answer"]),
            "truth": None,
            "fixture_truth": None,
        }
        for row in panel
    ]
    progress(started, "model_generation", "before", planned_calls=48)
    rows, runtime = native._measure(
        root,
        prepared,
        context,
        started,
        arms=ARMS,
        request_builder=make_request,
        response_reducer=reduce_model_response,
        raw_path=RAW,
        task_id="experiment_7716_v672_qwen_semantic_pilot",
    )
    for row in rows:
        row["family_id"] = row["unit_id"]
        row["latency_s"] = row["generation_s"]
        row["denominator"] = 1
        row["exclusion_reason"] = None
        row["human_annotation"] = row["metrics"]["human_annotation"]
    runtime.update(
        {"model_path": str(context["model_path"]), "model_sha256": context["model_sha256"]}
    )
    progress(started, "model_generation", "after", completed_units=len(rows))
    return rows, runtime


def cold_reduce(path: Path) -> dict[str, Any]:
    """Reload frozen panel and independently score exact request/response bytes."""
    artifact = json.loads(path.read_text())
    rows = artifact["rows"]
    if artifact["verdict_class"] == "blocked":
        return {"passed": not rows and not artifact["model_invoked"], "calls": 0}
    panel_path = ROOT / RAW / "frozen_panel.json"
    panel = {row["family_id"]: row for row in json.loads(panel_path.read_text())}
    seen = set()
    for row in rows:
        key = (row["family_id"], row["arm"])
        if key in seen or key[0] not in panel:
            return {"passed": False, "reason": "panel_identity"}
        seen.add(key)
        request_path, response_path = Path(row["request_path"]), Path(row["raw_response_path"])
        if (
            custody.sha256_file(request_path) != row["request_sha256"]
            or custody.sha256_file(response_path) != row["raw_response_sha256"]
        ):
            return {"passed": False, "reason": "raw_hash"}
        if json.loads(request_path.read_text()) != make_request(panel[key[0]], key[1]):
            return {"passed": False, "reason": "request_mismatch"}
        response = json.loads(response_path.read_text())
        choice = response["choices"][0]
        content = str(choice["message"].get("content") or "")
        finish = str(choice.get("finish_reason") or "unknown")
        if content != row["response_text"] or finish != row["finish_reason"]:
            return {"passed": False, "reason": "response_mismatch"}
        if reduce_model_response(panel[key[0]], key[1], content, finish) != row["metrics"]:
            return {"passed": False, "reason": "metrics_mismatch"}
    paired = evidence.reduce_pairs(rows)
    return {
        "passed": len(rows) == 48 and paired == artifact["paired_family_results"],
        "calls": len(rows),
        "paired_families": paired["paired_families"],
    }


def terminal_commands(root: Path, candidate: Path) -> list[validation.CommandSpec]:
    """Declare fresh replay and both strict final-reader commands."""
    python = str(root / ".venv/bin/python")
    cli = str(root / SCOPE["static_paths"][0])
    return [
        validation.CommandSpec(
            "independent_cold_replay",
            (python, "-u", cli, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
    ]


def _span(name: str, started: float, begin: float, units: int, checkpoint: Path) -> dict[str, Any]:
    """Close one monotonic phase with its checkpoint and completed units."""
    end = time.monotonic()
    return {
        "phase": name,
        "start_s": begin - started,
        "end_s": end - started,
        "duration_s": end - begin,
        "heartbeat_timestamp_s": end - started,
        "completed_units": units,
        "checkpoint": str(checkpoint),
    }


def run_experiment(root: Path, date: str, output: Path) -> int:
    """Measure owned calls, validate changed behavior, then publish atomically."""
    started = time.monotonic()
    progress(started, "startup", "before", root=root.resolve())
    root = root.resolve(strict=True)
    if date != "20260926":
        raise ValueError("date_must_be_20260926")
    destination = output if output.is_absolute() else root / output
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7716-"))
    scope_path = raw_dir / "frozen_affected_scope.json"
    custody.atomic_json(scope_path, SCOPE)
    spans: list[dict[str, Any]] = []
    begin = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, context = preflight(root, started)
    failures = [item for item in checks if not item["passed"]]
    spans.append(_span("preconditions", started, begin, len(checks), scope_path))
    progress(started, "preconditions", "after", completed_units=len(checks), failures=len(failures))
    panel: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    if not failures:
        begin = time.monotonic()
        progress(started, "freeze_panel", "before")
        try:
            panel, receipt = freeze_panel(root, started)
            checks.append(
                gate(
                    "exposed_panel_size",
                    "RAGTruth",
                    raw_dir,
                    "distinct_families",
                    24,
                    len({r["family_id"] for r in panel}),
                )
            )
            checks.append(
                gate(
                    "fresh_roster_disjoint",
                    "exp7715",
                    raw_dir,
                    "fresh_roster_intersection",
                    0,
                    receipt["fresh_roster_count"],
                )
            )
            failures = [item for item in checks if not item["passed"]]
            if not failures:
                panel_path = raw_dir / "frozen_panel.json"
                custody.atomic_json(panel_path, panel)
                hashes["pre_gate_receipts"][str(RAW / "frozen_panel.json")] = custody.sha256_file(
                    panel_path
                )
        except (ValueError, KeyError, OSError) as error:
            failure = gate(
                "panel_authentication",
                "RAGTruth",
                raw_dir,
                "complete_exposed_panel",
                True,
                f"{type(error).__name__}:{error}",
            )
            checks.append(failure)
            failures.append(failure)
        spans.append(
            _span("freeze_panel", started, begin, len(panel), raw_dir / "frozen_panel.json")
        )
        progress(
            started, "freeze_panel", "after", completed_units=len(panel), failures=len(failures)
        )
    if not failures:
        begin = time.monotonic()
        try:
            rows, runtime = measure(root, panel, context, started)
        except BaseException as error:
            run_dirs = sorted((raw_dir / "runs").glob(f"*-{os.getpid()}"))
            checkpoint = run_dirs[-1] / "checkpoint.json" if run_dirs else None
            if checkpoint and checkpoint.is_file():
                rows = json.loads(checkpoint.read_text())["rows"]
                for row in rows:
                    row["family_id"] = row["unit_id"]
                    row["latency_s"] = row["generation_s"]
            runtime = {
                "model_load_attempted": 1,
                "generation_attempted": len(rows) + 1,
                "model_path": str(context["model_path"]),
                "model_sha256": context["model_sha256"],
                "device_uuid": context["selected"]["uuid"],
                "error": f"{type(error).__name__}:{error}",
            }
            failure = gate(
                "owned_generation",
                "exp7630_owned_qwen_server",
                raw_dir,
                "terminal_calls",
                48,
                len(rows),
            )
            checks.append(failure)
            failures.append(failure)
        spans.append(_span("measurement", started, begin, len(rows), raw_dir / "runs"))
        progress(started, "measurement", "after", completed_units=len(rows), failures=len(failures))
    rows_path = raw_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    hashes["pre_gate_receipts"][str(RAW / "rows.jsonl")] = custody.sha256_file(rows_path)
    artifact = build_artifact(rows, failures, hashes, runtime, spans, time.monotonic() - started)
    artifact["preconditions_checked"] = checks
    begin = time.monotonic()
    progress(started, "scoped_validation", "before")
    commands = validation.build_scoped_commands(
        root,
        SCOPE["tests"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7716",
    )
    receipts = validation.run_commands(
        root, commands, log_dir=raw_dir / "validation" / "affected", heartbeat_s=45
    )
    spans.append(
        _span(
            "scoped_validation", started, begin, len(receipts), raw_dir / "validation" / "affected"
        )
    )
    progress(
        started,
        "scoped_validation",
        "after",
        completed_units=len(receipts),
        passed=all(r["passed"] for r in receipts),
    )
    artifact["validation_receipts"]["required_commands"] = receipts
    if not all(row["passed"] for row in receipts):
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    candidate = raw_dir / "terminal_candidate.json"
    custody.atomic_json(candidate, artifact)
    begin = time.monotonic()
    progress(started, "terminal_readers", "before")
    terminal = validation.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=raw_dir / "validation" / "terminal",
        heartbeat_s=45,
    )
    spans.append(_span("terminal_readers", started, begin, len(terminal), candidate))
    progress(
        started,
        "terminal_readers",
        "after",
        completed_units=len(terminal),
        passed=all(r["passed"] for r in terminal),
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["flagged_adversarial"] = not next(
        r["passed"] for r in terminal if r["name"] == "adversarial_verify"
    )
    if not all(row["passed"] for row in receipts + terminal):
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["verdict_class"] = "disqualified"
        for item in artifact["acceptance_gate_results"].values():
            if item["passed"] is not None:
                item["passed"] = False
    artifact["qwen_pilot_complete_score"] = int(
        len(rows) == 48 and all(r["passed"] for r in receipts + terminal)
    )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    custody.atomic_json(destination, artifact)
    progress(
        started,
        "publication",
        "after",
        path=destination,
        verdict=artifact["honest_verdict"],
        sha256=custody.sha256_file(destination),
    )
    return 0 if artifact["verdict_class"] != "disqualified" else 1


def main(argv: list[str] | None = None) -> int:
    """Accept the fixed date or a read-only cold replay request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = cold_reduce(args.cold_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    return run_experiment(ROOT, args.date, args.output)
