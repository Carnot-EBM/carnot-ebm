"""Measure a CPU record-address protocol on fixed fixtures and exposed pilots.

Fixture truth tests plumbing only. Eight old pilot sources show natural coverage
gaps but cannot establish fresh accuracy. Spec: REQ-REPORT-7700.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.experiment_7672_v669_bound_relations import fixture_cases
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify.record_addresses import (
    FEATURE_VOCABULARY,
    analyze_answer,
    bind_proposition,
    index_records,
    replay_analysis,
    resolve_address,
)
from carnot.verify.tool_source_atoms import digest
from carnot.verify.tool_source_relations import verify_relations


ROOT = Path(__file__).resolve().parents[2]
PILOT = Path("results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl")
RAW = Path("results/raw/experiment_7700_v671_record_span_protocol")
RESULT = Path("results/experiment_7700_v671_record_span_protocol.json")
MODULE = "python/carnot/experiment_7700_v671_record_span_protocol.py"
CAPABILITY = "python/carnot/verify/record_addresses.py"
TEST = "tests/python/test_experiment_7700_v671_record_span_protocol.py"
WRAPPER = "scripts/experiments/experiment_7700_v671_record_span_protocol.py"
SCOPE = {
    "tests": [TEST],
    "changed_modules": [MODULE, CAPABILITY],
    "static_paths": [WRAPPER],
    "specs": ["REQ-REPORT-7700", "REQ-VERIFY-7700"],
    "e2e": ["task_original_source_answer_replay"],
}
MODEL_SPECS: list[str] = []


def progress(started: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Make long owned work visible to the conductor and operator."""

    print(
        f"[exp7700] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(
    name: str, upstream: str, path: str, field: str, operator: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep exact observed operands even when an external input is absent."""

    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if operator == "==" else observed != expected,
    }


def preconditions(root: Path) -> list[dict[str, Any]]:
    """Authenticate required local resources without treating output as input."""

    root = root.resolve()
    paths = (
        PILOT,
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("openspec/capabilities/research-reporting/spec.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path(MODULE),
        Path(CAPABILITY),
        Path(TEST),
        Path(WRAPPER),
        Path(".venv/bin/pytest"),
        Path(".venv/bin/coverage"),
        Path(".venv/bin/ruff"),
        Path(".venv/bin/mypy"),
    )
    checks = [
        _check(
            "input_exists",
            "exp7700",
            str(path),
            "readable_nonempty_bytes",
            "==",
            True,
            (root / path).is_file() and (root / path).stat().st_size > 0,
        )
        for path in paths
    ]
    checks.append(
        _check(
            "absolute_root",
            "exp7700",
            str(root),
            "is_absolute_dir",
            "==",
            True,
            root.is_absolute() and root.is_dir(),
        )
    )
    checks.append(
        _check(
            "owned_process",
            "exp7700",
            f"/proc/{os.getpid()}",
            "exists",
            "==",
            True,
            Path(f"/proc/{os.getpid()}").is_dir(),
        )
    )
    return checks


def freeze_panel(pilots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fix source roles before decisions and keep evaluator labels out."""

    if len(pilots) != 8 or len({row["component_hash"] for row in pilots}) != 8:
        raise ValueError("eight_distinct_pilots_required")
    panel = []
    for case in fixture_cases():
        panel.append(
            {
                "unit_id": case["id"],
                "population": "fixture",
                "split": case["split"],
                "source": case["source"],
                "answer": case["answer"],
                "truth": case["truth"],
                "attack": case["attack"],
                "prior_exposure": True,
            }
        )
    for pilot in pilots:
        source, answer = pilot["complete_source"], pilot["complete_answer"]
        if pilot["source_sha256"] != digest(source):
            raise ValueError("pilot_source_authentication_failure")
        if pilot["answer_sha256"] != digest(answer):
            raise ValueError("pilot_answer_authentication_failure")
        panel.append(
            {
                "unit_id": pilot["component_hash"],
                "population": "pilot",
                "split": "prior_exposure",
                "source": source,
                "answer": answer,
                "truth": None,
                "attack": None,
                "prior_exposure": True,
            }
        )
    return panel


def feature_schema() -> dict[str, Any]:
    """Freeze the exact feature names that later tasks may consume."""

    return {
        **FEATURE_VOCABULARY,
        "schema": "carnot.exp7700.record_features.v1",
        "unit": "original source and complete answer",
        "sentence_id": "S-prefixed full answer span",
        "proposition_id": "P-prefixed narrow checkable span",
        "tuple_truth": "Narrow independent status after claim binding; address alone is not support.",
        "pilot_role": "exposed diagnostic with unknown evaluator truth",
    }


def build_rows(panel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare old and new readers on identical untouched unit bytes."""

    rows = []
    for unit in panel:
        source, answer = unit["source"], unit["answer"]
        analysis = analyze_answer(source, answer)
        legacy = verify_relations(source, answer)
        records = index_records(source)
        proposition = analysis["propositions"][0] if analysis["propositions"] else None
        selected = records[0] if records else None
        if proposition is not None:
            for record in records:
                trial = resolve_address(source, records, span=[record.byte_start, record.byte_end])
                if bind_proposition(analysis, trial, proposition["proposition_id"])[
                    "claim_binding"
                ]:
                    selected = record
                    break
        numeric = (
            resolve_address(source, records, span=[selected.byte_start, selected.byte_end])
            if selected
            else resolve_address(source, records, span=[0, 1])
        )
        quoted = (
            resolve_address(source, records, quote=selected.source_bytes)
            if selected
            else resolve_address(source, records, quote="")
        )
        binding = (
            bind_proposition(analysis, numeric, proposition["proposition_id"])
            if proposition
            else None
        )
        common = {
            "unit_id": unit["unit_id"],
            "population": unit["population"],
            "split": unit["split"],
            "truth": unit["truth"],
            "attack": unit["attack"],
            "source_sha256": digest(source),
            "answer_sha256": digest(answer),
            "excluded": unit["population"] == "pilot",
            "prior_exposure": unit["prior_exposure"],
            "provenance": "exact_fixture_oracle" if unit["population"] == "fixture" else str(PILOT),
        }
        rows.append(
            {
                **common,
                "arm": "old_relation",
                "observed": legacy["status"],
                "censored": legacy["status"] == "unknown",
                "raw_metrics": {
                    "relations": len(legacy["relations"]),
                    "checked_spans": len(legacy["checked_spans"]),
                    "unknown_spans": len(legacy["unknown_spans"]),
                },
                "addressing_metrics": None,
                "certificate_types": [],
            }
        )
        rows.append(
            {
                **common,
                "arm": "record_span",
                "observed": analysis["whole_answer_status"],
                "censored": analysis["whole_answer_status"] == "unknown",
                "raw_metrics": {
                    "records": len(analysis["records"]),
                    "sentences": len(analysis["sentences"]),
                    "propositions": len(analysis["propositions"]),
                    "narrow_supported": sum(
                        item["narrow_status"] == "supported" for item in analysis["propositions"]
                    ),
                    "narrow_contradicted": sum(
                        item["narrow_status"] == "contradicted" for item in analysis["propositions"]
                    ),
                    "narrow_unknown": sum(
                        item["narrow_status"] == "unknown" for item in analysis["propositions"]
                    ),
                    "residual_unknown": bool(analysis["residual_unknown_text"]),
                },
                "addressing_metrics": {
                    "exact_quote_span": quoted.exact_span,
                    "numeric_exact_span": numeric.exact_span,
                    "unique_quote_containment": quoted.unique_containment,
                    "unique_numeric_containment": numeric.unique_containment,
                    "claim_binding": binding["claim_binding"] if binding else False,
                    "tuple_truth": binding["tuple_truth"] if binding else "unknown",
                    "quote_reason": quoted.reason,
                    "numeric_reason": numeric.reason,
                    "source_span": list(numeric.span) if numeric.span else None,
                },
                "certificate_types": sorted(
                    {item["certificate_type"] for item in analysis["propositions"]}
                ),
                "sentences": analysis["sentences"],
                "propositions": analysis["propositions"],
                "residual_unknown_text": analysis["residual_unknown_text"],
            }
        )
    return rows


def cold_reduce(candidate: Path, panel_path: Path) -> dict[str, Any]:
    """Reload raw source bytes and independently check the saved unit decisions."""

    artifact = json.loads(candidate.read_text(encoding="utf-8"))
    panel = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines()]
    if len(panel) != 80 or len(artifact["rows"]) != 2 * len(panel):
        raise ValueError("row_reduction_count_mismatch")
    for unit, old, new in zip(panel, artifact["rows"][::2], artifact["rows"][1::2], strict=True):
        source, answer = unit["source"], unit["answer"]
        narrow = analyze_answer(source, answer)
        previous = verify_relations(source, answer)
        if (
            old["unit_id"] != unit["unit_id"]
            or new["unit_id"] != unit["unit_id"]
            or old["arm"] != "old_relation"
            or new["arm"] != "record_span"
            or old["observed"] != previous["status"]
            or new["observed"] != narrow["whole_answer_status"]
            or new["source_sha256"] != digest(source)
            or new["answer_sha256"] != digest(answer)
            or new["sentences"] != narrow["sentences"]
            or new["propositions"] != narrow["propositions"]
            or new["residual_unknown_text"] != narrow["residual_unknown_text"]
        ):
            raise ValueError("row_reduction_mismatch")
        replay_analysis(source, answer, narrow)
    return {"passed": True, "fixture_groups": 72, "pilot_groups": 8, "rows": len(artifact["rows"])}


def build_artifact(
    rows: list[dict[str, Any]],
    panel: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    date: str,
    duration: float,
    *,
    spans: list[dict[str, Any]] | None = None,
    terminal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep protocol readiness separate from oracle-distinct science."""

    failed = [check for check in checks if not check["passed"]]
    blocked = any(check["check"] == "input_exists" for check in failed)
    valid = not failed and bool(receipts) and all(row.get("passed") for row in receipts)
    fixture = [
        row for row in rows if row["population"] == "fixture" and row["arm"] == "record_span"
    ]
    pilots = [row for row in rows if row["population"] == "pilot" and row["arm"] == "record_span"]
    false_support = sum(
        row["observed"] == "supported" and row["truth"] != "supported" for row in fixture
    )
    protocol = (
        len(fixture) == 72
        and len(pilots) == 8
        and sum(row["split"] == "held_out" for row in fixture) == 24
        and false_support == 0
    )
    ready = valid and protocol
    verdict_class = (
        "blocked"
        if blocked
        else "disqualified"
        if not valid
        else "circular_positive"
        if ready
        else "null"
    )
    verdict = {
        "blocked": "complete_blocked_missing_external_input",
        "disqualified": "complete_disqualified_required_checks",
        "circular_positive": "complete_circular_positive_record_protocol_ready",
        "null": "complete_null_record_protocol_not_ready",
    }[verdict_class]
    addressing = {
        key: sum(bool(row["addressing_metrics"].get(key)) for row in fixture)
        for key in (
            "exact_quote_span",
            "numeric_exact_span",
            "unique_quote_containment",
            "unique_numeric_containment",
            "claim_binding",
        )
    }
    addressing["tuple_truth"] = {
        status: sum(row["addressing_metrics"]["tuple_truth"] == status for row in fixture)
        for status in ("supported", "contradicted", "unknown")
    }
    gate_values = {
        "validity": (
            valid,
            {
                "failed_preconditions": len(failed),
                "required_receipts": len(receipts),
                "failed_receipts": sum(not row.get("passed") for row in receipts),
            },
        ),
        "readiness": (
            ready,
            {
                "fixture_groups": len(fixture),
                "pilot_groups": len(pilots),
                "false_support": false_support,
                "held_out": sum(row["split"] == "held_out" for row in fixture),
            },
        ),
        "coverage": (
            protocol,
            {
                "independent_fixture_groups": len(fixture),
                "exposed_pilots": len(pilots),
                "unknown_remainders": sum(bool(row["residual_unknown_text"]) for row in fixture),
            },
        ),
        "freshness": (None, {"fresh_natural_groups": 0, "exposed_pilots": len(pilots)}),
        "probability": (None, {"independent_probability_labels": 0}),
        "utility": (None, {"measured_decision_outcomes": 0}),
        "retention": (None, {"delayed_replay_groups": 0}),
        "efficiency": (None, {"current_model_tokens": 0, "duration_s": duration}),
    }
    gate_principles = {
        "validity": "Failed custody or required checks cannot propagate evidence.",
        "readiness": "Tested addressing and typed claims require valid terminal checks.",
        "coverage": "Quality thresholds prevent a plumbing result from becoming an effect claim.",
        "freshness": "Previously exposed pilots do not measure fresh accuracy.",
        "probability": "A probability effect needs independent probability labels.",
        "utility": "A decision effect needs measured actions and outcomes.",
        "retention": "Retention bounds prevent apparent improvement by forgetting.",
        "efficiency": "Resource benefit requires a comparable measured workload.",
    }
    gates = [
        {
            "gate": name,
            "passed": passed,
            "measured_operands": operands,
            "principle": gate_principles[name],
        }
        for name, (passed, operands) in gate_values.items()
    ]
    hashes = {
        "producers": {str(PILOT): sha256_file(ROOT / PILOT)} if (ROOT / PILOT).is_file() else {},
        "pre_gate_receipts": {},
        "missing_evidence": [str(PILOT)] if not (ROOT / PILOT).is_file() else [],
    }
    principles = {
        name: "This record makes the stated claim independently checkable and limits its scope."
        for name in (
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
            "source_artifact_hashes",
            "preconditions_checked",
            "validation_receipts",
            "verifier_is_oracle",
            "field_principles",
            "record_protocol_ready_score",
            "feature_schema_path",
            "addressing_metrics",
            "natural_coverage_diagnostic",
        )
    }
    principles.update({f"acceptance_gate_{name}": value for name, value in gate_principles.items()})
    backend = "codex" if os.environ.get("CODEX_SESSION_ID") else "unknown"
    return {
        "schema": "carnot.exp7700.v671.record_span_protocol.v1",
        "experiment_id": "exp7700-record-span-protocol",
        "milestone": "2026.09.671",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": {"fixtures": 72, "held_out": 24, "exposed_pilots": 8},
            "observed": len(panel),
            "eligible": len(fixture),
            "excluded": len(pilots),
            "censored": sum(row["censored"] for row in fixture),
            "effective_blocks": {"fixture": len(fixture), "fresh_natural": 0},
            "prior_exposure": "Fixtures are exact oracles; all eight natural pilots were exposed in V664/V669.",
            "inference_limits": "No natural accuracy, learned-policy benefit or causal effect claim.",
        },
        "inference_substrate": "deterministic_tool_source_atoms_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [{"model": "none", "reason": "CPU deterministic record protocol"}],
        "model_invoked": False,
        "invocation_counts": {
            name: 0
            for name in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": platform.node(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans or [],
        "duration_s": duration,
        "random_seed": {"fixture_order": "fixed dialect, index enumeration; no random draw"},
        "reproducibility_checksum": canonical_hash(
            {
                "pilot_sha256": hashes["producers"],
                "scope": SCOPE,
                "feature_schema": feature_schema(),
                "reducer_sha256": sha256_file(ROOT / MODULE),
                "capability_sha256": sha256_file(ROOT / CAPABILITY),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": receipts,
            "terminal_readers": terminal or {},
            "unrelated_full_suite_debt": [],
        },
        "verifier_is_oracle": True,
        "field_principles": principles,
        "record_protocol_ready_score": int(ready),
        "feature_schema_path": str(RAW / "feature_schema.json"),
        "addressing_metrics": addressing,
        "natural_coverage_diagnostic": [
            {
                "unit_id": row["unit_id"],
                "whole_answer_status": row["observed"],
                "unknown_remainder": bool(row["residual_unknown_text"]),
                "fresh_accuracy_claim": False,
            }
            for row in pilots
        ],
        "execution_recovery_receipt": {
            "declared_backend": "codex",
            "effective_backend": backend,
            "force_experiments": os.environ.get("CODEX_FORCE_EXPERIMENTS"),
            "successful_invocation": {
                "kind": "current_api_assistant_invocation",
                "backend": backend,
                "session_id": os.environ.get("CODEX_SESSION_ID"),
            },
            "future_quota_guaranteed": False,
        },
        "prior_failures": [
            {
                "experiment_id": "exp7686-record-span-protocol",
                "custody": "not_emitted_usage_limit_three_attempts",
                "scientific_verdict": None,
            },
            {
                "experiment_id": "exp7676-qwen-quote-relations",
                "custody": "terminal_disqualified_required_checks",
                "scientific_verdict": "complete_disqualified_required_checks",
            },
            {
                "experiment_id": "exp7672-bound-relations",
                "custody": "terminal_fixture_oracle",
                "scientific_verdict": "complete_circular_positive_bound_relation_protocol_ready",
            },
        ],
        "same_verdict_retirements": {
            "exp7686": {
                "applied": False,
                "mechanism": "usage-limit external block; current invocation succeeded",
            },
            "exp7676": {
                "applied": verdict_class == "disqualified",
                "mechanism": "whole-record quote equality",
            },
            "exp7672": {
                "applied": verdict_class == "circular_positive",
                "mechanism": "standalone fixture-only bound-relation benefit claim",
            },
        },
    }


def run_experiment(date: str, output: Path) -> int:  # pragma: no cover - exercised by CLI E2E
    """Validate first, checkpoint fixed units, then publish only checked bytes."""

    started = time.monotonic()
    progress(started, "preconditions", "start")
    root = ROOT.resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []
    phase_start = started

    def boundary(name: str, units: int = 0) -> None:
        nonlocal phase_start
        now = time.monotonic()
        spans.append(
            {
                "phase": name,
                "start_s": phase_start - started,
                "end_s": now - started,
                "duration_s": now - phase_start,
                "completed_units": units,
                "heartbeat_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        phase_start = now
        progress(started, name, "end", units)

    checks = preconditions(root)
    if any(not row["passed"] for row in checks):
        boundary("preconditions")
        atomic_json(
            output,
            build_artifact([], [], checks, [], date, time.monotonic() - started, spans=spans),
        )
        return 0
    pilots = [json.loads(line) for line in (root / PILOT).read_text(encoding="utf-8").splitlines()]
    try:
        panel = freeze_panel(pilots)
    except ValueError as error:
        checks.append(
            _check(
                "pilot_authentication",
                "exp7602",
                str(PILOT),
                "source_answer_hashes",
                "==",
                True,
                str(error),
            )
        )
        boundary("preconditions")
        atomic_json(
            output,
            build_artifact([], [], checks, [], date, time.monotonic() - started, spans=spans),
        )
        return 0
    boundary("preconditions", len(panel))

    progress(started, "scope", "start")
    atomic_json(raw / "feature_schema.json", feature_schema())
    panel_path = raw / "panel.jsonl"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=raw, delete=False) as stream:
        for unit in panel:
            stream.write(json.dumps(unit, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary_panel = Path(stream.name)
    temporary_panel.replace(panel_path)
    boundary("scope", len(panel))

    progress(started, "validation", "before_subprocess")
    private = Path(tempfile.mkdtemp(prefix="exp7700-validation-", dir="/tmp"))
    (private / "basetemp").mkdir()
    outcome = validation.run_scoped_validation(
        root,
        SCOPE["tests"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage",
        log_dir=raw / "validation_logs",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    receipts = outcome["validation_receipts"]
    progress(started, "validation", "after_subprocess", len(receipts))
    boundary("validation", len(receipts))
    if not outcome["required_checks_passed"]:
        atomic_json(
            output,
            build_artifact(
                [], panel, checks, receipts, date, time.monotonic() - started, spans=spans
            ),
        )
        return 0

    progress(started, "measurement", "start")
    rows: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    for unit in panel:
        rows.extend(build_rows([unit]))
        if len(rows) % 16 == 0:
            atomic_json(
                raw / "checkpoint.json", {"completed_units": len(rows) // 2, "row_count": len(rows)}
            )
        if time.monotonic() - last_heartbeat >= 60:
            progress(started, "measurement", "heartbeat", len(rows) // 2)
            last_heartbeat = time.monotonic()
    boundary("measurement", len(panel))
    artifact = build_artifact(
        rows, panel, checks, receipts, date, time.monotonic() - started, spans=spans
    )
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)

    progress(started, "terminal", "before_subprocess")
    python = str(root / ".venv/bin/python")

    def terminal_commands(path: Path) -> list[validation.CommandSpec]:
        return [
            validation.CommandSpec(
                "cold_reduction",
                (
                    python,
                    "-m",
                    "carnot.experiment_7700_v671_record_span_protocol",
                    "--cold-reduce",
                    str(path),
                    "--panel",
                    str(panel_path),
                ),
                "raw_rows",
                300,
            ),
            validation.CommandSpec(
                "adversarial_verify",
                (python, "scripts/adversarial_verify.py", "--json", str(path)),
                "exact_candidate",
                300,
            ),
            validation.CommandSpec(
                "strict_row_lint",
                (python, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
                "exact_candidate",
                300,
            ),
        ]

    preliminary = validation.run_commands(
        root,
        terminal_commands(candidate),
        log_dir=raw / "terminal_preliminary",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    progress(started, "terminal", "after_subprocess", len(preliminary))
    artifact = build_artifact(
        rows,
        panel,
        checks,
        receipts,
        date,
        time.monotonic() - started,
        spans=spans,
        terminal={
            "preliminary": preliminary,
            "exact_receipts_path": str(RAW / "terminal_exact_receipts.json"),
        },
    )
    if not all(row["passed"] for row in preliminary):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_terminal_readers"
        artifact["record_protocol_ready_score"] = 0
        artifact["flagged_adversarial"] = not preliminary[1]["passed"]
        for gate in artifact["acceptance_gate_results"]:
            if gate["gate"] in {"validity", "readiness"}:
                gate["passed"] = False
    atomic_json(candidate, artifact)
    exact = validation.run_commands(
        root,
        terminal_commands(candidate),
        log_dir=raw / "terminal_exact",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    atomic_json(
        raw / "terminal_exact_receipts.json",
        {"candidate_sha256": sha256_file(candidate), "receipts": exact},
    )
    if not all(row["passed"] for row in exact):
        progress(started, "terminal", "failed", len(exact))
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, artifact)
    if sha256_file(output) != sha256_file(candidate):
        raise ValueError("published_candidate_bytes_drifted")
    progress(started, "publication", "complete", len(panel))
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exercised by CLI E2E
    """Dispatch the bounded experiment or an independent cold replay."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--panel", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        if args.panel is None:
            parser.error("--panel is required with --cold-reduce")
        print(json.dumps(cold_reduce(args.cold_reduce, args.panel)), flush=True)
        return 0
    return run_experiment(args.date, args.output)
