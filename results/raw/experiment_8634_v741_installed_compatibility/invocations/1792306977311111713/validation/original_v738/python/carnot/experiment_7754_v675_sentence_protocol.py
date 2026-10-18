"""Qualify the exposed sentence protocol without changing its cohort (REQ-REPORT-7754)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

from carnot import experiment_7740_v674_sentence_label_protocol as prior
from carnot.experiment_7727_v673_development_corpus import COUNTS
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7754_v675_sentence_protocol"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
SOURCE = ROOT / "results/experiment_7727_v673_development_corpus.json"
MANIFEST = ROOT / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"
SCOPE = (
    json.loads((RAW / "frozen_affected_scope.json").read_text())
    if (RAW / "frozen_affected_scope.json").is_file()
    else {}
)
PRINCIPLES = {
    "experiment_id": "An artifact must have a unique current owner.",
    "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
    "verdict_class": "The claim class travels with the evidence.",
    "flagged_adversarial": "Invalid evidence must not open downstream gates.",
    "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
    "rows": "Aggregates must be recomputable without rerunning science.",
    "acceptance_gate_results": "A working protocol is not evidence of benefit.",
    "duration_s": "Duration must describe actual work without padding.",
    "phase_spans": "Duration must describe actual work without padding.",
    "random_seed": "A third party needs the same experiment inputs.",
    "reproducibility_checksum": "A third party needs the same experiment inputs.",
    "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
    "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
    "preconditions_checked": "Access and validity must be established before expensive work.",
    "validation_receipts": "All registered checks must pass before readiness opens.",
    "verifier_is_oracle": "Execution truth and independent semantic verification are distinct claims.",
    "claim_scope": "Execution truth and independent semantic verification are distinct claims.",
    "inference_substrate": "Duration floors must match the invoked substrate.",
    "inference_substrate_class": "Duration floors must match the invoked substrate.",
    "MODEL_SPECS": "A cited upstream model is not a current model invocation.",
    "model_specs": "A cited upstream model is not a current model invocation.",
    "model_invocation_counts": "A cited upstream model is not a current model invocation.",
    "sentence_protocol_ready_score": "An execution defect must be corrected before scientific use.",
    "sentence_protocol_manifest_path": "Every downstream label must bind to the same answer bytes.",
    "annotation_coverage_rows": "Missing annotations must remain visible.",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush each boundary so an operator can see actual completed work."""
    print(
        f"[exp7754] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def prepare_basetemp(base: Path) -> None:
    """Pytest removes its leaf directory but requires each parent to exist."""
    base.mkdir(parents=True, exist_ok=True)


def feature_row(public: dict[str, Any]) -> dict[str, Any]:
    """Build source and answer features from public bytes alone."""
    return prior._feature_row(public)


def map_byte_targets(answer: bytes, annotations: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Retain old target semantics and bind Unicode spans to original UTF-8 bytes."""
    result = prior.map_targets(answer, annotations)
    text = answer.decode("utf-8", "strict")
    positions = [len(text[:index].encode("utf-8")) for index in range(len(text) + 1)]
    result["sentence_byte_offsets"] = [
        [positions[a], positions[b]] for a, b in result["char_offsets"]
    ]
    result["annotation_byte_offsets"] = (
        None
        if annotations is None
        else [[positions[item["start"]], positions[item["end"]]] for item in annotations]
    )
    return result


def _check(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep the actual operand when a declared producer fails custody."""
    return {
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_hash": sha256_file(path) if path.is_file() else None,
        "field": field,
        "expected": expected,
        "observed": observed,
        "operator": "==",
        "passed": expected == observed,
    }


def preconditions(
    manifest_path: Path, fixture: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Verify the declared producer separately from the conductor pre-gate."""
    checks: list[dict[str, Any]] = []
    if not fixture:
        exists = SOURCE.is_file()
        checks.append(_check("exp7727", SOURCE, "is_file", True, exists))
        if exists:
            source = json.loads(SOURCE.read_text())
            checks.append(
                _check(
                    "exp7727",
                    SOURCE,
                    "development_cohort_ready_score",
                    1,
                    source.get("development_cohort_ready_score"),
                )
            )
            checks.append(
                _check(
                    "exp7727",
                    SOURCE,
                    "development_manifest_sha256",
                    sha256_file(manifest_path) if manifest_path.is_file() else None,
                    source.get("development_manifest_sha256"),
                )
            )
        gate = ROOT / "results/experiment_7753_v675_contract_methods.json"
        checks.append(_check("exp7753_conductor_pre_gate", gate, "is_file", True, gate.is_file()))
    checks.append(_check("exp7727", manifest_path, "is_file", True, manifest_path.is_file()))
    if any(not item["passed"] for item in checks):
        return checks, [], {}, []
    public, manifest, shard_checks = prior.authenticate(
        manifest_path, {role: 1 for role in COUNTS} if fixture else COUNTS
    )
    checks.extend(shard_checks)
    return checks, public, manifest, shard_checks


def seal_offsets(
    raw: Path,
    manifest_path: Path,
    public: list[dict[str, Any]],
    manifest: dict[str, Any],
    start: float,
) -> str:
    """Open evaluator annotations only after the public feature shard is sealed."""
    by_family = {row["family_id"]: row for row in public}
    offset_rows = []
    for role, meta in manifest["roles"].items():
        evaluator = prior._read_jsonl(manifest_path.parent / meta["evaluator_path"])
        for label in evaluator:
            row = by_family[label["family_id"]]
            mapped = map_byte_targets(row["complete_response"].encode(), label["annotations"])
            offset_rows.append(
                {
                    "family_id": row["family_id"],
                    "role": role,
                    "response_sha256": row["response_sha256"],
                    "sentence_byte_offsets": mapped["sentence_byte_offsets"],
                    "annotation_byte_offsets": mapped["annotation_byte_offsets"],
                }
            )
        progress(start, "offsets", role, len(offset_rows))
    path = raw / "byte_offsets.jsonl"
    digest = prior._jsonl(path, offset_rows)
    os.chmod(path, 0o600)
    evidence_path = raw / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["byte_offsets_sha256"] = digest
    atomic_json(evidence_path, evidence)
    return digest


def cold_reduce(raw: Path, candidate_path: Path) -> dict[str, Any]:
    """Recompute features and offsets from sealed inputs in a fresh process."""
    summary = prior.cold_reduce(raw, candidate_path)
    candidate = json.loads(candidate_path.read_text())
    evidence = json.loads((raw / "evidence_manifest.json").read_text())
    offset_path = raw / "byte_offsets.jsonl"
    if sha256_file(offset_path) != evidence["byte_offsets_sha256"]:
        raise ValueError("byte_offsets_sha256")
    manifest_path = Path(
        json.loads((raw / "sentence_protocol_manifest.json").read_text())[
            "development_manifest_path"
        ]
    )
    public, manifest, _ = prior.authenticate(
        manifest_path, candidate["sample_size_budget"]["role_counts"]
    )
    features = prior._read_jsonl(raw / "features.jsonl")
    offsets = prior._read_jsonl(offset_path)
    by_family = {row["family_id"]: row for row in public}
    labels_by_role = {
        role: {
            item["family_id"]: item
            for item in prior._read_jsonl(manifest_path.parent / meta["evaluator_path"])
        }
        for role, meta in manifest["roles"].items()
    }
    if len(offsets) != len(public):
        raise ValueError("offset_count")
    for feature in features:
        if feature != feature_row(by_family[feature["family_id"]]):
            raise ValueError("feature_recomputation_mismatch")
    for index, offset in enumerate(offsets, 1):
        row = by_family[offset["family_id"]]
        role = offset["role"]
        label = labels_by_role[role][row["family_id"]]
        mapped = map_byte_targets(row["complete_response"].encode(), label["annotations"])
        if offset != {
            "family_id": row["family_id"],
            "role": row["role"],
            "response_sha256": row["response_sha256"],
            "sentence_byte_offsets": mapped["sentence_byte_offsets"],
            "annotation_byte_offsets": mapped["annotation_byte_offsets"],
        }:
            raise ValueError("byte_offset_mapping_mismatch")
        if index % 64 == 0:
            print(f"[exp7754] cold_replay heartbeat completed={index}", flush=True)
    return summary


def _blocked(
    date: str, raw: Path, output: Path, checks: list[dict[str, Any]], origin: float
) -> dict[str, Any]:
    """Publish a terminal external block with the failed exact operands."""
    value: dict[str, Any] = {
        "experiment_id": "exp7754-sentence-protocol",
        "milestone": "2026.09.675",
        "run_date": date,
        "honest_verdict": "complete_blocked_external_prerequisite",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": [c for c in checks if not c["passed"]],
        "rows": [],
        "annotation_coverage_rows": [],
        "acceptance_gate_results": {
            k: (False if k == "validity" else None)
            for k in (
                "validity",
                "readiness",
                "probability_quality",
                "decision_benefit",
                "retention",
                "efficiency",
            )
        },
        "sample_size_budget": {
            "intended": sum(COUNTS.values()),
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "excluded": 0,
            "censored": 0,
            "effective_independent_N": 0,
        },
        "duration_s": time.monotonic() - origin,
        "phase_spans": [],
        "random_seed": {"cohort_selection": "Exp7727 frozen salt"},
        "reproducibility_checksum": prior.digest(json.dumps(checks, sort_keys=True).encode()),
        "source_artifact_hashes": [],
        "preconditions_checked": checks,
        "validation_receipts": {"affected_scope": SCOPE, "commands": [], "cold_replay": None},
        "verifier_is_oracle": False,
        "claim_scope": {"value": "development_only", "fresh_generalization_eligible": False},
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            k: 0
            for k in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "sentence_protocol_ready_score": 0,
        "sentence_protocol_manifest_path": None,
        "field_principles": PRINCIPLES,
    }
    atomic_json(raw / "terminal_candidate.json", value)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, value)
    return value


def run_experiment(
    manifest_path: Path,
    raw: Path,
    output: Path,
    date: str,
    *,
    fixture: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Capture one task-owned receipt and publish only after required checks."""
    origin = time.monotonic()
    progress(origin, "preconditions", "begin")
    if date != "20260927":
        raise ValueError("run_date")
    raw.mkdir(parents=True, exist_ok=True)
    scope = SCOPE
    atomic_json(raw / "frozen_affected_scope.json", scope)
    checks, public, manifest, _ = preconditions(manifest_path, fixture)
    if any(not item["passed"] for item in checks):
        progress(origin, "preconditions", "blocked", 0)
        return _blocked(date, raw, output, checks, origin)
    protocol = prior._protocol(manifest_path, manifest, scope)
    protocol["schema"] = "carnot.exp7754.sentence_protocol.v1"
    protocol["output_annotation_authority"] = "human output spans; no gold source alignment"
    protocol["label_open_chronology"] = [
        "authenticate shard hashes",
        "freeze public features",
        "open fit evaluator for supervised fit",
        "open tune for selection",
        "open policy for diagnostics",
        "open online_update on admitted update",
        "open online_admission for admission",
        "open evaluation and retention only for final evaluation",
    ]
    protocol["evaluator_paths"] = {
        role: str(manifest_path.parent / meta["evaluator_path"])
        for role, meta in manifest["roles"].items()
    }
    protocol_path = raw / "sentence_protocol_manifest.json"
    atomic_json(protocol_path, protocol)
    spans = [
        prior._span(
            "preconditions", origin, time.monotonic(), origin, date, len(checks), protocol_path
        )
    ]
    progress(origin, "preconditions", "complete", len(checks))
    phase = time.monotonic()
    progress(origin, "capture", "begin")
    coverage, evidence = prior.capture(manifest_path, raw, public, manifest, origin)
    offsets_hash = seal_offsets(raw, manifest_path, public, manifest, origin)
    protocol["byte_offsets_path"] = str(raw / "byte_offsets.jsonl")
    protocol["byte_offsets_sha256"] = offsets_hash
    protocol["features_path"] = str(raw / "features.jsonl")
    protocol["features_sha256"] = evidence["features_sha256"]
    atomic_json(protocol_path, protocol)
    spans.append(
        prior._span(
            "capture",
            phase,
            time.monotonic(),
            origin,
            date,
            len(public),
            raw / "evidence_manifest.json",
        )
    )
    progress(origin, "capture", "complete", len(public))
    receipts: list[dict[str, Any]] = []
    if validate:
        phase = time.monotonic()
        progress(origin, "validation", "begin")
        private = Path("/tmp") / f"carnot-7754-{os.getpid()}"
        base = private / "basetemp"
        prepare_basetemp(base)
        commands = build_scoped_commands(
            ROOT,
            scope["test_paths"],
            scope["changed_modules"],
            static_paths=scope["static_paths"],
            basetemp=base,
            coverage_file=private / ".coverage",
        )
        commands.append(
            CommandSpec(
                "full_python_suite",
                (
                    str(ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={base / 'full'}",
                    "tests/python",
                    "-q",
                ),
                "all_python_tests",
                3600,
            )
        )
        receipts = run_commands(
            ROOT,
            commands,
            log_dir=raw / "validation_logs",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        spans.append(
            prior._span(
                "validation",
                phase,
                time.monotonic(),
                origin,
                date,
                len(receipts),
                raw / "frozen_affected_scope.json",
            )
        )
        progress(origin, "validation", "complete", len(receipts))
    result = prior._artifact(
        date,
        raw,
        manifest_path,
        public,
        checks,
        coverage,
        evidence,
        scope,
        spans,
        fixture,
        receipts,
    )
    result.update(
        {
            "experiment_id": "exp7754-sentence-protocol",
            "milestone": "2026.09.675",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "duration_s": time.monotonic() - origin,
            "field_principles": PRINCIPLES,
            "preconditions_checked": checks
            + [
                {
                    "check": "host_resources",
                    "cpu_count": os.cpu_count(),
                    "host": platform.node(),
                    "backend": "cpu",
                }
            ],
            "reproducibility_checksum": prior.digest(
                json.dumps(
                    {
                        "inputs": result["source_artifact_hashes"]["eligible_producers"],
                        "protocol": sha256_file(protocol_path),
                        "code": sha256_file(Path(__file__)),
                        "roles": manifest["counts"],
                    },
                    sort_keys=True,
                ).encode()
            ),
        }
    )
    result["source_artifact_hashes"]["imported_sources"] = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "date": date,
            "imported_fields": fields,
            "eligible": True,
        }
        for path, fields in (
            (manifest_path, ["counts", "roles", "hashes"]),
            (SOURCE, ["development_cohort_ready_score", "development_manifest_sha256"]),
        )
        if path.is_file()
    ]
    result["source_artifact_hashes"]["conductor_pre_gate_receipt"] = {
        "path": str(ROOT / "results/experiment_7753_v675_contract_methods.json"),
        "sha256": sha256_file(ROOT / "results/experiment_7753_v675_contract_methods.json")
        if (ROOT / "results/experiment_7753_v675_contract_methods.json").is_file()
        else None,
        "imported_fields": ["contract_ready_score"],
        "producer_role": False,
    }
    for row in result["rows"]:
        row["raw_paths"] = {
            "features": str(raw / "features.jsonl"),
            "targets": str(raw / f"{row['role']}_targets.jsonl"),
            "byte_offsets": str(raw / "byte_offsets.jsonl"),
        }
    result["validation_receipts"]["private_basetemp"] = str(base) if validate else None
    result["validation_receipts"]["child_directory_regression"] = "test_child_basetemp_regression"
    result["validation_receipts"]["full_python_suite"] = next(
        (r for r in receipts if r["name"] == "full_python_suite"), None
    )
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, result)
    if validate:
        phase = time.monotonic()
        terminal = [
            CommandSpec(name, argv, "exact_candidate", 900)
            for name, argv in (
                (
                    "cold_replay",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        "-m",
                        f"carnot.{NAME}",
                        "--cold-reduce",
                        str(raw),
                        "--candidate",
                        str(candidate),
                    ),
                ),
                (
                    "adversarial_verify",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/adversarial_verify.py",
                        "--json",
                        str(candidate),
                    ),
                ),
                (
                    "verdict_row_consistency_strict",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ),
                ),
            )
        ]
        progress(origin, "terminal", "begin")
        terminal_receipts = run_commands(
            ROOT,
            terminal,
            log_dir=raw / "terminal_logs",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        result["validation_receipts"]["terminal_commands"] = terminal_receipts
        result["validation_receipts"]["cold_replay"] = terminal_receipts[0]
        result["validation_receipts"]["terminal_checks_path"] = str(raw / "terminal_checks.json")
        spans.append(
            prior._span(
                "terminal", phase, time.monotonic(), origin, date, len(terminal_receipts), candidate
            )
        )
        result["duration_s"] = time.monotonic() - origin
        failed = [item for item in [*receipts, *terminal_receipts] if not item["passed"]]
        if failed:
            result["verdict_class"] = "disqualified"
            result["honest_verdict"] = "complete_disqualified_required_validation"
            result["sentence_protocol_ready_score"] = 0
            result["acceptance_gate_results"]["validity"] = False
            result["gate_check_summary"] = [
                _check(
                    NAME, Path(item["log_path"]), item["name"] + ".exit_code", 0, item["exit_code"]
                )
                for item in failed
            ]
        result["flagged_adversarial"] = not terminal_receipts[1]["passed"]
        atomic_json(candidate, result)
        final_receipts = run_commands(
            ROOT,
            terminal,
            log_dir=raw / "final_terminal_logs",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        atomic_json(
            raw / "terminal_checks.json",
            {"candidate_sha256": sha256_file(candidate), "commands": final_receipts},
        )
        if not all(item["passed"] for item in final_receipts):
            result["verdict_class"] = "disqualified"
            result["honest_verdict"] = "complete_disqualified_final_reader"
            result["sentence_protocol_ready_score"] = 0
            result["flagged_adversarial"] = not final_receipts[1]["passed"]
            atomic_json(candidate, result)
        progress(origin, "terminal", "complete", len(final_receipts))
    else:
        cold_reduce(raw, candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(candidate.read_bytes())
    os.replace(temporary, output)
    progress(origin, "publish", "complete", len(public))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the dated producer or verify an already sealed candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce is not None:
        if args.candidate is None:
            parser.error("--candidate is required for cold reduction")
        cold_reduce(args.cold_reduce, args.candidate)
        print("cold reduction passed", flush=True)
        return 0
    fixture = args.fixture_manifest is not None
    result = run_experiment(
        args.fixture_manifest or MANIFEST,
        args.raw or RAW,
        args.output or OUTPUT,
        args.date,
        fixture=fixture,
        validate=not fixture,
    )
    return 0 if result["verdict_class"] in {"null", "circular_positive", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
