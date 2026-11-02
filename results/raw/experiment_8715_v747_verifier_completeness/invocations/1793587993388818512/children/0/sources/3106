"""V668 CPU orchestration for native tool-source evidence atoms.

Fixture truth is written before calling the extractor. It is an exact oracle
for this bounded protocol, never a model-accuracy or learned-verifier result.
Spec: REQ-REPORT-7658 and SCENARIO-REPORT-7658-TERMINAL.
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

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)
from carnot.reporting.experiment_7646_source_features import validate_model_row
from carnot.verify.source_claim_witness import verify_claim
from carnot.verify.tool_source_atoms import digest, parse_atoms, replay, verify_answer


ROOT = Path(__file__).resolve().parents[2]
PILOT = Path("results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl")
RAW = Path("results/raw/experiment_7658_v668_evidence_atoms")
SCHEMA = RAW / "schema.json"
MODEL_SPECS: list[dict[str, Any]] = []


def fixture_cases() -> list[dict[str, str]]:
    """Return independently assigned truth for 64 distinct source units."""
    cases = []
    for dialect in ("plain", "numbered", "grep", "stack"):
        for index in range(16):
            name = f"alpha_{index}"
            path = f"pkg{index}/mod.py"
            line = 1 if index < 8 else 9
            if dialect == "plain":
                source = f'Tool output:\n```\ndef {name}():\n    return "café {index}"\n```'
                answer = f"`{name}` is defined at line {line}."
            elif dialect == "numbered":
                source = f'Tool output:\n```\n1: def {name}():\n2:     return "café {index}"\n```'
                answer = f"`{name}` is defined at line {line}."
            elif dialect == "grep":
                source = f"Tool output:\n```\n{path}:7: {name} = 'café'\nother{index}/mod.py:7: sentinel = 1\n```"
                answer = f"`{path}` at line {7 if index < 8 else 9}."
            else:
                source = f"Tool output:\n```\nError: café {index}\n    at handler ({path}:5:2)\n```"
                answer = (
                    f"The frame is at `{path}:5:2`."
                    if index < 8
                    else f"The frame is at `{path}:9:2`."
                )
            cases.append(
                {
                    "id": f"{dialect}-{index:02d}",
                    "dialect": dialect,
                    "source": source,
                    "answer": answer,
                    "truth": "observed" if index < 8 else "unknown",
                }
            )
    cases.extend(
        [
            {
                "id": "plain-nested-scope",
                "dialect": "plain",
                "source": "```\nclass Outer:\n    def inner(self):\n        pass\n```",
                "answer": "`inner` is defined at line 2.",
                "truth": "observed",
            },
            {
                "id": "plain-negation",
                "dialect": "plain",
                "source": "```\ndef alpha():\n    pass\n```",
                "answer": "`alpha` is not defined at line 1.",
                "truth": "unknown",
            },
            {
                "id": "numbered-wrong-range",
                "dialect": "numbered",
                "source": "```\n1: def alpha():\n2:     pass\n```",
                "answer": "`alpha` is defined at lines 2-3.",
                "truth": "scoped_contradiction",
            },
            {
                "id": "grep-duplicate-basename",
                "dialect": "grep",
                "source": "```\na/mod.py:7: x = 1\nb/mod.py:7: y = 2\n```",
                "answer": "`mod.py` at line 7.",
                "truth": "unknown",
            },
            {
                "id": "grep-source-swap",
                "dialect": "grep",
                "source": "```\nb/mod.py:7: x = 1\n```",
                "answer": "`a/mod.py` at line 7.",
                "truth": "unknown",
            },
            {
                "id": "grep-omitted-record",
                "dialect": "grep",
                "source": "```\na/mod.py:7: x = 1\n```",
                "answer": "`a/mod.py` at line 9.",
                "truth": "unknown",
            },
            {
                "id": "stack-duplicate-response",
                "dialect": "stack",
                "source": "```\n    at one (a/mod.js:5:2)\n```\n```\n    at two (b/mod.js:9:2)\n```",
                "answer": "`b/mod.js:9:2`",
                "truth": "unknown",
            },
            {
                "id": "stack-truncated-source",
                "dialect": "stack",
                "source": "```\n    at one (a/mod.js:5:2)",
                "answer": "`a/mod.js:5:2`",
                "truth": "unknown",
            },
        ]
    )
    return cases


def pilot_rows(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve all original groups, even when their prose is unsupported."""
    rows = []
    for index, row in enumerate(inputs):
        validate_model_row(row)
        source, answer = row["complete_source"], row["complete_answer"]
        old = [verify_claim(source, part["text"])["status"] for part in row["answer_sentences"]]
        atoms = parse_atoms(source)
        checked = verify_answer(source, answer)
        replay(source, answer, row["source_sha256"], row["answer_sha256"], atoms, checked)
        rows.append(
            {
                "unit_id": row["component_hash"],
                "arm": "original_source",
                "pilot_index": index,
                "dialect": atoms[0]["dialect"] if atoms else "unknown",
                "source_sha256": row["source_sha256"],
                "answer_sha256": row["answer_sha256"],
                "old_checked": sum(status != "unknown" for status in old),
                "checked_atoms": checked["checked_propositions"],
                "source_atom_count": len(atoms),
                "witnesses": checked["witnesses"],
                "whole_answer_certified": False,
                "excluded": False,
                "censored": checked["checked_propositions"] == 0,
                "raw_metrics": {"checked_propositions": checked["checked_propositions"]},
                "provenance": str(PILOT),
            }
        )
        print(
            f"[exp7658] pilot_unit={index + 1}/8 checked={checked['checked_propositions']}",
            flush=True,
        )
    return rows


def fixture_rows(cases: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Keep oracle truth, original spans, and the residual answer separately."""
    rows = []
    for index, case in enumerate(cases):
        atoms = parse_atoms(case["source"])
        result = verify_answer(case["source"], case["answer"])
        replay(
            case["source"],
            case["answer"],
            digest(case["source"]),
            digest(case["answer"]),
            atoms,
            result,
        )
        rows.append(
            {
                "unit_id": case["id"],
                "arm": "original_source",
                "dialect": case["dialect"],
                "truth": case["truth"],
                "observed": result["status"],
                "source_sha256": digest(case["source"]),
                "answer_sha256": digest(case["answer"]),
                "original_claim_spans": [
                    [item["byte_start"], item["byte_end"]] for item in result["witnesses"]
                ],
                "evidence_spans": [item["evidence_span"] for item in result["witnesses"]],
                "completeness": [item["completeness"] for item in result["witnesses"]],
                "checked_proposition": [item["kind"] for item in result["witnesses"]],
                "residual_unverified_text": result["residual_unverified_text"],
                "excluded": False,
                "censored": result["status"] == "unknown",
                "raw_metrics": {"checked_propositions": result["checked_propositions"]},
                "provenance": "independent_fixture_truth_v1",
            }
        )
        if (index + 1) % 8 == 0:
            print(f"[exp7658] fixture_unit={index + 1}/{len(cases)}", flush=True)
    return rows


def cold_reduce(candidate: Path) -> dict[str, Any]:
    """Rebuild every row from authenticated bytes in a fresh process."""
    artifact = json.loads(candidate.read_text())
    input_path = ROOT / PILOT
    for label, expected in artifact["source_artifact_hashes"]["producers"].items():
        if sha256_file(ROOT / label) != expected:
            raise ValueError("source_hash_mismatch")
    inputs = [json.loads(line) for line in input_path.read_text().splitlines()]
    rebuilt_pilots = pilot_rows(inputs)
    rebuilt_fixtures = fixture_rows(fixture_cases())
    if artifact["rows"] != rebuilt_pilots + rebuilt_fixtures:
        raise ValueError("row_reduction_mismatch")
    if len(rebuilt_pilots) != 8 or len(rebuilt_fixtures) < 64:
        raise ValueError("sample_size_mismatch")
    return {"passed": True, "pilot_groups": 8, "fixture_groups": len(rebuilt_fixtures)}


def _progress(phase: str, event: str, start: float, detail: str = "") -> None:
    print(
        f"[exp7658] {phase} {event} elapsed_s={time.monotonic() - start:.3f} {detail}", flush=True
    )


def _gate(
    name: str, passed: bool | None, operands: dict[str, Any], principle: str
) -> dict[str, Any]:
    return {"gate": name, "passed": passed, "measured_operands": operands, "principle": principle}


def _artifact(
    rows: list[dict[str, Any]],
    fixtures: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    start: float,
    run_date: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build a terminal candidate with separate validity and science gates."""
    pilots = rows[:8]
    false_certified = sum(
        row["observed"] == "observed" and row["truth"] != "observed" for row in fixtures
    )
    covered = sum(row["checked_atoms"] > 0 for row in pilots)
    required = reduce_required_checks(receipts)
    ready = not any(not check["passed"] for check in checks) and (
        len(fixtures) >= 64 and false_certified == 0 and covered >= 4
    )
    valid = ready and required["required_checks_passed"]
    failed_inputs = [check for check in checks if not check["passed"]]
    kind = "blocked" if failed_inputs else "circular_positive" if valid else "disqualified"
    verdict = {
        "blocked": "complete_blocked_missing_external_evidence",
        "circular_positive": "complete_circular_positive_fixture_protocol_ready",
        "disqualified": "complete_disqualified_required_validation",
    }[kind]
    gates = [
        _gate(
            "validity",
            valid,
            {
                "required_checks_passed": required["required_checks_passed"],
                "failed_inputs": len(failed_inputs),
            },
            "Authenticated inputs and required checks govern validity.",
        ),
        _gate(
            "readiness",
            ready and valid,
            {
                "fixture_groups": len(fixtures),
                "false_certified": false_certified,
                "covered_pilot_groups": covered,
            },
            "Fixture soundness and four covered pilots permit narrow protocol readiness.",
        ),
        _gate(
            "coverage",
            covered >= 4 and valid,
            {"covered_pilot_groups": covered, "intended": 8},
            "Only original pilot groups count toward coverage.",
        ),
        _gate(
            "probability_benefit",
            None,
            {"paired_probabilities": 0},
            "Proper loss needs independent labels and paired probabilities.",
        ),
        _gate(
            "utility", None, {"typed_decisions": 0}, "Utility needs actions, costs, and outcomes."
        ),
        _gate(
            "retention",
            None,
            {"delayed_feedback_events": 0},
            "Retention needs delayed feedback and later replay.",
        ),
        _gate(
            "freshness",
            None,
            {"fresh_independent_groups": 0},
            "Previously exposed pilots and fixtures are not fresh confirmation.",
        ),
    ]
    producer_paths = [
        PILOT,
        Path("results/experiment_7644_v667_source_witness_prototype.json"),
        Path("results/experiment_7646_v667_source_feature_corpus.json"),
    ]
    hashes = {
        "producers": {
            str(path): sha256_file(ROOT / path)
            for path in producer_paths
            if (ROOT / path).is_file()
        },
        "pre_gate_receipts": {},
        "missing_inputs": [],
    }
    checksum = digest(
        json.dumps(
            {
                "input_hashes": hashes,
                "manifest": manifest,
                "reducer_sha256": sha256_file(Path(__file__)),
            },
            sort_keys=True,
        )
    )
    return {
        "schema": "carnot.exp7658.v668.evidence_atoms.v1",
        "experiment_id": "exp7658-v668-evidence-atoms",
        "milestone": "2026.09.668",
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": kind,
        "flagged_adversarial": False,
        "gate_check_summary": failed_inputs,
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended_independent_groups": 8 + len(fixtures),
            "observed_independent_groups": len(rows),
            "eligible": len(rows),
            "excluded": 0,
            "censored": sum(row["censored"] for row in rows),
            "prior_exposure": "Eight pilots were previously exposed; fixture truth is an exact oracle.",
            "claim_limits": "No natural-answer accuracy, probability, utility, retention, or freshness inference.",
        },
        "inference_substrate": "deterministic_tool_source_atoms_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no model used or planned for current CPU work",
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in ("loads", "forwards", "generations", "tokens", "attempted", "cancelled")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": platform.node(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - start,
        "random_seed": {"fixture_generation": "deterministic indexed cases; no random sampling"},
        "reproducibility_checksum": checksum,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": manifest,
            "required_commands": receipts,
            **required,
            "unrelated_repository_suite_debt": [],
        },
        "verifier_is_oracle": True,
        "field_principles": {
            "honest_verdict": "Completion does not imply scientific benefit.",
            "verdict_class": "Required validation and terminal readers govern class.",
            "rows": "One original-source row per independent group; no repeated views enlarge the sample.",
            "atom_protocol_ready_score": "Sound fixtures, replay, pilot coverage and readers govern readiness.",
            "flagged_adversarial": "A flagged reader cannot open a gate.",
            "duration_s": "Measured current CPU time only.",
            "inference_substrate_class": "No model is loaded in current work.",
            "sample_size_budget": "Pilot and fixture units are distinct but oracle fixtures do not measure natural accuracy.",
        },
        "atom_protocol_ready_score": int(valid),
        "atom_schema_path": str(SCHEMA),
        "fixture_rows": fixtures,
        "pilot_coverage": [
            {
                "unit_id": row["unit_id"],
                "dialect": row["dialect"],
                "checked_atom_count": row["checked_atoms"],
            }
            for row in pilots
        ],
        "whole_answer_certified": False,
        "historical_model_id": "unsloth/Qwen3.8-27B-GGUF",
        "retirement": "Retire unchanged V667 In-file grammar; native tool-source atoms are the new mechanism.",
    }


def run_experiment(run_date: str, output: Path) -> dict[str, Any]:  # pragma: no cover - E2E
    """Checkpoint each unit, stream checks, and publish one terminal record."""
    start = time.monotonic()
    root = ROOT.resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []

    def phase(name: str, begin: float, units: int, checkpoint: Path) -> None:
        end = time.monotonic() - start
        spans.append(
            {
                "phase": name,
                "start_offset_s": begin,
                "end_offset_s": end,
                "duration_s": end - begin,
                "completed_units": units,
                "heartbeat_times_s": [begin, end],
                "checkpoint": str(checkpoint.relative_to(root)),
            }
        )
        _progress(name, "complete", start, f"units={units}")

    _progress("preconditions", "start", start, f"root={root}")
    begin = time.monotonic() - start
    required_inputs = [
        PILOT,
        Path("results/experiment_7644_v667_source_witness_prototype.json"),
        Path("results/experiment_7646_v667_source_feature_corpus.json"),
    ]
    checks = [
        {
            "check": "input_exists",
            "upstream": str(path),
            "path": str(path),
            "field": "exists",
            "operator": "==",
            "expected": True,
            "observed": (root / path).is_file(),
            "passed": (root / path).is_file(),
        }
        for path in required_inputs
    ]
    checks.append(
        {
            "check": "host_cpu_available",
            "upstream": "local",
            "path": "/proc/self",
            "field": "cpu_count",
            "operator": ">=",
            "expected": 1,
            "observed": os.cpu_count(),
            "passed": (os.cpu_count() or 0) >= 1,
        }
    )
    atomic_json(raw / "preconditions.json", checks)
    phase("preconditions", begin, len(checks), raw / "preconditions.json")
    tests = ["tests/python/test_experiment_7658_v668_evidence_atoms.py"]
    modules = [
        "python/carnot/verify/tool_source_atoms.py",
        "python/carnot/experiment_7658_v668_evidence_atoms.py",
    ]
    static = ["scripts/experiments/experiment_7658_v668_evidence_atoms.py"]
    with tempfile.TemporaryDirectory(prefix="exp7658-", dir="/tmp") as private_dir:
        private = Path(private_dir)
        commands = build_scoped_commands(
            root,
            tests,
            modules,
            static_paths=static,
            basetemp=private / "basetemp",
            coverage_file=private / ".coverage",
        )
        (private / "basetemp").mkdir()
        manifest = {
            "tests": tests,
            "changed_modules": modules,
            "static_paths": static,
            "commands": [{"name": item.name, "argv": list(item.argv)} for item in commands],
        }
        atomic_json(raw / "frozen_validation_manifest.json", manifest)
        _progress("source_atoms", "start", start)
        begin = time.monotonic() - start
        inputs = (
            [json.loads(line) for line in (root / PILOT).read_text().splitlines()]
            if checks[0]["passed"]
            else []
        )
        pilots = pilot_rows(inputs)
        fixtures = fixture_rows(fixture_cases())
        rows = pilots + fixtures
        for index, row in enumerate(rows):
            atomic_json(raw / "checkpoints" / f"{index:03d}.json", row)
        atomic_json(raw / "rows.json", rows)
        atomic_json(
            root / SCHEMA,
            {
                "schema": "carnot.exp7658.v668.evidence_atoms.v1",
                "atom_fields": [
                    "dialect",
                    "source_id",
                    "line",
                    "byte_start",
                    "byte_end",
                    "text",
                    "complete",
                    "definitions",
                    "scopes",
                ],
                "claim_fields": ["kind", "byte_start", "byte_end", "status", "evidence_span"],
            },
        )
        phase("source_atoms", begin, len(rows), raw / "rows.json")
        _progress("affected_validation", "before_subprocess", start)
        begin = time.monotonic() - start
        receipts = run_commands(
            root,
            commands,
            log_dir=raw / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        phase("affected_validation", begin, len(receipts), raw / "validation/affected")
        artifact = _artifact(rows, fixtures, checks, receipts, spans, start, run_date, manifest)
        candidate = raw / "exact_terminal_candidate.json"
        atomic_json(candidate, artifact)
        python = str(root / ".venv/bin/python")
        readers = [
            CommandSpec(
                "cold_reduce",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7658_v668_evidence_atoms",
                    "--cold-reduce",
                    str(candidate),
                ),
                "exact_candidate",
            ),
            CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "exact_candidate",
            ),
            CommandSpec(
                "verdict_row_consistency",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "exact_candidate",
            ),
        ]
        _progress("terminal_readers", "before_subprocess", start)
        begin = time.monotonic() - start
        terminal = run_commands(
            root,
            readers,
            log_dir=raw / "validation/terminal",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        phase("terminal_readers", begin, len(terminal), raw / "validation/terminal")
        artifact["validation_receipts"]["terminal_readers"] = terminal
        terminal_pass = all(item["passed"] for item in terminal)
        if not terminal_pass:
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["atom_protocol_ready_score"] = 0
            for gate in artifact["acceptance_gate_results"][:3]:
                gate["passed"] = False
        artifact["flagged_adversarial"] = not next(
            item["passed"] for item in terminal if item["name"] == "adversarial_verify"
        )
        atomic_json(candidate, artifact)
        _progress("exact_terminal_replay", "before_subprocess", start)
        begin = time.monotonic() - start
        exact = run_commands(
            root,
            readers,
            log_dir=raw / "validation/exact",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        atomic_json(raw / "exact_terminal_reader_outcomes.json", {"outcomes": exact})
        phase(
            "exact_terminal_replay", begin, len(exact), raw / "exact_terminal_reader_outcomes.json"
        )
        if not all(item["passed"] for item in exact):
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["atom_protocol_ready_score"] = 0
            for gate in artifact["acceptance_gate_results"][:3]:
                gate["passed"] = False
        artifact["terminal_reader_outcomes_path"] = str(RAW / "exact_terminal_reader_outcomes.json")
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - start
        _progress("publication", "before_atomic", start)
        atomic_json(output, artifact)
        _progress("publication", "after_atomic", start, artifact["honest_verdict"])
        return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - E2E
    """Select producer or independent cold-reduction mode."""
    print("[exp7658] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260925")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/experiment_7658_v668_evidence_atoms.json"
    )
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        print(json.dumps(cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    result = run_experiment(args.date, args.output)
    return 0 if result["verdict_class"] == "circular_positive" else 1


if __name__ == "__main__":  # pragma: no cover - E2E
    raise SystemExit(main())
