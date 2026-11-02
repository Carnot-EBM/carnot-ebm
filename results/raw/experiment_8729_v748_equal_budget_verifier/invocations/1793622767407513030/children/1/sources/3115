"""V669 CPU protocol for byte-bound source relations.

Fixture labels are fixed by construction and cannot establish answer accuracy.
Spec: REQ-REPORT-7672 and SCENARIO-REPORT-7672-FIXTURES/TERMINAL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify.tool_source_atoms import verify_answer
from carnot.verify.tool_source_relations import replay_relations, verify_relations


ROOT = Path(__file__).resolve().parents[2]
PILOT = Path("results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl")
RAW = Path("results/raw/experiment_7672_v669_bound_relations")
MODEL_SPECS: list[dict[str, Any]] = []
SCOPE = {
    "tests": ["tests/python/test_experiment_7672_v669_bound_relations.py"],
    "changed_modules": [
        "python/carnot/verify/tool_source_relations.py",
        "python/carnot/experiment_7672_v669_bound_relations.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7672_v669_bound_relations.py"],
    "specs": ["REQ-REPORT-7672", "REQ-VERIFY-7672"],
}
PRIMITIVES = [
    {
        "predicate": "stack_frame(function,path,line)",
        "applicability": "explicit stack frame",
        "contradiction_authority": "none for partial stacks",
        "unknown_rule": "missing, conflicting or truncated frame",
    },
    {
        "predicate": "grep_quote(path,line,text)",
        "applicability": "quoted visible grep row",
        "contradiction_authority": "none for partial search",
        "unknown_rule": "missing, conflicting or truncated row",
    },
    {
        "predicate": "definition_in_scope(name,scope,line)",
        "applicability": "parseable complete Python fence",
        "contradiction_authority": "complete local definition line",
        "unknown_rule": "unparsed, missing or ambiguous line",
    },
]


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _progress(phase: str, event: str, start: float, detail: str = "") -> None:
    print(
        f"[exp7672] {phase} {event} elapsed_s={time.monotonic() - start:.3f} {detail}", flush=True
    )


def fixture_cases() -> list[dict[str, str]]:
    """Freeze three dialects and eight attack types before any pilot inspection."""
    cases = []
    for dialect in ("stack", "grep", "ast"):
        for index in range(24):
            repeat, attack = divmod(index, 8)
            name = f"run{repeat}"
            path = f"pkg{repeat}/a.js" if dialect == "stack" else f"pkg{repeat}/a.py"
            other = f"pkg{repeat}/b.js" if dialect == "stack" else f"pkg{repeat}/b.py"
            if dialect == "stack":
                lines = [f" at {name} ({path}:5:2)", f" at other ({other}:9:1)"]
                answer = f"`{name}` at `{path}:5`."
                if attack == 1:
                    answer = f"`{name}` at `{other}:9`."
                elif attack == 2:
                    answer = f"`{name}` is not at `{path}:5`."
                elif attack == 3:
                    lines.append(f" at alias ({path}:5:2)")
                elif attack == 4:
                    lines = lines[1:]
                elif attack == 6:
                    lines.reverse()
                elif attack == 7:
                    answer = f"`{name}` at `{path}:5` might fail."
            elif dialect == "grep":
                lines = [f"{path}:7: 'café{repeat}'", f"{other}:7: 'other'"]
                answer = f"`{path}:7` contains 'café{repeat}'."
                if attack == 1:
                    answer = f"`{other}:7` contains 'café{repeat}'."
                elif attack == 2:
                    answer = f"`{path}:7` does not contain 'café{repeat}'."
                elif attack == 3:
                    lines.append(f"{path}:7: 'other'")
                elif attack == 4:
                    lines = lines[1:]
                elif attack == 6:
                    lines.reverse()
                elif attack == 7:
                    answer = f"`{path}:7` contains 'café{repeat}' because of a bug."
            else:
                lines = [
                    "class A:",
                    f"    def {name}(self):",
                    "        pass",
                    "class B:",
                    "    def stop(self):",
                    "        pass",
                ]
                answer = f"`{name}` is defined in `A` at line 2."
                if attack == 1:
                    answer = f"`{name}` is defined in `B` at line 2."
                elif attack == 2:
                    answer = f"`{name}` is not defined in `A` at line 2."
                elif attack == 3:
                    lines[1] = f"    def {name}(self): pass; def alias(self): pass"
                    lines.pop(2)
                elif attack == 4:
                    lines = ["class A:", "    pass"]
                elif attack == 6:
                    lines = [
                        "class B:",
                        "    def stop(self):",
                        "        pass",
                        "class A:",
                        f"    def {name}(self):",
                        "        pass",
                    ]
                    answer = f"`{name}` is defined in `A` at line 5."
                elif attack == 7:
                    answer = f"`{name}` is defined in `A` at line 2 and breaks startup."
            source = "```\n" + "\n".join(lines) + ("" if attack == 5 else "\n```")
            truth = (
                "contradicted"
                if dialect == "ast" and attack in {1, 2}
                else "supported"
                if attack in {0, 6}
                else "unknown"
            )
            cases.append(
                {
                    "id": f"{dialect}-{index:02d}",
                    "dialect": dialect,
                    "source": source,
                    "answer": answer,
                    "truth": truth,
                    "split": "held_out" if index >= 16 else "development",
                    "attack": str(attack),
                }
            )
    return cases


def _decision(source: str, answer: str) -> dict[str, Any]:
    """Compute both arms from identical bytes, without consulting truth."""
    bound = verify_relations(source, answer)
    replay_relations(source, answer, json.loads(json.dumps(bound)))
    old = verify_answer(source, answer)
    mapping = {
        "observed": "supported",
        "scoped_contradiction": "contradicted",
        "unknown": "unknown",
    }
    return {"bound": bound, "membership": mapping[old["status"]], "membership_detail": old}


def fixture_rows(cases: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Keep one row per group and arm; fixture truth stays outside predictor."""
    rows = []
    for case in cases:
        result = _decision(case["source"], case["answer"])
        for arm in ("membership_only", "bound_relation"):
            bound = result["bound"]
            observed = result["membership"] if arm == "membership_only" else bound["status"]
            rows.append(
                {
                    "unit_id": case["id"],
                    "arm": arm,
                    "population": "fixture",
                    "split": case["split"],
                    "dialect": case["dialect"],
                    "attack": case["attack"],
                    "truth": case["truth"],
                    "observed": observed,
                    "source_sha256": bound["source_sha256"],
                    "answer_sha256": bound["answer_sha256"],
                    "checked_spans": bound["checked_spans"] if arm == "bound_relation" else [],
                    "unknown_spans": bound["unknown_spans"] if arm == "bound_relation" else [],
                    "findings": bound["findings"]
                    if arm == "bound_relation"
                    else result["membership_detail"]["witnesses"],
                    "raw_metrics": {
                        "relations": len(bound["relations"]),
                        "checked": len(bound["checked_spans"]),
                        "unknown": len(bound["unknown_spans"]),
                    },
                    "excluded": False,
                    "censored": observed == "unknown",
                    "provenance": "exact_fixture_oracle",
                }
            )
    return rows


def pilot_findings(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inspect exposed bytes for failure types without opening evaluator labels."""
    findings = []
    for index, row in enumerate(inputs):
        source, answer = row["complete_source"], row["complete_answer"]
        if row["source_sha256"] != "sha256:" + hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("pilot_source_authentication_failure")
        if row["answer_sha256"] != "sha256:" + hashlib.sha256(answer.encode()).hexdigest():
            raise ValueError("pilot_answer_authentication_failure")
        decision = _decision(source, answer)
        bound = decision["bound"]
        findings.append(
            {
                "unit_id": row["component_hash"],
                "pilot_index": index,
                "prior_exposure": True,
                "source_sha256": bound["source_sha256"],
                "answer_sha256": bound["answer_sha256"],
                "membership_decision": decision["membership"],
                "bound_decision": bound["status"],
                "checked_spans": bound["checked_spans"],
                "unknown_spans": bound["unknown_spans"],
                "findings": bound["findings"],
                "error_categories": sorted(
                    {
                        finding["reason"]
                        for finding in bound["findings"]
                        if finding["status"] == "unknown"
                    }
                ),
                "fresh_accuracy_claim": False,
            }
        )
    return findings


def pilot_rows(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain both pilot arms while excluding them from fresh inference."""
    rows = []
    for finding in findings:
        for arm, key in (
            ("membership_only", "membership_decision"),
            ("bound_relation", "bound_decision"),
        ):
            observed = finding[key]
            rows.append(
                {
                    "unit_id": finding["unit_id"],
                    "arm": arm,
                    "population": "exposed_pilot",
                    "split": "prior_exposure",
                    "dialect": "mixed_native",
                    "truth": None,
                    "observed": observed,
                    "source_sha256": finding["source_sha256"],
                    "answer_sha256": finding["answer_sha256"],
                    "checked_spans": finding["checked_spans"] if arm == "bound_relation" else [],
                    "unknown_spans": finding["unknown_spans"] if arm == "bound_relation" else [],
                    "raw_metrics": {
                        "checked": len(finding["checked_spans"]),
                        "unknown": len(finding["unknown_spans"]),
                    },
                    "excluded": True,
                    "censored": observed == "unknown",
                    "provenance": str(PILOT),
                }
            )
    return rows


def cold_reduce(candidate: Path) -> dict[str, Any]:
    """Reload authenticated inputs and rebuild decisions in a fresh process."""
    value = json.loads(candidate.read_text())
    for label, expected in value["source_artifact_hashes"]["producers"].items():
        if sha256_file(ROOT / label) != expected:
            raise ValueError("source_hash_mismatch")
    cases = fixture_cases()
    inputs = [json.loads(line) for line in (ROOT / PILOT).read_text(encoding="utf-8").splitlines()]
    pilots = pilot_findings(inputs)
    if value["rows"] != fixture_rows(cases) + pilot_rows(pilots):
        raise ValueError("row_reduction_mismatch")
    if value["pilot_findings"] != pilots:
        raise ValueError("pilot_reduction_mismatch")
    return {"passed": True, "fixture_groups": len(cases), "pilot_groups": len(inputs)}


def _gate(
    name: str, passed: bool | None, operands: dict[str, Any], principle: str
) -> dict[str, Any]:
    return {"gate": name, "passed": passed, "measured_operands": operands, "principle": principle}


def _artifact(
    rows: list[dict[str, Any]],
    pilots: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    start: float,
    run_date: str,
    hashes: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Keep protocol readiness separate from any scientific benefit claim."""
    from carnot.reporting.experiment_7303_validation_scope import reduce_required_checks

    fixture = [
        row for row in rows if row["population"] == "fixture" and row["arm"] == "bound_relation"
    ]
    false_support = sum(
        row["observed"] == "supported" and row["truth"] != "supported" for row in fixture
    )
    wrong_unknown = sum(
        row["truth"] == "unknown" and row["observed"] != "unknown" for row in fixture
    )
    correct = sum(row["observed"] == row["truth"] for row in fixture)
    required = reduce_required_checks(receipts)
    blocked = [row for row in checks if not row["passed"] and row["check"] == "input_exists"]
    failed_auth = [row for row in checks if not row["passed"] and row["check"] != "input_exists"]
    valid = not blocked and not failed_auth and required["required_checks_passed"]
    protocol = (
        len(fixture) == 72
        and len(pilots) == 8
        and false_support == 0
        and wrong_unknown == 0
        and correct == 72
    )
    ready = valid and protocol
    classification = (
        "blocked"
        if blocked
        else "disqualified"
        if not valid
        else "circular_positive"
        if ready
        else "null"
    )
    verdict = {
        "blocked": "complete_blocked_missing_external_evidence",
        "disqualified": "complete_disqualified_required_validation",
        "circular_positive": "complete_circular_positive_bound_relation_protocol_ready",
        "null": "complete_null_bound_relation_protocol_not_ready",
    }[classification]
    gates = [
        _gate(
            "validity",
            valid,
            {
                "required_checks_passed": required["required_checks_passed"],
                "failed_preconditions": len(blocked) + len(failed_auth),
            },
            "Input custody and required validation govern validity.",
        ),
        _gate(
            "readiness",
            ready,
            {
                "fixture_groups": len(fixture),
                "correct": correct,
                "false_support": false_support,
                "wrong_unknown": wrong_unknown,
            },
            "Only bound tuples, qualifiers, unknowns, replay and validation open protocol readiness.",
        ),
        _gate(
            "coverage",
            len(fixture) == 72,
            {
                "fixture_groups": len(fixture),
                "held_out": sum(row["split"] == "held_out" for row in fixture),
                "exposed_pilots": len(pilots),
            },
            "Source groups, not arms or seeds, govern coverage.",
        ),
        _gate(
            "freshness",
            None,
            {"fresh_natural_groups": 0, "exposed_pilots": len(pilots)},
            "Fixtures and exposed pilots cannot establish natural accuracy.",
        ),
        _gate(
            "probability",
            None,
            {"paired_probabilities": 0},
            "Probability benefit requires independent labels and probabilities.",
        ),
        _gate(
            "decision_utility",
            None,
            {"typed_actions": 0},
            "Decision utility requires actions and measured outcomes.",
        ),
        _gate(
            "retention",
            None,
            {"delayed_feedback_groups": 0},
            "Retention needs delayed feedback and replay.",
        ),
        _gate(
            "efficiency",
            None,
            {"current_model_tokens": 0, "duration_s": time.monotonic() - start},
            "CPU protocol cost alone is not a comparative efficiency result.",
        ),
    ]
    return {
        "schema": "carnot.exp7672.v669.bound_relations.v1",
        "experiment_id": "exp7672-v669-bound-relations",
        "milestone": "2026.09.669",
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": classification,
        "flagged_adversarial": False,
        "gate_check_summary": blocked + failed_auth,
        "acceptance_gate_results": gates,
        "rows": rows,
        "fixture_results": fixture,
        "pilot_findings": pilots,
        "sample_size_budget": {
            "intended_independent_groups": 80,
            "observed_independent_groups": len(fixture) + len(pilots),
            "eligible": len(fixture),
            "excluded": len(pilots),
            "censored": sum(row["censored"] for row in fixture),
            "prior_exposure": "Eight V664 pilots exposed before this run; fixture oracle truth is constructed.",
            "effective_blocks": {"fixture": len(fixture), "natural_fresh": 0},
            "limits": "No natural-answer accuracy or learned-verifier advantage.",
        },
        "inference_substrate": "deterministic_tool_source_atoms_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no model used or planned in current CPU work",
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
        "random_seed": {"fixture_order": "fixed indexed enumeration; no random draws"},
        "reproducibility_checksum": _hash(
            {"inputs": hashes, "configuration": manifest, "reducer": sha256_file(Path(__file__))}
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": manifest,
            "required_commands": receipts,
            **required,
            "unrelated_full_suite_debt": [],
        },
        "verifier_is_oracle": True,
        "field_principles": {
            "honest_verdict": "Completion differs from scientific benefit.",
            "verdict_class": "Only exact required checks govern terminal class.",
            "rows": "One source group per arm with raw spans and provenance.",
            "sample_size_budget": "Repeated arms do not enlarge independent n.",
            "relation_protocol_ready_score": "A complete checked and reloaded tuple protocol is required.",
            "flagged_adversarial": "Reader flags cannot open a gate.",
            "duration_s": "Actual current CPU work only.",
            "inference_substrate_class": "No current model load.",
            "acceptance_gate_results": "Each gate retains its measured operands.",
        },
        "relation_protocol_ready_score": int(ready),
        "relation_schema_path": str(RAW / "schema.json"),
        "primitive_vocabulary": PRIMITIVES,
        "whole_answer_certified": False,
        "retirement": "Retire unchanged membership-only mechanism after same-verdict tuple failures; resource absence is a separate block.",
    }


def run_experiment(
    run_date: str, output: Path
) -> dict[str, Any]:  # pragma: no cover - exercised by CLI E2E
    """Checkpoint each unit and publish only after exact-candidate readers."""
    from carnot.reporting.experiment_7303_validation_scope import (
        CommandSpec,
        build_scoped_commands,
        run_commands,
    )

    start = time.monotonic()
    root = ROOT.resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []

    def finish(name: str, begin: float, units: int, checkpoint: Path) -> None:
        end = time.monotonic() - start
        spans.append(
            {
                "phase": name,
                "start_offset_s": begin,
                "end_offset_s": end,
                "duration_s": end - begin,
                "heartbeat_times_s": [begin, end],
                "completed_units": units,
                "checkpoint": str(checkpoint.relative_to(root)),
            }
        )
        _progress(name, "complete", start, f"units={units}")

    _progress("preconditions", "start", start, f"root={root}")
    begin = time.monotonic() - start
    cases = fixture_cases()
    manifest = {
        **SCOPE,
        "fixture_hash": _hash(cases),
        "holdout_groups": 24,
        "oracle_scope": "exact_fixture_only",
    }
    atomic_json(raw / "frozen_scope.json", manifest)
    input_path = root / PILOT
    checks = [
        {
            "check": "input_exists",
            "upstream": "Exp7602 pilot model inputs",
            "path": str(PILOT),
            "field": "exists",
            "operator": "==",
            "expected": True,
            "observed": input_path.is_file(),
            "passed": input_path.is_file(),
        }
    ]
    hashes = {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []}
    if input_path.is_file():
        hashes["producers"][str(PILOT)] = sha256_file(input_path)
        inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
        observed = len(inputs)
        checks.append(
            {
                "check": "pilot_group_count",
                "upstream": "Exp7602 pilot model inputs",
                "path": str(PILOT),
                "field": "independent_groups",
                "operator": "==",
                "expected": 8,
                "observed": observed,
                "passed": observed == 8,
            }
        )
    else:
        inputs = []
        hashes["missing_evidence"].append(str(PILOT))
    atomic_json(raw / "preconditions.json", checks)
    finish("preconditions", begin, len(checks), raw / "preconditions.json")

    _progress("relations", "start", start)
    begin = time.monotonic() - start
    fixture = fixture_rows(cases)
    for index in range(0, len(fixture), 2):
        atomic_json(
            raw / "checkpoints" / f"fixture_{index // 2:03d}.json", fixture[index : index + 2]
        )
        if index % 16 == 0:
            _progress("relations", "heartbeat", start, f"completed_groups={index // 2 + 1}/72")
    try:
        pilots = pilot_findings(inputs) if len(inputs) == 8 else []
    except ValueError as exc:
        pilots = []
        checks.append(
            {
                "check": "pilot_byte_authentication",
                "upstream": "Exp7602 pilot model inputs",
                "path": str(PILOT),
                "field": "source_and_answer_sha256",
                "operator": "==",
                "expected": "declared row hashes",
                "observed": str(exc),
                "passed": False,
            }
        )
    for index, finding in enumerate(pilots):
        atomic_json(raw / "checkpoints" / f"pilot_{index:03d}.json", finding)
        _progress("relations", "heartbeat", start, f"completed_pilots={index + 1}/8")
    rows = fixture + pilot_rows(pilots)
    atomic_json(raw / "rows.json", rows)
    atomic_json(
        raw / "schema.json",
        {
            "schema": "carnot.exp7672.v669.relations.v1",
            "relation_fields": [
                "kind",
                "dialect",
                "arguments",
                "polarity",
                "complete",
                "evidence_span",
                "source_id",
            ],
            "finding_fields": [
                "kind",
                "arguments",
                "polarity",
                "answer_span",
                "status",
                "reason",
                "evidence_span",
            ],
            "primitive_vocabulary": PRIMITIVES,
        },
    )
    finish("relations", begin, len(rows), raw / "rows.json")

    _progress("validation", "start", start)
    begin = time.monotonic() - start
    with tempfile.TemporaryDirectory(prefix="exp7672-", dir="/tmp") as private_dir:
        private = Path(private_dir)
        (private / "basetemp").mkdir()
        commands = build_scoped_commands(
            root,
            SCOPE["tests"],
            SCOPE["changed_modules"],
            static_paths=SCOPE["static_paths"],
            basetemp=private / "basetemp",
            coverage_file=private / ".coverage",
        )
        manifest["commands"] = [{"name": item.name, "argv": list(item.argv)} for item in commands]
        atomic_json(raw / "frozen_validation_manifest.json", manifest)
        _progress("validation", "before_subprocess", start, f"commands={len(commands)}")
        receipts = run_commands(
            root,
            commands,
            log_dir=raw / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
    finish("validation", begin, len(receipts), raw / "validation/affected")
    artifact = _artifact(rows, pilots, checks, receipts, spans, start, run_date, hashes, manifest)
    health_log = raw / "validation/full_suite_health.log"
    if health_log.is_file():
        artifact["validation_receipts"]["unrelated_full_suite_debt"] = [
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 2,
                "observed": "4 failed, 1271 passed, 5 skipped, 67 errors before owned interrupt",
                "primary_error": "KeyError: unsloth/Qwen3.6-35B-A3B-GGUF",
                "log_path": str(health_log.relative_to(root)),
                "log_sha256": sha256_file(health_log),
                "acceptance_gate": False,
            }
        ]
    if artifact["verdict_class"] == "blocked":
        atomic_json(output, artifact)
        _progress("publication", "complete", start, f"verdict=blocked output={output}")
        return artifact
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
                "carnot.experiment_7672_v669_bound_relations",
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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
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
    finish("terminal_readers", begin, len(terminal), raw / "validation/terminal")
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - start
    artifact["flagged_adversarial"] = not terminal[1]["passed"]
    if not all(row["passed"] for row in terminal):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["relation_protocol_ready_score"] = 0
        for gate in artifact["acceptance_gate_results"]:
            if gate["gate"] in {"validity", "readiness"}:
                gate["passed"] = False
    atomic_json(output, artifact)
    _progress(
        "publication", "complete", start, f"verdict={artifact['verdict_class']} output={output}"
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Select the owned producer or an independent cold-reduction process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260926")
    parser.add_argument(
        "--output", type=Path, default=Path("results/experiment_7672_v669_bound_relations.json")
    )
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        print(json.dumps(cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    run_experiment(args.date, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI E2E
    raise SystemExit(main())
