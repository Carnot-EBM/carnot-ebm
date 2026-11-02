"""CPU orchestration for the inherited V668 source atom corpus.

REQ-REPORT-7659 keeps measurement custody separate from scientific benefit.
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
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)
from carnot.reporting.experiment_7646_source_features import ARMS, immutable_jsonl, read_jsonl
from carnot.reporting.experiment_7659_atom_corpus import build_role_rows, validate_input


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path("results/raw/experiment_7602_v664_evidence_requalification/protocol.json")
RAW = Path("results/raw/experiment_7659_v668_atom_corpus")
MANIFEST = RAW / "manifest.json"
ROLES = ("fit", "tune", "policy", "online", "evaluation", "pilot")
MODEL_SPECS: list[dict[str, Any]] = []


def _check(check: str, upstream: str, path: str, field: str, expected: Any, observed: Any) -> dict:
    """Name the exact operand so a missing external input can be repaired."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def authenticate(root: Path) -> tuple[dict, list[dict], dict[str, str]]:
    """Verify immutable protocol and both isolated store sets before work."""

    protocol_path = root / PROTOCOL
    checks = [
        _check("input_exists", "Exp7602", str(PROTOCOL), "exists", True, protocol_path.is_file())
    ]
    if not protocol_path.is_file():
        return (
            {},
            checks,
            {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": [str(PROTOCOL)]},
        )
    protocol = json.loads(protocol_path.read_text())
    hashes = {
        "producers": {str(PROTOCOL): sha256_file(protocol_path)},
        "pre_gate_receipts": {},
        "missing_inputs": [],
    }
    for kind in ("model_inputs", "evaluator_stores"):
        for role in ROLES:
            receipt = protocol["reader_sidecars"][kind][role]
            relative = receipt["path"]
            path = root / relative
            exists = path.is_file()
            checks.append(_check("input_exists", "Exp7602", relative, "exists", True, exists))
            if not exists:
                hashes["missing_inputs"].append(relative)
                continue
            observed = sha256_file(path)
            checks.append(
                _check("input_hash", "Exp7602", relative, "sha256", receipt["sha256"], observed)
            )
            hashes["producers" if kind == "model_inputs" else "pre_gate_receipts"][relative] = (
                observed
            )
    schema = Path("results/raw/experiment_7658_v668_evidence_atoms/schema.json")
    schema_path = root / schema
    checks.append(
        _check("atom_schema_exists", "Exp7658", str(schema), "exists", True, schema_path.is_file())
    )
    if schema_path.is_file():
        hashes["producers"][str(schema)] = sha256_file(schema_path)
    else:
        hashes["missing_inputs"].append(str(schema))
    checks.append(
        _check("host_cpu", "local", "/proc/self", "cpu_available", True, (os.cpu_count() or 0) >= 1)
    )
    return protocol, checks, hashes


def _derangement(protocol: dict, role: str, rows: list[dict]) -> dict[str, str]:
    """Reuse the scored mapping; rotate pilots without looking at labels."""

    ids = [row["component_hash"] for row in rows]
    if role == "pilot":
        return dict(zip(ids, ids[1:] + ids[:1], strict=True))
    return {group: protocol["within_role_derangement"][group] for group in ids}


def _role_inputs(root: Path, protocol: dict, role: str) -> list[dict]:
    """Read only the predictor store and compare its roster with protocol facts."""

    path = root / protocol["reader_sidecars"]["model_inputs"][role]["path"]
    rows = read_jsonl(path)
    roster = {item["component_hash"]: item for item in protocol["roster"] if item["role"] == role}
    if len(rows) != protocol["role_counts"][role] or {row["component_hash"] for row in rows} != set(
        roster
    ):
        raise ValueError("roster_mismatch")
    for row in rows:
        expected = dict(row)
        expected["role"] = roster[row["component_hash"]]["role"]
        expected["source_sha256"] = roster[row["component_hash"]]["source_sha256"]
        expected["answer_sha256"] = roster[row["component_hash"]]["response_sha256"]
        validate_input(row, expected, role)
    return rows


def _coverage(rows: list[dict]) -> dict[str, dict]:
    """Count groups once in the original arm; arms are paired controls."""

    result = {}
    for role in ROLES:
        own = [row for row in rows if row["role"] == role and row["arm"] == "original_source"]
        result[role] = {
            "groups": len(own),
            "dialects": sorted({dialect for row in own for dialect in row["dialects"]}),
            "checked_propositions": sum(row["checked_structural_propositions"] for row in own),
            "checked_groups": sum(row["checked_structural_propositions"] > 0 for row in own),
            "lexical_only_evidence": sum(row["lexical_membership"] for row in own),
            "unknowns": sum(row["unknown_claims"] for row in own),
            "unverified_spans": sum(row["unsupported_semantic_spans"] for row in own),
        }
    return result


def _gate(name: str, passed: bool | None, operands: dict, principle: str) -> dict:
    """Put the reason next to every measured acceptance decision."""

    return {"gate": name, "passed": passed, "measured_operands": operands, "principle": principle}


def build_artifact(
    protocol: dict,
    checks: list[dict],
    hashes: dict,
    rows: list[dict],
    manifest: dict,
    receipts: list[dict],
    spans: list[dict],
    started: float,
    run_date: str,
) -> dict:
    """Keep accounting, learning readiness, and unmeasured science distinct."""

    coverage = _coverage(rows)
    fit = [
        row["checked_structural_propositions"]
        for row in rows
        if row["role"] == "fit" and row["arm"] == "original_source"
    ]
    coverage_pass = coverage["fit"]["checked_groups"] >= 16
    nonconstant = len(set(fit)) > 1
    complete = len(rows) == 248 * len(ARMS) and all(
        coverage[role]["groups"] == protocol["role_counts"][role] for role in ROLES
    )
    required = reduce_required_checks(receipts) if receipts else {"required_checks_passed": False}
    inputs_pass = all(check["passed"] for check in checks)
    valid = inputs_pass and required["required_checks_passed"] and complete
    ready = valid and coverage_pass and nonconstant
    if not inputs_pass:
        kind, verdict = "blocked", "complete_blocked_missing_external_evidence"
    elif not required["required_checks_passed"] or not complete:
        kind, verdict = "disqualified", "complete_disqualified_required_validation"
    else:
        kind, verdict = (
            "null",
            "complete_null_atom_corpus_ready"
            if ready
            else "complete_null_learning_readiness_closed",
        )
    gates = [
        _gate(
            "validity",
            valid,
            {
                "authenticated_inputs": inputs_pass,
                "required_checks_passed": required["required_checks_passed"],
                "rows": len(rows),
            },
            "Only authenticated rows and passing required checks establish validity.",
        ),
        _gate(
            "readiness",
            ready,
            {
                "fit_covered_groups": coverage["fit"]["checked_groups"],
                "fit_nonconstant": nonconstant,
                "complete": complete,
            },
            "Learning requires sixteen covered fit groups and nonconstant features.",
        ),
        _gate(
            "coverage",
            valid and coverage_pass,
            {"fit_covered_groups": coverage["fit"]["checked_groups"], "threshold": 16},
            "Count each fit source group once, using original source only.",
        ),
        _gate(
            "probability_benefit",
            None,
            {"paired_probabilities": 0},
            "Probability benefit needs paired predictions and independent outcomes.",
        ),
        _gate(
            "utility", None, {"typed_decisions": 0}, "Utility needs decisions, costs, and outcomes."
        ),
        _gate(
            "retention",
            None,
            {"delayed_feedback_events": 0},
            "Retention needs delayed feedback and later replay.",
        ),
        _gate(
            "freshness",
            False,
            {"fresh_confirmatory_groups": 0},
            "The inherited roster was previously exposed.",
        ),
    ]
    reducer = ROOT / "python/carnot/reporting/experiment_7659_atom_corpus.py"
    checksum_payload = json.dumps(
        {"hashes": hashes, "manifest": manifest, "reducer_sha256": sha256_file(reducer)},
        sort_keys=True,
    ).encode()
    return {
        "schema": "carnot.exp7659.v668.atom_corpus.v1",
        "experiment_id": "exp7659-v668-atom-corpus",
        "milestone": "2026.09.668",
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": kind,
        "flagged_adversarial": False,
        "gate_check_summary": [check for check in checks if not check["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended_independent_groups": 248,
            "observed_independent_groups": len({row["unit_id"] for row in rows}),
            "eligible": len({row["unit_id"] for row in rows}),
            "excluded": 0,
            "censored": sum(row["censored"] for row in rows if row["arm"] == "original_source"),
            "prior_exposure": "All 248 inherited source groups were previously exposed.",
            "claim_limits": "Arms and repeated claims do not enlarge the sample or establish probability benefit.",
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
            "cpu_class": platform.processor() or platform.machine(),
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - started,
        "random_seed": {
            "scored_derangement": protocol.get("selection_salt", "Exp7602 fixed mapping"),
            "pilot_derangement": "fixed roster rotation by one",
        },
        "reproducibility_checksum": "sha256:" + hashlib.sha256(checksum_payload).hexdigest(),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": manifest["validation_scope"],
            "required_commands": receipts,
            **required,
            "unrelated_repository_suite_debt": [
                {
                    "command": ".venv/bin/pytest tests/python -q",
                    "collected": 71079,
                    "observation": "At least five failures appeared before interruption near 1% completion.",
                    "exit_code": 130,
                    "scope": "repository health, outside frozen affected acceptance",
                }
            ],
        },
        "verifier_is_oracle": False,
        "field_principles": {
            "rows": "Every original source group remains in each paired arm; claims do not enlarge the sample.",
            "coverage_by_role": "Count checked original-source groups once and retain unknown prose.",
            "atom_features_ready_score": "Authentic complete rows, variable fit features, sixteen covered fit groups, and terminal checks are required.",
            "honest_verdict": "Completion and scientific benefit are separate.",
            "flagged_adversarial": "A flagged terminal reader closes readiness.",
            "duration_s": "Measure actual current CPU work without a model-duration floor.",
            "inference_substrate_class": "No model load or generation occurs in this run.",
            "sample_size_budget": "Independent units are source groups, not arms, claims, or duplicated tool responses.",
        },
        "atom_features_ready_score": int(ready),
        "feature_manifest_path": str(MANIFEST),
        "coverage_by_role": coverage,
        "fresh_confirmatory_claim_allowed": False,
        "historical_model_id": "unsloth/Qwen3.8-27B-GGUF",
        "retirement": "Retire unchanged V667 source and claim grammar after its zero-check result; atom features replace that mechanism.",
    }


def cold_reduce(candidate: Path) -> dict:
    """Recreate all feature rows from raw predictor bytes in a fresh process."""

    artifact = json.loads(candidate.read_text())
    manifest_path = ROOT / artifact["feature_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in artifact["source_artifact_hashes"]["producers"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError("source_hash_mismatch")
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    rebuilt = []
    for role in ROLES:
        info = manifest["roles"][role]
        inputs = _role_inputs(ROOT, protocol, role)
        if [row["component_hash"] for row in inputs] != info["group_ids"]:
            raise ValueError("role_roster_mismatch")
        if sha256_file(ROOT / info["feature_path"]) != info["feature_sha256"]:
            raise ValueError("feature_hash_mismatch")
        role_rows = build_role_rows(inputs, role, _derangement(protocol, role, inputs))
        if read_jsonl(ROOT / info["feature_path"]) != role_rows:
            raise ValueError("feature_reconstruction_mismatch")
        rebuilt.extend(role_rows)
    if artifact["rows"] != rebuilt or len({row["unit_id"] for row in rebuilt}) != 248:
        raise ValueError("row_reduction_mismatch")
    return {"passed": True, "groups": 248, "rows": len(rebuilt)}


def run_experiment(run_date: str, output: Path) -> dict:  # pragma: no cover - exercised by task E2E
    """Freeze each source group, run scoped checks, and publish terminal bytes."""

    started = time.monotonic()
    root = ROOT.resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict] = []

    def progress(phase: str, event: str, detail: str = "") -> None:
        print(
            f"[exp7659] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {detail}",
            flush=True,
        )

    def close_phase(name: str, begin: float, units: int, checkpoint: Path) -> None:
        end = time.monotonic() - started
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
        progress(name, "complete", f"units={units}")

    progress("preconditions", "start", f"root={root}")
    begin = time.monotonic() - started
    protocol, checks, hashes = authenticate(root)
    atomic_json(raw / "preconditions.json", {"checks": checks, "source_artifact_hashes": hashes})
    close_phase("preconditions", begin, len(checks), raw / "preconditions.json")
    tests = ["tests/python/test_experiment_7659_v668_atom_corpus.py"]
    modules = ["python/carnot/reporting/experiment_7659_atom_corpus.py"]
    static = [
        "python/carnot/experiment_7659_v668_atom_corpus.py",
        "scripts/experiments/experiment_7659_v668_atom_corpus.py",
    ]
    with tempfile.TemporaryDirectory(prefix="exp7659-", dir="/tmp") as private_dir:
        private = Path(private_dir)
        basetemp = private / "basetemp"
        basetemp.mkdir()
        commands = build_scoped_commands(
            root,
            tests,
            modules,
            static_paths=static,
            basetemp=basetemp,
            coverage_file=private / ".coverage",
        )
        commands.append(
            CommandSpec(
                "orchestration_mypy",
                (str(root / ".venv/bin/mypy"), "python/carnot/experiment_7659_v668_atom_corpus.py"),
                "changed_orchestration",
            )
        )
        scope = {
            "tests": tests,
            "changed_modules": modules,
            "static_paths": static,
            "commands": [{"name": item.name, "argv": list(item.argv)} for item in commands],
        }
        atomic_json(raw / "frozen_validation_manifest.json", scope)
        manifest: dict[str, Any] = {
            "schema": "carnot.exp7659.feature_manifest.v1",
            "atom_schema_path": "results/raw/experiment_7658_v668_evidence_atoms/schema.json",
            "roles": {},
            "validation_scope": scope,
        }
        rows: list[dict] = []
        if all(check["passed"] for check in checks):
            progress("extract", "start")
            begin = time.monotonic() - started
            for role in ROLES:
                inputs = _role_inputs(root, protocol, role)
                mapping = _derangement(protocol, role, inputs)
                role_rows = build_role_rows(inputs, role, mapping)
                rows.extend(role_rows)
                feature_path = RAW / f"{role}_features.jsonl"
                feature_hash = immutable_jsonl(root / feature_path, role_rows)
                manifest["roles"][role] = {
                    "input_path": protocol["reader_sidecars"]["model_inputs"][role]["path"],
                    "input_sha256": hashes["producers"][
                        protocol["reader_sidecars"]["model_inputs"][role]["path"]
                    ],
                    "feature_path": str(feature_path),
                    "feature_sha256": feature_hash,
                    "group_ids": [row["component_hash"] for row in inputs],
                    "derangement": mapping,
                }
                atomic_json(
                    raw / "checkpoints" / f"{role}.json",
                    {"completed_groups": len(inputs), "feature_sha256": feature_hash},
                )
                progress(
                    "extract",
                    "role_complete",
                    f"role={role} groups={len(inputs)} rows={len(role_rows)}",
                )
            close_phase("extract", begin, len(rows), raw / "checkpoints")
        atomic_json(root / MANIFEST, manifest)
        progress("affected_validation", "before_subprocess")
        begin = time.monotonic() - started
        receipts = run_commands(
            root,
            commands,
            log_dir=raw / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        close_phase("affected_validation", begin, len(receipts), raw / "validation/affected")
        if not protocol:
            protocol = {"role_counts": {role: 0 for role in ROLES}}
        artifact = build_artifact(
            protocol, checks, hashes, rows, manifest, receipts, spans, started, run_date
        )
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
                    "carnot.experiment_7659_v668_atom_corpus",
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
        progress("terminal_readers", "before_subprocess")
        begin = time.monotonic() - started
        terminal = run_commands(
            root,
            readers,
            log_dir=raw / "validation/terminal",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        close_phase("terminal_readers", begin, len(terminal), raw / "validation/terminal")
        artifact["validation_receipts"]["terminal_readers"] = terminal
        artifact["flagged_adversarial"] = not next(
            item["passed"] for item in terminal if item["name"] == "adversarial_verify"
        )
        if not all(item["passed"] for item in terminal):
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["atom_features_ready_score"] = 0
            for gate in artifact["acceptance_gate_results"][:3]:
                gate["passed"] = False
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        atomic_json(candidate, artifact)
        progress("exact_terminal_replay", "before_subprocess")
        begin = time.monotonic() - started
        exact = run_commands(
            root,
            readers,
            log_dir=raw / "validation/exact",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        atomic_json(raw / "exact_terminal_reader_outcomes.json", {"outcomes": exact})
        close_phase(
            "exact_terminal_replay", begin, len(exact), raw / "exact_terminal_reader_outcomes.json"
        )
        if not all(item["passed"] for item in exact):
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["atom_features_ready_score"] = 0
            for gate in artifact["acceptance_gate_results"][:3]:
                gate["passed"] = False
        artifact["terminal_reader_outcomes_path"] = str(RAW / "exact_terminal_reader_outcomes.json")
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        progress("publication", "before_atomic")
        atomic_json(output, artifact)
        progress("publication", "after_atomic", artifact["honest_verdict"])
        return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exercised by task E2E
    """Dispatch production or independent read-only reduction."""

    print("[exp7659] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260925")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/experiment_7659_v668_atom_corpus.json"
    )
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        print(json.dumps(cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    result = run_experiment(args.date, args.output)
    return 0 if result["verdict_class"] == "null" else 1


if __name__ == "__main__":  # pragma: no cover - exercised by task E2E
    raise SystemExit(main())
