"""Cold-replay Exp6813 row arithmetic from Exp6812 raw bytes.

The runner owns only the bounded row-replay shard. It uses fresh parser,
arbiter, and reducer modules, invokes no model, and runs no authority attacks.
Completion measures evidence coverage rather than agreement with the producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from types import ModuleType
from typing import Any, Mapping, Sequence

from carnot import experiment_6824_cold_arbiter as cold_arbiter
from carnot import experiment_6824_cold_parser as cold_parser
from carnot import experiment_6824_cold_reducer as cold_reducer


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260831"
SCHEMA = "carnot.experiment_6824.selective_arbiter_cold_row_replay.v1"
INFERENCE_SUBSTRATE = "fresh-process deterministic CPU replay, no LLM"
BLOCKED_STATUS = "complete_blocked_selective_arbiter_cold_row_replay"
EXPECTED_REPLAY_ROW_COUNT = 576
NUMERICAL_TOLERANCE = 1e-12
RANDOM_SEED = {"audit_seed": 6_824_001, "interval_seed": 681_302}

SOURCE_RELATIVE_PATHS = {
    "exp6811": Path("results/experiment_6811_operational_obligation_automaton_v3.json"),
    "exp6812": Path("results/experiment_6812_sota_operational_handoff_corpus_v2.json"),
    "exp6813": Path("results/experiment_6813_selective_priority_arbiter_ab.json"),
}
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6824_selective_arbiter_cold_row_replay.json")
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6824_selective_arbiter_cold_row_replay.py"
)
MODULE_RELATIVE_PATHS = (
    Path(cold_parser.MODULE_PATH),
    Path(cold_arbiter.MODULE_PATH),
    Path(cold_reducer.MODULE_PATH),
    Path("python/carnot/experiment_6824_selective_arbiter_cold_row_replay.py"),
)
OPEN_SPEC_IDS = (
    "REQ-CONSTRAINT-6824",
    "SCENARIO-CONSTRAINT-6824-PRECONDITIONS",
    "SCENARIO-CONSTRAINT-6824-PARSER",
    "SCENARIO-CONSTRAINT-6824-ORDER-AND-IDENTITY",
    "SCENARIO-CONSTRAINT-6824-BUDGETS-AND-JOINS",
    "SCENARIO-CONSTRAINT-6824-INTERVALS",
    "SCENARIO-CONSTRAINT-6824-ROW-FAULTS",
    "SCENARIO-CONSTRAINT-6824-AGGREGATION",
)
TASK_REQUIRED_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "independent_parser_id",
    "independent_arbiter_id",
    "independent_reducer_id",
    "rows",
    "row_coverage",
    "aggregate_recomputation",
    "headline_differences",
    "budget_recomputation",
    "safe_action_identity_recomputation",
    "hard_violation_recomputation",
    "false_intervention_recomputation",
    "source_verdict_supported",
    "cold_replay_shard_complete",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "title",
    "run_date",
    "status",
    "openspec_requirement_ids",
    "numerical_tolerance",
    "replay_commands",
    *TASK_REQUIRED_FIELDS,
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ARMS = (cold_arbiter.SELECTIVE_ARM, cold_arbiter.FLAT_ARM)
RECOMPUTED_ROW_FIELDS = (
    "abstention",
    "accepted_hard_violation",
    "accepted_progress",
    "arm",
    "base_already_valid",
    "candidate_count",
    "candidate_evidence",
    "certificate",
    "certificate_complete",
    "cpu_allowance_us",
    "exact_check_count",
    "false_intervention",
    "handoff_arm",
    "harmful_selection",
    "legality",
    "model_id",
    "outcome_certificate",
    "outcome_check_count",
    "pair_id",
    "post_selection_hard_violation_count",
    "random_seed",
    "retry_cap",
    "retry_count",
    "row_id",
    "safe_action_identity",
    "scenario_id",
    "selected_action",
    "selected_action_bytes_b64",
    "selected_action_sha256",
    "selected_candidate_id",
    "split",
    "work_units",
)
REPLAY_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6824_selective_arbiter_cold_row_replay.py -q",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/experiment_6824_*.py tests/python/test_experiment_6824_selective_arbiter_cold_row_replay.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6824_selective_arbiter_cold_row_replay.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6824_selective_arbiter_cold_row_replay.json",
    ".venv/bin/python scripts/artifact_convention_audit.py --dry-run results/experiment_6824_selective_arbiter_cold_row_replay.json",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6824_selective_arbiter_cold_row_replay.json",
    ".venv/bin/python scripts/root_clutter_sweep.py --check",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets later readers reject incompatible evidence.",
    "experiment_id": "A stable identity prevents this shard from being confused with the failed monolith.",
    "title": "A plain title states the bounded claim under audit.",
    "run_date": "The execution date distinguishes this replay from later source revisions.",
    "status": "A closed status separates terminal evidence from an interrupted task.",
    "openspec_requirement_ids": "Requirement links make each tested behavior traceable.",
    "numerical_tolerance": "A frozen tolerance prevents agreement from being redefined after comparison.",
    "replay_commands": "Recorded commands bind the artifact to its reproducible execution path.",
    "field_principles": "One reason per field makes the evidence contract self-explanatory.",
    "inference_substrate": "The substrate states that cached bytes, not a model call, produced the audit.",
    "duration_s": "Measured wall time makes an implausible execution claim detectable.",
    "random_seed": "Fixed interval and audit seeds make stochastic arithmetic repeatable.",
    "reproducibility_checksum": "One digest binds inputs, code, rows, commands, and output payload.",
    "source_artifact_hashes": "Content hashes prevent silent substitution of any source experiment.",
    "independent_parser_id": "A source hash proves which parser reparsed the raw bytes.",
    "independent_arbiter_id": "A source hash identifies the fresh authority implementation.",
    "independent_reducer_id": "A source hash identifies the fresh aggregation implementation.",
    "rows": "Per-unit rows keep every aggregate claim falsifiable.",
    "row_coverage": "Exact identities expose a missing, extra, or repeated unit before reduction.",
    "aggregate_recomputation": "Fresh row reduction prevents producer headlines from certifying themselves.",
    "headline_differences": "Explicit differences preserve agreement and contradiction without selection.",
    "budget_recomputation": "Matched budgets prevent unequal work from masquerading as an arm effect.",
    "safe_action_identity_recomputation": "Exact bytes detect needless changes to already-valid proposals.",
    "hard_violation_recomputation": "Exact hard counts keep safety authority outside soft scoring.",
    "false_intervention_recomputation": "A closed numerator measures unnecessary changes to safe inputs.",
    "source_verdict_supported": "A closed row decision states whether cold arithmetic supports the producer claim.",
    "cold_replay_shard_complete": "Completeness depends on evidence coverage, not a favorable effect.",
    "gate_check_summary": "Failed gates name expected and observed values instead of failing silently.",
    "verifier_is_oracle": "False distinguishes an audit of a producer claim from the outcome oracle.",
    "verdict_class": "A closed enum preserves positive, null, blocked, disqualified, and partial outcomes.",
    "honest_verdict": "A terminal sentence gives the conductor an unambiguous evidence state.",
}


class ColdReplayError(ValueError):
    """Report an invalid source or output artifact."""


def canonical_json(value: Any) -> bytes:
    """Encode stable ASCII JSON for hashes and exact identity checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    """Hash one JSON value independently of source whitespace."""

    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash source bytes or retain an explicit missing state."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json_object(path: Path) -> JsonDict:
    """Load one JSON object and reject arrays or scalar substitutes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ColdReplayError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ColdReplayError("JSON object required")
    return value


def load_sources(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Load all source objects while retaining readable failure receipts."""

    loaded: dict[str, JsonDict] = {}
    errors: dict[str, str] = {}
    for name, path in paths.items():
        try:
            loaded[name] = read_json_object(path)
        except ColdReplayError as exc:
            loaded[name] = {}
            errors[name] = str(exc)
    if errors:
        loaded["__load_errors__"] = errors
    return loaded


def source_artifact_hashes(paths: Mapping[str, Path]) -> JsonDict:
    """Record each source file path and exact byte hash."""

    return {
        name: {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def module_identity(module: ModuleType, relative_path: Path) -> JsonDict:
    """Bind one independent module name to its source bytes and import boundary."""

    path = REPO_ROOT / relative_path
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {
        "imports_exp6813": "experiment_6813_selective_priority_arbiter_ab" in text,
        "module": module.__name__,
        "path": relative_path.as_posix(),
        "sha256": sha256_file(path),
        "version": getattr(
            module,
            "PARSER_VERSION",
            getattr(module, "ARBITER_VERSION", getattr(module, "REDUCER_VERSION", "v1")),
        ),
    }


def _module_identities() -> JsonDict:
    """Return the three task-owned code identities consumed by completion."""

    return {
        "independent_parser_id": module_identity(cold_parser, MODULE_RELATIVE_PATHS[0]),
        "independent_arbiter_id": module_identity(cold_arbiter, MODULE_RELATIVE_PATHS[1]),
        "independent_reducer_id": module_identity(cold_reducer, MODULE_RELATIVE_PATHS[2]),
    }


def _check_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Use one uniform gate receipt for successful and blocked artifacts."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def check_preconditions(sources: Mapping[str, JsonDict], paths: Mapping[str, Path]) -> JsonDict:
    """Check all owned evidence before any cold arbitration starts."""

    exp6812 = sources.get("exp6812", {})
    exp6813 = sources.get("exp6813", {})
    raw_manifest = exp6812.get("raw_output_manifest")
    expected_cells = exp6812.get("frozen_manifest", {}).get("expected_cell_ids", [])
    raw_failures: list[str] = []
    if isinstance(raw_manifest, list):
        for receipt in raw_manifest:
            try:
                cold_parser.parse_receipt(receipt)
            except (cold_parser.ColdParseError, TypeError) as exc:
                raw_failures.append(str(exc))
    raw_ids = [row.get("cell_id") for row in raw_manifest] if isinstance(raw_manifest, list) else []
    raw_passed = (
        isinstance(raw_manifest, list)
        and len(raw_manifest) == 288
        and len(set(raw_ids)) == 288
        and set(raw_ids) == set(expected_cells)
        and not raw_failures
    )

    source_rows = exp6812.get("rows")
    expected_source_ids = exp6812.get("frozen_manifest", {}).get("expected_row_ids", [])
    observed_source_ids = (
        [row.get("row_id") for row in source_rows] if isinstance(source_rows, list) else []
    )
    source_rows_passed = (
        isinstance(source_rows, list)
        and len(source_rows) == EXPECTED_REPLAY_ROW_COUNT
        and len(set(observed_source_ids)) == EXPECTED_REPLAY_ROW_COUNT
        and set(observed_source_ids) == set(expected_source_ids)
    )

    hashes = source_artifact_hashes(paths)
    hash_observed = {
        "exp6811_file": hashes["exp6811"]["sha256"],
        "exp6812_file": hashes["exp6812"]["sha256"],
        "exp6813_compiler_receipt": exp6813.get("compiler_artifact_sha256"),
        "exp6813_file": hashes["exp6813"]["sha256"],
        "exp6813_source_receipt": exp6813.get("source_artifact_sha256"),
    }
    hashes_passed = (
        all(value != "missing" for value in (hashes[name]["sha256"] for name in paths))
        and exp6813.get("compiler_artifact_sha256") == hashes["exp6811"]["sha256"]
        and exp6813.get("source_artifact_sha256") == hashes["exp6812"]["sha256"]
        and not sources.get("__load_errors__")
    )

    frozen = exp6813.get("frozen_manifest", {})
    frozen_observed = {
        "arms": frozen.get("arms"),
        "development_count": len(frozen.get("development_scenario_ids", [])),
        "frozen_before_reduction": frozen.get("frozen_before_reduction"),
        "held_count": len(frozen.get("held_scenario_ids", [])),
        "source_handoff_arms": frozen.get("source_handoff_arms"),
    }
    frozen_passed = (
        frozen_observed
        == {
            "arms": list(ARMS),
            "development_count": 24,
            "frozen_before_reduction": True,
            "held_count": 24,
            "source_handoff_arms": ["direct_typed", "compressed_prose"],
        }
        and isinstance(frozen.get("public_constants"), dict)
        and frozen["public_constants"].get("candidate_count") == 2
    )
    completed = exp6813.get("selective_arbiter_ab_completed") is True
    checks = [
        _check_row(
            "raw_byte_manifests_readable",
            {"cell_count": 288, "hashes_valid": True},
            {"cell_count": len(raw_ids), "failures": raw_failures[:3]},
            raw_passed,
        ),
        _check_row(
            "complete_source_rows",
            EXPECTED_REPLAY_ROW_COUNT,
            len(observed_source_ids),
            source_rows_passed,
        ),
        _check_row(
            "source_artifact_hashes",
            "Exp6813 receipts equal readable Exp6811 and Exp6812 file hashes",
            hash_observed,
            hashes_passed,
        ),
        _check_row(
            "frozen_arm_manifest",
            {
                "arms": list(ARMS),
                "development_count": 24,
                "frozen_before_reduction": True,
                "held_count": 24,
                "source_handoff_arms": ["direct_typed", "compressed_prose"],
            },
            frozen_observed,
            frozen_passed,
        ),
        _check_row(
            "selective_arbiter_ab_completed",
            True,
            exp6813.get("selective_arbiter_ab_completed"),
            completed,
        ),
    ]
    failed = [row["check"] for row in checks if not row["passed"]]
    return {
        "checks": checks,
        "failed_checks": failed,
        "failures": [row for row in checks if not row["passed"]],
        "passed": not failed,
    }


def _candidate_evidence(
    evaluated: Mapping[str, Any], source_row: Mapping[str, Any], raw_hash: str
) -> JsonDict:
    """Keep source identity beside independently recomputed proposal evidence."""

    return {
        "authority_preserved": evaluated["authority_preserved"],
        "binding_violation_vector": evaluated["binding_violation_vector"],
        "candidate_id": evaluated["candidate_id"],
        "candidate_index": evaluated["candidate_index"],
        "hard_violation_count": evaluated["hard_violation_count"],
        "legal_support": evaluated["legal_support"],
        "parse_state": evaluated["parse_state"],
        "raw_output_sha256": raw_hash,
        "row_sha256": source_row.get("row_sha256"),
        "soft_score": evaluated["soft_score"],
    }


def _row_differences(cold: Mapping[str, Any], producer: Mapping[str, Any]) -> list[str]:
    """Name each recomputable producer row field that differs from cold arithmetic."""

    return [field for field in RECOMPUTED_ROW_FIELDS if cold.get(field) != producer.get(field)]


def replay_rows(sources: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Reparse every raw cell and independently rebuild all 576 arm rows."""

    exp6812 = sources["exp6812"]
    exp6813 = sources["exp6813"]
    scenarios = {row["scenario_id"]: row for row in exp6812["frozen_manifest"]["scenarios"]}
    receipts = {
        f"{row['model_id']}|{row['scenario_id']}|seed-{row['random_seed']}|{row['arm']}": row
        for row in exp6812["raw_output_manifest"]
    }
    source_candidates = {row["row_id"]: row for row in exp6812["rows"]}
    parsed_cells: dict[str, list[JsonDict]] = {}
    constants = exp6813["frozen_manifest"]["public_constants"]
    development = set(exp6813["frozen_manifest"]["development_scenario_ids"])
    rows: list[JsonDict] = []
    for producer in exp6813["rows"]:
        pair_id = str(producer["pair_id"])
        scenario = scenarios[str(producer["scenario_id"])]
        if pair_id not in parsed_cells:
            parsed = cold_parser.parse_receipt(receipts[pair_id])
            parsed_cells[pair_id] = [
                cold_arbiter.evaluate_candidate(scenario, candidate)
                for candidate in parsed["candidates"]
            ]
        evaluated = parsed_cells[pair_id]
        selected = cold_arbiter.select_arm(
            scenario,
            evaluated,
            arm=str(producer["arm"]),
            soft_clip=int(constants["soft_score_clip"]),
        )
        base_action = cold_arbiter.valid_base_action(scenario, evaluated)
        candidate_evidence = []
        source_candidate_ids = []
        for candidate in evaluated:
            source_id = f"{receipts[pair_id]['cell_id']}|candidate-{candidate['candidate_index']}"
            source_candidate_ids.append(source_id)
            candidate_evidence.append(
                _candidate_evidence(
                    candidate,
                    source_candidates[source_id],
                    str(receipts[pair_id]["raw_output_sha256"]),
                )
            )
        cold: JsonDict = {
            "abstention": selected["abstention"],
            "accepted_hard_violation": selected["accepted_hard_violation"],
            "accepted_progress": selected["accepted_progress"],
            "arm": producer["arm"],
            "base_already_valid": base_action is not None,
            "candidate_count": int(constants["candidate_count"]),
            "candidate_evidence": candidate_evidence,
            "certificate": selected["certificate"],
            "certificate_complete": selected["certificate_complete"],
            "cpu_allowance_us": int(constants["cpu_allowance_us"]),
            "exact_check_count": int(constants["exact_check_allowance"]),
            "false_intervention": selected["false_intervention"],
            "handoff_arm": receipts[pair_id]["arm"],
            "harmful_selection": selected["harmful_selection"],
            "legality": selected["legality"],
            "model_id": receipts[pair_id]["model_id"],
            "outcome_certificate": selected["outcome_certificate"],
            "outcome_check_count": int(constants["outcome_check_allowance"]),
            "outcome_evaluator": "Exp6824 fresh exact transition replay after selection",
            "pair_id": pair_id,
            "post_selection_hard_violation_count": selected["post_selection_hard_violation_count"],
            "random_seed": receipts[pair_id]["random_seed"],
            "retry_cap": int(constants["retry_cap"]),
            "retry_count": selected["retry_count"],
            "row_id": producer["row_id"],
            "row_type": "cold_replay",
            "safe_action_identity": selected["safe_action_identity"],
            "scenario_id": producer["scenario_id"],
            "selected_action": selected["selected_action"],
            "selected_action_bytes_b64": selected["selected_action_bytes_b64"],
            "selected_action_sha256": selected["selected_action_sha256"],
            "selected_candidate_id": selected["selected_candidate_id"],
            "source_candidate_row_ids": source_candidate_ids,
            "source_producer_row_id": producer["row_id"],
            "split": "development" if producer["scenario_id"] in development else "held",
            "work_units": int(constants["candidate_count"])
            + int(constants["outcome_check_allowance"]),
        }
        differences = _row_differences(cold, producer)
        cold["producer_row_differences"] = differences
        cold["producer_row_match"] = not differences
        rows.append(cold)
    return rows


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    """Bind inputs, code, rows, commands, and the non-self-referential output."""

    output = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return {
        "code": {
            field: artifact.get(field)
            for field in (
                "independent_parser_id",
                "independent_arbiter_id",
                "independent_reducer_id",
            )
        },
        "commands": artifact.get("replay_commands"),
        "inputs": artifact.get("source_artifact_hashes"),
        "output_payload": output,
        "rows_sha256": sha256_json(artifact.get("rows", [])),
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Return the stable replay checksum used by validation."""

    return sha256_json(_checksum_payload(artifact))


def _base_artifact(
    *,
    paths: Mapping[str, Path],
    run_date: str,
    duration_s: float,
    gate_summary: Mapping[str, Any],
) -> JsonDict:
    """Create fields shared by blocked and completed terminal artifacts."""

    modules = _module_identities()
    return {
        "schema": SCHEMA,
        "experiment_id": "6824",
        "title": "Selective arbiter cold row replay",
        "run_date": run_date,
        "status": "complete",
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": dict(RANDOM_SEED),
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": source_artifact_hashes(paths),
        **modules,
        "rows": [],
        "row_coverage": {},
        "aggregate_recomputation": {},
        "headline_differences": {},
        "budget_recomputation": {},
        "safe_action_identity_recomputation": {},
        "hard_violation_recomputation": {},
        "false_intervention_recomputation": {},
        "source_verdict_supported": False,
        "cold_replay_shard_complete": False,
        "gate_check_summary": dict(gate_summary),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "complete_partial_cold_row_replay_not_reduced",
    }


def _finish(artifact: JsonDict) -> JsonDict:
    """Attach one principle per field and compute the final content checksum."""

    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def terminal_decision(completed: bool, cold_positive: bool, source_positive: bool) -> JsonDict:
    """Classify effects without allowing effect sign to control completion."""

    if not completed:
        return {
            "cold_replay_shard_complete": False,
            "source_verdict_supported": False,
            "verdict_class": "partial",
            "honest_verdict": "complete_partial_cold_row_replay_evidence_incomplete",
        }
    supported = cold_positive == source_positive
    if cold_positive:
        verdict = "positive"
        text = (
            "complete: cold row replay supports the positive producer verdict"
            if supported
            else "complete: cold row replay contradicts the non-positive producer verdict"
        )
    else:
        verdict = "null"
        text = (
            "complete: cold row replay supports the non-positive producer verdict"
            if supported
            else "complete: cold row replay does not support the positive producer verdict"
        )
    return {
        "cold_replay_shard_complete": True,
        "source_verdict_supported": supported,
        "verdict_class": verdict,
        "honest_verdict": text,
    }


def build_artifact(
    sources: Mapping[str, JsonDict],
    *,
    source_paths: Mapping[str, Path],
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Build a blocked artifact or the complete bounded cold replay."""

    gate_summary = check_preconditions(sources, source_paths)
    artifact = _base_artifact(
        paths=source_paths,
        run_date=run_date,
        duration_s=duration_s,
        gate_summary=gate_summary,
    )
    if not gate_summary["passed"]:
        artifact.update(
            {
                "status": BLOCKED_STATUS,
                "verdict_class": "blocked",
                "honest_verdict": f"{BLOCKED_STATUS}: " + ", ".join(gate_summary["failed_checks"]),
            }
        )
        return _finish(artifact)

    cold_rows = replay_rows(sources)
    producer_rows = sources["exp6813"]["rows"]
    expected_ids = [str(row["row_id"]) for row in producer_rows]
    roster = cold_reducer.validate_roster(cold_rows, expected_ids)
    fault_rows = cold_reducer.run_row_fault_audits(cold_rows, expected_ids)
    frozen = sources["exp6813"]["frozen_manifest"]
    public_constants = {
        **frozen["public_constants"],
        "interval_seed": RANDOM_SEED["interval_seed"],
    }
    aggregate = cold_reducer.recompute_aggregates(
        cold_rows,
        held_scenario_ids=frozen["held_scenario_ids"],
        constants=public_constants,
    )
    differences = cold_reducer.compare_headlines(
        sources["exp6813"], aggregate, tolerance=NUMERICAL_TOLERANCE
    )
    budgets = cold_reducer.recompute_budgets(
        cold_rows, expected_pair_count=EXPECTED_REPLAY_ROW_COUNT // 2
    )
    modules_independent = all(
        not artifact[field]["imports_exp6813"] and artifact[field]["sha256"] != "missing"
        for field in (
            "independent_parser_id",
            "independent_arbiter_id",
            "independent_reducer_id",
        )
    )
    row_coverage = {
        **roster,
        "all_producer_rows_recomputed": len(cold_rows) == len(producer_rows),
        "audit_cases_complete": all(row["detected"] for row in fault_rows),
        "complete_recomputation": set(aggregate) == set(cold_reducer.HEADLINE_FIELDS),
        "expected_identity_sha256": sha256_json(expected_ids),
        "held_scenario_ids": list(frozen["held_scenario_ids"]),
        "independent_code": modules_independent,
        "observed_identity_sha256": sha256_json([row["row_id"] for row in cold_rows]),
        "public_constants": public_constants,
        "raw_cell_count": len(sources["exp6812"]["raw_output_manifest"]),
        "source_candidate_row_count": len(sources["exp6812"]["rows"]),
        "source_hash_identity": gate_summary["passed"],
    }
    row_coverage["passed"] = all(
        (
            roster["passed"],
            row_coverage["all_producer_rows_recomputed"],
            row_coverage["audit_cases_complete"],
            row_coverage["complete_recomputation"],
            row_coverage["independent_code"],
            row_coverage["source_hash_identity"],
        )
    )
    completed = row_coverage["passed"] and budgets["passed"]
    source_positive = bool(sources["exp6813"]["acceptance_gate_positive"]["passed"])
    decision = terminal_decision(
        completed,
        bool(aggregate["acceptance_gate_positive"]["passed"]),
        source_positive,
    )
    artifact.update(
        {
            "rows": [*cold_rows, *fault_rows],
            "row_coverage": row_coverage,
            "aggregate_recomputation": aggregate,
            "headline_differences": differences,
            "budget_recomputation": budgets,
            "safe_action_identity_recomputation": aggregate["safe_action_identity_by_arm"],
            "hard_violation_recomputation": aggregate["hard_violation_rate_by_arm"],
            "false_intervention_recomputation": aggregate["false_intervention_rate_by_arm"],
            **decision,
        }
    )
    return _finish(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject malformed terminals and claims detached from their cold rows."""

    findings: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS).difference(artifact))
    if missing:
        findings.append(f"missing required fields: {missing}")
    if set(artifact.get("field_principles", {})) != set(artifact):
        findings.append("field principle coverage mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        findings.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        findings.append("verdict class outside closed enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete:", "complete_", BLOCKED_STATUS)
    ):
        findings.append("honest verdict lacks terminal prefix")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        findings.append("duration_s must be non-negative")
    if artifact.get("cold_replay_shard_complete") and not artifact.get("row_coverage", {}).get(
        "passed"
    ):
        findings.append("row coverage is incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        findings.append("reproducibility checksum mismatch")
    return findings


def write_output(artifact: Mapping[str, Any], target: Path) -> None:
    """Validate and atomically replace the task-owned output file."""

    findings = validate_artifact(artifact)
    if findings:
        raise ColdReplayError("; ".join(findings))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, prefix=f".{target.name}."
    ) as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def parse_run_date(value: str) -> str:
    """Require the conductor's compact execution-date format."""

    if not re.fullmatch(r"\d{8}", value):
        raise ValueError("run date must use YYYYMMDD")
    return value


def build_from_repo(*, run_date: str) -> JsonDict:
    """Measure one fresh-process replay against repository source artifacts."""

    started = time.perf_counter()
    paths = {name: REPO_ROOT / path for name, path in SOURCE_RELATIVE_PATHS.items()}
    artifact = build_artifact(
        load_sources(paths),
        source_paths=paths,
        run_date=run_date,
        duration_s=0.0,
    )
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded replay and write one terminal JSON artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = build_from_repo(run_date=parse_run_date(args.date))
    write_output(artifact, args.output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
