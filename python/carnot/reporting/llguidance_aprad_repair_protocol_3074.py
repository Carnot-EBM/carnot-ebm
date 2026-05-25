"""Build the Exp 3074 LLGuidance/AprAD repair protocol artifact.

Spec refs: REQ-REPORT-3074, SCENARIO-REPORT-3074.

This module is a protocol step, not a repair generator. It preserves the
de-tautology blockers from Exp 3056 and the gate-blocked Exp 3059 rerun result,
then turns LLGuidance-style grammar constraints and AprAD-style intent
preservation into fields that Exp 3075 can consume before any live SOTA repair
candidate is generated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.287"
SCHEMA = "carnot.llguidance_aprad_repair_protocol.v1"
ARTIFACT = "experiment_3074_llguidance_aprad_repair_protocol_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCRIPT_FILENAME = f"{ARTIFACT}.py"
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME

EXP3056_REL_PATH = Path("results/experiment_3056_repair_de_tautology_protocol_v1.json")
EXP3059_REL_PATH = Path("results/experiment_3059_gated_sota_repair_de_tautology_rerun.json")
CAPSTONE_V286_REL_PATH = Path("results/experiment_3066_capstone_v286.json")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")

REQUIRED_ARTIFACT_FIELDS = (
    "grammar_constrained_repair_protocol_ready",
    "schema_syntax_failure_targets",
    "exact_semantic_validation_required",
    "aprad_intent_preservation_rules",
    "llguidance_runtime_plan",
    "de_tautology_disqualifiers",
    "exp3075_required_fields",
    "inference_substrate",
    "honest_verdict",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

JSON_SOURCES = (
    ("exp3056", EXP3056_REL_PATH, "repair_de_tautology_protocol", True),
    ("exp3059", EXP3059_REL_PATH, "gated_sota_repair_gate_result", True),
    ("exp3066", CAPSTONE_V286_REL_PATH, "capstone_v286_boundary_context", False),
)
TEXT_SOURCES = (
    ("research_references", RESEARCH_REFERENCES_REL_PATH, "llguidance_aprad_reference_context", False),
)

EXP3075_REQUIRED_FIELDS = (
    "schema",
    "artifact",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "matrix_version",
    "protocol_source_artifact",
    "grammar_constrained_repair_protocol_ready",
    "schema_syntax_failure_targets",
    "exact_semantic_validation_required",
    "aprad_intent_preservation_rules",
    "llguidance_runtime_plan",
    "de_tautology_disqualifiers",
    "exp3075_required_fields",
    "blocked_prior_fields_checked",
    "task_intent_hash",
    "behavioral_tests",
    "semantic_drift_checks",
    "verifier_authority",
    "checker_authority",
    "verifier_gain_gate_passed",
    "repair_generation_blocked",
    "clean_blocked_outcome",
    "gate_check_summary",
    "blocked_at_layer",
    "n_tasks",
    "candidate_count",
    "model_specs",
    "decode_config",
    "live_repair_generation_attempted",
    "inference_substrate",
    "tests_run",
    "honest_verdict",
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating missing or malformed evidence as absent.

    Why this is fail-closed: a protocol that cannot parse a required upstream
    artifact must block rather than silently infer that the repair gate is safe.
    """

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_text(path: Path) -> str:
    """Read optional reference text used only to annotate protocol provenance."""

    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    """Return a checksum so downstream artifacts can prove which protocol inputs they used."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3074: build the artifact-only constrained repair protocol."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    json_payloads = {
        experiment_id: read_json_object(root_path / rel_path)
        for experiment_id, rel_path, _role, _required in JSON_SOURCES
    }
    text_payloads = {
        experiment_id: read_text(root_path / rel_path)
        for experiment_id, rel_path, _role, _required in TEXT_SOURCES
    }
    source_artifacts = _source_artifacts(root_path, json_payloads, text_payloads)
    source_errors = _source_errors(source_artifacts)
    exp3056 = json_payloads["exp3056"]
    exp3059 = json_payloads["exp3059"]
    disqualifiers = _de_tautology_disqualifiers(exp3056)
    verifier_gain_gate = _verifier_gain_gate(exp3059)
    failure_targets = _schema_syntax_failure_targets()
    aprad_rules = _aprad_intent_preservation_rules(text_payloads["research_references"])
    runtime_plan = _llguidance_runtime_plan()
    clean_block = _clean_blocked_outcome(source_errors, exp3059, verifier_gain_gate)
    required_fields = list(EXP3075_REQUIRED_FIELDS)
    ready = _ready(
        source_errors=source_errors,
        exp3056=exp3056,
        disqualifiers=disqualifiers,
        failure_targets=failure_targets,
        aprad_rules=aprad_rules,
        runtime_plan=runtime_plan,
        required_fields=required_fields,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "grammar_constrained_repair_protocol_ready": ready,
        "schema_syntax_failure_targets": failure_targets,
        "exact_semantic_validation_required": True,
        "aprad_intent_preservation_rules": aprad_rules,
        "llguidance_runtime_plan": runtime_plan,
        "de_tautology_disqualifiers": disqualifiers,
        "de_tautology_disqualifier_count": len(disqualifiers),
        "exp3075_required_fields": required_fields,
        "exp3075_consumer_contract": _exp3075_consumer_contract(required_fields),
        "clean_blocked_outcome": clean_block,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "missing_source_artifacts": [
            row["path"] for row in source_artifacts if row["required"] and not row["present"]
        ],
        "blocked_reasons": source_errors,
        "no_live_llm_inference": True,
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_historical_artifact_rewrite": True,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "inference_substrate": _inference_substrate(),
        "duration_s": _duration(start, now_s),
        "honest_verdict": _honest_verdict(ready, source_errors, disqualifiers),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3074 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the protocol could be mistaken for a live repair result."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("live_llm_inference") is not False:
        raise ValueError("inference_substrate.live_llm_inference must be false")
    if not artifact.get("de_tautology_disqualifiers"):
        raise ValueError("de_tautology_disqualifiers must not be empty")
    if artifact.get("exact_semantic_validation_required") is not True:
        raise ValueError("exact_semantic_validation_required must be true")
    required_fields = artifact.get("exp3075_required_fields")
    if not isinstance(required_fields, list) or not required_fields:
        raise ValueError("exp3075_required_fields must list required_fields")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("grammar_constrained_repair_protocol_ready") is True:
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must start with a terminal success prefix")
        return
    if not verdict.startswith("blocked_missing_source:"):
        raise ValueError("honest_verdict must disclose blocked_missing_source")


def _source_artifacts(
    root: Path,
    json_payloads: Mapping[str, Mapping[str, Any]],
    text_payloads: Mapping[str, str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for experiment_id, rel_path, role, required in JSON_SOURCES:
        path = root / rel_path
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": rel_path.as_posix(),
                "role": role,
                "kind": "json",
                "required": required,
                "present": path.is_file(),
                "readable": bool(json_payloads[experiment_id]),
                "sha256": sha256_file(path),
            }
        )
    for experiment_id, rel_path, role, required in TEXT_SOURCES:
        path = root / rel_path
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": rel_path.as_posix(),
                "role": role,
                "kind": "text",
                "required": required,
                "present": path.is_file(),
                "readable": bool(text_payloads[experiment_id]),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _source_errors(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {"path": str(row["path"]), "reason": "missing_or_unreadable_required_source"}
        for row in source_artifacts
        if row["required"] and (not row["present"] or not row["readable"])
    ]


def _de_tautology_disqualifiers(exp3056: Mapping[str, Any]) -> list[JsonDict]:
    rows = exp3056.get("promotion_disqualifiers")
    if not isinstance(rows, list):
        return []
    return [
        {
            **dict(row),
            "source_artifact": EXP3056_REL_PATH.as_posix(),
            "carried_forward_to": "exp3075_matrix_v21",
        }
        for row in rows
        if isinstance(row, Mapping) and row.get("id")
    ]


def _schema_syntax_failure_targets() -> JsonDict:
    return {
        "schema_errors": {
            "artifact_field": "schema_failure_rate_delta",
            "measured": True,
            "promotion_target": "<= 0 versus baseline",
            "principle": "schema-invalid repairs cannot promote",
        },
        "syntax_errors": {
            "artifact_field": "syntax_failure_rate_delta",
            "measured": True,
            "promotion_target": "<= 0 versus baseline",
            "principle": "parser-invalid code cannot promote",
        },
        "parse_failures": {
            "artifact_field": "parse_failure_count",
            "measured": True,
            "fallback_behavior": "deterministic_schema_validation",
            "principle": "grammar backend failures stay visible in matrix v21",
        },
        "grammar_backend_failures": {
            "artifact_field": "grammar_backend_failure_count",
            "measured": True,
            "promotion_target": "0 for live grammar-constrained headline repair",
        },
    }


def _aprad_intent_preservation_rules(research_references: str) -> list[JsonDict]:
    reference_available = "AprAD" in research_references or "intent preservation" in research_references
    return [
        {
            "id": "task_intent_hash_required",
            "field": "task_intent_hash",
            "required": True,
            "rule": "Hash the original prompt, entry point, tests, and expected behavior before repair.",
            "reference_available": reference_available,
        },
        {
            "id": "behavioral_tests_required",
            "field": "behavioral_tests",
            "required": True,
            "rule": "Run original and independent behavioral tests before accepting a syntax-valid patch.",
        },
        {
            "id": "semantic_drift_checks_required",
            "field": "semantic_drift_checks",
            "required": True,
            "rule": "Reject repairs that pass tests by changing requested behavior or dropping constraints.",
        },
        {
            "id": "independent_verifier_authority_required",
            "field": "verifier_authority",
            "required": True,
            "rule": "Use deterministic tests or exact verifier authority; the generator cannot grade itself.",
        },
    ]


def _llguidance_runtime_plan() -> JsonDict:
    return {
        "grammar_source": "exp3074_json_schema_to_llguidance_or_gbnf",
        "constrained_syntax_target": "single_repair_candidate_json",
        "schema_validation": {
            "required": True,
            "validator": "deterministic_json_schema_validation",
            "required_fields": [
                "task_id",
                "task_intent_hash",
                "patch",
                "behavioral_tests",
                "semantic_drift_checks",
                "verifier_authority",
            ],
        },
        "parse_failures": {
            "record_field": "parse_failure_count",
            "record_examples_field": "parse_failure_examples",
            "must_block_promotion": True,
        },
        "fallback_behavior": {
            "on_backend_unavailable": (
                "emit unconstrained draft to deterministic JSON-schema validator only; do not promote"
            ),
            "on_parse_failure": "record parse failure and block candidate acceptance",
            "on_schema_failure": "record schema failure and block candidate acceptance",
        },
        "backend_order": ["llguidance", "llama_cpp_gbnf", "deterministic_schema_validation"],
        "live_generation_allowed_by_this_protocol": False,
        "claims_llguidance_implementation": False,
    }


def _verifier_gain_gate(exp3059: Mapping[str, Any]) -> JsonDict:
    for row in exp3059.get("gates_evaluated") or []:
        if isinstance(row, Mapping) and row.get("artifact_field") == "verifier_gain_delta":
            return {
                "upstream": str(row.get("upstream") or ""),
                "artifact_field": "verifier_gain_delta",
                "op": str(row.get("op") or ""),
                "expected": row.get("expected"),
                "actual": row.get("actual"),
                "passed": bool(row.get("passed")),
            }
    return {
        "upstream": "exp3057-local-sota-solution-verifier-gain-panel",
        "artifact_field": "verifier_gain_delta",
        "op": ">",
        "expected": 0.0,
        "actual": None,
        "passed": False,
    }


def _clean_blocked_outcome(
    source_errors: list[Mapping[str, Any]],
    exp3059: Mapping[str, Any],
    verifier_gain_gate: Mapping[str, Any],
) -> JsonDict:
    if source_errors:
        return {
            "outcome": "blocked_missing_source",
            "repair_generation_blocked": True,
            "triggered_by_exp3059": False,
            "blocked_reasons": list(source_errors),
        }
    gate_failed = verifier_gain_gate.get("passed") is not True
    return {
        "outcome": "blocked_verifier_gain_gate_failed" if gate_failed else "eligible_for_exp3075",
        "repair_generation_blocked": gate_failed,
        "triggered_by_exp3059": str(exp3059.get("status") or "") == "blocked" or gate_failed,
        "blocked_at_layer": str(exp3059.get("blocked_at_layer") or "protocol_pre_gate"),
        "gate_check_summary": str(exp3059.get("gate_check_summary") or ""),
        "verifier_gain_gate": dict(verifier_gain_gate),
        "exp3075_action": (
            "write terminal blocked artifact without live repair generation"
            if gate_failed
            else "may evaluate remaining preconditions before live generation"
        ),
    }


def _exp3075_consumer_contract(required_fields: list[str]) -> JsonDict:
    return {
        "consumer_experiment": "exp3075",
        "matrix_version": "v21",
        "consumer_ready": True,
        "required_fields": required_fields,
        "clean_blocked_outcome_required": True,
        "blocked_outcome_rule": (
            "If verifier_gain_delta <= 0 or a verifier-gain gate is failed, Exp 3075 must "
            "write repair_generation_blocked=true and skip live repair generation."
        ),
    }


def _inference_substrate() -> JsonDict:
    return {
        "mode": "artifact_only_repair_protocol",
        "protocol_only": True,
        "live_llm_inference": False,
        "local_gguf_inference": False,
        "model_load_attempted": False,
        "fresh_verifier_scoring": False,
        "fresh_solver_execution": False,
        "conductor_invoked": False,
        "source_artifacts_only": True,
    }


def _ready(
    *,
    source_errors: list[Mapping[str, Any]],
    exp3056: Mapping[str, Any],
    disqualifiers: list[Mapping[str, Any]],
    failure_targets: Mapping[str, Any],
    aprad_rules: list[Mapping[str, Any]],
    runtime_plan: Mapping[str, Any],
    required_fields: list[str],
) -> bool:
    return (
        not source_errors
        and exp3056.get("repair_de_tautology_protocol_ready") is True
        and bool(disqualifiers)
        and bool(failure_targets)
        and len(aprad_rules) >= 4
        and bool(runtime_plan.get("grammar_source"))
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(required_fields)
    )


def _honest_verdict(
    ready: bool,
    source_errors: list[Mapping[str, Any]],
    disqualifiers: list[Mapping[str, Any]],
) -> str:
    if ready:
        return (
            "complete: grammar_constrained_repair_protocol_ready=true; "
            f"de_tautology_disqualifiers={len(disqualifiers)}; exp3075_fields={len(EXP3075_REQUIRED_FIELDS)}"
        )
    paths = ",".join(str(row["path"]) for row in source_errors) or "protocol_preconditions"
    return f"blocked_missing_source: {paths}"


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
