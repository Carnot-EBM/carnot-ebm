"""Exp 2963 DCCD structured-repair protocol manifest.

This module prepares the next live SOTA code-repair replication without running
a model and without treating the small flagged Exp 2952 delta as a result.  It
aggregates the .278 repair taxonomy, structured candidate schema, threshold
policy, and known false-accept warnings into one pre-registered DCCD protocol.

Spec: REQ-CODE-2963, SCENARIO-CODE-2963.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2963_dccd_repair_protocol_manifest_v1.json"
ARTIFACT = "experiment_2963_dccd_repair_protocol_manifest_v1"
SCHEMA = "carnot.dccd_repair_protocol_manifest.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP2950_REL_PATH = Path("results/experiment_2950_code_taxonomy_repair_prompt_manifest_v1.json")
EXP2951_REL_PATH = Path("results/experiment_2951_structured_candidate_manifest_adapter_v1.json")
EXP2952_REL_PATH = Path("results/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json")
EXP2953_REL_PATH = Path("results/experiment_2953_code_verifier_threshold_policy_v1.json")

MODEL_SPECS = (
    {
        "name": "Qwen3.6-35B-A3B-GGUF",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "downstream_live_dccd_repair_generation",
    },
    {
        "name": "gemma-4-31B-it-GGUF",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "downstream_live_dccd_repair_generation",
    },
    {
        "name": "gemma-4-26B-A4B-it-GGUF",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "downstream_live_dccd_repair_generation",
    },
)

N_TASKS_PLANNED_MIN = 20
FIXED_SEED_PLAN = tuple(range(296300, 296320))
ACCOUNTING_BUCKETS = (
    "pass_at_1",
    "pass_at_k",
    "syntax_failures",
    "schema_failures",
    "test_failures",
    "verifier_only_accepts",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "dccd_repair_protocol_ready",
    "source_artifacts",
    "model_specs",
    "legacy_models_only_for_smoke",
    "n_tasks_planned_min",
    "fixed_seed_plan",
    "dccd_steps",
    "structured_backends_to_check",
    "deterministic_acceptance_checks",
    "false_accept_audit_plan",
    "downstream_gate",
    "inference_substrate",
    "duration_s",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the deterministic Exp 2963 artifact builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float = field(default_factory=time.time)
    clock: Callable[[], float] = time.time

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the DCCD protocol manifest from checked-in .278 artifacts."""

    config = config or ExperimentConfig()
    source_artifacts = _source_artifacts(config)
    missing_sources = [
        source["experiment_id"]
        for source in source_artifacts
        if source["required"] and not source["present"]
    ]
    if missing_sources:
        return _blocked_artifact(
            config,
            "blocked_missing_required_source_artifact",
            source_artifacts,
            missing_sources,
            [],
        )

    payloads: dict[str, JsonDict] = {}
    malformed_sources: list[str] = []
    for source in source_artifacts:
        try:
            payloads[source["experiment_id"]] = _read_json(
                _repo_path(config.repo_root, Path(source["path"]))
            )
        except ValueError:
            malformed_sources.append(source["experiment_id"])
    if malformed_sources:
        return _blocked_artifact(
            config,
            "blocked_malformed_source_artifact",
            source_artifacts,
            [],
            malformed_sources,
        )

    threshold_policy = _threshold_policy(payloads["exp2953"])
    return _final_artifact(
        config=config,
        ready=True,
        verdict="complete: DCCD structured-repair protocol ready; no pass-rate improvement claimed",
        source_artifacts=source_artifacts,
        missing_sources=[],
        malformed_sources=[],
        extracted_failure_taxonomy=_extracted_failure_taxonomy(payloads["exp2950"]),
        candidate_schema_summary=_candidate_schema_summary(payloads["exp2951"]),
        repair_delta_summary=_repair_delta_summary(payloads["exp2952"]),
        threshold_policy=threshold_policy,
        dccd_steps=_dccd_steps(),
        structured_backends_to_check=_structured_backends(payloads["exp2951"]),
        deterministic_acceptance_checks=_deterministic_checks(payloads["exp2950"]),
        false_accept_audit_plan=_false_accept_audit_plan(payloads["exp2952"]),
        downstream_gate=_downstream_gate(threshold_policy["selected_default_threshold"]),
    )


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 2963 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _blocked_artifact(
    config: ExperimentConfig,
    verdict: str,
    source_artifacts: list[JsonDict],
    missing_sources: list[str],
    malformed_sources: list[str],
) -> JsonDict:
    return _final_artifact(
        config=config,
        ready=False,
        verdict=verdict,
        source_artifacts=source_artifacts,
        missing_sources=missing_sources,
        malformed_sources=malformed_sources,
        extracted_failure_taxonomy={},
        candidate_schema_summary={
            "schema_version": None,
            "schema_fields": [],
            "required_fields": [],
        },
        repair_delta_summary={},
        threshold_policy=_empty_threshold_policy(),
        dccd_steps=[],
        structured_backends_to_check=[],
        deterministic_acceptance_checks=[],
        false_accept_audit_plan=_empty_false_accept_plan(),
        downstream_gate=_downstream_gate(None),
    )


def _final_artifact(
    *,
    config: ExperimentConfig,
    ready: bool,
    verdict: str,
    source_artifacts: list[JsonDict],
    missing_sources: list[str],
    malformed_sources: list[str],
    extracted_failure_taxonomy: JsonDict,
    candidate_schema_summary: JsonDict,
    repair_delta_summary: JsonDict,
    threshold_policy: JsonDict,
    dccd_steps: list[JsonDict],
    structured_backends_to_check: list[JsonDict],
    deterministic_acceptance_checks: list[JsonDict],
    false_accept_audit_plan: JsonDict,
    downstream_gate: JsonDict,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "dccd_repair_protocol_ready": ready,
        "source_artifacts": source_artifacts,
        "model_specs": [dict(model) for model in MODEL_SPECS],
        "legacy_models_only_for_smoke": True,
        "n_tasks_planned_min": N_TASKS_PLANNED_MIN,
        "fixed_seed_plan": list(FIXED_SEED_PLAN),
        "dccd_steps": dccd_steps,
        "structured_backends_to_check": structured_backends_to_check,
        "deterministic_acceptance_checks": deterministic_acceptance_checks,
        "false_accept_audit_plan": false_accept_audit_plan,
        "downstream_gate": downstream_gate,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config),
        "extracted_failure_taxonomy": extracted_failure_taxonomy,
        "candidate_schema_summary": candidate_schema_summary,
        "repair_delta_summary": repair_delta_summary,
        "threshold_policy": threshold_policy,
        "missing_source_artifacts": missing_sources,
        "malformed_source_artifacts": malformed_sources,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    specs = (
        (
            "exp2950",
            EXP2950_REL_PATH,
            "failure_taxonomy_and_repair_checks",
            ["taxonomy_labels", "deterministic_checks", "downstream_eval_plan"],
        ),
        (
            "exp2951",
            EXP2951_REL_PATH,
            "candidate_schema_and_structured_backends",
            ["schema_fields", "candidate_manifest_schema", "local_backends_checked"],
        ),
        (
            "exp2952",
            EXP2952_REL_PATH,
            "flagged_repair_delta_and_false_accepts",
            ["candidate_evaluations", "corrigendum_pending", "false_accept_audit_notes"],
        ),
        (
            "exp2953",
            EXP2953_REL_PATH,
            "verifier_threshold_policy",
            ["selected_default_threshold", "operating_points", "deployment_boundary"],
        ),
    )
    return [_source_artifact(config.repo_root, *spec) for spec in specs]


def _source_artifact(
    repo_root: Path,
    experiment_id: str,
    rel_path: Path,
    role: str,
    fields_imported: list[str],
) -> JsonDict:
    path = _repo_path(repo_root, rel_path)
    present = path.is_file()
    return {
        "experiment_id": experiment_id,
        "path": rel_path.as_posix(),
        "role": role,
        "required": True,
        "present": present,
        "sha256": _sha256(path) if present else None,
        "fields_imported": fields_imported,
    }


def _extracted_failure_taxonomy(exp2950: Mapping[str, Any]) -> JsonDict:
    taxonomy: JsonDict = {}
    for row in exp2950.get("taxonomy_labels", []):
        label = row["label"]
        taxonomy[label] = {
            "description": row.get("description"),
            "evidence_count": int(row.get("evidence_count", 0)),
            "sample_ids": list(row.get("sample_ids", [])),
            "deterministic_checks": list(row.get("deterministic_checks", [])),
        }
    return taxonomy


def _candidate_schema_summary(exp2951: Mapping[str, Any]) -> JsonDict:
    schema = exp2951.get("candidate_manifest_schema", {})
    return {
        "schema_version": exp2951.get("schema_version"),
        "schema_fields": list(exp2951.get("schema_fields", [])),
        "required_fields": list(schema.get("required", [])),
        "structured_decode_manifest_ready": bool(exp2951.get("structured_decode_manifest_ready")),
    }


def _repair_delta_summary(exp2952: Mapping[str, Any]) -> JsonDict:
    return {
        "upstream_n_tasks": exp2952.get("n_tasks"),
        "upstream_selected_task_ids": list(exp2952.get("selected_task_ids", [])),
        "flagged_adversarial": bool(exp2952.get("flagged_adversarial")),
        "baseline_pass_at_1": exp2952.get("baseline_pass_at_1"),
        "repair_pass_at_1": exp2952.get("repair_pass_at_1"),
        "pass_at_1_delta": exp2952.get("pass_at_1_delta"),
        "baseline_pass_at_k": exp2952.get("baseline_pass_at_k"),
        "repair_pass_at_k": exp2952.get("repair_pass_at_k"),
        "pass_at_k_delta": exp2952.get("pass_at_k_delta"),
        "syntax_failure_rate_delta": exp2952.get("syntax_failure_rate_delta"),
        "schema_failure_rate_delta": exp2952.get("schema_failure_rate_delta"),
        "false_accept_delta": exp2952.get("false_accept_delta"),
        "may_claim_pass_rate_improvement": False,
        "reason": "Exp 2952 was positive but adversarially flagged; Exp 2963 only pre-registers repair.",
    }


def _threshold_policy(exp2953: Mapping[str, Any]) -> JsonDict:
    return {
        "threshold_policy_ready": bool(exp2953.get("threshold_policy_ready")),
        "selected_default_threshold": exp2953.get("selected_default_threshold"),
        "expected_ppv_at_default": exp2953.get("expected_ppv_at_default"),
        "expected_recall_at_default": exp2953.get("expected_recall_at_default"),
        "expected_false_accept_rate_at_default": exp2953.get(
            "expected_false_accept_rate_at_default"
        ),
        "deployment_boundary": exp2953.get("deployment_boundary"),
        "operating_points": list(exp2953.get("operating_points", [])),
    }


def _empty_threshold_policy() -> JsonDict:
    return {
        "threshold_policy_ready": False,
        "selected_default_threshold": None,
        "expected_ppv_at_default": None,
        "expected_recall_at_default": None,
        "expected_false_accept_rate_at_default": None,
        "deployment_boundary": None,
        "operating_points": [],
    }


def _dccd_steps() -> list[JsonDict]:
    return [
        {
            "step_id": "unconstrained_semantic_draft",
            "purpose": "Generate an unconstrained draft so semantic intent is not damaged by early masks.",
            "source": "DCCD",
        },
        {
            "step_id": "taxonomy_conditioned_repair",
            "purpose": "Repair the draft using the Exp 2950 failure label and repair focus.",
            "source": "exp2950",
        },
        {
            "step_id": "constrained_manifest_emission",
            "purpose": "Emit only the Exp 2951 structured candidate manifest fields.",
            "source": "exp2951",
        },
        {
            "step_id": "parser_static_test_checks",
            "purpose": "Run parser, static import/name/API checks, and task tests where present.",
            "source": "exp2950_exp2952",
        },
        {
            "step_id": "verifier_threshold_check",
            "purpose": "Apply the Exp 2953 conservative threshold after deterministic code checks.",
            "source": "exp2953",
        },
        {
            "step_id": "false_accept_audit",
            "purpose": "Separate verifier-only accepts from pass@1/pass@k and audit known false-accept patterns.",
            "source": "exp2952",
        },
    ]


def _structured_backends(exp2951: Mapping[str, Any]) -> list[JsonDict]:
    upstream = {row["backend_name"]: row for row in exp2951.get("local_backends_checked", [])}
    return [
        {
            "backend_name": "llguidance",
            "must_check_downstream": True,
            "upstream_available": bool(upstream.get("llguidance", {}).get("available")),
            "role": "preferred local JSON-schema/CFG constrained decoding backend when installed",
        },
        {
            "backend_name": "llama_cpp_grammar",
            "must_check_downstream": True,
            "upstream_available": bool(upstream.get("llama_cpp_grammar", {}).get("available")),
            "role": "llama.cpp grammar emission path for local GGUF generation",
        },
        {
            "backend_name": "json_schema_validation_fallback",
            "must_check_downstream": True,
            "upstream_available": bool(upstream.get("jsonschema", {}).get("available")),
            "role": "deterministic post-decode schema validation fallback",
        },
    ]


def _deterministic_checks(exp2950: Mapping[str, Any]) -> list[JsonDict]:
    checks = [dict(row) for row in exp2950.get("deterministic_checks", [])]
    checks.extend(
        [
            {
                "check_id": "structured_candidate_schema_validation",
                "required": True,
                "description": "The constrained emission must validate against the Exp 2951 manifest schema.",
            },
            {
                "check_id": "verifier_only_accept_rejection",
                "required": True,
                "description": "Verifier-approved rows that fail parser/static/tests stay in verifier-only accounting.",
            },
            {
                "check_id": "pass_metric_bucket_separation",
                "required": True,
                "description": "pass@1/pass@k, syntax, schema, test, and verifier-only buckets are reported separately.",
            },
        ]
    )
    return checks


def _false_accept_audit_plan(exp2952: Mapping[str, Any]) -> JsonDict:
    known_constraints = [
        {
            "mode": row.get("mode"),
            "sample_id": row.get("sample_id"),
            "seed": row.get("seed"),
            "parser_status": row.get("parser_status"),
            "test_status": row.get("test_status"),
            "verifier_score": row.get("verifier_score"),
            "verifier_accepted": row.get("verifier_accepted"),
            "passed": row.get("passed"),
        }
        for row in exp2952.get("candidate_evaluations", [])
        if row.get("false_accept")
    ]
    corrigendum_kinds = sorted(
        {row.get("kind") for row in exp2952.get("corrigendum_pending", []) if row.get("kind")}
    )
    return {
        "upstream_known_false_accept_count": len(known_constraints),
        "known_false_accept_constraints": known_constraints,
        "corrigendum_kinds_to_audit": corrigendum_kinds,
        "audit_dimensions": [
            "verifier_only_accepts",
            "parser_or_schema_failure_accepted",
            "test_failure_accepted",
            "metric_tautology",
            "missing_seed_or_checksum",
        ],
        "required_conditions": {
            "verifier_only_accepts_must_not_count_as_pass": True,
            "false_accept_delta_must_be_nonpositive": True,
            "every_accept_requires_parser_static_and_available_tests": True,
        },
        "upstream_notes": list(exp2952.get("false_accept_audit_notes", [])),
    }


def _empty_false_accept_plan() -> JsonDict:
    return {
        "upstream_known_false_accept_count": 0,
        "known_false_accept_constraints": [],
        "corrigendum_kinds_to_audit": [],
        "audit_dimensions": [],
        "required_conditions": {},
        "upstream_notes": [],
    }


def _downstream_gate(threshold: float | None) -> JsonDict:
    return {
        "requires_fresh_live_replication": True,
        "may_claim_pass_rate_improvement": False,
        "n_tasks_min": N_TASKS_PLANNED_MIN,
        "fixed_seed_plan": list(FIXED_SEED_PLAN),
        "required_model_hf_ids": [model["hf_id"] for model in MODEL_SPECS],
        "legacy_small_models_allowed_only_for_cpu_smoke": True,
        "selected_default_threshold": threshold,
        "accounting_buckets": list(ACCOUNTING_BUCKETS),
        "acceptance_before_claim": [
            "Run the downstream live replication; this manifest is protocol-only.",
            "Report pass@1 and pass@k separately from syntax, schema, test, and verifier-only failures.",
            "Reject any candidate that is verifier-approved but fails parser/static/available task tests.",
        ],
    }


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON source artifact: {path}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _elapsed(config: ExperimentConfig) -> float:
    return max(0.0, config.clock() - config.started_at)
