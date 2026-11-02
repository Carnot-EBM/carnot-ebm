"""Exp 2950 repair-prompt manifest for failed SOTA code candidates.

This module does not run a model and does not claim a pass-rate improvement.
It aggregates the checked-in Exp 2940, Exp 2943, and Exp 2946 artifacts into a
repair manifest that downstream live SOTA GGUF evaluation can consume.

Spec: REQ-CODE-2950, SCENARIO-CODE-2950.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2950_code_taxonomy_repair_prompt_manifest_v1.json"
ARTIFACT = "experiment_2950_code_taxonomy_repair_prompt_manifest_v1"
SCHEMA = "carnot.code_taxonomy_repair_prompt_manifest.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP2940_REL_PATH = Path("results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json")
EXP2943_REL_PATH = Path("results/experiment_2943_cross_corpus_matrix_v11.json")
EXP2946_REL_PATH = Path("results/experiment_2946_sota_code_generation_continuation_v1.json")
NESTED_EXP2946_REL_PATH = Path("results/experiment_2946_nested_exp2910_protocol.json")

TAXONOMY_LABELS = (
    "syntax_error",
    "missing_symbol",
    "wrong_return_type",
    "failed_tests",
    "unsafe_import",
    "unsupported_api_hallucination",
)

MODEL_SPECS = (
    {
        "name": "Qwen3.6-35B-A3B-GGUF",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "downstream_live_repair_generation",
    },
    {
        "name": "gemma-4-31B-it-GGUF",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "downstream_live_repair_generation",
    },
    {
        "name": "gemma-4-26B-A4B-it-GGUF",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "downstream_live_repair_generation",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "repair_prompt_manifest_ready",
    "source_artifacts",
    "model_specs",
    "legacy_models_only_for_smoke",
    "taxonomy_labels",
    "repair_prompt_templates",
    "deterministic_checks",
    "downstream_eval_plan",
    "inference_substrate",
    "duration_s",
)

UNSAFE_IMPORTS = frozenset(
    {
        "os",
        "pathlib",
        "shutil",
        "socket",
        "subprocess",
        "sys",
    }
)

LABEL_DESCRIPTIONS = {
    "syntax_error": "Candidate cannot be parsed as Python after extraction.",
    "missing_symbol": "Candidate references a name that is not defined in scope.",
    "wrong_return_type": "Candidate returns a value incompatible with the task contract.",
    "failed_tests": "Candidate parses and runs, but deterministic task tests fail.",
    "unsafe_import": "Candidate imports or reaches unsafe local-system functionality.",
    "unsupported_api_hallucination": "Candidate calls an invented or unsupported API surface.",
}

LABEL_CHECKS = {
    "syntax_error": ("parser_ast_parse", "function_extraction"),
    "missing_symbol": ("parser_ast_parse", "static_import_name_checks"),
    "wrong_return_type": ("parser_ast_parse", "return_type_contract_probe"),
    "failed_tests": ("parser_ast_parse", "tests_where_present"),
    "unsafe_import": ("parser_ast_parse", "static_import_name_checks"),
    "unsupported_api_hallucination": (
        "parser_ast_parse",
        "unsupported_api_static_attr_check",
    ),
}

TEMPLATE_FOCUS = {
    "syntax_error": "Repair only malformed Python structure, indentation, fences, or truncation.",
    "missing_symbol": "Define the missing local symbol or replace it with an existing variable.",
    "wrong_return_type": "Keep the original signature and return the expected type directly.",
    "failed_tests": "Use the failing assertion evidence to correct behavior, not formatting.",
    "unsafe_import": "Remove unsafe imports and replace side effects with pure local logic.",
    "unsupported_api_hallucination": "Replace invented APIs with supported standard-library calls.",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2950 manifest builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp2940_path: Path = EXP2940_REL_PATH
    exp2943_path: Path = EXP2943_REL_PATH
    exp2946_path: Path = EXP2946_REL_PATH
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the repair-prompt manifest from checked-in upstream artifacts."""

    config = config or ExperimentConfig()
    started = config.start_time()
    source_artifacts = _required_source_artifacts(config)
    missing_required = any(source["required"] and not source["present"] for source in source_artifacts)
    if missing_required:
        return _blocked_artifact(config, started, source_artifacts)

    exp2940 = _read_json(_repo_path(config.repo_root, config.exp2940_path))
    exp2943 = _read_json(_repo_path(config.repo_root, config.exp2943_path))
    exp2946 = _read_json(_repo_path(config.repo_root, config.exp2946_path))
    nested_rel_path = _nested_protocol_path(exp2946)
    source_artifacts.append(
        _source_artifact(config.repo_root, nested_rel_path, "exp2946_nested_candidate_rows", False)
    )
    nested_path = _repo_path(config.repo_root, nested_rel_path)
    nested_protocol = _read_json(nested_path) if nested_path.is_file() else {}
    candidate_rows = _candidate_rows(nested_protocol)
    grouped = _group_candidates(candidate_rows)
    threshold = _number(
        (exp2940.get("max_f1_operating_point") or {}).get("threshold"),
        1.0,
    )

    return _final_artifact(
        config=config,
        started=started,
        ready=True,
        source_artifacts=source_artifacts,
        taxonomy_labels=_taxonomy_rows(grouped),
        repair_prompt_templates=_repair_prompt_templates(),
        deterministic_checks=_deterministic_checks(threshold),
        downstream_eval_plan=_downstream_eval_plan(threshold, blocked_reason=None),
        upstream_metrics={
            "code_corpus_auprc": _number(exp2940.get("code_corpus_auprc"), 0.0),
            "cross_corpus_code_auprc": _number(
                ((exp2943.get("per_corpus_auprc") or {}).get("code_corpora") or {}).get("value"),
                0.0,
            ),
            "pass_at_1": _number(exp2946.get("pass_at_1"), 0.0),
            "pass_at_k": _number(exp2946.get("pass_at_k"), 0.0),
        },
        failure_evidence_summary={
            label: {
                "evidence_count": len(rows),
                "sample_ids": [_sample_id(row) for row in rows[:5]],
            }
            for label, rows in grouped.items()
        },
    )


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 2950 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _required_source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    return [
        _source_artifact(config.repo_root, config.exp2940_path, "code_corpus_verifier_threshold", True),
        _source_artifact(config.repo_root, config.exp2943_path, "cross_corpus_matrix_status", True),
        _source_artifact(config.repo_root, config.exp2946_path, "sota_codegen_continuation", True),
    ]


def _source_artifact(repo_root: Path, rel_path: Path, role: str, required: bool) -> JsonDict:
    path = _repo_path(repo_root, rel_path)
    present = path.is_file()
    return {
        "path": str(rel_path),
        "role": role,
        "required": required,
        "present": present,
        "sha256": _sha256(path) if present else None,
    }


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    source_artifacts: list[JsonDict],
) -> JsonDict:
    return _final_artifact(
        config=config,
        started=started,
        ready=False,
        source_artifacts=source_artifacts,
        taxonomy_labels=[],
        repair_prompt_templates={},
        deterministic_checks=[],
        downstream_eval_plan=_downstream_eval_plan(1.0, blocked_reason="missing_required_source"),
        upstream_metrics={},
        failure_evidence_summary={},
    )


def _final_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    ready: bool,
    source_artifacts: list[JsonDict],
    taxonomy_labels: list[JsonDict],
    repair_prompt_templates: dict[str, JsonDict],
    deterministic_checks: list[JsonDict],
    downstream_eval_plan: JsonDict,
    upstream_metrics: JsonDict,
    failure_evidence_summary: JsonDict,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": (
            "complete: repair prompt manifest ready; no pass-rate improvement claimed"
            if ready
            else "blocked_upstream_artifact_missing"
        ),
        "repair_prompt_manifest_ready": ready,
        "source_artifacts": source_artifacts,
        "model_specs": [dict(model) for model in MODEL_SPECS],
        "legacy_models_only_for_smoke": True,
        "taxonomy_labels": taxonomy_labels,
        "repair_prompt_templates": repair_prompt_templates,
        "deterministic_checks": deterministic_checks,
        "downstream_eval_plan": downstream_eval_plan,
        "acceptance_criteria": downstream_eval_plan["acceptance_criteria"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "upstream_metrics": upstream_metrics,
        "failure_evidence_summary": failure_evidence_summary,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "run_date": RUN_DATE,
        "duration_s": _elapsed(config, started),
    }


def _candidate_rows(nested_protocol: Mapping[str, Any]) -> list[JsonDict]:
    rows = nested_protocol.get("candidate_results")
    return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _group_candidates(candidate_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = {label: [] for label in TAXONOMY_LABELS}
    for row in candidate_rows:
        for label in _candidate_labels(row):
            grouped[label].append(dict(row))
    return grouped


def _candidate_labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    error_type = str(row.get("error_type") or "")
    message = str(row.get("error_message") or "").lower()
    row_status = str(row.get("row_status") or "")
    source = str(row.get("extracted_code") or "")
    labels: set[str] = set()
    if row.get("syntax_success") is False or error_type in {"SyntaxError", "IndentationError"}:
        labels.add("syntax_error")
    if error_type == "NameError" or "not defined" in message:
        labels.add("missing_symbol")
    if "wrong return type" in message or "expected type" in message:
        labels.add("wrong_return_type")
    if row.get("runtime_success") is True and row.get("passed") is False:
        labels.add("failed_tests")
    if error_type == "AssertionError" and row_status == "candidate_failed":
        labels.add("failed_tests")
    if _has_unsafe_import(source) or "unsafe import" in message:
        labels.add("unsafe_import")
    if "has no attribute" in message or "unsupported api" in message:
        labels.add("unsupported_api_hallucination")
    return tuple(label for label in TAXONOMY_LABELS if label in labels)


def _has_unsafe_import(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    imported_roots = [
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imported_roots.extend(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    return any(root in UNSAFE_IMPORTS for root in imported_roots)


def _taxonomy_rows(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for label in TAXONOMY_LABELS:
        evidence_rows = list(grouped.get(label) or [])
        rows.append(
            {
                "label": label,
                "description": LABEL_DESCRIPTIONS[label],
                "evidence_count": len(evidence_rows),
                "evidence_status": (
                    "observed_in_exp2946"
                    if evidence_rows
                    else "template_defined_no_upstream_sample"
                ),
                "sample_ids": [_sample_id(row) for row in evidence_rows[:5]],
                "deterministic_checks": list(LABEL_CHECKS[label]),
            }
        )
    return rows


def _sample_id(row: Mapping[str, Any]) -> str:
    corpus = str(row.get("corpus") or "unknown")
    stable_id = str(row.get("stable_id") or "unknown")
    candidate_index = str(row.get("candidate_index") if row.get("candidate_index") is not None else "na")
    random_seed = str(row.get("random_seed") if row.get("random_seed") is not None else "na")
    return f"{corpus}:{stable_id}:c{candidate_index}:s{random_seed}"


def _repair_prompt_templates() -> dict[str, JsonDict]:
    return {
        label: {
            "template_id": f"{label}_repair_v1",
            "required_context_fields": [
                "sample_id",
                "task_prompt",
                "candidate_code",
                "failure_evidence",
                "deterministic_checks",
            ],
            "template": (
                f"Taxonomy label: {label}\n"
                "Sample: {sample_id}\n"
                "Failure evidence: {failure_evidence}\n"
                "Task context: {task_prompt}\n"
                "Candidate code:\n{candidate_code}\n"
                f"Repair focus: {TEMPLATE_FOCUS[label]}\n"
                "Return only corrected Python code. Do not introduce new imports "
                "unless the task already permits them. Preserve the public function "
                "signature and satisfy the listed deterministic checks."
            ),
        }
        for label in TAXONOMY_LABELS
    }


def _deterministic_checks(exp2940_threshold: float) -> list[JsonDict]:
    return [
        {
            "check_id": "parser_ast_parse",
            "description": "The repaired candidate must parse with Python ast.parse.",
            "required": True,
            "applies_to": list(TAXONOMY_LABELS),
        },
        {
            "check_id": "function_extraction",
            "description": "The repaired output must contain the expected entry-point function.",
            "required": True,
            "applies_to": ["syntax_error"],
        },
        {
            "check_id": "static_import_name_checks",
            "description": "Static import and name checks must find no unsafe imports or missing names.",
            "required": True,
            "applies_to": ["missing_symbol", "unsafe_import"],
        },
        {
            "check_id": "return_type_contract_probe",
            "description": "Where a signature or oracle states a type, probes must return that type.",
            "required": True,
            "applies_to": ["wrong_return_type"],
        },
        {
            "check_id": "tests_where_present",
            "description": "Official or manifest-local tests must pass when the row provides tests.",
            "required": True,
            "applies_to": ["failed_tests"],
        },
        {
            "check_id": "unsupported_api_static_attr_check",
            "description": "Imported module attributes must exist on the supported local API surface.",
            "required": True,
            "applies_to": ["unsupported_api_hallucination"],
        },
        {
            "check_id": "exp2940_verifier_threshold",
            "description": "The code-corpus verifier score must meet the retained Exp 2940 threshold.",
            "required": True,
            "threshold": exp2940_threshold,
            "source_artifact": str(EXP2940_REL_PATH),
            "applies_to": list(TAXONOMY_LABELS),
        },
    ]


def _downstream_eval_plan(exp2940_threshold: float, blocked_reason: str | None) -> JsonDict:
    return {
        "blocked_reason": blocked_reason,
        "substrate_required_for_next_step": "live_sota_gguf_repair_evaluation",
        "model_hf_ids": [model["hf_id"] for model in MODEL_SPECS],
        "legacy_model_policy": "Legacy tiny models are allowed only for CPU smoke tests.",
        "may_claim_this_manifest_improves_pass_rate": False,
        "acceptance_criteria": _acceptance_criteria(exp2940_threshold),
    }


def _acceptance_criteria(exp2940_threshold: float) -> list[str]:
    return [
        "Every repair candidate must pass parser_ast_parse.",
        "Rows with local tests must pass tests_where_present before acceptance.",
        "Static import, name, and API checks must be clean for the assigned label.",
        f"Verifier approval must meet the Exp 2940 threshold >= {exp2940_threshold:.4f}.",
        "Downstream reports baseline and repaired pass metrics separately from this manifest.",
    ]


def _nested_protocol_path(exp2946: Mapping[str, Any]) -> Path:
    raw_path = str(exp2946.get("protocol_artifact_path") or NESTED_EXP2946_REL_PATH)
    return Path(raw_path)


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)
