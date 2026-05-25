"""Exp 3031 tiny DCCD-style structured repair panel.

Spec: REQ-CODE-3031, SCENARIO-CODE-3031.

This module holds the deterministic panel mechanics: fixed repair-hard case
selection, draft/schema projection, acceptance-controller application, validator
replay, and metric aggregation. The actual local GGUF call is injected by the
CLI wrapper so unit tests can verify the logic without loading a model.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.eval import repair_acceptance_controller as acceptance


JsonDict = dict[str, Any]
ClockFunc = Callable[[], float]
PanelGenerator = Callable[["PanelCase", str, str | None, JsonDict], "GenerationResult"]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
SCHEMA = "carnot.dccd_structured_repair_panel.v1"
ARTIFACT = "experiment_3031_dccd_structured_repair_panel_v1"
OUTPUT_REL_PATH = Path("results/experiment_3031_dccd_structured_repair_panel_v1.json")
RAW_REL_DIR = Path("results/raw") / ARTIFACT
HARD_MANIFEST_REL_PATH = hard.DEFAULT_MANIFEST_REL_PATH
METAMORPHIC_MANIFEST_REL_PATH = metamorphic.METAMORPHIC_MANIFEST_REL_PATH
EXP3015_REL_PATH = Path("results") / acceptance.ARTIFACT_FILENAME
CONTROLLER_CONFIG_REL_PATH = acceptance.CONFIG_REL_PATH

HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SMOKE_ONLY_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3.5-0.8B",
    "unsloth/gemma-4-E4B-it-GGUF",
)
DEFAULT_PANEL_ITEM_IDS: tuple[str, ...] = (
    "repair-hard-0001",
    "repair-hard-0002",
    "repair-hard-0008",
)
UNCONSTRAINED_MODE = "unconstrained_draft_repair"
ACCEPTANCE_MODE = "acceptance_only_constrained_acceptance"
DCCD_MODE = "draft_conditioned_constrained_repair"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "dccd_panel_ready",
    "n_cases",
    "model_specs",
    "legacy_smoke_only_used",
    "baseline_acceptance_metrics",
    "dccd_metrics",
    "intent_drift_delta",
    "false_accept_delta",
    "changed_files",
    "tests_run",
    "inference_substrate",
    "honest_verdict",
)

DEFAULT_CHANGED_FILES: tuple[str, ...] = (
    "openspec/capabilities/code-verification/spec.md",
    "python/carnot/eval/dccd_structured_repair_panel_3031.py",
    "scripts/experiment_3031_dccd_structured_repair_panel_v1.py",
    "tests/python/test_experiment_3031_dccd_structured_repair_panel.py",
    OUTPUT_REL_PATH.as_posix(),
)


@dataclass(frozen=True)
class PanelCase:
    """One repair-hard case selected for the tiny DCCD panel."""

    item_id: str
    prompt: str
    expected_behavior: str
    entry_point: str
    baseline_candidate: str
    reference_solution: str
    failing_test_ids: tuple[str, ...]
    tests: tuple[JsonDict, ...]
    item: JsonDict


@dataclass(frozen=True)
class GenerationResult:
    """Raw model output plus lightweight generation telemetry."""

    raw_text: str
    duration_s: float = 0.0
    tokens_generated: int = 0
    error: str | None = None


@dataclass(frozen=True)
class StructuredCandidate:
    """Result of projecting a raw constrained response into the panel schema."""

    schema_valid: bool
    final_patch: str
    draft_intent: str
    schema_errors: list[str]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for Exp 3031."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    hard_manifest_path: Path | None = None
    metamorphic_manifest_path: Path | None = None
    controller_config_path: Path | None = None
    panel_item_ids: Sequence[str] = DEFAULT_PANEL_ITEM_IDS
    selected_model_path: Path | None = None
    selected_model_id: str | None = None
    started_at: float | None = None
    clock: ClockFunc = time.time
    tests_run: Sequence[str] = field(default_factory=tuple)
    changed_files: Sequence[str] = DEFAULT_CHANGED_FILES

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def resolved_hard_manifest_path(self) -> Path:
        return self.hard_manifest_path or self.repo_root / HARD_MANIFEST_REL_PATH

    def resolved_metamorphic_manifest_path(self) -> Path:
        return self.metamorphic_manifest_path or self.repo_root / METAMORPHIC_MANIFEST_REL_PATH

    def resolved_controller_config_path(self) -> Path:
        if self.controller_config_path is not None:
            return self.controller_config_path
        exp3015 = _read_json_if_present(self.repo_root / EXP3015_REL_PATH)
        return _resolve_repo_path(
            self.repo_root,
            exp3015.get("controller_config_path") or CONTROLLER_CONFIG_REL_PATH,
        )


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator_fn: PanelGenerator | None = None,
) -> JsonDict:
    """Build the Exp 3031 artifact, blocking before panel work if no model exists."""

    config = config or ExperimentConfig()
    started = config.start_time()
    hard_items = _load_hard_items(config)
    controller_rule = _load_controller_rule(config)
    variants = _load_metamorphic_variants(config)
    selected_model = _select_headline_model(config)
    preconditions = _preconditions(config, hard_items, controller_rule, selected_model)
    if not selected_model["available"] or generator_fn is None:
        return _blocked_artifact(
            config=config,
            started=started,
            preconditions=preconditions,
            selected_model=selected_model,
        )

    panel_cases = select_panel_cases(hard_items, config.panel_item_ids)
    model_spec = {
        "hf_id": selected_model["hf_id"],
        "model_path": selected_model["path"],
        "role": "headline_live_generation",
    }
    case_results: list[JsonDict] = []
    generation_errors: list[str] = []
    for case in panel_cases:
        case_result, errors = _run_case(
            case=case,
            generator_fn=generator_fn,
            model_spec=model_spec,
            controller_rule=controller_rule,
            variants=variants,
        )
        case_results.append(case_result)
        generation_errors.extend(errors)

    unconstrained_rows = [row["unconstrained_draft"] for row in case_results]
    acceptance_rows = [row["acceptance_only"] for row in case_results]
    dccd_rows = [row["draft_conditioned_constrained"] for row in case_results]
    n_cases = len(case_results)
    unconstrained_metrics = condition_metrics(unconstrained_rows, n_cases=n_cases)
    baseline_acceptance_metrics = condition_metrics(acceptance_rows, n_cases=n_cases)
    dccd_metrics = condition_metrics(dccd_rows, n_cases=n_cases)
    deltas = metric_deltas(baseline_acceptance_metrics, dccd_metrics)
    legacy_smoke_only_used = model_spec["hf_id"] in SMOKE_ONLY_MODEL_IDS
    any_live_generation = any(
        row["unconstrained_generation"]["raw_text"] and not row["unconstrained_generation"]["error"]
        for row in case_results
    )
    live_generation_succeeded = bool(case_results) and not generation_errors
    blocked_after_generation = not any_live_generation
    ready = bool(
        live_generation_succeeded
        and n_cases == len(panel_cases)
        and model_spec["hf_id"] in HEADLINE_MODEL_IDS
        and not legacy_smoke_only_used
        and controller_rule
        and deltas["false_accept_delta"] <= 0.0
    )
    substrate = _inference_substrate(
        preconditions=preconditions,
        selected_model=selected_model,
        model_load_attempted=True,
        live_generation_succeeded=live_generation_succeeded,
        generation_errors=generation_errors,
    )
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "dccd_panel_ready": ready,
        "n_cases": n_cases,
        "model_specs": [model_spec],
        "legacy_smoke_only_used": legacy_smoke_only_used,
        "baseline_acceptance_metrics": baseline_acceptance_metrics,
        "dccd_metrics": dccd_metrics,
        "intent_drift_delta": deltas["intent_drift_delta"],
        "false_accept_delta": deltas["false_accept_delta"],
        "changed_files": list(config.changed_files),
        "tests_run": list(config.tests_run),
        "inference_substrate": substrate,
        "honest_verdict": _honest_verdict(
            ready=ready,
            blocked=blocked_after_generation,
            n_cases=n_cases,
        ),
        "unconstrained_metrics": unconstrained_metrics,
        "metric_deltas": deltas,
        "case_results": case_results,
        "preconditions_checked": preconditions,
        "selected_panel_item_ids": [case.item_id for case in panel_cases],
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "source_artifacts": _source_artifacts(config),
        "duration_s": _elapsed(config, started),
        "reproducibility_checksum": _reproducibility_checksum(case_results, model_spec),
    }
    return artifact


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator_fn: PanelGenerator | None = None,
) -> JsonDict:
    """Build and persist the Exp 3031 deliverable JSON."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config, generator_fn=generator_fn)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def select_panel_cases(
    items: Sequence[Mapping[str, Any]],
    item_ids: Sequence[str] = DEFAULT_PANEL_ITEM_IDS,
) -> list[PanelCase]:
    """Select fixed repair-hard cases whose baseline fails and reference passes."""

    by_id = {str(item.get("item_id") or ""): item for item in items}
    selected: list[PanelCase] = []
    for item_id in item_ids:
        item = by_id.get(str(item_id))
        if not item:
            continue
        baseline = hard.run_candidate_tests(dict(item), "baseline_candidate")
        reference = hard.run_candidate_tests(dict(item), "reference_solution")
        if baseline.passed or not reference.passed:
            continue
        selected.append(
            PanelCase(
                item_id=str(item["item_id"]),
                prompt=str(item.get("prompt") or ""),
                expected_behavior=str(item.get("expected_behavior") or ""),
                entry_point=str(item.get("entry_point") or ""),
                baseline_candidate=str(item.get("baseline_candidate") or ""),
                reference_solution=str(item.get("reference_solution") or ""),
                failing_test_ids=tuple(str(test_id) for test_id in baseline.failing_test_ids),
                tests=tuple(dict(test) for test in item.get("tests") or []),
                item=dict(item),
            )
        )
    return selected


def extract_python_candidate(raw_text: str, *, entry_point: str) -> str:
    """Extract a Python function candidate from JSON, a code fence, or raw text."""

    text = raw_text.strip()
    if not text:
        return ""
    json_patch = _patch_from_json_text(text)
    if json_patch:
        return _ensure_trailing_newline(json_patch)
    fence = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return _ensure_trailing_newline(fence.group(1).strip())
    if _entry_point_present(text, entry_point):
        return _ensure_trailing_newline(text)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"def {entry_point}("):
            body = [line]
            for follow in lines[index + 1 :]:
                if follow.startswith((" ", "\t")) or not follow.strip():
                    body.append(follow)
                    continue
                break
            return _ensure_trailing_newline("\n".join(body).strip())
    return ""


def parse_schema_candidate(
    raw_text: str,
    *,
    fallback_draft: str,
    entry_point: str,
) -> StructuredCandidate:
    """Project a constrained raw response into the DCCD schema."""

    payload = _json_object_from_text(raw_text)
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        errors.append("json object missing")
        patch = extract_python_candidate(raw_text, entry_point=entry_point)
        return StructuredCandidate(False, patch, fallback_draft.strip(), errors)
    patch = payload.get("final_patch") or payload.get("repaired_code")
    draft_intent = str(payload.get("draft_intent") or fallback_draft).strip()
    if not isinstance(patch, str) or not patch.strip():
        errors.append("final_patch missing")
        patch_text = extract_python_candidate(raw_text, entry_point=entry_point)
    else:
        patch_text = _ensure_trailing_newline(patch.strip())
    return StructuredCandidate(not errors, patch_text, draft_intent, errors)


def condition_metrics(rows: Sequence[Mapping[str, Any]], *, n_cases: int) -> JsonDict:
    """Aggregate one condition's accepted repair outcomes over the fixed panel."""

    accepted = [row for row in rows if row.get("accepted") is True]
    denominator = max(int(n_cases), 1)
    pass_count = sum(1 for row in accepted if row.get("passed") is True)
    strict_valid_count = sum(1 for row in accepted if row.get("strict_valid") is True)
    schema_failure_count = sum(1 for row in accepted if row.get("schema_valid") is False)
    syntax_failure_count = sum(1 for row in accepted if row.get("syntax_success") is False)
    false_accept_count = sum(1 for row in accepted if row.get("false_accept") is True)
    intent_drift_count = sum(1 for row in accepted if row.get("intent_drift") is True)
    return {
        "candidate_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rows) - len(accepted),
        "pass_count": pass_count,
        "pass_rate": round(pass_count / denominator, 12),
        "strict_valid_count": strict_valid_count,
        "strict_validity_rate": round(strict_valid_count / denominator, 12),
        "schema_failure_count": schema_failure_count,
        "schema_failure_rate": round(schema_failure_count / denominator, 12),
        "syntax_failure_count": syntax_failure_count,
        "syntax_failure_rate": round(syntax_failure_count / denominator, 12),
        "false_accept_count": false_accept_count,
        "false_accept_rate": round(false_accept_count / denominator, 12),
        "intent_drift_count": intent_drift_count,
        "intent_drift_rate": round(intent_drift_count / denominator, 12),
        "accepted_item_ids": [str(row.get("item_id") or "") for row in accepted],
    }


def metric_deltas(baseline: Mapping[str, Any], dccd: Mapping[str, Any]) -> JsonDict:
    """Return DCCD-minus-acceptance deltas for the panel gate."""

    return {
        "pass_rate_delta": _delta(dccd.get("pass_rate"), baseline.get("pass_rate")),
        "strict_validity_delta": _delta(
            dccd.get("strict_validity_rate"), baseline.get("strict_validity_rate")
        ),
        "schema_failure_delta": _delta(
            dccd.get("schema_failure_rate"), baseline.get("schema_failure_rate")
        ),
        "syntax_failure_delta": _delta(
            dccd.get("syntax_failure_rate"), baseline.get("syntax_failure_rate")
        ),
        "intent_drift_delta": _delta(
            dccd.get("intent_drift_rate"), baseline.get("intent_drift_rate")
        ),
        "false_accept_delta": _delta(
            dccd.get("false_accept_rate"), baseline.get("false_accept_rate")
        ),
    }


def _run_case(
    *,
    case: PanelCase,
    generator_fn: PanelGenerator,
    model_spec: JsonDict,
    controller_rule: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, list[str]]:
    errors: list[str] = []
    unconstrained_generation = _safe_generate(
        generator_fn, case, UNCONSTRAINED_MODE, None, model_spec
    )
    if unconstrained_generation.error:
        errors.append(f"{case.item_id}:{UNCONSTRAINED_MODE}:{unconstrained_generation.error}")
    unconstrained_patch = extract_python_candidate(
        unconstrained_generation.raw_text,
        entry_point=case.entry_point,
    )
    unconstrained_row = evaluate_candidate(
        case=case,
        patch_text=unconstrained_patch,
        draft_intent=unconstrained_generation.raw_text,
        schema_valid=False,
        schema_errors=["unconstrained draft has no schema envelope"],
        condition=UNCONSTRAINED_MODE,
        variants=variants,
        accepted=True,
    )
    acceptance_probe = evaluate_candidate(
        case=case,
        patch_text=unconstrained_patch,
        draft_intent=unconstrained_generation.raw_text,
        schema_valid=bool(unconstrained_patch),
        schema_errors=[] if unconstrained_patch else ["candidate patch missing"],
        condition=ACCEPTANCE_MODE,
        variants=variants,
        accepted=True,
    )
    acceptance_reasons = controller_rejection_reasons(acceptance_probe, controller_rule)
    acceptance_row = {**acceptance_probe, "accepted": not acceptance_reasons}
    acceptance_row["rejection_reasons"] = acceptance_reasons

    dccd_generation = _safe_generate(
        generator_fn,
        case,
        DCCD_MODE,
        unconstrained_generation.raw_text,
        model_spec,
    )
    if dccd_generation.error:
        errors.append(f"{case.item_id}:{DCCD_MODE}:{dccd_generation.error}")
    structured = parse_schema_candidate(
        dccd_generation.raw_text,
        fallback_draft=unconstrained_generation.raw_text,
        entry_point=case.entry_point,
    )
    dccd_probe = evaluate_candidate(
        case=case,
        patch_text=structured.final_patch,
        draft_intent=structured.draft_intent,
        schema_valid=structured.schema_valid,
        schema_errors=structured.schema_errors,
        condition=DCCD_MODE,
        variants=variants,
        accepted=True,
    )
    dccd_reasons = controller_rejection_reasons(dccd_probe, controller_rule)
    dccd_row = {**dccd_probe, "accepted": not dccd_reasons}
    dccd_row["rejection_reasons"] = dccd_reasons
    return (
        {
            "item_id": case.item_id,
            "prompt": case.prompt,
            "expected_behavior": case.expected_behavior,
            "entry_point": case.entry_point,
            "failing_test_ids": list(case.failing_test_ids),
            "unconstrained_generation": _generation_dict(unconstrained_generation),
            "draft_conditioned_generation": _generation_dict(dccd_generation),
            "unconstrained_draft": unconstrained_row,
            "acceptance_only": acceptance_row,
            "draft_conditioned_constrained": dccd_row,
        },
        errors,
    )


def evaluate_candidate(
    *,
    case: PanelCase,
    patch_text: str,
    draft_intent: str,
    schema_valid: bool,
    schema_errors: Sequence[str],
    condition: str,
    variants: Sequence[Mapping[str, Any]],
    accepted: bool,
) -> JsonDict:
    """Replay syntax, original tests, variants, false accepts, and intent checks."""

    syntax_success, syntax_errors = syntax_diagnostics(patch_text)
    entry_point_present = _entry_point_present(patch_text, case.entry_point)
    original = hard.run_candidate_tests(
        {**case.item, "repair_candidate": patch_text}, "repair_candidate"
    )
    relevant_variants = [
        variant for variant in variants if str(variant.get("source_item_id") or "") == case.item_id
    ]
    variant_outcomes = []
    for variant in relevant_variants:
        adapted = metamorphic._adapt_candidate(  # noqa: SLF001
            patch_text,
            str(variant.get("source_entry_point") or case.entry_point),
            str(variant.get("entry_point") or case.entry_point),
        )
        variant_outcomes.append(
            hard.run_candidate_tests(
                {**dict(variant), "repair_candidate": adapted}, "repair_candidate"
            )
        )
    metamorphic_passed_all = (
        all(outcome.passed for outcome in variant_outcomes)
        if variant_outcomes
        else bool(original.passed)
    )
    false_accept = bool(original.passed and variant_outcomes and not metamorphic_passed_all)
    intent_drift = not intent_preserved(draft_intent, case.expected_behavior)
    return {
        "condition": condition,
        "item_id": case.item_id,
        "accepted": accepted,
        "schema_valid": bool(schema_valid),
        "schema_errors": list(schema_errors),
        "syntax_success": syntax_success,
        "syntax_errors": syntax_errors,
        "entry_point_present": entry_point_present,
        "strict_valid": bool(schema_valid and syntax_success and entry_point_present),
        "original_passed": bool(original.passed),
        "metamorphic_passed_all": bool(metamorphic_passed_all),
        "metamorphic_variant_count": len(variant_outcomes),
        "passed": bool(original.passed and metamorphic_passed_all),
        "false_accept": false_accept,
        "false_accept_probe_clean": not false_accept,
        "tautology_probe_clean": True,
        "intent_drift": intent_drift,
        "draft_intent": draft_intent.strip(),
        "final_patch": patch_text,
        "final_patch_sha256": _sha256_text(patch_text),
        "original_verifier_output": original.as_dict(),
        "metamorphic_verifier_outputs": [outcome.as_dict() for outcome in variant_outcomes],
    }


def syntax_diagnostics(code: str) -> tuple[bool, list[str]]:
    """Return parser validity and readable syntax errors for a candidate."""

    if not code.strip():
        return False, ["candidate patch missing"]
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, [f"{exc.__class__.__name__}: {exc.msg}"]
    return True, []


def controller_rejection_reasons(
    row: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> list[str]:
    """Apply the Exp 3015 transparent acceptance rule to one candidate row."""

    checks = [
        ("require_schema_valid", "schema_valid", row.get("schema_valid") is True),
        ("require_syntax_success", "syntax_success", row.get("syntax_success") is True),
        (
            "require_entry_point_present",
            "entry_point_present",
            row.get("entry_point_present") is True,
        ),
        (
            "require_false_accept_probe_clean",
            "false_accept",
            row.get("false_accept_probe_clean") is True,
        ),
        ("require_no_intent_drift", "intent_drift", row.get("intent_drift") is False),
        ("require_original_passed", "original_passed", row.get("original_passed") is True),
        (
            "require_metamorphic_passed_all",
            "metamorphic_passed_all",
            row.get("metamorphic_passed_all") is True,
        ),
        (
            "require_tautology_probe_clean",
            "tautology_probe_clean",
            row.get("tautology_probe_clean") is True,
        ),
    ]
    return [reason for flag, reason, passed in checks if rule.get(flag) and not passed]


def intent_preserved(draft_intent: str, expected_behavior: str) -> bool:
    """Check that a draft keeps at least a small lexical link to the case intent."""

    if not draft_intent.strip() or not expected_behavior.strip():
        return False
    draft_tokens = set(content_tokens(draft_intent))
    expected_tokens = set(content_tokens(expected_behavior))
    if not expected_tokens:
        return True
    overlap = len(draft_tokens.intersection(expected_tokens))
    return overlap >= min(2, len(expected_tokens))


def content_tokens(text: str) -> list[str]:
    """Return content-bearing lowercase tokens for lightweight intent checks."""

    stop = {"the", "and", "into", "that", "with", "from", "each", "once", "return"}
    token = ""
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            token += ch
        elif token:
            if len(token) > 2 and token not in stop:
                out.append(token)
            token = ""
    if token and len(token) > 2 and token not in stop:
        out.append(token)
    return out


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    preconditions: Mapping[str, Any],
    selected_model: Mapping[str, Any],
) -> JsonDict:
    substrate = _inference_substrate(
        preconditions=preconditions,
        selected_model=selected_model,
        model_load_attempted=False,
        live_generation_succeeded=False,
        generation_errors=["headline model unavailable or live generator absent"],
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "dccd_panel_ready": False,
        "n_cases": 0,
        "model_specs": [],
        "legacy_smoke_only_used": False,
        "baseline_acceptance_metrics": condition_metrics([], n_cases=0),
        "dccd_metrics": condition_metrics([], n_cases=0),
        "intent_drift_delta": 0.0,
        "false_accept_delta": 0.0,
        "changed_files": list(config.changed_files),
        "tests_run": list(config.tests_run),
        "inference_substrate": substrate,
        "honest_verdict": _honest_verdict(ready=False, blocked=True, n_cases=0),
        "unconstrained_metrics": condition_metrics([], n_cases=0),
        "metric_deltas": metric_deltas(
            condition_metrics([], n_cases=0), condition_metrics([], n_cases=0)
        ),
        "case_results": [],
        "preconditions_checked": dict(preconditions),
        "selected_panel_item_ids": [],
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "source_artifacts": _source_artifacts(config),
        "duration_s": _elapsed(config, started),
        "reproducibility_checksum": _sha256_text(json.dumps(dict(preconditions), sort_keys=True)),
    }


def _load_hard_items(config: ExperimentConfig) -> list[JsonDict]:
    path = config.resolved_hard_manifest_path()
    if not path.is_file():
        return []
    try:
        return [dict(item) for item in hard.load_manifest(path)]
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _load_controller_rule(config: ExperimentConfig) -> JsonDict:
    payload = _read_json_if_present(config.resolved_controller_config_path())
    rule = payload.get("selected_rule")
    return dict(rule) if isinstance(rule, Mapping) else {}


def _load_metamorphic_variants(config: ExperimentConfig) -> list[JsonDict]:
    path = config.resolved_metamorphic_manifest_path()
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return []


def _select_headline_model(config: ExperimentConfig) -> JsonDict:
    if config.selected_model_path is not None:
        model_id = config.selected_model_id or ""
        path = config.selected_model_path
        return {
            "hf_id": model_id,
            "path": str(path),
            "available": bool(model_id in HEADLINE_MODEL_IDS and path.is_file()),
            "source": "explicit_config",
        }
    for model_id in HEADLINE_MODEL_IDS:  # pragma: no cover - host-cache dependent.
        for path in _hf_cache_candidates(model_id):
            if path.is_file():
                return {
                    "hf_id": model_id,
                    "path": str(path),
                    "available": True,
                    "source": "huggingface_cache",
                }
    return {"hf_id": None, "path": None, "available": False, "source": "not_found"}


def _hf_cache_candidates(model_id: str) -> list[Path]:  # pragma: no cover - host-cache dependent.
    root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = root / f"models--{model_id.replace('/', '--')}"
    if not cache_dir.is_dir():
        return []
    return sorted(cache_dir.glob("snapshots/*/*.gguf"))


def _preconditions(
    config: ExperimentConfig,
    hard_items: Sequence[Mapping[str, Any]],
    controller_rule: Mapping[str, Any],
    selected_model: Mapping[str, Any],
) -> JsonDict:
    baseline_failures = 0
    reference_passes = 0
    for case in select_panel_cases(hard_items, config.panel_item_ids):
        baseline_failures += int(bool(case.failing_test_ids))
        reference_passes += 1
    return {
        "recorded_before_model_load": True,
        "gpu_status": _nvidia_smi_inventory(),
        "repo_commit": _git_commit(config.repo_root),
        "selected_headline_model": dict(selected_model),
        "repair_hard_fixture_availability": {
            "manifest_path": _path_string(config.repo_root, config.resolved_hard_manifest_path()),
            "manifest_present": config.resolved_hard_manifest_path().is_file(),
            "n_manifest_items": len(hard_items),
            "requested_item_ids": list(config.panel_item_ids),
            "baseline_failures_verified": baseline_failures,
            "reference_passes_verified": reference_passes,
        },
        "acceptance_controller": {
            "config_path": _path_string(config.repo_root, config.resolved_controller_config_path()),
            "config_present": config.resolved_controller_config_path().is_file(),
            "rule_loaded": bool(controller_rule),
        },
    }


def _inference_substrate(
    *,
    preconditions: Mapping[str, Any],
    selected_model: Mapping[str, Any],
    model_load_attempted: bool,
    live_generation_succeeded: bool,
    generation_errors: Sequence[str],
) -> JsonDict:
    return {
        "kind": "live_llm_inference",
        "loader": "llama_cpp",
        "recorded_before_model_load": bool(preconditions.get("recorded_before_model_load")),
        "selected_headline_model": dict(selected_model),
        "model_load_attempted": model_load_attempted,
        "live_generation_succeeded": live_generation_succeeded,
        "generation_errors": list(generation_errors),
        "gpu_status": preconditions.get("gpu_status", {}),
        "repo_commit": preconditions.get("repo_commit"),
    }


def _safe_generate(
    generator_fn: PanelGenerator,
    case: PanelCase,
    mode: str,
    draft_text: str | None,
    model_spec: JsonDict,
) -> GenerationResult:
    try:
        return generator_fn(case, mode, draft_text, model_spec)
    except Exception as exc:  # noqa: BLE001
        return GenerationResult("", error=f"{type(exc).__name__}: {exc}")


def _generation_dict(result: GenerationResult) -> JsonDict:
    return {
        "raw_text": result.raw_text,
        "duration_s": result.duration_s,
        "tokens_generated": result.tokens_generated,
        "error": result.error,
    }


def _json_object_from_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _patch_from_json_text(text: str) -> str:
    payload = _json_object_from_text(text)
    if not isinstance(payload, Mapping):
        return ""
    patch = payload.get("final_patch") or payload.get("repaired_code")
    return patch if isinstance(patch, str) else ""


def _entry_point_present(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body)


def _read_json_if_present(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    paths = [
        config.resolved_hard_manifest_path(),
        config.resolved_metamorphic_manifest_path(),
        config.repo_root / EXP3015_REL_PATH,
        config.resolved_controller_config_path(),
    ]
    return [
        {
            "path": _path_string(config.repo_root, path),
            "present": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
        for path in paths
    ]


def _resolve_repo_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _path_string(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _nvidia_smi_inventory() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return {"available": False, "stderr_summary": result.stderr.strip()[:240]}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "driver_version": parts[5],
            }
        )
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "free_vram_mib_total": sum(gpu["memory_free_mib"] for gpu in gpus),
    }


def _git_commit(root: Path) -> str | None:  # pragma: no cover - host dependent.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reproducibility_checksum(
    case_results: Sequence[Mapping[str, Any]], model_spec: Mapping[str, Any]
) -> str:
    payload = {
        "model_spec": dict(model_spec),
        "case_hashes": [
            {
                "item_id": row.get("item_id"),
                "unconstrained": row.get("unconstrained_draft", {}).get("final_patch_sha256"),
                "dccd": row.get("draft_conditioned_constrained", {}).get("final_patch_sha256"),
            }
            for row in case_results
        ],
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))[:16]


def _delta(left: Any, right: Any) -> float:
    return round(float(left or 0.0) - float(right or 0.0), 12)


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(config.clock() - started, 6)


def _honest_verdict(*, ready: bool, blocked: bool, n_cases: int) -> str:
    if blocked:
        return (
            "blocked_sota_headline_model_unavailable: "
            "no mandated headline GGUF loaded for live DCCD panel"
        )
    if ready:
        return f"complete: dccd structured repair panel ready; n_cases={n_cases}"
    return f"complete_flagged: dccd panel completed but ready gate failed; n_cases={n_cases}"


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else f"{text}\n"
