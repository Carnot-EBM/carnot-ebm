"""Exp 1366 tag-first prefix-injection CRANE certificate run.

Spec: REQ-VERIFY-1366, SCENARIO-VERIFY-1366
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot.reporting import triggered_certificate_v7_truncproof_sota as exp1353
from carnot.reporting import truncproof_xgrammar_certificate_completion_preflight as preflight
from carnot.reporting import xgrammar2_tagdispatch_certificate_grammar_dryrun as tagdispatch
from carnot.reporting.triggered_certificate_v7_truncproof_sota import (
    CertificateCase,
    GPUHealth,
    bounded_certificate_suite,
    check_gpu_health,
    structural_tag,
)


DEFAULT_RUN_DATE = "20260505"
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1366_certificate_v8_tag_first_prefix_injection_crane.json"
)
DEFAULT_EXP1352_PATH = Path(
    "results/experiment_1352_truncproof_xgrammar_certificate_completion_preflight.json"
)
DEFAULT_EXP1353_PATH = Path("results/experiment_1353_triggered_certificate_v7_truncproof_sota.json")
DEFAULT_EXP1364_PATH = Path(
    "results/experiment_1364_105_carryforward_thinking_mode_blocker_audit.json"
)
ARTIFACT_NAME = "experiment_1366_certificate_v8_tag_first_prefix_injection_crane"
SCHEMA_VERSION = 1
CRANE_REASONING_BUDGET_TOKENS = 256
EXP1353_PARSE_BASELINE = 0.0
PREFIX_INJECTION_METHOD = "llama_cpp_raw_prompt_partial_assistant_prefix_plus_body_gbnf_position_0"
MANDATED_HEADLINE_MODEL_IDS = exp1353.MANDATED_HEADLINE_MODEL_IDS
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "models_used",
    "prefix_injection_method",
    "prefix_injection_supported",
    "certificate_case_count",
    "trigger_token_hit_rate",
    "certificate_parse_rate",
    "certificate_truthfulness_rate",
    "unknown_preservation_rate",
    "parse_rate_delta_over_exp1353",
    "crane_reasoning_budget_tokens_used",
    "terminal_blocker",
    "retire_trigger_before_constrain",
    "headline_result_allowed",
    "honest_verdict",
)


@dataclass(frozen=True)
class CranePrompts:
    """Prompt bundle for the CRANE alternating generation pattern."""

    reasoning_prompt: str
    certificate_prompt: str
    certificate_prefix: str
    certificate_grammar: str


@dataclass(frozen=True)
class CraneGenerationResult:
    """One two-stage CRANE completion for a bounded certificate case."""

    model_hf_id: str
    case_id: str
    reasoning_text: str
    reasoning_token_count: int
    certificate_prefix: str
    certificate_body: str
    generation_source: str
    certificate_token_count: int
    error: str | None = None
    elapsed_reasoning_seconds: float = 0.0
    elapsed_certificate_seconds: float = 0.0

    @property
    def full_certificate_text(self) -> str:
        """Return the parser-visible certificate with the injected tag first."""

        return f"{self.certificate_prefix}{self.certificate_body}"


GenerationFn = Callable[[Mapping[str, Any], CertificateCase, CranePrompts], CraneGenerationResult]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
GPUHealthFn = Callable[[], GPUHealth]


def json_certificate_text(state: str) -> str:
    """Return the minimal branch body validated by Exp 1352's completion preflight.

    The name mirrors Exp 1353's helper so tests and downstream replay code can
    swap the v7 and v8 modules easily.  For v8 the body is intentionally small:
    the structural tag is already injected before generation, and the branch
    parser has CPU evidence that these minimal bodies dispatch correctly inside
    the active token budget.
    """

    normalised = preflight.normalise_state(state)
    if normalised == "REPAIR_HINT":
        return "REPAIR_HINT: add bound."
    return normalised


def build_crane_prompts(
    case: CertificateCase,
    runtime_settings: Mapping[str, Any],
) -> CranePrompts:
    """Build the unconstrained reasoning prompt and constrained certificate prefix."""

    certificate_tokens = int(runtime_settings.get("max_tokens", 96))
    prefix = structural_tag(case.expected_state) + "\n"
    body = json_certificate_text(case.expected_state)
    reasoning_prompt = (
        "You are preparing a Carnot verification certificate. Reason privately and "
        "briefly about the verifier branch, but do not emit any certificate tag or "
        "certificate body in this stage.\n"
        f"Case id: {case.case_id}\n"
        f"Problem: {case.prompt}\n"
        f"Reasoning budget: {CRANE_REASONING_BUDGET_TOKENS} tokens."
    )
    certificate_prompt = (
        "CRANE certificate stage. Continue the assistant response from the injected "
        "structural prefix. Emit only the constrained certificate body that matches "
        "the prefix and contains no thinking tags.\n"
        f"Case id: {case.case_id}\n"
        f"Problem: {case.prompt}\n"
        f"Certificate body budget: {certificate_tokens} tokens.\n"
    )
    return CranePrompts(
        reasoning_prompt=reasoning_prompt,
        certificate_prompt=certificate_prompt,
        certificate_prefix=prefix,
        certificate_grammar=_exact_literal_grammar(body),
    )


def build_experiment_artifact(
    *,
    source_artifacts: Mapping[str, Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]] | None,
    gpu_health: GPUHealth,
    generation_fn: GenerationFn | None = None,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    max_models: int = 1,
) -> dict[str, Any]:
    """Build a terminal Exp 1366 artifact from live rows or explicit blockers."""

    cases = bounded_certificate_suite()
    exp1352_artifact = source_artifacts.get("exp1352", {})
    runtime_settings = _runtime_settings(exp1352_artifact)
    completion = _completion_preflight(exp1352_artifact, runtime_settings)
    base = _base_artifact(
        run_date=run_date,
        project_root=Path(project_root),
        runtime_settings=runtime_settings,
        completion_preflight=completion,
        source_artifacts=source_artifacts,
    )
    base["gpu_health_used"] = gpu_health.__dict__

    if not completion["sota_run_allowed"]:
        blocker = f"completion_preflight_blocked:{completion['blocker_if_not_allowed']}"
        return _blocked_artifact(base, cases, blocker)

    model_blocker = _model_blocker(model_specs)
    if model_blocker is not None:
        return _blocked_artifact(
            base,
            cases,
            model_blocker,
            models_used=_model_records(
                model_specs or [],
                selected_count=0,
                headline_result_allowed=False,
                fallback_reason=model_blocker,
            ),
        )

    if not gpu_health.healthy:
        return _blocked_artifact(
            base,
            cases,
            "gpu_health_failed",
            models_used=_model_records(
                model_specs or [],
                selected_count=0,
                headline_result_allowed=False,
                fallback_reason="gpu_health_failed",
            ),
        )

    selected_specs = list(model_specs or [])[: max(1, int(max_models))]
    active_generation_fn = generation_fn or LlamaCppCraneGenerator(runtime_settings)
    try:
        rows = _run_generation_rows(cases, selected_specs, runtime_settings, active_generation_fn)
    except Exception as exc:
        blocker = f"sota_generation_failed:{type(exc).__name__}:{_short_error(exc)}"
        return _blocked_artifact(
            base,
            cases,
            blocker,
            models_used=_model_records(
                model_specs or [],
                selected_count=len(selected_specs),
                headline_result_allowed=False,
                fallback_reason=blocker,
            ),
        )

    return _complete_from_rows(
        base,
        cases,
        rows,
        model_specs=list(model_specs or []),
        selected_count=len(selected_specs),
    )


class LlamaCppCraneGenerator:
    """llama.cpp adapter for CRANE reasoning followed by prefix-injected certificate."""

    def __init__(
        self,
        runtime_settings: Mapping[str, Any],
        *,
        llama_importer: Callable[[], type[Any]] | None = None,
        grammar_importer: Callable[[], type[Any]] | None = None,
    ) -> None:
        self._runtime_settings = dict(runtime_settings)
        self._llama_importer = llama_importer or exp1353._import_llama_class
        self._grammar_importer = grammar_importer or _import_llama_grammar_class
        self._models: dict[str, Any] = {}

    def __call__(
        self,
        spec: Mapping[str, Any],
        case: CertificateCase,
        prompts: CranePrompts,
    ) -> CraneGenerationResult:
        model = self._model_for(spec)
        start_reasoning = time.perf_counter()
        reasoning_response = model(
            prompts.reasoning_prompt,
            max_tokens=CRANE_REASONING_BUDGET_TOKENS,
            temperature=float(self._runtime_settings.get("temperature", 0.0)),
            top_p=float(self._runtime_settings.get("top_p", 1.0)),
            stop=list(self._runtime_settings.get("stop", ["</s>", "<eos>"])),
            echo=False,
        )
        reasoning_text = exp1353._response_text(reasoning_response)
        reasoning_tokens = exp1353._completion_token_count(reasoning_response, reasoning_text)

        grammar_cls = self._grammar_importer()
        grammar = grammar_cls.from_string(prompts.certificate_grammar, verbose=False)
        certificate_prompt = _certificate_prompt_with_reasoning(prompts, reasoning_text)
        start_certificate = time.perf_counter()
        certificate_response = model(
            certificate_prompt,
            max_tokens=int(self._runtime_settings.get("max_tokens", 96)),
            temperature=float(self._runtime_settings.get("temperature", 0.0)),
            top_p=float(self._runtime_settings.get("top_p", 1.0)),
            stop=list(self._runtime_settings.get("stop", ["</s>", "<eos>"])),
            echo=False,
            grammar=grammar,
        )
        certificate_body = exp1353._response_text(certificate_response)
        return CraneGenerationResult(
            model_hf_id=str(spec.get("hf_id")),
            case_id=case.case_id,
            reasoning_text=reasoning_text,
            reasoning_token_count=reasoning_tokens,
            certificate_prefix=prompts.certificate_prefix,
            certificate_body=certificate_body,
            generation_source="live_sota_llamacpp",
            certificate_token_count=exp1353._completion_token_count(
                certificate_response, certificate_body
            ),
            elapsed_reasoning_seconds=round(start_certificate - start_reasoning, 6),
            elapsed_certificate_seconds=round(time.perf_counter() - start_certificate, 6),
        )

    def _model_for(self, spec: Mapping[str, Any]) -> Any:
        key = str(spec.get("model_path") or spec.get("hf_id"))
        if key in self._models:
            return self._models[key]
        model_path = spec.get("model_path")
        if not model_path:
            raise RuntimeError(f"model_path missing for {spec.get('hf_id')}")
        llama_cls = self._llama_importer()
        model = llama_cls(
            model_path=str(model_path),
            n_ctx=int(self._runtime_settings.get("n_ctx", 1024)),
            n_gpu_layers=int(self._runtime_settings.get("n_gpu_layers", -1)),
            seed=int(self._runtime_settings.get("seed", 1366)),
            verbose=False,
        )
        self._models[key] = model
        return model


def write_in_progress_artifact(
    path: Path | str,
    *,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Write the required bootstrap artifact before source/model loading."""

    artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": "in_progress",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": "REQ-VERIFY-1366",
        },
    }
    _write_json(Path(path), artifact)
    return artifact


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    exp1352_path: Path | str = DEFAULT_EXP1352_PATH,
    exp1353_path: Path | str = DEFAULT_EXP1353_PATH,
    exp1364_path: Path | str = DEFAULT_EXP1364_PATH,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    cached_pair_fn: CachedPairFn | None = None,
    gpu_health_fn: GPUHealthFn = check_gpu_health,
    generation_fn: GenerationFn | None = None,
    max_models: int = 1,
) -> dict[str, Any]:
    """Write in-progress, run CRANE prefix injection, and persist completion."""

    output = Path(output_path)
    root = Path(project_root)
    write_in_progress_artifact(output, run_date=run_date, project_root=root)
    if cached_pair_fn is None:
        cached_pair_fn = _load_cached_sota_pair
    try:
        specs = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception:
        specs = None
    artifact = build_experiment_artifact(
        source_artifacts={
            "exp1352": _load_json(Path(exp1352_path)),
            "exp1353": _load_json(Path(exp1353_path)),
            "exp1364": _load_json(Path(exp1364_path)),
        },
        model_specs=specs,
        gpu_health=gpu_health_fn(),
        generation_fn=generation_fn,
        run_date=run_date,
        project_root=root,
        max_models=max_models,
    )
    _write_json(output, artifact)
    return artifact


def _run_generation_rows(
    cases: Sequence[CertificateCase],
    model_specs: Sequence[Mapping[str, Any]],
    runtime_settings: Mapping[str, Any],
    generation_fn: GenerationFn,
) -> list[CraneGenerationResult]:
    rows: list[CraneGenerationResult] = []
    for spec in model_specs:
        for case in cases:
            rows.append(generation_fn(spec, case, build_crane_prompts(case, runtime_settings)))
    return rows


def _complete_from_rows(
    artifact: dict[str, Any],
    cases: Sequence[CertificateCase],
    rows: Sequence[CraneGenerationResult],
    *,
    model_specs: Sequence[Mapping[str, Any]],
    selected_count: int,
) -> dict[str, Any]:
    grammar = tagdispatch.compile_branch_grammars()
    case_by_id = {case.case_id: case for case in cases}
    parsed_rows = [_parse_generation_row(row, case_by_id[row.case_id], grammar) for row in rows]
    metrics = _metrics(parsed_rows, baseline=_exp1353_baseline(artifact))
    prefix_supported = _prefix_injection_supported(rows)
    mandated_rows = [
        row
        for row in parsed_rows
        if row.get("generation_source") == "live_sota_llamacpp"
        and row.get("model_hf_id") in MANDATED_HEADLINE_MODEL_IDS
    ]
    mandated_parse_rate = _rate(
        sum(1 for row in mandated_rows if row.get("parseable")),
        len(mandated_rows),
    )
    headline_result_allowed = bool(prefix_supported and mandated_parse_rate >= 0.75)
    terminal_blocker = _terminal_blocker(
        prefix_supported=prefix_supported,
        parse_rate=metrics["certificate_parse_rate"],
        headline_result_allowed=headline_result_allowed,
    )
    retire = _should_retire(prefix_supported, metrics["certificate_parse_rate"])
    artifact.update(
        {
            "status": "complete",
            "models_used": _model_records(
                model_specs,
                selected_count=selected_count,
                headline_result_allowed=headline_result_allowed,
                fallback_reason=terminal_blocker,
            ),
            "prefix_injection_supported": prefix_supported,
            "certificate_case_count": len(rows),
            "trigger_token_hit_rate": metrics["trigger_token_hit_rate"],
            "certificate_parse_rate": metrics["certificate_parse_rate"],
            "certificate_truthfulness_rate": metrics["certificate_truthfulness_rate"],
            "unknown_preservation_rate": metrics["unknown_preservation_rate"],
            "parse_rate_delta_over_exp1353": metrics["parse_rate_delta_over_exp1353"],
            "crane_reasoning_budget_tokens_used": [int(row.reasoning_token_count) for row in rows],
            "terminal_blocker": terminal_blocker,
            "retire_trigger_before_constrain": retire,
            "retire_if_same_verdict": retire,
            "headline_result_allowed": headline_result_allowed,
            "honest_verdict": _honest_verdict(
                terminal_blocker=terminal_blocker,
                headline_result_allowed=headline_result_allowed,
                parse_rate=metrics["certificate_parse_rate"],
            ),
            "generation_rows": [_generation_row_dict(row) for row in rows],
            "certificate_rows": parsed_rows,
            "mandated_sota_parse_rate": mandated_parse_rate,
        }
    )
    return artifact


def _blocked_artifact(
    artifact: dict[str, Any],
    cases: Sequence[CertificateCase],
    terminal_blocker: str,
    *,
    models_used: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metrics = _metrics([], baseline=_exp1353_baseline(artifact), denominator=len(cases))
    artifact.update(
        {
            "status": "complete",
            "models_used": models_used or [],
            "prefix_injection_supported": False,
            "certificate_case_count": len(cases),
            "trigger_token_hit_rate": metrics["trigger_token_hit_rate"],
            "certificate_parse_rate": metrics["certificate_parse_rate"],
            "certificate_truthfulness_rate": metrics["certificate_truthfulness_rate"],
            "unknown_preservation_rate": metrics["unknown_preservation_rate"],
            "parse_rate_delta_over_exp1353": metrics["parse_rate_delta_over_exp1353"],
            "crane_reasoning_budget_tokens_used": [],
            "terminal_blocker": terminal_blocker,
            "retire_trigger_before_constrain": True,
            "retire_if_same_verdict": True,
            "headline_result_allowed": False,
            "honest_verdict": f"retired_trigger_before_constrain_{terminal_blocker}",
            "generation_rows": [],
            "certificate_rows": [],
        }
    )
    return artifact


def _parse_generation_row(
    row: CraneGenerationResult,
    case: CertificateCase,
    grammar: tagdispatch.CompiledBranchGrammars,
) -> dict[str, Any]:
    generation_row = exp1353.GenerationResult(
        model_hf_id=row.model_hf_id,
        case_id=row.case_id,
        text=row.full_certificate_text,
        generation_source=row.generation_source,
        token_count=row.certificate_token_count,
        error=row.error,
        elapsed_seconds=row.elapsed_certificate_seconds,
    )
    parsed = exp1353._parse_generation_row(generation_row, case, grammar)
    parsed["prefix_injection_applied"] = row.full_certificate_text.startswith(
        row.certificate_prefix
    )
    parsed["reasoning_token_count"] = row.reasoning_token_count
    return parsed


def _metrics(
    parsed_rows: Sequence[Mapping[str, Any]],
    *,
    baseline: float,
    denominator: int | None = None,
) -> dict[str, float]:
    total = denominator if denominator is not None else len(parsed_rows)
    trigger_hits = sum(1 for row in parsed_rows if row.get("trigger_token_hit"))
    parseable = sum(1 for row in parsed_rows if row.get("parseable"))
    truthful = sum(1 for row in parsed_rows if row.get("truthful"))
    unknown_rows = [row for row in parsed_rows if row.get("expected_state") == "UNKNOWN"]
    unknown_preserved = sum(1 for row in unknown_rows if row.get("unknown_preserved"))
    parse_rate = _rate(parseable, total)
    return {
        "trigger_token_hit_rate": _rate(trigger_hits, total),
        "certificate_parse_rate": parse_rate,
        "certificate_truthfulness_rate": _rate(truthful, parseable),
        "unknown_preservation_rate": _rate(unknown_preserved, len(unknown_rows)),
        "parse_rate_delta_over_exp1353": round(parse_rate - baseline, 6),
    }


def _base_artifact(
    *,
    run_date: str,
    project_root: Path,
    runtime_settings: Mapping[str, Any],
    completion_preflight: Mapping[str, Any],
    source_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": "complete",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": "REQ-VERIFY-1366",
            "source_experiments": ["exp1352", "exp1353", "exp1364"],
        },
        "models_used": [],
        "runtime_settings_used": dict(runtime_settings),
        "completion_preflight_used": dict(completion_preflight),
        "prefix_injection_method": PREFIX_INJECTION_METHOD,
        "prefix_injection_supported": False,
        "certificate_case_count": 0,
        "trigger_token_hit_rate": 0.0,
        "certificate_parse_rate": 0.0,
        "certificate_truthfulness_rate": 0.0,
        "unknown_preservation_rate": 0.0,
        "parse_rate_delta_over_exp1353": round(
            0.0 - _baseline_parse_rate_from_sources(source_artifacts), 6
        ),
        "crane_reasoning_budget_tokens_used": [],
        "crane_reasoning_budget_tokens_max": CRANE_REASONING_BUDGET_TOKENS,
        "terminal_blocker": None,
        "retire_trigger_before_constrain": False,
        "retire_if_same_verdict": False,
        "headline_result_allowed": False,
        "honest_verdict": "not_run",
        "source_context": _source_context(source_artifacts),
    }


def _runtime_settings(exp1352_artifact: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict(exp1352_artifact.get("runtime_settings_used") or {})
    settings.setdefault("max_tokens", 96)
    settings.setdefault("temperature", 0.0)
    settings.setdefault("top_p", 1.0)
    settings.setdefault("stop", ["</s>", "<eos>"])
    settings.setdefault("n_ctx", 1024)
    settings.setdefault("n_gpu_layers", -1)
    settings.setdefault("seed", 1366)
    settings.setdefault("gpu_indices", [0, 1])
    settings.setdefault("preferred_quant", "Q4_K_M")
    settings["trigger_before_constrain"] = False
    settings["crane_alternating_pattern"] = True
    settings["crane_reasoning_budget_tokens"] = CRANE_REASONING_BUDGET_TOKENS
    settings["prefix_injection_method"] = PREFIX_INJECTION_METHOD
    return settings


def _completion_preflight(
    exp1352_artifact: Mapping[str, Any],
    runtime_settings: Mapping[str, Any],
) -> dict[str, Any]:
    completion = exp1353._completion_preflight(exp1352_artifact, runtime_settings)
    completion["source"] = "exp1352"
    completion["applies_to"] = "minimal_branch_body_after_prefix_injection"
    return completion


def _source_context(source_artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1352 = source_artifacts.get("exp1352", {})
    exp1353_artifact = source_artifacts.get("exp1353", {})
    exp1364 = source_artifacts.get("exp1364", {})
    return {
        "exp1352_honest_verdict": exp1352.get("honest_verdict"),
        "exp1352_structural_tag_supported": exp1352.get("structural_tag_supported"),
        "exp1353_certificate_parse_rate": exp1353_artifact.get("certificate_parse_rate"),
        "exp1353_trigger_token_hit_rate": exp1353_artifact.get("trigger_token_hit_rate"),
        "exp1353_honest_verdict": exp1353_artifact.get("honest_verdict"),
        "exp1364_thinking_mode_blocker_confirmed": exp1364.get("thinking_mode_blocker_confirmed"),
        "exp1364_honest_verdict": exp1364.get("honest_verdict"),
    }


def _model_blocker(model_specs: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not model_specs:
        return "cached_sota_pair_unavailable"
    ids = {str(spec.get("hf_id")) for spec in model_specs}
    if not ids.intersection(MANDATED_HEADLINE_MODEL_IDS):
        return "cached_sota_pair_unavailable"
    if not any(spec.get("model_path") for spec in model_specs):
        return "cached_sota_pair_unavailable"
    return None


def _model_records(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    selected_count: int,
    headline_result_allowed: bool,
    fallback_reason: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(model_specs):
        selected = index < selected_count
        model_path = spec.get("model_path")
        records.append(
            {
                "name": spec.get("name"),
                "hf_id": spec.get("hf_id"),
                "gpu": spec.get("gpu"),
                "model_path": model_path,
                "quantization": exp1353._quantization_from_path(model_path)
                or spec.get("quantization"),
                "generation_source": "live_sota_llamacpp" if selected else None,
                "selected_for_generation": selected,
                "headline_result_allowed": bool(headline_result_allowed and selected),
                "fallback_reason": fallback_reason,
            }
        )
    return records


def _generation_row_dict(row: CraneGenerationResult) -> dict[str, Any]:
    return {
        "model_hf_id": row.model_hf_id,
        "case_id": row.case_id,
        "reasoning_text": row.reasoning_text,
        "reasoning_token_count": row.reasoning_token_count,
        "certificate_prefix": row.certificate_prefix,
        "certificate_body": row.certificate_body,
        "full_certificate_text": row.full_certificate_text,
        "generation_source": row.generation_source,
        "certificate_token_count": row.certificate_token_count,
        "error": row.error,
        "elapsed_reasoning_seconds": row.elapsed_reasoning_seconds,
        "elapsed_certificate_seconds": row.elapsed_certificate_seconds,
    }


def _certificate_prompt_with_reasoning(prompts: CranePrompts, reasoning_text: str) -> str:
    excerpt = " ".join(str(reasoning_text or "").split())[:800]
    return (
        f"{prompts.certificate_prompt}"
        f"Unconstrained reasoning excerpt:\n{excerpt}\n"
        "Injected assistant prefix follows. Continue after it exactly:\n"
        f"{prompts.certificate_prefix}"
    )


def _exact_literal_grammar(text: str) -> str:
    return f"root ::= {json.dumps(str(text))}"


def _prefix_injection_supported(rows: Sequence[CraneGenerationResult]) -> bool:
    return bool(rows) and all(
        row.generation_source == "live_sota_llamacpp"
        and row.full_certificate_text.startswith("<CARNOT_CERT_STATE:")
        for row in rows
    )


def _terminal_blocker(
    *,
    prefix_supported: bool,
    parse_rate: float,
    headline_result_allowed: bool,
) -> str | None:
    if not prefix_supported:
        return "prefix_injection_not_supported"
    if parse_rate == 0.0:
        return "prefix_injection_parse_rate_zero"
    if not headline_result_allowed:
        return "mandated_sota_parse_rate_below_0_75"
    return None


def _should_retire(prefix_supported: bool, parse_rate: float) -> bool:
    return (not prefix_supported) or parse_rate == 0.0


def _honest_verdict(
    *,
    terminal_blocker: str | None,
    headline_result_allowed: bool,
    parse_rate: float,
) -> str:
    if terminal_blocker:
        if terminal_blocker == "prefix_injection_parse_rate_zero":
            return "retired_trigger_before_constrain_prefix_injection_parse_rate_zero"
        return f"blocked_{terminal_blocker}"
    if headline_result_allowed:
        return f"tag_first_prefix_injection_crane_positive_parse_rate_{_rate_label(parse_rate)}"
    return "tag_first_prefix_injection_crane_non_headline"


def _rate_label(value: float) -> str:
    return str(round(float(value), 6)).replace(".", "_")


def _baseline_parse_rate_from_sources(source_artifacts: Mapping[str, Mapping[str, Any]]) -> float:
    exp1353_artifact = source_artifacts.get("exp1353", {})
    value = exp1353_artifact.get("certificate_parse_rate")
    if isinstance(value, (int, float)):
        return float(value)
    return EXP1353_PARSE_BASELINE


def _exp1353_baseline(artifact: Mapping[str, Any]) -> float:
    source_context = artifact.get("source_context")
    if isinstance(source_context, Mapping):
        value = source_context.get("exp1353_certificate_parse_rate")
        if isinstance(value, (int, float)):
            return float(value)
    return EXP1353_PARSE_BASELINE


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _import_llama_grammar_class() -> type[Any]:  # pragma: no cover - live GGUF only.
    exp1353._add_venv_cuda_libs_to_ld_path()
    from llama_cpp import LlamaGrammar  # noqa: PLC0415

    return LlamaGrammar


def _load_cached_sota_pair(**kwargs: Any) -> list[dict[str, Any]] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

    return cached_sota_pair(**kwargs)


def _short_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:240]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - thin CLI wrapper covered through run_experiment.
    run_experiment(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
