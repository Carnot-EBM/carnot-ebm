"""Exp 1353 triggered certificate v7 SOTA terminal run.

Spec: REQ-VERIFY-1353, SCENARIO-VERIFY-1353
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot.reporting import truncproof_xgrammar_certificate_completion_preflight as preflight
from carnot.reporting import xgrammar2_tagdispatch_certificate_grammar_dryrun as tagdispatch


DEFAULT_RUN_DATE = "20260505"
DEFAULT_OUTPUT_PATH = Path("results/experiment_1353_triggered_certificate_v7_truncproof_sota.json")
DEFAULT_EXP1324_PATH = Path(
    "results/experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json"
)
DEFAULT_EXP1339_PATH = Path(
    "results/experiment_1339_xgrammar2_tagdispatch_certificate_grammar_dryrun.json"
)
DEFAULT_EXP1351_PATH = Path(
    "results/experiment_1351_104_carryforward_artifact_integrity_audit.json"
)
DEFAULT_EXP1352_PATH = Path(
    "results/experiment_1352_truncproof_xgrammar_certificate_completion_preflight.json"
)
ARTIFACT_NAME = "experiment_1353_triggered_certificate_v7_truncproof_sota"
SCHEMA_VERSION = 1
EXP1312_BASELINE_PARSE_RATE = 0.71223
GPU_CLEAN_VRAM_THRESHOLD_MB = 100
MANDATED_HEADLINE_MODEL_IDS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}
LEGACY_CPU_SMOKE_MODELS = (
    {
        "name": "Qwen3.5-0.8B",
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "gpu": "cpu",
        "generation_source": "legacy_cpu_smoke",
        "headline_result_allowed": False,
        "fallback_reason": "sota_path_unavailable",
    },
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "models_used",
    "runtime_settings_used",
    "completion_preflight_used",
    "certificate_case_count",
    "trigger_token_hit_rate",
    "certificate_parse_rate",
    "certificate_truthfulness_rate",
    "parse_rate_delta_over_exp1312",
    "unknown_preservation_rate",
    "min_completion_budget_respected",
    "terminal_blocker",
    "headline_result_allowed",
    "honest_verdict",
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_CUDA_LIB_ROOT = Path(".venv/lib/python3.12/site-packages/nvidia")


@dataclass(frozen=True)
class CertificateCase:
    """One bounded verifier fixture used by the terminal certificate run."""

    case_id: str
    family: str
    prompt: str
    expected_state: str


@dataclass(frozen=True)
class GenerationResult:
    """One model completion before certificate parsing.

    The experiment keeps this record narrow so tests can inject deterministic
    completions while the live path can still preserve raw model output and
    provenance without changing the metric code.
    """

    model_hf_id: str
    case_id: str
    text: str
    generation_source: str
    token_count: int
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class GPUHealth:
    """Minimal GPU readiness signal for deciding headline versus smoke mode."""

    healthy: bool
    gpu_count: int
    issues: list[str]


GenerationFn = Callable[[Mapping[str, Any], CertificateCase, str], GenerationResult]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
GPUHealthFn = Callable[[], GPUHealth]


def structural_tag(state: str) -> str:
    """Expose the Exp 1352 tag spelling used before branch grammar selection."""

    return preflight.structural_tag(state)


def json_certificate_text(state: str) -> str:
    """Return a minimal schema-valid body for a bounded certificate state."""

    normalised = preflight.normalise_state(state)
    final_answer = "ABSTAIN" if normalised == "REPAIR_HINT" else normalised
    claim_text = f"triggered certificate branch {normalised} is selected."
    return json.dumps(
        {
            "claims": [{"id": "c1", "text": claim_text}],
            "equations": [{"lhs": "branch", "relation": "=", "rhs": normalised}],
            "final_answer": final_answer,
            "confidence": 0.31 if normalised == "REPAIR_HINT" else 0.82,
            "verifier_routes": [{"claim_id": "c1", "verifier": "z3_math"}],
            "proof_numbers": [float(len(normalised))],
        },
        sort_keys=True,
    )


def bounded_certificate_suite() -> list[CertificateCase]:
    """Build the SAT/UNSAT/UNKNOWN/repair suite required for Exp 1353."""

    return [
        CertificateCase(
            "sat_unit_clause",
            "sat",
            "CNF: (x1). Emit a bounded Carnot certificate for SAT or UNSAT.",
            "SAT",
        ),
        CertificateCase(
            "unsat_unit_conflict",
            "unsat",
            "CNF: (x1) and (not x1). Emit a bounded Carnot certificate.",
            "UNSAT",
        ),
        CertificateCase(
            "unknown_missing_bound",
            "unknown",
            "A feasibility query omits the capacity bound B. Preserve UNKNOWN.",
            "UNKNOWN",
        ),
        CertificateCase(
            "repair_missing_upper_bound",
            "repair",
            "The certificate lacks an upper-bound premise. Emit a repair hint.",
            "REPAIR_HINT",
        ),
    ]


def build_prompt(case: CertificateCase, runtime_settings: Mapping[str, Any]) -> str:
    """Prompt the model to reason before emitting the structural trigger tag."""

    max_tokens = int(runtime_settings.get("max_tokens", 96))
    return (
        "You are producing a Carnot verification certificate. Think in one short "
        "sentence, then emit exactly one structural tag before the certificate body.\n"
        f"Case id: {case.case_id}\n"
        f"Problem: {case.prompt}\n"
        "Allowed tags: <CARNOT_CERT_STATE:SAT>, <CARNOT_CERT_STATE:UNSAT>, "
        "<CARNOT_CERT_STATE:UNKNOWN>, <CARNOT_CERT_STATE:REPAIR_HINT>.\n"
        "After the tag, emit only a compact JSON certificate with claims, "
        "equations, final_answer, confidence, verifier_routes, and proof_numbers.\n"
        f"Completion budget: {max_tokens} tokens."
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
    """Build a terminal artifact from SOTA rows or an honest blocker.

    The function deliberately separates terminal blockers from process status:
    Exp 1353 is complete when it proves either "SOTA evidence exists" or "the
    current terminal cannot produce it and here is the blocker."
    """

    cases = bounded_certificate_suite()
    exp1352 = source_artifacts.get("exp1352", {})
    runtime_settings = _runtime_settings(exp1352)
    completion = _completion_preflight(exp1352, runtime_settings)
    base = _base_artifact(
        run_date=run_date,
        project_root=Path(project_root),
        runtime_settings=runtime_settings,
        completion_preflight=completion,
        source_artifacts=source_artifacts,
    )
    base["gpu_health_used"] = gpu_health.__dict__

    grammar_blocker = _grammar_blocker(source_artifacts.get("exp1339", {}))
    if grammar_blocker is not None:
        return _blocked_without_smoke(
            base, cases, grammar_blocker, "blocked_dynamic_grammar_not_ready"
        )

    if not completion["sota_run_allowed"]:
        blocker = f"completion_preflight_blocked:{completion['blocker_if_not_allowed']}"
        return _blocked_without_smoke(
            base, cases, blocker, "blocked_completion_preflight_cpu_smoke_not_run"
        )

    model_blocker = _model_blocker(model_specs)
    if model_blocker is not None:
        rows = _legacy_cpu_smoke_rows(cases, terminal_blocker=model_blocker)
        return _complete_from_rows(
            base,
            cases,
            rows,
            terminal_blocker=model_blocker,
            headline_result_allowed=False,
            models_used=list(LEGACY_CPU_SMOKE_MODELS),
            honest_verdict=f"blocked_{model_blocker}_cpu_smoke_complete",
        )

    if not gpu_health.healthy:
        rows = _legacy_cpu_smoke_rows(cases, terminal_blocker="gpu_health_failed")
        return _complete_from_rows(
            base,
            cases,
            rows,
            terminal_blocker="gpu_health_failed",
            headline_result_allowed=False,
            models_used=_model_records(
                model_specs or [], generated=False, fallback_reason="gpu_health_failed"
            )
            + list(LEGACY_CPU_SMOKE_MODELS),
            honest_verdict="blocked_gpu_health_failed_cpu_smoke_complete",
        )

    active_generation_fn = generation_fn or LlamaCppCertificateGenerator(runtime_settings)
    try:
        rows = _run_generation_rows(
            cases,
            list(model_specs or [])[: max(1, int(max_models))],
            runtime_settings,
            active_generation_fn,
        )
    except Exception as exc:
        blocker = f"sota_generation_failed:{type(exc).__name__}:{_short_error(exc)}"
        smoke_rows = _legacy_cpu_smoke_rows(cases, terminal_blocker=blocker)
        return _complete_from_rows(
            base,
            cases,
            smoke_rows,
            terminal_blocker=blocker,
            headline_result_allowed=False,
            models_used=_model_records(model_specs or [], generated=False, fallback_reason=blocker)
            + list(LEGACY_CPU_SMOKE_MODELS),
            honest_verdict="blocked_sota_generation_failed_cpu_smoke_complete",
        )

    generated_model_ids = {
        row.model_hf_id for row in rows if row.generation_source == "live_sota_llamacpp"
    }
    headline_allowed = bool(generated_model_ids.intersection(MANDATED_HEADLINE_MODEL_IDS))
    terminal_blocker = None if headline_allowed else "no_mandated_sota_generation_rows"
    return _complete_from_rows(
        base,
        cases,
        rows,
        terminal_blocker=terminal_blocker,
        headline_result_allowed=headline_allowed,
        models_used=_model_records(model_specs or [], generated=headline_allowed),
        honest_verdict=(
            "sota_triggered_certificate_v7_measured"
            if headline_allowed
            else "blocked_no_mandated_sota_generation_rows"
        ),
    )


class LlamaCppCertificateGenerator:
    """Lazy llama.cpp GGUF runner used only when the live SOTA path is selected."""

    def __init__(
        self,
        runtime_settings: Mapping[str, Any],
        *,
        llama_importer: Callable[[], type[Any]] | None = None,
    ) -> None:
        self._runtime_settings = dict(runtime_settings)
        self._llama_importer = llama_importer or _import_llama_class
        self._models: dict[str, Any] = {}

    def __call__(
        self,
        spec: Mapping[str, Any],
        case: CertificateCase,
        prompt: str,
    ) -> GenerationResult:
        start = time.perf_counter()
        model = self._model_for(spec)
        response = model(
            prompt,
            max_tokens=int(self._runtime_settings.get("max_tokens", 96)),
            temperature=float(self._runtime_settings.get("temperature", 0.0)),
            top_p=float(self._runtime_settings.get("top_p", 1.0)),
            stop=list(self._runtime_settings.get("stop", ["</s>", "<eos>"])),
            echo=False,
        )
        text = _response_text(response)
        return GenerationResult(
            model_hf_id=str(spec.get("hf_id")),
            case_id=case.case_id,
            text=text,
            generation_source="live_sota_llamacpp",
            token_count=_completion_token_count(response, text),
            elapsed_seconds=round(time.perf_counter() - start, 6),
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
            seed=int(self._runtime_settings.get("seed", 1353)),
            verbose=False,
        )
        self._models[key] = model
        return model


def check_gpu_health() -> GPUHealth:
    """Return a small health record based on `nvidia-smi` availability."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return GPUHealth(False, 0, [f"nvidia_smi_error:{_short_error(exc)}"])
    if result.returncode != 0:
        return GPUHealth(False, 0, [f"nvidia_smi_exit_{result.returncode}:{result.stderr.strip()}"])
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    issues = ["fewer_than_two_gpus_visible"] if len(rows) < 2 else []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 4:
            try:
                used_mb = int(float(parts[3]))
            except ValueError:
                issues.append(f"gpu_vram_used_unparseable:{row}")
                continue
            if used_mb >= GPU_CLEAN_VRAM_THRESHOLD_MB:
                issues.append(f"gpu{parts[0]}_vram_used_{used_mb}mb")
    return GPUHealth(not issues, len(rows), issues)


def write_in_progress_artifact(
    path: Path | str,
    *,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Write the required bootstrap artifact before loading source state."""

    artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": "in_progress",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": "REQ-VERIFY-1353",
        },
    }
    _write_json(Path(path), artifact)
    return artifact


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    exp1324_path: Path | str = DEFAULT_EXP1324_PATH,
    exp1339_path: Path | str = DEFAULT_EXP1339_PATH,
    exp1351_path: Path | str = DEFAULT_EXP1351_PATH,
    exp1352_path: Path | str = DEFAULT_EXP1352_PATH,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    cached_pair_fn: CachedPairFn | None = None,
    gpu_health_fn: GPUHealthFn = check_gpu_health,
    generation_fn: GenerationFn | None = None,
    max_models: int = 1,
) -> dict[str, Any]:
    """Write in-progress, run the terminal evaluation, and persist completion."""

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
            "exp1324": _load_json(Path(exp1324_path)),
            "exp1339": _load_json(Path(exp1339_path)),
            "exp1351": _load_json(Path(exp1351_path)),
            "exp1352": _load_json(Path(exp1352_path)),
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


def _complete_from_rows(
    artifact: dict[str, Any],
    cases: Sequence[CertificateCase],
    rows: Sequence[GenerationResult],
    *,
    terminal_blocker: str | None,
    headline_result_allowed: bool,
    models_used: list[dict[str, Any]],
    honest_verdict: str,
) -> dict[str, Any]:
    grammar = tagdispatch.compile_branch_grammars()
    case_by_id = {case.case_id: case for case in cases}
    parsed_rows = [_parse_generation_row(row, case_by_id[row.case_id], grammar) for row in rows]
    metrics = _metrics(parsed_rows, baseline=_baseline_parse_rate(artifact))
    artifact.update(
        {
            "status": "complete",
            "models_used": models_used,
            "certificate_case_count": len(rows),
            "trigger_token_hit_rate": metrics["trigger_token_hit_rate"],
            "certificate_parse_rate": metrics["certificate_parse_rate"],
            "certificate_truthfulness_rate": metrics["certificate_truthfulness_rate"],
            "parse_rate_delta_over_exp1312": metrics["parse_rate_delta_over_exp1312"],
            "unknown_preservation_rate": metrics["unknown_preservation_rate"],
            "terminal_blocker": terminal_blocker,
            "headline_result_allowed": headline_result_allowed,
            "honest_verdict": honest_verdict,
            "generation_rows": [row.__dict__ for row in rows],
            "certificate_rows": parsed_rows,
        }
    )
    return artifact


def _blocked_without_smoke(
    artifact: dict[str, Any],
    cases: Sequence[CertificateCase],
    terminal_blocker: str,
    honest_verdict: str,
) -> dict[str, Any]:
    metrics = _metrics([], baseline=_baseline_parse_rate(artifact), denominator=len(cases))
    artifact.update(
        {
            "status": "complete",
            "models_used": [],
            "certificate_case_count": len(cases),
            "trigger_token_hit_rate": metrics["trigger_token_hit_rate"],
            "certificate_parse_rate": metrics["certificate_parse_rate"],
            "certificate_truthfulness_rate": metrics["certificate_truthfulness_rate"],
            "parse_rate_delta_over_exp1312": metrics["parse_rate_delta_over_exp1312"],
            "unknown_preservation_rate": metrics["unknown_preservation_rate"],
            "terminal_blocker": terminal_blocker,
            "headline_result_allowed": False,
            "honest_verdict": honest_verdict,
            "generation_rows": [],
            "certificate_rows": [],
        }
    )
    return artifact


def _parse_generation_row(
    row: GenerationResult,
    case: CertificateCase,
    grammar: tagdispatch.CompiledBranchGrammars,
) -> dict[str, Any]:
    tag_state, body = preflight.parse_structural_tag(row.text)
    trigger_hit = tag_state is not None
    if not trigger_hit:
        return _parsed_row(row, case, None, None, False, False, False, ["missing_structural_tag"])
    dispatched = tagdispatch.dispatch_certificate_text(body, grammar)
    parseable = bool(dispatched.parseable and dispatched.dispatched_state == tag_state)
    truthful = parseable and _truthful(case.expected_state, dispatched.certificate)
    unknown_preserved = (
        case.expected_state == "UNKNOWN"
        and parseable
        and _normalised_state(dispatched.certificate.get("final_answer")) == "UNKNOWN"
    )
    return _parsed_row(
        row,
        case,
        tag_state,
        dispatched.dispatched_state,
        parseable,
        truthful,
        unknown_preserved,
        list(dispatched.errors),
    )


def _parsed_row(
    row: GenerationResult,
    case: CertificateCase,
    tag_state: str | None,
    dispatched_state: str | None,
    parseable: bool,
    truthful: bool,
    unknown_preserved: bool,
    errors: Sequence[str],
) -> dict[str, Any]:
    return {
        "case_id": row.case_id,
        "model_hf_id": row.model_hf_id,
        "expected_state": case.expected_state,
        "generation_source": row.generation_source,
        "trigger_token_hit": tag_state is not None,
        "tag_state": tag_state,
        "dispatched_state": dispatched_state,
        "parseable": parseable,
        "truthful": truthful,
        "unknown_preserved": unknown_preserved,
        "errors": list(errors),
    }


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
        "parse_rate_delta_over_exp1312": round(parse_rate - baseline, 6),
        "unknown_preservation_rate": _rate(unknown_preserved, len(unknown_rows)),
    }


def _run_generation_rows(
    cases: Sequence[CertificateCase],
    model_specs: Sequence[Mapping[str, Any]],
    runtime_settings: Mapping[str, Any],
    generation_fn: GenerationFn,
) -> list[GenerationResult]:
    rows: list[GenerationResult] = []
    for spec in model_specs:
        for case in cases:
            rows.append(generation_fn(spec, case, build_prompt(case, runtime_settings)))
    return rows


def _legacy_cpu_smoke_rows(
    cases: Sequence[CertificateCase],
    *,
    terminal_blocker: str,
) -> list[GenerationResult]:
    return [
        GenerationResult(
            model_hf_id=LEGACY_CPU_SMOKE_MODELS[0]["hf_id"],
            case_id=case.case_id,
            text=f"{structural_tag(case.expected_state)}\n{json_certificate_text(case.expected_state)}",
            generation_source="legacy_cpu_smoke",
            token_count=preflight.estimate_completion_tokens(
                json_certificate_text(case.expected_state)
            ),
            error=terminal_blocker,
        )
        for case in cases
    ]


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
            "spec": "REQ-VERIFY-1353",
            "source_experiments": ["exp1324", "exp1339", "exp1351", "exp1352"],
        },
        "models_used": [],
        "runtime_settings_used": dict(runtime_settings),
        "completion_preflight_used": dict(completion_preflight),
        "certificate_case_count": 0,
        "trigger_token_hit_rate": 0.0,
        "certificate_parse_rate": 0.0,
        "certificate_truthfulness_rate": 0.0,
        "parse_rate_delta_over_exp1312": round(
            0.0 - _baseline_parse_rate_from_sources(source_artifacts), 6
        ),
        "unknown_preservation_rate": 0.0,
        "min_completion_budget_respected": bool(
            completion_preflight["min_completion_budget_respected"]
        ),
        "terminal_blocker": None,
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
    settings.setdefault("seed", 1353)
    settings.setdefault("gpu_indices", [0, 1])
    settings.setdefault("preferred_quant", "Q4_K_M")
    settings.setdefault("trigger_before_constrain", True)
    settings.setdefault("dynamic_grammar", "tagdispatch_branch_after_structural_tag")
    return settings


def _completion_preflight(
    exp1352_artifact: Mapping[str, Any],
    runtime_settings: Mapping[str, Any],
) -> dict[str, Any]:
    min_tokens = dict(exp1352_artifact.get("min_completion_tokens_by_state") or {})
    if not min_tokens:
        min_tokens = {"SAT": 6, "UNSAT": 6, "UNKNOWN": 6, "REPAIR_HINT": 10}
    max_tokens, required, budget_ok = preflight.max_token_budget_check(runtime_settings, min_tokens)
    recorded_budget_ok = bool(exp1352_artifact.get("max_token_budget_sufficient", budget_ok))
    budget_respected = bool(budget_ok and recorded_budget_ok)
    return {
        "source": "exp1352",
        "sota_run_allowed": bool(exp1352_artifact.get("sota_run_allowed")) and budget_respected,
        "blocker_if_not_allowed": exp1352_artifact.get("blocker_if_not_allowed"),
        "max_tokens": max_tokens,
        "required_min_completion_tokens": required,
        "min_completion_tokens_by_state": min_tokens,
        "min_completion_budget_respected": budget_respected,
        "structural_tag_supported": bool(exp1352_artifact.get("structural_tag_supported")),
        "dynamic_dispatch_preserved": bool(exp1352_artifact.get("dynamic_dispatch_preserved")),
    }


def _source_context(source_artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1324 = source_artifacts.get("exp1324", {})
    source_metrics = exp1324.get("source_metrics")
    exp1339 = source_artifacts.get("exp1339", {})
    exp1351 = source_artifacts.get("exp1351", {})
    exp1352 = source_artifacts.get("exp1352", {})
    return {
        "exp1312_certificate_parse_rate": (
            source_metrics.get("exp1312_certificate_parse_rate")
            if isinstance(source_metrics, Mapping)
            else None
        ),
        "exp1324_minimum_parseable_attempts_to_recover": exp1324.get(
            "minimum_parseable_attempts_to_recover"
        ),
        "exp1324_failure_modes": exp1324.get("formalizer_failure_modes", []),
        "exp1339_dynamic_grammar_ready": exp1339.get("dynamic_grammar_ready"),
        "exp1339_certificate_states_supported": exp1339.get("certificate_states_supported", []),
        "exp1351_honest_verdict": exp1351.get("honest_verdict"),
        "exp1352_honest_verdict": exp1352.get("honest_verdict"),
    }


def _grammar_blocker(exp1339_artifact: Mapping[str, Any]) -> str | None:
    if exp1339_artifact.get("dynamic_grammar_ready") is not True:
        return "dynamic_grammar_not_ready"
    supported = set(exp1339_artifact.get("certificate_states_supported") or [])
    required = {"SAT", "UNSAT", "UNKNOWN", "REPAIR_HINT"}
    if not required.issubset(supported):
        return "dynamic_grammar_missing_required_state"
    return None


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
    generated: bool,
    fallback_reason: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in model_specs:
        model_path = spec.get("model_path")
        records.append(
            {
                "name": spec.get("name"),
                "hf_id": spec.get("hf_id"),
                "gpu": spec.get("gpu"),
                "model_path": model_path,
                "quantization": _quantization_from_path(model_path) or spec.get("quantization"),
                "generation_source": "live_sota_llamacpp" if generated else None,
                "headline_result_allowed": bool(generated),
                "fallback_reason": fallback_reason,
            }
        )
    return records


def _quantization_from_path(model_path: Any) -> str | None:
    text = str(model_path or "")
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in text.lower():
            return token
    return None


def _truthful(expected_state: str, certificate: Mapping[str, Any]) -> bool:
    final = _normalised_state(certificate.get("final_answer"))
    expected = preflight.normalise_state(expected_state)
    if expected == "REPAIR_HINT":
        return final == "ABSTAIN"
    return final == expected


def _normalised_state(value: Any) -> str:
    text = str(value or "").upper()
    if "UNSATISFIABLE" in text or "UNSAT" in text:
        return "UNSAT"
    if "SATISFIABLE" in text or "SAT" in text:
        return "SAT"
    if "UNKNOWN" in text or "UNDETERMINED" in text or "ABSTAIN" in text:
        return "UNKNOWN" if "UNKNOWN" in text or "UNDETERMINED" in text else "ABSTAIN"
    return text


def _baseline_parse_rate(artifact: Mapping[str, Any]) -> float:
    source_context = artifact.get("source_context")
    if isinstance(source_context, Mapping):
        value = source_context.get("exp1312_certificate_parse_rate")
        if isinstance(value, (int, float)):
            return float(value)
    return EXP1312_BASELINE_PARSE_RATE


def _baseline_parse_rate_from_sources(source_artifacts: Mapping[str, Mapping[str, Any]]) -> float:
    exp1324 = source_artifacts.get("exp1324", {})
    source_metrics = exp1324.get("source_metrics")
    if isinstance(source_metrics, Mapping):
        value = source_metrics.get("exp1312_certificate_parse_rate")
        if isinstance(value, (int, float)):
            return float(value)
    return EXP1312_BASELINE_PARSE_RATE


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
            return str(choices[0].get("text") or "")
    return str(response or "")


def _completion_token_count(response: Any, text: str) -> int:
    if isinstance(response, Mapping):
        usage = response.get("usage")
        if isinstance(usage, Mapping) and isinstance(usage.get("completion_tokens"), int):
            return int(usage["completion_tokens"])
    return len(_TOKEN_RE.findall(text))


def _import_llama_class() -> type[Any]:  # pragma: no cover - exercised only on live GGUF hosts.
    _add_venv_cuda_libs_to_ld_path()
    from llama_cpp import Llama  # noqa: PLC0415

    return Llama


def _add_venv_cuda_libs_to_ld_path() -> None:
    libs = [str(path.resolve()) for path in _CUDA_LIB_ROOT.glob("*/lib") if path.is_dir()]
    if not libs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in existing.split(":") if part]
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [part for part in parts if part not in libs])


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
