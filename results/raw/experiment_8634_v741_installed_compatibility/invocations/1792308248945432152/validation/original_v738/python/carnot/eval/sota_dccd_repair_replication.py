"""Exp 2964 live SOTA DCCD structured code-repair replication.

The replication is the first gate after Exp 2963's protocol-only manifest.  It
does not inherit the small Exp 2952 positive delta as a claim; it reruns
baseline repair, taxonomy-guided repair, and DCCD structured repair on the same
failed Exp 2946 candidates, then promotes only when the pre-registered sample,
improvement, and false-accept gates all clear.

Spec: REQ-CODE-2964, SCENARIO-CODE-2964.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.eval import sota_taxonomy_guided_code_repair_eval as exp2952
from carnot.eval import structured_candidate_manifest_adapter as exp2951
from carnot.reporting.verifier_ensemble_auprc_code_corpora_2940 import (
    approval_score_from_energy,
    candidate_status_energy,
)


JsonDict = dict[str, Any]
GenerationOutcome = exp2952.GenerationOutcome
ExecutionOutcome = exp2952.ExecutionOutcome
PreconditionReport = exp2952.PreconditionReport
RepairGenerator = exp2952.RepairGenerator
Executor = exp2952.Executor
PreconditionProbe = Callable[["ExperimentConfig"], PreconditionReport]
TaskRowProvider = Callable[["ExperimentConfig"], dict[tuple[str, str], JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2964_sota_dccd_repair_replication_v1.json"
ARTIFACT = "experiment_2964_sota_dccd_repair_replication_v1"
SCHEMA = "carnot.sota_dccd_repair_replication.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"

EXP2946_REL_PATH = exp2952.EXP2946_REL_PATH
NESTED_EXP2946_REL_PATH = exp2952.NESTED_EXP2946_REL_PATH
EXP2950_REL_PATH = exp2952.EXP2950_REL_PATH
EXP2951_REL_PATH = exp2952.EXP2951_REL_PATH
EXP2952_REL_PATH = Path("results/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json")
EXP2953_REL_PATH = Path("results/experiment_2953_code_verifier_threshold_policy_v1.json")
EXP2963_REL_PATH = Path("results/experiment_2963_dccd_repair_protocol_manifest_v1.json")
RAW_RESPONSE_REL_DIR = Path("results/raw/experiment_2964_sota_dccd_repair_replication_v1")

BASELINE_MODE = exp2952.BASELINE_MODE
TAXONOMY_MODE = exp2952.REPAIR_MODE
DCCD_MODE = "dccd_structured"
MODES = (BASELINE_MODE, TAXONOMY_MODE, DCCD_MODE)

DEFAULT_RANDOM_SEED = 296300
DEFAULT_N_TASKS = 20
DEFAULT_SAMPLES_PER_MODE = 2
DEFAULT_MAX_TOKENS = 384
DEFAULT_TEMPERATURE = 0.2
DEFAULT_SANDBOX_TIMEOUT_S = 10.0

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "model_specs",
    "headline_models_used",
    "legacy_models_only_for_smoke",
    "n_tasks",
    "baseline_pass_at_1",
    "taxonomy_repair_pass_at_1",
    "dccd_repair_pass_at_1",
    "pass_at_1_delta",
    "baseline_pass_at_k",
    "dccd_repair_pass_at_k",
    "pass_at_k_delta",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "false_accept_delta",
    "dccd_repair_replication_clean",
    "candidate_manifest_sha256",
    "reproducibility_checksum",
    "duration_s",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2964 replication runner."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    raw_response_dir: Path | None = None
    exp2946_path: Path = EXP2946_REL_PATH
    nested_exp2946_path: Path = NESTED_EXP2946_REL_PATH
    exp2950_path: Path = EXP2950_REL_PATH
    exp2951_path: Path = EXP2951_REL_PATH
    exp2952_path: Path = EXP2952_REL_PATH
    exp2953_path: Path = EXP2953_REL_PATH
    exp2963_path: Path = EXP2963_REL_PATH
    n_tasks: int = DEFAULT_N_TASKS
    samples_per_mode: int = DEFAULT_SAMPLES_PER_MODE
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    random_seed: int = DEFAULT_RANDOM_SEED
    sandbox_timeout_s: float = DEFAULT_SANDBOX_TIMEOUT_S
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def raw_dir(self) -> Path:
        return self.raw_response_dir or self.repo_root / RAW_RESPONSE_REL_DIR


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator: RepairGenerator | None = None,
    executor: Executor = exp2952.exp2910.execute_script_in_sandbox,
    precondition_probe: PreconditionProbe = None,
    task_row_provider: TaskRowProvider = None,
) -> JsonDict:
    """Build the Exp 2964 artifact and run live repair when gates are open."""

    config = config or ExperimentConfig()
    started = config.start_time()
    precondition_probe = precondition_probe or default_precondition_probe
    task_row_provider = task_row_provider or default_task_row_provider
    source_checks = _source_precondition_checks(config)
    report = precondition_probe(config)
    preconditions_checked = source_checks + [dict(row) for row in report.checks]

    if not all(row["available"] for row in preconditions_checked) or not report.runnable_model_specs:
        return _blocked_artifact(
            config=config,
            started=started,
            verdict="blocked_preconditions_failed",
            preconditions_checked=preconditions_checked,
            model_specs=report.model_specs,
        )

    exp2946 = _read_json(_repo_path(config.repo_root, config.exp2946_path))
    exp2950_payload = _read_json(_repo_path(config.repo_root, config.exp2950_path))
    exp2953_payload = _read_json(_repo_path(config.repo_root, config.exp2953_path))
    nested = _read_json(_repo_path(config.repo_root, _nested_protocol_path(config, exp2946)))
    selected = exp2952.select_repair_set(nested, task_row_provider(config), config.n_tasks)
    if not selected:
        return _blocked_artifact(
            config=config,
            started=started,
            verdict="blocked_no_failed_or_low_scoring_candidates",
            preconditions_checked=preconditions_checked,
            model_specs=report.model_specs,
        )

    model_spec = dict(report.runnable_model_specs[0])
    live_generator = generator or exp2952.llama_cpp_repair_generator(
        model_path=str(model_spec["model_path"]),
        main_gpu=int(model_spec.get("gpu") or 0),
        temperature=config.temperature,
    )
    threshold = _verifier_threshold(exp2953_payload)
    templates = dict(exp2950_payload.get("repair_prompt_templates") or {})
    evaluations: list[JsonDict] = []
    manifests: list[JsonDict] = []

    for source in selected:
        for mode in MODES:
            for sample_index in range(config.samples_per_mode):
                seed = _candidate_seed(config, mode, source.task_index, sample_index)
                prompt = _prompt_for_mode(source, mode, templates)
                generation = live_generator(prompt, seed, config.max_tokens, model_spec)
                if mode == DCCD_MODE:
                    evaluation, manifest = evaluate_dccd_candidate(
                        config=config,
                        source=source,
                        sample_index=sample_index,
                        seed=seed,
                        prompt=prompt,
                        generation=generation,
                        model_spec=model_spec,
                        threshold=threshold,
                        executor=executor,
                    )
                else:
                    evaluation, manifest = exp2952.evaluate_repair_candidate(
                        config=config,
                        source=source,
                        mode=mode,
                        sample_index=sample_index,
                        seed=seed,
                        prompt=prompt,
                        generation=generation,
                        model_spec=model_spec,
                        threshold=threshold,
                        executor=executor,
                    )
                evaluations.append(evaluation)
                manifests.append(manifest)

    return _complete_artifact(
        config=config,
        started=started,
        preconditions_checked=preconditions_checked,
        model_specs=report.model_specs,
        selected=selected,
        evaluations=evaluations,
        manifests=manifests,
    )


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator: RepairGenerator | None = None,
    executor: Executor = exp2952.exp2910.execute_script_in_sandbox,
    precondition_probe: PreconditionProbe = None,
    task_row_provider: TaskRowProvider = None,
) -> JsonDict:
    """Build and persist the Exp 2964 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(
        config,
        generator=generator,
        executor=executor,
        precondition_probe=precondition_probe or default_precondition_probe,
        task_row_provider=task_row_provider or default_task_row_provider,
    )
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def evaluate_dccd_candidate(
    *,
    config: ExperimentConfig,
    source: exp2952.RepairSource,
    sample_index: int,
    seed: int,
    prompt: str,
    generation: GenerationOutcome,
    model_spec: Mapping[str, Any],
    threshold: float,
    executor: Executor,
) -> tuple[JsonDict, JsonDict]:
    """Validate one DCCD structured candidate before deterministic code checks."""

    raw_response_ref = exp2952._write_raw_response(
        config,
        source=source,
        mode=DCCD_MODE,
        sample_index=sample_index,
        seed=seed,
        text=generation.text,
    )
    parsed, parse_errors = _parse_json_object(generation.text)
    adapter = exp2951.StructuredCandidateManifestAdapter()
    raw_manifest = parsed if isinstance(parsed, Mapping) else {}
    raw_validation = adapter.validate_record(raw_manifest) if raw_manifest else None
    schema_errors = parse_errors + ([] if raw_validation is None else raw_validation.errors)
    schema_valid = raw_validation.ok if raw_validation is not None else False
    if not schema_valid:
        manifest = dict(raw_manifest) if raw_manifest else {"raw_model_output": generation.text}
        evaluation = _schema_failed_evaluation(
            source=source,
            sample_index=sample_index,
            seed=seed,
            prompt=prompt,
            generation=generation,
            model_spec=model_spec,
            raw_response_ref=raw_response_ref,
            schema_errors=schema_errors,
        )
        return evaluation, manifest

    repaired_code = str(raw_manifest.get("repaired_code") or "")
    extraction = exp2952.exp2910.extract_python_candidate(repaired_code)
    code_to_run = extraction.code or repaired_code.strip()
    static_checks = exp2952._static_checks(code_to_run) if extraction.syntax_success else exp2952._syntax_static_checks()
    outcome = exp2952._execute_candidate(
        config,
        source,
        code_to_run,
        extraction.syntax_success,
        static_checks,
        executor,
    )
    test_status = "passed" if outcome.passed else "failed" if extraction.syntax_success else "not_run"
    runtime_success = exp2952._runtime_success(extraction.syntax_success, outcome)
    verifier_score = approval_score_from_energy(
        candidate_status_energy(
            {
                "extraction_success": extraction.extraction_success,
                "syntax_success": extraction.syntax_success,
                "runtime_success": runtime_success,
                "passed": outcome.passed,
            }
        )
    )
    manifest = exp2952._candidate_manifest(
        source=source,
        mode=DCCD_MODE,
        sample_index=sample_index,
        seed=seed,
        model_id=str(model_spec.get("hf_id") or ""),
        raw_response_ref=raw_response_ref,
        raw_response_text=generation.text,
        repaired_code=code_to_run,
        failure_taxonomy=exp2952._post_repair_taxonomy(
            extraction.syntax_success,
            static_checks,
            outcome.passed,
        ),
        parser_status="parsed" if extraction.syntax_success else "syntax_error",
        test_status=test_status,
        verifier_score=verifier_score,
    )
    normalized_validation = adapter.validate_record(manifest)
    verifier_accepted = normalized_validation.ok and verifier_score >= threshold
    false_accept = verifier_accepted and not outcome.passed
    evaluation = {
        "mode": DCCD_MODE,
        "task_id": source.task_key,
        "stable_id": source.stable_id,
        "corpus": source.corpus,
        "sample_id": source.sample_id,
        "sample_index": sample_index,
        "seed": seed,
        "model_hf_id": str(model_spec.get("hf_id") or ""),
        "model_path": str(model_spec.get("model_path") or ""),
        "prompt_sha256": _sha256_text(prompt),
        "raw_response_ref": raw_response_ref,
        "raw_response_sha256": _sha256_text(generation.text),
        "generation_backend": generation.backend,
        "generation_backend_detail": generation.backend_detail,
        "generation_duration_s": float(generation.duration_s),
        "tokens_generated": int(generation.tokens_generated),
        "generation_error": generation.error,
        "original_failure_categories": list(source.original_failure_categories),
        "parser_status": manifest["parser_status"],
        "syntax_success": extraction.syntax_success,
        "static_checks": static_checks,
        "test_status": test_status,
        "passed": bool(outcome.passed),
        "execution_error_type": outcome.error_type,
        "execution_error_message": outcome.error_message,
        "verifier_score": verifier_score,
        "verifier_threshold": threshold,
        "verifier_accepted": verifier_accepted,
        "false_accept": false_accept,
        "schema_valid": normalized_validation.ok,
        "schema_errors": schema_errors + normalized_validation.errors,
        "candidate_manifest_sha256": exp2952._sha256_payload(manifest),
    }
    return evaluation, manifest


def default_precondition_probe(config: ExperimentConfig) -> PreconditionReport:  # pragma: no cover - hardware path.
    """Reuse the Exp 2952 live repair probe for GPU, runtime, sandbox, and GGUFs."""

    return exp2952.default_precondition_probe(config)


def default_task_row_provider(config: ExperimentConfig) -> dict[tuple[str, str], JsonDict]:  # pragma: no cover - filesystem path.
    """Load checked-in MBPP/HumanEval task rows for available task tests."""

    return exp2952.default_task_row_provider(config)


def _complete_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    preconditions_checked: list[JsonDict],
    model_specs: list[JsonDict],
    selected: Sequence[exp2952.RepairSource],
    evaluations: list[JsonDict],
    manifests: list[JsonDict],
) -> JsonDict:
    baseline = exp2952._mode_metrics(evaluations, BASELINE_MODE, selected)
    taxonomy = exp2952._mode_metrics(evaluations, TAXONOMY_MODE, selected)
    dccd = exp2952._mode_metrics(evaluations, DCCD_MODE, selected)
    deltas = _dccd_deltas(baseline, dccd)
    candidate_manifest_sha = exp2952._sha256_payload(manifests)
    selected_task_ids = [source.task_key for source in selected]
    clean = _dccd_replication_clean(len(selected), deltas)
    artifact = _base_artifact(
        config=config,
        started=started,
        verdict=_complete_verdict(clean, deltas),
        preconditions_checked=preconditions_checked,
        model_specs=model_specs,
        n_tasks=len(selected),
        candidate_manifest_sha256=candidate_manifest_sha,
        evaluations=evaluations,
        manifests=manifests,
    )
    artifact.update(
        {
            "headline_models_used": sorted(
                {str(row.get("model_hf_id")) for row in evaluations if row.get("model_hf_id")}
            ),
            "selected_task_ids": selected_task_ids,
            "selected_repair_set": [exp2952._selected_source_row(source) for source in selected],
            "sample_budget_per_mode": len(selected) * config.samples_per_mode,
            "baseline_pass_at_1": baseline["pass_at_1"],
            "taxonomy_repair_pass_at_1": taxonomy["pass_at_1"],
            "dccd_repair_pass_at_1": dccd["pass_at_1"],
            "pass_at_1_delta": deltas["pass_at_1_delta"],
            "baseline_pass_at_k": baseline["pass_at_k"],
            "dccd_repair_pass_at_k": dccd["pass_at_k"],
            "pass_at_k_delta": deltas["pass_at_k_delta"],
            "baseline_syntax_failure_rate": baseline["syntax_failure_rate"],
            "dccd_repair_syntax_failure_rate": dccd["syntax_failure_rate"],
            "syntax_failure_rate_delta": deltas["syntax_failure_rate_delta"],
            "baseline_schema_failure_rate": baseline["schema_failure_rate"],
            "dccd_repair_schema_failure_rate": dccd["schema_failure_rate"],
            "schema_failure_rate_delta": deltas["schema_failure_rate_delta"],
            "baseline_false_accept_rate": baseline["false_accept_rate"],
            "dccd_repair_false_accept_rate": dccd["false_accept_rate"],
            "false_accept_delta": deltas["false_accept_delta"],
            "dccd_repair_replication_clean": clean,
            "mode_metrics": {
                BASELINE_MODE: baseline,
                TAXONOMY_MODE: taxonomy,
                DCCD_MODE: dccd,
            },
            "false_accept_audit_notes": _false_accept_notes(deltas),
            "reproducibility_checksum": _reproducibility_checksum(
                selected_task_ids=selected_task_ids,
                candidate_manifest_sha256=candidate_manifest_sha,
                model_specs=model_specs,
                deltas=deltas,
            ),
        }
    )
    return artifact


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    verdict: str,
    preconditions_checked: list[JsonDict],
    model_specs: list[JsonDict],
) -> JsonDict:
    candidate_manifest_sha = exp2952._sha256_payload([])
    artifact = _base_artifact(
        config=config,
        started=started,
        verdict=verdict,
        preconditions_checked=preconditions_checked,
        model_specs=model_specs,
        n_tasks=0,
        candidate_manifest_sha256=candidate_manifest_sha,
        evaluations=[],
        manifests=[],
    )
    artifact.update(
        {
            "selected_task_ids": [],
            "selected_repair_set": [],
            "sample_budget_per_mode": 0,
            "baseline_pass_at_1": 0.0,
            "taxonomy_repair_pass_at_1": 0.0,
            "dccd_repair_pass_at_1": 0.0,
            "pass_at_1_delta": 0.0,
            "baseline_pass_at_k": 0.0,
            "dccd_repair_pass_at_k": 0.0,
            "pass_at_k_delta": 0.0,
            "baseline_syntax_failure_rate": 0.0,
            "dccd_repair_syntax_failure_rate": 0.0,
            "syntax_failure_rate_delta": 0.0,
            "baseline_schema_failure_rate": 0.0,
            "dccd_repair_schema_failure_rate": 0.0,
            "schema_failure_rate_delta": 0.0,
            "baseline_false_accept_rate": 0.0,
            "dccd_repair_false_accept_rate": 0.0,
            "false_accept_delta": 0.0,
            "dccd_repair_replication_clean": False,
            "mode_metrics": {},
            "false_accept_audit_notes": [verdict],
            "reproducibility_checksum": _reproducibility_checksum(
                selected_task_ids=[],
                candidate_manifest_sha256=candidate_manifest_sha,
                model_specs=model_specs,
                deltas={},
            ),
        }
    )
    return artifact


def _base_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    verdict: str,
    preconditions_checked: list[JsonDict],
    model_specs: list[JsonDict],
    n_tasks: int,
    candidate_manifest_sha256: str,
    evaluations: list[JsonDict],
    manifests: list[JsonDict],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "source_artifacts": _source_artifacts(config),
        "model_specs": model_specs,
        "headline_models_used": [],
        "legacy_models_only_for_smoke": True,
        "n_tasks": n_tasks,
        "samples_per_mode": config.samples_per_mode,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "candidate_manifests": manifests,
        "candidate_evaluations": evaluations,
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _source_precondition_checks(config: ExperimentConfig) -> list[JsonDict]:
    specs: tuple[tuple[str, Path, Callable[[Mapping[str, Any]], bool]], ...] = (
        ("exp2946_failed_candidates", config.exp2946_path, lambda payload: bool(payload)),
        ("exp2946_nested_protocol", config.nested_exp2946_path, lambda payload: bool(payload)),
        (
            "exp2950_repair_prompt_manifest",
            config.exp2950_path,
            lambda payload: payload.get("repair_prompt_manifest_ready") is True,
        ),
        (
            "exp2951_structured_candidate_manifest_adapter",
            config.exp2951_path,
            lambda payload: payload.get("structured_decode_manifest_ready") is True,
        ),
        ("exp2952_taxonomy_repair_reference", config.exp2952_path, lambda payload: bool(payload)),
        (
            "exp2953_threshold_policy",
            config.exp2953_path,
            lambda payload: payload.get("threshold_policy_ready") is True,
        ),
        (
            "exp2963_dccd_repair_protocol_ready",
            config.exp2963_path,
            lambda payload: payload.get("dccd_repair_protocol_ready") is True,
        ),
    )
    checks: list[JsonDict] = []
    for resource, rel_path, ready_fn in specs:
        path = _repo_path(config.repo_root, rel_path)
        payload = _read_json(path) if path.is_file() else {}
        checks.append(
            {
                "resource": resource,
                "available": path.is_file() and ready_fn(payload),
                "detail": str(rel_path),
                "sha256": exp2952._sha256_file(path) if path.is_file() else None,
            }
        )
    return checks


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    return [
        {
            "path": str(rel_path),
            "present": _repo_path(config.repo_root, rel_path).is_file(),
            "sha256": (
                exp2952._sha256_file(_repo_path(config.repo_root, rel_path))
                if _repo_path(config.repo_root, rel_path).is_file()
                else None
            ),
        }
        for rel_path in (
            config.exp2946_path,
            config.nested_exp2946_path,
            config.exp2950_path,
            config.exp2951_path,
            config.exp2952_path,
            config.exp2953_path,
            config.exp2963_path,
        )
    ]


def _prompt_for_mode(
    source: exp2952.RepairSource,
    mode: str,
    templates: Mapping[str, Any],
) -> str:
    if mode in {BASELINE_MODE, TAXONOMY_MODE}:
        return exp2952._repair_prompt(source, mode, templates)
    repair_prompt = exp2952._repair_prompt(source, TAXONOMY_MODE, templates)
    schema = exp2951.candidate_manifest_schema()
    return (
        "mode_id: dccd_structured\n"
        "DCCD structured repair: first form an unconstrained semantic draft, "
        "then apply the taxonomy repair focus, then emit exactly one JSON object "
        "matching the candidate manifest schema. Do not include markdown.\n"
        f"Candidate manifest schema:\n{json.dumps(schema, sort_keys=True)}\n"
        f"Repair context:\n{repair_prompt}\n"
    )


def _schema_failed_evaluation(
    *,
    source: exp2952.RepairSource,
    sample_index: int,
    seed: int,
    prompt: str,
    generation: GenerationOutcome,
    model_spec: Mapping[str, Any],
    raw_response_ref: str,
    schema_errors: list[str],
) -> JsonDict:
    return {
        "mode": DCCD_MODE,
        "task_id": source.task_key,
        "stable_id": source.stable_id,
        "corpus": source.corpus,
        "sample_id": source.sample_id,
        "sample_index": sample_index,
        "seed": seed,
        "model_hf_id": str(model_spec.get("hf_id") or ""),
        "model_path": str(model_spec.get("model_path") or ""),
        "prompt_sha256": _sha256_text(prompt),
        "raw_response_ref": raw_response_ref,
        "raw_response_sha256": _sha256_text(generation.text),
        "generation_backend": generation.backend,
        "generation_backend_detail": generation.backend_detail,
        "generation_duration_s": float(generation.duration_s),
        "tokens_generated": int(generation.tokens_generated),
        "generation_error": generation.error,
        "original_failure_categories": list(source.original_failure_categories),
        "parser_status": "not_run",
        "syntax_success": False,
        "static_checks": exp2952._syntax_static_checks(),
        "test_status": "not_run",
        "passed": False,
        "execution_error_type": "SchemaValidationError",
        "execution_error_message": "; ".join(schema_errors),
        "verifier_score": 0.0,
        "verifier_threshold": 1.0,
        "verifier_accepted": False,
        "false_accept": False,
        "schema_valid": False,
        "schema_errors": schema_errors,
        "candidate_manifest_sha256": exp2952._sha256_payload(
            {"raw_response_ref": raw_response_ref, "schema_errors": schema_errors}
        ),
    }


def _parse_json_object(text: str) -> tuple[Any | None, list[str]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    decoder = json.JSONDecoder()
    try:
        value, _end = decoder.raw_decode(stripped)
        return value, []
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            return None, ["no JSON object found"]
        try:
            value, _end = decoder.raw_decode(stripped[start:])
            return value, []
        except json.JSONDecodeError as exc:
            return None, [f"invalid JSON object: {exc.msg}"]


def _dccd_deltas(baseline: Mapping[str, Any], dccd: Mapping[str, Any]) -> JsonDict:
    return {
        "pass_at_1_delta": exp2952._delta(dccd.get("pass_at_1"), baseline.get("pass_at_1")),
        "pass_at_k_delta": exp2952._delta(dccd.get("pass_at_k"), baseline.get("pass_at_k")),
        "syntax_failure_rate_delta": exp2952._delta(
            dccd.get("syntax_failure_rate"),
            baseline.get("syntax_failure_rate"),
        ),
        "schema_failure_rate_delta": exp2952._delta(
            dccd.get("schema_failure_rate"),
            baseline.get("schema_failure_rate"),
        ),
        "false_accept_delta": exp2952._delta(
            dccd.get("false_accept_rate"),
            baseline.get("false_accept_rate"),
        ),
    }


def _dccd_replication_clean(n_tasks: int, deltas: Mapping[str, Any]) -> bool:
    false_accept_delta = deltas.get("false_accept_delta")
    return bool(
        n_tasks >= 20
        and isinstance(false_accept_delta, int | float)
        and false_accept_delta <= 0
        and (
            exp2952._positive(deltas.get("pass_at_1_delta"))
            or exp2952._negative(deltas.get("syntax_failure_rate_delta"))
        )
    )


def _candidate_seed(config: ExperimentConfig, mode: str, task_index: int, sample_index: int) -> int:
    mode_offset = {BASELINE_MODE: 0, TAXONOMY_MODE: 10_000, DCCD_MODE: 20_000}[mode]
    return config.random_seed + mode_offset + task_index * config.samples_per_mode + sample_index


def _nested_protocol_path(config: ExperimentConfig, exp2946: Mapping[str, Any]) -> Path:
    return Path(str(exp2946.get("protocol_artifact_path") or config.nested_exp2946_path))


def _verifier_threshold(exp2953_payload: Mapping[str, Any]) -> float:
    value = exp2953_payload.get("selected_default_threshold")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 1.0


def _false_accept_notes(deltas: Mapping[str, Any]) -> list[str]:
    delta = deltas.get("false_accept_delta")
    if isinstance(delta, int | float) and delta < 0:
        return ["false accepts decreased under DCCD structured repair"]
    if delta == 0:
        return ["false accepts unchanged under DCCD structured repair"]
    return ["false accepts increased or unavailable; DCCD promotion gate remains closed"]


def _complete_verdict(clean: bool, deltas: Mapping[str, Any]) -> str:
    status = (
        "complete: DCCD repair replication clean"
        if clean
        else "complete: DCCD repair replication did not clear promotion gates"
    )
    return (
        f"{status}; pass@1_delta={deltas.get('pass_at_1_delta')}, "
        f"syntax_failure_rate_delta={deltas.get('syntax_failure_rate_delta')}, "
        f"false_accept_delta={deltas.get('false_accept_delta')}"
    )


def _reproducibility_checksum(
    *,
    selected_task_ids: Sequence[str],
    candidate_manifest_sha256: str,
    model_specs: Sequence[Mapping[str, Any]],
    deltas: Mapping[str, Any],
) -> str:
    return exp2952._sha256_payload(
        {
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "deltas": dict(deltas),
            "model_specs": [dict(row) for row in model_specs],
            "selected_task_ids": list(selected_task_ids),
        }
    )


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


def main() -> int:  # pragma: no cover - script entrypoint.
    artifact = write_artifact(
        ExperimentConfig(
            tests_run=(
                ".venv/bin/pytest tests/python/test_experiment_2964_sota_dccd_repair_replication.py -q",
                ".venv/bin/pytest tests/python -q",
            )
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact["honest_verdict"].startswith("blocked_") else 1


if __name__ == "__main__":  # pragma: no cover - script entrypoint.
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "BASELINE_MODE",
    "DCCD_MODE",
    "EXP2946_REL_PATH",
    "EXP2950_REL_PATH",
    "EXP2951_REL_PATH",
    "EXP2952_REL_PATH",
    "EXP2953_REL_PATH",
    "EXP2963_REL_PATH",
    "ExecutionOutcome",
    "ExperimentConfig",
    "GenerationOutcome",
    "INFERENCE_SUBSTRATE",
    "NESTED_EXP2946_REL_PATH",
    "OUTPUT_FILENAME",
    "PreconditionReport",
    "REPO_ROOT",
    "REQUIRED_ARTIFACT_FIELDS",
    "TAXONOMY_MODE",
    "build_artifact",
    "evaluate_dccd_candidate",
    "write_artifact",
]
