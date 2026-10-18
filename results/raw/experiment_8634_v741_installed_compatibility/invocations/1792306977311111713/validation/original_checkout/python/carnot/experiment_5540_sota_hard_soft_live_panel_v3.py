"""Exp5540 compact local SOTA GGUF hard/soft live panel v3.

Spec refs: REQ-VERIFY-5540, SCENARIO-VERIFY-5540.

This module is intentionally a small orchestration layer. The row schema comes
from Exp5512, model-output extraction comes from Exp5513, and correctness comes
from the exact Exp5527 validator handoff. Exp5540 only checks the two required
preflight gates, runs or records local GGUF receipts, and refuses to promote
missing, repaired, schema-invalid, or exact-invalid rows into a SOTA claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5513_sota_hard_soft_structured_panel as panel5513
from carnot import experiment_5527_sota_hard_soft_panel_v2 as panel5527
from carnot import experiment_5538_sota_panel_duration_substrate_corrigendum as gate5538
from carnot import experiment_5539_gram2token_grammar_table_preflight as gate5539
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
PairResolver = Callable[[], Sequence[Mapping[str, Any]] | None]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json")
DURATION_GATE_RELATIVE_PATH = gate5538.RESULT_RELATIVE_PATH
GRAMMAR_GATE_RELATIVE_PATH = gate5539.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5540.sota_hard_soft_live_panel_v3.v502"
EXPERIMENT = 5540
EXPERIMENT_ID = "exp5540-sota-hard-soft-live-panel-v3"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5540
DURATION_FLOOR_S = 60.0
N_GPU_LAYERS = -1
DEFAULT_MAX_LIVE_MODELS = 2
DEFAULT_MAX_TOKENS = 1024
INFERENCE_SUBSTRATE = "live_local_sota_gguf_exact_validated_panel"
SPEC_REFS = ("REQ-VERIFY-5540", "SCENARIO-VERIFY-5540")

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "models_attempted",
    "rows_requested",
    "rows_emitted",
    "schema_validity_rate",
    "exact_validator_accuracy",
    "preference_optimality_rate",
    "missing_candidate_rows",
    "abstention_rate",
    "confident_wrong_rate",
    "prompt_hashes",
    "output_hashes",
    "random_seed",
    "measured_duration_s",
    "gpu_offload_evidence",
    "adversarial_clean",
    "sota_hard_soft_claim_allowed",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5540_sota_hard_soft_live_panel_v3.py",
    "tests/python/test_experiment_5538_sota_panel_duration_substrate_corrigendum.py",
    "tests/python/test_experiment_5539_gram2token_grammar_table_preflight.py",
    "tests/python/test_experiment_5527_sota_hard_soft_panel_v2.py",
    "tests/python/test_experiment_5513_sota_hard_soft_structured_panel.py",
    "tests/python/test_experiment_5512_structured_output_positive_control.py",
    "tests/python/test_experiment_5499_preference_maxsat_minimal_fixture_v499.py",
)

FIELD_PRINCIPLES: JsonDict = {
    "model_specs": "Names the only SOTA GGUF model IDs allowed for live panel rows.",
    "models_attempted": "Records which mandated local models actually received the prompt.",
    "rows_requested": "Fixes the exact fixture denominator before generation.",
    "rows_emitted": "Separates raw row volume from correctness.",
    "schema_validity_rate": "Keeps schema validation upstream of exact scoring.",
    "exact_validator_accuracy": "Uses deterministic validators as the correctness authority.",
    "preference_optimality_rate": "Measures soft preference only for feasible assignment rows.",
    "missing_candidate_rows": "Keeps absent expected rows visible.",
    "abstention_rate": "Counts only explicit schema-valid abstentions.",
    "confident_wrong_rate": "Separates high-confidence failures from abstention.",
    "prompt_hashes": "Pins prompts used for live generation.",
    "output_hashes": "Pins raw model outputs without relying on prose summaries.",
    "random_seed": "Records deterministic generation seed.",
    "measured_duration_s": "Records wall-clock runtime for live or gated-null evidence.",
    "gpu_offload_evidence": "Records offload evidence instead of trusting model names.",
    "adversarial_clean": "States whether the panel avoids known overclaim modes.",
    "sota_hard_soft_claim_allowed": "Controls whether the live hard/soft quality result may be cited.",
    "tests_added_or_reused": "Links the artifact to parser and validator tests.",
    "field_principles": "Explains why every headline and gate field exists.",
    "inference_substrate": "Declares the local GGUF exact-validated panel substrate.",
    "honest_verdict": "Provides a terminal, non-ambiguous evidence boundary.",
}

PRODUCTION_MODES = ("grammar_masking", "post_hoc_extraction", "repair")


def canonical_json(value: Any) -> str:
    """Serialize JSON in a stable form for hashes and reproducibility."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for a text receipt."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for a JSON-compatible value."""

    return sha256_text(canonical_json(value))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_live_prompt(target_rows: Sequence[Mapping[str, Any]] | None = None) -> str:
    """Build the compact JSON-only hard/soft row prompt for local GGUFs."""

    if target_rows is None:
        fixture = positive.load_fixture_artifact()["fixture"]
        target_rows = positive.build_fixture_candidate_payloads(fixture)
    return (
        "Return exactly one JSON object and no Markdown. The object must have "
        "candidate_rows, an array of rows matching the schema. Use abstain only "
        "when the exact hard constraints are infeasible. Do not omit requested "
        "instance_ids.\n\n"
        f"Row schema version: {positive.CANDIDATE_SCHEMA_VERSION}\n"
        f"JSON schema: {canonical_json(positive.candidate_schema())}\n"
        f"Requested rows: {canonical_json(list(target_rows))}\n"
        "Final answer shape: {\"candidate_rows\": [...]}\n"
    )


def build_artifact(
    *,
    duration_gate_path: Path = REPO_ROOT / DURATION_GATE_RELATIVE_PATH,
    grammar_gate_path: Path = REPO_ROOT / GRAMMAR_GATE_RELATIVE_PATH,
    duration_gate_artifact: Mapping[str, Any] | None = None,
    grammar_gate_artifact: Mapping[str, Any] | None = None,
    live_receipts: Sequence[Mapping[str, Any]] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    pair_resolver: PairResolver = cached_sota_pair,
    max_live_models: int = DEFAULT_MAX_LIVE_MODELS,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5540 artifact from clean gates and live or injected receipts."""

    started = time.perf_counter()
    duration_gate = dict(duration_gate_artifact) if duration_gate_artifact is not None else _load_gate_artifact(duration_gate_path)
    grammar_gate = dict(grammar_gate_artifact) if grammar_gate_artifact is not None else _load_gate_artifact(grammar_gate_path)
    gate_report = gate_status(duration_gate, grammar_gate)

    fixture = positive.load_fixture_artifact()["fixture"]
    target_rows = positive.build_fixture_candidate_payloads(fixture)
    requested_ids = [str(row["instance_id"]) for row in target_rows]
    prompt = build_live_prompt(target_rows)
    prompt_hash = sha256_text(prompt)
    specs = _model_specs(model_specs)

    receipts: list[JsonDict] = []
    if gate_report["gates_clean"] is True:
        if live_receipts is None:
            receipts = run_live_local_sota_panel(
                model_specs=specs,
                prompt=prompt,
                prompt_hash=prompt_hash,
                grammar_backend=str(grammar_gate.get("selected_backend", "")),
                pair_resolver=pair_resolver,
                max_live_models=max_live_models,
            )
        else:
            receipts = [dict(row) for row in live_receipts]

    normalized_receipts = [
        parse_live_receipt(row, default_prompt_hash=prompt_hash) for row in receipts
    ]
    per_model_reports = [
        _evaluate_receipt(receipt, fixture=fixture, requested_ids=requested_ids)
        for receipt in normalized_receipts
    ]
    metrics = _aggregate_reports(
        per_model_reports=per_model_reports,
        receipts=normalized_receipts,
        requested_ids=requested_ids,
    )
    gpu_evidence = _aggregate_gpu_offload_evidence(normalized_receipts)
    token_budgets = _token_budgets(normalized_receipts)
    models_attempted = _models_attempted_from_receipts(normalized_receipts)
    prompt_hashes = _unique_hashes(
        [str(row.get("prompt_hash") or prompt_hash) for row in normalized_receipts]
    )
    if gate_report["gates_clean"] is True and not prompt_hashes:
        prompt_hashes = [prompt_hash]
    output_hashes = [
        str(row["output_hash"])
        for row in normalized_receipts
        if row.get("output_hash") and row.get("raw_output")
    ]
    measured_duration = _measured_duration(normalized_receipts, started)
    blockers = _readiness_blockers(
        gates_clean=bool(gate_report["gates_clean"]),
        gate_blockers=gate_report["gate_blockers"],
        metrics=metrics,
        models_attempted=models_attempted,
        receipts=normalized_receipts,
        gpu_offload_evidence=gpu_evidence,
        measured_duration_s=measured_duration,
    )
    claim_allowed = not blockers

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "model_specs": specs,
        "models_attempted": models_attempted,
        "rows_requested": metrics["rows_requested"],
        "rows_emitted": metrics["rows_emitted"],
        "schema_validity_rate": metrics["schema_validity_rate"],
        "exact_validator_accuracy": metrics["exact_validator_accuracy"],
        "preference_optimality_rate": metrics["preference_optimality_rate"],
        "missing_candidate_rows": metrics["missing_candidate_rows"],
        "abstention_rate": metrics["abstention_rate"],
        "confident_wrong_rate": metrics["confident_wrong_rate"],
        "prompt_hashes": prompt_hashes,
        "output_hashes": output_hashes,
        "measured_duration_s": measured_duration,
        "gpu_offload_evidence": gpu_evidence,
        "adversarial_clean": True,
        "sota_hard_soft_claim_allowed": claim_allowed,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(
            claim_allowed=claim_allowed,
            gates_clean=bool(gate_report["gates_clean"]),
            blockers=blockers,
        ),
        "duration_gate_path": DURATION_GATE_RELATIVE_PATH.as_posix(),
        "grammar_gate_path": GRAMMAR_GATE_RELATIVE_PATH.as_posix(),
        "duration_gate_clean": gate_report["duration_gate_clean"],
        "grammar_gate_clean": gate_report["grammar_gate_clean"],
        "gates_clean": gate_report["gates_clean"],
        "gate_blockers": gate_report["gate_blockers"],
        "readiness_blockers": blockers,
        "duration_gate_honest_verdict": str(duration_gate.get("honest_verdict", "")),
        "grammar_gate_honest_verdict": str(grammar_gate.get("honest_verdict", "")),
        "prompt_builder_path": "carnot.experiment_5540_sota_hard_soft_live_panel_v3.build_live_prompt",
        "schema_parser_path": "carnot.experiment_5513_sota_hard_soft_structured_panel.extract_candidate_payloads",
        "schema_validator_path": "carnot.experiment_5512_structured_output_positive_control.classify_candidate_payload",
        "exact_validator_path": "carnot.experiment_5527_sota_hard_soft_panel_v2.evaluate_candidate_rows",
        "command_helper_path": "carnot.inference.sota_models.cached_sota_pair + llama_cpp.Llama.create_chat_completion",
        "token_budgets": token_budgets,
        "live_receipts": [_receipt_summary(row) for row in normalized_receipts],
        "per_model_reports": per_model_reports,
        "panel_rows": metrics["panel_rows"],
        "extra_emitted_rows": metrics["extra_emitted_rows"],
        "missing_instance_ids": metrics["missing_instance_ids"],
        "schema_valid_rows": metrics["schema_valid_rows"],
        "exact_validator_rows_scored": metrics["exact_validator_rows_scored"],
        "assignment_rows_scored": metrics["assignment_rows_scored"],
        "exact_correct_rows": metrics["exact_correct_rows"],
        "abstention_rows": metrics["abstention_rows"],
        "confident_wrong_rows": metrics["confident_wrong_rows"],
        "parse_failures": metrics["parse_failures"],
        "row_production_mode_counts": metrics["row_production_mode_counts"],
        "structured_repair_applied": metrics["row_production_mode_counts"]["repair"] > 0,
        "grammar_masking_rows": metrics["row_production_mode_counts"]["grammar_masking"],
        "post_hoc_extraction_rows": metrics["row_production_mode_counts"]["post_hoc_extraction"],
        "no_autotokenizer_on_gguf": True,
        "legacy_smoke_models_used": [],
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_gate_path: Path = REPO_ROOT / DURATION_GATE_RELATIVE_PATH,
    grammar_gate_path: Path = REPO_ROOT / GRAMMAR_GATE_RELATIVE_PATH,
    duration_gate_artifact: Mapping[str, Any] | None = None,
    grammar_gate_artifact: Mapping[str, Any] | None = None,
    live_receipts: Sequence[Mapping[str, Any]] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    pair_resolver: PairResolver = cached_sota_pair,
    max_live_models: int = DEFAULT_MAX_LIVE_MODELS,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5540 deliverable JSON."""

    artifact = build_artifact(
        duration_gate_path=duration_gate_path,
        grammar_gate_path=grammar_gate_path,
        duration_gate_artifact=duration_gate_artifact,
        grammar_gate_artifact=grammar_gate_artifact,
        live_receipts=live_receipts,
        model_specs=model_specs,
        pair_resolver=pair_resolver,
        max_live_models=max_live_models,
        tests_run=tests_run,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate Exp5540 fields and fail closed on hard/soft overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})), "field_principles")
    _require(_model_specs_match_mandated(artifact.get("model_specs", [])), "model_specs")
    _require(set(artifact.get("models_attempted", [])).issubset(positive.MANDATED_HEADLINE_MODEL_IDS), "models_attempted")
    _require(int(artifact.get("random_seed", -1)) == RANDOM_SEED, "random_seed")
    for field in (
        "schema_validity_rate",
        "exact_validator_accuracy",
        "preference_optimality_rate",
        "abstention_rate",
        "confident_wrong_rate",
    ):
        _require(0.0 <= float(artifact.get(field, -1.0)) <= 1.0, field)
    for field in ("rows_requested", "rows_emitted", "missing_candidate_rows"):
        _require(int(artifact.get(field, -1)) >= 0, field)
    _require(isinstance(artifact.get("prompt_hashes"), list), "prompt_hashes")
    _require(isinstance(artifact.get("output_hashes"), list), "output_hashes")
    _require(float(artifact.get("measured_duration_s", -1.0)) >= 0.0, "measured_duration_s")
    _require(isinstance(artifact.get("gpu_offload_evidence"), Mapping), "gpu_offload_evidence")
    _require(isinstance(artifact.get("adversarial_clean"), bool), "adversarial_clean")
    _require(isinstance(artifact.get("sota_hard_soft_claim_allowed"), bool), "sota_hard_soft_claim_allowed")
    _require(artifact.get("no_autotokenizer_on_gguf") is True, "no_autotokenizer_on_gguf")
    _require(artifact.get("legacy_smoke_models_used") == [], "legacy_smoke_models_used")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")

    if artifact.get("sota_hard_soft_claim_allowed") is True:
        _require(artifact.get("gates_clean") is True, "sota_hard_soft_claim_allowed")
        _require(artifact.get("readiness_blockers") == [], "sota_hard_soft_claim_allowed")
        _require(artifact.get("adversarial_clean") is True, "adversarial_clean")
        _require(bool(artifact.get("models_attempted")), "models_attempted")
        _require(float(artifact.get("measured_duration_s", 0.0)) >= DURATION_FLOOR_S, "measured_duration_s")
        _require(_gpu_offload_verified(artifact.get("gpu_offload_evidence", {})), "gpu_offload_evidence")
        _require(int(artifact.get("rows_requested", 0)) > 0, "rows_requested")
        _require(int(artifact.get("rows_emitted", 0)) >= int(artifact.get("rows_requested", 0)), "rows_emitted")
        _require(artifact.get("schema_validity_rate") == 1.0, "schema_validity_rate")
        _require(artifact.get("exact_validator_accuracy") == 1.0, "exact_validator_accuracy")
        _require(artifact.get("preference_optimality_rate") == 1.0, "preference_optimality_rate")
        _require(artifact.get("missing_candidate_rows") == 0, "missing_candidate_rows")
        _require(artifact.get("confident_wrong_rate") == 0.0, "confident_wrong_rate")
        _require(artifact.get("row_production_mode_counts", {}).get("repair") == 0, "row_production_mode_counts")
        _require(bool(artifact.get("prompt_hashes")), "prompt_hashes")
        _require(bool(artifact.get("output_hashes")), "output_hashes")
    else:
        _require(bool(artifact.get("readiness_blockers")), "sota_hard_soft_claim_allowed")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def gate_status(
    duration_gate: Mapping[str, Any],
    grammar_gate: Mapping[str, Any],
) -> JsonDict:
    """Return clean/blocked status for the two required upstream gates."""

    duration_clean = _duration_gate_clean(duration_gate)
    grammar_clean = _grammar_gate_clean(grammar_gate)
    blockers = []
    if not duration_clean:
        blockers.append("duration_substrate_gate_not_clean")
    if not grammar_clean:
        blockers.append("grammar_table_preflight_not_clean")
    return {
        "duration_gate_clean": duration_clean,
        "grammar_gate_clean": grammar_clean,
        "gates_clean": duration_clean and grammar_clean,
        "gate_blockers": blockers,
    }


def parse_live_receipt(
    receipt: Mapping[str, Any],
    *,
    default_prompt_hash: str,
) -> JsonDict:
    """Normalize a live run receipt before parsing and claim gating."""

    raw_output = str(receipt.get("raw_output", receipt.get("output_text", "")) or "")
    output_hash = sha256_text(raw_output) if raw_output else ""
    mode = _production_mode(receipt)
    attempted = _models_attempted(receipt.get("models_attempted", receipt.get("model_hf_id", [])))
    if not attempted and receipt.get("model_hf_id"):
        attempted = _models_attempted([receipt["model_hf_id"]])
    return {
        "live_model_invoked": bool(receipt.get("live_model_invoked")),
        "models_attempted": attempted,
        "model_hf_id": attempted[0] if attempted else str(receipt.get("model_hf_id", "")),
        "raw_output": raw_output,
        "output_hash": output_hash,
        "measured_duration_s": _safe_float(receipt.get("measured_duration_s", receipt.get("duration_s")), 0.0),
        "backend": str(receipt.get("backend", receipt.get("runtime_backend", "unavailable"))),
        "helper_path": str(receipt.get("helper_path", "carnot.inference.sota_models.cached_sota_pair")),
        "binding": receipt.get("binding", receipt.get("llama_cpp_binding")),
        "command": receipt.get("command", receipt.get("llama_cpp_command")),
        "random_seed": int(receipt.get("random_seed", RANDOM_SEED) or RANDOM_SEED),
        "prompt_hash": str(receipt.get("prompt_hash", receipt.get("prompt_sha256", default_prompt_hash)) or default_prompt_hash),
        "prompt_tokens": int(receipt.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(receipt.get("completion_tokens", 0) or 0),
        "max_tokens": int(receipt.get("max_tokens", DEFAULT_MAX_TOKENS) or DEFAULT_MAX_TOKENS),
        "n_ctx": int(receipt.get("n_ctx", 4096) or 4096),
        "n_batch": int(receipt.get("n_batch", 128) or 128),
        "n_gpu_layers": int(receipt.get("n_gpu_layers", N_GPU_LAYERS) or N_GPU_LAYERS),
        "production_mode": mode,
        "grammar_masking_used": mode == "grammar_masking",
        "gpu_offload_evidence": _normalize_gpu_offload_evidence(
            receipt.get("gpu_offload_evidence", receipt)
        ),
        "runtime_error": receipt.get("runtime_error"),
    }


def honest_verdict(
    *,
    claim_allowed: bool,
    gates_clean: bool,
    blockers: Sequence[str],
) -> str:
    """Return a terminal verdict that distinguishes claim, null, and gate block."""

    if claim_allowed:
        return "complete: sota_hard_soft_live_panel_v3_exact_validated_claim_allowed"
    suffix = "_".join(blockers) if blockers else "insufficient_live_evidence"
    if not gates_clean:
        return f"blocked: sota_hard_soft_live_panel_v3_gated_null_{suffix}"
    return f"complete: sota_hard_soft_live_panel_v3_honest_null_no_claim_{suffix}"


def run_live_local_sota_panel(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    prompt: str,
    prompt_hash: str,
    grammar_backend: str,
    pair_resolver: PairResolver,
    max_live_models: int,
) -> list[JsonDict]:  # pragma: no cover
    """Run bounded local GGUF calls for one or two mandated SOTA models."""

    selected = _select_live_model_specs(
        model_specs=model_specs,
        pair_resolver=pair_resolver,
        max_live_models=max_live_models,
    )
    if not selected:
        return []
    use_grammar = grammar_backend == "llama_cpp_gbnf"
    return [
        _run_one_live_model(spec=spec, prompt=prompt, prompt_hash=prompt_hash, use_grammar=use_grammar)
        for spec in selected
    ]


def _run_one_live_model(
    *,
    spec: Mapping[str, Any],
    prompt: str,
    prompt_hash: str,
    use_grammar: bool,
) -> JsonDict:  # pragma: no cover
    start = time.perf_counter()
    before = _gpu_memory_total_mb()
    attempted = [str(spec["hf_id"])]
    try:
        from llama_cpp import Llama, LlamaGrammar  # noqa: PLC0415

        llm = Llama(
            model_path=str(spec["model_path"]),
            n_ctx=4096,
            n_batch=128,
            n_gpu_layers=N_GPU_LAYERS,
            seed=RANDOM_SEED,
            verbose=False,
        )
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
        grammar = LlamaGrammar.from_string(positive.build_llama_cpp_json_grammar()) if use_grammar else None
        kwargs: JsonDict = {
            "messages": [
                {"role": "system", "content": "Return only the requested JSON object."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": RANDOM_SEED,
        }
        if grammar is not None:
            kwargs["grammar"] = grammar
        try:
            result = llm.create_chat_completion(**kwargs)
            production_mode = "grammar_masking" if grammar is not None else "post_hoc_extraction"
        except TypeError:
            kwargs.pop("grammar", None)
            result = llm.create_chat_completion(**kwargs)
            production_mode = "post_hoc_extraction"
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        message = choices[0].get("message", {}) if choices else {}
        raw_output = str(message.get("content", "")) if isinstance(message, Mapping) else ""
        completion_tokens = len(llm.tokenize(raw_output.encode("utf-8"))) if raw_output else 0
        after = _gpu_memory_total_mb()
        delta = max(0.0, after - before)
        return {
            "live_model_invoked": True,
            "models_attempted": attempted,
            "model_hf_id": attempted[0],
            "raw_output": raw_output,
            "measured_duration_s": round(time.perf_counter() - start, 6),
            "backend": "llama_cpp_python_cuda_gguf",
            "helper_path": "carnot.inference.sota_models.cached_sota_pair",
            "binding": "llama_cpp.Llama.create_chat_completion",
            "command": None,
            "random_seed": RANDOM_SEED,
            "prompt_hash": prompt_hash,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "n_ctx": 4096,
            "n_batch": 128,
            "n_gpu_layers": N_GPU_LAYERS,
            "production_mode": production_mode,
            "grammar_masking_used": production_mode == "grammar_masking",
            "gpu_offload_evidence": {
                "gpu_offload_verified": delta > 128.0,
                "offload_evidence": delta > 128.0,
                "gpu_memory_before_mb": before,
                "gpu_memory_after_mb": after,
                "gpu_memory_delta_mb": delta,
                "n_gpu_layers": N_GPU_LAYERS,
                "model_path": str(spec["model_path"]),
            },
            "runtime_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "live_model_invoked": False,
            "models_attempted": attempted,
            "model_hf_id": attempted[0],
            "raw_output": "",
            "measured_duration_s": round(time.perf_counter() - start, 6),
            "backend": "llama_cpp_python_cuda_gguf",
            "helper_path": "carnot.inference.sota_models.cached_sota_pair",
            "binding": "llama_cpp.Llama.create_chat_completion",
            "command": None,
            "random_seed": RANDOM_SEED,
            "prompt_hash": prompt_hash,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "n_ctx": 4096,
            "n_batch": 128,
            "n_gpu_layers": N_GPU_LAYERS,
            "production_mode": "post_hoc_extraction",
            "grammar_masking_used": False,
            "gpu_offload_evidence": {
                "gpu_offload_verified": False,
                "offload_evidence": False,
                "gpu_memory_delta_mb": 0.0,
                "n_gpu_layers": N_GPU_LAYERS,
            },
            "runtime_error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        llm = None  # type: ignore[possibly-used-before-assignment]
        gc.collect()


def _evaluate_receipt(
    receipt: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    requested_ids: Sequence[str],
) -> JsonDict:
    records, parse_failures = _candidate_records_from_receipt(receipt)
    report = panel5527.evaluate_candidate_rows(
        records,
        fixture=fixture,
        requested_instance_ids=requested_ids,
    )
    model_id = str(receipt.get("model_hf_id", ""))
    mode = str(receipt.get("production_mode", "post_hoc_extraction"))
    for row in report["panel_rows"]:
        row["model_hf_id"] = model_id
        row["production_mode"] = mode
        row["row_output_hash"] = receipt.get("output_hash", "")
    for row in report["extra_emitted_rows"]:
        row["model_hf_id"] = model_id
        row["production_mode"] = mode
        row["row_output_hash"] = receipt.get("output_hash", "")
    if receipt.get("runtime_error"):
        parse_failures.append({"parse_status": "runtime_error", "detail": str(receipt["runtime_error"])})
    return {
        "model_hf_id": model_id,
        "production_mode": mode,
        "rows_requested": report["rows_requested"],
        "rows_emitted": report["rows_emitted"],
        "schema_validity_rate": report["schema_validity_rate"],
        "missing_candidate_rows": report["missing_candidate_rows"],
        "exact_validator_accuracy": report["exact_validator_accuracy"],
        "preference_optimality_rate": report["preference_optimality_rate"],
        "abstention_rate": report["abstention_rate"],
        "confident_wrong_rate": report["confident_wrong_rate"],
        "panel_rows": report["panel_rows"],
        "extra_emitted_rows": report["extra_emitted_rows"],
        "missing_instance_ids": report["missing_instance_ids"],
        "schema_valid_rows": report["schema_valid_rows"],
        "exact_validator_rows_scored": report["exact_validator_rows_scored"],
        "assignment_rows_scored": report["assignment_rows_scored"],
        "confident_wrong_rows": report["confident_wrong_rows"],
        "parse_failures": parse_failures,
    }


def _candidate_records_from_receipt(receipt: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    if receipt.get("live_model_invoked") is not True:
        return [], []
    parsed = panel5513.extract_candidate_payloads(str(receipt.get("raw_output", "")))
    model_id = str(receipt.get("model_hf_id", ""))
    mode = str(receipt.get("production_mode", "post_hoc_extraction"))
    records = [
        {
            "parsed_payload": dict(payload),
            "model_hf_id": model_id,
            "production_mode": mode,
        }
        for payload in parsed.get("candidate_payloads", [])
        if isinstance(payload, Mapping)
    ]
    failures = [dict(row) for row in parsed.get("parse_failures", []) if isinstance(row, Mapping)]
    return records, failures


def _aggregate_reports(
    *,
    per_model_reports: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    requested_ids: Sequence[str],
) -> JsonDict:
    if per_model_reports:
        rows_requested = sum(int(row.get("rows_requested", 0)) for row in per_model_reports)
        missing_rows = sum(int(row.get("missing_candidate_rows", 0)) for row in per_model_reports)
    else:
        rows_requested = len(requested_ids)
        missing_rows = len(requested_ids)

    panel_rows = [dict(row) for report in per_model_reports for row in report.get("panel_rows", [])]
    extra_rows = [dict(row) for report in per_model_reports for row in report.get("extra_emitted_rows", [])]
    parse_failures = [dict(row) for report in per_model_reports for row in report.get("parse_failures", [])]
    missing_instance_ids = [
        {"model_hf_id": str(report.get("model_hf_id", "")), "instance_id": str(instance_id)}
        for report in per_model_reports
        for instance_id in report.get("missing_instance_ids", [])
    ]
    schema_valid_rows = [row for row in panel_rows if row.get("schema_valid") is True]
    assignment_rows = [
        row for row in schema_valid_rows if row.get("parse_status") == "schema_valid_assignment"
    ]
    abstention_rows = [
        row for row in schema_valid_rows if row.get("parse_status") == "schema_valid_abstention"
    ]
    exact_correct_rows = [
        row for row in schema_valid_rows if row.get("exact_validator_correct") is True
    ]
    soft_optimal_rows = [row for row in assignment_rows if row.get("soft_optimal") is True]
    confident_wrong_count = sum(int(row.get("confident_wrong_rows", 0)) for row in per_model_reports)
    production_counts = {mode: 0 for mode in PRODUCTION_MODES}
    for row in panel_rows:
        mode = str(row.get("production_mode", "post_hoc_extraction"))
        if mode in production_counts:
            production_counts[mode] += 1
    return {
        "rows_requested": rows_requested,
        "rows_emitted": sum(int(row.get("rows_emitted", 0)) for row in per_model_reports),
        "schema_validity_rate": _rate(len(schema_valid_rows), rows_requested),
        "exact_validator_accuracy": _rate(len(exact_correct_rows), len(schema_valid_rows)),
        "preference_optimality_rate": _rate(len(soft_optimal_rows), len(assignment_rows)),
        "missing_candidate_rows": missing_rows,
        "abstention_rate": _rate(len(abstention_rows), rows_requested),
        "confident_wrong_rate": _rate(confident_wrong_count, len(schema_valid_rows)),
        "panel_rows": panel_rows,
        "extra_emitted_rows": extra_rows,
        "missing_instance_ids": missing_instance_ids,
        "schema_valid_rows": len(schema_valid_rows),
        "exact_validator_rows_scored": len(schema_valid_rows),
        "assignment_rows_scored": len(assignment_rows),
        "exact_correct_rows": len(exact_correct_rows),
        "abstention_rows": len(abstention_rows),
        "confident_wrong_rows": confident_wrong_count,
        "parse_failures": parse_failures,
        "row_production_mode_counts": production_counts,
        "receipts_seen": len(receipts),
    }


def _readiness_blockers(
    *,
    gates_clean: bool,
    gate_blockers: Sequence[str],
    metrics: Mapping[str, Any],
    models_attempted: Sequence[str],
    receipts: Sequence[Mapping[str, Any]],
    gpu_offload_evidence: Mapping[str, Any],
    measured_duration_s: float,
) -> list[str]:
    blockers = list(gate_blockers)
    if not gates_clean:
        return sorted(set(blockers))
    if not receipts:
        blockers.append("no_live_model_invoked")
    if not models_attempted:
        blockers.append("no_mandated_sota_model_attempted")
    if any(row.get("live_model_invoked") is not True for row in receipts):
        blockers.append("live_model_runtime_failed")
    if measured_duration_s < DURATION_FLOOR_S:
        blockers.append("duration_below_live_claim_floor")
    if not _gpu_offload_verified(gpu_offload_evidence):
        blockers.append("gpu_offload_evidence_absent_or_false")
    if int(metrics.get("rows_requested", 0)) <= 0:
        blockers.append("no_rows_requested")
    if int(metrics.get("rows_emitted", 0)) <= 0:
        blockers.append("no_rows_emitted")
    if int(metrics.get("missing_candidate_rows", 0)) > 0:
        blockers.append("missing_candidate_rows")
    if float(metrics.get("schema_validity_rate", 0.0)) < 1.0:
        blockers.append("schema_invalid_or_missing_rows")
    if float(metrics.get("exact_validator_accuracy", 0.0)) < 1.0:
        blockers.append("exact_validator_mismatch")
    if float(metrics.get("preference_optimality_rate", 0.0)) < 1.0:
        blockers.append("preference_suboptimal_or_unscored")
    if float(metrics.get("confident_wrong_rate", 0.0)) > 0.0:
        blockers.append("confident_wrong_rows")
    if metrics.get("parse_failures"):
        blockers.append("parse_failures")
    if metrics.get("row_production_mode_counts", {}).get("repair", 0) > 0:
        blockers.append("repair_rows_not_headline_eligible")
    return sorted(set(blockers))


def _load_gate_artifact(path: Path) -> JsonDict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"load_error": f"{type(exc).__name__}: {exc}"}


def _duration_gate_clean(artifact: Mapping[str, Any]) -> bool:
    return bool(
        artifact.get("sota_panel_duration_corrigendum_ready") is True
        and artifact.get("adversarial_clean") is True
        and artifact.get("inference_substrate") == gate5538.INFERENCE_SUBSTRATE
        and artifact.get("research_conductor_modified") is not True
        and "load_error" not in artifact
    )


def _grammar_gate_clean(artifact: Mapping[str, Any]) -> bool:
    return bool(
        artifact.get("grammar_table_preflight_ready") is True
        and artifact.get("backend_available") is True
        and artifact.get("llm_invoked") is False
        and artifact.get("decoding_speedup_claim") is False
        and artifact.get("inference_substrate") == gate5539.INFERENCE_SUBSTRATE
        and artifact.get("research_conductor_modified") is not True
        and "load_error" not in artifact
    )


def _model_specs(rows: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if rows is None:
        return positive.resolve_model_specs()
    return [dict(row) for row in rows]


def _model_specs_match_mandated(rows: Any) -> bool:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    ids = [str(row.get("hf_id")) for row in rows if isinstance(row, Mapping)]
    return ids == list(positive.MANDATED_HEADLINE_MODEL_IDS)


def _select_live_model_specs(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    pair_resolver: PairResolver,
    max_live_models: int,
) -> list[JsonDict]:  # pragma: no cover
    by_id = {str(row.get("hf_id")): dict(row) for row in model_specs}
    selected: list[JsonDict] = []
    selected_ids: set[str] = set()
    for pair_row in pair_resolver() or []:
        spec = by_id.get(str(pair_row.get("hf_id")))
        if spec and spec.get("local_model_present") is True:
            selected.append(spec)
            selected_ids.add(str(spec["hf_id"]))
    for spec in model_specs:
        hf_id = str(spec.get("hf_id"))
        if spec.get("local_model_present") is True and hf_id not in selected_ids:
            selected.append(dict(spec))
            selected_ids.add(hf_id)
    return selected[: max(0, int(max_live_models))]


def _models_attempted_from_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    attempted: list[str] = []
    for receipt in receipts:
        for hf_id in _models_attempted(receipt.get("models_attempted", [])):
            if hf_id not in attempted:
                attempted.append(hf_id)
    return attempted


def _models_attempted(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        return []
    attempted = []
    for value in values:
        hf_id = str(value)
        if hf_id in positive.MANDATED_HEADLINE_MODEL_IDS and hf_id not in attempted:
            attempted.append(hf_id)
    return attempted


def _aggregate_gpu_offload_evidence(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_model = []
    for receipt in receipts:
        evidence = dict(receipt.get("gpu_offload_evidence", {}))
        evidence["model_hf_id"] = receipt.get("model_hf_id", "")
        per_model.append(evidence)
    deltas = [_safe_float(row.get("gpu_memory_delta_mb"), 0.0) for row in per_model]
    verified = any(_gpu_offload_verified(row) for row in per_model)
    return {
        "gpu_offload_verified": verified,
        "offload_evidence": verified,
        "gpu_memory_delta_mb": round(max(deltas) if deltas else 0.0, 6),
        "n_gpu_layers": N_GPU_LAYERS,
        "per_model": per_model,
    }


def _normalize_gpu_offload_evidence(value: Any) -> JsonDict:
    evidence = dict(value) if isinstance(value, Mapping) else {}
    verified = bool(
        evidence.get("gpu_offload_verified")
        or evidence.get("offload_evidence")
        or evidence.get("load_offload_evidence")
    )
    evidence["gpu_offload_verified"] = verified
    evidence["offload_evidence"] = verified
    evidence["gpu_memory_delta_mb"] = _safe_float(evidence.get("gpu_memory_delta_mb"), 0.0)
    evidence["n_gpu_layers"] = int(evidence.get("n_gpu_layers", N_GPU_LAYERS) or N_GPU_LAYERS)
    return evidence


def _gpu_offload_verified(evidence: Any) -> bool:
    return isinstance(evidence, Mapping) and bool(
        evidence.get("gpu_offload_verified") or evidence.get("offload_evidence")
    )


def _token_budgets(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "model_hf_id": str(row.get("model_hf_id", "")),
            "prompt_tokens": int(row.get("prompt_tokens", 0)),
            "completion_tokens": int(row.get("completion_tokens", 0)),
            "max_tokens": int(row.get("max_tokens", DEFAULT_MAX_TOKENS)),
            "n_ctx": int(row.get("n_ctx", 4096)),
            "n_batch": int(row.get("n_batch", 128)),
        }
        for row in receipts
    ]


def _receipt_summary(receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "live_model_invoked": receipt.get("live_model_invoked"),
        "models_attempted": list(receipt.get("models_attempted", [])),
        "model_hf_id": receipt.get("model_hf_id"),
        "output_hash": receipt.get("output_hash"),
        "measured_duration_s": receipt.get("measured_duration_s"),
        "backend": receipt.get("backend"),
        "helper_path": receipt.get("helper_path"),
        "binding": receipt.get("binding"),
        "command": receipt.get("command"),
        "random_seed": receipt.get("random_seed"),
        "prompt_hash": receipt.get("prompt_hash"),
        "production_mode": receipt.get("production_mode"),
        "grammar_masking_used": receipt.get("grammar_masking_used"),
        "runtime_error": receipt.get("runtime_error"),
    }


def _unique_hashes(values: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _measured_duration(receipts: Sequence[Mapping[str, Any]], started: float) -> float:
    if receipts:
        return round(sum(_safe_float(row.get("measured_duration_s"), 0.0) for row in receipts), 6)
    return round(time.perf_counter() - started, 6)


def _production_mode(receipt: Mapping[str, Any]) -> str:
    mode = str(receipt.get("production_mode", "") or "")
    if mode in PRODUCTION_MODES:
        return mode
    if receipt.get("grammar_masking_used") is True:
        return "grammar_masking"
    return "post_hoc_extraction"


def _rate(numerator: int | float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gpu_memory_total_mb() -> float:  # pragma: no cover
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return 0.0
    return sum(float(line.strip()) for line in result.stdout.splitlines() if line.strip())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
                "sota_hard_soft_claim_allowed": artifact["sota_hard_soft_claim_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
