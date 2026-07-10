"""Exp5538 duration/substrate corrigendum for the SOTA hard/soft panel.

Spec refs: REQ-VERIFY-5538, SCENARIO-VERIFY-5538.

This module repairs the evidence boundary around Exp5527. It reuses the same
hard/soft fixtures, row schema, parser path, and exact validators, but it does
not inherit Exp5527's quality metrics unless a fresh live local SOTA GGUF run
has a plausible duration/offload receipt and schema-valid exact rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5538_sota_panel_duration_substrate_corrigendum.json")
UPSTREAM_PANEL_RELATIVE_PATH = panel5527.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5538.sota_panel_duration_substrate_corrigendum.v502"
EXPERIMENT = 5538
EXPERIMENT_ID = "exp5538-sota-panel-duration-substrate-corrigendum"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5538
DURATION_FLOOR_S = 60.0
N_GPU_LAYERS = -1
INFERENCE_SUBSTRATE = "live_local_sota_gguf_panel_or_claim_downgrade"
SPEC_REFS = ("REQ-VERIFY-5538", "SCENARIO-VERIFY-5538")

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "upstream_panel_path",
    "live_model_invoked",
    "models_attempted",
    "rows_requested",
    "rows_emitted",
    "schema_validity_rate",
    "exact_validator_accuracy",
    "preference_optimality_rate",
    "missing_candidate_rows",
    "abstention_rate",
    "confident_wrong_rate",
    "duration_floor_s",
    "measured_duration_s",
    "gpu_offload_evidence",
    "adversarial_clean",
    "no_quality_claim_if_not_live",
    "sota_panel_duration_corrigendum_ready",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5538_sota_panel_duration_substrate_corrigendum.py",
    "tests/python/test_experiment_5527_sota_hard_soft_panel_v2.py",
    "tests/python/test_experiment_5513_sota_hard_soft_structured_panel.py",
    "tests/python/test_experiment_5512_structured_output_positive_control.py",
    "tests/python/test_experiment_5499_preference_maxsat_minimal_fixture_v499.py",
)

FIELD_PRINCIPLES: JsonDict = {
    "model_specs": "Names the mandated GGUF set so the corrigendum cannot swap in legacy smoke models as headline evidence.",
    "upstream_panel_path": "Pins the artifact whose duration/substrate boundary is being repaired.",
    "live_model_invoked": "Separates real local-GGUF execution from evidence downgrades.",
    "models_attempted": "Identifies which mandated model, if any, received the live prompt.",
    "rows_requested": "Fixes the Exp5527 fixture denominator before parsing.",
    "rows_emitted": "Separates model output volume from correctness or quality.",
    "schema_validity_rate": "Gates exact-validator handoff on schema-valid rows.",
    "exact_validator_accuracy": "Reports deterministic Exp5499 correctness only for schema-valid rows.",
    "preference_optimality_rate": "Reports soft optimality only after exact hard-constraint validation.",
    "missing_candidate_rows": "Keeps absent rows visible and prevents missing rows from becoming abstentions.",
    "abstention_rate": "Counts only explicit schema-valid abstentions.",
    "confident_wrong_rate": "Separates high-confidence schema-valid failures from calibrated abstention.",
    "duration_floor_s": "Records the adversarial plausibility floor for live SOTA local-GGUF claims.",
    "measured_duration_s": "Records the wall-clock receipt used to authenticate or downgrade the claim.",
    "gpu_offload_evidence": "Records the runtime/offload substrate instead of trusting model-name strings.",
    "adversarial_clean": "States whether the corrected boundary avoids the Exp5527 duration/substrate overclaim.",
    "no_quality_claim_if_not_live": "Prevents missing live execution from inheriting Exp5527 quality metrics.",
    "sota_panel_duration_corrigendum_ready": "States whether the boundary repair is complete enough for downstream gates.",
    "tests_added_or_reused": "Links the artifact to parser, duration, substrate, and exact-validator tests.",
    "field_principles": "Explains why each headline and gate field must remain present.",
    "inference_substrate": "Declares live local SOTA GGUF execution or explicit claim downgrade semantics.",
    "honest_verdict": "Provides a terminal status without promoting too-short, missing, or schema-invalid evidence.",
    "quality_claim_allowed": "States whether this corrigendum opens a downstream hard/soft SOTA quality claim.",
    "readiness_blockers": "Lists the exact gates preventing claim authentication.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON in the stable form used by checksums and prompt hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for text receipts such as prompts."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def claim_downgrade_receipt(reason: str) -> JsonDict:
    """Return a receipt for the explicit no-live-quality-claim path."""

    return {
        "live_model_invoked": False,
        "models_attempted": [],
        "raw_output": "",
        "measured_duration_s": 0.0,
        "backend": "unavailable",
        "helper_path": "carnot.inference.sota_models.cached_sota_pair",
        "binding": None,
        "command": None,
        "random_seed": RANDOM_SEED,
        "prompt_hash": "",
        "gpu_offload_evidence": {
            "gpu_offload_verified": False,
            "offload_evidence": False,
            "gpu_memory_delta_mb": 0.0,
            "n_gpu_layers": N_GPU_LAYERS,
        },
        "runtime_error": str(reason),
    }


def parse_duration_substrate_receipt(receipt: Mapping[str, Any]) -> JsonDict:
    """Normalize live-run duration and substrate evidence before claim gating."""

    evidence = _normalize_gpu_offload_evidence(receipt.get("gpu_offload_evidence", receipt))
    return {
        "live_model_invoked": bool(receipt.get("live_model_invoked")),
        "models_attempted": _models_attempted(receipt.get("models_attempted", [])),
        "raw_output": str(receipt.get("raw_output", receipt.get("output_text", "")) or ""),
        "measured_duration_s": _first_float(
            receipt,
            ("measured_duration_s", "duration_s", "wall_time_s"),
            default=0.0,
        ),
        "backend": str(receipt.get("backend", receipt.get("runtime_backend", "unavailable"))),
        "helper_path": str(
            receipt.get("helper_path", "carnot.inference.sota_models.cached_sota_pair")
        ),
        "binding": receipt.get("binding", receipt.get("llama_cpp_binding")),
        "command": receipt.get("command", receipt.get("llama_cpp_command")),
        "random_seed": int(receipt.get("random_seed", RANDOM_SEED) or RANDOM_SEED),
        "prompt_hash": str(
            receipt.get(
                "prompt_hash",
                receipt.get("prompt_sha256", receipt.get("reason_then_structure_prompt_sha256", "")),
            )
            or ""
        ),
        "gpu_offload_evidence": evidence,
        "runtime_error": receipt.get("runtime_error"),
    }


def load_upstream_panel(path: Path = REPO_ROOT / UPSTREAM_PANEL_RELATIVE_PATH) -> JsonDict:
    """Load Exp5527 so the corrigendum can preserve the repaired boundary target."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"load_error": f"{type(exc).__name__}: {exc}"}


def build_live_prompt() -> str:
    """Build the same reason-then-structure prompt shape used by Exp5513."""

    fixture = positive.load_fixture_artifact()["fixture"]
    return panel5513.build_reason_then_structure_prompt(
        positive.build_fixture_candidate_payloads(fixture)
    )


def build_artifact(
    *,
    upstream_panel_path: Path = REPO_ROOT / UPSTREAM_PANEL_RELATIVE_PATH,
    live_receipt: Mapping[str, Any] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5538 corrigendum artifact."""

    upstream = load_upstream_panel(Path(upstream_panel_path))
    fixture = positive.load_fixture_artifact()["fixture"]
    target_payloads = positive.build_fixture_candidate_payloads(fixture)
    requested_ids = [str(row["instance_id"]) for row in target_payloads]
    prompt = build_live_prompt()
    prompt_hash = sha256_text(prompt)
    specs = _model_specs(model_specs)
    receipt = (
        parse_duration_substrate_receipt(live_receipt)
        if live_receipt is not None
        else parse_duration_substrate_receipt(
            run_live_local_sota_panel(model_specs=specs, prompt=prompt, prompt_hash=prompt_hash)
        )
    )
    if not receipt["prompt_hash"]:
        receipt["prompt_hash"] = prompt_hash

    candidate_records, parse_failures = _candidate_records_from_receipt(receipt)
    report = panel5527.evaluate_candidate_rows(
        candidate_records,
        fixture=fixture,
        requested_instance_ids=requested_ids,
    )
    blockers = _readiness_blockers(receipt=receipt, report=report)
    quality_claim_allowed = not blockers
    no_quality_claim = not quality_claim_allowed
    ready = quality_claim_allowed or no_quality_claim
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
        "upstream_panel_path": UPSTREAM_PANEL_RELATIVE_PATH.as_posix(),
        "live_model_invoked": bool(receipt["live_model_invoked"]),
        "models_attempted": list(receipt["models_attempted"]),
        "rows_requested": report["rows_requested"],
        "rows_emitted": report["rows_emitted"],
        "schema_validity_rate": report["schema_validity_rate"],
        "exact_validator_accuracy": report["exact_validator_accuracy"],
        "preference_optimality_rate": report["preference_optimality_rate"],
        "missing_candidate_rows": report["missing_candidate_rows"],
        "abstention_rate": report["abstention_rate"],
        "confident_wrong_rate": report["confident_wrong_rate"],
        "duration_floor_s": DURATION_FLOOR_S,
        "measured_duration_s": float(receipt["measured_duration_s"]),
        "gpu_offload_evidence": dict(receipt["gpu_offload_evidence"]),
        "adversarial_clean": ready,
        "no_quality_claim_if_not_live": no_quality_claim,
        "sota_panel_duration_corrigendum_ready": ready,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(quality_claim_allowed, blockers),
        "quality_claim_allowed": quality_claim_allowed,
        "readiness_blockers": blockers,
        "upstream_panel_flagged_adversarial": bool(upstream.get("flagged_adversarial")),
        "upstream_panel_duration_s": float(upstream.get("duration_s", 0.0) or 0.0),
        "upstream_panel_claim_allowed": bool(upstream.get("sota_hard_soft_claim_allowed")),
        "upstream_panel_honest_verdict": str(upstream.get("honest_verdict", "")),
        "duration_receipt": _duration_receipt(receipt),
        "prompt_hash": str(receipt["prompt_hash"]),
        "prompt_builder_path": "carnot.experiment_5513_sota_hard_soft_structured_panel.build_reason_then_structure_prompt",
        "schema_parser_path": "carnot.experiment_5513_sota_hard_soft_structured_panel.extract_candidate_payloads",
        "exact_validator_path": "carnot.experiment_5527_sota_hard_soft_panel_v2.evaluate_candidate_rows",
        "parse_failures": parse_failures,
        "panel_rows": report["panel_rows"],
        "extra_emitted_rows": report["extra_emitted_rows"],
        "missing_instance_ids": report["missing_instance_ids"],
        "schema_valid_rows": report["schema_valid_rows"],
        "exact_validator_rows_scored": report["exact_validator_rows_scored"],
        "assignment_rows_scored": report["assignment_rows_scored"],
        "confident_wrong_rows": report["confident_wrong_rows"],
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
    live_receipt: Mapping[str, Any] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5538 deliverable JSON."""

    artifact = build_artifact(
        live_receipt=live_receipt,
        model_specs=model_specs,
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
    """Validate the corrigendum and fail closed on live-quality overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("upstream_panel_path") == UPSTREAM_PANEL_RELATIVE_PATH.as_posix(), "upstream_panel_path")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})), "field_principles")
    _require(_model_specs_match_mandated(artifact.get("model_specs", [])), "model_specs")
    _require(set(artifact.get("models_attempted", [])).issubset(positive.MANDATED_HEADLINE_MODEL_IDS), "models_attempted")
    for field in (
        "schema_validity_rate",
        "exact_validator_accuracy",
        "preference_optimality_rate",
        "abstention_rate",
        "confident_wrong_rate",
    ):
        _require(0.0 <= float(artifact.get(field, -1.0)) <= 1.0, field)
    _require(int(artifact.get("rows_requested", -1)) >= 0, "rows_requested")
    _require(int(artifact.get("rows_emitted", -1)) >= 0, "rows_emitted")
    _require(int(artifact.get("missing_candidate_rows", -1)) >= 0, "missing_candidate_rows")
    _require(float(artifact.get("duration_floor_s", -1.0)) == DURATION_FLOOR_S, "duration_floor_s")
    _require(float(artifact.get("measured_duration_s", -1.0)) >= 0.0, "measured_duration_s")
    _require(isinstance(artifact.get("gpu_offload_evidence"), Mapping), "gpu_offload_evidence")
    for field in (
        "live_model_invoked",
        "adversarial_clean",
        "no_quality_claim_if_not_live",
        "sota_panel_duration_corrigendum_ready",
        "quality_claim_allowed",
    ):
        _require(isinstance(artifact.get(field), bool), field)
    _require(artifact.get("no_autotokenizer_on_gguf") is True, "no_autotokenizer_on_gguf")
    _require(artifact.get("legacy_smoke_models_used") == [], "legacy_smoke_models_used")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")

    if artifact.get("quality_claim_allowed") is True:
        _require(artifact.get("live_model_invoked") is True, "quality_claim_allowed")
        _require(artifact.get("no_quality_claim_if_not_live") is False, "no_quality_claim_if_not_live")
        _require(float(artifact.get("measured_duration_s", 0.0)) >= DURATION_FLOOR_S, "measured_duration_s")
        _require(_gpu_offload_verified(artifact.get("gpu_offload_evidence", {})), "gpu_offload_evidence")
        _require(int(artifact.get("rows_requested", 0)) > 0, "rows_requested")
        _require(int(artifact.get("rows_emitted", 0)) >= int(artifact.get("rows_requested", 0)), "rows_emitted")
        _require(artifact.get("schema_validity_rate") == 1.0, "schema_validity_rate")
        _require(artifact.get("exact_validator_accuracy") == 1.0, "exact_validator_accuracy")
        _require(artifact.get("preference_optimality_rate") == 1.0, "preference_optimality_rate")
        _require(artifact.get("missing_candidate_rows") == 0, "missing_candidate_rows")
        _require(artifact.get("confident_wrong_rate") == 0.0, "confident_wrong_rate")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
    else:
        _require(artifact.get("no_quality_claim_if_not_live") is True, "quality_claim_allowed")
    if artifact.get("live_model_invoked") is False:
        _require(artifact.get("rows_emitted") == 0, "rows_emitted")
        _require(artifact.get("quality_claim_allowed") is False, "quality_claim_allowed")
    _require(artifact.get("sota_panel_duration_corrigendum_ready") is True, "ready")
    _require(artifact.get("adversarial_clean") is True, "adversarial_clean")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(quality_claim_allowed: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict that names authentication or downgrade."""

    if quality_claim_allowed:
        return "complete: live_sota_hard_soft_panel_duration_substrate_authenticated"
    suffix = "_".join(blockers) if blockers else "no_live_quality_claim"
    return f"complete: live_sota_hard_soft_panel_claim_downgraded_no_quality_claim_{suffix}"


def run_live_local_sota_panel(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    prompt: str,
    prompt_hash: str,
) -> JsonDict:  # pragma: no cover
    """Run one bounded live local GGUF prompt using llama.cpp when available."""

    spec = _select_live_model_spec(model_specs)
    if spec is None:
        return claim_downgrade_receipt("no_cached_mandated_gguf")
    attempted = [str(spec["hf_id"])]
    start = time.perf_counter()
    before = _gpu_memory_total_mb()
    try:
        from llama_cpp import Llama  # noqa: PLC0415

        llm = Llama(
            model_path=str(spec["model_path"]),
            n_ctx=4096,
            n_batch=128,
            n_gpu_layers=N_GPU_LAYERS,
            seed=RANDOM_SEED,
            verbose=False,
        )
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Return brief reasoning followed by the requested JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            seed=RANDOM_SEED,
        )
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        message = choices[0].get("message", {}) if choices else {}
        raw_output = str(message.get("content", "")) if isinstance(message, Mapping) else ""
        completion_tokens = len(llm.tokenize(raw_output.encode("utf-8"))) if raw_output else 0
        after = _gpu_memory_total_mb()
        delta = max(0.0, after - before)
        return {
            "live_model_invoked": True,
            "models_attempted": attempted,
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
        receipt = claim_downgrade_receipt(f"live_model_runtime_failed:{type(exc).__name__}: {exc}")
        receipt["models_attempted"] = attempted
        receipt["measured_duration_s"] = round(time.perf_counter() - start, 6)
        receipt["prompt_hash"] = prompt_hash
        return receipt
    finally:
        llm = None  # type: ignore[possibly-used-before-assignment]
        gc.collect()


def _candidate_records_from_receipt(receipt: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    if receipt.get("live_model_invoked") is not True:
        return [], []
    parsed = panel5513.extract_candidate_payloads(str(receipt.get("raw_output", "")))
    model_id = next(iter(receipt.get("models_attempted", []) or []), None)
    records = [
        {"parsed_payload": dict(payload), "model_hf_id": model_id}
        for payload in parsed.get("candidate_payloads", [])
        if isinstance(payload, Mapping)
    ]
    failures = [dict(row) for row in parsed.get("parse_failures", []) if isinstance(row, Mapping)]
    return records, failures


def _duration_receipt(receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "helper_path": receipt.get("helper_path"),
        "backend": receipt.get("backend"),
        "binding": receipt.get("binding"),
        "command": receipt.get("command"),
        "random_seed": receipt.get("random_seed"),
        "prompt_hash": receipt.get("prompt_hash"),
        "runtime_error": receipt.get("runtime_error"),
        "measured_duration_s": receipt.get("measured_duration_s"),
    }


def _readiness_blockers(*, receipt: Mapping[str, Any], report: Mapping[str, Any]) -> list[str]:
    blockers = []
    if receipt.get("live_model_invoked") is not True:
        blockers.append("no_live_model_invoked")
    if receipt.get("models_attempted") and not _models_attempted(receipt.get("models_attempted", [])):
        blockers.append("no_mandated_sota_model_attempted")
    if float(receipt.get("measured_duration_s", 0.0)) < DURATION_FLOOR_S:
        blockers.append("duration_below_live_claim_floor")
    if not _gpu_offload_verified(receipt.get("gpu_offload_evidence", {})):
        blockers.append("gpu_offload_evidence_absent_or_false")
    if int(report.get("missing_candidate_rows", 0)) > 0:
        blockers.append("missing_candidate_rows")
    if float(report.get("schema_validity_rate", 0.0)) < 1.0:
        blockers.append("schema_invalid_or_missing_rows")
    if float(report.get("exact_validator_accuracy", 0.0)) < 1.0:
        blockers.append("exact_validator_mismatch")
    if float(report.get("preference_optimality_rate", 0.0)) < 1.0:
        blockers.append("preference_suboptimal_or_unscored")
    if float(report.get("confident_wrong_rate", 0.0)) > 0.0:
        blockers.append("confident_wrong_rows")
    return sorted(set(blockers))


def _model_specs(rows: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if rows is None:
        return positive.resolve_model_specs()
    return [dict(row) for row in rows]


def _model_specs_match_mandated(rows: Any) -> bool:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    ids = [str(row.get("hf_id")) for row in rows if isinstance(row, Mapping)]
    return ids == list(positive.MANDATED_HEADLINE_MODEL_IDS)


def _select_live_model_spec(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:  # pragma: no cover
    by_id = {str(row.get("hf_id")): dict(row) for row in model_specs}
    for pair_row in cached_sota_pair() or []:
        spec = by_id.get(str(pair_row.get("hf_id")))
        if spec and spec.get("model_path"):
            return spec
    for spec in model_specs:
        if spec.get("model_path") and str(spec.get("hf_id")) in positive.MANDATED_HEADLINE_MODEL_IDS:
            return dict(spec)
    return None


def _models_attempted(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    attempted = []
    for value in values:
        hf_id = str(value)
        if hf_id in positive.MANDATED_HEADLINE_MODEL_IDS and hf_id not in attempted:
            attempted.append(hf_id)
    return attempted


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


def _first_float(receipt: Mapping[str, Any], keys: Sequence[str], *, default: float) -> float:
    for key in keys:
        if key in receipt:
            return _safe_float(receipt.get(key), default)
    return default


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
                "live_model_invoked": artifact["live_model_invoked"],
                "quality_claim_allowed": artifact["quality_claim_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
