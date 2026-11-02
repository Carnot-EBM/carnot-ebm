"""Exp5525 taxonomy for live SOTA structured-row schema failures.

Spec refs: REQ-VERIFY-5525, SCENARIO-VERIFY-5525.

This module diagnoses the interface between the local GGUF completion path and
the existing hard/soft structured-row validator. It deliberately reuses the
Exp5512 schema, Exp5513 extractor, and Exp5499 exact validators so the artifact
separates parser health from live model-quality claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod
from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5513_sota_hard_soft_structured_panel as panel


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5525_sota_schema_failure_taxonomy.json")
LIVE_ARTIFACT_RELATIVE_PATH = panel.RESULT_RELATIVE_PATH
POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH = positive.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5525.sota_schema_failure_taxonomy.v501"
EXPERIMENT = 5525
EXPERIMENT_ID = "exp5525-sota-schema-failure-taxonomy"
MILESTONE = "2026.07.501"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5525
INFERENCE_SUBSTRATE = "structured_output_fixture_plus_live_llm_smoke"
SPEC_REFS = ("REQ-VERIFY-5525", "SCENARIO-VERIFY-5525")
PARSER_BACKEND = (
    "experiment_5513.extract_candidate_payloads -> "
    "experiment_5512.classify_candidate_payload"
)
DEFAULT_LIVE_MAX_TOKENS = 2048

FAILURE_CATEGORIES = (
    "prompt_contract_miss",
    "grammar_runtime_unavailable",
    "grammar_mask_not_applied",
    "max_tokens_truncation",
    "json_extraction_failure",
    "json_schema_invalid",
    "required_field_missing",
    "exact_validator_mismatch",
    "semantic_candidate_absent",
    "runtime_unavailable",
)

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "smoke_models_used",
    "fixture_rows_checked",
    "live_rows_checked",
    "failure_taxonomy_counts",
    "prompt_prefix_hashes",
    "grammar_runtime_available",
    "grammar_mask_applied",
    "truncation_detected",
    "json_extraction_success_rate",
    "schema_validity_rate",
    "exact_validator_handoff_ready",
    "gpu_offload_evidence",
    "sota_schema_failure_taxonomy_ready",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5525_sota_schema_failure_taxonomy.py",
    "tests/python/test_experiment_5512_structured_output_positive_control.py",
    "tests/python/test_experiment_5513_sota_hard_soft_structured_panel.py",
)

FIELD_PRINCIPLES: JsonDict = {
    "model_specs": "Names the exact mandated local GGUF candidates so fixture success is not mistaken for an unnamed model claim.",
    "smoke_models_used": "Identifies which model actually supplied live evidence and keeps legacy smoke models out of headline credit.",
    "fixture_rows_checked": "Confirms the deterministic positive-control rows still pass before interpreting live failures.",
    "live_rows_checked": "Counts emitted and missing live row slots so absent candidates cannot disappear from rates.",
    "failure_taxonomy_counts": "Makes the first observed interface failure auditable instead of collapsing all failures into invalid JSON.",
    "prompt_prefix_hashes": "Content-addresses the prompt contract that the live model saw before candidate-specific payloads.",
    "grammar_runtime_available": "Separates missing constrained-decoding infrastructure from model behavior.",
    "grammar_mask_applied": "Shows whether an available grammar was actually passed into the live generation call.",
    "truncation_detected": "Flags token-budget exhaustion separately from semantic or schema mistakes.",
    "json_extraction_success_rate": "Distinguishes finding any JSON payload from validating the required schema.",
    "schema_validity_rate": "Measures live schema conformance without letting fixture rows mask live invalidity.",
    "exact_validator_handoff_ready": "Records whether live rows reached the deterministic hard/soft validator authority.",
    "gpu_offload_evidence": "Preserves CUDA and memory-delta evidence for any live SOTA GGUF row.",
    "sota_schema_failure_taxonomy_ready": "States whether the diagnostic covered fixture and live evidence without dropping rows.",
    "tests_added_or_reused": "Links the artifact to tests that exercise the taxonomy and reused parser contracts.",
    "field_principles": "Explains why each headline and gate field exists so future agents preserve the audit boundary.",
    "inference_substrate": "Declares fixture plus live-smoke diagnosis rather than a full headline SOTA panel.",
    "honest_verdict": "Provides a terminal status that cannot promote malformed live rows into success.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON in a stable form for hashes and checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest for a UTF-8 string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return the SHA-256 hex digest for a JSON-compatible value."""

    return sha256_text(canonical_json(value))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def prompt_prefix() -> str:
    """Return the Exp5513 prompt prefix before candidate-row payloads."""

    marker = "Candidate rows to transcribe and verify:"
    full_prompt = panel.build_reason_then_structure_prompt([])
    return full_prompt.split(marker, 1)[0]


def prompt_prefix_hash() -> str:
    """Hash the prompt prefix that controls the live structured-output contract."""

    return sha256_text(prompt_prefix())


def classify_first_failure(
    row: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> str | None:
    """Classify the first visible structured-output failure for one row."""

    ctx = dict(context or {})
    parse_status = str(row.get("parse_status", ""))
    schema_errors = [str(error) for error in row.get("schema_errors", [])]

    if row.get("runtime_error") or parse_status == "runtime_error":
        return "runtime_unavailable"
    if _prompt_contract_miss(row, ctx):
        return "prompt_contract_miss"
    if ctx.get("grammar_runtime_available") is False:
        return "grammar_runtime_unavailable"
    if (
        ctx.get("grammar_mask_applied") is False
        and parse_status != "missing_candidate_row"
        and row.get("schema_valid") is not True
    ):
        return "grammar_mask_not_applied"
    if ctx.get("truncated") is True and parse_status in {
        "no_json_payload",
        "no_json_object",
        "truncated_json",
    }:
        return "max_tokens_truncation"
    if parse_status in {"no_json_payload", "no_json_object"}:
        return "json_extraction_failure"
    if parse_status == "missing_candidate_row":
        return "semantic_candidate_absent"
    if schema_errors:
        if any(" is required" in error for error in schema_errors):
            return "required_field_missing"
        return "json_schema_invalid"
    if row.get("parseable") is True and row.get("exact_validator_correct") is False:
        return "exact_validator_mismatch"
    return None


def build_fixture_diagnostic_rows(
    fixture: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Run deterministic fixture rows through the Exp5513 extractor path."""

    fixture_payload = dict(fixture or positive.load_fixture_artifact()["fixture"])
    fixture_mod.validate_fixture(fixture_payload)
    target_rows = positive.build_fixture_candidate_payloads(fixture_payload)
    wrapper_text = json.dumps({"candidate_rows": target_rows, "proof_claims": []}, sort_keys=True)
    extracted = panel.extract_candidate_payloads(wrapper_text)
    output_len = len(wrapper_text.encode("utf-8"))
    rows = []
    for payload in extracted["candidate_payloads"]:
        classified = positive.classify_candidate_payload(payload, fixture=fixture_payload)
        rows.append(
            _diagnostic_row(
                classified,
                row_source="fixture",
                context={
                    "grammar_runtime_available": True,
                    "grammar_mask_applied": True,
                    "truncated": False,
                },
                max_tokens=None,
                stop_strings=[],
                output_byte_length=output_len,
                output_byte_length_source="fixture_wrapper",
                truncation_marker=None,
                grammar_backend="deterministic_fixture",
                grammar_mask_applied=True,
                gpu_offload_receipt={},
                exact_validator_target=payload.get("validator_target"),
                source_artifact=positive.FIXTURE_ARTIFACT_RELATIVE_PATH.as_posix(),
                json_extraction_succeeded=True,
            )
        )
    return rows


def build_live_diagnostic_rows(
    *,
    live_artifact_path: Path = REPO_ROOT / LIVE_ARTIFACT_RELATIVE_PATH,
    runtime_status: Mapping[str, Any],
    fixture: Mapping[str, Any] | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Build diagnostic rows from the preserved Exp5513 live smoke artifact."""

    fixture_payload = dict(fixture or positive.load_fixture_artifact()["fixture"])
    fixture_mod.validate_fixture(fixture_payload)
    grammar_backend = _grammar_backend(runtime_status)
    if not live_artifact_path.exists():
        row = _diagnostic_row(
            {
                "parse_status": "runtime_error",
                "runtime_error": f"missing_live_artifact:{live_artifact_path.as_posix()}",
                "schema_valid": False,
                "parseable": False,
                "schema_errors": [],
                "exact_validator_correct": False,
                "exact_validator_verdict": "not_handed_off",
            },
            row_source="live_runtime_unavailable",
            context={
                "grammar_runtime_available": bool(
                    runtime_status.get("grammar_runtime_available")
                ),
                "grammar_mask_applied": False,
                "truncated": False,
            },
            max_tokens=DEFAULT_LIVE_MAX_TOKENS,
            stop_strings=[],
            output_byte_length=0,
            output_byte_length_source="missing",
            truncation_marker=None,
            grammar_backend=grammar_backend,
            grammar_mask_applied=False,
            gpu_offload_receipt=_gpu_offload_evidence({}, runtime_status),
            exact_validator_target=None,
            source_artifact=live_artifact_path.as_posix(),
            json_extraction_succeeded=False,
        )
        return [row], {}

    live_artifact = json.loads(live_artifact_path.read_text(encoding="utf-8"))
    gpu_evidence = _gpu_offload_evidence(live_artifact, runtime_status)
    rows: list[JsonDict] = []
    for run in live_artifact.get("model_runs", []):
        if not isinstance(run, Mapping):
            continue
        rows.extend(
            _rows_from_model_run(
                run=run,
                fixture=fixture_payload,
                runtime_status=runtime_status,
                grammar_backend=grammar_backend,
                gpu_offload_receipt=gpu_evidence,
                source_artifact=live_artifact_path.as_posix(),
            )
        )
    return rows, live_artifact


def build_artifact(
    *,
    live_artifact_path: Path = REPO_ROOT / LIVE_ARTIFACT_RELATIVE_PATH,
    positive_control_path: Path = REPO_ROOT / POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5525 diagnostic taxonomy artifact."""

    fixture = positive.load_fixture_artifact()["fixture"]
    runtime_status = _positive_runtime_status(positive_control_path)
    fixture_rows = build_fixture_diagnostic_rows(fixture)
    live_rows, live_artifact = build_live_diagnostic_rows(
        live_artifact_path=Path(live_artifact_path),
        runtime_status=runtime_status,
        fixture=fixture,
    )
    model_specs = _model_specs(live_artifact)
    smoke_models_used = _smoke_models_used(live_artifact)
    counts = _failure_taxonomy_counts([*fixture_rows, *live_rows])
    taxonomy_ready = _taxonomy_ready(fixture_rows, live_rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "model_specs": model_specs,
        "smoke_models_used": smoke_models_used,
        "fixture_rows_checked": len(fixture_rows),
        "live_rows_checked": len(live_rows),
        "failure_taxonomy_counts": counts,
        "prompt_prefix_hashes": _prompt_hashes([*fixture_rows, *live_rows]),
        "grammar_runtime_available": bool(runtime_status.get("grammar_runtime_available")),
        "grammar_mask_applied": any(
            row.get("grammar_mask_applied") is True for row in live_rows
        ),
        "truncation_detected": any(row.get("truncation_marker") for row in live_rows),
        "json_extraction_success_rate": _json_extraction_success_rate(live_rows),
        "schema_validity_rate": _schema_validity_rate(live_rows),
        "exact_validator_handoff_ready": _exact_validator_handoff_ready(live_rows),
        "gpu_offload_evidence": _gpu_offload_evidence(live_artifact, runtime_status),
        "sota_schema_failure_taxonomy_ready": taxonomy_ready,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(taxonomy_ready, counts),
        "structured_positive_control_artifact": POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH.as_posix(),
        "live_artifact": Path(live_artifact_path).as_posix(),
        "fixture_schema_validity_rate": _schema_validity_rate(fixture_rows),
        "fixture_exact_validator_handoff_ready": _exact_validator_handoff_ready(fixture_rows),
        "grammar_backend": _grammar_backend(runtime_status),
        "parser_backend": PARSER_BACKEND,
        "diagnostic_rows": {"fixture": fixture_rows, "live": live_rows},
        "runtime_status": runtime_status,
        "no_autotokenizer_on_gguf": True,
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
    live_artifact_path: Path = REPO_ROOT / LIVE_ARTIFACT_RELATIVE_PATH,
    positive_control_path: Path = REPO_ROOT / POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5525 result JSON."""

    artifact = build_artifact(
        live_artifact_path=Path(live_artifact_path),
        positive_control_path=Path(positive_control_path),
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
    """Validate Exp5525 fields and fail closed on diagnostic overclaiming."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    _require(isinstance(artifact.get("model_specs"), list), "model_specs")
    _require(isinstance(artifact.get("smoke_models_used"), list), "smoke_models_used")
    _require(int(artifact.get("fixture_rows_checked", -1)) >= 0, "fixture_rows_checked")
    _require(int(artifact.get("live_rows_checked", -1)) >= 0, "live_rows_checked")
    _require(
        set(artifact.get("failure_taxonomy_counts", {})) == set(FAILURE_CATEGORIES),
        "failure_taxonomy_counts",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})),
        "field_principles",
    )
    for field in (
        "grammar_runtime_available",
        "grammar_mask_applied",
        "truncation_detected",
        "exact_validator_handoff_ready",
        "sota_schema_failure_taxonomy_ready",
    ):
        _require(isinstance(artifact.get(field), bool), field)
    for field in ("json_extraction_success_rate", "schema_validity_rate"):
        value = float(artifact.get(field, -1.0))
        _require(0.0 <= value <= 1.0, field)
    for digest in artifact.get("prompt_prefix_hashes", []):
        _require(isinstance(digest, str) and len(digest) == 64, "prompt_prefix_hashes")
    _require(isinstance(artifact.get("gpu_offload_evidence"), Mapping), "gpu_offload_evidence")
    _require(artifact.get("no_autotokenizer_on_gguf") is True, "no_autotokenizer_on_gguf")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    if artifact.get("sota_schema_failure_taxonomy_ready") is True:
        _require(int(artifact.get("fixture_rows_checked", 0)) > 0, "fixture_rows_checked")
        _require(int(artifact.get("live_rows_checked", 0)) > 0, "live_rows_checked")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(ready: bool, counts: Mapping[str, int]) -> str:
    """Return a terminal verdict for the taxonomy artifact."""

    if not ready:
        return "blocked: sota_schema_failure_taxonomy_not_ready_no_live_smoke_rows"
    observed = [name for name in FAILURE_CATEGORIES if counts.get(name, 0) > 0]
    suffix = "_and_".join(observed) if observed else "no_failures_observed"
    return f"complete: sota_schema_failure_taxonomy_ready_{suffix}"


def _rows_from_model_run(
    *,
    run: Mapping[str, Any],
    fixture: Mapping[str, Any],
    runtime_status: Mapping[str, Any],
    grammar_backend: str,
    gpu_offload_receipt: Mapping[str, Any],
    source_artifact: str,
) -> list[JsonDict]:
    max_tokens = int(run.get("max_tokens", DEFAULT_LIVE_MAX_TOKENS))
    completion_tokens = int(run.get("completion_tokens", 0))
    truncated = max_tokens > 0 and completion_tokens >= max_tokens
    output = str(run.get("raw_output", run.get("raw_output_preview", "")))
    output_source = "raw_output" if run.get("raw_output") is not None else "raw_output_preview"
    output_len = len(output.encode("utf-8"))
    truncation_marker = "completion_tokens_reached_max_tokens" if truncated else None
    grammar_mask_applied = bool(run.get("grammar_mask_applied", False))
    base_context = {
        "grammar_runtime_available": bool(runtime_status.get("grammar_runtime_available")),
        "grammar_mask_applied": grammar_mask_applied,
        "truncated": truncated,
    }
    rows: list[JsonDict] = []
    for failure in run.get("parse_failures", []):
        if not isinstance(failure, Mapping):
            continue
        failure_row = {
            "model_hf_id": run.get("model_hf_id"),
            "parse_status": failure.get("parse_status", "no_json_payload"),
            "schema_valid": False,
            "parseable": False,
            "schema_errors": [],
            "exact_validator_correct": False,
            "exact_validator_verdict": "not_handed_off",
            "runtime_error": failure.get("detail") if failure.get("parse_status") == "runtime_error" else None,
        }
        rows.append(
            _diagnostic_row(
                failure_row,
                row_source="live_parse_failure",
                context=base_context,
                max_tokens=max_tokens,
                stop_strings=run.get("stop_strings", []),
                output_byte_length=output_len,
                output_byte_length_source=output_source,
                truncation_marker=truncation_marker,
                grammar_backend=grammar_backend,
                grammar_mask_applied=grammar_mask_applied,
                gpu_offload_receipt=gpu_offload_receipt,
                exact_validator_target=None,
                source_artifact=source_artifact,
                json_extraction_succeeded=False,
            )
        )
    for row in run.get("candidate_rows", []):
        if not isinstance(row, Mapping):
            continue
        payload = row.get("parsed_payload", {})
        exact_target = payload.get("validator_target") if isinstance(payload, Mapping) else None
        rows.append(
            _diagnostic_row(
                row,
                row_source="live_emitted",
                context=base_context,
                max_tokens=max_tokens,
                stop_strings=run.get("stop_strings", []),
                output_byte_length=output_len,
                output_byte_length_source=output_source,
                truncation_marker=truncation_marker,
                grammar_backend=grammar_backend,
                grammar_mask_applied=grammar_mask_applied,
                gpu_offload_receipt=gpu_offload_receipt,
                exact_validator_target=exact_target,
                source_artifact=source_artifact,
                json_extraction_succeeded=True,
            )
        )
    for instance_id in run.get("missing_instance_ids", []):
        missing_id = str(instance_id)
        rows.append(
            _diagnostic_row(
                {
                    "model_hf_id": run.get("model_hf_id"),
                    "instance_id": missing_id,
                    "candidate_id": None,
                    "parse_status": "missing_candidate_row",
                    "schema_valid": False,
                    "parseable": False,
                    "schema_errors": [],
                    "exact_validator_correct": False,
                    "exact_validator_verdict": "not_handed_off",
                },
                row_source="live_missing",
                context=base_context,
                max_tokens=max_tokens,
                stop_strings=run.get("stop_strings", []),
                output_byte_length=output_len,
                output_byte_length_source=output_source,
                truncation_marker=truncation_marker,
                grammar_backend=grammar_backend,
                grammar_mask_applied=grammar_mask_applied,
                gpu_offload_receipt=gpu_offload_receipt,
                exact_validator_target=_expected_validator_target(fixture, missing_id),
                source_artifact=source_artifact,
                json_extraction_succeeded=False,
            )
        )
    return rows


def _diagnostic_row(
    row: Mapping[str, Any],
    *,
    row_source: str,
    context: Mapping[str, Any],
    max_tokens: int | None,
    stop_strings: Sequence[Any],
    output_byte_length: int,
    output_byte_length_source: str,
    truncation_marker: str | None,
    grammar_backend: str,
    grammar_mask_applied: bool,
    gpu_offload_receipt: Mapping[str, Any],
    exact_validator_target: Any,
    source_artifact: str,
    json_extraction_succeeded: bool,
) -> JsonDict:
    enriched = dict(row)
    enriched["row_source"] = row_source
    enriched["first_failure"] = classify_first_failure(enriched, context)
    enriched["secondary_failures"] = _secondary_failures(enriched, context)
    enriched["prompt_prefix_hash"] = prompt_prefix_hash()
    enriched["max_tokens"] = max_tokens
    enriched["stop_strings"] = [str(item) for item in stop_strings]
    enriched["output_byte_length"] = output_byte_length
    enriched["output_byte_length_source"] = output_byte_length_source
    enriched["truncation_marker"] = truncation_marker
    enriched["grammar_backend"] = grammar_backend
    enriched["grammar_mask_applied"] = grammar_mask_applied
    enriched["parser_backend"] = PARSER_BACKEND
    enriched["gpu_offload_receipt"] = dict(gpu_offload_receipt)
    enriched["exact_validator_target"] = exact_validator_target
    enriched["source_artifact"] = source_artifact
    enriched["json_extraction_succeeded"] = json_extraction_succeeded
    return enriched


def _secondary_failures(row: Mapping[str, Any], context: Mapping[str, Any]) -> list[str]:
    failures = []
    first = row.get("first_failure")
    for category in FAILURE_CATEGORIES:
        probe_context = dict(context)
        if category == "prompt_contract_miss" and _prompt_contract_miss(row, probe_context):
            failures.append(category)
        elif category == "grammar_runtime_unavailable" and probe_context.get("grammar_runtime_available") is False:
            failures.append(category)
        elif (
            category == "grammar_mask_not_applied"
            and probe_context.get("grammar_mask_applied") is False
            and row.get("parse_status") != "missing_candidate_row"
            and row.get("schema_valid") is not True
        ):
            failures.append(category)
        elif category == "max_tokens_truncation" and probe_context.get("truncated") is True:
            failures.append(category)
        elif category == "json_extraction_failure" and row.get("parse_status") in {
            "no_json_payload",
            "no_json_object",
        }:
            failures.append(category)
        elif category == "semantic_candidate_absent" and row.get("parse_status") == "missing_candidate_row":
            failures.append(category)
        elif category in {"required_field_missing", "json_schema_invalid"}:
            schema_errors = [str(error) for error in row.get("schema_errors", [])]
            if category == "required_field_missing" and any(
                " is required" in error for error in schema_errors
            ):
                failures.append(category)
            elif category == "json_schema_invalid" and schema_errors:
                failures.append(category)
        elif (
            category == "exact_validator_mismatch"
            and row.get("parseable") is True
            and row.get("exact_validator_correct") is False
        ):
            failures.append(category)
        elif category == "runtime_unavailable" and (
            row.get("runtime_error") or row.get("parse_status") == "runtime_error"
        ):
            failures.append(category)
    return [failure for failure in failures if failure != first]


def _prompt_contract_miss(row: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    del context
    payload = row.get("parsed_payload", {})
    values = [row.get("candidate_id")]
    if isinstance(payload, Mapping):
        values.extend([payload.get("candidate_id"), payload.get("claimed_exact_validator_verdict")])
    return any(str(value).strip() == "..." for value in values if value is not None)


def _positive_runtime_status(path: Path) -> JsonDict:
    if not path.exists():
        return {
            "grammar_runtime_available": False,
            "parser_only_fallback_used": True,
            "llama_cpp_cuda_available": False,
            "llama_cpp_grammar_available": False,
            "runtime_blockers": ["structured_positive_control_artifact_missing"],
        }
    artifact = json.loads(path.read_text(encoding="utf-8"))
    runtime = artifact.get("runtime_status", {})
    if isinstance(runtime, Mapping):
        return dict(runtime)
    return {
        "grammar_runtime_available": bool(artifact.get("grammar_runtime_available")),
        "parser_only_fallback_used": bool(artifact.get("parser_only_fallback_used")),
        "llama_cpp_cuda_available": bool(artifact.get("llama_cpp_cuda_available")),
        "llama_cpp_grammar_available": False,
        "runtime_blockers": [],
    }


def _grammar_backend(runtime_status: Mapping[str, Any]) -> str:
    if runtime_status.get("llama_cpp_grammar_available") is True:
        return "llama_cpp_gbnf"
    if runtime_status.get("llguidance_available") is True:
        return "llguidance_json_schema"
    if runtime_status.get("xgrammar_available") is True:
        return "xgrammar"
    return "none"


def _gpu_offload_evidence(
    live_artifact: Mapping[str, Any],
    runtime_status: Mapping[str, Any],
) -> JsonDict:
    runtime = live_artifact.get("runtime_status", {}) if isinstance(live_artifact, Mapping) else {}
    if not isinstance(runtime, Mapping):
        runtime = {}
    diagnostics = live_artifact.get("offload_diagnostics", []) if isinstance(live_artifact, Mapping) else []
    if not diagnostics:
        diagnostics = runtime.get("offload_diagnostics", runtime_status.get("offload_diagnostics", []))
    return {
        "llama_cpp_cuda_available": bool(
            runtime.get(
                "llama_cpp_cuda_available",
                runtime_status.get("llama_cpp_cuda_available", False),
            )
        ),
        "gpu_offload_verified": bool(
            runtime.get("gpu_offload_verified", runtime_status.get("gpu_offload_verified", False))
        ),
        "gpu_memory_delta_mb": float(
            runtime.get("gpu_memory_delta_mb", runtime_status.get("gpu_memory_delta_mb", 0.0))
        ),
        "offload_diagnostics": list(diagnostics) if isinstance(diagnostics, Sequence) else [],
    }


def _model_specs(live_artifact: Mapping[str, Any]) -> list[JsonDict]:
    specs = live_artifact.get("model_specs", []) if isinstance(live_artifact, Mapping) else []
    if isinstance(specs, list) and specs:
        return [dict(row) for row in specs if isinstance(row, Mapping)]
    return positive.resolve_model_specs()


def _smoke_models_used(live_artifact: Mapping[str, Any]) -> list[str]:
    used = live_artifact.get("headline_models_used", []) if isinstance(live_artifact, Mapping) else []
    if isinstance(used, list) and used:
        return [str(row) for row in used]
    runs = live_artifact.get("model_runs", []) if isinstance(live_artifact, Mapping) else []
    if not isinstance(runs, list):
        return []
    return [
        str(run.get("model_hf_id"))
        for run in runs
        if isinstance(run, Mapping) and run.get("model_hf_id") and run.get("runtime_error") is None
    ]


def _expected_validator_target(fixture: Mapping[str, Any], instance_id: str) -> JsonDict | None:
    for instance in fixture.get("instances", []):
        if str(instance.get("instance_id")) == instance_id:
            return {
                "instance_id": str(instance["instance_id"]),
                "expected_status": str(instance["expected_status"]),
                "reference_solver_path": fixture_mod.REFERENCE_SOLVER_PATH,
                "hard_constraint_ids": [str(row["id"]) for row in instance["hard_constraints"]],
                "soft_preference_ids": [str(row["id"]) for row in instance["soft_preferences"]],
                "typed_claim_names": [str(row["name"]) for row in instance["typed_claims"]],
            }
    return None


def _failure_taxonomy_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = {category: 0 for category in FAILURE_CATEGORIES}
    for row in rows:
        failure = row.get("first_failure")
        if failure in counts:
            counts[str(failure)] += 1
    return counts


def _prompt_hashes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row["prompt_prefix_hash"]) for row in rows if row.get("prompt_prefix_hash")})


def _json_extraction_success_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    extraction_rows = [
        row
        for row in rows
        if row.get("row_source") in {"live_emitted", "live_parse_failure", "live_runtime_unavailable"}
    ]
    if not extraction_rows:
        return 0.0
    successes = sum(int(row.get("json_extraction_succeeded") is True) for row in extraction_rows)
    return _rate(successes, len(extraction_rows))


def _schema_validity_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return _rate(sum(int(row.get("schema_valid") is True) for row in rows), len(rows))


def _exact_validator_handoff_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    return all(
        row.get("exact_validator_verdict") != "not_handed_off"
        and row.get("first_failure") is None
        for row in rows
    )


def _taxonomy_ready(
    fixture_rows: Sequence[Mapping[str, Any]],
    live_rows: Sequence[Mapping[str, Any]],
) -> bool:
    fixture_ready = bool(fixture_rows) and all(row.get("first_failure") is None for row in fixture_rows)
    live_has_smoke = any(
        row.get("row_source") in {"live_emitted", "live_missing", "live_parse_failure"}
        for row in live_rows
    )
    classified = all(
        row.get("first_failure") in FAILURE_CATEGORIES or row.get("first_failure") is None
        for row in [*fixture_rows, *live_rows]
    )
    return fixture_ready and live_has_smoke and classified


def _rate(numerator: int | float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


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
                "failure_taxonomy_counts": artifact["failure_taxonomy_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
