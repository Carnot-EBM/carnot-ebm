"""Exp5605 append-only raw response evidence envelope.

Spec refs: REQ-VERIFY-5605, SCENARIO-VERIFY-5605.

This module builds the contract that Exp5580 proved was missing: every future
model call must keep the raw response bytes, not only a response hash.  The
positive-control rows here are deliberately small fixture calls.  They exercise
the envelope reader, parser replay, truncation visibility, and corruption
rejection before another expensive local-SOTA panel depends on the contract.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import zlib
from typing import Any

from carnot import experiment_5567_local_sota_solve_verify_asymmetry as exp5567
from carnot import experiment_5580_parser_forensics_positive_control as parser5580
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5605_raw_response_evidence_envelope.json")
EXP5567_RELATIVE_PATH = exp5567.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5605.raw_response_evidence_envelope.v506"
ENVELOPE_SCHEMA_VERSION = "carnot.raw_response_evidence_envelope.v1"
EXPERIMENT = 5605
EXPERIMENT_ID = "exp5605-raw-response-evidence-envelope"
MILESTONE = "2026.07.506"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5605
INFERENCE_SUBSTRATE = "local_gguf_llamacpp_cuda_evidence_fixture"
PARSER_NAME = "carnot.exp5580.deterministic_json_parser"
PARSER_VERSION = parser5580.SCHEMA
QWEN_ID = exp5567.QWEN_ID
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_HF_IDS = (QWEN_ID, GEMMA26_ID)
SPEC_REFS = ("REQ-VERIFY-5605", "SCENARIO-VERIFY-5605", "REQ-VERIFY-5580")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "model_specs",
    "gpu_offload_authenticated",
    "envelope_schema_version",
    "response_rows_written",
    "raw_payloads_preserved",
    "lossless_replay_rate",
    "truncation_controls_detected",
    "payload_corruption_rejected",
    "semantic_false_accept_count",
    "parser_version_replay_passed",
    "envelope_ready",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Every headline and gate field names the evidence boundary it protects.",
    "model_specs": "Mandated local families are explicit and cannot be replaced by legacy CPU smoke models.",
    "gpu_offload_authenticated": "CPU fallback is not headline evidence.",
    "envelope_schema_version": "Replay has a stable contract.",
    "response_rows_written": "Every call has a row.",
    "raw_payloads_preserved": "Hashes alone are insufficient.",
    "lossless_replay_rate": "Preservation is positively tested.",
    "truncation_controls_detected": "Stop failures stay visible.",
    "payload_corruption_rejected": "Tampering fails closed.",
    "semantic_false_accept_count": "Parser repair cannot invent correctness.",
    "parser_version_replay_passed": "Aggregates can be regenerated.",
    "envelope_ready": "Downstream inference requires every control.",
    "inference_substrate": "Execution is local and authenticated.",
    "honest_verdict": "Failed preservation blocks the panel.",
}

REQUIRED_ROW_FIELDS = (
    "envelope_schema_version",
    "sequence_index",
    "call_id",
    "control_kind",
    "model_family",
    "model_hf_id",
    "model_local_path_hash",
    "prompt_payload",
    "prompt_hash",
    "raw_response_payload",
    "payload_hash",
    "llama_cpp_version",
    "llama_cpp_arguments",
    "device_offload_receipt",
    "sampling_parameters",
    "seed",
    "stop_reason",
    "token_counts",
    "truncation_flag",
    "parser_name",
    "parser_version",
    "parsed_object",
    "exact_validator_outcome",
    "timestamp_utc",
    "previous_row_hash",
    "row_hash",
)


class EnvelopeIntegrityError(ValueError):
    """Raised when a stored envelope row cannot be replayed exactly."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically before hashing or writing artifacts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest for byte evidence."""

    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping in canonical form."""

    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def utc_timestamp(sequence_index: int) -> str:
    """Return a stable fixture timestamp for deterministic artifacts."""

    return f"2026-07-14T00:00:{sequence_index:02d}Z"


def encode_prompt(prompt: str) -> JsonDict:
    """Encode prompt bytes losslessly without external storage."""

    data = prompt.encode("utf-8")
    return {
        "encoding": "base64",
        "bytes_b64": base64.b64encode(data).decode("ascii"),
        "byte_length": len(data),
    }


def decode_prompt(payload: Mapping[str, Any]) -> bytes:
    """Decode prompt bytes from the local envelope payload."""

    if payload.get("encoding") != "base64":
        raise EnvelopeIntegrityError("prompt_encoding")  # pragma: no cover
    return base64.b64decode(str(payload.get("bytes_b64", "")).encode("ascii"))


def encode_lossless_payload(raw: bytes) -> JsonDict:
    """Compress response bytes losslessly for local artifact storage."""

    compressed = zlib.compress(raw)
    return {
        "encoding": "base64",
        "compression": "zlib",
        "bytes_b64": base64.b64encode(compressed).decode("ascii"),
        "uncompressed_byte_length": len(raw),
    }


def decode_lossless_payload(payload: Mapping[str, Any]) -> bytes:
    """Recover response bytes exactly from the envelope payload."""

    if payload.get("compression") != "zlib" or payload.get("encoding") != "base64":
        raise EnvelopeIntegrityError("payload_encoding")  # pragma: no cover
    compressed = base64.b64decode(str(payload.get("bytes_b64", "")).encode("ascii"))
    return zlib.decompress(compressed)


def local_path_hash(model_path: str) -> str:
    """Hash the resolved local GGUF path rather than the multi-GB model file."""

    return sha256_bytes(str(Path(model_path).expanduser().resolve()).encode("utf-8"))


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return only the two mandated headline model specs with local path hashes."""

    normalized: list[JsonDict] = []
    for spec in model_specs:
        hf_id = str(spec.get("hf_id", ""))
        if hf_id not in MANDATED_HF_IDS:
            continue
        path = str(spec.get("model_path", ""))
        family = "qwen" if hf_id == QWEN_ID else "gemma"
        normalized.append(
            {
                "name": str(spec.get("name", "")),
                "hf_id": hf_id,
                "family": family,
                "role": str(spec.get("role", "moe")),
                "gpu": int(spec.get("gpu", len(normalized)) or 0),
                "model_path": path,
                "local_path_hash": local_path_hash(path) if path else "",
                "local_model_present": bool(path and Path(path).is_file()),
                "headline_eligible": spec.get("headline_eligible") is not False,
            }
        )
    return sorted(normalized, key=lambda row: MANDATED_HF_IDS.index(row["hf_id"]))


def build_envelope_rows(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    device_receipt: Mapping[str, Any],
) -> list[JsonDict]:
    """Build two valid and two malformed/truncated fixture rows per model family."""

    rows: list[JsonDict] = []
    previous_hash = ""
    for spec in normalize_model_specs(model_specs):
        for fixture in _fixtures_for_family(str(spec["family"])):
            sequence_index = len(rows)
            row = _build_row(
                sequence_index=sequence_index,
                spec=spec,
                fixture=fixture,
                device_receipt=device_receipt,
                previous_hash=previous_hash,
            )
            previous_hash = row["row_hash"]
            rows.append(row)
    return rows


def replay_envelope_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Verify hashes, decompress payloads, replay parser output, and summarize gates."""

    replayed: list[JsonDict] = []
    previous_hash = ""
    for row in rows:
        if row.get("previous_row_hash") != previous_hash:
            raise EnvelopeIntegrityError("previous_row_hash")
        if row.get("row_hash") != row_hash(row):
            raise EnvelopeIntegrityError("row_hash")
        prompt_bytes = decode_prompt(_as_mapping(row.get("prompt_payload"), "prompt_payload"))
        raw_bytes = decode_lossless_payload(
            _as_mapping(row.get("raw_response_payload"), "raw_response_payload")
        )
        if row.get("prompt_hash") != sha256_bytes(prompt_bytes):
            raise EnvelopeIntegrityError("prompt_hash")
        if row.get("payload_hash") != sha256_bytes(raw_bytes):
            raise EnvelopeIntegrityError("payload_hash")
        parsed = parse_fixture_response(raw_bytes.decode("utf-8"), str(row.get("expected_label") or ""))
        if row.get("parser_name") != PARSER_NAME or row.get("parser_version") != PARSER_VERSION:
            raise EnvelopeIntegrityError("parser_version")
        if row.get("parsed_object") != parsed["parsed_object"]:
            raise EnvelopeIntegrityError("parsed_object")
        if row.get("exact_validator_outcome") != parsed["exact_validator_outcome"]:
            raise EnvelopeIntegrityError("exact_validator_outcome")
        copy_row = dict(row)
        copy_row["replayed_prompt_text"] = prompt_bytes.decode("utf-8")
        replayed.append(copy_row)
        previous_hash = str(row["row_hash"])
    truncation_count = sum(1 for row in replayed if row.get("truncation_flag") is True)
    semantic_false_accepts = sum(
        1
        for row in replayed
        if row.get("control_kind") != "known_valid"
        and row.get("exact_validator_outcome", {}).get("accepted") is True
    )
    return {
        "rows": replayed,
        "lossless_replay_rate": 1.0 if rows else 0.0,
        "raw_payloads_preserved": all(_row_preserves_payload(row) for row in rows),
        "truncation_controls_detected": truncation_count,
        "semantic_false_accept_count": semantic_false_accepts,
        "parser_version_replay_passed": True,
    }


def corrupt_first_payload(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return a deep-copied row list with one stored payload tampered."""

    corrupted = copy.deepcopy([dict(row) for row in rows])
    payload = dict(corrupted[0]["raw_response_payload"])
    payload["bytes_b64"] = base64.b64encode(zlib.compress(b'{"verdict":"invalid"}')).decode("ascii")
    payload["uncompressed_byte_length"] = len(b'{"verdict":"invalid"}')
    corrupted[0]["raw_response_payload"] = payload
    corrupted[0]["row_hash"] = row_hash(corrupted[0])
    return corrupted


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    device_receipt: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5605 artifact from local fixture evidence."""

    normalized_specs = normalize_model_specs(model_specs)
    rows = build_envelope_rows(model_specs=normalized_specs, device_receipt=device_receipt)
    replay = replay_envelope_rows(rows)
    corruption_rejected = payload_corruption_is_rejected(rows)
    gpu_ok = gpu_offload_authenticated(device_receipt, normalized_specs)
    model_ok = model_specs_ready(normalized_specs)
    ready = bool(
        model_ok
        and gpu_ok
        and replay["raw_payloads_preserved"]
        and replay["lossless_replay_rate"] == 1.0
        and replay["truncation_controls_detected"] >= 2
        and corruption_rejected
        and replay["semantic_false_accept_count"] == 0
        and replay["parser_version_replay_passed"]
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "model_specs": normalized_specs,
        "gpu_offload_authenticated": gpu_ok,
        "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "response_rows_written": len(rows),
        "raw_payloads_preserved": replay["raw_payloads_preserved"],
        "lossless_replay_rate": replay["lossless_replay_rate"],
        "truncation_controls_detected": replay["truncation_controls_detected"],
        "payload_corruption_rejected": corruption_rejected,
        "semantic_false_accept_count": replay["semantic_false_accept_count"],
        "parser_version_replay_passed": replay["parser_version_replay_passed"],
        "envelope_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, model_ok=model_ok, gpu_ok=gpu_ok),
        "response_envelope_rows": rows,
        "device_receipt": dict(device_receipt),
        "privacy_local_storage": {
            "payload_storage": "inline_local_json_zlib_base64",
            "external_upload": False,
            "raw_payload_scope": "fixture_controls_only",
        },
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on missing fields, overclaims, or invalid ready gates."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]), "field_principles")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("envelope_schema_version") == ENVELOPE_SCHEMA_VERSION, "envelope_schema_version")
    specs = artifact.get("model_specs", [])
    _require(model_specs_ready(specs), "model_specs")
    rows = artifact.get("response_envelope_rows", [])
    replay = replay_envelope_rows(rows if isinstance(rows, Sequence) else [])
    _require(artifact.get("response_rows_written") == len(rows), "response_rows_written")
    for key in (
        "raw_payloads_preserved",
        "lossless_replay_rate",
        "truncation_controls_detected",
        "semantic_false_accept_count",
        "parser_version_replay_passed",
    ):
        _require(artifact.get(key) == replay[key], key)
    _require(artifact.get("payload_corruption_rejected") is True, "payload_corruption_rejected")
    gpu_ok = artifact.get("gpu_offload_authenticated") is True
    expected_ready = bool(
        gpu_ok
        and replay["raw_payloads_preserved"]
        and replay["lossless_replay_rate"] == 1.0
        and replay["truncation_controls_detected"] >= 2
        and artifact.get("payload_corruption_rejected") is True
        and replay["semantic_false_accept_count"] == 0
        and replay["parser_version_replay_passed"] is True
    )
    _require(artifact.get("envelope_ready") is expected_ready, "envelope_ready")
    verdict = str(artifact.get("honest_verdict", ""))
    if expected_ready:
        _require(verdict.startswith("complete:"), "honest_verdict")
    else:
        _require(verdict.startswith("blocked_"), "honest_verdict")
    _require(artifact.get("reproducibility_checksum") == artifact_checksum(artifact), "checksum")


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    device_receipt: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5605 evidence-envelope artifact."""

    if model_specs is None:  # pragma: no cover
        model_specs = resolve_mandated_model_specs()
    if device_receipt is None:  # pragma: no cover
        device_receipt = authenticate_gpu_offload_from_prior_receipt(repo_root, model_specs)
    artifact = build_artifact(model_specs=model_specs, device_receipt=device_receipt, tests_run=tests_run)
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def resolve_mandated_model_specs() -> list[JsonDict]:  # pragma: no cover
    """Resolve the two required cached SOTA GGUF paths without downloading."""

    specs = cached_sota_pair(model_indices=(0, 1)) or []
    return normalize_model_specs(specs)


def authenticate_gpu_offload_from_prior_receipt(
    repo_root: Path,
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover
    """Reuse the latest local Exp5567 CUDA/offload receipt for this fixture gate."""

    receipt: JsonDict = {
        "source": EXP5567_RELATIVE_PATH.as_posix(),
        "torch_cuda_available": False,
        "torch_device_count": 0,
        "llama_cpp_supports_gpu_offload": False,
        "gpu_offload_authenticated": False,
        "devices": [],
        "model_receipts": [],
    }
    path = repo_root / EXP5567_RELATIVE_PATH
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_receipt = prior.get("device_receipt", {})
        if isinstance(prior_receipt, Mapping):
            receipt.update(dict(prior_receipt))
            receipt["source"] = EXP5567_RELATIVE_PATH.as_posix()
    receipt["gpu_offload_authenticated"] = gpu_offload_authenticated(receipt, model_specs)
    receipt["llama_cpp_version"] = str(receipt.get("llama_cpp_version") or llama_cpp_version())
    return receipt


def llama_cpp_version() -> str:  # pragma: no cover
    """Return the installed llama-cpp-python version when available."""

    try:
        import llama_cpp  # noqa: PLC0415

        return str(getattr(llama_cpp, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}"


def _build_row(
    *,
    sequence_index: int,
    spec: Mapping[str, Any],
    fixture: Mapping[str, Any],
    device_receipt: Mapping[str, Any],
    previous_hash: str,
) -> JsonDict:
    prompt = str(fixture["prompt"])
    raw = str(fixture["response"]).encode("utf-8")
    parsed = parse_fixture_response(raw.decode("utf-8"), str(fixture.get("expected_label") or ""))
    row: JsonDict = {
        "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "sequence_index": sequence_index,
        "call_id": f"{spec['family']}-{fixture['control_kind']}-{fixture['index']}",
        "control_kind": fixture["control_kind"],
        "model_family": spec["family"],
        "model_hf_id": spec["hf_id"],
        "model_local_path_hash": spec["local_path_hash"],
        "prompt_payload": encode_prompt(prompt),
        "prompt_hash": sha256_bytes(prompt.encode("utf-8")),
        "raw_response_payload": encode_lossless_payload(raw),
        "payload_hash": sha256_bytes(raw),
        "llama_cpp_version": str(device_receipt.get("llama_cpp_version") or "receipt_version_unrecorded"),
        "llama_cpp_arguments": {"n_gpu_layers": -1, "n_ctx": 8192, "n_batch": 256},
        "device_offload_receipt": _receipt_for_model(device_receipt, str(spec["hf_id"])),
        "sampling_parameters": {"temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.0},
        "seed": RANDOM_SEED + sequence_index,
        "stop_reason": fixture["stop_reason"],
        "token_counts": _token_counts(prompt, raw.decode("utf-8")),
        "truncation_flag": parsed["truncation_flag"],
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "parsed_object": parsed["parsed_object"],
        "exact_validator_outcome": parsed["exact_validator_outcome"],
        "expected_label": fixture.get("expected_label"),
        "timestamp_utc": utc_timestamp(sequence_index),
        "previous_row_hash": previous_hash,
        "row_hash": "",
    }
    row["row_hash"] = row_hash(row)
    return row


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash a row while excluding the self-referential row hash field."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def parse_fixture_response(text: str, expected_label: str) -> JsonDict:
    """Replay the Exp5580 parser and bind it to a fixture exact label."""

    parsed = parser5580.parse_verifier_label(text, "discrete_verdict")
    parser_ok = parsed.get("parser_ok") is True
    label = parsed.get("label") if parser_ok else None
    accepted = bool(expected_label and label == expected_label)
    return {
        "parsed_object": {"label": label} if accepted else None,
        "truncation_flag": parsed.get("error_type") == "truncation",
        "exact_validator_outcome": {
            "validator": "fixture_exact_label_match_v1",
            "accepted": accepted,
            "expected_label": expected_label or None,
            "observed_label": label,
            "parser_ok": parser_ok,
            "parser_error_type": parsed.get("error_type", ""),
        },
    }


def payload_corruption_is_rejected(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Prove that a tampered stored payload cannot pass the reader."""

    try:
        replay_envelope_rows(corrupt_first_payload(rows))
    except EnvelopeIntegrityError:
        return True
    return False  # pragma: no cover


def gpu_offload_authenticated(
    device_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> bool:
    """Return true only when the receipt covers every mandated local model."""

    if device_receipt.get("gpu_offload_authenticated") is not True:
        return False
    receipts = device_receipt.get("model_receipts", [])
    covered = {
        str(row.get("model_hf_id"))
        for row in receipts
        if isinstance(row, Mapping) and row.get("gpu_offload_authenticated") is True
    }
    expected = {str(spec.get("hf_id")) for spec in model_specs}
    return set(MANDATED_HF_IDS).issubset(expected) and expected.issubset(covered)


def model_specs_ready(model_specs: Any) -> bool:
    """Check that both mandated families resolve to local non-legacy GGUF paths."""

    if not isinstance(model_specs, Sequence) or isinstance(model_specs, (str, bytes)):
        return False
    ids = {str(spec.get("hf_id")) for spec in model_specs if isinstance(spec, Mapping)}
    if set(MANDATED_HF_IDS) != ids:
        return False
    return all(
        isinstance(spec, Mapping)
        and spec.get("local_model_present") is True
        and str(spec.get("model_path", "")).endswith(".gguf")
        for spec in model_specs
    )


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking the self-referential checksum field."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def honest_verdict(ready: bool, *, model_ok: bool, gpu_ok: bool) -> str:
    """Return the terminal verdict for the envelope gate."""

    if ready:
        return "complete: raw response evidence envelope fixture passed replay and fail-closed controls"
    if not model_ok:
        return "blocked_missing_mandated_local_gguf_paths"
    if not gpu_ok:
        return "blocked_no_cuda_offload_authenticated_cpu_diagnostics_only"
    return "blocked_raw_response_evidence_envelope_controls_failed"


def _fixtures_for_family(family: str) -> list[JsonDict]:
    return [
        {
            "index": 0,
            "control_kind": "known_valid",
            "prompt": f"{family}: return a strict valid verifier object.",
            "response": '{"verdict":"valid"}',
            "expected_label": "valid",
            "stop_reason": "stop_sequence",
        },
        {
            "index": 1,
            "control_kind": "known_valid",
            "prompt": f"{family}: return a wrapped accepted verifier object.",
            "response": '{"answer":{"decision":"accepted"}}',
            "expected_label": "valid",
            "stop_reason": "stop_sequence",
        },
        {
            "index": 2,
            "control_kind": "truncated_control",
            "prompt": f"{family}: deliberately truncate the verifier object.",
            "response": '{"verdict":"valid"',
            "expected_label": None,
            "stop_reason": "length",
        },
        {
            "index": 3,
            "control_kind": "malformed_control",
            "prompt": f"{family}: emit an unknown verdict token.",
            "response": '{"verdict":"maybe"}',
            "expected_label": None,
            "stop_reason": "stop_sequence",
        },
    ]


def _receipt_for_model(device_receipt: Mapping[str, Any], hf_id: str) -> JsonDict:
    receipts = device_receipt.get("model_receipts", [])
    for row in receipts:
        if isinstance(row, Mapping) and row.get("model_hf_id") == hf_id:
            return dict(row)
    return {
        "model_hf_id": hf_id,
        "gpu_offload_authenticated": False,
        "receipt_missing": True,
    }


def _token_counts(prompt: str, response: str) -> JsonDict:
    prompt_tokens = len(prompt.split())
    completion_tokens = len(response.split())
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "source": "whitespace_fixture_count",
    }


def _row_preserves_payload(row: Mapping[str, Any]) -> bool:
    payload = row.get("raw_response_payload")
    return isinstance(payload, Mapping) and payload.get("compression") == "zlib" and bool(
        payload.get("bytes_b64")
    )


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvelopeIntegrityError(field)
    return value


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


if __name__ == "__main__":  # pragma: no cover
    run()
