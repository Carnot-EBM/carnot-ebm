"""Compare direct, self-check, and SQL-grounded source verification.

The model sees only the source and candidate response from Exp7138. The
RAGTruth outcome opens after all model calls are frozen. SQL and relation text
remain untrusted model output, and only the Exp7138 read-only sandbox executes
queries.

Spec refs: REQ-VERIFY-7139 and SCENARIO-VERIFY-7139-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any
from urllib import request

from carnot.experiment_7138_v627_relational_fixture import (
    RELATION_SCHEMA,
    create_relation_database,
    execute_bounded_select,
    validate_artifact as validate_fixture_artifact,
    validate_relation_bundle,
)
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    llama_cpp_build_receipt,
    nvidia_smi_gpu_snapshot,
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    supervisor_contract,
)
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260908"
RANDOM_SEED = 7_139_202_609_08
RESULT_PATH = Path("results/experiment_7139_v627_symbolic_grounding_ab.json")
FIXTURE_PATH = Path("results/experiment_7138_v627_relational_fixture.json")
INFERENCE_SUBSTRATE = "live_llm_inference: three-family source-grounded comparison"
INFERENCE_SUBSTRATE_CLASS = "model_full_generation"
EXECUTION_VENUE = "host"
PREFERRED_QUANT = "Q4_K_M"
OUTPUT_TOKEN_LIMIT = 512
BOOTSTRAP_REPLICATES = 2_000
RAW_DIR = Path("results/raw/experiment_7139_v627_symbolic_grounding_ab")

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_NAMES = {
    REQUIRED_MODEL_IDS[0]: "Qwen3.6-35B-A3B",
    REQUIRED_MODEL_IDS[1]: "gemma-4-31B-it",
    REQUIRED_MODEL_IDS[2]: "gemma-4-26B-A4B-it",
}
MODEL_FAMILIES = {
    REQUIRED_MODEL_IDS[0]: "qwen3.6_moe",
    REQUIRED_MODEL_IDS[1]: "gemma4_dense",
    REQUIRED_MODEL_IDS[2]: "gemma4_moe",
}
ARM_IDS = ("direct", "self_verification", "relational_sql")
CALL_PLAN = (
    ("direct", 1),
    ("self_verification", 1),
    ("self_verification", 2),
    ("relational_sql", 1),
    ("relational_sql", 2),
)
REQUIRED_SQL_QUERY = (
    "SELECT relation_id, predicate, unknown_reason FROM grounded_relations "
    "WHERE object_type = 'unknown' LIMIT 32"
)

# This declaration names the required roster without touching the host cache.
# Cache resolution occurs only after the first artifact write.
MODEL_SPECS: list[JsonDict] = [
    {
        "name": MODEL_NAMES[model_id],
        "hf_id": model_id,
        "model_path": "",
        "gpu": index % 2,
        "preferred_quant": PREFERRED_QUANT,
        "resolution_method": "cached_sota_pair",
        "remote_allowed": False,
    }
    for index, model_id in enumerate(REQUIRED_MODEL_IDS)
]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_game_results",
    "MODEL_SPECS",
    "model_identity_rows",
    "model_load_receipts",
    "prompt_rows",
    "raw_output_rows",
    "parse_rows",
    "relation_rows",
    "sql_query_rows",
    "sql_execution_rows",
    "arm_rows",
    "source_family_rows",
    "model_family_rows",
    "metric_rows",
    "bootstrap_rows",
    "token_rows",
    "latency_rows",
    "cost_rows",
    "invalid_sql_rate",
    "unknown_rate",
    "useful_detection_rows",
    "exact_label_blinding_passed",
    "symbolic_grounding_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for each field exposes missing scientific evidence.",
    "preconditions_checked": "Exact gates stop inference when the frozen input or host is unsafe.",
    "run_date": "The fixed date identifies the requested execution window.",
    "inference_substrate": "The declaration states that three live local models generated the evidence.",
    "inference_substrate_class": "The class separates full generation from a blocked no-run.",
    "execution_venue": "The host venue prevents an unsupported remote execution claim.",
    "duration_s": "Measured wall time exposes interruption and implausible model work.",
    "source_artifact_hashes": "Byte hashes bind the result to its fixture, code, tests, and spec.",
    "rows": "The complete row projection prevents aggregate-only claims.",
    "per_game_results": "The empty field confirms this factual task makes no game claim.",
    "MODEL_SPECS": "The ordered local roster prevents model or quantization substitution.",
    "model_identity_rows": "File and template identities prove which embedded chat format was used.",
    "model_load_receipts": "A real canary generation proves each model loaded before the matrix.",
    "prompt_rows": "Exact prompts make matching and label blindness auditable.",
    "raw_output_rows": "Unedited model text prevents later reconstruction of convenient outputs.",
    "parse_rows": "Typed parse results preserve rejection without repair.",
    "relation_rows": "Validated relation bundles retain source-span grounding evidence.",
    "sql_query_rows": "Exact model-written SQL remains visible as untrusted input.",
    "sql_execution_rows": "Sandbox receipts distinguish execution from model assertion.",
    "arm_rows": "One scored row per opportunity prevents denominator drift.",
    "source_family_rows": "Per-source results expose a pooled source-family reversal.",
    "model_family_rows": "Per-model results expose a pooled model-family reversal.",
    "metric_rows": "Accuracy, discrimination, error, resource, and rejection metrics stay together.",
    "bootstrap_rows": "Paired fixture resampling measures uncertainty without breaking arm matching.",
    "token_rows": "Token totals reveal unequal generation budgets.",
    "latency_rows": "Latency totals retain the operational cost of each arm.",
    "cost_rows": "Local inference cost remains explicit instead of silently assumed.",
    "invalid_sql_rate": "Rejected queries measure the cost of untrusted SQL proposals.",
    "unknown_rate": "Abstentions remain visible and cannot become correct clean answers.",
    "useful_detection_rows": "Unique catches separate intervention value from shared detections.",
    "exact_label_blinding_passed": "True means external outcomes never entered a model prompt.",
    "symbolic_grounding_complete_score": "One means the planned matrix completed, independent of uplift.",
    "random_seed": "A fixed seed makes decoding and bootstrap sampling repeatable.",
    "reproducibility_checksum": "A canonical digest detects later receipt mutation.",
    "gate_check_summary": "The first failed check retains its expected and observed values.",
    "verifier_is_oracle": "False prevents SQL execution from becoming correctness authority.",
    "verdict_class": "A closed class separates completion from supported improvement.",
    "honest_verdict": "A matching prefix states the scientific result without overclaiming.",
}

_PROMPT_FORBIDDEN_RE = re.compile(
    r"(?i)(?:\b(?:clean|hallucinated|hallucination|label)\b|ragtruth|"
    r"source_info\.jsonl|response\.jsonl|sealed[_ -]?scorer|scorer[_ -]?view)"
)


def canonical_json(value: Any) -> str:
    """Serialize evidence with one stable byte spelling."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text so prompt and output changes remain visible."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:  # pragma: no cover - hashes live multi-GB files.
    """Hash exact file bytes without loading a GGUF into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every artifact field except the digest that stores the result."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_text(canonical_json(payload))


def gate_row(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Retain one exact expected-observed precondition decision."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": expected == observed if passed is None else bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Promote the first failure into the stable automation shape."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "failed_check": None,
            "expected_value": "all checks pass",
            "observed_value": "all checks pass",
            "passed": True,
        }
    return {
        "failed_check": failed.get("check"),
        "expected_value": deepcopy(failed.get("expected_value")),
        "observed_value": deepcopy(failed.get("observed_value")),
        "passed": False,
    }


def base_artifact(run_date: str) -> JsonDict:
    """Return the complete first-write shape without reading any input."""

    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "per_game_results": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_identity_rows": [],
        "model_load_receipts": [],
        "prompt_rows": [],
        "raw_output_rows": [],
        "parse_rows": [],
        "relation_rows": [],
        "sql_query_rows": [],
        "sql_execution_rows": [],
        "arm_rows": [],
        "source_family_rows": [],
        "model_family_rows": [],
        "metric_rows": [],
        "bootstrap_rows": [],
        "token_rows": [],
        "latency_rows": [],
        "cost_rows": [],
        "invalid_sql_rate": None,
        "unknown_rate": None,
        "useful_detection_rows": [],
        "exact_label_blinding_passed": False,
        "symbolic_grounding_complete_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "experiment_complete",
            "expected_value": True,
            "observed_value": False,
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_symbolic_grounding_comparison",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> Path:
    """Atomically replace one artifact so every checkpoint remains valid JSON."""

    return atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def initialize_artifact(path: Path | str, run_date: str) -> JsonDict:
    """Write the schema before any gate, model, GPU, server, or fixture check."""

    artifact = base_artifact(run_date)
    write_artifact(path, artifact)
    return artifact


def finish_blocked(
    artifact: Mapping[str, Any],
    path: Path | str,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Finish once with the first failed field and no arm generation rows."""

    blocked = deepcopy(dict(artifact))
    rows = [deepcopy(dict(row)) for row in checks]
    summary = _gate_summary(rows)
    failed_check = str(summary.get("failed_check") or "unknown_precondition")
    blocked.update(
        {
            "preconditions_checked": rows,
            "inference_substrate_class": "blocked_no_run",
            "duration_s": round(max(0.0, float(duration_s)), 6),
            "rows": [],
            "prompt_rows": [],
            "raw_output_rows": [],
            "parse_rows": [],
            "relation_rows": [],
            "sql_query_rows": [],
            "sql_execution_rows": [],
            "arm_rows": [],
            "source_family_rows": [],
            "model_family_rows": [],
            "metric_rows": [],
            "bootstrap_rows": [],
            "token_rows": [],
            "latency_rows": [],
            "cost_rows": [],
            "invalid_sql_rate": None,
            "unknown_rate": None,
            "useful_detection_rows": [],
            "exact_label_blinding_passed": False,
            "symbolic_grounding_complete_score": 0,
            "gate_check_summary": summary,
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failed_check}",
        }
    )
    blocked["reproducibility_checksum"] = artifact_checksum(blocked)
    write_artifact(path, blocked)
    return blocked


def resolve_model_specs(
    *,
    pair_provider: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the cached pair first, then fill the third required Q4 file."""

    pair = (
        pair_provider(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT, model_indices=(0, 2))
        or []
    )
    paths = {str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair}
    rows = []
    for index, model_id in enumerate(REQUIRED_MODEL_IDS):
        path = paths.get(model_id) or resolver(model_id, PREFERRED_QUANT) or ""
        rows.append(
            {
                "name": MODEL_NAMES[model_id],
                "hf_id": model_id,
                "model_path": str(path),
                "gpu": index % 2,
                "preferred_quant": PREFERRED_QUANT,
                "resolution_method": "cached_sota_pair",
                "remote_allowed": False,
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject roster, local path, quantization, or remote-fallback drift."""

    errors: list[str] = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_ids_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id") or "")
        path = str(row.get("model_path") or "")
        if not path:
            errors.append(f"model_path_missing:{model_id}")
        elif Path(path).suffix.lower() != ".gguf" or "mmproj" in Path(path).name.lower():
            errors.append(f"model_path_not_primary_gguf:{model_id}")
        if row.get("preferred_quant") != PREFERRED_QUANT:
            errors.append(f"model_quantization_mismatch:{model_id}")
        if row.get("resolution_method") != "cached_sota_pair":
            errors.append(f"model_resolution_mismatch:{model_id}")
        if row.get("remote_allowed") is not False:
            errors.append(f"remote_fallback_enabled:{model_id}")
    return list(dict.fromkeys(errors))


def _documents(row: Mapping[str, Any]) -> str:
    """Render the two model-visible documents without adding scorer state."""

    return f"SOURCE DOCUMENT:\n{row['source_text']}\n\nCANDIDATE RESPONSE:\n{row['response_text']}"


def _prompt_for(row: Mapping[str, Any], arm: str, pass_index: int) -> str:
    """Freeze one label-free instruction for each arm and pass."""

    documents = _documents(row)
    if arm == "direct":
        return (
            "Assess whether every claim in the candidate response is supported by the source "
            "document. Return only JSON with status supported, unsupported, or unknown and "
            "confidence from 0 to 1.\n\n" + documents
        )
    if arm == "self_verification" and pass_index == 1:
        return (
            "Draft a support assessment for the candidate response. Check each claim against "
            "the source. Return only JSON with status supported, unsupported, or unknown, "
            "confidence from 0 to 1, and brief evidence_notes.\n\n" + documents
        )
    if arm == "self_verification":
        return (
            "Recheck the prior assessment against the same source and response. Correct your own "
            "mistakes. Return only JSON with status supported, unsupported, or unknown and "
            "confidence from 0 to 1.\n\n"
            + documents
            + "\n\nPRIOR ASSESSMENT:\n{{FIRST_PASS_OUTPUT}}"
        )
    if arm == "relational_sql" and pass_index == 1:
        return (
            "Build a closed relation bundle for claims in the candidate response. Use a typed "
            "unknown object when the source does not state a response claim. Cite exact source "
            "spans for supported values and exact response spans for unstated values. Return only "
            "one JSON object with entities and relations. The schema is:\n"
            + canonical_json(RELATION_SCHEMA)
            + "\n\n"
            + documents
        )
    return (
        "Write the bounded SQL check for the proposed relation bundle. Select relation_id, "
        "predicate, and unknown_reason from grounded_relations where object_type equals "
        "'unknown', with LIMIT 32. Return only JSON with one query field.\n\n"
        + documents
        + "\n\nPROPOSED RELATION BUNDLE:\n{{FIRST_PASS_OUTPUT}}"
    )


def build_schedule(
    model_rows: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze the matched call opportunity before any external label opens."""

    schedule = []
    for spec in specs:
        model_id = str(spec["hf_id"])
        for row in model_rows:
            fixture_id = str(row["fixture_id"])
            for arm, pass_index in CALL_PLAN:
                call_id = f"{model_id}|{fixture_id}|{arm}|pass-{pass_index}"
                schedule.append(
                    {
                        "call_id": call_id,
                        "model_id": model_id,
                        "model_family": MODEL_FAMILIES.get(model_id),
                        "fixture_id": fixture_id,
                        "arm": arm,
                        "pass_index": pass_index,
                        "source_text": str(row["source_text"]),
                        "response_text": str(row["response_text"]),
                        "source_text_sha256": row["source_text_sha256"],
                        "response_text_sha256": row["response_text_sha256"],
                        "output_token_limit": OUTPUT_TOKEN_LIMIT,
                        "prompt": _prompt_for(row, arm, pass_index),
                    }
                )
    return schedule


def prompt_exposure_errors(prompts: Sequence[str]) -> list[str]:
    """Reject reserved scorer names or outcomes from model input text."""

    return [
        f"prompt_exposure:{index}"
        for index, prompt in enumerate(prompts)
        if _PROMPT_FORBIDDEN_RE.search(prompt)
    ]


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]], expected_fixture_ids: Sequence[str]
) -> list[str]:
    """Reject missing calls, budget drift, changed text, or label exposure."""

    errors: list[str] = []
    expected = {
        f"{model_id}|{fixture_id}|{arm}|pass-{pass_index}"
        for model_id in REQUIRED_MODEL_IDS
        for fixture_id in expected_fixture_ids
        for arm, pass_index in CALL_PLAN
    }
    observed = [str(row.get("call_id")) for row in schedule]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        errors.append("schedule_identity_mismatch")
    for row in schedule:
        call_id = str(row.get("call_id"))
        if row.get("output_token_limit") != OUTPUT_TOKEN_LIMIT:
            errors.append(f"output_limit_mismatch:{call_id}")
        if row.get("source_text_sha256") != sha256_text(str(row.get("source_text", ""))):
            errors.append(f"source_hash_mismatch:{call_id}")
        if row.get("response_text_sha256") != sha256_text(str(row.get("response_text", ""))):
            errors.append(f"response_hash_mismatch:{call_id}")
        if _PROMPT_FORBIDDEN_RE.search(str(row.get("prompt", ""))):
            errors.append(f"prompt_exposure:{call_id}")
    return list(dict.fromkeys(errors))


def _extract_json_object(text: str) -> Mapping[str, Any] | None:
    """Read the first JSON object without repairing model text."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def parse_decision_output(text: str) -> JsonDict:
    """Parse a direct decision while preserving invalid and unknown states."""

    value = _extract_json_object(text)
    if value is None:
        return {"status": "rejected", "reason": "decision_json_missing"}
    status = value.get("status")
    confidence = value.get("confidence")
    valid_confidence = (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 0.0 <= float(confidence) <= 1.0
    )
    if status not in {"supported", "unsupported", "unknown"} or not valid_confidence:
        return {"status": "rejected", "reason": "decision_schema_invalid"}
    prediction = {
        "supported": "clean",
        "unsupported": "hallucinated",
        "unknown": "unknown",
    }[str(status)]
    score = 1.0 - float(confidence) if status == "supported" else float(confidence)
    if status == "unknown":
        score = 0.5
    return {
        "status": "ok",
        "prediction": prediction,
        "hallucination_score": score,
    }


def parse_relation_output(text: str, documents: Mapping[str, str]) -> JsonDict:
    """Parse and validate one closed Exp7138 relation bundle without repair."""

    value = _extract_json_object(text)
    if value is None:
        return {"status": "rejected", "relation_errors": ["relation_json_missing"]}
    errors = validate_relation_bundle(value, documents)
    if errors:
        return {"status": "rejected", "relation_errors": errors}
    return {"status": "ok", "bundle": deepcopy(dict(value)), "relation_errors": []}


def parse_sql_output(text: str) -> JsonDict:
    """Accept only the one frozen Exp7138 SELECT query without repair."""

    value = _extract_json_object(text)
    if value is None or set(value) != {"query"}:
        return {"status": "rejected", "reason": "sql_json_invalid"}
    query = value.get("query")
    if query != REQUIRED_SQL_QUERY:
        return {"status": "rejected", "reason": "sql_query_not_frozen_select"}
    return {"status": "ok", "query": query}


def execute_sql_candidate(
    bundle: Mapping[str, Any], documents: Mapping[str, str], query: str
) -> JsonDict:
    """Execute untrusted relations and SQL only through the Exp7138 sandbox."""

    try:
        connection = create_relation_database(bundle, documents)
    except (TypeError, ValueError) as exc:
        return {"status": "rejected", "reason": f"invalid_relation_bundle:{exc}"}
    try:
        return execute_bounded_select(connection, query)
    finally:
        connection.close()


def reduce_sql_prediction(
    relation_errors: Sequence[str], query_parse: Mapping[str, Any], execution: Mapping[str, Any]
) -> JsonDict:
    """Map rejection to unknown and prevent unknown relations from becoming clean."""

    if relation_errors or query_parse.get("status") != "ok" or execution.get("status") != "ok":
        return {
            "prediction": "unknown",
            "hallucination_score": 0.5,
            "invalid_sql": True,
            "reason": "invalid_relation_or_query",
        }
    if int(execution.get("row_count", 0) or 0) > 0:
        return {
            "prediction": "hallucinated",
            "hallucination_score": 1.0,
            "invalid_sql": False,
            "reason": "unsupported_relation_returned",
        }
    return {
        "prediction": "clean",
        "hallucination_score": 0.0,
        "invalid_sql": False,
        "reason": "no_unsupported_relation_returned",
    }


def _auroc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Compute binary AUROC from pair comparisons so ties stay explicit."""

    positive = [
        float(row["hallucination_score"])
        for row in rows
        if row.get("truth_label") == "hallucinated"
    ]
    negative = [
        float(row["hallucination_score"]) for row in rows if row.get("truth_label") == "clean"
    ]
    if not positive or not negative:
        return None
    wins = sum(
        1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positive for neg in negative
    )
    return wins / (len(positive) * len(negative))


def binary_metric_row(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute outcome and resource metrics without dropping unknown rows."""

    count = len(rows)
    clean_rows = [row for row in rows if row.get("truth_label") == "clean"]
    hallucinated_rows = [row for row in rows if row.get("truth_label") == "hallucinated"]
    return {
        "row_count": count,
        "accuracy": sum(bool(row.get("correct")) for row in rows) / count if count else None,
        "auroc": _auroc(rows),
        "false_positive_rate": (
            sum(row.get("prediction") == "hallucinated" for row in clean_rows) / len(clean_rows)
            if clean_rows
            else None
        ),
        "hallucination_catch_rate": (
            sum(row.get("prediction") == "hallucinated" for row in hallucinated_rows)
            / len(hallucinated_rows)
            if hallucinated_rows
            else None
        ),
        "invalid_query_rate": (
            sum(bool(row.get("invalid_sql")) for row in rows) / count if count else None
        ),
        "unknown_rate": (
            sum(row.get("prediction") == "unknown" for row in rows) / count if count else None
        ),
        "mean_latency_s": (
            sum(float(row.get("latency_s", 0.0) or 0.0) for row in rows) / count if count else None
        ),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens", 0) or 0) for row in rows),
        "cost_usd": sum(float(row.get("cost_usd", 0.0) or 0.0) for row in rows),
    }


def build_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep overall, model, and source-family metrics as separate rows."""

    metrics = []
    model_ids = [
        model_id
        for model_id in REQUIRED_MODEL_IDS
        if any(row.get("model_id") == model_id for row in rows)
    ]
    source_families = sorted({str(row.get("source_family")) for row in rows})
    for arm in ARM_IDS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        metrics.append(
            {"scope": "overall", "scope_value": "all", "arm": arm, **binary_metric_row(arm_rows)}
        )
        for model_id in model_ids:
            selected = [row for row in arm_rows if row.get("model_id") == model_id]
            metrics.append(
                {
                    "scope": "model",
                    "scope_value": model_id,
                    "arm": arm,
                    **binary_metric_row(selected),
                }
            )
        for family in source_families:
            selected = [row for row in arm_rows if row.get("source_family") == family]
            metrics.append(
                {
                    "scope": "source_family",
                    "scope_value": family,
                    "arm": arm,
                    **binary_metric_row(selected),
                }
            )
    return metrics


def build_source_family_rows(metrics: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project source metrics with an explicit source-family key."""

    return [
        {
            **{
                key: deepcopy(value)
                for key, value in row.items()
                if key not in {"scope", "scope_value"}
            },
            "source_family": row.get("scope_value"),
        }
        for row in metrics
        if row.get("scope") == "source_family"
    ]


def build_model_family_rows(metrics: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project model metrics without pooling dense and routed families."""

    return [
        {
            **{
                key: deepcopy(value)
                for key, value in row.items()
                if key not in {"scope", "scope_value"}
            },
            "model_id": row.get("scope_value"),
            "model_family": MODEL_FAMILIES.get(str(row.get("scope_value"))),
        }
        for row in metrics
        if row.get("scope") == "model"
    ]


def build_token_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report token totals for every model and arm pair."""

    output = []
    for model_id in REQUIRED_MODEL_IDS:
        for arm in ARM_IDS:
            selected = [
                row for row in rows if row.get("model_id") == model_id and row.get("arm") == arm
            ]
            if not selected:
                continue
            prompt = sum(int(row.get("prompt_tokens", 0) or 0) for row in selected)
            completion = sum(int(row.get("completion_tokens", 0) or 0) for row in selected)
            output.append(
                {
                    "model_id": model_id,
                    "arm": arm,
                    "row_count": len(selected),
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                }
            )
    return output


def build_latency_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report total and mean latency for each arm."""

    output = []
    for arm in ARM_IDS:
        selected = [row for row in rows if row.get("arm") == arm]
        if not selected:
            continue
        total = sum(float(row.get("latency_s", 0.0) or 0.0) for row in selected)
        output.append(
            {
                "arm": arm,
                "row_count": len(selected),
                "total_latency_s": total,
                "mean_latency_s": total / len(selected),
            }
        )
    return output


def build_cost_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report explicit local cost for every model and arm pair."""

    output = []
    for model_id in REQUIRED_MODEL_IDS:
        for arm in ARM_IDS:
            selected = [
                row for row in rows if row.get("model_id") == model_id and row.get("arm") == arm
            ]
            if selected:
                output.append(
                    {
                        "model_id": model_id,
                        "arm": arm,
                        "row_count": len(selected),
                        "cost_usd": sum(float(row.get("cost_usd", 0.0) or 0.0) for row in selected),
                    }
                )
    return output


def _quantile(values: Sequence[float], probability: float) -> float:
    """Return one linearly interpolated quantile for a non-empty sample."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_scope_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, list[Mapping[str, Any]]]]:
    """List every scope that must retain a paired confidence interval."""

    scopes: list[tuple[str, str, list[Mapping[str, Any]]]] = [("overall", "all", list(rows))]
    for model_id in REQUIRED_MODEL_IDS:
        selected = [row for row in rows if row.get("model_id") == model_id]
        if selected:
            scopes.append(("model", model_id, selected))
    for family in sorted({str(row.get("source_family")) for row in rows}):
        scopes.append(
            (
                "source_family",
                family,
                [row for row in rows if row.get("source_family") == family],
            )
        )
    return scopes


def build_bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int, n_boot: int
) -> list[JsonDict]:
    """Bootstrap SQL-minus-control accuracy by whole frozen fixture ID."""

    import random

    output = []
    for scope, scope_value, selected in _bootstrap_scope_rows(rows):
        fixture_ids = sorted({str(row.get("fixture_id")) for row in selected})
        by_fixture_arm: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in selected:
            by_fixture_arm[(str(row.get("fixture_id")), str(row.get("arm")))].append(
                float(bool(row.get("correct")))
            )
        for comparator in ("direct", "self_verification"):
            differences = []
            for fixture_id in fixture_ids:
                sql = by_fixture_arm[(fixture_id, "relational_sql")]
                control = by_fixture_arm[(fixture_id, comparator)]
                if sql and control:
                    differences.append(
                        (fixture_id, sum(sql) / len(sql) - sum(control) / len(control))
                    )
            point = sum(value for _fixture_id, value in differences) / len(differences)
            rng = random.Random(f"{seed}|{scope}|{scope_value}|{comparator}")
            samples = [
                sum(rng.choice(differences)[1] for _ in differences) / len(differences)
                for _replicate in range(n_boot)
            ]
            output.append(
                {
                    "scope": scope,
                    "scope_value": scope_value,
                    "comparator": comparator,
                    "sql_minus_control_accuracy": point,
                    "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
                    "n_boot": n_boot,
                    "resampling_unit": "fixture_id",
                    "paired": True,
                }
            )
    return output


def build_useful_detection_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Count SQL catches that both matched controls missed for each model."""

    output = []
    for model_id in REQUIRED_MODEL_IDS:
        selected = [row for row in rows if row.get("model_id") == model_id]
        if not selected:
            continue
        by_key = {(str(row.get("fixture_id")), str(row.get("arm"))): row for row in selected}
        fixture_ids = sorted({str(row.get("fixture_id")) for row in selected})
        useful = 0
        harmful = 0
        for fixture_id in fixture_ids:
            sql = by_key[(fixture_id, "relational_sql")]
            direct = by_key[(fixture_id, "direct")]
            self_check = by_key[(fixture_id, "self_verification")]
            if (
                sql.get("truth_label") == "hallucinated"
                and sql.get("prediction") == "hallucinated"
                and direct.get("prediction") != "hallucinated"
                and self_check.get("prediction") != "hallucinated"
            ):
                useful += 1
            if (
                sql.get("truth_label") == "clean"
                and sql.get("prediction") == "hallucinated"
                and direct.get("prediction") != "hallucinated"
                and self_check.get("prediction") != "hallucinated"
            ):
                harmful += 1
        output.append(
            {
                "model_id": model_id,
                "sql_unique_hallucination_catches": useful,
                "sql_unique_false_positives": harmful,
            }
        )
    return output


def select_verdict(
    *, matrix_complete: bool, bootstrap_rows: Sequence[Mapping[str, Any]], family_loss: bool
) -> tuple[int, str, str]:
    """Separate matrix completion from supported external-label improvement."""

    if not matrix_complete:
        return 0, "partial", "partial_incomplete_symbolic_grounding_matrix"
    overall = {
        str(row.get("comparator")): row for row in bootstrap_rows if row.get("scope") == "overall"
    }
    supported = all(
        comparator in overall
        and isinstance(overall[comparator].get("ci95"), list)
        and float(overall[comparator]["ci95"][0]) > 0.0
        for comparator in ("direct", "self_verification")
    )
    if supported and not family_loss:
        return 1, "positive", "positive_sql_grounding_external_label_uplift_supported"
    return 1, "null", "null_symbolic_grounding_complete_without_supported_uplift"


def _family_loss(metrics: Sequence[Mapping[str, Any]]) -> bool:
    """Return true when SQL accuracy loses within any model or source family."""

    grouped = {
        (str(row.get("scope")), str(row.get("scope_value")), str(row.get("arm"))): row
        for row in metrics
    }
    scopes = {
        (str(row.get("scope")), str(row.get("scope_value")))
        for row in metrics
        if row.get("scope") in {"model", "source_family"}
    }
    return any(
        float(grouped[(scope, value, "relational_sql")]["accuracy"])
        < float(grouped[(scope, value, comparator)]["accuracy"])
        for scope, value in scopes
        for comparator in ("direct", "self_verification")
    )


def _expected_call_ids(fixture_ids: Sequence[str]) -> set[str]:
    """Build the exact call identity set without needing model-visible text."""

    return {
        f"{model_id}|{fixture_id}|{arm}|pass-{pass_index}"
        for model_id in REQUIRED_MODEL_IDS
        for fixture_id in fixture_ids
        for arm, pass_index in CALL_PLAN
    }


def _matrix_complete(
    *,
    arm_rows: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]],
    raw_output_rows: Sequence[Mapping[str, Any]],
    parse_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    sql_query_rows: Sequence[Mapping[str, Any]],
    sql_execution_rows: Sequence[Mapping[str, Any]],
    model_identity_rows: Sequence[Mapping[str, Any]],
    model_load_receipts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    fixture_ids: Sequence[str],
) -> bool:
    """Require every matched call, SQL cell, scored arm, model, and gate."""

    calls = _expected_call_ids(fixture_ids)
    cells = {
        f"{model_id}|{fixture_id}" for model_id in REQUIRED_MODEL_IDS for fixture_id in fixture_ids
    }
    arms = {
        f"{model_id}|{fixture_id}|{arm}"
        for model_id in REQUIRED_MODEL_IDS
        for fixture_id in fixture_ids
        for arm in ARM_IDS
    }
    receipt_sets = [
        {str(row.get("call_id")) for row in prompt_rows},
        {str(row.get("call_id")) for row in raw_output_rows},
        {str(row.get("call_id")) for row in parse_rows},
    ]
    cell_sets = [
        {str(row.get("cell_id")) for row in relation_rows},
        {str(row.get("cell_id")) for row in sql_query_rows},
        {str(row.get("cell_id")) for row in sql_execution_rows},
    ]
    arm_ids = {str(row.get("arm_row_id")) for row in arm_rows}
    identity_models = {
        str(row.get("model_id")) for row in model_identity_rows if row.get("passed") is True
    }
    load_models = {
        str(row.get("model_id")) for row in model_load_receipts if row.get("passed") is True
    }
    return (
        all(row.get("passed") is True for row in preconditions_checked)
        and all(row.get("terminal_state") != "failed" for row in raw_output_rows)
        and len(prompt_rows) == len(raw_output_rows) == len(parse_rows) == len(calls)
        and all(receipts == calls for receipts in receipt_sets)
        and len(relation_rows) == len(sql_query_rows) == len(sql_execution_rows) == len(cells)
        and all(receipts == cells for receipts in cell_sets)
        and len(arm_rows) == len(arms)
        and arm_ids == arms
        and identity_models == set(REQUIRED_MODEL_IDS)
        and load_models == set(REQUIRED_MODEL_IDS)
    )


def _blinding_passed(
    prompt_rows: Sequence[Mapping[str, Any]], raw_output_rows: Sequence[Mapping[str, Any]]
) -> bool:
    """Confirm scorer fields occur in neither prompts nor raw receipt metadata."""

    prompt_errors = prompt_exposure_errors([str(row.get("prompt", "")) for row in prompt_rows])
    forbidden_keys = {"truth_label", "response_label", "span_labels", "sealed_scorer_rows"}
    return not prompt_errors and all(
        not forbidden_keys.intersection(row) for row in raw_output_rows
    )


def finalize_artifact(
    artifact: Mapping[str, Any],
    *,
    arm_rows: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]],
    raw_output_rows: Sequence[Mapping[str, Any]],
    parse_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    sql_query_rows: Sequence[Mapping[str, Any]],
    sql_execution_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    model_identity_rows: Sequence[Mapping[str, Any]],
    model_load_receipts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    duration_s: float,
    expected_fixture_ids: Sequence[str],
    n_boot: int = BOOTSTRAP_REPLICATES,
) -> JsonDict:
    """Score frozen outputs and build one complete terminal artifact."""

    rows = [deepcopy(dict(row)) for row in arm_rows]
    metrics = build_metric_rows(rows)
    bootstrap = build_bootstrap_rows(rows, seed=RANDOM_SEED, n_boot=n_boot)
    complete = _matrix_complete(
        arm_rows=rows,
        prompt_rows=prompt_rows,
        raw_output_rows=raw_output_rows,
        parse_rows=parse_rows,
        relation_rows=relation_rows,
        sql_query_rows=sql_query_rows,
        sql_execution_rows=sql_execution_rows,
        model_identity_rows=model_identity_rows,
        model_load_receipts=model_load_receipts,
        preconditions_checked=preconditions_checked,
        fixture_ids=expected_fixture_ids,
    )
    score, verdict_class, verdict = select_verdict(
        matrix_complete=complete,
        bootstrap_rows=bootstrap,
        family_loss=_family_loss(metrics),
    )
    sql_rows = [row for row in rows if row.get("arm") == "relational_sql"]
    result = deepcopy(dict(artifact))
    result.update(
        {
            "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "duration_s": round(max(0.0, float(duration_s)), 6),
            "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
            "rows": deepcopy(rows),
            "per_game_results": [],
            "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
            "model_identity_rows": [deepcopy(dict(row)) for row in model_identity_rows],
            "model_load_receipts": [deepcopy(dict(row)) for row in model_load_receipts],
            "prompt_rows": [deepcopy(dict(row)) for row in prompt_rows],
            "raw_output_rows": [deepcopy(dict(row)) for row in raw_output_rows],
            "parse_rows": [deepcopy(dict(row)) for row in parse_rows],
            "relation_rows": [deepcopy(dict(row)) for row in relation_rows],
            "sql_query_rows": [deepcopy(dict(row)) for row in sql_query_rows],
            "sql_execution_rows": [deepcopy(dict(row)) for row in sql_execution_rows],
            "arm_rows": deepcopy(rows),
            "source_family_rows": build_source_family_rows(metrics),
            "model_family_rows": build_model_family_rows(metrics),
            "metric_rows": metrics,
            "bootstrap_rows": bootstrap,
            "token_rows": build_token_rows(rows),
            "latency_rows": build_latency_rows(rows),
            "cost_rows": build_cost_rows(rows),
            "invalid_sql_rate": (
                sum(bool(row.get("invalid_sql")) for row in sql_rows) / len(sql_rows)
                if sql_rows
                else None
            ),
            "unknown_rate": (
                sum(row.get("prediction") == "unknown" for row in rows) / len(rows)
                if rows
                else None
            ),
            "useful_detection_rows": build_useful_detection_rows(rows),
            "exact_label_blinding_passed": _blinding_passed(prompt_rows, raw_output_rows),
            "symbolic_grounding_complete_score": score,
            "gate_check_summary": _gate_summary(preconditions_checked),
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
        }
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def _load_artifact_value(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any] | None:
    """Load a path or return an existing object for cold validation."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, Path):
        if not value.is_file():
            return None
        try:
            loaded = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, Mapping) else None
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, Mapping) else None
    return None


def validate_artifact(value: Mapping[str, Any] | str | Path | object) -> list[str]:
    """Cold-check shape, receipts, metrics, blinding, verdict, and checksum."""

    artifact = _load_artifact_value(value)  # type: ignore[arg-type]
    if artifact is None:
        return ["artifact_not_object"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    extra = [field for field in artifact if field not in REQUIRED_ARTIFACT_FIELDS]
    if missing or extra:
        return [f"artifact fields mismatch missing={missing} extra={extra}"]

    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle mismatch")
    if artifact.get("per_game_results") != []:
        errors.append("per_game_results mismatch")
    if (
        not isinstance(artifact.get("duration_s"), (int, float))
        or float(artifact.get("duration_s", -1)) < 0
    ):
        errors.append("duration_s invalid")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(f"{verdict_class}_"):
        errors.append("honest_verdict mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")

    checks = [row for row in artifact.get("preconditions_checked", []) if isinstance(row, Mapping)]
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None or verdict_class == "blocked":
        if failed is None:
            errors.append("blocked precondition missing")
        elif artifact.get("gate_check_summary") != _gate_summary(checks):
            errors.append("blocked gate_check_summary mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked inference_substrate_class mismatch")
        if verdict_class != "blocked":
            errors.append("blocked verdict_class mismatch")
        if any(artifact.get(field) for field in ("prompt_rows", "raw_output_rows", "arm_rows")):
            errors.append("blocked arm generation rows present")
        if artifact.get("symbolic_grounding_complete_score") != 0:
            errors.append("blocked completion score mismatch")
        return list(dict.fromkeys(errors))

    specs = list(artifact.get("MODEL_SPECS", []))
    errors.extend(model_spec_errors(specs))
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class mismatch")
    if artifact.get("rows") != artifact.get("arm_rows"):
        errors.append("rows mismatch")

    prompt_rows = list(artifact.get("prompt_rows", []))
    raw_rows = list(artifact.get("raw_output_rows", []))
    parse_rows = list(artifact.get("parse_rows", []))
    for row in prompt_rows:
        if row.get("prompt_sha256") != sha256_text(str(row.get("prompt", ""))):
            errors.append(f"prompt_hash mismatch:{row.get('call_id')}")
    for row in raw_rows:
        if row.get("raw_output_sha256") != sha256_text(str(row.get("raw_output", ""))):
            errors.append(f"raw_output_hash mismatch:{row.get('call_id')}")
    if not _blinding_passed(prompt_rows, raw_rows):
        errors.append("exact_label_blinding mismatch")

    arm_rows = list(artifact.get("arm_rows", []))
    fixture_ids = sorted({str(row.get("fixture_id")) for row in arm_rows})
    complete = _matrix_complete(
        arm_rows=arm_rows,
        prompt_rows=prompt_rows,
        raw_output_rows=raw_rows,
        parse_rows=parse_rows,
        relation_rows=list(artifact.get("relation_rows", [])),
        sql_query_rows=list(artifact.get("sql_query_rows", [])),
        sql_execution_rows=list(artifact.get("sql_execution_rows", [])),
        model_identity_rows=list(artifact.get("model_identity_rows", [])),
        model_load_receipts=list(artifact.get("model_load_receipts", [])),
        preconditions_checked=checks,
        fixture_ids=fixture_ids,
    )
    expected_metrics = build_metric_rows(arm_rows)
    if artifact.get("metric_rows") != expected_metrics:
        errors.append("metric_rows mismatch")
    if artifact.get("source_family_rows") != build_source_family_rows(expected_metrics):
        errors.append("source_family_rows mismatch")
    if artifact.get("model_family_rows") != build_model_family_rows(expected_metrics):
        errors.append("model_family_rows mismatch")
    if artifact.get("token_rows") != build_token_rows(arm_rows):
        errors.append("token_rows mismatch")
    if artifact.get("latency_rows") != build_latency_rows(arm_rows):
        errors.append("latency_rows mismatch")
    if artifact.get("cost_rows") != build_cost_rows(arm_rows):
        errors.append("cost_rows mismatch")
    if artifact.get("useful_detection_rows") != build_useful_detection_rows(arm_rows):
        errors.append("useful_detection_rows mismatch")

    sql_rows = [row for row in arm_rows if row.get("arm") == "relational_sql"]
    invalid_rate = (
        sum(bool(row.get("invalid_sql")) for row in sql_rows) / len(sql_rows) if sql_rows else None
    )
    unknown_rate = (
        sum(row.get("prediction") == "unknown" for row in arm_rows) / len(arm_rows)
        if arm_rows
        else None
    )
    if artifact.get("invalid_sql_rate") != invalid_rate:
        errors.append("invalid_sql_rate mismatch")
    if artifact.get("unknown_rate") != unknown_rate:
        errors.append("unknown_rate mismatch")

    bootstrap_rows = list(artifact.get("bootstrap_rows", []))
    n_boot = int(bootstrap_rows[0].get("n_boot", 0)) if bootstrap_rows else 0
    expected_bootstrap = (
        build_bootstrap_rows(arm_rows, seed=RANDOM_SEED, n_boot=n_boot)
        if arm_rows and n_boot > 0
        else []
    )
    if bootstrap_rows != expected_bootstrap:
        errors.append("bootstrap_rows mismatch")
    score, expected_class, expected_verdict = select_verdict(
        matrix_complete=complete,
        bootstrap_rows=expected_bootstrap,
        family_loss=_family_loss(expected_metrics),
    )
    if artifact.get("symbolic_grounding_complete_score") != score:
        errors.append("symbolic_grounding_complete_score mismatch")
    if verdict_class != expected_class or artifact.get("honest_verdict") != expected_verdict:
        errors.append("terminal verdict mismatch")
    if artifact.get("gate_check_summary") != _gate_summary(checks):
        errors.append("gate_check_summary mismatch")
    if artifact.get("exact_label_blinding_passed") is not _blinding_passed(prompt_rows, raw_rows):
        errors.append("exact_label_blinding_passed mismatch")
    return list(dict.fromkeys(errors))


def _progress(phase: str, **fields: Any) -> None:  # pragma: no cover - CLI progress boundary.
    """Print one compact progress row so long inference never appears silent."""

    print(canonical_json({"phase": phase, **fields}), flush=True)


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live repository bytes.
    """Hash the exact fixture, code, tests, spec, and named context inputs."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("research-references.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("python/carnot/experiment_7138_v627_relational_fixture.py"),
        Path("python/carnot/experiment_7139_v627_symbolic_grounding_ab.py"),
        Path("scripts/experiments/experiment_7139_v627_symbolic_grounding_ab.py"),
        Path("tests/python/test_experiment_7139_v627_symbolic_grounding_ab.py"),
        Path("results/experiment_7138_v627_relational_fixture.json"),
        Path("results/experiment_3670_facts_row_real_benchmark.json"),
        Path("results/experiment_5163_mmlu_pro_verifier_rescale_v473.json"),
    )
    return {
        str(path): sha256_file(root / path) if (root / path).is_file() else None for path in paths
    }


def _load_fixture(path: Path) -> tuple[JsonDict | None, list[JsonDict]]:  # pragma: no cover
    """Load the frozen artifact and fail on its exact structured readiness gate."""

    checks = []
    exists = path.is_file()
    checks.append(
        gate_row(
            "fixture_artifact_path",
            "readable_file",
            "readable_file" if exists else "missing_or_unreadable",
            exists,
        )
    )
    if not exists:
        return None, checks
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        checks.append(gate_row("fixture_artifact_json", "object", type(exc).__name__, False))
        return None, checks
    if not isinstance(value, dict):
        checks.append(gate_row("fixture_artifact_json", "object", type(value).__name__, False))
        return None, checks
    fixture_errors = validate_fixture_artifact(value)
    checks.append(gate_row("fixture_artifact_validation", [], fixture_errors, not fixture_errors))
    ready = value.get("source_grounding_fixture_ready_score")
    checks.append(
        gate_row(
            "source_grounding_fixture_ready_score",
            1,
            ready,
            type(ready) is int and ready == 1,
        )
    )
    checks.append(
        gate_row(
            "fixture_row_count",
            72,
            value.get("fixture_row_count"),
            value.get("fixture_row_count") == 72,
        )
    )
    checks.append(
        gate_row(
            "fixture_label_exposure_count",
            0,
            value.get("label_exposure_count"),
            value.get("label_exposure_count") == 0,
        )
    )
    return value, checks


def _model_file_receipts(
    specs: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Bind every cached Q4 path to bytes, revision, and embedded template metadata."""

    resolved: list[JsonDict] = []
    identities: list[JsonDict] = []
    checks: list[JsonDict] = []
    spec_errors = model_spec_errors(specs)
    checks.append(gate_row("MODEL_SPECS", [], spec_errors, not spec_errors))
    if spec_errors:
        return [deepcopy(dict(row)) for row in specs], identities, checks
    for spec in specs:
        row = deepcopy(dict(spec))
        path = Path(str(row["model_path"]))
        exists = path.is_file()
        observed = {
            "path": str(path),
            "exists": exists,
            "q4_k_m": "q4_k_m" in path.name.lower(),
            "size_bytes": path.stat().st_size if exists else 0,
        }
        passed = exists and observed["q4_k_m"] and int(observed["size_bytes"]) > 0
        checks.append(
            gate_row(
                f"cached_q4_file:{row['hf_id']}",
                {"exists": True, "q4_k_m": True, "size_bytes_gt": 0},
                observed,
                passed,
            )
        )
        if not passed:
            resolved.append(row)
            continue
        metadata = read_gguf_metadata(path)
        file_hash = sha256_file(path)
        row.update(
            {
                "loaded_path": str(path.resolve()),
                "revision": snapshot_revision(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_hash,
                "chat_template_source": "embedded_gguf",
                "chat_template_sha256": metadata.get("chat_template_sha256"),
            }
        )
        identity = {
            "model_id": row["hf_id"],
            "loaded_path": row["loaded_path"],
            "revision": row["revision"],
            "size_bytes": row["size_bytes"],
            "sha256": file_hash,
            "template_source": "embedded_gguf",
            "template_sha256": metadata.get("chat_template_sha256"),
            "template_metadata_keys": metadata.get("metadata_keys", []),
            "template_detail": metadata.get("tokenizer_detail"),
            "passed": bool(metadata.get("chat_template_present")),
        }
        identities.append(identity)
        checks.append(
            gate_row(
                f"embedded_chat_template:{row['hf_id']}",
                {"source": "embedded_gguf", "present": True},
                {
                    "source": identity["template_source"],
                    "present": identity["passed"],
                    "sha256": identity["template_sha256"],
                },
                identity["passed"],
            )
        )
        resolved.append(row)
    return resolved, identities, checks


def _host_runtime_receipt() -> tuple[JsonDict, JsonDict, list[JsonDict]]:  # pragma: no cover
    """Check two idle GPUs, CUDA llama.cpp, and the native server binary."""

    gpu = nvidia_smi_gpu_snapshot()
    backend = llama_cpp_build_receipt()
    devices = list(gpu.get("devices", []))
    external_apps = [row for row in gpu.get("compute_apps", []) if not row.get("owned_by_task")]
    checks = [
        gate_row(
            "gpu_available",
            {"gpu_count": 2, "external_compute_apps": 0},
            {
                "query_ok": gpu.get("ok"),
                "gpu_count": gpu.get("gpu_count"),
                "devices": devices,
                "external_compute_apps": external_apps,
            },
            gpu.get("ok") is True and len(devices) == 2 and not external_apps,
        ),
        gate_row(
            "llama_cpp_backend",
            {"python_gpu_offload": True},
            {
                "python_version": backend.get("llama_cpp_python_version"),
                "python_gpu_offload": backend.get("llama_cpp_python_gpu_offload"),
                "system_info": backend.get("llama_cpp_python_system_info"),
            },
            backend.get("llama_cpp_python_gpu_offload") is True,
        ),
        gate_row(
            "native_llama_server",
            {"exists": True, "cuda_build": True, "version_returncode": 0},
            {
                "path": backend.get("native_llama_server_path"),
                "exists": backend.get("native_llama_server_exists"),
                "cuda_build": backend.get("native_llama_server_cuda_build"),
                "version": backend.get("native_llama_server_version"),
                "version_returncode": backend.get("native_llama_server_version_returncode"),
            },
            backend.get("native_llama_server_exists") is True
            and backend.get("native_llama_server_cuda_build") is True
            and backend.get("native_llama_server_version_returncode") == 0,
        ),
    ]
    return gpu, backend, checks


def _free_port() -> int:  # pragma: no cover - live socket boundary.
    """Reserve one loopback port number for the next owned server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(
    server: Path, model_path: Path, port: int, *, n_ctx: int
) -> list[str]:  # pragma: no cover
    """Use the established offline two-GPU llama-server command."""

    return [
        str(server),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(n_ctx),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--fit",
        "off",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-webui",
        "--log-verbosity",
        "3",
    ]


def _server_request(
    port: int, prompt: str, *, max_tokens: int, seed: int
) -> JsonDict:  # pragma: no cover
    """Send one deterministic chat request and retain the complete response."""

    payload = {
        "messages": [
            {"role": "system", "content": "Return only the requested JSON object."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": int(seed),
        "cache_prompt": False,
    }
    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(http_request, timeout=600.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = list(body.get("choices") or [{}])
    message = dict(choices[0].get("message") or {})
    usage = dict(body.get("usage") or {})
    raw = str(message.get("content") or "")
    return {
        "raw_output": raw,
        "reasoning_output": str(message.get("reasoning_content") or message.get("reasoning") or ""),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "latency_s": time.perf_counter() - started,
        "raw_response": body,
    }


def _call_seed(call_id: str) -> int:
    """Derive one stable positive request seed from the frozen call ID."""

    digest = hashlib.sha256(f"{RANDOM_SEED}|{call_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _preflight_canaries(
    specs: Sequence[Mapping[str, Any]],
    *,
    server_path: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Require one real template-rendered generation from every model."""

    receipts = []
    checks = []
    canary_row = {
        "source_text": "Paris is in France.",
        "response_text": "Paris is in France.",
    }
    prompt = _prompt_for(canary_row, "direct", 1)
    for spec in specs:
        model_id = str(spec["hf_id"])
        _progress("preflight_model_request", model_id=model_id)
        port = _free_port()
        command = _server_command(server_path, Path(str(spec["model_path"])), port, n_ctx=4_096)
        contract = supervisor_contract(
            outer_deadline_s=900,
            health_timeout_s=420,
            token_timeout_s=600,
            cleanup_grace_s=30,
            kill_after_cleanup_timeout_s=10,
            retry_budget=0,
            endurance_interval_s=0,
            endurance_sample_count=1,
        )
        supervisor = NativeLlamaServerSupervisor(command, raw_dir / "preflight", contract)
        started = time.perf_counter()
        health: JsonDict = {"ok": False, "classification": "not_started"}
        response_row: JsonDict = {}
        error = None
        during_gpu: JsonDict = {}
        try:
            identity = supervisor.launch()
            health = supervisor.wait_for_health()
            during_gpu = nvidia_smi_gpu_snapshot()
            if health.get("ok") is not True:
                raise RuntimeError(f"server_health:{health.get('classification')}")
            response_row = _server_request(
                port,
                prompt,
                max_tokens=32,
                seed=_call_seed(f"{model_id}|preflight"),
            )
        except Exception as exc:  # noqa: BLE001 - the exact preflight error must survive.
            identity = supervisor.identity or {}
            error = f"{type(exc).__name__}: {exc}"
        cleanup = supervisor.cleanup()
        parsed = parse_decision_output(str(response_row.get("raw_output", "")))
        passed = (
            health.get("ok") is True
            and parsed.get("status") == "ok"
            and cleanup.get("leak_free") is True
        )
        receipt = {
            "model_id": model_id,
            "loaded_path": spec.get("loaded_path", spec.get("model_path")),
            "revision": spec.get("revision"),
            "size_bytes": spec.get("size_bytes"),
            "sha256": spec.get("sha256"),
            "template_source": spec.get("chat_template_source"),
            "template_sha256": spec.get("chat_template_sha256"),
            "device": during_gpu.get("devices", []),
            "backend": {
                "kind": "native_llama_server",
                "path": str(server_path),
                "command": command,
                "pid": identity.get("pid"),
            },
            "health": health,
            "canary_prompt": prompt,
            "canary_prompt_sha256": sha256_text(prompt),
            "canary_raw_output": response_row.get("raw_output", ""),
            "canary_raw_output_sha256": sha256_text(str(response_row.get("raw_output", ""))),
            "canary_parse": parsed,
            "prompt_tokens": response_row.get("prompt_tokens", 0),
            "completion_tokens": response_row.get("completion_tokens", 0),
            "latency_s": response_row.get("latency_s", 0.0),
            "server_log_path": str(supervisor.log_path),
            "server_log_tail": supervisor.stderr_tail(),
            "cleanup": cleanup,
            "duration_s": time.perf_counter() - started,
            "error": error,
            "passed": passed,
        }
        receipts.append(receipt)
        checks.append(
            gate_row(
                f"real_generation_canary:{model_id}",
                {"health": True, "parsed": True, "cleanup_leak_free": True},
                {
                    "health": health.get("ok"),
                    "parsed": parsed.get("status") == "ok",
                    "cleanup_leak_free": cleanup.get("leak_free"),
                    "error": error,
                },
                passed,
            )
        )
        _progress(
            "preflight_model_response",
            model_id=model_id,
            passed=passed,
            completion_tokens=response_row.get("completion_tokens", 0),
        )
        if not passed:
            break
    return receipts, checks


def _run_matrix(
    schedule: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    *,
    server_path: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Generate every scheduled call and keep failures as durable raw rows."""

    prompt_rows: list[JsonDict] = []
    raw_rows: list[JsonDict] = []
    first_outputs: dict[tuple[str, str, str], str] = {}
    for spec in specs:
        model_id = str(spec["hf_id"])
        model_schedule = [row for row in schedule if row.get("model_id") == model_id]
        port = _free_port()
        command = _server_command(server_path, Path(str(spec["model_path"])), port, n_ctx=16_384)
        contract = supervisor_contract(
            outer_deadline_s=7_200,
            health_timeout_s=420,
            token_timeout_s=600,
            cleanup_grace_s=30,
            kill_after_cleanup_timeout_s=10,
            retry_budget=0,
            endurance_interval_s=0,
            endurance_sample_count=1,
        )
        supervisor = NativeLlamaServerSupervisor(command, raw_dir / "matrix", contract)
        server_error = None
        try:
            _progress("matrix_model_load", model_id=model_id, request_count=len(model_schedule))
            supervisor.launch()
            health = supervisor.wait_for_health()
            if health.get("ok") is not True:
                raise RuntimeError(f"server_health:{health.get('classification')}")
        except Exception as exc:  # noqa: BLE001 - every planned call still gets a receipt.
            server_error = f"{type(exc).__name__}: {exc}"
        for scheduled in model_schedule:
            call_id = str(scheduled["call_id"])
            key = (model_id, str(scheduled["fixture_id"]), str(scheduled["arm"]))
            prompt = str(scheduled["prompt"])
            if int(scheduled["pass_index"]) == 2:
                prompt = prompt.replace("{{FIRST_PASS_OUTPUT}}", first_outputs.get(key, ""))
            prompt_rows.append(
                {
                    "call_id": call_id,
                    "model_id": model_id,
                    "fixture_id": scheduled["fixture_id"],
                    "arm": scheduled["arm"],
                    "pass_index": scheduled["pass_index"],
                    "output_token_limit": scheduled["output_token_limit"],
                    "source_text_sha256": scheduled["source_text_sha256"],
                    "response_text_sha256": scheduled["response_text_sha256"],
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                }
            )
            _progress(
                "model_request",
                model_id=model_id,
                fixture_id=scheduled["fixture_id"],
                arm=scheduled["arm"],
                pass_index=scheduled["pass_index"],
            )
            response_row: JsonDict = {}
            error = server_error
            if error is None:
                try:
                    response_row = _server_request(
                        port,
                        prompt,
                        max_tokens=int(scheduled["output_token_limit"]),
                        seed=_call_seed(call_id),
                    )
                except Exception as exc:  # noqa: BLE001 - raw failure remains a scored abstention.
                    error = f"{type(exc).__name__}: {exc}"
            raw_output = str(response_row.get("raw_output", ""))
            raw_row = {
                "call_id": call_id,
                "model_id": model_id,
                "fixture_id": scheduled["fixture_id"],
                "arm": scheduled["arm"],
                "pass_index": scheduled["pass_index"],
                "raw_output": raw_output,
                "raw_output_sha256": sha256_text(raw_output),
                "reasoning_output": response_row.get("reasoning_output", ""),
                "raw_response": response_row.get("raw_response", {}),
                "raw_response_sha256": sha256_text(
                    canonical_json(response_row.get("raw_response", {}))
                ),
                "prompt_tokens": response_row.get("prompt_tokens", 0),
                "completion_tokens": response_row.get("completion_tokens", 0),
                "latency_s": response_row.get("latency_s", 0.0),
                "terminal_state": "complete" if error is None else "failed",
                "error": error,
            }
            raw_rows.append(raw_row)
            if int(scheduled["pass_index"]) == 1:
                first_outputs[key] = raw_output
            _progress(
                "model_response",
                model_id=model_id,
                fixture_id=scheduled["fixture_id"],
                arm=scheduled["arm"],
                pass_index=scheduled["pass_index"],
                completion_tokens=raw_row["completion_tokens"],
                terminal_state=raw_row["terminal_state"],
            )
        cleanup = supervisor.cleanup()
        _progress(
            "matrix_model_cleanup",
            model_id=model_id,
            leak_free=cleanup.get("leak_free"),
        )
        gc.collect()
    return prompt_rows, raw_rows


def _score_frozen_outputs(
    *,
    model_views: Sequence[Mapping[str, Any]],
    fixture_rows: Sequence[Mapping[str, Any]],
    sealed_rows: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
]:  # pragma: no cover - full live receipt reduction.
    """Parse all raw text, run SQL, then join the only external outcomes."""

    model_by_id = {str(row["fixture_id"]): row for row in model_views}
    family_by_id = {str(row["fixture_id"]): str(row["source_family"]) for row in fixture_rows}
    truth_by_id = {str(row["fixture_id"]): str(row["response_label"]) for row in sealed_rows}
    raw_by_id = {str(row["call_id"]): row for row in raw_rows}
    prompt_by_id = {str(row["call_id"]): row for row in prompt_rows}
    parse_rows: list[JsonDict] = []
    parsed: dict[str, JsonDict] = {}
    for call_id, raw in raw_by_id.items():
        arm = str(raw["arm"])
        pass_index = int(raw["pass_index"])
        fixture_id = str(raw["fixture_id"])
        documents = {
            "source": str(model_by_id[fixture_id]["source_text"]),
            "response": str(model_by_id[fixture_id]["response_text"]),
        }
        if arm == "relational_sql" and pass_index == 1:
            result = parse_relation_output(str(raw["raw_output"]), documents)
            parser = "relation_bundle"
        elif arm == "relational_sql":
            result = parse_sql_output(str(raw["raw_output"]))
            parser = "sql_query"
        else:
            result = parse_decision_output(str(raw["raw_output"]))
            parser = "decision"
        parsed[call_id] = result
        parse_rows.append(
            {
                "call_id": call_id,
                "model_id": raw["model_id"],
                "fixture_id": fixture_id,
                "arm": arm,
                "pass_index": pass_index,
                "parser": parser,
                "raw_output_sha256": raw["raw_output_sha256"],
                "status": result.get("status"),
                "result": deepcopy(result),
            }
        )

    relation_rows: list[JsonDict] = []
    query_rows: list[JsonDict] = []
    execution_rows: list[JsonDict] = []
    arm_rows: list[JsonDict] = []
    for model_id in REQUIRED_MODEL_IDS:
        for fixture_id in model_by_id:
            documents = {
                "source": str(model_by_id[fixture_id]["source_text"]),
                "response": str(model_by_id[fixture_id]["response_text"]),
            }
            cell_id = f"{model_id}|{fixture_id}"
            relation_call = f"{cell_id}|relational_sql|pass-1"
            query_call = f"{cell_id}|relational_sql|pass-2"
            relation = parsed[relation_call]
            query = parsed[query_call]
            relation_row = {
                "cell_id": cell_id,
                "model_id": model_id,
                "fixture_id": fixture_id,
                "call_id": relation_call,
                "status": relation.get("status"),
                "bundle": deepcopy(relation.get("bundle")),
                "relation_errors": deepcopy(relation.get("relation_errors", [])),
            }
            relation_rows.append(relation_row)
            query_row = {
                "cell_id": cell_id,
                "model_id": model_id,
                "fixture_id": fixture_id,
                "call_id": query_call,
                "status": query.get("status"),
                "query": query.get("query"),
                "reason": query.get("reason"),
            }
            query_rows.append(query_row)
            if relation.get("status") != "ok":
                execution = {"status": "rejected", "reason": "invalid_relation_bundle"}
            elif query.get("status") != "ok":
                execution = {"status": "rejected", "reason": "unsupported_sql"}
            else:
                execution = execute_sql_candidate(
                    relation["bundle"], documents, str(query["query"])
                )
            execution_row = {
                "cell_id": cell_id,
                "model_id": model_id,
                "fixture_id": fixture_id,
                "sandbox": "Exp7138.create_relation_database+execute_bounded_select",
                **deepcopy(execution),
            }
            execution_rows.append(execution_row)

            decisions: dict[str, JsonDict] = {}
            direct_call = f"{cell_id}|direct|pass-1"
            self_call = f"{cell_id}|self_verification|pass-2"
            for arm, call_id in (("direct", direct_call), ("self_verification", self_call)):
                decision = parsed[call_id]
                decisions[arm] = (
                    deepcopy(decision)
                    if decision.get("status") == "ok"
                    else {
                        "prediction": "unknown",
                        "hallucination_score": 0.5,
                        "invalid_sql": False,
                        "reason": "invalid_decision_output",
                    }
                )
            decisions["relational_sql"] = reduce_sql_prediction(
                relation_row["relation_errors"], query, execution
            )
            for arm in ARM_IDS:
                arm_call_rows = [
                    row
                    for row in raw_rows
                    if row.get("model_id") == model_id
                    and row.get("fixture_id") == fixture_id
                    and row.get("arm") == arm
                ]
                truth = truth_by_id[fixture_id]
                decision = decisions[arm]
                arm_rows.append(
                    {
                        "arm_row_id": f"{cell_id}|{arm}",
                        "fixture_id": fixture_id,
                        "model_id": model_id,
                        "model_family": MODEL_FAMILIES[model_id],
                        "source_family": family_by_id[fixture_id],
                        "arm": arm,
                        "prediction": decision["prediction"],
                        "hallucination_score": decision["hallucination_score"],
                        "truth_label": truth,
                        "correct": decision["prediction"] == truth,
                        "invalid_sql": bool(decision.get("invalid_sql", False)),
                        "unknown": decision["prediction"] == "unknown",
                        "latency_s": sum(
                            float(row.get("latency_s", 0.0) or 0.0) for row in arm_call_rows
                        ),
                        "prompt_tokens": sum(
                            int(row.get("prompt_tokens", 0) or 0) for row in arm_call_rows
                        ),
                        "completion_tokens": sum(
                            int(row.get("completion_tokens", 0) or 0) for row in arm_call_rows
                        ),
                        "cost_usd": 0.0,
                        "decision_reason": decision.get("reason"),
                        "call_ids": [row["call_id"] for row in arm_call_rows],
                        "prompt_hashes": [
                            prompt_by_id[str(row["call_id"])]["prompt_sha256"]
                            for row in arm_call_rows
                        ],
                    }
                )
    return parse_rows, relation_rows, query_rows, execution_rows, arm_rows


def run_experiment(  # pragma: no cover - required live host and model boundary.
    *,
    root: Path,
    run_date: str,
    result_path: Path,
    fixture_path: Path,
    raw_dir: Path,
) -> JsonDict:
    """Write first, preflight all models, run the matrix, and score once."""

    started = time.perf_counter()
    artifact = initialize_artifact(result_path, run_date)
    checks = [gate_row("run_date", RUN_DATE, run_date, run_date == RUN_DATE)]
    sources = _source_hashes(root)
    artifact["source_artifact_hashes"] = sources
    if not checks[-1]["passed"]:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress("fixture_preflight")
    fixture, fixture_checks = _load_fixture(fixture_path)
    checks.extend(fixture_checks)
    if fixture is None or any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress("model_cache_resolution")
    specs = resolve_model_specs()
    artifact["MODEL_SPECS"] = deepcopy(specs)
    specs, identities, model_checks = _model_file_receipts(specs)
    checks.extend(model_checks)
    artifact["MODEL_SPECS"] = deepcopy(specs)
    artifact["model_identity_rows"] = deepcopy(identities)
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress("gpu_backend_server_preflight")
    _gpu, backend, host_checks = _host_runtime_receipt()
    checks.extend(host_checks)
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    server_path = resolve_native_llama_server()
    load_receipts, canary_checks = _preflight_canaries(
        specs, server_path=server_path, raw_dir=raw_dir
    )
    checks.extend(canary_checks)
    artifact["preconditions_checked"] = deepcopy(checks)
    artifact["model_load_receipts"] = deepcopy(load_receipts)
    for receipt in artifact["model_load_receipts"]:
        receipt["backend_build"] = deepcopy(backend)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    write_artifact(result_path, artifact)
    if len(load_receipts) != len(REQUIRED_MODEL_IDS) or any(
        row["passed"] is not True for row in checks
    ):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    model_views = list(fixture["model_view_rows"])
    fixture_ids = [str(row["fixture_id"]) for row in model_views]
    schedule = build_schedule(model_views, specs)
    schedule_failures = schedule_errors(schedule, fixture_ids)
    schedule_check = gate_row("frozen_call_schedule", [], schedule_failures, not schedule_failures)
    checks.append(schedule_check)
    if schedule_failures:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress("matrix_generation", call_count=len(schedule))
    prompt_rows, raw_rows = _run_matrix(schedule, specs, server_path=server_path, raw_dir=raw_dir)
    # The sealed scorer view is first accessed after all planned raw calls freeze.
    sealed_rows = list(fixture["sealed_scorer_rows"])
    parse_rows, relation_rows, query_rows, execution_rows, arm_rows = _score_frozen_outputs(
        model_views=model_views,
        fixture_rows=list(fixture["fixture_rows"]),
        sealed_rows=sealed_rows,
        prompt_rows=prompt_rows,
        raw_rows=raw_rows,
    )
    result = finalize_artifact(
        artifact,
        arm_rows=arm_rows,
        prompt_rows=prompt_rows,
        raw_output_rows=raw_rows,
        parse_rows=parse_rows,
        relation_rows=relation_rows,
        sql_query_rows=query_rows,
        sql_execution_rows=execution_rows,
        model_specs=specs,
        model_identity_rows=identities,
        model_load_receipts=load_receipts,
        preconditions_checked=checks,
        source_artifact_hashes=sources,
        duration_s=time.perf_counter() - started,
        expected_fixture_ids=fixture_ids,
    )
    write_artifact(result_path, result)
    _progress(
        "complete",
        verdict_class=result["verdict_class"],
        symbolic_grounding_complete_score=result["symbolic_grounding_complete_score"],
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the fixed-date experiment or cold-validate one artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        print(canonical_json({"valid": not errors, "errors": errors}))
        return int(bool(errors))
    root = find_repo_root()
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    fixture_path = root / FIXTURE_PATH
    raw_dir = root / RAW_DIR
    result = run_experiment(
        root=root,
        run_date=args.date,
        result_path=result_path,
        fixture_path=fixture_path,
        raw_dir=raw_dir,
    )
    errors = validate_artifact(result)
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "valid": not errors,
                "errors": errors,
                "verdict_class": result.get("verdict_class"),
                "symbolic_grounding_complete_score": result.get(
                    "symbolic_grounding_complete_score"
                ),
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - use the experiment wrapper.
    raise SystemExit(main())
