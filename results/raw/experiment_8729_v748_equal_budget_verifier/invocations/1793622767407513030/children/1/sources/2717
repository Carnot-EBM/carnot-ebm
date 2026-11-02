"""Run a paired Qwen pilot with source-side and SQL-side checks.

The model proposes decisions, typed relations, and SQL. The runner treats all
three as untrusted. It opens the exact external labels only after every prompt
and raw output has a stable hash.

Spec refs: REQ-VERIFY-7154 and SCENARIO-VERIFY-7154-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Any
from urllib import request

from carnot import experiment_7138_v627_relational_fixture as fixture_v627
from carnot import experiment_7139_v627_symbolic_grounding_ab as grounding_v627
from carnot import experiment_7150_v628_grounding_preflight as preflight_v628
from carnot import experiment_7153_v629_grounding_runtime as runtime_v629
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    resolve_native_llama_server,
)
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    supervisor_contract,
)
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260909"
RANDOM_SEED = 7_154_202_609_09
RESULT_PATH = Path("results/experiment_7154_v629_qwen_dual_side_grounding.json")
RUNTIME_PATH = Path("results/experiment_7153_v629_grounding_runtime.json")
FIXTURE_PATH = Path("results/experiment_7138_v627_relational_fixture.json")
RAW_DIR = Path("results/raw/experiment_7154_v629_qwen_dual_side_grounding")
INFERENCE_SUBSTRATE = "live_llm_inference: paired Qwen dual-side grounding pilot"
EXECUTION_VENUE = "host"
QWEN_MODEL_ID = runtime_v629.QWEN_MODEL_ID
PREFERRED_QUANT = runtime_v629.PREFERRED_QUANT
FROZEN_FIXTURE_IDS = runtime_v629.FROZEN_FIXTURE_IDS
ARM_IDS = ("direct", "self_check", "relational_sql", "dual_side")
SQL_ARM_IDS = ("relational_sql", "dual_side")
REQUIRED_SQL_QUERY = grounding_v627.REQUIRED_SQL_QUERY
OUTPUT_TOKEN_LIMIT = grounding_v627.OUTPUT_TOKEN_LIMIT
BOOTSTRAP_REPLICATES = 2_000
CONDUCTOR_ANNOTATION_FIELDS = frozenset(
    {"flagged_adversarial", "flagged_adversarial_provenance", "corrigendum_pending"}
)

# The declaration is safe to write before cache inspection. Resolution remains
# local-only and occurs after the first artifact checkpoint.
MODEL_SPECS: list[JsonDict] = deepcopy(runtime_v629.MODEL_SPECS)

REQUIRED_PRECONDITION_CHECKS = (
    "run_date",
    "output_directory",
    "runtime_artifact",
    "runtime_gate",
    "model_hash",
    "fixture_hash",
    "frozen_fixture_ids",
    "frozen_schedule_hash",
    "exact_label_authority",
    "cached_qwen_q4",
    "embedded_chat_template",
    "native_cuda_linkage",
    "gpu_available",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "MODEL_SPECS",
    "model_identity_rows",
    "gpu_rows",
    "model_load_receipts",
    "frozen_fixture_ids",
    "frozen_schedule_hash",
    "prompt_hash_rows",
    "raw_output_rows",
    "parse_rows",
    "source_structure_rows",
    "sql_execution_rows",
    "arm_metric_rows",
    "source_family_metric_rows",
    "paired_comparison_rows",
    "bootstrap_rows",
    "harmful_flip_rows",
    "abstention_rows",
    "latency_rows",
    "token_rows",
    "label_exposure_count",
    "schedule_identity_score",
    "qwen_dual_side_pilot_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "status": "A durable state prevents interrupted work from appearing complete.",
    "field_principles": "A principle for each field makes missing evidence visible.",
    "preconditions_checked": "Exact comparisons show why inference was allowed or blocked.",
    "run_date": "The fixed date binds the result to the requested execution window.",
    "inference_substrate": "The declaration states that one local Qwen generated every output.",
    "inference_substrate_class": "The class separates full generation from a blocked no-run.",
    "execution_venue": "The host venue prevents an unsupported remote execution claim.",
    "duration_s": "Measured wall time exposes interruption and implausible model work.",
    "source_artifact_hashes": "Byte hashes bind the result to the exact upstream evidence.",
    "rows": "One row per fixture and arm prevents denominator drift.",
    "MODEL_SPECS": "The local-only Q4 declaration prevents model substitution.",
    "model_identity_rows": "File and template hashes prove which model bytes were used.",
    "gpu_rows": "GPU snapshots show availability, live use, and release after cleanup.",
    "model_load_receipts": "One lifecycle receipt proves that Qwen loaded only once.",
    "frozen_fixture_ids": "Ordered IDs prevent data-dependent row selection.",
    "frozen_schedule_hash": "The upstream digest binds all 168 planned opportunities.",
    "prompt_hash_rows": "Template and executed prompt hashes make prompt changes visible.",
    "raw_output_rows": "Unedited model output preserves failures and parse errors.",
    "parse_rows": "Typed parse receipts prevent silent repair of model text.",
    "source_structure_rows": "Source-side checks expose each formal-structure requirement.",
    "sql_execution_rows": "Read-only receipts preserve exact solution-side query output.",
    "arm_metric_rows": "Separate arm metrics prevent pooled intervention claims.",
    "source_family_metric_rows": "Family metrics expose source-specific losses.",
    "paired_comparison_rows": "Matched deltas compare the same external-label rows.",
    "bootstrap_rows": "Whole-row resampling keeps all four arms paired.",
    "harmful_flip_rows": "Explicit flips prevent gains from hiding damaged correct controls.",
    "abstention_rows": "Every unknown result stays in the denominator.",
    "latency_rows": "Per-arm latency retains the operational intervention cost.",
    "token_rows": "Per-arm tokens expose unequal generation work.",
    "label_exposure_count": "Zero means no sealed outcome entered model evidence.",
    "schedule_identity_score": "One means the executed call identities match Exp7153.",
    "qwen_dual_side_pilot_complete_score": "One reports matrix completion, not improvement.",
    "random_seed": "A fixed seed makes generation and resampling repeatable.",
    "reproducibility_checksum": "A canonical digest detects later artifact mutation.",
    "gate_check_summary": "The first failure names its upstream field and exact values.",
    "verifier_is_oracle": "False keeps SQL execution distinct from external correctness.",
    "verdict_class": "A closed class separates value, null, blocked, and disqualified runs.",
    "honest_verdict": "A matching prefix states the outcome without model-judge claims.",
}

canonical_json = grounding_v627.canonical_json
sha256_text = grounding_v627.sha256_text
sha256_file = grounding_v627.sha256_file
artifact_checksum = grounding_v627.artifact_checksum
parse_decision_output = grounding_v627.parse_decision_output
parse_relation_output = grounding_v627.parse_relation_output


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
    *,
    upstream: str | None,
    field: str | None,
) -> JsonDict:
    """Keep the upstream and field next to one exact gate comparison."""

    return {
        "upstream": upstream,
        "field": field,
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": expected == observed if passed is None else bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Promote the first failed gate into the stable terminal shape."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "upstream": None,
            "field": None,
            "failed_check": None,
            "expected_value": 1,
            "observed_value": 1,
            "passed": True,
        }
    return {
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "failed_check": failed.get("check"),
        "expected_value": deepcopy(failed.get("expected_value")),
        "observed_value": deepcopy(failed.get("observed_value")),
        "passed": False,
    }


def base_artifact(run_date: str) -> JsonDict:
    """Create every final field before any path or model check occurs."""

    artifact: JsonDict = {
        "status": "running",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_identity_rows": [],
        "gpu_rows": [],
        "model_load_receipts": [],
        "frozen_fixture_ids": [],
        "frozen_schedule_hash": "",
        "prompt_hash_rows": [],
        "raw_output_rows": [],
        "parse_rows": [],
        "source_structure_rows": [],
        "sql_execution_rows": [],
        "arm_metric_rows": [],
        "source_family_metric_rows": [],
        "paired_comparison_rows": [],
        "bootstrap_rows": [],
        "harmful_flip_rows": [],
        "abstention_rows": [],
        "latency_rows": [],
        "token_rows": [],
        "label_exposure_count": 0,
        "schedule_identity_score": 0,
        "qwen_dual_side_pilot_complete_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "upstream": None,
            "field": "experiment_complete",
            "failed_check": "experiment_complete",
            "expected_value": True,
            "observed_value": False,
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_qwen_dual_side_pilot",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> None:
    """Write one atomic checkpoint through the established experiment helper."""

    runtime_v629.write_artifact(path, dict(artifact))


def initialize_artifact(path: Path | str, run_date: str) -> JsonDict:
    """Write the complete schema before the first fallible check."""

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
    """Finish a failed gate as blocked and keep its exact upstream field."""

    result = deepcopy(dict(artifact))
    copied = [deepcopy(dict(row)) for row in checks]
    summary = _gate_summary(copied)
    failed = str(summary.get("failed_check") or "unknown_precondition")
    result.update(
        {
            "status": "blocked",
            "preconditions_checked": copied,
            "inference_substrate_class": "blocked_no_run",
            "duration_s": round(max(0.0, float(duration_s)), 6),
            "qwen_dual_side_pilot_complete_score": 0,
            "gate_check_summary": summary,
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failed}",
        }
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    write_artifact(path, result)
    return result


def _load_value(value: Mapping[str, Any] | str | Path | object) -> JsonDict | None:
    """Load a JSON object while preserving a simple cold-validation boundary."""

    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _runtime_without_annotations(value: Mapping[str, Any]) -> JsonDict:
    """Remove only conductor audit stamps before producer validation."""

    return {
        key: deepcopy(item) for key, item in value.items() if key not in CONDUCTOR_ANNOTATION_FIELDS
    }


def expected_call_schedule(runtime_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Flatten the qualified Exp7153 schedule without changing its order."""

    rows: list[JsonDict] = []
    for schedule_row in runtime_artifact.get("schedule_rows", []):
        for opportunity in schedule_row.get("call_opportunities", []):
            for call in opportunity.get("calls", []):
                rows.append(
                    {
                        "call_id": call.get("call_id"),
                        "model_id": schedule_row.get("model_id"),
                        "fixture_id": schedule_row.get("fixture_id"),
                        "source_family": schedule_row.get("source_family"),
                        "arm": opportunity.get("arm"),
                        "pass_index": call.get("pass_index"),
                        "output_token_limit": call.get("output_token_limit"),
                        "prompt": call.get("prompt"),
                        "prompt_sha256": call.get("prompt_sha256"),
                        "source_text_sha256": schedule_row.get("source_text_sha256"),
                        "response_text_sha256": schedule_row.get("response_text_sha256"),
                    }
                )
    return rows


def schedule_identity_errors(
    schedule: Sequence[Mapping[str, Any]], runtime_artifact: Mapping[str, Any]
) -> list[str]:
    """Reject any difference from the exact Exp7153 call opportunities."""

    expected = expected_call_schedule(runtime_artifact)
    errors: list[str] = []
    if len(schedule) != 168 or len(schedule) != len(expected):
        errors.append("schedule_call_count_mismatch")
    if [row.get("call_id") for row in schedule] != [row.get("call_id") for row in expected]:
        errors.append("schedule_call_order_mismatch")
    if [row.get("fixture_id") for row in schedule] != [row.get("fixture_id") for row in expected]:
        errors.append("schedule_fixture_order_mismatch")
    if [row.get("arm") for row in schedule] != [row.get("arm") for row in expected]:
        errors.append("schedule_arm_order_mismatch")
    for observed, wanted in zip(schedule, expected, strict=False):
        if observed.get("prompt_sha256") != wanted.get("prompt_sha256"):
            errors.append("schedule_prompt_hash_mismatch")
        if observed.get("output_token_limit") != wanted.get("output_token_limit"):
            errors.append("schedule_output_limit_mismatch")
        prompt = str(observed.get("prompt", ""))
        if observed.get("prompt_sha256") != sha256_text(prompt):
            errors.append("schedule_prompt_content_mismatch")
    expected_hash = sha256_text(canonical_json(runtime_artifact.get("schedule_rows", [])))
    if runtime_artifact.get("frozen_schedule_hash") != expected_hash:
        errors.append("runtime_schedule_hash_mismatch")
    return list(dict.fromkeys(errors))


def _label_exposure_errors(value: Any) -> list[str]:
    """Use typed checks so ordinary uses of the word label remain allowed."""

    return runtime_v629.typed_blinding_errors(value)


def model_evidence_seal(
    prompt_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Hash all model evidence before the exact labels become accessible."""

    errors: list[str] = []
    expected_ids = [str(row.get("call_id")) for row in schedule]
    prompt_ids = [str(row.get("call_id")) for row in prompt_rows]
    raw_ids = [str(row.get("call_id")) for row in raw_rows]
    if prompt_ids != expected_ids or len(prompt_ids) != 168:
        errors.append("prompt_receipt_identity_mismatch")
    if raw_ids != expected_ids or len(raw_ids) != 168:
        errors.append("raw_receipt_identity_mismatch")
    for row, planned in zip(prompt_rows, schedule, strict=False):
        if row.get("template_prompt_sha256") != planned.get("prompt_sha256"):
            errors.append("template_prompt_hash_mismatch")
        prompt = str(row.get("prompt", ""))
        # Tests can omit prompt text when they retain only its executed hash.
        if prompt and row.get("prompt_sha256") != sha256_text(prompt):
            errors.append("executed_prompt_hash_mismatch")
    for row in raw_rows:
        if row.get("raw_output_sha256") != sha256_text(str(row.get("raw_output", ""))):
            errors.append("raw_output_hash_mismatch")
        if row.get("terminal_state") not in {"complete", "failed"}:
            errors.append("raw_terminal_state_invalid")
    exposures = _label_exposure_errors([prompt_rows, raw_rows])
    errors.extend(exposures)
    evidence = {
        "prompt_hash_rows": [deepcopy(dict(row)) for row in prompt_rows],
        "raw_output_rows": [deepcopy(dict(row)) for row in raw_rows],
        "schedule_call_ids": expected_ids,
    }
    return {
        "sealed": not errors,
        "call_count": len(expected_ids),
        "label_exposure_count": len(exposures),
        "errors": list(dict.fromkeys(errors)),
        "evidence_sha256": sha256_text(canonical_json(evidence)),
    }


def _seal_matches(
    seal: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> bool:
    """Recompute the seal so later mutation cannot precede label access."""

    return dict(seal) == model_evidence_seal(prompt_rows, raw_rows, schedule)


def open_exact_labels(
    fixture_artifact: Mapping[str, Any], seal: Mapping[str, Any]
) -> dict[str, str]:
    """Open only the 24 exact labels after a valid 168-call evidence seal."""

    if seal.get("sealed") is not True or seal.get("call_count") != 168:
        raise ValueError("model evidence is not sealed")
    labels: dict[str, str] = {}
    for row in fixture_artifact.get("sealed_scorer_rows", []):
        fixture_id = str(row.get("fixture_id"))
        if fixture_id not in FROZEN_FIXTURE_IDS:
            continue
        label = str(row.get("response_label"))
        if label not in {"clean", "hallucinated"} or fixture_id in labels:
            raise ValueError(f"exact label authority invalid:{fixture_id}")
        labels[fixture_id] = label
    if list(labels) != list(FROZEN_FIXTURE_IDS):
        raise ValueError("exact label authority IDs differ from the frozen order")
    return labels


def parse_sql_output(text: str) -> JsonDict:
    """Keep the proposed query visible when the exact SQL parser rejects it."""

    result = grounding_v627.parse_sql_output(text)
    value = grounding_v627._extract_json_object(text)
    candidate = value.get("query") if isinstance(value, Mapping) else None
    return {**result, "candidate_query": candidate}


_SELECT_FIELDS_RE = re.compile(
    r"(?is)^\s*SELECT\s+relation_id\s*,\s*predicate\s*,\s*unknown_reason\s+"
)
_TABLE_RE = re.compile(r"(?is)\bFROM\s+grounded_relations\b")
_FILTER_RE = re.compile(r"(?is)\bWHERE\s+object_type\s*=\s*'unknown'(?![A-Za-z0-9_])")
_COMPARISON_RE = re.compile(r"(?is)\bobject_type\s*=\s*'unknown'(?![A-Za-z0-9_])")
_JOIN_RE = re.compile(r"(?is)\bJOIN\b")
_AGGREGATION_RE = re.compile(r"(?is)\b(?:GROUP\s+BY|HAVING|COUNT|SUM|AVG|MIN|MAX)\b")


def _source_provenance_valid(bundle: Mapping[str, Any]) -> bool:
    """Require supported values from source spans and unknowns from response spans."""

    for relation in bundle.get("relations", []):
        value_type = dict(relation.get("object") or {}).get("type")
        documents = {span.get("document") for span in relation.get("provenance", [])}
        required = "response" if value_type == "unknown" else "source"
        if required not in documents:
            return False
    return True


def check_source_structure(
    relation_parse: Mapping[str, Any], query_parse: Mapping[str, Any]
) -> JsonDict:
    """Check formal structure against source rules before SQL can run."""

    relation_valid = relation_parse.get("status") == "ok"
    bundle = relation_parse.get("bundle") if relation_valid else {}
    provenance_valid = relation_valid and _source_provenance_valid(dict(bundle or {}))
    query = str(query_parse.get("candidate_query") or query_parse.get("query") or "")
    table_valid = bool(_TABLE_RE.search(query))
    fields_valid = bool(_SELECT_FIELDS_RE.search(query))
    join_valid = not bool(_JOIN_RE.search(query))
    filter_valid = bool(_FILTER_RE.search(query))
    aggregation_valid = not bool(_AGGREGATION_RE.search(query))
    comparison_valid = bool(_COMPARISON_RE.search(query))
    errors: list[str] = []
    if not relation_valid:
        errors.append("relation_schema_invalid")
    if not provenance_valid:
        errors.append("source_provenance_invalid")
    if query_parse.get("status") != "ok":
        errors.append("sql_query_invalid")
    for name, passed in (
        ("table", table_valid),
        ("fields", fields_valid),
        ("join", join_valid),
        ("filter", filter_valid),
        ("aggregation", aggregation_valid),
        ("comparison", comparison_valid),
    ):
        if not passed:
            errors.append(f"{name}_invalid")
    row = {
        "relation_schema_valid": relation_valid,
        "source_provenance_valid": provenance_valid,
        "table_valid": table_valid,
        "fields_valid": fields_valid,
        "join_valid": join_valid,
        "filter_valid": filter_valid,
        "aggregation_valid": aggregation_valid,
        "comparison_valid": comparison_valid,
        "structure_valid": not errors,
        "errors": errors,
    }
    return row


def execute_checked_sql(
    relation_parse: Mapping[str, Any],
    query_parse: Mapping[str, Any],
    documents: Mapping[str, str],
    structure: Mapping[str, Any],
) -> JsonDict:
    """Execute only a source-valid bundle and the exact read-only query."""

    if structure.get("structure_valid") is not True:
        return {"status": "rejected", "reason": "source_structure_invalid"}
    return grounding_v627.execute_sql_candidate(
        dict(relation_parse["bundle"]), documents, str(query_parse["query"])
    )


def reduce_sql_prediction(execution: Mapping[str, Any]) -> JsonDict:
    """Map rejected SQL to abstention and exact unknown rows to detection."""

    if execution.get("status") != "ok":
        return {
            "prediction": "unknown",
            "hallucination_score": 0.5,
            "invalid_sql": True,
            "reason": str(execution.get("reason") or "sql_rejected"),
        }
    if int(execution.get("row_count", 0) or 0) > 0:
        return {
            "prediction": "hallucinated",
            "hallucination_score": 1.0,
            "invalid_sql": False,
            "reason": "exact_unknown_rows_returned",
        }
    return {
        "prediction": "clean",
        "hallucination_score": 0.0,
        "invalid_sql": False,
        "reason": "exact_query_returned_no_unknown_rows",
    }


def _decision_or_abstention(value: Mapping[str, Any]) -> JsonDict:
    """Keep rejected decision text as an abstention instead of guessing."""

    if value.get("status") == "ok":
        return deepcopy(dict(value))
    return {
        "prediction": "unknown",
        "hallucination_score": 0.5,
        "invalid_sql": False,
        "reason": str(value.get("reason") or "decision_parse_rejected"),
    }


def score_frozen_outputs(
    runtime_artifact: Mapping[str, Any],
    fixture_artifact: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Parse all outputs, open labels, then build complete paired arm rows."""

    schedule = expected_call_schedule(runtime_artifact)
    if not _seal_matches(seal, prompt_rows, raw_rows, schedule):
        raise ValueError("model evidence changed after sealing")
    labels = open_exact_labels(fixture_artifact, seal)
    model_by_id = {
        str(row.get("fixture_id")): row for row in fixture_artifact.get("model_view_rows", [])
    }
    family_by_id = {
        str(row.get("fixture_id")): str(row.get("source_family"))
        for row in fixture_artifact.get("fixture_rows", [])
    }
    raw_by_id = {str(row.get("call_id")): row for row in raw_rows}
    prompt_by_id = {str(row.get("call_id")): row for row in prompt_rows}
    parsed: dict[str, JsonDict] = {}
    parse_rows: list[JsonDict] = []
    for call in schedule:
        call_id = str(call["call_id"])
        raw = raw_by_id[call_id]
        fixture_id = str(call["fixture_id"])
        documents = {
            "source": str(model_by_id[fixture_id]["source_text"]),
            "response": str(model_by_id[fixture_id]["response_text"]),
        }
        arm = str(call["arm"])
        pass_index = int(call["pass_index"])
        if arm in SQL_ARM_IDS and pass_index == 1:
            result = parse_relation_output(str(raw.get("raw_output", "")), documents)
            parser = "typed_relation_schema"
        elif arm in SQL_ARM_IDS:
            result = parse_sql_output(str(raw.get("raw_output", "")))
            parser = "exact_sql_query"
        else:
            result = parse_decision_output(str(raw.get("raw_output", "")))
            parser = "decision"
        parsed[call_id] = result
        parse_rows.append(
            {
                "call_id": call_id,
                "fixture_id": fixture_id,
                "arm": arm,
                "pass_index": pass_index,
                "parser": parser,
                "raw_output_sha256": raw.get("raw_output_sha256"),
                "status": result.get("status"),
                "result": deepcopy(result),
            }
        )

    structures: list[JsonDict] = []
    executions: list[JsonDict] = []
    decisions: dict[tuple[str, str], JsonDict] = {}
    for fixture_id in FROZEN_FIXTURE_IDS:
        documents = {
            "source": str(model_by_id[fixture_id]["source_text"]),
            "response": str(model_by_id[fixture_id]["response_text"]),
        }
        direct_call = f"{QWEN_MODEL_ID}|{fixture_id}|direct|pass-1"
        self_call = f"{QWEN_MODEL_ID}|{fixture_id}|self_check|pass-2"
        decisions[(fixture_id, "direct")] = _decision_or_abstention(parsed[direct_call])
        decisions[(fixture_id, "self_check")] = _decision_or_abstention(parsed[self_call])
        for arm in SQL_ARM_IDS:
            relation_call = f"{QWEN_MODEL_ID}|{fixture_id}|{arm}|pass-1"
            query_call = f"{QWEN_MODEL_ID}|{fixture_id}|{arm}|pass-2"
            relation = parsed[relation_call]
            query = parsed[query_call]
            structure = check_source_structure(relation, query)
            structures.append(
                {
                    "fixture_id": fixture_id,
                    "arm": arm,
                    "relation_call_id": relation_call,
                    "query_call_id": query_call,
                    "enforced": arm == "dual_side",
                    **deepcopy(structure),
                }
            )
            if relation.get("status") != "ok":
                execution = {"status": "rejected", "reason": "relation_schema_invalid"}
            elif query.get("status") != "ok":
                execution = {"status": "rejected", "reason": "exact_sql_invalid"}
            elif arm == "dual_side":
                execution = execute_checked_sql(relation, query, documents, structure)
            else:
                execution = grounding_v627.execute_sql_candidate(
                    dict(relation["bundle"]), documents, str(query["query"])
                )
            executions.append(
                {
                    "fixture_id": fixture_id,
                    "arm": arm,
                    "sandbox": "Exp7138.create_relation_database+execute_bounded_select",
                    "query": query.get("candidate_query"),
                    "structure_enforced": arm == "dual_side",
                    **deepcopy(execution),
                }
            )
            decisions[(fixture_id, arm)] = reduce_sql_prediction(execution)

    rows: list[JsonDict] = []
    for fixture_id in FROZEN_FIXTURE_IDS:
        truth = labels[fixture_id]
        direct = decisions[(fixture_id, "direct")]
        direct_correct = direct["prediction"] == truth
        for arm in ARM_IDS:
            decision = decisions[(fixture_id, arm)]
            prediction = str(decision["prediction"])
            accepted = prediction != "unknown"
            correct = prediction == truth
            call_rows = [
                row
                for row in raw_rows
                if row.get("fixture_id") == fixture_id and row.get("arm") == arm
            ]
            structure_row = next(
                (
                    row
                    for row in structures
                    if row["fixture_id"] == fixture_id and row["arm"] == arm
                ),
                None,
            )
            execution_row = next(
                (
                    row
                    for row in executions
                    if row["fixture_id"] == fixture_id and row["arm"] == arm
                ),
                None,
            )
            rows.append(
                {
                    "arm_row_id": f"{QWEN_MODEL_ID}|{fixture_id}|{arm}",
                    "fixture_id": fixture_id,
                    "model_id": QWEN_MODEL_ID,
                    "source_family": family_by_id[fixture_id],
                    "arm": arm,
                    "prediction": prediction,
                    "hallucination_score": decision["hallucination_score"],
                    "truth_label": truth,
                    "correct": correct,
                    "accepted": accepted,
                    "accepted_correct": correct if accepted else None,
                    "detected": truth == "hallucinated" and prediction == "hallucinated",
                    "repaired": arm != "direct" and not direct_correct and correct,
                    "harmful_flip": arm != "direct" and direct_correct and not correct,
                    "abstained": not accepted,
                    "structure_valid": (
                        structure_row.get("structure_valid") if structure_row else None
                    ),
                    "sql_valid": (execution_row.get("status") == "ok" if execution_row else None),
                    "invalid_sql": bool(decision.get("invalid_sql", False)),
                    "latency_s": sum(float(row.get("latency_s", 0.0) or 0.0) for row in call_rows),
                    "prompt_tokens": sum(
                        int(row.get("prompt_tokens", 0) or 0) for row in call_rows
                    ),
                    "completion_tokens": sum(
                        int(row.get("completion_tokens", 0) or 0) for row in call_rows
                    ),
                    "call_ids": [str(row.get("call_id")) for row in call_rows],
                    "prompt_hashes": [
                        prompt_by_id[str(row.get("call_id"))].get("prompt_sha256")
                        for row in call_rows
                    ],
                    "decision_reason": decision.get("reason"),
                }
            )
    return parse_rows, structures, executions, rows


def arm_accounting_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require one row per frozen fixture and arm with no duplicate identity."""

    errors: list[str] = []
    identities = [str(row.get("arm_row_id")) for row in rows]
    if len(rows) != 96 or len(identities) != len(set(identities)):
        errors.append(f"arm_row_count:{len(rows)}")
    by_fixture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fixture[str(row.get("fixture_id"))].append(row)
    if list(by_fixture) != list(FROZEN_FIXTURE_IDS):
        errors.append("fixture_row_order")
    for fixture_id in FROZEN_FIXTURE_IDS:
        observed = [str(row.get("arm")) for row in by_fixture.get(fixture_id, [])]
        if observed != list(ARM_IDS):
            errors.append(f"fixture_arm_set:{fixture_id}")
    return list(dict.fromkeys(errors))


def _metric_row(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce correctness, acceptance, detection, structure, and resource data."""

    count = len(rows)
    accepted = [row for row in rows if row.get("accepted") is True]
    hallucinated = [row for row in rows if row.get("truth_label") == "hallucinated"]
    structured = [row for row in rows if row.get("structure_valid") is not None]
    sql_rows = [row for row in rows if row.get("sql_valid") is not None]
    return {
        "row_count": count,
        "correct_count": sum(row.get("correct") is True for row in rows),
        "accuracy": sum(row.get("correct") is True for row in rows) / count if count else None,
        "accepted_count": len(accepted),
        "accepted_accuracy": (
            sum(row.get("correct") is True for row in accepted) / len(accepted)
            if accepted
            else None
        ),
        "detection_count": sum(row.get("detected") is True for row in rows),
        "detection_rate": (
            sum(row.get("detected") is True for row in hallucinated) / len(hallucinated)
            if hallucinated
            else None
        ),
        "repair_count": sum(row.get("repaired") is True for row in rows),
        "harmful_flip_count": sum(row.get("harmful_flip") is True for row in rows),
        "abstention_count": sum(row.get("abstained") is True for row in rows),
        "abstention_rate": (
            sum(row.get("abstained") is True for row in rows) / count if count else None
        ),
        "structure_valid_count": sum(row.get("structure_valid") is True for row in structured),
        "structure_valid_rate": (
            sum(row.get("structure_valid") is True for row in structured) / len(structured)
            if structured
            else None
        ),
        "sql_valid_count": sum(row.get("sql_valid") is True for row in sql_rows),
        "sql_valid_rate": (
            sum(row.get("sql_valid") is True for row in sql_rows) / len(sql_rows)
            if sql_rows
            else None
        ),
        "latency_s": sum(float(row.get("latency_s", 0.0) or 0.0) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens", 0) or 0) for row in rows),
    }


def build_arm_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report each arm on the same complete 24-row denominator."""

    return [
        {"arm": arm, **_metric_row([row for row in rows if row.get("arm") == arm])}
        for arm in ARM_IDS
    ]


def build_source_family_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report every source family and arm without pooled reversals."""

    families = sorted({str(row.get("source_family")) for row in rows})
    return [
        {
            "source_family": family,
            "arm": arm,
            **_metric_row(
                [
                    row
                    for row in rows
                    if row.get("source_family") == family and row.get("arm") == arm
                ]
            ),
        }
        for family in families
        for arm in ARM_IDS
    ]


def _paired_delta(
    rows: Sequence[Mapping[str, Any]], comparator: str, fixture_ids: Sequence[str]
) -> float:
    """Compute dual-side minus comparator accuracy on matched fixture IDs."""

    keyed = {(str(row.get("fixture_id")), str(row.get("arm"))): row for row in rows}
    deltas = [
        float(bool(keyed[(fixture_id, "dual_side")].get("correct")))
        - float(bool(keyed[(fixture_id, comparator)].get("correct")))
        for fixture_id in fixture_ids
    ]
    return sum(deltas) / len(deltas) if deltas else math.nan


def build_paired_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Compare dual-side with each control on exact external-label pairs."""

    keyed = {(str(row.get("fixture_id")), str(row.get("arm"))): row for row in rows}
    result = []
    for comparator in ARM_IDS[:-1]:
        pairs = [
            (keyed[(fixture_id, comparator)], keyed[(fixture_id, "dual_side")])
            for fixture_id in FROZEN_FIXTURE_IDS
            if (fixture_id, comparator) in keyed and (fixture_id, "dual_side") in keyed
        ]
        result.append(
            {
                "comparator": comparator,
                "intervention": "dual_side",
                "row_count": len(pairs),
                "accuracy_delta": (
                    sum(
                        float(bool(dual.get("correct"))) - float(bool(control.get("correct")))
                        for control, dual in pairs
                    )
                    / len(pairs)
                    if pairs
                    else None
                ),
                "detection_delta": (
                    sum(
                        float(bool(dual.get("detected"))) - float(bool(control.get("detected")))
                        for control, dual in pairs
                    )
                    / len(pairs)
                    if pairs
                    else None
                ),
                "repair_count": sum(
                    control.get("correct") is not True and dual.get("correct") is True
                    for control, dual in pairs
                ),
                "harmful_flip_count": sum(
                    control.get("correct") is True and dual.get("correct") is not True
                    for control, dual in pairs
                ),
            }
        )
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    """Return one linear quantile from a non-empty ordered bootstrap sample."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int, n_boot: int
) -> list[JsonDict]:
    """Resample fixture IDs so each replicate retains all four arm rows."""

    if n_boot <= 0 or arm_accounting_errors(rows):
        return []
    families = sorted({str(row.get("source_family")) for row in rows})
    scopes = [("overall", "all", list(FROZEN_FIXTURE_IDS))]
    for family in families:
        ids = [
            fixture_id
            for fixture_id in FROZEN_FIXTURE_IDS
            if any(
                row.get("fixture_id") == fixture_id and row.get("source_family") == family
                for row in rows
            )
        ]
        scopes.append(("source_family", family, ids))
    rng = random.Random(seed)
    result = []
    for comparator in ARM_IDS[:-1]:
        for scope, scope_value, fixture_ids in scopes:
            observed = _paired_delta(rows, comparator, fixture_ids)
            samples = []
            for _index in range(n_boot):
                sampled = [rng.choice(fixture_ids) for _item in fixture_ids]
                samples.append(_paired_delta(rows, comparator, sampled))
            result.append(
                {
                    "scope": scope,
                    "scope_value": scope_value,
                    "comparator": comparator,
                    "intervention": "dual_side",
                    "row_count": len(fixture_ids),
                    "metric": "accuracy_delta",
                    "observed_delta": observed,
                    "ci95_low": _quantile(samples, 0.025),
                    "ci95_high": _quantile(samples, 0.975),
                    "n_boot": n_boot,
                    "seed": seed,
                }
            )
    return result


def build_harmful_flip_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep every non-direct comparison, including rows with no harmful flip."""

    direct = {str(row.get("fixture_id")): row for row in rows if row.get("arm") == "direct"}
    return [
        {
            "fixture_id": row.get("fixture_id"),
            "source_family": row.get("source_family"),
            "arm": row.get("arm"),
            "direct_prediction": direct[str(row.get("fixture_id"))].get("prediction"),
            "arm_prediction": row.get("prediction"),
            "truth_label": row.get("truth_label"),
            "harmful_flip": row.get("harmful_flip"),
        }
        for row in rows
        if row.get("arm") != "direct" and str(row.get("fixture_id")) in direct
    ]


def build_abstention_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retain the abstention state for every arm and fixture."""

    return [
        {
            "fixture_id": row.get("fixture_id"),
            "source_family": row.get("source_family"),
            "arm": row.get("arm"),
            "prediction": row.get("prediction"),
            "abstained": row.get("abstained"),
        }
        for row in rows
    ]


def build_latency_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report total and mean latency for every arm."""

    result = []
    for arm in ARM_IDS:
        selected = [row for row in rows if row.get("arm") == arm]
        total = sum(float(row.get("latency_s", 0.0) or 0.0) for row in selected)
        result.append(
            {
                "arm": arm,
                "row_count": len(selected),
                "total_latency_s": total,
                "mean_latency_s": total / len(selected) if selected else None,
            }
        )
    return result


def build_token_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report prompt and completion token totals for every arm."""

    return [
        {
            "arm": arm,
            "row_count": len([row for row in rows if row.get("arm") == arm]),
            "prompt_tokens": sum(
                int(row.get("prompt_tokens", 0) or 0) for row in rows if row.get("arm") == arm
            ),
            "completion_tokens": sum(
                int(row.get("completion_tokens", 0) or 0) for row in rows if row.get("arm") == arm
            ),
        }
        for arm in ARM_IDS
    ]


def select_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedule_identity_score: int,
    label_exposure_count: int,
    bootstrap_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, str, str, list[str]]:
    """Separate comparison completion from supported scientific value."""

    reasons: list[str] = []
    accounting = arm_accounting_errors(rows)
    if accounting:
        reasons.append("rows_missing")
    if schedule_identity_score != 1:
        reasons.append("schedule_identity_failed")
    if label_exposure_count != 0:
        reasons.append("label_leak")
    direct = [row for row in rows if row.get("arm") == "direct"]
    if not direct or all(row.get("correct") is True for row in direct):
        reasons.append("control_headroom_absent")
    keyed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        keyed[str(row.get("fixture_id"))].append(row)
    if keyed and all(
        len({row.get("prediction") for row in fixture_rows}) == 1
        and len(fixture_rows) == len(ARM_IDS)
        for fixture_rows in keyed.values()
    ):
        reasons.append("all_arms_identical")
    complete = not accounting and schedule_identity_score == 1 and label_exposure_count == 0
    completion_score = int(complete)
    if reasons:
        return completion_score, "disqualified", "disqualified_" + "_and_".join(reasons), reasons

    overall = [row for row in bootstrap_rows if row.get("scope") == "overall"]
    supported = (
        len(overall) == 3
        and all(float(row.get("ci95_low", 0.0) or 0.0) > 0.0 for row in overall)
        and not any(
            row.get("arm") == "dual_side" and row.get("harmful_flip") is True for row in rows
        )
    )
    family_losses = any(
        row.get("scope") == "source_family" and float(row.get("observed_delta", 0.0) or 0.0) < 0.0
        for row in bootstrap_rows
    )
    if supported and not family_losses:
        return 1, "positive", "positive_external_label_dual_side_improvement", []
    return 1, "null", "null_complete_qwen_dual_side_no_supported_improvement", []


def finalize_artifact(
    artifact: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    n_boot: int = BOOTSTRAP_REPLICATES,
) -> JsonDict:
    """Build all row-derived fields and select one consistent terminal verdict."""

    if any(row.get("passed") is not True for row in checks):
        result = deepcopy(dict(artifact))
        copied = [deepcopy(dict(row)) for row in checks]
        summary = _gate_summary(copied)
        failed = str(summary.get("failed_check") or "precondition")
        result.update(
            {
                "status": "blocked",
                "preconditions_checked": copied,
                "inference_substrate_class": "blocked_no_run",
                "duration_s": round(max(0.0, float(duration_s)), 6),
                "qwen_dual_side_pilot_complete_score": 0,
                "gate_check_summary": summary,
                "verdict_class": "blocked",
                "honest_verdict": f"blocked_{failed}",
            }
        )
        result["reproducibility_checksum"] = artifact_checksum(result)
        return result
    result = deepcopy(dict(artifact))
    rows = list(result.get("rows") or [])
    bootstrap = build_bootstrap_rows(rows, seed=RANDOM_SEED, n_boot=n_boot)
    score, verdict_class, verdict, _reasons = select_verdict(
        rows,
        schedule_identity_score=int(result.get("schedule_identity_score", 0) or 0),
        label_exposure_count=int(result.get("label_exposure_count", 0) or 0),
        bootstrap_rows=bootstrap,
    )
    result.update(
        {
            "status": "completed",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "inference_substrate_class": "model_full_generation",
            "duration_s": round(max(0.0, float(duration_s)), 6),
            "arm_metric_rows": build_arm_metric_rows(rows),
            "source_family_metric_rows": build_source_family_metric_rows(rows),
            "paired_comparison_rows": build_paired_comparison_rows(rows),
            "bootstrap_rows": bootstrap,
            "harmful_flip_rows": build_harmful_flip_rows(rows),
            "abstention_rows": build_abstention_rows(rows),
            "latency_rows": build_latency_rows(rows),
            "token_rows": build_token_rows(rows),
            "qwen_dual_side_pilot_complete_score": score,
            "gate_check_summary": _gate_summary(checks),
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
        }
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def _completed_model_evidence_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Check the one-model lifecycle receipt without trusting aggregate metrics."""

    errors: list[str] = []
    specs = list(artifact.get("MODEL_SPECS") or [])
    identities = list(artifact.get("model_identity_rows") or [])
    loads = list(artifact.get("model_load_receipts") or [])
    if len(specs) != 1 or specs[0].get("hf_id") != QWEN_MODEL_ID:
        errors.append("MODEL_SPECS_mismatch")
    if len(identities) != 1 or identities[0].get("sha256") != (
        specs[0].get("sha256") if specs else None
    ):
        errors.append("model_identity_mismatch")
    if len(loads) != 1:
        errors.append("model_load_receipt_count")
    elif specs:
        load = loads[0]
        if load.get("model_id") != QWEN_MODEL_ID or load.get("sha256") != specs[0].get("sha256"):
            errors.append("model_load_identity_mismatch")
        if dict(load.get("health") or {}).get("ok") is not True:
            errors.append("model_load_health_failed")
        if load.get("gpu_offload_confirmed") is not True:
            errors.append("model_gpu_offload_unconfirmed")
        if dict(load.get("cleanup") or {}).get("leak_free") is not True:
            errors.append("model_cleanup_failed")
    phases = {str(row.get("phase")) for row in artifact.get("gpu_rows", [])}
    if not {"before", "model_loaded", "after_teardown"}.issubset(phases):
        errors.append("gpu_phase_receipts_missing")
    return errors


def validate_artifact(
    value: Mapping[str, Any] | str | Path | object,
    *,
    runtime_artifact: Mapping[str, Any] | str | Path | None = None,
    fixture_artifact: Mapping[str, Any] | str | Path | None = None,
) -> list[str]:
    """Cold-check gates, evidence, metrics, verdict, and exact upstream rows."""

    artifact = _load_value(value)
    if artifact is None:
        return ["artifact_not_object"]
    required = set(REQUIRED_ARTIFACT_FIELDS)
    if set(artifact) != required:
        return [f"artifact_fields_mismatch:{sorted(set(artifact) ^ required)}"]
    errors: list[str] = []
    if set(artifact.get("field_principles", {})) != required or any(
        not str(item).strip() for item in artifact.get("field_principles", {}).values()
    ):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    verdict_class = str(artifact.get("verdict_class"))
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(f"{verdict_class}_"):
        errors.append("honest_verdict_prefix_mismatch")
    checks = list(artifact.get("preconditions_checked") or [])
    if artifact.get("gate_check_summary") != _gate_summary(checks):
        errors.append("gate_check_summary_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")

    if verdict_class == "blocked":
        if artifact.get("status") != "blocked":
            errors.append("blocked_status_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("qwen_dual_side_pilot_complete_score") != 0:
            errors.append("blocked_completion_score_mismatch")
        summary = dict(artifact.get("gate_check_summary") or {})
        if (
            summary.get("passed") is not False
            or not summary.get("upstream")
            or not summary.get("field")
            or not summary.get("failed_check")
        ):
            errors.append("blocked_gate_detail_missing")
        return list(dict.fromkeys(errors))

    if artifact.get("status") != "completed":
        errors.append("completed_status_mismatch")
    if artifact.get("inference_substrate_class") != "model_full_generation":
        errors.append("completed_substrate_class_mismatch")
    check_map = {str(row.get("check")): row.get("passed") for row in checks}
    if any(check_map.get(name) is not True for name in REQUIRED_PRECONDITION_CHECKS):
        errors.append("preconditions_incomplete")
    errors.extend(_completed_model_evidence_errors(artifact))

    runtime_value = _load_value(runtime_artifact or RUNTIME_PATH)
    fixture_value = _load_value(fixture_artifact or FIXTURE_PATH)
    if runtime_value is None:
        errors.append("runtime_artifact_missing")
        return list(dict.fromkeys(errors))
    if fixture_value is None:
        errors.append("fixture_artifact_missing")
        return list(dict.fromkeys(errors))
    schedule = expected_call_schedule(runtime_value)
    errors.extend(schedule_identity_errors(schedule, runtime_value))
    if artifact.get("frozen_fixture_ids") != list(FROZEN_FIXTURE_IDS):
        errors.append("frozen_fixture_ids_mismatch")
    if artifact.get("frozen_schedule_hash") != runtime_value.get("frozen_schedule_hash"):
        errors.append("frozen_schedule_hash_mismatch")
    if artifact.get("schedule_identity_score") != int(
        not schedule_identity_errors(schedule, runtime_value)
    ):
        errors.append("schedule_identity_score_mismatch")

    prompt_rows = list(artifact.get("prompt_hash_rows") or [])
    raw_rows = list(artifact.get("raw_output_rows") or [])
    for row in raw_rows:
        if row.get("raw_output_sha256") != sha256_text(str(row.get("raw_output", ""))):
            errors.append("raw_output_hash_mismatch")
    seal = model_evidence_seal(prompt_rows, raw_rows, schedule)
    if seal.get("sealed") is not True:
        errors.extend(str(item) for item in seal.get("errors", []))
    if artifact.get("label_exposure_count") != seal.get("label_exposure_count"):
        errors.append("label_exposure_count_mismatch")
    if seal.get("sealed") is True:
        try:
            parse_rows, structure_rows, execution_rows, arm_rows = score_frozen_outputs(
                runtime_value, fixture_value, prompt_rows, raw_rows, seal
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"scoring_replay_failed:{type(exc).__name__}")
        else:
            if artifact.get("parse_rows") != parse_rows:
                errors.append("parse_rows_mismatch")
            if artifact.get("source_structure_rows") != structure_rows:
                errors.append("source_structure_rows_mismatch")
            if artifact.get("sql_execution_rows") != execution_rows:
                errors.append("sql_execution_rows_mismatch")
            if artifact.get("rows") != arm_rows:
                errors.append("rows_replay_mismatch")

    rows = list(artifact.get("rows") or [])
    errors.extend(arm_accounting_errors(rows))
    if artifact.get("arm_metric_rows") != build_arm_metric_rows(rows):
        errors.append("arm_metric_rows_mismatch")
    if artifact.get("source_family_metric_rows") != build_source_family_metric_rows(rows):
        errors.append("source_family_metric_rows_mismatch")
    if artifact.get("paired_comparison_rows") != build_paired_comparison_rows(rows):
        errors.append("paired_comparison_rows_mismatch")
    bootstrap = list(artifact.get("bootstrap_rows") or [])
    n_boot = int(bootstrap[0].get("n_boot", 0) or 0) if bootstrap else 0
    expected_bootstrap = (
        build_bootstrap_rows(rows, seed=RANDOM_SEED, n_boot=n_boot) if n_boot else []
    )
    if bootstrap != expected_bootstrap:
        errors.append("bootstrap_rows_mismatch")
    if artifact.get("harmful_flip_rows") != build_harmful_flip_rows(rows):
        errors.append("harmful_flip_rows_mismatch")
    if artifact.get("abstention_rows") != build_abstention_rows(rows):
        errors.append("abstention_rows_mismatch")
    if artifact.get("latency_rows") != build_latency_rows(rows):
        errors.append("latency_rows_mismatch")
    if artifact.get("token_rows") != build_token_rows(rows):
        errors.append("token_rows_mismatch")
    score, expected_class, expected_verdict, _reasons = select_verdict(
        rows,
        schedule_identity_score=int(artifact.get("schedule_identity_score", 0) or 0),
        label_exposure_count=int(artifact.get("label_exposure_count", 0) or 0),
        bootstrap_rows=expected_bootstrap,
    )
    if artifact.get("qwen_dual_side_pilot_complete_score") != score:
        errors.append("completion_score_mismatch")
    if verdict_class != expected_class or artifact.get("honest_verdict") != expected_verdict:
        errors.append("terminal_verdict_mismatch")
    return list(dict.fromkeys(errors))


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    """Print and flush one compact line at each observable phase boundary."""

    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def _checkpoint(path: Path, artifact: JsonDict) -> None:  # pragma: no cover
    """Refresh the checksum before each fallible live phase."""

    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    write_artifact(path, artifact)


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover
    """Bind the result to the exact code, tests, specs, and upstream files."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("research-references.md"),
        Path("openspec/capabilities/verification/spec.md"),
        FIXTURE_PATH,
        RUNTIME_PATH,
        Path("python/carnot/experiment_7138_v627_relational_fixture.py"),
        Path("python/carnot/experiment_7139_v627_symbolic_grounding_ab.py"),
        Path("python/carnot/experiment_7150_v628_grounding_preflight.py"),
        Path("python/carnot/experiment_7153_v629_grounding_runtime.py"),
        Path("python/carnot/experiment_7154_v629_qwen_dual_side_grounding.py"),
        Path("python/carnot/inference/llama_server_supervisor.py"),
        Path("scripts/experiment_template.py"),
        Path("scripts/experiments/experiment_7154_v629_qwen_dual_side_grounding.py"),
        Path("tests/python/test_experiment_7154_v629_qwen_dual_side_grounding.py"),
    )
    return {
        str(path): sha256_file(root / path) if (root / path).is_file() else None for path in paths
    }


def _output_precondition(result_path: Path, raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Require a writable result parent and an unused task-specific raw directory."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(str(path) for path in raw_dir.iterdir())
    return {
        "result_parent_writable": result_path.parent.is_dir(),
        "raw_dir_writable": raw_dir.is_dir(),
        "raw_dir_isolated": not existing,
        "existing_raw_entries": existing,
    }


def _upstream_checks(
    runtime_path: Path, fixture_path: Path, runtime_value: Mapping[str, Any] | None
) -> tuple[list[JsonDict], JsonDict | None]:  # pragma: no cover
    """Check the qualified schedule and exact label authority before startup."""

    checks: list[JsonDict] = []
    if runtime_value is None:
        checks.append(
            gate_row(
                "runtime_artifact",
                "readable_json_object",
                "missing_or_invalid",
                False,
                upstream=str(RUNTIME_PATH),
                field="artifact",
            )
        )
        return checks, None
    canonical_runtime = _runtime_without_annotations(runtime_value)
    validation_errors = runtime_v629.validate_artifact(canonical_runtime)
    checks.append(
        gate_row(
            "runtime_artifact",
            [],
            validation_errors,
            not validation_errors,
            upstream=str(RUNTIME_PATH),
            field="artifact_validation",
        )
    )
    checks.append(
        gate_row(
            "runtime_gate",
            1,
            runtime_value.get("grounding_runtime_ready_score"),
            runtime_value.get("grounding_runtime_ready_score") == 1,
            upstream=str(RUNTIME_PATH),
            field="grounding_runtime_ready_score",
        )
    )
    specs = list(runtime_value.get("MODEL_SPECS") or [])
    observed_model = {
        "model_id": specs[0].get("hf_id") if len(specs) == 1 else None,
        "sha256": specs[0].get("sha256") if len(specs) == 1 else None,
    }
    expected_model = {
        "model_id": QWEN_MODEL_ID,
        "sha256": runtime_v629.MODEL_SPECS[0].get("sha256")
        or (
            runtime_value.get("model_identity_rows", [{}])[0].get("sha256")
            if runtime_value.get("model_identity_rows")
            else None
        ),
    }
    # The unresolved declaration has no hash. The qualified identity supplies it.
    expected_model["sha256"] = (
        runtime_value.get("model_identity_rows", [{}])[0].get("sha256")
        if runtime_value.get("model_identity_rows")
        else None
    )
    checks.append(
        gate_row(
            "model_hash",
            expected_model,
            observed_model,
            observed_model == expected_model and bool(expected_model["sha256"]),
            upstream=str(RUNTIME_PATH),
            field="MODEL_SPECS[0].sha256",
        )
    )
    fixture_exists = fixture_path.is_file()
    fixture_hash = sha256_file(fixture_path) if fixture_exists else "missing"
    runtime_fixture_hash = dict(runtime_value.get("source_artifact_hashes") or {}).get(
        str(FIXTURE_PATH)
    )
    checks.append(
        gate_row(
            "fixture_hash",
            runtime_fixture_hash,
            fixture_hash,
            fixture_exists and fixture_hash == runtime_fixture_hash,
            upstream=str(RUNTIME_PATH),
            field="source_artifact_hashes.results/experiment_7138_v627_relational_fixture.json",
        )
    )
    checks.append(
        gate_row(
            "frozen_fixture_ids",
            list(FROZEN_FIXTURE_IDS),
            runtime_value.get("frozen_fixture_ids"),
            runtime_value.get("frozen_fixture_ids") == list(FROZEN_FIXTURE_IDS),
            upstream=str(RUNTIME_PATH),
            field="frozen_fixture_ids",
        )
    )
    expected_schedule_hash = sha256_text(canonical_json(runtime_value.get("schedule_rows", [])))
    checks.append(
        gate_row(
            "frozen_schedule_hash",
            expected_schedule_hash,
            runtime_value.get("frozen_schedule_hash"),
            runtime_value.get("frozen_schedule_hash") == expected_schedule_hash,
            upstream=str(RUNTIME_PATH),
            field="frozen_schedule_hash",
        )
    )
    fixture_value = _load_value(fixture_path) if fixture_exists else None
    fixture_errors = (
        fixture_v627.validate_artifact(fixture_value) if fixture_value is not None else ["missing"]
    )
    authority_ids = (
        [str(row.get("fixture_id")) for row in fixture_value.get("sealed_scorer_rows", [])]
        if fixture_value is not None
        else []
    )
    authority_observed = {
        "fixture_validation_errors": fixture_errors,
        "authority_field": "sealed_scorer_rows" if fixture_value is not None else None,
        "frozen_ids_present": all(item in authority_ids for item in FROZEN_FIXTURE_IDS),
    }
    authority_expected = {
        "fixture_validation_errors": [],
        "authority_field": "sealed_scorer_rows",
        "frozen_ids_present": True,
    }
    checks.append(
        gate_row(
            "exact_label_authority",
            authority_expected,
            authority_observed,
            authority_observed == authority_expected,
            upstream=str(FIXTURE_PATH),
            field="sealed_scorer_rows",
        )
    )
    return checks, fixture_value


def _run_subprocess(
    command: list[str], *, timeout_s: float = 15.0, phase: int = 4
) -> JsonDict:  # pragma: no cover
    """Run one bounded subprocess with visible start and end lines."""

    _progress(phase, "subprocess_start", command=command)
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        row = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_s": time.perf_counter() - started,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        row = {
            "returncode": None,
            "stdout": getattr(exc, "stdout", "") or "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": time.perf_counter() - started,
        }
    _progress(phase, "subprocess_end", command=command, returncode=row["returncode"])
    return row


def _request_generation(
    port: int, prompt: str, *, max_tokens: int, seed: int
) -> JsonDict:  # pragma: no cover
    """Send one deterministic embedded-template chat request with a finite timeout."""

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
    with request.urlopen(http_request, timeout=240.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = list(body.get("choices") or [{}])
    message = dict(choices[0].get("message") or {})
    usage = dict(body.get("usage") or {})
    return {
        "raw_output": str(message.get("content") or ""),
        "reasoning_output": str(message.get("reasoning_content") or message.get("reasoning") or ""),
        "raw_response": body,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "latency_s": time.perf_counter() - started,
    }


def _run_matrix_once(
    schedule: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    server_path: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict, list[JsonDict]]:  # pragma: no cover
    """Load Qwen once and run all 168 calls in their frozen order."""

    port = preflight_v628._free_port()
    command = grounding_v627._server_command(
        server_path, Path(str(spec["model_path"])), port, n_ctx=8_192
    )
    contract = supervisor_contract(
        outer_deadline_s=4_200,
        health_timeout_s=480,
        token_timeout_s=240,
        cleanup_grace_s=30,
        kill_after_cleanup_timeout_s=10,
        retry_budget=0,
        endurance_interval_s=0,
        endurance_sample_count=1,
    )
    supervisor = NativeLlamaServerSupervisor(command, raw_dir, contract)
    identity: JsonDict = {}
    health: JsonDict = {"ok": False, "classification": "not_started"}
    prompt_rows: list[JsonDict] = []
    raw_rows: list[JsonDict] = []
    gpu_rows: list[JsonDict] = []
    first_outputs: dict[tuple[str, str], str] = {}
    load_error: str | None = None
    cleanup: JsonDict = {"action": "not_started", "leak_free": False}
    started = time.perf_counter()
    _progress(6, "subprocess_start", command=command)
    _progress(6, "model_load_start", model_id=QWEN_MODEL_ID)
    try:
        identity = supervisor.launch()
        health = runtime_v629.prior._wait_for_health(supervisor, port, timeout_s=480.0)
        _progress(6, "model_load_end", model_id=QWEN_MODEL_ID, health=health.get("ok"))
        gpu_rows.append(runtime_v629.prior._gpu_snapshot("model_loaded", phase=6))
        if health.get("ok") is not True:
            raise RuntimeError(f"server health failed:{health.get('classification')}")
    except Exception as exc:  # noqa: BLE001 - every scheduled failure remains a raw row.
        load_error = f"{type(exc).__name__}: {exc}"
        _progress(6, "model_load_end", model_id=QWEN_MODEL_ID, error=load_error)

    _progress(6, "generation_batch_start", call_count=len(schedule))
    last_heartbeat = time.monotonic()
    for index, call in enumerate(schedule, 1):
        call_id = str(call["call_id"])
        fixture_id = str(call["fixture_id"])
        arm = str(call["arm"])
        pass_index = int(call["pass_index"])
        prompt = str(call["prompt"])
        if pass_index == 2:
            prompt = prompt.replace(
                "{{FIRST_PASS_OUTPUT}}", first_outputs.get((fixture_id, arm), "")
            )
        prompt_rows.append(
            {
                "call_id": call_id,
                "fixture_id": fixture_id,
                "arm": arm,
                "pass_index": pass_index,
                "output_token_limit": call["output_token_limit"],
                "template_prompt_sha256": call["prompt_sha256"],
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
            }
        )
        _progress(
            6,
            "generation_start",
            call_index=index,
            call_id=call_id,
            arm=arm,
            pass_index=pass_index,
        )
        response_row: JsonDict = {}
        error = load_error
        if error is None:
            try:
                response_row = _request_generation(
                    port,
                    prompt,
                    max_tokens=int(call["output_token_limit"]),
                    seed=grounding_v627._call_seed(call_id),
                )
            except Exception as exc:  # noqa: BLE001 - failures remain abstention rows.
                error = f"{type(exc).__name__}: {exc}"
        raw_output = str(response_row.get("raw_output", ""))
        raw_response = response_row.get("raw_response", {})
        raw_row = {
            "call_id": call_id,
            "fixture_id": fixture_id,
            "arm": arm,
            "pass_index": pass_index,
            "raw_output": raw_output,
            "raw_output_sha256": sha256_text(raw_output),
            "reasoning_output": response_row.get("reasoning_output", ""),
            "raw_response": raw_response,
            "raw_response_sha256": sha256_text(canonical_json(raw_response)),
            "prompt_tokens": int(response_row.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(response_row.get("completion_tokens", 0) or 0),
            "latency_s": float(response_row.get("latency_s", 0.0) or 0.0),
            "terminal_state": "complete" if error is None else "failed",
            "error": error,
        }
        raw_rows.append(raw_row)
        if pass_index == 1:
            first_outputs[(fixture_id, arm)] = raw_output
        _progress(
            6,
            "generation_end",
            call_index=index,
            call_id=call_id,
            completion_tokens=raw_row["completion_tokens"],
            terminal_state=raw_row["terminal_state"],
        )
        now = time.monotonic()
        if index % 8 == 0 or now - last_heartbeat >= 300:
            _progress(6, "heartbeat", completed=index, total=len(schedule))
            last_heartbeat = now
    _progress(6, "generation_batch_end", call_count=len(raw_rows))
    if health.get("ok") is True:
        gpu_rows.append(runtime_v629.prior._gpu_snapshot("after_generation", phase=6))
    _progress(6, "subprocess_cleanup_start", pid=identity.get("pid"))
    cleanup = supervisor.cleanup()
    _progress(6, "subprocess_cleanup_end", leak_free=cleanup.get("leak_free"))
    process_returncode = None
    if supervisor.proc is not None:
        try:
            process_returncode = supervisor.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process_returncode = supervisor.proc.poll()
    _progress(6, "subprocess_end", command=command, returncode=process_returncode)
    gpu_rows.append(runtime_v629.prior._gpu_snapshot("after_teardown", phase=6))
    server_log = (
        supervisor.log_path.read_text(encoding="utf-8", errors="replace")
        if supervisor.log_path.is_file()
        else ""
    )
    loaded_gpu = next((row for row in gpu_rows if row.get("phase") == "model_loaded"), {})
    cuda = runtime_v629.cuda_offload_receipt(
        server_log, loaded_gpu, pid=int(identity.get("pid", -1)), command=command
    )
    load_receipt = {
        "model_id": QWEN_MODEL_ID,
        "loaded_path": spec.get("loaded_path", spec.get("model_path")),
        "revision": spec.get("revision"),
        "size_bytes": spec.get("size_bytes"),
        "sha256": spec.get("sha256"),
        "template_source": spec.get("chat_template_source"),
        "template_sha256": spec.get("chat_template_sha256"),
        "backend": "native_llama_server",
        "command": command,
        "pid": identity.get("pid"),
        "health": health,
        "server_log_path": str(supervisor.log_path),
        "server_log": server_log,
        "server_log_sha256": sha256_text(server_log),
        "process_returncode": process_returncode,
        "cleanup": cleanup,
        "gpu_offload_confirmed": cuda.get("gpu_offload_confirmed", False),
        "cuda_receipt": cuda,
        "duration_s": time.perf_counter() - started,
        "error": load_error,
        "load_count": 1,
        "generation_call_count": len(raw_rows),
    }
    gc.collect()
    return prompt_rows, raw_rows, load_receipt, gpu_rows


def run_experiment(  # pragma: no cover - this boundary owns the live local model.
    *,
    root: Path,
    run_date: str,
    result_path: Path,
    runtime_path: Path,
    fixture_path: Path,
    raw_dir: Path,
) -> JsonDict:
    """Gate inputs, run Qwen once, seal evidence, then open exact labels."""

    started = time.perf_counter()
    _progress(0, "phase_start", name="schema_complete_first_write")
    artifact = initialize_artifact(result_path, run_date)
    _progress(0, "phase_end", name="schema_complete_first_write", path=str(result_path))

    _progress(1, "phase_start", name="upstream_output_and_label_authority")
    artifact["source_artifact_hashes"] = _source_hashes(root)
    output_observed = _output_precondition(result_path, raw_dir)
    output_expected = {
        "result_parent_writable": True,
        "raw_dir_writable": True,
        "raw_dir_isolated": True,
        "existing_raw_entries": [],
    }
    checks = [
        gate_row(
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            upstream="task",
            field="run_date",
        ),
        gate_row(
            "output_directory",
            output_expected,
            output_observed,
            output_observed == output_expected,
            upstream=str(RAW_DIR),
            field="output_directory",
        ),
    ]
    runtime_value = _load_value(runtime_path)
    upstream_checks, _fixture_precheck = _upstream_checks(runtime_path, fixture_path, runtime_value)
    checks.extend(upstream_checks)
    _checkpoint(result_path, artifact)
    _progress(
        1,
        "phase_end",
        name="upstream_output_and_label_authority",
        passed=all(row["passed"] for row in checks),
    )
    if runtime_value is None or any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(2, "phase_start", name="frozen_schedule_replay")
    schedule = expected_call_schedule(runtime_value)
    schedule_errors = schedule_identity_errors(schedule, runtime_value)
    artifact["frozen_fixture_ids"] = list(FROZEN_FIXTURE_IDS)
    artifact["frozen_schedule_hash"] = str(runtime_value["frozen_schedule_hash"])
    artifact["schedule_identity_score"] = int(not schedule_errors)
    _checkpoint(result_path, artifact)
    _progress(
        2,
        "phase_end",
        name="frozen_schedule_replay",
        call_count=len(schedule),
        passed=not schedule_errors,
    )
    if schedule_errors:
        checks.append(
            gate_row(
                "frozen_schedule_hash",
                [],
                schedule_errors,
                False,
                upstream=str(RUNTIME_PATH),
                field="schedule_rows",
            )
        )
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(3, "phase_start", name="cached_qwen_identity")
    specs = runtime_v629.resolve_model_specs()
    _progress(3, "benchmark_start", name="model_hash_and_embedded_template")
    specs, identities, model_errors = runtime_v629.model_identity_receipts(specs)
    _progress(3, "benchmark_end", name="model_hash_and_embedded_template", errors=model_errors)
    artifact["MODEL_SPECS"] = specs
    artifact["model_identity_rows"] = identities
    qualified_hash = runtime_value["MODEL_SPECS"][0]["sha256"]
    observed_hash = specs[0].get("sha256") if specs else None
    non_template_errors = [
        error for error in model_errors if error != "embedded_chat_template_missing"
    ]
    checks.extend(
        [
            gate_row(
                "cached_qwen_q4",
                {"errors": [], "sha256": qualified_hash},
                {"errors": non_template_errors, "sha256": observed_hash},
                not non_template_errors and observed_hash == qualified_hash,
                upstream=str(RUNTIME_PATH),
                field="MODEL_SPECS[0].sha256",
            ),
            gate_row(
                "embedded_chat_template",
                True,
                bool(identities and identities[0].get("template_present")),
                bool(identities and identities[0].get("template_present")),
                upstream=str(RUNTIME_PATH),
                field="model_identity_rows[0].template_present",
            ),
        ]
    )
    _checkpoint(result_path, artifact)
    _progress(
        3, "phase_end", name="cached_qwen_identity", passed=all(row["passed"] for row in checks)
    )
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(4, "phase_start", name="native_cuda_linkage")
    server_path = resolve_native_llama_server()
    linkage, backend = runtime_v629.binary_runtime_receipts(
        server_path, command_runner=_run_subprocess
    )
    linkage_errors = runtime_v629.cuda_linkage_errors(linkage)
    artifact["model_load_receipts"] = [
        {"preload_binary_linkage": linkage, "preload_backend": backend}
    ]
    checks.append(
        gate_row(
            "native_cuda_linkage",
            [],
            linkage_errors,
            not linkage_errors and bool(backend and backend[0].get("exists")),
            upstream=str(server_path),
            field="ldd_cuda_linkage",
        )
    )
    _checkpoint(result_path, artifact)
    _progress(4, "phase_end", name="native_cuda_linkage", passed=not linkage_errors)
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(5, "phase_start", name="gpu_gate_and_preconditions_checkpoint")
    before_gpu = runtime_v629.prior._gpu_snapshot("before", phase=5)
    artifact["gpu_rows"] = [before_gpu]
    external_apps = [
        row for row in before_gpu.get("compute_apps", []) if not row.get("owned_by_task")
    ]
    gpu_expected = {"query_ok": True, "gpu_count": 2, "external_compute_apps": []}
    gpu_observed = {
        "query_ok": before_gpu.get("ok"),
        "gpu_count": before_gpu.get("gpu_count"),
        "external_compute_apps": external_apps,
    }
    gpu_passed = (
        before_gpu.get("ok") is True and before_gpu.get("gpu_count") == 2 and not external_apps
    )
    checks.append(
        gate_row(
            "gpu_available",
            gpu_expected,
            gpu_observed,
            gpu_passed,
            upstream="host",
            field="gpu_rows.before",
        )
    )
    if not gpu_passed:
        _progress(5, "phase_end", name="gpu_gate_and_preconditions_checkpoint", passed=False)
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )
    artifact["status"] = "preconditions_passed"
    artifact["preconditions_checked"] = deepcopy(checks)
    _checkpoint(result_path, artifact)
    _progress(5, "status", status="preconditions_passed")
    _progress(5, "phase_end", name="gpu_gate_and_preconditions_checkpoint", passed=True)

    _progress(6, "phase_start", name="single_load_full_generation_matrix")
    prompt_rows, raw_rows, load_receipt, runtime_gpu_rows = _run_matrix_once(
        schedule, specs[0], server_path=server_path, raw_dir=raw_dir
    )
    artifact["prompt_hash_rows"] = prompt_rows
    artifact["raw_output_rows"] = raw_rows
    artifact["model_load_receipts"] = [load_receipt]
    artifact["gpu_rows"].extend(runtime_gpu_rows)
    _checkpoint(result_path, artifact)
    load_passed = (
        dict(load_receipt.get("health") or {}).get("ok") is True
        and load_receipt.get("gpu_offload_confirmed") is True
        and dict(load_receipt.get("cleanup") or {}).get("leak_free") is True
        and len(raw_rows) == 168
    )
    _progress(6, "phase_end", name="single_load_full_generation_matrix", passed=load_passed)
    if not load_passed:
        checks.append(
            gate_row(
                "model_generation",
                {"health": True, "gpu_offload": True, "cleanup": True, "rows": 168},
                {
                    "health": dict(load_receipt.get("health") or {}).get("ok"),
                    "gpu_offload": load_receipt.get("gpu_offload_confirmed"),
                    "cleanup": dict(load_receipt.get("cleanup") or {}).get("leak_free"),
                    "rows": len(raw_rows),
                    "error": load_receipt.get("error"),
                },
                False,
                upstream="live_qwen_process",
                field="model_load_receipts[0]",
            )
        )
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(7, "phase_start", name="immutable_model_evidence_seal")
    seal = model_evidence_seal(prompt_rows, raw_rows, schedule)
    artifact["label_exposure_count"] = seal["label_exposure_count"]
    _checkpoint(result_path, artifact)
    _progress(
        7,
        "phase_end",
        name="immutable_model_evidence_seal",
        sealed=seal["sealed"],
        evidence_sha256=seal["evidence_sha256"],
    )
    if seal["sealed"] is not True:
        checks.append(
            gate_row(
                "model_evidence_seal",
                [],
                seal["errors"],
                False,
                upstream="live_qwen_process",
                field="prompt_hash_rows+raw_output_rows",
            )
        )
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(8, "phase_start", name="label_open_structure_sql_and_metrics")
    # This is the first access that joins exact outcome values to model output.
    fixture_value = _load_value(fixture_path)
    if fixture_value is None:
        checks.append(
            gate_row(
                "exact_label_authority",
                "readable_json_object_after_seal",
                "missing_or_invalid",
                False,
                upstream=str(FIXTURE_PATH),
                field="sealed_scorer_rows",
            )
        )
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )
    _progress(8, "benchmark_start", name="structure_sql_and_paired_reduction")
    parse_rows, structure_rows, execution_rows, arm_rows = score_frozen_outputs(
        runtime_value, fixture_value, prompt_rows, raw_rows, seal
    )
    _progress(8, "benchmark_end", name="structure_sql_and_paired_reduction", arm_rows=len(arm_rows))
    artifact.update(
        {
            "parse_rows": parse_rows,
            "source_structure_rows": structure_rows,
            "sql_execution_rows": execution_rows,
            "rows": arm_rows,
        }
    )
    _checkpoint(result_path, artifact)
    _progress(8, "phase_end", name="label_open_structure_sql_and_metrics", rows=len(arm_rows))

    _progress(9, "phase_start", name="terminal_reduction")
    result = finalize_artifact(artifact, checks, duration_s=time.perf_counter() - started)
    write_artifact(result_path, result)
    _progress(
        9,
        "phase_end",
        name="terminal_reduction",
        verdict_class=result["verdict_class"],
        complete_score=result["qwen_dual_side_pilot_complete_score"],
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the dated pilot or cold-validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        _progress(10, "subprocess_start", name="artifact_validation", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(10, "subprocess_end", name="artifact_validation", valid=not errors)
        print(canonical_json({"valid": not errors, "errors": errors}), flush=True)
        return int(bool(errors))
    root = find_repo_root()
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    result = run_experiment(
        root=root,
        run_date=args.date,
        result_path=result_path,
        runtime_path=root / RUNTIME_PATH,
        fixture_path=root / FIXTURE_PATH,
        raw_dir=root / RAW_DIR,
    )
    errors = validate_artifact(result)
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "valid": not errors,
                "errors": errors,
                "verdict_class": result.get("verdict_class"),
                "qwen_dual_side_pilot_complete_score": result.get(
                    "qwen_dual_side_pilot_complete_score"
                ),
            }
        ),
        flush=True,
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
